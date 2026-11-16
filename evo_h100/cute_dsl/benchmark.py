"""Benchmark local EvoAttention + NCCL against fused distributed EvoAttention.

The measured boundaries are deliberately narrow:

* forward baseline: local CuTe EvoAttention, then a packed NCCL row-to-column
  All-to-All of the attention output;
* forward fused: distributed CuTe EvoAttention, whose epilogue remotely writes
  both the attention output and the residual input;
* backward baseline: a packed NCCL column-to-row All-to-All of ``dO``, then the
  local CuTe EvoAttention backward;
* backward fused: remote reconstruction of ``dO`` and ``dX``, followed by the
  CuTe EvoAttention backward mainloop.

Compilation, tensor allocation, NVSHMEM initialization, and peer-pointer setup
all happen before timing.  NCCL pack/unpack copies are intentionally timed.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.distributed as dist
from cuda.core import Device
import nvshmem.core as nvshmem

from CuteEvoAttention import evoattention_cute_d32 as local_forward_module
from CuteEvoAttentionDist.benchmark_support import (
    column_to_row,
    measure_distributed,
    row_to_column,
)
from CuteEvoAttentionDist.evoattention_dist_bwd_d32 import (
    get_distributed_evoattention_backward,
    get_distributed_evoattention_backward_no_preprocess,
)
from CuteEvoAttentionDist.evoattention_dist_d32 import (
    get_distributed_evoattention_forward,
)
from CuteEvoAttentionDist.nvshmem_utils import broadcast_unique_id
from CuteEvoAttentionDist.remote_backward_preprocess import (
    get_remote_backward_preprocess,
)


HEAD_DIMENSION = 32
RESIDUAL_DIMENSION = 128
SUPPORTED_WORLD_SIZES = (2, 4, 8)
SUPPORTED_BATCH_SIZES = (1, 4)
SUPPORTED_HEAD_COUNTS = (4, 16)
SUPPORTED_CONTEXT_LENGTHS = (128, 256, 384, 512, 640, 768, 896, 1024)


@dataclass(frozen=True)
class ProblemShape:
    batch: int
    heads: int
    context_length: int


@dataclass(frozen=True)
class TimingResult:
    baseline_ms: float
    fused_ms: float

    @property
    def speedup(self) -> float:
        return self.baseline_ms / self.fused_ms


@dataclass(frozen=True)
class ShapeResult:
    batch: int
    heads: int
    context_length: int
    sequence_count: int
    world_size: int
    forward: TimingResult | None
    backward: TimingResult | None
    combined: TimingResult | None


class _RectangularLocalForward(local_forward_module.EvoAttentionForward):
    """Use the production local kernel for a row-sharded ``S x N`` problem.

    The production launcher restricts its public single-GPU API to ``S == N``.
    Its kernel and dynamic layouts already use S and N independently, so the
    benchmark only relaxes that API check; no kernel code is duplicated.
    """

    @staticmethod
    def _validate_arguments(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pair_bias: torch.Tensor,
        residual_mask: torch.Tensor,
        output: torch.Tensor,
        logsumexp: torch.Tensor,
    ) -> None:
        if query.ndim != 5:
            raise ValueError("query must have shape [B, S, N, H, 32]")
        batch, sequence_count, context_length, heads, dimension = query.shape
        if dimension != HEAD_DIMENSION:
            raise ValueError(f"head dimension must be {HEAD_DIMENSION}")
        if context_length % local_forward_module.QUERY_TILE_SIZE:
            raise ValueError(
                "N must be divisible by "
                f"{local_forward_module.QUERY_TILE_SIZE}"
            )
        if sequence_count % local_forward_module.CONSUMER_WARPGROUP_COUNT:
            raise ValueError("S must be even")
        if key.shape != query.shape or value.shape != query.shape:
            raise ValueError("query, key, and value must have identical shapes")
        if output.shape != query.shape:
            raise ValueError("output must have the same shape as query")
        if pair_bias.shape != (
            batch,
            1,
            heads,
            context_length,
            context_length,
        ):
            raise ValueError("pair_bias must have shape [B, 1, H, N, N]")
        if residual_mask.shape != (
            batch,
            sequence_count,
            1,
            1,
            context_length,
        ):
            raise ValueError("residual_mask must have shape [B, S, 1, 1, N]")
        if logsumexp.shape != (
            batch,
            sequence_count,
            heads,
            context_length,
        ):
            raise ValueError("logsumexp must have shape [B, S, H, N]")

        tensors = (
            query,
            key,
            value,
            pair_bias,
            residual_mask,
            output,
            logsumexp,
        )
        if any(tensor.device != query.device for tensor in tensors):
            raise ValueError("all tensors must be on the same CUDA device")
        if query.device.type != "cuda":
            raise ValueError("EvoAttention requires CUDA tensors")
        if any(
            not tensor.is_contiguous()
            for tensor in (query, key, value, residual_mask, output, logsumexp)
        ):
            raise ValueError("Q/K/V/mask/output/LSE must be contiguous")
        if pair_bias.stride(-1) != 1 or any(
            stride < 0 for stride in pair_bias.stride()
        ):
            raise ValueError("pair_bias must have a contiguous K axis")
        if any(tensor.dtype != torch.bfloat16 for tensor in tensors[:-1]):
            raise TypeError("all attention tensors except LSE must be BF16")
        if logsumexp.dtype != torch.float32:
            raise TypeError("logsumexp must be FP32")


_LOCAL_FORWARD = _RectangularLocalForward()


class _SymmetricAllocationPool:
    """Track collectively allocated NVSHMEM tensors for deterministic cleanup."""

    def __init__(self) -> None:
        self._allocations: list[torch.Tensor] = []

    def tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        allocation = nvshmem.tensor(shape, dtype=dtype)
        self._allocations.append(allocation)
        return allocation

    def close(self) -> None:
        torch.cuda.synchronize()
        dist.barrier()
        for allocation in reversed(self._allocations):
            nvshmem.free_tensor(allocation)
        self._allocations.clear()


def _random_tensor(
    shape: tuple[int, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        shape,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )


def _row_to_column_reference(
    local_row: torch.Tensor,
    rank: int,
    sequence_count: int,
) -> torch.Tensor:
    """Construct a correctness reference outside the measured region."""
    gathered = [torch.empty_like(local_row) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_row)
    return torch.cat(gathered, dim=1)[
        :, :, rank * sequence_count : (rank + 1) * sequence_count
    ].contiguous()


def _column_to_row_reference(
    local_column: torch.Tensor,
    rank: int,
    sequence_count: int,
) -> torch.Tensor:
    """Construct the inverse-reshard reference outside the measured region."""
    gathered = [
        torch.empty_like(local_column) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered, local_column)
    return torch.cat(
        [
            peer[:, rank * sequence_count : (rank + 1) * sequence_count]
            for peer in gathered
        ],
        dim=2,
    ).contiguous()


def _check_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    rank: int,
) -> None:
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(f"rank {rank}: {name} mismatch\n{error}") from error
    if rank == 0:
        maximum_error = (actual.float() - expected.float()).abs().max().item()
        print(f"    {name:12s} max_abs={maximum_error:.6f}")


def _median_time(
    operation: Callable[[], None],
    warmup: int,
    iterations: int,
    repeats: int,
) -> float:
    return statistics.median(
        measure_distributed(operation, warmup, iterations, repeats)
    )


def _measure_pair(
    baseline: Callable[[], None],
    fused: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    fused_first: bool,
) -> TimingResult:
    """Measure both paths, alternating order across shapes to reduce drift."""
    if fused_first:
        fused_ms = _median_time(fused, warmup, iterations, repeats)
        baseline_ms = _median_time(baseline, warmup, iterations, repeats)
    else:
        baseline_ms = _median_time(baseline, warmup, iterations, repeats)
        fused_ms = _median_time(fused, warmup, iterations, repeats)
    return TimingResult(baseline_ms=baseline_ms, fused_ms=fused_ms)


def _run_shape(
    shape: ProblemShape,
    *,
    rank: int,
    world_size: int,
    warmup: int,
    iterations: int,
    repeats: int,
    directions: frozenset[str],
    check_correctness: bool,
    fused_first: bool,
) -> ShapeResult:
    batch = shape.batch
    heads = shape.heads
    context_length = shape.context_length
    sequence_count = context_length // world_size
    generator = torch.Generator(device="cuda").manual_seed(
        20260930 + 1009 * rank + context_length + 17 * heads + batch
    )
    attention_shape = (
        batch,
        sequence_count,
        context_length,
        heads,
        HEAD_DIMENSION,
    )
    statistics_shape = (batch, sequence_count, heads, context_length)
    pair_shape = (batch, heads, context_length, context_length)
    peer_output_shape = (
        batch,
        sequence_count,
        sequence_count,
        heads,
        HEAD_DIMENSION,
    )
    peer_residual_shape = (
        batch,
        sequence_count,
        sequence_count,
        RESIDUAL_DIMENSION,
    )

    query = _random_tensor(attention_shape, generator)
    key = _random_tensor(attention_shape, generator)
    value = _random_tensor(attention_shape, generator)
    pair_bias = _random_tensor((batch, 1, *pair_shape[1:]), generator)
    residual_mask = torch.empty(
        (batch, sequence_count, 1, 1, context_length),
        dtype=torch.bfloat16,
        device="cuda",
    )
    residual_mask.bernoulli_(0.875, generator=generator)
    residual_mask[..., 0] = 1.0
    residual_input = _random_tensor(
        (batch, sequence_count, context_length, RESIDUAL_DIMENSION),
        generator,
    )

    local_forward = _LOCAL_FORWARD
    baseline_backward = get_distributed_evoattention_backward()
    distributed_forward = get_distributed_evoattention_forward(world_size, rank)
    distributed_backward = (
        get_distributed_evoattention_backward_no_preprocess()
    )
    remote_preprocess = get_remote_backward_preprocess(world_size, rank)

    baseline_local_output = torch.empty_like(query)
    baseline_logsumexp = torch.empty(
        statistics_shape, dtype=torch.float32, device="cuda"
    )
    baseline_column_output = torch.empty(
        (batch, context_length, sequence_count, heads, HEAD_DIMENSION),
        dtype=torch.bfloat16,
        device="cuda",
    )
    baseline_output_gradient = torch.empty_like(query)
    baseline_delta = torch.empty_like(baseline_logsumexp)
    baseline_dq = torch.empty_like(query)
    baseline_dk = torch.empty_like(query)
    baseline_dv = torch.empty_like(query)
    baseline_dpb = torch.empty(pair_shape, dtype=torch.float32, device="cuda")
    baseline_dpb_output = torch.empty(
        pair_shape, dtype=torch.bfloat16, device="cuda"
    )

    distributed_local_output = torch.empty_like(query)
    distributed_logsumexp = torch.empty_like(baseline_logsumexp)
    distributed_output_gradient = torch.empty_like(query)
    distributed_residual_gradient = torch.empty_like(residual_input)
    distributed_delta = torch.empty_like(distributed_logsumexp)
    distributed_dq = torch.empty_like(query)
    distributed_dk = torch.empty_like(query)
    distributed_dv = torch.empty_like(query)
    distributed_dpb = torch.empty_like(baseline_dpb)
    distributed_dpb_output = torch.empty_like(baseline_dpb_output)

    # The same buffers are safely reused in the two communication directions;
    # operations are ordered on one CUDA stream.
    communication_buffers = tuple(
        torch.empty(peer_output_shape, dtype=torch.bfloat16, device="cuda")
        for _ in range(2 * world_size)
    )
    send_buffers = communication_buffers[:world_size]
    receive_buffers = communication_buffers[world_size:]

    symmetric = _SymmetricAllocationPool()
    try:
        output_storage = symmetric.tensor(
            (
                batch,
                world_size,
                sequence_count,
                sequence_count,
                heads,
                HEAD_DIMENSION,
            ),
            dtype=torch.bfloat16,
        )
        output_peers = tuple(
            nvshmem.get_peer_tensor(output_storage, peer)[:, rank]
            for peer in range(world_size)
        )
        distributed_column_output = output_storage.reshape(
            batch,
            context_length,
            sequence_count,
            heads,
            HEAD_DIMENSION,
        )

        residual_output_storage = symmetric.tensor(
            (
                batch,
                world_size,
                sequence_count,
                sequence_count,
                RESIDUAL_DIMENSION,
            ),
            dtype=torch.bfloat16,
        )
        residual_output_peers = tuple(
            nvshmem.get_peer_tensor(residual_output_storage, peer)[:, rank]
            for peer in range(world_size)
        )
        distributed_column_residual = residual_output_storage.reshape(
            batch,
            context_length,
            sequence_count,
            RESIDUAL_DIMENSION,
        )

        completion_storage = symmetric.tensor(
            (world_size,), dtype=torch.int32
        )
        completion_storage.zero_()
        completion_peers = tuple(
            nvshmem.get_peer_tensor(completion_storage, peer)[rank : rank + 1]
            for peer in range(world_size)
        )
        completion_counter = torch.zeros(
            (1,), dtype=torch.int32, device="cuda"
        )

        column_gradient_storage = symmetric.tensor(
            (
                batch,
                world_size,
                sequence_count,
                sequence_count,
                heads,
                HEAD_DIMENSION,
            ),
            dtype=torch.bfloat16,
        )
        local_column_output_gradient = column_gradient_storage.reshape(
            batch,
            context_length,
            sequence_count,
            heads,
            HEAD_DIMENSION,
        )
        local_column_output_gradient.copy_(
            _random_tensor(tuple(local_column_output_gradient.shape), generator)
        )
        output_gradient_peers = tuple(
            nvshmem.get_peer_tensor(column_gradient_storage, peer)[:, rank]
            for peer in range(world_size)
        )

        residual_gradient_storage = symmetric.tensor(
            (
                batch,
                world_size,
                sequence_count,
                sequence_count,
                RESIDUAL_DIMENSION,
            ),
            dtype=torch.bfloat16,
        )
        local_column_residual_gradient = residual_gradient_storage.reshape(
            batch,
            context_length,
            sequence_count,
            RESIDUAL_DIMENSION,
        )
        local_column_residual_gradient.copy_(
            _random_tensor(tuple(local_column_residual_gradient.shape), generator)
        )
        residual_gradient_peers = tuple(
            nvshmem.get_peer_tensor(residual_gradient_storage, peer)[:, rank]
            for peer in range(world_size)
        )
        dist.barrier()

        query_gradient_accumulator = distributed_backward._workspace(
            batch,
            sequence_count,
            context_length,
            heads,
            query.device,
        )
        completion_epoch = 0

        def baseline_forward_step() -> None:
            local_forward(
                query,
                key,
                value,
                pair_bias,
                residual_mask,
                baseline_local_output,
                baseline_logsumexp,
            )
            row_to_column(
                baseline_local_output,
                send_buffers,
                receive_buffers,
                baseline_column_output,
            )

        def fused_forward_step() -> None:
            nonlocal completion_epoch
            completion_epoch += 1
            distributed_forward(
                query,
                key,
                value,
                pair_bias,
                residual_mask,
                residual_input,
                distributed_local_output,
                output_peers,
                residual_output_peers,
                completion_peers,
                completion_storage,
                completion_counter,
                completion_epoch,
                distributed_logsumexp,
            )

        def baseline_backward_step() -> None:
            column_to_row(
                local_column_output_gradient,
                send_buffers,
                receive_buffers,
                baseline_output_gradient,
            )
            baseline_backward(
                query,
                key,
                value,
                baseline_local_output,
                baseline_output_gradient,
                pair_bias,
                residual_mask,
                baseline_logsumexp,
                baseline_delta,
                baseline_dq,
                baseline_dk,
                baseline_dv,
                baseline_dpb,
                baseline_dpb_output,
            )
            # The inherited square launcher converts only S rows of dPairBias
            # when S != N.  Complete the benchmark-only rectangular adapter.
            baseline_dpb_output.copy_(baseline_dpb)

        def fused_backward_step() -> None:
            remote_preprocess(
                output_gradient_peers,
                residual_gradient_peers,
                distributed_local_output,
                distributed_output_gradient,
                distributed_residual_gradient,
                distributed_delta,
                query_gradient_accumulator,
                distributed_dpb,
            )
            distributed_backward(
                query,
                key,
                value,
                distributed_local_output,
                distributed_output_gradient,
                pair_bias,
                residual_mask,
                distributed_logsumexp,
                distributed_delta,
                distributed_dq,
                distributed_dk,
                distributed_dv,
                distributed_dpb,
                distributed_dpb_output,
            )

        def baseline_combined_step() -> None:
            baseline_forward_step()
            baseline_backward_step()

        def fused_combined_step() -> None:
            fused_forward_step()
            fused_backward_step()

        # Compilation, workspace creation, NVSHMEM pointer setup, and initial
        # allocations all finish before either correctness checking or timing.
        baseline_forward_step()
        fused_forward_step()
        baseline_backward_step()
        fused_backward_step()
        torch.cuda.synchronize()
        dist.barrier()

        if check_correctness:
            if rank == 0:
                print(
                    f"  correctness B={batch} H={heads} N={context_length} "
                    f"S={sequence_count}:"
                )
            _check_close(
                "O_col",
                distributed_column_output,
                baseline_column_output,
                atol=2e-2,
                rtol=2e-2,
                rank=rank,
            )
            expected_column_residual = _row_to_column_reference(
                residual_input, rank, sequence_count
            )
            _check_close(
                "X_col",
                distributed_column_residual,
                expected_column_residual,
                atol=0.0,
                rtol=0.0,
                rank=rank,
            )
            _check_close(
                "dO_row",
                distributed_output_gradient,
                baseline_output_gradient,
                atol=0.0,
                rtol=0.0,
                rank=rank,
            )
            expected_residual_gradient = _column_to_row_reference(
                local_column_residual_gradient, rank, sequence_count
            )
            _check_close(
                "dX_row",
                distributed_residual_gradient,
                expected_residual_gradient,
                atol=0.0,
                rtol=0.0,
                rank=rank,
            )
            for name, actual, expected in (
                ("dQ", distributed_dq, baseline_dq),
                ("dK", distributed_dk, baseline_dk),
                ("dV", distributed_dv, baseline_dv),
                ("dPairBias", distributed_dpb_output, baseline_dpb_output),
            ):
                _check_close(
                    name,
                    actual,
                    expected,
                    atol=6e-2,
                    rtol=6e-2,
                    rank=rank,
                )

        forward_result = None
        backward_result = None
        combined_result = None
        if "forward" in directions:
            forward_result = _measure_pair(
                baseline_forward_step,
                fused_forward_step,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
                fused_first=fused_first,
            )
        if "backward" in directions:
            # Refresh the forward state consumed by each backward before timing.
            baseline_forward_step()
            fused_forward_step()
            backward_result = _measure_pair(
                baseline_backward_step,
                fused_backward_step,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
                fused_first=fused_first,
            )
        if "combined" in directions:
            combined_result = _measure_pair(
                baseline_combined_step,
                fused_combined_step,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
                fused_first=fused_first,
            )
        torch.cuda.synchronize()
        return ShapeResult(
            batch=batch,
            heads=heads,
            context_length=context_length,
            sequence_count=sequence_count,
            world_size=world_size,
            forward=forward_result,
            backward=backward_result,
            combined=combined_result,
        )
    finally:
        symmetric.close()
        gc.collect()
        torch.cuda.empty_cache()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare local CuTe EvoAttention + NCCL All-to-All with the fused "
            "NVSHMEM distributed kernel. Launch with torchrun."
        )
    )
    parser.add_argument("--batch", type=int, choices=SUPPORTED_BATCH_SIZES, default=1)
    parser.add_argument("--heads", type=int, choices=SUPPORTED_HEAD_COUNTS, default=4)
    parser.add_argument(
        "--n", type=int, choices=SUPPORTED_CONTEXT_LENGTHS, default=384
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="sweep B={1,4}, H={4,16}, and every supported N",
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "combined", "all"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="write rank-0 results to this JSON file",
    )
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.iterations <= 0 or arguments.repeats <= 0:
        parser.error("warmup must be non-negative; iterations/repeats must be positive")
    return arguments


def _select_shapes(arguments: argparse.Namespace) -> tuple[ProblemShape, ...]:
    if not arguments.all_shapes:
        return (ProblemShape(arguments.batch, arguments.heads, arguments.n),)
    return tuple(
        ProblemShape(batch, heads, context_length)
        for batch in SUPPORTED_BATCH_SIZES
        for heads in SUPPORTED_HEAD_COUNTS
        for context_length in SUPPORTED_CONTEXT_LENGTHS
    )


def _print_result(result: ShapeResult) -> None:
    prefix = (
        f"P={result.world_size} B={result.batch} H={result.heads} "
        f"N={result.context_length} S={result.sequence_count} D={HEAD_DIMENSION}"
    )
    fields = [prefix]
    for name in ("forward", "backward", "combined"):
        timing = getattr(result, name)
        if timing is not None:
            fields.append(
                f"{name}: baseline={timing.baseline_ms:.4f} ms "
                f"fused={timing.fused_ms:.4f} ms {timing.speedup:.3f}x"
            )
    print(" | ".join(fields))


def _result_as_dict(result: ShapeResult) -> dict[str, object]:
    payload = asdict(result)
    for direction in ("forward", "backward", "combined"):
        timing = getattr(result, direction)
        if timing is not None:
            payload[direction]["speedup"] = timing.speedup  # type: ignore[index]
    return payload


def main() -> None:
    arguments = _parse_arguments()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = Device(local_rank)
    device.set_current()

    nvshmem_initialized = False
    try:
        if world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"world size must be one of {SUPPORTED_WORLD_SIZES}, got {world_size}"
            )
        nvshmem.init(
            device=device,
            uid=broadcast_unique_id(rank),
            rank=rank,
            nranks=world_size,
            initializer_method="uid",
        )
        nvshmem_initialized = True

        shapes = _select_shapes(arguments)
        for shape in shapes:
            if shape.context_length % world_size:
                raise ValueError("N must be divisible by P")
            if shape.context_length // world_size % 16:
                raise ValueError("N/P must be divisible by 16")

        if arguments.direction == "all":
            directions = frozenset(("forward", "backward", "combined"))
        else:
            directions = frozenset((arguments.direction,))

        if rank == 0:
            print(
                "baseline: local CuTe EvoAttention + packed NCCL All-to-All; "
                "fused: distributed CuTe remote epilogue/preprocess"
            )
            print(
                "timing: CUDA events, slowest rank, median of repeats; "
                "pack/unpack included; compile/allocation/NVSHMEM setup excluded"
            )

        results = []
        for shape_index, shape in enumerate(shapes):
            result = _run_shape(
                shape,
                rank=rank,
                world_size=world_size,
                warmup=arguments.warmup,
                iterations=arguments.iterations,
                repeats=arguments.repeats,
                directions=directions,
                check_correctness=not arguments.skip_check,
                fused_first=bool(shape_index % 2),
            )
            results.append(result)
            if rank == 0:
                _print_result(result)

        if rank == 0 and arguments.json is not None:
            with open(arguments.json, "w", encoding="utf-8") as output_file:
                json.dump(
                    [_result_as_dict(result) for result in results],
                    output_file,
                    indent=2,
                )
                output_file.write("\n")
    finally:
        if nvshmem_initialized:
            nvshmem.finalize()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
