"""Combined EvoAttention and sequence-parallel communication benchmark.

The baseline preserves MegaFold's communication placement: one C-wide
row-to-column exchange after the forward computation and one inverse exchange
before backward.  The CuTe path routes the raw attention output and residual
input in its forward epilogue, then reconstructs dO and dX during backward
preprocessing.  Layer normalization, gating, output projection, dropout, and
pair-bias collectives are deliberately outside this microbenchmark boundary.
"""

from __future__ import annotations

import argparse
import os
import statistics

import torch
import torch.distributed as dist
from cuda.core import Device

import nvshmem.core as nvshmem

if __package__:
    from .benchmark_support import (
        column_to_row,
        launch_triton_backward,
        launch_triton_forward,
        load_reference_module,
        measure_distributed,
        row_to_column,
    )
    from .evoattention_dist_bwd_d32 import (
        get_distributed_evoattention_backward_no_preprocess,
    )
    from .evoattention_dist_d32 import get_distributed_evoattention_forward
    from .nvshmem_utils import broadcast_unique_id
    from .remote_backward_preprocess import get_remote_backward_preprocess
else:
    from benchmark_support import (
        column_to_row,
        launch_triton_backward,
        launch_triton_forward,
        load_reference_module,
        measure_distributed,
        row_to_column,
    )
    from evoattention_dist_bwd_d32 import (
        get_distributed_evoattention_backward_no_preprocess,
    )
    from evoattention_dist_d32 import get_distributed_evoattention_forward
    from nvshmem_utils import broadcast_unique_id
    from remote_backward_preprocess import get_remote_backward_preprocess


HEAD_DIMENSION = 32
MODEL_DIMENSION = 128


def _all_gather_col_to_row_reference(
    local_column: torch.Tensor,
    rank: int,
    sequence_count: int,
) -> torch.Tensor:
    """Reference inverse reshard used outside the timed region."""
    gathered = [torch.empty_like(local_column) for _ in range(dist.get_world_size())]
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
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if rank == 0:
        error = (actual.float() - expected.float()).abs().max().item()
        print(f"  {name:18s} max_abs={error:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--n", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--breakdown", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = Device(local_rank)
    device.set_current()
    nvshmem.init(
        device=device,
        uid=broadcast_unique_id(rank),
        rank=rank,
        nranks=world_size,
        initializer_method="uid",
    )

    if world_size not in (2, 4, 8):
        raise ValueError("launch with P=2, 4, or 8")
    if args.n % 128 or args.n % world_size:
        raise ValueError("N must be divisible by both 128 and P")
    if args.heads not in (4, 16):
        raise ValueError("H must be 4 or 16")
    sequence_count = args.n // world_size
    if sequence_count % 16:
        raise ValueError("N/P must be divisible by 16")

    generator = torch.Generator(device="cuda").manual_seed(20260923 + rank)
    attention_shape = (
        args.batch,
        sequence_count,
        args.n,
        args.heads,
        HEAD_DIMENSION,
    )
    query = torch.randn(
        attention_shape,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key_value_storage = torch.randn(
        (
            args.batch,
            sequence_count,
            args.n,
            2,
            args.heads,
            HEAD_DIMENSION,
        ),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    key = key_value_storage[:, :, :, 0]
    value = key_value_storage[:, :, :, 1]
    pair_bias_head_last = torch.randn(
        (args.batch, args.n, args.n, args.heads),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    pair_bias_source = pair_bias_head_last.permute(0, 3, 1, 2)
    residual_mask = torch.empty(
        (args.batch, sequence_count, 1, 1, args.n),
        dtype=torch.bfloat16,
        device="cuda",
    )
    residual_mask.bernoulli_(0.875, generator=generator)
    residual_mask[..., 0] = 1
    residual_input = torch.randn(
        (args.batch, sequence_count, args.n, MODEL_DIMENSION),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    # Triton forward workspaces.  These copies reproduce the original wrapper's
    # transpose-and-contiguous contract, including K/V compaction.
    triton_internal_shape = (
        args.batch,
        sequence_count,
        args.heads,
        args.n,
        HEAD_DIMENSION,
    )
    triton_query = torch.empty(
        triton_internal_shape, dtype=torch.bfloat16, device="cuda"
    )
    triton_key_public = torch.empty_like(query)
    triton_value_public = torch.empty_like(query)
    triton_key = torch.empty_like(triton_query)
    triton_value = torch.empty_like(triton_query)
    triton_output = torch.empty_like(triton_query)
    triton_output_public = torch.empty_like(query)
    triton_lse = torch.empty(
        (args.batch, sequence_count, args.heads, args.n),
        dtype=torch.float32,
        device="cuda",
    )
    triton_pair_bias = torch.empty(
        (args.batch, args.heads, args.n, args.n),
        dtype=torch.bfloat16,
        device="cuda",
    )
    additive_mask = torch.empty(
        (args.batch, sequence_count, args.n),
        dtype=torch.float32,
        device="cuda",
    )

    # The original MegaFold placement communicates one post-residual C-wide
    # tensor in each direction.  The omitted pointwise/projection operations do
    # not change the communication volume.
    baseline_message_shape = (
        args.batch,
        sequence_count,
        sequence_count,
        MODEL_DIMENSION,
    )
    forward_send = tuple(
        torch.empty(baseline_message_shape, dtype=torch.bfloat16, device="cuda")
        for _ in range(world_size)
    )
    forward_receive = tuple(torch.empty_like(forward_send[0]) for _ in range(world_size))
    baseline_column_output = torch.empty(
        (args.batch, args.n, sequence_count, MODEL_DIMENSION),
        dtype=torch.bfloat16,
        device="cuda",
    )
    baseline_column_gradient = torch.randn_like(
        baseline_column_output, generator=generator
    )
    backward_send = tuple(
        torch.empty(baseline_message_shape, dtype=torch.bfloat16, device="cuda")
        for _ in range(world_size)
    )
    backward_receive = tuple(
        torch.empty_like(backward_send[0]) for _ in range(world_size)
    )
    baseline_row_gradient = torch.empty_like(residual_input)

    # The attention gradient is prepared outside the timed region because the
    # omitted output projection would otherwise map C to H*D.
    column_gradient_storage = nvshmem.tensor(
        (
            args.batch,
            world_size,
            sequence_count,
            sequence_count,
            args.heads,
            HEAD_DIMENSION,
        ),
        dtype=torch.bfloat16,
    )
    local_column_output_gradient = column_gradient_storage.reshape(
        args.batch,
        args.n,
        sequence_count,
        args.heads,
        HEAD_DIMENSION,
    )
    local_column_output_gradient.normal_()
    output_gradient_peers = tuple(
        nvshmem.get_peer_tensor(column_gradient_storage, peer)[:, rank]
        for peer in range(world_size)
    )
    reference_output_gradient = _all_gather_col_to_row_reference(
        local_column_output_gradient, rank, sequence_count
    )
    triton_output_gradient = torch.empty_like(triton_output)
    triton_delta = torch.empty_like(triton_lse)
    triton_dq = torch.empty_like(triton_query)
    triton_dk = torch.empty_like(triton_query)
    triton_dv = torch.empty_like(triton_query)
    pair_shape = (args.batch, args.heads, args.n, args.n)
    triton_dpb = torch.empty(pair_shape, dtype=torch.float32, device="cuda")
    triton_dq_public = torch.empty_like(query)
    triton_dk_public = torch.empty_like(query)
    triton_dv_public = torch.empty_like(query)
    triton_dpb_output = torch.empty(
        pair_shape, dtype=torch.bfloat16, device="cuda"
    )

    # CuTe forward output and synchronization buffers.
    cute_pair_bias = torch.empty(
        (args.batch, 1, args.heads, args.n, args.n),
        dtype=torch.bfloat16,
        device="cuda",
    )
    cute_local_output = torch.empty_like(query)
    cute_lse = torch.empty_like(triton_lse)
    output_storage = nvshmem.tensor(
        (
            args.batch,
            world_size,
            sequence_count,
            sequence_count,
            args.heads,
            HEAD_DIMENSION,
        ),
        dtype=torch.bfloat16,
    )
    output_peers = tuple(
        nvshmem.get_peer_tensor(output_storage, peer)[:, rank]
        for peer in range(world_size)
    )
    residual_output_storage = nvshmem.tensor(
        (
            args.batch,
            world_size,
            sequence_count,
            sequence_count,
            MODEL_DIMENSION,
        ),
        dtype=torch.bfloat16,
    )
    residual_output_peers = tuple(
        nvshmem.get_peer_tensor(residual_output_storage, peer)[:, rank]
        for peer in range(world_size)
    )
    completion_storage = nvshmem.tensor((world_size,), dtype=torch.int32)
    completion_storage.zero_()
    completion_peers = tuple(
        nvshmem.get_peer_tensor(completion_storage, peer)[rank : rank + 1]
        for peer in range(world_size)
    )
    completion_counter = torch.zeros((1,), dtype=torch.int32, device="cuda")

    residual_column_gradient_storage = nvshmem.tensor(
        (
            args.batch,
            world_size,
            sequence_count,
            sequence_count,
            MODEL_DIMENSION,
        ),
        dtype=torch.bfloat16,
    )
    local_column_residual_gradient = residual_column_gradient_storage.reshape(
        args.batch,
        args.n,
        sequence_count,
        MODEL_DIMENSION,
    )
    local_column_residual_gradient.normal_()
    residual_gradient_peers = tuple(
        nvshmem.get_peer_tensor(residual_column_gradient_storage, peer)[:, rank]
        for peer in range(world_size)
    )
    dist.barrier()

    cute_output_gradient = torch.empty_like(query)
    cute_residual_gradient = torch.empty_like(residual_input)
    cute_delta = torch.empty_like(cute_lse)
    cute_dq = torch.empty_like(query)
    cute_dk = torch.empty_like(query)
    cute_dv = torch.empty_like(query)
    cute_dpb = torch.empty(pair_shape, dtype=torch.float32, device="cuda")
    cute_dpb_output = torch.empty(
        pair_shape, dtype=torch.bfloat16, device="cuda"
    )

    reference = load_reference_module()
    cute_forward = get_distributed_evoattention_forward(world_size, rank)
    cute_backward = get_distributed_evoattention_backward_no_preprocess()
    remote_preprocess = get_remote_backward_preprocess(world_size, rank)
    query_gradient_accumulator = cute_backward._workspace(
        args.batch,
        sequence_count,
        args.n,
        args.heads,
        query.device,
    )
    epoch = 0

    def triton_forward_step() -> None:
        triton_key_public.copy_(key)
        triton_value_public.copy_(value)
        launch_triton_forward(
            reference,
            query,
            triton_key_public,
            triton_value_public,
            pair_bias_source,
            residual_mask,
            triton_query,
            triton_key,
            triton_value,
            triton_pair_bias,
            additive_mask,
            triton_output,
            triton_output_public,
            triton_lse,
        )

    def baseline_forward_communication() -> None:
        row_to_column(
            residual_input,
            forward_send,
            forward_receive,
            baseline_column_output,
        )

    def baseline_backward_communication() -> None:
        column_to_row(
            baseline_column_gradient,
            backward_send,
            backward_receive,
            baseline_row_gradient,
        )

    def triton_backward_step() -> None:
        triton_output_gradient.copy_(reference_output_gradient.transpose(2, 3))
        launch_triton_backward(
            reference,
            triton_query,
            triton_key,
            triton_value,
            triton_output,
            triton_output_gradient,
            triton_pair_bias,
            additive_mask,
            triton_lse,
            triton_delta,
            triton_dq,
            triton_dk,
            triton_dv,
            triton_dpb,
        )
        triton_dq_public.copy_(triton_dq.transpose(2, 3))
        triton_dk_public.copy_(triton_dk.transpose(2, 3))
        triton_dv_public.copy_(triton_dv.transpose(2, 3))
        triton_dpb_output.copy_(triton_dpb)

    def baseline_step() -> None:
        triton_forward_step()
        baseline_forward_communication()
        baseline_backward_communication()
        triton_backward_step()

    def cute_prepare_step() -> None:
        cute_pair_bias[:, 0].copy_(pair_bias_source)

    def cute_forward_step() -> None:
        nonlocal epoch
        epoch += 1
        cute_forward(
            query,
            key,
            value,
            cute_pair_bias,
            residual_mask,
            residual_input,
            cute_local_output,
            output_peers,
            residual_output_peers,
            completion_peers,
            completion_storage,
            completion_counter,
            epoch,
            cute_lse,
        )

    def cute_remote_preprocess_step() -> None:
        remote_preprocess(
            output_gradient_peers,
            residual_gradient_peers,
            cute_local_output,
            cute_output_gradient,
            cute_residual_gradient,
            cute_delta,
            query_gradient_accumulator,
            cute_dpb,
        )

    def cute_physical_backward_step() -> None:
        cute_backward(
            query,
            key,
            value,
            cute_local_output,
            cute_output_gradient,
            cute_pair_bias,
            residual_mask,
            cute_lse,
            cute_delta,
            cute_dq,
            cute_dk,
            cute_dv,
            cute_dpb,
            cute_dpb_output,
        )

    def cute_backward_step() -> None:
        cute_remote_preprocess_step()
        cute_physical_backward_step()

    def cute_step() -> None:
        cute_prepare_step()
        cute_forward_step()
        cute_backward_step()

    # Compile both paths and establish correctness before timing.
    triton_forward_step()
    cute_prepare_step()
    cute_forward_step()
    cute_backward_step()
    triton_backward_step()
    torch.cuda.synchronize()

    if not args.skip_check:
        gathered_outputs = [
            torch.empty_like(triton_output_public) for _ in range(world_size)
        ]
        dist.all_gather(gathered_outputs, triton_output_public)
        expected_column_output = torch.cat(gathered_outputs, dim=1)[
            :, :, rank * sequence_count : (rank + 1) * sequence_count
        ]
        actual_column_output = output_storage.reshape(
            args.batch,
            args.n,
            sequence_count,
            args.heads,
            HEAD_DIMENSION,
        )
        gathered_inputs = [torch.empty_like(residual_input) for _ in range(world_size)]
        dist.all_gather(gathered_inputs, residual_input)
        expected_column_input = torch.cat(gathered_inputs, dim=1)[
            :, :, rank * sequence_count : (rank + 1) * sequence_count
        ]
        actual_column_input = residual_output_storage.reshape(
            args.batch, args.n, sequence_count, MODEL_DIMENSION
        )
        expected_residual_gradient = _all_gather_col_to_row_reference(
            local_column_residual_gradient, rank, sequence_count
        )

        if rank == 0:
            print("correctness:")
        _check_close(
            "O_col",
            actual_column_output,
            expected_column_output,
            atol=2e-2,
            rtol=2e-2,
            rank=rank,
        )
        _check_close(
            "X_col",
            actual_column_input,
            expected_column_input,
            atol=0,
            rtol=0,
            rank=rank,
        )
        _check_close(
            "dX_row",
            cute_residual_gradient,
            expected_residual_gradient,
            atol=0,
            rtol=0,
            rank=rank,
        )
        for name, actual, expected in (
            ("dQ", cute_dq, triton_dq_public),
            ("dK", cute_dk, triton_dk_public),
            ("dV", cute_dv, triton_dv_public),
            ("dPairBias", cute_dpb_output, triton_dpb_output),
        ):
            _check_close(
                name,
                actual,
                expected,
                atol=6e-2,
                rtol=6e-2,
                rank=rank,
            )

    baseline_ms = statistics.median(
        measure_distributed(
            baseline_step, args.warmup, args.iterations, args.repeats
        )
    )
    cute_ms = statistics.median(
        measure_distributed(cute_step, args.warmup, args.iterations, args.repeats)
    )
    speedup = baseline_ms / cute_ms
    if rank == 0:
        print(
            f"B={args.batch} H={args.heads} N={args.n} D={HEAD_DIMENSION} "
            f"C={MODEL_DIMENSION} P={world_size} "
            f"baseline={baseline_ms:.4f} ms cute={cute_ms:.4f} ms "
            f"speedup={speedup:.3f}x target={'PASS' if speedup >= 2.5 else 'FAIL'}"
        )

    if args.breakdown:
        components = (
            ("triton_forward", triton_forward_step),
            ("baseline_row_to_col", baseline_forward_communication),
            ("baseline_col_to_row", baseline_backward_communication),
            ("triton_backward", triton_backward_step),
            ("cute_prepare", cute_prepare_step),
            ("cute_forward_fused", cute_forward_step),
            ("cute_remote_preprocess", cute_remote_preprocess_step),
            ("cute_physical_backward", cute_physical_backward_step),
            ("cute_backward_fused", cute_backward_step),
        )
        for name, step in components:
            milliseconds = statistics.median(
                measure_distributed(step, 3, args.iterations, 3)
            )
            if rank == 0:
                print(f"  {name:24s} {milliseconds:.4f} ms")

    nvshmem.free_tensor(residual_column_gradient_storage)
    nvshmem.free_tensor(column_gradient_storage)
    nvshmem.free_tensor(completion_storage)
    nvshmem.free_tensor(residual_output_storage)
    nvshmem.free_tensor(output_storage)
    nvshmem.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
