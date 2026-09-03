"""Kernel-only EvoAttention forward benchmark: Triton versus CuTe-DSL.

Compilation, tensor allocation, layout preparation, and warmup are excluded.
Both kernels receive the same prepared buffers described in
``test_kernel.py`` and write BF16 output plus FP32 logsumexp in place. Prepared
addresses rotate through a memory-bounded pool to avoid benchmarking an
unrealistically hot single allocation. Timing uses CUDA events; the default run
covers all 32 acceptance shapes and exits nonzero unless CuTe is at least 2x
faster on every shape.

Examples:
    CUDA_VISIBLE_DEVICES=2 python benchmark.py
    CUDA_VISIBLE_DEVICES=3 python benchmark.py --quick --no-enforce
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from test_kernel import (
    HEAD_DIMENSION,
    SEQUENCE_LENGTHS,
    EvoAttentionInputs,
    EvoAttentionOutputs,
    EvoAttentionShape,
    ForwardKernel,
    assert_single_compilation,
    launch_cute,
    launch_reference,
    load_cute_forward,
    make_inputs,
    make_outputs,
    select_shapes,
)


DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 5
DEFAULT_TARGET_MS = 200.0
MIN_ITERS = 3
MAX_ITERS = 1000
REQUIRED_SPEEDUP = 2.0
INPUT_POOL_BUDGET_BYTES = 2 * 1024**3
MAX_POOL_SLOTS = 8


@dataclass(frozen=True)
class TimingStats:
    median_ms: float
    mean_ms: float
    minimum_ms: float
    maximum_ms: float
    iterations: int
    repeats: int


@dataclass(frozen=True)
class BenchmarkResult:
    batch_size: int
    num_heads: int
    context_length: int
    pool_slots: int
    triton: TimingStats
    cute: TimingStats

    @property
    def speedup(self) -> float:
        return self.triton.median_ms / self.cute.median_ms

    @property
    def triton_tflops(self) -> float:
        return _flops_from_dimensions(
            self.batch_size,
            self.num_heads,
            self.context_length,
        ) / (self.triton.median_ms * 1.0e9)

    @property
    def cute_tflops(self) -> float:
        return _flops_from_dimensions(
            self.batch_size,
            self.num_heads,
            self.context_length,
        ) / (self.cute.median_ms * 1.0e9)


@dataclass(frozen=True)
class PreparedSlot:
    """One allocation-free benchmark slot with prepared inputs and outputs."""

    inputs: EvoAttentionInputs
    outputs: EvoAttentionOutputs


def _slot_bytes(shape: EvoAttentionShape) -> int:
    """Return the exact tensor bytes allocated for one benchmark slot."""
    activation_elements = (
        shape.batch_size
        * shape.num_sequences
        * shape.num_heads
        * shape.context_length
        * HEAD_DIMENSION
    )
    pair_bias_elements = (
        shape.batch_size * shape.num_heads * shape.context_length * shape.context_length
    )
    residual_mask_elements = (
        shape.batch_size * shape.num_sequences * shape.context_length
    )
    logsumexp_elements = (
        shape.batch_size * shape.num_sequences * shape.num_heads * shape.context_length
    )
    return (
        4 * activation_elements * 2
        + pair_bias_elements * 4
        + residual_mask_elements * 4
        + logsumexp_elements * 4
    )


def _pool_size(shape: EvoAttentionShape) -> int:
    """Use multiple addresses when memory permits, without risking large OOMs."""
    return max(
        1,
        min(MAX_POOL_SLOTS, INPUT_POOL_BUDGET_BYTES // _slot_bytes(shape)),
    )


def _make_pool(
    shape: EvoAttentionShape,
    device: torch.device,
    *,
    seed: int,
) -> tuple[PreparedSlot, ...]:
    pool = []
    for slot in range(_pool_size(shape)):
        inputs = make_inputs(shape, device, seed=seed + slot)
        pool.append(PreparedSlot(inputs=inputs, outputs=make_outputs(inputs)))
    return tuple(pool)


BenchmarkStep = Callable[[int], None]


def _elapsed_ms(
    step: BenchmarkStep,
    *,
    first_iteration: int,
    iterations: int,
) -> float:
    """Average device time per launch, measured by a pair of CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for offset in range(iterations):
        step(first_iteration + offset)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _calibrated_iterations(steps: Sequence[BenchmarkStep], target_ms: float) -> int:
    # Use the slower implementation, keeping every repeat long enough to dwarf
    # event resolution while preventing the largest cubic cases from dominating
    # the whole suite.
    single_ms = max(
        _elapsed_ms(step, first_iteration=0, iterations=1) for step in steps
    )
    if single_ms <= 0.0:
        return MAX_ITERS
    return max(MIN_ITERS, min(MAX_ITERS, round(target_ms / single_ms)))


def time_cuda_events(
    step: BenchmarkStep,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> TimingStats:
    """Time an already-compiled, allocation-free callable with CUDA events."""
    for iteration in range(warmup):
        step(iteration)
    torch.cuda.synchronize()
    samples = [
        _elapsed_ms(
            step,
            first_iteration=warmup + repeat * iterations,
            iterations=iterations,
        )
        for repeat in range(repeats)
    ]
    return TimingStats(
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
        iterations=iterations,
        repeats=repeats,
    )


def _flops_from_dimensions(
    batch_size: int,
    num_heads: int,
    context_length: int,
) -> int:
    # QK^T and PV, two FLOPs per multiply-accumulate.  The softmax and bias
    # arithmetic are intentionally omitted, matching standard FA reporting.
    return (
        4 * batch_size * context_length * num_heads * context_length**2 * HEAD_DIMENSION
    )


def benchmark_shape(
    forward: ForwardKernel,
    shape: EvoAttentionShape,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    target_ms: float,
    fixed_iters: int | None,
    seed: int,
) -> BenchmarkResult:
    pool = _make_pool(shape, device, seed=seed)

    def triton_step(iteration: int) -> None:
        slot = pool[iteration % len(pool)]
        launch_reference(slot.inputs, slot.outputs)

    def cute_step(iteration: int) -> None:
        slot = pool[iteration % len(pool)]
        launch_cute(forward, slot.inputs, slot.outputs)

    # The first call can compile/JIT.  Synchronization places all compilation
    # and initialization firmly before calibration and timed event ranges.
    with torch.inference_mode():
        triton_step(0)
        cute_step(0)
    torch.cuda.synchronize(device)

    iterations = fixed_iters or _calibrated_iterations(
        (triton_step, cute_step), target_ms
    )

    # Alternate order by shape so clock/temperature drift does not systematically
    # favor either implementation across the table.
    if (shape.batch_size + shape.num_heads + shape.context_length // 128) % 2:
        cute_timing = time_cuda_events(
            cute_step, warmup=warmup, iterations=iterations, repeats=repeats
        )
        triton_timing = time_cuda_events(
            triton_step, warmup=warmup, iterations=iterations, repeats=repeats
        )
    else:
        triton_timing = time_cuda_events(
            triton_step, warmup=warmup, iterations=iterations, repeats=repeats
        )
        cute_timing = time_cuda_events(
            cute_step, warmup=warmup, iterations=iterations, repeats=repeats
        )

    result = BenchmarkResult(
        batch_size=shape.batch_size,
        num_heads=shape.num_heads,
        context_length=shape.context_length,
        pool_slots=len(pool),
        triton=triton_timing,
        cute=cute_timing,
    )

    del pool
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _print_header() -> None:
    print(
        f"{'B':>2} {'H':>3} {'N=S':>5} {'slots':>5}  "
        f"{'Triton ms':>11} {'CuTe ms':>11} {'speedup':>9}  "
        f"{'Triton TF/s':>12} {'CuTe TF/s':>10} {'iters':>6} {'pass':>5}"
    )
    print("-" * 106)


def _print_result(result: BenchmarkResult, required_speedup: float) -> None:
    passed = result.speedup >= required_speedup
    print(
        f"{result.batch_size:2d} {result.num_heads:3d} "
        f"{result.context_length:5d} {result.pool_slots:5d}  "
        f"{result.triton.median_ms:11.4f} {result.cute.median_ms:11.4f} "
        f"{result.speedup:8.3f}x  {result.triton_tflops:12.2f} "
        f"{result.cute_tflops:10.2f} {result.cute.iterations:6d} "
        f"{'yes' if passed else 'NO':>5}",
        flush=True,
    )


def _write_json(
    path: Path,
    results: Sequence[BenchmarkResult],
    args: argparse.Namespace,
) -> None:
    serialized_results = []
    for result in results:
        serialized = asdict(result)
        serialized.update(
            speedup=result.speedup,
            triton_tflops=result.triton_tflops,
            cute_tflops=result.cute_tflops,
        )
        serialized_results.append(serialized)
    payload = {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "required_speedup": args.required_speedup,
        "timing": (
            "CUDA events, prepared-layout kernel only, median of repeat "
            "averages, rotating address pool"
        ),
        "results": serialized_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run three smoke shapes")
    parser.add_argument("--n", type=int, choices=SEQUENCE_LENGTHS)
    parser.add_argument("--batch", type=int, choices=(1, 4))
    parser.add_argument("--heads", type=int, choices=(4, 16))
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--target-ms",
        type=float,
        default=DEFAULT_TARGET_MS,
        help="target duration of each repeat when --iters is omitted",
    )
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--required-speedup", type=float, default=REQUIRED_SPEEDUP)
    parser.add_argument(
        "--no-enforce",
        action="store_true",
        help="report sub-2x shapes without returning failure",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.iters is not None and args.iters < 1:
        parser.error("--iters must be positive")
    if args.warmup < 0 or args.repeats < 1 or args.target_ms <= 0:
        parser.error("warmup must be nonnegative; repeats and target-ms positive")

    device = torch.device("cuda", torch.cuda.current_device())
    print(f"device: {torch.cuda.get_device_name(device)} ({device})")
    print("timing: prepared-layout kernel only, CUDA events, compile/warmup excluded")
    print(f"required speedup: {args.required_speedup:.3f}x on every selected shape\n")

    forward = load_cute_forward()
    results: list[BenchmarkResult] = []
    _print_header()
    for shape in select_shapes(
        quick=args.quick,
        context_length=args.n,
        batch_size=args.batch,
        num_heads=args.heads,
    ):
        result = benchmark_shape(
            forward,
            shape,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            target_ms=args.target_ms,
            fixed_iters=args.iters,
            seed=args.seed,
        )
        results.append(result)
        _print_result(result, args.required_speedup)

    assert_single_compilation(forward)
    failures = [r for r in results if r.speedup < args.required_speedup]
    print(
        f"\nsummary: {len(results) - len(failures)}/{len(results)} shapes meet "
        f"{args.required_speedup:.3f}x; CuTe compile_count={forward.compile_count}"
    )
    if args.json is not None:
        _write_json(args.json, results, args)
        print(f"wrote {args.json}")

    if failures and not args.no_enforce:
        failed = ", ".join(
            f"B{result.batch_size}/H{result.num_heads}/N{result.context_length}"
            for result in failures
        )
        raise SystemExit(f"speed target missed for: {failed}")


if __name__ == "__main__":
    main()
