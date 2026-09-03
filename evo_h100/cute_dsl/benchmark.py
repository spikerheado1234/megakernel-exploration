"""EvoAttention forward, backward, and combined benchmark: Triton vs CuTe-DSL.

Compilation, allocation, input layout preparation, and warmup are excluded.
Prepared addresses rotate through a memory-bounded pool to avoid measuring an
unrealistically hot allocation. Forward timing writes BF16 output and FP32 LSE.
Backward timing includes every launch/reduction needed for model-layout BF16
dQ/dK/dV/dPairBias; this includes the reference's three output transposes.
Combined timing encloses a dependent forward followed by the complete backward
pass in one CUDA-event range. One CUDA graph is captured per rotating address
slot so Python launch gaps are excluded. The default run covers all 32 shapes
and enforces a 2x CuTe speedup.

Examples:
    CUDA_VISIBLE_DEVICES=1 python benchmark.py --direction combined
    CUDA_VISIBLE_DEVICES=2 python benchmark.py --direction forward
    CUDA_VISIBLE_DEVICES=3 python benchmark.py --direction backward --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch
import triton

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
    reference_module,
    select_shapes,
)


DEFAULT_WARMUP = 10
DEFAULT_REPEATS = 5
DEFAULT_TARGET_MS = 200.0
MIN_ITERS = 3
MAX_ITERS = 1000
REQUIRED_SPEEDUP = 2.0
SOFTMAX_SCALE = HEAD_DIMENSION**-0.5
INPUT_POOL_BUDGET_BYTES = 2 * 1024**3
BACKWARD_POOL_BUDGET_BYTES = 70 * 1024**3
COMBINED_POOL_BUDGET_BYTES = 80 * 1024**3
GPU_MEMORY_RESERVE_BYTES = 8 * 1024**3
TRITON_FULL_BATCH_LIVE_BYTES_PER_ELEMENT = 18
TRITON_FULL_BATCH_MEMORY_LIMIT_BYTES = 60 * 1024**3
MAX_POOL_SLOTS = 8
MAX_GRID_Y = 65_535


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


@dataclass(frozen=True)
class BackwardBuffers:
    """Internal-layout buffers used by the authoritative Triton backward."""

    inputs: EvoAttentionInputs
    forward: EvoAttentionOutputs
    output_gradient: torch.Tensor
    delta: torch.Tensor
    query_gradient: torch.Tensor
    key_gradient: torch.Tensor
    value_gradient: torch.Tensor
    pair_bias_accumulator: torch.Tensor
    pair_bias_gradient: torch.Tensor


@dataclass(frozen=True)
class TritonBackwardSlot:
    buffers: BackwardBuffers
    public_query_gradient: torch.Tensor
    public_key_gradient: torch.Tensor
    public_value_gradient: torch.Tensor
    gradient_batch_capacity: int


@dataclass(frozen=True)
class CuteBackwardSlot:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    output: torch.Tensor
    output_gradient: torch.Tensor
    pair_bias: torch.Tensor
    residual_mask: torch.Tensor
    logsumexp: torch.Tensor
    delta: torch.Tensor
    query_gradient: torch.Tensor
    key_gradient: torch.Tensor
    value_gradient: torch.Tensor
    pair_bias_accumulator: torch.Tensor
    pair_bias_gradient: torch.Tensor


@dataclass(frozen=True)
class CuteCombinedChunk:
    """One batch chunk connecting the forward output to backward inputs."""

    forward_output: torch.Tensor
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    output: torch.Tensor
    output_gradient: torch.Tensor
    pair_bias: torch.Tensor
    residual_mask: torch.Tensor
    logsumexp: torch.Tensor
    delta: torch.Tensor
    query_gradient: torch.Tensor
    key_gradient: torch.Tensor
    value_gradient: torch.Tensor
    pair_bias_accumulator: torch.Tensor
    pair_bias_gradient: torch.Tensor


@dataclass(frozen=True)
class CuteCombinedSlot:
    """Prepared buffers for one dependent CuTe forward-plus-backward call."""

    forward_inputs: EvoAttentionInputs
    forward_outputs: EvoAttentionOutputs
    backward_chunks: tuple[CuteCombinedChunk, ...]


class BackwardKernel(Protocol):
    """Prepared-layout interface exposed by the CuTe backward launcher."""

    @property
    def compile_count(self) -> int: ...

    def clear_workspace_cache(self) -> None: ...

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        output_gradient: torch.Tensor,
        pair_bias: torch.Tensor,
        residual_mask: torch.Tensor,
        logsumexp: torch.Tensor,
        delta: torch.Tensor,
        query_gradient: torch.Tensor,
        key_gradient: torch.Tensor,
        value_gradient: torch.Tensor,
        pair_bias_accumulator: torch.Tensor,
        pair_bias_gradient: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class BackwardBenchmarkResult:
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
    def cute_tflops(self) -> float:
        # Five mathematical GEMMs; recomputation is intentionally excluded.
        return _backward_flops_from_dimensions(
            self.batch_size,
            self.num_heads,
            self.context_length,
        ) / (self.cute.median_ms * 1.0e9)


@dataclass(frozen=True)
class CombinedBenchmarkResult:
    batch_size: int
    num_heads: int
    context_length: int
    pool_slots: int
    backward_batch_chunk: int
    triton: TimingStats
    cute: TimingStats

    @property
    def speedup(self) -> float:
        return self.triton.median_ms / self.cute.median_ms

    @property
    def triton_tflops(self) -> float:
        return _combined_flops_from_dimensions(
            self.batch_size,
            self.num_heads,
            self.context_length,
        ) / (self.triton.median_ms * 1.0e9)

    @property
    def cute_tflops(self) -> float:
        return _combined_flops_from_dimensions(
            self.batch_size,
            self.num_heads,
            self.context_length,
        ) / (self.cute.median_ms * 1.0e9)


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


def _backward_flops_from_dimensions(
    batch_size: int,
    num_heads: int,
    context_length: int,
) -> int:
    """Return algorithmic backward FLOPs for its five matrix products."""
    return (
        10
        * batch_size
        * context_length
        * num_heads
        * context_length**2
        * HEAD_DIMENSION
    )


def _combined_flops_from_dimensions(
    batch_size: int,
    num_heads: int,
    context_length: int,
) -> int:
    """Return forward plus backward algorithmic FLOPs."""
    return _flops_from_dimensions(
        batch_size,
        num_heads,
        context_length,
    ) + _backward_flops_from_dimensions(
        batch_size,
        num_heads,
        context_length,
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


def _make_public_inputs(
    shape: EvoAttentionShape,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    """Create one logical problem directly in the model-facing layout."""
    generator = torch.Generator(device=device)
    generator.manual_seed(
        seed
        ^ (shape.batch_size * 0x1F1F)
        ^ (shape.num_heads * 0x101)
        ^ shape.context_length
    )
    public_shape = (
        shape.batch_size,
        shape.num_sequences,
        shape.context_length,
        shape.num_heads,
        HEAD_DIMENSION,
    )
    q = torch.randn(
        public_shape, dtype=torch.bfloat16, device=device, generator=generator
    )
    k = torch.randn(
        public_shape, dtype=torch.bfloat16, device=device, generator=generator
    )
    v = torch.randn(
        public_shape, dtype=torch.bfloat16, device=device, generator=generator
    )
    pair_bias = torch.randn(
        (
            shape.batch_size,
            shape.num_heads,
            shape.context_length,
            shape.context_length,
        ),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    residual_mask = torch.empty(
        (shape.batch_size, shape.num_sequences, shape.context_length),
        dtype=torch.float32,
        device=device,
    )
    residual_mask.bernoulli_(0.125, generator=generator).mul_(-1.0e9)
    residual_mask[..., 0] = 0.0
    return q, k, v, pair_bias, residual_mask, generator


def _to_triton_layout(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(2, 3).contiguous()


def _make_triton_backward_slot(
    shape: EvoAttentionShape,
    device: torch.device,
    seed: int,
    *,
    gradient_batch_capacity: int | None = None,
) -> TritonBackwardSlot:
    q_public, k_public, v_public, pair_bias, residual_mask, generator = (
        _make_public_inputs(shape, device, seed)
    )
    public_shape = q_public.shape
    q, k, v = map(_to_triton_layout, (q_public, k_public, v_public))
    inputs = EvoAttentionInputs(q, k, v, pair_bias, residual_mask)
    forward = make_outputs(inputs)
    launch_reference(inputs, forward)
    output_gradient = _to_triton_layout(
        torch.randn(
            public_shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
    )
    del q_public, k_public, v_public
    pair_shape = pair_bias.shape
    activation_elements = _backward_activation_elements(shape)
    # Keep the full model-layout outputs but bound Triton's internal-layout
    # gradient scratch at the maximum shape.
    if gradient_batch_capacity is None:
        gradient_batch_capacity = (
            1
            if TRITON_FULL_BATCH_LIVE_BYTES_PER_ELEMENT * activation_elements
            > TRITON_FULL_BATCH_MEMORY_LIMIT_BYTES
            else shape.batch_size
        )
    if not 1 <= gradient_batch_capacity <= shape.batch_size:
        raise ValueError("gradient batch capacity must be in [1, batch_size]")
    gradient_shape = (
        gradient_batch_capacity,
        shape.num_sequences,
        shape.num_heads,
        shape.context_length,
        HEAD_DIMENSION,
    )
    statistics_shape = gradient_shape[:-1]
    buffers = BackwardBuffers(
        inputs=inputs,
        forward=forward,
        output_gradient=output_gradient,
        delta=torch.empty(statistics_shape, dtype=torch.float32, device=device),
        query_gradient=torch.empty(gradient_shape, dtype=torch.bfloat16, device=device),
        key_gradient=torch.empty(gradient_shape, dtype=torch.bfloat16, device=device),
        value_gradient=torch.empty(gradient_shape, dtype=torch.bfloat16, device=device),
        pair_bias_accumulator=torch.empty(
            pair_shape, dtype=torch.float32, device=device
        ),
        pair_bias_gradient=torch.empty(pair_shape, dtype=torch.bfloat16, device=device),
    )
    return TritonBackwardSlot(
        buffers=buffers,
        public_query_gradient=torch.empty(
            public_shape, dtype=torch.bfloat16, device=device
        ),
        public_key_gradient=torch.empty(
            public_shape, dtype=torch.bfloat16, device=device
        ),
        public_value_gradient=torch.empty(
            public_shape, dtype=torch.bfloat16, device=device
        ),
        gradient_batch_capacity=gradient_batch_capacity,
    )


def _make_cute_backward_slot(
    shape: EvoAttentionShape,
    device: torch.device,
    seed: int,
) -> CuteBackwardSlot:
    q, k, v, pair_bias, residual_mask, generator = _make_public_inputs(
        shape, device, seed
    )
    internal_inputs = EvoAttentionInputs(
        _to_triton_layout(q),
        _to_triton_layout(k),
        _to_triton_layout(v),
        pair_bias,
        residual_mask,
    )
    internal_forward = make_outputs(internal_inputs)
    launch_reference(internal_inputs, internal_forward)
    output = internal_forward.output.transpose(2, 3).contiguous()
    logsumexp = internal_forward.logsumexp
    del internal_inputs, internal_forward
    output_gradient = torch.randn(
        q.shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    pair_shape = pair_bias.shape
    return CuteBackwardSlot(
        q=q,
        k=k,
        v=v,
        output=output,
        output_gradient=output_gradient,
        pair_bias=pair_bias,
        residual_mask=residual_mask,
        logsumexp=logsumexp,
        delta=torch.empty_like(logsumexp),
        query_gradient=torch.empty_like(q),
        key_gradient=torch.empty_like(k),
        value_gradient=torch.empty_like(v),
        pair_bias_accumulator=torch.empty(
            pair_shape, dtype=torch.float32, device=device
        ),
        pair_bias_gradient=torch.empty(pair_shape, dtype=torch.bfloat16, device=device),
    )


def _make_cute_combined_slot(
    shape: EvoAttentionShape,
    device: torch.device,
    seed: int,
    backward_batch_chunk: int,
) -> CuteCombinedSlot:
    """Prepare a true forward-to-backward CuTe dataflow without allocations.

    The forward kernel consumes the prepared ``[B,S,H,N,D]`` layout, whereas
    the backward kernel consumes ``[B,S,N,H,D]``. Both Q/K/V layouts are
    prepared before timing. Only the required forward-output transpose remains
    inside the measured collective.

    Backward output buffers are sized to one batch chunk and reused serially.
    This keeps the largest B=4/H=16/N=1024 collective below H100 memory limits
    without changing the full-batch forward launch.
    """
    q, k, v, pair_bias, residual_mask, generator = _make_public_inputs(
        shape, device, seed
    )
    forward_inputs = EvoAttentionInputs(
        q=_to_triton_layout(q),
        k=_to_triton_layout(k),
        v=_to_triton_layout(v),
        pair_bias=pair_bias,
        residual_mask=residual_mask,
    )
    forward_outputs = make_outputs(forward_inputs)

    public_shape = q.shape
    chunk_shape = (backward_batch_chunk, *public_shape[1:])
    pair_chunk_shape = (
        backward_batch_chunk,
        shape.num_heads,
        shape.context_length,
        shape.context_length,
    )
    statistic_chunk_shape = (
        backward_batch_chunk,
        shape.num_sequences,
        shape.num_heads,
        shape.context_length,
    )
    output_buffer = torch.empty(chunk_shape, dtype=torch.bfloat16, device=device)
    query_gradient = torch.empty_like(output_buffer)
    key_gradient = torch.empty_like(output_buffer)
    value_gradient = torch.empty_like(output_buffer)
    output_gradient = torch.randn(
        public_shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    delta = torch.empty(statistic_chunk_shape, dtype=torch.float32, device=device)
    pair_bias_accumulator = torch.empty(
        pair_chunk_shape, dtype=torch.float32, device=device
    )
    pair_bias_gradient = torch.empty(
        pair_chunk_shape, dtype=torch.bfloat16, device=device
    )

    chunks = []
    for start in range(0, shape.batch_size, backward_batch_chunk):
        stop = min(shape.batch_size, start + backward_batch_chunk)
        chunk_batch = stop - start
        chunks.append(
            CuteCombinedChunk(
                forward_output=forward_outputs.output[start:stop],
                q=q[start:stop],
                k=k[start:stop],
                v=v[start:stop],
                output=output_buffer[:chunk_batch],
                output_gradient=output_gradient[start:stop],
                pair_bias=pair_bias[start:stop],
                residual_mask=residual_mask[start:stop],
                logsumexp=forward_outputs.logsumexp[start:stop],
                delta=delta[:chunk_batch],
                query_gradient=query_gradient[:chunk_batch],
                key_gradient=key_gradient[:chunk_batch],
                value_gradient=value_gradient[:chunk_batch],
                pair_bias_accumulator=pair_bias_accumulator[:chunk_batch],
                pair_bias_gradient=pair_bias_gradient[:chunk_batch],
            )
        )
    return CuteCombinedSlot(
        forward_inputs=forward_inputs,
        forward_outputs=forward_outputs,
        backward_chunks=tuple(chunks),
    )


def _backward_activation_elements(shape: EvoAttentionShape) -> int:
    return (
        shape.batch_size
        * shape.num_sequences
        * shape.context_length
        * shape.num_heads
        * HEAD_DIMENSION
    )


def _backward_slot_bytes(
    shape: EvoAttentionShape,
    *,
    triton_reference: bool,
) -> int:
    activation = _backward_activation_elements(shape)
    row = (
        shape.batch_size * shape.num_sequences * shape.num_heads * shape.context_length
    )
    pair = (
        shape.batch_size * shape.num_heads * shape.context_length * shape.context_length
    )
    mask = shape.batch_size * shape.num_sequences * shape.context_length
    slot_bytes = 16 * activation + 8 * row + 10 * pair + 4 * mask
    if triton_reference:
        slot_bytes += 6 * activation
    else:
        # CuTe's largest cached temporary is an FP32 dQ accumulator.  The
        # launcher bounds it to 8 GiB by splitting B when necessary.
        slot_bytes += min(8 * 1024**3, 4 * activation)
        slot_bytes += 4 * row
    return slot_bytes


def _backward_pool_size(
    shape: EvoAttentionShape,
    *,
    triton_reference: bool,
) -> int:
    slot_bytes = _backward_slot_bytes(shape, triton_reference=triton_reference)
    return max(1, min(MAX_POOL_SLOTS, BACKWARD_POOL_BUDGET_BYTES // slot_bytes))


def _cute_combined_slot_bytes(
    shape: EvoAttentionShape,
    backward_batch_chunk: int,
) -> int:
    """Estimate retained tensors plus cached CuTe workspace for one slot."""
    activation = _backward_activation_elements(shape)
    activation_bytes = activation * 2
    chunk_fraction_numerator = backward_batch_chunk
    chunk_fraction_denominator = shape.batch_size
    chunk_activation_bytes = (
        activation_bytes * chunk_fraction_numerator // chunk_fraction_denominator
    )
    row = (
        shape.batch_size * shape.num_sequences * shape.num_heads * shape.context_length
    )
    chunk_row = row * chunk_fraction_numerator // chunk_fraction_denominator
    pair = (
        shape.batch_size * shape.num_heads * shape.context_length * shape.context_length
    )
    chunk_pair = pair * chunk_fraction_numerator // chunk_fraction_denominator
    mask = shape.batch_size * shape.num_sequences * shape.context_length

    # Full-size buffers: public Q/K/V/dO and prepared forward Q/K/V/O.
    full_activation_buffers = 8 * activation_bytes
    # Reused chunk buffers: public O and dQ/dK/dV.
    chunk_activation_buffers = 4 * chunk_activation_bytes
    # Backward's cached FP32 dQ accumulator is twice a BF16 activation buffer.
    backward_workspace = 2 * chunk_activation_bytes + 4 * chunk_row
    return (
        full_activation_buffers
        + chunk_activation_buffers
        + backward_workspace
        + 4 * pair  # pair-bias input
        + 6 * chunk_pair  # FP32 accumulator and BF16 dPairBias
        + 4 * mask  # residual mask
        + 4 * row  # forward LSE
        + 4 * chunk_row  # backward delta
    )


def _combined_pool_plan(
    shape: EvoAttentionShape,
    device: torch.device,
) -> tuple[int, int]:
    """Choose a memory-safe backward chunk and common address-pool size."""
    with torch.cuda.device(device):
        free_bytes, _ = torch.cuda.mem_get_info()
    usable_bytes = min(
        COMBINED_POOL_BUDGET_BYTES,
        max(1, free_bytes - GPU_MEMORY_RESERVE_BYTES),
    )
    backward_batch_chunk = next(
        (
            chunk
            for chunk in range(shape.batch_size, 0, -1)
            if _cute_combined_slot_bytes(shape, chunk) <= usable_bytes
        ),
        None,
    )
    if backward_batch_chunk is None:
        required_gib = _cute_combined_slot_bytes(shape, 1) / 1024**3
        available_gib = usable_bytes / 1024**3
        raise RuntimeError(
            "insufficient free GPU memory for the combined benchmark: "
            f"requires about {required_gib:.1f} GiB, "
            f"has {available_gib:.1f} GiB after reserve"
        )

    cute_bytes = _cute_combined_slot_bytes(shape, backward_batch_chunk)
    triton_bytes = _backward_slot_bytes(shape, triton_reference=True)
    pool_slots = max(
        1,
        min(MAX_POOL_SLOTS, usable_bytes // max(cute_bytes, triton_bytes)),
    )
    return backward_batch_chunk, pool_slots


def _slice_backward_buffers(
    buffers: BackwardBuffers,
    start: int,
    stop: int,
) -> BackwardBuffers:
    inputs = buffers.inputs
    forward = buffers.forward
    return BackwardBuffers(
        inputs=EvoAttentionInputs(
            inputs.q[start:stop],
            inputs.k[start:stop],
            inputs.v[start:stop],
            inputs.pair_bias[start:stop],
            inputs.residual_mask[start:stop],
        ),
        forward=EvoAttentionOutputs(
            forward.output[start:stop],
            forward.logsumexp[start:stop],
        ),
        output_gradient=buffers.output_gradient[start:stop],
        delta=buffers.delta[: stop - start],
        query_gradient=buffers.query_gradient[: stop - start],
        key_gradient=buffers.key_gradient[: stop - start],
        value_gradient=buffers.value_gradient[: stop - start],
        pair_bias_accumulator=buffers.pair_bias_accumulator[start:stop],
        pair_bias_gradient=buffers.pair_bias_gradient[start:stop],
    )


def _launch_triton_backward_chunk(reference, buffers: BackwardBuffers) -> None:
    q = buffers.inputs.q
    k = buffers.inputs.k
    v = buffers.inputs.v
    pair_bias = buffers.inputs.pair_bias
    residual_mask = buffers.inputs.residual_mask
    output = buffers.forward.output
    output_gradient = buffers.output_gradient
    logsumexp = buffers.forward.logsumexp
    delta = buffers.delta
    dq = buffers.query_gradient
    dk = buffers.key_gradient
    dv = buffers.value_gradient
    dpb = buffers.pair_bias_accumulator
    batch, sequence_count, heads, context_length, dimension = q.shape
    problem_count = batch * sequence_count * heads

    reference._attn_bwd_preprocess[(triton.cdiv(context_length, 32), problem_count, 1)](
        O=output,
        dO=output_gradient,
        D=delta,
        SEQ_LEN=context_length,
        DIM=dimension,
        BLOCK_DIM=64,
        BLOCK_SIZE_Q=32,
        num_stages=2,
        num_warps=4,
    )
    common = dict(
        Q=q,
        K=k,
        V=v,
        res_mask=residual_mask,
        pair_bias=pair_bias,
        softmax_scale=SOFTMAX_SCALE,
        dO=output_gradient,
        dQ=dq,
        dK=dk,
        dV=dv,
        M=logsumexp,
        D=delta,
        stride_batch=q.stride(0),
        stride_msa=q.stride(1),
        stride_head=q.stride(2),
        stride_seq=q.stride(3),
        stride_pair_bias_batch=pair_bias.stride(0),
        stride_pair_bias_head=pair_bias.stride(1),
        stride_pair_bias_seq1=pair_bias.stride(2),
        stride_pair_bias_seq2=pair_bias.stride(3),
        stride_mask_batch=residual_mask.stride(0),
        stride_mask_msa=residual_mask.stride(1),
        stride_mask_seq=residual_mask.stride(2),
        HEAD=heads,
        N_SEQ=sequence_count,
        SEQ_LEN=context_length,
        BLOCK_DIM=64,
        DIM=dimension,
    )
    reference._attn_bwd_dk_dv[(triton.cdiv(context_length, 16), problem_count, 1)](
        **common,
        BLOCK_SIZE_Q=64,
        BLOCK_SIZE_KV=16,
        num_stages=3,
        num_warps=4,
    )
    reference._attn_bwd_dq[(triton.cdiv(context_length, 16), problem_count, 1)](
        **common,
        d_pair_bias=dpb,
        stride_d_pair_bias_batch=dpb.stride(0),
        stride_d_pair_bias_head=dpb.stride(1),
        stride_d_pair_bias_seq1=dpb.stride(2),
        stride_d_pair_bias_seq2=dpb.stride(3),
        BLOCK_SIZE_Q=16,
        BLOCK_SIZE_KV=64,
        num_stages=3,
        num_warps=4,
    )


def _launch_triton_public_backward(
    reference,
    slot: TritonBackwardSlot,
) -> None:
    buffers = slot.buffers
    buffers.pair_bias_accumulator.zero_()
    batch, sequence_count, heads, _, _ = buffers.inputs.q.shape
    batch_chunk = min(
        slot.gradient_batch_capacity,
        max(1, MAX_GRID_Y // (sequence_count * heads)),
    )
    for start in range(0, batch, batch_chunk):
        chunk = _slice_backward_buffers(
            buffers,
            start,
            min(batch, start + batch_chunk),
        )
        _launch_triton_backward_chunk(reference, chunk)
        slot.public_query_gradient[start : start + chunk.inputs.q.shape[0]].copy_(
            chunk.query_gradient.transpose(2, 3)
        )
        slot.public_key_gradient[start : start + chunk.inputs.q.shape[0]].copy_(
            chunk.key_gradient.transpose(2, 3)
        )
        slot.public_value_gradient[start : start + chunk.inputs.q.shape[0]].copy_(
            chunk.value_gradient.transpose(2, 3)
        )
    buffers.pair_bias_gradient.copy_(buffers.pair_bias_accumulator)


def _launch_cute_backward(backward: BackwardKernel, slot: CuteBackwardSlot) -> None:
    backward(
        slot.q,
        slot.k,
        slot.v,
        slot.output,
        slot.output_gradient,
        slot.pair_bias,
        slot.residual_mask,
        slot.logsumexp,
        slot.delta,
        slot.query_gradient,
        slot.key_gradient,
        slot.value_gradient,
        slot.pair_bias_accumulator,
        slot.pair_bias_gradient,
        stream=torch.cuda.current_stream(slot.q.device),
    )


def _launch_triton_combined(reference, slot: TritonBackwardSlot) -> None:
    """Launch a dependent reference forward and complete public backward."""
    launch_reference(slot.buffers.inputs, slot.buffers.forward)
    _launch_triton_public_backward(reference, slot)


def _launch_cute_combined(
    forward: ForwardKernel,
    backward: BackwardKernel,
    slot: CuteCombinedSlot,
) -> None:
    """Launch forward, bridge its output layout, then launch full backward."""
    launch_cute(forward, slot.forward_inputs, slot.forward_outputs)
    stream = torch.cuda.current_stream(slot.forward_outputs.output.device)
    for chunk in slot.backward_chunks:
        # Forward writes [B,S,H,N,D], while backward consumes [B,S,N,H,D].
        # This is the only layout operation required between the two CuTe APIs.
        chunk.output.copy_(chunk.forward_output.transpose(2, 3))
        backward(
            chunk.q,
            chunk.k,
            chunk.v,
            chunk.output,
            chunk.output_gradient,
            chunk.pair_bias,
            chunk.residual_mask,
            chunk.logsumexp,
            chunk.delta,
            chunk.query_gradient,
            chunk.key_gradient,
            chunk.value_gradient,
            chunk.pair_bias_accumulator,
            chunk.pair_bias_gradient,
            stream=stream,
        )


def _benchmark_implementation(
    make_slot,
    launch,
    shape: EvoAttentionShape,
    *,
    device: torch.device,
    seed: int,
    target_ms: float,
    warmup: int,
    repeats: int,
    fixed_iters: int | None,
    pool_slots: int,
    capture_graph: bool = False,
) -> TimingStats:
    slots = tuple(make_slot(shape, device, seed + index) for index in range(pool_slots))

    def eager_step(iteration: int) -> None:
        launch(slots[iteration % len(slots)])

    # Compile every code path and populate pointer-keyed launcher caches before
    # timing or graph capture. This also keeps capture-time allocations illegal.
    for slot in slots:
        launch(slot)
    torch.cuda.synchronize(device)

    if capture_graph:
        graphs = []
        for slot in slots:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(slot)
            graphs.append(graph)
        torch.cuda.synchronize(device)

        def step(iteration: int) -> None:
            graphs[iteration % len(graphs)].replay()

    else:
        step = eager_step

    iterations = fixed_iters or _calibrated_iterations((step,), target_ms)
    result = time_cuda_events(
        step,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    del step
    if capture_graph:
        del graphs
    del eager_step, slots
    gc.collect()
    torch.cuda.empty_cache()
    return result


def benchmark_backward_shape(
    backward: BackwardKernel,
    shape: EvoAttentionShape,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    target_ms: float,
    fixed_iters: int | None,
    seed: int,
) -> BackwardBenchmarkResult:
    clear_cache = getattr(backward, "clear_workspace_cache", None)
    if clear_cache is not None:
        clear_cache()
    reference = reference_module()
    # Rotate both implementations through the same number of addresses.  The
    # smaller memory-safe capacity wins so neither side receives a warmer cache.
    pool_slots = min(
        _backward_pool_size(shape, triton_reference=True),
        _backward_pool_size(shape, triton_reference=False),
    )
    common = dict(
        shape=shape,
        device=device,
        seed=seed,
        target_ms=target_ms,
        warmup=warmup,
        repeats=repeats,
        pool_slots=pool_slots,
    )

    def benchmark_triton(iterations: int | None = None) -> TimingStats:
        return _benchmark_implementation(
            _make_triton_backward_slot,
            lambda slot: _launch_triton_public_backward(reference, slot),
            fixed_iters=fixed_iters if fixed_iters is not None else iterations,
            **common,
        )

    def benchmark_cute(iterations: int | None = None) -> TimingStats:
        return _benchmark_implementation(
            _make_cute_backward_slot,
            lambda slot: _launch_cute_backward(backward, slot),
            fixed_iters=fixed_iters if fixed_iters is not None else iterations,
            **common,
        )

    if (shape.batch_size + shape.num_heads + shape.context_length // 128) % 2:
        cute_timing = benchmark_cute()
        if clear_cache is not None:
            clear_cache()
        triton_timing = benchmark_triton(cute_timing.iterations)
    else:
        triton_timing = benchmark_triton()
        cute_timing = benchmark_cute(triton_timing.iterations)
        if clear_cache is not None:
            clear_cache()
    return BackwardBenchmarkResult(
        batch_size=shape.batch_size,
        num_heads=shape.num_heads,
        context_length=shape.context_length,
        pool_slots=pool_slots,
        triton=triton_timing,
        cute=cute_timing,
    )


def benchmark_combined_shape(
    forward: ForwardKernel,
    backward: BackwardKernel,
    shape: EvoAttentionShape,
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    target_ms: float,
    fixed_iters: int | None,
    seed: int,
    capture_graph: bool,
) -> CombinedBenchmarkResult:
    """Benchmark one dependent forward plus complete backward operation."""
    clear_cache = getattr(backward, "clear_workspace_cache", None)
    if clear_cache is not None:
        clear_cache()
    reference = reference_module()
    backward_batch_chunk, pool_slots = _combined_pool_plan(shape, device)
    common = dict(
        shape=shape,
        device=device,
        seed=seed,
        target_ms=target_ms,
        warmup=warmup,
        repeats=repeats,
        pool_slots=pool_slots,
    )

    def benchmark_triton(iterations: int | None = None) -> TimingStats:
        return _benchmark_implementation(
            lambda current_shape, current_device, current_seed: (
                _make_triton_backward_slot(
                    current_shape,
                    current_device,
                    current_seed,
                    gradient_batch_capacity=backward_batch_chunk,
                )
            ),
            lambda slot: _launch_triton_combined(reference, slot),
            fixed_iters=fixed_iters if fixed_iters is not None else iterations,
            capture_graph=capture_graph,
            **common,
        )

    def benchmark_cute(iterations: int | None = None) -> TimingStats:
        return _benchmark_implementation(
            lambda current_shape, current_device, current_seed: (
                _make_cute_combined_slot(
                    current_shape,
                    current_device,
                    current_seed,
                    backward_batch_chunk,
                )
            ),
            lambda slot: _launch_cute_combined(forward, backward, slot),
            fixed_iters=fixed_iters if fixed_iters is not None else iterations,
            capture_graph=capture_graph,
            **common,
        )

    # Alternate implementation order by shape to avoid consistently assigning
    # the cooler GPU clock state to one side. Both sides use the same address
    # count, iteration count, and backward batch partition.
    if (shape.batch_size + shape.num_heads + shape.context_length // 128) % 2:
        cute_timing = benchmark_cute()
        if clear_cache is not None:
            clear_cache()
        triton_timing = benchmark_triton(cute_timing.iterations)
    else:
        triton_timing = benchmark_triton()
        cute_timing = benchmark_cute(triton_timing.iterations)
        if clear_cache is not None:
            clear_cache()

    return CombinedBenchmarkResult(
        batch_size=shape.batch_size,
        num_heads=shape.num_heads,
        context_length=shape.context_length,
        pool_slots=pool_slots,
        backward_batch_chunk=backward_batch_chunk,
        triton=triton_timing,
        cute=cute_timing,
    )


def _print_forward_header() -> None:
    print(
        f"{'B':>2} {'H':>3} {'N=S':>5} {'slots':>5}  "
        f"{'Triton ms':>11} {'CuTe ms':>11} {'speedup':>9}  "
        f"{'Triton TF/s':>12} {'CuTe TF/s':>10} {'iters':>6} {'pass':>5}"
    )
    print("-" * 106)


def _print_forward_result(result: BenchmarkResult, required_speedup: float) -> None:
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


def _print_combined_header() -> None:
    print(
        f"{'B':>2} {'H':>3} {'N=S':>5} {'slots':>5} {'bwd chunk':>9}  "
        f"{'Triton ms':>11} {'CuTe ms':>11} {'speedup':>9}  "
        f"{'Triton TF/s':>12} {'CuTe TF/s':>10} {'iters':>6} {'pass':>5}"
    )
    print("-" * 117)


def _print_combined_result(
    result: CombinedBenchmarkResult,
    required_speedup: float,
) -> None:
    passed = result.speedup >= required_speedup
    print(
        f"{result.batch_size:2d} {result.num_heads:3d} "
        f"{result.context_length:5d} {result.pool_slots:5d} "
        f"{result.backward_batch_chunk:9d}  "
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


def _write_backward_json(
    path: Path,
    results: Sequence[BackwardBenchmarkResult],
    args: argparse.Namespace,
    compile_count: int,
) -> None:
    serialized_results = []
    for result in results:
        serialized = asdict(result)
        serialized.update(
            speedup=result.speedup,
            cute_tflops=result.cute_tflops,
        )
        serialized_results.append(serialized)
    payload = {
        "direction": "backward",
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "required_speedup": args.required_speedup,
        "compile_count": compile_count,
        "timing": (
            "CUDA events, prepared model layouts, all backward launches and "
            "output conversions, median of repeat averages, matched rotating "
            "pool and iteration count"
        ),
        "results": serialized_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_combined_json(
    path: Path,
    results: Sequence[CombinedBenchmarkResult],
    args: argparse.Namespace,
    *,
    forward_compile_count: int,
    backward_compile_count: int,
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
    execution = "eager launches" if args.eager_combined else "CUDA-graph replay"
    payload = {
        "direction": "combined",
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "required_speedup": args.required_speedup,
        "compile_count": {
            "forward": forward_compile_count,
            "backward": backward_compile_count,
        },
        "flops": (
            "14*B*S*H*N^2*D algorithmic FLOPs: two forward and five "
            "backward matrix products, counting each FMA as two FLOPs"
        ),
        "timing": (
            "CUDA-event timing of dependent forward plus complete backward; "
            f"execution={execution}; "
            "prepared input layouts, allocation, compilation, graph capture, and "
            "warmup excluded; required CuTe "
            "forward-output layout bridge and Triton gradient output conversions "
            "included; matched rotating address pool, iteration count, and "
            "backward batch chunk"
        ),
        "results": serialized_results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_backward_benchmark(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    from evoattention_cute_bwd import get_evoattention_backward

    backward = get_evoattention_backward()
    results: list[BackwardBenchmarkResult] = []
    print(
        f"{'B':>2} {'H':>3} {'N=S':>5} {'slots':>5}  "
        f"{'Triton ms':>11} {'CuTe ms':>11} {'speedup':>9}  "
        f"{'CuTe TF/s':>10} {'iters':>6} {'pass':>5}"
    )
    print("-" * 104)
    for shape in select_shapes(
        quick=args.quick,
        context_length=args.n,
        batch_size=args.batch,
        num_heads=args.heads,
    ):
        result = benchmark_backward_shape(
            backward,
            shape,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            target_ms=args.target_ms,
            fixed_iters=args.iters,
            seed=args.seed,
        )
        results.append(result)
        passed = result.speedup >= args.required_speedup
        print(
            f"{result.batch_size:2d} {result.num_heads:3d} "
            f"{result.context_length:5d} {result.pool_slots:5d}  "
            f"{result.triton.median_ms:11.4f} "
            f"{result.cute.median_ms:11.4f} {result.speedup:8.3f}x  "
            f"{result.cute_tflops:10.2f} {result.cute.iterations:6d} "
            f"{'yes' if passed else 'NO':>5}",
            flush=True,
        )

    failures = [result for result in results if result.speedup < args.required_speedup]
    print(
        f"\nsummary: {len(results) - len(failures)}/{len(results)} shapes meet "
        f"{args.required_speedup:.3f}x; CuTe compile_count={backward.compile_count}"
    )
    if backward.compile_count != 1:
        raise AssertionError(
            "backward must use one runtime-dynamic compiled artifact, got "
            f"{backward.compile_count}"
        )
    if args.json is not None:
        _write_backward_json(args.json, results, args, backward.compile_count)
        print(f"wrote {args.json}")
    if failures and not args.no_enforce:
        failed = ", ".join(
            f"B{result.batch_size}/H{result.num_heads}/N{result.context_length}"
            for result in failures
        )
        raise SystemExit(f"speed target missed for: {failed}")


def _run_combined_benchmark(
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    from evoattention_cute_bwd import get_evoattention_backward

    forward = load_cute_forward()
    backward = get_evoattention_backward()
    results: list[CombinedBenchmarkResult] = []
    _print_combined_header()
    for shape in select_shapes(
        quick=args.quick,
        context_length=args.n,
        batch_size=args.batch,
        num_heads=args.heads,
    ):
        result = benchmark_combined_shape(
            forward,
            backward,
            shape,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            target_ms=args.target_ms,
            fixed_iters=args.iters,
            seed=args.seed,
            capture_graph=not args.eager_combined,
        )
        results.append(result)
        _print_combined_result(result, args.required_speedup)

    assert_single_compilation(forward)
    if backward.compile_count != 1:
        raise AssertionError(
            "backward must use one runtime-dynamic compiled artifact, got "
            f"{backward.compile_count}"
        )
    failures = [result for result in results if result.speedup < args.required_speedup]
    print(
        f"\nsummary: {len(results) - len(failures)}/{len(results)} shapes meet "
        f"{args.required_speedup:.3f}x; CuTe compile_count="
        f"forward:{forward.compile_count}, backward:{backward.compile_count}"
    )
    if args.json is not None:
        _write_combined_json(
            args.json,
            results,
            args,
            forward_compile_count=forward.compile_count,
            backward_compile_count=backward.compile_count,
        )
        print(f"wrote {args.json}")
    if failures and not args.no_enforce:
        failed = ", ".join(
            f"B{result.batch_size}/H{result.num_heads}/N{result.context_length}"
            for result in failures
        )
        raise SystemExit(f"speed target missed for: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=("combined", "forward", "backward"),
        default="combined",
    )
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
    parser.add_argument(
        "--eager-combined",
        action="store_true",
        help=(
            "launch each operation eagerly in combined mode instead of replaying "
            "a pre-captured CUDA graph; this includes host enqueue gaps"
        ),
    )
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
    if args.direction == "combined":
        print(
            "timing: dependent forward + complete backward, prepared inputs, "
            f"{'eager launches' if args.eager_combined else 'CUDA-graph replay'}, "
            "CUDA events, compile/allocation/warmup excluded"
        )
    else:
        print(
            "timing: prepared-layout kernel only, CUDA events, "
            "compile/warmup excluded"
        )
    print(f"required speedup: {args.required_speedup:.3f}x on every selected shape\n")

    if args.direction == "combined":
        _run_combined_benchmark(args, device)
        return
    if args.direction == "backward":
        _run_backward_benchmark(args, device)
        return

    forward = load_cute_forward()
    results: list[BenchmarkResult] = []
    _print_forward_header()
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
        _print_forward_result(result, args.required_speedup)

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
