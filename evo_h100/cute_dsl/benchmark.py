"""Kernel-only EvoAttention forward/backward benchmark: Triton vs CuTe-DSL.

Compilation, allocation, input layout preparation, and warmup are excluded.
Prepared addresses rotate through a memory-bounded pool to avoid measuring an
unrealistically hot allocation. Forward timing writes BF16 output and FP32 LSE.
Backward timing includes every launch/reduction needed for model-layout BF16
dQ/dK/dV/dPairBias; this includes the reference's three output transposes. The
default run covers all 32 shapes and enforces a 2x CuTe speedup.

Examples:
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
        flops = (
            10
            * self.batch_size
            * self.context_length
            * self.num_heads
            * self.context_length**2
            * HEAD_DIMENSION
        )
        return flops / (self.cute.median_ms * 1.0e9)


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
    gradient_batch_capacity = (
        1
        if TRITON_FULL_BATCH_LIVE_BYTES_PER_ELEMENT * activation_elements
        > TRITON_FULL_BATCH_MEMORY_LIMIT_BYTES
        else shape.batch_size
    )
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


def _backward_activation_elements(shape: EvoAttentionShape) -> int:
    return (
        shape.batch_size
        * shape.num_sequences
        * shape.context_length
        * shape.num_heads
        * HEAD_DIMENSION
    )


def _backward_pool_size(
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
    return max(1, min(MAX_POOL_SLOTS, BACKWARD_POOL_BUDGET_BYTES // slot_bytes))


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


def _benchmark_backward_implementation(
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
) -> TimingStats:
    slots = tuple(make_slot(shape, device, seed + index) for index in range(pool_slots))

    def step(iteration: int) -> None:
        launch(slots[iteration % len(slots)])

    step(0)
    torch.cuda.synchronize(device)
    iterations = fixed_iters or _calibrated_iterations((step,), target_ms)
    result = time_cuda_events(
        step,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    del slots
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
        return _benchmark_backward_implementation(
            _make_triton_backward_slot,
            lambda slot: _launch_triton_public_backward(reference, slot),
            fixed_iters=fixed_iters if fixed_iters is not None else iterations,
            **common,
        )

    def benchmark_cute(iterations: int | None = None) -> TimingStats:
        return _benchmark_backward_implementation(
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direction",
        choices=("forward", "backward"),
        default="forward",
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

    if args.direction == "backward":
        _run_backward_benchmark(args, device)
        return

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
