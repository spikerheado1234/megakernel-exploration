"""Correctness tests for model-facing CuTe-DSL EvoAttention training.

Both implementations receive MegaFold's public layout and dtype contract:

    q, k, v, out:  contiguous [B, S, N, H, D],     bfloat16
    pair_bias:     contiguous [B, 1, H, N, N],      bfloat16
    res_mask:      contiguous [B, S, 1, 1, N],      bfloat16, values 0 or 1
    lse:           contiguous [B, S, H, N],          float32

The authoritative Triton kernel still requires internal-layout Q/K/V and an
additive mask.  ``launch_reference`` performs those conversions, whereas CuTe
uses rank-five TMA views and converts the Boolean-valued mask while loading it.

Here S == N for every target workload.  ``evoattention_cute`` must export
``get_evoattention_forward()``, which returns a cached launcher whose
allocation-free entry point has this signature::

    forward.forward_into(q, k, v, res_mask, pair_bias, out, lse, *, stream=None)

The returned object must also expose an integer ``compile_count`` property.
All B/H/N modes are runtime dynamic, so it must remain one after all 32 shapes
have run. D is selected once per test run from the static D=32 and D=64 kernel
specializations.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

import torch
import triton
import triton.language as tl


HERE = Path(__file__).resolve().parent
REFERENCE_PATH = Path(
    os.environ.get(
        "EVOATTENTION_REFERENCE",
        "/home/ahangupta/MegaFold/megafold/model/FusedEvoAttention/evoattention.py",
    )
)

SUPPORTED_HEAD_DIMENSIONS = (32, 64)
HEAD_DIMENSION = 32
SEQUENCE_LENGTHS = (128, 256, 384, 512, 640, 768, 896, 1024)
BATCH_SIZES = (1, 4)
HEAD_COUNTS = (4, 16)

OUTPUT_ATOL = 2.0e-2
OUTPUT_RTOL = 2.0e-2
LSE_ATOL = 2.0e-2
LSE_RTOL = 2.0e-2
GRADIENT_ATOL = 2.0e-2
GRADIENT_RTOL = 2.0e-2

# Bounds peak comparison memory even for B=4, H=16, N=S=1024.  In particular,
# do not cast an entire 8 GiB BF16 output to FP32 at once.
COMPARE_CHUNK_ELEMENTS = 16 * 1024 * 1024


@dataclass(frozen=True, order=True)
class EvoAttentionShape:
    batch_size: int
    num_heads: int
    context_length: int
    head_dimension: int = HEAD_DIMENSION

    @property
    def num_sequences(self) -> int:
        return self.context_length

    @property
    def tag(self) -> str:
        return (
            f"B={self.batch_size}, S=N={self.context_length}, "
            f"H={self.num_heads}, D={self.head_dimension}"
        )


@dataclass(frozen=True)
class EvoAttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    pair_bias: torch.Tensor
    residual_mask: torch.Tensor


@dataclass(frozen=True)
class EvoAttentionOutputs:
    output: torch.Tensor
    logsumexp: torch.Tensor


@dataclass(frozen=True)
class ReferenceForwardWorkspace:
    """Preallocated internal-layout buffers for the MegaFold Triton baseline."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    additive_mask: torch.Tensor
    output: torch.Tensor


class ForwardKernel(Protocol):
    """Model-facing forward interface exposed by ``evoattention_cute``."""

    @property
    def compile_count(self) -> int: ...

    def forward_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual_mask: torch.Tensor,
        pair_bias: torch.Tensor,
        output: torch.Tensor,
        logsumexp: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None: ...


def target_shapes(
    head_dimension: int = HEAD_DIMENSION,
) -> tuple[EvoAttentionShape, ...]:
    """The exact 32-shape acceptance matrix."""
    if head_dimension not in SUPPORTED_HEAD_DIMENSIONS:
        raise ValueError(
            f"head dimension must be one of {SUPPORTED_HEAD_DIMENSIONS}"
        )
    return tuple(
        EvoAttentionShape(
            batch_size=batch,
            num_heads=heads,
            context_length=n,
            head_dimension=head_dimension,
        )
        for batch in BATCH_SIZES
        for heads in HEAD_COUNTS
        for n in SEQUENCE_LENGTHS
    )


def _load_reference_module() -> ModuleType:
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            f"Triton EvoAttention reference not found at {REFERENCE_PATH}. "
            "Set EVOATTENTION_REFERENCE to override it."
        )
    spec = importlib.util.spec_from_file_location(
        "_megafold_evoattention_reference", REFERENCE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Triton reference from {REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REFERENCE_MODULE: ModuleType | None = None


def reference_module() -> ModuleType:
    global _REFERENCE_MODULE
    if _REFERENCE_MODULE is None:
        _REFERENCE_MODULE = _load_reference_module()
    return _REFERENCE_MODULE


def load_cute_forward() -> ForwardKernel:
    """Load the stable, cached TVM-FFI/CuTe launcher interface."""
    try:
        from evoattention_cute import get_evoattention_forward
    except ImportError as exc:
        raise ImportError(
            f"expected {HERE / 'evoattention_cute.py'} to export "
            "get_evoattention_forward()"
        ) from exc

    forward = get_evoattention_forward()
    if not callable(forward) or not callable(getattr(forward, "forward_into", None)):
        raise TypeError(
            "get_evoattention_forward() must return a callable with forward_into()"
        )
    if not hasattr(forward, "compile_count"):
        raise TypeError(
            "the CuTe forward callable must expose compile_count so the test can "
            "verify that B/H/N do not trigger new specializations"
        )
    return forward


def make_inputs(
    shape: EvoAttentionShape,
    device: torch.device,
    seed: int = 0,
) -> EvoAttentionInputs:
    """Create deterministic inputs in MegaFold's public layout."""
    if shape.context_length % 128:
        raise ValueError("N must be divisible by the M=128 CTA tile")

    generator = torch.Generator(device=device)
    generator.manual_seed(
        seed
        ^ (shape.batch_size * 0x1F1F)
        ^ (shape.num_heads * 0x101)
        ^ shape.context_length
        ^ (shape.head_dimension * 0x10001)
    )
    qkv_shape = (
        shape.batch_size,
        shape.num_sequences,
        shape.context_length,
        shape.num_heads,
        shape.head_dimension,
    )
    q = torch.randn(qkv_shape, dtype=torch.bfloat16, device=device, generator=generator)
    k = torch.randn(qkv_shape, dtype=torch.bfloat16, device=device, generator=generator)
    v = torch.randn(qkv_shape, dtype=torch.bfloat16, device=device, generator=generator)
    pair_bias = torch.randn(
        (
            shape.batch_size,
            1,
            shape.num_heads,
            shape.context_length,
            shape.context_length,
        ),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    # About one eighth of keys are invalid. Setting key zero valid avoids a
    # degenerate all-masked row while exercising MegaFold's Boolean-mask contract.
    residual_mask = torch.empty(
        (shape.batch_size, shape.num_sequences, 1, 1, shape.context_length),
        dtype=torch.bfloat16,
        device=device,
    )
    residual_mask.bernoulli_(0.875, generator=generator)
    residual_mask[..., 0] = 1.0

    tensors = (q, k, v, pair_bias, residual_mask)
    if not all(t.is_contiguous() for t in tensors):
        raise AssertionError("model-facing inputs must be contiguous")
    return EvoAttentionInputs(
        q=q,
        k=k,
        v=v,
        pair_bias=pair_bias,
        residual_mask=residual_mask,
    )


def make_outputs(inputs: EvoAttentionInputs) -> EvoAttentionOutputs:
    q = inputs.q
    batch, sequence_count, n_ctx, heads, _ = q.shape
    return EvoAttentionOutputs(
        output=torch.empty_like(q),
        logsumexp=torch.empty(
            (batch, sequence_count, heads, n_ctx),
            dtype=torch.float32,
            device=q.device,
        ),
    )


def make_reference_workspace(inputs: EvoAttentionInputs) -> ReferenceForwardWorkspace:
    """Allocate reusable buffers for MegaFold's internal Triton layout."""
    batch, n_seq, n_ctx, heads, dim = inputs.q.shape
    internal_shape = (batch, n_seq, heads, n_ctx, dim)
    return ReferenceForwardWorkspace(
        q=torch.empty(internal_shape, dtype=torch.bfloat16, device=inputs.q.device),
        k=torch.empty(internal_shape, dtype=torch.bfloat16, device=inputs.q.device),
        v=torch.empty(internal_shape, dtype=torch.bfloat16, device=inputs.q.device),
        additive_mask=torch.empty(
            (batch, n_seq, n_ctx), dtype=torch.float32, device=inputs.q.device
        ),
        output=torch.empty(internal_shape, dtype=torch.bfloat16, device=inputs.q.device),
    )


@triton.jit
def _convert_boolean_mask_kernel(
    source,
    destination,
    element_count,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < element_count
    values = tl.load(source + offsets, mask=active)
    additive = tl.where(values != 0, 0.0, -1.0e9)
    tl.store(destination + offsets, additive, mask=active)


def prepare_reference_inputs(
    inputs: EvoAttentionInputs,
    workspace: ReferenceForwardWorkspace,
) -> None:
    """Materialize the exact input transforms required by MegaFold Triton."""
    workspace.q.copy_(inputs.q.transpose(2, 3))
    workspace.k.copy_(inputs.k.transpose(2, 3))
    workspace.v.copy_(inputs.v.transpose(2, 3))
    element_count = workspace.additive_mask.numel()
    _convert_boolean_mask_kernel[(triton.cdiv(element_count, 256),)](
        inputs.residual_mask,
        workspace.additive_mask,
        element_count,
        BLOCK_SIZE=256,
    )


def launch_reference_prepared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pair_bias: torch.Tensor,
    residual_mask: torch.Tensor,
    output: torch.Tensor,
    logsumexp: torch.Tensor,
) -> None:
    """Launch MegaFold's unmodified Triton kernel on internal-layout buffers."""
    batch, n_seq, heads, n_ctx, dim = q.shape
    if n_seq != n_ctx or dim not in SUPPORTED_HEAD_DIMENSIONS:
        raise ValueError(f"unsupported prepared Q shape: {tuple(q.shape)}")

    # The unmodified reference flattens B*S*H into CUDA grid.y.  CUDA limits
    # grid.y to 65,535, while the largest requested shape has 65,536 problems.
    # Preserve the exact kernel and split that otherwise-unlaunchable case into
    # contiguous per-batch calls.  Benchmark timing includes all of these calls.
    if batch * n_seq * heads > 65_535:
        for batch_idx in range(batch):
            launch_reference_prepared(
                q[batch_idx : batch_idx + 1],
                k[batch_idx : batch_idx + 1],
                v[batch_idx : batch_idx + 1],
                pair_bias[batch_idx : batch_idx + 1],
                residual_mask[batch_idx : batch_idx + 1],
                output[batch_idx : batch_idx + 1],
                logsumexp[batch_idx : batch_idx + 1],
            )
        return

    reference = reference_module()
    grid = (triton.cdiv(n_ctx, 16), batch * n_seq * heads, 1)
    reference._attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        res_mask=residual_mask,
        pair_bias=pair_bias,
        softmax_scale=dim**-0.5,
        M=logsumexp,
        O=output,
        stride_Q_batch=q.stride(0),
        stride_Q_msa=q.stride(1),
        stride_Q_head=q.stride(2),
        stride_Q_seq=q.stride(3),
        stride_Q_dim=q.stride(4),
        stride_K_batch=k.stride(0),
        stride_K_msa=k.stride(1),
        stride_K_head=k.stride(2),
        stride_K_seq=k.stride(3),
        stride_K_dim=k.stride(4),
        stride_V_batch=v.stride(0),
        stride_V_msa=v.stride(1),
        stride_V_head=v.stride(2),
        stride_V_seq=v.stride(3),
        stride_V_dim=v.stride(4),
        stride_O_batch=output.stride(0),
        stride_O_msa=output.stride(1),
        stride_O_head=output.stride(2),
        stride_O_seq=output.stride(3),
        stride_O_dim=output.stride(4),
        stride_pair_bias_batch=pair_bias.stride(0),
        stride_pair_bias_head=pair_bias.stride(1),
        stride_pair_bias_seq1=pair_bias.stride(2),
        stride_pair_bias_seq2=pair_bias.stride(3),
        stride_mask_batch=residual_mask.stride(0),
        stride_mask_msa=residual_mask.stride(1),
        stride_mask_seq=residual_mask.stride(2),
        BATCH_SIZE=batch,
        HEAD=heads,
        N_SEQ=n_seq,
        SEQ_LEN=n_ctx,
        DIM=dim,
        BLOCK_DIM=dim,
        BLOCK_SIZE_Q=16,
        BLOCK_SIZE_KV=64,
        num_stages=3,
        num_warps=4,
    )


def launch_reference(
    inputs: EvoAttentionInputs,
    outputs: EvoAttentionOutputs,
    workspace: ReferenceForwardWorkspace | None = None,
) -> None:
    """Run corrected MegaFold Triton from the model-facing tensor contract."""
    if workspace is None:
        workspace = make_reference_workspace(inputs)
    prepare_reference_inputs(inputs, workspace)
    launch_reference_prepared(
        workspace.q,
        workspace.k,
        workspace.v,
        inputs.pair_bias[:, 0],
        workspace.additive_mask,
        workspace.output,
        outputs.logsumexp,
    )
    outputs.output.copy_(workspace.output.transpose(2, 3))


def launch_cute(
    forward: ForwardKernel,
    inputs: EvoAttentionInputs,
    outputs: EvoAttentionOutputs,
) -> None:
    """Launch only the in-place CuTe kernel through its TVM-FFI wrapper."""
    forward.forward_into(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.residual_mask,
        inputs.pair_bias,
        outputs.output,
        outputs.logsumexp,
        stream=torch.cuda.current_stream(inputs.q.device),
    )


def _chunks(tensor: torch.Tensor, size: int) -> Iterator[torch.Tensor]:
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), size):
        yield flat[start : start + size]


def assert_close_chunked(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
    name: str,
    chunk_elements: int = COMPARE_CHUNK_ELEMENTS,
) -> tuple[float, float]:
    """FP32 allclose with bounded temporary memory and useful diagnostics."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name} shape mismatch: got {actual.shape}, expected {expected.shape}"
        )
    if actual.device != expected.device:
        raise AssertionError(
            f"{name} device mismatch: got {actual.device}, expected {expected.device}"
        )

    max_abs = 0.0
    sum_abs = 0.0
    bad = 0
    total = actual.numel()
    for actual_chunk, expected_chunk in zip(
        _chunks(actual, chunk_elements), _chunks(expected, chunk_elements)
    ):
        actual_f32 = actual_chunk.float()
        expected_f32 = expected_chunk.float()
        diff = (actual_f32 - expected_f32).abs()
        finite = torch.isfinite(actual_f32) & torch.isfinite(expected_f32)
        allowed = atol + rtol * expected_f32.abs()
        bad += int((~finite | (diff > allowed)).sum().item())
        max_abs = max(max_abs, float(diff.nan_to_num(nan=math.inf).max().item()))
        sum_abs += float(diff.nan_to_num(nan=math.inf).sum(dtype=torch.float64).item())

    mean_abs = sum_abs / total
    if bad:
        raise AssertionError(
            f"{name}: {bad}/{total} values outside atol={atol}, rtol={rtol}; "
            f"max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}"
        )
    return max_abs, mean_abs


def run_correctness_case(
    forward: ForwardKernel,
    shape: EvoAttentionShape,
    *,
    device: torch.device,
    seed: int = 0,
) -> dict[str, float]:
    inputs = make_inputs(shape, device, seed=seed)
    reference_outputs = make_outputs(inputs)
    cute_outputs = make_outputs(inputs)

    with torch.inference_mode():
        launch_reference(inputs, reference_outputs)
        launch_cute(forward, inputs, cute_outputs)
    torch.cuda.synchronize(device)

    if cute_outputs.output.dtype != torch.bfloat16:
        raise AssertionError(f"output must be BF16, got {cute_outputs.output.dtype}")
    if cute_outputs.logsumexp.dtype != torch.float32:
        raise AssertionError(f"LSE must be FP32, got {cute_outputs.logsumexp.dtype}")
    out_max, out_mean = assert_close_chunked(
        cute_outputs.output,
        reference_outputs.output,
        atol=OUTPUT_ATOL,
        rtol=OUTPUT_RTOL,
        name=f"output ({shape.tag})",
    )
    lse_max, lse_mean = assert_close_chunked(
        cute_outputs.logsumexp,
        reference_outputs.logsumexp,
        atol=LSE_ATOL,
        rtol=LSE_RTOL,
        name=f"LSE ({shape.tag})",
    )

    del inputs, reference_outputs, cute_outputs
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "out_max_abs": out_max,
        "out_mean_abs": out_mean,
        "lse_max_abs": lse_max,
        "lse_mean_abs": lse_mean,
    }


def run_training_correctness_case(
    shape: EvoAttentionShape,
    *,
    device: torch.device,
    seed: int = 0,
) -> dict[str, float]:
    """Compare the public autograd entry point with corrected MegaFold Triton."""
    from evoattention_cute import TritonEvoformer

    inputs = make_inputs(shape, device, seed=seed)
    upstream_gradient = torch.randn_like(inputs.q)

    cute_q = inputs.q.detach().clone().requires_grad_()
    cute_k = inputs.k.detach().clone().requires_grad_()
    cute_v = inputs.v.detach().clone().requires_grad_()
    cute_pair_bias = inputs.pair_bias.detach().clone().requires_grad_()
    cute_output = TritonEvoformer(
        cute_q,
        cute_k,
        cute_v,
        inputs.residual_mask,
        cute_pair_bias,
    )
    cute_output.backward(upstream_gradient)

    reference_q = inputs.q.detach().clone().requires_grad_()
    reference_k = inputs.k.detach().clone().requires_grad_()
    reference_v = inputs.v.detach().clone().requires_grad_()
    reference_pair_bias = inputs.pair_bias.detach().clone().requires_grad_()
    additive_mask = (inputs.residual_mask.float() - 1.0).mul_(1.0e9)
    reference_output = reference_module().TritonEvoformer(
        reference_q,
        reference_k,
        reference_v,
        additive_mask,
        reference_pair_bias,
    )
    reference_output.backward(upstream_gradient)
    torch.cuda.synchronize(device)

    comparisons = {
        "output": (cute_output, reference_output, OUTPUT_ATOL, OUTPUT_RTOL),
        "dQ": (cute_q.grad, reference_q.grad, GRADIENT_ATOL, GRADIENT_RTOL),
        "dK": (cute_k.grad, reference_k.grad, GRADIENT_ATOL, GRADIENT_RTOL),
        "dV": (cute_v.grad, reference_v.grad, GRADIENT_ATOL, GRADIENT_RTOL),
        "dPairBias": (
            cute_pair_bias.grad,
            reference_pair_bias.grad,
            GRADIENT_ATOL,
            GRADIENT_RTOL,
        ),
    }
    statistics: dict[str, float] = {}
    for name, (actual, expected, atol, rtol) in comparisons.items():
        if actual is None or expected is None:
            raise AssertionError(f"{name} was not produced by autograd")
        max_abs, mean_abs = assert_close_chunked(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            name=f"{name} ({shape.tag})",
        )
        statistics[f"{name}_max_abs"] = max_abs
        statistics[f"{name}_mean_abs"] = mean_abs
    return statistics


def assert_single_compilation(forward: ForwardKernel) -> None:
    count = forward.compile_count
    if callable(count):
        count = count()
    if not isinstance(count, int):
        raise AssertionError(f"compile_count must be an int, got {count!r}")
    if count != 1:
        raise AssertionError(
            f"expected one CuTe compilation for dynamic B/H/N, observed {count}"
        )


def test_forward_all_target_shapes() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EvoAttention correctness tests")
    device = torch.device("cuda", torch.cuda.current_device())
    forward = load_cute_forward()
    for shape in target_shapes():
        run_correctness_case(forward, shape, device=device)
    assert_single_compilation(forward)


def test_training_drop_in_contract() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for EvoAttention correctness tests")
    device = torch.device("cuda", torch.cuda.current_device())
    for head_dimension in SUPPORTED_HEAD_DIMENSIONS:
        run_training_correctness_case(
            EvoAttentionShape(1, 4, 128, head_dimension),
            device=device,
            seed=17,
        )


def select_shapes(
    *,
    quick: bool = False,
    context_length: int | None = None,
    batch_size: int | None = None,
    num_heads: int | None = None,
    head_dimension: int = HEAD_DIMENSION,
) -> tuple[EvoAttentionShape, ...]:
    """Filter the acceptance matrix for command-line smoke runs."""
    shapes: Sequence[EvoAttentionShape] = target_shapes(head_dimension)
    if quick:
        shapes = (
            EvoAttentionShape(1, 4, 128, head_dimension),
            EvoAttentionShape(1, 16, 384, head_dimension),
            EvoAttentionShape(4, 4, 640, head_dimension),
        )
    if context_length is not None:
        shapes = tuple(
            shape for shape in shapes if shape.context_length == context_length
        )
    if batch_size is not None:
        shapes = tuple(shape for shape in shapes if shape.batch_size == batch_size)
    if num_heads is not None:
        shapes = tuple(shape for shape in shapes if shape.num_heads == num_heads)
    return tuple(shapes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run three smoke shapes")
    parser.add_argument("--n", type=int, choices=SEQUENCE_LENGTHS)
    parser.add_argument("--batch", type=int, choices=BATCH_SIZES)
    parser.add_argument("--heads", type=int, choices=HEAD_COUNTS)
    parser.add_argument(
        "--head-dim",
        "--d",
        type=int,
        choices=SUPPORTED_HEAD_DIMENSIONS,
        default=HEAD_DIMENSION,
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    forward = load_cute_forward()
    for shape in select_shapes(
        quick=args.quick,
        context_length=args.n,
        batch_size=args.batch,
        num_heads=args.heads,
        head_dimension=args.head_dim,
    ):
        stats = run_correctness_case(forward, shape, device=device, seed=args.seed)
        print(
            f"PASS {shape.tag}: "
            f"out max/mean={stats['out_max_abs']:.6g}/{stats['out_mean_abs']:.6g}, "
            f"lse max/mean={stats['lse_max_abs']:.6g}/{stats['lse_mean_abs']:.6g}",
            flush=True,
        )
    training_shape = EvoAttentionShape(1, 4, 128, args.head_dim)
    training_stats = run_training_correctness_case(
        training_shape,
        device=device,
        seed=args.seed + 17,
    )
    print(
        f"PASS training {training_shape.tag}: "
        f"dQ max={training_stats['dQ_max_abs']:.6g}, "
        f"dK max={training_stats['dK_max_abs']:.6g}, "
        f"dV max={training_stats['dV_max_abs']:.6g}, "
        f"dPairBias max={training_stats['dPairBias_max_abs']:.6g}",
        flush=True,
    )
    assert_single_compilation(forward)
    print(f"PASS dynamic-shape compilation count: {forward.compile_count}")


__all__ = [
    "BATCH_SIZES",
    "HEAD_COUNTS",
    "HEAD_DIMENSION",
    "SEQUENCE_LENGTHS",
    "SUPPORTED_HEAD_DIMENSIONS",
    "EvoAttentionInputs",
    "EvoAttentionOutputs",
    "EvoAttentionShape",
    "ForwardKernel",
    "ReferenceForwardWorkspace",
    "assert_single_compilation",
    "launch_cute",
    "launch_reference",
    "launch_reference_prepared",
    "load_cute_forward",
    "make_inputs",
    "make_outputs",
    "make_reference_workspace",
    "prepare_reference_inputs",
    "run_correctness_case",
    "run_training_correctness_case",
    "select_shapes",
    "target_shapes",
]


if __name__ == "__main__":
    main()
