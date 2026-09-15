"""Autograd-compatible CuTe-DSL EvoAttention for MegaFold training.

The forward interface matches MegaFold's fused-attention call boundary:

* query/key/value/output: ``[B, S, N, H, D]`` BF16
* pair bias: ``[B, 1, H, N, N]`` BF16
* residual mask: ``[B, S, 1, 1, N]`` BF16 with ``1`` for valid keys
* log-sum-exp: ``[B, S, H, N]`` FP32

``D`` is inspected on every call and dispatched to independently tuned,
runtime-dynamic D=32 or D=64 forward and backward specializations. Importing
this module does not compile a kernel; each specialization compiles on first
use. The exported ``TritonEvoformer`` alias matches MegaFold's existing call
site and can replace its Triton autograd function directly.
"""

from __future__ import annotations

import importlib
import threading

import torch


SUPPORTED_HEAD_DIMENSIONS = (32, 64)


def _specialization_module(head_dimension: int) -> str:
    module = f"evoattention_cute_d{head_dimension}"
    return f"{__package__}.{module}" if __package__ else module


class EvoAttentionForward:
    """Dispatch model-facing forward calls by ``q.shape[-1]``."""

    def __init__(self) -> None:
        self._kernels: dict[int, object] = {}
        self._lock = threading.RLock()

    @property
    def compile_count(self) -> int:
        """Return the total number of compiled head-dimension artifacts."""
        with self._lock:
            return sum(
                int(getattr(kernel, "compile_count"))
                for kernel in self._kernels.values()
            )

    @property
    def compile_counts(self) -> dict[int, int]:
        """Return compilation counts separately for D=32 and D=64."""
        with self._lock:
            return {
                dimension: int(
                    getattr(self._kernels.get(dimension), "compile_count", 0)
                )
                for dimension in SUPPORTED_HEAD_DIMENSIONS
            }

    def _kernel(self, head_dimension: int):
        if head_dimension not in SUPPORTED_HEAD_DIMENSIONS:
            raise ValueError(
                "unsupported head dimension "
                f"{head_dimension}; expected one of {SUPPORTED_HEAD_DIMENSIONS}"
            )
        with self._lock:
            kernel = self._kernels.get(head_dimension)
            if kernel is None:
                module = importlib.import_module(
                    _specialization_module(head_dimension)
                )
                kernel = module.get_evoattention_forward()
                self._kernels[head_dimension] = kernel
            return kernel

    def forward_into(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual_mask: torch.Tensor,
        pair_bias: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Write output and FP32 log-sum-exp into preallocated tensors."""
        if q.ndim != 5:
            raise ValueError(
                "q must have model-facing shape [B, S, N, H, D], got "
                f"{tuple(q.shape)}"
            )
        self._kernel(q.shape[-1])(
            q,
            k,
            v,
            pair_bias,
            residual_mask,
            out,
            lse,
            stream=stream,
        )

    def forward_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual_mask: torch.Tensor,
        pair_bias: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return model-layout output together with backward-compatible LSE."""
        if q.ndim != 5:
            raise ValueError(
                "q must have model-facing shape [B, S, N, H, D], got "
                f"{tuple(q.shape)}"
            )
        batch, sequence_count, n_ctx, heads, _ = q.shape
        out = torch.empty_like(q)
        lse = torch.empty(
            (batch, sequence_count, heads, n_ctx),
            dtype=torch.float32,
            device=q.device,
        )
        self.forward_into(
            q,
            k,
            v,
            residual_mask,
            pair_bias,
            out,
            lse,
            stream=stream,
        )
        return out, lse

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual_mask: torch.Tensor,
        pair_bias: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run the forward pass with the same visible contract as MegaFold."""
        out, _ = self.forward_with_lse(
            q,
            k,
            v,
            residual_mask,
            pair_bias,
            stream=stream,
        )
        return out


_FORWARD_KERNEL = EvoAttentionForward()


def _backward_dispatcher():
    module_name = "evoattention_cute_bwd"
    qualified_name = f"{__package__}.{module_name}" if __package__ else module_name
    module = importlib.import_module(qualified_name)
    return module.get_evoattention_backward()


class EvoformerAttention(torch.autograd.Function):
    """MegaFold-compatible autograd boundary backed by CuTe-DSL kernels."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual_mask: torch.Tensor,
        pair_bias: torch.Tensor,
    ) -> torch.Tensor:
        # MegaFold obtains these tensors through an einops axis permutation.
        # In BF16 that permutation can remain a non-compact view because the
        # subsequent dtype cast is a no-op. Normalize only the tensors whose
        # tuned forward/backward descriptors require compact storage.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        residual_mask = residual_mask.contiguous()
        output, logsumexp = _FORWARD_KERNEL.forward_with_lse(
            q, k, v, residual_mask, pair_bias
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            output,
            residual_mask,
            pair_bias,
            logsumexp,
        )
        return output

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        q, k, v, output, residual_mask, pair_bias, logsumexp = ctx.saved_tensors
        output_gradient = output_gradient.contiguous()
        query_gradient = torch.empty_like(q)
        key_gradient = torch.empty_like(k)
        value_gradient = torch.empty_like(v)
        delta = torch.empty_like(logsumexp)
        batch, _, context_length, heads, _ = q.shape
        pair_shape = (batch, heads, context_length, context_length)
        pair_bias_accumulator = torch.empty(
            pair_shape, dtype=torch.float32, device=q.device
        )
        pair_bias_gradient = torch.empty(
            pair_shape, dtype=torch.bfloat16, device=q.device
        )
        _backward_dispatcher()(
            q,
            k,
            v,
            output,
            output_gradient,
            pair_bias,
            residual_mask,
            logsumexp,
            delta,
            query_gradient,
            key_gradient,
            value_gradient,
            pair_bias_accumulator,
            pair_bias_gradient,
        )
        return (
            query_gradient,
            key_gradient,
            value_gradient,
            None,
            pair_bias_gradient.unsqueeze(1),
        )


def get_evoattention_forward() -> EvoAttentionForward:
    """Return the process-wide D=32/D=64 forward dispatcher."""
    return _FORWARD_KERNEL


def evoattention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    residual_mask: torch.Tensor,
    pair_bias: torch.Tensor,
) -> torch.Tensor:
    """Run the differentiable, drop-in MegaFold EvoAttention operation."""
    return EvoformerAttention.apply(q, k, v, residual_mask, pair_bias)


TritonEvoformer = EvoformerAttention.apply


__all__ = [
    "EvoAttentionForward",
    "EvoformerAttention",
    "SUPPORTED_HEAD_DIMENSIONS",
    "TritonEvoformer",
    "evoattention",
    "get_evoattention_forward",
]
