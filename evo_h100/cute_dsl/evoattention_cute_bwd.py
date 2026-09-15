"""Runtime head-dimension dispatch for CuTe-DSL EvoAttention backward.

The backward interface accepts model-layout BF16 activations and gradients
with shape ``[B, S, N, H, D]``. ``D`` selects the independently tuned D=32 or
D=64 implementation; all launches and reductions for dQ, dK, dV, and
dPairBias remain owned by the selected specialization.
"""

from __future__ import annotations

import importlib
import threading

import torch


SUPPORTED_HEAD_DIMENSIONS = (32, 64)


def _specialization_module(head_dimension: int) -> str:
    module = f"evoattention_cute_bwd_d{head_dimension}"
    return f"{__package__}.{module}" if __package__ else module


class EvoAttentionBackward:
    """Dispatch complete prepared backward calls by ``q.shape[-1]``."""

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
                kernel = module.get_evoattention_backward()
                self._kernels[head_dimension] = kernel
            return kernel

    def clear_workspace_cache(self) -> None:
        """Release workspaces held by any loaded specialization."""
        with self._lock:
            for kernel in self._kernels.values():
                kernel.clear_workspace_cache()

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
        dq: torch.Tensor,
        dk: torch.Tensor,
        dv: torch.Tensor,
        pair_bias_gradient: torch.Tensor,
        pair_bias_gradient_output: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if q.ndim != 5:
            raise ValueError(
                "q must have model shape [B, S, N, H, D], got "
                f"{tuple(q.shape)}"
            )
        self._kernel(q.shape[-1])(
            q,
            k,
            v,
            output,
            output_gradient,
            pair_bias,
            residual_mask,
            logsumexp,
            delta,
            dq,
            dk,
            dv,
            pair_bias_gradient,
            pair_bias_gradient_output,
            stream=stream,
        )


_BACKWARD_KERNEL = EvoAttentionBackward()


def get_evoattention_backward() -> EvoAttentionBackward:
    """Return the process-wide D=32/D=64 backward dispatcher."""
    return _BACKWARD_KERNEL


__all__ = [
    "EvoAttentionBackward",
    "SUPPORTED_HEAD_DIMENSIONS",
    "get_evoattention_backward",
]
