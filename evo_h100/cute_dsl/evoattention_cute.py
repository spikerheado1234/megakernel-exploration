"""Runtime head-dimension dispatch for CuTe-DSL EvoAttention forward.

The prepared forward interface is unchanged:

* query/key/value/output: ``[B, S, H, N, D]`` BF16
* pair bias: ``[B, H, N, N]`` FP32
* residual mask: ``[B, S, N]`` FP32
* log-sum-exp: ``[B, S, H, N]`` FP32

``D`` is inspected on every call and dispatched to an independently tuned,
runtime-dynamic D=32 or D=64 CuTe specialization. Importing this module does
not compile either kernel; each specialization is imported and compiled only
when first used.
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
    """Dispatch prepared forward calls by ``q.shape[-1]``."""

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

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_bias: torch.Tensor,
        residual_mask: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if q.ndim != 5:
            raise ValueError(
                "q must have prepared shape [B, S, H, N, D], got "
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


_FORWARD_KERNEL = EvoAttentionForward()


def get_evoattention_forward() -> EvoAttentionForward:
    """Return the process-wide D=32/D=64 forward dispatcher."""
    return _FORWARD_KERNEL


__all__ = [
    "EvoAttentionForward",
    "SUPPORTED_HEAD_DIMENSIONS",
    "get_evoattention_forward",
]
