"""Reference launches, collectives, and timing for the distributed benchmark."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch
import torch.distributed as dist
import triton
import triton.language as tl


DEFAULT_REFERENCE_PATH = Path("/home/ahangupta/Debug/MegaFold/debug/evoattention.py")


def load_reference_module() -> ModuleType:
    """Load MegaFold's Triton EvoAttention (``TritonEvoformer`` autograd op)."""
    reference_path = Path(
        os.environ.get("EVOATTENTION_REFERENCE", DEFAULT_REFERENCE_PATH)
    )
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"Triton EvoAttention reference not found at {reference_path}; "
            "set EVOATTENTION_REFERENCE to override it"
        )
    spec = importlib.util.spec_from_file_location(
        "_distributed_triton_evoattention", reference_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Triton reference from {reference_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@triton.jit
def _convert_mask(source, destination, count, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    active = offsets < count
    values = tl.load(source + offsets, mask=active)
    tl.store(destination + offsets, tl.where(values != 0, 0.0, -1.0e9), mask=active)


def convert_mask(residual_mask: torch.Tensor, additive_mask: torch.Tensor) -> None:
    """Convert a 0/1 residual mask into the additive ``0 / -1e9`` Triton mask."""
    mask_count = additive_mask.numel()
    _convert_mask[(triton.cdiv(mask_count, 256),)](
        residual_mask, additive_mask, mask_count, BLOCK_SIZE=256
    )


def row_to_column(
    local_tensor: torch.Tensor,
    send_buffers: tuple[torch.Tensor, ...],
    receive_buffers: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> None:
    """Reshard ``[B,S,N,C]`` to ``[B,N,S,C]`` with NCCL All-to-All."""
    world_size = len(send_buffers)
    column_shard = local_tensor.shape[2] // world_size
    for peer in range(world_size):
        send_buffers[peer].copy_(
            local_tensor[:, :, peer * column_shard : (peer + 1) * column_shard]
        )
    dist.all_to_all(list(receive_buffers), list(send_buffers))
    source_rows = local_tensor.shape[1]
    for source in range(world_size):
        output[:, source * source_rows : (source + 1) * source_rows].copy_(
            receive_buffers[source]
        )


def column_to_row(
    column_shard: torch.Tensor,
    send_buffers: tuple[torch.Tensor, ...],
    receive_buffers: tuple[torch.Tensor, ...],
    output: torch.Tensor,
) -> None:
    """Invert the sequence-parallel row-to-column reshard."""
    world_size = len(send_buffers)
    sequence_count = output.shape[1]
    for source in range(world_size):
        send_buffers[source].copy_(
            column_shard[
                :, source * sequence_count : (source + 1) * sequence_count
            ]
        )
    dist.all_to_all(list(receive_buffers), list(send_buffers))
    for peer in range(world_size):
        output[:, :, peer * sequence_count : (peer + 1) * sequence_count].copy_(
            receive_buffers[peer]
        )


def measure_distributed(
    operation: Callable[[], None],
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, ...]:
    """Time an operation with CUDA events and report the slowest rank."""
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    dist.barrier()

    samples = []
    for _ in range(repeats):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            [start.elapsed_time(end) / iterations],
            device="cuda",
            dtype=torch.float64,
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(float(elapsed.item()))
    return tuple(samples)


__all__ = [
    "column_to_row",
    "convert_mask",
    "load_reference_module",
    "measure_distributed",
    "row_to_column",
]
