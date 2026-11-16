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


DEFAULT_REFERENCE_PATH = Path(
    "/home/ahangupta/MegaFold/megafold/model/FusedEvoAttention/evoattention.py"
)


def load_reference_module() -> ModuleType:
    """Load the authoritative MegaFold Triton implementation."""
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


def launch_triton_forward(
    reference: ModuleType,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pair_bias_source: torch.Tensor,
    residual_mask: torch.Tensor,
    query_workspace: torch.Tensor,
    key_workspace: torch.Tensor,
    value_workspace: torch.Tensor,
    pair_bias_workspace: torch.Tensor,
    mask_workspace: torch.Tensor,
    output_workspace: torch.Tensor,
    output: torch.Tensor,
    logsumexp: torch.Tensor,
) -> None:
    """Prepare model-layout inputs and launch the Triton forward baseline."""
    query_workspace.copy_(query.transpose(2, 3))
    key_workspace.copy_(key.transpose(2, 3))
    value_workspace.copy_(value.transpose(2, 3))
    pair_bias_workspace.copy_(pair_bias_source)
    mask_count = mask_workspace.numel()
    _convert_mask[(triton.cdiv(mask_count, 256),)](
        residual_mask, mask_workspace, mask_count, BLOCK_SIZE=256
    )

    batch, sequence_count, heads, context_length, head_dimension = (
        query_workspace.shape
    )
    reference._attn_fwd[
        (triton.cdiv(context_length, 16), batch * sequence_count * heads, 1)
    ](
        Q=query_workspace,
        K=key_workspace,
        V=value_workspace,
        res_mask=mask_workspace,
        pair_bias=pair_bias_workspace,
        softmax_scale=head_dimension**-0.5,
        M=logsumexp,
        O=output_workspace,
        stride_Q_batch=query_workspace.stride(0),
        stride_Q_msa=query_workspace.stride(1),
        stride_Q_head=query_workspace.stride(2),
        stride_Q_seq=query_workspace.stride(3),
        stride_Q_dim=query_workspace.stride(4),
        stride_K_batch=key_workspace.stride(0),
        stride_K_msa=key_workspace.stride(1),
        stride_K_head=key_workspace.stride(2),
        stride_K_seq=key_workspace.stride(3),
        stride_K_dim=key_workspace.stride(4),
        stride_V_batch=value_workspace.stride(0),
        stride_V_msa=value_workspace.stride(1),
        stride_V_head=value_workspace.stride(2),
        stride_V_seq=value_workspace.stride(3),
        stride_V_dim=value_workspace.stride(4),
        stride_O_batch=output_workspace.stride(0),
        stride_O_msa=output_workspace.stride(1),
        stride_O_head=output_workspace.stride(2),
        stride_O_seq=output_workspace.stride(3),
        stride_O_dim=output_workspace.stride(4),
        stride_pair_bias_batch=pair_bias_workspace.stride(0),
        stride_pair_bias_head=pair_bias_workspace.stride(1),
        stride_pair_bias_seq1=pair_bias_workspace.stride(2),
        stride_pair_bias_seq2=pair_bias_workspace.stride(3),
        stride_mask_batch=mask_workspace.stride(0),
        stride_mask_msa=mask_workspace.stride(1),
        stride_mask_seq=mask_workspace.stride(2),
        BATCH_SIZE=batch,
        HEAD=heads,
        N_SEQ=sequence_count,
        SEQ_LEN=context_length,
        DIM=head_dimension,
        BLOCK_DIM=max(triton.next_power_of_2(head_dimension), 16),
        BLOCK_SIZE_Q=16,
        BLOCK_SIZE_KV=64,
        num_stages=3,
        num_warps=4,
    )
    output.copy_(output_workspace.transpose(2, 3))


def launch_triton_backward(
    reference: ModuleType,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gradient: torch.Tensor,
    pair_bias: torch.Tensor,
    residual_mask: torch.Tensor,
    logsumexp: torch.Tensor,
    delta: torch.Tensor,
    query_gradient: torch.Tensor,
    key_gradient: torch.Tensor,
    value_gradient: torch.Tensor,
    pair_bias_gradient: torch.Tensor,
) -> None:
    """Launch every kernel in the authoritative Triton backward path."""
    batch, sequence_count, heads, context_length, head_dimension = query.shape
    problem_count = batch * sequence_count * heads
    pair_bias_gradient.zero_()
    reference._attn_bwd_preprocess[
        (triton.cdiv(context_length, 32), problem_count, 1)
    ](
        O=output,
        dO=output_gradient,
        D=delta,
        SEQ_LEN=context_length,
        DIM=head_dimension,
        BLOCK_DIM=head_dimension,
        BLOCK_SIZE_Q=32,
        num_stages=2,
        num_warps=4,
    )
    common_arguments = dict(
        Q=query,
        K=key,
        V=value,
        res_mask=residual_mask,
        pair_bias=pair_bias,
        softmax_scale=head_dimension**-0.5,
        dO=output_gradient,
        dQ=query_gradient,
        dK=key_gradient,
        dV=value_gradient,
        M=logsumexp,
        D=delta,
        stride_batch=query.stride(0),
        stride_msa=query.stride(1),
        stride_head=query.stride(2),
        stride_seq=query.stride(3),
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
        BLOCK_DIM=head_dimension,
        DIM=head_dimension,
    )
    reference._attn_bwd_dk_dv[
        (triton.cdiv(context_length, 16), problem_count, 1)
    ](
        **common_arguments,
        BLOCK_SIZE_Q=64,
        BLOCK_SIZE_KV=16,
        num_stages=3,
        num_warps=4,
    )
    reference._attn_bwd_dq[
        (triton.cdiv(context_length, 16), problem_count, 1)
    ](
        **common_arguments,
        d_pair_bias=pair_bias_gradient,
        stride_d_pair_bias_batch=pair_bias_gradient.stride(0),
        stride_d_pair_bias_head=pair_bias_gradient.stride(1),
        stride_d_pair_bias_seq1=pair_bias_gradient.stride(2),
        stride_d_pair_bias_seq2=pair_bias_gradient.stride(3),
        BLOCK_SIZE_Q=16,
        BLOCK_SIZE_KV=64,
        num_stages=3,
        num_warps=4,
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
    "launch_triton_backward",
    "launch_triton_forward",
    "load_reference_module",
    "measure_distributed",
    "row_to_column",
]
