"""Rectangular D=32 EvoAttention backward for sequence parallelism."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.base_dsl.compiler import CompileCallable, EnableTVMFFI, PtxasOptions


_CUTE_DSL_ROOT = Path(__file__).resolve().parent.parent
if str(_CUTE_DSL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CUTE_DSL_ROOT))

from CuteEvoAttention import evoattention_cute_bwd_d32 as _local
from CuteEvoAttention.evoattention_cute_bwd_d32 import (
    EvoAttentionBackward as _LocalBackward,
)


HEAD_DIMENSION = 32
KEY_TILE_SIZE = 128


@cute.kernel
def _finalize_rectangular_query_gradient_kernel(
    accumulator: cute.Tensor,
    query_gradient: cute.Tensor,
    pair_bias_gradient: cute.Tensor,
    pair_bias_gradient_output: cute.Tensor,
    scale: Float32,
    tiled_mma: cute.TiledMma,
    accumulator_layout: cute.Layout,
    output_layout: cute.ComposedLayout,
    global_to_shared_copy: cute.TiledCopy,
    shared_to_register_copy: cute.TiledCopy,
    global_store_copy: cute.TiledCopy,
):
    """Finalize dQ and convert the complete replicated dPairBias matrix."""
    thread_idx, _, _ = cute.arch.thread_idx()
    query_block, head_idx, problem_idx = cute.arch.block_idx()
    sequence_count = query_gradient.shape[0] // pair_bias_gradient.shape[0]
    sequence_idx = problem_idx % sequence_count
    batch_idx = problem_idx // sequence_count
    context_length = query_gradient.shape[1]
    blocks_per_sequence = context_length // _local.QUERY_TILE_SIZE
    pair_chunk = sequence_idx * blocks_per_sequence + query_block
    chunks_per_cta = context_length // sequence_count
    pair_offset = (
        (batch_idx * pair_bias_gradient.shape[1] + head_idx)
        * context_length
        * context_length
        + pair_chunk * _local.QUERY_TILE_SIZE * chunks_per_cta
    )
    conversion_iterations = chunks_per_cta // 2
    for chunk in cutlass.range(conversion_iterations, unroll=1):
        pair_index = thread_idx + chunk * 128
        if pair_index < _local.QUERY_TILE_SIZE * chunks_per_cta:
            pair_bias_gradient_output.iterator[pair_offset + pair_index] = (
                pair_bias_gradient.iterator[pair_offset + pair_index]
            ).to(cutlass.BFloat16)

    allocator = cutlass.utils.SmemAllocator()
    shared_accumulator = allocator.allocate_tensor(
        Float32, accumulator_layout, byte_alignment=1024
    )
    shared_accumulator_flat = cute.make_tensor(
        shared_accumulator.iterator,
        cute.make_layout(cute.size(shared_accumulator)),
    )
    shared_output = cute.make_tensor(
        cute.recast_ptr(shared_accumulator.iterator, dtype=cutlass.BFloat16),
        output_layout,
    )
    accumulator_tile = cute.local_tile(
        accumulator[problem_idx, head_idx, None],
        (_local.QUERY_TILE_SIZE * HEAD_DIMENSION,),
        (query_block,),
    )
    output_tile = cute.local_tile(
        query_gradient[problem_idx, None, head_idx, None],
        (_local.QUERY_TILE_SIZE, HEAD_DIMENSION),
        (query_block, 0),
    )
    load_thread = global_to_shared_copy.get_slice(thread_idx)
    cute.copy(
        global_to_shared_copy,
        load_thread.partition_S(accumulator_tile),
        load_thread.partition_D(shared_accumulator_flat),
    )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()

    shared_thread = shared_to_register_copy.get_slice(thread_idx)
    shared_fragment = shared_thread.partition_S(shared_accumulator)
    accumulator_fragment = cute.make_fragment(
        tiled_mma.partition_shape_C(
            (_local.QUERY_TILE_SIZE, HEAD_DIMENSION)
        ),
        Float32,
    )
    accumulator_view = cute.make_tensor(
        accumulator_fragment.iterator,
        cute.make_layout(shared_fragment.shape),
    )
    cute.autovec_copy(shared_fragment, accumulator_view)
    output_fragment = cute.make_fragment_like(
        accumulator_fragment, cutlass.BFloat16
    )
    output_fragment.store(
        (accumulator_fragment.load() * scale).to(cutlass.BFloat16)
    )
    cute.arch.barrier()

    register_store = cute.make_tiled_copy_C(
        cute.make_copy_atom(
            _local.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        ),
        tiled_mma,
    ).get_slice(thread_idx)
    cute.copy(
        register_store,
        register_store.retile(output_fragment),
        register_store.partition_D(shared_output),
    )
    cute.arch.barrier()

    store_thread = global_store_copy.get_slice(thread_idx)
    shared_partition = store_thread.partition_S(shared_output)
    register_partition = cute.make_fragment_like(
        shared_partition, cutlass.BFloat16
    )
    cute.autovec_copy(shared_partition, register_partition)
    cute.copy(
        global_store_copy,
        register_partition,
        store_thread.partition_D(output_tile),
    )


@cute.jit
def _launch_rectangular_query_gradient_finalize(
    accumulator: cute.Tensor,
    query_gradient: cute.Tensor,
    pair_bias_gradient: cute.Tensor,
    pair_bias_gradient_output: cute.Tensor,
    scale: Float32,
    stream: cuda.CUstream,
):
    tiled_mma = _local.sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        _local.warpgroup.OperandMajorMode.K,
        _local.warpgroup.OperandMajorMode.K,
        Float32,
        atom_layout_mnk=(1, 1, 1),
        tiler_mn=(_local.QUERY_TILE_SIZE, HEAD_DIMENSION),
    )
    accumulator_layout = cute.make_layout(
        (_local.QUERY_TILE_SIZE * HEAD_DIMENSION, 1)
    )
    output_layout = _local._sm90_make_smem_layout(
        cutlass.BFloat16,
        _local.LayoutEnum.ROW_MAJOR,
        (_local.QUERY_TILE_SIZE, HEAD_DIMENSION),
        major_mode_size=HEAD_DIMENSION,
    )
    vector_bytes = 16
    accumulator_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            _local.cpasync.CopyG2SOp(
                cache_mode=_local.cpasync.LoadCacheMode.GLOBAL
            ),
            Float32,
            num_bits_per_copy=vector_bytes * 8,
        ),
        cute.make_layout(128),
        cute.make_layout(vector_bytes * 8 // Float32.width),
    )
    shared_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Float32,
            num_bits_per_copy=vector_bytes * 8,
        ),
        cute.make_layout((128, 1)),
        cute.make_layout(vector_bytes * 8 // Float32.width),
    )
    output_copy = _local._copy_tiled_copy_2d(
        cutlass.BFloat16,
        threads_per_row=HEAD_DIMENSION // 8,
        num_threads=128,
        num_copy_elems=8,
    )
    smem_size = max(
        cute.size_in_bytes(Float32, accumulator_layout),
        cute.size_in_bytes(cutlass.BFloat16, output_layout),
    )
    _finalize_rectangular_query_gradient_kernel(
        accumulator,
        query_gradient,
        pair_bias_gradient,
        pair_bias_gradient_output,
        scale,
        tiled_mma,
        accumulator_layout,
        output_layout,
        accumulator_copy,
        shared_copy,
        output_copy,
    ).launch(
        grid=(
            query_gradient.shape[1] // _local.QUERY_TILE_SIZE,
            query_gradient.shape[2],
            query_gradient.shape[0],
        ),
        block=(128, 1, 1),
        smem=smem_size,
        stream=stream,
    )


class _NoPreprocessBackwardLaunch:
    """Launch the main/final kernels after fused remote preprocessing."""

    def __init__(self) -> None:
        self.main = _local._EvoAttentionBackwardMain()

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        output: cute.Tensor,
        output_gradient: cute.Tensor,
        logsumexp: cute.Tensor,
        delta: cute.Tensor,
        query_gradient_accumulator: cute.Tensor,
        query_gradient: cute.Tensor,
        key_gradient: cute.Tensor,
        value_gradient: cute.Tensor,
        pair_bias: cute.Tensor,
        residual_mask: cute.Tensor,
        pair_bias_gradient: cute.Tensor,
        pair_bias_gradient_output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        pair_bias_gradient_flat = cute.make_tensor(
            pair_bias_gradient.iterator,
            cute.make_layout(cute.size(pair_bias_gradient)),
        )
        pair_bias_gradient_output_flat = cute.make_tensor(
            pair_bias_gradient_output.iterator,
            cute.make_layout(cute.size(pair_bias_gradient_output)),
        )
        element_count = cute.size(pair_bias_gradient_flat)
        elementwise_grid = (
            cute.ceil_div(
                element_count,
                _local.ELEMENTWISE_THREADS * _local.ELEMENTS_PER_THREAD,
            ),
            1,
            1,
        )

        use_split_pair_bias = (
            pair_bias.shape[2] == 16 and pair_bias.shape[3] >= 640
        )
        if use_split_pair_bias:
            self.main(
                q, k, v, output_gradient, logsumexp, delta,
                query_gradient_accumulator, query_gradient, key_gradient,
                value_gradient, Float32(_local.SOFTMAX_SCALE), pair_bias,
                residual_mask, pair_bias_gradient, False, stream,
            )
            _local._launch_split_sequence_pair_bias_gradient(
                q, k, v, output_gradient, pair_bias, residual_mask,
                logsumexp, delta, pair_bias_gradient, stream,
            )
        else:
            self.main(
                q, k, v, output_gradient, logsumexp, delta,
                query_gradient_accumulator, query_gradient, key_gradient,
                value_gradient, Float32(_local.SOFTMAX_SCALE), pair_bias,
                residual_mask, pair_bias_gradient, True, stream,
            )

        if pair_bias.shape[3] == KEY_TILE_SIZE:
            _local._convert_fp32_to_bf16(
                pair_bias_gradient_flat,
                pair_bias_gradient_output_flat,
            ).launch(
                grid=elementwise_grid,
                block=(_local.ELEMENTWISE_THREADS, 1, 1),
                stream=stream,
            )
        else:
            _launch_rectangular_query_gradient_finalize(
                query_gradient_accumulator,
                query_gradient,
                pair_bias_gradient,
                pair_bias_gradient_output,
                Float32(_local.SOFTMAX_SCALE),
                stream,
            )


class DistributedEvoAttentionBackward(_LocalBackward):
    """Use the optimized local backward with a row-sharded sequence axis."""

    clear_pair_bias_gradient = True

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
        if self.clear_pair_bias_gradient:
            # The inherited preprocess distributes its clear across S=N CTAs.
            # With sequence parallelism S=N/P, clear the replicated matrix here.
            pair_bias_gradient.zero_()
        super().__call__(
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

    @staticmethod
    def _validate(
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
    ) -> tuple[int, int, int, int]:
        if q.ndim != 5:
            raise ValueError("q must have shape [B, S, N, H, 32]")
        batch, sequence_count, context_length, heads, dimension = q.shape
        if dimension != HEAD_DIMENSION or context_length % KEY_TILE_SIZE:
            raise ValueError("D must be 32 and N must be divisible by 128")
        if sequence_count % 16:
            raise ValueError("the row-sharded sequence count must be divisible by 16")
        if heads % 4:
            raise ValueError("H must be divisible by 4")

        activations = (q, k, v, output, output_gradient, dq, dk, dv)
        if any(
            tensor.shape != q.shape or tensor.dtype != torch.bfloat16
            for tensor in activations
        ):
            raise ValueError(
                "all activation and gradient tensors must match q and be BF16"
            )

        pair_shape = (batch, heads, context_length, context_length)
        if (
            pair_bias.shape != (batch, 1, heads, context_length, context_length)
            or pair_bias.dtype != torch.bfloat16
        ):
            raise ValueError("pair_bias must be BF16 [B, 1, H, N, N]")
        if (
            pair_bias_gradient.shape != pair_shape
            or pair_bias_gradient.dtype != torch.float32
        ):
            raise ValueError("pair_bias_gradient must be FP32 [B, H, N, N]")
        if (
            pair_bias_gradient_output.shape != pair_shape
            or pair_bias_gradient_output.dtype != torch.bfloat16
        ):
            raise ValueError("pair_bias_gradient_output must be BF16 [B, H, N, N]")

        expected_mask_shape = (batch, sequence_count, 1, 1, context_length)
        if (
            residual_mask.shape != expected_mask_shape
            or residual_mask.dtype != torch.bfloat16
        ):
            raise ValueError("residual_mask must be BF16 [B, S, 1, 1, N]")
        statistics_shape = (batch, sequence_count, heads, context_length)
        if (
            logsumexp.shape != statistics_shape
            or delta.shape != statistics_shape
            or logsumexp.dtype != torch.float32
            or delta.dtype != torch.float32
        ):
            raise ValueError("logsumexp and delta must be FP32 [B, S, H, N]")

        tensors = activations + (
            pair_bias,
            residual_mask,
            logsumexp,
            delta,
            pair_bias_gradient,
            pair_bias_gradient_output,
        )
        if any(tensor.device != q.device for tensor in tensors):
            raise ValueError("all tensors must reside on the same CUDA device")
        compact_tensors = (
            q,
            output,
            output_gradient,
            dq,
            dk,
            dv,
            residual_mask,
            logsumexp,
            delta,
            pair_bias_gradient,
            pair_bias_gradient_output,
        )
        if any(
            not tensor.is_cuda or not tensor.is_contiguous()
            for tensor in compact_tensors
        ):
            raise ValueError("all outputs, workspaces, Q, dO, and O must be compact")
        if any(
            not tensor.is_cuda
            or tensor.stride(-1) != 1
            or any(stride < 0 for stride in tensor.stride())
            for tensor in (k, v)
        ):
            raise ValueError("K/V must have positive strides and a contiguous D axis")
        if not pair_bias.is_cuda:
            raise ValueError("pair_bias must be a CUDA tensor")
        if pair_bias.stride(-1) != 1 or any(
            stride < 0 for stride in pair_bias.stride()
        ):
            raise ValueError("pair_bias must have a positive-stride contiguous K axis")
        return batch, sequence_count, context_length, heads

    def _make_launch(self):
        return _local._BackwardLaunch()

    def _compile(
        self,
        tensors: tuple[torch.Tensor, ...],
        cuda_stream: cuda.CUstream,
    ) -> object:
        activation_divisibility = (16, 128, 4)
        statistic_divisibility = (16, 4, 128)
        pair_divisibility = (1, 4, 128, 128)
        mask_divisibility = (1, 16, 1, 1, 128)
        compile_arguments = (
            _local._dynamic_tensor(tensors[0], activation_divisibility),
            _local._dynamic_layout_tensor(tensors[1], leading_dim=3),
            _local._dynamic_layout_tensor(tensors[2], leading_dim=3),
            _local._dynamic_tensor(tensors[3], activation_divisibility),
            _local._dynamic_tensor(tensors[4], activation_divisibility),
            *(
                _local._dynamic_tensor(tensor, statistic_divisibility)
                for tensor in tensors[5:8]
            ),
            *(
                _local._dynamic_tensor(tensor, activation_divisibility)
                for tensor in tensors[8:11]
            ),
            _local._dynamic_layout_tensor(tensors[11], leading_dim=4),
            _local._dynamic_tensor(tensors[12], mask_divisibility),
            _local._dynamic_tensor(tensors[13], pair_divisibility),
            _local._dynamic_tensor(tensors[14], pair_divisibility),
            cuda_stream,
        )
        return CompileCallable((PtxasOptions("--maxrregcount=255"), EnableTVMFFI))(
            self._make_launch(),
            *compile_arguments,
        )


class DistributedEvoAttentionBackwardNoPreprocess(DistributedEvoAttentionBackward):
    """Backward mainloop variant paired with remote fused preprocessing."""

    clear_pair_bias_gradient = False

    def _make_launch(self):
        return _NoPreprocessBackwardLaunch()


_BACKWARD = DistributedEvoAttentionBackward()
_BACKWARD_NO_PREPROCESS = DistributedEvoAttentionBackwardNoPreprocess()


def get_distributed_evoattention_backward() -> DistributedEvoAttentionBackward:
    """Return the cached rectangular backward launcher."""
    return _BACKWARD


def get_distributed_evoattention_backward_no_preprocess(
) -> DistributedEvoAttentionBackwardNoPreprocess:
    """Return the launcher whose dO/delta preprocessing is fused remotely."""
    return _BACKWARD_NO_PREPROCESS


__all__ = [
    "DistributedEvoAttentionBackward",
    "DistributedEvoAttentionBackwardNoPreprocess",
    "get_distributed_evoattention_backward",
    "get_distributed_evoattention_backward_no_preprocess",
]
