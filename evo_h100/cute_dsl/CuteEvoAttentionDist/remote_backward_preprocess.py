"""Fuse inverse remote gradient reshards with backward delta preprocessing."""

from __future__ import annotations

import threading

import torch
from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.compiler import CompileCallable, EnableTVMFFI, PtxasOptions
from cutlass.cute.nvgpu.warpgroup import SmemLayoutAtomKind
from cutlass.cute.runtime import from_dlpack


HEAD_DIMENSION = 32
RESIDUAL_DIMENSION = 128
THREADS = 256
MAX_WORLD_SIZE = 8


def _dynamic_layout_tensor(tensor: torch.Tensor, leading_dim: int) -> cute.Tensor:
    return from_dlpack(
        tensor.detach(), assumed_align=16, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=leading_dim)


@cute.kernel
def _remote_preprocess_kernel(
    input_tmas: tuple,
    input_tensors: tuple,
    residual_input_tmas: tuple,
    residual_input_tensors: tuple,
    output: cute.Tensor,
    output_gradient: cute.Tensor,
    residual_gradient: cute.Tensor,
    delta: cute.Tensor,
    query_gradient_accumulator: cute.Tensor,
    pair_bias_gradient: cute.Tensor,
    shared_layout: cute.ComposedLayout,
    shared_residual_layout: cute.ComposedLayout,
    tile_rows: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
):
    thread_idx, _, _ = cute.arch.thread_idx()
    local_query_block, head_idx, problem_idx = cute.arch.block_idx()
    sequence_count = delta.shape[1]
    context_length = delta.shape[3]
    sequence_idx = problem_idx % sequence_count
    batch_idx = problem_idx // sequence_count
    blocks_per_peer = sequence_count // tile_rows
    peer_rank = local_query_block // blocks_per_peer
    peer_query_block = local_query_block - peer_rank * blocks_per_peer
    query_base = local_query_block * tile_rows

    allocator = cutlass.utils.SmemAllocator()
    shared_gradient = allocator.allocate_tensor(
        cutlass.BFloat16,
        shared_layout.outer,
        swizzle=shared_layout.inner,
    )
    shared_residual_gradient = allocator.allocate_tensor(
        cutlass.BFloat16,
        shared_residual_layout.outer,
        swizzle=shared_residual_layout.inner,
    )
    output_ready = allocator.allocate_array(cutlass.Int64, 1)
    residual_ready = allocator.allocate_array(cutlass.Int64, 1)
    if thread_idx == 0:
        cute.arch.mbarrier_init(output_ready, 1)
        cute.arch.mbarrier_init(residual_ready, 1)
        for peer in cutlass.range_constexpr(world_size):
            cute.nvgpu.cpasync.prefetch_descriptor(input_tmas[peer])
            cute.nvgpu.cpasync.prefetch_descriptor(residual_input_tmas[peer])
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    if thread_idx < 32:
        for peer in cutlass.range_constexpr(world_size):
            if peer_rank == peer:
                input_tiles = cute.zipped_divide(
                    input_tensors[peer], (HEAD_DIMENSION, tile_rows)
                )
                input_problem = input_tiles[
                    None, (0, None, head_idx, sequence_idx, batch_idx)
                ]
                shared_partition, input_partition = (
                    cute.nvgpu.cpasync.tma_partition(
                        input_tmas[peer],
                        0,
                        cute.make_layout(1),
                        cute.group_modes(shared_gradient, 0, 2),
                        input_problem,
                    )
                )
                if thread_idx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        output_ready, HEAD_DIMENSION * tile_rows * 2
                    )
                cute.copy(
                    input_tmas[peer],
                    input_partition[None, peer_query_block],
                    shared_partition,
                    tma_bar_ptr=output_ready,
                )
                if head_idx == 0:
                    residual_tiles = cute.zipped_divide(
                        residual_input_tensors[peer],
                        (RESIDUAL_DIMENSION, tile_rows),
                    )
                    residual_problem = residual_tiles[
                        None, (0, None, sequence_idx, batch_idx)
                    ]
                    residual_shared_partition, residual_input_partition = (
                        cute.nvgpu.cpasync.tma_partition(
                            residual_input_tmas[peer],
                            0,
                            cute.make_layout(1),
                            cute.group_modes(shared_residual_gradient, 0, 2),
                            residual_problem,
                        )
                    )
                    if thread_idx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            residual_ready,
                            RESIDUAL_DIMENSION * tile_rows * 2,
                        )
                    cute.copy(
                        residual_input_tmas[peer],
                        residual_input_partition[None, peer_query_block],
                        residual_shared_partition,
                        tma_bar_ptr=residual_ready,
                    )
    cute.arch.mbarrier_wait(output_ready, 0)

    lane = thread_idx % 32
    warp = thread_idx // 32
    values_per_thread = tile_rows * HEAD_DIMENSION // THREADS
    for value_idx in cutlass.range_constexpr(values_per_thread):
        row = warp + value_idx * (THREADS // 32)
        gradient = shared_gradient[lane, row]
        output_gradient[
            batch_idx,
            sequence_idx,
            query_base + row,
            head_idx,
            lane,
        ] = gradient
        product = (
            gradient.to(cutlass.Float32)
            * output[
                batch_idx,
                sequence_idx,
                query_base + row,
                head_idx,
                lane,
            ].to(cutlass.Float32)
        )
        for offset in (16, 8, 4, 2, 1):
            product += cute.arch.shuffle_sync_bfly(product, offset=offset)
        if lane == 0:
            delta[
                batch_idx,
                sequence_idx,
                head_idx,
                query_base + row,
            ] = product
        if context_length != 128:
            query_gradient_accumulator[
                problem_idx,
                head_idx,
                (query_base + row) * HEAD_DIMENSION + lane,
            ] = cutlass.Float32(0.0)

    if head_idx == 0:
        cute.arch.mbarrier_wait(residual_ready, 0)
        residual_values_per_thread = tile_rows * RESIDUAL_DIMENSION // THREADS
        for value_idx in cutlass.range_constexpr(residual_values_per_thread):
            linear_idx = thread_idx + value_idx * THREADS
            residual_dimension_idx = linear_idx % RESIDUAL_DIMENSION
            row = linear_idx // RESIDUAL_DIMENSION
            residual_gradient[
                batch_idx,
                sequence_idx,
                query_base + row,
                residual_dimension_idx,
            ] = shared_residual_gradient[residual_dimension_idx, row]

    if thread_idx < 128:
        blocks_per_head = context_length // tile_rows
        clear_block = sequence_idx * blocks_per_head + local_query_block
        linear_idx = clear_block * 128 + thread_idx
        pair_offset = (
            (batch_idx * pair_bias_gradient.shape[1] + head_idx)
            * context_length
            * context_length
        )
        pair_bias_gradient.iterator[pair_offset + linear_idx] = cutlass.Float32(0.0)


@cute.jit
def _build_and_launch(
    peer_0: cute.Tensor,
    peer_1: cute.Tensor,
    peer_2: cute.Tensor,
    peer_3: cute.Tensor,
    peer_4: cute.Tensor,
    peer_5: cute.Tensor,
    peer_6: cute.Tensor,
    peer_7: cute.Tensor,
    residual_peer_0: cute.Tensor,
    residual_peer_1: cute.Tensor,
    residual_peer_2: cute.Tensor,
    residual_peer_3: cute.Tensor,
    residual_peer_4: cute.Tensor,
    residual_peer_5: cute.Tensor,
    residual_peer_6: cute.Tensor,
    residual_peer_7: cute.Tensor,
    output: cute.Tensor,
    output_gradient: cute.Tensor,
    residual_gradient: cute.Tensor,
    delta: cute.Tensor,
    query_gradient_accumulator: cute.Tensor,
    pair_bias_gradient: cute.Tensor,
    tile_rows: cutlass.Constexpr[int],
    world_size: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    peers = (peer_0, peer_1, peer_2, peer_3, peer_4, peer_5, peer_6, peer_7)
    residual_peers = (
        residual_peer_0,
        residual_peer_1,
        residual_peer_2,
        residual_peer_3,
        residual_peer_4,
        residual_peer_5,
        residual_peer_6,
        residual_peer_7,
    )
    peer_views = tuple(
        cute.make_tensor(
            peer.iterator,
            cute.select(peer.layout, mode=[4, 2, 3, 1, 0]),
        )
        for peer in peers
    )
    residual_peer_views = tuple(
        cute.make_tensor(
            peer.iterator,
            cute.select(peer.layout, mode=[3, 2, 1, 0]),
        )
        for peer in residual_peers
    )
    layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        SmemLayoutAtomKind.MN_SW64,
        cutlass.BFloat16,
    )
    shared_layout = cute.tile_to_shape(
        layout_atom,
        (HEAD_DIMENSION, tile_rows),
        order=(1, 0),
    )
    residual_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        SmemLayoutAtomKind.MN_SW128,
        cutlass.BFloat16,
    )
    shared_residual_layout = cute.tile_to_shape(
        residual_layout_atom,
        (RESIDUAL_DIMENSION, tile_rows),
        order=(1, 0),
    )
    load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    input_pairs = tuple(
        cute.nvgpu.cpasync.make_tiled_tma_atom(
            load_op,
            peer,
            shared_layout,
            (HEAD_DIMENSION, tile_rows),
        )
        for peer in peer_views
    )
    input_tmas = tuple(pair[0] for pair in input_pairs)
    input_tensors = tuple(pair[1] for pair in input_pairs)
    residual_input_pairs = tuple(
        cute.nvgpu.cpasync.make_tiled_tma_atom(
            load_op,
            peer,
            shared_residual_layout,
            (RESIDUAL_DIMENSION, tile_rows),
        )
        for peer in residual_peer_views
    )
    residual_input_tmas = tuple(pair[0] for pair in residual_input_pairs)
    residual_input_tensors = tuple(pair[1] for pair in residual_input_pairs)
    _remote_preprocess_kernel(
        input_tmas,
        input_tensors,
        residual_input_tmas,
        residual_input_tensors,
        output,
        output_gradient,
        residual_gradient,
        delta,
        query_gradient_accumulator,
        pair_bias_gradient,
        shared_layout,
        shared_residual_layout,
        tile_rows,
        world_size,
    ).launch(
        grid=(
            output.shape[2] // tile_rows,
            output.shape[3],
            output.shape[0] * output.shape[1],
        ),
        block=(THREADS, 1, 1),
        stream=stream,
    )


class RemoteBackwardPreprocess:
    """Remote dO gather, local materialization, delta, and dQ clear."""

    def __init__(self, world_size: int, source_rank: int) -> None:
        if world_size not in (2, 4, 8):
            raise ValueError(f"world_size must be 2, 4, or 8, got {world_size}")
        if not 0 <= source_rank < world_size:
            raise ValueError("source_rank must identify one PE")
        self.world_size = world_size
        self.source_rank = source_rank
        self.tile_rows = 128 // world_size
        self._compiled = None
        self._device_index: int | None = None
        self._lock = threading.Lock()

    def __call__(
        self,
        peer_inputs: tuple[torch.Tensor, ...],
        residual_peer_inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        output_gradient: torch.Tensor,
        residual_gradient: torch.Tensor,
        delta: torch.Tensor,
        query_gradient_accumulator: torch.Tensor,
        pair_bias_gradient: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if len(peer_inputs) != self.world_size:
            raise ValueError(f"expected {self.world_size} peer input views")
        if len(residual_peer_inputs) != self.world_size:
            raise ValueError(
                f"expected {self.world_size} residual peer input views"
            )
        batch, sequence_count, context_length, heads, dimension = output.shape
        if output_gradient.shape != output.shape or dimension != HEAD_DIMENSION:
            raise ValueError("output and dO must match [B,S,N,H,32]")
        if context_length != self.world_size * sequence_count:
            raise ValueError("N must equal world_size*S")
        expected_peer_shape = (
            batch,
            sequence_count,
            sequence_count,
            heads,
            dimension,
        )
        if any(tuple(peer.shape) != expected_peer_shape for peer in peer_inputs):
            raise ValueError(f"peer inputs must each have shape {expected_peer_shape}")
        expected_residual_peer_shape = (
            batch,
            sequence_count,
            sequence_count,
            RESIDUAL_DIMENSION,
        )
        if any(
            tuple(peer.shape) != expected_residual_peer_shape
            for peer in residual_peer_inputs
        ):
            raise ValueError(
                "residual peer inputs must each have shape "
                f"{expected_residual_peer_shape}"
            )
        expected_residual_gradient_shape = (
            batch,
            sequence_count,
            context_length,
            RESIDUAL_DIMENSION,
        )
        if tuple(residual_gradient.shape) != expected_residual_gradient_shape:
            raise ValueError(
                "residual_gradient must have shape "
                f"{expected_residual_gradient_shape}"
            )
        if delta.shape != (batch, sequence_count, heads, context_length):
            raise ValueError("delta must have shape [B,S,H,N]")
        expected_accumulator_shape = (
            batch * sequence_count,
            heads,
            context_length * dimension,
        )
        if tuple(query_gradient_accumulator.shape) != expected_accumulator_shape:
            raise ValueError(
                "query_gradient_accumulator must have shape "
                f"{expected_accumulator_shape}"
            )
        expected_pair_bias_gradient_shape = (
            batch,
            heads,
            context_length,
            context_length,
        )
        if (
            tuple(pair_bias_gradient.shape) != expected_pair_bias_gradient_shape
            or pair_bias_gradient.dtype != torch.float32
        ):
            raise ValueError(
                "pair_bias_gradient must be FP32 with shape "
                f"{expected_pair_bias_gradient_shape}"
            )

        device_index = output.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if stream is None:
            stream = torch.cuda.current_stream(device_index)
        cuda_stream = cuda.CUstream(stream.cuda_stream)
        padded_peers = peer_inputs + (peer_inputs[0],) * (
            MAX_WORLD_SIZE - self.world_size
        )
        padded_residual_peers = residual_peer_inputs + (
            residual_peer_inputs[0],
        ) * (MAX_WORLD_SIZE - self.world_size)
        arguments = (
            *padded_peers,
            *padded_residual_peers,
            output,
            output_gradient,
            residual_gradient,
            delta,
            query_gradient_accumulator,
            pair_bias_gradient,
        )
        if self._compiled is None:
            with self._lock:
                if self._compiled is None:
                    compile_arguments = (
                        *(
                            _dynamic_layout_tensor(peer, leading_dim=4)
                            for peer in padded_peers
                        ),
                        *(
                            _dynamic_layout_tensor(peer, leading_dim=3)
                            for peer in padded_residual_peers
                        ),
                        _dynamic_layout_tensor(output, leading_dim=4),
                        _dynamic_layout_tensor(output_gradient, leading_dim=4),
                        _dynamic_layout_tensor(residual_gradient, leading_dim=3),
                        _dynamic_layout_tensor(delta, leading_dim=3),
                        _dynamic_layout_tensor(
                            query_gradient_accumulator, leading_dim=2
                        ),
                        _dynamic_layout_tensor(pair_bias_gradient, leading_dim=3),
                        self.tile_rows,
                        self.world_size,
                        cuda_stream,
                    )
                    self._compiled = CompileCallable(
                        (PtxasOptions("--maxrregcount=64"), EnableTVMFFI)
                    )(_build_and_launch, *compile_arguments)
                    self._device_index = device_index
        self._compiled(*arguments, cuda_stream)


_KERNELS: dict[tuple[int, int], RemoteBackwardPreprocess] = {}


def get_remote_backward_preprocess(
    world_size: int,
    source_rank: int,
) -> RemoteBackwardPreprocess:
    key = (world_size, source_rank)
    if key not in _KERNELS:
        _KERNELS[key] = RemoteBackwardPreprocess(world_size, source_rank)
    return _KERNELS[key]


__all__ = ["RemoteBackwardPreprocess", "get_remote_backward_preprocess"]
