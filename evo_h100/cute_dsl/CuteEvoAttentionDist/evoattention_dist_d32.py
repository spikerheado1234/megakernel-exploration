"""Distributed fused EvoAttention forward for NVIDIA Hopper.

The public entry point, :func:`get_distributed_evoattention_forward`, returns a cached
TVM-FFI callable. It accepts tensors in MegaFold's model-facing contract:

- query/key/value: ``[B, S, N, H, 32]`` BF16, where ``N = P * S``
- each remote attention-output view: ``[B, S, S, H, 32]`` BF16
- each remote residual-output view: ``[B, S, S, 128]`` BF16
- pair bias: ``[B, 1, H, N, N]`` BF16
- residual mask: ``[B, S, 1, 1, N]`` BF16 containing ``0`` or ``1``
- log-sum-exp: ``[B, S, H, N]`` FP32

The implementation uses one TMA producer warp and two 128-thread WGMMA
consumer warpgroups. The consumers process the same M=64 query rows for two
adjacent sequences, sharing the BF16 pair-bias tile while keeping independent
Q/K/V, residual-mask, output, and softmax state. B, S, H, and N remain runtime
dynamic, while D=32 and the tile schedule are static. The epilogue divides
each M=64 output tile into M=16 TMA stores and routes those stores directly to
the symmetric output allocation on the owning NVSHMEM PE.
"""

from __future__ import annotations

import math
import threading

import torch
from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.base_dsl.compiler import CompileCallable, EnableTVMFFI, PtxasOptions
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu.warpgroup import Field, OperandMajorMode, OperandSource
from cutlass.cute.nvgpu.warpgroup import SmemLayoutAtomKind
from cutlass.pipeline import NamedBarrier
from cutlass.utils import hopper_helpers as sm90_utils
from cutlass.utils.layout import LayoutEnum


HEAD_DIMENSION = 32
RESIDUAL_DIMENSION = 128
QUERY_TILE_SIZE = 64
KEY_TILE_SIZE = 64
CONSUMER_WARPGROUP_COUNT = 2
THREADS_PER_WARPGROUP = 128
CONSUMER_THREAD_COUNT = CONSUMER_WARPGROUP_COUNT * THREADS_PER_WARPGROUP
PRODUCER_WARP_INDEX = 8
THREADS_PER_BLOCK = CONSUMER_THREAD_COUNT + 32
CTA_OUTPUT_ROW_SLOTS = CONSUMER_WARPGROUP_COUNT * QUERY_TILE_SIZE
MAX_WORLD_SIZE = 8
REMOTE_STORE_ROWS = 16
SHORT_SEQUENCE_LENGTH = 2 * KEY_TILE_SIZE
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)
KEY_VALUE_STAGES = 3
PAIR_BIAS_STAGES = 1
PAIR_BIAS_EVICT_LAST_POLICY = 0x14F0000000000000
QK0_TO_QK1_BARRIER_ID = 3
QK1_TO_PV0_BARRIER_ID = 4
OUTPUT_READY_BARRIER_ID = 5


def _as_dynamic_tensor(
    tensor: torch.Tensor,
    modes: tuple[int, ...],
    *,
    alignment: int,
    tiled_modes: tuple[int, ...] = (),
) -> cute.Tensor:
    """Convert a compact PyTorch view and mark selected modes dynamic."""
    dynamic_tensor = from_dlpack(
        tensor.detach(),
        assumed_align=alignment,
        enable_tvm_ffi=True,
    )
    for mode in modes:
        divisibility = QUERY_TILE_SIZE if mode in tiled_modes else 1
        dynamic_tensor = dynamic_tensor.mark_compact_shape_dynamic(
            mode=mode,
            stride_order=tensor.dim_order(),
            divisibility=divisibility,
        )
    return dynamic_tensor


def _as_dynamic_layout_tensor(
    tensor: torch.Tensor,
    *,
    alignment: int,
    leading_dim: int,
) -> cute.Tensor:
    """Preserve a runtime-strided flat layout with one known unit stride."""
    return from_dlpack(
        tensor.detach(),
        assumed_align=alignment,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=leading_dim)


@cute.jit
def _store_remote_output_chunk(
    output_tma: cute.CopyAtom,
    output_tensor: cute.Tensor,
    shared_output: cute.Tensor,
    chunk: cutlass.Constexpr[int],
    store_rows: cutlass.Constexpr[int],
    local_query_block: cutlass.Int32,
    head_idx: cutlass.Int32,
    sequence_0: cutlass.Int32,
    batch_idx: cutlass.Int32,
):
    """Store one destination-aligned chunk for both consumer warpgroups."""
    output_tiles = cute.zipped_divide(
        output_tensor, (HEAD_DIMENSION, store_rows)
    )
    output_problem_0 = output_tiles[
        None, (0, None, head_idx, sequence_0, batch_idx)
    ]
    output_problem_1 = output_tiles[
        None, (0, None, head_idx, sequence_0 + 1, batch_idx)
    ]
    chunks_per_consumer = QUERY_TILE_SIZE // store_rows
    shared_output_0 = cute.local_tile(
        shared_output,
        (HEAD_DIMENSION, store_rows),
        (0, chunk),
    )
    shared_output_1 = cute.local_tile(
        shared_output,
        (HEAD_DIMENSION, store_rows),
        (0, chunks_per_consumer + chunk),
    )
    shared_partition_0, global_partition_0 = cute.nvgpu.cpasync.tma_partition(
        output_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_output_0, 0, 2),
        output_problem_0,
    )
    shared_partition_1, global_partition_1 = cute.nvgpu.cpasync.tma_partition(
        output_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_output_1, 0, 2),
        output_problem_1,
    )
    cute.copy(
        output_tma,
        shared_partition_0,
        global_partition_0[None, local_query_block],
    )
    cute.copy(
        output_tma,
        shared_partition_1,
        global_partition_1[None, local_query_block],
    )


@cute.jit
def _load_residual_chunk(
    input_tma: cute.CopyAtom,
    input_tensor: cute.Tensor,
    shared_input: cute.Tensor,
    global_query_block: cutlass.Int32,
    sequence_0: cutlass.Int32,
    batch_idx: cutlass.Int32,
    barrier_ptr: cute.Pointer,
):
    """Load one M=16 residual-input chunk for both owned sequences."""
    input_tiles = cute.zipped_divide(
        input_tensor, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    input_problem_0 = input_tiles[
        None, (0, None, sequence_0, batch_idx)
    ]
    input_problem_1 = input_tiles[
        None, (0, None, sequence_0 + 1, batch_idx)
    ]
    shared_input_0 = cute.local_tile(
        shared_input,
        (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS),
        (0, 0),
    )
    shared_input_1 = cute.local_tile(
        shared_input,
        (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS),
        (0, 1),
    )
    shared_partition_0, global_partition_0 = cute.nvgpu.cpasync.tma_partition(
        input_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_input_0, 0, 2),
        input_problem_0,
    )
    shared_partition_1, global_partition_1 = cute.nvgpu.cpasync.tma_partition(
        input_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_input_1, 0, 2),
        input_problem_1,
    )
    cute.copy(
        input_tma,
        global_partition_0[None, global_query_block],
        shared_partition_0,
        tma_bar_ptr=barrier_ptr,
    )
    cute.copy(
        input_tma,
        global_partition_1[None, global_query_block],
        shared_partition_1,
        tma_bar_ptr=barrier_ptr,
    )


@cute.jit
def _store_remote_residual_chunk(
    output_tma: cute.CopyAtom,
    output_tensor: cute.Tensor,
    shared_input: cute.Tensor,
    local_query_block: cutlass.Int32,
    sequence_0: cutlass.Int32,
    batch_idx: cutlass.Int32,
):
    """Store both M=16 residual-input chunks to one destination PE."""
    output_tiles = cute.zipped_divide(
        output_tensor, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    output_problem_0 = output_tiles[
        None, (0, None, sequence_0, batch_idx)
    ]
    output_problem_1 = output_tiles[
        None, (0, None, sequence_0 + 1, batch_idx)
    ]
    shared_input_0 = cute.local_tile(
        shared_input,
        (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS),
        (0, 0),
    )
    shared_input_1 = cute.local_tile(
        shared_input,
        (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS),
        (0, 1),
    )
    shared_partition_0, global_partition_0 = cute.nvgpu.cpasync.tma_partition(
        output_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_input_0, 0, 2),
        output_problem_0,
    )
    shared_partition_1, global_partition_1 = cute.nvgpu.cpasync.tma_partition(
        output_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_input_1, 0, 2),
        output_problem_1,
    )
    cute.copy(
        output_tma,
        shared_partition_0,
        global_partition_0[None, local_query_block],
    )
    cute.copy(
        output_tma,
        shared_partition_1,
        global_partition_1[None, local_query_block],
    )


@cute.kernel
def _evoattention_forward_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    query_tma: cute.CopyAtom,
    key_tma: cute.CopyAtom,
    value_tma: cute.CopyAtom,
    pair_bias_tma: cute.CopyAtom,
    pair_bias_tensor: cute.Tensor,
    residual_mask_tma: cute.CopyAtom,
    residual_mask_tensor: cute.Tensor,
    residual_input_tma: cute.CopyAtom,
    residual_input_tensor: cute.Tensor,
    local_output_tma: cute.CopyAtom,
    local_output_tensor: cute.Tensor,
    output_tmas: tuple,
    output_tensors: tuple,
    residual_output_tmas: tuple,
    residual_output_tensors: tuple,
    completion_peers: tuple,
    completion: cute.Tensor,
    completion_counter: cute.Tensor,
    completion_epoch: cutlass.Int32,
    lse: cute.Tensor,
    query_layout: cute.ComposedLayout,
    key_layout: cute.ComposedLayout,
    value_layout: cute.ComposedLayout,
    pair_bias_layout: cute.ComposedLayout,
    output_layout: cute.ComposedLayout,
    residual_io_layout: cute.ComposedLayout,
    score_mma: cute.TiledMma,
    output_mma: cute.TiledMma,
    output_register_copy: cute.TiledCopy,
    world_size: cutlass.Constexpr[int],
):
    """Run the warp-specialized SM90 forward pass.

    Warps 0-7 form two consumer warpgroups. Warp 8 is the sole TMA producer.
    Each consumer owns one sequence's M=64 query tile and a three-stage K/V
    pipeline; both consumers reuse one pair-bias tile.
    """
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    linear_block, _, _ = cute.arch.block_idx()
    n_ctx = q.shape[0]
    heads = q.shape[2]
    sequence_count = q.shape[3]
    batch = q.shape[4]
    query_blocks = n_ctx // QUERY_TILE_SIZE
    sequence_pairs = sequence_count // CONSUMER_WARPGROUP_COUNT
    query_block = linear_block % query_blocks
    sequence_pair_linear = linear_block // query_blocks
    sequence_pair = sequence_pair_linear % sequence_pairs
    batch_head_idx = sequence_pair_linear // sequence_pairs
    batch_idx = batch_head_idx // heads
    head_idx = batch_head_idx % heads
    sequence_0 = sequence_pair * CONSUMER_WARPGROUP_COUNT

    allocator = cutlass.utils.SmemAllocator()
    shared_query = allocator.allocate_tensor(
        cutlass.BFloat16, query_layout.outer, swizzle=query_layout.inner
    )
    shared_key = allocator.allocate_tensor(
        cutlass.BFloat16, key_layout.outer, swizzle=key_layout.inner
    )
    shared_pair_bias = allocator.allocate_tensor(
        cutlass.BFloat16,
        pair_bias_layout.outer,
        swizzle=pair_bias_layout.inner,
    )
    shared_value = allocator.allocate_tensor(
        cutlass.BFloat16, value_layout.outer, swizzle=value_layout.inner
    )
    shared_mask = allocator.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout(
            (KEY_TILE_SIZE, CONSUMER_WARPGROUP_COUNT * KEY_VALUE_STAGES),
            stride=(1, KEY_TILE_SIZE),
        ),
        swizzle=None,
    )
    # Residual mask is physically one [Bc] vector per stage. This zero-stride
    # M mode represents the logical [Mq,Bc] broadcast without replication.
    shared_broadcast_mask = cute.make_tensor(
        shared_mask.iterator,
        cute.make_layout(
            (
                QUERY_TILE_SIZE,
                KEY_TILE_SIZE,
                CONSUMER_WARPGROUP_COUNT * KEY_VALUE_STAGES,
            ),
            stride=(0, 1, KEY_TILE_SIZE),
        ),
    )
    shared_output = allocator.allocate_tensor(
        cutlass.BFloat16,
        output_layout.outer,
        swizzle=output_layout.inner,
    )
    # Once the attention output has been issued to TMA, reinterpret the same
    # 8 KiB allocation as two [C=128, M=16] residual-input tiles.
    shared_residual_input = cute.make_tensor(
        shared_output.iterator,
        residual_io_layout.outer,
    )
    query_ready = allocator.allocate_array(cutlass.Int64, CONSUMER_WARPGROUP_COUNT)
    key_value_ready = allocator.allocate_array(cutlass.Int64, KEY_VALUE_STAGES)
    key_value_free = allocator.allocate_array(cutlass.Int64, KEY_VALUE_STAGES)
    pair_bias_ready = allocator.allocate_array(
        cutlass.Int64, PAIR_BIAS_STAGES
    )
    pair_bias_free = allocator.allocate_array(cutlass.Int64, PAIR_BIAS_STAGES)

    if thread_idx == 0:
        for consumer in cutlass.range_constexpr(CONSUMER_WARPGROUP_COUNT):
            cute.arch.mbarrier_init(query_ready + consumer, 1)
        for stage in cutlass.range_constexpr(KEY_VALUE_STAGES):
            cute.arch.mbarrier_init(key_value_ready + stage, 1)
            # One arrival from each consumer warpgroup releases a stage.
            if n_ctx != SHORT_SEQUENCE_LENGTH:
                cute.arch.mbarrier_init(
                    key_value_free + stage, CONSUMER_WARPGROUP_COUNT
                )
        for slot in cutlass.range_constexpr(PAIR_BIAS_STAGES):
            cute.arch.mbarrier_init(pair_bias_ready + slot, 1)
            # Every consumer warp arrives after copying its PB fragment to
            # registers. Waiting for all eight warps prevents a fast warp from
            # releasing the single shared slot while a sibling still reads it.
            cute.arch.mbarrier_init(
                pair_bias_free + slot,
                CONSUMER_WARPGROUP_COUNT * 4,
            )
        cute.nvgpu.cpasync.prefetch_descriptor(query_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(key_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(value_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(pair_bias_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(residual_mask_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(residual_input_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(local_output_tma)
        for peer in cutlass.range_constexpr(world_size):
            cute.nvgpu.cpasync.prefetch_descriptor(output_tmas[peer])
            cute.nvgpu.cpasync.prefetch_descriptor(residual_output_tmas[peer])
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    query_tiles = cute.zipped_divide(q, (QUERY_TILE_SIZE, HEAD_DIMENSION))
    query_problem_0 = query_tiles[
        None, (None, 0, head_idx, sequence_0, batch_idx)
    ]
    query_problem_1 = query_tiles[
        None, (None, 0, head_idx, sequence_0 + 1, batch_idx)
    ]
    shared_query_partition, global_query_partition_0 = cute.nvgpu.cpasync.tma_partition(
        query_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_query, 0, 2),
        query_problem_0,
    )
    _, global_query_partition_1 = cute.nvgpu.cpasync.tma_partition(
        query_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_query, 0, 2),
        query_problem_1,
    )
    key_tiles = cute.zipped_divide(k, (KEY_TILE_SIZE, HEAD_DIMENSION))
    key_problem_0 = key_tiles[None, (None, 0, head_idx, sequence_0, batch_idx)]
    key_problem_1 = key_tiles[
        None, (None, 0, head_idx, sequence_0 + 1, batch_idx)
    ]
    shared_key_partition, global_key_partition_0 = cute.nvgpu.cpasync.tma_partition(
        key_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_key, 0, 2),
        key_problem_0,
    )
    _, global_key_partition_1 = cute.nvgpu.cpasync.tma_partition(
        key_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_key, 0, 2),
        key_problem_1,
    )
    value_tiles = cute.zipped_divide(v, (HEAD_DIMENSION, KEY_TILE_SIZE))
    value_problem_0 = value_tiles[None, (0, None, head_idx, sequence_0, batch_idx)]
    value_problem_1 = value_tiles[
        None, (0, None, head_idx, sequence_0 + 1, batch_idx)
    ]
    shared_value_partition, global_value_partition_0 = cute.nvgpu.cpasync.tma_partition(
        value_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_value, 0, 2),
        value_problem_0,
    )
    _, global_value_partition_1 = cute.nvgpu.cpasync.tma_partition(
        value_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_value, 0, 2),
        value_problem_1,
    )
    pair_bias_tiles = cute.zipped_divide(
        pair_bias_tensor, (KEY_TILE_SIZE, QUERY_TILE_SIZE)
    )
    pair_bias_problem = pair_bias_tiles[
        None, (None, None, head_idx, batch_idx)
    ]
    shared_pair_bias_partition, global_pair_bias_partition = (
        cute.nvgpu.cpasync.tma_partition(
            pair_bias_tma,
            0,
            cute.make_layout(1),
            cute.group_modes(shared_pair_bias, 0, 2),
            pair_bias_problem,
        )
    )
    residual_mask_tiles = cute.zipped_divide(residual_mask_tensor, (KEY_TILE_SIZE,))
    residual_mask_problem_0 = residual_mask_tiles[
        None, (None, sequence_0, batch_idx)
    ]
    residual_mask_problem_1 = residual_mask_tiles[
        None, (None, sequence_0 + 1, batch_idx)
    ]
    shared_mask_partition, global_mask_partition_0 = cute.nvgpu.cpasync.tma_partition(
        residual_mask_tma,
        0,
        cute.make_layout(1),
        shared_mask,
        residual_mask_problem_0,
    )
    _, global_mask_partition_1 = cute.nvgpu.cpasync.tma_partition(
        residual_mask_tma,
        0,
        cute.make_layout(1),
        shared_mask,
        residual_mask_problem_1,
    )
    num_key_blocks = n_ctx // KEY_TILE_SIZE

    if warp_idx == PRODUCER_WARP_INDEX:
        warpgroup_thread_idx = thread_idx - CONSUMER_THREAD_COUNT

        # One warp owns producer control flow. The CuTe TMA copy atom has a
        # one-thread logical copy layout, so this warp-uniform call does not
        # create 32 independent transfers. Lane 0 also programs expected bytes.
        if warpgroup_thread_idx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                query_ready, QUERY_TILE_SIZE * HEAD_DIMENSION * 2
            )
            cute.arch.mbarrier_arrive_and_expect_tx(
                query_ready + 1, QUERY_TILE_SIZE * HEAD_DIMENSION * 2
            )
        cute.copy(
            query_tma,
            global_query_partition_0[None, query_block],
            shared_query_partition[None, 0],
            tma_bar_ptr=query_ready,
        )
        cute.copy(
            query_tma,
            global_query_partition_1[None, query_block],
            shared_query_partition[None, 1],
            tma_bar_ptr=query_ready + 1,
        )

        key_block = cutlass.Int32(0)
        for _ in cutlass.range(num_key_blocks):
            stage = key_block % KEY_VALUE_STAGES
            if key_block >= KEY_VALUE_STAGES:
                free_phase = ((key_block // KEY_VALUE_STAGES) - 1) % 2
                cute.arch.mbarrier_wait(key_value_free + stage, free_phase)
            pair_bias_stage = key_block % PAIR_BIAS_STAGES
            if key_block >= PAIR_BIAS_STAGES:
                pair_bias_free_phase = (
                    (key_block // PAIR_BIAS_STAGES) - 1
                ) % 2
                cute.arch.mbarrier_wait(
                    pair_bias_free + pair_bias_stage,
                    pair_bias_free_phase,
                )
            if warpgroup_thread_idx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    key_value_ready + stage,
                    CONSUMER_WARPGROUP_COUNT
                    * (
                        2 * KEY_TILE_SIZE * HEAD_DIMENSION * 2
                        + KEY_TILE_SIZE * 2
                    ),
                )
                cute.arch.mbarrier_arrive_and_expect_tx(
                    pair_bias_ready + pair_bias_stage,
                    QUERY_TILE_SIZE * KEY_TILE_SIZE * 2,
                )
            cute.copy(
                key_tma,
                global_key_partition_0[None, key_block],
                shared_key_partition[None, stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            cute.copy(
                key_tma,
                global_key_partition_1[None, key_block],
                shared_key_partition[None, KEY_VALUE_STAGES + stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            cute.copy(
                value_tma,
                global_value_partition_0[None, key_block],
                shared_value_partition[None, stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            cute.copy(
                value_tma,
                global_value_partition_1[None, key_block],
                shared_value_partition[None, KEY_VALUE_STAGES + stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            cute.copy(
                residual_mask_tma,
                global_mask_partition_0[None, key_block],
                shared_mask_partition[None, stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            cute.copy(
                residual_mask_tma,
                global_mask_partition_1[None, key_block],
                shared_mask_partition[None, KEY_VALUE_STAGES + stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            cute.copy(
                pair_bias_tma,
                global_pair_bias_partition[None, key_block, query_block],
                shared_pair_bias_partition[None, pair_bias_stage],
                tma_bar_ptr=pair_bias_ready + pair_bias_stage,
                cache_policy=cutlass.Int64(PAIR_BIAS_EVICT_LAST_POLICY),
            )
            if key_block + KEY_VALUE_STAGES < num_key_blocks:
                cute.prefetch(
                    key_tma,
                    global_key_partition_0[None, key_block + KEY_VALUE_STAGES],
                )
                cute.prefetch(
                    value_tma,
                    global_value_partition_0[None, key_block + KEY_VALUE_STAGES],
                )
                cute.prefetch(
                    key_tma,
                    global_key_partition_1[None, key_block + KEY_VALUE_STAGES],
                )
                cute.prefetch(
                    value_tma,
                    global_value_partition_1[None, key_block + KEY_VALUE_STAGES],
                )
            key_block += 1

    else:
        consumer_warpgroup = thread_idx // THREADS_PER_WARPGROUP
        warpgroup_thread_idx = thread_idx - consumer_warpgroup * THREADS_PER_WARPGROUP
        query_base = query_block * QUERY_TILE_SIZE
        cute.arch.mbarrier_wait(query_ready + consumer_warpgroup, 0)
        qk_thread = score_mma.get_slice(warpgroup_thread_idx)
        query_fragment = score_mma.make_fragment_A(qk_thread.partition_A(shared_query))
        key_fragment = score_mma.make_fragment_B(qk_thread.partition_B(shared_key))
        score_fragment = score_mma.make_fragment_C(
            score_mma.partition_shape_C((QUERY_TILE_SIZE, KEY_TILE_SIZE))
        )
        score_coords = qk_thread.partition_C(
            cute.make_identity_tensor((QUERY_TILE_SIZE, KEY_TILE_SIZE))
        )
        # PB is staged in transposed MN-major form.  Reversing the two logical
        # axes here recovers [row,col], while partition_C gives the same vector
        # ownership as the score accumulator and avoids scalar bank-conflicted
        # LDS instructions.
        shared_pair_bias_row_col = cute.make_tensor(
            shared_pair_bias.iterator,
            cute.select(shared_pair_bias.layout, mode=[1, 0, 2]),
        )
        pair_bias_fragment = score_mma.make_fragment_C(
            score_mma.partition_shape_C((QUERY_TILE_SIZE, KEY_TILE_SIZE))
        )

        pv_thread = output_mma.get_slice(warpgroup_thread_idx)
        probability_fragment = output_mma.make_fragment_A(
            output_mma.partition_shape_A((QUERY_TILE_SIZE, KEY_TILE_SIZE))
        )
        value_fragment = output_mma.make_fragment_B(pv_thread.partition_B(shared_value))
        output_accumulator = output_mma.make_fragment_C(
            output_mma.partition_shape_C((QUERY_TILE_SIZE, HEAD_DIMENSION))
        )
        output_accumulator.fill(0.0)
        shared_output_mn = cute.make_tensor(
            shared_output.iterator, cute.select(shared_output.layout, mode=[1, 0])
        )
        shared_output_consumer = cute.local_tile(
            shared_output_mn,
            (QUERY_TILE_SIZE, HEAD_DIMENSION),
            (consumer_warpgroup, 0),
        )
        thread_copy_c = output_register_copy.get_slice(warpgroup_thread_idx)
        output_partition = thread_copy_c.partition_D(shared_output_consumer)

        first_row, _ = score_coords[0]
        second_row, _ = score_coords[2]
        running_max_0 = cutlass.Float32(float("-inf"))
        running_max_1 = cutlass.Float32(float("-inf"))
        running_sum_0 = cutlass.Float32(0.0)
        running_sum_1 = cutlass.Float32(0.0)
        qk0_to_qk1 = NamedBarrier(
            barrier_id=QK0_TO_QK1_BARRIER_ID,
            num_threads=CONSUMER_THREAD_COUNT,
        )
        qk1_to_pv0 = NamedBarrier(
            barrier_id=QK1_TO_PV0_BARRIER_ID,
            num_threads=CONSUMER_THREAD_COUNT,
        )

        key_block = cutlass.Int32(0)
        for _ in cutlass.range(num_key_blocks):
            stage = key_block % KEY_VALUE_STAGES
            key_value_slot = consumer_warpgroup * KEY_VALUE_STAGES + stage
            full_phase = (key_block // KEY_VALUE_STAGES) % 2
            pair_bias_stage = key_block % PAIR_BIAS_STAGES
            pair_bias_phase = (key_block // PAIR_BIAS_STAGES) % 2

            cute.arch.mbarrier_wait(key_value_ready + stage, full_phase)
            # Diagonal FA3 ping-pong. WG1's QK(k) follows WG0's QK(k).
            # Later, WG1's PV(k) waits for WG0's QK(k+1), allowing the latter
            # to overlap WG1's softmax(k).
            # Delay only the first QK; the producer independently stages the
            # pair-bias tile shared by both consumers.
            if (
                n_ctx != SHORT_SEQUENCE_LENGTH
                and consumer_warpgroup == 1
                and key_block == 0
            ):
                qk0_to_qk1.arrive_and_wait()
            score_mma.set(Field.ACCUMULATE, False)
            cute.nvgpu.warpgroup.fence()
            cute.gemm(
                score_mma,
                score_fragment,
                query_fragment[None, None, None, consumer_warpgroup],
                key_fragment[None, None, None, key_value_slot],
                score_fragment,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            if n_ctx != SHORT_SEQUENCE_LENGTH:
                if consumer_warpgroup == 0:
                    qk0_to_qk1.arrive()
                else:
                    qk1_to_pv0.arrive()
            cute.arch.mbarrier_wait(
                pair_bias_ready + pair_bias_stage, pair_bias_phase
            )
            pair_bias_partition = qk_thread.partition_C(
                shared_pair_bias_row_col[None, None, pair_bias_stage]
            )
            pair_bias_fragment.store(pair_bias_partition.load())
            if warpgroup_thread_idx % 32 == 0:
                cute.arch.mbarrier_arrive(pair_bias_free + pair_bias_stage)

            local_max_0 = cutlass.Float32(float("-inf"))
            local_max_1 = cutlass.Float32(float("-inf"))
            # e and e+2 own the same key column for the two query rows in a
            # four-lane WGMMA quartet.  Load the zero-stride broadcast mask
            # once per column and fold the additive work into the max pass.
            for group in cutlass.range_constexpr(cute.size(score_fragment) // 4):
                elem_0 = group * 4
                elem_1 = elem_0 + 1
                elem_2 = elem_0 + 2
                elem_3 = elem_0 + 3
                _, col_0 = score_coords[elem_0]
                _, col_1 = score_coords[elem_1]
                mask_0 = (
                    cutlass.Float32(
                        shared_broadcast_mask[0, col_0, key_value_slot]
                    )
                    - 1.0
                ) * 1.0e9
                mask_1 = (
                    cutlass.Float32(
                        shared_broadcast_mask[0, col_1, key_value_slot]
                    )
                    - 1.0
                ) * 1.0e9
                score_0 = (
                    score_fragment[elem_0] * SOFTMAX_SCALE
                    + pair_bias_fragment[elem_0]
                    + mask_0
                )
                score_1 = (
                    score_fragment[elem_1] * SOFTMAX_SCALE
                    + pair_bias_fragment[elem_1]
                    + mask_1
                )
                score_2 = (
                    score_fragment[elem_2] * SOFTMAX_SCALE
                    + pair_bias_fragment[elem_2]
                    + mask_0
                )
                score_3 = (
                    score_fragment[elem_3] * SOFTMAX_SCALE
                    + pair_bias_fragment[elem_3]
                    + mask_1
                )
                score_fragment[elem_0] = score_0
                score_fragment[elem_1] = score_1
                score_fragment[elem_2] = score_2
                score_fragment[elem_3] = score_3
                local_max_0 = cutlass.max(local_max_0, score_0)
                local_max_0 = cutlass.max(local_max_0, score_1)
                local_max_1 = cutlass.max(local_max_1, score_2)
                local_max_1 = cutlass.max(local_max_1, score_3)
            local_max_0 = cutlass.max(
                local_max_0, cute.arch.shuffle_sync_bfly(local_max_0, offset=1)
            )
            local_max_0 = cutlass.max(
                local_max_0, cute.arch.shuffle_sync_bfly(local_max_0, offset=2)
            )
            local_max_1 = cutlass.max(
                local_max_1, cute.arch.shuffle_sync_bfly(local_max_1, offset=1)
            )
            local_max_1 = cutlass.max(
                local_max_1, cute.arch.shuffle_sync_bfly(local_max_1, offset=2)
            )

            new_max_0 = cutlass.max(running_max_0, local_max_0)
            new_max_1 = cutlass.max(running_max_1, local_max_1)
            alpha_0 = cutlass.Float32(0.0)
            alpha_1 = cutlass.Float32(0.0)
            if warpgroup_thread_idx % 4 == 0:
                alpha_0 = cute.math.exp(running_max_0 - new_max_0, fastmath=True)
                alpha_1 = cute.math.exp(running_max_1 - new_max_1, fastmath=True)
            alpha_0 = cute.arch.shuffle_sync(alpha_0, 0, mask_and_clamp=0x1C03)
            alpha_1 = cute.arch.shuffle_sync(alpha_1, 0, mask_and_clamp=0x1C03)
            local_sum_0 = cutlass.Float32(0.0)
            local_sum_1 = cutlass.Float32(0.0)
            for elem in cutlass.range_constexpr(cute.size(score_fragment)):
                probability = cutlass.Float32(0.0)
                if elem % 4 < 2:
                    exponent = score_fragment[elem] - new_max_0
                    probability = cute.math.exp(exponent, fastmath=True)
                    local_sum_0 += probability
                else:
                    exponent = score_fragment[elem] - new_max_1
                    probability = cute.math.exp(exponent, fastmath=True)
                    local_sum_1 += probability
                score_fragment[elem] = probability

            local_sum_0 += cute.arch.shuffle_sync_bfly(local_sum_0, offset=1)
            local_sum_0 += cute.arch.shuffle_sync_bfly(local_sum_0, offset=2)
            local_sum_1 += cute.arch.shuffle_sync_bfly(local_sum_1, offset=1)
            local_sum_1 += cute.arch.shuffle_sync_bfly(local_sum_1, offset=2)
            running_sum_0 = running_sum_0 * alpha_0 + local_sum_0
            running_sum_1 = running_sum_1 * alpha_1 + local_sum_1
            running_max_0 = new_max_0
            running_max_1 = new_max_1
            probability_fragment.store(score_fragment.load().to(cutlass.BFloat16))

            if n_ctx != SHORT_SEQUENCE_LENGTH:
                if consumer_warpgroup == 0:
                    qk1_to_pv0.arrive_and_wait()
                else:
                    if key_block + 1 < num_key_blocks:
                        qk0_to_qk1.arrive_and_wait()
            for elem in cutlass.range_constexpr(cute.size(output_accumulator)):
                output_accumulator[elem] *= alpha_0 if elem % 4 < 2 else alpha_1

            output_mma.set(Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.fence()
            cute.gemm(
                output_mma,
                output_accumulator,
                probability_fragment,
                value_fragment[None, None, None, key_value_slot],
                output_accumulator,
            )
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            if n_ctx != SHORT_SEQUENCE_LENGTH:
                if warpgroup_thread_idx == 0:
                    cute.arch.mbarrier_arrive(key_value_free + stage)
            key_block += 1

        inv_sum_0 = cutlass.Float32(0.0)
        inv_sum_1 = cutlass.Float32(0.0)
        if warpgroup_thread_idx % 4 == 0:
            inv_sum_0 = cute.arch.rcp_approx(running_sum_0)
            inv_sum_1 = cute.arch.rcp_approx(running_sum_1)
        inv_sum_0 = cute.arch.shuffle_sync(inv_sum_0, 0, mask_and_clamp=0x1C03)
        inv_sum_1 = cute.arch.shuffle_sync(inv_sum_1, 0, mask_and_clamp=0x1C03)
        for elem in cutlass.range_constexpr(cute.size(output_accumulator)):
            output_accumulator[elem] *= inv_sum_0 if elem % 4 < 2 else inv_sum_1
        output_bf16 = cute.make_fragment_like(output_accumulator, cutlass.BFloat16)
        output_bf16.store(output_accumulator.load().to(cutlass.BFloat16))
        output_retile = output_register_copy.retile(output_bf16)
        cute.copy(output_register_copy, output_retile, output_partition)
        if warpgroup_thread_idx % 4 == 0:
            sequence_idx = sequence_0 + consumer_warpgroup
            lse[
                query_base + first_row, head_idx, sequence_idx, batch_idx
            ] = running_max_0 + cute.math.log(running_sum_0, fastmath=True)
            lse[
                query_base + second_row, head_idx, sequence_idx, batch_idx
            ] = running_max_1 + cute.math.log(running_sum_1, fastmath=True)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(
            barrier_id=OUTPUT_READY_BARRIER_ID,
            number_of_threads=CONSUMER_THREAD_COUNT,
        )
        if consumer_warpgroup == 0 and warpgroup_thread_idx < 32:
            remote_store_rows = REMOTE_STORE_ROWS
            _store_remote_output_chunk(
                local_output_tma,
                local_output_tensor,
                shared_output,
                0,
                QUERY_TILE_SIZE,
                query_block,
                head_idx,
                sequence_0,
                batch_idx,
            )
            for chunk in cutlass.range_constexpr(
                QUERY_TILE_SIZE // remote_store_rows
            ):
                global_query = (
                    query_block * QUERY_TILE_SIZE + chunk * remote_store_rows
                )
                destination = global_query // sequence_count
                local_query = global_query - destination * sequence_count
                local_query_block = local_query // remote_store_rows
                for peer in cutlass.range_constexpr(world_size):
                    if destination == peer:
                        _store_remote_output_chunk(
                            output_tmas[peer],
                            output_tensors[peer],
                            shared_output,
                            chunk,
                            remote_store_rows,
                            local_query_block,
                            head_idx,
                            sequence_0,
                            batch_idx,
                        )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)
            if head_idx == 0:
                for chunk in cutlass.range_constexpr(
                    QUERY_TILE_SIZE // remote_store_rows
                ):
                    global_query_block = (
                        query_block * QUERY_TILE_SIZE
                        + chunk * remote_store_rows
                    ) // remote_store_rows
                    destination = (
                        query_block * QUERY_TILE_SIZE
                        + chunk * remote_store_rows
                    ) // sequence_count
                    local_query = (
                        query_block * QUERY_TILE_SIZE
                        + chunk * remote_store_rows
                        - destination * sequence_count
                    )
                    local_query_block = local_query // remote_store_rows
                    if warpgroup_thread_idx == 0:
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            query_ready,
                            2
                            * RESIDUAL_DIMENSION
                            * remote_store_rows
                            * 2,
                        )
                    _load_residual_chunk(
                        residual_input_tma,
                        residual_input_tensor,
                        shared_residual_input,
                        global_query_block,
                        sequence_0,
                        batch_idx,
                        query_ready,
                    )
                    cute.arch.mbarrier_wait(query_ready, (chunk + 1) % 2)
                    for peer in cutlass.range_constexpr(world_size):
                        if destination == peer:
                            _store_remote_residual_chunk(
                                residual_output_tmas[peer],
                                residual_output_tensors[peer],
                                shared_residual_input,
                                local_query_block,
                                sequence_0,
                                batch_idx,
                            )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0)
            if warpgroup_thread_idx == 0:
                cute.arch.fence_proxy("async.global")
                completed_before = cute.arch.atomic_add(
                    completion_counter.iterator,
                    cutlass.Int32(1),
                    sem="acq_rel",
                    scope="gpu",
                )
                total_ctas = query_blocks * sequence_pairs * batch * heads
                if completed_before == completion_epoch * total_ctas - 1:
                    cute.arch.fence_acq_rel_sys()
                    for peer in cutlass.range_constexpr(world_size):
                        cute.arch.store(
                            completion_peers[peer].iterator,
                            completion_epoch,
                            sem="release",
                            scope="sys",
                        )
                    for source in cutlass.range_constexpr(world_size):
                        signal = cute.domain_offset((source,), completion)
                        observed = cute.arch.load(
                            signal.iterator,
                            cutlass.Int32,
                            sem="acquire",
                            scope="sys",
                        )
                        while observed < completion_epoch:
                            observed = cute.arch.load(
                                signal.iterator,
                                cutlass.Int32,
                                sem="acquire",
                                scope="sys",
                            )


@cute.jit
def _build_and_launch_forward(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    pair_bias: cute.Tensor,
    residual_mask: cute.Tensor,
    residual_input: cute.Tensor,
    local_output: cute.Tensor,
    out_peer_0: cute.Tensor,
    out_peer_1: cute.Tensor,
    out_peer_2: cute.Tensor,
    out_peer_3: cute.Tensor,
    out_peer_4: cute.Tensor,
    out_peer_5: cute.Tensor,
    out_peer_6: cute.Tensor,
    out_peer_7: cute.Tensor,
    residual_out_peer_0: cute.Tensor,
    residual_out_peer_1: cute.Tensor,
    residual_out_peer_2: cute.Tensor,
    residual_out_peer_3: cute.Tensor,
    residual_out_peer_4: cute.Tensor,
    residual_out_peer_5: cute.Tensor,
    residual_out_peer_6: cute.Tensor,
    residual_out_peer_7: cute.Tensor,
    completion_peer_0: cute.Tensor,
    completion_peer_1: cute.Tensor,
    completion_peer_2: cute.Tensor,
    completion_peer_3: cute.Tensor,
    completion_peer_4: cute.Tensor,
    completion_peer_5: cute.Tensor,
    completion_peer_6: cute.Tensor,
    completion_peer_7: cute.Tensor,
    completion: cute.Tensor,
    completion_counter: cute.Tensor,
    completion_epoch: cutlass.Int32,
    lse: cute.Tensor,
    world_size: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    batch = q.shape[0]
    sequence_count = q.shape[1]
    n_ctx = q.shape[2]
    heads = q.shape[3]
    problem_count = batch * sequence_count * heads

    score_mma = sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        OperandMajorMode.K,
        OperandMajorMode.K,
        cutlass.Float32,
        atom_layout_mnk=(1, 1, 1),
        tiler_mn=(QUERY_TILE_SIZE, KEY_TILE_SIZE),
        a_source=OperandSource.SMEM,
    )
    output_mma = sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        OperandMajorMode.K,
        OperandMajorMode.MN,
        cutlass.Float32,
        atom_layout_mnk=(1, 1, 1),
        tiler_mn=(QUERY_TILE_SIZE, HEAD_DIMENSION),
        a_source=OperandSource.RMEM,
    )
    output_store_atom = cute.make_copy_atom(
        cute.nvgpu.warp.StMatrix8x8x16bOp(
            transpose=False,
            num_matrices=4,
        ),
        cutlass.BFloat16,
    )
    output_register_copy = cute.make_tiled_copy_C(output_store_atom, output_mma)
    query_layout = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        (QUERY_TILE_SIZE, QUERY_TILE_SIZE, HEAD_DIMENSION),
        a_dtype=cutlass.BFloat16,
        num_stages=2,
    )
    key_layout = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        (QUERY_TILE_SIZE, KEY_TILE_SIZE, HEAD_DIMENSION),
        b_dtype=cutlass.BFloat16,
        num_stages=CONSUMER_WARPGROUP_COUNT * KEY_VALUE_STAGES,
    )
    value_layout = sm90_utils.make_smem_layout_b(
        LayoutEnum.COL_MAJOR,
        (QUERY_TILE_SIZE, HEAD_DIMENSION, KEY_TILE_SIZE),
        b_dtype=cutlass.BFloat16,
        num_stages=CONSUMER_WARPGROUP_COUNT * KEY_VALUE_STAGES,
    )

    query_view = cute.make_tensor(
        q.iterator, cute.select(q.layout, mode=[2, 4, 3, 1, 0])
    )
    key_view = cute.make_tensor(
        k.iterator, cute.select(k.layout, mode=[2, 4, 3, 1, 0])
    )
    value_view = cute.make_tensor(
        v.iterator, cute.select(v.layout, mode=[4, 2, 3, 1, 0])
    )
    pair_bias_view = cute.make_tensor(
        pair_bias.iterator,
        cute.select(pair_bias.layout, mode=[4, 3, 2, 0]),
    )
    residual_mask_view = cute.make_tensor(
        residual_mask.iterator,
        cute.select(residual_mask.layout, mode=[4, 1, 0]),
    )
    residual_input_view = cute.make_tensor(
        residual_input.iterator,
        cute.select(residual_input.layout, mode=[3, 2, 1, 0]),
    )
    local_output_view = cute.make_tensor(
        local_output.iterator,
        cute.select(local_output.layout, mode=[4, 2, 3, 1, 0]),
    )
    output_view_0 = cute.make_tensor(
        out_peer_0.iterator,
        cute.select(out_peer_0.layout, mode=[4, 2, 3, 1, 0]),
    )
    output_view_1 = cute.make_tensor(
        out_peer_1.iterator,
        cute.select(out_peer_1.layout, mode=[4, 2, 3, 1, 0]),
    )
    output_view_2 = cute.make_tensor(
        out_peer_2.iterator,
        cute.select(out_peer_2.layout, mode=[4, 2, 3, 1, 0]),
    )
    output_view_3 = cute.make_tensor(
        out_peer_3.iterator,
        cute.select(out_peer_3.layout, mode=[4, 2, 3, 1, 0]),
    )
    output_view_4 = cute.make_tensor(
        out_peer_4.iterator,
        cute.select(out_peer_4.layout, mode=[4, 2, 3, 1, 0]),
    )
    output_view_5 = cute.make_tensor(
        out_peer_5.iterator,
        cute.select(out_peer_5.layout, mode=[4, 2, 3, 1, 0]),
    )
    output_view_6 = cute.make_tensor(
        out_peer_6.iterator,
        cute.select(out_peer_6.layout, mode=[4, 2, 3, 1, 0]),
    )
    output_view_7 = cute.make_tensor(
        out_peer_7.iterator,
        cute.select(out_peer_7.layout, mode=[4, 2, 3, 1, 0]),
    )
    residual_output_view_0 = cute.make_tensor(
        residual_out_peer_0.iterator,
        cute.select(residual_out_peer_0.layout, mode=[3, 2, 1, 0]),
    )
    residual_output_view_1 = cute.make_tensor(
        residual_out_peer_1.iterator,
        cute.select(residual_out_peer_1.layout, mode=[3, 2, 1, 0]),
    )
    residual_output_view_2 = cute.make_tensor(
        residual_out_peer_2.iterator,
        cute.select(residual_out_peer_2.layout, mode=[3, 2, 1, 0]),
    )
    residual_output_view_3 = cute.make_tensor(
        residual_out_peer_3.iterator,
        cute.select(residual_out_peer_3.layout, mode=[3, 2, 1, 0]),
    )
    residual_output_view_4 = cute.make_tensor(
        residual_out_peer_4.iterator,
        cute.select(residual_out_peer_4.layout, mode=[3, 2, 1, 0]),
    )
    residual_output_view_5 = cute.make_tensor(
        residual_out_peer_5.iterator,
        cute.select(residual_out_peer_5.layout, mode=[3, 2, 1, 0]),
    )
    residual_output_view_6 = cute.make_tensor(
        residual_out_peer_6.iterator,
        cute.select(residual_out_peer_6.layout, mode=[3, 2, 1, 0]),
    )
    residual_output_view_7 = cute.make_tensor(
        residual_out_peer_7.iterator,
        cute.select(residual_out_peer_7.layout, mode=[3, 2, 1, 0]),
    )
    lse_view = cute.make_tensor(
        lse.iterator, cute.select(lse.layout, mode=[3, 2, 1, 0])
    )
    load_tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    query_tma_layout = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        (QUERY_TILE_SIZE, QUERY_TILE_SIZE, HEAD_DIMENSION),
        a_dtype=cutlass.BFloat16,
        num_stages=1,
    )
    key_tma_layout = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        (QUERY_TILE_SIZE, KEY_TILE_SIZE, HEAD_DIMENSION),
        b_dtype=cutlass.BFloat16,
        num_stages=1,
    )
    value_tma_layout = sm90_utils.make_smem_layout_b(
        LayoutEnum.COL_MAJOR,
        (QUERY_TILE_SIZE, HEAD_DIMENSION, KEY_TILE_SIZE),
        b_dtype=cutlass.BFloat16,
        num_stages=1,
    )
    pair_bias_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        SmemLayoutAtomKind.MN_SW128,
        cutlass.BFloat16,
    )
    pair_bias_layout = cute.tile_to_shape(
        pair_bias_layout_atom,
        (
            KEY_TILE_SIZE,
            QUERY_TILE_SIZE,
            PAIR_BIAS_STAGES,
        ),
        order=(1, 0, 2),
    )
    pair_bias_tma_layout = cute.tile_to_shape(
        pair_bias_layout_atom,
        (KEY_TILE_SIZE, QUERY_TILE_SIZE, 1),
        order=(1, 0, 2),
    )
    residual_mask_tma_layout = cute.make_layout(KEY_TILE_SIZE, stride=1)
    output_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        SmemLayoutAtomKind.MN_SW64, cutlass.BFloat16
    )
    output_layout = cute.tile_to_shape(
        output_layout_atom,
        (HEAD_DIMENSION, CTA_OUTPUT_ROW_SLOTS),
        order=(1, 0),
    )
    output_tma_layout = cute.tile_to_shape(
        output_layout_atom,
        (HEAD_DIMENSION, REMOTE_STORE_ROWS),
        order=(1, 0),
    )
    residual_io_layout_atom = output_layout_atom
    residual_io_layout = cute.tile_to_shape(
        residual_io_layout_atom,
        (
            RESIDUAL_DIMENSION,
            CONSUMER_WARPGROUP_COUNT * REMOTE_STORE_ROWS,
        ),
        order=(1, 0),
    )
    residual_io_tma_layout = cute.tile_to_shape(
        residual_io_layout_atom,
        (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS),
        order=(1, 0),
    )
    local_output_tma_layout = cute.tile_to_shape(
        output_layout_atom,
        (HEAD_DIMENSION, QUERY_TILE_SIZE),
        order=(1, 0),
    )
    query_tma, query_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        load_tma_op,
        query_view,
        query_tma_layout,
        (QUERY_TILE_SIZE, HEAD_DIMENSION),
    )
    key_tma, key_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        load_tma_op,
        key_view,
        key_tma_layout,
        (KEY_TILE_SIZE, HEAD_DIMENSION),
    )
    value_tma, value_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        load_tma_op,
        value_view,
        value_tma_layout,
        (HEAD_DIMENSION, KEY_TILE_SIZE),
    )
    pair_bias_tma, pair_bias_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        load_tma_op,
        pair_bias_view,
        pair_bias_tma_layout,
        (KEY_TILE_SIZE, QUERY_TILE_SIZE),
    )
    residual_mask_tma, residual_mask_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        load_tma_op,
        residual_mask_view,
        residual_mask_tma_layout,
        (KEY_TILE_SIZE,),
    )
    residual_input_tma, residual_input_tensor = (
        cute.nvgpu.cpasync.make_tiled_tma_atom(
            load_tma_op,
            residual_input_view,
            residual_io_tma_layout,
            (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS),
        )
    )
    local_output_tma, local_output_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        local_output_view,
        local_output_tma_layout,
        (HEAD_DIMENSION, QUERY_TILE_SIZE),
    )
    output_tma_0, output_tensor_0 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        output_view_0,
        output_tma_layout,
        (HEAD_DIMENSION, REMOTE_STORE_ROWS),
    )
    output_tma_1, output_tensor_1 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), output_view_1,
        output_tma_layout, (HEAD_DIMENSION, REMOTE_STORE_ROWS)
    )
    output_tma_2, output_tensor_2 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), output_view_2,
        output_tma_layout, (HEAD_DIMENSION, REMOTE_STORE_ROWS)
    )
    output_tma_3, output_tensor_3 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), output_view_3,
        output_tma_layout, (HEAD_DIMENSION, REMOTE_STORE_ROWS)
    )
    output_tma_4, output_tensor_4 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), output_view_4,
        output_tma_layout, (HEAD_DIMENSION, REMOTE_STORE_ROWS)
    )
    output_tma_5, output_tensor_5 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), output_view_5,
        output_tma_layout, (HEAD_DIMENSION, REMOTE_STORE_ROWS)
    )
    output_tma_6, output_tensor_6 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), output_view_6,
        output_tma_layout, (HEAD_DIMENSION, REMOTE_STORE_ROWS)
    )
    output_tma_7, output_tensor_7 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), output_view_7,
        output_tma_layout, (HEAD_DIMENSION, REMOTE_STORE_ROWS)
    )
    residual_output_tma_0, residual_output_tensor_0 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), residual_output_view_0,
        residual_io_tma_layout, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    residual_output_tma_1, residual_output_tensor_1 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), residual_output_view_1,
        residual_io_tma_layout, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    residual_output_tma_2, residual_output_tensor_2 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), residual_output_view_2,
        residual_io_tma_layout, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    residual_output_tma_3, residual_output_tensor_3 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), residual_output_view_3,
        residual_io_tma_layout, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    residual_output_tma_4, residual_output_tensor_4 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), residual_output_view_4,
        residual_io_tma_layout, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    residual_output_tma_5, residual_output_tensor_5 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), residual_output_view_5,
        residual_io_tma_layout, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    residual_output_tma_6, residual_output_tensor_6 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), residual_output_view_6,
        residual_io_tma_layout, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    residual_output_tma_7, residual_output_tensor_7 = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(), residual_output_view_7,
        residual_io_tma_layout, (RESIDUAL_DIMENSION, REMOTE_STORE_ROWS)
    )
    output_tmas = (
        output_tma_0, output_tma_1, output_tma_2, output_tma_3,
        output_tma_4, output_tma_5, output_tma_6, output_tma_7,
    )
    output_tensors = (
        output_tensor_0, output_tensor_1, output_tensor_2, output_tensor_3,
        output_tensor_4, output_tensor_5, output_tensor_6, output_tensor_7,
    )
    residual_output_tmas = (
        residual_output_tma_0, residual_output_tma_1,
        residual_output_tma_2, residual_output_tma_3,
        residual_output_tma_4, residual_output_tma_5,
        residual_output_tma_6, residual_output_tma_7,
    )
    residual_output_tensors = (
        residual_output_tensor_0, residual_output_tensor_1,
        residual_output_tensor_2, residual_output_tensor_3,
        residual_output_tensor_4, residual_output_tensor_5,
        residual_output_tensor_6, residual_output_tensor_7,
    )
    completion_peers = (
        completion_peer_0,
        completion_peer_1,
        completion_peer_2,
        completion_peer_3,
        completion_peer_4,
        completion_peer_5,
        completion_peer_6,
        completion_peer_7,
    )

    _evoattention_forward_kernel(
        query_tensor,
        key_tensor,
        value_tensor,
        query_tma,
        key_tma,
        value_tma,
        pair_bias_tma,
        pair_bias_tensor,
        residual_mask_tma,
        residual_mask_tensor,
        residual_input_tma,
        residual_input_tensor,
        local_output_tma,
        local_output_tensor,
        output_tmas,
        output_tensors,
        residual_output_tmas,
        residual_output_tensors,
        completion_peers,
        completion,
        completion_counter,
        completion_epoch,
        lse_view,
        query_layout,
        key_layout,
        value_layout,
        pair_bias_layout,
        output_layout,
        residual_io_layout,
        score_mma,
        output_mma,
        output_register_copy,
        world_size,
    ).launch(
        # Flatten the work grid into x: grid.y is limited to 65535, while the
        # largest target has P=B*S*H=65536 before accounting for query tiles.
        grid=(
            (n_ctx // QUERY_TILE_SIZE)
            * (problem_count // CONSUMER_WARPGROUP_COUNT),
            1,
            1,
        ),
        block=(THREADS_PER_BLOCK, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
    )


class DistributedEvoAttentionForward:
    """Compile one distributed forward specialization for a fixed world size."""

    def __init__(self, world_size: int, source_rank: int) -> None:
        if world_size not in (2, 4, 8):
            raise ValueError(f"world_size must be 2, 4, or 8, got {world_size}")
        self.world_size = world_size
        if not 0 <= source_rank < world_size:
            raise ValueError(
                f"source_rank must be in [0, {world_size}), got {source_rank}"
            )
        self.source_rank = source_rank
        self._compiled = None
        self._compile_count = 0
        self._device_index: int | None = None
        self._lock = threading.Lock()

    @property
    def compile_count(self) -> int:
        return self._compile_count

    def _validate_arguments(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_bias: torch.Tensor,
        residual_mask: torch.Tensor,
        residual_input: torch.Tensor,
        local_output: torch.Tensor,
        out_peers: tuple[torch.Tensor, ...],
        residual_out_peers: tuple[torch.Tensor, ...],
        completion_peers: tuple[torch.Tensor, ...],
        completion: torch.Tensor,
        completion_counter: torch.Tensor,
        completion_epoch: int,
        lse: torch.Tensor,
    ) -> None:
        if q.ndim != 5:
            raise ValueError(
                f"q must have shape [B, S, N, H, 32], got {tuple(q.shape)}"
            )
        batch, sequence_count, n_ctx, heads, dim = q.shape
        if n_ctx != self.world_size * sequence_count:
            raise ValueError(
                "distributed forward requires N=world_size*S, got "
                f"N={n_ctx}, world_size={self.world_size}, S={sequence_count}"
            )
        if dim != HEAD_DIMENSION:
            raise ValueError(f"head dimension must be {HEAD_DIMENSION}, got {dim}")
        if n_ctx % QUERY_TILE_SIZE:
            raise ValueError(
                f"N must be divisible by {QUERY_TILE_SIZE}, got {n_ctx}"
            )
        if sequence_count % CONSUMER_WARPGROUP_COUNT:
            raise ValueError("S must be even so the two consumers can pair sequences")
        if k.shape != q.shape or v.shape != q.shape:
            raise ValueError("q, k, and v must have identical shapes")
        if local_output.shape != q.shape:
            raise ValueError("local_output must have the same shape as q")
        expected_residual_input_shape = (
            batch,
            sequence_count,
            n_ctx,
            RESIDUAL_DIMENSION,
        )
        if tuple(residual_input.shape) != expected_residual_input_shape:
            raise ValueError(
                "residual_input must have shape [B,S,N,128] = "
                f"{expected_residual_input_shape}"
            )
        if len(out_peers) != self.world_size:
            raise ValueError(
                f"expected {self.world_size} peer output views, got {len(out_peers)}"
            )
        expected_output_shape = (
            batch,
            sequence_count,
            sequence_count,
            heads,
            HEAD_DIMENSION,
        )
        if any(tuple(out.shape) != expected_output_shape for out in out_peers):
            raise ValueError(
                "each peer output must have shape [B,S,S,H,D] = "
                f"{expected_output_shape}"
            )
        if len(residual_out_peers) != self.world_size:
            raise ValueError(
                f"expected {self.world_size} residual-output peer views, "
                f"got {len(residual_out_peers)}"
            )
        expected_residual_output_shape = (
            batch,
            sequence_count,
            sequence_count,
            RESIDUAL_DIMENSION,
        )
        if any(
            tuple(out.shape) != expected_residual_output_shape
            for out in residual_out_peers
        ):
            raise ValueError(
                "each residual-output peer must have shape [B,S,S,128] = "
                f"{expected_residual_output_shape}"
            )
        if len(completion_peers) != self.world_size:
            raise ValueError(
                f"expected {self.world_size} completion peers, "
                f"got {len(completion_peers)}"
            )
        if any(
            tuple(completion.shape) != (1,)
            for completion in completion_peers
        ):
            raise ValueError("completion peers must each have shape [1]")
        if tuple(completion.shape) != (self.world_size,):
            raise ValueError(
                f"completion must have shape [{self.world_size}], "
                f"got {tuple(completion.shape)}"
            )
        if tuple(completion_counter.shape) != (1,):
            raise ValueError("completion_counter must have shape [1]")
        if not 0 < completion_epoch < 2**31:
            raise ValueError("completion_epoch must be a positive int32 value")
        if tuple(pair_bias.shape) != (batch, 1, heads, n_ctx, n_ctx):
            raise ValueError(
                "pair_bias must have shape [B, 1, H, N, N], got "
                f"{tuple(pair_bias.shape)}"
            )
        if tuple(residual_mask.shape) != (batch, sequence_count, 1, 1, n_ctx):
            raise ValueError(
                "residual_mask must have shape [B, S, 1, 1, N], got "
                f"{tuple(residual_mask.shape)}"
            )
        if tuple(lse.shape) != (batch, sequence_count, heads, n_ctx):
            raise ValueError(
                "lse must have shape [B, S, H, N], got " f"{tuple(lse.shape)}"
            )

        tensors = (
            q,
            k,
            v,
            pair_bias,
            residual_mask,
            residual_input,
            local_output,
            *out_peers,
            *residual_out_peers,
            *completion_peers,
            completion,
            completion_counter,
            lse,
        )
        if any(t.device != q.device for t in tensors):
            raise ValueError("all tensors must be on the same CUDA device")
        if q.device.type != "cuda":
            raise ValueError("EvoAttention requires CUDA tensors")
        compact_tensors = (q, residual_mask, residual_input, local_output, lse)
        if any(not tensor.is_contiguous() for tensor in compact_tensors):
            raise ValueError("Q/mask/LSE tensors must be contiguous")
        if any(
            tensor.stride(-1) != 1
            or any(stride < 0 for stride in tensor.stride())
            for tensor in (k, v)
        ):
            raise ValueError("K/V must have positive strides and a contiguous D axis")
        if pair_bias.stride(-1) != 1 or any(stride < 0 for stride in pair_bias.stride()):
            raise ValueError("pair_bias must have a positive-stride contiguous K axis")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise TypeError("q, k and v must be bfloat16")
        if local_output.dtype != torch.bfloat16:
            raise TypeError("local_output must be bfloat16")
        if any(out.dtype != torch.bfloat16 for out in out_peers):
            raise TypeError("peer outputs must be bfloat16")
        if any(out.stride(-1) != 1 for out in out_peers):
            raise ValueError("peer outputs must have a contiguous head-dimension axis")
        if residual_input.dtype != torch.bfloat16:
            raise TypeError("residual_input must be bfloat16")
        if any(out.dtype != torch.bfloat16 for out in residual_out_peers):
            raise TypeError("residual-output peers must be bfloat16")
        if any(out.stride(-1) != 1 for out in residual_out_peers):
            raise ValueError("residual-output peers must have a contiguous C axis")
        if any(completion.dtype != torch.int32 for completion in completion_peers):
            raise TypeError("completion peers must be int32")
        if any(not completion.is_contiguous() for completion in completion_peers):
            raise ValueError("completion peers must be contiguous")
        if completion_counter.dtype != torch.int32:
            raise TypeError("completion_counter must be int32")
        if not completion_counter.is_contiguous():
            raise ValueError("completion_counter must be contiguous")
        if pair_bias.dtype != torch.bfloat16 or residual_mask.dtype != torch.bfloat16:
            raise TypeError("pair_bias and residual_mask must be bfloat16")
        if lse.dtype != torch.float32:
            raise TypeError("lse must be float32")

    def _make_compile_arguments(
        self,
        arguments: tuple[object, ...],
        stream: cuda.CUstream,
    ) -> tuple[object, ...]:
        q, k, v, pair_bias, residual_mask, residual_input, local_output, *tail = arguments
        out_peers = tail[:MAX_WORLD_SIZE]
        residual_out_peers = tail[MAX_WORLD_SIZE : 2 * MAX_WORLD_SIZE]
        completion_peers = tail[2 * MAX_WORLD_SIZE : 3 * MAX_WORLD_SIZE]
        completion = tail[3 * MAX_WORLD_SIZE]
        completion_counter = tail[3 * MAX_WORLD_SIZE + 1]
        completion_epoch = tail[3 * MAX_WORLD_SIZE + 2]
        lse = tail[3 * MAX_WORLD_SIZE + 3]
        return (
            _as_dynamic_tensor(q, (0, 1, 2, 3), alignment=16, tiled_modes=(2,)),
            _as_dynamic_layout_tensor(k, alignment=16, leading_dim=4),
            _as_dynamic_layout_tensor(v, alignment=16, leading_dim=4),
            _as_dynamic_layout_tensor(pair_bias, alignment=16, leading_dim=4),
            _as_dynamic_tensor(
                residual_mask, (0, 1, 4), alignment=16, tiled_modes=(4,)
            ),
            _as_dynamic_tensor(
                residual_input, (0, 1, 2), alignment=16, tiled_modes=(2,)
            ),
            _as_dynamic_tensor(
                local_output, (0, 1, 2, 3), alignment=16, tiled_modes=(2,)
            ),
            *(
                _as_dynamic_layout_tensor(out, alignment=16, leading_dim=4)
                for out in out_peers
            ),
            *(
                _as_dynamic_layout_tensor(out, alignment=16, leading_dim=3)
                for out in residual_out_peers
            ),
            *(
                _as_dynamic_tensor(completion, (0,), alignment=4)
                for completion in completion_peers
            ),
            _as_dynamic_tensor(completion, (0,), alignment=4),
            _as_dynamic_tensor(completion_counter, (0,), alignment=4),
            completion_epoch,
            _as_dynamic_tensor(lse, (0, 1, 2, 3), alignment=16, tiled_modes=(3,)),
            self.world_size,
            stream,
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_bias: torch.Tensor,
        residual_mask: torch.Tensor,
        residual_input: torch.Tensor,
        local_output: torch.Tensor,
        out_peers: tuple[torch.Tensor, ...],
        residual_out_peers: tuple[torch.Tensor, ...],
        completion_peers: tuple[torch.Tensor, ...],
        completion: torch.Tensor,
        completion_counter: torch.Tensor,
        completion_epoch: int,
        lse: torch.Tensor,
        *,
        stream=None,
    ) -> None:
        self._validate_arguments(
            q,
            k,
            v,
            pair_bias,
            residual_mask,
            residual_input,
            local_output,
            out_peers,
            residual_out_peers,
            completion_peers,
            completion,
            completion_counter,
            completion_epoch,
            lse,
        )
        device_index = q.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if self._device_index is not None and self._device_index != device_index:
            raise RuntimeError(
                "the cached forward is bound to CUDA device "
                f"{self._device_index}, but received device {device_index}"
            )

        if stream is None:
            stream = torch.cuda.current_stream(device_index)
        cuda_stream = cuda.CUstream(stream.cuda_stream)
        padded_peers = out_peers + (out_peers[0],) * (MAX_WORLD_SIZE - self.world_size)
        padded_residual_peers = residual_out_peers + (residual_out_peers[0],) * (
            MAX_WORLD_SIZE - self.world_size
        )
        padded_completions = completion_peers + (completion_peers[0],) * (
            MAX_WORLD_SIZE - self.world_size
        )
        arguments = (
            q,
            k,
            v,
            pair_bias,
            residual_mask,
            residual_input,
            local_output,
            *padded_peers,
            *padded_residual_peers,
            *padded_completions,
            completion,
            completion_counter,
            completion_epoch,
            lse,
        )

        if self._compiled is None:
            with self._lock:
                if self._compiled is None:
                    compile_arguments = self._make_compile_arguments(
                        arguments, cuda_stream
                    )
                    self._compiled = CompileCallable(
                        (PtxasOptions("--maxrregcount=255"), EnableTVMFFI)
                    )(_build_and_launch_forward, *compile_arguments)
                    self._device_index = device_index
                    self._compile_count += 1

        self._compiled(*arguments, cuda_stream)

_FORWARD_KERNELS: dict[tuple[int, int], DistributedEvoAttentionForward] = {}


def get_distributed_evoattention_forward(
    world_size: int,
    source_rank: int,
) -> DistributedEvoAttentionForward:
    """Return the cached launcher for one distributed source rank."""
    key = (world_size, source_rank)
    if key not in _FORWARD_KERNELS:
        _FORWARD_KERNELS[key] = DistributedEvoAttentionForward(
            world_size, source_rank
        )
    return _FORWARD_KERNELS[key]


__all__ = ["get_distributed_evoattention_forward"]
