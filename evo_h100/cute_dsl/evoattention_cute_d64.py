"""Fused EvoAttention forward kernel for NVIDIA Hopper.

The public entry point, :func:`get_evoattention_forward`, returns a cached
TVM-FFI callable. It accepts prepared contiguous tensors with this contract:

- query/key/value/output: ``[B, S, H, N, 64]`` BF16
- pair bias: ``[B, H, N, N]`` FP32
- residual mask: ``[B, S, N]`` FP32 containing ``0`` or ``-1e9``
- log-sum-exp: ``[B, S, H, N]`` FP32

The implementation uses one TMA producer warp and two 128-thread WGMMA
consumer warpgroups. Each consumer owns an M=64 query tile. B, S, H, and N
remain runtime dynamic, while D=64 and the tile schedule are static.
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


HEAD_DIMENSION = 64
QUERY_TILE_SIZE = 64
KEY_TILE_SIZE = 64
CONSUMER_WARPGROUP_COUNT = 2
THREADS_PER_WARPGROUP = 128
CONSUMER_THREAD_COUNT = CONSUMER_WARPGROUP_COUNT * THREADS_PER_WARPGROUP
PRODUCER_WARP_INDEX = 8
THREADS_PER_BLOCK = CONSUMER_THREAD_COUNT + 32
CTA_QUERY_TILE_SIZE = CONSUMER_WARPGROUP_COUNT * QUERY_TILE_SIZE
SHORT_SEQUENCE_LENGTH = 2 * KEY_TILE_SIZE
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)
KEY_VALUE_STAGES = 2
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
    output_tma: cute.CopyAtom,
    output_tensor: cute.Tensor,
    pair_bias: cute.Tensor,
    residual_mask: cute.Tensor,
    out: cute.Tensor,
    lse: cute.Tensor,
    query_layout: cute.ComposedLayout,
    key_layout: cute.ComposedLayout,
    value_layout: cute.ComposedLayout,
    pair_bias_layout: cute.ComposedLayout,
    output_layout: cute.ComposedLayout,
    score_mma: cute.TiledMma,
    output_mma: cute.TiledMma,
    output_register_copy: cute.TiledCopy,
):
    """Run the warp-specialized SM90 forward pass.

    Warps 0-7 form two consumer warpgroups. Warp 8 is the sole TMA producer.
    Each consumer owns an M=64 query tile and shares a two-stage K/V pipeline.
    """
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    linear_block, _, _ = cute.arch.block_idx()
    n_ctx = out.shape[1]
    query_blocks = n_ctx // CTA_QUERY_TILE_SIZE
    query_block = linear_block % query_blocks
    problem_idx = linear_block // query_blocks
    problem_count = out.shape[0]

    batch_heads = pair_bias.shape[0]
    batch_sequences = residual_mask.shape[0]
    batch = (batch_heads * batch_sequences) // problem_count
    heads = batch_heads // batch
    sequence_count = batch_sequences // batch
    head_idx = problem_idx % heads
    batch_seq_idx = problem_idx // heads
    batch_idx = batch_seq_idx // sequence_count
    pair_bias_idx = batch_idx * heads + head_idx

    allocator = cutlass.utils.SmemAllocator()
    shared_query = allocator.allocate_tensor(
        cutlass.BFloat16, query_layout.outer, swizzle=query_layout.inner
    )
    shared_key = allocator.allocate_tensor(
        cutlass.BFloat16, key_layout.outer, swizzle=key_layout.inner
    )
    shared_pair_bias = allocator.allocate_tensor(
        cutlass.Float32,
        pair_bias_layout.outer,
        swizzle=pair_bias_layout.inner,
    )
    shared_value = allocator.allocate_tensor(
        cutlass.BFloat16, value_layout.outer, swizzle=value_layout.inner
    )
    shared_mask = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((KEY_TILE_SIZE, KEY_VALUE_STAGES), stride=(1, KEY_TILE_SIZE)),
        swizzle=None,
    )
    # Residual mask is physically one [Bc] vector per stage. This zero-stride
    # M mode represents the logical [Mq,Bc] broadcast without replication.
    shared_broadcast_mask = cute.make_tensor(
        shared_mask.iterator,
        cute.make_layout(
            (QUERY_TILE_SIZE, KEY_TILE_SIZE, KEY_VALUE_STAGES),
            stride=(0, 1, KEY_TILE_SIZE),
        ),
    )
    shared_output = allocator.allocate_tensor(
        cutlass.BFloat16,
        output_layout.outer,
        swizzle=output_layout.inner,
    )
    query_ready = allocator.allocate_array(cutlass.Int64, CONSUMER_WARPGROUP_COUNT)
    key_value_ready = allocator.allocate_array(cutlass.Int64, KEY_VALUE_STAGES)
    key_value_free = allocator.allocate_array(cutlass.Int64, KEY_VALUE_STAGES)
    pair_bias_ready = allocator.allocate_array(cutlass.Int64, CONSUMER_WARPGROUP_COUNT)

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
        for consumer in cutlass.range_constexpr(CONSUMER_WARPGROUP_COUNT):
            cute.arch.mbarrier_init(pair_bias_ready + consumer, 1)
        cute.nvgpu.cpasync.prefetch_descriptor(query_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(key_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(value_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(pair_bias_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(residual_mask_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(output_tma)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    query_tiles = cute.zipped_divide(q, (QUERY_TILE_SIZE, HEAD_DIMENSION))
    query_problem = query_tiles[None, (None, 0, problem_idx)]
    shared_query_partition, global_query_partition = cute.nvgpu.cpasync.tma_partition(
        query_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_query, 0, 2),
        query_problem,
    )
    key_tiles = cute.zipped_divide(k, (KEY_TILE_SIZE, HEAD_DIMENSION))
    key_problem = key_tiles[None, (None, 0, problem_idx)]
    shared_key_partition, global_key_partition = cute.nvgpu.cpasync.tma_partition(
        key_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_key, 0, 2),
        key_problem,
    )
    value_tiles = cute.zipped_divide(v, (HEAD_DIMENSION, KEY_TILE_SIZE))
    value_problem = value_tiles[None, (0, None, problem_idx)]
    shared_value_partition, global_value_partition = cute.nvgpu.cpasync.tma_partition(
        value_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_value, 0, 2),
        value_problem,
    )
    pair_bias_tiles = cute.zipped_divide(
        pair_bias_tensor, (KEY_TILE_SIZE, QUERY_TILE_SIZE)
    )
    pair_bias_problem = pair_bias_tiles[None, (None, None, pair_bias_idx)]
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
    residual_mask_problem = residual_mask_tiles[None, (None, batch_seq_idx)]
    shared_mask_partition, global_mask_partition = cute.nvgpu.cpasync.tma_partition(
        residual_mask_tma,
        0,
        cute.make_layout(1),
        shared_mask,
        residual_mask_problem,
    )
    output_tiles = cute.zipped_divide(
        output_tensor, (HEAD_DIMENSION, CTA_QUERY_TILE_SIZE)
    )
    output_problem = output_tiles[None, (0, None, problem_idx)]
    shared_output_partition, global_output_partition = cute.nvgpu.cpasync.tma_partition(
        output_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_output, 0, 2),
        output_problem,
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
            global_query_partition[None, query_block * 2],
            shared_query_partition[None, 0],
            tma_bar_ptr=query_ready,
        )
        cute.copy(
            query_tma,
            global_query_partition[None, query_block * 2 + 1],
            shared_query_partition[None, 1],
            tma_bar_ptr=query_ready + 1,
        )

        key_block = cutlass.Int32(0)
        for _ in cutlass.range(num_key_blocks):
            stage = key_block % KEY_VALUE_STAGES
            if key_block >= KEY_VALUE_STAGES:
                free_phase = ((key_block // KEY_VALUE_STAGES) - 1) % 2
                cute.arch.mbarrier_wait(key_value_free + stage, free_phase)
            if warpgroup_thread_idx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    key_value_ready + stage,
                    2 * KEY_TILE_SIZE * HEAD_DIMENSION * 2 + KEY_TILE_SIZE * 4,
                )
            cute.copy(
                key_tma,
                global_key_partition[None, key_block],
                shared_key_partition[None, stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            cute.copy(
                value_tma,
                global_value_partition[None, key_block],
                shared_value_partition[None, stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            cute.copy(
                residual_mask_tma,
                global_mask_partition[None, key_block],
                shared_mask_partition[None, stage],
                tma_bar_ptr=key_value_ready + stage,
            )
            if key_block + KEY_VALUE_STAGES < num_key_blocks:
                cute.prefetch(
                    key_tma,
                    global_key_partition[None, key_block + KEY_VALUE_STAGES],
                )
                cute.prefetch(
                    value_tma,
                    global_value_partition[None, key_block + KEY_VALUE_STAGES],
                )
            key_block += 1

    else:
        consumer_warpgroup = thread_idx // THREADS_PER_WARPGROUP
        warpgroup_thread_idx = thread_idx - consumer_warpgroup * THREADS_PER_WARPGROUP
        query_base = (
            query_block * CTA_QUERY_TILE_SIZE + consumer_warpgroup * QUERY_TILE_SIZE
        )

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
        pair_bias_partition = qk_thread.partition_C(
            shared_pair_bias_row_col[None, None, consumer_warpgroup]
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
        pair_bias_phase = cutlass.Int32(0)
        for _ in cutlass.range(num_key_blocks):
            stage = key_block % KEY_VALUE_STAGES
            full_phase = (key_block // KEY_VALUE_STAGES) % 2
            # Pair bias is private to this M=64 consumer tile.  Start its TMA
            # before waiting on K/V so the independent transfers overlap.
            if warpgroup_thread_idx < 32:
                if warpgroup_thread_idx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        pair_bias_ready + consumer_warpgroup,
                        QUERY_TILE_SIZE * KEY_TILE_SIZE * 4,
                    )
                cute.copy(
                    pair_bias_tma,
                    global_pair_bias_partition[
                        None,
                        key_block,
                        query_block * 2 + consumer_warpgroup,
                    ],
                    shared_pair_bias_partition[None, consumer_warpgroup],
                    tma_bar_ptr=pair_bias_ready + consumer_warpgroup,
                    cache_policy=cutlass.Int64(PAIR_BIAS_EVICT_LAST_POLICY),
                )

            cute.arch.mbarrier_wait(key_value_ready + stage, full_phase)
            # Diagonal FA3 ping-pong. WG1's QK(k) follows WG0's QK(k).
            # Later, WG1's PV(k) waits for WG0's QK(k+1), allowing the latter
            # to overlap WG1's softmax(k).
            # Delay only the first QK, not WG1's independent PB(0) transfer.
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
                key_fragment[None, None, None, stage],
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
                pair_bias_ready + consumer_warpgroup, pair_bias_phase
            )
            pair_bias_phase ^= 1
            pair_bias_fragment.store(pair_bias_partition.load())

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
                mask_0 = shared_broadcast_mask[0, col_0, stage]
                mask_1 = shared_broadcast_mask[0, col_1, stage]
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
                    probability = cute.math.exp(
                        score_fragment[elem] - new_max_0, fastmath=True
                    )
                    local_sum_0 += probability
                else:
                    probability = cute.math.exp(
                        score_fragment[elem] - new_max_1, fastmath=True
                    )
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
                value_fragment[None, None, None, stage],
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
            lse[problem_idx, query_base + first_row] = running_max_0 + cute.math.log(
                running_sum_0, fastmath=True
            )
            lse[problem_idx, query_base + second_row] = running_max_1 + cute.math.log(
                running_sum_1, fastmath=True
            )
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(
            barrier_id=OUTPUT_READY_BARRIER_ID,
            number_of_threads=CONSUMER_THREAD_COUNT,
        )
        if consumer_warpgroup == 0 and warpgroup_thread_idx < 32:
            cute.copy(
                output_tma,
                shared_output_partition,
                global_output_partition[None, query_block],
            )
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)


@cute.jit
def _build_and_launch_forward(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    pair_bias: cute.Tensor,
    residual_mask: cute.Tensor,
    out: cute.Tensor,
    lse: cute.Tensor,
    stream: cuda.CUstream,
):
    problem_count = q.shape[0]
    n_ctx = q.shape[1]

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
        cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
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
        num_stages=KEY_VALUE_STAGES,
    )
    value_layout = sm90_utils.make_smem_layout_b(
        LayoutEnum.COL_MAJOR,
        (QUERY_TILE_SIZE, HEAD_DIMENSION, KEY_TILE_SIZE),
        b_dtype=cutlass.BFloat16,
        num_stages=KEY_VALUE_STAGES,
    )

    query_view = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[1, 2, 0]))
    key_view = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 2, 0]))
    value_view = cute.make_tensor(v.iterator, cute.select(v.layout, mode=[2, 1, 0]))
    pair_bias_view = cute.make_tensor(
        pair_bias.iterator,
        cute.select(pair_bias.layout, mode=[2, 1, 0]),
    )
    residual_mask_view = cute.make_tensor(
        residual_mask.iterator,
        cute.select(residual_mask.layout, mode=[1, 0]),
    )
    output_view = cute.make_tensor(
        out.iterator, cute.select(out.layout, mode=[2, 1, 0])
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
        SmemLayoutAtomKind.MN_SW128, cutlass.Float32
    )
    pair_bias_layout = cute.tile_to_shape(
        pair_bias_layout_atom,
        (KEY_TILE_SIZE, QUERY_TILE_SIZE, 2),
        order=(1, 0, 2),
    )
    pair_bias_tma_layout = cute.tile_to_shape(
        pair_bias_layout_atom,
        (KEY_TILE_SIZE, QUERY_TILE_SIZE, 1),
        order=(1, 0, 2),
    )
    residual_mask_tma_layout = cute.make_layout(KEY_TILE_SIZE, stride=1)
    output_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        SmemLayoutAtomKind.MN_SW128, cutlass.BFloat16
    )
    output_layout = cute.tile_to_shape(
        output_layout_atom,
        (HEAD_DIMENSION, CTA_QUERY_TILE_SIZE),
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
    output_tma, output_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        output_view,
        output_layout,
        (HEAD_DIMENSION, CTA_QUERY_TILE_SIZE),
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
        output_tma,
        output_tensor,
        pair_bias,
        residual_mask,
        out,
        lse,
        query_layout,
        key_layout,
        value_layout,
        pair_bias_layout,
        output_layout,
        score_mma,
        output_mma,
        output_register_copy,
    ).launch(
        # Flatten the work grid into x: grid.y is limited to 65535, while the
        # largest target has P=B*S*H=65536 before accounting for query tiles.
        grid=((n_ctx // CTA_QUERY_TILE_SIZE) * problem_count, 1, 1),
        block=(THREADS_PER_BLOCK, 1, 1),
        min_blocks_per_mp=2,
        stream=stream,
    )


class EvoAttentionForward:
    """Validate inputs and lazily compile one dynamic TVM-FFI specialization."""

    def __init__(self) -> None:
        self._compiled = None
        self._compile_count = 0
        self._device_index: int | None = None
        self._lock = threading.Lock()

    @property
    def compile_count(self) -> int:
        return self._compile_count

    @staticmethod
    def _validate_arguments(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_bias: torch.Tensor,
        residual_mask: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
    ) -> None:
        if q.ndim != 5:
            raise ValueError(
                f"q must have shape [B, S, H, N, 64], got {tuple(q.shape)}"
            )
        batch, sequence_count, heads, n_ctx, dim = q.shape
        if sequence_count != n_ctx:
            raise ValueError(
                f"this forward requires S=N, got S={sequence_count}, N={n_ctx}"
            )
        if dim != HEAD_DIMENSION:
            raise ValueError(f"head dimension must be {HEAD_DIMENSION}, got {dim}")
        if n_ctx % CTA_QUERY_TILE_SIZE:
            raise ValueError(
                f"N must be divisible by {CTA_QUERY_TILE_SIZE}, got {n_ctx}"
            )
        if k.shape != q.shape or v.shape != q.shape or out.shape != q.shape:
            raise ValueError("q, k, v and out must have identical shapes")
        if tuple(pair_bias.shape) != (batch, heads, n_ctx, n_ctx):
            raise ValueError(
                "pair_bias must have shape [B, H, N, N], got "
                f"{tuple(pair_bias.shape)}"
            )
        if tuple(residual_mask.shape) != (batch, sequence_count, n_ctx):
            raise ValueError(
                "residual_mask must have shape [B, S, N], got "
                f"{tuple(residual_mask.shape)}"
            )
        if tuple(lse.shape) != (batch, sequence_count, heads, n_ctx):
            raise ValueError(
                "lse must have shape [B, S, H, N], got " f"{tuple(lse.shape)}"
            )

        tensors = (q, k, v, pair_bias, residual_mask, out, lse)
        if any(t.device != q.device for t in tensors):
            raise ValueError("all tensors must be on the same CUDA device")
        if q.device.type != "cuda":
            raise ValueError("EvoAttention requires CUDA tensors")
        if any(not t.is_contiguous() for t in tensors):
            raise ValueError("all prepared tensors must be contiguous")
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            raise TypeError("q, k and v must be bfloat16")
        if out.dtype != torch.bfloat16:
            raise TypeError("out must be bfloat16")
        if pair_bias.dtype != torch.float32 or residual_mask.dtype != torch.float32:
            raise TypeError("pair_bias and residual_mask must be float32")
        if lse.dtype != torch.float32:
            raise TypeError("lse must be float32")

    @staticmethod
    def _make_runtime_views(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pair_bias: torch.Tensor,
        residual_mask: torch.Tensor,
        out: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch, sequence_count, heads, n_ctx, dim = q.shape
        problems = batch * sequence_count * heads
        return (
            q.view(problems, n_ctx, dim),
            k.view(problems, n_ctx, dim),
            v.view(problems, n_ctx, dim),
            pair_bias.view(batch * heads, n_ctx, n_ctx),
            residual_mask.view(batch * sequence_count, n_ctx),
            out.view(problems, n_ctx, dim),
            lse.view(problems, n_ctx),
        )

    @staticmethod
    def _make_compile_arguments(
        views: tuple[torch.Tensor, ...],
        stream: cuda.CUstream,
    ) -> tuple[object, ...]:
        q, k, v, pair_bias, residual_mask, out, lse = views
        return (
            _as_dynamic_tensor(q, (0, 1), alignment=16, tiled_modes=(1,)),
            _as_dynamic_tensor(k, (0, 1), alignment=16, tiled_modes=(1,)),
            _as_dynamic_tensor(v, (0, 1), alignment=16, tiled_modes=(1,)),
            _as_dynamic_tensor(pair_bias, (0, 1, 2), alignment=16, tiled_modes=(1, 2)),
            _as_dynamic_tensor(residual_mask, (0, 1), alignment=16, tiled_modes=(1,)),
            _as_dynamic_tensor(out, (0, 1), alignment=16, tiled_modes=(1,)),
            _as_dynamic_tensor(lse, (0, 1), alignment=16, tiled_modes=(1,)),
            stream,
        )

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
        stream=None,
    ) -> None:
        self._validate_arguments(q, k, v, pair_bias, residual_mask, out, lse)
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
        views = self._make_runtime_views(q, k, v, pair_bias, residual_mask, out, lse)

        if self._compiled is None:
            with self._lock:
                if self._compiled is None:
                    compile_arguments = self._make_compile_arguments(views, cuda_stream)
                    self._compiled = CompileCallable(
                        (PtxasOptions("--maxrregcount=255"), EnableTVMFFI)
                    )(_build_and_launch_forward, *compile_arguments)
                    self._device_index = device_index
                    self._compile_count += 1

        self._compiled(*views, cuda_stream)


_FORWARD_KERNEL = EvoAttentionForward()


def get_evoattention_forward() -> EvoAttentionForward:
    """Return the process-wide cached EvoAttention forward launcher."""
    return _FORWARD_KERNEL


__all__ = ["get_evoattention_forward"]
