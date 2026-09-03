"""CuTe-DSL EvoAttention backward for NVIDIA Hopper.

Prepared tensor contract:

* q/k/v/output/output_gradient and dQ/dK/dV: ``[B, S, N, H, 64]`` BF16
* pair bias and its accumulator: ``[B, H, N, N]`` FP32
* residual mask: ``[B, S, N]`` FP32 containing ``0`` or ``-1e9``
* logsumexp and delta workspace: ``[B, S, H, N]`` FP32
* final pair-bias gradient: ``[B, H, N, N]`` BF16

The fixed main schedule is M64 x N128 with two consumer warpgroups and two
stages for Q, dO, P, and dS. For H=16 and N >= 640, dPairBias uses a separate
64 x 128 x 16 split-sequence reduction while the main kernel omits its
per-sequence atomics; H=4 retains the cheaper fused update. Preprocessing,
gradient kernels, dQ postprocessing, and pair-bias initialization/conversion
are emitted behind one TVM-FFI call. B, S, H, and N are runtime dynamic; D=64
is static.

This module uses the SM90 CuTe FlashAttention primitives installed with vLLM.
Set ``EVOATTENTION_CUTE_SITE_PACKAGES`` if that package is not in the sibling
``vllm`` Conda environment.
"""

from __future__ import annotations

import math
import os
import sys
import sysconfig
import threading
import types
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable

import torch
from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.compiler import CompileCallable, EnableTVMFFI, PtxasOptions
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import hopper_helpers as sm90_utils
from cutlass.utils.layout import LayoutEnum


HEAD_DIMENSION = 64
QUERY_TILE_SIZE = 64
KEY_TILE_SIZE = 128
ELEMENTWISE_THREADS = 256
ELEMENTS_PER_THREAD = 4
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)
MAX_DQ_WORKSPACE_BYTES = 8 * 1024**3
MAX_WORKSPACE_CACHE_BYTES = 16 * 1024**3
MAX_PREPARED_CACHE_BYTES = 4 * 1024**3
MAX_GRID_Y = 65_535
MAX_CACHED_BUFFER_SETS = 8
SPLIT_SEQUENCE_CHUNK_SIZE = 16
SPLIT_PAIR_BIAS_THREADS = 512


def _serialized(method):
    """Serialize launcher metadata updates while allowing queued GPU overlap."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


def _load_upstream_cute() -> tuple[object, ...]:
    """Load pure-Python CuTe primitives without importing the vLLM runtime."""
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    configured = os.environ.get("EVOATTENTION_CUTE_SITE_PACKAGES")
    candidates = (
        Path(configured) if configured else None,
        Path(sysconfig.get_paths()["purelib"]),
        Path(sys.executable).resolve().parent.parent
        / "lib"
        / python_version
        / "site-packages",
    )
    site_packages = next(
        (
            candidate
            for candidate in candidates
            if candidate is not None
            and (candidate / "vllm/vllm_flash_attn/cute/flash_bwd_sm90.py").is_file()
        ),
        None,
    )
    if site_packages is None:
        raise ImportError(
            "CuTe FlashAttention sources were not found; set "
            "EVOATTENTION_CUTE_SITE_PACKAGES to their site-packages directory"
        )

    site_string = str(site_packages)
    if site_string not in sys.path:
        sys.path.append(site_string)
    if "quack" not in sys.modules:
        module = types.ModuleType("quack")
        module.__path__ = [str(site_packages / "quack")]
        module.__package__ = "quack"
        sys.modules["quack"] = module
    for name, relative_path in (
        ("vllm", "vllm"),
        ("vllm.vllm_flash_attn", "vllm/vllm_flash_attn"),
        ("vllm.vllm_flash_attn.cute", "vllm/vllm_flash_attn/cute"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(site_packages / relative_path)]
            module.__package__ = name
            sys.modules[name] = module

    from quack import copy_utils
    from vllm.vllm_flash_attn.cute import utils
    from vllm.vllm_flash_attn.cute.flash_bwd_postprocess import (
        FlashAttentionBackwardPostprocess,
    )
    from vllm.vllm_flash_attn.cute.flash_bwd_preprocess import (
        FlashAttentionBackwardPreprocess,
    )
    from vllm.vllm_flash_attn.cute.flash_bwd_sm90 import (
        FlashAttentionBackwardSm90,
    )
    from vllm.vllm_flash_attn.cute.named_barrier import NamedBarrierBwd

    return (
        copy_utils,
        utils,
        FlashAttentionBackwardPreprocess,
        FlashAttentionBackwardSm90,
        FlashAttentionBackwardPostprocess,
        NamedBarrierBwd,
    )


(
    _copy_utils,
    _flash_utils,
    _FlashAttentionBackwardPreprocess,
    _FlashAttentionBackwardSm90,
    _FlashAttentionBackwardPostprocess,
    _NamedBarrierBwd,
) = _load_upstream_cute()


@cute.jit
def _evo_score_mod(
    scores,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    """Add pair bias and the broadcast residual mask to one score."""
    del seqlen_info
    pair_bias, residual_mask, _ = aux_tensors
    flat_batch = batch_idx[0]
    sequence_count = residual_mask.shape[1]
    batch = flat_batch // sequence_count
    sequence = flat_batch - batch * sequence_count
    bias = pair_bias[batch, head_idx[0], q_idx[0], kv_idx[0]]
    mask = residual_mask[batch, sequence, kv_idx[0]]
    return scores + _flash_utils.scalar_to_ssa(bias + mask, Float32)


@cute.jit
def _evo_score_mod_backward(
    score_gradient,
    score,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    """Accumulate the additive score gradient into FP32 dPairBias."""
    del score, seqlen_info
    _, residual_mask, pair_bias_gradient = aux_tensors
    flat_batch = batch_idx[0]
    sequence_count = residual_mask.shape[1]
    batch = flat_batch // sequence_count
    cute.arch.atomic_add(
        _flash_utils.elem_pointer(
            pair_bias_gradient,
            (batch, head_idx[0], q_idx[0], kv_idx[0]),
        ),
        score_gradient[0],
        sem="relaxed",
        scope="gpu",
    )
    return score_gradient


@cute.jit(preprocess=False)
def _evo_score_mod_backward_no_dpair_bias(
    score_gradient,
    score,
    batch_idx,
    head_idx,
    q_idx,
    kv_idx,
    seqlen_info,
    aux_tensors,
):
    """Return dS unchanged when dPairBias is reduced by a separate kernel."""
    del score, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors
    return score_gradient


@dsl_user_op
def _direct_bulk_store_f32(
    shared_pointer, global_pointer, store_bytes, *, loc=None, ip=None
):
    """Store dQ without reduction when N=128 has exactly one KV tile."""
    shared_pointer_i32 = shared_pointer.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [global_pointer.llvm_ptr, shared_pointer_i32, Int32(store_bytes).ir_value()],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
    )


class _RuntimeDQStoreMain(_FlashAttentionBackwardSm90):
    """Dense SM90 dQ epilogue with a runtime direct-store fast path."""

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        blocksparse_tensors=None,
        mdQ_semaphore=None,
    ):
        # EvoAttention is dense, global, fixed-length, and nondeterministic.
        # The upstream deterministic, local, variable-length, and block-sparse
        # branches are deliberately omitted from this specialized override.
        del blocksparse_tensors, mdQ_semaphore

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            sequence_info = SeqlenInfoCls(batch_idx)
            query_gradient = mdQaccum[None, head_idx, batch_idx]
            query_gradient_tiles = cute.local_tile(
                query_gradient,
                (
                    cute.make_layout(
                        (
                            self.tile_m * self.tile_hdim // self.num_wg_dQ,
                            self.num_wg_dQ,
                        )
                    ),
                ),
                (None,),
            )

            minimum_query_block, maximum_query_block = block_info.get_m_block_min_max(
                sequence_info, n_block
            )
            loop_count = maximum_query_block - minimum_query_block
            for iteration in cutlass.range(loop_count, unroll=1):
                query_block = minimum_query_block + iteration
                for warp_group_idx in cutlass.range_constexpr(self.num_wg_dQ):
                    cute.arch.cp_async_bulk_wait_group(
                        self.num_wg_dQ - 1 - warp_group_idx,
                        read=True,
                    )
                    cute.arch.barrier_arrive(
                        barrier_id=int(_NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                        number_of_threads=self.num_threads_per_warp_group
                        + cute.arch.WARP_SIZE,
                    )

                for warp_group_idx in cutlass.range_constexpr(self.num_wg_dQ):
                    cute.arch.barrier(
                        barrier_id=int(_NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                        number_of_threads=self.num_threads_per_warp_group
                        + cute.arch.WARP_SIZE,
                    )
                    with cute.arch.elect_one():
                        store_source = sdQaccum[None, warp_group_idx].iterator
                        store_destination = query_gradient_tiles[
                            (None, warp_group_idx), query_block
                        ].iterator
                        if sequence_info.seqlen_k == self.tile_n:
                            _direct_bulk_store_f32(
                                store_source,
                                store_destination,
                                self.tma_copy_bytes["dQ"],
                            )
                        else:
                            _copy_utils.cpasync_reduce_bulk_add_f32(
                                store_source,
                                store_destination,
                                self.tma_copy_bytes["dQ"],
                            )
                    cute.arch.cp_async_bulk_commit_group()

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        cute.arch.cp_async_bulk_wait_group(0, read=True)


def _make_runtime_dq_main(score_mod_backward):
    return _RuntimeDQStoreMain(
        cutlass.BFloat16,
        HEAD_DIMENSION,
        HEAD_DIMENSION,
        qhead_per_kvhead=1,
        tile_m=QUERY_TILE_SIZE,
        tile_n=KEY_TILE_SIZE,
        Q_stage=2,
        dO_stage=2,
        PdS_stage=2,
        SdP_swapAB=True,
        dKV_swapAB=False,
        dQ_swapAB=False,
        AtomLayoutMSdP=1,
        AtomLayoutNdKV=2,
        AtomLayoutMdQ=1,
        num_threads=384,
        V_in_regs=False,
        score_mod=_evo_score_mod,
        score_mod_bwd=score_mod_backward,
        has_aux_tensors=True,
        dQ_single_wg=True,
    )


@cute.kernel
def _zero_fp32(destination: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    base = (block_idx * ELEMENTWISE_THREADS + thread_idx) * ELEMENTS_PER_THREAD
    for lane in cutlass.range_constexpr(ELEMENTS_PER_THREAD):
        if base + lane < cute.size(destination):
            destination[base + lane] = Float32(0.0)


@cute.kernel
def _convert_fp32_to_bf16(source: cute.Tensor, destination: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    base = (block_idx * ELEMENTWISE_THREADS + thread_idx) * ELEMENTS_PER_THREAD
    for lane in cutlass.range_constexpr(ELEMENTS_PER_THREAD):
        if base + lane < cute.size(source):
            destination[base + lane] = source[base + lane].to(cutlass.BFloat16)


@cute.kernel
def _split_sequence_pair_bias_gradient_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    output_gradient: cute.Tensor,
    query_tma: cute.CopyAtom,
    key_tma: cute.CopyAtom,
    value_tma: cute.CopyAtom,
    output_gradient_tma: cute.CopyAtom,
    pair_bias: cute.Tensor,
    residual_mask: cute.Tensor,
    logsumexp: cute.Tensor,
    delta: cute.Tensor,
    pair_bias_gradient: cute.Tensor,
    score_mma: cute.TiledMma,
    query_copy: cute.TiledCopy,
    key_copy: cute.TiledCopy,
    query_layout: cute.ComposedLayout,
    key_layout: cute.ComposedLayout,
):
    """Reduce dS over 16 sequences per CTA for one 64 x 128 pair tile."""
    thread_idx, _, _ = cute.arch.thread_idx()
    pair_tile_idx, batch_head_idx, sequence_chunk_idx = cute.arch.block_idx()

    context_length = pair_bias.shape[2]
    head_count = pair_bias.shape[1]
    sequence_count = residual_mask.shape[1]
    key_block_count = context_length // KEY_TILE_SIZE
    query_block_idx = pair_tile_idx // key_block_count
    key_block_idx = pair_tile_idx - query_block_idx * key_block_count
    batch_idx = batch_head_idx // head_count
    head_idx = batch_head_idx - batch_idx * head_count
    query_base = query_block_idx * QUERY_TILE_SIZE
    key_base = key_block_idx * KEY_TILE_SIZE

    allocator = cutlass.utils.SmemAllocator()
    shared_query = allocator.allocate_tensor(
        cutlass.BFloat16, query_layout.outer, swizzle=query_layout.inner
    )
    shared_output_gradient = allocator.allocate_tensor(
        cutlass.BFloat16, query_layout.outer, swizzle=query_layout.inner
    )
    shared_key = allocator.allocate_tensor(
        cutlass.BFloat16, key_layout.outer, swizzle=key_layout.inner
    )
    shared_value = allocator.allocate_tensor(
        cutlass.BFloat16, key_layout.outer, swizzle=key_layout.inner
    )
    shared_mask = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((KEY_TILE_SIZE, 2), stride=(1, KEY_TILE_SIZE)),
    )
    shared_lse = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((QUERY_TILE_SIZE, 2), stride=(1, QUERY_TILE_SIZE)),
    )
    shared_delta = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((QUERY_TILE_SIZE, 2), stride=(1, QUERY_TILE_SIZE)),
    )
    activation_ready = allocator.allocate_array(cutlass.Int64, 2)

    if thread_idx == 0:
        for stage in cutlass.range_constexpr(2):
            cute.arch.mbarrier_init(activation_ready + stage, 1)
        cute.nvgpu.cpasync.prefetch_descriptor(query_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(key_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(value_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(output_gradient_tma)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    query_tiles = cute.zipped_divide(q, (QUERY_TILE_SIZE, HEAD_DIMENSION))
    query_problem = query_tiles[None, (None, 0, head_idx, None)]
    shared_query_partition, global_query_partition = cute.nvgpu.cpasync.tma_partition(
        query_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_query, 0, 2),
        query_problem,
    )
    output_gradient_tiles = cute.zipped_divide(
        output_gradient, (QUERY_TILE_SIZE, HEAD_DIMENSION)
    )
    output_gradient_problem = output_gradient_tiles[None, (None, 0, head_idx, None)]
    shared_output_gradient_partition, global_output_gradient_partition = (
        cute.nvgpu.cpasync.tma_partition(
            output_gradient_tma,
            0,
            cute.make_layout(1),
            cute.group_modes(shared_output_gradient, 0, 2),
            output_gradient_problem,
        )
    )
    key_tiles = cute.zipped_divide(k, (KEY_TILE_SIZE, HEAD_DIMENSION))
    key_problem = key_tiles[None, (None, 0, head_idx, None)]
    shared_key_partition, global_key_partition = cute.nvgpu.cpasync.tma_partition(
        key_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_key, 0, 2),
        key_problem,
    )
    value_tiles = cute.zipped_divide(v, (KEY_TILE_SIZE, HEAD_DIMENSION))
    value_problem = value_tiles[None, (None, 0, head_idx, None)]
    shared_value_partition, global_value_partition = cute.nvgpu.cpasync.tma_partition(
        value_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(shared_value, 0, 2),
        value_problem,
    )

    mma_thread = score_mma.get_slice(thread_idx)
    score_coordinates = mma_thread.partition_C(
        cute.make_identity_tensor((QUERY_TILE_SIZE, KEY_TILE_SIZE))
    )
    pair_bias_fragment = score_mma.make_fragment_C(
        score_mma.partition_shape_C((QUERY_TILE_SIZE, KEY_TILE_SIZE))
    )
    for element_idx in cutlass.range_constexpr(cute.size(pair_bias_fragment)):
        row, column = score_coordinates[element_idx]
        pair_bias_fragment[element_idx] = pair_bias[
            batch_idx, head_idx, query_base + row, key_base + column
        ]
    gradient_sum = score_mma.make_fragment_C(
        score_mma.partition_shape_C((QUERY_TILE_SIZE, KEY_TILE_SIZE))
    )
    gradient_sum.fill(0.0)

    query_partition = mma_thread.partition_A(shared_query)
    key_partition = mma_thread.partition_B(shared_key)
    output_gradient_partition = mma_thread.partition_A(shared_output_gradient)
    value_partition = mma_thread.partition_B(shared_value)
    query_fragment = score_mma.make_fragment_A(query_partition)
    key_fragment = score_mma.make_fragment_B(key_partition)
    output_gradient_fragment = score_mma.make_fragment_A(output_gradient_partition)
    value_fragment = score_mma.make_fragment_B(value_partition)
    query_copy_thread = query_copy.get_slice(thread_idx)
    key_copy_thread = key_copy.get_slice(thread_idx)
    query_copy_source = query_copy_thread.partition_S(shared_query)
    output_gradient_copy_source = query_copy_thread.partition_S(shared_output_gradient)
    key_copy_source = key_copy_thread.partition_S(shared_key)
    value_copy_source = key_copy_thread.partition_S(shared_value)
    query_copy_destination = query_copy_thread.retile(query_fragment)
    output_gradient_copy_destination = query_copy_thread.retile(
        output_gradient_fragment
    )
    key_copy_destination = key_copy_thread.retile(key_fragment)
    value_copy_destination = key_copy_thread.retile(value_fragment)

    sequence = sequence_chunk_idx * SPLIT_SEQUENCE_CHUNK_SIZE
    initial_batch_sequence_idx = batch_idx * sequence_count + sequence
    if thread_idx < 32:
        if thread_idx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(
                activation_ready,
                2
                * (
                    2 * QUERY_TILE_SIZE * HEAD_DIMENSION
                    + 2 * KEY_TILE_SIZE * HEAD_DIMENSION
                ),
            )
        cute.copy(
            query_tma,
            global_query_partition[None, query_block_idx, initial_batch_sequence_idx],
            shared_query_partition[None, 0],
            tma_bar_ptr=activation_ready,
        )
        cute.copy(
            output_gradient_tma,
            global_output_gradient_partition[
                None, query_block_idx, initial_batch_sequence_idx
            ],
            shared_output_gradient_partition[None, 0],
            tma_bar_ptr=activation_ready,
        )
        cute.copy(
            key_tma,
            global_key_partition[None, key_block_idx, initial_batch_sequence_idx],
            shared_key_partition[None, 0],
            tma_bar_ptr=activation_ready,
        )
        cute.copy(
            value_tma,
            global_value_partition[None, key_block_idx, initial_batch_sequence_idx],
            shared_value_partition[None, 0],
            tma_bar_ptr=activation_ready,
        )
    if thread_idx < KEY_TILE_SIZE:
        shared_mask[thread_idx, 0] = residual_mask[
            batch_idx, sequence, key_base + thread_idx
        ]
    if thread_idx < QUERY_TILE_SIZE:
        shared_lse[thread_idx, 0] = logsumexp[
            initial_batch_sequence_idx, head_idx, query_base + thread_idx
        ]
        shared_delta[thread_idx, 0] = delta[
            initial_batch_sequence_idx, head_idx, query_base + thread_idx
        ]

    sequence_iteration = cutlass.Int32(0)
    for _ in cutlass.range_constexpr(SPLIT_SEQUENCE_CHUNK_SIZE):
        stage = sequence_iteration % 2
        activation_phase = (sequence_iteration // 2) % 2
        cute.arch.mbarrier_wait(activation_ready + stage, activation_phase)
        if sequence_iteration == 0:
            cute.arch.sync_threads()

        next_sequence = sequence + 1
        next_stage = next_sequence % 2
        next_batch_sequence_idx = batch_idx * sequence_count + next_sequence
        if sequence_iteration + 1 < SPLIT_SEQUENCE_CHUNK_SIZE:
            if thread_idx < 32:
                if thread_idx == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        activation_ready + next_stage,
                        2
                        * (
                            2 * QUERY_TILE_SIZE * HEAD_DIMENSION
                            + 2 * KEY_TILE_SIZE * HEAD_DIMENSION
                        ),
                    )
                cute.copy(
                    query_tma,
                    global_query_partition[
                        None, query_block_idx, next_batch_sequence_idx
                    ],
                    shared_query_partition[None, next_stage],
                    tma_bar_ptr=activation_ready + next_stage,
                )
                cute.copy(
                    output_gradient_tma,
                    global_output_gradient_partition[
                        None, query_block_idx, next_batch_sequence_idx
                    ],
                    shared_output_gradient_partition[None, next_stage],
                    tma_bar_ptr=activation_ready + next_stage,
                )
                cute.copy(
                    key_tma,
                    global_key_partition[None, key_block_idx, next_batch_sequence_idx],
                    shared_key_partition[None, next_stage],
                    tma_bar_ptr=activation_ready + next_stage,
                )
                cute.copy(
                    value_tma,
                    global_value_partition[
                        None, key_block_idx, next_batch_sequence_idx
                    ],
                    shared_value_partition[None, next_stage],
                    tma_bar_ptr=activation_ready + next_stage,
                )
            if thread_idx < KEY_TILE_SIZE:
                shared_mask[thread_idx, next_stage] = residual_mask[
                    batch_idx, next_sequence, key_base + thread_idx
                ]
            if thread_idx < QUERY_TILE_SIZE:
                shared_lse[thread_idx, next_stage] = logsumexp[
                    next_batch_sequence_idx, head_idx, query_base + thread_idx
                ]
                shared_delta[thread_idx, next_stage] = delta[
                    next_batch_sequence_idx, head_idx, query_base + thread_idx
                ]

        cute.copy(
            query_copy,
            query_copy_source[None, None, None, stage],
            query_copy_destination[None, None, None, stage],
        )
        cute.copy(
            key_copy,
            key_copy_source[None, None, None, stage],
            key_copy_destination[None, None, None, stage],
        )
        cute.copy(
            query_copy,
            output_gradient_copy_source[None, None, None, stage],
            output_gradient_copy_destination[None, None, None, stage],
        )
        cute.copy(
            key_copy,
            value_copy_source[None, None, None, stage],
            value_copy_destination[None, None, None, stage],
        )
        score = score_mma.make_fragment_C(
            score_mma.partition_shape_C((QUERY_TILE_SIZE, KEY_TILE_SIZE))
        )
        probability_gradient = score_mma.make_fragment_C(
            score_mma.partition_shape_C((QUERY_TILE_SIZE, KEY_TILE_SIZE))
        )
        score.fill(0.0)
        probability_gradient.fill(0.0)
        cute.gemm(
            score_mma,
            score,
            query_fragment[None, None, None, stage],
            key_fragment[None, None, None, stage],
            score,
        )
        cute.gemm(
            score_mma,
            probability_gradient,
            output_gradient_fragment[None, None, None, stage],
            value_fragment[None, None, None, stage],
            probability_gradient,
        )

        for element_idx in cutlass.range_constexpr(cute.size(score)):
            row, column = score_coordinates[element_idx]
            probability = cute.math.exp(
                score[element_idx] * SOFTMAX_SCALE
                + pair_bias_fragment[element_idx]
                + shared_mask[column, stage]
                - shared_lse[row, stage],
                fastmath=True,
            )
            gradient_sum[element_idx] += probability * (
                probability_gradient[element_idx] - shared_delta[row, stage]
            )
        cute.arch.sync_threads()
        sequence += 1
        sequence_iteration += 1

    for element_idx in cutlass.range_constexpr(cute.size(gradient_sum)):
        row, column = score_coordinates[element_idx]
        output_offset = (
            ((batch_idx * head_count + head_idx) * context_length + query_base + row)
            * context_length
            + key_base
            + column
        )
        cute.arch.atomic_add(
            pair_bias_gradient.iterator + output_offset,
            gradient_sum[element_idx],
            sem="relaxed",
            scope="gpu",
        )


@cute.jit
def _launch_split_sequence_pair_bias_gradient(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    output_gradient: cute.Tensor,
    pair_bias: cute.Tensor,
    residual_mask: cute.Tensor,
    logsumexp: cute.Tensor,
    delta: cute.Tensor,
    pair_bias_gradient: cute.Tensor,
    stream: cuda.CUstream,
):
    query_view = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[1, 3, 2, 0]))
    key_view = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 3, 2, 0]))
    value_view = cute.make_tensor(v.iterator, cute.select(v.layout, mode=[1, 3, 2, 0]))
    output_gradient_view = cute.make_tensor(
        output_gradient.iterator,
        cute.select(output_gradient.layout, mode=[1, 3, 2, 0]),
    )
    query_layout = sm90_utils.make_smem_layout_a(
        LayoutEnum.ROW_MAJOR,
        (QUERY_TILE_SIZE, KEY_TILE_SIZE, HEAD_DIMENSION),
        a_dtype=cutlass.BFloat16,
        num_stages=2,
    )
    key_layout = sm90_utils.make_smem_layout_b(
        LayoutEnum.ROW_MAJOR,
        (QUERY_TILE_SIZE, KEY_TILE_SIZE, HEAD_DIMENSION),
        b_dtype=cutlass.BFloat16,
        num_stages=2,
    )
    load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    query_tma, query_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        load_op,
        query_view,
        query_layout,
        (QUERY_TILE_SIZE, HEAD_DIMENSION),
    )
    key_tma, key_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        load_op,
        key_view,
        key_layout,
        (KEY_TILE_SIZE, HEAD_DIMENSION),
    )
    value_tma, value_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        load_op,
        value_view,
        key_layout,
        (KEY_TILE_SIZE, HEAD_DIMENSION),
    )
    output_gradient_tma, output_gradient_tensor = (
        cute.nvgpu.cpasync.make_tiled_tma_atom(
            load_op,
            output_gradient_view,
            query_layout,
            (QUERY_TILE_SIZE, HEAD_DIMENSION),
        )
    )
    score_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
        atom_layout_mnk=(4, 4, 1),
        permutation_mnk=(QUERY_TILE_SIZE, KEY_TILE_SIZE, 16),
    )
    query_copy = cute.make_tiled_copy_A(
        cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        ),
        score_mma,
    )
    key_copy = cute.make_tiled_copy_B(
        cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            cutlass.BFloat16,
        ),
        score_mma,
    )
    context_length = pair_bias.shape[2]
    pair_tile_count = (context_length // QUERY_TILE_SIZE) * (
        context_length // KEY_TILE_SIZE
    )
    _split_sequence_pair_bias_gradient_kernel(
        query_tensor,
        key_tensor,
        value_tensor,
        output_gradient_tensor,
        query_tma,
        key_tma,
        value_tma,
        output_gradient_tma,
        pair_bias,
        residual_mask,
        logsumexp,
        delta,
        pair_bias_gradient,
        score_mma,
        query_copy,
        key_copy,
        query_layout,
        key_layout,
    ).launch(
        grid=(
            pair_tile_count,
            pair_bias.shape[0] * pair_bias.shape[1],
            residual_mask.shape[1] // SPLIT_SEQUENCE_CHUNK_SIZE,
        ),
        block=(SPLIT_PAIR_BIAS_THREADS, 1, 1),
        min_blocks_per_mp=1,
        stream=stream,
    )


class _BackwardPipeline:
    """Compile-time fixed CuTe pipeline issued by one host dispatch."""

    def __init__(
        self,
        *,
        direct_query_store: bool,
        score_mod_backward=_evo_score_mod_backward,
    ) -> None:
        self.direct_query_store = direct_query_store
        self.preprocess = _FlashAttentionBackwardPreprocess(
            cutlass.BFloat16,
            HEAD_DIMENSION,
            HEAD_DIMENSION,
            tile_m=QUERY_TILE_SIZE,
            num_threads=ELEMENTWISE_THREADS,
            use_padded_offsets=False,
        )
        self.main = _FlashAttentionBackwardSm90(
            cutlass.BFloat16,
            HEAD_DIMENSION,
            HEAD_DIMENSION,
            qhead_per_kvhead=1,
            tile_m=QUERY_TILE_SIZE,
            tile_n=KEY_TILE_SIZE,
            Q_stage=2,
            dO_stage=2,
            PdS_stage=2,
            SdP_swapAB=True,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=1,
            AtomLayoutNdKV=2,
            AtomLayoutMdQ=1,
            num_threads=384,
            V_in_regs=False,
            score_mod=_evo_score_mod,
            score_mod_bwd=score_mod_backward,
            has_aux_tensors=True,
            dQ_single_wg=True,
        )
        self.postprocess = _FlashAttentionBackwardPostprocess(
            cutlass.BFloat16,
            HEAD_DIMENSION,
            90,
            tile_m=QUERY_TILE_SIZE,
            num_threads=128,
            AtomLayoutMdQ=1,
            dQ_swapAB=False,
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        output: cute.Tensor,
        output_gradient: cute.Tensor,
        logsumexp: cute.Tensor,
        logsumexp_log2: cute.Tensor,
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
            pair_bias_gradient.iterator, cute.make_layout(cute.size(pair_bias_gradient))
        )
        pair_bias_gradient_output_flat = cute.make_tensor(
            pair_bias_gradient_output.iterator,
            cute.make_layout(cute.size(pair_bias_gradient_output)),
        )
        element_count = cute.size(pair_bias_gradient_flat)
        elementwise_grid = (
            cute.ceil_div(element_count, ELEMENTWISE_THREADS * ELEMENTS_PER_THREAD),
            1,
            1,
        )
        _zero_fp32(pair_bias_gradient_flat).launch(
            grid=elementwise_grid,
            block=(ELEMENTWISE_THREADS, 1, 1),
            stream=stream,
        )
        self.preprocess(
            output,
            output_gradient,
            delta,
            logsumexp,
            logsumexp_log2,
            None if self.direct_query_store else query_gradient_accumulator,
            None,
            None,
            None,
            stream,
        )
        self.main(
            q,
            k,
            v,
            output_gradient,
            logsumexp_log2,
            delta,
            query_gradient_accumulator,
            key_gradient,
            value_gradient,
            Float32(SOFTMAX_SCALE),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            [pair_bias, residual_mask, pair_bias_gradient],
            None,
            stream,
        )
        self.postprocess(
            query_gradient_accumulator,
            query_gradient,
            Float32(SOFTMAX_SCALE),
            None,
            None,
            stream,
        )
        _convert_fp32_to_bf16(
            pair_bias_gradient_flat, pair_bias_gradient_output_flat
        ).launch(
            grid=elementwise_grid,
            block=(ELEMENTWISE_THREADS, 1, 1),
            stream=stream,
        )


class _SplitPairBiasBackwardPipeline(_BackwardPipeline):
    """Large-N pipeline with a separate split-sequence dPairBias reduction.

    This remains a distinct decorated callable because sharing one mutable
    CuTe-JIT function between score modifiers corrupts closure preprocessing
    when both runtime branches are compiled into the same artifact.
    """

    def __init__(self) -> None:
        super().__init__(
            direct_query_store=False,
            score_mod_backward=_evo_score_mod_backward_no_dpair_bias,
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        output: cute.Tensor,
        output_gradient: cute.Tensor,
        logsumexp: cute.Tensor,
        logsumexp_log2: cute.Tensor,
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
            pair_bias_gradient.iterator, cute.make_layout(cute.size(pair_bias_gradient))
        )
        pair_bias_gradient_output_flat = cute.make_tensor(
            pair_bias_gradient_output.iterator,
            cute.make_layout(cute.size(pair_bias_gradient_output)),
        )
        element_count = cute.size(pair_bias_gradient_flat)
        elementwise_grid = (
            cute.ceil_div(element_count, ELEMENTWISE_THREADS * ELEMENTS_PER_THREAD),
            1,
            1,
        )
        _zero_fp32(pair_bias_gradient_flat).launch(
            grid=elementwise_grid,
            block=(ELEMENTWISE_THREADS, 1, 1),
            stream=stream,
        )
        self.preprocess(
            output,
            output_gradient,
            delta,
            logsumexp,
            logsumexp_log2,
            query_gradient_accumulator,
            None,
            None,
            None,
            stream,
        )
        self.main(
            q,
            k,
            v,
            output_gradient,
            logsumexp_log2,
            delta,
            query_gradient_accumulator,
            key_gradient,
            value_gradient,
            Float32(SOFTMAX_SCALE),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            [pair_bias, residual_mask, pair_bias_gradient],
            None,
            stream,
        )
        _launch_split_sequence_pair_bias_gradient(
            q,
            k,
            v,
            output_gradient,
            pair_bias,
            residual_mask,
            logsumexp,
            delta,
            pair_bias_gradient,
            stream,
        )
        self.postprocess(
            query_gradient_accumulator,
            query_gradient,
            Float32(SOFTMAX_SCALE),
            None,
            None,
            stream,
        )
        _convert_fp32_to_bf16(
            pair_bias_gradient_flat, pair_bias_gradient_output_flat
        ).launch(
            grid=elementwise_grid,
            block=(ELEMENTWISE_THREADS, 1, 1),
            stream=stream,
        )


class _RuntimeDispatchPipeline:
    """Select all three schedules from runtime H/N in one compiled artifact."""

    def __init__(self) -> None:
        atomic_main = _make_runtime_dq_main(_evo_score_mod_backward)
        self.direct = _BackwardPipeline(direct_query_store=True)
        self.direct.main = atomic_main
        self.fused = _BackwardPipeline(direct_query_store=False)
        self.fused.main = atomic_main
        self.split = _SplitPairBiasBackwardPipeline()
        self.split.main = _make_runtime_dq_main(_evo_score_mod_backward_no_dpair_bias)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        output: cute.Tensor,
        output_gradient: cute.Tensor,
        logsumexp: cute.Tensor,
        logsumexp_log2: cute.Tensor,
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
        if pair_bias.shape[2] == KEY_TILE_SIZE:
            self.direct(
                q,
                k,
                v,
                output,
                output_gradient,
                logsumexp,
                logsumexp_log2,
                delta,
                query_gradient_accumulator,
                query_gradient,
                key_gradient,
                value_gradient,
                pair_bias,
                residual_mask,
                pair_bias_gradient,
                pair_bias_gradient_output,
                stream,
            )
        elif pair_bias.shape[1] == 16 and pair_bias.shape[2] >= 640:
            self.split(
                q,
                k,
                v,
                output,
                output_gradient,
                logsumexp,
                logsumexp_log2,
                delta,
                query_gradient_accumulator,
                query_gradient,
                key_gradient,
                value_gradient,
                pair_bias,
                residual_mask,
                pair_bias_gradient,
                pair_bias_gradient_output,
                stream,
            )
        else:
            self.fused(
                q,
                k,
                v,
                output,
                output_gradient,
                logsumexp,
                logsumexp_log2,
                delta,
                query_gradient_accumulator,
                query_gradient,
                key_gradient,
                value_gradient,
                pair_bias,
                residual_mask,
                pair_bias_gradient,
                pair_bias_gradient_output,
                stream,
            )


@dataclass
class _Workspace:
    logsumexp_log2: torch.Tensor
    query_gradient_accumulator: torch.Tensor


def _dynamic_tensor(
    tensor: torch.Tensor, divisibilities: tuple[int, ...]
) -> cute.Tensor:
    """Make semantic sizes dynamic with shape-independent divisibility facts."""
    result = from_dlpack(tensor.detach(), assumed_align=16, enable_tvm_ffi=True)
    for mode, divisibility in enumerate(divisibilities):
        result = result.mark_compact_shape_dynamic(
            mode=mode,
            stride_order=tensor.dim_order(),
            divisibility=divisibility,
        )
    return result


class EvoAttentionBackward:
    """Lazy compiler and allocation-free steady-state backward launcher."""

    def __init__(self) -> None:
        self._compiled: object | None = None
        self._device_index: int | None = None
        self._workspaces: OrderedDict[
            tuple[int, int, int, int, int, int], _Workspace
        ] = OrderedDict()
        self._prepared: OrderedDict[
            tuple[int, ...], tuple[tuple[object, tuple[torch.Tensor, ...]], ...]
        ] = OrderedDict()
        self._active_shape: tuple[int, ...] | None = None
        self._workspace_bytes = 0
        self._prepared_capacity = MAX_CACHED_BUFFER_SETS
        self._lock = threading.RLock()

    @property
    def compile_count(self) -> int:
        """Return whether the single dynamic backward artifact was compiled."""
        return int(self._compiled is not None)

    @_serialized
    def clear_workspace_cache(self) -> None:
        """Release cached per-buffer launch views and temporary allocations."""
        self._prepared.clear()
        self._workspaces.clear()
        self._active_shape = None
        self._workspace_bytes = 0

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
            raise ValueError("q must have shape [B, S, N, H, 64]")
        batch, sequence_count, context_length, heads, dimension = q.shape
        if sequence_count != context_length:
            raise ValueError("this kernel requires S == N")
        if dimension != HEAD_DIMENSION or context_length % KEY_TILE_SIZE:
            raise ValueError("D must be 64 and N must be divisible by 128")
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
        if pair_bias.shape != pair_shape or pair_bias.dtype != torch.float32:
            raise ValueError("pair_bias must be FP32 [B, H, N, N]")
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
        if (
            residual_mask.shape != (batch, sequence_count, context_length)
            or residual_mask.dtype != torch.float32
        ):
            raise ValueError("residual_mask must be FP32 [B, S, N]")
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
        if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("all tensors must be contiguous CUDA tensors")
        if any(tensor.device != q.device for tensor in tensors):
            raise ValueError("all tensors must reside on the same CUDA device")
        return batch, sequence_count, context_length, heads

    @staticmethod
    def _batch_chunk_size(
        batch: int,
        sequence_count: int,
        context_length: int,
        heads: int,
    ) -> int:
        workspace_bytes_per_batch = (
            sequence_count * heads * context_length * HEAD_DIMENSION * 4
        )
        memory_limit = max(1, MAX_DQ_WORKSPACE_BYTES // workspace_bytes_per_batch)
        grid_limit = max(1, MAX_GRID_Y // (sequence_count * heads))
        return min(batch, memory_limit, grid_limit)

    def _workspace(
        self,
        slot_pointer: int,
        batch_chunk: int,
        sequence_count: int,
        context_length: int,
        heads: int,
        device: torch.device,
    ) -> _Workspace:
        key = (
            device.index if device.index is not None else torch.cuda.current_device(),
            slot_pointer,
            batch_chunk,
            sequence_count,
            context_length,
            heads,
        )
        workspace = self._workspaces.get(key)
        if workspace is None:
            problems = batch_chunk * sequence_count
            workspace_bytes = (
                problems * heads * context_length * (4 + HEAD_DIMENSION * 4)
            )
            while self._workspaces and (
                len(self._workspaces) >= MAX_CACHED_BUFFER_SETS
                or self._workspace_bytes + workspace_bytes > MAX_WORKSPACE_CACHE_BYTES
            ):
                old_key, old_workspace = self._workspaces.popitem(last=False)
                self._workspace_bytes -= sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in (
                        old_workspace.logsumexp_log2,
                        old_workspace.query_gradient_accumulator,
                    )
                )
                old_slot_pointer = old_key[1]
                for prepared_key in tuple(self._prepared):
                    if prepared_key[10] == old_slot_pointer:
                        del self._prepared[prepared_key]
            workspace = _Workspace(
                logsumexp_log2=torch.empty(
                    (problems, heads, context_length),
                    dtype=torch.float32,
                    device=device,
                ),
                query_gradient_accumulator=torch.empty(
                    (problems, heads, context_length * HEAD_DIMENSION),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            self._workspaces[key] = workspace
            self._workspace_bytes += workspace_bytes
        else:
            self._workspaces.move_to_end(key)
        return workspace

    def _compile(
        self,
        tensors: tuple[torch.Tensor, ...],
        cuda_stream: cuda.CUstream,
    ) -> object:
        activation_divisibility = (128, 128, 4)
        statistic_divisibility = (128, 4, 128)
        pair_divisibility = (1, 4, 128, 128)
        mask_divisibility = (1, 128, 128)
        dynamic_divisibilities = (
            *(activation_divisibility,) * 5,
            *(statistic_divisibility,) * 4,
            *(activation_divisibility,) * 3,
            pair_divisibility,
            mask_divisibility,
            pair_divisibility,
            pair_divisibility,
        )
        compile_arguments = tuple(
            _dynamic_tensor(tensor, divisibilities)
            for tensor, divisibilities in zip(
                tensors, dynamic_divisibilities, strict=True
            )
        ) + (cuda_stream,)
        return CompileCallable((PtxasOptions("--maxrregcount=255"), EnableTVMFFI))(
            _RuntimeDispatchPipeline(),
            *compile_arguments,
        )

    @_serialized
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
        device_index = q.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if self._device_index is not None and self._device_index != device_index:
            raise RuntimeError(
                "the cached backward is bound to CUDA device "
                f"{self._device_index}, but received device {device_index}"
            )
        shape = (device_index, *q.shape)
        tensor_key = (device_index,) + tuple(
            tensor.data_ptr()
            for tensor in (
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
            )
        )
        torch_stream = stream or torch.cuda.current_stream(q.device)
        if torch_stream.device != q.device:
            raise ValueError("stream and tensors must reside on the same CUDA device")
        cuda_stream = cuda.CUstream(torch_stream.cuda_stream)
        prepared = self._prepared.get(tensor_key)
        if prepared is not None and shape == self._active_shape:
            self._prepared.move_to_end(tensor_key)
            for compiled, chunk_tensors in prepared:
                chunk_tensors[6].record_stream(torch_stream)
                chunk_tensors[8].record_stream(torch_stream)
                compiled(*chunk_tensors, cuda_stream)
            return

        batch, sequence_count, context_length, heads = self._validate(
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
        )
        if self._active_shape != shape:
            self._prepared.clear()
            self._workspaces.clear()
            self._workspace_bytes = 0
            self._active_shape = shape
            retained_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
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
                )
            )
            self._prepared_capacity = max(
                1,
                min(
                    MAX_CACHED_BUFFER_SETS,
                    MAX_PREPARED_CACHE_BYTES // retained_bytes,
                ),
            )
        batch_chunk = self._batch_chunk_size(
            batch, sequence_count, context_length, heads
        )
        workspace = self._workspace(
            dq.data_ptr(),
            batch_chunk,
            sequence_count,
            context_length,
            heads,
            q.device,
        )
        launches: list[tuple[object, tuple[torch.Tensor, ...]]] = []
        for batch_start in range(0, batch, batch_chunk):
            batch_stop = min(batch, batch_start + batch_chunk)
            problem_count = (batch_stop - batch_start) * sequence_count
            activation_shape = (
                problem_count,
                context_length,
                heads,
                HEAD_DIMENSION,
            )
            chunk_tensors = (
                q[batch_start:batch_stop].view(activation_shape),
                k[batch_start:batch_stop].view(activation_shape),
                v[batch_start:batch_stop].view(activation_shape),
                output[batch_start:batch_stop].view(activation_shape),
                output_gradient[batch_start:batch_stop].view(activation_shape),
                logsumexp[batch_start:batch_stop].view(
                    problem_count, heads, context_length
                ),
                workspace.logsumexp_log2[:problem_count],
                delta[batch_start:batch_stop].view(
                    problem_count, heads, context_length
                ),
                workspace.query_gradient_accumulator[:problem_count],
                dq[batch_start:batch_stop].view(activation_shape),
                dk[batch_start:batch_stop].view(activation_shape),
                dv[batch_start:batch_stop].view(activation_shape),
                pair_bias[batch_start:batch_stop],
                residual_mask[batch_start:batch_stop],
                pair_bias_gradient[batch_start:batch_stop],
                pair_bias_gradient_output[batch_start:batch_stop],
            )
            compiled = self._compiled
            if compiled is None:
                with self._lock:
                    compiled = self._compiled
                    if compiled is None:
                        compiled = self._compile(
                            chunk_tensors,
                            cuda_stream,
                        )
                        self._compiled = compiled
                        self._device_index = device_index
            launches.append((compiled, chunk_tensors))
        prepared = tuple(launches)
        if len(self._prepared) >= self._prepared_capacity:
            self._prepared.popitem(last=False)
        self._prepared[tensor_key] = prepared
        for compiled, chunk_tensors in prepared:
            chunk_tensors[6].record_stream(torch_stream)
            chunk_tensors[8].record_stream(torch_stream)
            compiled(*chunk_tensors, cuda_stream)


_BACKWARD_KERNEL = EvoAttentionBackward()


def get_evoattention_backward() -> EvoAttentionBackward:
    """Return the process-wide cached EvoAttention backward launcher."""
    return _BACKWARD_KERNEL


__all__ = ["EvoAttentionBackward", "get_evoattention_backward"]
