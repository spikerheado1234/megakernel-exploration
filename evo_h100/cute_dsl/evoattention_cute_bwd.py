"""CuTe-DSL EvoAttention backward for NVIDIA Hopper.

Prepared tensor contract:

* q/k/v/output/output_gradient and dQ/dK/dV: ``[B, S, N, H, 64]`` BF16
* pair bias and its accumulator: ``[B, H, N, N]`` FP32
* residual mask: ``[B, S, N]`` FP32 containing ``0`` or ``-1e9``
* logsumexp and delta workspace: ``[B, S, H, N]`` FP32
* final pair-bias gradient: ``[B, H, N, N]`` BF16

The fixed main schedule is M64 x N128 with two consumer warpgroups, two-stage
Q/dO/dS storage, and the probability tile retained in registers. Large
H=16, N>=640 cases use a separate 64 x 128 x 16 split-sequence dPairBias
reduction while the main kernel omits its per-sequence atomics; H=4 retains
the cheaper fused update. Preprocessing, gradient kernels, dQ postprocessing,
and pair-bias initialization/conversion are emitted behind one TVM-FFI call.
B, S, H, and N are runtime dynamic; D=64 is static.

Each ``(batch, sequence, head)`` slice is one dense attention problem with
``logits[i,j] = dot(q[i], k[j]) / sqrt(64) + pair_bias[i,j] + mask[j]``.
The main kernel maps ``batch * sequence`` to its problem axis, recomputes those
modified probabilities, produces dQ/dK/dV, and reduces dS over the sequence
axis to form dPairBias.

The SM90 attention-backward pipeline and all supporting primitives are defined
in this module. Runtime dependencies are limited to PyTorch, CUDA Python, and
the NVIDIA CUTLASS CuTe-DSL package.
"""

import enum
import math
import operator
import threading
from functools import partial, wraps
from typing import Callable, Optional, Tuple, Type, Union

import torch
from cuda.bindings import driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.compiler import CompileCallable, EnableTVMFFI, PtxasOptions
from cutlass.cutlass_dsl import (
    Arch,
    BaseDSL,
    Numeric,
    T,
    dsl_user_op,
)
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
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
MAX_GRID_Y = 65_535
SPLIT_SEQUENCE_CHUNK_SIZE = 16
SPLIT_PAIR_BIAS_THREADS = 512
ACTIVATION_TILE_BYTES = QUERY_TILE_SIZE * HEAD_DIMENSION * 2
KEY_VALUE_TILE_BYTES = KEY_TILE_SIZE * HEAD_DIMENSION * 2
STATISTIC_TILE_BYTES = QUERY_TILE_SIZE * 4
QUERY_GRADIENT_TILE_BYTES = QUERY_TILE_SIZE * HEAD_DIMENSION * 4


def _serialized(method):
    """Serialize launcher metadata updates while allowing queued GPU overlap."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return locked


# Minimal SM90 primitives for the fixed EvoAttention schedule below. These are
# deliberately local and single-purpose: there is no general attention helper
# layer, scheduler framework, score-modifier API, or external kernel source.


def _layout_transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def _layout_select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


def _layout_convert_layout_acc_mn(
    acc_layout: cute.Layout, transpose: bool = False
) -> cute.Layout:
    """Reshape an MMA accumulator into logical row and column modes."""
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    shape = (
        (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
        (
            acc_layout_col_major.shape[0][0],
            *acc_layout_col_major.shape[0][2:],
            acc_layout_col_major.shape[2],
        ),
        *acc_layout_col_major.shape[3:],
    )
    stride = (
        (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
        (
            acc_layout_col_major.stride[0][0],
            *acc_layout_col_major.stride[0][2:],
            acc_layout_col_major.stride[2],
        ),
        *acc_layout_col_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    acc_layout_mn = cute.make_layout(shape, stride=stride)
    return cute.composition(acc_layout, acc_layout_mn)


def _layout_reshape_acc_to_mn(acc: cute.Tensor, transpose: bool = False) -> cute.Tensor:
    return cute.make_tensor(
        acc.iterator, _layout_convert_layout_acc_mn(acc.layout, transpose=transpose)
    )


@cute.jit
def _layout_convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    if const_expr(cute.rank(acc_layout.shape[0]) == 3):
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        l = cute.logical_divide(acc_layout, ((None, None, div), None, None))
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
    else:
        assert acc_layout.shape[2] % 2 == 0
        l = cute.logical_divide(acc_layout, (None, None, 2))
        rA_mma_view = cute.make_layout(
            ((l.shape[0][0], l.shape[0][1], l.shape[2][0]), l.shape[1], l.shape[2][1]),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[2][0]),
                l.stride[1],
                l.stride[2][1],
            ),
        )
    return rA_mma_view


def _layout_reshape_acc_to_frgA(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, _layout_convert_layout_acc_frgA(acc.layout))


def _layout_mma_partition_C_vec(
    sVec: cute.Tensor, thr_mma: cute.core.ThrMma, expand_shape: int, is_colvec: bool
) -> cute.Tensor:
    assert cute.rank(sVec) == 2
    assert sVec.stride[0] == 1
    stage = sVec.shape[1]
    shape = (
        (sVec.shape[0], expand_shape, stage)
        if const_expr(is_colvec)
        else (expand_shape, sVec.shape[0], stage)
    )
    stride = (1, 0, sVec.stride[1]) if const_expr(is_colvec) else (0, 1, sVec.stride[1])
    sVec_mma = cute.make_tensor(sVec.iterator, cute.make_layout(shape, stride=stride))
    tC_sVec = _layout_reshape_acc_to_mn(thr_mma.partition_C(sVec_mma))
    return tC_sVec[None, 0, None] if const_expr(is_colvec) else tC_sVec[0, None, None]


@dsl_user_op
def _copy_load_s2r(src: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    dst = cute.make_rmem_tensor_like(src, src.element_type, loc=loc, ip=ip)
    cute.autovec_copy(src, dst, loc=loc, ip=ip)
    return dst


def _copy_tiled_copy_1d(
    dtype: Type[cutlass.Numeric],
    num_threads: int,
    num_copy_elems: int = 1,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=num_copy_bits
    )
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def _copy_tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    threads_per_row: int,
    num_threads: int,
    num_copy_elems: int = 1,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=num_copy_bits
    )
    assert num_threads % threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // threads_per_row, threads_per_row), order=(1, 0)
    )
    val_layout = cute.make_layout((1, num_copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def _copy_swizzle_int(ptr_int: Int32, b: int, m: int, s: int) -> Int32:
    bit_msk = (1 << b) - 1
    yyy_msk = bit_msk << m + s
    return ptr_int ^ (ptr_int & yyy_msk) >> s


def _copy_swizzle_ptr(ptr: cute.Pointer):
    swz = ptr.type.swizzle_type
    ptr_int = _copy_swizzle_int(ptr.toint(), swz.num_bits, swz.num_base, swz.num_shift)
    return cute.make_ptr(ptr.dtype, ptr_int, ptr.memspace, assumed_align=ptr.alignment)


def _copy_as_position_independent_swizzle_tensor(tensor: cute.Tensor) -> cute.Tensor:
    outer = tensor.layout
    width = tensor.element_type.width
    swizzle_type = tensor.iterator.type.swizzle_type
    inner = cute.make_swizzle(
        swizzle_type.num_bits, swizzle_type.num_base, swizzle_type.num_shift
    )
    new_layout = cute.recast_layout(
        width,
        8,
        cute.make_composed_layout(inner, 0, cute.recast_layout(8, width, outer)),
    )
    return cute.make_tensor(
        cute.recast_ptr(tensor.iterator, dtype=tensor.element_type), new_layout
    )


def _copy_partition_D_position_independent(
    thr_copy: cute.core.ThrCopy, tensor: cute.Tensor
) -> cute.Tensor:
    return cute.make_tensor(
        _copy_swizzle_ptr(thr_copy.partition_D(tensor).iterator),
        thr_copy.partition_D(
            _copy_as_position_independent_swizzle_tensor(tensor)
        ).layout,
    )


def _copy_get_smem_store_C(
    tiled_mma: cute.TiledMma,
    sC: cute.Tensor,
    tidx: Int32,
    transpose: bool = False,
) -> Tuple[Callable, cute.TiledCopy, cute.Tensor]:
    dtype = sC.element_type
    copy_atom = cute.make_copy_atom(
        warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4), dtype
    )
    tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
    thr_copy = tiled_copy.get_slice(tidx)
    tRS_sC = _copy_partition_D_position_independent(thr_copy, sC)

    def copy_fn(src: cute.Tensor, dst_idx: Optional[Int32] = None, **new_kwargs):
        dst_tensor = (
            tRS_sC if const_expr(dst_idx is None) else tRS_sC[None, None, None, dst_idx]
        )
        converted = cute.make_rmem_tensor_like(src, dst_tensor.element_type)
        converted.store(src.load().to(dst_tensor.element_type))
        cute.copy(tiled_copy, tiled_copy.retile(converted), dst_tensor, **new_kwargs)

    return (copy_fn, thr_copy, tRS_sC)


@dsl_user_op
def _copy_cpasync_reduce_bulk_add_f32(
    smem_ptr: cute.Pointer,
    gmem_ptr: cute.Pointer,
    store_bytes: int | Int32,
    *,
    loc=None,
    ip=None,
):
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [gmem_ptr.llvm_ptr, smem_ptr_i32, Int32(store_bytes).ir_value()],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
    )


def _copy_cpasync_bulk_get_copy_fn(
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    **kwargs,
) -> Callable:
    group_rank_src = const_expr(cute.rank(src_tensor) - 1)
    group_rank_dst = const_expr(cute.rank(dst_tensor) - 1)
    src = cute.group_modes(src_tensor, 0, group_rank_src)
    dst = cute.group_modes(dst_tensor, 0, group_rank_dst)

    def copy_bulk(src_idx, dst_idx, tma_bar_ptr: cute.Pointer, **new_kwargs):
        atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), src.element_type)
        with cute.arch.elect_one():
            cute.copy(
                atom,
                src[None, src_idx],
                dst[None, dst_idx],
                mbar_ptr=tma_bar_ptr,
                **new_kwargs,
                **kwargs,
            )

    return copy_bulk


@dsl_user_op
def _copy_tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    single_stage: bool = False,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (
        (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    )
    group_rank_smem = const_expr(
        cute.rank(smem_tensor) - (1 if not single_stage else 0)
    )
    group_rank_gmem = const_expr(
        cute.rank(gmem_tensor) - (1 if not single_stage else 0)
    )
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem),
        loc=loc,
        ip=ip,
    )
    src, dst = (s, g) if src_is_smem else (g, s)

    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **new_kwargs):
        cute.copy(
            atom,
            src[None, src_idx],
            dst[None, dst_idx],
            **new_kwargs,
            **kwargs,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def copy_tma_single_stage(*, loc=None, ip=None, **new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs, loc=loc, ip=ip)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage, s, g)


def _bind_tma_copy_to_pipeline(copy, pipeline):
    """Bind a staged TMA copy to the producer state's stage and barrier."""

    def copy_fn(source_stage, producer_state, **kwargs):
        copy(
            src_idx=source_stage,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **kwargs,
        )

    return copy_fn


@dsl_user_op
def _sm90_make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    tile: cute.Tile,
    stage: Optional[int] = None,
    major_mode_size: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    shape = cute.product_each(cute.shape(tile, loc=loc, ip=ip), loc=loc, ip=ip)
    if const_expr(major_mode_size is None):
        major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(layout, dtype, major_mode_size), dtype
    )
    order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2],
    )
    return smem_layout_staged


@cute.jit
def _sm90_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    if const_expr(swap_AB):
        _sm90_gemm(
            tiled_mma,
            acc,
            tCrB,
            tCrA,
            zero_init=zero_init,
            wg_wait=wg_wait,
            swap_AB=False,
        )
    else:
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
            cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)


def _sm90_gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> cute.Tensor:
    if const_expr(swap_AB):
        return _sm90_gemm_zero_init(
            tiled_mma, shape[::-1], tCrB, tCrA, B_idx, A_idx, wg_wait, swap_AB=False
        )
    else:
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        _sm90_gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
        return acc


def _sm90_gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> None:
    if const_expr(swap_AB):
        _sm90_gemm_w_idx(
            tiled_mma, acc, tCrB, tCrA, zero_init, B_idx, A_idx, wg_wait, swap_AB=False
        )
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        _sm90_gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)


def _sm90_partition_fragment_ABC(
    thr_mma: cute.ThrMma,
    shape_mnk: cute.Shape,
    sA: Optional[cute.Tensor],
    sB: Optional[cute.Tensor],
    swap_AB: bool = False,
):
    is_rs = thr_mma.op.a_src == warpgroup.OperandSource.RMEM
    if const_expr(not swap_AB):
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
        if const_expr(not is_rs):
            assert sA is not None
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        else:
            tCrA = thr_mma.make_fragment_A(
                thr_mma.partition_shape_A((shape_mnk[0], shape_mnk[2]))
            )
        assert sB is not None
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
    else:
        acc = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((shape_mnk[1], shape_mnk[0])), Float32
        )
        if const_expr(not is_rs):
            assert sB is not None
            tCrB = thr_mma.make_fragment_A(thr_mma.partition_A(sB))
        else:
            tCrB = thr_mma.make_fragment_A(
                thr_mma.partition_shape_A((shape_mnk[1], shape_mnk[2]))
            )
        assert sA is not None
        tCrA = thr_mma.make_fragment_B(thr_mma.partition_B(sA))
    return (acc, tCrA, tCrB)


@cute.jit
def _utils_warp_reduce(
    val: cute.TensorSSA | cute.Numeric,
    op: Callable,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_fragment(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = _utils_warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val


@cute.jit
def _utils_shuffle_sync(
    value: cute.Numeric,
    offset: cute.typing.Int,
    width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE,
) -> cute.Numeric:
    assert value.width % 32 == 0, "value type must be a multiple of 32 bits"
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp
    val = cute.make_rmem_tensor(cute.make_layout((1,), stride=(1,)), type(value))
    val[0] = value
    val_i32 = cute.recast_tensor(val, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(val_i32)):
        val_i32[i] = cute.arch.shuffle_sync(
            val_i32[i], offset, mask_and_clamp=mask_and_clamp
        )
    return val[0]


@dsl_user_op
def _pack_bf16_pair(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def _convert_fragment_to_bf16(source: cute.Tensor) -> cute.Tensor:
    """Pack an even-sized FP32 register fragment into BF16 pairs."""
    result = cute.make_fragment(source.shape, cutlass.BFloat16)
    result_i32 = cute.recast_tensor(result, cutlass.Int32)
    for index in cutlass.range_constexpr(cute.size(result_i32)):
        result_i32[index] = _pack_bf16_pair(source[2 * index], source[2 * index + 1])
    return result


@dsl_user_op
def _acquire_tma_stage(
    pipeline,
    state,
    extra_transaction_bytes=0,
    *,
    loc=None,
    ip=None,
):
    """Acquire one stage, including a one-off K/V transfer when requested."""
    pipeline.sync_object_empty.wait(state.index, state.phase, loc=loc, ip=ip)
    if const_expr(extra_transaction_bytes == 0):
        pipeline.sync_object_full.arrive(
            state.index, pipeline.producer_mask, loc=loc, ip=ip
        )
    else:
        pipeline.sync_object_full.arrive_and_expect_tx(
            state.index,
            pipeline.sync_object_full.tx_count + extra_transaction_bytes,
            loc=loc,
            ip=ip,
        )


class _BackwardBarrier(enum.IntEnum):
    EPILOGUE = 1
    SCORE_GRADIENT = 2
    QUERY_GRADIENT_FULL = 3
    QUERY_GRADIENT_EMPTY = 4


@cute.kernel
def _preprocess_backward_kernel(
    output: cute.Tensor,
    output_gradient: cute.Tensor,
    delta: cute.Tensor,
    query_gradient_accumulator: Optional[cute.Tensor],
    activation_copy: cute.TiledCopy,
    accumulator_copy: cute.TiledCopy,
    use_pdl: cutlass.Constexpr[bool],
):
    """Compute delta and optionally clear the FP32 dQ workspace."""
    thread_idx, _, _ = cute.arch.thread_idx()
    query_block, head_idx, problem_idx = cute.arch.block_idx()
    query_base = query_block * QUERY_TILE_SIZE

    if const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()

    output_tile = cute.local_tile(
        output[problem_idx, None, head_idx, None],
        (QUERY_TILE_SIZE, HEAD_DIMENSION),
        (query_block, 0),
    )
    output_gradient_tile = cute.local_tile(
        output_gradient[problem_idx, None, head_idx, None],
        (QUERY_TILE_SIZE, HEAD_DIMENSION),
        (query_block, 0),
    )
    copy_thread = activation_copy.get_slice(thread_idx)
    output_partition = copy_thread.partition_S(output_tile)
    gradient_partition = copy_thread.partition_S(output_gradient_tile)
    output_fragment = cute.make_rmem_tensor_like(output_partition)
    gradient_fragment = cute.make_rmem_tensor_like(gradient_partition)
    cute.copy(activation_copy, output_partition, output_fragment)
    cute.copy(activation_copy, gradient_partition, gradient_fragment)

    if const_expr(use_pdl):
        cute.arch.griddepcontrol_launch_dependents()

    row_products = (
        output_fragment.load().to(Float32) * gradient_fragment.load().to(Float32)
    ).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1))
    row_products = _utils_warp_reduce(
        row_products,
        operator.add,
        width=activation_copy.layout_src_tv_tiled[0].shape[0],
    )
    row_product_fragment = cute.make_rmem_tensor(
        cute.size(output_fragment, mode=[1]), Float32
    )
    row_product_fragment.store(row_products)

    coordinates = copy_thread.partition_S(
        cute.make_identity_tensor((QUERY_TILE_SIZE, HEAD_DIMENSION))
    )
    if coordinates[0, 0, 0][1] == 0:
        for row_slot in cutlass.range(
            cute.size(row_product_fragment), unroll_full=True
        ):
            row = coordinates[0, row_slot, 0][0]
            delta[problem_idx, head_idx, query_base + row] = row_product_fragment[
                row_slot
            ]

    if const_expr(query_gradient_accumulator is not None):
        accumulator_tile = cute.local_tile(
            query_gradient_accumulator[problem_idx, head_idx, None],
            (QUERY_TILE_SIZE * HEAD_DIMENSION,),
            (query_block,),
        )
        accumulator_thread = accumulator_copy.get_slice(thread_idx)
        accumulator_partition = accumulator_thread.partition_S(accumulator_tile)
        zero = cute.make_rmem_tensor_like(accumulator_partition)
        zero.fill(0.0)
        cute.copy(accumulator_copy, zero, accumulator_partition)


@cute.jit
def _launch_backward_preprocess(
    output: cute.Tensor,
    output_gradient: cute.Tensor,
    delta: cute.Tensor,
    query_gradient_accumulator: Optional[cute.Tensor],
    stream: cuda.CUstream,
):
    activation_copy = _copy_tiled_copy_2d(
        cutlass.BFloat16,
        HEAD_DIMENSION // (128 // cutlass.BFloat16.width),
        ELEMENTWISE_THREADS,
        128 // cutlass.BFloat16.width,
    )
    accumulator_copy = _copy_tiled_copy_1d(
        Float32, ELEMENTWISE_THREADS, 128 // Float32.width
    )
    _preprocess_backward_kernel(
        output,
        output_gradient,
        delta,
        query_gradient_accumulator,
        activation_copy,
        accumulator_copy,
        BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90a,
    ).launch(
        grid=(
            output.shape[1] // QUERY_TILE_SIZE,
            output.shape[2],
            output.shape[0],
        ),
        block=(ELEMENTWISE_THREADS, 1, 1),
        stream=stream,
        use_pdl=BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90a,
    )


class _EvoAttentionBackwardMain:
    """Fixed-shape-class SM90 pipeline for dense EvoAttention gradients."""

    def __init__(self) -> None:
        self.dtype = cutlass.BFloat16
        self.tile_hdim = HEAD_DIMENSION
        self.tile_m = QUERY_TILE_SIZE
        self.tile_n = KEY_TILE_SIZE
        self.num_threads = 384
        self.Q_stage = 2
        self.dO_stage = 2
        self.num_wg_mma = 2
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.sQ_layout = _sm90_make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_hdim),
            self.Q_stage,
            self.tile_hdim,
        )
        self.sdO_layout = _sm90_make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_hdim),
            self.dO_stage,
            self.tile_hdim,
        )
        self.sK_layout = _sm90_make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=None,
            major_mode_size=self.tile_hdim,
        )
        self.sV_layout = _sm90_make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), None
        )
        self.sPdS_layout = _sm90_make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=2,
            major_mode_size=64,
        )
        self.sdQaccum_layout = cute.make_layout((self.tile_m * self.tile_hdim, 1))
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            ),
            cute.make_layout((self.num_threads_per_warp_group, 1)),
            cute.make_layout(128 // Float32.width),
        )

    def _get_tiled_mma(self):
        tiled_mma_SdP = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(2, 1, 1),
            tiler_mn=(64, 64),
        )
        tiled_mma_dKV = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(2, 1, 1),
            tiler_mn=(64, 64),
            a_source=warpgroup.OperandSource.RMEM,
        )
        tiled_mma_dQ = sm90_utils.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(1, 1, 1),
            tiler_mn=(64, 64),
        )
        return (tiled_mma_SdP, tiled_mma_dKV, tiled_mma_dKV, tiled_mma_dQ)

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[
                cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout, t in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
                (self.sdQaccum_layout, Float32),
            ]
        ]
        cosize_sdS = cute.cosize(self.sPdS_layout)
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.Q_stage
            ],
            128,
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.dO_stage
            ],
            128,
        ]

        @cute.struct
        class SharedStorageQKV:
            query_barriers: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            output_gradient_barriers: cute.struct.MemRange[
                cutlass.Int64, self.dO_stage * 2
            ]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        pair_bias: cute.Tensor,
        residual_mask: cute.Tensor,
        pair_bias_gradient: cute.Tensor,
        accumulate_pair_bias: cutlass.Constexpr[bool],
        stream: cuda.CUstream = None,
    ):
        def _qkv_transpose(t):
            return _layout_select(
                t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1]
            )

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        LSE_dPsum_dQaccum_transpose = (
            [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        )
        mLSE, mdPsum, mdQaccum = [
            _layout_select(t, LSE_dPsum_dQaccum_transpose)
            for t in (mLSE, mdPsum, mdQaccum)
        ]
        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads
        self.num_threads_per_warp_group = 128
        self.num_mma_regs_wg0 = 256
        self.num_mma_regs_wg1 = 224
        self.num_producer_regs = 24
        assert (
            self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs
            <= 504
        )
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        grid_dim = (
            cute.size(mK.shape[0]) // self.tile_n,
            cute.size(mQ.shape[2]),
            cute.size(mK.shape[3]),
        )
        softmax_scale_log2 = math.log2(math.e)
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK,
            tma_tensor_dV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            SharedStorage,
            pair_bias,
            residual_mask,
            pair_bias_gradient,
            accumulate_pair_bias,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        SharedStorage: cutlass.Constexpr[Callable],
        pair_bias: cute.Tensor,
        residual_mask: cute.Tensor,
        pair_bias_gradient: cute.Tensor,
        accumulate_pair_bias: cutlass.Constexpr[bool],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dK,
                tma_atom_dV,
            ):
                cpasync.prefetch_descriptor(atom)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        query_pipeline = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.query_barriers.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=ACTIVATION_TILE_BYTES + STATISTIC_TILE_BYTES,
            defer_sync=True,
        )
        output_gradient_pipeline = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.output_gradient_barriers.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=ACTIVATION_TILE_BYTES + STATISTIC_TILE_BYTES,
            defer_sync=False,
        )
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage), stride=(1, cute.round_up(self.tile_m, 64))
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)
        n_block, head_idx, problem_idx = cute.arch.block_idx()
        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sQ,
                    sK,
                    sV,
                    sdO,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_dO,
                    query_pipeline,
                    output_gradient_pipeline,
                    n_block,
                    head_idx,
                    problem_idx,
                )
            if warp_idx == 1:
                self.dQaccum_store(
                    mdQaccum,
                    sdQaccum,
                    n_block,
                    head_idx,
                    problem_idx,
                    mQ.shape[0],
                )
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                query_pipeline,
                output_gradient_pipeline,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                n_block,
                head_idx,
                problem_idx,
                mQ.shape[0],
                pair_bias,
                residual_mask,
                pair_bias_gradient,
                accumulate_pair_bias,
            )
            warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
            if warp_idx_in_mma < 4:
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=True)
            else:
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                self.mma(*mma_args, is_dQ_wg=False)

    @cute.jit
    def load(
        self,
        query: cute.Tensor,
        key: cute.Tensor,
        value: cute.Tensor,
        output_gradient: cute.Tensor,
        logsumexp: cute.Tensor,
        delta: cute.Tensor,
        shared_query: cute.Tensor,
        shared_key: cute.Tensor,
        shared_value: cute.Tensor,
        shared_output_gradient: cute.Tensor,
        shared_logsumexp: cute.Tensor,
        shared_delta: cute.Tensor,
        query_tma: cute.CopyAtom,
        key_tma: cute.CopyAtom,
        value_tma: cute.CopyAtom,
        output_gradient_tma: cute.CopyAtom,
        query_pipeline: cutlass.pipeline.PipelineAsync,
        output_gradient_pipeline: cutlass.pipeline.PipelineAsync,
        key_block: Int32,
        head_idx: Int32,
        problem_idx: Int32,
    ):
        query_problem = query[None, None, head_idx, problem_idx]
        key_problem = key[None, None, head_idx, problem_idx]
        value_problem = value[None, None, head_idx, problem_idx]
        output_gradient_problem = output_gradient[None, None, head_idx, problem_idx]
        logsumexp_problem = logsumexp[None, head_idx, problem_idx]
        delta_problem = delta[None, head_idx, problem_idx]

        key_tile = cute.local_tile(
            key_problem, (self.tile_n, self.tile_hdim), (key_block, 0)
        )
        value_tile = cute.local_tile(
            value_problem, (self.tile_n, self.tile_hdim), (key_block, 0)
        )
        query_tiles = cute.local_tile(
            query_problem, (self.tile_m, self.tile_hdim), (None, 0)
        )
        output_gradient_tiles = cute.local_tile(
            output_gradient_problem,
            (self.tile_m, self.tile_hdim),
            (None, 0),
        )
        logsumexp_tiles = cute.local_tile(logsumexp_problem, (self.tile_m,), (None,))
        delta_tiles = cute.local_tile(delta_problem, (self.tile_m,), (None,))

        load_key, _, _ = _copy_tma_get_copy_fn(
            key_tma, 0, cute.make_layout(1), key_tile, shared_key, single_stage=True
        )
        load_value, _, _ = _copy_tma_get_copy_fn(
            value_tma,
            0,
            cute.make_layout(1),
            value_tile,
            shared_value,
            single_stage=True,
        )
        load_query, _, _ = _copy_tma_get_copy_fn(
            query_tma, 0, cute.make_layout(1), query_tiles, shared_query
        )
        load_output_gradient, _, _ = _copy_tma_get_copy_fn(
            output_gradient_tma,
            0,
            cute.make_layout(1),
            output_gradient_tiles,
            shared_output_gradient,
        )
        load_query = _bind_tma_copy_to_pipeline(load_query, query_pipeline)
        load_output_gradient = _bind_tma_copy_to_pipeline(
            load_output_gradient, output_gradient_pipeline
        )
        load_logsumexp = _bind_tma_copy_to_pipeline(
            _copy_cpasync_bulk_get_copy_fn(logsumexp_tiles, shared_logsumexp),
            query_pipeline,
        )
        load_delta = _bind_tma_copy_to_pipeline(
            _copy_cpasync_bulk_get_copy_fn(delta_tiles, shared_delta),
            output_gradient_pipeline,
        )

        producer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        _acquire_tma_stage(
            query_pipeline,
            producer_state,
            extra_transaction_bytes=KEY_VALUE_TILE_BYTES,
        )
        load_key(tma_bar_ptr=query_pipeline.producer_get_barrier(producer_state))
        load_query(0, producer_state=producer_state)
        cute.arch.griddepcontrol_wait()
        load_logsumexp(0, producer_state=producer_state)
        _acquire_tma_stage(
            output_gradient_pipeline,
            producer_state,
            extra_transaction_bytes=KEY_VALUE_TILE_BYTES,
        )
        load_value(
            tma_bar_ptr=output_gradient_pipeline.producer_get_barrier(producer_state)
        )
        load_output_gradient(0, producer_state=producer_state)
        load_delta(0, producer_state=producer_state)
        producer_state.advance()

        query_block_count = query.shape[0] // self.tile_m
        for query_block in cutlass.range(1, query_block_count, unroll=1):
            _acquire_tma_stage(query_pipeline, producer_state)
            load_query(query_block, producer_state=producer_state)
            load_logsumexp(query_block, producer_state=producer_state)
            _acquire_tma_stage(output_gradient_pipeline, producer_state)
            load_output_gradient(query_block, producer_state=producer_state)
            load_delta(query_block, producer_state=producer_state)
            producer_state.advance()

    @cute.jit
    def apply_score_mod(
        self,
        scores: cute.Tensor,
        thr_mma: cute.core.ThrMma,
        problem_idx: Int32,
        head_idx: Int32,
        query_block: Int32,
        key_block: Int32,
        softmax_scale: Float32,
        pair_bias: cute.Tensor,
        residual_mask: cute.Tensor,
    ):
        coordinates = thr_mma.partition_C(
            cute.domain_offset(
                (
                    key_block * self.tile_n,
                    query_block * self.tile_m,
                ),
                cute.make_identity_tensor((self.tile_n, self.tile_m)),
            )
        )
        sequence_count = residual_mask.shape[1]
        batch_idx = problem_idx // sequence_count
        sequence_idx = problem_idx - batch_idx * sequence_count
        for element in cutlass.range_constexpr(cute.size(scores)):
            key_idx = coordinates[element][0]
            query_idx = coordinates[element][1]
            scores[element] = (
                scores[element] * softmax_scale
                + pair_bias[batch_idx, head_idx, query_idx, key_idx]
                + residual_mask[batch_idx, sequence_idx, key_idx]
            )

    @cute.jit
    def accumulate_score_gradient(
        self,
        score_gradient: cute.Tensor,
        thr_mma: cute.core.ThrMma,
        problem_idx: Int32,
        head_idx: Int32,
        query_block: Int32,
        key_block: Int32,
        pair_bias_gradient: cute.Tensor,
        residual_mask: cute.Tensor,
        accumulate_pair_bias: cutlass.Constexpr[bool],
    ):
        if const_expr(accumulate_pair_bias):
            coordinates = thr_mma.partition_C(
                cute.domain_offset(
                    (
                        key_block * self.tile_n,
                        query_block * self.tile_m,
                    ),
                    cute.make_identity_tensor((self.tile_n, self.tile_m)),
                )
            )
            sequence_count = residual_mask.shape[1]
            batch_idx = problem_idx // sequence_count
            context_length = pair_bias_gradient.shape[2]
            for element in cutlass.range_constexpr(cute.size(score_gradient)):
                key_idx = coordinates[element][0]
                query_idx = coordinates[element][1]
                offset = (
                    (batch_idx * pair_bias_gradient.shape[1] + head_idx)
                    * context_length
                    + query_idx
                ) * context_length + key_idx
                cute.arch.atomic_add(
                    pair_bias_gradient.iterator + offset,
                    score_gradient[element],
                    sem="relaxed",
                    scope="gpu",
                )

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        query_pipeline: cutlass.pipeline.PipelineAsync,
        output_gradient_pipeline: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        key_block: Int32,
        head_idx: Int32,
        problem_idx: Int32,
        context_length: Int32,
        pair_bias: cute.Tensor,
        residual_mask: cute.Tensor,
        pair_bias_gradient: cute.Tensor,
        accumulate_pair_bias: cutlass.Constexpr[bool],
        is_dQ_wg: cutlass.Constexpr[bool] = True,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = None
        if const_expr(is_dQ_wg):
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(0))
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tSrQ, tSrK = _sm90_partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=True
        )
        mma_qk_fn = partial(
            _sm90_gemm_zero_init,
            tiled_mma_SdP,
            shape_mnk_S[:2],
            tSrQ,
            tSrK,
            swap_AB=True,
        )
        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdim)
        _, tdPrdO, tdPrV = _sm90_partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=True
        )
        mma_dov_fn = partial(
            _sm90_gemm_zero_init,
            tiled_mma_SdP,
            shape_mnk_dP[:2],
            tdPrdO,
            tdPrV,
            swap_AB=True,
        )
        sdOt = _layout_transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dV, _, tdVrdOt = _sm90_partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, None, sdOt, swap_AB=False
        )
        mma_pdo_fn = partial(_sm90_gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
        sdSt = _layout_transpose_view(sdS)
        sQt = _layout_transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, _, tdKrQt = _sm90_partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=False
        )
        mma_dsq_fn = partial(_sm90_gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        sKt = _layout_transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        mma_dsk_fn = None
        if const_expr(is_dQ_wg):
            _, tdQrdS, tdQrKt = _sm90_partition_fragment_ABC(
                wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=False
            )
            mma_dsk_fn = partial(
                _sm90_gemm_zero_init,
                tiled_mma_dQ,
                shape_mnk_dQ[:2],
                tdQrdS,
                tdQrKt,
                swap_AB=False,
            )
        copy_dS_r2s, _, _ = _copy_get_smem_store_C(
            tiled_mma_SdP,
            sdSt,
            tidx,
            transpose=True,
        )
        tLSEsLSE = _layout_mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=False
        )
        tLSEsdPsum = _layout_mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=False
        )
        shfl_copy = _copy_tiled_copy_1d(
            sLSE.element_type, num_threads=8, num_copy_elems=2
        )
        shuffle_slice = shfl_copy.get_slice(cute.arch.lane_idx() // 4)
        tLSEsLSE = cute.group_modes(shuffle_slice.partition_S(tLSEsLSE), 0, 2)
        tLSEsdPsum = cute.group_modes(shuffle_slice.partition_S(tLSEsdPsum), 0, 2)
        tdQsdQaccum = None
        if const_expr(is_dQ_wg):
            smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)
        PdS_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(_BackwardBarrier.SCORE_GRADIENT),
            num_threads=self.num_mma_threads,
        )
        score_mod_fn = partial(
            self.apply_score_mod,
            thr_mma=thr_mma_SdP,
            softmax_scale=softmax_scale,
            pair_bias=pair_bias,
            residual_mask=residual_mask,
        )
        score_mod_bwd_fn = partial(
            self.accumulate_score_gradient,
            thr_mma=thr_mma_SdP,
            pair_bias_gradient=pair_bias_gradient,
            residual_mask=residual_mask,
            accumulate_pair_bias=accumulate_pair_bias,
        )
        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            copy_dS_r2s=copy_dS_r2s,
            query_pipeline=query_pipeline,
            output_gradient_pipeline=output_gradient_pipeline,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tdQsdQaccum=tdQsdQaccum,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            is_dQ_wg=is_dQ_wg,
        )
        consumer_state = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        score_mod_fn = partial(
            score_mod_fn,
            problem_idx=problem_idx,
            head_idx=head_idx,
            key_block=key_block,
        )
        score_mod_bwd_fn = partial(
            score_mod_bwd_fn,
            problem_idx=problem_idx,
            head_idx=head_idx,
            key_block=key_block,
        )
        dkv_accumulate = False
        for query_block in cutlass.range(context_length // self.tile_m, unroll=1):
            consumer_state = mma_one_m_block_all(
                query_block,
                consumer_state,
                score_mod_fn=score_mod_fn,
                score_mod_bwd_fn=score_mod_bwd_fn,
                dKV_accumulate=dkv_accumulate,
            )
            dkv_accumulate = True
        acc_dK.store(acc_dK.load() * softmax_scale)
        self.epilogue_dKV(
            acc_dV,
            mdV,
            sV,
            acc_dK,
            mdK,
            sK,
            tma_atom_dK,
            tma_atom_dV,
            tiled_mma_dK,
            tiled_mma_dV,
            tidx,
            key_block,
            head_idx,
            problem_idx,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32) -> Float32:
        """Shuffle the statistic from the thread that owns an accumulator row."""
        vecsize = cute.size(tSrS, mode=[0, 0])
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
        return _utils_shuffle_sync(
            tSrS[idx0 + idx1 * vecsize], offset=off * 4 + lane % 4
        )

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state: cutlass.pipeline.PipelineState,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        copy_dS_r2s: Callable,
        query_pipeline: cutlass.pipeline.PipelineAsync,
        output_gradient_pipeline: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tdQsdQaccum: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        is_dQ_wg: cutlass.Constexpr[bool],
        score_mod_fn: Callable,
        score_mod_bwd_fn: Callable,
        dKV_accumulate: Boolean,
    ):
        smem_idx_Q = consumer_state.index
        smem_idx_dO = consumer_state.index
        smem_idx_PdS = smem_idx_Q
        query_pipeline.consumer_wait(
            consumer_state, query_pipeline.consumer_try_wait(consumer_state)
        )
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        tLSErLSE = _copy_load_s2r(tLSEsLSE[None, smem_idx_Q])
        output_gradient_pipeline.consumer_wait(
            consumer_state,
            output_gradient_pipeline.consumer_try_wait(consumer_state),
        )
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)
        score_mod_fn(acc_S, query_block=m_block)
        acc_S_mn = _layout_reshape_acc_to_mn(acc_S, transpose=True)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    (acc_S_mn[r, c] - lse_val) * softmax_scale_log2,
                    fastmath=True,
                )
        tLSErdPsum = _copy_load_s2r(tLSEsdPsum[None, smem_idx_dO])
        tdVrP = _convert_fragment_to_bf16(_layout_reshape_acc_to_frgA(acc_S))
        warpgroup.wait_group(0)
        acc_dP_mn = _layout_reshape_acc_to_mn(acc_dP, transpose=True)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(tLSErdPsum, r, lane_idx)
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)
        score_mod_bwd_fn(acc_dP, query_block=m_block)
        tdKrdS = _convert_fragment_to_bf16(_layout_reshape_acc_to_frgA(acc_dP))
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)
        mma_pdo_fn(
            tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
        )
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()
        if const_expr(is_dQ_wg):
            acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
            output_gradient_pipeline.consumer_release(consumer_state)
            mma_dsq_fn(
                tCrA=tdKrdS,
                B_idx=smem_idx_Q,
                zero_init=not dKV_accumulate,
                wg_wait=1,
            )
            cute.arch.barrier(
                barrier_id=int(_BackwardBarrier.QUERY_GRADIENT_EMPTY),
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            tdQrdQaccum_flat = cute.make_tensor(
                acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape)
            )
            cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(_BackwardBarrier.QUERY_GRADIENT_FULL),
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            warpgroup.wait_group(0)
            query_pipeline.consumer_release(consumer_state)
        else:
            mma_dsq_fn(
                tCrA=tdKrdS,
                B_idx=smem_idx_Q,
                zero_init=not dKV_accumulate,
                wg_wait=1,
            )
            output_gradient_pipeline.consumer_release(consumer_state)
            warpgroup.wait_group(0)
            query_pipeline.consumer_release(consumer_state)
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(_BackwardBarrier.EPILOGUE), num_threads=self.num_mma_threads
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        mdK_cur = mdK[None, None, head_idx, batch_idx]
        mdV_cur = mdV[None, None, head_idx, batch_idx]
        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.tile_hdim), (n_block, 0))
        store_dK, _, _ = _copy_tma_get_copy_fn(
            tma_atom_dK, 0, cute.make_layout(1), sK, gdK, single_stage=True
        )
        store_dV, _, _ = _copy_tma_get_copy_fn(
            tma_atom_dV, 0, cute.make_layout(1), sV, gdV, single_stage=True
        )
        copy_dV_r2s, _, _ = _copy_get_smem_store_C(
            tiled_mma_dV,
            sV,
            tidx,
            transpose=False,
        )
        copy_dK_r2s, _, _ = _copy_get_smem_store_C(
            tiled_mma_dK,
            sK,
            tidx,
            transpose=False,
        )
        cute.arch.cp_async_bulk_wait_group(1, read=True)
        epi_barrier.arrive_and_wait()
        copy_dV_r2s(acc_dV, dst_idx=None)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx == 4:
            store_dV()
            cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(1, read=True)
        epi_barrier.arrive_and_wait()
        copy_dK_r2s(acc_dK, dst_idx=None)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx == 4:
            store_dK()
            cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def dQaccum_store(
        self,
        query_gradient_accumulator: cute.Tensor,
        shared_query_gradient: cute.Tensor,
        key_block: Int32,
        head_idx: Int32,
        problem_idx: Int32,
        context_length: Int32,
    ):
        """Store one dQ contribution or reduce it into the FP32 workspace."""
        query_gradient = query_gradient_accumulator[None, head_idx, problem_idx]
        query_gradient_tiles = cute.local_tile(
            query_gradient,
            (cute.make_layout((self.tile_m * self.tile_hdim, 1)),),
            (None,),
        )

        for query_block in cutlass.range(context_length // self.tile_m, unroll=1):
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier_arrive(
                barrier_id=int(_BackwardBarrier.QUERY_GRADIENT_EMPTY),
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            cute.arch.barrier(
                barrier_id=int(_BackwardBarrier.QUERY_GRADIENT_FULL),
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            with cute.arch.elect_one():
                store_source = shared_query_gradient[None, 0].iterator
                store_destination = query_gradient_tiles[
                    (None, 0), query_block
                ].iterator
                if context_length == self.tile_n:
                    _direct_bulk_store_f32(
                        store_source,
                        store_destination,
                        QUERY_GRADIENT_TILE_BYTES,
                    )
                else:
                    _copy_cpasync_reduce_bulk_add_f32(
                        store_source,
                        store_destination,
                        QUERY_GRADIENT_TILE_BYTES,
                    )
            cute.arch.cp_async_bulk_commit_group()

        cute.arch.cp_async_bulk_wait_group(0, read=True)


@cute.kernel
def _finalize_query_gradient_kernel(
    accumulator: cute.Tensor,
    query_gradient: cute.Tensor,
    scale: Float32,
    tiled_mma: cute.TiledMma,
    accumulator_layout: cute.Layout,
    output_layout: cute.ComposedLayout,
    global_to_shared_copy: cute.TiledCopy,
    shared_to_register_copy: cute.TiledCopy,
    global_store_copy: cute.TiledCopy,
):
    """Reorder one WGMMA accumulator tile into model-layout BF16 dQ."""
    thread_idx, _, _ = cute.arch.thread_idx()
    query_block, head_idx, problem_idx = cute.arch.block_idx()

    allocator = cutlass.utils.SmemAllocator()
    shared_accumulator = allocator.allocate_tensor(
        Float32, accumulator_layout, byte_alignment=1024
    )
    shared_accumulator_flat = cute.make_tensor(
        shared_accumulator.iterator, cute.make_layout(cute.size(shared_accumulator))
    )
    shared_output = cute.make_tensor(
        cute.recast_ptr(shared_accumulator.iterator, dtype=cutlass.BFloat16),
        output_layout,
    )

    accumulator_problem = accumulator[problem_idx, head_idx, None]
    accumulator_tile = cute.local_tile(
        accumulator_problem,
        (QUERY_TILE_SIZE * HEAD_DIMENSION,),
        (query_block,),
    )
    output_problem = query_gradient[problem_idx, None, head_idx, None]
    output_tile = cute.local_tile(
        output_problem,
        (QUERY_TILE_SIZE, HEAD_DIMENSION),
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
        tiled_mma.partition_shape_C((QUERY_TILE_SIZE, HEAD_DIMENSION)), Float32
    )
    accumulator_view = cute.make_tensor(
        accumulator_fragment.iterator, cute.make_layout(shared_fragment.shape)
    )
    cute.autovec_copy(shared_fragment, accumulator_view)
    output_fragment = cute.make_fragment_like(accumulator_fragment, cutlass.BFloat16)
    output_fragment.store((accumulator_fragment.load() * scale).to(cutlass.BFloat16))
    # The BF16 output aliases the FP32 staging buffer. All threads must finish
    # reading the accumulator before any STSM instruction overwrites it.
    cute.arch.barrier()

    register_store = cute.make_tiled_copy_C(
        cute.make_copy_atom(
            warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4),
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
    register_partition = cute.make_fragment_like(shared_partition, cutlass.BFloat16)
    cute.autovec_copy(shared_partition, register_partition)
    cute.copy(
        global_store_copy,
        register_partition,
        store_thread.partition_D(output_tile),
    )


@cute.jit
def _launch_query_gradient_finalize(
    accumulator: cute.Tensor,
    query_gradient: cute.Tensor,
    scale: Float32,
    stream: cuda.CUstream,
):
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        warpgroup.OperandMajorMode.K,
        warpgroup.OperandMajorMode.K,
        Float32,
        atom_layout_mnk=(1, 1, 1),
        tiler_mn=(QUERY_TILE_SIZE, HEAD_DIMENSION),
    )
    accumulator_layout = cute.make_layout((QUERY_TILE_SIZE * HEAD_DIMENSION, 1))
    output_layout = _sm90_make_smem_layout(
        cutlass.BFloat16,
        LayoutEnum.ROW_MAJOR,
        (QUERY_TILE_SIZE, HEAD_DIMENSION),
        major_mode_size=HEAD_DIMENSION,
    )
    vector_bytes = 16
    accumulator_copy = cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
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
    output_copy = _copy_tiled_copy_2d(
        cutlass.BFloat16,
        threads_per_row=8,
        num_threads=128,
        num_copy_elems=8,
    )
    smem_size = max(
        cute.size_in_bytes(Float32, accumulator_layout),
        cute.size_in_bytes(cutlass.BFloat16, output_layout),
    )
    _finalize_query_gradient_kernel(
        accumulator,
        query_gradient,
        scale,
        tiled_mma,
        accumulator_layout,
        output_layout,
        accumulator_copy,
        shared_copy,
        output_copy,
    ).launch(
        grid=(
            query_gradient.shape[1] // QUERY_TILE_SIZE,
            query_gradient.shape[2],
            query_gradient.shape[0],
        ),
        block=(128, 1, 1),
        smem=smem_size,
        stream=stream,
    )


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


class _BackwardLaunch:
    """Issue the complete backward pass from one TVM-FFI dispatch."""

    def __init__(self) -> None:
        self.main = _EvoAttentionBackwardMain()

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
                ELEMENTWISE_THREADS * ELEMENTS_PER_THREAD,
            ),
            1,
            1,
        )
        _zero_fp32(pair_bias_gradient_flat).launch(
            grid=elementwise_grid,
            block=(ELEMENTWISE_THREADS, 1, 1),
            stream=stream,
        )

        if pair_bias.shape[2] == KEY_TILE_SIZE:
            _launch_backward_preprocess(output, output_gradient, delta, None, stream)
        else:
            _launch_backward_preprocess(
                output,
                output_gradient,
                delta,
                query_gradient_accumulator,
                stream,
            )

        if pair_bias.shape[1] == 16 and pair_bias.shape[2] >= 640:
            self.main(
                q,
                k,
                v,
                output_gradient,
                logsumexp,
                delta,
                query_gradient_accumulator,
                key_gradient,
                value_gradient,
                Float32(SOFTMAX_SCALE),
                pair_bias,
                residual_mask,
                pair_bias_gradient,
                False,
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
        else:
            self.main(
                q,
                k,
                v,
                output_gradient,
                logsumexp,
                delta,
                query_gradient_accumulator,
                key_gradient,
                value_gradient,
                Float32(SOFTMAX_SCALE),
                pair_bias,
                residual_mask,
                pair_bias_gradient,
                True,
                stream,
            )

        _launch_query_gradient_finalize(
            query_gradient_accumulator,
            query_gradient,
            Float32(SOFTMAX_SCALE),
            stream,
        )
        _convert_fp32_to_bf16(
            pair_bias_gradient_flat,
            pair_bias_gradient_output_flat,
        ).launch(
            grid=elementwise_grid,
            block=(ELEMENTWISE_THREADS, 1, 1),
            stream=stream,
        )


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
        self._workspace_key: tuple[int, int, int, int, int] | None = None
        self._query_gradient_accumulator: torch.Tensor | None = None
        self._lock = threading.RLock()

    @property
    def compile_count(self) -> int:
        """Return whether the single dynamic backward artifact was compiled."""
        return int(self._compiled is not None)

    @_serialized
    def clear_workspace_cache(self) -> None:
        """Release the cached FP32 dQ accumulation workspace."""
        self._workspace_key = None
        self._query_gradient_accumulator = None

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
        batch_chunk: int,
        sequence_count: int,
        context_length: int,
        heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (
            device.index if device.index is not None else torch.cuda.current_device(),
            batch_chunk,
            sequence_count,
            context_length,
            heads,
        )
        if self._workspace_key != key:
            problems = batch_chunk * sequence_count
            self._query_gradient_accumulator = torch.empty(
                (problems, heads, context_length * HEAD_DIMENSION),
                dtype=torch.float32,
                device=device,
            )
            self._workspace_key = key
        assert self._query_gradient_accumulator is not None
        return self._query_gradient_accumulator

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
            *(statistic_divisibility,) * 3,
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
            _BackwardLaunch(),
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
        torch_stream = stream or torch.cuda.current_stream(q.device)
        if torch_stream.device != q.device:
            raise ValueError("stream and tensors must reside on the same CUDA device")
        cuda_stream = cuda.CUstream(torch_stream.cuda_stream)
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
        batch_chunk = self._batch_chunk_size(
            batch, sequence_count, context_length, heads
        )
        workspace = self._workspace(
            batch_chunk,
            sequence_count,
            context_length,
            heads,
            q.device,
        )
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
                delta[batch_start:batch_stop].view(
                    problem_count, heads, context_length
                ),
                workspace[:problem_count],
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
            workspace.record_stream(torch_stream)
            compiled(*chunk_tensors, cuda_stream)


_BACKWARD_KERNEL = EvoAttentionBackward()


def get_evoattention_backward() -> EvoAttentionBackward:
    """Return the process-wide cached EvoAttention backward launcher."""
    return _BACKWARD_KERNEL


__all__ = ["EvoAttentionBackward", "get_evoattention_backward"]
