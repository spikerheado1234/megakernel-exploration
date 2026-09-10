## A testing ground for blackwell gemm with tmem. ##
import cutlass
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as blackwell_utils
import torch

@cute.kernel
def gemm_kernel(
    a: cute.Tensor, a_tma: cute.CopyAtom,
    a_s: cute.ComposedLayout, b_s: cute.ComposedLayout,
    b: cute.Tensor, b_tma: cute.CopyAtom,
    c: cute.Tensor,
    tiled_mma: cute.TiledMma,
    m: cutlass.Int32, n: cutlass.Int32, k: cutlass.Int32
):

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, tidy, _ = cute.arch.thread_idx()
    dimx, dimy, _ = cute.arch.block_dim()

    warpidx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    @cute.struct
    class Storage:
        ## For loading inputs. ##
        a_smem: cute.struct.Align[
            cute.struct.MemRange[
                cutlass.BFloat16, cute.cosize(a_s)
            ],
                                    128]

        b_smem: cute.struct.Align[cute.struct.MemRange[
            cutlass.BFloat16, cute.cosize(b_s)
        ],
                                  128]

        c_smem: cute.struct.Align[cute.struct.MemRange[
            cutlass.Float32, 64*64
        ],
                                  128]

        ## Synchronization points. ##
        a_sync: cutlass.Int64
        b_sync: cutlass.Int64
        mma_sync: cutlass.Int64

        ## Tmem pointer. ##
        tmem_ptr: cutlass.Int32 ## Holds the base TMEM pointer. ##
        tmem_dealloc_mbar: cutlass.Int64 ## Used for waiting on Tmem deallocation. ##

    alloc = cutlass.utils.SmemAllocator()
    smem = alloc.allocate(Storage)

    if warpidx == 0:
        ## Then, we have to prefetch
        cute.nvgpu.cpasync.prefetch_descriptor(a_tma)
        cute.nvgpu.cpasync.prefetch_descriptor(b_tma)

    a_prod_sync = smem.a_sync
    b_prod_sync = smem.b_sync
    mma_sync = smem.mma_sync

    cute.arch.sync_threads()

    if tidx == 0:
        ## Here we initialize the mbarrier. ##
        cute.arch.mbarrier_init(a_prod_sync, 1)
        cute.arch.mbarrier_init(b_prod_sync, 1)
        cute.arch.mbarrier_init(mma_sync, 1)

    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    ## Now, we first have to make the cute-tensors. ##
    aS = cute.make_tensor(
        cute.recast_ptr(smem.a_smem.data_ptr(), a_s.inner, dtype=cutlass.BFloat16),
        a_s.outer
    )

    bS = cute.make_tensor(
        cute.recast_ptr(smem.b_smem.data_ptr(), b_s.inner, dtype=cutlass.BFloat16),
        b_s.outer
    )

    cs = cute.make_tensor(
        cute.recast_ptr(smem.c_smem.data_ptr(), dtype=cutlass.Float32),
        cute.make_layout((64, 64), stride=(64, 1))
    )

    ## Next, we prepare the TMA gmem layouts. ##
    ## This is different in blackwell compared to hopper. ##

    aTMA = tiled_mma.get_slice(0).partition_A(
        cute.zipped_divide(a, (64, 64))[((None, None), (bidy, None))]
    )

    bTMA = tiled_mma.get_slice(0).partition_B(
        cute.zipped_divide(b, (64, 64))[((None, None), (bidx, None))]
    )

    ## Next we have to prepare all the TMA partitions. ##
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        a_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(aS, 0, 3), ## Note, this is interestingly grouped from 0 -> 3 as it is organised in MMA partition layotu: (MMA values, rest_M, rest_K, stages)
        cute.group_modes(aTMA, 0, 3)
    )

    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        b_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(bS, 0, 3),
        cute.group_modes(bTMA, 0, 3)
    )

    loop_iter = k // 64
    phaseA = cutlass.Int32(0)
    phaseB = cutlass.Int32(0)
    mmaPhase = cutlass.Int32(0)

    ## Lastly, we prepare the TMEM allocation. ##
    tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
        barrier_id=2,
        num_threads=128
    )

    tmem_alloc = cutlass.utils.TmemAllocator(
        smem.tmem_ptr.ptr,
        tmem_alloc_barrier,
        allocator_warp_id=0
    )

    ## Finally, we call on the allocation. ##

    ## First we have to generate the shape. ##
    mma_tiler_mn = (64, 64)
    tCtC = tiled_mma.partition_shape_C(mma_tiler_mn)
    tCtC_fake = tiled_mma.make_fragment_C(tCtC)
    tmem_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtC_fake)

    tmem_alloc.allocate(tmem_cols)

    tmem_alloc.wait_for_alloc()

    tmem_ptr = tmem_alloc.retrieve_ptr(cutlass.Float32)
    cTmem = cute.make_tensor(
        tmem_ptr, tCtC_fake.layout
    )

    ## Warp 0 owns the TMA loads and UMMA issue path. The other warps are
    ## reserved for the cooperative TMEM epilogue below.
    if warpidx == 0:
        for i in cutlass.range(loop_iter):

            ## First we have to prepare the expected mbarrier bytes. ##
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    a_prod_sync, 64 * 64 * 2
                )
                cute.arch.mbarrier_arrive_and_expect_tx(
                    b_prod_sync, 64 * 64 * 2
                )

            cute.copy(
                a_tma, tAgA[(None, i)], tAsA[(None, 0)], tma_bar_ptr=a_prod_sync
            )
            cute.copy(
                b_tma, tBgB[(None, i)], tBsB[(None, 0)], tma_bar_ptr=b_prod_sync
            )

            ## Wait until both TMA transfers have completed. ##
            cute.arch.mbarrier_wait(a_prod_sync, phaseA)
            cute.arch.mbarrier_wait(b_prod_sync, phaseB)

            tCrA = tiled_mma.make_fragment_A(aS)
            tCrB = tiled_mma.make_fragment_B(bS)
            tCrC = tiled_mma.make_fragment_C(cTmem)

            cute.gemm(
                tiled_mma,
                tCrC,
                tCrA[None, None, None, 0],
                tCrB[None, None, None, 0],
                tCrC
            )

            tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)

            with cute.arch.elect_one():
                cute.nvgpu.tcgen05.commit(
                    mma_sync, None,
                    cute.nvgpu.tcgen05.CtaGroup.ONE
                )

            ## Wait for this UMMA group before reusing the single SMEM stage. ##
            cute.arch.mbarrier_wait(mma_sync, mmaPhase)

            phaseA ^= 1
            phaseB ^= 1
            mmaPhase ^= 1

    ## Prepare the global-memory C tile. The CTA tile and TMEM load tile are both
    ## 64x64, so no additional epilogue sub-tiling is needed.
    cZipped = tiled_mma.get_slice(0).partition_C(
        c[((None, None), (bidy, bidx))]
    )

    ## We will directly build the copy atom. ##
    copy_atom_t2r = cute.make_copy_atom(
        cute.nvgpu.tcgen05.Ld16x256bOp(cute.nvgpu.tcgen05.Repetition(8)),
        cutlass.Float32
    )

    tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
        copy_atom_t2r, cTmem
    )

    thr_copy_t2r = tiled_copy_t2r.get_slice(tidy * dimx + tidx)

    sCopyTmem = thr_copy_t2r.partition_S(cTmem)
    dGmem = thr_copy_t2r.partition_D(cZipped)

    dRmem = cute.make_rmem_tensor(
        dGmem.shape, dtype=cutlass.Float32
    )

    ## Epilogue writeback needs special care. ##
    tmem_alloc.relinquish_alloc_permit()
    cute.arch.sync_threads()
    cute.copy(tiled_copy_t2r, sCopyTmem, dRmem)
    dGmem[None] = dRmem.load().to(cutlass.BFloat16)
    cute.arch.sync_threads()
    ## Finally, we free up the tmem memory. ##
    tmem_alloc.free(tmem_ptr, num_columns=tmem_cols)

@cute.jit
def gemm(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):

    b = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 0]))
    ## First we make the tiled_mma atom. ##
    op = cute.nvgpu.tcgen05.MmaF16BF16Op(
        cutlass.BFloat16,
        cutlass.Float32,
        (64, 64, 16),
        cute.nvgpu.tcgen05.CtaGroup.ONE,
        cute.nvgpu.tcgen05.OperandSource.SMEM,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.MN
    )
    tiled_mma = cute.make_tiled_mma(op)

    ## We will manually build the smem layout. ##
    a_lyt = cutlass.utils.make_smem_layout_a(
        tiled_mma,
        (64, 64, 64),
        a_dtype=cutlass.BFloat16,
        num_stages=1
    )

    b_lyt = cutlass.utils.make_smem_layout_b(
        tiled_mma,
        (64, 64, 64),
        b_dtype=cutlass.BFloat16,
        num_stages=1
    )

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

    mma_tiler_mnk = (64, 64, 64)
    tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        op,
        a,
        a_lyt,
        mma_tiler_mnk,
        tiled_mma
    )

    ## In blackwell we have to use atom_A/atom_B for tma construction. ##
    ## This is the old hopper style of making the template. ##
    #tma_atom_a, tma_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
    #    op,
    #    a,
    #    cute.slice_(a_lyt.outer, (None, None, None, 0)), ## Phase out the stages.
    #    (64, 64),
    #)

    tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        op,
        b,
        b_lyt,
        mma_tiler_mnk,
        tiled_mma
    )

    #tma_atom_b, tma_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
    #    op,
    #    b,
    #    cute.slice_(b_lyt.outer, (None, None, None, 0)), ## Phase out the stages.
    #    (64, 64)
    #)

    ## Now, we have to figure out how to launch. ##
    c_zip = cute.zipped_divide(c, (64, 64))

    blocks_y = cute.size(c_zip.layout, mode=[1, 0])
    blocks_x = cute.size(c_zip.layout, mode=[1, 1])

    gemm_kernel(
        tma_a, tma_atom_a,
        a_lyt, b_lyt,
        tma_b, tma_atom_b,
        c_zip,
        tiled_mma,
        128, 128, 128
    ).launch(
        grid=[blocks_x, blocks_y, 1],
        block=[128, 1, 1]
    )

    torch.cuda.synchronize()


if __name__ == '__main__':
    ## A sample gemm. ##
    torch.manual_seed(0)
    a = torch.randn((128, 128), dtype=torch.bfloat16).to(0)
    b = torch.randn((128, 128), dtype=torch.bfloat16).to(0)
    c = torch.zeros((128, 128), dtype=torch.bfloat16).to(0)
    reference = a.float() @ b.float()
    a_dl = from_dlpack(a, assumed_align=16)
    b_dl = from_dlpack(b, assumed_align=16)
    c_dl = from_dlpack(c, assumed_align=16)

    compiled_gemm = cute.compile(gemm, a_dl, b_dl, c_dl)
    compiled_gemm(a_dl, b_dl, c_dl)

    result = c.float()
    abs_error = (result - reference).abs()
    print(f"max absolute error:  {abs_error.max().item():.6f}")
    print(f"mean absolute error: {abs_error.mean().item():.6f}")
    torch.testing.assert_close(result, reference, rtol=2e-2, atol=2.5e-1)
    print("PASS: CuTe Blackwell GEMM matches torch.matmul")
