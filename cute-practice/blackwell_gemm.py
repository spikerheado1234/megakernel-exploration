## A testing ground for blackwell gemm with tmem. ##
import cutlass
from cutlass.cute.runtime import from_dlpack
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as blackwell_utils
import torch

@cute.kernel
def gemm_kernel(
    a: cute.Tensor, a_tma: cute.CopyAtom,
    a_smem: cute.ComposedLayout, b_smem: cute.ComposedLayout,
    b: cute.Tensor, b_tma: cute.CopyAtom,
    c: cute.Tensor,
    tiled_mma: cute.TiledMma,
    m: cutlass.Int32, n: cutlass.Int32, k: cutlass.Int32
):

    bidx, bidy, _ = cute.arch.block_idx()
    tidx, tidy, _ = cute.arch.thread_idx()
    dimx, dimy, _ = cute.arch.block_dim()

    warpidx = cute.arch.warp_idx()

    @cute.struct
    class Storage:
        ## For loading inputs. ##
        a_smem: cute.struct.Align[
            cute.struct.MemRange[
                cutlass.BFloat16, cute.cosize(a_smem)
            ],
                                    128]

        b_smem: cute.struct.Align[cute.struct.MemRange[
            cutlass.BFloat16, cute.cosize(b_smem)
        ],
                                  128]

        c_smem: cute.struct.Align[cute.struct.MemRange[
            cutlass.BFloat16, 64*64
        ],
                                  128]

        ## Synchronization points. ##
        a_sync: cutlass.Int64
        b_sync: cutlass.Int64

        ## Tmem pointer. ##
        tmem_ptr: cutlass.Int32 ## Holds the base TMEM pointer. ##
        tmem_dealloc_mbar: cutlass.Int64 ## Used for waiting on Tmem deallocation. ##

    alloc = cutlass.utils.SmemAllocator()
    smem = alloc.allocate(Storage)

    if warpidx == 0:
        ## Then, we have to prefetch
        cute.arch.nvgpu.cpasync.prefetch_descriptor(a_tma)
        cute.arch.nvgpu.cpasync.prefetch_descriptor(b_tma)

    a_prod_sync = smem.a_sync.data_ptr()
    b_prod_sync = smem.b_sync.data_ptr()

    cute.arch.sync_threads()

    if tidx == 0:
        ## Here we initialize the mbarrier. ##
        cute.arch.mbarrier_init(a_prod_sync, 1)
        cute.arch.mbarrier_init(b_prod_sync, 1)

    ## Now, we first have to make the cute-tensors. ##
    aS = cute.make_tensor(
        cute.recast_ptr(smem.a_smem.data_ptr(), a_lyt.inner, dtype=cutlass.BFloat16),
        a_lyt.outer
    )

    bS = cute.make_tensor(
        cute.recast_ptr(smem.b_smem.data_ptr(), b_lyt.inner, dtype=cutlass.BFloat16),
        b_lyt.outer
    )

    cs = cute.make_tensor(smem.c_smem.iterator, cute.make_layout(64, 64), dtype=cutlass.BFloat16)

    ## Next, we prepare the TMA gmem layouts. ##

    aTMA = cute.zipped_divide(a, (64, 64))
    aTMA = aTMA[((None), (bidy, None))]

    bTMA = cute.zipped_divide(b, (64, 64))
    bTMA = bTMA[((None), (None, bidx))]

    ## Next we have to prepare all the TMA partitions. ##
    tAsA, tAgA = cute.tma_partition(
        a_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(aS, 0, 2),
        aTMA
    )

    tBsB, tBgB = cute.tma_partition(
        b_tma,
        0,
        cute.make_layout(1),
        cute.group_modes(bS, 0, 2),
        bTMA
    )

    loop_iter = k // 64
    phaseA = cutlass.Int32(0)
    phaseB = cutlass.Int32(0)

    ## Lastly, we prepare the TMEM allocation. ##
    tmem_alloc_barrier = cutlass.pipeline.NamedBarrier(
        barrier_id=0,
        num_threads=32 ## Single warp in the end. ##
    )

    tmem_alloc = cutlass.utils.TmemAllocator(
        smem.tmem_ptr.ptr(), ## Should I use data_ptr or ptr? Probably ptr as it will be overwritten. ##
        tmem_alloc_barrier,
        allocator_warp_id=0
    )

    ## Finally, we call on the allocation. ##
    tmem_alloc.allocate(64)

    tmem_alloc.wait_for_alloc()

    tmem_ptr = tmem_alloc.retrieve_ptr(cutlass.Float32)
    cTmem = cute.make_tensor(tmem_ptr, cute.make_layout(64, 64))

    ## Now this should be the main loop. ##
    for i in cutlass.range(loop_iter):

        ## First we have to prepare the expect mbarriere bytes. ##
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(a_prod_sync, 64 * 64 * 2)
            cute.arch.mbarrier_expect_tx(b_prod_sync, 64 * 64 * 2)

        cute.copy(
            a_tma, tAgA[(None, i)], tAsA[(None, 0)], tma_bar_ptr=a_prod_sync
        )
        cute.copy(
            b_tma, tBgB[(None, i)], tBsB[(None, 0)], tma_bar_ptr=b_prod_sync
        )

        ## Next, the compute warp. ##

        ## First: wait on the current memory txs to finish. ##
        cute.arch.mbarrier_wait(a_prod_sync, phaseA)
        cute.arch.mbarrier_wait(b_prod_sync, phaseB)

        ## Second, prepare the partitions. ##
        thr_mma = tiled_mma.get_slice(tidy * dimx + tidx)
        tArA = tiled_mma.make_fragment_A(
            thr_mma.partition_A(aS)
        )
        tBrB = tiled_mma.make_fragment_B(
            thr_mma.partition_B(bS)
        )
        ## This has to shift to TMem. ##
        tCrC = tiled_mma.make_fragment_C(
            thr_mma.partition_C(cTmem)
        )

        ## Now we have to launch the gemm. ##
        cute.gemm(
            tiled_mma,
            tCrC,
            tArA[None, None, None, 0],
            tBrB[None, None, None, 0],
            tCrC
        )

        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

        ## Now, what we do here. ##

        a_prod_sync ^= 1
        b_prod_sync ^= 1


    ## Finally we have to take from tmem -> registers -> store to gmem. ##
    epi_tile = cute.make_layout((64, 64), stride=(64, 1))

    ## We will directly build the copy atom. ##
    copy_atom_t2r = cute.make_copy_atom(
        cute.tcgen05.Ld32x32bOp(cute.tcgen05.Repetitionx4),
        cutlass.Float32
    )

    tiled_copy_t2r = cute.tcgen05.make_tmem_copy(
        copy_atom_t2r, tCrC
    )

    thr_copy_t2r = tiled_copy_t2r.get_slice(tidy * dimx + tidx)

    sCopyTmem = thr_copy_t2r.partition_S(tCrC)

    dCopySmem = thr_copy_t2r.partition_D(smem.c_smem)

    op = cute.nvgpu.CopyUniversalOp()

    cute.copy(op, sCopyTmem, dCopySmem)

    ## Finally, we free up the tmem memory. ##
    tmem_alloc.free(smem.tmem_ptr.ptr, num_columns=64)

    ## Finally, we can store from smem -> gmem! ##

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

    import pdb
    pdb.set_trace()

    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()

    tma_atom_a, tma_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op,
        a,
        cute.slice_(a_lyt.outer, (None, None, None, 0)), ## Phase out the stages.
        (64, 64),
    )

    tma_atom_b, tma_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op,
        b,
        cute.slice_(b_lyt.outer, (None, None, None, 0)), ## Phase out the stages.
        (64, 64)
    )

    ## Now, we have to figure out how to launch. ##
    tma_atom_a = cute.zipped_divide(tma_atom_a, (64, 64))
    tma_atom_b = cute.zipped_divide(tma_atom_b, (64, 64))
    c = cute.zipped_divide(c, (64, 64))

    blocks_y = cute.cosize(c.layout, mode=[1, 0])
    blocks_x = cute.cosize(c.layout, mode=[1, 1])

    gemm_kernel(
        tma_a, tma_atom_a,
        a_lyt, b_layt,
        tma_b, tma_atom_b,
        c,
        tiled_mma,
        128, 128, 128
    ).launch(
        grid=[blocks_x, blocks_y, 1],
        block=[32, 1, 1] ## A single warp per CTA schedule. Let's see if we can get this correct!
    )

    torch.cuda.synchronize()


if __name__ == '__main__':
    ## A sample gemm. ##
    a = torch.randn((128, 128)).to(0)
    b = torch.randn((128, 128)).to(0)
    c = torch.zeros((128, 128)).to(0)
    a_dl = from_dlpack(a, assumed_align=16)
    b_dl = from_dlpack(b, assumed_align=16)
    c_dl = from_dlpack(c, assumed_align=16)

    compiled_gemm = cute.compile(gemm, a_dl, b_dl, c_dl)
    compiled_gemm(a_dl, b_dl, c_dl)
