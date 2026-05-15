import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

#######################################################################
##  Multi-part CuTe-DSL kernels building up to TMA + WGMMA usage.
#######################################################################

@cute.kernel
def gemm_v3a(left: cute.Tensor, right: cute.Tensor, answer: cute.Tensor, ## At this point, what is the layout of left, right and answer?
                M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
                tv_lyt_A: cute.Layout, tv_lyt_B: cute.Layout, 
                tv_lyt_C, val_lyt_C: cute.Layout,
                tma_atom_left: cute.CopyAtom, tma_atom_right: cute.CopyAtom,
                BLKM: cutlass.Constexpr, BLKN: cutlass.Constexpr, BLKK: cutlass.Constexpr):
    """
    Uses tv_layouts and TMA to start having a more performant kernel.
    """

    ## Some preprocessing to get the layouts in the right mode. ##
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    gidx, gidy, _ = cute.arch.block_dim()

    alloc = cutlass.utils.SmemAllocator()
    left_lyt = cute.make_layout(shape=(BLKM, BLKK), stride=(BLKK, 1))
    right_lyt = cute.make_layout(shape=(BLKK, BLKN), stride=(BLKN, 1))
    answer_lyt = cute.make_layout(shape=(BLKM, BLKN), stride=(BLKN, 1))

    leftSmem = alloc.allocate_tensor(cutlass.BFloat16, layout=left_lyt, swizzle=None)
    rightSmem = alloc.allocate_tensor(cutlass.BFloat16, layout=right_lyt, swizzle=None)
    answerSmem = alloc.allocate_tensor(cutlass.BFloat16, layout=answer_lyt, swizzle=None)

    ## Next, we initialize the memory bars for TMA. ##
    mbar = alloc.allocate_array(cutlass.Int64, num_elems=1)
    mbar_write = alloc.allocate_array(cutlass.Int64, num_elems=1)

    ## Recover base tid. ##
    tid = tidx + tidy * gidx
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    ## First initialize the memory barrier. ##
    if tid == 0:
        cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init(mbar_write, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    ## Now, we prepare the TMA partition.
    sA_grp = cute.group_modes(leftSmem, 0, 2)
    sB_grp = cute.group_modes(rightSmem, 0, 2)

    tma_left_lyt = cute.local_tile(left, (BLKM, BLKK), (bidy, None))
    tma_left_lyt = cute.group_modes(tma_left_lyt, 0, 2)
    tma_right_lyt = cute.local_tile(right, (BLKK, BLKN), (None, bidx))
    tma_right_lyt = cute.group_modes(tma_right_lyt, 0, 2)

    left_grp = cute.group_modes(tma_left_lyt, 0, 2)
    right_grp = cute.group_modes(tma_right_lyt, 0, 2)

    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_left, (0,), cute.make_layout((1,)), sA_grp, tma_left_lyt
            )

    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_right, (0,), cute.make_layout((1,)), sB_grp, tma_right_lyt 
            )

    ### Next, we get the view for computing the matmul. ##

    ## Now we get compA and compB. Cute layouts for smem -> regs for matmul. ##
    compA = cute.local_tile(leftSmem, (val_lyt_C.shape[0], BLKK), (tidy, 0))
    compB = cute.local_tile(rightSmem, (BLKK, val_lyt_C.shape[-1]), (0, tidx))

    ## Finally, we instantiate the register file to store output. ##
    gC_base = cute.composition(answer[None, (bidy, bidx)], tv_lyt_C) # ((64, 32) : (32, 1)) o tv_layout.

    answerRegs = cute.make_fragment_like(gC_base[(tid, None)], dtype=cutlass.Float32)

    answerRegs.fill(0.0)

    phase = 0
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_left)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_right)

    cute.arch.sync_threads()

    for k in range(tAgA.shape[-1]):

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, (BLKK * BLKM + BLKK * BLKN) * 2)
            cute.copy(tma_atom_left, tAgA[(None, k)], tAsA, tma_bar_ptr=mbar)
            cute.copy(tma_atom_right, tBgB[(None, k)], tBsB, tma_bar_ptr=mbar)

        ## Then we have to wait on the phase of the mbar to flip. ##
        cute.arch.mbarrier_wait(mbar, phase)
        ## Here's let's go ahead and print out the relevant information. ##
        if bidx == 1 and bidy == 1 and tid == 0 and k == 0:
            cute.printf('leftSmem: {}', leftSmem)
            cute.printf('rightSmem: {}', rightSmem)

        ## Finally, we do the gemm here. ##
        ## Only the first 4 warps (tidy 0..7, i.e. tid < 128) participate in
        ## compute/store. Warp 4 (tidy 8,9) is the TMA-only producer warp; if
        ## it falls through here, compA[(tidy, 0)] indexes past leftSmem's 8
        ## M-tiles and gC_base[(tid, None)] wraps around the (16,8) T-mode of
        ## tv_lyt_C, racing valid threads on the same output positions.
        if warp_idx < 4:
            for m_iter in range(val_lyt_C.shape[0]):
                for n_iter in range(val_lyt_C.shape[1]):
                    for k_iter in range(BLKK):
                        answerRegs[((n_iter, m_iter),)] += compA[(m_iter, k_iter)].to(cutlass.Float32) * compB[(k_iter, n_iter)].to(cutlass.Float32)

        cute.arch.sync_threads()  ## For now, we have synchronization over here. ##
        phase ^= 1

    ## finally we store to memory. ##
    if warp_idx < 4:
        gC_base[(tid, None)] = answerRegs.load().to(cutlass.BFloat16)

@cute.kernel
def gemm_v2(left: cute.Tensor, right: cute.Tensor, answer: cute.Tensor,
                M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
                tv_lytA: cute.Layout, tv_lytB: cute.Layout, tv_lytC: cute.Layout, val_C: cute.Layout,
                BLKM: cutlass.Constexpr, BLKN: cutlass.Constexpr, BLKK: cutlass.Constexpr):
    """
    Uses tv_layouts to make separation of concerns palatable.
    This makes loading from gmem -> shmem more efficient.
    """

    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    gidx, gidy, _ = cute.arch.block_dim()

    alloc = cutlass.utils.SmemAllocator()
    left_lyt = cute.make_layout(shape=(BLKM, BLKK), stride=(BLKK, 1))
    right_lyt = cute.make_layout(shape=(BLKK, BLKN), stride=(BLKN, 1))
    answer_lyt = cute.make_layout(shape=(BLKM, BLKN), stride=(BLKN, 1))
    leftSmem = alloc.allocate_tensor(cutlass.BFloat16, layout=left_lyt, swizzle=None)
    rightSmem = alloc.allocate_tensor(cutlass.BFloat16, layout=right_lyt, swizzle=None)

    answerSmem = alloc.allocate_tensor(cutlass.BFloat16, layout=answer_lyt, swizzle=None)

    ## Recover base tid. ##
    tid = tidx + tidy * gidx

    ## What is the implication of using one over the other? In the above? ##

    ## Then, we get this thread's view into smem. ##
    sA = cute.composition(leftSmem, tv_lytA)
    sB = cute.composition(rightSmem, tv_lytB)

    ## Next, we get the view for computing the matmul. ##
    compA = cute.local_tile(leftSmem, (val_C.shape[0], BLKK), (tidy, 0))
    compB = cute.local_tile(rightSmem, (BLKK, val_C.shape[1]), (0, tidx))

    ## For writing, create composition and register file to match view of output. ##
    bC = cute.composition(answer[((None, None), (bidy, bidx))], tv_lytC)

    ## Finally, we make the regsiter file. ##
    answerRegs = cute.make_fragment_like(bC[(tid, None)], cutlass.Float32)

    if tidx == 0 and tidy == 0 and bidx == 0 and bidy == 0:
        cute.printf('regsLayout: {}', answerRegs.layout)
    answerRegs.fill(0.0)

    for k in range(left.shape[1][1]):
        
        ## load into shmem. ##
        bA = left[((None, None), (bidy, k))]
        bB = right[((None, None), (k, bidx))]

        refA = cute.composition(bA, tv_lytA) 
        refB = cute.composition(bB, tv_lytB)
        ## Finally load into smem. ##
        sA[(tid, None)] = refA[(tid, None)].load()
        sB[(tid, None)] = refB[(tid, None)].load()

        cute.arch.sync_threads()

        ## finally, we can do the physical matmul. ##
        for n_iter in range(compB.shape[1]):
            for m_iter in range(compA.shape[0]):
                for k_iter in range(compA.shape[1]):
                    answerRegs[((n_iter, m_iter),)] += compA[(m_iter, k_iter)].to(cutlass.Float32) * compB[(k_iter, n_iter)].to(cutlass.Float32)

        cute.arch.sync_threads()

    ## Lastly, we have to store back to gmem. ##

    bC[(tid, None)] = answerRegs.load().to(cutlass.BFloat16)

@cute.kernel
def gemm_v1(left: cute.Tensor, right: cute.Tensor, answer: cute.Tensor,
            M : cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
            left_mat_layout: cute.Layout, right_mat_layout: cute.Layout,
            answer_layout: cute.Layout):

    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    gidx, gidy, _ = cute.arch.block_dim()

    ## Here we can print out the rudimentary layout. ##

    ## Create shared-memory. ##
    allocator = cutlass.utils.SmemAllocator()

    left_mat_shmem = allocator.allocate_tensor(cutlass.BFloat16, layout=left.layout[0], swizzle=None)
    right_mat_shmem = allocator.allocate_tensor(cutlass.BFloat16, layout=right.layout[0], swizzle=None)

    internal_value = cutlass.Float32(0.0)

    for k in range(left.shape[0][1]):

        ## Here we have to mask out things that aren't compute. We should theoretically do it, but lazy, the next part is tedious, so we do it there. ##
        left_mat_shmem[(tidy, tidx)] = left[((tidy, tidx), (bidx, k))] 
        right_mat_shmem[(tidx, tidy)] = right[((tidx, tidy), (k, bidy))] 

        cute.arch.sync_threads()

        for k_iter in range(left.shape[0][1]):
            ## Then we do a matmul ##
            internal_value += (left_mat_shmem[(tidx, k_iter)].to(cutlass.Float32) * right_mat_shmem[(k_iter, tidy)].to(cutlass.Float32))

        cute.arch.sync_threads()

    ## Then we store answer_shmem to memory. ##
    answer[((tidx, tidy), (bidx, bidy))] = internal_value.to(cutlass.BFloat16)

@cute.jit
def wrapper_gemm_simple(a_ : cute.Tensor, b_: cute.Tensor, c_: cute.Tensor,
                            BLK_M : cutlass.Constexpr, BLK_N: cutlass.Constexpr, BLK_K: cutlass.Constexpr):

    ## This is a simple wrapper, no tv_layouts introduced here. ##

    m, k = a_.shape
    _, n = b_.shape

    #assert m % BLK_M == 0 and n % BLK_N == 0 and k % BLK_K == 0, 'Incorrect tensor sizes/shapes.'

    a_lyt = cute.make_layout(shape=(BLK_M, BLK_K), stride=(BLK_K, 1))

    b_lyt = cute.make_layout(shape=(BLK_K, BLK_N), stride=(BLK_N, 1))

    c_lyt = cute.make_layout(shape=(BLK_M, BLK_N), stride=(BLK_M, 1))

    tA = cute.zipped_divide(a_, (10, 10)) ## Produces a: (REST_M, REST_K), (BLK_M, BLK_K) : (...) cuteified tensor.
    tB = cute.zipped_divide(b_, (10, 10)) ## Produces a: (REST_N, REST_K), (BLK_N, BLK_K) : (...) cuteified tensor.
    tC = cute.zipped_divide(c_, (10, 10)) ## Produces a: (REST_M, REST_N), (BLK_M, BLK_N) : (...) cuteified tensor.

    ## Now, we launch the kernel. ##
    gemm_v1(tA, tB, tC, m, n, k, a_lyt, b_lyt, c_lyt).launch(
              grid=(m // BLK_M, n // BLK_N, 1),
              block=(BLK_M, BLK_N, 1)
              )

@cute.jit
def wrapper_gemm_v2(a_ : cute.Tensor, b_: cute.Tensor, c_: cute.Tensor):
    """
    How should we construct the tv_layouts? 

    Let's do a block computes [64, 32] chunk of data.

    this means we multiply [64, 16] X [16, 32]-sized tiles.

    Then what do the tile sizes look like?

    threads -> 4 * 32 (One warpgroup), easier to change later.
    values -> 16 (dispersed however we want them).
    """
    M, N = c_.shape
    _, K = a_.shape

    ## Let's construct two different tv_layouts. ##

    ## tv_layout for a_. ##
    thr_layout_a = cute.make_layout(shape=(8, 16), stride=(16, 1))
    val_layout_a = cute.make_layout(shape=(8, 1), stride=(1, 0))

    a_tiler, tv_layout_a = cute.make_layout_tv(thr_layout_a, val_layout_a)

    ## tv_layout for b_. ##
    thr_layout_b = cute.make_layout(shape=(8, 16), stride=(16, 1))
    val_layout_b = cute.make_layout(shape=(2, 2), stride=(2, 1))

    b_tiler, tv_layout_b = cute.make_layout_tv(thr_layout_b, val_layout_b)

    ## tv_layout for c_. ##
    thr_layout_c = cute.make_layout(shape=(8, 16), stride=(16, 1))
    val_layout_c = cute.make_layout(shape=(8, 2), stride=(2, 1))

    c_tiler, tv_layout_c = cute.make_layout_tv(thr_layout_c, val_layout_c)

    tA = cute.zipped_divide(a_, a_tiler)
    tB = cute.zipped_divide(b_, b_tiler)
    tC = cute.zipped_divide(c_, c_tiler)

    grid=(cute.size(tC, mode=[1, 1]), cute.size(tC, mode=[1, 0]))

    gemm_v2(tA, tB, tC, M, N, K, tv_layout_a, tv_layout_b, tv_layout_c,
            val_layout_c,c_tiler[0],c_tiler[1],b_tiler[0]).launch(
            grid=(cute.size(tC, mode=[1, 1]), cute.size(tC, mode=[1, 0])),
            block=(16, 8, 1)
            )

    torch.cuda.synchronize()

@cute.jit
def wrapper_gemm_v3a(a_: cute.Tensor, b_: cute.Tensor, c_: cute.Tensor):
    """
    A simple TMA path.
    """
    M, K = a_.shape
    _, N = b_.shape

    ## First we build the TMA atom. ##

    ## To do this, we encode the shmem sizes. ##

    ## We keep smemA = [64, 16] & smemB = [16, 32].
    BLKM, BLKN, BLKK= 64, 32, 16
    smemA_lyt = cute.make_layout(shape=(BLKM, BLKK), stride=(16, 1))
    smemB_lyt = cute.make_layout(shape=(BLKK, BLKN), stride=(32, 1))

    ## Next we build the TMA atom. ##
    op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
    tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op, a_, smemA_lyt, (BLKM, BLKK)
            )
    tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op, b_, smemB_lyt, (BLKK, BLKN)
            )

    ## We still need to manipulate smem right, so we need tv_layouts generated. ##
    thr_A_lyt = cute.make_layout((8, 16))
    val_A_lyt = cute.make_layout((8, 1))

    tiler_a, tv_lyt_a = cute.make_layout_tv(thr_A_lyt, val_A_lyt)

    thr_B_lyt = cute.make_layout((8, 16))
    val_B_lyt = cute.make_layout((2, 2))

    tiler_b, tv_lyt_b = cute.make_layout_tv(thr_B_lyt, val_B_lyt)

    ## We no longer need to TV partition A/B, but let's partition C ##
    ##  and avoid using TMA to store to C. ##
    thr_C_lyt = cute.make_layout(shape=(8, 16), stride=(16, 1))
    val_C_lyt = cute.make_layout(shape=(8, 2), stride=(2, 1))

    tiler_c, tv_c = cute.make_layout_tv(thr_C_lyt, val_C_lyt)

    tC = cute.zipped_divide(c_, tiler_c)

    ## Let's see what things look like here. ##
    print(f'tma_tensor_a: {tma_tensor_a}')
    print(f'tma_tensor_b: {tma_tensor_b}')
    print(f'tma_atom_a: {tma_atom_a}')
    print(f'tiler_c: {tiler_c}, tv_c: {tv_c}, layout_c: {tC}')

    ## We launch the gemm. ##
    gemm_v3a(tma_tensor_a, tma_tensor_b, tC, M, N, K, tv_lyt_a, tv_lyt_b, tv_c, val_C_lyt,
                tma_atom_a, tma_atom_b, BLKM, BLKN, BLKK).launch(
                grid=(cute.size(tC, mode=[1, 1]), cute.size(tC, mode=[1, 0]), 1),
                block=(16, 8 + 2, 1) ## We spawn an extra warp for issuing TMA only.
            )

if __name__ == '__main__':
    ## Here we launch some simple test cases. ##
    torch.manual_seed(42)
    a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
    c = torch.zeros(a.shape[0], b.shape[1], device='cuda', dtype=torch.bfloat16)

    a_ = from_dlpack(a)
    b_= from_dlpack(b)
    c_ = from_dlpack(c)
    BLK_M, BLK_N, BLK_K = 10, 10, 10

    #comp_fn = cute.compile(wrapper_gemm_simple, a_, b_, c_, BLK_M, BLK_N, BLK_K)
    comp_fn = cute.compile(wrapper_gemm_v3a, a_, b_, c_)

    comp_fn(a_, b_, c_)

    #truth = torch.mm(a, b)
    truth = torch.einsum('ab,bc->ac',a,b)
    print(f'a: {a[64:64+64, :16]}, b: {b[:16, 32:32+32]}')

    print(f'out: {c}, truth: {truth}')
    print(f'{c}')
    print(f'is_corect: {torch.allclose(c, truth, atol=1e-2, rtol=1e-2)}')
    print(f'max-value: {torch.abs(c - truth).argmax()}')
    print(f'c: {c[8672 // 128, 8672 % 128]}, truth: {truth[8672 // 128, 8672 % 128]}')
