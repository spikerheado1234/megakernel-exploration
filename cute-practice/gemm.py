import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

@cute.kernel
def gemm_v2(left: cute.Tensor, right: cute.Tensor, answer: cute.Tensor,
                M: cutlass.Int32, N: cutlass.Int32, K: cutlass.Int32,
                tv_lytA: cute.Layout, tv_lytB: cute.Layout, tv_lytC: cute.Layout, thr_C: cute.Layout,
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

    ## Think about why this could go wrong then. ##
    tid = tidx + tidy * gidx
    tid = tidx + tidy * thr_C.shape[0]

    ## What is the implication of using one over the other? In the above? ##

    ## Then, we get this thread's view into smem. ##
    sA = cute.composition(leftSmem, tv_lytA)
    sB = cute.composition(rightSmem, tv_lytB)

    ## Next, we get the view for computing the matmul. ##
    compA = cute.local_partition(leftSmem, thr_C, tid, proj=(1, None))
    compB = cute.local_partition(rightSmem, thr_C, tid, proj=(None, 1))

    ## Finally, we make the regsiter file. ##
    answerRegs = cute.make_fragment(cute.make_layout(shape=(8,2), stride=(2,1)), cutlass.Float32)

    answerRegs.fill(0.0)

    for k in range(left.shape[1][1]):

        ## load into shmem. ##
        bA = left[((None, None), (bidy, k))]
        bB = right[((None, None), (k, bidy))]

        refA = cute.composition(bA, tv_lytA) 
        refB = cute.composition(bB, tv_lytB)
        ## Finally load into smem. ##
        sA[(tid, None)] = refA[(tid, None)].load()
        sB[(tid, None)] = refB[(tid, None)].load()

        cute.arch.sync_threads()

        ## finally, we can do the physical matmul. ##
        for m_iter in range(compA.shape[0]):
            for n_iter in range(compB.shape[1]):
                for k_iter in range(compA.shape[1]):
                    answerRegs[(m_iter, n_iter)] += compA[(m_iter, k_iter)].to(cutlass.Float32) * compB[(k_iter, n_iter)].to(cutlass.Float32)

        cute.arch.sync_threads()

    ## Lastly, we have to store back to gmem. ##

    ## Step one, create composition. ##
    bC = cute.composition(answer[((None, None), (bidy, bidx))], tv_lytC)

    bC[(tid, None)] = answerRegs.load().to(cutlass.BFloat16)  ## is this correct?

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

    gemm_v2(tA, tB, tC, M, N, K, tv_layout_a, tv_layout_b, tv_layout_c,
            thr_layout_c,c_tiler[0],c_tiler[1],b_tiler[0]).launch(
            grid=(cute.size(tC, mode=[1, 0]), cute.size(tC, mode=[1, 1])),
            block=(8, 16, 1)
            )


if __name__ == '__main__':
    ## Here we launch some simple test cases. ##
    a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
    c = torch.zeros(a.shape[0], b.shape[1], device='cuda', dtype=torch.bfloat16)

    a_ = from_dlpack(a)
    b_= from_dlpack(b)
    c_ = from_dlpack(c)
    BLK_M, BLK_N, BLK_K = 10, 10, 10

    #comp_fn = cute.compile(wrapper_gemm_simple, a_, b_, c_, BLK_M, BLK_N, BLK_K)
    comp_fn = cute.compile(wrapper_gemm_v2, a_, b_, c_)

    comp_fn(a_, b_, c_)

    truth = torch.einsum('ab,bc->ac',a,b)

    print(f'out: {c}, truth: {truth}')
    print(f'is_corect: {torch.allclose(c, truth, atol=1e-2, rtol=1e-2)}')
