import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

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
    answer_shmem = allocator.allocate_tensor(cutlass.BFloat16, layout=answer.layout[0], swizzle=None)

    answer_shmem[(tidy, tidx)] = cutlass.BFloat16(0)

    internal_value = cutlass.Float32(0.0)
    cute.arch.sync_threads()

    for k in range(left.shape[0][1]):

        left_mat_shmem[(tidy, tidx)] = left[((tidy, tidx), (bidx, k))] 
        right_mat_shmem[(tidx, tidy)] = right[((tidx, tidy), (k, bidy))] 

        cute.arch.sync_threads()

        for k_iter in range(left.shape[0][1]):
            ## Then we do a matmul ##
            #internal_value += (left_mat_shmem[(tidx, k_iter)] * right_mat_shmem[(k_iter, tidy)]).to(cutlass.Float32)
            internal_value += (left_mat_shmem[(tidx, k_iter)].to(cutlass.Float32) * right_mat_shmem[(k_iter, tidy)].to(cutlass.Float32))
            #answer_shmem[(tidy, tidx)] += left_mat_shmem[(tidy, k_iter)] * right_mat_shmem[(k_iter, tidx)]

        cute.arch.sync_threads()

    ## Then we store answer_shmem to memory. ##
    answer[((tidx, tidy), (bidx, bidy))] = internal_value.to(cutlass.BFloat16)
    #answer[((tidx, tidy), (bidx, bidy))] = answer_shmem[(tidx, tidy)]

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


if __name__ == '__main__':
    ## Here we launch some simple test cases. ##
    a = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(100, 100, device='cuda', dtype=torch.bfloat16)
    c = torch.zeros(a.shape[0], b.shape[1], device='cuda', dtype=torch.bfloat16)

    print(f'a: {a[:10, :10]}, b: {b[:10, :10]}')

    a_ = from_dlpack(a)
    b_= from_dlpack(b)
    c_ = from_dlpack(c)
    BLK_M, BLK_N, BLK_K = 10, 10, 10

    comp_fn = cute.compile(wrapper_gemm_simple, a_, b_, c_, BLK_M, BLK_N, BLK_K)

    ## BLKs removed. ##
    comp_fn(a_, b_, c_)

    truth = torch.einsum('ab,bc->ac',a,b)

    print(f'out: {c}, truth: {truth}')
    print(f'is_corect: {torch.allclose(c, truth, atol=1e-2, rtol=1e-2)}')
