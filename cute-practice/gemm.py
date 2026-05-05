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
    cute.printf('Layout of left: {}, right: {}, answer: {}', left, right, answer)

    ## Create shared-memory. ##
    allocator = cutlass.utils.ShmemAllocator()

    left_mat_shmem = allocator.allocate_tensor(cutlass.Bfloat16, layout=left_mat_layout[1], swizzle=None)
    right_mat_shmem = allocator.allocate_tensor(cutlass.Bfloat16, layout=right_mat_layout[1], swizzle=None)
    answer_shmem = allocator.allocate_tensor(cutlass.Bfloat16, layout=answer_layout[1], swizzle=None)


    for k in range(left.shape[0][1]):

        left_mat_shmem[(tidy, tidx)] = left[((bidy, bidx), (tidy, tidx))].load()
        right_mat_shmem[(tidx, tidy)] = right[((bidy, bidx), (tidx, tidy))].load()

        cute.arch.syncthreads()

        for k_iter in range(left_mat_layout[1][1]):
            ## Then we do a matmul ##
            answer_shmem[(tidy, tidx)] += left_mat_shmem[(tidy, k_iter)] * right_mat_shmem[(k_iter, tidx)]

        cute.arch.syncthreads()

    ## Then we store answer_shmem to memory. ##
    answer[((bidy, bidx), (tidy, tidx))] = answer_shmem[(tidy, tidx)]



@cute.jit
def wrapper_gemm_simple(a_ : cute.Tensor, b_: cute.Tensor, c_: cute.Tensor,
                            BLK_M : cutlass.Int32, BLK_N: cutlass.Int32, BLK_K: cutlass.Int32):

    ## This is a simple wrapper, no tv_layouts introduced here. ##

    m, k = a_.shape
    _, n = b_.shape

    assert m % BLK_M == 0 and n % BLK_N == 0 and k $ BLK_K == 0, 'Incorrect tensor sizes/shapes.'

    a_lyt = cute.make_layout(shape=(BLK_M, BLK_K), stride=(BLK_K, 1))

    b_lyt = cute.make_layout(shape=(BLK_K, BLK_N), stride=(BLK_N, 1))

    c_lyt = cute.make_layout(shape=(BLK_M, BLK_N), stride=(BLK_M, 1))

    tA = cute.zipped_divide(a_, a_lyt) ## Produces a: (REST_M, REST_K), (BLK_M, BLK_K) : (...) cuteified tensor.
    tB = cute.zipped_divide(b_, b_lyt) ## Produces a: (REST_N, REST_K), (BLK_N, BLK_K) : (...) cuteified tensor.
    tC = cute.zipped_divide(c_, c_lyt) ## Produces a: (REST_M, REST_N), (BLK_M, BLK_N) : (...) cuteified tensor.

    print(f'a_lyt: {a_lyt}, b_lyt: {b_lyt}, c_lyt: {c_lyt}')


    ## Now, we launch the kernel. ##
    gemm_v1(a_, b_, c_, a_lyt, b_lyt, c_lyt).launch(
              grid=(m // BLK_M, n // BLK_N, 1),
              block=(BLK_M, BLK_N, 1)
              )




if __name__ == '__main__':
    ## Here we launch some simple test cases. ##
    a = torch.randn(100, 100, device='cuda')
    b = torch.randn(100, 100, device='cuda')
    c = torch.zeros(a.shape[0], b.shape[1], device='cuda')

    a_ = from_dlpack(a)
    b_= from_dlpack(b)
    c_ = from_dlpack(c)
    BLK_M, BLK_N, BLK_K = 10, 10, 10

    comp_fn = cute.compile(wrapper_gemm_simple, a_, b_, c_, BLK_M, BLK_N, BLK_K)

    comp_fn(a_, b_, c_, BLK_M, BLK_N, BLK_K)
