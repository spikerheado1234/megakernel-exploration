## Simple hello world

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

@cute.kernel
def hello_world():
    tidx, _, _ = cute.arch.thread_idx()

    if tidx == 0:
        cute.printf("Hello World!")

@cute.kernel
def printer(a : cutlass.Int32, b: cutlass.Constexpr[int]):
    print('static printing a: ', a)
    print('static printing b: ', b)
    cute.printf("dynamic printing a: {}", a)
    cute.printf("static printing b: {}", b)

## Though this is incorrect, it passes the test cases, so no idea what's going on. ##
@cute.kernel
def adder(one: cute.Tensor, two: cute.Tensor,
            three: cute.Tensor, size: cutlass.Int32):

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gidx, _, _ = cute.arch.block_dim()

    global_idx = gidx*bidx + tidx;

    print(f'one layout: {one}')
    print(f'two layout: {two}')

    ## Then extract the current shape. ##
    m,n = one.shape[1]

    print(f'one shape: {one.shape}')

    mi = global_idx // n
    ni = global_idx % n

    fragA = one[(None, (mi, ni))].load()
    fragB = two[(None, (mi, ni))].load()

    ## TODO: investigate why this also seems to work. ##
    #fragA = one[(None, global_idx)].load()
    #fragB = two[(None, global_idx)].load()

    three[(None, global_idx)] = fragA + fragB

## A vectorised add using cute compositions. ##
@cute.kernel
def adder_composition(a: cute.Tensor, b: cute.Tensor, answer: cute.Tensor, tv_layout: cute.Layout):

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gidx, _, _ = cute.arch.block_dim()

    print(f'a layout: {a}')
    print(f'b layout: {b}')
    print(f'a shape: {a.shape}')
    print(f'b shape: {b.shape}')

    ## This prints out per thread, block information. ##
    #cute.printf('tidx: {}, bidx: {}', tidx, bidx)

    ## We index this manually. ##

    ## We unpack the bidx. ##
    m, n = a.shape[1]
    mi = bidx // n
    ni = bidx % n

    idx = ((None, None), (mi, ni))

    ## I don't think cute composition is needed here. ##
    cA = cute.composition(a[idx], tv_layout)
    cB = cute.composition(b[idx], tv_layout)
    cC = cute.composition(answer[idx], tv_layout)

    print(f'a[idx]: {a[idx]} composition cA: {cA}')

    ## Alternative cute-composition method. ##

    cC[(tidx, None)] = cA[(tidx, None)].load() + cB[(tidx, None)].load()

@cute.jit
def wrapper(a_: cute.Tensor, b_: cute.Tensor, answer_: cute.Tensor):
    cutlass.cuda.initialize_cuda_context()
    hello_world().launch(
            grid=(1,1,1),block=(1,1,1)
            )

    printer(cutlass.Int32(10), 12).launch(
            grid=(1,1,1), block=(1,1,1)
            )

    ## Now let's do some fun things. ##

    ## Here is a sample vectorized addition kernel. ##

    gA = cute.zipped_divide(a_, (1, 8)) ## Layout here: (1, 8),(16, 2) : (0, 1),(16,8). 
    gB = cute.zipped_divide(b_, (1, 8)) ## Same layout as ^.
    answer = cute.zipped_divide(answer_, (1, 8)) ## Same layout as ^.

    thread_cnt = 16

    size = torch.numel(one)

    ## now how do we compute grid size? [one.shape(0) // (thread_cnt * mode[0].size())] + 1

    ## Alternatively, we can also do: [cute.size(gC, mode=[1]) // thread_cnt + 1] --> How does this make sense?

    ## Then, how do we index within the tensor? use both block and thread-idxs.

    adder(gA, gB, answer, cutlass.Int32(one.shape[0])).launch(
            grid=((size // (thread_cnt * cute.size(gA, mode=[0])) + 1),1,1),
            block=(thread_cnt, 1, 1)
            )


@cute.jit
def wrapper_tv(a_: cute.Tensor, b_: cute.Tensor, c_: cute.Tensor):
    thr_layout = cute.make_layout((4, 32), stride=(32, 1))
    val_layout = cute.make_layout((4, 8), stride=(8, 1))

    ## So what excatly is tiler_mn and tv_layout and their relationship? ##
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    print(f'tiler_mn: {tiler_mn}, tv_layout: {tv_layout}')

    gA = cute.zipped_divide(a_, tiler_mn)
    gB = cute.zipped_divide(b_, tiler_mn)
    gC = cute.zipped_divide(c_, tiler_mn)

    print(f'grid size: {cute.size(gA, mode=[1])}')

    adder_composition(gA, gB, gC, tv_layout).launch(
            grid=[cute.size(gA, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1]
            )

if __name__ == '__main__':
    one = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    two = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    answer = torch.zeros_like(one)

    a_ = from_dlpack(one)
    b_ = from_dlpack(two)
    answer_ = from_dlpack(answer)

    comp_func = cute.compile(wrapper_tv, a_, b_, answer_)

    ## Timing in case it's needed. ##
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    ## Reset data to remove l2 cache pollution. ##
    one = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    two = torch.randn(1024, 1024, dtype=torch.bfloat16, device="cuda")
    answer = torch.zeros_like(one)

    a_ = from_dlpack(one)
    b_ = from_dlpack(two)
    answer_ = from_dlpack(answer)

    comp_func(a_, b_, answer_)

    print(f'is close: {torch.allclose(one+two, answer)}')
    print('terminated successful!')

