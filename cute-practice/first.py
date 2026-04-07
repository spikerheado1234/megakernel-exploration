## Simple hello world

import cutlass
import cutlass.cute as cute

@cute.kernel
def hello_world():
    tidx, _, _ = cute.arch.thread_idx()

    if tidx == 0:
        cute.printf("Hello World!")


@cute.jit
def wrapper():
    cutlass.cuda.initialize_cuda_context()
    hello_kernel().launch(
            grid=(1,1,1),block=(1,1,1)
            )


if __name__ == '__main__':
    comp_func = cute.compile(wrapper)

    comp_func()
