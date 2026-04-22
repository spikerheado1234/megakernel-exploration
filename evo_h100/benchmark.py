"""Forward-pass throughput comparison: Triton EvoAttention vs. TK EvoAttention.

Mirrors the shape schedule and FLOP accounting of
~/MegaFold/benchmarks/evoattention_speed.py (forward only).

Prints a table:
    N_CTX    triton (TFLOP/s)    tk (TFLOP/s)

FLOP count (forward):
    QK^T : 2 * B * N_SEQ * H * N_CTX * N_CTX * D     (mul-add)
    P@V  : 2 * B * N_SEQ * H * N_CTX * N_CTX * D
    total = 4 * B * N_SEQ * H * N_CTX^2 * D
(pair-bias add, mask add, and softmax are negligible vs. the two GEMMs.)
"""

import os
import sys

import torch
import triton.testing

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MEGAFOLD = os.path.expanduser("~/MegaFold/megafold/model/FusedEvoAttention")
sys.path.insert(0, MEGAFOLD)

from tk_evo_attention_layer import TKEvoAttention
from evoattention import TritonEvoformer


# ---- Config (matches evoattention_speed.py defaults) ------------------------
BATCH       = 4
N_HEADS     = 16
HEAD_DIM    = 64
N_SEQ       = 1
N_CTX_VALS  = [128, 256, 384, 512, 640, 768, 1024]
DTYPE       = torch.bfloat16
DEVICE      = "cuda"
REP_MS      = 5000
WARMUP_MS   = 200


def fwd_flops(B, N_SEQ, H, N_CTX, D):
    # 2 matmuls of shape (N_CTX, D) * (D, N_CTX) + (N_CTX, N_CTX) * (N_CTX, D)
    # each contributes 2 * B * N_SEQ * H * N_CTX * N_CTX * D
    return 4.0 * B * N_SEQ * H * N_CTX * N_CTX * D


def make_inputs(N_CTX, pair_bias_dtype=torch.float32):
    q = torch.randn((BATCH, N_SEQ, N_CTX, N_HEADS, HEAD_DIM),
                    dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    res_mask_bool = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX),
                                  dtype=torch.bool, device=DEVICE)
    pair_bias = torch.randn((BATCH, 1, N_HEADS, N_CTX, N_CTX),
                            dtype=pair_bias_dtype, device=DEVICE,
                            requires_grad=False)
    return q, k, v, res_mask_bool, pair_bias


def bench_triton(N_CTX):
    q, k, v, res_mask, pair_bias = make_inputs(N_CTX)
    fn = lambda: TritonEvoformer(q, k, v, res_mask, pair_bias)
    fn()  # warm up triton autotune / JIT
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, rep=REP_MS, warmup=WARMUP_MS)
    return ms


def bench_tk(N_CTX):
    # pair_bias in bf16 so the layer's `.to(bf16)` is a no-op. A real caller
    # would keep the bias in the kernel's native dtype between calls rather
    # than recasting fp32->bf16 every forward pass; mirroring that here.
    q, k, v, res_mask, pair_bias = make_inputs(N_CTX, pair_bias_dtype=torch.bfloat16)
    fn = lambda: TKEvoAttention.apply(q, k, v, res_mask, pair_bias)
    fn()  # warm up: first call builds TMA descriptors etc.
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, rep=REP_MS, warmup=WARMUP_MS)
    return ms


def main():
    flops_const = None
    print(f"BATCH={BATCH}  H={N_HEADS}  D={HEAD_DIM}  N_SEQ={N_SEQ}  "
          f"dtype={DTYPE}  device={DEVICE}")
    print()
    header = f"{'N_CTX':>7}  {'triton (TFLOP/s)':>18}  {'tk (TFLOP/s)':>14}"
    print(header)
    print("-" * len(header))

    for n in N_CTX_VALS:
        flops = fwd_flops(BATCH, N_SEQ, N_HEADS, n, HEAD_DIM)

        tri_ms = bench_triton(n)
        tri_tflops = flops * 1e-12 / (tri_ms * 1e-3)

        tk_ms = bench_tk(n)
        tk_tflops = flops * 1e-12 / (tk_ms * 1e-3)

        print(f"{n:>7}  {tri_tflops:>18.2f}  {tk_tflops:>14.2f}")


if __name__ == "__main__":
    main()
