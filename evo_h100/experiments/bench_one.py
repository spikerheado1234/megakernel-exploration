"""Bench a single (B, N_CTX, candidate). Fast iteration.

Usage:
    python3 bench_one.py B N_CTX cand_name
"""

import os
import sys

import torch
import triton.testing

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

MEGAFOLD = os.path.expanduser("~/MegaFold/megafold/model/FusedEvoAttention")
sys.path.insert(0, MEGAFOLD)

from tk_fwd_layer import load_candidate, make_fwd_apply
from tk_evo_attention_layer import TKEvoAttention
from evoattention import TritonEvoformer


N_HEADS  = 16
HEAD_DIM = 64
N_SEQ    = 1


def main(B, N_CTX, name):
    print(f"B={B}  N_CTX={N_CTX}  cand={name}", flush=True)
    q = torch.randn((B, N_SEQ, N_CTX, N_HEADS, HEAD_DIM),
                    dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q); v = torch.randn_like(q)
    rm = torch.randint(0, 2, (B, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device="cuda")
    pb_bf = torch.randn((B, 1, N_HEADS, N_CTX, N_CTX), dtype=torch.bfloat16, device="cuda")
    pb_fp = pb_bf.float()

    mod = load_candidate(name)
    cand_fn = make_fwd_apply(mod)

    # one warmup each
    print(" warm cand", flush=True);  cand_fn(q, k, v, rm, pb_bf); torch.cuda.synchronize()
    print(" warm prod", flush=True);  TKEvoAttention.apply(q, k, v, rm, pb_bf); torch.cuda.synchronize()
    print(" warm tri", flush=True);  TritonEvoformer(q, k, v, rm, pb_fp); torch.cuda.synchronize()
    print(" bench", flush=True)

    flops = 4.0 * B * N_SEQ * N_HEADS * N_CTX * N_CTX * HEAD_DIM

    cand_ms = triton.testing.do_bench(lambda: cand_fn(q, k, v, rm, pb_bf),       rep=500, warmup=100)
    prod_ms = triton.testing.do_bench(lambda: TKEvoAttention.apply(q, k, v, rm, pb_bf), rep=500, warmup=100)
    tri_ms  = triton.testing.do_bench(lambda: TritonEvoformer(q, k, v, rm, pb_fp),     rep=500, warmup=100)

    tris = flops * 1e-12 / (tri_ms * 1e-3)
    pros = flops * 1e-12 / (prod_ms * 1e-3)
    cas = flops * 1e-12 / (cand_ms * 1e-3)
    print(f"   triton {tris:8.2f}   prod-tk {pros:8.2f}   {name} {cas:8.2f}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: bench_one.py B N_CTX cand")
        sys.exit(2)
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
