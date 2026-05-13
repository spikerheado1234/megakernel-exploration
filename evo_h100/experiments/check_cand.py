"""Run a single candidate kernel once with timeout (no triton)."""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tk_fwd_layer import load_candidate, make_fwd_apply


def main(B, N_CTX, name):
    H, D, N_SEQ = 16, 64, 1
    print(f"B={B}  N_CTX={N_CTX}  cand={name}", flush=True)
    q = torch.randn((B, N_SEQ, N_CTX, H, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q); v = torch.randn_like(q)
    rm = torch.randint(0, 2, (B, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device="cuda")
    pb = torch.randn((B, 1, H, N_CTX, N_CTX), dtype=torch.bfloat16, device="cuda")

    fn = make_fwd_apply(load_candidate(name))
    print(" calling...", flush=True)
    out = fn(q, k, v, rm, pb)
    torch.cuda.synchronize()
    print(f" OK, out shape {out.shape}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
