"""Correctness test for an experimental candidate kernel.

Usage:
    python3 test_candidate.py v1_baseline
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MEGAFOLD = os.path.expanduser("~/MegaFold/megafold/model/FusedEvoAttention")
sys.path.insert(0, MEGAFOLD)

from tk_fwd_layer import load_candidate, make_fwd_apply
from evoattention import TritonEvoformer


CASES = [
    # D=64 native
    dict(B=1, N_SEQ=1,   H=4,  N_CTX=128, DIM=64),
    dict(B=1, N_SEQ=1,   H=4,  N_CTX=256, DIM=64),
    dict(B=1, N_SEQ=1,   H=4,  N_CTX=384, DIM=64),
    dict(B=2, N_SEQ=1,   H=16, N_CTX=384, DIM=64),
    dict(B=4, N_SEQ=1,   H=16, N_CTX=256, DIM=64),
    dict(B=1, N_SEQ=1,   H=16, N_CTX=640, DIM=64),
    dict(B=1, N_SEQ=1,   H=16, N_CTX=768, DIM=64),
    dict(B=1, N_SEQ=1,   H=16, N_CTX=1024, DIM=64),
    dict(B=1, N_SEQ=32,  H=4,  N_CTX=384, DIM=64),
    # D=128 native
    dict(B=1, N_SEQ=1,   H=4,  N_CTX=256, DIM=128),
    dict(B=1, N_SEQ=4,   H=4,  N_CTX=384, DIM=128),
    # D=96 -> pad to 128 (Evo native head_dim)
    dict(B=1, N_SEQ=1,   H=4,  N_CTX=384, DIM=96),
    # D=32 -> pad to 64
    dict(B=1, N_SEQ=1,   H=4,  N_CTX=384, DIM=32),
]


def run_case(fwd_apply, *, B, N_SEQ, H, N_CTX, DIM, seed=0, verbose=True):
    torch.manual_seed(seed)
    device = "cuda"

    Q = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)
    K = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)
    V = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)

    mask = torch.randint(0, 2, (B, N_SEQ, 1, 1, N_CTX), device=device)
    res_mask_fp = (1e9 * (mask - 1)).to(torch.float32)

    pair_bias_fp = torch.randn((B, 1, H, N_CTX, N_CTX), dtype=torch.float32, device=device)

    Q_ref = Q.clone()
    K_ref = K.clone()
    V_ref = V.clone()
    pair_bias_ref = pair_bias_fp.clone()
    with torch.no_grad():
        ref_O = TritonEvoformer(Q_ref, K_ref, V_ref, res_mask_fp, pair_bias_ref)

    tk_O = fwd_apply(Q, K, V, res_mask_fp, pair_bias_fp)

    ref_f32 = ref_O.float()
    tk_f32 = tk_O.float()
    diff = (ref_f32 - tk_f32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()

    tag = f"B={B} N_SEQ={N_SEQ} H={H} N_CTX={N_CTX} D={DIM}"
    atol, rtol = 1e-2, 2e-2
    ok = torch.allclose(ref_f32, tk_f32, atol=atol, rtol=rtol)
    if verbose:
        flag = "OK " if ok else "BAD"
        print(f"[{flag}] {tag}  max_abs={max_abs:.4f}  mean_abs={mean_abs:.5f}")
    if not ok:
        mismatch = (diff > (atol + rtol * ref_f32.abs())).sum().item()
        print(f"     MISMATCH {mismatch}/{ref_f32.numel()}")
    return ok, max_abs


def main(name):
    print(f"Loading candidate: {name}")
    module = load_candidate(name)
    fwd = make_fwd_apply(module)

    fails = 0
    for c in CASES:
        ok, _ = run_case(fwd, **c)
        if not ok:
            fails += 1
    if fails:
        print(f"\n{fails} CASES FAILED")
        sys.exit(1)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: test_candidate.py <candidate_name>")
        sys.exit(2)
    main(sys.argv[1])
