"""Unit tests for TKEvoAttention forward pass.

Compares the TK layer against the Triton reference at
~/MegaFold/megafold/model/FusedEvoAttention/evoattention.py

Input shapes (Triton / MegaFold convention, passed directly to the layer):
    Q, K, V:   (B, N_SEQ, N_CTX, H, D)
    res_mask:  (B, N_SEQ, 1, 1, N_CTX)
    pair_bias: (B, 1, H, N_CTX, N_CTX)
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Triton reference
MEGAFOLD = os.path.expanduser("~/MegaFold/megafold/model/FusedEvoAttention")
sys.path.insert(0, MEGAFOLD)

from tk_evo_attention_layer import TKEvoAttention
from evoattention import TritonEvoformer


def run_case(B, N_SEQ, H, N_CTX, DIM, *, seed=0, verbose=True):
    torch.manual_seed(seed)
    device = "cuda"

    # Raw (B, N_SEQ, N_CTX, H, DIM) inputs, bf16 for Q/K/V
    Q = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)
    K = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)
    V = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)

    # res_mask in {-1e9, 0}, shape (B, N_SEQ, 1, 1, N_CTX)
    mask = torch.randint(0, 2, (B, N_SEQ, 1, 1, N_CTX), device=device)
    res_mask_fp = (1e9 * (mask - 1)).to(torch.float32)

    # pair_bias shape (B, 1, H, N_CTX, N_CTX)
    pair_bias_fp = torch.randn((B, 1, H, N_CTX, N_CTX), dtype=torch.float32, device=device)

    # ------- Triton reference -------
    Q_ref = Q.clone().requires_grad_(False)
    K_ref = K.clone().requires_grad_(False)
    V_ref = V.clone().requires_grad_(False)
    pair_bias_ref = pair_bias_fp.clone().requires_grad_(False)
    with torch.no_grad():
        ref_O = TritonEvoformer(Q_ref, K_ref, V_ref, res_mask_fp, pair_bias_ref)
    # ref_O shape: (B, N_SEQ, N_CTX, H, DIM)

    # ------- TK layer -------
    tk_O = TKEvoAttention.apply(Q, K, V, res_mask_fp, pair_bias_fp)

    # ------- Compare -------
    ref_f32 = ref_O.float()
    tk_f32 = tk_O.float()
    diff = (ref_f32 - tk_f32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = ref_f32.abs().clamp_min(1e-6)
    max_rel = (diff / denom).max().item()

    tag = f"B={B} N_SEQ={N_SEQ} H={H} N_CTX={N_CTX} D={DIM}"
    if verbose:
        print(f"[{tag}] max_abs={max_abs:.4f}  mean_abs={mean_abs:.5f}  max_rel={max_rel:.3f}")

    atol, rtol = 1e-2, 2e-2
    ok = torch.allclose(ref_f32, tk_f32, atol=atol, rtol=rtol)
    if not ok:
        mismatch = (diff > (atol + rtol * ref_f32.abs())).sum().item()
        total = ref_f32.numel()
        print(f"  MISMATCH: {mismatch}/{total} elements outside tol (atol={atol}, rtol={rtol})")
    assert ok, f"[{tag}] output mismatch  max_abs={max_abs} max_rel={max_rel}"
    return max_abs, mean_abs


def main():
    cases = [
        # D=64 native
        dict(B=1, N_SEQ=1,   H=4,  N_CTX=128, DIM=64),
        dict(B=1, N_SEQ=1,   H=4,  N_CTX=256, DIM=64),
        dict(B=1, N_SEQ=1,   H=4,  N_CTX=384, DIM=64),
        dict(B=2, N_SEQ=1,   H=16, N_CTX=384, DIM=64),
        dict(B=4, N_SEQ=1,   H=16, N_CTX=256, DIM=64),
        dict(B=1, N_SEQ=32,  H=4,  N_CTX=384, DIM=64),
        dict(B=1, N_SEQ=64,  H=4,  N_CTX=256, DIM=64),
        # D=128 native
        dict(B=1, N_SEQ=1,   H=4,  N_CTX=256, DIM=128),
        dict(B=1, N_SEQ=4,   H=4,  N_CTX=384, DIM=128),
        # D=96 -> pad to 128 (EvoAttention native)
        dict(B=1, N_SEQ=1,   H=4,  N_CTX=384, DIM=96),
        dict(B=2, N_SEQ=1,   H=16, N_CTX=256, DIM=96),
        dict(B=1, N_SEQ=16,  H=4,  N_CTX=384, DIM=96),
        # D=32 -> pad to 64 (triangle-attention native)
        dict(B=1, N_SEQ=1,   H=4,  N_CTX=384, DIM=32),
        dict(B=1, N_SEQ=32,  H=4,  N_CTX=256, DIM=32),
        dict(B=2, N_SEQ=4,   H=8,  N_CTX=384, DIM=32),
        # D=16 -> pad to 64
        dict(B=1, N_SEQ=1,   H=4,  N_CTX=384, DIM=16),
        dict(B=1, N_SEQ=16,  H=4,  N_CTX=256, DIM=16),
    ]
    for c in cases:
        run_case(**c)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
