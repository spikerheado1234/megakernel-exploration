"""Correctness tests for TKEvoAttention backward pass.

Compares TK backward gradients (dQ, dK, dV, d_pair_bias) against the Triton
reference at ~/MegaFold/megafold/model/FusedEvoAttention/evoattention.py.

Tolerance matches the MegaFold test (atol=1e-2, rtol=2e-2).
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MEGAFOLD = os.path.expanduser("~/MegaFold/megafold/model/FusedEvoAttention")
sys.path.insert(0, MEGAFOLD)

from tk_evo_attention_layer import TKEvoAttention
from evoattention import TritonEvoformer


def _stats(name, ref, tk, atol, rtol):
    diff = (ref - tk).abs()
    denom = ref.abs().clamp_min(1e-6)
    return dict(
        name=name,
        max_abs=diff.max().item(),
        mean_abs=diff.mean().item(),
        max_rel=(diff / denom).max().item(),
        ok=torch.allclose(ref, tk, atol=atol, rtol=rtol),
        mismatch=int((diff > (atol + rtol * ref.abs())).sum().item()),
        total=ref.numel(),
    )


def run_case(B, N_SEQ, H, N_CTX, DIM, *, seed=0, verbose=True):
    torch.manual_seed(seed)
    device = "cuda"

    Q = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)
    K = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)
    V = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)

    mask = torch.randint(0, 2, (B, N_SEQ, 1, 1, N_CTX), device=device)
    res_mask_fp = (1e9 * (mask - 1)).to(torch.float32)

    pair_bias_fp = torch.randn((B, 1, H, N_CTX, N_CTX), dtype=torch.float32, device=device)

    # Use a deterministic dO; easier to reason about numerics across backends.
    torch.manual_seed(seed + 1)
    dO = torch.randn((B, N_SEQ, N_CTX, H, DIM), dtype=torch.bfloat16, device=device)

    # ----- Triton reference -----
    Qr = Q.clone().detach().requires_grad_(True)
    Kr = K.clone().detach().requires_grad_(True)
    Vr = V.clone().detach().requires_grad_(True)
    PBr = pair_bias_fp.clone().detach().requires_grad_(True)

    Or = TritonEvoformer(Qr, Kr, Vr, res_mask_fp, PBr)
    Or.backward(dO, retain_graph=False)
    dQ_ref, dK_ref, dV_ref = Qr.grad, Kr.grad, Vr.grad
    dPB_ref = PBr.grad  # fp32, shape (B, 1, H, N, N)

    # ----- TK -----
    Qt = Q.clone().detach().requires_grad_(True)
    Kt = K.clone().detach().requires_grad_(True)
    Vt = V.clone().detach().requires_grad_(True)
    PBt = pair_bias_fp.clone().detach().requires_grad_(True)

    Ot = TKEvoAttention.apply(Qt, Kt, Vt, res_mask_fp, PBt)
    Ot.backward(dO, retain_graph=False)
    dQ_tk, dK_tk, dV_tk = Qt.grad, Kt.grad, Vt.grad
    dPB_tk = PBt.grad  # bf16, shape (B, 1, H, N, N)

    atol, rtol = 1e-2, 2e-2
    all_ok = True
    tag = f"B={B} N_SEQ={N_SEQ} H={H} N_CTX={N_CTX} D={DIM}"
    results = [
        _stats("dQ",        dQ_ref.float(),  dQ_tk.float(),  atol, rtol),
        _stats("dK",        dK_ref.float(),  dK_tk.float(),  atol, rtol),
        _stats("dV",        dV_ref.float(),  dV_tk.float(),  atol, rtol),
        _stats("d_pair_bias", dPB_ref.float(), dPB_tk.float(), atol, rtol),
    ]
    if verbose:
        print(f"[{tag}]")
        for r in results:
            marker = "OK  " if r["ok"] else "FAIL"
            print(f"  {marker} {r['name']:>11}  max_abs={r['max_abs']:.4f}"
                  f"  mean_abs={r['mean_abs']:.5f}  max_rel={r['max_rel']:.3f}"
                  f"  mismatch={r['mismatch']}/{r['total']}")
    for r in results:
        if not r["ok"]:
            all_ok = False
    assert all_ok, f"[{tag}] backward mismatch"
    return results


def main():
    # Start with modest cases (D=64 and D=128 native).
    cases = [
        dict(B=1, N_SEQ=1, H=4,  N_CTX=128, DIM=64),
        dict(B=1, N_SEQ=1, H=4,  N_CTX=256, DIM=64),
        dict(B=1, N_SEQ=1, H=4,  N_CTX=384, DIM=64),
        dict(B=2, N_SEQ=1, H=16, N_CTX=256, DIM=64),
        dict(B=1, N_SEQ=4, H=4,  N_CTX=256, DIM=64),   # exercises N_SEQ > 1 (pair_bias broadcast)
        dict(B=1, N_SEQ=1, H=4,  N_CTX=256, DIM=128),
        dict(B=1, N_SEQ=1, H=4,  N_CTX=384, DIM=128),
        # padded
        dict(B=1, N_SEQ=1, H=4,  N_CTX=384, DIM=96),
    ]
    for c in cases:
        run_case(**c)
    print("\nALL BWD TESTS PASSED")


if __name__ == "__main__":
    main()
