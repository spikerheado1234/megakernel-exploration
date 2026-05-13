"""Quick benchmark — like bench_candidate.py but with smaller rep/warmup
and live (unbuffered) output. Use for fast iteration.

Usage:
    python3 bench_quick.py v2_kv64 [v3_xxx ...]   # bench one or more candidates
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
from tk_evo_attention_layer import TKEvoAttention   # production
from evoattention import TritonEvoformer


BATCH_VALS  = [int(x) for x in os.environ.get("BATCHES", "1,4,8").split(",")]
N_HEADS     = 16
HEAD_DIM    = 64
N_SEQ       = 1
N_CTX_VALS  = [128, 256, 384, 512, 640, 768, 1024]
DTYPE       = torch.bfloat16
DEVICE      = "cuda"
REP_MS      = 200
WARMUP_MS   = 50

FWD_MULT = 4.0


def _flops(B, N_SEQ, H, N_CTX, D):
    return FWD_MULT * B * N_SEQ * H * N_CTX * N_CTX * D


def make_inputs(B, N_CTX, pair_bias_dtype):
    q = torch.randn((B, N_SEQ, N_CTX, N_HEADS, HEAD_DIM),
                    dtype=DTYPE, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    res_mask_bool = torch.randint(0, 2, (B, N_SEQ, 1, 1, N_CTX),
                                  dtype=torch.bool, device=DEVICE)
    pair_bias = torch.randn((B, 1, N_HEADS, N_CTX, N_CTX),
                            dtype=pair_bias_dtype, device=DEVICE)
    return q, k, v, res_mask_bool, pair_bias


def _bench_fwd(run_forward, pair_bias_dtype, B, N_CTX):
    q, k, v, res_mask, pair_bias = make_inputs(B, N_CTX, pair_bias_dtype)
    fn = lambda: run_forward(q, k, v, res_mask, pair_bias)
    fn()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, rep=REP_MS, warmup=WARMUP_MS)
    del q, k, v, res_mask, pair_bias
    torch.cuda.empty_cache()
    return ms


def main(names):
    cands = []
    for name in names:
        module = load_candidate(name)
        cands.append((name, make_fwd_apply(module)))

    header_names = ["triton", "prod-tk"] + [n for n, _ in cands]
    for B in BATCH_VALS:
        print(f"\n=== BATCH={B} ===", flush=True)
        hdr = f"{'N_CTX':>7}  " + "  ".join(f"{n:>10}" for n in header_names) + "  best"
        print(hdr, flush=True)
        for n in N_CTX_VALS:
            flops = _flops(B, N_SEQ, N_HEADS, n, HEAD_DIM)
            cells = []
            tri_ms  = _bench_fwd(TritonEvoformer,      pair_bias_dtype=torch.float32,  B=B, N_CTX=n)
            prod_ms = _bench_fwd(TKEvoAttention.apply, pair_bias_dtype=torch.bfloat16, B=B, N_CTX=n)
            tris = flops * 1e-12 / (tri_ms * 1e-3)
            pros = flops * 1e-12 / (prod_ms * 1e-3)
            cells.append(tris); cells.append(pros)
            for _, fn in cands:
                cms = _bench_fwd(fn, pair_bias_dtype=torch.bfloat16, B=B, N_CTX=n)
                cells.append(flops * 1e-12 / (cms * 1e-3))
            best_idx = max(range(len(cells)), key=lambda i: cells[i])
            best = header_names[best_idx]
            row = f"{n:>7}  " + "  ".join(f"{c:>10.2f}" for c in cells) + f"  {best}"
            print(row, flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: bench_quick.py <cand1> [<cand2> ...]")
        sys.exit(2)
    main(sys.argv[1:])
