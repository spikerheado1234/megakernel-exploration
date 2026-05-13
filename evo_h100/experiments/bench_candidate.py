"""Benchmark a single candidate kernel against the triton reference and the
production tk kernel, using the same shape sweep as benchmark_fwd.py.

Usage:
    python3 bench_candidate.py v1_baseline
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


BATCH_VALS  = [1, 4, 8]
N_HEADS     = 16
HEAD_DIM    = 64
N_SEQ       = 1
N_CTX_VALS  = [128, 256, 384, 512, 640, 768, 1024]
DTYPE       = torch.bfloat16
DEVICE      = "cuda"
REP_MS      = 3000
WARMUP_MS   = 200

FWD_MULT = 4.0


def _flops(B, N_SEQ, H, N_CTX, D):
    return FWD_MULT * B * N_SEQ * H * N_CTX * N_CTX * D


def make_inputs(B, N_CTX, pair_bias_dtype):
    q = torch.randn((B, N_SEQ, N_CTX, N_HEADS, HEAD_DIM),
                    dtype=DTYPE, device=DEVICE, requires_grad=False)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    res_mask_bool = torch.randint(0, 2, (B, N_SEQ, 1, 1, N_CTX),
                                  dtype=torch.bool, device=DEVICE)
    pair_bias = torch.randn((B, 1, N_HEADS, N_CTX, N_CTX),
                            dtype=pair_bias_dtype, device=DEVICE,
                            requires_grad=False)
    return q, k, v, res_mask_bool, pair_bias


def _bench_fwd(run_forward, pair_bias_dtype, B, N_CTX):
    q, k, v, res_mask, pair_bias = make_inputs(B, N_CTX, pair_bias_dtype)
    fn = lambda: run_forward(q, k, v, res_mask, pair_bias)
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, rep=REP_MS, warmup=WARMUP_MS)


def _print_table(title, rows):
    print(f"=== {title} ===")
    header = (f"{'N_CTX':>7}  {'triton (TFLOP/s)':>18}  "
              f"{'prod-tk (TFLOP/s)':>18}  {'cand (TFLOP/s)':>16}  "
              f"{'cand/tri':>9}  {'cand/prod':>10}")
    print(header)
    print("-" * len(header))
    for (n, tri, prod, cand) in rows:
        ratio_tri = cand / tri if tri > 0 else float("nan")
        ratio_pr  = cand / prod if prod > 0 else float("nan")
        print(f"{n:>7}  {tri:>18.2f}  {prod:>18.2f}  {cand:>16.2f}  "
              f"{ratio_tri:>8.2f}x  {ratio_pr:>9.2f}x")
    print()


def main(name):
    print(f"Candidate: {name}")
    module = load_candidate(name)
    cand_fwd = make_fwd_apply(module)

    print(f"H={N_HEADS}  D={HEAD_DIM}  N_SEQ={N_SEQ}  dtype={DTYPE}")
    for B in BATCH_VALS:
        rows = []
        for n in N_CTX_VALS:
            flops = _flops(B, N_SEQ, N_HEADS, n, HEAD_DIM)
            tri_ms  = _bench_fwd(TritonEvoformer,      pair_bias_dtype=torch.float32,  B=B, N_CTX=n)
            prod_ms = _bench_fwd(TKEvoAttention.apply, pair_bias_dtype=torch.bfloat16, B=B, N_CTX=n)
            cand_ms = _bench_fwd(cand_fwd,             pair_bias_dtype=torch.bfloat16, B=B, N_CTX=n)
            rows.append((n,
                         flops * 1e-12 / (tri_ms * 1e-3),
                         flops * 1e-12 / (prod_ms * 1e-3),
                         flops * 1e-12 / (cand_ms * 1e-3)))
        _print_table(f"Forward only — BATCH={B}", rows)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: bench_candidate.py <candidate_name>")
        sys.exit(2)
    main(sys.argv[1])
