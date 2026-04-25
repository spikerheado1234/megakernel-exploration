"""Throughput comparison: Triton EvoAttention vs. TK EvoAttention, in three modes.

Mirrors the shape schedule and modes of
    ~/MegaFold/benchmarks/evoattention_speed.py
but compares against the ThunderKittens kernel instead of DeepSpeed / torch.

Prints three tables (forward, backward, combined end-to-end):
    N_CTX   triton (TFLOP/s)   tk (TFLOP/s)   ratio

FLOP counting (per (B, N_SEQ, H, N, D)):
    fwd  : 4  * B*N_SEQ*H*N^2*D     (two matmuls Q@K^T and P@V)
    bwd  : 10 * B*N_SEQ*H*N^2*D     (five matmuls: S^T, dP^T, dV, dK, dQ)
    full : 14 * B*N_SEQ*H*N^2*D     (fwd + bwd)
(pair-bias add, mask add, softmax, and any recomputation are omitted.)
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
REP_MS      = 3000
WARMUP_MS   = 200


# FLOP multipliers (per B*N_SEQ*H*N^2*D).
FWD_MULT  = 4.0
BWD_MULT  = 10.0
FULL_MULT = FWD_MULT + BWD_MULT


def _flops(mult, B, N_SEQ, H, N_CTX, D):
    return mult * B * N_SEQ * H * N_CTX * N_CTX * D


def make_inputs(N_CTX, pair_bias_dtype, requires_grad):
    q = torch.randn((BATCH, N_SEQ, N_CTX, N_HEADS, HEAD_DIM),
                    dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    k = torch.randn_like(q, requires_grad=requires_grad)
    v = torch.randn_like(q, requires_grad=requires_grad)
    res_mask_bool = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, N_CTX),
                                  dtype=torch.bool, device=DEVICE)
    pair_bias = torch.randn((BATCH, 1, N_HEADS, N_CTX, N_CTX),
                            dtype=pair_bias_dtype, device=DEVICE,
                            requires_grad=requires_grad)
    return q, k, v, res_mask_bool, pair_bias


# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------

def _bench_fwd(run_forward, pair_bias_dtype, N_CTX):
    q, k, v, res_mask, pair_bias = make_inputs(N_CTX, pair_bias_dtype, requires_grad=False)
    fn = lambda: run_forward(q, k, v, res_mask, pair_bias)
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, rep=REP_MS, warmup=WARMUP_MS)


def _bench_bwd(run_forward, pair_bias_dtype, N_CTX):
    q, k, v, res_mask, pair_bias = make_inputs(N_CTX, pair_bias_dtype, requires_grad=True)
    o = run_forward(q, k, v, res_mask, pair_bias)
    do = torch.randn_like(o)
    # Clear `.grad` each iteration so the backward is measured in
    # "assign new grad" mode (same as _bench_full). Without this, every
    # iteration after the first pays the cost of `param.grad += new_grad`,
    # which is a meaningful memory-bandwidth tax on large tensors like
    # pair_bias.grad (~268 MB at N_CTX=1024). That asymmetry used to
    # overstate bwd cost relative to full and produce the misleading
    # "fwd fast + bwd fast but full slow" pattern.
    grad_leaves = (q, k, v, pair_bias)
    def fn():
        for t in grad_leaves:
            t.grad = None
        o.backward(do, retain_graph=True)
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, rep=REP_MS, warmup=WARMUP_MS)


def _bench_full(run_forward, pair_bias_dtype, N_CTX):
    q, k, v, res_mask, pair_bias = make_inputs(N_CTX, pair_bias_dtype, requires_grad=True)

    # We need `do` to have a stable shape; generate once.
    with torch.no_grad():
        o_probe = run_forward(q, k, v, res_mask, pair_bias)
    do = torch.randn_like(o_probe)
    del o_probe

    def step():
        # Zero grads so each iteration is an independent fwd+bwd.
        for t in (q, k, v, pair_bias):
            if t.grad is not None:
                t.grad = None
        o = run_forward(q, k, v, res_mask, pair_bias)
        o.backward(do, retain_graph=False)

    step()
    torch.cuda.synchronize()
    return triton.testing.do_bench(step, rep=REP_MS, warmup=WARMUP_MS)


BENCHES = {
    "fwd":  (_bench_fwd,  FWD_MULT,  "Forward only"),
    "bwd":  (_bench_bwd,  BWD_MULT,  "Backward only"),
    "full": (_bench_full, FULL_MULT, "Full (fwd + bwd)"),
}


def _print_table(title, rows):
    print(f"=== {title} ===")
    header = f"{'N_CTX':>7}  {'triton (TFLOP/s)':>18}  {'tk (TFLOP/s)':>14}  {'ratio':>8}"
    print(header)
    print("-" * len(header))
    for (n, tri_tf, tk_tf) in rows:
        ratio = tk_tf / tri_tf if tri_tf > 0 else float("nan")
        print(f"{n:>7}  {tri_tf:>18.2f}  {tk_tf:>14.2f}  {ratio:>7.2f}x")
    print()


def main():
    print(f"BATCH={BATCH}  H={N_HEADS}  D={HEAD_DIM}  N_SEQ={N_SEQ}  "
          f"dtype={DTYPE}  device={DEVICE}")
    print()

    for mode, (bench_fn, mult, title) in BENCHES.items():
        rows = []
        for n in N_CTX_VALS:
            flops = _flops(mult, BATCH, N_SEQ, N_HEADS, n, HEAD_DIM)
            tri_ms = bench_fn(TritonEvoformer,      pair_bias_dtype=torch.float32,  N_CTX=n)
            tk_ms  = bench_fn(TKEvoAttention.apply, pair_bias_dtype=torch.bfloat16, N_CTX=n)
            rows.append((n, flops * 1e-12 / (tri_ms * 1e-3), flops * 1e-12 / (tk_ms * 1e-3)))
        _print_table(title, rows)


if __name__ == "__main__":
    main()
