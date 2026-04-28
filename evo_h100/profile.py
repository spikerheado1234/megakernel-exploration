"""Single-invocation NCU profiling harness for TK EvoAttention.

Shape config matches benchmark.py (BATCH=4, N_HEADS=16, HEAD_DIM=64, N_SEQ=1,
bf16) so results compare directly. The script warms up (compile + autotune
+ backward graph build) outside the profiled region, then runs exactly one
fwd/bwd/full step between cudaProfilerStart/Stop.

Intended invocation with --profile-from-start off so only the target region
is sampled:

    ncu --set full --target-processes all --profile-from-start off \\
        -o tk_bwd_n512 \\
        python profile.py --n-ctx 512 --mode bwd

For Triton, pass --impl triton. Kernel filtering can be done inside ncu
(e.g. -k evo_bwd_ker) since the wrapper launches a few ancillary kernels
(padding/transpose, bwd prep) alongside the main one.
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MEGAFOLD = os.path.expanduser("~/MegaFold/megafold/model/FusedEvoAttention")
sys.path.insert(0, MEGAFOLD)

from tk_evo_attention_layer import TKEvoAttention
from evoattention import TritonEvoformer


# --- Config (matches benchmark.py) ------------------------------------------
BATCH    = 1
N_HEADS  = 16
HEAD_DIM = 64
N_SEQ    = 1
DTYPE    = torch.bfloat16
DEVICE   = "cuda"


def make_inputs(n_ctx, pair_bias_dtype, requires_grad):
    q = torch.randn((BATCH, N_SEQ, n_ctx, N_HEADS, HEAD_DIM),
                    dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)
    k = torch.randn_like(q, requires_grad=requires_grad)
    v = torch.randn_like(q, requires_grad=requires_grad)
    res_mask = torch.randint(0, 2, (BATCH, N_SEQ, 1, 1, n_ctx),
                             dtype=torch.bool, device=DEVICE)
    pair_bias = torch.randn((BATCH, 1, N_HEADS, n_ctx, n_ctx),
                            dtype=pair_bias_dtype, device=DEVICE,
                            requires_grad=requires_grad)
    return q, k, v, res_mask, pair_bias


def zero_grads(tensors):
    for t in tensors:
        t.grad = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, required=True,
                    help="Sequence length. For D=64 CW=2, must be divisible by 128.")
    ap.add_argument("--mode", choices=["fwd", "bwd", "full"], default="fwd",
                    help="Which step to profile.")
    ap.add_argument("--impl", choices=["tk", "triton"], default="tk")
    args = ap.parse_args()

    if args.impl == "tk":
        runner = TKEvoAttention.apply
        pb_dtype = torch.bfloat16
    else:
        runner = TritonEvoformer
        pb_dtype = torch.float32

    needs_grad = args.mode != "fwd"
    q, k, v, rm, pb = make_inputs(args.n_ctx, pb_dtype, needs_grad)

    # ---- Warmup (excluded from profile) ------------------------------------
    # Runs the same code path once so JIT/autotune/allocator land before the
    # profiled region. For bwd/full we do a complete fwd+bwd to build the
    # autograd graph and cache any backward kernels.
    if args.mode == "fwd":
        with torch.no_grad():
            _ = runner(q, k, v, rm, pb)
    else:
        o_w  = runner(q, k, v, rm, pb)
        do_w = torch.randn_like(o_w)
        o_w.backward(do_w)
        zero_grads((q, k, v, pb))
        del o_w, do_w
    torch.cuda.synchronize()

    # ---- Pre-stage tensors whose creation we don't want to profile ---------
    if args.mode == "bwd":
        # Forward runs outside the profiled region so only the backward path
        # lands in the trace.
        o  = runner(q, k, v, rm, pb)
        do = torch.randn_like(o)
        zero_grads((q, k, v, pb))
    elif args.mode == "full":
        # Pre-generate the output-shaped grad so torch.randn_like doesn't
        # show up inside the profiled region.
        do = torch.randn((BATCH, N_SEQ, args.n_ctx, N_HEADS, HEAD_DIM),
                         dtype=DTYPE, device=DEVICE)
        zero_grads((q, k, v, pb))
    torch.cuda.synchronize()

    # ---- Profiled region ---------------------------------------------------
    torch.cuda.cudart().cudaProfilerStart()
    if args.mode == "fwd":
        with torch.no_grad():
            _ = runner(q, k, v, rm, pb)
    elif args.mode == "bwd":
        o.backward(do)
    else:  # full
        o_f = runner(q, k, v, rm, pb)
        o_f.backward(do)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
