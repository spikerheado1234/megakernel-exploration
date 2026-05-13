"""Single-file benchmark: best.cu vs production tk_evoattention.cu vs the
Triton reference, forward-only.

Usage:
    # Default sweep (B = {1,4,8}, N_CTX = {128…1024}, H=16, D=64, bf16):
    python3 bench_fwd.py

    # Custom shapes:
    python3 bench_fwd.py --batches 1,4 --n-ctx 256,512,1024

    # Custom heads / head_dim:
    python3 bench_fwd.py --heads 16 --head-dim 64

Each (B, N_CTX) shape is benchmarked in its own python subprocess. Running
multiple shapes in one process triggers an intermittent hang inside
triton.testing.do_bench — we never tracked down whether it's a triton
autotune-cache thing or PyTorch caching allocator interaction. The
subprocess-per-shape pattern dodges it reliably.

Prints a six-column table:
    B   N_CTX   triton (TFLOP/s)   prod-tk (TFLOP/s)   best (TFLOP/s)   best/triton   best/prod
"""

import argparse
import os
import subprocess
import sys


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Worker mode: run one (B, N_CTX) and print "tri <x>  prod <y>  best <z>".
# Invoked as a subprocess by the parent.
# ---------------------------------------------------------------------------
def _run_worker(B, N_CTX, H, D, N_SEQ):
    import torch
    import triton.testing

    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))   # parent dir for prod TKEvoAttention
    sys.path.insert(0, os.path.expanduser("~/MegaFold/megafold/model/FusedEvoAttention"))

    from tk_evo_attention_layer import TKEvoAttention            # production
    from evoattention import TritonEvoformer                      # Triton ref
    from tk_fwd_layer import load_candidate, make_fwd_apply       # best.cu
    best_fn = make_fwd_apply(load_candidate("best"))

    q = torch.randn((B, N_SEQ, N_CTX, H, D), dtype=torch.bfloat16, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    rm = torch.randint(0, 2, (B, N_SEQ, 1, 1, N_CTX), dtype=torch.bool, device="cuda")
    pb_bf = torch.randn((B, 1, H, N_CTX, N_CTX), dtype=torch.bfloat16, device="cuda")
    pb_fp = pb_bf.float()

    # warmup once each
    best_fn(q, k, v, rm, pb_bf);            torch.cuda.synchronize()
    TKEvoAttention.apply(q, k, v, rm, pb_bf); torch.cuda.synchronize()
    TritonEvoformer(q, k, v, rm, pb_fp);    torch.cuda.synchronize()

    flops = 4.0 * B * N_SEQ * H * N_CTX * N_CTX * D

    best_ms = triton.testing.do_bench(lambda: best_fn(q, k, v, rm, pb_bf),         rep=500, warmup=100)
    prod_ms = triton.testing.do_bench(lambda: TKEvoAttention.apply(q, k, v, rm, pb_bf), rep=500, warmup=100)
    tri_ms  = triton.testing.do_bench(lambda: TritonEvoformer(q, k, v, rm, pb_fp),     rep=500, warmup=100)

    tri  = flops * 1e-12 / (tri_ms  * 1e-3)
    prod = flops * 1e-12 / (prod_ms * 1e-3)
    best = flops * 1e-12 / (best_ms * 1e-3)
    # Single, easy-to-parse line:
    print(f"RESULT triton={tri:.4f} prod={prod:.4f} best={best:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Driver mode: spawn one subprocess per shape, collect, print table.
# ---------------------------------------------------------------------------
def _parse_int_list(s):
    return [int(x) for x in s.split(",") if x]


def _main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches",  type=_parse_int_list, default=[1, 4, 8],
                        help="comma-separated batch sizes (default: 1,4,8)")
    parser.add_argument("--n-ctx",    type=_parse_int_list, default=[128, 256, 384, 512, 640, 768, 1024],
                        help="comma-separated context lengths (default: 128,256,384,512,640,768,1024)")
    parser.add_argument("--heads",    type=int, default=16, help="number of heads")
    parser.add_argument("--head-dim", type=int, default=64, help="head dim")
    parser.add_argument("--n-seq",    type=int, default=1,  help="MSA dim")
    parser.add_argument("--timeout",  type=int, default=120, help="per-shape timeout (s)")
    parser.add_argument("--worker",   nargs=5, metavar=("B", "N_CTX", "H", "D", "N_SEQ"),
                        help="(internal) run a single shape in this process")
    args = parser.parse_args()

    if args.worker is not None:
        B, N_CTX, H, D, N_SEQ = map(int, args.worker)
        _run_worker(B, N_CTX, H, D, N_SEQ)
        return

    print(f"H={args.heads}  D={args.head_dim}  N_SEQ={args.n_seq}  dtype=bf16  device=cuda")
    print(f"BATCHES={args.batches}  N_CTX={args.n_ctx}")
    print()

    header = (f"{'B':>4} {'N_CTX':>6}  {'triton (TFLOP/s)':>18}  "
              f"{'prod-tk (TFLOP/s)':>18}  {'best (TFLOP/s)':>16}  "
              f"{'best/triton':>11}  {'best/prod':>10}")
    print(header)
    print("-" * len(header))

    for B in args.batches:
        for N in args.n_ctx:
            cmd = [sys.executable, "-u", os.path.abspath(__file__), "--worker",
                   str(B), str(N), str(args.heads), str(args.head_dim), str(args.n_seq)]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True,
                                     timeout=args.timeout, check=False)
            except subprocess.TimeoutExpired:
                print(f"{B:>4} {N:>6}  {'TIMEOUT':>18}")
                continue

            line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT")), None)
            if line is None:
                print(f"{B:>4} {N:>6}  FAILED  (stderr: {out.stderr.strip()[:120]})")
                continue

            kv = dict(t.split("=") for t in line.split()[1:])
            tri, prod, best = float(kv["triton"]), float(kv["prod"]), float(kv["best"])
            r_tri  = best / tri  if tri  > 0 else float("nan")
            r_prod = best / prod if prod > 0 else float("nan")
            print(f"{B:>4} {N:>6}  {tri:>18.2f}  {prod:>18.2f}  {best:>16.2f}  "
                  f"{r_tri:>10.2f}x  {r_prod:>9.2f}x")


if __name__ == "__main__":
    _main()
