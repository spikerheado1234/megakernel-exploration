# EvoAttention forward-kernel experiments

Self-contained workspace for trying optimisations on the EvoAttention
forward kernel without touching the production `tk_evoattention.cu`.

## Layout

```
experiments/
├── Makefile               # parameterised build: `make CAND=v<n>_xxx`
├── common_wrapper.cuh     # shared pybind11 + launcher; each .cu #includes it
│
├── v1_baseline.cu         # forward-only copy of tk_evoattention.cu
├── v2_kv64.cu             # kv_height 128 → 64 (kills register spill)
├── v3_no_pp.cu            # remove FA3 ping-pong barriers
├── v4_stages3.cu          # stages 4 → 3 (smaller smem)
├── v5_combined_sem.cu     # pb+rm share one mbarrier
├── v6_one_sem.cu          # K+V+pb+rm all share one mbarrier  ← winner
├── v7_intra_pipe.cu       # intra-WG QK/PV overlap (slower — see log)
├── v8_intra_stages4.cu    # v7 with stages=4 (still slower)
├── v9_more_regs.cu        # bump consumer reg cap to 224 (no-op)
├── v10_overlap_load.cu    # hoist pb/rm load before mma_wait (no-op)
├── v11_cw1_2cta.cu        # CW=1, 2 CTAs/SM (helps tiny shapes only)
├── v12_mma_sync.cu        # FlashKDA-style: TMA + warp mma.sync + 2 CTAs/SM
├── v13_mma_stages4.cu     # v12 with stages=4 (wash)
├── v14_mma_tune.cu        # v12 + producer=24 / consumer=232 + hoisted pb,rm
├── best.cu                # final pick: ≡ v6_one_sem.cu
│
├── tk_fwd_layer.py        # python wrapper that takes an _C_<cand> module
├── tk_evo_best_layer.py   # drop-in autograd.Function around best.cu (fwd only)
│
├── test_candidate.py      # correctness vs Triton reference (atol=1e-2,
│                          #   rtol=2e-2 like the production test)
├── bench_one.py           # bench one (B, N_CTX, candidate) in one python proc
├── bench_sweep.sh         # full (B, N_CTX) sweep — one subprocess per shape
├── bench_final.sh         # full sweep for `best` only — populates best.txt
│
├── log.md                 # what was tried + per-candidate numbers + why
└── best.txt               # final triton / prod / best comparison table
```

## Build a candidate

```
make CAND=v6_one_sem
# produces _C_v6_one_sem.cpython-310-x86_64-linux-gnu.so
```

## Test correctness vs Triton

```
python3 test_candidate.py v6_one_sem
```

## Benchmark one shape

```
python3 bench_one.py 8 1024 v6_one_sem
# B   N_CTX  candidate
```

## Full sweep

```
bash bench_sweep.sh v6_one_sem v11_cw1_2cta     # 1 or more candidates
bash bench_final.sh                              # `best` vs triton + prod
```

The sweep launches every (B, N_CTX) pair in its own python subprocess —
running many shapes in one process triggers a hang inside triton.testing
that we never tracked down (suspect: triton autotune caching state).

## Use the best kernel from Python

```python
from experiments.tk_evo_best_layer import TKEvoAttentionBest

O = TKEvoAttentionBest.apply(Q, K, V, res_mask, pair_bias)
```

(No backward. Use the parent dir's `TKEvoAttention` for autograd.)

## Headline result

best.cu is ~15-20 % faster than the shipping production kernel and
1.0 – 1.78× faster than the Triton reference across the standard
(B ∈ {1,4,8}) × (N_CTX ∈ {128…1024}) sweep at H=16, D=64, bf16. See
`best.txt`.
