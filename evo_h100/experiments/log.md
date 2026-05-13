# EvoAttention forward-kernel optimization log

Hardware: NVIDIA H100 80GB HBM3. Driver 580.126.09, CUDA 13.0.
Bench: `bench_candidate.py` runs `triton.testing.do_bench` (rep=3000ms, warmup=200ms)
on every (B, N_CTX) in the standard sweep:

    BATCH = [1, 4, 8]
    N_HEADS = 16
    HEAD_DIM = 64
    N_CTX = [128, 256, 384, 512, 640, 768, 1024]

All numbers below are **TFLOP/s** (higher = better).

## Baseline (no changes)

`v1_baseline.cu` — forward-only copy of `tk_evoattention.cu`, compiled into its
own TU. Built with the same flags but **without the bwd kernels in the same
file**. This alone yields ~5-20% lift over the shipping `_C.so` binary, simply
because the compiler picks a different register/spill allocation when the file
is smaller. We use v1_baseline as our "starting line" — the production
benchmark column (`prod-tk`) is from the existing `_C.so`.

| B | N_CTX | triton | prod-tk | v1 (forward-only TU) | v1/prod |
|---|-------|--------|---------|----------------------|---------|
| 1 |  128  |  0.44  |  0.57   |  0.76                | 1.34x   |
| 1 |  256  |  1.84  |  2.36   |  2.86                | 1.21x   |
| 1 |  384  |  4.11  |  5.08   |  6.03                | 1.19x   |
| 1 |  512  |  7.32  |  8.75   | 10.35                | 1.18x   |
| 1 |  640  | 11.48  | 13.63   | 15.62                | 1.15x   |
| 1 |  768  | 17.04  | 18.91   | 21.63                | 1.14x   |
| 1 | 1024  | 29.02  | 31.19   | 35.11                | 1.13x   |
| 4 |  128  |  1.94  |  2.50   |  2.94                | 1.18x   |
| 4 |  256  |  7.66  |  9.40   | 10.61                | 1.13x   |
| 4 |  384  | 17.20  | 18.23   | 19.47                | 1.07x   |
| 4 |  512  | 28.92  | 28.85   | 30.84                | 1.07x   |
| 4 |  640  | 41.99  | 37.52   | 39.66                | 1.06x   |
| 4 |  768  | 46.03  | 49.45   | 51.73                | 1.05x   |
| 4 | 1024  | 51.56  | 67.01   | 69.42                | 1.04x   |
| 8 |  128  |  3.78  |  4.78   |  5.55                | 1.16x   |
| 8 |  256  | 14.93  | 16.33   | 17.60                | 1.08x   |
| 8 |  384  | 31.80  | 29.21   | 30.47                | 1.04x   |
| 8 |  512  | 41.54  | 42.59   | 44.29                | 1.04x   |
| 8 |  640  | 46.23  | 54.57   | 55.57                | 1.02x   |
| 8 |  768  | 49.50  | 64.98   | 66.67                | 1.03x   |
| 8 | 1024  | 54.14  | 81.87   | 83.71                | 1.02x   |

H100 BF16 tensor-core peak is ~989 TFLOP/s.  At B=8 N=1024 we sit at ~8.5 %.
Most of that gap is the fundamental cost of EvoAttention (pair-bias load +
add, res-mask load + add) on top of the attention math.

Production has spilling: `ptxas warning : Registers are spilled to local memory
... evo_fwd_ker (D=64): 36 bytes spill stores, 36 bytes spill loads`. That's
also present in v1_baseline so it's a kernel-structural issue, not a TU
artifact.

---

## v2_kv64 — kv_height 128 → 64, stages 3 → 4

Single change: tile_h drops from 128 to 64 in the inner-loop K/V/PB cycle.
Side effect: `att_block`, `pb_reg` are `rt_fl<16, 64>` (32 fp32 regs/thread)
instead of `rt_fl<16, 128>` (64 fp32 regs/thread). With ~32 regs/thread freed
the D=64 kernel **drops to 0 bytes spill stores** (was 36 bytes). Loop count
doubles (kv_iters = N/64 instead of N/128) but per-iter work is light enough
that the extra iterations are cheap.

Numbers (TFLOP/s, all measured via `bench_sweep.sh` which runs every shape
in its own python process to dodge a cross-shape hang we hit with multiple
shapes in one process):

| B | N_CTX | triton | prod-tk | v1 (fwd-only TU) | v2_kv64 | v2/v1  | v2/triton |
|---|-------|--------|---------|------------------|---------|--------|-----------|
| 1 |  128  | 0.48   | 0.62    | 0.74             | 0.73    | 0.99x  | 1.52x     |
| 1 |  256  | 1.97   | 2.43    | 2.79             | 2.85    | 1.02x  | 1.45x     |
| 1 |  384  | 4.09   | 4.87    | 5.73             | 5.99    | 1.05x  | 1.46x     |
| 1 |  512  | 7.15   | 8.47    | 9.95             | 10.12   | 1.02x  | 1.42x     |
| 1 |  640  | 11.77  | 13.45   | 15.44            | 15.45   | 1.00x  | 1.31x     |
| 1 |  768  | 17.27  | 17.79   | 20.98            | 21.48   | 1.02x  | 1.24x     |
| 1 | 1024  | 28.54  | 30.98   | 34.19            | 35.49   | 1.04x  | 1.24x     |
| 4 |  128  | 1.94   | 2.35    | 2.82             | 2.82    | 1.00x  | 1.45x     |
| 4 |  256  | 7.53   | 8.99    | 10.35            | 10.29   | 0.99x  | 1.37x     |
| 4 |  384  | 16.95  | 17.55   | 18.94            | 19.52   | 1.03x  | 1.15x     |
| 4 |  512  | 28.64  | 28.58   | 30.30            | 30.93   | 1.02x  | 1.08x     |
| 4 |  640  | 42.26  | 37.30   | 39.20            | 42.08   | 1.07x  | 1.00x     |
| 4 |  768  | 45.89  | 47.64   | 50.24            | 53.67   | 1.07x  | 1.17x     |
| 4 | 1024  | 51.80  | 65.78   | 68.59            | 74.06   | 1.08x  | 1.43x     |
| 8 |  128  | 3.72   | 4.78    | 5.34             | 5.60    | 1.05x  | 1.51x     |
| 8 |  256  | 15.00  | 15.90   | 17.16            | 17.6*   | 1.03x  | 1.17x     |
| 8 |  384  | 31.25  | 28.53   | 30.14            | 31.53   | 1.05x  | 1.01x     |
| 8 |  512  | 41.66  | 41.71   | 43.51            | 46.11   | 1.06x  | 1.11x     |
| 8 |  640  | 46.41  | 54.46   | 55.60            | 60.27   | 1.08x  | 1.30x     |
| 8 |  768  | 49.58  | 65.10   | 65.80            | 72.76   | 1.11x  | 1.47x     |
| 8 | 1024  | 54.30  | 82.15   | 83.32            | 91.81   | 1.10x  | 1.69x     |

\* one shape (B=8, N=256) timed out in the sweep so v2 number is from an
isolated retry.

**Verdict:** v2 wins or ties v1 at every shape, with the biggest wins on big
sequences (B=8 N=1024 +10 %). The win is essentially "stop spilling". Now
v2_kv64 is the working best.

The kernel is still ~9 % of H100 BF16 peak at B=8 N=1024 — the extra
pair-bias add and res-mask add on top of attention put a hard ceiling on
this. v1's ceiling was ~8.4 %; v2 lifts it to ~9.3 %.

---

## v3_no_pp — drop FA3 ping-pong barriers

Same kernel as v2_kv64 but the `fwd_named_bar_*` calls that gate WG0/WG1
ping-pong are removed. The hypothesis: with kv_height now 64 the att_block
is half-size and softmax is short enough that the named-bar synchronisation
overhead costs more than the inter-WG overlap buys.

It wins or ties v2_kv64 at every shape (~1-7 % lift on the big-N cases).

| B | N_CTX | v2_kv64 | v3_no_pp |
|---|-------|---------|----------|
| 1 |  512  | 10.12   | 10.82    |
| 1 |  640  | 15.45   | 16.15    |
| 4 |  512  | 30.93   | 31.98    |
| 4 |  768  | 53.67   | 55.86    |
| 4 | 1024  | 74.06   | 75.46    |
| 8 |  512  | 46.11   | 47.35    |
| 8 |  768  | 72.76   | 74.71    |
| 8 | 1024  | 91.81   | 95.01    |

So FA3-style ping-pong is **net negative** for EvoAttention at this
config. The barrier overhead exceeds the overlap benefit. Keeping CW=2 for
the smem layout but letting the two consumer WGs run independently is
faster.

---

## v4_stages3 — drop K/V/pb pipeline depth from 4 → 3

With kv=64, stages=4 used 145 KB shared mem. Dropping to stages=3 (~113 KB)
keeps the same producer/consumer pattern but uses less smem. Numbers are
essentially a wash with v3 (mostly within ±2 %) but slightly better on
average. We adopt stages=3 going forward — frees ~32 KB smem for future
candidates.

---

## v5_combined_sem — merge pb_arrived and rm_arrived

Combine the pair-bias and res-mask "load arrived" mbarriers into one
(`pbrm_smem_arrived[stage]`) — drops one `wait()` per inner-loop iteration
from 4 to 3. Numbers are essentially tied with v4 (some shapes ±1-2 %).
Keep the change going forward because it strictly reduces work.

---

## v6_one_sem — pool ALL per-stage loads on one mbarrier

K, V, pair-bias, res-mask all share one `load_arrived[stage]` semaphore.
Inner loop drops to **one** `wait()` (was 4 in v2/v3, 3 in v5). Cost: K
finishing fast no longer lets QK MMA start early without V also being
ready — but in practice K and V are issued together and finish ≈ together,
so the lost overlap is negligible.

| B | N_CTX | v4 stages3 | v6 one_sem |
|---|-------|------------|------------|
| 1 |  128  | 0.72       | 0.75       |
| 1 |  384  | 6.00       | 6.23       |
| 1 |  768  | 22.10      | 22.63      |
| 1 | 1024  | 35.87      | 37.42      |
| 4 |  768  | 55.83      | 56.51      |
| 4 | 1024  | 77.07      | 77.57      |
| 8 |  128  | 5.38       | 5.55       |
| 8 |  640  | 61.18      | 63.52      |
| 8 |  768  | 74.94      | 75.59      |
| 8 | 1024  | 95.59      | 97.05      |

Wins 1-4 % at most shapes, ties or ≤1 % loss elsewhere. **New best.**

---

## v7_intra_pipe — intra-WG QK/PV pipelining (did not work)

FA3-style trick within a single consumer warpgroup: issue PV(k) and QK(k+1)
back-to-back so two WGMMAs are queued, then drain both at the top of the
next iter. The pseudo-shape per iter goes from

    QK(k) → softmax(k) → PV(k)   (3 stages, sequential)

to

    drain[PV(k-1), QK(k)] → softmax(k) → issue PV(k) + QK(k+1)  (queued)

It compiled and was correct, but ran 5-7 % **slower** at the big shapes
(B=8, N=1024: 96.7 → 90.4 TFLOP/s). Root cause: the per-iter `arrive
compute_done[k-1]` was moved one iteration later (to right after the
drain), so the producer's `wait(compute_done[…])` lags by one iter. That
shrinks the producer's effective pre-fetch headroom from 2 iters (v6)
to 1 iter (v7). At the latency a single TMA pair takes, that gap can't
be hidden any more and the consumer stalls on `wait(load_arrived[s+1])`
before issuing QK(k+1). v8 tried stages=4 to widen the buffer; it was
still slower than v6 (same producer-lag mechanism).

Lesson: intra-WG pipelining is only a win if the producer can be made
to stay 2+ iters ahead. With our current producer-consumer split that
would need a real protocol rework, not just reordering the consumer.

---

## v9_more_regs — bump consumer regs 160 → 224

ptxas obliges and uses 168 registers instead of 160. No spill in either
version. Numbers are within ±1-2 % of v6 in both directions — noise. The
extra regs aren't unlocking better scheduling on top of v6's already-no-
spill code, so we leave the cap where it is.

---

## v10_overlap_load — move pb / rm smem load before mma_async_wait

In v6 the pb_reg / rm_reg load + LOG2E scaling sits AFTER the
mma_async_wait that drains the QK WGMMA. v10 hoists those reads + scales
before the wait so they overlap with the in-flight QK MMA on the tensor
cores.

Within ±2 % of v6 on every shape. ptxas-level scheduling on H100 is good
enough that this reordering is mostly a no-op — the compiler already
slots the swizzled smem load alongside the WGMMA where it can.

---

## v11_cw1_2cta — CW=1, 2 CTAs per SM

A targeted attack on small-grid shapes (B=1, N≤512). Drop CW from 2 to 1
and set `__launch_bounds__(256, 2)` so two CTAs share each SM. Each SM
holds 2 small CTAs (256 threads each) instead of one big CTA (384), which
helps hide WGMMA + softmax latency by giving the SM scheduler a second
CTA to switch to.

Smem per CTA falls to ~80 KB, so 2 × 80 KB = 160 KB fits in the 228 KB
SM budget. Consumer regs lifted to 224 (kittens recipe for CW=1, 2
blocks/SM).

The trade-off: we lose CW=2's "two consumer WGs share one K/V load"
smem reuse, so per-CTA work goes up. We hope the 2-CTA latency hiding
covers the loss.

| B | N_CTX | v6   | v11  | best |
|---|-------|------|------|------|
| 1 |  128  | 0.72 | 0.78 | v11  |
| 1 |  256  | 2.94 | 2.86 | v6   |
| 1 |  384  | 6.29 | 6.41 | v11  |
| 1 |  512  | 10.51| 10.74| v11  |
| 1 |  640  | 15.86| 14.93| v6   |
| 1 | 1024  | 37.03| 33.53| v6   |
| 4 | 1024  | 77.03| 66.71| v6   |
| 8 | 1024  | 96.76| 81.07| v6   |

v11 wins B=1 at small N by 2-8 %. v6 wins almost everywhere else,
sometimes by 15 %+ (B=4 N=1024, B=8 N=1024). The wins for v11 are within
do_bench noise (±5 %) and within do_bench variance from run to run; the
losses for v11 at large N are real and large.

**Decision: keep v6 as the single universal kernel.** A dispatcher (v11
on small grids, v6 elsewhere) would carry two binaries + a runtime
branch for an at-most 8 % win on shapes that are already < 1 TFLOP/s in
absolute terms. Not worth it; the absolute small-N performance is
limited by total work (B=1 N=128 is 8 GFLOPs of math, runs in ~10 µs
no matter what kernel).

---

## Final pick: best.cu (≡ v6_one_sem.cu)

Going-into-the-sweep numbers: prod-tk hit 4.78 – 82.15 TFLOP/s across
(B, N_CTX). best.cu hits 5.66 – 96.65 TFLOP/s on the same sweep — a
clean 13 – 28 % lift over production at every shape and a 1.0 – 1.78×
lift over Triton.

Headline changes from the production kernel:

1. `kv_height` lowered from 128 → 64. Halves the live size of `att_block`,
   `att_block_mma`, `pb_reg` and eliminates the 36-byte D=64 register
   spill that production has. (v2)
2. FA3-style ping-pong barriers removed. With kv_height now 64, softmax
   is short enough that the named-bar overhead between WG0 and WG1
   exceeds the overlap it was buying. (v3)
3. Pipeline depth dropped 4 → 3 stages (saves ~32 KB smem; perf wash). (v4)
4. `K`, `V`, `pair_bias`, `res_mask` per-stage mbarriers pooled onto a
   single combined `load_arrived[stage]`. Inner loop drops from four
   `wait()` calls to one. (v5 → v6)

Things that **did not pan out** (negative results worth keeping):
* Intra-WG QK/PV pipelining (v7/v8): starved the producer.
* Bumping consumer regs to 224 (v9): no change, compiler doesn't want
  them.
* Hoisting pb/rm smem loads before mma_async_wait (v10): compiler
  already does this scheduling.
* CW=1 with 2 CTAs/SM (v11): only wins at B=1 small-N by a few %;
  loses 5-15 % at large N. Not worth dispatch complexity.

What we did **not** try that would still be on the table:
* `qo_height` 64 → 128 with CW=1 (single big Q tile per CTA). Would
  amortize each kv iter's softmax over more Q rows. Needs care because
  att_block grows back to rt_fl<32,64> per warp = 64 fp32 regs/thread
  which probably re-spills at D=64.
* Persistent kernel that loops over heads × q-tiles within one CTA.
  Mostly a small-grid micro-opt and small-grid is already capped by
  total work, so unlikely to move the needle.
* Pre-scaling pair_bias by LOG2E into a separate fp32 smem buffer to
  cut one `mul` from the inner loop. Costs smem.
* WG-level cluster TMA multicast for pair_bias. Pair-bias is broadcast
  over N_SEQ; in our benchmark N_SEQ=1 so multicast has nothing to
  share. Could help when N_SEQ > 1 (the actual MegaFold call sites).

---

## v12_mma_sync — FlashKDA-style: TMA + warp-level mma.sync, 2 CTAs/SM

Motivated by FlashKDA's fwd_kernel2.cuh: keep TMA for bulk async loads
but switch the inner-loop GEMMs from warpgroup WGMMA to **warp-level
mma.sync** (`SM80_16x8x16_F32BF16BF16F32_TN`, the m16n8k16 PTX). The
register pressure per warp goes way down (no wgmma accumulator state),
which lets us run two CTAs per SM via `__launch_bounds__(256, 2)`.

Shape: CW=1 (1 producer + 1 consumer warpgroup, 256 threads/CTA),
qo=64, kv=64, stages=3. Each of the 4 consumer warps owns 16 Q-rows
and pulls the FULL 64×D K and V tiles into its own registers per kv
iter — the same per-warp register-broadcast pattern that production
`evo_bwd_mma_ker` already uses for backward at seq_len ≤ 512.

ptxas: 128 regs/thread (vs 160 for v6) at D=64, zero spills. Smem
~80 KB → 2 CTAs/SM = 160 KB, fits in the 228 KB H100 SM budget after
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize, MAX_SMEM/blocks_sm)`
(common_wrapper now reads blocks_sm out of the tile_dims).

Quick aside on intermediate steps:
* v13_mma_kv32 (kv_height 64 → 32) deadlocked the kernel — `sv_bf<32>`
  is below the TMA 128-byte 1-D vector minimum and the load never
  arrives. Stayed at kv=64.
* v13_mma_stages4 (stages 3 → 4): a wash with v12.

| B | N_CTX | best (v6 WGMMA) | v12 (mma.sync, 2 CTAs/SM) | v12/best |
|---|-------|-----------------|---------------------------|----------|
| 1 |  128  | 0.75            | 0.73                      | 0.97x    |
| 1 |  256  | 2.74            | 2.88                      | 1.05x    |
| 1 |  384  | 6.22            | 6.13                      | 0.99x    |
| 1 |  512  | 10.68           | 10.41                     | 0.97x    |
| 1 | 1024  | 36.19           | 36.53                     | 1.01x    |
| 4 |  640  | 42.48           | 43.03                     | 1.01x    |
| 4 | 1024  | 78.30           | 74.66                     | 0.95x    |
| 8 |  128  | 5.40            | 5.41                      | 1.00x    |
| 8 | 1024  | 96.68           | 93.28                     | 0.96x    |

v12 is **competitive** — within 5 % of v6 on every shape — and wins on a
few shapes (B=1 N=256 +5 %, B=8 N=128 ~tie, B=1 N=1024 ~tie). It loses
3-5 % at large N where WGMMA's higher per-instruction throughput beats
the occupancy advantage.

## v14_mma_tune — v12 + producer=24, consumer=232, hoisted pb/rm load

Push the register split to its max (24 producer, 232 consumer = 32768
regs total per CTA at 2 CTAs/SM); hoist pb/rm shared-mem reads above
the QK MMA so the compiler can interleave their ldmatrix-style reads
with the early m16n8k16 instances inside `warp::mma_ABt`.

Modest, shape-dependent improvement over v12:

| B | N_CTX | best  | v12   | v14   |
|---|-------|-------|-------|-------|
| 1 |  128  | 0.73  | 0.76  | 0.77  |
| 1 |  512  | 10.36 | 10.75 | 10.84 |
| 1 |  768  | 21.78 | 21.85 | 21.88 |
| 1 | 1024  | 35.56 | 35.28 | 35.72 |
| 4 |  128  | 2.86  | 2.90  | 2.96  |
| 4 |  384  | 19.94 | 20.06 | 20.10 |
| 4 | 1024  | 77.67 | 74.14 | 74.34 |
| 8 | 1024  | 97.06 | 92.68 | 94.01 |

v14 wins at most B=1 shapes (small N especially, +5 % at N=128 and
+5 % at N=512). Loses 3-5 % at large compute-bound shapes
(B=4–8, N=1024).

**Verdict.** Neither v12 nor v14 is a universal upgrade over v6. The
WGMMA throughput advantage at large N keeps v6 ahead overall. The
mma.sync + 2-CTAs/SM design is a meaningful alternative specifically
for very small grids (B=1, small N) where there are not enough total
work units to fill the SMs even at 1 CTA/SM, and the extra in-flight
CTAs per SM bought by 2-CTAs/SM helps hide WGMMA-vs-mma.sync latency
differences. The win there is 3-5 %.

I'm **leaving best.cu pointed at v6 (WGMMA)** — a clean ~15-20 % lift
over the production kernel at every shape — and keeping v12 / v14 as
documented alternatives with their own .cu / .so files. A dispatcher
that switches to v14 only when total CTAs in v6's grid drops below
~132 (one wave on H100) would buy back the B=1 small-N delta, but the
absolute work at those shapes is so small (< 10 GFLOPs) that the
end-to-end impact on EvoAttention forward in MegaFold-scale calls is
negligible.



