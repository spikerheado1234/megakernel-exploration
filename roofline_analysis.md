# Memory vs. Compute: A Per-Pipe Roofline Framework

From GEMM to FlashAttention to EvoAttention on H100.

---

## 1. The model

Roofline is **not** a statement about where operands come from. It is a contention
argument at a cut.

Draw a cut anywhere in the machine. Let `b_X` = bytes (or ops) that cross cut `X`
during the kernel, `R_X` = that cut's rate. The kernel cannot finish faster than
`b_X / R_X`, because that much data physically has to traverse a link of finite
capacity. This holds for every cut, **independently**:

```
t  >=  max( F/P_tensor,  b_dram/BW_dram,  b_L2/BW_L2,
            b_smem/BW_smem,  n_mufu/R_mufu,  n_cuda/R_cuda )
```

Divide through by `F` (tensor FLOPs) and each term becomes a roofline test with its
own intensity `I_X = F / b_X` and its own ridge point `P_tensor / R_X`.

**Two consequences that get missed:**

- A cut's intensity is meaningful even though every operand eventually lands in
  registers. DRAM bandwidth is a finite shared resource; if the kernel demands bytes
  across it faster than it can deliver, the ALUs starve regardless of the operand path.
- The intensity number alone means nothing. Only the **ratio to that level's ridge**
  does. Intensity 32 at the register boundary is fine; intensity 32 at DRAM is fatal.

### Why byte counts differ across cuts

If every level moved the same bytes, only the last hop would matter. They don't,
because reuse happens *between* the levels:

```
b_dram  <<  b_L2->smem  <<  b_smem->RF
```

One element of A crosses DRAM once (if L2 does its job), gets pulled from L2 into
SMEM once per tile that needs it, and gets read out of SMEM once per warp-tile that
touches it. Same element, three different multiplicities. Each ratio is an
**amplification factor**, and each is a tuning knob.

The ordering holds when (a) all data originates in DRAM and (b) each cache captures
the reuse presented to it. It is not a theorem. It breaks on register spills,
split-K partials, and fused kernels that manufacture on-chip intermediates
(FlashAttention's `S` and `P` are `O(N^2)` SMEM traffic that never touches DRAM).

---

## 2. The five ceilings (H100 SXM5)

132 SMs, ~1.755 GHz, bf16/fp16 dense.

| Pipe | Rate | Demand unit | Ridge (tensor flop/unit) | Set by |
|---|---|---|---|---|
| DRAM <-> L2 | 3.35 TB/s | byte | **295** | L2-resident working set, swizzle, clusters |
| L2 -> SMEM | ~7 TB/s [1] | byte | **141** | block tile, TMA multicast |
| SMEM <-> RF | ~30 TB/s | byte | **33** | warp tile; ~0 if wgmma reads SMEM |
| CUDA core fp32 | ~29.7 T op/s | fp32 op | **33** | epilogue, softmax, rescaling |
| MUFU (SFU) | 3.71 T op/s | `ex2`/`rcp`/`rsqrt` | **267** | softmax, GELU, norms |
| **Tensor core** | 990 TFLOP/s | — | reference | — |

SMEM: 128 B/cyc/SM. CUDA fp32: 128 ops/cyc/SM. MUFU: **16** results/cyc/SM.

[1] The L2 figure is the least trustworthy row: published measurements span
5–7.5 TB/s depending on access pattern and read/write mix. At 5 TB/s the ridge is 198.

**On the MUFU rate.** A common error is quoting ~16.75 TFLOP/s for `ex2` on H100.
That is `67 / 4` — the A100 SFU ratio (1:4) applied to Hopper, where FP32 lanes
doubled to 128/SM but SFU lanes stayed at 16/SM, making it **1:8**. The remaining
2x is a units error: FLOPS counts an FMA as 2 flops, but `ex2` produces 1 result.
Correct figure: 16 results/cyc/SM = **3.71 T/s**.

Also distinguish `expf(x)` (range reduction + degree-6 polynomial on FP32 cores,
~15–20 FFMA) from `__expf(x)` / `ex2.approx.f32` (one `MUFU.EX2`). Attention kernels
emit the latter and fold `log2(e)` into the `1/sqrt(d)` prescale.

---

## 3. GEMM: intensity is a design variable

### Derivation

For `C = A·B` with M, N, K, tile `BM x BN`, K-loop chunked by `BK`, element size `e`:

**Per output tile**, the k-loop walks a full `BM x K` row panel of A and a
`K x BN` column panel of B, then writes `BM x BN` of C:

```
bytes/tile = e * (BM*K + K*BN + BM*BN)
```

`BK` cancels. It sets SMEM footprint and pipeline depth, never byte volume.

**Times `(M/BM)(N/BN)` tiles**, distributing the prefactor across each term:

```
bytes = M*N*K*e * (1/BM + 1/BN)  +  M*N*e
          \___ redundant re-reads ___/     \_ C, compulsory _/
```

The A term keeps `1/BN` because each of the `N/BN` tile-*columns* re-reads all of A.
The C term cancels completely — C is written once regardless of tiling.

### Intensity

With `M = N = K` and `BM = BN = T`:

```
I = 2N^3 / [ N^3*e*(2/T) + N^2*e ]  =  (2/e) * TN/(2N + T)  -->  T/e
```

**N appears only in the C write-back term.** fp16, T=128: N=4096 gives 63.0,
N=8192 gives 63.5, asymptote 64. Doubling the problem size does not change the
intensity, because both FLOPs and bytes scale as `N^3`.

### The three intensities

| | Formula | N=4096, fp16, T=128 |
|---|---|---|
| Tile-induced (L2->SMEM) | `T/e` | 64 |
| Achieved DRAM (G-tile L2 wave) | `G*T/e` | ~512 (G=8) |
| Compulsory (infinite cache) | `2N/(3e)` | 1365 |

Compulsory is a **ceiling, not a rung**: infinite cache captures all available reuse.
Adding cache levels moves you *up toward* it. And it is one continuous ladder — set
`G = N/T` and `G*T/e = N/e`, which is compulsory up to the C term.

Compulsory is unreachable (it needs `Theta(N^2)` fast memory). The real floor is the
communication lower bound: with `S` elements of fast memory, matmul DRAM traffic is
`Omega(N^3 / sqrt(S))`.

### The square-root law

At every level, a square tile of dimension `T` needs `T^2` elements of fast memory
and delivers intensity `T/e`. So with capacity `S`:

```
I  ~  sqrt(S) / e
```

**Intensity scales as the square root of the capacity you block into.** Doubling
intensity costs 4x the SRAM. This is why the hierarchy exists: you cannot buy
300 flop/byte from one level, so you stack levels.

Corollary: square tiles are optimal. Minimizing `1/BM + 1/BN` subject to
`BM + BN = S` gives `BM = BN` (the expression `2*BM*BN/(BM+BN)` is the harmonic mean,
maximized at equality for fixed sum).

### Design consequences

- **Don't maximize intensity, buy exactly enough.** Pick T so `G*T/e` clears the
  ridge with 2–3x margin, then spend everything else on latency hiding.
- **Precision changes nothing on the roofline.** fp8 halves `e` but doubles peak
  FLOP/s; the ridge moves out just as fast. The invariant is tile *bytes*, not
  tile *elements*.
- **Shapes with no reuse can't be rescued.** GEMV has intensity `2/e` no matter the
  tiling, because there is no reuse of B to block for.
- **Fusion raises top-level intensity for free.** A fused epilogue removes a full
  `N^2` round trip.

### Worked example, N=4096, fp16, T=128, G=8, warp tile 64

| Cut | Bytes | Intensity | Amplification vs. cut above |
|---|---|---|---|
| compulsory floor | 100 MB | 1365 | — |
| DRAM <-> L2 | 268 MB | 512 | 2.7x (imperfect L2 reuse) |
| L2 -> SMEM | 2.1 GB | 64 | 8x (= G) |
| SMEM -> RF | 4.3 GB | 32 | 2x (= warp grid dim) |

`T/e` was never the DRAM intensity — it is the **L2->SMEM** intensity, because the
block tile *is* the SMEM staging granularity. The DRAM number is `T/e` amplified by
whatever L2 gives you.

**Diagnostic:** if a measured ratio is *below* the model's prediction, you have
invented traffic — spills, split-K, or a bug.

| Ratio | Should be | If too big | Fix |
|---|---|---|---|
| `b_dram` / compulsory | ~1–3 | L2 not capturing wave reuse | swizzle, cluster launch |
| `b_L2` / `b_dram` | = G | tiles not sharing panels | rasterization order, multicast |
| `b_smem` / `b_L2` | = warp grid dim | warp tile too small | larger warp tile, wgmma |

---

## 4. FlashAttention

Forward, non-causal, per (batch, head), sequence `S`, head dim `d`,
Q-tile `B_r`, KV-tile `B_c`. `F = 4*S^2*d`.

### Closed forms

```
I_dram  =  4S^2 d / (4 S d e)          =  S / e
I_L2    =  4 S B_r / ( e (B_r + 2S) )  ->  2 B_r / e
I_mufu  =  4 S^2 d / S^2               =  4 d      (tensor flops per exp)
```

Three observations:

1. **`I_dram = S/e` has no tile parameter in it.** Attention's DRAM intensity is
   sequence length over element size. Nothing in the kernel can change it.
2. `I_L2 = 2*B_r/e` is **double** GEMM's `T/e`, because one K/V staging feeds both
   GEMMs. A structural gift from fusion. It is independent of `d` and `S`.
3. `I_mufu = 4d` depends on **head dimension alone**. Softmax pressure is a `d`
   phenomenon, not an `S` phenomenon.

### The critical sequence length

Setting `S/e = P/BW_dram`:

```
S_crit = e * P / BW_dram  =  2 * 990/3.35  ~  590 tokens
```

Below ~590, attention is DRAM bound on compulsory traffic and **no tiling decision
helps**. fp8 halves `e` and doubles `P`, leaving `S_crit` unchanged.

### SMEM <-> RF is architecturally deleted on Hopper

`wgmma` reads Q, K, V from SMEM via descriptors. `P` is born in the accumulator and
consumed as the A operand of the second GEMM without leaving registers. Softmax
reductions span 4 threads in a quad -> `shfl`, not SMEM. So `b_smem ~ 0`.

On Ampere the same kernel needs `ldmatrix` for every K and V fragment. This cut is
not a fixed feature of the machine; it has been actively engineered away.

### Two regimes

| | S = 8192, d = 128, `B_r`=128 | S = 384, d = 128, `B_r`=128 |
|---|---|---|
| DRAM | 8 MB, I=4295, 2.4 us — 15x headroom | 384 KB, **I=192 < 295** — bound |
| L2 -> SMEM | 270 MB, I=127, **38.6 us — binding** | 672 KB, I=110 |
| SMEM -> RF | ~0 | ~0 |
| MUFU | I=512 vs 267 — **1.9x over budget** | same, 40 ns |
| CUDA fp32 | ~6 us | ~15 ns |
| Tensor | 34.7 us | 76 ns |
| Ceiling | ~90% (fixable) | **65% (structural)** |

**Long S** is a *throughput* problem: enough work to saturate every pipe, and the job
is arranging overlap. The L2 cut is fixed by 2-CTA cluster + TMA multicast on K/V
(270 -> 136 MB, I=253). MUFU cannot be reduced — `S^2` exponentials are the algorithm
— so it must be **hidden**, which is exactly FA3's ping-pong warpgroup schedule.

**Short S** is a *latency and occupancy* problem. The DRAM floor is compulsory, so the
65% ceiling cannot be lifted from inside the kernel. Worse, every fixed cost stops
amortizing: 3 K-loop iterations against a 3–4 stage pipeline means the mainloop *is*
the prologue; ping-pong gets one overlapped iteration out of three; a 3–5 us launch
overhead dwarfs 76 ns of tensor work; and batch-1 x 16 heads = 48 CTAs leaves 64% of
the GPU idle.

Fixes at short S live outside the kernel: **fuse with the QKV projection** (the only
thing that moves the ceiling), **GQA** (8 query heads per KV head gives
`I_dram = 1.78 S/e = 341 > 295`, moving `S_crit` to ~330), persistent kernels, and
CUDA graphs.

### `B_r` and `B_c` are different knobs

| | `B_r` (Q rows) | `B_c` (KV cols) |
|---|---|---|
| L2 intensity | `2 B_r / e` | **no effect** (cancels) |
| CTA count / occupancy | `S / B_r` | none |
| Warpgroup count, ping-pong | 64 rows per WG | none |
| CUDA-core rescale cost | none | `S^2 d / B_c` |
| K-loop trips (pipelining) | none | `S / B_c` |
| SMEM per stage | Q only, once | `2 B_c d e` |

`B_r` is the intensity + occupancy knob; `B_c` is the pipelining + softmax-overhead
knob and is invisible to the roofline. Tying them together is what makes shrinking
the tile look like a bad trade. `B_r`=128 with `B_c`=64 dominates `B_r`=`B_c`=64 on
every axis except CTA count.

Note that `B_r`=64 crosses the `wgmma` M=64 boundary: one consumer warpgroup instead
of two, which **removes the mechanism that hides softmax behind MMA**. The bytes say
"1.9x worse"; the scheduling says something the bytes don't.

---

## 5. EvoAttention: when a tensor has no reuse

AlphaFold-3 style attention. Same structure as FlashAttention plus:

- `pair_bias`: `(B, H, N, N)`, added to the logits, **broadcast across the MSA
  dimension `N_SEQ`**
- `res_mask`: `(B*N_SEQ, 1, 1, N)`, broadcast across heads and the Q dimension

### The structural fact

`pair_bias` has **one element per logit**. Within a `(batch_msa, head)` it has
**zero reuse** — each element is consumed exactly once, by exactly one `(q_i, k_j)`
pair. Its only reuse axis is `N_SEQ`, and that reuse can only ever be captured in L2,
never in SMEM.

Per logit element the kernel does `4d` tensor flops and must move `e` bytes of bias:

```
I_L2(ceiling)  =  4 N^2 d / (N^2 e)  =  4d / e
```

**At d=64 that is 128, against an L2 ridge of 141.** No tile size reaches it.
EvoAttention at D=64 **cannot be made L2-compute-bound on H100.** In FlashAttention
`I_L2 = 2 B_r/e` grows without bound in `B_r`; here the tile only interpolates toward
a wall.

### Exact forms

Per `(batch_msa, head)`, with `R` = Q rows per CTA, `M = B*N_SEQ`:

```
b_L2   = e*N * [ 2d + 2Nd/R + N + N/R ]
         Q,O    K,V restream   pb    rm

I_L2  -->  4d / ( e * (1 + 2d/R) )          [ N >> d ]

b_dram = M*H*4*N*d*e  +  B*H*N^2*e  +  M*N*e
                          pb, once per (b,h)

I_dram = 4Nd / ( e * (4d + N/N_SEQ + 1/H) )
```

`N_SEQ` is the **only** amortizer of pair-bias at the DRAM level. At `N_SEQ`=1 you pay
`N^2 e` of DRAM traffic for `4 N^2 d` flops. As `N_SEQ -> inf` you recover
FlashAttention's `N/e`.

### Per-pipe times, D=64, R=128 (times in microseconds, full machine)

| Case | | DRAM | L2->SMEM | SMEM->RF | MUFU | CUDA | Tensor | Binding | Ceiling |
|---|---|---|---|---|---|---|---|---|---|
| B1 S1 H4 N384 (12 CTAs) | Evo | **0.59** | 0.39 | 0.05 | 0.16 | 0.19 | 0.15 | DRAM | 26% |
| | FA | 0.23 | 0.22 | 0.01 | 0.16 | 0.11 | 0.15 | DRAM | 65% |
| B2 S1 H16 N384 (96) | Evo | **4.70** | 3.16 | 0.37 | 1.28 | 1.51 | 1.22 | DRAM | 26% |
| | FA | 1.88 | 1.80 | 0.05 | 1.28 | 0.88 | 1.22 | DRAM | 65% |
| B1 S32 H4 N384 (384) | Evo | 7.87 | **12.63** | 1.49 | 5.13 | 6.05 | 4.88 | **L2** | 39% |
| | FA | 7.51 | 7.19 | 0.21 | 5.13 | 3.50 | 4.88 | DRAM | 65% |
| B1 S64 H4 N256 (512) | Evo | 10.18 | **12.02** | 1.41 | 4.56 | 5.38 | 4.34 | **L2** | 36% |
| | FA | 10.02 | 7.19 | 0.28 | 4.56 | 3.11 | 4.34 | DRAM | 43% |
| B1 S4 H4 N384 D128 | Evo | 2.23 | **3.83** | 0.21 | 0.64 | 0.80 | 1.22 | **L2** | 32% |
| B1 S16 H4 N384 D96->128 | Evo | 7.87 | **15.32** | 0.85 | 2.57 | 3.18 | 4.88 | **L2** | 32% |

Intensities are shape-invariant: Evo `I_L2` = 54.7, `I_dram` = 76.8 at `N_SEQ`=1
rising to 183 at `N_SEQ`=32. FA: 96 and 192.

### How the bottleneck shifts

**1. `N_SEQ` flips DRAM <-> L2.** At `N_SEQ`=1, pair-bias is read once from HBM per
`(b,h)` and never amortized: `I_dram` = 77 vs. FA's 192, DRAM binds at a 26% ceiling.
At `N_SEQ`=32 the pb DRAM cost divides by 32, `I_dram` climbs to 183, and the binding
cut moves down to **L2->SMEM** — where it stays forever, because L2 traffic does not
amortize at all. FlashAttention never shows this transition.

**2. SMEM->RF comes back from the dead.** In FA3 this cut is ~0. Here pair-bias must
be explicitly loaded into registers to be added to the logits: a full `N^2` bf16
tensor every iteration. Intensity 110 vs. a 33 ridge, so not binding — but it costs
an `rt_fl<16,128>` register tile = **64 registers/thread**, which with the attention
accumulator (64) and output accumulator (32) is exactly a 160-register budget.
Register pressure is a second-order cost of the same fact.

**3. The compute pipes bind at small `d`.** Per logit you get `4d` = 256 tensor flops
to pay for the softmax:

| Pipe | Demand/logit | Budget (ridge) | Evo d=64 | FA d=128 |
|---|---|---|---|---|
| MUFU | 1 `ex2` | 267 flop | 256 -> **1.04x over** | 512 -> 0.52x |
| CUDA fp32 | 9 ops (Evo) / 5 (FA) | 33.3 flop/op | 28.4 -> **1.17x over** | 102 -> 0.33x |

Evo's 9 ops/logit: scale, pb convert, pb x log2e, +pb, +res_mask, row_max, sub_row,
row_sum, convert-to-bf16. FlashAttention has 5 — the extra 4 are all bias/mask
application.

Both non-tensor pipes exceed the tensor-core time at d=64. Ping-pong scheduling
therefore runs **backwards** from FA3: you are hiding MMA behind softmax, not softmax
behind MMA. Even with perfect overlap the CUDA-core pipe alone caps you at ~81% of
tensor peak before any memory consideration.

**4. Padding is pure loss.** D=96->128 does 1.33x the tensor work and 1.33x the QKV
traffic; D=32->64 does 2x both.

### Tile-size sweep (N=384, d=64, `B_c`=128, stages=3)

| R (Q rows/CTA) | Evo `I_L2` | FA `I_L2` | Evo SMEM/CTA | FA SMEM/CTA | Feasible |
|---|---|---|---|---|---|
| 64 | 38 | 55 | 152 KB | 104 KB | yes |
| **128** | **55** | **96** | **208 KB** | 112 KB | yes (at the wall) |
| 256 | 70 | 154 | 320 KB | 128 KB | no |
| 512 | 81 | 219 | 544 KB | 160 KB | no |
| inf | **128 (hard ceiling)** | unbounded | — | — | — |

Two things collapse at once, and they are the same fact twice:

- **The payoff saturates.** Doubling R takes FA 96 -> 154 (+60%) but Evo 55 -> 70
  (+27%), heading for a wall at `4d/e` = 128.
- **The price explodes.** FA's tile cost is `R*d*e` (Q only) — 8 KB per doubling.
  Evo's is `R*B_c*e*stages` — **96 KB per doubling**, because pair-bias staging scales
  with `R x B_c`, not `R x d`. Since `B_c` = 128 > d = 64, pb dominates SMEM.

The usual lever — buy intensity with a bigger tile — is unavailable in both directions
simultaneously. R=128 with stages=3 is the constrained optimum.

### What moves the needle

1. **Fold pair-bias into the wgmma accumulator.** Pre-scale `pair_bias` by
   `1/softmax_scale` offline, load it directly into the attention accumulator, then use
   an *accumulating* `mma_ABt` instead of an overwriting `mm_ABt`. Deletes the separate
   bias register tile (**64 registers/thread back**) plus two `N^2` elementwise ops.
   CUDA-core count 9 -> 7, bringing that pipe to parity with the tensor cores.
2. **Fix occupancy at `N_SEQ`=1.** 12 CTAs on 132 SMs means every SM-local pipe runs
   at 11x the tabulated time: tensor 1.65 us, CUDA 2.09 us, against 0.59 us of DRAM.
   The roofline is irrelevant there — it is pure parallelism starvation, ~7% of peak.
   Split-KV with a logsumexp combine pass is the standard fix.
3. **Exploit `res_mask` block sparsity.** It is binary and, in real MSA data,
   contiguous padding. Skipping fully-masked `B_c` blocks cuts *everything*
   proportionally. Random per-element masks in a test harness destroy this structure.
4. **Cluster + TMA multicast on K/V.** Cuts the `2Nd/R` term: `I_L2` 55 -> 70. Does
   not help pb (each CTA needs different pb rows), which is why the gain is modest.
5. **fp8 pair-bias.** Halves the dominant L2 term and raises the ceiling to
   `4d/e_pb` = 256. The only lever that moves the wall itself.

---

## 6. Summary: what the framework buys

Roofline's role is **not** to find the bottleneck — a profiler does that better, with
real numbers. Its role is:

- **Before writing code**, predict which cut will bind from tile parameters alone, so
  you build the right kernel first. A 128x128 tile with G=1 caps you at intensity 64
  against a 295 ridge; that is knowable in thirty seconds on paper and takes a day to
  discover by implementing.
- **Quantify the fix**, not just its direction. "You need `T/e >= 140`, so `T >= 280`,
  so 256x256, which needs 128 KB of SMEM at BK=64 double-buffered" — including whether
  it fits, and whether it is even sufficient.
- **Distinguish the two kinds of failure.** A cut fails either because `b_X` is too
  big (fix the *tiling*: bigger tile, better swizzle, multicast — this changes the
  intensity) or because achieved bandwidth is far below peak (fix the *access pattern*:
  vectorization, XOR-swizzled SMEM layouts, coalescing — this does not change the
  intensity at all). A profiler's "SMEM bound" verdict does not tell you which world
  you are in.
- **Recognize unfixable bottlenecks.** GEMV's `2/e`, attention's `S/e`, EvoAttention's
  `4d/e`. Knowing when to stop optimizing and change algorithms is worth more than any
  individual kernel win.

### Decision procedure

1. Compute `I_X` for every cut from tile parameters; compare each to its ridge.
2. Find the first failing cut. Fixing anything above it buys nothing.
3. Classify the failure: too many bytes (retile) or too slow (re-pattern)?
4. Check whether the fix fits in SMEM / registers / occupancy before implementing it.
5. Verify with `dram__bytes.sum`, `lts__t_bytes.sum`,
   `l1tex__data_pipe_lsu_wavefronts_mem_shared.sum`,
   `smsp__inst_executed_pipe_fp32`.
6. Compare measured amplification ratios to predicted ones. A ratio *below* prediction
   means invented traffic.

### What the model does not see

Wave quantization, pipeline fill and drain, launch overhead, MMA issue rate,
register bank conflicts, and warp scheduling. At long sequence these are second-order.
At short sequence they are most of the gap: the model may predict a 65% ceiling where
you measure 40%, and the missing 25 points live entirely in the machinery above.
Roofline gives you the ceiling; it does not tell you how far below it you are standing.
