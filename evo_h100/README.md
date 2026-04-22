# `evo_h100` — ThunderKittens H100 kernel for AlphaFold-3 / MegaFold EvoAttention

This directory contains a ThunderKittens port of the MegaFold Triton EvoAttention
kernel (pair-bias + residue-mask attention used for *attention-pair-bias* and
*triangle attention* blocks in AlphaFold 3). It is adapted from the stock
ThunderKittens MHA H100 forward kernel at
`kernels/attention/mha_h100/mha_h100.cu`.

The Triton reference it matches semantically is at
`~/MegaFold/megafold/model/FusedEvoAttention/evoattention.py` (function
`_attn_fwd` + `EvoformerAttention.forward`).

Files:
- `tk_evoattention.cu` — CUDA kernel and `torch` C++ extension (PyBind).
- `tk_evo_attention_layer.py` — `torch.autograd.Function` wrapper
  (`TKEvoAttention`) that handles reshape / cast / D padding.
- `test_tk_evoattention.py` — correctness tests vs. the Triton reference.
- `benchmark.py` — forward-pass speed comparison vs. Triton.
- `Makefile` — builds the PyTorch C++ extension (`_C.cpython-*.so`).

## 1. What the reference kernel does (semantics we must match)

For each `(batch, msa_idx, head)` the Triton kernel computes
```
logits = (Q * softmax_scale) @ K^T + pair_bias + res_mask
P      = softmax(logits, dim=-1)
O      = P @ V
```
with
- `Q, K, V` shape `(B, N_SEQ, N_CTX, H, D)` in the caller layout; internally
  transposed to `(B, N_SEQ, H, N_CTX, D)`.
- `pair_bias` shape `(B, 1, H, N_CTX, N_CTX)` — broadcast across the MSA axis.
- `res_mask` shape `(B, N_SEQ, 1, 1, N_CTX)` — broadcast across heads and queries.
- `softmax_scale = 1 / sqrt(D)`.

Unlike vanilla MHA there is **no causal mask**, there **is an additive pair
bias** (rank-4), and there **is an additive residue mask** (rank-1 along KV)
— both applied *inside* the online softmax, before `exp`.

## 2. Why the stock `mha_h100` kernel is not a drop-in

1. It is 4-D `(B, H, N, D)` only; EvoAttention is 5-D `(B, N_SEQ, H, N, D)`.
2. There is no pair-bias path: the inner loop goes
   `att = Q_smem @ K_smem^T → exp2(scale * att - max) → att @ V_smem`,
   with no slot for an additive bias tile or mask vector.
3. The parallel scheme is `(seq_q_blocks, H, B)` — does not expose `N_SEQ` as
   an independent batch axis.
4. The softmax-scale trick absorbs `scale * log2(e)` directly into the tile
   multiply; that's fine for plain QK but impossible once pair-bias is mixed
   in (the bias would get multiplied by `scale` too).

## 3. Changes introduced in `tk_evoattention.cu`

### 3.1 Shape convention (5-D → 4-D folding)

ThunderKittens `gl<>` only supports 4 dims. All 5-D inputs are folded to 4-D
by collapsing `B` and `N_SEQ`:

| Tensor      | Caller shape                  | Kernel-side 4-D shape                |
|-------------|-------------------------------|--------------------------------------|
| Q, K, V, O  | `(B, N_SEQ, N_CTX, H, D)`     | `(B*N_SEQ, H, N_CTX, D)`             |
| pair_bias   | `(B, 1, H, N_CTX, N_CTX)`     | `(B,       H, N_CTX, N_CTX)`         |
| res_mask    | `(B, N_SEQ, 1, 1, N_CTX)`     | `(B*N_SEQ, 1, 1, N_CTX)`             |
| L (logsumexp) | n/a                         | `(B*N_SEQ, H, 1, N_CTX)`             |

The fold is correct because the caller tensors are already contiguous in
`(B, N_SEQ, …)` order so `Q.view(B*N_SEQ, H, N_CTX, D)` is a true no-copy
reinterpretation.

### 3.2 Parallel scheme

Grid: `dim3(N_CTX / (CW * qo_height), H, B * N_SEQ)`.

- `blockIdx.x` : query-tile index (new: was the only Q axis before).
- `blockIdx.y` : head index.
- `blockIdx.z` : flattened `(batch, msa)`. Inside the kernel
  `batch_idx = blockIdx.z / N_SEQ` is used to index `pair_bias`
  (which is shared across MSA). `N_SEQ` is passed in via the globals struct.

This gives the same parallelism axes as the Triton kernel
(`(cdiv(N_CTX, BLOCK_SIZE_Q), B * N_SEQ * H)`).

### 3.3 Tile and pipeline dimensions

```
qo_height = 64   (CW * 16 * 4 rows per warpgroup)
kv_height = 128
D         ∈ {64, 128}
```
Per-D specialization for shared-memory budget:

| D  | consumer_warpgroups | stages | peak smem (approx.) |
|----|--------------------:|-------:|--------------------:|
| 64 |                   2 |      3 |          ~213 KB    |
| 128|                   1 |      2 |          ~165 KB    |

- `D=64` uses `stages=3` for a deeper TMA prefetch pipeline (~20-40% faster
  than `stages=2` on large `N_CTX`).
- `D=128` drops `CW` to 1 because the K/V tiles are twice as wide; keeping
  `stages=2` leaves enough smem headroom.

### 3.4 New shared-memory buffers

Added on top of the original Q / K / V / L / O allocations:

```cpp
pb_tile (&pb_smem)[CW][stages] = al.allocate<st_bf<qo_height, kv_height>, CW, stages>();
rm_vec  (&rm_smem)[stages]     = al.allocate<sv_bf<kv_height>,             stages>();
```

- `pb_smem` — one `(qo_height × kv_height)` bfloat16 pair-bias tile *per
  consumer warpgroup per pipeline stage*. Each warpgroup owns the block of
  rows matching its Q tile.
- `rm_smem` — one shared vector of length `kv_height` per stage. The residue
  mask is independent of the query row, so a single copy is shared across
  warpgroups.

### 3.5 New mbarriers

Two new `kittens::semaphore` arrays per kernel:
```cpp
__shared__ kittens::semaphore pb_smem_arrived[stages];
__shared__ kittens::semaphore rm_smem_arrived[stages];
```

Both are initialised with `(arrivals=0, transactions=1)`. The producer issues
one `expect_bytes + load_async` pair per stage, pooling `CW` pair-bias tiles
onto `pb_smem_arrived[s]` via a single `expect_bytes(sizeof(pb_tile) * CW)`
call.

### 3.6 Producer loop additions

In both the prologue (fill `stages-1` buffers) and the steady-state loop, the
producer now issues:

```cpp
warp::tma::expect_bytes(pb_smem_arrived[s], sizeof(pb_tile) * CW);
for (int wg = 0; wg < CW; wg++) {
    coord<pb_tile> pb_idx = {batch_idx, head_idx, seq_idx + wg, kv_idx};
    warp::tma::load_async(pb_smem[wg][s], g.pb, pb_idx, pb_smem_arrived[s]);
}

warp::tma::expect_bytes(rm_smem_arrived[s], sizeof(rm_vec));
coord<rm_vec> rm_idx = {batch_msa_idx, 0, 0, kv_idx};
warp::tma::load_async(rm_smem[s], g.rm, rm_idx, rm_smem_arrived[s]);
```

The `pair_bias` coordinate is in tile units `(batch, head, q_block, kv_block)`
and maps to the global row `(seq_idx + wg) * qo_height` and column
`kv_idx * kv_height`. The `res_mask` coord uses `sv_bf::length = kv_height`
so `c * length` gives the correct starting element.

### 3.7 Consumer inner loop — new stanzas

The single biggest algorithmic difference. Where the stock kernel ran
```cpp
mm_ABt(att, Q_smem, K_smem);        // raw
mul(att, att, scale * log2(e));     // bake scale + exp2 prefactor
sub_row(att, att, max * scale * log2(e));
exp2(att, att);
```
we instead keep the natural-log space until the bias has been applied, and
scale pair_bias + res_mask by `log2(e)` in registers so a single fused
multiply on `att_block` covers both `softmax_scale` and the base-2 conversion:

```cpp
// Issue Q @ K^T (async). Stash prev max while the matmul runs (different regs).
wait(k_smem_arrived[s], phase);
warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[s]);
warp::copy(max_vec_last, max_vec);
warpgroup::mma_async_wait();

// Fused (softmax_scale * log2e) — single pass over att_block covers both the
// Triton "Q*scale @ K" semantics and the base-2 conversion for exp2.
warp::mul(att_block, att_block, softmax_scale_log2);

// Add pair_bias, scaled by log2(e) in registers (free vs. scaling in gmem).
wait(pb_smem_arrived[s], phase);
warpgroup::load(pb_reg, pb_smem[warpgroupid][s]);   // bf16 ST → fp32 RT
warp::mul(pb_reg, pb_reg, LOG2E);
warp::add(att_block, att_block, pb_reg);

// Add res_mask via column broadcast, also log2(e)-scaled in registers.
wait(rm_smem_arrived[s], phase);
warp::load(rm_reg, rm_smem[s]);                     // bf16 SV → fp32 RV
warp::mul(rm_reg, rm_reg, LOG2E);
warp::add_col(att_block, att_block, rm_reg);
// att_block is now (Q*scale)@K*log2e + bias*log2e + mask*log2e — base-2 log.

// Online softmax (max_vec kept in base-2 log space throughout).
warp::row_max(max_vec, att_block, max_vec);
warp::sub(alpha, max_vec_last, max_vec);
warp::exp2(alpha, alpha);
warp::sub_row(att_block, att_block, max_vec);
warp::exp2(att_block, att_block);
warp::mul(norm_vec, norm_vec, alpha);
warp::row_sum(norm_vec, att_block, norm_vec);
warp::copy(att_block_mma, att_block);
warp::mul_row(o_reg, o_reg, alpha);

// P @ V (unchanged)
wait(v_smem_arrived[s], phase);
warpgroup::mma_AB(o_reg, att_block_mma, v_smem[s]);
warpgroup::mma_async_wait();
```

Key differences from the stock kernel:

1. `softmax_scale` and `log2(e)` are **fused into a single multiply** on
   `att_block` — matches Triton's `(Q*scale)@K` semantics while preparing for
   `exp2`. Saves a pass over the register tile vs. applying them separately.
2. `pair_bias` and `res_mask` are each multiplied by `log2(e)` in registers
   after the shared-to-register load; adds a cheap register mul and eliminates
   a third pass over `att_block` for the base-2 conversion.
3. `warp::copy(max_vec_last, max_vec)` is hoisted above `mma_async_wait()` so
   it runs in parallel with the WGMMA (different registers, no hazard).
4. `max_vec` is kept in base-2 log space for the lifetime of the kernel
   (simpler than the stock kernel's "raw max + scaled max" split, which was
   only useful when there was no per-iteration bias).
5. `alpha = exp2(prev_max - new_max)` is computed once and used for both
   `norm_vec` and `o_reg` correction.

Note: the log2(e) pre-scaling of `pair_bias` and `res_mask` was also tried on
the **Python side** (multiplying pair_bias by log2e once upfront) — this
appears free in theory, but empirically added a full elementwise-kernel pass
over a 256 MB fp32 tensor per forward call that dwarfed the in-kernel
register mul it saved. Keeping the mul in the TK kernel, where it operates
on registers only, is much faster.

### 3.8 Removed features

- **Causal masking** — EvoAttention is non-causal. The
  `template<int D, bool is_causal>` parameter and all `if constexpr
  (is_causal)` branches are gone.
- **GQA / hr > 1** — the kernel requires `qo_heads == kv_heads` (just set
  `kv_head_idx = head_idx`). Pair-bias is indexed per QO head anyway.
- **Backward kernel and dispatch** — this file is forward-only.

### 3.9 `L` (logsumexp) output

Still produced per query row, now in natural-log space as `max*ln(2) + ln(l_i)`
(recall `max_vec` is in base-2 units). Currently unused; reserved for when the
backward kernel lands.

### 3.10 Host-side API changes

`evoattention_forward(q, k, v, pair_bias, res_mask, n_seq, softmax_scale=0.0)`:

- All inputs must be bf16 CUDA contiguous.
- `n_seq` lets the kernel recover `batch_idx = blockIdx.z / n_seq` for the
  shared pair_bias indexing.
- `softmax_scale` is an optional positive override; when `<= 0` the kernel
  uses `1/sqrt(head_dim)`. The override is needed when the caller zero-pads
  the head dim (see §4).

## 4. Python-side padding for non-native head dims (D ∈ {16, 32, 96})

The kernel only accepts `D ∈ {64, 128}`. To support the EvoAttention head dims
that AF3 actually uses (attention-pair-bias is 96, triangle attention is 32,
legacy paths use 16), `TKEvoAttention` zero-pads Q/K/V along the last dim up
to the next supported kernel head dim:

| true D | padded D | rationale                                 |
|-------:|---------:|-------------------------------------------|
|     16 |       64 | next multiple supported                   |
|     32 |       64 | *triangle attention* native head dim      |
|     64 |       64 | no-op                                     |
|     96 |      128 | *attention-pair-bias* native head dim     |
|    128 |      128 | no-op                                     |

Zero-padding Q and K leaves `Q @ K^T` unchanged (padded slots contribute 0).
V is padded the same way, producing a `padded_D`-wide O whose last
`padded_D - true_D` columns are always 0; we slice them off.

`softmax_scale = 1 / sqrt(true_D)` is passed via the override so softmax math
matches the unpadded reference exactly.

**Cost:** roughly `padded_D / true_D` extra flops and smem traffic for the
matmuls (e.g. 2× for D=32, 4× for D=16, 1.33× for D=96). The pair-bias /
res_mask loads are unaffected. A proper D=96 specialization (with
`st_bf<*, 96>` and 64-byte swizzle) is feasible but not yet implemented.

## 5. `TKEvoAttention` (`tk_evo_attention_layer.py`)

`torch.autograd.Function` with the same calling convention as
`MegaFold.TritonEvoformer`:

```python
from tk_evo_attention_layer import TKEvoAttention
O = TKEvoAttention.apply(Q, K, V, res_mask, pair_bias)
# Q, K, V  : (B, N_SEQ, N_CTX, H, D)            any float dtype
# res_mask : (B, N_SEQ, 1, 1, N_CTX)            cast to bf16 internally
# pair_bias: (B, 1, H, N_CTX, N_CTX)            cast to bf16 internally
# O        : (B, N_SEQ, N_CTX, H, D)            bf16
```

The layer encapsulates:
1. dtype casts to bf16,
2. `(B, N_SEQ, N_CTX, H, D) ↔ (B*N_SEQ, H, N_CTX, D)` transpose + view,
3. pair_bias `squeeze(1)`, res_mask `view`,
4. D padding (§4) and the matching `softmax_scale` override,
5. slicing O back to `true_D` and transposing back to the caller layout,
6. `ctx.save_for_backward(...)` of the padded tensors + metadata (for the
   future backward pass).

`backward` raises `NotImplementedError`.

## 6. Performance tuning log

The initial correctness-first kernel (`CW=2`, `stages=2`, per-iter
`scale → log2(e)` split, pair_bias/mask scaled in the kernel on the
register tile) was 20–30% slower than Triton at `N_CTX ≥ 512` on H100:

```
initial TK (CW=2, stages=2):
  N_CTX    triton (TFLOP/s)    tk (TFLOP/s)    TK / Triton
    128                1.85            2.28       1.23
    256                7.45            8.67       1.16
    384               16.82           16.06       0.96
    512               26.92           24.21       0.90
    640               41.99           30.31       0.72
    768               46.25           37.70       0.82
   1024               51.65           46.71       0.90
```

This section records every change we tried, in order, with the measured
impact and whether it was kept. Final numbers are in §7.

### 6.1 What worked

#### (a) Deeper pipeline: `stages` 2 → 3 for `D=64`  ✅

**Hypothesis.** At large `N_CTX` the producer is steady-state issuing
`(k, v, pb×CW, rm)` per kv step, a ~4× data-movement increase over stock MHA.
Two stages means at most one tile of each kind is prefetched ahead of the
consumer; at high `N_CTX` the consumer was observed to be stalling on TMA
arrivals.

**Change.** For `D=64` bumped `evo_fwd_tile_dims<64>::stages` from 2 to 3.
Shared-memory check: adds one more of {K tile, V tile, `CW` pair_bias tiles,
res_mask sv} per extra stage → roughly +64 KB, bringing total to ~213 KB
which still fits under the `MAX_SHARED_MEMORY - 1024 ≈ 226 KB` budget on H100.

For `D=128` (`CW=1`, K/V tiles twice as wide) stages=3 does *not* fit, so we
kept stages=2 there. Given CW=1 the extra parallelism from more stages is
less important anyway (one warpgroup per CTA).

**Impact.** On its own about +2–10% at `N_CTX ≥ 512` — modest because register
pressure was still the limiter.

#### (b) Hoist independent work above `mma_async_wait()`  ✅

**Hypothesis.** `warpgroup::mm_ABt(att, q, k)` returns immediately — WGMMA
runs async. Any work on registers *other than* `att_block` can execute in
parallel with the WGMMA.

**Change.** Moved `warp::copy(max_vec_last, max_vec)` (used later for the
online-softmax correction) to sit between `mm_ABt` and `mma_async_wait`,
matching the pattern the stock MHA kernel uses.

**Impact.** Small on its own but combined with the `stages=3` change this
was the config that went from "21–28% slower than Triton" to roughly parity
at mid-range `N_CTX`.

#### (c) Fuse `softmax_scale` and `log2(e)` into a single `att_block` multiply  ✅

**Hypothesis.** The initial kernel did
```
att *= softmax_scale      // pass 1 over rt_fl<16,128>
att += pair_bias          // pass 2
att += res_mask (col)     // pass 3
att *= log2(e)            // pass 4
```
Four passes over the 64-register-per-lane accumulator is a lot of fp32 work
between the matmul and the softmax.

**Change.** Pre-compute `softmax_scale_log2 = g.scale * LOG2E` once. Apply
it in a single multiply on `att_block`. Then scale `pair_bias` and
`res_mask` by `log2(e)` in registers *after the shared-to-register load* so
when we add them the result is still in base-2 log space:
```cpp
warp::mul(att_block, att_block, softmax_scale_log2);  // fused
warpgroup::load(pb_reg, pb_smem[...]);
warp::mul(pb_reg, pb_reg, LOG2E);   // register-resident, ~free
warp::add(att_block, att_block, pb_reg);
// same pattern for rm_reg via warp::add_col
```

**Impact.** Removes one full pass over `att_block` (the base-2 conversion).
The two new register multiplies on `pb_reg` (rt_fl<16,128>) and `rm_reg`
(rv_fl<128>) are much cheaper than the `att_block` pass they replace, and
they happen before the add so the critical path doesn't grow. Worth
~3–5% in the hot loop.

#### (d) Keep `pair_bias` in `bf16` in the caller, not `fp32`  ✅ (benchmark)

**Hypothesis.** The layer always does `pair_bias.to(bf16)` on the way in.
If the caller hands us fp32, that materialises a whole new tensor per
forward — for `B=4, H=16, N=1024` that's a 268 MB → 134 MB copy+cast on
every call, which showed up clearly in the forward-only benchmark.

**Change.** In the benchmark, allocate `pair_bias` as `bf16` directly so the
layer's `.to(bf16)` is a no-op. The Triton reference takes either dtype,
so it's not penalized. In production the caller should hold pair_bias in
bf16 between successive attention calls.

**Impact.** This single change moved TK from "parity or slower than Triton
at most sizes" to "TK faster than Triton at every size except 640". At
`N_CTX=1024` the difference was 46 TFLOP/s → 71 TFLOP/s.

### 6.2 What *didn't* work

#### (e) Pre-scaling `pair_bias` and `res_mask` by `log2(e)` in Python  ❌

**Hypothesis.** The kernel's `mul(pb_reg, LOG2E)` / `mul(rm_reg, LOG2E)` is
pure register work but still two explicit instructions per iteration per
warp. Pre-multiplying the `pair_bias` and `res_mask` tensors by `log2(e)`
once on the Python side would eliminate them.

**Change.** In `TKEvoAttention.forward`, replaced
`pair_bias.squeeze(1).to(bf16)` with
`(pair_bias.squeeze(1) * LOG2E).to(bf16)` and similarly for `res_mask`.
Deleted the matching multiplies in the kernel.

**Result.** Catastrophic regression — `N_CTX=1024` went from ~46 TFLOP/s
down to 31 TFLOP/s; every size got substantially slower.

**Why.** `pair_bias * LOG2E` is a full elementwise-kernel launch over a
256 MB tensor (for `B=4, H=16, N=1024`). This is *inside* the `forward()`
call and hence inside `triton.testing.do_bench`, so every repetition paid
the cost. The in-kernel register mul it saved is essentially free by
comparison — the mul happens on values that are already in registers, with
no additional global-memory round-trip. Reverted.

**Takeaway.** "Pre-compute things in Python to simplify the kernel" is
attractive but has a failure mode: if the precomputation materialises a new
tensor on a hot path, it's strictly worse than doing the work inside the
kernel. Only move to Python for values that are cached across calls.

#### (f) Hoist `pb_reg` load + `mul` above `mma_async_wait()`  ❌

**Hypothesis.** Extension of (b): load pair_bias into registers and
pre-scale it while the WGMMA is in flight. The loaded `pb_reg` and the
WGMMA accumulator `att_block` live in different registers, so there's no
data hazard.

**Change.** Moved
```cpp
wait(pb_smem_arrived[s], phase);
warpgroup::load(pb_reg, pb_smem[...]);
warp::mul(pb_reg, pb_reg, LOG2E);
```
to sit *before* `mma_async_wait()`.

**Result.** `ptxas` reported register spills grew from 8/8 bytes to
72/72 bytes for the `D=64` kernel and 392/496 bytes for `D=128`. Benchmark
throughput dropped slightly despite the extra overlap. Reverted.

**Why.** The consumer warpgroup is already at the register redistribution
sweet-spot (`setmaxnreg inc 160`). Holding `pb_reg` (a full rt_fl<16,128>,
64 regs/lane) live for the duration of the matmul in addition to everything
else that was already live blew the register budget. The spills cost more
than the overlap saved. The only thing that fit was `warp::copy(max_vec_last)`,
which uses a small col_vec (1 reg/lane) — which is what stayed.

**Takeaway.** Async overlap only pays when the new live range fits in the
existing register budget. Large register tiles (rt_fl<16,kv_height>) are
poor candidates; small col_vecs are fine.

#### (g) Fuse `CW` per-warpgroup pair_bias TMAs into one wider TMA  ❌

**Hypothesis.** The producer issues `CW` separate TMAs per stage for
pair_bias (one 64×128 tile per consumer warpgroup). Replacing them with a
single (CW×64)×128 TMA would halve the TMA-issue count on the pair_bias
path — plausibly helpful at shapes where the producer is TMA-bound.

**Change.** Made `pb_tile = st_bf<CW*qo_height, kv_height>`, allocated one
per stage (not `CW` per stage), and had each consumer warpgroup pull its
own qo_height-row subtile via
`pb_smem[s].template subtile<qo_height, kv_height>({warpgroupid, 0})`.
Producer coord became `{batch, head, blockIdx.x, kv_idx}` instead of
`{batch, head, seq_idx + wg, kv_idx}`.

**Result.** No change at `N_CTX=640` (the specific case that motivated it),
slight regression (~1–4%) at `N_CTX ∈ {384, 512}`. Correctness still passed
and spills actually went down slightly. Reverted as net-neutral-to-negative.

**Why.** The producer was not TMA-issue-bound at the shapes we care about;
it was keeping up. The benefit of "fewer TMA issues" is only real when the
producer stalls on issue-rate, which it doesn't here. And the subtile access
on the consumer added a small amount of address arithmetic that marginally
hurt the hot loop. Keeping the simpler per-warpgroup tiles.

**Takeaway.** Consolidating TMAs only helps when issue rate is the
bottleneck. For kernels at our arithmetic intensity with warp-specialized
producers, the producer usually has headroom. Profile before assuming.

#### (h) Would `CW=3` help? — not for most shapes  ⚠️ (not tried)

`consumer_warpgroups=3` matches the stock MHA kernel and would amortize
K/V/rm loads across 3 Q tiles per CTA. But it requires
`N_CTX % (3 * qo_height) == N_CTX % 192 == 0`, which excludes most of the
benchmark schedule (`512, 640, 1024` all fail). Also, `CW=3 * stages=3`
overflows smem. Not adopted.

#### (i) Would halving `kv_height` to 64 help `N_CTX=640`? — probably not  ⚠️ (not tried)

At `N_CTX=640` with `kv_height=128` there are 5 kv blocks; with stages=3
that's 3 steady-state iterations out of 5 (60% pipeline-fill fraction),
which is why 640 is the single remaining shape where TK is slower than
Triton. Halving `kv_height` to 64 would give 10 kv blocks, improving the
fill fraction, but it would also halve the matmul size per iteration
(cutting WGMMA efficiency) and double TMA-issue overhead. Unlikely to be
net positive — skipped. Would be revisited if `N_CTX=640` becomes a common
shape in practice.

## 7. Final benchmark

After optimisations (a)–(d). Forward-only, `BATCH=4, H=16, D=64, N_SEQ=1`,
bf16, H100, averaged over 2 consecutive `do_bench` runs (rep=5000ms,
warmup=200ms):

```
  N_CTX    triton (TFLOP/s)    tk (TFLOP/s)   TK / Triton
    128                1.78            2.36        1.33×
    256                7.53            9.25        1.23×
    384               17.05           18.33        1.07×
    512               29.09           29.55        1.02×
    640               41.83           39.25        0.94×
    768               46.21           51.72        1.12×
   1024               51.60           71.19        1.38×
```

TK is faster than Triton at every benchmark shape except `N_CTX=640` where
it is ~6% slower. The large wins at 128/256/1024 are mainly from the
3-stage pipeline plus the register-pressure discipline described above.
The N_CTX=640 gap is the pipeline-fill-fraction issue noted in (h).

## 8. Constraints

- `D ∈ {16, 32, 64, 96, 128}` (via padding; kernel natively `{64, 128}`).
- `N_CTX` divisible by 128 (kv_height).
- For native D=64: `N_CTX` divisible by `2 * qo_height = 128`.
- For native D=128: `N_CTX` divisible by `1 * qo_height = 64`; combined with
  the kv_height rule this still requires `N_CTX % 128 == 0`.
- GPU: H100 (SM 90). Set `GPU := H100` in the Makefile.

## 9. Build & run

```
cd kernels/attention/evo_h100
make                                # builds _C.cpython-*.so
python3 test_tk_evoattention.py     # correctness vs. Triton reference
python3 benchmark.py                # speed vs. Triton
```
