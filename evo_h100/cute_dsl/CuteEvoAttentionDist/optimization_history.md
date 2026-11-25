# Distributed CuTe EvoAttention optimization history

## 1. Scope and status

This document records both halves of the D=32 sequence-parallel prototype:
the remote-TMA forward epilogue and the inverse-reshard/backward path. The
production files beside this document contain only the retained design.
Rejected variants and generated sweep artifacts were removed during the
production refactor.

The benchmark boundary is intentionally narrower than a complete MegaFold
block. It compares:

- baseline forward: model-layout preparation, Triton EvoAttention, then one
  C=128 NCCL row-to-column All-to-All. The Triton baseline is MegaFold's
  `TritonEvoformer` autograd op (`debug/evoattention.py`: 64x64 WGMMA tiles,
  strided Q/K/V views); `EVOATTENTION_REFERENCE` overrides its path;
- CuTe forward: EvoAttention plus remote-TMA resharding of both attention
  output and residual input;
- baseline backward: one C=128 NCCL column-to-row All-to-All, then the full
  Triton EvoAttention backward;
- CuTe backward: peer-load inverse resharding of dO and dX fused with delta
  preprocessing and workspace clears, then the physical CuTe backward.

Layer normalization, gating, output projection, dropout, SwiGLU, residual
addition, and pair-bias collectives are outside this microbenchmark. The
current implementation supports P in {2, 4, 8}, D=32, H in {4, 16}, and
N divisible by both 128 and P. B, H, S=N/P, and N are runtime dynamic.

The 2.5x combined target was reached on most sampled configurations, but not
on every point of the full matrix. In particular, repeated P=8, B=1, H=4,
N=640 measurements were approximately 2.44--2.47x. This is therefore a
correct working prototype, not a claim that the complete acceptance target
has been closed.

## 2. Tensor contracts and sharding

Each rank owns a row shard before attention:

```text
q, k, v       [B, S, N, H, 32]  BF16
pair_bias     [B, 1, H, N, N]   BF16, replicated and already gathered
residual_mask [B, S, 1, 1, N]   BF16, values 0 or 1
X_row         [B, S, N, 128]    BF16
S = N / P
```

The fused forward produces symmetric NVSHMEM allocations viewed locally as:

```text
attention_col [B, N, S, H, 32]
X_col         [B, N, S, 128]
```

For destination rank `r`, every source rank sends columns
`[r*S:(r+1)*S]`. Concatenating the source-row shards along their first
sequence axis constructs the destination column shard. The backward applies
the inverse map to `dAttention_col` and `dX_col`; `dX_row` is an explicit
output because this fused operator accepts `X_row` as an input.

dPairBias is the local attention contribution for a replicated pair bias.
A model integration must retain the existing autograd collective that sums
those per-rank contributions. That collective is deliberately outside this
prototype's timed boundary.

## 3. Retained forward design

### 3.1 Attention mainloop

The forward uses the following D=32 schedule:

| Quantity | Value |
|---|---:|
| query tile per consumer | 64 rows |
| key/value tile | 64 rows |
| consumer warpgroups | 2 |
| producer warps | 1 |
| epilogue warps | 1 |
| threads per CTA | 320 |
| K/V stages | 3 |
| pair-bias stages | 2 |
| Q / output staging buffers | 2 / 2 |
| CTAs per SM (persistent grid) | 2 |
| output transfer granularity | 16 rows |

One producer warp feeds two 128-thread WGMMA consumers. The consumers work on
two adjacent sequence instances while sharing the same pair-bias tile. Query,
key, value, mask, online-softmax state, and output accumulators remain private
where their logical indices differ.

The mainloop reads the pair-bias tile with non-transposed `ldmatrix` directly
into the score-accumulator layout, and combines row maxima with `max.f32`
(`cute.arch.fmax`) instead of the NaN-propagating `cutlass.max`, which lowers
to a compare-and-select pair. Together these remove about 21% of the executed
instructions.

### 3.1.1 Persistent schedule

The grid launches two CTAs per SM, and each CTA loops over the linear task
space with a grid stride (`task = CTA + i * grid`), so the tasks resident at
any moment are contiguous. Within a task index, heads vary fastest after the
query block: Q/K/V store all heads of a token contiguously, and this order lets
concurrently resident tasks share those rows in L2. A head-slowest order
roughly doubled DRAM reads at H=16.

Pipeline barriers are initialized once per CTA. K/V and pair-bias phases follow
a CTA-wide pipeline block index, and Q and output phases follow the per-CTA
task iteration, so the producer stages the next task's Q while the consumers
finish the current task. A dedicated epilogue warp waits on the output staging
buffer, issues the local and remote TMA stores, and releases the buffer through
an mbarrier, so neither consumer warpgroup ever waits on TMA stores. The
residual-input chunk `c` of a task is staged by the producer into its own tile
and stored by the epilogue warp of the task whose head index is `c % H`, which
spreads that traffic across heads.

At P=1, compared with the previous non-persistent kernel, this schedule is
1.45x faster at B=1, H=4, N=384, 1.43x at H=16, N=384, and 1.26x at H=4,
N=640, with bit-identical output, LSE, peer output and residual copy.

### 3.2 Remote-TMA epilogue

The epilogue warp divides the output tile into 16-row destination-aligned
chunks. For each
chunk, the rank owning that column range is known from the global query-row
coordinate. TMA stores therefore target the appropriate symmetric peer view
directly instead of first materializing row-sharded output and launching a
second All-to-All kernel.

The same epilogue routes the C=128 residual input. This is important for the
eventual `EvoAttentionDist(X, q, k, v, pair_bias, mask) -> (O_col, X_col)`
boundary: one synchronization epoch covers both products.

After issuing the remote stores, each source publishes its epoch through a
system-scope release. The destination waits until it observes every source's
epoch with acquire semantics before returning. The release orders the remote
writes before the completion publication; the matching acquire prevents
subsequent local consumers from passing the observed publication.

## 4. Retained backward design

### 4.1 Fused remote preprocessing

The retained preprocessor reads the relevant peer slices of
`dAttention_col` and `dX_col` directly. In one launch it:

1. reconstructs row-sharded dO;
2. reconstructs row-sharded dX;
3. computes `delta = rowsum(dO * O)` in FP32;
4. clears the FP32 dQ accumulation workspace; and
5. clears the complete replicated FP32 dPairBias accumulation buffer.

This replaces a standalone inverse collective, a backward preprocessing
kernel, and separate clears. The dX result is not added to dQ: it is the
gradient of the independent residual input returned by the distributed API.

### 4.2 Physical backward

The physical kernel uses the tuned D=32 single-GPU five-MMA backward schedule
over the rectangular local problem S x N. It computes dQ, dK, dV, and the
local dPairBias contribution. The retained schedule fixes Q/dO pipeline depth
at three; stage-count environment switches used during exploration were
removed.

For H=16 and N>=640, the retained split-sequence pair-bias reduction avoids
the main-kernel dPairBias atomic bottleneck. The experimental override that
forced this path for H=4 was removed because it regressed performance.

### 4.3 Rectangular dQ/dPairBias finalization fix

The inherited square-problem finalizer assumed S=N and only converted S*N
dPairBias values. Under sequence parallelism, dPairBias remains N*N, so this
silently left most of the output unconverted. The retained rectangular
finalizer assigns each existing dQ CTA an additional contiguous fraction of
the N*N conversion. Across S/64 dQ CTAs, each CTA converts P times its former
share, covering the full replicated matrix without a standalone launch.

After the fix, sampled dPairBias agreement with the Triton reference had a
maximum absolute error of 0.03125 under the benchmark tolerance.

## 5. Experiments retained or rejected

### 5.1 Direct peer-load inverse reshard — retained

Directly loading the symmetric peer views in the preprocessing kernel was the
best measured backward transport. At P=8, N=640, attention-gradient remote
preprocessing was about 0.094 ms in the representative comparison and was
substantially faster than the push-based alternative.

### 5.2 Source-push inverse reshard — rejected

A production-shaped push protocol was correct at P=2 and P=8, but its
additional remote stores and completion protocol cost approximately 0.291 ms
at P=8, N=640. Combined speedup fell to 1.862x there, versus 2.454x at the
small P=2, N=128 point. The peer-load preprocessor was restored.

### 5.3 External registration of PyTorch tensors — rejected

Registering ordinary PyTorch allocations as peer-addressable buffers failed
because the allocator did not supply the virtual-memory-management backing
required by this NVSHMEM path. The prototype consequently uses symmetric
`nvshmem.tensor` allocations at its communication boundary.

### 5.4 Separate dX stream — rejected

Splitting residual-gradient transport from attention preprocessing produced
component times near 0.0614 ms and 0.0242 ms, but concurrent execution plus
events increased the combined step from roughly 0.4506 ms to 0.4604 ms. The
extra synchronization and fabric contention outweighed overlap.

### 5.5 Grouped peer tiles — rejected

Grouping multiple peer tiles into one loop increased the attention-only
preprocessor from about 0.0614 ms to 0.0855 ms and the combined step to
roughly 0.4924 ms. The simple one-tile mapping was restored.

### 5.6 Pipeline-depth sweep — neutral

Q/dO stage counts 2, 3, and 4 did not produce a decisive improvement across
the distributed matrix. Stage 3 remains the fixed production choice because
it matches the validated single-GPU schedule and avoids configuration state.

### 5.7 Forced split dPairBias at H=4 — rejected

At N=640, forcing the split reduction increased the representative physical
backward from approximately 0.357 ms to 0.4192 ms. The branch is therefore
limited to H=16, N>=640, where atomics are the measured problem.

## 6. Representative combined results

### 6.1 Persistent forward vs `TritonEvoformer` + NCCL

`benchmark.py` with the persistent forward, H100, one process per GPU;
combined forward + backward step, slowest rank:

| B, H | N | P=2 | P=4 | P=8 |
|---|---:|---:|---:|---:|
| 1, 4 | 128 | 5.15x | 5.70x | 7.16x |
| 1, 4 | 256 | 3.20x | 5.40x | 6.58x |
| 1, 4 | 384 | 1.57x | 2.43x | 4.33x |
| 1, 4 | 512 | 1.46x | 1.53x | 2.78x |
| 1, 4 | 640 | 1.38x | 1.43x | 1.56x |
| 1, 4 | 768 | 1.31x | 1.36x | 1.42x |
| 1, 4 | 896 | 1.24x | 1.30x | 1.35x |
| 1, 4 | 1024 | 1.28x | 1.27x | 1.29x |
| 1, 16 | 384 | 1.25x | 1.33x | 1.47x |
| 1, 16 | 1024 | 2.10x | 2.06x | 1.94x |
| 4, 16 | 1024 | 1.99x | 2.07x | 1.92x |

Below roughly 0.2 ms per rank the baseline is bounded by a fixed cost of about
0.14 ms for the Triton forward and 0.2 ms for its backward, plus the NCCL
exchanges, so the largest ratios mostly reflect launch and communication
overhead. At H=16, N=1024 the Triton backward alone is 2.4x slower than the
CuTe physical backward.

### 6.2 Earlier development results

These are representative development measurements, not a claim of a fresh
post-refactor rerun. They predate the persistent forward and were measured
against the original upstream MegaFold Triton kernel
(`megafold/model/FusedEvoAttention/evoattention.py`, 16-row query tiles), not
the current `TritonEvoformer` baseline. CUDA events measured the complete dependency chain and
the reported sample was reduced with `MAX` across ranks.

| P | B | H | N | speedup vs Triton + NCCL |
|---:|---:|---:|---:|---:|
| 2 | 1 | 4 | 128 | 2.729x |
| 2 | 1 | 4 | 384 | 2.658x |
| 2 | 1 | 4 | 640 | 2.695x |
| 2 | 1 | 4 | 1024 | 2.586x |
| 4 | 1 | 4 | 128 | 3.012x |
| 4 | 1 | 4 | 384 | 2.560x |
| 4 | 1 | 4 | 640 | 2.582x |
| 4 | 1 | 4 | 1024 | 2.591x |
| 8 | 1 | 4 | 128 | 3.843x |
| 8 | 1 | 4 | 384 | 2.785x |
| 8 | 1 | 4 | 640 | 2.44--2.47x |
| 8 | 1 | 4 | 1024 | 2.582x |
| 2 | 4 | 16 | 1024 | 2.972x |

The development sweep also recorded P=2 speedups of 2.698x, 2.711x, 2.658x,
and 2.583x at N=256, 512, 768, and 896; P=4 speedups of 3.001x, 2.568x,
2.608x, and 2.546x; and P=8 speedups of 3.797x, 2.501x, 2.515x, and 2.515x.

## 7. NCU diagnosis at P=8, N=640

The retained physical backward showed approximately:

| Metric | Value |
|---|---:|
| L2 throughput | 70% |
| SM throughput | 33.64% |
| memory throughput | 69.36% |
| DRAM throughput | 11.41% |
| L2 hit rate | 93.26% |
| registers per thread | 168 |
| shared memory per CTA | 110,592 bytes |
| achieved occupancy | 15.23% |
| long-scoreboard share | 29.2% |
| barrier-stall share | 15.3% |

This identifies the remaining issue as L2/latency exposure at one CTA per SM,
not DRAM bandwidth saturation. Adding stages alone did not help because the
kernel is constrained by shared memory and register residency. Future work
should reduce repeated L2 transactions or restructure the reduction before
adding more buffering.

## 8. Reproduction

Run from this directory with one process per GPU:

```bash
torchrun --standalone --nproc-per-node=2 benchmark.py --n 384
torchrun --standalone --nproc-per-node=4 benchmark.py --n 384
torchrun --standalone --nproc-per-node=8 benchmark.py --n 384
```

Correctness runs before timing unless `--skip-check` is supplied. CUDA events
exclude compilation, allocation, and warmup, and the timing result is the
slowest rank for each repeat. Use `--breakdown` to report the forward,
communication, remote-preprocess, and physical-backward components.
