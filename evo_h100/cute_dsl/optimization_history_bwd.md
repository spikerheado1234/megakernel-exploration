# EvoAttention CuTe backward optimization history

## 1. Executive summary

The production backward in `evoattention_cute_bwd.py` uses one fixed M64 x
N128 Hopper main schedule.  One TVM-FFI dispatch emits FP32 dPairBias
initialization, FlashAttention backward preprocessing, the fused five-MMA main
kernel, dQ postprocessing, and FP32-to-BF16 dPairBias conversion.  For H=16
and N>=640, the same dispatch disables main-kernel dPairBias atomics and adds a
Q64 x K128 x S16 split-sequence CuTe reduction.  B, S, H, and N are runtime
dynamic; D=64 is static.

The final production kernel passed sampled correctness checks and exceeded 2x
the complete authoritative Triton backward on all 32 requested shapes.  The
definitive CUDA-event sweep uses the rotating public-layout benchmark: input
preparation and saved-forward generation are outside timing; all backward
launches, final dPairBias conversion, and the three Triton-only gradient
transpose-copy launches are inside timing.  The minimum speedup was 2.249x at
B4/H16/N640. The complete acceptance table appears at the end of this log.

Representative final-source results:

| B | H | N | Triton ms | CuTe ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 128 | 0.1887 | 0.0744 | 2.534x |
| 1 | 4 | 1024 | 42.4384 | 16.6270 | 2.552x |
| 1 | 16 | 640 | 48.9424 | 21.3854 | 2.289x |
| 1 | 16 | 1024 | 210.1223 | 89.6514 | 2.344x |
| 4 | 4 | 640 | 42.5450 | 15.6176 | 2.724x |
| 4 | 4 | 1024 | 164.7169 | 66.0500 | 2.494x |
| 4 | 16 | 640 | 194.6316 | 86.5380 | 2.249x |
| 4 | 16 | 1024 | 841.0437 | 358.2977 | 2.347x |

For correctness, every target shape used identical logical inputs and
Triton-forward O/LSE, then sampled 65,536 evenly spaced values from every
output.  All dQ/dK/dV/dPairBias checks passed `atol=1e-2, rtol=2e-2` with zero
mismatches.  The largest absolute errors over the 32-shape sweep were 0.001953
for dQ, 0.003906 for dK, 0.001953 for dV, and 0.031250 for dPairBias.

There are three runtime paths in one compiled artifact: N=128 uses a
non-reducing dQ bulk store, the general fused-atomic path handles N>=256 plus
every H=4 shape, and the split-dPairBias path handles H=16/N>=640.  A
single-process N=128 -> 256 -> 640 -> 1024 test keeps compile count at one and
passes all gradient checks.  The runtime dQ epilogue is a local subclass of
the upstream SM90 mainloop, avoiding the earlier process-global helper patch.

The fully validated pre-cleanup source was SHA-256
`b633ec8356583854024bceaa1c92ed0910e46dce8b7ba15cf38de1288b9b4e8c`.
The behavior-preserving cleanup removed 88 lines of unreachable upstream
variable-length, local, deterministic, and block-sparse dQ epilogue handling;
the retained source is SHA-256
`a1051737567070821d320a15cee1cc1f00b4ad7e3adb9c8f2e9a51560216bc5d`.
After cleanup, the mixed-path runtime suite still passed with compile count
one, and focused CUDA-event checks measured 2.607x at B1/H4/N128 and 2.308x
at B1/H16/N640.

## 2. Operation and reference baseline

For one `(batch, sequence, head)` matrix, with `S=N`:

```text
delta_i = sum_d O[i,d] * dO[i,d]
P       = exp(Q K^T / sqrt(64) + pair_bias + residual_mask - LSE)
dP      = dO V^T
dS      = P * (dP - delta)
dQ      = dS K / sqrt(64)
dK      = dS^T Q / sqrt(64)
dV      = P^T dO
dPB     = sum_sequence dS
```

The authoritative Triton implementation launches preprocess, dK/dV, and
dQ/dPairBias kernels.  It reconstructs QK, P, dP, and dS twice, performing
seven GEMMs or `14 * B*S*H*N^2*D` FLOPs.  Representative original kernel-only
times for B1/H4 were 0.140 ms, 2.326 ms, and 38.348 ms at N=128, 384, and
1024.  The public-layout benchmark is intentionally larger because it also
includes final BF16 dPairBias conversion and the three output transpose-copy
launches required by Triton's internal `[B,S,H,N,D]` layout.

The authoritative baseline is the unmodified
`~/MegaFold/megafold/model/FusedEvoAttention/evoattention.py` implementation.

## 3. Roofline: optimize data motion before tensor math

The initial roofline and NCU study rejected the premise that the reference is
tensor-core-bound.  At B1/H4/N1024, Triton's dK/dV and dQ kernels reached
87.4% and 88.4% of the L1TEX data-pipe limit, while tensor-core activity was
only 16.6% and 13.6%.  HBM read utilization was only 3.2% and 5.3%, with
93-95% aggregate L2 hit rate.  The practical bottleneck was repeated
L2/L1/shared movement, shared conflicts, and insufficient eligible warps.

The parameterized traffic model predicted 44.75 logical L2 bytes per score
for Triton's 16-row ownership tiles, 20.19 bytes for 64-row ownership, and
14.19 bytes for a later two-way multicast/reduction design.  Semantic
SMEM-to-register operand traffic similarly falls from about 70 to 28 bytes per
score when ownership grows from 16 to 64 rows.

That evidence fixed the non-negotiable design choices:

- M=64 query tiles, avoiding padding and extending the inner pipeline loop;
- N=128 key ownership in the fused kernel;
- one TMA producer warpgroup and two WGMMA consumer warpgroups;
- 128-byte-swizzled shared layouts;
- physical storage only once for broadcast pair bias and residual mask;
- one score/dS reconstruction shared by all five gradient GEMMs.

At B1/H4/N1024, the post-M64 ideal lower bounds were 1.32 ms for compulsory
DRAM traffic, 7.39 ms for L2-to-shared movement, 3.89 ms for
shared-to-register movement, and 3.89 ms for tensor work. This reinforced the
measured L1TEX/shared-memory diagnosis.

## 4. First CuTe implementation: correct but only about 1.3x

The first self-contained CuTe implementation used two separate M64 attention
sweeps: one q-owned sweep for dQ/dPairBias and one key-owned sweep for dK/dV.
It was correct at N=128/256/384 but achieved only about 1.33-1.40x.  At N=128,
its two main kernels took about 53.3 us and 45.3 us, plus initialization and
conversion.

NCU explained the miss:

- the two-pass design duplicated QK, dP, exponentials, and score data motion;
- at N=384 it moved 6.85 GB of L2 sectors versus 3.93 GB for a fused pass;
- dPairBias atomics were lane-scattered, producing 28.3 million excessive
  global sectors;
- the two kernels used 132-149 KiB shared memory and only 6.6-13.5% achieved
  occupancy.

This result established that a cleaner split-pass implementation could not
reach 2x.

## 5. Naive full fusion: less work, worse schedule

A second prototype fused all five GEMMs into one key-owned CTA but initially
used one consumer warpgroup and 64-key ownership.  It was mathematically
correct but regressed to approximately Triton speed: B1/H4 timings were about
0.129, 0.748, 2.286, and 5.05 ms for N=128/256/384/512.  NCU measured 206
registers per thread, 165.4 KiB shared memory, 7.27% achieved occupancy, and
only 7.35% tensor activity.

The failure was not fusion itself; it was fusion without the FlashAttention-3
warp-specialized mapping.  One warpgroup could not hide score/softmax work
behind tensor work, and its oversized live state removed latency-hiding
capacity.

## 6. Winning fused topology

The best fixed core configuration is:

```python
BwdConfig(
    m_block_size=64,
    n_block_size=128,
    num_stages_Q=2,
    num_stages_dO=2,
    num_stages_PdS=2,
    SdP_swapAB=True,
    dKV_swapAB=False,
    dQ_swapAB=False,
    AtomLayoutMSdP=1,
    AtomLayoutNdKV=2,
    AtomLayoutMdQ=1,
    num_wg=2,
    dQ_single_wg=True,
)
```

This is a 384-thread CTA: one 128-thread producer warpgroup and two consumer
warpgroups.  One consumer computes the full dQ GEMM, while both cooperate on
dK/dV.  The main loop performs QK, dO*V, P*dO, dS*K, and dS*Q from one P/dS
reconstruction.  It uses 168 registers/thread and 117,760 bytes shared memory.

EvoAttention is injected through score modifiers rather than materializing a
broadcast tensor:

```python
@cute.jit
def score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, _, aux):
    pair_bias, residual_mask, _ = aux
    flat_batch = batch_idx[0]
    sequence_count = residual_mask.shape[1]
    batch = flat_batch // sequence_count
    sequence = flat_batch - batch * sequence_count
    return scores + scalar_to_ssa(
        pair_bias[batch, head_idx[0], q_idx[0], kv_idx[0]]
        + residual_mask[batch, sequence, kv_idx[0]],
        Float32,
    )
```

The backward score modifier atomically accumulates FP32 dPairBias and returns
dS unchanged.  This preserves the authoritative `-1e9` residual mask and
avoids any physical broadcast replication in shared memory.

## 7. N=128 direct dQ store

At N=128 there is exactly one N128 key tile, hence exactly one writer for each
dQ element.  Replacing FP32 bulk reduction with a plain bulk store avoids
clearing the dQ accumulator:

```python
cp.async.bulk.global.shared::cta.bulk_group [dst], [src], bytes;
```

In the true cold benchmark the general reduction path took 0.0796-0.0814 ms
at B1/H4/N128; direct store reduced this to 0.0735-0.0760 ms.  The specialization
is correct because ownership, not an assumed launch order, proves there is one
writer.  It is also required by the per-shape target: at B4/H16/N128 the
general path measured 1.2393 ms versus Triton's 2.4557 ms, only 1.982x, while
the direct path measured about 0.978 ms and exceeded 2.5x.

The final implementation does not compile N=128 separately.  A local subclass
of the upstream SM90 mainloop overrides only its dQ epilogue and lowers the
choice against the runtime sequence length:

```python
with cute.arch.elect_one():
    if sequence_info.seqlen_k == KEY_TILE_SIZE:
        direct_bulk_store_f32(shared_dq, global_dq, store_bytes)
    else:
        copy_utils.cpasync_reduce_bulk_add_f32(
            shared_dq, global_dq, store_bytes
        )
```

The enclosing host JIT similarly dispatches on dynamic tensor extents:

```python
if pair_bias.shape[2] == 128:
    direct_pipeline(...)
elif pair_bias.shape[1] == 16 and pair_bias.shape[2] >= 640:
    split_pair_bias_pipeline(...)
else:
    fused_atomic_pipeline(...)
```

CuTe lowers both conditions to runtime control flow, so all three paths live
in one compiled TVM-FFI artifact and `compile_count` remains one.

An alternative one-BF16x2-atomic-per-K64-partial scheme was rejected.  It had
rare tolerance failures at large shapes and changed billions of values across
identical launches.  Combining two K64 partials in FP32 before a K128 BF16x2
atomic passed the sampled campaign, but FP32 accumulation plus final BF16
conversion remains the robust general path.

## 8. The rotating-address regression was a host launch gap

The initial fused candidate looked fast with one address set but failed the
production rotating benchmark:

| B | H | N | Triton ms | high-level CuTe wrapper ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 128 | 0.1866 | 0.1444 | 1.293x |
| 1 | 16 | 128 | 0.6527 | 0.2686 | 2.430x |

The GPU kernels themselves still summed to about 71-73 us.  Torch profiler
showed that the high-level helper performed three `torch.empty` allocations,
multiple slices/views/detaches, and three independent compiled calls on every
invocation.  Its CPU enqueue time was about 100-115 us per call, so B1/H4's
GPU work drained before the next kernel arrived.  CUDA events correctly
included that idle interval, yielding 144 us despite only about 72 us of
kernel execution.

The fix was to preallocate explicit workspaces per rotating slot and compile
the complete launch sequence as one TVM-FFI host callable:

```python
@cute.jit
def __call__(..., stream):
    zero_dpb(...).launch(stream=stream)
    preprocess(..., stream)
    fused_main(..., stream)
    postprocess_dq(..., stream)
    convert_dpb(...).launch(stream=stream)
```

This still emits five GPU kernels, but Python makes one FFI call and performs
no steady-state allocation.  B1/H4/N128 fell from 0.1444 to 0.0760 ms, or
2.457x over the complete Triton path.  Moving dPB zero/conversion into the same
outer call did not change GPU work materially, but it removed the last two
Python CUDA launches and made dispatch predictable.

## 9. Runtime-dynamic shapes unexpectedly removed spills

Making compact B/H/N modes runtime dynamic was required to avoid recompilation,
but it also improved the main kernel.  The same dynamic binary passed N=128,
256, and 384 with changing B/H while compile count stayed one for the general
path.

At B1/H4/N512:

| compilation | rotating total ms | NCU main ms | local loads | local stores | long-scoreboard cycles |
|---|---:|---:|---:|---:|---:|
| fully static shape | 2.3504 | 2.7832 | 4,980,736 | 3,112,960 | 3.276 |
| runtime-dynamic B/H/N | 2.1461 | 2.3532 | 0 | 0 | 1.511 |

Registers/thread (168), shared memory (117,760 B), grid (8192 CTAs), and
theoretical occupancy (18.75%) are identical.  The dynamic form prevents the
compiler from specializing/unrolling the runtime M loop into a spill-heavy
version.  It executes more loop-control instructions, but eliminates all local
traffic, raises tensor-active cycles from 18.40% to 21.67%, and reduces NCU
main duration by 15.5%.

The conversion must leave D=64 static and preserve 16-byte pointer alignment:

```python
tensor = from_dlpack(torch_tensor, assumed_align=16, enable_tvm_ffi=True)
for mode in semantic_dynamic_modes:
    tensor = tensor.mark_compact_shape_dynamic(
        mode=mode,
        stride_order=torch_tensor.dim_order(),
        divisibility=tile_divisibility,
    )
```

## 10. dPairBias becomes the large-H critical path

Fused FP32 atomics are the fastest universal path through N=512.  At large N
and H=16, all S matrices contend on the same `[B,H,N,N]` destinations and the
atomic becomes dominant:

| shape | fused atomic total | same core without dPB | atomic contribution |
|---|---:|---:|---:|
| B1/H16/N640 | 27.058 ms | 14.210 ms | 12.849 ms |
| B1/H16/N1024 | 129.529 ms | 58.050 ms | 71.480 ms |

This is why the current fused-atomic production path cannot meet 2x for every
N>=640/H16 shape by core tuning alone.

### Reduction experiments

Isolated dPairBias ownership validates a real reduction opportunity:

| N | per-S atomic | one all-S owner | owner speedup | best split-S |
|---:|---:|---:|---:|---:|
| 128 | 0.0317 ms | 0.1527 ms | 0.21x | 0.0344 ms, chunk 4 |
| 384 | 0.6309 ms | 0.5665 ms | 1.11x | 0.4084 ms, chunk 32 |
| 1024 | 12.6489 ms | 5.7321 ms | 2.21x | 6.70 ms, chunk 128 |

However, a separate owner pass must reconstruct QK/dP/dS and therefore lost
end-to-end when paired with a separate dQ pass.  Moving atomics from dPB to dQ
also lost because it introduced a huge FP32 dQ workspace and an equally large
reduction problem.

The successful large-H strategy is instead:

1. run the fused five-MMA main with dPairBias atomics disabled;
2. recompute dPairBias with a Q64 x K128 x S16 CuTe reduction;
3. retain the fused atomic path for N<=512 and for all H=4 shapes.

The split kernel assigns one CTA to a 64 x 128 pair tile and 16 consecutive
MSA sequences.  Sixteen `mma.sync.m16n8k16` warps reconstruct QK and dO*V^T,
accumulate the 16 dS contributions in FP32 registers, and issue one global
atomic per output element per sequence chunk.  Q, K, V, and dO use a two-stage
TMA pipeline with 128-byte-swizzled shared layouts.  The next stage is issued
before the current MMAs, so its transfers overlap tensor-core and softmax
work.  The invariant FP32 pair-bias tile is loaded directly into each MMA
lane's accumulator fragment; this avoids a 32 KiB shared-memory allocation and
its store/load round trip.

The tile/chunk progression shows why the final combination won:

| isolated CuTe variant | N=640 ms | N=1024 ms | result |
|---|---:|---:|---|
| Q16/K128/S32, scalar shared loads | 84.78 | — | unusable bank conflicts |
| Q16/K128/S32, swizzled TMA | 12.48 | 46.46 | correct, synchronous |
| Q16/K128/S32, two-stage TMA | 11.32 | — | overlap helps, still too slow |
| Q16/K128/S16, pipelined | 9.270 | 37.482 | smaller instruction footprint |
| Q32/K128/S16, pipelined | 9.545 | 37.564 | no reduction in time |
| Q64/K128/S16, shared pair bias | 7.491 | 28.853 | fewer CTAs and K/V rereads |
| Q64/K128/S16, direct pair bias | **6.815** | **27.769** | final |

For comparison, the Triton split-S kernel took 8.793 ms at B1/H16/N640 and
36.141 ms at B1/H16/N1024, so the final CuTe reduction is 1.290x and 1.302x
faster in isolation.  At B4/H16/N640 it took 27.086 ms versus Triton's
35.022 ms, a 1.293x speedup.  The isolated FP32 outputs agreed with Triton to
maximum absolute errors of 9.31e-9, 1.12e-8, and 1.30e-8, respectively.
After integration and BF16 conversion, the full backward had no failures at
`atol=1e-2, rtol=2e-2`; the largest exhaustive difference observed in the
B1/H16/N640 check was 0.03125.

The final N640 NCU profile reports 512 threads, 124 registers/thread, 100.37
KiB dynamic shared memory, 25.0% achieved occupancy, 65.27% L1/TEX throughput,
36.79% L2 throughput, 27.00% DRAM throughput, and 38.53% compute throughput.
This is a substantial shift from the Q16/S32 version's 12.45% occupancy,
42.33% L1/TEX throughput, 14.30% DRAM throughput, and 20.23% compute
throughput.  The final kernel remains primarily constrained by the shared/L1
path and memory latency rather than tensor throughput.

H=4 deliberately stays on fused atomics.  Its lower contention does not repay
the extra two-GEMM reconstruction and launch: the existing atomic path already
achieves 2.732x at N=640 and 2.533x at N=1024.  Split-S is selected only for
H=16/N>=640, where atomic serialization had become the critical path.

## 11. Head flattening does not remove the dPB bottleneck

An alternate prepared layout `[B,S,H,N,D]` allows a no-copy view
`[B*S*H,N,1,D]`, moving H into the problem index.  This changes grid ordering
and slightly spreads dPB atomic arrivals, but it is not enough:

| shape/path | atomic total | no-dPB core | inferred atomic penalty |
|---|---:|---:|---:|
| N640 current `[BS,N,H,D]` | 27.058 | 14.210 | 12.849 |
| N640 flat internal, no output copies | 25.124 | 13.962 | 11.162 |
| N640 flat plus three public-layout copies | 28.757 | 17.526 | 11.232 |
| N1024 current | 129.529 | 58.050 | 71.480 |
| N1024 flat internal, no output copies | 127.774 | 57.119 | 70.655 |
| N1024 flat plus public-layout copies | 137.073 | not separately retained | — |

At N640, flattening saves 1.69 ms of atomic contention but still reaches only
1.92x even if consumers accept the internal layout.  Returning the required
public layout makes it slower than the current representation.  At N1024 the
atomic saving shrinks below 1 ms.  Flattening cannot replace split-S dPB.

## 12. Variants that did not improve the fixed core

All rows below use the rotating B1/H4/N128 benchmark unless noted.

| variant | CuTe ms | outcome |
|---|---:|---|
| M64/N128, 2/2/2 stages, direct dQ | 0.0755-0.0760 | winner |
| same, general dQ reduction | 0.0814 | correct, slower but still >2x |
| 1/1/1 stages | 0.0793 | pipeline too shallow |
| Q2/dO1/PdS1 | 0.0769 | close at N128, loses at larger N |
| Q2/dO2/PdS1 | 0.0757 | close at N128; N256/N512 slower than 2/2/2 |
| 3/3/3 stages | 0.0758 | no gain; more shared state |
| N64 key tile | 0.1077 | slower and duplicates dPB ownership |
| one consumer warpgroup | 0.2278 | insufficient latency hiding |
| both WGs compute dQ | 0.0768 | no benefit over single-dQ-WG |
| `SdP_swapAB=False` | 0.0956 | loses accumulator/shared orientation |
| `dKV_swapAB=True` | 0.0787 | slower |
| `dQ_swapAB=True` | 0.0765 | noise-level change |
| `V_in_regs=True` | 0.0755 | noise-level change |
| score-mod vector width 2/4 | 0.0761 / 0.0757 | no material gain |
| max registers 240/224 | 0.0761 / 0.0762 | no gain |
| TMA evict-last hint | 0.0755 vs 0.0755 default | no measurable effect |

For stage selection at larger N, 2/2/2 is clearly preferable:

| N | 2/2/2 ms | 2/2/1 ms |
|---:|---:|---:|
| 256 | 0.4127 | 0.4202 |
| 512 | 2.3401 | 2.3839 |

Those two rows were collected before the dynamic-shape spill fix; the relative
stage conclusion remains valid.

## 13. Benchmarking lessons

- CUDA events must enclose every required launch.  Per-kernel profiler sums do
  not include stream starvation between Python calls.
- Rotate prepared address sets.  One-address microbenchmarks hide L2 reuse and
  allocator/dispatch behavior; B1/H4/N128 was the shape most sensitive to this.
- Warm every pool slot before timing.  Three warmups with an eight-slot pool
  accidentally timed workspace construction and produced 2-3x variance.
- Generate large-tensor sample indices with integer arithmetic.  FP32-backed
  `linspace` can round `numel-1` up to `numel` beyond 2^24.
- Keep Triton layout conversion symmetric: its three dQ/dK/dV transpose-copy
  launches and final dPB cast belong inside the reference interval.
- Do not retain one FP32 dQ workspace per maximum-size rotating slot.  Production
  chunks batches to cap each workspace at 8 GiB and exposes a cache-clear hook
  to the benchmark.

## 14. Runtime-shape and stream-safety hardening

The final launcher marks dimensions dynamic with semantic, shape-independent
divisibility facts: batch has divisibility 1, heads 4, and S/N 128.  Inferring
divisibility from the first input would make a binary first compiled at B=4
invalid for a later B=1 call.  A one-process B4/N128 -> B1/N128 -> N256 ->
H16/N640 -> N1024 test now passes while all three runtime paths remain inside
one `CompileCallable`.  The benchmark also uses one common rotating-pool size
and iteration count for both implementations at each shape.

Prepared launch records and FP32 workspaces use pointer/device keys, bounded
LRU caches, a 16 GiB workspace budget, and an explicit cache-clear API.  The
two internally allocated workspaces are recorded on every launch stream before
dispatch.  This matters when a custom stream is itself current: comparing the
requested stream only with `torch.cuda.current_stream()` incorrectly labels it
as safe, then an immediate cache clear can recycle storage while the kernel is
still running.  The final stress test clears the cache, deliberately poisons
reused allocations, and observes zero gradient mismatches for both explicit
and current custom-stream forms.  Twelve rotating N128 buffer sets also verify
that both caches plateau at eight entries.

## 15. Final acceptance matrix

The final benchmark uses CUDA events around the complete backward operation,
including all launches, reductions, zeroing, BF16 dPairBias conversion, and
the Triton baseline's three model-layout transpose copies. Allocation, forward
preparation, and compilation are excluded. Both implementations use the same
rotating-pool size and iteration count for each shape, with 10 warmups and the
median of seven repeat averages.

| B | H | N | Triton ms | CuTe ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 128 | 0.1887 | 0.0744 | 2.534x |
| 1 | 4 | 256 | 0.9155 | 0.3749 | 2.442x |
| 1 | 4 | 384 | 2.5983 | 1.0078 | 2.578x |
| 1 | 4 | 512 | 5.7452 | 2.1558 | 2.665x |
| 1 | 4 | 640 | 10.7400 | 3.9387 | 2.727x |
| 1 | 4 | 768 | 18.2108 | 6.6429 | 2.741x |
| 1 | 4 | 896 | 28.1342 | 10.8846 | 2.585x |
| 1 | 4 | 1024 | 42.4384 | 16.6270 | 2.552x |
| 1 | 16 | 128 | 0.6511 | 0.2689 | 2.422x |
| 1 | 16 | 256 | 3.5063 | 1.4347 | 2.444x |
| 1 | 16 | 384 | 10.2620 | 3.9908 | 2.571x |
| 1 | 16 | 512 | 23.0401 | 9.8401 | 2.341x |
| 1 | 16 | 640 | 48.9424 | 21.3854 | 2.289x |
| 1 | 16 | 768 | 91.3163 | 38.0559 | 2.400x |
| 1 | 16 | 896 | 143.2949 | 60.5471 | 2.367x |
| 1 | 16 | 1024 | 210.1223 | 89.6514 | 2.344x |
| 4 | 4 | 128 | 0.6481 | 0.2642 | 2.453x |
| 4 | 4 | 256 | 3.4899 | 1.4323 | 2.437x |
| 4 | 4 | 384 | 10.2362 | 3.8984 | 2.626x |
| 4 | 4 | 512 | 22.6536 | 8.4103 | 2.694x |
| 4 | 4 | 640 | 42.5450 | 15.6176 | 2.724x |
| 4 | 4 | 768 | 71.5239 | 26.1604 | 2.734x |
| 4 | 4 | 896 | 111.5652 | 44.2230 | 2.523x |
| 4 | 4 | 1024 | 164.7169 | 66.0500 | 2.494x |
| 4 | 16 | 128 | 2.4559 | 0.9828 | 2.499x |
| 4 | 16 | 256 | 13.8917 | 5.6299 | 2.467x |
| 4 | 16 | 384 | 41.1653 | 16.0091 | 2.571x |
| 4 | 16 | 512 | 92.4065 | 39.5033 | 2.339x |
| 4 | 16 | 640 | 194.6316 | 86.5380 | 2.249x |
| 4 | 16 | 768 | 366.1703 | 156.7875 | 2.335x |
| 4 | 16 | 896 | 573.9297 | 244.3882 | 2.348x |
| 4 | 16 | 1024 | 841.0437 | 358.2977 | 2.347x |

All 32 shapes exceeded 2x, with a minimum of 2.249x. Correctness sampled
65,536 evenly spaced elements from each of dQ, dK, dV, and dPairBias at every
shape using `atol=1e-2, rtol=2e-2`. There were zero mismatches. Maximum
absolute errors were 0.001953, 0.003906, 0.001953, and 0.031250 respectively.
The single-process runtime and stream-safety suite also kept compile count at
one across direct, fused-atomic, and split-dPairBias paths.

## 16. Remaining large-shape opportunity

The split-S kernel is now integrated behind the H=16/N>=640 branch and closes
the former dPairBias acceptance gap.  The principal remaining opportunity is
to reduce its repeated K/V traffic rather than to change the reduction
algorithm.

The longer-term Hopper-native design is a two-CTA cluster over adjacent query
tiles in the split-S kernel.  Both ranks use the same K, V, residual mask, and
pair-bias tile, so rank 0 can multicast those operands.  Q and dO remain
private.  Every target N has an even number of 128-query tiles, including
N=384, 640, and 896, so no odd cluster tail is required.

KernelWiki informed the warp-specialized, TMA, swizzle, and ping-pong choices.
The Blackwell FlashAttention-4 two-CTA implementation is not directly portable
to Hopper, so all SM90 multicast/reduction claims above remain experimental
until verified by H100 NCU counters.
