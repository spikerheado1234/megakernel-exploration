# EvoAttention forward: optimization history and final design

## 1. Executive summary

This directory contains the retained CuTe-DSL forward implementation for
EvoAttention on NVIDIA H100 (SM90a). The implementation is intentionally
specialized to head dimension 64 and to sequence lengths divisible by 128, but
batch size, head count, and sequence length remain runtime dynamic. One
TVM-FFI compilation serves the complete target matrix.

The retained kernel uses:

- BF16 query, key, value, and output tensors;
- FP32 pair bias, additive residual mask, and log-sum-exp output;
- one TMA producer warp and two 128-thread WGMMA consumer warpgroups;
- M=64 per consumer and an effective M=128 CTA tile;
- Bc=64 and two K/V stages;
- register-sourced probability operands for the PV WGMMA;
- a strict FA3-style diagonal ping-pong schedule for N greater than 128;
- an N=128 fast path that removes steady-state synchronization which cannot
  overlap useful work when there are only two K/V tiles;
- MN_SW128 pair-bias and output layouts;
- StMatrix register-to-shared output staging followed by one 128x64 TMA store;
- a logical zero-stride residual-mask broadcast, backed by only one FP32
  vector per K/V stage;
- an evict-last TMA L2 policy for pair bias;
- a 288-thread launch with min_blocks_per_mp=2.

The final pre-refactor candidate was SHA-256
3278052699cd1da337adcf4402e9f99c23854c99ad00514651f5a0bb6ad7e25e.
It passed all 32 correctness cases with one dynamic compilation. The maximum
absolute output error was 0.0078125 and the maximum FP32 LSE error was
1.90735e-6.

The behavior-preserving cleaned kernel is SHA-256
19d404a7f778bf4dca42918530116df55fe6f1d5f2e86b2f23f075b4a71c55dd.

The original fixed-buffer acceptance benchmark reached 2x or better on 30 of
32 shapes before the final N=128 changes. The retained short path subsequently
measured 2.0001x at B=1, H=16, N=128 and 2.016x at B=4, H=4, N=128 in its
focused run. These two results sit on the noise boundary and should not be
interpreted as a guaranteed 2x on every H100.

The cleaned benchmark now follows the repository microbenchmark convention of
rotating prepared input and output addresses through a bounded pool. This is a
stricter cache-residency test than the historical fixed-buffer runs, so its
numbers must not be mixed with the tables below.

## 2. Operation and tensor contract

For each batch b, sequence row s, head h, query i, and key j:

    score[b,s,h,i,j] =
        dot(q[b,s,h,i,:], k[b,s,h,j,:]) / sqrt(64)
        + pair_bias[b,h,i,j]
        + residual_mask[b,s,j]

    probability[b,s,h,i,:] = softmax(score[b,s,h,i,:])
    output[b,s,h,i,:] = probability[b,s,h,i,:] @ value[b,s,h,:,:]

The prepared-layout API is:

| Tensor | Shape | Type |
|---|---|---|
| query, key, value, output | [B, S, H, N, 64] | BF16 |
| pair bias | [B, H, N, N] | FP32 |
| residual mask | [B, S, N] | FP32, values 0 or -1e9 |
| log-sum-exp | [B, S, H, N] | FP32 |

All accepted workloads use S=N:

- B in {1, 4}
- H in {4, 16}
- N in {128, 256, 384, 512, 640, 768, 896, 1024}

The fused implementation never materializes the B x S x H x N x N score
tensor. It computes 4*B*S*H*N^2*D tensor FLOPs for QK and PV.

## 3. Initial roofline diagnosis

The initial model used these H100 rates:

| Resource | Assumed peak |
|---|---:|
| HBM to L2 | 3.35 TB/s |
| L2 to shared memory | 7 TB/s |
| Shared memory to registers | 30 TB/s |
| FP32 CUDA cores | 29.7 T scalar operations/s |
| MUFU exponential | 3.71 T results/s |
| BF16 tensor cores | 990 TFLOP/s |

With M=64, Bc=64, and no multicast, the predicted B=1, H=4 lower bounds were:

| N | DRAM | L2 to SMEM | SMEM to RF | MUFU | CUDA | tensor |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 10.2 us | 12.1 us | 1.4 us | 2.3 us | 2.8 us | 2.2 us |
| 384 | 91.7 us | 282.8 us | 33.2 us | 62.0 us | 76.3 us | 58.6 us |
| 1024 | 652.3 us | 5102.7 us | 599.5 us | 1175.8 us | 1446.1 us | 1110.6 us |

The important difference from ordinary FlashAttention is the FP32 pair-bias
term. Even with ideal K/V reuse, L2 intensity asymptotically approaches only
64 tensor FLOP/byte. The modeled H100 L2 ridge is about 141 FLOP/byte.
Therefore pair-bias traffic moves the medium and long shapes toward an L2
bottleneck, while N=128 remains dominated by fixed pipeline and scheduling
latency.

The model selected M=64 for each consumer because every target N is divisible
by 64 and m64 is a native Hopper WGMMA extent. Two consumers share K/V, giving
an effective CTA reuse width of 128 without padding a larger WGMMA query tile.

## 4. Final kernel schedule

The CTA contains nine warps:

    warps 0-3: consumer warpgroup 0, query rows [0, 64)
    warps 4-7: consumer warpgroup 1, query rows [64, 128)
    warp 8:    sole TMA producer

The producer loads two Q tiles and advances a two-stage ring containing K, V,
and the residual-mask vector. Each consumer independently loads its own
64x64 FP32 pair-bias tile. For N greater than 128, named barriers offset the
two consumers:

    key block k:
      WG0: QK(k) -> softmax(k) -> waits for WG1 QK(k) -> PV(k)
      WG1: waits for WG0 QK(k) -> QK(k) -> softmax(k)
           -> waits for WG0 QK(k+1) -> PV(k)

This alternates tensor-core and CUDA/SFU work between the consumer groups.
With N=128 there are only two K/V blocks, so there is no useful steady state.
The final short path skips those named-barrier operations and the K/V-free
barriers whose stages are never reused.

### 4.1 Logical mask broadcast

Only 64 FP32 mask values are physically stored per K/V stage. A zero-stride
query mode exposes the logical 64x64 broadcast:

    shared_broadcast_mask = cute.make_tensor(
        shared_mask.iterator,
        cute.make_layout(
            (QUERY_TILE_SIZE, KEY_TILE_SIZE, KEY_VALUE_STAGES),
            stride=(0, 1, KEY_TILE_SIZE),
        ),
    )

Two score rows sharing a WGMMA quartet load each mask column once:

    mask_0 = shared_broadcast_mask[0, column_0, stage]
    mask_1 = shared_broadcast_mask[0, column_1, stage]
    score_0 = qk_0 * SOFTMAX_SCALE + bias_0 + mask_0
    score_2 = qk_2 * SOFTMAX_SCALE + bias_2 + mask_0

This saves 63/64 of the shared-memory capacity that a materialized MxK mask
would require and avoids redundant shared loads.

Pair bias is also logically broadcast over S, but S belongs to distinct CTAs
in this schedule. A local CuTe zero-stride layout cannot share physical SMEM
between CTAs. Cluster multicast was therefore the correct mechanism tested for
that reuse; it was not retained because cluster scheduling cost outweighed the
traffic reduction on the complete shape set.

### 4.2 Split query completion barriers

Each consumer has its own Q completion barrier:

    query_ready = allocator.allocate_array(
        cutlass.Int64, CONSUMER_WARPGROUP_COUNT
    )
    cute.arch.mbarrier_arrive_and_expect_tx(
        query_ready + consumer, QUERY_TILE_SIZE * HEAD_DIMENSION * 2
    )
    cute.arch.mbarrier_wait(query_ready + consumer_warpgroup, 0)

The earlier implementation used one barrier covering both Q transfers. That
forced consumer 0 to wait for consumer 1's independent tile. Splitting the
barrier improved the two difficult N=128 cases by about 1.05-1.07%.

### 4.3 Register-sourced PV

The probability tile remains in the QK accumulator registers, is converted to
BF16, and feeds PV as an RMEM A operand:

    probability_fragment = output_mma.make_fragment_A(
        output_mma.partition_shape_A((QUERY_TILE_SIZE, KEY_TILE_SIZE))
    )
    probability_fragment.store(
        score_fragment.load().to(cutlass.BFloat16)
    )
    cute.gemm(
        output_mma,
        output_accumulator,
        probability_fragment,
        value_fragment[None, None, None, stage],
        output_accumulator,
    )

This removed a large probability shared-memory round trip. At B=1, H=4,
N=384 the NCU duration fell from 729.5 us in the prior shared-heavy kernel to
637.4 us, registers fell from 141 to 108 per thread, and shared memory fell
from 149,608 to 133,224 bytes per CTA.

### 4.4 Quartet softmax work

An m64n64 WGMMA accumulator maps two logical rows to each thread and four
adjacent lanes cover their columns. The implementation performs the row max
and sum reductions within each aligned four-lane subgroup. Only lane zero of
each quartet evaluates the online-rescale exponential and reciprocal; the
result is broadcast with shuffle_sync.

This avoids four identical transcendental operations for each logical row
without changing the online-softmax recurrence:

    new_max = max(running_max, local_max)
    alpha = exp(running_max - new_max)
    running_sum = running_sum * alpha + local_sum
    output_accumulator = output_accumulator * alpha + probability @ value

### 4.5 Output epilogue

Each consumer converts its FP32 m64n64 accumulator and stores it into one half
of a combined MN_SW128 128x64 BF16 shared tile using StMatrix:

    output_bf16 = cute.make_fragment_like(
        output_accumulator, cutlass.BFloat16
    )
    output_bf16.store(output_accumulator.load().to(cutlass.BFloat16))
    output_retile = output_register_copy.retile(output_bf16)
    cute.copy(output_register_copy, output_retile, output_partition)

After one 256-thread barrier, consumer 0 issues one TMA S2G operation for the
whole tile. The isolated epilogue emitted four STMATRIX instructions, one
UTMASTG, zero shared-store conflicts, and 384 total SASS instructions. Its
median was 3.3838 us on GPU2 and 3.0985 us on GPU3.

### 4.6 Dynamic TVM-FFI dispatch

The public wrapper flattens the prepared tensors without copying and marks
B*S*H and N dimensions dynamic. Tiled dimensions carry divisibility 64. The
compiled callable is cached under a lock and rejects cross-device reuse:

    dynamic_tensor = from_dlpack(
        tensor.detach(),
        assumed_align=16,
        enable_tvm_ffi=True,
    )
    dynamic_tensor = dynamic_tensor.mark_compact_shape_dynamic(
        mode=mode,
        stride_order=tensor.dim_order(),
        divisibility=64,
    )

All 32 target shapes execute through one compiled specialization.

## 5. Optimization campaign

The measurements in this section use the original fixed prepared-buffer
methodology: CUDA events, compilation/allocation/layout preparation excluded,
and medians across repeated launch batches.

### 5.1 Initial WGMMA implementation

The first tensor-core implementation established the correct dynamic ABI,
TMA staging, two consumers, and FP32 online softmax.

Result: all shapes were correct, but speedup ranged from 1.212x to 1.529x.
Long H=16 cases fell toward 1.25x because the kernel restreamed too much data
and did not overlap the non-matmul work effectively.

### 5.2 Output TMA and early ping-pong

The scalar output path was replaced with a shared-memory staging tile and TMA
store, and the consumer ordering was converted toward diagonal ping-pong.

Result: speedup improved to 1.339-1.825x. This established that output
transactions and consumer scheduling mattered, but no shape reached 2x
reliably.

### 5.3 Strict ping-pong and pair-bias swizzle

The schedule used named barriers rather than full-CTA synchronization in the
main loop, and pair bias used an MN-swizzled layout.

Result: 1.278-1.858x. The schedule was directionally useful at medium/large N,
but the SW64 layout and shared probability path still left too much shared
traffic.

### 5.4 Register-sourced PV

The probability operand moved from shared memory to RMEM WGMMA input.

Result: seven of 32 cases reached 2x, with a range of 1.379-2.143x.
NCU at B=1, H=4, N=384 showed:

| Metric | Before | RMEM PV |
|---|---:|---:|
| Duration | 729.5 us | 637.4 us |
| Registers/thread | 141 | 108 |
| Shared memory/CTA | 149,608 B | 133,224 B |
| Shared-store pressure | high | materially lower |

This was the first decisive optimization.

### 5.5 Two-way pair-bias multicast

A correct SM90 cluster-two multicast was implemented. Cluster peers varied S
while holding batch, head, query tile, and key tile fixed. Both ranks issued
their TMA partitions and each local barrier accounted for the full tile.

The transport worked: one 16 KiB global source tile served two CTAs. However,
the full kernel reached 2x on only five of 32 shapes, with a range of
1.332-2.035x. Cluster launch and scheduling costs outweighed saved traffic for
short and medium shapes. It helped selected long H=16 cases but was rejected
for the required universal configuration.

### 5.6 Bc=128

Doubling the K/V tile halved loop and rescale overhead. At B=1, H=4, N=384,
NCU duration fell from 637.4 to 544.8 us, long-scoreboard samples fell from
17,200 to 5,031, and tensor-pipe active time rose from 13.75% to 16.07%.

The cost was 168 registers/thread and 199,240 bytes of shared memory per CTA.
The full sweep reached 2x on 20/32 shapes. Later SW128 and two-stage changes
raised that to 22/32, but N=128/256 and the largest H=16 cases remained weak.
Bc=128 was rejected as the universal tile because its one-CTA residency and
register footprint were worse than the final Bc=64 two-resident-CTA design.

### 5.7 Bc=64, 288 threads, and two CTAs per SM

The decisive configuration aligned the consumers to warps 0-7 and assigned
only warp 8 to TMA production. With Bc=64, two K/V stages, and
min_blocks_per_mp=2, the compiled kernel used:

| Resource | Value |
|---|---:|
| Threads/CTA | 288 |
| Registers/thread | 96 |
| Shared memory/CTA | 98,872 B |
| Local-memory spills | 0 |
| Resident CTAs/SM | 2 |
| Resident warps/SM | 18 |

Two CTAs require 55,296 registers and 197,744 shared-memory bytes, both within
H100 limits. This configuration, together with SW128 pair bias, RMEM PV,
quartet softmax, and the StMatrix epilogue, produced the 30/32 robust result.

### 5.8 N=128 fixed-cost reductions

The two remaining misses had identical effective work: 2,048 CTAs and only two
K/V loop iterations. They were neither tensor-throughput nor HBM-bandwidth
limited; fixed barriers and fill/drain latency dominated.

Three changes were retained:

1. Split Q readiness so each consumer waits only for its own TMA tile.
2. Disable named consumer ping-pong at N=128 because there is no steady state
   to establish.
3. Do not initialize or arrive on K/V-free barriers at N=128 because neither
   of the two stages is reused.

Focused measurements:

| Case | Triton | CuTe | Speedup |
|---|---:|---:|---:|
| B=1, H=16, N=128 | 0.15821 ms | 0.07910 ms | 2.0001x |
| B=4, H=4, N=128 | 0.15936 ms | 0.07905 ms | 2.0160x |

The gain is real but small enough that clock state and benchmark cache policy
can move the ratio across 2x.

## 6. Experiments that did not survive

### 6.1 S-fast CTA raster

The pair bias is invariant across S, so CTAs were reordered to visit S
consecutively for a fixed batch/head/query tile. This was intended to improve
L2 reuse without physical replication.

Result: N=128 regressed from about 81.5 us to 83.0-83.6 us, roughly 2%.
The original raster was restored. L2 already retained enough pair-bias data,
while the altered CTA ordering harmed scheduling locality elsewhere.

### 6.2 Universal cluster multicast

Cluster-two and cluster-four pair-bias multicast were both functionally
correct. Cluster four reduced source traffic per recipient further and helped
some long H=16 cases. It substantially hurt small shapes and could not be
enabled selectively under the one-configuration requirement, so the final
kernel uses ordinary TMA with pair-bias evict-last caching.

### 6.3 K/V evict-first policy

The 0x12F0000000000000 TMA policy emitted the intended L2 cache-hint SASS, but
slightly regressed long shapes. It was removed. Pair bias retained the
0x14F0000000000000 evict-last hint because it is reused across S.

### 6.4 Combined Q and pair-bias transactions

Combining adjacent logical tiles reduced the number of TMA instructions but
did not improve the end-to-end critical path; variants were neutral or
slightly slower. Independent transactions provide better overlap.

### 6.5 Head-grouped grid ordering

Grouping CTAs by head to encourage pair-bias locality produced no measurable
gain and was removed.

### 6.6 First-PV overwrite and first-iteration alpha specialization

Special-casing the first online-softmax iteration can theoretically remove an
output rescale by zero. In practice the extra dynamic branch and code size were
neutral or slightly regressive at N=128. The uniform recurrence was retained.

### 6.7 Direct global output epilogue

Writing the WGMMA accumulator directly to global memory was neutral or slower.
The combined StMatrix plus TMA epilogue has cleaner code generation and fewer
transactions.

### 6.8 Q/output shared-memory alias

The lifetimes permit output scratch to reuse Q storage after the final QK.
Aliasing saves 16 KiB, but does not create a third resident CTA because the
register allocation remains the limiting resource. It added complexity
without an occupancy gain and was not retained.

### 6.9 More stages

With FP32 pair bias, deeper full staging consumes shared memory quickly.
Three-stage Bc=128 or four independent pair-bias slots either approach or
exceed the 227 KiB opt-in limit. The final two-stage Bc=64 layout is smaller
and enables two resident CTAs.

### 6.10 Software exponential

The roofline showed H100 MUFU close to, but not clearly beyond, the tensor
budget. The measured bottlenecks were L2/shared latency and scheduling rather
than saturated exponential throughput. A software exponential, useful in
FlashAttention-4 on Blackwell, was therefore not prioritized for this Hopper
kernel.

## 7. Pair-bias layout experiments

An isolated coordinate-tagged probe compared six FP32 layouts. All were exact.
The best path was MN_SW128 with order (1,0):

| Layout | Order | TMA loads | LDS instructions | NCU duration |
|---|---|---:|---:|---:|
| MN_SW32 | (0,1) | 128 | 32 | 14.592 us |
| MN_SW32 | (1,0) | 16 | 32 | 5.536 us |
| MN_SW64 | (0,1) | 64 | 32 | 9.728 us |
| MN_SW64 | (1,0) | 8 | 32 | 5.184 us |
| MN_SW128 | (0,1) | 32 | 32 | 7.168 us |
| MN_SW128 | (1,0) | 4 | 32 | 5.120 us |

The physical TMA view is [K,Q,problem]. Consumers expose an inverse [Q,K]
logical view to partition_C:

    shared_pair_bias_row_col = cute.make_tensor(
        shared_pair_bias.iterator,
        cute.select(shared_pair_bias.layout, mode=[1, 0, 2]),
    )
    pair_bias_partition = score_thread.partition_C(
        shared_pair_bias_row_col[None, None, consumer_warpgroup]
    )

This is an example where the CuTe layout, TMA descriptor, and consumer
partition must be designed together. A swizzle enum alone is not sufficient.

## 8. Benchmark history

The main fixed-buffer checkpoints were:

| Checkpoint | Shapes at least 2x | Speedup range | Main change |
|---|---:|---:|---|
| Initial WGMMA | 0/32 | 1.212-1.529x | tensor-core baseline |
| Output TMA + early ping | 0/32 | 1.339-1.825x | combined output store |
| Strict ping + SW64 PB | 0/32 | 1.278-1.858x | named barriers |
| RMEM PV | 7/32 | 1.379-2.143x | eliminate P shared round trip |
| PB multicast | 5/32 | 1.332-2.035x | cluster-two PB reuse |
| Bc=128 | 20/32 | not retained | halve loop count |
| Bc=128, SW128, KV2 | 22/32 | not retained | better PB load path |
| Final Bc=64 occupancy | 30/32 | 1.936-2.889x | two CTAs/SM |
| Final N=128 fast path | focused only | 2.000-2.016x | remove fixed sync |

The robust pre-fast-path table was:

| B | H | N | Triton ms | CuTe ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 128 | 0.0416 | 0.0203 | 2.050x |
| 1 | 4 | 256 | 0.2580 | 0.1043 | 2.474x |
| 1 | 4 | 384 | 0.8007 | 0.3026 | 2.646x |
| 1 | 4 | 512 | 1.8267 | 0.6463 | 2.826x |
| 1 | 4 | 640 | 3.4628 | 1.2653 | 2.737x |
| 1 | 4 | 768 | 5.9060 | 2.0552 | 2.874x |
| 1 | 4 | 896 | 9.2494 | 3.3146 | 2.790x |
| 1 | 4 | 1024 | 13.6898 | 4.8474 | 2.824x |
| 1 | 16 | 128 | 0.1582 | 0.0817 | 1.936x |
| 1 | 16 | 256 | 1.0263 | 0.3793 | 2.706x |
| 1 | 16 | 384 | 3.1794 | 1.1848 | 2.684x |
| 1 | 16 | 512 | 7.3092 | 2.6014 | 2.810x |
| 1 | 16 | 640 | 14.0950 | 5.6063 | 2.514x |
| 1 | 16 | 768 | 25.0403 | 9.6744 | 2.588x |
| 1 | 16 | 896 | 40.8433 | 16.1358 | 2.531x |
| 1 | 16 | 1024 | 61.1600 | 22.5718 | 2.710x |
| 4 | 4 | 128 | 0.1591 | 0.0814 | 1.953x |
| 4 | 4 | 256 | 1.0167 | 0.3964 | 2.565x |
| 4 | 4 | 384 | 3.2004 | 1.1381 | 2.812x |
| 4 | 4 | 512 | 7.2678 | 2.6272 | 2.766x |
| 4 | 4 | 640 | 13.9843 | 4.8970 | 2.856x |
| 4 | 4 | 768 | 23.8898 | 8.4175 | 2.838x |
| 4 | 4 | 896 | 37.4564 | 12.9669 | 2.889x |
| 4 | 4 | 1024 | 56.3831 | 20.5749 | 2.740x |
| 4 | 16 | 128 | 0.6202 | 0.2861 | 2.168x |
| 4 | 16 | 256 | 4.0866 | 1.5487 | 2.639x |
| 4 | 16 | 384 | 12.8778 | 4.5824 | 2.810x |
| 4 | 16 | 512 | 29.5929 | 10.9202 | 2.710x |
| 4 | 16 | 640 | 58.0796 | 22.2651 | 2.609x |
| 4 | 16 | 768 | 102.7112 | 39.5735 | 2.595x |
| 4 | 16 | 896 | 172.9935 | 62.7385 | 2.757x |
| 4 | 16 | 1024 | 256.2859 | 93.2586 | 2.748x |

## 9. Correctness and benchmark methodology

Correctness compares directly with the original fused MegaFold Triton kernel,
not with a separately reconstructed PyTorch equation. It verifies BF16 output
and FP32 log-sum-exp using atol=0.02 and rtol=0.02. Comparisons are chunked to
bound temporary FP32 memory at the largest shape.

The largest reference launch would exceed CUDA's grid-y limit of 65,535.
The harness handles only that reference limitation by issuing contiguous
per-batch calls; the CuTe kernel keeps one flattened x grid.

The cleaned benchmark:

1. creates contiguous prepared-layout tensors before timing;
2. JIT-compiles both implementations before calibration;
3. uses CUDA events, so Python launch overhead is excluded;
4. chooses an iteration count targeting about 200 ms per repeat;
5. reports the median of five repeat averages;
6. alternates implementation order by shape to reduce thermal bias;
7. rotates through up to eight prepared address slots under a 2 GiB pool
   budget, avoiding an unrealistically hot single-address working set;
8. checks that the CuTe wrapper compiled exactly once.

Because address rotation is stricter than the historical benchmark, compare
new runs only with other runs from the cleaned benchmark.

The first full run of the cleaned benchmark passed 29/32 shapes. Its three
misses were all N=128:

| B | H | N | Pool slots | Triton | CuTe | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 128 | 8 | 0.04816 ms | 0.02676 ms | 1.800x |
| 1 | 16 | 128 | 8 | 0.15965 ms | 0.08260 ms | 1.933x |
| 4 | 4 | 128 | 8 | 0.16131 ms | 0.08264 ms | 1.952x |

All N>=256 shapes passed 2x under address rotation. The measured range there
was 2.378-2.877x; B=4, H=16, N=128 also passed at 2.141x. This confirms that
the remaining weakness is the shallow two-iteration pipeline, not the long
sequence data path.

## 10. Lessons

- The largest theoretical reuse tile was not the fastest universal tile.
  Bc=64 won because two CTAs could reside per SM with no spills.
- Removing a shared-memory round trip was more valuable than adding stages.
- Logical broadcasting should be represented by zero strides. Materializing
  the residual mask would waste both capacity and shared-memory bandwidth.
- Broadcast across CTAs requires multicast or persistent work reuse; a local
  layout cannot make shared memory cross CTA boundaries.
- Pair-bias multicast reduced source traffic exactly as expected, but launch
  and cluster scheduling costs made it a net loss for a universal kernel.
- SW128 reduced TMA decomposition even though its isolated shared-load bank
  conflict counter was not zero.
- Ping-pong is valuable only when the loop is deep enough to establish steady
  state. N=128 is faster without it.
- Occupancy changes must be checked against both shared memory and registers.
  Saving 16 KiB through aliasing has no value if registers still prevent an
  additional CTA.
- Source-level simplifications must be measured at SASS or end-to-end level.
  Combined TMA requests, first-iteration special cases, and direct stores all
  looked cheaper in source but were neutral or slower.

## 11. External design references

The implementation choices are consistent with these KernelWiki pages:

- kernel-flash-attention-4, wiki/kernels/flash-attention-4.md
- hw-tma, wiki/hardware/tma.md
- lang-cute-dsl, wiki/languages/cute-dsl.md

Those pages support the general use of TMA, warp specialization, layout-aware
swizzling, and ping-pong scheduling. All EvoAttention performance and
correctness claims in this document come from measurements in this repository,
not from the external references.
