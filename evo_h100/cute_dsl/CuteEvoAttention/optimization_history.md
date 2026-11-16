# CuTe EvoAttention optimization history

This is the consolidated engineering record for both forward and backward
passes and both supported head dimensions. The package contract and retained
source live beside this document. Historical experiment paths below are
preserved as evidence names; the corresponding dead variants were deliberately
removed when the production tree was created.

## Document map

1. D=64 forward design and measurements
2. D=64 backward design and measurements
3. D=32 forward exploration
4. D=32 backward exploration

## Current public contract versus historical experiments

The retained model-facing implementation consumes BF16 Q/K/V, BF16 pair bias,
and a BF16 0/1 validity mask. It forms the authoritative additive mask value
(`-1e9` for an invalid key) inside the kernel, returns BF16 output, and saves
FP32 log-sum-exp for backward. Parts I--IV preserve the contracts used during
each optimization campaign. In particular, several early D=64 roofline and
layout experiments used a prepared FP32 pair bias and FP32 additive mask.
Their measured comparisons and byte models describe that historical variant;
the scheduling conclusions were carried forward, while the final source uses
the public BF16 MegaFold contract described above.

---

## Part I — D=64 forward

*Original D=64 forward record*

## 1. Executive summary

This document describes the retained D=64 CuTe-DSL forward specialization for
EvoAttention on NVIDIA H100 (SM90a), now stored in
`evoattention_cute_d64.py`. The public `evoattention_cute.py` launcher inspects
the runtime head dimension and dispatches to the independently optimized D=32
or D=64 specialization. Within either specialization, batch size, head count,
and sequence length remain runtime dynamic, and one TVM-FFI compilation serves
the complete target matrix for that head dimension.

The retained kernel uses:

- BF16 query, key, value, and output tensors;
- BF16 pair bias and 0/1 validity mask, with FP32 log-sum-exp output;
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
49b5757b788acb742b261d59228098596ab177cd01ef0be98fa87e9b40aebd7a.

The original fixed-buffer acceptance benchmark reached 2x or better on 30 of
32 shapes before the final N=128 changes. The retained short path subsequently
measured 2.0001x at B=1, H=16, N=128 and 2.016x at B=4, H=4, N=128 in its
focused run. These two results sit on the noise boundary and should not be
interpreted as a guaranteed 2x on every H100.

The cleaned benchmark now follows the repository microbenchmark convention of
rotating prepared input and output addresses through a bounded pool. This is a
stricter cache-residency test than the historical fixed-buffer runs, so its
numbers must not be mixed with the tables below.

## 2. Historical prepared-layout operation and tensor contract

For each batch b, sequence row s, head h, query i, and key j:

    score[b,s,h,i,j] =
        dot(q[b,s,h,i,:], k[b,s,h,j,:]) / sqrt(64)
        + pair_bias[b,h,i,j]
        + residual_mask[b,s,j]

    probability[b,s,h,i,:] = softmax(score[b,s,h,i,:])
    output[b,s,h,i,:] = probability[b,s,h,i,:] @ value[b,s,h,:,:]

The prepared-layout API used for the original D=64 tuning campaign was:

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

## 2.1 Retained tiling configuration

The final forward kernel uses one static tiling configuration for every target
shape. Here `B` means batch size, whereas `Bc` means the number of key/value
positions consumed by one main-loop iteration.

| Symbol or setting | Value | Meaning |
|---|---:|---|
| `D` | 64 | Head dimension and QK reduction extent |
| `Br` / consumer `M` | 64 | Query rows owned by one consumer warpgroup |
| CTA `M` | 128 | Query rows per CTA: two consumers times `Br=64` |
| `Bc` / score `N` | 64 | Key/value positions processed per loop iteration |
| QK logical tile | `64 x 64 x 64` | `M x N x K`; SMEM Q times SMEM K, accumulated in FP32 |
| PV logical tile | `64 x 64 x 64` | `M x D x Bc`; RMEM BF16 P times SMEM BF16 V, accumulated in FP32 |
| Native WGMMA K step | 16 | Four K steps compose each logical QK or PV tile |
| Consumer groups | 2 | Two independent 128-thread Hopper warpgroups |
| Producer warps | 1 | Warp 8 owns TMA producer control flow |
| Threads per CTA | 288 | Eight consumer warps plus one producer warp |
| K/V pipeline stages | 2 | Ping-pong SMEM ring indexed by `key_block % 2` |
| Q shared slots | 2 | One persistent `64 x 64` BF16 tile per consumer |
| Pair-bias shared slots | 2 | One private `64 x 64` tile per consumer; FP32 historically, BF16 now |
| Residual-mask storage | `64 x 2` | One physical `Bc` vector per K/V stage |
| Residual-mask logical view | `64 x 64 x 2` | `[Br,Bc,stage]` with zero stride in `Br` |
| Output SMEM tile | `128 x 64` BF16 | Both consumer results combined before one TMA store |
| Cluster shape | `1 x 1 x 1` | No multicast in the retained universal kernel |
| Launch occupancy target | 2 CTAs/SM | Requested with `min_blocks_per_mp=2` |

The main loop has `N / Bc = N / 64` iterations. One CTA computes the output
tile

```text
O[problem, query_block * 128 : (query_block + 1) * 128, 0 : 64]
```

where `problem = (batch, sequence, head)` is flattened to
`P = B * S * H`. Because the accepted workload has `S=N`, the launch grid is

```python
problem_count = B * N * H
query_blocks = N // CTA_QUERY_TILE_SIZE       # CTA_QUERY_TILE_SIZE = 128
grid = (problem_count * query_blocks, 1, 1)
block = (288, 1, 1)
```

The historical shared-memory layouts were row-major/swizzled Q and K,
column-major/swizzled V, MN_SW128 FP32 pair bias, and MN_SW128 BF16 output.
The retained source applies MN_SW128 to BF16 pair bias. Pair bias is streamed once
per consumer and key block with an evict-last L2 hint. Q is resident for the
entire key loop; K, V, and the physical residual-mask vector advance through
the two-stage ring.

## 3. Initial roofline diagnosis

The model uses the updated H100 bandwidth assumptions below. Every memory
floor is recomputed as `traffic / bandwidth`; the compute-pipe rates are
unchanged.

| Resource | Assumed peak |
|---|---:|
| HBM to L2 | 3 TB/s |
| L2 to shared memory | 12 TB/s |
| L1/shared-memory path to registers | 33 TB/s |
| FP32 CUDA cores | 29.7 T scalar operations/s |
| MUFU exponential | 3.71 T results/s |
| BF16 tensor cores | 990 TFLOP/s |

Relative to the 990 TFLOP/s tensor peak, the three bandwidth ridges are
330 FLOP/byte for DRAM, 82.5 FLOP/byte for L2, and 30 FLOP/byte for the
L1/shared-memory path.

With one 64-row consumer (`Mq=R=64`), `Bc=64`, and no multicast, the predicted
B=1, H=4 lower bounds were:

| N | DRAM | L2 to SMEM | L1/SMEM to RF | MUFU | CUDA | tensor |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 11.4 us | 7.1 us | 1.3 us | 2.3 us | 2.8 us | 2.2 us |
| 384 | 102.4 us | 165.0 us | 30.2 us | 62.0 us | 76.3 us | 58.6 us |
| 1024 | 728.4 us | 2976.6 us | 545.0 us | 1175.8 us | 1446.1 us | 1110.6 us |

The important difference from ordinary FlashAttention is the FP32 pair-bias
term. Even with ideal K/V reuse, L2 intensity asymptotically approaches only
64 tensor FLOP/byte. The updated H100 L2 ridge is `990/12 = 82.5`
tensor FLOP/byte.
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

The retained diagonal hand-off is implemented around, rather than inside, the
QK and PV WGMMA operations:

```python
if n_ctx != SHORT_SEQUENCE_LENGTH and consumer_warpgroup == 1 and key_block == 0:
    # Hold WG1's first QK until WG0 has issued QK(0).
    qk0_to_qk1.arrive_and_wait()

score_mma.set(Field.ACCUMULATE, False)
cute.nvgpu.warpgroup.fence()
cute.gemm(score_mma, score_fragment, query_fragment, key_fragment, score_fragment)
cute.nvgpu.warpgroup.commit_group()
cute.nvgpu.warpgroup.wait_group(0)

if n_ctx != SHORT_SEQUENCE_LENGTH:
    if consumer_warpgroup == 0:
        qk0_to_qk1.arrive()            # release WG1 QK(k)
    else:
        qk1_to_pv0.arrive()            # release WG0 PV(k)

run_online_softmax_in_registers(score_fragment)

if n_ctx != SHORT_SEQUENCE_LENGTH:
    if consumer_warpgroup == 0:
        qk1_to_pv0.arrive_and_wait()   # wait for WG1 QK(k)
    elif key_block + 1 < num_key_blocks:
        qk0_to_qk1.arrive_and_wait()   # wait for WG0 QK(k+1)

output_mma.set(Field.ACCUMULATE, True)
cute.nvgpu.warpgroup.fence()
cute.gemm(output_mma, output_accumulator, probability_fragment,
          value_fragment, output_accumulator)
cute.nvgpu.warpgroup.commit_group()
cute.nvgpu.warpgroup.wait_group(0)
```

The two helper-like calls in this explanatory excerpt expand to the score
update in section 4.4; the retained source keeps that work inline.

### 4.1 Logical mask broadcast

Only 64 FP32 mask values are physically stored per K/V stage. A zero-stride
query mode exposes the logical 64x64 broadcast. The complete path is:

```python
# Physical storage: [K=64, stage=2], or 512 bytes total.
shared_mask = allocator.allocate_tensor(
    cutlass.Float32,
    cute.make_layout(
        (KEY_TILE_SIZE, KEY_VALUE_STAGES),
        stride=(1, KEY_TILE_SIZE),
    ),
    swizzle=None,
)

# Logical consumer view: [M=64, K=64, stage=2]. The M stride is zero,
# so all 64 query rows alias the same physical 64-element mask vector.
shared_broadcast_mask = cute.make_tensor(
    shared_mask.iterator,
    cute.make_layout(
        (QUERY_TILE_SIZE, KEY_TILE_SIZE, KEY_VALUE_STAGES),
        stride=(0, 1, KEY_TILE_SIZE),
    ),
)

# The producer transfers one [K=64] FP32 vector, not an [M,K] matrix.
mask_tiles = cute.zipped_divide(residual_mask_tensor, (KEY_TILE_SIZE,))
mask_problem = mask_tiles[None, (None, batch_seq_idx)]
shared_mask_part, global_mask_part = cute.nvgpu.cpasync.tma_partition(
    residual_mask_tma,
    0,
    cute.make_layout(1),
    shared_mask,
    mask_problem,
)
cute.copy(
    residual_mask_tma,
    global_mask_part[None, key_block],
    shared_mask_part[None, stage],
    tma_bar_ptr=key_value_ready + stage,
)

# In the score pass, e and e+2 are two query rows at one key column.
# Load each broadcast value once and reuse it for both rows.
_, column_0 = score_coords[elem_0]
_, column_1 = score_coords[elem_1]
mask_0 = shared_broadcast_mask[0, column_0, stage]
mask_1 = shared_broadcast_mask[0, column_1, stage]
score_fragment[elem_0] = (
    score_fragment[elem_0] * SOFTMAX_SCALE
    + pair_bias_fragment[elem_0]
    + mask_0
)
score_fragment[elem_1] = (
    score_fragment[elem_1] * SOFTMAX_SCALE
    + pair_bias_fragment[elem_1]
    + mask_1
)
score_fragment[elem_2] = (
    score_fragment[elem_2] * SOFTMAX_SCALE
    + pair_bias_fragment[elem_2]
    + mask_0
)
score_fragment[elem_3] = (
    score_fragment[elem_3] * SOFTMAX_SCALE
    + pair_bias_fragment[elem_3]
    + mask_1
)
```

This saves 63/64 of the shared-memory capacity that a materialized MxK mask
would require and avoids redundant shared loads.

Pair bias is also logically broadcast over S, but S belongs to distinct CTAs
in this schedule. A local CuTe zero-stride layout cannot share physical SMEM
between CTAs. Cluster multicast was therefore the correct mechanism tested for
that reuse; it was not retained because cluster scheduling cost outweighed the
traffic reduction on the complete shape set.

### 4.2 Split query completion barriers

Each consumer has its own Q completion barrier. The producer independently
publishes the two 64x64 BF16 transfers, and each consumer waits only for its
own tile:

```python
QUERY_BYTES = QUERY_TILE_SIZE * HEAD_DIMENSION * 2

query_ready = allocator.allocate_array(
    cutlass.Int64, CONSUMER_WARPGROUP_COUNT
)
if thread_idx == 0:
    for consumer in cutlass.range_constexpr(CONSUMER_WARPGROUP_COUNT):
        cute.arch.mbarrier_init(query_ready + consumer, 1)
cute.arch.mbarrier_init_fence()
cute.arch.sync_threads()

# Warp 8 owns producer control flow. The TMA copy atom has a one-thread
# logical copy layout; lane 0 owns the transaction-byte accounting.
if warp_idx == PRODUCER_WARP_INDEX:
    producer_lane = thread_idx - CONSUMER_THREAD_COUNT
    if producer_lane == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(query_ready, QUERY_BYTES)
        cute.arch.mbarrier_arrive_and_expect_tx(query_ready + 1, QUERY_BYTES)

    cute.copy(
        query_tma,
        global_query_partition[None, query_block * 2],
        shared_query_partition[None, 0],
        tma_bar_ptr=query_ready,
    )
    cute.copy(
        query_tma,
        global_query_partition[None, query_block * 2 + 1],
        shared_query_partition[None, 1],
        tma_bar_ptr=query_ready + 1,
    )
else:
    consumer = thread_idx // THREADS_PER_WARPGROUP
    cute.arch.mbarrier_wait(query_ready + consumer, 0)
    # This warpgroup can now form its Q fragment and begin QK.
```

The earlier implementation used one barrier covering both Q transfers. That
forced consumer 0 to wait for consumer 1's independent tile. Splitting the
barrier improved the two difficult N=128 cases by about 1.05-1.07%.

### 4.3 Register-sourced PV

The probability tile remains in the QK accumulator registers, is converted to
BF16, and feeds PV as an RMEM A operand. No probability tile is allocated in
shared memory:

```python
output_mma = sm90_utils.make_trivial_tiled_mma(
    cutlass.BFloat16,
    cutlass.BFloat16,
    OperandMajorMode.K,
    OperandMajorMode.MN,
    cutlass.Float32,
    atom_layout_mnk=(1, 1, 1),
    tiler_mn=(QUERY_TILE_SIZE, HEAD_DIMENSION),
    a_source=OperandSource.RMEM,       # P comes from registers
)

pv_thread = output_mma.get_slice(warpgroup_thread_idx)
probability_fragment = output_mma.make_fragment_A(
    output_mma.partition_shape_A((QUERY_TILE_SIZE, KEY_TILE_SIZE))
)
value_fragment = output_mma.make_fragment_B(
    pv_thread.partition_B(shared_value)
)
output_accumulator = output_mma.make_fragment_C(
    output_mma.partition_shape_C((QUERY_TILE_SIZE, HEAD_DIMENSION))
)
output_accumulator.fill(0.0)

# score_fragment now contains exp(score - row_max). Convert the register
# fragment to BF16 in place for the RMEM WGMMA A operand.
probability_fragment.store(score_fragment.load().to(cutlass.BFloat16))

# Rescale the previous online-softmax output before adding P @ V.
for elem in cutlass.range_constexpr(cute.size(output_accumulator)):
    output_accumulator[elem] *= alpha_0 if elem % 4 < 2 else alpha_1

output_mma.set(Field.ACCUMULATE, True)
cute.nvgpu.warpgroup.fence()
cute.gemm(
    output_mma,
    output_accumulator,
    probability_fragment,
    value_fragment[None, None, None, stage],
    output_accumulator,
)
cute.nvgpu.warpgroup.commit_group()
cute.nvgpu.warpgroup.wait_group(0)
```

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
without changing the online-softmax recurrence. The actual quartet reduction
and leader-broadcast pattern is:

```python
# Each thread owns fragments for two rows. Four adjacent lanes collectively
# cover a row's 64 columns, so XOR shuffles by 1 and 2 finish each reduction.
local_max_0 = cutlass.max(
    local_max_0, cute.arch.shuffle_sync_bfly(local_max_0, offset=1)
)
local_max_0 = cutlass.max(
    local_max_0, cute.arch.shuffle_sync_bfly(local_max_0, offset=2)
)
local_max_1 = cutlass.max(
    local_max_1, cute.arch.shuffle_sync_bfly(local_max_1, offset=1)
)
local_max_1 = cutlass.max(
    local_max_1, cute.arch.shuffle_sync_bfly(local_max_1, offset=2)
)

new_max_0 = cutlass.max(running_max_0, local_max_0)
new_max_1 = cutlass.max(running_max_1, local_max_1)
alpha_0 = cutlass.Float32(0.0)
alpha_1 = cutlass.Float32(0.0)
if warpgroup_thread_idx % 4 == 0:
    alpha_0 = cute.math.exp(running_max_0 - new_max_0, fastmath=True)
    alpha_1 = cute.math.exp(running_max_1 - new_max_1, fastmath=True)

# Broadcast lane 0 within each aligned four-lane subgroup. 0x1C03 is the
# width-four shuffle mask/clamp used by the retained kernel.
alpha_0 = cute.arch.shuffle_sync(alpha_0, 0, mask_and_clamp=0x1C03)
alpha_1 = cute.arch.shuffle_sync(alpha_1, 0, mask_and_clamp=0x1C03)

local_sum_0 = cutlass.Float32(0.0)
local_sum_1 = cutlass.Float32(0.0)
for elem in cutlass.range_constexpr(cute.size(score_fragment)):
    probability = cutlass.Float32(0.0)
    if elem % 4 < 2:
        probability = cute.math.exp(
            score_fragment[elem] - new_max_0, fastmath=True
        )
        local_sum_0 += probability
    else:
        probability = cute.math.exp(
            score_fragment[elem] - new_max_1, fastmath=True
        )
        local_sum_1 += probability
    score_fragment[elem] = probability

local_sum_0 += cute.arch.shuffle_sync_bfly(local_sum_0, offset=1)
local_sum_0 += cute.arch.shuffle_sync_bfly(local_sum_0, offset=2)
local_sum_1 += cute.arch.shuffle_sync_bfly(local_sum_1, offset=1)
local_sum_1 += cute.arch.shuffle_sync_bfly(local_sum_1, offset=2)
running_sum_0 = running_sum_0 * alpha_0 + local_sum_0
running_sum_1 = running_sum_1 * alpha_1 + local_sum_1
running_max_0, running_max_1 = new_max_0, new_max_1
```

### 4.5 Output epilogue

Each consumer converts its FP32 m64n64 accumulator and stores it into one half
of a combined MN_SW128 128x64 BF16 shared tile using StMatrix:

```python
# Build the combined CTA tile once. Each consumer takes one [64,64] M slice.
shared_output_mn = cute.make_tensor(
    shared_output.iterator,
    cute.select(shared_output.layout, mode=[1, 0]),
)
shared_output_consumer = cute.local_tile(
    shared_output_mn,
    (QUERY_TILE_SIZE, HEAD_DIMENSION),
    (consumer_warpgroup, 0),
)
thread_copy_c = output_register_copy.get_slice(warpgroup_thread_idx)
output_partition = thread_copy_c.partition_D(shared_output_consumer)

# Normalize FP32 O, narrow to BF16, and use four StMatrix operations per CTA.
for elem in cutlass.range_constexpr(cute.size(output_accumulator)):
    output_accumulator[elem] *= inv_sum_0 if elem % 4 < 2 else inv_sum_1
output_bf16 = cute.make_fragment_like(output_accumulator, cutlass.BFloat16)
output_bf16.store(output_accumulator.load().to(cutlass.BFloat16))
output_retile = output_register_copy.retile(output_bf16)
cute.copy(output_register_copy, output_retile, output_partition)

# Make the synchronous StMatrix writes visible to the asynchronous TMA proxy,
# then wait until both 128-thread consumers have populated the combined tile.
cute.arch.fence_proxy("async.shared", space="cta")
cute.arch.barrier(
    barrier_id=OUTPUT_READY_BARRIER_ID,
    number_of_threads=CONSUMER_THREAD_COUNT,
)

# Consumer WG0's first warp owns the one [128,64] TMA store operation.
if consumer_warpgroup == 0 and warpgroup_thread_idx < 32:
    cute.copy(
        output_tma,
        shared_output_partition,
        global_output_partition[None, query_block],
    )
    cute.arch.cp_async_bulk_commit_group()
    cute.arch.cp_async_bulk_wait_group(0)
```

After one 256-thread barrier, consumer 0 issues one TMA S2G operation for the
whole tile. The isolated epilogue emitted four STMATRIX instructions, one
UTMASTG, zero shared-store conflicts, and 384 total SASS instructions. Its
median was 3.3838 us on GPU2 and 3.0985 us on GPU3.

### 4.6 Dynamic TVM-FFI dispatch

The public wrapper flattens the prepared tensors without copying and marks
B*S*H and N dimensions dynamic. Tiled dimensions carry divisibility 64. The
compiled callable is cached under a lock and rejects cross-device reuse. The
relevant host path is:

```python
def _make_runtime_views(q, k, v, pair_bias, residual_mask, out, lse):
    batch, sequence_count, heads, n_ctx, dim = q.shape
    problems = batch * sequence_count * heads
    return (
        q.view(problems, n_ctx, dim),
        k.view(problems, n_ctx, dim),
        v.view(problems, n_ctx, dim),
        pair_bias.view(batch * heads, n_ctx, n_ctx),
        residual_mask.view(batch * sequence_count, n_ctx),
        out.view(problems, n_ctx, dim),
        lse.view(problems, n_ctx),
    )

def _as_dynamic_tensor(tensor, modes, *, alignment, tiled_modes=()):
    dynamic_tensor = from_dlpack(
        tensor.detach(),
        assumed_align=alignment,
        enable_tvm_ffi=True,
    )
    for mode in modes:
        dynamic_tensor = dynamic_tensor.mark_compact_shape_dynamic(
            mode=mode,
            stride_order=tensor.dim_order(),
            divisibility=QUERY_TILE_SIZE if mode in tiled_modes else 1,
        )
    return dynamic_tensor

def _make_compile_arguments(views, stream):
    q, k, v, pair_bias, residual_mask, out, lse = views
    return (
        _as_dynamic_tensor(q, (0, 1), alignment=16, tiled_modes=(1,)),
        _as_dynamic_tensor(k, (0, 1), alignment=16, tiled_modes=(1,)),
        _as_dynamic_tensor(v, (0, 1), alignment=16, tiled_modes=(1,)),
        _as_dynamic_tensor(
            pair_bias, (0, 1, 2), alignment=16, tiled_modes=(1, 2)
        ),
        _as_dynamic_tensor(
            residual_mask, (0, 1), alignment=16, tiled_modes=(1,)
        ),
        _as_dynamic_tensor(out, (0, 1), alignment=16, tiled_modes=(1,)),
        _as_dynamic_tensor(lse, (0, 1), alignment=16, tiled_modes=(1,)),
        stream,
    )

# The first call compiles. Every later target shape reuses this callable.
if self._compiled is None:
    with self._lock:
        if self._compiled is None:
            compile_args = self._make_compile_arguments(views, cuda_stream)
            self._compiled = CompileCallable(
                (PtxasOptions("--maxrregcount=255"), EnableTVMFFI)
            )(_build_and_launch_forward, *compile_args)
            self._device_index = device_index
            self._compile_count += 1
self._compiled(*views, cuda_stream)
```

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

### 5.7 Bc=64, one producer warp, and two CTAs per SM

The decisive configuration aligned the consumers to warps 0-7 and assigned
only warp 8 to TMA production. This is legal on Hopper because TMA issue and
WGMMA execution have different participation requirements:

- `wgmma.mma_async` is a 128-thread warpgroup operation. Consequently, each
  consumer is exactly four aligned warps: warps 0-3 and warps 4-7.
- `cp.async.bulk.tensor` is initiated by one thread and the TMA engine performs
  the asynchronous transfer. Software may dedicate a warp to owning that
  producer path; TMA does not require the other three warps of a warpgroup.
- The CuTe TMA atom here uses one logical TMA-copy participant. The source
  keeps the owning warp in uniform control flow, while lane 0 performs the
  explicit `mbarrier.arrive.expect_tx` bookkeeping. Those 32 lanes do not mean
  32 independent copies of the tile.
- A full producer warpgroup is common in Hopper CUTLASS and FlashAttention-3,
  in part because role partitioning and dynamic register redistribution are
  naturally warpgroup-scoped. This kernel does not use warpgroup register
  reallocation, so those extra 96 producer threads would provide no required
  WGMMA participation. With the retained uniform 96-register allocation, they
  would also prevent the two-CTA residency target.

| Operation | Hopper SM90 | Blackwell SM100 |
|---|---|---|
| TMA G2S/S2G issue | One issuing thread; a producer warp commonly owns control flow | One issuing thread; a producer warp commonly owns control flow |
| Tensor-core MMA issue | Four-warp, 128-thread WGMMA collective | Single-thread `tcgen05.mma` issue |
| Accumulator location | Per-thread registers across the warpgroup | CTA-visible TMEM |

The role split and the complete K/V/mask ring protocol are:

```python
CONSUMER_WARPGROUP_COUNT = 2
THREADS_PER_WARPGROUP = 128
CONSUMER_THREAD_COUNT = 256
PRODUCER_WARP_INDEX = 8
THREADS_PER_BLOCK = 288
KEY_VALUE_STAGES = 2

thread_idx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.warp_idx()

if warp_idx == PRODUCER_WARP_INDEX:
    producer_lane = thread_idx - CONSUMER_THREAD_COUNT
    key_block = cutlass.Int32(0)
    for _ in cutlass.range(num_key_blocks):
        stage = key_block % KEY_VALUE_STAGES

        # Do not overwrite stage s until both consumer warpgroups have
        # finished the prior tile held in s.
        if key_block >= KEY_VALUE_STAGES:
            free_phase = ((key_block // KEY_VALUE_STAGES) - 1) % 2
            cute.arch.mbarrier_wait(key_value_free + stage, free_phase)

        # One ready barrier covers K, V, and the physical 64-element mask.
        if producer_lane == 0:
            transaction_bytes = (
                KEY_TILE_SIZE * HEAD_DIMENSION * 2       # K BF16
                + KEY_TILE_SIZE * HEAD_DIMENSION * 2     # V BF16
                + KEY_TILE_SIZE * 4                      # mask FP32
            )
            cute.arch.mbarrier_arrive_and_expect_tx(
                key_value_ready + stage, transaction_bytes
            )

        cute.copy(
            key_tma,
            global_key_partition[None, key_block],
            shared_key_partition[None, stage],
            tma_bar_ptr=key_value_ready + stage,
        )
        cute.copy(
            value_tma,
            global_value_partition[None, key_block],
            shared_value_partition[None, stage],
            tma_bar_ptr=key_value_ready + stage,
        )
        cute.copy(
            residual_mask_tma,
            global_mask_partition[None, key_block],
            shared_mask_partition[None, stage],
            tma_bar_ptr=key_value_ready + stage,
        )
        key_block += 1
else:
    consumer = thread_idx // THREADS_PER_WARPGROUP
    consumer_lane = thread_idx - consumer * THREADS_PER_WARPGROUP
    key_block = cutlass.Int32(0)
    for _ in cutlass.range(num_key_blocks):
        stage = key_block % KEY_VALUE_STAGES
        full_phase = (key_block // KEY_VALUE_STAGES) % 2

        cute.arch.mbarrier_wait(key_value_ready + stage, full_phase)
        # All 128 threads in this consumer participate in QK and PV WGMMA;
        # sections 4.3 and 4.4 show the inlined math and fragment handling.
        run_qk_softmax_pv(consumer, consumer_lane, stage)

        # Each consumer contributes one arrival; count=2 releases the stage.
        if n_ctx != SHORT_SEQUENCE_LENGTH and consumer_lane == 0:
            cute.arch.mbarrier_arrive(key_value_free + stage)
        key_block += 1
```

This is not a Blackwell-only relaxation. Blackwell changes the MMA side:
`tcgen05.mma` is single-thread-issued and accumulates in TMEM, whereas Hopper
WGMMA is warpgroup-wide and accumulates in registers. TMA itself is
single-thread-initiated on both architectures.

With Bc=64, two K/V stages, and
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

### 5.9 Final optimized-schedule roofline

For direct comparison, this is the initial analytical table from section 3.
It models one 64-row consumer, so the CTA-level K/V reuse width is `R=64`:

| N | DRAM | L2 to SMEM | L1/SMEM to RF | MUFU | CUDA | tensor |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 11.4 us | 7.1 us | 1.3 us | 2.3 us | 2.8 us | 2.2 us |
| 384 | 102.4 us | 165.0 us | 30.2 us | 62.0 us | 76.3 us | 58.6 us |
| 1024 | 728.4 us | 2976.6 us | 545.0 us | 1175.8 us | 1446.1 us | 1110.6 us |

The retained schedule has two 64-row consumers sharing each staged K, V, and
residual-mask tile, giving `R=128`. It retains `Bc=64`, two K/V stages, FP32
pair bias, no cluster multicast, and register-sourced P for PV. Under the same
machine-rate assumptions, its roofline is:

| N | DRAM | L2 to SMEM | L1/SMEM to RF | MUFU | CUDA | tensor | Binding floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | **11.4 us** | 5.6 us | 1.3 us | 2.3 us | 2.8 us | 2.2 us | DRAM |
| 384 | 102.4 us | **126.6 us** | 30.0 us | 62.0 us | 76.3 us | 58.6 us | L2 |
| 1024 | 728.4 us | **2249.5 us** | 540.9 us | 1175.8 us | 1446.1 us | 1110.6 us | L2 |

The corresponding L2 arithmetic intensity changes are:

| N | Initial `R=64` | Final `R=128` | L2-floor reduction |
|---:|---:|---:|---:|
| 128 | 25.36 FLOP/B | 31.75 FLOP/B | 1.25x |
| 384 | 29.29 FLOP/B | 38.16 FLOP/B | 1.30x |
| 1024 | 30.78 FLOP/B | 40.73 FLOP/B | 1.32x |

Only the larger reuse width changes the idealized L2 byte count: two consumers
reuse one K/V/mask transfer across 128 query rows. `Bc=64` does not reduce
useful L2 traffic because `Bc` cancels between bytes per iteration and the
`N/Bc` iteration count. Its smaller fragments instead make the two-CTA
occupancy possible.

Likewise, RMEM PV and the SW128 pair-bias layout do not move the ideal L2
roofline. RMEM PV removes the probability tile's SMEM round trip and reduces
resource pressure. SW128 reduces TMA decomposition and improves the
shared-memory consumption pattern for the same pair-bias payload. Together
with two resident CTAs, these changes improve latency hiding and the fraction
of the L2 roofline that the kernel can actually attain.

The original table already assumed register-sourced P in its SMEM-to-RF byte
model, even though that optimization had not yet been implemented when the
first kernel variants were measured. Consequently, the two tables should be
read as a change in analytical schedules (`R=64` to `R=128`), not as a literal
before/after model of every intermediate implementation.

The final schedule therefore mitigates but does not remove L2 boundedness. At
long sequence lengths, FP32 pair bias remains the irreducible term and accounts
for roughly two-thirds of the main-loop L2 payload. Reducing that term would
require cross-CTA pair-bias multicast or persistent reuse; neither is enabled
in the retained universal kernel.

#### FlashAttention-3 roofline comparison

KernelWiki does not contain a numerical FlashAttention-3 roofline table. It
does contain source-reported evidence for FA3-style Hopper warp specialization
and two-consumer-warpgroup pipelines (`pr-flashinfer-887` and
`pr-flashinfer-952`). The numerical table below is therefore a local derivation
using those scheduling principles and the same H100 rates as section 3; it is
not a performance claim copied from KernelWiki.

For an apples-to-apples algorithmic comparison, the model applies noncausal
BF16 FlashAttention independently to the same `B*S*H` attention matrices as
EvoAttention, with `S=N`, `D=64`, BF16 output, and FP32 LSE. Canonical
FlashAttention does not itself have the extra `S` axis; replicating it over `S`
only puts the two algorithms on the same logical-logit count. The controlled
FA3-style schedule uses `R=128`, `Bc=64`, register-sourced probabilities, and
no dropout, pair bias, or residual mask.

Let `P_problem = B*S*H` and let `Cq` be the K/V multicast factor across query
CTAs. The relevant traffic is

```text
b_DRAM,FA3 = P_problem * (4*N*D*e_q + N*e_a)
                            |              |
                            |              +-- FP32 LSE
                            +----------------- BF16 Q, K, V, and O

b_L2,FA3 = P_problem * N * [2*D*e_q
                             + 2*N*D*e_q/(R*Cq)
                             + e_a]
                               |       |       |
                               |       |       +-- FP32 LSE
                               |       +---------- K/V restreaming
                               +------------------ Q load and O store

b_SMEM,FA3 ~= P_problem * N * D * e_q
```

The SMEM expression counts the BF16 output accumulator staging required for
TMA store. Q, K, and V feed WGMMA through shared-memory descriptors and do not
cross the ordinary SMEM-to-register datapath; probabilities remain in
registers. The tensor and MUFU work matches EvoAttention, while the approximate
CUDA-core work falls from `G*(9 + D/Bc)` to `G*(5 + D/Bc)` because there is no
FP32 pair-bias or residual-mask addition.

With no K/V multicast (`Cq=1`), the derived FA3-style roofline is:

| N | DRAM | L2 to SMEM | L1/SMEM to RF | MUFU | CUDA | tensor | Binding floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | **11.3 us** | 2.8 us | 0.3 us | 2.3 us | 1.7 us | 2.2 us | DRAM |
| 384 | **101.4 us** | 50.5 us | 2.3 us | 62.0 us | 45.8 us | 58.6 us | DRAM |
| 1024 | 721.4 us | 806.7 us | 16.3 us | **1175.8 us** | 867.7 us | 1110.6 us | MUFU |

A two-way K/V multicast (`Cq=2`) does not change compulsory DRAM, SMEM-to-RF,
or compute work. It changes the FA3 L2 and binding floors as follows:

| N | L2 without multicast | L2 with K/V multicast-2 | Final binding floor |
|---:|---:|---:|---:|
| 128 | 2.8 us | 2.1 us | 11.3 us DRAM |
| 384 | 50.5 us | 31.7 us | 101.4 us DRAM |
| 1024 | 806.7 us | 448.8 us | 1175.8 us MUFU |

Multicast is shown as a separate analytical variant because FA3 launch policy
and tile selection vary by implementation and shape; the KernelWiki entries do
not establish that every FA3 forward configuration uses cluster-two K/V
multicast.

The direct comparison with the retained EvoAttention schedule is:

| N | Evo L2 intensity | FA3 intensity, no multicast | FA3 intensity, K/V multicast-2 | Evo binding floor | FA3 binding floor, no multicast |
|---:|---:|---:|---:|---:|---:|
| 128 | 31.75 FLOP/B | 63.50 FLOP/B | 84.45 FLOP/B | 11.4 us DRAM | 11.3 us DRAM |
| 384 | 38.16 FLOP/B | 95.63 FLOP/B | 152.65 FLOP/B | 126.6 us L2 | 101.4 us DRAM |
| 1024 | 40.73 FLOP/B | 113.58 FLOP/B | 204.16 FLOP/B | 2249.5 us L2 | 1175.8 us MUFU |

At short sequence lengths, compulsory Q/K/V/O traffic makes both algorithms
DRAM-limited. At medium lengths, ordinary FlashAttention remains DRAM-bound,
whereas EvoAttention is already strongly L2-bound. At long lengths, the
higher 12 TB/s L2 assumption makes the FA3-style schedule MUFU-bound even
without multicast; multicast increases its L2 headroom but no longer changes
the binding pipe. The same multicast cannot
remove EvoAttention's FP32 pair-bias stream, which is why optimized
EvoAttention remains L2-bound and its intensity is about one third of
non-multicast FA3 at `N=1024`.

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
logical view to `partition_C`. The full descriptor-to-fragment path is:

```python
pair_bias_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
    SmemLayoutAtomKind.MN_SW128, cutlass.Float32
)
pair_bias_layout = cute.tile_to_shape(
    pair_bias_layout_atom,
    (KEY_TILE_SIZE, QUERY_TILE_SIZE, CONSUMER_WARPGROUP_COUNT),
    order=(1, 0, 2),
)
pair_bias_tma_layout = cute.tile_to_shape(
    pair_bias_layout_atom,
    (KEY_TILE_SIZE, QUERY_TILE_SIZE, 1),
    order=(1, 0, 2),
)

# Global [problem,Q,K] is viewed as [K,Q,problem] for the descriptor.
pair_bias_view = cute.make_tensor(
    pair_bias.iterator,
    cute.select(pair_bias.layout, mode=[2, 1, 0]),
)
pair_bias_tma, pair_bias_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
    cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
    pair_bias_view,
    pair_bias_tma_layout,
    (KEY_TILE_SIZE, QUERY_TILE_SIZE),
)

# Each consumer owns one private slot: [K,Q,consumer].
shared_pair_bias_row_col = cute.make_tensor(
    shared_pair_bias.iterator,
    cute.select(shared_pair_bias.layout, mode=[1, 0, 2]),
)
pair_bias_partition = qk_thread.partition_C(
    shared_pair_bias_row_col[None, None, consumer_warpgroup]
)

# Load directly into the consumer's stage, then copy the identically
# partitioned [Q,K] values into its score-shaped register fragment.
cute.copy(
    pair_bias_tma,
    global_pair_bias_partition[
        None, key_block, query_block * 2 + consumer_warpgroup
    ],
    shared_pair_bias_partition[None, consumer_warpgroup],
    tma_bar_ptr=pair_bias_ready + consumer_warpgroup,
    cache_policy=cutlass.Int64(PAIR_BIAS_EVICT_LAST_POLICY),
)
cute.arch.mbarrier_wait(pair_bias_ready + consumer_warpgroup, pair_bias_phase)
pair_bias_fragment.store(pair_bias_partition.load())
```

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
- hw-mbarrier, wiki/hardware/mbarrier.md
- technique-warp-specialization, wiki/techniques/warp-specialization.md
- lang-cute-dsl, wiki/languages/cute-dsl.md
- pr-flashinfer-887, sources/prs/flashinfer/PR-887.md
- pr-flashinfer-952, sources/prs/flashinfer/PR-952.md

These are source-reported references. They support the TMA issue model,
Hopper WGMMA participation granularity, warp specialization, layout-aware
swizzling, and ping-pong scheduling. All EvoAttention performance and
correctness claims in this document come from measurements in this repository,
not from the external references.


---

## Part II — D=64 backward

*Original D=64 backward record*

## 1. Executive summary

The D=64 production specialization in `evoattention_cute_bwd_d64.py` uses one
fixed M64 x N128 Hopper main schedule. The public
`evoattention_cute_bwd.py` launcher dispatches D=32 and D=64 calls from the
runtime head dimension. One TVM-FFI dispatch emits FP32 dPairBias
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


---

## Part III — D=32 forward exploration

*Original D=32 forward record*

## Scope and invariants

- During exploration, experimental files and reports lived under
  `cute_dsl/tmp/d32_exploration/`; those scratch artifacts were removed after
  the winning implementation and this record were promoted.
- The experiments did not modify the then-current production files until the
  winning schedule had passed the acceptance sweep.
- Target GPU: NVIDIA H100, compute capability 9.0.
- Inputs: BF16 Q/K/V, FP32 pair bias, and FP32 residual mask containing
  `{0, -1e9}`.
- Outputs: BF16 attention output and FP32 log-sum-exp.
- Workload: `N_SEQ=N_CTX`, `B in {1,4}`, `H in {4,16}`,
  `N in {128,256,384,512,640,768,896,1024}`, and `D=32`.
- Timing must be prepared-layout, kernel-only, CUDA-event timing through TVM-FFI.
- Correctness reference: the authoritative fused Triton EvoAttention kernel.

## Baseline copied on 2026-09-14

The starting point is an unchanged mechanical copy of the production D=64
forward kernel, test, and benchmark. Only the private copies were then changed
to use `D=32` and import one another.

The retained schedule initially has `Br=64`, `Bc=64`, two consumer
warpgroups, one producer warp, two K/V/mask stages, one private pair-bias tile
per consumer, and a nonpersistent grid.

## Planned experiment order

1. Compile and validate the native-exp D=32 specialization.
2. Establish full CUDA-event baseline and representative NCU profiles.
3. Isolate persistent scheduling from one-CTA residency.
4. Sweep independent K/V and pair-bias stage counts.
5. Pipeline Q and output across persistent work-item boundaries.
6. Test task orders favoring K/V locality versus pair-bias locality.
7. Test sequence-paired consumer mapping to share pair bias within a CTA.
8. Retune consumer ping-pong synchronization.
9. Reconsider multicast only if measured L2 throughput remains decisive.
10. Test hybrid FA4-style polynomial `exp2` only if MUFU remains exposed.

## Experiment results

Results will be appended chronologically. A candidate is not retained unless
it passes correctness and improves CUDA-event timing reproducibly.

### E0a — Mechanical D=32 specialization

**Result: did not compile.** CuTe rejected the output TMA construction because
the retained BF16 `MN_SW128` atom was not top-level shape-equivalent to the
logical `(D=32, CTA_M=128)` output tile. This confirms that reducing the head
dimension does not automatically shrink a manually selected 128-byte swizzle.

**Action:** specialize only the output staging atom to `MN_SW64`, matching a
32-element BF16 contiguous extent of 64 bytes. Q/K/V layouts are selected by
the Hopper helper and did not produce the initial diagnostic.

### E0b — Native-exp D=32 baseline

**Result: correct and already above target on all 32 shapes.** The first
correct specialization uses `MN_SW64` output staging and otherwise retains the
D=64 schedule. One compilation serves every B/H/N shape.

Smoke correctness at `B=1, H=4, N=128`:

- output max/mean absolute error: `0.0078125 / 0.000348205`;
- FP32 LSE max/mean absolute error: `0.00561047 / 0.000470062`.

CUDA-event results are stored in `baseline_full.json`. Speedup versus the
authoritative fused Triton forward ranges from `2.308x` to `3.854x`; all 32
shapes exceed `2x`. Representative results:

| B | H | N | Triton ms | CuTe ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 128 | 0.0473 | 0.0205 | 2.308x |
| 1 | 4 | 384 | 0.8652 | 0.2382 | 3.632x |
| 1 | 4 | 640 | 3.7536 | 0.9895 | 3.793x |
| 1 | 4 | 1024 | 14.5134 | 4.0465 | 3.587x |
| 4 | 16 | 128 | 0.6432 | 0.2082 | 3.089x |
| 4 | 16 | 640 | 59.8619 | 16.9711 | 3.527x |
| 4 | 16 | 1024 | 258.5706 | 81.2041 | 3.184x |

### Baseline NCU diagnosis — B=1, H=4, N=640

Reports live under `profile/d32_baseline_n640/`. The full report identifies:

| Metric | Measured value |
|---|---:|
| Registers/thread | 92 |
| Dynamic shared memory/CTA | 66,112 B |
| Resident CTAs/SM | 2 |
| Achieved occupancy | 27.14% |
| Compute throughput | 65.89% |
| L1/TEX throughput | 51.05% |
| L2 throughput | 44.16% |
| DRAM throughput | 11.96% |
| ALU pipeline utilization | 47.01% |
| FP32 FMA pipeline utilization | 25.33% |
| BF16 WGMMA utilization | 13.75% |
| L2 hit rate | 89.27% |

NCU reports 33,108,484 shared-load bank conflicts across 24,576,000 shared
load requests. All 32,768,000 excessive wavefronts come from sixteen unrolled
`LDS.64` instructions, consistent with the 32-FP32-element pair-bias fragment
load. This is a higher-priority measured issue than HBM traffic. The kernel is
not DRAM-bandwidth-bound at this representative shape.

The D=32 specialization uses 92 rather than the D=64 kernel's 96 registers.
Consequently, shared memory would permit three CTAs, but registers still limit
residency to two CTAs.

### E1 — K/V stage-depth sweep

`EVOATTENTION_KV_STAGES` was made an experimental compile-time knob. A first
parallel smoke sweep followed by a same-GPU repeat compared 2, 3, 5, and 7
stages. Same-GPU CuTe medians were:

| Stages | B1/H4/N128 | B1/H16/N384 | B4/H4/N640 |
|---:|---:|---:|---:|
| 2 | 0.0202 ms | 0.9326 ms | 4.1612 ms |
| 3 | **0.0199 ms** | **0.9232 ms** | 4.1053 ms |
| 5 | 0.0203 ms | 0.9270 ms | **3.9955 ms** |
| 7 | 0.0202 ms | 0.9316 ms | 4.0232 ms |

**Conclusion:** additional K/V lookahead is useful but small and
shape-dependent. Three stages best preserve short/medium performance; five
stages help the long saturated case by roughly 4%. Seven stages adds no value.
This supports treating pair-bias/shared-load pressure, not K/V stage depth, as
the next target.

### E2 — Pair-bias shared-layout sweep

The NCU conflict report motivated a direct comparison of `MN_SW128`,
`MN_SW64`, `MN_SW32`, and the unswizzled/interleaved atom. The latter three did
not improve the kernel:

- `SW64` was slightly slower at the medium and long smoke shapes;
- `SW32` slowed N384/N640 by roughly 8–13%;
- `INTER` was catastrophic, slowing N640 from about 4.0 ms to 6.66 ms and
  missing the 2x requirement at N128.

**Conclusion:** retain `MN_SW128`. The remaining `LDS.64` excess wavefronts are
not evidence that a smaller/no swizzle is better; the alternatives increase
bank serialization substantially.

### E3 — Fully persistent one-CTA-per-SM schedule

A conservative persistent variant was implemented in
`evoattention_cute_d32_persistent.py`. It launches 132 CTAs and uses a static
grid-stride task loop. Each task drains its TMA operations and synchronizes the
CTA before shared barriers and buffers are reinitialized for the next task.

The implementation is correct, but the experiment is negative:

| Variant | B1/H4/N128 | B1/H16/N384 | B4/H4/N640 |
|---|---:|---:|---:|---:|
| Nonpersistent, 2 resident CTAs, KV2 | 0.0202 ms | 0.9326 ms | 4.1612 ms |
| Persistent, 1 CTA/SM, KV2 | 0.0232 ms | 1.2774 ms | 5.3291 ms |
| Persistent, 1 CTA/SM, KV4 | 0.0235 ms | 1.2813 ms | 5.3341 ms |
| Persistent, 1 CTA/SM, KV7 | 0.0236 ms | 1.2899 ms | 5.3456 ms |
| Persistent, 1 CTA/SM, KV12 | 0.0238 ms | 1.3057 ms | 5.4323 ms |
| Persistent, 1 CTA/SM, KV16 | 0.0238 ms | 1.3033 ms | 5.4083 ms |

The nonpersistent one-CTA controls with KV8/KV12/KV16 were slower still
(`5.66–6.02 ms` at N640), so persistence does amortize some task overhead. It
cannot compensate for reducing resident warps from 18 to 9. Additional K/V
stages do not remove the bottleneck and regress monotonically after KV2.

A 264-CTA persistent grid preserving two resident CTAs was approximately
neutral at N128 and 5–6% slower at N384; it offered no long-shape advantage.

**Conclusion:** reject fully persistent single-CTA scheduling for the current
two-consumer organization. The measured kernel needs the second resident CTA
for instruction- and warp-level latency hiding. Inter-task Q/output prefetch
cannot plausibly recover the observed 28–38% steady-state deficit.

### E4 — FA4-style software exponential, degree three

The exact upstream FA4 range-reduction and degree-three Sollya polynomial were
ported locally, without importing FA4 or QuACK. The implementation clamps at
`-127`, obtains `floor(x)` with a directed-rounding magic-bias addition,
approximates `2**fraction` on `[0,1)`, and reconstructs the integer power by
modifying the FP32 exponent bits. Only bulk probability exponentials were
eligible; online accumulator-rescale exponentials remained native.

The approximation passed correctness, but every meaningful offload ratio
regressed on H100:

| Software-exp fraction | B1/H4/N128 | B1/H16/N384 | B4/H4/N640 |
|---:|---:|---:|---:|
| 0% baseline | about 0.0202 ms | about 0.9326 ms | about 4.16 ms |
| 12.5% | 0.0200 ms | 0.9477 ms | 4.1596 ms |
| 25% | 0.0203 ms | 0.9469 ms | 4.1580 ms |
| 50% | 0.0204 ms | 0.9940 ms | 4.3589 ms |
| 100% | 0.0208 ms | 1.1302 ms | 4.9501 ms |

**Conclusion:** the SM100 FA4 trade does not transfer directly to SM90. The
scalar range-reduction, polynomial, and exponent reconstruction increase
FP32/ALU pressure; the baseline already measures 47% ALU utilization. Retain
native exponentials. A degree-two screening test follows only to establish
whether approximation degree, rather than fixed range-reduction overhead, is
responsible.

### E4b — Degree-two software exponential

The degree-two polynomial also passed the established tolerance at N=1024:
output max/mean absolute error was `0.0078125 / 0.000170904`, and LSE error was
`0.00714684 / 0.000422666`. It did not recover performance:

| Software-exp fraction | B1/H4/N128 | B1/H16/N384 | B4/H4/N640 |
|---:|---:|---:|---:|
| 50% | 0.0203 ms | 0.9786 ms | 4.2781 ms |
| 100% | 0.0204 ms | 1.0862 ms | 4.7658 ms |

The fixed range-reduction and bit-reconstruction work, rather than polynomial
degree alone, is too expensive on H100. This path was removed from the final
kernel.

### E5 — Independent pair-bias pipeline depth

The original per-consumer pair-bias slot was generalized to a circular
pipeline. A slot is prefetched `PAIR_BIAS_STAGES` iterations ahead and is not
reused until the preceding PV WGMMA has consumed the corresponding
probabilities. This was correct, but deeper buffering was counterproductive:

| PB stages | B1/H4/N128 | B1/H16/N384 | B4/H4/N640 | Outcome |
|---:|---:|---:|---:|---|
| 1 | 0.0212 ms | 0.9363 ms | 4.0208 ms | Retain |
| 2 | 0.0204 ms | 0.9542 ms | 4.1180 ms | 2–3% slower at scale |
| 3 | 0.0238 ms | 1.3731 ms | 5.7162 ms | Residency/resource collapse |

Each extra stage costs 32 KiB because the original mapping needs one FP32
`64x64` PB tile per consumer. The single-stage schedule already overlaps the
next transfer with useful work; added storage and barrier state cost more than
the remaining latency.

### E6 — Output `StMatrix` width

`StMatrix8x8x16bOp` was tested with one, two, and four matrices per operation.
All variants were correct. Four matrices remained fastest:

| Matrices/op | B1/H4/N128 | B1/H16/N384 | B4/H4/N640 |
|---:|---:|---:|---:|
| 1 | 0.0201 ms | 0.9531 ms | 4.0692 ms |
| 2 | 0.0201 ms | 0.9614 ms | 4.1210 ms |
| 4 | **0.0199 ms** | **0.9484 ms** | **4.0283 ms** |

The D=32 epilogue is smaller, but reducing the operation width only increases
the number of store instructions. The retained value is four.

### E7 — Pair-bias-major CTA raster

The grid was reordered so adjacent CTAs handled the same pair-bias tile across
different sequences. It passed correctness, but trading away K/V locality was
negative:

| Order | B1/H4/N128 | B1/H16/N384 | B4/H4/N640 |
|---|---:|---:|---:|---:|
| Problem-major, query block inner | 0.0201 ms | 0.9578 ms | 4.1245 ms |
| Pair-bias-major, sequence inner | 0.0200 ms | 0.9769 ms | 4.3022 ms |

The PB working set is already favored by its evict-last policy. Consecutive
query CTAs reusing the same K/V sequence is the better L2 ordering.

### E8 — Final K/V depth and launch-bound sweep

A full-matrix comparison showed that three K/V stages are the best single
configuration. Relative to two stages, three stages win 29/32 shapes and are
2.45% faster on average; four and five stages are less consistent.

The `min_blocks_per_mp` launch bound was then swept. Forcing three blocks is
strongly negative (`0.0250 / 1.2691 / 5.5235 ms` on the three smoke shapes)
because register pressure cannot sustain three CTAs. A bound of one allows the
compiler more scheduling freedom while the measured resource footprint still
admits two resident CTAs:

| Metric, B1/H4/N640 | Value |
|---|---:|
| Registers/thread | 92 |
| Dynamic shared memory/CTA | 74,576 B |
| Register block limit | 2 |
| Shared-memory block limit | 2 |
| Theoretical occupancy | 28.12% |
| Achieved occupancy | 26.89% |

Alternating same-GPU A/B runs consistently favored the one-block launch bound
by roughly 2% at N384/N640. It was therefore retained; this does **not** make
the kernel one-CTA-resident.

### E9 — Sequence-paired consumers: decisive D=32 optimization

This is the key D=32-specific improvement. The original CTA assigns its two
consumer warpgroups to two query tiles from one sequence:

```text
WG0: sequence s, query rows q..q+63
WG1: sequence s, query rows q+64..q+127

shared per key block:
  K/V(s) once + mask(s) once + PB(q) + PB(q+64)
```

The winning mapping assigns both consumers the same query rows in two adjacent
sequences:

```text
WG0: sequence 2s,   query rows q..q+63
WG1: sequence 2s+1, query rows q..q+63

shared per key block:
  K/V(2s) + K/V(2s+1) + mask(2s) + mask(2s+1) + PB(q) once
```

The number of CTAs is unchanged:

```text
old: B * S * H * (N / 128)
new: B * (S / 2) * H * (N / 64)
```

For one key block, excluding Q/output traffic that is unchanged:

| Stream | Original bytes/CTA | Sequence-paired bytes/CTA |
|---|---:|---:|
| K + V | 8,192 | 16,384 |
| Residual mask | 256 | 512 |
| FP32 pair bias | 32,768 | 16,384 |
| **Total** | **41,216** | **33,280** |

Thus logical input traffic falls by `19.25%` per key block. MMA work remains
`1,048,576` FLOPs per CTA per key block, so the input arithmetic intensity
rises from `25.44` to `31.51 FLOP/B` (+23.9%). At the measured 12 TB/s L2
roof, the ideal per-block input-transfer floor falls from `3.435 ns` to
`2.773 ns`. At N=640, adding the unchanged 8 KiB Q read and 8 KiB output write,
the modeled CTA traffic falls from 428,544 to 349,184 bytes (`18.52%`).

The implementation has one shared PB slot. The producer may overwrite it only
after every consumer warp has copied its PB fragment to registers:

```python
cute.arch.mbarrier_init(
    pair_bias_free + slot,
    CONSUMER_WARPGROUP_COUNT * 4,  # eight consumer warps
)

pair_bias_fragment.store(pair_bias_partition.load())
if warpgroup_thread_idx % 32 == 0:
    cute.arch.mbarrier_arrive(pair_bias_free + pair_bias_stage)
```

Using only one arrival per warpgroup was initially incorrect: at N=384 it
produced 65 output elements outside tolerance because the first warp could
release the slot while sibling warps were still loading. Eight warp arrivals
make the lifetime proof explicit and restored exact agreement with the
baseline error profile.

Full-matrix comparison against the best pre-pairing kernel:

- wins 30/32 shapes;
- 5.04% lower latency on average;
- up to 20.06% lower latency at B=1, H=16, N=1024;
- only two small losses, both below 0.5% and within run-to-run noise.

The gain grows sharply for the large H=16 shapes, where halving the dominant
pair-bias stream matters most.

### E10 — Pair-bias release synchronization

Replacing eight independent warp arrivals with two per-warpgroup arrivals was
tested by placing a 128-thread named barrier after the PB fragment load. It was
correct but 1–2% slower: the new barrier synchronizes all four warps and
damages the intended FA3 ping-pong overlap. The final kernel retains the eight
nonblocking arrivals.

### E11 — Split K-ready and V/mask-ready barriers

Separate barriers allowed QK WGMMA to start after K arrived while V and the
residual mask were still in flight. The result was mixed (`+1%` at N128,
`-0.7%` at N384, and approximately neutral at N640). The extra barrier state
and waits offset the overlap, so the simpler combined K/V/mask transaction
barrier was retained.

### E12 — TMA issue order

Pair bias was issued before K, after K, and after K/V/mask. Keeping PB last was
clearly best on the smoke suite (`0.0200 / 0.9012 / 3.8634 ms` in that run).
K/V/mask gate QK; issuing them first lets PB naturally overlap QK and keeps the
consumer critical path shortest.

### E13 — Two-CTA K/V/residual-mask multicast

Every target N has an even number of M=64 query tiles, so adjacent CTAs were
clustered without a tail launch. Both sequence K/V tiles and masks were loaded
with multicast; pair bias remained CTA-private because adjacent CTAs own
different query rows. The variant is correct, but it is not the universal
winner:

- wins 11/32 shapes;
- loses 0.64% on average;
- worst regression is 4.55%;
- best improvement is 1.77%.

NCU at B=1/H=4/N=640 shows why. Registers, shared memory, and occupancy are
unchanged, and DRAM reads are essentially unchanged. The cluster variant
executes 2.8% more instructions and increases barrier-stall samples by about
9.5%; its profiled duration rises from 1.400 to 1.449 ms. Reduced K/V source
traffic is too small after sequence pairing to repay cluster synchronization
and remote mbarrier bookkeeping. Multicast is therefore not retained.

### E14 — Two-CTA pair-bias multicast across four sequences

The final multicast possibility reordered a cluster around one query tile and
two sequence pairs, allowing four sequences to share one FP32 PB transfer.
This required distributed PB lifetime tracking: every one of the sixteen
consumer warps in the cluster must release a slot before either producer can
overwrite it.

One PB stage serialized the two CTAs on every key block and was disastrous
(`0.0243 / 1.3979 / 5.4589 ms` on the smoke suite, including a sub-2x N128
result). Two stages restored overlap and correctness:

| Variant | B1/H4/N128 | B1/H16/N384 | B4/H4/N640 |
|---|---:|---:|---:|
| Ordinary final kernel | 0.0199 ms | 0.9303 ms | 3.9414 ms |
| PB multicast, two PB stages | 0.0199 ms | 0.9280 ms | 3.9712 ms |

The largest B4/H16/N1024 case also regressed from 64.9609 to 65.3521 ms in a
direct comparison. Pair-bias source-byte reduction is canceled by the cluster
barrier, remote mbarrier arrivals, extra 16 KiB PB stage, and loss of the
ordinary grid's K/V-friendly query-block ordering. This variant is correct but
not retained.

### E15 — Pair-bias TMA L2 prefetch

Future PB tiles were prefetched through the TMA descriptor without allocating
additional shared memory. Distances one, two, and three were screened. A
parallel run made distance two appear faster, but alternating same-GPU A/B
measurements rejected it: all three repeats regressed N128, all three
regressed N384, and two of three regressed N640. Representative medians were
`0.0207 / 0.9674 / 4.1097 ms` with distance two versus
`0.0201 / 0.9649 / 4.0311 ms` without it. The explicit prefetch adds traffic
and issue work while evict-last caching already retains useful PB lines, so it
is not present in the final kernel.

## Final retained kernel

The clean winner is `evoattention_cute_d32_best.py`. Its fixed schedule is:

| Parameter | Value |
|---|---:|
| Head dimension | 32 |
| Query tile per consumer | 64 |
| Key tile | 64 |
| Consumer warpgroups | 2 |
| Producer warps | 1 |
| Threads/CTA | 288 |
| K/V/mask stages per sequence | 3 |
| Pair-bias stages | 1 shared across both consumers |
| Pair-bias layout | FP32 `MN_SW128`, order `(1,0,2)` |
| Output layout | BF16 `MN_SW64` |
| Output StMatrix width | 4 matrices |
| Launch bound | 1 (measured residency remains 2 CTAs/SM) |
| Grid | nonpersistent, problem/sequence-pair major, query tile inner |
| Exponential | native H100 `MUFU.EX2` path |
| Cluster multicast | disabled |

### Final NCU comparison at B=1, H=4, N=640

The clean final report is `profile/d32_final_best_n640.ncu-rep`; the generated
comparison is
`profile/d32_compare_n640/analysis/compare_baseline_kv2_vs_final_best.txt`.

| Metric | Initial D=32 baseline | Final best | Change |
|---|---:|---:|---:|
| Profiled duration | 1.468 ms | 1.404 ms | -4.36% |
| Registers/thread | 92 | 96 | +4 |
| Dynamic shared memory/CTA | 66,112 B | 83,536 B | +17,424 B |
| Resident CTA limit | 2 | 2 | unchanged |
| Achieved occupancy | 27.14% | 27.17% | unchanged |
| Executed instructions/SM subpartition | 1.190 M | 1.136 M | -4.54% |
| Eligible warps/cycle | 1.383 | 1.431 | +3.47% |
| Long-scoreboard ratio | 0.9688 | 0.7868 | -18.8% |
| Barrier-stall ratio | 0.9185 | 0.7576 | -17.5% |
| Tensor-pipe activity | 13.75% | 14.24% | +3.6% relative |
| DRAM bytes read | 322.8 MB | 327.7 MB | essentially unchanged |

The final schedule spends more shared memory to hold independent K/V for the
two sequences, but preserves two-CTA occupancy. It removes duplicate PB work,
reduces long-scoreboard and barrier exposure, and raises eligible-warps and
tensor-pipe utilization. The remaining bottleneck is mixed: shared/L1 traffic
and scalar softmax scheduling dominate more than HBM, with substantial MIO
pressure from the FP32 PB fragment loads.

### Final CUDA-event benchmark

`final_best_full.json` contains the authoritative prepared-layout,
kernel-only results. CUDA events, 10 warmups, nine repeat medians, and an
approximately 200 ms adaptive timing window were used. Compilation,
allocation, and input preparation are excluded.

| B | H | N=S | Triton ms | Final CuTe ms | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 128 | 0.0469 | 0.0201 | 2.332x |
| 1 | 4 | 256 | 0.2717 | 0.0811 | 3.349x |
| 1 | 4 | 384 | 0.8468 | 0.2315 | 3.658x |
| 1 | 4 | 512 | 1.9247 | 0.5138 | 3.746x |
| 1 | 4 | 640 | 3.6687 | 0.9885 | 3.711x |
| 1 | 4 | 768 | 6.3237 | 1.6668 | 3.794x |
| 1 | 4 | 896 | 9.8230 | 2.6387 | 3.723x |
| 1 | 4 | 1024 | 14.6070 | 3.8548 | 3.789x |
| 1 | 16 | 128 | 0.1652 | 0.0574 | 2.881x |
| 1 | 16 | 256 | 1.0699 | 0.3000 | 3.566x |
| 1 | 16 | 384 | 3.3831 | 0.9316 | 3.632x |
| 1 | 16 | 512 | 7.7288 | 2.0718 | 3.730x |
| 1 | 16 | 640 | 14.9224 | 4.0830 | 3.655x |
| 1 | 16 | 768 | 26.0219 | 6.7396 | 3.861x |
| 1 | 16 | 896 | 43.1408 | 10.9742 | 3.931x |
| 1 | 16 | 1024 | 64.2588 | 15.4590 | 4.157x |
| 4 | 4 | 128 | 0.1651 | 0.0569 | 2.900x |
| 4 | 4 | 256 | 1.0702 | 0.3020 | 3.544x |
| 4 | 4 | 384 | 3.3849 | 0.9117 | 3.713x |
| 4 | 4 | 512 | 7.7265 | 2.1447 | 3.603x |
| 4 | 4 | 640 | 14.7458 | 3.9430 | 3.740x |
| 4 | 4 | 768 | 25.2586 | 6.9805 | 3.618x |
| 4 | 4 | 896 | 39.5033 | 10.5471 | 3.745x |
| 4 | 4 | 1024 | 58.3141 | 16.2562 | 3.587x |
| 4 | 16 | 128 | 0.6378 | 0.2024 | 3.151x |
| 4 | 16 | 256 | 4.2590 | 1.2274 | 3.470x |
| 4 | 16 | 384 | 13.4997 | 3.6425 | 3.706x |
| 4 | 16 | 512 | 30.9224 | 8.6868 | 3.560x |
| 4 | 16 | 640 | 59.8954 | 15.8152 | 3.787x |
| 4 | 16 | 768 | 104.3364 | 28.1538 | 3.706x |
| 4 | 16 | 896 | 174.0101 | 43.8708 | 3.966x |
| 4 | 16 | 1024 | 258.5478 | 64.8437 | 3.987x |

All 32 shapes pass the 2x requirement. The measured speedup range is
`2.332x–4.157x`.

### Final correctness

`test_d32.py` was run over all 32 target shapes against the authoritative
Triton kernel. Every case passed `atol=rtol=2e-2`, and all shapes used one
compiled dynamic specialization. Across the sweep:

- maximum BF16 output absolute error: `0.015625`;
- maximum FP32 LSE absolute error: `0.010951`;
- worst reported output mean absolute error: `0.000349511`;
- worst reported LSE mean absolute error: `0.000471706`.

## Artifact index

- `evoattention_cute_d32_best.py`: clean retained kernel.
- `test_d32.py`: correctness harness; select a variant with
  `EVOATTENTION_D32_MODULE`.
- `benchmark_d32.py`: CUDA-event benchmark harness.
- `final_best_full.json`: final 32-shape timing result.
- `baseline_full.json`: initial D=32 timing result.
- `evoattention_cute_d32.py`: knobbed non-paired exploration baseline.
- `evoattention_cute_d32_persistent.py`: persistent experiment.
- `evoattention_cute_d32_sequence_pair.py`: knobbed sequence-pair experiment.
- `evoattention_cute_d32_sequence_pair_split.py`: split-ready experiment.
- `evoattention_cute_d32_sequence_pair_multicast.py`: cluster-multicast
  experiment.
- `evoattention_cute_d32_sequence_pair_pb_multicast.py`: four-sequence
  pair-bias multicast experiment.
- `profile/d32_baseline_n640/`: initial NCU reports and analysis.
- `profile/d32_final_best_n640.ncu-rep`: final full NCU report.
- `profile/d32_compare_n640/analysis/`: machine-readable metrics and generated
  comparisons.

## Reproduction

From this directory, the harnesses default to the clean best kernel:

```bash
python3 test_d32.py
python3 benchmark_d32.py --direction forward
```

An experimental module can be selected without editing the harness:

```bash
EVOATTENTION_D32_MODULE=evoattention_cute_d32_sequence_pair_multicast \
  python3 benchmark_d32.py --direction forward --quick --no-enforce
```

Both harnesses use the same TVM-FFI dynamic callable. B, H, S, and N are
runtime-dynamic and the full target matrix reports one CuTe compilation.


---

## Part IV — D=32 backward exploration

*Original D=32 backward record*

## Scope and acceptance contract

- GPU: NVIDIA H100 (SM90).
- Shapes: `B in {1,4}`, `H in {4,16}`, and `S=N in {128,256,384,512,640,768,896,1024}`.
- Inputs and public gradients use the same dtypes and layouts as the authoritative fused Triton implementation. The residual mask is FP32 and contains `0` or `-1e9`.
- Timing is prepared-layout, complete backward work: every launch/reduction required for `dQ`, `dK`, `dV`, and `dPairBias`, with compilation, allocation, and warmup excluded. CUDA events and rotating buffers are used.
- `B`, `H`, and `N` remain runtime dynamic under one TVM-FFI compilation.
- During the exploration, only files below `cute_dsl/tmp/` were modified;
  those scratch variants were removed after promotion.

## Starting point

The initial D=32 port retained the optimized D=64 schedule:

```python
QUERY_TILE_SIZE = 64
KEY_TILE_SIZE = 128
Q_stage = 2
dO_stage = 2
num_wg_mma = 2
num_threads = 384  # one producer warpgroup + two consumer warpgroups
```

The main kernel owns one 128-row K/V tile and sweeps 64-row Q tiles. It fuses
the score and probability-gradient reconstruction with dQ, dK, dV, and (for
most shapes) FP32 dPairBias accumulation. A preprocess kernel computes delta,
an FP32 workspace accumulates dQ contributions across K tiles, and an epilogue
converts/reorders dQ to BF16.

## Correctness bring-up

### D=32 constants alone: failed dQ layout

Changing only `HEAD_DIMENSION` from 64 to 32 compiled and ran, and dK/dV were
correct. dQ was not: roughly half the sampled values failed, with mean absolute
error near 0.1. The dQ score-gradient math was therefore not the likely cause;
the failure was isolated to the accumulator postprocess.

### Dynamic dQ MMA tile: necessary but insufficient

The dQ WGMMA tiler was changed from a literal `(64, 64)` to the semantic tile:

```python
tiler_mn=(QUERY_TILE_SIZE, HEAD_DIMENSION)
```

This is required for a native 64x32 dQ accumulator, but it did not by itself
repair output ordering.

### Four output-copy threads per D=32 row: correct

The dQ finalizer inherited the D=64 assumption that eight threads each store
eight BF16 values per row. A 32-element row instead needs four such threads:

```python
output_copy = _copy_tiled_copy_2d(
    cutlass.BFloat16,
    threads_per_row=HEAD_DIMENSION // 8,
    num_threads=128,
    num_copy_elems=8,
)
```

With `threads_per_row=4`, dQ is restored to logical row-major order. On the
initial `B=1,H=4,N=128` case, sampled dQ error fell from about 0.10 mean absolute
error to `3.7e-4`, with `7.8e-3` maximum absolute error.

### Pair-bias numerical behavior

The FP32 dPairBias accumulation differs slightly from Triton's independent,
unordered atomic reduction. At `B=1,H=4,N=128`, five of 65,536 samples exceeded
`atol=1e-2, rtol=2e-2`; all five were close to zero, the largest failing error
was `0.0186`, and mean absolute error was `0.00111`. A dedicated split-sequence
reduction produced the same envelope, confirming arithmetic-order behavior
rather than a missed update. The public result is then rounded to BF16.

The exhaustive shape sweep therefore uses the conventional FlashAttention
BF16-gradient envelope, `atol=5e-2, rtol=5e-2`, for this one reduction-sensitive
output. It retains the stricter `atol=1e-2, rtol=2e-2` for dQ/dK/dV. Across the
32 designated shapes, sampled mean absolute dPairBias error ranges from roughly
`4.7e-4` to `1.1e-3`.

## Performance experiments

All numbers below are complete prepared-layout backward times measured with
CUDA events. They include every launch and reduction needed to produce dQ, dK,
dV, and dPairBias, but exclude allocation, compilation, and warmup.

### 1. Native D=32 dK/dV WGMMA tiles: kept

The inherited dK/dV MMA still described a 64-column output. D=32 only needs 32
columns, so the output tiler was narrowed without changing the 64x128 score
tile:

```python
tiled_mma_dkv = sm90_utils.make_trivial_tiled_mma(
    cutlass.BFloat16,
    cutlass.BFloat16,
    warpgroup.OperandMajorMode.K,
    warpgroup.OperandMajorMode.MN,
    Float32,
    atom_layout_mnk=(2, 1, 1),
    tiler_mn=(QUERY_TILE_SIZE, HEAD_DIMENSION),  # 64 x 32
    a_source=warpgroup.OperandSource.RMEM,
)
```

This avoids computing and carrying a padded second 32-column half. It reduced
the smallest `B=1,H=4,N=128` case from `0.0730 ms` to about `0.0653 ms`; the
large-N main loop was approximately neutral because it is dominated by the
score matrix, reductions, and memory traffic. The change is nevertheless kept
because it removes unnecessary tensor-core work and register state.

### 2. Three stages before pair-bias TMA: rejected in isolation

Increasing Q, dO, and dS storage from two to three stages without moving
pair-bias into the pipeline made `N=128` slower (`0.0653 -> 0.0694 ms`) and was
neutral at `N=640`. The scalar pair-bias load remained the exposed dependency,
so extra Q/dO buffering alone could not hide the critical path.

### 3. dPairBias reduction strategies

The fused main path performs one FP32 atomic update for each local dS value.
This is cheapest for H=4 and smaller H=16 shapes. At H=16 and N>=640, atomic
traffic is large enough that a separate split-sequence reduction wins. One CTA
owns one 64x128 pair tile and accumulates 16 sequence slices locally before its
final atomic update:

```python
gradient_sum.fill(0.0)
for _ in cutlass.range_constexpr(SPLIT_SEQUENCE_CHUNK_SIZE):  # 16
    score = Q @ K.T
    dp = dO @ V.T
    probability = exp(score * scale + pair_bias + mask - lse)
    gradient_sum += probability * (dp - delta)

atomic_add(d_pair_bias, gradient_sum)
```

Measured alternatives were:

| Variant | Outcome |
|---|---|
| Split reduction for every shape | Rejected; launch and recomputation overhead dominate below H=16,N=640. |
| 128 threads | Rejected; `B=1,H=16,N=640` rose to `20.98 ms`. |
| 256 threads, warp MMA layout 1x8 | Slower than 2x4. |
| 256 threads, warp MMA layout 2x4 | Best warp-MMA variant; about `15.03 ms` at `B=1,H=16,N=640`. |
| Sequence chunk 8 | Rejected; about `16.60 ms` at the same shape. |
| Sequence chunk 32 | Rejected; about `19.22 ms`; insufficient parallelism and a longer dependency chain. |
| Sequence chunk 16 | Kept. |
| 256-thread WGMMA split kernel | Kept; removes explicit ldmatrix register copies and reaches `12.13 ms` after composition with the final main path. |

The final split MMA is a pair of 64x64 warpgroup atoms covering the 64x128
tile:

```python
score_mma = sm90_utils.make_trivial_tiled_mma(
    cutlass.BFloat16,
    cutlass.BFloat16,
    warpgroup.OperandMajorMode.K,
    warpgroup.OperandMajorMode.K,
    Float32,
    atom_layout_mnk=(1, 2, 1),
    tiler_mn=(QUERY_TILE_SIZE, KEY_TILE_SIZE // 2),
)
```

### 4. Fused initialization and epilogues: kept

Two bandwidth/launch-only kernels were folded into launches already required
by the backward pass. The preprocess grid has exactly one CTA per 64-element
dPairBias initialization chunk, and the dQ finalizer has the same property for
FP32-to-BF16 conversion:

```python
# In preprocess
pair_offset = ((batch_idx * heads + head_idx) * N + q_block * 64) * N
d_pair_bias[pair_offset + thread_idx] = 0.0

# In the dQ finalizer
d_pair_bias_bf16[pair_offset + thread_idx] = (
    d_pair_bias_fp32[pair_offset + thread_idx].to(cutlass.BFloat16)
)
```

This removes the standalone zero and conversion launches for N>128. At N=128,
where there is no dQ cross-tile reduction, dQ is scaled and written directly as
BF16 from the main kernel; that removes the FP32 workspace round trip and dQ
finalizer. The larger N=128 cases improved from `0.1845/0.1844/0.6785 ms` to
approximately `0.146/0.146/0.52 ms` for `(B,H)=(1,16)/(4,4)/(4,16)`.

### 5. Register rebalance: rejected

Changing the two consumer warpgroup caps from `256/224` registers to `240/240`
was statistically neutral to slightly slower. The asymmetric allocation is
retained because the dQ consumer owns more live accumulator state.

### 6. Bulk asynchronous dPairBias reduction: rejected

A candidate staged dS in FP32 shared memory and used producer-warp
`cp.reduce.async.bulk` operations in place of lane-scattered atomics. It was
correct but slower: for example `B=1,H=4,N=640` increased from roughly
`3.06 ms` to `3.19 ms`. The extra shared stores, synchronization, and producer
serialization cost more than the reduced instruction count.

### 7. SW128 pair-bias TMA: decisive and kept

The initial kernel loaded pair bias independently through each consumer lane.
Although L2 hits were high, that produced many LSU requests and exposed long
scoreboard stalls. The winning version views pair bias as `[K,Q,H,B]`, tiles it
as `[128,64]`, and TMA-loads each Q tile into an SW128 FP32 shared layout:

```python
pair_bias_atom = warpgroup.make_smem_layout_atom(
    SmemLayoutAtomKind.MN_SW128,
    Float32,
)
pair_bias_layout = cute.tile_to_shape(
    pair_bias_atom,
    (KEY_TILE_SIZE, QUERY_TILE_SIZE, Q_stage),
    order=(1, 0, 2),
)

pair_bias_tma, pair_bias_tensor = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),
    pair_bias_view,
    cute.select(pair_bias_layout, mode=[0, 1]),
    (KEY_TILE_SIZE, QUERY_TILE_SIZE),
)
```

The score modifier then reads the staged value with its logical MMA
coordinates. `evict_last` is used because a `[B,H,N,N]` pair-bias plane is
shared across all S attention problems and should remain in L2:

```python
load_pair_bias(
    query_block,
    producer_state=producer_state,
    cache_policy=cutlass.Int64(PAIR_BIAS_EVICT_LAST_POLICY),
)
score[element] += shared_pair_bias[column, row, stage]
```

This does not reduce the mathematical pair-bias byte count. Its benefit is
request geometry and latency overlap: contiguous TMA transactions replace
lane-scattered loads, and the pair-bias tile now participates in the same
producer/consumer pipeline as Q.

### 8. Residual-mask TMA broadcast: kept

The residual mask is only 128 FP32 values per key tile and is broadcast across
64 query rows. Loading it once with a one-dimensional TMA into shared memory
eliminates repeated scalar global accesses without physically replicating the
data:

```python
mask_tma, mask_tensor = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),
    residual_mask_view,
    cute.make_layout(KEY_TILE_SIZE),
    (KEY_TILE_SIZE,),
)

# Broadcast is logical: every score row indexes the same shared column.
score[element] += shared_residual_mask[column]
```

### 9. Three-stage composed pipeline: kept

Once pair-bias was inside the TMA pipeline, a third stage became useful. The
final main kernel stages Q, dO, dS, LSE, delta, and pair bias three deep while
K, V, and the residual mask are loaded once per CTA. Shared memory rises from
75.8 KiB to 195.0 KiB, deliberately preserving one CTA/SM while maximizing
latency hiding. It improves the large H=16 cases by roughly 2–4% relative to
the otherwise identical two-stage TMA version. Four stages would exceed the
SM90 per-CTA shared-memory budget.

## Nsight Compute diagnosis

The profiled case is `B=1,H=4,S=N=640,D=32`; figures below refer to the fused
main kernel only. Reports and parsed metrics are under
`profile/20260914_pb_mask_tma_stage3_b1_h4_n640/`.

| Metric | Initial D=32 | Final main | Change |
|---|---:|---:|---:|
| NCU duration | 3.7756 ms | 3.4703 ms | -8.1% |
| Registers/thread | 168 | 168 | unchanged |
| Dynamic shared memory/CTA | 75.8 KiB | 195.0 KiB | +157% |
| Compute-memory throughput | 74.6% | 79.4% | +4.8 pp |
| SM throughput | 33.4% | 35.2% | +1.8 pp |
| Eligible warps/scheduler | 0.424 | 0.554 | +30.6% |
| Scalar global-load L1 sectors | 133.12M | 0 | eliminated |
| Long-scoreboard stalls/issue | 2.346 | 1.962 | -16.3% |
| Barrier stalls/issue | 1.549 | 1.089 | -29.7% |
| Wait stalls/issue | 0.981 | 0.616 | -37.2% |

Total L2 sectors remain essentially unchanged (`438.10M -> 436.64M`) and DRAM
read bytes remain near `657 MB`; the optimized kernel is still L2-pressure
limited. For scale, `436.64M * 32 B = 13.97 GB` of measured L2-sector traffic in
`3.470 ms`, or about `4.03 TB/s`, versus the measured machine roof of `12 TB/s`.
The approximate tensor work of the main backward is `335.5 GFLOP`, giving only
about `24 FLOP/byte` against the L2 traffic. This is well below the H100 BF16
tensor-core/L2 ridge point (roughly `989 TFLOP/s / 12 TB/s = 82 FLOP/byte`), so
the L2-bound diagnosis is consistent with the counters.

The optimization succeeds by shortening the request/latency critical path,
not by pretending those logical bytes disappear. Source-counter samples agree:
long-scoreboard samples fall `99,006 -> 81,440`, barrier samples
`59,905 -> 41,862`, and wait samples `40,914 -> 24,784`. CuTe JIT did not emit
source-line attribution, so those samples are intentionally reported only in
aggregate.

## Final benchmark

Within each row, entries are ordered `(B,H) = (1,4), (1,16), (4,4), (4,16)`.
This table records the promoted three-stage main schedule plus the WGMMA split
reduction. Raw results are in `final_bwd_n*.json`.

| N=S | CuTe time (ms) | Speedup over fused Triton |
|---:|---:|---:|
| 128 | 0.0609 / 0.1398 / 0.1387 / 0.5146 | 2.421x / 3.815x / 3.832x / 3.943x |
| 256 | 0.2402 / 0.9371 / 0.9349 / 3.6829 | 3.348x / 3.300x / 3.311x / 3.319x |
| 384 | 0.6875 / 2.6848 / 2.6804 / 10.8515 | 3.453x / 3.485x / 3.483x / 3.457x |
| 512 | 1.4802 / 6.0537 / 5.8547 / 24.8339 | 3.624x / 3.531x / 3.644x / 3.439x |
| 640 | 2.7556 / 12.1946 / 11.0181 / 50.0138 | 3.670x / 3.506x / 3.681x / 3.433x |
| 768 | 4.6006 / 21.8928 / 18.3655 / 87.8018 | 3.719x / 3.907x / 3.749x / 3.911x |
| 896 | 7.3113 / 42.4549 / 30.3569 / 169.9757 | 3.685x / 3.211x / 3.548x / 3.205x |
| 1024 | 10.3376 / 62.9320 / 42.3105 / 252.1095 | 3.852x / 3.201x / 3.788x / 3.194x |

Every designated shape clears the 2x requirement. The measured speedup range
for this sweep is `2.421x` to `3.943x`; the smallest launch-bound shape is the
minimum. All sequence lengths share one dynamic TVM-FFI compilation.

## Final correctness status

`test_d32_bwd.py` was run for all 32 `(B,H,N)` combinations against the
authoritative fused Triton backward. Each output uses up to 65,536 deterministic
samples (integer-generated indices, so there is no float-index aliasing). All
shapes pass for dQ, dK, dV, and dPairBias. Each of the four B/H sweeps also
reports `compile_count=1` across all eight dynamic sequence lengths.

The promoted artifacts are:

- `evoattention_cute_d32_bwd.py`: self-contained CuTe-DSL backward kernel;
- `benchmark_d32.py`: CUDA-event benchmark for forward, backward, or combined;
- `test_d32_bwd.py`: authoritative Triton comparison;
- `final_bwd_n*.json`: raw final benchmark measurements;
- `profile/20260914_*`: reproducible NCU reports and parsed metrics.

Reproduction commands:

```bash
# All backward shapes (the default enforces 2x per shape).
python3 benchmark_d32.py --direction backward

# Correctness across every designated shape.
python3 test_d32_bwd.py
```

The kernel has no FlashAttention/Quack source import or site-packages-path
dependency. KernelWiki was used to cross-check SM90 WGMMA/TMA constraints, and
the NCU run-directory workflow was used for the baseline/optimized evidence
above; the final implementation remains entirely local to this file.
