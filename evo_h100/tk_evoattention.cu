// ThunderKittens H100 kernel for AlphaFold-3 style attention
// (pair-bias + residue-mask triangle / pair-bias attention).
//
// Forward pass only. Semantics mirror the Triton reference kernel at
//   MegaFold/megafold/model/FusedEvoAttention/evoattention.py
//
// Tensor layout convention (all 4D after flattening B*N_SEQ):
//   Q, K, V, O : (B*N_SEQ, H,       SEQ_LEN, D)
//   pair_bias  : (B,       H,       SEQ_LEN, SEQ_LEN)  -- shared across MSA
//   res_mask   : (B*N_SEQ, 1,       1,       SEQ_LEN)  -- broadcast across heads & Q dim
//   L          : (B*N_SEQ, H,       1,       SEQ_LEN)  -- logsumexp (placeholder for bwd)
//
// Softmax semantics (matching Triton):
//   logits = (Q * scale) @ K^T + pair_bias + res_mask     // natural log space
//   O      = softmax(logits, dim=-1) @ V

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

using namespace kittens;
namespace cg = cooperative_groups;

// log2(e)
__device__ constexpr float LOG2E = 1.44269504089f;

// Per-D tile/pipeline config. CW is per-D so we can fit D=128
// within the shared memory budget.
template<int D> struct evo_fwd_tile_dims {};
template<> struct evo_fwd_tile_dims<64> {
    constexpr static int tile_width          = 64;
    constexpr static int qo_height           = 64;
    constexpr static int kv_height           = 128;
    constexpr static int stages              = 3;  // 3-stage pipeline fits in smem for D=64
    constexpr static int consumer_warpgroups = 2;
};
template<> struct evo_fwd_tile_dims<128> {
    constexpr static int tile_width          = 128;
    constexpr static int qo_height           = 64;
    constexpr static int kv_height           = 128;
    constexpr static int stages              = 2;
    constexpr static int consumer_warpgroups = 1;
};

template<int D>
constexpr int evo_num_workers() {
    return (evo_fwd_tile_dims<D>::consumer_warpgroups + 1) * kittens::WARPGROUP_WARPS;
}

template<int D> struct evo_fwd_globals {
    using q_tile    =         st_bf<evo_fwd_tile_dims<D>::qo_height, evo_fwd_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<evo_fwd_tile_dims<D>::kv_height, evo_fwd_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<evo_fwd_tile_dims<D>::kv_height, evo_fwd_tile_dims<D>::tile_width>;
    using o_tile    =         st_bf<evo_fwd_tile_dims<D>::qo_height, evo_fwd_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<evo_fwd_tile_dims<D>::qo_height, evo_fwd_tile_dims<D>::tile_width>>;
    using pb_tile   =         st_bf<evo_fwd_tile_dims<D>::qo_height, evo_fwd_tile_dims<D>::kv_height>;
    using rm_vec    =         sv_bf<evo_fwd_tile_dims<D>::kv_height>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using l_gl  = gl<float, -1, -1, -1, -1, l_col_vec>;
    using pb_gl = gl<bf16,  -1, -1, -1, -1, pb_tile>;
    using rm_gl = gl<bf16,  -1, -1, -1, -1, rm_vec>;

    q_gl  q;
    k_gl  k;
    v_gl  v;
    pb_gl pb;
    rm_gl rm;
    l_gl  l;
    o_gl  o;

    const int N;       // sequence length (SEQ_LEN)
    const int N_SEQ;   // MSA dimension, used to decompose blockIdx.z -> (batch, msa)
    const float scale; // 1/sqrt(D)
};

template<int D>
__global__ __launch_bounds__(evo_num_workers<D>() * kittens::WARP_THREADS, 1)
void evo_fwd_ker(const __grid_constant__ evo_fwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();
    int warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    using K = evo_fwd_tile_dims<D>;
    constexpr int CW              = K::consumer_warpgroups;
    constexpr int NUM_WARPGROUPS  = CW + 1;                       // +1 producer
    constexpr int NUM_WORKERS     = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using pb_tile   =         st_bf<K::qo_height, K::kv_height>;
    using rm_vec    =         sv_bf<K::kv_height>;

    q_tile    (&q_smem) [CW]              = al.allocate<q_tile,    CW>();
    k_tile    (&k_smem) [K::stages]       = al.allocate<k_tile,    K::stages>();
    v_tile    (&v_smem) [K::stages]       = al.allocate<v_tile,    K::stages>();
    pb_tile   (&pb_smem)[CW][K::stages]   = al.allocate<pb_tile,   CW, K::stages>();
    rm_vec    (&rm_smem)[K::stages]       = al.allocate<rm_vec,    K::stages>();
    l_col_vec (&l_smem) [CW]              = al.allocate<l_col_vec, CW>();
    auto      (*o_smem)                   = reinterpret_cast<o_tile(*)>(q_smem);

    const int kv_blocks     = g.N / K::kv_height;
    const int batch_msa_idx = blockIdx.z;
    const int head_idx      = blockIdx.y;
    const int batch_idx     = batch_msa_idx / g.N_SEQ;
    const int seq_idx       = blockIdx.x * CW;

    __shared__ kittens::semaphore qsmem_semaphore;
    __shared__ kittens::semaphore k_smem_arrived[K::stages];
    __shared__ kittens::semaphore v_smem_arrived[K::stages];
    __shared__ kittens::semaphore pb_smem_arrived[K::stages];
    __shared__ kittens::semaphore rm_smem_arrived[K::stages];
    __shared__ kittens::semaphore compute_done[K::stages];

    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        for (int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j],  0, 1);
            init_semaphore(v_smem_arrived[j],  0, 1);
            init_semaphore(pb_smem_arrived[j], 0, 1);
            init_semaphore(rm_smem_arrived[j], 0, 1);
            init_semaphore(compute_done[j], CW, 0);
        }

        // Q: one tile per consumer warpgroup, single arrival
        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
        for (int wg = 0; wg < CW; wg++) {
            coord<q_tile> q_tile_idx = {batch_msa_idx, head_idx, seq_idx + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        // Prologue: fill first (stages-1) stages with K, V, pair_bias, res_mask
        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {batch_msa_idx, head_idx, j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);

            // pair_bias: one tile per consumer warpgroup, all pooled on same mbarrier
            tma::expect_bytes(pb_smem_arrived[j], sizeof(pb_tile) * CW);
            for (int wg = 0; wg < CW; wg++) {
                coord<pb_tile> pb_tile_idx = {batch_idx, head_idx, seq_idx + wg, j};
                tma::load_async(pb_smem[wg][j], g.pb, pb_tile_idx, pb_smem_arrived[j]);
            }

            // res_mask: single shared vector (kv_height) per stage
            tma::expect_bytes(rm_smem_arrived[j], sizeof(rm_vec));
            coord<rm_vec> rm_idx = {batch_msa_idx, 0, 0, j};
            tma::load_async(rm_smem[j], g.rm, rm_idx, rm_smem_arrived[j]);
        }
    }
    __syncthreads();

    const int pipe_idx = K::stages - 1;

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        // ---- Producer warpgroup ----
        warpgroup::decrease_registers<32>();

        const int kv_iters = kv_blocks - 2;

        if (warpid == NUM_WORKERS - 4) {
            // Single thread issuing TMAs for the remaining stages
            for (int kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                const int nxt = kv_idx + 1;
                const int s   = nxt % K::stages;

                coord<k_tile> kv_tile_idx = {batch_msa_idx, head_idx, nxt, 0};
                warp::tma::expect_bytes(k_smem_arrived[s], sizeof(k_tile));
                warp::tma::load_async(k_smem[s], g.k, kv_tile_idx, k_smem_arrived[s]);

                warp::tma::expect_bytes(v_smem_arrived[s], sizeof(v_tile));
                warp::tma::load_async(v_smem[s], g.v, kv_tile_idx, v_smem_arrived[s]);

                warp::tma::expect_bytes(pb_smem_arrived[s], sizeof(pb_tile) * CW);
                for (int wg = 0; wg < CW; wg++) {
                    coord<pb_tile> pb_tile_idx = {batch_idx, head_idx, seq_idx + wg, nxt};
                    warp::tma::load_async(pb_smem[wg][s], g.pb, pb_tile_idx, pb_smem_arrived[s]);
                }

                warp::tma::expect_bytes(rm_smem_arrived[s], sizeof(rm_vec));
                coord<rm_vec> rm_idx = {batch_msa_idx, 0, 0, nxt};
                warp::tma::load_async(rm_smem[s], g.rm, rm_idx, rm_smem_arrived[s]);

                wait(compute_done[kv_idx % K::stages], (kv_idx / K::stages) % 2);
            }
        }
    } else {
        // ---- Consumer warpgroup ----
        warpgroup::increase_registers<160>();

        using att_tile_t = rt_fl<16, K::kv_height>;

        att_tile_t                       att_block;
        rt_bf<16, K::kv_height>          att_block_mma;
        rt_fl<16, K::tile_width>         o_reg;
        att_tile_t                       pb_reg;
        typename att_tile_t::row_vec     rm_reg;

        col_vec<rt_fl<16, K::kv_height>> max_vec;          // running max (in base-2 log space)
        col_vec<rt_fl<16, K::kv_height>> norm_vec;         // running denom (sum of exp2)
        col_vec<rt_fl<16, K::kv_height>> max_vec_last;     // previous max
        col_vec<rt_fl<16, K::kv_height>> alpha;            // correction factor

        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
        warp::zero(o_reg);

        const int kv_iters = kv_blocks - 1;
        // softmax_scale * log2(e): pre-compute for combining Q@K scaling with base-2 conversion
        const float softmax_scale_log2 = g.scale * LOG2E;

        wait(qsmem_semaphore, 0);

        for (int kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
            const int s = kv_idx % K::stages;
            const int phase = (kv_idx / K::stages) % 2;

            // Issue Q @ K^T (async); stash prev max for online-softmax correction
            // meanwhile (runs on different registers from att_block).
            wait(k_smem_arrived[s], phase);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[s]);
            warp::copy(max_vec_last, max_vec);
            warpgroup::mma_async_wait();

            // Fuse (scale * log2e) into a single multiply on att_block.
            warp::mul(att_block, att_block, softmax_scale_log2);

            // Add pair_bias (scale by log2(e) in registers so att stays in base-2 space).
            wait(pb_smem_arrived[s], phase);
            warpgroup::load(pb_reg, pb_smem[warpgroupid][s]);
            warp::mul(pb_reg, pb_reg, LOG2E);
            warp::add(att_block, att_block, pb_reg);

            // Add res_mask (broadcast across Q rows; also log2(e)-scaled).
            wait(rm_smem_arrived[s], phase);
            warp::load(rm_reg, rm_smem[s]);
            warp::mul(rm_reg, rm_reg, LOG2E);
            warp::add_col(att_block, att_block, rm_reg);

            // Online softmax (max_vec in base-2 log space throughout).
            warp::row_max(max_vec, att_block, max_vec);

            warp::sub(alpha, max_vec_last, max_vec);
            warp::exp2(alpha, alpha);

            warp::sub_row(att_block, att_block, max_vec);
            warp::exp2(att_block, att_block);

            warp::mul(norm_vec, norm_vec, alpha);
            warp::row_sum(norm_vec, att_block, norm_vec);

            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, alpha);

            wait(v_smem_arrived[s], phase);
            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[s]);
            warpgroup::mma_async_wait();

            if (warpgroup::laneid() == 0) arrive(compute_done[s], 1);
        }

        // Final divide by norm
        warp::div_row(o_reg, o_reg, norm_vec);

        warpgroup::store(o_smem[warpgroupid], o_reg);
        warpgroup::sync(warpgroupid + 4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {batch_msa_idx, head_idx, seq_idx + warpgroupid, 0};
            warp::tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        // L (logsumexp) — stored in natural log space: max*ln(2) + ln(norm)
        //   (recall max is in base-2 log space; ln(2)*max_base2 = max_natural)
        warp::mul(max_vec, max_vec, 0.69314718056f); // ln(2)
        warp::log(norm_vec, norm_vec);
        warp::add(norm_vec, norm_vec, max_vec);

        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid + 4);

        if (warpid % 4 == 0) {
            coord<l_col_vec> tile_idx = {batch_msa_idx, head_idx, 0, seq_idx + warpgroupid};
            warp::tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        warp::tma::store_async_wait();
    }
}

// =========================================================================================
// Backward preparation kernel: D[q] = sum_d(dO[q,d] * O[q,d])
// (Identical to mha_h100's bwd_attend_prep_ker; shape convention is (B*N_SEQ, H, N, D).)
// =========================================================================================

template<int D>
struct evo_bwd_prep_globals {
    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    og_gl og;
    o_gl  o;
    d_gl  d;
};

template<int D>
__global__ __launch_bounds__(4*kittens::WARP_THREADS, (D == 64) ? 2 : 1)
void evo_bwd_prep_ker(const __grid_constant__ evo_bwd_prep_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    og_tile (&og_smem)[4] = al.allocate<og_tile, 4>();
    o_tile  (&o_smem) [4] = al.allocate<o_tile , 4>();
    d_tile  (&d_smem) [4] = al.allocate<d_tile , 4>();

    rt_fl<4*16, D>          og_reg, o_reg;
    col_vec<rt_fl<4*16, D>> d_reg;

    __shared__ kittens::semaphore smem_semaphore;
    if (threadIdx.x == 0) {
        init_semaphore(smem_semaphore, 0, 1);
        tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * 4 * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            coord<o_tile> tile_idx = {blockIdx.z, blockIdx.y, (blockIdx.x * 4) + w, 0};
            warp::tma::load_async(o_smem[w],  g.o,  tile_idx, smem_semaphore);
            warp::tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
        }
    }

    wait(smem_semaphore, 0);
    warp::load(o_reg, o_smem[warpid]);
    warp::load(og_reg, og_smem[warpid]);
    warp::mul(og_reg, og_reg, o_reg);
    warp::row_sum(d_reg, og_reg);
    warp::store(d_smem[warpid], d_reg);
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < 4; w++) {
            coord<d_tile> tile_idx = {blockIdx.z, blockIdx.y, 0, (blockIdx.x * 4) + w};
            warp::tma::store_async(g.d, d_smem[w], tile_idx);
        }
    }
    warp::tma::store_async_wait();
}

// =========================================================================================
// Main backward kernel.
//
// Adapted from mha_h100.cu's bwd_attend_ker. Changes vs. the base kernel:
//   1. Grouped-query attention (hr) and causal masking removed (evoattention has neither).
//   2. Per-iteration pair_bias TMA load (tic/toc double-buffered) and add into S^T before
//      softmax recompute. Added in natural log space; the (scale * log2e) factor is
//      applied once by multiplying after all additions.
//   3. Per-CTA res_mask vector TMA load (once). Broadcast across Q via warp::add_row.
//   4. L recomputation: our forward saves the plain natural-log LSE (= ln(sum exp(logits))),
//      so we keep Q-K^T as the bare matmul, apply softmax_scale, add pair_bias + res_mask,
//      subtract L[q], and multiply by log2(e) to do exp2 for P.
//   5. d_pair_bias accumulation: per-element global atomicAdd from the ds register tile
//      (mirroring Triton's tl.atomic_add). This handles the N_SEQ broadcast (multiple MSA
//      batch slices contributing gradients to the same (batch, head, q, kv) entry).
//   6. CONSUMER_WARPGROUPS = 1 for both D=64 and D=128 to keep smem budget comfortable
//      with the added pair_bias buffer; kept same tile_h_qo / tile_h = 64 as the base.
// =========================================================================================

template<int D> struct evo_bwd_tile_dims {};
template<> struct evo_bwd_tile_dims<64> {
    constexpr static int tile_width          = 64;
    constexpr static int tile_h               = 64;
    constexpr static int tile_h_qo            = 64;
    constexpr static int consumer_warpgroups  = 2;   // CW=2 at D=64: ~130 KB smem, 2× CTAs for dK/dV.
    constexpr static int blocks_sm            = 1;
};
template<> struct evo_bwd_tile_dims<128> {
    constexpr static int tile_width          = 128;
    constexpr static int tile_h               = 64;
    constexpr static int tile_h_qo            = 64;
    constexpr static int consumer_warpgroups  = 1;   // CW=1 at D=128: smem limits prevent CW=2.
    constexpr static int blocks_sm            = 1;
};

template<int D>
constexpr int evo_bwd_num_workers() {
    return (evo_bwd_tile_dims<D>::consumer_warpgroups + 1) * kittens::WARPGROUP_WARPS;
}

template<int D>
struct evo_bwd_globals {
    using G = evo_bwd_tile_dims<D>;

    using q_tile  =         st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using v_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using og_tile =         st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile =         st_fl<G::tile_h_qo, G::tile_width>;
    using kg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using vg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using pb_tile =         st_bf<G::tile_h_qo, G::tile_h>;
    using rm_vec  =         sv_bf<G::tile_h>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
    using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
    using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;
    using l_gl  = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;
    using pb_gl = gl<bf16,  -1, -1, -1, -1, pb_tile>;
    using rm_gl = gl<bf16,  -1, -1, -1, -1, rm_vec>;

    q_gl  q;
    k_gl  k;
    v_gl  v;
    og_gl og;
    qg_gl qg;
    kg_gl kg;
    vg_gl vg;
    l_gl  l;
    d_gl  d;
    pb_gl pb;
    rm_gl rm;

    float* d_pair_bias;  // raw pointer for atomicAdd (shape (B, H, N, N) float32)

    const int N;
    const int N_SEQ;
    const int H;       // num heads (for d_pair_bias stride calc)
    const float scale;
};

// Stream a length-tile_h row_vec (indexed by col-axis of the register tile) across rows.
// reg_tile[r, c] <- smem_vec[tic][c].  Matches mha_h100.cu's stream_tile.
__device__ static inline void
evo_stream_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
        reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
    }
}

// Stream a length-tile_h row_vec (indexed by col-axis of the register tile) and subtract.
// Matches mha_h100.cu's stream_sub_tile.
__device__ static inline void
evo_stream_sub_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
    }
}

// Add pair_bias[q, kv] into s_block_t[kv, q].
// s_block_t has layout (kv × q) with the consumer warpgroup's 4 warps owning 16 kv-rows each.
// pb_smem has layout (tile_h_qo × tile_h), i.e. rows = q, cols = kv.
// Per-thread register element layout (standard rt_fl row layout for rt_fl<16, 64>):
//   tiles[0][i], i ∈ [0,4): base-tile along q-axis, covering q cols [16*i, 16*i+16)
//   Within each base tile, lane owns 8 elements:
//     data[0]: (kv_row0, q_lo0..q_lo0+1)    where kv_row0 = (laneid/4) + 0
//     data[1]: (kv_row1, q_lo0..q_lo0+1)    where kv_row1 = (laneid/4) + 8
//     data[2]: (kv_row0, q_hi0..q_hi0+1)
//     data[3]: (kv_row1, q_hi0..q_hi0+1)
//   q_lo0 = 16*i + 2*(laneid%4) + 0,  q_hi0 = q_lo0 + 8
template<typename ST>
__device__ static inline void
evo_add_pb_transposed(rt_fl<16, ST::rows /*=tile_h*/> &s_block_t, ST &pb_smem) {
    // NOTE: ST is st_bf<tile_h_qo, tile_h>; ST::rows == tile_h_qo == tile_h in our config.
    const int lane          = kittens::laneid();
    const int warp_in_wg    = kittens::warpid() % kittens::WARPGROUP_WARPS;
    const int kv_row0       = warp_in_wg * 16 + (lane / 4) + 0;
    const int kv_row1       = warp_in_wg * 16 + (lane / 4) + 8;
    const int col_lane_base = 2 * (lane % 4);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int q_lo = 16*i + col_lane_base;
        const int q_hi = q_lo + 8;

        // 8 scalar bf16 reads per base tile per lane.
        float pb00 = __bfloat162float(pb_smem[int2{q_lo + 0, kv_row0}]);
        float pb01 = __bfloat162float(pb_smem[int2{q_lo + 1, kv_row0}]);
        float pb10 = __bfloat162float(pb_smem[int2{q_lo + 0, kv_row1}]);
        float pb11 = __bfloat162float(pb_smem[int2{q_lo + 1, kv_row1}]);
        float pb20 = __bfloat162float(pb_smem[int2{q_hi + 0, kv_row0}]);
        float pb21 = __bfloat162float(pb_smem[int2{q_hi + 1, kv_row0}]);
        float pb30 = __bfloat162float(pb_smem[int2{q_hi + 0, kv_row1}]);
        float pb31 = __bfloat162float(pb_smem[int2{q_hi + 1, kv_row1}]);

        s_block_t.tiles[0][i].data[0].x += pb00;
        s_block_t.tiles[0][i].data[0].y += pb01;
        s_block_t.tiles[0][i].data[1].x += pb10;
        s_block_t.tiles[0][i].data[1].y += pb11;
        s_block_t.tiles[0][i].data[2].x += pb20;
        s_block_t.tiles[0][i].data[2].y += pb21;
        s_block_t.tiles[0][i].data[3].x += pb30;
        s_block_t.tiles[0][i].data[3].y += pb31;
    }
}

// atomicAdd ds_block_t[kv_local, q_local] into d_pair_bias[batch, head, q_base + q_local, kv_base + kv_local].
// Uses the same register layout as evo_add_pb_transposed.
__device__ static inline void
evo_atomic_add_dpair_bias(const rt_fl<16, 64> &ds_block_t,
                          float* dpb_base_bh, int q_base, int kv_base, int N)
{
    const int lane          = kittens::laneid();
    const int warp_in_wg    = kittens::warpid() % kittens::WARPGROUP_WARPS;
    const int kv_row0       = warp_in_wg * 16 + (lane / 4) + 0;
    const int kv_row1       = warp_in_wg * 16 + (lane / 4) + 8;
    const int col_lane_base = 2 * (lane % 4);

    const int kv_abs0 = kv_base + kv_row0;
    const int kv_abs1 = kv_base + kv_row1;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int q_lo = q_base + 16*i + col_lane_base;
        const int q_hi = q_lo + 8;

        // d_pair_bias[q, kv] in a (B, H, N, N) row-major tensor: offset = q*N + kv.
        atomicAdd(dpb_base_bh + (q_lo + 0) * N + kv_abs0, ds_block_t.tiles[0][i].data[0].x);
        atomicAdd(dpb_base_bh + (q_lo + 1) * N + kv_abs0, ds_block_t.tiles[0][i].data[0].y);
        atomicAdd(dpb_base_bh + (q_lo + 0) * N + kv_abs1, ds_block_t.tiles[0][i].data[1].x);
        atomicAdd(dpb_base_bh + (q_lo + 1) * N + kv_abs1, ds_block_t.tiles[0][i].data[1].y);
        atomicAdd(dpb_base_bh + (q_hi + 0) * N + kv_abs0, ds_block_t.tiles[0][i].data[2].x);
        atomicAdd(dpb_base_bh + (q_hi + 1) * N + kv_abs0, ds_block_t.tiles[0][i].data[2].y);
        atomicAdd(dpb_base_bh + (q_hi + 0) * N + kv_abs1, ds_block_t.tiles[0][i].data[3].x);
        atomicAdd(dpb_base_bh + (q_hi + 1) * N + kv_abs1, ds_block_t.tiles[0][i].data[3].y);
    }
}

template<int D>
__global__ __launch_bounds__(evo_bwd_num_workers<D>()*kittens::WARP_THREADS, evo_bwd_tile_dims<D>::blocks_sm)
void evo_bwd_ker(const __grid_constant__ evo_bwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int N       = g.N;
    const int N_SEQ   = g.N_SEQ;
    const float scale = g.scale;
    using G = evo_bwd_tile_dims<D>;
    constexpr int CW              = G::consumer_warpgroups;
    constexpr int NUM_WARPGROUPS  = CW + 1;   // +1 producer
    constexpr int NUM_WORKERS     = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;

    using kg_tile = st_fl<G::tile_h,    G::tile_width>;
    using vg_tile = st_fl<G::tile_h,    G::tile_width>;
    using k_tile  = st_bf<G::tile_h,    G::tile_width>;
    using v_tile  = st_bf<G::tile_h,    G::tile_width>;
    using q_tile  = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;
    using pb_tile   = st_bf<G::tile_h_qo, G::tile_h>;
    using rm_vec    = sv_bf<G::tile_h>;

    // Allocation order matters: kg_smem must alias (k_smem + v_smem) and
    // vg_smem must alias (q_smem + og_smem) — same pattern as mha_h100 bwd.
    k_tile   (&k_smem) [CW] = al.allocate<k_tile,  CW>();
    v_tile   (&v_smem) [CW] = al.allocate<v_tile,  CW>();
    q_tile   (&q_smem) [2]  = al.allocate<q_tile,  2>();
    og_tile  (&og_smem)[2]  = al.allocate<og_tile, 2>();
    qg_tile  (&qg_smem)     = al.allocate<qg_tile>();

    l_tile   (&l_smem) [2]      = al.allocate<l_tile,    2>();
    d_tile   (&d_smem) [2]      = al.allocate<d_tile,    2>();
    attn_tile(&ds_smem)[CW]     = al.allocate<attn_tile, CW>();
    pb_tile  (&pb_smem)[CW][2]  = al.allocate<pb_tile,   CW, 2>();
    rm_vec   (&rm_smem)[CW]     = al.allocate<rm_vec,    CW>();

    // Alias kg/vg onto k/q smem for the final write-back.
    kg_tile (*kg_smem) = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]);
    vg_tile (*vg_smem) = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]);

    const int warpid        = kittens::warpid();
    const int warpgroupid   = warpid / kittens::WARPGROUP_WARPS;
    const int qo_blocks     = N / G::tile_h_qo;
    const int batch_msa_idx = blockIdx.z;
    const int head_idx      = blockIdx.y;
    const int kv_block      = blockIdx.x;  // each CTA handles CW * tile_h kv rows
    const int batch_idx     = batch_msa_idx / N_SEQ;

    // d_pair_bias[batch, head, *, *] base pointer (row-major, strides H*N*N, N*N, N, 1).
    const int64_t N64 = static_cast<int64_t>(N);
    float* dpb_base_bh = g.d_pair_bias
                        + static_cast<int64_t>(batch_idx) * static_cast<int64_t>(g.H) * N64 * N64
                        + static_cast<int64_t>(head_idx)  * N64 * N64;

    __shared__ kittens::semaphore kv_b, rm_b, q_b[2], o_b[2], vec_b[2], pb_b[2];
    __shared__ kittens::semaphore compute_done[2], qg_ready;

    int tic = 0, toc = 1;
    const int q_start = 0;  // non-causal: every CTA sweeps all Q blocks.

    if (threadIdx.x == 0) {
        init_semaphore(kv_b,     0, 1);
        init_semaphore(rm_b,     0, 1);
        init_semaphore(qg_ready, 1, 0);
        for (int s = 0; s < 2; s++) {
            init_semaphore(q_b[s],          0, 1);
            init_semaphore(o_b[s],          0, 1);
            init_semaphore(vec_b[s],        0, 1);
            init_semaphore(pb_b[s],         0, 1);
            // Only warpgroup 0 arrives on compute_done (wg0 syncs wg1 via
            // group<8>::sync inside the loop, so one arrival covers both).
            init_semaphore(compute_done[s], 1, 0);
        }

        // KV (per-CTA, once) — one tile per consumer warpgroup, pooled on kv_b.
        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * CW);
        for (int w = 0; w < CW; w++) {
            coord<k_tile> tile_idx = {batch_msa_idx, head_idx, (kv_block * CW) + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }

        // res_mask (per-CTA, once) — one vec per consumer warpgroup, pooled on rm_b.
        tma::expect_bytes(rm_b, sizeof(rm_vec) * CW);
        for (int w = 0; w < CW; w++) {
            coord<rm_vec> rm_idx = {batch_msa_idx, 0, 0, (kv_block * CW) + w};
            tma::load_async(rm_smem[w], g.rm, rm_idx, rm_b);
        }

        // Prologue: kick off tic for Q, og, L, D, pair_bias
        coord<q_tile> q_tile_idx = {batch_msa_idx, head_idx, q_start, 0};
        tma::expect_bytes(q_b[tic],   sizeof(q_smem[0]));
        tma::load_async(q_smem[tic],  g.q,  q_tile_idx, q_b[tic]);
        tma::expect_bytes(o_b[tic],   sizeof(og_smem[0]));
        tma::load_async(og_smem[tic], g.og, q_tile_idx, o_b[tic]);

        coord<l_tile> vec_idx = {batch_msa_idx, head_idx, 0, q_start};
        tma::expect_bytes(vec_b[tic], sizeof(l_smem[0]) + sizeof(d_smem[0]));
        tma::load_async(l_smem[tic], g.l, vec_idx, vec_b[tic]);
        tma::load_async(d_smem[tic], g.d, vec_idx, vec_b[tic]);

        // pair_bias: one tile per consumer warpgroup per stage, pooled on pb_b.
        tma::expect_bytes(pb_b[tic], sizeof(pb_tile) * CW);
        for (int w = 0; w < CW; w++) {
            coord<pb_tile> pb_tile_idx = {batch_idx, head_idx, q_start, (kv_block * CW) + w};
            tma::load_async(pb_smem[w][tic], g.pb, pb_tile_idx, pb_b[tic]);
        }
    }
    __syncthreads();

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        // ----- Producer warpgroup -----
        warpgroup::decrease_registers<24>();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                if (qo_idx + 1 < qo_blocks) {
                    coord<q_tile> tile_idx = {batch_msa_idx, head_idx, qo_idx + 1, 0};
                    warp::tma::expect_bytes(q_b[toc],   sizeof(q_smem[0]));
                    warp::tma::load_async(q_smem[toc],  g.q,  tile_idx, q_b[toc]);
                    warp::tma::expect_bytes(o_b[toc],   sizeof(og_smem[0]));
                    warp::tma::load_async(og_smem[toc], g.og, tile_idx, o_b[toc]);

                    coord<l_tile> vec_idx = {batch_msa_idx, head_idx, 0, qo_idx + 1};
                    warp::tma::expect_bytes(vec_b[toc], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                    warp::tma::load_async(l_smem[toc], g.l, vec_idx, vec_b[toc]);
                    warp::tma::load_async(d_smem[toc], g.d, vec_idx, vec_b[toc]);

                    warp::tma::expect_bytes(pb_b[toc], sizeof(pb_tile) * CW);
                    for (int w = 0; w < CW; w++) {
                        coord<pb_tile> pb_tile_idx = {batch_idx, head_idx, qo_idx + 1, (kv_block * CW) + w};
                        warp::tma::load_async(pb_smem[w][toc], g.pb, pb_tile_idx, pb_b[toc]);
                    }
                }

                wait(compute_done[tic], ((qo_idx - q_start)/2)%2);
            }
        }
        else if (warpid % kittens::WARPGROUP_WARPS == 1) {
            for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
                wait(compute_done[tic], ((qo_idx - q_start)/2)%2);

                coord<qg_tile> tile_idx = {batch_msa_idx, head_idx, qo_idx, 0};
                warp::tma::store_add_async(g.qg, qg_smem, tile_idx);
                warp::tma::store_async_wait();

                if (laneid() == 0) arrive(qg_ready);
            }
        }
    }
    else {
        // ----- Consumer warpgroup(s) -----
        // Shared consumer setup — each wg has its own copy.
        rt_fl<16, G::tile_width> kg_reg, vg_reg;
        rt_fl<16, 64>            s_block_t,  p_block_t;
        rt_fl<16, 64>            ds_block_t, dp_block_t;
        rt_bf<16, 64>            p_block_t_mma, ds_block_t_mma;

        warp::zero(kg_reg);
        warp::zero(vg_reg);

        // res_mask: one sv_bf<tile_h> per consumer warpgroup (for its kv slice).
        wait(rm_b, 0);
        typename rt_fl<16, 64>::col_vec rm_reg_f;
        warpgroup::load(rm_reg_f, rm_smem[warpgroupid]);

        wait(kv_b, 0);

        const float log2e = 1.44269504089f;

        // Per-warpgroup register budget. wg0 needs more (qg_reg for dQ).
        if (warpgroupid == 0) warpgroup::increase_registers<256>();
        else                  warpgroup::increase_registers<224>();

        for (int qo_idx = q_start; qo_idx < qo_blocks; qo_idx++, tic ^= 1, toc ^= 1) {
            // --- Recompute P^T = softmax(Q K^T * scale + pair_bias + res_mask)^T --
            wait(q_b[tic],   ((qo_idx - q_start)/2)%2);
            warpgroup::mm_ABt(s_block_t, k_smem[warpgroupid], q_smem[tic]);
            warpgroup::mma_commit_group();

            wait(vec_b[tic], ((qo_idx - q_start)/2)%2);
            warpgroup::mma_async_wait();

            warp::mul(s_block_t, s_block_t, scale);

            wait(pb_b[tic], ((qo_idx - q_start)/2)%2);
            evo_add_pb_transposed(s_block_t, pb_smem[warpgroupid][tic]);

            warp::add_row(s_block_t, s_block_t, rm_reg_f);

            evo_stream_sub_tile(s_block_t, l_smem, tic);

            warp::mul(s_block_t, s_block_t, log2e);
            warp::exp2(s_block_t, s_block_t);

            warp::copy(p_block_t,     s_block_t);
            warp::copy(p_block_t_mma, s_block_t);

            // --- dP^T = V @ dO^T ---
            wait(o_b[tic], ((qo_idx - q_start)/2)%2);
            warpgroup::mm_ABt(dp_block_t, v_smem[warpgroupid], og_smem[tic]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            evo_stream_sub_tile(dp_block_t, d_smem, tic);

            warp::mul(ds_block_t, p_block_t, dp_block_t);

            // Atomic-add dS (pre-scale) into d_pair_bias.
            evo_atomic_add_dpair_bias(ds_block_t,
                                       dpb_base_bh,
                                       /*q_base*/ qo_idx * G::tile_h_qo,
                                       /*kv_base*/ (kv_block * CW + warpgroupid) * G::tile_h,
                                       N);

            warp::mul(ds_block_t, ds_block_t, scale);

            // dV += P^T @ dO
            warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[tic]);
            warpgroup::mma_commit_group();

            warp::copy(ds_block_t_mma, ds_block_t);
            warpgroup::store(ds_smem[warpgroupid], ds_block_t);

            // dK += dS^T @ Q
            warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[tic]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            // Cross-warpgroup sync so wg0 can read all ds_smem[*].
            group<CW * kittens::WARPGROUP_WARPS>::sync(10);

            if (warpgroupid == 0) {
                // --- dQ = sum_wg (ds_smem[wg]^T @ k_smem[wg])  via mm_AtB + mma_AtB ---
                rt_fl<16, G::tile_width> qg_reg;
                warpgroup::mm_AtB(qg_reg, ds_smem[0], k_smem[0]);
                if constexpr (CW >= 2) warpgroup::mma_AtB(qg_reg, ds_smem[1], k_smem[1]);
                if constexpr (CW >= 3) warpgroup::mma_AtB(qg_reg, ds_smem[2], k_smem[2]);
                warpgroup::mma_commit_group();

                wait(qg_ready, toc);
                if (qo_idx > 0) warp::tma::store_async_wait();

                warpgroup::mma_async_wait();
                warpgroup::store(qg_smem, qg_reg);
                group<4>::sync(warpgroup::groupid() + 4);

                if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
            }
        }

        // ----- Final dK, dV writes (both warpgroups participate, each writes its own slice) -----
        // Sync all consumers before aliasing kg/vg over k/v/q/og smem.
        group<CW * kittens::WARPGROUP_WARPS>::sync(10);
        warpgroup::store(kg_smem[warpgroupid], kg_reg);
        group<4>::sync(warpgroup::groupid() + 4);
        if (warpid % 4 == 0) {
            coord<kg_tile> tile_idx = {batch_msa_idx, head_idx, (kv_block * CW) + warpgroupid, 0};
            warp::tma::store_add_async(g.kg, kg_smem[warpgroupid], tile_idx);
            warp::tma::store_commit_group();
        }

        wait(qg_ready, toc);
        warpgroup::store(vg_smem[warpgroupid], vg_reg);
        group<4>::sync(warpgroup::groupid() + 4);
        if (warpid % 4 == 0) {
            coord<vg_tile> tile_idx = {batch_msa_idx, head_idx, (kv_block * CW) + warpgroupid, 0};
            warp::tma::store_add_async(g.vg, vg_smem[warpgroupid], tile_idx);
            warp::tma::store_commit_group();
        }
        warp::tma::store_async_wait();
    }
}

#ifdef TORCH_COMPILE

#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <cmath>

// Forward wrapper. Accepts tensors in 4D layout (B*N_SEQ, H, SEQ_LEN, D) etc.
// All input tensors must be bfloat16, contiguous, CUDA.
//
// pair_bias shape: (B, H, SEQ_LEN, SEQ_LEN) -- broadcast across MSA
// res_mask  shape: (B*N_SEQ, 1, 1, SEQ_LEN)
//
// Returns {O, L} where
//   O: (B*N_SEQ, H, SEQ_LEN, D) bfloat16
//   L: (B*N_SEQ, H, 1, SEQ_LEN) float32 (logsumexp, for backward)
// If softmax_scale <= 0, we fall back to 1/sqrt(head_dim). Callers that pad
// (e.g. pad D=96 -> D=128 in shared memory) must pass 1/sqrt(true_D) here so
// the softmax matches the unpadded reference.
std::vector<at::Tensor>
evoattention_forward(at::Tensor q,
                     at::Tensor k,
                     at::Tensor v,
                     at::Tensor pair_bias,
                     at::Tensor res_mask,
                     int64_t    n_seq,
                     double     softmax_scale_override = 0.0)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(pair_bias);
    CHECK_INPUT(res_mask);

    TORCH_CHECK(q.dim() == 4, "Q must be 4D: (B*N_SEQ, H, SEQ_LEN, D)");
    TORCH_CHECK(k.dim() == 4, "K must be 4D: (B*N_SEQ, H, SEQ_LEN, D)");
    TORCH_CHECK(v.dim() == 4, "V must be 4D: (B*N_SEQ, H, SEQ_LEN, D)");
    TORCH_CHECK(pair_bias.dim() == 4, "pair_bias must be 4D: (B, H, SEQ_LEN, SEQ_LEN)");
    TORCH_CHECK(res_mask.dim() == 4, "res_mask must be 4D: (B*N_SEQ, 1, 1, SEQ_LEN)");

    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "V must be bfloat16");
    TORCH_CHECK(pair_bias.scalar_type() == at::kBFloat16, "pair_bias must be bfloat16");
    TORCH_CHECK(res_mask.scalar_type() == at::kBFloat16, "res_mask must be bfloat16");

    const auto batch_msa = q.size(0);
    const auto heads     = q.size(1);
    const auto seq_len   = q.size(2);
    const auto head_dim  = q.size(3);

    TORCH_CHECK(k.size(0) == batch_msa && k.size(1) == heads && k.size(2) == seq_len && k.size(3) == head_dim,
                "K shape must match Q shape");
    TORCH_CHECK(v.size(0) == batch_msa && v.size(1) == heads && v.size(2) == seq_len && v.size(3) == head_dim,
                "V shape must match Q shape");

    TORCH_CHECK(batch_msa % n_seq == 0, "batch_msa must be divisible by N_SEQ");
    const int64_t batch = batch_msa / n_seq;

    TORCH_CHECK(pair_bias.size(0) == batch,
                "pair_bias batch (dim 0) must equal batch_msa / N_SEQ");
    TORCH_CHECK(pair_bias.size(1) == heads,
                "pair_bias head dim (dim 1) must equal qo_heads");
    TORCH_CHECK(pair_bias.size(2) == seq_len && pair_bias.size(3) == seq_len,
                "pair_bias must be (B, H, SEQ_LEN, SEQ_LEN)");

    TORCH_CHECK(res_mask.size(0) == batch_msa, "res_mask dim 0 must equal B*N_SEQ");
    TORCH_CHECK(res_mask.size(1) == 1 && res_mask.size(2) == 1,
                "res_mask dims 1 and 2 must be 1");
    TORCH_CHECK(res_mask.size(3) == seq_len, "res_mask dim 3 must equal SEQ_LEN");

    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "Only head_dim=64 or 128 supported; got ", head_dim);

    // Tile constraints (the per-D consumer_warpgroups count is checked during launch)
    constexpr int QO_H = 64;
    constexpr int KV_H = 128;
    TORCH_CHECK(seq_len % KV_H == 0, "SEQ_LEN must be divisible by kv_height=", KV_H);

    bf16* d_q  = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k  = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v  = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_pb = reinterpret_cast<bf16*>(pair_bias.data_ptr<c10::BFloat16>());
    bf16* d_rm = reinterpret_cast<bf16*>(res_mask.data_ptr<c10::BFloat16>());

    at::Tensor o = at::empty({batch_msa, heads, seq_len, head_dim}, q.options());
    at::Tensor l = at::empty({batch_msa, heads, 1, seq_len}, q.options().dtype(at::kFloat));

    bf16*  d_o = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    float* d_l = l.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // If the caller passed a positive override, use it. Needed when the caller
    // zero-pads Q/K/V along D (e.g. true D=96 padded to D=128): they must
    // request 1/sqrt(96) here so softmax semantics match the unpadded ref.
    const float softmax_scale = (softmax_scale_override > 0.0)
        ? static_cast<float>(softmax_scale_override)
        : 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto launch = [&](auto D_tag) {
        constexpr int D   = decltype(D_tag)::value;
        constexpr int D_CW = evo_fwd_tile_dims<D>::consumer_warpgroups;
        constexpr int D_NUM_WORKERS = evo_num_workers<D>();

        TORCH_CHECK(seq_len % (D_CW * QO_H) == 0,
                    "SEQ_LEN must be divisible by consumer_warpgroups*qo_height = ",
                    D_CW * QO_H, " for head_dim=", D);

        using globals = evo_fwd_globals<D>;
        using q_global  = typename globals::q_gl;
        using k_global  = typename globals::k_gl;
        using v_global  = typename globals::v_gl;
        using o_global  = typename globals::o_gl;
        using l_global  = typename globals::l_gl;
        using pb_global = typename globals::pb_gl;
        using rm_global = typename globals::rm_gl;

        q_global  qg_arg{d_q,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        k_global  kg_arg{d_k,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        v_global  vg_arg{d_v,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        o_global  og_arg{d_o,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        l_global  lg_arg{d_l,  (unsigned)batch_msa, (unsigned)heads, 1u,                 (unsigned)seq_len};
        pb_global pbg_arg{d_pb,(unsigned)batch,     (unsigned)heads, (unsigned)seq_len, (unsigned)seq_len};
        rm_global rmg_arg{d_rm,(unsigned)batch_msa, 1u,              1u,                 (unsigned)seq_len};

        globals g{qg_arg, kg_arg, vg_arg, pbg_arg, rmg_arg, lg_arg, og_arg,
                  static_cast<int>(seq_len), static_cast<int>(n_seq), softmax_scale};

        auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;

        dim3 grid(seq_len / (D_CW * QO_H),
                  static_cast<unsigned>(heads),
                  static_cast<unsigned>(batch_msa));

        cudaFuncSetAttribute(evo_fwd_ker<D>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             mem_size);
        evo_fwd_ker<D><<<grid, 32 * D_NUM_WORKERS, mem_size, stream>>>(g);
        CHECK_CUDA_ERROR(cudaGetLastError());
    };

    if (head_dim == 64) {
        launch(std::integral_constant<int, 64>{});
    } else {
        launch(std::integral_constant<int, 128>{});
    }

    cudaStreamSynchronize(stream);

    return {o, l};
}

// Backward wrapper. Accepts tensors in the same 4D layout as forward, plus O, L from fwd,
// and dO. Returns {dQ, dK, dV, d_pair_bias} where
//   dQ, dK, dV    : (B*N_SEQ, H, N, D) float32 (caller casts to bf16)
//   d_pair_bias   : (B, H, N, N)       float32 (caller reshapes to (B,1,H,N,N))
// If softmax_scale <= 0, falls back to 1/sqrt(head_dim). Pass an override when D is padded
// (must match the fwd's override so L is consistent).
std::vector<at::Tensor>
evoattention_backward(at::Tensor q,
                      at::Tensor k,
                      at::Tensor v,
                      at::Tensor pair_bias,
                      at::Tensor res_mask,
                      at::Tensor o,
                      at::Tensor l_vec,
                      at::Tensor og,
                      int64_t    n_seq,
                      double     softmax_scale_override = 0.0)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(pair_bias);
    CHECK_INPUT(res_mask);
    CHECK_INPUT(o);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(og);

    TORCH_CHECK(q.dim() == 4, "Q must be 4D: (B*N_SEQ, H, SEQ_LEN, D)");
    TORCH_CHECK(q.scalar_type()         == at::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(k.scalar_type()         == at::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(v.scalar_type()         == at::kBFloat16, "V must be bfloat16");
    TORCH_CHECK(pair_bias.scalar_type() == at::kBFloat16, "pair_bias must be bfloat16");
    TORCH_CHECK(res_mask.scalar_type()  == at::kBFloat16, "res_mask must be bfloat16");
    TORCH_CHECK(o.scalar_type()         == at::kBFloat16, "O must be bfloat16");
    TORCH_CHECK(og.scalar_type()        == at::kBFloat16, "dO must be bfloat16");
    TORCH_CHECK(l_vec.scalar_type()     == at::kFloat,    "L must be float32");

    const auto batch_msa = q.size(0);
    const auto heads     = q.size(1);
    const auto seq_len   = q.size(2);
    const auto head_dim  = q.size(3);

    TORCH_CHECK(batch_msa % n_seq == 0, "batch_msa must be divisible by N_SEQ");
    const int64_t batch = batch_msa / n_seq;

    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "Only head_dim=64 or 128 supported; got ", head_dim);

    constexpr int QO_H = 64;
    constexpr int KV_H = 64;  // bwd tile_h
    TORCH_CHECK(seq_len % QO_H == 0, "SEQ_LEN must be divisible by ", QO_H);
    TORCH_CHECK(seq_len % KV_H == 0, "SEQ_LEN must be divisible by ", KV_H);

    bf16* d_q  = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k  = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v  = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_pb = reinterpret_cast<bf16*>(pair_bias.data_ptr<c10::BFloat16>());
    bf16* d_rm = reinterpret_cast<bf16*>(res_mask.data_ptr<c10::BFloat16>());
    bf16* d_o  = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16* d_og = reinterpret_cast<bf16*>(og.data_ptr<c10::BFloat16>());
    float* d_l = l_vec.data_ptr<float>();

    // Outputs: zero-initialized because bwd accumulates (TMA store-add for dQ/dK/dV,
    // atomicAdd for d_pair_bias).
    at::Tensor qg = at::zeros({batch_msa, heads, seq_len, head_dim}, q.options().dtype(at::kFloat));
    at::Tensor kg = at::zeros({batch_msa, heads, seq_len, head_dim}, q.options().dtype(at::kFloat));
    at::Tensor vg = at::zeros({batch_msa, heads, seq_len, head_dim}, q.options().dtype(at::kFloat));
    at::Tensor d_vec   = at::empty({batch_msa, heads, 1, seq_len}, q.options().dtype(at::kFloat));
    at::Tensor dpb_out = at::zeros({batch, heads, seq_len, seq_len}, q.options().dtype(at::kFloat));

    float* d_qg  = qg.data_ptr<float>();
    float* d_kg  = kg.data_ptr<float>();
    float* d_vg  = vg.data_ptr<float>();
    float* d_d   = d_vec.data_ptr<float>();
    float* d_dpb = dpb_out.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const float softmax_scale = (softmax_scale_override > 0.0)
        ? static_cast<float>(softmax_scale_override)
        : 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto launch = [&](auto D_tag) {
        constexpr int D = decltype(D_tag)::value;

        // ---- Preparation kernel: compute D[q] = sum(dO * O) ----
        {
            using prep = evo_bwd_prep_globals<D>;
            using og_global = typename prep::og_gl;
            using o_global  = typename prep::o_gl;
            using d_global  = typename prep::d_gl;

            og_global prep_og{d_og, (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
            o_global  prep_o {d_o,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
            d_global  prep_d {d_d,  (unsigned)batch_msa, (unsigned)heads, 1u,                 (unsigned)seq_len};

            prep bwd_g{prep_og, prep_o, prep_d};

            using prep_og_tile = st_bf<64, D>;
            using prep_o_tile  = st_bf<64, D>;
            using prep_d_tile  = col_vec<st_fl<64, D>>;
            size_t prep_mem = sizeof(prep_og_tile) * 4 + sizeof(prep_o_tile) * 4 + sizeof(prep_d_tile) * 4 + 4096;

            cudaFuncSetAttribute(evo_bwd_prep_ker<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, prep_mem);

            dim3 grid_prep(seq_len / (4 * 16), (unsigned)heads, (unsigned)batch_msa);
            evo_bwd_prep_ker<D><<<grid_prep, 4 * kittens::WARP_THREADS, prep_mem, stream>>>(bwd_g);
            CHECK_CUDA_ERROR(cudaGetLastError());
        }

        // ---- Main backward kernel ----
        using globals = evo_bwd_globals<D>;

        typename globals::q_gl  qa {d_q,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::k_gl  ka {d_k,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::v_gl  va {d_v,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::og_gl oga{d_og, (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::qg_gl qga{d_qg, (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::kg_gl kga{d_kg, (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::vg_gl vga{d_vg, (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::l_gl  la {d_l,  (unsigned)batch_msa, (unsigned)heads, 1u,                 (unsigned)seq_len};
        typename globals::d_gl  da {d_d,  (unsigned)batch_msa, (unsigned)heads, 1u,                 (unsigned)seq_len};
        typename globals::pb_gl pba{d_pb, (unsigned)batch,     (unsigned)heads, (unsigned)seq_len,  (unsigned)seq_len};
        typename globals::rm_gl rma{d_rm, (unsigned)batch_msa, 1u,              1u,                 (unsigned)seq_len};

        globals g{qa, ka, va, oga, qga, kga, vga, la, da, pba, rma,
                  d_dpb,
                  static_cast<int>(seq_len), static_cast<int>(n_seq),
                  static_cast<int>(heads), softmax_scale};

        auto mem_size = kittens::MAX_SHARED_MEMORY - 1024;
        cudaFuncSetAttribute(evo_bwd_ker<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

        constexpr int D_CW          = evo_bwd_tile_dims<D>::consumer_warpgroups;
        constexpr int D_NUM_WORKERS = evo_bwd_num_workers<D>();
        TORCH_CHECK(seq_len % (D_CW * KV_H) == 0,
                    "SEQ_LEN must be divisible by consumer_warpgroups*tile_h = ",
                    D_CW * KV_H, " for head_dim=", D);

        dim3 grid_bwd(seq_len / (D_CW * KV_H),
                      static_cast<unsigned>(heads),
                      static_cast<unsigned>(batch_msa));
        evo_bwd_ker<D><<<grid_bwd, 32 * D_NUM_WORKERS, mem_size, stream>>>(g);
        CHECK_CUDA_ERROR(cudaGetLastError());
    };

    if (head_dim == 64) { launch(std::integral_constant<int, 64>{}); }
    else                { launch(std::integral_constant<int, 128>{}); }

    cudaStreamSynchronize(stream);

    return {qg, kg, vg, dpb_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("evoattention_forward", &evoattention_forward,
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("pair_bias"),
          pybind11::arg("res_mask"),
          pybind11::arg("n_seq"),
          pybind11::arg("softmax_scale") = 0.0,
          "AlphaFold-3 style attention forward (pair_bias + res_mask). "
          "Inputs: Q,K,V (B*N_SEQ,H,N,D) bf16; pair_bias (B,H,N,N) bf16; res_mask (B*N_SEQ,1,1,N) bf16. "
          "Returns {O (B*N_SEQ,H,N,D) bf16, L (B*N_SEQ,H,1,N) fp32}. "
          "softmax_scale <= 0 -> kernel uses 1/sqrt(head_dim). Pass an override when D is padded.");

    m.def("evoattention_backward", &evoattention_backward,
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("pair_bias"),
          pybind11::arg("res_mask"),
          pybind11::arg("o"),
          pybind11::arg("l_vec"),
          pybind11::arg("og"),
          pybind11::arg("n_seq"),
          pybind11::arg("softmax_scale") = 0.0,
          "AlphaFold-3 style attention backward. "
          "Inputs: saved Q,K,V,O,L,dO + pair_bias + res_mask (all in fwd layout). "
          "Returns {dQ (B*N_SEQ,H,N,D) fp32, dK fp32, dV fp32, d_pair_bias (B,H,N,N) fp32}. "
          "d_pair_bias is atomic-add accumulated across MSA batch (pair_bias is broadcast across N_SEQ).");
}

#endif // TORCH_COMPILE
