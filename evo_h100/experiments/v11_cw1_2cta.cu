// v11_cw1_2cta: CW=1 (single consumer warpgroup) per CTA, with 2 CTAs/SM.
// Total threads per CTA = 256 (1 consumer + 1 producer WG); two CTAs share
// an SM. Hypothesis: smaller CTAs let H100's scheduler interleave two
// kernels on each SM, hiding WGMMA + softmax latency on either side.
//
// Same outer-loop structure as v6 (one combined load semaphore, no ping-pong).
// Loses CW=2's "two Q tiles share one K load" smem reuse, but gains 2x in-
// flight CTAs/SM.

#include "kittens.cuh"
#include <cooperative_groups.h>

using namespace kittens;

__device__ constexpr float LOG2E = 1.44269504089f;

template<int D> struct evo_fwd_tile_dims {};
template<> struct evo_fwd_tile_dims<64> {
    constexpr static int tile_width          = 64;
    constexpr static int qo_height           = 64;
    constexpr static int kv_height           = 64;
    constexpr static int consumer_warpgroups = 1;   // CW=1
    constexpr static int stages              = 3;
    constexpr static int blocks_sm           = 2;   // 2 CTAs/SM
};
template<> struct evo_fwd_tile_dims<128> {
    constexpr static int tile_width          = 128;
    constexpr static int qo_height           = 64;
    constexpr static int kv_height           = 128;
    constexpr static int stages              = 2;
    constexpr static int consumer_warpgroups = 1;
    constexpr static int blocks_sm           = 1;
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

    const int N;
    const int N_SEQ;
    const float scale;
};

template<int D>
__global__ __launch_bounds__(evo_num_workers<D>() * kittens::WARP_THREADS, evo_fwd_tile_dims<D>::blocks_sm)
void evo_fwd_ker(const __grid_constant__ evo_fwd_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();
    int warpgroupid = warpid / kittens::WARPGROUP_WARPS;

    using K = evo_fwd_tile_dims<D>;
    constexpr int CW              = K::consumer_warpgroups;
    constexpr int NUM_WARPGROUPS  = CW + 1;
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
    __shared__ kittens::semaphore load_arrived[K::stages];
    __shared__ kittens::semaphore compute_done[K::stages];

    const int load_bytes_per_stage = (int)(sizeof(k_tile) + sizeof(v_tile) + sizeof(pb_tile) * CW + sizeof(rm_vec));

    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        for (int j = 0; j < K::stages; j++) {
            init_semaphore(load_arrived[j], 0, 1);
            init_semaphore(compute_done[j], CW, 0);
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
        for (int wg = 0; wg < CW; wg++) {
            coord<q_tile> q_tile_idx = {batch_msa_idx, head_idx, seq_idx + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            tma::expect_bytes(load_arrived[j], load_bytes_per_stage);
            coord<k_tile> kv_tile_idx = {batch_msa_idx, head_idx, j, 0};
            tma::load_async(k_smem[j], g.k, kv_tile_idx, load_arrived[j]);
            tma::load_async(v_smem[j], g.v, kv_tile_idx, load_arrived[j]);

            for (int wg = 0; wg < CW; wg++) {
                coord<pb_tile> pb_tile_idx = {batch_idx, head_idx, seq_idx + wg, j};
                tma::load_async(pb_smem[wg][j], g.pb, pb_tile_idx, load_arrived[j]);
            }
            coord<rm_vec> rm_idx = {batch_msa_idx, 0, 0, j};
            tma::load_async(rm_smem[j], g.rm, rm_idx, load_arrived[j]);
        }
    }
    __syncthreads();

    const int pipe_idx = K::stages - 1;

    if (warpgroupid == NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<32>();

        const int kv_iters = kv_blocks - 2;

        if (warpid == NUM_WORKERS - 4) {
            for (int kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                const int nxt = kv_idx + 1;
                const int s   = nxt % K::stages;

                warp::tma::expect_bytes(load_arrived[s], load_bytes_per_stage);
                coord<k_tile> kv_tile_idx = {batch_msa_idx, head_idx, nxt, 0};
                warp::tma::load_async(k_smem[s], g.k, kv_tile_idx, load_arrived[s]);
                warp::tma::load_async(v_smem[s], g.v, kv_tile_idx, load_arrived[s]);

                for (int wg = 0; wg < CW; wg++) {
                    coord<pb_tile> pb_tile_idx = {batch_idx, head_idx, seq_idx + wg, nxt};
                    warp::tma::load_async(pb_smem[wg][s], g.pb, pb_tile_idx, load_arrived[s]);
                }
                coord<rm_vec> rm_idx = {batch_msa_idx, 0, 0, nxt};
                warp::tma::load_async(rm_smem[s], g.rm, rm_idx, load_arrived[s]);

                wait(compute_done[kv_idx % K::stages], (kv_idx / K::stages) % 2);
            }
        }
    } else {
        warpgroup::increase_registers<224>();   // 224 regs OK with 2 CTAs/SM (per recipe)

        using att_tile_t = rt_fl<16, K::kv_height>;

        att_tile_t                       att_block;
        rt_bf<16, K::kv_height>          att_block_mma;
        rt_fl<16, K::tile_width>         o_reg;
        att_tile_t                       pb_reg;
        typename att_tile_t::row_vec     rm_reg;

        col_vec<rt_fl<16, K::kv_height>> max_vec;
        col_vec<rt_fl<16, K::kv_height>> norm_vec;
        col_vec<rt_fl<16, K::kv_height>> max_vec_last;
        col_vec<rt_fl<16, K::kv_height>> alpha;

        warp::neg_infty(max_vec);
        warp::zero(norm_vec);
        warp::zero(o_reg);

        const int kv_iters = kv_blocks - 1;
        const float softmax_scale_log2 = g.scale * LOG2E;

        wait(qsmem_semaphore, 0);

        for (int kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
            const int s = kv_idx % K::stages;
            const int phase = (kv_idx / K::stages) % 2;

            wait(load_arrived[s], phase);

            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[s]);
            warp::copy(max_vec_last, max_vec);
            warpgroup::mma_async_wait();

            warp::mul(att_block, att_block, softmax_scale_log2);

            warpgroup::load(pb_reg, pb_smem[warpgroupid][s]);
            warp::mul(pb_reg, pb_reg, LOG2E);
            warp::add(att_block, att_block, pb_reg);

            warp::load(rm_reg, rm_smem[s]);
            warp::mul(rm_reg, rm_reg, LOG2E);
            warp::add_col(att_block, att_block, rm_reg);

            warp::row_max(max_vec, att_block, max_vec);

            warp::sub(alpha, max_vec_last, max_vec);
            warp::exp2(alpha, alpha);

            warp::sub_row(att_block, att_block, max_vec);
            warp::exp2(att_block, att_block);

            warp::mul(norm_vec, norm_vec, alpha);
            warp::row_sum(norm_vec, att_block, norm_vec);

            warp::copy(att_block_mma, att_block);
            warp::mul_row(o_reg, o_reg, alpha);

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[s]);
            warpgroup::mma_async_wait();

            if (warpgroup::laneid() == 0) arrive(compute_done[s], 1);
        }

        warp::div_row(o_reg, o_reg, norm_vec);

        warpgroup::store(o_smem[warpgroupid], o_reg);
        warpgroup::sync(warpgroupid + 4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {batch_msa_idx, head_idx, seq_idx + warpgroupid, 0};
            warp::tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        warp::mul(max_vec, max_vec, 0.69314718056f);
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

#include "common_wrapper.cuh"
