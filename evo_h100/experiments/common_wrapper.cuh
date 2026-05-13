// Shared forward-only wrapper boilerplate for evo_h100 experiments.
// Each candidate .cu file is expected to provide a kernel template
// `void evo_fwd_ker<D>(...)` and a `evo_fwd_tile_dims<D>` struct
// exposing tile_width / qo_height / kv_height / consumer_warpgroups.
//
// Include this once at the bottom of each candidate .cu file.

#ifndef EVO_FWD_COMMON_WRAPPER_CUH
#define EVO_FWD_COMMON_WRAPPER_CUH

#include "pyutils/torchutils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <cmath>

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
    TORCH_CHECK(q.scalar_type() == at::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(v.scalar_type() == at::kBFloat16, "V must be bfloat16");
    TORCH_CHECK(pair_bias.scalar_type() == at::kBFloat16, "pair_bias must be bfloat16");
    TORCH_CHECK(res_mask.scalar_type() == at::kBFloat16, "res_mask must be bfloat16");

    const auto batch_msa = q.size(0);
    const auto heads     = q.size(1);
    const auto seq_len   = q.size(2);
    const auto head_dim  = q.size(3);

    TORCH_CHECK(batch_msa % n_seq == 0, "batch_msa must be divisible by N_SEQ");
    const int64_t batch = batch_msa / n_seq;

    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "Only head_dim=64 or 128 supported; got ", head_dim);

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

    const float softmax_scale = (softmax_scale_override > 0.0)
        ? static_cast<float>(softmax_scale_override)
        : 1.0f / std::sqrt(static_cast<float>(head_dim));

    auto launch = [&](auto D_tag) {
        constexpr int D = decltype(D_tag)::value;
        constexpr int D_CW = evo_fwd_tile_dims<D>::consumer_warpgroups;
        constexpr int D_QO = evo_fwd_tile_dims<D>::qo_height;
        constexpr int D_KV = evo_fwd_tile_dims<D>::kv_height;
        constexpr int D_NUM_WORKERS = evo_num_workers<D>();

        TORCH_CHECK(seq_len % (D_CW * D_QO) == 0,
                    "SEQ_LEN must be divisible by consumer_warpgroups*qo_height");
        TORCH_CHECK(seq_len % D_KV == 0,
                    "SEQ_LEN must be divisible by kv_height");

        using globals = evo_fwd_globals<D>;

        typename globals::q_gl  qg_arg{d_q,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::k_gl  kg_arg{d_k,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::v_gl  vg_arg{d_v,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::o_gl  og_arg{d_o,  (unsigned)batch_msa, (unsigned)heads, (unsigned)seq_len, (unsigned)D};
        typename globals::l_gl  lg_arg{d_l,  (unsigned)batch_msa, (unsigned)heads, 1u,                 (unsigned)seq_len};
        typename globals::pb_gl pbg_arg{d_pb,(unsigned)batch,     (unsigned)heads, (unsigned)seq_len,  (unsigned)seq_len};
        typename globals::rm_gl rmg_arg{d_rm,(unsigned)batch_msa, 1u,              1u,                 (unsigned)seq_len};

        globals g{qg_arg, kg_arg, vg_arg, pbg_arg, rmg_arg, lg_arg, og_arg,
                  static_cast<int>(seq_len), static_cast<int>(n_seq), softmax_scale};

        // If the candidate's tile_dims defines a blocks_sm member, divide
        // the smem budget so two (or more) CTAs actually fit on one SM.
        // Otherwise default to ~MAX_SHARED-1024 (single CTA per SM).
        constexpr int D_BLOCKS_SM = []() constexpr {
            if constexpr (requires { evo_fwd_tile_dims<D>::blocks_sm; }) {
                return evo_fwd_tile_dims<D>::blocks_sm;
            } else {
                return 1;
            }
        }();
        auto mem_size = (kittens::MAX_SHARED_MEMORY / D_BLOCKS_SM) - 1024;

        dim3 grid(seq_len / (D_CW * D_QO),
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("evoattention_forward", &evoattention_forward,
          pybind11::arg("q"),
          pybind11::arg("k"),
          pybind11::arg("v"),
          pybind11::arg("pair_bias"),
          pybind11::arg("res_mask"),
          pybind11::arg("n_seq"),
          pybind11::arg("softmax_scale") = 0.0,
          "EvoAttention forward (experiments).");
}

#endif // EVO_FWD_COMMON_WRAPPER_CUH
