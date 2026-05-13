"""torch.autograd.Function-style wrapper around the experimental `best`
forward kernel. Mirrors the API of TKEvoAttention (in the parent dir) but
only implements the forward pass — the backward returns NotImplementedError.

This is intentionally drop-in to make it easy to A/B against the production
TKEvoAttention layer.
"""

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _C_best as _C   # built from best.cu


TK_SUPPORTED_DIMS = (64, 128)


def _next_supported_dim(d: int) -> int:
    for t in TK_SUPPORTED_DIMS:
        if d <= t:
            return t
    raise ValueError(f"head_dim {d} > max supported {TK_SUPPORTED_DIMS[-1]}")


class TKEvoAttentionBest(torch.autograd.Function):
    """ThunderKittens EvoAttention (experimental `best` forward kernel).

    forward(Q, K, V, res_mask, pair_bias) -> O

    Input shapes (Triton / MegaFold convention):
        Q, K, V   : (B, N_SEQ, N_CTX, H, D)        bf16 (other floats cast)
        res_mask  : (B, N_SEQ, 1, 1, N_CTX)
        pair_bias : (B, 1, H, N_CTX, N_CTX)

    Output:
        O : (B, N_SEQ, N_CTX, H, D) bf16
    """

    @staticmethod
    def forward(ctx, Q, K, V, res_mask, pair_bias):
        B, N_SEQ, N_CTX, H, D = Q.shape
        bf16 = torch.bfloat16

        def to_tk_qkv(x):
            x = x.to(bf16) if x.dtype != bf16 else x
            x = x.transpose(-2, -3).contiguous()
            return x.view(B * N_SEQ, H, N_CTX, D)

        Q_tk = to_tk_qkv(Q)
        K_tk = to_tk_qkv(K)
        V_tk = to_tk_qkv(V)

        pair_bias_tk = pair_bias.squeeze(1).contiguous().to(bf16)
        res_mask_tk = res_mask.reshape(B * N_SEQ, 1, 1, N_CTX).contiguous().to(bf16)

        padded_D = _next_supported_dim(D)
        if padded_D != D:
            pad = padded_D - D
            Q_tk = F.pad(Q_tk, (0, pad), value=0.0).contiguous()
            K_tk = F.pad(K_tk, (0, pad), value=0.0).contiguous()
            V_tk = F.pad(V_tk, (0, pad), value=0.0).contiguous()

        softmax_scale = 1.0 / (D ** 0.5)

        O_pad, _L = _C.evoattention_forward(
            Q_tk, K_tk, V_tk, pair_bias_tk, res_mask_tk, N_SEQ, softmax_scale
        )

        O = O_pad[..., :D].contiguous()
        O = O.view(B, N_SEQ, H, N_CTX, D).transpose(-2, -3).contiguous()
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError(
            "TKEvoAttentionBest is forward-only. Use the production "
            "TKEvoAttention (parent dir) for forward+backward."
        )
