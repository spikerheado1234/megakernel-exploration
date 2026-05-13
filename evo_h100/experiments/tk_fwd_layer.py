"""Forward-only wrapper around an experimental TK EvoAttention kernel.

Lets you swap in different compiled candidate modules:
    import _C_v1_baseline as m
    layer = make_fwd_apply(m)
    O = layer(Q, K, V, res_mask, pair_bias)
"""

import os
import sys
import importlib

import torch
import torch.nn.functional as F


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


TK_SUPPORTED_DIMS = (64, 128)


def _next_supported_dim(d: int) -> int:
    for t in TK_SUPPORTED_DIMS:
        if d <= t:
            return t
    raise ValueError(f"head_dim {d} > max supported {TK_SUPPORTED_DIMS[-1]}")


def load_candidate(name: str):
    """Import a candidate module (e.g. 'v1_baseline'). Returns the python module."""
    mod_name = f"_C_{name}"
    return importlib.import_module(mod_name)


def make_fwd_apply(module):
    """Return a callable that mimics TKEvoAttention.apply but only does forward."""

    def fwd(Q, K, V, res_mask, pair_bias):
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

        O_pad, _L = module.evoattention_forward(
            Q_tk, K_tk, V_tk, pair_bias_tk, res_mask_tk, N_SEQ, softmax_scale
        )

        O = O_pad[..., :D].contiguous()
        O = O.view(B, N_SEQ, H, N_CTX, D).transpose(-2, -3).contiguous()
        return O

    return fwd
