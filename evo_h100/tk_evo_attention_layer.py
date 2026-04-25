"""torch.autograd.Function wrapper around the TK EvoAttention forward kernel.

Forward only for now; backward raises NotImplementedError.

Input shapes (mirror the Triton MegaFold API):
    Q, K, V   : (B, N_SEQ, N_CTX, H, D)          bf16 (other float dtypes cast)
    res_mask  : (B, N_SEQ, 1, 1, N_CTX)          any float/int/bool; cast to bf16
    pair_bias : (B, 1, H, N_CTX, N_CTX)          any float; cast to bf16

Output:
    O : (B, N_SEQ, N_CTX, H, D)                  bf16
"""

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _C  # built tk_evoattention extension


# Head-dim values the CUDA kernel supports natively. Anything else is
# zero-padded up to the next supported value on the way in; output is sliced
# back on the way out.
TK_SUPPORTED_DIMS = (64, 128)


def _next_supported_dim(d: int) -> int:
    for t in TK_SUPPORTED_DIMS:
        if d <= t:
            return t
    raise ValueError(
        f"head_dim {d} exceeds max supported TK head_dim {TK_SUPPORTED_DIMS[-1]}"
    )


class TKEvoAttention(torch.autograd.Function):
    """ThunderKittens Evo/Triangle attention.

    forward(Q, K, V, res_mask, pair_bias) -> O
    """

    @staticmethod
    def forward(ctx, Q, K, V, res_mask, pair_bias):
        # ---- Shape & dtype setup ---------------------------------------------
        B, N_SEQ, N_CTX, H, D = Q.shape
        assert K.shape == (B, N_SEQ, N_CTX, H, D), f"K shape mismatch: {K.shape}"
        assert V.shape == (B, N_SEQ, N_CTX, H, D), f"V shape mismatch: {V.shape}"
        assert res_mask.shape == (B, N_SEQ, 1, 1, N_CTX), (
            f"res_mask shape mismatch: expected (B,N_SEQ,1,1,N_CTX)={(B,N_SEQ,1,1,N_CTX)}, got {res_mask.shape}"
        )
        assert pair_bias.shape == (B, 1, H, N_CTX, N_CTX), (
            f"pair_bias shape mismatch: expected (B,1,H,N,N)={(B,1,H,N_CTX,N_CTX)}, got {pair_bias.shape}"
        )

        bf16 = torch.bfloat16

        # Q/K/V: (B, N_SEQ, N_CTX, H, D) -> (B, N_SEQ, H, N_CTX, D) -> (B*N_SEQ, H, N_CTX, D)
        def to_tk_qkv(x):
            x = x.to(bf16) if x.dtype != bf16 else x
            x = x.transpose(-2, -3).contiguous()
            return x.view(B * N_SEQ, H, N_CTX, D)

        Q_tk = to_tk_qkv(Q)
        K_tk = to_tk_qkv(K)
        V_tk = to_tk_qkv(V)

        # pair_bias: (B, 1, H, N, N) -> (B, H, N, N) bf16
        pair_bias_tk = pair_bias.squeeze(1).contiguous().to(bf16)

        # res_mask: (B, N_SEQ, 1, 1, N) -> (B*N_SEQ, 1, 1, N) bf16
        res_mask_tk = res_mask.reshape(B * N_SEQ, 1, 1, N_CTX).contiguous().to(bf16)

        # ---- Pad D to the next supported kernel head_dim ---------------------
        padded_D = _next_supported_dim(D)
        if padded_D != D:
            pad = padded_D - D
            Q_tk = F.pad(Q_tk, (0, pad), value=0.0).contiguous()
            K_tk = F.pad(K_tk, (0, pad), value=0.0).contiguous()
            V_tk = F.pad(V_tk, (0, pad), value=0.0).contiguous()

        # softmax_scale is 1/sqrt(true_D), not 1/sqrt(padded_D)
        softmax_scale = 1.0 / (D ** 0.5)

        O_pad, L = _C.evoattention_forward(
            Q_tk, K_tk, V_tk, pair_bias_tk, res_mask_tk, N_SEQ, softmax_scale
        )

        # Slice off padded D columns, reshape/transpose back to caller layout
        O = O_pad[..., :D].contiguous()
        O = O.view(B, N_SEQ, H, N_CTX, D).transpose(-2, -3).contiguous()

        # Save for (future) backward
        ctx.save_for_backward(Q_tk, K_tk, V_tk, pair_bias_tk, res_mask_tk, O_pad, L)
        ctx.softmax_scale = softmax_scale
        ctx.true_D = D
        ctx.padded_D = padded_D
        ctx.N_SEQ = N_SEQ
        ctx.B = B
        ctx.H = H
        ctx.N_CTX = N_CTX

        return O

    @staticmethod
    def backward(ctx, dO):
        # ctx layout (set in forward):
        #   Q_tk, K_tk, V_tk, pair_bias_tk, res_mask_tk : padded (if needed), (B*N_SEQ, H, N, padded_D) bf16
        #   O_pad : (B*N_SEQ, H, N, padded_D) bf16
        #   L     : (B*N_SEQ, H, 1, N) fp32
        Q_tk, K_tk, V_tk, pair_bias_tk, res_mask_tk, O_pad, L = ctx.saved_tensors
        B       = ctx.B
        H       = ctx.H
        N_CTX   = ctx.N_CTX
        N_SEQ   = ctx.N_SEQ
        true_D  = ctx.true_D
        padded_D = ctx.padded_D
        softmax_scale = ctx.softmax_scale

        bf16 = torch.bfloat16

        # dO arrives in caller layout (B, N_SEQ, N_CTX, H, D=true_D); bring to
        # (B*N_SEQ, H, N_CTX, padded_D) with the same pad we did in forward.
        dO_tk = dO
        if dO_tk.dtype != bf16:
            dO_tk = dO_tk.to(bf16)
        dO_tk = dO_tk.transpose(-2, -3).contiguous().view(B * N_SEQ, H, N_CTX, true_D)
        if padded_D != true_D:
            pad = padded_D - true_D
            dO_tk = F.pad(dO_tk, (0, pad), value=0.0).contiguous()

        dQ_pad, dK_pad, dV_pad, d_pair_bias = _C.evoattention_backward(
            Q_tk, K_tk, V_tk, pair_bias_tk, res_mask_tk,
            O_pad, L, dO_tk,
            N_SEQ, softmax_scale,
        )
        # dQ_pad, dK_pad, dV_pad: (B*N_SEQ, H, N_CTX, padded_D) fp32
        # d_pair_bias:            (B, H, N_CTX, N_CTX)          fp32

        # Slice off padded D columns, cast to bf16 FIRST (halves the bytes moved
        # by the subsequent transpose+contiguous), then reshape/transpose back.
        # This knocks ~30 μs off per-call wrapper overhead at N_CTX=384 D=64.
        def from_tk_grad(x):
            x = x[..., :true_D].contiguous()     # fp32, (B*N_SEQ, H, N, true_D)
            x = x.to(bf16)                        # bf16 cast, same layout
            x = x.view(B, N_SEQ, H, N_CTX, true_D).transpose(-2, -3).contiguous()
            return x

        dQ = from_tk_grad(dQ_pad)
        dK = from_tk_grad(dK_pad)
        dV = from_tk_grad(dV_pad)

        # pair_bias caller layout is (B, 1, H, N, N). We received (B, H, N, N) fp32.
        d_pair_bias = d_pair_bias.unsqueeze(1).to(bf16)  # (B, 1, H, N, N) bf16

        # forward signature: forward(ctx, Q, K, V, res_mask, pair_bias) -> O
        # so backward returns one grad per input (+ None for non-diff tensors).
        return dQ, dK, dV, None, d_pair_bias
