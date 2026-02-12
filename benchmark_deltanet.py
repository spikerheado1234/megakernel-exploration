import torch
import torch.nn as nn
import time
import mirage as mi

# ============================================================
# 1. BASELINE PYTORCH IMPLEMENTATION
# ============================================================

def chunk_batched_delta_rule_forward(Q, K, V, beta, C):
    """
    Q, K, V: (B, L, d)
    beta:    (B, L, 1)
    C:       chunk size, must divide L
    """
    B, L, d = Q.shape
    Q, K, V = map(lambda x: x.reshape(B, -1, C, d), [Q, K, V])
    beta = beta.reshape(B, -1, C)
    K_beta = K * beta.unsqueeze(-1)
    V_beta = V * beta.unsqueeze(-1)

    mask = torch.triu(torch.ones(C, C, device=Q.device), diagonal=0).bool()

    K_t = torch.transpose(K, 2, 3)
    T = -(K_beta[:] @ K_t[:]).masked_fill(mask, 0)

    for k in range(L // C):
        for i in range(1, C):
            T_new = T.clone()
            T_new[:, k, i, :i] = T[:, k, i, :i] + (T[:, k, i, :, None] * T[:, k, :, :i]).sum(-2)
            T = T_new
        T[:, k] = T[:, k] + torch.eye(C, device=T.device)

    W = T @ K_beta
    U = T @ V_beta

    S = torch.zeros((B, d, d), device=Q.device, dtype=Q.dtype)
    O = torch.empty_like(V)
    mask = torch.triu(torch.ones(C, C, device=Q.device), diagonal=1).bool()

    for i in range(L // C):
        q_i, k_i, w_i = Q[:, i], K[:, i], W[:, i]
        u_i = U[:, i] - w_i @ S
        o_inter = q_i @ S
        A_i = (q_i @ k_i.transpose(1, 2)).masked_fill(mask, 0)
        o_intra = A_i @ u_i
        S = S + k_i.transpose(1, 2) @ u_i
        O[:, i] = o_intra + o_inter

    return O.reshape(B, L, d)


# ============================================================
# 2. MIRAGE KERNEL BUILDERS
# ============================================================

def make_dual_matmul_kernel(B, C, d):
    """
    Fuse q_i @ S and w_i @ S (two matmuls sharing the same RHS).
    q_i, w_i: (B, C, d)
    S:        (B, d, d)
    outputs:  (B, C, d), (B, C, d)
    """
    graph = mi.new_kernel_graph()
    q = graph.new_input(dims=(B, C, d), dtype=mi.float16)
    w = graph.new_input(dims=(B, C, d), dtype=mi.float16)
    S = graph.new_input(dims=(B, d, d), dtype=mi.float16)
    o_inter = graph.matmul(q, S)
    u_correction = graph.matmul(w, S)
    graph.mark_output(o_inter)
    graph.mark_output(u_correction)
    return graph.superoptimize()


def make_attn_matmul_kernel(B, C, d):
    """
    Compute A_i = q_i @ k_i^T.
    q_i:  (B, C, d)
    k_iT: (B, d, C)  <-- pre-transposed in PyTorch before calling
    output: (B, C, C)
    """
    graph = mi.new_kernel_graph()
    q = graph.new_input(dims=(B, C, d), dtype=mi.float16)
    kt = graph.new_input(dims=(B, d, C), dtype=mi.float16)
    A = graph.matmul(q, kt)
    graph.mark_output(A)
    return graph.superoptimize()


def make_state_update_kernel(B, C, d):
    """
    Fuse: S_new = S + k_i^T @ u_i.
    S:    (B, d, d)
    k_iT: (B, d, C)  <-- pre-transposed in PyTorch before calling
    u_i:  (B, C, d)
    output: (B, d, d)
    """
    graph = mi.new_kernel_graph()
    S = graph.new_input(dims=(B, d, d), dtype=mi.float16)
    kt = graph.new_input(dims=(B, d, C), dtype=mi.float16)
    u = graph.new_input(dims=(B, C, d), dtype=mi.float16)
    update = graph.matmul(kt, u)
    S_new = graph.add(S, update)
    graph.mark_output(S_new)
    return graph.superoptimize()


# ============================================================
# 3. MIRAGE-OPTIMIZED DELTA RULE
# ============================================================

def chunk_delta_rule_mirage(Q, K, V, beta, C,
                            dual_mm_kernel,
                            attn_kernel,
                            state_kernel):
    B, L, d = Q.shape
    Q, K, V = map(lambda x: x.reshape(B, -1, C, d), [Q, K, V])
    beta = beta.reshape(B, -1, C)
    K_beta = K * beta.unsqueeze(-1)
    V_beta = V * beta.unsqueeze(-1)

    # T matrix computation — forward substitution stays in Python
    mask = torch.triu(torch.ones(C, C, device=Q.device), diagonal=0).bool()
    K_t = K.transpose(2, 3)
    T = -(K_beta @ K_t).masked_fill(mask, 0)

    for k in range(L // C):
        for i in range(1, C):
            T_new = T.clone()
            T_new[:, k, i, :i] = T[:, k, i, :i] + (T[:, k, i, :, None] * T[:, k, :, :i]).sum(-2)
            T = T_new
        T[:, k] = T[:, k] + torch.eye(C, device=T.device)

    W = T @ K_beta
    U = T @ V_beta

    S = torch.zeros((B, d, d), dtype=Q.dtype, device=Q.device)
    O = torch.empty_like(V)
    causal_mask = torch.triu(torch.ones(C, C, device=Q.device), diagonal=1).bool()

    # Main chunk loop — uses Mirage kernels for the heavy matmuls
    for i in range(L // C):
        q_i, k_i, w_i = Q[:, i], K[:, i], W[:, i]

        # Pre-transpose k_i in PyTorch (zero-cost view, contiguous for Mirage)
        k_iT = k_i.transpose(1, 2).contiguous()  # (B, d, C)

        # Fused: o_inter = q_i @ S, wS = w_i @ S
        dual_out = dual_mm_kernel(inputs=[q_i, w_i, S])
        if isinstance(dual_out, (list, tuple)):
            o_inter, wS = dual_out[0], dual_out[1]
        else:
            o_inter, wS = dual_out, None

        u_i = U[:, i] - wS

        # Attention scores: A_i = q_i @ k_i^T
        attn_out = attn_kernel(inputs=[q_i, k_iT])
        if isinstance(attn_out, (list, tuple)):
            A_i = attn_out[0]
        else:
            A_i = attn_out
        A_i = A_i.masked_fill(causal_mask, 0)

        # Small matmul, fine in torch
        o_intra = A_i @ u_i

        # Fused state update: S = S + k_i^T @ u_i
        state_out = state_kernel(inputs=[S, k_iT, u_i])
        if isinstance(state_out, (list, tuple)):
            S = state_out[0]
        else:
            S = state_out

        O[:, i] = o_intra + o_inter

    return O.reshape(B, L, d)


# ============================================================
# 4. BENCHMARK HARNESS
# ============================================================

def benchmark_fn(fn, warmup=5, repeats=20):
    """Time a GPU function using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


def main():
    # ----- Config -----
    B = 2        # batch size
    L = 128      # sequence length
    d = 64       # embedding dimension
    C = 16       # chunk size (must divide L)

    warmup = 5
    repeats = 50

    device = "cuda"
    dtype = torch.float16

    print("=" * 60)
    print("DeltaNet: PyTorch Baseline vs Mirage-Optimized")
    print("=" * 60)
    print(f"  B={B}, L={L}, d={d}, C={C}")
    print(f"  dtype={dtype}, device={device}")
    print(f"  warmup={warmup}, repeats={repeats}")
    print()

    # ----- Generate inputs -----
    torch.manual_seed(42)
    Q = torch.randn(B, L, d, device=device, dtype=dtype)
    K = torch.randn(B, L, d, device=device, dtype=dtype)
    V = torch.randn(B, L, d, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, L, 1, device=device, dtype=dtype))

    # ----- Build Mirage kernels -----
    print("Building Mirage kernels (superoptimize)...")
    print("  This may take a few minutes on first run...")
    t0 = time.time()

    dual_mm_kernel = make_dual_matmul_kernel(B, C, d)
    print(f"  dual_matmul kernel built ({time.time() - t0:.1f}s)")

    t1 = time.time()
    attn_kernel = make_attn_matmul_kernel(B, C, d)
    print(f"  attn_matmul kernel built ({time.time() - t1:.1f}s)")

    t2 = time.time()
    state_kernel = make_state_update_kernel(B, C, d)
    print(f"  state_update kernel built ({time.time() - t2:.1f}s)")

    build_time = time.time() - t0
    print(f"  Total kernel build time: {build_time:.2f}s")
    print()

    # ----- Correctness check -----
    print("-" * 60)
    print("CORRECTNESS CHECK")
    print("-" * 60)

    with torch.no_grad():
        out_baseline = chunk_batched_delta_rule_forward(
            Q.clone(), K.clone(), V.clone(), beta.clone(), C
        )
        out_mirage = chunk_delta_rule_mirage(
            Q.clone(), K.clone(), V.clone(), beta.clone(), C,
            dual_mm_kernel, attn_kernel, state_kernel
        )

    abs_diff = (out_baseline - out_mirage).abs()
    rel_diff = abs_diff / (out_baseline.abs() + 1e-8)

    print(f"  Output shape:        {out_baseline.shape}")
    print(f"  Max absolute error:  {abs_diff.max().item():.6e}")
    print(f"  Mean absolute error: {abs_diff.mean().item():.6e}")
    print(f"  Max relative error:  {rel_diff.max().item():.6e}")
    print(f"  Mean relative error: {rel_diff.mean().item():.6e}")

    # fp16 tolerance
    atol = 1e-2
    if abs_diff.max().item() < atol:
        print(f"  PASS (max abs error < {atol})")
    else:
        print(f"  FAIL (max abs error >= {atol})")
        print("     First 5 mismatches:")
        mismatch_mask = abs_diff > atol
        idxs = mismatch_mask.nonzero()[:5]
        for idx in idxs:
            t = tuple(idx.tolist())
            print(f"       idx={t}: "
                  f"baseline={out_baseline[t].item():.6f}, "
                  f"mirage={out_mirage[t].item():.6f}, "
                  f"diff={abs_diff[t].item():.6e}")
    print()

    # ----- Speed benchmark -----
    print("-" * 60)
    print("SPEED BENCHMARK")
    print("-" * 60)

    def run_baseline():
        return chunk_batched_delta_rule_forward(Q, K, V, beta, C)

    times_baseline = benchmark_fn(run_baseline, warmup=warmup, repeats=repeats)
    mean_bl = sum(times_baseline) / len(times_baseline)
    min_bl = min(times_baseline)
    std_bl = (sum((t - mean_bl) ** 2 for t in times_baseline) / len(times_baseline)) ** 0.5

    print(f"  PyTorch baseline:")
    print(f"    mean = {mean_bl:.3f} ms")
    print(f"    min  = {min_bl:.3f} ms")
    print(f"    std  = {std_bl:.3f} ms")
    print()

    def run_mirage():
        return chunk_delta_rule_mirage(Q, K, V, beta, C,
                                       dual_mm_kernel, attn_kernel, state_kernel)

    times_mirage = benchmark_fn(run_mirage, warmup=warmup, repeats=repeats)
    mean_mi = sum(times_mirage) / len(times_mirage)
    min_mi = min(times_mirage)
    std_mi = (sum((t - mean_mi) ** 2 for t in times_mirage) / len(times_mirage)) ** 0.5

    print(f"  Mirage optimized:")
    print(f"    mean = {mean_mi:.3f} ms")
    print(f"    min  = {min_mi:.3f} ms")
    print(f"    std  = {std_mi:.3f} ms")
    print()

    # ----- Summary -----
    print("-" * 60)
    print("SUMMARY")
    print("-" * 60)
    speedup = mean_bl / mean_mi if mean_mi > 0 else float("inf")
    print(f"  Speedup (mean): {speedup:.2f}x")
    if speedup > 1.0:
        print(f"  Mirage is {speedup:.2f}x faster")
    else:
        print(f"  Baseline is {1.0 / speedup:.2f}x faster")
    print()


if __name__ == "__main__":
    main()
