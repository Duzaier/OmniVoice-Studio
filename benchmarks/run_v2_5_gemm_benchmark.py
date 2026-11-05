"""
OmniVoice V2.5 - CPU Kernel & GEMM Micro-Benchmark
Forensic Read-Only Analysis of CPU Linear, Attention, and Activation Kernels
"""

import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_backend_info():
    info = {
        "torch_version": torch.__version__,
        "mkl_available": torch.backends.mkl.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
        "cpu_capability": torch._C._cpu_capability() if hasattr(torch._C, "_cpu_capability") else "N/A",
        "cpu_count": torch.get_num_threads(),
    }
    return info

def benchmark_op(func, warmup=10, iters=50):
    for _ in range(warmup):
        func()
    
    t0 = time.perf_counter()
    for _ in range(iters):
        func()
    t1 = time.perf_counter()
    return ((t1 - t0) / iters) * 1000.0 # ms

def run_micro_benchmarks(threads_list=[1, 4, 6, 8]):
    # Qwen3 dimensions
    L = 410       # Token sequence length
    D = 1024      # Hidden dimension
    FFN = 3072    # Intermediate dimension
    N_HEADS = 16  # Query heads
    KV_HEADS = 8  # KV heads (GQA)
    HEAD_DIM = 64 # Head dim
    NUM_LAYERS = 28 # Number of transformer layers

    results = {}

    for t in threads_list:
        torch.set_num_threads(t)
        results[t] = {}

        # 1. QKV Linear Projection: [410, 1024] -> [410, 2048]
        x_qkv = torch.randn(1, L, D, dtype=torch.float32)
        w_qkv = nn.Linear(D, (N_HEADS + 2 * KV_HEADS) * HEAD_DIM, bias=False)
        flops_qkv = 2 * L * D * ((N_HEADS + 2 * KV_HEADS) * HEAD_DIM)
        ms_qkv = benchmark_op(lambda: w_qkv(x_qkv))
        gflops_qkv = (flops_qkv / (ms_qkv / 1000.0)) / 1e9

        # 2. Attention Output Linear Projection: [410, 1024] -> [410, 1024]
        x_out = torch.randn(1, L, D, dtype=torch.float32)
        w_out = nn.Linear(D, D, bias=False)
        flops_out = 2 * L * D * D
        ms_out = benchmark_op(lambda: w_out(x_out))
        gflops_out = (flops_out / (ms_out / 1000.0)) / 1e9

        # 3. MLP Gate+Up Projection: [410, 1024] -> [410, 6144]
        x_mlp = torch.randn(1, L, D, dtype=torch.float32)
        w_gate_up = nn.Linear(D, 2 * FFN, bias=False)
        flops_gate_up = 2 * L * D * (2 * FFN)
        ms_gate_up = benchmark_op(lambda: w_gate_up(x_mlp))
        gflops_gate_up = (flops_gate_up / (ms_gate_up / 1000.0)) / 1e9

        # 4. MLP Down Projection: [410, 3072] -> [410, 1024]
        x_down = torch.randn(1, L, FFN, dtype=torch.float32)
        w_down = nn.Linear(FFN, D, bias=False)
        flops_down = 2 * L * FFN * D
        ms_down = benchmark_op(lambda: w_down(x_down))
        gflops_down = (flops_down / (ms_down / 1000.0)) / 1e9

        # 5. Scaled Dot-Product Attention (SDPA): Q [1, 16, 410, 64], K [1, 8, 410, 64] -> expand K to 16
        q = torch.randn(1, N_HEADS, L, HEAD_DIM, dtype=torch.float32)
        k = torch.randn(1, KV_HEADS, L, HEAD_DIM, dtype=torch.float32).repeat_interleave(2, dim=1)
        v = torch.randn(1, KV_HEADS, L, HEAD_DIM, dtype=torch.float32).repeat_interleave(2, dim=1)
        # FLOPs for Attention: 2 * (Q @ K.T) + 2 * (Attn @ V) = 4 * N_HEADS * L * L * HEAD_DIM
        flops_sdpa = 4 * N_HEADS * L * L * HEAD_DIM
        ms_sdpa = benchmark_op(lambda: F.scaled_dot_product_attention(q, k, v))
        gflops_sdpa = (flops_sdpa / (ms_sdpa / 1000.0)) / 1e9

        # 6. RMSNorm: [1, 410, 1024]
        x_norm = torch.randn(1, L, D, dtype=torch.float32)
        weight_norm = torch.ones(D, dtype=torch.float32)
        def run_rmsnorm():
            variance = x_norm.pow(2).mean(-1, keepdim=True)
            return x_norm * torch.rsqrt(variance + 1e-6) * weight_norm
        ms_rmsnorm = benchmark_op(run_rmsnorm)

        # 7. SiLU Activation & Gating: [1, 410, 3072]
        gate = torch.randn(1, L, FFN, dtype=torch.float32)
        up = torch.randn(1, L, FFN, dtype=torch.float32)
        def run_silu_gate():
            return F.silu(gate) * up
        ms_silu_gate = benchmark_op(run_silu_gate)

        # Theoretical 1-layer Transformer block time (sum of ops)
        ms_layer = (ms_qkv + ms_out + ms_sdpa + ms_gate_up + ms_down + 2 * ms_rmsnorm + ms_silu_gate)
        # Full 28 layers * 4 steps
        ms_4steps = ms_layer * NUM_LAYERS * 4

        results[t] = {
            "qkv_proj_ms": ms_qkv,
            "qkv_gflops": gflops_qkv,
            "out_proj_ms": ms_out,
            "out_gflops": gflops_out,
            "gate_up_proj_ms": ms_gate_up,
            "gate_up_gflops": gflops_gate_up,
            "down_proj_ms": ms_down,
            "down_gflops": gflops_down,
            "sdpa_ms": ms_sdpa,
            "sdpa_gflops": gflops_sdpa,
            "rmsnorm_ms": ms_rmsnorm,
            "silu_gate_ms": ms_silu_gate,
            "single_layer_est_ms": ms_layer,
            "four_steps_est_s": ms_4steps / 1000.0
        }

    return results

if __name__ == "__main__":
    backend = get_backend_info()
    print("Backend info:", json.dumps(backend, indent=2))
    print("\nRunning micro-benchmarks across threads [1, 4, 6, 8]...")
    res = run_micro_benchmarks([1, 4, 6, 8])
    print(json.dumps(res, indent=2))
