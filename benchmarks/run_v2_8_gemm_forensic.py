"""
OmniVoice V2.8 - GEMM Backend & Threading Forensic Benchmark
Low-level CPU investigation of BLAS/MKL, threading pools, memory layouts, and GEMM bottleneck
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

def get_env_forensics():
    env_vars = {
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "Unset"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "Unset"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "Unset"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "Unset"),
        "MKL_DYNAMIC": os.environ.get("MKL_DYNAMIC", "Unset"),
        "OMP_PROC_BIND": os.environ.get("OMP_PROC_BIND", "Unset"),
        "OMP_SCHEDULE": os.environ.get("OMP_SCHEDULE", "Unset"),
        "TORCH_NUM_THREADS": os.environ.get("TORCH_NUM_THREADS", "Unset"),
    }
    
    config_show = torch.__config__.show()
    parallel_info = torch.__config__.parallel_info() if hasattr(torch.__config__, "parallel_info") else "N/A"
    
    info = {
        "torch_version": torch.__version__,
        "mkl_available": torch.backends.mkl.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
        "default_intraop_threads": torch.get_num_threads(),
        "default_interop_threads": torch.get_num_interop_threads(),
        "env_vars": env_vars,
        "config_show": config_show,
        "parallel_info": parallel_info
    }
    return info

def benchmark_gemm(A, B, is_linear=False, warmup=15, iters=30):
    if is_linear:
        # B is weight [out_features, in_features]
        op = lambda: F.linear(A, B)
    else:
        op = lambda: torch.mm(A, B)

    for _ in range(warmup):
        _ = op()

    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = op()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # ms

    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    min_ms = float(np.min(latencies))
    std_ms = float(np.std(latencies))
    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "std_ms": std_ms,
        "latencies": latencies
    }

def run_gemm_forensic_suite():
    print("=" * 80, flush=True)
    print("OMNIVOICE V2.8 — GEMM BACKEND & THREADING FORENSIC BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    # 1. Environment Forensics
    print("\n[STEP 1] ENVIRONMENT & RUNTIME FORENSICS", flush=True)
    env_info = get_env_forensics()
    print(f"PyTorch Version   : {env_info['torch_version']}")
    print(f"MKL Available     : {env_info['mkl_available']}")
    print(f"oneDNN Available  : {env_info['mkldnn_available']}")
    print(f"OpenMP Available  : {env_info['openmp_available']}")
    print(f"Intra-op Threads  : {env_info['default_intraop_threads']}")
    print(f"Interop Threads   : {env_info['default_interop_threads']}")
    print(f"Environment Vars  : {json.dumps(env_info['env_vars'], indent=2)}")
    print("\ntorch.__config__.show():\n", env_info['config_show'])
    print("\ntorch.__config__.parallel_info():\n", env_info['parallel_info'])

    # 2. Real Model Shapes & Memory Layout
    print("\n[STEP 2] REAL GEMM SHAPES & MEMORY LAYOUT INSPECTION", flush=True)
    # Real Qwen3 dimensions
    M = 410
    D = 1024
    FFN = 3072

    # Shape A: QKV Proj: [410, 1024] x [1024, 2048]
    A_qkv = torch.randn(M, D, dtype=torch.float32)
    W_qkv = torch.randn(2048, D, dtype=torch.float32) # nn.Linear weight format
    B_qkv = W_qkv.t().contiguous()                     # torch.mm format

    # Shape B: MLP Gate+Up Proj: [410, 1024] x [1024, 6144]
    A_mlp_up = torch.randn(M, D, dtype=torch.float32)
    W_mlp_up = torch.randn(2 * FFN, D, dtype=torch.float32)
    B_mlp_up = W_mlp_up.t().contiguous()

    # Shape C: MLP Down Proj: [410, 3072] x [3072, 1024]
    A_mlp_down = torch.randn(M, FFN, dtype=torch.float32)
    W_mlp_down = torch.randn(D, FFN, dtype=torch.float32)
    B_mlp_down = W_mlp_down.t().contiguous()

    def inspect_layout(name, x, w, b):
        print(f"\nLayout Inspection for {name}:")
        print(f"  • Input Tensor A : shape={list(x.shape)}, stride={x.stride()}, contiguous={x.is_contiguous()}, bytes={x.nbytes}")
        print(f"  • Weight Tensor W: shape={list(w.shape)}, stride={w.stride()}, contiguous={w.is_contiguous()}, bytes={w.nbytes}")
        print(f"  • Matmul Matrix B: shape={list(b.shape)}, stride={b.stride()}, contiguous={b.is_contiguous()}, bytes={b.nbytes}")
        print(f"  • W.t() view     : shape={list(w.t().shape)}, stride={w.t().stride()}, contiguous={w.t().is_contiguous()}")

    inspect_layout("QKV Projection", A_qkv, W_qkv, B_qkv)
    inspect_layout("MLP Gate+Up Projection", A_mlp_up, W_mlp_up, B_mlp_up)
    inspect_layout("MLP Down Projection", A_mlp_down, W_mlp_down, B_mlp_down)

    # 3. Thread Scaling Microbenchmarks [1, 2, 4, 6, 8]
    print("\n[STEP 3] GEMM THREAD SCALING MICROBENCHMARK [1, 2, 4, 6, 8]", flush=True)
    threads = [1, 2, 4, 6, 8]
    shapes = {
        "QKV_Proj": {"A": A_qkv, "W": W_qkv, "B": B_qkv, "flops": 2 * M * D * 2048, "bytes": (A_qkv.nbytes + W_qkv.nbytes + M * 2048 * 4)},
        "MLP_GateUp": {"A": A_mlp_up, "W": W_mlp_up, "B": B_mlp_up, "flops": 2 * M * D * (2 * FFN), "bytes": (A_mlp_up.nbytes + W_mlp_up.nbytes + M * 2 * FFN * 4)},
        "MLP_Down": {"A": A_mlp_down, "W": W_mlp_down, "B": B_mlp_down, "flops": 2 * M * FFN * D, "bytes": (A_mlp_down.nbytes + W_mlp_down.nbytes + M * D * 4)},
    }

    scaling_results = {}
    for s_name, s_data in shapes.items():
        scaling_results[s_name] = {}
        flops = s_data["flops"]
        mem_bytes = s_data["bytes"]
        intensity = flops / mem_bytes
        print(f"\n--- Shape: {s_name} | FLOPs: {flops/1e6:.2f} MFLOPs | Memory: {mem_bytes/1e6:.2f} MB | Arithmetic Intensity: {intensity:.2f} FLOP/Byte ---", flush=True)

        for t in threads:
            torch.set_num_threads(t)
            with torch.inference_mode():
                res_mm = benchmark_gemm(s_data["A"], s_data["B"], is_linear=False, warmup=15, iters=30)
                res_lin = benchmark_gemm(s_data["A"], s_data["W"], is_linear=True, warmup=15, iters=30)

            gflops_mm = (flops / (res_mm["mean_ms"] / 1000.0)) / 1e9
            gflops_lin = (flops / (res_lin["mean_ms"] / 1000.0)) / 1e9
            bandwidth_gb_s = (mem_bytes / (res_mm["mean_ms"] / 1000.0)) / 1e9

            scaling_results[s_name][t] = {
                "mm": res_mm,
                "linear": res_lin,
                "gflops_mm": gflops_mm,
                "gflops_linear": gflops_lin,
                "bandwidth_gb_s": bandwidth_gb_s
            }

            speedup_1t = scaling_results[s_name][1]["mm"]["mean_ms"] / res_mm["mean_ms"] if 1 in scaling_results[s_name] else 1.0
            efficiency = (speedup_1t / t) * 100.0

            print(f"  Threads = {t:2d} | mm: {res_mm['mean_ms']:6.2f} ms ({gflops_mm:5.2f} GFLOPS, {bandwidth_gb_s:5.2f} GB/s) | Linear: {res_lin['mean_ms']:6.2f} ms ({gflops_lin:5.2f} GFLOPS) | Speedup: {speedup_1t:5.2f}x (Eff: {efficiency:5.1f}%)", flush=True)

    # 4. PyTorch Profiler: Real 1-Step Diffusion Forward Pass
    print("\n[STEP 4] PYTORCH PROFILER: REAL 1-STEP DIFFUSION (6 THREADS)...", flush=True)
    torch.set_num_threads(6)
    
    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    
    print("Loading OmniVoice model for live 1-step profiling...", flush=True)
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    vp = vm.load_prompt("default", map_location="cpu")
    text = TextPreprocessor.process("Hôm nay tôi xin trình bày báo cáo về tiến độ phát triển hệ thống trí tuệ nhân tạo.")

    cfg = OmniVoiceGenerationConfig(
        num_step=1,
        guidance_scale=0.0,
        denoise=False,
        postprocess_output=False,
    )

    # Warmup
    print("Warmup 1-step...", flush=True)
    with torch.inference_mode():
        _ = model.generate(text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)

    # Profile
    print("Profiling 1-step generation pass...", flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        with torch.inference_mode():
            _ = model.generate(text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)

    key_avgs = prof.key_averages(group_by_input_shape=False)
    print("\n" + "=" * 90)
    print("TOP 15 CPU OPERATORS IN 1-STEP GENERATION (BY SELF CPU TIME):")
    print("=" * 90)
    print(key_avgs.table(sort_by="self_cpu_time_total", row_limit=15))

    total_self_cpu = sum(item.self_cpu_time_total for item in key_avgs)
    sorted_ops = sorted(key_avgs, key=lambda x: x.self_cpu_time_total, reverse=True)
    
    gemm_self_time = 0
    non_gemm_self_time = 0
    profiler_breakdown = []
    
    for item in key_avgs:
        if item.key in ["aten::mm", "aten::addmm", "aten::bmm"]:
            gemm_self_time += item.self_cpu_time_total
        else:
            non_gemm_self_time += item.self_cpu_time_total

    print("\n" + "=" * 90)
    print(f"TOTAL PROFILED SELF CPU TIME : {total_self_cpu / 1000.0:.2f} ms")
    print(f"  • Pure GEMM Self Time (mm/addmm/bmm) : {gemm_self_time / 1000.0:.2f} ms ({gemm_self_time/total_self_cpu*100:.2f}%)")
    print(f"  • Non-GEMM Self Time                 : {non_gemm_self_time / 1000.0:.2f} ms ({non_gemm_self_time/total_self_cpu*100:.2f}%)")
    print("=" * 90)

    # 5. Measure Step-level Latency directly with Python Timers (5 runs)
    print("\n[STEP 5] STEP-LEVEL DIRECT LATENCY TIMING (5 RUNS AT 6 THREADS)...", flush=True)
    step_timings = []
    for r in range(1, 6):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)
        t_el = (time.perf_counter() - t0) * 1000.0
        step_timings.append(t_el)
        print(f"  Run {r}: 1-Step Latency = {t_el:6.2f} ms", flush=True)

    mean_step_ms = float(np.mean(step_timings))
    std_step_ms = float(np.std(step_timings))
    print(f"\n1-Step Mean Latency: {mean_step_ms:.2f} ms (Std: {std_step_ms:.2f} ms)", flush=True)

    # Save comprehensive results to JSON
    report_data = {
        "environment": env_info,
        "scaling_microbenchmarks": scaling_results,
        "profiler": {
            "total_self_cpu_ms": total_self_cpu / 1000.0,
            "gemm_self_cpu_ms": gemm_self_time / 1000.0,
            "gemm_pct": (gemm_self_time / total_self_cpu) * 100.0,
            "non_gemm_self_cpu_ms": non_gemm_self_time / 1000.0,
            "top_10_ops": [
                {
                    "name": op.key,
                    "self_cpu_ms": op.self_cpu_time_total / 1000.0,
                    "self_cpu_pct": (op.self_cpu_time_total / total_self_cpu) * 100.0,
                    "calls": op.count,
                    "avg_ms": (op.self_cpu_time_total / op.count / 1000.0) if op.count > 0 else 0
                }
                for op in sorted_ops[:10]
            ]
        },
        "step_timings_ms": step_timings,
        "mean_step_ms": mean_step_ms
    }

    with open("benchmarks/v2_8_gemm_forensic_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print("\nForensic results saved to benchmarks/v2_8_gemm_forensic_results.json", flush=True)

if __name__ == "__main__":
    run_gemm_forensic_suite()
