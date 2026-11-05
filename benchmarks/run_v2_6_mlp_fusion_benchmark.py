"""
OmniVoice V2.6 - MLP Operator Fusion Forensic Benchmark
Evaluates baseline Qwen3MLP vs potential fusion paths in PyTorch 2.8.0+cpu across threads [1, 2, 4, 6, 8]
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

from omnivoice import OmniVoice

def create_mlp_modules():
    # Load model weights from 1 layer to get exact real weights
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    mlp_orig = model.llm.layers[0].mlp
    mlp_orig.eval()
    return mlp_orig

class FusedGateUpMLP(nn.Module):
    """Candidate 1: Fused Gate+Up Linear Projection (1 single GEMM)"""
    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.hidden_size = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features
        
        # Concat weights: [6144, 1024]
        fused_weight = torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0)
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.gate_up_proj.weight.data.copy_(fused_weight)
        
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.down_proj.weight.data.copy_(down_proj.weight.data)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class InplaceBufferMLP(nn.Module):
    """Candidate 2: In-place buffer reuse for SiLU and Mul"""
    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # in-place silu and in-place mul into gate buffer
        gate = F.silu(gate, inplace=False)
        gate.mul_(up)
        return self.down_proj(gate)

class FusedGateUpInplaceMLP(nn.Module):
    """Candidate 3: Fused Gate+Up + In-place Mul"""
    def __init__(self, gate_proj, up_proj, down_proj):
        super().__init__()
        self.hidden_size = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features
        fused_weight = torch.cat([gate_proj.weight.data, up_proj.weight.data], dim=0)
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.gate_up_proj.weight.data.copy_(fused_weight)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.down_proj.weight.data.copy_(down_proj.weight.data)

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        # In-place mul
        act_gate = F.silu(gate)
        act_gate.mul_(up)
        return self.down_proj(act_gate)

def benchmark_fn(func, x, warmup=10, iters=30):
    for _ in range(warmup):
        _ = func(x)
    
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = func(x)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # ms
        
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "std_ms": float(np.std(latencies)),
        "runs": len(latencies)
    }

def run_forensic_suite():
    print("=" * 80, flush=True)
    print("OMNIVOICE V2.6 — MLP OPERATOR FUSION FORENSIC BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    mlp_baseline = create_mlp_modules()
    c1_fused_gate_up = FusedGateUpMLP(mlp_baseline.gate_proj, mlp_baseline.up_proj, mlp_baseline.down_proj)
    c2_inplace = InplaceBufferMLP(mlp_baseline.gate_proj, mlp_baseline.up_proj, mlp_baseline.down_proj)
    c3_fused_inplace = FusedGateUpInplaceMLP(mlp_baseline.gate_proj, mlp_baseline.up_proj, mlp_baseline.down_proj)

    # Candidate 4: TorchScript Frozen Module
    x_sample = torch.randn(1, 410, 1024, dtype=torch.float32)
    try:
        traced_mlp = torch.jit.trace(mlp_baseline, x_sample)
        frozen_mlp = torch.jit.freeze(traced_mlp)
        c4_torchscript = frozen_mlp
        ts_available = True
    except Exception as e:
        print(f"TorchScript tracing failed: {e}", flush=True)
        ts_available = False

    # 1. Numerical Equivalence Verification
    print("\n--- 1. NUMERICAL EQUIVALENCE TEST ---", flush=True)
    with torch.inference_mode():
        out_base = mlp_baseline(x_sample)
        out_c1 = c1_fused_gate_up(x_sample)
        out_c2 = c2_inplace(x_sample)
        out_c3 = c3_fused_inplace(x_sample)
        if ts_available:
            out_c4 = c4_torchscript(x_sample)

    def check_error(name, out_cand, out_ref):
        diff = torch.abs(out_cand - out_ref)
        max_err = float(torch.max(diff))
        mean_err = float(torch.mean(diff))
        rel_err = float(torch.max(diff / (torch.abs(out_ref) + 1e-7)))
        passed = max_err < 1e-4
        print(f"  • {name:<25}: Max Err = {max_err:.2e}, Mean Err = {mean_err:.2e}, Rel Err = {rel_err:.2e} -> {'PASS' if passed else 'FAIL'}")
        return {"max_err": max_err, "mean_err": mean_err, "rel_err": rel_err, "pass": passed}

    equiv_results = {
        "C1_FusedGateUp": check_error("C1 Fused Gate+Up", out_c1, out_base),
        "C2_InplaceBuffer": check_error("C2 Inplace Buffer", out_c2, out_base),
        "C3_FusedInplace": check_error("C3 Fused+Inplace", out_c3, out_base),
    }
    if ts_available:
        equiv_results["C4_TorchScript"] = check_error("C4 TorchScript", out_c4, out_base)

    # 2. Detailed Component Breakdown of Baseline MLP
    print("\n--- 2. DETAILED BASELINE MLP BREAKDOWN (6 THREADS) ---", flush=True)
    torch.set_num_threads(6)
    with torch.inference_mode():
        res_gate = benchmark_fn(lambda x: mlp_baseline.gate_proj(x), x_sample)
        res_up = benchmark_fn(lambda x: mlp_baseline.up_proj(x), x_sample)
        
        gate_out = mlp_baseline.gate_proj(x_sample)
        up_out = mlp_baseline.up_proj(x_sample)
        
        res_act = benchmark_fn(lambda g: mlp_baseline.act_fn(g), gate_out)
        act_out = mlp_baseline.act_fn(gate_out)
        
        res_mul = benchmark_fn(lambda a: a * up_out, act_out)
        mul_out = act_out * up_out
        
        res_down = benchmark_fn(lambda m: mlp_baseline.down_proj(m), mul_out)
        res_full_mlp = benchmark_fn(lambda x: mlp_baseline(x), x_sample)

    print(f"  • Gate Projection   : {res_gate['mean_ms']:.3f} ms (median: {res_gate['median_ms']:.3f} ms)")
    print(f"  • Up Projection     : {res_up['mean_ms']:.3f} ms (median: {res_up['median_ms']:.3f} ms)")
    print(f"  • SiLU Activation   : {res_act['mean_ms']:.3f} ms (median: {res_act['median_ms']:.3f} ms)")
    print(f"  • Elementwise Mul   : {res_mul['mean_ms']:.3f} ms (median: {res_mul['median_ms']:.3f} ms)")
    print(f"  • Down Projection   : {res_down['mean_ms']:.3f} ms (median: {res_down['median_ms']:.3f} ms)")
    print(f"  -------------------------------------------------------------")
    sum_individual = res_gate['mean_ms'] + res_up['mean_ms'] + res_act['mean_ms'] + res_mul['mean_ms'] + res_down['mean_ms']
    print(f"  • Sum of Components : {sum_individual:.3f} ms")
    print(f"  • Full Block (Eager): {res_full_mlp['mean_ms']:.3f} ms (std: {res_full_mlp['std_ms']:.3f} ms)")

    # 3. Thread Scaling Benchmark [1, 2, 4, 6, 8]
    threads = [1, 2, 4, 6, 8]
    scaling_data = {}

    print("\n--- 3. THREAD SCALING BENCHMARK [1, 2, 4, 6, 8] ---", flush=True)
    for t in threads:
        torch.set_num_threads(t)
        scaling_data[t] = {}
        with torch.inference_mode():
            base_t = benchmark_fn(lambda x: mlp_baseline(x), x_sample)
            c1_t = benchmark_fn(lambda x: c1_fused_gate_up(x), x_sample)
            c2_t = benchmark_fn(lambda x: c2_inplace(x), x_sample)
            c3_t = benchmark_fn(lambda x: c3_fused_inplace(x), x_sample)
            c4_t = benchmark_fn(lambda x: c4_torchscript(x), x_sample) if ts_available else None

        speedup_c1 = base_t["mean_ms"] / c1_t["mean_ms"]
        speedup_c2 = base_t["mean_ms"] / c2_t["mean_ms"]
        speedup_c3 = base_t["mean_ms"] / c3_t["mean_ms"]
        speedup_c4 = (base_t["mean_ms"] / c4_t["mean_ms"]) if c4_t else 1.0

        scaling_data[t] = {
            "baseline": base_t,
            "c1_fused_gate_up": c1_t,
            "c2_inplace": c2_t,
            "c3_fused_inplace": c3_t,
            "c4_torchscript": c4_t,
            "speedup_c1": speedup_c1,
            "speedup_c2": speedup_c2,
            "speedup_c3": speedup_c3,
            "speedup_c4": speedup_c4,
        }

        print(f"\n[Threads = {t}]", flush=True)
        print(f"  Baseline MLP         : {base_t['mean_ms']:7.3f} ms", flush=True)
        print(f"  C1 Fused Gate+Up     : {c1_t['mean_ms']:7.3f} ms (Speedup: {speedup_c1:.3f}x, {(speedup_c1-1)*100:+.2f}%)", flush=True)
        print(f"  C2 Inplace Buffer    : {c2_t['mean_ms']:7.3f} ms (Speedup: {speedup_c2:.3f}x, {(speedup_c2-1)*100:+.2f}%)", flush=True)
        print(f"  C3 Fused+Inplace     : {c3_t['mean_ms']:7.3f} ms (Speedup: {speedup_c3:.3f}x, {(speedup_c3-1)*100:+.2f}%)", flush=True)
        if c4_t:
            print(f"  C4 TorchScript Freeze: {c4_t['mean_ms']:7.3f} ms (Speedup: {speedup_c4:.3f}x, {(speedup_c4-1)*100:+.2f}%)", flush=True)

    # Output full json for reporting
    with open("benchmarks/v2_6_mlp_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "breakdown_6t": {
                "gate": res_gate,
                "up": res_up,
                "act": res_act,
                "mul": res_mul,
                "down": res_down,
                "full_mlp": res_full_mlp
            },
            "scaling": scaling_data,
            "equivalence": equiv_results
        }, f, indent=2)

    print("\nBenchmark results written to benchmarks/v2_6_mlp_benchmark_results.json", flush=True)

if __name__ == "__main__":
    run_forensic_suite()
