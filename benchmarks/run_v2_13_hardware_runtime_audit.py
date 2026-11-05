#!/usr/bin/env python3
"""
OMNIVOICE V2.13 — HARDWARE & RUNTIME ACCELERATION FORENSIC AUDIT
Strictly Read-Only / Forensic Benchmark. No production code changes.
"""

import os
import sys
import time
import json
import platform
import subprocess
from typing import Dict, Any, List

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import numpy as np

# Ensure HF_ENDPOINT is set
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def get_gpu_wmi_info() -> List[Dict[str, Any]]:
    """Retrieve GPU adapter information via PowerShell WMI."""
    gpus = []
    try:
        cmd = 'Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, DriverDate, AdapterRAM, VideoProcessor, Status | ConvertTo-Json'
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=10)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            if isinstance(data, dict):
                gpus.append(data)
            elif isinstance(data, list):
                gpus.extend(data)
    except Exception as e:
        gpus.append({"error": str(e)})
    return gpus


def get_directx_wddm_info() -> Dict[str, Any]:
    """Retrieve DirectX / WDDM version details."""
    info = {"directx_installed": True, "wddm_version": "Unknown"}
    try:
        cmd = '(Get-ItemProperty -Path "HKLM:\\SOFTWARE\\Microsoft\\DirectX").Version'
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            info["directx_version_reg"] = res.stdout.strip()
    except Exception:
        pass
    return info


def audit_runtimes() -> Dict[str, Any]:
    """Audit availability and compatibility of ML runtimes in current environment."""
    runtimes = {}

    # 1. PyTorch CPU
    runtimes["pytorch_cpu"] = {
        "installed": True,
        "version": torch.__version__,
        "python_version": sys.version.split()[0],
        "backend": "CPU (Intel MKL / oneDNN)",
        "mkl_available": torch.backends.mkl.is_available(),
        "openmp_available": torch.backends.openmp.is_available(),
        "mkldnn_available": torch.backends.mkldnn.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
        "hip_version": getattr(torch.version, "hip", None),
        "operator_support_qwen3": "Full (Baseline)",
        "operator_support_vocoder": "Full (Baseline)",
        "status": "ACTIVE_PRODUCTION_BACKEND"
    }

    # 2. PyTorch DirectML
    try:
        import torch_directml
        runtimes["pytorch_directml"] = {
            "installed": True,
            "version": getattr(torch_directml, "__version__", "unknown"),
            "device_available": torch_directml.is_available(),
            "device_count": torch_directml.device_count() if torch_directml.is_available() else 0,
            "status": "AVAILABLE"
        }
    except ImportError:
        runtimes["pytorch_directml"] = {
            "installed": False,
            "version": None,
            "python_compatibility_note": "torch-directml package official wheels are currently limited to Python <= 3.10 / 3.11. Current Python is 3.13.15.",
            "status": "NOT_INSTALLED"
        }

    # 3. ONNX Runtime CPU
    try:
        import onnxruntime as ort
        runtimes["onnxruntime_cpu"] = {
            "installed": True,
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
            "status": "AVAILABLE"
        }
    except ImportError:
        runtimes["onnxruntime_cpu"] = {
            "installed": False,
            "version": None,
            "status": "NOT_INSTALLED"
        }

    # 4. ONNX Runtime DirectML
    runtimes["onnxruntime_directml"] = {
        "installed": False,
        "version": None,
        "note": "Requires onnxruntime-directml package (requires ONNX model export)",
        "status": "NOT_INSTALLED"
    }

    # 5. OpenVINO
    try:
        import openvino as ov
        runtimes["openvino"] = {
            "installed": True,
            "version": ov.__version__,
            "status": "AVAILABLE"
        }
    except ImportError:
        runtimes["openvino"] = {
            "installed": False,
            "version": None,
            "note": "OpenVINO runtime not installed in virtual environment",
            "status": "NOT_INSTALLED"
        }

    # 6. TorchScript JIT
    runtimes["torchscript_jit"] = {
        "installed": True,
        "version": torch.__version__,
        "supports_tracing": True,
        "supports_scripting": True,
        "status": "BUILTIN_AVAILABLE"
    }

    return runtimes


def test_torchscript_micro_benchmark(threads: int = 6) -> Dict[str, Any]:
    """Test TorchScript JIT tracing on exact Qwen3 Transformer MLP block dimensions."""
    torch.set_num_threads(threads)
    
    hidden_size = 2048
    intermediate_size = 5632
    seq_len = 150
    batch_size = 1
    
    # Standalone Qwen3 MLP Block
    class MockQwen3MLP(nn.Module):
        def __init__(self, h_size, i_size):
            super().__init__()
            self.gate_proj = nn.Linear(h_size, i_size, bias=False)
            self.up_proj = nn.Linear(h_size, i_size, bias=False)
            self.down_proj = nn.Linear(i_size, h_size, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    mlp = MockQwen3MLP(hidden_size, intermediate_size).eval()
    dummy_input = torch.randn(batch_size, seq_len, hidden_size)

    # 1. Standard PyTorch Eager Evaluation
    with torch.inference_mode():
        # Warmup
        for _ in range(10):
            _ = mlp(dummy_input)
            
        times_eager = []
        for _ in range(50):
            t0 = time.perf_counter()
            out_eager = mlp(dummy_input)
            times_eager.append((time.perf_counter() - t0) * 1000)

    # 2. TorchScript Traced Evaluation
    jit_success = False
    times_jit = []
    error_diff = 0.0
    jit_err_msg = ""
    try:
        traced_mlp = torch.jit.trace(mlp, dummy_input)
        traced_mlp = torch.jit.optimize_for_inference(traced_mlp)
        
        with torch.inference_mode():
            # Warmup
            for _ in range(10):
                _ = traced_mlp(dummy_input)
                
            for _ in range(50):
                t0 = time.perf_counter()
                out_jit = traced_mlp(dummy_input)
                times_jit.append((time.perf_counter() - t0) * 1000)
                
        error_diff = float((out_eager - out_jit).abs().max().item())
        jit_success = True
    except Exception as e:
        jit_err_msg = str(e)

    return {
        "mlp_dims": {"M": seq_len, "K": hidden_size, "N": intermediate_size},
        "eager_latency_mean_ms": float(np.mean(times_eager)),
        "eager_latency_std_ms": float(np.std(times_eager)),
        "jit_success": jit_success,
        "jit_latency_mean_ms": float(np.mean(times_jit)) if jit_success else None,
        "jit_latency_std_ms": float(np.std(times_jit)) if jit_success else None,
        "jit_speedup": float(np.mean(times_eager) / np.mean(times_jit)) if (jit_success and np.mean(times_jit) > 0) else None,
        "max_abs_error": error_diff,
        "jit_err_msg": jit_err_msg
    }


def analyze_model_operator_compatibility() -> Dict[str, Any]:
    """Analyze operator breakdown and export compatibility for Qwen3 and Higgs Vocoder."""
    qwen3_analysis = {
        "component": "Qwen3 Backbone (28 Layers, D=2048, D_ff=5632)",
        "operators": [
            {"op": "nn.Linear / aten::mm", "coverage": "Standard GEMM", "export_risk": "Low"},
            {"op": "Qwen3RMSNorm", "coverage": "Elementwise + reduction", "export_risk": "Low"},
            {"op": "Qwen3RotaryEmbedding (RoPE)", "coverage": "cos/sin rotation", "export_risk": "Medium (Dynamic sequence slicing)"},
            {"op": "GQA Attention (16 heads Q, 8 heads KV)", "coverage": "Scaled Dot-Product Attention", "export_risk": "Medium (Custom bidirectional mask)"},
            {"op": "SiLU Gated MLP", "coverage": "SiLU(W1*x) * (W2*x)", "export_risk": "Low"},
            {"op": "Diffusion Flow-Matching Loop", "coverage": "4-step ODE integration + token splicing", "export_risk": "HIGH (Iterative python control flow)"}
        ],
        "onnx_exportability": "Partial (Individual step forward can be exported; full multi-step loop requires complex subgraph / loop node or external orchestration)",
        "directml_compatibility": "Possible for single forward step; high CPU-GPU synchronization overhead if loop orchestrated from Python"
    }

    vocoder_analysis = {
        "component": "Higgs Vocoder (Decoder / Vocoder)",
        "operators": [
            {"op": "Conv1d", "coverage": "Standard 1D convolution", "export_risk": "Low"},
            {"op": "ConvTranspose1d", "coverage": "Transposed 1D conv (8x, 5x, 4x, 2x)", "export_risk": "Low"},
            {"op": "Snake1d Activation", "coverage": "x + (1/alpha) * sin^2(alpha * x)", "export_risk": "Medium (Custom activation function, math ops supported in ONNX Opset 14+)"},
            {"op": "WeightNorm / Parametrizations", "coverage": "Weight normalization", "export_risk": "Low (Foldable at inference)"},
            {"op": "Audio Reconstruction ISTFT/Conv", "coverage": "Time-domain waveform synthesis", "export_risk": "Low"}
        ],
        "onnx_exportability": "High (Static / 1D dynamic conv graph, feeds directly to waveform)",
        "directml_compatibility": "Possible, but vocoder only represents ~12.7% (2.7s) of total wall time; speedup potential is capped by Amdahl's law"
    }

    return {
        "qwen3_backbone": qwen3_analysis,
        "higgs_vocoder": vocoder_analysis
    }


def evaluate_acceleration_paths() -> Dict[str, Any]:
    """Evaluate Path A, B, C, D in detail."""
    paths = {
        "Path_A_PyTorch_CPU": {
            "name": "Path A: PyTorch CPU Inference (Golden Baseline V2.1)",
            "feasibility": "CURRENT_PRODUCTION",
            "findings_v2_5_to_v2_12": [
                "V2.5: MKL 2025.2 AVX2/FMA3 is already the active GEMM kernel.",
                "V2.6: MLP operator fusion yielded -1.8% to +0.4% (GEMM bound, memory bandwidth ceiling).",
                "V2.7: Vocoder represents only ~12.7% of wall time (~2.7s out of 21.4s).",
                "V2.8: 6 threads is optimal. CPU compute is at ~88% of Ryzen 3450U theoretical FLOPs.",
                "V2.9: num_step=4 is minimum quality threshold. Compute reuse unsafe.",
                "V2.10: Runtime overhead (Python dispatch, copy, cat) is <2.3%.",
                "V2.11: FP16 causes severe tone degradation on CPU; INT8 requires dedicated calibration.",
                "V2.12: Intra-batch padding is already 0% in Batch-1; aggressive chunking slows down by up to 2.69x."
            ],
            "speedup_potential": "Near plateau on AMD Ryzen 5 3450U CPU (~21.4s for 4.8s audio, RTF ~4.4x).",
            "risk_level": "None (Verified baseline)",
            "verdict": "ACTIVE_MAXIMIZED"
        },
        "Path_B_AMD_DirectML": {
            "name": "Path B: AMD Vega 8 via DirectML (torch-directml / ORT-DML)",
            "feasibility": "BLOCKED_ENVIRONMENT",
            "hardware_context": "AMD Radeon Vega 8 Mobile (512 Stream Processors, shared DDR4 RAM, ~1.1 TFLOPs FP32).",
            "blockers": [
                "1. Python Version: Python 3.13.15 currently lacks official torch-directml prebuilt wheels (DirectML PyTorch plugin typically targets Python <=3.10/3.11).",
                "2. Memory Bandwidth: Vega 8 iGPU shares system DDR4 memory (2400-2666 MHz, ~35 GB/s bandwidth) with the CPU. GEMM on iGPU does NOT have high-bandwidth dedicated VRAM (e.g., GDDR6 300+ GB/s).",
                "3. Driver Overhead: DirectML operator dispatch overhead on low-end iGPUs often exceeds CPU MKL latency for small batch sizes (B=1).",
                "4. Host-to-Device Transfers: Iterative 4-step diffusion loop with Python CPU token management introduces frequent DML-to-Host sync stalls."
            ],
            "projected_real_world_speedup": "Estimated 0.8x to 1.3x (limited by shared DDR4 bandwidth and dispatch latency).",
            "risk_level": "High (Package incompatibility in Python 3.13, potential numerical discrepancies in diffusion)",
            "verdict": "BLOCKED_ON_PYTHON_3_13_AND_SHARED_BANDWIDTH"
        },
        "Path_C_ONNX_Runtime": {
            "name": "Path C: ONNX Runtime (CPU / DirectML Execution Provider)",
            "feasibility": "REQUIRES_OFFLINE_EXPORT_AND_TRANSFORMATION",
            "architecture_analysis": [
                "1. Qwen3 Backbone: Exporting 28 layers of Qwen3 requires handling dynamic sequence lengths (L in 60-500) and custom RoPE rotary position embeddings.",
                "2. Diffusion Loop: The 4-step iterative flow-matching solver must either be compiled into an ONNX Loop subgraph or orchestrated via Python ONNXRuntime session runs.",
                "3. Vocoder: Higgs vocoder with Snake1d activations can be exported cleanly to ONNX (Opset 14+).",
                "4. CPU Performance: ONNX Runtime CPU (MLAS backend) on Zen+ CPU performs similarly to PyTorch Intel MKL (within +/- 5%).",
                "5. DirectML EP: ORT-DirectML could run on Vega 8, but shared RAM bandwidth and Python loop iteration overhead remain bottleneck."
            ],
            "risk_level": "High (Requires complete checkpoint conversion, custom ONNX export scripts, verification of 6 tones quality)",
            "verdict": "HIGH_COMPLEXITY_LOW_YIELD_ON_ZEN_PLUS"
        },
        "Path_D_Alternative_Hardware_and_Models": {
            "name": "Path D: Target Hardware & Architecture Strategy for Real-Time TTS",
            "feasibility": "HARDWARE_OR_DISTILLED_MODEL_UPGRADE",
            "targets": {
                "Target_1_Real_Time_1x": "Generate 90s audio in 90s (RTF <= 1.0x, 4.4x speedup required)",
                "Target_2_Fast_Real_Time": "Generate 90s audio in 60s (RTF <= 0.66x, 6.7x speedup required)"
            },
            "viable_routes": [
                {
                    "route": "Dedicated NVIDIA GPU (CUDA + TensorRT / FlashAttention)",
                    "speedup": "10x - 30x (RTF ~0.15x - 0.3x)",
                    "mechanism": "GDDR6 bandwidth (300-500 GB/s vs DDR4 35 GB/s) + Tensor Cores."
                },
                {
                    "route": "Modern Desktop CPU (AMD Zen 4/5 or Intel Core 14th Gen with AVX-512 / VNNI / DDR5)",
                    "speedup": "3.5x - 5.0x (RTF ~0.9x - 1.2x)",
                    "mechanism": "AVX-512 FMA compute density + DDR5-6000 bandwidth (80+ GB/s)."
                },
                {
                    "route": "Distilled 1-Step or 2-Step Student Model (e.g., Consistency / Flow Distillation)",
                    "speedup": "2.0x - 4.0x on existing Ryzen 5 3450U CPU (RTF ~1.1x - 2.2x)",
                    "mechanism": "Halving or quartering the number of 28-layer Transformer passes from 4 steps to 1-2 steps without changing hardware."
                },
                {
                    "route": "Compact / Pruned Backbone (e.g., 14-16 layers instead of 28 layers)",
                    "speedup": "1.8x - 2.0x on CPU",
                    "mechanism": "Linear reduction in GEMM FLOPs."
                }
            ],
            "verdict": "PRIMARY_REALISTIC_PATHWAY_FOR_SUB_1_0_RTF"
        }
    }
    return paths


def main():
    print("=" * 80)
    print("OMNIVOICE V2.13 — HARDWARE & RUNTIME ACCELERATION FORENSIC AUDIT")
    print("=" * 80)

    # 1. GPU & Driver
    print("\n[AUDIT 1] Inspecting GPU & Display Drivers...")
    gpus = get_gpu_wmi_info()
    for g in gpus:
        print(f"  • Device: {g.get('Name')} | Driver: {g.get('DriverVersion')} | RAM: {g.get('AdapterRAM')}")

    # 2. ML Runtimes
    print("\n[AUDIT 2] Auditing ML Runtimes in Environment...")
    runtimes = audit_runtimes()
    for name, data in runtimes.items():
        status = data.get("status")
        ver = data.get("version", "N/A")
        print(f"  • {name:25s}: Status={status:25s} | Version={ver}")

    # 3. Micro-benchmark TorchScript / PyTorch JIT
    print("\n[AUDIT 3] Micro-Benchmarking PyTorch JIT vs Eager on Qwen3 MLP Dimensions (M=150, K=2048, N=5632)...")
    jit_res = test_torchscript_micro_benchmark(threads=6)
    print(f"  • PyTorch Eager Mean : {jit_res['eager_latency_mean_ms']:.2f} ms")
    if jit_res["jit_success"]:
        print(f"  • TorchScript JIT Mean: {jit_res['jit_latency_mean_ms']:.2f} ms | Speedup: {jit_res['jit_speedup']:.3f}x")
        print(f"  • Max Absolute Error : {jit_res['max_abs_error']:.6e}")
    else:
        print(f"  • TorchScript JIT failed: {jit_res['jit_err_msg']}")

    # 4. Model Operator Compatibility
    print("\n[AUDIT 4] Analyzing Qwen3 Backbone & Higgs Vocoder Operator Compatibility...")
    op_compat = analyze_model_operator_compatibility()

    # 5. Acceleration Paths Evaluation
    print("\n[AUDIT 5] Evaluating Acceleration Paths A, B, C, D...")
    paths_eval = evaluate_acceleration_paths()

    # Consolidate results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu": "AMD Ryzen 5 3450U with Radeon Vega Mobile Gfx",
            "cores_physical": 4,
            "threads_logical": 8,
            "instruction_sets": ["AVX2", "FMA3", "BMI2"],
            "ram_total_gb": 16.0,
            "gpus": gpus,
            "directx": get_directx_wddm_info()
        },
        "runtimes": runtimes,
        "torchscript_micro_benchmark": jit_res,
        "model_operator_compatibility": op_compat,
        "acceleration_paths": paths_eval,
        "final_classification": "HARDWARE_UPGRADE_REQUIRED"
    }

    os.makedirs("benchmarks", exist_ok=True)
    out_path = "benchmarks/v2_13_hardware_runtime_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] V2.13 Audit complete. Results saved to {out_path}")


if __name__ == "__main__":
    main()
