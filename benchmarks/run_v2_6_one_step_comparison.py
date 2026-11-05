"""
OmniVoice V2.6 - One Diffusion-Step Impact Benchmark
Compares real 1-step diffusion with baseline MLP vs Candidate 1 (Fused Gate+Up)
"""

import os
import sys
import time
import json
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

class FusedGateUpMLP(nn.Module):
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
        return self.down_proj(F.silu(gate) * up)

def run_one_step_comparison(num_threads=6):
    torch.set_num_threads(num_threads)
    print(f"=== 1-STEP DIFFUSION IMPACT BENCHMARK ({num_threads} THREADS) ===", flush=True)

    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    
    print("Loading baseline model...", flush=True)
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    vp = vm.load_prompt("default", map_location="cpu")
    target_text = TextPreprocessor.process("Hôm nay tôi xin trình bày báo cáo về tiến độ phát triển hệ thống trí tuệ nhân tạo.")

    cfg = OmniVoiceGenerationConfig(
        num_step=1,
        guidance_scale=0.0,
        denoise=False,
        postprocess_output=False,
    )

    # 1. Measure Baseline 1-step
    print("\n[1] Measuring Baseline 1-Step (Warmup + 5 timed runs)...", flush=True)
    with torch.inference_mode():
        _ = model.generate(target_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)

    baseline_step_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(target_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)
        baseline_step_times.append((time.perf_counter() - t0) * 1000.0)

    mean_base_ms = sum(baseline_step_times) / len(baseline_step_times)
    print(f"  • Baseline 1-Step Mean: {mean_base_ms:.2f} ms ({baseline_step_times})", flush=True)

    # 2. In-memory patch all 28 MLP blocks to FusedGateUpMLP
    print("\n[2] Applying In-Memory Candidate 1 (Fused Gate+Up) to all 28 layers...", flush=True)
    for i, layer in enumerate(model.llm.layers):
        layer.mlp = FusedGateUpMLP(layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)

    # Measure Candidate 1 1-step
    print("Measuring Candidate 1-Step (Warmup + 5 timed runs)...", flush=True)
    with torch.inference_mode():
        _ = model.generate(target_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)

    cand_step_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(target_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)
        cand_step_times.append((time.perf_counter() - t0) * 1000.0)

    mean_cand_ms = sum(cand_step_times) / len(cand_step_times)
    print(f"  • Candidate 1-Step Mean: {mean_cand_ms:.2f} ms ({cand_step_times})", flush=True)

    diff_ms = mean_base_ms - mean_cand_ms
    speedup = mean_base_ms / mean_cand_ms
    pct_gain = (speedup - 1.0) * 100.0

    print("\n" + "=" * 60, flush=True)
    print("1-STEP DIFFUSION COMPARISON SUMMARY:", flush=True)
    print(f"  • Baseline 1-Step Latency : {mean_base_ms:.2f} ms", flush=True)
    print(f"  • Candidate 1-Step Latency: {mean_cand_ms:.2f} ms", flush=True)
    print(f"  • Absolute Delta          : {diff_ms:.2f} ms ({pct_gain:+.2f}%)", flush=True)
    print(f"  • End-to-End Speedup      : {speedup:.3f}x", flush=True)
    print("=" * 60, flush=True)

    with open("benchmarks/v2_6_one_step_comparison.json", "w", encoding="utf-8") as f:
        json.dump({
            "baseline_ms": mean_base_ms,
            "candidate_ms": mean_cand_ms,
            "delta_ms": diff_ms,
            "speedup": speedup,
            "pct_gain": pct_gain
        }, f, indent=2)

if __name__ == "__main__":
    run_one_step_comparison(6)
