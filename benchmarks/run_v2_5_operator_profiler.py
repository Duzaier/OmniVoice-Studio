"""
OmniVoice V2.5 - PyTorch Operator Profiling
Using torch.profiler.profile to rank TOP CPU operators during a 1-step diffusion forward pass
"""

import os
import sys
import time
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

def profile_operators(num_threads=6):
    torch.set_num_threads(num_threads)
    print(f"Loading OmniVoice model for operator profiling with {num_threads} threads...", flush=True)
    
    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    vp = vm.load_prompt("default", map_location="cpu")
    target_text = TextPreprocessor.process("Hôm nay tôi xin trình bày báo cáo về tiến độ phát triển hệ thống trí tuệ nhân tạo.")

    gen_config = OmniVoiceGenerationConfig(
        num_step=1,
        guidance_scale=0.0,
        denoise=False,
        postprocess_output=False
    )

    print("Running warmup pass...", flush=True)
    with torch.inference_mode():
        _ = model.generate(target_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=gen_config)

    print("Starting PyTorch Profiler for 1-step diffusion...", flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        with torch.inference_mode():
            _ = model.generate(target_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=gen_config)

    # Key averages
    key_avgs = prof.key_averages(group_by_input_shape=False)
    print("\n" + "="*90, flush=True)
    print("TOP 15 CPU OPERATORS BY TOTAL CPU TIME:", flush=True)
    print("="*90, flush=True)
    print(key_avgs.table(sort_by="cpu_time_total", row_limit=15), flush=True)

    # Calculate percentages
    total_cpu_time = sum(item.cpu_time_total for item in key_avgs)
    sorted_items = sorted(key_avgs, key=lambda x: x.cpu_time_total, reverse=True)
    
    print("\n" + "="*90, flush=True)
    print("TOP 10 OPERATORS BREAKDOWN (RANKED):", flush=True)
    print("="*90, flush=True)
    for idx, item in enumerate(sorted_items[:10], 1):
        pct = (item.cpu_time_total / total_cpu_time) * 100.0 if total_cpu_time > 0 else 0
        print(f"{idx}. {item.key:<35} | Total: {item.cpu_time_total / 1000.0:>8.2f} ms | {pct:>6.2f}% | Count: {item.count:>5} | Avg: {item.cpu_time_total / item.count / 1000.0:>7.3f} ms", flush=True)

if __name__ == "__main__":
    profile_operators(6)
