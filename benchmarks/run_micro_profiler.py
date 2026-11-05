#!/usr/bin/env python3
"""
Micro-profiler to measure exact per-step forward pass latency on CPU.
"""

import os
import sys
import time
import torch

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.hardware import profile_hardware, apply_runtime_optimizations

hw = profile_hardware()
apply_runtime_optimizations(hw)

print("Loading model...")
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32, load_asr=False)
model.eval()

text = "Xin chào các bạn, hôm nay thời tiết rất đẹp."

# Test 1: num_step=8, guidance_scale=0.0
cfg1 = OmniVoiceGenerationConfig(num_step=8, guidance_scale=0.0, denoise=True, postprocess_output=False)
t0 = time.perf_counter()
with torch.inference_mode():
    audios1 = model.generate(text, language="Vietnamese", generation_config=cfg1)
t1 = time.perf_counter() - t0
dur1 = len(audios1[0]) / model.sampling_rate
print(f"Test 1 (8 steps, CFG 0.0): gen_time = {t1:.2f}s, audio_dur = {dur1:.2f}s, RTF = {t1/dur1:.3f}x")

# Test 2: num_step=8, guidance_scale=2.0
cfg2 = OmniVoiceGenerationConfig(num_step=8, guidance_scale=2.0, denoise=True, postprocess_output=False)
t0 = time.perf_counter()
with torch.inference_mode():
    audios2 = model.generate(text, language="Vietnamese", generation_config=cfg2)
t2 = time.perf_counter() - t0
dur2 = len(audios2[0]) / model.sampling_rate
print(f"Test 2 (8 steps, CFG 2.0): gen_time = {t2:.2f}s, audio_dur = {dur2:.2f}s, RTF = {t2/dur2:.3f}x")
