#!/usr/bin/env python3
"""
Urgent OmniVoice Performance Diagnostic Tool.
Instruments every single execution stage to identify the exact bottleneck.
Answers all Phase 1-6 diagnostic questions with high-resolution timings.
"""

import os
import sys
import time
import json
import logging
import psutil
import torch
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

# 1. Inspect Hardware / PyTorch Device
cuda_avail = torch.cuda.is_available()
cuda_ver = torch.version.cuda if hasattr(torch.version, 'cuda') else None
torch_ver = torch.__version__
gpu_count = torch.cuda.device_count() if cuda_avail else 0
gpu_name = torch.cuda.get_device_name(0) if cuda_avail and gpu_count > 0 else "None (No CUDA GPU)"
vram_alloc = torch.cuda.memory_allocated(0) / (1024*1024) if cuda_avail else 0
vram_res = torch.cuda.memory_reserved(0) / (1024*1024) if cuda_avail else 0

cpu_count_phys = psutil.cpu_count(logical=False)
cpu_count_log = psutil.cpu_count(logical=True)
num_threads = torch.get_num_threads()

print("=" * 80)
print("      🚨 URGENT OMNIVOICE PERFORMANCE DIAGNOSTIC")
print("=" * 80)
print(f"PyTorch Version:         {torch_ver}")
print(f"CUDA Available:          {cuda_avail}")
print(f"CUDA Version:            {cuda_ver}")
print(f"GPU Name:                {gpu_name}")
print(f"GPU VRAM Allocated:      {vram_alloc} MB")
print(f"GPU VRAM Reserved:       {vram_res} MB")
print(f"CPU Physical Cores:      {cpu_count_phys}")
print(f"CPU Logical Threads:     {cpu_count_log}")
print(f"PyTorch CPU Threads:     {num_threads}")
print("=" * 80)

# Import OmniVoice
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

storage = StorageManager()
vm = VoiceManager(str(storage.voices_dir))

# -------------------------------------------------------------
# PHASE 2 & 5: HIGH-RESOLUTION INSTRUMENTED TEST
# -------------------------------------------------------------

# Test sentences
ONE_SENTENCE = "Xin chào các bạn."
TWO_SENTENCES = (
    "Có một nghịch lý khá buồn về tình yêu. "
    "Có những người xuất hiện đúng lúc mình muốn yêu một ai đó, nhưng lại không phải lúc cả hai biết cách ở bên nhau."
)

print("\n[PHASE 2 & 4] INSTRUMENTING STAGES ON ONE SENTENCE:")
print(f"Text: '{ONE_SENTENCE}'\n")

# Stage 1: Model Loading
t0 = time.perf_counter()
device = "cuda" if cuda_avail else "cpu"
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=torch.float32, load_asr=False)
model.eval()
t_model_load = time.perf_counter() - t0
print(f"[PERF] 1. model_load:              {t_model_load:.4f} s (Device: {model.device}, Dtype: {model.dtype})")

# Stage 2: Prompt Lookup / Load
t0 = time.perf_counter()
voice_prompt = vm.load_prompt("default", map_location=device)
t_prompt_lookup = time.perf_counter() - t0
print(f"[PERF] 2. prompt_lookup:           {t_prompt_lookup*1000:.4f} ms (Found: {voice_prompt is not None})")

# Stage 3: Text Preprocessing
t0 = time.perf_counter()
clean_1 = TextPreprocessor.process(ONE_SENTENCE)
t_prep = time.perf_counter() - t0
print(f"[PERF] 3. text_preprocess:         {t_prep*1000:.4f} ms (Cleaned: '{clean_1}')")

# Stage 4: Execution Config & Parameter Audit
cfg_default = OmniVoiceGenerationConfig(
    num_step=32,
    guidance_scale=2.0,
    t_shift=0.1,
    class_temperature=0.0,
    denoise=True,
    preprocess_prompt=True,
    postprocess_output=True,
    audio_chunk_duration=15.0,
    audio_chunk_threshold=30.0,
)

print(f"\n[PHASE 4] RUNTIME GENERATION PARAMETERS AUDIT:")
print(f"  • num_step:              {cfg_default.num_step}")
print(f"  • guidance_scale:        {cfg_default.guidance_scale}")
print(f"  • t_shift:               {cfg_default.t_shift}")
print(f"  • class_temperature:     {cfg_default.class_temperature}")
print(f"  • denoise:               {cfg_default.denoise}")
print(f"  • preprocess_prompt:     {cfg_default.preprocess_prompt}")
print(f"  • postprocess_output:    {cfg_default.postprocess_output}")
print(f"  • audio_chunk_duration:  {cfg_default.audio_chunk_duration} s")
print(f"  • audio_chunk_threshold: {cfg_default.audio_chunk_threshold} s")
print(f"  • device:                {model.device}")
print(f"  • dtype:                 {model.dtype}")

# Stage 5: Measure ONE short sentence with num_step=16 vs 32
print(f"\n▶ Running ONE short sentence (num_step=16)...")
cfg_16 = OmniVoiceGenerationConfig(
    num_step=16, guidance_scale=2.0, t_shift=0.1, class_temperature=0.0,
    denoise=True, preprocess_prompt=True, postprocess_output=True,
)
t0 = time.perf_counter()
with torch.inference_mode():
    audios_1 = model.generate(
        text=clean_1,
        language="Vietnamese",
        voice_clone_prompt=voice_prompt,
        generation_config=cfg_16,
    )
t_gen_1 = time.perf_counter() - t0
dur_1 = len(audios_1[0]) / model.sampling_rate
print(f"[PERF] ONE SENTENCE RESULT:")
print(f"  • Total Time:            {t_gen_1:.2f} s")
print(f"  • Audio Duration:        {dur_1:.2f} s")
print(f"  • RTF:                   {t_gen_1 / dur_1:.3f}x")
print(f"  • Per-Step Latency:      {t_gen_1 / 16:.3f} s/step")

# Stage 6: Measure TWO sentences (num_step=16)
print(f"\n▶ Running TWO sentences (num_step=16)...")
clean_2 = TextPreprocessor.process(TWO_SENTENCES)
print(f"Input: '{clean_2}' ({len(clean_2)} chars)")

t0 = time.perf_counter()
with torch.inference_mode():
    audios_2 = model.generate(
        text=clean_2,
        language="Vietnamese",
        voice_clone_prompt=voice_prompt,
        generation_config=cfg_16,
    )
t_gen_2 = time.perf_counter() - t0
dur_2 = len(audios_2[0]) / model.sampling_rate
print(f"[PERF] TWO SENTENCES RESULT:")
print(f"  • Total Time:            {t_gen_2:.2f} s")
print(f"  • Audio Duration:        {dur_2:.2f} s")
print(f"  • RTF:                   {t_gen_2 / dur_2:.3f}x")
print(f"  • Per-Step Latency:      {t_gen_2 / 16:.3f} s/step")

# Stage 7: Measure with num_step=8 (TURBO)
print(f"\n▶ Running TWO sentences with num_step=8 (TURBO)...")
cfg_8 = OmniVoiceGenerationConfig(
    num_step=8, guidance_scale=1.5, t_shift=0.1, class_temperature=0.0,
    denoise=True, preprocess_prompt=True, postprocess_output=True,
)
t0 = time.perf_counter()
with torch.inference_mode():
    audios_8 = model.generate(
        text=clean_2,
        language="Vietnamese",
        voice_clone_prompt=voice_prompt,
        generation_config=cfg_8,
    )
t_gen_8 = time.perf_counter() - t0
dur_8 = len(audios_8[0]) / model.sampling_rate
print(f"[PERF] TWO SENTENCES (8 STEPS) RESULT:")
print(f"  • Total Time:            {t_gen_8:.2f} s")
print(f"  • Audio Duration:        {dur_8:.2f} s")
print(f"  • RTF:                   {t_gen_8 / dur_8:.3f}x")
print(f"  • Per-Step Latency:      {t_gen_8 / 8:.3f} s/step")

print("\n" + "=" * 80)
print("DIAGNOSTIC TEST COMPLETE")
print("=" * 80)
