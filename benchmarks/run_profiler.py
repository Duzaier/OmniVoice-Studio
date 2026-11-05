#!/usr/bin/env python3
"""
OmniVoice Deep Profiler: Measures exact millisecond timings for every stage in the TTS pipeline.
"""

import os
import sys
import time

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import psutil
import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import VoiceClonePrompt


def profile_pipeline(model, text, ref_audio=None, ref_text=None, voice_prompt=None, num_step=16):
    timings = {}
    proc = psutil.Process()

    print(f"\n[PROFILER] Running detailed profiling on text: '{text[:40]}...' (steps={num_step})")
    
    # Stage 1: Text preprocessing & Normalization
    t0 = time.perf_counter()
    gen_config = OmniVoiceGenerationConfig(num_step=num_step, guidance_scale=2.0)
    task = model._preprocess_all(
        text=text,
        language="Vietnamese",
        ref_text=ref_text,
        ref_audio=ref_audio,
        voice_clone_prompt=voice_prompt,
        preprocess_prompt=True,
    )
    timings["1_preprocess_and_prompt"] = time.perf_counter() - t0

    # Stage 2: Short/Long Chunk decision
    t0 = time.perf_counter()
    short_idx, long_idx = task.get_indices(gen_config, model.audio_tokenizer.config.frame_rate)
    timings["2_chunking_decision"] = time.perf_counter() - t0

    # Stage 3: Iterative diffusion forward passes
    t0 = time.perf_counter()
    if short_idx:
        sub_task = task.slice_task(short_idx)
        step_times = []
        raw_tokens = model._generate_iterative(sub_task, gen_config)
    else:
        raw_tokens = model._generate_chunked(task, gen_config)
    timings["3_iterative_diffusion"] = time.perf_counter() - t0

    # Stage 4: Decoding through Audio Tokenizer (Vocoder)
    t0 = time.perf_counter()
    audio = model._decode_and_post_process(raw_tokens[0], task.ref_rms[0], gen_config)
    timings["4_vocoder_decode_postprocess"] = time.perf_counter() - t0

    # Stage 5: Audio export
    t0 = time.perf_counter()
    out_path = os.path.join(os.path.dirname(__file__), "outputs", "profile_out.wav")
    sf.write(out_path, audio, model.sampling_rate)
    timings["5_file_export"] = time.perf_counter() - t0

    audio_dur = len(audio) / model.sampling_rate
    total_time = sum(timings.values())
    rtf = total_time / audio_dur if audio_dur > 0 else 0

    print("\n--- STAGE TIMING BREAKDOWN ---")
    for stage, t in timings.items():
        pct = (t / total_time) * 100
        print(f"  {stage:<30}: {t:>6.3f} s ({pct:>5.1f}%)")
    print("-" * 40)
    print(f"  TOTAL INFERENCE TIME:          {total_time:>6.3f} s")
    print(f"  GENERATED AUDIO DURATION:      {audio_dur:>6.3f} s")
    print(f"  REAL-TIME FACTOR (RTF):        {rtf:>6.3f}x")
    print(f"  PEAK RAM (RSS):                {proc.memory_info().rss / (1024**2):.1f} MB")
    print("=" * 40)

    return timings, rtf, audio_dur


if __name__ == "__main__":
    print("Loading model...")
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32, load_asr=False)
    
    sample_text = "Xin chào, đây là bài kiểm tra đo lường hiệu năng chuyên sâu cho hệ thống tổng hợp giọng nói OmniVoice."
    profile_pipeline(model, sample_text, num_step=16)
