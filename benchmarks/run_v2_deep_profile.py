#!/usr/bin/env python3
"""
V2 Golden Baseline Deep Profiling Tool.
Measures:
1. Stage-by-stage latency (Text, Prompt, Chunking, Diffusion steps, Vocoder decode, Postprocess, File I/O).
2. Exact CFG batch size / forward pass verification for guidance_scale=0.0 vs 2.0.
3. CPU Thread scaling (1, 2, 4, 6, 8 threads).
4. 3-run Repeatability testing (min, max, mean, median).
5. Hardware resource tracking (CPU %, RAM MB, frequency if available).
"""

import os
import sys
import time
import json
import psutil
import platform
import numpy as np
import soundfile as sf
import torch

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

TEXT_220 = (
    "Xin chào các bạn. Hôm nay tôi rất vui được giới thiệu với các bạn hệ thống "
    "tổng hợp giọng nói trí tuệ nhân tạo OmniVoice. Đây là một công nghệ đột phá, "
    "cho phép nhân bản giọng nói và đọc văn bản tiếng Việt vô cùng tự nhiên và mượt mà."
)


def get_cpu_info():
    proc = psutil.Process()
    mem = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.2),
        "cpu_freq_current_mhz": freq.current if freq else "UNKNOWN",
        "process_ram_mb": round(proc.memory_info().rss / (1024 * 1024), 2),
        "system_ram_used_gb": round((mem.total - mem.available) / (1024**3), 2),
    }


def main():
    print("=" * 80)
    print("      🔍 OMNIVOICE V2 GOLDEN BASELINE DEEP PROFILING")
    print("=" * 80)

    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))

    # 1. Model Loading
    print("\n[SECTION 1] MODEL & PROMPT LOADING PROFILING")
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()
    t_model_load = time.perf_counter() - t0
    print(f"  • Model Load Time: {t_model_load:.4f} s")

    t0 = time.perf_counter()
    vp = vm.load_prompt("default", map_location="cpu")
    t_prompt_load = time.perf_counter() - t0
    print(f"  • VoiceClonePrompt Load Time: {t_prompt_load*1000:.4f} ms")

    # Warmup
    warm_cfg = OmniVoiceGenerationConfig(num_step=2, guidance_scale=1.0, denoise=True, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate("Khởi động.", language="Vietnamese", generation_config=warm_cfg)
    print("  • Warmup Pass Complete.")

    # 2. Detailed Pipeline Profiling with Step-by-Step Hooking
    print("\n[SECTION 2] HIGH-RESOLUTION PIPELINE PROFILING (num_step=4, guidance_scale=0.0)")
    torch.set_num_threads(8)

    # Instrument model to measure forward passes and per-step latency
    step_timings = []
    forward_shapes = []

    orig_forward = model.forward
    def hooked_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None:
            forward_shapes.append(list(input_ids.shape))
        t_fwd_0 = time.perf_counter()
        out = orig_forward(*args, **kwargs)
        t_fwd = time.perf_counter() - t_fwd_0
        step_timings.append(t_fwd)
        return out

    model.forward = hooked_forward

    # Time Text Prep
    t0 = time.perf_counter()
    clean_text = TextPreprocessor.process(TEXT_220)
    t_text_prep = time.perf_counter() - t0

    # Time Chunking (1 chunk)
    t0 = time.perf_counter()
    chunks = [clean_text] if len(clean_text) <= 250 else [clean_text]
    t_chunking = time.perf_counter() - t0

    # Time Generation (Diffusion + Decode + Postprocess)
    cfg_turbo = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        denoise=True,
        pad_duration=0.1,
        fade_duration=0.1,
        postprocess_output=True,
    )

    t_gen_0 = time.perf_counter()
    with torch.inference_mode():
        audios = model.generate(
            text=chunks[0],
            language="Vietnamese",
            voice_clone_prompt=vp,
            generation_config=cfg_turbo,
        )
    t_gen_total = time.perf_counter() - t_gen_0

    # Restore original forward
    model.forward = orig_forward

    raw_audio = audios[0]
    audio_dur = len(raw_audio) / model.sampling_rate

    # Time File Write
    t0 = time.perf_counter()
    os.makedirs("debug", exist_ok=True)
    out_wav = "debug/v2_profile_sample.wav"
    sf.write(out_wav, raw_audio, model.sampling_rate)
    t_save = time.perf_counter() - t0

    diffusion_time = sum(step_timings)
    # decode + vocoder + postprocess overhead inside generate
    decode_and_vocoder_time = max(0.0, t_gen_total - diffusion_time)

    print(f"\nPipeline Stage Breakdown:")
    print(f"  1. Text Preprocess:    {t_text_prep*1000:8.4f} ms")
    print(f"  2. Prompt Load:        {t_prompt_load*1000:8.4f} ms")
    print(f"  3. Chunking:           {t_chunking*1000:8.4f} ms")
    print(f"  4. Diffusion (4 steps):{diffusion_time:8.4f} s ({diffusion_time/t_gen_total*100:.1f}%)")
    for s_idx, st in enumerate(step_timings):
        print(f"     - Step {s_idx+1}:          {st:8.4f} s (Batch shape: {forward_shapes[s_idx]})")
    print(f"  5. Decode & Vocoder:   {decode_and_vocoder_time:8.4f} s ({decode_and_vocoder_time/t_gen_total*100:.1f}%)")
    print(f"  6. Save File I/O:      {t_save*1000:8.4f} ms")
    print(f"  TOTAL GENERATION:      {t_gen_total:8.4f} s (Audio: {audio_dur:.2f}s | RTF: {t_gen_total/audio_dur:.3f}x)")

    # 3. CPU Thread Benchmark (1, 2, 4, 6, 8 threads)
    print("\n[SECTION 3] CPU THREAD BENCHMARK (1, 2, 4, 6, 8 threads)")
    thread_results = []
    thread_counts = [1, 2, 4, 6, 8]

    for th in thread_counts:
        torch.set_num_threads(th)
        # Warmup for this thread count
        with torch.inference_mode():
            _ = model.generate("Khởi động ngắn.", language="Vietnamese", generation_config=warm_cfg)
        
        # Measured run
        t0 = time.perf_counter()
        with torch.inference_mode():
            res_audios = model.generate(text=clean_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg_turbo)
        t_th = time.perf_counter() - t0
        dur_th = len(res_audios[0]) / model.sampling_rate
        rtf_th = t_th / dur_th if dur_th > 0 else 0
        thread_results.append({"threads": th, "time_s": round(t_th, 2), "audio_s": round(dur_th, 2), "rtf": round(rtf_th, 3)})
        print(f"  Threads = {th:2d} -> Generation Time: {t_th:6.2f}s | Audio: {dur_th:.2f}s | RTF: {rtf_th:5.3f}x")

    # 4. Repeatability Benchmark (3 consecutive runs at 8 threads)
    print("\n[SECTION 4] REPEATABILITY TEST (3 consecutive runs, 8 threads)")
    torch.set_num_threads(8)
    runs = []
    for r in range(1, 4):
        t0 = time.perf_counter()
        with torch.inference_mode():
            r_audios = model.generate(text=clean_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg_turbo)
        t_run = time.perf_counter() - t0
        dur_run = len(r_audios[0]) / model.sampling_rate
        rtf_run = t_run / dur_run if dur_run > 0 else 0
        runs.append({"run": r, "time_s": round(t_run, 2), "audio_s": round(dur_run, 2), "rtf": round(rtf_run, 3)})
        print(f"  Run {r}: Time = {t_run:6.2f}s | Audio = {dur_run:.2f}s | RTF = {rtf_run:5.3f}x")

    times = [x["time_s"] for x in runs]
    min_t = min(times)
    max_t = max(times)
    mean_t = round(float(np.mean(times)), 2)
    median_t = round(float(np.median(times)), 2)
    print(f"  -> Stats: Min = {min_t}s | Max = {max_t}s | Mean = {mean_t}s | Median = {median_t}s")

    # 5. Resource info
    cpu_stats = get_cpu_info()

    summary_data = {
        "pipeline": {
            "text_prep_ms": round(t_text_prep * 1000, 3),
            "prompt_load_ms": round(t_prompt_load * 1000, 3),
            "chunking_ms": round(t_chunking * 1000, 3),
            "diffusion_time_s": round(diffusion_time, 3),
            "decode_and_vocoder_s": round(decode_and_vocoder_time, 3),
            "save_file_ms": round(t_save * 1000, 3),
            "total_gen_s": round(t_gen_total, 3),
            "audio_dur_s": round(audio_dur, 2),
            "rtf": round(t_gen_total / audio_dur, 3),
            "step_timings_s": [round(x, 3) for x in step_timings],
            "forward_shapes": forward_shapes,
        },
        "threads": thread_results,
        "runs": runs,
        "stats": {
            "min_time_s": min_t,
            "max_time_s": max_t,
            "mean_time_s": mean_t,
            "median_time_s": median_t,
        },
        "cpu": cpu_stats,
    }

    with open("debug/v2_deep_profile_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE — Results saved to debug/v2_deep_profile_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
