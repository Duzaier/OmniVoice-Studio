#!/usr/bin/env python3
"""
Single-Variable Batch-1 Experiment Benchmark.
Tests:
1. Shape verification: guidance_scale=0.0 -> [1, 8, 410], guidance_scale=1.5 -> [2, 8, 410].
2. CFG Safety verification.
3. Benchmark: Warmup (1 run) + Measured (3 runs) for Batch-1 V2 Turbo (num_step=4, guidance_scale=0.0).
4. Detailed latency tracking (diffusion time vs vocoder time).
5. Numerical audio metrics and phoneme/tone quality checks.
"""

import os
import sys
import time
import json
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
from omnivoice.core.pitch_analyzer import PitchAndEnergyAnalyzer
from omnivoice.core.phoneme_validator import PhonemeValidator

TEXT_220 = (
    "Xin chào các bạn. Hôm nay tôi rất vui được giới thiệu với các bạn hệ thống "
    "tổng hợp giọng nói trí tuệ nhân tạo OmniVoice. Đây là một công nghệ đột phá, "
    "cho phép nhân bản giọng nói và đọc văn bản tiếng Việt vô cùng tự nhiên và mượt mà."
)


def main():
    print("=" * 80)
    print("      🧪 OMNIVOICE V2 TURBO — BATCH-1 SINGLE-VARIABLE EXPERIMENT")
    print("=" * 80)

    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    torch.set_num_threads(8)

    print("\n[STEP 1] LOADING MODEL & CACHED PROMPT")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    vp = vm.load_prompt("default", map_location="cpu")
    clean_text = TextPreprocessor.process(TEXT_220)

    # 1. Shape Verification & CFG Safety Test
    print("\n[STEP 2] TENSOR SHAPE & CFG SAFETY VERIFICATION")
    captured_shapes = []
    orig_forward = model.forward

    def diagnostic_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None:
            captured_shapes.append(list(input_ids.shape))
        return orig_forward(*args, **kwargs)

    model.forward = diagnostic_forward

    # Test A: guidance_scale = 0.0 (Must produce batch = 1)
    captured_shapes.clear()
    cfg_b1 = OmniVoiceGenerationConfig(num_step=1, guidance_scale=0.0, denoise=True, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate("Kiểm tra.", language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg_b1)
    b1_shape = captured_shapes[0] if captured_shapes else []
    print(f"  • guidance_scale = 0.0 -> Transformer Input Shape: {b1_shape} (Batch = {b1_shape[0] if b1_shape else 'N/A'})")

    # Test B: guidance_scale = 1.5 (Must produce batch = 2)
    captured_shapes.clear()
    cfg_b2 = OmniVoiceGenerationConfig(num_step=1, guidance_scale=1.5, denoise=True, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate("Kiểm tra CFG.", language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg_b2)
    b2_shape = captured_shapes[0] if captured_shapes else []
    print(f"  • guidance_scale = 1.5 -> Transformer Input Shape: {b2_shape} (Batch = {b2_shape[0] if b2_shape else 'N/A'})")

    model.forward = orig_forward

    if b1_shape and b1_shape[0] == 1 and b2_shape and b2_shape[0] == 2:
        print("  ✅ SHAPE VERIFICATION & CFG ISOLATION PASS!")
    else:
        print("  ❌ SHAPE VERIFICATION FAILED! b1:", b1_shape, "b2:", b2_shape)
        return

    # 2. Benchmark Batch-1 V2 Turbo (num_step=4, guidance_scale=0.0)
    print("\n[STEP 3] RUNNING BENCHMARK (Warmup + 3 Measured Runs)")
    cfg_turbo_4 = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        denoise=True,
        pad_duration=0.1,
        fade_duration=0.1,
        postprocess_output=True,
    )

    # Warmup
    print("▶ Running Warmup Pass...")
    with torch.inference_mode():
        _ = model.generate(clean_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg_turbo_4)
    print("  ✅ Warmup Complete.")

    # 3 Measured Runs with Step-by-Step Latency Instrumentation
    runs = []
    audios = []
    
    for r in range(1, 4):
        step_times = []
        def step_timer(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_forward(*args, **kwargs)
            step_times.append(time.perf_counter() - t0)
            return out

        model.forward = step_timer
        t_start = time.perf_counter()
        with torch.inference_mode():
            res_audios = model.generate(
                text=clean_text,
                language="Vietnamese",
                voice_clone_prompt=vp,
                generation_config=cfg_turbo_4,
            )
        t_total = time.perf_counter() - t_start
        model.forward = orig_forward

        diff_time = sum(step_times)
        vocoder_time = max(0.0, t_total - diff_time)
        audio = res_audios[0]
        audio_dur = len(audio) / model.sampling_rate
        rtf = t_total / audio_dur if audio_dur > 0 else 0

        runs.append({
            "run": r,
            "total_time_s": round(t_total, 2),
            "audio_dur_s": round(audio_dur, 2),
            "rtf": round(rtf, 3),
            "diffusion_time_s": round(diff_time, 2),
            "vocoder_time_s": round(vocoder_time, 2),
            "step_times_s": [round(x, 2) for x in step_times],
        })
        audios.append(audio)
        print(f"  Run {r}: Total = {t_total:5.2f}s | Diffusion = {diff_time:5.2f}s | Vocoder = {vocoder_time:4.2f}s | Audio = {audio_dur:5.2f}s | RTF = {rtf:5.3f}x")

    # Summary Statistics
    total_times = [x["total_time_s"] for x in runs]
    diff_times = [x["diffusion_time_s"] for x in runs]
    voc_times = [x["vocoder_time_s"] for x in runs]
    rtfs = [x["rtf"] for x in runs]

    stats = {
        "mean_time_s": round(float(np.mean(total_times)), 2),
        "median_time_s": round(float(np.median(total_times)), 2),
        "min_time_s": min(total_times),
        "max_time_s": max(total_times),
        "mean_diff_s": round(float(np.mean(diff_times)), 2),
        "mean_voc_s": round(float(np.mean(voc_times)), 2),
        "mean_rtf": round(float(np.mean(rtfs)), 3),
    }

    print("\n" + "-" * 60)
    print(f"BENCHMARK SUMMARY (Batch-1 V2 Turbo):")
    print(f"  • Mean Total Time:     {stats['mean_time_s']:.2f} s")
    print(f"  • Mean Diffusion Time: {stats['mean_diff_s']:.2f} s")
    print(f"  • Mean Vocoder Time:   {stats['mean_voc_s']:.2f} s")
    print(f"  • Mean RTF:            {stats['mean_rtf']:.3f}x")
    print(f"  • Min/Max Time:        {stats['min_time_s']:.2f}s / {stats['max_time_s']:.2f}s")
    print("-" * 60)

    # 3. Save Output & Quality Analysis
    os.makedirs("debug", exist_ok=True)
    out_wav_path = "debug/v2_batch1_test_a.wav"
    sf.write(out_wav_path, audios[0], model.sampling_rate)

    # Numerical Audio Metrics
    data = audios[0]
    sr = model.sampling_rate
    dur = len(data) / sr
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))
    silence_pct = float((np.sum(np.abs(data) < 0.01) / len(data)) * 100)

    # Acoustic and Phoneme Checks
    prosody_metrics = PitchAndEnergyAnalyzer.analyze_prosody_metrics(data, sr)
    phoneme_metrics = PhonemeValidator.evaluate_audio_completeness(data, sr, clean_text)

    print("\n[STEP 4] QUALITY & ACOUSTIC EVALUATION")
    print(f"  • Duration:          {dur:.2f} s")
    print(f"  • RMS Energy:        {rms:.4f}")
    print(f"  • Peak Amplitude:    {peak:.4f}")
    print(f"  • Silence Ratio:     {silence_pct:.1f}%")
    print(f"  • F0 Mean / Std:     {prosody_metrics['f0_mean_hz']:.1f} Hz / {prosody_metrics['f0_std_hz']:.1f} Hz")
    print(f"  • Expressiveness:    {prosody_metrics['expressiveness_grade']}")
    print(f"  • Phoneme Status:    {phoneme_metrics['status']}")

    results = {
        "shapes": {
            "b1_guidance_0": b1_shape,
            "b2_guidance_1_5": b2_shape,
        },
        "runs": runs,
        "stats": stats,
        "audio_metrics": {
            "duration_s": round(dur, 2),
            "rms": round(rms, 4),
            "peak": round(peak, 4),
            "silence_pct": round(silence_pct, 1),
            "f0_mean_hz": prosody_metrics["f0_mean_hz"],
            "f0_std_hz": prosody_metrics["f0_std_hz"],
            "expressiveness": prosody_metrics["expressiveness_grade"],
            "phoneme_status": phoneme_metrics["status"],
        }
    }

    with open("debug/v2_batch1_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE — Results saved to debug/v2_batch1_experiment_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
