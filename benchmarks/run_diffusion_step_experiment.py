#!/usr/bin/env python3
"""
OmniVoice V2.2 Single-Variable Diffusion Step Experiment Benchmark.
Tests:
- Config A: num_step = 4 (Baseline)
- Config B: num_step = 3
- Config C: num_step = 2
Under exact identical conditions (guidance_scale=0.0, Batch-1, cached prompt, Test A 220 chars).
Runs Warmup (1 run) + Measured (3 runs) per configuration.
Logs detailed latency, audio duration, RTF, RMS, peak, silence ratio, F0 pitch metrics, and phoneme completeness.
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


def run_experiment():
    print("=" * 80)
    print("      🧪 OMNIVOICE V2.2 — SINGLE-VARIABLE DIFFUSION STEP EXPERIMENT")
    print("=" * 80)

    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    torch.set_num_threads(8)

    print("\n[PHASE 1] INITIALIZING MODEL & CACHED VOICE PROMPT")
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    vp = vm.load_prompt("default", map_location="cpu")
    clean_text = TextPreprocessor.process(TEXT_220)

    step_configs = [4, 3, 2]
    all_results = {}
    os.makedirs("debug", exist_ok=True)

    orig_forward = model.forward

    for steps in step_configs:
        print("\n" + "=" * 65)
        print(f"▶ TESTING CONFIGURATION: num_step = {steps}")
        print("=" * 65)

        cfg = OmniVoiceGenerationConfig(
            num_step=steps,
            guidance_scale=0.0,
            denoise=True,
            pad_duration=0.1,
            fade_duration=0.1,
            postprocess_output=True,
        )

        # 1. Warmup
        print(f"  • Running Warmup Pass (num_step={steps})...")
        with torch.inference_mode():
            _ = model.generate(clean_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)
        print(f"  • Warmup Pass Complete.")

        # 2. Measured 3 Runs
        runs_data = []
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
                    generation_config=cfg,
                )
            t_total = time.perf_counter() - t_start
            model.forward = orig_forward

            diff_time = sum(step_times)
            vocoder_time = max(0.0, t_total - diff_time)
            audio = res_audios[0]
            audio_dur = len(audio) / model.sampling_rate
            rtf = t_total / audio_dur if audio_dur > 0 else 0

            runs_data.append({
                "run": r,
                "total_time_s": round(t_total, 2),
                "audio_dur_s": round(audio_dur, 2),
                "rtf": round(rtf, 3),
                "diffusion_time_s": round(diff_time, 2),
                "vocoder_time_s": round(vocoder_time, 2),
                "step_times_s": [round(x, 2) for x in step_times],
            })
            audios.append(audio)
            print(f"    Run {r}: Total = {t_total:5.2f}s | Diffusion = {diff_time:5.2f}s | Vocoder = {vocoder_time:4.2f}s | Audio = {audio_dur:5.2f}s | RTF = {rtf:5.3f}x")

        # Save representative audio
        out_wav_path = f"debug/diffusion_step_{steps}.wav"
        sf.write(out_wav_path, audios[0], model.sampling_rate)
        file_size_bytes = os.path.getsize(out_wav_path)

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

        total_times = [x["total_time_s"] for x in runs_data]
        diff_times = [x["diffusion_time_s"] for x in runs_data]
        voc_times = [x["vocoder_time_s"] for x in runs_data]
        rtfs = [x["rtf"] for x in runs_data]

        config_summary = {
            "num_step": steps,
            "runs": runs_data,
            "stats": {
                "mean_total_time_s": round(float(np.mean(total_times)), 2),
                "median_total_time_s": round(float(np.median(total_times)), 2),
                "min_total_time_s": min(total_times),
                "max_total_time_s": max(total_times),
                "mean_diffusion_time_s": round(float(np.mean(diff_times)), 2),
                "mean_vocoder_time_s": round(float(np.mean(voc_times)), 2),
                "mean_rtf": round(float(np.mean(rtfs)), 3),
            },
            "audio_metrics": {
                "file_path": out_wav_path,
                "file_size_bytes": file_size_bytes,
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

        all_results[f"step_{steps}"] = config_summary

        print(f"\n  ✅ Summary for num_step={steps}:")
        print(f"     • Mean Time:      {config_summary['stats']['mean_total_time_s']} s")
        print(f"     • Mean Diffusion: {config_summary['stats']['mean_diffusion_time_s']} s")
        print(f"     • Mean Vocoder:   {config_summary['stats']['mean_vocoder_time_s']} s")
        print(f"     • Mean RTF:       {config_summary['stats']['mean_rtf']}x")
        print(f"     • RMS / Peak:     {rms:.4f} / {peak:.4f}")
        print(f"     • F0 Mean / Std:  {prosody_metrics['f0_mean_hz']:.1f} Hz / {prosody_metrics['f0_std_hz']:.1f} Hz")
        print(f"     • Phoneme Status: {phoneme_metrics['status']}")

    # Save complete experiment results JSON
    json_path = "debug/v2_2_diffusion_step_experiment.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"DIFFUSION STEP EXPERIMENT COMPLETE — Saved to {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
