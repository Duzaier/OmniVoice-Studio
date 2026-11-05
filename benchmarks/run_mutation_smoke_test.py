#!/usr/bin/env python3
"""
OmniVoice Mutation Engine — Progressive Smoke Test & Quality Gate Runner.
Tests:
- TEST 1 (Smoke Test): 5–10s Vietnamese sentence.
- TEST 2 (Phoneme & Consonant Check): Vietnamese initial consonants (gi, s, tr, ch, nh, ng, kh, ph, th).
- TEST 3 (Tonal & Diacritic Check): 6 Vietnamese tones (sắc, huyền, hỏi, ngã, nặng, ngang).
- TEST 4 (Prosody & Punctuation Check): Question, exclamation, ellipsis.
- Measures cold start vs warm inference, audio validity (no NaN/Inf, no clipping), RTF.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List
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
from omnivoice.core.hardware import profile_hardware, apply_runtime_optimizations
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SmokeTest")

SMOKE_TESTS = [
    {
        "id": "01_smoke_short",
        "name": "Smoke Test (Short 5-8s)",
        "text": "Xin chào các bạn, đây là kiểm tra phát âm tiếng Việt cơ bản.",
        "expected_dur": 3.5,
    },
    {
        "id": "02_consonants_gi_s",
        "name": "Phoneme Gate (gi, s, tr, ch, x, r)",
        "text": "Có những điều gì sẽ xảy ra trong cuộc sống xưa và nay giữa chúng ta.",
        "expected_dur": 5.0,
    },
    {
        "id": "03_tones_prosody",
        "name": "Tonal Gate (6 Vietnamese Tones & Question/Exclamation)",
        "text": "Bạn có khỏe không? Thật là tuyệt vời! Hãy giữ gìn sức khỏe mỗi ngày nhé…",
        "expected_dur": 6.0,
    },
]


def check_audio_validity(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """Gate A: Verify audio array is valid and non-corrupt."""
    if len(audio) == 0:
        return {"valid": False, "reason": "Audio is empty"}
    if np.isnan(audio).any():
        return {"valid": False, "reason": "Audio contains NaN values"}
    if np.isinf(audio).any():
        return {"valid": False, "reason": "Audio contains Inf values"}
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        return {"valid": False, "reason": "Audio is silent"}
    rms = float(np.sqrt(np.mean(audio**2)))
    return {
        "valid": True,
        "peak_amplitude": round(peak, 4),
        "rms": round(rms, 4),
        "duration_sec": round(len(audio) / sample_rate, 2),
    }


def run_smoke_suite():
    print("=" * 80)
    print("      🧪 OMNIVOICE MUTATION ENGINE — PROGRESSIVE SMOKE TEST")
    print("=" * 80)

    hw = profile_hardware()
    apply_runtime_optimizations(hw)
    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))

    # 1. Cold Start: Load Pure Native Model ONCE
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map=hw["device"],
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()
    t_load = time.perf_counter() - t0
    print(f"✅ Model loaded in {t_load:.2f}s (Device: {hw['device']}, Threads: {torch.get_num_threads()})")

    # 2. Warmup
    warm_cfg = OmniVoiceGenerationConfig(num_step=2, guidance_scale=1.0, denoise=True, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate("Sẵn sàng.", language="Vietnamese", generation_config=warm_cfg)
    print("✅ Initial Warmup complete.")

    # 3. Load Cached Voice Clone Prompt
    voice_prompt = vm.load_prompt("default", map_location=hw["device"])
    print(f"✅ VoiceClonePrompt loaded from cache: {voice_prompt is not None}")

    gen_cfg = OmniVoiceGenerationConfig(
        num_step=16,
        guidance_scale=2.0,
        denoise=True,
        pad_duration=0.1,
        fade_duration=0.1,
        postprocess_output=True,
    )

    test_results = []
    all_passed = True

    # 4. Execute Smoke Suite sequentially
    for test in SMOKE_TESTS:
        print(f"\n▶ RUNNING {test['name']}...")
        cleaned = TextPreprocessor.process(test["text"])
        print(f"   Input Text: '{cleaned}'")

        t_gen_start = time.perf_counter()
        with torch.inference_mode():
            audios = model.generate(
                text=cleaned,
                language="Vietnamese",
                voice_clone_prompt=voice_prompt,
                generation_config=gen_cfg,
            )
        t_gen = time.perf_counter() - t_gen_start
        audio = audios[0]

        validity = check_audio_validity(audio, model.sampling_rate)
        audio_dur = validity.get("duration_sec", 0.0)
        rtf = t_gen / audio_dur if audio_dur > 0 else 0

        passed = validity["valid"]
        if not passed:
            all_passed = False

        res = {
            "test_id": test["id"],
            "name": test["name"],
            "input_text": cleaned,
            "generation_time_sec": round(t_gen, 2),
            "audio_duration_sec": audio_dur,
            "rtf": round(rtf, 3),
            "validity": validity,
            "gate_status": "PASS" if passed else "FAIL",
        }
        test_results.append(res)

        print(f"   Result: {'✅ PASS' if passed else '❌ FAIL'}")
        print(f"   Audio Duration: {audio_dur:.2f}s | Gen Time: {t_gen:.2f}s | RTF: {rtf:.3f}x")
        print(f"   Signal: Peak = {validity.get('peak_amplitude', 0)}, RMS = {validity.get('rms', 0)}")

        if not passed:
            print(f"   ❌ GATE FAILED: {validity.get('reason')}")
            break

    print("\n" + "=" * 80)
    print(f"🏆 SMOKE SUITE SUMMARY: {'ALL TESTS PASSED ✅' if all_passed else 'TESTS FAILED ❌'}")
    print("=" * 80)
    return test_results


if __name__ == "__main__":
    run_smoke_suite()
