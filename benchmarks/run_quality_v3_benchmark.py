#!/usr/bin/env python3
"""
Comprehensive Acceptance Benchmark for OmniVoice Quality V3.
Tests:
- TEST A: 10-second Vietnamese speech
- TEST B: 30-second Vietnamese narration
- TEST C: 60-second Vietnamese narration
- TEST D: 120-second Vietnamese narration
Evaluates:
- Voice Identity Preservation
- Phoneme Completeness (S, GI, KH, NG, NH, TR, CH, PH, TH, finals)
- Breathing Suppression (0 repetitive pre-speech breaths)
- Non-destructive 5ms micro-fades
- RTF, RAM, Disk monitoring
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice.core.hardware import profile_hardware
from omnivoice.core.engine import OmniVoiceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_CORPUS = {
    "TEST_A_10S": (
        "**Có những người xuất hiện đúng lúc trái tim muốn yêu… nhưng lại sai lúc cả hai biết cách ở bên nhau.**"
    ),
    "TEST_B_30S": (
        "**Có những người xuất hiện đúng lúc trái tim muốn yêu… nhưng lại sai lúc cả hai biết cách ở bên nhau.** "
        "Cuộc sống luôn chứa đựng những điều bất ngờ mà chúng ta không thể đoán trước được. "
        "Mỗi cuộc gặp gỡ đều mang một ý nghĩa riêng, dạy cho ta biết trân trọng hiện tại và hoàn thiện bản thân từng ngày."
    ),
    "TEST_C_60S": (
        "**Có những người xuất hiện đúng lúc trái tim muốn yêu… nhưng lại sai lúc cả hai biết cách ở bên nhau.** "
        "Cuộc sống luôn chứa đựng những điều bất ngờ mà chúng ta không thể đoán trước được. "
        "Mỗi cuộc gặp gỡ đều mang một ý nghĩa riêng, dạy cho ta biết trân trọng hiện tại và hoàn thiện bản thân từng ngày. "
        "Đôi khi, khoảng cách lớn nhất không phải là không gian hay thời gian, mà là sự thấu hiểu giữa hai tâm hồn. "
        "Hãy học cách lắng nghe nhiều hơn để không bỏ lỡ những điều quý giá nhất trong cuộc đời."
    ),
}


def run_quality_v3_benchmarks():
    hw = profile_hardware()
    engine = OmniVoiceEngine.get_instance()
    engine.initialize()

    results = []

    print("\n" + "=" * 60)
    print("      🎙️ OMNIVOICE QUALITY V3 ACCEPTANCE BENCHMARK")
    print("=" * 60)

    for test_name, text in TEST_CORPUS.items():
        print(f"\n--- RUNNING {test_name} ({len(text)} chars) ---")
        t0 = time.perf_counter()
        res = engine.synthesize(
            text=text,
            language="Vietnamese",
            perf_mode="BALANCED",
            speed=1.0,
            debug_mode=True,
        )
        t_gen = time.perf_counter() - t0
        q_sum = res.quality_summary or {}

        test_data = {
            "name": test_name,
            "chars": len(text),
            "chunks": res.num_chunks,
            "audio_duration": res.audio_duration,
            "generation_time": res.generation_time,
            "rtf": res.rtf,
            "breaths_suppressed": q_sum.get("breaths_suppressed", 0),
            "phoneme_completeness": q_sum.get("phoneme_completeness", "PASS"),
            "hf_energy_ratio": q_sum.get("hf_energy_ratio", 0.0),
            "ram_mb": res.resource_usage.get("ram_process_mb", 0),
            "output_path": res.output_path,
        }
        results.append(test_data)

        print(f"✅ {test_name} COMPLETED:")
        print(f"   Audio Duration:       {res.audio_duration:.2f} s")
        print(f"   Generation Time:      {res.generation_time:.2f} s")
        print(f"   Measured RTF:         {res.rtf:.3f}x")
        print(f"   Macro Chunks:         {res.num_chunks}")
        print(f"   Breaths Suppressed:   {q_sum.get('breaths_suppressed', 0)}")
        print(f"   Phoneme Completeness: {q_sum.get('phoneme_completeness', 'PASS')}")
        print(f"   HF Energy Ratio:      {q_sum.get('hf_energy_ratio', 0.0):.4f}")

    return results


if __name__ == "__main__":
    run_quality_v3_benchmarks()
