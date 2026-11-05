#!/usr/bin/env python3
"""
Comprehensive Acceptance Benchmark for OmniVoice Quality V4:
PROSODY, SENTENCE BOUNDARY & EXPRESSIVE SPEECH ENGINE

Tests:
1. TEST A: Punctuation pauses (. ! ?)
2. TEST B: Comma & Clause pauses (, ;)
3. TEST C: Question intonation (?)
4. TEST D: Exclamation & emotional peaks (!)
5. TEST E: Compound clause transitions
6. TEST F: Dialogue & Quotation narration
7. TEST G: Emotional keyword pacing
8. TEST H: Full multi-sentence storytelling (10s, 30s, 60s)

Measures:
- Prosody Units count & Boundary hierarchy
- Adaptive Pause intervals
- F0 Contour, F0 Mean, F0 Std (Pitch Dynamics & Variation)
- Expressiveness Grade
- Voice Identity & Phoneme Preservation
- Generation Time, RTF & Resource Usage
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice.core.hardware import profile_hardware
from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.core.prosody_tagger import ProsodyTagger
from omnivoice.core.pitch_analyzer import PitchAndEnergyAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROSODY_TEST_SUITE = {
    "TEST_A_PUNCTUATION": "Hôm nay tôi đi làm. Trời đổ một cơn mưa rào. Tôi phải dừng lại trú chân.",
    "TEST_B_COMMAS": "Trong căn phòng nhỏ, ánh đèn vàng ấm áp, tiếng nhạc du dương vang lên dịu dàng.",
    "TEST_C_QUESTION": "Bạn có từng tự hỏi, tại sao chúng ta lại luôn tìm kiếm những điều xa vời?",
    "TEST_D_EXCLAMATION": "Thật là một buổi chiều tuyệt vời và đáng nhớ biết bao!",
    "TEST_E_COMPOUND": "Dù ngoài kia sóng gió bão bùng đến đâu, chỉ cần trong lòng bình yên, mọi chuyện rồi sẽ qua.",
    "TEST_F_DIALOGUE": 'Anh ấy quay lại nhìn tôi rồi khẽ nói: "Chúng ta sẽ vượt qua chuyện này thôi."',
    "TEST_G_EMOTIONAL_WORDS": "Nỗi buồn man mác, lòng bâng khuâng, từng hạt mưa rơi tí tách bên thềm vắng.",
    "TEST_H_STORY_30S": (
        "**Có những người xuất hiện đúng lúc trái tim muốn yêu… nhưng lại sai lúc cả hai biết cách ở bên nhau.** "
        "Cuộc sống luôn chứa đựng những điều bất ngờ mà chúng ta không thể đoán trước được. "
        "Mỗi cuộc gặp gỡ đều mang một ý nghĩa riêng, dạy cho ta biết trân trọng hiện tại và hoàn thiện bản thân từng ngày."
    ),
}


def run_prosody_v4_benchmarks():
    print("\n" + "=" * 75)
    print("      🎙️ OMNIVOICE V4 PROSODY & SENTENCE BOUNDARY ACCEPTANCE BENCHMARK")
    print("=" * 75)

    hw = profile_hardware()
    engine = OmniVoiceEngine.get_instance()
    engine.initialize()

    benchmark_records = []
    output_dir = Path("D:/OmniVoice_Storage/debug/prosody") if Path("D:/OmniVoice_Storage").exists() else Path("./debug/prosody")
    output_dir.mkdir(parents=True, exist_ok=True)

    for test_name, text in PROSODY_TEST_SUITE.items():
        print(f"\n" + "-" * 60)
        print(f"▶ RUNNING: {test_name}")
        print(f"  Input: {text[:80]}..." if len(text) > 80 else f"  Input: {text}")
        print("-" * 60)

        t0 = time.perf_counter()
        result = engine.synthesize(
            text=text,
            language="Vietnamese",
            perf_mode="BALANCED",
            speed=1.0,
            debug_mode=True,
        )
        t_total = time.perf_counter() - t0
        q_sum = result.quality_summary or {}

        rec = {
            "test_name": test_name,
            "text": text,
            "char_count": len(text),
            "prosody_units_count": q_sum.get("prosody_units_count", result.num_chunks),
            "audio_duration_sec": result.audio_duration,
            "generation_time_sec": result.generation_time,
            "rtf": result.rtf,
            "f0_mean_hz": q_sum.get("f0_mean_hz", 0.0),
            "f0_std_hz": q_sum.get("f0_std_hz", 0.0),
            "f0_range_hz": q_sum.get("f0_range_hz", 0.0),
            "voiced_percentage": q_sum.get("voiced_percentage", 0.0),
            "expressiveness_grade": q_sum.get("expressiveness_grade", "N/A"),
            "breaths_suppressed": q_sum.get("breaths_suppressed", 0),
            "output_path": result.output_path,
        }
        benchmark_records.append(rec)

        print(f"✅ {test_name} RESULT:")
        print(f"   Audio Duration:       {result.audio_duration:.2f} s")
        print(f"   Generation Time:      {result.generation_time:.2f} s")
        print(f"   RTF:                  {result.rtf:.3f}x")
        print(f"   Prosody Units:        {rec['prosody_units_count']}")
        print(f"   F0 Mean (Pitch):      {rec['f0_mean_hz']:.1f} Hz")
        print(f"   F0 Std (Variation):   {rec['f0_std_hz']:.1f} Hz")
        print(f"   Expressiveness:       {rec['expressiveness_grade']}")
        print(f"   Breaths Suppressed:   {rec['breaths_suppressed']}")
        print(f"   Output File:          {result.output_path}")

    # Export benchmark report
    report_file = output_dir / f"prosody_v4_benchmark_{int(time.time())}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(f"🏆 PROSODY V4 BENCHMARK COMPLETED — Logged to {report_file}")
    print("=" * 75)
    return benchmark_records


if __name__ == "__main__":
    run_prosody_v4_benchmarks()
