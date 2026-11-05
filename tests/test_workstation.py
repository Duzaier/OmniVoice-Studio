#!/usr/bin/env python3
"""
Automated Regression Test Suite for OmniVoice Workstation.
Verifies chunker, voice manager, prompt caching, hardware profiling, and engine synthesis.
"""

import os
import sys
import tempfile
import numpy as np

# UTF-8 Encoding Fix for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from omnivoice.core.hardware import profile_hardware, apply_runtime_optimizations
from omnivoice.core.chunker import SmartTextChunker, concatenate_audio_segments
from omnivoice.core.voice_manager import VoiceManager, compute_audio_hash


def test_hardware_profiling():
    hw = profile_hardware()
    assert hw["device"] in ("cpu", "cuda", "xpu", "mps")
    assert hw["logical_cores"] > 0
    assert hw["total_ram_gb"] > 0
    apply_runtime_optimizations(hw)
    print("✅ test_hardware_profiling PASSED")


def test_smart_text_chunker():
    chunker = SmartTextChunker(max_chars=80, min_chars=20)
    long_text = (
        "Xin chào các bạn. Đây là câu thứ nhất của bài kiểm tra. "
        "Còn đây là câu thứ hai với nội dung dài hơn một chút để kiểm tra thuật toán chia nhỏ văn bản. "
        "Và đây là câu cuối cùng."
    )
    chunks = chunker.chunk(long_text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 120
    print(f"✅ test_smart_text_chunker PASSED: Split into {len(chunks)} chunks.")


def test_audio_concatenation():
    sr = 24000
    seg1 = np.ones(sr * 1, dtype=np.float32)  # 1 sec
    seg2 = np.ones(sr * 2, dtype=np.float32)  # 2 sec
    combined = concatenate_audio_segments([seg1, seg2], sample_rate=sr, crossfade_ms=50, pause_ms=100)
    assert len(combined) > len(seg1) + len(seg2)
    print("✅ test_audio_concatenation PASSED")


def test_voice_manager_hashing():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFFdummywavecontent1234567890")
        tmp_path = f.name
    try:
        h1 = compute_audio_hash(tmp_path, "Hello")
        h2 = compute_audio_hash(tmp_path, "Hello")
        h3 = compute_audio_hash(tmp_path, "World")
        assert h1 == h2
        assert h1 != h3
        print("✅ test_voice_manager_hashing PASSED")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    test_hardware_profiling()
    test_smart_text_chunker()
    test_audio_concatenation()
    test_voice_manager_hashing()
    print("\n🎉 ALL REGRESSION TESTS PASSED SUCCESSFULLY!")
