#!/usr/bin/env python3
"""
Unit & Regression Tests for Chunker Quality and Audio Concatenation.
"""

import numpy as np
from omnivoice.core.chunker import SmartTextChunker, concatenate_audio_segments


def test_chunker_prevents_tiny_fragments():
    text = (
        "Cuộc sống là một hành trình dài với nhiều thử thách và cơ hội. "
        "Mỗi ngày trôi qua đều mang lại cho chúng ta những bài học quý giá. "
        "Đừng ngần ngại bước tiếp khi gặp khó khăn, bởi vì sau cơn mưa trời lại sáng. "
        "Hãy luôn giữ vững niềm tin và nỗ lực hết mình vì mục tiêu đã chọn."
    )
    chunker = SmartTextChunker(target_chars=140, min_chars=60, max_chars=240)
    chunks = chunker.chunk(text)
    
    assert len(chunks) >= 1
    for c in chunks:
        # Each chunk should have sufficient context (unless very end)
        assert len(c) >= 50, f"Chunk too small: '{c}' ({len(c)} chars)"
        # No Markdown asterisks
        assert "**" not in c


def test_concatenation_and_fades():
    sr = 24000
    seg1 = np.ones(sr * 1, dtype=np.float32) * 0.5
    seg2 = np.ones(sr * 1, dtype=np.float32) * 0.5
    
    result = concatenate_audio_segments([seg1, seg2], sample_rate=sr, pause_ms=100.0, fade_ms=5.0)
    # Total duration = 1s + 0.1s pause + 1s = 2.1s = 50400 samples
    expected_len = int((1.0 + 0.1 + 1.0) * sr)
    assert abs(len(result) - expected_len) <= 10


if __name__ == "__main__":
    test_chunker_prevents_tiny_fragments()
    test_concatenation_and_fades()
    print("✅ ALL CHUNKER QUALITY REGRESSION TESTS PASSED!")
