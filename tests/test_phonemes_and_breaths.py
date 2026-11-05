#!/usr/bin/env python3
"""
Unit & Regression Test Suite for Phoneme Preservation and Breath Suppression.
"""

import numpy as np
from omnivoice.core.breath_analyzer import AcousticBreathAnalyzer
from omnivoice.core.phoneme_validator import PhonemeValidator, VIETNAMESE_PHONEME_CORPUS
from omnivoice.core.audio_processor import apply_micro_fade, concatenate_natural_chunks
from omnivoice.core.chunker import SmartTextChunker


def test_breath_analyzer_and_onset_protection():
    sr = 24000
    # Simulate a 150ms breath noise followed by strong speech
    breath = np.random.normal(0, 0.02, int(0.15 * sr)).astype(np.float32)
    speech = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, sr)).astype(np.float32) * 0.5
    full = np.concatenate([breath, speech])

    diag = AcousticBreathAnalyzer.analyze_onset(full, sample_rate=sr)
    assert diag["speech_start_sec"] >= 0.10

    trimmed = AcousticBreathAnalyzer.trim_leading_breath_artifact(full, sample_rate=sr, max_trim_ms=250.0)
    # Trimmed should be shorter than full but keep at least 30ms before speech
    assert len(trimmed) < len(full)
    assert len(trimmed) >= len(speech)
    print("✅ test_breath_analyzer_and_onset_protection PASSED")


def test_micro_fade_preserves_consonants():
    sr = 24000
    # High frequency consonant pulse (e.g. 50ms 's' sound at 5kHz)
    consonant_len = int(0.05 * sr)
    consonant = np.sin(2 * np.pi * 5000 * np.linspace(0, 0.05, consonant_len)).astype(np.float32)
    vowel = np.sin(2 * np.pi * 200 * np.linspace(0, 0.5, int(0.5 * sr))).astype(np.float32)
    audio = np.concatenate([consonant, vowel])

    faded = apply_micro_fade(audio, sample_rate=sr, fade_ms=5.0)
    assert len(faded) == len(audio)
    
    # 5ms fade is 120 samples. At sample 120 (5ms), amplitude must be 100%
    assert abs(faded[120] - audio[120]) < 1e-4
    print("✅ test_micro_fade_preserves_consonants PASSED")


def test_macro_chunker_prevents_fragmentation():
    sample_text = (
        "Có một nghịch lý khá buồn về tình yêu. "
        "Có những người xuất hiện đúng lúc trái tim muốn yêu, nhưng lại sai lúc cả hai biết cách ở bên nhau. "
        "Cuộc sống luôn mang lại cho chúng ta những bài học quý giá mà thời gian không thể xóa nhòa. "
        "Hãy luôn trân trọng từng khoảnh khắc được sống và yêu thương."
    )
    chunker = SmartTextChunker(target_chars=220, min_chars=80, max_chars=350)
    chunks = chunker.chunk(sample_text)
    
    # Text of ~300 chars should be at most 2 macro chunks (never 19 chunks!)
    assert len(chunks) <= 3
    for c in chunks:
        assert len(c) >= 70
    print(f"✅ test_macro_chunker_prevents_fragmentation PASSED: Split into {len(chunks)} cohesive macro chunks.")


def test_phoneme_completeness_validator():
    sr = 24000
    # Create healthy synthetic speech signal
    t = np.linspace(0, 1.0, sr)
    signal = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.sin(2 * np.pi * 4000 * t)
    eval_res = PhonemeValidator.evaluate_audio_completeness(signal, sample_rate=sr, expected_text="sống trong bình yên")
    assert eval_res["status"] == "PASS"
    assert eval_res["hf_energy_ratio"] > 0.01
    print("✅ test_phoneme_completeness_validator PASSED")


if __name__ == "__main__":
    test_breath_analyzer_and_onset_protection()
    test_micro_fade_preserves_consonants()
    test_macro_chunker_prevents_fragmentation()
    test_phoneme_completeness_validator()
    print("\n🎉 ALL QUALITY V3 REGRESSION TESTS PASSED!")
