#!/usr/bin/env python3
"""
Vietnamese Phoneme Corpus & Acoustic Completeness Validator for OmniVoice.
Validates initial consonants (S, GI, KH, NG, NH, TR, CH, PH, TH),
final consonants (-T, -C, -P, -M, -N, -NG, -NH), and diacritics.
"""

import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

# Standard Vietnamese Phonetic Benchmark Corpus
VIETNAMESE_PHONEME_CORPUS = {
    "S": ["sao", "say", "sâu", "suối", "sống"],
    "GI": ["gia", "giữa", "giọng", "giải", "giữ"],
    "KH": ["không", "khó", "khác", "không gian"],
    "NG": ["người", "nghe", "nghĩ", "ngày"],
    "NH": ["nhớ", "nhìn", "những"],
    "TR": ["trái", "trưởng", "trời"],
    "CH": ["chúng", "chẳng", "chuyện"],
    "PH": ["phải", "phút", "phát"],
    "TH": ["thời", "thương", "thật"],
    "FINALS": ["sống", "thương", "mất", "khác", "mình", "người", "đúng", "biết", "cách", "hết"],
}


class PhonemeValidator:
    """Validates acoustic completeness and spectral properties for phoneme reproduction."""

    @staticmethod
    def evaluate_audio_completeness(
        audio: np.ndarray,
        sample_rate: int = 24000,
        expected_text: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate if generated audio is acoustically complete, non-clipped,
        has sufficient duration, and contains healthy high-frequency spectral content.
        """
        if audio.ndim > 1:
            audio = audio[0]

        duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
        peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        rms = float(np.sqrt(np.mean(audio**2))) if len(audio) > 0 else 0.0

        is_silent = peak < 0.01 or rms < 0.001
        is_clipped = peak >= 0.999

        # High Frequency Energy Ratio (for fricatives 's', 'kh', 'th' in 3kHz - 10kHz band)
        hf_energy_ratio = 0.0
        if len(audio) >= 1024 and not is_silent:
            # Perform FFT
            fft_mag = np.abs(np.fft.rfft(audio[: min(len(audio), sample_rate * 3)]))
            freqs = np.fft.rfftfreq(min(len(audio), sample_rate * 3), 1.0 / sample_rate)
            
            hf_mask = (freqs >= 2500) & (freqs <= 9000)
            total_power = np.sum(fft_mag**2) + 1e-8
            hf_power = np.sum(fft_mag[hf_mask] ** 2)
            hf_energy_ratio = float(hf_power / total_power)

        word_count = len(expected_text.split()) if expected_text else 1
        expected_min_duration = word_count * 0.15  # ~150ms per word minimum

        completeness_status = "PASS"
        if is_silent:
            completeness_status = "FAIL (Silent Audio)"
        elif is_clipped:
            completeness_status = "WARNING (Audio Clipped)"
        elif duration < expected_min_duration:
            completeness_status = "FAIL (Audio Truncated / Swallowed)"
        elif hf_energy_ratio < 0.01 and any(c in expected_text.lower() for c in ["s", "x", "kh", "th"]):
            completeness_status = "WARNING (Low Fricative Energy)"

        return {
            "duration_sec": round(duration, 3),
            "peak": round(peak, 4),
            "rms": round(rms, 5),
            "hf_energy_ratio": round(hf_energy_ratio, 4),
            "is_silent": is_silent,
            "is_clipped": is_clipped,
            "status": completeness_status,
        }
