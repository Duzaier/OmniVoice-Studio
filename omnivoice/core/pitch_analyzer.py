#!/usr/bin/env python3
"""
Acoustic Pitch (F0) and Energy Contour Analyzer for OmniVoice TTS.
Measures:
- F0 Mean, F0 Standard Deviation (Pitch Variation / Expressiveness)
- RMS Energy Contour & Dynamics
- Sentence Cadence & Pause Duration Verification
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class PitchAndEnergyAnalyzer:
    """Analyzes acoustic prosody, pitch variation (F0), and dynamic energy contours."""

    @staticmethod
    def estimate_f0_autocorr(
        audio: np.ndarray,
        sample_rate: int = 24000,
        fmin: float = 60.0,
        fmax: float = 400.0,
        frame_ms: float = 30.0,
        hop_ms: float = 10.0,
    ) -> np.ndarray:
        """
        Estimate Fundamental Frequency (F0) contour across time using normalized autocorrelation.
        """
        if audio.ndim > 1:
            audio = audio[0]

        frame_len = int((frame_ms / 1000.0) * sample_rate)
        hop_len = int((hop_ms / 1000.0) * sample_rate)

        min_lag = int(sample_rate / fmax)
        max_lag = int(sample_rate / fmin)

        num_frames = max(1, (len(audio) - frame_len) // hop_len)
        f0_contour = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            frame = audio[i * hop_len : i * hop_len + frame_len]
            # Zero-center and window
            frame = frame - np.mean(frame)
            if np.max(np.abs(frame)) < 1e-4:
                continue

            # Autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(frame) - 1 :]  # Non-negative lags only
            
            # Avoid division by zero
            if corr[0] <= 1e-6:
                continue

            norm_corr = corr / (corr[0] + 1e-8)
            search_window = norm_corr[min_lag : min(max_lag, len(norm_corr))]
            if len(search_window) == 0:
                continue

            peak_lag_rel = np.argmax(search_window)
            peak_val = search_window[peak_lag_rel]
            peak_lag = min_lag + peak_lag_rel

            # Voiced threshold: correlation peak >= 0.35
            if peak_val >= 0.35 and peak_lag > 0:
                f0_contour[i] = sample_rate / peak_lag

        return f0_contour

    @classmethod
    def analyze_prosody_metrics(
        cls,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive prosody statistics: F0 Mean, F0 Std, Dynamic Range, RMS Energy.
        """
        if audio.ndim > 1:
            audio = audio[0]

        f0_contour = cls.estimate_f0_autocorr(audio, sample_rate=sample_rate)
        voiced_f0 = f0_contour[f0_contour > 0]

        if len(voiced_f0) > 0:
            f0_mean = float(np.mean(voiced_f0))
            f0_std = float(np.std(voiced_f0))
            f0_min = float(np.min(voiced_f0))
            f0_max = float(np.max(voiced_f0))
            f0_range = f0_max - f0_min
            voiced_pct = float(len(voiced_f0) / len(f0_contour) * 100.0)
        else:
            f0_mean = 0.0
            f0_std = 0.0
            f0_min = 0.0
            f0_max = 0.0
            f0_range = 0.0
            voiced_pct = 0.0

        # RMS Energy Contour (100ms frames)
        frame_len = int(0.1 * sample_rate)
        rms_frames = []
        for i in range(0, max(1, len(audio) - frame_len), frame_len // 2):
            rms_frames.append(float(np.sqrt(np.mean(audio[i : i + frame_len] ** 2))))

        rms_mean = float(np.mean(rms_frames)) if rms_frames else 0.0
        rms_std = float(np.std(rms_frames)) if rms_frames else 0.0

        # Assess Expressiveness Grade
        # Flat speech typically has f0_std < 10 Hz; expressive natural speech has f0_std >= 18 Hz
        if f0_std >= 18.0:
            expressiveness = "EXPRESSIVE (Healthy pitch dynamics)"
        elif f0_std >= 10.0:
            expressiveness = "MODERATE (Standard narrative)"
        else:
            expressiveness = "MONOTONE (Low pitch variation)"

        return {
            "f0_mean_hz": round(f0_mean, 2),
            "f0_std_hz": round(f0_std, 2),
            "f0_range_hz": round(f0_range, 2),
            "f0_min_hz": round(f0_min, 2),
            "f0_max_hz": round(f0_max, 2),
            "voiced_percentage": round(voiced_pct, 1),
            "rms_mean": round(rms_mean, 4),
            "rms_std": round(rms_std, 4),
            "expressiveness_grade": expressiveness,
        }
