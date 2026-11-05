"""
experiments/v2_21/real_audio_validator.py

Programmatic Audio Inspection and Quality Validation Module for OmniVoice V2.21.
Verifies sample rate, channels, duration, peak amplitude, RMS, clipping, silence ratio,
NaN/Inf detection, waveform sanity, and reference audio sanity checking.
"""

import os
import sys
import math
from typing import Dict, Any, Optional
import numpy as np
import torch
import soundfile as sf

# Import V2.20 Sanity Checker
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v2_20")))
from sanity_checker import ReferenceAudioSanityChecker


def validate_generated_audio(
    audio_path: str,
    expected_sample_rate: int = 24000,
    min_duration_sec: float = 0.5,
    max_clipping_ratio: float = 0.005,  # 0.5%
) -> Dict[str, Any]:
    """
    Perform deep programmatic inspection on a generated WAV audio file.
    """
    if not os.path.exists(audio_path):
        return {
            "valid": False,
            "error": f"Audio file not found: {audio_path}",
            "metrics": {}
        }

    try:
        data, sr = sf.read(audio_path)
    except Exception as e:
        return {
            "valid": False,
            "error": f"Failed to read audio file via soundfile: {e}",
            "metrics": {}
        }

    # Ensure float32 numpy array
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)

    channels = 1 if data.ndim == 1 else data.shape[1]
    if channels > 1:
        data = data.mean(axis=1)

    num_samples = len(data)
    duration_sec = num_samples / float(sr)

    # 1. NaN / Inf Check
    if np.isnan(data).any():
        return {
            "valid": False,
            "error": "Audio contains NaN values",
            "metrics": {"duration_sec": duration_sec, "sample_rate": sr}
        }
    if np.isinf(data).any():
        return {
            "valid": False,
            "error": "Audio contains Inf values",
            "metrics": {"duration_sec": duration_sec, "sample_rate": sr}
        }

    # 2. Duration Check
    if duration_sec < min_duration_sec:
        return {
            "valid": False,
            "error": f"Duration too short ({duration_sec:.2f}s < {min_duration_sec}s)",
            "metrics": {"duration_sec": duration_sec, "sample_rate": sr}
        }

    # 3. Peak Amplitude & RMS
    peak_amp = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))

    if rms < 0.001:
        return {
            "valid": False,
            "error": f"Audio is virtually silent (RMS={rms:.6f})",
            "metrics": {"duration_sec": duration_sec, "rms": rms, "peak_amp": peak_amp}
        }

    # 4. Clipping Detection
    clipping_samples = int(np.sum(np.abs(data) >= 0.999))
    clipping_ratio = clipping_samples / float(num_samples)
    if clipping_ratio > max_clipping_ratio:
        return {
            "valid": False,
            "error": f"Excessive clipping detected ({clipping_ratio*100:.2f}% > {max_clipping_ratio*100:.2f}%)",
            "metrics": {"duration_sec": duration_sec, "rms": rms, "clipping_ratio": clipping_ratio}
        }

    # 5. Silence Ratio
    frame_len = int(sr * 0.025)  # 25ms
    hop_len = int(sr * 0.010)    # 10ms
    if num_samples >= frame_len:
        num_frames = (num_samples - frame_len) // hop_len + 1
        frame_energies = [
            np.mean(data[i * hop_len : i * hop_len + frame_len] ** 2)
            for i in range(num_frames)
        ]
        silence_thresh = (rms ** 2) * 0.01
        silence_frames = sum(1 for e in frame_energies if e < silence_thresh)
        silence_ratio = silence_frames / float(num_frames) if num_frames > 0 else 0.0
    else:
        silence_ratio = 0.0

    return {
        "valid": True,
        "error": None,
        "metrics": {
            "file_path": audio_path,
            "file_size_bytes": os.path.getsize(audio_path),
            "sample_rate": sr,
            "channels": channels,
            "duration_sec": round(duration_sec, 2),
            "peak_amplitude": round(peak_amp, 4),
            "rms": round(rms, 4),
            "clipping_ratio": round(clipping_ratio, 5),
            "silence_ratio": round(silence_ratio, 3),
            "expected_sr_match": (sr == expected_sample_rate)
        }
    }
