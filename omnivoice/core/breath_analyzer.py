#!/usr/bin/env python3
"""
Acoustic Breath & Transient Analyzer for OmniVoice TTS.
Detects and distinguishes natural speech pauses from repetitive pre-speech
inhalation artifacts without clipping initial unvoiced consonants ('s', 'kh', 'th', 'tr').
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class AcousticBreathAnalyzer:
    """Analyzes audio waveforms for onset latency, breathing energy, and speech start."""

    @staticmethod
    def analyze_onset(
        waveform: np.ndarray,
        sample_rate: int = 24000,
        frame_ms: float = 20.0,
        energy_threshold_db: float = -22.0,
    ) -> Dict[str, Any]:
        """
        Analyze the first 800ms of a generated waveform to locate speech start
        and identify whether pre-speech breath/noise is present.
        """
        if waveform.ndim > 1:
            audio = waveform[0] if waveform.shape[0] < waveform.shape[1] else waveform[:, 0]
        else:
            audio = waveform

        if len(audio) == 0:
            return {
                "speech_start_sec": 0.0,
                "breath_detected": False,
                "breath_duration_sec": 0.0,
                "breath_energy_rms": 0.0,
                "max_peak": 0.0,
            }

        max_peak = float(np.max(np.abs(audio)))
        if max_peak < 1e-5:
            return {
                "speech_start_sec": 0.0,
                "breath_detected": False,
                "breath_duration_sec": 0.0,
                "breath_energy_rms": 0.0,
                "max_peak": 0.0,
            }

        frame_size = int((frame_ms / 1000.0) * sample_rate)
        num_frames = min(len(audio) // frame_size, int(1.0 * sample_rate // frame_size))

        frame_rms = []
        for i in range(num_frames):
            frame = audio[i * frame_size : (i + 1) * frame_size]
            rms = np.sqrt(np.mean(frame**2))
            frame_rms.append(rms)

        peak_rms = max(frame_rms) if frame_rms else 1e-4
        threshold_rms = max(1e-4, peak_rms * (10 ** (energy_threshold_db / 20.0)))

        # Find first speech frame
        speech_frame_idx = 0
        for i, rms in enumerate(frame_rms):
            if rms >= threshold_rms:
                speech_frame_idx = i
                break

        speech_start_sec = (speech_frame_idx * frame_size) / sample_rate

        # Analyze pre-speech region (is there a low-energy breath transient?)
        breath_detected = False
        breath_duration = 0.0
        breath_energy = 0.0

        if speech_frame_idx > 1:
            pre_speech_samples = audio[: speech_frame_idx * frame_size]
            pre_rms = float(np.sqrt(np.mean(pre_speech_samples**2)))
            # If pre-speech has noticeable energy (> -55dB of peak) but below speech threshold
            if pre_rms > (peak_rms * 0.01) and pre_rms < threshold_rms:
                breath_detected = True
                breath_duration = speech_start_sec
                breath_energy = pre_rms

        return {
            "speech_start_sec": round(speech_start_sec, 3),
            "breath_detected": breath_detected,
            "breath_duration_sec": round(breath_duration, 3),
            "breath_energy_rms": round(breath_energy, 5),
            "max_peak": round(max_peak, 4),
        }

    @staticmethod
    def trim_leading_breath_artifact(
        waveform: np.ndarray,
        sample_rate: int = 24000,
        max_trim_ms: float = 200.0,
    ) -> np.ndarray:
        """
        Gently trim pre-speech breath/silence up to max_trim_ms, preserving at least 30ms
        headroom before voiced onset so initial consonants ('s', 'kh', 'th') are never clipped.
        """
        if waveform.ndim > 1:
            audio = waveform[0]
            is_2d = True
        else:
            audio = waveform
            is_2d = False

        if len(audio) < int(0.1 * sample_rate):
            return waveform

        frame_size = int(0.01 * sample_rate)  # 10ms frame
        max_search_frames = int((max_trim_ms / 1000.0) * sample_rate // frame_size)
        
        # Calculate frame RMS across the first 1 second
        search_samples = min(len(audio), int(1.0 * sample_rate))
        frame_rms = [
            np.sqrt(np.mean(audio[i : i + frame_size] ** 2))
            for i in range(0, search_samples - frame_size, frame_size)
        ]
        if not frame_rms:
            return waveform

        peak_rms = max(frame_rms)
        if peak_rms < 1e-4:
            return waveform

        threshold_rms = peak_rms * 0.15  # Speech onset at 15% of peak RMS
        onset_frame = 0
        for f, rms in enumerate(frame_rms[:max_search_frames]):
            if rms >= threshold_rms:
                onset_frame = f
                break

        # Keep 30ms buffer before onset to protect unvoiced fricatives
        safety_buffer_samples = int(0.03 * sample_rate)
        trim_samples = max(0, onset_frame * frame_size - safety_buffer_samples)

        if trim_samples > 0:
            trimmed = audio[trim_samples:]
            return trimmed[np.newaxis, :] if is_2d else trimmed
        return waveform
