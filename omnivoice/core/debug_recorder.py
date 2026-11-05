#!/usr/bin/env python3
"""
Chunk Quality Debug Recorder for OmniVoice.
Saves granular per-chunk audio, input text, and timing/memory metadata
to pinpoint any muffled syllables, swallowed phonemes, or breathing anomalies.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import soundfile as sf

from omnivoice.core.breath_analyzer import AcousticBreathAnalyzer
from omnivoice.core.phoneme_validator import PhonemeValidator

logger = logging.getLogger(__name__)


class QualityDebugRecorder:
    """Records per-chunk debug information to disk."""

    def __init__(self, debug_dir: Optional[Path] = None):
        if debug_dir is None:
            debug_dir = Path(__file__).parent.parent.parent / "debug" / "chunks"
        self.debug_dir = Path(debug_dir).resolve()
        self.debug_dir.mkdir(parents=True, exist_ok=True)

    def record_chunk(
        self,
        session_id: str,
        chunk_idx: int,
        text: str,
        audio: np.ndarray,
        sample_rate: int,
        generation_time: float,
        device: str,
        dtype: str = "float32",
        model: str = "k2-fsa/OmniVoice",
        voice: str = "default",
        num_step: int = 16,
        guidance_scale: float = 2.0,
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """Save chunk artifact files (001_input.txt, 001.wav, 001.json)."""
        session_folder = self.debug_dir / session_id
        session_folder.mkdir(parents=True, exist_ok=True)

        prefix = f"{chunk_idx:03d}"
        wav_file = session_folder / f"{prefix}.wav"
        txt_file = session_folder / f"{prefix}_input.txt"
        meta_file = session_folder / f"{prefix}.json"

        # Save audio
        sf.write(str(wav_file), audio, sample_rate)

        # Save input text
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(text)

        duration = len(audio) / sample_rate if sample_rate > 0 else 0
        rtf = generation_time / duration if duration > 0 else 0

        # Breath and Phoneme Analysis
        onset_diag = AcousticBreathAnalyzer.analyze_onset(audio, sample_rate=sample_rate)
        phoneme_diag = PhonemeValidator.evaluate_audio_completeness(audio, sample_rate=sample_rate, expected_text=text)

        metadata = {
            "chunk_id": chunk_idx,
            "session_id": session_id,
            "text": text,
            "text_length": len(text),
            "duration": round(duration, 3),
            "generation_time": round(generation_time, 3),
            "rtf": round(rtf, 3),
            "device": device,
            "dtype": dtype,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "speed": speed,
            "model": model,
            "voice": voice,
            "breath_detected": onset_diag["breath_detected"],
            "speech_start": onset_diag["speech_start_sec"],
            "breath_duration": onset_diag["breath_duration_sec"],
            "breath_energy": onset_diag["breath_energy_rms"],
            "hf_energy_ratio": phoneme_diag["hf_energy_ratio"],
            "status": phoneme_diag["status"],
            "wav_path": str(wav_file),
            "txt_path": str(txt_file),
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.debug(f"[QualityDebugRecorder] Recorded {prefix} to {session_folder}")
        return metadata
