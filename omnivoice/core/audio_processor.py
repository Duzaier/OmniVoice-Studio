#!/usr/bin/env python3
"""
Acoustic Audio Processor & Seamless Concatenator for OmniVoice TTS.
Provides non-destructive micro-fading (5ms), natural pause structure,
and prevents consonant truncation or clipping across chunks.
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


def prepare_reference_audio(
    audio_path: str,
    target_sr: int = 24000,
    max_duration: float = 8.0,
) -> Tuple[np.ndarray, str]:
    """
    Load, convert to mono, resample, trim pre-speech breath/silence,
    and cap reference audio to optimal duration (3-8 seconds).
    """
    import tempfile
    from omnivoice.core.breath_analyzer import AcousticBreathAnalyzer

    data, sr = sf.read(audio_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        tensor_wav = torch.from_numpy(data).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor_wav, orig_freq=sr, new_freq=target_sr)
        data = resampled.squeeze(0).numpy()

    # Trim leading breath artifact from reference
    data = AcousticBreathAnalyzer.trim_leading_breath_artifact(data, sample_rate=target_sr, max_trim_ms=300.0)

    # Cap to max_duration
    max_samples = int(max_duration * target_sr)
    if len(data) > max_samples:
        data = data[:max_samples]

    # Save to a temporary clean wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        clean_path = tmp.name
    sf.write(clean_path, data, target_sr)
    return data, clean_path


def apply_micro_fade(
    audio: np.ndarray,
    sample_rate: int = 24000,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """
    Apply a transparent 5ms fade-in and fade-out at the absolute edges.
    5ms (120 samples @ 24kHz) eliminates DAC step clicks/pops while preserving
    100% of unvoiced consonant attacks ('s', 'gi', 'kh', 'th', 'tr', 'ch').
    """
    if len(audio) == 0:
        return audio

    fade_samples = max(1, int((fade_ms / 1000.0) * sample_rate))
    if len(audio) < fade_samples * 2:
        return audio

    processed = audio.copy().astype(np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    processed[:fade_samples] *= fade_in
    processed[-fade_samples:] *= fade_out
    return processed


def concatenate_natural_chunks(
    chunk_audios: List[np.ndarray],
    chunk_texts: List[str],
    sample_rate: int = 24000,
    sentence_pause_ms: float = 160.0,
    clause_pause_ms: float = 80.0,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """
    Concatenates speech chunks with prosody-aware natural pauses.
    Inserts:
    - 160ms pause after sentence-ending punctuation (. ! ?)
    - 80ms pause after commas / semicolons / clauses (, ;)
    """
    if not chunk_audios:
        return np.zeros(0, dtype=np.float32)
    if len(chunk_audios) == 1:
        return apply_micro_fade(chunk_audios[0], sample_rate, fade_ms)

    result = []
    for i, (audio, text) in enumerate(zip(chunk_audios, chunk_texts)):
        if len(audio) == 0:
            continue
        cleaned_seg = apply_micro_fade(audio, sample_rate, fade_ms)
        result.append(cleaned_seg)

        # Determine pause duration based on ending punctuation of this chunk
        if i < len(chunk_audios) - 1:
            txt_strip = text.strip()
            if txt_strip and txt_strip[-1] in (".", "!", "?"):
                pause_len = int((sentence_pause_ms / 1000.0) * sample_rate)
            else:
                pause_len = int((clause_pause_ms / 1000.0) * sample_rate)
            
            silence = np.zeros(pause_len, dtype=np.float32)
            result.append(silence)

    return np.concatenate(result, axis=0) if result else np.zeros(0, dtype=np.float32)
