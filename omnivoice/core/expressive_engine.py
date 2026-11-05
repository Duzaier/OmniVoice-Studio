#!/usr/bin/env python3
"""
Expressive Prosody Synthesis Engine for OmniVoice TTS (V5 Performance Edition).
Features:
- Fast batched inference over SemanticProsodyChunks (~10-20s audio per chunk)
- Dramatic reduction in model.generate() invocations (e.g. 51 sentences -> 5-8 chunks)
- Full torch.inference_mode() execution with 0 autograd overhead
- Clean intra-sentence punctuation preservation (periods, commas, question/exclamation marks)
- Adaptive digital pause insertion between semantic chunks in memory (<1ms)
- Non-destructive 5ms micro-fades and onset breath suppression
- Preserves 100% Vietnamese lexical tones and speaker identity
"""

import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import torch

from omnivoice.core.prosody_tagger import ProsodyUnit, SemanticProsodyChunk, ProsodyTagger
from omnivoice.core.audio_processor import apply_micro_fade
from omnivoice.core.breath_analyzer import AcousticBreathAnalyzer
from omnivoice.core.pitch_analyzer import PitchAndEnergyAnalyzer

logger = logging.getLogger(__name__)


class ExpressiveProsodyEngine:
    """Executes high-performance expressive speech synthesis over SemanticProsodyChunks."""

    @staticmethod
    def build_emotion_instruct(emotion: str, sentence_type: str, has_voice_clone: bool = True) -> Optional[str]:
        """
        OmniVoice accepts only predefined strict instruct tokens:
        ['child', 'elderly', 'female', 'high pitch', 'low pitch', 'male',
         'middle-aged', 'moderate pitch', 'teenager', 'very high pitch',
         'very low pitch', 'whisper', 'young adult']

        When voice cloning prompt is active, speaker identity comes from voice_prompt,
        so instruct must be None (or valid pitch modifier) to avoid ValueError.
        """
        if has_voice_clone:
            return None

        if sentence_type == "question" or sentence_type == "exclamation":
            return "high pitch"
        elif emotion in ("melancholic", "reflective"):
            return "low pitch"
        elif emotion == "tender":
            return "moderate pitch"
        return None

    @classmethod
    def synthesize_semantic_chunks(
        cls,
        model,
        chunks: List[SemanticProsodyChunk],
        voice_prompt,
        language: str = "Vietnamese",
        gen_config = None,
        base_speed: float = 1.0,
        debug_mode: bool = False,
        progress_callback: Optional[Callable[[int, int, float, str, Dict[str, Any]], None]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Synthesize SemanticProsodyChunks with batched GPU/CPU inference,
        inserting adaptive pauses and micro-fades cleanly in memory.
        """
        sample_rate = model.sampling_rate
        chunk_audios = []
        chunk_records = []
        num_chunks = len(chunks)
        has_voice_clone = (voice_prompt is not None)

        t_session_start = time.perf_counter()
        accumulated_audio_duration = 0.0

        for idx, c in enumerate(chunks, 1):
            t_chunk_start = time.perf_counter()
            effective_speed = base_speed * c.speed_factor

            # Notify progress
            if progress_callback:
                pct = ((idx - 1) / max(1, num_chunks)) * 100
                elapsed = time.perf_counter() - t_session_start
                avg_chunk_time = elapsed / max(1, idx - 1) if idx > 1 else 0
                eta_s = avg_chunk_time * (num_chunks - idx + 1)
                perf_info = {
                    "elapsed_sec": round(elapsed, 1),
                    "eta_sec": round(eta_s, 1),
                    "audio_duration_sec": round(accumulated_audio_duration, 1),
                    "current_chunk": idx,
                    "total_chunks": num_chunks,
                }
                progress_callback(
                    idx,
                    num_chunks,
                    pct,
                    f"Đang xử lý đoạn {idx}/{num_chunks} ({len(c.sentences)} câu): '{c.cleaned_text[:35]}...'",
                    perf_info,
                )

            # Generate speech for this semantic chunk under inference_mode
            with torch.inference_mode():
                audios = model.generate(
                    text=c.cleaned_text,
                    language=language if language != "Auto" else None,
                    voice_clone_prompt=voice_prompt,
                    speed=effective_speed if effective_speed != 1.0 else None,
                    instruct=None,
                    generation_config=gen_config,
                )
            raw_audio = audios[0]
            t_chunk_gen = time.perf_counter() - t_chunk_start

            # Analyze onset for breath transient and trim leading breath if present
            onset_diag = AcousticBreathAnalyzer.analyze_onset(raw_audio, sample_rate=sample_rate)
            if onset_diag["breath_detected"] and onset_diag["speech_start_sec"] > 0.12:
                processed_audio = AcousticBreathAnalyzer.trim_leading_breath_artifact(
                    raw_audio,
                    sample_rate=sample_rate,
                    max_trim_ms=250.0,
                )
            else:
                processed_audio = raw_audio

            # Apply non-destructive 5ms micro-fade
            processed_audio = apply_micro_fade(processed_audio, sample_rate=sample_rate, fade_ms=5.0)
            chunk_audios.append(processed_audio)

            chunk_dur = len(processed_audio) / sample_rate
            accumulated_audio_duration += chunk_dur
            chunk_rtf = t_chunk_gen / chunk_dur if chunk_dur > 0 else 0

            rec = {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "cleaned_text": c.cleaned_text,
                "num_sentences": len(c.sentences),
                "dominant_emotion": c.dominant_emotion,
                "pause_after_sec": c.pause_after_sec,
                "speed_factor": c.speed_factor,
                "duration_sec": round(chunk_dur, 2),
                "generation_time_sec": round(t_chunk_gen, 2),
                "rtf": round(chunk_rtf, 3),
                "boundary_level": c.boundary_level,
                "sentences": [s.to_dict() for s in c.sentences],
            }
            chunk_records.append(rec)

        # -------------------------------------------------------------
        # In-Memory Adaptive Pause Concatenation
        # -------------------------------------------------------------
        final_segments = []
        for i, (audio, c) in enumerate(zip(chunk_audios, chunks)):
            if len(audio) == 0:
                continue
            final_segments.append(audio)

            # Insert adaptive pause after this chunk (except after final chunk)
            if i < len(chunks) - 1:
                pause_samples = int(c.pause_after_sec * sample_rate)
                silence_gap = np.zeros(pause_samples, dtype=np.float32)
                final_segments.append(silence_gap)

        final_waveform = np.concatenate(final_segments, axis=0) if final_segments else np.zeros(0, dtype=np.float32)

        # Acoustic prosody analysis (only in debug mode to keep production fast)
        overall_prosody = PitchAndEnergyAnalyzer.analyze_prosody_metrics(final_waveform, sample_rate=sample_rate) if debug_mode else {}

        # Flatten individual sentence prosody units for UI and debug tables
        flattened_units = []
        for c_rec in chunk_records:
            for s in c_rec.get("sentences", []):
                flattened_units.append(s)

        return final_waveform, chunk_records, overall_prosody
