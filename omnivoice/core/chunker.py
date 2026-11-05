#!/usr/bin/env python3
"""
Acoustic-Aware Smart Text Chunker & Audio Segment Concatenator for OmniVoice.
Ensures optimal context window (80-250 chars), avoids cutting phrases,
and concatenates audio cleanly with boundary micro-fades and natural pause gaps.
"""

import re
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class SmartTextChunker:
    """
    Hierarchical text chunker prioritizing Paragraph -> Sentence -> Clause -> Length.
    Prevents tiny muffled fragments (<60 chars) while avoiding sequence length bloat (>280 chars).
    """

    def __init__(self, target_chars: int = 150, min_chars: int = 60, max_chars: int = 250):
        self.target_chars = target_chars
        self.min_chars = min_chars
        self.max_chars = max_chars

    def chunk(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        # If text is already comfortably within limits, return as single chunk
        if len(text) <= self.max_chars:
            return [text]

        # Step 1: Split by sentence boundaries (. ! ?)
        # Lookbehind pattern to keep punctuation with the sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sent in sentences:
            if not current_chunk:
                current_chunk = sent
                continue

            # If adding this sentence stays within max_chars
            if len(current_chunk) + len(sent) + 1 <= self.max_chars:
                current_chunk += " " + sent
                # If reached target_chars, close this chunk
                if len(current_chunk) >= self.target_chars:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            else:
                # Current chunk is full
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # If single sentence itself is larger than max_chars, split by clause (; : ,)
                if len(sent) > self.max_chars:
                    sub_clauses = self._split_by_clause(sent)
                    for sc in sub_clauses:
                        if len(sc) >= self.min_chars or not chunks:
                            chunks.append(sc)
                        elif chunks:
                            # Merge with previous chunk if small
                            chunks[-1] += " " + sc
                else:
                    current_chunk = sent

        if current_chunk:
            # If remaining trailing chunk is very small and previous chunk has space, merge it
            if len(current_chunk) < self.min_chars and chunks:
                if len(chunks[-1]) + len(current_chunk) + 1 <= self.max_chars + 50:
                    chunks[-1] += " " + current_chunk
                else:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(current_chunk.strip())

        logger.info(f"[SmartTextChunker] Split text ({len(text)} chars) into {len(chunks)} cohesive chunk(s).")
        return chunks

    def _split_by_clause(self, long_sent: str) -> List[str]:
        """Split very long sentence by comma, semicolon or colon."""
        clauses = re.split(r"(?<=[;,:])\s+", long_sent)
        merged = []
        cur = ""
        for c in clauses:
            c = c.strip()
            if not c:
                continue
            if not cur:
                cur = c
            elif len(cur) + len(c) + 1 <= self.max_chars:
                cur += " " + c
            else:
                merged.append(cur)
                cur = c
        if cur:
            merged.append(cur)
        return merged if merged else [long_sent]


def concatenate_audio_segments(
    segments: List[np.ndarray],
    sample_rate: int = 24000,
    pause_ms: float = 90.0,
    fade_ms: float = 6.0,
    crossfade_ms: float = 0.0,
) -> np.ndarray:
    """
    Concatenate audio segments cleanly without destroying phonemes or causing edge pops.
    Applies a micro fade-in/out (6ms) at segment edges and inserts a natural silence gap (90ms).
    """
    if not segments:
        return np.zeros(0, dtype=np.float32)
    if len(segments) == 1:
        return segments[0]

    pause_samples = int((pause_ms / 1000.0) * sample_rate)
    silence = np.zeros(pause_samples, dtype=np.float32)

    fade_samples = max(1, int((fade_ms / 1000.0) * sample_rate))
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)

    processed_segments = []
    for seg in segments:
        if len(seg) == 0:
            continue
        seg_copy = seg.copy().astype(np.float32)
        # Apply gentle boundary fade to prevent clicks
        if len(seg_copy) > fade_samples * 2:
            seg_copy[:fade_samples] *= fade_in
            seg_copy[-fade_samples:] *= fade_out
        processed_segments.append(seg_copy)

    # Interleave with natural silence pause
    result = []
    for i, seg in enumerate(processed_segments):
        result.append(seg)
        if i < len(processed_segments) - 1:
            result.append(silence)

    return np.concatenate(result, axis=0)
