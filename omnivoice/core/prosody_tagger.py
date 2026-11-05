#!/usr/bin/env python3
"""
Prosody Tagging & Semantic Sentence Boundary Engine for Vietnamese TTS.
Converts raw text into an internal Prosody-Aware Representation with:
- Boundary hierarchy (Level 1: Micro-pause, Level 2: Clause, Level 3: Sentence, Level 4: Paragraph)
- Adaptive Pause Calculation (based on punctuation, length, and emotion)
- Semantic Emotion Intent Detection (reflective, melancholic, tender, dramatic, hopeful, neutral)
- Sentence Intonation Type (statement, question, exclamation, suspensive)
- Keyword / Emphasis Identification
"""

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Emotion keywords dictionary for Vietnamese storytelling
EMOTION_KEYWORDS = {
    "melancholic": ["buồn", "nghịch lý", "đau", "mất", "xa", "khóc", "cô đơn", "chia tay", "tiếc nuối", "rời đi", "mệt mỏi"],
    "reflective": ["nghĩ", "hiểu", "nhận ra", "bài học", "cuộc sống", "thời gian", "trải nghiệm", "hoàn thiện", "trân trọng"],
    "tender": ["yêu", "thương", "nhớ", "trái tim", "ở bên", "chăm sóc", "dịu dàng", "ôm", "bình yên", "ấm áp"],
    "hopeful": ["hy vọng", "tương lai", "tin tưởng", "vươn lên", "bước tiếp", "ngày mai", "niềm tin", "ánh sáng"],
    "dramatic": ["bất ngờ", "không thể", "quyết định", "đột ngột", "sự thật", "nguy hiểm", "bùng nổ"],
}


@dataclass
class ProsodyUnit:
    """Internal representation of a speech prosody segment."""
    unit_id: int
    text: str
    cleaned_text: str
    boundary_level: int  # 1: Comma, 2: Semicolon/Colon, 3: Period/End, 4: Paragraph
    sentence_type: str   # "statement", "question", "exclamation", "suspensive"
    emotion: str         # "reflective", "melancholic", "tender", "hopeful", "neutral", etc.
    pause_after_sec: float
    speed_factor: float = 1.0
    emphasis_words: List[str] = field(default_factory=list)
    pitch_offset_semitones: float = 0.0
    energy_scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticProsodyChunk:
    """Group of consecutive ProsodyUnits formed for fast batched model inference (~15-20s)."""
    chunk_id: int
    text: str
    cleaned_text: str
    sentences: List[ProsodyUnit]
    pause_after_sec: float
    speed_factor: float = 1.0
    dominant_emotion: str = "neutral"
    boundary_level: int = 3  # 3: Sentence, 4: Paragraph

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "cleaned_text": self.cleaned_text,
            "num_sentences": len(self.sentences),
            "pause_after_sec": self.pause_after_sec,
            "speed_factor": self.speed_factor,
            "dominant_emotion": self.dominant_emotion,
            "boundary_level": self.boundary_level,
            "sentences": [s.to_dict() for s in self.sentences],
        }



class ProsodyTagger:
    """Parses text into hierarchical prosody units with adaptive pauses and emotion tags."""

    @staticmethod
    def detect_emotion(text: str) -> str:
        """Detect subtle emotion from text keywords."""
        text_lower = text.lower()
        scores = {emo: 0 for emo in EMOTION_KEYWORDS}
        for emo, words in EMOTION_KEYWORDS.items():
            for w in words:
                if w in text_lower:
                    scores[emo] += 1

        best_emo = max(scores, key=scores.get)
        return best_emo if scores[best_emo] > 0 else "neutral"

    @staticmethod
    def detect_sentence_type(text: str) -> str:
        """Identify intonation type from punctuation."""
        text_strip = text.strip()
        if not text_strip:
            return "statement"
        if text_strip.endswith("?"):
            return "question"
        if text_strip.endswith("!"):
            return "exclamation"
        if text_strip.endswith("...") or text_strip.endswith("…"):
            return "suspensive"
        return "statement"

    @staticmethod
    def extract_emphasis_words(text: str) -> List[str]:
        """Find key contrastive or emotional words to subtly emphasize."""
        words = text.split()
        emphasis = []
        for w in words:
            clean_w = w.strip(".,!?;:\"'()").lower()
            if any(clean_w in kw_list for kw_list in EMOTION_KEYWORDS.values()):
                emphasis.append(clean_w)
        return list(set(emphasis))[:3]

    @classmethod
    def calculate_adaptive_pause(
        cls,
        boundary_level: int,
        sentence_type: str,
        emotion: str,
        text_length: int,
    ) -> float:
        """
        Calculate adaptive pause duration in seconds.
        Accounts for boundary level, emotion, length, and sentence type.
        """
        # Base pauses per boundary level
        if boundary_level == 1:  # Micro pause (Comma)
            base_pause = 0.14
        elif boundary_level == 2:  # Clause pause (Semicolon, Colon, Dash)
            base_pause = 0.24
        elif boundary_level == 3:  # Sentence boundary (. ! ?)
            if sentence_type == "question":
                base_pause = 0.42
            elif sentence_type == "exclamation":
                base_pause = 0.38
            elif sentence_type == "suspensive":
                base_pause = 0.52
            else:
                base_pause = 0.36
        elif boundary_level >= 4:  # Paragraph / Major Section
            base_pause = 0.65
        else:
            base_pause = 0.05

        # Emotion adjustments
        if emotion in ("reflective", "melancholic"):
            base_pause *= 1.15
        elif emotion in ("dramatic", "hopeful"):
            base_pause *= 0.95

        # Length scaling (longer sentence earns slightly more breathing room after)
        len_factor = min(1.2, max(0.9, 0.9 + (text_length / 200.0)))
        final_pause = round(base_pause * len_factor, 3)
        return final_pause

    @classmethod
    def parse_text_to_prosody_units(
        cls,
        raw_text: str,
        default_speed: float = 1.0,
    ) -> List[ProsodyUnit]:
        """
        Segment text by paragraphs and sentences, producing structured ProsodyUnits.
        """
        from omnivoice.core.text_preprocessor import TextPreprocessor

        if not raw_text or not raw_text.strip():
            return []

        paragraphs = re.split(r"\n\s*\n+", raw_text.strip())
        units = []
        unit_idx = 1

        for p_idx, para in enumerate(paragraphs):
            is_last_para = (p_idx == len(paragraphs) - 1)
            para_clean = para.strip()
            if not para_clean:
                continue

            # Split paragraph into sentences (. ! ? … \n)
            # Lookbehind to keep sentence ending punctuation
            sentence_splits = re.split(r"(?<=[.!?…\n])\s+", para_clean)
            sentences = [s.strip() for s in sentence_splits if s.strip()]

            for s_idx, sent in enumerate(sentences):
                is_last_sent = (s_idx == len(sentences) - 1)
                cleaned = TextPreprocessor.process(sent)
                if not cleaned:
                    continue

                sent_type = cls.detect_sentence_type(sent)
                emotion = cls.detect_emotion(cleaned)
                emphasis = cls.extract_emphasis_words(cleaned)

                # Determine boundary level
                if is_last_sent and not is_last_para:
                    boundary = 4  # Paragraph boundary
                else:
                    boundary = 3  # Sentence boundary

                pause_after = cls.calculate_adaptive_pause(
                    boundary_level=boundary,
                    sentence_type=sent_type,
                    emotion=emotion,
                    text_length=len(cleaned),
                )

                # Speaking rate nuance per emotion
                speed = default_speed
                if emotion in ("reflective", "melancholic"):
                    speed *= 0.96
                elif emotion in ("hopeful", "dramatic"):
                    speed *= 1.02
                elif sent_type == "question":
                    speed *= 0.98

                # Pitch & energy adjustment
                pitch_offset = 0.0
                energy_scale = 1.0
                if sent_type == "question":
                    pitch_offset = 0.8  # Slight subtle rise on question
                elif sent_type == "exclamation":
                    energy_scale = 1.08  # Slight energy boost on exclamation
                    pitch_offset = 0.5
                elif emotion == "melancholic":
                    energy_scale = 0.94
                    pitch_offset = -0.4

                unit = ProsodyUnit(
                    unit_id=unit_idx,
                    text=sent,
                    cleaned_text=cleaned,
                    boundary_level=boundary,
                    sentence_type=sent_type,
                    emotion=emotion,
                    pause_after_sec=pause_after,
                    speed_factor=round(speed, 3),
                    emphasis_words=emphasis,
                    pitch_offset_semitones=round(pitch_offset, 2),
                    energy_scale=round(energy_scale, 2),
                )
                units.append(unit)
                unit_idx += 1

        logger.info(f"[ProsodyTagger] Parsed {len(units)} prosody units with adaptive pauses & emotional dynamics.")
        return units

    @classmethod
    def parse_text_to_semantic_chunks(
        cls,
        raw_text: str,
        target_chars: int = 220,
        max_chars: int = 320,
        default_speed: float = 1.0,
    ) -> List[SemanticProsodyChunk]:
        """
        Group individual ProsodyUnits into cohesive macro-chunks (~10-20s speech)
        while preserving internal sentence boundaries, punctuation, and emotions.
        """
        units = cls.parse_text_to_prosody_units(raw_text, default_speed=default_speed)
        if not units:
            return []

        chunks = []
        cur_units: List[ProsodyUnit] = []
        cur_chars = 0
        chunk_idx = 1

        for i, u in enumerate(units):
            cur_units.append(u)
            cur_chars += len(u.cleaned_text) + 1

            is_paragraph_end = (u.boundary_level >= 4)
            is_last = (i == len(units) - 1)
            is_long_enough = (cur_chars >= target_chars)

            if is_paragraph_end or is_long_enough or is_last:
                joined_clean = " ".join(item.cleaned_text for item in cur_units)
                joined_raw = " ".join(item.text for item in cur_units)
                trailing_pause = cur_units[-1].pause_after_sec
                avg_speed = sum(item.speed_factor for item in cur_units) / len(cur_units)
                emotions = [item.emotion for item in cur_units if item.emotion != "neutral"]
                dominant_emo = emotions[0] if emotions else "neutral"

                chunk_obj = SemanticProsodyChunk(
                    chunk_id=chunk_idx,
                    text=joined_raw,
                    cleaned_text=joined_clean,
                    sentences=list(cur_units),
                    pause_after_sec=trailing_pause,
                    speed_factor=round(avg_speed, 3),
                    dominant_emotion=dominant_emo,
                    boundary_level=cur_units[-1].boundary_level,
                )
                chunks.append(chunk_obj)
                chunk_idx += 1
                cur_units = []
                cur_chars = 0

        logger.info(f"[ProsodyTagger] Grouped {len(units)} sentences into {len(chunks)} semantic chunks for fast batched inference.")
        return chunks

