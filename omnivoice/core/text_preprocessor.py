#!/usr/bin/env python3
"""
Smart Text Preprocessor & Vietnamese Normalizer for OmniVoice TTS.
Includes TextCleaner, TextNormalizer, and ProsodyPreserver.
Ensures Markdown formatting and non-speech symbols are NEVER spoken,
while preserving Vietnamese diacritics, structural pauses, and natural intonation.
"""

import re
import unicodedata
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class TextCleaner:
    """Removes Markdown syntax, URL links, and non-spoken formatting symbols."""

    @staticmethod
    def remove_markdown(text: str) -> str:
        if not text:
            return ""

        # Remove code blocks ```code```
        text = re.sub(r"```[\s\S]*?```", " ", text)
        # Remove inline code `code`
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Remove Markdown image syntax ![alt](url)
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", " ", text)
        # Remove Markdown links [text](url) -> text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove headers # Header -> Header
        text = re.sub(r"(?:^|\s)#{1,6}\s*", " ", text)
        # Remove bold/italic **text**, *text*, __text__, _text_
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)
        # Remove strikethrough ~~text~~
        text = re.sub(r"~~([^~]+)~~", r"\1", text)
        # Remove blockquotes > Quote
        text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
        # Remove bullet lists - item, * item, + item
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
        # Remove numbered lists 1. item -> item
        text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)

        return text


class ProsodyPreserver:
    """Converts formatting and punctuation into natural prosody pauses."""

    @staticmethod
    def preserve_prosody(text: str) -> str:
        if not text:
            return ""

        # Convert ellipsis '...' and '…' to a gentle pause marker (comma or period based on surrounding)
        text = re.sub(r"\.{3,}|…", ", ", text)
        
        # Convert em-dash / en-dash / double dash to comma pause
        text = re.sub(r"[\u2013\u2014\u2015]|\s--\s", ", ", text)

        # Remove non-speech quotes: " ' “ ” « » ` '
        text = re.sub(r'["\'“”«»„‟`]', "", text)

        # Remove parentheses, brackets, braces but keep text inside with pause: (abc) -> , abc ,
        text = re.sub(r"[\(\[\{]([^\)\]\}]+)[\)\]\}]", r", \1, ", text)

        # Remove standalone special symbols: @, #, $, %, ^, &, *, ~, |, \, <, >, /
        text = re.sub(r"[@#\$%\^&\*~\|\\<>/]", " ", text)

        # Normalize multiple punctuation marks (e.g., '??' -> '?', '!!' -> '!', ',,' -> ',')
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"!+", "!", text)
        text = re.sub(r",+", ",", text)
        text = re.sub(r"\.+", ".", text)
        text = re.sub(r":+", ":", text)
        text = re.sub(r";+", ";", text)

        return text


class TextNormalizer:
    """Normalizes Vietnamese text, whitespace, and Unicode NFC encodings."""

    # Vietnamese number words dictionary for basic cardinal numbers
    VIET_DIGITS = {
        "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
        "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín", "10": "mười"
    }

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Ensure standard Unicode NFC composition for Vietnamese diacritics."""
        if not text:
            return ""
        return unicodedata.normalize("NFC", text)

    @classmethod
    def normalize_numbers(cls, text: str) -> str:
        """Convert standalone single/double digits to Vietnamese words for smoother speech."""
        def replace_digit(m):
            d = m.group(0)
            if d in cls.VIET_DIGITS:
                return cls.VIET_DIGITS[d]
            return d
        
        # Replace standalone 0-10
        return re.sub(r"\b(10|[0-9])\b", replace_digit, text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Clean extra spaces, newlines, and ensure proper spacing after punctuation."""
        if not text:
            return ""
        # Replace newlines with spaces or periods
        text = re.sub(r"\r?\n+", " ", text)
        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?;:])(?=[^\s\d])", r"\1 ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class TextPreprocessor:
    """Complete Pipeline: Raw Text -> Spoken Text Representation."""

    @classmethod
    def process(cls, raw_text: str, normalize_nums: bool = False) -> str:
        """
        Executes full preprocessing pipeline:
        RAW TEXT -> Remove Markdown -> Preserve Prosody -> Normalize Unicode & Numbers -> Normalize Whitespace
        """
        if not raw_text:
            return ""

        # Step 1: Unicode NFC Normalization first
        t = TextNormalizer.normalize_unicode(raw_text)

        # Step 2: Clean Markdown formatting
        t = TextCleaner.remove_markdown(t)

        # Step 3: Prosody & punctuation structure
        t = ProsodyPreserver.preserve_prosody(t)

        # Step 4: Optional number normalization
        if normalize_nums:
            t = TextNormalizer.normalize_numbers(t)

        # Step 5: Whitespace normalization
        t = TextNormalizer.normalize_whitespace(t)

        return t
