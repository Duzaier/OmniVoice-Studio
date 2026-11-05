#!/usr/bin/env python3
"""
Voice Profile & Persistent Prompt Cache Manager for OmniVoice.
Stores reference audio, transcripts, pre-encoded prompts (.pt), and metadata.
Guarantees zero redundant re-encoding of reference voices.
"""

import os
import json
import shutil
import hashlib
import logging
import unicodedata
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

import torch
from omnivoice.models.omnivoice import VoiceClonePrompt

logger = logging.getLogger(__name__)


def slugify_name(name: str) -> str:
    """Convert any Unicode/Vietnamese string into a safe ASCII filesystem folder name."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in ascii_name).strip("_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe or ("voice_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:8])


def compute_audio_hash(audio_path: str, ref_text: str = "") -> str:
    """Compute SHA256 fingerprint for reference audio file + text."""
    h = hashlib.sha256()
    with open(audio_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    h.update(ref_text.strip().encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class VoiceProfile:
    name: str
    folder_name: str
    ref_audio_path: str
    ref_text: str
    prompt_path: str
    fingerprint: str
    created_at: str
    description: str = ""
    language: str = "Vietnamese"


class VoiceManager:
    """Manages persistent voice profiles and prompt caching."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "voices")
        self.voices_dir = Path(base_dir).resolve()
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_cache: Dict[str, VoiceClonePrompt] = {}
        logger.info(f"[VoiceManager] Storage initialized at: {self.voices_dir}")

    def list_voices(self) -> List[str]:
        """Return sorted list of available voice profile display names."""
        profiles = []
        for p in self.voices_dir.iterdir():
            if p.is_dir():
                meta_file = p / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            profiles.append(data.get("name", p.name))
                    except Exception:
                        profiles.append(p.name)
        return sorted(list(set(profiles)))

    def _find_voice_dir(self, name_or_folder: str) -> Optional[Path]:
        """Find directory by folder name or display name."""
        direct = self.voices_dir / slugify_name(name_or_folder)
        if direct.exists() and (direct / "metadata.json").exists():
            return direct

        # Search all subdirectories
        for p in self.voices_dir.iterdir():
            if p.is_dir() and (p / "metadata.json").exists():
                try:
                    with open(p / "metadata.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("name") == name_or_folder or p.name == name_or_folder:
                            return p
                except Exception:
                    pass
        return None

    def get_profile(self, voice_name: str) -> Optional[VoiceProfile]:
        """Load voice profile metadata."""
        vdir = self._find_voice_dir(voice_name)
        if not vdir:
            return None
        meta_file = vdir / "metadata.json"
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "folder_name" not in data:
                data["folder_name"] = vdir.name
            return VoiceProfile(**data)
        except Exception as e:
            logger.error(f"Failed to read metadata for voice {voice_name}: {e}")
            return None

    def load_prompt(self, voice_name: str, map_location: str = "cpu") -> Optional[VoiceClonePrompt]:
        """Load pre-encoded VoiceClonePrompt from memory cache or disk."""
        if voice_name in self._prompt_cache:
            return self._prompt_cache[voice_name]

        vdir = self._find_voice_dir(voice_name)
        if not vdir:
            return None

        prompt_file = vdir / "prompt.pt"
        if not prompt_file.exists():
            return None

        try:
            prompt = VoiceClonePrompt.load(str(prompt_file), map_location=map_location)
            self._prompt_cache[voice_name] = prompt
            logger.info(f"[VoiceManager] Loaded cached prompt for '{voice_name}' from {prompt_file}")
            return prompt
        except Exception as e:
            logger.error(f"Failed to load prompt.pt for voice {voice_name}: {e}")
            return None

    def create_or_update_voice(
        self,
        name: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
        model = None,
        description: str = "",
        language: str = "Vietnamese",
    ) -> VoiceProfile:
        """
        Encode reference audio, save prompt.pt, copy reference audio and save metadata.
        """
        import datetime
        import soundfile as sf

        display_name = name.strip()
        if not display_name:
            raise ValueError("Tên hồ sơ giọng không được để trống.")

        folder_slug = slugify_name(display_name)
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            raise ValueError("Không tìm thấy file audio tham chiếu.")

        vdir = self.voices_dir / folder_slug
        vdir.mkdir(parents=True, exist_ok=True)

        target_wav = vdir / "reference.wav"
        target_txt = vdir / "reference.txt"
        target_prompt = vdir / "prompt.pt"
        target_meta = vdir / "metadata.json"

        # Clean reference audio (mono conversion, leading breath trimming, optimal duration)
        from omnivoice.core.audio_processor import prepare_reference_audio
        from omnivoice.core.text_preprocessor import TextPreprocessor

        ref_data, clean_ref_path = prepare_reference_audio(
            ref_audio_path,
            target_sr=getattr(model, "sampling_rate", 24000),
            max_duration=8.0,
        )
        shutil.copy2(clean_ref_path, str(target_wav))

        # Transcript handling with markdown cleaning
        raw_text = ref_text.strip() if ref_text and ref_text.strip() else "Xin chào các bạn."
        final_text = TextPreprocessor.process(raw_text)
        if not final_text:
            final_text = "Xin chào các bạn."

        with open(target_txt, "w", encoding="utf-8") as f:
            f.write(final_text)

        # Generate and save VoiceClonePrompt
        prompt = model.create_voice_clone_prompt(
            ref_audio=str(target_wav),
            ref_text=final_text,
            preprocess_prompt=False,  # Already cleanly preprocessed
        )
        prompt.save(str(target_prompt))
        self._prompt_cache[display_name] = prompt
        self._prompt_cache[folder_slug] = prompt

        fingerprint = compute_audio_hash(str(target_wav), final_text)
        profile = VoiceProfile(
            name=display_name,
            folder_name=folder_slug,
            ref_audio_path=str(target_wav),
            ref_text=final_text,
            prompt_path=str(target_prompt),
            fingerprint=fingerprint,
            created_at=datetime.datetime.now().isoformat(),
            description=description,
            language=language,
        )

        with open(target_meta, "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)

        logger.info(f"[VoiceManager] Created voice profile '{display_name}' ({folder_slug}) with cached prompt.")
        return profile

    def delete_voice(self, name: str) -> bool:
        """Delete a voice profile and its cached files."""
        vdir = self._find_voice_dir(name)
        if vdir and vdir.exists() and vdir.is_dir():
            shutil.rmtree(vdir, ignore_errors=True)
            self._prompt_cache.pop(name, None)
            logger.info(f"[VoiceManager] Deleted voice profile '{name}'")
            return True
        return False
