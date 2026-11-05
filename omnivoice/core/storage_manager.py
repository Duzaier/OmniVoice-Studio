#!/usr/bin/env python3
"""
Storage Manager & Disk Safety Controller for OmniVoice.
Protects C: drive from overflow by redirecting large caches, models, outputs,
Gradio temp files, and temporary audio/video files to D:\\ALL PROJECT\\OJT PROJECT\\OmniVoice\\cache.
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

MIN_REQUIRED_FREE_DISK_GB = 1.0  # Safe threshold before generation


class DiskSpaceError(Exception):
    """Raised when available disk space is dangerously low."""
    pass


class StorageManager:
    """Manages workspace storage, directories, caches, and disk safety."""

    def __init__(self, preferred_base_dir: Optional[str] = None):
        self.base_dir = self._resolve_best_base_dir(preferred_base_dir)
        self.outputs_dir = self.base_dir / "outputs"
        self.voices_dir = self.base_dir / "voices"
        self.temp_dir = self.base_dir / "temp"
        self.debug_dir = self.base_dir / "debug"
        self.cache_dir = self.base_dir / "cache"
        self.cache_temp_dir = self.cache_dir / "temp"
        self.cache_gradio_dir = self.cache_dir / "gradio"
        self.cache_audio_dir = self.cache_dir / "audio_segments"
        self.cache_video_dir = self.cache_dir / "video_assets"
        self.benchmarks_dir = self.base_dir / "benchmarks"
        self.logs_dir = self.base_dir / "logs"

        # Create necessary directories
        for d in [
            self.outputs_dir,
            self.voices_dir,
            self.temp_dir,
            self.debug_dir,
            self.cache_dir,
            self.cache_temp_dir,
            self.cache_gradio_dir,
            self.cache_audio_dir,
            self.cache_video_dir,
            self.benchmarks_dir,
            self.logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # CRITICAL C: DRIVE PROTECTION:
        # Redirect tempfile and Gradio temp directories to D: drive cache so C: (which has <1 GB free) never overflows.
        self._redirect_system_temp_to_safe_drive()

        logger.info(f"[StorageManager] Active Workstation Data Directory: {self.base_dir}")

    def get_outputs_dir(self) -> Path:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self.outputs_dir

    def get_temp_dir(self) -> Path:
        self.cache_temp_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_temp_dir

    def _redirect_system_temp_to_safe_drive(self):
        """Redirect Python tempfile, OS TEMP/TMP, and GRADIO_TEMP_DIR to self.cache_temp_dir."""
        safe_tmp = str(self.cache_temp_dir.resolve())
        safe_gradio = str(self.cache_gradio_dir.resolve())
        os.environ["TEMP"] = safe_tmp
        os.environ["TMP"] = safe_tmp
        os.environ["TMPDIR"] = safe_tmp
        os.environ["GRADIO_TEMP_DIR"] = safe_gradio
        tempfile.tempdir = safe_tmp

    def _resolve_best_base_dir(self, preferred: Optional[str] = None) -> Path:
        """
        Choose base directory prioritizing the project root if it resides on a spacious drive (e.g. D:).
        """
        if preferred and os.path.exists(os.path.dirname(preferred) or preferred):
            return Path(preferred).resolve()

        default_path = Path(__file__).parent.parent.parent.resolve()

        # If default_path is already on D: (e.g., D:\ALL PROJECT\OJT PROJECT\OmniVoice), use it directly!
        if str(default_path).upper().startswith("D:"):
            return default_path

        # Check drive space if running from C:
        d_exists = os.path.exists("D:\\")
        c_usage = shutil.disk_usage("C:\\") if os.path.exists("C:\\") else None

        if d_exists:
            d_usage = shutil.disk_usage("D:\\")
            d_free_gb = d_usage.free / (1024**3)
            c_free_gb = (c_usage.free / (1024**3)) if c_usage else 0
            if d_free_gb > 20.0 and c_free_gb < 15.0:
                ojt_path = Path(r"D:\ALL PROJECT\OJT PROJECT\OmniVoice")
                if ojt_path.exists():
                    return ojt_path.resolve()
                return Path("D:\\OmniVoice_Storage").resolve()

        return default_path

    def check_disk_safety(self, drive_path: Optional[str] = None) -> Tuple[bool, float, str]:
        """
        Check if there is sufficient disk space before starting a generation.
        Returns (is_safe, free_gb, message).
        """
        target = drive_path or str(self.base_dir)
        try:
            usage = shutil.disk_usage(target)
            free_gb = usage.free / (1024**3)
            if free_gb < MIN_REQUIRED_FREE_DISK_GB:
                msg = (
                    f"Dung lượng đĩa khả dụng quá thấp: {free_gb:.2f} GB "
                    f"(Yêu cầu tối thiểu {MIN_REQUIRED_FREE_DISK_GB} GB để đảm bảo an toàn). "
                    f"Vui lòng giải phóng ổ đĩa hoặc chuyển thư mục lưu trữ sang ổ D:."
                )
                return False, free_gb, msg
            return True, free_gb, f"Đĩa khả dụng: {free_gb:.1f} GB (Đạt yêu cầu an toàn)"
        except Exception as e:
            logger.warning(f"Could not verify disk space on {target}: {e}")
            return True, 999.0, "Không thể kiểm tra dung lượng đĩa."

    def ensure_disk_safe_for_generation(self):
        """Raise DiskSpaceError if disk space is below minimum safe threshold."""
        is_safe, free_gb, msg = self.check_disk_safety()
        if not is_safe:
            raise DiskSpaceError(msg)

    def clean_temp_files(self, max_age_hours: float = 24.0):
        """Clean temporary audio/video files older than max_age_hours."""
        import time
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        cleaned_bytes = 0
        for tdir in [self.temp_dir, self.cache_temp_dir]:
            if tdir.exists():
                for f in tdir.iterdir():
                    try:
                        if f.is_file() and f.stat().st_mtime < cutoff:
                            sz = f.stat().st_size
                            f.unlink()
                            cleaned_bytes += sz
                    except Exception:
                        pass
        if cleaned_bytes > 0:
            logger.info(f"[StorageManager] Cleaned {cleaned_bytes / (1024**2):.2f} MB of old temp files.")
