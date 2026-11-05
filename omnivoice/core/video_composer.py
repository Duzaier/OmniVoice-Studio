#!/usr/bin/env python3
"""
OmniVoice Hardware-Aware Video Composer & Rendering Engine.
Designed for Dell Inspiron 15 3515 (AMD Ryzen 5 3450U + AMD Radeon Vega 8 Mobile).

Supports & Benchmarks 5 Video Rendering Presets:
1. TURBO_CACHED_MUX: Pre-renders a reusable H.264 visual master segment into cache/video_assets/
   and stream-copies (-c:v copy) over any audio duration (30s, 60s, 120s+) in ~1-2 seconds.
2. TURBO_DIRECT_ENCODE: Single-pass libx264 -preset ultrafast -tune stillimage from static/waveform slide.
3. AMD_AMF_HW: Hardware-accelerated H.264 encoding via AMD AMF (h264_amf) on Radeon Vega 8 Mobile.
4. MEDIAFOUNDATION_HW: Windows MediaFoundation H.264 encoder (h264_mf).
5. BASELINE_SW: Unoptimized full-frame software encoding (libx264 -preset medium, 1080p@30fps).
"""

import os
import time
import json
import shutil
import hashlib
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf
import psutil

logger = logging.getLogger(__name__)


@dataclass
class VideoExportResult:
    output_video_path: str
    audio_path: str
    video_duration_sec: float
    total_export_time_sec: float
    model_init_time_sec: float
    voice_generation_time_sec: float
    video_composition_time_sec: float
    video_encoding_time_sec: float
    rendering_fps: float
    total_frames: int
    output_file_size_mb: float
    resolution: str
    frame_rate: int
    video_codec: str
    audio_codec: str
    audio_quality: str
    preset_used: str
    cpu_utilization_pct: float
    ram_process_mb: float
    ram_system_used_mb: float
    gpu_utilization_note: str
    cached_visual_used: bool
    cached_audio_used: bool


class VideoComposer:
    """Hardware-aware video composition and FFmpeg encoding pipeline for OmniVoice."""

    PRESETS = {
        "TURBO_CACHED_MUX": {
            "label": "Turbo Cached Visual Mux (Stream-Copy H.264 + AAC)",
            "video_codec": "libx264 (Cached Stream Copy -c:v copy)",
            "resolution": "1280x720",
            "fps": 24,
            "audio_bitrate": "192k",
            "description": "Fastest (~1-2s encode for 120s video): reuses pre-encoded H.264 visual loop + AAC muxing.",
        },
        "TURBO_DIRECT_ENCODE": {
            "label": "Turbo Direct Still-Image Encode (libx264 ultrafast + stillimage)",
            "video_codec": "libx264 (-preset ultrafast -tune stillimage)",
            "resolution": "1280x720",
            "fps": 24,
            "audio_bitrate": "192k",
            "description": "Fast single-pass encode (~3-5s for 120s video) using AVX2 stillimage optimization.",
        },
        "AMD_AMF_HW": {
            "label": "AMD Radeon Vega 8 AMF Hardware H.264 (h264_amf)",
            "video_codec": "h264_amf (AMD Hardware Encoder)",
            "resolution": "1280x720",
            "fps": 24,
            "audio_bitrate": "192k",
            "description": "Offloads H.264 video encoding to integrated AMD Radeon Vega 8 VCN hardware engine.",
        },
        "MEDIAFOUNDATION_HW": {
            "label": "Windows MediaFoundation H.264 (h264_mf)",
            "video_codec": "h264_mf (MediaFoundation MFT)",
            "resolution": "1280x720",
            "fps": 24,
            "audio_bitrate": "192k",
            "description": "Uses Windows 11 native MediaFoundation H.264 MFT encoder.",
        },
        "BASELINE_SW": {
            "label": "Original Unoptimized Baseline (libx264 medium, 1080p@30fps)",
            "video_codec": "libx264 (-preset medium)",
            "resolution": "1920x1080",
            "fps": 30,
            "audio_bitrate": "192k",
            "description": "Reference unoptimized software encoding at 1080p 30fps without loop/stillimage caching.",
        },
    }

    def __init__(self, storage_manager):
        self.storage = storage_manager
        self.ffmpeg_bin = self._locate_ffmpeg()
        self.ffprobe_bin = self._locate_ffprobe()
        self.assets_dir = self.storage.cache_video_dir
        self.assets_dir.mkdir(parents=True, exist_ok=True)

    def _locate_ffmpeg(self) -> str:
        path = shutil.which("ffmpeg")
        if path:
            return path
        candidates = [
            r"C:\Users\Duzaier\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe",
            r"D:\vMix\streaming\ffmpeg.exe",
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "ffmpeg"

    def _locate_ffprobe(self) -> str:
        path = shutil.which("ffprobe")
        if path:
            return path
        candidate = r"C:\Users\Duzaier\AppData\Local\Microsoft\WinGet\Links\ffprobe.exe"
        if os.path.exists(candidate):
            return candidate
        return "ffprobe"

    def generate_visual_slide(
        self,
        title: str = "OmniVoice AI Production Studio",
        subtitle: str = "Vietnamese Zero-Shot Voice Cloning & Video Workflow",
        resolution: str = "1280x720",
        custom_image_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        use_cache: bool = True,
    ) -> Tuple[str, bool]:
        """
        Generate or retrieve a 16:9 visual card PNG for video composition.
        Returns (image_path, was_cached).
        """
        if custom_image_path and os.path.exists(custom_image_path):
            return custom_image_path, True

        w, h = [int(x) for x in resolution.lower().split("x")]
        key_str = f"slide_v2_{w}x{h}_{title}_{subtitle}"
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:12]
        slide_path = self.assets_dir / f"slide_{w}x{h}_{key_hash}.png"

        if use_cache and slide_path.exists() and slide_path.stat().st_size > 1000:
            return str(slide_path), True

        try:
            from PIL import Image, ImageDraw, ImageFont

            # Create modern dark studio background
            img = Image.new("RGB", (w, h), color=(11, 15, 25))
            draw = ImageDraw.Draw(img)

            # Subtle gradient bars & decorative border
            draw.rectangle([24, 24, w - 24, h - 24], outline=(56, 189, 248), width=3)
            draw.rectangle([40, 40, w - 40, 120], fill=(30, 41, 59))

            # Draw waveform decorative bars in center
            bar_count = 64
            margin_x = int(w * 0.1)
            usable_w = w - 2 * margin_x
            bar_w = max(4, usable_w // (bar_count * 2))
            center_y = int(h * 0.58)
            max_bar_h = int(h * 0.22)

            if audio_array is not None and len(audio_array) > bar_count:
                chunks = np.array_split(np.abs(audio_array), bar_count)
                amps = [float(np.mean(c)) for c in chunks]
                max_a = max(amps) if max(amps) > 1e-6 else 1.0
                norm_amps = [min(1.0, max(0.08, a / max_a)) for a in amps]
            else:
                rng = np.random.default_rng(42)
                norm_amps = [float(0.15 + 0.85 * np.sin(i / bar_count * np.pi) * rng.uniform(0.4, 1.0)) for i in range(bar_count)]

            for i, amp in enumerate(norm_amps):
                bx = margin_x + i * (usable_w // bar_count)
                bh = max(6, int(amp * max_bar_h))
                draw.rectangle(
                    [bx, center_y - bh, bx + bar_w, center_y + bh],
                    fill=(56, 189, 248) if i % 2 == 0 else (99, 102, 241),
                )

            # Text labels
            try:
                font_lg = ImageFont.truetype("arial.ttf", size=max(24, w // 36))
                font_md = ImageFont.truetype("arial.ttf", size=max(18, w // 54))
                font_sm = ImageFont.truetype("arial.ttf", size=max(14, w // 72))
            except Exception:
                font_lg = ImageFont.load_default()
                font_md = font_lg
                font_sm = font_lg

            draw.text((64, 62), title[:70], fill=(255, 255, 255), font=font_lg)
            draw.text((64, 150), subtitle[:95], fill=(148, 163, 184), font=font_md)
            draw.text(
                (64, h - 75),
                f"OmniVoice V2.22 Workstation | {resolution} | Dell Inspiron 15 3515 (AMD Ryzen 5 3450U + Vega 8)",
                fill=(100, 116, 139),
                font=font_sm,
            )

            img.save(str(slide_path), format="PNG", optimize=False)
            return str(slide_path), False

        except Exception as e:
            logger.warning(f"[VideoComposer] PIL slide generation fallback to FFmpeg lavfi: {e}")
            cmd = [
                self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c=#0b0f19:s={resolution}:d=1",
                "-frames:v", "1", str(slide_path)
            ]
            subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
            return str(slide_path), False

    def get_or_build_cached_visual_loop(
        self,
        slide_path: str,
        resolution: str = "1280x720",
        fps: int = 24,
        loop_seconds: int = 5,
        force_rebuild: bool = False,
    ) -> Tuple[str, bool]:
        """
        Pre-encode a 5-second GOP-aligned H.264 visual loop MP4 in cache/video_assets/.
        When muxed with -stream_loop -1 -c:v copy, a 120s video requires ZERO video re-encoding!
        """
        slide_stat = os.stat(slide_path)
        key_str = f"{slide_path}_{slide_stat.st_size}_{resolution}_{fps}_{loop_seconds}_v2"
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:12]
        loop_mp4 = self.assets_dir / f"visual_loop_{resolution}_{fps}fps_{key_hash}.mp4"

        if not force_rebuild and loop_mp4.exists() and loop_mp4.stat().st_size > 1000:
            return str(loop_mp4), True

        gop = fps  # 1-second keyframe interval for clean stream looping and seeking
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-loop", "1",
            "-framerate", str(fps),
            "-t", str(loop_seconds),
            "-i", slide_path,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "stillimage",
            "-pix_fmt", "yuv420p",
            "-g", str(gop),
            "-keyint_min", str(gop),
            "-sc_threshold", "0",
            "-an",
            str(loop_mp4),
        ]
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        return str(loop_mp4), False

    def compose_and_export(
        self,
        audio_path: Optional[str] = None,
        audio_array: Optional[np.ndarray] = None,
        sample_rate: int = 24000,
        output_filename: Optional[str] = None,
        preset: str = "TURBO_CACHED_MUX",
        resolution: Optional[str] = None,
        fps: Optional[int] = None,
        title: str = "OmniVoice AI Production Studio",
        subtitle: str = "High-Fidelity Vietnamese Voice Cloning & Video Workflow",
        custom_image_path: Optional[str] = None,
        model_init_time_sec: float = 0.0,
        voice_generation_time_sec: float = 0.0,
        cached_audio_used: bool = False,
        use_visual_cache: bool = True,
    ) -> VideoExportResult:
        """
        Compose and encode a complete MP4 video from audio + visual assets.
        Measures composition time and encoding time separately and accurately.
        """
        t_export_start = time.perf_counter()
        proc = psutil.Process()
        psutil.cpu_percent(interval=None)  # Prime CPU counter

        preset_cfg = self.PRESETS.get(preset, self.PRESETS["TURBO_CACHED_MUX"])
        res_str = resolution or preset_cfg["resolution"]
        target_fps = int(fps or preset_cfg["fps"])
        audio_bitrate = preset_cfg["audio_bitrate"]

        # 1. Ensure audio is on disk (on D: cache/outputs) and measure exact audio duration
        t_comp_start = time.perf_counter()
        if audio_path and os.path.exists(audio_path):
            info = sf.info(audio_path)
            audio_duration = float(info.duration)
            wav_path = audio_path
        elif audio_array is not None:
            audio_duration = float(len(audio_array)) / float(sample_rate)
            wav_path = str(self.storage.cache_temp_dir / f"temp_audio_{int(time.time()*1000)}.wav")
            sf.write(wav_path, audio_array, sample_rate)
        else:
            raise ValueError("Either audio_path or audio_array must be provided for video export.")

        # 2. Visual Composition: Prepare 16:9 visual slide + optional cached visual loop
        slide_path, slide_was_cached = self.generate_visual_slide(
            title=title,
            subtitle=subtitle,
            resolution=res_str,
            custom_image_path=custom_image_path,
            audio_array=audio_array,
            use_cache=use_visual_cache,
        )

        cached_visual_used = slide_was_cached
        loop_mp4_path = None
        if preset == "TURBO_CACHED_MUX":
            loop_mp4_path, loop_was_cached = self.get_or_build_cached_visual_loop(
                slide_path=slide_path,
                resolution=res_str,
                fps=target_fps,
                loop_seconds=5,
                force_rebuild=not use_visual_cache,
            )
            cached_visual_used = cached_visual_used and loop_was_cached

        video_composition_time = time.perf_counter() - t_comp_start

        # 3. Final Video Encoding / Muxing via FFmpeg
        if not output_filename:
            output_filename = f"omnivoice_video_{preset.lower()}_{int(audio_duration)}s_{int(time.time())}.mp4"
        output_video_path = str(self.storage.outputs_dir / output_filename)

        t_enc_start = time.perf_counter()
        w, h = [int(x) for x in res_str.lower().split("x")]

        if preset == "TURBO_CACHED_MUX" and loop_mp4_path:
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-stream_loop", "-1",
                "-i", loop_mp4_path,
                "-i", wav_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-t", f"{audio_duration:.3f}",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", audio_bitrate,
                "-ar", "24000",
                "-movflags", "+faststart",
                output_video_path,
            ]
            codec_label = "h264 (Stream Copy from Cached H.264 Loop)"
            gpu_note = "0% GPU (Zero-Re-encode Stream Mux)"
        elif preset == "AMD_AMF_HW":
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-loop", "1",
                "-framerate", str(target_fps),
                "-i", slide_path,
                "-i", wav_path,
                "-t", f"{audio_duration:.3f}",
                "-vf", f"scale={w}:{h},format=nv12",
                "-c:v", "h264_amf",
                "-usage", "ultralowlatency",
                "-quality", "speed",
                "-c:a", "aac",
                "-b:a", audio_bitrate,
                "-ar", "24000",
                "-movflags", "+faststart",
                "-shortest",
                output_video_path,
            ]
            codec_label = "h264_amf (AMD Radeon Vega 8 VCN Hardware)"
            gpu_note = "Active (AMD Radeon Vega 8 AMF VCN Engine)"
        elif preset == "MEDIAFOUNDATION_HW":
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-loop", "1",
                "-framerate", str(target_fps),
                "-i", slide_path,
                "-i", wav_path,
                "-t", f"{audio_duration:.3f}",
                "-vf", f"scale={w}:{h},format=nv12",
                "-c:v", "h264_mf",
                "-c:a", "aac",
                "-b:a", audio_bitrate,
                "-ar", "24000",
                "-movflags", "+faststart",
                "-shortest",
                output_video_path,
            ]
            codec_label = "h264_mf (Windows MediaFoundation)"
            gpu_note = "Active (MediaFoundation MFT)"
        elif preset == "BASELINE_SW":
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-loop", "1",
                "-framerate", str(target_fps),
                "-i", slide_path,
                "-i", wav_path,
                "-t", f"{audio_duration:.3f}",
                "-vf", f"scale={w}:{h},format=yuv420p",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-threads", "6",
                "-c:a", "aac",
                "-b:a", audio_bitrate,
                "-ar", "24000",
                "-movflags", "+faststart",
                "-shortest",
                output_video_path,
            ]
            codec_label = "libx264 (-preset medium, Full Software)"
            gpu_note = "0% (CPU Software Encoding)"
        else:  # TURBO_DIRECT_ENCODE
            cmd = [
                self.ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-loop", "1",
                "-framerate", str(target_fps),
                "-i", slide_path,
                "-i", wav_path,
                "-t", f"{audio_duration:.3f}",
                "-vf", f"scale={w}:{h},format=yuv420p",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "stillimage",
                "-crf", "22",
                "-threads", "6",
                "-c:a", "aac",
                "-b:a", audio_bitrate,
                "-ar", "24000",
                "-movflags", "+faststart",
                "-shortest",
                output_video_path,
            ]
            codec_label = "libx264 (-preset ultrafast -tune stillimage)"
            gpu_note = "0% (CPU AVX2 Still-Image Fast Path)"

        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        video_encoding_time = time.perf_counter() - t_enc_start

        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_proc_mb = round(proc.memory_info().rss / (1024**2), 1)
        ram_sys_mb = round((mem.total - mem.available) / (1024**2), 1)

        total_frames = int(round(audio_duration * target_fps))
        render_stage_sec = max(0.001, video_composition_time + video_encoding_time)
        rendering_fps = round(total_frames / render_stage_sec, 1)

        file_size_mb = round(os.path.getsize(output_video_path) / (1024**2), 2) if os.path.exists(output_video_path) else 0.0
        total_export_time = model_init_time_sec + voice_generation_time_sec + video_composition_time + video_encoding_time

        return VideoExportResult(
            output_video_path=output_video_path,
            audio_path=wav_path,
            video_duration_sec=round(audio_duration, 2),
            total_export_time_sec=round(total_export_time, 3),
            model_init_time_sec=round(model_init_time_sec, 3),
            voice_generation_time_sec=round(voice_generation_time_sec, 3),
            video_composition_time_sec=round(video_composition_time, 3),
            video_encoding_time_sec=round(video_encoding_time, 3),
            rendering_fps=rendering_fps,
            total_frames=total_frames,
            output_file_size_mb=file_size_mb,
            resolution=res_str,
            frame_rate=target_fps,
            video_codec=codec_label,
            audio_codec="AAC-LC",
            audio_quality=f"24kHz Mono / {audio_bitrate} AAC",
            preset_used=preset,
            cpu_utilization_pct=round(cpu_pct, 1),
            ram_process_mb=ram_proc_mb,
            ram_system_used_mb=ram_sys_mb,
            gpu_utilization_note=gpu_note,
            cached_visual_used=cached_visual_used,
            cached_audio_used=cached_audio_used,
        )
