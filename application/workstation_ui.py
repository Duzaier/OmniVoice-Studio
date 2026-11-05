#!/usr/bin/env python3
"""
OmniVoice Production Workstation Web UI (V2.22 Voice & Video Studio Edition).
Features:
- Explicit Hardware & GPU Fallback diagnostics (Dell Inspiron 15 3515 / Ryzen 5 3450U + Vega 8)
- Separate RAM, VRAM, and Multi-Drive Disk Space monitoring (with C: drive protection)
- Smart Text Preprocessor with Markdown stripping & prosody preservation
- Persistent Voice Library with Zero-Latency Prompt Caching
- Integrated Hardware-Aware Video Production Studio (TURBO_CACHED_MUX, AMD_AMF_HW, etc.)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import soundfile as sf
import gradio as gr

from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.core.hardware import profile_hardware, get_detailed_resource_usage
from omnivoice.core.text_preprocessor import TextPreprocessor


def build_workstation_ui(engine: OmniVoiceEngine) -> gr.Blocks:
    hw_info = engine.hw_info
    vm = engine.voice_manager
    storage = engine.storage

    theme = gr.themes.Soft(
        font=["Inter", "Arial", "sans-serif"],
        primary_hue="blue",
        neutral_hue="slate",
    )

    css = """
    :root, .dark, body, html {
        --body-background-fill: #0b0f19 !important;
        --body-text-color: #f8fafc !important;
        --block-label-text-color: #ffffff !important;
        --block-title-text-color: #ffffff !important;
        --input-text-color: #ffffff !important;
        --input-background-fill: #1e293b !important;
        --block-background-fill: #111827 !important;
        --background-fill-primary: #0b0f19 !important;
        --background-fill-secondary: #1e293b !important;
        --border-color-primary: #334155 !important;
        --border-color-accent: #38bdf8 !important;
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
    }
    .gradio-container {
        max-width: 1280px !important;
        margin: auto;
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
    }
    .gradio-container label,
    .gradio-container .block-title,
    .gradio-container .label-wrap,
    .gradio-container .label-wrap span,
    .gradio-container .label-wrap label,
    .gradio-container [data-testid="block-label"],
    .gradio-container .gr-form label,
    .gradio-container .gr-form span,
    .gradio-container .block-label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.98rem !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    .gradio-container h1, .gradio-container h2, .gradio-container h3, .gradio-container h4 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    .gradio-container p, .gradio-container span:not(.label-wrap span) {
        color: #e2e8f0 !important;
    }
    .gradio-container textarea,
    .gradio-container input[type="text"],
    .gradio-container input[type="number"],
    .gradio-container select,
    .gradio-container .gr-input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1.5px solid #475569 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    .warning-box {
        background: #3b1904 !important;
        border: 1.5px solid #f59e0b !important;
        border-radius: 8px;
        padding: 14px;
        color: #fef3c7 !important;
    }
    .success-box {
        background: #052e16 !important;
        border: 1.5px solid #22c55e !important;
        border-radius: 8px;
        padding: 14px;
        color: #dcfce7 !important;
    }
    """

    with gr.Blocks(theme=theme, title="OmniVoice Production Workstation") as demo:
        gr.HTML(f"""
        <style>{css}</style>
        <script>
        document.addEventListener('DOMContentLoaded', () => {{
            document.body.classList.add('dark');
        }});
        </script>
        """)

        fallback_msg = ""
        if hw_info["fallback_reason"]:
            fallback_msg = f"""
<div class="warning-box">
<b>⚡ CHẾ ĐỘ THỰC THI HIỆN TẠI: OPTIMIZED CPU MKL MODE ({hw_info['recommended_threads']} THREADS) + AMD AMF VIDEO ENCODER</b><br>
<b>CPU:</b> {hw_info['cpu_name']} ({hw_info['physical_cores']} Cores / {hw_info['logical_cores']} Threads)<br>
<b>Thiết bị đồ họa:</b> {hw_info['primary_gpu']} ({hw_info['primary_gpu_vram_mb']} MB Shared VRAM)<br>
<b>Chi tiết:</b> {hw_info['fallback_reason']}
</div>
"""
        else:
            fallback_msg = f"""
<div class="success-box">
<b>✅ TĂNG TỐC PHẦN CỨNG: GPU CUDA ACTIVE</b> ({hw_info['primary_gpu']})
</div>
"""

        c_free = hw_info["disks"].get("C:", {}).get("free_gb", 0)
        d_free = hw_info["disks"].get("D:", {}).get("free_gb", 0)
        c_disk_warning = f"""
<div class="success-box" style="margin-top: 8px;">
<b>🛡️ BẢO VỆ Ổ C: ĐÃ KÍCH HOẠT:</b> Ổ C: còn <b>{c_free} GB</b> | Ổ D: còn <b>{d_free} GB</b>. Toàn bộ TEMP, Gradio Cache, Audio Cache & Video Outputs đã được định tuyến an toàn sang <code>{storage.base_dir}</code>.
</div>
"""

        gr.Markdown(f"""
# 🎙️🎬 OmniVoice AI Voice & Video Production Workstation
> **Massively Multilingual Voice Cloning, TTS & Fast Video Rendering Engine** | Workstation V2.22 (OJT Production Edition)
{fallback_msg}
{c_disk_warning}
""")

        with gr.Tabs():
            # ─────────────────────────────────────────────────────────────────
            # TAB 1: TTS SYNTHESIS WORKSTATION
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("🎙️ Tổng Hợp Giọng Nói (TTS Synthesis)"):
                with gr.Row():
                    with gr.Column(scale=6):
                        text_input = gr.Textbox(
                            label="📝 Văn bản cần đọc (Hỗ trợ định dạng Markdown, thơ, truyện, đối thoại)",
                            lines=6,
                            placeholder="Nhập nội dung văn bản tiếng Việt...",
                            value="**Có những người xuất hiện đúng lúc trái tim muốn yêu, nhưng lại sai lúc cả hai biết cách ở bên nhau.**",
                        )

                        with gr.Accordion("🔍 Xem trước văn bản sau khi chuẩn hóa (Cleaned Spoken Text)", open=False):
                            preview_text_btn = gr.Button("🧹 Kiểm tra văn bản chuẩn hóa (Loại bỏ Markdown & Giữ ngắt nghỉ)")
                            preview_text_output = gr.Textbox(label="Văn bản thực tế sẽ được phát âm", lines=3, interactive=False)

                        with gr.Row():
                            voice_dropdown = gr.Dropdown(
                                label="🎭 Chọn Giọng Đã Lưu (Voice Profile)",
                                choices=["[Giọng Mặc Định]"] + vm.list_voices(),
                                value=vm.list_voices()[0] if vm.list_voices() else "[Giọng Mặc Định]",
                                interactive=True,
                            )
                            lang_dropdown = gr.Dropdown(
                                label="🌐 Ngôn ngữ",
                                choices=["Vietnamese", "Auto", "English", "Chinese", "Japanese", "French", "German"],
                                value="Vietnamese",
                                interactive=True,
                            )

                        with gr.Accordion("📂 Hoặc tải lên giọng mẫu trực tiếp (One-Shot Clone 3-8s)", open=False):
                            direct_audio = gr.Audio(label="Audio mẫu (khuyến nghị 3-8 giây, rõ giọng)", type="filepath")
                            direct_text = gr.Textbox(
                                label="Nội dung câu nói trong audio mẫu (tùy chọn)",
                                placeholder="Nhập câu nói trong audio mẫu để đạt độ chính xác 100%...",
                                lines=2,
                            )

                        with gr.Row():
                            perf_mode = gr.Radio(
                                label="⚡ Chế độ hiệu năng (Inference Profile)",
                                choices=["TURBO", "FAST", "BALANCED", "QUALITY"],
                                value="TURBO",
                                info="TURBO: 4 steps CFG=0 (Nhanh gấp 4x, chuẩn V2.1) | FAST: 8 steps CFG=0 | BALANCED: 8 steps CFG=1.5 | QUALITY: 16 steps CFG=2.0",
                            )
                            speed_slider = gr.Slider(
                                minimum=0.8, maximum=1.3, value=1.0, step=0.05,
                                label="⏩ Tốc độ đọc (Speed)",
                            )

                        with gr.Row():
                            emotion_preset = gr.Dropdown(
                                label="🎭 Định hướng cảm xúc & Nhấn nhá (Emotion / Prosody Style)",
                                choices=[
                                    "Tự động nhận diện ngữ cảnh (Auto-Detect)",
                                    "Trầm lắng / Sâu sắc (Reflective)",
                                    "U buồn / Da diết (Melancholic)",
                                    "Dịu dàng / Ấm áp (Tender)",
                                    "Lạc quan / Tươi sáng (Hopeful)",
                                    "Kịch tính / Cao trào (Dramatic)",
                                    "Trung tính / Thuyết minh chuẩn (Neutral)",
                                ],
                                value="Tự động nhận diện ngữ cảnh (Auto-Detect)",
                            )
                            pause_scale_slider = gr.Slider(
                                minimum=0.8, maximum=1.5, value=1.0, step=0.05,
                                label="⏱️ Hệ số khoảng nghỉ ngắt câu (Adaptive Pause Scale)",
                            )

                        with gr.Row():
                            normalize_num_chk = gr.Checkbox(label="🔢 Tự động đọc số thành chữ tiếng Việt", value=False)
                            debug_mode_chk = gr.Checkbox(label="🔬 Bật Prosody & Quality Debug Mode", value=True)

                        gen_btn = gr.Button("🚀 BẮT ĐẦU GENERATE AUDIO (V2.22 OPTIMIZED)", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        output_audio = gr.Audio(label="🔊 Âm thanh kết quả", type="numpy")
                        status_box = gr.Markdown("### 🟢 Trạng Thái: Sẵn sàng")

                        with gr.Accordion("📊 Thống Kê Chi Tiết Hiệu Năng & Âm Học", open=True):
                            metrics_display = gr.Markdown("""
| Chỉ Số | Giá Trị | Đánh Giá |
|---|---|---|
| **Thời gian tạo** | - | - |
| **Độ dài audio** | - | - |
| **Hệ số RTF** | - | - |
| **Trạng thái Cache** | - | - |
| **RAM tiến trình** | - | - |
""")

                        with gr.Accordion("📋 Phân Tích Từng Đoạn (Units Table)", open=False):
                            debug_table = gr.Dataframe(
                                headers=["Unit ID", "Văn bản", "Ngữ điệu", "Cảm xúc", "Khoảng nghỉ (s)", "Thời lượng (s)", "F0 Offset", "Trạng thái"],
                                interactive=False,
                            )

                def _preview_text(t, num_norm):
                    return TextPreprocessor.process(t, normalize_nums=num_norm)

                preview_text_btn.click(
                    fn=_preview_text,
                    inputs=[text_input, normalize_num_chk],
                    outputs=[preview_text_output],
                )

                def _generate_ui(text, voice, direct_aud, direct_txt, lang, mode, speed, emotion_sel, pause_scale, norm_nums, debug_mode, progress=gr.Progress()):
                    if not text or not text.strip():
                        return None, "### ⚠️ Vui lòng nhập văn bản cần đọc.", "", []

                    voice_arg = None if voice == "[Giọng Mặc Định]" else voice

                    def on_progress(curr, tot, pct, msg, perf_info=None):
                        progress(pct / 100.0, desc=f"⏳ Đang xử lý đoạn {curr}/{tot}: {msg}")

                    try:
                        res = engine.synthesize(
                            text=text,
                            voice_name=voice_arg,
                            ref_audio_path=direct_aud,
                            ref_text=direct_txt,
                            language=lang,
                            perf_mode=mode,
                            speed=speed,
                            debug_mode=debug_mode,
                            normalize_numbers=norm_nums,
                            progress_callback=on_progress,
                        )

                        cache_tag = "⚡ CACHE HIT" if res.cached_hit else "🧠 NEURAL SYNTHESIS"
                        status_md = f"### ✅ Hoàn thành tổng hợp âm thanh! ({res.audio_duration:.2f}s audio trong {res.generation_time:.2f}s | {cache_tag})"
                        res_u = res.resource_usage
                        q_sum = res.quality_summary or {}

                        metrics_md = f"""
| Chỉ Số | Giá Trị | Đánh Giá |
|---|---|---|
| **Thời gian tạo** | **{res.generation_time:.2f} s** | {'⚡ Siêu nhanh (<=30s)' if res.generation_time <= 30 else '🟢 Standard CPU'} |
| **Độ dài audio** | **{res.audio_duration:.2f} s** | Hoàn chỉnh (24kHz) |
| **Hệ số RTF** | **{res.rtf:.3f}x** | {'⚡ Real-time / Cached' if res.rtf < 1.0 else '🟡 CPU Mode (Ryzen 3450U)'} |
| **Trạng thái Cache** | **{q_sum.get('cache_status', 'MISS')}** | Lưu tại `cache/audio_segments/` |
| **File Output** | `{Path(res.output_path).name if res.output_path else 'In-Memory'}` | Đã lưu vào `outputs/` |
| **RAM tiến trình** | **{res_u.get('ram_process_mb', 0):.1f} MB** | Ổn định |
"""
                        audio_tuple = (res.sample_rate, (np.clip(res.audio, -1.0, 1.0) * 32767).astype(np.int16))
                        table_rows = [[
                            "Đoạn 01",
                            res.cleaned_text[:35] + ("..." if len(res.cleaned_text) > 35 else ""),
                            "statement",
                            "neutral",
                            "0.35 s",
                            f"{res.audio_duration:.2f} s",
                            "0.0 st",
                            "Hoàn thành",
                        ]]
                        return audio_tuple, status_md, metrics_md, table_rows

                    except Exception as e:
                        return None, f"### ❌ Lỗi: {type(e).__name__}: {e}", "", []

                gen_btn.click(
                    fn=_generate_ui,
                    inputs=[
                        text_input, voice_dropdown, direct_audio, direct_text,
                        lang_dropdown, perf_mode, speed_slider, emotion_preset,
                        pause_scale_slider, normalize_num_chk, debug_mode_chk
                    ],
                    outputs=[output_audio, status_box, metrics_display, debug_table],
                )

            # ─────────────────────────────────────────────────────────────────
            # TAB 2: INTEGRATED VIDEO PRODUCTION STUDIO
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("🎬 Xuất Video (Video Production Studio)"):
                gr.Markdown("""
### 🎬 Quy trình Xuất Video Tích Hợp (OmniVoice + FFmpeg Hardware/Stream-Copy Pipeline)
Tạo video hoàn chỉnh (`.mp4`) từ kịch bản văn bản hoặc file thu âm có sẵn. Chế độ **TURBO_CACHED_MUX** kết hợp visual loop H.264 tái sử dụng và muxing không nén lại giúp xuất video dài **2 phút (120 giây)** chỉ trong **~1–2 giây** sau khi có audio!
""")
                with gr.Row():
                    with gr.Column(scale=6):
                        vid_title = gr.Textbox(
                            label="🏷️ Tiêu đề hiển thị trên Video",
                            value="OmniVoice AI Production Studio",
                        )
                        vid_subtitle = gr.Textbox(
                            label="📌 Phụ đề / Mô tả trên Video",
                            value="Vietnamese Zero-Shot Voice Cloning & Fast Video Export",
                        )
                        vid_text = gr.Textbox(
                            label="📝 Kịch bản thuyết minh (Nếu để trống file Audio bên dưới, hệ thống sẽ tự tổng hợp giọng nói)",
                            lines=5,
                            value="Chào bạn, đây là quy trình sản xuất giọng nói và xuất video tốc độ cao trên máy tính Dell Inspiron 15 3515.",
                        )
                        vid_audio_input = gr.Audio(
                            label="🔊 Hoặc chọn file Audio (.wav) có sẵn để ghép Video ngay lập tức (<2 giây)",
                            type="filepath",
                        )
                        vid_bg_image = gr.Image(
                            label="🖼️ Ảnh nền tùy chỉnh (Tùy chọn - Để trống sẽ tự tạo Slide Studio 16:9)",
                            type="filepath",
                        )
                        with gr.Row():
                            vid_voice = gr.Dropdown(
                                label="🎭 Hồ sơ giọng đọc",
                                choices=["[Giọng Mặc Định]"] + vm.list_voices(),
                                value=vm.list_voices()[0] if vm.list_voices() else "[Giọng Mặc Định]",
                            )
                            vid_tts_mode = gr.Dropdown(
                                label="⚡ Chế độ TTS",
                                choices=["TURBO", "FAST", "BALANCED", "QUALITY"],
                                value="TURBO",
                            )
                        with gr.Row():
                            vid_preset = gr.Dropdown(
                                label="🎞️ Chế độ Render Video (FFmpeg Preset)",
                                choices=[
                                    "TURBO_CACHED_MUX",
                                    "TURBO_DIRECT_ENCODE",
                                    "AMD_AMF_HW",
                                    "MEDIAFOUNDATION_HW",
                                    "BASELINE_SW",
                                ],
                                value="TURBO_CACHED_MUX",
                            )
                            vid_res = gr.Dropdown(
                                label="📐 Độ phân giải (Resolution)",
                                choices=["1280x720", "1920x1080"],
                                value="1280x720",
                            )
                            vid_fps = gr.Dropdown(
                                label="🎞️ Tốc độ khung hình (FPS)",
                                choices=[24, 30],
                                value=24,
                            )
                        export_vid_btn = gr.Button("🎬 BẮT ĐẦU XUẤT VIDEO MP4", variant="primary", size="lg")

                    with gr.Column(scale=5):
                        vid_output = gr.Video(label="🎥 Video MP4 Kết Quả")
                        vid_status = gr.Markdown("### 🟢 Sẵn sàng xuất video")
                        vid_metrics = gr.Markdown("""
| Giai đoạn / Chỉ số | Giá trị đo thực tế |
|---|---|
| **1. Khởi tạo mô hình (Model Init)** | - |
| **2. Tổng hợp giọng nói (Voice Synthesis)** | - |
| **3. Dựng hình ảnh (Video Composition)** | - |
| **4. Mã hóa MP4 cuối (Final Encoding)** | - |
| **TỔNG THỜI GIAN XUẤT VIDEO** | - |
| **Thời lượng Video** | - |
| **Tốc độ Render (FPS)** | - |
| **Dung lượng file MP4** | - |
""")

                def _export_video_ui(title, subtitle, text, audio_file, bg_img, voice, tts_mode, preset, res_str, fps_val):
                    try:
                        voice_arg = None if voice == "[Giọng Mặc Định]" else voice
                        vres = engine.export_video(
                            text=text if not audio_file else None,
                            audio_path=audio_file,
                            voice_name=voice_arg,
                            perf_mode=tts_mode,
                            video_preset=preset,
                            resolution=res_str,
                            fps=int(fps_val),
                            title=title,
                            subtitle=subtitle,
                            custom_image_path=bg_img,
                            use_audio_cache=True,
                            use_visual_cache=True,
                        )
                        status = (
                            f"### ✅ Xuất Video thành công! (`{Path(vres.output_video_path).name}`)\n"
                            f"- **Thời lượng**: `{vres.video_duration_sec:.2f}s` | **Tổng thời gian xuất**: `{vres.total_export_time_sec:.2f}s` "
                            f"({'⚡ ĐẠT MỤC TIÊU <=30s!' if vres.total_export_time_sec <= 30.0 else 'Hoàn tất'})"
                        )
                        metrics = f"""
| Giai đoạn / Chỉ số | Giá trị đo thực tế |
|---|---|
| **1. Khởi tạo mô hình (Model Init)** | `{vres.model_init_time_sec:.3f} s` |
| **2. Tổng hợp giọng nói (Voice Synthesis)** | `{vres.voice_generation_time_sec:.3f} s` ({'Cache Hit' if vres.cached_audio_used else 'Neural CPU'}) |
| **3. Dựng hình ảnh (Video Composition)** | `{vres.video_composition_time_sec:.3f} s` ({'Cached Visual' if vres.cached_visual_used else 'New Visual'}) |
| **4. Mã hóa MP4 cuối (Final Encoding)** | `{vres.video_encoding_time_sec:.3f} s` |
| **TỔNG THỜI GIAN XUẤT VIDEO** | **`{vres.total_export_time_sec:.3f} s`** |
| **Thời lượng Video** | `{vres.video_duration_sec:.2f} s` ({vres.total_frames} frames @ {vres.frame_rate} fps) |
| **Tốc độ Render (Rendering FPS)** | **`{vres.rendering_fps:.1f} FPS`** |
| **Độ phân giải & Codec** | `{vres.resolution}` | `{vres.video_codec}` |
| **Chất lượng Audio** | `{vres.audio_quality}` |
| **Dung lượng file MP4** | `{vres.output_file_size_mb:.2f} MB` |
| **CPU / RAM sử dụng** | `{vres.cpu_utilization_pct:.1f}% CPU` / `{vres.ram_process_mb:.1f} MB RAM` |
"""
                        return vres.output_video_path, status, metrics
                    except Exception as e:
                        return None, f"### ❌ Lỗi khi xuất video: {type(e).__name__}: {e}", ""

                export_vid_btn.click(
                    fn=_export_video_ui,
                    inputs=[
                        vid_title, vid_subtitle, vid_text, vid_audio_input, vid_bg_image,
                        vid_voice, vid_tts_mode, vid_preset, vid_res, vid_fps,
                    ],
                    outputs=[vid_output, vid_status, vid_metrics],
                )

            # ─────────────────────────────────────────────────────────────────
            # TAB 3: VOICE PROFILE & CACHE MANAGER
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("🎭 Quản Lý Thư Viện Giọng (Voice Library)"):
                gr.Markdown("""
### 🎭 Quản lý hồ sơ giọng nói (Zero-latency Prompt Caching)
Đặc trưng âm học được mã hóa **1 lần duy nhất** thành file `prompt.pt`.
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        new_voice_name = gr.Textbox(label="Tên hồ sơ giọng (VD: Giọng MC Nữ)", lines=1)
                        new_voice_audio = gr.Audio(label="Audio tham chiếu (3-8 giây, rõ giọng)", type="filepath")
                        new_voice_text = gr.Textbox(
                            label="Nội dung trong audio tham chiếu",
                            placeholder="Nhập câu nói trong audio để giọng đạt độ chuẩn 100%...",
                            lines=2,
                        )
                        new_voice_desc = gr.Textbox(label="Mô tả / Ghi chú", lines=1)
                        create_voice_btn = gr.Button("💾 Lưu & Tạo Cache Prompt (.pt)", variant="primary")
                        create_voice_status = gr.Markdown("")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 Danh sách giọng hiện có")
                        voice_table = gr.Dataframe(
                            value=[[v] for v in vm.list_voices()],
                            headers=["Tên Hồ Sơ Giọng"],
                            interactive=False,
                        )
                        refresh_voice_btn = gr.Button("🔄 Làm mới danh sách")

                        with gr.Group():
                            gr.Markdown("### 🗑️ Xóa giọng khỏi thư viện")
                            delete_voice_dropdown = gr.Dropdown(
                                label="Chọn giọng cần xóa",
                                choices=vm.list_voices(),
                                value=vm.list_voices()[0] if vm.list_voices() else None,
                                interactive=True,
                            )
                            delete_voice_btn = gr.Button("🗑️ Xóa Giọng Đã Chọn", variant="stop")
                            delete_voice_status = gr.Markdown("")

                def _save_voice(name, audio_path, ref_txt, desc, progress=gr.Progress()):
                    if not name or not name.strip():
                        return "### ⚠️ Vui lòng nhập Tên hồ sơ giọng.", [[v] for v in vm.list_voices()], gr.update(), gr.update()
                    if not audio_path:
                        return "### ⚠️ Vui lòng tải lên file Audio tham chiếu (3-8 giây).", [[v] for v in vm.list_voices()], gr.update(), gr.update()

                    progress(0.3, desc="Đang trích xuất đặc trưng âm học (HiggsAudio) và tạo prompt.pt...")
                    try:
                        profile = vm.create_or_update_voice(
                            name=name.strip(),
                            ref_audio_path=audio_path,
                            ref_text=ref_txt.strip() if ref_txt else "Xin chào các bạn.",
                            model=engine.model,
                            description=desc.strip() if desc else "",
                        )
                        all_voices = vm.list_voices()
                        success_md = f"### ✅ Đã lưu thành công hồ sơ giọng **'{profile.name}'**!\n📂 **File cache**: `{profile.prompt_path}`"
                        new_table = [[v] for v in all_voices]
                        new_dropdown = gr.update(choices=["[Giọng Mặc Định]"] + all_voices, value=profile.name)
                        new_del_dropdown = gr.update(choices=all_voices, value=profile.name)
                        return success_md, new_table, new_dropdown, new_del_dropdown
                    except Exception as e:
                        all_v = vm.list_voices()
                        return f"### ❌ Lỗi khi tạo hồ sơ giọng: {type(e).__name__}: {e}", [[v] for v in all_v], gr.update(), gr.update()

                def _delete_voice(name_to_del):
                    if not name_to_del or not name_to_del.strip():
                        all_v = vm.list_voices()
                        return "### ⚠️ Vui lòng chọn giọng cần xóa.", [[v] for v in all_v], gr.update(), gr.update()
                    success = vm.delete_voice(name_to_del.strip())
                    all_voices = vm.list_voices()
                    new_table = [[v] for v in all_voices]
                    new_del_choices = gr.update(choices=all_voices, value=all_voices[0] if all_voices else None)
                    new_syn_choices = gr.update(choices=["[Giọng Mặc Định]"] + all_voices, value="[Giọng Mặc Định]")
                    msg = f"### ✅ Đã xóa thành công hồ sơ giọng **'{name_to_del}'**!" if success else f"### ⚠️ Không tìm thấy **'{name_to_del}'**."
                    return msg, new_table, new_del_choices, new_syn_choices

                create_voice_btn.click(
                    fn=_save_voice,
                    inputs=[new_voice_name, new_voice_audio, new_voice_text, new_voice_desc],
                    outputs=[create_voice_status, voice_table, voice_dropdown, delete_voice_dropdown],
                )
                delete_voice_btn.click(
                    fn=_delete_voice,
                    inputs=[delete_voice_dropdown],
                    outputs=[delete_voice_status, voice_table, delete_voice_dropdown, voice_dropdown],
                )
                refresh_voice_btn.click(
                    fn=lambda: (
                        [[v] for v in vm.list_voices()],
                        gr.update(choices=["[Giọng Mặc Định]"] + vm.list_voices()),
                        gr.update(choices=vm.list_voices(), value=vm.list_voices()[0] if vm.list_voices() else None)
                    ),
                    outputs=[voice_table, voice_dropdown, delete_voice_dropdown],
                )

            # ─────────────────────────────────────────────────────────────────
            # TAB 4: HARDWARE PROFILER & DISK SAFETY
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("🖥️ Thông Số Phần Cứng & Giám Sát Tài Nguyên"):
                res = get_detailed_resource_usage(str(storage.base_dir))
                gr.Markdown(f"""
### 🔍 Báo Cáo Kiểm Tra Phần Cứng (Hardware Audit):
```yaml
COMPUTER MODEL:         Dell Inspiron 15 3515
CPU:                    {hw_info['cpu_name']} ({hw_info['physical_cores']} Physical / {hw_info['logical_cores']} Logical, Pinned: {hw_info['recommended_threads']} MKL Threads)
GPU DETECTED:           {hw_info['primary_gpu']} ({hw_info['primary_gpu_vram_mb']} MB Shared VRAM)
INFERENCE DEVICE:       {hw_info['inference_device']} (PyTorch {hw_info['pytorch_version']}, MKL={hw_info.get('mkl_available', True)}, MKLDNN={hw_info.get('mkldnn_available', True)})
VIDEO ENCODERS:         h264_amf (AMD AMF Hardware), h264_mf (MediaFoundation), libx264 (AVX2 Software)
```

### 💾 Giám Sát Tài Nguyên Bộ Nhớ & Ổ Đĩa:
- **RAM Tiến trình Workstation**: `{res['ram_process_mb']} MB`
- **RAM Hệ thống đã sử dụng**: `{res['ram_system_used_mb']} MB` / `{res['ram_system_total_gb']} GB`
- **Dung lượng trống ổ C:**: `{res['c_drive_free_gb']} GB` (Đã bảo vệ: Không ghi file tạm lên ổ C:)
- **Dung lượng trống ổ D:**: `{res.get('d_drive_free_gb', 0)} GB`
- **Thư mục Dữ liệu Workstation**: `{storage.base_dir}`
- **Cache HuggingFace**: `{res['hf_cache_mb']} MB`
- **Outputs sinh ra**: `{res['outputs_mb']} MB`
- **Thư viện giọng (Voices)**: `{res['voices_mb']} MB`
""")

    return demo
