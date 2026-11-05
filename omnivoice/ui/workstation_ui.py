#!/usr/bin/env python3
"""
OmniVoice Production Studio UI (V2.24 Swiss Minimalist Edition).
Engineered with:
- antigravity-design-engineering (Swiss Minimalist archetype, WCAG 2.2 AA, 4px/8px grid, progressive disclosure)
- uiverse-ui-engineering (Scoped tokens, tactile CTA button, zero global CSS leakage)
- morphicons-icon-motion (Semantic stroke SVG state transitions with reduced-motion support)
- gsap-core (matchMedia-aware micro-transitions & prefers-reduced-motion compliance)

Zero duplication: 1 Unified Workspace, 1 Script Composer, 1 Primary Trigger.
All secondary parameters, voice cloning tools, and telemetry are tucked into closed drawers by default.
"""

import time
import socket
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import gradio as gr

from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.core.hardware import get_detailed_resource_usage
from omnivoice.core.text_preprocessor import TextPreprocessor


def _get_lan_ips() -> List[str]:
    ips = []
    try:
        _, _, addr_list = socket.gethostbyname_ex(socket.gethostname())
        for ip in addr_list:
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                ips.append(ip)
    except Exception:
        pass
    return ips or ["localhost"]


def build_workstation_ui(engine: OmniVoiceEngine) -> gr.Blocks:
    hw_info = engine.hw_info
    vm = engine.voice_manager
    storage = engine.storage
    lan_ips = _get_lan_ips()
    primary_lan_url = f"http://{lan_ips[0]}:8001"

    theme = gr.themes.Base(
        font=["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
        primary_hue="sky",
        neutral_hue="slate",
    )

    css = """
    :root, .dark, body, html {
        --ov-bg: #090d16;
        --ov-surface: #111827;
        --ov-surface-elevated: #1e293b;
        --ov-border: rgba(148, 163, 184, 0.16);
        --ov-border-focus: #38bdf8;
        --ov-text: #f8fafc;
        --ov-muted: #94a3b8;
        --ov-accent: #38bdf8;
        --body-background-fill: var(--ov-bg) !important;
        --body-text-color: var(--ov-text) !important;
        --block-background-fill: var(--ov-surface) !important;
        --block-border-color: var(--ov-border) !important;
        --input-background-fill: var(--ov-surface-elevated) !important;
        --input-border-color: var(--ov-border) !important;
        --input-text-color: var(--ov-text) !important;
        --block-label-text-color: var(--ov-text) !important;
        --block-title-text-color: var(--ov-text) !important;
        background-color: var(--ov-bg) !important;
        color: var(--ov-text) !important;
        -webkit-tap-highlight-color: transparent;
    }

    /* Hide Gradio footer clutter */
    footer { display: none !important; }

    .gradio-container {
        max-width: 980px !important;
        margin: 0 auto !important;
        padding: 12px 16px 32px !important;
        background-color: var(--ov-bg) !important;
        transition: max-width 0.22s cubic-bezier(0.22, 1, 0.36, 1);
    }

    /* 1-Click Mobile Frame Preview on Desktop */
    body.ov-mobile-frame .gradio-container {
        max-width: 430px !important;
        margin: 14px auto !important;
        padding: 12px !important;
        border: 2px solid rgba(56, 189, 248, 0.45) !important;
        border-radius: 26px !important;
        box-shadow: 0 24px 64px rgba(0, 0, 0, 0.75), 0 0 0 5px #111827 !important;
    }

    /* Minimalist Top Bar */
    .ov-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 8px 14px;
        background: rgba(17, 24, 39, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid var(--ov-border);
        border-radius: 14px;
        margin-bottom: 12px;
    }
    .ov-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: clamp(0.98rem, 1.4vw, 1.1rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--ov-text);
    }
    .ov-status-dot {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #86efac;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        padding: 3px 9px;
        border-radius: 999px;
    }
    .ov-actions {
        display: flex;
        align-items: center;
        gap: 6px;
        flex-wrap: wrap;
    }
    .ov-pill-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--ov-surface-elevated);
        color: var(--ov-text) !important;
        border: 1px solid var(--ov-border);
        border-radius: 9px;
        padding: 6px 11px;
        font-size: 0.78rem;
        font-weight: 600;
        text-decoration: none !important;
        cursor: pointer;
        transition: border-color 0.15s ease, background 0.15s ease, transform 0.1s ease;
        min-height: 34px;
    }
    .ov-pill-btn:hover {
        border-color: var(--ov-accent);
        background: rgba(56, 189, 248, 0.12);
    }
    .ov-pill-btn:active {
        transform: scale(0.97);
    }
    .ov-pill-btn.accent {
        background: rgba(2, 132, 199, 0.22);
        border-color: rgba(56, 189, 248, 0.45);
        color: #e0f2fe !important;
    }

    /* Typography & Inputs (16px minimum prevents iOS auto-zoom) */
    .gradio-container label,
    .gradio-container .block-title,
    .gradio-container [data-testid="block-label"] {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        letter-spacing: 0.01em;
    }
    .gradio-container textarea,
    .gradio-container input[type="text"],
    .gradio-container input[type="number"],
    .gradio-container select {
        background-color: var(--ov-surface-elevated) !important;
        color: var(--ov-text) !important;
        border: 1px solid rgba(148, 163, 184, 0.24) !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
    }
    .gradio-container textarea:focus,
    .gradio-container input:focus,
    .gradio-container select:focus {
        border-color: var(--ov-border-focus) !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.18) !important;
    }

    /* Uiverse Tactile Primary Button */
    .gradio-container button.primary {
        min-height: 50px !important;
        font-size: 0.98rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.28) !important;
        transition: transform 0.12s ease, box-shadow 0.15s ease !important;
    }
    .gradio-container button.primary:hover {
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.42) !important;
    }
    .gradio-container button.primary:active {
        transform: scale(0.985) !important;
    }

    /* Quick Sample Chips */
    .ov-chip-row button {
        min-height: 32px !important;
        padding: 4px 10px !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        border-radius: 999px !important;
        background: var(--ov-surface-elevated) !important;
        border: 1px solid var(--ov-border) !important;
        color: #cbd5e1 !important;
    }
    .ov-chip-row button:hover {
        border-color: var(--ov-accent) !important;
        color: #ffffff !important;
    }

    /* Progressive Disclosure Accordions */
    .gradio-container .accordion {
        border: 1px solid var(--ov-border) !important;
        border-radius: 12px !important;
        background: rgba(17, 24, 39, 0.6) !important;
        margin-top: 6px !important;
    }

    /* Compact Status Banner */
    .ov-status-line {
        background: var(--ov-surface);
        border: 1px solid var(--ov-border);
        border-radius: 12px;
        padding: 10px 14px;
        font-size: 0.88rem;
        margin-bottom: 6px;
    }
    .ov-status-line p {
        margin: 0 !important;
        color: var(--ov-text) !important;
    }

    /* Reduced Motion Accessibility */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }

    /* Mobile Responsive Breakpoint */
    @media (max-width: 768px) {
        .gradio-container {
            padding: 8px 8px 24px !important;
        }
        .ov-topbar {
            padding: 8px 10px;
        }
        .ov-hide-mobile {
            display: none !important;
        }
        .gradio-container button.primary {
            min-height: 54px !important;
            font-size: 1.02rem !important;
        }
    }
    """

    d_free = hw_info["disks"].get("D:", {}).get("free_gb", 0)

    with gr.Blocks(theme=theme, title="OmniVoice Studio") as demo:
        gr.HTML(f"""
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="theme-color" content="#090d16">
        <style>{css}</style>
        <script>
        document.addEventListener('DOMContentLoaded', () => {{
            document.body.classList.add('dark');
            setTimeout(async () => {{
                try {{
                    const r = await fetch('/api/mobile/info');
                    const d = await r.json();
                    if (d.public_url) {{
                        const btn = document.getElementById('ov-pub-btn');
                        if (btn) {{
                            btn.style.display = 'inline-flex';
                            btn.href = d.public_url + '/mobile';
                            btn.title = 'Link HTTPS dùng trên mọi điện thoại (4G/5G/Wi-Fi): ' + d.public_url + '/mobile';
                        }}
                    }}
                }} catch (e) {{}}
            }}, 3500);
        }});
        function ovToggleMobileFrame() {{
            document.body.classList.toggle('ov-mobile-frame');
            const lbl = document.getElementById('ov-frame-label');
            const icon = document.getElementById('ov-frame-icon');
            const isMob = document.body.classList.contains('ov-mobile-frame');
            if (lbl) lbl.textContent = isMob ? 'Khung PC' : 'Khung ĐT';
            if (icon) {{
                icon.innerHTML = isMob
                    ? '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>'
                    : '<rect x="5" y="2" width="14" height="20" rx="2"/><line x1="12" y1="18" x2="12.01" y2="18"/>';
            }}
        }}
        function ovCopyLanUrl(url) {{
            if (navigator.clipboard) {{
                navigator.clipboard.writeText(url);
            }}
            const el = document.getElementById('ov-lan-text');
            if (el) {{
                const orig = el.textContent;
                el.textContent = 'Đã chép link!';
                setTimeout(() => {{ el.textContent = orig; }}, 1800);
            }}
        }}
        </script>
        <header class="ov-topbar" role="banner">
            <div class="ov-brand">
                <span>🎙️ OmniVoice</span>
                <span class="ov-status-dot" title="6-Thread MKL Ready · C: Drive Protected">● D: {d_free}GB</span>
            </div>
            <nav class="ov-actions" aria-label="Công cụ nhanh">
                <a id="ov-pub-btn" href="/mobile" target="_blank" class="ov-pill-btn" style="display:none;border-color:rgba(34,197,94,0.45);color:#86efac !important;">
                    <span>🌐 Link 4G/HTTPS</span>
                </a>
                <button type="button" class="ov-pill-btn" onclick="ovCopyLanUrl('{primary_lan_url}/mobile')" title="Bấm để sao chép link mở trên điện thoại cùng Wi-Fi">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/><path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/></svg>
                    <span id="ov-lan-text">{lan_ips[0]}:8001</span>
                </button>
                <button type="button" class="ov-pill-btn ov-hide-mobile" onclick="ovToggleMobileFrame()" aria-label="Chuyển đổi khung xem thử điện thoại">
                    <svg id="ov-frame-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="5" y="2" width="14" height="20" rx="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>
                    <span id="ov-frame-label">Khung ĐT</span>
                </button>
                <a href="/mobile" class="ov-pill-btn accent" title="Mở giao diện Mobile App tối giản chuyên biệt">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
                    <span>Bản Mobile</span>
                </a>
            </nav>
        </header>
        """)

        # ─────────────────────────────────────────────────────────────────────
        # SINGLE UNIFIED STUDIO WORKSPACE (ZERO REPETITION)
        # ─────────────────────────────────────────────────────────────────────
        with gr.Row():
            # ── LEFT COLUMN: COMPOSER & CONTROLS ──
            with gr.Column(scale=6):
                with gr.Row(elem_classes=["ov-chip-row"]):
                    tpl_fast_btn = gr.Button("⚡ Mẫu 2.5s (Tức thì)", size="sm")
                    tpl_news_btn = gr.Button("📰 Bản tin 30s", size="sm")
                    tpl_story_btn = gr.Button("🌙 Truyện ngắn", size="sm")

                text_input = gr.Textbox(
                    label="Kịch bản văn bản",
                    lines=4,
                    placeholder="Nhập nội dung cần tạo giọng đọc & xuất video...",
                    value="Chào bạn, hôm nay thời tiết Cần Thơ rất đẹp.",
                )

                with gr.Row():
                    voice_dropdown = gr.Dropdown(
                        label="Giọng đọc",
                        choices=["[Giọng Mặc Định]"] + vm.list_voices(),
                        value=vm.list_voices()[0] if vm.list_voices() else "[Giọng Mặc Định]",
                        interactive=True,
                        scale=3,
                    )
                    perf_mode = gr.Dropdown(
                        label="Tốc độ AI",
                        choices=["TURBO", "FAST", "BALANCED", "QUALITY"],
                        value="TURBO",
                        interactive=True,
                        scale=2,
                    )

                with gr.Row():
                    make_video_chk = gr.Checkbox(
                        label="🎬 Tự động xuất kèm Video MP4 (Siêu tốc)",
                        value=True,
                    )

                run_btn = gr.Button("🚀 Tạo Giọng Đọc & Video", variant="primary", size="lg")

                # ── PROGRESSIVE DISCLOSURE DRAWER 1: ADVANCED AUDIO & VIDEO SETTINGS ──
                with gr.Accordion("🎛️ Tùy chỉnh nâng cao (Giọng nói & Video)", open=False):
                    with gr.Row():
                        speed_slider = gr.Slider(
                            minimum=0.8, maximum=1.3, value=1.0, step=0.05,
                            label="Tốc độ đọc (Speed)",
                        )
                        pause_scale_slider = gr.Slider(
                            minimum=0.8, maximum=1.5, value=1.0, step=0.05,
                            label="Khoảng nghỉ ngắt câu",
                        )
                    with gr.Row():
                        lang_dropdown = gr.Dropdown(
                            label="Ngôn ngữ",
                            choices=["Vietnamese", "Auto", "English", "Chinese", "Japanese", "French", "German"],
                            value="Vietnamese",
                        )
                        emotion_preset = gr.Dropdown(
                            label="Sắc thái cảm xúc",
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
                    with gr.Row():
                        normalize_num_chk = gr.Checkbox(label="Đọc số thành chữ tiếng Việt", value=False)
                        debug_mode_chk = gr.Checkbox(label="Phân tích âm học chi tiết", value=False)

                    gr.Markdown("##### 🎞️ Cấu hình Render Video MP4")
                    with gr.Row():
                        vid_preset = gr.Dropdown(
                            label="Chế độ Encode",
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
                            label="Độ phân giải",
                            choices=["1280x720", "1920x1080"],
                            value="1280x720",
                        )
                        vid_fps = gr.Dropdown(
                            label="FPS",
                            choices=[24, 30],
                            value=24,
                        )
                    with gr.Row():
                        vid_title = gr.Textbox(label="Tiêu đề trên Video", value="OmniVoice Studio", lines=1)
                        vid_subtitle = gr.Textbox(label="Phụ đề trên Video", value="AI Voice & Fast Video Workflow", lines=1)

                    with gr.Row():
                        vid_bg_image = gr.Image(label="Ảnh nền Video tùy chỉnh (Tùy chọn)", type="filepath")
                        existing_wav_input = gr.Audio(
                            label="Hoặc chọn file Audio (.wav) có sẵn để ghép Video ngay",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )

                    with gr.Accordion("📂 Clone nhanh 1-chạm (Không lưu vào thư viện) & Kiểm tra văn bản", open=False):
                        direct_audio = gr.Audio(
                            label="Audio mẫu 3-8s (One-Shot Clone)",
                            sources=["upload", "microphone"],
                            type="filepath",
                        )
                        direct_text = gr.Textbox(label="Câu nói trong audio mẫu", lines=1)
                        preview_text_btn = gr.Button("🧹 Xem văn bản sau khi lọc ký tự", size="sm")
                        preview_text_output = gr.Textbox(label="Văn bản phát âm thực tế", lines=2, interactive=False)

                # ── PROGRESSIVE DISCLOSURE DRAWER 2: VOICE CLONING & LIBRARY ──
                with gr.Accordion("🎙️ Quản lý & Thu âm Giọng mới (Voice Clone)", open=False):
                    with gr.Row():
                        new_voice_name = gr.Textbox(label="Tên giọng mới (VD: Giọng Của Tôi)", lines=1)
                        new_voice_text = gr.Textbox(
                            label="Câu nói trong mẫu thu âm",
                            value="Xin chào các bạn, hôm nay trời rất đẹp.",
                            lines=1,
                        )
                    new_voice_audio = gr.Audio(
                        label="Thu âm trực tiếp từ Micro hoặc tải file (3-8 giây)",
                        sources=["microphone", "upload"],
                        type="filepath",
                    )
                    with gr.Row():
                        create_voice_btn = gr.Button("💾 Lưu Giọng Mới (.pt)", variant="secondary", size="sm")
                        refresh_voice_btn = gr.Button("🔄 Làm mới", size="sm")
                    create_voice_status = gr.Markdown("")

                    with gr.Accordion("🗑️ Xóa giọng khỏi thư viện", open=False):
                        with gr.Row():
                            delete_voice_dropdown = gr.Dropdown(
                                label="Chọn giọng cần xóa",
                                choices=vm.list_voices(),
                                value=vm.list_voices()[0] if vm.list_voices() else None,
                            )
                            delete_voice_btn = gr.Button("🗑️ Xóa", variant="stop", size="sm")

            # ── RIGHT COLUMN: UNIFIED OUTPUT STAGE ──
            with gr.Column(scale=5):
                status_box = gr.Markdown(
                    "🟢 **Sẵn sàng** — Nhập kịch bản và bấm **Tạo Giọng Đọc & Video**",
                    elem_classes=["ov-status-line"],
                )
                output_audio = gr.Audio(label="🔊 Âm thanh (24kHz WAV)", type="filepath")
                output_video = gr.Video(label="🎥 Video Hoàn Chỉnh (MP4)")

                # ── PROGRESSIVE DISCLOSURE DRAWER 3: TELEMETRY & HARDWARE AUDIT ──
                with gr.Accordion("📊 Chi tiết Hiệu năng & Hệ thống", open=False):
                    metrics_display = gr.Markdown("Chưa có dữ liệu chạy mới.")
                    debug_table = gr.Dataframe(
                        headers=["Đoạn", "Văn bản", "Thời lượng", "Trạng thái"],
                        interactive=False,
                    )
                    res_info = get_detailed_resource_usage(str(storage.base_dir))
                    gr.Markdown(f"""
**Phần cứng:** `{hw_info['cpu_name']}` ({hw_info['recommended_threads']} Threads MKL) · `{hw_info['primary_gpu']}`  
**Bộ nhớ:** RAM `{res_info['ram_process_mb']} MB` · Ổ C: trống `{res_info['c_drive_free_gb']} GB` (Đã bảo vệ) · Ổ D: trống `{res_info.get('d_drive_free_gb', 0)} GB`  
**Link Điện thoại (Cùng Wi-Fi):** [`{primary_lan_url}/mobile`]({primary_lan_url}/mobile)
""")

        # ─────────────────────────────────────────────────────────────────────
        # EVENT WIRING
        # ─────────────────────────────────────────────────────────────────────
        tpl_fast_btn.click(
            fn=lambda: "Chào bạn, hôm nay thời tiết Cần Thơ rất đẹp.",
            outputs=[text_input],
        )
        tpl_news_btn.click(
            fn=lambda: (
                "Chào mừng các bạn đến với hệ thống sản xuất giọng nói và video tự động OmniVoice. "
                "Phiên bản mới nhất đã được tối ưu hóa riêng cho máy tính Dell Inspiron và điện thoại di động."
            ),
            outputs=[text_input],
        )
        tpl_story_btn.click(
            fn=lambda: "Có những người xuất hiện đúng lúc trái tim muốn yêu, nhưng lại sai lúc cả hai biết cách ở bên nhau.",
            outputs=[text_input],
        )

        preview_text_btn.click(
            fn=lambda t, n: TextPreprocessor.process(t, normalize_nums=n),
            inputs=[text_input, normalize_num_chk],
            outputs=[preview_text_output],
        )

        def _unified_run(
            text, voice, mode, make_vid,
            speed, pause_scale, lang, emotion_sel, norm_nums, debug_mode,
            v_preset, v_res, v_fps, v_title, v_sub, v_bg, existing_wav,
            dir_aud, dir_txt,
            progress=gr.Progress(),
        ):
            if not existing_wav and (not text or not text.strip()):
                return "⚠️ **Vui lòng nhập kịch bản văn bản hoặc chọn file WAV có sẵn.**", None, None, "", []

            voice_arg = None if voice == "[Giọng Mặc Định]" else voice
            t_start = time.perf_counter()

            try:
                audio_file_path = existing_wav
                tts_time = 0.0
                audio_dur = 0.0
                rtf_val = 0.0
                cache_hit = True
                cleaned_txt = text.strip() if text else ""

                if not audio_file_path:
                    def on_progress(curr, tot, pct, msg, perf_info=None):
                        progress(pct / 100.0, desc=f"🎙️ Đang xử lý đoạn {curr}/{tot}...")

                    res = engine.synthesize(
                        text=text,
                        voice_name=voice_arg,
                        ref_audio_path=dir_aud,
                        ref_text=dir_txt,
                        language=lang,
                        perf_mode=mode,
                        speed=speed,
                        debug_mode=debug_mode,
                        normalize_numbers=norm_nums,
                        progress_callback=on_progress,
                    )
                    tts_time = res.generation_time
                    audio_dur = res.audio_duration
                    rtf_val = res.rtf
                    cache_hit = res.cached_hit
                    cleaned_txt = res.cleaned_text
                    audio_file_path = res.output_path
                    if not audio_file_path or not Path(audio_file_path).exists():
                        audio_file_path = str(storage.cache_temp_dir / f"studio_tts_{int(time.time())}.wav")
                        sf.write(audio_file_path, res.audio, res.sample_rate)

                vid_path = None
                vid_comp_s = 0.0
                vid_enc_s = 0.0
                vid_fps_val = 0.0
                vid_size_mb = 0.0

                if make_vid:
                    progress(0.82, desc="🎬 Đang kết xuất Video MP4...")
                    vres = engine.export_video(
                        text=cleaned_txt or "OmniVoice Video",
                        audio_path=audio_file_path,
                        voice_name=voice_arg,
                        perf_mode=mode,
                        video_preset=v_preset,
                        resolution=v_res,
                        fps=int(v_fps),
                        title=v_title or "OmniVoice Studio",
                        subtitle=v_sub or f"Voice: {voice} | Mode: {mode}",
                        custom_image_path=v_bg,
                        use_audio_cache=True,
                        use_visual_cache=True,
                    )
                    vid_path = vres.output_video_path
                    vid_comp_s = vres.video_composition_time_sec
                    vid_enc_s = vres.video_encoding_time_sec
                    vid_fps_val = vres.rendering_fps
                    vid_size_mb = vres.output_file_size_mb
                    if audio_dur <= 0:
                        audio_dur = vres.video_duration_sec

                total_elapsed = time.perf_counter() - t_start
                badge = "⚡ CACHE HIT" if cache_hit else "🧠 CPU AI"
                vid_summary = f" · 🎬 Video `{vid_comp_s + vid_enc_s:.2f}s` (`{vid_fps_val:.0f} FPS`)" if make_vid else ""
                status_md = (
                    f"✅ **Hoàn tất trong `{total_elapsed:.2f}s`** ({badge}) — "
                    f"Thời lượng `{audio_dur:.1f}s` · 🎙️ Audio `{tts_time:.2f}s`{vid_summary}"
                )

                metrics_md = f"""
| Giai đoạn | Thời gian / Chỉ số |
|---|---|
| **1. Tổng hợp Giọng nói (TTS)** | `{tts_time:.3f} s` (`RTF: {rtf_val:.3f}x` · `{badge}`) |
| **2. Dựng hình Video (Composition)** | `{vid_comp_s:.3f} s` |
| **3. Mã hóa MP4 (FFmpeg {v_preset})** | `{vid_enc_s:.3f} s` (`{vid_fps_val:.1f} FPS` · `{vid_size_mb:.2f} MB`) |
| **TỔNG THỜI GIAN PHẢN HỒI** | **`{total_elapsed:.3f} s`** |
"""
                rows = [[
                    "#1",
                    cleaned_txt[:48] + ("..." if len(cleaned_txt) > 48 else ""),
                    f"{audio_dur:.2f}s",
                    "OK",
                ]]
                return status_md, audio_file_path, vid_path, metrics_md, rows

            except Exception as e:
                return f"❌ **Lỗi:** `{type(e).__name__}: {e}`", None, None, "", []

        run_btn.click(
            fn=_unified_run,
            inputs=[
                text_input, voice_dropdown, perf_mode, make_video_chk,
                speed_slider, pause_scale_slider, lang_dropdown, emotion_preset, normalize_num_chk, debug_mode_chk,
                vid_preset, vid_res, vid_fps, vid_title, vid_subtitle, vid_bg_image, existing_wav_input,
                direct_audio, direct_text,
            ],
            outputs=[status_box, output_audio, output_video, metrics_display, debug_table],
        )

        # Voice Library Handlers
        def _save_voice(name, audio_path, ref_txt):
            if not name or not name.strip() or not audio_path:
                return "⚠️ Nhập tên giọng và thu âm/chọn file mẫu (3-8s).", gr.update(), gr.update()
            try:
                profile = vm.create_or_update_voice(
                    name=name.strip(),
                    ref_audio_path=audio_path,
                    ref_text=ref_txt.strip() if ref_txt else "Xin chào các bạn.",
                    model=engine.model,
                    description="Minimal Studio",
                )
                all_v = vm.list_voices()
                return (
                    f"✅ Đã lưu giọng **'{profile.name}'**",
                    gr.update(choices=["[Giọng Mặc Định]"] + all_v, value=profile.name),
                    gr.update(choices=all_v, value=profile.name),
                )
            except Exception as e:
                return f"❌ Lỗi: {e}", gr.update(), gr.update()

        def _delete_voice(name_to_del):
            if not name_to_del:
                return "⚠️ Chọn giọng cần xóa.", gr.update(), gr.update()
            vm.delete_voice(name_to_del.strip())
            all_v = vm.list_voices()
            return (
                f"🗑️ Đã xóa giọng **'{name_to_del}'**",
                gr.update(choices=["[Giọng Mặc Định]"] + all_v, value=all_v[0] if all_v else "[Giọng Mặc Định]"),
                gr.update(choices=all_v, value=all_v[0] if all_v else None),
            )

        create_voice_btn.click(
            fn=_save_voice,
            inputs=[new_voice_name, new_voice_audio, new_voice_text],
            outputs=[create_voice_status, voice_dropdown, delete_voice_dropdown],
        )
        delete_voice_btn.click(
            fn=_delete_voice,
            inputs=[delete_voice_dropdown],
            outputs=[create_voice_status, voice_dropdown, delete_voice_dropdown],
        )
        refresh_voice_btn.click(
            fn=lambda: (
                gr.update(choices=["[Giọng Mặc Định]"] + vm.list_voices()),
                gr.update(choices=vm.list_voices(), value=vm.list_voices()[0] if vm.list_voices() else None),
            ),
            outputs=[voice_dropdown, delete_voice_dropdown],
        )

    return demo
