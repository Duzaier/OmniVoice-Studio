#!/usr/bin/env python3
"""
OmniVoice Fast Demo — CPU-optimized launcher.

Cải tiến so với demo gốc:
  1. Tăng số CPU threads tối đa cho PyTorch
  2. Mặc định num_step = 16 thay vì 32 (nhanh 2x, chất lượng vẫn tốt)
  3. Áp dụng torch.compile() để tối ưu graph (tăng ~20-40% tốc độ sau lần đầu)
  4. Bỏ load Whisper ASR mặc định (--no-asr) để tiết kiệm RAM và thời gian khởi động
     -> Nếu muốn ASR, thêm flag --asr khi chạy

Usage:
    uv run python omnivoice/cli/demo_fast.py --port 8001
    uv run python omnivoice/cli/demo_fast.py --port 8001 --asr   # bật lại ASR
    uv run python omnivoice/cli/demo_fast.py --port 8001 --steps 8   # nhanh hơn nữa
"""

import argparse
import logging
import os
import torch

# -----------------------------------------------------------------------
# 1. Tối ưu CPU threads TRƯỚC khi import torch nn
# -----------------------------------------------------------------------
_cpu_count = os.cpu_count() or 4
torch.set_num_threads(_cpu_count)
torch.set_num_interop_threads(max(1, _cpu_count // 2))

import numpy as np
import gradio as gr

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name

_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "Gender / 性别": ["Male / 男", "Female / 女"],
    "Age / 年龄": [
        "Child / 儿童",
        "Teenager / 少年",
        "Young Adult / 青年",
        "Middle-aged / 中年",
        "Elderly / 老年",
    ],
    "Pitch / 音调": [
        "Very Low Pitch / 极低音调",
        "Low Pitch / 低音调",
        "Moderate Pitch / 中音调",
        "High Pitch / 高音调",
        "Very High Pitch / 极高音调",
    ],
    "Style / 风格": ["Whisper / 耳语"],
    "English Accent / 英文口音": [
        "American Accent / 美式口音",
        "Australian Accent / 澳大利亚口音",
        "British Accent / 英国口音",
        "Chinese Accent / 中国口音",
        "Canadian Accent / 加拿大口音",
        "Indian Accent / 印度口音",
        "Korean Accent / 韩国口音",
        "Portuguese Accent / 葡萄牙口音",
        "Russian Accent / 俄罗斯口音",
        "Japanese Accent / 日本口音",
    ],
    "Chinese Dialect / 中文方言": [
        "Henan Dialect / 河南话",
        "Shaanxi Dialect / 陕西话",
        "Sichuan Dialect / 四川话",
        "Guizhou Dialect / 贵州话",
        "Yunnan Dialect / 云南话",
        "Guilin Dialect / 桂林话",
        "Jinan Dialect / 济南话",
        "Shijiazhuang Dialect / 石家庄话",
        "Gansu Dialect / 甘肃话",
        "Ningxia Dialect / 宁夏话",
        "Qingdao Dialect / 青岛话",
        "Northeast Dialect / 东北话",
    ],
}

_ATTR_INFO = {
    "English Accent / 英文口音": "Only effective for English speech.",
    "Chinese Dialect / 中文方言": "Only effective for Chinese speech.",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="omnivoice-demo-fast",
        description="OmniVoice Fast Demo (CPU-optimized).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--root-path", default=None)
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
        help="Default inference steps (default: 16). Lower = faster, higher = better quality.",
    )
    parser.add_argument(
        "--asr",
        action="store_true",
        default=False,
        help="Load Whisper ASR model for auto-transcription. Disabled by default to save RAM.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Apply torch.compile() to the model for faster CPU inference (first run will be slow).",
    )
    return parser


def build_demo(model: OmniVoice, default_steps: int = 16) -> gr.Blocks:
    sampling_rate = model.sampling_rate

    def _gen_core(text, language, ref_audio, instruct, num_step, guidance_scale,
                  denoise, speed, duration, preprocess_prompt, postprocess_output,
                  mode, ref_text=None):
        if not text or not text.strip():
            return None, "Vui lòng nhập văn bản cần tổng hợp."

        gen_config = OmniVoiceGenerationConfig(
            num_step=int(num_step or default_steps),
            guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
            denoise=bool(denoise) if denoise is not None else True,
            preprocess_prompt=bool(preprocess_prompt),
            postprocess_output=bool(postprocess_output),
        )

        lang = language if (language and language != "Auto") else None
        kw = dict(text=text.strip(), language=lang, generation_config=gen_config)

        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        if mode == "clone":
            if not ref_audio:
                return None, "Vui lòng upload audio tham chiếu."
            kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
                ref_audio=ref_audio, ref_text=ref_text,
            )

        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Lỗi: {type(e).__name__}: {e}"

        waveform = (audio[0] * 32767).astype(np.int16)
        return (sampling_rate, waveform), "✅ Hoàn thành!"

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------
    theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
    css = """
    .gradio-container {max-width: 100% !important; font-size: 16px !important;}
    .gradio-container h1 {font-size: 1.5em !important;}
    .compact-audio audio {height: 60px !important;}
    .compact-audio .waveform {min-height: 80px !important;}
    """

    def _lang_dropdown(label="Language / 语种", value="Auto"):
        return gr.Dropdown(
            label=label, choices=_ALL_LANGUAGES, value=value,
            allow_custom_value=False, interactive=True,
            info="Để 'Auto' để tự nhận diện ngôn ngữ.",
        )

    def _gen_settings(default_ns):
        with gr.Accordion("⚙️ Cài đặt nâng cao (tùy chọn)", open=False):
            sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Tốc độ (Speed)")
            du = gr.Number(value=None, label="Thời lượng cố định (giây) — để trống = tự động")
            ns = gr.Slider(
                4, 64, value=default_ns, step=1,
                label=f"Inference Steps (mặc định: {default_ns})",
                info=f"💡 {default_ns} bước = nhanh & chất lượng tốt. Giảm xuống 8 để nhanh hơn nữa.",
            )
            dn = gr.Checkbox(label="Denoise", value=True)
            gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
            pp = gr.Checkbox(label="Preprocess Prompt", value=True)
            po = gr.Checkbox(label="Postprocess Output", value=True)
        return ns, gs, dn, sp, du, pp, po

    with gr.Blocks(theme=theme, css=css, title="OmniVoice Fast Demo") as demo:
        gr.Markdown(f"""
# 🎙️ OmniVoice — Fast Demo (CPU)

Mô hình TTS đa ngôn ngữ 600+ ngôn ngữ, bao gồm **tiếng Việt**.

> ⚡ **Chế độ nhanh**: Mặc định {default_steps} bước inference (thay vì 32).  
> 💡 **Mẹo**: Văn bản ngắn hơn = tạo âm thanh nhanh hơn.
""")

        with gr.Tabs():
            # -- Voice Clone Tab --
            with gr.TabItem("🎤 Voice Clone (Nhân bản giọng)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vc_text = gr.Textbox(
                            label="Văn bản cần tổng hợp", lines=4,
                            placeholder="Nhập văn bản tại đây...",
                        )
                        vc_ref_audio = gr.Audio(
                            label="Audio tham chiếu (3–10 giây)", type="filepath",
                            elem_classes="compact-audio",
                        )
                        vc_ref_text = gr.Textbox(
                            label="Nội dung audio tham chiếu (tùy chọn)", lines=2,
                            placeholder="Để trống sẽ tự động nhận dạng (nếu có ASR).",
                        )
                        vc_lang = _lang_dropdown("Ngôn ngữ")
                        with gr.Accordion("Instruct (tùy chọn)", open=False):
                            vc_instruct = gr.Textbox(label="Instruct", lines=2)
                        vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings(default_steps)
                        vc_btn = gr.Button("🎵 Tạo giọng nói", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        vc_audio = gr.Audio(label="Kết quả", type="numpy")
                        vc_status = gr.Textbox(label="Trạng thái", lines=2)

                def _clone_fn(text, lang, ref_aud, ref_text, instruct, ns, gs, dn, sp, du, pp, po):
                    return _gen_core(text, lang, ref_aud, instruct, ns, gs, dn, sp, du, pp, po,
                                     mode="clone", ref_text=ref_text or None)

                vc_btn.click(_clone_fn,
                             inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text,
                                     vc_instruct, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                             outputs=[vc_audio, vc_status])

            # -- Voice Design Tab --
            with gr.TabItem("🎨 Voice Design (Thiết kế giọng)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vd_text = gr.Textbox(
                            label="Văn bản cần tổng hợp", lines=4,
                            placeholder="Nhập văn bản tại đây...",
                        )
                        vd_lang = _lang_dropdown()
                        _AUTO = "Auto"
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            vd_groups.append(gr.Dropdown(
                                label=_cat, choices=[_AUTO] + _choices, value=_AUTO,
                                info=_ATTR_INFO.get(_cat),
                            ))
                        vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings(default_steps)
                        vd_btn = gr.Button("🎵 Tạo giọng nói", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        vd_audio = gr.Audio(label="Kết quả", type="numpy")
                        vd_status = gr.Textbox(label="Trạng thái", lines=2)

                def _build_instruct(groups):
                    selected = [g for g in groups if g and g != "Auto"]
                    if not selected:
                        return None
                    parts = []
                    for v in selected:
                        if " / " in v:
                            en, zh = v.split(" / ", 1)
                            parts.append(zh.strip() if "Dialect" in v.split(" / ")[0] else en.strip())
                        else:
                            parts.append(v)
                    return ", ".join(parts)

                def _design_fn(text, lang, ns, gs, dn, sp, du, pp, po, *groups):
                    return _gen_core(text, lang, None, _build_instruct(groups),
                                     ns, gs, dn, sp, du, pp, po, mode="design")

                vd_btn.click(_design_fn,
                             inputs=[vd_text, vd_lang, vd_ns, vd_gs, vd_dn,
                                     vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                             outputs=[vd_audio, vd_status])

    return demo


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    device = args.device or "cpu"

    logging.info(f"CPU threads: {torch.get_num_threads()}")
    logging.info(f"Loading model from {args.model}, device={device}, default_steps={args.steps}")

    model = OmniVoice.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.float32,   # CPU phải dùng float32 (float16 không hỗ trợ trên CPU)
        load_asr=args.asr,
        asr_model_name="openai/whisper-large-v3-turbo",
    )

    # -----------------------------------------------------------------------
    # 3. torch.compile() — tối ưu graph cho CPU (tùy chọn)
    # -----------------------------------------------------------------------
    if args.compile:
        logging.info("Applying torch.compile() to the model (first inference will be slow)...")
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead", backend="inductor")
            logging.info("torch.compile() applied successfully.")
        except Exception as e:
            logging.warning(f"torch.compile() failed: {e}. Continuing without compilation.")

    logging.info("Model loaded. Starting Gradio demo...")
    demo = build_demo(model, default_steps=args.steps)

    demo.queue().launch(
        server_name=args.ip,
        server_port=args.port,
        share=args.share,
        root_path=args.root_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
