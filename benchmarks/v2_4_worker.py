#!/usr/bin/env python3
"""
V2.4 Threading Benchmark Isolated Subprocess Worker.
Runs 1 warmup + 3 measured runs for a specific thread count with component-level timing.
Outputs JSON to stdout.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

TEXT_220 = (
    "Xin chào các bạn. Hôm nay tôi rất vui được giới thiệu với các bạn hệ thống "
    "tổng hợp giọng nói trí tuệ nhân tạo OmniVoice. Đây là một công nghệ đột phá, "
    "cho phép nhân bản giọng nói và đọc văn bản tiếng Việt vô cùng tự nhiên và mượt mà."
)


def run_worker(num_threads: int):
    # Set PyTorch threading
    torch.set_num_threads(num_threads)
    interop = max(1, min(4, num_threads // 2))
    torch.set_num_interop_threads(interop)

    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))

    # Load Model & Prompt
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    vp = vm.load_prompt("default", map_location="cpu")
    clean_text = TextPreprocessor.process(TEXT_220)

    cfg = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        denoise=True,
        pad_duration=0.1,
        fade_duration=0.1,
        postprocess_output=True,
    )

    # 1. Warmup Run (Untracked)
    with torch.inference_mode():
        _ = model.generate(clean_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)

    # 2. Instrument Model for 3 Measured Runs
    runs = []

    for r in range(1, 4):
        step_metrics = {
            "attn_time": 0.0,
            "mlp_time": 0.0,
            "rmsnorm_time": 0.0,
            "lm_head_time": 0.0,
            "transformer_time": 0.0,
        }

        # Hook Attention & MLP on all 28 layers
        orig_hooks = []
        for layer in model.llm.layers:
            # Attention
            orig_attn = layer.self_attn.forward
            def make_attn_hook(fn):
                def hooked(*args, **kwargs):
                    t0 = time.perf_counter()
                    out = fn(*args, **kwargs)
                    step_metrics["attn_time"] += time.perf_counter() - t0
                    return out
                return hooked
            layer.self_attn.forward = make_attn_hook(orig_attn)
            orig_hooks.append((layer.self_attn, "forward", orig_attn))

            # MLP
            orig_mlp = layer.mlp.forward
            def make_mlp_hook(fn):
                def hooked(*args, **kwargs):
                    t0 = time.perf_counter()
                    out = fn(*args, **kwargs)
                    step_metrics["mlp_time"] += time.perf_counter() - t0
                    return out
                return hooked
            layer.mlp.forward = make_mlp_hook(orig_mlp)
            orig_hooks.append((layer.mlp, "forward", orig_mlp))

            # Norms
            orig_in_norm = layer.input_layernorm.forward
            def make_norm_hook(fn):
                def hooked(*args, **kwargs):
                    t0 = time.perf_counter()
                    out = fn(*args, **kwargs)
                    step_metrics["rmsnorm_time"] += time.perf_counter() - t0
                    return out
                return hooked
            layer.input_layernorm.forward = make_norm_hook(orig_in_norm)
            orig_hooks.append((layer.input_layernorm, "forward", orig_in_norm))

            orig_post_norm = layer.post_attention_layernorm.forward
            layer.post_attention_layernorm.forward = make_norm_hook(orig_post_norm)
            orig_hooks.append((layer.post_attention_layernorm, "forward", orig_post_norm))

        # Audio Head
        orig_head = model.audio_heads.forward
        def make_head_hook(fn):
            def hooked(*args, **kwargs):
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                step_metrics["lm_head_time"] += time.perf_counter() - t0
                return out
            return hooked
        model.audio_heads.forward = make_head_hook(orig_head)
        orig_hooks.append((model.audio_heads, "forward", orig_head))

        # Transformer Forward
        orig_model_fwd = model.forward
        def make_model_hook(fn):
            def hooked(*args, **kwargs):
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                step_metrics["transformer_time"] += time.perf_counter() - t0
                return out
            return hooked
        model.forward = make_model_hook(orig_model_fwd)
        orig_hooks.append((model, "forward", orig_model_fwd))

        # Measured Generation Call
        t_gen_start = time.perf_counter()
        with torch.inference_mode():
            audios = model.generate(
                text=clean_text,
                language="Vietnamese",
                voice_clone_prompt=vp,
                generation_config=cfg,
            )
        t_gen_total = time.perf_counter() - t_gen_start

        # Unhook
        for obj, attr, orig_fn in orig_hooks:
            setattr(obj, attr, orig_fn)

        raw_audio = audios[0]
        audio_dur = len(raw_audio) / model.sampling_rate
        rtf = t_gen_total / audio_dur if audio_dur > 0 else 0
        vocoder_time = max(0.0, t_gen_total - step_metrics["transformer_time"])

        runs.append({
            "run": r,
            "total_time_s": round(t_gen_total, 3),
            "transformer_time_s": round(step_metrics["transformer_time"], 3),
            "attention_time_s": round(step_metrics["attn_time"], 3),
            "mlp_time_s": round(step_metrics["mlp_time"], 3),
            "rmsnorm_time_s": round(step_metrics["rmsnorm_time"], 3),
            "lm_head_time_s": round(step_metrics["lm_head_time"], 3),
            "vocoder_time_s": round(vocoder_time, 3),
            "audio_dur_s": round(audio_dur, 2),
            "rtf": round(rtf, 3),
        })

    def calc_stats(metric_key):
        vals = [x[metric_key] for x in runs]
        return {
            "mean": round(float(np.mean(vals)), 3),
            "median": round(float(np.median(vals)), 3),
            "min": round(float(np.min(vals)), 3),
            "max": round(float(np.max(vals)), 3),
            "std": round(float(np.std(vals)), 3),
        }

    output = {
        "threads": num_threads,
        "interop_threads": interop,
        "runs": runs,
        "stats": {
            "total_time": calc_stats("total_time_s"),
            "transformer": calc_stats("transformer_time_s"),
            "attention": calc_stats("attention_time_s"),
            "mlp": calc_stats("mlp_time_s"),
            "rmsnorm": calc_stats("rmsnorm_time_s"),
            "lm_head": calc_stats("lm_head_time_s"),
            "vocoder": calc_stats("vocoder_time_s"),
            "rtf": calc_stats("rtf"),
        }
    }

    # Print pure JSON marker
    print("===RESULT_JSON_START===")
    print(json.dumps(output))
    print("===RESULT_JSON_END===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    run_worker(args.threads)
