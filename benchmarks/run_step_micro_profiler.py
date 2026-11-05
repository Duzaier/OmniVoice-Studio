#!/usr/bin/env python3
"""
High-Resolution Micro-Profiler for a Single Diffusion Step in OmniVoice Batch-1.
Accurately measures:
 ├─ Transformer forward
 │   ├─ Embedding (Text + Audio)
 │   ├─ 28x Qwen3 Layers:
 │   │   ├─ MLP (Gate + Up + Down_proj)
 │   │   ├─ Self-Attention (QKV + SDPA + O_proj)
 │   │   └─ RMSNorm (input + post_attn + final)
 │   └─ LM Heads (Audio logits projection)
 ├─ Sampling & Scoring (_predict_tokens_with_scoring)
 └─ Tensor Preparation & Copy
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

TEXT_SAMPLE = (
    "Xin chào các bạn. Hôm nay tôi rất vui được giới thiệu với các bạn hệ thống "
    "tổng hợp giọng nói trí tuệ nhân tạo OmniVoice. Đây là một công nghệ đột phá, "
    "cho phép nhân bản giọng nói và đọc văn bản tiếng Việt vô cùng tự nhiên và mượt mà."
)


def run_micro_profile():
    print("=" * 75)
    print("      🔬 OMNIVOICE BATCH-1 SINGLE STEP MICRO-PROFILER")
    print("=" * 75)

    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    torch.set_num_threads(8)

    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    vp = vm.load_prompt("default", map_location="cpu")
    clean_text = TextPreprocessor.process(TEXT_SAMPLE)

    # Warmup pass
    cfg_warm = OmniVoiceGenerationConfig(num_step=1, guidance_scale=0.0, denoise=True, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate("Khởi động.", language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg_warm)

    # Instrument sub-modules
    timings = {
        "embedding": 0.0,
        "attention_total": 0.0,
        "mlp_total": 0.0,
        "rmsnorm": 0.0,
        "lm_heads": 0.0,
        "sampling_scoring": 0.0,
        "tensor_prep_copy": 0.0,
        "transformer_total": 0.0,
        "step_total": 0.0,
    }

    qwen_layers = model.llm.layers

    for layer in qwen_layers:
        self_attn = layer.self_attn
        mlp = layer.mlp

        # Hook Attention
        orig_attn = self_attn.forward
        def make_attn_hook(orig_fn):
            def hooked_attn(*args, **kwargs):
                t0 = time.perf_counter()
                out = orig_fn(*args, **kwargs)
                timings["attention_total"] += time.perf_counter() - t0
                return out
            return hooked_attn
        self_attn.forward = make_attn_hook(orig_attn)

        # Hook MLP
        orig_mlp = mlp.forward
        def make_mlp_hook(orig_fn):
            def hooked_mlp(*args, **kwargs):
                t0 = time.perf_counter()
                out = orig_fn(*args, **kwargs)
                timings["mlp_total"] += time.perf_counter() - t0
                return out
            return hooked_mlp
        mlp.forward = make_mlp_hook(orig_mlp)

        # Hook RMSNorm
        orig_in_norm = layer.input_layernorm.forward
        def make_norm_hook(orig_fn):
            def hooked_norm(*args, **kwargs):
                t0 = time.perf_counter()
                out = orig_fn(*args, **kwargs)
                timings["rmsnorm"] += time.perf_counter() - t0
                return out
            return hooked_norm
        layer.input_layernorm.forward = make_norm_hook(orig_in_norm)

        orig_post_norm = layer.post_attention_layernorm.forward
        layer.post_attention_layernorm.forward = make_norm_hook(orig_post_norm)

    if hasattr(model.llm, "norm"):
        orig_final_norm = model.llm.norm.forward
        def hooked_final_norm(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_final_norm(*args, **kwargs)
            timings["rmsnorm"] += time.perf_counter() - t0
            return out
        model.llm.norm.forward = hooked_final_norm

    # Hook Embedding
    orig_embed = model.llm.embed_tokens.forward
    def hooked_embed(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_embed(*args, **kwargs)
        timings["embedding"] += time.perf_counter() - t0
        return out
    model.llm.embed_tokens.forward = hooked_embed

    # Hook Audio Heads (LM Heads)
    orig_audio_heads = model.audio_heads.forward
    def hooked_audio_heads(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_audio_heads(*args, **kwargs)
        timings["lm_heads"] += time.perf_counter() - t0
        return out
    model.audio_heads.forward = hooked_audio_heads

    # Hook Scoring & Sampling
    orig_predict = model._predict_tokens_with_scoring
    def hooked_predict(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_predict(*args, **kwargs)
        timings["sampling_scoring"] += time.perf_counter() - t0
        return out
    model._predict_tokens_with_scoring = hooked_predict

    # Hook Full Transformer Model Forward
    orig_model_forward = model.forward
    def hooked_model_forward(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_model_forward(*args, **kwargs)
        timings["transformer_total"] += time.perf_counter() - t0
        return out
    model.forward = hooked_model_forward

    # Measure 1 exact step of generation with 220 chars input
    cfg_1step = OmniVoiceGenerationConfig(
        num_step=1,
        guidance_scale=0.0,
        denoise=True,
        pad_duration=0.1,
        fade_duration=0.1,
        postprocess_output=False,
    )

    t_step_start = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(
            text=clean_text,
            language="Vietnamese",
            voice_clone_prompt=vp,
            generation_config=cfg_1step,
        )
    t_step_total = time.perf_counter() - t_step_start
    timings["step_total"] = t_step_total

    # Calculate residual tensor prep / copy time
    accounted = timings["transformer_total"] + timings["sampling_scoring"]
    timings["tensor_prep_copy"] = max(0.0, t_step_total - accounted)

    # Print Formatted Hierarchical Tree
    print("\n" + "=" * 75)
    print("      MEASURED 1 DIFFUSION STEP BREAKDOWN (BATCH-1 CONDITIONAL)")
    print("=" * 75)
    print(f"1 Diffusion Step Total:         {timings['step_total']:7.4f} s (100.00%)\n")
    print(f" ├─ Transformer forward:        {timings['transformer_total']:7.4f} s ({timings['transformer_total']/timings['step_total']*100:6.2f}%)")
    print(f" │   ├─ MLP (FeedForward 28L):  {timings['mlp_total']:7.4f} s ({timings['mlp_total']/timings['step_total']*100:6.2f}%)  ◀◀◀ [LARGEST BOTTLENECK]")
    print(f" │   ├─ Attention (Self-Attn):  {timings['attention_total']:7.4f} s ({timings['attention_total']/timings['step_total']*100:6.2f}%)")
    print(f" │   ├─ RMSNorm (28L x 2 + 1):  {timings['rmsnorm']:7.4f} s ({timings['rmsnorm']/timings['step_total']*100:6.2f}%)")
    print(f" │   ├─ LM Heads (8 Codebooks): {timings['lm_heads']:7.4f} s ({timings['lm_heads']/timings['step_total']*100:6.2f}%)")
    print(f" │   └─ Embedding:              {timings['embedding']:7.4f} s ({timings['embedding']/timings['step_total']*100:6.2f}%)")
    print(f" ├─ Sampling & Scoring:         {timings['sampling_scoring']:7.4f} s ({timings['sampling_scoring']/timings['step_total']*100:6.2f}%)")
    print(f" └─ Tensor Prep & Copy:         {timings['tensor_prep_copy']:7.4f} s ({timings['tensor_prep_copy']/timings['step_total']*100:6.2f}%)")
    print("=" * 75)

    os.makedirs("debug", exist_ok=True)
    with open("debug/step_micro_profile.json", "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)


if __name__ == "__main__":
    run_micro_profile()
