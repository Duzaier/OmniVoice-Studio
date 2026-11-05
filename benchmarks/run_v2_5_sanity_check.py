"""
OmniVoice V2.5 - Sanity Benchmark
Running full V2.1 golden baseline inference at optimal 6 threads to verify latency and quality
"""

import os
import sys
import time
import json
import numpy as np
import soundfile as sf
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

TEXT_220 = (
    "Hôm nay tôi xin trình bày báo cáo về tiến độ phát triển hệ thống trí tuệ nhân tạo. "
    "Mô hình xử lý ngôn ngữ và tổng hợp âm thanh đa ngôn ngữ đang được tối ưu hóa liên tục, "
    "nhằm đạt được hiệu suất cao nhất trên các nền tảng thiết bị cá nhân."
)

def run_sanity_check():
    torch.set_num_threads(6)
    print("=== OMNIVOICE V2.5 SANITY VERIFICATION (6 THREADS) ===", flush=True)
    
    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    
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

    # Warmup
    print("Warmup pass...", flush=True)
    with torch.inference_mode():
        _ = model.generate(clean_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)

    # Measured Run
    print("Executing measured run...", flush=True)
    t0 = time.perf_counter()
    with torch.inference_mode():
        res_audios = model.generate(clean_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg)
    t1 = time.perf_counter()

    gen_time = t1 - t0
    audio_arr = res_audios[0]
    if isinstance(audio_arr, torch.Tensor):
        audio_arr = audio_arr.cpu().numpy()
    audio_dur = len(audio_arr) / model.sampling_rate
    rtf = gen_time / audio_dur if audio_dur > 0 else 0

    print(f"Generation Time: {gen_time:.2f}s | Audio Duration: {audio_dur:.2f}s | RTF: {rtf:.2f}x", flush=True)

    # Save output for verification
    out_path = "benchmarks/outputs/v2_5_sanity_check_6threads.wav"
    os.makedirs("benchmarks/outputs", exist_ok=True)
    sf.write(out_path, audio_arr, model.sampling_rate)
    print(f"Saved audio to: {out_path}", flush=True)

    # Check signal
    rms = np.sqrt(np.mean(audio_arr**2))
    max_val = np.max(np.abs(audio_arr))
    print(f"Audio RMS: {rms:.4f}, Peak: {max_val:.4f}", flush=True)
    assert not np.isnan(audio_arr).any(), "NaN in audio output!"
    assert rms > 0.01, "Audio is silent!"
    print("✅ SANITY CHECK PASSED!", flush=True)

if __name__ == "__main__":
    run_sanity_check()
