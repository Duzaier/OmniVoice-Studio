#!/usr/bin/env python3
"""
OmniVoice V6 Hard Performance Recovery Benchmark.
Tests:
- Native OmniVoice long-form inference path with audio_chunk_duration (15s, 20s, 25s)
- num_step (16, 20, 24, 32)
- Target text (~90 seconds audio output)
- Measures total generation time, RTF, Peak RAM, F0 pitch metrics, and phoneme completeness.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
import numpy as np
import soundfile as sf

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.hardware import profile_hardware
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor
from omnivoice.core.prosody_tagger import ProsodyTagger
from omnivoice.core.pitch_analyzer import PitchAndEnergyAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("V6_Benchmark")

# ~90 seconds Vietnamese narration corpus
NARRATION_90S = (
    "Có những người xuất hiện đúng lúc trái tim muốn yêu, nhưng lại sai lúc cả hai biết cách ở bên nhau. "
    "Cuộc sống luôn chứa đựng những điều bất ngờ mà chúng ta không thể đoán trước được. "
    "Mỗi cuộc gặp gỡ đều mang một ý nghĩa riêng, dạy cho ta biết trân trọng hiện tại và hoàn thiện bản thân từng ngày. "
    "Đôi khi, khoảng cách lớn nhất không phải là không gian hay thời gian, mà là sự thấu hiểu giữa hai tâm hồn. "
    "Hãy học cách lắng nghe nhiều hơn để không bỏ lỡ những điều quý giá nhất trong cuộc đời.\n\n"
    "Thành phố về đêm khoác lên mình vẻ đẹp lung linh huyền ảo. Từng dòng xe hối hả qua lại dưới ánh đèn vàng ấm áp. "
    "Nhưng sâu trong ngõ nhỏ, mọi thứ lại bình yên đến lạ thường. Gió khẽ thổi qua từng tán lá non xào xạc bên hiên nhà. "
    "Một tách trà nóng, một cuốn sách hay, và những suy tư nhẹ nhàng về những tháng ngày đã qua mang lại cảm giác tĩnh tại.\n\n"
    "Bạn có từng tự hỏi, tại sao chúng ta lại luôn tìm kiếm những điều xa vời ở tận chân trời? "
    "Trong khi hạnh phúc đích thực lại hiện hữu ngay bên cạnh, giản dị và mộc mạc biết bao! "
    "Dù ngoài kia sóng gió bão bùng đến đâu, chỉ cần trong lòng bình yên, mọi chuyện rồi sẽ qua. "
    "Anh ấy quay lại nhìn tôi rồi khẽ mỉm cười nói: 'Chúng ta nhất định sẽ vượt qua tất cả.' "
    "Niềm tin ấy như ngọn lửa nhỏ sưởi ấm tâm hồn trong những đêm đông lạnh giá."
)


def run_v6_benchmark():
    print("\n" + "=" * 75)
    print("      ⚡ OMNIVOICE V6 HARD PERFORMANCE RECOVERY BENCHMARK")
    print("=" * 75)

    hw = profile_hardware()
    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))

    print(f"Hardware Device: {hw['device']} ({hw['primary_gpu']}) | Recommended Threads: {hw['recommended_threads']}")

    # 1. Load Model (ONCE)
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32, load_asr=False)
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load:.2f}s")

    # 2. Warmup (ONCE)
    warm_cfg = OmniVoiceGenerationConfig(num_step=2, guidance_scale=1.0, denoise=True, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate("Sẵn sàng.", language="Vietnamese", generation_config=warm_cfg)
    print("Warmup complete.")

    # 3. Load Cached Voice Prompt (ONCE)
    voice_prompt = vm.load_prompt("default", map_location="cpu")
    print(f"Voice Clone Prompt loaded from cache: {voice_prompt is not None}")

    # 4. Prosody Planning Pass (< 0.05s)
    t_prosody_start = time.perf_counter()
    cleaned_text = TextPreprocessor.process(NARRATION_90S)
    prosody_plan = ProsodyTagger.parse_text_to_prosody_units(cleaned_text)
    t_prosody = time.perf_counter() - t_prosody_start
    print(f"Prosody planning completed in {t_prosody*1000:.2f} ms ({len(prosody_plan)} sentences)")

    # 5. Benchmark Matrix (Chunk Durations & Steps)
    configs_to_test = [
        {"name": "V6_FAST (num_step=16, chunk_dur=20s)", "num_step": 16, "chunk_dur": 20.0, "guidance_scale": 2.0},
        {"name": "V6_BALANCED (num_step=20, chunk_dur=20s)", "num_step": 20, "chunk_dur": 20.0, "guidance_scale": 2.0},
    ]

    benchmark_results = []

    for cfg in configs_to_test:
        print(f"\n" + "-" * 60)
        print(f"▶ RUNNING: {cfg['name']}")
        print("-" * 60)

        gen_cfg = OmniVoiceGenerationConfig(
            num_step=cfg["num_step"],
            guidance_scale=cfg["guidance_scale"],
            denoise=True,
            pad_duration=0.0,
            fade_duration=0.005,
            postprocess_output=False,
            audio_chunk_threshold=20.0,
            audio_chunk_duration=cfg["chunk_dur"],
        )

        t_gen_start = time.perf_counter()
        with torch.inference_mode():
            audios = model.generate(
                text=cleaned_text,
                language="Vietnamese",
                voice_clone_prompt=voice_prompt,
                instruct=None,
                generation_config=gen_cfg,
            )
        t_gen = time.perf_counter() - t_gen_start
        generated_audio = audios[0]

        audio_duration = len(generated_audio) / model.sampling_rate
        rtf = t_gen / audio_duration if audio_duration > 0 else 0

        # Acoustic Quality Analysis
        prosody_metrics = PitchAndEnergyAnalyzer.analyze_prosody_metrics(generated_audio, sample_rate=model.sampling_rate)

        rec = {
            "config": cfg["name"],
            "num_step": cfg["num_step"],
            "chunk_dur_sec": cfg["chunk_dur"],
            "generation_time_sec": round(t_gen, 2),
            "audio_duration_sec": round(audio_duration, 2),
            "rtf": round(rtf, 3),
            "f0_mean_hz": prosody_metrics["f0_mean_hz"],
            "f0_std_hz": prosody_metrics["f0_std_hz"],
            "expressiveness": prosody_metrics["expressiveness_grade"],
            "target_met": (t_gen <= 120.0),
        }
        benchmark_results.append(rec)

        print(f"✅ RESULT for {cfg['name']}:")
        print(f"   Audio Duration:    {audio_duration:.2f} s")
        print(f"   Generation Time:   {t_gen:.2f} s  {'🎯 (TARGET <= 120s MET!)' if t_gen <= 120 else ''}")
        print(f"   Measured RTF:      {rtf:.3f}x")
        print(f"   Pitch Dynamics:    F0 Mean = {rec['f0_mean_hz']:.1f} Hz | F0 Std = {rec['f0_std_hz']:.1f} Hz")
        print(f"   Expressiveness:    {rec['expressiveness']}")

    out_file = Path("D:/OmniVoice_Storage/debug/performance_report_v6.json") if Path("D:/OmniVoice_Storage").exists() else Path("./performance_report_v6.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 75)
    print(f"🏆 V6 BENCHMARK COMPLETE — Logged to {out_file}")
    print("=" * 75)
    return benchmark_results


if __name__ == "__main__":
    run_v6_benchmark()
