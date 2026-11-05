#!/usr/bin/env python3
"""
Forensic Audit Script: Measure Native OmniVoice vs Chunking Path.
Runs in a single isolated process with detailed instrumentation.
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
import soundfile as sf
import psutil

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.core.hardware import profile_hardware, apply_runtime_optimizations
from omnivoice.core.storage_manager import StorageManager
from omnivoice.core.voice_manager import VoiceManager
from omnivoice.core.text_preprocessor import TextPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ForensicAudit")

# ~90 seconds benchmark text
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


def run_audit():
    print("=" * 75)
    print("      🔍 OMNIVOICE FORENSIC AUDIT & SPEED MEASUREMENT")
    print("=" * 75)

    hw = profile_hardware()
    apply_runtime_optimizations(hw)
    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))

    print(f"Device: {hw['device']} | Physical Cores: {hw['physical_cores']} | Logical: {hw['logical_cores']} | Threads: {torch.get_num_threads()}")

    # 1. Load Model
    t0 = time.perf_counter()
    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu", dtype=torch.float32, load_asr=False)
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load:.2f}s")

    # 2. Warmup
    warm_cfg = OmniVoiceGenerationConfig(num_step=2, guidance_scale=1.0, denoise=True, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate("Sẵn sàng.", language="Vietnamese", generation_config=warm_cfg)
    print("Warmup complete.")

    # 3. Load Cloned Voice Prompt
    voice_prompt = vm.load_prompt("default", map_location="cpu")
    print(f"Loaded voice prompt: {voice_prompt is not None}")

    # 4. Clean text
    cleaned_text = TextPreprocessor.process(NARRATION_90S)
    print(f"Cleaned Text: {len(cleaned_text)} characters")

    # 5. Test Native OmniVoice generate with num_step=16
    gen_cfg = OmniVoiceGenerationConfig(
        num_step=16,
        guidance_scale=2.0,
        denoise=True,
        pad_duration=0.0,
        fade_duration=0.005,
        postprocess_output=False,
        audio_chunk_threshold=25.0,
        audio_chunk_duration=20.0,
    )

    print("\n▶ Running Native OmniVoice generate (num_step=16, audio_chunk_duration=20s)...")
    proc = psutil.Process()
    ram_before = proc.memory_info().rss / (1024 * 1024)

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
    ram_after = proc.memory_info().rss / (1024 * 1024)

    generated_audio = audios[0]
    audio_duration = len(generated_audio) / model.sampling_rate
    rtf = t_gen / audio_duration if audio_duration > 0 else 0

    print("\n" + "=" * 75)
    print(f"📊 FORENSIC BENCHMARK RESULT:")
    print(f"   Audio Duration:       {audio_duration:.2f} s")
    print(f"   Total Generation Time:{t_gen:.2f} s")
    print(f"   RTF:                  {rtf:.3f}x")
    print(f"   RAM Used:             {ram_after:.1f} MB (delta: {ram_after - ram_before:.1f} MB)")
    print(f"   Target <= 120s:       {'✅ PASS' if t_gen <= 120 else '❌ FAIL'}")
    print("=" * 75)

    out_path = Path("D:/OmniVoice_Storage/outputs/forensic_test.wav") if Path("D:/OmniVoice_Storage").exists() else Path("./forensic_test.wav")
    sf.write(str(out_path), generated_audio, model.sampling_rate)
    print(f"Output saved to: {out_path}")


if __name__ == "__main__":
    run_audit()
