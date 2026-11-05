#!/usr/bin/env python3
"""
OmniVoice Comprehensive Performance Benchmark Suite.
Measures cold start, warm generation, stage-by-stage latency breakdown, RTF, CPU/RAM usage.
"""

import os
import sys
import time

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import psutil
import platform
import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.models.omnivoice import VoiceClonePrompt

# Ensure output directory exists
BENCH_OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(BENCH_OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Test Corpora (Realistic Vietnamese Texts)
# ---------------------------------------------------------------------------
TEST_CORPUS = {
    "Test_A_30s": (
        "Xin chào các bạn. Hôm nay tôi rất vui được giới thiệu với các bạn hệ thống "
        "tổng hợp giọng nói trí tuệ nhân tạo OmniVoice. Đây là một công nghệ đột phá, "
        "cho phép nhân bản giọng nói và đọc văn bản tiếng Việt vô cùng tự nhiên và mượt mà."
    ),
    "Test_B_60s": (
        "Trí tuệ nhân tạo đang thay đổi cách chúng ta tương tác với thế giới công nghệ mỗi ngày. "
        "Từ những trợ lý ảo thông minh cho đến các mô hình ngôn ngữ lớn, AI mang lại hiệu suất "
        "làm việc vượt trội cho con người. Trong lĩnh vực xử lý âm thanh và tổng hợp giọng nói, "
        "các mô hình khuếch tán mới nhất đã đạt được độ chân thực gần như không thể phân biệt "
        "với giọng người thật. Điều này mở ra nhiều cơ hội ứng dụng trong giáo dục, giải trí "
        "và sản xuất nội dung tự động với chi phí tối ưu nhất."
    ),
    "Test_C_120s": (
        "Việt Nam là một đất nước có bề dày lịch sử và nền văn hóa đa dạng, phong phú. "
        "Từ những dãy núi hùng vĩ miền Tây Bắc, qua vùng đồng bằng sông Hồng màu mỡ, "
        "dải đất miền Trung kiên cường và duyên hải trù phú, cho đến vùng đồng bằng sông Cửu Long "
        "bạt ngàn phù sa và hoa trái. Mỗi vùng miền đều mang những nét đặc trưng văn hóa riêng biệt, "
        "từ âm nhạc dân gian, ẩm thực truyền thống cho đến giọng điệu mộc mạc, chân tình của con người. "
        "Trong thời đại chuyển đổi số và công nghệ số phát triển mạnh mẽ, việc gìn giữ và phát huy "
        "bản sắc ngôn ngữ dân tộc thông qua các công nghệ giọng nói tiếng Việt là vô cùng cần thiết. "
        "Các hệ thống trí tuệ nhân tạo hiện đại giờ đây có thể học và tái tạo chính xác từng thanh điệu, "
        "nhịp ngắt và cảm xúc trong từng câu chữ, giúp đưa văn hóa và tri thức Việt Nam vươn xa hơn "
        "trên trường quốc tế một cách bền vững."
    ),
}


def get_hardware_info():
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "device": "cpu",
        "cpu_name": platform.processor(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "torch_threads": torch.get_num_threads(),
    }


def run_benchmark(model_path="k2-fsa/OmniVoice", num_steps=32, device="cpu"):
    hw = get_hardware_info()
    print("=" * 60)
    print("      OMNIVOICE COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"OS:            {hw['os']}")
    print(f"CPU:           {hw['cpu_name']}")
    print(f"Cores/Threads: {hw['physical_cores']}C / {hw['logical_cores']}T")
    print(f"RAM:           {hw['total_ram_gb']} GB")
    print(f"PyTorch:       {hw['pytorch']} (device={device})")
    print(f"Torch Threads: {hw['torch_threads']}")
    print("=" * 60)

    # 1. Cold Start: Model Load
    print("\n[1/4] Measuring Model Cold Start Loading...")
    t0 = time.perf_counter()
    mem_before = psutil.Process().memory_info().rss / (1024**2)

    model = OmniVoice.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float32,
        load_asr=False,
    )
    t_load = time.perf_counter() - t0
    mem_after = psutil.Process().memory_info().rss / (1024**2)
    print(f"  -> Model Load Time: {t_load:.2f} s | RAM Added: {mem_after - mem_before:.1f} MB")

    # 2. Warm-up
    print("\n[2/4] Running Warm-up Inference...")
    t0 = time.perf_counter()
    warm_cfg = OmniVoiceGenerationConfig(num_step=4)
    _ = model.generate("Khởi động hệ thống.", language="Vietnamese", generation_config=warm_cfg)
    t_warm = time.perf_counter() - t0
    print(f"  -> Warm-up Time: {t_warm:.2f} s")

    # 3. Standard Tests
    results = {}
    print("\n[3/4] Executing Benchmark Tests (Standard Steps: {})...".format(num_steps))

    for test_name, text in TEST_CORPUS.items():
        print(f"\n--- Running {test_name} (Chars: {len(text)}, Words: {len(text.split())}) ---")
        
        proc = psutil.Process()
        cpu_percentages = []
        
        gen_cfg = OmniVoiceGenerationConfig(
            num_step=num_steps,
            guidance_scale=2.0,
            denoise=True,
            audio_chunk_threshold=25.0,
            audio_chunk_duration=12.0,
        )

        t_start = time.perf_counter()
        
        # Generation
        audios = model.generate(
            text=text,
            language="Vietnamese",
            generation_config=gen_cfg,
        )
        t_gen = time.perf_counter() - t_start
        
        audio = audios[0]
        sr = model.sampling_rate
        audio_dur = len(audio) / sr
        rtf = t_gen / audio_dur if audio_dur > 0 else 0
        
        # Audio export
        out_file = os.path.join(BENCH_OUT_DIR, f"{test_name}_step{num_steps}.wav")
        sf.write(out_file, audio, sr)

        res = {
            "text_chars": len(text),
            "text_words": len(text.split()),
            "gen_time_s": round(t_gen, 2),
            "audio_dur_s": round(audio_dur, 2),
            "rtf": round(rtf, 3),
            "ram_rss_mb": round(proc.memory_info().rss / (1024**2), 1),
        }
        results[test_name] = res

        print(f"  Audio Duration:  {res['audio_dur_s']:.2f} s")
        print(f"  Generation Time: {res['gen_time_s']:.2f} s")
        print(f"  RTF:             {res['rtf']:.3f} ({'PASS <= 0.25' if rtf <= 0.25 else 'SLOW > 0.25'})")
        print(f"  RAM Usage:       {res['ram_rss_mb']:.1f} MB")

    # 4. Summary Table
    print("\n" + "=" * 60)
    print("                 BENCHMARK SUMMARY RESULTS")
    print("=" * 60)
    print(f"Model Load: {t_load:.2f} s | Warm-up: {t_warm:.2f} s")
    print("-" * 60)
    print(f"{'Test':<15} | {'Audio Dur':<10} | {'Gen Time':<10} | {'RTF':<8} | {'Status':<10}")
    print("-" * 60)
    for k, v in results.items():
        status = "PASS" if v["gen_time_s"] <= 30.0 else "FAIL"
        print(f"{k:<15} | {v['audio_dur_s']:>8.1f} s | {v['gen_time_s']:>8.1f} s | {v['rtf']:>8.3f} | {status:<10}")
    print("=" * 60)

    return {
        "hardware": hw,
        "load_time": t_load,
        "warm_time": t_warm,
        "tests": results,
    }


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    run_benchmark(num_steps=steps)
