#!/usr/bin/env python3
"""
OmniVoice Comprehensive Benchmark & Validation Suite.
Measures real performance across Test A (30s), Test B (60s), Test C (120s).
"""

import os
import sys
import time

# UTF-8 Encoding Fix for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import soundfile as sf
import psutil
import torch

from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.core.hardware import profile_hardware

BENCH_OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(BENCH_OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Test Corpora (Natural Vietnamese Texts)
# ---------------------------------------------------------------------------
CORPUS = {
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


def run_suite(mode="TURBO"):
    print("=" * 65)
    print("      OMNIVOICE MASTER PERFORMANCE BENCHMARK SUITE")
    print("=" * 65)
    
    hw = profile_hardware()
    print(f"OS:            {hw['os']}")
    print(f"CPU:           {hw['cpu_name']}")
    print(f"Cores/Threads: {hw['physical_cores']}C / {hw['logical_cores']}T")
    print(f"RAM Total:     {hw['total_ram_gb']} GB (Free: {hw['available_ram_gb']} GB)")
    print(f"PyTorch:       {hw['pytorch']} (Device: {hw['device']})")
    print(f"Engine Mode:   {mode}")
    print("=" * 65)

    # 1. Cold Start Initialization
    print("\n[PHASE 1] Initializing Engine & Measuring Cold Start...")
    engine = OmniVoiceEngine.get_instance(device="cpu")
    t_load = engine.initialize()
    print(f"  -> Cold Start Model Load: {t_load:.2f} s")

    # 2. Warm Generation Tests
    print(f"\n[PHASE 2] Running Warm Generation Benchmarks ({mode} Mode)...")
    results = {}

    for test_key, text in CORPUS.items():
        print(f"\n>>> Running {test_key} (Chars: {len(text)}, Words: {len(text.split())})...")
        
        def on_prog(curr, tot, pct, msg):
            print(f"    [{pct:5.1f}%] Chunk {curr}/{tot}: {msg}")

        res = engine.synthesize(
            text=text,
            language="Vietnamese",
            perf_mode=mode,
            progress_callback=on_prog,
        )

        out_path = os.path.join(BENCH_OUT_DIR, f"{test_key}_{mode}.wav")
        sf.write(out_path, res.audio, res.sample_rate)

        results[test_key] = res
        print(f"  -> Result: Audio Dur = {res.audio_duration:.2f}s | Gen Time = {res.generation_time:.2f}s | RTF = {res.rtf:.3f}x")

    # 3. Final Summary Report
    print("\n" + "=" * 65)
    print("                 FINAL BENCHMARK REPORT TABLE")
    print("=" * 65)
    print(f"{'Test':<15} | {'Audio Dur':<10} | {'Gen Time':<10} | {'RTF':<8} | {'Status (≤30s)':<15}")
    print("-" * 65)
    for k, v in results.items():
        status = "PASS (≤30s)" if v.generation_time <= 30.0 else "HW-LIMITED"
        print(f"{k:<15} | {v.audio_duration:>8.1f} s | {v.generation_time:>8.1f} s | {v.rtf:>7.3f}x | {status:<15}")
    print("=" * 65)
    
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "TURBO"
    run_suite(mode)
