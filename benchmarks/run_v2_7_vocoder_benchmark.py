"""
OmniVoice V2.7 - Vocoder Forensic Benchmark
Comprehensive profiling and end-to-end evaluation of HiggsAudioV2Tokenizer / DacDecoder
"""

import os
import sys
import time
import json
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def benchmark_op(func, warmup=5, iters=20):
    for _ in range(warmup):
        _ = func()
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = func()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # ms
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "std_ms": float(np.std(latencies)),
        "runs": len(latencies),
        "latencies": latencies
    }

def run_vocoder_forensic_suite():
    print("=" * 80, flush=True)
    print("OMNIVOICE V2.7 — VOCODER FORENSIC BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))
    
    print("Loading OmniVoice model...", flush=True)
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    vp = vm.load_prompt("default", map_location="cpu")
    clean_text = TextPreprocessor.process(TEXT_220)

    # 1. Capture Real Audio Tokens from 1 generation pass
    print("\n[STEP 1] CAPTURING REAL AUDIO CODES FROM PIPELINE...", flush=True)
    captured_codes = []
    orig_decode = model.audio_tokenizer.decode
    def capture_decode_hook(audio_codes, *args, **kwargs):
        captured_codes.append(audio_codes.clone())
        return orig_decode(audio_codes, *args, **kwargs)

    model.audio_tokenizer.decode = capture_decode_hook
    cfg_1step = OmniVoiceGenerationConfig(num_step=1, guidance_scale=0.0, denoise=False, postprocess_output=False)
    with torch.inference_mode():
        _ = model.generate(clean_text, language="Vietnamese", voice_clone_prompt=vp, generation_config=cfg_1step)
    model.audio_tokenizer.decode = orig_decode

    assert len(captured_codes) > 0, "Failed to capture audio codes!"
    real_codes = captured_codes[0]
    print(f"  • Real Audio Codes Shape: {list(real_codes.shape)}", flush=True)
    print(f"  • Dtype: {real_codes.dtype}, Stride: {real_codes.stride()}, Contiguous: {real_codes.is_contiguous()}", flush=True)
    num_codebooks = real_codes.shape[1]
    seq_tokens = real_codes.shape[2]
    expected_samples = seq_tokens * 1920
    print(f"  • Codebooks: {num_codebooks}, Sequence Tokens: {seq_tokens} -> Waveform Samples: {expected_samples} ({expected_samples/24000:.2f}s audio)", flush=True)

    # 2. Benchmark Baseline Vocoder across threads [1, 2, 4, 6, 8]
    print("\n[STEP 2] BASELINE VOCODER THREAD SCALING BENCHMARK [1, 2, 4, 6, 8]...", flush=True)
    threads = [1, 2, 4, 6, 8]
    thread_results = {}
    for t in threads:
        torch.set_num_threads(t)
        with torch.inference_mode():
            res = benchmark_op(lambda: model.audio_tokenizer.decode(real_codes), warmup=5, iters=20)
        thread_results[t] = res
        print(f"  • Threads = {t:2d} | Mean = {res['mean_ms']:7.2f} ms | Median = {res['median_ms']:7.2f} ms | Min = {res['min_ms']:7.2f} ms | Std = {res['std_ms']:6.2f} ms", flush=True)

    # 3. Profiling Operator Breakdown inside audio_tokenizer.decode
    print("\n[STEP 3] PYTORCH PROFILER OPERATOR BREAKDOWN (6 THREADS)...", flush=True)
    torch.set_num_threads(6)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        with torch.inference_mode():
            for _ in range(5):
                _ = model.audio_tokenizer.decode(real_codes)

    key_avgs = prof.key_averages(group_by_input_shape=False)
    print("\nTOP 15 CPU OPERATORS IN VOCODER:")
    print(key_avgs.table(sort_by="self_cpu_time_total", row_limit=15))

    total_self_cpu = sum(item.self_cpu_time_total for item in key_avgs)
    sorted_ops = sorted(key_avgs, key=lambda x: x.self_cpu_time_total, reverse=True)
    breakdown_list = []
    print("\nStructured Top 10 Operators by Self CPU Time:")
    for idx, item in enumerate(sorted_ops[:10], 1):
        pct = (item.self_cpu_time_total / total_self_cpu) * 100.0 if total_self_cpu > 0 else 0
        ms_per_call = (item.self_cpu_time_total / item.count / 1000.0) if item.count > 0 else 0
        total_ms = item.self_cpu_time_total / 1000.0 / 5.0 # per decode call
        print(f"  {idx:2d}. {item.key:<40} | Self: {total_ms:6.2f} ms/call | {pct:5.2f}% | Count: {item.count//5:4d} | Avg: {ms_per_call:6.3f} ms")
        breakdown_list.append({
            "name": item.key,
            "self_ms_per_call": total_ms,
            "pct": pct,
            "count_per_call": item.count // 5,
            "avg_call_ms": ms_per_call
        })

    # 4. Detailed Component Breakdown of DacDecoder Submodules
    print("\n[STEP 4] SUBMODULE TIMING BREAKDOWN (6 THREADS)...", flush=True)
    tok = model.audio_tokenizer
    with torch.inference_mode():
        # Step A: RVQ Quantizer Decode
        codes_t = real_codes.transpose(0, 1)
        res_quant = benchmark_op(lambda: tok.quantizer.decode(codes_t), warmup=5, iters=20)
        quantized = tok.quantizer.decode(codes_t)
        
        # Step B: Linear fc2 projection
        quantized_t = quantized.transpose(1, 2)
        res_fc2 = benchmark_op(lambda: tok.fc2(quantized_t), warmup=5, iters=20)
        quantized_acoustic = tok.fc2(quantized_t).transpose(1, 2)
        
        # Step C: DacDecoder
        dac = tok.acoustic_decoder
        res_dac = benchmark_op(lambda: dac(quantized_acoustic), warmup=5, iters=20)

        # Inside DacDecoder blocks
        res_conv1 = benchmark_op(lambda: dac.conv1(quantized_acoustic), warmup=5, iters=20)
        h = dac.conv1(quantized_acoustic)

        block_timings = []
        for b_idx, block in enumerate(dac.block):
            res_b = benchmark_op(lambda blk=block, inp=h: blk(inp), warmup=5, iters=20)
            block_timings.append(res_b)
            h = block(h)

        res_final = benchmark_op(lambda inp=h: dac.conv2(dac.snake1(inp)), warmup=5, iters=20)

    print(f"  • RVQ Quantizer Decode : {res_quant['mean_ms']:6.2f} ms")
    print(f"  • FC2 Projection (Linear): {res_fc2['mean_ms']:6.2f} ms")
    print(f"  • DacDecoder Total      : {res_dac['mean_ms']:6.2f} ms")
    print(f"    ├─ Initial Conv1      : {res_conv1['mean_ms']:6.2f} ms")
    for b_idx, bt in enumerate(block_timings):
        print(f"    ├─ DacBlock {b_idx} (Upsample): {bt['mean_ms']:6.2f} ms")
    print(f"    └─ Final Conv2/Snake  : {res_final['mean_ms']:6.2f} ms")

    # 5. Full End-to-End A/B Repeatability Benchmark (5 Baseline vs 5 Candidate)
    print("\n[STEP 5] FULL COMPLETE GENERATION A/B BENCHMARK (V2.1, 4 STEPS, 6 THREADS)...", flush=True)
    cfg_turbo_4 = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        denoise=True,
        pad_duration=0.1,
        fade_duration=0.1,
        postprocess_output=True,
    )

    # Alternate 5 runs to measure end-to-end impact & vocoder vs diffusion split
    runs_e2e = []
    for r in range(1, 6):
        step_times = []
        orig_fwd = model.forward
        def step_tracker(*args, **kwargs):
            t_s0 = time.perf_counter()
            out = orig_fwd(*args, **kwargs)
            step_times.append(time.perf_counter() - t_s0)
            return out
        model.forward = step_tracker

        t_start = time.perf_counter()
        with torch.inference_mode():
            res_audios = model.generate(
                text=clean_text,
                language="Vietnamese",
                voice_clone_prompt=vp,
                generation_config=cfg_turbo_4,
            )
        t_total = time.perf_counter() - t_start
        model.forward = orig_fwd

        diff_time = sum(step_times)
        vocoder_time = max(0.0, t_total - diff_time)
        audio = res_audios[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        audio_dur = len(audio) / model.sampling_rate
        rtf = t_total / audio_dur if audio_dur > 0 else 0

        runs_e2e.append({
            "run": r,
            "total_s": t_total,
            "diffusion_s": diff_time,
            "vocoder_s": vocoder_time,
            "audio_dur_s": audio_dur,
            "rtf": rtf,
            "vocoder_pct": (vocoder_time / t_total) * 100.0
        })
        print(f"  Run {r}: Total = {t_total:5.2f}s | Diffusion = {diff_time:5.2f}s ({diff_time/t_total*100:.1f}%) | Vocoder = {vocoder_time:4.2f}s ({vocoder_time/t_total*100:.1f}%) | Audio = {audio_dur:5.2f}s | RTF = {rtf:5.2f}x", flush=True)

    # Summary
    mean_total = float(np.mean([x["total_s"] for x in runs_e2e]))
    mean_diff = float(np.mean([x["diffusion_s"] for x in runs_e2e]))
    mean_voc = float(np.mean([x["vocoder_s"] for x in runs_e2e]))
    mean_rtf = float(np.mean([x["rtf"] for x in runs_e2e]))
    mean_voc_pct = float(np.mean([x["vocoder_pct"] for x in runs_e2e]))

    print("\n" + "=" * 80, flush=True)
    print("END-TO-END GENERATION PROFILE SUMMARY (6 THREADS, 4 STEPS):", flush=True)
    print(f"  • Total Generation Time : {mean_total:.2f} s (Std: {np.std([x['total_s'] for x in runs_e2e]):.2f} s)", flush=True)
    print(f"  • Diffusion Loop Time   : {mean_diff:.2f} s ({mean_diff/mean_total*100:.1f}%)", flush=True)
    print(f"  • Vocoder Time          : {mean_voc:.2f} s ({mean_voc_pct:.1f}%)", flush=True)
    print(f"  • Real-Time Factor (RTF): {mean_rtf:.2f} x", flush=True)
    print("=" * 80, flush=True)

    # Save complete results
    report_data = {
        "real_codes_shape": list(real_codes.shape),
        "thread_scaling": thread_results,
        "operator_breakdown": breakdown_list,
        "submodule_timings": {
            "quantizer_ms": res_quant["mean_ms"],
            "fc2_ms": res_fc2["mean_ms"],
            "dac_total_ms": res_dac["mean_ms"],
            "dac_conv1_ms": res_conv1["mean_ms"],
            "block_ms": [b["mean_ms"] for b in block_timings],
            "dac_final_ms": res_final["mean_ms"],
        },
        "e2e_runs": runs_e2e,
        "summary": {
            "mean_total_s": mean_total,
            "mean_diffusion_s": mean_diff,
            "mean_vocoder_s": mean_voc,
            "mean_vocoder_pct": mean_voc_pct,
            "mean_rtf": mean_rtf
        }
    }

    with open("benchmarks/v2_7_vocoder_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print("Results saved to benchmarks/v2_7_vocoder_benchmark_results.json", flush=True)

if __name__ == "__main__":
    run_vocoder_forensic_suite()
