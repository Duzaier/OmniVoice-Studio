"""
OMNIVOICE V2.12 — SEQUENCE LENGTH & CHUNKING FORENSIC BENCHMARK
Forensic investigation of sequence length, token scaling, padding,
and native chunking across text length regimes (Short, Medium, Long, Multi-Sentence).
"""

import gc
import json
import math
import os
import sys
import time
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
from omnivoice.utils.text import chunk_text_punctuation

NUM_THREADS = 6

TEXT_REGIMES = {
    "short": "Chào bạn, hôm nay bạn thấy thế nào?",
    "medium": "Hôm nay tôi xin trình bày báo cáo về tiến độ phát triển hệ thống trí tuệ nhân tạo.",
    "long": "Trí tuệ nhân tạo đang thay đổi thế giới một cách nhanh chóng và sâu sắc. Từ các mô hình ngôn ngữ lớn đến tổng hợp giọng nói tự nhiên, công nghệ này mở ra những tiềm năng to lớn trong giáo dục, y tế và tự động hóa công việc hàng ngày.",
    "multi_sentence": "Hệ thống trí tuệ nhân tạo thế hệ mới đang được ứng dụng rộng rãi trong nhiều lĩnh vực quan trọng. Các nhà nghiên cứu liên tục cải tiến kiến trúc mô hình để tối ưu hóa hiệu năng và độ chính xác. Bên cạnh đó, việc tối ưu hóa suy luận trên các thiết bị phần cứng phổ thông như máy tính cá nhân giúp công nghệ này tiếp cận được với đông đảo người dùng hơn bao giờ hết. Chúng tôi cam kết mang lại trải nghiệm giọng nói tự nhiên, mượt mà và trung thực nhất.",
}


def setup_env():
    torch.set_num_threads(NUM_THREADS)
    try:
        torch.set_num_interop_threads(NUM_THREADS)
    except RuntimeError:
        pass
    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)


def load_model_and_prompt():
    print("Loading OmniVoice model for V2.12 sequence & chunking forensic...", flush=True)
    storage = StorageManager()
    vm = VoiceManager(str(storage.voices_dir))

    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cpu",
        dtype=torch.float32,
        load_asr=False,
    )
    model.eval()

    prompt = vm.load_prompt("default", map_location="cpu")
    print("VoiceClonePrompt 'default' loaded successfully.", flush=True)

    return model, prompt


def run_baseline_reproduction(model, prompt, warmup=2, runs=5):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 1] BASELINE REPRODUCTION & SEQUENCE METRICS", flush=True)
    print("=" * 80, flush=True)

    setup_env()
    text = TextPreprocessor.process(TEXT_REGIMES["medium"])
    gen_config = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        postprocess_output=True,
    )

    # Warmup
    for _ in range(warmup):
        with torch.inference_mode():
            _ = model.generate(text, language="Vietnamese", voice_clone_prompt=prompt, generation_config=gen_config)

    wall_times = []
    diff_times = []
    voc_times = []
    durations = []
    rtfs = []
    rms_list = []
    peak_list = []
    last_audio = None

    # Inspect token preparation metrics
    with torch.inference_mode():
        task = model._preprocess_all(
            text=text,
            language="Vietnamese",
            voice_clone_prompt=prompt,
            preprocess_prompt=gen_config.preprocess_prompt,
        )
        inputs_list = [
            model._prepare_inference_inputs(
                task.texts[i],
                task.target_lens[i],
                task.ref_texts[i],
                task.ref_audio_tokens[i],
                task.langs[i],
                task.instructs[i],
                gen_config.denoise,
            )
            for i in range(task.batch_size)
        ]
        seq_len = inputs_list[0]["input_ids"].size(2)
        target_tokens = task.target_lens[0]
        prefix_tokens = seq_len - target_tokens
        char_count = len(text)
        text_tokens_count = model.text_tokenizer(text, return_tensors="pt").input_ids.size(1)

    for r in range(runs):
        gc.collect()
        t0_wall = time.perf_counter()

        with torch.inference_mode():
            full_task = model._preprocess_all(
                text=text,
                language="Vietnamese",
                voice_clone_prompt=prompt,
                preprocess_prompt=gen_config.preprocess_prompt,
            )

            t0_diff = time.perf_counter()
            gen_tokens = model._generate_iterative(full_task, gen_config)
            t_diff = time.perf_counter() - t0_diff

            t0_voc = time.perf_counter()
            audio = model._decode_and_post_process(
                gen_tokens[0], full_task.ref_rms[0], gen_config
            )
            t_voc = time.perf_counter() - t0_voc

        t_wall = time.perf_counter() - t0_wall
        audio_dur = len(audio) / model.sampling_rate
        rtf = t_wall / audio_dur
        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))

        wall_times.append(t_wall)
        diff_times.append(t_diff)
        voc_times.append(t_voc)
        durations.append(audio_dur)
        rtfs.append(rtf)
        rms_list.append(rms)
        peak_list.append(peak)
        last_audio = audio

        print(
            f"  Run {r+1}: Wall={t_wall:5.2f}s | Diff={t_diff:5.2f}s | Voc={t_voc:4.2f}s | "
            f"Audio={audio_dur:5.2f}s | RTF={rtf:5.2f}x | RMS={rms:.4f}",
            flush=True,
        )

    res = {
        "text": text,
        "char_count": char_count,
        "text_tokens_count": text_tokens_count,
        "prefix_tokens": prefix_tokens,
        "target_tokens": target_tokens,
        "total_seq_len": seq_len,
        "padding_in_batch_1": 0,
        "wall_mean_s": float(np.mean(wall_times)),
        "wall_std_s": float(np.std(wall_times)),
        "diff_mean_s": float(np.mean(diff_times)),
        "diff_std_s": float(np.std(diff_times)),
        "voc_mean_s": float(np.mean(voc_times)),
        "voc_std_s": float(np.std(voc_times)),
        "audio_dur_mean_s": float(np.mean(durations)),
        "rtf_mean": float(np.mean(rtfs)),
        "rms_mean": float(np.mean(rms_list)),
        "peak_mean": float(np.mean(peak_list)),
    }

    print(f"\nBaseline Sequence Metrics:", flush=True)
    print(f"  • Input characters  : {char_count} chars", flush=True)
    print(f"  • Text tokens       : {text_tokens_count} tokens", flush=True)
    print(f"  • Prefix tokens     : {prefix_tokens} tokens (style + text + ref audio)", flush=True)
    print(f"  • Target audio codes: {target_tokens} frames ({target_tokens * 8} discrete codes)", flush=True)
    print(f"  • Total sequence (L): {seq_len} tokens (Zero padding in Batch-1)", flush=True)
    print(f"  • Mean Latency      : Wall={res['wall_mean_s']:.2f}s (Diff={res['diff_mean_s']:.2f}s), RTF={res['rtf_mean']:.2f}x", flush=True)

    return res, last_audio


def benchmark_sequence_length_scaling(model, prompt):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 2] SEQUENCE LENGTH LATENCY SCALING CURVE", flush=True)
    print("=" * 80, flush=True)

    seq_lengths = [60, 90, 120, 150, 200, 250, 300, 410]
    scaling_curve = {}

    print("Measuring 1-step Transformer Forward Latency across Sequence Lengths L...", flush=True)
    for L in seq_lengths:
        dummy_inputs = torch.randn(1, L, 1024)
        dummy_mask = torch.ones(1, 1, L, L, dtype=torch.bool)

        # Warmup
        for _ in range(5):
            with torch.inference_mode():
                _ = model.llm(inputs_embeds=dummy_inputs, attention_mask=dummy_mask, return_dict=True)

        timings = []
        for _ in range(10):
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model.llm(inputs_embeds=dummy_inputs, attention_mask=dummy_mask, return_dict=True)
            t_el = (time.perf_counter() - t0) * 1000.0
            timings.append(t_el)

        mean_ms = float(np.mean(timings))
        std_ms = float(np.std(timings))
        scaling_curve[L] = {
            "seq_len": L,
            "1step_tf_mean_ms": mean_ms,
            "1step_tf_std_ms": std_ms,
            "4step_tf_projected_s": (mean_ms * 4) / 1000.0,
            "latency_per_token_ms": mean_ms / L,
        }

        print(f"  • L = {L:3d} tokens | 1-Step TF: {mean_ms:6.1f} ms | 4-Step TF: {mean_ms*4/1000:5.2f} s | Cost/Token: {mean_ms/L:5.2f} ms", flush=True)

    return scaling_curve


def benchmark_text_length_regimes_and_chunking(model, prompt):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 3] TEXT REGIMES & CHUNKING BEHAVIOR BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    regime_results = {}

    for regime_name, raw_text in TEXT_REGIMES.items():
        processed_text = TextPreprocessor.process(raw_text)
        print(f"\nEvaluating Regime: {regime_name.upper()} ({len(processed_text)} chars)...", flush=True)

        # 1. Unchunked / Default Native (threshold=20.0s)
        cfg_default = OmniVoiceGenerationConfig(
            num_step=4,
            guidance_scale=0.0,
            audio_chunk_threshold=20.0,
            audio_chunk_duration=10.0,
            postprocess_output=True,
        )

        with torch.inference_mode():
            task_def = model._preprocess_all(
                text=processed_text,
                language="Vietnamese",
                voice_clone_prompt=prompt,
            )

        # Measure 3 runs
        wall_times = []
        durations = []
        rtfs = []

        for r in range(3):
            gc.collect()
            t0 = time.perf_counter()
            with torch.inference_mode():
                audios = model.generate(
                    text=processed_text,
                    language="Vietnamese",
                    voice_clone_prompt=prompt,
                    generation_config=cfg_default,
                )
            t_wall = time.perf_counter() - t0
            audio_dur = len(audios[0]) / model.sampling_rate
            rtf = t_wall / audio_dur

            wall_times.append(t_wall)
            durations.append(audio_dur)
            rtfs.append(rtf)
            print(f"  [Default Config] Run {r+1}: Wall={t_wall:5.2f}s | Audio={audio_dur:5.2f}s | RTF={rtf:5.2f}x", flush=True)

        # 2. Aggressive Chunking (threshold=6.0s, duration=5.0s) for long/multi-sentence
        aggressive_res = None
        if regime_name in ["long", "multi_sentence"]:
            cfg_aggr = OmniVoiceGenerationConfig(
                num_step=4,
                guidance_scale=0.0,
                audio_chunk_threshold=6.0,
                audio_chunk_duration=5.0,
                postprocess_output=True,
            )
            aggr_wall_times = []
            aggr_durations = []
            aggr_rtfs = []

            for r in range(3):
                gc.collect()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    audios = model.generate(
                        text=processed_text,
                        language="Vietnamese",
                        voice_clone_prompt=prompt,
                        generation_config=cfg_aggr,
                    )
                t_wall = time.perf_counter() - t0
                audio_dur = len(audios[0]) / model.sampling_rate
                rtf = t_wall / audio_dur
                aggr_wall_times.append(t_wall)
                aggr_durations.append(audio_dur)
                aggr_rtfs.append(rtf)
                print(f"  [Aggressive Chunking] Run {r+1}: Wall={t_wall:5.2f}s | Audio={audio_dur:5.2f}s | RTF={rtf:5.2f}x", flush=True)

            aggressive_res = {
                "wall_mean_s": float(np.mean(aggr_wall_times)),
                "audio_dur_mean_s": float(np.mean(aggr_durations)),
                "rtf_mean": float(np.mean(aggr_rtfs)),
            }

        regime_results[regime_name] = {
            "char_count": len(processed_text),
            "estimated_target_tokens": task_def.target_lens[0],
            "default_chunking": {
                "wall_mean_s": float(np.mean(wall_times)),
                "audio_dur_mean_s": float(np.mean(durations)),
                "rtf_mean": float(np.mean(rtfs)),
            },
            "aggressive_chunking": aggressive_res,
        }

    return regime_results


def benchmark_target_token_budgeting(model, prompt):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 4] TARGET TOKEN FRAME BUDGETING EXPERIMENT", flush=True)
    print("=" * 80, flush=True)

    text = TextPreprocessor.process(TEXT_REGIMES["medium"])
    target_budget_candidates = [118, 105, 95, 85, 75]
    budget_results = {}

    gen_config = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        postprocess_output=True,
    )

    for budget in target_budget_candidates:
        print(f"\nEvaluating Target Token Budget = {budget} frames (vs 118 baseline)...", flush=True)

        wall_times = []
        diff_times = []
        durations = []
        rtfs = []
        last_audio = None

        for r in range(3):
            gc.collect()
            t0 = time.perf_counter()
            with torch.inference_mode():
                full_task = model._preprocess_all(
                    text=text,
                    language="Vietnamese",
                    voice_clone_prompt=prompt,
                    preprocess_prompt=gen_config.preprocess_prompt,
                )
                full_task.target_lens[0] = budget

                t0_diff = time.perf_counter()
                gen_tokens = model._generate_iterative(full_task, gen_config)
                t_diff = time.perf_counter() - t0_diff

                audio = model._decode_and_post_process(
                    gen_tokens[0], full_task.ref_rms[0], gen_config
                )

            t_wall = time.perf_counter() - t0
            audio_dur = len(audio) / model.sampling_rate
            rtf = t_wall / audio_dur

            wall_times.append(t_wall)
            diff_times.append(t_diff)
            durations.append(audio_dur)
            rtfs.append(rtf)
            last_audio = audio

            print(f"  Run {r+1}: Wall={t_wall:5.2f}s | Diff={t_diff:5.2f}s | Audio={audio_dur:5.2f}s | RTF={rtf:5.2f}x", flush=True)

        audio_save_path = f"benchmarks/audio_budget_{budget}.wav"
        sf.write(audio_save_path, last_audio, model.sampling_rate)

        speedup_vs_118 = budget_results[118]["wall_mean_s"] / np.mean(wall_times) if 118 in budget_results else 1.0

        budget_results[budget] = {
            "target_frames": budget,
            "wall_mean_s": float(np.mean(wall_times)),
            "diff_mean_s": float(np.mean(diff_times)),
            "audio_dur_mean_s": float(np.mean(durations)),
            "rtf_mean": float(np.mean(rtfs)),
            "speedup_vs_baseline": float(speedup_vs_118),
            "audio_path": audio_save_path,
        }

    return budget_results


def main():
    setup_env()
    model, prompt = load_model_and_prompt()

    baseline_res, sample_audio = run_baseline_reproduction(model, prompt)
    seq_scaling_res = benchmark_sequence_length_scaling(model, prompt)
    regimes_res = benchmark_text_length_regimes_and_chunking(model, prompt)
    budget_res = benchmark_target_token_budgeting(model, prompt)

    os.makedirs("benchmarks", exist_ok=True)
    audio_path = "benchmarks/v2_12_baseline.wav"
    sf.write(audio_path, sample_audio, model.sampling_rate)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "golden_baseline": baseline_res,
        "sequence_scaling_curve": seq_scaling_res,
        "text_regimes_and_chunking": regimes_res,
        "target_token_budgeting": budget_res,
    }

    results_path = "benchmarks/v2_12_sequence_chunking_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] V2.12 Sequence Length & Chunking Forensic completed. Results saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
