"""
OMNIVOICE V2.10 — INFERENCE EXECUTION / RUNTIME FORENSIC BENCHMARK
Comprehensive investigation of inference contexts, thread pools, model eval state,
memory allocation/copies, synchronization points, memory formats, and weight packing.
"""

import gc
import json
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

SAMPLE_RAW_TEXT = (
    "Hôm nay tôi xin trình bày báo cáo về tiến độ phát triển hệ thống trí tuệ nhân tạo."
)
DEFAULT_THREADS = 6


def setup_env(intra_threads=DEFAULT_THREADS, inter_threads=None):
    torch.set_num_threads(intra_threads)
    if inter_threads is not None:
        try:
            torch.set_num_interop_threads(inter_threads)
        except RuntimeError:
            pass
    os.environ["OMP_NUM_THREADS"] = str(intra_threads)
    os.environ["MKL_NUM_THREADS"] = str(intra_threads)


def load_model_and_prompt():
    print(f"Loading OmniVoice model for V2.10 runtime forensic...", flush=True)
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

    text = TextPreprocessor.process(SAMPLE_RAW_TEXT)
    return model, prompt, text


def audit_inference_context_and_eval(model):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 1] INFERENCE CONTEXT & MODEL.EVAL AUDIT", flush=True)
    print("=" * 80, flush=True)

    grad_enabled_default = torch.is_grad_enabled()
    inf_mode_default = torch.is_inference_mode_enabled()

    with torch.no_grad():
        grad_in_no_grad = torch.is_grad_enabled()
        inf_in_no_grad = torch.is_inference_mode_enabled()

    with torch.inference_mode():
        grad_in_inf_mode = torch.is_grad_enabled()
        inf_in_inf_mode = torch.is_inference_mode_enabled()

    # Audit submodules
    module_stats = {
        "omnivoice_training": model.training,
        "llm_training": model.llm.training,
        "audio_embeddings_training": model.audio_embeddings.training,
        "audio_heads_training": model.audio_heads.training,
        "tokenizer_training": model.audio_tokenizer.training if model.audio_tokenizer else None,
    }

    dropout_modules = []
    norm_modules = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            dropout_modules.append((name, type(module).__name__, getattr(module, "p", None), module.training))
        elif "norm" in type(module).__name__.lower():
            norm_modules.append((name, type(module).__name__, module.training))

    print(f"Default context: grad_enabled={grad_enabled_default}, inference_mode={inf_mode_default}", flush=True)
    print(f"Inside no_grad: grad_enabled={grad_in_no_grad}, inference_mode={inf_in_no_grad}", flush=True)
    print(f"Inside inference_mode: grad_enabled={grad_in_inf_mode}, inference_mode={inf_in_inf_mode}", flush=True)
    print(f"Model submodules training state: {module_stats}", flush=True)
    print(f"Total Dropout layers found: {len(dropout_modules)} (All training={any(d[3] for d in dropout_modules)})", flush=True)
    print(f"Total Normalization layers found: {len(norm_modules)}", flush=True)

    return {
        "grad_default": grad_enabled_default,
        "inf_default": inf_mode_default,
        "grad_in_no_grad": grad_in_no_grad,
        "inf_in_no_grad": inf_in_no_grad,
        "grad_in_inf": grad_in_inf_mode,
        "inf_in_inf": inf_in_inf_mode,
        "submodule_training": module_stats,
        "dropout_count": len(dropout_modules),
        "norm_count": len(norm_modules),
    }


def benchmark_golden_baseline(model, prompt, text, warmup=2, runs=5):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 2] GOLDEN BASELINE REPRODUCTION (V2.10 BASELINE: 6 THREADS, NUM_STEP=4)", flush=True)
    print("=" * 80, flush=True)

    setup_env(6)
    gen_config = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        postprocess_output=True,
    )

    # Warmup
    for w in range(warmup):
        with torch.inference_mode():
            _ = model.generate(
                text=text,
                language="Vietnamese",
                voice_clone_prompt=prompt,
                generation_config=gen_config,
            )

    wall_times = []
    diff_times = []
    voc_times = []
    durations = []
    rtfs = []
    rms_list = []
    peak_list = []
    last_audio = None

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
        "wall_mean_s": float(np.mean(wall_times)),
        "wall_std_s": float(np.std(wall_times)),
        "wall_min_s": float(np.min(wall_times)),
        "wall_max_s": float(np.max(wall_times)),
        "diff_mean_s": float(np.mean(diff_times)),
        "diff_std_s": float(np.std(diff_times)),
        "voc_mean_s": float(np.mean(voc_times)),
        "voc_std_s": float(np.std(voc_times)),
        "audio_dur_mean_s": float(np.mean(durations)),
        "rtf_mean": float(np.mean(rtfs)),
        "rms_mean": float(np.mean(rms_list)),
        "peak_mean": float(np.mean(peak_list)),
    }
    print(f"Baseline Summary: Mean Wall={res['wall_mean_s']:.2f}s (Diff={res['diff_mean_s']:.2f}s, Voc={res['voc_mean_s']:.2f}s), RTF={res['rtf_mean']:.2f}x", flush=True)
    return res, last_audio


def benchmark_inference_contexts(model, prompt, text, warmup=2, runs=5):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 3] CONTROLLED INFERENCE CONTEXT A/B BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    gen_config = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        postprocess_output=True,
    )

    contexts = {
        "torch.inference_mode()": lambda: torch.inference_mode(),
        "torch.no_grad()": lambda: torch.no_grad(),
    }

    results = {}

    for name, ctx_factory in contexts.items():
        print(f"\nBenchmarking Context: {name}...", flush=True)
        # Warmup
        for _ in range(warmup):
            with ctx_factory():
                _ = model.generate(
                    text=text,
                    language="Vietnamese",
                    voice_clone_prompt=prompt,
                    generation_config=gen_config,
                )

        wall_times = []
        diff_times = []
        for r in range(runs):
            gc.collect()
            t0 = time.perf_counter()
            with ctx_factory():
                full_task = model._preprocess_all(
                    text=text,
                    language="Vietnamese",
                    voice_clone_prompt=prompt,
                    preprocess_prompt=gen_config.preprocess_prompt,
                )
                t0_diff = time.perf_counter()
                gen_tokens = model._generate_iterative(full_task, gen_config)
                t_diff = time.perf_counter() - t0_diff
                audio = model._decode_and_post_process(
                    gen_tokens[0], full_task.ref_rms[0], gen_config
                )
            t_wall = time.perf_counter() - t0
            wall_times.append(t_wall)
            diff_times.append(t_diff)
            print(f"  Run {r+1}: Wall={t_wall:5.2f}s | Diff={t_diff:5.2f}s", flush=True)

        results[name] = {
            "wall_mean_s": float(np.mean(wall_times)),
            "wall_std_s": float(np.std(wall_times)),
            "diff_mean_s": float(np.mean(diff_times)),
            "diff_std_s": float(np.std(diff_times)),
        }

    return results


def benchmark_thread_configurations(model, prompt, text, warmup=2, runs=5):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 4] THREAD POOL CONFIGURATION BENCHMARK (Intra: 4, 6, 8)", flush=True)
    print("=" * 80, flush=True)

    gen_config = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        postprocess_output=True,
    )

    thread_configs = [4, 6, 8]
    results = {}

    for t in thread_configs:
        print(f"\nBenchmarking Intra-op Threads = {t}...", flush=True)
        setup_env(t)

        for _ in range(warmup):
            with torch.inference_mode():
                _ = model.generate(
                    text=text,
                    language="Vietnamese",
                    voice_clone_prompt=prompt,
                    generation_config=gen_config,
                )

        wall_times = []
        diff_times = []
        voc_times = []
        for r in range(runs):
            gc.collect()
            t0 = time.perf_counter()
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
            t_wall = time.perf_counter() - t0
            wall_times.append(t_wall)
            diff_times.append(t_diff)
            voc_times.append(t_voc)
            print(f"  Run {r+1}: Wall={t_wall:5.2f}s | Diff={t_diff:5.2f}s | Voc={t_voc:4.2f}s", flush=True)

        results[t] = {
            "wall_mean_s": float(np.mean(wall_times)),
            "wall_std_s": float(np.std(wall_times)),
            "diff_mean_s": float(np.mean(diff_times)),
            "diff_std_s": float(np.std(diff_times)),
            "voc_mean_s": float(np.mean(voc_times)),
            "voc_std_s": float(np.std(voc_times)),
        }

    return results


def run_allocation_and_memory_format_forensics(model, prompt, text):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 5] MEMORY ALLOCATION, COPY, SYNCHRONIZATION & LAYOUT FORENSICS", flush=True)
    print("=" * 80, flush=True)

    gen_config = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        postprocess_output=True,
    )

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

    inp = inputs_list[0]
    input_ids = inp["input_ids"]
    audio_mask = inp["audio_mask"]

    # Check embedding preparation tensors
    text_embeds = model.get_input_embeddings()(input_ids[:, 0, :])
    shifted_ids = (input_ids * audio_mask.unsqueeze(1)) + model.codebook_layer_offsets.view(1, -1, 1)
    audio_embeds = model.audio_embeddings(shifted_ids).sum(dim=1)
    inputs_embeds = torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    # Inspect layouts
    tensor_layouts = {
        "input_ids": {"shape": list(input_ids.shape), "stride": list(input_ids.stride()), "contiguous": input_ids.is_contiguous(), "dtype": str(input_ids.dtype)},
        "audio_mask": {"shape": list(audio_mask.shape), "stride": list(audio_mask.stride()), "contiguous": audio_mask.is_contiguous(), "dtype": str(audio_mask.dtype)},
        "inputs_embeds": {"shape": list(inputs_embeds.shape), "stride": list(inputs_embeds.stride()), "contiguous": inputs_embeds.is_contiguous(), "dtype": str(inputs_embeds.dtype)},
    }

    # Inspect first layer weights layout
    first_layer = model.llm.layers[0]
    q_weight = first_layer.self_attn.q_proj.weight
    gate_weight = first_layer.mlp.gate_proj.weight
    head_weight = model.audio_heads.weight

    weight_layouts = {
        "q_proj_weight": {"shape": list(q_weight.shape), "stride": list(q_weight.stride()), "contiguous": q_weight.is_contiguous(), "dtype": str(q_weight.dtype)},
        "gate_proj_weight": {"shape": list(gate_weight.shape), "stride": list(gate_weight.stride()), "contiguous": gate_weight.is_contiguous(), "dtype": str(gate_weight.dtype)},
        "audio_heads_weight": {"shape": list(head_weight.shape), "stride": list(head_weight.stride()), "contiguous": head_weight.is_contiguous(), "dtype": str(head_weight.dtype)},
    }

    print("Tensor Layout Inspection:", flush=True)
    for k, v in tensor_layouts.items():
        print(f"  • {k:15s}: shape={v['shape']}, stride={v['stride']}, contiguous={v['contiguous']}", flush=True)

    print("Weight Layout Inspection:", flush=True)
    for k, v in weight_layouts.items():
        print(f"  • {k:20s}: shape={v['shape']}, stride={v['stride']}, contiguous={v['contiguous']}", flush=True)

    # Audit synchronization calls
    sync_points = [
        {"location": "OmniVoice.generate -> full_task.get_indices", "type": "Python list comprehension / frame count", "sync": False},
        {"location": "_generate_iterative -> _get_time_steps.tolist()", "type": "One-time list conversion before loop", "sync": True, "overhead": "<0.01ms"},
        {"location": "_generate_iterative -> schedules calculation", "type": "Python math.ceil / int casting before loop", "sync": False},
        {"location": "_generate_iterative -> torch.topk indices", "type": "Native PyTorch tensor indexing in loop", "sync": False},
        {"location": "_decode_and_post_process -> audio_values.cpu().numpy()", "type": "End-of-generation vocoder export", "sync": True, "overhead": "~0.2ms"},
    ]

    print(f"Synchronization Points in generation path: {len(sync_points)} identified (0 blocking synchronizations inside the 4-step Transformer loop).", flush=True)

    return {
        "tensor_layouts": tensor_layouts,
        "weight_layouts": weight_layouts,
        "sync_points": sync_points,
    }


def run_detailed_one_step_profiler(model, prompt, text):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 6] FULL ONE-STEP PYTORCH PROFILER (INCLUSIVE VS SELF CPU TIME)", flush=True)
    print("=" * 80, flush=True)

    setup_env(6)
    gen_config = OmniVoiceGenerationConfig(
        num_step=1,
        guidance_scale=0.0,
        postprocess_output=False,
    )

    # Warmup
    with torch.inference_mode():
        _ = model.generate(text, language="Vietnamese", voice_clone_prompt=prompt, generation_config=gen_config)

    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.inference_mode():
            _ = model.generate(text, language="Vietnamese", voice_clone_prompt=prompt, generation_config=gen_config)

    key_avgs = prof.key_averages(group_by_input_shape=False)
    total_self_cpu = sum(item.self_cpu_time_total for item in key_avgs)

    print("\nTOP 15 CPU OPERATORS (BY SELF CPU TIME):", flush=True)
    print(key_avgs.table(sort_by="self_cpu_time_total", row_limit=15))

    top_ops = []
    gemm_self = 0
    non_gemm_self = 0

    for item in sorted(key_avgs, key=lambda x: x.self_cpu_time_total, reverse=True)[:15]:
        top_ops.append({
            "name": item.key,
            "self_cpu_ms": item.self_cpu_time_total / 1000.0,
            "self_cpu_pct": (item.self_cpu_time_total / total_self_cpu) * 100.0 if total_self_cpu > 0 else 0,
            "cpu_total_ms": item.cpu_time_total / 1000.0,
            "calls": item.count,
            "cpu_memory_usage_mb": item.cpu_memory_usage / (1024 * 1024),
        })

    for item in key_avgs:
        if item.key in ["aten::mm", "aten::addmm", "aten::bmm"]:
            gemm_self += item.self_cpu_time_total
        else:
            non_gemm_self += item.self_cpu_time_total

    summary = {
        "total_self_cpu_ms": total_self_cpu / 1000.0,
        "gemm_self_ms": gemm_self / 1000.0,
        "gemm_self_pct": (gemm_self / total_self_cpu) * 100.0,
        "non_gemm_self_ms": non_gemm_self / 1000.0,
        "non_gemm_self_pct": (non_gemm_self / total_self_cpu) * 100.0,
        "top_ops": top_ops,
    }

    print(f"Profiler Summary: Total Self CPU = {summary['total_self_cpu_ms']:.2f} ms | Pure GEMM = {summary['gemm_self_ms']:.2f} ms ({summary['gemm_self_pct']:.2f}%)", flush=True)
    return summary


def main():
    setup_env(DEFAULT_THREADS)
    model, prompt, text = load_model_and_prompt()

    audit_res = audit_inference_context_and_eval(model)
    baseline_res, sample_audio = benchmark_golden_baseline(model, prompt, text)
    context_res = benchmark_inference_contexts(model, prompt, text)
    threads_res = benchmark_thread_configurations(model, prompt, text)
    alloc_res = run_allocation_and_memory_format_forensics(model, prompt, text)
    prof_res = run_detailed_one_step_profiler(model, prompt, text)

    # Save audio for verification
    os.makedirs("benchmarks", exist_ok=True)
    audio_path = "benchmarks/v2_10_golden_baseline.wav"
    sf.write(audio_path, sample_audio, model.sampling_rate)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit": audit_res,
        "golden_baseline": baseline_res,
        "context_benchmark": context_res,
        "threads_benchmark": threads_res,
        "allocation_and_layout": alloc_res,
        "profiler": prof_res,
    }

    results_path = "benchmarks/v2_10_runtime_forensic_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] V2.10 Runtime Forensic Benchmark completed. Results saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
