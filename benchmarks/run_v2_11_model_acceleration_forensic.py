"""
OMNIVOICE V2.11 — MODEL-LEVEL ACCELERATION FORENSIC BENCHMARK
Forensic investigation of model-level, representation-level, and precision acceleration.
Audits FP32, FP16, BF16, INT8 Dynamic Quantization, torch.compile, TorchScript,
external runtimes, repository checkpoints, and Amdahl's Law theoretical limits.
"""

import copy
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
NUM_THREADS = 6


def setup_env():
    torch.set_num_threads(NUM_THREADS)
    try:
        torch.set_num_interop_threads(NUM_THREADS)
    except RuntimeError:
        pass
    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)


def load_model_and_prompt():
    print("Loading OmniVoice model for V2.11 forensic investigation...", flush=True)
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


def audit_model_architecture_and_parameters(model):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 1] MODEL ARCHITECTURE & PARAMETER ACCOUNTING", flush=True)
    print("=" * 80, flush=True)

    llm = model.llm
    cfg = model.config
    llm_cfg = cfg.llm_config

    num_layers = llm_cfg.num_hidden_layers
    hidden_size = llm_cfg.hidden_size
    num_heads = llm_cfg.num_attention_heads
    num_kv_heads = getattr(llm_cfg, "num_key_value_heads", num_heads)
    intermediate_size = llm_cfg.intermediate_size
    vocab_size = cfg.audio_vocab_size
    num_codebooks = cfg.num_audio_codebook
    head_dim = hidden_size // num_heads

    # Parameter accounting
    llm_params = sum(p.numel() for p in llm.parameters())
    embed_params = sum(p.numel() for p in model.audio_embeddings.parameters())
    heads_params = sum(p.numel() for p in model.audio_heads.parameters())
    tokenizer_params = sum(p.numel() for p in model.audio_tokenizer.parameters()) if model.audio_tokenizer else 0
    total_params = sum(p.numel() for p in model.parameters()) + tokenizer_params

    # Memory accounting (FP32)
    fp32_model_mb = (total_params * 4) / (1024 * 1024)

    # FLOPs per step for L=150 (current benchmark) and L=410 (max chunk)
    def calc_step_flops(seq_len):
        qkv = 2 * seq_len * hidden_size * (num_heads + 2 * num_kv_heads) * head_dim
        attn_s = 2 * num_heads * seq_len * seq_len * head_dim
        attn_c = 2 * num_heads * seq_len * seq_len * head_dim
        out_p = 2 * seq_len * hidden_size * hidden_size
        mlp = 6 * seq_len * hidden_size * intermediate_size
        tf_step = num_layers * (qkv + attn_s + attn_c + out_p + mlp)
        heads_step = 2 * seq_len * hidden_size * (num_codebooks * vocab_size)
        return tf_step + heads_step

    flops_150 = calc_step_flops(150)
    flops_410 = calc_step_flops(410)

    arch_info = {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "num_codebooks": num_codebooks,
        "audio_vocab_size": vocab_size,
        "llm_params": llm_params,
        "embed_params": embed_params,
        "heads_params": heads_params,
        "tokenizer_params": tokenizer_params,
        "total_params": total_params,
        "fp32_model_mb": fp32_model_mb,
        "step_flops_150_gflop": flops_150 / 1e9,
        "step_flops_410_gflop": flops_410 / 1e9,
        "total_4step_flops_150_gflop": (flops_150 * 4) / 1e9,
        "total_4step_flops_410_gflop": (flops_410 * 4) / 1e9,
    }

    print(f"Architecture: Qwen3 Transformer Backbone | {num_layers} Layers, Hidden={hidden_size}, Heads={num_heads}/{num_kv_heads}, FFN={intermediate_size}", flush=True)
    print(f"Parameter Counts:", flush=True)
    print(f"  • LLM Transformer Backbone : {llm_params / 1e6:.2f} M params ({llm_params * 4 / (1024*1024):.1f} MB)", flush=True)
    print(f"  • Audio Codebook Embeddings: {embed_params / 1e6:.2f} M params ({embed_params * 4 / (1024*1024):.1f} MB)", flush=True)
    print(f"  • Audio Heads Linear Layer : {heads_params / 1e6:.2f} M params ({heads_params * 4 / (1024*1024):.1f} MB)", flush=True)
    print(f"  • Higgs Vocoder Tokenizer  : {tokenizer_params / 1e6:.2f} M params ({tokenizer_params * 4 / (1024*1024):.1f} MB)", flush=True)
    print(f"  • Total Model Parameters   : {total_params / 1e6:.2f} M params ({fp32_model_mb:.1f} MB FP32)", flush=True)
    print(f"Computational Workload:", flush=True)
    print(f"  • 1-Step FLOPs @ L=150 tokens: {flops_150 / 1e9:.2f} GFLOP (4 Steps = {flops_150 * 4 / 1e9:.2f} GFLOP)", flush=True)
    print(f"  • 1-Step FLOPs @ L=410 tokens: {flops_410 / 1e9:.2f} GFLOP (4 Steps = {flops_410 * 4 / 1e9:.2f} GFLOP)", flush=True)

    return arch_info


def audit_current_dtypes(model):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 2] CURRENT DTYPES & TENSOR PRECISION AUDIT", flush=True)
    print("=" * 80, flush=True)

    dtypes_found = set()
    for name, param in model.named_parameters():
        dtypes_found.add(str(param.dtype))

    tokenizer_dtypes = set()
    if model.audio_tokenizer:
        for name, param in model.audio_tokenizer.named_parameters():
            tokenizer_dtypes.add(str(param.dtype))

    print(f"OmniVoice Model Parameters Dtype(s): {dtypes_found}", flush=True)
    print(f"Audio Tokenizer Parameters Dtype(s): {tokenizer_dtypes}", flush=True)

    is_pure_fp32 = (dtypes_found == {"torch.float32"}) and (tokenizer_dtypes == {"torch.float32"})
    print(f"Pure FP32 Precision: {is_pure_fp32}", flush=True)

    return {
        "model_param_dtypes": list(dtypes_found),
        "tokenizer_param_dtypes": list(tokenizer_dtypes),
        "is_pure_fp32": is_pure_fp32,
    }


def run_baseline_benchmark(model, prompt, text, warmup=2, runs=5):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 3] REPRODUCE GOLDEN BASELINE (FP32, 6 THREADS, NUM_STEP=4)", flush=True)
    print("=" * 80, flush=True)

    setup_env()
    gen_config = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
        postprocess_output=True,
    )

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
        "diff_mean_s": float(np.mean(diff_times)),
        "diff_std_s": float(np.std(diff_times)),
        "voc_mean_s": float(np.mean(voc_times)),
        "voc_std_s": float(np.std(voc_times)),
        "audio_dur_mean_s": float(np.mean(durations)),
        "rtf_mean": float(np.mean(rtfs)),
        "rms_mean": float(np.mean(rms_list)),
        "peak_mean": float(np.mean(peak_list)),
    }
    print(f"Golden Baseline Summary: Mean Wall={res['wall_mean_s']:.2f}s (Diff={res['diff_mean_s']:.2f}s), RTF={res['rtf_mean']:.2f}x", flush=True)
    return res, last_audio


def audit_cpu_precision_and_quantization_support():
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 4] CPU PRECISION & QUANTIZATION FEASIBILITY AUDIT", flush=True)
    print("=" * 80, flush=True)

    has_cpu_amp = hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast")

    # Test BF16 GEMM execution natively
    bf16_gemm_status = "Unknown"
    try:
        a_bf16 = torch.randn(10, 10, dtype=torch.bfloat16)
        b_bf16 = torch.randn(10, 10, dtype=torch.bfloat16)
        c_bf16 = torch.mm(a_bf16, b_bf16)
        bf16_gemm_status = "Executable (Software Emulation on Zen+)"
    except Exception as e:
        bf16_gemm_status = f"Failed: {e}"

    # Test FP16 CPU GEMM execution
    fp16_gemm_status = "Unknown"
    try:
        a_fp16 = torch.randn(10, 10, dtype=torch.float16)
        b_fp16 = torch.randn(10, 10, dtype=torch.float16)
        c_fp16 = torch.mm(a_fp16, b_fp16)
        fp16_gemm_status = "Executable"
    except Exception as e:
        fp16_gemm_status = f"Failed: {e}"

    has_dynamic_quant = hasattr(torch.ao.quantization, "quantize_dynamic") or hasattr(torch.quantization, "quantize_dynamic")
    supported_qengines = torch.backends.quantized.supported_engines

    print(f"PyTorch CPU AMP available: {has_cpu_amp}", flush=True)
    print(f"BF16 CPU GEMM status    : {bf16_gemm_status}", flush=True)
    print(f"FP16 CPU GEMM status    : {fp16_gemm_status}", flush=True)
    print(f"Dynamic Quantization API : {has_dynamic_quant}", flush=True)
    print(f"Quantized Engines        : {supported_qengines} (Current: {torch.backends.quantized.engine})", flush=True)

    return {
        "has_cpu_amp": has_cpu_amp,
        "bf16_gemm_status": bf16_gemm_status,
        "fp16_gemm_status": fp16_gemm_status,
        "has_dynamic_quant": has_dynamic_quant,
        "supported_qengines": list(supported_qengines),
        "active_qengine": torch.backends.quantized.engine,
    }


def benchmark_bf16_and_int8_in_isolation(model, prompt, text):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 5] ISOLATED LOWER PRECISION & INT8 BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    gen_config = OmniVoiceGenerationConfig(
        num_step=1,  # 1 step for precision screening
        guidance_scale=0.0,
        postprocess_output=False,
    )

    results = {}

    # A. Test BF16 via torch.autocast('cpu', dtype=torch.bfloat16)
    print("Testing BF16 CPU Autocast (1-step generation)...", flush=True)
    try:
        for _ in range(2):
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                _ = model.generate(text, language="Vietnamese", voice_clone_prompt=prompt, generation_config=gen_config)

        bf16_timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                _ = model.generate(text, language="Vietnamese", voice_clone_prompt=prompt, generation_config=gen_config)
            t_el = (time.perf_counter() - t0) * 1000
            bf16_timings.append(t_el)

        results["bf16_autocast_1step_ms"] = float(np.mean(bf16_timings))
        print(f"  • BF16 Autocast 1-step Mean: {results['bf16_autocast_1step_ms']:.2f} ms", flush=True)
    except Exception as e:
        results["bf16_autocast_1step_ms"] = f"Failed: {e}"
        print(f"  • BF16 Autocast Failed: {e}", flush=True)

    # B. Test INT8 Dynamic Quantization on temporary copy of Qwen3 LLM
    print("\nTesting INT8 Dynamic Quantization on isolated Qwen3 Linear layers...", flush=True)
    try:
        t0_quant = time.perf_counter()
        quantized_llm = torch.ao.quantization.quantize_dynamic(
            model.llm,
            {nn.Linear},
            dtype=torch.qint8,
        )
        quant_time_s = time.perf_counter() - t0_quant
        print(f"  • INT8 Dynamic Quantization Time: {quant_time_s:.2f} s", flush=True)

        qlinear_count = sum(1 for m in quantized_llm.modules() if "QuantizedLinear" in type(m).__name__ or "Dynamic" in type(m).__name__)
        print(f"  • Quantized Linear Layers: {qlinear_count}", flush=True)

        # Benchmark isolated 1-step pass with quantized LLM
        original_llm = model.llm
        model.llm = quantized_llm

        for _ in range(2):
            with torch.inference_mode():
                _ = model.generate(text, language="Vietnamese", voice_clone_prompt=prompt, generation_config=gen_config)

        int8_timings = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model.generate(text, language="Vietnamese", voice_clone_prompt=prompt, generation_config=gen_config)
            t_el = (time.perf_counter() - t0) * 1000
            int8_timings.append(t_el)

        results["int8_dynamic_quant_1step_ms"] = float(np.mean(int8_timings))
        results["int8_quant_time_s"] = quant_time_s
        results["int8_qlinear_count"] = qlinear_count
        print(f"  • INT8 Dynamic Quant 1-step Mean: {results['int8_dynamic_quant_1step_ms']:.2f} ms", flush=True)

        # Restore original LLM
        model.llm = original_llm
        del quantized_llm
        gc.collect()

    except Exception as e:
        results["int8_dynamic_quant_1step_ms"] = f"Failed: {e}"
        print(f"  • INT8 Quantization Failed: {e}", flush=True)

    return results


def audit_torch_compile_and_runtimes():
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 6] COMPILER & EXTERNAL RUNTIME AUDIT", flush=True)
    print("=" * 80, flush=True)

    backends = torch._dynamo.list_backends() if hasattr(torch, "_dynamo") else []

    def check_pkg(pkg_name):
        try:
            __import__(pkg_name)
            return True
        except ImportError:
            return False

    runtimes = {
        "onnxruntime": check_pkg("onnxruntime"),
        "openvino": check_pkg("openvino"),
        "directml": check_pkg("torch_directml"),
        "torchscript": hasattr(torch.jit, "trace"),
        "executorch": check_pkg("executorch"),
    }

    print(f"torch.compile Backends: {backends}", flush=True)
    for k, v in runtimes.items():
        print(f"  • {k:15s}: {'Installed' if v else 'Not Installed'}", flush=True)

    return {
        "dynamo_backends": backends,
        "external_runtimes": runtimes,
    }


def audit_repository_checkpoints():
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 7] REPOSITORY CHECKPOINT & DISTILLATION AUDIT", flush=True)
    print("=" * 80, flush=True)

    checkpoint_files = []
    for root, dirs, files in os.walk("."):
        if any(ign in root for ign in [".git", ".venv", "__pycache__", "benchmarks"]):
            continue
        for f in files:
            if f.endswith((".pt", ".bin", ".safetensors", ".onnx", ".tflite")):
                checkpoint_files.append(os.path.join(root, f))

    print(f"Checkpoints found in repository tree: {len(checkpoint_files)}", flush=True)
    for cf in checkpoint_files:
        print(f"  • {cf} ({os.path.getsize(cf) / (1024*1024):.2f} MB)", flush=True)

    has_distilled = any("distill" in cf.lower() or "tiny" in cf.lower() for cf in checkpoint_files)
    print(f"Distilled / Reduced Model Checkpoints present: {has_distilled}", flush=True)

    return {
        "checkpoints": checkpoint_files,
        "has_distilled": has_distilled,
    }


def calculate_amdahl_bounds(gemm_pct=0.515, tf_pct=0.975):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 8] AMDAHL'S LAW THEORETICAL SPEEDUP BOUNDS", flush=True)
    print("=" * 80, flush=True)

    gemm_2x = 1.0 / ((1.0 - gemm_pct) + gemm_pct / 2.0)
    gemm_3x = 1.0 / ((1.0 - gemm_pct) + gemm_pct / 3.0)
    gemm_4x = 1.0 / ((1.0 - gemm_pct) + gemm_pct / 4.0)
    gemm_inf = 1.0 / (1.0 - gemm_pct)

    tf_2x = 1.0 / ((1.0 - tf_pct) + tf_pct / 2.0)
    tf_4x = 1.0 / ((1.0 - tf_pct) + tf_pct / 4.0)
    tf_inf = 1.0 / (1.0 - tf_pct)

    bounds = {
        "gemm_fraction_p": gemm_pct,
        "transformer_fraction_p": tf_pct,
        "gemm_speedups": {
            "2x_gemm": gemm_2x,
            "3x_gemm": gemm_3x,
            "4x_gemm": gemm_4x,
            "inf_gemm_limit": gemm_inf,
        },
        "transformer_speedups": {
            "2x_transformer": tf_2x,
            "4x_transformer": tf_4x,
            "inf_transformer_limit": tf_inf,
        },
    }

    print(f"GEMM Acceleration (P = {gemm_pct*100:.1f}%):", flush=True)
    print(f"  • 2x Faster GEMM -> End-to-End Speedup = {gemm_2x:.2f}x (Max latency reduction: {(1 - 1/gemm_2x)*100:.1f}%)", flush=True)
    print(f"  • 3x Faster GEMM -> End-to-End Speedup = {gemm_3x:.2f}x (Max latency reduction: {(1 - 1/gemm_3x)*100:.1f}%)", flush=True)
    print(f"  • 4x Faster GEMM -> End-to-End Speedup = {gemm_4x:.2f}x (Max latency reduction: {(1 - 1/gemm_4x)*100:.1f}%)", flush=True)
    print(f"  • Infinite (0s) GEMM -> Max Theoretical Limit = {gemm_inf:.2f}x", flush=True)

    print(f"\nEntire Transformer Acceleration (P = {tf_pct*100:.1f}%):", flush=True)
    print(f"  • 2x Faster Transformer -> End-to-End Speedup = {tf_2x:.2f}x", flush=True)
    print(f"  • 4x Faster Transformer -> End-to-End Speedup = {tf_4x:.2f}x", flush=True)
    print(f"  • Infinite (0s) Transformer -> Max Theoretical Limit = {tf_inf:.2f}x (Vocoder & misc floor)", flush=True)

    return bounds


def main():
    setup_env()
    model, prompt, text = load_model_and_prompt()

    arch_res = audit_model_architecture_and_parameters(model)
    dtype_res = audit_current_dtypes(model)
    baseline_res, sample_audio = run_baseline_benchmark(model, prompt, text)
    precision_res = audit_cpu_precision_and_quantization_support()
    quant_bench_res = benchmark_bf16_and_int8_in_isolation(model, prompt, text)
    runtimes_res = audit_torch_compile_and_runtimes()
    repo_res = audit_repository_checkpoints()
    amdahl_res = calculate_amdahl_bounds()

    # Save sample audio
    os.makedirs("benchmarks", exist_ok=True)
    audio_path = "benchmarks/v2_11_baseline.wav"
    sf.write(audio_path, sample_audio, model.sampling_rate)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": arch_res,
        "dtypes": dtype_res,
        "golden_baseline": baseline_res,
        "precision_audit": precision_res,
        "quant_benchmarks": quant_bench_res,
        "runtimes": runtimes_res,
        "repo_checkpoints": repo_res,
        "amdahl_bounds": amdahl_res,
    }

    results_path = "benchmarks/v2_11_model_acceleration_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] V2.11 Model Acceleration Forensic completed. Results saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
