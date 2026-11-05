"""
OMNIVOICE V2.9 — DIFFUSION STEP / COMPUTE REUSE FORENSIC BENCHMARK
Strictly read-only / forensic analysis of diffusion step computation,
tensor invariance, num_step scaling, quality gates, and compute reuse feasibility.
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
from omnivoice.models.omnivoice import (
    _get_time_steps,
    _filter_top_k,
    _gumbel_sample,
)

SAMPLE_RAW_TEXT = (
    "Hôm nay tôi xin trình bày báo cáo về tiến độ phát triển hệ thống trí tuệ nhân tạo."
)
NUM_THREADS = 6


def setup_env():
    torch.set_num_threads(NUM_THREADS)
    torch.set_num_interop_threads(NUM_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)


def load_model_and_prompt():
    print(f"Loading OmniVoice model (num_threads={NUM_THREADS})...", flush=True)
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


@torch.inference_mode()
def run_step_loop_forensics(model, prompt, text):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 1] DIFFUSION STEP LOOP & TENSOR INVARIANCE INSTRUMENTATION", flush=True)
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

    B = task.batch_size
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
        for i in range(B)
    ]

    c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
    max_c_len = max(c_lens)
    t_len = task.target_lens[0]
    c_len = c_lens[0]
    prefix_len = c_len - t_len

    use_cfg = gen_config.guidance_scale != 0
    batch_mult = 2 if use_cfg else 1

    batch_input_ids = torch.full(
        (batch_mult * B, model.config.num_audio_codebook, max_c_len),
        model.config.audio_mask_id,
        dtype=torch.long,
        device=model.device,
    )
    batch_audio_mask = torch.zeros(
        (batch_mult * B, max_c_len), dtype=torch.bool, device=model.device
    )
    batch_attention_mask = torch.zeros(
        (batch_mult * B, 1, max_c_len, max_c_len), dtype=torch.bool, device=model.device
    )

    for i, inp in enumerate(inputs_list):
        batch_input_ids[i, :, :c_len] = inp["input_ids"]
        batch_audio_mask[i, :c_len] = inp["audio_mask"]
        batch_attention_mask[i, :, :c_len, :c_len] = True

    tokens = torch.full(
        (B, model.config.num_audio_codebook, max(task.target_lens)),
        model.config.audio_mask_id,
        dtype=torch.long,
        device=model.device,
    )

    timesteps = _get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=gen_config.num_step,
        t_shift=gen_config.t_shift,
    ).tolist()

    schedules = []
    for t_l in task.target_lens:
        total_mask = t_l * model.config.num_audio_codebook
        rem = total_mask
        sched = []
        for step in range(gen_config.num_step):
            num = (
                rem
                if step == gen_config.num_step - 1
                else min(
                    math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                    rem,
                )
            )
            sched.append(int(num))
            rem -= int(num)
        schedules.append(sched)

    layer_ids = torch.arange(
        model.config.num_audio_codebook, device=model.device
    ).view(1, -1, 1)

    print(f"Sequence dimensions: Total={c_len} (Prefix={prefix_len}, Target={t_len})", flush=True)
    print(f"Target audio codes to unmask: {t_len * model.config.num_audio_codebook} (8 codebooks x {t_len} frames)", flush=True)
    print(f"Unmasking schedule across 4 steps: {schedules[0]}", flush=True)
    print(f"Timesteps: {[round(t, 4) for t in timesteps]}", flush=True)

    step_records = []
    initial_prefix_ids = batch_input_ids[:, :, :prefix_len].clone()
    initial_audio_mask = batch_audio_mask.clone()
    initial_attn_mask = batch_attention_mask.clone()

    previous_hidden_prefix = None

    for step in range(gen_config.num_step):
        t0_step = time.perf_counter()

        # Check tensor states before forward
        cur_prefix_ids = batch_input_ids[:, :, :prefix_len]
        prefix_ids_identical = torch.equal(cur_prefix_ids, initial_prefix_ids)
        audio_mask_identical = torch.equal(batch_audio_mask, initial_audio_mask)
        attn_mask_identical = torch.equal(batch_attention_mask, initial_attn_mask)

        # Count masked tokens in target
        cur_target_tokens = batch_input_ids[0, :, prefix_len:]
        num_masked_tokens = int((cur_target_tokens == model.config.audio_mask_id).sum().item())

        # Embed input preparation
        t0_embed = time.perf_counter()
        inputs_embeds = model._prepare_embed_inputs(batch_input_ids, batch_audio_mask)
        t_embed = (time.perf_counter() - t0_embed) * 1000

        # LLM forward pass
        t0_tf = time.perf_counter()
        llm_outputs = model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=batch_attention_mask,
            return_dict=True,
        )
        hidden_states = llm_outputs[0]
        t_tf = (time.perf_counter() - t0_tf) * 1000

        # Audio heads forward pass
        t0_heads = time.perf_counter()
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = model.audio_heads(hidden_states)
        batch_logits = logits_flat.view(
            batch_size,
            seq_len,
            model.config.num_audio_codebook,
            model.config.audio_vocab_size,
        ).permute(0, 2, 1, 3).to(torch.float32)
        t_heads = (time.perf_counter() - t0_heads) * 1000

        # Measure prefix hidden states drift across steps (due to bidirectional attention)
        cur_prefix_hidden = hidden_states[:, :prefix_len, :].detach().clone()
        if previous_hidden_prefix is not None:
            l2_diff = torch.norm(cur_prefix_hidden - previous_hidden_prefix).item()
            max_diff = torch.max(torch.abs(cur_prefix_hidden - previous_hidden_prefix)).item()
            cos_sim = F.cosine_similarity(
                cur_prefix_hidden.flatten(), previous_hidden_prefix.flatten(), dim=0
            ).item()
        else:
            l2_diff = 0.0
            max_diff = 0.0
            cos_sim = 1.0
        previous_hidden_prefix = cur_prefix_hidden

        # Sampling logic
        t0_sample = time.perf_counter()
        k = schedules[0][step]
        c_logits = batch_logits[0:1, :, prefix_len:c_len, :]
        u_logits = None

        pred_tokens, scores = model._predict_tokens_with_scoring(
            c_logits, u_logits, gen_config
        )
        scores = scores - (layer_ids * gen_config.layer_penalty_factor)

        if gen_config.position_temperature > 0.0:
            scores = _gumbel_sample(scores, gen_config.position_temperature)

        sample_tokens = tokens[0:1, :, :t_len]
        scores.masked_fill_(
            sample_tokens != model.config.audio_mask_id, -float("inf")
        )

        _, topk_idx = torch.topk(scores.flatten(), k)
        flat_tokens = sample_tokens.flatten()
        flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
        sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

        tokens[0:1, :, :t_len] = sample_tokens
        batch_input_ids[0:1, :, prefix_len:c_len] = sample_tokens
        t_sample = (time.perf_counter() - t0_sample) * 1000

        t_total_step = (time.perf_counter() - t0_step) * 1000

        record = {
            "step": step,
            "timestep_start": timesteps[step],
            "timestep_end": timesteps[step + 1],
            "unmask_k": k,
            "remaining_masked_before": num_masked_tokens,
            "prefix_ids_invariant": prefix_ids_identical,
            "audio_mask_invariant": audio_mask_identical,
            "attn_mask_invariant": attn_mask_identical,
            "embed_time_ms": t_embed,
            "transformer_time_ms": t_tf,
            "heads_time_ms": t_heads,
            "sampling_time_ms": t_sample,
            "total_step_time_ms": t_total_step,
            "prefix_hidden_drift_l2": l2_diff,
            "prefix_hidden_drift_max": max_diff,
            "prefix_hidden_cosine_sim": cos_sim,
        }
        step_records.append(record)

        print(
            f"Step {step} | t=[{timesteps[step]:.2f}->{timesteps[step+1]:.2f}] | Unmask: {k:3d} | "
            f"TF: {t_tf:6.1f}ms | Heads: {t_heads:5.1f}ms | Sample: {t_sample:4.1f}ms | "
            f"Total: {t_total_step:6.1f}ms | Prefix Invariant: {prefix_ids_identical} | "
            f"Prefix CosSim to Prev: {cos_sim:.6f}",
            flush=True,
        )

    return step_records


@torch.inference_mode()
def run_num_step_scaling_benchmark(model, prompt, text):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 2] NUM_STEP SCALING BENCHMARK (1, 2, 3, 4, 6, 8 STEPS)", flush=True)
    print("=" * 80, flush=True)

    step_candidates = [1, 2, 3, 4, 6, 8]
    scaling_results = {}
    WARMUP_RUNS = 2
    MEASURED_RUNS = 5

    os.makedirs("benchmarks", exist_ok=True)

    for num_step in step_candidates:
        print(f"\nEvaluating num_step = {num_step} (Warmup={WARMUP_RUNS}, Runs={MEASURED_RUNS})...", flush=True)
        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=0.0,
            postprocess_output=True,
        )

        # Warmup
        for w in range(WARMUP_RUNS):
            _ = model.generate(
                text=text,
                language="Vietnamese",
                voice_clone_prompt=prompt,
                generation_config=gen_config,
            )

        wall_times = []
        diffusion_times = []
        vocoder_times = []
        durations = []
        rtfs = []
        rms_list = []
        peak_list = []
        last_audio = None

        for r in range(MEASURED_RUNS):
            gc.collect()

            t0_wall = time.perf_counter()

            # Preprocess
            full_task = model._preprocess_all(
                text=text,
                language="Vietnamese",
                voice_clone_prompt=prompt,
                preprocess_prompt=gen_config.preprocess_prompt,
            )

            # Iterative Diffusion
            t0_diff = time.perf_counter()
            gen_tokens = model._generate_iterative(full_task, gen_config)
            t_diff = time.perf_counter() - t0_diff

            # Vocoder / Decode
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
            diffusion_times.append(t_diff)
            vocoder_times.append(t_voc)
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

        # Save sample wav for quality gate verification
        audio_save_path = f"benchmarks/audio_step{num_step}.wav"
        sf.write(audio_save_path, last_audio, model.sampling_rate)

        scaling_results[num_step] = {
            "wall_mean_s": float(np.mean(wall_times)),
            "wall_std_s": float(np.std(wall_times)),
            "wall_min_s": float(np.min(wall_times)),
            "wall_max_s": float(np.max(wall_times)),
            "diff_mean_s": float(np.mean(diffusion_times)),
            "diff_std_s": float(np.std(diffusion_times)),
            "voc_mean_s": float(np.mean(vocoder_times)),
            "voc_std_s": float(np.std(vocoder_times)),
            "audio_dur_mean_s": float(np.mean(durations)),
            "rtf_mean": float(np.mean(rtfs)),
            "rms_mean": float(np.mean(rms_list)),
            "peak_mean": float(np.mean(peak_list)),
            "audio_path": audio_save_path,
        }

    return scaling_results


def run_transformer_execution_analysis(model):
    print("\n" + "=" * 80, flush=True)
    print("[FORENSIC 3] TRANSFORMER ARCHITECTURE & FLOP EXECUTION FORENSICS", flush=True)
    print("=" * 80, flush=True)

    llm_config = model.config.llm_config
    num_layers = llm_config.num_hidden_layers
    hidden_size = llm_config.hidden_size
    num_heads = llm_config.num_attention_heads
    num_kv_heads = getattr(llm_config, "num_key_value_heads", num_heads)
    intermediate_size = llm_config.intermediate_size
    vocab_size = model.config.audio_vocab_size
    num_codebooks = model.config.num_audio_codebook
    head_dim = hidden_size // num_heads

    seq_len = 410

    # FLOP calculations per layer
    qkv_flops = 2 * seq_len * hidden_size * (num_heads + 2 * num_kv_heads) * head_dim
    attn_score_flops = 2 * num_heads * seq_len * seq_len * head_dim
    attn_ctx_flops = 2 * num_heads * seq_len * seq_len * head_dim
    out_proj_flops = 2 * seq_len * hidden_size * hidden_size
    mlp_flops = 6 * seq_len * hidden_size * intermediate_size

    layer_flops = qkv_flops + attn_score_flops + attn_ctx_flops + out_proj_flops + mlp_flops
    total_tf_flops = num_layers * layer_flops

    # Audio head FLOPs
    heads_flops = 2 * seq_len * hidden_size * (num_codebooks * vocab_size)

    total_step_flops = total_tf_flops + heads_flops

    info = {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "intermediate_size": intermediate_size,
        "num_codebooks": num_codebooks,
        "audio_vocab_size": vocab_size,
        "seq_len": seq_len,
        "layer_flops_m": layer_flops / 1e6,
        "total_tf_flops_m": total_tf_flops / 1e6,
        "audio_heads_flops_m": heads_flops / 1e6,
        "total_step_flops_m": total_step_flops / 1e6,
        "attn_implementation": getattr(llm_config, "_attn_implementation", "sdpa"),
    }

    print(f"Architecture: Qwen3 Transformer | Layers={num_layers}, Hidden={hidden_size}, Heads={num_heads}/{num_kv_heads}, FFN={intermediate_size}", flush=True)
    print(f"FLOPs per layer @ L=410 : {layer_flops / 1e6:.2f} MFLOP", flush=True)
    print(f"Total Transformer FLOPs : {total_tf_flops / 1e6:.2f} MFLOP ({total_tf_flops / 1e9:.2f} GFLOP)", flush=True)
    print(f"Audio Heads FLOPs       : {heads_flops / 1e6:.2f} MFLOP", flush=True)
    print(f"Total FLOPs per step    : {total_step_flops / 1e9:.2f} GFLOP per diffusion step", flush=True)

    return info


def main():
    setup_env()
    model, prompt, text = load_model_and_prompt()

    step_records = run_step_loop_forensics(model, prompt, text)
    scaling_results = run_num_step_scaling_benchmark(model, prompt, text)
    arch_info = run_transformer_execution_analysis(model)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step_records": step_records,
        "scaling_results": scaling_results,
        "arch_info": arch_info,
    }

    results_path = "benchmarks/v2_9_diffusion_forensic_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] V2.9 Diffusion Forensic Benchmark completed. Results saved to {results_path}", flush=True)


if __name__ == "__main__":
    main()
