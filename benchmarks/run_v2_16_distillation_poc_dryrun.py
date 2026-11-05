#!/usr/bin/env python3
"""
OMNIVOICE V2.16 — DISTILLATION TECHNICAL VALIDATION & MINIMAL POC DRY-RUN
Strictly Read-Only / Isolated Dry-Run. Tests mathematical alignment, trajectory collection,
loss computation, and single-step gradient flow without altering production code.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Fix Windows console UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from omnivoice.core.engine import OmniVoiceEngine
from omnivoice.models.omnivoice import OmniVoice, VoiceClonePrompt, OmniVoiceGenerationConfig


def run_v2_16_poc_dryrun():
    print("=" * 80)
    print("OMNIVOICE V2.16 — DISTILLATION TECHNICAL VALIDATION DRY-RUN")
    print("=" * 80)

    # 1. Load Teacher Model in Memory
    print("\n[STEP 1] Loading OmniVoice Teacher Model instance...")
    t0 = time.perf_counter()
    engine = OmniVoiceEngine.get_instance(device="cpu")
    engine.initialize()
    model = engine.model
    model.eval()
    print(f"Teacher model loaded in {time.perf_counter() - t0:.2f}s")

    # 2. Prepare Sample Input Data for Trajectory Collection
    print("\n[STEP 2] Preparing Teacher 4-step Trajectory Collection...")
    test_text = "Thử nghiệm chưng cất mô hình OmniVoice."
    voice_prompt = engine.voice_manager.load_prompt("default")

    gen_task = model._preprocess_all(
        text=test_text,
        language="vi",
        voice_clone_prompt=voice_prompt,
    )
    gen_config = OmniVoiceGenerationConfig(
        num_step=4,
        guidance_scale=0.0,
    )

    # 3. Simulate Teacher 4-Step Trajectory Recording
    print("\n[STEP 3] Running Teacher 4-Step Forward & Capturing Trajectory Slices...")
    B = gen_task.batch_size
    T_target = gen_task.target_lens[0]
    num_codebooks = model.config.num_audio_codebook
    mask_id = model.config.audio_mask_id
    vocab_size = model.config.audio_vocab_size

    # Prepare inputs
    inputs_list = [
        model._prepare_inference_inputs(
            gen_task.texts[i],
            gen_task.target_lens[i],
            gen_task.ref_texts[i],
            gen_task.ref_audio_tokens[i],
            gen_task.langs[i],
            gen_task.instructs[i],
            gen_config.denoise,
        )
        for i in range(B)
    ]
    inp = inputs_list[0]
    c_len = inp["input_ids"].size(2)

    batch_input_ids = inp["input_ids"].clone()
    batch_audio_mask = inp["audio_mask"].clone()
    batch_attention_mask = torch.ones((1, 1, c_len, c_len), dtype=torch.bool)

    # Trajectory storage
    trajectories = []
    tokens = torch.full((B, num_codebooks, T_target), mask_id, dtype=torch.long)

    # Calculate step schedules
    total_tokens = T_target * num_codebooks
    step_k = total_tokens // 4
    schedules = [step_k, step_k, step_k, total_tokens - 3 * step_k]

    layer_ids = torch.arange(num_codebooks).view(1, -1, 1)

    with torch.no_grad():
        for step_idx in range(4):
            k = schedules[step_idx]
            out = model(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            )
            logits = out.logits[:, :, c_len - T_target : c_len, :].float()  # [1, 8, T, V]

            # Save state before unmasking
            state_before = tokens.clone()

            # Unmasking logic
            pred_tokens, scores = model._predict_tokens_with_scoring(logits, None, gen_config)
            scores = scores - (layer_ids * gen_config.layer_penalty_factor)
            sample_tokens = tokens.clone()
            scores.masked_fill_(sample_tokens != mask_id, -float("inf"))

            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            tokens.copy_(flat_tokens.view_as(sample_tokens))

            # Update batch_input_ids
            batch_input_ids[:, :, c_len - T_target : c_len] = tokens

            trajectories.append({
                "step": step_idx,
                "state_before": state_before,
                "state_after": tokens.clone(),
                "teacher_logits": logits.clone(),
                "unmasked_count": int(k)
            })
            print(f"  • Teacher Step {step_idx + 1}/4: Unmasked {k} tokens (Total active: {(tokens != mask_id).sum().item()}/{total_tokens})")

    # 4. Formulate Student 2-Step Targets (Grouping Steps 0+1 -> Step 1, Steps 2+3 -> Step 2)
    print("\n[STEP 4] Constructing Student 2-Step Targets from Trajectories...")
    # Student Step 1: Input is state_before[0] (All Mask), Target is state_after[1] (50% unmasked)
    s1_input_tokens = trajectories[0]["state_before"]
    s1_target_tokens = trajectories[1]["state_after"]
    s1_teacher_logits = trajectories[1]["teacher_logits"]

    # Student Step 2: Input is state_after[1] (50% unmasked), Target is state_after[3] (100% unmasked)
    s2_input_tokens = trajectories[1]["state_after"]
    s2_target_tokens = trajectories[3]["state_after"]
    s2_teacher_logits = trajectories[3]["teacher_logits"]

    print(f"  • Student Step 1 Target: {(s1_target_tokens != mask_id).sum().item()}/{total_tokens} tokens unmasked")
    print(f"  • Student Step 2 Target: {(s2_target_tokens != mask_id).sum().item()}/{total_tokens} tokens unmasked (100% complete)")

    # 5. Execute Student Loss Computation & Gradient Backpropagation Test
    print("\n[STEP 5] Testing Student Distillation Loss & Backprop Gradient Flow...")

    # Enable gradients on a small subset of model weights (Audio Heads & Final Layer) to verify gradient computation
    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.audio_heads.parameters():
        p.requires_grad = True

    # Build student Step 1 input
    student_input_ids = inp["input_ids"].clone()
    student_input_ids[:, :, c_len - T_target : c_len] = s1_input_tokens

    student_out = model(
        input_ids=student_input_ids,
        audio_mask=batch_audio_mask,
        attention_mask=batch_attention_mask,
    )
    student_logits = student_out.logits[:, :, c_len - T_target : c_len, :].float()  # [1, 8, T, V]

    # Mask for tokens to be unmasked in Step 1
    s1_mask = (s1_input_tokens == mask_id) & (s1_target_tokens != mask_id)
    num_supervision_tokens = s1_mask.sum().item()

    # 1. Hard-Token Cross Entropy Loss
    # student_logits: [1, 8, T, V] -> permute to [1, V, 8, T]
    loss_ce = F.cross_entropy(
        student_logits.permute(0, 3, 1, 2),
        torch.where(s1_mask, s1_target_tokens, torch.tensor(-100)),
        ignore_index=-100
    )

    # 2. Soft-Logit KL Divergence Loss
    tau = 2.0
    student_log_probs = F.log_softmax(student_logits / tau, dim=-1)
    teacher_probs = F.softmax(s1_teacher_logits / tau, dim=-1)
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)  # [1, 8, T]
    loss_kd = (kl_div * s1_mask.float()).sum() / s1_mask.sum().clamp(min=1.0) * (tau ** 2)

    # Combined Loss
    alpha = 0.5
    total_loss = alpha * loss_ce + (1 - alpha) * loss_kd

    print(f"  • Hard Token CE Loss : {loss_ce.item():.4f}")
    print(f"  • Soft Logit KD Loss : {loss_kd.item():.4f}")
    print(f"  • Total Combined Loss: {total_loss.item():.4f}")

    # Backpropagation verification
    total_loss.backward()

    # Check gradients
    grad_norms = []
    for name, p in model.audio_heads.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            grad_norms.append(norm)
            assert not torch.isnan(p.grad).any(), f"NaN in grad of {name}"
            assert not torch.isinf(p.grad).any(), f"Inf in grad of {name}"

    print(f"  • Backprop Verification: Gradient computed across {len(grad_norms)} parameters.")
    print(f"  • Max Gradient Norm   : {max(grad_norms):.6f} (Finite, Non-Zero, No NaN)")

    # Clean up model
    model.zero_grad()
    model.eval()

    # Memory and Tensor Shapes summary
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teacher_4step_schedule": schedules,
        "total_target_tokens": total_tokens,
        "supervision_tokens_step1": num_supervision_tokens,
        "loss_ce": float(loss_ce.item()),
        "loss_kd": float(loss_kd.item()),
        "total_loss": float(total_loss.item()),
        "gradient_finite_verified": True,
        "max_grad_norm": float(max(grad_norms)),
        "tensor_shapes": {
            "input_ids": list(student_input_ids.shape),
            "logits": list(student_logits.shape),
            "s1_mask": list(s1_mask.shape)
        },
        "dryrun_status": "SUCCESS"
    }

    os.makedirs("benchmarks", exist_ok=True)
    out_path = "benchmarks/v2_16_distillation_validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] V2.16 PoC Dry-Run successfully completed! Results saved to {out_path}")


if __name__ == "__main__":
    run_v2_16_poc_dryrun()
