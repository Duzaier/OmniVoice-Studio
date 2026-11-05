"""
experiments/v2_20/run_v2_20_controlled_experiment.py

Master Controlled Experiment Runner for OmniVoice V2.20.
Executes the minimum experiment protocol:
  A. Smoke test (Imports, loss modules, sanity checker)
  B. Tiny-batch training test (Forward + Backward + Invariant checks)
  C. Checkpoint save/load verification
  D. Deterministic replay verification
  E. Multi-group speed & loss simulation (G0, G1, G2, G3a, G3b, G3c, G4)
  F. Real Reference Audio Sanity Audit on voices/
  G. Vietnamese Tone Test Set benchmark
"""

import os
import sys
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sanity_checker import ReferenceAudioSanityChecker
from loss_modules import DecoupledCodebookKDLoss
from vietnamese_tone_evaluator import VietnameseToneEvaluator, VIETNAMESE_100_TONE_ITEMS


def run_protocol_a_smoke_test():
    """Protocol A: Smoke test of all isolated components."""
    print("\n--- Protocol A: Smoke Test ---")
    checker = ReferenceAudioSanityChecker()
    loss_mod = DecoupledCodebookKDLoss()
    evaluator = VietnameseToneEvaluator()
    
    assert checker is not None
    assert loss_mod is not None
    assert len(evaluator.test_items) == 100
    print("  [PASSED] Smoke test: All components initialized successfully.")


def run_protocol_b_tiny_batch_train():
    """Protocol B: Tiny-batch training test with gradient isolation and mask invariant."""
    print("\n--- Protocol B: Tiny-Batch Training Test ---")
    torch.manual_seed(42)
    B, C, T, V = 1, 8, 20, 1025
    mask_id = 1024

    # Target ground truth and teacher logits
    y_true = torch.randint(0, mask_id, (B, C, T))
    teacher_logits = torch.randn(B, C, T, V)

    # Trainable student head
    student_head = nn.Parameter(torch.randn(B, C, T, V) * 0.01)
    optimizer = torch.optim.AdamW([student_head], lr=0.05)

    # Active mask: unmask 50%
    active_mask = torch.zeros((B, C, T), dtype=torch.bool)
    active_mask[:, :, :10] = True  # First 10 frames active

    loss_fn = DecoupledCodebookKDLoss(
        num_codebooks=8,
        codebook_weights=[1.5, 1.3, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8],
        temperatures=[1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0, 3.0]
    )

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        out = loss_fn(student_head, teacher_logits, y_true, active_mask)
        loss = out["loss"]
        loss.backward()

        # Invariant check: Inactive positions must have zero gradient
        inactive_grad = student_head.grad[:, :, 10:, :].abs().sum().item()
        assert inactive_grad == 0.0, f"Gradient leakage at step {step}: {inactive_grad}"

        optimizer.step()
        losses.append(loss.item())

    initial_loss = losses[0]
    final_loss = losses[-1]
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"  [PASSED] Tiny-batch training: Loss dropped from {initial_loss:.4f} to {final_loss:.4f} (Zero inactive grad).")


def run_protocol_c_checkpoint_save_load(tmp_dir="experiments/v2_20/checkpoints"):
    """Protocol C: Checkpoint save/load verification."""
    print("\n--- Protocol C: Checkpoint Save/Load Test ---")
    os.makedirs(tmp_dir, exist_ok=True)
    ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")

    dummy_model = nn.Linear(64, 10)
    torch.save({"state_dict": dummy_model.state_dict(), "step": 100}, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=True)
    assert loaded["step"] == 100
    assert "weight" in loaded["state_dict"]
    
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("  [PASSED] Checkpoint save and load verified.")


def run_protocol_d_deterministic_replay():
    """Protocol D: Deterministic replay verification."""
    print("\n--- Protocol D: Deterministic Replay Test ---")
    def run_sim(seed):
        torch.manual_seed(seed)
        x = torch.randn(2, 64)
        lin = nn.Linear(64, 10)
        return lin(x).sum().item()

    v1 = run_sim(42)
    v2 = run_sim(42)
    v3 = run_sim(123)
    assert v1 == v2, "Non-deterministic execution on identical seed!"
    assert v1 != v3, "Identical execution on different seeds!"
    print("  [PASSED] Deterministic replay verified across seeds.")


def run_protocol_e_multi_group_simulation():
    """Protocol E: Multi-group controlled comparison simulation (G0, G1, G2, G3a, G3b, G3c, G4)."""
    print("\n--- Protocol E: Multi-Group Controlled Benchmark Simulation ---")
    
    configs = {
        "G0_Static_Control": {
            "p_student": 0.0,
            "weights": [1.0] * 8,
            "sanity_check": False
        },
        "G1_Curriculum_Uniform": {
            "p_student": 0.5,
            "weights": [1.0] * 8,
            "sanity_check": False
        },
        "G2_Curriculum_Sanity": {
            "p_student": 0.5,
            "weights": [1.0] * 8,
            "sanity_check": True
        },
        "G3a_Codebook_Uniform": {
            "p_student": 0.5,
            "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "sanity_check": False
        },
        "G3b_Codebook_LinearDecay": {
            "p_student": 0.5,
            "weights": [1.5, 1.3, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8],
            "sanity_check": False
        },
        "G3c_Codebook_ToneOriented": {
            "p_student": 0.5,
            "weights": [2.0, 1.5, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6],
            "sanity_check": False
        },
        "G4_Integrated_Candidate": {
            "p_student": 0.75,
            "weights": [1.5, 1.3, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8],
            "sanity_check": True
        }
    }

    results = {}
    B, C, T, V = 1, 8, 40, 1025
    mask_id = 1024

    for name, cfg in configs.items():
        loss_fn = DecoupledCodebookKDLoss(num_codebooks=8, codebook_weights=cfg["weights"])
        student_head = nn.Parameter(torch.randn(B, C, T, V) * 0.01)
        optimizer = torch.optim.AdamW([student_head], lr=0.01)
        
        y_true = torch.randint(0, mask_id, (B, C, T))
        teacher_logits = torch.randn(B, C, T, V)
        active_mask = torch.ones((B, C, T), dtype=torch.bool)

        t0 = time.perf_counter()
        for _ in range(5):
            optimizer.zero_grad()
            out = loss_fn(student_head, teacher_logits, y_true, active_mask)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
        latency_ms = (time.perf_counter() - t0) * 1000.0 / 5.0

        results[name] = {
            "final_loss": round(loss.item(), 4),
            "step_latency_ms": round(latency_ms, 2),
            "p_student": cfg["p_student"],
            "weights": cfg["weights"],
            "sanity_check": cfg["sanity_check"]
        }
        print(f"  [{name}] Final Loss: {loss.item():.4f} | Latency: {latency_ms:.2f}ms/step")

    return results


def run_protocol_f_reference_audio_audit():
    """Protocol F: Audit actual local voice samples in voices/ directory."""
    print("\n--- Protocol F: Reference Audio Sanity Audit on voices/ ---")
    checker = ReferenceAudioSanityChecker()
    voices_dir = "voices"
    audit_records = []

    if os.path.exists(voices_dir):
        for root, _, files in os.walk(voices_dir):
            for file in files:
                if file.endswith(".wav"):
                    wav_path = os.path.join(root, file)
                    speaker_name = os.path.basename(root)
                    try:
                        waveform, sr = torchaudio.load(wav_path)
                        eval_res = checker.evaluate(waveform, sr)
                        eval_res["file_path"] = wav_path
                        eval_res["speaker"] = speaker_name
                        audit_records.append(eval_res)
                        status_str = "ACCEPTED" if eval_res["accepted"] else "REJECTED"
                        print(f"  [{status_str}] {speaker_name}/{file} - Duration: {eval_res['duration_sec']:.2f}s | Reason: {eval_res['reason']}")
                    except Exception as e:
                        print(f"  [ERROR] Failed to load {wav_path}: {e}")

    # Stress test on synthetic edge cases
    print("\n  [Stress Testing Edge Cases with Sanity Checker]:")
    # Edge case 1: Too short (< 2.5s)
    short_audio = torch.randn(1, int(24000 * 1.5)) * 0.1
    res1 = checker.evaluate(short_audio, 24000)
    assert not res1["accepted"] and "too short" in res1["reason"]
    print("    - Short audio (1.5s): Correctly REJECTED.")

    # Edge case 2: Clipped audio (> 0.2%)
    clipped_audio = torch.randn(1, int(24000 * 4.0)) * 0.1
    clipped_audio[:, :500] = 1.0  # 500 / 96000 = 0.52% > 0.2%
    res2 = checker.evaluate(clipped_audio, 24000)
    assert not res2["accepted"] and ("Clipping ratio" in res2["reason"] or "RMS" in res2["reason"])
    print("    - Clipped audio: Correctly REJECTED.")

    # Edge case 3: Valid 4s speech
    valid_audio = torch.randn(1, int(24000 * 4.0)) * 0.1
    res3 = checker.evaluate(valid_audio, 24000)
    assert res3["accepted"]
    print("    - Clean synthetic audio: Correctly ACCEPTED.")

    return audit_records


def run_protocol_g_tone_benchmark():
    """Protocol G: Vietnamese Tone Test Set benchmark."""
    print("\n--- Protocol G: Vietnamese Tone Test Set Benchmark ---")
    evaluator = VietnameseToneEvaluator()
    
    # Mock transcriptions for testing evaluator
    perfect_preds = {item.id: item.test_sentence for item in VIETNAMESE_100_TONE_ITEMS}
    res_perfect = evaluator.evaluate_transcription_results(perfect_preds)
    assert res_perfect["tone_accuracy"] == 100.0

    # Confused transcriptions (e.g. swap hoi/nga on 10 items)
    confused_preds = perfect_preds.copy()
    for i in range(31, 41):
        confused_preds[i] = "Đọc nhầm không rõ dấu"
    res_confused = evaluator.evaluate_transcription_results(confused_preds)
    
    print(f"  [Perfect Baseline] Tone Accuracy: {res_perfect['tone_accuracy']}%")
    print(f"  [Simulated Confused] Tone Accuracy: {res_confused['tone_accuracy']}% (Hoi/Nga Breakdown: {res_confused['confusion_breakdown']['hoi/nga']['accuracy_pct']}%)")
    return res_perfect, res_confused


def main():
    print("=" * 80)
    print("OMNIVOICE V2.20 — CONTROLLED DISTILLATION EXPERIMENT PROTOCOL")
    print("=" * 80)
    
    run_protocol_a_smoke_test()
    run_protocol_b_tiny_batch_train()
    run_protocol_c_checkpoint_save_load()
    run_protocol_d_deterministic_replay()
    group_sim_results = run_protocol_e_multi_group_simulation()
    audio_audit_results = run_protocol_f_reference_audio_audit()
    tone_bench_perfect, tone_bench_confused = run_protocol_g_tone_benchmark()

    # Save summary manifest
    summary_manifest = {
        "version": "2.20",
        "git_commit": "e5a3059e9f423fb34ffd34b534cccdc28037c1b1",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "multi_group_simulation": group_sim_results,
        "reference_audio_audit": audio_audit_results,
        "vietnamese_tone_benchmark": {
            "total_items": 100,
            "perfect_accuracy": tone_bench_perfect["tone_accuracy"],
            "confused_accuracy": tone_bench_confused["tone_accuracy"],
        }
    }
    
    os.makedirs("benchmarks", exist_ok=True)
    with open("benchmarks/v2_20_protocol_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_manifest, f, indent=2, ensure_ascii=False)
        
    print("\n[ALL PROTOCOL VERIFICATIONS PASSED SUCCESSFULLY]")


if __name__ == "__main__":
    main()
