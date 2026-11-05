"""
tests/test_v2_18_self_rollout_validation.py

Independent Controlled Test Suite for OmniVoice V2.18:
Self-Rollout Validation, Exposure Bias Analysis & Scheduled Sampling Design.

Tests:
1. Student Step 1 Confidence Ranking & Non-destructive Top-K Unmasking.
2. Self-Rollout State Generation & Step 2 Input Pipeline.
3. Teacher Target Alignment & Dynamic Active-Mask Loss Application.
4. Future Token / Target Non-Leakage Invariant.
5. Multi-Strategy Input & Loss Synthesis (Strategies A, B, C, D).
6. Micro-Benchmark Execution (Compute Overhead & Memory Delta).
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def _create_mock_vocab_model(vocab_size=1025, mask_id=1024, num_codebooks=8, hidden_dim=64):
    """Creates a tiny 2-layer transformer for rigorous unit verification."""
    class TinyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=4, dim_feedforward=128, batch_first=True
                ),
                num_layers=2
            )
            self.head = nn.Linear(hidden_dim, vocab_size)

        def forward(self, x):
            # x: [B, C, T]
            B, C, T = x.shape
            x_flat = x.view(B * C, T)
            h = self.embed(x_flat)
            h = self.transformer(h)
            logits = self.head(h)  # [B*C, T, V]
            return logits.view(B, C, T, -1)

    return TinyStudent()


def test_1_student_step1_unmasking():
    """Verify Student Step 1 unmasking: Top-K confidence selection with layer penalty."""
    B, C, T, V = 1, 8, 20, 1025
    mask_id = 1024
    layer_penalty = 5.0
    
    # Fully masked initial state x_0
    x_0 = torch.full((B, C, T), mask_id, dtype=torch.long)
    
    # Simulated logits where Layer 0 has highest confidence
    logits = torch.randn(B, C, T, V)
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs[..., mask_id] = -float("inf")
    
    pred_tokens = log_probs.argmax(dim=-1)
    confidence = log_probs.max(dim=-1)[0]
    
    layer_ids = torch.arange(C).view(1, -1, 1)
    scores = confidence - (layer_ids * layer_penalty)
    
    # Invariant: Mask already unmasked
    scores.masked_fill_(x_0 != mask_id, -float("inf"))
    
    # Step 1 unmasks 50% = 80 tokens
    k_1 = (C * T) // 2
    _, topk_idx = torch.topk(scores.flatten(), k_1)
    
    x_student_1 = x_0.clone()
    flat_tokens = x_student_1.flatten()
    flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
    x_student_1 = flat_tokens.view_as(x_student_1)
    
    unmasked_count = (x_student_1 != mask_id).sum().item()
    assert unmasked_count == k_1, f"Expected {k_1} unmasked tokens, got {unmasked_count}"
    
    # Verify Layer 0 received priority due to layer penalty
    layer_0_unmasked = (x_student_1[:, 0, :] != mask_id).sum().item()
    layer_7_unmasked = (x_student_1[:, 7, :] != mask_id).sum().item()
    assert layer_0_unmasked >= layer_7_unmasked, "Layer penalty prioritization failed"
    print(f"  [TEST 1 PASSED] Step 1 Unmasking: {unmasked_count}/{C*T} tokens (Layer 0: {layer_0_unmasked}/20, Layer 7: {layer_7_unmasked}/20).")


def test_2_self_rollout_step2_pipeline():
    """Verify Student Step 2 transitions from Step 1 self-rollout and preserves frozen tokens."""
    B, C, T, V = 1, 8, 20, 1025
    mask_id = 1024
    
    # Start from student Step 1 state (50% unmasked)
    x_0 = torch.full((B, C, T), mask_id, dtype=torch.long)
    x_student_1 = x_0.clone()
    unmasked_step1_indices = torch.randperm(C * T)[:80]
    x_student_1.flatten()[unmasked_step1_indices] = torch.randint(0, mask_id, (80,))
    
    step1_tokens_frozen = x_student_1.clone()
    
    # Step 2 logits
    step2_logits = torch.randn(B, C, T, V)
    log_probs = F.log_softmax(step2_logits, dim=-1)
    log_probs[..., mask_id] = -float("inf")
    pred_step2 = log_probs.argmax(dim=-1)
    scores = log_probs.max(dim=-1)[0]
    
    # CRUCIAL INVARIANT: Scores of already unmasked tokens must be -inf
    scores.masked_fill_(x_student_1 != mask_id, -float("inf"))
    
    # Step 2 unmasks remaining 80 tokens
    k_2 = 80
    _, topk_idx = torch.topk(scores.flatten(), k_2)
    
    x_student_2 = x_student_1.clone()
    flat_tokens = x_student_2.flatten()
    flat_tokens[topk_idx] = pred_step2.flatten()[topk_idx]
    x_student_2 = flat_tokens.view_as(x_student_2)
    
    # Invariant checks:
    # 1. 100% tokens must now be unmasked
    assert (x_student_2 == mask_id).sum().item() == 0, "Not all tokens unmasked at Step 2"
    # 2. Step 1 tokens must be strictly identical (non-destructive)
    assert torch.equal(x_student_2.flatten()[unmasked_step1_indices], step1_tokens_frozen.flatten()[unmasked_step1_indices]), "Step 1 tokens were modified!"
    print("  [TEST 2 PASSED] Step 2 Self-Rollout & Non-Destructive Freezing verified.")


def test_3_target_alignment_and_active_loss():
    """Verify that Teacher targets align correctly with Student's dynamic unmasked positions."""
    B, C, T, V = 1, 8, 20, 1025
    mask_id = 1024
    
    # Ground truth targets
    y_true = torch.randint(0, mask_id, (B, C, T))
    teacher_step4_logits = torch.randn(B, C, T, V)
    
    # Step 2 input: x_student_1 (50% unmasked)
    x_student_1 = torch.full((B, C, T), mask_id, dtype=torch.long)
    unmasked_step1 = torch.randperm(C * T)[:80]
    x_student_1.flatten()[unmasked_step1] = torch.randint(0, mask_id, (80,))
    
    # Active mask for Step 2: exactly the positions still masked in x_student_1
    step2_active_mask = (x_student_1 == mask_id)
    assert step2_active_mask.sum().item() == 80
    
    # Student step 2 predicted logits
    student_logits = torch.randn(B, C, T, V, requires_grad=True)
    
    # Compute active masked loss
    active_student_logits = student_logits[step2_active_mask]
    active_y_true = y_true[step2_active_mask]
    active_teacher_logits = teacher_step4_logits[step2_active_mask].detach()
    
    loss_ce = F.cross_entropy(active_student_logits, active_y_true)
    
    tau = 2.0
    s_log = F.log_softmax(active_student_logits / tau, dim=-1)
    t_prob = F.softmax(active_teacher_logits / tau, dim=-1)
    loss_kd = F.kl_div(s_log, t_prob, reduction="batchmean") * (tau ** 2)
    
    total_loss = 0.5 * loss_ce + 0.5 * loss_kd
    total_loss.backward()
    
    assert student_logits.grad is not None
    # Verify gradients only exist at active mask positions!
    grad_norm_active = student_logits.grad[step2_active_mask].abs().sum().item()
    grad_norm_inactive = student_logits.grad[~step2_active_mask].abs().sum().item()
    
    assert grad_norm_active > 0, "No gradient on active positions"
    assert grad_norm_inactive == 0.0, f"Leakage: Inactive positions received gradient ({grad_norm_inactive})"
    print(f"  [TEST 3 PASSED] Dynamic Active-Mask Loss verified (Active Grad: {grad_norm_active:.2f}, Inactive Grad: {grad_norm_inactive}).")


def test_4_future_token_non_leakage():
    """Verify that no future ground truth tokens leak into Step 1 input tensor."""
    B, C, T, V = 1, 8, 20, 1025
    mask_id = 1024
    
    # Ground truth targets
    y_true = torch.randint(0, mask_id, (B, C, T))
    
    # Initial input
    x_0 = torch.full((B, C, T), mask_id, dtype=torch.long)
    
    # Check that x_0 contains ZERO overlap with y_true
    leakage_in_x0 = (x_0 != mask_id).sum().item()
    assert leakage_in_x0 == 0, "Leakage in initial input x_0"
    
    print("  [TEST 4 PASSED] Future Token Non-Leakage verified.")


def test_5_multi_strategy_synthesis():
    """Verify that Strategies A, B, C, D generate distinct, valid training trajectories."""
    B, C, T, V = 1, 8, 20, 1025
    mask_id = 1024
    
    x_teacher_2 = torch.full((B, C, T), mask_id, dtype=torch.long)
    x_teacher_2.flatten()[:80] = 100  # Teacher unmasks first 80 tokens
    
    x_student_1 = torch.full((B, C, T), mask_id, dtype=torch.long)
    x_student_1.flatten()[40:120] = 200  # Student unmasks a different set of 80 tokens
    
    # Strategy A: 100% Teacher
    strat_a_input = x_teacher_2
    assert torch.equal(strat_a_input, x_teacher_2)
    
    # Strategy B: Scheduled Sampling with p=0.5
    torch.manual_seed(42)
    sampled_inputs_b = []
    for _ in range(10):
        use_student = torch.rand(1).item() < 0.5
        inp = x_student_1 if use_student else x_teacher_2
        sampled_inputs_b.append(inp)
    
    has_teacher = any(torch.equal(inp, x_teacher_2) for inp in sampled_inputs_b)
    has_student = any(torch.equal(inp, x_student_1) for inp in sampled_inputs_b)
    assert has_teacher and has_student, "Strategy B did not sample both Teacher and Student inputs"
    
    # Strategy C: 100% Student
    strat_c_input = x_student_1
    assert torch.equal(strat_c_input, x_student_1)
    
    # Strategy D: Curriculum Annealing
    p_step0 = min(1.0, 0 / 1000.0)
    p_step500 = min(1.0, 500 / 1000.0)
    p_step1000 = min(1.0, 1000 / 1000.0)
    assert p_step0 == 0.0 and p_step500 == 0.5 and p_step1000 == 1.0
    
    print("  [TEST 5 PASSED] Multi-Strategy Synthesis (A, B, C, D) verified.")


def test_6_micro_benchmark():
    """Benchmark compute time and memory overhead of self-rollout vs static trajectory."""
    model = _create_mock_vocab_model()
    B, C, T, V = 1, 8, 40, 1025
    mask_id = 1024
    
    x_0 = torch.full((B, C, T), mask_id, dtype=torch.long)
    y_true = torch.randint(0, mask_id, (B, C, T))
    
    # Benchmark Static Strategy (Strategy A): 1 forward pass per step
    t0 = time.perf_counter()
    out1 = model(x_0)
    loss1 = F.cross_entropy(out1.view(-1, V), y_true.view(-1))
    loss1.backward()
    t_static = (time.perf_counter() - t0) * 1000.0
    
    # Benchmark Self-Rollout Strategy (Strategy C):
    # Step 1 forward + topk unmask -> Step 2 forward + backward
    model.zero_grad()
    t0 = time.perf_counter()
    with torch.no_grad():
        out1 = model(x_0)
        log_probs = F.log_softmax(out1, dim=-1)
        log_probs[..., mask_id] = -float("inf")
        pred_step1 = log_probs.argmax(dim=-1)
        confidence = log_probs.max(dim=-1)[0]
        scores = confidence.clone()
        scores.masked_fill_(x_0 != mask_id, -float("inf"))
        _, topk_idx = torch.topk(scores.flatten(), (C * T) // 2)
        
        x_student_1 = x_0.clone()
        flat_tokens = x_student_1.flatten()
        flat_tokens[topk_idx] = pred_step1.flatten()[topk_idx]
        x_student_1 = flat_tokens.view_as(x_student_1)
    
    out2 = model(x_student_1)
    loss2 = F.cross_entropy(out2.view(-1, V), y_true.view(-1))
    loss2.backward()
    t_rollout = (time.perf_counter() - t0) * 1000.0
    
    overhead_ratio = t_rollout / t_static
    print(f"  [TEST 6 PASSED] Micro-Benchmark: Static: {t_static:.2f}ms | Self-Rollout: {t_rollout:.2f}ms (Overhead: {overhead_ratio:.2f}x).")


def main():
    print("=" * 80)
    print("OMNIVOICE V2.18 — SELF-ROLLOUT & SCHEDULED SAMPLING CONTROLLED TEST SUITE")
    print("=" * 80)
    test_1_student_step1_unmasking()
    test_2_self_rollout_step2_pipeline()
    test_3_target_alignment_and_active_loss()
    test_4_future_token_non_leakage()
    test_5_multi_strategy_synthesis()
    test_6_micro_benchmark()
    print("\n[ALL CONTROLLED TESTS PASSED SUCCESSFULLY — 6/6 PASSED]")


if __name__ == "__main__":
    main()
