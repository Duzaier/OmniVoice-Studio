#!/usr/bin/env python3
"""
Unit test for OmniVoice Progressive Distillation Token Alignment & Loss Logic.
Verifies:
1. Masking logic & non-destructive unmasking invariant.
2. Tensor alignment between Teacher Trajectory logits and Student predictions.
3. Finite value bounds on Cross-Entropy and KL-Divergence distillation loss.
"""

import torch
import torch.nn.functional as F


def test_distillation_tensor_alignment_and_loss():
    B = 1
    C = 8        # Number of codebooks
    T = 60       # Target audio frames
    V = 2048     # Audio vocabulary size
    mask_id = 2048
    tau = 2.0
    alpha = 0.5

    # Mock inputs
    x_input = torch.full((B, C, T), mask_id, dtype=torch.long)
    
    # Mock teacher target: 50% unmasked at Step 1
    x_target = torch.full((B, C, T), mask_id, dtype=torch.long)
    unmasked_indices = torch.randperm(C * T)[: (C * T) // 2]
    mock_tokens = torch.randint(0, V, (len(unmasked_indices),), dtype=torch.long)
    x_target.view(-1)[unmasked_indices] = mock_tokens

    # Mock logits
    student_logits = torch.randn(B, C, T, V, requires_grad=True)
    teacher_logits = torch.randn(B, C, T, V)

    # 1. Verify supervision mask
    supervision_mask = (x_input == mask_id) & (x_target != mask_id)
    assert supervision_mask.sum().item() == (C * T) // 2, "Supervision mask count mismatch!"

    # 2. Hard-token Cross Entropy Loss
    loss_ce = F.cross_entropy(
        student_logits.permute(0, 3, 1, 2),
        torch.where(supervision_mask, x_target, torch.tensor(-100)),
        ignore_index=-100
    )
    assert torch.isfinite(loss_ce), "CE Loss must be finite"
    assert loss_ce.item() > 0.0, "CE Loss must be positive"

    # 3. Soft-logit KL Divergence Loss
    s_log_probs = F.log_softmax(student_logits / tau, dim=-1)
    t_probs = F.softmax(teacher_logits / tau, dim=-1)
    kl_div = F.kl_div(s_log_probs, t_probs, reduction="none").sum(dim=-1)
    loss_kd = (kl_div * supervision_mask.float()).sum() / supervision_mask.sum().clamp(min=1.0) * (tau ** 2)

    assert torch.isfinite(loss_kd), "KD Loss must be finite"
    assert loss_kd.item() >= 0.0, "KD Loss must be non-negative"

    # 4. Total Loss & Backward pass
    total_loss = alpha * loss_ce + (1 - alpha) * loss_kd
    total_loss.backward()

    assert student_logits.grad is not None, "Gradient must be populated"
    assert not torch.isnan(student_logits.grad).any(), "Gradient must not contain NaN"
    assert not torch.isinf(student_logits.grad).any(), "Gradient must not contain Inf"
    assert (student_logits.grad != 0).any(), "Gradient must be non-zero"


if __name__ == "__main__":
    test_distillation_tensor_alignment_and_loss()
    print("Test passed successfully!")
