"""
experiments/v2_20/loss_modules.py

Decoupled Codebook Knowledge Distillation (DC-KD) Loss Module for OmniVoice V2.20.
Supports uniform weighting, fixed linear decay, decoupled two-tier, and custom layer weights.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledCodebookKDLoss(nn.Module):
    def __init__(
        self,
        num_codebooks: int = 8,
        codebook_weights: Optional[List[float]] = None,
        temperatures: Optional[List[float]] = None,
        alpha_ce: float = 0.5,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        
        # Default to uniform if None
        if codebook_weights is None:
            codebook_weights = [1.0] * num_codebooks
        assert len(codebook_weights) == num_codebooks, f"Expected {num_codebooks} weights, got {len(codebook_weights)}"
        self.register_buffer("codebook_weights", torch.tensor(codebook_weights, dtype=torch.float32))

        # Default temperatures to 2.0 if None
        if temperatures is None:
            temperatures = [2.0] * num_codebooks
        assert len(temperatures) == num_codebooks, f"Expected {num_codebooks} temperatures, got {len(temperatures)}"
        self.register_buffer("temperatures", torch.tensor(temperatures, dtype=torch.float32))

        self.alpha_ce = alpha_ce

    def forward(
        self,
        student_logits: torch.Tensor,       # [B, C, T, V]
        teacher_logits: torch.Tensor,       # [B, C, T, V]
        target_tokens: torch.Tensor,        # [B, C, T]
        active_mask: torch.Tensor,          # [B, C, T] (bool: True where loss is calculated)
    ) -> dict:
        """
        Compute decoupled active-mask CE and KD loss across all RVQ codebook layers.
        """
        B, C, T, V = student_logits.shape
        assert C == self.num_codebooks

        layer_losses_ce = []
        layer_losses_kd = []
        layer_losses_total = []

        total_loss = torch.tensor(0.0, device=student_logits.device, dtype=torch.float32)

        for c in range(C):
            c_mask = active_mask[:, c, :]  # [B, T]
            w_c = self.codebook_weights[c]
            tau_c = self.temperatures[c]

            if not c_mask.any():
                layer_losses_ce.append(0.0)
                layer_losses_kd.append(0.0)
                layer_losses_total.append(0.0)
                continue

            s_logits_c = student_logits[:, c, :, :][c_mask]      # [N_active, V]
            t_logits_c = teacher_logits[:, c, :, :][c_mask].detach()  # [N_active, V] strictly detached
            targets_c = target_tokens[:, c, :][c_mask]           # [N_active]

            # 1. Cross-Entropy Loss
            loss_ce = F.cross_entropy(s_logits_c, targets_c, reduction="mean")

            # 2. Softmax KD with temperature tau_c scaled by tau_c^2
            s_log = F.log_softmax(s_logits_c / tau_c, dim=-1)
            t_prob = F.softmax(t_logits_c / tau_c, dim=-1)
            loss_kd = F.kl_div(s_log, t_prob, reduction="batchmean") * (tau_c ** 2)

            # 3. Layer Total
            layer_loss = self.alpha_ce * loss_ce + (1.0 - self.alpha_ce) * loss_kd
            weighted_layer_loss = w_c * layer_loss

            total_loss = total_loss + weighted_layer_loss

            layer_losses_ce.append(loss_ce.item())
            layer_losses_kd.append(loss_kd.item())
            layer_losses_total.append(layer_loss.item())

        # Normalize total loss by sum of active codebook weights
        active_weight_sum = self.codebook_weights.sum().item()
        normalized_total_loss = total_loss / max(active_weight_sum, 1e-6)

        return {
            "loss": normalized_total_loss,
            "layer_ce": layer_losses_ce,
            "layer_kd": layer_losses_kd,
            "layer_total": layer_losses_total,
        }
