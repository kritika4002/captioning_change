"""
Structural-Semantic Alignment Loss (SSAL) — Novel Loss Function
================================================================

This module defines a custom composite loss function designed specifically
for disaster change captioning.  It is the **core novel contribution**
of this project.

The SSAL loss addresses three key challenges:

1. **Caption Accuracy** (L_CE):
   Standard cross-entropy for next-token prediction.

2. **Structural Damage Consistency** (L_structural):
   Ensures the model's internal damage representation is consistent
   with the ground-truth damage distribution from building annotations.
   This forces the visual encoder to learn damage-aware features.

3. **Temporal Coherence Penalty** (L_temporal):
   Penalises the model when generated embeddings violate the expected
   pre→post temporal ordering of damage severity. In a valid disaster
   scenario, severity should increase (or stay same) from pre to post,
   never decrease.

Mathematical Formulation:
    L_total = λ₁ · L_CE + λ₂ · L_structural + λ₃ · L_temporal

Where:
    L_CE         = CrossEntropy(predicted_tokens, target_tokens)
    L_structural = KL(pred_damage_dist ‖ gt_damage_dist)
                   + MSE(pred_severity, gt_severity)
    L_temporal   = max(0, severity_pre - severity_post)²

References:
    - Label smoothing: Szegedy et al., "Rethinking the Inception
      Architecture for Computer Vision," CVPR 2016.
    - KL divergence for distribution matching: Hinton et al.,
      "Distilling the Knowledge in a Neural Network," 2015.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class StructuralDamageConsistencyLoss(nn.Module):
    """
    Ensures predicted damage distributions match ground-truth annotations.

    Combines:
      - KL divergence between predicted and GT damage distributions
      - MSE between predicted and GT scalar severity scores

    This forces the vision encoder to learn features that accurately
    represent the severity of structural damage.
    """

    def __init__(self, damage_embed_dim: int = 128):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.mse_loss = nn.MSELoss()

        # Learnable severity projector: damage vector → scalar severity
        self.severity_projector = nn.Sequential(
            nn.Linear(4, damage_embed_dim),
            nn.ReLU(),
            nn.Linear(damage_embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pred_damage: torch.Tensor,
        gt_damage: torch.Tensor,
        pred_severity: Optional[torch.Tensor] = None,
        gt_severity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_damage:  [B, 4]  predicted damage distribution (softmax)
            gt_damage:    [B, 4]  ground-truth damage distribution
            pred_severity: [B]   optional predicted severity scalar
            gt_severity:   [B]   optional ground-truth severity scalar

        Returns:
            Scalar loss value.
        """
        # Ensure gt_damage sums to 1 (already normalised in dataset)
        gt_damage = gt_damage.clamp(min=1e-8)
        gt_damage = gt_damage / gt_damage.sum(dim=-1, keepdim=True)

        # KL divergence: pred should match GT distribution
        log_pred = F.log_softmax(pred_damage, dim=-1)
        kl = self.kl_loss(log_pred, gt_damage)

        # Severity MSE (if provided)
        if pred_severity is not None and gt_severity is not None:
            sev_loss = self.mse_loss(pred_severity.squeeze(), gt_severity)
        else:
            # Derive severity from damage vectors via projector
            pred_sev = self.severity_projector(pred_damage).squeeze(-1)
            gt_sev = self.severity_projector(gt_damage).squeeze(-1)
            sev_loss = self.mse_loss(pred_sev, gt_sev.detach())

        return kl + sev_loss


class TemporalCoherencePenalty(nn.Module):
    """
    Penalises violations of temporal damage ordering.

    In a disaster scenario, the post-disaster severity must be ≥
    pre-disaster severity. This loss penalises cases where the model
    predicts a decrease in severity from pre → post.

    L_temporal = mean( max(0, S_pre - S_post)² )

    This acts as a soft constraint that respects the causal direction
    of damage: disasters cause damage, they don't undo it.
    """

    def __init__(self, margin: float = 0.0):
        """
        Args:
            margin: Minimum required severity increase (default 0).
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        severity_post: torch.Tensor,
        severity_pre: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            severity_post: [B]  post-disaster severity scores (0-1)
            severity_pre:  [B]  pre-disaster severity scores (0-1)
                                If None, pre is assumed to be 0.

        Returns:
            Scalar penalty loss.
        """
        if severity_pre is None:
            # Pre-disaster: no damage (severity = 0)
            severity_pre = torch.zeros_like(severity_post)

        # Violation: pre-severity exceeds post-severity
        violation = F.relu(severity_pre - severity_post + self.margin)
        return (violation ** 2).mean()


class StructuralSemanticAlignmentLoss(nn.Module):
    """
    Structural-Semantic Alignment Loss (SSAL) — Composite Loss Function.

    Combines three complementary objectives:
      1. Cross-Entropy Loss (L_CE)       — caption token prediction
      2. Structural Consistency (L_struct) — damage distribution matching
      3. Temporal Coherence (L_temporal)   — pre→post ordering constraint

    L_total = λ₁ · L_CE + λ₂ · L_structural + λ₃ · L_temporal

    This multi-objective loss ensures that the model not only generates
    fluent captions but also maintains physical consistency with the
    observed structural damage.

    Attributes:
        lambda_ce:          Weight for cross-entropy loss.
        lambda_structural:  Weight for structural damage consistency.
        lambda_temporal:    Weight for temporal coherence penalty.
    """

    def __init__(
        self,
        lambda_ce: float = 1.0,
        lambda_structural: float = 0.3,
        lambda_temporal: float = 0.2,
        label_smoothing: float = 0.1,
        damage_embed_dim: int = 128,
        vocab_size: int = 50257,  # GPT-2 vocabulary size
    ):
        """
        Args:
            lambda_ce:         Weight for the cross-entropy component.
            lambda_structural: Weight for structural consistency.
            lambda_temporal:   Weight for temporal coherence.
            label_smoothing:   Label smoothing factor for CE loss.
            damage_embed_dim:  Hidden dim for the severity projector.
            vocab_size:        Vocabulary size for the language model.
        """
        super().__init__()

        self.lambda_ce = lambda_ce
        self.lambda_structural = lambda_structural
        self.lambda_temporal = lambda_temporal

        # Component 1: Cross-Entropy with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100,  # Ignore padding tokens
        )

        # Component 2: Structural Damage Consistency
        self.structural_loss = StructuralDamageConsistencyLoss(
            damage_embed_dim=damage_embed_dim
        )

        # Component 3: Temporal Coherence Penalty
        self.temporal_loss = TemporalCoherencePenalty()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        pred_damage: torch.Tensor,
        gt_damage: torch.Tensor,
        pred_severity: Optional[torch.Tensor] = None,
        gt_severity: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the composite SSAL loss.

        Args:
            logits:        [B, L, V]  language model logits
            labels:        [B, L]     target token IDs
            pred_damage:   [B, 4]     predicted damage distribution
            gt_damage:     [B, 4]     ground-truth damage distribution
            pred_severity: [B]        predicted severity (optional)
            gt_severity:   [B]        ground-truth severity (optional)

        Returns:
            dict with:
                - loss:        Total weighted loss
                - loss_ce:     Cross-entropy component
                - loss_struct: Structural consistency component
                - loss_temp:   Temporal coherence component
        """
        # ── 1. Cross-Entropy Loss ──
        # Shift logits and labels for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_ce = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        # ── 2. Structural Damage Consistency ──
        loss_struct = self.structural_loss(
            pred_damage, gt_damage, pred_severity, gt_severity
        )

        # ── 3. Temporal Coherence Penalty ──
        # Post-disaster severity should be ≥ pre (pre assumed 0)
        if pred_severity is not None:
            loss_temp = self.temporal_loss(pred_severity)
        else:
            # Derive severity from damage prediction
            # Weighted sum: no=0, minor=0.33, major=0.67, destroyed=1.0
            weights = torch.tensor(
                [0.0, 0.33, 0.67, 1.0],
                device=pred_damage.device, dtype=pred_damage.dtype,
            )
            derived_severity = (pred_damage * weights).sum(dim=-1)
            loss_temp = self.temporal_loss(derived_severity)

        # ── Weighted combination ──
        total_loss = (
            self.lambda_ce * loss_ce
            + self.lambda_structural * loss_struct
            + self.lambda_temporal * loss_temp
        )

        return {
            "loss": total_loss,
            "loss_ce": loss_ce.detach(),
            "loss_struct": loss_struct.detach(),
            "loss_temp": loss_temp.detach(),
        }

    def extra_repr(self) -> str:
        return (
            f"λ_ce={self.lambda_ce}, "
            f"λ_struct={self.lambda_structural}, "
            f"λ_temp={self.lambda_temporal}"
        )
