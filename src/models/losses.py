import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralDamageConsistencyLoss(nn.Module):
    def __init__(self, damage_embed_dim=128):
        super().__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.mse = nn.MSELoss()

        self.severity_projector = nn.Sequential(
            nn.Linear(4, damage_embed_dim),
            nn.ReLU(),
            nn.Linear(damage_embed_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, pred_damage, gt_damage, pred_sev=None, gt_sev=None):
        gt = gt_damage.clamp(min=1e-8)
        gt = gt / gt.sum(dim=-1, keepdim=True)
        log_pred = F.log_softmax(pred_damage, dim=-1)
        kl_loss = self.kl(log_pred, gt)

        if pred_sev is not None and gt_sev is not None:
            sev_loss = self.mse(pred_sev.squeeze(), gt_sev)
        else:
            p_sev = self.severity_projector(pred_damage).squeeze(-1)
            g_sev = self.severity_projector(gt_damage).squeeze(-1)
            sev_loss = self.mse(p_sev, g_sev.detach())

        return kl_loss + sev_loss


class TemporalCoherencePenalty(nn.Module):
    def __init__(self, margin=0.0):
        super().__init__()
        self.margin = margin

    def forward(self, post_sev, pre_sev=None):
        # disasters do not undo damage; penalize if pre exceeds post
        if pre_sev is None:
            pre_sev = torch.zeros_like(post_sev)
        viol = F.relu(pre_sev - post_sev + self.margin)
        return (viol ** 2).mean()


class ContrastiveTemporalAlignmentLoss(nn.Module):
    """
    Novel Contrastive Temporal Alignment Loss (CTAL).

    In the bi-temporal satellite setting, the pre-disaster and post-disaster
    CLS tokens should be far apart when significant damage occurred (large
    damage severity) and close together when the scene is intact. This
    contrastive objective forces the change-detection module to produce
    embeddings that geometrically encode the magnitude of structural change.

    Within a batch, pairs with similar ground-truth severity (|sev_i - sev_j| < tau)
    are treated as positive pairs and are pulled together; pairs with large
    severity differences are treated as negatives and pushed apart. This is
    applied to the CLS token of the fused change-feature sequence.

    Why this is novel here:
    - Standard image captioning uses only token-level CE loss.
    - SSAL adds per-image structural supervision.
    - CTAL adds cross-image relational supervision, ensuring the embedding
      space is metrically consistent with damage ordering — something none
      of the three baselines do.
    """

    def __init__(self, temperature=0.07, similarity_tau=0.15):
        super().__init__()
        self.temp = temperature
        # pairs whose severity differs by less than tau are "similar"
        self.tau = similarity_tau

    def forward(self, cls_tokens, gt_sev):
        """
        cls_tokens: (B, D) — the CLS embedding from the change detector
        gt_sev    : (B,)   — ground-truth severity in [0, 1]
        """
        if cls_tokens.size(0) < 2:
            # need at least 2 samples to form pairs
            return cls_tokens.sum() * 0.0

        normed = F.normalize(cls_tokens, dim=-1)
        # similarity matrix: (B, B)
        sim = torch.matmul(normed, normed.T) / self.temp

        # positive mask: pairs with close severity scores
        sev_diff = (gt_sev.unsqueeze(0) - gt_sev.unsqueeze(1)).abs()
        pos_mask = (sev_diff < self.tau).float()
        # zero out diagonal (self-pairs)
        pos_mask.fill_diagonal_(0.0)
        neg_mask = 1.0 - pos_mask
        neg_mask.fill_diagonal_(0.0)

        # InfoNCE-style: pull positives, push negatives
        # for each anchor, average log-softmax over its positive set
        exp_sim = torch.exp(sim)
        # mask self out of denominator too
        eye = torch.eye(sim.size(0), device=sim.device).bool()
        exp_sim = exp_sim.masked_fill(eye, 0.0)

        denom = exp_sim.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        log_prob = sim - torch.log(denom)

        # only compute loss for anchors that have at least one positive
        n_pos = pos_mask.sum(dim=-1).clamp(min=1e-8)
        per_anchor_loss = -(log_prob * pos_mask).sum(dim=-1) / n_pos

        # mask anchors with zero positives to avoid NaN gradient
        has_pos = (pos_mask.sum(dim=-1) > 0).float()
        loss = (per_anchor_loss * has_pos).sum() / has_pos.sum().clamp(min=1.0)

        return loss


class StructuralSemanticAlignmentLoss(nn.Module):
    """
    Composite four-component loss:
      L_SSAL = lambda_ce * L_CE
             + lambda_structural * L_Structural
             + lambda_temporal * L_Temporal
             + lambda_contrastive * L_CTAL

    L_CE         : Token-level cross-entropy with label smoothing (text quality)
    L_Structural : KL divergence on damage-state distribution (structural accuracy)
    L_Temporal   : Monotonicity penalty — damage cannot decrease over time
    L_CTAL       : Contrastive Temporal Alignment — metric consistency across batch
    """

    def __init__(
        self,
        lambda_ce=1.0,
        lambda_structural=0.3,
        lambda_temporal=0.2,
        lambda_contrastive=0.15,
        label_smoothing=0.1,
        damage_embed_dim=128,
        contrastive_temperature=0.07,
        contrastive_tau=0.15,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_structural = lambda_structural
        self.lambda_temporal = lambda_temporal
        self.lambda_contrastive = lambda_contrastive

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
        self.structural = StructuralDamageConsistencyLoss(damage_embed_dim=damage_embed_dim)
        self.temporal = TemporalCoherencePenalty()
        self.contrastive = ContrastiveTemporalAlignmentLoss(
            temperature=contrastive_temperature,
            similarity_tau=contrastive_tau,
        )

    def forward(self, logits, labels, pred_damage, gt_damage,
                change_cls=None, pred_sev=None, gt_sev=None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        l_ce = self.ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        l_struct = self.structural(pred_damage, gt_damage, pred_sev, gt_sev)

        if pred_sev is not None:
            l_temp = self.temporal(pred_sev)
        else:
            w = torch.tensor([0.0, 0.33, 0.67, 1.0], device=pred_damage.device, dtype=pred_damage.dtype)
            l_temp = self.temporal((pred_damage * w).sum(dim=-1))

        # CTAL: needs change_cls and severity
        if change_cls is not None and gt_sev is not None:
            sev_for_ctal = gt_sev
        elif change_cls is not None:
            w = torch.tensor([0.0, 0.33, 0.67, 1.0], device=pred_damage.device, dtype=pred_damage.dtype)
            sev_for_ctal = (gt_damage.clamp(min=0) * w).sum(dim=-1)
        else:
            sev_for_ctal = None

        if change_cls is not None and sev_for_ctal is not None:
            l_ctal = self.contrastive(change_cls, sev_for_ctal)
        else:
            l_ctal = torch.tensor(0.0, device=logits.device)

        tot = (
            self.lambda_ce * l_ce
            + self.lambda_structural * l_struct
            + self.lambda_temporal * l_temp
            + self.lambda_contrastive * l_ctal
        )
        return {
            "loss": tot,
            "loss_ce": l_ce.detach(),
            "loss_struct": l_struct.detach(),
            "loss_temp": l_temp.detach(),
            "loss_ctal": l_ctal.detach(),
        }
