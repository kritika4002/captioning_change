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


class StructuralSemanticAlignmentLoss(nn.Module):
    def __init__(
        self,
        lambda_ce=1.0,
        lambda_structural=0.3,
        lambda_temporal=0.2,
        label_smoothing=0.1,
        damage_embed_dim=128,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_structural = lambda_structural
        self.lambda_temporal = lambda_temporal

        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
        self.structural = StructuralDamageConsistencyLoss(damage_embed_dim=damage_embed_dim)
        self.temporal = TemporalCoherencePenalty()

    def forward(self, logits, labels, pred_damage, gt_damage, pred_sev=None, gt_sev=None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        l_ce = self.ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        l_struct = self.structural(pred_damage, gt_damage, pred_sev, gt_sev)

        if pred_sev is not None:
            l_temp = self.temporal(pred_sev)
        else:
            w = torch.tensor([0.0, 0.33, 0.67, 1.0], device=pred_damage.device, dtype=pred_damage.dtype)
            l_temp = self.temporal((pred_damage * w).sum(dim=-1))

        tot = self.lambda_ce * l_ce + self.lambda_structural * l_struct + self.lambda_temporal * l_temp
        return {
            "loss": tot,
            "loss_ce": l_ce.detach(),
            "loss_struct": l_struct.detach(),
            "loss_temp": l_temp.detach(),
        }
