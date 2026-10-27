import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast
    def _create_scaler(enabled=True):
        return GradScaler("cuda", enabled=enabled)
    def _autocast_context(enabled=True):
        return autocast("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import GradScaler, autocast
    def _create_scaler(enabled=True):
        return GradScaler(enabled=enabled)
    def _autocast_context(enabled=True):
        return autocast(enabled=enabled)
from transformers import get_scheduler, AutoTokenizer

from src.models.losses import StructuralSemanticAlignmentLoss
from src.training.sam import SAM
from src.evaluation.metrics import CaptionEvaluator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Dual-Optimization Trainer
    =========================

    Two optimizers run every training step:

      1. AdamW (base)  — fast, adaptive moment updates for convergence.
      2. SAM           — Sharpness-Aware Minimization wrapping AdamW.
                         Finds flat minima for better generalization.

    The training loop performs two forward+backward passes per accumulation step:
      - Pass 1: compute SSAL loss, backprop, SAM.first_step() (perturbs weights)
      - Pass 2: recompute SSAL at perturbed weights, backprop, SAM.second_step()

    Loss function is the four-component SSAL:
      L = lambda_ce * L_CE
        + lambda_structural * L_Structural
        + lambda_temporal * L_Temporal
        + lambda_contrastive * L_CTAL
    """

    def __init__(self, model, train_loader, val_loader, config, tokenizer_name="gpt2"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(
            config.get("project", {}).get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        t_cfg = config.get("training", {})
        self.epochs = t_cfg.get("epochs", 30)
        self.lr = t_cfg.get("learning_rate", 2e-4)
        self.weight_decay = t_cfg.get("weight_decay", 0.01)
        self.grad_accum_steps = t_cfg.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = t_cfg.get("max_grad_norm", 1.0)
        self.warmup_steps = t_cfg.get("warmup_steps", 500)
        self.use_fp16 = t_cfg.get("fp16", True) and self.device.type == "cuda"
        self.save_every = t_cfg.get("save_every", 5)
        self.eval_every = t_cfg.get("eval_every", 1)
        self.patience = t_cfg.get("early_stopping_patience", 7)

        self.out_dir = Path(t_cfg.get("output_dir", "./outputs"))
        self.ckpt_dir = Path(t_cfg.get("checkpoint_dir", "./checkpoints"))
        self.log_dir = Path(t_cfg.get("log_dir", "./logs"))
        for d in [self.out_dir, self.ckpt_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # --- Loss: 4-component SSAL with contrastive alignment ---
        l_cfg = config.get("loss", {})
        self.criterion = StructuralSemanticAlignmentLoss(
            lambda_ce=l_cfg.get("lambda_ce", 1.0),
            lambda_structural=l_cfg.get("lambda_structural", 0.3),
            lambda_temporal=l_cfg.get("lambda_temporal", 0.2),
            lambda_contrastive=l_cfg.get("lambda_contrastive", 0.15),
            label_smoothing=l_cfg.get("label_smoothing", 0.1),
            damage_embed_dim=l_cfg.get("damage_embed_dim", 128),
            contrastive_temperature=l_cfg.get("contrastive_temperature", 0.07),
            contrastive_tau=l_cfg.get("contrastive_tau", 0.15),
        ).to(self.device)

        # --- Dual Optimization: SAM wrapping AdamW ---
        sam_rho = t_cfg.get("sam_rho", 0.05)
        sam_adaptive = t_cfg.get("sam_adaptive", False)
        self.optimizer = SAM(
            model.get_trainable_params(),
            base_optimizer=torch.optim.AdamW,
            rho=sam_rho,
            adaptive=sam_adaptive,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # expose base optimizer for scheduler
        base_opt = self.optimizer.base_optimizer

        n_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        self.scheduler = get_scheduler(
            t_cfg.get("scheduler", "cosine"),
            optimizer=base_opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=n_steps,
        )

        # AMP scaler — only on pass-1 gradients (SAM pass-2 reuses them)
        self.scaler = _create_scaler(enabled=self.use_fp16)
        e_cfg = config.get("evaluation", {})
        self.evaluator = CaptionEvaluator(metrics=e_cfg.get("metrics", ["bleu", "meteor", "rouge_l"]))

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = []
        self.global_step = 0

    def train(self):
        for ep in range(1, self.epochs + 1):
            t_metrics = self._train_epoch(ep)
            v_metrics = self._validate(ep) if ep % self.eval_every == 0 else {}

            rec = {
                "epoch": ep,
                "train": t_metrics,
                "val": v_metrics,
                "lr": self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.lr,
            }
            self.history.append(rec)
            self._log_epoch(ep, t_metrics, v_metrics)

            if ep % self.save_every == 0:
                self._save_checkpoint(ep)

            val_loss = v_metrics.get("loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(ep, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"early stopping at epoch {ep}")
                    break

        self._save_checkpoint(ep, tag="final")
        self._save_history()
        return self.history

    def _compute_loss(self, batch):
        """Forward pass + 4-component SSAL loss. Returns losses dict."""
        pre = batch["pre_image"].to(self.device)
        post = batch["post_image"].to(self.device)
        cap_ids = batch["caption_ids"].to(self.device)
        mask = batch["attention_mask"].to(self.device)
        dmg = batch["damage_vector"].to(self.device)
        sev = batch["severity_score"].to(self.device)

        with _autocast_context(enabled=self.use_fp16):
            out = self.model(pre_img=pre, post_img=post, caption_ids=cap_ids, attention_mask=mask)
            n_vis = out["logits"].size(1) - cap_ids.size(1)
            vis_pad = torch.full((cap_ids.size(0), n_vis), -100, device=self.device, dtype=cap_ids.dtype)
            lbls = torch.cat([vis_pad, cap_ids], dim=1)

            losses = self.criterion(
                logits=out["logits"],
                labels=lbls,
                pred_damage=out["pred_damage"],
                gt_damage=dmg,
                change_cls=out.get("change_cls"),
                pred_sev=None,
                gt_sev=sev,
            )
        return losses

    def _train_epoch(self, ep):
        self.model.train()
        tot = {"loss": 0.0, "loss_ce": 0.0, "loss_struct": 0.0, "loss_temp": 0.0, "loss_ctal": 0.0}
        n_batches = 0
        self.optimizer.zero_grad()

        for b_idx, batch in enumerate(self.train_loader):
            # ---- SAM Pass 1: gradient at current weights ----
            losses = self._compute_loss(batch)
            loss1 = losses["loss"] / self.grad_accum_steps
            self.scaler.scale(loss1).backward()

            if (b_idx + 1) % self.grad_accum_steps == 0:
                if self.use_fp16:
                    with torch.no_grad():
                        inv_scale = 1.0 / max(self.scaler.get_scale(), 1e-8)
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad.mul_(inv_scale)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                # perturb weights to worst-case neighbor
                self.optimizer.first_step(zero_grad=True)

                # ---- SAM Pass 2: gradient at perturbed weights ----
                self.model.train()
                losses2 = self._compute_loss(batch)
                loss2 = losses2["loss"] / self.grad_accum_steps
                if self.use_fp16:
                    self.scaler.scale(loss2).backward()
                    self.scaler.unscale_(self.optimizer.base_optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    # revert perturbation and apply base AdamW step through scaler
                    with torch.no_grad():
                        for group in self.optimizer.param_groups:
                            for p in group["params"]:
                                if p.grad is not None and p in self.optimizer.state and "e_w" in self.optimizer.state[p]:
                                    p.sub_(self.optimizer.state[p]["e_w"])
                    self.scaler.step(self.optimizer.base_optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else:
                    loss2.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.second_step(zero_grad=True)

                self.scheduler.step()
                self.global_step += 1

            for k in tot:
                tot[k] += losses[k].item()
            n_batches += 1

        avg = lambda x: round(x / max(n_batches, 1), 4)
        return {k: avg(v) for k, v in tot.items()}

    @torch.no_grad()
    def _validate(self, ep):
        self.model.eval()
        tot_loss, n_batches = 0.0, 0
        preds, refs = [], []

        for batch in self.val_loader:
            pre = batch["pre_image"].to(self.device)
            post = batch["post_image"].to(self.device)
            cap_ids = batch["caption_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            dmg = batch["damage_vector"].to(self.device)
            sev = batch["severity_score"].to(self.device)

            out = self.model(pre_img=pre, post_img=post, caption_ids=cap_ids, attention_mask=mask)
            n_vis = out["logits"].size(1) - cap_ids.size(1)
            vis_pad = torch.full((cap_ids.size(0), n_vis), -100, device=self.device, dtype=cap_ids.dtype)
            lbls = torch.cat([vis_pad, cap_ids], dim=1)

            losses = self.criterion(
                logits=out["logits"],
                labels=lbls,
                pred_damage=out["pred_damage"],
                gt_damage=dmg,
                change_cls=out.get("change_cls"),
                gt_sev=sev,
            )
            tot_loss += losses["loss"].item()
            n_batches += 1

            pred_ids = out["logits"][:, n_vis:, :].argmax(dim=-1)
            for i in range(pred_ids.size(0)):
                preds.append(self.tokenizer.decode(pred_ids[i], skip_special_tokens=True))
                refs.append(self.tokenizer.decode(cap_ids[i], skip_special_tokens=True))

        val_loss = round(tot_loss / max(n_batches, 1), 4)
        metrics = self.evaluator.evaluate(preds, refs)
        return {"loss": val_loss, **metrics}

    def _log_epoch(self, ep, train, val):
        msg = (
            f"ep {ep:02d}/{self.epochs} | "
            f"loss={train['loss']:.4f} "
            f"(ce={train['loss_ce']:.4f} "
            f"struct={train['loss_struct']:.4f} "
            f"temp={train['loss_temp']:.4f} "
            f"ctal={train['loss_ctal']:.4f})"
        )
        if val:
            msg += f" | val_loss={val.get('loss', 0):.4f} b4={val.get('bleu_4', 0):.4f} meteor={val.get('meteor', 0):.4f}"
        logger.info(msg)

    def _save_checkpoint(self, ep, is_best=False, tag=""):
        name = "best" if is_best else (tag or f"epoch_{ep}")
        sdir = self.ckpt_dir / name
        sdir.mkdir(parents=True, exist_ok=True)

        self.model.save_adapter(str(sdir / "lora_adapter"))

        torch.save({
            "epoch": ep,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer": self.optimizer.base_optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }, sdir / "training_state.pt")

        trainable = {
            k: v for k, v in self.model.state_dict().items()
            if "lora" not in k.lower() and any(p.data_ptr() == v.data_ptr() for p in self.model.parameters() if p.requires_grad)
        }
        torch.save(trainable, sdir / "trainable_weights.pt")

    def _save_history(self):
        with open(self.out_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
