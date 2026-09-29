import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler, AutoTokenizer

from src.models.losses import StructuralSemanticAlignmentLoss
from src.evaluation.metrics import CaptionEvaluator

logger = logging.getLogger(__name__)


class Trainer:
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

        l_cfg = config.get("loss", {})
        self.criterion = StructuralSemanticAlignmentLoss(
            lambda_ce=l_cfg.get("lambda_ce", 1.0),
            lambda_structural=l_cfg.get("lambda_structural", 0.3),
            lambda_temporal=l_cfg.get("lambda_temporal", 0.2),
            label_smoothing=l_cfg.get("label_smoothing", 0.1),
            damage_embed_dim=l_cfg.get("damage_embed_dim", 128),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(), lr=self.lr, weight_decay=self.weight_decay
        )

        n_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        self.scheduler = get_scheduler(
            t_cfg.get("scheduler", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=n_steps,
        )

        self.scaler = GradScaler(enabled=self.use_fp16)
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
                "lr": self.scheduler.get_last_lr()[0],
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

    def _train_epoch(self, ep):
        self.model.train()
        tot_loss, tot_ce, tot_struct, tot_temp = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for b_idx, batch in enumerate(self.train_loader):
            pre = batch["pre_image"].to(self.device)
            post = batch["post_image"].to(self.device)
            cap_ids = batch["caption_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            dmg = batch["damage_vector"].to(self.device)
            sev = batch["severity_score"].to(self.device)

            with autocast(enabled=self.use_fp16):
                out = self.model(pre_image=pre, post_image=post, caption_ids=cap_ids, attention_mask=mask)
                n_vis = out["logits"].size(1) - cap_ids.size(1)
                vis_pad = torch.full((cap_ids.size(0), n_vis), -100, device=self.device, dtype=cap_ids.dtype)
                lbls = torch.cat([vis_pad, cap_ids], dim=1)

                losses = self.criterion(
                    logits=out["logits"],
                    labels=lbls,
                    pred_damage=out["pred_damage"],
                    gt_damage=dmg,
                    gt_sev=sev,
                )
                loss = losses["loss"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (b_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            tot_loss += losses["loss"].item()
            tot_ce += losses["loss_ce"].item()
            tot_struct += losses["loss_struct"].item()
            tot_temp += losses["loss_temp"].item()
            n_batches += 1

        avg = lambda x: round(x / max(n_batches, 1), 4)
        return {"loss": avg(tot_loss), "loss_ce": avg(tot_ce), "loss_struct": avg(tot_struct), "loss_temp": avg(tot_temp)}

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

            out = self.model(pre_image=pre, post_image=post, caption_ids=cap_ids, attention_mask=mask)
            n_vis = out["logits"].size(1) - cap_ids.size(1)
            vis_pad = torch.full((cap_ids.size(0), n_vis), -100, device=self.device, dtype=cap_ids.dtype)
            lbls = torch.cat([vis_pad, cap_ids], dim=1)

            losses = self.criterion(
                logits=out["logits"],
                labels=lbls,
                pred_damage=out["pred_damage"],
                gt_damage=dmg,
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
        msg = f"ep {ep:02d}/{self.epochs} | loss={train['loss']:.4f} (ce={train['loss_ce']:.4f} struct={train['loss_struct']:.4f})"
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
            "optimizer": self.optimizer.state_dict(),
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
