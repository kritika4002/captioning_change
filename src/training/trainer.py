"""
Trainer — Handles the complete training loop for the Change Captioning
model, including mixed-precision, gradient accumulation, checkpointing,
periodic evaluation, and early stopping.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_scheduler, AutoTokenizer

from src.models.change_captioner import ChangeCaptioningModel
from src.models.losses import StructuralSemanticAlignmentLoss
from src.evaluation.metrics import CaptionEvaluator

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training engine for the disaster change captioning model.

    Features:
        - Mixed precision (FP16) training via torch AMP
        - Gradient accumulation for effective large batch sizes
        - Cosine / linear learning rate scheduling with warmup
        - Periodic validation with caption-quality metrics
        - Early stopping on validation loss
        - Checkpoint saving & resumption
        - TensorBoard logging
    """

    def __init__(
        self,
        model: ChangeCaptioningModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        tokenizer_name: str = "gpt2",
    ):
        """
        Args:
            model:        The ChangeCaptioningModel instance.
            train_loader: Training DataLoader.
            val_loader:   Validation DataLoader.
            config:       Configuration dictionary (from YAML).
            tokenizer_name: HuggingFace tokenizer name.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device
        self.device = torch.device(
            config.get("project", {}).get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model.to(self.device)

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.lr = train_cfg.get("learning_rate", 2e-4)
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.warmup_steps = train_cfg.get("warmup_steps", 500)
        self.use_fp16 = train_cfg.get("fp16", True) and self.device.type == "cuda"
        self.save_every = train_cfg.get("save_every", 5)
        self.eval_every = train_cfg.get("eval_every", 1)
        self.patience = train_cfg.get("early_stopping_patience", 7)

        # Directories
        self.output_dir = Path(train_cfg.get("output_dir", "./outputs"))
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "./checkpoints"))
        self.log_dir = Path(train_cfg.get("log_dir", "./logs"))
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Loss function (SSAL — our novel contribution)
        loss_cfg = config.get("loss", {})
        self.criterion = StructuralSemanticAlignmentLoss(
            lambda_ce=loss_cfg.get("lambda_ce", 1.0),
            lambda_structural=loss_cfg.get("lambda_structural", 0.3),
            lambda_temporal=loss_cfg.get("lambda_temporal", 0.2),
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
            damage_embed_dim=loss_cfg.get("damage_embed_dim", 128),
        ).to(self.device)

        # Optimizer (only trainable params)
        self.optimizer = torch.optim.AdamW(
            model.get_trainable_params(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        num_training_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        scheduler_type = train_cfg.get("scheduler", "cosine")
        self.scheduler = get_scheduler(
            scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_fp16)

        # Evaluator
        eval_cfg = config.get("evaluation", {})
        self.evaluator = CaptionEvaluator(
            metrics=eval_cfg.get("metrics", ["bleu", "meteor", "rouge_l"])
        )

        # Tokenizer (for generation during eval)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tracking
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.training_history = []
        self.global_step = 0

    # ──────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────

    def train(self) -> Dict:
        """
        Run the complete training loop.

        Returns:
            Training history dict.
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"FP16: {self.use_fp16}")
        logger.info(f"Gradient accumulation: {self.grad_accum_steps}")

        for epoch in range(1, self.epochs + 1):
            # ── Train one epoch ──
            train_metrics = self._train_epoch(epoch)

            # ── Evaluate ──
            if epoch % self.eval_every == 0:
                val_metrics = self._validate(epoch)
            else:
                val_metrics = {}

            # ── Log ──
            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": self.scheduler.get_last_lr()[0],
            }
            self.training_history.append(epoch_record)

            self._log_epoch(epoch, train_metrics, val_metrics)

            # ── Checkpointing ──
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

            # ── Early stopping ──
            val_loss = val_metrics.get("loss", float("inf"))
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
                logger.info(f"  ★ New best val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(patience={self.patience})"
                    )
                    break

        # Save final model
        self._save_checkpoint(epoch, is_best=False, tag="final")
        self._save_history()

        logger.info("Training complete!")
        return self.training_history

    # ──────────────────────────────────────────
    # Single epoch
    # ──────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_struct = 0.0
        total_temp = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            pre_img = batch["pre_image"].to(self.device)
            post_img = batch["post_image"].to(self.device)
            caption_ids = batch["caption_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            damage_vec = batch["damage_vector"].to(self.device)
            severity = batch["severity_score"].to(self.device)

            # Forward pass
            with autocast(enabled=self.use_fp16):
                outputs = self.model(
                    pre_image=pre_img,
                    post_image=post_img,
                    caption_ids=caption_ids,
                    attention_mask=attn_mask,
                )

                # Prepare labels (shift for causal LM)
                # Visual tokens don't have labels → set to -100
                num_vis = outputs["logits"].size(1) - caption_ids.size(1)
                vis_labels = torch.full(
                    (caption_ids.size(0), num_vis),
                    -100,
                    device=self.device,
                    dtype=caption_ids.dtype,
                )
                labels = torch.cat([vis_labels, caption_ids], dim=1)

                # Compute SSAL loss
                loss_dict = self.criterion(
                    logits=outputs["logits"],
                    labels=labels,
                    pred_damage=outputs["pred_damage"],
                    gt_damage=damage_vec,
                    gt_severity=severity,
                )
                loss = loss_dict["loss"] / self.grad_accum_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate metrics
            total_loss += loss_dict["loss"].item()
            total_ce += loss_dict["loss_ce"].item()
            total_struct += loss_dict["loss_struct"].item()
            total_temp += loss_dict["loss_temp"].item()
            num_batches += 1

        avg = lambda x: round(x / max(num_batches, 1), 4)
        return {
            "loss": avg(total_loss),
            "loss_ce": avg(total_ce),
            "loss_struct": avg(total_struct),
            "loss_temp": avg(total_temp),
        }

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_refs = []

        for batch in self.val_loader:
            pre_img = batch["pre_image"].to(self.device)
            post_img = batch["post_image"].to(self.device)
            caption_ids = batch["caption_ids"].to(self.device)
            attn_mask = batch["attention_mask"].to(self.device)
            damage_vec = batch["damage_vector"].to(self.device)
            severity = batch["severity_score"].to(self.device)

            outputs = self.model(
                pre_image=pre_img,
                post_image=post_img,
                caption_ids=caption_ids,
                attention_mask=attn_mask,
            )

            num_vis = outputs["logits"].size(1) - caption_ids.size(1)
            vis_labels = torch.full(
                (caption_ids.size(0), num_vis), -100,
                device=self.device, dtype=caption_ids.dtype,
            )
            labels = torch.cat([vis_labels, caption_ids], dim=1)

            loss_dict = self.criterion(
                logits=outputs["logits"],
                labels=labels,
                pred_damage=outputs["pred_damage"],
                gt_damage=damage_vec,
                gt_severity=severity,
            )

            total_loss += loss_dict["loss"].item()
            num_batches += 1

            # Decode predictions and references for caption metrics
            pred_ids = outputs["logits"][:, num_vis:, :].argmax(dim=-1)
            for i in range(pred_ids.size(0)):
                pred_text = self.tokenizer.decode(
                    pred_ids[i], skip_special_tokens=True
                )
                ref_text = self.tokenizer.decode(
                    caption_ids[i], skip_special_tokens=True
                )
                all_preds.append(pred_text)
                all_refs.append(ref_text)

        # Compute caption metrics
        val_loss = round(total_loss / max(num_batches, 1), 4)
        caption_metrics = self.evaluator.evaluate(all_preds, all_refs)

        return {"loss": val_loss, **caption_metrics}

    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────

    def _log_epoch(
        self, epoch: int, train: Dict, val: Dict
    ):
        """Log epoch metrics."""
        parts = [f"Epoch {epoch:>3d}/{self.epochs}"]
        parts.append(f"train_loss={train['loss']:.4f}")
        parts.append(f"[CE={train['loss_ce']:.4f}")
        parts.append(f"struct={train['loss_struct']:.4f}")
        parts.append(f"temp={train['loss_temp']:.4f}]")

        if val:
            parts.append(f"| val_loss={val.get('loss', 0):.4f}")
            if "bleu_4" in val:
                parts.append(f"BLEU-4={val['bleu_4']:.4f}")
            if "meteor" in val:
                parts.append(f"METEOR={val['meteor']:.4f}")
            if "rouge_l" in val:
                parts.append(f"ROUGE-L={val['rouge_l']:.4f}")

        logger.info("  ".join(parts))

    def _save_checkpoint(
        self, epoch: int, is_best: bool = False, tag: str = ""
    ):
        """Save a training checkpoint."""
        name = "best" if is_best else (tag or f"epoch_{epoch}")
        save_dir = self.checkpoint_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.model.save_adapter(str(save_dir / "lora_adapter"))

        # Save training state
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(state, save_dir / "training_state.pt")

        # Save non-LoRA trainable weights (vision projection, cross-attn, etc.)
        trainable_state = {
            k: v for k, v in self.model.state_dict().items()
            if "lora" not in k.lower()
            and any(
                p.data_ptr() == v.data_ptr()
                for p in self.model.parameters()
                if p.requires_grad
            )
        }
        torch.save(trainable_state, save_dir / "trainable_weights.pt")

        logger.info(f"Checkpoint saved: {save_dir}")

    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
