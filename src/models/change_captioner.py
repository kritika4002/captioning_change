"""
Change Captioning Model — End-to-end architecture for disaster
change captioning from bi-temporal satellite imagery.

Architecture Overview:
  ┌────────────────────────────────────────────────────┐
  │  VisionChangeEncoder                                │
  │  (Frozen CLIP ViT + Trainable Cross-Attention)     │
  │                                                     │
  │  pre_image, post_image  →  change_features [B,N,D] │
  └───────────────┬────────────────────────────────────┘
                  │
           ┌──────▼──────┐
           │  Projection  │  (visual tokens → LM embedding space)
           └──────┬──────┘
                  │
  ┌───────────────▼────────────────────────────────────┐
  │  GPT-2 Language Decoder (with QLoRA adapters)       │
  │                                                     │
  │  [visual_tokens, caption_tokens]  →  logits [B,L,V] │
  └────────────────────────────────────────────────────┘
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .vision_encoder import VisionChangeEncoder

logger = logging.getLogger(__name__)


class VisualProjection(nn.Module):
    """
    Projects visual change features into the language model's
    embedding space.

    Adds a learnable [CLS]-like token and positional information
    before feeding into the decoder.
    """

    def __init__(
        self,
        visual_dim: int,
        lm_dim: int,
        num_visual_tokens: int = 197,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, lm_dim),
            nn.GELU(),
            nn.LayerNorm(lm_dim),
            nn.Dropout(dropout),
        )

        # Learnable visual position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_visual_tokens, lm_dim) * 0.02
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [B, N, D_visual]

        Returns:
            projected: [B, N, D_lm]
        """
        projected = self.proj(visual_features)  # [B, N, D_lm]

        # Add positional embeddings (truncate/pad if needed)
        seq_len = projected.size(1)
        pos = self.pos_embed[:, :seq_len, :]
        projected = projected + pos

        return projected


class ChangeCaptioningModel(nn.Module):
    """
    End-to-end model for generating change captions from satellite
    image pairs.

    Modules:
        1. VisionChangeEncoder — extracts change-aware visual features
        2. VisualProjection    — maps to LM embedding space
        3. GPT-2 + LoRA       — generates captions autoregressively

    Training:
        - CLIP ViT is frozen
        - GPT-2 base weights are frozen
        - Only LoRA adapters + projection layers are trained
    """

    def __init__(
        self,
        # Vision encoder config
        clip_model_name: str = "openai/clip-vit-base-patch16",
        vision_hidden: int = 512,
        num_heads: int = 8,
        num_cross_attn_layers: int = 2,
        freeze_clip: bool = True,
        # Decoder config
        decoder_name: str = "gpt2",
        max_length: int = 128,
        # LoRA config
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
    ):
        super().__init__()

        self.max_length = max_length

        # ── Vision Change Encoder ──
        self.vision_encoder = VisionChangeEncoder(
            clip_model_name=clip_model_name,
            hidden_size=vision_hidden,
            num_heads=num_heads,
            num_cross_attn_layers=num_cross_attn_layers,
            freeze_clip=freeze_clip,
        )

        # ── Language Decoder (GPT-2 with LoRA) ──
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_name)
        lm_hidden = self.decoder.config.n_embd  # 768 for GPT-2

        # Apply LoRA adapters
        if lora_target_modules is None:
            lora_target_modules = ["c_attn", "c_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.decoder = get_peft_model(self.decoder, lora_config)

        # ── Visual Projection ──
        num_visual_tokens = self.vision_encoder.get_num_tokens()
        self.visual_projection = VisualProjection(
            visual_dim=vision_hidden,
            lm_dim=lm_hidden,
            num_visual_tokens=num_visual_tokens,
        )

        # Log trainable parameters
        self._log_params()

    def _log_params(self):
        """Log trainable vs total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        logger.info(
            f"Model Parameters:\n"
            f"  Total:     {total:>12,}\n"
            f"  Trainable: {trainable:>12,}  ({100*trainable/total:.2f}%)\n"
            f"  Frozen:    {frozen:>12,}  ({100*frozen/total:.2f}%)"
        )

    def forward(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        caption_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            pre_image:      [B, 3, H, W]
            post_image:     [B, 3, H, W]
            caption_ids:    [B, L]  tokenised caption
            attention_mask: [B, L]  attention mask

        Returns:
            dict with:
                - logits:       [B, N+L, V]  prediction logits
                - pred_damage:  [B, 4]       damage distribution
                - change_cls:   [B, D]       change embedding
        """
        # ── 1. Encode visual changes ──
        vision_out = self.vision_encoder(pre_image, post_image)
        change_features = vision_out["change_features"]  # [B, N, D]
        pred_damage = vision_out["pred_damage"]           # [B, 4]
        change_cls = vision_out["change_cls"]             # [B, D]

        # ── 2. Project to LM space ──
        visual_embeds = self.visual_projection(change_features)  # [B, N, D_lm]
        batch_size = visual_embeds.size(0)
        num_vis_tokens = visual_embeds.size(1)

        # ── 3. Get text embeddings ──
        text_embeds = self.decoder.get_input_embeddings()(caption_ids)  # [B, L, D_lm]

        # ── 4. Concatenate: [visual_tokens | caption_tokens] ──
        inputs_embeds = torch.cat(
            [visual_embeds, text_embeds], dim=1
        )  # [B, N+L, D_lm]

        # Build attention mask: visual tokens always attended
        visual_mask = torch.ones(
            batch_size, num_vis_tokens,
            device=caption_ids.device, dtype=attention_mask.dtype
            if attention_mask is not None
            else torch.long,
        )
        if attention_mask is not None:
            full_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            text_mask = torch.ones_like(caption_ids)
            full_mask = torch.cat([visual_mask, text_mask], dim=1)

        # ── 5. Decoder forward ──
        decoder_out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
        )

        logits = decoder_out.logits  # [B, N+L, V]

        return {
            "logits": logits,
            "pred_damage": pred_damage,
            "change_cls": change_cls,
        }

    @torch.no_grad()
    def generate_caption(
        self,
        pre_image: torch.Tensor,
        post_image: torch.Tensor,
        tokenizer,
        max_length: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        beam_size: int = 5,
    ) -> str:
        """
        Generate a caption for a single image pair (inference).

        Args:
            pre_image:   [1, 3, H, W]
            post_image:  [1, 3, H, W]
            tokenizer:   HuggingFace tokenizer
            max_length:  Maximum tokens to generate
            temperature: Sampling temperature
            top_k:       Top-k filtering
            top_p:       Nucleus sampling threshold
            beam_size:   Number of beams (if > 1, beam search)

        Returns:
            Generated caption string.
        """
        self.eval()

        # Encode visual changes
        vision_out = self.vision_encoder(pre_image, post_image)
        change_features = vision_out["change_features"]

        # Project to LM space
        visual_embeds = self.visual_projection(change_features)  # [1, N, D]

        # Start with BOS token
        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        input_ids = torch.tensor(
            [[bos_token_id]], device=pre_image.device
        )

        # Autoregressive generation
        generated = []
        for step in range(max_length):
            text_embeds = self.decoder.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat(
                [visual_embeds, text_embeds], dim=1
            )

            outputs = self.decoder(inputs_embeds=inputs_embeds)
            next_token_logits = outputs.logits[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token], dim=1)

        caption = tokenizer.decode(generated, skip_special_tokens=True)
        return caption.strip()

    def get_trainable_params(self):
        """Yield only trainable parameters (for optimizer)."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param

    def save_adapter(self, save_dir: str):
        """Save only the LoRA adapter weights."""
        self.decoder.save_pretrained(save_dir)
        logger.info(f"LoRA adapter saved to {save_dir}")

    def load_adapter(self, adapter_dir: str):
        """Load LoRA adapter weights."""
        from peft import PeftModel
        self.decoder = PeftModel.from_pretrained(
            self.decoder.base_model, adapter_dir
        )
        logger.info(f"LoRA adapter loaded from {adapter_dir}")
