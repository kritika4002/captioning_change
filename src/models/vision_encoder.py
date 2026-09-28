"""
Vision Change Encoder — Extracts and compares visual features from
bi-temporal satellite image pairs.

Architecture:
  1. Frozen CLIP ViT encodes both pre- and post-disaster images
  2. Feature differencing captures pixel-level changes
  3. Cross-attention module refines change representations
  4. Outputs a fused change embedding for the language decoder
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig


class CrossAttentionBlock(nn.Module):
    """
    Multi-head cross-attention for comparing pre/post feature maps.

    The post-disaster features attend to the pre-disaster features,
    enabling the model to identify what has changed.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query:     Post-disaster features [B, N, D]
            key_value: Pre-disaster features  [B, N, D]

        Returns:
            Change-aware features [B, N, D]
        """
        # Cross-attention: post attends to pre
        attn_out, _ = self.cross_attn(
            query=query, key=key_value, value=key_value
        )
        x = self.norm1(query + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class VisionChangeEncoder(nn.Module):
    """
    Encodes bi-temporal satellite imagery into change-aware embeddings.

    Pipeline:
        pre_image  ─► CLIP ViT ─► pre_features  ─┐
                                                   ├─► Diff + CrossAttn ─► change_emb
        post_image ─► CLIP ViT ─► post_features ─┘

    The CLIP backbone is frozen; only the projection and cross-attention
    layers are trainable.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch16",
        hidden_size: int = 512,
        num_heads: int = 8,
        num_cross_attn_layers: int = 2,
        dropout: float = 0.1,
        freeze_clip: bool = True,
    ):
        super().__init__()

        # ── Frozen CLIP Vision Encoder ──
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_model_name)
        clip_hidden = self.clip_vision.config.hidden_size  # 768 for ViT-B/16

        if freeze_clip:
            for param in self.clip_vision.parameters():
                param.requires_grad = False
            self.clip_vision.eval()

        # ── Feature projection (CLIP hidden → our hidden) ──
        self.pre_proj = nn.Linear(clip_hidden, hidden_size)
        self.post_proj = nn.Linear(clip_hidden, hidden_size)

        # ── Difference projection ──
        self.diff_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Cross-attention stack ──
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_cross_attn_layers)
        ])

        # ── Fusion: combine diff + cross-attn output ──
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        # ── Severity head (auxiliary, used by SSAL loss) ──
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 damage classes
            nn.Softmax(dim=-1),
        )

        self.hidden_size = hidden_size

    def forward(
        self, pre_image: torch.Tensor, post_image: torch.Tensor
    ) -> dict:
        """
        Args:
            pre_image:  [B, 3, H, W]  pre-disaster image batch
            post_image: [B, 3, H, W]  post-disaster image batch

        Returns:
            dict with:
                - change_features: [B, N, D]  sequence of change tokens
                - change_cls:      [B, D]     pooled change embedding
                - pred_damage:     [B, 4]     predicted damage distribution
        """
        # Extract CLIP features (last hidden state)
        with torch.no_grad() if not any(p.requires_grad for p in self.clip_vision.parameters()) else torch.enable_grad():
            pre_out = self.clip_vision(pixel_values=pre_image)
            post_out = self.clip_vision(pixel_values=post_image)

        pre_features = pre_out.last_hidden_state    # [B, N, 768]
        post_features = post_out.last_hidden_state  # [B, N, 768]

        # Project to working dimension
        pre_proj = self.pre_proj(pre_features)    # [B, N, D]
        post_proj = self.post_proj(post_features)  # [B, N, D]

        # ── Feature differencing ──
        diff = self.diff_proj(post_proj - pre_proj)  # [B, N, D]

        # ── Cross-attention refinement ──
        cross = post_proj
        for layer in self.cross_attn_layers:
            cross = layer(query=cross, key_value=pre_proj)

        # ── Fusion ──
        fused = self.fusion(
            torch.cat([diff, cross], dim=-1)  # [B, N, 2D] → [B, N, D]
        )

        # CLS token (first token) as pooled representation
        change_cls = fused[:, 0, :]  # [B, D]

        # Auxiliary: predict damage distribution
        pred_damage = self.severity_head(change_cls)  # [B, 4]

        return {
            "change_features": fused,
            "change_cls": change_cls,
            "pred_damage": pred_damage,
        }

    def get_num_tokens(self) -> int:
        """Number of visual tokens output per image (depends on ViT patch size)."""
        # ViT-B/16 with 224×224 input: (224/16)² + 1 = 197
        return (self.clip_vision.config.image_size // self.clip_vision.config.patch_size) ** 2 + 1
