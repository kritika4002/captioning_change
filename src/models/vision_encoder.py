import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, p_drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p_drop, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p_drop),
        )

    def forward(self, q, kv, return_weights=False):
        out, weights = self.attn(query=q, key=kv, value=kv, need_weights=return_weights)
        x = self.norm1(q + out)
        res = self.norm2(x + self.mlp(x))
        if return_weights:
            return res, weights
        return res


class VisionChangeEncoder(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch16",
        hidden_size=512,
        num_heads=8,
        num_cross_attn_layers=2,
        dropout=0.1,
        freeze_clip=True,
    ):
        super().__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained(clip_model_name, use_safetensors=True)
        d_clip = self.clip_vision.config.hidden_size

        if freeze_clip:
            for p in self.clip_vision.parameters():
                p.requires_grad = False
            self.clip_vision.eval()

        self.pre_proj = nn.Linear(d_clip, hidden_size)
        self.post_proj = nn.Linear(d_clip, hidden_size)

        self.diff_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_cross_attn_layers)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        # 4 damage classes: no-damage, minor, major, destroyed
        self.severity_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=-1),
        )

    def forward(self, pre_img, post_img):
        # clip feature extraction
        is_frozen = not any(p.requires_grad for p in self.clip_vision.parameters())
        with torch.no_grad() if is_frozen else torch.enable_grad():
            f_pre = self.clip_vision(pixel_values=pre_img).last_hidden_state
            f_post = self.clip_vision(pixel_values=post_img).last_hidden_state

        h_pre = self.pre_proj(f_pre)
        h_post = self.post_proj(f_post)

        diff = self.diff_proj(h_post - h_pre)

        cross = h_post
        for layer in self.layers:
            cross = layer(cross, h_pre)

        fused = self.fusion(torch.cat([diff, cross], dim=-1))
        cls_token = fused[:, 0, :]
        pred_damage = self.severity_head(cls_token)

        return {
            "change_features": fused,
            "change_cls": cls_token,
            "pred_damage": pred_damage,
        }

    def compute_saliency_heatmap(self, pre_img, post_img):
        """
        Computes 2D spatial change visual attention attribution heatmap over the 14x14 patch grid.
        Returns a normalized 2D numpy array [224, 224] in range [0, 1].
        """
        self.eval()
        with torch.no_grad():
            f_pre = self.clip_vision(pixel_values=pre_img).last_hidden_state
            f_post = self.clip_vision(pixel_values=post_img).last_hidden_state

            h_pre = self.pre_proj(f_pre)
            h_post = self.post_proj(f_post)

            diff = self.diff_proj(h_post - h_pre)

            # Exclude CLS token (index 0) to get 196 spatial patch tokens
            # Calculate difference activation magnitude
            diff_spatial = diff[:, 1:, :]
            diff_norm = torch.norm(diff_spatial, p=2, dim=-1)  # shape: (B, 196)

            # Extract cross-attention weights
            cross = h_post
            last_attn = None
            for layer in self.layers:
                cross, weights = layer(cross, h_pre, return_weights=True)
                last_attn = weights

            if last_attn is not None:
                # Saliency of cross-attention from post to pre patches: (B, 196)
                cross_weights = last_attn[:, 1:, 1:].mean(dim=1)
                sal = diff_norm * (1.0 + cross_weights)
            else:
                sal = diff_norm

            B = sal.shape[0]
            # Reshape to 14x14 grid
            sal_grid = sal.view(B, 1, 14, 14)

            # Upsample smoothly to 224x224
            sal_upsampled = F.interpolate(sal_grid, size=(224, 224), mode="bicubic", align_corners=False)
            
            # Min-Max normalization per image
            s_min = sal_upsampled.amin(dim=(2, 3), keepdim=True)
            s_max = sal_upsampled.amax(dim=(2, 3), keepdim=True)
            sal_norm = (sal_upsampled - s_min) / (s_max - s_min + 1e-8)

            return sal_norm.squeeze(0).squeeze(0).cpu().numpy()

    def get_num_tokens(self):
        # vit-b/16 on 224x224: (224//16)^2 + 1 = 197 tokens
        cfg = self.clip_vision.config
        return (cfg.image_size // cfg.patch_size) ** 2 + 1
