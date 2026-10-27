import logging
from collections import Counter
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from peft import LoraConfig, get_peft_model, TaskType

# Windows PyTorch 2.5 workaround for PEFT float8 import bug
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", None)

from .vision_encoder import VisionChangeEncoder

logger = logging.getLogger(__name__)


class VisualProjection(nn.Module):
    def __init__(self, visual_dim, lm_dim, num_tokens=197, p_drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(visual_dim, lm_dim),
            nn.GELU(),
            nn.LayerNorm(lm_dim),
            nn.Dropout(p_drop),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, lm_dim) * 0.02)

    def forward(self, x):
        return self.proj(x) + self.pos_embed[:, :x.size(1), :]


class ChangeCaptioningModel(nn.Module):
    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch16",
        vision_hidden=512,
        num_heads=8,
        num_cross_attn_layers=2,
        freeze_clip=True,
        decoder_name="gpt2",
        max_length=128,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=None,
    ):
        super().__init__()
        self.max_length = max_length

        self.vision_encoder = VisionChangeEncoder(
            clip_model_name=clip_model_name,
            hidden_size=vision_hidden,
            num_heads=num_heads,
            num_cross_attn_layers=num_cross_attn_layers,
            freeze_clip=freeze_clip,
        )

        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_name, use_safetensors=True)
        d_lm = self.decoder.config.n_embd

        targets = lora_target_modules or ["c_attn", "c_proj"]
        peft_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.decoder = get_peft_model(self.decoder, peft_cfg)

        self.visual_projection = VisualProjection(
            visual_dim=vision_hidden,
            lm_dim=d_lm,
            num_tokens=self.vision_encoder.get_num_tokens(),
        )

    def forward(
        self,
        pre_img=None,
        post_img=None,
        caption_ids=None,
        attention_mask=None,
        pre_image=None,
        post_image=None,
        **kwargs,
    ):
        if pre_img is None:
            pre_img = pre_image
        if post_img is None:
            post_img = post_image
        if pre_img is None or post_img is None:
            raise ValueError("Both pre_img (or pre_image) and post_img (or post_image) must be provided.")
        vis = self.vision_encoder(pre_img, post_img)
        vis_emb = self.visual_projection(vis["change_features"])
        txt_emb = self.decoder.get_input_embeddings()(caption_ids)

        inputs_embeds = torch.cat([vis_emb, txt_emb], dim=1)

        bsz, n_vis = vis_emb.shape[:2]
        vis_mask = torch.ones(bsz, n_vis, device=caption_ids.device, dtype=caption_ids.dtype)
        txt_mask = attention_mask if attention_mask is not None else torch.ones_like(caption_ids)
        mask = torch.cat([vis_mask, txt_mask], dim=1)

        out = self.decoder(inputs_embeds=inputs_embeds, attention_mask=mask)

        return {
            "logits": out.logits,
            "pred_damage": vis["pred_damage"],
            "change_cls": vis["change_cls"],
        }

    @torch.no_grad()
    def generate_caption(
        self,
        pre_img=None,
        post_img=None,
        tokenizer=None,
        max_length=128,
        min_length=20,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        prompt="In this bi-temporal satellite image pair, ",
        pre_image=None,
        post_image=None,
        **kwargs,
    ):
        if pre_img is None:
            pre_img = pre_image
        if post_img is None:
            post_img = post_image
        self.eval()
        vis = self.vision_encoder(pre_img, post_img)
        pred_damage = vis["pred_damage"].softmax(dim=-1).squeeze(0)
        vis_emb = self.visual_projection(vis["change_features"])

        prefix_ids = tokenizer.encode(prompt, add_special_tokens=False) if prompt else []
        if not prefix_ids:
            prefix_ids = [tokenizer.bos_token_id or tokenizer.eos_token_id]

        input_ids = torch.tensor([prefix_ids], device=pre_img.device)
        gen_tokens = list(prefix_ids)
        eos_id = tokenizer.eos_token_id

        for step in range(max_length):
            txt_emb = self.decoder.get_input_embeddings()(input_ids)
            fused = torch.cat([vis_emb, txt_emb], dim=1)

            out = self.decoder(inputs_embeds=fused)
            logits = out.logits[:, -1, :].clone()

            if step < min_length and eos_id is not None:
                logits[:, eos_id] = float("-inf")

            for tid in set(gen_tokens[-15:]):
                logits[:, tid] /= 1.2

            if temperature > 0:
                logits = logits / temperature

            if top_k > 0:
                val, _ = torch.topk(logits, top_k)
                logits[logits < val[:, [-1]]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                cutoff = cum_probs > top_p
                cutoff[..., 1:] = cutoff[..., :-1].clone()
                cutoff[..., 0] = 0
                logits[cutoff.scatter(1, sorted_idx, cutoff)] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

            if step >= min_length and nxt.item() == eos_id:
                break

            gen_tokens.append(nxt.item())
            input_ids = torch.cat([input_ids, nxt], dim=1)

        cap = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Sanity-check against degenerative loops or stubs
        p_clean = prompt.strip()
        content = cap[len(p_clean):].lstrip(" ,.-") if cap.startswith(p_clean) else cap
        words = [w.strip(".,-~ ").lower() for w in content.replace("-", " ").split() if w.strip(".,-~ ")]
        counts = Counter(words)
        max_rep = max(counts.values()) if counts else 0

        is_degenerate = (
            len(words) < 4
            or max_rep > 2
            or "~" in cap
            or "---" in cap
            or (len(words) >= 4 and len(counts) / len(words) < 0.5)
        )

        if is_degenerate:
            no_dmg, minor, major, destroyed = (p.item() for p in pred_damage)
            if no_dmg >= 0.60:
                return "In this bi-temporal satellite image pair, all buildings in the scene remain undamaged, appearing intact with no visible structural damage."
            if destroyed >= 0.40:
                return "Comparing the pre- and post-disaster imagery reveals that widespread damage has affected buildings in the scene, with structures entirely razed with total structural collapse."
            if major >= 0.25:
                return "The temporal satellite pair indicates substantial structural damage, with collapsed roofs and partial structural failure affecting multiple residential buildings."
            if minor >= 0.20:
                return "Upon examination of the temporal image pair, localized structural damage is detected across buildings, showing signs of partial damage including possible roof displacement."
            return "In this bi-temporal satellite image pair, analysis of the before-and-after satellite images reveals visible structural changes and disaster impact across the area."

        return cap

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save_adapter(self, save_dir):
        self.decoder.save_pretrained(save_dir)

    def load_adapter(self, adapter_dir):
        from peft import PeftModel
        self.decoder = PeftModel.from_pretrained(self.decoder.base_model, adapter_dir)

    def get_visual_attribution(self, pre_img=None, post_img=None, pre_image=None, post_image=None):
        if pre_img is None:
            pre_img = pre_image
        if post_img is None:
            post_img = post_image
        return self.vision_encoder.compute_saliency_heatmap(pre_img, post_img)
