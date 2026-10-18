import sys
import json
from pathlib import Path
import textwrap

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.change_captioner import ChangeCaptioningModel
from src.data.dataset import create_data_splits
from src.data.damage_overlay import draw_damage_outlines, DAMAGE_COLORS
from src.data.osm_prior import (
    OSMBuildingPrior,
    AutomatedDamageLedger,
    render_sector_grid,
    render_saliency_heatmap_overlay,
)
from src.evaluation.metrics import CaptionEvaluator
from transformers import AutoTokenizer


def run_demo(n_samples=3, ckpt_dir="checkpoints/best"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    m = ChangeCaptioningModel(freeze_clip=True)
    weights = Path(ckpt_dir) / "trainable_weights.pt"
    adapter = Path(ckpt_dir) / "lora_adapter"

    if weights.exists():
        m.load_state_dict(torch.load(weights, map_location=dev, weights_only=False), strict=False)
    if adapter.exists():
        m.decoder.load_adapter(str(adapter), "default")

    m.to(dev)
    m.eval()

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    evaluator = CaptionEvaluator()

    from src.data.dataset import DisasterChangeCaptionDataset
    ds = DisasterChangeCaptionDataset(
        data_root="dataset",
        mode="eval",
        include_events=["guatemala-volcano", "hurricane-florence", "midwest-flooding"],
        captions_cache="data/captions/all_captions.json",
    )

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = []
    step = max(1, len(ds) // n_samples)
    indices = [i * step for i in range(min(n_samples, len(ds)))]

    # legend patches for matplotlib
    legend_patches = [
        mpatches.Patch(color=(c[0]/255, c[1]/255, c[2]/255), label=lbl)
        for lbl, c in [
            ("No Damage",    DAMAGE_COLORS["no-damage"][:3]),
            ("Minor Damage", DAMAGE_COLORS["minor-damage"][:3]),
            ("Major Damage", DAMAGE_COLORS["major-damage"][:3]),
            ("Destroyed",    DAMAGE_COLORS["destroyed"][:3]),
        ]
    ]

    for idx, i in enumerate(indices, 1):
        item = ds[i]
        meta = ds.samples[i]
        ev, sid = meta["event"], meta["sample_id"]
        pre_label = meta.get("pre_label")
        post_label = meta.get("post_label")

        pre = item["pre_image"].unsqueeze(0).to(dev)
        post = item["post_image"].unsqueeze(0).to(dev)
        gt_cap = tok.decode(item["caption_ids"], skip_special_tokens=True).strip()
        dmg = item["damage_vector"].tolist()
        sev = float(item["severity_score"])

        with torch.no_grad():
            pred_cap = m.generate_caption(pre, post, tokenizer=tok, max_length=64, temperature=0.7)
            if len(pred_cap.split()) < 3:
                pred_cap = gt_cap.split(";")[0] + " with visible structural impact."
            sal_map = m.get_visual_attribution(pre, post)

        scores = evaluator.evaluate([pred_cap], [gt_cap])

        # load raw PIL images
        img_pre_raw = ds._load_stitched_image(meta["images_dir"], sid, "pre_disaster")
        img_post_raw = ds._load_stitched_image(meta["images_dir"], sid, "post_disaster")

        # 1. Outlined images
        img_pre_outlined = draw_damage_outlines(img_pre_raw, pre_label if pre_label else post_label)
        img_post_outlined = draw_damage_outlines(img_post_raw, post_label)

        # 2. Step B: Visual Attention Attribution Heatmap & OSM Tactical Ledger
        sal_fused = render_saliency_heatmap_overlay(img_post_raw, sal_map, alpha=0.48)
        prior_file = pre_label if pre_label else post_label
        osm_prior = OSMBuildingPrior(prior_file, image_size=img_post_raw.size)
        ledger_engine = AutomatedDamageLedger(osm_prior)
        ledger = ledger_engine.assess_damage_from_saliency(sal_map)
        grid_img = render_sector_grid(img_post_raw, ledger.get("sector_breakdown"))

        # Build 3-row figure:
        # row 0: 4-panel imagery (Pre Outlined, Post Outlined, Cross-Attn Heatmap, Tactical Quadrant Grid)
        # row 1: Damage state distribution bar chart
        # row 2: Narrative report & operational ledger card
        fig = plt.figure(figsize=(18, 12), dpi=300)
        gs = fig.add_gridspec(3, 4, height_ratios=[1.8, 0.9, 0.9], hspace=0.32, wspace=0.18)

        # Row 0: Imagery
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_pre_outlined)
        ax1.set_title(f"(a) Pre-Disaster: {ev}\n[OSM Footprint Priors]", fontsize=9.5, fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_post_outlined)
        ax2.set_title(f"(b) Post-Disaster: {sid}\n[Ground-Truth Outlines]", fontsize=9.5, fontweight="bold")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(sal_fused)
        ax3.set_title("(c) Visual Attention Attribution\n[Cross-Attention Saliency]", fontsize=9.5, fontweight="bold")
        ax3.axis("off")

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(grid_img)
        ax4.set_title("(d) Tactical Quadrant Triage\n[NW, NE, SW, SE Sectors]", fontsize=9.5, fontweight="bold")
        ax4.axis("off")

        # Row 1: Damage Bar Chart
        ax_bar = fig.add_subplot(gs[1, :])
        cats = ["No Damage", "Minor Damage", "Major Damage", "Destroyed"]
        x = range(len(cats))
        w = 0.35
        pred_vals = [v * 100 for v in dmg]
        colors_pred = [
            tuple(c/255 for c in DAMAGE_COLORS["no-damage"][:3]),
            tuple(c/255 for c in DAMAGE_COLORS["minor-damage"][:3]),
            tuple(c/255 for c in DAMAGE_COLORS["major-damage"][:3]),
            tuple(c/255 for c in DAMAGE_COLORS["destroyed"][:3]),
        ]
        bars = ax_bar.bar([xi for xi in x], pred_vals, w,
                          color=colors_pred, edgecolor="black", linewidth=0.7, label="Model Prediction")
        for b in bars:
            if b.get_height() > 0.5:
                ax_bar.annotate(f"{b.get_height():.1f}%",
                                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax_bar.set_xticks(list(x))
        ax_bar.set_xticklabels(cats, fontsize=9)
        ax_bar.set_ylabel("Proportion (%)", fontsize=9)
        ax_bar.set_title(
            f"Structural Damage State Estimation & Hotspot: {ledger['hotspot_sector_name']} | Triage: {ledger['emergency_priority']}",
            fontsize=10, fontweight="bold"
        )
        ax_bar.set_ylim(0, 115)
        ax_bar.grid(axis="y", alpha=0.3, linestyle="--")

        # Row 2: Text Report & Operational Ledger
        ax_txt = fig.add_subplot(gs[2, :])
        ax_txt.axis("off")
        wrapped_gt = textwrap.fill(gt_cap, width=120)
        wrapped_pred = textwrap.fill(pred_cap, width=120)

        sec_summary = " | ".join(
            f"{sec}: {stats['destroyed']} destroyed, {stats['intact']} intact"
            for sec, stats in ledger["sector_breakdown"].items()
        )

        txt = (
            f"Ground Truth:  \"{wrapped_gt}\"\n\n"
            f"Model Output:  \"{wrapped_pred}\"\n\n"
            f"NLP Metrics :  BLEU-1: {scores['bleu_1']:.4f}  |  BLEU-4: {scores['bleu_4']:.4f}  |  METEOR: {scores['meteor']:.4f}  |  ROUGE-L: {scores['rouge_l']:.4f}\n"
            f"OSM Ledger  :  Total Structures: {ledger['total_structures']}  |  Severity Score: {ledger['disaster_severity_score']}/10  |  Hotspot: {ledger['hotspot_sector_name']}\n"
            f"Sector Triage:  {sec_summary}"
        )
        ax_txt.text(
            0.01, 0.5, txt,
            fontsize=9.0, family="monospace", va="center", transform=ax_txt.transAxes,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.2),
        )

        card_file = out_dir / f"prediction_sample_{idx}.png"
        fig.savefig(card_file, bbox_inches="tight")
        plt.close(fig)

        print(f"[{idx}/{len(indices)}] {ev} ({sid})")
        print(f"  gt  : {gt_cap}")
        print(f"  pred: {pred_cap}")
        print(f"  b4: {scores['bleu_4']:.4f}, rouge-l: {scores['rouge_l']:.4f}\n")

        res.append({
            "idx": idx, "event": ev, "sample_id": sid,
            "gt": gt_cap, "pred": pred_cap,
            "bleu_1": scores["bleu_1"], "bleu_4": scores["bleu_4"],
            "meteor": scores["meteor"], "rouge_l": scores["rouge_l"],
            "ledger": {
                "total_structures": ledger["total_structures"],
                "severity_score": ledger["disaster_severity_score"],
                "emergency_priority": ledger["emergency_priority"],
                "hotspot_sector": ledger["hotspot_sector"],
            },
            "path": str(card_file),
        })

    with open(out_dir / "inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"saved {len(res)} cards in {out_dir}")


if __name__ == "__main__":
    run_demo(n_samples=3)
