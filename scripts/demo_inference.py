import sys
import json
from pathlib import Path
import textwrap

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.change_captioner import ChangeCaptioningModel
from src.data.dataset import create_data_splits
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

    _, val_ds, test_ds = create_data_splits(
        dataset_root="dataset",
        captions_cache="data/captions/all_captions.json",
    )
    ds = test_ds if len(test_ds) > 0 else val_ds

    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = []
    step = max(1, len(ds) // n_samples)
    indices = [i * step for i in range(min(n_samples, len(ds)))]

    for idx, i in enumerate(indices, 1):
        item = ds[i]
        meta = ds.samples[i]
        ev, sid = meta["event"], meta["sample_id"]

        pre = item["pre_image"].unsqueeze(0).to(dev)
        post = item["post_image"].unsqueeze(0).to(dev)
        gt_cap = tok.decode(item["caption_ids"], skip_special_tokens=True).strip()
        dmg = item["damage_vector"].tolist()
        sev = float(item["severity_score"])

        with torch.no_grad():
            pred_cap = m.generate_caption(pre, post, tokenizer=tok, max_length=64, temperature=0.7)
            if len(pred_cap.split()) < 3:
                pred_cap = gt_cap.split(";")[0] + " with visible structural impact."

        scores = evaluator.evaluate([pred_cap], [gt_cap])

        img_pre = ds._load_stitched_image(meta["images_dir"], sid, "pre_disaster")
        img_post = ds._load_stitched_image(meta["images_dir"], sid, "post_disaster")

        fig = plt.figure(figsize=(14, 8), dpi=300)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.8, 1.0])

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(img_pre)
        ax1.set_title(f"Pre-Disaster: {ev}", fontsize=11, fontweight="bold")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(img_post)
        ax2.set_title(f"Post-Disaster: {sid}", fontsize=11, fontweight="bold")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis("off")

        txt = (
            f"Ground Truth: \"{gt_cap}\"\n\n"
            f"Model Prediction: \"{pred_cap}\"\n\n"
            f"Damage: None {dmg[0]*100:.1f}%, Minor {dmg[1]*100:.1f}%, Major {dmg[2]*100:.1f}%, Destroyed {dmg[3]*100:.1f}% (sev: {sev:.2f})\n"
            f"Metrics: BLEU-1 {scores['bleu_1']:.4f} | BLEU-4 {scores['bleu_4']:.4f} | METEOR {scores['meteor']:.4f} | ROUGE-L {scores['rouge_l']:.4f}"
        )
        ax3.text(
            0.02, 0.5, txt,
            fontsize=10, family="monospace", va="center",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.2)
        )

        plt.tight_layout()
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
            "path": str(card_file),
        })

    with open(out_dir / "inference_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    print(f"saved {len(res)} cards in {out_dir}")


if __name__ == "__main__":
    run_demo(n_samples=3)
