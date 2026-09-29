import sys
import yaml
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_data_splits
from src.models.change_captioner import ChangeCaptioningModel
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg.get("training", {}).get("log_dir", "./logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f"{log_dir}/training.log")],
    )
    logger = logging.getLogger("train")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"config: {args.config} on {dev}")

    seed = cfg.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(seed)

    d_cfg = cfg.get("data", {})
    train_ds, val_ds, _ = create_data_splits(
        dataset_root=d_cfg.get("root_dir", "./dataset"),
        train_ratio=d_cfg.get("train_split", 0.8),
        val_ratio=d_cfg.get("val_split", 0.1),
        tokenizer_name=cfg.get("model", {}).get("decoder", {}).get("name", "gpt2"),
        max_length=d_cfg.get("max_caption_length", 128),
        image_size=d_cfg.get("image_size", 224),
        captions_cache=d_cfg.get("captions_cache", "./data/captions/all_captions.json"),
    )

    t_cfg = cfg.get("training", {})
    bs = t_cfg.get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=d_cfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=d_cfg.get("num_workers", 4), pin_memory=True)

    m_cfg = cfg.get("model", {})
    v_cfg = m_cfg.get("vision_encoder", {})
    dec_cfg = m_cfg.get("decoder", {})
    l_cfg = m_cfg.get("lora", {})
    c_cfg = m_cfg.get("change_detector", {})

    model = ChangeCaptioningModel(
        clip_model_name=v_cfg.get("name", "openai/clip-vit-base-patch16"),
        vision_hidden=c_cfg.get("hidden_size", 512),
        num_heads=c_cfg.get("num_heads", 8),
        num_cross_attn_layers=c_cfg.get("num_layers", 2),
        freeze_clip=v_cfg.get("freeze", True),
        decoder_name=dec_cfg.get("name", "gpt2"),
        max_length=dec_cfg.get("max_length", 128),
        lora_r=l_cfg.get("r", 16),
        lora_alpha=l_cfg.get("alpha", 32),
        lora_dropout=l_cfg.get("dropout", 0.05),
        lora_target_modules=l_cfg.get("target_modules", ["c_attn", "c_proj"]),
    )

    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
