"""
Train — Main entry point for training the Change Captioning model.

Usage:
    python scripts/train.py --config configs/config.yaml

This script:
    1. Loads configuration from YAML
    2. Creates train/val/test data splits
    3. Initialises the model with LoRA adapters
    4. Runs the training loop with the SSAL loss
    5. Evaluates and saves results
"""

import sys
import yaml
import logging
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import create_data_splits
from src.models.change_captioner import ChangeCaptioningModel
from src.training.trainer import Trainer


def setup_logging(log_dir: str = "./logs"):
    """Configure logging for training."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"{log_dir}/training.log"),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train Disaster Change Captioning Model"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # ── Load config ──
    config = load_config(args.config)
    setup_logging(config.get("training", {}).get("log_dir", "./logs"))

    logger = logging.getLogger("train")
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ── Seed ──
    seed = config.get("project", {}).get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Data ──
    data_cfg = config.get("data", {})
    logger.info("Creating data splits...")

    train_ds, val_ds, test_ds = create_data_splits(
        dataset_root=data_cfg.get("root_dir", "./dataset"),
        train_ratio=data_cfg.get("train_split", 0.8),
        val_ratio=data_cfg.get("val_split", 0.1),
        tokenizer_name=config.get("model", {}).get("decoder", {}).get("name", "gpt2"),
        max_length=data_cfg.get("max_caption_length", 128),
        image_size=data_cfg.get("image_size", 224),
        captions_cache=data_cfg.get("captions_dir", None),
    )

    train_cfg = config.get("training", {})
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_ds)} samples, {len(train_loader)} batches")
    logger.info(f"Val:   {len(val_ds)} samples, {len(val_loader)} batches")
    logger.info(f"Test:  {len(test_ds)} samples")

    # ── Model ──
    model_cfg = config.get("model", {})
    vision_cfg = model_cfg.get("vision_encoder", {})
    decoder_cfg = model_cfg.get("decoder", {})
    lora_cfg = model_cfg.get("lora", {})
    change_cfg = model_cfg.get("change_detector", {})

    logger.info("Initialising model...")
    model = ChangeCaptioningModel(
        clip_model_name=vision_cfg.get("name", "openai/clip-vit-base-patch16"),
        vision_hidden=change_cfg.get("hidden_size", 512),
        num_heads=change_cfg.get("num_heads", 8),
        num_cross_attn_layers=change_cfg.get("num_layers", 2),
        freeze_clip=vision_cfg.get("freeze", True),
        decoder_name=decoder_cfg.get("name", "gpt2"),
        max_length=decoder_cfg.get("max_length", 128),
        lora_r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        lora_target_modules=lora_cfg.get("target_modules", ["c_attn", "c_proj"]),
    )

    # ── Trainer ──
    logger.info("Starting training with SSAL loss...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        tokenizer_name=decoder_cfg.get("name", "gpt2"),
    )

    # ── Train ──
    history = trainer.train()

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Results saved to: {train_cfg.get('output_dir', './outputs')}")


if __name__ == "__main__":
    main()
