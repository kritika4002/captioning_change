"""
PyTorch Dataset — Disaster Change Caption Dataset.

Loads bi-temporal satellite image pairs from the xBD dataset along with
their auto-generated change captions.  Provides everything needed by the
training loop: image tensors, tokenised captions, and damage metadata
(used by the custom SSAL loss function).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .preprocessing import (
    ImagePreprocessor,
    stitch_tiles,
    find_image_tiles,
)
from .caption_generator import CaptionGenerator, DAMAGE_LEVELS

logger = logging.getLogger(__name__)


class DisasterChangeCaptionDataset(Dataset):
    """
    PyTorch Dataset for disaster change captioning.

    Each sample contains:
        - pre_image:      Tensor [3, H, W]  (pre-disaster)
        - post_image:     Tensor [3, H, W]  (post-disaster)
        - caption_ids:    Tensor [L]         (tokenised caption)
        - attention_mask: Tensor [L]         (attention mask)
        - damage_vector:  Tensor [4]         (normalised counts for
                                              no/minor/major/destroyed)
        - severity_score: float              (aggregate severity 0-1)
    """

    def __init__(
        self,
        data_root: str,
        tokenizer_name: str = "gpt2",
        max_length: int = 128,
        image_size: int = 224,
        mode: str = "train",
        include_events: Optional[List[str]] = None,
        exclude_events: Optional[List[str]] = None,
        captions_cache: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Args:
            data_root:       Root path to the dataset directory.
            tokenizer_name:  HuggingFace tokenizer identifier.
            max_length:      Maximum caption token length.
            image_size:      Target image resolution.
            mode:            "train" or "eval".
            include_events:  Whitelist of disaster events (None = all).
            exclude_events:  Blacklist of disaster events.
            captions_cache:  Path to pre-generated captions JSON (faster).
            seed:            Random seed.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.max_length = max_length
        self.mode = mode

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Image preprocessor
        self.preprocessor = ImagePreprocessor(
            image_size=image_size, mode=mode
        )

        # Caption generator
        self.caption_gen = CaptionGenerator(seed=seed)

        # Build sample index
        self.samples = self._index_samples(
            include_events, exclude_events
        )

        # Optionally load pre-generated captions
        self._caption_cache: Dict[str, Dict] = {}
        if captions_cache and Path(captions_cache).exists():
            self._load_caption_cache(captions_cache)

        logger.info(
            f"Loaded {len(self.samples)} samples from "
            f"{len(set(s['event'] for s in self.samples))} events "
            f"(mode={mode})"
        )

    # ──────────────────────────────────────────
    # Dataset interface
    # ──────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # ── Load images ──
        pre_img = self._load_stitched_image(
            sample["images_dir"], sample["sample_id"], "pre_disaster"
        )
        post_img = self._load_stitched_image(
            sample["images_dir"], sample["sample_id"], "post_disaster"
        )

        pre_tensor = self.preprocessor(pre_img)
        post_tensor = self.preprocessor(post_img)

        # ── Generate or load caption ──
        cache_key = f"{sample['event']}_{sample['sample_id']}"
        if cache_key in self._caption_cache:
            caption_data = self._caption_cache[cache_key]
        else:
            caption_data = self.caption_gen.generate_caption(
                sample["post_label"], sample.get("pre_label")
            )

        caption_text = caption_data["caption"]

        # ── Tokenise caption ──
        encoding = self.tokenizer(
            caption_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # ── Damage vector (for SSAL loss) ──
        counts = caption_data.get("damage_counts", {})
        total = max(caption_data.get("total_buildings", 1), 1)
        damage_vector = torch.tensor(
            [
                counts.get("no-damage", 0) / total,
                counts.get("minor-damage", 0) / total,
                counts.get("major-damage", 0) / total,
                counts.get("destroyed", 0) / total,
            ],
            dtype=torch.float32,
        )

        severity = torch.tensor(
            caption_data.get("severity_score", 0.0), dtype=torch.float32
        )

        return {
            "pre_image": pre_tensor,
            "post_image": post_tensor,
            "caption_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "damage_vector": damage_vector,
            "severity_score": severity,
        }

    # ──────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────

    def _index_samples(
        self,
        include_events: Optional[List[str]],
        exclude_events: Optional[List[str]],
    ) -> List[Dict]:
        """
        Walk the dataset directory and index all valid sample pairs.
        """
        samples = []
        exclude = set(exclude_events or [])
        # Also exclude the xbd_subset meta-directory
        exclude.add("xbd_subset")

        for event_dir in sorted(self.data_root.iterdir()):
            if not event_dir.is_dir():
                continue

            event_name = event_dir.name

            if event_name in exclude:
                continue
            if include_events and event_name not in include_events:
                continue

            labels_dir = event_dir / "labels"
            images_dir = event_dir / "images"

            if not labels_dir.exists() or not images_dir.exists():
                logger.warning(f"Skipping {event_name}: missing labels/ or images/")
                continue

            # Find post-disaster label files
            for post_label in sorted(labels_dir.glob("*_post_disaster.json")):
                stem = post_label.stem.replace("_post_disaster", "")
                pre_label = labels_dir / f"{stem}_pre_disaster.json"

                # Check that corresponding image tiles exist
                post_tiles = find_image_tiles(str(images_dir), stem, "post_disaster")
                pre_tiles = find_image_tiles(str(images_dir), stem, "pre_disaster")

                if not post_tiles or not pre_tiles:
                    continue

                samples.append({
                    "event": event_name,
                    "sample_id": stem,
                    "images_dir": str(images_dir),
                    "post_label": str(post_label),
                    "pre_label": str(pre_label) if pre_label.exists() else None,
                })

        return samples

    def _load_stitched_image(
        self, images_dir: str, sample_id: str, phase: str
    ):
        """Load and stitch tile parts into a single PIL Image."""
        tiles = find_image_tiles(images_dir, sample_id, phase)

        if len(tiles) == 1:
            from PIL import Image
            return Image.open(tiles[0]).convert("RGB")
        elif len(tiles) == 4:
            return stitch_tiles(tiles)
        else:
            # Fallback: use the first tile
            from PIL import Image
            return Image.open(tiles[0]).convert("RGB")

    def _load_caption_cache(self, cache_path: str):
        """Load pre-generated captions from a JSON file."""
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_list = json.load(f)

        for item in cache_list:
            key = f"{item['event']}_{item['sample_id']}"
            self._caption_cache[key] = item

        logger.info(f"Loaded {len(self._caption_cache)} cached captions")


def create_data_splits(
    dataset_root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> Tuple[DisasterChangeCaptionDataset, DisasterChangeCaptionDataset, DisasterChangeCaptionDataset]:
    """
    Create train/val/test splits from the full dataset.

    Splits are done at the event level (not sample level) to prevent
    data leakage between splits.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    import random

    data_root = Path(dataset_root)
    all_events = sorted([
        d.name for d in data_root.iterdir()
        if d.is_dir() and d.name != "xbd_subset"
        and (d / "labels").exists() and (d / "images").exists()
    ])

    rng = random.Random(seed)
    rng.shuffle(all_events)

    n = len(all_events)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_events = all_events[:n_train]
    val_events = all_events[n_train : n_train + n_val]
    test_events = all_events[n_train + n_val :]

    logger.info(
        f"Split: train={len(train_events)} events, "
        f"val={len(val_events)} events, "
        f"test={len(test_events)} events"
    )

    train_ds = DisasterChangeCaptionDataset(
        data_root=dataset_root, mode="train",
        include_events=train_events, seed=seed, **kwargs
    )
    val_ds = DisasterChangeCaptionDataset(
        data_root=dataset_root, mode="eval",
        include_events=val_events, seed=seed, **kwargs
    )
    test_ds = DisasterChangeCaptionDataset(
        data_root=dataset_root, mode="eval",
        include_events=test_events, seed=seed, **kwargs
    )

    return train_ds, val_ds, test_ds
