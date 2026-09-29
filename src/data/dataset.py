import json
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .preprocessing import ImagePreprocessor, stitch_tiles, find_image_tiles
from .caption_generator import CaptionGenerator


class DisasterChangeCaptionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer_name="gpt2",
        max_length=128,
        image_size=224,
        mode="train",
        include_events=None,
        exclude_events=None,
        captions_cache=None,
        seed=42,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.max_length = max_length
        self.mode = mode

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.preprocessor = ImagePreprocessor(image_size=image_size, mode=mode)
        self.caption_gen = CaptionGenerator(seed=seed)
        self.samples = self._index_samples(include_events, exclude_events)

        self._caption_cache = {}
        self._cache = self._caption_cache
        if captions_cache and Path(captions_cache).exists():
            self._load_cache(captions_cache)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        pre_img = self._load_stitched_image(s["images_dir"], s["sample_id"], "pre_disaster")
        post_img = self._load_stitched_image(s["images_dir"], s["sample_id"], "post_disaster")

        key = f"{s['event']}_{s['sample_id']}"
        if key in self._cache:
            cap_data = self._cache[key]
        else:
            cap_data = self.caption_gen.generate_caption(s["post_label"], s.get("pre_label"))

        enc = self.tokenizer(
            cap_data["caption"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        counts = cap_data.get("damage_counts", {})
        total = max(cap_data.get("total_buildings", 1), 1)
        dmg_vec = torch.tensor(
            [
                counts.get("no-damage", 0) / total,
                counts.get("minor-damage", 0) / total,
                counts.get("major-damage", 0) / total,
                counts.get("destroyed", 0) / total,
            ],
            dtype=torch.float32,
        )

        return {
            "pre_image": self.preprocessor(pre_img),
            "post_image": self.preprocessor(post_img),
            "caption_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "damage_vector": dmg_vec,
            "severity_score": torch.tensor(cap_data.get("severity_score", 0.0), dtype=torch.float32),
        }

    def _index_samples(self, include_events, exclude_events):
        samples = []
        exclude = set(exclude_events or [])
        exclude.add("xbd_subset")

        for edir in sorted(self.data_root.iterdir()):
            if not edir.is_dir():
                continue
            ev = edir.name
            if ev in exclude:
                continue
            if include_events and ev not in include_events:
                continue

            ldir = edir / "labels"
            idir = edir / "images"
            if not ldir.exists() or not idir.exists():
                continue

            for post_lbl in sorted(ldir.glob("*_post_disaster.json")):
                sid = post_lbl.stem.replace("_post_disaster", "")
                pre_lbl = ldir / f"{sid}_pre_disaster.json"

                if not find_image_tiles(str(idir), sid, "post_disaster"):
                    continue
                if not find_image_tiles(str(idir), sid, "pre_disaster"):
                    continue

                samples.append({
                    "event": ev,
                    "sample_id": sid,
                    "images_dir": str(idir),
                    "post_label": str(post_lbl),
                    "pre_label": str(pre_lbl) if pre_lbl.exists() else None,
                })
        return samples

    def _load_stitched_image(self, images_dir, sample_id, phase):
        tiles = find_image_tiles(images_dir, sample_id, phase)
        if len(tiles) == 4:
            return stitch_tiles(tiles)
        if len(tiles) > 0:
            return Image.open(tiles[0]).convert("RGB")
        return Image.new("RGB", (512, 512), color="black")

    def _load_cache(self, path):
        p = Path(path)
        if p.is_dir():
            p = p / "all_captions.json"
        if not p.is_file():
            return
        with open(p, "r", encoding="utf-8") as f:
            for item in json.load(f):
                self._caption_cache[f"{item['event']}_{item['sample_id']}"] = item
                self._caption_cache[item["sample_id"]] = item


def create_data_splits(dataset_root, train_ratio=0.8, val_ratio=0.1, seed=42, **kwargs):
    droot = Path(dataset_root)
    all_events = sorted([
        d.name for d in droot.iterdir()
        if d.is_dir() and d.name != "xbd_subset" and (d / "labels").exists() and (d / "images").exists()
    ])

    rng = random.Random(seed)
    rng.shuffle(all_events)

    n = len(all_events)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ev = all_events[:n_train]
    val_ev = all_events[n_train:n_train + n_val]
    test_ev = all_events[n_train + n_val:]

    train_ds = DisasterChangeCaptionDataset(data_root=dataset_root, mode="train", include_events=train_ev, seed=seed, **kwargs)
    val_ds = DisasterChangeCaptionDataset(data_root=dataset_root, mode="eval", include_events=val_ev, seed=seed, **kwargs)
    test_ds = DisasterChangeCaptionDataset(data_root=dataset_root, mode="eval", include_events=test_ev, seed=seed, **kwargs)

    return train_ds, val_ds, test_ds
