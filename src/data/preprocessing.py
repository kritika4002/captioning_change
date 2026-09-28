"""
Image Preprocessing — Utilities for satellite image preparation.

Handles:
  - Loading and resizing satellite imagery (pre/post pairs)
  - Normalisation for CLIP vision encoder
  - Data augmentation for training robustness
  - Tile stitching (xBD images are split into 4 parts)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

from PIL import Image

import torch
from torchvision import transforms


# ── CLIP normalisation statistics ──
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ImagePreprocessor:
    """
    Preprocesses satellite imagery for the change captioning pipeline.

    Supports two modes:
      - train: with data augmentation (random flip, colour jitter, etc.)
      - eval: deterministic centre-crop + normalise
    """

    def __init__(
        self,
        image_size: int = 224,
        mode: str = "train",
        mean: Tuple[float, ...] = CLIP_MEAN,
        std: Tuple[float, ...] = CLIP_STD,
    ):
        """
        Args:
            image_size: Target spatial resolution (square).
            mode:       "train" for augmented, "eval" for deterministic.
            mean:       Per-channel normalisation mean.
            std:        Per-channel normalisation std.
        """
        self.image_size = image_size
        self.mode = mode

        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply the preprocessing pipeline to a PIL Image."""
        return self.transform(image)

    def load_and_preprocess(self, image_path: str) -> torch.Tensor:
        """Load an image from disk and apply preprocessing."""
        img = Image.open(image_path).convert("RGB")
        return self(img)

    def load_image_pair(
        self, pre_path: str, post_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and preprocess a bi-temporal image pair.

        Args:
            pre_path:  Path to pre-disaster image.
            post_path: Path to post-disaster image.

        Returns:
            (pre_tensor, post_tensor) both of shape [3, H, W].
        """
        pre_img = self.load_and_preprocess(pre_path)
        post_img = self.load_and_preprocess(post_path)
        return pre_img, post_img


def stitch_tiles(tile_paths: List[str], grid: Tuple[int, int] = (2, 2)) -> Image.Image:
    """
    Stitch multiple image tiles back into a single image.

    The xBD dataset splits 1024×1024 images into 4 parts (512×512 each),
    named *_part1.png through *_part4.png in a 2×2 grid.

    Args:
        tile_paths: List of paths to tile images (must be in order).
        grid:       (rows, cols) of the tile grid.

    Returns:
        Stitched PIL Image.
    """
    rows, cols = grid
    tiles = [Image.open(p).convert("RGB") for p in tile_paths]

    if len(tiles) != rows * cols:
        raise ValueError(
            f"Expected {rows * cols} tiles, got {len(tiles)}"
        )

    tile_w, tile_h = tiles[0].size
    stitched = Image.new("RGB", (cols * tile_w, rows * tile_h))

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        stitched.paste(tile, (c * tile_w, r * tile_h))

    return stitched


def find_image_tiles(
    images_dir: str, sample_id: str, phase: str = "post_disaster"
) -> List[str]:
    """
    Find all tile parts for a given sample in the images directory.

    Args:
        images_dir: Path to the images directory.
        sample_id:  Sample identifier (e.g., "hurricane-harvey_00000000").
        phase:      "pre_disaster" or "post_disaster".

    Returns:
        Sorted list of tile file paths.
    """
    img_dir = Path(images_dir)
    pattern = f"{sample_id}_{phase}_part*.png"
    tiles = sorted(img_dir.glob(pattern))
    return [str(t) for t in tiles]
