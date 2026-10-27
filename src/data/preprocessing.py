from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ImagePreprocessor:
    def __init__(self, image_size=224, mode="train", mean=CLIP_MEAN, std=CLIP_STD):
        self.image_size = image_size
        self.mode = mode

        if isinstance(image_size, (tuple, list)):
            resize_size = (int(image_size[0]), int(image_size[1]))
        else:
            resize_size = (int(image_size), int(image_size))

        if mode == "train":
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __call__(self, img):
        return self.transform(img)

    def load_and_preprocess(self, path):
        return self(Image.open(path).convert("RGB"))

    def load_image_pair(self, pre_path, post_path):
        return self.load_and_preprocess(pre_path), self.load_and_preprocess(post_path)


def stitch_tiles(tile_paths, grid=(2, 2)):
    # xbd cuts 1024x1024 images into 4 512x512 quadrants
    rows, cols = grid
    tiles = [Image.open(p).convert("RGB") for p in tile_paths]
    if len(tiles) != rows * cols:
        raise ValueError(f"expected {rows * cols} tiles, got {len(tiles)}")

    tw, th = tiles[0].size
    stitched = Image.new("RGB", (cols * tw, rows * th))
    for idx, t in enumerate(tiles):
        r, c = idx // cols, idx % cols
        stitched.paste(t, (c * tw, r * th))
    return stitched


def find_image_tiles(images_dir, sample_id, phase="post_disaster"):
    idir = Path(images_dir)
    pattern = f"{sample_id}_{phase}_part*.png"
    tiles = [str(t) for t in sorted(idir.glob(pattern))]
    if not tiles:
        direct = idir / f"{sample_id}_{phase}.png"
        if direct.exists():
            return [str(direct)]
        for ext in [".jpg", ".jpeg", ".tif", ".tiff"]:
            direct_ext = idir / f"{sample_id}_{phase}{ext}"
            if direct_ext.exists():
                return [str(direct_ext)]
    return tiles

