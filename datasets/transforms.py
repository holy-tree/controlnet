"""Paired image transforms for (degraded, clean) supervision.

The model needs the degraded input *and* the clean ground-truth to be
spatially aligned and identically normalised, so every transform accepts
both tensors / PIL images at once and applies the *same* random operation
to both.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image


# -----------------------------------------------------------------------------
# PIL <-> tensor
# -----------------------------------------------------------------------------
class ToTensorMinusOneOne:
    """Convert a ``PIL.Image`` to a ``float32`` tensor in ``[-1, 1]``."""

    def __call__(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # HWC in [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)                 # CHW
        tensor = tensor * 2.0 - 1.0                                     # [-1, 1]
        return tensor


# -----------------------------------------------------------------------------
# Paired spatial augmentations
# -----------------------------------------------------------------------------
class PairedRandomFlip:
    """Random horizontal flip applied to both images with identical decision."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            degraded = degraded.transpose(Image.FLIP_LEFT_RIGHT)
            clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
        return degraded, clean


class PairedRandomCrop:
    """Random square crop of size ``size`` applied to both images."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = degraded.size
        if w < self.size or h < self.size:
            # Resize up if the image is smaller than the crop size.
            scale = max(self.size / w, self.size / h)
            new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
            degraded = degraded.resize((new_w, new_h), Image.BICUBIC)
            clean = clean.resize((new_w, new_h), Image.BICUBIC)
            w, h = degraded.size

        x = random.randint(0, w - self.size)
        y = random.randint(0, h - self.size)
        box = (x, y, x + self.size, y + self.size)
        return degraded.crop(box), clean.crop(box)


# -----------------------------------------------------------------------------
# Composition helpers
# -----------------------------------------------------------------------------
def build_train_transforms(image_size: int, random_flip: bool = True):
    """Compose a transform list for the training set."""
    transforms = [PairedRandomCrop(image_size)]
    if random_flip:
        transforms.append(PairedRandomFlip(p=0.5))
    transforms.append(_ToTensorBoth())
    return transforms


def build_val_transforms(image_size: int):
    """Compose a deterministic transform list for the validation set."""
    return [_CenterCropResize(image_size), _ToTensorBoth()]


# -----------------------------------------------------------------------------
# Composite PIL->Tensor transforms that take both images
# -----------------------------------------------------------------------------
class _ToTensorBoth:
    def __call__(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        converter = ToTensorMinusOneOne()
        return converter(degraded), converter(clean)


class _CenterCropResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        # Match ControlNet / SD2: force the short side then center-crop.
        w, h = degraded.size
        scale = self.size / min(w, h)
        new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
        degraded = degraded.resize((new_w, new_h), Image.BICUBIC)
        clean = clean.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - self.size) // 2
        top = (new_h - self.size) // 2
        box = (left, top, left + self.size, top + self.size)
        return degraded.crop(box), clean.crop(box)