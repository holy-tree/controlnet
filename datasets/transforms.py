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


class PairedResizeNative:
    """Resize short side to ``short_side``, round both dims to nearest multiple of 8.

    Preserves aspect ratio. Both H and W are guaranteed to be divisible by 8,
    which is required by the VAE / UNet pipeline.
    """

    def __init__(self, short_side: int = 512) -> None:
        self.short_side = short_side

    def __call__(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = degraded.size
        scale = self.short_side / min(w, h)
        new_w = int(w * scale + 0.5)
        new_h = int(h * scale + 0.5)
        new_w = max(8, new_w - new_w % 8)
        new_h = max(8, new_h - new_h % 8)
        degraded = degraded.resize((new_w, new_h), Image.BICUBIC)
        clean = clean.resize((new_w, new_h), Image.BICUBIC)
        return degraded, clean


class PairedCenterCropSquare:
    """Center-crop the image to a square of side ``min(W, H)``.

    Designed to be applied **after** :class:`PairedResizeNative`: the resize
    preserves aspect ratio, so the short side is exactly ``image_size`` and
    the long side is the multiple-of-8 rounded counterpart.  This transform
    trims the long side down to the short side by taking the central square,
    giving a square ``(image_size, image_size)`` tensor that is the canonical
    input shape for the SD2 / ControlNet pipeline.
    """

    def __call__(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = degraded.size
        crop = min(w, h)
        if w == h:
            return degraded, clean
        left = (w - crop) // 2
        top = (h - crop) // 2
        box = (left, top, left + crop, top + crop)
        return degraded.crop(box), clean.crop(box)


class PairedRandomCropSquare:
    """Random crop to a square of side ``min(W, H)``.

    Same as :class:`PairedCenterCropSquare` but the crop offset is sampled
    uniformly within the image.  Both images are cropped with the **same**
    random offset so LQ and GT remain spatially aligned.  Use this for the
    train split; for validation prefer the deterministic center crop.
    """

    def __call__(
        self,
        degraded: Image.Image,
        clean: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = degraded.size
        crop = min(w, h)
        if w == h:
            return degraded, clean
        x = random.randint(0, w - crop)
        y = random.randint(0, h - crop)
        box = (x, y, x + crop, y + crop)
        return degraded.crop(box), clean.crop(box)


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