"""Multi-weather image restoration dataset.

Expected directory layout (matches ``config/config.yaml``)::

    <data_root>/
      rain/
        train/
          LQ/   # low-quality (degraded) images
          GT/   # ground-truth (clean) images (filenames must match LQ/)
        test/
          LQ/
          GT/
      snow/
        train/{LQ,GT}
        test/{LQ,GT}
      haze/
        train/{LQ,GT}
        test/{LQ,GT}

The dataset returns ``(lq_tensor, gt_tensor, weather_label)`` for each item;
the trainer is responsible for building the text prompt.

Because the dataset does **not** ship a dedicated validation set, the
``test`` split is reused for periodic validation during training (see
``train.py``).  If you want stricter hygiene, split ``train`` yourself and
keep ``test`` only for the final report.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class WeatherRestorationDataset(Dataset):
    """Paired (LQ, GT, weather) dataset.

    The directory walk is *weather-first*: ``<data_root>/<weather>/<split>/{LQ,GT}/``.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",                 # "train" | "test"
        lq_subdir: str = "LQ",
        gt_subdir: str = "GT",
        train_subdir: str = "train",
        test_subdir: str = "test",
        image_extensions: Sequence[str] = SUPPORTED_EXTS,
        image_size: int = 512,
        random_flip: bool = True,
        random_crop: bool = True,
        max_samples: Optional[int] = None,
        samples_per_weather: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size

        # ------------------------------------------------------------------
        # Validate that the requested split is one of the known ones.
        # ------------------------------------------------------------------
        known_splits = {train_subdir, test_subdir}
        if split not in known_splits:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of: {sorted(known_splits)}. "
                f"Update dataset.{train_subdir}/{test_subdir} in config.yaml if you "
                f"want to use different folder names."
            )

        # ------------------------------------------------------------------
        # Discover (LQ, GT) pairs grouped by weather class.
        # ------------------------------------------------------------------
        if not os.path.isdir(data_root):
            layout = (
                f"{data_root}/<weather>/<train|test>/"
                f"{{'{lq_subdir}', '{gt_subdir}'}}/."
            )
            raise FileNotFoundError(
                f"Dataset root not found: {data_root}. Expected layout: {layout}"
            )

        exts = tuple(e.lower() for e in image_extensions)
        # Walk the filesystem grouped by weather class. Accumulating per
        # bucket first lets us apply an optional per-weather cap below, which
        # keeps the dataset balanced when raw folder counts are skewed.
        samples_by_weather: Dict[str, List[Tuple[str, str, str]]] = {}

        for weather in sorted(os.listdir(data_root)):
            weather_dir = os.path.join(data_root, weather)
            if not os.path.isdir(weather_dir):
                continue
            split_dir = os.path.join(weather_dir, split)
            if not os.path.isdir(split_dir):
                # Weather class does not contain the requested split - skip.
                continue
            lq_dir = os.path.join(split_dir, lq_subdir)
            gt_dir = os.path.join(split_dir, gt_subdir)
            if not (os.path.isdir(lq_dir) and os.path.isdir(gt_dir)):
                continue

            bucket: List[Tuple[str, str, str]] = []
            for lq_path in sorted(_list_images(lq_dir, exts)):
                fname = os.path.basename(lq_path)
                gt_path = os.path.join(gt_dir, fname)
                if not os.path.isfile(gt_path):
                    # Fall back to filename stem match (handles .png vs .jpg).
                    stem, _ = os.path.splitext(fname)
                    matches = _list_images(gt_dir, exts, stem=stem)
                    if not matches:
                        continue
                    gt_path = matches[0]
                bucket.append((lq_path, gt_path, weather))
            if bucket:
                samples_by_weather[weather] = bucket

        if not samples_by_weather:
            raise RuntimeError(
                f"No (LQ, GT) pairs found under {data_root}/<weather>/{split}/. "
                f"Check that filenames match between '{lq_subdir}/' and '{gt_subdir}/'."
            )

        # Optional per-weather sample cap (deterministic: keep first N of
        # each bucket by sorted filename). Set via YAML to balance classes.
        if samples_per_weather is not None and samples_per_weather > 0:
            for weather in list(samples_by_weather.keys()):
                bucket = samples_by_weather[weather]
                if len(bucket) > samples_per_weather:
                    samples_by_weather[weather] = bucket[:samples_per_weather]

        # Flatten back into a single list in sorted weather order so dataset
        # indexing is deterministic w.r.t. OS file ordering.
        self.samples: List[Tuple[str, str, str]] = []
        for weather in sorted(samples_by_weather):
            self.samples.extend(samples_by_weather[weather])

        # Optional overall cap (legacy max_samples parameter, applied after
        # the per-weather cap as a final global truncation).
        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

        # ------------------------------------------------------------------
        # Transforms
        # ------------------------------------------------------------------
        if split == train_subdir and random_crop:
            from .transforms import PairedRandomCrop, PairedRandomFlip
            self.transforms: List[Callable] = [PairedRandomCrop(image_size)]
            if random_flip:
                self.transforms.append(PairedRandomFlip(p=0.5))
        else:
            from .transforms import _CenterCropResize
            self.transforms = [_CenterCropResize(image_size)]

        from .transforms import _ToTensorBoth
        self.transforms.append(_ToTensorBoth())

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        lq_path, gt_path, weather = self.samples[idx]
        lq_img = Image.open(lq_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        for t in self.transforms:
            lq_img, gt_img = t(lq_img, gt_img)

        return {
            "lq": lq_img,                  # float tensor, [-1, 1], CHW
            "gt": gt_img,                  # float tensor, [-1, 1], CHW
            "weather": weather,            # str (e.g. "rain")
            "lq_path": lq_path,            # for debugging
        }

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def weather_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for _, _, w in self.samples:
            dist[w] = dist.get(w, 0) + 1
        return dist


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _list_images(
    folder: str,
    exts: Sequence[str],
    stem: Optional[str] = None,
) -> List[str]:
    files: List[str] = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, f"*{ext}")))
    if stem is not None:
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] == stem]
    return sorted(files)


# -----------------------------------------------------------------------------
# Dataloader factory
# -----------------------------------------------------------------------------
def create_dataloader(
    data_root: str,
    split: str = "train",                  # "train" | "test"
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    image_extensions: Sequence[str] = SUPPORTED_EXTS,
    lq_subdir: str = "LQ",
    gt_subdir: str = "GT",
    train_subdir: str = "train",
    test_subdir: str = "test",
    random_flip: bool = True,
    random_crop: bool = True,
    max_samples: Optional[int] = None,
    samples_per_weather: Optional[int] = None,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Build a DataLoader for the weather restoration dataset.

    Parameters
    ----------
    shuffle
        Defaults to ``True`` for the train split and ``False`` for test.
    samples_per_weather
        Optional per-weather-class cap (e.g. 1000 keeps 1000 images per
        ``rain`` / ``snow`` / ``haze`` bucket). Deterministic (first N by
        sorted filename). ``None`` keeps everything per class.
    """
    dataset = WeatherRestorationDataset(
        data_root=data_root,
        split=split,
        lq_subdir=lq_subdir,
        gt_subdir=gt_subdir,
        train_subdir=train_subdir,
        test_subdir=test_subdir,
        image_extensions=image_extensions,
        image_size=image_size,
        random_flip=random_flip,
        random_crop=random_crop,
        max_samples=max_samples,
        samples_per_weather=samples_per_weather,
    )

    if shuffle is None:
        shuffle = (split == train_subdir)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(split == train_subdir),
    )