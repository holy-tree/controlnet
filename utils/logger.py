"""Lightweight logger + seeding utilities."""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Optional

import numpy as np
import torch


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Initialise root logger with a consistent format.

    Logs go to stderr by default and optionally to ``log_file`` if provided.
    """
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)
    # remove duplicate handlers when re-initialised in the same process
    for h in list(root.handlers):
        root.removeHandler(h)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def seed_everything(seed: int) -> None:
    """Seed Python / NumPy / PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)