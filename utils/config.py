"""Single-file YAML configuration loader.

The training pipeline is driven by one consolidated config file
(``config/config.yaml``).  This module provides a thin wrapper around
``yaml.safe_load`` plus a deep-merge helper for CLI overrides.

Example
-------
>>> cfg = load_config("config/config.yaml")
>>> cfg["model"]["base_model_path"]
'stabilityai/stable-diffusion-2-base'
>>> cfg["weather_prompt"]["use_weather_prompt"]
True
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = dict(base)
    for k, v in override.items():
        if (
            k in merged
            and isinstance(merged[k], dict)
            and isinstance(v, dict)
        ):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge an arbitrary number of dict-style configs."""
    merged: Dict[str, Any] = {}
    for c in configs:
        merged = _deep_merge(merged, c)
    return merged


def apply_dotted_overrides(
    cfg: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply ``a.b.c = value`` style overrides without mutating ``cfg``."""
    out: Dict[str, Any] = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()
    }
    for dotted, value in overrides.items():
        keys = dotted.split(".")
        cur = out
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = value
    return out


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the unified YAML config and (optionally) apply overrides.

    Parameters
    ----------
    config_path
        Path to the consolidated config file (``config/config.yaml``).
    overrides
        Optional dict of dotted-key overrides, e.g.
        ``{"weather_prompt.use_weather_prompt": False}``.
    """
    config_path = os.path.abspath(config_path)
    cfg = _read_yaml(config_path)
    if overrides:
        cfg = apply_dotted_overrides(cfg, overrides)
    return cfg


def print_config(cfg: Dict[str, Any]) -> None:
    """Pretty-print a (possibly nested) config dict."""
    import json
    try:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
    except TypeError:
        print(str(cfg))