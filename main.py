"""Project entry point.

Everything is configured from YAML — startup needs only::

    python main.py --config config/config.yaml

Optional CLI flags (override the YAML when supplied)::

    --mode {train,infer}    default: read from ``project.mode`` in YAML
    --override key=value    dotted-key override, repeatable

Train / infer dispatch is decided by ``project.mode`` in the YAML (or by
the ``--mode`` flag if provided).  All paths, weights, weather classes
and sampling parameters used by inference live under ``infer:`` in the
YAML — no extra CLI args required.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import torch
from PIL import Image

import train as train_mod
from datasets.transforms import ToTensorMinusOneOne
from models import ControlNetRestorationModel
from utils import (
    get_logger,
    load_config,
    print_config,
    seed_everything,
    setup_logging,
)


logger = get_logger("main")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_overrides(items: Optional[List[str]]):
    if not items:
        return {}
    out = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Override must be key=value, got: {raw}")
        k, v = raw.split("=", 1)
        k, v = k.strip(), v.strip()
        if v.lower() in {"true", "false"}:
            out[k] = (v.lower() == "true")
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _resolve_dtype(mp: str) -> torch.dtype:
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, torch.float32)


def _to_minus_one_one(pil_image: Image.Image, image_size: int) -> torch.Tensor:
    """Match the validation pipeline: short-side resize -> center crop -> tensor."""
    w, h = pil_image.size
    scale = image_size / min(w, h)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    pil_image = pil_image.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - image_size) // 2
    top = (new_h - image_size) // 2
    pil_image = pil_image.crop((left, top, left + image_size, top + image_size))
    return ToTensorMinusOneOne()(pil_image).unsqueeze(0)  # (1, 3, H, W)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = ((t.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


# -----------------------------------------------------------------------------
# Sub-commands
# -----------------------------------------------------------------------------
def cmd_train(cfg: dict) -> None:
    """Delegate to train.main() with the same effective argv."""
    sys.argv = [sys.argv[0]]
    sys.argv += ["--config", cfg["__config_path__"]]
    train_mod.main()


def cmd_infer(cfg: dict) -> None:
    """Run single-image or batch-folder inference. All params come from YAML."""
    project = cfg["project"]
    model_cfg = cfg["model"]
    weather_cfg = cfg["weather_prompt"]
    train_cfg = cfg["train"]
    infer_cfg = cfg["infer"]

    mp = project["mixed_precision"]
    dtype = _resolve_dtype(mp)
    device = torch.device(project["device"])

    if project.get("seed_everything", True):
        seed_everything(project["seed"])

    # ------------------------------------------------------------------
    # Resolve ControlNet checkpoint
    # ------------------------------------------------------------------
    ckpt_path = infer_cfg.get("ckpt_path") or model_cfg.get("controlnet_path")
    if not ckpt_path:
        raise ValueError(
            "Inference requires a ControlNet checkpoint. "
            "Set 'infer.ckpt_path' (or 'model.controlnet_path') in the YAML."
        )

    logger.info("Loading SD2 + ControlNet from base=%s | ckpt=%s",
                model_cfg["base_model_path"], ckpt_path)

    model = ControlNetRestorationModel(
        base_model_path=model_cfg["base_model_path"],
        controlnet_path=None,           # load randomly; we restore below
        weather_prompt_cfg=weather_cfg,
        device=str(device),
        dtype=dtype,
        enable_xformers=project["enable_xformers"],
        gradient_checkpointing=False,
    )
    if os.path.isdir(ckpt_path):
        model.load_controlnet(ckpt_path)
    elif os.path.isfile(ckpt_path):
        model.load_controlnet(os.path.dirname(ckpt_path) or ".")
    else:
        raise FileNotFoundError(f"ControlNet checkpoint not found: {ckpt_path}")

    image_size = infer_cfg.get("image_size", train_cfg.get("image_size", 512)
                               or cfg["dataset"]["image_size"])
    guidance_scale = infer_cfg.get("guidance_scale", train_cfg["guidance_scale"])
    num_inference_steps = infer_cfg.get("num_inference_steps", train_cfg["num_inference_steps"])

    # ------------------------------------------------------------------
    # Dispatch: single image vs. batch folder
    # ------------------------------------------------------------------
    if infer_cfg.get("input_path"):
        _infer_single(model, infer_cfg, device, dtype,
                      image_size, guidance_scale, num_inference_steps)
    elif infer_cfg.get("input_dir"):
        _infer_batch(model, infer_cfg, device, dtype,
                     image_size, guidance_scale, num_inference_steps)
    else:
        raise ValueError(
            "Inference YAML must define either 'input_path' (single image) "
            "or 'input_dir' (batch folder)."
        )


def _infer_single(model, infer_cfg, device, dtype, image_size,
                  guidance_scale, num_inference_steps) -> None:
    input_path = infer_cfg["input_path"]
    output_path = infer_cfg["output_path"]
    weather = infer_cfg["weather"]

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    pil = Image.open(input_path).convert("RGB")
    tensor = _to_minus_one_one(pil, image_size).to(device, dtype=dtype)
    out = model.sample(
        degraded_pixel_values=tensor,
        weather_labels=[weather],
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )
    _tensor_to_pil(out[0]).save(output_path)
    logger.info("Saved restored image -> %s", output_path)


def _infer_batch(model, infer_cfg, device, dtype, image_size,
                 guidance_scale, num_inference_steps) -> None:
    from datasets.weather_restoration import _list_images, SUPPORTED_EXTS

    input_dir = infer_cfg["input_dir"]
    output_dir = infer_cfg["output_dir"]
    weather_from_subdir = infer_cfg.get("weather_from_subdir", False)
    default_weather = infer_cfg.get("weather", "rain")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    exts = tuple(e.lower() for e in infer_cfg.get("image_extensions",
                                                  cfg_default_image_exts()))
    images = _list_images(input_dir, exts)
    if not images:
        logger.warning("No images found in %s", input_dir)
        return

    for img_path in images:
        if weather_from_subdir:
            # weather = immediate parent folder name
            weather = os.path.basename(os.path.dirname(img_path))
        else:
            weather = default_weather

        # Mirror the input layout under output_dir so the user can sanity-check
        # which image came from where.
        rel = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

        pil = Image.open(img_path).convert("RGB")
        tensor = _to_minus_one_one(pil, image_size).to(device, dtype=dtype)
        out = model.sample(
            degraded_pixel_values=tensor,
            weather_labels=[weather],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        _tensor_to_pil(out[0]).save(out_path)
        logger.info("[%s/%s] %s -> %s", weather, os.path.basename(img_path),
                    img_path, out_path)


def cfg_default_image_exts():
    return (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SD2 + ControlNet multi-weather restoration. "
                    "All parameters live in the YAML; --config is the only required flag."
    )
    p.add_argument("--config", type=str, default="config/config.yaml",
                   help="Path to the unified YAML config.")
    p.add_argument("--mode", choices=["train", "infer"], default=None,
                   help="Override project.mode from the YAML.")
    p.add_argument("--override", "-o", action="append", default=None,
                   help="Dotted-key override, e.g. weather_prompt.use_weather_prompt=false.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = _parse_overrides(args.override)

    cfg = load_config(args.config, overrides=overrides)
    cfg["__config_path__"] = os.path.abspath(args.config)

    setup_logging(log_file=os.path.join(
        cfg["project"]["output_dir"], cfg["project"]["name"], "train.log"
    ))
    if cfg["project"].get("print_config_on_start", True):
        logger.info("===== Loaded configuration (%s) =====", cfg["__config_path__"])
        print_config(cfg)
        logger.info("====================================")

    mode = args.mode or cfg["project"].get("mode", "train")
    logger.info("Mode: %s", mode)

    if mode == "train":
        cmd_train(cfg)
    elif mode == "infer":
        cmd_infer(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    main()