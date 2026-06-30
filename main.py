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


def _to_minus_one_one(pil_image: Image.Image, image_size: int, crop: bool = True) -> torch.Tensor:
    """Convert PIL image to [-1, 1] tensor, optionally cropping to square.

    * For validation/training, we use ``crop=True`` (center crop) to match the
      square training patches.
    * For inference, we use ``crop=False`` (full-image resize) so that we restore
      the entire image area instead of throwing away the borders.
    """
    if crop:
        w, h = pil_image.size
        scale = image_size / min(w, h)
        new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
        pil_image = pil_image.resize((new_w, new_h), Image.BICUBIC)
        left = (new_w - image_size) // 2
        top = (new_h - image_size) // 2
        pil_image = pil_image.crop((left, top, left + image_size, top + image_size))
    else:
        # Resize entire image to square without cropping.
        pil_image = pil_image.resize((image_size, image_size), Image.BICUBIC)
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
    """Run batch-folder inference. All params come from YAML.

    The test directory is ``infer.input_dir``; the optional
    ``infer.num_test_images`` cap limits how many images get processed
    (per-weather when ``weather_from_subdir`` is True, otherwise global).
    """
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
        controlnet_hint_range=model_cfg.get("controlnet_hint_range", "zero_one"),
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
    # Dispatch: only batch-folder mode is supported.
    # ------------------------------------------------------------------
    if not infer_cfg.get("input_dir"):
        raise ValueError(
            "Inference requires 'infer.input_dir' (the test directory) "
            "and 'infer.output_dir' (where restored images are written)."
        )
    _infer_batch(model, infer_cfg, cfg["dataset"], device, dtype,
                 image_size, guidance_scale, num_inference_steps)


def _infer_batch(model, infer_cfg, dataset_cfg, device, dtype, image_size,
                 guidance_scale, num_inference_steps) -> None:
    """Run restoration on the first N test images.

    Directory contract (driven by ``dataset.*`` subdir names):

    * ``LQ_DIR = <input_dir>/<test_subdir>/<lq_subdir>`` — degraded inputs
    * ``GT_DIR = <input_dir>/<test_subdir>/<gt_subdir>`` — clean references

    Output goes to a single timestamped folder with prefixed filenames::

        <output_dir>/test/<YYYYMMDD_HHMMSS>/
            LQ_rain-0000.png    <- copied LQ input
            GT_rain-0000.png    <- copied GT reference (by stem match)
            PRED_rain-0000.png  <- model-restored output
            LQ_rain-0001.png
            GT_rain-0001.png
            PRED_rain-0001.png
            ...
    """
    import shutil
    from datetime import datetime
    from glob import glob

    input_dir: str = infer_cfg["input_dir"]
    output_dir: str = infer_cfg["output_dir"]
    weather: str = infer_cfg.get("weather", "rain")
    num_test_images = infer_cfg.get("num_test_images")

    # ------------------------------------------------------------------
    # Resolve LQ / GT directories from the dataset subdir convention.
    # ------------------------------------------------------------------
    test_subdir: str = dataset_cfg.get("test_subdir", "test")
    lq_subdir: str = dataset_cfg.get("lq_subdir", "LQ")
    gt_subdir: str = dataset_cfg.get("gt_subdir", "GT")

    lq_dir = os.path.join(input_dir, test_subdir, lq_subdir)
    gt_dir = os.path.join(input_dir, test_subdir, gt_subdir)

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not os.path.isdir(lq_dir):
        raise FileNotFoundError(
            f"LQ directory not found: {lq_dir}\n"
            f"(derived from input_dir={input_dir!r}, "
            f"test_subdir={test_subdir!r}, lq_subdir={lq_subdir!r})"
        )
    if not os.path.isdir(gt_dir):
        # GT missing is non-fatal — we just skip the GT copy and log it.
        logger.warning("GT directory not found (continuing without GT copies): %s",
                       gt_dir)
        gt_dir = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Discover LQ images (non-recursive: LQ_DIR is a flat folder of images).
    # ------------------------------------------------------------------
    exts = tuple(e.lower() for e in infer_cfg.get("image_extensions",
                                                  cfg_default_image_exts()))
    images: List[str] = []
    for ext in exts:
        images.extend(glob(os.path.join(lq_dir, f"*{ext}")))
    images = sorted(set(images))

    if not images:
        logger.warning("No images found in %s (exts=%s)", lq_dir, exts)
        return

    # ------------------------------------------------------------------
    # Optional cap on how many images to actually restore.  null/0/<=0 = no cap.
    # ------------------------------------------------------------------
    if num_test_images is not None and num_test_images > 0:
        images = images[:num_test_images]

    if not images:
        logger.warning("No images left after applying num_test_images=%s cap.",
                       num_test_images)
        return

    # ------------------------------------------------------------------
    # Create a single per-run timestamped output directory; LQ/GT/PRED
    # all land in this folder with a prefix on the filename so the user
    # can sort/glob them easily.
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, "test", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    logger.info(
        "Restoring %d image(s) | weather=%s | LQ=%s | GT=%s -> %s",
        len(images), weather, lq_dir, gt_dir or "<missing>", run_dir,
    )

    # ------------------------------------------------------------------
    # Per-image loop: write LQ_<name>, GT_<name>, PRED_<name> side by side.
    # ------------------------------------------------------------------
    for img_path in images:
        fname = os.path.basename(img_path)
        stem, _ = os.path.splitext(fname)
        prefixed = lambda tag: f"{tag}_{fname}"  # noqa: E731

        # Always save the LQ (preserves original filename even if extension
        # differs from GT, e.g. lq.png vs gt.jpg).
        shutil.copy2(img_path, os.path.join(run_dir, prefixed("LQ")))

        # Find a matching GT by stem (handles .png vs .jpg mismatches).
        gt_src: Optional[str] = None
        if gt_dir is not None:
            direct = os.path.join(gt_dir, fname)
            if os.path.isfile(direct):
                gt_src = direct
            else:
                stem_matches = glob(os.path.join(gt_dir, f"{stem}.*"))
                if stem_matches:
                    gt_src = sorted(stem_matches)[0]
            if gt_src is None:
                logger.warning("No matching GT for %s in %s", fname, gt_dir)
            else:
                shutil.copy2(gt_src, os.path.join(run_dir, prefixed("GT")))

        # Run the model and save PRED.
        pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image_size = pil.size         # remember the input size
        # We resize the entire image to square (crop=False) during inference,
        # so when we scale PRED back to (orig_w, orig_h) at the end, the
        # entire image is restored without border loss or aspect ratio stretching!
        tensor = _to_minus_one_one(pil, image_size, crop=False).to(device, dtype=dtype)
        out = model.sample(
            degraded_pixel_values=tensor,
            weather_labels=[weather],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        pred_pil = _tensor_to_pil(out[0])                   # PIL, image_size × image_size
        # Optional: stretch PRED back to the original input resolution
        # so all three of LQ/GT/PRED line up pixel-for-pixel.
        # The trade-off is that the model only saw the center crop, so
        # upscaling interpolates details that the model never produced.
        if infer_cfg.get("output_to_original_size", True):
            pred_pil = pred_pil.resize((orig_w, orig_h), Image.BICUBIC)
        pred_pil.save(os.path.join(run_dir, prefixed("PRED")))
        logger.info(
            "[%s] %s  ->  LQ/GT/PRED saved (%dx%d)",
            weather, fname, orig_w, orig_h,
        )


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