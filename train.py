"""Training entry point for the SD2 + ControlNet multi-weather pipeline.

Usage
-----

::

    python main.py                                   # uses config/config.yaml
    python main.py --config path/to/other.yaml
    python main.py weather_prompt.use_weather_prompt=false

The trainer deliberately stays dependency-light (no ``accelerate`` / no
``transformers.Trainer``) so the code is easy to read and adapt.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from glob import glob
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from datasets import create_dataloader
from models import ControlNetRestorationModel
from utils import (
    get_logger,
    load_config,
    print_config,
    seed_everything,
    setup_logging,
)


logger = get_logger("train")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_overrides(items: Optional[List[str]]) -> Dict[str, object]:
    """Parse ``key=value`` CLI args into a dict of dotted-key overrides.

    Type inference: int / float / bool / str (fallback).
    """
    out: Dict[str, object] = {}
    if not items:
        return out
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Override must be key=value, got: {raw}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.lower() in {"true", "false"}:
            out[key] = (value.lower() == "true")
        else:
            try:
                out[key] = int(value)
            except ValueError:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
    return out


def _resolve_dtype(mp: str) -> torch.dtype:
    return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(mp, torch.float32)


def _save_image_grid(images: torch.Tensor, ncols: int, path: str) -> None:
    """Save a batch of ``[-1, 1]`` images as a single grid PNG."""
    imgs = (images.clamp(-1, 1) + 1) / 2
    imgs = (imgs * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
    pil = [Image.fromarray(im) for im in imgs]
    if len(pil) == 0:
        return
    w, h = pil[0].size
    rows = math.ceil(len(pil) / ncols)
    grid = Image.new("RGB", (w * ncols, h * rows), color=(0, 0, 0))
    for i, im in enumerate(pil):
        r, c = divmod(i, ncols)
        grid.paste(im, (c * w, r * h))
    grid.save(path)


def _latest_checkpoint(output_dir: str) -> Optional[str]:
    ckpts = sorted(glob(os.path.join(output_dir, "checkpoint-*")))
    if not ckpts:
        return None
    return ckpts[-1]


def _prune_checkpoints(output_dir: str, keep: int) -> None:
    if keep <= 0:
        return
    ckpts = sorted(glob(os.path.join(output_dir, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
    for old in ckpts[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.project = cfg["project"]
        self.model_cfg = cfg["model"]
        self.dataset_cfg = cfg["dataset"]
        self.train_cfg = cfg["train"]
        self.weather_prompt_cfg = cfg["weather_prompt"]

        # ------------------------------------------------------------------
        # Output dirs
        # ------------------------------------------------------------------
        self.output_dir = os.path.join(
            self.project["output_dir"], self.project["name"]
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Device / dtype
        # ------------------------------------------------------------------
        if self.project["device"] == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.project["device"])
        self.dtype = _resolve_dtype(self.project["mixed_precision"])

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------
        logger.info("Building dataloaders ...")
        self.train_loader = create_dataloader(
            data_root=self.dataset_cfg["data_root"],
            split=self.dataset_cfg["train_subdir"],
            image_size=self.dataset_cfg["image_size"],
            batch_size=self.train_cfg["batch_size"],
            num_workers=self.dataset_cfg["num_workers"],
            image_extensions=self.dataset_cfg["image_extensions"],
            lq_subdir=self.dataset_cfg["lq_subdir"],
            gt_subdir=self.dataset_cfg["gt_subdir"],
            train_subdir=self.dataset_cfg["train_subdir"],
            test_subdir=self.dataset_cfg["test_subdir"],
            random_flip=self.dataset_cfg["random_flip"],
            random_crop=self.dataset_cfg["random_crop"],
            pin_memory=self.train_cfg["dataloader_pin_memory"],
        )
        # No dedicated validation set is shipped with this dataset, so we
        # reuse the "test" split for periodic monitoring during training.
        self.val_loader = create_dataloader(
            data_root=self.dataset_cfg["data_root"],
            split=self.dataset_cfg["test_subdir"],
            image_size=self.dataset_cfg["image_size"],
            batch_size=self.train_cfg["num_validation_images"],
            num_workers=self.dataset_cfg["num_workers"],
            image_extensions=self.dataset_cfg["image_extensions"],
            lq_subdir=self.dataset_cfg["lq_subdir"],
            gt_subdir=self.dataset_cfg["gt_subdir"],
            train_subdir=self.dataset_cfg["train_subdir"],
            test_subdir=self.dataset_cfg["test_subdir"],
            random_flip=False,
            random_crop=False,
            max_samples=self.dataset_cfg.get("test_max_samples"),
            shuffle=False,
            pin_memory=self.train_cfg["dataloader_pin_memory"],
        )
        logger.info(
            "  train samples=%d | test samples=%d | weather distribution=%s",
            len(self.train_loader.dataset),
            len(self.val_loader.dataset),
            self.train_loader.dataset.weather_distribution(),
        )

        # ------------------------------------------------------------------
        # Model
        # ------------------------------------------------------------------
        logger.info("Loading SD2 + ControlNet model ...")
        self.model = ControlNetRestorationModel(
            base_model_path=self.model_cfg["base_model_path"],
            controlnet_path=self.model_cfg["controlnet_path"],
            weather_prompt_cfg=self.weather_prompt_cfg,
            device=str(self.device),
            dtype=self.dtype,
            enable_xformers=self.project["enable_xformers"],
            gradient_checkpointing=self.project["gradient_checkpointing"],
        )

        # ------------------------------------------------------------------
        # Optimiser / scheduler
        # ------------------------------------------------------------------
        params = self.model.trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.train_cfg["learning_rate"],
            betas=(self.train_cfg["adam_beta1"], self.train_cfg["adam_beta2"]),
            weight_decay=self.train_cfg["adam_weight_decay"],
            eps=self.train_cfg["adam_epsilon"],
        )
        self.lr_scheduler = self._build_lr_scheduler()
        self.scaler = torch.amp.GradScaler("cuda") if self.dtype == torch.float16 else None

        # ------------------------------------------------------------------
        # Tracker (optional)
        # ------------------------------------------------------------------
        self.tracker = self._build_tracker()

        # ------------------------------------------------------------------
        # Resume
        # ------------------------------------------------------------------
        self.global_step = 0
        self.start_epoch = 0
        self._maybe_resume()

    # ------------------------------------------------------------------ #
    # Optimiser helpers
    # ------------------------------------------------------------------ #
    def _build_lr_scheduler(self):
        sched_name = self.train_cfg["lr_scheduler"]
        warmup = self.train_cfg["lr_warmup_steps"]
        if sched_name == "constant":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda step: 1.0
            )
        if sched_name == "linear":
            total = self._estimate_total_steps()
            def linear_lambda(step):
                if step < warmup:
                    return float(step) / max(1, warmup)
                return max(0.0, float(total - step) / max(1, total - warmup))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=linear_lambda)
        # Default: no decay
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: 1.0
        )

    def _estimate_total_steps(self) -> int:
        if self.train_cfg.get("max_train_steps"):
            return int(self.train_cfg["max_train_steps"])
        steps_per_epoch = max(1, len(self.train_loader) // max(1, self.train_cfg["gradient_accumulation_steps"]))
        return steps_per_epoch * int(self.train_cfg["num_train_epochs"])

    def _build_tracker(self):
        report_to = self.train_cfg.get("report_to", "none")
        if report_to == "none":
            return None
        if report_to == "wandb":
            try:
                import wandb
                wandb.init(
                    project=self.train_cfg["tracker_project_name"],
                    name=self.project["name"],
                    config=self.cfg,
                    dir=self.output_dir,
                )
                return wandb
            except ImportError:
                logger.warning("wandb not installed; falling back to tensorboard.")
                report_to = "tensorboard"
        if report_to == "tensorboard":
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = os.path.join(self.output_dir, "logs")
                return SummaryWriter(log_dir=log_dir)
            except ImportError:
                logger.warning("tensorboard not installed; disabling logging.")
                return None
        return None

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    def _maybe_resume(self) -> None:
        resume = self.train_cfg.get("resume_from_checkpoint")
        if not resume:
            return
        if resume == "latest":
            ckpt = _latest_checkpoint(self.output_dir)
        else:
            ckpt = resume
        if not ckpt or not os.path.isdir(ckpt):
            logger.warning("Resume checkpoint not found: %s", ckpt)
            return
        logger.info("Resuming from %s ...", ckpt)
        self.model.load_controlnet(ckpt)
        try:
            opt_path = os.path.join(ckpt, "optimizer.pt")
            sched_path = os.path.join(ckpt, "scheduler.pt")
            if os.path.isfile(opt_path):
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
            if os.path.isfile(sched_path):
                self.lr_scheduler.load_state_dict(torch.load(sched_path, map_location=self.device))
            self.global_step = int(os.path.basename(ckpt).split("-")[-1])
        except Exception as e:
            logger.warning("Could not fully restore optimizer/scheduler: %s", e)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log(self, payload: Dict[str, float], step: int) -> None:
        if self.tracker is None:
            return
        try:
            if hasattr(self.tracker, "log"):                  # wandb
                self.tracker.log(payload, step=step)
            else:                                              # tensorboard
                for k, v in payload.items():
                    self.tracker.add_scalar(k, v, step)
        except Exception as e:
            logger.debug("Tracker logging failed: %s", e)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        total_steps = self._estimate_total_steps()
        max_steps = self.train_cfg.get("max_train_steps") or total_steps
        logger.info("Starting training: epochs=%d | steps=%d | batch=%d",
                    self.train_cfg["num_train_epochs"], max_steps,
                    self.train_cfg["batch_size"])

        self.model.controlnet.train()
        accum = max(1, self.train_cfg["gradient_accumulation_steps"])
        log_every = self.train_cfg["log_every_n_steps"]
        grad_clip = self.train_cfg["max_grad_norm"]
        ckpt_every = self.train_cfg["checkpointing_steps"]
        val_every = self.train_cfg["validation_steps"]

        step = self.global_step
        start = time.time()
        done = False

        for epoch in range(self.start_epoch, self.train_cfg["num_train_epochs"]):
            if done:
                break
            for batch in self.train_loader:
                # Build prepared batch (encodes latents, prompts, etc.).
                prepared = self.model.prepare_batch(
                    clean_pixel_values=batch["gt"],
                    degraded_pixel_values=batch["lq"],
                    weather_labels=batch["weather"],
                )

                with torch.amp.autocast(
                    device_type="cuda" if self.device.type == "cuda" else "cpu",
                    enabled=(self.dtype != torch.float32),
                    dtype=self.dtype,
                ):
                    out = self.model.compute_loss(
                        prepared,
                        prediction_type=self.train_cfg["prediction_type"],
                        noise_offset=self.train_cfg["noise_offset"],
                    )
                    loss = out["loss"] / accum

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.trainable_parameters(), grad_clip)
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()

                # Periodic logging.
                if (step + 1) % log_every == 0:
                    elapsed = time.time() - start
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "step=%d | epoch=%d | loss=%.4f | loss_l1=%.4f | lr=%.2e | %.1fs",
                        step + 1, epoch, float(out["loss"]), float(out["loss_l1"]),
                        lr, elapsed,
                    )
                    self._log(
                        {
                            "train/loss": float(out["loss"]),
                            "train/loss_l1": float(out["loss_l1"]),
                            "train/lr": lr,
                        },
                        step=step + 1,
                    )

                # Periodic validation.
                if (step + 1) % val_every == 0:
                    self.validate(step + 1)

                # Periodic checkpoint.
                if (step + 1) % ckpt_every == 0:
                    self.save_checkpoint(step + 1)

                step += 1
                if step >= max_steps:
                    done = True
                    break

        # Final save + validation.
        self.save_checkpoint(step)
        self.validate(step)

    # ------------------------------------------------------------------ #
    # Validation: run ControlNet sampling on val samples, save image grids.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self, step: int) -> None:
        logger.info("Running validation at step=%d ...", step)
        self.model.controlnet.eval()

        n_samples = self.train_cfg["num_validation_images"]
        saved = 0
        for batch in self.val_loader:
            if saved >= n_samples:
                break
            lq = batch["lq"].to(self.device, dtype=self.dtype)
            weather = list(batch["weather"])

            preds = self.model.sample(
                degraded_pixel_values=lq,
                weather_labels=weather,
                guidance_scale=self.train_cfg["guidance_scale"],
                num_inference_steps=self.train_cfg["num_inference_steps"],
            )

            # Build a 3-row grid: LQ / GT / restored
            lq_vis = (lq + 1) / 2
            gt = batch["gt"].to(self.device, dtype=self.dtype)
            gt_vis = (gt + 1) / 2
            pred_vis = (preds + 1) / 2
            stacked = torch.cat([lq_vis, gt_vis, pred_vis], dim=0)

            out_path = os.path.join(self.output_dir, "validation")
            os.makedirs(out_path, exist_ok=True)
            _save_image_grid(stacked, ncols=deg.shape[0],
                             path=os.path.join(out_path, f"step-{step:06d}.png"))
            saved += deg.shape[0]

        self.model.controlnet.train()

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, step: int) -> None:
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_controlnet(ckpt_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
        logger.info("Saved checkpoint to %s", ckpt_dir)
        _prune_checkpoints(self.output_dir, self.train_cfg["checkpoints_total_limit"])


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SD2 + ControlNet multi-weather trainer.")
    p.add_argument("--config", type=str, default="config/config.yaml",
                   help="Path to the unified YAML config.")
    p.add_argument("--override", "-o", action="append", default=None,
                   help="Dotted-key override, e.g. weather_prompt.use_weather_prompt=false.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = _parse_overrides(args.override)

    cfg = load_config(args.config, overrides=overrides)

    setup_logging(log_file=os.path.join(cfg["project"]["output_dir"],
                                         cfg["project"]["name"], "train.log"))
    if cfg["project"].get("print_config_on_start", True):
        logger.info("===== Loaded configuration =====")
        print_config(cfg)
        logger.info("===============================")

    if cfg["project"].get("seed_everything", True):
        seed_everything(cfg["project"]["seed"])

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()