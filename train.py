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
import sys
import time
from glob import glob
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from datasets import create_dataloader
from models import ControlNetRestorationModel
from utils import (
    ImageQualityMeter,
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


def _fmt_eta(seconds: float) -> str:
    """Format seconds into a compact ``Hh Mm Ss`` string."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _latest_checkpoint(output_dir: str) -> Optional[str]:
    ckpts = sorted(glob(os.path.join(output_dir, "checkpoint-*")))
    if not ckpts:
        return None
    return ckpts[-1]


def _prune_checkpoints(output_dir: str, keep: int) -> None:
    if keep <= 0:
        return
    ckpts = sorted(
        glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.split("-")[-1]),
    )
    for old in ckpts[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    # ``best/`` is a single-alias directory and is never pruned.


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
            samples_per_weather=self.dataset_cfg.get("samples_per_weather"),
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
            samples_per_weather=self.dataset_cfg.get("samples_per_weather"),
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
            controlnet_hint_range=self.model_cfg.get("controlnet_hint_range", "zero_one"),
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
        # Best-model tracking state
        self.best_metric_value: Optional[float] = None
        self.best_step: Optional[int] = None
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
        steps_per_epoch = max(
            1, len(self.train_loader) // max(1, self.train_cfg["gradient_accumulation_steps"])
        )
        num_epochs = self.train_cfg["num_train_epochs"]
        n_samples = len(self.train_loader.dataset)
        batch_size = self.train_cfg["batch_size"]

        # ------------------------------------------------------------------
        # Sanity hint for the operator
        # ------------------------------------------------------------------
        logger.info(
            "Starting training: epochs=%d | steps=%d | steps/epoch=%d "
            "| dataset=%d samples | batch=%d | accum=%d",
            num_epochs, max_steps, steps_per_epoch,
            n_samples, batch_size, self.train_cfg["gradient_accumulation_steps"],
        )
        if max_steps > 50_000:
            logger.warning(
                "Total training steps is large (%d). If your dataset is small "
                "(<%d samples), consider lowering train.num_train_epochs or "
                "setting train.max_train_steps explicitly in the YAML.",
                max_steps, n_samples * 5,
            )
        logger.info(
            "Note: each training step = one ControlNet+UNet forward+backward pass "
            "with ONE random diffusion timestep sampled from the 1000-step schedule."
        )

        self.model.controlnet.train()
        accum = max(1, self.train_cfg["gradient_accumulation_steps"])
        log_every = self.train_cfg["log_every_n_steps"]
        grad_clip = self.train_cfg["max_grad_norm"]
        ckpt_every = self.train_cfg["checkpointing_steps"]
        val_every = self.train_cfg["validation_steps"]
        show_bar = bool(self.train_cfg.get("show_progress_bar", True))

        # ------------------------------------------------------------------
        # Live tqdm progress bar (one bar for the whole run).
        # ------------------------------------------------------------------
        bar_fmt = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] "
            "{postfix}"
        )
        pbar = tqdm(
            total=max_steps,
            initial=self.global_step,
            desc="train",
            disable=not show_bar,
            bar_format=bar_fmt,
            dynamic_ncols=True,
            file=sys.stderr,
            mininterval=0.5,        # redraw at most twice a second
            maxinterval=2.0,
        )

        step = self.global_step
        start = time.time()
        step_times: list = []
        done = False
        last_loss = float("nan")
        last_diff_loss = float("nan")
        last_pix_loss = float("nan")
        last_lpips = float("nan")

        ckpt_every_epochs = self.train_cfg.get("checkpointing_epochs")

        for epoch in range(self.start_epoch, num_epochs):
            if done:
                break

            epoch_step_start = time.time()
            steps_in_epoch = 0
            # Per-epoch metric accumulators (for the summary line and for
            # the best-model tracking).
            epoch_loss_sum = 0.0
            epoch_diff_sum = 0.0
            epoch_pix_sum  = 0.0
            epoch_lpips_sum = 0.0
            epoch_loss_count = 0

            for batch in self.train_loader:
                t0 = time.time()
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

                step_times.append(time.time() - t0)
                if len(step_times) > 50:
                    step_times.pop(0)
                steps_in_epoch += 1

                last_loss      = float(out["loss"])
                last_diff_loss = float(out["diffusion_loss"])
                last_pix_loss  = float(out["pixel_loss"])
                last_lpips     = float(out["lpips_loss"])
                lr_now = self.optimizer.param_groups[0]["lr"]

                # Accumulate for the per-epoch summary.
                epoch_loss_sum += last_loss
                epoch_diff_sum += last_diff_loss
                epoch_pix_sum  += last_pix_loss
                epoch_lpips_sum += last_lpips
                epoch_loss_count += 1

                # Live progress bar postfix.
                pbar.set_postfix(
                    epoch=epoch,
                    loss=f"{last_loss:.4f}",
                    diff=f"{last_diff_loss:.4f}",
                    pix=f"{last_pix_loss:.4f}",
                    lpips=f"{last_lpips:.4f}",
                    lr=f"{lr_now:.1e}",
                )

                # Periodic logging (also drives TensorBoard / wandb).
                if (step + 1) % log_every == 0:
                    avg_step = sum(step_times) / len(step_times)
                    eta_epoch_sec = avg_step * max(0, steps_per_epoch - steps_in_epoch)
                    eta_total_sec = avg_step * max(0, max_steps - (step + 1))
                    log_line = (
                        f"step={step + 1}/{max_steps} | epoch={epoch} "
                        f"({steps_in_epoch}/{steps_per_epoch}) "
                        f"| loss={last_loss:.4f} "
                        f"| diff={last_diff_loss:.4f} "
                        f"| pix={last_pix_loss:.4f} "
                        f"| lpips={last_lpips:.4f} "
                        f"| lr={lr_now:.2e} | step={avg_step:.2f}s "
                        f"| epoch_eta={_fmt_eta(eta_epoch_sec)} "
                        f"| total_eta={_fmt_eta(eta_total_sec)}"
                    )
                    # tqdm.write goes above the bar without breaking it.
                    pbar.write(log_line)
                    self._log(
                        {
                            "train/loss": last_loss,
                            "train/diffusion_loss": last_diff_loss,
                            "train/pixel_loss": last_pix_loss,
                            "train/lpips_loss": last_lpips,
                            "train/lr": lr_now,
                            "train/avg_step_sec": avg_step,
                        },
                        step=step + 1,
                    )

                # Periodic validation.
                if (step + 1) % val_every == 0:
                    self.validate(step + 1)

                # Periodic checkpoint (step-based).
                if (step + 1) % ckpt_every == 0:
                    self.save_checkpoint(step + 1)

                step += 1
                pbar.update(1)
                if step >= max_steps:
                    done = True
                    break

            # ------------------------------------------------------------------
            # End-of-epoch summary + epoch-level checkpoint + best tracking
            # ------------------------------------------------------------------
            epoch_sec = time.time() - epoch_step_start
            avg_loss    = epoch_loss_sum   / max(1, epoch_loss_count)
            avg_diff    = epoch_diff_sum   / max(1, epoch_loss_count)
            avg_pix     = epoch_pix_sum    / max(1, epoch_loss_count)
            avg_lpips   = epoch_lpips_sum  / max(1, epoch_loss_count)

            # ------------------------------------------------------------------
            # Run PSNR / SSIM on the test set at every epoch end so the
            # per-epoch summary line carries them and best-tracking can
            # optionally use them.
            # ------------------------------------------------------------------
            try:
                val_metrics = self.validate(step, return_metrics=True) or {}
            except Exception as e:
                logger.warning("End-of-epoch validation failed: %s", e)
                val_metrics = {}

            psnr = val_metrics.get("psnr")
            ssim = val_metrics.get("ssim")

            # Print a clear per-epoch metric line above the bar.
            metric_parts = [
                f"epoch {epoch}",
                f"loss={avg_loss:.4f}",
                f"diff={avg_diff:.4f}",
                f"pix={avg_pix:.4f}",
                f"lpips={avg_lpips:.4f}",
            ]
            if psnr is not None:
                metric_parts.append(f"val_psnr={psnr:.3f}dB")
            if ssim is not None:
                metric_parts.append(f"val_ssim={ssim:.4f}")
            metric_parts.append(f"steps={steps_in_epoch}")
            metric_parts.append(
                f"time={_fmt_eta(epoch_sec)} "
                f"({epoch_sec / max(1, steps_in_epoch):.2f}s/step)"
            )
            metric_line = " | ".join(metric_parts)
            pbar.write(metric_line)
            logger.info("METRIC %s", metric_line)
            log_payload = {
                "train/epoch_loss": avg_loss,
                "train/epoch_diffusion_loss": avg_diff,
                "train/epoch_pixel_loss": avg_pix,
                "train/epoch_lpips_loss": avg_lpips,
                "train/epoch_seconds": epoch_sec,
            }
            if psnr is not None:
                log_payload["val/epoch_psnr"] = psnr
            if ssim is not None:
                log_payload["val/epoch_ssim"] = ssim
            self._log(log_payload, step=step)

            # Epoch-level checkpoint.
            save_now = (
                ckpt_every_epochs is not None
                and ckpt_every_epochs > 0
                and ((epoch + 1) % ckpt_every_epochs == 0)
            )
            if save_now:
                pbar.write(f"Saving epoch checkpoint at epoch={epoch} ...")
                try:
                    self.save_checkpoint(step, tag=f"epoch-{epoch:03d}")
                except Exception as e:
                    logger.warning("Epoch checkpoint failed: %s", e)

            # Best-model tracking.  Resolves which metric to use from YAML.
            metric_name = self.train_cfg.get("best_metric", "epoch_loss")
            metric_value = self._select_best_metric(
                metric_name=metric_name,
                avg_loss=avg_loss,
                avg_loss_l1=avg_pix,    # backwards-compat: "epoch_loss_l1" -> pixel L1
                val_psnr=psnr,
                val_ssim=ssim,
            )
            if metric_value is not None:
                self._maybe_update_best(metric_value, step=step, epoch=epoch)

        pbar.close()

        # Final save + validation.
        try:
            self.save_checkpoint(step, tag="final")
        except Exception as e:
            logger.warning("Final checkpoint failed: %s", e)
        final_metrics = self.validate(step, return_metrics=True) or {}
        logger.info(
            "Training finished: %d steps in %s | best %s = %s @ step %s "
            "| final_psnr=%s | final_ssim=%s",
            step - self.global_step, _fmt_eta(time.time() - start),
            self.train_cfg.get("best_metric", "epoch_loss"),
            f"{self.best_metric_value:.6f}" if self.best_metric_value is not None else "n/a",
            self.best_step,
            f"{final_metrics['psnr']:.3f}" if "psnr" in final_metrics else "n/a",
            f"{final_metrics['ssim']:.4f}" if "ssim" in final_metrics else "n/a",
        )

    def _select_best_metric(
        self,
        metric_name: str,
        avg_loss: float,
        avg_loss_l1: float,
        val_psnr: Optional[float],
        val_ssim: Optional[float],
    ) -> Optional[float]:
        """Resolve the YAML-configured ``best_metric`` to a scalar value."""
        if metric_name == "epoch_loss_l1":
            return avg_loss_l1
        if metric_name == "epoch_loss":
            return avg_loss
        if metric_name == "val_psnr":
            return val_psnr
        if metric_name == "val_ssim":
            return val_ssim
        # Unknown metric name -> log warning, fall back to train loss.
        logger.warning(
            "Unknown best_metric '%s'; falling back to 'epoch_loss'. "
            "Valid options: epoch_loss | epoch_loss_l1 | val_psnr | val_ssim.",
            metric_name,
        )
        return avg_loss

    # ------------------------------------------------------------------ #
    # Validation: run ControlNet sampling on val samples, save image grids.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self, step: int, return_metrics: bool = True) -> Optional[Dict[str, float]]:
        """Render validation grids + compute PSNR/SSIM over the full test set.

        Returns
        -------
        dict with keys ``"psnr"`` and ``"ssim"`` (averaged over the entire
        test loader), or ``None`` when ``return_metrics=False``.
        """
        n_samples = self.train_cfg["num_validation_images"]
        out_dir = os.path.join(self.output_dir, "validation")
        path, metrics = self._render_test_grid(
            tag=f"step-{step:06d}",
            n_images=n_samples,
            out_dir=out_dir,
            compute_metrics=True,
        )
        if metrics:
            logger.info(
                "Validation @ step=%d | PSNR=%.3f dB | SSIM=%.4f | grid=%s",
                step, metrics["psnr"], metrics["ssim"], path or "(none)",
            )
            self._log(
                {
                    "val/psnr": metrics["psnr"],
                    "val/ssim": metrics["ssim"],
                },
                step=step,
            )
        self.model.controlnet.train()
        return metrics if return_metrics else None

    @torch.no_grad()
    def _render_test_grid(
        self,
        tag: str,
        n_images: int,
        out_dir: str,
        compute_metrics: bool = False,
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        """Run inference on the first ``n_images`` test samples and write a
        3-row ``LQ / GT / restored`` PNG.

        When ``compute_metrics=True``, PSNR/SSIM are also computed across
        the **entire** test loader (not just the first ``n_images``).
        Returns ``(png_path, metrics_dict_or_None)``.
        """
        self.model.controlnet.eval()
        os.makedirs(out_dir, exist_ok=True)

        iq_cfg = self.train_cfg.get("image_quality", {})
        meter = ImageQualityMeter(
            data_range=float(iq_cfg.get("data_range", 1.0)),
            use_torchmetrics=(iq_cfg.get("backend", "torchmetrics") == "torchmetrics"),
        )
        saved = 0
        written_path = ""
        n_total = 0

        for batch in self.val_loader:
            lq = batch["lq"].to(self.device, dtype=self.dtype)
            gt = batch["gt"].to(self.device, dtype=self.dtype)
            weather = list(batch["weather"])

            tile_size = self.cfg["infer"].get("tile_size", 0)
            tile_stride = self.cfg["infer"].get("tile_stride", 0)
            tile_blend_sigma = self.cfg["infer"].get("tile_blend_sigma", 0.0)
            preview_size = self.train_cfg.get("preview_image_size", 0)

            if tile_size > 0 and tile_stride > 0:
                preds = self.model.sample_tiled(
                    degraded_pixel_values=lq,
                    weather_labels=weather,
                    guidance_scale=self.train_cfg["guidance_scale"],
                    num_inference_steps=self.train_cfg["num_inference_steps"],
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                    blend_sigma=tile_blend_sigma,
                    upscale_to=preview_size,
                )
            else:
                preds = self.model.sample(
                    degraded_pixel_values=lq,
                    weather_labels=weather,
                    guidance_scale=self.train_cfg["guidance_scale"],
                    num_inference_steps=self.train_cfg["num_inference_steps"],
                )

            # Accumulate PSNR/SSIM over the full test set.
            if compute_metrics:
                pred_01 = ((preds + 1) / 2).clamp(0.0, 1.0)
                gt_01 = ((gt + 1) / 2).clamp(0.0, 1.0)
                meter.update(pred_01, gt_01)

            # Save the first ``n_images`` samples as a 3-row grid.
            if saved < n_images:
                lq_vis = (lq + 1) / 2
                gt_vis = (gt + 1) / 2
                pred_vis = (preds + 1) / 2
                stacked = torch.cat([lq_vis, gt_vis, pred_vis], dim=0)

                path = os.path.join(out_dir, f"{tag}.png")
                _save_image_grid(stacked, ncols=lq.shape[0], path=path)
                saved += lq.shape[0]
                if not written_path:
                    written_path = path

            n_total += lq.shape[0]

        metrics = meter.compute() if compute_metrics else None
        if compute_metrics and metrics is not None:
            metrics["n_samples"] = int(n_total)
        return written_path, metrics

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, step: int, tag: Optional[str] = None,
                        is_best: bool = False) -> str:
        """Persist ControlNet + optimiser + scheduler to ``checkpoint-<step>``.

        Also renders a visual preview grid (LQ / GT / restored) so the
        operator can sanity-check restoration quality over time.

        Parameters
        ----------
        step
            Optimizer step counter (used for naming + pruning).
        tag
            Optional explicit tag used for the preview filename; defaults
            to ``"step-<step:06d>"``.  Pass ``"best"`` when saving the
            best-so-far model.
        is_best
            When true, also write the same payload to ``<output>/best/``
            (overwriting any previous best).
        """
        if tag is None:
            tag = f"step-{step:06d}"

        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_controlnet(ckpt_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

        # Log what was written so the operator knows the exact files.
        written = sorted(os.listdir(ckpt_dir))
        logger.info("Saved checkpoint to %s (files: %s)",
                    ckpt_dir, ", ".join(written))

        # Visual preview: LQ / GT / restored on the first N test images.
        try:
            preview_dir = os.path.join(self.output_dir, "preview")
            preview_path = self._render_test_grid(
                tag=f"step-{step:06d}",
                n_images=self.train_cfg.get("num_preview_images", 4),
                out_dir=preview_dir,
            )
            if preview_path:
                logger.info("Saved preview grid -> %s", preview_path)
        except Exception as e:
            logger.warning("Preview rendering failed: %s", e)
        finally:
            self.model.controlnet.train()

        # Mirror to <output>/best/ if requested.
        if is_best:
            best_dir = os.path.join(self.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            # Remove stale contents so the best/ folder is always a single
            # snapshot of the latest "best" weights.
            for entry in os.listdir(best_dir):
                full = os.path.join(best_dir, entry)
                if os.path.isdir(full):
                    shutil.rmtree(full, ignore_errors=True)
                else:
                    try:
                        os.remove(full)
                    except OSError:
                        pass
            self.model.save_controlnet(best_dir)
            torch.save(self.optimizer.state_dict(), os.path.join(best_dir, "optimizer.pt"))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(best_dir, "scheduler.pt"))
            # Save the matching preview into best/ for direct comparison.
            try:
                best_preview = self._render_test_grid(
                    tag="best",
                    n_images=self.train_cfg.get("num_preview_images", 4),
                    out_dir=os.path.join(self.output_dir, "preview"),
                )
                if best_preview:
                    logger.info("Saved best-preview grid -> %s", best_preview)
            except Exception as e:
                logger.warning("Best-preview rendering failed: %s", e)
            finally:
                self.model.controlnet.train()

        _prune_checkpoints(self.output_dir, self.train_cfg["checkpoints_total_limit"])
        return ckpt_dir

    # ------------------------------------------------------------------ #
    # Best-model tracking
    # ------------------------------------------------------------------ #
    @staticmethod
    def _infer_metric_mode(metric_name: str) -> str:
        """Infer the natural comparison direction for known metrics.

        * loss-style metrics (``*_loss*``) -> ``"min"`` (lower is better)
        * psnr / ssim                       -> ``"max"`` (higher is better)
        """
        name = metric_name.lower()
        if "psnr" in name or "ssim" in name:
            return "max"
        return "min"

    def _resolve_best_mode(self) -> str:
        """Resolve ``best_metric_mode`` from explicit YAML or inference."""
        explicit = self.train_cfg.get("best_metric_mode")
        if explicit:
            return explicit
        name = self.train_cfg.get("best_metric", "epoch_loss").lower()
        return "max" if "psnr" in name or "ssim" in name else "min"

    def _is_better(self, candidate: float, best: Optional[float]) -> bool:
        """Return True iff ``candidate`` improves on ``best`` per the
        configured ``best_metric_mode``.
        """
        if best is None:
            return True
        mode = self._resolve_best_mode()
        return candidate < best if mode == "min" else candidate > best

    def _maybe_update_best(self, metric_value: float, step: int, epoch: int) -> None:
        """If ``metric_value`` is the best so far, persist a ``best/`` copy."""
        if not self.train_cfg.get("save_best", True):
            return
        if not self._is_better(metric_value, self.best_metric_value):
            return
        prev = self.best_metric_value
        self.best_metric_value = metric_value
        self.best_step = step
        try:
            self.save_checkpoint(step, tag=f"best-after-epoch-{epoch}", is_best=True)
            msg = (
                f"  >> NEW BEST {self.train_cfg.get('best_metric', 'epoch_loss')} "
                f"= {metric_value:.6f} (prev={prev}) @ epoch={epoch} step={step}"
            )
            logger.info(msg)
            tqdm.write(msg)
        except Exception as e:
            logger.warning("Failed to save best checkpoint: %s", e)


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