"""Stable Diffusion 2 + ControlNet wrapper for image restoration.

What this module does
---------------------

1. Loads a frozen SD2 pipeline (UNet + VAE + text-encoder + scheduler +
   tokenizer + feature-extractor).
2. Loads (or randomly initialises) a ControlNet that consumes the
   *degraded* image as a hint and predicts residual control signals for the
   SD2 UNet.
3. Exposes a thin API used by ``train.py``:

   * :meth:`prepare_batch` – encodes a raw batch (images + weather labels)
     into latents / prompt embeddings / ControlNet hints.
   * :meth:`compute_loss`  – runs the diffusion forward pass and returns a
     scalar loss plus optional auxiliary metrics for logging.
   * :meth:`sample`        – runs the full SD2 + ControlNet sampling loop
     for validation / inference.

Why ControlNet for restoration?
-------------------------------
Treating restoration as a *conditioned generation* problem lets us reuse
the rich image prior inside SD2.  The ControlNet branch learns the
degradation-specific residual that the SD2 UNet should remove.

Notes
-----
* We use HuggingFace ``diffusers``.  If you don't have internet access at
  training time, set ``base_model_path`` / ``controlnet_path`` to local
  checkpoint directories.
* Only the ControlNet parameters are trained.
* Mixed-precision is delegated to the surrounding ``accelerate`` launcher.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import warnings
warnings.filterwarnings("ignore", message=".*You have disabled the safety checker.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils import get_logger
from .weather_conditioning import WeatherPromptEncoder


logger = get_logger("model")


@dataclass
class PreparedBatch:
    """All tensors produced by :meth:`ControlNetRestorationModel.prepare_batch`."""

    latents: torch.Tensor                # clean image latents, scaled
    noisy_latents: torch.Tensor          # latents + noise
    noise: torch.Tensor                  # the added noise
    timesteps: torch.Tensor              # diffusion timesteps
    prompt_embeds: torch.Tensor          # text encoder outputs
    degraded_pixel_values: torch.Tensor  # kept for ControlNet hint
    clean_pixel_values: torch.Tensor      # GT pixels, used by the logging L1
    weather_labels: List[str]            # original labels (for logging)


class ControlNetRestorationModel:
    """Wrapper around SD2 + ControlNet for multi-weather restoration."""

    def __init__(
        self,
        base_model_path: str,
        controlnet_path: Optional[str],
        weather_prompt_cfg: Dict,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        enable_xformers: bool = True,
        gradient_checkpointing: bool = True,
        controlnet_hint_range: str = "zero_one",
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

        # Range expected by the ControlNet's ``conv_in`` (controls how we
        # rescale the LQ image before passing it as ``controlnet_cond``).
        if controlnet_hint_range not in {"zero_one", "minus_one_one"}:
            raise ValueError(
                f"controlnet_hint_range must be 'zero_one' or 'minus_one_one', "
                f"got {controlnet_hint_range!r}"
            )
        self.controlnet_hint_range = controlnet_hint_range

        # Lazy imports keep the rest of the codebase usable without diffusers.
        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            DDPMScheduler,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTokenizer

        # ------------------------------------------------------------------
        # 1. SD2 components (frozen)
        # ------------------------------------------------------------------
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer"
        )
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            base_model_path, subfolder="text_encoder"
        )
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            base_model_path, subfolder="vae"
        )
        self.unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            base_model_path, subfolder="unet"
        )
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
            base_model_path, subfolder="scheduler"
        )

        # ------------------------------------------------------------------
        # 2. ControlNet (trainable)
        # ------------------------------------------------------------------
        # ``from_pretrained`` accepts either a local directory or a
        # HuggingFace Hub repo id (e.g. "lllyasviel/sd-controlnet-canny"),
        # so we just hand the path through.  Falls back to random init only
        # when no path is configured at all.
        if controlnet_path:
            logger.info("Loading ControlNet weights from %s", controlnet_path)
            self.controlnet = ControlNetModel.from_pretrained(controlnet_path)
        else:
            logger.warning(
                "No ControlNet path configured — falling back to random init "
                "(ControlNetModel.from_unet). Training will be slow / unstable "
                "until enough data is seen; consider providing a pretrained "
                "checkpoint that matches model.base_model_path."
            )
            self.controlnet = ControlNetModel.from_unet(self.unet)
            self._random_init_customize()

        # ------------------------------------------------------------------
        # 3. Freeze everything except ControlNet
        # ------------------------------------------------------------------
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(True)

        # ------------------------------------------------------------------
        # 4. Memory optimisations
        # ------------------------------------------------------------------
        if gradient_checkpointing:
            self.controlnet.enable_gradient_checkpointing()
        if enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                self.controlnet.enable_xformers_memory_efficient_attention()
            except Exception:
                # xformers not installed – fall back silently.
                pass

        # ------------------------------------------------------------------
        # 5. Move to device / dtype
        # ------------------------------------------------------------------
        # Frozen components can sit in autocast dtype (FP16 / BF16) — that
        # saves memory and matches the autocast forward-pass dtype.
        self.vae.to(self.device, dtype=dtype)
        self.text_encoder.to(self.device, dtype=dtype)
        self.unet.to(self.device, dtype=dtype)

        # The trainable ControlNet MUST stay in FP32 — these are the
        # "master weights" that ``torch.amp.GradScaler`` updates.
        # ``GradScaler.unscale_()`` refuses to unscale FP16 gradients
        # ("Attempting to unscale FP16 gradients"), so leaving them in
        # FP16 breaks AMP.  Autocast still casts them to FP16 inside the
        # forward pass, so compute stays fast.
        self.controlnet.to(self.device, dtype=torch.float32)

        # ------------------------------------------------------------------
        # 6. Weather prompt builder
        # ------------------------------------------------------------------
        self.weather_prompt_encoder = WeatherPromptEncoder(**weather_prompt_cfg)

        # ------------------------------------------------------------------
        # 7. LPIPS perceptual loss (frozen VGG, lazy-loaded on first use so
        #    the constructor stays cheap and we don't pay the download
        #    cost when the user only wants inference / metric eval).
        # ------------------------------------------------------------------
        self._lpips_fn = None
        self._lpips_loss_weight = float(
            (weather_prompt_cfg.get("lpips_loss_weight") if isinstance(weather_prompt_cfg, dict) else None)
            or 0.05
        )
        self._pixel_loss_weight = float(
            (weather_prompt_cfg.get("pixel_loss_weight") if isinstance(weather_prompt_cfg, dict) else None)
            or 0.1
        )

    # ------------------------------------------------------------------ #
    # Parameters exposed for the optimiser
    # ------------------------------------------------------------------ #
    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [p for p in self.controlnet.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #
    def encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        """Tokenise + embed ``prompts`` with the frozen SD2 text encoder."""
        with torch.no_grad():
            tokens = self.tokenizer(
                list(prompts),
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            tokens = tokens.to(self.device)
            embeds = self.text_encoder(tokens)[0].to(dtype=self.dtype)
        return embeds

    # ------------------------------------------------------------------ #
    # LPIPS perceptual loss (lazy)
    # ------------------------------------------------------------------ #
    def _get_lpips(self):
        """Return a frozen LPIPS (VGG backbone) module, initialised once."""
        if self._lpips_fn is None:
            try:
                import lpips
            except ImportError as e:
                raise ImportError(
                    "LPIPS requires the `lpips` package. Install with: "
                    "`pip install lpips`.  You can also set "
                    "weather_prompt.lpips_loss_weight=0 in the YAML to skip it."
                ) from e
            self._lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
            for p in self._lpips_fn.parameters():
                p.requires_grad_(False)
            self._lpips_fn.eval()
            logger.info("LPIPS perceptual loss initialised (VGG backbone).")
        return self._lpips_fn

    def encode_images_to_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images in [-1, 1] to SD2 latent space (VAE is frozen)."""
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(dtype=self.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents back to image space in [-1, 1]."""
        with torch.no_grad():
            latents = latents / self.vae.config.scaling_factor
            images = self.vae.decode(latents.to(dtype=self.dtype)).sample
        return images.clamp(-1, 1)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def prepare_batch(
        self,
        clean_pixel_values: torch.Tensor,
        degraded_pixel_values: torch.Tensor,
        weather_labels: Sequence[str],
        generator: Optional[torch.Generator] = None,
    ) -> PreparedBatch:
        """Encode a training batch into latents / prompt embeds / noise."""
        device = self.device
        clean_pixel_values = clean_pixel_values.to(device, dtype=self.dtype)
        degraded_pixel_values = degraded_pixel_values.to(device, dtype=self.dtype)
        # Keep an FP32 copy of the clean pixels so the logging L1 in
        # ``compute_loss`` can compare against the GT without dtype drift.
        clean_pixel_values_fp32 = clean_pixel_values.to(torch.float32)

        # 1. Clean image -> latents.
        latents = self.encode_images_to_latents(clean_pixel_values)

        # 2. Sample noise + timesteps.
        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=self.dtype)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=device, dtype=torch.long,
        )
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 3. Weather-conditioned prompts -> text embeddings.
        wp = self.weather_prompt_encoder.build_prompts(weather_labels, generator=generator)
        prompt_embeds = self.encode_prompts(wp.prompts)

        return PreparedBatch(
            latents=latents,
            noisy_latents=noisy_latents,
            noise=noise,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            degraded_pixel_values=degraded_pixel_values,
            clean_pixel_values=clean_pixel_values_fp32,
            weather_labels=list(weather_labels),
        )

    def compute_loss(
        self,
        batch: PreparedBatch,
        prediction_type: str = "epsilon",
        noise_offset: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Run the ControlNet-conditional diffusion forward pass.

        Total objective:
            total_loss = diffusion_loss
                        + pixel_loss_weight * L1(recon, clean)
                        + lpips_loss_weight * LPIPS(recon, clean)

        Returns
        -------
        dict with keys
            ``loss``          - scalar total loss (used for backprop)
            ``diffusion_loss``- MSE on noise prediction (logging)
            ``pixel_loss``    - L1 between recon and clean pixels (logging)
            ``lpips_loss``    - LPIPS perceptual distance (logging)
        """
        # 1. ControlNet forward pass with the degraded image as the hint.
        #
        # ``batch.degraded_pixel_values`` lives in [-1, 1] (VAE convention).
        # The ControlNet's ``conv_in`` was trained on canny edge maps in
        # [0, 1] — feeding it [-1, 1] would shift the activation
        # distribution and waste the pretrained features.  Rescale here.
        if self.controlnet_hint_range == "zero_one":
            controlnet_cond = (batch.degraded_pixel_values + 1.0) / 2.0
        else:
            controlnet_cond = batch.degraded_pixel_values

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            batch.noisy_latents,
            batch.timesteps,
            encoder_hidden_states=batch.prompt_embeds,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        # 2. SD UNet forward pass, conditioned on the ControlNet residuals.
        model_pred = self.unet(
            batch.noisy_latents,
            batch.timesteps,
            encoder_hidden_states=batch.prompt_embeds,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        # 3. Diffusion loss (epsilon / v_prediction).
        if noise_offset > 0.0:
            offset_noise = noise_offset * torch.randn(
                (batch.noise.shape[0], batch.noise.shape[1], 1, 1),
                device=batch.noise.device, dtype=batch.noise.dtype,
            )
            target = batch.noise + offset_noise
        elif prediction_type == "epsilon":
            target = batch.noise
        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                batch.noisy_latents, batch.noise, batch.timesteps
            )
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

        diffusion_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # ------------------------------------------------------------------
        # Pixel-space recon: x0 -> VAE decode -> L1 vs clean GT.
        # VAE decode is precision-sensitive; run in FP32 explicitly.
        # ------------------------------------------------------------------
        if prediction_type == "epsilon":
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
                device=batch.noisy_latents.device, dtype=torch.float32,
            )
            alpha_bar_t = alphas_cumprod[batch.timesteps].view(-1, 1, 1, 1)
            sqrt_alpha_bar = alpha_bar_t.sqrt()
            sqrt_one_minus = (1.0 - alpha_bar_t).sqrt()
            pred_x0 = (batch.noisy_latents.float() - sqrt_one_minus * model_pred.float()) / sqrt_alpha_bar
        elif prediction_type == "v_prediction":
            if hasattr(self.noise_scheduler, "get_velocity_to_x0"):
                pred_x0 = self.noise_scheduler.get_velocity_to_x0(
                    batch.noisy_latents, model_pred, batch.timesteps
                )
            else:
                pred_x0 = model_pred.float()
        else:
            pred_x0 = model_pred.float()

        pred_x0_scaled = pred_x0 / self.vae.config.scaling_factor
        recon = self.vae.decode(pred_x0_scaled.to(dtype=self.dtype)).sample.clamp(-1, 1)
        pixel_loss = F.l1_loss(recon.float(), batch.clean_pixel_values)

        lpips_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self._lpips_loss_weight > 0.0:
            try:
                lpips_loss = self._get_lpips()(recon, batch.clean_pixel_values.float()).mean()
            except Exception as e:
                logger.debug("LPIPS forward failed; treating loss as zero. %s", e)

        total_loss = (
            diffusion_loss
            + self._pixel_loss_weight * pixel_loss
            + self._lpips_loss_weight * lpips_loss
        )

        return {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss.detach(),
            "pixel_loss": pixel_loss.detach(),
            "lpips_loss": lpips_loss.detach(),
        }

    # ------------------------------------------------------------------ #
    # Inference (used by the validation step in train.py)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        degraded_pixel_values: torch.Tensor,
        weather_labels: Sequence[str],
        guidance_scale: float = 2.5,
        num_inference_steps: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Run the full SD2 + ControlNet sampling loop.

        Notes
        -----
        The trainable ControlNet is kept in **FP32** as master weights
        (see ``__init__``).  When the frozen modules are in FP16/BF16
        we therefore wrap the pipeline call in ``torch.amp.autocast`` so
        the FP32 ControlNet activations are downcast on the fly to match
        the FP16 UNet, otherwise the matmul in the UNet fails with
        ``mat1 and mat2 must have the same dtype``.
        """
        from diffusers import StableDiffusionControlNetPipeline

        wp = self.weather_prompt_encoder.build_prompts(weather_labels)

        pipe = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe.set_progress_bar_config(disable=True)

        # Tell the pipeline NOT to normalise the PIL hint into [-1, 1] — the
        # canny ControlNet was trained on [0, 1] edge maps.  Otherwise
        # ``VaeImageProcessor.preprocess(do_normalize=True)`` would shift
        # the input distribution back into [-1, 1] and waste the
        # pretrained features, exactly the same mismatch we fixed on the
        # training side above.
        if self.controlnet_hint_range == "zero_one":
            pipe.image_processor.do_normalize = False

        pil_hint = self._tensor_to_pil(degraded_pixel_values)

        with torch.amp.autocast(
            device_type="cuda" if self.device.type == "cuda" else "cpu",
            enabled=(self.dtype != torch.float32),
            dtype=self.dtype,
        ):
            out = pipe(
                prompt=wp.prompts,
                negative_prompt=None,
                image=pil_hint,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                output_type="pt",
            ).images
        return out.clamp(-1, 1)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tensor_to_pil(images: torch.Tensor) -> List[Image.Image]:
        """Convert ``(B, 3, H, W)`` tensor in ``[-1, 1]`` to PIL images."""
        images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1).numpy()
        return [Image.fromarray(img) for img in images]

    # ------------------------------------------------------------------ #
    # Saving / loading
    # ------------------------------------------------------------------ #
    def save_controlnet(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.controlnet.save_pretrained(output_dir)

    # ------------------------------------------------------------------ #
    # Random-init customisation
    # ------------------------------------------------------------------ #
    def _random_init_customize(self) -> None:
        """Tune freshly-initialised ControlNet for [-1,1] image restoration.

        Called only when ControlNet was randomly initialised (no pretrained checkpoint).
        Zero-convolutions and conv_in receive special treatment so the network
        can forward a [-1,1] degraded hint without numerical mismatch.
        """
        def _init_conv_in(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-4)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def _init_zero_convs(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d) and child.out_channels == 4:
                    nn.init.zeros_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)
                else:
                    _init_zero_convs(child)

        embed = self.controlnet.controlnet_cond_embedding
        if hasattr(embed, 'conv_in'):
            embed.conv_in.apply(_init_conv_in)

        _init_zero_convs(embed)
        _init_zero_convs(self.controlnet.mid_block)

        logger.info("Random-init ControlNet customised for [-1,1] hint range.")

    # ------------------------------------------------------------------ #
    # Tiled / variable-resolution inference
    # ------------------------------------------------------------------ #
    @torch.amp.autocast("cuda", enabled=False)
    def sample_tiled(
        self,
        degraded_pixel_values: torch.Tensor,
        weather_labels: Sequence[str],
        guidance_scale: float = 2.5,
        num_inference_steps: int = 30,
        tile_size: int = 512,
        tile_stride: int = 384,
        blend_sigma: float = 32.0,
        upscale_to: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Variable-resolution inference via overlapping tiles.

        Parameters
        ----------
        degraded_pixel_values : (B, 3, H, W) in [-1, 1]
        weather_labels        : weather strings for prompt encoder
        guidance_scale        : CFG scale (2.5 typical for restoration)
        num_inference_steps   : diffusion steps per tile
        tile_size             : square window size (default 512, must be multiple of 8)
        tile_stride           : step between windows (default 384 → ~25% overlap)
        blend_sigma           : Gaussian sigma for feathered blending (default stride/12)
        upscale_to            : if > 0, upscale short side to this before inference,
                                 then crop back to original size at output
        """
        from diffusers import StableDiffusionControlNetPipeline

        B, C, H_orig, W_orig = degraded_pixel_values.shape
        device = degraded_pixel_values.device
        H_curr, W_curr = H_orig, W_orig

        if upscale_to > 0 and min(H_orig, W_orig) < upscale_to:
            scale = upscale_to / min(H_orig, W_orig)
            new_h = int(H_orig * scale + 0.5)
            new_w = int(W_orig * scale + 0.5)
            new_h = max(8, new_h - new_h % 8)
            new_w = max(8, new_w - new_w % 8)
            degraded_pixel_values = F.interpolate(
                degraded_pixel_values, size=(new_h, new_w), mode="bicubic", align_corners=False
            )
            H_curr, W_curr = new_h, new_w

        pad_h = (tile_stride - H_curr % tile_stride) % tile_stride
        pad_w = (tile_stride - W_curr % tile_stride) % tile_stride
        if pad_h or pad_w:
            degraded_pixel_values = F.pad(
                degraded_pixel_values, (0, pad_w, 0, pad_h), mode="reflect"
            )

        H_pad, W_pad = degraded_pixel_values.shape[2], degraded_pixel_values.shape[3]

        if blend_sigma <= 0:
            blend_sigma = tile_stride / 12.0
        weight_map = self._build_gaussian_weight(tile_size, blend_sigma, device)

        output_accum = torch.zeros((B, 3, H_pad, W_pad), device=device, dtype=self.dtype)
        weight_accum = torch.zeros((B, 1, H_pad, W_pad), device=device, dtype=self.dtype)

        num_rows = max(1, (H_pad - tile_size) // tile_stride + 1)
        num_cols = max(1, (W_pad - tile_size) // tile_stride + 1)

        wp = self.weather_prompt_encoder.build_prompts(weather_labels)

        pipe = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        pipe.set_progress_bar_config(disable=True)
        if self.controlnet_hint_range == "zero_one":
            pipe.image_processor.do_normalize = False

        y_positions = [r * tile_stride for r in range(num_rows)]
        x_positions = [c * tile_stride for c in range(num_cols)]
        if y_positions[-1] + tile_size > H_pad:
            y_positions[-1] = H_pad - tile_size
        if x_positions[-1] + tile_size > W_pad:
            x_positions[-1] = W_pad - tile_size

        for y in y_positions:
            for x in x_positions:
                tile = degraded_pixel_values[:, :, y:y+tile_size, x:x+tile_size]
                pil_tile = self._tensor_to_pil(tile)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*safety.*checker.*")
                    with torch.amp.autocast(
                        device_type="cuda" if self.device.type == "cuda" else "cpu",
                        enabled=(self.dtype != torch.float32),
                        dtype=self.dtype,
                    ):
                        out_tile = pipe(
                            prompt=wp.prompts,
                            negative_prompt=None,
                            image=pil_tile,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            output_type="pt",
                        ).images

                out_tile = out_tile.to(device=self.device, dtype=self.dtype)
                output_accum[:, :, y:y+tile_size, x:x+tile_size] += out_tile * weight_map
                weight_accum[:, :, y:y+tile_size, x:x+tile_size] += weight_map

        output_accum = output_accum / weight_accum.clamp(min=1e-8)
        output = output_accum[:, :, :H_curr, :W_curr]

        if upscale_to > 0 and (H_curr != H_orig or W_curr != W_orig):
            output = F.interpolate(
                output, size=(H_orig, W_orig), mode="bicubic", align_corners=False
            )

        return output.clamp(-1, 1)

    @staticmethod
    @staticmethod
    def _build_gaussian_weight(tile_size: int, sigma: float, device: torch.device):
        half = tile_size / 2.0
        coords = torch.linspace(-half, half, tile_size, device=device)
        y_g, x_g = torch.meshgrid(coords, coords, indexing="ij")
        g = torch.exp(-(x_g**2 + y_g**2) / (2 * sigma**2))
        return g.unsqueeze(0).unsqueeze(0)

    def load_controlnet(self, path: str) -> None:
        from diffusers import ControlNetModel
        # Keep trainable ControlNet in FP32 (master weights); see __init__ comment.
        self.controlnet = ControlNetModel.from_pretrained(path).to(self.device, dtype=torch.float32)
        self.controlnet.requires_grad_(True)