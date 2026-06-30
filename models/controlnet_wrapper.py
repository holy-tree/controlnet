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

import torch
import torch.nn.functional as F
from PIL import Image

from .weather_conditioning import WeatherPromptEncoder


@dataclass
class PreparedBatch:
    """All tensors produced by :meth:`ControlNetRestorationModel.prepare_batch`."""

    latents: torch.Tensor                # clean image latents, scaled
    noisy_latents: torch.Tensor          # latents + noise
    noise: torch.Tensor                  # the added noise
    timesteps: torch.Tensor              # diffusion timesteps
    prompt_embeds: torch.Tensor          # text encoder outputs
    degraded_pixel_values: torch.Tensor  # kept for ControlNet hint
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
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype

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
        if controlnet_path and os.path.isdir(controlnet_path):
            self.controlnet = ControlNetModel.from_pretrained(controlnet_path)
        else:
            # Random initialisation – fine for the "process first" goal.
            self.controlnet = ControlNetModel.from_unet(self.unet)

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
        self.vae.to(self.device, dtype=dtype)
        self.text_encoder.to(self.device, dtype=dtype)
        self.unet.to(self.device, dtype=dtype)
        self.controlnet.to(self.device, dtype=dtype)

        # ------------------------------------------------------------------
        # 6. Weather prompt builder
        # ------------------------------------------------------------------
        self.weather_prompt_encoder = WeatherPromptEncoder(**weather_prompt_cfg)

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
            weather_labels=list(weather_labels),
        )

    def compute_loss(
        self,
        batch: PreparedBatch,
        prediction_type: str = "epsilon",
        noise_offset: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Run the ControlNet-conditional diffusion forward pass.

        Returns
        -------
        dict
            ``"loss"`` is mandatory (used for backprop).  ``"loss_l1"`` is
            an auxiliary reconstruction metric for logging only.
        """
        # 1. ControlNet forward pass with the degraded image as the hint.
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            batch.noisy_latents,
            batch.timesteps,
            encoder_hidden_states=batch.prompt_embeds,
            controlnet_cond=batch.degraded_pixel_values,
            return_dict=False,
        )

        # 2. SD2 UNet forward pass, conditioned on the ControlNet residuals.
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
            # Offset the noise slightly to combat low-frequency bias.
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

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Auxiliary L1 in pixel space: approximate x0 and compare to degraded
        # input (cheap, detached; used purely as a readable logging metric).
        with torch.no_grad():
            if prediction_type == "epsilon":
                pred_x0 = (batch.noisy_latents - model_pred
                           * self.noise_scheduler.init_noise_sigma) / self.vae.config.scaling_factor
            else:
                pred_x0 = model_pred
            try:
                recon = self.vae.decode(pred_x0.to(dtype=self.dtype)).sample.clamp(-1, 1)
                loss_l1 = F.l1_loss(recon, batch.degraded_pixel_values)
            except Exception:
                loss_l1 = torch.tensor(0.0, device=self.device)

        return {"loss": loss, "loss_l1": loss_l1.detach()}

    # ------------------------------------------------------------------ #
    # Inference (used by the validation step in train.py)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        degraded_pixel_values: torch.Tensor,
        weather_labels: Sequence[str],
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Run the full SD2 + ControlNet sampling loop."""
        from diffusers import StableDiffusionControlNetPipeline

        wp = self.weather_prompt_encoder.build_prompts(weather_labels)
        uncond_prompt = [self.weather_prompt_encoder.get_unconditional_prompt()] * len(weather_labels)

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

        pil_hint = self._tensor_to_pil(degraded_pixel_values)

        out = pipe(
            prompt=wp.prompts,
            negative_prompt=uncond_prompt,
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

    def load_controlnet(self, path: str) -> None:
        from diffusers import ControlNetModel
        self.controlnet = ControlNetModel.from_pretrained(path).to(self.device, dtype=self.dtype)
        self.controlnet.requires_grad_(True)