"""Model wrappers around Stable Diffusion 2 + ControlNet.

Exposes two building blocks:

* :class:`WeatherPromptEncoder`  – builds text prompts conditioned on the
  current weather class (rain / snow / fog) and applies classifier-free
  guidance dropout.
* :class:`ControlNetRestorationModel` – wraps a frozen SD2 UNet + VAE +
  text-encoder together with a trainable ControlNet used to inject the
  degraded image as a restoration hint.
"""

from .controlnet_wrapper import ControlNetRestorationModel
from .weather_conditioning import WeatherPromptEncoder

__all__ = [
    "ControlNetRestorationModel",
    "WeatherPromptEncoder",
]