"""Weather-class prompt encoder.

Given a batch of weather labels (``"rain"``, ``"snow"``, ``"fog"`` ...) this
module produces the corresponding text prompts that are then tokenised by the
SD2 text encoder.

Two design choices make this module "pluggable":

1. ``use_weather_prompt=False`` forces the prompt to be a fixed empty string,
   effectively removing the weather-class signal from the model.  This lets
   you ablate the contribution of weather conditioning without changing any
   other code.

2. ``cfg_dropout_prob`` implements the unconditional branch of classifier-free
   guidance: with that probability the prompt is replaced by an empty string
   so the model also learns to restore images without weather information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch


@dataclass
class WeatherPromptOutput:
    """Container for prompts produced by :class:`WeatherPromptEncoder`."""

    prompts: List[str]      # final prompts fed to the text encoder
    raw_weather: List[str]  # original weather labels before dropout


class WeatherPromptEncoder:
    """Build SD2 text prompts from weather labels.

    Parameters
    ----------
    use_weather_prompt
        Master switch controlled by ``config/model.yaml``.
    prompt_template
        Format string that receives the weather token, e.g.
        ``"a clean photo after removing {weather}, high quality"``.
    empty_prompt
        Prompt used when ``use_weather_prompt`` is ``False`` or when CFG
        dropout fires.
    weather_tokens
        Mapping from dataset folder name (e.g. ``"rain"``) to natural
        language token (e.g. ``"rain"``).
    cfg_dropout_prob
        Probability of replacing the prompt with the empty string.
    """

    def __init__(
        self,
        use_weather_prompt: bool = True,
        prompt_template: str = "a clean photo after removing {weather}, high quality, sharp",
        empty_prompt: str = "",
        weather_tokens: Optional[Dict[str, str]] = None,
        cfg_dropout_prob: float = 0.1,
    ) -> None:
        self.use_weather_prompt = use_weather_prompt
        self.prompt_template = prompt_template
        self.empty_prompt = empty_prompt
        self.weather_tokens = weather_tokens or {"rain": "rain", "snow": "snow", "fog": "fog"}
        self.cfg_dropout_prob = float(cfg_dropout_prob)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build_prompts(
        self,
        weather_labels: Sequence[str],
        generator: Optional[torch.Generator] = None,
    ) -> WeatherPromptOutput:
        """Build the prompt list for a batch of weather labels.

        Parameters
        ----------
        weather_labels
            Iterable of length ``B`` with weather names (e.g. ``["rain",
            "snow", "fog", "rain"]``).
        generator
            Optional torch RNG used for the CFG dropout decision.  Falls
            back to the global RNG if not provided.
        """
        labels = list(weather_labels)

        if not self.use_weather_prompt:
            prompts = [self.empty_prompt] * len(labels)
            return WeatherPromptOutput(prompts=prompts, raw_weather=labels)

        prompts: List[str] = []
        for label in labels:
            token = self.weather_tokens.get(label, label)
            prompts.append(self.prompt_template.format(weather=token))

        # Classifier-free guidance dropout: replace some prompts with empty.
        if self.cfg_dropout_prob > 0.0:
            probs = torch.rand(len(prompts), generator=generator)
            for i, p in enumerate(probs.tolist()):
                if p < self.cfg_dropout_prob:
                    prompts[i] = self.empty_prompt

        return WeatherPromptOutput(prompts=prompts, raw_weather=labels)

    # ------------------------------------------------------------------ #
    # Convenience helpers
    # ------------------------------------------------------------------ #
    def get_unconditional_prompt(self) -> str:
        """Prompt used during validation-time CFG (negative branch)."""
        return self.empty_prompt

    def __repr__(self) -> str:
        return (
            f"WeatherPromptEncoder(use_weather_prompt={self.use_weather_prompt}, "
            f"cfg_dropout_prob={self.cfg_dropout_prob}, "
            f"tokens={list(self.weather_tokens.keys())})"
        )