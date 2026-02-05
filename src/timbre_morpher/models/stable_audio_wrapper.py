"""
Stable Audio Open (AutoencoderOobleck) wrapper.

Reference:
    Evans, Z., et al. (2024). Stable Audio Open.
    https://github.com/Stability-AI/stable-audio-tools
    https://huggingface.co/stabilityai/stable-audio-open-1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from timbre_morpher.models.base import AudioVAE, ModelConfig

logger = logging.getLogger(__name__)


# Pretrained Stable Audio models via HuggingFace.
# Only the VAE component (AutoencoderOobleck) is used for morphing.
STABLE_AUDIO_PRETRAINED_MODELS = {
    "stable-audio-open-1.0": {
        "model_id": "stabilityai/stable-audio-open-1.0",
        "subfolder": "vae",
    },
}


@dataclass
class StableAudioConfig(ModelConfig):
    """Configuration for Stable Audio models."""

    checkpoint: str = "stable-audio-open-1.0"
    sample_rate: int = 44100
    latent_dim: int = 64
    hop_length: int = 2048


class StableAudioWrapper(AudioVAE):
    """Wrapper for Stable Audio Open's AutoencoderOobleck VAE.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        config: StableAudioConfig | None = None,
    ) -> None:
        config = config or StableAudioConfig()
        super().__init__(config)

        self.stable_audio_config = config
        self._model = model
        self._is_loaded = model is not None

    @property
    def model(self) -> nn.Module:
        """Get the underlying AutoencoderOobleck model, loading if necessary."""
        if not self._is_loaded:
            # Protect against AttributeError swallowing by nn.Module.__getattr__
            try:
                self._load_model()
            except AttributeError as e:
                raise RuntimeError(f"Failed to load Stable Audio model: {e}") from e
        return self._model # pyright: ignore[reportReturnType]

    def _load_model(self) -> None:
        """Load the AutoencoderOobleck VAE from HuggingFace."""
        try:
            from diffusers import AutoencoderOobleck
        except ImportError:
            raise ImportError(
                "The 'diffusers' package is required for Stable Audio support. "
                "Install it with: pip install diffusers transformers"
            )

        checkpoint = self.stable_audio_config.checkpoint

        if checkpoint not in STABLE_AUDIO_PRETRAINED_MODELS:
            raise ValueError(
                f"Stable Audio checkpoint not found: {checkpoint}. "
                f"Available models: {list(STABLE_AUDIO_PRETRAINED_MODELS.keys())}"
            )

        model_spec = STABLE_AUDIO_PRETRAINED_MODELS[checkpoint]
        logger.info(f"Loading Stable Audio VAE: {checkpoint}")

        self._model = AutoencoderOobleck.from_pretrained(
            model_spec["model_id"],
            subfolder=model_spec["subfolder"],
        )

        # Sync config from loaded model attributes
        self.stable_audio_config.sample_rate = self._model.config.sampling_rate
        self.stable_audio_config.latent_dim = self._model.config.decoder_input_channels
        # hop_length = product of downsampling_ratios
        downsampling_ratios = self._model.config.downsampling_ratios
        self.stable_audio_config.hop_length = int(torch.tensor(downsampling_ratios).prod().item())

        self._model.eval()
        self._is_loaded = True

        logger.info(
            f"Stable Audio VAE loaded: {self.stable_audio_config.sample_rate}Hz, "
            f"latent_dim={self.stable_audio_config.latent_dim}, "
            f"hop_length={self.stable_audio_config.hop_length}"
        )

    @classmethod
    def from_pretrained(cls, name: str = "stable-audio-open-1.0", **kwargs: Any) -> "StableAudioWrapper":
        """Load a pretrained Stable Audio VAE.

        Args:
            name: Pretrained model name (currently only "stable-audio-open-1.0").
            **kwargs: Additional config options.

        Available models:
            - stable-audio-open-1.0: 44.1kHz stereo VAE from Stability AI
        """
        config = StableAudioConfig(checkpoint=name, **kwargs)
        wrapper = cls(config=config)
        _ = wrapper.model  # Force load
        return wrapper

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to continuous latent space.

        AutoencoderOobleck returns a DiagonalGaussianDistribution. We use .mode()
        to get the deterministic latent (mean), which is best for morphing.

        Args:
            audio: Input audio tensor of shape (samples,), (batch, samples),
                   or (batch, channels, samples).

        Returns:
            Latent tensor of shape (batch, latent_dim, time_frames).
        """
        audio = self._preprocess_audio(audio)

        with torch.no_grad():
            output = self.model.encode(audio)
            z = output.latent_dist.mode()

        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to audio.

        Args:
            latent: Latent tensor of shape (batch, latent_dim, time_frames).

        Returns:
            Audio tensor of shape (batch, 2, samples) (stereo).
        """
        with torch.no_grad():
            output = self.model.decode(latent)
            audio = output.sample

        return audio

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Prepare audio tensor for AutoencoderOobleck encoding.

        Returns stereo audio of shape (batch, 2, samples).
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # AutoencoderOobleck expects stereo (2 channels)
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]

        return audio


class MockStableAudio(AudioVAE):
    """Mock Stable Audio VAE for testing without loading real weights."""

    def __init__(self, config: StableAudioConfig | None = None) -> None:
        config = config or StableAudioConfig()
        super().__init__(config)

        self._encoder_proj = nn.Conv1d(
            2,  # Stereo input
            config.latent_dim,
            kernel_size=config.hop_length,
            stride=config.hop_length,
        )
        self._decoder_proj = nn.ConvTranspose1d(
            config.latent_dim,
            2,  # Stereo output
            kernel_size=config.hop_length,
            stride=config.hop_length,
        )
        nn.init.xavier_normal_(self._encoder_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self._decoder_proj.weight, gain=0.1)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Convert to stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]

        return self._encoder_proj(audio)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decoder_proj(latent)
