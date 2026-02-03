"""
DAC (Descript Audio Codec) wrapper.

Reference:
    Kumar, K., et al. (2024). High-Fidelity Audio Compression with Improved RVAE-GAN.
    https://github.com/descript/descript-audio-codec
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from timbre_morpher.models.base import AudioVAE, ModelConfig

logger = logging.getLogger(__name__)


# Parameters passed to dac.utils.load_model() for each pretrained variant.
# sample_rate / hop_length / latent_dim are synced from the loaded model itself.
DAC_PRETRAINED_MODELS = {
    "dac_16khz": {"model_type": "16khz", "model_bitrate": "8kbps"},
    "dac_24khz": {"model_type": "24khz", "model_bitrate": "8kbps"},
    "dac_44khz": {"model_type": "44khz", "model_bitrate": "8kbps"},
}


@dataclass
class DACConfig(ModelConfig):
    """Configuration for DAC models."""

    checkpoint: str = "dac_44khz"
    # DAC 44kHz defaults
    sample_rate: int = 44100
    latent_dim: int = 1024
    hop_length: int = 512


class DACWrapper(AudioVAE):
    """Wrapper for Descript Audio Codec (DAC) models.

    Like EnCodec, DAC uses Residual Vector Quantization (RVQ). For morphing
    we bypass quantization entirely by calling the encoder/decoder directly,
    giving us a continuous latent space suitable for interpolation.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        config: DACConfig | None = None,
    ) -> None:
        config = config or DACConfig()
        super().__init__(config)

        self.dac_config = config
        self._model = model
        self._is_loaded = model is not None

    @property
    def model(self) -> nn.Module:
        """Get the underlying DAC model, loading if necessary."""
        if not self._is_loaded:
            # Catch AttributeError explicitly: nn.Module's __getattr__ swallows
            # any AttributeError raised inside a @property, producing a confusing
            # "'X' object has no attribute 'model'" instead of the real error.
            try:
                self._load_model()
            except AttributeError as e:
                raise RuntimeError(f"Failed to load DAC model: {e}") from e
        return self._model

    def _load_model(self) -> None:
        """Load the DAC model from checkpoint."""
        try:
            from dac.utils import load_model
        except ImportError:
            raise ImportError(
                "The 'descript-audio-codec' package is required for DAC support. "
                "Install it with: pip install descript-audio-codec"
            )

        checkpoint = self.dac_config.checkpoint

        if checkpoint not in DAC_PRETRAINED_MODELS:
            raise ValueError(
                f"DAC checkpoint not found: {checkpoint}. "
                f"Available models: {list(DAC_PRETRAINED_MODELS.keys())}"
            )

        model_config = DAC_PRETRAINED_MODELS[checkpoint]
        logger.info(f"Loading DAC model: {checkpoint}")
        self._model = load_model(**model_config)

        # Sync config from the loaded model's own attributes
        self.dac_config.sample_rate = self._model.sample_rate
        self.dac_config.hop_length = self._model.hop_length
        self.dac_config.latent_dim = self._model.latent_dim

        self._model.eval()
        self._is_loaded = True

        logger.info(
            f"DAC model loaded: {self.dac_config.sample_rate}Hz, "
            f"latent_dim={self.dac_config.latent_dim}, "
            f"hop_length={self.dac_config.hop_length}"
        )

    @classmethod
    def from_pretrained(cls, name: str = "dac_44khz", **kwargs: Any) -> "DACWrapper":
        """Load a pretrained DAC model.

        Args:
            name: Pretrained model name ("dac_16khz", "dac_24khz", or "dac_44khz").
            **kwargs: Additional config options.

        Available models:
            - dac_16khz: 16kHz model
            - dac_24khz: 24kHz model
            - dac_44khz: 44.1kHz model (best for music)
        """
        config = DACConfig(checkpoint=name, **kwargs)
        wrapper = cls(config=config)
        _ = wrapper.model  # Force load
        return wrapper

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to continuous latent space (pre-quantization).

        Calls the encoder directly, bypassing RVQ, so the output is
        suitable for standard interpolation.

        Args:
            audio: Input audio tensor of shape (samples,), (batch, samples),
                   or (batch, channels, samples).

        Returns:
            Latent tensor of shape (batch, latent_dim, time_frames).
        """
        audio = self._preprocess_audio(audio)

        with torch.no_grad():
            z = self.model.encoder(audio)

        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to audio.

        Args:
            latent: Latent tensor of shape (batch, latent_dim, time_frames).

        Returns:
            Audio tensor of shape (batch, 1, samples).
        """
        with torch.no_grad():
            audio = self.model.decoder(latent)

        return audio

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Prepare audio tensor for DAC encoding.

        Returns mono audio of shape (batch, 1, samples).
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # DAC pretrained models expect mono
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        return audio


class MockDAC(AudioVAE):
    """Mock DAC model for testing without loading real weights."""

    def __init__(self, config: DACConfig | None = None) -> None:
        config = config or DACConfig()
        super().__init__(config)

        self._encoder_proj = nn.Conv1d(
            1,  # Mono input
            config.latent_dim,
            kernel_size=config.hop_length,
            stride=config.hop_length,
        )
        self._decoder_proj = nn.ConvTranspose1d(
            config.latent_dim,
            1,  # Mono output
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

        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        return self._encoder_proj(audio)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decoder_proj(latent)
