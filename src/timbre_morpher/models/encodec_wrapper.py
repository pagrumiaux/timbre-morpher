"""
EnCodec (Neural Audio Codec) wrapper.

Reference:
    Défossez, A., et al. (2022). High Fidelity Neural Audio Compression.
    https://github.com/facebookresearch/encodec
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from timbre_morpher.models.base import AudioVAE, ModelConfig

logger = logging.getLogger(__name__)


# Available pretrained EnCodec models
ENCODEC_PRETRAINED_MODELS = {
    "encodec_48khz": {
        "sample_rate": 48000,
        "bandwidth": 12.0,  # kbps
        "latent_dim": 128,
        "hop_length": 640,
    },
}


@dataclass
class EncodecConfig(ModelConfig):
    """Configuration for EnCodec models."""

    checkpoint: str = "encodec_48khz"
    bandwidth: float = 12.0  # Target bandwidth in kbps
    # EnCodec defaults
    sample_rate: int = 48000
    latent_dim: int = 128
    hop_length: int = 640
    n_quantizers: int = 32


class EncodecWrapper(AudioVAE):
    """Wrapper for Meta's EnCodec models."""

    def __init__(
        self,
        model: nn.Module | None = None,
        config: EncodecConfig | None = None,
    ) -> None:
        config = config or EncodecConfig()
        super().__init__(config)

        self.encodec_config = config
        self._model = model
        self._is_loaded = model is not None

    @property
    def model(self) -> nn.Module:
        """Get the underlying EnCodec model, loading if necessary."""
        if not self._is_loaded:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load the EnCodec model from checkpoint."""
        try:
            from encodec import EncodecModel
        except ImportError:
            raise ImportError(
                "The 'encodec' package is required for EnCodec support. "
                "Install it with: pip install encodec"
            )

        checkpoint = self.encodec_config.checkpoint

        # Load pretrained model (48kHz only)
        if checkpoint in ENCODEC_PRETRAINED_MODELS:
            model_config = ENCODEC_PRETRAINED_MODELS[checkpoint]
            logger.info(f"Loading EnCodec model: {checkpoint}")
            self._model = EncodecModel.encodec_model_48khz()

            # Update config with model-specific settings
            self.encodec_config.sample_rate = model_config["sample_rate"]
            self.encodec_config.hop_length = model_config["hop_length"]

        else:
            raise ValueError(
                f"EnCodec checkpoint not found: {checkpoint}. "
                f"Available models: {list(ENCODEC_PRETRAINED_MODELS.keys())}"
            )

        # Set bandwidth
        self._model.set_target_bandwidth(self.encodec_config.bandwidth)
        self._model.eval()
        self._is_loaded = True

        logger.info(
            f"EnCodec model loaded: {self.encodec_config.sample_rate}Hz, "
            f"{self.encodec_config.bandwidth}kbps bandwidth"
        )

    @classmethod
    def from_pretrained(cls, name: str = "encodec_48khz", **kwargs: Any) -> "EncodecWrapper":
        """Load a pretrained EnCodec model.

        Args:
            name: Pretrained model name ("encodec_48khz").
            **kwargs: Additional config options (e.g., bandwidth=6.0).

        Available models:
            - encodec_48khz: 48kHz model (high quality for music)
        """
        config = EncodecConfig(checkpoint=name, **kwargs)
        wrapper = cls(config=config)
        _ = wrapper.model  # Force load
        return wrapper

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to EnCodec latent space.

        Args:
            audio: Input audio tensor of shape (samples,), (batch, samples),
                   or (batch, channels, samples).

        Returns:
            Latent tensor of shape (batch, latent_dim, time_frames).
        """
        # Preprocess audio for EnCodec
        audio = self._preprocess_audio(audio)

        # Encode to latent space (quantized codes)
        with torch.no_grad():
            # EnCodec returns EncodedFrame with 'codes' (quantized) and 'scale'
            encoded = self.model.encode(audio)

            # Extract codes: (batch, n_quantizers, time_frames)
            codes = encoded[0][0]  # First frame, first batch element codes

            # For morphing, we use the continuous embeddings before quantization
            # Access the encoder's continuous output
            z = self.model.encoder(audio)

        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode EnCodec latent representation to audio.

        Args:
            latent: Latent tensor of shape (batch, latent_dim, time_frames).

        Returns:
            Audio tensor of shape (batch, channels, samples).
        """
        # Decode from latent space
        with torch.no_grad():
            audio = self.model.decoder(latent)

        return audio

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Prepare audio tensor for EnCodec encoding.

        Returns audio of shape (batch, channels, samples).
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # EnCodec 48kHz model expects stereo (2 channels)
        # Convert mono to stereo by duplicating the channel
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        # Convert to stereo if more than 2 channels
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]

        return audio


class MockEncodec(AudioVAE):
    """Mock EnCodec model for testing without loading real weights."""

    def __init__(self, config: EncodecConfig | None = None) -> None:
        config = config or EncodecConfig()
        super().__init__(config)

        self._encoder_proj = nn.Conv1d(
            2,  # Stereo input for 48kHz model
            config.latent_dim,
            kernel_size=config.hop_length,
            stride=config.hop_length,
        )
        self._decoder_proj = nn.ConvTranspose1d(
            config.latent_dim,
            2,  # Stereo output for 48kHz model
            kernel_size=config.hop_length,
            stride=config.hop_length,
        )
        nn.init.xavier_normal_(self._encoder_proj.weight, gain=0.1)
        nn.init.xavier_normal_(self._decoder_proj.weight, gain=0.1)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Convert mono to stereo for 48kHz model
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]

        return self._encoder_proj(audio)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decoder_proj(latent)
