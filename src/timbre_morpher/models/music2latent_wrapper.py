"""
music2latent (Consistency Autoencoder) wrapper.

Reference:
    Pasini, M. (2024). music2latent: Consistency Autoencoders for Latent Audio Compression.
    https://github.com/SonyCSLParis/music2latent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from timbre_morpher.models.base import AudioVAE, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class Music2LatentConfig(ModelConfig):
    """Configuration for music2latent models."""

    sample_rate: int = 44100
    latent_dim: int = 64
    hop_length: int = 4410  # ~10Hz at 44.1kHz


class Music2LatentWrapper(AudioVAE):
    """Wrapper for music2latent's Consistency Autoencoder."""

    def __init__(
        self,
        model: Any | None = None,
        config: Music2LatentConfig | None = None,
    ) -> None:
        config = config or Music2LatentConfig()
        super().__init__(config)

        self.m2l_config = config
        self._model = model
        self._is_loaded = model is not None

    @property
    def model(self) -> Any:
        """Get the underlying EncoderDecoder, loading if necessary."""
        if not self._is_loaded:
            try:
                self._load_model()
            except AttributeError as e:
                raise RuntimeError(f"Failed to load music2latent model: {e}") from e
        return self._model

    def _load_model(self) -> None:
        """Load the music2latent EncoderDecoder."""
        try:
            from music2latent import EncoderDecoder
        except ImportError:
            raise ImportError(
                "The 'music2latent' package is required. "
                "Install it with: pip install music2latent"
            )

        logger.info("Loading music2latent EncoderDecoder (weights auto-download on first use)...")
        self._model = EncoderDecoder()
        self._is_loaded = True
        logger.info(
            f"music2latent loaded: {self.m2l_config.sample_rate}Hz, "
            f"latent_dim={self.m2l_config.latent_dim}, "
            f"hop_length={self.m2l_config.hop_length}"
        )

    @classmethod
    def from_pretrained(cls, name: str = "music2latent", **kwargs: Any) -> "Music2LatentWrapper":
        """Load the music2latent model.

        Args:
            name: Ignored (only one model available). Kept for API consistency.
            **kwargs: Additional config options.
        """
        config = Music2LatentConfig(**kwargs)
        wrapper = cls(config=config)
        _ = wrapper.model  # Force load
        return wrapper

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent space.

        music2latent expects numpy arrays and treats channels as batch dim.
        We convert to mono, pass as numpy, and get back a torch tensor.

        Args:
            audio: Input tensor of shape (samples,), (batch, samples),
                   or (batch, channels, samples).

        Returns:
            Latent tensor of shape (batch, 64, time_frames).
        """
        audio = self._preprocess_audio(audio)
        batch_size = audio.shape[0]

        # music2latent expects numpy: (batch, samples)
        audio_np = audio.squeeze(1).cpu().numpy()

        with torch.no_grad():
            z = self.model.encode(audio_np)

        # Ensure torch tensor
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z)

        # z shape: (batch, 64, seq_len)
        assert z.shape[0] == batch_size
        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to audio.

        Args:
            latent: Latent tensor of shape (batch, 64, time_frames).

        Returns:
            Audio tensor of shape (batch, 1, samples) (mono).
        """
        with torch.no_grad():
            audio = self.model.decode(latent)

        # Ensure torch tensor
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio)

        # music2latent returns (batch, samples), add channel dim
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        return audio

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Prepare audio tensor for music2latent encoding.

        Returns mono audio of shape (batch, 1, samples).
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Convert to mono by averaging channels
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        return audio


class MockMusic2Latent(AudioVAE):
    """Mock music2latent for testing without loading real weights."""

    def __init__(self, config: Music2LatentConfig | None = None) -> None:
        config = config or Music2LatentConfig()
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

        # Convert to mono
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        return self._encoder_proj(audio)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self._decoder_proj(latent)
