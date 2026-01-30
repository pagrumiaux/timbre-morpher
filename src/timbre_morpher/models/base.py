"""
Abstract base classes for audio VAE models.

This module defines the interface that all audio encoder/decoder
backends must implement, allowing for easy swapping of different
model architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Base configuration for audio VAE models.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        latent_dim: Dimensionality of the latent space.
        hop_length: Hop length for encoding (samples per latent frame).
    """

    sample_rate: int = 48000
    latent_dim: int = 128
    hop_length: int = 512


class AudioVAE(nn.Module, ABC):
    """Abstract base class for audio variational autoencoders.

    Subclasses must implement encode() and decode() methods.
    This allows swapping different model architectures (RAVE, custom VAE, etc.)
    while maintaining a consistent interface.

    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the audio VAE.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        super().__init__()
        self.config = config or ModelConfig()

    @property
    def sample_rate(self) -> int:
        """Audio sample rate this model expects."""
        return self.config.sample_rate

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        return self.config.latent_dim

    @property
    def hop_length(self) -> int:
        """Number of audio samples per latent frame."""
        return self.config.hop_length

    @abstractmethod
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to latent representation.

        Args:
            audio: Input audio tensor of shape (batch, channels, samples)
                or (batch, samples) for mono.

        Returns:
            Latent tensor of shape (batch, latent_dim, time_frames).
        """
        pass

    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to audio waveform.

        Args:
            latent: Latent tensor of shape (batch, latent_dim, time_frames).

        Returns:
            Audio tensor of shape (batch, channels, samples).
        """
        pass

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode and decode (reconstruction).

        Args:
            audio: Input audio tensor.

        Returns:
            Reconstructed audio tensor.
        """
        z = self.encode(audio)
        return self.decode(z)

    def get_latent_size(self, audio_length: int) -> int:
        """Calculate latent sequence length for given audio length.

        Args:
            audio_length: Number of audio samples.

        Returns:
            Number of latent time frames.
        """
        return audio_length // self.hop_length

    def get_audio_size(self, latent_length: int) -> int:
        """Calculate audio length for given latent sequence length.

        Args:
            latent_length: Number of latent time frames.

        Returns:
            Number of audio samples.
        """
        return latent_length * self.hop_length

    @classmethod
    def from_pretrained(cls, path: str | Path, **kwargs: Any) -> "AudioVAE":
        """Load a pretrained model from a checkpoint.

        Args:
            path: Path to the checkpoint file or model identifier.
            **kwargs: Additional arguments for model initialization.

        Returns:
            Loaded model instance.

        Raises:
            NotImplementedError: If the subclass doesn't implement this.
        """
        raise NotImplementedError(
            f"{cls.__name__} does not support loading from pretrained checkpoints. "
            "Override from_pretrained() to enable this."
        )


class TestVAE(AudioVAE):
    """Dummy VAE for tests.

    Simply passes audio through a single linear layer in the latent space.
    Used for testing the pipeline without loading heavy models.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config)
        # Simple linear projection for testing
        self._encoder = nn.Conv1d(1, self.latent_dim, kernel_size=self.hop_length, stride=self.hop_length)
        self._decoder = nn.ConvTranspose1d(self.latent_dim, 1, kernel_size=self.hop_length, stride=self.hop_length)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent space."""
        # Ensure (batch, channels, samples) format
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)

        return self._encoder(audio)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to audio."""
        return self._decoder(latent)
