"""
RAVE (Realtime Audio Variational autoEncoder) wrapper.

Reference:
    Caillon, A., & Esling, P. (2021). RAVE: A variational autoencoder
    for fast and high-quality neural audio synthesis.
    https://github.com/acids-ircam/RAVE
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from timbre_morpher.models.base import AudioVAE, ModelConfig

logger = logging.getLogger(__name__)


# RAVE model API from IRCAM
RAVE_API_BASE = "https://play.forum.ircam.fr/rave-vst-api/get_model"

# Available pretrained models (as of 2024)
RAVE_PRETRAINED_MODELS = [
    "VCTK",           # Speech (VCTK dataset)
    "darbouka_onnx",  # Darbouka percussion
    "nasa",           # Apollo 11 recordings
    "percussion",     # Various percussion
    "vintage",        # 80h vintage music
    "isis",           # Vocal ISiS database
    "musicnet",       # MusicNet dataset
    "sol_ordinario",  # IRCAM Studio OnLine (ordinario)
    "sol_full",       # IRCAM Studio OnLine (full)
    "sol_ordinario_fast",  # Smaller/faster version
]


@dataclass
class RAVEConfig(ModelConfig):
    """Configuration for RAVE models."""

    checkpoint: str = "musicnet"
    cache_dir: str | None = None
    # RAVE defaults
    sample_rate: int = 48000
    latent_dim: int = 128
    hop_length: int = 2048


class RAVEWrapper(AudioVAE):
    """Wrapper for RAVE models."""

    def __init__(
        self,
        model: nn.Module | None = None,
        config: RAVEConfig | None = None,
    ) -> None:
        config = config or RAVEConfig()
        super().__init__(config)

        self.rave_config = config
        self._model = model
        self._is_loaded = model is not None

    @property
    def model(self) -> nn.Module:
        """Get the underlying RAVE model, loading if necessary."""
        if not self._is_loaded:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load the RAVE model from checkpoint."""
        # Download pretrained model if needed
        checkpoint = self.rave_config.checkpoint
        if checkpoint in RAVE_PRETRAINED_MODELS:
            path = self._download_pretrained(checkpoint)
        else:
            path = Path(checkpoint)
            if not path.exists():
                raise FileNotFoundError(
                    f"RAVE checkpoint not found: {checkpoint}. "
                    f"Available pretrained models: {RAVE_PRETRAINED_MODELS}"
                )

        # Load model
        logger.info(f"Loading RAVE model from {path}")
        self._model = torch.jit.load(str(path), map_location="cpu")
        self._model.eval()
        self._is_loaded = True
        logger.info("RAVE model loaded successfully")

    def _download_pretrained(self, name: str) -> Path:
        """Download a pretrained model if not cached."""
        # Create cache directory
        cache_dir = Path(self.rave_config.cache_dir or Path.home() / ".cache" / "rave")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download model
        model_path = cache_dir / f"{name}.ts"
        if model_path.exists():
            logger.info(f"Using cached model: {model_path}")
            return model_path

        url = f"{RAVE_API_BASE}/{name}"
        logger.info(f"Downloading RAVE model '{name}' from {url}")

        urllib.request.urlretrieve(url, model_path)
        logger.info(f"Model saved to {model_path}")

        return model_path
    
    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> "RAVEWrapper":
        """Load a pretrained RAVE model.

        Args:
            name_or_path: Pretrained model name (e.g., "vintage") or path to checkpoint.
            **kwargs: Additional config options.

        Available models: VCTK, darbouka_onnx, nasa, percussion, vintage, isis,
                         musicnet, sol_ordinario, sol_full, sol_ordinario_fast
        """
        config = RAVEConfig(checkpoint=name_or_path, **kwargs)
        wrapper = cls(config=config)
        _ = wrapper.model  # Force load
        return wrapper

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform to RAVE latent space.

        Args:
            audio: Input audio tensor of shape (samples,), (batch, samples),
                   or (batch, channels, samples).

        Returns:
            Latent tensor of shape (batch, latent_dim, time_frames).
        """
        # Preprocess audio for RAVE
        audio = self._preprocess_audio(audio)

        # Encode in latent space
        with torch.no_grad():
            z = self.model.encode(audio)

        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode RAVE latent representation to audio.

        Args:
            latent: Latent tensor of shape (batch, latent_dim, time_frames).

        Returns:
            Audio tensor of shape (batch, 1, samples).
        """
        # Decode from latent space
        with torch.no_grad():
            audio = self.model.decode(latent)

        return audio

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Prepare audio tensor for RAVE encoding.

        Returns audio of shape (batch, 1, samples).
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1)

        # Convert to mono if needed
        if audio.shape[1] > 1:
            audio = audio[:, :1, :]

        return audio


class MockRAVE(AudioVAE):
    """Mock RAVE model for testing without loading real weights."""

    def __init__(self, config: RAVEConfig | None = None) -> None:
        config = config or RAVEConfig()
        super().__init__(config)

        self._encoder_proj = nn.Conv1d(
            1, config.latent_dim,
            kernel_size=config.hop_length,
            stride=config.hop_length,
        )
        self._decoder_proj = nn.ConvTranspose1d(
            config.latent_dim, 1,
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
