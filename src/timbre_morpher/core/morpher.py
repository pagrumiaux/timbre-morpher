"""Main TimbreMorpher class - the high-level API for audio morphing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from timbre_morpher.core.interpolator import Interpolator
from timbre_morpher.models.base import AudioVAE
from timbre_morpher.models.rave_wrapper import RAVEWrapper, MockRAVE
from timbre_morpher.models.encodec_wrapper import EncodecWrapper
from timbre_morpher.models.dac_wrapper import DACWrapper
from timbre_morpher.utils.audio import (
    save_audio,
    match_length,
    crossfade_sequence,
)

logger = logging.getLogger(__name__)


@dataclass
class MorphConfig:
    """Configuration for morphing operations."""

    normalize_output: bool = True


@dataclass
class MorphResult:
    """Result of a morphing operation."""

    audio: list[torch.Tensor]
    latents: list[torch.Tensor]
    sample_rate: int

    def __len__(self) -> int:
        return len(self.audio)

    def concatenate(self, crossfade_samples: int = 0) -> torch.Tensor:
        """Concatenate all audio steps into single tensor."""
        if crossfade_samples > 0:
            return crossfade_sequence(self.audio, crossfade_samples)
        return torch.cat(self.audio, dim=-1)


class TimbreMorpher:
    """High-level interface for timbre morphing.

    Transforms audio smoothly from one timbre to another by interpolating
    through the latent space of a variational autoencoder.
    """

    def __init__(
        self,
        model: AudioVAE | Literal["rave", "encodec", "dac", "mock"] | None = None,
        checkpoint: str = "vintage",
        config: MorphConfig | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize TimbreMorpher.

        Args:
            model: Model backend - can be:
                - An AudioVAE instance
                - "rave" to use RAVE with specified checkpoint
                - "encodec" to use EnCodec (48kHz, high quality)
                - "dac" to use DAC (44.1kHz, optimized for music)
                - "mock" for testing without real model
                - None defaults to "mock" (for safety)
            checkpoint: Checkpoint name/path for model.
                For RAVE: "vintage", "musicnet", "VCTK", etc.
                For EnCodec: "encodec_48khz"
                For DAC: "dac_16khz", "dac_24khz", or "dac_44khz"
            config: Morphing configuration.
            device: Device to use (auto-detected if None).
        """
        self.config = config or MorphConfig()
        self.device = device or self._get_default_device()

        # Initialize model
        if model is None or model == "mock":
            logger.info("Using MockRAVE for testing (no model weights loaded)")
            self._model = MockRAVE()
        elif model == "rave":
            logger.info(f"Loading RAVE model: {checkpoint}")
            self._model = RAVEWrapper.from_pretrained(checkpoint)
        elif model == "encodec":
            checkpoint = checkpoint if checkpoint != "vintage" else "encodec_48khz"
            logger.info(f"Loading EnCodec model: {checkpoint}")
            self._model = EncodecWrapper.from_pretrained(checkpoint)
        elif model == "dac":
            checkpoint = checkpoint if checkpoint != "vintage" else "dac_44khz"
            logger.info(f"Loading DAC model: {checkpoint}")
            self._model = DACWrapper.from_pretrained(checkpoint)
        elif isinstance(model, AudioVAE):
            self._model = model
        else:
            raise ValueError(
                f"Invalid model specification: {model}. "
                "Use 'rave', 'encodec', 'dac', 'mock', or an AudioVAE instance."
            )

        self._model = self._model.to(self.device)
        self._model.eval()

        # Initialize interpolator
        self._interpolator = Interpolator()

    @staticmethod
    def _get_default_device() -> torch.device:
        """Determine best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @property
    def sample_rate(self) -> int:
        """Model's native sample rate."""
        return self._model.sample_rate

    def morph(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        steps: int = 10,
    ) -> MorphResult:
        """Morph between two audio tensors.

        Args:
            source: Source audio tensor of shape (samples,), (channels, samples),
                    or (batch, channels, samples).
            target: Target audio tensor (same shape options as source).
            steps: Number of intermediate morphing steps.

        Returns:
            MorphResult containing audio and latent sequences.
        """
        source_audio = self._ensure_batch_shape(source)
        target_audio = self._ensure_batch_shape(target)

        # Match lengths
        source_audio, target_audio = match_length(source_audio, target_audio)

        # Move to device
        source_audio = source_audio.to(self.device)
        target_audio = target_audio.to(self.device)

        # Encode
        logger.info("Encoding source audio...")
        z_source = self._model.encode(source_audio)

        logger.info("Encoding target audio...")
        z_target = self._model.encode(target_audio)

        # Generate trajectory
        logger.info(f"Generating {steps} interpolation steps...")
        z_trajectory = self._interpolator.trajectory(
            z_source,
            z_target,
            steps=steps,
        )

        # Decode multiple steps
        logger.info("Decoding morphed audio...")
        audio_sequence = []
        for z in z_trajectory:
            audio = self._model.decode(z)

            # Normalize
            if self.config.normalize_output:
                audio = audio / (audio.abs().max() + 1e-8)

            audio_sequence.append(audio.cpu())

        return MorphResult(
            audio=audio_sequence,
            latents=[z.cpu() for z in z_trajectory],
            sample_rate=self.sample_rate,
        )

    def _ensure_batch_shape(self, audio: torch.Tensor) -> torch.Tensor:
        """Ensure audio tensor has shape (batch, channels, samples)."""
        if audio.dim() == 1:
            return audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            return audio.unsqueeze(0)
        return audio

    def get_latent_trajectory(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        steps: int = 50,
    ) -> list[torch.Tensor]:
        """Get only the latent trajectory without decoding.

        Useful for visualization without the overhead of audio synthesis.
        """
        source_audio = self._ensure_batch_shape(source)
        target_audio = self._ensure_batch_shape(target)
        source_audio, target_audio = match_length(source_audio, target_audio)

        z_source = self._model.encode(source_audio.to(self.device))
        z_target = self._model.encode(target_audio.to(self.device))

        return self._interpolator.trajectory(z_source, z_target, steps=steps)

    def save_sequence(
        self,
        result: MorphResult,
        output_dir: str | Path,
        format: str = "wav",
        prefix: str = "morph",
    ) -> list[Path]:
        """Save morphing sequence to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        n_digits = len(str(len(result)))

        for i, audio in enumerate(result.audio):
            filename = f"{prefix}_{str(i).zfill(n_digits)}.{format}"
            path = output_dir / filename

            # Remove batch dimension if present
            if audio.dim() == 3:
                audio = audio.squeeze(0)

            save_audio(audio, path, result.sample_rate)
            paths.append(path)
            logger.debug(f"Saved: {path}")

        logger.info(f"Saved {len(paths)} audio files to {output_dir}")
        return paths

    def save_concatenated(
        self,
        result: MorphResult,
        output_path: str | Path,
        crossfade_ms: float = 20.0,
    ) -> Path:
        """Save entire sequence as single concatenated file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        crossfade_samples = int(crossfade_ms * result.sample_rate / 1000)
        audio = result.concatenate(crossfade_samples)

        if audio.dim() == 3:
            audio = audio.squeeze(0)

        save_audio(audio, output_path, result.sample_rate)
        logger.info(f"Saved concatenated audio to {output_path}")

        return output_path
