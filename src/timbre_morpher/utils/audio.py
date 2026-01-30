"""
Audio utility functions for loading, saving, and processing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchaudio
import soundfile as sf


def load_audio(
    path: str | Path,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load audio file to tensor.

    Args:
        path: Path to audio file.
        mono: Convert to mono if True.

    Returns:
        Tuple of (audio_tensor, sample_rate).
        Audio tensor shape: (channels, samples) or (samples,) if mono.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # Use soundfile (more reliable than torchaudio.load)
    # soundfile returns (samples, channels) format
    audio_np, sr = sf.read(str(path))

    # Convert to tensor and transpose to (channels, samples)
    audio = torch.from_numpy(audio_np.T).float()

    # Handle mono files (1D array from soundfile)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    if mono and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    return audio, sr


def save_audio(
    audio: torch.Tensor,
    path: str | Path,
    sample_rate: int,
    normalize: bool = False,
) -> None:
    """Save audio tensor to file.

    Args:
        audio: Audio tensor of shape (channels, samples) or (samples,).
        path: Output file path.
        sample_rate: Sample rate in Hz.
        normalize: Normalize audio to [-1, 1] before saving.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure 2D: (channels, samples)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    elif audio.dim() == 3:
        # (batch, channels, samples) -> (channels, samples)
        audio = audio.squeeze(0)

    if normalize:
        audio = audio / (audio.abs().max() + 1e-8)

    # Ensure float32 for saving
    audio = audio.float()

    # Use soundfile (more reliable than torchaudio.save)
    # soundfile expects (samples, channels) format
    audio_np = audio.detach().numpy().T
    sf.write(str(path), audio_np, sample_rate)


def resample(
    audio: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """Resample audio.

    Args:
        audio: Audio tensor.
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled audio tensor
    """
    # If already at target sample rate, no need to resample
    if orig_sr == target_sr:
        return audio

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(audio)


def match_length(
    audio1: torch.Tensor,
    audio2: torch.Tensor,
    mode: str = "min",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match the length of two audio tensors.

    Args:
        audio1: First audio tensor.
        audio2: Second audio tensor.
        mode: How to match lengths:
            - "min": Truncate to shorter length
            - "max": Pad shorter to longer length

    Returns:
        Tuple of length-matched audio tensors.
    """
    len1 = audio1.shape[-1]
    len2 = audio2.shape[-1]

    if len1 == len2:
        return audio1, audio2

    if mode == "min":
        target_len = min(len1, len2)
    elif mode == "max":
        target_len = max(len1, len2)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'min' or 'max'.")

    audio1 = _adjust_length(audio1, target_len)
    audio2 = _adjust_length(audio2, target_len)

    return audio1, audio2


def _adjust_length(audio: torch.Tensor, target_len: int) -> torch.Tensor:
    """Adjust audio length by truncating or padding.

    Args:
        audio: Audio tensor.
        target_len: Desired length.

    Returns:
        Adjusted audio tensor.
    """
    current_len = audio.shape[-1]

    if current_len == target_len:
        return audio
    elif current_len > target_len:
        # Truncate
        return audio[..., :target_len]
    else:
        # Pad with zeros
        pad_len = target_len - current_len
        padding = torch.zeros(*audio.shape[:-1], pad_len, device=audio.device)
        return torch.cat([audio, padding], dim=-1)


def crossfade_sequence(
    audio_list: list[torch.Tensor],
    crossfade_samples: int,
) -> torch.Tensor:
    """Concatenate audio tensors with crossfading.

    Args:
        audio_list: List of audio tensors to concatenate.
        crossfade_samples: Number of samples to crossfade between segments.

    Returns:
        Concatenated audio tensor with smooth transitions.
    """
    if len(audio_list) == 0:
        raise ValueError("Empty audio list")

    if len(audio_list) == 1:
        return audio_list[0]

    if crossfade_samples <= 0:
        return torch.cat(audio_list, dim=-1)

    result = audio_list[0]

    for next_audio in audio_list[1:]:
        result = _crossfade_pair(result, next_audio, crossfade_samples)

    return result


def _crossfade_pair(
    audio1: torch.Tensor,
    audio2: torch.Tensor,
    crossfade_samples: int,
) -> torch.Tensor:
    """Crossfade between two audio tensors.

    Args:
        audio1: First audio tensor.
        audio2: Second audio tensor.
        crossfade_samples: Number of samples in crossfade region.

    Returns:
        Crossfaded audio tensor.
    """
    len1 = audio1.shape[-1]
    len2 = audio2.shape[-1]

    # Don't crossfade more than available
    crossfade_samples = min(crossfade_samples, len1, len2)

    if crossfade_samples <= 0:
        return torch.cat([audio1, audio2], dim=-1)

    # Fade curves
    fade_out = torch.linspace(1, 0, crossfade_samples, device=audio1.device)
    fade_in = torch.linspace(0, 1, crossfade_samples, device=audio2.device)

    # Expand dimensions
    for _ in range(audio1.dim() - 1):
        fade_out = fade_out.unsqueeze(0)
        fade_in = fade_in.unsqueeze(0)

    # Crossfade regions
    end_of_first = audio1[..., -crossfade_samples:]
    start_of_second = audio2[..., :crossfade_samples]

    # Apply crossfade
    crossfaded = end_of_first * fade_out + start_of_second * fade_in

    # Concatenate
    result = torch.cat(
        [
            audio1[..., :-crossfade_samples],
            crossfaded,
            audio2[..., crossfade_samples:],
        ],
        dim=-1,
    )

    return result


def compute_rms(audio: torch.Tensor, frame_size: int = 2048) -> torch.Tensor:
    """Compute RMS energy of audio.

    Args:
        audio: Audio tensor.
        frame_size: Analysis frame size.

    Returns:
        RMS values per frame.
    """
    # Ensure 2D
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Square
    squared = audio ** 2

    # Unfold into frames
    frames = squared.unfold(-1, frame_size, frame_size // 2)

    # Mean and sqrt
    rms = frames.mean(dim=-1).sqrt()

    return rms


def normalize_loudness(
    audio: torch.Tensor,
    target_db: float = -14.0,
) -> torch.Tensor:
    """Normalize audio to target loudness (simple peak-based).

    Args:
        audio: Audio tensor.
        target_db: Target peak level in dB.

    Returns:
        Loudness-normalized audio.
    """
    current_peak = audio.abs().max()

    if current_peak < 1e-8:
        return audio

    target_linear = 10 ** (target_db / 20)
    gain = target_linear / current_peak

    return audio * gain
