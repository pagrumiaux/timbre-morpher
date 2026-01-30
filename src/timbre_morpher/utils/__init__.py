"""Utility functions for audio processing and visualization."""

from timbre_morpher.utils.audio import (
    load_audio,
    save_audio,
    resample,
    match_length,
    crossfade_sequence,
    normalize_loudness,
)
from timbre_morpher.utils.visualization import (
    plot_trajectory,
    plot_latent_dimensions,
)

__all__ = [
    "load_audio",
    "save_audio",
    "resample",
    "match_length",
    "crossfade_sequence",
    "normalize_loudness",
    "plot_trajectory",
    "plot_latent_dimensions",
    "create_trajectory_animation",
]
