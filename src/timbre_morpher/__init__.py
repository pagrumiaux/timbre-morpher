"""Timbre Morpher: Intelligent audio morphing for creative sound design."""

from timbre_morpher.core.morpher import TimbreMorpher
from timbre_morpher.core.interpolator import Interpolator
from timbre_morpher.models.base import AudioVAE

__version__ = "0.1.0"
__all__ = [
    "TimbreMorpher",
    "Interpolator",
    "AudioVAE",
]
