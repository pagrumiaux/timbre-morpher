"""Interpolation engine for latent space traversal."""

from __future__ import annotations

import torch


class Interpolator:
    """Interpolates between latent representations using linear interpolation."""

    def __call__(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """Interpolate between two latent vectors.

        Args:
            z1: Source latent tensor.
            z2: Target latent tensor.
            alpha: Interpolation factor (0 = z1, 1 = z2).

        Returns:
            Interpolated latent tensor.
        """
        return (1 - alpha) * z1 + alpha * z2

    def trajectory(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        steps: int,
    ) -> list[torch.Tensor]:
        """Generate a sequence of interpolated latents.

        Args:
            z1: Source latent tensor.
            z2: Target latent tensor.
            steps: Number of intermediate steps.

        Returns:
            List of interpolated latent tensors.
        """
        alphas = torch.linspace(0, 1, steps + 2)

        return [self(z1, z2, alpha.item()) for alpha in alphas]


def interpolate(
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Convenience function for one-off interpolation.

    Args:
        z1: Source latent tensor.
        z2: Target latent tensor.
        alpha: Interpolation factor (0 = z1, 1 = z2).

    Returns:
        Interpolated latent tensor.
    """
    return Interpolator()(z1, z2, alpha)
