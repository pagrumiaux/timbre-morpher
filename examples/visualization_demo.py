#!/usr/bin/env python3
"""
Latent space visualization example.

This script demonstrates how to visualize morphing trajectories
in the latent space using different dimensionality reduction methods.

Usage:
    python examples/visualization_demo.py
"""

import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run visualization demo."""
    from timbre_morpher import TimbreMorpher
    from timbre_morpher.utils.visualization import (
        plot_trajectory,
        plot_latent_dimensions,
    )

    # Output directory
    output_dir = Path("output/visualization_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize with mock model for quick demo
    logger.info("Initializing morpher...")
    morpher = TimbreMorpher(model="mock")

    # Generate some synthetic audio for demo
    torch.manual_seed(42)
    duration_samples = 48000 * 2  # 2 seconds at 48kHz
    source_audio = torch.randn(1, 1, duration_samples) * 0.5
    target_audio = torch.randn(1, 1, duration_samples) * 0.5

    # Generate trajectory
    logger.info("Generating morphing trajectory...")
    result = morpher.morph(
        source=source_audio,
        target=target_audio,
        steps=50,
    )

    # Show latent dimension evolution
    logger.info("Generating latent dimension plot...")
    _ = plot_latent_dimensions(
        result.latents,
        n_dims=8,
        save_path=output_dir / "latent_dimensions.png",
        show=False,
    )
    logger.info("Saved: latent_dimensions.png")

    # Try different reduction methods
    logger.info("Comparing dimensionality reduction methods...")

    for reduction_method in ["pca", "tsne"]:
        try:
            _ = plot_trajectory(
                result.latents,
                method=reduction_method,
                dimensions=2,
                title=f"Morphing trajectory ({reduction_method.upper()})",
                save_path=output_dir / f"trajectory_{reduction_method}.png",
                show=False,
            )
            logger.info(f"Saved: trajectory_{reduction_method}.png")
        except ImportError as e:
            logger.warning(f"Could not use {reduction_method}: {e}")

    # Try UMAP if available
    try:
        _ = plot_trajectory(
            result.latents,
            method="umap",
            dimensions=2,
            title="Morphing trajectory (UMAP)",
            save_path=output_dir / "trajectory_umap.png",
            show=False,
        )
        logger.info("Saved: trajectory_umap.png")
    except ImportError:
        logger.info("UMAP not available. Install with: pip install umap-learn")

    logger.info(f"\nAll visualizations saved to: {output_dir}")
    logger.info("\nFiles generated:")
    for f in sorted(output_dir.glob("*.png")):
        logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()
