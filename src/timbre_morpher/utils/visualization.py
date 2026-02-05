"""
Visualization utilities for latent space exploration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_trajectory(
    latents: list[torch.Tensor] | torch.Tensor,
    method: Literal["pca", "tsne", "umap"] = "pca",
    dimensions: Literal[2, 3] = 2,
    title: str = "Morphing trajectory in latent space",
    save_path: str | Path | None = None,
    show: bool = True,
    colormap: str = "viridis",
) -> Figure:
    """Plot morphing trajectory in reduced latent space.

    Args:
        latents: List of latent tensors from morphing trajectory.
        method: Dimensionality reduction method.
        dimensions: Number of dimensions for visualization (2 or 3).
        title: Plot title.
        save_path: Path to save figure (optional).
        show: Whether to display the plot.
        colormap: Matplotlib colormap for trajectory coloring.

    Returns:
        Matplotlib Figure object.
    """
    # Stack and flatten latents
    stacked = torch.stack(latents)  # (steps, batch, latent_dim, time)
    
    # Average over time dimension and remove batch
    if stacked.dim() == 4:
        stacked = stacked.mean(dim=-1).squeeze(1)  # (steps, latent_dim)
    elif stacked.dim() == 3:
        stacked = stacked.mean(dim=-1)  # (steps, latent_dim)
    
    points = stacked.detach().cpu().numpy()

    # Apply dimensionality reduction
    reduced = _reduce_dimensions(points, method, dimensions)

    # Create figure
    fig = plt.figure(figsize=(10,10))
    
    if dimensions == 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    # Color mapping based on position in trajectory
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(reduced)))

    # Plot trajectory line
    if dimensions == 2:
        ax.plot(reduced[:, 0], reduced[:, 1], "b-", alpha=0.3, linewidth=2, zorder=1)
        _ = ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=colors,
            s=100,
            zorder=2,
            edgecolors="white",
            linewidths=0.5,
        )
        
        # Mark start and end
        ax.scatter(
            reduced[0, 0],
            reduced[0, 1],
            c="green",
            s=300,
            marker="o",
            label="Source",
            zorder=3,
            edgecolors="black",
            linewidths=2,
        )
        ax.scatter(
            reduced[-1, 0],
            reduced[-1, 1],
            c="red",
            s=300,
            marker="s",
            label="Target",
            zorder=3,
            edgecolors="black",
            linewidths=2,
        )
        
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        
    else:  # 3D
        ax.plot(
            reduced[:, 0],
            reduced[:, 1],
            reduced[:, 2],
            "b-",
            alpha=0.3,
            linewidth=2,
        )
        ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            reduced[:, 2],
            c=colors,
            s=100,
            edgecolors="white",
            linewidths=0.5,
        )
        
        # Mark start and end
        ax.scatter(
            reduced[0, 0],
            reduced[0, 1],
            reduced[0, 2],
            c="green",
            s=300,
            marker="o",
            label="Source",
            edgecolors="black",
            linewidths=2,
        )
        ax.scatter(
            reduced[-1, 0],
            reduced[-1, 1],
            reduced[-1, 2],
            c="red",
            s=300,
            marker="s",
            label="Target",
            edgecolors="black",
            linewidths=2,
        )
        
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.set_zlabel(f"{method.upper()} 3")

    ax.legend(loc="upper right")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def _reduce_dimensions(
    points: np.ndarray,
    method: str,
    n_components: int,
) -> np.ndarray:
    """Apply dimensionality reduction.

    Args:
        points: High-dimensional points (n_samples, n_features).
        method: Reduction method ('pca', 'tsne', 'umap').
        n_components: Number of output dimensions.

    Returns:
        Reduced points (n_samples, n_components).
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(points)

        # For linear interpolation, data is collinear so PC2+ have ~zero variance
        # Replace noisy PC2+ with trajectory position to show clean line
        if reducer.explained_variance_ratio_[0] > 0.99:
            # Data is essentially 1D, use step index for second axis
            reduced[:, 1] = np.linspace(0, 1, len(points))

        return reduced
        
    elif method == "tsne":
        from sklearn.manifold import TSNE
        # t-SNE perplexity should be less than n_samples
        perplexity = min(30, len(points) - 1)
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
        )
        
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP not installed. Install with: pip install umap-learn"
            )
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
        )
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'.")

    return reducer.fit_transform(points)


def plot_latent_dimensions(
    latents: list[torch.Tensor],
    n_dims: int = 8,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot individual latent dimensions over the morphing trajectory.

    Args:
        latents: List of latent tensors.
        n_dims: Number of latent dimensions to plot.
        save_path: Path to save figure.
        show: Whether to display.

    Returns:
        Matplotlib Figure.
    """
    # Stack latents
    stacked = torch.stack(latents)
    
    # Average over time if needed
    if stacked.dim() == 4:
        stacked = stacked.mean(dim=-1).squeeze(1)
    elif stacked.dim() == 3:
        stacked = stacked.mean(dim=-1)
    
    points = stacked.detach().cpu().numpy()  # (steps, latent_dim)
    
    n_dims = min(n_dims, points.shape[1])
    
    fig, axes = plt.subplots(
        n_dims // 2,
        2,
        figsize=(14,8),
        sharex=True,
    )
    axes = axes.flatten()

    x = np.linspace(0, 1, len(points))

    for i in range(n_dims):
        ax = axes[i]
        ax.plot(x, points[:, i], linewidth=2)
        ax.fill_between(x, points[:, i], alpha=0.3)
        ax.set_ylabel(f"Dim {i}")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        
        if i >= n_dims - 2:
            ax.set_xlabel("Morphing progress")

    plt.suptitle("Latent dimension evolution during morphing", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig