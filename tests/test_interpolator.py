"""Tests for the interpolation engine."""

import pytest
import torch

from timbre_morpher.core.interpolator import Interpolator, interpolate


class TestInterpolator:
    """Tests for the Interpolator class."""

    @pytest.fixture
    def sample_latents(self):
        """Create sample latent tensors for testing."""
        torch.manual_seed(42)
        z1 = torch.randn(1, 128, 10)  # (batch, latent_dim, time)
        z2 = torch.randn(1, 128, 10)
        return z1, z2

    def test_linear_interpolation_endpoints(self, sample_latents):
        """Linear interpolation at endpoints should return original tensors."""
        z1, z2 = sample_latents
        interpolator = Interpolator()

        result_0 = interpolator(z1, z2, alpha=0.0)
        result_1 = interpolator(z1, z2, alpha=1.0)

        assert torch.allclose(result_0, z1, atol=1e-6)
        assert torch.allclose(result_1, z2, atol=1e-6)

    def test_linear_interpolation_midpoint(self, sample_latents):
        """Linear interpolation at midpoint should be average."""
        z1, z2 = sample_latents
        interpolator = Interpolator()

        result = interpolator(z1, z2, alpha=0.5)
        expected = (z1 + z2) / 2

        assert torch.allclose(result, expected, atol=1e-6)

    def test_trajectory_length(self, sample_latents):
        """Trajectory should have correct number of steps."""
        z1, z2 = sample_latents
        interpolator = Interpolator()

        # With endpoints
        trajectory = interpolator.trajectory(z1, z2, steps=10)
        assert len(trajectory) == 12  # 10 + 2 endpoints


    def test_trajectory_shape_preserved(self, sample_latents):
        """Trajectory tensors should have same shape as input."""
        z1, z2 = sample_latents
        interpolator = Interpolator()

        trajectory = interpolator.trajectory(z1, z2, steps=5)

        for z in trajectory:
            assert z.shape == z1.shape

    def test_interpolate_convenience_function(self, sample_latents):
        """Test the standalone interpolate function."""
        z1, z2 = sample_latents

        result = interpolate(z1, z2, alpha=0.5)
        expected = (z1 + z2) / 2

        assert torch.allclose(result, expected, atol=1e-6)

    def test_linear_preserves_convex_combination(self):
        """Linear interpolation should stay within convex hull."""
        torch.manual_seed(42)
        z1 = torch.randn(1, 64, 5)
        z2 = torch.randn(1, 64, 5)

        interpolator = Interpolator()

        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = interpolator(z1, z2, alpha)

            # Check that result is between z1 and z2 componentwise
            min_vals = torch.minimum(z1, z2)
            max_vals = torch.maximum(z1, z2)

            assert (result >= min_vals - 1e-6).all()
            assert (result <= max_vals + 1e-6).all()
