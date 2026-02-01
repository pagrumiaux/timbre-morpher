"""Tests for audio utility functions."""

import pytest
import torch

from timbre_morpher.utils.audio import (
    match_length,
    crossfade_sequence,
    normalize_loudness,
    _adjust_length,
    _crossfade_pair,
)


class TestMatchLength:
    """Tests for the match_length function."""

    def test_already_matching(self):
        """No change when lengths already match."""
        a1 = torch.randn(1, 1000)
        a2 = torch.randn(1, 1000)

        r1, r2 = match_length(a1, a2)

        assert r1.shape == a1.shape
        assert r2.shape == a2.shape
        assert torch.equal(r1, a1)
        assert torch.equal(r2, a2)

    def test_min_mode_truncates(self):
        """Min mode should truncate to shorter length."""
        a1 = torch.randn(1, 1000)
        a2 = torch.randn(1, 500)

        r1, r2 = match_length(a1, a2, mode="min")

        assert r1.shape[-1] == 500
        assert r2.shape[-1] == 500
        assert torch.equal(r1, a1[..., :500])
        assert torch.equal(r2, a2)

    def test_max_mode_pads(self):
        """Max mode should pad to longer length."""
        a1 = torch.randn(1, 1000)
        a2 = torch.randn(1, 500)

        r1, r2 = match_length(a1, a2, mode="max")

        assert r1.shape[-1] == 1000
        assert r2.shape[-1] == 1000
        # Original content preserved
        assert torch.equal(r2[..., :500], a2)
        # Padded with zeros
        assert torch.equal(r2[..., 500:], torch.zeros(1, 500))

    def test_invalid_mode(self):
        """Invalid mode should raise error."""
        a1 = torch.randn(1, 1000)
        a2 = torch.randn(1, 500)

        with pytest.raises(ValueError):
            match_length(a1, a2, mode="invalid")


class TestCrossfade:
    """Tests for crossfade functions."""

    def test_crossfade_pair_length(self):
        """Crossfaded audio should have correct length."""
        a1 = torch.randn(1, 1000)
        a2 = torch.randn(1, 1000)
        crossfade_samples = 100

        result = _crossfade_pair(a1, a2, crossfade_samples)

        # Length should be sum minus overlap
        expected_len = 1000 + 1000 - 100
        assert result.shape[-1] == expected_len

    def test_crossfade_pair_no_crossfade(self):
        """Zero crossfade should just concatenate."""
        a1 = torch.randn(1, 1000)
        a2 = torch.randn(1, 1000)

        result = _crossfade_pair(a1, a2, 0)
        expected = torch.cat([a1, a2], dim=-1)

        assert torch.equal(result, expected)

    def test_crossfade_sequence_empty(self):
        """Empty list should raise error."""
        with pytest.raises(ValueError):
            crossfade_sequence([], 100)

    def test_crossfade_sequence_single(self):
        """Single audio should return unchanged."""
        a1 = torch.randn(1, 1000)

        result = crossfade_sequence([a1], 100)

        assert torch.equal(result, a1)

    def test_crossfade_sequence_multiple(self):
        """Multiple audios should be crossfaded correctly."""
        audios = [torch.randn(1, 1000) for _ in range(3)]
        crossfade_samples = 100

        result = crossfade_sequence(audios, crossfade_samples)

        # Each junction loses crossfade_samples
        expected_len = 3 * 1000 - 2 * 100
        assert result.shape[-1] == expected_len


class TestNormalizeLoudness:
    """Tests for loudness normalization."""

    def test_normalize_loudness_peak(self):
        """Normalized audio should have specified peak."""
        audio = torch.randn(1, 10000)
        target_db = -6.0

        result = normalize_loudness(audio, target_db)

        # Convert dB to linear
        target_linear = 10 ** (target_db / 20)
        actual_peak = result.abs().max().item()

        assert abs(actual_peak - target_linear) < 1e-6

    def test_normalize_silent_audio(self):
        """Silent audio should remain silent."""
        audio = torch.zeros(1, 1000)

        result = normalize_loudness(audio, -14.0)

        assert torch.equal(result, audio)


class TestAdjustLength:
    """Tests for length adjustment helper."""

    def test_truncate(self):
        """Longer audio should be truncated."""
        audio = torch.randn(1, 1000)
        result = _adjust_length(audio, 500)

        assert result.shape[-1] == 500
        assert torch.equal(result, audio[..., :500])

    def test_pad(self):
        """Shorter audio should be padded."""
        audio = torch.randn(1, 500)
        result = _adjust_length(audio, 1000)

        assert result.shape[-1] == 1000
        assert torch.equal(result[..., :500], audio)
        assert torch.equal(result[..., 500:], torch.zeros(1, 500))

    def test_no_change(self):
        """Matching length should be unchanged."""
        audio = torch.randn(1, 1000)
        result = _adjust_length(audio, 1000)

        assert torch.equal(result, audio)
