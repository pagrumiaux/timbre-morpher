"""Tests for music2latent wrapper (MockMusic2Latent)."""

import pytest
import torch

from timbre_morpher.models.music2latent_wrapper import (
    Music2LatentConfig,
    MockMusic2Latent,
)


@pytest.fixture
def config():
    return Music2LatentConfig()


@pytest.fixture
def mock_m2l(config):
    torch.manual_seed(42)
    return MockMusic2Latent(config)


class TestMockMusic2LatentShapes:
    """Encode/decode shape contracts."""

    def test_encode_3d_input(self, mock_m2l, config):
        """Standard (batch, channels, samples) input."""
        audio = torch.randn(2, 1, config.hop_length * 4)
        z = mock_m2l.encode(audio)

        assert z.shape == (2, config.latent_dim, 4)

    def test_encode_2d_input(self, mock_m2l, config):
        """(batch, samples) input gets channel dim added."""
        audio = torch.randn(2, config.hop_length * 4)
        z = mock_m2l.encode(audio)

        assert z.shape == (2, config.latent_dim, 4)

    def test_encode_1d_input(self, mock_m2l, config):
        """(samples,) input gets both batch and channel dims added."""
        audio = torch.randn(config.hop_length * 4)
        z = mock_m2l.encode(audio)

        assert z.shape == (1, config.latent_dim, 4)

    def test_decode_shape(self, mock_m2l, config):
        """Decode produces mono audio."""
        z = torch.randn(2, config.latent_dim, 4)
        audio = mock_m2l.decode(z)

        assert audio.shape == (2, 1, config.hop_length * 4)

    def test_roundtrip_shape(self, mock_m2l, config):
        """Encode -> decode preserves batch size and length."""
        audio = torch.randn(3, 1, config.hop_length * 8)
        recon = mock_m2l(audio)

        assert recon.shape == audio.shape


class TestMockMusic2LatentMono:
    """Stereo-to-mono conversion in preprocessing."""

    def test_stereo_input_is_averaged(self, mock_m2l, config):
        """Stereo input should be averaged to mono before encoding."""
        mono = torch.ones(1, 1, config.hop_length * 2)
        stereo = torch.ones(1, 2, config.hop_length * 2)

        z_mono = mock_m2l.encode(mono)
        z_stereo = mock_m2l.encode(stereo)

        assert torch.allclose(z_mono, z_stereo)

    def test_multichannel_averaged(self, mock_m2l, config):
        """Multi-channel input should be averaged to mono."""
        # 4 channels all ones = mono ones after averaging
        multi = torch.ones(1, 4, config.hop_length * 2)
        mono = torch.ones(1, 1, config.hop_length * 2)

        z_multi = mock_m2l.encode(multi)
        z_mono = mock_m2l.encode(mono)

        assert torch.allclose(z_multi, z_mono)


class TestMockMusic2LatentConfig:
    """Config defaults."""

    def test_default_config(self, config):
        assert config.sample_rate == 44100
        assert config.latent_dim == 64
        assert config.hop_length == 4410

    def test_custom_config(self):
        cfg = Music2LatentConfig(latent_dim=128, hop_length=2205)
        model = MockMusic2Latent(cfg)

        audio = torch.randn(1, 1, 2205 * 5)
        z = model.encode(audio)
        assert z.shape == (1, 128, 5)


class TestMockMusic2LatentFromBase:
    """Inherited AudioVAE interface."""

    def test_sample_rate_property(self, mock_m2l, config):
        assert mock_m2l.sample_rate == config.sample_rate

    def test_latent_dim_property(self, mock_m2l, config):
        assert mock_m2l.latent_dim == config.latent_dim

    def test_hop_length_property(self, mock_m2l, config):
        assert mock_m2l.hop_length == config.hop_length

    def test_get_latent_size(self, mock_m2l, config):
        assert mock_m2l.get_latent_size(config.hop_length * 10) == 10

    def test_get_audio_size(self, mock_m2l, config):
        assert mock_m2l.get_audio_size(10) == config.hop_length * 10
