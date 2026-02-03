"""Tests for DAC wrapper (MockDAC)."""

import pytest
import torch

from timbre_morpher.models.dac_wrapper import DACConfig, MockDAC, DAC_PRETRAINED_MODELS


@pytest.fixture
def config():
    return DACConfig()


@pytest.fixture
def mock_dac(config):
    torch.manual_seed(42)
    return MockDAC(config)


class TestMockDACShapes:
    """Encode/decode shape contracts."""

    def test_encode_3d_input(self, mock_dac, config):
        """Standard (batch, channels, samples) input."""
        audio = torch.randn(2, 1, config.hop_length * 4)
        z = mock_dac.encode(audio)

        assert z.shape == (2, config.latent_dim, 4)

    def test_encode_2d_input(self, mock_dac, config):
        """(batch, samples) input gets channel dim added."""
        audio = torch.randn(2, config.hop_length * 4)
        z = mock_dac.encode(audio)

        assert z.shape == (2, config.latent_dim, 4)

    def test_encode_1d_input(self, mock_dac, config):
        """(samples,) input gets both batch and channel dims added."""
        audio = torch.randn(config.hop_length * 4)
        z = mock_dac.encode(audio)

        assert z.shape == (1, config.latent_dim, 4)

    def test_decode_shape(self, mock_dac, config):
        """Decode produces mono audio."""
        z = torch.randn(2, config.latent_dim, 4)
        audio = mock_dac.decode(z)

        assert audio.shape == (2, 1, config.hop_length * 4)

    def test_roundtrip_shape(self, mock_dac, config):
        """Encode → decode preserves batch size and length."""
        audio = torch.randn(3, 1, config.hop_length * 8)
        recon = mock_dac(audio)

        assert recon.shape == audio.shape


class TestMockDACMono:
    """Stereo-to-mono conversion in preprocessing."""

    def test_stereo_input_is_downmixed(self, mock_dac, config):
        """Stereo input should be averaged to mono before encoding."""
        left = torch.ones(1, 1, config.hop_length * 2)
        right = torch.ones(1, 1, config.hop_length * 2) * 3.0
        stereo = torch.cat([left, right], dim=1)  # (1, 2, samples)

        mono = torch.ones(1, 1, config.hop_length * 2) * 2.0  # expected mean

        z_stereo = mock_dac.encode(stereo)
        z_mono = mock_dac.encode(mono)

        assert torch.allclose(z_stereo, z_mono)

    def test_multichannel_input_is_downmixed(self, mock_dac, config):
        """More than 2 channels should also be averaged to mono."""
        audio = torch.randn(1, 4, config.hop_length * 2)
        z = mock_dac.encode(audio)

        expected_mono = audio.mean(dim=1, keepdim=True)
        z_expected = mock_dac.encode(expected_mono)

        assert torch.allclose(z, z_expected)


class TestMockDACConfig:
    """Config and pretrained model registry."""

    def test_default_config_is_44khz(self, config):
        assert config.sample_rate == 44100
        assert config.checkpoint == "dac_44khz"

    def test_pretrained_models_have_required_keys(self):
        required = {"model_type", "model_bitrate"}
        for name, spec in DAC_PRETRAINED_MODELS.items():
            assert set(spec.keys()) == required, f"{name} missing keys"

    def test_custom_config(self):
        cfg = DACConfig(checkpoint="dac_24khz", sample_rate=24000, hop_length=320)
        model = MockDAC(cfg)

        audio = torch.randn(1, 1, 320 * 5)
        z = model.encode(audio)
        assert z.shape == (1, 1024, 5)


class TestMockDACFromBase:
    """Inherited AudioVAE interface."""

    def test_sample_rate_property(self, mock_dac, config):
        assert mock_dac.sample_rate == config.sample_rate

    def test_latent_dim_property(self, mock_dac, config):
        assert mock_dac.latent_dim == config.latent_dim

    def test_hop_length_property(self, mock_dac, config):
        assert mock_dac.hop_length == config.hop_length

    def test_get_latent_size(self, mock_dac, config):
        assert mock_dac.get_latent_size(config.hop_length * 10) == 10

    def test_get_audio_size(self, mock_dac, config):
        assert mock_dac.get_audio_size(10) == config.hop_length * 10
