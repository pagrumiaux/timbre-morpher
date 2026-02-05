"""Tests for Stable Audio wrapper (MockStableAudio)."""

import pytest
import torch

from timbre_morpher.models.stable_audio_wrapper import (
    StableAudioConfig,
    MockStableAudio,
    STABLE_AUDIO_PRETRAINED_MODELS,
)


@pytest.fixture
def config():
    return StableAudioConfig()


@pytest.fixture
def mock_stable_audio(config):
    torch.manual_seed(42)
    return MockStableAudio(config)


class TestMockStableAudioShapes:
    """Encode/decode shape contracts."""

    def test_encode_3d_input(self, mock_stable_audio, config):
        """Standard (batch, channels, samples) input."""
        audio = torch.randn(2, 2, config.hop_length * 4)
        z = mock_stable_audio.encode(audio)

        assert z.shape == (2, config.latent_dim, 4)

    def test_encode_2d_input(self, mock_stable_audio, config):
        """(batch, samples) input gets channel dim added and duplicated to stereo."""
        audio = torch.randn(2, config.hop_length * 4)
        z = mock_stable_audio.encode(audio)

        assert z.shape == (2, config.latent_dim, 4)

    def test_encode_1d_input(self, mock_stable_audio, config):
        """(samples,) input gets both batch and channel dims added."""
        audio = torch.randn(config.hop_length * 4)
        z = mock_stable_audio.encode(audio)

        assert z.shape == (1, config.latent_dim, 4)

    def test_decode_shape(self, mock_stable_audio, config):
        """Decode produces stereo audio."""
        z = torch.randn(2, config.latent_dim, 4)
        audio = mock_stable_audio.decode(z)

        assert audio.shape == (2, 2, config.hop_length * 4)

    def test_roundtrip_shape(self, mock_stable_audio, config):
        """Encode → decode preserves batch size and length."""
        audio = torch.randn(3, 2, config.hop_length * 8)
        recon = mock_stable_audio(audio)

        assert recon.shape == audio.shape


class TestMockStableAudioStereo:
    """Mono-to-stereo conversion in preprocessing."""

    def test_mono_input_is_duplicated(self, mock_stable_audio, config):
        """Mono input should be duplicated to stereo before encoding."""
        mono = torch.ones(1, 1, config.hop_length * 2)
        stereo = torch.ones(1, 2, config.hop_length * 2)

        z_mono = mock_stable_audio.encode(mono)
        z_stereo = mock_stable_audio.encode(stereo)

        assert torch.allclose(z_mono, z_stereo)

    def test_multichannel_input_is_truncated(self, mock_stable_audio, config):
        """More than 2 channels should be truncated to first 2."""
        # Create 4-channel audio where channels 0,1 are ones, channels 2,3 are twos
        multi = torch.ones(1, 4, config.hop_length * 2)
        multi[:, 2:, :] = 2.0

        stereo = multi[:, :2, :]  # Expected: first 2 channels

        z_multi = mock_stable_audio.encode(multi)
        z_stereo = mock_stable_audio.encode(stereo)

        assert torch.allclose(z_multi, z_stereo)


class TestMockStableAudioConfig:
    """Config and pretrained model registry."""

    def test_default_config_is_stable_audio_open(self, config):
        assert config.sample_rate == 44100
        assert config.checkpoint == "stable-audio-open-1.0"
        assert config.latent_dim == 64
        assert config.hop_length == 2048

    def test_pretrained_models_have_required_keys(self):
        required = {"model_id", "subfolder"}
        for name, spec in STABLE_AUDIO_PRETRAINED_MODELS.items():
            assert set(spec.keys()) == required, f"{name} missing keys"

    def test_custom_config(self):
        cfg = StableAudioConfig(latent_dim=128, hop_length=1024)
        model = MockStableAudio(cfg)

        audio = torch.randn(1, 2, 1024 * 5)
        z = model.encode(audio)
        assert z.shape == (1, 128, 5)


class TestMockStableAudioFromBase:
    """Inherited AudioVAE interface."""

    def test_sample_rate_property(self, mock_stable_audio, config):
        assert mock_stable_audio.sample_rate == config.sample_rate

    def test_latent_dim_property(self, mock_stable_audio, config):
        assert mock_stable_audio.latent_dim == config.latent_dim

    def test_hop_length_property(self, mock_stable_audio, config):
        assert mock_stable_audio.hop_length == config.hop_length

    def test_get_latent_size(self, mock_stable_audio, config):
        assert mock_stable_audio.get_latent_size(config.hop_length * 10) == 10

    def test_get_audio_size(self, mock_stable_audio, config):
        assert mock_stable_audio.get_audio_size(10) == config.hop_length * 10
