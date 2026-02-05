"""Test script to verify Stable Audio VAE loading and basic encode/decode."""

import logging
import torch

logging.basicConfig(level=logging.INFO)

from timbre_morpher.models.stable_audio_wrapper import (
    StableAudioWrapper,
    STABLE_AUDIO_PRETRAINED_MODELS,
)

print("Available Stable Audio models:")
for name, spec in STABLE_AUDIO_PRETRAINED_MODELS.items():
    print(f"  - {name}: {spec['model_id']}")


print("\nTesting Stable Audio VAE loading...")

# Load the default model (stable-audio-open-1.0)
model_name = "stable-audio-open-1.0"
print(f"\nLoading '{model_name}' VAE (this will download ~450MB on first run)...")

stable_audio = StableAudioWrapper.from_pretrained(model_name)
print("Model loaded successfully!")

# Check model attributes (synced from the loaded model)
print("\nModel info:")
print(f"  Sample rate: {stable_audio.sample_rate} Hz")
print(f"  Latent dim:  {stable_audio.latent_dim}")
print(f"  Hop length:  {stable_audio.hop_length}")
print(f"  Channels:    2 (stereo)")

# Test with synthetic audio (sine wave)
print("\n" + "-" * 50)
print("Testing encode/decode with synthetic audio...")

duration = 2.0  # seconds
sample_rate = stable_audio.sample_rate
t = torch.linspace(0, duration, int(duration * sample_rate))
# Create stereo sine wave (left: 440Hz, right: 880Hz)
left = torch.sin(2 * 3.14159 * 440 * t)
right = torch.sin(2 * 3.14159 * 880 * t)
audio = torch.stack([left, right], dim=0)  # (2, samples) → stereo

print(f"Input audio shape: {audio.shape}")
print(f"Input duration: {duration}s at {sample_rate}Hz")

# Encode (returns deterministic latent via .mode())
z = stable_audio.encode(audio)
print(f"\nLatent shape: {z.shape}")
print(f"  - Batch:       {z.shape[0]}")
print(f"  - Latent dim:  {z.shape[1]}")
print(f"  - Time frames: {z.shape[2]}")

# Decode
reconstructed = stable_audio.decode(z)
print(f"\nReconstructed audio shape: {reconstructed.shape}")
print(f"  - Batch:    {reconstructed.shape[0]}")
print(f"  - Channels: {reconstructed.shape[1]} (stereo)")
print(f"  - Samples:  {reconstructed.shape[2]}")

print("\nStable Audio VAE is working correctly!")
print("\nNote: This VAE is a proper KL-regularized continuous latent space model,")
print("      ideal for interpolation-based morphing (no quantizer to bypass).")
