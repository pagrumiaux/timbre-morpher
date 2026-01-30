"""Test script to verify RAVE model loading and basic encode/decode."""

import logging
import torch

logging.basicConfig(level=logging.INFO)

from timbre_morpher.models.rave_wrapper import RAVEWrapper, RAVE_PRETRAINED_MODELS

print("Available RAVE models:")
for model in RAVE_PRETRAINED_MODELS:
    print(f"  - {model}")


print("\nTesting RAVE model loading...")

# Load the 'percussion' model
model_name = "percussion"
print(f"\nLoading '{model_name}' model (this will download ~50-100MB on first run)...")

rave = RAVEWrapper.from_pretrained(model_name)
print(f"Model loaded successfully!")

# Check model attributes
print(f"\nModel info:")
print(f"  Sample rate: {rave.sample_rate} Hz")
print(f"  Latent dim: {rave.latent_dim}")

# Test with synthetic audio (sine wave)
print("\n" + "-" * 50)
print("Testing encode/decode with synthetic audio...")

duration = 2.0  # seconds
sample_rate = rave.sample_rate
t = torch.linspace(0, duration, int(duration * sample_rate))
audio = torch.sin(2 * 3.14159 * 440 * t)

print(f"Input audio shape: {audio.shape}")
print(f"Input duration: {duration}s at {sample_rate}Hz")

# Encode
z = rave.encode(audio)
print(f"\nLatent shape: {z.shape}")
print(f"  - Batch: {z.shape[0]}")
print(f"  - Latent dim: {z.shape[1]}")
print(f"  - Time frames: {z.shape[2]}")

# Decode
reconstructed = rave.decode(z)
print(f"\nReconstructed audio shape: {reconstructed.shape}")

print("\nRAVE model is working correctly!")
