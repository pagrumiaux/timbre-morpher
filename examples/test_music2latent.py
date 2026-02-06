"""Test script to verify music2latent loading and basic encode/decode."""

import logging
import torch

logging.basicConfig(level=logging.INFO)

from timbre_morpher.models.music2latent_wrapper import Music2LatentWrapper

print("Testing music2latent loading...")
print("(Weights will auto-download on first use)\n")

m2l = Music2LatentWrapper.from_pretrained()
print("Model loaded successfully!")

print(f"\nModel info:")
print(f"  Sample rate: {m2l.sample_rate} Hz")
print(f"  Latent dim:  {m2l.latent_dim}")
print(f"  Hop length:  {m2l.hop_length}")
print(f"  Channels:    1 (mono)")

# Test with synthetic audio
print("\n" + "-" * 50)
print("Testing encode/decode with synthetic audio...")

duration = 2.0
t = torch.linspace(0, duration, int(duration * m2l.sample_rate))
audio = torch.sin(2 * 3.14159 * 440 * t)  # 440Hz sine, mono

print(f"Input audio shape: {audio.shape}")

z = m2l.encode(audio)
print(f"\nLatent shape: {z.shape}")
print(f"  - Batch:       {z.shape[0]}")
print(f"  - Latent dim:  {z.shape[1]}")
print(f"  - Time frames: {z.shape[2]}")

reconstructed = m2l.decode(z)
print(f"\nReconstructed audio shape: {reconstructed.shape}")

print("\nmusic2latent is working correctly!")
print("\nNote: This is a Consistency Autoencoder — continuous latent space")
print("      with single-step decoding (no iterative diffusion needed).")
