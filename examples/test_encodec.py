"""Test script to verify EnCodec model loading and basic encode/decode."""

import logging
import torch

logging.basicConfig(level=logging.INFO)

from timbre_morpher.models.encodec_wrapper import EncodecWrapper, ENCODEC_PRETRAINED_MODELS

print("Available EnCodec models:")
for model_name, config in ENCODEC_PRETRAINED_MODELS.items():
    print(f"  - {model_name}: {config['sample_rate']}Hz, {config['bandwidth']}kbps")


print("\nTesting EnCodec model loading...")

# Load the 48kHz model (default)
model_name = "encodec_48khz"
print(f"\nLoading '{model_name}' model...")

encodec = EncodecWrapper.from_pretrained(model_name)
print("Model loaded successfully!")

# Check model attributes
print("\nModel info:")
print(f"  Sample rate: {encodec.sample_rate} Hz")
print(f"  Latent dim: {encodec.latent_dim}")
print(f"  Bandwidth: {encodec.encodec_config.bandwidth} kbps")

# Test with synthetic audio (sine wave)
print("\n" + "-" * 50)
print("Testing encode/decode with synthetic audio...")

duration = 2.0  # seconds
sample_rate = encodec.sample_rate
t = torch.linspace(0, duration, int(duration * sample_rate))
audio = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz (A4 note)

print(f"Input audio shape: {audio.shape}")
print(f"Input duration: {duration}s at {sample_rate}Hz")

# Encode
z = encodec.encode(audio)
print(f"\nLatent shape: {z.shape}")
print(f"  - Batch: {z.shape[0]}")
print(f"  - Latent dim: {z.shape[1]}")
print(f"  - Time frames: {z.shape[2]}")

# Decode
reconstructed = encodec.decode(z)
print(f"\nReconstructed audio shape: {reconstructed.shape}")

print("\n" + "-" * 50)
print("Testing different bandwidths...")

# Test with different bandwidth settings
for bandwidth in [1.5, 3.0, 6.0, 12.0, 24.0]:
    print(f"\nTesting bandwidth: {bandwidth} kbps")
    encodec_bw = EncodecWrapper.from_pretrained(model_name, bandwidth=bandwidth)
    z = encodec_bw.encode(audio)
    recon = encodec_bw.decode(z)
    print(f"  Latent shape: {z.shape}, Audio shape: {recon.shape}")

print("\nEnCodec model is working correctly!")
