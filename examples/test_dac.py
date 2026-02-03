"""Test script to verify DAC model loading and basic encode/decode."""

import logging
import torch

logging.basicConfig(level=logging.INFO)

from timbre_morpher.models.dac_wrapper import DACWrapper, DAC_PRETRAINED_MODELS

print("Available DAC models:")
for name, config in DAC_PRETRAINED_MODELS.items():
    print(f"  - {name}: {config['model_type']} @ {config['model_bitrate']}")


print("\nTesting DAC model loading...")

# Load the 44kHz model (default, best for music)
model_name = "dac_44khz"
print(f"\nLoading '{model_name}' model (this will download on first run)...")

dac = DACWrapper.from_pretrained(model_name)
print("Model loaded successfully!")

# Check model attributes (synced from the loaded model)
print("\nModel info:")
print(f"  Sample rate: {dac.sample_rate} Hz")
print(f"  Latent dim:  {dac.latent_dim}")
print(f"  Hop length:  {dac.hop_length}")

# Test with synthetic audio (sine wave)
print("\n" + "-" * 50)
print("Testing encode/decode with synthetic audio...")

duration = 2.0  # seconds
sample_rate = dac.sample_rate
t = torch.linspace(0, duration, int(duration * sample_rate))
audio = torch.sin(2 * 3.14159 * 440 * t)  # 440 Hz (A4 note)

print(f"Input audio shape: {audio.shape}")
print(f"Input duration: {duration}s at {sample_rate}Hz")

# Encode (bypasses RVQ, returns continuous pre-quantization latent)
z = dac.encode(audio)
print(f"\nLatent shape: {z.shape}")
print(f"  - Batch:       {z.shape[0]}")
print(f"  - Latent dim:  {z.shape[1]}")
print(f"  - Time frames: {z.shape[2]}")

# Decode
reconstructed = dac.decode(z)
print(f"\nReconstructed audio shape: {reconstructed.shape}")

# Test the other pretrained checkpoints
print("\n" + "-" * 50)
print("Testing other checkpoints...")

for checkpoint in ["dac_24khz", "dac_16khz"]:
    print(f"\nLoading '{checkpoint}'...")
    model = DACWrapper.from_pretrained(checkpoint)
    print(f"  Sample rate: {model.sample_rate} Hz, "
          f"latent_dim: {model.latent_dim}, "
          f"hop_length: {model.hop_length}")

    # Sine wave at the model's own sample rate
    t = torch.linspace(0, duration, int(duration * model.sample_rate))
    audio = torch.sin(2 * 3.14159 * 440 * t)

    z = model.encode(audio)
    recon = model.decode(z)
    print(f"  Latent shape: {z.shape}, Audio shape: {recon.shape}")

print("\nDAC model is working correctly!")
