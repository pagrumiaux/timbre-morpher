# Timbre Morpher

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

Transform sounds smoothly from one timbre to another using latent space interpolation.

## Features

- Morph between two audio files (piano → violin, voice → synth, etc.)
- Model-agnostic architecture: designed to work with any audio autoencoder
- Latent space trajectory visualization (PCA, t-SNE, UMAP)
- Simple Python API
- Audio utilities (resampling, crossfade, normalization)

### Implemented Models

| Model | Description |
|-------|-------------|
| [RAVE](https://github.com/acids-ircam/RAVE) | Realtime Audio Variational autoEncoder from IRCAM |

## Installation

```bash
git clone https://github.com/music-pal/timbre-morpher.git
cd timbre-morpher
pip install -e .
```

## Audio Demo

Morphing from piano to violin using [`basic_morph.py`](examples/basic_morph.py):

<!-- markdownlint-disable MD033 -->
<table>
  <tr>
    <th>Source (Piano)</th>
    <th>Target (Violin)</th>
  </tr>
  <tr>
    <td><audio controls src="examples/audio/piano.wav"></audio></td>
    <td><audio controls src="examples/audio/violin.wav"></audio></td>
  </tr>
</table>

<table>
  <tr>
    <th>Morphed (10 steps, concatenated with crossfade)</th>
  </tr>
  <tr>
    <td><audio controls src="examples/audio/piano_to_violin_morph.wav"></audio></td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

## Quick Start

### 1. Test RAVE model loading

First, verify that RAVE models download and run correctly:

```bash
python examples/test_rave.py
```

This downloads a RAVE model and tests encode/decode with a synthetic sine wave.

### 2. Run a basic morph

Morph between piano and violin samples:

```bash
python examples/basic_morph.py
```

Results are saved to `output/basic_morph/`.

### Python API

```python
from timbre_morpher import TimbreMorpher
from timbre_morpher.utils.audio import load_audio, resample

# Initialize the morpher (model downloads on first use)
morpher = TimbreMorpher(model="rave", checkpoint="musicnet")

# Load audio files
source, sr = load_audio("piano.wav")
target, sr = load_audio("violin.wav")

# Resample to model's sample rate (48kHz)
source = resample(source, sr, morpher.sample_rate)
target = resample(target, sr, morpher.sample_rate)

# Generate morph sequence
result = morpher.morph(source, target, steps=10)

# Save individual steps
morpher.save_sequence(result, "output/")

# Or save as single concatenated file with crossfade
morpher.save_concatenated(result, "morph_full.wav", crossfade_ms=50)
```

## Available Models

### RAVE

[RAVE](https://github.com/acids-ircam/RAVE) (Realtime Audio Variational autoEncoder) is a fast neural audio synthesis model from IRCAM. Pretrained checkpoints are downloaded automatically on first use.

| Checkpoint       | Training Data          | Best For                                     |
|------------------|------------------------|----------------------------------------------|
| `musicnet`       | MusicNet dataset       | Classical instruments (piano, violin, cello) |
| `sol_ordinario`  | IRCAM Studio OnLine    | Orchestral (standard techniques)             |
| `sol_full`       | IRCAM Studio OnLine    | Full orchestral range                        |
| `vintage`        | 80h vintage recordings | General music, diverse sources               |
| `VCTK`           | VCTK speech corpus     | Speech and voice                             |
| `percussion`     | Percussion samples     | Drums and percussion                         |

**Specs:** 48kHz sample rate, 128-dim latent space, ~23ms latency

```python
morpher = TimbreMorpher(model="rave", checkpoint="vintage")
```

## Latent Space Visualization

```python
from timbre_morpher.utils.visualization import plot_trajectory, plot_latent_dimensions

# Get latent trajectory
trajectory = morpher.get_latent_trajectory(source, target, steps=50)

# 2D visualization
plot_trajectory(trajectory, method="pca", save_path="trajectory.png")

# Plot individual latent dimensions
plot_latent_dimensions(trajectory, n_dims=8, save_path="latent_dims.png")
```

## Custom Models

Implement your own encoder/decoder backend:

```python
from timbre_morpher.models.base import AudioVAE
import torch

class MyCustomVAE(AudioVAE):
    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: (batch, channels, samples)
        # return: (batch, latent_dim, time_frames)
        ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: (batch, latent_dim, time_frames)
        # return: (batch, channels, samples)
        ...

morpher = TimbreMorpher(model=MyCustomVAE())
```

## How It Works

Timbre Morpher encodes audio into a continuous latent space using a variational autoencoder. By linearly interpolating between latent representations, we generate smooth transitions between different timbres.

```
┌──────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌──────────┐
│  Source  │────▶│ Encoder │────▶│  Latent │────▶│ Decoder │────▶│ Morphed  │
│  Audio   │     │         │     │  Interp │     │         │     │  Audio   │
└──────────┘     └─────────┘     └─────────┘     └─────────┘     └──────────┘
                                      ▲
┌──────────┐     ┌─────────┐          │
│  Target  │────▶│ Encoder │──────────┘
│  Audio   │     │         │
└──────────┘     └─────────┘
```

## Roadmap

- [x] Core morphing engine
- [x] RAVE integration
- [x] Latent space visualization
- [ ] Additional interpolation methods (SLERP, Bezier)
- [ ] Pitch preservation
- [ ] Other autoencoder models:
  - [EnCodec](https://github.com/facebookresearch/encodec) (Meta)
  - [DAC](https://github.com/descriptinc/descript-audio-codec) (Descript Audio Codec)
  - [AudioMAE](https://github.com/facebookresearch/AudioMAE)
- [ ] CLI interface
- [ ] VST/AU plugin
