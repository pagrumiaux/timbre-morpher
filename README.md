# Timbre Morpher

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Transform sounds smoothly from one timbre to another using latent space interpolation.

## Features

- Morph between two audio files (piano → violin, voice → synth, etc.)
- Model-agnostic architecture: designed to work with any audio autoencoder
- Latent space trajectory visualization (PCA, t-SNE, UMAP)

### Implemented models

| Model | Description |
|-------|-------------|
| [RAVE](https://github.com/acids-ircam/RAVE) | Realtime Audio Variational autoEncoder from IRCAM |
| [EnCodec](https://github.com/facebookresearch/encodec) | High Fidelity Neural Audio Codec from Meta |
| [DAC](https://github.com/descriptinc/descript-audio-codec) | High-Fidelity Audio Codec from Descript |

## Installation

```bash
git clone https://github.com/music-pal/timbre-morpher.git
cd timbre-morpher
pip install -e .
```

### Optional dependencies

To use EnCodec models:

```bash
pip install -e ".[encodec]"
# or
pip install encodec
```

To use DAC models:

```bash
pip install -e ".[dac]"
# or
pip install descript-audio-codec
```

To install all optional backends at once:

```bash
pip install -e ".[all]"
```

## Audio demo

Morphing from piano to violin using [`morph.py`](scripts/morph.py):

| Audio                            | File                                                                   |
|----------------------------------|------------------------------------------------------------------------|
| Source (Piano)                   | [piano.wav](examples/audio/piano.wav)                                  |
| Target (Violin)                  | [violin.wav](examples/audio/violin.wav)                                |
| Morphed (10 steps, concatenated, DAC model) | [piano_to_violin_morph.wav](examples/audio/piano_to_violin_morph.wav)  |

## Quick start

### 1. Test model loading

First, verify that models download and run correctly:

```bash
# Test RAVE
python examples/test_rave.py

# Test EnCodec
python examples/test_encodec.py

# Test DAC
python examples/test_dac.py
```

These scripts download models and test encode/decode with a synthetic sine wave.

### 2. Morph between audio files

Use the `morph.py` script to morph between any two audio files:

```bash
# Basic usage with RAVE
python scripts/morph.py source.wav target.wav

# Use EnCodec model
python scripts/morph.py source.wav target.wav --model encodec

# Use DAC model (44.1kHz, good for music)
python scripts/morph.py source.wav target.wav --model dac

# Custom steps and output directory
python scripts/morph.py source.wav target.wav --steps 20 --output ./my_morph/

# Save individual steps
python scripts/morph.py source.wav target.wav --save-individual

# With visualization
python scripts/morph.py source.wav target.wav --visualize
```

Results are saved to `output/` by default.

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

## Available models

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

### EnCodec

[EnCodec](https://github.com/facebookresearch/encodec) (Neural Audio Codec) is Meta's state-of-the-art neural audio compression model. It uses a streaming encoder-decoder architecture with residual vector quantization.

**Specs:** 48kHz sample rate, configurable bandwidth (1.5-24 kbps), 128-dim latent space

**Note:** EnCodec requires the `encodec` package. Install with: `pip install encodec`

```python
morpher = TimbreMorpher(model="encodec")
```

### DAC

[DAC](https://github.com/descriptinc/descript-audio-codec) (Descript Audio Codec) is a high-fidelity neural audio codec optimised for music at 44.1kHz. Like EnCodec it uses residual vector quantisation internally, but for morphing we bypass the quantiser and interpolate in the continuous pre-quantisation latent space.

| Checkpoint    | Sample Rate | Bitrate |
|---------------|-------------|---------|
| `dac_44khz`   | 44.1 kHz    | 8 kbps  |
| `dac_24khz`   | 24 kHz      | 8 kbps  |
| `dac_16khz`   | 16 kHz      | 8 kbps  |

**Note:** DAC requires the `descript-audio-codec` package. Install with: `pip install descript-audio-codec`

```python
morpher = TimbreMorpher(model="dac")                          # 44.1kHz (default)
morpher = TimbreMorpher(model="dac", checkpoint="dac_24khz")  # 24kHz
```

## Latent space visualization

```python
from timbre_morpher.utils.visualization import plot_trajectory, plot_latent_dimensions

# Get latent trajectory
trajectory = morpher.get_latent_trajectory(source, target, steps=50)

# 2D visualization
plot_trajectory(trajectory, method="pca", save_path="trajectory.png")

# Plot individual latent dimensions
plot_latent_dimensions(trajectory, n_dims=8, save_path="latent_dims.png")
```

## Custom models

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

## How it works

Both the source and target audio are encoded into a continuous latent space by the same model. This produces two points in that space — one representing the source timbre, one representing the target. We then walk a straight line between those two points, sampling intermediate latent vectors along the way. Each intermediate vector is decoded back to audio, producing one step of the morph. String all the steps together and you get a smooth timbre transition.

For codecs that use discrete quantisation internally (EnCodec, DAC), we bypass the quantiser and work directly with the continuous encoder output. This keeps the latent space smooth and interpolation-friendly, at the cost of some reconstruction quality compared to the quantised path — a tradeoff worth making for morphing.

**Note:** The quality of the output depends on the underlying model's ability to reconstruct your audio. Pretrained models work best with audio similar to their training data. For optimal results, pick a model trained on the same domain as your source and target.

## Roadmap

- [x] Core morphing engine
- [x] RAVE integration
- [x] Latent space visualization
- [x] EnCodec integration
- [x] DAC integration
- [ ] Other autoencoder models:
  - [AudioMAE](https://github.com/facebookresearch/AudioMAE)
- [ ] Enforce pitch preservation
- [ ] CLI interface
- [ ] VST/AU plugin
