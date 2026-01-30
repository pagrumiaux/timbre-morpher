#!/usr/bin/env python
"""
Basic Timbre Morphing Example

This example demonstrates morphing between piano and violin audio files.
"""

from pathlib import Path


def main():
    from timbre_morpher import TimbreMorpher
    from timbre_morpher.utils.audio import load_audio, resample

    print("-" * 60)
    print("Timbre Morpher - Piano to Violin Morph")

    # Paths
    assets_dir = Path(__file__).parent.parent / "assets" / "audio"
    output_dir = Path("output/basic_morph")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize morpher with RAVE model (musicnet is best for classical instruments)
    print("\n1. Initializing TimbreMorpher with RAVE model (musicnet)...")
    morpher = TimbreMorpher(model="rave", checkpoint="sol_full")
    print(f"   Sample rate: {morpher.sample_rate} Hz")

    # Load audio files
    print("\n2. Loading audio files...")
    source_audio, source_sr = load_audio(assets_dir / "piano.wav")
    target_audio, target_sr = load_audio(assets_dir / "violin.wav")

    # Resample to model's sample rate
    source_audio = resample(source_audio, source_sr, morpher.sample_rate)
    target_audio = resample(target_audio, target_sr, morpher.sample_rate)

    print(f"   Source (piano): {source_audio.shape}")
    print(f"   Target (violin): {target_audio.shape}")

    # Perform morphing
    print("\n3. Performing morphing...")
    result = morpher.morph(
        source=source_audio,
        target=target_audio,
        steps=10,
    )

    print(f"   Generated {len(result)} audio frames")
    print(f"   Latent shapes: {[z.shape for z in result.latents[:2]]}...")

    # Save results
    _ = morpher.save_sequence(result, output_dir)
    print(f"   Saved to: {output_dir}")

    # Save concatenated version
    morpher.save_concatenated(result, output_dir / "morph_full.wav", crossfade_ms=50)
    print(f"   Saved concatenated: {output_dir / 'morph_full.wav'}")

    # Trajectory visualization
    print("\n4. Generating latent trajectory visualization...")
    try:
        from timbre_morpher.utils.visualization import plot_trajectory

        trajectory = morpher.get_latent_trajectory(
            source_audio, target_audio, steps=50
        )

        _ = plot_trajectory(
            trajectory,
            method="pca",
            save_path=output_dir / "trajectory.png",
            show=False,
            title="Piano → Violin Morphing Trajectory (PCA)",
        )
        print(f"   Saved visualization to: {output_dir / 'trajectory.png'}")

    except ImportError as e:
        print(f"   Skipping visualization (missing dependency: {e})")

    print("\n" + "=" * 60)
    print("Done! Check the output directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
