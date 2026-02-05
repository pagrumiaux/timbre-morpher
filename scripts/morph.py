#!/usr/bin/env python3
"""
Command-line tool for morphing between two audio files using neural audio codecs.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from timbre_morpher import TimbreMorpher
from timbre_morpher.utils.audio import load_audio, resample


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Morph between two audio files using neural audio codecs',
    )

    # Required arguments
    parser.add_argument(
        'source',
        type=Path,
        help='Path to source audio file'
    )
    parser.add_argument(
        'target',
        type=Path,
        help='Path to target audio file'
    )

    # Model configuration
    model_group = parser.add_argument_group('model configuration')
    model_group.add_argument(
        '--model', '-m',
        type=str,
        choices=['rave', 'encodec', 'dac', 'stable-audio'],
        default='rave',
        help='Model to use for morphing (default: rave)'
    )
    model_group.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Model checkpoint name. For RAVE: vintage, musicnet, VCTK, etc. '
             'For EnCodec: encodec_48khz. For DAC: dac_16khz, dac_24khz, dac_44khz. '
             'For Stable Audio: stable-audio-open-1.0 '
             '(default for each model if not specified)'
    )
    model_group.add_argument(
        '--bandwidth',
        type=float,
        default=None,
        help='Bandwidth for EnCodec model in kbps (default: 12.0). '
             'Valid range: 1.5-24.0'
    )

    # Morphing parameters
    morph_group = parser.add_argument_group('morphing parameters')
    morph_group.add_argument(
        '--steps', '-s',
        type=int,
        default=10,
        help='Number of interpolation steps (default: 10)'
    )
    morph_group.add_argument(
        '--crossfade',
        type=float,
        default=50.0,
        help='Crossfade duration in milliseconds for concatenated output (default: 50)'
    )

    # Output configuration
    output_group = parser.add_argument_group('output configuration')
    output_group.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('output/morph'),
        help='Output directory (default: output/morph/)'
    )
    output_group.add_argument(
        '--save-individual',
        action='store_true',
        help='Save individual morph steps as separate files'
    )
    output_group.add_argument(
        '--no-concatenate',
        action='store_true',
        help='Do not save concatenated output'
    )
    output_group.add_argument(
        '--prefix',
        type=str,
        default='morph',
        help='Prefix for output files (default: morph)'
    )

    # Device configuration
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help='Device to use for computation (default: auto-detect)'
    )

    # Visualization
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate latent space trajectory visualization'
    )

    # Misc
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.source.exists():
        parser.error(f"Source file not found: {args.source}")
    if not args.target.exists():
        parser.error(f"Target file not found: {args.target}")
    if args.steps < 2:
        parser.error("Number of steps must be at least 2")
    if args.bandwidth is not None and (args.bandwidth < 1.5 or args.bandwidth > 24.0):
        parser.error("Bandwidth must be between 1.5 and 24.0 kbps")
    if args.bandwidth is not None and args.model != 'encodec':
        parser.error("--bandwidth can only be used with --model encodec")

    return args


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Print configuration
        logger.info(f"Source: {args.source}")
        logger.info(f"Target: {args.target}")
        logger.info(f"Model: {args.model}")
        if args.checkpoint:
            logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Steps: {args.steps}")
        logger.info(f"Output directory: {args.output}")

        # Initialize morpher
        logger.info("\n" + "="*50)
        logger.info("Initializing model...")
        logger.info("="*50)

        # Prepare model kwargs
        model_kwargs = {
            'model': args.model,
            'device': args.device,
        }

        # Set checkpoint based on model
        if args.checkpoint:
            model_kwargs['checkpoint'] = args.checkpoint
        elif args.model == 'rave':
            model_kwargs['checkpoint'] = 'vintage'
        elif args.model == 'encodec':
            model_kwargs['checkpoint'] = 'encodec_48khz'
        elif args.model == 'dac':
            model_kwargs['checkpoint'] = 'dac_44khz'
        elif args.model == 'stable-audio':
            model_kwargs['checkpoint'] = 'stable-audio-open-1.0'

        morpher = TimbreMorpher(**model_kwargs)

        # Update bandwidth for EnCodec if specified
        if args.model == 'encodec' and args.bandwidth is not None:
            logger.info(f"Setting EnCodec bandwidth to {args.bandwidth} kbps")
            # Reload with custom bandwidth
            morpher = TimbreMorpher(
                model=args.model,
                checkpoint=model_kwargs['checkpoint'],
                device=args.device,
            )
            # Set bandwidth on the model
            morpher._model.encodec_config.bandwidth = args.bandwidth
            morpher._model._model.set_target_bandwidth(args.bandwidth)

        logger.info(f"Model loaded: {morpher._model.__class__.__name__}")
        logger.info(f"Sample rate: {morpher.sample_rate} Hz")

        # Load audio files
        logger.info("\n" + "="*50)
        logger.info("Loading audio files...")
        logger.info("="*50)

        source_audio, source_sr = load_audio(str(args.source))
        target_audio, target_sr = load_audio(str(args.target))

        logger.info(f"Source: {source_audio.shape} at {source_sr} Hz")
        logger.info(f"Target: {target_audio.shape} at {target_sr} Hz")

        # Resample if needed
        if source_sr != morpher.sample_rate:
            logger.info(f"Resampling source from {source_sr} Hz to {morpher.sample_rate} Hz")
            source_audio = resample(source_audio, source_sr, morpher.sample_rate)

        if target_sr != morpher.sample_rate:
            logger.info(f"Resampling target from {target_sr} Hz to {morpher.sample_rate} Hz")
            target_audio = resample(target_audio, target_sr, morpher.sample_rate)

        # Audio is already torch tensors from load_audio, ensure float
        source_audio = source_audio.float()
        target_audio = target_audio.float()

        # Perform morphing
        logger.info("\n" + "="*50)
        logger.info(f"Morphing with {args.steps} steps...")
        logger.info("="*50)

        result = morpher.morph(source_audio, target_audio, steps=args.steps)

        logger.info(f"Generated {len(result)} audio frames")

        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)

        # Save results
        logger.info("\n" + "="*50)
        logger.info("Saving results...")
        logger.info("="*50)

        output_files = []

        # Save individual files
        if args.save_individual:
            logger.info("Saving individual morph steps...")
            paths = morpher.save_sequence(result, args.output, prefix=args.prefix)
            output_files.extend(paths)
            logger.info(f"Saved {len(paths)} individual files")

        # Save concatenated file
        if not args.no_concatenate:
            concat_path = args.output / f"{args.prefix}_concatenated.wav"
            logger.info(f"Saving concatenated output (crossfade: {args.crossfade}ms)...")
            morpher.save_concatenated(result, concat_path, crossfade_ms=args.crossfade)
            output_files.append(concat_path)
            logger.info(f"Saved: {concat_path}")

        # Generate visualization
        if args.visualize:
            logger.info("\n" + "="*50)
            logger.info("Generating visualization...")
            logger.info("="*50)

            from timbre_morpher.utils.visualization import plot_trajectory

            # Get latent trajectory for visualization (more steps for smoothness)
            viz_steps = max(args.steps, 50)
            trajectory = morpher.get_latent_trajectory(
                source_audio,
                target_audio,
                steps=viz_steps
            )

            # Convert trajectory list to tensor
            trajectory_tensor = torch.stack(trajectory)

            # Plot using PCA
            viz_path = args.output / f"{args.prefix}_trajectory_pca.png"
            plot_trajectory(trajectory_tensor, method="pca", save_path=str(viz_path))
            logger.info(f"Saved PCA visualization: {viz_path}")

            # Plot using t-SNE
            viz_path = args.output / f"{args.prefix}_trajectory_tsne.png"
            plot_trajectory(trajectory_tensor, method="tsne", save_path=str(viz_path))
            logger.info(f"Saved t-SNE visualization: {viz_path}")

        # Summary
        logger.info("\n" + "="*50)
        logger.info("MORPHING COMPLETE!")
        logger.info("="*50)
        logger.info(f"Output directory: {args.output}")
        logger.info(f"Total files generated: {len(output_files)}")

        return 0

    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
