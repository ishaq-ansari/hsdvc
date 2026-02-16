#!/usr/bin/env python
"""
Motion Control Video Generation - Main CLI Script

Generate videos with character performing motions from reference video.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.video_generator import Wan21VideoGenerator
from pipeline.preprocessing import validate_inputs
from pipeline.postprocessing import postprocess_video
from pipeline.identity_preserving import preserve_identity
from utils.audio_handler import AudioHandler
from utils.video_utils import save_frames_as_video


def main():
    parser = argparse.ArgumentParser(
        description="Generate motion control videos using Wan2.1 I2V",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output
    parser.add_argument(
        "--character",
        type=str,
        help="Path to character image"
    )
    parser.add_argument(
        "--reference",
        type=str,
        help="Path to reference video"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path for output video"
    )

    # Batch processing
    parser.add_argument(
        "--batch",
        type=str,
        help="Path to batch manifest JSON file"
    )
    parser.add_argument(
        "--job-index",
        type=int,
        help="Job index for batch processing (used with SLURM arrays)"
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/checkpoints",
        help="Path to Wan2.1 model checkpoints"
    )

    # Generation parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=3.375,  # 81 frames at 24fps (default for Wan2.1)
        help="Video duration in seconds"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Frames per second"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=5.0,  # Default for i2v
        help="Guidance scale for diffusion"
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=40,  # Default for i2v
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to guide video generation (e.g., 'person walking, smooth motion')"
    )

    # Processing options
    parser.add_argument(
        "--preserve-identity",
        action="store_true",
        help="Apply identity preservation using CodeFormer"
    )
    parser.add_argument(
        "--preserve-audio",
        action="store_true",
        help="Extract and merge audio from reference video"
    )
    parser.add_argument(
        "--target-resolution",
        type=str,
        default="1920x1080",
        help="Target resolution (WIDTHxHEIGHT)"
    )
    parser.add_argument(
        "--upscale-method",
        type=str,
        default="lanczos",
        choices=["realesrgan", "bicubic", "lanczos"],
        help="Upscaling method"
    )
    parser.add_argument(
        "--temporal-smoothing",
        action="store_true",
        help="Apply temporal smoothing to reduce flicker"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type for inference"
    )

    args = parser.parse_args()

    # Batch processing mode
    if args.batch:
        process_batch(args)
        return

    # Single video mode
    if not args.character or not args.output:
        parser.error("--character and --output are required for single video mode")

    generate_single_video(args)


def generate_single_video(args):
    """Generate a single video."""
    import torch

    character_image = Path(args.character)
    reference_video = Path(args.reference) if args.reference else None
    output_video = Path(args.output)

    print("Motion Control Video Generation")
    print("=" * 60)

    # Validate inputs
    print("\n1. Validating inputs...")
    if not validate_inputs(character_image, reference_video):
        print("✗ Input validation failed")
        sys.exit(1)

    # Parse target resolution
    width, height = map(int, args.target_resolution.split('x'))
    target_res = (width, height)

    # Set dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Initialize generator
    print(f"\n2. Loading Wan2.1 model from {args.model_path}...")
    generator = Wan21VideoGenerator(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype
    )

    # Generate video
    print(f"\n3. Generating video ({args.duration}s at {args.fps}fps)...")
    if args.prompt:
        print(f"   Prompt: {args.prompt}")

    # Calculate num_frames and ensure it's valid (4n+1)
    num_frames = int(args.duration * args.fps)
    num_frames = ((num_frames - 1) // 4) * 4 + 1

    if args.duration <= 3.5:  # Single segment
        frames = generator.generate_video(
            character_image=character_image,
            num_frames=num_frames,
            fps=args.fps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            prompt=args.prompt
        )
    else:  # Multiple segments for longer videos
        frames = generator.generate_long_video(
            character_image=character_image,
            target_duration=args.duration,
            fps=args.fps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            prompt=args.prompt
        )

    # Identity preservation
    if args.preserve_identity:
        print("\n4. Applying identity preservation...")
        frames = preserve_identity(
            frames,
            character_image,
            device=args.device
        )

    # Postprocessing
    print(f"\n5. Postprocessing (target: {target_res})...")
    frames = postprocess_video(
        frames,
        target_resolution=target_res,
        upscale_method=args.upscale_method,
        apply_smoothing=args.temporal_smoothing,
        device=args.device
    )

    # Save video
    print(f"\n6. Saving video...")
    temp_output = output_video.parent / f"{output_video.stem}_temp.mp4"
    save_frames_as_video(frames, temp_output, fps=args.fps)

    # Add audio if requested
    if args.preserve_audio:
        print("\n7. Extracting and merging audio...")
        audio_handler = AudioHandler()
        audio_path = audio_handler.extract_audio(reference_video)

        # Trim audio to match video duration
        audio_duration = audio_handler.get_audio_duration(audio_path)
        if audio_duration > args.duration:
            audio_trimmed = output_video.parent / f"{output_video.stem}_audio_trimmed.m4a"
            audio_handler.trim_audio(audio_path, audio_trimmed, duration=args.duration)
            audio_path.unlink()
            audio_path = audio_trimmed

        audio_handler.merge_audio_video(temp_output, audio_path, output_video)
        temp_output.unlink()  # Remove temp video
        audio_path.unlink()  # Remove temp audio
    else:
        temp_output.rename(output_video)

    # Cleanup
    generator.clear_cache()

    print("\n" + "=" * 60)
    print(f"✓ Video generation complete!")
    print(f"✓ Saved to: {output_video}")
    print(f"✓ Duration: {args.duration}s")
    print(f"✓ Resolution: {target_res[0]}x{target_res[1]}")
    print("=" * 60)


def process_batch(args):
    """Process batch of videos."""
    from batch_processor import BatchProcessor

    manifest_path = Path(args.batch)

    if args.job_index is not None:
        # SLURM array job mode - process single job from manifest
        with open(manifest_path, 'r') as f:
            data = json.load(f)

        jobs = data['jobs']
        if args.job_index >= len(jobs):
            print(f"Error: Job index {args.job_index} out of range (0-{len(jobs)-1})")
            sys.exit(1)

        job = jobs[args.job_index]

        print(f"Processing job {args.job_index}: {job.get('id', f'job_{args.job_index}')}")

        # Create temporary single-job manifest
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'jobs': [job]}, f)
            temp_manifest = f.name

        # Process single job
        processor = BatchProcessor(temp_manifest)
        processor.process_sequential(
            model_path=args.model_path,
            device=args.device
        )

        # Cleanup
        Path(temp_manifest).unlink()

    else:
        # Local batch processing
        processor = BatchProcessor(manifest_path)
        processor.process_sequential(
            model_path=args.model_path,
            device=args.device
        )


if __name__ == "__main__":
    main()
