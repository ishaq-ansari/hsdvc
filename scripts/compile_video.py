#!/usr/bin/env python3
"""
Per-video compilation script.
Adapts the model to a specific input video.
"""

import argparse
from pathlib import Path
import torch

from hsdvc import VideoCompiler
from hsdvc.config import HSDVCConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Compile video for HSDVC")
    
    # Input/Output
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--identity_image", type=str, default=None, help="Identity reference image")
    
    # Compilation
    parser.add_argument("--num_steps", type=int, default=500, help="Number of adaptation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    
    # Model
    parser.add_argument("--base_model", type=str, default="cogvideox-5b", help="Base model name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("HSDVC Per-Video Compilation")
    print(f"{'='*60}\n")
    print(f"Input video: {args.video}")
    print(f"Output directory: {args.output_dir}")
    print(f"Base model: {args.base_model}")
    print(f"Compilation steps: {args.num_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {device}\n")
    
    # Load compiler
    print("Loading VideoCompiler...")
    compiler = VideoCompiler.from_pretrained(args.base_model)
    compiler = compiler.to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        compiler.load_state_dict(
            torch.load(args.checkpoint, map_location=device),
            strict=False
        )
    
    # Compile video
    results = compiler.compile_video(
        video_path=args.video,
        identity_image=args.identity_image,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        save_dir=str(output_dir)
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Compilation Summary")
    print("="*60)
    print(f"Number of frames: {results['num_frames']}")
    print(f"Resolution: {results['resolution']}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")
    
    print("You can now use this compiled video for:")
    print("  1. Character replacement: python scripts/replace_character.py")
    print("  2. Generation: python scripts/generate.py")
    print("  3. Style transfer: python scripts/style_transfer.py")


if __name__ == "__main__":
    main()
