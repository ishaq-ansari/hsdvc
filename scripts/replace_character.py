#!/usr/bin/env python3
"""
Character replacement script.
Replace character in compiled video while preserving motion.
"""

import argparse
from pathlib import Path
import torch

from hsdvc import VideoCompiler, CharacterReplacer


def parse_args():
    parser = argparse.ArgumentParser(description="Replace character in compiled video")
    
    # Input/Output
    parser.add_argument("--compiled_dir", type=str, required=True, help="Compiled video directory")
    parser.add_argument("--new_character", type=str, required=True, help="New character image")
    parser.add_argument("--output", type=str, required=True, help="Output video path")
    
    # Options
    parser.add_argument("--preserve_style", action="store_true", help="Preserve style from original")
    parser.add_argument("--identity_strength", type=float, default=1.0, help="Identity strength [0, 1]")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames (all if None)")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    
    # Model
    parser.add_argument("--base_model", type=str, default="cogvideox-5b", help="Base model name")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print("HSDVC Character Replacement")
    print(f"{'='*60}\n")
    print(f"Compiled directory: {args.compiled_dir}")
    print(f"New character: {args.new_character}")
    print(f"Output: {args.output}")
    print(f"Identity strength: {args.identity_strength}")
    print(f"Preserve style: {args.preserve_style}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Device: {device}\n")
    
    # Load compiler
    print("Loading VideoCompiler...")
    compiler = VideoCompiler.from_pretrained(args.base_model)
    compiler = compiler.to(device)
    
    # Load compiled data
    print(f"Loading compiled data from {args.compiled_dir}...")
    compiler.load_compiled(args.compiled_dir)
    
    # Create replacer
    print("Initializing CharacterReplacer...")
    replacer = CharacterReplacer(compiler)
    
    # Replace character
    video = replacer.replace(
        new_character_image=args.new_character,
        preserve_motion=True,
        preserve_style=args.preserve_style,
        identity_strength=args.identity_strength,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_path=args.output
    )
    
    print(f"\n✓ Character replacement complete!")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
