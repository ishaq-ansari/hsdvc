#!/usr/bin/env python3
"""
Quick demo script showing end-to-end workflow.
"""

import torch
from pathlib import Path

from hsdvc import VideoCompiler, CharacterReplacer


def main():
    """Run quick demo."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   HSDVC: Hybrid Structured-Diffusion Video Compiler      ║
    ║                                                           ║
    ║   Quick Demo - Character Replacement                     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check for sample data
    video_path = "data/videos/sample.mp4"
    character_path = "data/characters/new_character.jpg"
    output_dir = "outputs/demo"
    
    if not Path(video_path).exists():
        print(f"❌ Sample video not found: {video_path}")
        print("\nPlease place a sample video at 'data/videos/sample.mp4'")
        print("Or download sample data:")
        print("  python scripts/download_sample_data.py")
        return
    
    if not Path(character_path).exists():
        print(f"❌ Character image not found: {character_path}")
        print("\nPlease place a character image at 'data/characters/new_character.jpg'")
        return
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    
    # Step 1: Load compiler
    print("\n" + "="*60)
    print("Step 1: Loading VideoCompiler")
    print("="*60)
    
    compiler = VideoCompiler.from_pretrained("cogvideox-5b")
    compiler = compiler.to(device)
    print("✅ VideoCompiler loaded!")
    
    # Step 2: Compile video
    print("\n" + "="*60)
    print("Step 2: Compiling Video")
    print("="*60)
    print(f"📹 Input: {video_path}")
    
    results = compiler.compile_video(
        video_path=video_path,
        num_steps=100,  # Reduced for demo
        save_dir=output_dir + "/compiled"
    )
    
    print("\n✅ Video compiled successfully!")
    print(f"   Frames: {results['num_frames']}")
    print(f"   Resolution: {results['resolution']}")
    
    # Step 3: Replace character
    print("\n" + "="*60)
    print("Step 3: Replacing Character")
    print("="*60)
    print(f"🎭 New character: {character_path}")
    
    replacer = CharacterReplacer(compiler)
    
    output_video = replacer.replace(
        new_character_image=character_path,
        preserve_motion=True,
        num_inference_steps=20,  # Reduced for demo
        output_path=output_dir + "/output.mp4"
    )
    
    print("\n✅ Character replacement complete!")
    print(f"   Output: {output_dir}/output.mp4")
    
    # Summary
    print("\n" + "="*60)
    print("🎉 Demo Complete!")
    print("="*60)
    print("\nWhat you just did:")
    print("  1. ✅ Extracted motion from input video")
    print("  2. ✅ Compiled video-specific model")
    print("  3. ✅ Replaced character while preserving motion")
    print("\nNext steps:")
    print("  • Try different characters: python scripts/replace_character.py")
    print("  • Interpolate characters: See notebooks/04_character_interpolation.ipynb")
    print("  • Train on your data: python scripts/train_stage1.py")
    print("\nOutputs:")
    print(f"  📁 {output_dir}/compiled/ - Compiled video data")
    print(f"  🎬 {output_dir}/output.mp4 - Output video")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
