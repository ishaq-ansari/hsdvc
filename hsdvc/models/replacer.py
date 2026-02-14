"""
Character Replacer: Replace character identity while preserving motion.
"""

import torch
import torch.nn as nn
from typing import Optional, Union
from pathlib import Path

from hsdvc.models.compiler import VideoCompiler
from hsdvc.models.identity import IdentityEncoder, IdentityEmbedding
from hsdvc.config import HSDVCConfig


class CharacterReplacer:
    """
    Replace character in compiled video while preserving motion.
    """
    
    def __init__(self, compiler: VideoCompiler):
        """
        Initialize character replacer.
        
        Args:
            compiler: Compiled VideoCompiler instance
        """
        self.compiler = compiler
        
        if compiler.compiled_motion is None:
            raise ValueError(
                "Compiler must have compiled motion data. "
                "Run compiler.compile_video() first."
            )
    
    def replace(
        self,
        new_character_image: str,
        preserve_motion: bool = True,
        preserve_style: bool = False,
        identity_strength: float = 1.0,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Replace character with new identity.
        
        Args:
            new_character_image: Path to new character image
            preserve_motion: Whether to preserve exact motion
            preserve_style: Whether to preserve style from original
            identity_strength: Strength of identity injection [0, 1]
            num_frames: Number of frames (uses all if None)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            output_path: Path to save output video (optional)
            
        Returns:
            Generated video tensor [T, C, H, W]
        """
        print(f"\n{'='*60}")
        print(f"Replacing Character")
        print(f"{'='*60}\n")
        
        # Step 1: Extract new character identity
        print(f"Step 1/3: Extracting identity from {new_character_image}...")
        new_identity = self.compiler.identity_encoder.encode_from_path(
            new_character_image
        )
        print(f"  ✓ Shape: {new_identity.shape.shape}")
        print(f"  ✓ Appearance: {new_identity.appearance.shape}")
        print(f"  ✓ Texture: {new_identity.texture.shape}")
        
        # Step 2: Blend identities if needed
        if preserve_style and self.compiler.compiled_identity is not None:
            print("\nStep 2/3: Blending identities (preserving style)...")
            # Interpolate: keep new shape, blend appearance/texture
            blended_identity = IdentityEmbedding(
                shape=new_identity.shape,
                appearance=(
                    identity_strength * new_identity.appearance +
                    (1 - identity_strength) * self.compiler.compiled_identity.appearance
                ),
                texture=(
                    identity_strength * new_identity.texture +
                    (1 - identity_strength) * self.compiler.compiled_identity.texture
                )
            )
            final_identity = blended_identity
            print(f"  ✓ Identity blend ratio: {identity_strength:.2f}")
        else:
            print("\nStep 2/3: Using new identity directly ✓")
            final_identity = new_identity
        
        # Step 3: Generate video with new identity
        print(f"\nStep 3/3: Generating video with new character...")
        print(f"  Motion preservation: {'ON' if preserve_motion else 'OFF'}")
        print(f"  Inference steps: {num_inference_steps}")
        print(f"  Guidance scale: {guidance_scale}")
        
        video = self.compiler.generate(
            num_frames=num_frames,
            identity_embedding=final_identity,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Step 4: Save if output path provided
        if output_path is not None:
            print(f"\nSaving output to {output_path}...")
            self._save_video(video, output_path)
            print("  ✓ Video saved!")
        
        print(f"\n{'='*60}")
        print("✓ Character replacement complete!")
        print(f"{'='*60}\n")
        
        return video
    
    def interpolate_characters(
        self,
        character_images: list[str],
        num_frames_per_transition: int = 30,
        num_inference_steps: int = 50,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Interpolate between multiple characters over time.
        
        Args:
            character_images: List of character image paths
            num_frames_per_transition: Frames for each transition
            num_inference_steps: Diffusion steps
            output_path: Output path (optional)
            
        Returns:
            Video with character interpolation [T, C, H, W]
        """
        print(f"\nInterpolating between {len(character_images)} characters...")
        
        # Extract all identities
        identities = []
        for img_path in character_images:
            print(f"  Extracting: {img_path}")
            identity = self.compiler.identity_encoder.encode_from_path(img_path)
            identities.append(identity)
        
        # Generate video segments with interpolation
        all_videos = []
        
        for i in range(len(identities) - 1):
            print(f"\nGenerating transition {i+1}/{len(identities)-1}...")
            
            # Interpolate identities over time
            for t in range(num_frames_per_transition):
                alpha = t / num_frames_per_transition
                
                # Interpolate identity
                interp_identity = self.compiler.identity_encoder.interpolate(
                    identities[i],
                    identities[i + 1],
                    alpha=alpha
                )
                
                # Generate frame
                frame = self.compiler.generate(
                    num_frames=1,
                    identity_embedding=interp_identity,
                    num_inference_steps=num_inference_steps
                )
                
                all_videos.append(frame)
        
        # Concatenate all segments
        video = torch.cat(all_videos, dim=0)
        
        if output_path is not None:
            self._save_video(video, output_path)
        
        print(f"\n✓ Character interpolation complete!")
        return video
    
    def replace_with_style_transfer(
        self,
        new_character_image: str,
        style_image: str,
        style_strength: float = 0.5,
        num_inference_steps: int = 50,
        output_path: Optional[str] = None
    ) -> torch.Tensor:
        """
        Replace character with style transfer.
        
        Args:
            new_character_image: New character image
            style_image: Style reference image
            style_strength: Style strength [0, 1]
            num_inference_steps: Diffusion steps
            output_path: Output path
            
        Returns:
            Stylized video [T, C, H, W]
        """
        print(f"\nReplacing character with style transfer...")
        
        # Extract identities
        new_identity = self.compiler.identity_encoder.encode_from_path(
            new_character_image
        )
        style_identity = self.compiler.identity_encoder.encode_from_path(
            style_image
        )
        
        # Blend: use new shape, blend appearance/texture with style
        stylized_identity = IdentityEmbedding(
            shape=new_identity.shape,
            appearance=(
                (1 - style_strength) * new_identity.appearance +
                style_strength * style_identity.appearance
            ),
            texture=(
                (1 - style_strength) * new_identity.texture +
                style_strength * style_identity.texture
            )
        )
        
        # Generate
        video = self.compiler.generate(
            identity_embedding=stylized_identity,
            num_inference_steps=num_inference_steps
        )
        
        if output_path is not None:
            self._save_video(video, output_path)
        
        print(f"✓ Style transfer complete!")
        return video
    
    def _save_video(self, video: torch.Tensor, output_path: str):
        """
        Save video tensor to file.
        
        Args:
            video: [T, C, H, W] video tensor
            output_path: Output file path
        """
        import cv2
        import numpy as np
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        T, C, H, W = video.shape
        
        # Convert to numpy and scale to [0, 255]
        video_np = (video.cpu().numpy() * 255).astype(np.uint8)
        video_np = video_np.transpose(0, 2, 3, 1)  # [T, H, W, C]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.compiler.config.cogvideox.frame_rate
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (W, H)
        )
        
        # Write frames
        for frame in video_np:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        
        writer.release()
    
    def compare_identities(
        self,
        image1: str,
        image2: str
    ) -> dict:
        """
        Compare two character identities.
        
        Args:
            image1: First character image
            image2: Second character image
            
        Returns:
            Dictionary with similarity scores
        """
        # Extract identities
        identity1 = self.compiler.identity_encoder.encode_from_path(image1)
        identity2 = self.compiler.identity_encoder.encode_from_path(image2)
        
        # Compute similarities
        total_similarity = self.compiler.identity_encoder.similarity(
            identity1, identity2
        )
        
        shape_similarity = torch.nn.functional.cosine_similarity(
            identity1.shape, identity2.shape, dim=-1
        )
        
        appearance_similarity = torch.nn.functional.cosine_similarity(
            identity1.appearance, identity2.appearance, dim=-1
        )
        
        texture_similarity = torch.nn.functional.cosine_similarity(
            identity1.texture, identity2.texture, dim=-1
        )
        
        results = {
            "total_similarity": total_similarity.item(),
            "shape_similarity": shape_similarity.item(),
            "appearance_similarity": appearance_similarity.item(),
            "texture_similarity": texture_similarity.item(),
        }
        
        print("\nIdentity Comparison:")
        print(f"  Total: {results['total_similarity']:.3f}")
        print(f"  Shape: {results['shape_similarity']:.3f}")
        print(f"  Appearance: {results['appearance_similarity']:.3f}")
        print(f"  Texture: {results['texture_similarity']:.3f}")
        
        return results
