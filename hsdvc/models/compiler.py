"""
Video Compiler: Main class for per-video adaptation and compilation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
import json

from hsdvc.config import HSDVCConfig, CogVideoXConfig, ControlNetConfig
from hsdvc.models.motion import MotionExtractor, MotionData
from hsdvc.models.identity import IdentityEncoder, IdentityEmbedding
from hsdvc.models.geometry import create_geometry
from hsdvc.models.cogvideox import load_cogvideox_with_structure, CogVideoXWithStructure


class VideoCompiler(nn.Module):
    """
    Main video compiler that orchestrates all components.
    Performs fast per-video adaptation (5-10 minutes).
    """
    
    def __init__(self, config: Optional[HSDVCConfig] = None):
        super().__init__()
        
        if config is None:
            config = HSDVCConfig()
        
        self.config = config
        
        # Initialize components
        print("Initializing Motion Extractor...")
        self.motion_extractor = MotionExtractor(config.motion)
        
        print("Initializing Identity Encoder...")
        self.identity_encoder = IdentityEncoder(config.identity)
        
        print("Initializing Geometry Representation...")
        self.geometry = create_geometry(config.geometry)
        
        print("Initializing CogVideoX with Structure...")
        self.cogvideox = load_cogvideox_with_structure(
            video_config=config.cogvideox,
            control_config=config.controlnet,
            enable_lora=True
        )
        
        # Compiled video data
        self.compiled_motion: Optional[MotionData] = None
        self.compiled_identity: Optional[IdentityEmbedding] = None
        self.compiled_video_path: Optional[str] = None
    
    @classmethod
    def from_pretrained(cls, model_name: str = "cogvideox-5b") -> "VideoCompiler":
        """
        Load pre-trained video compiler.
        
        Args:
            model_name: Base model name
            
        Returns:
            VideoCompiler instance
        """
        config = HSDVCConfig()
        config.cogvideox.model_name = f"THUDM/{model_name}"
        
        return cls(config)
    
    def compile_video(
        self,
        video_path: str,
        motion_data: Optional[MotionData] = None,
        identity_image: Optional[str] = None,
        num_steps: int = 500,
        learning_rate: float = 5e-4,
        save_dir: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Compile video: extract structure and adapt model.
        
        Args:
            video_path: Path to input video
            motion_data: Pre-extracted motion data (optional)
            identity_image: Path to identity reference image (optional)
            num_steps: Number of adaptation steps
            learning_rate: Learning rate
            save_dir: Directory to save compiled data
            
        Returns:
            Dictionary with compilation results
        """
        print(f"\n{'='*60}")
        print(f"Compiling Video: {video_path}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract motion
        if motion_data is None:
            print("Step 1/4: Extracting motion...")
            motion_data = self.motion_extractor.extract(video_path)
            print(f"  ✓ Extracted {motion_data.num_frames} frames")
            print(f"  ✓ Poses: {motion_data.poses_3d.shape}")
            print(f"  ✓ Depth: {motion_data.depth_maps.shape}")
            print(f"  ✓ Flow: {motion_data.optical_flow.shape}")
        else:
            print("Step 1/4: Using provided motion data ✓")
        
        self.compiled_motion = motion_data
        
        # Step 2: Extract identity
        if identity_image is not None:
            print(f"\nStep 2/4: Extracting identity from {identity_image}...")
            identity_embedding = self.identity_encoder.encode_from_path(identity_image)
        else:
            print("\nStep 2/4: Extracting identity from video...")
            # Use first frame
            video_frames = self._load_video_frames(video_path)
            identity_embedding = self.identity_encoder(video_frames[:1])
        
        print(f"  ✓ Shape embedding: {identity_embedding.shape.shape}")
        print(f"  ✓ Appearance embedding: {identity_embedding.appearance.shape}")
        print(f"  ✓ Texture embedding: {identity_embedding.texture.shape}")
        
        self.compiled_identity = identity_embedding
        
        # Step 3: Initialize geometry
        print("\nStep 3/4: Initializing 3D geometry...")
        if hasattr(self.geometry, 'from_motion_data'):
            self.geometry.from_motion_data(motion_data)
            print(f"  ✓ Initialized {self.config.geometry.type}")
        
        # Step 4: Adapt CogVideoX model
        print(f"\nStep 4/4: Adapting CogVideoX model ({num_steps} steps)...")
        
        # Load video frames
        video_frames = self._load_video_frames(video_path)
        
        # Prepare control signals
        control_signals = self._prepare_control_signals(motion_data)
        
        # Run compilation
        self.cogvideox.compile_for_video(
            video_data=video_frames,
            motion_data=control_signals,
            identity_embedding=identity_embedding,
            num_steps=num_steps,
            learning_rate=learning_rate
        )
        
        self.compiled_video_path = video_path
        
        # Save compiled data
        if save_dir is not None:
            self.save_compiled(save_dir)
        
        print(f"\n{'='*60}")
        print("✓ Video compilation complete!")
        print(f"{'='*60}\n")
        
        return {
            "motion_data": motion_data,
            "identity_embedding": identity_embedding,
            "num_frames": motion_data.num_frames,
            "resolution": motion_data.resolution,
        }
    
    def save_compiled(self, save_dir: str):
        """Save compiled video data."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save motion data
        if self.compiled_motion is not None:
            torch.save(
                self.compiled_motion,
                save_path / "motion_data.pt"
            )
        
        # Save identity embedding
        if self.compiled_identity is not None:
            torch.save(
                self.compiled_identity.to_dict(),
                save_path / "identity_embedding.pt"
            )
        
        # Save model state (LoRA weights)
        if self.cogvideox.lora_adapters is not None:
            self.cogvideox.lora_adapters.save_pretrained(
                save_path / "lora_adapters"
            )
        
        # Save metadata
        metadata = {
            "video_path": self.compiled_video_path,
            "num_frames": self.compiled_motion.num_frames if self.compiled_motion else 0,
            "resolution": self.compiled_motion.resolution if self.compiled_motion else (0, 0),
        }
        
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Compiled data saved to {save_dir}")
    
    def load_compiled(self, load_dir: str):
        """Load compiled video data."""
        load_path = Path(load_dir)
        
        # Load motion data
        if (load_path / "motion_data.pt").exists():
            self.compiled_motion = torch.load(load_path / "motion_data.pt")
        
        # Load identity embedding
        if (load_path / "identity_embedding.pt").exists():
            identity_dict = torch.load(load_path / "identity_embedding.pt")
            self.compiled_identity = IdentityEmbedding(**identity_dict)
        
        # Load LoRA weights
        if (load_path / "lora_adapters").exists():
            from peft import PeftModel
            self.cogvideox.lora_adapters = PeftModel.from_pretrained(
                self.cogvideox.base_model,
                load_path / "lora_adapters"
            )
        
        print(f"✓ Compiled data loaded from {load_dir}")
    
    def _load_video_frames(self, video_path: str) -> torch.Tensor:
        """Load video frames as tensor."""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            target_size = self.config.cogvideox.image_size
            frame = cv2.resize(frame, (target_size[1], target_size[0]))
            
            # Convert to tensor and normalize
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        
        cap.release()
        
        return torch.stack(frames)
    
    def _prepare_control_signals(self, motion_data: MotionData) -> Dict[str, torch.Tensor]:
        """Prepare control signals for conditioning."""
        # Normalize and prepare control signals
        control_signals = {
            "poses": motion_data.poses_2d.unsqueeze(0),  # Add batch dim
            "depth": motion_data.depth_maps.unsqueeze(0).unsqueeze(2),  # [B, T, 1, H, W]
            "flow": motion_data.optical_flow.unsqueeze(0).permute(0, 1, 4, 2, 3),  # [B, T, 2, H, W]
        }
        
        # Create mask (all ones for now)
        B, T = 1, motion_data.num_frames
        H, W = motion_data.resolution
        control_signals["mask"] = torch.ones(B, T, 1, H, W)
        
        return control_signals
    
    def generate(
        self,
        num_frames: Optional[int] = None,
        identity_embedding: Optional[IdentityEmbedding] = None,
        prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        Generate video using compiled motion and identity.
        
        Args:
            num_frames: Number of frames to generate (uses compiled if None)
            identity_embedding: Identity to use (uses compiled if None)
            prompt: Text prompt (optional)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            Generated video tensor [T, C, H, W]
        """
        if self.compiled_motion is None:
            raise ValueError("No compiled motion data. Run compile_video() first.")
        
        if identity_embedding is None:
            identity_embedding = self.compiled_identity
        
        if num_frames is None:
            num_frames = self.compiled_motion.num_frames
        
        print(f"\nGenerating video with {num_frames} frames...")
        
        # Prepare control signals
        control_signals = self._prepare_control_signals(self.compiled_motion)
        
        # Initialize latents
        B = 1
        C = self.config.cogvideox.latent_channels
        T = num_frames
        H = self.config.cogvideox.image_size[0] // self.config.cogvideox.vae_scale_factor
        W = self.config.cogvideox.image_size[1] // self.config.cogvideox.vae_scale_factor
        
        latents = torch.randn(B, C, T, H, W, device=self.device)
        
        # Diffusion sampling loop (simplified DDPM)
        timesteps = torch.linspace(999, 0, num_inference_steps, device=self.device).long()
        
        for i, t in enumerate(timesteps):
            print(f"  Step {i+1}/{num_inference_steps}", end="\r")
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.cogvideox(
                    latents,
                    t.unsqueeze(0),
                    control_signals,
                    identity_embedding,
                    return_dict=False
                )
            
            # DDPM update (simplified)
            alpha = 1.0 - t.float() / 1000.0
            latents = latents - alpha * noise_pred * 0.01
        
        print("\n✓ Generation complete!")
        
        # Decode latents to pixels (simplified - use VAE in practice)
        video = self._decode_latents(latents)
        
        return video.squeeze(0)
    
    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to pixel space (simplified)."""
        # In practice, use CogVideoX VAE decoder
        # For now, just resize and normalize
        B, C, T, H, W = latents.shape
        
        target_h = self.config.cogvideox.image_size[0]
        target_w = self.config.cogvideox.image_size[1]
        
        # Reshape for upsampling
        latents = latents.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        latents = latents.reshape(B * T, C, H, W)
        
        # Upsample
        video = torch.nn.functional.interpolate(
            latents,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )
        
        # Project to RGB
        video = video[:, :3]  # Take first 3 channels
        
        # Reshape back
        video = video.reshape(B, T, 3, target_h, target_w)
        
        # Normalize to [0, 1]
        video = torch.sigmoid(video)
        
        return video
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
