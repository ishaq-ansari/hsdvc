"""
Video generator using Wan2.1 I2V model for motion control.
Core pipeline for generating videos from character images and reference motions.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# Add Wan2.1 model path
WAN21_PATH = Path(__file__).parent.parent.parent / "models" / "wan21"
if WAN21_PATH.exists():
    sys.path.insert(0, str(WAN21_PATH))

try:
    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS

    # Patch flash_attention to use fallback when not available
    from wan.modules import attention as attn_module

    if not (attn_module.FLASH_ATTN_2_AVAILABLE or attn_module.FLASH_ATTN_3_AVAILABLE):
        # Create a wrapper that adapts flash_attention signature to attention signature
        def flash_attention_fallback(
            q, k, v,
            q_lens=None,
            k_lens=None,
            dropout_p=0.,
            softmax_scale=None,
            q_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
            dtype=torch.bfloat16,
            version=None  # flash_attention uses 'version'
        ):
            # Call attention() with fa_version instead of version
            return attn_module.attention(
                q=q, k=k, v=v,
                q_lens=q_lens,
                k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                fa_version=version  # Map version -> fa_version
            )

        # Patch in all modules that imported flash_attention
        from wan import modules as wan_modules
        from wan.modules import clip as clip_module
        from wan.modules import model as model_module

        attn_module.flash_attention = flash_attention_fallback
        clip_module.flash_attention = flash_attention_fallback
        model_module.flash_attention = flash_attention_fallback
        wan_modules.flash_attention = flash_attention_fallback
        print("✓ Patched flash_attention to use PyTorch scaled_dot_product_attention fallback")

    WAN_AVAILABLE = True
except ImportError:
    WAN_AVAILABLE = False

from pipeline.preprocessing import preprocess_character_image, align_image_to_video
from utils.video_utils import VideoReader, load_image


class Wan21VideoGenerator:
    """
    Video generator using Wan2.1 I2V model.
    Generates videos where a character performs motions from a reference video.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enable_xformers: bool = True
    ):
        """
        Initialize Wan2.1 video generator.

        Args:
            model_path: Path to Wan2.1 model checkpoints
            device: Device for inference (cuda or cpu)
            dtype: Data type for inference (float16 or float32)
            enable_xformers: Enable xFormers memory efficient attention
        """
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.enable_xformers = enable_xformers

        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load Wan2.1 I2V model pipeline."""
        if not WAN_AVAILABLE:
            raise ImportError(
                "Wan2.1 module not available. "
                "Make sure you have:\n"
                "1. Cloned Wan2.1 repository to models/wan21\n"
                "2. Installed Wan2.1 dependencies"
            )

        try:
            print(f"Loading Wan2.1 I2V model from {self.model_path}...")

            # Get i2v-14B config
            cfg = WAN_CONFIGS['i2v-14B']

            # Determine device_id from device string
            if isinstance(self.device, str):
                device_id = int(self.device.split(':')[1]) if ':' in self.device else 0
            else:
                device_id = 0

            # Initialize Wan2.1 I2V pipeline
            self.pipeline = wan.WanI2V(
                config=cfg,
                checkpoint_dir=str(self.model_path),
                device_id=device_id,
                rank=0,  # Single GPU mode
                t5_fsdp=False,
                dit_fsdp=False,
                use_usp=False,
                t5_cpu=False,  # Keep T5 on GPU for better performance
                init_on_cpu=False,  # Direct GPU initialization
            )

            print("✓ Model loaded successfully")
            print(f"✓ Using device: cuda:{device_id}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load Wan2.1 model: {e}\n"
                f"Make sure you have:\n"
                f"1. Cloned Wan2.1 repository to models/wan21\n"
                f"2. Downloaded model checkpoints to {self.model_path}\n"
                f"3. Installed all dependencies"
            ) from e

    def generate_video(
        self,
        character_image: Union[str, Path, Image.Image, np.ndarray],
        reference_video: Optional[Union[str, Path]] = None,
        num_frames: int = 81,  # Default for Wan2.1 (must be 4n+1)
        fps: int = 24,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 40,  # Default for i2v
        seed: Optional[int] = None,
        prompt: Optional[str] = None,  # Text prompt for guidance
        motion_bucket_id: int = 127,  # Not used in Wan2.1
        noise_aug_strength: float = 0.02  # Not used in Wan2.1
    ) -> List[np.ndarray]:
        """
        Generate video from character image.

        Args:
            character_image: Input character image
            reference_video: Optional reference video for motion extraction (not used currently)
            num_frames: Number of frames to generate (must be 4n+1, e.g., 81, 121, 161)
            fps: Target frames per second
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducibility
            prompt: Text prompt to guide video generation (e.g., 'person walking slowly')
            motion_bucket_id: Not used (kept for API compatibility)
            noise_aug_strength: Not used (kept for API compatibility)

        Returns:
            List of generated frames as numpy arrays (H, W, C)
        """
        # Load and preprocess character image
        if isinstance(character_image, (str, Path)):
            character_image = load_image(character_image)
        elif isinstance(character_image, np.ndarray):
            character_image = Image.fromarray(character_image)

        # Validate frame_num (must be 4n+1)
        if (num_frames - 1) % 4 != 0:
            # Round to nearest valid frame_num
            num_frames = ((num_frames - 1) // 4) * 4 + 1
            print(f"Warning: num_frames adjusted to {num_frames} (must be 4n+1)")

        # Set random seed if provided
        if seed is None:
            seed = -1

        # Determine size based on image aspect ratio
        # Use lower resolution to avoid OOM during VAE decoding
        w, h = character_image.size
        aspect_ratio = h / w

        if aspect_ratio > 1.2:  # Portrait
            size_key = "480*832"
            max_area = MAX_AREA_CONFIGS["480*832"]
            shift = 3.0
            # Note: VAE decoder processes all frames at once
            # Memory usage: ~77GB for 49 frames, ~50GB for 33 frames
            if num_frames > 49:
                print(f"Warning: Reducing frames from {num_frames} to 49 for portrait orientation")
                num_frames = 49
        elif aspect_ratio < 0.8:  # Landscape
            size_key = "1280*720"
            max_area = MAX_AREA_CONFIGS["1280*720"]
            shift = 5.0
        else:  # Square-ish
            size_key = "720*1280"
            max_area = MAX_AREA_CONFIGS["720*1280"]
            shift = 5.0

        # Use provided prompt or default
        if prompt is None:
            prompt = "High quality, smooth motion, natural movement, consistent character"

        print(f"Generating {num_frames} frames ({num_frames/fps:.1f}s) at {size_key}...")
        print(f"Prompt: '{prompt}'")
        print(f"Using guidance_scale={guidance_scale}, steps={num_inference_steps}, seed={seed}")

        try:
            # Generate video using Wan2.1 I2V
            video_tensor = self.pipeline.generate(
                input_prompt=prompt,
                img=character_image,
                max_area=max_area,
                frame_num=num_frames,
                shift=shift,
                sample_solver='unipc',
                sampling_steps=num_inference_steps,
                guide_scale=guidance_scale,
                seed=seed,
                offload_model=True  # Offload to save VRAM
            )

            # Convert output to list of numpy arrays
            # video_tensor shape: (C, F, H, W) where C=3, F=num_frames
            print(f"Generation complete. Output shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
            print("Converting output to frames...")
            numpy_frames = []

            # Convert from tensor to numpy array
            if isinstance(video_tensor, torch.Tensor):
                # Clear GPU cache before moving to CPU
                torch.cuda.empty_cache()

                # Process frames in smaller chunks to reduce memory usage
                chunk_size = 10  # Process 10 frames at a time
                num_chunks = (video_tensor.shape[1] + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, video_tensor.shape[1])

                    # Extract chunk: (C, chunk_frames, H, W)
                    chunk_tensor = video_tensor[:, start_idx:end_idx, :, :]

                    # Move chunk to CPU and convert to numpy
                    chunk_np = chunk_tensor.cpu().numpy()

                    # Transpose: (C, F, H, W) -> (F, H, W, C)
                    chunk_np = np.transpose(chunk_np, (1, 2, 3, 0))

                    # Denormalize from [-1, 1] to [0, 255]
                    chunk_np = (chunk_np * 0.5 + 0.5) * 255
                    chunk_np = np.clip(chunk_np, 0, 255).astype(np.uint8)

                    # Add frames to list
                    for i in range(chunk_np.shape[0]):
                        numpy_frames.append(chunk_np[i])

                    # Delete chunk to free memory
                    del chunk_tensor, chunk_np

                    # Progress indicator
                    if (chunk_idx + 1) % 5 == 0 or chunk_idx == num_chunks - 1:
                        print(f"  Processed {min(end_idx, video_tensor.shape[1])}/{video_tensor.shape[1]} frames")

                # Delete original tensor and clear cache
                del video_tensor
                torch.cuda.empty_cache()
            else:
                # Handle other output formats if needed
                raise TypeError(f"Unexpected output type: {type(video_tensor)}")

            print(f"✓ Generated {len(numpy_frames)} frames")
            return numpy_frames

        except Exception as e:
            raise RuntimeError(f"Video generation failed: {e}") from e

    def generate_long_video(
        self,
        character_image: Union[str, Path, Image.Image, np.ndarray],
        target_duration: float = 30.0,
        fps: int = 24,
        segment_duration: float = 3.375,  # 81 frames at 24fps = 3.375s
        **generation_kwargs
    ) -> List[np.ndarray]:
        """
        Generate long video by creating and stitching multiple segments.

        Args:
            character_image: Input character image
            target_duration: Target video duration in seconds
            fps: Frames per second
            segment_duration: Duration of each segment (default 3.375s = 81 frames)
            **generation_kwargs: Additional arguments for generate_video

        Returns:
            List of frames for full video
        """
        num_segments = int(np.ceil(target_duration / segment_duration))
        frames_per_segment = int(segment_duration * fps)

        # Ensure frames_per_segment is valid (4n+1)
        frames_per_segment = ((frames_per_segment - 1) // 4) * 4 + 1

        print(f"Generating {num_segments} segments for {target_duration}s video...")
        print(f"Each segment: {frames_per_segment} frames ({frames_per_segment/fps:.2f}s)")

        all_segments = []

        # Generate first segment
        current_image = character_image
        for i in range(num_segments):
            print(f"\nGenerating segment {i+1}/{num_segments}...")

            # Generate segment
            segment_frames = self.generate_video(
                character_image=current_image,
                num_frames=frames_per_segment,
                fps=fps,
                **generation_kwargs
            )

            all_segments.append(segment_frames)

            # Use last frame as input for next segment (for continuity)
            if i < num_segments - 1:
                current_image = Image.fromarray(segment_frames[-1])

        # Stitch segments with blending
        from pipeline.postprocessing import extend_video_with_segments

        print("\nStitching segments...")
        extended_frames = extend_video_with_segments(
            all_segments,
            blend_frames_count=int(fps * 0.2)  # 0.2s blend
        )

        # Trim to exact duration
        target_frames = int(target_duration * fps)
        extended_frames = extended_frames[:target_frames]

        print(f"✓ Generated {len(extended_frames)} frames ({len(extended_frames)/fps:.1f}s)")
        return extended_frames

    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Convenience functions
def generate_video(
    character_image: Union[str, Path],
    output_path: Union[str, Path],
    model_path: Union[str, Path],
    duration: float = 5.0,
    fps: int = 24,
    device: str = "cuda",
    **kwargs
) -> Path:
    """
    Generate video from character image.

    Args:
        character_image: Path to character image
        output_path: Output video path
        model_path: Path to Wan2.1 model
        duration: Video duration in seconds
        fps: Frames per second
        device: Device for inference
        **kwargs: Additional generation parameters

    Returns:
        Path to output video
    """
    from utils.video_utils import save_frames_as_video

    # Initialize generator
    generator = Wan21VideoGenerator(model_path=model_path, device=device)

    # Generate frames
    if duration <= 5.0:
        frames = generator.generate_video(
            character_image=character_image,
            num_frames=int(duration * fps),
            fps=fps,
            **kwargs
        )
    else:
        frames = generator.generate_long_video(
            character_image=character_image,
            target_duration=duration,
            fps=fps,
            **kwargs
        )

    # Save video
    save_frames_as_video(frames, output_path, fps=fps)

    # Clean up
    generator.clear_cache()

    return Path(output_path)
