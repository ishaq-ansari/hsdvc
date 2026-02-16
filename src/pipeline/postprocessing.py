"""
Postprocessing utilities for motion control pipeline.
Handles upscaling, temporal smoothing, and video enhancement.
"""

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False


class VideoUpscaler:
    """Upscale video frames using Real-ESRGAN or other methods."""

    def __init__(
        self,
        model_name: str = "RealESRGAN_x2plus",
        device: str = "cuda",
        half_precision: bool = True
    ):
        """
        Initialize video upscaler.

        Args:
            model_name: Model name (RealESRGAN_x2plus, RealESRGAN_x4plus, etc.)
            device: Device for inference (cuda or cpu)
            half_precision: Use FP16 for faster inference
        """
        self.model_name = model_name
        self.device = device
        self.half_precision = half_precision and device == "cuda"

        if not REALESRGAN_AVAILABLE:
            raise ImportError(
                "Real-ESRGAN not available. "
                "Install with: pip install realesrgan"
            )

        self._load_model()

    def _load_model(self):
        """Load Real-ESRGAN model."""
        # Determine scale from model name
        if "x2" in self.model_name:
            scale = 2
        elif "x4" in self.model_name:
            scale = 4
        else:
            scale = 2

        # Initialize model
        if self.model_name == "RealESRGAN_x4plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        elif self.model_name == "RealESRGAN_x2plus":
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
        else:
            # Default to x2
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2

        # Create upsampler
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=None,  # Will download automatically
            model=model,
            tile=256,  # Tile size for GPU memory management
            tile_pad=10,
            pre_pad=0,
            half=self.half_precision,
            device=self.device
        )

    def upscale_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Upscale a single frame.

        Args:
            frame: Input frame (H, W, C) as numpy array

        Returns:
            Upscaled frame
        """
        output,  _ = self.upsampler.enhance(frame, outscale=None)
        return output

    def upscale_frames(
        self,
        frames: List[np.ndarray],
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Upscale multiple frames.

        Args:
            frames: List of frames to upscale
            show_progress: Show progress bar

        Returns:
            List of upscaled frames
        """
        upscaled = []

        iterator = tqdm(frames, desc="Upscaling frames") if show_progress else frames

        for frame in iterator:
            upscaled_frame = self.upscale_frame(frame)
            upscaled.append(upscaled_frame)

        return upscaled


def upscale_to_resolution(
    frames: List[np.ndarray],
    target_resolution: tuple,
    method: str = "realesrgan",
    device: str = "cuda"
) -> List[np.ndarray]:
    """
    Upscale frames to target resolution.

    Args:
        frames: List of input frames
        target_resolution: Target (width, height)
        method: Upscaling method ('realesrgan', 'bicubic', 'lanczos')
        device: Device for inference

    Returns:
        List of upscaled frames
    """
    if not frames:
        return frames

    current_h, current_w = frames[0].shape[:2]
    target_w, target_h = target_resolution

    if method == "realesrgan":
        # Use Real-ESRGAN
        # Determine scale needed
        scale_w = target_w / current_w
        scale_h = target_h / current_h
        scale = max(scale_w, scale_h)

        if scale <= 2:
            model_name = "RealESRGAN_x2plus"
        else:
            model_name = "RealESRGAN_x4plus"

        upscaler = VideoUpscaler(model_name=model_name, device=device)
        upscaled = upscaler.upscale_frames(frames)

        # Resize to exact target if needed
        if upscaled[0].shape[:2] != (target_h, target_w):
            upscaled = [
                cv2.resize(frame, target_resolution, interpolation=cv2.INTER_LANCZOS4)
                for frame in tqdm(upscaled, desc="Final resize")
            ]

    elif method == "bicubic":
        upscaled = [
            cv2.resize(frame, target_resolution, interpolation=cv2.INTER_CUBIC)
            for frame in tqdm(frames, desc="Bicubic upscaling")
        ]

    elif method == "lanczos":
        upscaled = [
            cv2.resize(frame, target_resolution, interpolation=cv2.INTER_LANCZOS4)
            for frame in tqdm(frames, desc="Lanczos upscaling")
        ]

    else:
        raise ValueError(f"Unknown upscaling method: {method}")

    return upscaled


def apply_temporal_smoothing(
    frames: List[np.ndarray],
    kernel_size: int = 3,
    sigma: float = 1.0
) -> List[np.ndarray]:
    """
    Apply temporal smoothing to reduce flickering.

    Args:
        frames: List of input frames
        kernel_size: Temporal kernel size (must be odd)
        sigma: Gaussian sigma

    Returns:
        Smoothed frames
    """
    if len(frames) < kernel_size:
        return frames

    if kernel_size % 2 == 0:
        kernel_size += 1

    half_kernel = kernel_size // 2

    # Create temporal gaussian weights
    weights = np.exp(-np.arange(-half_kernel, half_kernel + 1) ** 2 / (2 * sigma ** 2))
    weights /= weights.sum()

    smoothed = []

    for i in tqdm(range(len(frames)), desc="Temporal smoothing"):
        # Get temporal window
        start_idx = max(0, i - half_kernel)
        end_idx = min(len(frames), i + half_kernel + 1)

        # Adjust weights for boundary frames
        if i < half_kernel or i >= len(frames) - half_kernel:
            active_weights = weights[
                (half_kernel - (i - start_idx)):(half_kernel + (end_idx - i))
            ]
            active_weights /= active_weights.sum()
        else:
            active_weights = weights

        # Weighted average of frames
        window_frames = frames[start_idx:end_idx]
        smoothed_frame = np.zeros_like(frames[i], dtype=np.float32)

        for j, frame in enumerate(window_frames):
            smoothed_frame += frame.astype(np.float32) * active_weights[j]

        smoothed.append(smoothed_frame.astype(frames[i].dtype))

    return smoothed


def blend_frames(frame1: np.ndarray, frame2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend two frames together.

    Args:
        frame1: First frame
        frame2: Second frame
        alpha: Blend factor (0 = all frame1, 1 = all frame2)

    Returns:
        Blended frame
    """
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)


def extend_video_with_segments(
    frame_segments: List[List[np.ndarray]],
    blend_frames_count: int = 5
) -> List[np.ndarray]:
    """
    Extend video by stitching multiple segments with blending.

    Args:
        frame_segments: List of frame lists (each segment)
        blend_frames_count: Number of frames to blend between segments

    Returns:
        Extended video frames
    """
    if not frame_segments:
        return []

    if len(frame_segments) == 1:
        return frame_segments[0]

    extended = []

    for i, segment in enumerate(frame_segments):
        if i == 0:
            # First segment: add all frames except last few (for blending)
            extended.extend(segment[:-blend_frames_count])
        elif i == len(frame_segments) - 1:
            # Last segment: blend start, then add rest
            prev_segment_end = frame_segments[i - 1][-blend_frames_count:]

            for j in range(blend_frames_count):
                alpha = (j + 1) / (blend_frames_count + 1)
                blended = blend_frames(prev_segment_end[j], segment[j], alpha)
                extended.append(blended)

            extended.extend(segment[blend_frames_count:])
        else:
            # Middle segment: blend start, add middle, keep end for next blend
            prev_segment_end = frame_segments[i - 1][-blend_frames_count:]

            for j in range(blend_frames_count):
                alpha = (j + 1) / (blend_frames_count + 1)
                blended = blend_frames(prev_segment_end[j], segment[j], alpha)
                extended.append(blended)

            extended.extend(segment[blend_frames_count:-blend_frames_count])

    return extended


def enhance_colors(
    frames: List[np.ndarray],
    saturation: float = 1.1,
    brightness: float = 1.0,
    contrast: float = 1.0
) -> List[np.ndarray]:
    """
    Enhance colors in video frames.

    Args:
        frames: Input frames
        saturation: Saturation multiplier (>1 = more saturated)
        brightness: Brightness multiplier
        contrast: Contrast multiplier

    Returns:
        Enhanced frames
    """
    enhanced = []

    for frame in tqdm(frames, desc="Enhancing colors"):
        # Convert to float
        img = frame.astype(np.float32) / 255.0

        # Adjust brightness
        if brightness != 1.0:
            img = img * brightness

        # Adjust contrast
        if contrast != 1.0:
            img = (img - 0.5) * contrast + 0.5

        # Adjust saturation
        if saturation != 1.0:
            # Convert to HSV
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Adjust saturation channel
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)

            # Convert back to RGB
            img_uint8 = hsv.astype(np.uint8)
            img = cv2.cvtColor(img_uint8, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        # Clip and convert back
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        enhanced.append(img)

    return enhanced


# Convenience function
def postprocess_video(
    frames: List[np.ndarray],
    target_resolution: Optional[tuple] = None,
    upscale_method: str = "lanczos",
    apply_smoothing: bool = True,
    enhance: bool = False,
    device: str = "cuda"
) -> List[np.ndarray]:
    """
    Apply full postprocessing pipeline to video frames.

    Args:
        frames: Input frames
        target_resolution: Target (width, height). If None, no upscaling.
        upscale_method: Upscaling method
        apply_smoothing: Apply temporal smoothing
        enhance: Apply color enhancement
        device: Device for processing

    Returns:
        Processed frames
    """
    processed = frames

    # Upscale if needed
    if target_resolution is not None:
        print(f"Upscaling to {target_resolution}...")
        processed = upscale_to_resolution(
            processed,
            target_resolution,
            method=upscale_method,
            device=device
        )

    # Apply temporal smoothing
    if apply_smoothing:
        print("Applying temporal smoothing...")
        processed = apply_temporal_smoothing(processed)

    # Enhance colors
    if enhance:
        print("Enhancing colors...")
        processed = enhance_colors(processed)

    return processed
