"""
Preprocessing utilities for motion control pipeline.
Validates and prepares inputs (character images and reference videos).
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from utils.video_utils import VideoReader, get_video_info, load_image


class InputValidator:
    """Validates input files for motion control pipeline."""

    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    def __init__(
        self,
        min_image_size: Tuple[int, int] = (256, 256),
        max_image_size: Tuple[int, int] = (4096, 4096),
        min_video_fps: float = 10.0,
        max_video_duration: float = 60.0
    ):
        """
        Initialize input validator.

        Args:
            min_image_size: Minimum image dimensions (width, height)
            max_image_size: Maximum image dimensions
            min_video_fps: Minimum video FPS
            max_video_duration: Maximum video duration in seconds
        """
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.min_video_fps = min_video_fps
        self.max_video_duration = max_video_duration

    def validate_image(self, image_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate character image.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (is_valid, message)
        """
        image_path = Path(image_path)

        # Check file exists
        if not image_path.exists():
            return False, f"Image file not found: {image_path}"

        # Check format
        if image_path.suffix.lower() not in self.SUPPORTED_IMAGE_FORMATS:
            return False, f"Unsupported image format: {image_path.suffix}. Supported: {self.SUPPORTED_IMAGE_FORMATS}"

        try:
            # Load and check image
            img = load_image(image_path)
            width, height = img.size

            # Check dimensions
            if width < self.min_image_size[0] or height < self.min_image_size[1]:
                return False, f"Image too small: {width}x{height}. Minimum: {self.min_image_size}"

            if width > self.max_image_size[0] or height > self.max_image_size[1]:
                return False, f"Image too large: {width}x{height}. Maximum: {self.max_image_size}"

            return True, f"✓ Valid image: {width}x{height}"

        except Exception as e:
            return False, f"Failed to load image: {str(e)}"

    def validate_video(self, video_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate reference video.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, message)
        """
        video_path = Path(video_path)

        # Check file exists
        if not video_path.exists():
            return False, f"Video file not found: {video_path}"

        # Check format
        if video_path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
            return False, f"Unsupported video format: {video_path.suffix}. Supported: {self.SUPPORTED_VIDEO_FORMATS}"

        try:
            # Get video info
            info = get_video_info(video_path)

            # Check FPS
            if info['fps'] < self.min_video_fps:
                return False, f"Video FPS too low: {info['fps']}. Minimum: {self.min_video_fps}"

            # Check duration
            if info['duration'] > self.max_video_duration:
                return False, f"Video too long: {info['duration']:.1f}s. Maximum: {self.max_video_duration}"

            # Check has frames
            if info['frame_count'] == 0:
                return False, "Video has no frames"

            return True, f"✓ Valid video: {info['resolution']} @ {info['fps']}fps, {info['duration']:.1f}s"

        except Exception as e:
            return False, f"Failed to load video: {str(e)}"

    def validate_inputs(
        self,
        character_image: Union[str, Path],
        reference_video: Optional[Union[str, Path]] = None
    ) -> Tuple[bool, str]:
        """
        Validate both character image and reference video.

        Args:
            character_image: Path to character image
            reference_video: Path to reference video (optional)

        Returns:
            Tuple of (is_valid, message)
        """
        # Validate image
        img_valid, img_msg = self.validate_image(character_image)
        if not img_valid:
            return False, f"Image validation failed: {img_msg}"

        # Validate video (optional)
        if reference_video:
            vid_valid, vid_msg = self.validate_video(reference_video)
            if not vid_valid:
                return False, f"Video validation failed: {vid_msg}"
            return True, f"✓ All inputs valid\n  {img_msg}\n  {vid_msg}"
        else:
            return True, f"✓ All inputs valid\n  {img_msg}"


def preprocess_character_image(
    image: Union[str, Path, Image.Image, np.ndarray],
    target_size: Optional[Tuple[int, int]] = None,
    center_crop: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess character image for generation.

    Args:
        image: Input image (path or loaded image)
        target_size: Target size (width, height). If None, uses original size.
        center_crop: Whether to center crop to square
        normalize: Whether to normalize to [0, 1]

    Returns:
        Preprocessed image as numpy array (H, W, C)
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        image = load_image(image)

    # Convert to PIL if numpy
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Center crop to square if requested
    if center_crop:
        width, height = image.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        image = image.crop((left, top, right, bottom))

    # Resize if target size specified
    if target_size is not None:
        image = image.resize(target_size, Image.LANCZOS)

    # Convert to numpy array
    img_array = np.array(image)

    # Ensure RGB (3 channels)
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # Normalize if requested
    if normalize:
        img_array = img_array.astype(np.float32) / 255.0

    return img_array


def extract_first_frame(video_path: Union[str, Path]) -> np.ndarray:
    """
    Extract first frame from video.

    Args:
        video_path: Path to video file

    Returns:
        First frame as numpy array (H, W, C)
    """
    with VideoReader(video_path) as reader:
        return reader.get_frame(0)


def align_image_to_video(
    character_image: Union[str, Path, Image.Image, np.ndarray],
    reference_video: Union[str, Path],
    method: str = "resize"
) -> np.ndarray:
    """
    Align character image to reference video dimensions.

    Args:
        character_image: Character image
        reference_video: Reference video
        method: Alignment method ('resize', 'pad', 'crop')

    Returns:
        Aligned image as numpy array
    """
    # Load image
    if isinstance(character_image, (str, Path)):
        img = load_image(character_image)
    elif isinstance(character_image, np.ndarray):
        img = Image.fromarray(character_image)
    else:
        img = character_image

    # Get video dimensions
    video_info = get_video_info(reference_video)
    target_width = video_info['width']
    target_height = video_info['height']

    if method == "resize":
        # Simple resize
        img = img.resize((target_width, target_height), Image.LANCZOS)

    elif method == "pad":
        # Resize maintaining aspect ratio, then pad
        img.thumbnail((target_width, target_height), Image.LANCZOS)
        new_img = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        paste_x = (target_width - img.width) // 2
        paste_y = (target_height - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        img = new_img

    elif method == "crop":
        # Resize to cover target, then center crop
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            # Image is wider, fit height
            new_height = target_height
            new_width = int(target_height * img_ratio)
        else:
            # Image is taller, fit width
            new_width = target_width
            new_height = int(target_width / img_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)

        # Center crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        img = img.crop((left, top, left + target_width, top + target_height))

    else:
        raise ValueError(f"Unknown method: {method}. Use 'resize', 'pad', or 'crop'")

    return np.array(img)


# Convenience functions
def validate_inputs(character_image: Union[str, Path], reference_video: Optional[Union[str, Path]] = None) -> bool:
    """
    Validate inputs and print results.

    Args:
        character_image: Path to character image
        reference_video: Path to reference video (optional)

    Returns:
        True if valid, False otherwise
    """
    validator = InputValidator()
    is_valid, message = validator.validate_inputs(character_image, reference_video)
    print(message)
    return is_valid
