"""
Video utilities for motion control pipeline.
Handles video I/O, frame extraction, and basic video manipulations.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm


class VideoReader:
    """Read video files and extract frames."""

    def __init__(self, video_path: Union[str, Path]):
        """
        Initialize video reader.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

    def read_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> List[np.ndarray]:
        """
        Read frames from video.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (exclusive). If None, read to end.
            step: Frame step (1 = every frame, 2 = every other frame, etc.)

        Returns:
            List of frames as numpy arrays (H, W, C) in RGB format
        """
        if end_frame is None:
            end_frame = self.frame_count

        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i in range(start_frame, end_frame, step):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            # Skip frames if step > 1
            for _ in range(step - 1):
                self.cap.read()

        return frames

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Get specific frame from video.

        Args:
            frame_idx: Frame index

        Returns:
            Frame as numpy array (H, W, C) in RGB format
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx}")

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __del__(self):
        """Release video capture."""
        if hasattr(self, 'cap'):
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()


class VideoWriter:
    """Write frames to video file."""

    def __init__(
        self,
        output_path: Union[str, Path],
        fps: float = 24,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
        pix_fmt: str = "yuv420p"
    ):
        """
        Initialize video writer.

        Args:
            output_path: Output video path
            fps: Frames per second
            codec: Video codec (libx264, libx265, etc.)
            crf: Constant Rate Factor (0-51, 18 is visually lossless)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
            pix_fmt: Pixel format
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.pix_fmt = pix_fmt
        self.frames = []

    def add_frame(self, frame: Union[np.ndarray, Image.Image]):
        """
        Add frame to video.

        Args:
            frame: Frame as numpy array or PIL Image
        """
        if isinstance(frame, Image.Image):
            frame = np.array(frame)

        # Ensure RGB format
        if frame.ndim == 2:  # Grayscale
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[2] == 4:  # RGBA
            frame = frame[:, :, :3]

        self.frames.append(frame)

    def write(self):
        """Write all frames to video file using imageio."""
        if not self.frames:
            raise ValueError("No frames to write")

        print(f"Writing {len(self.frames)} frames to {self.output_path}...")

        imageio.mimsave(
            self.output_path,
            self.frames,
            fps=self.fps,
            codec=self.codec,
            quality=10 - (self.crf // 5),  # Convert CRF to quality (rough approximation)
            pixelformat=self.pix_fmt,
            ffmpeg_params=["-preset", self.preset]
        )

        print(f"✓ Video saved: {self.output_path}")

    def write_with_ffmpeg(self, ffmpeg_path: str = "ffmpeg"):
        """Write frames using ffmpeg (more control over encoding)."""
        if not self.frames:
            raise ValueError("No frames to write")

        height, width = self.frames[0].shape[:2]

        # Create ffmpeg process
        cmd = [
            ffmpeg_path,
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",  # Input from pipe
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", self.pix_fmt,
            str(self.output_path)
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Write frames
        for frame in tqdm(self.frames, desc="Writing video"):
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {process.stderr.read().decode()}")

        print(f"✓ Video saved: {self.output_path}")


def save_frames_as_video(
    frames: List[Union[np.ndarray, Image.Image]],
    output_path: Union[str, Path],
    fps: float = 24,
    **kwargs
) -> Path:
    """
    Save list of frames as video file.

    Args:
        frames: List of frames (numpy arrays or PIL Images)
        output_path: Output video path
        fps: Frames per second
        **kwargs: Additional arguments for VideoWriter

    Returns:
        Path to output video
    """
    writer = VideoWriter(output_path, fps=fps, **kwargs)
    for frame in frames:
        writer.add_frame(frame)
    writer.write()
    return Path(output_path)


def load_image(image_path: Union[str, Path], mode: str = "RGB") -> Image.Image:
    """
    Load image from file.

    Args:
        image_path: Path to image file
        mode: Color mode (RGB, RGBA, L, etc.)

    Returns:
        PIL Image
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    if mode:
        img = img.convert(mode)
    return img


def save_image(image: Union[np.ndarray, Image.Image], output_path: Union[str, Path]):
    """
    Save image to file.

    Args:
        image: Image as numpy array or PIL Image
        output_path: Output path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image.save(output_path)


def resize_image(
    image: Union[np.ndarray, Image.Image],
    size: Tuple[int, int],
    interpolation: str = "bilinear"
) -> Union[np.ndarray, Image.Image]:
    """
    Resize image.

    Args:
        image: Input image
        size: Target size (width, height)
        interpolation: Interpolation method (bilinear, bicubic, lanczos, nearest)

    Returns:
        Resized image (same type as input)
    """
    is_numpy = isinstance(image, np.ndarray)

    if is_numpy:
        image = Image.fromarray(image)

    # Map interpolation methods
    interp_map = {
        "bilinear": Image.BILINEAR,
        "bicubic": Image.BICUBIC,
        "lanczos": Image.LANCZOS,
        "nearest": Image.NEAREST
    }

    resized = image.resize(size, interp_map.get(interpolation, Image.BILINEAR))

    if is_numpy:
        return np.array(resized)
    return resized


def get_video_info(video_path: Union[str, Path]) -> dict:
    """
    Get video file information.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata
    """
    with VideoReader(video_path) as reader:
        return {
            "fps": reader.fps,
            "frame_count": reader.frame_count,
            "width": reader.width,
            "height": reader.height,
            "duration": reader.duration,
            "resolution": f"{reader.width}x{reader.height}"
        }


def extract_frames_to_folder(
    video_path: Union[str, Path],
    output_folder: Union[str, Path],
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    prefix: str = "frame"
) -> List[Path]:
    """
    Extract video frames to folder.

    Args:
        video_path: Input video path
        output_folder: Output folder for frames
        start_frame: Starting frame index
        end_frame: Ending frame index (exclusive)
        prefix: Filename prefix for frames

    Returns:
        List of saved frame paths
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    frame_paths = []

    with VideoReader(video_path) as reader:
        frames = reader.read_frames(start_frame, end_frame)

        for i, frame in enumerate(tqdm(frames, desc="Extracting frames")):
            frame_idx = start_frame + i
            frame_path = output_folder / f"{prefix}_{frame_idx:06d}.png"
            save_image(frame, frame_path)
            frame_paths.append(frame_path)

    return frame_paths
