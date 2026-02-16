"""
Audio handling utilities for motion control video generation.
Extracts audio from reference videos and merges with generated videos.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np


class AudioHandler:
    """Handles audio extraction and merging for video generation pipeline."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Initialize AudioHandler.

        Args:
            ffmpeg_path: Path to ffmpeg executable
        """
        self.ffmpeg_path = ffmpeg_path
        self._verify_ffmpeg()

    def _verify_ffmpeg(self):
        """Verify that ffmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f"ffmpeg not found at '{self.ffmpeg_path}'. "
                "Please install ffmpeg or specify correct path. "
                "On HPC: module load ffmpeg"
            ) from e

    def extract_audio(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        audio_codec: str = "aac",
        audio_bitrate: str = "192k"
    ) -> Path:
        """
        Extract audio from video file.

        Args:
            video_path: Path to input video file
            output_path: Path for output audio file. If None, creates temp file.
            audio_codec: Audio codec (aac, mp3, wav, etc.)
            audio_bitrate: Audio bitrate (e.g., '192k', '320k')

        Returns:
            Path to extracted audio file

        Raises:
            FileNotFoundError: If input video doesn't exist
            RuntimeError: If audio extraction fails
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Determine output path
        if output_path is None:
            ext = "m4a" if audio_codec == "aac" else audio_codec
            output_path = video_path.parent / f"{video_path.stem}_audio.{ext}"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", audio_codec,
            "-b:a", audio_bitrate,
            "-y",  # Overwrite output
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Audio extracted to: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Audio extraction failed: {e.stderr}"
            ) from e

    def get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """
        Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        cmd = [
            self.ffmpeg_path,
            "-i", str(audio_path),
            "-f", "null",
            "-"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            # Parse duration from ffmpeg output
            for line in result.stderr.split('\n'):
                if 'Duration:' in line:
                    time_str = line.split('Duration:')[1].split(',')[0].strip()
                    h, m, s = time_str.split(':')
                    duration = float(h) * 3600 + float(m) * 60 + float(s)
                    return duration

            raise RuntimeError("Could not parse audio duration")
        except Exception as e:
            raise RuntimeError(f"Failed to get audio duration: {e}") from e

    def trim_audio(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        start_time: float = 0.0,
        duration: Optional[float] = None
    ) -> Path:
        """
        Trim audio to specified duration.

        Args:
            audio_path: Path to input audio file
            output_path: Path for trimmed audio output
            start_time: Start time in seconds
            duration: Duration in seconds. If None, trim to end.

        Returns:
            Path to trimmed audio file
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-i", str(audio_path),
            "-ss", str(start_time),
        ]

        if duration is not None:
            cmd.extend(["-t", str(duration)])

        cmd.extend([
            "-c", "copy",  # Copy codec without re-encoding
            "-y",
            str(output_path)
        ])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Audio trimming failed: {e.stderr}") from e

    def merge_audio_video(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 18,  # Quality (lower = better, 18 is visually lossless)
        preset: str = "medium"
    ) -> Path:
        """
        Merge audio with video file.

        Args:
            video_path: Path to input video file (can be without audio)
            audio_path: Path to audio file to merge
            output_path: Path for output video with audio
            video_codec: Video codec (libx264, libx265, etc.)
            audio_codec: Audio codec (aac, mp3, etc.)
            crf: Constant Rate Factor for video quality (0-51, 18 recommended)
            preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)

        Returns:
            Path to output video with merged audio

        Raises:
            FileNotFoundError: If input files don't exist
            RuntimeError: If merging fails
        """
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        cmd = [
            self.ffmpeg_path,
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", video_codec,
            "-crf", str(crf),
            "-preset", preset,
            "-c:a", audio_codec,
            "-shortest",  # Match shortest stream (video or audio)
            "-y",
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Audio+video merged to: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Audio/video merge failed: {e.stderr}"
            ) from e

    def replace_audio(
        self,
        video_path: Union[str, Path],
        new_audio_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Replace audio track in video file.

        Args:
            video_path: Path to input video
            new_audio_path: Path to new audio file
            output_path: Path for output video

        Returns:
            Path to output video with replaced audio
        """
        return self.merge_audio_video(
            video_path=video_path,
            audio_path=new_audio_path,
            output_path=output_path,
            video_codec="copy"  # Copy video without re-encoding
        )


# Convenience functions
def extract_audio(video_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Path:
    """Extract audio from video. Convenience wrapper."""
    handler = AudioHandler()
    return handler.extract_audio(video_path, output_path)


def merge_audio_video(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path]
) -> Path:
    """Merge audio with video. Convenience wrapper."""
    handler = AudioHandler()
    return handler.merge_audio_video(video_path, audio_path, output_path)
