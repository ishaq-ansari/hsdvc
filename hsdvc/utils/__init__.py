"""
Utility functions for HSDVC.
"""

import torch
import numpy as np
from typing import Optional
import random


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.)
               If None, automatically selects available device.
    
    Returns:
        torch.device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def format_time(seconds: float) -> str:
    """Format seconds to human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def save_video(
    video: torch.Tensor,
    path: str,
    fps: int = 8,
    codec: str = "mp4v"
):
    """
    Save video tensor to file.
    
    Args:
        video: [T, C, H, W] video tensor
        path: Output path
        fps: Frame rate
        codec: Video codec
    """
    import cv2
    
    video_np = (video.cpu().numpy() * 255).astype(np.uint8)
    video_np = video_np.transpose(0, 2, 3, 1)  # [T, H, W, C]
    
    T, H, W, C = video_np.shape
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    
    for frame in video_np:
        if C == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        writer.write(frame_bgr)
    
    writer.release()


def load_video(path: str, max_frames: Optional[int] = None) -> torch.Tensor:
    """
    Load video from file.
    
    Args:
        path: Video file path
        max_frames: Maximum frames to load
        
    Returns:
        Video tensor [T, C, H, W]
    """
    import cv2
    
    cap = cv2.VideoCapture(path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    
    return torch.stack(frames)
