"""
HSDVC: Hybrid Structured-Diffusion Video Compiler
A novel video generation and manipulation system combining structured motion learning with diffusion models.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .models import (
    MotionExtractor,
    IdentityEncoder,
    VideoCompiler,
    CharacterReplacer,
)

__all__ = [
    "MotionExtractor",
    "IdentityEncoder", 
    "VideoCompiler",
    "CharacterReplacer",
]
