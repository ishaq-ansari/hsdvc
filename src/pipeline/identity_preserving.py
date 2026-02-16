"""
Identity preservation utilities using LoRA fine-tuning and CodeFormer face restoration.
Ensures generated character maintains consistent appearance across frames.
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add CodeFormer path
CODEFORMER_PATH = Path(__file__).parent.parent.parent / "models" / "CodeFormer"
if CODEFORMER_PATH.exists():
    sys.path.insert(0, str(CODEFORMER_PATH))


class IdentityPreserver:
    """Preserve character identity across generated video frames."""

    def __init__(
        self,
        method: str = "codeformer",
        device: str = "cuda",
        upscale_factor: float = 2.0,
        face_enhance_strength: float = 0.5
    ):
        """
        Initialize identity preserver.

        Args:
            method: Preservation method ('codeformer', 'gfpgan', 'both')
            device: Device for inference
            upscale_factor: Upscale factor for face region
            face_enhance_strength: Strength of face restoration (0-1)
        """
        self.method = method
        self.device = device
        self.upscale_factor = upscale_factor
        self.face_enhance_strength = face_enhance_strength

        self.restorer = None
        self.face_detector = None

        if method in ["codeformer", "both"]:
            self._load_codeformer()

    def _load_codeformer(self):
        """Load CodeFormer face restoration model."""
        try:
            # Import CodeFormer modules
            from basicsr.utils import img2tensor, tensor2img
            from facelib.utils.face_restoration_helper import FaceRestoreHelper

            # Create face restoration helper
            self.face_helper = FaceRestoreHelper(
                upscale_factor=self.upscale_factor,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=self.device
            )

            # Load CodeFormer model
            try:
                from basicsr.archs.codeformer_arch import CodeFormer

                model_path = CODEFORMER_PATH / "weights" / "CodeFormer" / "codeformer.pth"

                self.restorer = CodeFormer(
                    dim_embd=512,
                    codebook_size=1024,
                    n_head=8,
                    n_layers=9,
                    connect_list=['32', '64', '128', '256']
                ).to(self.device)

                checkpoint = torch.load(model_path, map_location=self.device)
                self.restorer.load_state_dict(checkpoint['params_ema'])
                self.restorer.eval()

                print("✓ CodeFormer loaded successfully")

            except Exception as e:
                print(f"Warning: Could not load CodeFormer model: {e}")
                print("Continuing without face restoration...")
                self.restorer = None

        except ImportError as e:
            print(f"Warning: CodeFormer not available: {e}")
            print("Install with: cd models && git clone https://github.com/sczhou/CodeFormer.git")
            self.restorer = None

    def restore_face(
        self,
        frame: np.ndarray,
        reference_face: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Restore face in a single frame.

        Args:
            frame: Input frame (H, W, C) as numpy array
            reference_face: Optional reference face for identity matching

        Returns:
            Frame with restored face
        """
        if self.restorer is None:
            return frame

        try:
            # Detect and align faces
            self.face_helper.clean_all()
            self.face_helper.read_image(frame)
            self.face_helper.get_face_landmarks_5(only_center_face=False)
            self.face_helper.align_warp_face()

            # No faces detected
            if len(self.face_helper.cropped_faces) == 0:
                return frame

            # Restore each detected face
            for cropped_face in self.face_helper.cropped_faces:
                # Prepare input
                cropped_face_t = torch.from_numpy(cropped_face).permute(2, 0, 1).unsqueeze(0).to(self.device)
                cropped_face_t = cropped_face_t / 255.0

                # Restore face
                with torch.no_grad():
                    output = self.restorer(
                        cropped_face_t,
                        w=self.face_enhance_strength,
                        adain=True
                    )[0]

                # Convert back to numpy
                restored_face = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored_face = np.clip(restored_face * 255, 0, 255).astype(np.uint8)

                self.face_helper.add_restored_face(restored_face)

            # Paste faces back
            self.face_helper.get_inverse_affine(None)
            restored_img = self.face_helper.paste_faces_to_input_image()

            return restored_img

        except Exception as e:
            print(f"Warning: Face restoration failed for frame: {e}")
            return frame

    def restore_faces_in_video(
        self,
        frames: List[np.ndarray],
        reference_face: Optional[np.ndarray] = None,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Restore faces in all video frames.

        Args:
            frames: List of input frames
            reference_face: Optional reference face image
            show_progress: Show progress bar

        Returns:
            List of frames with restored faces
        """
        restored = []

        iterator = tqdm(frames, desc="Restoring faces") if show_progress else frames

        for frame in iterator:
            restored_frame = self.restore_face(frame, reference_face)
            restored.append(restored_frame)

        return restored


class LoRATrainer:
    """Train LoRA adapters for character-specific fine-tuning."""

    def __init__(
        self,
        base_model,
        device: str = "cuda",
        rank: int = 8,
        alpha: int = 16
    ):
        """
        Initialize LoRA trainer.

        Args:
            base_model: Base diffusion model
            device: Device for training
            rank: LoRA rank (lower = less parameters)
            alpha: LoRA alpha (scaling factor)
        """
        self.base_model = base_model
        self.device = device
        self.rank = rank
        self.alpha = alpha

    def train_lora(
        self,
        character_images: List[Union[str, Path, Image.Image]],
        num_steps: int = 500,
        learning_rate: float = 1e-4,
        batch_size: int = 1
    ):
        """
        Train LoRA adapter on character images.

        Args:
            character_images: List of character images for training
            num_steps: Training steps
            learning_rate: Learning rate
            batch_size: Batch size

        Returns:
            Trained LoRA weights
        """
        print("TODO: Implement LoRA training")
        print("This requires:")
        print("1. PEFT library for LoRA integration")
        print("2. Training loop with character images")
        print("3. Proper loss function for identity preservation")

        # Placeholder - actual implementation would be more complex
        return None


def preserve_identity(
    frames: List[np.ndarray],
    reference_image: Union[str, Path, np.ndarray],
    method: str = "codeformer",
    device: str = "cuda",
    **kwargs
) -> List[np.ndarray]:
    """
    Preserve character identity across video frames.

    Args:
        frames: List of generated frames
        reference_image: Reference character image
        method: Preservation method
        device: Device for processing
        **kwargs: Additional arguments for IdentityPreserver

    Returns:
        Frames with preserved identity
    """
    # Load reference
    if isinstance(reference_image, (str, Path)):
        from utils.video_utils import load_image
        reference_image = np.array(load_image(reference_image))

    # Initialize preserver
    preserver = IdentityPreserver(method=method, device=device, **kwargs)

    # Restore faces
    restored_frames = preserver.restore_faces_in_video(
        frames,
        reference_face=reference_image
    )

    return restored_frames
