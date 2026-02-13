"""
Motion extraction module: extracts structured motion information from videos.
Includes pose, depth, flow, camera, contacts, and motion ODE parameters.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2

from ..config import MotionExtractionConfig


@dataclass
class MotionData:
    """Container for all extracted motion information."""
    # Per-frame data
    poses_3d: torch.Tensor  # [T, N, 3] - 3D keypoints
    poses_2d: torch.Tensor  # [T, N, 2] - 2D keypoints
    pose_confidence: torch.Tensor  # [T, N] - confidence scores
    
    depth_maps: torch.Tensor  # [T, H, W] - depth maps
    optical_flow: torch.Tensor  # [T-1, H, W, 2] - optical flow
    
    # Camera parameters
    camera_intrinsics: torch.Tensor  # [3, 3] - camera matrix
    camera_extrinsics: torch.Tensor  # [T, 4, 4] - camera poses
    camera_trajectory: torch.Tensor  # [T, 6] - [translation, rotation]
    
    # Contact and physics
    contact_labels: torch.Tensor  # [T, N] - binary contact labels
    contact_confidence: torch.Tensor  # [T, N] - contact confidence
    
    # Motion ODE parameters
    motion_ode_params: Optional[torch.Tensor] = None  # [T, D] - learned dynamics
    motion_basis: Optional[torch.Tensor] = None  # [K, D] - temporal basis
    motion_coefficients: Optional[torch.Tensor] = None  # [T, K] - basis coefficients
    
    # Metadata
    frame_rate: float = 30.0
    resolution: Tuple[int, int] = (720, 1280)
    num_frames: int = 0


class PoseEstimator(nn.Module):
    """
    3D pose estimation from video frames.
    Supports multiple backends: MediaPipe, ViTPose, etc.
    """
    
    def __init__(self, config: MotionExtractionConfig):
        super().__init__()
        self.config = config
        self.pose_dim = config.pose_dim
        
        if config.pose_model == "mediapipe":
            self._init_mediapipe()
        elif config.pose_model == "vitpose":
            self._init_vitpose()
        else:
            raise ValueError(f"Unknown pose model: {config.pose_model}")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Pose."""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.backend = "mediapipe"
        except ImportError:
            raise ImportError("MediaPipe not installed. Install with: pip install mediapipe")
    
    def _init_vitpose(self):
        """Initialize ViTPose (more accurate, but requires model download)."""
        from timm import create_model
        self.pose_detector = create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=self.pose_dim * 3  # x, y, confidence
        )
        self.backend = "vitpose"
    
    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract poses from video frames.
        
        Args:
            frames: [B, T, C, H, W] video frames
            
        Returns:
            Dictionary with pose data
        """
        B, T, C, H, W = frames.shape
        
        poses_2d = []
        poses_3d = []
        confidences = []
        
        for b in range(B):
            batch_poses_2d = []
            batch_poses_3d = []
            batch_confidences = []
            
            for t in range(T):
                frame = frames[b, t].permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(np.uint8)
                
                if self.backend == "mediapipe":
                    pose_2d, pose_3d, conf = self._extract_mediapipe(frame)
                else:
                    pose_2d, pose_3d, conf = self._extract_vitpose(frame)
                
                batch_poses_2d.append(pose_2d)
                batch_poses_3d.append(pose_3d)
                batch_confidences.append(conf)
            
            poses_2d.append(torch.stack(batch_poses_2d))
            poses_3d.append(torch.stack(batch_poses_3d))
            confidences.append(torch.stack(batch_confidences))
        
        return {
            "poses_2d": torch.stack(poses_2d),  # [B, T, N, 2]
            "poses_3d": torch.stack(poses_3d),  # [B, T, N, 3]
            "confidence": torch.stack(confidences),  # [B, T, N]
        }
    
    def _extract_mediapipe(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract pose using MediaPipe."""
        results = self.pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            pose_2d = torch.tensor([
                [lm.x * frame.shape[1], lm.y * frame.shape[0]]
                for lm in landmarks
            ])
            
            pose_3d = torch.tensor([
                [lm.x, lm.y, lm.z]
                for lm in landmarks
            ])
            
            confidence = torch.tensor([lm.visibility for lm in landmarks])
        else:
            # Return zeros if no pose detected
            pose_2d = torch.zeros(self.pose_dim, 2)
            pose_3d = torch.zeros(self.pose_dim, 3)
            confidence = torch.zeros(self.pose_dim)
        
        return pose_2d, pose_3d, confidence
    
    def _extract_vitpose(self, frame: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract pose using ViTPose."""
        # Preprocess frame
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # Run model
        with torch.no_grad():
            output = self.pose_detector(frame_tensor)
        
        # Parse output (simplified - actual ViTPose has more complex output format)
        output = output.reshape(self.pose_dim, 3)
        pose_2d = output[:, :2]
        confidence = output[:, 2]
        pose_3d = torch.cat([pose_2d, torch.zeros(self.pose_dim, 1)], dim=-1)
        
        return pose_2d, pose_3d, confidence


class DepthEstimator(nn.Module):
    """
    Monocular depth estimation.
    Supports Depth-Anything V2, MiDaS, etc.
    """
    
    def __init__(self, config: MotionExtractionConfig):
        super().__init__()
        self.config = config
        self.resolution = config.depth_resolution
        
        if config.depth_model == "depth_anything_v2":
            self._init_depth_anything()
        elif config.depth_model == "midas":
            self._init_midas()
        else:
            raise ValueError(f"Unknown depth model: {config.depth_model}")
    
    def _init_depth_anything(self):
        """Initialize Depth-Anything V2 (state-of-the-art)."""
        try:
            # Depth-Anything V2 from HuggingFace
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            self.processor = AutoImageProcessor.from_pretrained(
                "depth-anything/Depth-Anything-V2-Large"
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Large"
            )
            self.backend = "depth_anything"
        except Exception as e:
            print(f"Failed to load Depth-Anything V2: {e}")
            print("Falling back to MiDaS...")
            self._init_midas()
    
    def _init_midas(self):
        """Initialize MiDaS depth estimator."""
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        self.backend = "midas"
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Estimate depth from video frames.
        
        Args:
            frames: [B, T, C, H, W] video frames
            
        Returns:
            depth_maps: [B, T, H, W] depth maps
        """
        B, T, C, H, W = frames.shape
        
        depth_maps = []
        
        for b in range(B):
            batch_depths = []
            
            for t in range(T):
                frame = frames[b, t]
                
                if self.backend == "depth_anything":
                    depth = self._estimate_depth_anything(frame)
                else:
                    depth = self._estimate_midas(frame)
                
                batch_depths.append(depth)
            
            depth_maps.append(torch.stack(batch_depths))
        
        return torch.stack(depth_maps)
    
    def _estimate_depth_anything(self, frame: torch.Tensor) -> torch.Tensor:
        """Estimate depth using Depth-Anything."""
        # Convert to PIL
        from PIL import Image
        frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(frame_np)
        
        # Process
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            depth = outputs.predicted_depth
        
        # Resize to target resolution
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=self.resolution,
            mode="bilinear",
            align_corners=False
        ).squeeze(1)
        
        return depth.squeeze(0)
    
    def _estimate_midas(self, frame: torch.Tensor) -> torch.Tensor:
        """Estimate depth using MiDaS."""
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        input_batch = self.transform(frame_np)
        
        with torch.no_grad():
            depth = self.model(input_batch)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=self.resolution,
                mode="bicubic",
                align_corners=False
            ).squeeze(1)
        
        return depth.squeeze(0)


class OpticalFlowEstimator(nn.Module):
    """
    Optical flow estimation.
    Supports RAFT, GMFlow, etc.
    """
    
    def __init__(self, config: MotionExtractionConfig):
        super().__init__()
        self.config = config
        self.resolution = config.flow_resolution
        
        if config.flow_model == "raft":
            self._init_raft()
        else:
            raise ValueError(f"Unknown flow model: {config.flow_model}")
    
    def _init_raft(self):
        """Initialize RAFT optical flow."""
        # Use RAFT from torchvision (PyTorch 2.0+)
        from torchvision.models.optical_flow import raft_large
        self.model = raft_large(pretrained=True, progress=False)
        self.model.eval()
        self.backend = "raft"
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Estimate optical flow between consecutive frames.
        
        Args:
            frames: [B, T, C, H, W] video frames
            
        Returns:
            flows: [B, T-1, H, W, 2] optical flow
        """
        B, T, C, H, W = frames.shape
        
        flows = []
        
        for b in range(B):
            batch_flows = []
            
            for t in range(T - 1):
                frame1 = frames[b, t]
                frame2 = frames[b, t + 1]
                
                flow = self._estimate_raft(frame1, frame2)
                batch_flows.append(flow)
            
            flows.append(torch.stack(batch_flows))
        
        return torch.stack(flows)
    
    def _estimate_raft(self, frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Estimate flow using RAFT."""
        # Resize to flow resolution
        frame1_resized = torch.nn.functional.interpolate(
            frame1.unsqueeze(0),
            size=self.resolution,
            mode="bilinear",
            align_corners=False
        )
        frame2_resized = torch.nn.functional.interpolate(
            frame2.unsqueeze(0),
            size=self.resolution,
            mode="bilinear",
            align_corners=False
        )
        
        # Normalize to [0, 255]
        frame1_resized = frame1_resized * 255.0
        frame2_resized = frame2_resized * 255.0
        
        with torch.no_grad():
            flow_predictions = self.model(frame1_resized, frame2_resized)
            flow = flow_predictions[-1]  # Use final prediction
        
        # flow shape: [1, 2, H, W] -> [H, W, 2]
        flow = flow.squeeze(0).permute(1, 2, 0)
        
        return flow


class CameraEstimator(nn.Module):
    """
    Camera parameter estimation.
    Estimates intrinsics and extrinsics from video.
    """
    
    def __init__(self, config: MotionExtractionConfig):
        super().__init__()
        self.config = config
    
    def forward(
        self,
        frames: torch.Tensor,
        depth_maps: torch.Tensor,
        optical_flow: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate camera parameters.
        
        Args:
            frames: [B, T, C, H, W]
            depth_maps: [B, T, H, W]
            optical_flow: [B, T-1, H, W, 2]
            
        Returns:
            Dictionary with camera parameters
        """
        B, T, C, H, W = frames.shape
        
        # Estimate intrinsics (assume fixed for now)
        intrinsics = self._estimate_intrinsics(H, W)
        
        # Estimate extrinsics using flow and depth
        extrinsics = self._estimate_extrinsics(depth_maps, optical_flow, intrinsics)
        
        # Extract camera trajectory
        trajectory = self._extract_trajectory(extrinsics)
        
        return {
            "intrinsics": intrinsics.repeat(B, 1, 1),  # [B, 3, 3]
            "extrinsics": extrinsics,  # [B, T, 4, 4]
            "trajectory": trajectory,  # [B, T, 6]
        }
    
    def _estimate_intrinsics(self, H: int, W: int) -> torch.Tensor:
        """Estimate camera intrinsics (simplified - assume standard pinhole)."""
        focal_length = max(H, W) * 1.2  # Typical for standard cameras
        
        intrinsics = torch.tensor([
            [focal_length, 0, W / 2],
            [0, focal_length, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        return intrinsics
    
    def _estimate_extrinsics(
        self,
        depth_maps: torch.Tensor,
        optical_flow: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> torch.Tensor:
        """Estimate camera extrinsics from depth and flow."""
        B, T, H, W = depth_maps.shape
        
        # Initialize extrinsics (identity for first frame)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
        
        # Estimate relative poses from flow
        for t in range(1, T):
            # Simplified: use flow magnitude as proxy for camera motion
            flow_t = optical_flow[:, t - 1]  # [B, H, W, 2]
            
            # Estimate translation from average flow
            translation = flow_t.mean(dim=[1, 2]) * 0.01  # Scale factor
            
            # Update extrinsics
            extrinsics[:, t, :3, 3] = extrinsics[:, t - 1, :3, 3] + translation
        
        return extrinsics
    
    def _extract_trajectory(self, extrinsics: torch.Tensor) -> torch.Tensor:
        """Extract 6-DOF trajectory from extrinsics."""
        B, T = extrinsics.shape[:2]
        
        trajectory = torch.zeros(B, T, 6)
        
        for t in range(T):
            # Translation
            trajectory[:, t, :3] = extrinsics[:, t, :3, 3]
            
            # Rotation (convert rotation matrix to euler angles - simplified)
            R = extrinsics[:, t, :3, :3]
            # Simplified: extract approximate euler angles
            trajectory[:, t, 3:] = torch.zeros(B, 3)  # Placeholder
        
        return trajectory


class ContactDetector(nn.Module):
    """
    Detect contact events between body and environment.
    """
    
    def __init__(self, config: MotionExtractionConfig):
        super().__init__()
        self.config = config
        self.threshold = config.contact_threshold
    
    def forward(
        self,
        poses_3d: torch.Tensor,
        depth_maps: torch.Tensor,
        optical_flow: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Detect contact points.
        
        Args:
            poses_3d: [B, T, N, 3]
            depth_maps: [B, T, H, W]
            optical_flow: [B, T-1, H, W, 2]
            
        Returns:
            Dictionary with contact labels and confidence
        """
        B, T, N, _ = poses_3d.shape
        
        # Detect contacts based on velocity and depth
        velocities = torch.diff(poses_3d, dim=1)  # [B, T-1, N, 3]
        velocities = torch.cat([velocities, velocities[:, -1:]], dim=1)  # [B, T, N, 3]
        
        velocity_magnitude = torch.norm(velocities, dim=-1)  # [B, T, N]
        
        # Contact when velocity is low
        contact_labels = (velocity_magnitude < self.threshold).float()
        
        # Confidence based on velocity stability
        contact_confidence = 1.0 - torch.sigmoid(velocity_magnitude * 10)
        
        return {
            "labels": contact_labels,  # [B, T, N]
            "confidence": contact_confidence,  # [B, T, N]
        }


class MotionODEEncoder(nn.Module):
    """
    Encode motion as ODE parameters for smooth interpolation.
    """
    
    def __init__(self, config: MotionExtractionConfig):
        super().__init__()
        self.config = config
        self.ode_dim = config.ode_dim
        
        # Temporal basis learning
        self.basis_encoder = nn.Sequential(
            nn.Linear(config.pose_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, self.ode_dim)
        )
    
    def forward(self, poses_3d: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode motion as ODE parameters.
        
        Args:
            poses_3d: [B, T, N, 3]
            
        Returns:
            Dictionary with ODE parameters and basis
        """
        B, T, N, _ = poses_3d.shape
        
        # Flatten pose to vector
        pose_flat = poses_3d.reshape(B, T, -1)  # [B, T, N*3]
        
        # Encode to ODE space
        ode_params = self.basis_encoder(pose_flat)  # [B, T, D]
        
        # Learn temporal basis (PCA-like)
        motion_basis = self._compute_basis(ode_params)
        motion_coefficients = self._compute_coefficients(ode_params, motion_basis)
        
        return {
            "params": ode_params,  # [B, T, D]
            "basis": motion_basis,  # [B, K, D]
            "coefficients": motion_coefficients,  # [B, T, K]
        }
    
    def _compute_basis(self, ode_params: torch.Tensor, num_basis: int = 16) -> torch.Tensor:
        """Compute temporal basis using SVD."""
        B, T, D = ode_params.shape
        
        # SVD per batch
        basis_list = []
        for b in range(B):
            U, S, V = torch.svd(ode_params[b])  # [T, D]
            basis = V[:, :num_basis].T  # [K, D]
            basis_list.append(basis)
        
        return torch.stack(basis_list)  # [B, K, D]
    
    def _compute_coefficients(
        self,
        ode_params: torch.Tensor,
        basis: torch.Tensor
    ) -> torch.Tensor:
        """Compute coefficients by projecting onto basis."""
        # ode_params: [B, T, D]
        # basis: [B, K, D]
        
        # Project: coeffs = params @ basis.T
        coefficients = torch.bmm(
            ode_params,  # [B, T, D]
            basis.transpose(1, 2)  # [B, D, K]
        )  # [B, T, K]
        
        return coefficients


class MotionExtractor(nn.Module):
    """
    Main motion extraction module that combines all sub-modules.
    """
    
    def __init__(self, config: Optional[MotionExtractionConfig] = None):
        super().__init__()
        
        if config is None:
            config = MotionExtractionConfig()
        
        self.config = config
        
        # Initialize sub-modules
        self.pose_estimator = PoseEstimator(config)
        self.depth_estimator = DepthEstimator(config)
        self.flow_estimator = OpticalFlowEstimator(config)
        self.camera_estimator = CameraEstimator(config)
        
        if config.detect_contacts:
            self.contact_detector = ContactDetector(config)
        
        if config.use_motion_ode:
            self.motion_ode_encoder = MotionODEEncoder(config)
    
    def extract(self, video_path: str) -> MotionData:
        """
        Extract motion data from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            MotionData object with all extracted information
        """
        # Load video
        frames = self._load_video(video_path)
        
        return self.forward(frames)
    
    def forward(self, frames: torch.Tensor) -> MotionData:
        """
        Extract motion data from video frames.
        
        Args:
            frames: [B, T, C, H, W] video frames (normalized to [0, 1])
            
        Returns:
            MotionData object
        """
        B, T, C, H, W = frames.shape
        
        # 1. Pose estimation
        pose_data = self.pose_estimator(frames)
        
        # 2. Depth estimation
        depth_maps = self.depth_estimator(frames)
        
        # 3. Optical flow
        optical_flow = self.flow_estimator(frames)
        
        # 4. Camera estimation
        camera_data = self.camera_estimator(frames, depth_maps, optical_flow)
        
        # 5. Contact detection (optional)
        if self.config.detect_contacts:
            contact_data = self.contact_detector(
                pose_data["poses_3d"],
                depth_maps,
                optical_flow
            )
        else:
            contact_data = {
                "labels": torch.zeros(B, T, self.config.pose_dim),
                "confidence": torch.zeros(B, T, self.config.pose_dim),
            }
        
        # 6. Motion ODE encoding (optional)
        if self.config.use_motion_ode:
            ode_data = self.motion_ode_encoder(pose_data["poses_3d"])
        else:
            ode_data = {
                "params": None,
                "basis": None,
                "coefficients": None,
            }
        
        # Combine all data
        motion_data = MotionData(
            poses_3d=pose_data["poses_3d"][0],  # Remove batch dim
            poses_2d=pose_data["poses_2d"][0],
            pose_confidence=pose_data["confidence"][0],
            depth_maps=depth_maps[0],
            optical_flow=optical_flow[0],
            camera_intrinsics=camera_data["intrinsics"][0],
            camera_extrinsics=camera_data["extrinsics"][0],
            camera_trajectory=camera_data["trajectory"][0],
            contact_labels=contact_data["labels"][0],
            contact_confidence=contact_data["confidence"][0],
            motion_ode_params=ode_data["params"][0] if ode_data["params"] is not None else None,
            motion_basis=ode_data["basis"][0] if ode_data["basis"] is not None else None,
            motion_coefficients=ode_data["coefficients"][0] if ode_data["coefficients"] is not None else None,
            num_frames=T,
            resolution=(H, W),
        )
        
        return motion_data
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video from file."""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        
        cap.release()
        
        return torch.stack(frames).unsqueeze(0)  # [1, T, C, H, W]
