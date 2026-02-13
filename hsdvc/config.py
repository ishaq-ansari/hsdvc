"""
Configuration management using OmegaConf and dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from omegaconf import MISSING


@dataclass
class ModelConfig:
    """Base model configuration."""
    name: str = MISSING
    pretrained: bool = True
    checkpoint_path: Optional[str] = None


@dataclass
class MotionExtractionConfig:
    """Motion extraction configuration."""
    # Pose estimation
    pose_model: str = "mediapipe"  # mediapipe, openpose, vitpose
    pose_dim: int = 33  # Number of keypoints
    
    # Depth estimation
    depth_model: str = "depth_anything_v2"  # midas, depth_anything
    depth_resolution: Tuple[int, int] = (384, 512)
    
    # Optical flow
    flow_model: str = "raft"  # raft, gmflow, unimatch
    flow_resolution: Tuple[int, int] = (384, 512)
    
    # Camera estimation
    camera_model: str = "colmap_free"  # colmap, dust3r, colmap_free
    
    # Contact detection
    detect_contacts: bool = True
    contact_threshold: float = 0.05
    
    # Motion ODE
    use_motion_ode: bool = True
    ode_dim: int = 64
    

@dataclass
class GeometryConfig:
    """3D geometry representation configuration."""
    type: str = "gaussian_splatting"  # gaussian_splatting, nerf, dynamic_mesh
    
    # Gaussian Splatting
    num_gaussians: int = 100000
    gaussian_dim: int = 3
    sh_degree: int = 3
    
    # NeRF
    nerf_grid_size: int = 128
    nerf_num_layers: int = 8
    
    # Dynamic Mesh
    mesh_vertices: int = 5000
    mesh_rank: int = 32


@dataclass
class IdentityConfig:
    """Identity encoder configuration."""
    # Factorization
    shape_dim: int = 512
    appearance_dim: int = 512
    texture_dim: int = 1024
    
    # Encoder architecture
    encoder_type: str = "resnet50"  # resnet50, vit_base, dinov2
    
    # Training
    use_triplet_loss: bool = True
    triplet_margin: float = 0.2


@dataclass
class CogVideoXConfig:
    """CogVideoX model configuration."""
    model_name: str = "THUDM/CogVideoX-5b"
    
    # Architecture
    num_frames: int = 49
    frame_rate: int = 8
    image_size: Tuple[int, int] = (480, 720)
    
    # Latent space
    latent_channels: int = 16
    vae_scale_factor: int = 8
    
    # Attention
    num_attention_heads: int = 24
    attention_head_dim: int = 64
    
    # Temporal
    temporal_compression: int = 4
    
    # Control
    control_type: str = "full"  # full, lite, minimal
    control_channels: int = 12  # pose + depth + flow + mask
    
    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0",
        "proj_in", "proj_out"
    ])


@dataclass  
class ControlNetConfig:
    """ControlNet-style conditioning configuration."""
    num_control_layers: int = 12
    control_channels: int = 12
    
    # Architecture matches CogVideoX
    hidden_channels: List[int] = field(default_factory=lambda: [320, 640, 1280, 1280])
    num_res_blocks: int = 2
    attention_resolutions: List[int] = field(default_factory=lambda: [4, 2, 1])
    
    # Conditioning scale
    control_scale: float = 1.0
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    lr_warmup_steps: int = 500
    
    # Training
    num_epochs: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "fp16"  # fp16, bf16, no
    
    # Per-video compilation
    compilation_steps: int = 500
    compilation_lr: float = 5e-4
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Loss weights
    loss_weights: dict = field(default_factory=lambda: {
        "diffusion": 1.0,
        "motion_consistency": 0.5,
        "identity_preservation": 0.3,
        "temporal_consistency": 0.2,
        "perceptual": 0.1,
    })


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Paths
    train_data_dir: str = MISSING
    val_data_dir: Optional[str] = None
    
    # Video processing
    video_length: int = 49
    frame_rate: int = 8
    resolution: Tuple[int, int] = (480, 720)
    
    # Augmentation
    use_augmentation: bool = True
    random_flip: bool = True
    random_crop: bool = True
    color_jitter: bool = True
    
    # Data loading
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Generation
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    
    # Video
    num_frames: int = 49
    frame_rate: int = 8
    
    # Control
    control_scale: float = 1.0
    
    # Character replacement
    identity_strength: float = 1.0
    motion_preservation_strength: float = 1.0


@dataclass
class HSDVCConfig:
    """Main HSDVC system configuration."""
    # Sub-configs
    motion: MotionExtractionConfig = field(default_factory=MotionExtractionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    cogvideox: CogVideoXConfig = field(default_factory=CogVideoXConfig)
    controlnet: ControlNetConfig = field(default_factory=ControlNetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # System
    seed: int = 42
    device: str = "cuda"
    num_gpus: int = 1
    distributed: bool = False
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
