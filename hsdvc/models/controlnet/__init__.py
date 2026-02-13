"""
ControlNet-style conditioning for structured video generation.
Injects strong control signals (pose, depth, flow) into diffusion model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange

from ..config import ControlNetConfig, CogVideoXConfig


class ControlNetConditioner(nn.Module):
    """
    ControlNet-style conditioning branch.
    Processes control signals and injects into diffusion model.
    """
    
    def __init__(
        self,
        control_config: ControlNetConfig,
        video_config: CogVideoXConfig
    ):
        super().__init__()
        self.control_config = control_config
        self.video_config = video_config
        
        # Input projection
        self.input_proj = nn.Conv2d(
            control_config.control_channels,
            control_config.hidden_channels[0],
            kernel_size=3,
            padding=1
        )
        
        # Encoder blocks (mirror diffusion model structure)
        self.encoder_blocks = nn.ModuleList()
        
        for i, (in_ch, out_ch) in enumerate(zip(
            control_config.hidden_channels[:-1],
            control_config.hidden_channels[1:]
        )):
            self.encoder_blocks.append(
                ControlNetEncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_res_blocks=control_config.num_res_blocks,
                    downsample=(i < len(control_config.hidden_channels) - 2)
                )
            )
        
        # Zero convolutions for injection (ControlNet trick)
        self.zero_convs = nn.ModuleList([
            self._make_zero_conv(ch)
            for ch in control_config.hidden_channels
        ])
    
    def _make_zero_conv(self, channels: int) -> nn.Conv2d:
        """Create zero-initialized convolution."""
        conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv
    
    def forward(
        self,
        control_signals: Dict[str, torch.Tensor],
        timestep: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Process control signals and return features for injection.
        
        Args:
            control_signals: Dictionary with control tensors
                - poses: [B, T, N, 3] or rendered heatmaps [B, T, C, H, W]
                - depth: [B, T, 1, H, W]
                - flow: [B, T, 2, H, W]
                - mask: [B, T, 1, H, W]
            timestep: [B] diffusion timestep
            
        Returns:
            List of control features for each resolution level
        """
        # Concatenate control signals
        control_input = self._prepare_control_input(control_signals)
        
        # Process through encoder
        x = self.input_proj(control_input)
        
        control_features = []
        control_features.append(self.zero_convs[0](x))
        
        for i, block in enumerate(self.encoder_blocks):
            x = block(x, timestep)
            control_features.append(self.zero_convs[i + 1](x))
        
        return control_features
    
    def _prepare_control_input(
        self,
        control_signals: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Prepare control input by concatenating all signals.
        
        Expected control channels:
        - Pose heatmaps: 3-6 channels (main keypoints)
        - Depth: 1 channel
        - Flow: 2 channels
        - Mask: 1 channel
        Total: ~12 channels
        """
        B, T = None, None
        signals = []
        
        # Process pose
        if "poses" in control_signals:
            poses = control_signals["poses"]
            if poses.dim() == 4:  # Already heatmaps [B, T, C, H, W]
                pose_maps = poses
            else:  # Need to render heatmaps from keypoints
                pose_maps = self._render_pose_heatmaps(poses)
            
            B, T = pose_maps.shape[:2]
            pose_maps = rearrange(pose_maps, "b t c h w -> (b t) c h w")
            signals.append(pose_maps)
        
        # Process depth
        if "depth" in control_signals:
            depth = control_signals["depth"]
            if B is None:
                B, T = depth.shape[:2]
            depth = rearrange(depth, "b t c h w -> (b t) c h w")
            signals.append(depth)
        
        # Process flow
        if "flow" in control_signals:
            flow = control_signals["flow"]
            if B is None:
                B, T = flow.shape[:2]
            flow = rearrange(flow, "b t c h w -> (b t) c h w")
            signals.append(flow)
        
        # Process mask
        if "mask" in control_signals:
            mask = control_signals["mask"]
            if B is None:
                B, T = mask.shape[:2]
            mask = rearrange(mask, "b t c h w -> (b t) c h w")
            signals.append(mask)
        
        # Concatenate all signals
        control_input = torch.cat(signals, dim=1)
        
        # Pad or trim to expected number of channels
        current_channels = control_input.shape[1]
        if current_channels < self.control_config.control_channels:
            padding = torch.zeros(
                control_input.shape[0],
                self.control_config.control_channels - current_channels,
                *control_input.shape[2:],
                device=control_input.device
            )
            control_input = torch.cat([control_input, padding], dim=1)
        elif current_channels > self.control_config.control_channels:
            control_input = control_input[:, :self.control_config.control_channels]
        
        return control_input
    
    def _render_pose_heatmaps(
        self,
        poses: torch.Tensor,
        sigma: float = 2.0,
        target_size: Tuple[int, int] = (64, 64)
    ) -> torch.Tensor:
        """
        Render pose keypoints as Gaussian heatmaps.
        
        Args:
            poses: [B, T, N, 2] 2D keypoints (normalized to [-1, 1])
            sigma: Gaussian sigma
            target_size: Output size
            
        Returns:
            heatmaps: [B, T, N, H, W]
        """
        B, T, N, _ = poses.shape
        H, W = target_size
        
        # Create coordinate grid
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=poses.device),
            torch.linspace(-1, 1, W, device=poses.device),
            indexing='ij'
        )
        grid = torch.stack([x_grid, y_grid], dim=-1)  # [H, W, 2]
        
        # Expand for batch and keypoints
        grid = grid.view(1, 1, 1, H, W, 2).expand(B, T, N, -1, -1, -1)
        poses_expanded = poses.view(B, T, N, 1, 1, 2).expand(-1, -1, -1, H, W, -1)
        
        # Compute Gaussian heatmaps
        diff = grid - poses_expanded
        dist_sq = (diff ** 2).sum(dim=-1)
        heatmaps = torch.exp(-dist_sq / (2 * sigma ** 2))
        
        # Select top keypoints for efficiency (e.g., 6 main body parts)
        # torso, head, left/right shoulders, left/right hips
        key_indices = [0, 10, 11, 12, 23, 24]  # MediaPipe landmark indices
        heatmaps = heatmaps[:, :, key_indices]
        
        return heatmaps


class ControlNetEncoderBlock(nn.Module):
    """Encoder block for ControlNet."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
        downsample: bool = True
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResNetBlock(
                in_channels if i == 0 else out_channels,
                out_channels
            )
            for i in range(num_res_blocks)
        ])
        
        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        for block in self.res_blocks:
            x = block(x, timestep)
        
        if self.downsample is not None:
            x = self.downsample(x)
        
        return x


class ResNetBlock(nn.Module):
    """ResNet block with timestep embedding."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Timestep embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(512, out_channels)
        )
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        h = x
        
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add timestep embedding
        # Simplified: assume timestep is already embedded
        # In practice, use proper timestep embedding (sinusoidal + MLP)
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class StructuredControlInjector(nn.Module):
    """
    Injects structured control into CogVideoX diffusion model.
    """
    
    def __init__(
        self,
        control_config: ControlNetConfig,
        video_config: CogVideoXConfig
    ):
        super().__init__()
        self.control_config = control_config
        self.video_config = video_config
        
        # ControlNet conditioner
        self.controlnet = ControlNetConditioner(control_config, video_config)
        
        # Scaling for control influence
        self.register_buffer(
            "control_scales",
            torch.ones(control_config.num_control_layers)
        )
    
    def forward(
        self,
        noisy_latents: torch.Tensor,
        control_signals: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        diffusion_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Inject control into diffusion features.
        
        Args:
            noisy_latents: [B, C, T, H, W] noisy video latents
            control_signals: Dictionary of control signals
            timestep: [B] diffusion timestep
            diffusion_features: List of features from diffusion model
            
        Returns:
            List of conditioned features
        """
        # Get control features
        control_features = self.controlnet(control_signals, timestep)
        
        # Apply guidance schedule
        guidance_scale = self._compute_guidance_scale(timestep)
        
        # Add control features to diffusion features
        conditioned_features = []
        for i, (diff_feat, ctrl_feat) in enumerate(zip(diffusion_features, control_features)):
            # Scale control based on timestep and layer
            scale = guidance_scale * self.control_scales[i] * self.control_config.control_scale
            
            # Add control
            conditioned = diff_feat + scale * ctrl_feat
            conditioned_features.append(conditioned)
        
        return conditioned_features
    
    def _compute_guidance_scale(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Compute guidance scale based on timestep.
        More control at later stages (lower timesteps).
        """
        # Normalize timestep to [0, 1]
        t_norm = timestep.float() / 1000.0
        
        # Linear schedule
        start = self.control_config.control_guidance_start
        end = self.control_config.control_guidance_end
        
        scale = start + (end - start) * (1.0 - t_norm)
        
        return scale.view(-1, 1, 1, 1, 1)


class TemporalConsistencyRegularizer(nn.Module):
    """
    Ensures temporal consistency across video frames.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predictions: torch.Tensor,
        optical_flow: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            predictions: [B, T, C, H, W] predicted frames
            optical_flow: [B, T-1, 2, H, W] optical flow (optional)
            
        Returns:
            loss: Scalar temporal consistency loss
        """
        B, T, C, H, W = predictions.shape
        
        if T < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        # Simple frame difference loss
        frame_diff = predictions[:, 1:] - predictions[:, :-1]
        diff_loss = torch.abs(frame_diff).mean()
        
        # Flow-based warping loss (if flow provided)
        if optical_flow is not None:
            warp_loss = self._flow_warping_loss(predictions, optical_flow)
        else:
            warp_loss = 0.0
        
        total_loss = diff_loss + warp_loss
        
        return total_loss
    
    def _flow_warping_loss(
        self,
        predictions: torch.Tensor,
        optical_flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flow-based warping consistency loss.
        """
        B, T, C, H, W = predictions.shape
        
        loss = 0.0
        
        for t in range(T - 1):
            frame_t = predictions[:, t]
            frame_t1 = predictions[:, t + 1]
            flow = optical_flow[:, t]
            
            # Warp frame_t to frame_t+1 using flow
            frame_t_warped = self._warp_frame(frame_t, flow)
            
            # Compute difference
            loss += F.l1_loss(frame_t_warped, frame_t1)
        
        return loss / (T - 1)
    
    def _warp_frame(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp frame using optical flow.
        
        Args:
            frame: [B, C, H, W]
            flow: [B, 2, H, W]
            
        Returns:
            warped: [B, C, H, W]
        """
        B, C, H, W = frame.shape
        
        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=frame.device),
            torch.arange(W, device=frame.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()  # [2, H, W]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # Add flow
        grid = grid + flow
        
        # Normalize to [-1, 1]
        grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
        
        # Permute for grid_sample
        grid = grid.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # Warp
        warped = F.grid_sample(
            frame,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped
