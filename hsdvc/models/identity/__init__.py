"""
Identity encoder: Factorizes identity into shape, appearance, and texture embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from hsdvc.config import IdentityConfig


@dataclass
class IdentityEmbedding:
    """Container for factorized identity embeddings."""
    shape: torch.Tensor  # [D_shape] - geometric structure
    appearance: torch.Tensor  # [D_appearance] - color/lighting
    texture: torch.Tensor  # [D_texture] - fine details
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "shape": self.shape,
            "appearance": self.appearance,
            "texture": self.texture,
        }
    
    def concat(self) -> torch.Tensor:
        """Concatenate all embeddings."""
        return torch.cat([self.shape, self.appearance, self.texture], dim=-1)


class IdentityBackbone(nn.Module):
    """
    Feature extraction backbone for identity encoding.
    Supports ResNet, ViT, DINOv2, etc.
    """
    
    def __init__(self, encoder_type: str = "resnet50"):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == "resnet50":
            self._init_resnet()
        elif encoder_type == "vit_base":
            self._init_vit()
        elif encoder_type == "dinov2":
            self._init_dinov2()
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def _init_resnet(self):
        """Initialize ResNet50 backbone."""
        from torchvision.models import resnet50, ResNet50_Weights
        
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 2048
        self.backend = "resnet50"
    
    def _init_vit(self):
        """Initialize Vision Transformer backbone."""
        from timm import create_model
        
        self.features = create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,  # Remove head
        )
        self.feature_dim = 768
        self.backend = "vit_base"
    
    def _init_dinov2(self):
        """Initialize DINOv2 backbone (best for identity)."""
        try:
            self.features = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.feature_dim = 768
            self.backend = "dinov2"
        except:
            print("DINOv2 not available, falling back to ViT")
            self._init_vit()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.
        
        Args:
            images: [B, C, H, W] images
            
        Returns:
            features: [B, D] feature vectors
        """
        features = self.features(images)
        
        if self.backend == "resnet50":
            features = features.flatten(1)
        
        return features


class ShapeEncoder(nn.Module):
    """
    Encodes geometric shape/structure information.
    Focuses on body proportions, pose-invariant features.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, output_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode shape embedding.
        
        Args:
            features: [B, D] backbone features
            
        Returns:
            shape_emb: [B, D_shape]
        """
        return self.encoder(features)


class AppearanceEncoder(nn.Module):
    """
    Encodes appearance information: color, lighting, clothing.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, output_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode appearance embedding.
        
        Args:
            features: [B, D] backbone features
            
        Returns:
            appearance_emb: [B, D_appearance]
        """
        return self.encoder(features)


class TextureEncoder(nn.Module):
    """
    Encodes fine-grained texture details.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        # Use deeper network for texture details
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, output_dim),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode texture embedding.
        
        Args:
            features: [B, D] backbone features
            
        Returns:
            texture_emb: [B, D_texture]
        """
        return self.encoder(features)


class DisentanglementLoss(nn.Module):
    """
    Encourages disentanglement between shape, appearance, and texture.
    """
    
    def __init__(self, lambda_ortho: float = 0.01, lambda_sparsity: float = 0.001):
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_sparsity = lambda_sparsity
    
    def forward(self, identity_emb: IdentityEmbedding) -> torch.Tensor:
        """
        Compute disentanglement loss.
        
        Args:
            identity_emb: IdentityEmbedding object
            
        Returns:
            loss: Scalar loss value
        """
        shape = identity_emb.shape
        appearance = identity_emb.appearance
        texture = identity_emb.texture
        
        # Orthogonality loss: encourage different embeddings to be orthogonal
        ortho_loss = 0.0
        
        # Shape vs Appearance
        shape_norm = F.normalize(shape, dim=-1)
        appearance_norm = F.normalize(appearance, dim=-1)
        ortho_loss += torch.abs(torch.sum(shape_norm * appearance_norm, dim=-1)).mean()
        
        # Shape vs Texture
        texture_norm = F.normalize(texture, dim=-1)
        ortho_loss += torch.abs(torch.sum(shape_norm * texture_norm, dim=-1)).mean()
        
        # Appearance vs Texture
        ortho_loss += torch.abs(torch.sum(appearance_norm * texture_norm, dim=-1)).mean()
        
        # Sparsity loss: encourage sparse representations
        sparsity_loss = (
            torch.norm(shape, p=1, dim=-1).mean() +
            torch.norm(appearance, p=1, dim=-1).mean() +
            torch.norm(texture, p=1, dim=-1).mean()
        ) / 3.0
        
        total_loss = (
            self.lambda_ortho * ortho_loss +
            self.lambda_sparsity * sparsity_loss
        )
        
        return total_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for identity learning.
    Ensures same identity is close, different identities are far.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: [B, D] anchor embeddings
            positive: [B, D] positive embeddings (same identity)
            negative: [B, D] negative embeddings (different identity)
            
        Returns:
            loss: Scalar loss value
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        
        return loss


class IdentityEncoder(nn.Module):
    """
    Main identity encoder that factorizes identity into shape, appearance, and texture.
    """
    
    def __init__(self, config: Optional[IdentityConfig] = None):
        super().__init__()
        
        if config is None:
            config = IdentityConfig()
        
        self.config = config
        
        # Backbone feature extractor
        self.backbone = IdentityBackbone(config.encoder_type)
        feature_dim = self.backbone.feature_dim
        
        # Factorized encoders
        self.shape_encoder = ShapeEncoder(feature_dim, config.shape_dim)
        self.appearance_encoder = AppearanceEncoder(feature_dim, config.appearance_dim)
        self.texture_encoder = TextureEncoder(feature_dim, config.texture_dim)
        
        # Loss functions
        self.disentanglement_loss = DisentanglementLoss()
        
        if config.use_triplet_loss:
            self.triplet_loss = TripletLoss(margin=config.triplet_margin)
    
    def forward(self, images: torch.Tensor) -> IdentityEmbedding:
        """
        Encode identity from images.
        
        Args:
            images: [B, C, H, W] images (normalized)
            
        Returns:
            IdentityEmbedding object with shape, appearance, texture
        """
        # Extract backbone features
        features = self.backbone(images)
        
        # Encode factorized embeddings
        shape_emb = self.shape_encoder(features)
        appearance_emb = self.appearance_encoder(features)
        texture_emb = self.texture_encoder(features)
        
        return IdentityEmbedding(
            shape=shape_emb,
            appearance=appearance_emb,
            texture=texture_emb,
        )
    
    def encode_from_path(self, image_path: str) -> IdentityEmbedding:
        """
        Encode identity from image file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            IdentityEmbedding object
        """
        from PIL import Image
        import torchvision.transforms as T
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            identity_emb = self.forward(image_tensor)
        
        return identity_emb
    
    def compute_loss(
        self,
        identity_emb: IdentityEmbedding,
        anchor: Optional[torch.Tensor] = None,
        positive: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute identity learning losses.
        
        Args:
            identity_emb: IdentityEmbedding object
            anchor, positive, negative: For triplet loss (optional)
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Disentanglement loss
        losses["disentanglement"] = self.disentanglement_loss(identity_emb)
        
        # Triplet loss (if inputs provided)
        if self.config.use_triplet_loss and anchor is not None:
            # Use concatenated embeddings for triplet loss
            anchor_concat = anchor.concat() if isinstance(anchor, IdentityEmbedding) else anchor
            positive_concat = positive.concat() if isinstance(positive, IdentityEmbedding) else positive
            negative_concat = negative.concat() if isinstance(negative, IdentityEmbedding) else negative
            
            losses["triplet"] = self.triplet_loss(anchor_concat, positive_concat, negative_concat)
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def similarity(
        self,
        identity1: IdentityEmbedding,
        identity2: IdentityEmbedding,
        weights: Optional[Dict[str, float]] = None
    ) -> torch.Tensor:
        """
        Compute similarity between two identities.
        
        Args:
            identity1, identity2: IdentityEmbedding objects
            weights: Optional weights for each component
            
        Returns:
            similarity: Scalar similarity score [0, 1]
        """
        if weights is None:
            weights = {"shape": 0.4, "appearance": 0.3, "texture": 0.3}
        
        # Cosine similarity for each component
        shape_sim = F.cosine_similarity(identity1.shape, identity2.shape, dim=-1)
        appearance_sim = F.cosine_similarity(identity1.appearance, identity2.appearance, dim=-1)
        texture_sim = F.cosine_similarity(identity1.texture, identity2.texture, dim=-1)
        
        # Weighted average
        similarity = (
            weights["shape"] * shape_sim +
            weights["appearance"] * appearance_sim +
            weights["texture"] * texture_sim
        )
        
        return similarity
    
    def interpolate(
        self,
        identity1: IdentityEmbedding,
        identity2: IdentityEmbedding,
        alpha: float = 0.5,
        component: Optional[str] = None
    ) -> IdentityEmbedding:
        """
        Interpolate between two identities.
        
        Args:
            identity1, identity2: IdentityEmbedding objects
            alpha: Interpolation weight [0, 1]
            component: If specified, only interpolate this component
            
        Returns:
            Interpolated IdentityEmbedding
        """
        if component is None:
            # Interpolate all components
            shape = (1 - alpha) * identity1.shape + alpha * identity2.shape
            appearance = (1 - alpha) * identity1.appearance + alpha * identity2.appearance
            texture = (1 - alpha) * identity1.texture + alpha * identity2.texture
        elif component == "shape":
            shape = (1 - alpha) * identity1.shape + alpha * identity2.shape
            appearance = identity1.appearance
            texture = identity1.texture
        elif component == "appearance":
            shape = identity1.shape
            appearance = (1 - alpha) * identity1.appearance + alpha * identity2.appearance
            texture = identity1.texture
        elif component == "texture":
            shape = identity1.shape
            appearance = identity1.appearance
            texture = (1 - alpha) * identity1.texture + alpha * identity2.texture
        else:
            raise ValueError(f"Unknown component: {component}")
        
        return IdentityEmbedding(shape=shape, appearance=appearance, texture=texture)
