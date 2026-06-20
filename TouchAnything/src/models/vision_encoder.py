import torch
import torch.nn as nn
import sys
import os
from typing import Optional


class VisionEncoder(nn.Module):
    """
    Vision Encoder using DINOv2.
    Supports dinov2_vits14, dinov2_vitb14, dinov2_vitl14.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        pretrained_path: Optional[str] = None,
        freeze: bool = True,
        patch_size: int = 14,
        embed_dim: int = 384,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Load encoder based on model type
        if model_name.startswith('depth_anything'):
            self.encoder = self._load_depth_anything_v2(model_name, pretrained_path)
        else:
            self.encoder = self._load_dinov2(model_name, pretrained_path)
        
        # Freeze encoder if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        
        print(f"[VisionEncoder] Loaded {model_name}, embed_dim={embed_dim}, freeze={freeze}")
    
    def train(self, mode: bool = True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self
    
    def _load_dinov2(self, model_name: str, pretrained_path: Optional[str] = None):
        """Load DINOv2 model from a local hub cache to avoid implicit network access."""
        hub_dir = torch.hub.get_dir()
        local_repo_candidates = [
            os.path.join(hub_dir, 'facebookresearch_dinov2_main'),
            os.path.join(hub_dir, 'facebookresearch_dinov2_master'),
        ]
        local_repo = next((p for p in local_repo_candidates if os.path.isdir(p)), None)

        if local_repo is None:
            raise ValueError(
                f"Cannot load {model_name}: local DINOv2 torch.hub repo cache not found under {hub_dir}. "
                "Populate the hub cache first instead of relying on an implicit online download."
            )

        try:
            encoder = torch.hub.load(local_repo, model_name, source='local', pretrained=False)
        except Exception as e:
            raise ValueError(f"Cannot build local DINOv2 model {model_name} from {local_repo}: {e}") from e

        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"[VisionEncoder] Loading from local checkpoint: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            encoder.load_state_dict(state_dict, strict=True)
            print(f"[VisionEncoder] Successfully loaded {model_name} from local checkpoint")
            return encoder

        print(f"[VisionEncoder] Loaded {model_name} from local hub cache: {local_repo}")
        return encoder
    
    def _load_depth_anything_v2(self, model_name: str, pretrained_path: Optional[str] = None):
        """Load Depth Anything V2 encoder (fine-tuned DINOv2)."""
        # Map model_name to DA-V2 encoder variant
        variant_map = {
            'depth_anything_v2_vits': 'vits',
            'depth_anything_v2_vitb': 'vitb',
            'depth_anything_v2_vitl': 'vitl',
        }
        variant = variant_map.get(model_name)
        if variant is None:
            raise ValueError(f"Unknown DA-V2 model: {model_name}. Available: {list(variant_map.keys())}")
        
        # Import DA-V2's own DinoVisionTransformer (handles pos_embed for img_size=518)
        da_v2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ref', 'Depth-Anything-V2')
        if da_v2_path not in sys.path:
            sys.path.insert(0, da_v2_path)
        from depth_anything_v2.dinov2 import DINOv2 as DA_DINOv2
        
        # Build architecture using DA-V2's factory (img_size=518, patch_size=14)
        encoder = DA_DINOv2(model_name=variant)
        
        if pretrained_path is not None and os.path.exists(pretrained_path):
            print(f"[VisionEncoder] Loading DA-V2 from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Extract only encoder weights (pretrained.*) and strip prefix
            encoder_state = {}
            for k, v in state_dict.items():
                if k.startswith('pretrained.'):
                    encoder_state[k.replace('pretrained.', '')] = v
            encoder.load_state_dict(encoder_state, strict=True)
            print(f"[VisionEncoder] Successfully loaded DA-V2 {variant} encoder ({len(encoder_state)} params)")
        else:
            print(f"[VisionEncoder] WARNING: No pretrained weights for DA-V2, using random init")
        
        return encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 3, H, W) - batch of video clips
        
        Returns:
            features: (B, T, N, D) - frame-wise features
                N = number of patches (e.g., 16x16=256 for 224x224 image with patch_size=14)
                D = embed_dim
        """
        B, T, C, H, W = x.shape
        
        # Reshape to (B*T, 3, H, W) for batch processing
        x = x.view(B * T, C, H, W)
        
        # Extract features
        if self.freeze:
            with torch.no_grad():
                output = self.encoder.forward_features(x)
        else:
            output = self.encoder.forward_features(x)
        
        # DINOv2 returns a dict, extract patch tokens (without CLS token)
        if isinstance(output, dict):
            features = output['x_norm_patchtokens']  # (B*T, N, D)
        else:
            # Fallback: if it's a tensor, remove CLS token manually
            features = output[:, 1:, :]  # (B*T, N, D)
        
        # Reshape back to (B, T, N, D)
        N = features.shape[1]
        D = features.shape[2]
        features = features.view(B, T, N, D)
        
        return features
    
    def get_num_patches(self, image_size: int) -> int:
        """Calculate number of patches for given image size."""
        return (image_size // self.patch_size) ** 2


class DINOv2Config:
    """Configuration for different DINOv2 models."""
    
    CONFIGS = {
        'dinov2_vits14': {'embed_dim': 384, 'num_heads': 6, 'depth': 12},
        'dinov2_vitb14': {'embed_dim': 768, 'num_heads': 12, 'depth': 12},
        'dinov2_vitl14': {'embed_dim': 1024, 'num_heads': 16, 'depth': 24},
        'depth_anything_v2_vits': {'embed_dim': 384, 'num_heads': 6, 'depth': 12},
        'depth_anything_v2_vitb': {'embed_dim': 768, 'num_heads': 12, 'depth': 12},
        'depth_anything_v2_vitl': {'embed_dim': 1024, 'num_heads': 16, 'depth': 24},
    }
    
    @classmethod
    def get_config(cls, model_name: str) -> dict:
        if model_name not in cls.CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(cls.CONFIGS.keys())}")
        return cls.CONFIGS[model_name]
