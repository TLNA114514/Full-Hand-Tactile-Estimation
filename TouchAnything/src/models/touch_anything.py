import torch
import torch.nn as nn
from typing import Dict, Optional

from .vision_encoder import VisionEncoder
from .temporal_transformer import TemporalTransformer
from .pose_decoder import PoseDecoder, TactileDecoder, JointLevelTactileDecoder, build_tactile_decoder
from .pose_encoder import PoseEncoder, TransformerPoseEncoder, build_pose_encoder
from .fusion import ConcatFusion, CrossAttentionFusion, build_fusion
from .multi_view_encoder import MultiViewEncoder, build_multi_view_encoder


class TouchAnything(nn.Module):
    """
    Touch Anything: Vision-based Hand Pose and Tactile Prediction Model.
    
    Architecture:
        Single-view:
            Video -> VisionEncoder -> TemporalTransformer -> Decoder
        Multi-view:
            {ego, wrist_L, wrist_R} -> SharedEncoder + ViewEmbed + CrossViewAttn -> TemporalTransformer -> Decoder
        
        Pose prediction:    Visual features -> PoseDecoder -> (B,T,J,3)
        Tactile prediction: Visual features + Pose -> Fusion -> TactileDecoder -> (B,T,2,16,16)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.task = config['model']['task']
        embed_dim = config['model']['vision_encoder']['embed_dim']
        
        # Vision Encoder (DINOv2 or Depth Anything V2)
        vision_cfg = config['model']['vision_encoder']
        self.vision_encoder = VisionEncoder(
            model_name=vision_cfg['model_name'],
            pretrained_path=vision_cfg.get('pretrained_path'),
            freeze=vision_cfg['freeze'],
            patch_size=vision_cfg['patch_size'],
            embed_dim=embed_dim,
        )
        
        # Multi-view wrapper (optional)
        mv_cfg = config['model'].get('multi_view', {})
        self.multi_view_enabled = mv_cfg.get('enabled', False)
        if self.multi_view_enabled:
            self.multi_view_encoder = build_multi_view_encoder(
                self.vision_encoder, mv_cfg, embed_dim
            )
        
        # Temporal Transformer
        temporal_cfg = config['model']['temporal_transformer']
        self.temporal_transformer = TemporalTransformer(
            embed_dim=embed_dim,
            num_layers=temporal_cfg['num_layers'],
            num_heads=temporal_cfg['num_heads'],
            mlp_ratio=temporal_cfg['mlp_ratio'],
            dropout=temporal_cfg['dropout'],
            window_size=temporal_cfg['window_size'],
        )
        
        # Task-specific decoder
        if self.task == 'pose_prediction':
            pose_cfg = config['model']['pose_decoder']
            self.decoder = PoseDecoder(
                input_dim=embed_dim,
                hidden_dims=pose_cfg['hidden_dims'],
                output_dim=pose_cfg['output_dim'],
                dropout=pose_cfg['dropout'],
                decoder_type=pose_cfg['type'],
                num_query_layers=pose_cfg.get('num_query_layers', 2),
                num_query_heads=pose_cfg.get('num_query_heads', 8),
            )
        elif self.task == 'tactile_prediction':
            tactile_cfg = config['model'].get('tactile_decoder', {})
            num_joints = config['model'].get('num_input_joints', 48)
            
            # --- Pose Encoder ---
            pose_enc_cfg = config['model'].get('pose_encoder', {})
            # Backward compat: if no pose_encoder section, use old MLP defaults
            if not pose_enc_cfg or 'type' not in pose_enc_cfg:
                pose_enc_cfg = {
                    'type': 'mlp',
                    'num_joints': num_joints,
                    'hidden_dims': tactile_cfg.get('pose_encoder_dims', [512, 512]),
                    'dropout': tactile_cfg.get('dropout', 0.1),
                }
            pose_enc_cfg.setdefault('num_joints', num_joints)
            self.pose_encoder = build_pose_encoder(pose_enc_cfg, embed_dim)
            
            # --- Fusion ---
            fusion_cfg = config['model'].get('fusion', {})
            if not fusion_cfg or 'type' not in fusion_cfg:
                fusion_cfg = {'type': 'concat'}
            fusion_cfg.setdefault('dropout', tactile_cfg.get('dropout', 0.1))
            self.fusion = build_fusion(fusion_cfg, embed_dim)
            
            # Determine decoder input type based on fusion output
            self.fusion_output_type = self.fusion.output_type  # 'global' or 'per_joint'
            
            # For global fusion + temporal transformer path, keep backward compat
            if self.fusion_output_type == 'global':
                self.fusion_proj = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                )
            
            # --- Tactile Decoder ---
            dec_cfg = dict(tactile_cfg)  # shallow copy
            if 'type' not in dec_cfg:
                dec_cfg['type'] = 'conv'  # backward compat default
            dec_cfg.setdefault('num_joints', num_joints)
            self.decoder = build_tactile_decoder(dec_cfg, embed_dim)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        print(f"[TouchAnything] Initialized with task={self.task}")
    
    def forward(
        self,
        frames: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        views: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, H, W) - single-view video (backward compatible)
            poses: (B, T, J, 3) - hand poses (required for tactile_prediction)
            views: dict of {view_name: (B, T, 3, H, W)} - multi-view input
                   view_name in ['ego', 'wrist_left', 'wrist_right']
                   Any subset can be provided at inference time.
        
        Returns:
            outputs: dict containing predictions
                - 'poses': (B, T, J, 3) for pose prediction
                - 'tactile': (B, T, 2, 16, 16) for tactile prediction
        """
        # Extract visual features (multi-view or single-view)
        if self.multi_view_enabled and views is not None:
            visual_features = self.multi_view_encoder(views)  # (B, T, N, D)
        elif frames is not None:
            visual_features = self.vision_encoder(frames)  # (B, T, N, D)
        elif views is not None:
            # Multi-view input but multi_view not enabled: use ego view
            if 'ego' in views:
                visual_features = self.vision_encoder(views['ego'])
            else:
                visual_features = self.vision_encoder(next(iter(views.values())))
        else:
            raise ValueError("Must provide either 'frames' or 'views'")
        
        if self.task == 'pose_prediction':
            # Temporal modeling
            temporal_features = self.temporal_transformer(visual_features)  # (B, T, N, D)
            poses_out = self.decoder(temporal_features)  # (B, T, J, 3)
            return {'poses': poses_out}
        
        elif self.task == 'tactile_prediction':
            assert poses is not None, "Pose input required for tactile_prediction"
            
            # Encode pose input
            pose_features = self.pose_encoder(poses)  # (B,T,D) or (B,T,J,D)
            
            if self.fusion_output_type == 'global':
                # --- Original path: global fusion + temporal transformer ---
                # Pool pose features if per-joint
                if pose_features.dim() == 4:
                    pose_pooled = pose_features.mean(dim=2)  # (B, T, D)
                else:
                    pose_pooled = pose_features  # (B, T, D)
                
                visual_pooled = visual_features.mean(dim=2)  # (B, T, D)
                fused = torch.cat([visual_pooled, pose_pooled], dim=-1)  # (B, T, 2D)
                fused = self.fusion_proj(fused)  # (B, T, D)
                
                # Add fused token for temporal modeling
                fused_token = fused.unsqueeze(2)  # (B, T, 1, D)
                combined = torch.cat([visual_features, fused_token], dim=2)  # (B, T, N+1, D)
                temporal_features = self.temporal_transformer(combined)  # (B, T, N+1, D)
                tactile_features = temporal_features[:, :, -1, :]  # (B, T, D)
                
                tactile = self.decoder(tactile_features)  # (B, T, 2, 16, 16)
                return {'tactile': tactile}
            
            else:
                # --- New path: cross-attention fusion (per-joint) ---
                # Temporal modeling on visual features first
                temporal_visual = self.temporal_transformer(visual_features)  # (B, T, N, D)
                
                # Cross-attention: per-joint features query temporal visual features
                fused_joints = self.fusion(temporal_visual, pose_features)  # (B, T, J, D)
                
                # Decode from per-joint features
                tactile = self.decoder(fused_joints)  # (B, T, 2, 16, 16)
                return {'tactile': tactile}
    
    def freeze_encoder(self):
        """Freeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()
        print("[TouchAnything] Vision encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze vision encoder parameters."""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        self.vision_encoder.train()
        print("[TouchAnything] Vision encoder unfrozen")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get number of parameters for each component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        temporal_params = sum(p.numel() for p in self.temporal_transformer.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'vision_encoder': vision_params,
            'temporal_transformer': temporal_params,
            'decoder': decoder_params,
        }


def build_model(config: dict) -> TouchAnything:
    """Build TouchAnything model from config."""
    model = TouchAnything(config)
    
    # Print model info
    param_counts = model.get_num_parameters()
    print("\n" + "="*60)
    print("Model Parameters:")
    print(f"  Total:      {param_counts['total']:,}")
    print(f"  Trainable:  {param_counts['trainable']:,}")
    print(f"  Vision:     {param_counts['vision_encoder']:,}")
    print(f"  Temporal:   {param_counts['temporal_transformer']:,}")
    print(f"  Decoder:    {param_counts['decoder']:,}")
    print("="*60 + "\n")
    
    return model
