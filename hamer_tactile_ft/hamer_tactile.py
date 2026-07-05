import torch
import torch.nn as nn
from typing import Dict
from hamer.models.hamer import HAMER
from pathlib import Path

from losses import TactileLossConfig, compute_tactile_loss


def count_obj_vertices(obj_path: Path) -> int:
    count = 0
    with obj_path.open("r") as f:
        for line in f:
            if line.startswith("v "):
                count += 1
    return count


def default_tactile_dim() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    subdiv_obj = repo_root / "opentouch" / "preprocess" / "scratch" / "mano_right_neutral_subdiv.obj"
    return count_obj_vertices(subdiv_obj)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        res = x
        x = self.drop(self.act(self.norm1(self.fc1(x))))
        x = self.norm2(self.fc2(x))
        return self.act(x + res)

class AnatomicalSpatialPooling(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        # Mask: shape (1, 1, 5, 5)
        mask = torch.ones((1, 1, 5, 5), dtype=torch.bool)
        # Mask out the bottom-left two and bottom-right two, leaving the wrist at the bottom-center
        mask[0, 0, 4, 0] = False
        mask[0, 0, 4, 1] = False
        mask[0, 0, 4, 3] = False
        mask[0, 0, 4, 4] = False
        self.register_buffer('mask', mask)
        
    def forward(self, x):
        x = self.pool(x) # (B, C, 5, 5)
        B, C, H, W = x.shape
        x_view = x.view(B, C, H * W)
        mask_flat = self.mask.view(-1) # (25,)
        # x_view[:, :, mask_flat] perfectly extracts the 21 valid columns, resulting in (B, C, 21)
        valid_regions = x_view[:, :, mask_flat]
        # Flatten to (B, C * 21)
        x_flat = valid_regions.reshape(B, C * 21)
        return x_flat

class HAMER_Tactile(HAMER):
    def __init__(self, cfg, init_renderer=False):
        super().__init__(cfg, init_renderer=init_renderer)
        self.tactile_dim = default_tactile_dim()
        self.tactile_loss_config = TactileLossConfig()
        
        # 仿生手部解剖学瓶颈架构 (Anatomical Spatial Bottleneck)
        self.tactile_head = nn.Sequential(
            # 1. 剧烈降维 (Channel Squeeze)
            nn.LazyConv2d(out_channels=256, kernel_size=1),
            nn.GELU(),
            
            # 2. 解剖学空间池化 (Anatomical Masking)
            AnatomicalSpatialPooling(channels=256),
            nn.Dropout(p=0.5),
            
            # 3. 轻量化残差解码器 (Lightweight Residual Decoder)
            # 256 channels * 21 regions = 5376 dimensions
            nn.Linear(5376, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            ResidualBlock(512),
            nn.Linear(512, self.tactile_dim)
        )
        
        # Ensure we don't automatically optimize as we want to control freezing
        self.automatic_optimization = True

    def set_tactile_loss_config(self, config: TactileLossConfig):
        self.tactile_loss_config = config

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network, including the tactile head.
        """
        # Run original forward_step which calls backbone and mano_head
        output = super().forward_step(batch, train=train)
        
        # Now get the conditioning features to predict tactile signal
        x = batch['img']
        # We need to run the backbone again because the original HAMER doesn't 
        # store `conditioning_feats` in the `output` dict. 
        # But instead of running the backbone twice, we can just intercept or override.
        # However, for simplicity and since we froze the backbone, let's just 
        # extract features.
        
        with torch.set_grad_enabled(self.backbone.training and train):
            conditioning_feats = self.backbone(x[:, :, :, 32:-32])
            
        pred_logits = self.tactile_head(conditioning_feats)
        output['pred_logits'] = pred_logits
        output['pred_tactile'] = torch.sigmoid(pred_logits)
        
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute total loss including MANO and tactile loss.
        """
        # Get base loss from HAMER
        base_loss = super().compute_loss(batch, output, train=train)
        
        loss_tactile_mean, tactile_losses = compute_tactile_loss(
            pred=output['pred_tactile'],
            logits=output['pred_logits'],
            target=batch['tactile_signal'],
            palm_mask=batch['palm_mask'],
            valid_mask=batch['has_tactile'],
            dataset_batch=batch.get('dataset', None),
            config=self.tactile_loss_config,
            current_epoch=getattr(self, "current_epoch", 0),
        )
        output['losses'].update(tactile_losses)
        
        # Total loss combines base loss and tactile loss.
        # Since backbone is frozen, base loss will just be passed for logging, 
        # but the gradients will only backpropagate through tactile_head if other parts are frozen.
        total_loss = base_loss + 10.0 * loss_tactile_mean
        
        output['losses']['loss_total'] = total_loss.detach()
        
        return total_loss
