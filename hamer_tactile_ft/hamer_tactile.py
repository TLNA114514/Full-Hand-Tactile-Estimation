import torch
import torch.nn as nn
from typing import Dict
from hamer.models.hamer import HAMER

import torch.nn.functional as F

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

class HAMER_Tactile(HAMER):
    def __init__(self, cfg, init_renderer=False):
        super().__init__(cfg, init_renderer=init_renderer)
        
        # We define a tactile head. 
        # Since the backbone output channels (e.g. 1024 or 1280) are passed as `b c h w`,
        # we will use AdaptiveAvgPool2d and then LazyLinear to handle any backbone feature dimension.
        self.tactile_head = nn.Sequential(
            # 1. 优雅的通道降维：保留原始空间分辨率的同时，大幅缩减厚度
            nn.LazyConv2d(out_channels=256, kernel_size=3, padding=1),
            nn.GELU(),
            
            # 2. 释放空间分辨率：从 4x4 (16个网格) 提升至 8x8 (64个网格)，清晰度暴增 4 倍！
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            
            # 3. 强力丢弃，对抗大参数量记忆
            nn.Dropout(p=0.5),
            
            # 4. 后接我们之前部署好的抗过拟合残差集群
            nn.LazyLinear(1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(p=0.3),
            ResidualBlock(1024),
            ResidualBlock(1024),
            nn.Linear(1024, 778)
        )
        
        # Ensure we don't automatically optimize as we want to control freezing
        self.automatic_optimization = True

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
        
        # Compute tactile loss
        gt_tactile = batch['tactile_signal']
        has_tactile = batch['has_tactile']  # (B,) boolean or float
        pred_logits = output['pred_logits']
        pred_tactile = output['pred_tactile']
        
        # SmoothL1 directly aligns with RMSE
        loss_tactile_base = F.smooth_l1_loss(pred_tactile, gt_tactile, reduction='none')
        
        # Mask out non-palm vertices using palm_mask
        palm_mask = batch['palm_mask'] # (B, 778)
        loss_tactile = loss_tactile_base * palm_mask
        
        # Mask out samples that don't have tactile data
        # has_tactile shape is (B,) while loss_tactile shape is (B, 778)
        has_tactile_expanded = has_tactile.unsqueeze(1).expand_as(loss_tactile)
        
        loss_tactile_masked = loss_tactile * has_tactile_expanded
        
        # Average over valid samples and palm vertices
        valid_samples = has_tactile.sum()
        num_palm_vertices = palm_mask[0].sum() if palm_mask.shape[0] > 0 else 0
        if valid_samples > 0 and num_palm_vertices > 0:
            loss_tactile_mean = loss_tactile_masked.sum() / (valid_samples * num_palm_vertices)
        else:
            loss_tactile_mean = torch.tensor(0.0, device=pred_tactile.device, requires_grad=True)
            
        output['losses']['loss_tactile'] = loss_tactile_mean.detach()
        
        # Total loss combines base loss and tactile loss.
        # Since backbone is frozen, base loss will just be passed for logging, 
        # but the gradients will only backpropagate through tactile_head if other parts are frozen.
        total_loss = base_loss + 10.0 * loss_tactile_mean
        
        output['losses']['loss_total'] = total_loss.detach()
        
        return total_loss
