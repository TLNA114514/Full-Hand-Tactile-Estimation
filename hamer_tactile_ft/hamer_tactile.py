import torch
import torch.nn as nn
from typing import Dict
from hamer.models.hamer import HAMER

class HAMER_Tactile(HAMER):
    def __init__(self, cfg, init_renderer=False):
        super().__init__(cfg, init_renderer=init_renderer)
        
        # We define a tactile head. 
        # Since the backbone output channels (e.g. 1024 or 1280) are passed as `b c h w`,
        # we will use AdaptiveAvgPool2d and then LazyLinear to handle any backbone feature dimension.
        self.tactile_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(1024, 778),
            nn.Sigmoid()
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
            
        pred_tactile = self.tactile_head(conditioning_feats)
        output['pred_tactile'] = pred_tactile
        
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
        pred_tactile = output['pred_tactile']
        
        # We use Smooth L1 Loss for tactile signal regression
        tactile_loss_fn = nn.SmoothL1Loss(reduction='none')
        
        loss_tactile_base = tactile_loss_fn(pred_tactile, gt_tactile)
        
        # Asymmetric penalty: heavily penalize false positives in non-contact areas
        weight = torch.ones_like(gt_tactile)
        weight[gt_tactile < 0.05] = 2.0
        
        loss_tactile = loss_tactile_base * weight
        
        # Mask out samples that don't have tactile data
        # has_tactile shape is (B,) while loss_tactile shape is (B, 778)
        has_tactile_expanded = has_tactile.unsqueeze(1).expand_as(loss_tactile)
        
        loss_tactile_masked = loss_tactile * has_tactile_expanded
        
        # Average over valid samples
        valid_samples = has_tactile.sum()
        if valid_samples > 0:
            loss_tactile_mean = loss_tactile_masked.sum() / (valid_samples * 778.0)
        else:
            loss_tactile_mean = torch.tensor(0.0, device=pred_tactile.device, requires_grad=True)
            
        output['losses']['loss_tactile'] = loss_tactile_mean.detach()
        
        # Total loss combines base loss and tactile loss.
        # Since backbone is frozen, base loss will just be passed for logging, 
        # but the gradients will only backpropagate through tactile_head if other parts are frozen.
        total_loss = base_loss + 10.0 * loss_tactile_mean
        
        output['losses']['loss_total'] = total_loss.detach()
        
        return total_loss
