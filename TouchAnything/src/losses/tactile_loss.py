"""
Loss function for tactile pressure map prediction.
Combines MSE and L1 losses with optional spatial smoothness regularization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TactileLoss(nn.Module):
    """
    Loss for tactile pressure map prediction.
    
    Combines:
    - MSE loss (pixel-wise reconstruction)
    - L1 loss (sparsity-friendly)
    - Optional spatial smoothness (TV loss)
    - Weighted loss (emphasize high-pressure regions to avoid model collapse)
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.5,
        tv_weight: float = 0.01,
        contact_weight: float = 2.0,  # Weight for high-pressure regions
        pressure_threshold: float = 0.1,  # Threshold to consider as "contact"
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.tv_weight = tv_weight
        self.contact_weight = contact_weight
        self.pressure_threshold = pressure_threshold
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def _total_variation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total variation loss for spatial smoothness."""
        # x: (B, T, 2, H, W)
        tv_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).mean()
        tv_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).mean()
        return tv_h + tv_w
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred:   (B, T, 2, S, S) predicted pressure maps [0, 1]
            target: (B, T, 2, S, S) ground truth pressure maps [0, 1]
            mask:   (B, T, 2, S, S) optional sensor mask (1 = valid sensor, 0 = no sensor)
                    If None, loss is computed over all positions.
        
        Returns:
            loss: scalar loss value
        """
        # Compute per-pixel losses
        mse_per_pixel = self.mse_loss(pred, target)  # (B, T, 2, S, S)
        l1_per_pixel = self.l1_loss(pred, target)    # (B, T, 2, S, S)
        
        # Create weight map: higher weight for high-pressure regions
        # This prevents model from collapsing to predicting all zeros
        contact_mask = (target > self.pressure_threshold).float()
        weight_map = 1.0 + (self.contact_weight - 1.0) * contact_mask  # (B, T, 2, S, S)
        
        # Apply sensor mask if provided
        if mask is not None:
            weight_map = weight_map * mask
            valid_pixels = mask.sum()
        else:
            valid_pixels = pred.numel()
        
        # Weighted reconstruction losses
        loss = 0.0
        
        if self.mse_weight > 0:
            weighted_mse = (mse_per_pixel * weight_map).sum() / valid_pixels
            loss += self.mse_weight * weighted_mse
        
        if self.l1_weight > 0:
            weighted_l1 = (l1_per_pixel * weight_map).sum() / valid_pixels
            loss += self.l1_weight * weighted_l1
        
        if self.tv_weight > 0:
            loss += self.tv_weight * self._total_variation(pred)
        
        return loss
