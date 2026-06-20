import torch
import torch.nn as nn
from typing import Optional


class PoseLoss(nn.Module):
    """
    Loss function for hand pose prediction.
    Combines L1 and L2 losses with optional confidence weighting.
    """
    
    def __init__(
        self,
        use_l1: bool = True,
        use_l2: bool = True,
        l1_weight: float = 0.5,
        l2_weight: float = 0.5,
        use_confidence_weighting: bool = True,
    ):
        super().__init__()
        
        self.use_l1 = use_l1
        self.use_l2 = use_l2
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.use_confidence_weighting = use_confidence_weighting
        
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, T, J, 3) predicted poses
            target: (B, T, J, 3) ground truth poses
            confidences: (B, T, J) confidence scores for each joint
        
        Returns:
            loss: scalar loss value
        """
        total_loss = 0.0
        
        # L1 loss
        if self.use_l1:
            l1 = self.l1_loss(pred, target)  # (B, T, J, 3)
            l1 = l1.mean(dim=-1)  # (B, T, J)
            
            if self.use_confidence_weighting and confidences is not None:
                l1 = l1 * confidences
                l1 = l1.sum() / (confidences.sum() + 1e-8)
            else:
                l1 = l1.mean()
            
            total_loss += self.l1_weight * l1
        
        # L2 loss
        if self.use_l2:
            l2 = self.l2_loss(pred, target)  # (B, T, J, 3)
            l2 = l2.mean(dim=-1)  # (B, T, J)
            
            if self.use_confidence_weighting and confidences is not None:
                l2 = l2 * confidences
                l2 = l2.sum() / (confidences.sum() + 1e-8)
            else:
                l2 = l2.mean()
            
            total_loss += self.l2_weight * l2
        
        return total_loss
