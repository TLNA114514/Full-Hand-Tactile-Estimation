import torch
import sys
import os

ft_dir = "/code/users/jiangrui/Full-Hand-Tactile-Estimation/hamer_tactile_ft"
workspace_dir = "/code/users/jiangrui/Full-Hand-Tactile-Estimation"
sys.path.append(os.path.join(workspace_dir, "hamer"))
sys.path.append(ft_dir)

from hamer_tactile import HAMER_Tactile
from hamer.configs import get_config

model_cfg = get_config(os.path.join(workspace_dir, "hamer/hamer/configs/hamer_training.yaml"))
model = HAMER_Tactile(model_cfg, init_renderer=False)

# Dummy inputs
B = 2
batch = {
    'img': torch.randn(B, 3, 256, 256),
    'tactile_signal': torch.rand(B, 256),
    'has_tactile': torch.ones(B),
    'keypoints_3d': torch.zeros(B, 21, 4),
    'keypoints_2d': torch.zeros(B, 21, 3),
    'box_center': torch.zeros(B, 2),
    'box_size': torch.zeros(B),
    'img_size': torch.zeros(B, 2),
    'right': torch.ones(B),
    'mano_params': {
        'global_orient': torch.zeros(B, 3),
        'hand_pose': torch.zeros(B, 45),
        'betas': torch.zeros(B, 10)
    },
    'has_mano_params': {
        'global_orient': torch.ones(B),
        'hand_pose': torch.ones(B),
        'betas': torch.ones(B)
    },
    'mano_params_is_axis_angle': {
        'global_orient': True,
        'hand_pose': True,
        'betas': False
    }
}

model.train()
out = model.forward_step(batch, train=True)
print("pred_tactile shape:", out['pred_tactile'].shape)
loss = model.compute_loss(batch, out, train=True)
print("Loss computed successfully.")
