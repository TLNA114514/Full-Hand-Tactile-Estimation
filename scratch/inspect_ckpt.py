import torch
import sys

checkpoint_path = "/code/users/jiangrui/Full-Hand-Tactile-Estimation/opentouch_hamer_ft/checkpoints/regression_60/best_ft_model.ckpt"
try:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt['state_dict']
    for k, v in state_dict.items():
        if 'tactile_head' in k:
            print(f"{k}: {v.shape}")
except Exception as e:
    print(e)
