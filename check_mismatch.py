import torch
from mmcv import Config
import sys
import os

# add paths
sys.path.append('/code/users/jiangrui/Full-Hand-Tactile-Estimation/hamer')
sys.path.append('/code/users/jiangrui/.conda/envs/tactile/lib/python3.10/site-packages')

import importlib.util
vit_path = "/code/users/jiangrui/Full-Hand-Tactile-Estimation/hamer/third-party/ViTPose/mmpose/models/backbones/vit.py"
spec = importlib.util.spec_from_file_location("mmpose.models.backbones.vit", vit_path)
vit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vit_module)

from mmpose.models import build_posenet

config_path = '/code/users/jiangrui/Full-Hand-Tactile-Estimation/hamer/third-party/ViTPose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_huge_wholebody_256x192.py'
cfg = Config.fromfile(config_path)
cfg.model.pretrained = None
model = build_posenet(cfg.model)

ckpt_path = '/code/users/jiangrui/Full-Hand-Tactile-Estimation/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth'
state_dict = torch.load(ckpt_path, map_location='cpu')
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']

try:
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", len(missing_keys))
    if len(missing_keys) > 0:
        print(missing_keys[:5])
    print("Unexpected keys:", len(unexpected_keys))
    if len(unexpected_keys) > 0:
        print(unexpected_keys[:5])
except Exception as e:
    print("Error:", e)
