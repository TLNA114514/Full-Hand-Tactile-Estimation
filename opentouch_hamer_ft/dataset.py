import os
import sys
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from yacs.config import CfgNode

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))

from hamer.datasets.utils import get_example, expand_to_aspect_ratio

# Global keypoint permutation for hand flipping (MediaPipe format)
FLIP_KEYPOINT_PERMUTATION = list(range(21))

class OpenTouchHamerDataset(Dataset):
    def __init__(self, cfg: CfgNode, split: str = "train", 
                 split_json: str = None, bbox_json: str = None, 
                 data_dir: str = None, train: bool = True):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.train = train
        
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.rescale_factor = 2.0  # Same as used in eval_hamer.py
        
        # New base directory for the extracted dataset
        if data_dir is None:
            self.data_dir = "/data/jiangrui/OpenTouch Data/extracted_dataset"
        else:
            self.data_dir = data_dir
        
        if split_json is None:
            split_json = os.path.join(workspace_dir, "evaluation/opentouch_splits.json")
            
        with open(split_json, 'r') as f:
            all_splits = json.load(f)
            # 根据当前的 split (train 或 val) 提取列表，并转成 scene/clip_id 格式
            valid_clips = set(f"{item[0]}/{item[1]}" for item in all_splits.get(split, []))
            
        if bbox_json is None:
            bbox_json = os.path.join(ft_dir, "opentouch_train_val_bboxes.json")
            
        with open(bbox_json, 'r') as f:
            samples_dict = json.load(f)
            
        self.samples = []
        for scene_clip, frames in samples_dict.items():
            if scene_clip in valid_clips:
                scene, clip_id = scene_clip.split("/")
                for frame_idx_str, boxes in frames.items():
                    frame_idx = int(frame_idx_str)
                    for box_idx, sample in enumerate(boxes):
                        is_right = sample.get("is_right", 1)
                        self.samples.append({
                            "scene": scene,
                            "clip_id": clip_id,
                            "frame_idx": frame_idx,
                            "is_right": is_right,
                            "bbox": sample["bbox"],
                            "split_folder": split
                        })
        
        print(f"[{split}] Loaded {len(self.samples)} valid bounding boxes from {len(valid_clips)} valid clips.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_meta = self.samples[idx]
        scene = sample_meta["scene"]
        clip_id = sample_meta["clip_id"]
        frame_idx = sample_meta["frame_idx"]
        is_right = sample_meta["is_right"]
        split_folder = sample_meta["split_folder"]
        
        sample_folder_name = f"{scene}_{clip_id}_{frame_idx:04d}_{is_right}"
        sample_dir = os.path.join(self.data_dir, split_folder, sample_folder_name)
        
        img_path = os.path.join(sample_dir, "image.jpg")
        meta_path = os.path.join(sample_dir, "meta.json")
        
        # 1. Check if files exist (in case extraction is not finished)
        if not os.path.exists(img_path) or not os.path.exists(meta_path):
            print(f"Warning: Extracted sample missing at {sample_dir}")
            return self.__getitem__(np.random.randint(0, len(self.samples)))
            
        # 2. Load image using OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
            
        # 3. Load pre-computed metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        bbox = np.array(meta["bbox"], dtype=np.float32)
        is_right = meta["is_right"]
        landmarks_cam = np.array(meta["keypoints_3d_cam"], dtype=np.float32)
        valid_mask = np.array(meta["valid_mask"], dtype=bool)
        
        # 4. Format 3D keypoints for Hamer (N, 4)
        keypoints_3d = np.zeros((21, 4), dtype=np.float32)
        keypoints_3d[valid_mask, :3] = landmarks_cam[valid_mask]
        keypoints_3d[valid_mask, 3] = 1.0
        
        # 5. Calculate bounding box parameters
        if np.isnan(bbox).any() or len(bbox) < 4:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
            
        center = (bbox[2:4] + bbox[0:2]) / 2.0
        center_x, center_y = center[0], center[1]
        
        scale_pixels = np.max(bbox[2:4] - bbox[0:2])
        if np.isnan(scale_pixels) or scale_pixels <= 1.0:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
            
        bbox_size = self.rescale_factor * scale_pixels
        
        # Placeholders
        keypoints_2d = np.zeros((21, 3), dtype=np.float32)
        num_pose = 3 * (self.cfg.MANO.NUM_HAND_JOINTS + 1)
        mano_params = {
            'global_orient': np.zeros(3, dtype=np.float32),
            'hand_pose': np.zeros(num_pose - 3, dtype=np.float32),
            'betas': np.zeros(10, dtype=np.float32)
        }
        has_mano_params = {k: 0.0 for k in mano_params.keys()}
        mano_params_is_axis_angle = {k: True for k in mano_params.keys()}

        # Image Augmentation and cropping (same as original Hamer code)
        augm_config = self.cfg.DATASET.CONFIG
        img_patch, keypoints_2d, keypoints_3d, mano_params, has_mano_params, img_size = get_example(
            img_bgr, center_x, center_y, width=bbox_size, height=bbox_size,
            keypoints_2d=keypoints_2d, keypoints_3d=keypoints_3d,
            mano_params=mano_params, has_mano_params=has_mano_params,
            flip_kp_permutation=FLIP_KEYPOINT_PERMUTATION,
            patch_width=self.img_size, patch_height=self.img_size,
            mean=self.mean, std=self.std, do_augment=self.train,
            is_right=(is_right == 1), augm_config=augm_config, is_bgr=True
        )
        
        item = {
            'img': torch.from_numpy(img_patch).float(),
            'keypoints_3d': torch.from_numpy(keypoints_3d).float(),
            'keypoints_2d': torch.from_numpy(keypoints_2d).float(),
            'box_center': torch.tensor([center_x, center_y]).float(),
            'box_size': torch.tensor(bbox_size).float(),
            'img_size': torch.from_numpy(img_size).float(),
            'right': torch.tensor(float(is_right)).float(),
            'mano_params': {k: torch.from_numpy(v).float() for k, v in mano_params.items()},
            'has_mano_params': {k: torch.tensor(float(v)).float() for k, v in has_mano_params.items()},
            'mano_params_is_axis_angle': {k: torch.tensor(v).bool() for k, v in mano_params_is_axis_angle.items()},
        }
        return item
