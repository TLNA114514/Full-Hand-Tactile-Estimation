import os
import sys

# Disable HDF5 file locking to prevent C-level segmentation faults in multi-process/DDP environments
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import json
import h5py
import cv2
import numpy as np
import torch
import io
from PIL import Image
from torch.utils.data import Dataset
from yacs.config import CfgNode
from collections import OrderedDict

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
        
        if split_json is None:
            split_json = os.path.join(workspace_dir, "evaluation/opentouch_splits.json")
        if bbox_json is None:
            bbox_json = os.path.join(ft_dir, "opentouch_train_val_bboxes.json")
        if data_dir is None:
            data_dir = os.path.join(workspace_dir, "opentouch/data")
            
        self.data_dir = data_dir
        
        # Load splits
        with open(split_json, "r", encoding="utf-8") as f:
            splits = json.load(f)
        self.target_clips = splits.get(split, [])
        
        # Load bounding boxes
        if not os.path.exists(bbox_json):
            raise FileNotFoundError(f"Bounding box JSON not found at: {bbox_json}. Please run extract_bboxes.py first.")
        with open(bbox_json, "r", encoding="utf-8") as f:
            self.bbox_cache = json.load(f)
            
        # Build a flat list of samples: (scene_name, clip_id, frame_idx, bbox, is_right)
        self.samples = []
        
        # Blacklist corrupted clips that cause C-level segmentation faults in OpenCV/HDF5
        BLACKLISTED_CLIPS = {
            "grocery_target_p3/demo_148",
            "home_kitchen_p1/demo_077"
        }
        
        for scene, clip in self.target_clips:
            clip_key = f"{scene}/{clip}"
            if clip_key in BLACKLISTED_CLIPS:
                print(f"Skipping blacklisted corrupted clip: {clip_key}")
                continue
            if clip_key not in self.bbox_cache:
                continue
            
            clip_bboxes = self.bbox_cache[clip_key]
            for frame_idx_str, frame_data in clip_bboxes.items():
                frame_idx = int(frame_idx_str)
                for item in frame_data:
                    bbox = item["bbox"]
                    is_right = item["is_right"]
                    self.samples.append({
                        "scene": scene,
                        "clip_id": clip,
                        "frame_idx": frame_idx,
                        "bbox": bbox,
                        "is_right": is_right
                    })
                    
        print(f"Loaded OpenTouchHamerDataset | Split: {split} | Total Clips: {len(self.target_clips)} | Total Hand Samples: {len(self.samples)}")
        
        # Cache for h5py File handles (initialized lazily to avoid multiprocessing fork issues)
        self._h5_files = OrderedDict()
        self.MAX_OPEN_FILES = 8

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        scene = sample["scene"]
        clip_id = sample["clip_id"]
        frame_idx = sample["frame_idx"]
        bbox = np.array(sample["bbox"], dtype=np.float32)
        is_right = sample["is_right"]
        
        # 诊断段错误源：“行车记录仪”机制
        rank = os.environ.get("RANK", "0")
        debug_file = os.path.join(self.data_dir, f"last_sample_rank_{rank}.txt")
        with open(debug_file, "w", encoding="utf-8") as df:
            df.write(f"Scene: {scene}\nClip: {clip_id}\nFrame: {frame_idx}\nis_right: {is_right}\nbbox: {sample['bbox']}\n")
        
        img_bgr = None
        landmarks = None
        
        # Read from HDF5 efficiently: keep files open to avoid C-level resource exhaustion/segfaults 
        # from repeatedly opening/closing HDF5 files thousands of times per second.
        file_path = os.path.join(self.data_dir, f"{scene}.hdf5")
        try:
            if scene not in self._h5_files:
                if len(self._h5_files) >= self.MAX_OPEN_FILES:
                    # Pop the oldest (least recently used) file handle and close it
                    oldest_scene, oldest_f = self._h5_files.popitem(last=False)
                    try:
                        oldest_f.close()
                    except Exception:
                        pass
                
                # swmr=True enables Single Writer Multiple Reader, avoiding lock contention
                self._h5_files[scene] = h5py.File(file_path, "r", swmr=True)
            else:
                # Move to the end to mark as recently used
                self._h5_files.move_to_end(scene)
            
            f = self._h5_files[scene]
            clip_group = f[f"data/{clip_id}"]
            
            # 1. Decode JPEG image
            rgb_bytes_seq = clip_group["rgb_images_jpeg"]
            raw_bytes = rgb_bytes_seq[frame_idx]
            
            is_jpeg = False
            if raw_bytes is not None:
                # Convert numpy array or object to standard bytes for safe verification
                byte_data = raw_bytes.tobytes() if isinstance(raw_bytes, np.ndarray) else raw_bytes
                if isinstance(byte_data, (bytes, bytearray)) and len(byte_data) > 4:
                    if byte_data.startswith(b'\xff\xd8'):
                        is_jpeg = True
                        
            if is_jpeg:
                try:
                    # Decode JPEG using Pillow which is exception-safe and avoids C-level SIGSEGV on corrupt bodies
                    img_pil = Image.open(io.BytesIO(byte_data))
                    # Convert PIL RGB Image to OpenCV BGR numpy array
                    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                except Exception as pil_err:
                    print(f"Warning: Pillow failed to decode JPEG at {scene}/{clip_id} frame {frame_idx}: {pil_err}")
                    img_bgr = None
            else:
                print(f"Warning: Corrupted or invalid JPEG bytes encountered at {scene}/{clip_id} frame {frame_idx}")
            
            # 2. Load GT 3D joints (MediaPipe 21 landmarks, in meters)
            if is_right == 1:
                if "right_hand_landmarks" in clip_group:
                    landmarks = clip_group["right_hand_landmarks"][frame_idx]
            else:
                if "left_hand_landmarks" in clip_group:
                    landmarks = clip_group["left_hand_landmarks"][frame_idx]
        except Exception as e:
            print(f"Warning: Exception encountered while reading {scene}/{clip_id} frame {frame_idx}: {e}")
            
        # Fallback to dummy data if reading or decoding failed
        if img_bgr is None:
            img_bgr = np.zeros((256, 256, 3), dtype=np.uint8)
        if landmarks is None:
            landmarks = np.zeros((21, 3), dtype=np.float32)
            landmarks[:] = np.nan
            
        # Hamer always expects a 3D coordinate + confidence format: [21, 4]
        keypoints_3d = np.zeros((21, 4), dtype=np.float32)
        
        # Fill confidence column based on whether the landmarks are valid
        valid_mask = ~np.isnan(landmarks).any(axis=-1)
        keypoints_3d[valid_mask, :3] = landmarks[valid_mask]
        keypoints_3d[valid_mask, 3] = 1.0  # Valid joint
        keypoints_3d[~valid_mask, 3] = 0.0  # Invalid joint (NaN)
        
        # 3. Calculate bounding box parameters
        center = (bbox[2:4] + bbox[0:2]) / 2.0
        center_x = center[0]
        center_y = center[1]
        
        # Hamer scale calculation logic
        # scale_pixels = max(x2-x1, y2-y1)
        scale_pixels = np.max(bbox[2:4] - bbox[0:2])
        bbox_size = self.rescale_factor * scale_pixels
        
        # Placeholders for keys that we do not have annotations for, but Hamer expects
        # 2D keypoints are set to 0.0 confidence so that Keypoint2DLoss will ignore them
        keypoints_2d = np.zeros((21, 3), dtype=np.float32)
        
        # MANO parameters are set to 0.0 and has_mano_params is set to 0.0 so that ParameterLoss ignores them
        num_pose = 3 * (self.cfg.MANO.NUM_HAND_JOINTS + 1)
        mano_params = {
            'global_orient': np.zeros(3, dtype=np.float32),
            'hand_pose': np.zeros(num_pose - 3, dtype=np.float32),
            'betas': np.zeros(10, dtype=np.float32)
        }
        has_mano_params = {
            'global_orient': 0.0,
            'hand_pose': 0.0,
            'betas': 0.0
        }
        mano_params_is_axis_angle = {
            'global_orient': True,
            'hand_pose': True,
            'betas': False
        }
        
        # Call Hamer official get_example with augmentations enabled during training
        augm_config = self.cfg.DATASETS.CONFIG
        
        # Note: If it's a left hand (is_right=0), Hamer flips the hand horizontally inside get_example
        # so it acts like a right hand. It also processes 3D joints internally.
        img_patch, keypoints_2d, keypoints_3d_proc, mano_params, has_mano_params, img_size = get_example(
            img_path=img_bgr,
            center_x=center_x,
            center_y=center_y,
            width=bbox_size,
            height=bbox_size,
            keypoints_2d=keypoints_2d,
            keypoints_3d=keypoints_3d,
            mano_params=mano_params,
            has_mano_params=has_mano_params,
            flip_kp_permutation=FLIP_KEYPOINT_PERMUTATION,
            patch_width=self.img_size,
            patch_height=self.img_size,
            mean=self.mean,
            std=self.std,
            do_augment=self.train,
            is_right=(is_right == 1),
            augm_config=augm_config,
            is_bgr=True
        )
        
        # Convert dictionary formats
        item = {
            'img': torch.from_numpy(img_patch).float(),
            'keypoints_3d': torch.from_numpy(keypoints_3d_proc).float(),
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

    def __del__(self):
        # Close all cached HDF5 file handles to cleanly release resources
        if hasattr(self, '_h5_files'):
            for f in self._h5_files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._h5_files.clear()
