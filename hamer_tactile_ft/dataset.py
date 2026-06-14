import os
import sys
import json
import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset
from yacs.config import CfgNode

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))

# Global keypoint permutation for hand flipping (MediaPipe format)
FLIP_KEYPOINT_PERMUTATION = list(range(21))

class OpenTouchTactileDataset(Dataset):
    def __init__(self, cfg: CfgNode, split: str = "train", 
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
            
        print(f"[{split}] Computing W_sum for continuous pressure normalization...")
        self.W_sum = self._compute_W_sum()
        
        split_path = os.path.join(self.data_dir, self.split)
        
        # Find all sample directories in the split path
        sample_dirs = glob.glob(os.path.join(split_path, "*"))
        self.samples = [d for d in sample_dirs if os.path.isdir(d)]
        
        print(f"[{split}] Loaded {len(self.samples)} sample folders from {split_path}")

    def _compute_W_sum(self):
        import trimesh
        # 1. Load MANO mesh
        mesh_path = os.path.join(workspace_dir, "opentouch", "preprocess", "scratch", "mano_right_neutral.obj")
        mesh = trimesh.load(mesh_path, process=False)
        mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)

        # 2. Load palm faces/vertices
        palm_faces_path = os.path.join(workspace_dir, "opentouch", "preprocess", "scratch", "auto_calibrated_palm_faces.json")
        with open(palm_faces_path, "r") as f:
            palm_data = json.load(f)
            
        palm_vertices_set = set()
        for triplet in palm_data["group_positive"]["face_triplets"]:
            for vid in triplet:
                if vid <= 777:
                    palm_vertices_set.add(vid)
        palm_vertices = list(palm_vertices_set)
        
        # 3. Load layout
        layout_path = os.path.join(workspace_dir, "opentouch", "preprocess", "scratch", "handLayoutNewest_meshid_lowres.json")
        if not os.path.exists(layout_path):
            layout_path = os.path.join(workspace_dir, "opentouch", "preprocess", "scratch", "handLayoutNewest_meshid.json")
            
        with open(layout_path, "r") as f:
            layout_data = json.load(f)
        layout = layout_data["positions"]
        erased_nodes = set(layout_data.get("erasedNodes", []))

        valid_nodes = {}
        for nid, info in layout.items():
            if nid in erased_nodes:
                continue
            vids = info.get("mano_vid", [])
            vids = [v for v in vids if v <= 777]
            if len(vids) > 0:
                center = np.mean(mano_vertices[vids], axis=0)
                valid_nodes[nid] = center

        n_verts = mano_vertices.shape[0] 
        vert_weights_sum = np.zeros(n_verts, dtype=np.float32)
        sigma = 0.005
        two_sig2 = 2.0 * (sigma * sigma)
        
        centers = []
        for nid, center in valid_nodes.items():
            r, c = map(int, nid.split('-'))
            if r < 16 and c < 16:
                centers.append(center)
                
        if len(centers) > 0:
            centers = np.array(centers, dtype=np.float32) # (K, 3)
            palm_coords = mano_vertices[palm_vertices] # (P, 3)
            diff = palm_coords[:, np.newaxis, :] - centers[np.newaxis, :, :] # (P, K, 3)
            dist2 = np.sum(diff**2, axis=2) # (P, K)
            weights = np.exp(-dist2 / two_sig2) # (P, K)
            weights_sum_P = np.sum(weights, axis=1) # (P,)
            vert_weights_sum[palm_vertices] = weights_sum_P
            
        return vert_weights_sum

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

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
        
        # Extract tactile pressure signal (continuous 778-D vector)
        tactile_key = "right_pressure_continuous" if is_right else "left_pressure_continuous"
        tactile_signal = np.zeros(778, dtype=np.float32)
        has_tactile = 0.0
        if "original_hdf5_data" in meta and tactile_key in meta["original_hdf5_data"]:
            pressure_data = meta["original_hdf5_data"][tactile_key]
            if pressure_data is not None:
                raw_signal = np.array(pressure_data, dtype=np.float32)
                # Apply reverse normalization: target = W_sum - (raw_signal / 3072.0)
                tactile_signal = self.W_sum - (raw_signal / 3072.0)
                tactile_signal = np.clip(tactile_signal, 0.0, 1.0)
                has_tactile = 1.0

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
        mano_params_is_axis_angle = {'global_orient': True, 'hand_pose': True, 'betas': False}

        # Add basic augmentation during training
        if self.train:
            augm_config = self.cfg.DATASETS.CONFIG
            scale_aug = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.SCALE_FACTOR + 1.0
            tx = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.TRANS_FACTOR * bbox_size
            ty = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.TRANS_FACTOR * bbox_size
            
            bbox_size = bbox_size * scale_aug
            center_x += tx
            center_y += ty
            
        # Crop and resize image using affine transform
        res = self.img_size
        t = np.zeros((2, 3), dtype=np.float32)
        t[0, 0] = float(res) / bbox_size
        t[1, 1] = float(res) / bbox_size
        t[0, 2] = res * (-float(center_x) / bbox_size + 0.5)
        t[1, 2] = res * (-float(center_y) / bbox_size + 0.5)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_patch = cv2.warpAffine(img_rgb, t, (res, res), flags=cv2.INTER_LINEAR)
        
        # Normalize and convert to CHW
        img_patch = img_patch.astype(np.float32) / 255.0
        
        if is_right == 0:
            # Flip left hands to right hands for Hamer
            img_patch = cv2.flip(img_patch, 1)
            keypoints_3d[:, 0] = -keypoints_3d[:, 0]
            # Left continuous pressure is passed as is, since it's defined on the MANO vertices space.
            # But wait: if MANO vertices are physically on the right hand, is left_pressure_continuous correctly mapped to right hand vertices?
            # add_continuous_pressure.py uses mano_right_neutral.obj to diffuse BOTH right_pressure and left_pressure!
            # So the left_pressure_continuous is ALREADY computed on the right hand topology!
            # This is brilliant! We don't need to do anything to flip it.
            
        # Standard mean/std normalization
        img_patch = (img_patch - self.cfg.MODEL.IMAGE_MEAN) / self.cfg.MODEL.IMAGE_STD
        img_patch = img_patch.transpose(2, 0, 1)
        
        img_size_array = np.array([img_bgr.shape[1], img_bgr.shape[0]])
        
        item = {
            'img': torch.from_numpy(img_patch).float(),
            'keypoints_3d': torch.from_numpy(keypoints_3d).float(),
            'keypoints_2d': torch.from_numpy(keypoints_2d).float(),
            'tactile_signal': torch.from_numpy(tactile_signal).float(),
            'has_tactile': torch.tensor(has_tactile).float(),
            'box_center': torch.tensor([center_x, center_y]).float(),
            'box_size': torch.tensor(bbox_size).float(),
            'img_size': torch.from_numpy(img_size_array).float(),
            'right': torch.tensor(float(is_right)).float(),
            'mano_params': {k: torch.from_numpy(v).float() for k, v in mano_params.items()},
            'has_mano_params': {k: torch.tensor(float(v)).float() for k, v in has_mano_params.items()},
            'mano_params_is_axis_angle': {k: torch.tensor(v).bool() for k, v in mano_params_is_axis_angle.items()},
        }
        return item
