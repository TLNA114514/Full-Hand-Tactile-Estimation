import os
import sys
import json
import cv2
import numpy as np
import torch
import glob
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from torch.utils.data import Dataset
from yacs.config import CfgNode

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))

# Global keypoint permutation for hand flipping (MediaPipe format)
FLIP_KEYPOINT_PERMUTATION = list(range(21))
CANONICAL_SPLITS = ("train", "val", "test")

SUBDIV_OBJ_PATH = os.path.join(
    workspace_dir,
    "opentouch",
    "preprocess",
    "scratch",
    "mano_right_neutral_subdiv.obj",
)
SUBDIV_PALM_FACES_PATH = os.path.join(
    workspace_dir,
    "opentouch",
    "preprocess",
    "scratch",
    "auto_calibrated_palm_subdiv_faces.json",
)


def count_obj_vertices(obj_path):
    count = 0
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                count += 1
    return count


def canonical_dataset_name(value):
    raw_name = str(value or "OpenTouch")
    aliases = {
        "opentouch": "OpenTouch",
        "open_touch": "OpenTouch",
        "touchanything": "TouchAnything",
        "touch_anything": "TouchAnything",
        "egotouch": "TouchAnything",
        "ego_touch": "TouchAnything",
        "egotactile": "EgoTactile",
        "ego_tactile": "EgoTactile",
    }
    return aliases.get(raw_name.lower(), raw_name)


def valid_bbox(bbox):
    if bbox is None or bbox == "null":
        return False
    try:
        arr = np.array(bbox, dtype=np.float32)
    except Exception:
        return False
    return arr.shape == (4,) and np.isfinite(arr).all() and np.max(arr[2:4] - arr[0:2]) > 1.0


def has_pressure(meta, dataset_name, hand=None, is_right=None):
    if dataset_name == "TouchAnything":
        hand_meta = meta.get("hands", {}).get(hand or "", {})
        pressure = hand_meta.get("gaussian_pressure")
    else:
        if is_right is None:
            is_right = int(meta.get("is_right", 1))
        side = "right" if int(is_right) == 1 else "left"
        pressure = meta.get("original_hdf5_data", {}).get(f"{side}_pressure_continuous_subdiv")
        if pressure is None:
            pressure = meta.get("gaussian_pressure")
    return pressure is not None


def scan_sample_dir(sample_dir):
    if not os.path.isdir(sample_dir):
        return []
    meta_path = os.path.join(sample_dir, "meta.json")
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return []

    samples = []
    dataset_name = canonical_dataset_name(meta.get("dataset", "OpenTouch"))
    if dataset_name == "TouchAnything":
        for hand in ("left", "right"):
            hand_meta = meta.get("hands", {}).get(hand, {})
            bbox = hand_meta.get("bbox_chest")
            is_right = int(hand_meta.get("is_right", 1 if hand == "right" else 0))
            if valid_bbox(bbox) and has_pressure(meta, dataset_name, hand=hand):
                samples.append({
                    "sample_dir": sample_dir,
                    "dataset": dataset_name,
                    "hand": hand,
                    "is_right": is_right,
                })
    else:
        is_right = int(meta.get("is_right", 1))
        if valid_bbox(meta.get("bbox")) and has_pressure(meta, dataset_name, is_right=is_right):
            samples.append({
                "sample_dir": sample_dir,
                "dataset": dataset_name,
                "hand": "right" if is_right else "left",
                "is_right": is_right,
            })
    return samples


def write_jsonl_atomic(path, rows):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ddp_global_rank():
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


def wait_for_file(path, timeout_sec=3600, poll_sec=5):
    start = time.time()
    while True:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for index cache: {path}")
        time.sleep(poll_sec)


class OpenTouchTactileDataset(Dataset):
    def __init__(self, cfg: CfgNode, split: str = "train", 
                 data_dir: str = None, train: bool = True, index_workers: int = 1,
                 index_chunksize: int = 256, index_cache_dir: str = None,
                 rebuild_index: bool = False, index_cache_timeout: int = 3600,
                 sample_records=None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.train = train
        self.index_workers = max(1, int(index_workers))
        self.index_chunksize = max(1, int(index_chunksize))
        self.index_cache_dir = index_cache_dir
        self.rebuild_index = bool(rebuild_index)
        self.index_cache_timeout = int(index_cache_timeout)
        
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.rescale_factor = 2.0  # Same as used in eval_hamer.py
        
        if data_dir is None:
            data_dirs = ["/data1/jiangrui/OpenTouch Data/extracted_dataset"]
        elif isinstance(data_dir, (list, tuple)):
            data_dirs = [str(d) for d in data_dir if str(d).strip()]
        else:
            data_dirs = [d.strip() for d in str(data_dir).split(",") if d.strip()]
        self.data_dirs = data_dirs
            
        self.tactile_dim = count_obj_vertices(SUBDIV_OBJ_PATH)
        print(f"[{split}] Loading subdiv palm mask for evaluation and loss masking...")
        self.palm_mask = self._load_palm_mask()
        
        if sample_records is None:
            self.samples = self._load_or_build_index()
        else:
            self.samples = list(sample_records)
        source_counts = {}
        for sample in self.samples:
            source_counts[sample["dataset"]] = source_counts.get(sample["dataset"], 0) + 1
        print(f"[{split}] Loaded {len(self.samples)} hand samples from {len(self.data_dirs)} root(s): {source_counts}")

    def _load_palm_mask(self):
        with open(SUBDIV_PALM_FACES_PATH, "r") as f:
            palm_data = json.load(f)
            
        palm_vertices_set = set()
        for triplet in palm_data["group_negative"]["face_triplets"]:
            for vid in triplet:
                if 0 <= vid < self.tactile_dim:
                    palm_vertices_set.add(vid)
        palm_vertices = list(palm_vertices_set)
        
        palm_mask = np.zeros(self.tactile_dim, dtype=np.float32)
        palm_mask[palm_vertices] = 1.0
            
        return palm_mask

    def _has_sample_dirs(self, path):
        if not os.path.isdir(path):
            return False
        for child in glob.glob(os.path.join(path, "*")):
            if os.path.isdir(child) and os.path.exists(os.path.join(child, "meta.json")):
                return True
        return False

    def _split_dir(self, root):
        split_path = os.path.join(root, self.split)
        if os.path.isdir(split_path):
            return split_path

        has_any_split = any(os.path.isdir(os.path.join(root, name)) for name in CANONICAL_SPLITS)
        if has_any_split:
            return None

        if self.split != "train":
            return None

        all_path = os.path.join(root, "all")
        if self._has_sample_dirs(all_path):
            print(f"[{self.split}] No train/val/test under {root}; using {all_path} as train split.")
            return all_path

        if os.path.isdir(root):
            print(f"[{self.split}] No train/val/test under {root}; using the full root as train split.")
            return root

        return None

    def _infer_dataset_name(self, meta):
        return canonical_dataset_name(meta.get("dataset", "OpenTouch"))

    def _valid_bbox(self, bbox):
        return valid_bbox(bbox)

    def _has_pressure(self, meta, dataset_name, hand=None, is_right=None):
        return has_pressure(meta, dataset_name, hand=hand, is_right=is_right)

    def _cache_path(self):
        if not self.index_cache_dir:
            return None
        key_data = {
            "split": self.split,
            "data_dirs": [os.path.abspath(path) for path in self.data_dirs],
            "version": 2,
        }
        digest = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.index_cache_dir, f"{self.split}_{digest}.jsonl")

    def _load_or_build_index(self):
        cache_path = self._cache_path()
        if cache_path is None:
            return self._build_index()

        done_path = f"{cache_path}.done"
        rank = ddp_global_rank()
        if os.path.exists(cache_path) and os.path.exists(done_path) and not self.rebuild_index:
            print(f"[{self.split}] Loading index cache: {cache_path}")
            return read_jsonl(cache_path)

        if rank == 0:
            print(f"[{self.split}] Building index cache on rank 0: {cache_path}")
            if self.rebuild_index:
                for path in (cache_path, done_path):
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
            samples = self._build_index()
            write_jsonl_atomic(cache_path, samples)
            write_jsonl_atomic(done_path, [{"complete": True, "num_samples": len(samples)}])
            print(f"[{self.split}] Wrote index cache: {cache_path}")
            return samples

        print(f"[{self.split}] Rank {rank} waiting for index cache: {cache_path}")
        wait_for_file(done_path, timeout_sec=self.index_cache_timeout)
        return read_jsonl(cache_path)

    def _build_index(self):
        sample_dirs = []
        for root in self.data_dirs:
            split_path = self._split_dir(root)
            if split_path is None:
                print(f"[{self.split}] Warning: split directory not found under {root}")
                continue

            for sample_dir in sorted(glob.glob(os.path.join(split_path, "*"))):
                sample_dirs.append(sample_dir)

        print(
            f"[{self.split}] Index scan: {len(sample_dirs)} sample dirs with "
            f"{self.index_workers} worker(s), chunksize={self.index_chunksize}"
        )
        samples = []
        if self.index_workers == 1:
            for sample_dir in sample_dirs:
                samples.extend(scan_sample_dir(sample_dir))
        else:
            done = 0
            with ProcessPoolExecutor(max_workers=self.index_workers) as executor:
                for result in executor.map(scan_sample_dir, sample_dirs, chunksize=self.index_chunksize):
                    samples.extend(result)
                    done += 1
                    if done % 10000 == 0:
                        print(f"[{self.split}] Indexed {done}/{len(sample_dirs)} sample dirs...")
        samples.sort(key=lambda item: (item["sample_dir"], item["hand"]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_record = self.samples[idx]
        sample_dir = sample_record["sample_dir"]

        meta_path = os.path.join(sample_dir, "meta.json")
        
        # 1. Check if files exist (in case extraction is not finished)
        if not os.path.exists(meta_path):
            print(f"Warning: Extracted sample missing at {sample_dir}")
            return self.__getitem__(np.random.randint(0, len(self.samples)))
            
        # 2. Load pre-computed metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        dataset_name = sample_record.get("dataset", self._infer_dataset_name(meta))
        hand = sample_record.get("hand")
        is_right = int(sample_record.get("is_right", meta.get("is_right", 1)))

        if dataset_name == "TouchAnything":
            hand_meta = meta.get("hands", {}).get(hand, {})
            image_name = meta.get("views", {}).get("chest", "chest.jpg")
            bbox = np.array(hand_meta["bbox_chest"], dtype=np.float32)
            pressure_data = hand_meta.get("gaussian_pressure")
            landmarks_cam = np.zeros((21, 3), dtype=np.float32)
            valid_mask = np.zeros(21, dtype=bool)
        else:
            image_name = meta.get("image", "image.jpg")
            bbox = np.array(meta["bbox"], dtype=np.float32)
            landmarks_cam = np.array(meta.get("keypoints_3d_cam", np.zeros((21, 3))), dtype=np.float32)
            valid_mask = np.array(meta.get("valid_mask", np.zeros(21, dtype=bool)), dtype=bool)
            side = "right" if is_right else "left"
            tactile_key = f"{side}_pressure_continuous_subdiv"
            pressure_data = meta.get("original_hdf5_data", {}).get(tactile_key)
            if pressure_data is None:
                pressure_data = meta.get("gaussian_pressure")

        img_path = os.path.join(sample_dir, image_name)
        if not os.path.exists(img_path):
            print(f"Warning: image missing at {img_path}")
            return self.__getitem__(np.random.randint(0, len(self.samples)))

        # 3. Load image using OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        
        # Extract tactile pressure signal on the subdiv MANO mesh.
        tactile_signal = np.zeros(self.tactile_dim, dtype=np.float32)
        has_tactile = 0.0
        if pressure_data is not None:
            raw_signal = np.array(pressure_data, dtype=np.float32)
            if raw_signal.shape == (self.tactile_dim,):
                tactile_signal = np.clip(raw_signal, 0.0, 1.0)
                has_tactile = 1.0
            else:
                print(
                    f"Warning: tactile signal in {meta_path} has shape {raw_signal.shape}, "
                    f"expected ({self.tactile_dim},). Treating as no tactile data."
                )

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
            # Continuous pressure is already generated on the canonical MANO topology.
            
        # Standard mean/std normalization
        img_patch = (img_patch - self.cfg.MODEL.IMAGE_MEAN) / self.cfg.MODEL.IMAGE_STD
        img_patch = img_patch.transpose(2, 0, 1)
        
        img_size_array = np.array([img_bgr.shape[1], img_bgr.shape[0]])
        
        item = {
            'dataset': dataset_name,
            'img': torch.from_numpy(img_patch).float(),
            'keypoints_3d': torch.from_numpy(keypoints_3d).float(),
            'keypoints_2d': torch.from_numpy(keypoints_2d).float(),
            'tactile_signal': torch.from_numpy(tactile_signal).float(),
            'has_tactile': torch.tensor(has_tactile).float(),
            'palm_mask': torch.from_numpy(self.palm_mask).float(),
            'box_center': torch.tensor([center_x, center_y]).float(),
            'box_size': torch.tensor(bbox_size).float(),
            'img_size': torch.from_numpy(img_size_array).float(),
            'right': torch.tensor(float(is_right)).float(),
            'mano_params': {k: torch.from_numpy(v).float() for k, v in mano_params.items()},
            'has_mano_params': {k: torch.tensor(float(v)).float() for k, v in has_mano_params.items()},
            'mano_params_is_axis_angle': {k: torch.tensor(v).bool() for k, v in mano_params_is_axis_angle.items()},
        }
        return item
