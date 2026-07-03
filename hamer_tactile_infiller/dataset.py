import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .common import count_obj_vertices, load_palm_mask, read_jsonl, valid_bbox
except ImportError:
    from common import count_obj_vertices, load_palm_mask, read_jsonl, valid_bbox

FT_DIR = Path(__file__).resolve().parents[1] / "hamer_tactile_ft"
if str(FT_DIR) not in sys.path:
    sys.path.append(str(FT_DIR))


def expand_manifest_paths(manifest_path):
    if isinstance(manifest_path, (list, tuple)):
        raw_items = [str(item) for item in manifest_path]
    else:
        raw_items = [item.strip() for item in str(manifest_path).split(",") if item.strip()]
    paths = []
    for item in raw_items:
        path = Path(item)
        if any(ch in item for ch in "*?[]"):
            parent = path.parent if str(path.parent) else Path(".")
            matches = sorted(parent.glob(path.name))
            paths.extend(matches)
        else:
            paths.append(path)
    deduped = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def get_nested(data, dotted_key):
    if not dotted_key:
        return None
    cur = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def first_existing_pressure(meta, is_right, priority):
    side = "right" if int(is_right) == 1 else "left"
    candidates = {
        "continuous_subdiv": f"original_hdf5_data.{side}_pressure_continuous_subdiv",
        "gaussian_pressure": "gaussian_pressure",
        "original_hdf5_data": f"original_hdf5_data.{side}_pressure_continuous",
    }
    for name in priority:
        key = candidates.get(name, name)
        value = get_nested(meta, key)
        if value is not None:
            return value
    return None


class TactileSequenceDataset(Dataset):
    def __init__(
        self,
        cfg,
        manifest_path,
        split="train",
        train=True,
        seq_len=16,
        seq_stride=None,
        sample_frame_rate=1,
        min_observed_bbox=1,
        allow_missing_bbox=True,
        mask_prob=0.5,
        target_policy="has_tactile",
        missing_bbox_weight=1.0,
        observed_bbox_weight=0.5,
        pressure_key_priority=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.manifest_paths = expand_manifest_paths(manifest_path)
        if not self.manifest_paths:
            raise FileNotFoundError(f"No manifest paths matched: {manifest_path}")
        self.manifest_path = self.manifest_paths[0]
        self.split = split
        self.train = train
        self.seq_len = int(seq_len)
        self.seq_stride = int(seq_stride if seq_stride is not None else (8 if train else seq_len))
        self.sample_frame_rate = int(sample_frame_rate)
        self.min_observed_bbox = int(min_observed_bbox)
        self.allow_missing_bbox = bool(allow_missing_bbox)
        self.mask_prob = float(mask_prob)
        self.target_policy = str(target_policy)
        self.missing_bbox_weight = float(missing_bbox_weight)
        self.observed_bbox_weight = float(observed_bbox_weight)
        self.pressure_key_priority = pressure_key_priority or [
            "continuous_subdiv",
            "gaussian_pressure",
            "original_hdf5_data",
        ]

        self.img_size = int(cfg.MODEL.IMAGE_SIZE)
        self.rescale_factor = 2.0
        self.tactile_dim = count_obj_vertices()
        self.palm_mask = np.array(load_palm_mask(self.tactile_dim), dtype=np.float32)

        self.sequences = []
        for path in self.manifest_paths:
            if not path.exists():
                raise FileNotFoundError(f"Manifest not found: {path}")
            self.sequences.extend(read_jsonl(path))
        self.windows = self._build_windows()
        manifest_desc = ",".join(str(path) for path in self.manifest_paths)
        print(
            f"[{split}] TactileSequenceDataset loaded {len(self.sequences)} sequences "
            f"and {len(self.windows)} windows from {manifest_desc}"
        )

    def _build_windows(self):
        windows = []
        span = (self.seq_len - 1) * self.sample_frame_rate + 1
        for seq_idx, seq in enumerate(self.sequences):
            frames = seq.get("frames", [])
            if not frames:
                continue
            if len(frames) < span:
                start_candidates = [0]
            else:
                start_candidates = range(0, len(frames) - span + 1, max(1, self.seq_stride))
            for start in start_candidates:
                indices = [start + i * self.sample_frame_rate for i in range(self.seq_len)]
                indices = [min(i, len(frames) - 1) for i in indices]
                window_frames = [frames[i] for i in indices]
                bbox_count = sum(1 for frame in window_frames if frame.get("bbox_valid"))
                tactile_count = sum(1 for frame in window_frames if frame.get("tactile_valid"))
                if bbox_count < self.min_observed_bbox:
                    continue
                if tactile_count <= 0:
                    continue
                if not self.allow_missing_bbox and bbox_count < len(window_frames):
                    continue
                windows.append((seq_idx, indices))
        return windows

    def __len__(self):
        return len(self.windows)

    def _load_meta(self, sample_dir):
        with open(os.path.join(sample_dir, "meta.json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_pressure(self, meta, frame, is_right):
        pressure = get_nested(meta, frame.get("tactile_key"))
        if pressure is None:
            pressure = first_existing_pressure(meta, is_right, self.pressure_key_priority)
        tactile = np.zeros(self.tactile_dim, dtype=np.float32)
        has_tactile = 0.0
        if pressure is not None:
            arr = np.asarray(pressure, dtype=np.float32)
            if arr.shape == (self.tactile_dim,):
                tactile = np.clip(arr, 0.0, 1.0)
                has_tactile = 1.0
        return tactile, has_tactile

    def _load_crop(self, sample_dir, image_name, bbox, is_right, bbox_valid):
        res = self.img_size
        if not bbox_valid:
            return np.zeros((3, res, res), dtype=np.float32)

        img_path = os.path.join(sample_dir, image_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return np.zeros((3, res, res), dtype=np.float32)

        bbox = np.array(bbox, dtype=np.float32)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            return np.zeros((3, res, res), dtype=np.float32)

        center = (bbox[2:4] + bbox[0:2]) / 2.0
        scale_pixels = np.max(bbox[2:4] - bbox[0:2])
        if not np.isfinite(scale_pixels) or scale_pixels <= 1.0:
            return np.zeros((3, res, res), dtype=np.float32)
        bbox_size = self.rescale_factor * scale_pixels

        if self.train:
            augm_config = self.cfg.DATASETS.CONFIG
            scale_aug = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.SCALE_FACTOR + 1.0
            tx = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.TRANS_FACTOR * bbox_size
            ty = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.TRANS_FACTOR * bbox_size
            bbox_size *= scale_aug
            center[0] += tx
            center[1] += ty

        t = np.zeros((2, 3), dtype=np.float32)
        t[0, 0] = float(res) / bbox_size
        t[1, 1] = float(res) / bbox_size
        t[0, 2] = res * (-float(center[0]) / bbox_size + 0.5)
        t[1, 2] = res * (-float(center[1]) / bbox_size + 0.5)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_patch = cv2.warpAffine(img_rgb, t, (res, res), flags=cv2.INTER_LINEAR)
        img_patch = img_patch.astype(np.float32) / 255.0
        if int(is_right) == 0:
            img_patch = cv2.flip(img_patch, 1)
        img_patch = (img_patch - self.cfg.MODEL.IMAGE_MEAN) / self.cfg.MODEL.IMAGE_STD
        return img_patch.transpose(2, 0, 1).astype(np.float32)

    def _observed_mask(self, bbox_valid):
        observed = bbox_valid.astype(np.float32).copy()
        if self.train and self.mask_prob > 0:
            for i in range(len(observed)):
                if observed[i] > 0.5 and random.random() < self.mask_prob:
                    observed[i] = 0.0
            if observed.sum() < self.min_observed_bbox:
                valid_indices = np.where(bbox_valid > 0.5)[0]
                if valid_indices.size:
                    keep = np.random.choice(valid_indices, size=min(self.min_observed_bbox, valid_indices.size), replace=False)
                    observed[keep] = 1.0
        return observed

    def __getitem__(self, idx):
        seq_idx, indices = self.windows[idx]
        seq = self.sequences[seq_idx]
        frames = [seq["frames"][i] for i in indices]
        is_right = int(seq.get("is_right", 1))

        imgs = []
        tactiles = []
        has_tactile = []
        bbox_valid = []
        frame_idx = []
        loss_weight = []
        dataset_names = []

        for frame in frames:
            sample_dir = frame["sample_dir"]
            meta = self._load_meta(sample_dir)
            bbox_ok = bool(frame.get("bbox_valid")) and valid_bbox(frame.get("bbox"))
            tactile, has_t = self._load_pressure(meta, frame, is_right)
            imgs.append(self._load_crop(sample_dir, frame.get("image", "image.jpg"), frame.get("bbox"), is_right, bbox_ok))
            tactiles.append(tactile)
            has_tactile.append(has_t)
            bbox_valid.append(1.0 if bbox_ok else 0.0)
            frame_idx.append(int(frame.get("frame_idx", 0)))
            loss_weight.append(self.observed_bbox_weight if bbox_ok else self.missing_bbox_weight)
            dataset_names.append(seq.get("dataset", "OpenTouch"))

        bbox_valid = np.array(bbox_valid, dtype=np.float32)
        has_tactile = np.array(has_tactile, dtype=np.float32)
        observed_mask = self._observed_mask(bbox_valid)
        if self.target_policy != "has_tactile":
            raise ValueError(f"Unsupported target_policy: {self.target_policy}")
        target_mask = has_tactile.copy()

        return {
            "img": torch.from_numpy(np.stack(imgs, axis=0)).float(),
            "bbox_valid": torch.from_numpy(bbox_valid).float(),
            "observed_mask": torch.from_numpy(observed_mask).float(),
            "target_mask": torch.from_numpy(target_mask).float(),
            "tactile_signal": torch.from_numpy(np.stack(tactiles, axis=0)).float(),
            "has_tactile": torch.from_numpy(has_tactile).float(),
            "loss_weight": torch.tensor(loss_weight, dtype=torch.float32),
            "palm_mask": torch.from_numpy(np.repeat(self.palm_mask[None, :], self.seq_len, axis=0)).float(),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
            "dataset": dataset_names,
            "sequence_id": seq.get("sequence_id", ""),
            "hand": seq.get("hand", "right"),
            "right": torch.tensor(float(is_right)).float(),
        }
