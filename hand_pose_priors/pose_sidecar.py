"""Read and project compact HaMeR pose sidecars."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np


SCHEMA_NAME = "touchanything_hamer_pose_sidecar"
SUPPORTED_SCHEMA_VERSIONS = ("1.0.0", "1.1.0")


def project_camera_vertices(
    vertices_camera: np.ndarray,
    focal_length: np.ndarray | float,
    image_wh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project source-camera vertices into original image pixel coordinates."""

    vertices = np.asarray(vertices_camera, dtype=np.float32)
    if vertices.shape[-2:] != (778, 3):
        raise ValueError(f"Expected [...,778,3] camera vertices, got {vertices.shape}")
    focal = np.asarray(focal_length, dtype=np.float32)
    image_size = np.asarray(image_wh, dtype=np.float32)
    prefix = vertices.shape[:-2]
    focal = np.broadcast_to(focal, prefix)[..., None]
    image_size = np.broadcast_to(image_size, (*prefix, 2))
    depth = vertices[..., 2]
    valid = np.isfinite(vertices).all(axis=-1) & (depth > 1e-6)
    safe_depth = np.maximum(depth, 1e-6)
    uv = np.empty((*prefix, 778, 2), dtype=np.float32)
    uv[..., 0] = focal * vertices[..., 0] / safe_depth + image_size[..., 0, None] / 2.0
    uv[..., 1] = focal * vertices[..., 1] / safe_depth + image_size[..., 1, None] / 2.0
    uv[~valid] = np.nan
    return uv, valid


def tactile_crop_affine(
    bbox_xyxy: np.ndarray,
    *,
    scale: float = 1.2,
    output_hw: Sequence[int] = (256, 192),
) -> np.ndarray:
    """Return the original-image to tactile-crop affine used by crop-1.2."""

    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    if bbox.shape[-1] != 4:
        raise ValueError(f"Expected bbox [...,4], got {bbox.shape}")
    height, width = (int(value) for value in output_hw)
    center = (bbox[..., :2] + bbox[..., 2:]) * 0.5
    box_size = np.max(bbox[..., 2:] - bbox[..., :2], axis=-1) * float(scale)
    if not np.isfinite(box_size).all() or np.any(box_size <= 1.0):
        raise ValueError("Cannot project an invalid tactile bbox")
    affine = np.zeros((*bbox.shape[:-1], 2, 3), dtype=np.float32)
    affine[..., 0, 0] = height / box_size
    affine[..., 1, 1] = height / box_size
    affine[..., 0, 2] = -height * center[..., 0] / box_size + width * 0.5
    affine[..., 1, 2] = height * (-center[..., 1] / box_size + 0.5)
    return affine


def transform_uv(uv: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a batched 2x3 affine to matching vertex UV arrays."""

    coordinates = np.asarray(uv, dtype=np.float32)
    matrix = np.asarray(affine, dtype=np.float32)
    if coordinates.shape[-1] != 2 or matrix.shape[-2:] != (2, 3):
        raise ValueError("Expected UV [...,N,2] and affine [...,2,3]")
    ones = np.ones((*coordinates.shape[:-1], 1), dtype=np.float32)
    homogeneous = np.concatenate((coordinates, ones), axis=-1)
    transformed = np.einsum("...ni,...ji->...nj", homogeneous, matrix)
    transformed[~np.isfinite(coordinates).all(axis=-1)] = np.nan
    return transformed


class HaMeRPoseSidecar:
    """O(1) split/source-row access with a small HDF5 handle cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        require_complete: bool = True,
        max_open_shards: int = 4,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        self.config = json.loads(
            (self.root / "sidecar_config.json").read_text(encoding="utf-8")
        )
        if (
            self.config.get("schema") != SCHEMA_NAME
            or self.config.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS
        ):
            raise ValueError(f"Unsupported HaMeR sidecar schema under {self.root}")
        if require_complete and not (self.root / "SIDECAR_DONE.json").is_file():
            raise RuntimeError(f"HaMeR sidecar is not finalized: {self.root}")
        self.max_open_shards = max(int(max_open_shards), 1)
        self.shard_size = int(self.config["request"]["shard_size"])
        self.sources = {str(source["split"]): source for source in self.config["sources"]}
        self.work_items = {
            (str(item["split"]), int(item["shard_index"])): item
            for item in self.config["work_items"]
        }
        self._handles: OrderedDict[tuple[str, int], h5py.File] = OrderedDict()

    def close(self) -> None:
        while self._handles:
            _, handle = self._handles.popitem(last=False)
            handle.close()

    def __enter__(self) -> "HaMeRPoseSidecar":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def __len__(self) -> int:
        return int(self.config["sample_count"])

    def split_count(self, split: str) -> int:
        return int(self.sources[str(split)]["count"])

    def _handle(self, split: str, shard_index: int) -> h5py.File:
        key = (str(split), int(shard_index))
        handle = self._handles.pop(key, None)
        if handle is None or not handle.id.valid:
            item = self.work_items[key]
            handle = h5py.File(self.root / item["path"], "r", libver="latest")
        self._handles[key] = handle
        while len(self._handles) > self.max_open_shards:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def get(self, split: str, source_row: int, *, derive_uv: bool = False) -> dict[str, Any]:
        split = str(split)
        source_row = int(source_row)
        count = self.split_count(split)
        if source_row < 0 or source_row >= count:
            raise IndexError(f"{split} source_row={source_row} outside [0,{count})")
        shard_index = source_row // self.shard_size
        row = source_row % self.shard_size
        handle = self._handle(split, shard_index)
        result: dict[str, Any] = {
            "sample_uid": str(handle["queries/sample_uid"].asstr()[row]),
            "source_row": int(handle["queries/source_row"][row]),
            "query_row": int(handle["queries/query_row"][row]),
            "frame_row": int(handle["queries/frame_row"][row]),
            "is_right": bool(handle["queries/is_right"][row]),
            "bbox_source_code": int(handle["queries/bbox_source_code"][row]),
            "bbox_xyxy": np.asarray(handle["queries/bbox_xyxy"][row]),
            "bbox_score": float(handle["queries/bbox_score"][row]),
            "image_wh": np.asarray(handle["camera/image_wh"][row]),
            "focal_length": float(handle["camera/focal_length"][row]),
            "camera_translation": np.asarray(handle["camera/translation"][row]),
            "global_orient": np.asarray(handle["mano/global_orient"][row]),
            "hand_pose": np.asarray(handle["mano/hand_pose"][row]),
            "betas": np.asarray(handle["mano/betas"][row]),
            "status": int(handle["quality/status"][row]),
            "positive_depth_fraction": float(
                handle["quality/positive_depth_fraction"][row]
            ),
            "in_frame_fraction": float(handle["quality/in_frame_fraction"][row]),
        }
        if "geometry/vertices_camera" in handle:
            vertices = np.asarray(handle["geometry/vertices_camera"][row], dtype=np.float32)
            result["vertices_camera"] = vertices
            if derive_uv:
                uv, depth_valid = project_camera_vertices(
                    vertices, result["focal_length"], result["image_wh"]
                )
                affine = tactile_crop_affine(result["bbox_xyxy"])
                result.update(
                    {
                        "vertices_uv_full": uv,
                        "vertex_depth_valid": depth_valid,
                        "crop12_affine": affine,
                        "vertices_uv_crop12": transform_uv(uv, affine),
                    }
                )
        elif derive_uv:
            raise RuntimeError(
                "This sidecar was built without camera vertices; regenerate MANO geometry "
                "from the stored right-canonical pose before deriving UV"
            )
        return result
