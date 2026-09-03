#!/usr/bin/env python3
"""Validate HaMeR inference on a TouchAnything HDF5 frame and SAM3 box."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import cv2
import h5py
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
HAMER_ROOT = REPO_ROOT / "hamer"
DEFAULT_CHECKPOINT = HAMER_ROOT / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
DEFAULT_MANO = HAMER_ROOT / "_DATA/data/mano/MANO_RIGHT.pkl"
DEFAULT_OUTPUT = REPO_ROOT / "hand_pose_priors/outputs/hamer_smoke"
PROCESSED_ROOT_CANDIDATES = (
    Path("/home/ma-user/work/cfzhao/EgoTouch/extracted_frames"),
    REPO_ROOT / "EgoTouch/extracted_frames",
)


def _add_hamer_to_path() -> None:
    value = str(HAMER_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def _resolve_processed_root(raw: str | None) -> Path:
    candidates = []
    if raw:
        candidates.append(Path(raw).expanduser())
    env_value = os.environ.get("TOUCHANYTHING_PROCESSED_ROOT", "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(PROCESSED_ROOT_CANDIDATES)
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    for candidate in unique:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the TouchAnything processed root; pass --processed-root. "
        f"Checked: {[str(value) for value in unique]}"
    )


def _manifest_row(path: Path, row_index: int) -> dict[str, Any]:
    if row_index < 0:
        raise ValueError("--row-index must be non-negative")
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if seen == row_index:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TypeError(f"Manifest row {line_number} is not an object")
                return dict(value)
            seen += 1
    raise IndexError(f"Manifest has no non-empty row at index {row_index}: {path}")


def _decode_hdf5_frame(path: Path, frame_row: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        offsets = handle["images/rgb/jpeg_offsets"]
        data = handle["images/rgb/jpeg_data"]
        if frame_row < 0 or frame_row + 1 >= len(offsets):
            raise IndexError(f"frame_row={frame_row} is outside {path}")
        start = int(offsets[frame_row])
        end = int(offsets[frame_row + 1])
        encoded = np.asarray(data[start:end], dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode frame_row={frame_row} from {path}")
    return image


def _batch_to_device(sample: Mapping[str, torch.Tensor], device: torch.device):
    return {key: value.unsqueeze(0).to(device) for key, value in sample.items()}


def _crop_camera_to_full(
    crop_camera: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    image_size: torch.Tensor,
    focal_length: torch.Tensor,
) -> torch.Tensor:
    image_width, image_height = image_size[:, 0], image_size[:, 1]
    center_x, center_y = box_center[:, 0], box_center[:, 1]
    scale = box_size * crop_camera[:, 0] + 1e-9
    depth = 2.0 * focal_length / scale
    x = 2.0 * (center_x - image_width / 2.0) / scale + crop_camera[:, 1]
    y = 2.0 * (center_y - image_height / 2.0) / scale + crop_camera[:, 2]
    return torch.stack((x, y, depth), dim=-1)


def _project(points_camera: np.ndarray, focal_length: float, image_wh: np.ndarray):
    depth = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (depth > 1e-6)
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float32)
    uv[valid, 0] = focal_length * points_camera[valid, 0] / depth[valid] + image_wh[0] / 2.0
    uv[valid, 1] = focal_length * points_camera[valid, 1] / depth[valid] + image_wh[1] / 2.0
    return uv, valid


def _crop_affine(bbox_xyxy: np.ndarray, scale: float, output_hw=(256, 192)) -> np.ndarray:
    output_height, output_width = (int(value) for value in output_hw)
    center = (bbox_xyxy[:2] + bbox_xyxy[2:]) * 0.5
    box_size = float(np.max(bbox_xyxy[2:] - bbox_xyxy[:2])) * float(scale)
    if not np.isfinite(box_size) or box_size <= 1.0:
        raise ValueError(f"Invalid bbox for crop projection: {bbox_xyxy.tolist()}")
    affine = np.zeros((2, 3), dtype=np.float32)
    affine[0, 0] = output_height / box_size
    affine[1, 1] = output_height / box_size
    affine[0, 2] = -output_height * center[0] / box_size + output_width * 0.5
    affine[1, 2] = output_height * (-center[1] / box_size + 0.5)
    return affine


def _transform_uv(uv: np.ndarray, affine: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((uv, np.ones((len(uv), 1), dtype=np.float32)), axis=1)
    transformed = homogeneous @ affine.T
    transformed[~np.isfinite(uv).all(axis=1)] = np.nan
    return transformed.astype(np.float32, copy=False)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _runtime_check(checkpoint: Path, mano_path: Path, device: torch.device) -> None:
    minimum_sizes = ((checkpoint, 2_000_000_000), (mano_path, 1_000_000))
    for path, minimum_size in minimum_sizes:
        if not path.is_file() or path.stat().st_size < minimum_size:
            raise FileNotFoundError(f"Required HaMeR asset is missing or incomplete: {path}")
    _add_hamer_to_path()
    from hamer.models import load_hamer

    model, config = load_hamer(str(checkpoint), init_renderer=False)
    model = model.eval().to(device)
    print(
        "HAMER_RUNTIME_OK",
        f"checkpoint={checkpoint}",
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}",
        f"device={device}",
        f"image_size={config.MODEL.IMAGE_SIZE}",
        flush=True,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    mano_path = Path(args.mano).expanduser().resolve(strict=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if args.check_only:
        _runtime_check(checkpoint, mano_path, device)
        return

    processed_root = _resolve_processed_root(args.processed_root)
    manifest = (
        Path(args.manifest).expanduser().resolve(strict=True)
        if args.manifest
        else processed_root / "manifests/touchanything_test_seen.queries.jsonl"
    )
    row = _manifest_row(manifest, args.row_index)
    required = ("h5_relpath", "frame_row", "bbox_xyxy", "is_right", "sample_uid")
    missing = [key for key in required if key not in row]
    if missing:
        raise KeyError(f"Manifest row is missing {missing}: {manifest}")
    h5_path = (processed_root / str(row["h5_relpath"])).resolve(strict=True)
    image = _decode_hdf5_frame(h5_path, int(row["frame_row"]))
    bbox = np.asarray(row["bbox_xyxy"], dtype=np.float32)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError(f"Invalid SAM3 bbox: {row['bbox_xyxy']!r}")

    _add_hamer_to_path()
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.models import load_hamer

    model, config = load_hamer(str(checkpoint), init_renderer=False)
    model = model.eval().to(device)
    dataset = ViTDetDataset(
        config,
        image,
        bbox[None],
        np.asarray([int(row["is_right"])], dtype=np.float32),
        rescale_factor=float(args.hamer_bbox_scale),
    )
    batch = _batch_to_device(dataset[0], device)
    if args.precision == "fp16":
        amp_context = torch.autocast(device_type=device.type, dtype=torch.float16)
    elif args.precision == "bf16":
        amp_context = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        amp_context = nullcontext()
    with torch.inference_mode(), amp_context:
        output = model(batch)

    handedness_multiplier = 2.0 * batch["right"].float() - 1.0
    crop_camera = output["pred_cam"].detach().float().clone()
    crop_camera[:, 1] *= handedness_multiplier
    image_size = batch["img_size"].float()
    focal_length = (
        float(config.EXTRA.FOCAL_LENGTH)
        / float(config.MODEL.IMAGE_SIZE)
        * image_size.max(dim=1).values
    )
    full_camera = _crop_camera_to_full(
        crop_camera,
        batch["box_center"].float(),
        batch["box_size"].float(),
        image_size,
        focal_length,
    )
    vertices = output["pred_vertices"].detach().float().clone()
    joints = output["pred_keypoints_3d"].detach().float().clone()
    vertices[:, :, 0] *= handedness_multiplier[:, None]
    joints[:, :, 0] *= handedness_multiplier[:, None]
    vertices_camera = vertices + full_camera[:, None]
    joints_camera = joints + full_camera[:, None]
    tensors = {
        "pred_cam": crop_camera,
        "pred_cam_t_full": full_camera,
        "vertices_mano": vertices,
        "vertices_camera": vertices_camera,
        "joints_mano": joints,
        "joints_camera": joints_camera,
    }
    for key, value in tensors.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"HaMeR returned non-finite {key}")

    vertices_camera_np = vertices_camera[0].cpu().numpy()
    joints_camera_np = joints_camera[0].cpu().numpy()
    image_wh = image_size[0].cpu().numpy()
    focal = float(focal_length[0].item())
    vertex_uv_full, vertex_depth_valid = _project(vertices_camera_np, focal, image_wh)
    joint_uv_full, joint_depth_valid = _project(joints_camera_np, focal, image_wh)
    crop12_affine = _crop_affine(bbox, float(args.tactile_bbox_scale))
    vertex_uv_crop12 = _transform_uv(vertex_uv_full, crop12_affine)
    joint_uv_crop12 = _transform_uv(joint_uv_full, crop12_affine)
    in_frame = (
        vertex_depth_valid
        & (vertex_uv_full[:, 0] >= 0.0)
        & (vertex_uv_full[:, 0] < image_wh[0])
        & (vertex_uv_full[:, 1] >= 0.0)
        & (vertex_uv_full[:, 1] < image_wh[1])
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "hamer_sam3_smoke.npz"
    mano_parameters = {
        key: value.detach().float().cpu().numpy()
        for key, value in output["pred_mano_params"].items()
    }
    np.savez_compressed(
        npz_path,
        vertices_mano=vertices[0].cpu().numpy(),
        vertices_camera=vertices_camera_np,
        vertices_uv_full=vertex_uv_full,
        vertices_uv_crop12=vertex_uv_crop12,
        vertex_depth_valid=vertex_depth_valid,
        vertex_in_frame=in_frame,
        joints_mano=joints[0].cpu().numpy(),
        joints_camera=joints_camera_np,
        joints_uv_full=joint_uv_full,
        joints_uv_crop12=joint_uv_crop12,
        joint_depth_valid=joint_depth_valid,
        pred_cam=crop_camera[0].cpu().numpy(),
        pred_cam_t_full=full_camera[0].cpu().numpy(),
        focal_length=np.asarray(focal, dtype=np.float32),
        bbox_xyxy=bbox,
        crop12_affine=crop12_affine,
        global_orient=mano_parameters["global_orient"][0],
        hand_pose=mano_parameters["hand_pose"][0],
        betas=mano_parameters["betas"][0],
    )

    overlay = image.copy()
    for x, y in vertex_uv_full[in_frame]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 1, (0, 220, 255), -1)
    cv2.rectangle(
        overlay,
        tuple(np.rint(bbox[:2]).astype(int)),
        tuple(np.rint(bbox[2:]).astype(int)),
        (80, 255, 80),
        2,
    )
    overlay_path = output_dir / "hamer_sam3_overlay.jpg"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Could not write {overlay_path}")

    summary = {
        "schema": "hamer_sam3_smoke_v1",
        "sample_uid": str(row["sample_uid"]),
        "manifest": str(manifest),
        "h5_path": str(h5_path),
        "frame_row": int(row["frame_row"]),
        "bbox_xyxy": bbox.tolist(),
        "bbox_source": row.get("bbox_source"),
        "is_right": bool(int(row["is_right"])),
        "mano_parameter_space": "right_hand_canonical",
        "checkpoint": str(checkpoint),
        "precision": args.precision,
        "image_hw": [int(image.shape[0]), int(image.shape[1])],
        "vertex_count": int(len(vertices_camera_np)),
        "joint_count": int(len(joints_camera_np)),
        "positive_depth_fraction": float(vertex_depth_valid.mean()),
        "in_frame_fraction": float(in_frame.mean()),
        "focal_length_pixels": focal,
        "npz": str(npz_path),
        "overlay": str(overlay_path),
    }
    _atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("HAMER_SAM3_SMOKE_OK", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root")
    parser.add_argument("--manifest")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mano", default=str(DEFAULT_MANO))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--hamer-bbox-scale", type=float, default=2.0)
    parser.add_argument("--tactile-bbox-scale", type=float, default=1.2)
    parser.add_argument("--check-only", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
