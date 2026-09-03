#!/usr/bin/env python3
"""Compare a Dyn-HaMR sequence result with its per-frame HaMeR initialization."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import cv2
import numpy as np
import torch


HAND_EDGES = tuple(
    (offset, offset + 1)
    for start in (1, 5, 9, 13, 17)
    for offset in range(start, start + 3)
) + ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17))
JOINT_MAP = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)
FINGERTIP_INDICES = np.asarray([744, 320, 443, 554, 671], dtype=np.int64)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _latest_result(output_dir: Path) -> Path:
    candidates = list((output_dir / "smooth_fit").glob("*_world_results.npz"))
    if not candidates:
        raise FileNotFoundError(f"No smooth-fit result found under {output_dir}")

    def key(path: Path) -> tuple[int, int]:
        fields = path.stem.split("_")
        iterations = [int(field) for field in fields if field.isdigit()]
        return (max(iterations, default=-1), path.stat().st_mtime_ns)

    return max(candidates, key=key)


def _as_bt(value: np.ndarray, batch: int, frames: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[:2] == (batch, frames):
        return array
    if array.shape[0] == frames:
        return np.broadcast_to(array[None], (batch, *array.shape)).copy()
    raise ValueError(f"Cannot normalize temporal tensor with shape {array.shape}")


def _reconstruct(
    dynhamr_root: Path,
    mano_path: Path,
    result: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    source_root = dynhamr_root / "dyn-hamr"
    sys.path.insert(0, str(source_root))
    from body_model import MANO, run_mano  # type: ignore

    pose = torch.from_numpy(np.asarray(result["pose_body"], dtype=np.float32))
    root = torch.from_numpy(np.asarray(result["root_orient"], dtype=np.float32))
    trans = torch.from_numpy(np.asarray(result["trans"], dtype=np.float32))
    betas = torch.from_numpy(np.asarray(result["betas"], dtype=np.float32))
    is_right = torch.from_numpy(np.asarray(result["is_right"], dtype=np.float32))
    batch, frames = pose.shape[:2]
    model = MANO(
        model_path=str(mano_path),
        batch_size=batch * frames,
        pose2rot=True,
        use_pca=False,
        flat_hand_mean=False,
        create_body_pose=False,
    )
    with torch.no_grad():
        geometry = run_mano(model, trans, root, pose, is_right, betas=betas)
    joints_world = geometry["joints"].cpu().numpy().astype(np.float32)
    vertices_world = geometry["vertices"].cpu().numpy().astype(np.float32)
    cam_r = _as_bt(np.asarray(result["cam_R"]), batch, frames)
    cam_t = _as_bt(np.asarray(result["cam_t"]), batch, frames)
    joints_camera = np.einsum("btij,btnj->btni", cam_r, joints_world) + cam_t[:, :, None]
    vertices_camera = np.einsum("btij,btnj->btni", cam_r, vertices_world) + cam_t[:, :, None]
    return joints_camera, vertices_camera


def _intrinsics(value: np.ndarray, batch: int, frames: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (4,):
        return np.broadcast_to(array, (batch, frames, 4)).copy()
    if array.shape == (frames, 4):
        return np.broadcast_to(array[None], (batch, frames, 4)).copy()
    if array.shape == (batch, frames, 4):
        return array
    raise ValueError(f"Unexpected intrinsics shape: {array.shape}")


def _project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = points[..., 2]
    safe_depth = np.maximum(depth, 1e-6)
    uv = np.empty((*points.shape[:-1], 2), dtype=np.float32)
    uv[..., 0] = intrinsics[..., 0, None] * points[..., 0] / safe_depth + intrinsics[..., 2, None]
    uv[..., 1] = intrinsics[..., 1, None] * points[..., 1] / safe_depth + intrinsics[..., 3, None]
    uv[~(np.isfinite(points).all(axis=-1) & (depth > 1e-6))] = np.nan
    return uv


def _joints_from_vertices(vertices: np.ndarray, mano_path: Path) -> np.ndarray:
    with mano_path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    regressor = model["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor, dtype=np.float32)
    if regressor.shape != (16, 778) or vertices.shape[-2:] != (778, 3):
        raise ValueError(
            f"Unexpected MANO geometry shapes: regressor={regressor.shape}, "
            f"vertices={vertices.shape}"
        )
    base_joints = np.einsum("jv,btvc->btjc", regressor, vertices)
    tips = vertices[:, :, FINGERTIP_INDICES]
    return np.concatenate((base_joints, tips), axis=2)[:, :, JOINT_MAP]


def _hand_scales(joints: np.ndarray, minimum: float = 1.0) -> np.ndarray:
    lower = np.nanmin(joints, axis=-2)
    upper = np.nanmax(joints, axis=-2)
    return np.linalg.norm(upper - lower, axis=-1).clip(float(minimum))


def _motion_rms(joints: np.ndarray, order: int, scales: np.ndarray) -> float:
    delta = np.diff(joints, n=order, axis=1)
    normalizer = scales[:, order:, None, None]
    values = delta / normalizer
    finite = np.isfinite(values).all(axis=-1)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values[finite]))))


def _draw_skeleton(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    for hand in range(points.shape[0]):
        uv = points[hand]
        finite = np.isfinite(uv).all(axis=1)
        inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        for first, second in HAND_EDGES:
            if inside[first] and inside[second]:
                cv2.line(
                    output,
                    tuple(np.rint(uv[first]).astype(int)),
                    tuple(np.rint(uv[second]).astype(int)),
                    color,
                    2,
                    cv2.LINE_AA,
                )
        for point in uv[inside]:
            cv2.circle(output, tuple(np.rint(point).astype(int)), 2, color, -1, cv2.LINE_AA)
    cv2.rectangle(output, (0, 0), (235, 32), (0, 0, 0), -1)
    cv2.putText(output, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return output


def _visualize(
    image_dir: Path,
    raw_uv: np.ndarray,
    final_uv: np.ndarray,
    output_dir: Path,
    fps: float,
    priority_frames: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = raw_uv.shape[1]
    selected = sorted(
        set(np.linspace(0, frame_count - 1, 7).round().astype(int).tolist())
        | set(priority_frames)
    )
    writer = None
    for frame in range(frame_count):
        image = cv2.imread(str(image_dir / f"{frame:06d}.jpg"), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read visualization frame {frame}")
        raw_panel = _draw_skeleton(image, raw_uv[:, frame], (0, 190, 255), "HaMeR init")
        final_panel = _draw_skeleton(image, final_uv[:, frame], (80, 230, 80), "Dyn-HaMR")
        overlay = _draw_skeleton(image, raw_uv[:, frame], (0, 190, 255), "overlay")
        overlay = _draw_skeleton(overlay, final_uv[:, frame], (80, 230, 80), "overlay")
        panel = np.concatenate((raw_panel, final_panel, overlay), axis=1)
        if writer is None:
            writer = cv2.VideoWriter(
                str(output_dir / "hamer_vs_dynhamr.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (panel.shape[1], panel.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError("Could not initialize comparison video writer")
        writer.write(panel)
        if frame in selected:
            cv2.imwrite(str(output_dir / f"frame_{frame:06d}.jpg"), panel)
    if writer is not None:
        writer.release()


def audit(args: argparse.Namespace) -> None:
    dynhamr_root = Path(args.checkout).expanduser().resolve(strict=True)
    trial_root = Path(args.trial_root).expanduser().resolve(strict=True)
    run_dir = Path(args.run_dir).expanduser().resolve(strict=True)
    mano_path = Path(args.mano).expanduser().resolve(strict=True)
    result_path = (
        Path(args.result).expanduser().resolve(strict=True)
        if args.result
        else _latest_result(run_dir)
    )
    report_dir = (
        Path(args.report_dir).expanduser().resolve()
        if args.report_dir
        else run_dir / "audit"
    )
    reference_file = np.load(trial_root / "hamer_reference.npz")
    raw_uv = np.asarray(reference_file["joints2d"], dtype=np.float32)
    raw_vertices_camera = np.asarray(
        reference_file["vertices_camera"], dtype=np.float32
    )
    raw_betas = np.asarray(reference_file["betas"], dtype=np.float32)
    raw_joints_camera = _joints_from_vertices(raw_vertices_camera, mano_path)
    visibility = np.asarray(reference_file["visibility"], dtype=bool)
    with np.load(result_path) as loaded:
        result = {key: np.asarray(loaded[key]) for key in loaded.files}
    joints_camera, vertices_camera = _reconstruct(dynhamr_root, mano_path, result)
    batch, frames = joints_camera.shape[:2]
    intrinsics = _intrinsics(result["intrins"], batch, frames)
    final_uv = _project(joints_camera, intrinsics)
    if raw_uv.shape != final_uv.shape:
        raise ValueError(f"Raw/final keypoint shapes differ: {raw_uv.shape} vs {final_uv.shape}")
    valid = visibility[:, :, None] & np.isfinite(raw_uv).all(axis=-1) & np.isfinite(final_uv).all(axis=-1)
    difference = np.linalg.norm(final_uv - raw_uv, axis=-1)
    scales = _hand_scales(raw_uv)
    normalized_difference = difference / scales[:, :, None]
    positive_depth = np.isfinite(vertices_camera).all(axis=-1) & (vertices_camera[..., 2] > 1e-6)
    valid_3d = (
        visibility[:, :, None]
        & np.isfinite(raw_joints_camera).all(axis=-1)
        & np.isfinite(joints_camera).all(axis=-1)
    )
    difference_3d_mm = np.linalg.norm(
        joints_camera - raw_joints_camera, axis=-1
    ) * 1000.0
    scales_3d = _hand_scales(raw_joints_camera, minimum=1e-6)
    final_scales_3d = _hand_scales(joints_camera, minimum=1e-6)
    raw_root = raw_joints_camera[:, :, :1]
    final_root = joints_camera[:, :, :1]
    root_difference_mm = np.linalg.norm(final_root - raw_root, axis=-1) * 1000.0
    root_xy_difference_mm = np.linalg.norm(
        final_root[..., :2] - raw_root[..., :2], axis=-1
    ) * 1000.0
    root_depth_difference_mm = (
        final_root[..., 2] - raw_root[..., 2]
    ) * 1000.0
    root_depth_relative_difference = np.abs(
        final_root[..., 2] - raw_root[..., 2]
    ) / np.maximum(np.abs(raw_root[..., 2]), 1e-6)
    raw_root_relative = raw_joints_camera - raw_root
    final_root_relative = joints_camera - final_root
    articulation_difference_mm = np.linalg.norm(
        final_root_relative - raw_root_relative, axis=-1
    ) * 1000.0
    reference_betas = np.stack(
        [
            np.mean(raw_betas[hand, visibility[hand]], axis=0)
            for hand in range(raw_betas.shape[0])
        ],
        axis=0,
    )
    result_betas = np.asarray(result["betas"], dtype=np.float32)
    beta_l2_change = np.linalg.norm(result_betas - reference_betas, axis=-1)

    raw_velocity = _motion_rms(raw_uv, 1, scales)
    final_velocity = _motion_rms(final_uv, 1, scales)
    raw_acceleration = _motion_rms(raw_uv, 2, scales)
    final_acceleration = _motion_rms(final_uv, 2, scales)
    raw_jerk = _motion_rms(raw_uv, 3, scales)
    final_jerk = _motion_rms(final_uv, 3, scales)
    raw_velocity_3d = _motion_rms(raw_joints_camera, 1, scales_3d)
    final_velocity_3d = _motion_rms(joints_camera, 1, scales_3d)
    raw_acceleration_3d = _motion_rms(raw_joints_camera, 2, scales_3d)
    final_acceleration_3d = _motion_rms(joints_camera, 2, scales_3d)
    raw_jerk_3d = _motion_rms(raw_joints_camera, 3, scales_3d)
    final_jerk_3d = _motion_rms(joints_camera, 3, scales_3d)
    raw_root_acceleration_3d = _motion_rms(raw_root, 2, scales_3d)
    final_root_acceleration_3d = _motion_rms(final_root, 2, scales_3d)
    raw_root_velocity_3d = _motion_rms(raw_root, 1, scales_3d)
    final_root_velocity_3d = _motion_rms(final_root, 1, scales_3d)
    raw_root_jerk_3d = _motion_rms(raw_root, 3, scales_3d)
    final_root_jerk_3d = _motion_rms(final_root, 3, scales_3d)
    raw_articulation_velocity_3d = _motion_rms(
        raw_root_relative, 1, scales_3d
    )
    final_articulation_velocity_3d = _motion_rms(
        final_root_relative, 1, scales_3d
    )
    raw_articulation_acceleration_3d = _motion_rms(
        raw_root_relative, 2, scales_3d
    )
    final_articulation_acceleration_3d = _motion_rms(
        final_root_relative, 2, scales_3d
    )
    raw_articulation_jerk_3d = _motion_rms(raw_root_relative, 3, scales_3d)
    final_articulation_jerk_3d = _motion_rms(final_root_relative, 3, scales_3d)
    summary = {
        "schema": "touchanything_dynhamr_audit",
        "schema_version": "1.1.0",
        "result": str(result_path),
        "frames": frames,
        "tracks": batch,
        "valid_keypoints": int(valid.sum()),
        "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(difference[valid])))),
        "reprojection_median_px": float(np.median(difference[valid])),
        "reprojection_rmse_hand_normalized": float(
            np.sqrt(np.mean(np.square(normalized_difference[valid])))
        ),
        "reprojection_percentiles_px": {
            f"p{percentile}": float(np.percentile(difference[valid], percentile))
            for percentile in (50, 90, 95, 99)
        },
        "positive_depth_fraction": float(positive_depth.mean()),
        "geometry_change_3d": {
            "joint_rmse_mm": float(
                np.sqrt(np.mean(np.square(difference_3d_mm[valid_3d])))
            ),
            "joint_median_mm": float(np.median(difference_3d_mm[valid_3d])),
            "joint_p95_mm": float(np.percentile(difference_3d_mm[valid_3d], 95)),
            "root_rmse_mm": float(
                np.sqrt(np.mean(np.square(root_difference_mm[visibility])))
            ),
            "root_relative_joint_rmse_mm": float(
                np.sqrt(np.mean(np.square(articulation_difference_mm[valid_3d])))
            ),
            "root_relative_joint_median_mm": float(
                np.median(articulation_difference_mm[valid_3d])
            ),
            "root_relative_joint_p95_mm": float(
                np.percentile(articulation_difference_mm[valid_3d], 95)
            ),
            "wrist_xy_shift_median_mm": float(
                np.median(root_xy_difference_mm[visibility])
            ),
            "wrist_xy_shift_p95_mm": float(
                np.percentile(root_xy_difference_mm[visibility], 95)
            ),
            "wrist_depth_shift_median_mm": float(
                np.median(root_depth_difference_mm[visibility])
            ),
            "wrist_depth_shift_abs_median_mm": float(
                np.median(np.abs(root_depth_difference_mm[visibility]))
            ),
            "wrist_depth_shift_abs_p95_mm": float(
                np.percentile(np.abs(root_depth_difference_mm[visibility]), 95)
            ),
            "wrist_depth_shift_relative_abs_median": float(
                np.median(root_depth_relative_difference[visibility])
            ),
            "wrist_depth_shift_relative_abs_p95": float(
                np.percentile(root_depth_relative_difference[visibility], 95)
            ),
            "hand_extent_ratio_median": float(
                np.median((final_scales_3d / scales_3d)[visibility])
            ),
            "beta_l2_change_median": float(np.median(beta_l2_change)),
            "beta_l2_change_max": float(np.max(beta_l2_change)),
        },
        "temporal_2d": {
            "velocity_rms_hamer": raw_velocity,
            "velocity_rms_dynhamr": final_velocity,
            "velocity_ratio": final_velocity / max(raw_velocity, 1e-12),
            "acceleration_rms_hamer": raw_acceleration,
            "acceleration_rms_dynhamr": final_acceleration,
            "acceleration_ratio": final_acceleration / max(raw_acceleration, 1e-12),
            "jerk_rms_hamer": raw_jerk,
            "jerk_rms_dynhamr": final_jerk,
            "jerk_ratio": final_jerk / max(raw_jerk, 1e-12),
        },
        "temporal_3d": {
            "velocity_rms_hamer": raw_velocity_3d,
            "velocity_rms_dynhamr": final_velocity_3d,
            "velocity_ratio": final_velocity_3d / max(raw_velocity_3d, 1e-12),
            "acceleration_rms_hamer": raw_acceleration_3d,
            "acceleration_rms_dynhamr": final_acceleration_3d,
            "acceleration_ratio": final_acceleration_3d
            / max(raw_acceleration_3d, 1e-12),
            "jerk_rms_hamer": raw_jerk_3d,
            "jerk_rms_dynhamr": final_jerk_3d,
            "jerk_ratio": final_jerk_3d / max(raw_jerk_3d, 1e-12),
            "root_acceleration_ratio": final_root_acceleration_3d
            / max(raw_root_acceleration_3d, 1e-12),
            "root_velocity_ratio": final_root_velocity_3d
            / max(raw_root_velocity_3d, 1e-12),
            "root_jerk_ratio": final_root_jerk_3d
            / max(raw_root_jerk_3d, 1e-12),
            "articulation_velocity_ratio": final_articulation_velocity_3d
            / max(raw_articulation_velocity_3d, 1e-12),
            "articulation_acceleration_ratio": final_articulation_acceleration_3d
            / max(raw_articulation_acceleration_3d, 1e-12),
            "articulation_jerk_ratio": final_articulation_jerk_3d
            / max(raw_articulation_jerk_3d, 1e-12),
        },
    }
    summary["per_hand"] = {}
    for hand, name in enumerate(("left", "right")):
        hand_valid = valid[hand]
        hand_difference = difference[hand][hand_valid]
        hand_normalized = normalized_difference[hand][hand_valid]
        summary["per_hand"][name] = {
            "valid_keypoints": int(hand_valid.sum()),
            "reprojection_rmse_px": float(
                np.sqrt(np.mean(np.square(hand_difference)))
            ),
            "reprojection_median_px": float(np.median(hand_difference)),
            "reprojection_rmse_hand_normalized": float(
                np.sqrt(np.mean(np.square(hand_normalized)))
            ),
            "positive_depth_fraction": float(positive_depth[hand].mean()),
        }
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame in range(frames):
        mask = valid[:, frame]
        values = difference[:, frame][mask]
        normalized = normalized_difference[:, frame][mask]
        rows.append(
            {
                "frame": frame,
                "valid_keypoints": int(mask.sum()),
                "reprojection_rmse_px": float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan"),
                "reprojection_rmse_hand_normalized": float(np.sqrt(np.mean(np.square(normalized)))) if len(normalized) else float("nan"),
                "joint_3d_rmse_mm": float(
                    np.sqrt(np.mean(np.square(difference_3d_mm[:, frame][valid_3d[:, frame]])))
                )
                if valid_3d[:, frame].any()
                else float("nan"),
                "positive_depth_fraction": float(positive_depth[:, frame].mean()),
            }
        )
    with (report_dir / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    preparation = json.loads((trial_root / "PREPARE_DONE.json").read_text(encoding="utf-8"))
    _visualize(
        Path(preparation["images"]),
        raw_uv,
        final_uv,
        report_dir / "visuals",
        float(preparation["request"]["fps"]),
        sorted(
            range(frames),
            key=lambda frame: rows[frame]["reprojection_rmse_px"],
            reverse=True,
        )[:5],
    )
    _atomic_json(report_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--trial-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mano", required=True)
    parser.add_argument("--result", default="")
    parser.add_argument("--report-dir", default="")
    return parser


if __name__ == "__main__":
    audit(build_parser().parse_args())
