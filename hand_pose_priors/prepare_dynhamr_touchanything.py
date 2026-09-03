#!/usr/bin/env python3
"""Prepare a compact TouchAnything sequence for Dyn-HaMR optimization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Mapping

import cv2
import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_pose_priors.pose_sidecar import HaMeRPoseSidecar  # noqa: E402


DEFAULT_SIDECAR = Path(
    "/home/ma-user/work/cfzhao/hand_pose_sidecars/touchanything_hamer_v1"
)
DEFAULT_PROCESSED_ROOT = Path("/home/ma-user/work/cfzhao/EgoTouch/extracted_frames")
DEFAULT_MANO = REPO_ROOT / "hamer/_DATA/data/mano/MANO_RIGHT.pkl"
DEFAULT_OUTPUT = Path(
    "/home/ma-user/work/cfzhao/hand_pose_dynhamr/trials/"
    "touchanything_arrange_pillow_v1"
)
DEFAULT_SEQUENCE = "Home/arrange_pillow/20260412_101243_753"
VALID_STATUS = 1
TRANSLATION_CONVENTION = "world_translation_after_canonical_hand_mirror_v2"
JOINT_MAP = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _rotation_matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    vector, _ = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64))
    return vector.reshape(3).astype(np.float32)


def _load_joint_regressor(mano_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with mano_path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    regressor = model["J_regressor"]
    if hasattr(regressor, "toarray"):
        regressor = regressor.toarray()
    regressor = np.asarray(regressor, dtype=np.float32)
    if regressor.shape != (16, 778):
        raise ValueError(f"Unexpected MANO joint regressor shape: {regressor.shape}")
    fingertip_indices = np.asarray([744, 320, 443, 554, 671], dtype=np.int64)
    return regressor, fingertip_indices


def _joints_from_vertices(
    vertices: np.ndarray,
    regressor: np.ndarray,
    fingertip_indices: np.ndarray,
) -> np.ndarray:
    joints = np.concatenate(
        (regressor @ vertices, vertices[fingertip_indices]), axis=0
    )
    return joints[JOINT_MAP]


def _project(points: np.ndarray, focal: float, image_wh: np.ndarray) -> np.ndarray:
    depth = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (depth > 1e-6)
    safe_depth = np.maximum(depth, 1e-6)
    uv = np.empty((len(points), 2), dtype=np.float32)
    uv[:, 0] = focal * points[:, 0] / safe_depth + float(image_wh[0]) * 0.5
    uv[:, 1] = focal * points[:, 1] / safe_depth + float(image_wh[1]) * 0.5
    uv[~valid] = np.nan
    return uv


def _lower_bound_uid(
    sidecar: HaMeRPoseSidecar, split: str, target: str
) -> int:
    low, high = 0, sidecar.split_count(split)
    while low < high:
        middle = (low + high) // 2
        uid = str(sidecar.get(split, middle)["sample_uid"])
        if uid < target:
            low = middle + 1
        else:
            high = middle
    return low


def _sequence_records(
    sidecar: HaMeRPoseSidecar, split: str, sequence_key: str
) -> list[dict[str, Any]]:
    prefix = f"TouchAnything/{split}/{sequence_key}/"
    source_row = _lower_bound_uid(sidecar, split, prefix)
    records: list[dict[str, Any]] = []
    count = sidecar.split_count(split)
    while source_row < count:
        record = sidecar.get(split, source_row)
        if not str(record["sample_uid"]).startswith(prefix):
            break
        records.append(record)
        source_row += 1
    if not records:
        raise RuntimeError(
            f"No sidecar records found for split={split!r}, sequence={sequence_key!r}"
        )
    return records


def _record_table(
    records: list[dict[str, Any]], frame_count: int
) -> list[list[dict[str, Any] | None]]:
    table: list[list[dict[str, Any] | None]] = [
        [None, None] for _ in range(frame_count)
    ]
    for record in records:
        frame_row = int(record["frame_row"])
        hand = int(bool(record["is_right"]))
        if 0 <= frame_row < frame_count:
            if table[frame_row][hand] is not None:
                raise RuntimeError(
                    f"Duplicate sidecar query for frame_row={frame_row}, hand={hand}"
                )
            table[frame_row][hand] = record
    return table


def _auto_start(
    table: list[list[dict[str, Any] | None]],
    volumes: np.ndarray,
    num_frames: int,
) -> tuple[int, dict[str, float]]:
    frame_count = len(table)
    if num_frames > frame_count:
        raise ValueError(f"Requested {num_frames} frames from a {frame_count}-frame clip")
    best: tuple[float, int, dict[str, float]] | None = None
    for start in range(frame_count - num_frames + 1):
        stop = start + num_frames
        validity = []
        motion = []
        for hand in (0, 1):
            hand_records = [table[index][hand] for index in range(start, stop)]
            valid = np.asarray(
                [record is not None and int(record["status"]) == VALID_STATUS for record in hand_records],
                dtype=bool,
            )
            validity.append(float(valid.mean()))
            centers = np.full((num_frames, 2), np.nan, dtype=np.float32)
            scales = np.full(num_frames, np.nan, dtype=np.float32)
            for index, record in enumerate(hand_records):
                if record is None or not valid[index]:
                    continue
                bbox = np.asarray(record["bbox_xyxy"], dtype=np.float32)
                centers[index] = 0.5 * (bbox[:2] + bbox[2:])
                scales[index] = max(float(np.max(bbox[2:] - bbox[:2])), 1.0)
            pair_valid = valid[1:] & valid[:-1]
            if pair_valid.any():
                displacement = np.linalg.norm(np.diff(centers, axis=0), axis=1)
                normalization = 0.5 * (scales[1:] + scales[:-1])
                motion.append(float(np.nanmedian(displacement[pair_valid] / normalization[pair_valid])))
            else:
                motion.append(0.0)
        minimum_validity = min(validity)
        if minimum_validity < 0.90:
            continue
        pressure = volumes[start:stop]
        pressure_variation = float(np.nanstd(np.log1p(np.maximum(pressure, 0.0))))
        motion_score = float(np.mean(motion))
        score = 10.0 * minimum_validity + motion_score + 0.05 * pressure_variation
        details = {
            "score": score,
            "left_valid_fraction": validity[0],
            "right_valid_fraction": validity[1],
            "median_normalized_bbox_motion": motion_score,
            "log_pressure_variation": pressure_variation,
        }
        candidate = (score, -start, details)
        if best is None or candidate[:2] > (best[0], -best[1]):
            best = (score, start, details)
    if best is None:
        raise RuntimeError(
            f"No {num_frames}-frame window has at least 90% valid HaMeR rows for both hands"
        )
    return best[1], best[2]


def _encode_video(image_dir: Path, path: Path, fps: float) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(image_dir / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(command, check=True)


def prepare(args: argparse.Namespace) -> None:
    sidecar_root = Path(args.sidecar_root).expanduser().resolve(strict=True)
    processed_root = Path(args.processed_root).expanduser().resolve(strict=True)
    mano_path = Path(args.mano).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    h5_path = processed_root / args.split / f"{args.sequence_key}.h5"
    h5_path = h5_path.resolve(strict=True)
    sequence_name = str(args.sequence_name).strip()
    if not sequence_name or "/" in sequence_name:
        raise ValueError("--sequence-name must be a non-empty directory-safe name")

    request = {
        "schema": "touchanything_dynhamr_trial",
        "schema_version": "1.0.0",
        "sidecar_root": str(sidecar_root),
        "processed_root": str(processed_root),
        "mano": str(mano_path),
        "split": str(args.split),
        "sequence_key": str(args.sequence_key),
        "sequence_name": sequence_name,
        "start": str(args.start),
        "num_frames": int(args.num_frames),
        "fps": float(args.fps),
        "translation_convention": TRANSLATION_CONVENTION,
    }
    config_path = output_root / "prepare_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("request") != request:
            raise RuntimeError(
                f"Existing Dyn-HaMR trial has a different preparation contract: {output_root}"
            )
        done_path = output_root / "PREPARE_DONE.json"
        if done_path.is_file():
            print(f"[dynhamr-prepare] Reusing completed trial: {output_root}")
            print(done_path.read_text(encoding="utf-8").strip())
            return
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(config_path, {"request": request})

    with HaMeRPoseSidecar(sidecar_root) as sidecar, h5py.File(
        h5_path, "r", libver="latest"
    ) as source:
        frame_count = int(source["frames/frame_idx"].shape[0])
        records = _sequence_records(sidecar, args.split, args.sequence_key)
        table = _record_table(records, frame_count)
        per_frame_volume = np.zeros(frame_count, dtype=np.float32)
        query_volume = source["targets/volume"]
        for frame_row, hand_records in enumerate(table):
            values = [
                float(query_volume[int(record["query_row"])])
                for record in hand_records
                if record is not None
            ]
            if values:
                per_frame_volume[frame_row] = float(np.sum(values))
        if str(args.start).lower() == "auto":
            start, selection = _auto_start(table, per_frame_volume, args.num_frames)
        else:
            start = int(args.start)
            selection = {"score": float("nan")}
        stop = start + int(args.num_frames)
        if start < 0 or stop > frame_count:
            raise ValueError(f"Selected frame range [{start},{stop}) outside [0,{frame_count})")

        image_dir = output_root / "images" / sequence_name
        track_root = output_root / "dynhamr/track_preds" / sequence_name
        shot_path = output_root / "dynhamr/shot_idcs" / f"{sequence_name}.json"
        camera_dir = output_root / "dynhamr/cameras" / sequence_name / "shot-0"
        frame_names = [f"{index:06d}.jpg" for index in range(args.num_frames)]
        jpeg_offsets = source["images/rgb/jpeg_offsets"]
        jpeg_data = source["images/rgb/jpeg_data"]
        source_frame_indices = source["frames/source_frame_idx"][start:stop].astype(np.int64)
        timestamps = source["frames/timestamp"][start:stop].astype(np.float64)

        regressor, fingertip_indices = _load_joint_regressor(mano_path)
        raw_joints2d = np.full((2, args.num_frames, 21, 2), np.nan, np.float32)
        raw_vertices = np.full((2, args.num_frames, 778, 3), np.nan, np.float32)
        raw_body_pose = np.full((2, args.num_frames, 15, 3), np.nan, np.float32)
        raw_root_orient = np.full((2, args.num_frames, 3), np.nan, np.float32)
        raw_translation = np.full((2, args.num_frames, 3), np.nan, np.float32)
        raw_betas = np.full((2, args.num_frames, 10), np.nan, np.float32)
        visibility = np.zeros((2, args.num_frames), dtype=bool)
        image_wh = None
        focal_values: list[float] = []
        mapping: list[dict[str, Any]] = []

        for local_index, frame_row in enumerate(range(start, stop)):
            first, last = np.asarray(jpeg_offsets[frame_row : frame_row + 2], dtype=np.uint64)
            encoded = np.asarray(jpeg_data[int(first) : int(last)], dtype=np.uint8).tobytes()
            image_path = image_dir / frame_names[local_index]
            _atomic_bytes(image_path, encoded)
            image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not decode frame_row={frame_row} from {h5_path}")
            current_wh = np.asarray([image.shape[1], image.shape[0]], dtype=np.int32)
            if image_wh is None:
                image_wh = current_wh
            elif not np.array_equal(image_wh, current_wh):
                raise RuntimeError("Dyn-HaMR trial frames do not share one image resolution")

            mapped_hands = []
            for hand in (0, 1):
                record = table[frame_row][hand]
                if record is None or int(record["status"]) != VALID_STATUS:
                    continue
                vertices = np.asarray(record["vertices_camera"], dtype=np.float32)
                joints = _joints_from_vertices(vertices, regressor, fingertip_indices)
                focal = float(record["focal_length"])
                uv = _project(joints, focal, np.asarray(record["image_wh"]))
                if not np.isfinite(uv).all():
                    continue
                pose = np.stack(
                    [_rotation_matrix_to_axis_angle(matrix) for matrix in record["hand_pose"]],
                    axis=0,
                )
                orient = _rotation_matrix_to_axis_angle(record["global_orient"][0])
                track_dir = track_root / f"{hand:03d}"
                # The runtime compatibility patch mirrors canonical left-hand
                # geometry before adding this shared camera/world translation,
                # matching HaMeR's full-image coordinate convention.
                dynhamr_translation = np.asarray(
                    record["camera_translation"], dtype=np.float32
                ).copy()
                mano_record = {
                    "betas": np.asarray(record["betas"], dtype=np.float32).tolist(),
                    "body_pose": pose.tolist(),
                    "global_orient": orient.tolist(),
                    "cam_trans": dynhamr_translation.tolist(),
                    "is_right": hand,
                }
                keypoints = np.concatenate(
                    (uv, np.ones((21, 1), dtype=np.float32)), axis=1
                )
                keypoint_record = {
                    "people": [{"pose_keypoints_2d": keypoints.reshape(-1).tolist()}]
                }
                stem = Path(frame_names[local_index]).stem
                _atomic_json(track_dir / f"{stem}_mano.json", mano_record)
                _atomic_json(track_dir / f"{stem}_keypoints.json", keypoint_record)
                raw_joints2d[hand, local_index] = uv
                raw_vertices[hand, local_index] = vertices
                raw_body_pose[hand, local_index] = pose
                raw_root_orient[hand, local_index] = orient
                raw_translation[hand, local_index] = dynhamr_translation
                raw_betas[hand, local_index] = np.asarray(record["betas"], dtype=np.float32)
                visibility[hand, local_index] = True
                focal_values.append(focal)
                mapped_hands.append("right" if hand else "left")
            mapping.append(
                {
                    "local_frame": local_index,
                    "image": frame_names[local_index],
                    "source_frame_row": frame_row,
                    "source_frame_idx": int(source_frame_indices[local_index]),
                    "timestamp": float(timestamps[local_index]),
                    "valid_hands": mapped_hands,
                }
            )

    if image_wh is None or not focal_values:
        raise RuntimeError("Prepared trial contains no valid frame geometry")
    focal = float(np.median(np.asarray(focal_values, dtype=np.float64)))
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], args.num_frames, axis=0)
    intrins = np.repeat(
        np.asarray([[focal, focal, image_wh[0] * 0.5, image_wh[1] * 0.5]], np.float32),
        args.num_frames,
        axis=0,
    )
    camera_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        camera_dir / "cameras.npz",
        w2c=w2c,
        intrins=intrins,
        width=np.int32(image_wh[0]),
        height=np.int32(image_wh[1]),
        focal=np.float32(focal),
    )
    _atomic_json(shot_path, {name: 0 for name in frame_names})
    _atomic_json(output_root / "frame_mapping.json", {"frames": mapping})
    np.savez_compressed(
        output_root / "hamer_reference.npz",
        joints2d=raw_joints2d,
        vertices_camera=raw_vertices.astype(np.float16),
        pose_body=raw_body_pose,
        root_orient=raw_root_orient,
        trans=raw_translation,
        betas=raw_betas,
        visibility=visibility,
        intrins=intrins,
        source_frame_indices=source_frame_indices,
        timestamps=timestamps,
    )
    video_path = output_root / "videos" / f"{sequence_name}.mp4"
    if not args.skip_video:
        _encode_video(image_dir, video_path, args.fps)
    summary = {
        "schema": "touchanything_dynhamr_prepare_done",
        "schema_version": "1.0.0",
        "request": request,
        "selected_start": start,
        "selected_stop": stop,
        "selection": selection,
        "frame_count": int(args.num_frames),
        "valid_left": int(visibility[0].sum()),
        "valid_right": int(visibility[1].sum()),
        "image_wh": image_wh.tolist(),
        "focal_length": focal,
        "h5_path": str(h5_path),
        "images": str(image_dir),
        "tracks": str(track_root),
        "cameras": str(camera_dir),
        "shots": str(shot_path),
        "video": str(video_path) if video_path.is_file() else None,
        "size_bytes": sum(
            path.stat().st_size for path in output_root.rglob("*") if path.is_file()
        ),
    }
    _atomic_json(output_root / "PREPARE_DONE.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", default=str(DEFAULT_SIDECAR))
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--mano", default=str(DEFAULT_MANO))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence-key", default=DEFAULT_SEQUENCE)
    parser.add_argument("--sequence-name", default="ta_arrange_pillow_dynhamr")
    parser.add_argument("--start", default="auto")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--skip-video", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_frames < 61:
        raise ValueError("Dyn-HaMR rejects tracks of 60 frames or fewer")
    prepare(args)


if __name__ == "__main__":
    main()
