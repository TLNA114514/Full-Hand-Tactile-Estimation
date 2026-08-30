#!/usr/bin/env python3
"""Build train-only AceData sequence HDF5 containers from current SAM3 boxes.

The legacy ``samples/all`` tree embeds obsolete detector boxes, so it is used
only as a lossless JPEG cache. Query identity, trainability, pressure and bbox
provenance are rebuilt from the synchronized NPZ files and active SAM3 output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from hamer_tactile_ft.data.hdf5_backend import (  # noqa: E402
    AtomicJsonlWriter,
    SequenceHDF5Writer,
    canonical_json,
    manifest_rows_from_hdf5,
    sequence_manifest_row,
    sha256_file,
    verify_sequence_hdf5,
    write_json_atomic,
)


BUILDER_SCHEMA = "acedata_sequence_hdf5_builder_v2"
BBOX_SOURCE_SCHEMA = "sam3_bbox_source_v1"
DATASET_NAME = "AceData"
SPLIT = "train"
TACTILE_DIM = 13614
PRESSURE_KEYS = {
    "left": "left_pressure_continuous_subdiv",
    "right": "right_pressure_continuous_subdiv",
}
VALID_KEYS = {
    "left": "left_sensor_valid",
    "right": "right_sensor_valid",
}


def _atomic_json(path: Path, value: Any) -> None:
    write_json_atomic(path, value)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _safe_sequence_parts(sequence_key: str) -> tuple[str, str]:
    parts = Path(sequence_key).parts
    if len(parts) != 2 or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid AceData sequence_key={sequence_key!r}")
    return str(parts[0]), str(parts[1])


def _valid_bbox(value: Any) -> bool:
    try:
        bbox = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return bool(
        bbox.shape == (4,)
        and np.isfinite(bbox).all()
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _palm_vertex_mask() -> np.ndarray:
    faces_path = (
        WORKSPACE_ROOT
        / "opentouch"
        / "preprocess"
        / "scratch"
        / "auto_calibrated_palm_subdiv_faces.json"
    )
    payload = _read_json(faces_path)
    mask = np.zeros(TACTILE_DIM, dtype=bool)
    for face in payload["group_negative"]["face_triplets"]:
        for vertex in face:
            if 0 <= int(vertex) < TACTILE_DIM:
                mask[int(vertex)] = True
    if not mask.any():
        raise RuntimeError(f"Palm mask is empty: {faces_path}")
    return mask


def _mesh_paths() -> tuple[Path, Path]:
    scratch = WORKSPACE_ROOT / "opentouch" / "preprocess" / "scratch"
    return (
        scratch / "mano_right_neutral_subdiv.obj",
        scratch / "auto_calibrated_palm_subdiv_faces.json",
    )


def _source_fingerprint(
    row: dict[str, Any],
    *,
    video_path: Path,
    pressure_path: Path,
    bbox_path: Path,
    provenance_path: Path,
    source_manifest_sha256: str,
    pressure_compression: str,
    jpeg_quality: int,
    image_source: str,
) -> str:
    value = {
        "schema": BUILDER_SCHEMA,
        "sequence_key": str(row["sequence_key"]),
        "video": _file_stat(video_path),
        "pressure": _file_stat(pressure_path),
        "bbox_sha256": sha256_file(bbox_path),
        "bbox_provenance_sha256": sha256_file(provenance_path),
        "source_manifest_sha256": source_manifest_sha256,
        "pressure_compression": pressure_compression,
        "jpeg_quality": int(jpeg_quality),
        "image_source": str(image_source),
        "missing_query_policy": "exclude_if_sensor_or_sam3_missing",
        "image_policy": (
            "sequential_source_video_decode"
            if image_source == "video"
            else "reuse_legacy_full_frame_jpeg_else_decode_source_video"
        ),
    }
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sequence_paths(
    row: dict[str, Any], processed_root: Path, output_root: Path
) -> dict[str, Path]:
    subject, clip = _safe_sequence_parts(str(row["sequence_key"]))
    return {
        "video": Path(row["resource_path"]).expanduser().resolve(),
        "pressure": Path(row["pressure_path"]).expanduser().resolve(),
        "bbox": processed_root / "bboxes" / subject / f"{clip}.json",
        "provenance": (
            processed_root / "bboxes" / ".provenance" / subject / f"{clip}.json"
        ),
        "h5": output_root / SPLIT / subject / f"{clip}.h5",
    }


def _legacy_sample_dir(
    processed_root: Path, subject: str, clip: str, frame_index: int, hand: str
) -> Path:
    return (
        processed_root
        / "samples"
        / "all"
        / f"{subject}__{clip}__{frame_index:06d}__{hand}"
    )


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


class _SequentialVideoReader:
    def __init__(self, path: Path, jpeg_quality: int):
        import cv2

        cv2.setNumThreads(1)
        self.cv2 = cv2
        self.path = path
        self.quality = int(jpeg_quality)
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open AceData video: {path}")
        self.current_index = -1

    def jpeg_at(self, target_index: int) -> bytes:
        if target_index <= self.current_index:
            raise RuntimeError(
                f"Video fallback requested out of order: {target_index} <= {self.current_index}"
            )
        frame = None
        while self.current_index < target_index:
            ok, frame = self.capture.read()
            self.current_index += 1
            if not ok or frame is None:
                raise RuntimeError(
                    f"Could not decode frame {self.current_index} from {self.path}"
                )
        ok, encoded = self.cv2.imencode(
            ".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        if not ok:
            raise RuntimeError(f"Could not JPEG-encode frame {target_index} from {self.path}")
        return encoded.tobytes()

    def close(self) -> None:
        self.capture.release()


@dataclass(frozen=True)
class BuildTask:
    row: dict[str, Any]
    processed_root: str
    output_root: str
    source_manifest: str
    source_manifest_sha256: str
    mode: str
    pressure_compression: str
    jpeg_quality: int
    image_source: str
    deep_verify: bool


def _load_sequence_arrays(
    pressure_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    pressures: dict[str, np.ndarray] = {}
    validity: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    with np.load(pressure_path, allow_pickle=False) as archive:
        missing = [
            key
            for key in (*PRESSURE_KEYS.values(), *VALID_KEYS.values())
            if key not in archive
        ]
        if missing:
            raise KeyError(f"{pressure_path}: missing arrays {missing}")
        for hand in ("left", "right"):
            pressure = np.asarray(archive[PRESSURE_KEYS[hand]], dtype=np.float32)
            valid = np.asarray(archive[VALID_KEYS[hand]], dtype=bool)
            if pressure.ndim != 2 or pressure.shape[1] != TACTILE_DIM:
                raise ValueError(
                    f"{pressure_path}: {PRESSURE_KEYS[hand]} has shape {pressure.shape}"
                )
            if valid.shape != (pressure.shape[0],):
                raise ValueError(
                    f"{pressure_path}: {VALID_KEYS[hand]} has shape {valid.shape}"
                )
            pressures[hand] = pressure
            validity[hand] = valid
        for key in (
            "frame_count",
            "fps",
            "video_width",
            "video_height",
            "normalization_max_N",
            "normalization_formula",
            "sync_formula",
            "mapping_left_sha256",
            "mapping_right_sha256",
        ):
            if key in archive:
                value = np.asarray(archive[key])
                metadata[key] = value.item() if value.ndim == 0 else value.tolist()
    return pressures, validity, metadata


def _bbox_source(
    *,
    task: BuildTask,
    bbox_path: Path,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": BBOX_SOURCE_SCHEMA,
        "materialization_schema": str(provenance.get("schema", "")),
        "prompt_preset": str(provenance.get("prompt_preset", "gloved")),
        "association_policy": str(provenance.get("association_policy", "")),
        "association_confidence": "high",
        "source_manifest": task.source_manifest,
        "source_manifest_sha256": task.source_manifest_sha256,
        "source_bbox_jsonl": str(provenance.get("tracking_bbox_jsonl", "")),
        "source_bbox_jsonl_sha256": str(
            provenance.get("tracking_bbox_jsonl_sha256", "")
        ),
        "source_materialized_bbox": str(bbox_path),
        "source_materialized_bbox_sha256": sha256_file(bbox_path),
        "missing_frames_are_excluded": True,
    }


def _eligible_queries(
    boxes: dict[str, Any],
    validity: dict[str, np.ndarray],
    frame_count: int,
) -> tuple[list[int], list[tuple[int, str]]]:
    frame_indices = []
    query_specs = []
    for frame_index in range(frame_count):
        frame_boxes = boxes.get(str(frame_index), {})
        has_query = False
        for hand in ("left", "right"):
            hand_box = frame_boxes.get(hand, {}) if isinstance(frame_boxes, dict) else {}
            if bool(validity[hand][frame_index]) and _valid_bbox(hand_box.get("bbox")):
                query_specs.append((frame_index, hand))
                has_query = True
        if has_query:
            frame_indices.append(frame_index)
    return frame_indices, query_specs


def build_sequence(task: BuildTask) -> dict[str, Any]:
    processed_root = Path(task.processed_root)
    output_root = Path(task.output_root)
    row = task.row
    subject, clip = _safe_sequence_parts(str(row["sequence_key"]))
    paths = _sequence_paths(row, processed_root, output_root)
    for key in ("video", "pressure", "bbox", "provenance"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])
    fingerprint = _source_fingerprint(
        row,
        video_path=paths["video"],
        pressure_path=paths["pressure"],
        bbox_path=paths["bbox"],
        provenance_path=paths["provenance"],
        source_manifest_sha256=task.source_manifest_sha256,
        pressure_compression=task.pressure_compression,
        jpeg_quality=task.jpeg_quality,
        image_source=task.image_source,
    )
    h5_path = paths["h5"]
    if h5_path.is_file() and task.mode in ("resume", "verify"):
        result = verify_sequence_hdf5(
            h5_path,
            expected_source_fingerprint=fingerprint,
            expected_pressure_compression=task.pressure_compression,
            deep=task.deep_verify,
        )
        result["status"] = "verified" if task.mode == "verify" else "resumed"
        return result
    if task.mode == "verify":
        raise FileNotFoundError(h5_path)
    if h5_path.exists() and task.mode == "create":
        raise FileExistsError(f"{h5_path} exists; use resume or overwrite")

    pressures, validity, pressure_metadata = _load_sequence_arrays(paths["pressure"])
    frame_count = int(pressures["left"].shape[0])
    if pressures["right"].shape[0] != frame_count:
        raise ValueError(f"{paths['pressure']}: left/right frame counts differ")
    expected_frame_count = int(row["frame_count"])
    if frame_count != expected_frame_count:
        raise ValueError(
            f"{row['sequence_key']}: pressure={frame_count}, manifest={expected_frame_count}"
        )
    boxes = _read_json(paths["bbox"])
    if len(boxes) != frame_count:
        raise ValueError(
            f"{paths['bbox']}: bbox rows={len(boxes)}, expected={frame_count}"
        )
    provenance = _read_json(paths["provenance"])
    if provenance.get("association_policy") != "initial_screen_order":
        raise ValueError(f"{paths['provenance']}: unexpected association policy")
    frame_indices, query_specs = _eligible_queries(boxes, validity, frame_count)
    if not frame_indices or not query_specs:
        raise RuntimeError(f"{row['sequence_key']}: no eligible SAM3 + pressure queries")
    frame_row_by_source = {
        source_frame_index: frame_row
        for frame_row, source_frame_index in enumerate(frame_indices)
    }
    bbox_source = _bbox_source(
        task=task, bbox_path=paths["bbox"], provenance=provenance
    )
    fps = float(pressure_metadata.get("fps", row.get("fps", 0.0)))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{row['sequence_key']}: invalid fps={fps}")
    height = int(pressure_metadata.get("video_height", row.get("frame_height", 0)))
    width = int(pressure_metadata.get("video_width", row.get("frame_width", 0)))
    if height <= 0 or width <= 0:
        raise ValueError(f"{row['sequence_key']}: invalid video dimensions {width}x{height}")
    metadata_value = {
        "schema": BUILDER_SCHEMA,
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "sequence_key": str(row["sequence_key"]),
        "source_frame_count": frame_count,
        "stored_frame_count": len(frame_indices),
        "train_query_count": len(query_specs),
        "missing_query_policy": "exclude_if_sensor_or_sam3_missing",
        "image_source": task.image_source,
        "pressure_metadata": pressure_metadata,
        "bbox_source": bbox_source,
    }
    metadata_bytes = canonical_json(metadata_value).encode("utf-8")
    palm_mask = _palm_vertex_mask()
    zero_keypoints = np.zeros((21, 3), dtype=np.float32)
    zero_keypoint_validity = np.zeros(21, dtype=np.uint8)
    extra_attrs = {
        "mesh_sha256": sha256_file(_mesh_paths()[0]),
        "palm_faces_sha256": sha256_file(_mesh_paths()[1]),
        "target_construction": "acedata_pressure_mean_float32_normalized_5N",
        "image_storage": (
            "source_video_sequential_jpeg_concat_offsets"
            if task.image_source == "video"
            else "legacy_jpeg_or_source_video_jpeg_concat_offsets"
        ),
        "archive_scope": "eligible_query_frames_only",
        "archive_frame_index_contract": "contiguous_rows_source_index_preserved",
        "archive_sequence_complete": False,
        "metadata_storage": "generated_sequence_provenance_json",
        "source_frame_count": frame_count,
        "missing_query_policy": "exclude_if_sensor_or_sam3_missing",
        "bbox_materialization_sha256": bbox_source[
            "source_materialized_bbox_sha256"
        ],
        "bbox_tracking_sha256": bbox_source["source_bbox_jsonl_sha256"],
        "mapping_left_sha256": str(pressure_metadata.get("mapping_left_sha256", "")),
        "mapping_right_sha256": str(pressure_metadata.get("mapping_right_sha256", "")),
    }
    video_reader: _SequentialVideoReader | None = None
    legacy_jpeg_frames = 0
    video_encoded_frames = 0
    started_at = time.monotonic()
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with SequenceHDF5Writer(
            h5_path,
            dataset=DATASET_NAME,
            split=SPLIT,
            sequence_key=str(row["sequence_key"]),
            frame_count=len(frame_indices),
            query_count=len(query_specs),
            jpeg_total_bytes=None,
            metadata_record_count=1,
            metadata_total_bytes=len(metadata_bytes),
            source_fingerprint=fingerprint,
            pressure_compression=task.pressure_compression,
            extra_attrs=extra_attrs,
        ) as writer:
            for frame_row, source_frame_index in enumerate(frame_indices):
                jpeg_bytes = None
                if task.image_source == "legacy_jpeg":
                    image_candidates = (
                        _legacy_sample_dir(
                            processed_root, subject, clip, source_frame_index, "left"
                        )
                        / "image.jpg",
                        _legacy_sample_dir(
                            processed_root, subject, clip, source_frame_index, "right"
                        )
                        / "image.jpg",
                    )
                    image_path = next(
                        (
                            candidate
                            for candidate in image_candidates
                            if candidate.is_file()
                        ),
                        None,
                    )
                    if image_path is not None:
                        jpeg_bytes = image_path.read_bytes()
                        legacy_jpeg_frames += 1
                if jpeg_bytes is None:
                    if video_reader is None:
                        video_reader = _SequentialVideoReader(
                            paths["video"], task.jpeg_quality
                        )
                    jpeg_bytes = video_reader.jpeg_at(source_frame_index)
                    video_encoded_frames += 1
                if len(jpeg_bytes) < 4 or jpeg_bytes[:2] != b"\xff\xd8":
                    raise ValueError(
                        f"{row['sequence_key']} frame={source_frame_index}: "
                        "source image is not a JPEG payload"
                    )
                writer.write_frame(
                    frame_row,
                    frame_idx=frame_row,
                    source_frame_idx=source_frame_index,
                    timestamp=source_frame_index / fps,
                    timestamp_kind=1,
                    image_hw=(height, width),
                    jpeg_bytes=jpeg_bytes,
                )

            writer.write_metadata_record(
                0,
                frame_row=0,
                query_alias="",
                source_sample_relpath=(
                    f"{subject}/{clip}#generated_sequence_provenance"
                ),
                meta_json_bytes=metadata_bytes,
            )

            for query_row, (source_frame_index, hand) in enumerate(query_specs):
                pressure = np.asarray(
                    pressures[hand][source_frame_index], dtype=np.float32
                )
                if not np.isfinite(pressure).all():
                    raise ValueError(
                        f"{row['sequence_key']} frame={source_frame_index} hand={hand}: "
                        "non-finite pressure"
                    )
                minimum = float(pressure.min())
                maximum = float(pressure.max())
                if minimum < -1e-6 or maximum > 1.0 + 1e-6:
                    raise ValueError(
                        f"{row['sequence_key']} frame={source_frame_index} hand={hand}: "
                        f"pressure outside [0,1]: [{minimum},{maximum}]"
                    )
                pressure = np.clip(pressure, 0.0, 1.0)
                palm_pressure = pressure[palm_mask]
                box_record = boxes[str(source_frame_index)][hand]
                source_sample = (
                    f"source_video/{subject}/{clip}/stereo1.mp4"
                    f"#frame={source_frame_index}"
                )
                if task.image_source == "legacy_jpeg":
                    sample_dir = _legacy_sample_dir(
                        processed_root, subject, clip, source_frame_index, hand
                    )
                    if sample_dir.is_dir():
                        source_sample = _relative_or_absolute(
                            sample_dir, processed_root
                        )
                uid = (
                    f"AceData/train/{subject}/{clip}/"
                    f"{source_frame_index:08d}/{hand}"
                )
                writer.write_query(
                    query_row,
                    frame_row=frame_row_by_source[source_frame_index],
                    frame_idx=source_frame_index,
                    hand_code=1 if hand == "right" else 0,
                    is_right=1 if hand == "right" else 0,
                    bbox_xyxy=np.asarray(box_record["bbox"], dtype=np.float32),
                    bbox_score=float(box_record.get("score", 0.0) or 0.0),
                    keypoints_3d_cam=zero_keypoints,
                    keypoints_valid=zero_keypoint_validity,
                    query_uid=uid,
                    query_alias=hand,
                    source_sample_relpath=source_sample,
                    bbox_source=bbox_source,
                    pressure_source_key=PRESSURE_KEYS[hand],
                    pressure=pressure,
                    max_pressure=float(palm_pressure.max()),
                    volume=float(palm_pressure.sum(dtype=np.float64)),
                    active_count=int(np.count_nonzero(palm_pressure >= 0.05)),
                    is_trainable=True,
                )
    finally:
        if video_reader is not None:
            video_reader.close()
        del pressures

    result = verify_sequence_hdf5(
        h5_path,
        expected_source_fingerprint=fingerprint,
        expected_pressure_compression=task.pressure_compression,
        deep=task.deep_verify,
    )
    elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
    result.update(
        {
            "status": "converted",
            "legacy_jpeg_frames": legacy_jpeg_frames,
            "video_encoded_frames": video_encoded_frames,
            "elapsed_seconds": elapsed_seconds,
            "stored_frames_per_second": len(frame_indices) / elapsed_seconds,
        }
    )
    return result


def _selected_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path, str]:
    manifest = args.manifest.expanduser().resolve()
    rows = _read_jsonl(manifest)
    if args.expected_sequences and len(rows) != args.expected_sequences:
        raise RuntimeError(
            f"Expected {args.expected_sequences} AceData sequences in {manifest}, "
            f"found {len(rows)}"
        )
    seen = set()
    for row in rows:
        sequence_key = str(row.get("sequence_key", ""))
        _safe_sequence_parts(sequence_key)
        if sequence_key in seen:
            raise ValueError(f"Duplicate sequence in {manifest}: {sequence_key}")
        seen.add(sequence_key)
    if args.only_sequence:
        requested = set(args.only_sequence)
        missing = requested - seen
        if missing:
            raise ValueError(f"Unknown --only-sequence values: {sorted(missing)}")
        rows = [row for row in rows if row["sequence_key"] in requested]
    if args.max_sequences:
        rows = rows[: args.max_sequences]
    return rows, manifest, sha256_file(manifest)


def _tasks(args: argparse.Namespace) -> tuple[list[BuildTask], Path, str]:
    rows, manifest, manifest_sha256 = _selected_rows(args)
    tasks = [
        BuildTask(
            row=row,
            processed_root=str(args.processed_root.expanduser().resolve()),
            output_root=str(args.output_root.expanduser().resolve()),
            source_manifest=str(manifest),
            source_manifest_sha256=manifest_sha256,
            mode=args.mode,
            pressure_compression=args.pressure_compression,
            jpeg_quality=args.jpeg_quality,
            image_source=args.image_source,
            deep_verify=args.deep_verify,
        )
        for row in rows
    ]
    return tasks, manifest, manifest_sha256


def _publish_manifests(
    tasks: Iterable[BuildTask],
    manifest: Path,
    manifest_sha256: str,
    expected_queries: int,
) -> dict[str, Any]:
    tasks = list(tasks)
    if not tasks:
        raise ValueError("Cannot publish an empty AceData manifest")
    output_root = Path(tasks[0].output_root).resolve()
    manifest_dir = output_root / "manifests"
    query_path = manifest_dir / "acedata_train.queries.jsonl"
    sequence_path = manifest_dir / "acedata_train.sequences.jsonl"
    query_count = 0
    frame_count = 0
    with AtomicJsonlWriter(query_path) as query_writer, AtomicJsonlWriter(
        sequence_path
    ) as sequence_writer:
        for completed, task in enumerate(tasks, start=1):
            h5_path = _sequence_paths(
                task.row, Path(task.processed_root), output_root
            )["h5"]
            fingerprint = _source_fingerprint(
                task.row,
                video_path=Path(task.row["resource_path"]).expanduser().resolve(),
                pressure_path=Path(task.row["pressure_path"]).expanduser().resolve(),
                bbox_path=_sequence_paths(
                    task.row, Path(task.processed_root), output_root
                )["bbox"],
                provenance_path=_sequence_paths(
                    task.row, Path(task.processed_root), output_root
                )["provenance"],
                source_manifest_sha256=manifest_sha256,
                pressure_compression=task.pressure_compression,
                jpeg_quality=task.jpeg_quality,
                image_source=task.image_source,
            )
            verify_sequence_hdf5(
                h5_path,
                expected_source_fingerprint=fingerprint,
                expected_pressure_compression=task.pressure_compression,
                deep=False,
            )
            sequence_row = sequence_manifest_row(h5_path, output_root)
            sequence_writer.write(sequence_row)
            frame_count += int(sequence_row["frame_count"])
            rows = manifest_rows_from_hdf5(h5_path, output_root)
            for row in rows:
                query_writer.write(row)
            query_count += len(rows)
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"[AceData manifest] {completed}/{len(tasks)} sequences, "
                    f"queries={query_count}",
                    flush=True,
                )
        if expected_queries and query_count != expected_queries:
            raise RuntimeError(
                f"AceData train query contract failed: expected={expected_queries}, "
                f"observed={query_count}. Existing official manifests were preserved."
            )
    summary = {
        "schema": "acedata_train_hdf5_manifest_summary_v1",
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "train_only": True,
        "sequence_count": len(tasks),
        "stored_frame_count": frame_count,
        "query_count": query_count,
        "source_manifest": str(manifest),
        "source_manifest_sha256": manifest_sha256,
        "query_manifest": str(query_path),
        "query_manifest_sha256": sha256_file(query_path),
        "sequence_manifest": str(sequence_path),
        "sequence_manifest_sha256": sha256_file(sequence_path),
        "bbox_source_schema": BBOX_SOURCE_SCHEMA,
        "association_policy": "initial_screen_order",
        "missing_queries_are_excluded": True,
        "validation_manifest": None,
    }
    _atomic_json(manifest_dir / "acedata_train.summary.json", summary)
    _atomic_json(output_root / ".acedata_hdf5_complete.json", summary)
    return summary


def command_build(args: argparse.Namespace) -> None:
    tasks, manifest, manifest_sha256 = _tasks(args)
    if not tasks:
        raise RuntimeError("No AceData sequences selected")
    workers = min(max(int(args.workers), 1), len(tasks))
    failures = []
    counts: dict[str, int] = {}
    legacy_jpeg_frames = 0
    video_encoded_frames = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {executor.submit(build_sequence, task): task for task in tasks}
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                    status = str(result.get("status", "complete"))
                    counts[status] = counts.get(status, 0) + 1
                    legacy_jpeg_frames += int(
                        result.get("legacy_jpeg_frames", 0)
                    )
                    video_encoded_frames += int(
                        result.get("video_encoded_frames", 0)
                    )
                except Exception as exc:
                    failures.append(
                        f"{task.row.get('sequence_key')}: {type(exc).__name__}: {exc}"
                    )
                print(
                    f"[AceData HDF5] {completed}/{len(tasks)} "
                    f"status={counts} failures={len(failures)} "
                    f"legacy_jpeg_frames={legacy_jpeg_frames} "
                    f"video_encoded_frames={video_encoded_frames}",
                    flush=True,
                )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
    if failures:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            f"AceData HDF5 failed for {len(failures)}/{len(tasks)} sequences:\n{preview}"
        )
    full_selection = not args.only_sequence and not args.max_sequences
    if args.publish and not full_selection and not args.publish_partial:
        raise RuntimeError(
            "Refusing to publish an incomplete official train manifest; remove "
            "--max-sequences/--only-sequence or pass --publish-partial explicitly"
        )
    if args.publish:
        summary = _publish_manifests(
            tasks, manifest, manifest_sha256, args.expected_queries
        )
        print(canonical_json(summary), flush=True)


def command_publish(args: argparse.Namespace) -> None:
    tasks, manifest, manifest_sha256 = _tasks(args)
    if args.only_sequence or args.max_sequences:
        raise ValueError("publish requires the complete source manifest")
    print(
        canonical_json(
            _publish_manifests(
                tasks, manifest, manifest_sha256, args.expected_queries
            )
        ),
        flush=True,
    )


def command_status(args: argparse.Namespace) -> None:
    rows, manifest, manifest_sha256 = _selected_rows(args)
    output_root = args.output_root.expanduser().resolve()
    existing = 0
    missing = []
    query_count = 0
    for row in rows:
        path = _sequence_paths(row, args.processed_root, output_root)["h5"]
        if path.is_file():
            existing += 1
            try:
                import h5py

                with h5py.File(path, "r") as handle:
                    query_count += int(handle.attrs.get("train_query_count", 0))
            except Exception:
                pass
        else:
            missing.append(str(row["sequence_key"]))
    print(
        canonical_json(
            {
                "schema": BUILDER_SCHEMA,
                "source_manifest": str(manifest),
                "source_manifest_sha256": manifest_sha256,
                "expected_sequences": len(rows),
                "existing_sequences": existing,
                "missing_sequences": len(missing),
                "known_train_queries": query_count,
                "missing_preview": missing[:20],
                "published": (
                    output_root / "manifests" / "acedata_train.queries.jsonl"
                ).is_file(),
            }
        ),
        flush=True,
    )


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/home/ma-user/work/hy/acedata-sam3-reconstruction-v1/"
            "jobs/acedata_gloved_jobs.jsonl"
        ),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/home/ma-user/work/hy/acedata-processed"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ma-user/work/hy/acedata-processed-hdf5"),
    )
    parser.add_argument("--only-sequence", action="append", default=[])
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--expected-sequences", type=int, default=494)
    parser.add_argument("--expected-queries", type=int, default=1987236)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build/resume HDF5 and publish train manifests")
    _common_parser(build)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--mode", choices=("create", "resume", "overwrite", "verify"), default="resume")
    build.add_argument("--pressure-compression", choices=("lzf", "gzip1"), default="lzf")
    build.add_argument("--jpeg-quality", type=int, default=95)
    build.add_argument(
        "--image-source",
        choices=("video", "legacy_jpeg"),
        default="video",
        help=(
            "video sequentially decodes the canonical MP4 and avoids millions "
            "of shared-filesystem directory lookups"
        ),
    )
    build.add_argument("--deep-verify", action="store_true")
    build.add_argument("--publish", action=argparse.BooleanOptionalAction, default=True)
    build.add_argument("--publish-partial", action="store_true")
    build.set_defaults(func=command_build)

    publish = subparsers.add_parser("publish", help="Rebuild deterministic train manifests")
    _common_parser(publish)
    publish.add_argument("--mode", default="verify")
    publish.add_argument("--pressure-compression", choices=("lzf", "gzip1"), default="lzf")
    publish.add_argument("--jpeg-quality", type=int, default=95)
    publish.add_argument(
        "--image-source", choices=("video", "legacy_jpeg"), default="video"
    )
    publish.add_argument("--deep-verify", action="store_false")
    publish.set_defaults(func=command_publish)

    status = subparsers.add_parser("status", help="Report converted sequence coverage")
    _common_parser(status)
    status.set_defaults(func=command_status)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "max_sequences", 0) < 0:
        raise ValueError("--max-sequences cannot be negative")
    args.func(args)


if __name__ == "__main__":
    main()
