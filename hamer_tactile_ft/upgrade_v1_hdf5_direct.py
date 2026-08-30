#!/usr/bin/env python3
"""Upgrade sequence HDF5 v1 archives directly from original sequence sources.

This bypasses the temporary per-frame folder representation. Source datasets
are read-only; each v1 target is replaced atomically only after the v2 file has
been completed and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

try:
    from convert_sequence_hdf5 import (
        bounded_ordered_map,
        canonical_dataset_name,
        load_existing_train_query_overlay,
        load_palm_assets,
        parse_jpeg_hw,
    )
    from hdf5_storage import (
        AtomicJsonlWriter,
        SCHEMA_VERSION,
        SequenceHDF5Writer,
        canonical_json,
        manifest_rows_from_hdf5,
        sequence_manifest_row,
        sha256_file,
        verify_sequence_hdf5,
        write_json_atomic,
    )
    from process_lifecycle import initialize_worker_parent_death_signal
except ImportError:
    from .convert_sequence_hdf5 import (
        bounded_ordered_map,
        canonical_dataset_name,
        load_existing_train_query_overlay,
        load_palm_assets,
        parse_jpeg_hw,
    )
    from .hdf5_storage import (
        AtomicJsonlWriter,
        SCHEMA_VERSION,
        SequenceHDF5Writer,
        canonical_json,
        manifest_rows_from_hdf5,
        sequence_manifest_row,
        sha256_file,
        verify_sequence_hdf5,
        write_json_atomic,
    )
    from .process_lifecycle import initialize_worker_parent_death_signal


DATASETS = ("opentouch", "egotactile")
CONTINUOUS_KEYS = (
    "{hand}_pressure_continuous_subdiv",
    "{hand}_pressure_continuous",
)


def _initialize_upgrade_worker() -> None:
    initialize_worker_parent_death_signal()
    cv2.setNumThreads(1)


@dataclass(frozen=True)
class UpgradeTask:
    dataset: str
    split: str
    sequence_key: str
    target_h5: str
    processed_root: str
    source_root: str
    pressure_compression: str
    deep_verify: bool
    palm_mask: np.ndarray
    mesh_sha256: str
    palm_faces_sha256: str
    ego_npz_name: str


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _jpeg_bytes(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value, dtype=np.uint8).tobytes()
    if isinstance(value, np.void):
        return bytes(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError(f"Unsupported encoded JPEG value type: {type(value).__name__}")


def _load_existing_frame_overlay(h5_path: Path) -> dict[int, bytes]:
    if not h5_path.is_file():
        return {}
    payloads: dict[int, bytes] = {}
    with h5py.File(h5_path, "r") as handle:
        frame_indices = np.asarray(handle["frames/frame_idx"][:], dtype=np.int64)
        offsets = np.asarray(
            handle["images/rgb/jpeg_offsets"][:], dtype=np.uint64
        )
        data = handle["images/rgb/jpeg_data"]
        for row, frame_idx in enumerate(frame_indices):
            start, end = int(offsets[row]), int(offsets[row + 1])
            payloads[int(frame_idx)] = _jpeg_bytes(data[start:end])
    return payloads


def _validate_overlay_bounds(
    *,
    target_h5: Path,
    frame_count: int,
    query_overlay: dict[tuple[int, str], dict[str, Any]],
    frame_overlay: dict[int, bytes],
) -> None:
    invalid_queries = sorted(
        key
        for key in query_overlay
        if key[0] < 0 or key[0] >= frame_count or key[1] not in ("left", "right")
    )
    invalid_frames = sorted(
        frame_idx
        for frame_idx in frame_overlay
        if frame_idx < 0 or frame_idx >= frame_count
    )
    if invalid_queries or invalid_frames:
        raise RuntimeError(
            f"{target_h5}: v1 rows do not align with the original sequence; "
            f"invalid queries={invalid_queries[:10]}, "
            f"invalid frames={invalid_frames[:10]}, frame_count={frame_count}"
        )


def _file_fingerprint(paths: list[Path], extra: dict[str, Any]) -> str:
    rows = []
    for path in paths:
        stat = path.stat()
        rows.append(
            {
                "path": str(path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return hashlib.sha256(
        canonical_json({"files": rows, "extra": extra}).encode("utf-8")
    ).hexdigest()


def _pressure_from_dataset(group: h5py.Group, hand: str, frame_idx: int):
    for template in CONTINUOUS_KEYS:
        key = template.format(hand=hand)
        if key not in group:
            continue
        dataset = group[key]
        if dataset.ndim < 2 or frame_idx >= dataset.shape[0]:
            continue
        values = np.asarray(dataset[frame_idx], dtype=np.float32)
        if values.shape == (13614,) and np.isfinite(values).all():
            return np.clip(values, 0.0, 1.0), key
    return None, ""


def _frame_timestamp(group: h5py.Group, frame_idx: int) -> float | None:
    for key in ("timestamps", "timestamp"):
        if key not in group:
            continue
        dataset = group[key]
        if dataset.ndim == 0 or frame_idx >= dataset.shape[0]:
            continue
        try:
            value = float(np.asarray(dataset[frame_idx]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(value):
            return value
    return None


def _pressure_from_npz(npz, hand: str, frame_idx: int):
    if npz is None:
        return None, ""
    for template in CONTINUOUS_KEYS:
        key = template.format(hand=hand)
        if key not in npz or frame_idx >= npz[key].shape[0]:
            continue
        values = np.asarray(npz[key][frame_idx], dtype=np.float32)
        if values.shape == (13614,) and np.isfinite(values).all():
            return np.clip(values, 0.0, 1.0), key
    return None, ""


def _query_records(
    *,
    frame_count: int,
    source_pressure,
    overlay: dict[tuple[int, str], dict[str, Any]],
    preserve_overlay_pressure: bool = True,
) -> list[dict[str, Any]]:
    records = []
    for frame_idx in range(frame_count):
        for alias in ("left", "right"):
            existing = overlay.get((frame_idx, alias))
            pressure, pressure_key = source_pressure(alias, frame_idx)
            if existing is not None and preserve_overlay_pressure:
                pressure = np.asarray(existing["pressure"], dtype=np.float32)
                pressure_key = existing.get("pressure_source_key") or pressure_key
            if pressure is None:
                continue
            records.append(
                {
                    "frame_idx": frame_idx,
                    "alias": alias,
                    "is_trainable": existing is not None,
                    "pressure": pressure,
                    "pressure_key": pressure_key,
                    "bbox": (
                        np.asarray(existing["bbox"], dtype=np.float32)
                        if existing is not None
                        else np.full(4, np.nan, dtype=np.float32)
                    ),
                    "bbox_score": (
                        float(existing["bbox_score"]) if existing is not None else 0.0
                    ),
                    "bbox_source": (
                        existing.get("bbox_source") if existing is not None else None
                    ),
                    "keypoints": (
                        np.asarray(existing["keypoints_3d_cam"], dtype=np.float32)
                        if existing is not None
                        else np.zeros((21, 3), dtype=np.float32)
                    ),
                    "keypoints_valid": (
                        np.asarray(existing["keypoints_valid"], dtype=np.uint8)
                        if existing is not None
                        else np.zeros(21, dtype=np.uint8)
                    ),
                }
            )
    return records


def _write_queries(
    writer: SequenceHDF5Writer,
    records: list[dict[str, Any]],
    task: UpgradeTask,
) -> None:
    for row, record in enumerate(records):
        pressure = record["pressure"]
        palm = pressure[task.palm_mask]
        alias = record["alias"]
        frame_idx = record["frame_idx"]
        writer.write_query(
            row,
            frame_row=frame_idx,
            frame_idx=frame_idx,
            hand_code=1 if alias == "right" else 0,
            is_right=1 if alias == "right" else 0,
            bbox_xyxy=record["bbox"],
            bbox_score=record["bbox_score"],
            keypoints_3d_cam=record["keypoints"],
            keypoints_valid=record["keypoints_valid"],
            query_uid=(
                f"{canonical_dataset_name(task.dataset)}/{task.split}/"
                f"{task.sequence_key}/{frame_idx:08d}/{alias}"
            ),
            query_alias=alias,
            source_sample_relpath="",
            bbox_source=record["bbox_source"],
            pressure_source_key=record["pressure_key"],
            pressure=pressure,
            max_pressure=float(palm.max()),
            volume=float(palm.sum()),
            active_count=int(np.count_nonzero(palm >= 0.05)),
            is_trainable=record["is_trainable"],
        )


def _copy_hdf5_group_children(
    source: h5py.Group, destination: h5py.Group, *, skip: set[str]
) -> None:
    for name in source:
        if name in skip:
            continue
        source.file.copy(source[name], destination, name=name)
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _upgrade_opentouch(task: UpgradeTask) -> None:
    parts = task.sequence_key.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid OpenTouch sequence key: {task.sequence_key!r}")
    scene, clip = parts
    source_root = Path(task.source_root)
    source_path = source_root / f"{scene}.hdf5"
    if not source_path.is_file():
        source_path = source_root / f"{scene}.h5"
    if not source_path.is_file():
        raise FileNotFoundError(f"OpenTouch source scene is missing: {source_path}")

    target_h5 = Path(task.target_h5)
    overlay = load_existing_train_query_overlay(target_h5)
    frame_overlay = _load_existing_frame_overlay(target_h5)
    with h5py.File(source_path, "r", swmr=True) as source:
        group_path = f"data/{clip}"
        if group_path not in source:
            raise KeyError(f"{source_path}: missing {group_path}")
        group = source[group_path]
        jpeg_dataset = group["rgb_images_jpeg"]
        frame_count = len(jpeg_dataset)
        if frame_count <= 0:
            raise RuntimeError(f"{source_path}:{group_path} contains no RGB frames")
        _validate_overlay_bounds(
            target_h5=target_h5,
            frame_count=frame_count,
            query_overlay=overlay,
            frame_overlay=frame_overlay,
        )
        metadata = [
            _json_bytes(
                {
                    "dataset": "OpenTouch",
                    "split": task.split,
                    "scene": scene,
                    "demo": clip,
                    "frame_idx": row,
                    "source_hdf5": str(source_path),
                    "source_group": group_path,
                    "archive_only": True,
                }
            )
            for row in range(frame_count)
        ]
        records = _query_records(
            frame_count=frame_count,
            source_pressure=lambda hand, row: _pressure_from_dataset(
                group, hand, row
            ),
            overlay=overlay,
        )
        fingerprint = _file_fingerprint(
            [source_path],
            {"group": group_path, "v1": str(target_h5.resolve())},
        )
        with SequenceHDF5Writer(
            target_h5,
            dataset="OpenTouch",
            split=task.split,
            sequence_key=task.sequence_key,
            frame_count=frame_count,
            query_count=len(records),
            jpeg_total_bytes=None,
            metadata_record_count=frame_count,
            metadata_total_bytes=sum(map(len, metadata)),
            source_fingerprint=fingerprint,
            pressure_compression=task.pressure_compression,
            extra_attrs={
                "mesh_sha256": task.mesh_sha256,
                "palm_faces_sha256": task.palm_faces_sha256,
                "archive_scope": "complete_source_sequence",
                "archive_sequence_complete": True,
                "archive_source_kind": "opentouch_hdf5",
                "archive_frame_index_contract": "contiguous_zero_based",
                "upgrade_overlay": "v1_train_queries",
            },
        ) as writer:
            for row in range(frame_count):
                payload = frame_overlay.get(row)
                if payload is None:
                    payload = _jpeg_bytes(jpeg_dataset[row])
                timestamp = _frame_timestamp(group, row)
                writer.write_frame(
                    row,
                    frame_idx=row,
                    source_frame_idx=row,
                    timestamp=timestamp,
                    timestamp_kind=1 if timestamp is not None else 0,
                    image_hw=parse_jpeg_hw(payload),
                    jpeg_bytes=payload,
                )
                writer.write_metadata_record(
                    row,
                    frame_row=row,
                    query_alias="",
                    source_sample_relpath="",
                    meta_json_bytes=metadata[row],
                )
            _write_queries(writer, records, task)
            archive = writer.file.require_group("archive")
            copied = archive.create_group("source_clip")
            _copy_hdf5_group_children(
                group, copied, skip={"rgb_images_jpeg"}
            )
            if "calibration/rgb" in source:
                calibration = archive.create_group("source_calibration")
                _copy_hdf5_group_children(
                    source["calibration/rgb"], calibration, skip=set()
                )


def _load_ego_frames(data_json: Path) -> list[dict[str, Any]]:
    text = data_json.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("frames", "data", "records"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"{data_json}: could not find frame list")


def _upgrade_egotactile(task: UpgradeTask) -> None:
    sequence_dir = Path(task.source_root) / task.sequence_key
    data_json = sequence_dir / "data.json"
    video_path = sequence_dir / "video.mp4"
    npz_path = sequence_dir / task.ego_npz_name
    if not data_json.is_file() or not video_path.is_file():
        raise FileNotFoundError(
            f"EgoTactile source files are missing under {sequence_dir}"
        )
    frames = _load_ego_frames(data_json)
    if not frames:
        raise RuntimeError(f"{data_json}: sequence contains no frame records")
    target_h5 = Path(task.target_h5)
    overlay = load_existing_train_query_overlay(target_h5)
    frame_overlay = _load_existing_frame_overlay(target_h5)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(len(frames), video_count if video_count > 0 else len(frames))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"{video_path}: sequence contains no decodable frames")
    try:
        _validate_overlay_bounds(
            target_h5=target_h5,
            frame_count=frame_count,
            query_overlay=overlay,
            frame_overlay=frame_overlay,
        )
    except BaseException:
        cap.release()
        raise
    with tempfile.TemporaryFile(prefix="tactile_ego_jpeg_") as jpeg_spool:
        jpeg_offsets = [0]
        jpeg_hw = []
        try:
            for row in range(frame_count):
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(f"{video_path}: decode failed at frame {row}")
                payload = frame_overlay.get(row)
                if payload is None:
                    ok, encoded = cv2.imencode(".jpg", image)
                    if not ok:
                        raise RuntimeError(
                            f"{video_path}: JPEG encode failed at frame {row}"
                        )
                    payload = encoded.tobytes()
                jpeg_spool.write(payload)
                jpeg_offsets.append(jpeg_offsets[-1] + len(payload))
                jpeg_hw.append((int(image.shape[0]), int(image.shape[1])))
        finally:
            cap.release()

        npz = np.load(npz_path) if npz_path.is_file() else None
        try:
            records = _query_records(
                frame_count=frame_count,
                source_pressure=lambda hand, row: _pressure_from_npz(
                    npz, hand, row
                ),
                overlay=overlay,
                preserve_overlay_pressure=False,
            )
        finally:
            if npz is not None:
                npz.close()
        trainable_frames = {
            item["frame_idx"] for item in records if item["is_trainable"]
        }
        metadata = [
            _json_bytes(
                {
                    "dataset": "EgoTactile",
                    "split": task.split,
                    "rel_seq": task.sequence_key,
                    "frame_idx": row,
                    "original_frame_record": frames[row],
                    "archive_only": row not in trainable_frames,
                }
            )
            for row in range(frame_count)
        ]
        source_files = [data_json, video_path]
        if npz_path.is_file():
            source_files.append(npz_path)
        fingerprint = _file_fingerprint(
            source_files, {"v1": str(target_h5.resolve())}
        )
        with SequenceHDF5Writer(
            target_h5,
            dataset="EgoTactile",
            split=task.split,
            sequence_key=task.sequence_key,
            frame_count=frame_count,
            query_count=len(records),
            jpeg_total_bytes=jpeg_offsets[-1],
            metadata_record_count=frame_count,
            metadata_total_bytes=sum(map(len, metadata)),
            source_fingerprint=fingerprint,
            pressure_compression=task.pressure_compression,
            extra_attrs={
                "mesh_sha256": task.mesh_sha256,
                "palm_faces_sha256": task.palm_faces_sha256,
                "archive_scope": "complete_source_sequence",
                "archive_sequence_complete": True,
                "archive_source_kind": "egotactile_video_json_npz",
                "archive_frame_index_contract": "contiguous_zero_based",
                "upgrade_overlay": "v1_train_queries",
            },
        ) as writer:
            for row in range(frame_count):
                start, end = jpeg_offsets[row], jpeg_offsets[row + 1]
                jpeg_spool.seek(start)
                payload = jpeg_spool.read(end - start)
                if len(payload) != end - start:
                    raise RuntimeError(
                        f"{video_path}: temporary JPEG spool truncated at frame {row}"
                    )
                timestamp = frames[row].get("timestamp", frames[row].get("ts"))
                try:
                    timestamp = float(timestamp)
                except (TypeError, ValueError):
                    timestamp = math.nan
                writer.write_frame(
                    row,
                    frame_idx=row,
                    source_frame_idx=row,
                    timestamp=timestamp if math.isfinite(timestamp) else None,
                    timestamp_kind=1 if math.isfinite(timestamp) else 0,
                    image_hw=jpeg_hw[row],
                    jpeg_bytes=payload,
                )
                writer.write_metadata_record(
                    row,
                    frame_row=row,
                    query_alias="",
                    source_sample_relpath="",
                    meta_json_bytes=metadata[row],
                )
            _write_queries(writer, records, task)
            archive = writer.file.require_group("archive")
            raw_json = data_json.read_bytes()
            archive.create_dataset(
                "source_data_json",
                data=np.frombuffer(raw_json, dtype=np.uint8),
                compression="lzf",
                shuffle=True,
            )
            archive.attrs["source_data_json_sha256"] = hashlib.sha256(
                raw_json
            ).hexdigest()
            if npz_path.is_file():
                raw_npz = np.memmap(npz_path, mode="r", dtype=np.uint8)
                try:
                    archive.create_dataset(
                        "source_npz_file",
                        data=raw_npz,
                        chunks=(min(4 * 1024 * 1024, max(1, raw_npz.size)),),
                        fletcher32=True,
                    )
                finally:
                    del raw_npz
                archive.attrs["source_npz_name"] = npz_path.name


def _schema_version(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        return _decode_text(handle.attrs.get("schema_version", "1.0.0"))


def _v1_backup_path(target_h5: Path) -> Path:
    return target_h5.with_name(f".{target_h5.name}.v1-backup")


def _ensure_v1_backup(target_h5: Path) -> Path:
    backup = _v1_backup_path(target_h5)
    if not target_h5.exists() and backup.exists():
        os.replace(backup, target_h5)
    if not target_h5.is_file():
        raise FileNotFoundError(target_h5)
    if backup.exists():
        if _schema_version(backup) != "1.0.0":
            raise RuntimeError(f"Unexpected non-v1 rollback file: {backup}")
        return backup
    os.link(target_h5, backup)
    return backup


def _restore_v1_backup(target_h5: Path, backup: Path) -> None:
    if not backup.exists():
        return
    target_h5.unlink(missing_ok=True)
    os.replace(backup, target_h5)


def upgrade_task(task: UpgradeTask) -> dict[str, Any]:
    target_h5 = Path(task.target_h5)
    backup = _v1_backup_path(target_h5)
    if not target_h5.exists() and backup.exists():
        os.replace(backup, target_h5)
    current_version = _schema_version(target_h5)
    if current_version == SCHEMA_VERSION:
        verify_sequence_hdf5(target_h5, deep=task.deep_verify)
        backup.unlink(missing_ok=True)
        status = "already_v2"
    else:
        if current_version != "1.0.0":
            raise RuntimeError(
                f"{target_h5}: expected v1 or v2, got {current_version!r}"
            )
        backup = _ensure_v1_backup(target_h5)
        try:
            if task.dataset == "opentouch":
                _upgrade_opentouch(task)
            else:
                _upgrade_egotactile(task)
            verify_sequence_hdf5(target_h5, deep=task.deep_verify)
        except BaseException:
            _restore_v1_backup(target_h5, backup)
            raise
        backup.unlink(missing_ok=True)
        status = "upgraded"
    return {
        "status": status,
        "query_rows": manifest_rows_from_hdf5(target_h5, task.processed_root),
        "sequence_row": sequence_manifest_row(target_h5, task.processed_root),
    }


def _manifest_paths(root: Path, dataset: str, split: str):
    prefix = root / "manifests" / f"{dataset}_{split}"
    return (
        Path(f"{prefix}.queries.jsonl"),
        Path(f"{prefix}.sequences.jsonl"),
        Path(f"{prefix}.summary.json"),
    )


def _discover_manifest_splits(root: Path, dataset: str) -> list[str]:
    splits = set()
    manifest_dir = root / "manifests"
    if manifest_dir.is_dir():
        prefix = f"{dataset}_"
        suffix = ".sequences.jsonl"
        splits.update(
            name[len(prefix) : -len(suffix)]
            for name in os.listdir(manifest_dir)
            if name.startswith(prefix) and name.endswith(suffix)
        )
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False) or entry.name == "manifests":
                continue
            directory = Path(entry.path)
            if next(directory.rglob("*.h5"), None) is not None:
                splits.add(entry.name)
    return sorted(splits)


def _load_sequence_rows(root: Path, dataset: str, split: str) -> list[dict[str, Any]]:
    _, sequence_manifest, _ = _manifest_paths(root, dataset, split)
    if sequence_manifest.is_file():
        rows = []
        with sequence_manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("dataset") != canonical_dataset_name(dataset):
                    raise RuntimeError(
                        f"{sequence_manifest}:{line_number}: dataset mismatch"
                    )
                rows.append(row)
        return rows

    candidates = sorted((root / split).rglob("*.h5"))
    rows = []
    for target in candidates:
        with h5py.File(target, "r") as handle:
            found_dataset = _decode_text(handle.attrs.get("dataset", ""))
            found_split = _decode_text(handle.attrs.get("split", ""))
            if found_dataset != canonical_dataset_name(dataset) or found_split != split:
                continue
            rows.append(
                {
                    "dataset": found_dataset,
                    "split": found_split,
                    "sequence_key": _decode_text(handle.attrs["sequence_key"]),
                    "h5_relpath": target.relative_to(root).as_posix(),
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"No {dataset}/{split} sequence manifest or HDF5 files under {root}"
        )
    return rows


def load_tasks(args, split: str, palm_mask, mesh_sha, palm_faces_sha):
    root = Path(args.processed_root)
    tasks = []
    for row in _load_sequence_rows(root, args.dataset, split):
        target = (root / row["h5_relpath"]).resolve()
        target.relative_to(root)
        tasks.append(
            UpgradeTask(
                dataset=args.dataset,
                split=split,
                sequence_key=row["sequence_key"],
                target_h5=str(target),
                processed_root=str(root),
                source_root=str(Path(args.source_root).resolve()),
                pressure_compression=args.compression,
                deep_verify=args.deep_verify_after_write,
                palm_mask=palm_mask,
                mesh_sha256=mesh_sha,
                palm_faces_sha256=palm_faces_sha,
                ego_npz_name=args.ego_npz_name,
            )
        )
    if not tasks:
        raise RuntimeError(f"No {args.dataset}/{split} v1 sequence rows")
    return tasks


def process_split(args, split, tasks):
    root = Path(args.processed_root)
    query_manifest, sequence_manifest, summary_path = _manifest_paths(
        root, args.dataset, split
    )
    query_writer = AtomicJsonlWriter(query_manifest)
    sequence_writer = AtomicJsonlWriter(sequence_manifest)
    counts = {}
    query_count = 0
    started = time.monotonic()
    with query_writer, sequence_writer:
        kwargs = {
            "max_workers": args.workers,
            "initializer": _initialize_upgrade_worker,
        }
        with ProcessPoolExecutor(**kwargs) as executor:
            results = bounded_ordered_map(
                executor, upgrade_task, tasks, max_pending=args.workers * 2
            )
            for completed, result in enumerate(results, start=1):
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                for row in result["query_rows"]:
                    query_writer.write(row)
                    query_count += 1
                sequence_writer.write(result["sequence_row"])
                if completed % 10 == 0 or completed == len(tasks):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"[{split}] {completed}/{len(tasks)} sequences "
                        f"({completed / elapsed:.2f} seq/s), queries={query_count}",
                        flush=True,
                    )
    summary = {
        "schema": "tactile_direct_v1_upgrade_summary_v1",
        "schema_version": SCHEMA_VERSION,
        "dataset_key": args.dataset,
        "dataset": canonical_dataset_name(args.dataset),
        "split": split,
        "source_root": str(Path(args.source_root).resolve()),
        "processed_root": str(root),
        "sequence_count": len(tasks),
        "query_count": query_count,
        "status_counts": counts,
        "query_manifest_sha256": sha256_file(query_manifest),
        "sequence_manifest_sha256": sha256_file(sequence_manifest),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(summary_path, summary)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Directly upgrade OpenTouch/EgoTactile v1 sequence HDF5 to v2."
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--splits", default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compression", choices=("lzf", "gzip1"), default="lzf")
    parser.add_argument(
        "--ego-npz-name", default="pressure_grids_egotactile.npz"
    )
    parser.add_argument("--deep-verify-after-write", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    processed_root = Path(args.processed_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    if not processed_root.is_dir() or not source_root.is_dir():
        raise FileNotFoundError(
            f"processed/source root missing: {processed_root}, {source_root}"
        )
    args.processed_root = str(processed_root)
    args.source_root = str(source_root)
    splits = (
        _discover_manifest_splits(processed_root, args.dataset)
        if args.splits.strip().lower() == "auto"
        else [value.strip() for value in args.splits.split(",") if value.strip()]
    )
    if not splits:
        raise RuntimeError("No v1 manifest splits were found")
    palm_mask, mesh_sha, palm_faces_sha = load_palm_assets()
    print(
        f"Direct v1 upgrade: dataset={args.dataset}, splits={splits}, "
        f"workers={args.workers}",
        flush=True,
    )
    for split in splits:
        tasks = load_tasks(args, split, palm_mask, mesh_sha, palm_faces_sha)
        process_split(args, split, tasks)
    print("Direct v1 upgrade complete.", flush=True)


if __name__ == "__main__":
    main()
