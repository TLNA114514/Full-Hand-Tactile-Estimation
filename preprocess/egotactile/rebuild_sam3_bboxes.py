#!/usr/bin/env python3
"""Build, audit, and activate a SAM3-bbox EgoTactile HDF5 dataset view."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SAM3_BBOX_SOURCE_SCHEMA = "sam3_bbox_source_v1"
MIGRATION_SCHEMA = "egotactile_sam3_hdf5_migration_v2"
REQUIRED_SPLITS = ("bare_hand", "gloved_hand")
MAX_INTERIOR_INTERPOLATION_GAP = 8
MAX_EDGE_EXTENSION_GAP = 4
OFFICIAL_SPLIT_SCHEMA = "egotactile_official_split_v1"
OFFICIAL_HELD_OUT_OBJECTS = (
    "Apple",
    "CocaCola-330ml",
    "Corn",
    "Dumbbell",
    "TennisBall",
)
OFFICIAL_HELD_OUT_SUBJECTS = ("p007", "p011")
OFFICIAL_PROTOCOLS = {
    "gloved_object_held_out": {
        "source_split": "gloved_hand",
        "held_out_field": "object_name",
        "held_out_values": OFFICIAL_HELD_OUT_OBJECTS,
        "paper_table_clip_counts": {"train": 638, "test": 55},
    },
    "gloved_subject_held_out": {
        "source_split": "gloved_hand",
        "held_out_field": "participant_id",
        "held_out_values": OFFICIAL_HELD_OUT_SUBJECTS,
        "paper_table_clip_counts": {"train": 630, "test": 63},
        "paper_definition_note": (
            "Appendix A.1.5 explicitly names p007 and p011. Its two-subject "
            "definition is authoritative here even though Table 3 reports clip "
            "counts corresponding to a one-subject test partition."
        ),
    },
    "bare_object_held_out": {
        "source_split": "bare_hand",
        "held_out_field": "object_name",
        "held_out_values": OFFICIAL_HELD_OUT_OBJECTS,
        "paper_table_clip_counts": {"train": 60, "test": 15},
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    payload = bytearray()
    count = 0
    for row in rows:
        payload.extend(canonical_json(row).encode("utf-8"))
        payload.extend(b"\n")
        count += 1
    atomic_write(path, bytes(payload))
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def load_frame_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("frames", "data", "records"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError(f"{path}: no frame list found")


def normalize_task_hand(value: Any) -> str | None:
    token = str(value or "").strip().lower()
    if token in {"l", "lh", "left"}:
        return "left"
    if token in {"r", "rh", "right"}:
        return "right"
    return None


def safe_job_id(sequence_key: str) -> str:
    readable = "__".join(Path(sequence_key).parts)
    readable = "".join(char if char.isalnum() or char in "._-" else "_" for char in readable)
    digest = hashlib.sha256(sequence_key.encode("utf-8")).hexdigest()[:12]
    return f"ego__{readable[:150]}__{digest}"


def discover_raw_sequences(raw_root: Path) -> list[Path]:
    paths = []
    for split in REQUIRED_SPLITS:
        split_root = raw_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        for data_json in split_root.rglob("data.json"):
            sequence_dir = data_json.parent
            if (sequence_dir / "video.mp4").is_file():
                paths.append(sequence_dir)
    return sorted(paths)


def command_build(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.expanduser().resolve()
    hdf5_root = args.hdf5_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records = []
    counts = Counter()
    frame_counts = Counter()
    hand_counts = Counter()
    for sequence_dir in discover_raw_sequences(raw_root):
        sequence_key = sequence_dir.relative_to(raw_root).as_posix()
        split = sequence_key.split("/", 1)[0]
        frames = load_frame_records(sequence_dir / "data.json")
        if not frames:
            raise RuntimeError(f"{sequence_dir}: empty data.json")
        task_hands = {normalize_task_hand(row.get("task_hand")) for row in frames}
        if None in task_hands or len(task_hands) != 1:
            raise RuntimeError(
                f"{sequence_dir}: task_hand must be constant left/right, got {task_hands}"
            )
        target_hand = next(iter(task_hands))
        pressure_npz = sequence_dir / "pressure_grids_egotactile.npz"
        if not pressure_npz.is_file():
            raise FileNotFoundError(
                f"Formal pressure archive is missing for {sequence_key}: {pressure_npz}"
            )
        validity_key = f"{target_hand}_sensor_valid"
        with np.load(pressure_npz, allow_pickle=False) as pressure_archive:
            if validity_key not in pressure_archive:
                raise KeyError(f"{pressure_npz}: missing {validity_key}")
            validity = np.asarray(pressure_archive[validity_key], dtype=bool).reshape(-1)
        if len(validity) != len(frames):
            raise RuntimeError(
                f"{sequence_key}: {validity_key} length={len(validity)}, "
                f"raw frames={len(frames)}"
            )
        trainable_frame_indices = np.flatnonzero(validity).astype(int).tolist()
        h5_relpath = f"{split}/{sequence_key}.h5"
        h5_path = hdf5_root / h5_relpath
        if not h5_path.is_file():
            raise FileNotFoundError(
                f"Current HDF5 sequence is missing for {sequence_key}: {h5_path}"
            )
        with h5py.File(h5_path, "r") as handle:
            h5_frame_count = int(handle.attrs["frame_count"])
            train_query_count = int(handle.attrs["train_query_count"])
            found_sequence = str(handle.attrs["sequence_key"])
            if found_sequence != sequence_key:
                raise RuntimeError(
                    f"{h5_path}: sequence_key={found_sequence!r}, expected={sequence_key!r}"
                )
        if h5_frame_count != len(frames):
            raise RuntimeError(
                f"{sequence_key}: raw/HDF5 frame counts differ: "
                f"{len(frames)}/{h5_frame_count}"
            )
        prompt_preset = "bare" if split == "bare_hand" else "gloved"
        prompt = "bare human hand" if prompt_preset == "bare" else "gloved hand"
        stat = h5_path.stat()
        row = {
            "schema": "egotactile_sam3_tracking_job_v1",
            "job_id": safe_job_id(sequence_key),
            "dataset": "egotactile",
            "tracker_dataset": "generic",
            "split": split,
            "sequence_key": sequence_key,
            "resource_path": str(sequence_dir / "video.mp4"),
            "resource_type": "video",
            "source_path": str(sequence_dir / "video.mp4"),
            "data_json": str(sequence_dir / "data.json"),
            "source_h5_relpath": h5_relpath,
            "source_h5_size": int(stat.st_size),
            "frame_count": len(frames),
            "old_train_query_count": train_query_count,
            "valid_pressure_frame_count": len(trainable_frame_indices),
            "trainable_frame_indices": trainable_frame_indices,
            "target_hand": target_hand,
            "expected_gloved_hands": 1,
            "prompt_preset": prompt_preset,
            "prompt": prompt,
        }
        records.append(row)
        counts[split] += 1
        frame_counts[split] += len(frames)
        hand_counts[f"{split}/{target_hand}"] += 1
    if len(records) != args.expected_clips:
        raise RuntimeError(
            f"Expected {args.expected_clips} EgoTactile clips, discovered {len(records)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "egotactile_jobs.jsonl"
    write_jsonl(master_path, records)
    split_paths = {}
    for split in REQUIRED_SPLITS:
        path = output_dir / f"egotactile_{split}_jobs.jsonl"
        write_jsonl(path, (row for row in records if row["split"] == split))
        split_paths[split] = str(path)
    summary = {
        "schema": "egotactile_sam3_tracking_manifest_summary_v1",
        "created_utc": utc_now(),
        "raw_root": str(raw_root),
        "source_hdf5_root": str(hdf5_root),
        "clip_count": len(records),
        "split_clip_counts": dict(counts),
        "split_frame_counts": dict(frame_counts),
        "task_hand_clip_counts": dict(hand_counts),
        "master_manifest": str(master_path),
        "master_manifest_sha256": sha256_file(master_path),
        "split_manifests": split_paths,
    }
    atomic_write_json(output_dir / "manifest_summary.json", summary)
    print(canonical_json(summary), flush=True)


def valid_bbox(value: Any) -> list[float] | None:
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(bbox) != 4 or not all(math.isfinite(item) for item in bbox):
        return None
    if bbox[2] <= bbox[0] + 1.0 or bbox[3] <= bbox[1] + 1.0:
        return None
    return bbox


def bbox_jump(previous: list[float], current: list[float]) -> float:
    px = (previous[0] + previous[2]) * 0.5
    py = (previous[1] + previous[3]) * 0.5
    cx = (current[0] + current[2]) * 0.5
    cy = (current[1] + current[3]) * 0.5
    area = max(
        1.0,
        (previous[2] - previous[0]) * (previous[3] - previous[1]),
        (current[2] - current[0]) * (current[3] - current[1]),
    )
    return math.hypot(cx - px, cy - py) / math.sqrt(area)


def tracking_job_dir(tracking_roots: list[Path], row: dict[str, Any]) -> Path:
    relative = Path("results") / row["dataset"] / row["split"] / row["job_id"]
    matches = [root / relative for root in tracking_roots if (root / relative).is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one result directory for {row['sequence_key']}, found {matches}"
        )
    return matches[0]


def missing_runs(missing: set[int]) -> list[tuple[int, int]]:
    if not missing:
        return []
    ordered = sorted(missing)
    runs = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def interpolate_tracking_gaps(
    boxes: dict[int, dict], expected_frames: int
) -> tuple[dict[int, dict], set[int], dict[str, Any]]:
    missing = set(range(expected_frames)) - set(boxes)
    audit = {
        "interpolated_frame_count": 0,
        "interior_interpolated_frame_count": 0,
        "edge_extended_frame_count": 0,
        "maximum_interpolated_gap": 0,
        "untracked_frame_count": 0,
        "untracked_frame_runs": [],
    }
    if not missing:
        return boxes, set(), audit
    untracked = set()
    for start, end in missing_runs(missing):
        length = end - start + 1
        previous_index = start - 1 if start > 0 and start - 1 in boxes else None
        next_index = end + 1 if end + 1 < expected_frames and end + 1 in boxes else None
        audit["maximum_interpolated_gap"] = max(
            audit["maximum_interpolated_gap"], length
        )
        if previous_index is not None and next_index is not None:
            if length > MAX_INTERIOR_INTERPOLATION_GAP:
                untracked.update(range(start, end + 1))
                audit["untracked_frame_runs"].append([start, end])
                continue
            previous = boxes[previous_index]
            following = boxes[next_index]
            for frame_index in range(start, end + 1):
                alpha = (frame_index - previous_index) / (next_index - previous_index)
                bbox = [
                    (1.0 - alpha) * left + alpha * right
                    for left, right in zip(previous["bbox"], following["bbox"])
                ]
                boxes[frame_index] = {
                    "bbox": bbox,
                    "bbox_score": min(
                        float(previous["bbox_score"]), float(following["bbox_score"])
                    ),
                    "raw_track_id": (
                        previous["raw_track_id"]
                        if previous["raw_track_id"] == following["raw_track_id"]
                        else -1
                    ),
                    "tracking_bbox_source": "sam3_temporal_interpolation",
                    "flow_confidence": None,
                    "flow_bbox_iou": None,
                    "flow_anchor_frames": [previous_index, next_index],
                }
            audit["interior_interpolated_frame_count"] += length
        else:
            if length > MAX_EDGE_EXTENSION_GAP:
                untracked.update(range(start, end + 1))
                audit["untracked_frame_runs"].append([start, end])
                continue
            anchor_index = previous_index if previous_index is not None else next_index
            if anchor_index is None:
                untracked.update(range(start, end + 1))
                audit["untracked_frame_runs"].append([start, end])
                continue
            anchor = boxes[anchor_index]
            for frame_index in range(start, end + 1):
                boxes[frame_index] = {
                    **anchor,
                    "bbox": list(anchor["bbox"]),
                    "tracking_bbox_source": "sam3_temporal_edge_extension",
                    "flow_anchor_frames": [anchor_index],
                }
            audit["edge_extended_frame_count"] += length
        audit["interpolated_frame_count"] += length
    audit["untracked_frame_count"] = len(untracked)
    return boxes, untracked, audit


def load_tracking_boxes(job_dir: Path, expected_frames: int) -> tuple[dict[int, dict], dict]:
    summary_path = job_dir / "summary.json"
    bbox_path = job_dir / "bboxes.jsonl"
    if not summary_path.is_file() or not bbox_path.is_file():
        raise FileNotFoundError(f"Incomplete SAM3 outputs under {job_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError(f"{job_dir}: SAM3 status={summary.get('status')!r}")
    boxes = {}
    track_ids = Counter()
    bbox_sources = Counter()
    multiple = []
    with bbox_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            frame = json.loads(line)
            frame_index = int(frame["frame_index"])
            tracks = frame.get("tracks") or []
            if not tracks:
                continue
            if len(tracks) != 1:
                multiple.append((frame_index, len(tracks)))
                continue
            track = tracks[0]
            bbox = valid_bbox(track.get("bbox"))
            if bbox is None:
                raise ValueError(f"{bbox_path}:{line_number}: invalid bbox")
            if frame_index in boxes:
                raise RuntimeError(f"{bbox_path}: duplicate frame {frame_index}")
            boxes[frame_index] = {
                "bbox": bbox,
                "bbox_score": float(track.get("prompt_score") or 0.0),
                "raw_track_id": int(track.get("track_id", -1)),
                "tracking_bbox_source": str(track.get("bbox_source", "sam3_native")),
                "flow_confidence": track.get("flow_confidence"),
                "flow_bbox_iou": track.get("flow_bbox_iou"),
                "flow_anchor_frames": list(track.get("flow_anchor_frames") or ()),
            }
            track_ids[boxes[frame_index]["raw_track_id"]] += 1
            bbox_sources[boxes[frame_index]["tracking_bbox_source"]] += 1
    expected = set(range(expected_frames))
    extras = sorted(set(boxes) - expected)
    if multiple or extras:
        raise RuntimeError(
            f"{job_dir}: invalid one-track coverage: boxes={len(boxes)}/"
            f"{expected_frames}, multiple={multiple[:10]}, extras={extras[:10]}"
        )
    boxes, untracked, gap_audit = interpolate_tracking_gaps(boxes, expected_frames)
    if set(boxes) | untracked != expected or set(boxes) & untracked:
        raise RuntimeError(f"{job_dir}: inconsistent tracked/untracked frame partition")
    track_ids = Counter(item["raw_track_id"] for item in boxes.values())
    bbox_sources = Counter(item["tracking_bbox_source"] for item in boxes.values())
    tracked_indices = sorted(boxes)
    jumps = [
        bbox_jump(boxes[left]["bbox"], boxes[right]["bbox"])
        for left, right in zip(tracked_indices, tracked_indices[1:])
        if right == left + 1
    ]
    metrics = {
        "frame_count": expected_frames,
        "tracked_frame_count": len(boxes),
        "tracked_frame_fraction": len(boxes) / max(expected_frames, 1),
        "track_id_count": len(track_ids),
        "track_ids": dict(track_ids),
        "bbox_source_counts": dict(bbox_sources),
        "mean_center_jump": float(np.mean(jumps)) if jumps else 0.0,
        "p95_center_jump": float(np.quantile(jumps, 0.95)) if jumps else 0.0,
        "maximum_center_jump": max(jumps, default=0.0),
        "summary_path": str(summary_path),
        "bbox_path": str(bbox_path),
        "bbox_sha256": sha256_file(bbox_path),
        **gap_audit,
    }
    return boxes, metrics


def load_jobs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        path = path.expanduser().resolve()
        manifest_sha256 = sha256_file(path)
        for source_row in read_jsonl(path):
            row = dict(source_row)
            row["_tracking_manifest"] = str(path)
            row["_tracking_manifest_sha256"] = manifest_sha256
            identity = (row["split"], row["sequence_key"])
            if identity in seen:
                raise RuntimeError(f"Duplicate tracking job {identity}")
            seen.add(identity)
            rows.append(row)
    return sorted(rows, key=lambda row: (row["split"], row["sequence_key"]))


def audit_tracking(
    manifest_paths: list[Path],
    tracking_roots: list[Path],
    output_dir: Path,
    *,
    expected_clips: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[int, dict]]]:
    jobs = load_jobs(manifest_paths)
    if len(jobs) != expected_clips:
        raise RuntimeError(f"Expected {expected_clips} jobs, found {len(jobs)}")
    metrics = []
    boxes_by_sequence = {}
    failures = []
    for completed, row in enumerate(jobs, start=1):
        try:
            job_dir = tracking_job_dir(tracking_roots, row)
            boxes, job_metrics = load_tracking_boxes(job_dir, int(row["frame_count"]))
            boxes_by_sequence[(row["split"], row["sequence_key"])] = boxes
            metrics.append(
                {
                    "status": "complete",
                    "split": row["split"],
                    "sequence_key": row["sequence_key"],
                    "prompt_preset": row["prompt_preset"],
                    "target_hand": row["target_hand"],
                    **job_metrics,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "status": "failed",
                    "split": row["split"],
                    "sequence_key": row["sequence_key"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if completed % 25 == 0 or completed == len(jobs):
            print(
                f"[SAM3 audit] {completed}/{len(jobs)} clips, failures={len(failures)}",
                flush=True,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = metrics + failures
    fields = sorted({key for row in all_rows for key in row})
    csv_path = output_dir / "sequence_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(temporary, csv_path)
    total_frames = sum(int(row["frame_count"]) for row in metrics)
    tracked_frames = sum(int(row["tracked_frame_count"]) for row in metrics)
    summary = {
        "schema": "egotactile_sam3_tracking_audit_v1",
        "created_utc": utc_now(),
        "job_count": len(jobs),
        "complete_clip_count": len(metrics),
        "failed_clip_count": len(failures),
        "frame_count": total_frames,
        "tracked_frame_count": tracked_frames,
        "untracked_frame_count": total_frames - tracked_frames,
        "tracked_frame_fraction": tracked_frames / max(total_frames, 1),
        "split_clip_counts": dict(Counter(row["split"] for row in metrics)),
        "split_frame_counts": {
            split: sum(
                int(row["frame_count"])
                for row in metrics
                if row["split"] == split
            )
            for split in REQUIRED_SPLITS
        },
        "maximum_p95_center_jump": max(
            (float(row["p95_center_jump"]) for row in metrics), default=None
        ),
        "clips_with_untracked_frames": sum(
            int(row.get("untracked_frame_count", 0)) > 0 for row in metrics
        ),
        "failures": failures,
        "manifest_sha256": {
            str(path.resolve()): sha256_file(path.resolve()) for path in manifest_paths
        },
    }
    atomic_write_json(output_dir / "summary.json", summary)
    if failures:
        raise RuntimeError(
            f"SAM3 tracking audit failed for {len(failures)}/{len(jobs)} clips; "
            f"see {output_dir / 'summary.json'}"
        )
    return jobs, boxes_by_sequence


def command_audit(args: argparse.Namespace) -> None:
    audit_tracking(
        args.manifests,
        args.tracking_roots,
        args.output_dir.expanduser().resolve(),
        expected_clips=args.expected_clips,
    )
    print(f"SAM3 tracking audit passed: {args.output_dir}", flush=True)


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _migrate_h5_task(task: dict[str, Any]) -> dict[str, Any]:
    source = Path(task["source"])
    target = Path(task["target"])
    boxes = {int(key): value for key, value in task["boxes"].items()}
    trainable_frames = {int(value) for value in task["trainable_frame_indices"]}
    target_hand = str(task["target_hand"])
    marker_hash = str(task["tracking_manifest_sha256"])
    if target.is_file():
        try:
            with h5py.File(target, "r") as handle:
                if (
                    _decode_text(handle.attrs.get("bbox_migration_schema", ""))
                    == MIGRATION_SCHEMA
                    and _decode_text(handle.attrs.get("bbox_tracking_manifest_sha256", ""))
                    == marker_hash
                ):
                    return {
                        "status": "resumed",
                        "target": str(target),
                        "tracked_frames": int(
                            handle.attrs.get("bbox_tracked_frame_count", 0)
                        ),
                        "untracked_frames": int(
                            handle.attrs.get("bbox_untracked_frame_count", 0)
                        ),
                        "trainable_queries": int(
                            handle.attrs.get("train_query_count", 0)
                        ),
                    }
        except OSError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.name}.sam3.", suffix=".tmp", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        with h5py.File(temporary, "r+") as archive:
            queries = archive["queries"]
            frame_indices = np.asarray(queries["frame_idx"][:], dtype=np.int64)
            aliases = np.asarray(
                [_decode_text(value) for value in queries["query_alias"][:]],
                dtype=object,
            )
            target_rows = np.flatnonzero(aliases == target_hand)
            target_query_count = len(target_rows)
            queries["is_trainable"][:] = np.zeros(len(frame_indices), dtype=np.uint8)
            updated_frames = set()
            usable_frames = set()
            for query_row in target_rows:
                frame_index = int(frame_indices[query_row])
                item = boxes.get(frame_index)
                bbox_source = {
                    "schema": SAM3_BBOX_SOURCE_SCHEMA,
                    "association_policy": "egotactile_task_hand_single_track",
                    "association_confidence": "known_task_hand" if item else "untracked",
                    "raw_track_id": item["raw_track_id"] if item else -1,
                    "tracking_bbox_source": (
                        item["tracking_bbox_source"] if item else "sam3_untracked"
                    ),
                    "flow_confidence": item["flow_confidence"] if item else None,
                    "flow_bbox_iou": item["flow_bbox_iou"] if item else None,
                    "flow_anchor_frames": item["flow_anchor_frames"] if item else [],
                    "prompt_preset": task["prompt_preset"],
                    "source_manifest": task["tracking_manifest"],
                    "source_manifest_sha256": marker_hash,
                    "source_bbox_jsonl_sha256": task["bbox_sha256"],
                }
                if item is None:
                    queries["bbox_xyxy"][query_row] = np.full(4, np.nan, dtype=np.float32)
                    queries["bbox_score"][query_row] = 0.0
                else:
                    queries["bbox_xyxy"][query_row] = np.asarray(
                        item["bbox"], dtype=np.float32
                    )
                    queries["bbox_score"][query_row] = float(item["bbox_score"])
                queries["bbox_source_json"][query_row] = canonical_json(bbox_source)
                is_trainable = item is not None and frame_index in trainable_frames
                queries["is_trainable"][query_row] = int(is_trainable)
                if is_trainable:
                    usable_frames.add(frame_index)
                updated_frames.add(frame_index)
            if not set(boxes).issubset(updated_frames):
                raise RuntimeError(f"{source}: SAM frames do not match HDF5 frame identities")
            archive.attrs["bbox_migration_schema"] = MIGRATION_SCHEMA
            archive.attrs["bbox_tracking_manifest_sha256"] = marker_hash
            archive.attrs["bbox_prompt_preset"] = str(task["prompt_preset"])
            archive.attrs["bbox_updated_utc"] = utc_now()
            archive.attrs["train_query_count"] = len(usable_frames)
            archive.attrs["bbox_tracked_frame_count"] = len(boxes)
            archive.attrs["bbox_untracked_frame_count"] = target_query_count - len(boxes)
            archive.flush()
        with h5py.File(temporary, "r") as archive:
            queries = archive["queries"]
            aliases = np.asarray(
                [_decode_text(value) for value in queries["query_alias"][:]],
                dtype=object,
            )
            target_rows = aliases == target_hand
            trainable = np.asarray(queries["is_trainable"][:], dtype=bool)
            trainable_bboxes = np.asarray(
                queries["bbox_xyxy"][:], dtype=np.float32
            )[trainable]
            if not np.isfinite(trainable_bboxes).all():
                raise RuntimeError(f"{temporary}: migrated trainable bboxes are invalid")
            expected_trainable = len(set(boxes) & trainable_frames)
            if int(trainable.sum()) != expected_trainable:
                raise RuntimeError(f"{temporary}: trainable query count is inconsistent")
            for source_text in queries["bbox_source_json"][:][target_rows]:
                provenance = json.loads(_decode_text(source_text))
                if provenance.get("schema") != SAM3_BBOX_SOURCE_SCHEMA:
                    raise RuntimeError(f"{temporary}: stale trainable bbox provenance")
        os.replace(temporary, target)
        return {
            "status": "migrated",
            "target": str(target),
            "tracked_frames": len(boxes),
            "untracked_frames": target_query_count - len(boxes),
            "trainable_queries": len(set(boxes) & trainable_frames),
        }
    finally:
        temporary.unlink(missing_ok=True)


def rebuild_hdf5_manifests(output_root: Path) -> dict[str, Any]:
    from hamer_tactile_ft.data.hdf5_backend import (
        manifest_rows_from_hdf5,
        sequence_manifest_row,
    )

    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split in REQUIRED_SPLITS:
        h5_paths = sorted((output_root / split).rglob("*.h5"))
        query_rows = []
        sequence_rows = []
        for completed, h5_path in enumerate(h5_paths, start=1):
            query_rows.extend(manifest_rows_from_hdf5(h5_path, output_root))
            sequence_rows.append(sequence_manifest_row(h5_path, output_root))
            if completed % 50 == 0 or completed == len(h5_paths):
                print(
                    f"[manifest {split}] {completed}/{len(h5_paths)} HDF5 sequences, "
                    f"queries={len(query_rows)}",
                    flush=True,
                )
        query_path = manifest_dir / f"egotactile_{split}.queries.jsonl"
        sequence_path = manifest_dir / f"egotactile_{split}.sequences.jsonl"
        write_jsonl(query_path, query_rows)
        write_jsonl(sequence_path, sequence_rows)
        summary = {
            "schema": "egotactile_sam3_manifest_summary_v1",
            "dataset": "EgoTactile",
            "split": split,
            "sequence_count": len(sequence_rows),
            "query_count": len(query_rows),
            "query_manifest_sha256": sha256_file(query_path),
            "sequence_manifest_sha256": sha256_file(sequence_path),
            "bbox_source_schema": SAM3_BBOX_SOURCE_SCHEMA,
        }
        atomic_write_json(manifest_dir / f"egotactile_{split}.summary.json", summary)
        result[split] = summary
    return result


def _official_sequence_identity(
    row: dict[str, Any],
    *,
    expected_source_split: str,
    location: str,
) -> tuple[str, str, str]:
    sequence_key = str(row.get("sequence_key", "")).strip()
    parts = Path(sequence_key).parts
    if len(parts) != 4:
        raise ValueError(
            f"{location}: expected sequence_key="
            "<source_split>/<participant>/<object>/<repeat>, "
            f"got {sequence_key!r}"
        )
    source_split, participant_id, object_name, _ = parts
    row_split = str(row.get("source_split", row.get("split", ""))).strip()
    if source_split != expected_source_split or row_split != expected_source_split:
        raise ValueError(
            f"{location}: source split mismatch: key={source_split!r}, "
            f"row={row_split!r}, expected={expected_source_split!r}"
        )
    if not participant_id.startswith("p") or not participant_id[1:].isdigit():
        raise ValueError(f"{location}: invalid participant id {participant_id!r}")
    if not object_name:
        raise ValueError(f"{location}: object name is empty")
    return sequence_key, participant_id, object_name


def _official_partition(
    participant_id: str,
    object_name: str,
    protocol: dict[str, Any],
) -> str:
    field = protocol["held_out_field"]
    value = participant_id if field == "participant_id" else object_name
    return "test" if value in protocol["held_out_values"] else "train"


def _official_manifest_row(
    row: dict[str, Any],
    *,
    protocol_name: str,
    source_split: str,
    partition: str,
    participant_id: str,
    object_name: str,
) -> dict[str, Any]:
    updated = dict(row)
    updated.update(
        {
            "source_split": source_split,
            "split": partition,
            "official_split_schema": OFFICIAL_SPLIT_SCHEMA,
            "official_protocol": protocol_name,
            "official_partition": partition,
            "participant_id": participant_id,
            "object_name": object_name,
        }
    )
    return updated


def _build_one_official_protocol(
    *,
    manifest_dir: Path,
    output_dir: Path,
    protocol_name: str,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    from hamer_tactile_ft.data.hdf5_backend import AtomicJsonlWriter

    source_split = str(protocol["source_split"])
    source_query_path = manifest_dir / f"egotactile_{source_split}.queries.jsonl"
    source_sequence_path = manifest_dir / f"egotactile_{source_split}.sequences.jsonl"
    if not source_query_path.is_file() or not source_sequence_path.is_file():
        raise FileNotFoundError(
            f"Official split source manifests are missing for {source_split}: "
            f"{source_query_path}, {source_sequence_path}"
        )

    protocol_dir = output_dir / protocol_name
    protocol_dir.mkdir(parents=True, exist_ok=True)
    partition_paths = {
        partition: {
            "queries": protocol_dir / f"{partition}.queries.jsonl",
            "sequences": protocol_dir / f"{partition}.sequences.jsonl",
        }
        for partition in ("train", "test")
    }
    sequence_partition: dict[str, str] = {}
    sequence_identity: dict[str, tuple[str, str]] = {}
    partition_sequence_keys: dict[str, set[str]] = {
        "train": set(),
        "test": set(),
    }
    partition_stats = {
        partition: {
            "sequence_count": 0,
            "query_count": 0,
            "frame_count": 0,
            "archive_query_count": 0,
            "objects": set(),
            "participants": set(),
        }
        for partition in ("train", "test")
    }

    with ExitStack() as stack:
        sequence_writers = {
            partition: stack.enter_context(
                AtomicJsonlWriter(partition_paths[partition]["sequences"])
            )
            for partition in ("train", "test")
        }
        for line_number, row in iter_jsonl(source_sequence_path):
            location = f"{source_sequence_path}:{line_number}"
            sequence_key, participant_id, object_name = _official_sequence_identity(
                row,
                expected_source_split=source_split,
                location=location,
            )
            if sequence_key in sequence_partition:
                raise RuntimeError(f"{location}: duplicate sequence_key {sequence_key!r}")
            partition = _official_partition(participant_id, object_name, protocol)
            sequence_partition[sequence_key] = partition
            sequence_identity[sequence_key] = (participant_id, object_name)
            partition_sequence_keys[partition].add(sequence_key)
            stats = partition_stats[partition]
            stats["sequence_count"] += 1
            stats["frame_count"] += int(row.get("frame_count", 0))
            stats["archive_query_count"] += int(row.get("archive_query_count", 0))
            stats["objects"].add(object_name)
            stats["participants"].add(participant_id)
            sequence_writers[partition].write(
                _official_manifest_row(
                    row,
                    protocol_name=protocol_name,
                    source_split=source_split,
                    partition=partition,
                    participant_id=participant_id,
                    object_name=object_name,
                )
            )

    overlap = partition_sequence_keys["train"] & partition_sequence_keys["test"]
    if overlap:
        raise RuntimeError(
            f"{protocol_name}: train/test sequence overlap: {sorted(overlap)[:5]}"
        )
    if not all(partition_sequence_keys.values()):
        raise RuntimeError(f"{protocol_name}: train and test must both be non-empty")

    seen_queries: set[str] = set()
    with ExitStack() as stack:
        query_writers = {
            partition: stack.enter_context(
                AtomicJsonlWriter(partition_paths[partition]["queries"])
            )
            for partition in ("train", "test")
        }
        for line_number, row in iter_jsonl(source_query_path):
            location = f"{source_query_path}:{line_number}"
            sequence_key, participant_id, object_name = _official_sequence_identity(
                row,
                expected_source_split=source_split,
                location=location,
            )
            if sequence_key not in sequence_partition:
                raise RuntimeError(
                    f"{location}: query references unknown sequence {sequence_key!r}"
                )
            if sequence_identity[sequence_key] != (participant_id, object_name):
                raise RuntimeError(f"{location}: query/sequence identity mismatch")
            sample_uid = str(row.get("sample_uid", "")).strip()
            if not sample_uid:
                raise ValueError(f"{location}: sample_uid is missing")
            if sample_uid in seen_queries:
                raise RuntimeError(f"{location}: duplicate sample_uid {sample_uid!r}")
            seen_queries.add(sample_uid)
            partition = sequence_partition[sequence_key]
            partition_stats[partition]["query_count"] += 1
            query_writers[partition].write(
                _official_manifest_row(
                    row,
                    protocol_name=protocol_name,
                    source_split=source_split,
                    partition=partition,
                    participant_id=participant_id,
                    object_name=object_name,
                )
            )

    if sum(item["query_count"] for item in partition_stats.values()) != len(seen_queries):
        raise RuntimeError(f"{protocol_name}: query partition accounting is inconsistent")

    serialized_stats = {}
    for partition, stats in partition_stats.items():
        query_path = partition_paths[partition]["queries"]
        sequence_path = partition_paths[partition]["sequences"]
        serialized_stats[partition] = {
            "sequence_count": int(stats["sequence_count"]),
            "usable_query_count": int(stats["query_count"]),
            "frame_count": int(stats["frame_count"]),
            "archive_query_count": int(stats["archive_query_count"]),
            "object_count": len(stats["objects"]),
            "objects": sorted(stats["objects"]),
            "participant_count": len(stats["participants"]),
            "participants": sorted(stats["participants"]),
            "query_manifest": str(query_path),
            "query_manifest_sha256": sha256_file(query_path),
            "sequence_manifest": str(sequence_path),
            "sequence_manifest_sha256": sha256_file(sequence_path),
        }

    observed_clip_counts = {
        partition: serialized_stats[partition]["sequence_count"]
        for partition in ("train", "test")
    }
    paper_counts = dict(protocol["paper_table_clip_counts"])
    summary = {
        "schema": "egotactile_official_split_summary_v1",
        "created_utc": utc_now(),
        "official_split_schema": OFFICIAL_SPLIT_SCHEMA,
        "official_source": "EgoTactile arXiv:2606.09243v1 Appendix A.1.5",
        "protocol": protocol_name,
        "source_split": source_split,
        "held_out_field": protocol["held_out_field"],
        "held_out_values": list(protocol["held_out_values"]),
        "source_query_manifest": str(source_query_path),
        "source_query_manifest_sha256": sha256_file(source_query_path),
        "source_sequence_manifest": str(source_sequence_path),
        "source_sequence_manifest_sha256": sha256_file(source_sequence_path),
        "paper_table_clip_counts": paper_counts,
        "observed_clip_counts": observed_clip_counts,
        "paper_table_clip_count_delta": {
            partition: observed_clip_counts[partition] - int(paper_counts[partition])
            for partition in ("train", "test")
        },
        "paper_definition_note": protocol.get("paper_definition_note", ""),
        "partitions": serialized_stats,
        "train_test_sequence_overlap_count": 0,
        "all_source_sequences_assigned": True,
        "all_source_queries_assigned": True,
    }
    summary_path = protocol_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    print(
        f"[official split {protocol_name}] sequences={observed_clip_counts}, "
        f"queries={{'train': {serialized_stats['train']['usable_query_count']}, "
        f"'test': {serialized_stats['test']['usable_query_count']}}}",
        flush=True,
    )
    return summary


def build_official_split_manifests(
    output_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    marker = output_root / ".sam3_bbox_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"Official manifests require a completed SAM3 HDF5 root: {marker}"
        )
    manifest_dir = output_root / "manifests"
    destination = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else manifest_dir / "official"
    )
    destination.mkdir(parents=True, exist_ok=True)
    summaries = {
        protocol_name: _build_one_official_protocol(
            manifest_dir=manifest_dir,
            output_dir=destination,
            protocol_name=protocol_name,
            protocol=protocol,
        )
        for protocol_name, protocol in OFFICIAL_PROTOCOLS.items()
    }
    index = {
        "schema": "egotactile_official_split_index_v1",
        "created_utc": utc_now(),
        "dataset_root": str(output_root),
        "completion_marker": str(marker),
        "completion_marker_sha256": sha256_file(marker),
        "official_split_schema": OFFICIAL_SPLIT_SCHEMA,
        "protocols": summaries,
        "validation_policy": (
            "The official benchmark defines train/test only. No validation "
            "manifest is synthesized and test must not be used for checkpoint selection."
        ),
    }
    atomic_write_json(destination / "index.json", index)
    return index


def command_official_splits(args: argparse.Namespace) -> None:
    result = build_official_split_manifests(args.output_root, args.output_dir)
    print(canonical_json(result), flush=True)


def command_materialize(args: argparse.Namespace) -> None:
    source_root = args.source_hdf5_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root == source_root:
        raise ValueError("SAM3 HDF5 output root must differ from the source root")
    audit_dir = args.audit_dir.expanduser().resolve()
    jobs, boxes_by_sequence = audit_tracking(
        args.manifests,
        args.tracking_roots,
        audit_dir,
        expected_clips=args.expected_clips,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {str(path.resolve()): sha256_file(path.resolve()) for path in args.manifests}
    tasks = []
    for row in jobs:
        identity = (row["split"], row["sequence_key"])
        job_dir = tracking_job_dir(args.tracking_roots, row)
        bbox_path = job_dir / "bboxes.jsonl"
        manifest_path = Path(row["_tracking_manifest"])
        source = source_root / row["source_h5_relpath"]
        target = output_root / row["source_h5_relpath"]
        tasks.append(
            {
                "source": str(source),
                "target": str(target),
                "boxes": boxes_by_sequence[identity],
                "target_hand": row["target_hand"],
                "trainable_frame_indices": row["trainable_frame_indices"],
                "prompt_preset": row["prompt_preset"],
                "tracking_manifest": str(manifest_path),
                "tracking_manifest_sha256": row["_tracking_manifest_sha256"],
                "bbox_sha256": sha256_file(bbox_path),
            }
        )
    counts = Counter()
    migration_totals = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for completed, result in enumerate(executor.map(_migrate_h5_task, tasks), start=1):
            counts[result["status"]] += 1
            for key in ("tracked_frames", "untracked_frames", "trainable_queries"):
                migration_totals[key] += int(result.get(key, 0))
            if completed % 20 == 0 or completed == len(tasks):
                print(
                    f"[HDF5 SAM3 migration] {completed}/{len(tasks)} {dict(counts)}",
                    flush=True,
                )
    manifest_summaries = rebuild_hdf5_manifests(output_root)
    total_sequences = sum(item["sequence_count"] for item in manifest_summaries.values())
    total_queries = sum(item["query_count"] for item in manifest_summaries.values())
    if total_sequences != args.expected_clips:
        raise RuntimeError(
            f"Materialized sequence count {total_sequences} != {args.expected_clips}"
        )
    completion = {
        "schema": MIGRATION_SCHEMA,
        "completed_utc": utc_now(),
        "source_hdf5_root": str(source_root),
        "output_root": str(output_root),
        "sequence_count": total_sequences,
        "query_count": total_queries,
        "migration_status_counts": dict(counts),
        "migration_totals": dict(migration_totals),
        "tracking_manifest_sha256": manifest_hashes,
        "manifest_summaries": manifest_summaries,
        "audit_summary": str(audit_dir / "summary.json"),
        "official_split_index": str(
            output_root / "manifests" / "official" / "index.json"
        ),
        "official_protocols": sorted(OFFICIAL_PROTOCOLS),
    }
    completion_path = output_root / ".sam3_bbox_complete.json"
    atomic_write_json(completion_path, completion)
    build_official_split_manifests(output_root)
    print(canonical_json(completion), flush=True)


def command_activate(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    marker = output_root / ".sam3_bbox_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(
            f"Refusing to activate an incomplete SAM3 dataset: {marker}"
        )
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if int(completion.get("sequence_count", -1)) != args.expected_clips:
        raise RuntimeError("SAM3 dataset completion marker has the wrong sequence count")
    active_link = args.active_link.expanduser().absolute()
    active_link.parent.mkdir(parents=True, exist_ok=True)
    if active_link.exists() and not active_link.is_symlink():
        raise RuntimeError(f"Active dataset path exists and is not a symlink: {active_link}")
    temporary = active_link.with_name(f".{active_link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    os.symlink(output_root, temporary, target_is_directory=True)
    os.replace(temporary, active_link)
    activation = {
        "schema": "egotactile_active_dataset_v1",
        "activated_utc": utc_now(),
        "active_link": str(active_link),
        "target": str(output_root),
        "completion_sha256": sha256_file(marker),
    }
    atomic_write_json(output_root / "activation.json", activation)
    print(canonical_json(activation), flush=True)


def add_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--tracking-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-clips", type=int, default=767)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifests")
    build.add_argument("--raw-root", type=Path, required=True)
    build.add_argument("--hdf5-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--expected-clips", type=int, default=767)
    build.set_defaults(func=command_build)

    audit = subparsers.add_parser("audit")
    add_manifest_arguments(audit)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.set_defaults(func=command_audit)

    materialize = subparsers.add_parser("materialize")
    add_manifest_arguments(materialize)
    materialize.add_argument("--source-hdf5-root", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--audit-dir", type=Path, required=True)
    materialize.add_argument("--workers", type=int, default=8)
    materialize.set_defaults(func=command_materialize)

    official_splits = subparsers.add_parser("official-splits")
    official_splits.add_argument("--output-root", type=Path, required=True)
    official_splits.add_argument("--output-dir", type=Path)
    official_splits.set_defaults(func=command_official_splits)

    activate = subparsers.add_parser("activate")
    activate.add_argument("--output-root", type=Path, required=True)
    activate.add_argument("--active-link", type=Path, required=True)
    activate.add_argument("--expected-clips", type=int, default=767)
    activate.set_defaults(func=command_activate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
