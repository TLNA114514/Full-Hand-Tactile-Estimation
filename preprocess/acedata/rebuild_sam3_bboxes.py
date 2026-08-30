#!/usr/bin/env python3
"""Build, audit, materialize, and activate SAM3 boxes for AceData.

AceData videos contain two gloved hands. Hand identity is assigned without
using the old detector: at the first reliable two-track frame, the screen-left
track becomes ``left`` and the screen-right track becomes ``right``. Missing or
ambiguous frames remain untracked and must be excluded downstream.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from sam3_bbox_reconstruction.associate_tracks import (
    assign_touchanything_screen_order,
    observations_from_frames,
    reconnect_tracklets,
    split_tracklets,
)


SCHEMA = "acedata_sam3_gloved_screen_order_v1"
EXPECTED_CLIPS = 494


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
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")
    atomic_write(path, payload)


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    payload = bytearray()
    count = 0
    for row in rows:
        payload.extend(canonical_json(row).encode("utf-8"))
        payload.extend(b"\n")
        count += 1
    atomic_write(path, bytes(payload))
    return count


def safe_job_id(sequence_key: str) -> str:
    readable = "__".join(Path(sequence_key).parts)
    readable = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in readable
    )
    digest = hashlib.sha256(sequence_key.encode("utf-8")).hexdigest()[:12]
    return f"ace__{readable[:150]}__{digest}"


def video_metadata(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frame_count <= 0 or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(
            f"Invalid video metadata for {path}: frames={frame_count}, "
            f"size={width}x{height}, fps={fps}"
        )
    return {"frame_count": frame_count, "width": width, "height": height, "fps": fps}


def pressure_frame_count(path: Path) -> int:
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if "frame_count" in archive:
            return int(np.asarray(archive["frame_count"]).reshape(-1)[0])
        for key in (
            "left_pressure_continuous_subdiv",
            "right_pressure_continuous_subdiv",
            "left_force_norm_5N",
            "right_force_norm_5N",
        ):
            if key in archive:
                return int(archive[key].shape[0])
    raise KeyError(f"{path}: no pressure array from which to infer frame count")


def discover_clips(raw_root: Path) -> list[tuple[str, str, Path]]:
    clips = []
    for video in raw_root.glob("*/CLIP*/EgoCap/stereo1.mp4"):
        clip_dir = video.parents[1]
        clips.append((clip_dir.parent.name, clip_dir.name, video.resolve()))
    return sorted(clips)


def command_build(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    records = []
    total_frames = 0
    for completed, (subject, clip, video) in enumerate(discover_clips(raw_root), start=1):
        sequence_key = f"{subject}/{clip}"
        metadata = video_metadata(video)
        pressure_candidates = (
            processed_root / "clips" / subject / clip / "pressure_mean.npz",
            processed_root / "pressure_mean" / subject / f"{clip}.npz",
        )
        pressure_path = next(
            (candidate for candidate in pressure_candidates if candidate.is_file()),
            pressure_candidates[0],
        )
        pressure_frames = pressure_frame_count(pressure_path)
        old_bbox_path = processed_root / "bboxes" / subject / f"{clip}.json"
        if not old_bbox_path.is_file():
            raise FileNotFoundError(old_bbox_path)
        old_boxes = json.loads(old_bbox_path.read_text(encoding="utf-8"))
        if metadata["frame_count"] != pressure_frames or len(old_boxes) != pressure_frames:
            raise RuntimeError(
                f"{sequence_key}: frame-count mismatch: video={metadata['frame_count']}, "
                f"pressure={pressure_frames}, old_bbox={len(old_boxes)}"
            )
        records.append(
            {
                "schema": SCHEMA,
                "job_id": safe_job_id(sequence_key),
                "dataset": "acedata",
                "tracker_dataset": "generic",
                "split": "all",
                "sequence_key": sequence_key,
                "subject": subject,
                "clip": clip,
                "resource_path": str(video),
                "resource_type": "video",
                "source_path": str(video),
                "pressure_path": str(pressure_path),
                "old_bbox_relpath": f"{subject}/{clip}.json",
                "frame_count": metadata["frame_count"],
                "frame_width": metadata["width"],
                "frame_height": metadata["height"],
                "fps": metadata["fps"],
                "expected_gloved_hands": 2,
                "prompt_preset": "gloved",
                "prompt": "gloved hand",
                "association_policy": "initial_screen_order",
            }
        )
        total_frames += metadata["frame_count"]
        if completed % 25 == 0:
            print(f"[AceData SAM3 build] {completed} clips", flush=True)
    if len(records) != args.expected_clips:
        raise RuntimeError(
            f"Expected {args.expected_clips} AceData clips, discovered {len(records)}"
        )
    write_jsonl(output, records)
    summary = {
        "schema": f"{SCHEMA}_manifest",
        "created_utc": utc_now(),
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "clip_count": len(records),
        "frame_count": total_frames,
        "manifest": str(output),
        "manifest_sha256": sha256_file(output),
        "prompt_preset": "gloved",
        "expected_gloved_hands": 2,
        "association_policy": "initial_screen_order",
    }
    atomic_write_json(output.with_suffix(".summary.json"), summary)
    print(canonical_json(summary), flush=True)


def tracking_job_dir(tracking_root: Path, row: dict[str, Any]) -> Path:
    return (
        tracking_root.expanduser().resolve()
        / "results"
        / row["dataset"]
        / row["split"]
        / row["job_id"]
    )


def _center_jump(left: list[float], right: list[float]) -> float:
    left_width = max(left[2] - left[0], 1.0)
    left_height = max(left[3] - left[1], 1.0)
    right_width = max(right[2] - right[0], 1.0)
    right_height = max(right[3] - right[1], 1.0)
    scale = max(math.hypot(left_width, left_height), math.hypot(right_width, right_height), 1.0)
    left_center = (0.5 * (left[0] + left[2]), 0.5 * (left[1] + left[3]))
    right_center = (0.5 * (right[0] + right[2]), 0.5 * (right[1] + right[3]))
    return math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]) / scale


def associate_tracking_job(
    row: dict[str, Any], tracking_root: Path
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    job_dir = tracking_job_dir(tracking_root, row)
    summary_path = job_dir / "summary.json"
    bbox_path = job_dir / "bboxes.jsonl"
    if not summary_path.is_file() or not bbox_path.is_file():
        raise FileNotFoundError(f"Incomplete SAM3 output under {job_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise RuntimeError(f"{job_dir}: tracker status={summary.get('status')!r}")
    frame_rows = read_jsonl(bbox_path)
    expected_frames = int(row["frame_count"])
    observed_indices = {int(item["frame_index"]) for item in frame_rows}
    extras = sorted(index for index in observed_indices if not 0 <= index < expected_frames)
    if extras:
        raise RuntimeError(f"{bbox_path}: out-of-range frame indices {extras[:10]}")

    observations = observations_from_frames(frame_rows)
    associations = reconnect_tracklets(split_tracklets(observations))
    assign_touchanything_screen_order(associations)
    candidates: dict[tuple[int, str], list[tuple[Any, Any]]] = defaultdict(list)
    assignment_rows = []
    for association in associations:
        first = min(association.observations, key=lambda item: item.frame_index)
        assignment_rows.append(
            {
                "association_id": association.association_id,
                "target_hand": association.target_hand,
                "confidence": association.confidence,
                "observation_count": len(association.observations),
                "first_frame": first.frame_index,
                "first_center_x": 0.5 * (first.bbox[0] + first.bbox[2]),
                "evidence": association.evidence,
            }
        )
        if association.target_hand not in {"left", "right"}:
            continue
        for observation in association.observations:
            candidates[(observation.frame_index, association.target_hand)].append(
                (association, observation)
            )

    empty = {"bbox": None, "score": 0.0}
    output = {
        str(frame_index): {"left": dict(empty), "right": dict(empty)}
        for frame_index in range(expected_frames)
    }
    collisions = []
    source_counts = Counter()
    track_ids = Counter()
    hand_indices: dict[str, list[int]] = {"left": [], "right": []}
    scores: dict[str, list[float]] = {"left": [], "right": []}
    for (frame_index, hand), rows in sorted(candidates.items()):
        if len(rows) != 1:
            collisions.append(
                {
                    "frame_index": frame_index,
                    "hand": hand,
                    "association_ids": sorted(item.association_id for item, _ in rows),
                }
            )
            continue
        _, observation = rows[0]
        score = 0.0 if observation.prompt_score is None else float(observation.prompt_score)
        output[str(frame_index)][hand] = {
            "bbox": [float(value) for value in observation.bbox],
            "score": score,
        }
        hand_indices[hand].append(frame_index)
        scores[hand].append(score)
        source_counts[observation.bbox_source] += 1
        track_ids[observation.raw_track_id] += 1

    if not hand_indices["left"] or not hand_indices["right"]:
        raise RuntimeError(
            f"{row['sequence_key']}: initial screen-order association did not recover "
            f"both hands (left={len(hand_indices['left'])}, right={len(hand_indices['right'])})"
        )
    jumps = {}
    for hand in ("left", "right"):
        indices = hand_indices[hand]
        values = []
        for previous, current in zip(indices, indices[1:]):
            if current == previous + 1:
                values.append(
                    _center_jump(
                        output[str(previous)][hand]["bbox"],
                        output[str(current)][hand]["bbox"],
                    )
                )
        jumps[hand] = values
    both = sum(
        output[str(index)]["left"]["bbox"] is not None
        and output[str(index)]["right"]["bbox"] is not None
        for index in range(expected_frames)
    )
    any_hand = sum(
        output[str(index)]["left"]["bbox"] is not None
        or output[str(index)]["right"]["bbox"] is not None
        for index in range(expected_frames)
    )
    metrics = {
        "status": "complete",
        "sequence_key": row["sequence_key"],
        "frame_count": expected_frames,
        "raw_observation_count": len(observations),
        "raw_track_count": len(track_ids),
        "association_count": len(associations),
        "left_tracked_frames": len(hand_indices["left"]),
        "right_tracked_frames": len(hand_indices["right"]),
        "both_hands_tracked_frames": both,
        "any_hand_tracked_frames": any_hand,
        "left_coverage": len(hand_indices["left"]) / expected_frames,
        "right_coverage": len(hand_indices["right"]) / expected_frames,
        "both_hands_coverage": both / expected_frames,
        "any_hand_coverage": any_hand / expected_frames,
        "collision_count": len(collisions),
        "left_mean_score": mean(scores["left"]) if scores["left"] else 0.0,
        "right_mean_score": mean(scores["right"]) if scores["right"] else 0.0,
        "left_mean_center_jump": mean(jumps["left"]) if jumps["left"] else 0.0,
        "right_mean_center_jump": mean(jumps["right"]) if jumps["right"] else 0.0,
        "left_max_center_jump": max(jumps["left"], default=0.0),
        "right_max_center_jump": max(jumps["right"], default=0.0),
        "bbox_source_counts": dict(source_counts),
        "tracking_bbox_jsonl": str(bbox_path),
        "tracking_bbox_jsonl_sha256": sha256_file(bbox_path),
        "assignment_policy": "initial_screen_order",
        "assignments": assignment_rows,
        "collisions": collisions,
    }
    return output, metrics


def load_manifest(path: Path, expected_clips: int) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    rows = read_jsonl(path)
    if len(rows) != expected_clips:
        raise RuntimeError(f"Expected {expected_clips} jobs in {path}, found {len(rows)}")
    return rows


def audit_all(
    manifest: Path,
    tracking_root: Path,
    output_dir: Path,
    expected_clips: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, dict[str, Any]]]]]:
    rows = load_manifest(manifest, expected_clips)
    metrics = []
    boxes = {}
    failures = []
    for completed, row in enumerate(rows, start=1):
        try:
            sequence_boxes, sequence_metrics = associate_tracking_job(row, tracking_root)
            boxes[row["sequence_key"]] = sequence_boxes
            metrics.append(sequence_metrics)
        except Exception as exc:
            failures.append(
                {
                    "status": "failed",
                    "sequence_key": row.get("sequence_key"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if completed % 25 == 0 or completed == len(rows):
            print(
                f"[AceData SAM3 audit] {completed}/{len(rows)}, failures={len(failures)}",
                flush=True,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = metrics + failures
    fields = sorted({key for item in csv_rows for key in item if key not in {"assignments", "collisions"}})
    csv_path = output_dir / "sequence_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    os.replace(temporary, csv_path)
    total_frames = sum(int(item["frame_count"]) for item in metrics)
    totals = {
        key: sum(int(item[key]) for item in metrics)
        for key in (
            "left_tracked_frames",
            "right_tracked_frames",
            "both_hands_tracked_frames",
            "any_hand_tracked_frames",
            "collision_count",
        )
    }
    summary = {
        "schema": f"{SCHEMA}_audit",
        "created_utc": utc_now(),
        "manifest": str(manifest.expanduser().resolve()),
        "manifest_sha256": sha256_file(manifest.expanduser().resolve()),
        "tracking_root": str(tracking_root.expanduser().resolve()),
        "clip_count": len(rows),
        "successful_clip_count": len(metrics),
        "failed_clip_count": len(failures),
        "frame_count": total_frames,
        **totals,
        "left_coverage": totals["left_tracked_frames"] / max(total_frames, 1),
        "right_coverage": totals["right_tracked_frames"] / max(total_frames, 1),
        "both_hands_coverage": totals["both_hands_tracked_frames"] / max(total_frames, 1),
        "any_hand_coverage": totals["any_hand_tracked_frames"] / max(total_frames, 1),
        "missing_frames_are_excluded": True,
        "failures": failures,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "sequence_details.jsonl", metrics)
    if failures:
        raise RuntimeError(
            f"AceData SAM3 audit failed for {len(failures)}/{len(rows)} clips; "
            f"see {output_dir / 'summary.json'}"
        )
    return rows, boxes


def command_audit(args: argparse.Namespace) -> None:
    audit_all(
        args.manifest,
        args.tracking_root,
        args.output_dir.expanduser().resolve(),
        args.expected_clips,
    )
    print(f"AceData SAM3 audit passed: {args.output_dir}", flush=True)


def command_materialize(args: argparse.Namespace) -> None:
    output_root = args.output_bbox_root.expanduser().resolve()
    rows, boxes = audit_all(
        args.manifest,
        args.tracking_root,
        args.audit_dir.expanduser().resolve(),
        args.expected_clips,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    provenance_root = output_root / ".provenance"
    counts = Counter()
    for completed, row in enumerate(rows, start=1):
        sequence_key = row["sequence_key"]
        subject, clip = Path(sequence_key).parts
        target = output_root / subject / f"{clip}.json"
        provenance = provenance_root / subject / f"{clip}.json"
        sequence_boxes = boxes[sequence_key]
        if target.is_file() and provenance.is_file():
            prior = json.loads(provenance.read_text(encoding="utf-8"))
            expected_hash = prior.get("tracking_bbox_jsonl_sha256")
            current_hash = sha256_file(
                tracking_job_dir(args.tracking_root, row) / "bboxes.jsonl"
            )
            if expected_hash == current_hash and len(json.loads(target.read_text())) == int(row["frame_count"]):
                counts["resumed"] += 1
                continue
        details = associate_tracking_job(row, args.tracking_root)[1]
        atomic_write_json(target, sequence_boxes)
        atomic_write_json(
            provenance,
            {
                "schema": SCHEMA,
                "created_utc": utc_now(),
                "sequence_key": sequence_key,
                "prompt_preset": "gloved",
                "expected_gloved_hands": 2,
                "association_policy": "initial_screen_order",
                "missing_frames_are_excluded": True,
                **details,
            },
        )
        counts["written"] += 1
        if completed % 25 == 0 or completed == len(rows):
            print(
                f"[AceData SAM3 materialize] {completed}/{len(rows)} {dict(counts)}",
                flush=True,
            )
    bbox_paths = sorted(
        path for path in output_root.glob("*/*.json") if ".provenance" not in path.parts
    )
    if len(bbox_paths) != args.expected_clips:
        raise RuntimeError(
            f"Materialized bbox count {len(bbox_paths)} != {args.expected_clips}"
        )
    completion = {
        "schema": SCHEMA,
        "completed_utc": utc_now(),
        "output_bbox_root": str(output_root),
        "clip_count": len(bbox_paths),
        "status_counts": dict(counts),
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": sha256_file(args.manifest.expanduser().resolve()),
        "audit_summary": str(args.audit_dir.expanduser().resolve() / "summary.json"),
        "association_policy": "initial_screen_order",
        "missing_frames_are_excluded": True,
    }
    atomic_write_json(output_root / ".sam3_bbox_complete.json", completion)
    print(canonical_json(completion), flush=True)


def command_activate(args: argparse.Namespace) -> None:
    processed_root = args.processed_root.expanduser().resolve()
    output_root = args.output_bbox_root.expanduser().resolve()
    marker = output_root / ".sam3_bbox_complete.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Refusing to activate incomplete boxes: {marker}")
    completion = json.loads(marker.read_text(encoding="utf-8"))
    if int(completion.get("clip_count", -1)) != args.expected_clips:
        raise RuntimeError("SAM3 completion marker has the wrong clip count")
    active_path = processed_root / "bboxes"
    backup_path = args.backup_bbox_root.expanduser().resolve()
    if active_path.is_symlink() and active_path.resolve() == output_root:
        print(f"AceData SAM3 boxes already active: {active_path} -> {output_root}")
        return
    if active_path.exists() and not active_path.is_symlink():
        if backup_path.exists():
            raise RuntimeError(
                f"Cannot preserve current bbox directory because backup exists: {backup_path}"
            )
        os.replace(active_path, backup_path)
    elif active_path.is_symlink():
        active_path.unlink()
    temporary = active_path.with_name(f".{active_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    os.symlink(output_root, temporary, target_is_directory=True)
    os.replace(temporary, active_path)

    manifest_path = processed_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_backup = processed_root / "manifest.pre_sam3_bbox.json"
    if not manifest_backup.exists():
        atomic_write_json(manifest_backup, manifest)
    artifacts = manifest.setdefault("artifacts", {})
    artifacts["bbox_dir"] = str(active_path)
    artifacts["bbox_json_count"] = args.expected_clips
    artifacts["bbox_source"] = SCHEMA
    artifacts["legacy_sample_meta_bbox_source"] = "vitdet_vitpose_wholebody_v1"
    artifacts["legacy_sample_meta_requires_rebuild"] = True
    manifest["sam3_bbox"] = {
        "schema": SCHEMA,
        "activated_utc": utc_now(),
        "active_bbox_root": str(active_path),
        "target_bbox_root": str(output_root),
        "backup_bbox_root": str(backup_path),
        "prompt_preset": "gloved",
        "expected_gloved_hands": 2,
        "association_policy": "initial_screen_order",
        "missing_frames_are_excluded": True,
        "completion_sha256": sha256_file(marker),
    }
    atomic_write_json(manifest_path, manifest)
    activation = {
        "schema": f"{SCHEMA}_activation",
        "activated_utc": utc_now(),
        "active_bbox_root": str(active_path),
        "target_bbox_root": str(output_root),
        "backup_bbox_root": str(backup_path),
        "warning": (
            "Existing samples/all/*/meta.json files still embed legacy boxes and must "
            "not be used to build the next training manifest. Rebuild HDF5/sample "
            "records from the active bbox root; missing SAM3 rows are non-trainable."
        ),
    }
    atomic_write_json(output_root / "activation.json", activation)
    print(canonical_json(activation), flush=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--expected-clips", type=int, default=EXPECTED_CLIPS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    build.add_argument("--raw-root", type=Path, required=True)
    build.add_argument("--processed-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--expected-clips", type=int, default=EXPECTED_CLIPS)
    build.set_defaults(func=command_build)

    audit = subparsers.add_parser("audit")
    add_common(audit)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.set_defaults(func=command_audit)

    materialize = subparsers.add_parser("materialize")
    add_common(materialize)
    materialize.add_argument("--output-bbox-root", type=Path, required=True)
    materialize.add_argument("--audit-dir", type=Path, required=True)
    materialize.set_defaults(func=command_materialize)

    activate = subparsers.add_parser("activate")
    activate.add_argument("--processed-root", type=Path, required=True)
    activate.add_argument("--output-bbox-root", type=Path, required=True)
    activate.add_argument("--backup-bbox-root", type=Path, required=True)
    activate.add_argument("--expected-clips", type=int, default=EXPECTED_CLIPS)
    activate.set_defaults(func=command_activate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
