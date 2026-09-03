#!/usr/bin/env python3
"""Build and materialize HumanTouch head-camera SAM3 hand boxes.

Only ``observation.images.cam_head`` videos are admitted. Anonymous SAM tracks
are assigned to left/right from the first reliable screen-order pair. Missing
or ambiguous frames remain null so a later HDF5 builder can exclude them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from preprocess.acedata.rebuild_sam3_bboxes import associate_tracking_job


SCHEMA = "humantouch_sam3_head_gloved_screen_order_v1"
HEAD_VIDEO_KEY = "observation.images.cam_head"


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
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


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
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sequence_key).strip("_")
    digest = hashlib.sha256(sequence_key.encode()).hexdigest()[:12]
    return f"ht__{readable[:150]}__{digest}"


def episode_index(path: Path) -> int:
    match = re.fullmatch(r"episode_(\d+)\.mp4", path.name)
    if match is None:
        raise ValueError(f"Unexpected HumanTouch video name: {path}")
    return int(match.group(1))


def load_subject_metadata(subject_root: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    info_path = subject_root / "meta" / "info.json"
    episodes_path = subject_root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata below {subject_root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = {}
    for row in read_jsonl(episodes_path):
        index = int(row["episode_index"])
        if index in episodes:
            raise ValueError(f"{episodes_path}: duplicate episode_index={index}")
        episodes[index] = row
    return info, episodes


def find_pressure_path(processed_root: Path, subject: str, video: Path) -> Path:
    relative_chunk = video.parents[1].name
    return (
        processed_root
        / subject
        / "data"
        / relative_chunk
        / video.with_suffix(".npz").name
    )


def command_build(args: argparse.Namespace) -> None:
    raw_root = args.raw_root.expanduser().resolve()
    processed_root = args.processed_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    subject_roots = sorted(
        path for path in raw_root.glob("X*") if path.is_dir() and (path / "meta").is_dir()
    )
    if not subject_roots:
        raise FileNotFoundError(f"No HumanTouch X*/meta roots found below {raw_root}")

    records: list[dict[str, Any]] = []
    subject_summaries = []
    pressure_available = 0
    total_frames = 0
    for subject_root in subject_roots:
        subject = subject_root.name
        info, episodes = load_subject_metadata(subject_root)
        feature = info.get("features", {}).get(HEAD_VIDEO_KEY)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"{subject_root}: missing video feature {HEAD_VIDEO_KEY}")
        video_info = feature.get("info", {})
        shape = feature.get("shape", ())
        if len(shape) != 3:
            raise ValueError(f"{subject_root}: invalid head-video shape {shape!r}")
        videos = sorted(
            subject_root.glob(
                f"videos/chunk-*/{HEAD_VIDEO_KEY}/episode_*.mp4"
            )
        )
        expected = int(info.get("total_episodes", -1))
        if len(videos) != expected or len(episodes) != expected:
            raise RuntimeError(
                f"{subject}: head videos={len(videos)}, episode metadata={len(episodes)}, "
                f"info.total_episodes={expected}"
            )
        observed_indices = {episode_index(video) for video in videos}
        if observed_indices != set(episodes):
            missing = sorted(set(episodes) - observed_indices)
            extra = sorted(observed_indices - set(episodes))
            raise RuntimeError(
                f"{subject}: head video/metadata indices differ; "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        subject_frames = 0
        subject_pressure = 0
        for video in videos:
            index = episode_index(video)
            episode = episodes[index]
            frame_count = int(episode["length"])
            if frame_count <= 0:
                raise ValueError(f"{video}: non-positive episode length {frame_count}")
            pressure_path = find_pressure_path(processed_root, subject, video)
            has_pressure = pressure_path.is_file()
            pressure_available += int(has_pressure)
            subject_pressure += int(has_pressure)
            total_frames += frame_count
            subject_frames += frame_count
            episode_name = video.stem
            sequence_key = f"{subject}/{episode_name}"
            records.append(
                {
                    "schema": SCHEMA,
                    "job_id": safe_job_id(sequence_key),
                    "dataset": "humantouch",
                    "tracker_dataset": "generic",
                    "split": "all",
                    "sequence_key": sequence_key,
                    "subject": subject,
                    "episode_index": index,
                    "episode_key": str(episode.get("episode_key", f"{index:06d}")),
                    "tasks": episode.get("tasks", []),
                    "resource_path": str(video.resolve()),
                    "resource_type": "video",
                    "source_path": str(video.resolve()),
                    "video_key": HEAD_VIDEO_KEY,
                    "pressure_path": str(pressure_path),
                    "pressure_available_at_manifest_build": has_pressure,
                    "frame_count": frame_count,
                    "frame_width": int(shape[1]),
                    "frame_height": int(shape[0]),
                    "fps": float(video_info.get("video.fps", info.get("fps", 60))),
                    "expected_gloved_hands": 2,
                    "prompt_preset": "gloved",
                    "prompt": "gloved hand",
                    "association_policy": "initial_screen_order",
                }
            )
        subject_summaries.append(
            {
                "subject": subject,
                "episode_count": len(videos),
                "frame_count": subject_frames,
                "processed_pressure_count": subject_pressure,
            }
        )
        print(
            f"[HumanTouch SAM3 build] {subject}: {len(videos)} head videos, "
            f"pressure={subject_pressure}",
            flush=True,
        )

    records.sort(key=lambda row: (row["subject"], int(row["episode_index"])))
    write_jsonl(output, records)
    summary = {
        "schema": f"{SCHEMA}_manifest",
        "created_utc": utc_now(),
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "head_video_key": HEAD_VIDEO_KEY,
        "wrist_videos_included": False,
        "subject_count": len(subject_roots),
        "episode_count": len(records),
        "frame_count": total_frames,
        "processed_pressure_count": pressure_available,
        "manifest": str(output),
        "manifest_sha256": sha256_file(output),
        "prompt_preset": "gloved",
        "expected_gloved_hands": 2,
        "association_policy": "initial_screen_order",
        "subjects": subject_summaries,
    }
    atomic_write_json(output.with_suffix(".summary.json"), summary)
    print(canonical_json(summary), flush=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty HumanTouch SAM3 manifest: {path}")
    for line_number, row in enumerate(rows, start=1):
        if row.get("dataset") != "humantouch" or row.get("video_key") != HEAD_VIDEO_KEY:
            raise ValueError(f"{path}:{line_number}: manifest is not head-only HumanTouch")
        resource = Path(str(row["resource_path"]))
        if resource.parent.name != HEAD_VIDEO_KEY:
            raise ValueError(f"{path}:{line_number}: wrist/non-head resource rejected: {resource}")
    return rows


def tracking_job_dir(tracking_root: Path, row: dict[str, Any]) -> Path:
    return (
        tracking_root.expanduser().resolve()
        / "results"
        / str(row["dataset"])
        / str(row["split"])
        / str(row["job_id"])
    )


def write_audit(
    output_dir: Path,
    manifest: Path,
    tracking_root: Path,
    rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = metrics + failures
    fields = sorted(
        {key for item in csv_rows for key in item if key not in {"assignments", "collisions"}}
    )
    csv_path = output_dir / "sequence_metrics.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    os.replace(temporary, csv_path)
    total_frames = sum(int(row["frame_count"]) for row in rows)
    totals = {
        key: sum(int(item.get(key, 0)) for item in metrics)
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
        "head_video_key": HEAD_VIDEO_KEY,
        "wrist_videos_included": False,
        "episode_count": len(rows),
        "successful_episode_count": len(metrics),
        "failed_episode_count": len(failures),
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
    return summary


def process_tracking_rows(
    rows: list[dict[str, Any]], tracking_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = []
    failures = []
    for completed, row in enumerate(rows, start=1):
        try:
            _, details = associate_tracking_job(row, tracking_root)
            metrics.append(details)
        except Exception as exc:
            failures.append(
                {
                    "status": "failed",
                    "sequence_key": row.get("sequence_key"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if completed % 100 == 0 or completed == len(rows):
            print(
                f"[HumanTouch SAM3 audit] {completed}/{len(rows)}, failures={len(failures)}",
                flush=True,
            )
    return metrics, failures


def command_audit(args: argparse.Namespace) -> None:
    rows = load_manifest(args.manifest)
    metrics, failures = process_tracking_rows(rows, args.tracking_root)
    summary = write_audit(
        args.output_dir.expanduser().resolve(),
        args.manifest,
        args.tracking_root,
        rows,
        metrics,
        failures,
    )
    print(canonical_json(summary), flush=True)
    if args.require_all and failures:
        raise RuntimeError(f"HumanTouch SAM3 audit has {len(failures)} failed episodes")


def command_materialize(args: argparse.Namespace) -> None:
    rows = load_manifest(args.manifest)
    output_root = args.output_bbox_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    provenance_root = output_root / ".provenance"
    metrics = []
    failures = []
    counts = Counter()
    for completed, row in enumerate(rows, start=1):
        subject = str(row["subject"])
        episode_name = f"episode_{int(row['episode_index']):06d}"
        target = output_root / subject / f"{episode_name}.json"
        provenance = provenance_root / subject / f"{episode_name}.json"
        bbox_source = tracking_job_dir(args.tracking_root, row) / "bboxes.jsonl"
        try:
            current_hash = sha256_file(bbox_source)
            if target.is_file() and provenance.is_file():
                prior = json.loads(provenance.read_text(encoding="utf-8"))
                if (
                    prior.get("tracking_bbox_jsonl_sha256") == current_hash
                    and int(prior.get("frame_count", -1)) == int(row["frame_count"])
                ):
                    metrics.append(prior["metrics"])
                    counts["resumed"] += 1
                    continue
            boxes, details = associate_tracking_job(row, args.tracking_root)
            atomic_write_json(target, boxes)
            atomic_write_json(
                provenance,
                {
                    "schema": SCHEMA,
                    "created_utc": utc_now(),
                    "sequence_key": row["sequence_key"],
                    "video_key": HEAD_VIDEO_KEY,
                    "source_path": row["source_path"],
                    "pressure_path": row.get("pressure_path"),
                    "frame_count": int(row["frame_count"]),
                    "prompt_preset": "gloved",
                    "expected_gloved_hands": 2,
                    "association_policy": "initial_screen_order",
                    "missing_frames_are_excluded": True,
                    "tracking_bbox_jsonl_sha256": current_hash,
                    "metrics": details,
                },
            )
            metrics.append(details)
            counts["written"] += 1
        except Exception as exc:
            failures.append(
                {
                    "status": "failed",
                    "sequence_key": row.get("sequence_key"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            counts["failed"] += 1
        if completed % 100 == 0 or completed == len(rows):
            print(
                f"[HumanTouch SAM3 materialize] {completed}/{len(rows)} {dict(counts)}",
                flush=True,
            )

    summary = write_audit(
        args.audit_dir.expanduser().resolve(),
        args.manifest,
        args.tracking_root,
        rows,
        metrics,
        failures,
    )
    completion = {
        "schema": SCHEMA,
        "completed_utc": utc_now(),
        "status": "complete" if not failures else "partial",
        "output_bbox_root": str(output_root),
        "head_video_key": HEAD_VIDEO_KEY,
        "wrist_videos_included": False,
        "episode_count": len(rows),
        "materialized_episode_count": len(metrics),
        "failed_episode_count": len(failures),
        "status_counts": dict(counts),
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": sha256_file(args.manifest.expanduser().resolve()),
        "audit_summary": str(args.audit_dir.expanduser().resolve() / "summary.json"),
        "association_policy": "initial_screen_order",
        "missing_frames_are_excluded": True,
    }
    atomic_write_json(output_root / ".sam3_bbox_status.json", completion)
    print(canonical_json(completion), flush=True)
    if args.require_all and failures:
        raise RuntimeError(f"HumanTouch bbox materialization has {len(failures)} failures")


def add_tracking_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    build.add_argument("--raw-root", type=Path, required=True)
    build.add_argument("--processed-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(func=command_build)

    audit = subparsers.add_parser("audit")
    add_tracking_input(audit)
    audit.add_argument("--output-dir", type=Path, required=True)
    audit.set_defaults(func=command_audit)

    materialize = subparsers.add_parser("materialize")
    add_tracking_input(materialize)
    materialize.add_argument("--output-bbox-root", type=Path, required=True)
    materialize.add_argument("--audit-dir", type=Path, required=True)
    materialize.set_defaults(func=command_materialize)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
