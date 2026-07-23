#!/usr/bin/env python3
"""Summarize anonymous SAM hand tracking separately for OpenTouch and TA."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

try:
    from .pilot_manifest import parse_dataset_selection
except ImportError:
    from pilot_manifest import parse_dataset_selection


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bbox_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def normalized_center_jump(previous: Iterable[float], current: Iterable[float]) -> float:
    p = tuple(float(value) for value in previous)
    c = tuple(float(value) for value in current)
    p_center = ((p[0] + p[2]) * 0.5, (p[1] + p[3]) * 0.5)
    c_center = ((c[0] + c[2]) * 0.5, (c[1] + c[3]) * 0.5)
    distance = math.hypot(p_center[0] - c_center[0], p_center[1] - c_center[1])
    reference = max(1.0, math.sqrt(max(bbox_area(p), bbox_area(c))))
    return distance / reference


def sequence_metrics(job_dir: Path, manifest_row: dict) -> dict:
    bbox_path = job_dir / "bboxes.jsonl"
    summary_path = job_dir / "summary.json"
    if not bbox_path.is_file() or not summary_path.is_file():
        return {
            "dataset": manifest_row["dataset"],
            "split": manifest_row["split"],
            "sequence_key": manifest_row["sequence_key"],
            "status": "missing",
        }
    frames = read_jsonl(bbox_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit_path = job_dir / "track_audit.json"
    track_audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else {}
    )
    expected = int(manifest_row["expected_gloved_hands"])
    observations_by_id: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    cardinalities = []
    scores = []
    rejected = 0
    bbox_source_counts: Counter[str] = Counter()
    for frame in frames:
        tracks = frame.get("tracks", [])
        cardinalities.append(len(tracks))
        rejected += len(frame.get("rejected_tracks", []))
        for track in tracks:
            bbox_source_counts[str(track.get("bbox_source", "sam3_native"))] += 1
            observations_by_id[int(track["track_id"])].append(
                (int(frame["frame_index"]), track)
            )
            score = track.get("prompt_score")
            if score is not None and math.isfinite(float(score)):
                scores.append(float(score))
    jumps = []
    area_log_changes = []
    fragment_count = 0
    for observations in observations_by_id.values():
        observations.sort(key=lambda item: item[0])
        fragment_count += 1 if observations else 0
        for (prev_frame, previous), (frame_index, current) in zip(
            observations, observations[1:]
        ):
            if frame_index != prev_frame + 1:
                fragment_count += 1
                continue
            jumps.append(normalized_center_jump(previous["bbox"], current["bbox"]))
            previous_area = max(1.0, bbox_area(previous["bbox"]))
            current_area = max(1.0, bbox_area(current["bbox"]))
            area_log_changes.append(abs(math.log(current_area / previous_area)))
    frame_count = len(frames)
    nonempty = sum(value > 0 for value in cardinalities)
    exact = sum(value == expected for value in cardinalities)
    chunks = summary.get("resolved_session_policy", {}).get("chunks", [])
    boundary_indices: set[int] = set()
    for previous, current in zip(chunks, chunks[1:]):
        boundary = int(current["start"])
        overlap = max(0, int(previous["end"]) - boundary)
        radius = max(1, min(16, overlap // 2 if overlap else 4))
        boundary_indices.update(range(max(0, boundary - radius), boundary + radius))
    boundary_values = [
        len(frame.get("tracks", []))
        for frame in frames
        if int(frame.get("frame_index", -1)) in boundary_indices
    ]
    interior_values = [
        len(frame.get("tracks", []))
        for frame in frames
        if int(frame.get("frame_index", -1)) not in boundary_indices
    ]
    reentry_ids = track_audit.get("selection", {}).get(
        "reentry_selected_track_ids", []
    )

    def coverage(values: list[int], predicate) -> float | None:
        return sum(predicate(value) for value in values) / len(values) if values else None

    return {
        "dataset": manifest_row["dataset"],
        "split": manifest_row["split"],
        "sequence_key": manifest_row["sequence_key"],
        "status": summary.get("status", "unknown"),
        "frame_count": frame_count,
        "expected_gloved_hands": expected,
        "nonempty_rate": nonempty / frame_count if frame_count else 0.0,
        "expected_cardinality_rate": exact / frame_count if frame_count else 0.0,
        "mean_retained_tracks": sum(cardinalities) / frame_count if frame_count else 0.0,
        "chunk_count": len(chunks),
        "chunk_boundary_frame_count": len(boundary_values),
        "chunk_boundary_nonempty_rate": coverage(boundary_values, lambda value: value > 0),
        "chunk_boundary_expected_cardinality_rate": coverage(
            boundary_values, lambda value: value == expected
        ),
        "chunk_interior_frame_count": len(interior_values),
        "chunk_interior_nonempty_rate": coverage(interior_values, lambda value: value > 0),
        "chunk_interior_expected_cardinality_rate": coverage(
            interior_values, lambda value: value == expected
        ),
        "reentry_selected_track_count": len(reentry_ids),
        "median_prompt_score": median(scores) if scores else None,
        "track_id_count": len(observations_by_id),
        "track_fragment_count": fragment_count,
        "median_center_jump": median(jumps) if jumps else None,
        "p95_center_jump": sorted(jumps)[int(0.95 * (len(jumps) - 1))] if jumps else None,
        "median_abs_log_area_change": median(area_log_changes) if area_log_changes else None,
        "rejected_observation_count": rejected,
        "sam3_native_observation_count": bbox_source_counts["sam3_native"],
        "sam3_flow_agreed_observation_count": bbox_source_counts["sam3_flow_agreed"],
        "flow_short_bridge_observation_count": bbox_source_counts["flow_short_bridge"],
        "semantic_motion_conflict_observation_count": bbox_source_counts[
            "semantic_motion_conflict"
        ],
    }


def aggregate(rows: list[dict]) -> dict:
    valid = [row for row in rows if row.get("status") == "complete"]
    total_frames = sum(int(row.get("frame_count", 0)) for row in valid)
    weighted_fields = (
        "nonempty_rate",
        "expected_cardinality_rate",
        "mean_retained_tracks",
    )
    result = {
        "sequence_count": len(rows),
        "complete_sequence_count": len(valid),
        "frame_count": total_frames,
    }
    for field in (
        "sam3_native_observation_count",
        "sam3_flow_agreed_observation_count",
        "flow_short_bridge_observation_count",
        "semantic_motion_conflict_observation_count",
    ):
        result[field] = sum(int(row.get(field, 0)) for row in valid)
    for field in weighted_fields:
        result[field] = (
            sum(float(row[field]) * int(row["frame_count"]) for row in valid) / total_frames
            if total_frames
            else None
        )
    for region in ("chunk_boundary", "chunk_interior"):
        count_field = f"{region}_frame_count"
        region_frames = sum(int(row.get(count_field, 0)) for row in valid)
        result[count_field] = region_frames
        for suffix in ("nonempty_rate", "expected_cardinality_rate"):
            field = f"{region}_{suffix}"
            result[field] = (
                sum(
                    float(row[field]) * int(row[count_field])
                    for row in valid
                    if row.get(field) is not None
                )
                / region_frames
                if region_frames
                else None
            )
    result["reentry_selected_track_count"] = sum(
        int(row.get("reentry_selected_track_count", 0)) for row in valid
    )
    for field in ("median_prompt_score", "median_center_jump", "p95_center_jump"):
        values = [float(row[field]) for row in valid if row.get(field) is not None]
        result[f"sequence_median_{field}"] = median(values) if values else None
    result["track_fragment_count"] = sum(
        int(row.get("track_fragment_count", 0)) for row in valid
    )
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--datasets", default="opentouch,touchanything")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_root / "reports" / "track_quality",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pilot_dir = args.pilot_dir.expanduser().resolve()
    manifest_path = pilot_dir / "pilot_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    datasets = set(parse_dataset_selection(args.datasets))
    manifest_rows = [
        row for row in read_jsonl(manifest_path) if row.get("dataset") in datasets
    ]
    metrics = []
    for row in manifest_rows:
        job_dir = pilot_dir / "results" / row["dataset"] / row["split"] / row["job_id"]
        metrics.append(sequence_metrics(job_dir, row))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "sequence_metrics.csv", metrics)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in metrics:
        grouped[f"{row['dataset']}/{row['split']}"].append(row)
        grouped[row["dataset"]].append(row)
    report = {key: aggregate(value) for key, value in sorted(grouped.items())}
    report["all"] = aggregate(metrics)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Track quality report: {output_dir}")
    for key, value in report.items():
        print(
            f"  {key}: complete={value['complete_sequence_count']}/"
            f"{value['sequence_count']} nonempty={value.get('nonempty_rate')} "
            f"cardinality={value.get('expected_cardinality_rate')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
