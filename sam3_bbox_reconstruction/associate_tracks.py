#!/usr/bin/env python3
"""Associate anonymous SAM tracks with tactile queries without model identity input.

OpenTouch has one tactile glove, so every accepted SAM observation can be
exported directly. TouchAnything has two pressure channels. Its default domain
policy maps the initially left image track to the left channel and the initially
right track to the right channel. Sparse legacy boxes remain available as an
optional audit/control policy but do not affect the default assignment.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable

try:
    from .defaults import (
        DEFAULT_OPENTOUCH_DATA_ROOT,
        DEFAULT_OPENTOUCH_EXTRACTED_ROOT,
        DEFAULT_OPENTOUCH_SPLITS,
        DEFAULT_TOUCHANYTHING_EXTRACTED_ROOT,
    )
    from .progress import progress
    from .track_selection import bbox_iou
except ImportError:
    from defaults import (
        DEFAULT_OPENTOUCH_DATA_ROOT,
        DEFAULT_OPENTOUCH_EXTRACTED_ROOT,
        DEFAULT_OPENTOUCH_SPLITS,
        DEFAULT_TOUCHANYTHING_EXTRACTED_ROOT,
    )
    from progress import progress
    from track_selection import bbox_iou


BBox = tuple[float, float, float, float]
PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Observation:
    frame_index: int
    raw_track_id: int
    bbox: BBox
    prompt_score: float | None
    bbox_source: str = "sam3_native"
    flow_confidence: float | None = None
    flow_bbox_iou: float | None = None
    flow_anchor_frames: tuple[int, ...] = ()


@dataclass
class Tracklet:
    tracklet_id: int
    raw_track_id: int
    observations: list[Observation]

    @property
    def start(self) -> int:
        return self.observations[0].frame_index

    @property
    def end(self) -> int:
        return self.observations[-1].frame_index


@dataclass
class Association:
    association_id: int
    tracklets: list[Tracklet]
    observations: list[Observation] = field(default_factory=list)
    target_hand: str | None = None
    confidence: str = "low"
    evidence: dict = field(default_factory=dict)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def resolve_pilot_dir(value: Path) -> Path:
    """Resolve current and early pilot output layouts without guessing silently."""

    requested = value.expanduser().resolve(strict=False)
    candidates = [requested]
    if requested.name == "results":
        candidates.append(requested.parent)
    # Early runs wrote pilot_manifest.jsonl and results/ directly below the
    # package directory. Newer domain wrappers use outputs/pilot_<domain>/.
    candidates.append(PACKAGE_ROOT)
    seen = set()
    valid = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "results").is_dir():
            valid.append(candidate)
    if requested in valid:
        return requested
    if len(valid) == 1:
        print(f"Resolved pilot directory {value} -> {valid[0]}")
        return valid[0]
    discovered = sorted(
        path.parent
        for path in (PACKAGE_ROOT / "outputs").glob("*/pilot_manifest.jsonl")
        if (path.parent / "results").is_dir()
    )
    if len(discovered) == 1:
        print(f"Resolved pilot directory {value} -> {discovered[0]}")
        return discovered[0]
    choices = [*valid, *discovered]
    detail = "\n  - ".join(str(path) for path in choices) or "(none found)"
    raise FileNotFoundError(
        f"Could not resolve a pilot root from {value}. A pilot root must contain results/. "
        f"Candidates found:\n  - {detail}"
    )


def _opentouch_job_lookup(split_path: Path, data_root: Path) -> dict[tuple[str, str], dict]:
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    lookup = {}
    for split, entries in split_data.items():
        for scene, demo in entries:
            sequence_key = f"{scene}/{demo}"
            job_id = f"opentouch__{split}__{sanitize_component(sequence_key)}"
            source = None
            for suffix in (".hdf5", ".h5"):
                candidate = data_root / f"{scene}{suffix}"
                if candidate.is_file():
                    source = candidate
                    break
            lookup[(str(split), job_id)] = {
                "sequence_key": sequence_key,
                "source_path": (
                    f"{source.resolve()}::data/{demo}/rgb_images_jpeg"
                    if source is not None
                    else None
                ),
            }
    return lookup


def rebuild_manifest_from_results(
    pilot_dir: Path,
    *,
    opentouch_splits: Path = DEFAULT_OPENTOUCH_SPLITS,
    opentouch_data_root: Path = DEFAULT_OPENTOUCH_DATA_ROOT,
) -> Path:
    """Recover a deleted pilot manifest from completed per-job summaries."""

    summaries = sorted((pilot_dir / "results").glob("*/*/*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No result summaries found below {pilot_dir / 'results'}")
    ot_lookup = _opentouch_job_lookup(
        opentouch_splits.expanduser().resolve(),
        opentouch_data_root.expanduser().resolve(),
    )
    records = []
    for summary_path in progress(
        summaries,
        desc="Recover pilot manifest",
        unit="job",
    ):
        job_dir = summary_path.parent
        split = job_dir.parent.name
        dataset = job_dir.parent.parent.name
        job_id = job_dir.name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        resource = Path(summary.get("resource", "")).expanduser().resolve(strict=False)
        if dataset == "opentouch":
            source = ot_lookup.get((split, job_id))
            if source is None:
                raise RuntimeError(
                    f"Could not map existing OpenTouch result {job_id!r} through "
                    f"{opentouch_splits}"
                )
            sequence_key = source["sequence_key"]
            source_path = source["source_path"] or str(resource)
            resource_type = "jpeg_directory"
            expected_gloved_hands = 1
        elif dataset == "touchanything":
            if resource.name != "chest.mp4" or len(resource.parent.parts) < 3:
                raise RuntimeError(f"Cannot recover TouchAnything sequence from {resource}")
            sequence_key = "/".join(resource.parent.parts[-3:])
            source_path = str(resource)
            resource_type = "video"
            expected_gloved_hands = 2
        else:
            continue
        records.append(
            {
                "job_id": job_id,
                "dataset": dataset,
                "split": split,
                "sequence_key": sequence_key,
                "resource_path": str(resource),
                "resource_type": resource_type,
                "source_path": source_path,
                "expected_gloved_hands": expected_gloved_hands,
                "prompt_preset": summary.get("prompt_preset", "gloved"),
            }
        )
    if not records:
        raise RuntimeError(f"No supported completed jobs found below {pilot_dir / 'results'}")
    records.sort(key=lambda row: (row["dataset"], row["split"], row["sequence_key"]))
    manifest_path = pilot_dir / "pilot_manifest.jsonl"
    write_jsonl(manifest_path, records)
    audit = {
        "schema": "sam3_recovered_pilot_manifest_v1",
        "record_count": len(records),
        "source": "completed_result_summaries",
        "manifest": str(manifest_path),
    }
    (pilot_dir / "pilot_manifest.recovered.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(f"Recovered {len(records)} manifest rows from existing results: {manifest_path}")
    return manifest_path


def as_bbox(value: Iterable[float] | None) -> BBox | None:
    if value is None:
        return None
    try:
        box = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(box) != 4 or not all(math.isfinite(item) for item in box):
        return None
    if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
        return None
    return box  # type: ignore[return-value]


def bbox_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center_distance_ratio(lhs: BBox, rhs: BBox) -> float:
    lhs_center = ((lhs[0] + lhs[2]) * 0.5, (lhs[1] + lhs[3]) * 0.5)
    rhs_center = ((rhs[0] + rhs[2]) * 0.5, (rhs[1] + rhs[3]) * 0.5)
    distance = math.hypot(lhs_center[0] - rhs_center[0], lhs_center[1] - rhs_center[1])
    reference = max(1.0, math.sqrt(max(bbox_area(lhs), bbox_area(rhs))))
    return distance / reference


def area_ratio(lhs: BBox, rhs: BBox) -> float:
    lhs_area = max(1.0, bbox_area(lhs))
    rhs_area = max(1.0, bbox_area(rhs))
    return max(lhs_area / rhs_area, rhs_area / lhs_area)


def compatible_transition(
    previous: Observation,
    current: Observation,
    *,
    max_gap: int,
    max_center_ratio: float,
    max_area_ratio: float,
) -> bool:
    gap = current.frame_index - previous.frame_index
    if gap < 1 or gap > max_gap:
        return False
    if area_ratio(previous.bbox, current.bbox) > max_area_ratio:
        return False
    center_ratio = center_distance_ratio(previous.bbox, current.bbox)
    return center_ratio <= max_center_ratio or bbox_iou(previous.bbox, current.bbox) >= 0.05


def observations_from_frames(frame_rows: Iterable[dict]) -> list[Observation]:
    observations = []
    for frame in frame_rows:
        frame_index = int(frame["frame_index"])
        for track in frame.get("tracks", []):
            bbox = as_bbox(track.get("bbox"))
            if bbox is None:
                continue
            score = track.get("prompt_score")
            observations.append(
                Observation(
                    frame_index=frame_index,
                    raw_track_id=int(track["track_id"]),
                    bbox=bbox,
                    prompt_score=None if score is None else float(score),
                    bbox_source=str(track.get("bbox_source", "sam3_native")),
                    flow_confidence=(
                        None
                        if track.get("flow_confidence") is None
                        else float(track["flow_confidence"])
                    ),
                    flow_bbox_iou=(
                        None
                        if track.get("flow_bbox_iou") is None
                        else float(track["flow_bbox_iou"])
                    ),
                    flow_anchor_frames=tuple(
                        int(value) for value in track.get("flow_anchor_frames", ())
                    ),
                )
            )
    return sorted(observations, key=lambda row: (row.raw_track_id, row.frame_index))


def split_tracklets(
    observations: Iterable[Observation],
    *,
    max_gap: int = 3,
    max_center_ratio: float = 1.25,
    max_area_ratio: float = 3.0,
) -> list[Tracklet]:
    by_id: dict[int, list[Observation]] = defaultdict(list)
    for observation in observations:
        by_id[observation.raw_track_id].append(observation)
    tracklets: list[Tracklet] = []
    for raw_track_id, rows in sorted(by_id.items()):
        rows.sort(key=lambda row: row.frame_index)
        current: list[Observation] = []
        for row in rows:
            if current and not compatible_transition(
                current[-1],
                row,
                max_gap=max_gap,
                max_center_ratio=max_center_ratio,
                max_area_ratio=max_area_ratio,
            ):
                tracklets.append(Tracklet(len(tracklets), raw_track_id, current))
                current = []
            current.append(row)
        if current:
            tracklets.append(Tracklet(len(tracklets), raw_track_id, current))
    return tracklets


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, lhs: int, rhs: int) -> None:
        lhs_root, rhs_root = self.find(lhs), self.find(rhs)
        if lhs_root != rhs_root:
            self.parent[rhs_root] = lhs_root


def reconnect_tracklets(
    tracklets: list[Tracklet],
    *,
    max_link_gap: int = 30,
    max_center_ratio: float = 1.0,
    max_area_ratio: float = 2.5,
) -> list[Association]:
    """Reconnect compatible fragments without using handedness or image side."""

    union = _UnionFind(len(tracklets))
    claimed_successors: set[int] = set()
    for previous in sorted(tracklets, key=lambda item: item.end):
        candidates = []
        endpoint = previous.observations[-1]
        for current in tracklets:
            if current.tracklet_id == previous.tracklet_id or current.tracklet_id in claimed_successors:
                continue
            gap = current.start - previous.end
            if gap < 1 or gap > max_link_gap:
                continue
            startpoint = current.observations[0]
            center = center_distance_ratio(endpoint.bbox, startpoint.bbox)
            areas = area_ratio(endpoint.bbox, startpoint.bbox)
            if center > max_center_ratio or areas > max_area_ratio:
                continue
            identity_bonus = -0.25 if previous.raw_track_id == current.raw_track_id else 0.0
            candidates.append((center + 0.05 * gap + identity_bonus, current))
        if candidates:
            _, current = min(candidates, key=lambda item: item[0])
            union.union(previous.tracklet_id, current.tracklet_id)
            claimed_successors.add(current.tracklet_id)
    grouped: dict[int, list[Tracklet]] = defaultdict(list)
    for tracklet in tracklets:
        grouped[union.find(tracklet.tracklet_id)].append(tracklet)
    associations = []
    for association_id, (_, parts) in enumerate(sorted(grouped.items())):
        observations = sorted(
            (row for part in parts for row in part.observations),
            key=lambda row: row.frame_index,
        )
        associations.append(
            Association(
                association_id=association_id,
                tracklets=sorted(parts, key=lambda item: item.start),
                observations=observations,
            )
        )
    return associations


def sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def load_touchanything_anchors(
    extracted_root: Path,
    split: str,
    sequence_key: str,
    frame_indices: Iterable[int],
) -> tuple[dict[int, dict[str, BBox]], dict[int, str]]:
    parts = sequence_key.split("/")
    if len(parts) != 3:
        raise ValueError(f"Expected TouchAnything sequence scene/task/clip, got {sequence_key!r}")
    scene, task, clip = parts
    prefix = "__".join(sanitize_component(item) for item in (scene, task, clip))
    anchors: dict[int, dict[str, BBox]] = {}
    sample_dirs: dict[int, str] = {}
    for frame_index in sorted(set(int(value) for value in frame_indices)):
        sample_dir = extracted_root / split / f"{prefix}__{frame_index:06d}"
        meta_path = sample_dir / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        frame_anchors = {}
        for hand in ("left", "right"):
            bbox = as_bbox(meta.get("hands", {}).get(hand, {}).get("bbox_chest"))
            if bbox is not None:
                frame_anchors[hand] = bbox
        if frame_anchors:
            anchors[frame_index] = frame_anchors
        sample_dirs[frame_index] = str(sample_dir)
    return anchors, sample_dirs


def touchanything_sample_dirs(
    extracted_root: Path,
    split: str,
    sequence_key: str,
    frame_indices: Iterable[int],
) -> dict[int, str]:
    """Resolve deterministic TA sample paths without reading legacy bbox metadata."""

    parts = sequence_key.split("/")
    if len(parts) != 3:
        raise ValueError(f"Expected TouchAnything sequence scene/task/clip, got {sequence_key!r}")
    prefix = "__".join(sanitize_component(item) for item in parts)
    return {
        frame_index: str(extracted_root / split / f"{prefix}__{frame_index:06d}")
        for frame_index in sorted(set(int(value) for value in frame_indices))
    }


def anchor_similarity(observation: Observation, anchor: BBox) -> float:
    overlap = bbox_iou(observation.bbox, anchor)
    center = center_distance_ratio(observation.bbox, anchor)
    return 0.75 * overlap + 0.25 * math.exp(-center)


def score_association(
    association: Association,
    anchors: dict[int, dict[str, BBox]],
) -> dict:
    similarities = {"left": [], "right": []}
    paired_rows = []
    for observation in association.observations:
        frame_anchors = anchors.get(observation.frame_index, {})
        row_scores = {}
        for hand in ("left", "right"):
            anchor = frame_anchors.get(hand)
            if anchor is not None:
                score = anchor_similarity(observation, anchor)
                similarities[hand].append(score)
                row_scores[hand] = score
        if row_scores:
            paired_rows.append(row_scores)
    hand_scores = {
        hand: mean(values) if values else None for hand, values in similarities.items()
    }
    finite_scores = {hand: value for hand, value in hand_scores.items() if value is not None}
    local_best_hand = max(finite_scores, key=finite_scores.get) if finite_scores else None
    available = [value for value in hand_scores.values() if value is not None]
    if len(available) == 2:
        margin = abs(hand_scores["left"] - hand_scores["right"])
    elif len(available) == 1:
        # A single reliable legacy hand channel is still useful evidence.  The
        # 0.25 reference is the centre-only floor of anchor_similarity; a
        # strong overlap must exceed it before becoming high confidence.
        margin = max(0.0, float(available[0]) - 0.25)
    else:
        margin = 0.0
    return {
        "anchor_frame_count": len(paired_rows),
        "hand_scores": hand_scores,
        "local_best_hand": local_best_hand,
        "margin": margin,
        "paired_rows": paired_rows,
    }


def observation_center_x(observation: Observation) -> float:
    return 0.5 * (observation.bbox[0] + observation.bbox[2])


def assign_touchanything_screen_order(
    associations: list[Association],
) -> list[Association]:
    """Assign the initially left/right visible tracks to left/right channels.

    The primary pair is taken from the earliest frame containing two distinct
    associations. This is independent of SAM object/query IDs and legacy bbox
    anchors. Additional non-overlapping fragments are assigned by their initial
    x position relative to that primary pair. A sequence with only one detected
    association remains unassigned because image side alone cannot establish a
    two-hand ordering.
    """

    for association in associations:
        association.target_hand = None
        association.confidence = "low"
        first = min(association.observations, key=lambda row: row.frame_index)
        association.evidence = {
            "assignment_policy": "initial_screen_order",
            "initial_frame": first.frame_index,
            "initial_center_x": observation_center_x(first),
            "legacy_anchor_used": False,
        }

    by_frame: dict[int, list[tuple[Association, Observation]]] = defaultdict(list)
    for association in associations:
        for observation in association.observations:
            by_frame[observation.frame_index].append((association, observation))

    primary_rows: list[tuple[Association, Observation]] | None = None
    for frame_index in sorted(by_frame):
        rows = by_frame[frame_index]
        distinct = {item.association_id for item, _ in rows}
        if len(distinct) < 2:
            continue
        # Upstream tracking is capped at two TA queries. Keep the two longest
        # associations if malformed/manual input contains more candidates.
        rows = sorted(
            rows,
            key=lambda pair: (-len(pair[0].observations), observation_center_x(pair[1])),
        )
        unique_rows = []
        seen_ids = set()
        for pair in rows:
            if pair[0].association_id in seen_ids:
                continue
            unique_rows.append(pair)
            seen_ids.add(pair[0].association_id)
            if len(unique_rows) == 2:
                break
        if len(unique_rows) == 2:
            primary_rows = unique_rows
            break

    if primary_rows is None:
        # This fallback covers two non-overlapping fragments. It deliberately
        # does not guess when only one association exists in the whole clip.
        dominant = sorted(
            associations,
            key=lambda item: (-len(item.observations), item.observations[0].frame_index),
        )[:2]
        if len(dominant) < 2:
            return associations
        primary_rows = [
            (item, min(item.observations, key=lambda row: row.frame_index))
            for item in dominant
        ]
        primary_source = "nonoverlap_initial_fallback"
    else:
        primary_source = "earliest_simultaneous_pair"

    ordered = sorted(primary_rows, key=lambda pair: observation_center_x(pair[1]))
    left_pair, right_pair = ordered
    left_x = observation_center_x(left_pair[1])
    right_x = observation_center_x(right_pair[1])
    midpoint = 0.5 * (left_x + right_x)
    for hand, (association, observation) in zip(("left", "right"), ordered):
        association.target_hand = hand
        association.confidence = "high"
        association.evidence.update(
            {
                "assignment_source": primary_source,
                "assignment_frame": observation.frame_index,
                "assignment_center_x": observation_center_x(observation),
                "primary_left_center_x": left_x,
                "primary_right_center_x": right_x,
            }
        )

    primary_ids = {left_pair[0].association_id, right_pair[0].association_id}
    for association in associations:
        if association.association_id in primary_ids:
            continue
        first = min(association.observations, key=lambda row: row.frame_index)
        center_x = observation_center_x(first)
        association.target_hand = "left" if center_x < midpoint else "right"
        association.confidence = "high"
        association.evidence.update(
            {
                "assignment_source": "fragment_initial_x_relative_to_primary_pair",
                "assignment_frame": first.frame_index,
                "assignment_center_x": center_x,
                "primary_left_center_x": left_x,
                "primary_right_center_x": right_x,
            }
        )

    # A duplicated/reconnected fragment must never silently create two boxes for
    # the same pressure channel in one frame. Keep the conflict visible and out
    # of the high-confidence training manifest.
    by_frame_hand: dict[tuple[int, str], list[Association]] = defaultdict(list)
    for association in associations:
        if association.target_hand is None:
            continue
        for observation in association.observations:
            by_frame_hand[(observation.frame_index, association.target_hand)].append(association)
    for collisions in by_frame_hand.values():
        if len({item.association_id for item in collisions}) > 1:
            for association in collisions:
                association.confidence = "low"
                association.evidence["simultaneous_same_hand_collision"] = True
    return associations


def assign_touchanything_associations(
    associations: list[Association],
    anchors: dict[int, dict[str, BBox]],
    *,
    high_min_anchors: int = 3,
    high_margin: float = 0.15,
    medium_margin: float = 0.05,
) -> list[Association]:
    """Legacy control: assign left/right from sparse bbox anchors."""

    for association in associations:
        association.evidence = score_association(association, anchors)
        scores = association.evidence["hand_scores"]
        finite = {hand: value for hand, value in scores.items() if value is not None}
        if finite:
            association.target_hand = max(finite, key=finite.get)  # type: ignore[arg-type]

    dominant = sorted(associations, key=lambda item: len(item.observations), reverse=True)[:2]
    if len(dominant) == 2:
        lhs, rhs = dominant
        lhs_scores, rhs_scores = lhs.evidence["hand_scores"], rhs.evidence["hand_scores"]

        def score(item: dict, hand: str) -> float:
            value = item.get(hand)
            return float(value) if value is not None else 0.0

        normal = score(lhs_scores, "left") + score(rhs_scores, "right")
        swapped = score(lhs_scores, "right") + score(rhs_scores, "left")
        if max(normal, swapped) > 0:
            lhs.target_hand, rhs.target_hand = (
                ("left", "right") if normal >= swapped else ("right", "left")
            )
            lhs.evidence["two_query_assignment_margin"] = abs(normal - swapped)
            rhs.evidence["two_query_assignment_margin"] = abs(normal - swapped)
            for item in dominant:
                if item.evidence["hand_scores"].get(item.target_hand) is None:
                    item.evidence["assignment_inferred_from_other_query"] = True

    for association in associations:
        evidence = association.evidence
        chosen = association.target_hand
        anchor_count = int(evidence["anchor_frame_count"])
        margin = float(evidence["margin"])
        scores = evidence["hand_scores"]
        chosen_score = scores.get(chosen) if chosen else None
        contradictions = 0
        comparable = 0
        if chosen:
            other = "right" if chosen == "left" else "left"
            for row in evidence["paired_rows"]:
                if chosen in row and other in row:
                    comparable += 1
                    contradictions += row[other] > row[chosen]
        contradiction_rate = contradictions / comparable if comparable else 0.0
        evidence["contradiction_rate"] = contradiction_rate
        if (
            chosen is not None
            and anchor_count >= high_min_anchors
            and margin >= high_margin
            and chosen == evidence.get("local_best_hand")
            and chosen_score is not None
            and chosen_score >= 0.25
            and contradiction_rate <= 0.20
        ):
            association.confidence = "high"
        elif chosen is not None and (
            (anchor_count >= 1 and margin >= medium_margin)
            or bool(evidence.get("assignment_inferred_from_other_query"))
        ):
            association.confidence = "medium"
        else:
            association.confidence = "low"

    by_frame_hand: dict[tuple[int, str], list[Association]] = defaultdict(list)
    for association in associations:
        if association.target_hand is None:
            continue
        for observation in association.observations:
            by_frame_hand[(observation.frame_index, association.target_hand)].append(association)
    for collisions in by_frame_hand.values():
        if len({item.association_id for item in collisions}) > 1:
            for association in collisions:
                association.confidence = "low"
                association.evidence["simultaneous_same_hand_collision"] = True
    return associations


def association_output_rows(
    manifest_row: dict,
    associations: Iterable[Association],
    sample_dirs: dict[int, str],
    *,
    compact: bool = False,
) -> list[dict]:
    rows = []
    for association in associations:
        compact_evidence = {
            key: value
            for key, value in association.evidence.items()
            if key != "paired_rows"
        }
        for observation in association.observations:
            row = {
                "dataset": "touchanything",
                "split": manifest_row["split"],
                "sequence_key": manifest_row["sequence_key"],
                "frame_idx": observation.frame_index,
                "target_hand": association.target_hand,
                "bbox": list(observation.bbox),
                "prompt_score": observation.prompt_score,
                "bbox_source": observation.bbox_source,
                "flow_confidence": observation.flow_confidence,
                "flow_bbox_iou": observation.flow_bbox_iou,
                "flow_anchor_frames": list(observation.flow_anchor_frames),
                "association_id": association.association_id,
                "association_confidence": association.confidence,
                "sample_dir": sample_dirs.get(observation.frame_index),
            }
            if not compact:
                row.update(
                    {
                        "schema": "sam3_touchanything_association_v1",
                        "raw_track_id": observation.raw_track_id,
                        "association_evidence": compact_evidence,
                        "source_video": manifest_row["resource_path"],
                    }
                )
            rows.append(row)
    return sorted(rows, key=lambda row: (row["frame_idx"], row["association_id"]))


def load_opentouch_sample_dirs(
    extracted_root: Path,
    split: str,
    sequence_key: str,
    frame_indices: Iterable[int],
) -> tuple[dict[int, str], dict[int, list[str]]]:
    parts = sequence_key.split("/")
    if len(parts) != 2:
        raise ValueError(f"Expected OpenTouch sequence scene/demo, got {sequence_key!r}")
    scene, demo = parts
    sample_dirs = {}
    ambiguous = {}
    split_root = extracted_root / split
    for frame_index in sorted(set(int(value) for value in frame_indices)):
        prefix = f"{scene}_{demo}_{frame_index:04d}_"
        candidates = sorted(
            path
            for path in split_root.glob(f"{prefix}*")
            if path.is_dir() and (path / "meta.json").is_file()
        )
        if len(candidates) == 1:
            sample_dirs[frame_index] = str(candidates[0])
        elif len(candidates) > 1:
            ambiguous[frame_index] = [str(path) for path in candidates]
    return sample_dirs, ambiguous


def build_opentouch_sample_index(
    extracted_root: Path,
    manifest_rows: Iterable[dict],
) -> dict[tuple[str, str, int], list[str]]:
    """Index an OT extracted split once instead of globbing once per frame."""

    prefixes_by_split: dict[str, dict[str, str]] = defaultdict(dict)
    for row in manifest_rows:
        if row.get("dataset") != "opentouch":
            continue
        sequence_key = str(row["sequence_key"])
        scene, demo = sequence_key.split("/", 1)
        prefix = f"{scene}_{demo}"
        existing = prefixes_by_split[str(row["split"])].get(prefix)
        if existing is not None and existing != sequence_key:
            raise RuntimeError(f"Ambiguous OpenTouch extracted prefix {prefix!r}")
        prefixes_by_split[str(row["split"])][prefix] = sequence_key
    result: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    pattern = re.compile(r"^(.+)_([0-9]+)_([01])$")
    for split, prefixes in prefixes_by_split.items():
        split_root = extracted_root / split
        if not split_root.is_dir():
            continue
        for entry in progress(
            os.scandir(split_root),
            desc=f"Index OpenTouch extracted {split}",
            unit="sample",
        ):
            if not entry.is_dir(follow_symlinks=False):
                continue
            match = pattern.match(entry.name)
            if match is None:
                continue
            sequence_key = prefixes.get(match.group(1))
            if sequence_key is None:
                continue
            meta_path = Path(entry.path) / "meta.json"
            if meta_path.is_file():
                result[(split, sequence_key, int(match.group(2)))].append(entry.path)
    return dict(result)


def export_opentouch_rows(
    manifest_row: dict,
    frame_rows: Iterable[dict],
    sample_dirs: dict[int, str],
    *,
    compact: bool = False,
) -> list[dict]:
    frame_rows = sorted(frame_rows, key=lambda row: int(row["frame_index"]))
    candidates_by_frame = {
        int(frame["frame_index"]): [
            track for track in frame.get("tracks", []) if as_bbox(track.get("bbox")) is not None
        ]
        for frame in frame_rows
    }

    def source_score(track: dict) -> float:
        return {
            "sam3_flow_agreed": 2.0,
            "sam3_native": 1.0,
            "flow_short_bridge": 0.5,
            "semantic_motion_conflict": -4.0,
        }.get(str(track.get("bbox_source", "sam3_native")), 0.0)

    def choose_track(frame_index: int, candidates: list[dict]) -> tuple[dict, dict | None]:
        if len(candidates) == 1:
            return candidates[0], None
        neighbour_frames = [
            other
            for other in range(frame_index - 2, frame_index + 3)
            if other != frame_index and candidates_by_frame.get(other)
        ]
        ranked = []
        for track in candidates:
            bbox = as_bbox(track["bbox"])
            assert bbox is not None
            track_id = int(track["track_id"])
            same_id_support = 0
            overlap_support = 0.0
            for other_frame in neighbour_frames:
                neighbours = candidates_by_frame[other_frame]
                same_id_support += any(
                    int(neighbour["track_id"]) == track_id for neighbour in neighbours
                )
                overlap_support += max(
                    bbox_iou(bbox, as_bbox(neighbour["bbox"]))
                    for neighbour in neighbours
                    if as_bbox(neighbour["bbox"]) is not None
                )
            prompt_score = float(track.get("prompt_score") or 0.0)
            flow_confidence = float(track.get("flow_confidence") or 0.0)
            evidence_score = (
                5.0 * same_id_support
                + 2.0 * overlap_support
                + source_score(track)
                + prompt_score
                + 0.25 * flow_confidence
            )
            ranked.append((evidence_score, prompt_score, -track_id, track))
        ranked.sort(key=lambda item: item[:3], reverse=True)
        winner = ranked[0][3]
        resolution = {
            "policy": "single_query_temporal_evidence_v1",
            "candidate_count": len(candidates),
            "selected_track_id": int(winner["track_id"]),
            "discarded_track_ids": [
                int(item[3]["track_id"]) for item in ranked[1:]
            ],
            "candidate_evidence_scores": {
                str(int(item[3]["track_id"])): float(item[0]) for item in ranked
            },
        }
        return winner, resolution

    rows = []
    for frame in frame_rows:
        frame_index = int(frame["frame_index"])
        candidates = candidates_by_frame.get(frame_index, [])
        if not candidates:
            continue
        track, resolution = choose_track(frame_index, candidates)
        for track in (track,):
            bbox = as_bbox(track.get("bbox"))
            if bbox is None:
                continue
            row = {
                "dataset": "opentouch",
                "split": manifest_row["split"],
                "sequence_key": manifest_row["sequence_key"],
                "frame_idx": int(frame["frame_index"]),
                "target_hand": "source_pressure_hand",
                "bbox": list(bbox),
                "prompt_score": track.get("prompt_score"),
                "bbox_source": track.get("bbox_source", "sam3_native"),
                "flow_confidence": track.get("flow_confidence"),
                "flow_bbox_iou": track.get("flow_bbox_iou"),
                "flow_anchor_frames": list(track.get("flow_anchor_frames", ())),
                "association_confidence": "single_gloved_query",
                "sample_dir": sample_dirs.get(int(frame["frame_index"])),
                "single_slot_resolution": resolution,
            }
            if not compact:
                row.update(
                    {
                        "schema": "sam3_opentouch_bbox_v1",
                        "raw_track_id": int(track["track_id"]),
                        "source_resource": manifest_row["resource_path"],
                    }
                )
            rows.append(row)
    return rows


def render_touchanything_association_preview(
    video_path: Path,
    rows: Iterable[dict],
    output_path: Path,
) -> Path | None:
    """Render offline left/right pressure-channel assignments over the video."""

    rows_by_frame: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_frame[int(row["frame_idx"])].append(row)
    if not rows_by_frame:
        return None
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open TouchAnything video for association preview: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create association preview: {output_path}")
    confidence_colors = {
        "high": (80, 220, 90),
        "medium": (60, 210, 240),
        "low": (70, 90, 245),
    }
    max_frame = max(rows_by_frame)
    frame_index = 0
    try:
        while frame_index <= max_frame:
            ok, frame = capture.read()
            if not ok:
                break
            for row in rows_by_frame.get(frame_index, []):
                x1, y1, x2, y2 = (int(round(value)) for value in row["bbox"])
                confidence = str(row["association_confidence"])
                target_hand = row.get("target_hand")
                hand_label = str(target_hand) if target_hand is not None else "unassigned"
                uncertainty = "" if confidence == "high" else "?"
                label = (
                    f"{hand_label}{uncertainty} [{confidence}] "
                    f"track={row['raw_track_id']} assoc={row['association_id']}"
                )
                color = confidence_colors.get(confidence, confidence_colors["low"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"offline pressure-channel association | frame {frame_index}",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()
    return output_path


def write_association_gallery(pilot_dir: Path, audits: Iterable[dict]) -> Path:
    """Write one inspectable page for all TA handedness previews."""

    cards = []
    for audit in audits:
        preview_value = audit.get("association_preview")
        if not preview_value:
            continue
        preview_path = Path(preview_value)
        try:
            source = preview_path.relative_to(pilot_dir).as_posix()
        except ValueError:
            source = preview_path.as_uri()
        title = f"{audit.get('split', 'unknown')} | {audit.get('sequence_key', 'unknown')}"
        cards.append(
            "\n".join(
                [
                    '<article class="card">',
                    f"<h2>{html.escape(title)}</h2>",
                    f'<video controls preload="metadata" src="{html.escape(source)}"></video>',
                    "</article>",
                ]
            )
        )
    page = "\n".join(
        [
            "<!doctype html>",
            '<html><head><meta charset="utf-8"><title>TA association previews</title>',
            "<style>body{font-family:sans-serif;margin:24px;background:#111;color:#eee}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}",
            ".card{background:#1c1c1c;padding:14px;border-radius:6px}video{width:100%}",
            "h1{font-size:24px}h2{font-size:15px;font-weight:500}</style></head><body>",
            "<h1>TouchAnything offline left/right association</h1>",
            '<main class="grid">',
            *cards,
            "</main></body></html>",
        ]
    )
    output_path = pilot_dir / "association_index.html"
    output_path.write_text(page, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument(
        "--opentouch-extracted-root",
        type=Path,
        default=DEFAULT_OPENTOUCH_EXTRACTED_ROOT,
    )
    parser.add_argument(
        "--touchanything-extracted-root",
        type=Path,
        default=DEFAULT_TOUCHANYTHING_EXTRACTED_ROOT,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--touchanything-association",
        choices=("screen_order", "legacy_anchor"),
        default="screen_order",
        help=(
            "TA handedness policy. screen_order maps the initially left/right visible "
            "tracks directly; legacy_anchor is retained only as a controlled audit."
        ),
    )
    parser.add_argument("--high-min-anchors", type=int, default=3)
    parser.add_argument("--high-margin", type=float, default=0.15)
    parser.add_argument("--medium-margin", type=float, default=0.05)
    parser.add_argument(
        "--association-previews",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render left/right/uncertain labels over each TA pilot video.",
    )
    parser.add_argument(
        "--compact-manifests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Omit repeated tracking-debug fields from writeback manifests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pilot_dir = resolve_pilot_dir(args.pilot_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pilot_dir / "manifests"
    )
    manifest_path = pilot_dir / "pilot_manifest.jsonl"
    if not manifest_path.is_file():
        manifest_path = rebuild_manifest_from_results(pilot_dir)
    manifest = read_jsonl(manifest_path)
    opentouch_extracted_root = args.opentouch_extracted_root.expanduser().resolve()
    opentouch_sample_index = build_opentouch_sample_index(
        opentouch_extracted_root,
        manifest,
    )
    touchanything_extracted_root = args.touchanything_extracted_root.expanduser().resolve()
    opentouch_rows: list[dict] = []
    ta_high: list[dict] = []
    ta_uncertain: list[dict] = []
    audits = []
    for manifest_row in progress(
        manifest,
        desc="Associate bbox sequences",
        unit="seq",
    ):
        job_dir = (
            pilot_dir
            / "results"
            / manifest_row["dataset"]
            / manifest_row["split"]
            / manifest_row["job_id"]
        )
        bbox_path = job_dir / "bboxes.jsonl"
        if not bbox_path.is_file():
            continue
        frame_rows = read_jsonl(bbox_path)
        if manifest_row["dataset"] == "opentouch":
            frame_indices = [int(row["frame_index"]) for row in frame_rows]
            sample_dirs = {}
            ambiguous = {}
            for frame_index in sorted(set(frame_indices)):
                candidates = opentouch_sample_index.get(
                    (manifest_row["split"], manifest_row["sequence_key"], frame_index),
                    [],
                )
                if len(candidates) == 1:
                    sample_dirs[frame_index] = candidates[0]
                elif len(candidates) > 1:
                    ambiguous[frame_index] = candidates
            exported = export_opentouch_rows(
                manifest_row,
                frame_rows,
                sample_dirs,
                compact=args.compact_manifests,
            )
            opentouch_rows.extend(exported)
            audits.append(
                {
                    "dataset": "opentouch",
                    "split": manifest_row["split"],
                    "sequence_key": manifest_row["sequence_key"],
                    "observation_count": len(exported),
                    "resolved_sample_count": sum(row.get("sample_dir") is not None for row in exported),
                    "ambiguous_sample_frames": ambiguous,
                    "association_policy": "single_gloved_query",
                }
            )
            continue
        observations = observations_from_frames(frame_rows)
        tracklets = split_tracklets(observations)
        associations = reconnect_tracklets(tracklets)
        if args.touchanything_association == "screen_order":
            anchors = {}
            sample_dirs = touchanything_sample_dirs(
                touchanything_extracted_root,
                manifest_row["split"],
                manifest_row["sequence_key"],
                (row.frame_index for row in observations),
            )
            assign_touchanything_screen_order(associations)
        else:
            anchors, sample_dirs = load_touchanything_anchors(
                touchanything_extracted_root,
                manifest_row["split"],
                manifest_row["sequence_key"],
                (row.frame_index for row in observations),
            )
            assign_touchanything_associations(
                associations,
                anchors,
                high_min_anchors=args.high_min_anchors,
                high_margin=args.high_margin,
                medium_margin=args.medium_margin,
            )
        rows = association_output_rows(
            manifest_row,
            associations,
            sample_dirs,
            compact=args.compact_manifests,
        )
        ta_high.extend(
            row
            for row in rows
            if row["association_confidence"] == "high" and row["target_hand"] is not None
        )
        ta_uncertain.extend(
            row
            for row in rows
            if row["association_confidence"] != "high" or row["target_hand"] is None
        )
        preview_path = None
        preview_error = None
        if args.association_previews:
            try:
                preview_path = render_touchanything_association_preview(
                    Path(manifest_row["resource_path"]),
                    rows,
                    job_dir / "association_preview.mp4",
                )
            except Exception as exc:
                preview_error = str(exc)
                print(
                    f"Warning: association preview failed for {manifest_row['sequence_key']}: {exc}"
                )
        else:
            (job_dir / "association_preview.mp4").unlink(missing_ok=True)
        audits.append(
            {
                "split": manifest_row["split"],
                "sequence_key": manifest_row["sequence_key"],
                "observation_count": len(observations),
                "legacy_anchor_frame_count": len(anchors),
                "tracklet_count": len(tracklets),
                "association_count": len(associations),
                "association_policy": args.touchanything_association,
                "association_preview": str(preview_path) if preview_path else None,
                "association_preview_error": preview_error,
                "associations": [
                    {
                        "association_id": item.association_id,
                        "raw_track_ids": sorted({part.raw_track_id for part in item.tracklets}),
                        "start": min(row.frame_index for row in item.observations),
                        "end": max(row.frame_index for row in item.observations),
                        "observation_count": len(item.observations),
                        "target_hand": item.target_hand,
                        "confidence": item.confidence,
                        "evidence": {
                            key: value
                            for key, value in item.evidence.items()
                            if key != "paired_rows"
                        },
                    }
                    for item in associations
                    if item.observations
                ],
            }
        )
    counts = {
        "opentouch": write_jsonl(output_dir / "opentouch_sam3_v1.jsonl", opentouch_rows),
        "touchanything_highconf": write_jsonl(
            output_dir / "touchanything_sam3_v1_highconf.jsonl", ta_high
        ),
        "touchanything_uncertain": write_jsonl(
            output_dir / "touchanything_sam3_v1_uncertain.jsonl", ta_uncertain
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "touchanything_association_audit.json").write_text(
        json.dumps(
            {
                "touchanything_association": args.touchanything_association,
                "compact_manifests": bool(args.compact_manifests),
                "counts": counts,
                "sequences": audits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    gallery_path = write_association_gallery(pilot_dir, audits)
    print(f"Association manifests: {output_dir}")
    print(f"Association preview gallery: {gallery_path}")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
