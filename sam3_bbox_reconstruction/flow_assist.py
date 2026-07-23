"""Conservative optical-flow assistance for prompt-validated SAM tracks.

SAM remains the semantic authority.  This module only measures short-term
motion agreement and fills a short gap when independent forward and backward
LK projections agree.  It never creates a track without two accepted SAM
anchors.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from math import exp, isfinite, sqrt
from typing import Any, Callable

try:
    from .progress import progress as progress_bar
    from .track_selection import TrackObservation, bbox_iou
except ImportError:
    from progress import progress as progress_bar
    from track_selection import TrackObservation, bbox_iou


@dataclass(frozen=True)
class FlowAssistConfig:
    max_gap: int = 5
    fb_error: float = 1.5
    min_points: int = 12
    min_inlier_ratio: float = 0.60
    min_confidence: float = 0.45
    sam_iou_accept: float = 0.50
    conflict_iou: float = 0.15
    bridge_policy: str = "off"
    score_decay: float = 0.98
    cache_frames: int = 16
    max_tracks_per_frame: int = 0


class _GrayFrameCache:
    def __init__(self, get_frame: Callable[[int], Any], capacity: int):
        self.get_frame = get_frame
        self.capacity = max(2, int(capacity))
        self.cache: OrderedDict[int, Any] = OrderedDict()

    def get(self, frame_index: int):
        import cv2

        if frame_index in self.cache:
            value = self.cache.pop(frame_index)
            self.cache[frame_index] = value
            return value
        frame = self.get_frame(frame_index)
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.cache[frame_index] = gray
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return gray


def _clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    values = (x1, y1, x2, y2)
    if not all(isfinite(float(value)) for value in values):
        return None
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _feature_points(gray, bbox: tuple[int, int, int, int], max_corners: int = 96):
    import cv2
    import numpy as np

    height, width = gray.shape[:2]
    x1, y1, x2, y2 = bbox
    inset_x = max(2, int(round((x2 - x1) * 0.08)))
    inset_y = max(2, int(round((y2 - y1) * 0.08)))
    left, right = max(0, x1 + inset_x), min(width, x2 - inset_x + 1)
    top, bottom = max(0, y1 + inset_y), min(height, y2 - inset_y + 1)
    if right <= left + 2 or bottom <= top + 2:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=7,
        useHarrisDetector=False,
    )


def estimate_lk_bbox_step(
    previous_gray,
    current_gray,
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    config: FlowAssistConfig,
) -> dict[str, Any]:
    """Project one bbox by pyramidal LK with forward/backward rejection."""

    import cv2
    import numpy as np

    points = _feature_points(previous_gray, bbox)
    if points is None or len(points) < config.min_points:
        return {"ok": False, "reason": "insufficient_features", "point_count": 0 if points is None else len(points)}
    lk = {
        "winSize": (21, 21),
        "maxLevel": 3,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, points, None, **lk
    )
    if forward is None or forward_status is None:
        return {"ok": False, "reason": "forward_flow_failed", "point_count": len(points)}
    backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, forward, None, **lk
    )
    if backward is None or backward_status is None:
        return {"ok": False, "reason": "backward_flow_failed", "point_count": len(points)}
    source = points.reshape(-1, 2)
    target = forward.reshape(-1, 2)
    returned = backward.reshape(-1, 2)
    fb = np.linalg.norm(source - returned, axis=1)
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(returned).all(axis=1)
        & (fb <= config.fb_error)
    )
    source, target, fb = source[valid], target[valid], fb[valid]
    if len(source) < config.min_points:
        return {
            "ok": False,
            "reason": "insufficient_fb_points",
            "point_count": int(len(points)),
            "valid_point_count": int(len(source)),
        }

    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=1000,
        confidence=0.99,
        refineIters=10,
    )
    fallback = False
    if matrix is None or inliers is None:
        displacement = np.median(target - source, axis=0)
        matrix = np.asarray(
            [[1.0, 0.0, displacement[0]], [0.0, 1.0, displacement[1]]],
            dtype=np.float32,
        )
        inlier_ratio = 0.5
        fallback = True
    else:
        inlier_ratio = float(inliers.reshape(-1).mean())
    scale = sqrt(float(matrix[0, 0]) ** 2 + float(matrix[0, 1]) ** 2)
    if not 0.75 <= scale <= 1.35:
        displacement = np.median(target - source, axis=0)
        matrix = np.asarray(
            [[1.0, 0.0, displacement[0]], [0.0, 1.0, displacement[1]]],
            dtype=np.float32,
        )
        scale = 1.0
        fallback = True
    if inlier_ratio < config.min_inlier_ratio and not fallback:
        return {
            "ok": False,
            "reason": "low_inlier_ratio",
            "valid_point_count": int(len(source)),
            "inlier_ratio": inlier_ratio,
        }

    x1, y1, x2, y2 = bbox
    corners = np.asarray(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=np.float32,
    )
    transformed = corners @ matrix.T
    projected = _clip_bbox(
        (
            float(transformed[:, 0].min()),
            float(transformed[:, 1].min()),
            float(transformed[:, 0].max()),
            float(transformed[:, 1].max()),
        ),
        width,
        height,
    )
    if projected is None:
        return {"ok": False, "reason": "invalid_projected_bbox"}
    median_fb = float(np.median(fb))
    point_factor = min(1.0, len(source) / max(float(config.min_points * 2), 1.0))
    fb_factor = exp(-median_fb / max(config.fb_error, 1e-6))
    confidence = max(0.0, min(1.0, point_factor * inlier_ratio * fb_factor))
    return {
        "ok": confidence >= config.min_confidence,
        "reason": "ok" if confidence >= config.min_confidence else "low_confidence",
        "bbox": projected,
        "confidence": confidence,
        "point_count": int(len(points)),
        "valid_point_count": int(len(source)),
        "inlier_ratio": inlier_ratio,
        "median_fb_error": median_fb,
        "scale": scale,
        "translation_fallback": fallback,
    }


def _propagate(
    cache: _GrayFrameCache,
    start: int,
    end: int,
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    config: FlowAssistConfig,
) -> dict[int, dict[str, Any]]:
    direction = 1 if end > start else -1
    current_bbox = bbox
    result: dict[int, dict[str, Any]] = {}
    frame_index = start
    while frame_index != end:
        next_index = frame_index + direction
        previous_gray = cache.get(frame_index)
        current_gray = cache.get(next_index)
        if previous_gray is None or current_gray is None:
            result[next_index] = {"ok": False, "reason": "frame_decode_failed"}
            break
        step = estimate_lk_bbox_step(
            previous_gray,
            current_gray,
            current_bbox,
            width=width,
            height=height,
            config=config,
        )
        result[next_index] = step
        if not step.get("ok"):
            break
        current_bbox = tuple(step["bbox"])
        frame_index = next_index
    return result


def apply_optical_flow_assist(
    accepted_by_frame: dict[int, list[TrackObservation]],
    *,
    get_frame: Callable[[int], Any],
    width: int,
    height: int,
    config: FlowAssistConfig,
) -> tuple[dict[int, list[TrackObservation]], dict[str, Any]]:
    if config.max_gap < 0:
        raise ValueError("flow max_gap must be >= 0")
    if config.bridge_policy not in {"off", "short_bridge"}:
        raise ValueError("flow bridge_policy must be off or short_bridge")
    grouped: dict[int, list[TrackObservation]] = defaultdict(list)
    for rows in accepted_by_frame.values():
        for row in rows:
            grouped[row.object_id].append(row)
    mutable = {
        frame: {row.object_id: row for row in rows}
        for frame, rows in accepted_by_frame.items()
    }
    cache = _GrayFrameCache(get_frame, max(config.cache_frames, 2 * config.max_gap + 4))
    audit: dict[str, Any] = {
        "enabled": True,
        "backend": "opencv_pyramidal_lk_bidirectional",
        "bridge_policy": config.bridge_policy,
        "pair_count": 0,
        "agreed_pair_count": 0,
        "conflict_pair_count": 0,
        "indeterminate_pair_count": 0,
        "candidate_gap_count": 0,
        "bridged_frame_count": 0,
        "rejected_gap_count": 0,
        "bridge_slot_collision_count": 0,
        "failure_reasons": defaultdict(int),
    }

    pair_tasks: list[tuple[TrackObservation, TrackObservation]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: row.frame_index)
        for left, right in zip(rows, rows[1:]):
            delta = right.frame_index - left.frame_index
            if delta < 1 or delta > config.max_gap + 1:
                continue
            pair_tasks.append((left, right))

    # Frame-major ordering lets both TouchAnything tracks share decoded gray
    # frames and keeps MP4 access sequential instead of seeking once per track.
    pair_tasks.sort(
        key=lambda pair: (
            pair[0].frame_index,
            pair[1].frame_index,
            pair[0].object_id,
        )
    )
    with progress_bar(
        total=len(pair_tasks) or None,
        unit="pair",
        desc="optical-flow-assist",
    ) as progress:
        for left, right in pair_tasks:
            track_id = left.object_id
            delta = right.frame_index - left.frame_index
            audit["pair_count"] += 1
            forward = _propagate(
                cache,
                left.frame_index,
                right.frame_index,
                left.bbox,
                width=width,
                height=height,
                config=config,
            )
            backward = _propagate(
                cache,
                right.frame_index,
                left.frame_index,
                right.bbox,
                width=width,
                height=height,
                config=config,
            )
            forward_end = forward.get(right.frame_index, {})
            backward_end = backward.get(left.frame_index, {})
            if forward_end.get("ok") and backward_end.get("ok"):
                endpoint_iou = min(
                    bbox_iou(tuple(forward_end["bbox"]), right.bbox),
                    bbox_iou(tuple(backward_end["bbox"]), left.bbox),
                )
                endpoint_confidence = min(
                    float(forward_end["confidence"]),
                    float(backward_end["confidence"]),
                )
                if endpoint_iou >= config.sam_iou_accept:
                    audit["agreed_pair_count"] += 1
                    source = "sam3_flow_agreed"
                elif endpoint_iou <= config.conflict_iou:
                    audit["conflict_pair_count"] += 1
                    source = "semantic_motion_conflict"
                else:
                    audit["indeterminate_pair_count"] += 1
                    source = "sam3_native"
                mutable[right.frame_index][track_id] = replace(
                    mutable[right.frame_index][track_id],
                    bbox_source=source,
                    flow_confidence=endpoint_confidence,
                    flow_bbox_iou=endpoint_iou,
                    flow_anchor_frames=(left.frame_index, right.frame_index),
                )
            else:
                audit["indeterminate_pair_count"] += 1
                for endpoint in (forward_end, backward_end):
                    if not endpoint.get("ok"):
                        audit["failure_reasons"][endpoint.get("reason", "unknown")] += 1

            if delta <= 1:
                progress.update(1)
                continue
            audit["candidate_gap_count"] += 1
            inserted = 0
            if config.bridge_policy == "short_bridge":
                for frame_index in range(left.frame_index + 1, right.frame_index):
                    lhs = forward.get(frame_index, {})
                    rhs = backward.get(frame_index, {})
                    if not lhs.get("ok") or not rhs.get("ok"):
                        continue
                    agreement_iou = bbox_iou(tuple(lhs["bbox"]), tuple(rhs["bbox"]))
                    confidence = min(float(lhs["confidence"]), float(rhs["confidence"]))
                    if agreement_iou < config.sam_iou_accept or confidence < config.min_confidence:
                        continue
                    existing_tracks = mutable.get(frame_index, {})
                    if (
                        config.max_tracks_per_frame > 0
                        and track_id not in existing_tracks
                        and len(existing_tracks) >= config.max_tracks_per_frame
                    ):
                        audit["bridge_slot_collision_count"] += 1
                        continue
                    lhs_bbox, rhs_bbox = lhs["bbox"], rhs["bbox"]
                    bbox = tuple(
                        int(round((float(lhs_bbox[index]) + float(rhs_bbox[index])) * 0.5))
                        for index in range(4)
                    )
                    bbox = _clip_bbox(bbox, width, height)
                    if bbox is None:
                        continue
                    left_score = 1.0 if left.prompt_score is None else float(left.prompt_score)
                    right_score = 1.0 if right.prompt_score is None else float(right.prompt_score)
                    distance = min(frame_index - left.frame_index, right.frame_index - frame_index)
                    prompt_score = min(left_score, right_score) * config.score_decay**distance
                    glove_votes = tuple(
                        sorted(set(left.glove_verifier_prompts) & set(right.glove_verifier_prompts))
                    )
                    bare_votes = tuple(
                        sorted(set(left.bare_verifier_prompts) & set(right.bare_verifier_prompts))
                    )
                    area_ratio = (
                        (frame_index - left.frame_index) / max(1, right.frame_index - left.frame_index)
                    )
                    mask_area = int(round(left.mask_area + area_ratio * (right.mask_area - left.mask_area)))
                    mutable.setdefault(frame_index, {})[track_id] = TrackObservation(
                        frame_index=frame_index,
                        object_id=track_id,
                        bbox=bbox,
                        mask_area=max(1, mask_area),
                        prompt_score=prompt_score,
                        mask_centroid=((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5),
                        glove_verifier_prompts=glove_votes,
                        bare_verifier_prompts=bare_votes,
                        bbox_source="flow_short_bridge",
                        flow_confidence=confidence,
                        flow_bbox_iou=agreement_iou,
                        flow_anchor_frames=(left.frame_index, right.frame_index),
                    )
                    inserted += 1
            if inserted:
                audit["bridged_frame_count"] += inserted
            else:
                audit["rejected_gap_count"] += 1
            progress.update(1)

    audit["failure_reasons"] = dict(sorted(audit["failure_reasons"].items()))
    result = {
        frame: sorted(rows.values(), key=lambda row: row.object_id)
        for frame, rows in mutable.items()
    }
    return result, audit
