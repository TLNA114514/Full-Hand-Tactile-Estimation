"""Prompt-aware selection and retrospective validation for SAM video tracks.

This module intentionally has no SAM, Torch, OpenCV, or NumPy dependency.  It
only works with the compact metadata emitted by ``track_video.py`` so its
selection policy can be tested without a GPU or a model checkout.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from math import hypot, isfinite, sqrt
from statistics import median
from typing import Iterable


BARE_REJECTION_POLICIES = ("hard", "bare_only", "off")


@dataclass(frozen=True)
class TrackObservation:
    """One prompt-conditioned SAM object observed in one video frame."""

    frame_index: int
    object_id: int
    bbox: tuple[int, int, int, int]
    mask_area: int
    prompt_score: float | None
    mask_centroid: tuple[float, float] | None = None
    # Prompt-vote fields are filled by independent text-prompt sessions.  They
    # are metadata only: neither hand identity nor these signals ever enter
    # the tactile model.
    glove_verifier_prompts: tuple[str, ...] = ()
    bare_verifier_prompts: tuple[str, ...] = ()
    bbox_source: str = "sam3_native"
    flow_confidence: float | None = None
    flow_bbox_iou: float | None = None
    flow_anchor_frames: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        row = asdict(self)
        row["bbox"] = list(self.bbox)
        if self.mask_centroid is not None:
            row["mask_centroid"] = list(self.mask_centroid)
        row["glove_verifier_prompts"] = list(self.glove_verifier_prompts)
        row["bare_verifier_prompts"] = list(self.bare_verifier_prompts)
        row["flow_anchor_frames"] = list(self.flow_anchor_frames)
        return row


def _finite_score(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if isfinite(value) else None


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile of no values")
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * fraction))))
    return sorted(values)[index]


def bbox_iou(lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]) -> float:
    left = max(lhs[0], rhs[0])
    top = max(lhs[1], rhs[1])
    right = min(lhs[2], rhs[2])
    bottom = min(lhs[3], rhs[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    lhs_area = max(0, lhs[2] - lhs[0]) * max(0, lhs[3] - lhs[1])
    rhs_area = max(0, rhs[2] - rhs[0]) * max(0, rhs[3] - rhs[1])
    union = lhs_area + rhs_area - intersection
    return float(intersection / union) if union else 0.0


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)


def bbox_diagonal(bbox: tuple[int, int, int, int]) -> float:
    return hypot(max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1]))


def _observation_center(observation: TrackObservation) -> tuple[float, float]:
    """Use a mask centroid when available, otherwise the bbox centre."""

    return observation.mask_centroid or bbox_center(observation.bbox)


def _spatially_matches(
    primary: TrackObservation,
    verifier: TrackObservation,
    *,
    match_iou_floor: float,
    max_centroid_distance_ratio: float,
) -> bool:
    """Conservatively associate two prompt sessions in the same frame.

    Object IDs are local to a SAM video session.  Bbox IoU alone is too loose
    for nearby hands, especially if a mask includes wrist or forearm pixels.
    Requiring a compatible mask/bbox centre prevents a nearby bare hand from
    becoming counter-evidence for a distinct gloved hand.
    """

    if bbox_iou(primary.bbox, verifier.bbox) < match_iou_floor:
        return False
    primary_center = _observation_center(primary)
    verifier_center = _observation_center(verifier)
    center_distance = hypot(
        primary_center[0] - verifier_center[0],
        primary_center[1] - verifier_center[1],
    )
    reference_size = max(
        1.0,
        min(bbox_diagonal(primary.bbox), bbox_diagonal(verifier.bbox)),
    )
    return center_distance / reference_size <= max_centroid_distance_ratio


def _centroid_distance_ratio(
    primary: TrackObservation,
    verifier: TrackObservation,
) -> float:
    """Return the same normalized centre distance used by spatial matching."""

    primary_center = _observation_center(primary)
    verifier_center = _observation_center(verifier)
    center_distance = hypot(
        primary_center[0] - verifier_center[0],
        primary_center[1] - verifier_center[1],
    )
    reference_size = max(
        1.0,
        min(bbox_diagonal(primary.bbox), bbox_diagonal(verifier.bbox)),
    )
    return float(center_distance / reference_size)


def _bare_rejection_applies(
    observation: TrackObservation,
    bare_rejection_policy: str,
) -> bool:
    """Decide whether a bare-prompt vote is strong enough to reject a row.

    A text prompt can call a gloved hand a ``bare hand`` even when no useful
    skin is visible.  ``bare_only`` therefore treats conflicting independent
    glove evidence as ambiguity, not proof that the primary candidate is bare.
    """

    if bare_rejection_policy not in BARE_REJECTION_POLICIES:
        raise ValueError(
            "bare_rejection_policy must be one of "
            f"{BARE_REJECTION_POLICIES}, got {bare_rejection_policy!r}"
        )
    if not observation.bare_verifier_prompts:
        return False
    if bare_rejection_policy == "hard":
        return True
    if bare_rejection_policy == "bare_only":
        return not bool(observation.glove_verifier_prompts)
    return False


def attach_semantic_prompt_votes(
    gloved_observations: Iterable[TrackObservation],
    glove_verifier_observations: dict[str, Iterable[TrackObservation]],
    bare_verifier_observations: dict[str, Iterable[TrackObservation]],
    *,
    match_iou_floor: float,
    min_verifier_score: float,
    max_centroid_distance_ratio: float,
    include_match_details: bool = False,
) -> tuple[list[TrackObservation], dict]:
    """Attach independent glove/bare prompt votes to primary candidates.

    SAM object IDs are session-local.  A vote therefore requires a high-IoU
    mask box in the same frame from an *independent* text-prompt run.  We never
    compare scores across different text prompts: those scores share a detector
    but are not calibrated as a glove-vs-bare probability.  A primary candidate
    needs positive glove evidence to become a tactile query; any matched bare
    evidence is a hard per-frame rejection in filter mode.
    """

    if not 0 <= match_iou_floor <= 1:
        raise ValueError("match_iou_floor must lie in [0, 1]")
    if not 0 <= min_verifier_score <= 1:
        raise ValueError("min_verifier_score must lie in [0, 1]")
    if max_centroid_distance_ratio <= 0:
        raise ValueError("max_centroid_distance_ratio must be positive")
    gloved_rows = list(gloved_observations)

    def _by_frame(
        prompt_rows: dict[str, Iterable[TrackObservation]],
    ) -> dict[str, dict[int, list[TrackObservation]]]:
        result: dict[str, dict[int, list[TrackObservation]]] = {}
        for prompt, rows in prompt_rows.items():
            grouped: dict[int, list[TrackObservation]] = defaultdict(list)
            for candidate in rows:
                if (_finite_score(candidate.prompt_score) or -1.0) >= min_verifier_score:
                    grouped[candidate.frame_index].append(candidate)
            result[prompt] = grouped
        return result

    glove_by_prompt = _by_frame(glove_verifier_observations)
    bare_by_prompt = _by_frame(bare_verifier_observations)

    def _match_details(
        row: TrackObservation,
        candidates: Iterable[TrackObservation],
    ) -> list[dict]:
        details: list[dict] = []
        for candidate in candidates:
            iou = bbox_iou(row.bbox, candidate.bbox)
            centroid_ratio = _centroid_distance_ratio(row, candidate)
            if iou < match_iou_floor or centroid_ratio > max_centroid_distance_ratio:
                continue
            details.append(
                {
                    "verifier_object_id": int(candidate.object_id),
                    "verifier_bbox": list(candidate.bbox),
                    "verifier_mask_centroid": (
                        list(candidate.mask_centroid)
                        if candidate.mask_centroid is not None
                        else None
                    ),
                    "verifier_prompt_score": _finite_score(candidate.prompt_score),
                    "verifier_mask_area": int(candidate.mask_area),
                    "bbox_iou": float(iou),
                    "centroid_distance_ratio": float(centroid_ratio),
                }
            )
        return details

    enriched: list[TrackObservation] = []
    glove_match_counts = {prompt: 0 for prompt in glove_by_prompt}
    bare_match_counts = {prompt: 0 for prompt in bare_by_prompt}
    observation_match_details: list[dict] = []
    for row in gloved_rows:
        glove_details = {
            prompt: _match_details(row, by_frame.get(row.frame_index, ()))
            for prompt, by_frame in glove_by_prompt.items()
        }
        bare_details = {
            prompt: _match_details(row, by_frame.get(row.frame_index, ()))
            for prompt, by_frame in bare_by_prompt.items()
        }
        glove_votes = tuple(prompt for prompt, details in glove_details.items() if details)
        bare_votes = tuple(prompt for prompt, details in bare_details.items() if details)
        for prompt in glove_votes:
            glove_match_counts[prompt] += 1
        for prompt in bare_votes:
            bare_match_counts[prompt] += 1
        if include_match_details:
            if glove_votes and bare_votes:
                semantic_state = "both"
            elif glove_votes:
                semantic_state = "glove_only"
            elif bare_votes:
                semantic_state = "bare_only"
            else:
                semantic_state = "neither"
            observation_match_details.append(
                {
                    "frame_index": int(row.frame_index),
                    "primary_object_id": int(row.object_id),
                    "primary_bbox": list(row.bbox),
                    "primary_mask_centroid": (
                        list(row.mask_centroid) if row.mask_centroid is not None else None
                    ),
                    "primary_prompt_score": _finite_score(row.prompt_score),
                    "primary_mask_area": int(row.mask_area),
                    "semantic_state": semantic_state,
                    "glove_matches": {
                        prompt: details for prompt, details in glove_details.items() if details
                    },
                    "bare_matches": {
                        prompt: details for prompt, details in bare_details.items() if details
                    },
                }
            )
        enriched.append(
            replace(
                row,
                glove_verifier_prompts=glove_votes,
                bare_verifier_prompts=bare_votes,
            )
        )
    audit = {
        "match_iou_floor": match_iou_floor,
        "max_centroid_distance_ratio": max_centroid_distance_ratio,
        "min_verifier_score": min_verifier_score,
        "matching_rule": "same_frame_bbox_iou_and_mask_centroid_agreement",
        "gloved_observation_count": len(gloved_rows),
        "glove_prompt_match_counts": glove_match_counts,
        "bare_prompt_match_counts": bare_match_counts,
        "positive_vote_observation_count": sum(
            bool(row.glove_verifier_prompts) for row in enriched
        ),
        "negative_vote_observation_count": sum(
            bool(row.bare_verifier_prompts) for row in enriched
        ),
    }
    if include_match_details:
        audit["observation_match_details"] = observation_match_details
    return enriched, audit


def _track_summary(
    object_id: int,
    observations: list[TrackObservation],
    total_frames: int,
    min_prompt_score: float,
    min_track_frames: int,
    min_glove_verifier_fraction: float,
    max_bare_evidence_fraction: float,
    bare_rejection_policy: str,
) -> dict:
    observations = sorted(observations, key=lambda row: row.frame_index)
    scores = [_finite_score(row.prompt_score) for row in observations]
    finite_scores = [score for score in scores if score is not None]
    score_available = len(finite_scores) == len(observations)
    high_score_fraction = (
        sum(score >= min_prompt_score for score in finite_scores) / len(finite_scores)
        if finite_scores
        else 0.0
    )
    gaps = [
        later.frame_index - earlier.frame_index
        for earlier, later in zip(observations, observations[1:])
    ]
    continuity = (
        sum(gap == 1 for gap in gaps) / len(gaps) if gaps else float(bool(observations))
    )
    score_median = median(finite_scores) if finite_scores else None
    score_p10 = _quantile(finite_scores, 0.10) if finite_scores else None
    glove_verifier_fraction = sum(
        bool(row.glove_verifier_prompts) for row in observations
    ) / max(1, len(observations))
    glove_verifier_vote_mean = sum(
        len(row.glove_verifier_prompts) for row in observations
    ) / max(1, len(observations))
    bare_evidence_fraction = sum(
        bool(row.bare_verifier_prompts) for row in observations
    ) / max(1, len(observations))
    bare_only_evidence_fraction = sum(
        bool(row.bare_verifier_prompts) and not bool(row.glove_verifier_prompts)
        for row in observations
    ) / max(1, len(observations))
    ambiguous_glove_bare_fraction = sum(
        bool(row.bare_verifier_prompts) and bool(row.glove_verifier_prompts)
        for row in observations
    ) / max(1, len(observations))
    bare_verifier_vote_mean = sum(
        len(row.bare_verifier_prompts) for row in observations
    ) / max(1, len(observations))
    effective_bare_evidence_fraction = sum(
        _bare_rejection_applies(row, bare_rejection_policy) for row in observations
    ) / max(1, len(observations))
    glove_verifier_conformant = (
        glove_verifier_fraction >= min_glove_verifier_fraction
    )
    bare_evidence_conformant = (
        effective_bare_evidence_fraction <= max_bare_evidence_fraction
    )
    contrast_conformant = glove_verifier_conformant and bare_evidence_conformant
    conformant = bool(
        score_available
        and len(observations) >= min_track_frames
        and score_median is not None
        and score_median >= min_prompt_score
        and high_score_fraction >= 0.5
        and contrast_conformant
    )
    # Rank independently matched semantics first, then broad temporal coverage
    # and native prompt confidence. Tiny continuity differences are only a
    # late tie-breaker: a stable held object must not beat a longer hand track
    # merely because it has one fewer gap. Area remains deliberately excluded.
    rank_key = (
        float(contrast_conformant),
        glove_verifier_fraction,
        glove_verifier_vote_mean,
        -effective_bare_evidence_fraction,
        -bare_evidence_fraction,
        -bare_verifier_vote_mean,
        len(observations) / max(1, total_frames),
        -1.0 if score_median is None else score_median,
        high_score_fraction,
        continuity,
        -object_id,
    )
    return {
        "object_id": object_id,
        "frame_count": len(observations),
        "coverage": len(observations) / max(1, total_frames),
        "frame_first": observations[0].frame_index if observations else None,
        "frame_last": observations[-1].frame_index if observations else None,
        "median_mask_area": median(row.mask_area for row in observations) if observations else 0.0,
        "prompt_score_available": score_available,
        "prompt_score_median": score_median,
        "prompt_score_p10": score_p10,
        "prompt_score_high_fraction": high_score_fraction,
        "glove_verifier_fraction": glove_verifier_fraction,
        "glove_verifier_vote_mean": glove_verifier_vote_mean,
        "bare_evidence_fraction": bare_evidence_fraction,
        "bare_only_evidence_fraction": bare_only_evidence_fraction,
        "ambiguous_glove_bare_fraction": ambiguous_glove_bare_fraction,
        "effective_bare_evidence_fraction": effective_bare_evidence_fraction,
        "bare_verifier_vote_mean": bare_verifier_vote_mean,
        "bare_rejection_policy": bare_rejection_policy,
        "glove_verifier_observation_count": sum(
            bool(row.glove_verifier_prompts) for row in observations
        ),
        "bare_verifier_observation_count": sum(
            bool(row.bare_verifier_prompts) for row in observations
        ),
        "bare_contrast_conformant": contrast_conformant,
        "glove_verifier_conformant": glove_verifier_conformant,
        "bare_evidence_conformant": bare_evidence_conformant,
        "adjacent_frame_continuity": continuity,
        "max_frame_gap": max(gaps) if gaps else 0,
        "prompt_conformant": conformant,
        "rank_key": list(rank_key),
    }


def select_prompt_tracks(
    observations: Iterable[TrackObservation],
    *,
    total_frames: int,
    max_tracks: int,
    min_prompt_score: float,
    min_track_frames: int,
    require_prompt_score: bool,
    min_glove_verifier_fraction: float = 0.0,
    max_bare_evidence_fraction: float = 1.0,
    bare_rejection_policy: str = "hard",
    duplicate_iou_floor: float = 0.80,
    duplicate_overlap_fraction: float = 0.60,
    duplicate_match_fraction: float = 0.80,
    duplicate_centroid_ratio: float = 0.18,
    duplicate_area_ratio: float = 1.50,
    duplicate_min_frames: int = 2,
    allow_nonoverlapping_fragments: bool = False,
) -> tuple[list[int], dict]:
    """Select stable SAM object IDs without any per-frame area replacement."""

    if max_tracks < 1:
        raise ValueError("max_tracks must be positive")
    if not 0 <= min_glove_verifier_fraction <= 1:
        raise ValueError("min_glove_verifier_fraction must lie in [0, 1]")
    if not 0 <= max_bare_evidence_fraction <= 1:
        raise ValueError("max_bare_evidence_fraction must lie in [0, 1]")
    if bare_rejection_policy not in BARE_REJECTION_POLICIES:
        raise ValueError(
            "bare_rejection_policy must be one of "
            f"{BARE_REJECTION_POLICIES}, got {bare_rejection_policy!r}"
        )
    if not 0 <= duplicate_iou_floor <= 1:
        raise ValueError("duplicate_iou_floor must lie in [0, 1]")
    if not 0 <= duplicate_overlap_fraction <= 1:
        raise ValueError("duplicate_overlap_fraction must lie in [0, 1]")
    if not 0 <= duplicate_match_fraction <= 1:
        raise ValueError("duplicate_match_fraction must lie in [0, 1]")
    if duplicate_centroid_ratio <= 0:
        raise ValueError("duplicate_centroid_ratio must be positive")
    if duplicate_area_ratio <= 1:
        raise ValueError("duplicate_area_ratio must be > 1")
    if duplicate_min_frames < 1:
        raise ValueError("duplicate_min_frames must be positive")
    grouped: dict[int, list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.object_id].append(observation)

    all_observations = [row for rows in grouped.values() for row in rows]
    missing_score_count = sum(_finite_score(row.prompt_score) is None for row in all_observations)
    if require_prompt_score and all_observations and missing_score_count:
        raise RuntimeError(
            "SAM did not provide a finite per-object prompt confidence for every "
            "candidate. Refusing to promote an unverified track to a tactile query. "
            "Inspect summary.json response_output_keys, or use "
            "--allow-missing-prompt-score only for diagnostics."
        )

    summaries = [
        _track_summary(
            object_id,
            rows,
            total_frames=total_frames,
            min_prompt_score=min_prompt_score,
            min_track_frames=min_track_frames,
            min_glove_verifier_fraction=min_glove_verifier_fraction,
            max_bare_evidence_fraction=max_bare_evidence_fraction,
            bare_rejection_policy=bare_rejection_policy,
        )
        for object_id, rows in grouped.items()
    ]
    summaries.sort(key=lambda row: tuple(row["rank_key"]), reverse=True)
    eligible = [row for row in summaries if row["prompt_conformant"]]

    parent = {int(row["object_id"]): int(row["object_id"]) for row in eligible}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(lhs: int, rhs: int) -> None:
        lhs_root, rhs_root = find(lhs), find(rhs)
        if lhs_root != rhs_root:
            parent[rhs_root] = lhs_root

    duplicate_pairs = []
    for index, lhs_summary in enumerate(eligible):
        lhs_id = int(lhs_summary["object_id"])
        lhs_by_frame = {row.frame_index: row for row in grouped[lhs_id]}
        for rhs_summary in eligible[index + 1 :]:
            rhs_id = int(rhs_summary["object_id"])
            rhs_by_frame = {row.frame_index: row for row in grouped[rhs_id]}
            common_frames = sorted(set(lhs_by_frame) & set(rhs_by_frame))
            overlap_fraction = len(common_frames) / max(
                1, min(len(lhs_by_frame), len(rhs_by_frame))
            )
            matched = 0
            for frame_index in common_frames:
                lhs = lhs_by_frame[frame_index]
                rhs = rhs_by_frame[frame_index]
                areas = max(lhs.mask_area, rhs.mask_area) / max(
                    1.0, min(lhs.mask_area, rhs.mask_area)
                )
                if (
                    bbox_iou(lhs.bbox, rhs.bbox) >= duplicate_iou_floor
                    and _centroid_distance_ratio(lhs, rhs) <= duplicate_centroid_ratio
                    and areas <= duplicate_area_ratio
                ):
                    matched += 1
            match_fraction = matched / len(common_frames) if common_frames else 0.0
            duplicate = bool(
                len(common_frames) >= duplicate_min_frames
                and overlap_fraction >= duplicate_overlap_fraction
                and match_fraction >= duplicate_match_fraction
            )
            if duplicate:
                union(lhs_id, rhs_id)
            duplicate_pairs.append(
                {
                    "lhs_object_id": lhs_id,
                    "rhs_object_id": rhs_id,
                    "common_frame_count": len(common_frames),
                    "overlap_fraction": overlap_fraction,
                    "matched_frame_count": matched,
                    "match_fraction": match_fraction,
                    "duplicate": duplicate,
                }
            )

    clusters: dict[int, list[int]] = defaultdict(list)
    for row in eligible:
        object_id = int(row["object_id"])
        clusters[find(object_id)].append(object_id)
    summary_rank = {int(row["object_id"]): index for index, row in enumerate(summaries)}
    representative_by_id: dict[int, int] = {}
    cluster_rows = []
    for cluster_index, members in enumerate(clusters.values()):
        members.sort(key=lambda object_id: summary_rank[object_id])
        representative = members[0]
        for object_id in members:
            representative_by_id[object_id] = representative
        cluster_rows.append(
            {
                "cluster_id": cluster_index,
                "representative_object_id": representative,
                "member_object_ids": members,
            }
        )
    representatives = [
        row
        for row in eligible
        if representative_by_id[int(row["object_id"])] == int(row["object_id"])
    ]
    selected: list[int] = []
    occupied_frames: dict[int, int] = defaultdict(int)
    reentry_selected: set[int] = set()
    for row in representatives:
        object_id = int(row["object_id"])
        frame_indices = {item.frame_index for item in grouped[object_id]}
        is_primary_slot = len(selected) < max_tracks
        fits_open_slots = bool(
            allow_nonoverlapping_fragments
            and frame_indices
            and all(occupied_frames[index] < max_tracks for index in frame_indices)
        )
        if not is_primary_slot and not fits_open_slots:
            continue
        selected.append(object_id)
        if not is_primary_slot:
            reentry_selected.add(object_id)
        for frame_index in frame_indices:
            occupied_frames[frame_index] += 1
    selected_set = set(selected)
    cluster_by_id = {
        object_id: cluster
        for cluster in cluster_rows
        for object_id in cluster["member_object_ids"]
    }
    for summary in summaries:
        object_id = int(summary["object_id"])
        representative = representative_by_id.get(object_id)
        summary["duplicate_cluster_id"] = (
            cluster_by_id[object_id]["cluster_id"] if object_id in cluster_by_id else None
        )
        summary["duplicate_of"] = (
            representative if representative is not None and representative != object_id else None
        )
        summary["selected"] = object_id in selected_set
        if summary["selected"]:
            summary["selection_reason"] = (
                "prompt_conformant_nonoverlap_reentry_fragment"
                if object_id in reentry_selected
                else "prompt_conformant_global_track"
            )
        elif summary["duplicate_of"] is not None:
            summary["selection_reason"] = "spatiotemporal_duplicate_track"
        elif not summary["prompt_score_available"]:
            summary["selection_reason"] = "missing_prompt_score"
        elif summary["frame_count"] < min_track_frames:
            summary["selection_reason"] = "insufficient_track_frames"
        elif not summary["bare_evidence_conformant"]:
            summary["selection_reason"] = "bare_prompt_matched_too_often"
        elif not summary["glove_verifier_conformant"]:
            summary["selection_reason"] = "missing_independent_glove_verification"
        elif not summary["prompt_conformant"]:
            summary["selection_reason"] = "below_prompt_conformity_threshold"
        else:
            summary["selection_reason"] = "lower_prompt_track_rank"
    return selected, {
        "max_tracks": max_tracks,
        "allow_nonoverlapping_fragments": bool(allow_nonoverlapping_fragments),
        "reentry_selected_track_ids": sorted(reentry_selected),
        "min_prompt_score": min_prompt_score,
        "min_track_frames": min_track_frames,
        "require_prompt_score": require_prompt_score,
        "min_glove_verifier_fraction": min_glove_verifier_fraction,
        "max_bare_evidence_fraction": max_bare_evidence_fraction,
        "bare_rejection_policy": bare_rejection_policy,
        "candidate_observation_count": len(all_observations),
        "missing_prompt_score_count": missing_score_count,
        "selected_track_ids": selected,
        "duplicate_track_aliases": {
            str(object_id): representative
            for object_id, representative in sorted(representative_by_id.items())
            if object_id != representative
            and representative in selected_set
        },
        "duplicate_track_clusters": cluster_rows,
        "duplicate_track_pairs": duplicate_pairs,
        "duplicate_policy": {
            "iou_floor": duplicate_iou_floor,
            "overlap_fraction": duplicate_overlap_fraction,
            "match_fraction": duplicate_match_fraction,
            "centroid_ratio": duplicate_centroid_ratio,
            "area_ratio": duplicate_area_ratio,
            "min_frames": duplicate_min_frames,
        },
        "track_summaries": summaries,
    }


def consolidate_duplicate_track_observations(
    observations: Iterable[TrackObservation],
    duplicate_track_aliases: dict[str | int, int],
) -> list[TrackObservation]:
    """Merge duplicate SAM IDs into their selected physical-query representative.

    The best same-frame semantic/prompt observation is retained. Mask area is
    intentionally absent from the ranking so a larger duplicate cannot win by
    absorbing more of the object or forearm.
    """

    aliases = {int(key): int(value) for key, value in duplicate_track_aliases.items()}
    by_frame_id: dict[tuple[int, int], TrackObservation] = {}

    def rank(row: TrackObservation) -> tuple[float, float, float, int]:
        score = _finite_score(row.prompt_score)
        return (
            float(bool(row.glove_verifier_prompts)),
            -float(bool(row.bare_verifier_prompts)),
            score if score is not None else -1.0,
            -int(row.object_id),
        )

    for row in observations:
        representative = aliases.get(row.object_id, row.object_id)
        merged = replace(row, object_id=representative)
        key = (merged.frame_index, representative)
        previous = by_frame_id.get(key)
        if previous is None or rank(merged) > rank(previous):
            by_frame_id[key] = merged
    return sorted(
        by_frame_id.values(), key=lambda row: (row.frame_index, row.object_id)
    )


def stitch_overlapping_chunk_tracks(
    observations: Iterable[TrackObservation],
    *,
    namespace_stride: int = 1_000_000,
    min_common_frames: int = 3,
    match_iou_floor: float = 0.30,
    match_fraction: float = 0.60,
    max_centroid_ratio: float = 0.50,
) -> tuple[list[TrackObservation], dict]:
    """Join adjacent bounded-session IDs using only their overlap frames."""

    rows = list(observations)
    by_chunk_id: dict[int, dict[int, list[TrackObservation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_chunk_id[row.object_id // namespace_stride][row.object_id].append(row)
    aliases: dict[int, int] = {}
    pair_audit = []

    def resolve(object_id: int) -> int:
        while object_id in aliases:
            object_id = aliases[object_id]
        return object_id

    chunks = sorted(by_chunk_id)
    for lhs_chunk, rhs_chunk in zip(chunks, chunks[1:]):
        if rhs_chunk != lhs_chunk + 1:
            continue
        candidates = []
        for lhs_id, lhs_rows in by_chunk_id[lhs_chunk].items():
            lhs_by_frame = {row.frame_index: row for row in lhs_rows}
            for rhs_id, rhs_rows in by_chunk_id[rhs_chunk].items():
                rhs_by_frame = {row.frame_index: row for row in rhs_rows}
                common = sorted(set(lhs_by_frame) & set(rhs_by_frame))
                matched_ious = []
                for frame_index in common:
                    lhs = lhs_by_frame[frame_index]
                    rhs = rhs_by_frame[frame_index]
                    iou = bbox_iou(lhs.bbox, rhs.bbox)
                    if (
                        iou >= match_iou_floor
                        and _centroid_distance_ratio(lhs, rhs) <= max_centroid_ratio
                    ):
                        matched_ious.append(iou)
                fraction = len(matched_ious) / len(common) if common else 0.0
                accepted = (
                    len(common) >= min_common_frames
                    and fraction >= match_fraction
                )
                mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
                pair_audit.append(
                    {
                        "lhs_object_id": lhs_id,
                        "rhs_object_id": rhs_id,
                        "common_frame_count": len(common),
                        "matched_frame_count": len(matched_ious),
                        "match_fraction": fraction,
                        "mean_matched_iou": mean_iou,
                        "accepted_candidate": accepted,
                    }
                )
                if accepted:
                    candidates.append((len(matched_ious), mean_iou, lhs_id, rhs_id))
        used_lhs: set[int] = set()
        used_rhs: set[int] = set()
        for _, _, lhs_id, rhs_id in sorted(candidates, reverse=True):
            if lhs_id in used_lhs or rhs_id in used_rhs:
                continue
            aliases[rhs_id] = resolve(lhs_id)
            used_lhs.add(lhs_id)
            used_rhs.add(rhs_id)

    stitched = [replace(row, object_id=resolve(row.object_id)) for row in rows]
    stitched.sort(key=lambda row: (row.frame_index, row.object_id))
    return stitched, {
        "namespace_stride": namespace_stride,
        "min_common_frames": min_common_frames,
        "match_iou_floor": match_iou_floor,
        "match_fraction": match_fraction,
        "max_centroid_ratio": max_centroid_ratio,
        "aliases": {str(key): value for key, value in sorted(aliases.items())},
        "pairs": pair_audit,
    }


def _return_jump_reason(
    previous: TrackObservation,
    current: TrackObservation,
    following: TrackObservation,
    *,
    max_frame_gap: int,
    center_residual_ratio: float,
    area_ratio: float,
    neighbor_iou_floor: float,
) -> str | None:
    if (
        current.frame_index - previous.frame_index > max_frame_gap
        or following.frame_index - current.frame_index > max_frame_gap
    ):
        return None
    # A jump and immediate return is distinguishable from ordinary fast motion:
    # the two neighbours agree, while the middle observation is far away.
    if bbox_iou(previous.bbox, following.bbox) < neighbor_iou_floor:
        return None
    before_center = bbox_center(previous.bbox)
    after_center = bbox_center(following.bbox)
    current_center = bbox_center(current.bbox)
    expected_center = ((before_center[0] + after_center[0]) * 0.5, (before_center[1] + after_center[1]) * 0.5)
    residual = hypot(
        current_center[0] - expected_center[0],
        current_center[1] - expected_center[1],
    )
    reference_size = max(1.0, bbox_diagonal(previous.bbox), bbox_diagonal(following.bbox))
    if residual / reference_size >= center_residual_ratio:
        return "return_center_jump"

    neighbour_area_ratio = max(previous.mask_area, following.mask_area) / max(
        1.0, min(previous.mask_area, following.mask_area)
    )
    expected_area = sqrt(max(1.0, previous.mask_area) * max(1.0, following.mask_area))
    current_area_ratio = max(current.mask_area, expected_area) / max(1.0, min(current.mask_area, expected_area))
    if neighbour_area_ratio <= 1.5 and current_area_ratio >= area_ratio:
        return "return_area_jump"
    return None


def _unverified_return_excursion_indices(
    rows: list[TrackObservation],
    *,
    max_segment_frames: int,
    max_frame_gap: int,
    center_residual_ratio: float,
    neighbor_iou_floor: float,
) -> set[int]:
    """Find short, semantically unverified excursions bracketed by one glove.

    This intentionally requires evidence on both sides. It cannot reject a
    legitimate newly appearing hand merely because it is far from a stale box.
    """

    rejected: set[int] = set()
    if max_segment_frames < 2:
        return rejected
    index = 1
    while index < len(rows) - 1:
        before = rows[index - 1]
        if not before.glove_verifier_prompts:
            index += 1
            continue
        accepted_end = None
        for end in range(index + 1, min(len(rows) - 1, index + max_segment_frames)):
            segment = rows[index : end + 1]
            after = rows[end + 1]
            chain = [before, *segment, after]
            if any(
                current.frame_index - previous.frame_index > max_frame_gap
                for previous, current in zip(chain, chain[1:])
            ):
                break
            if any(row.glove_verifier_prompts for row in segment):
                continue
            if not after.glove_verifier_prompts:
                continue
            if bbox_iou(before.bbox, after.bbox) < neighbor_iou_floor:
                continue
            before_center = bbox_center(before.bbox)
            after_center = bbox_center(after.bbox)
            reference_size = max(
                1.0, bbox_diagonal(before.bbox), bbox_diagonal(after.bbox)
            )
            far_count = 0
            for offset, row in enumerate(segment, start=1):
                alpha = offset / (len(segment) + 1)
                expected = (
                    before_center[0] + alpha * (after_center[0] - before_center[0]),
                    before_center[1] + alpha * (after_center[1] - before_center[1]),
                )
                current = bbox_center(row.bbox)
                residual = hypot(current[0] - expected[0], current[1] - expected[1])
                far_count += residual / reference_size >= center_residual_ratio
            if far_count == len(segment):
                accepted_end = end
                break
        if accepted_end is None:
            index += 1
            continue
        rejected.update(range(index, accepted_end + 1))
        index = accepted_end + 1
    return rejected


def filter_selected_tracks(
    observations: Iterable[TrackObservation],
    *,
    selected_track_ids: Iterable[int],
    min_prompt_score: float,
    require_prompt_score: bool,
    max_frame_gap: int,
    center_residual_ratio: float,
    area_ratio: float,
    neighbor_iou_floor: float,
    return_excursion_max_frames: int = 0,
    reject_unverified_return_excursions: bool = False,
    reject_bare_prompt_matches: bool = False,
    bare_rejection_policy: str = "hard",
) -> tuple[dict[int, list[TrackObservation]], dict[int, list[dict]], dict]:
    """Apply prompt and retrospective return-jump checks to locked IDs.

    A rejected frame is intentionally left empty.  The function never replaces
    it with a different SAM ID just because that other object looks larger.
    """

    if bare_rejection_policy not in BARE_REJECTION_POLICIES:
        raise ValueError(
            "bare_rejection_policy must be one of "
            f"{BARE_REJECTION_POLICIES}, got {bare_rejection_policy!r}"
        )
    selected = {int(track_id) for track_id in selected_track_ids}
    grouped: dict[int, list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        if observation.object_id in selected:
            grouped[observation.object_id].append(observation)

    accepted_by_frame: dict[int, list[TrackObservation]] = defaultdict(list)
    rejected_by_frame: dict[int, list[dict]] = defaultdict(list)
    rejection_counts: dict[str, int] = defaultdict(int)
    for track_id in sorted(selected):
        rows = sorted(grouped.get(track_id, []), key=lambda row: row.frame_index)
        initially_valid: list[TrackObservation] = []
        for row in rows:
            score = _finite_score(row.prompt_score)
            if score is None and require_prompt_score:
                reason = "missing_prompt_score"
            elif score is not None and score < min_prompt_score:
                reason = "prompt_score_below_threshold"
            elif (
                reject_bare_prompt_matches
                and _bare_rejection_applies(row, bare_rejection_policy)
            ):
                reason = (
                    "bare_prompt_matched"
                    if bare_rejection_policy == "hard"
                    else "bare_only_prompt_matched"
                )
            else:
                reason = None
            if reason is None:
                initially_valid.append(row)
            else:
                rejected_by_frame[row.frame_index].append(
                    {
                        "track_id": track_id,
                        "reason": reason,
                        "prompt_score": score,
                        "glove_verifier_prompts": list(row.glove_verifier_prompts),
                        "bare_verifier_prompts": list(row.bare_verifier_prompts),
                        "bare_rejection_policy": bare_rejection_policy,
                    }
                )
                rejection_counts[reason] += 1

        rejected_indices: set[int] = set()
        for index in range(1, len(initially_valid) - 1):
            reason = _return_jump_reason(
                initially_valid[index - 1],
                initially_valid[index],
                initially_valid[index + 1],
                max_frame_gap=max_frame_gap,
                center_residual_ratio=center_residual_ratio,
                area_ratio=area_ratio,
                neighbor_iou_floor=neighbor_iou_floor,
            )
            if reason is not None:
                rejected_indices.add(index)
                row = initially_valid[index]
                rejected_by_frame[row.frame_index].append(
                    {"track_id": track_id, "reason": reason, "prompt_score": row.prompt_score}
                )
                rejection_counts[reason] += 1
        if reject_unverified_return_excursions:
            excursion_indices = _unverified_return_excursion_indices(
                initially_valid,
                max_segment_frames=return_excursion_max_frames,
                max_frame_gap=max_frame_gap,
                center_residual_ratio=center_residual_ratio,
                neighbor_iou_floor=max(0.30, neighbor_iou_floor),
            )
            for index in sorted(excursion_indices - rejected_indices):
                rejected_indices.add(index)
                row = initially_valid[index]
                reason = "return_excursion_without_glove_evidence"
                rejected_by_frame[row.frame_index].append(
                    {"track_id": track_id, "reason": reason, "prompt_score": row.prompt_score}
                )
                rejection_counts[reason] += 1
        for index, row in enumerate(initially_valid):
            if index not in rejected_indices:
                accepted_by_frame[row.frame_index].append(row)

    for rows in accepted_by_frame.values():
        rows.sort(key=lambda row: row.object_id)
    return dict(accepted_by_frame), dict(rejected_by_frame), {
        "selected_track_count": len(selected),
        "accepted_observation_count": sum(len(rows) for rows in accepted_by_frame.values()),
        "rejected_observation_count": sum(rejection_counts.values()),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "max_frame_gap": max_frame_gap,
        "center_residual_ratio": center_residual_ratio,
        "area_ratio": area_ratio,
        "neighbor_iou_floor": neighbor_iou_floor,
        "return_excursion_max_frames": return_excursion_max_frames,
        "reject_unverified_return_excursions": reject_unverified_return_excursions,
        "reject_bare_prompt_matches": reject_bare_prompt_matches,
        "bare_rejection_policy": bare_rejection_policy,
        "ambiguous_glove_bare_observation_count": sum(
            bool(row.glove_verifier_prompts) and bool(row.bare_verifier_prompts)
            for rows in grouped.values()
            for row in rows
        ),
    }
