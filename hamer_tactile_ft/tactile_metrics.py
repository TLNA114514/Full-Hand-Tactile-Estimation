from dataclasses import dataclass
import hashlib
from math import isfinite

import numpy as np


TOUCHANYTHING_CONTACT_THRESHOLD = 0.1
TOUCHANYTHING_MIN_CONTACT_RATIO = 0.05
_FRAME_STAT_SIZE = 8
TOUCHANYTHING_SCENE_CATEGORIES = ("Home", "Workbench", "Office", "Retail", "Outdoor")


def touchanything_protocol_group_key(sequence_key, query_alias=None):
    """Collapse canonical per-query keys back to the source trajectory."""
    sequence_key = str(sequence_key)
    query_alias = str(query_alias or "")
    suffix = f"/{query_alias}" if query_alias else ""
    if suffix and sequence_key.endswith(suffix):
        return sequence_key[: -len(suffix)]
    return sequence_key


def touchanything_scene_category(sequence_key):
    """Return the canonical TouchAnything scene encoded in a sequence key."""
    parts = [part for part in str(sequence_key).replace("\\", "/").split("/") if part]
    aliases = {name.casefold(): name for name in TOUCHANYTHING_SCENE_CATEGORIES}
    for part in parts:
        match = aliases.get(part.casefold())
        if match is not None:
            return match
    return "Unknown"


def _sequence_hash(sequence_key):
    digest = hashlib.blake2b(str(sequence_key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


class CompactTouchAnythingProtocolAccumulator:
    """Compact per-frame protocol storage for large validation sets.

    The legacy accumulator stores one Python dictionary entry per trajectory
    frame. This representation keeps the same sufficient statistics in dense
    arrays and retains only one string per trajectory.
    """

    def __init__(self):
        self._sequence_hash_chunks = []
        self._frame_index_chunks = []
        self._frame_stat_chunks = []
        self._sequence_keys = {}

    def add(self, sequence_keys, frame_indices, frame_stats):
        frame_stats = np.asarray(frame_stats, dtype=np.float64)
        if frame_stats.ndim != 2 or frame_stats.shape[1] != _FRAME_STAT_SIZE:
            raise ValueError(
                f"frame_stats must have shape [N,{_FRAME_STAT_SIZE}], got {frame_stats.shape}"
            )
        if len(sequence_keys) != len(frame_stats) or len(frame_indices) != len(frame_stats):
            raise ValueError("sequence_keys, frame_indices, and frame_stats must have the same length")
        hashes = np.empty(len(frame_stats), dtype=np.uint64)
        for index, sequence_key in enumerate(sequence_keys):
            sequence_key = str(sequence_key)
            if not sequence_key:
                raise ValueError("TouchAnything-compatible metrics require a non-empty sequence_key")
            sequence_hash = _sequence_hash(sequence_key)
            previous = self._sequence_keys.setdefault(sequence_hash, sequence_key)
            if previous != sequence_key:
                raise RuntimeError(
                    "TouchAnything sequence hash collision between "
                    f"{previous!r} and {sequence_key!r}"
                )
            hashes[index] = sequence_hash
        self._sequence_hash_chunks.append(hashes)
        self._frame_index_chunks.append(np.asarray(frame_indices, dtype=np.int64))
        self._frame_stat_chunks.append(frame_stats)

    def pack(self):
        if self._frame_stat_chunks:
            sequence_hashes = np.concatenate(self._sequence_hash_chunks)
            frame_indices = np.concatenate(self._frame_index_chunks)
            frame_stats = np.concatenate(self._frame_stat_chunks, axis=0)
        else:
            sequence_hashes = np.zeros(0, dtype=np.uint64)
            frame_indices = np.zeros(0, dtype=np.int64)
            frame_stats = np.zeros((0, _FRAME_STAT_SIZE), dtype=np.float64)
        return {
            "sequence_hashes": sequence_hashes,
            "frame_indices": frame_indices,
            "frame_stats": frame_stats,
            "sequence_keys": dict(self._sequence_keys),
        }


def summarize_compact_touchanything_protocol(
    packed_items,
    *,
    min_contact_ratio=TOUCHANYTHING_MIN_CONTACT_RATIO,
    include_rows=True,
):
    """Merge compact rank-local arrays and reproduce sequence-macro metrics."""
    packed_items = [item for item in packed_items if item is not None]
    sequence_keys = {}
    hash_parts = []
    frame_parts = []
    stat_parts = []
    for item in packed_items:
        hashes = np.asarray(item.get("sequence_hashes", ()), dtype=np.uint64)
        frames = np.asarray(item.get("frame_indices", ()), dtype=np.int64)
        stats = np.asarray(item.get("frame_stats", ()), dtype=np.float64)
        if stats.size == 0:
            stats = np.zeros((0, _FRAME_STAT_SIZE), dtype=np.float64)
        if stats.shape != (len(hashes), _FRAME_STAT_SIZE) or len(frames) != len(hashes):
            raise ValueError("Invalid compact TouchAnything protocol payload")
        hash_parts.append(hashes)
        frame_parts.append(frames)
        stat_parts.append(stats)
        for raw_hash, raw_key in item.get("sequence_keys", {}).items():
            sequence_hash = int(raw_hash)
            sequence_key = str(raw_key)
            previous = sequence_keys.setdefault(sequence_hash, sequence_key)
            if previous != sequence_key:
                raise RuntimeError("TouchAnything sequence hash collision while merging ranks")

    if not hash_parts or sum(len(part) for part in hash_parts) == 0:
        return _summarize_touchanything_sequence_rows([])

    hashes = np.concatenate(hash_parts)
    frames = np.concatenate(frame_parts)
    stats = np.concatenate(stat_parts, axis=0)
    order = np.lexsort((frames, hashes))
    hashes = hashes[order]
    frames = frames[order]
    stats = stats[order]

    frame_starts = np.flatnonzero(
        np.r_[True, (hashes[1:] != hashes[:-1]) | (frames[1:] != frames[:-1])]
    )
    frame_hashes = hashes[frame_starts]
    frame_stats = np.add.reduceat(stats, frame_starts, axis=0)
    count = frame_stats[:, 1]
    pred_contact = frame_stats[:, 6] >= float(min_contact_ratio) * np.maximum(count, 1.0)
    gt_contact = frame_stats[:, 7] >= float(min_contact_ratio) * np.maximum(count, 1.0)
    sequence_frame_stats = np.column_stack(
        (
            frame_stats[:, :6],
            (pred_contact == gt_contact).astype(np.float64),
            np.ones(len(frame_stats), dtype=np.float64),
            (pred_contact & gt_contact).astype(np.float64),
            (pred_contact & ~gt_contact).astype(np.float64),
            (~pred_contact & gt_contact).astype(np.float64),
        )
    )
    sequence_starts = np.flatnonzero(np.r_[True, frame_hashes[1:] != frame_hashes[:-1]])
    unique_hashes = frame_hashes[sequence_starts]
    sequence_stats = np.add.reduceat(sequence_frame_stats, sequence_starts, axis=0)
    rows = [
        _touchanything_sequence_row(sequence_keys.get(int(key), str(int(key))), values)
        for key, values in zip(unique_hashes, sequence_stats)
    ]
    return _summarize_touchanything_sequence_rows(rows, include_rows=include_rows)


def merge_compact_touchanything_protocol_stats(packed_items):
    """Concatenate rank-local compact payloads without materializing frame dictionaries."""
    accumulator = {
        "sequence_hashes": [],
        "frame_indices": [],
        "frame_stats": [],
        "sequence_keys": {},
    }
    for item in packed_items:
        if item is None:
            continue
        hashes = np.asarray(item.get("sequence_hashes", ()), dtype=np.uint64)
        frames = np.asarray(item.get("frame_indices", ()), dtype=np.int64)
        stats = np.asarray(item.get("frame_stats", ()), dtype=np.float64)
        if stats.size == 0:
            stats = np.zeros((0, _FRAME_STAT_SIZE), dtype=np.float64)
        if stats.shape != (len(hashes), _FRAME_STAT_SIZE) or len(frames) != len(hashes):
            raise ValueError("Invalid compact TouchAnything protocol payload")
        accumulator["sequence_hashes"].append(hashes)
        accumulator["frame_indices"].append(frames)
        accumulator["frame_stats"].append(stats)
        for raw_hash, raw_key in item.get("sequence_keys", {}).items():
            sequence_hash = int(raw_hash)
            sequence_key = str(raw_key)
            previous = accumulator["sequence_keys"].setdefault(sequence_hash, sequence_key)
            if previous != sequence_key:
                raise RuntimeError("TouchAnything sequence hash collision while merging ranks")
    return {
        "sequence_hashes": (
            np.concatenate(accumulator["sequence_hashes"])
            if accumulator["sequence_hashes"]
            else np.zeros(0, dtype=np.uint64)
        ),
        "frame_indices": (
            np.concatenate(accumulator["frame_indices"])
            if accumulator["frame_indices"]
            else np.zeros(0, dtype=np.int64)
        ),
        "frame_stats": (
            np.concatenate(accumulator["frame_stats"], axis=0)
            if accumulator["frame_stats"]
            else np.zeros((0, _FRAME_STAT_SIZE), dtype=np.float64)
        ),
        "sequence_keys": accumulator["sequence_keys"],
    }


@dataclass(frozen=True)
class VolumetricIoUStats:
    per_frame: np.ndarray
    frame_macro: float
    global_micro: float
    intersection_sum: float
    union_sum: float


@dataclass(frozen=True)
class LocationDistributionStats:
    distribution_viou: np.ndarray
    pred_mass_on_gt_support: np.ndarray
    gt_mass_in_pred_support: np.ndarray
    eligible: np.ndarray


def location_distribution_stats(
    pred,
    gt,
    *,
    value_axis=-1,
    support_threshold=0.05,
    min_gt_volume=1.0,
    distribution_power=1.0,
    min_gt_peak=0.0,
    eps=1e-12,
):
    """Compute scale-invariant pressure-location metrics per frame.

    Metrics are defined only for frames meeting the raw GT volume and peak
    thresholds. ``distribution_power`` emphasizes pressure cores before
    normalization without changing the total-pressure scale. Ineligible
    entries are returned as NaN so callers retain frame alignment.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have identical shapes, got {pred.shape} and {gt.shape}")
    if pred.ndim == 0:
        raise ValueError("pred and gt must contain at least one value axis")
    if min_gt_volume < 0.0:
        raise ValueError("min_gt_volume must be nonnegative")
    if not isfinite(float(distribution_power)) or not 1.0 <= float(distribution_power) <= 4.0:
        raise ValueError("distribution_power must be finite and lie in [1, 4]")
    if not isfinite(float(min_gt_peak)) or not 0.0 <= float(min_gt_peak) <= 1.0:
        raise ValueError("min_gt_peak must be finite and lie in [0, 1]")

    pred_raw = np.maximum(pred, 0.0)
    gt_raw = np.maximum(gt, 0.0)
    if float(distribution_power) == 1.0:
        pred, gt = pred_raw, gt_raw
    else:
        pred = np.power(pred_raw, float(distribution_power))
        gt = np.power(gt_raw, float(distribution_power))
    pred_volume = pred.sum(axis=value_axis, dtype=np.float64)
    gt_volume = gt.sum(axis=value_axis, dtype=np.float64)
    gt_volume_raw = gt_raw.sum(axis=value_axis, dtype=np.float64)
    gt_peak_raw = gt_raw.max(axis=value_axis)
    eligible = (gt_volume_raw >= float(min_gt_volume)) & (gt_peak_raw >= float(min_gt_peak))

    pred_denom = np.expand_dims(np.maximum(pred_volume, float(eps)), axis=value_axis)
    gt_denom = np.expand_dims(np.maximum(gt_volume, float(eps)), axis=value_axis)
    pred_dist = pred / pred_denom
    gt_dist = gt / gt_denom
    intersection = np.minimum(pred_dist, gt_dist).sum(axis=value_axis, dtype=np.float64)
    union = np.maximum(pred_dist, gt_dist).sum(axis=value_axis, dtype=np.float64)
    distribution_viou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > float(eps),
    )

    gt_support = gt_raw >= float(support_threshold)
    pred_support = pred_raw >= float(support_threshold)
    pred_mass_on_gt_support = np.divide(
        np.where(gt_support, pred, 0.0).sum(axis=value_axis, dtype=np.float64),
        pred_volume,
        out=np.zeros_like(pred_volume, dtype=np.float64),
        where=pred_volume > float(eps),
    )
    gt_mass_in_pred_support = np.divide(
        np.where(pred_support, gt, 0.0).sum(axis=value_axis, dtype=np.float64),
        gt_volume,
        out=np.zeros_like(gt_volume, dtype=np.float64),
        where=gt_volume > float(eps),
    )

    def mask_ineligible(values):
        return np.where(eligible, values, np.nan).astype(np.float64, copy=False)

    return LocationDistributionStats(
        distribution_viou=mask_ineligible(distribution_viou),
        pred_mass_on_gt_support=mask_ineligible(pred_mass_on_gt_support),
        gt_mass_in_pred_support=mask_ineligible(gt_mass_in_pred_support),
        eligible=np.asarray(eligible, dtype=bool),
    )


def volumetric_iou_stats(pred, gt, *, value_axis=-1, eps=1e-12):
    """Compute frame-macro and global-micro soft volumetric IoU."""
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have identical shapes, got {pred.shape} and {gt.shape}")
    if pred.ndim == 0:
        raise ValueError("pred and gt must contain at least one value axis")

    intersection = np.minimum(pred, gt).sum(axis=value_axis, dtype=np.float64)
    union = np.maximum(pred, gt).sum(axis=value_axis, dtype=np.float64)
    per_frame = np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union > float(eps),
    )
    intersection_sum = float(np.asarray(intersection, dtype=np.float64).sum())
    union_sum = float(np.asarray(union, dtype=np.float64).sum())
    global_micro = intersection_sum / union_sum if union_sum > float(eps) else 1.0
    return VolumetricIoUStats(
        per_frame=np.asarray(per_frame, dtype=np.float64),
        frame_macro=float(np.asarray(per_frame, dtype=np.float64).mean()),
        global_micro=float(global_micro),
        intersection_sum=intersection_sum,
        union_sum=union_sum,
    )


def touchanything_protocol_frame_stats(
    pred,
    gt,
    *,
    value_axis=-1,
    contact_threshold=TOUCHANYTHING_CONTACT_THRESHOLD,
):
    """Return per-frame sufficient statistics for TouchAnything-style aggregation.

    The caller must first select the valid canonical palm vertices. TouchAnything's
    bend-sensor mask has no direct equivalent on a canonical mesh, so it is not
    applied here.
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have identical shapes, got {pred.shape} and {gt.shape}")
    if pred.ndim == 0:
        raise ValueError("pred and gt must contain at least one value axis")

    valid = np.isfinite(gt)
    pred_valid = np.where(valid, pred, 0.0)
    gt_valid = np.where(valid, gt, 0.0)
    value_count = valid.sum(axis=value_axis, dtype=np.float64)
    abs_sum = np.abs(pred_valid - gt_valid).sum(axis=value_axis, dtype=np.float64)
    vol_intersection = np.minimum(pred_valid, gt_valid).sum(axis=value_axis, dtype=np.float64)
    vol_union = np.maximum(pred_valid, gt_valid).sum(axis=value_axis, dtype=np.float64)

    pred_binary = (pred_valid > float(contact_threshold)) & valid
    gt_binary = (gt_valid > float(contact_threshold)) & valid
    contact_intersection = (pred_binary & gt_binary).sum(axis=value_axis, dtype=np.float64)
    contact_union = (pred_binary | gt_binary).sum(axis=value_axis, dtype=np.float64)
    pred_active_count = pred_binary.sum(axis=value_axis, dtype=np.float64)
    gt_active_count = gt_binary.sum(axis=value_axis, dtype=np.float64)

    return np.stack(
        (
            abs_sum,
            value_count,
            vol_intersection,
            vol_union,
            contact_intersection,
            contact_union,
            pred_active_count,
            gt_active_count,
        ),
        axis=-1,
    )


def accumulate_touchanything_protocol(sequence_stats, sequence_keys, frame_indices, frame_stats):
    frame_stats = np.asarray(frame_stats, dtype=np.float64)
    if frame_stats.ndim != 2 or frame_stats.shape[1] != _FRAME_STAT_SIZE:
        raise ValueError(
            f"frame_stats must have shape [N,{_FRAME_STAT_SIZE}], got {frame_stats.shape}"
        )
    if len(sequence_keys) != len(frame_stats) or len(frame_indices) != len(frame_stats):
        raise ValueError("sequence_keys, frame_indices, and frame_stats must have the same length")
    for sequence_key, frame_index, values in zip(sequence_keys, frame_indices, frame_stats):
        sequence_key = str(sequence_key)
        if not sequence_key:
            raise ValueError("TouchAnything-compatible metrics require a non-empty sequence_key")
        key = (sequence_key, int(frame_index))
        if key not in sequence_stats:
            sequence_stats[key] = np.zeros(_FRAME_STAT_SIZE, dtype=np.float64)
        sequence_stats[key] += values
    return sequence_stats


def merge_touchanything_protocol_stats(items):
    merged = {}
    for item in items:
        for key, values in item.items():
            values = np.asarray(values, dtype=np.float64)
            if values.shape != (_FRAME_STAT_SIZE,):
                raise ValueError(f"Invalid sequence statistics for {key}: {values.shape}")
            if key not in merged:
                merged[key] = np.zeros(_FRAME_STAT_SIZE, dtype=np.float64)
            merged[key] += values
    return merged


def _touchanything_sequence_row(sequence_key, values):
    values = np.asarray(values, dtype=np.float64)
    abs_sum, count, vol_inter, vol_union, con_inter, con_union, correct, frames, tp, fp, fn = values
    mae = abs_sum / count if count > 0 else float("nan")
    volumetric_iou = vol_inter / vol_union if vol_union > 0 else float("nan")
    contact_iou = con_inter / con_union if con_union > 0 else float("nan")
    temporal_accuracy = correct / frames if frames > 0 else float("nan")
    temporal_precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    temporal_recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    temporal_f1 = (
        2.0 * temporal_precision * temporal_recall / (temporal_precision + temporal_recall)
        if temporal_precision + temporal_recall > 0
        else 0.0
    )
    return {
        "sequence_key": str(sequence_key),
        "scene_category": touchanything_scene_category(sequence_key),
        "frame_count": int(frames),
        "mae": float(mae),
        "volumetric_iou": float(volumetric_iou),
        "contact_iou": float(contact_iou),
        "temporal_accuracy": float(temporal_accuracy),
        "temporal_precision": float(temporal_precision),
        "temporal_recall": float(temporal_recall),
        "temporal_f1": float(temporal_f1),
    }


def _summarize_touchanything_sequence_rows(rows, *, include_rows=True):
    rows = list(rows)

    def finite_mean(name, selected_rows=rows):
        values = [row[name] for row in selected_rows if isfinite(row[name])]
        return float(np.mean(values)) if values else float("nan")

    by_scene = {}
    for scene in TOUCHANYTHING_SCENE_CATEGORIES:
        selected = [row for row in rows if row.get("scene_category") == scene]
        by_scene[scene] = {
            "sequence_count": len(selected),
            "frame_count": int(sum(row["frame_count"] for row in selected)),
            "mae": finite_mean("mae", selected),
            "volumetric_iou": finite_mean("volumetric_iou", selected),
            "contact_iou": finite_mean("contact_iou", selected),
            "temporal_accuracy": finite_mean("temporal_accuracy", selected),
        }

    summary = {
        "sequence_count": len(rows),
        "mae": finite_mean("mae"),
        "volumetric_iou": finite_mean("volumetric_iou"),
        "contact_iou": finite_mean("contact_iou"),
        "temporal_accuracy": finite_mean("temporal_accuracy"),
        "temporal_precision": finite_mean("temporal_precision"),
        "temporal_recall": finite_mean("temporal_recall"),
        "temporal_f1": finite_mean("temporal_f1"),
        "by_scene": by_scene,
    }
    summary["rows"] = rows if include_rows else []
    return summary


def summarize_touchanything_protocol(
    sequence_stats,
    *,
    min_contact_ratio=TOUCHANYTHING_MIN_CONTACT_RATIO,
):
    aggregated = {}
    for key, frame_values in sequence_stats.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(f"Expected (sequence_key, frame_idx), got {key!r}")
        sequence_key, _frame_index = key
        values = np.asarray(frame_values, dtype=np.float64)
        abs_sum, count, vol_inter, vol_union, con_inter, con_union, pred_active, gt_active = values
        pred_contact = pred_active >= float(min_contact_ratio) * max(count, 1.0)
        gt_contact = gt_active >= float(min_contact_ratio) * max(count, 1.0)
        if sequence_key not in aggregated:
            aggregated[sequence_key] = np.zeros(11, dtype=np.float64)
        aggregated[sequence_key] += np.asarray(
            (
                abs_sum,
                count,
                vol_inter,
                vol_union,
                con_inter,
                con_union,
                float(pred_contact == gt_contact),
                1.0,
                float(pred_contact and gt_contact),
                float(pred_contact and not gt_contact),
                float(not pred_contact and gt_contact),
            ),
            dtype=np.float64,
        )

    rows = []
    for key in sorted(aggregated):
        rows.append(_touchanything_sequence_row(key, aggregated[key]))
    return _summarize_touchanything_sequence_rows(rows)
