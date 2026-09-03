"""Query-safe tactile history cache, pairing, and bounded temporal residual."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hamer_tactile_ft.hamer_tactile import (
    _canonical_mesh_assets,
    _canonical_rbf_assignment,
)

from .feature_cache import FeatureCacheDataset, canonical_json, sha256_file, sha256_json


TEMPORAL_PAIR_SCHEMA = "tactile_temporal_pairs_v8"
TEMPORAL_MODEL_FORMAT = "tactile_temporal_flow_v1"
TEMPORAL_SELECTOR_FORMAT = "tactile_temporal_action_selector_v2"
PREDICTION_CONTROL_SCHEMA = "tactile_temporal_prediction_control_v1"
PREDICTION_PRESSURE_BIN_EDGES = (0.005, 0.05, 0.2, 0.5, 0.7)
PER_LAG_QUALITY_DIM = 5
TEMPORAL_REQUIRED_FIELDS = ("palm_base_logits", "palm_tactile_signal", "has_tactile")


def _homogeneous_affine(value: np.ndarray | torch.Tensor) -> np.ndarray:
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape == (3, 3):
        result = affine
    elif affine.shape == (2, 3):
        result = np.vstack((affine, np.asarray((0.0, 0.0, 1.0))))
    else:
        raise ValueError(f"Expected a 2x3 or 3x3 affine, got {affine.shape}")
    if not np.isfinite(result).all() or abs(float(np.linalg.det(result))) < 1e-12:
        raise ValueError("Crop affine must be finite and invertible")
    return result


def tactile_crop_affine(
    bbox_xyxy: Sequence[float],
    *,
    input_resolution: Sequence[int],
    bbox_rescale_factor: float,
    is_right: bool,
) -> np.ndarray:
    """Return source-image to canonical RGB-crop coordinates.

    This mirrors ``OpenTouchTactileDataset.__getitem__`` including the
    post-crop horizontal flip used to anonymize left hands.
    """

    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError(f"Expected one finite xyxy bbox, got {bbox_xyxy!r}")
    height, width = (int(value) for value in input_resolution)
    side = float(np.max(bbox[2:] - bbox[:2]))
    crop_size = float(bbox_rescale_factor) * side
    if min(height, width) <= 0 or crop_size <= 1.0:
        raise ValueError("Invalid crop resolution or bbox size")
    center = (bbox[:2] + bbox[2:]) * 0.5
    affine = np.asarray(
        (
            (height / crop_size, 0.0, -height * center[0] / crop_size + width * 0.5),
            (0.0, height / crop_size, height * (-center[1] / crop_size + 0.5)),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    if not bool(is_right):
        flip = np.asarray(
            ((-1.0, 0.0, width - 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        affine = flip @ affine
    return affine.astype(np.float32)


def current_to_history_crop_affine(
    current_affine: np.ndarray | torch.Tensor,
    history_affine: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Map current-crop pixel coordinates into a history crop."""

    current = _homogeneous_affine(current_affine)
    history = _homogeneous_affine(history_affine)
    return (history @ np.linalg.inv(current)).astype(np.float32)


def _bbox_metrics(previous: Sequence[float], current: Sequence[float]) -> tuple[float, float, float]:
    a = np.asarray(previous, dtype=np.float64)
    b = np.asarray(current, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.nan, math.nan, math.nan
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if min(aw, ah, bw, bh) <= 1.0:
        return math.nan, math.nan, math.nan
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    union = aw * ah + bw * bh - intersection
    iou = intersection / union if union > 0 else 0.0
    ac = np.asarray(((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5))
    bc = np.asarray(((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5))
    center_jump = float(np.linalg.norm(ac - bc) / max(math.sqrt(aw * ah), 1.0))
    area_ratio = abs(math.log(max(bw * bh, 1.0) / max(aw * ah, 1.0)))
    return float(iou), center_jump, area_ratio


class PartitionedPalmCache:
    """Interleave stride-built cache partitions back into global dataset order."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        *,
        max_open_shards: int = 4,
        optional_fields: Sequence[str] = (),
    ):
        self.root = Path(root).expanduser().resolve(strict=True)
        if (self.root / "CACHE_DONE.json").is_file():
            paths = (self.root,)
        else:
            paths = tuple(
                child
                for child in sorted(self.root.glob("part-*-of-*"))
                if (child / "CACHE_DONE.json").is_file()
            )
        if not paths:
            raise FileNotFoundError(f"No finalized feature cache under {self.root}")
        requested_fields = tuple(
            dict.fromkeys((*TEMPORAL_REQUIRED_FIELDS, *(str(value) for value in optional_fields)))
        )
        self.parts = tuple(
            FeatureCacheDataset(
                path,
                fields=requested_fields,
                max_open_shards=max_open_shards,
                copy_arrays=False,
            )
            for path in paths
        )
        self.fields = requested_fields
        expected = len(self.parts)
        for index, path in enumerate(paths):
            if expected > 1 and path.name != f"part-{index:02d}-of-{expected:02d}":
                raise RuntimeError(f"Non-contiguous cache partition set under {self.root}")
        self.sample_count = sum(len(part) for part in self.parts)
        base_hashes = {
            str(part.config.get("provenance", {}).get("base_checkpoint_sha256", ""))
            for part in self.parts
        }
        base_hashes.discard("")
        if len(base_hashes) != 1:
            raise RuntimeError(f"Temporal cache mixes frozen bases: {sorted(base_hashes)}")
        self.base_checkpoint_sha256 = next(iter(base_hashes))
        palm_sets = {
            tuple(part.config.get("provenance", {}).get("palm_vertex_indices", ()))
            for part in self.parts
        }
        if len(palm_sets) != 1 or not next(iter(palm_sets)):
            raise RuntimeError("Temporal cache lacks one consistent palm vertex definition")
        self.palm_vertex_indices = np.asarray(next(iter(palm_sets)), dtype=np.int64)
        resolutions = {
            tuple(part.config.get("provenance", {}).get("input_resolution", ()))
            for part in self.parts
        }
        bbox_scales = {
            float(part.config.get("provenance", {}).get("bbox_rescale_factor", math.nan))
            for part in self.parts
        }
        if len(resolutions) != 1 or len(next(iter(resolutions), ())) != 2:
            raise RuntimeError("Temporal cache lacks one consistent input resolution")
        if len(bbox_scales) != 1 or not math.isfinite(next(iter(bbox_scales))):
            raise RuntimeError("Temporal cache lacks one consistent bbox scale")
        self.input_resolution = tuple(int(value) for value in next(iter(resolutions)))
        self.bbox_rescale_factor = float(next(iter(bbox_scales)))
        self.config_sha256 = sha256_json(
            {
                "parts": [part.config_sha256 for part in self.parts],
            }
        )

    def __len__(self) -> int:
        return self.sample_count

    def _location(self, index: int) -> tuple[FeatureCacheDataset, int]:
        index = int(index)
        if index < 0:
            index += self.sample_count
        if not 0 <= index < self.sample_count:
            raise IndexError(index)
        part_index = index % len(self.parts)
        local_index = index // len(self.parts)
        if local_index >= len(self.parts[part_index]):
            raise IndexError(index)
        return self.parts[part_index], local_index

    def values(self, index: int) -> dict[str, np.ndarray]:
        part, local_index = self._location(index)
        return part.field_values(local_index, self.fields)

    def sample_records(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        part_count = len(self.parts)
        for part_index, part in enumerate(self.parts):
            connection = sqlite3.connect(
                f"file:{part.sample_index_path.as_posix()}?mode=ro&immutable=1", uri=True
            )
            try:
                columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(samples)")
                }
                required = {
                    "source_frame_idx", "timestamp", "is_right", "bbox_xyxy",
                    "bbox_association_id",
                }
                if not required.issubset(columns):
                    raise RuntimeError(
                        "Temporal cache predates strict pair metadata; rebuild it with "
                        "the current cache_tactile_features.py"
                    )
                query = (
                    "SELECT sample_id, ordinal, dataset, sequence_key, query_alias, "
                    "frame_idx, source_frame_idx, timestamp, is_right, bbox_xyxy, "
                    "bbox_association_id FROM samples ORDER BY ordinal"
                )
                for row in connection.execute(query):
                    sample_id, ordinal = row[:2]
                    global_index = int(ordinal) * part_count + part_index
                    if global_index >= self.sample_count:
                        raise RuntimeError("Cache partition ordinal exceeds global sample count")
                    key = str(sample_id)
                    if key in result:
                        raise RuntimeError(f"Duplicate cached sample UID: {key}")
                    result[key] = {
                        "cache_index": global_index,
                        "dataset": str(row[2]),
                        "sequence_key": str(row[3]),
                        "query_alias": str(row[4]),
                        "frame_idx": int(row[5]),
                        "source_frame_idx": None if row[6] is None else int(row[6]),
                        "timestamp": None if row[7] is None else float(row[7]),
                        "is_right": int(row[8]),
                        "bbox_xyxy": json.loads(str(row[9])),
                        "bbox_association_id": str(row[10]),
                    }
            finally:
                connection.close()
        if len(result) != self.sample_count:
            raise RuntimeError(
                f"Cache ID index covers {len(result)}/{self.sample_count} samples"
            )
        return result

    def sample_id_index(self) -> dict[str, int]:
        return {
            sample_id: int(record["cache_index"])
            for sample_id, record in self.sample_records().items()
        }


def temporal_manifest_key(manifests: Iterable[os.PathLike[str] | str]) -> str:
    """Return one relocation-independent key for an ordered manifest set."""

    hashes = [sha256_file(Path(path).expanduser().resolve(strict=True)) for path in manifests]
    if not hashes:
        raise ValueError("At least one query manifest is required")
    return sha256_json({"manifest_sha256": hashes})[:12]


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _pressure_bin(value: Any) -> int:
    value = float(value or 0.0)
    return int(np.searchsorted(PREDICTION_PRESSURE_BIN_EDGES, value, side="right"))


def build_prediction_control_bins(
    cache: PartitionedPalmCache,
    pair_index: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    *,
    force: bool = False,
) -> Path:
    """Cache label-free control bins from the frozen RGB maximum prediction."""

    pair_path = Path(pair_index).expanduser().resolve(strict=True)
    output_path = Path(output_path).expanduser().resolve(strict=False)
    metadata_path = output_path.with_suffix(".json")
    contract = {
        "schema": PREDICTION_CONTROL_SCHEMA,
        "cache_config_sha256": cache.config_sha256,
        "pair_index_sha256": sha256_file(pair_path),
        "score": "sigmoid(max(palm_base_logits))",
        "bin_edges": list(PREDICTION_PRESSURE_BIN_EDGES),
    }
    contract_sha256 = sha256_json(contract)
    if not force and output_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("contract_sha256") == contract_sha256:
            return output_path

    with np.load(pair_path, allow_pickle=False) as payload:
        current_indices = np.asarray(payload["current_index"], dtype=np.int64)
    predicted_max = np.empty(len(current_indices), dtype=np.float32)
    for position, cache_index in enumerate(current_indices):
        logits = np.asarray(
            cache.values(int(cache_index))["palm_base_logits"], dtype=np.float32
        )
        maximum_logit = float(np.max(logits))
        predicted_max[position] = 1.0 / (
            1.0 + math.exp(-max(-60.0, min(60.0, maximum_logit)))
        )
    bins = np.searchsorted(
        np.asarray(PREDICTION_PRESSURE_BIN_EDGES, dtype=np.float32),
        predicted_max,
        side="right",
    ).astype(np.int8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            prediction_pressure_bin=bins,
            prediction_max=predicted_max,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    counts = np.bincount(
        bins.astype(np.int64), minlength=len(PREDICTION_PRESSURE_BIN_EDGES) + 1
    )
    metadata = {
        **contract,
        "contract_sha256": contract_sha256,
        "pair_count": int(len(current_indices)),
        "bin_counts": counts.tolist(),
    }
    temporary_meta = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    temporary_meta.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_meta, metadata_path)
    return output_path


def _strict_control_indices(
    sequence_keys: Sequence[str], sides: Sequence[int], pressure_bins: Sequence[int], seed: int
) -> np.ndarray:
    output = np.full(len(sequence_keys), -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    broad: dict[int, list[int]] = defaultdict(list)
    fine: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (side, pressure_bin) in enumerate(zip(sides, pressure_bins)):
        broad[int(side)].append(index)
        fine[(int(side), int(pressure_bin))].append(index)

    def assignment(indices: list[int]) -> dict[int, int] | None:
        if len(indices) < 2:
            return None
        by_sequence: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            by_sequence[str(sequence_keys[index])].append(index)
        maximum = max(map(len, by_sequence.values()))
        if maximum * 2 > len(indices):
            return None
        ordered = []
        for key in sorted(by_sequence):
            values = by_sequence[key]
            rng.shuffle(values)
            ordered.extend(values)
        result = {}
        for position, index in enumerate(ordered):
            candidate = ordered[(position + maximum) % len(ordered)]
            if sequence_keys[index] == sequence_keys[candidate]:
                return None
            result[index] = candidate
        return result

    for side, indices in broad.items():
        candidate_assignments = []
        fine_ok = True
        for pressure_bin in sorted({int(pressure_bins[index]) for index in indices}):
            values = fine[(side, pressure_bin)]
            current = assignment(values)
            if current is None:
                fine_ok = False
                break
            candidate_assignments.append(current)
        if fine_ok:
            for current in candidate_assignments:
                for index, candidate in current.items():
                    output[index] = candidate
            continue
        current = assignment(indices)
        if current is None:
            raise RuntimeError(
                f"Cannot build no-replacement cross-sequence control for side={side}"
            )
        for index, candidate in current.items():
            output[index] = candidate
    if np.any(output < 0):
        raise RuntimeError("Strict temporal control construction left uncovered pairs")
    return output


def build_temporal_pair_index(
    cache: PartitionedPalmCache,
    manifests: Iterable[os.PathLike[str] | str],
    output_path: os.PathLike[str] | str,
    *,
    max_time_gap: float = 0.05,
    min_bbox_iou: float = 0.05,
    max_bbox_center_jump: float = 0.5,
    max_bbox_area_ratio: float = 2.0,
    seed: int = 521,
    label_free_controls: bool = False,
    force: bool = False,
) -> Path:
    output_path = Path(output_path).expanduser().resolve(strict=False)
    metadata_path = output_path.with_suffix(".json")
    manifest_paths = tuple(Path(path).expanduser().resolve(strict=True) for path in manifests)
    contract = {
        "schema": TEMPORAL_PAIR_SCHEMA,
        "cache_config_sha256": cache.config_sha256,
        "manifest_sha256": [sha256_file(path) for path in manifest_paths],
        "max_time_gap": float(max_time_gap),
        "min_bbox_iou": float(min_bbox_iou),
        "max_bbox_center_jump": float(max_bbox_center_jump),
        "max_bbox_area_ratio": float(max_bbox_area_ratio),
        "seed": int(seed),
        "control_pressure_source": (
            "none_label_free" if label_free_controls else "manifest_max_pressure"
        ),
    }
    contract_sha = sha256_json(contract)
    if not force and output_path.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") == contract_sha:
            return output_path

    cached_records = cache.sample_records()
    previous_by_group: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    history_by_sequence_side: dict[
        tuple[str, str, str, int], dict[int, dict[str, Any]]
    ] = defaultdict(dict)
    current_indices: list[int] = []
    previous_indices: list[int] = []
    time_gaps: list[float] = []
    bbox_ious: list[float] = []
    center_jumps: list[float] = []
    area_ratios: list[float] = []
    sequence_keys: list[str] = []
    sides: list[int] = []
    pressure_bins: list[int] = []
    contralateral_previous_indices: list[int] = []
    current_crop_affines: list[np.ndarray] = []
    previous_crop_affines: list[np.ndarray] = []
    contralateral_previous_crop_affines: list[np.ndarray] = []
    reset_counts: Counter[str] = Counter()
    scanned = 0
    for manifest in manifest_paths:
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                scanned += 1
                uid = str(row.get("sample_uid") or "")
                cached = cached_records.get(uid)
                if cached is None:
                    reset_counts["not_in_cache"] += 1
                    continue
                cache_index = int(cached["cache_index"])
                alias = str(cached["query_alias"] or "query").lower()
                side = int(cached["is_right"])
                group = (
                    str(cached["dataset"] or row.get("dataset") or ""),
                    str(row.get("split") or ""),
                    str(cached["sequence_key"] or row.get("sequence_key") or ""),
                    alias,
                    side,
                )
                current = {
                    "cache_index": cache_index,
                    "frame": int(
                        cached["source_frame_idx"]
                        if cached["source_frame_idx"] is not None
                        else cached["frame_idx"]
                    ),
                    "timestamp": _finite_float(cached["timestamp"]),
                    "bbox": cached["bbox_xyxy"],
                    "association": str(cached["bbox_association_id"] or ""),
                    "h5": str(row.get("h5_path") or row.get("h5_relpath") or ""),
                    "sequence": group[2],
                    "side": side,
                    "pressure_bin": (
                        0
                        if label_free_controls
                        else _pressure_bin(row.get("max_pressure"))
                    ),
                    "crop_affine": tactile_crop_affine(
                        cached["bbox_xyxy"],
                        input_resolution=cache.input_resolution,
                        bbox_rescale_factor=cache.bbox_rescale_factor,
                        is_right=bool(side),
                    ),
                }
                previous = previous_by_group.get(group)
                previous_by_group[group] = current
                opposite_history = history_by_sequence_side[
                    (group[0], group[1], group[2], 1 - side)
                ]
                opposite = opposite_history.get(int(current["frame"]) - 1)
                own_history = history_by_sequence_side[
                    (group[0], group[1], group[2], side)
                ]
                own_history[int(current["frame"])] = current
                for stale_frame in tuple(own_history):
                    if stale_frame < int(current["frame"]) - 2:
                        del own_history[stale_frame]
                if previous is None:
                    reset_counts["cold_start"] += 1
                    continue
                if current["frame"] - previous["frame"] != 1:
                    reset_counts["frame_gap"] += 1
                    continue
                if current["h5"] != previous["h5"]:
                    reset_counts["source_change"] += 1
                    continue
                if current["timestamp"] is None or previous["timestamp"] is None:
                    reset_counts["missing_timestamp"] += 1
                    continue
                time_gap = current["timestamp"] - previous["timestamp"]
                if not 0.0 < time_gap <= float(max_time_gap):
                    reset_counts["time_gap"] += 1
                    continue
                if (
                    current["association"]
                    and previous["association"]
                    and current["association"] != previous["association"]
                ):
                    reset_counts["association_change"] += 1
                    continue
                bbox_iou, center_jump, area_ratio = _bbox_metrics(
                    previous["bbox"], current["bbox"]
                )
                if not (
                    math.isfinite(bbox_iou)
                    and bbox_iou >= float(min_bbox_iou)
                    and center_jump <= float(max_bbox_center_jump)
                    and area_ratio <= math.log(float(max_bbox_area_ratio))
                ):
                    reset_counts["bbox_unstable"] += 1
                    continue
                current_indices.append(cache_index)
                previous_indices.append(int(previous["cache_index"]))
                time_gaps.append(time_gap)
                bbox_ious.append(bbox_iou)
                center_jumps.append(center_jump)
                area_ratios.append(area_ratio)
                sequence_keys.append(current["sequence"])
                sides.append(side)
                pressure_bins.append(current["pressure_bin"])
                contralateral_previous_indices.append(
                    int(opposite["cache_index"])
                    if opposite is not None
                    and int(opposite["frame"]) == int(current["frame"]) - 1
                    and str(opposite["h5"]) == str(current["h5"])
                    else -1
                )
                current_crop_affines.append(np.asarray(current["crop_affine"], dtype=np.float32))
                previous_crop_affines.append(
                    np.asarray(previous["crop_affine"], dtype=np.float32)
                )
                contralateral_previous_crop_affines.append(
                    np.asarray(opposite["crop_affine"], dtype=np.float32)
                    if opposite is not None
                    and int(opposite["frame"]) == int(current["frame"]) - 1
                    and str(opposite["h5"]) == str(current["h5"])
                    else np.full((3, 3), np.nan, dtype=np.float32)
                )

    if not current_indices:
        raise RuntimeError("Strict temporal pairing produced no eligible records")
    control_pair_indices = _strict_control_indices(
        sequence_keys, sides, pressure_bins, seed
    )
    control_previous_indices = np.asarray(previous_indices, dtype=np.int64)[
        control_pair_indices
    ]
    arrays = {
        "current_index": np.asarray(current_indices, dtype=np.int64),
        "previous_index": np.asarray(previous_indices, dtype=np.int64),
        "control_previous_index": control_previous_indices,
        "control_pair_index": np.asarray(control_pair_indices, dtype=np.int64),
        "contralateral_previous_index": np.asarray(
            contralateral_previous_indices, dtype=np.int64
        ),
        "current_crop_affine": np.stack(current_crop_affines).astype(np.float32),
        "previous_crop_affine": np.stack(previous_crop_affines).astype(np.float32),
        "contralateral_previous_crop_affine": np.stack(
            contralateral_previous_crop_affines
        ).astype(np.float32),
        "time_gap": np.asarray(time_gaps, dtype=np.float32),
        "bbox_iou": np.asarray(bbox_ious, dtype=np.float32),
        "bbox_center_jump": np.asarray(center_jumps, dtype=np.float32),
        "bbox_abs_log_area_ratio": np.asarray(area_ratios, dtype=np.float32),
        "side": np.asarray(sides, dtype=np.int8),
        "sequence_key": np.asarray(sequence_keys, dtype=np.str_),
        "pressure_bin": np.asarray(pressure_bins, dtype=np.int8),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    metadata = {
        **contract,
        "contract_sha256": contract_sha,
        "pair_count": len(current_indices),
        "scanned_record_count": scanned,
        "reset_counts": dict(reset_counts),
        "cross_sequence_control_fraction": float(
            np.mean(
                [
                    sequence_keys[index] != sequence_keys[int(control_pair_indices[index])]
                    for index in range(len(sequence_keys))
                ]
            )
        ),
    }
    temporary_meta = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    temporary_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_meta, metadata_path)
    return output_path


def strict_lag_history_indices(
    cache_length: int,
    current_indices: np.ndarray,
    previous_indices: np.ndarray,
    lags: Sequence[int],
) -> np.ndarray:
    """Chain only validated adjacent edges; unavailable histories remain -1."""

    current = np.asarray(current_indices, dtype=np.int64)
    previous = np.asarray(previous_indices, dtype=np.int64)
    if current.shape != previous.shape or current.ndim != 1:
        raise ValueError("current/previous pair arrays must be matching vectors")
    if len(np.unique(current)) != len(current):
        raise ValueError("adjacent pair index contains duplicate current samples")
    lags = tuple(int(value) for value in lags)
    if not lags or len(set(lags)) != len(lags) or any(value <= 0 for value in lags):
        raise ValueError("history lags must be unique positive integers")
    if len(current) and (
        current.min() < 0
        or previous.min() < 0
        or current.max() >= int(cache_length)
        or previous.max() >= int(cache_length)
    ):
        raise ValueError("pair index refers outside the temporal cache")
    previous_by_index = np.full(int(cache_length), -1, dtype=np.int64)
    previous_by_index[current] = previous
    histories = np.full((len(current), len(lags)), -1, dtype=np.int64)
    requested = {lag: column for column, lag in enumerate(lags)}
    cursor = current.copy()
    for step in range(1, max(lags) + 1):
        valid = cursor >= 0
        next_cursor = np.full_like(cursor, -1)
        next_cursor[valid] = previous_by_index[cursor[valid]]
        cursor = next_cursor
        if step in requested:
            histories[:, requested[step]] = cursor
    if 1 in requested and not np.array_equal(histories[:, requested[1]], previous):
        raise RuntimeError("strict lag-1 history differs from the validated pair index")
    return histories


def strict_lag_history_metadata(
    cache_length: int,
    current_indices: np.ndarray,
    previous_indices: np.ndarray,
    lags: Sequence[int],
    *,
    time_gap: np.ndarray,
    bbox_iou: np.ndarray,
    bbox_center_jump: np.ndarray,
    bbox_abs_log_area_ratio: np.ndarray,
    contralateral_previous_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Accumulate timing and association quality along validated lag-1 edges.

    Every reported long-lag value is built only from edges that already passed
    the strict timestamp, source, association, and bbox checks.  Missing chains
    retain neutral metadata and an index of ``-1``.
    """

    current = np.asarray(current_indices, dtype=np.int64)
    previous = np.asarray(previous_indices, dtype=np.int64)
    lags = tuple(int(value) for value in lags)
    histories = strict_lag_history_indices(
        cache_length, current, previous, lags
    )
    edge_values = {
        "time_gap": np.asarray(time_gap, dtype=np.float64),
        "bbox_iou": np.asarray(bbox_iou, dtype=np.float64),
        "bbox_center_jump": np.asarray(bbox_center_jump, dtype=np.float64),
        "bbox_abs_log_area_ratio": np.asarray(
            bbox_abs_log_area_ratio, dtype=np.float64
        ),
    }
    for name, values in edge_values.items():
        if values.shape != current.shape or not np.isfinite(values).all():
            raise ValueError(
                f"strict temporal edge metadata {name!r} must be one finite vector"
            )
    contralateral = None
    if contralateral_previous_indices is not None:
        contralateral = np.asarray(
            contralateral_previous_indices, dtype=np.int64
        )
        if contralateral.shape != current.shape:
            raise ValueError(
                "contralateral previous indices must match strict temporal edges"
            )
        if len(contralateral) and (
            contralateral.min() < -1
            or contralateral.max() >= int(cache_length)
        ):
            raise ValueError("contralateral temporal index refers outside the cache")

    previous_by_index = np.full(int(cache_length), -1, dtype=np.int64)
    edge_row_by_index = np.full(int(cache_length), -1, dtype=np.int64)
    previous_by_index[current] = previous
    edge_row_by_index[current] = np.arange(len(current), dtype=np.int64)
    requested = {lag: column for column, lag in enumerate(lags)}
    shape = (len(current), len(lags))
    cumulative_time = np.zeros(shape, dtype=np.float32)
    minimum_iou = np.ones(shape, dtype=np.float32)
    maximum_center_jump = np.zeros(shape, dtype=np.float32)
    maximum_area_change = np.zeros(shape, dtype=np.float32)
    contralateral_histories = np.full(shape, -1, dtype=np.int64)

    cursor = current.copy()
    running_time = np.zeros(len(current), dtype=np.float64)
    running_iou = np.ones(len(current), dtype=np.float64)
    running_center = np.zeros(len(current), dtype=np.float64)
    running_area = np.zeros(len(current), dtype=np.float64)
    for step in range(1, max(lags) + 1):
        valid_cursor = cursor >= 0
        rows = np.full(len(current), -1, dtype=np.int64)
        rows[valid_cursor] = edge_row_by_index[cursor[valid_cursor]]
        valid_edge = rows >= 0
        if bool(valid_edge.any()):
            selected = rows[valid_edge]
            running_time[valid_edge] += edge_values["time_gap"][selected]
            running_iou[valid_edge] = np.minimum(
                running_iou[valid_edge], edge_values["bbox_iou"][selected]
            )
            running_center[valid_edge] = np.maximum(
                running_center[valid_edge],
                edge_values["bbox_center_jump"][selected],
            )
            running_area[valid_edge] = np.maximum(
                running_area[valid_edge],
                edge_values["bbox_abs_log_area_ratio"][selected],
            )
        next_cursor = np.full_like(cursor, -1)
        next_cursor[valid_edge] = previous_by_index[cursor[valid_edge]]
        if step in requested:
            column = requested[step]
            available = next_cursor >= 0
            cumulative_time[available, column] = running_time[available]
            minimum_iou[available, column] = running_iou[available]
            maximum_center_jump[available, column] = running_center[available]
            maximum_area_change[available, column] = running_area[available]
            if contralateral is not None:
                contralateral_histories[available, column] = contralateral[
                    rows[available]
                ]
        cursor = next_cursor

    if not np.array_equal(histories >= 0, cumulative_time > 0.0):
        raise RuntimeError("Long-lag time metadata disagrees with strict history coverage")
    return {
        "history_indices": histories,
        "history_time_gap": cumulative_time,
        "history_min_bbox_iou": minimum_iou,
        "history_max_bbox_center_jump": maximum_center_jump,
        "history_max_bbox_abs_log_area_ratio": maximum_area_change,
        "contralateral_history_indices": contralateral_histories,
    }


def strict_history_control_pair_indices(
    sequence_keys: Sequence[str],
    sides: Sequence[int],
    pressure_bins: Sequence[int],
    history_available: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Match every control's lag availability while changing its sequence."""

    availability = np.asarray(history_available, dtype=np.bool_)
    if availability.ndim != 2 or availability.shape[0] != len(sequence_keys):
        raise ValueError("history availability must be [pair,lag]")
    if len(sides) != len(sequence_keys) or len(pressure_bins) != len(sequence_keys):
        raise ValueError("control metadata arrays have different lengths")
    output = np.full(len(sequence_keys), -1, dtype=np.int64)
    patterns = np.packbits(availability, axis=1, bitorder="little")
    pattern_keys = [bytes(row) for row in patterns]
    for pattern_number, pattern in enumerate(sorted(set(pattern_keys))):
        indices = np.asarray(
            [index for index, value in enumerate(pattern_keys) if value == pattern],
            dtype=np.int64,
        )
        local = _strict_control_indices(
            [str(sequence_keys[index]) for index in indices],
            [int(sides[index]) for index in indices],
            [int(pressure_bins[index]) for index in indices],
            int(seed) + 1009 * pattern_number,
        )
        output[indices] = indices[local]
    if np.any(output < 0):
        raise RuntimeError("Availability-matched temporal control is incomplete")
    if np.any(availability != availability[output]):
        raise RuntimeError("Temporal controls do not preserve lag availability")
    if any(
        str(sequence_keys[index]) == str(sequence_keys[int(output[index])])
        for index in range(len(output))
    ):
        raise RuntimeError("Temporal control contains a same-sequence history")
    return output


def _pair_control_seed(pair_index_path: Path) -> int:
    metadata_path = pair_index_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise RuntimeError(f"Temporal pair metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != TEMPORAL_PAIR_SCHEMA:
        raise RuntimeError(
            f"Temporal pair metadata schema is not current: {metadata.get('schema')!r}"
        )
    return int(metadata.get("seed", 521))


def _pair_crop_affine_lookup(
    cache_length: int, arrays: Mapping[str, np.ndarray]
) -> np.ndarray:
    required = {
        "current_index",
        "previous_index",
        "contralateral_previous_index",
        "current_crop_affine",
        "previous_crop_affine",
        "contralateral_previous_crop_affine",
    }
    if not required.issubset(arrays):
        raise RuntimeError(
            "Historical DINO evidence requires a v8 temporal pair index with crop affines"
        )
    lookup = np.full((int(cache_length), 3, 3), np.nan, dtype=np.float32)
    for indices_name, affine_name in (
        ("current_index", "current_crop_affine"),
        ("previous_index", "previous_crop_affine"),
        (
            "contralateral_previous_index",
            "contralateral_previous_crop_affine",
        ),
    ):
        indices = np.asarray(arrays[indices_name], dtype=np.int64)
        affines = np.asarray(arrays[affine_name], dtype=np.float32)
        if affines.shape != (len(indices), 3, 3):
            raise RuntimeError(f"Malformed {affine_name}: {affines.shape}")
        for index, affine in zip(indices, affines):
            if int(index) < 0:
                continue
            old = lookup[int(index)]
            if np.isfinite(old).all() and not np.allclose(old, affine, atol=1e-5):
                raise RuntimeError(f"Cache sample {int(index)} has inconsistent crop affines")
            lookup[int(index)] = affine
    return lookup


def _history_crop_transforms(
    crop_affines: np.ndarray,
    current_index: int,
    history_indices: Sequence[int],
) -> np.ndarray:
    history_indices = np.asarray(history_indices, dtype=np.int64)
    if history_indices.ndim != 1 or not len(history_indices):
        raise ValueError("history_indices must be one non-empty vector")
    # Complete-split replay includes cold-start/reset rows that have no entry
    # in the temporal pair index. Their history tensors copy the current frame
    # but are masked unavailable, so identity is the only neutral transform and
    # no crop affine is needed. Keep strict affine validation whenever at least
    # one real history frame is present.
    if bool(np.all(history_indices < 0)):
        return np.repeat(
            np.eye(3, dtype=np.float32)[None], len(history_indices), axis=0
        )
    current = crop_affines[int(current_index)]
    if not np.isfinite(current).all():
        raise RuntimeError(f"Current cache sample {int(current_index)} lacks a crop affine")
    transforms = []
    for history_index in history_indices:
        if int(history_index) < 0:
            transforms.append(np.eye(3, dtype=np.float32))
            continue
        history = crop_affines[int(history_index)]
        if not np.isfinite(history).all():
            raise RuntimeError(
                f"History cache sample {int(history_index)} lacks a crop affine"
            )
        transforms.append(current_to_history_crop_affine(current, history))
    return np.stack(transforms).astype(np.float32)


class TemporalPairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_root: os.PathLike[str] | str,
        pair_index: os.PathLike[str] | str,
        *,
        include_control: bool = False,
        include_dino_grid: bool = False,
        include_crop_transform: bool = True,
        include_control_current_grid: bool = True,
        history_lags: Sequence[int] = (1,),
        max_open_shards: int = 4,
        control_pressure_bins: Sequence[int] | None = None,
        control_crop_transform_from_current: bool = False,
    ):
        self.include_dino_grid = bool(include_dino_grid)
        self.include_crop_transform = bool(
            include_dino_grid and include_crop_transform
        )
        self.include_control_current_grid = bool(
            include_dino_grid and include_control_current_grid
        )
        self.control_crop_transform_from_current = bool(
            control_crop_transform_from_current
        )
        self.cache = PartitionedPalmCache(
            cache_root,
            max_open_shards=max_open_shards,
            optional_fields=("z_rgb",) if self.include_dino_grid else (),
        )
        self.pair_index_path = Path(pair_index).expanduser().resolve(strict=True)
        with np.load(self.pair_index_path, allow_pickle=False) as payload:
            self.arrays = {name: np.asarray(payload[name]) for name in payload.files}
        length = len(self.arrays["current_index"])
        if any(len(value) != length for value in self.arrays.values()):
            raise RuntimeError("Temporal pair-index arrays have different lengths")
        self.include_control = bool(include_control)
        self.history_lags = tuple(int(value) for value in history_lags)
        self.pair_sequence_ids = np.unique(
            np.asarray(self.arrays["sequence_key"], dtype=np.str_),
            return_inverse=True,
        )[1].astype(np.int64)
        self.sequence_count = int(self.pair_sequence_ids.max()) + 1 if length else 0
        self.history_metadata = strict_lag_history_metadata(
            len(self.cache),
            self.arrays["current_index"],
            self.arrays["previous_index"],
            self.history_lags,
            time_gap=self.arrays["time_gap"],
            bbox_iou=self.arrays["bbox_iou"],
            bbox_center_jump=self.arrays["bbox_center_jump"],
            bbox_abs_log_area_ratio=self.arrays["bbox_abs_log_area_ratio"],
            contralateral_previous_indices=self.arrays.get(
                "contralateral_previous_index"
            ),
        )
        self.history_indices = self.history_metadata["history_indices"]
        self.crop_affines = (
            _pair_crop_affine_lookup(len(self.cache), self.arrays)
            if self.include_crop_transform
            else None
        )
        self.control_history_indices = None
        self.control_pair_indices = None
        if self.include_control:
            required = {"sequence_key", "pressure_bin", "side"}
            if not required.issubset(self.arrays):
                raise RuntimeError(
                    "History controls require the current temporal pair index"
                )
            if control_pressure_bins is None:
                pressure_bins = np.asarray(self.arrays["pressure_bin"], dtype=np.int64)
                self.control_bin_source = "target_max_pressure"
            else:
                pressure_bins = np.asarray(control_pressure_bins, dtype=np.int64)
                if pressure_bins.shape != (length,):
                    raise ValueError(
                        "control_pressure_bins must contain one value per temporal pair"
                    )
                if np.any(pressure_bins < 0):
                    raise ValueError("control_pressure_bins cannot contain negative values")
                self.control_bin_source = "external_label_free"
            self.control_pressure_bins = pressure_bins
            control_pairs = strict_history_control_pair_indices(
                self.arrays["sequence_key"],
                self.arrays["side"],
                pressure_bins,
                self.history_indices >= 0,
                seed=_pair_control_seed(self.pair_index_path),
            )
            self.control_pair_indices = control_pairs
            self.control_history_indices = self.history_indices[control_pairs]

    def __len__(self) -> int:
        return len(self.arrays["current_index"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        current_index = int(self.arrays["current_index"][index])
        current = self.cache.values(current_index)

        def tensor(value, dtype=None):
            result = torch.from_numpy(np.array(value, copy=True))
            return result if dtype is None else result.to(dtype=dtype)

        history_logits = []
        history_targets = []
        history_available = []
        history_grids = []
        for history_index in self.history_indices[index]:
            available = int(history_index) >= 0
            history = self.cache.values(int(history_index)) if available else current
            history_logits.append(tensor(history["palm_base_logits"]))
            history_targets.append(
                tensor(history["palm_tactile_signal"], torch.float32)
            )
            history_available.append(float(available))
            if self.include_dino_grid:
                history_grids.append(tensor(history["z_rgb"]))
        result = {
            "current_logits": tensor(current["palm_base_logits"]),
            "history_logits": torch.stack(history_logits),
            "history_available": torch.tensor(
                history_available, dtype=torch.float32
            ),
            "history_time_gap": tensor(
                self.history_metadata["history_time_gap"][index], torch.float32
            ),
            "history_min_bbox_iou": tensor(
                self.history_metadata["history_min_bbox_iou"][index],
                torch.float32,
            ),
            "history_max_bbox_center_jump": tensor(
                self.history_metadata["history_max_bbox_center_jump"][index],
                torch.float32,
            ),
            "history_max_bbox_abs_log_area_ratio": tensor(
                self.history_metadata["history_max_bbox_abs_log_area_ratio"][index],
                torch.float32,
            ),
            "tactile_signal": tensor(current["palm_tactile_signal"], torch.float32),
            "history_tactile_signal": torch.stack(history_targets),
            "has_tactile": tensor(current["has_tactile"], torch.float32).reshape(()),
            "time_gap": torch.tensor(float(self.arrays["time_gap"][index])),
            "bbox_iou": torch.tensor(float(self.arrays["bbox_iou"][index])),
            "bbox_center_jump": torch.tensor(
                float(self.arrays["bbox_center_jump"][index])
            ),
            "bbox_abs_log_area_ratio": torch.tensor(
                float(self.arrays["bbox_abs_log_area_ratio"][index])
            ),
            "current_index": torch.tensor(current_index),
            "sequence_id": torch.tensor(int(self.pair_sequence_ids[index])),
        }
        # Preserve the v1 single-lag batch contract for historical checkpoints.
        result["previous_logits"] = result["history_logits"][0]
        result["previous_tactile_signal"] = result["history_tactile_signal"][0]
        if self.include_dino_grid:
            result.update(
                {
                    "current_grid": tensor(current["z_rgb"]),
                    "history_grids": torch.stack(history_grids),
                }
            )
            if self.include_crop_transform:
                result["history_crop_transform"] = tensor(
                    _history_crop_transforms(
                        self.crop_affines,
                        current_index,
                        self.history_indices[index],
                    ),
                    torch.float32,
                )
        if self.include_control:
            if self.control_history_indices is None:
                control_indices = (int(self.arrays["control_previous_index"][index]),)
            else:
                control_indices = self.control_history_indices[index]
            control_logits = []
            control_available = []
            control_grids = []
            for history_index in control_indices:
                available = int(history_index) >= 0
                history = self.cache.values(int(history_index)) if available else current
                control_logits.append(tensor(history["palm_base_logits"]))
                control_available.append(float(available))
                if self.include_dino_grid:
                    control_grids.append(tensor(history["z_rgb"]))
            result["control_history_logits"] = torch.stack(control_logits)
            result["control_history_available"] = torch.tensor(
                control_available, dtype=torch.float32
            )
            control_pair = int(self.control_pair_indices[index])
            if self.include_dino_grid:
                control_current_index = int(self.arrays["current_index"][control_pair])
                result["control_history_grids"] = torch.stack(control_grids)
                if self.include_control_current_grid:
                    control_current = self.cache.values(control_current_index)
                    result["control_current_grid"] = tensor(control_current["z_rgb"])
                if self.include_crop_transform:
                    transform_current_index = (
                        current_index
                        if self.control_crop_transform_from_current
                        else control_current_index
                    )
                    result["control_history_crop_transform"] = tensor(
                        _history_crop_transforms(
                            self.crop_affines,
                            transform_current_index,
                            control_indices,
                        ),
                        torch.float32,
                    )
            for name in (
                "history_time_gap",
                "history_min_bbox_iou",
                "history_max_bbox_center_jump",
                "history_max_bbox_abs_log_area_ratio",
            ):
                result[f"control_{name}"] = tensor(
                    self.history_metadata[name][control_pair], torch.float32
                )
            result["control_previous_logits"] = result["control_history_logits"][0]
            contralateral_index = int(
                self.arrays["contralateral_previous_index"][index]
            )
            result["contralateral_available"] = torch.tensor(
                contralateral_index >= 0, dtype=torch.float32
            )
            if contralateral_index >= 0:
                contralateral = self.cache.values(contralateral_index)
                result["contralateral_previous_logits"] = tensor(
                    contralateral["palm_base_logits"]
                )
            else:
                result["contralateral_previous_logits"] = result[
                    "previous_logits"
                ].clone()
            contralateral_indices = self.history_metadata[
                "contralateral_history_indices"
            ][index]
            contralateral_history = result["history_logits"].clone()
            contralateral_available = torch.zeros_like(result["history_available"])
            contralateral_grids = (
                result["history_grids"].clone() if self.include_dino_grid else None
            )
            contralateral_transforms = (
                np.repeat(
                    np.eye(3, dtype=np.float32)[None],
                    len(contralateral_indices),
                    axis=0,
                )
                if self.include_crop_transform
                else None
            )
            for column, history_index in enumerate(contralateral_indices):
                if int(history_index) < 0:
                    continue
                opposite = self.cache.values(int(history_index))
                contralateral_history[column] = tensor(
                    opposite["palm_base_logits"]
                )
                contralateral_available[column] = 1.0
                if self.include_dino_grid:
                    contralateral_grids[column] = tensor(opposite["z_rgb"])
                    if self.include_crop_transform:
                        contralateral_transforms[column] = _history_crop_transforms(
                            self.crop_affines, current_index, (int(history_index),)
                        )[0]
            result["contralateral_history_logits"] = contralateral_history
            result["contralateral_history_available"] = contralateral_available
            if self.include_dino_grid:
                result["contralateral_history_grids"] = contralateral_grids
                if self.include_crop_transform:
                    result["contralateral_history_crop_transform"] = tensor(
                        contralateral_transforms, torch.float32
                    )
        return result


class TemporalReplayDataset(torch.utils.data.Dataset):
    """Replay a complete split, falling back exactly to RGB at every reset."""

    def __init__(
        self,
        cache_root: os.PathLike[str] | str,
        pair_index: os.PathLike[str] | str,
        *,
        include_control: bool = True,
        include_dino_grid: bool = False,
        include_crop_transform: bool = True,
        include_control_current_grid: bool = True,
        history_lags: Sequence[int] = (1,),
        max_open_shards: int = 4,
        control_pressure_bins: Sequence[int] | None = None,
        control_crop_transform_from_current: bool = False,
    ):
        self.include_dino_grid = bool(include_dino_grid)
        self.include_crop_transform = bool(
            include_dino_grid and include_crop_transform
        )
        self.include_control_current_grid = bool(
            include_dino_grid and include_control_current_grid
        )
        self.control_crop_transform_from_current = bool(
            control_crop_transform_from_current
        )
        self.cache = PartitionedPalmCache(
            cache_root,
            max_open_shards=max_open_shards,
            optional_fields=("z_rgb",) if self.include_dino_grid else (),
        )
        self.pair_index_path = Path(pair_index).expanduser().resolve(strict=True)
        with np.load(self.pair_index_path, allow_pickle=False) as payload:
            self.arrays = {name: np.asarray(payload[name]) for name in payload.files}
        pair_count = len(self.arrays["current_index"])
        if any(len(value) != pair_count for value in self.arrays.values()):
            raise RuntimeError("Temporal pair-index arrays have different lengths")
        self.pair_lookup = np.full(len(self.cache), -1, dtype=np.int64)
        current = np.asarray(self.arrays["current_index"], dtype=np.int64)
        if len(np.unique(current)) != len(current):
            raise RuntimeError("Temporal pair index assigns multiple histories to one sample")
        self.pair_lookup[current] = np.arange(pair_count, dtype=np.int64)
        self.pair_sequence_ids = np.unique(
            np.asarray(self.arrays["sequence_key"], dtype=np.str_),
            return_inverse=True,
        )[1].astype(np.int64)
        self.sequence_count = (
            int(self.pair_sequence_ids.max()) + 1
            if self.pair_sequence_ids.size
            else 0
        )
        self.include_control = bool(include_control)
        self.history_lags = tuple(int(value) for value in history_lags)
        self.history_metadata = strict_lag_history_metadata(
            len(self.cache),
            current,
            self.arrays["previous_index"],
            self.history_lags,
            time_gap=self.arrays["time_gap"],
            bbox_iou=self.arrays["bbox_iou"],
            bbox_center_jump=self.arrays["bbox_center_jump"],
            bbox_abs_log_area_ratio=self.arrays["bbox_abs_log_area_ratio"],
            contralateral_previous_indices=self.arrays.get(
                "contralateral_previous_index"
            ),
        )
        self.history_indices = self.history_metadata["history_indices"]
        self.crop_affines = (
            _pair_crop_affine_lookup(len(self.cache), self.arrays)
            if self.include_crop_transform
            else None
        )
        self.control_history_indices = None
        self.control_pair_indices = None
        if self.include_control:
            required = {"sequence_key", "pressure_bin", "side"}
            if not required.issubset(self.arrays):
                raise RuntimeError(
                    "History controls require the current temporal pair index"
                )
            if control_pressure_bins is None:
                pressure_bins = np.asarray(self.arrays["pressure_bin"], dtype=np.int64)
                self.control_bin_source = "target_max_pressure"
            else:
                pressure_bins = np.asarray(control_pressure_bins, dtype=np.int64)
                if pressure_bins.shape != (pair_count,):
                    raise ValueError(
                        "control_pressure_bins must contain one value per temporal pair"
                    )
                if np.any(pressure_bins < 0):
                    raise ValueError("control_pressure_bins cannot contain negative values")
                self.control_bin_source = "external_label_free"
            self.control_pressure_bins = pressure_bins
            control_pairs = strict_history_control_pair_indices(
                self.arrays["sequence_key"],
                self.arrays["side"],
                pressure_bins,
                self.history_indices >= 0,
                seed=_pair_control_seed(self.pair_index_path),
            )
            self.control_pair_indices = control_pairs
            self.control_history_indices = self.history_indices[control_pairs]

    @property
    def pair_count(self) -> int:
        return int(len(self.arrays["current_index"]))

    def __len__(self) -> int:
        return len(self.cache)

    @staticmethod
    def _tensor(value, dtype=None):
        result = torch.from_numpy(np.array(value, copy=True))
        return result if dtype is None else result.to(dtype=dtype)

    def __getitem__(self, current_index: int) -> dict[str, Any]:
        current_index = int(current_index)
        current = self.cache.values(current_index)
        pair_position = int(self.pair_lookup[current_index])
        eligible = pair_position >= 0
        history_indices = (
            self.history_indices[pair_position]
            if eligible
            else np.full(len(self.history_lags), -1, dtype=np.int64)
        )
        history_logits = []
        history_targets = []
        history_available = []
        history_grids = []
        for history_index in history_indices:
            available = int(history_index) >= 0
            history = self.cache.values(int(history_index)) if available else current
            history_logits.append(self._tensor(history["palm_base_logits"]))
            history_targets.append(
                self._tensor(history["palm_tactile_signal"], torch.float32)
            )
            history_available.append(float(available))
            if self.include_dino_grid:
                history_grids.append(self._tensor(history["z_rgb"]))
        result = {
            "current_logits": self._tensor(current["palm_base_logits"]),
            "history_logits": torch.stack(history_logits),
            "history_tactile_signal": torch.stack(history_targets),
            "history_available": torch.tensor(
                history_available, dtype=torch.float32
            ),
            "history_time_gap": self._tensor(
                self.history_metadata["history_time_gap"][pair_position]
                if eligible
                else np.zeros(len(self.history_lags), dtype=np.float32),
                torch.float32,
            ),
            "history_min_bbox_iou": self._tensor(
                self.history_metadata["history_min_bbox_iou"][pair_position]
                if eligible
                else np.ones(len(self.history_lags), dtype=np.float32),
                torch.float32,
            ),
            "history_max_bbox_center_jump": self._tensor(
                self.history_metadata["history_max_bbox_center_jump"][pair_position]
                if eligible
                else np.zeros(len(self.history_lags), dtype=np.float32),
                torch.float32,
            ),
            "history_max_bbox_abs_log_area_ratio": self._tensor(
                self.history_metadata["history_max_bbox_abs_log_area_ratio"][pair_position]
                if eligible
                else np.zeros(len(self.history_lags), dtype=np.float32),
                torch.float32,
            ),
            "tactile_signal": self._tensor(
                current["palm_tactile_signal"], torch.float32
            ),
            "has_tactile": self._tensor(
                current["has_tactile"], torch.float32
            ).reshape(()),
            "temporal_eligible": torch.tensor(
                eligible and all(value > 0.5 for value in history_available),
                dtype=torch.float32,
            ),
            "lag1_temporal_eligible": torch.tensor(
                eligible, dtype=torch.float32
            ),
            "time_gap": torch.tensor(
                float(self.arrays["time_gap"][pair_position]) if eligible else 0.0
            ),
            "bbox_iou": torch.tensor(
                float(self.arrays["bbox_iou"][pair_position]) if eligible else 1.0
            ),
            "bbox_center_jump": torch.tensor(
                float(self.arrays["bbox_center_jump"][pair_position])
                if eligible
                else 0.0
            ),
            "bbox_abs_log_area_ratio": torch.tensor(
                float(self.arrays["bbox_abs_log_area_ratio"][pair_position])
                if eligible
                else 0.0
            ),
            "current_index": torch.tensor(current_index),
            # Cold-start rows do not enter matched temporal bootstrap. Their
            # sequence identity is intentionally left unknown instead of
            # guessing it from neighboring cache order.
            "sequence_id": torch.tensor(
                int(self.pair_sequence_ids[pair_position]) if eligible else -1
            ),
        }
        result["previous_logits"] = result["history_logits"][0]
        if self.include_dino_grid:
            result.update(
                {
                    "current_grid": self._tensor(current["z_rgb"]),
                    "history_grids": torch.stack(history_grids),
                }
            )
            if self.include_crop_transform:
                result["history_crop_transform"] = self._tensor(
                    _history_crop_transforms(
                        self.crop_affines, current_index, history_indices
                    ),
                    torch.float32,
                )
        if self.include_control:
            if eligible and self.control_history_indices is not None:
                control_indices = self.control_history_indices[pair_position]
            elif eligible:
                control_indices = np.asarray(
                    [self.arrays["control_previous_index"][pair_position]],
                    dtype=np.int64,
                )
            else:
                control_indices = np.full(
                    len(self.history_lags), -1, dtype=np.int64
                )
            control_logits = []
            control_available = []
            control_grids = []
            for history_index in control_indices:
                available = int(history_index) >= 0
                history = self.cache.values(int(history_index)) if available else current
                control_logits.append(self._tensor(history["palm_base_logits"]))
                control_available.append(float(available))
                if self.include_dino_grid:
                    control_grids.append(self._tensor(history["z_rgb"]))
            result["control_history_logits"] = torch.stack(control_logits)
            result["control_history_available"] = torch.tensor(
                control_available, dtype=torch.float32
            )
            control_pair = (
                int(self.control_pair_indices[pair_position])
                if eligible and self.control_pair_indices is not None
                else -1
            )
            if self.include_dino_grid:
                control_current_index = (
                    int(self.arrays["current_index"][control_pair])
                    if control_pair >= 0
                    else current_index
                )
                result["control_history_grids"] = torch.stack(control_grids)
                if self.include_control_current_grid:
                    control_current = self.cache.values(control_current_index)
                    result["control_current_grid"] = self._tensor(
                        control_current["z_rgb"]
                    )
                if self.include_crop_transform:
                    transform_current_index = (
                        current_index
                        if self.control_crop_transform_from_current
                        else control_current_index
                    )
                    result["control_history_crop_transform"] = self._tensor(
                        _history_crop_transforms(
                            self.crop_affines,
                            transform_current_index,
                            control_indices,
                        ),
                        torch.float32,
                    )
            for name in (
                "history_time_gap",
                "history_min_bbox_iou",
                "history_max_bbox_center_jump",
                "history_max_bbox_abs_log_area_ratio",
            ):
                default = (
                    np.ones(len(self.history_lags), dtype=np.float32)
                    if name == "history_min_bbox_iou"
                    else np.zeros(len(self.history_lags), dtype=np.float32)
                )
                result[f"control_{name}"] = self._tensor(
                    self.history_metadata[name][control_pair]
                    if control_pair >= 0
                    else default,
                    torch.float32,
                )
            result["control_previous_logits"] = result["control_history_logits"][0]
            contralateral_index = (
                int(self.arrays["contralateral_previous_index"][pair_position])
                if eligible
                else -1
            )
            result["contralateral_available"] = torch.tensor(
                contralateral_index >= 0, dtype=torch.float32
            )
            if contralateral_index >= 0:
                opposite = self.cache.values(contralateral_index)
                result["contralateral_previous_logits"] = self._tensor(
                    opposite["palm_base_logits"]
                )
            else:
                result["contralateral_previous_logits"] = result[
                    "current_logits"
                ].clone()
            contralateral_indices = (
                self.history_metadata["contralateral_history_indices"][pair_position]
                if eligible
                else np.full(len(self.history_lags), -1, dtype=np.int64)
            )
            contralateral_history = result["history_logits"].clone()
            contralateral_available = torch.zeros_like(result["history_available"])
            contralateral_grids = (
                result["history_grids"].clone() if self.include_dino_grid else None
            )
            contralateral_transforms = (
                np.repeat(
                    np.eye(3, dtype=np.float32)[None],
                    len(contralateral_indices),
                    axis=0,
                )
                if self.include_crop_transform
                else None
            )
            for column, history_index in enumerate(contralateral_indices):
                if int(history_index) < 0:
                    continue
                opposite = self.cache.values(int(history_index))
                contralateral_history[column] = self._tensor(
                    opposite["palm_base_logits"]
                )
                contralateral_available[column] = 1.0
                if self.include_dino_grid:
                    contralateral_grids[column] = self._tensor(opposite["z_rgb"])
                    if self.include_crop_transform:
                        contralateral_transforms[column] = _history_crop_transforms(
                            self.crop_affines, current_index, (int(history_index),)
                        )[0]
            result["contralateral_history_logits"] = contralateral_history
            result["contralateral_history_available"] = contralateral_available
            if self.include_dino_grid:
                result["contralateral_history_grids"] = contralateral_grids
                if self.include_crop_transform:
                    result["contralateral_history_crop_transform"] = self._tensor(
                        contralateral_transforms, torch.float32
                    )
        return result


class AnchorGraphBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        neighbor_mean = torch.zeros_like(hidden)
        for column in range(neighbors.shape[1]):
            neighbor_mean.add_(hidden[:, neighbors[:, column]])
        neighbor_mean.div_(float(neighbors.shape[1]))
        return hidden + self.mlp(self.norm(torch.cat((hidden, neighbor_mean), dim=-1)))


class QueryAwareTemporalResidual(nn.Module):
    """Bounded temporal residual with legacy or signed additive lag fusion."""

    def __init__(
        self,
        palm_vertex_indices: Sequence[int],
        *,
        anchor_count: int = 512,
        anchor_neighbors: int = 4,
        graph_neighbors: int = 4,
        hidden_dim: int = 64,
        graph_layers: int = 2,
        dropout: float = 0.05,
        max_history_alpha: float = 0.75,
        max_logit_delta: float = 0.50,
        architecture: str = "legacy_product",
        history_lags: Sequence[int] = (1,),
        use_per_lag_quality: bool = False,
        nominal_fps: float = 30.0,
    ):
        super().__init__()
        architecture = str(architecture).strip().lower()
        if architecture not in ("legacy_product", "signed_additive"):
            raise ValueError(
                "architecture must be legacy_product or signed_additive"
            )
        history_lags = tuple(int(value) for value in history_lags)
        if (
            not history_lags
            or len(set(history_lags)) != len(history_lags)
            or any(value <= 0 for value in history_lags)
        ):
            raise ValueError("history_lags must be unique positive integers")
        if architecture == "legacy_product" and history_lags != (1,):
            raise ValueError("legacy_product supports only history_lags=(1,)")
        palm_vertex_indices = torch.as_tensor(palm_vertex_indices, dtype=torch.long)
        tactile_dim = int(palm_vertex_indices.max().item()) + 1
        vertices, _ = _canonical_mesh_assets()
        tactile_dim = int(vertices.shape[0])
        anchor_vertices, vertex_anchor_indices, vertex_anchor_weights, valid_mask = (
            _canonical_rbf_assignment(
                tactile_dim=tactile_dim,
                anchor_count=int(anchor_count),
                neighbor_count=int(anchor_neighbors),
            )
        )
        if not torch.equal(torch.nonzero(valid_mask).flatten(), palm_vertex_indices):
            raise RuntimeError("Temporal cache palm vertices differ from canonical palm mask")
        full_to_palm = torch.full((tactile_dim,), -1, dtype=torch.long)
        full_to_palm[palm_vertex_indices] = torch.arange(len(palm_vertex_indices))
        anchor_local = full_to_palm[anchor_vertices]
        if bool((anchor_local < 0).any()):
            raise RuntimeError("Canonical temporal anchor lies outside the valid palm")
        anchor_xyz = vertices[anchor_vertices]
        anchor_xyz = (anchor_xyz - anchor_xyz.mean(0)) / anchor_xyz.std(0).clamp_min(1e-6)
        distances = torch.cdist(anchor_xyz, anchor_xyz)
        graph_index = torch.topk(
            distances, k=min(int(graph_neighbors) + 1, int(anchor_count)), largest=False
        ).indices[:, 1:]
        self.register_buffer("palm_vertex_indices", palm_vertex_indices)
        self.register_buffer("anchor_local_indices", anchor_local)
        self.register_buffer("vertex_anchor_indices", vertex_anchor_indices[palm_vertex_indices])
        self.register_buffer("vertex_anchor_weights", vertex_anchor_weights[palm_vertex_indices])
        self.register_buffer("anchor_xyz", anchor_xyz)
        self.register_buffer("graph_neighbors", graph_index)
        self.anchor_count = int(anchor_count)
        self.max_history_alpha = float(max_history_alpha)
        self.max_logit_delta = float(max_logit_delta)
        self.architecture = architecture
        self.history_lags = history_lags
        self.use_per_lag_quality = bool(use_per_lag_quality)
        self.nominal_fps = float(nominal_fps)
        if self.nominal_fps <= 0.0:
            raise ValueError("nominal_fps must be positive")
        if self.architecture == "legacy_product" and self.use_per_lag_quality:
            raise ValueError("per-lag quality is supported only by signed_additive")
        quality_dim = (
            PER_LAG_QUALITY_DIM * len(self.history_lags)
            if self.use_per_lag_quality
            else 0
        )
        input_dim = (
            6 + 4 + 3
            if self.architecture == "legacy_product"
            else 2 + 6 * len(self.history_lags) + quality_dim + 4 + 3
        )
        self.input_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_blocks = nn.ModuleList(
            AnchorGraphBlock(hidden_dim, dropout) for _ in range(int(graph_layers))
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        if self.architecture == "legacy_product":
            self.transition_head = nn.Linear(hidden_dim, 3)
            self.history_gate_head = nn.Linear(hidden_dim, 1)
            self.global_rezero_gate = nn.Parameter(torch.zeros(()))
        else:
            lag_count = len(self.history_lags)
            self.transition_head = nn.Linear(hidden_dim, lag_count * 3)
            self.history_gate_head = nn.Linear(hidden_dim, lag_count)
            self.coefficient_head = nn.Linear(hidden_dim, lag_count)
            nn.init.zeros_(self.coefficient_head.weight)
            nn.init.zeros_(self.coefficient_head.bias)
        nn.init.zeros_(self.history_gate_head.weight)
        nn.init.zeros_(self.history_gate_head.bias)

    def _interpolate(self, anchor_values: torch.Tensor) -> torch.Tensor:
        values = anchor_values[:, self.vertex_anchor_indices]
        weights = self.vertex_anchor_weights.to(values)[None]
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        return (values * weights).sum(dim=2)

    def _signed_additive_forward(
        self,
        current_logits: torch.Tensor,
        history_logits: torch.Tensor,
        pair_context: torch.Tensor,
        history_available: torch.Tensor | None,
        history_quality: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch_size, vertex_count = current_logits.shape
        lag_count = len(self.history_lags)
        if history_logits.shape != (batch_size, lag_count, vertex_count):
            raise ValueError(
                "signed_additive history logits must be "
                f"[B,{lag_count},V], got {tuple(history_logits.shape)}"
            )
        if history_available is None:
            history_available = torch.ones(
                (batch_size, lag_count),
                device=current_logits.device,
                dtype=current_logits.dtype,
            )
        if history_available.shape != (batch_size, lag_count):
            raise ValueError(
                "history_available must match [B,L], got "
                f"{tuple(history_available.shape)}"
            )
        available = history_available.float().clamp(0.0, 1.0)
        if self.use_per_lag_quality:
            expected = (batch_size, lag_count, PER_LAG_QUALITY_DIM)
            if history_quality is None or tuple(history_quality.shape) != expected:
                actual = None if history_quality is None else tuple(history_quality.shape)
                raise ValueError(
                    f"history_quality must be {expected} when enabled, got {actual}"
                )
            quality = history_quality.float() * available[:, :, None]
        else:
            quality = None
        current = current_logits.float()
        history = history_logits.float()
        history = torch.where(available[:, :, None] > 0.5, history, current[:, None])
        current_anchor = current[:, self.anchor_local_indices]
        history_anchor = history[:, :, self.anchor_local_indices].transpose(1, 2)
        difference = history_anchor - current_anchor[:, :, None]
        local_features = [
            current_anchor[:, :, None],
            torch.sigmoid(current_anchor)[:, :, None],
        ]
        maximum_lag = float(max(self.history_lags))
        for column, lag in enumerate(self.history_lags):
            lag_available = available[:, None, column].expand(-1, self.anchor_count)
            local_features.extend(
                (
                    history_anchor[:, :, column : column + 1],
                    difference[:, :, column : column + 1],
                    difference[:, :, column : column + 1].abs(),
                    torch.sigmoid(history_anchor[:, :, column : column + 1]),
                    lag_available[:, :, None],
                    torch.full_like(
                        history_anchor[:, :, column : column + 1],
                        float(lag) / maximum_lag,
                    ),
                )
            )
            if quality is not None:
                local_features.append(
                    quality[:, None, column, :].expand(-1, self.anchor_count, -1)
                )
        context = pair_context.float()[:, None].expand(-1, self.anchor_count, -1)
        xyz = self.anchor_xyz[None].expand(batch_size, -1, -1)
        hidden = self.input_mlp(torch.cat((*local_features, context, xyz), dim=-1))
        for block in self.graph_blocks:
            hidden = block(hidden, self.graph_neighbors)
        hidden = self.output_norm(hidden)

        # Auxiliary classifiers train only their own heads. They diagnose
        # dynamics without steering or multiplicatively suppressing pressure.
        auxiliary_hidden = hidden.detach()
        transition_logits = self.transition_head(auxiliary_hidden).reshape(
            batch_size, self.anchor_count, lag_count, 3
        )
        history_gate_logits = self.history_gate_head(auxiliary_hidden)
        raw_coefficients = self.coefficient_head(hidden)
        anchor_alpha = (
            self.max_history_alpha
            * torch.tanh(raw_coefficients)
            * available[:, None, :]
        )
        vertex_alpha_per_lag = self._interpolate(anchor_alpha).permute(0, 2, 1)
        raw_delta = (
            vertex_alpha_per_lag * (history - current[:, None])
        ).sum(dim=1)
        bounded_delta = self.max_logit_delta * torch.tanh(
            raw_delta / self.max_logit_delta
        )
        fused_logits = current + bounded_delta
        return {
            "pred_logits": fused_logits,
            "pred_tactile": torch.sigmoid(fused_logits),
            "base_pred_logits": current,
            "base_pred_tactile": torch.sigmoid(current),
            "bounded_logit_delta": bounded_delta,
            "vertex_history_alpha": vertex_alpha_per_lag.sum(dim=1),
            "vertex_history_alpha_per_lag": vertex_alpha_per_lag,
            "anchor_history_probability": torch.sigmoid(history_gate_logits),
            "anchor_history_gate_logits": history_gate_logits,
            "anchor_transition_logits": transition_logits,
            "anchor_signed_coefficients": anchor_alpha,
            "anchor_raw_coefficients": raw_coefficients,
            "anchor_local_indices": self.anchor_local_indices,
        }

    def forward(
        self,
        current_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        pair_context: torch.Tensor,
        history_available: torch.Tensor | None = None,
        history_quality: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if current_logits.ndim != 2:
            raise ValueError("Temporal current logits must be [B,V]")
        if current_logits.shape[1] != len(self.palm_vertex_indices):
            raise ValueError("Temporal logits do not use the cached palm vertex definition")
        if pair_context.shape != (current_logits.shape[0], 4):
            raise ValueError("pair_context must be [B,4]")
        if self.architecture == "signed_additive":
            if previous_logits.ndim == 2 and len(self.history_lags) == 1:
                previous_logits = previous_logits[:, None]
            return self._signed_additive_forward(
                current_logits,
                previous_logits,
                pair_context,
                history_available,
                history_quality,
            )
        if current_logits.shape != previous_logits.shape:
            raise ValueError(
                "Legacy temporal current/previous logits must be matching [B,V] tensors"
            )
        current_anchor = current_logits[:, self.anchor_local_indices].float()
        previous_anchor = previous_logits[:, self.anchor_local_indices].float()
        current_probability = torch.sigmoid(current_anchor)
        previous_probability = torch.sigmoid(previous_anchor)
        local = torch.stack(
            (
                current_anchor,
                previous_anchor,
                previous_anchor - current_anchor,
                (previous_anchor - current_anchor).abs(),
                current_probability,
                previous_probability,
            ),
            dim=-1,
        )
        context = pair_context.float()[:, None].expand(-1, self.anchor_count, -1)
        xyz = self.anchor_xyz[None].expand(current_logits.shape[0], -1, -1)
        hidden = self.input_mlp(torch.cat((local, context, xyz), dim=-1))
        for block in self.graph_blocks:
            hidden = block(hidden, self.graph_neighbors)
        hidden = self.output_norm(hidden)
        transition_logits = self.transition_head(hidden)
        stable_probability = transition_logits.softmax(dim=-1)[..., 0]
        history_gate_logits = self.history_gate_head(hidden).squeeze(-1)
        history_probability = torch.sigmoid(history_gate_logits)
        signed_global_gate = torch.tanh(self.global_rezero_gate)
        anchor_alpha = (
            self.max_history_alpha
            * signed_global_gate
            * stable_probability
            * history_probability
        )
        vertex_alpha = self._interpolate(anchor_alpha)
        raw_delta = vertex_alpha * (previous_logits.float() - current_logits.float())
        bounded_delta = self.max_logit_delta * torch.tanh(
            raw_delta / self.max_logit_delta
        )
        fused_logits = current_logits.float() + bounded_delta
        return {
            "pred_logits": fused_logits,
            "pred_tactile": torch.sigmoid(fused_logits),
            "base_pred_logits": current_logits.float(),
            "base_pred_tactile": torch.sigmoid(current_logits.float()),
            "bounded_logit_delta": bounded_delta,
            "vertex_history_alpha": vertex_alpha,
            "anchor_history_probability": history_probability,
            "anchor_history_gate_logits": history_gate_logits,
            "anchor_transition_logits": transition_logits,
            "anchor_local_indices": self.anchor_local_indices,
            "effective_global_gate": signed_global_gate,
        }


class TemporalActionSelectorV2(nn.Module):
    """Diagnostic canonical-anchor classifier for down/hold/up actions.

    This module deliberately has no pressure decoder.  It learns whether the
    frozen RGB prediction at each anchor is too high, already close enough, or
    too low, while history and its quality metadata are only evidence.
    """

    ACTION_NAMES = ("down", "hold", "up")

    def __init__(
        self,
        palm_vertex_indices: Sequence[int],
        *,
        anchor_count: int = 512,
        anchor_neighbors: int = 4,
        graph_neighbors: int = 4,
        hidden_dim: int = 96,
        graph_layers: int = 2,
        dropout: float = 0.05,
        history_lags: Sequence[int] = (1, 2),
        use_per_lag_quality: bool = True,
        nominal_fps: float = 30.0,
        dino_grid_channels: int = 0,
        dino_grid_size: Sequence[int] = (16, 12),
        dino_input_resolution: Sequence[int] = (256, 192),
        dino_attention_heads: int = 4,
        dino_alignment_mode: str = "aligned",
        dino_shuffle_seed: int = 521,
    ):
        super().__init__()
        history_lags = tuple(int(value) for value in history_lags)
        if (
            not history_lags
            or len(set(history_lags)) != len(history_lags)
            or any(value <= 0 for value in history_lags)
        ):
            raise ValueError("history_lags must be unique positive integers")
        if float(nominal_fps) <= 0.0:
            raise ValueError("nominal_fps must be positive")
        palm_vertex_indices = torch.as_tensor(palm_vertex_indices, dtype=torch.long)
        vertices, _ = _canonical_mesh_assets()
        tactile_dim = int(vertices.shape[0])
        anchor_vertices, vertex_anchor_indices, vertex_anchor_weights, valid_mask = (
            _canonical_rbf_assignment(
                tactile_dim=tactile_dim,
                anchor_count=int(anchor_count),
                neighbor_count=int(anchor_neighbors),
            )
        )
        if not torch.equal(torch.nonzero(valid_mask).flatten(), palm_vertex_indices):
            raise RuntimeError("Temporal cache palm vertices differ from canonical palm mask")
        full_to_palm = torch.full((tactile_dim,), -1, dtype=torch.long)
        full_to_palm[palm_vertex_indices] = torch.arange(len(palm_vertex_indices))
        anchor_local = full_to_palm[anchor_vertices]
        if bool((anchor_local < 0).any()):
            raise RuntimeError("Canonical temporal anchor lies outside the valid palm")
        anchor_xyz = vertices[anchor_vertices]
        anchor_xyz = (anchor_xyz - anchor_xyz.mean(0)) / anchor_xyz.std(0).clamp_min(1e-6)
        distances = torch.cdist(anchor_xyz, anchor_xyz)
        graph_index = torch.topk(
            distances,
            k=min(int(graph_neighbors) + 1, int(anchor_count)),
            largest=False,
        ).indices[:, 1:]
        self.register_buffer("palm_vertex_indices", palm_vertex_indices)
        self.register_buffer("anchor_local_indices", anchor_local)
        self.register_buffer("vertex_anchor_indices", vertex_anchor_indices[palm_vertex_indices])
        self.register_buffer("vertex_anchor_weights", vertex_anchor_weights[palm_vertex_indices])
        self.register_buffer("anchor_xyz", anchor_xyz)
        self.register_buffer("graph_neighbors", graph_index)
        self.register_buffer("class_prior", torch.full((3,), 1.0 / 3.0))
        self.anchor_count = int(anchor_count)
        self.history_lags = history_lags
        self.use_per_lag_quality = bool(use_per_lag_quality)
        self.nominal_fps = float(nominal_fps)
        self.dino_grid_channels = int(dino_grid_channels)
        self.uses_dino_history = self.dino_grid_channels > 0
        self.dino_grid_size = tuple(int(value) for value in dino_grid_size)
        self.dino_input_resolution = tuple(int(value) for value in dino_input_resolution)
        self.dino_attention_heads = int(dino_attention_heads)
        self.dino_alignment_mode = str(dino_alignment_mode).strip().lower()
        self.dino_shuffle_seed = int(dino_shuffle_seed)
        if self.dino_alignment_mode not in {"aligned", "unwarped"}:
            raise ValueError("dino_alignment_mode must be aligned or unwarped")
        if self.uses_dino_history:
            if len(self.dino_grid_size) != 2 or min(self.dino_grid_size) <= 0:
                raise ValueError("dino_grid_size must contain two positive values")
            if len(self.dino_input_resolution) != 2 or min(self.dino_input_resolution) <= 0:
                raise ValueError("dino_input_resolution must contain two positive values")
            expected_grid = tuple(value // 16 for value in self.dino_input_resolution)
            if expected_grid != self.dino_grid_size:
                raise ValueError(
                    "DINO grid must match the /16 input patch layout: "
                    f"grid={self.dino_grid_size}, input={self.dino_input_resolution}"
                )
            if hidden_dim % self.dino_attention_heads:
                raise ValueError("hidden_dim must be divisible by dino_attention_heads")
        # Keep quality/no-quality controls parameter matched. The control gets
        # identically shaped zero quality channels instead of a smaller MLP.
        quality_dim = PER_LAG_QUALITY_DIM * len(history_lags)
        input_dim = 2 + 6 * len(history_lags) + quality_dim + 3
        self.input_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_blocks = nn.ModuleList(
            AnchorGraphBlock(hidden_dim, dropout) for _ in range(int(graph_layers))
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        # Preserve the legacy selector's construction order so a fixed seed
        # initializes every pre-existing parameter identically when DINO
        # evidence is enabled behind its zero gate.
        self.action_head = nn.Linear(hidden_dim, len(self.ACTION_NAMES))
        if self.uses_dino_history:
            self.dino_current_projection = nn.Sequential(
                nn.LayerNorm(self.dino_grid_channels),
                nn.Linear(self.dino_grid_channels, hidden_dim),
            )
            self.dino_motion_projection = nn.Sequential(
                nn.LayerNorm(2 * self.dino_grid_channels + 4),
                nn.Linear(2 * self.dino_grid_channels + 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.dino_lag_embedding = nn.Embedding(len(self.history_lags), hidden_dim)
            self.dino_anchor_query = nn.Linear(3, hidden_dim, bias=False)
            self.dino_query_norm = nn.LayerNorm(hidden_dim)
            self.dino_token_norm = nn.LayerNorm(hidden_dim)
            self.dino_cross_attention = nn.MultiheadAttention(
                hidden_dim,
                self.dino_attention_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.dino_attention_norm = nn.LayerNorm(hidden_dim)
            self.dino_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.dino_ffn_norm = nn.LayerNorm(hidden_dim)
            self.dino_output = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.dino_rezero_gate = nn.Parameter(torch.zeros(()))
            positional = self._fixed_2d_sincos(
                self.dino_grid_size[0], self.dino_grid_size[1], hidden_dim
            )
            self.register_buffer("dino_position_encoding", positional, persistent=True)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.dino_shuffle_seed)
            self.register_buffer(
                "dino_spatial_permutation",
                torch.randperm(self.dino_grid_size[0] * self.dino_grid_size[1], generator=generator),
                persistent=True,
            )

    @staticmethod
    def _fixed_2d_sincos(height: int, width: int, dim: int) -> torch.Tensor:
        if dim % 4:
            raise ValueError("DINO token hidden dimension must be divisible by four")
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        frequency = torch.arange(dim // 4, dtype=torch.float32)
        frequency = 1.0 / (10000.0 ** (frequency / max(dim // 4, 1)))
        x = x.reshape(-1, 1) * frequency[None]
        y = y.reshape(-1, 1) * frequency[None]
        return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

    def _warp_history_grid(
        self,
        history_grids: torch.Tensor,
        current_to_history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, lag_count, channels, grid_height, grid_width = history_grids.shape
        if (grid_height, grid_width) != self.dino_grid_size:
            raise ValueError(
                f"DINO history grid is {grid_height}x{grid_width}, expected "
                f"{self.dino_grid_size}"
            )
        if current_to_history.shape != (batch_size, lag_count, 3, 3):
            raise ValueError(
                "history_crop_transform must be [B,L,3,3], got "
                f"{tuple(current_to_history.shape)}"
            )
        input_height, input_width = self.dino_input_resolution
        # grid_sample's autocast wrapper cannot safely prioritize every cached
        # feature dtype against the geometric grid. Keep coordinate construction
        # and interpolation in one FP32 numerical island, then return to the
        # cache/model dtype. This also prevents BF16 from rounding away small
        # differences in crop transforms before interpolation.
        with torch.autocast(device_type=history_grids.device.type, enabled=False):
            center_y = (
                torch.arange(
                    grid_height, device=history_grids.device, dtype=torch.float32
                )
                + 0.5
            ) * (input_height / grid_height)
            center_x = (
                torch.arange(
                    grid_width, device=history_grids.device, dtype=torch.float32
                )
                + 0.5
            ) * (input_width / grid_width)
            y, x = torch.meshgrid(center_y, center_x, indexing="ij")
            points = torch.stack((x, y, torch.ones_like(x)), dim=-1).reshape(-1, 3)
            mapped = torch.einsum(
                "blij,nj->blni", current_to_history.float(), points
            )
            denominator = mapped[..., 2].clamp_min(1e-8)
            mapped_x = mapped[..., 0] / denominator
            mapped_y = mapped[..., 1] / denominator
            history_x = mapped_x / (input_width / grid_width) - 0.5
            history_y = mapped_y / (input_height / grid_height) - 0.5
            norm_x = 2.0 * history_x / max(grid_width - 1, 1) - 1.0
            norm_y = 2.0 * history_y / max(grid_height - 1, 1) - 1.0
            sample_grid = torch.stack((norm_x, norm_y), dim=-1).reshape(
                batch_size * lag_count, grid_height, grid_width, 2
            )
            flat_history = history_grids.reshape(
                batch_size * lag_count, channels, grid_height, grid_width
            )
            warped = F.grid_sample(
                flat_history.float(),
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            valid = (
                (norm_x >= -1.0)
                & (norm_x <= 1.0)
                & (norm_y >= -1.0)
                & (norm_y <= 1.0)
            ).reshape(batch_size, lag_count, 1, grid_height, grid_width)
        warped = warped.to(dtype=history_grids.dtype).reshape(
            batch_size, lag_count, channels, grid_height, grid_width
        )
        return warped, valid

    def _dino_motion_residual(
        self,
        hidden: torch.Tensor,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_available: torch.Tensor,
        history_crop_transform: torch.Tensor,
        *,
        evidence_current_grid: torch.Tensor | None,
        control: str,
        residual_mode: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        batch_size, lag_count = history_available.shape
        expected = (batch_size, self.dino_grid_channels, *self.dino_grid_size)
        if tuple(current_grid.shape) != expected:
            raise ValueError(f"current_grid must be {expected}, got {tuple(current_grid.shape)}")
        if tuple(history_grids.shape) != (
            batch_size,
            lag_count,
            self.dino_grid_channels,
            *self.dino_grid_size,
        ):
            raise ValueError("history_grids do not match the configured DINO grid")
        motion_current = current_grid if evidence_current_grid is None else evidence_current_grid
        if tuple(motion_current.shape) != expected:
            raise ValueError("evidence_current_grid does not match current_grid")
        control = str(control).strip().lower()
        if control not in {"aligned", "unwarped", "spatial_shuffle"}:
            raise ValueError(f"Unsupported DINO history control: {control!r}")
        residual_mode = str(residual_mode).strip().lower()
        if residual_mode not in {"full", "gate_zero", "zero_motion"}:
            raise ValueError(
                f"Unsupported DINO residual mode: {residual_mode!r}"
            )
        if control == "unwarped":
            aligned_history = history_grids
            spatial_valid = torch.ones(
                (batch_size, lag_count, 1, *self.dino_grid_size),
                device=history_grids.device,
                dtype=torch.bool,
            )
        else:
            aligned_history, spatial_valid = self._warp_history_grid(
                history_grids, history_crop_transform
            )
        available = history_available[:, :, None, None, None] > 0.5
        valid = spatial_valid & available
        current_for_motion = motion_current[:, None].expand_as(aligned_history)
        if residual_mode == "zero_motion":
            # Preserve lag availability and crop-support metadata while
            # removing all historical image content and feature motion. The
            # remaining branch measures current-frame DINO capacity.
            aligned_history = current_for_motion
        else:
            aligned_history = torch.where(valid, aligned_history, current_for_motion)
        difference = aligned_history - current_for_motion
        difference = difference * valid.to(difference.dtype)
        if residual_mode == "gate_zero":
            residual = torch.zeros_like(hidden)
            diagnostics = {
                "dino_effective_gate": hidden.new_zeros(()),
                "dino_residual_rms": hidden.new_zeros(()),
                "dino_valid_token_fraction": valid.float().mean(),
                "dino_motion_rms": difference.float().square().mean().sqrt(),
            }
            return hidden, diagnostics, residual
        current_tokens = current_grid.flatten(2).transpose(1, 2)
        # Cached DINO grids are stored in FP16, while standalone evaluation
        # keeps selector parameters in FP32 and does not run under Lightning's
        # mixed-precision autocast. Match each projection's parameter dtype at
        # the module boundary so train, validation, and independent evaluation
        # share the same input contract.
        current_tokens = current_tokens.to(
            dtype=self.dino_current_projection[0].weight.dtype
        )
        current_tokens = self.dino_current_projection(current_tokens)
        current_tokens = current_tokens[:, None].expand(-1, lag_count, -1, -1)
        difference_tokens = difference.flatten(3).permute(0, 1, 3, 2)
        dot = (aligned_history * current_for_motion).sum(dim=2, keepdim=True)
        denominator = (
            aligned_history.square().sum(dim=2, keepdim=True).clamp_min(1e-12).sqrt()
            * current_for_motion.square().sum(dim=2, keepdim=True).clamp_min(1e-12).sqrt()
        )
        cosine = (dot / denominator).flatten(3).permute(0, 1, 3, 2)
        valid_tokens = valid.flatten(3).permute(0, 1, 3, 2).to(difference_tokens)
        available_tokens = history_available[:, :, None, None].expand(
            -1, -1, difference_tokens.shape[2], 1
        ).to(difference_tokens)
        maximum_lag = float(max(self.history_lags))
        lag_value = torch.as_tensor(
            self.history_lags, device=difference_tokens.device, dtype=difference_tokens.dtype
        ) / maximum_lag
        lag_value = lag_value[None, :, None, None].expand_as(available_tokens)
        motion_input = torch.cat(
            (
                difference_tokens,
                difference_tokens.abs(),
                cosine,
                valid_tokens,
                available_tokens,
                lag_value,
            ),
            dim=-1,
        )
        motion_input = motion_input.to(
            dtype=self.dino_motion_projection[0].weight.dtype
        )
        motion_tokens = self.dino_motion_projection(motion_input)
        position = self.dino_position_encoding.to(motion_tokens)[None, None]
        lag_embedding = self.dino_lag_embedding.weight.to(motion_tokens)[None, :, None]
        tokens = current_tokens + motion_tokens + lag_embedding
        if control == "spatial_shuffle":
            tokens = tokens[:, :, self.dino_spatial_permutation]
        tokens = tokens + position
        tokens = self.dino_token_norm(tokens.flatten(1, 2))
        query = self.dino_query_norm(
            hidden + self.dino_anchor_query(self.anchor_xyz.to(hidden))[None]
        )
        attended, _ = self.dino_cross_attention(
            query, tokens, tokens, need_weights=False
        )
        attended = self.dino_attention_norm(query + attended)
        attended = self.dino_ffn_norm(attended + self.dino_ffn(attended))
        gate = torch.tanh(self.dino_rezero_gate)
        residual = gate * self.dino_output(attended)
        diagnostics = {
            "dino_effective_gate": gate,
            "dino_residual_rms": residual.float().square().mean().sqrt(),
            "dino_valid_token_fraction": valid.float().mean(),
            "dino_motion_rms": difference.float().square().mean().sqrt(),
        }
        # ``hidden`` has already passed ``output_norm``. A second normalization
        # here would change the legacy selector even while the ReZero gate is
        # exactly zero, invalidating the parameter-matched initialization.
        return hidden + residual, diagnostics, residual

    def set_class_prior(self, counts_or_prior: torch.Tensor) -> None:
        values = torch.as_tensor(
            counts_or_prior, device=self.class_prior.device, dtype=self.class_prior.dtype
        )
        if values.shape != (3,) or bool((values < 0).any()) or not torch.isfinite(values).all():
            raise ValueError("Selector class prior must be one finite non-negative 3-vector")
        total = values.sum()
        if float(total) <= 0.0:
            raise ValueError("Selector class prior must have positive mass")
        self.class_prior.copy_((values / total).clamp_min(1e-8))

    def forward(
        self,
        current_logits: torch.Tensor,
        history_logits: torch.Tensor,
        history_available: torch.Tensor,
        history_quality: torch.Tensor | None = None,
        *,
        apply_prior_correction: bool = True,
        current_grid: torch.Tensor | None = None,
        history_grids: torch.Tensor | None = None,
        history_crop_transform: torch.Tensor | None = None,
        evidence_current_grid: torch.Tensor | None = None,
        dino_control: str | None = None,
        dino_residual_mode: str = "full",
        return_dino_residual: bool = False,
    ) -> dict[str, torch.Tensor]:
        if current_logits.ndim != 2 or current_logits.shape[1] != len(
            self.palm_vertex_indices
        ):
            raise ValueError("Selector current logits must be cached palm logits [B,V]")
        batch_size, vertex_count = current_logits.shape
        lag_count = len(self.history_lags)
        if history_logits.shape != (batch_size, lag_count, vertex_count):
            raise ValueError(
                f"Selector history logits must be [B,{lag_count},V], "
                f"got {tuple(history_logits.shape)}"
            )
        if history_available.shape != (batch_size, lag_count):
            raise ValueError(
                f"Selector history availability must be [B,{lag_count}], "
                f"got {tuple(history_available.shape)}"
            )
        available = history_available.float().clamp(0.0, 1.0)
        if self.use_per_lag_quality:
            expected = (batch_size, lag_count, PER_LAG_QUALITY_DIM)
            if history_quality is None or tuple(history_quality.shape) != expected:
                actual = None if history_quality is None else tuple(history_quality.shape)
                raise ValueError(
                    f"Selector history quality must be {expected}, got {actual}"
                )
            quality = history_quality.float() * available[:, :, None]
        else:
            quality = current_logits.new_zeros(
                (batch_size, lag_count, PER_LAG_QUALITY_DIM), dtype=torch.float32
            )
        current = current_logits.float()
        history = history_logits.float()
        history = torch.where(available[:, :, None] > 0.5, history, current[:, None])
        current_anchor = current[:, self.anchor_local_indices]
        history_anchor = history[:, :, self.anchor_local_indices].transpose(1, 2)
        difference = history_anchor - current_anchor[:, :, None]
        features = [
            current_anchor[:, :, None],
            torch.sigmoid(current_anchor)[:, :, None],
        ]
        maximum_lag = float(max(self.history_lags))
        for column, lag in enumerate(self.history_lags):
            lag_available = available[:, None, column].expand(-1, self.anchor_count)
            features.extend(
                (
                    history_anchor[:, :, column : column + 1],
                    difference[:, :, column : column + 1],
                    difference[:, :, column : column + 1].abs(),
                    torch.sigmoid(history_anchor[:, :, column : column + 1]),
                    lag_available[:, :, None],
                    torch.full_like(
                        history_anchor[:, :, column : column + 1],
                        float(lag) / maximum_lag,
                    ),
                )
            )
            features.append(
                quality[:, None, column, :].expand(-1, self.anchor_count, -1)
            )
        xyz = self.anchor_xyz[None].expand(batch_size, -1, -1)
        hidden = self.input_mlp(torch.cat((*features, xyz), dim=-1))
        for block in self.graph_blocks:
            hidden = block(hidden, self.graph_neighbors)
        hidden = self.output_norm(hidden)
        dino_diagnostics: dict[str, torch.Tensor] = {}
        if self.uses_dino_history:
            if current_grid is None or history_grids is None or history_crop_transform is None:
                raise KeyError(
                    "DINO-history selector requires current_grid, history_grids, and "
                    "history_crop_transform"
                )
            hidden, dino_diagnostics, dino_residual = self._dino_motion_residual(
                hidden,
                current_grid,
                history_grids,
                available,
                history_crop_transform,
                evidence_current_grid=evidence_current_grid,
                control=dino_control or self.dino_alignment_mode,
                residual_mode=dino_residual_mode,
            )
        balanced_logits = self.action_head(hidden)
        if apply_prior_correction:
            action_logits = balanced_logits + self.class_prior.clamp_min(1e-8).log()
        else:
            action_logits = balanced_logits
        output = {
            "balanced_action_logits": balanced_logits,
            "action_logits": action_logits,
            "action_probability": action_logits.softmax(dim=-1),
            "anchor_local_indices": self.anchor_local_indices,
            **dino_diagnostics,
        }
        if self.uses_dino_history and return_dino_residual:
            output["dino_hidden_residual"] = dino_residual
        return output


def temporal_action_targets(
    current_logits: torch.Tensor,
    tactile_signal: torch.Tensor,
    anchor_local_indices: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Return down=0, hold=1, up=2 labels around a probability dead zone."""

    if float(margin) < 0.0:
        raise ValueError("Selector action margin must be non-negative")
    current = torch.sigmoid(current_logits.float())[:, anchor_local_indices]
    target = tactile_signal.float()[:, anchor_local_indices]
    error = target - current
    labels = torch.full_like(error, 1, dtype=torch.long)
    labels[error < -float(margin)] = 0
    labels[error > float(margin)] = 2
    return labels


def pair_context(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        (
            batch["time_gap"].float() / 0.05,
            batch["bbox_iou"].float(),
            batch["bbox_center_jump"].float(),
            batch["bbox_abs_log_area_ratio"].float() / math.log(2.0),
        ),
        dim=1,
    )


def history_quality_context(
    batch: Mapping[str, torch.Tensor],
    history_lags: Sequence[int],
    *,
    prefix: str = "",
    availability: torch.Tensor | None = None,
    nominal_fps: float = 30.0,
) -> torch.Tensor:
    """Build bounded, fixed-semantics quality features for every history lag.

    The five channels are absolute elapsed time, deviation from the nominal
    lag time, cumulative minimum bbox IoU, cumulative maximum center jump, and
    cumulative maximum area change.  Availability is already an explicit model
    input; unavailable rows are zeroed here so stale neutral metadata cannot be
    interpreted as evidence.
    """

    lags = tuple(int(value) for value in history_lags)
    if not lags or any(value <= 0 for value in lags):
        raise ValueError("history_lags must contain positive integers")
    if float(nominal_fps) <= 0.0:
        raise ValueError("nominal_fps must be positive")
    names = (
        "history_time_gap",
        "history_min_bbox_iou",
        "history_max_bbox_center_jump",
        "history_max_bbox_abs_log_area_ratio",
    )
    values = []
    for name in names:
        key = f"{prefix}{name}"
        if key not in batch:
            raise KeyError(f"Batch is missing per-lag temporal metadata {key!r}")
        value = batch[key].float()
        if value.ndim != 2 or value.shape[1] != len(lags):
            raise ValueError(
                f"{key} must be [B,{len(lags)}], got {tuple(value.shape)}"
            )
        values.append(value)
    elapsed, minimum_iou, center_jump, area_change = values
    lag_tensor = elapsed.new_tensor(lags)[None]
    nominal_time = lag_tensor / float(nominal_fps)
    absolute_time = torch.log1p(elapsed.clamp_min(0.0) / 0.05) / math.log(2.0)
    time_ratio = torch.log(
        (elapsed.clamp_min(0.0) + 1e-6) / (nominal_time + 1e-6)
    ).clamp(-2.0, 2.0) / 2.0
    quality = torch.stack(
        (
            absolute_time.clamp(0.0, 8.0) / 8.0,
            time_ratio,
            minimum_iou.clamp(0.0, 1.0),
            center_jump.clamp(0.0, 2.0) / 2.0,
            (area_change.clamp_min(0.0) / math.log(2.0)).clamp(0.0, 4.0) / 4.0,
        ),
        dim=-1,
    )
    if availability is None:
        availability_key = f"{prefix}history_available"
        if availability_key not in batch:
            raise KeyError(
                f"Batch is missing temporal availability {availability_key!r}"
            )
        availability = batch[availability_key]
    if availability.shape != elapsed.shape:
        raise ValueError(
            f"history availability must be {tuple(elapsed.shape)}, "
            f"got {tuple(availability.shape)}"
        )
    quality = quality * availability.to(quality).clamp(0.0, 1.0)[..., None]
    if not torch.isfinite(quality).all():
        raise FloatingPointError("Per-lag temporal quality contains non-finite values")
    return quality


def temporal_checkpoint_payload(
    model: QueryAwareTemporalResidual,
    *,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    base_checkpoint: str,
    epoch: int,
    global_step: int,
    monitor: str,
    score: float,
) -> dict[str, Any]:
    return {
        "format": TEMPORAL_MODEL_FORMAT,
        "model_config": dict(model_config),
        "data_config": dict(data_config),
        "base_checkpoint": str(Path(base_checkpoint).expanduser().resolve(strict=True)),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "epoch": int(epoch),
        "global_step": int(global_step),
        "monitor": str(monitor),
        "score": float(score),
    }


def temporal_selector_checkpoint_payload(
    model: TemporalActionSelectorV2,
    *,
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    base_checkpoint: str,
    action_margin: float,
    epoch: int,
    global_step: int,
    monitor: str,
    score: float,
) -> dict[str, Any]:
    return {
        "format": TEMPORAL_SELECTOR_FORMAT,
        "model_config": dict(model_config),
        "data_config": dict(data_config),
        "base_checkpoint": str(Path(base_checkpoint).expanduser().resolve(strict=True)),
        "base_checkpoint_sha256": sha256_file(base_checkpoint),
        "action_margin": float(action_margin),
        "class_prior": model.class_prior.detach().cpu().tolist(),
        "action_names": list(model.ACTION_NAMES),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "monitor": str(monitor),
        "score": float(score),
    }
