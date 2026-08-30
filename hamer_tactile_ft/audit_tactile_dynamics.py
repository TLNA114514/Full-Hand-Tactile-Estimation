#!/usr/bin/env python3
"""Audit whether current sequence-HDF5 data can support tactile dynamics.

The audit is deliberately model-free. It first scans authoritative query
manifests without decoding RGB, then reads a deterministic subset of pressure
pairs. Adjacent mode measures persistence, transport, and loading/release;
bilateral mode adds exact lags, per-hand marginals, anonymous association
provenance, and synchronized opposite-hand counterfactual histories.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import heapq
import json
import math
import os
import sys
import tempfile
import time
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactile_input_priors.hdf5_manifest import (  # noqa: E402
    open_readonly,
    sha256_file,
    write_json_atomic,
)
from tactile_input_priors.resolve_depth_manifests import (  # noqa: E402
    resolve_manifests,
)


AUDIT_SCHEMA = "tactile_dynamics_audit_v1"
BILATERAL_AUDIT_SCHEMA = "bilateral_tactile_dynamics_audit_v2"
_H5_PATH_CACHE: dict[tuple[str, str, str, str], Path] = {}
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "TACTILE_FLOW_AUDIT_ROOT",
        "/home/ma-user/work/cfzhao/input_prior_full/tactile_flow_audits",
    )
)
FRAME_GAP_BINS = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-8", 4, 8),
    ("9-15", 9, 15),
    ("16-30", 16, 30),
    ("31+", 31, None),
)
TIME_GAP_BINS = (
    ("<=0.04", 0.0, 0.04),
    ("0.04-0.10", 0.04, 0.10),
    ("0.10-0.25", 0.10, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    (">0.50", 0.50, None),
)
PRESSURE_BINS = (
    ("<0.005", -math.inf, 0.005),
    ("0.005-0.05", 0.005, 0.05),
    ("0.05-0.2", 0.05, 0.2),
    ("0.2-0.5", 0.2, 0.5),
    ("0.5-0.7", 0.5, 0.7),
    (">=0.7", 0.7, math.inf),
)
VOLUME_BINS = (
    ("<10", -math.inf, 10.0),
    ("10-50", 10.0, 50.0),
    ("50-150", 50.0, 150.0),
    ("150-300", 150.0, 300.0),
    (">=300", 300.0, math.inf),
)


try:
    import orjson
except ImportError:  # pragma: no cover - optional speedup
    orjson = None


def _loads(value: bytes) -> Mapping[str, Any]:
    decoded = orjson.loads(value) if orjson is not None else json.loads(value)
    if not isinstance(decoded, dict):
        raise TypeError("manifest row must be a JSON object")
    return decoded


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _frame_gap_bin(value: int) -> str:
    for name, lower, upper in FRAME_GAP_BINS:
        if value >= lower and (upper is None or value <= upper):
            return name
    return "nonpositive"


def _time_gap_bin(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    if value <= 0.0:
        return "nonpositive"
    for name, lower, upper in TIME_GAP_BINS:
        if value > lower and (upper is None or value <= upper):
            return name
    return "unknown"


def _numeric_bin(value: float, bins: Sequence[tuple[str, float, float]]) -> str:
    for name, lower, upper in bins:
        if value >= lower and value < upper:
            return name
    return "nonfinite" if not math.isfinite(value) else "out_of_range"


def _bbox_metrics(previous: Sequence[float], current: Sequence[float]) -> tuple[float, float, float]:
    a = np.asarray(previous, dtype=np.float64)
    b = np.asarray(current, dtype=np.float64)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan"), float("nan"), float("nan")
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if min(aw, ah, bw, bh) <= 1.0:
        return float("nan"), float("nan"), float("nan")
    intersection_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = intersection_w * intersection_h
    union = aw * ah + bw * bh - intersection
    iou = intersection / union if union > 0.0 else 0.0
    ac = np.asarray(((a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5))
    bc = np.asarray(((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5))
    center_jump = float(np.linalg.norm(ac - bc) / max(math.sqrt(aw * ah), 1.0))
    log_area_ratio = abs(math.log(max(bw * bh, 1.0) / max(aw * ah, 1.0)))
    return float(iou), center_jump, log_area_ratio


@dataclasses.dataclass(frozen=True, slots=True)
class SampleRecord:
    manifest_index: int
    line_number: int
    sample_uid: str
    dataset: str
    split: str
    sequence_key: str
    query_alias: str
    is_right: int
    frame_idx: int
    source_frame_idx: Optional[int]
    timestamp: Optional[float]
    bbox_xyxy: tuple[float, float, float, float]
    bbox_score: Optional[float]
    bbox_source_schema: str
    bbox_raw_track_id: Optional[str]
    bbox_association_id: Optional[str]
    bbox_association_confidence: Optional[str]
    bbox_association_policy: Optional[str]
    pressure_source_key: str
    h5_path: str
    query_row: int
    max_pressure: float
    target_volume: float
    target_active_count: int

    @property
    def group_key(self) -> tuple[str, str, str, str, int]:
        return self.dataset, self.split, self.sequence_key, self.query_alias, self.is_right

    @property
    def side_name(self) -> str:
        return "right" if self.is_right else "left"

    @property
    def temporal_frame_idx(self) -> int:
        return self.source_frame_idx if self.source_frame_idx is not None else self.frame_idx


@dataclasses.dataclass(frozen=True, slots=True)
class PairRecord:
    previous: SampleRecord
    current: SampleRecord
    frame_gap: int
    time_gap: Optional[float]
    bbox_iou: float
    bbox_center_jump: float
    bbox_abs_log_area_ratio: float
    bbox_stable: bool
    temporal_eligible: bool
    eligible: bool
    transition: str

    @property
    def frame_gap_bin(self) -> str:
        return _frame_gap_bin(self.frame_gap)

    @property
    def time_gap_bin(self) -> str:
        return _time_gap_bin(self.time_gap)

    @property
    def stratum(self) -> tuple[str, str, str, str]:
        return (
            self.current.dataset,
            self.current.split,
            self.time_gap_bin,
            self.transition,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class BilateralPairRecord:
    self_pair: PairRecord
    requested_lag: int
    contralateral_previous: Optional[SampleRecord]

    @property
    def stratum(self) -> tuple[str, str, str, str, str, str, str, str]:
        current = self.self_pair.current
        return (
            current.dataset,
            current.split,
            str(self.requested_lag),
            current.side_name,
            self.self_pair.transition,
            _numeric_bin(current.max_pressure, PRESSURE_BINS),
            _numeric_bin(current.target_volume, VOLUME_BINS),
            "contralateral" if self.contralateral_previous is not None else "self_only",
        )


@dataclasses.dataclass(slots=True)
class LoadedBilateralPressureBatch:
    valid_records: list[BilateralPairRecord]
    own_previous_rows: list[np.ndarray]
    current_rows: list[np.ndarray]
    other_positions: list[int]
    other_previous_rows: list[np.ndarray]
    other_current_rows: list[np.ndarray]
    other_pairs: list[PairRecord]
    errors: list[tuple[BilateralPairRecord, Exception]]
    read_seconds: float


@dataclasses.dataclass(slots=True)
class AuditConfig:
    manifests: tuple[Path, ...]
    data_root: Optional[Path]
    output_dir: Path
    max_pressure_pairs: int
    max_frame_gap: int
    max_time_gap: float
    allow_missing_timestamps: bool
    stable_bbox_only: bool
    min_bbox_iou: float
    max_bbox_center_jump: float
    max_bbox_area_ratio: float
    contact_threshold: float
    active_threshold: float
    seed: int
    progress_every: int
    max_open_hdf5: int
    pair_csv_limit: int
    mode: str
    controlled_lags: tuple[int, ...]
    max_bilateral_pressure_pairs: int
    association_iou_ambiguity_margin: float
    association_center_ambiguity_margin: float
    pressure_row_cache_size: int = 512
    pressure_metric_device: str = "cpu"
    pressure_batch_size: int = 1024
    pressure_metric_chunk_size: int = 0
    pressure_metric_dtype: str = "float32"
    pressure_prefetch: bool = True


def _resolve_h5_path(raw: Mapping[str, Any], manifest_path: Path, data_root: Optional[Path]) -> Path:
    absolute = str(raw.get("h5_path") or "").strip()
    relative = str(raw.get("h5_relpath") or "").strip()
    root_value = str(raw.get("data_root") or raw.get("hdf5_root") or "").strip()
    cache_key = (
        absolute,
        relative,
        root_value,
        str(data_root or manifest_path.parent.parent),
    )
    cached = _H5_PATH_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if absolute:
        candidate = Path(absolute).expanduser()
        if candidate.is_file():
            resolved = candidate.resolve()
            _H5_PATH_CACHE[cache_key] = resolved
            return resolved
    if not relative:
        raise ValueError("row has neither a usable h5_path nor h5_relpath")
    roots = []
    if data_root is not None:
        roots.append(data_root)
    if root_value:
        roots.append(Path(str(root_value)).expanduser())
    roots.append(manifest_path.parent.parent)
    existing = []
    seen = set()
    for root in roots:
        root = root.resolve(strict=False)
        candidate = (root / relative).resolve(strict=False)
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        key = os.path.normcase(os.fspath(candidate))
        if key not in seen and candidate.is_file():
            seen.add(key)
            existing.append(candidate)
    if len(existing) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve h5_relpath={relative!r} from manifest "
            f"{manifest_path}; candidates={existing}"
        )
    _H5_PATH_CACHE[cache_key] = existing[0]
    return existing[0]


def _sample_from_raw(
    raw: Mapping[str, Any],
    *,
    manifest_index: int,
    line_number: int,
    manifest_path: Path,
    data_root: Optional[Path],
) -> SampleRecord:
    bbox = raw.get("bbox_xyxy", raw.get("bbox"))
    if bbox is None or len(bbox) != 4:
        raise ValueError("missing bbox_xyxy")
    required = (
        "sample_uid",
        "dataset",
        "split",
        "sequence_key",
        "frame_idx",
        "query_alias",
        "query_row",
        "max_pressure",
        "target_volume",
        "target_active_count",
    )
    missing = [key for key in required if raw.get(key) is None]
    if missing:
        raise ValueError(f"missing required fields {missing}")
    bbox_source = raw.get("bbox_source")
    bbox_source_schema = (
        str(bbox_source.get("schema") or "missing")
        if isinstance(bbox_source, dict)
        else str(raw.get("bbox_source_schema") or bbox_source or "missing")
    )
    bbox_raw_track_id = None
    bbox_association_id = None
    bbox_association_confidence = None
    bbox_association_policy = None
    if isinstance(bbox_source, dict):
        if bbox_source.get("raw_track_id") is not None:
            bbox_raw_track_id = str(bbox_source["raw_track_id"])
        if bbox_source.get("association_id") is not None:
            bbox_association_id = str(bbox_source["association_id"])
        if bbox_source.get("association_confidence") is not None:
            bbox_association_confidence = str(
                bbox_source["association_confidence"]
            ).strip().lower()
        if bbox_source.get("association_policy") is not None:
            bbox_association_policy = str(bbox_source["association_policy"])
    return SampleRecord(
        manifest_index=manifest_index,
        line_number=line_number,
        sample_uid=str(raw["sample_uid"]),
        dataset=str(raw["dataset"]),
        split=str(raw["split"]),
        sequence_key=str(raw["sequence_key"]),
        query_alias=str(raw["query_alias"]).strip().lower(),
        is_right=int(raw.get("is_right", str(raw["query_alias"]).lower() == "right")),
        frame_idx=int(raw["frame_idx"]),
        source_frame_idx=_finite_int(raw.get("source_frame_idx")),
        timestamp=_finite_float(raw.get("timestamp")),
        bbox_xyxy=tuple(float(value) for value in bbox),
        bbox_score=_finite_float(raw.get("bbox_score")),
        bbox_source_schema=bbox_source_schema,
        bbox_raw_track_id=bbox_raw_track_id,
        bbox_association_id=bbox_association_id,
        bbox_association_confidence=bbox_association_confidence,
        bbox_association_policy=bbox_association_policy,
        pressure_source_key=str(raw.get("pressure_source_key") or "missing"),
        h5_path=str(_resolve_h5_path(raw, manifest_path, data_root)),
        query_row=int(raw["query_row"]),
        max_pressure=float(raw["max_pressure"]),
        target_volume=float(raw["target_volume"]),
        target_active_count=int(raw["target_active_count"]),
    )


def iter_samples(config: AuditConfig, *, pass_name: str) -> Iterator[SampleRecord]:
    seen = 0
    for manifest_index, manifest_path in enumerate(config.manifests):
        with manifest_path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = _loads(line)
                    sample = _sample_from_raw(
                        raw,
                        manifest_index=manifest_index,
                        line_number=line_number,
                        manifest_path=manifest_path,
                        data_root=config.data_root,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid temporal audit row at {manifest_path}:{line_number}: {exc}"
                    ) from exc
                seen += 1
                if config.progress_every and seen % config.progress_every == 0:
                    print(f"[{pass_name}] scanned {seen:,} manifest rows", flush=True)
                yield sample


def _pair_from_samples(previous: SampleRecord, current: SampleRecord, config: AuditConfig) -> PairRecord:
    if previous.source_frame_idx is not None and current.source_frame_idx is not None:
        frame_gap = current.source_frame_idx - previous.source_frame_idx
    else:
        frame_gap = current.frame_idx - previous.frame_idx
    time_gap = None
    if previous.timestamp is not None and current.timestamp is not None:
        time_gap = current.timestamp - previous.timestamp
    bbox_iou, center_jump, log_area_ratio = _bbox_metrics(
        previous.bbox_xyxy, current.bbox_xyxy
    )
    bbox_stable = bool(
        math.isfinite(bbox_iou)
        and bbox_iou >= config.min_bbox_iou
        and center_jump <= config.max_bbox_center_jump
        and log_area_ratio <= math.log(config.max_bbox_area_ratio)
    )
    time_ok = (
        config.allow_missing_timestamps
        if time_gap is None
        else 0.0 < time_gap <= config.max_time_gap
    )
    temporal_eligible = bool(
        previous.h5_path == current.h5_path
        and 0 < frame_gap <= config.max_frame_gap
        and time_ok
    )
    eligible = temporal_eligible and (bbox_stable or not config.stable_bbox_only)
    previous_contact = previous.max_pressure >= config.contact_threshold
    current_contact = current.max_pressure >= config.contact_threshold
    transition = (
        ("contact" if previous_contact else "empty")
        + "_to_"
        + ("contact" if current_contact else "empty")
    )
    return PairRecord(
        previous=previous,
        current=current,
        frame_gap=frame_gap,
        time_gap=time_gap,
        bbox_iou=bbox_iou,
        bbox_center_jump=center_jump,
        bbox_abs_log_area_ratio=log_area_ratio,
        bbox_stable=bbox_stable,
        temporal_eligible=temporal_eligible,
        eligible=eligible,
        transition=transition,
    )


def iter_pairs(config: AuditConfig, *, pass_name: str) -> Iterator[PairRecord]:
    previous_by_group: dict[tuple[str, str, str, str, int], SampleRecord] = {}
    for sample in iter_samples(config, pass_name=pass_name):
        previous = previous_by_group.get(sample.group_key)
        if previous is not None:
            yield _pair_from_samples(previous, sample, config)
        previous_by_group[sample.group_key] = sample


class _ControlledPairBuilder:
    def __init__(self, config: AuditConfig):
        self.config = config
        self.max_lag = max(config.controlled_lags)
        self.history_by_track: dict[
            tuple[str, str, str, str, int], dict[int, SampleRecord]
        ] = defaultdict(dict)
        self.history_by_side: dict[
            tuple[str, str, str, int], dict[int, SampleRecord]
        ] = defaultdict(dict)

    @staticmethod
    def _side_key(sample: SampleRecord, is_right: Optional[int] = None):
        return (
            sample.dataset,
            sample.split,
            sample.sequence_key,
            sample.is_right if is_right is None else int(is_right),
        )

    @staticmethod
    def _prune(history: dict[int, SampleRecord], minimum: int) -> None:
        stale = [index for index in history if index < minimum]
        for index in stale:
            del history[index]

    def push(self, current: SampleRecord) -> list[BilateralPairRecord]:
        frame_idx = current.temporal_frame_idx
        own_history = self.history_by_track[current.group_key]
        opposite_history = self.history_by_side[
            self._side_key(current, 1 - current.is_right)
        ]
        result = []
        for lag in self.config.controlled_lags:
            previous = own_history.get(frame_idx - lag)
            if previous is None:
                continue
            pair = _pair_from_samples(previous, current, self.config)
            result.append(
                BilateralPairRecord(
                    self_pair=pair,
                    requested_lag=lag,
                    contralateral_previous=opposite_history.get(frame_idx - lag),
                )
            )
        own_history[frame_idx] = current
        side_history = self.history_by_side[self._side_key(current)]
        side_history[frame_idx] = current
        minimum = frame_idx - self.max_lag
        self._prune(own_history, minimum)
        self._prune(side_history, minimum)
        return result


def iter_bilateral_pairs(
    config: AuditConfig, *, pass_name: str
) -> Iterator[BilateralPairRecord]:
    builder = _ControlledPairBuilder(config)
    for sample in iter_samples(config, pass_name=pass_name):
        yield from builder.push(sample)


def _identifier_purity(
    values: Mapping[tuple[str, str, str, str], Counter]
) -> dict[str, Any]:
    observation_count = sum(sum(counter.values()) for counter in values.values())
    matched_count = sum(max(counter.values(), default=0) for counter in values.values())
    mixed = sum(len(counter) > 1 for counter in values.values())
    return {
        "identifier_count": len(values),
        "observation_count": observation_count,
        "mixed_alias_identifier_count": mixed,
        "weighted_alias_purity": (
            None if observation_count == 0 else matched_count / observation_count
        ),
    }


def scan_bilateral_structure(
    config: AuditConfig,
) -> tuple[dict[str, Any], Counter, Counter]:
    counts: Counter = Counter()
    strata: Counter = Counter()
    builder = _ControlledPairBuilder(config)
    previous_by_group: dict[tuple[str, str, str, str, int], SampleRecord] = {}
    raw_track_aliases: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    association_aliases: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
    previous_frame_by_sequence: dict[tuple[str, str, str], int] = {}
    record_count = 0
    for sample in iter_samples(config, pass_name="bilateral-structure"):
        record_count += 1
        scope = _scope(sample)
        side = sample.side_name
        counts[(scope, "records")] += 1
        counts[(scope, f"side/{side}")] += 1
        counts[(scope, f"bbox_source/{sample.bbox_source_schema}")] += 1
        expected_alias = side
        counts[(scope, "side_alias_consistent")] += sample.query_alias == expected_alias
        if sample.bbox_raw_track_id is not None:
            counts[(scope, "raw_track_id_present")] += 1
            raw_track_aliases[
                (sample.dataset, sample.split, sample.sequence_key, sample.bbox_raw_track_id)
            ][sample.query_alias] += 1
        if sample.bbox_association_id is not None:
            counts[(scope, "association_id_present")] += 1
            association_aliases[
                (
                    sample.dataset,
                    sample.split,
                    sample.sequence_key,
                    sample.bbox_association_id,
                )
            ][sample.query_alias] += 1
        if sample.bbox_association_confidence:
            counts[
                (scope, f"association_confidence/{sample.bbox_association_confidence}")
            ] += 1
        sequence_identity = (sample.dataset, sample.split, sample.sequence_key)
        previous_sequence_frame = previous_frame_by_sequence.get(sequence_identity)
        if (
            previous_sequence_frame is not None
            and sample.temporal_frame_idx < previous_sequence_frame
        ):
            counts[(scope, "sequence_frame_order_regression")] += 1
        previous_frame_by_sequence[sequence_identity] = sample.temporal_frame_idx
        previous = previous_by_group.get(sample.group_key)
        if previous is not None:
            if (
                previous.bbox_raw_track_id is not None
                and sample.bbox_raw_track_id is not None
            ):
                counts[(scope, "raw_track_adjacent_comparable")] += 1
                counts[(scope, "raw_track_adjacent_same")] += (
                    previous.bbox_raw_track_id == sample.bbox_raw_track_id
                )
            if (
                previous.bbox_association_id is not None
                and sample.bbox_association_id is not None
            ):
                counts[(scope, "association_adjacent_comparable")] += 1
                counts[(scope, "association_adjacent_same")] += (
                    previous.bbox_association_id == sample.bbox_association_id
                )
        previous_by_group[sample.group_key] = sample
        for record in builder.push(sample):
            pair = record.self_pair
            lag = record.requested_lag
            counts[(scope, "controlled_pairs")] += 1
            counts[(scope, f"controlled_lag/{lag}")] += 1
            counts[(scope, f"controlled_lag/{lag}/side/{side}")] += 1
            counts[(scope, f"controlled_lag/{lag}/bbox_stable")] += pair.bbox_stable
            counts[(scope, f"controlled_lag/{lag}/temporal_eligible")] += (
                pair.temporal_eligible
            )
            counts[(scope, f"controlled_lag/{lag}/eligible")] += pair.eligible
            counts[(scope, f"controlled_lag/{lag}/contralateral_available")] += (
                record.contralateral_previous is not None
            )
            if pair.eligible:
                counts[
                    (scope, f"controlled_lag/{lag}/eligible_contralateral_available")
                ] += record.contralateral_previous is not None
                strata[record.stratum] += 1
    raw_comparable = _sum_structure_metric(counts, "raw_track_adjacent_comparable")
    association_comparable = _sum_structure_metric(
        counts, "association_adjacent_comparable"
    )
    summary = {
        "record_count": record_count,
        "controlled_pair_count": sum(
            value for (scope, metric), value in counts.items()
            if metric == "controlled_pairs"
        ),
        "eligible_controlled_pair_count": sum(strata.values()),
        "sequence_frame_order_regression_count": sum(
            value
            for (_scope_name, metric), value in counts.items()
            if metric == "sequence_frame_order_regression"
        ),
        "raw_track_identity": _identifier_purity(raw_track_aliases),
        "association_identity": _identifier_purity(association_aliases),
        "raw_track_adjacent_same_fraction": _ratio_or_none(
            _sum_structure_metric(counts, "raw_track_adjacent_same"), raw_comparable
        ),
        "association_adjacent_same_fraction": _ratio_or_none(
            _sum_structure_metric(counts, "association_adjacent_same"),
            association_comparable,
        ),
    }
    return summary, counts, strata


def _bilateral_pair_hash(pair: BilateralPairRecord, seed: int) -> int:
    payload = (
        f"{seed}\0{pair.requested_lag}\0{pair.self_pair.previous.sample_uid}"
        f"\0{pair.self_pair.current.sample_uid}"
    ).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def select_bilateral_pairs(
    config: AuditConfig, strata: Counter
) -> tuple[list[BilateralPairRecord], dict[tuple[str, ...], int]]:
    if config.max_bilateral_pressure_pairs == 0:
        return [], {key: int(value) for key, value in strata.items()}
    quotas = _largest_remainder(strata, config.max_bilateral_pressure_pairs)
    heaps: dict[
        tuple[str, ...], list[tuple[int, str, BilateralPairRecord]]
    ] = defaultdict(list)
    for record in iter_bilateral_pairs(config, pass_name="bilateral-selection"):
        if not record.self_pair.eligible:
            continue
        quota = quotas.get(record.stratum, 0)
        if quota <= 0:
            continue
        score = _bilateral_pair_hash(record, config.seed)
        identity = (
            f"{record.requested_lag}\0{record.self_pair.previous.sample_uid}"
            f"\0{record.self_pair.current.sample_uid}"
        )
        item = (-score, identity, record)
        heap = heaps[record.stratum]
        if len(heap) < quota:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = [item[2] for heap in heaps.values() for item in heap]
    expected = sum(quotas.values())
    if len(selected) != expected:
        raise RuntimeError(
            f"selected {len(selected)} bilateral pairs, expected {expected}"
        )
    selected.sort(
        key=lambda record: (
            record.self_pair.current.h5_path,
            record.self_pair.current.query_row,
            record.requested_lag,
        )
    )
    return selected, quotas


def _scope(sample: SampleRecord) -> str:
    return f"{sample.dataset}/{sample.split}"


def scan_structure(config: AuditConfig) -> tuple[dict[str, Any], Counter]:
    counts: Counter = Counter()
    groups = set()
    sequences = set()
    strata: Counter = Counter()
    previous_by_group: dict[tuple[str, str, str, str, int], SampleRecord] = {}
    for sample in iter_samples(config, pass_name="structure"):
        scope = _scope(sample)
        counts[(scope, "records")] += 1
        counts[(scope, "timestamp_present")] += sample.timestamp is not None
        counts[(scope, "source_frame_present")] += sample.source_frame_idx is not None
        counts[(scope, "bbox_score_present")] += sample.bbox_score is not None
        counts[(scope, f"bbox_source/{sample.bbox_source_schema}")] += 1
        counts[(scope, f"pressure_source/{sample.pressure_source_key}")] += 1
        counts[(scope, f"record_pressure_bin/{_numeric_bin(sample.max_pressure, PRESSURE_BINS)}")] += 1
        counts[(scope, f"record_volume_bin/{_numeric_bin(sample.target_volume, VOLUME_BINS)}")] += 1
        groups.add(sample.group_key)
        sequences.add((sample.dataset, sample.split, sample.sequence_key))
        previous = previous_by_group.get(sample.group_key)
        previous_by_group[sample.group_key] = sample
        if previous is None:
            continue
        pair = _pair_from_samples(previous, sample, config)
        scope = _scope(pair.current)
        counts[(scope, "adjacent_pairs")] += 1
        counts[(scope, f"frame_gap/{pair.frame_gap_bin}")] += 1
        counts[(scope, f"time_gap/{pair.time_gap_bin}")] += 1
        counts[(scope, f"transition/{pair.transition}")] += 1
        counts[(scope, "positive_frame_order")] += pair.frame_gap > 0
        counts[(scope, "positive_time_order")] += (
            pair.time_gap is not None and pair.time_gap > 0.0
        )
        counts[(scope, "same_hdf5")] += pair.previous.h5_path == pair.current.h5_path
        counts[(scope, "bbox_stable")] += pair.bbox_stable
        counts[(scope, "temporal_eligible")] += pair.temporal_eligible
        counts[(scope, "eligible")] += pair.eligible
        if pair.eligible:
            counts[(scope, f"eligible_pressure_bin/{_numeric_bin(pair.current.max_pressure, PRESSURE_BINS)}")] += 1
            counts[(scope, f"eligible_volume_bin/{_numeric_bin(pair.current.target_volume, VOLUME_BINS)}")] += 1
            strata[pair.stratum] += 1

    scopes = sorted({key[0] for key in counts})
    summary = {
        "record_count": sum(counts[(scope, "records")] for scope in scopes),
        "sequence_count": len(sequences),
        "query_track_count": len(groups),
        "adjacent_pair_count": sum(counts[(scope, "adjacent_pairs")] for scope in scopes),
        "temporal_eligible_pair_count": sum(
            counts[(scope, "temporal_eligible")] for scope in scopes
        ),
        "eligible_pair_count": sum(counts[(scope, "eligible")] for scope in scopes),
        "scopes": scopes,
    }
    return summary, strata | Counter({("__structure__",) + key: value for key, value in counts.items()})


def _extract_structure_counts(combined: Counter) -> Counter:
    result = Counter()
    for key, value in combined.items():
        if len(key) == 3 and key[0] == "__structure__":
            result[(key[1], key[2])] = value
    return result


def _extract_strata_counts(combined: Counter) -> Counter:
    return Counter({key: value for key, value in combined.items() if key and key[0] != "__structure__"})


def _marginal_shift_rows(counts: Counter, scopes: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for scope in scopes:
        record_total = counts.get((scope, "records"), 0)
        eligible_total = counts.get((scope, "eligible"), 0)
        for family, bins in (("pressure", PRESSURE_BINS), ("volume", VOLUME_BINS)):
            differences = []
            for name, _, _ in bins:
                record_fraction = (
                    counts.get((scope, f"record_{family}_bin/{name}"), 0) / record_total
                    if record_total
                    else 0.0
                )
                eligible_fraction = (
                    counts.get((scope, f"eligible_{family}_bin/{name}"), 0) / eligible_total
                    if eligible_total
                    else 0.0
                )
                differences.append(abs(eligible_fraction - record_fraction))
            rows.append(
                {
                    "scope": scope,
                    "family": family,
                    "record_count": record_total,
                    "eligible_target_count": eligible_total,
                    "max_absolute_bin_fraction_shift": max(differences, default=0.0),
                    "total_variation_distance": 0.5 * sum(differences),
                }
            )
    return rows


def _largest_remainder(counts: Counter, total: int) -> dict[tuple[str, ...], int]:
    available = sum(counts.values())
    target = min(max(0, int(total)), available)
    if target == 0 or available == 0:
        return {key: 0 for key in counts}
    exact = {key: target * value / available for key, value in counts.items()}
    quotas = {key: min(counts[key], int(math.floor(value))) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    for key in order:
        if remaining <= 0:
            break
        if quotas[key] < counts[key]:
            quotas[key] += 1
            remaining -= 1
    if sum(quotas.values()) != target:
        raise RuntimeError("largest-remainder pressure-pair allocation failed")
    return quotas


def _pair_hash(pair: PairRecord, seed: int) -> int:
    payload = f"{seed}\0{pair.previous.sample_uid}\0{pair.current.sample_uid}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def select_pairs(config: AuditConfig, strata: Counter) -> list[PairRecord]:
    if config.max_pressure_pairs == 0:
        return []
    quotas = _largest_remainder(strata, config.max_pressure_pairs)
    heaps: dict[tuple[str, ...], list[tuple[int, str, PairRecord]]] = defaultdict(list)
    for pair in iter_pairs(config, pass_name="pressure-selection"):
        if not pair.eligible:
            continue
        quota = quotas.get(pair.stratum, 0)
        if quota <= 0:
            continue
        score = _pair_hash(pair, config.seed)
        identity = pair.previous.sample_uid + "\0" + pair.current.sample_uid
        item = (-score, identity, pair)
        heap = heaps[pair.stratum]
        if len(heap) < quota:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    selected = [item[2] for heap in heaps.values() for item in heap]
    expected = sum(quotas.values())
    if len(selected) != expected:
        raise RuntimeError(f"selected {len(selected)} pressure pairs, expected {expected}")
    return sorted(
        selected,
        key=lambda pair: (
            pair.current.h5_path,
            pair.current.query_row,
            pair.previous.query_row,
        ),
    )


class PressureReader:
    PRESSURE_DATASETS = (
        "targets/pressure",
        "queries/pressure/gaussian_subdiv",
        "tactile/pressure",
    )
    MAX_BULK_RUN_ROWS = 2048

    def __init__(
        self,
        max_handles: int,
        *,
        max_cached_rows: int = 0,
        palm_mask: Optional[np.ndarray] = None,
        cached_dtype: Any = np.float64,
    ):
        self.max_handles = max(1, int(max_handles))
        self.max_cached_rows = max(0, int(max_cached_rows))
        self.cached_dtype = np.dtype(cached_dtype)
        self.handles: OrderedDict[str, h5py.File] = OrderedDict()
        self.rows: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self.palm_mask = None
        self.palm_vertex_count = None
        if palm_mask is not None:
            mask = np.asarray(palm_mask, dtype=bool)
            if mask.ndim != 1:
                raise ValueError(f"palm mask must be 1-D, got {mask.shape}")
            self.palm_mask = mask
            self.palm_vertex_count = int(np.count_nonzero(mask))
        self.read_requests = 0
        self.cache_hits = 0
        self.disk_reads = 0
        self.cache_evictions = 0
        self.batch_reuses = 0
        self.hdf5_read_calls = 0
        self.bulk_read_calls = 0
        self.bulk_read_rows = 0
        self.bulk_fallback_rows = 0

    def close(self) -> None:
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self.handles.clear()
        self.rows.clear()

    def _handle(self, path: str) -> h5py.File:
        handle = self.handles.get(path)
        if handle is not None and handle.id.valid:
            self.handles.move_to_end(path)
            return handle
        if handle is not None:
            self.handles.pop(path, None)
            try:
                handle.close()
            except Exception:
                pass
        handle = open_readonly(path, raw_chunk_cache_bytes=4 * 1024 * 1024)
        self.handles[path] = handle
        while len(self.handles) > self.max_handles:
            _, old_handle = self.handles.popitem(last=False)
            old_handle.close()
        return handle

    @staticmethod
    def _cache_key(sample: SampleRecord) -> tuple[str, int]:
        return sample.h5_path, int(sample.query_row)

    def _store_cache(
        self, cache_key: tuple[str, int], value: np.ndarray, *, copy: bool
    ) -> None:
        if not self.max_cached_rows:
            return
        cached = np.array(value, dtype=self.cached_dtype, copy=copy)
        cached.setflags(write=False)
        self.rows[cache_key] = cached
        self.rows.move_to_end(cache_key)
        while len(self.rows) > self.max_cached_rows:
            self.rows.popitem(last=False)
            self.cache_evictions += 1

    def _prepare_pressure_block(
        self, value: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 2:
            raise ValueError(f"pressure block must be 2-D, got {value.shape}")
        finite = np.isfinite(value).all(axis=1)
        value = np.clip(value, 0.0, 1.0)
        if self.palm_mask is not None:
            if value.shape[1] == self.palm_mask.size:
                value = value[:, self.palm_mask]
            elif value.shape[1] != self.palm_vertex_count:
                raise ValueError(
                    f"pressure rows have {value.shape[1]} vertices; expected "
                    f"{self.palm_mask.size} full or "
                    f"{self.palm_vertex_count} palm vertices"
                )
        return np.asarray(value, dtype=self.cached_dtype), finite

    def _read_uncached(self, sample: SampleRecord) -> np.ndarray:
        cache_key = (sample.h5_path, int(sample.query_row))
        handle = self._handle(sample.h5_path)
        for name in self.PRESSURE_DATASETS:
            if name in handle:
                self.hdf5_read_calls += 1
                value = np.asarray(handle[name][sample.query_row], dtype=np.float32)
                if value.ndim == 1 and np.isfinite(value).all():
                    self.disk_reads += 1
                    value = np.clip(value, 0.0, 1.0)
                    if self.palm_mask is not None:
                        if value.shape == self.palm_mask.shape:
                            value = value[self.palm_mask]
                        elif value.size != self.palm_vertex_count:
                            raise ValueError(
                                f"pressure row has {value.size} vertices; expected "
                                f"{self.palm_mask.size} full or "
                                f"{self.palm_vertex_count} palm vertices"
                            )
                    # Convert once so repeated lag/control comparisons do not repeat
                    # masking and casting. CPU legacy metrics use FP64; the batched
                    # GPU path can retain source FP32 for lower transfer volume.
                    value = np.asarray(value, dtype=self.cached_dtype)
                    value.setflags(write=False)
                    self._store_cache(cache_key, value, copy=False)
                    return value
        raise KeyError(f"No finite pressure row {sample.query_row} in {sample.h5_path}")

    def read(self, sample: SampleRecord) -> np.ndarray:
        self.read_requests += 1
        cache_key = self._cache_key(sample)
        cached = self.rows.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            self.rows.move_to_end(cache_key)
            return cached
        return self._read_uncached(sample)

    def read_many(
        self, samples: Sequence[SampleRecord]
    ) -> list[np.ndarray | Exception]:
        """Read one pressure batch with file-local contiguous HDF5 slices."""

        if not samples:
            return []
        self.read_requests += len(samples)
        results: dict[tuple[str, int], np.ndarray | Exception] = {}
        scheduled: set[tuple[str, int]] = set()
        missing_by_path: dict[str, list[SampleRecord]] = defaultdict(list)
        cache_candidates: list[tuple[tuple[str, int], np.ndarray]] = []
        for sample in samples:
            cache_key = self._cache_key(sample)
            if cache_key in results or cache_key in scheduled:
                self.cache_hits += 1
                self.batch_reuses += 1
                continue
            cached = self.rows.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                self.rows.move_to_end(cache_key)
                results[cache_key] = cached
                continue
            scheduled.add(cache_key)
            missing_by_path[sample.h5_path].append(sample)

        for path, path_samples in missing_by_path.items():
            ordered = sorted(path_samples, key=lambda sample: int(sample.query_row))
            try:
                handle = self._handle(path)
                dataset = next(
                    (handle[name] for name in self.PRESSURE_DATASETS if name in handle),
                    None,
                )
                if dataset is None:
                    raise KeyError(f"No pressure dataset exists in {path}")
            except Exception as exc:
                for sample in ordered:
                    results[self._cache_key(sample)] = exc
                continue

            runs: list[list[SampleRecord]] = []
            current_run: list[SampleRecord] = []
            previous_row: Optional[int] = None
            for sample in ordered:
                row = int(sample.query_row)
                if (
                    current_run
                    and (
                        previous_row is None
                        or row != previous_row + 1
                        or len(current_run) >= self.MAX_BULK_RUN_ROWS
                    )
                ):
                    runs.append(current_run)
                    current_run = []
                current_run.append(sample)
                previous_row = row
            if current_run:
                runs.append(current_run)

            for run in runs:
                start = int(run[0].query_row)
                stop = int(run[-1].query_row) + 1
                try:
                    self.hdf5_read_calls += 1
                    self.bulk_read_calls += 1
                    block, finite = self._prepare_pressure_block(dataset[start:stop])
                    if block.shape[0] != len(run):
                        raise ValueError(
                            f"pressure slice {path}:{start}:{stop} returned "
                            f"{block.shape[0]} rows"
                        )
                except Exception:
                    self.bulk_fallback_rows += len(run)
                    for sample in run:
                        cache_key = self._cache_key(sample)
                        try:
                            results[cache_key] = self._read_uncached(sample)
                        except Exception as row_exc:
                            results[cache_key] = row_exc
                    continue

                self.bulk_read_rows += len(run)
                for offset, sample in enumerate(run):
                    cache_key = self._cache_key(sample)
                    if not bool(finite[offset]):
                        self.bulk_fallback_rows += 1
                        try:
                            results[cache_key] = self._read_uncached(sample)
                        except Exception as row_exc:
                            results[cache_key] = row_exc
                        continue
                    value = block[offset]
                    value.setflags(write=False)
                    results[cache_key] = value
                    self.disk_reads += 1
                    cache_candidates.append((cache_key, value))

        # Cache only the newest rows that can survive the LRU capacity. Copying
        # every row before immediately evicting most of them is expensive for
        # large audit batches, while caching a view would retain its whole block.
        if self.max_cached_rows:
            for cache_key, value in cache_candidates[-self.max_cached_rows :]:
                self._store_cache(cache_key, value, copy=True)

        return [results[self._cache_key(sample)] for sample in samples]

    def stats(self) -> dict[str, Any]:
        hit_rate = (
            self.cache_hits / self.read_requests if self.read_requests else 0.0
        )
        return {
            "read_requests": self.read_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": hit_rate,
            "disk_reads": self.disk_reads,
            "batch_reuses": self.batch_reuses,
            "hdf5_read_calls": self.hdf5_read_calls,
            "bulk_read_calls": self.bulk_read_calls,
            "bulk_read_rows": self.bulk_read_rows,
            "bulk_fallback_rows": self.bulk_fallback_rows,
            "cache_evictions": self.cache_evictions,
            "max_cached_rows": self.max_cached_rows,
            "cached_row_dtype": self.cached_dtype.name,
            "cached_row_vertex_count": self.palm_vertex_count,
        }


@dataclasses.dataclass(slots=True)
class RunningMetric:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: Optional[float]) -> None:
        if value is None or not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total_squared += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def merge_statistics(
        self,
        count: int,
        total: float,
        total_squared: float,
        minimum: float,
        maximum: float,
    ) -> None:
        if count <= 0:
            return
        self.count += int(count)
        self.total += float(total)
        self.total_squared += float(total_squared)
        self.minimum = min(self.minimum, float(minimum))
        self.maximum = max(self.maximum, float(maximum))

    def payload(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


class MetricTable:
    def __init__(self):
        self.values: dict[str, dict[str, RunningMetric]] = defaultdict(
            lambda: defaultdict(RunningMetric)
        )

    def add(self, scopes: Iterable[str], metrics: Mapping[str, Optional[float]]) -> None:
        for scope in scopes:
            for name, value in metrics.items():
                self.values[scope][name].add(value)

    def rows(self) -> list[dict[str, Any]]:
        result = []
        for scope in sorted(self.values):
            for metric in sorted(self.values[scope]):
                result.append(
                    {"scope": scope, "metric": metric, **self.values[scope][metric].payload()}
                )
        return result

    def mean(self, scope: str, metric: str) -> Optional[float]:
        running = self.values.get(scope, {}).get(metric)
        return None if running is None else running.payload()["mean"]


class BilateralMetricTable:
    """Metric cube with explicit wildcard marginals for direct CSV analysis."""

    WILDCARD = "*"

    def __init__(self):
        self.values: dict[
            tuple[str, str, str, str, str, str], dict[str, RunningMetric]
        ] = defaultdict(lambda: defaultdict(RunningMetric))

    def add(
        self,
        relation: str,
        record: BilateralPairRecord,
        self_dynamics: str,
        metrics: Mapping[str, Optional[float]],
    ) -> None:
        keys = self._keys(relation, record, self_dynamics)
        for key in keys:
            for name, value in metrics.items():
                self.values[key][name].add(value)

    def _keys(
        self,
        relation: str,
        record: BilateralPairRecord,
        self_dynamics: str,
    ) -> set[tuple[str, str, str, str, str, str]]:
        current = record.self_pair.current
        return self._keys_from_dimensions(
            relation,
            current.dataset,
            current.split,
            str(record.requested_lag),
            current.side_name,
            self_dynamics,
        )

    @classmethod
    def _keys_from_dimensions(
        cls,
        relation: str,
        dataset: str,
        split: str,
        lag: str,
        side: str,
        dynamics: str,
    ) -> set[tuple[str, str, str, str, str, str]]:
        w = cls.WILDCARD
        return {
            (relation, w, w, w, w, w),
            (relation, dataset, split, w, w, w),
            (
                relation,
                dataset,
                split,
                lag,
                w,
                w,
            ),
            (relation, dataset, split, w, side, w),
            (
                relation,
                dataset,
                split,
                lag,
                side,
                w,
            ),
            (relation, w, w, lag, w, w),
            (relation, w, w, lag, side, w),
            (relation, w, w, w, side, w),
            (relation, w, w, w, w, dynamics),
            (
                relation,
                dataset,
                split,
                lag,
                side,
                dynamics,
            ),
        }

    def add_batch(
        self,
        relation: str,
        records: Sequence[BilateralPairRecord],
        self_dynamics: Sequence[str],
        metrics: Mapping[str, Sequence[Optional[float]]],
    ) -> None:
        batch_size = len(records)
        if batch_size == 0:
            return
        if len(self_dynamics) != batch_size:
            raise ValueError("batched dynamics labels do not match records")
        names = tuple(metrics)
        if not names:
            return
        columns = []
        for name in names:
            values = np.asarray(metrics[name], dtype=np.float64)
            if values.shape != (batch_size,):
                raise ValueError(
                    f"batched metric {name!r} has shape {values.shape}; "
                    f"expected {(batch_size,)}"
                )
            columns.append(values)
        matrix = np.stack(columns, axis=1)
        groups: dict[tuple[str, str, str, str, str], list[int]] = defaultdict(list)
        for index, (record, dynamics) in enumerate(zip(records, self_dynamics)):
            current = record.self_pair.current
            groups[
                (
                    current.dataset,
                    current.split,
                    str(record.requested_lag),
                    current.side_name,
                    dynamics,
                )
            ].append(index)
        for dimensions, indices in groups.items():
            values = matrix[np.asarray(indices, dtype=np.int64)]
            finite = np.isfinite(values)
            counts = finite.sum(axis=0)
            safe = np.where(finite, values, 0.0)
            totals = safe.sum(axis=0, dtype=np.float64)
            total_squared = np.square(safe).sum(axis=0, dtype=np.float64)
            minima = np.where(finite, values, np.inf).min(axis=0)
            maxima = np.where(finite, values, -np.inf).max(axis=0)
            for key in self._keys_from_dimensions(relation, *dimensions):
                for metric_index, name in enumerate(names):
                    running = self.values[key][name]
                    count = int(counts[metric_index])
                    if count:
                        running.merge_statistics(
                            count,
                            float(totals[metric_index]),
                            float(total_squared[metric_index]),
                            float(minima[metric_index]),
                            float(maxima[metric_index]),
                        )

    def rows(self) -> list[dict[str, Any]]:
        rows = []
        for key in sorted(self.values):
            relation, dataset, split, lag, side, dynamics = key
            for metric in sorted(self.values[key]):
                rows.append(
                    {
                        "relation": relation,
                        "dataset": dataset,
                        "split": split,
                        "lag": lag,
                        "side": side,
                        "self_dynamics_class": dynamics,
                        "metric": metric,
                        **self.values[key][metric].payload(),
                    }
                )
        return rows

    def metric(
        self,
        relation: str,
        metric: str,
        *,
        dataset: str = WILDCARD,
        split: str = WILDCARD,
        lag: str = WILDCARD,
        side: str = WILDCARD,
        dynamics: str = WILDCARD,
    ) -> Optional[RunningMetric]:
        return self.values.get(
            (relation, dataset, split, lag, side, dynamics), {}
        ).get(metric)

    def mean(self, relation: str, metric: str, **dimensions: str) -> Optional[float]:
        running = self.metric(relation, metric, **dimensions)
        return None if running is None else running.payload()["mean"]

    def count(self, relation: str, metric: str, **dimensions: str) -> int:
        running = self.metric(relation, metric, **dimensions)
        return 0 if running is None else running.count


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return None if denominator <= 1e-12 else float(numerator / denominator)


def pressure_pair_metrics(
    previous: np.ndarray,
    current: np.ndarray,
    palm_mask: np.ndarray,
    pair: PairRecord,
    config: AuditConfig,
    *,
    already_masked: bool = False,
) -> tuple[dict[str, Optional[float]], str]:
    if previous.shape != current.shape:
        raise ValueError(
            f"pressure shapes differ: {previous.shape}, {current.shape}"
        )
    if already_masked:
        if previous.ndim != 1:
            raise ValueError(
                f"pre-masked pressure rows must be 1-D, got {previous.shape}"
            )
        previous = previous.astype(np.float64, copy=False)
        current = current.astype(np.float64, copy=False)
    else:
        if previous.shape != palm_mask.shape:
            raise ValueError(
                f"pressure/mask shapes differ: {previous.shape}, {palm_mask.shape}"
            )
        previous = previous[palm_mask].astype(np.float64, copy=False)
        current = current[palm_mask].astype(np.float64, copy=False)
    delta = current - previous
    previous_volume = float(previous.sum())
    current_volume = float(current.sum())
    union_volume = float(np.maximum(previous, current).sum())
    persistence_rmse = float(np.sqrt(np.mean(delta * delta)))
    persistence_mae = float(np.mean(np.abs(delta)))
    zero_rmse = float(np.sqrt(np.mean(current * current)))
    previous_dist = previous / previous_volume if previous_volume > 1e-12 else None
    current_dist = current / current_volume if current_volume > 1e-12 else None
    distribution_viou = None
    if previous_dist is not None and current_dist is not None:
        distribution_viou = _safe_ratio(
            float(np.minimum(previous_dist, current_dist).sum()),
            float(np.maximum(previous_dist, current_dist).sum()),
        )
    oracle_scaled = np.zeros_like(previous)
    if previous_volume > 1e-12:
        oracle_scaled = np.clip(previous * (current_volume / previous_volume), 0.0, 1.0)
    oracle_delta = current - oracle_scaled

    threshold = config.contact_threshold
    previous_support = previous >= threshold
    current_support = current >= threshold
    support_intersection = int(np.count_nonzero(previous_support & current_support))
    current_support_count = int(np.count_nonzero(current_support))
    previous_support_count = int(np.count_nonzero(previous_support))
    support_union = (
        previous_support_count + current_support_count - support_intersection
    )
    previous_active = previous >= config.active_threshold
    current_active = current >= config.active_threshold
    active_intersection = int(np.count_nonzero(previous_active & current_active))
    current_active_count = int(np.count_nonzero(current_active))
    previous_active_count = int(np.count_nonzero(previous_active))
    active_union = previous_active_count + current_active_count - active_intersection
    relative_volume_change = abs(current_volume - previous_volume) / max(
        current_volume, previous_volume, 1.0
    )
    if previous_volume <= 1e-12 and current_volume <= 1e-12:
        dynamics_class = "empty_stable"
    elif previous_volume <= 1e-12 or current_volume <= 1e-12 or relative_volume_change > 0.25:
        dynamics_class = "source_sink_dominant"
    elif distribution_viou is not None and distribution_viou >= 0.75:
        dynamics_class = "spatially_stable"
    elif distribution_viou is not None and distribution_viou >= 0.25:
        dynamics_class = "transport_candidate"
    else:
        dynamics_class = "large_spatial_change"

    metrics: dict[str, Optional[float]] = {
        "persistence_mae": persistence_mae,
        "persistence_rmse": persistence_rmse,
        "zero_mae": float(np.mean(np.abs(current))),
        "zero_rmse": zero_rmse,
        "persistence_rmse_gain_vs_zero": zero_rmse - persistence_rmse,
        "persistence_relative_rmse_gain_vs_zero": _safe_ratio(
            zero_rmse - persistence_rmse, zero_rmse
        ),
        "persistence_viou": _safe_ratio(
            float(np.minimum(previous, current).sum()), union_volume
        ),
        "distribution_viou": distribution_viou,
        "oracle_volume_scaled_rmse": float(np.sqrt(np.mean(oracle_delta * oracle_delta))),
        "oracle_volume_scaled_viou": _safe_ratio(
            float(np.minimum(oracle_scaled, current).sum()),
            float(np.maximum(oracle_scaled, current).sum()),
        ),
        "support_iou": _safe_ratio(float(support_intersection), float(support_union)),
        "current_support_recall_from_previous": _safe_ratio(
            float(support_intersection), float(current_support_count)
        ),
        "previous_support_retained": _safe_ratio(
            float(support_intersection), float(previous_support_count)
        ),
        "current_mass_on_previous_support": _safe_ratio(
            float(current[previous_support].sum()), current_volume
        ),
        "previous_mass_on_current_support": _safe_ratio(
            float(previous[current_support].sum()), previous_volume
        ),
        "birth_fraction_of_union": _safe_ratio(
            float(current_support_count - support_intersection), float(support_union)
        ),
        "death_fraction_of_union": _safe_ratio(
            float(previous_support_count - support_intersection), float(support_union)
        ),
        "active_support_iou": _safe_ratio(
            float(active_intersection), float(active_union)
        ),
        "active_birth_fraction_of_union": _safe_ratio(
            float(current_active_count - active_intersection), float(active_union)
        ),
        "active_death_fraction_of_union": _safe_ratio(
            float(previous_active_count - active_intersection), float(active_union)
        ),
        "previous_volume": previous_volume,
        "current_volume": current_volume,
        "signed_volume_delta": current_volume - previous_volume,
        "absolute_volume_delta": abs(current_volume - previous_volume),
        "relative_volume_change": relative_volume_change,
        "delta_rms": persistence_rmse,
        "delta_mae_per_second": (
            None
            if pair.time_gap is None or pair.time_gap <= 0.0
            else persistence_mae / pair.time_gap
        ),
        "bbox_iou": pair.bbox_iou,
        "bbox_center_jump": pair.bbox_center_jump,
        "bbox_abs_log_area_ratio": pair.bbox_abs_log_area_ratio,
        "frame_gap": float(pair.frame_gap),
        "time_gap": pair.time_gap,
    }
    return metrics, dynamics_class


class BatchedPressureMetricEngine:
    """Vectorized pressure-pair metrics for CUDA or batched Torch CPU runs."""

    def __init__(self, config: AuditConfig):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The batched pressure metric backend requires PyTorch"
            ) from exc
        requested = config.pressure_metric_device.strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda":
            requested = "cuda:0"
        self.torch = torch
        self.device = torch.device(requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA pressure metrics were requested on {requested}, but CUDA is unavailable"
            )
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(
                "--pressure-metric-device must be cpu, auto, cuda, or cuda:N"
            )
        self.dtype = {
            "float32": torch.float32,
            "float64": torch.float64,
        }[config.pressure_metric_dtype]
        self.numpy_dtype = {
            "float32": np.float32,
            "float64": np.float64,
        }[config.pressure_metric_dtype]
        self.contact_threshold = float(config.contact_threshold)
        self.active_threshold = float(config.active_threshold)

    @staticmethod
    def row(
        metrics: Mapping[str, np.ndarray], index: int
    ) -> dict[str, Optional[float]]:
        row: dict[str, Optional[float]] = {}
        for name, values in metrics.items():
            value = float(values[index])
            row[name] = value if math.isfinite(value) else None
        return row

    def compute(
        self,
        previous_rows: Sequence[np.ndarray],
        current_rows: Sequence[np.ndarray],
        pairs: Sequence[PairRecord],
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        if not previous_rows:
            return {}, []
        if not (len(previous_rows) == len(current_rows) == len(pairs)):
            raise ValueError("batched pressure inputs have different lengths")
        previous_array = np.stack(previous_rows).astype(self.numpy_dtype, copy=False)
        current_array = np.stack(current_rows).astype(self.numpy_dtype, copy=False)
        torch = self.torch

        with torch.inference_mode():
            previous = torch.from_numpy(previous_array).to(
                device=self.device, dtype=self.dtype, non_blocking=False
            )
            current = torch.from_numpy(current_array).to(
                device=self.device, dtype=self.dtype, non_blocking=False
            )
            if previous.ndim != 2 or previous.shape != current.shape:
                raise ValueError(
                    f"batched pressure shapes differ: {tuple(previous.shape)}, "
                    f"{tuple(current.shape)}"
                )
            epsilon = 1e-12

            def ratio(numerator, denominator):
                nan = torch.full_like(numerator, float("nan"))
                return torch.where(
                    denominator > epsilon,
                    numerator / denominator.clamp_min(epsilon),
                    nan,
                )

            delta = current - previous
            absolute_delta = delta.abs()
            previous_volume = previous.sum(dim=1)
            current_volume = current.sum(dim=1)
            union_volume = torch.maximum(previous, current).sum(dim=1)
            persistence_rmse = delta.square().mean(dim=1).sqrt()
            persistence_mae = absolute_delta.mean(dim=1)
            zero_rmse = current.square().mean(dim=1).sqrt()
            zero_mae = current.abs().mean(dim=1)
            persistence_minimum = torch.minimum(previous, current).sum(dim=1)

            previous_positive = previous_volume > epsilon
            current_positive = current_volume > epsilon
            previous_dist = previous / previous_volume.clamp_min(epsilon)[:, None]
            current_dist = current / current_volume.clamp_min(epsilon)[:, None]
            distribution_viou = ratio(
                torch.minimum(previous_dist, current_dist).sum(dim=1),
                torch.maximum(previous_dist, current_dist).sum(dim=1),
            )
            distribution_viou = torch.where(
                previous_positive & current_positive,
                distribution_viou,
                torch.full_like(distribution_viou, float("nan")),
            )

            oracle_scale = torch.where(
                previous_positive,
                current_volume / previous_volume.clamp_min(epsilon),
                torch.zeros_like(previous_volume),
            )
            oracle_scaled = torch.clamp(previous * oracle_scale[:, None], 0.0, 1.0)
            oracle_delta = current - oracle_scaled

            previous_support = previous >= self.contact_threshold
            current_support = current >= self.contact_threshold
            support_intersection = (previous_support & current_support).sum(dim=1).to(self.dtype)
            previous_support_count = previous_support.sum(dim=1).to(self.dtype)
            current_support_count = current_support.sum(dim=1).to(self.dtype)
            support_union = (
                previous_support_count + current_support_count - support_intersection
            )
            previous_active = previous >= self.active_threshold
            current_active = current >= self.active_threshold
            active_intersection = (previous_active & current_active).sum(dim=1).to(self.dtype)
            previous_active_count = previous_active.sum(dim=1).to(self.dtype)
            current_active_count = current_active.sum(dim=1).to(self.dtype)
            active_union = previous_active_count + current_active_count - active_intersection

            relative_volume_change = (current_volume - previous_volume).abs() / torch.maximum(
                torch.maximum(current_volume, previous_volume),
                torch.ones_like(current_volume),
            )
            metrics_t = {
                "persistence_mae": persistence_mae,
                "persistence_rmse": persistence_rmse,
                "zero_mae": zero_mae,
                "zero_rmse": zero_rmse,
                "persistence_rmse_gain_vs_zero": zero_rmse - persistence_rmse,
                "persistence_relative_rmse_gain_vs_zero": ratio(
                    zero_rmse - persistence_rmse, zero_rmse
                ),
                "persistence_viou": ratio(persistence_minimum, union_volume),
                "distribution_viou": distribution_viou,
                "oracle_volume_scaled_rmse": oracle_delta.square().mean(dim=1).sqrt(),
                "oracle_volume_scaled_viou": ratio(
                    torch.minimum(oracle_scaled, current).sum(dim=1),
                    torch.maximum(oracle_scaled, current).sum(dim=1),
                ),
                "support_iou": ratio(support_intersection, support_union),
                "current_support_recall_from_previous": ratio(
                    support_intersection, current_support_count
                ),
                "previous_support_retained": ratio(
                    support_intersection, previous_support_count
                ),
                "current_mass_on_previous_support": ratio(
                    (current * previous_support).sum(dim=1), current_volume
                ),
                "previous_mass_on_current_support": ratio(
                    (previous * current_support).sum(dim=1), previous_volume
                ),
                "birth_fraction_of_union": ratio(
                    current_support_count - support_intersection, support_union
                ),
                "death_fraction_of_union": ratio(
                    previous_support_count - support_intersection, support_union
                ),
                "active_support_iou": ratio(active_intersection, active_union),
                "active_birth_fraction_of_union": ratio(
                    current_active_count - active_intersection, active_union
                ),
                "active_death_fraction_of_union": ratio(
                    previous_active_count - active_intersection, active_union
                ),
                "previous_volume": previous_volume,
                "current_volume": current_volume,
                "signed_volume_delta": current_volume - previous_volume,
                "absolute_volume_delta": (current_volume - previous_volume).abs(),
                "relative_volume_change": relative_volume_change,
                "delta_rms": persistence_rmse,
            }
            metric_names = tuple(metrics_t)
            metric_matrix = torch.stack(
                [metrics_t[name] for name in metric_names], dim=1
            ).cpu().numpy()
            del previous, current

        metrics_np = {
            name: metric_matrix[:, index]
            for index, name in enumerate(metric_names)
        }
        time_gaps = np.asarray(
            [float("nan") if pair.time_gap is None else pair.time_gap for pair in pairs],
            dtype=np.float64,
        )
        delta_mae = metrics_np["persistence_mae"].astype(np.float64, copy=False)
        metrics_np["delta_mae_per_second"] = np.divide(
            delta_mae,
            time_gaps,
            out=np.full_like(time_gaps, np.nan),
            where=np.isfinite(time_gaps) & (time_gaps > 0.0),
        )
        metrics_np["bbox_iou"] = np.asarray(
            [pair.bbox_iou for pair in pairs], dtype=np.float64
        )
        metrics_np["bbox_center_jump"] = np.asarray(
            [pair.bbox_center_jump for pair in pairs], dtype=np.float64
        )
        metrics_np["bbox_abs_log_area_ratio"] = np.asarray(
            [pair.bbox_abs_log_area_ratio for pair in pairs], dtype=np.float64
        )
        metrics_np["frame_gap"] = np.asarray(
            [pair.frame_gap for pair in pairs], dtype=np.float64
        )
        metrics_np["time_gap"] = time_gaps

        previous_volume_np = metrics_np["previous_volume"]
        current_volume_np = metrics_np["current_volume"]
        relative_change_np = metrics_np["relative_volume_change"]
        distribution_np = metrics_np["distribution_viou"]
        dynamics = []
        for index in range(len(pairs)):
            previous_volume_value = float(previous_volume_np[index])
            current_volume_value = float(current_volume_np[index])
            distribution_value = float(distribution_np[index])
            if previous_volume_value <= 1e-12 and current_volume_value <= 1e-12:
                label = "empty_stable"
            elif (
                previous_volume_value <= 1e-12
                or current_volume_value <= 1e-12
                or float(relative_change_np[index]) > 0.25
            ):
                label = "source_sink_dominant"
            elif math.isfinite(distribution_value) and distribution_value >= 0.75:
                label = "spatially_stable"
            elif math.isfinite(distribution_value) and distribution_value >= 0.25:
                label = "transport_candidate"
            else:
                label = "large_spatial_change"
            dynamics.append(label)
        return metrics_np, dynamics

    def validate_against_scalar(self, config: AuditConfig) -> None:
        previous_rows = (
            np.zeros(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
            np.asarray([0.0, 0.02, 0.08, 0.12, 0.3, 0.7, 1.0, 0.0], dtype=np.float32),
        )
        current_rows = (
            np.zeros(8, dtype=np.float32),
            np.asarray([0.0, 0.01, 0.06, 0.2, 0.5, 0.8, 0.9, 0.0], dtype=np.float32),
            np.asarray([0.0, 0.04, 0.02, 0.18, 0.25, 0.9, 0.6, 0.1], dtype=np.float32),
        )
        pair = PairRecord(
            previous=None,  # type: ignore[arg-type]
            current=None,  # type: ignore[arg-type]
            frame_gap=1,
            time_gap=1.0 / 30.0,
            bbox_iou=0.75,
            bbox_center_jump=0.1,
            bbox_abs_log_area_ratio=0.05,
            bbox_stable=True,
            temporal_eligible=True,
            eligible=True,
            transition="steady",
        )
        pairs = (pair,) * len(previous_rows)
        batched_metrics, batched_dynamics = self.compute(
            previous_rows, current_rows, pairs
        )
        tolerance = 3e-5 if self.dtype == self.torch.float32 else 1e-10
        mask = np.ones(8, dtype=bool)
        for index, (previous, current) in enumerate(
            zip(previous_rows, current_rows)
        ):
            scalar, scalar_dynamics = pressure_pair_metrics(
                previous,
                current,
                mask,
                pair,
                config,
                already_masked=True,
            )
            if scalar_dynamics != batched_dynamics[index]:
                raise RuntimeError(
                    "Batched pressure metric preflight changed the dynamics class: "
                    f"{scalar_dynamics} vs {batched_dynamics[index]}"
                )
            for name, expected in scalar.items():
                raw_actual = float(batched_metrics[name][index])
                actual = raw_actual if math.isfinite(raw_actual) else None
                if expected is None or actual is None:
                    if expected is not None or actual is not None:
                        raise RuntimeError(
                            f"Batched pressure metric preflight differs for {name}: "
                            f"{expected} vs {actual}"
                        )
                    continue
                if not math.isclose(
                    float(expected), float(actual), rel_tol=tolerance, abs_tol=tolerance
                ):
                    raise RuntimeError(
                        f"Batched pressure metric preflight differs for {name}: "
                        f"{expected} vs {actual}"
                    )

        audit_records = []
        for index in range(len(previous_rows)):
            current_sample = SampleRecord(
                manifest_index=0,
                line_number=index + 1,
                sample_uid=f"preflight/current/{index}",
                dataset="TouchAnything",
                split="train" if index < 2 else "val",
                sequence_key="preflight",
                query_alias="right" if index % 2 == 0 else "left",
                is_right=int(index % 2 == 0),
                frame_idx=index + 1,
                source_frame_idx=index + 1,
                timestamp=(index + 1) / 30.0,
                bbox_xyxy=(10.0, 10.0, 110.0, 110.0),
                bbox_score=1.0,
                bbox_source_schema="preflight",
                bbox_raw_track_id=None,
                bbox_association_id=None,
                bbox_association_confidence=None,
                bbox_association_policy=None,
                pressure_source_key="preflight",
                h5_path="preflight.h5",
                query_row=index,
                max_pressure=float(np.max(current_rows[index])),
                target_volume=float(np.sum(current_rows[index])),
                target_active_count=int(
                    np.count_nonzero(current_rows[index] >= self.active_threshold)
                ),
            )
            previous_sample = dataclasses.replace(
                current_sample,
                sample_uid=f"preflight/previous/{index}",
                frame_idx=index,
                source_frame_idx=index,
                timestamp=index / 30.0,
            )
            audit_records.append(
                BilateralPairRecord(
                    self_pair=dataclasses.replace(
                        pair,
                        previous=previous_sample,
                        current=current_sample,
                    ),
                    requested_lag=1,
                    contralateral_previous=None,
                )
            )

        with tempfile.TemporaryDirectory(prefix="tactile-pressure-reader-") as root:
            h5_path = Path(root) / "preflight.h5"
            pressure_data = np.stack(current_rows).astype(np.float32, copy=False)
            with h5py.File(h5_path, "w", libver="latest") as handle:
                targets = handle.create_group("targets")
                targets.create_dataset(
                    "pressure",
                    data=pressure_data,
                    chunks=(1, pressure_data.shape[1]),
                    compression="lzf",
                )
            pressure_samples = [
                dataclasses.replace(
                    record.self_pair.current,
                    h5_path=str(h5_path),
                    query_row=index,
                )
                for index, record in enumerate(audit_records)
            ]
            request_order = (2, 0, 1, 2, 0)
            pressure_reader = PressureReader(
                1,
                max_cached_rows=2,
                palm_mask=None,
                cached_dtype=self.numpy_dtype,
            )
            try:
                loaded = pressure_reader.read_many(
                    [pressure_samples[index] for index in request_order]
                )
            finally:
                pressure_reader.close()
            for loaded_row, source_index in zip(loaded, request_order):
                if isinstance(loaded_row, Exception):
                    raise RuntimeError(
                        "Batched pressure reader preflight failed"
                    ) from loaded_row
                if not np.array_equal(
                    loaded_row,
                    pressure_data[source_index].astype(
                        self.numpy_dtype, copy=False
                    ),
                ):
                    raise RuntimeError(
                        "Batched pressure reader preflight changed a pressure row"
                    )
            if pressure_reader.bulk_read_calls != 1:
                raise RuntimeError(
                    "Batched pressure reader failed to merge contiguous query rows"
                )

        scalar_table = BilateralMetricTable()
        batched_table = BilateralMetricTable()
        for index, record in enumerate(audit_records):
            scalar_table.add(
                "preflight",
                record,
                batched_dynamics[index],
                self.row(batched_metrics, index),
            )
        batched_table.add_batch(
            "preflight", audit_records, batched_dynamics, batched_metrics
        )
        if scalar_table.values.keys() != batched_table.values.keys():
            raise RuntimeError("Batched metric aggregation changed report scopes")
        for key, expected_metrics in scalar_table.values.items():
            actual_metrics = batched_table.values[key]
            if expected_metrics.keys() != actual_metrics.keys():
                raise RuntimeError("Batched metric aggregation changed metric names")
            for name, expected in expected_metrics.items():
                actual = actual_metrics[name]
                if expected.count != actual.count:
                    raise RuntimeError(
                        f"Batched metric aggregation changed count for {name}"
                    )
                for attribute in (
                    "total",
                    "total_squared",
                    "minimum",
                    "maximum",
                ):
                    expected_value = float(getattr(expected, attribute))
                    actual_value = float(getattr(actual, attribute))
                    if not math.isclose(
                        expected_value,
                        actual_value,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        raise RuntimeError(
                            "Batched metric aggregation changed "
                            f"{attribute} for {name}: "
                            f"{expected_value} vs {actual_value}"
                        )


def _load_palm_mask() -> tuple[np.ndarray, dict[str, str]]:
    obj_path = REPO_ROOT / "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj"
    faces_path = REPO_ROOT / "opentouch/preprocess/scratch/auto_calibrated_palm_subdiv_faces.json"
    if not obj_path.is_file() or not faces_path.is_file():
        raise FileNotFoundError("Canonical subdiv mesh or palm-face definition is missing")
    vertex_count = sum(1 for line in obj_path.open("r", encoding="utf-8") if line.startswith("v "))
    payload = json.loads(faces_path.read_text(encoding="utf-8"))
    mask = np.zeros(vertex_count, dtype=bool)
    for face in payload["group_negative"]["face_triplets"]:
        for vertex in face:
            mask[int(vertex)] = True
    if not mask.any():
        raise RuntimeError("Canonical palm mask is empty")
    return mask, {
        "mesh_sha256": sha256_file(obj_path),
        "palm_faces_sha256": sha256_file(faces_path),
        "vertex_count": str(vertex_count),
        "valid_palm_vertex_count": str(int(mask.sum())),
    }


def _pair_detail(pair: PairRecord, metrics: Mapping[str, Optional[float]], dynamics: str) -> dict[str, Any]:
    return {
        "dataset": pair.current.dataset,
        "split": pair.current.split,
        "sequence_key": pair.current.sequence_key,
        "query_alias": pair.current.query_alias,
        "previous_sample_uid": pair.previous.sample_uid,
        "current_sample_uid": pair.current.sample_uid,
        "previous_frame_idx": pair.previous.frame_idx,
        "current_frame_idx": pair.current.frame_idx,
        "frame_gap": pair.frame_gap,
        "time_gap": pair.time_gap,
        "frame_gap_bin": pair.frame_gap_bin,
        "time_gap_bin": pair.time_gap_bin,
        "bbox_stable": int(pair.bbox_stable),
        "transition": pair.transition,
        "dynamics_class": dynamics,
        **metrics,
    }


def audit_pressures(
    config: AuditConfig,
    palm_mask: np.ndarray,
    selected: Sequence[PairRecord],
) -> tuple[
    MetricTable,
    Counter,
    list[dict[str, Any]],
    int,
    list[str],
    dict[str, Any],
]:
    table = MetricTable()
    classes: Counter = Counter()
    details: list[dict[str, Any]] = []
    errors: list[str] = []
    error_count = 0
    pressure_started = time.time()
    reader = PressureReader(
        config.max_open_hdf5,
        max_cached_rows=config.pressure_row_cache_size,
        palm_mask=palm_mask,
    )
    processed = 0
    try:
        pairs: Iterable[PairRecord]
        if config.max_pressure_pairs == 0:
            pairs = (pair for pair in iter_pairs(config, pass_name="pressure-full") if pair.eligible)
        else:
            pairs = selected
        for pair in pairs:
            try:
                previous = reader.read(pair.previous)
                current = reader.read(pair.current)
                metrics, dynamics = pressure_pair_metrics(
                    previous,
                    current,
                    palm_mask,
                    pair,
                    config,
                    already_masked=True,
                )
            except Exception as exc:
                error_count += 1
                if len(errors) < 50:
                    errors.append(
                        f"{pair.previous.sample_uid} -> {pair.current.sample_uid}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                continue
            scopes = (
                "overall",
                f"dataset_split/{_scope(pair.current)}",
                f"frame_gap/{pair.frame_gap_bin}",
                f"time_gap/{pair.time_gap_bin}",
                f"transition/{pair.transition}",
                f"dynamics/{dynamics}",
            )
            table.add(scopes, metrics)
            classes[(pair.current.dataset, pair.current.split, dynamics)] += 1
            classes[("overall", "overall", dynamics)] += 1
            processed += 1
            if len(details) < config.pair_csv_limit:
                details.append(_pair_detail(pair, metrics, dynamics))
            if config.progress_every and processed % config.progress_every == 0:
                stats = reader.stats()
                print(
                    f"[pressure] audited {processed:,} pairs; "
                    f"rate={processed / max(time.time() - pressure_started, 1e-9):,.1f}/s; "
                    f"cache_hit={stats['cache_hit_rate']:.1%}; "
                    f"HDF5_reads={stats['disk_reads']:,}",
                    flush=True,
                )
    finally:
        reader.close()
    reader_stats = reader.stats()
    pressure_elapsed = time.time() - pressure_started
    reader_stats.update(
        {
            "processed_pairs": processed,
            "elapsed_seconds": pressure_elapsed,
            "pairs_per_second": processed / max(pressure_elapsed, 1e-9),
        }
    )
    return table, classes, details, error_count, errors, reader_stats


_PAIRED_COMPARISON_METRICS = (
    "persistence_rmse",
    "persistence_viou",
    "distribution_viou",
    "support_iou",
    "active_support_iou",
    "relative_volume_change",
)


def _paired_metric_delta(
    contralateral: Mapping[str, Optional[float]],
    own: Mapping[str, Optional[float]],
) -> dict[str, Optional[float]]:
    result = {}
    for name in _PAIRED_COMPARISON_METRICS:
        lhs = contralateral.get(name)
        rhs = own.get(name)
        result[f"contralateral_minus_self_{name}"] = (
            None if lhs is None or rhs is None else lhs - rhs
        )
    return result


def _bilateral_association_metrics(
    record: BilateralPairRecord,
    own_metrics: Mapping[str, Optional[float]],
    contralateral_metrics: Mapping[str, Optional[float]],
    config: AuditConfig,
) -> dict[str, Optional[float]]:
    opposite = record.contralateral_previous
    if opposite is None:
        return {}
    pair = record.self_pair
    other_iou, other_center, other_area = _bbox_metrics(
        opposite.bbox_xyxy, pair.current.bbox_xyxy
    )
    interhand_iou, interhand_center, _ = _bbox_metrics(
        pair.previous.bbox_xyxy, opposite.bbox_xyxy
    )
    self_iou = pair.bbox_iou
    self_center = pair.bbox_center_jump
    iou_margin = self_iou - other_iou
    center_margin = other_center - self_center
    own_viou = own_metrics.get("persistence_viou")
    other_viou = contralateral_metrics.get("persistence_viou")
    own_rmse = own_metrics.get("persistence_rmse")
    other_rmse = contralateral_metrics.get("persistence_rmse")
    return {
        "self_bbox_iou": self_iou,
        "contralateral_bbox_iou": other_iou,
        "bbox_iou_self_margin": iou_margin,
        "bbox_iou_prefers_self": float(iou_margin > 0.0),
        "bbox_iou_ambiguous": float(
            abs(iou_margin) <= config.association_iou_ambiguity_margin
        ),
        "self_bbox_center_jump": self_center,
        "contralateral_bbox_center_jump": other_center,
        "bbox_center_self_margin": center_margin,
        "bbox_center_prefers_self": float(center_margin > 0.0),
        "bbox_center_ambiguous": float(
            abs(center_margin) <= config.association_center_ambiguity_margin
        ),
        "previous_interhand_bbox_iou": interhand_iou,
        "previous_interhand_center_distance": interhand_center,
        "contralateral_bbox_abs_log_area_ratio": other_area,
        "pressure_viou_self_margin": (
            None if own_viou is None or other_viou is None else own_viou - other_viou
        ),
        "pressure_viou_prefers_self": (
            None if own_viou is None or other_viou is None else float(own_viou > other_viou)
        ),
        "wrong_history_rmse_penalty": (
            None if own_rmse is None or other_rmse is None else other_rmse - own_rmse
        ),
    }


def _slice_metric_columns(
    metrics: Mapping[str, np.ndarray], start: int, stop: Optional[int] = None
) -> dict[str, np.ndarray]:
    return {name: np.asarray(values[start:stop]) for name, values in metrics.items()}


def _paired_metric_delta_columns(
    contralateral: Mapping[str, np.ndarray],
    own: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        f"contralateral_minus_self_{name}": (
            np.asarray(contralateral[name], dtype=np.float64)
            - np.asarray(own[name], dtype=np.float64)
        )
        for name in _PAIRED_COMPARISON_METRICS
    }


def _bbox_metrics_batch(
    previous: np.ndarray, current: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous = np.asarray(previous, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if previous.ndim != 2 or current.shape != previous.shape or previous.shape[1] != 4:
        raise ValueError(
            f"batched bbox arrays must both be [B,4], got {previous.shape}, {current.shape}"
        )
    aw = previous[:, 2] - previous[:, 0]
    ah = previous[:, 3] - previous[:, 1]
    bw = current[:, 2] - current[:, 0]
    bh = current[:, 3] - current[:, 1]
    valid = (
        np.isfinite(previous).all(axis=1)
        & np.isfinite(current).all(axis=1)
        & (aw > 1.0)
        & (ah > 1.0)
        & (bw > 1.0)
        & (bh > 1.0)
    )
    intersection_w = np.maximum(
        0.0, np.minimum(previous[:, 2], current[:, 2]) - np.maximum(previous[:, 0], current[:, 0])
    )
    intersection_h = np.maximum(
        0.0, np.minimum(previous[:, 3], current[:, 3]) - np.maximum(previous[:, 1], current[:, 1])
    )
    intersection = intersection_w * intersection_h
    previous_area = aw * ah
    current_area = bw * bh
    union = previous_area + current_area - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(union),
        where=union > 0.0,
    )
    previous_center_x = (previous[:, 0] + previous[:, 2]) * 0.5
    previous_center_y = (previous[:, 1] + previous[:, 3]) * 0.5
    current_center_x = (current[:, 0] + current[:, 2]) * 0.5
    current_center_y = (current[:, 1] + current[:, 3]) * 0.5
    center = np.sqrt(
        np.square(previous_center_x - current_center_x)
        + np.square(previous_center_y - current_center_y)
    ) / np.maximum(np.sqrt(np.maximum(previous_area, 0.0)), 1.0)
    area_ratio = np.abs(
        np.log(
            np.maximum(current_area, 1.0)
            / np.maximum(previous_area, 1.0)
        )
    )
    for values in (iou, center, area_ratio):
        values[~valid] = np.nan
    return iou, center, area_ratio


def _bilateral_association_metric_columns(
    records: Sequence[BilateralPairRecord],
    own_metrics: Mapping[str, np.ndarray],
    contralateral_metrics: Mapping[str, np.ndarray],
    config: AuditConfig,
) -> dict[str, np.ndarray]:
    if not records:
        return {}
    if any(record.contralateral_previous is None for record in records):
        raise ValueError("Association metric batch contains an unpaired record")
    previous_boxes = np.asarray(
        [record.self_pair.previous.bbox_xyxy for record in records], dtype=np.float64
    )
    current_boxes = np.asarray(
        [record.self_pair.current.bbox_xyxy for record in records], dtype=np.float64
    )
    opposite_boxes = np.asarray(
        [record.contralateral_previous.bbox_xyxy for record in records],  # type: ignore[union-attr]
        dtype=np.float64,
    )
    other_iou, other_center, other_area = _bbox_metrics_batch(
        opposite_boxes, current_boxes
    )
    interhand_iou, interhand_center, _ = _bbox_metrics_batch(
        previous_boxes, opposite_boxes
    )
    self_iou = np.asarray(
        [record.self_pair.bbox_iou for record in records], dtype=np.float64
    )
    self_center = np.asarray(
        [record.self_pair.bbox_center_jump for record in records], dtype=np.float64
    )
    iou_margin = self_iou - other_iou
    center_margin = other_center - self_center
    own_viou = np.asarray(own_metrics["persistence_viou"], dtype=np.float64)
    other_viou = np.asarray(
        contralateral_metrics["persistence_viou"], dtype=np.float64
    )
    own_rmse = np.asarray(own_metrics["persistence_rmse"], dtype=np.float64)
    other_rmse = np.asarray(
        contralateral_metrics["persistence_rmse"], dtype=np.float64
    )
    finite_viou = np.isfinite(own_viou) & np.isfinite(other_viou)
    viou_preference = np.full(len(records), np.nan, dtype=np.float64)
    viou_preference[finite_viou] = (
        own_viou[finite_viou] > other_viou[finite_viou]
    ).astype(np.float64)
    return {
        "self_bbox_iou": self_iou,
        "contralateral_bbox_iou": other_iou,
        "bbox_iou_self_margin": iou_margin,
        "bbox_iou_prefers_self": (iou_margin > 0.0).astype(np.float64),
        "bbox_iou_ambiguous": (
            np.abs(iou_margin) <= config.association_iou_ambiguity_margin
        ).astype(np.float64),
        "self_bbox_center_jump": self_center,
        "contralateral_bbox_center_jump": other_center,
        "bbox_center_self_margin": center_margin,
        "bbox_center_prefers_self": (center_margin > 0.0).astype(np.float64),
        "bbox_center_ambiguous": (
            np.abs(center_margin) <= config.association_center_ambiguity_margin
        ).astype(np.float64),
        "previous_interhand_bbox_iou": interhand_iou,
        "previous_interhand_center_distance": interhand_center,
        "contralateral_bbox_abs_log_area_ratio": other_area,
        "pressure_viou_self_margin": own_viou - other_viou,
        "pressure_viou_prefers_self": viou_preference,
        "wrong_history_rmse_penalty": other_rmse - own_rmse,
    }


def _bilateral_detail(
    record: BilateralPairRecord,
    self_dynamics: str,
    own_metrics: Mapping[str, Optional[float]],
    contralateral_metrics: Optional[Mapping[str, Optional[float]]],
    association_metrics: Optional[Mapping[str, Optional[float]]],
) -> dict[str, Any]:
    pair = record.self_pair
    row: dict[str, Any] = {
        "dataset": pair.current.dataset,
        "split": pair.current.split,
        "sequence_key": pair.current.sequence_key,
        "side": pair.current.side_name,
        "query_alias": pair.current.query_alias,
        "requested_lag": record.requested_lag,
        "previous_sample_uid": pair.previous.sample_uid,
        "current_sample_uid": pair.current.sample_uid,
        "contralateral_previous_sample_uid": (
            None
            if record.contralateral_previous is None
            else record.contralateral_previous.sample_uid
        ),
        "transition": pair.transition,
        "self_dynamics_class": self_dynamics,
        "bbox_raw_track_id": pair.current.bbox_raw_track_id,
        "bbox_association_id": pair.current.bbox_association_id,
        "bbox_association_confidence": pair.current.bbox_association_confidence,
    }
    for name in _PAIRED_COMPARISON_METRICS:
        row[f"self_{name}"] = own_metrics.get(name)
        row[f"contralateral_{name}"] = (
            None if contralateral_metrics is None else contralateral_metrics.get(name)
        )
    if association_metrics:
        row.update(association_metrics)
    return row


def _accumulate_bilateral_metrics(
    history_table: BilateralMetricTable,
    association_table: BilateralMetricTable,
    classes: Counter,
    record: BilateralPairRecord,
    own_metrics: Mapping[str, Optional[float]],
    self_dynamics: str,
    other_metrics: Optional[Mapping[str, Optional[float]]],
    config: AuditConfig,
) -> Optional[dict[str, Optional[float]]]:
    pair = record.self_pair
    history_table.add("self_history", record, self_dynamics, own_metrics)
    classes[
        (
            pair.current.dataset,
            pair.current.split,
            str(record.requested_lag),
            pair.current.side_name,
            self_dynamics,
        )
    ] += 1
    if other_metrics is None:
        return None
    history_table.add("self_history_paired", record, self_dynamics, own_metrics)
    history_table.add(
        "contralateral_history", record, self_dynamics, other_metrics
    )
    history_table.add(
        "contralateral_minus_self",
        record,
        self_dynamics,
        _paired_metric_delta(other_metrics, own_metrics),
    )
    association_metrics = _bilateral_association_metrics(
        record, own_metrics, other_metrics, config
    )
    association_table.add(
        "anonymous_bbox_association",
        record,
        self_dynamics,
        association_metrics,
    )
    return association_metrics


def audit_bilateral_pressures(
    config: AuditConfig,
    palm_mask: np.ndarray,
    selected: Sequence[BilateralPairRecord],
    metric_engine: Optional[BatchedPressureMetricEngine] = None,
) -> tuple[
    BilateralMetricTable,
    BilateralMetricTable,
    Counter,
    list[dict[str, Any]],
    int,
    list[str],
    dict[str, Any],
]:
    history_table = BilateralMetricTable()
    association_table = BilateralMetricTable()
    classes: Counter = Counter()
    details: list[dict[str, Any]] = []
    errors: list[str] = []
    error_count = 0
    processed = 0
    pressure_started = time.time()
    pressure_read_seconds = 0.0
    pressure_metric_seconds = 0.0
    pressure_aggregate_seconds = 0.0
    pressure_metric_calls = 0
    pressure_metric_max_rows = 0
    expects_batched = config.pressure_metric_device.strip().lower() != "cpu"
    if expects_batched != (metric_engine is not None):
        raise RuntimeError("Pressure metric engine does not match the audit configuration")
    reader = PressureReader(
        config.max_open_hdf5,
        max_cached_rows=config.pressure_row_cache_size,
        palm_mask=palm_mask,
        cached_dtype=(np.float64 if metric_engine is None else metric_engine.numpy_dtype),
    )
    print(
        "[bilateral-pressure] metric backend="
        + (
            "numpy_scalar device=cpu dtype=float64 batch=1"
            if metric_engine is None
            else (
                f"torch_batched device={metric_engine.device} "
                f"dtype={config.pressure_metric_dtype} "
                f"read_batch_pairs={config.pressure_batch_size} "
                f"metric_chunk_rows="
                f"{config.pressure_metric_chunk_size or 'full'} "
                f"bulk_hdf5=true prefetch={str(config.pressure_prefetch).lower()}"
            )
        ),
        flush=True,
    )

    def report_progress() -> None:
        stats = reader.stats()
        elapsed_seconds = time.time() - pressure_started
        measured_seconds = (
            pressure_read_seconds
            + pressure_metric_seconds
            + pressure_aggregate_seconds
        )
        phase_summary = ""
        if metric_engine is not None and elapsed_seconds > 0.0:
            metric_layout = (
                f"; metric_calls={pressure_metric_calls:,}; "
                f"max_rows/call={pressure_metric_max_rows:,}"
            )
            if config.pressure_prefetch:
                phase_summary = (
                    "; active read/metric/aggregate="
                    f"{pressure_read_seconds / elapsed_seconds:.0%}/"
                    f"{pressure_metric_seconds / elapsed_seconds:.0%}/"
                    f"{pressure_aggregate_seconds / elapsed_seconds:.0%} (overlap)"
                    f"{metric_layout}"
                )
            else:
                other_seconds = max(0.0, elapsed_seconds - measured_seconds)
                phase_summary = (
                    "; phases read/metric/aggregate/other="
                    f"{pressure_read_seconds / elapsed_seconds:.0%}/"
                    f"{pressure_metric_seconds / elapsed_seconds:.0%}/"
                    f"{pressure_aggregate_seconds / elapsed_seconds:.0%}/"
                    f"{other_seconds / elapsed_seconds:.0%}"
                    f"{metric_layout}"
                )
        print(
            f"[bilateral-pressure] audited {processed:,} pairs; "
            f"rate={processed / max(elapsed_seconds, 1e-9):,.1f}/s; "
            f"cache_hit={stats['cache_hit_rate']:.1%}; "
            f"HDF5_rows/calls={stats['disk_reads']:,}/"
            f"{stats['hdf5_read_calls']:,}{phase_summary}",
            flush=True,
        )

    def record_error(record: BilateralPairRecord, exc: Exception) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 50:
            pair = record.self_pair
            errors.append(
                f"lag={record.requested_lag} "
                f"{pair.previous.sample_uid} -> {pair.current.sample_uid}: "
                f"{type(exc).__name__}: {exc}"
            )

    def accumulate_record(
        record: BilateralPairRecord,
        own_metrics: Mapping[str, Optional[float]],
        self_dynamics: str,
        other_metrics: Optional[Mapping[str, Optional[float]]],
    ) -> None:
        nonlocal processed
        association_metrics = _accumulate_bilateral_metrics(
            history_table,
            association_table,
            classes,
            record,
            own_metrics,
            self_dynamics,
            other_metrics,
            config,
        )
        processed += 1
        if len(details) < config.pair_csv_limit:
            details.append(
                _bilateral_detail(
                    record,
                    self_dynamics,
                    own_metrics,
                    other_metrics,
                    association_metrics,
                )
            )
        if config.progress_every and processed % config.progress_every == 0:
            report_progress()

    def load_batched(
        records_batch: Sequence[BilateralPairRecord],
    ) -> LoadedBilateralPressureBatch:
        phase_started = time.perf_counter()
        requested_samples: list[SampleRecord] = []
        for record in records_batch:
            requested_samples.extend(
                (record.self_pair.previous, record.self_pair.current)
            )
            if record.contralateral_previous is not None:
                requested_samples.append(record.contralateral_previous)
        loaded_rows = reader.read_many(requested_samples)
        cursor = 0
        valid_records: list[BilateralPairRecord] = []
        own_previous_rows: list[np.ndarray] = []
        current_rows: list[np.ndarray] = []
        other_positions: list[int] = []
        other_previous_rows: list[np.ndarray] = []
        other_current_rows: list[np.ndarray] = []
        other_pairs: list[PairRecord] = []
        read_errors: list[tuple[BilateralPairRecord, Exception]] = []
        for record in records_batch:
            pair = record.self_pair
            own_previous = loaded_rows[cursor]
            current = loaded_rows[cursor + 1]
            cursor += 2
            other_previous: np.ndarray | Exception | None = None
            if record.contralateral_previous is not None:
                other_previous = loaded_rows[cursor]
                cursor += 1
            row_error = next(
                (
                    value
                    for value in (own_previous, current, other_previous)
                    if isinstance(value, Exception)
                ),
                None,
            )
            if row_error is not None:
                read_errors.append((record, row_error))
                continue
            if not isinstance(own_previous, np.ndarray) or not isinstance(
                current, np.ndarray
            ):
                read_errors.append(
                    (record, TypeError("pressure batch returned a non-array row"))
                )
                continue
            position = len(valid_records)
            valid_records.append(record)
            own_previous_rows.append(own_previous)
            current_rows.append(current)
            if isinstance(other_previous, np.ndarray):
                other_positions.append(position)
                other_previous_rows.append(other_previous)
                other_current_rows.append(current)
                other_pairs.append(pair)
        if cursor != len(loaded_rows):
            raise RuntimeError("Pressure batch row layout was not consumed exactly")
        return LoadedBilateralPressureBatch(
            valid_records=valid_records,
            own_previous_rows=own_previous_rows,
            current_rows=current_rows,
            other_positions=other_positions,
            other_previous_rows=other_previous_rows,
            other_current_rows=other_current_rows,
            other_pairs=other_pairs,
            errors=read_errors,
            read_seconds=time.perf_counter() - phase_started,
        )

    def process_batched(loaded: LoadedBilateralPressureBatch) -> None:
        nonlocal processed
        nonlocal pressure_read_seconds
        nonlocal pressure_metric_seconds
        nonlocal pressure_aggregate_seconds
        nonlocal pressure_metric_calls
        nonlocal pressure_metric_max_rows
        if metric_engine is None:
            raise RuntimeError("batched processing requested without a metric engine")
        pressure_read_seconds += loaded.read_seconds
        for record, exc in loaded.errors:
            record_error(record, exc)
        valid_records = loaded.valid_records
        if not valid_records:
            return
        own_pairs = [record.self_pair for record in valid_records]
        combined_previous = loaded.own_previous_rows + loaded.other_previous_rows
        combined_current = loaded.current_rows + loaded.other_current_rows
        combined_pairs = own_pairs + loaded.other_pairs
        phase_started = time.perf_counter()
        metric_chunk_size = int(config.pressure_metric_chunk_size)
        if metric_chunk_size <= 0:
            metric_chunk_size = len(combined_pairs)
        metric_parts: dict[str, list[np.ndarray]] = {}
        combined_dynamics: list[str] = []
        for start in range(0, len(combined_pairs), metric_chunk_size):
            stop = min(start + metric_chunk_size, len(combined_pairs))
            chunk_metrics, chunk_dynamics = metric_engine.compute(
                combined_previous[start:stop],
                combined_current[start:stop],
                combined_pairs[start:stop],
            )
            if not metric_parts:
                metric_parts = {name: [] for name in chunk_metrics}
            elif tuple(metric_parts) != tuple(chunk_metrics):
                raise RuntimeError(
                    "Pressure metric columns changed between GPU microbatches"
                )
            for name, values in chunk_metrics.items():
                metric_parts[name].append(np.asarray(values))
            combined_dynamics.extend(chunk_dynamics)
            pressure_metric_calls += 1
            pressure_metric_max_rows = max(
                pressure_metric_max_rows, stop - start
            )
        combined_metrics = {
            name: np.concatenate(parts, axis=0)
            for name, parts in metric_parts.items()
        }
        if len(combined_dynamics) != len(combined_pairs) or any(
            len(values) != len(combined_pairs)
            for values in combined_metrics.values()
        ):
            raise RuntimeError(
                "GPU pressure metric microbatches were not reassembled exactly"
            )
        pressure_metric_seconds += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        own_count = len(valid_records)
        own_metrics = _slice_metric_columns(combined_metrics, 0, own_count)
        dynamics = combined_dynamics[:own_count]
        history_table.add_batch(
            "self_history", valid_records, dynamics, own_metrics
        )
        classes.update(
            (
                record.self_pair.current.dataset,
                record.self_pair.current.split,
                str(record.requested_lag),
                record.self_pair.current.side_name,
                self_dynamics,
            )
            for record, self_dynamics in zip(valid_records, dynamics)
        )

        other_metrics: dict[str, np.ndarray] = {}
        association_metrics: dict[str, np.ndarray] = {}
        if loaded.other_positions:
            paired_indices = np.asarray(loaded.other_positions, dtype=np.int64)
            paired_records = [
                valid_records[position] for position in loaded.other_positions
            ]
            paired_dynamics = [
                dynamics[position] for position in loaded.other_positions
            ]
            paired_own_metrics = {
                name: np.asarray(values)[paired_indices]
                for name, values in own_metrics.items()
            }
            other_metrics = _slice_metric_columns(
                combined_metrics, own_count, None
            )
            history_table.add_batch(
                "self_history_paired",
                paired_records,
                paired_dynamics,
                paired_own_metrics,
            )
            history_table.add_batch(
                "contralateral_history",
                paired_records,
                paired_dynamics,
                other_metrics,
            )
            history_table.add_batch(
                "contralateral_minus_self",
                paired_records,
                paired_dynamics,
                _paired_metric_delta_columns(other_metrics, paired_own_metrics),
            )
            association_metrics = _bilateral_association_metric_columns(
                paired_records, paired_own_metrics, other_metrics, config
            )
            association_table.add_batch(
                "anonymous_bbox_association",
                paired_records,
                paired_dynamics,
                association_metrics,
            )

        remaining_details = max(0, config.pair_csv_limit - len(details))
        if remaining_details:
            paired_lookup = {
                position: paired_index
                for paired_index, position in enumerate(loaded.other_positions)
            }
            for position, record in enumerate(valid_records[:remaining_details]):
                own_row = metric_engine.row(own_metrics, position)
                paired_index = paired_lookup.get(position)
                other_row = (
                    None
                    if paired_index is None
                    else metric_engine.row(other_metrics, paired_index)
                )
                association_row = (
                    None
                    if paired_index is None
                    else metric_engine.row(association_metrics, paired_index)
                )
                details.append(
                    _bilateral_detail(
                        record,
                        dynamics[position],
                        own_row,
                        other_row,
                        association_row,
                    )
                )

        previous_processed = processed
        processed += len(valid_records)
        pressure_aggregate_seconds += time.perf_counter() - phase_started
        if (
            config.progress_every
            and processed // config.progress_every
            > previous_processed // config.progress_every
        ):
            report_progress()

    try:
        records: Iterable[BilateralPairRecord]
        if config.max_bilateral_pressure_pairs == 0:
            records = (
                record
                for record in iter_bilateral_pairs(
                    config, pass_name="bilateral-pressure-full"
                )
                if record.self_pair.eligible
            )
        else:
            records = selected
        if metric_engine is None:
            for record in records:
                pair = record.self_pair
                try:
                    own_previous = reader.read(pair.previous)
                    current = reader.read(pair.current)
                    own_metrics, self_dynamics = pressure_pair_metrics(
                        own_previous,
                        current,
                        palm_mask,
                        pair,
                        config,
                        already_masked=True,
                    )
                    other_metrics = None
                    if record.contralateral_previous is not None:
                        other_previous = reader.read(record.contralateral_previous)
                        other_metrics, _ = pressure_pair_metrics(
                            other_previous,
                            current,
                            palm_mask,
                            pair,
                            config,
                            already_masked=True,
                        )
                except Exception as exc:
                    record_error(record, exc)
                    continue
                accumulate_record(
                    record, own_metrics, self_dynamics, other_metrics
                )
        else:
            def iter_record_batches() -> Iterator[tuple[BilateralPairRecord, ...]]:
                pending: list[BilateralPairRecord] = []
                for record in records:
                    pending.append(record)
                    if len(pending) >= config.pressure_batch_size:
                        yield tuple(pending)
                        pending.clear()
                if pending:
                    yield tuple(pending)

            batch_iterator = iter(iter_record_batches())
            first_batch = next(batch_iterator, None)
            if first_batch is not None and config.pressure_prefetch:
                # One producer owns every h5py call. The main thread consumes
                # the previous batch on CUDA while the next batch is read.
                with ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="pressure-reader"
                ) as executor:
                    read_future = executor.submit(load_batched, first_batch)
                    for next_batch in batch_iterator:
                        loaded = read_future.result()
                        read_future = executor.submit(load_batched, next_batch)
                        process_batched(loaded)
                    process_batched(read_future.result())
            elif first_batch is not None:
                process_batched(load_batched(first_batch))
                for next_batch in batch_iterator:
                    process_batched(load_batched(next_batch))
    finally:
        reader.close()
    reader_stats = reader.stats()
    pressure_elapsed = time.time() - pressure_started
    pressure_measured = (
        pressure_read_seconds
        + pressure_metric_seconds
        + pressure_aggregate_seconds
    )
    reader_stats.update(
        {
            "processed_pairs": processed,
            "elapsed_seconds": pressure_elapsed,
            "pairs_per_second": processed / max(pressure_elapsed, 1e-9),
            "metric_backend": (
                "numpy_scalar" if metric_engine is None else "torch_batched"
            ),
            "metric_device": (
                "cpu" if metric_engine is None else str(metric_engine.device)
            ),
            "metric_dtype": (
                "float64"
                if metric_engine is None
                else config.pressure_metric_dtype
            ),
            "metric_batch_size": (
                1 if metric_engine is None else config.pressure_batch_size
            ),
            "read_pair_batch_size": (
                1 if metric_engine is None else config.pressure_batch_size
            ),
            "metric_chunk_size": (
                1
                if metric_engine is None
                else config.pressure_metric_chunk_size
            ),
            "metric_calls": (
                processed if metric_engine is None else pressure_metric_calls
            ),
            "metric_max_rows_per_call": (
                1 if metric_engine is None else pressure_metric_max_rows
            ),
            "metric_aggregation": (
                "python_scalar" if metric_engine is None else "numpy_batched"
            ),
            "pressure_prefetch": bool(
                metric_engine is not None and config.pressure_prefetch
            ),
            "pressure_read_seconds": pressure_read_seconds,
            "pressure_metric_seconds": pressure_metric_seconds,
            "pressure_aggregate_seconds": pressure_aggregate_seconds,
            "pressure_pairing_and_other_seconds": (
                None
                if metric_engine is not None and config.pressure_prefetch
                else max(0.0, pressure_elapsed - pressure_measured)
            ),
            "pressure_phase_seconds_may_overlap": bool(
                metric_engine is not None and config.pressure_prefetch
            ),
        }
    )
    return (
        history_table,
        association_table,
        classes,
        details,
        error_count,
        errors,
        reader_stats,
    )


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        fieldnames = sorted({key for row in rows for key in row})
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def _structure_rows(counts: Counter) -> list[dict[str, Any]]:
    rows = []
    for (scope, metric), count in sorted(counts.items()):
        denominator_name = (
            "records"
            if metric in {"records", "timestamp_present", "source_frame_present", "bbox_score_present"}
            or metric.startswith("bbox_source/")
            or metric.startswith("pressure_source/")
            or metric.startswith("record_pressure_bin/")
            or metric.startswith("record_volume_bin/")
            else "adjacent_pairs"
        )
        if metric.startswith("eligible_pressure_bin/") or metric.startswith("eligible_volume_bin/"):
            denominator_name = "eligible"
        denominator = counts.get((scope, denominator_name), 0)
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "count": int(count),
                "fraction": None if denominator <= 0 else float(count / denominator),
                "denominator": denominator_name,
            }
        )
    return rows


def _class_rows(classes: Counter) -> list[dict[str, Any]]:
    totals = Counter()
    for (dataset, split, _), count in classes.items():
        totals[(dataset, split)] += count
    return [
        {
            "dataset": dataset,
            "split": split,
            "dynamics_class": dynamics,
            "count": count,
            "fraction": count / totals[(dataset, split)],
        }
        for (dataset, split, dynamics), count in sorted(classes.items())
    ]


def _bilateral_structure_rows(counts: Counter) -> list[dict[str, Any]]:
    rows = []
    for (scope, metric), count in sorted(counts.items()):
        denominator_name = "records"
        if metric == "raw_track_adjacent_same":
            denominator_name = "raw_track_adjacent_comparable"
        elif metric == "association_adjacent_same":
            denominator_name = "association_adjacent_comparable"
        elif metric.startswith("controlled_lag/"):
            parts = metric.split("/")
            denominator_name = f"controlled_lag/{parts[1]}"
        elif metric == "controlled_pairs":
            denominator_name = "controlled_pairs"
        denominator = counts.get((scope, denominator_name), 0)
        rows.append(
            {
                "scope": scope,
                "metric": metric,
                "count": int(count),
                "fraction": None if denominator <= 0 else float(count / denominator),
                "denominator": denominator_name,
            }
        )
    return rows


def _bilateral_class_rows(classes: Counter) -> list[dict[str, Any]]:
    totals = Counter()
    for dataset, split, lag, side, _dynamics in classes:
        totals[(dataset, split, lag, side)] += classes[
            (dataset, split, lag, side, _dynamics)
        ]
    return [
        {
            "dataset": dataset,
            "split": split,
            "lag": lag,
            "side": side,
            "dynamics_class": dynamics,
            "count": int(count),
            "fraction": float(count / totals[(dataset, split, lag, side)]),
        }
        for (dataset, split, lag, side, dynamics), count in sorted(classes.items())
    ]


def _bilateral_sampling_rows(
    strata: Counter, quotas: Mapping[tuple[str, ...], int]
) -> list[dict[str, Any]]:
    rows = []
    for key, count in sorted(strata.items()):
        rows.append(
            {
                "dataset": key[0],
                "split": key[1],
                "lag": key[2],
                "side": key[3],
                "transition": key[4],
                "pressure_bin": key[5],
                "volume_bin": key[6],
                "contralateral_availability": key[7],
                "eligible_pair_count": int(count),
                "requested_pressure_pair_count": int(quotas.get(key, 0)),
            }
        )
    return rows


def _sum_structure_metric(counts: Counter, metric: str) -> int:
    return int(
        sum(value for (_scope_name, name), value in counts.items() if name == metric)
    )


def _ratio_or_none(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator <= 0 else float(numerator / denominator)


def _format_metric(value: Optional[float], *, percent: bool = False) -> str:
    if value is None:
        return "unavailable"
    return f"{value:.2%}" if percent else f"{value:.4f}"


def _interpret_bilateral(
    structure: Mapping[str, Any],
    counts: Counter,
    history_table: BilateralMetricTable,
    association_table: BilateralMetricTable,
) -> list[str]:
    notes = []
    self_viou = history_table.mean("self_history_paired", "persistence_viou")
    other_viou = history_table.mean("contralateral_history", "persistence_viou")
    self_rmse = history_table.mean("self_history_paired", "persistence_rmse")
    other_rmse = history_table.mean("contralateral_history", "persistence_rmse")
    if self_viou is not None and other_viou is not None:
        viou_margin = self_viou - other_viou
        rmse_margin = (
            None if self_rmse is None or other_rmse is None else other_rmse - self_rmse
        )
        if viou_margin >= 0.05 and (rmse_margin is None or rmse_margin > 0.0):
            notes.append(
                "Same-instance pressure history clearly beats the other hand at the "
                f"same lag (V-IoU margin={viou_margin:.4f}, wrong-history RMSE "
                f"penalty={_format_metric(rmse_margin)}). A tactile-flow model must "
                "carry an anonymous per-instance state instead of one sequence-wide state."
            )
        elif viou_margin > 0.0:
            notes.append(
                "Same-instance history is better than the other hand, but the margin is "
                f"limited (V-IoU={viou_margin:.4f}). Identity-aware state remains useful, "
                "while much of the apparent persistence may be a shared interaction prior."
            )
        else:
            notes.append(
                "The other hand matches or exceeds same-instance persistence. Direct "
                "autoregressive tactile state is not causally supported until query "
                "association and paired-frame construction are fixed."
            )
    available = sum(
        value
        for (_scope_name, metric), value in counts.items()
        if metric.endswith("/eligible_contralateral_available")
    )
    controlled = int(structure["eligible_controlled_pair_count"])
    notes.append(
        "A synchronized opposite-hand counterfactual exists for "
        f"{_format_metric(_ratio_or_none(available, controlled), percent=True)} of "
        "controlled pairs. Missing counterfactuals are reported rather than filled."
    )
    bbox_preference = association_table.mean(
        "anonymous_bbox_association", "bbox_iou_prefers_self"
    )
    bbox_ambiguity = association_table.mean(
        "anonymous_bbox_association", "bbox_iou_ambiguous"
    )
    if bbox_preference is not None:
        notes.append(
            "Previous-box IoU selects the same canonical query in "
            f"{bbox_preference:.2%} of paired cases; "
            f"{_format_metric(bbox_ambiguity, percent=True)} "
            "are ambiguous under the configured margin. This is an association oracle "
            "audit, not evidence that semantic left/right should enter the network."
        )
    raw_purity = structure["raw_track_identity"].get("weighted_alias_purity")
    association_purity = structure["association_identity"].get(
        "weighted_alias_purity"
    )
    notes.append(
        "BBox provenance identifier purity is "
        f"raw-track={_format_metric(raw_purity, percent=True)}, "
        f"association={_format_metric(association_purity, percent=True)}. These IDs may "
        "maintain anonymous A/B state, but are not treated as handedness ground truth."
    )
    notes.append(
        "The current samples are canonicalized by flipping left-hand RGB into the "
        "right-hand view. Side labels are used only to construct this audit's paired "
        "counterfactual; a future model should reset and route state by tracked instance."
    )
    return notes


def _config_payload(config: AuditConfig) -> dict[str, Any]:
    payload = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if field.name == "manifests":
            value = [str(path) for path in value]
        elif isinstance(value, Path):
            value = str(value)
        payload[field.name] = value
    return payload


def _interpret(table: MetricTable, classes: Counter) -> list[str]:
    notes = []
    relative_gain = table.mean("overall", "persistence_relative_rmse_gain_vs_zero")
    distribution = table.mean("overall", "distribution_viou")
    if relative_gain is not None:
        if relative_gain >= 0.25:
            notes.append(
                "Short-history pressure persistence is substantially more informative "
                "than an empty map; temporal state is plausible as a main "
                "representation input."
            )
        elif relative_gain > 0.0:
            notes.append(
                "Previous pressure has positive but limited persistence skill; a "
                "temporal branch should model updates and confidence rather than copy "
                "the previous map."
            )
        else:
            notes.append(
                "Previous pressure does not beat the empty-map RMSE baseline on the "
                "audited pairs; direct autoregressive pressure input is not supported."
            )
    if distribution is not None:
        if distribution >= 0.6:
            notes.append(
                "Normalized pressure location is often preserved across short gaps, "
                "supporting a transport/update formulation on the canonical mesh."
            )
        else:
            notes.append(
                "Normalized pressure location changes strongly; explicit source/sink "
                "or uncertainty is necessary and brightness-style conservation would "
                "be unsafe."
            )
    overall_total = sum(
        count for (dataset, split, _), count in classes.items()
        if dataset == "overall" and split == "overall"
    )
    if overall_total:
        source_sink = classes.get(("overall", "overall", "source_sink_dominant"), 0)
        notes.append(
            f"Source/sink-dominant pairs comprise "
            f"{source_sink / overall_total:.2%} of audited pressure pairs; tactile "
            "flow must represent loading and release, not only spatial transport."
        )
    return notes


def run_bilateral_audit(config: AuditConfig) -> dict[str, Any]:
    started = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    palm_mask, palm_provenance = _load_palm_mask()
    metric_engine = (
        None
        if config.pressure_metric_device.strip().lower() == "cpu"
        else BatchedPressureMetricEngine(config)
    )
    if metric_engine is not None:
        metric_engine.validate_against_scalar(config)
    structure_summary, structure_counts, strata = scan_bilateral_structure(config)
    if structure_summary["sequence_frame_order_regression_count"]:
        raise RuntimeError(
            "Bilateral audit requires each query manifest to be ordered by sequence "
            "and temporal frame, but observed "
            f"{structure_summary['sequence_frame_order_regression_count']:,} frame-order "
            "regressions. Rebuild/sort the authoritative query manifest before using "
            "lag or contralateral-history statistics."
        )
    selected, quotas = select_bilateral_pairs(config, strata)
    (
        history_table,
        association_table,
        classes,
        details,
        error_count,
        errors,
        pressure_reader_stats,
    ) = audit_bilateral_pressures(
        config, palm_mask, selected, metric_engine=metric_engine
    )
    pressure_count = history_table.count("self_history", "persistence_rmse")
    contralateral_count = history_table.count(
        "contralateral_history", "persistence_rmse"
    )
    interpretation = _interpret_bilateral(
        structure_summary, structure_counts, history_table, association_table
    )
    manifest_provenance = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in config.manifests
    ]
    class_rows = _bilateral_class_rows(classes)
    structure_rows = _bilateral_structure_rows(structure_counts)
    history_rows = history_table.rows()
    association_rows = association_table.rows()
    sampling_rows = _bilateral_sampling_rows(strata, quotas)
    payload = {
        "schema": BILATERAL_AUDIT_SCHEMA,
        "structure": structure_summary,
        "pressure_pair_count": pressure_count,
        "contralateral_pressure_pair_count": contralateral_count,
        "pressure_pair_mode": (
            "all_eligible"
            if config.max_bilateral_pressure_pairs == 0
            else "deterministic_stratified_sample"
        ),
        "pressure_read_error_count": error_count,
        "pressure_read_errors_preview": errors,
        "pressure_reader": pressure_reader_stats,
        "interpretation": interpretation,
        "history_metrics": history_rows,
        "association_metrics": association_rows,
        "dynamics_classes": class_rows,
        "provenance": {
            "manifests": manifest_provenance,
            "palm": palm_provenance,
            "script_sha256": sha256_file(Path(__file__)),
            "config": _config_payload(config),
            "semantic_contract": {
                "canonicalization": (
                    "left-hand RGB is horizontally flipped into the canonical "
                    "right-hand view by the current dataset pipeline"
                ),
                "identity": (
                    "query side/provenance is used only for audit pairing; a learned "
                    "temporal state should be keyed by anonymous tracked instance"
                ),
                "contralateral_control": (
                    "opposite canonical query at the exact previous source-frame index"
                ),
            },
            "elapsed_seconds": time.time() - started,
        },
    }
    write_json_atomic(config.output_dir / "summary.json", payload)
    _atomic_csv(config.output_dir / "bilateral_structure.csv", structure_rows)
    _atomic_csv(config.output_dir / "bilateral_history_metrics.csv", history_rows)
    _atomic_csv(
        config.output_dir / "bilateral_association_metrics.csv", association_rows
    )
    _atomic_csv(config.output_dir / "bilateral_dynamics_classes.csv", class_rows)
    _atomic_csv(config.output_dir / "bilateral_sampling_strata.csv", sampling_rows)
    if details:
        _atomic_csv(config.output_dir / "sampled_bilateral_pairs.csv", details)

    self_viou = history_table.mean("self_history_paired", "persistence_viou")
    other_viou = history_table.mean("contralateral_history", "persistence_viou")
    self_rmse = history_table.mean("self_history_paired", "persistence_rmse")
    other_rmse = history_table.mean("contralateral_history", "persistence_rmse")
    pressure_other_seconds = pressure_reader_stats[
        "pressure_pairing_and_other_seconds"
    ]
    pressure_other_text = (
        "overlapped"
        if pressure_other_seconds is None
        else f"{pressure_other_seconds:.1f}"
    )
    lines = [
        "Bilateral tactile dynamics audit",
        f"records: {structure_summary['record_count']:,}",
        f"controlled lag pairs: {structure_summary['controlled_pair_count']:,}",
        (
            "eligible controlled lag pairs: "
            f"{structure_summary['eligible_controlled_pair_count']:,}"
        ),
        f"audited self-history pressure pairs: {pressure_count:,}",
        f"paired contralateral controls: {contralateral_count:,}",
        (
            "pressure row cache: "
            f"{pressure_reader_stats['cache_hits']:,}/"
            f"{pressure_reader_stats['read_requests']:,} hits "
            f"({pressure_reader_stats['cache_hit_rate']:.2%}); "
            f"HDF5 rows/calls={pressure_reader_stats['disk_reads']:,}/"
            f"{pressure_reader_stats['hdf5_read_calls']:,}; "
            f"throughput={pressure_reader_stats['pairs_per_second']:,.1f} pairs/s"
        ),
        (
            "pressure phase seconds (read / metric / aggregate / other): "
            f"{pressure_reader_stats['pressure_read_seconds']:.1f} / "
            f"{pressure_reader_stats['pressure_metric_seconds']:.1f} / "
            f"{pressure_reader_stats['pressure_aggregate_seconds']:.1f} / "
            f"{pressure_other_text}"
        ),
        (
            "pressure batching (read pairs / configured metric rows / "
            "observed max rows / metric calls): "
            f"{pressure_reader_stats['read_pair_batch_size']:,} / "
            f"{pressure_reader_stats['metric_chunk_size']:,} / "
            f"{pressure_reader_stats['metric_max_rows_per_call']:,} / "
            f"{pressure_reader_stats['metric_calls']:,}"
        ),
        (
            "paired same-instance V-IoU/RMSE: "
            f"{_format_metric(self_viou)} / {_format_metric(self_rmse)}"
        ),
        (
            "contralateral-history V-IoU/RMSE: "
            f"{_format_metric(other_viou)} / {_format_metric(other_rmse)}"
        ),
        "",
        "Interpretation:",
        *[f"- {note}" for note in interpretation],
        "",
        "Important: this is a model-free identity/lag audit. It does not establish "
        "that semantic handedness should enter the tactile model or that a temporal "
        "network beats the RGB baseline.",
    ]
    (config.output_dir / "summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    print(f"Reports: {config.output_dir}", flush=True)
    return payload


def run_audit(config: AuditConfig) -> dict[str, Any]:
    started = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    palm_mask, palm_provenance = _load_palm_mask()
    structure_summary, combined_counts = scan_structure(config)
    structure_counts = _extract_structure_counts(combined_counts)
    strata = _extract_strata_counts(combined_counts)
    marginal_shifts = _marginal_shift_rows(
        structure_counts, structure_summary["scopes"]
    )
    structure_summary["target_marginal_shifts"] = marginal_shifts
    embedded_sam3_count = sum(
        structure_counts.get((scope, "bbox_source/sam3_bbox_source_v1"), 0)
        for scope in structure_summary["scopes"]
    )
    embedded_sam3_fraction = (
        embedded_sam3_count / structure_summary["record_count"]
        if structure_summary["record_count"]
        else None
    )
    structure_summary["embedded_sam3_bbox_provenance_fraction"] = embedded_sam3_fraction
    selected = select_pairs(config, strata)
    quotas = (
        {key: value for key, value in strata.items()}
        if config.max_pressure_pairs == 0
        else _largest_remainder(strata, config.max_pressure_pairs)
    )
    table, classes, details, error_count, errors, pressure_reader_stats = audit_pressures(
        config, palm_mask, selected
    )
    manifest_provenance = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in config.manifests
    ]
    pressure_count = table.values.get("overall", {}).get("persistence_rmse", RunningMetric()).count
    interpretation = _interpret(table, classes)
    maximum_target_tv = max(
        (row["total_variation_distance"] for row in marginal_shifts), default=0.0
    )
    if maximum_target_tv <= 0.02:
        interpretation.append(
            "Eligible target frames closely preserve the original pressure/volume "
            "marginals; target-first temporal sampling should require little "
            "reweighting."
        )
    else:
        interpretation.append(
            "Temporal eligibility shifts at least one target pressure/volume marginal "
            f"(maximum TV={maximum_target_tv:.3f}); future training should sample "
            "targets from the original frame-uniform distribution and attach context "
            "conditionally."
        )
    if embedded_sam3_fraction is not None and embedded_sam3_fraction < 0.99:
        interpretation.append(
            "Some query rows do not embed SAM3 bbox provenance. Pressure/timestamp "
            "conclusions remain valid, but bbox-stability statistics describe the "
            "HDF5 manifest rows rather than any runtime bbox overlay."
        )
    payload = {
        "schema": AUDIT_SCHEMA,
        "structure": structure_summary,
        "pressure_pair_count": pressure_count,
        "pressure_pair_mode": (
            "all_eligible" if config.max_pressure_pairs == 0 else "deterministic_stratified_sample"
        ),
        "pressure_read_error_count": error_count,
        "pressure_read_errors_preview": errors,
        "pressure_reader": pressure_reader_stats,
        "interpretation": interpretation,
        "metrics": table.rows(),
        "dynamics_classes": _class_rows(classes),
        "provenance": {
            "manifests": manifest_provenance,
            "palm": palm_provenance,
            "script_sha256": sha256_file(Path(__file__)),
            "config": _config_payload(config),
            "elapsed_seconds": time.time() - started,
        },
    }
    write_json_atomic(config.output_dir / "summary.json", payload)
    _atomic_csv(config.output_dir / "structure.csv", _structure_rows(structure_counts))
    _atomic_csv(config.output_dir / "target_marginal_shifts.csv", marginal_shifts)
    _atomic_csv(config.output_dir / "pressure_metrics.csv", table.rows())
    _atomic_csv(config.output_dir / "dynamics_classes.csv", _class_rows(classes))
    _atomic_csv(
        config.output_dir / "pressure_sampling_strata.csv",
        [
            {
                "dataset": key[0],
                "split": key[1],
                "time_gap_bin": key[2],
                "transition": key[3],
                "eligible_pair_count": int(count),
                "requested_pressure_pair_count": int(quotas.get(key, 0)),
            }
            for key, count in sorted(strata.items())
        ],
    )
    if details:
        _atomic_csv(config.output_dir / "sampled_pressure_pairs.csv", details)
    lines = [
        "Tactile dynamics data audit",
        f"records: {structure_summary['record_count']:,}",
        f"query tracks: {structure_summary['query_track_count']:,}",
        f"adjacent pairs: {structure_summary['adjacent_pair_count']:,}",
        f"temporal eligible pairs: {structure_summary['temporal_eligible_pair_count']:,}",
        f"audited pressure pairs: {pressure_count:,}",
        (
            "pressure row cache: "
            f"{pressure_reader_stats['cache_hits']:,}/"
            f"{pressure_reader_stats['read_requests']:,} hits "
            f"({pressure_reader_stats['cache_hit_rate']:.2%}); "
            f"HDF5 row reads={pressure_reader_stats['disk_reads']:,}; "
            f"throughput={pressure_reader_stats['pairs_per_second']:,.1f} pairs/s"
        ),
        (
            "embedded SAM3 bbox provenance fraction: unavailable"
            if embedded_sam3_fraction is None
            else f"embedded SAM3 bbox provenance fraction: {embedded_sam3_fraction:.2%}"
        ),
        "",
        "Interpretation:",
        *[f"- {note}" for note in payload["interpretation"]],
        "",
        "Important: this audit measures data support and oracle persistence. It does "
        "not prove that a learned temporal model beats the current RGB model.",
    ]
    (config.output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Reports: {config.output_dir}", flush=True)
    return payload


def _resolve_cli_manifests(args: argparse.Namespace) -> tuple[Path, ...]:
    explicit = []
    for value in args.manifest:
        explicit.extend(item.strip() for item in str(value).split(",") if item.strip())
    if explicit:
        resolved = tuple(Path(value).expanduser().resolve(strict=True) for value in explicit)
        if len(resolved) != len(set(resolved)):
            raise ValueError("The same query manifest was supplied more than once")
        return resolved
    resolver_args = argparse.Namespace(
        dataset=args.dataset,
        processed_root=args.processed_root,
        splits=args.splits,
        create_missing=True,
        lock_timeout=args.manifest_lock_timeout,
        print_paths=False,
    )
    return tuple(resolve_manifests(resolver_args, emit_paths=False))


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "controlled lags must be comma-separated positive integers"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "controlled lags must contain at least one positive integer"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("adjacent", "bilateral"),
        default="adjacent",
        help="Original adjacent-pair audit or controlled-lag bilateral audit.",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Explicit query JSONL; repeat or comma-separate.",
    )
    parser.add_argument("--dataset", choices=("touchanything", "opentouch"), default="touchanything")
    parser.add_argument("--splits", default="train,val,test_seen,test_unseen")
    parser.add_argument("--processed-root", default="")
    parser.add_argument("--manifest-lock-timeout", type=float, default=3600.0)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--max-pressure-pairs",
        type=int,
        default=50000,
        help="Deterministic pressure-pair sample; 0 audits all eligible pairs.",
    )
    parser.add_argument(
        "--controlled-lags",
        type=_parse_positive_ints,
        default=(1, 2, 4, 8),
        help="Exact source-frame lags for bilateral mode (default: 1,2,4,8).",
    )
    parser.add_argument(
        "--max-bilateral-pressure-pairs",
        type=int,
        default=200000,
        help=(
            "Deterministic bilateral pressure-pair sample across all lags; "
            "0 audits every eligible pair."
        ),
    )
    parser.add_argument("--max-frame-gap", type=int, default=8)
    parser.add_argument("--max-time-gap", type=float, default=0.5)
    parser.add_argument("--allow-missing-timestamps", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stable-bbox-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-bbox-iou", type=float, default=0.05)
    parser.add_argument("--max-bbox-center-jump", type=float, default=0.5)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=2.0)
    parser.add_argument(
        "--association-iou-ambiguity-margin", type=float, default=0.05
    )
    parser.add_argument(
        "--association-center-ambiguity-margin", type=float, default=0.05
    )
    parser.add_argument("--contact-threshold", type=float, default=0.10)
    parser.add_argument("--active-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--max-open-hdf5", type=int, default=4)
    parser.add_argument(
        "--pressure-row-cache-size",
        type=int,
        default=512,
        help=(
            "Number of pre-masked pressure rows retained in an LRU cache "
            "(default: 512; 0 disables)."
        ),
    )
    parser.add_argument(
        "--pressure-metric-device",
        default="cpu",
        help=(
            "Pressure metric backend: cpu preserves the scalar NumPy path; "
            "auto, cuda, or cuda:N enables batched Torch metrics in bilateral mode."
        ),
    )
    parser.add_argument(
        "--pressure-batch-size",
        type=int,
        default=1024,
        help=(
            "Outer pair batch used for HDF5 bulk reads and prefetch "
            "(default: 1024)."
        ),
    )
    parser.add_argument(
        "--pressure-metric-chunk-size",
        type=int,
        default=0,
        help=(
            "Maximum combined self/contralateral rows per Torch metric call; "
            "0 processes the whole outer batch at once."
        ),
    )
    parser.add_argument(
        "--pressure-metric-dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Torch metric precision; float32 is the fast audit default.",
    )
    parser.add_argument(
        "--pressure-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Overlap one HDF5 bulk-read batch with CUDA metric computation "
            "(default: enabled)."
        ),
    )
    parser.add_argument("--pair-csv-limit", type=int, default=100000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.max_pressure_pairs < 0
        or args.max_bilateral_pressure_pairs < 0
        or args.max_frame_gap <= 0
        or args.max_time_gap <= 0
    ):
        raise ValueError("pair count and temporal gap arguments must be non-negative/positive")
    if args.pressure_row_cache_size < 0:
        raise ValueError("--pressure-row-cache-size must be non-negative")
    if args.pressure_batch_size <= 0:
        raise ValueError("--pressure-batch-size must be positive")
    if args.pressure_metric_chunk_size < 0:
        raise ValueError("--pressure-metric-chunk-size must be non-negative")
    metric_device = args.pressure_metric_device.strip().lower()
    valid_metric_device = metric_device in {"cpu", "auto", "cuda"} or (
        metric_device.startswith("cuda:")
        and metric_device.removeprefix("cuda:").isdigit()
    )
    if not valid_metric_device:
        raise ValueError(
            "--pressure-metric-device must be cpu, auto, cuda, or cuda:N"
        )
    if args.mode != "bilateral" and metric_device != "cpu":
        raise ValueError(
            "Batched Torch pressure metrics currently require --mode bilateral"
        )
    if args.mode == "bilateral" and args.max_frame_gap < max(args.controlled_lags):
        raise ValueError(
            "--max-frame-gap must be at least the largest --controlled-lags value"
        )
    if not 0.0 <= args.min_bbox_iou <= 1.0:
        raise ValueError("--min-bbox-iou must lie in [0,1]")
    if args.max_bbox_center_jump <= 0.0 or args.max_bbox_area_ratio < 1.0:
        raise ValueError("bbox stability thresholds are invalid")
    if (
        args.association_iou_ambiguity_margin < 0.0
        or args.association_center_ambiguity_margin < 0.0
    ):
        raise ValueError("association ambiguity margins must be non-negative")
    manifests = _resolve_cli_manifests(args)
    if not manifests:
        raise RuntimeError("No query manifests were resolved")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT
        / (
            f"{_safe_component(args.dataset)}_bilateral_v2"
            if args.mode == "bilateral"
            else f"{_safe_component(args.dataset)}_current"
        )
    ).resolve(strict=False)
    data_root = Path(args.processed_root).expanduser().resolve(strict=True) if args.processed_root else None
    config = AuditConfig(
        manifests=manifests,
        data_root=data_root,
        output_dir=output_dir,
        max_pressure_pairs=args.max_pressure_pairs,
        max_frame_gap=args.max_frame_gap,
        max_time_gap=args.max_time_gap,
        allow_missing_timestamps=args.allow_missing_timestamps,
        stable_bbox_only=args.stable_bbox_only,
        min_bbox_iou=args.min_bbox_iou,
        max_bbox_center_jump=args.max_bbox_center_jump,
        max_bbox_area_ratio=args.max_bbox_area_ratio,
        contact_threshold=args.contact_threshold,
        active_threshold=args.active_threshold,
        seed=args.seed,
        progress_every=args.progress_every,
        max_open_hdf5=args.max_open_hdf5,
        pair_csv_limit=args.pair_csv_limit,
        mode=args.mode,
        controlled_lags=args.controlled_lags,
        max_bilateral_pressure_pairs=args.max_bilateral_pressure_pairs,
        association_iou_ambiguity_margin=args.association_iou_ambiguity_margin,
        association_center_ambiguity_margin=args.association_center_ambiguity_margin,
        pressure_row_cache_size=args.pressure_row_cache_size,
        pressure_metric_device=metric_device,
        pressure_batch_size=args.pressure_batch_size,
        pressure_metric_chunk_size=args.pressure_metric_chunk_size,
        pressure_metric_dtype=args.pressure_metric_dtype,
        pressure_prefetch=args.pressure_prefetch,
    )
    if config.mode == "bilateral":
        run_bilateral_audit(config)
    else:
        run_audit(config)


if __name__ == "__main__":
    main()
