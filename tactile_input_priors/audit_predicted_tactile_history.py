#!/usr/bin/env python3
"""Audit whether cached RGB predictions provide useful temporal state.

The validation split selects one convex history coefficient per exact lag by
minimizing vertex-weighted squared error. The coefficients are then replayed
unchanged on validation, seen, and unseen data. Controls distinguish correctly
routed history from generic smoothing or pressure suppression.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.audit_tactile_dynamics import (  # noqa: E402
    AuditConfig,
    BatchedPressureMetricEngine,
    BilateralMetricTable,
    BilateralPairRecord,
    PairRecord,
    PressureReader,
    SampleRecord,
    _atomic_csv,
    _load_palm_mask,
    _pair_from_samples,
    iter_bilateral_pairs,
    iter_samples,
)
from tactile_input_priors.hdf5_manifest import (  # noqa: E402
    sha256_file,
    write_json_atomic,
)


SCHEMA = "predicted_tactile_history_replay_v2"


def _parse_lags(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("lags must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("lags must contain positive integers")
    return result


def _safe_float(value: Optional[float]) -> Optional[float]:
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _format_metric(value: Optional[float], digits: int) -> str:
    value = _safe_float(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


class PredictionArchive:
    """In-memory view of exact prediction shards with strict provenance checks."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve(strict=True)
        config_path = self.root / "prediction_config.json"
        complete_path = self.root / "_COMPLETE"
        if not config_path.is_file() or not complete_path.is_file():
            raise FileNotFoundError(f"Incomplete prediction export: {self.root}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if self.config.get("schema") not in {
            "tactile_exact_prediction_shards_v1",
            "tactile_exact_prediction_shards_v2",
        }:
            raise ValueError(f"Unsupported prediction schema: {self.config.get('schema')!r}")
        if self.config.get("status") != "complete":
            raise ValueError(f"Prediction export is not complete: {self.root}")
        self.manifest = Path(self.config["sample_records"]).expanduser().resolve(strict=True)
        manifest_sha = sha256_file(self.manifest)
        if manifest_sha != str(self.config["sample_records_sha256"]):
            raise RuntimeError(f"Prediction manifest changed after export: {self.manifest}")
        if complete_path.read_text(encoding="utf-8").strip() != manifest_sha:
            raise RuntimeError(f"Prediction completion marker is stale: {self.root}")

        shard_paths = []
        for raw_path in self.config.get("shards", ()):
            path = Path(raw_path).expanduser()
            if not path.is_file():
                path = self.root / "shards" / path.name
            shard_paths.append(path.resolve(strict=True))
        if not shard_paths:
            raise RuntimeError(f"Prediction export has no shards: {self.root}")

        self.predictions: list[np.ndarray] = []
        self.locations: dict[str, tuple[int, int]] = {}
        self.artifact_indices: dict[str, int] = {}
        record_count = int(self.config["record_count"])
        self.shard_by_artifact_index = np.full(record_count, -1, dtype=np.int16)
        self.row_by_artifact_index = np.full(record_count, -1, dtype=np.int32)
        vertex_indices: Optional[np.ndarray] = None
        observed_indices: list[np.ndarray] = []
        for shard_index, shard_path in enumerate(shard_paths):
            with np.load(shard_path, allow_pickle=False) as shard:
                indices = np.asarray(shard["indices"], dtype=np.int64)
                sample_uids = np.asarray(shard["sample_uids"], dtype=str)
                predictions = np.asarray(shard["predictions"], dtype=np.float16)
                if "vertex_indices" in shard:
                    current_vertices = np.asarray(shard["vertex_indices"], dtype=np.int32)
                else:
                    current_vertices = np.arange(predictions.shape[1], dtype=np.int32)
            if predictions.ndim != 2 or predictions.shape[0] != len(indices):
                raise RuntimeError(f"Malformed prediction shard: {shard_path}")
            if len(sample_uids) != len(indices):
                raise RuntimeError(f"UID count differs from prediction rows: {shard_path}")
            if len(current_vertices) != predictions.shape[1]:
                raise RuntimeError(f"Vertex count differs from prediction width: {shard_path}")
            if vertex_indices is None:
                vertex_indices = current_vertices
            elif not np.array_equal(vertex_indices, current_vertices):
                raise RuntimeError("Prediction shards use different vertex indices")
            self.predictions.append(predictions)
            observed_indices.append(indices)
            for row_index, (sample_uid, artifact_index) in enumerate(
                zip(sample_uids, indices)
            ):
                key = str(sample_uid)
                if key in self.locations:
                    raise RuntimeError(f"Duplicate prediction sample_uid: {key}")
                self.locations[key] = (shard_index, row_index)
                self.artifact_indices[key] = int(artifact_index)
                if artifact_index < 0 or artifact_index >= record_count:
                    raise RuntimeError(
                        f"Prediction artifact index is out of range: {artifact_index}"
                    )
                if self.shard_by_artifact_index[artifact_index] >= 0:
                    raise RuntimeError(
                        f"Duplicate prediction artifact index: {artifact_index}"
                    )
                self.shard_by_artifact_index[artifact_index] = shard_index
                self.row_by_artifact_index[artifact_index] = row_index

        if vertex_indices is None:
            raise RuntimeError("Prediction export has no vertex definition")
        self.vertex_indices = vertex_indices
        indices = np.concatenate(observed_indices)
        expected = np.arange(int(self.config["record_count"]), dtype=np.int64)
        if not np.array_equal(np.sort(indices), expected):
            raise RuntimeError("Prediction shards do not cover every manifest row exactly once")
        if len(self.locations) != len(expected):
            raise RuntimeError("Prediction UID coverage differs from artifact index coverage")
        if np.any(self.shard_by_artifact_index < 0) or np.any(
            self.row_by_artifact_index < 0
        ):
            raise RuntimeError("Prediction artifact lookup contains uncovered indices")
        expected_vertex_sha = str(self.config.get("vertex_indices_sha256") or "")
        actual_vertex_sha = hashlib.sha256(
            np.ascontiguousarray(self.vertex_indices).tobytes()
        ).hexdigest()
        if expected_vertex_sha and expected_vertex_sha != actual_vertex_sha:
            raise RuntimeError("Prediction vertex-index hash differs from export provenance")

    @property
    def split(self) -> str:
        with self.manifest.open("rb") as handle:
            for line in handle:
                if line.strip():
                    return str(json.loads(line).get("split", ""))
        raise RuntimeError(f"Empty prediction manifest: {self.manifest}")

    def lookup(self, sample_uid: str, *, dtype=np.float32) -> np.ndarray:
        try:
            shard_index, row_index = self.locations[str(sample_uid)]
        except KeyError as exc:
            raise KeyError(f"Prediction missing sample_uid={sample_uid!r}") from exc
        return self.predictions[shard_index][row_index].astype(dtype, copy=False)

    def lookup_many(
        self, sample_uids: Sequence[str], *, dtype=np.float32
    ) -> np.ndarray:
        output = np.empty((len(sample_uids), len(self.vertex_indices)), dtype=dtype)
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_index, sample_uid in enumerate(sample_uids):
            try:
                shard_index, row_index = self.locations[str(sample_uid)]
            except KeyError as exc:
                raise KeyError(f"Prediction missing sample_uid={sample_uid!r}") from exc
            grouped[shard_index].append((output_index, row_index))
        for shard_index, entries in grouped.items():
            output_indices = np.asarray([item[0] for item in entries], dtype=np.int64)
            row_indices = np.asarray([item[1] for item in entries], dtype=np.int64)
            output[output_indices] = self.predictions[shard_index][row_indices]
        return output

    def artifact_index(self, sample_uid: str) -> int:
        try:
            return self.artifact_indices[str(sample_uid)]
        except KeyError as exc:
            raise KeyError(f"Prediction missing sample_uid={sample_uid!r}") from exc

    def lookup_artifact_indices(
        self, artifact_indices: Sequence[int] | np.ndarray, *, dtype=np.float32
    ) -> np.ndarray:
        indices = np.asarray(artifact_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("artifact_indices must be one-dimensional")
        if len(indices) and (indices.min() < 0 or indices.max() >= len(self.locations)):
            raise IndexError("Prediction artifact index is out of range")
        output = np.empty((len(indices), len(self.vertex_indices)), dtype=dtype)
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for output_index, artifact_index in enumerate(indices):
            shard_index = int(self.shard_by_artifact_index[artifact_index])
            row_index = int(self.row_by_artifact_index[artifact_index])
            grouped[shard_index].append((output_index, row_index))
        for shard_index, entries in grouped.items():
            output_indices = np.asarray([item[0] for item in entries], dtype=np.int64)
            row_indices = np.asarray([item[1] for item in entries], dtype=np.int64)
            output[output_indices] = self.predictions[shard_index][row_indices]
        return output


def _audit_config(
    archive: PredictionArchive,
    output_dir: Path,
    lags: tuple[int, ...],
    args: argparse.Namespace,
) -> AuditConfig:
    return AuditConfig(
        manifests=(archive.manifest,),
        data_root=None,
        output_dir=output_dir,
        max_pressure_pairs=0,
        max_frame_gap=max(lags),
        max_time_gap=float(args.max_time_gap),
        allow_missing_timestamps=False,
        stable_bbox_only=False,
        min_bbox_iou=float(args.min_bbox_iou),
        max_bbox_center_jump=float(args.max_bbox_center_jump),
        max_bbox_area_ratio=float(args.max_bbox_area_ratio),
        contact_threshold=float(args.contact_threshold),
        active_threshold=float(args.active_threshold),
        seed=int(args.seed),
        progress_every=int(args.progress_every),
        max_open_hdf5=int(args.max_open_hdf5),
        pair_csv_limit=0,
        mode="bilateral",
        controlled_lags=lags,
        max_bilateral_pressure_pairs=0,
        association_iou_ambiguity_margin=0.05,
        association_center_ambiguity_margin=0.05,
        pressure_row_cache_size=int(args.pressure_row_cache_size),
        pressure_metric_device=str(args.metric_device),
        pressure_batch_size=int(args.pair_batch_size),
        pressure_metric_chunk_size=int(args.pair_batch_size),
        pressure_metric_dtype="float32",
        pressure_prefetch=False,
    )


def _iter_batches(values: Iterable[Any], batch_size: int) -> Iterator[list[Any]]:
    pending = []
    for value in values:
        pending.append(value)
        if len(pending) >= batch_size:
            yield pending
            pending = []
    if pending:
        yield pending


def _require_pressure_rows(values: Sequence[np.ndarray | Exception]) -> list[np.ndarray]:
    result = []
    for value in values:
        if isinstance(value, Exception):
            raise RuntimeError("Failed to read replay target pressure") from value
        result.append(value)
    return result


@dataclasses.dataclass
class AlphaMoments:
    e2: float = 0.0
    ed: float = 0.0
    d2: float = 0.0
    value_count: int = 0
    pair_count: int = 0

    def update(self, e2: np.ndarray, ed: np.ndarray, d2: np.ndarray) -> None:
        self.e2 += float(np.asarray(e2, dtype=np.float64).sum())
        self.ed += float(np.asarray(ed, dtype=np.float64).sum())
        self.d2 += float(np.asarray(d2, dtype=np.float64).sum())
        self.pair_count += int(len(e2))

    def alpha(self) -> float:
        return float(np.clip(-self.ed / max(self.d2, 1e-24), 0.0, 1.0))

    def payload(self, vertex_count: int) -> dict[str, Any]:
        alpha = self.alpha()
        denominator = max(self.pair_count * vertex_count, 1)
        selected_mse = (self.e2 + 2.0 * alpha * self.ed + alpha * alpha * self.d2) / denominator
        return {
            "alpha": alpha,
            "pair_count": self.pair_count,
            "vertex_count": vertex_count,
            "rgb_rmse": math.sqrt(max(self.e2 / denominator, 0.0)),
            "selected_blend_rmse": math.sqrt(max(selected_mse, 0.0)),
            "quadratic_e2": self.e2,
            "quadratic_ed": self.ed,
            "quadratic_d2": self.d2,
        }


def fit_validation_alphas(
    archive: PredictionArchive,
    config: AuditConfig,
    engine: BatchedPressureMetricEngine,
    palm_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    if archive.split not in {"val", "validation"}:
        raise ValueError(f"Alpha selection requires validation predictions, got {archive.split!r}")
    moments = {lag: AlphaMoments() for lag in config.controlled_lags}
    reader = PressureReader(
        config.max_open_hdf5,
        max_cached_rows=config.pressure_row_cache_size,
        palm_mask=palm_mask,
        cached_dtype=np.float32,
    )
    torch = engine.torch
    processed = 0
    started = time.time()
    try:
        records = (
            record
            for record in iter_bilateral_pairs(config, pass_name="replay-alpha-fit")
            if record.self_pair.eligible
        )
        for batch in _iter_batches(records, int(args.pair_batch_size)):
            current_gt = np.stack(
                _require_pressure_rows(
                    reader.read_many([record.self_pair.current for record in batch])
                )
            ).astype(np.float32, copy=False)
            current_pred = archive.lookup_many(
                [record.self_pair.current.sample_uid for record in batch]
            )
            previous_pred = archive.lookup_many(
                [record.self_pair.previous.sample_uid for record in batch]
            )
            with torch.inference_mode():
                current_tensor = torch.from_numpy(current_pred).to(engine.device)
                previous_tensor = torch.from_numpy(previous_pred).to(engine.device)
                target_tensor = torch.from_numpy(current_gt).to(engine.device)
                error = current_tensor - target_tensor
                delta = previous_tensor - current_tensor
                e2 = error.square().sum(dim=1).cpu().numpy()
                ed = (error * delta).sum(dim=1).cpu().numpy()
                d2 = delta.square().sum(dim=1).cpu().numpy()
            lags = np.asarray([record.requested_lag for record in batch])
            for lag in config.controlled_lags:
                selected = lags == lag
                moments[lag].update(e2[selected], ed[selected], d2[selected])
            processed += len(batch)
            if args.progress_every and processed // args.progress_every != (
                processed - len(batch)
            ) // args.progress_every:
                print(
                    f"[replay-alpha-fit] pairs={processed:,} "
                    f"rate={processed / max(time.time() - started, 1e-9):,.1f}/s",
                    flush=True,
                )
    finally:
        reader.close()
    return {
        lag: moments[lag].payload(len(archive.vertex_indices))
        for lag in config.controlled_lags
    }


def _prediction_columns(raw: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    rename = {
        "persistence_mae": "mae",
        "persistence_rmse": "rmse",
        "persistence_viou": "viou",
        "distribution_viou": "distribution_viou",
        "support_iou": "contact_iou",
        "active_support_iou": "active_iou",
        "current_support_recall_from_previous": "gt_contact_recall",
        "previous_support_retained": "pred_contact_precision",
        "previous_volume": "pred_volume",
        "current_volume": "gt_volume",
        "relative_volume_change": "relative_volume_error",
    }
    return {
        output_name: np.asarray(raw[input_name])
        for input_name, output_name in rename.items()
    }


def _candidate_metrics(
    predictions: np.ndarray,
    targets: Sequence[np.ndarray],
    records: Sequence[BilateralPairRecord],
    engine: BatchedPressureMetricEngine,
) -> dict[str, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float32)
    target_array = np.stack(targets).astype(np.float32, copy=False)
    raw, _ = engine.compute(
        list(predictions),
        list(target_array),
        [record.self_pair for record in records],
    )
    metrics = _prediction_columns(raw)
    false_high = (target_array < 0.005) & (predictions >= 0.30)
    low_gt_count = (target_array < 0.005).sum(axis=1)
    false_high_count = false_high.sum(axis=1).astype(np.float64)
    false_high_excess = np.where(
        false_high,
        np.maximum(predictions - target_array, 0.0),
        0.0,
    ).sum(axis=1, dtype=np.float64)
    metrics.update(
        {
            "false_high_count": false_high_count,
            "false_high_rate": np.divide(
                false_high_count,
                low_gt_count,
                out=np.full(len(records), np.nan, dtype=np.float64),
                where=low_gt_count > 0,
            ),
            "false_high_excess_volume": false_high_excess,
        }
    )
    pred_volume = predictions.sum(axis=1, dtype=np.float64)
    gt_volume = target_array.sum(axis=1, dtype=np.float64)
    catastrophic_over_eligible = gt_volume < 10.0
    catastrophic_under_eligible = gt_volume >= 150.0
    metrics["catastrophic_over"] = np.where(
        catastrophic_over_eligible,
        (pred_volume > 300.0).astype(np.float64),
        np.nan,
    )
    metrics["catastrophic_under"] = np.where(
        catastrophic_under_eligible,
        (pred_volume < 50.0).astype(np.float64),
        np.nan,
    )
    return metrics


def _add_candidate(
    table: BilateralMetricTable,
    relation: str,
    records: Sequence[BilateralPairRecord],
    dynamics: Sequence[str],
    predictions: np.ndarray,
    targets: Sequence[np.ndarray],
    engine: BatchedPressureMetricEngine,
) -> dict[str, np.ndarray]:
    if not records:
        return {}
    metrics = _candidate_metrics(predictions, targets, records, engine)
    table.add_batch(relation, records, dynamics, metrics)
    return metrics


@dataclasses.dataclass
class GlobalHistoryShufflePlan:
    source_indices: dict[int, np.ndarray]
    metadata: dict[int, dict[str, Any]]
    cursors: dict[int, int] = dataclasses.field(default_factory=dict)

    def take(self, lag: int, count: int) -> np.ndarray:
        lag = int(lag)
        start = int(self.cursors.get(lag, 0))
        end = start + int(count)
        values = self.source_indices[lag]
        if end > len(values):
            raise RuntimeError(
                f"Cross-sequence shuffle plan exhausted for lag={lag}: "
                f"requested={end}, available={len(values)}"
            )
        self.cursors[lag] = end
        return values[start:end]

    def assert_consumed(self) -> None:
        for lag, values in self.source_indices.items():
            consumed = int(self.cursors.get(lag, 0))
            if consumed != len(values):
                raise RuntimeError(
                    f"Cross-sequence shuffle plan was not consumed exactly for "
                    f"lag={lag}: consumed={consumed}, available={len(values)}"
                )


def build_global_history_shuffle_plan(
    archive: PredictionArchive,
    config: AuditConfig,
) -> GlobalHistoryShufflePlan:
    sources: dict[int, list[int]] = {
        int(lag): [] for lag in config.controlled_lags
    }
    sequence_ids: dict[int, list[int]] = {
        int(lag): [] for lag in config.controlled_lags
    }
    sequence_lookup: dict[tuple[str, str, str], int] = {}
    processed = 0
    for record in iter_bilateral_pairs(config, pass_name=f"shuffle-plan-{archive.split}"):
        if not record.self_pair.eligible:
            continue
        lag = int(record.requested_lag)
        previous = record.self_pair.previous
        sequence_key = (previous.dataset, previous.split, previous.sequence_key)
        sequence_id = sequence_lookup.setdefault(sequence_key, len(sequence_lookup))
        sources[lag].append(archive.artifact_index(previous.sample_uid))
        sequence_ids[lag].append(sequence_id)
        processed += 1

    planned: dict[int, np.ndarray] = {}
    metadata: dict[int, dict[str, Any]] = {}
    for lag in config.controlled_lags:
        lag = int(lag)
        source = np.asarray(sources[lag], dtype=np.int64)
        groups = np.asarray(sequence_ids[lag], dtype=np.int32)
        if len(source) == 0:
            raise RuntimeError(f"No eligible records exist for shuffle lag={lag}")
        unique_groups, counts = np.unique(groups, return_counts=True)
        if len(unique_groups) < 2:
            raise RuntimeError(
                f"Cross-sequence shuffle requires at least two sequences for lag={lag}"
            )
        order = np.argsort(groups, kind="stable")
        maximum_group = int(counts.max())
        shifted_order = np.roll(order, -maximum_group)
        selected_groups = groups[shifted_order]
        destination_groups = groups[order]
        collisions = selected_groups == destination_groups
        replacement_count = int(collisions.sum())
        if replacement_count:
            first_index_by_group = {
                int(group): int(order[np.flatnonzero(groups[order] == group)[0]])
                for group in unique_groups
            }
            next_group = {
                int(group): int(unique_groups[(index + 1) % len(unique_groups)])
                for index, group in enumerate(unique_groups)
            }
            for sorted_position in np.flatnonzero(collisions):
                destination_group = int(destination_groups[sorted_position])
                source_index = first_index_by_group[next_group[destination_group]]
                shifted_order[sorted_position] = source_index
        shuffled = np.empty_like(source)
        shuffled[order] = source[shifted_order]
        shuffled_groups = np.empty_like(groups)
        shuffled_groups[order] = groups[shifted_order]
        if np.any(shuffled_groups == groups):
            raise RuntimeError(
                f"Failed to build a strict cross-sequence shuffle for lag={lag}"
            )
        planned[lag] = shuffled
        metadata[lag] = {
            "pair_count": int(len(source)),
            "sequence_count": int(len(unique_groups)),
            "maximum_sequence_pair_count": maximum_group,
            "replacement_count": replacement_count,
            "cross_sequence_fraction": 1.0,
        }
    print(
        f"[shuffle-plan-{archive.split}] planned {processed:,} strict "
        "cross-sequence histories",
        flush=True,
    )
    return GlobalHistoryShufflePlan(planned, metadata)


class SequenceDeltaAccumulator:
    METRICS = (
        "rmse",
        "viou",
        "distribution_viou",
        "contact_iou",
        "active_iou",
        "false_high_count",
        "false_high_rate",
        "false_high_excess_volume",
        "catastrophic_over",
        "catastrophic_under",
    )

    def __init__(self):
        self.values: dict[
            tuple[str, str, str, str, str], list[float | int]
        ] = defaultdict(lambda: [0.0, 0])

    def update(
        self,
        comparison: str,
        records: Sequence[BilateralPairRecord],
        candidate: Mapping[str, np.ndarray],
        baseline: Mapping[str, np.ndarray],
    ) -> None:
        grouped_indices: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            current = record.self_pair.current
            grouped_indices[
                (current.split, str(record.requested_lag), current.sequence_key)
            ].append(index)
        for metric in self.METRICS:
            if metric not in candidate or metric not in baseline:
                continue
            candidate_values = np.asarray(candidate[metric], dtype=np.float64)
            baseline_values = np.asarray(baseline[metric], dtype=np.float64)
            for dimensions, indices in grouped_indices.items():
                selected = np.asarray(indices, dtype=np.int64)
                differences = candidate_values[selected] - baseline_values[selected]
                differences = differences[np.isfinite(differences)]
                if not len(differences):
                    continue
                split, lag, sequence_key = dimensions
                key = (comparison, split, lag, sequence_key, metric)
                running = self.values[key]
                running[0] = float(running[0]) + float(
                    differences.sum(dtype=np.float64)
                )
                running[1] = int(running[1]) + int(len(differences))

    def rows(
        self,
        *,
        iterations: int,
        confidence: float,
        seed: int,
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str, str], list[tuple[float, int]]] = defaultdict(list)
        for (comparison, split, lag, _sequence, metric), (total, count) in self.values.items():
            if int(count) > 0:
                grouped[(comparison, split, lag, metric)].append(
                    (float(total) / int(count), int(count))
                )
        rows = []
        tail = (1.0 - float(confidence)) * 0.5
        for group_index, (key, sequence_values) in enumerate(sorted(grouped.items())):
            comparison, split, lag, metric = key
            means = np.asarray([item[0] for item in sequence_values], dtype=np.float64)
            counts = np.asarray([item[1] for item in sequence_values], dtype=np.int64)
            rng = np.random.default_rng(int(seed) + group_index * 1009)
            frame_bootstrap = np.empty(int(iterations), dtype=np.float64)
            macro_bootstrap = np.empty(int(iterations), dtype=np.float64)
            for index in range(int(iterations)):
                selected = rng.integers(0, len(means), size=len(means))
                frame_bootstrap[index] = float(
                    np.average(means[selected], weights=counts[selected])
                )
                macro_bootstrap[index] = float(means[selected].mean())
            frame_ci_low = float(np.quantile(frame_bootstrap, tail))
            frame_ci_high = float(np.quantile(frame_bootstrap, 1.0 - tail))
            rows.append(
                {
                    "comparison": comparison,
                    "split": split,
                    "lag": lag,
                    "metric": metric,
                    "sequence_count": int(len(means)),
                    "frame_count": int(counts.sum()),
                    "frame_weighted_delta": float(
                        np.average(means, weights=counts)
                    ),
                    "sequence_macro_delta": float(means.mean()),
                    "ci_low": frame_ci_low,
                    "ci_high": frame_ci_high,
                    "frame_weighted_ci_low": frame_ci_low,
                    "frame_weighted_ci_high": frame_ci_high,
                    "sequence_macro_ci_low": float(
                        np.quantile(macro_bootstrap, tail)
                    ),
                    "sequence_macro_ci_high": float(
                        np.quantile(macro_bootstrap, 1.0 - tail)
                    ),
                    "confidence": float(confidence),
                    "bootstrap_iterations": int(iterations),
                }
            )
        return rows


def evaluate_pair_replay(
    archive: PredictionArchive,
    config: AuditConfig,
    engine: BatchedPressureMetricEngine,
    alpha_selection: Mapping[int, Mapping[str, Any]],
    palm_mask: np.ndarray,
    args: argparse.Namespace,
    sequence_deltas: SequenceDeltaAccumulator,
) -> tuple[BilateralMetricTable, Counter, dict[str, Any]]:
    table = BilateralMetricTable()
    classes: Counter = Counter()
    oracle_decisions: Counter = Counter()
    shuffle_plan = build_global_history_shuffle_plan(archive, config)
    reader = PressureReader(
        config.max_open_hdf5,
        max_cached_rows=config.pressure_row_cache_size,
        palm_mask=palm_mask,
        cached_dtype=np.float32,
    )
    processed = 0
    started = time.time()
    try:
        records = (
            record
            for record in iter_bilateral_pairs(config, pass_name=f"replay-{archive.split}")
            if record.self_pair.eligible
        )
        for batch in _iter_batches(records, int(args.pair_batch_size)):
            current_rows = _require_pressure_rows(
                reader.read_many([record.self_pair.current for record in batch])
            )
            previous_rows = _require_pressure_rows(
                reader.read_many([record.self_pair.previous for record in batch])
            )
            current_pred = archive.lookup_many(
                [record.self_pair.current.sample_uid for record in batch]
            )
            previous_pred = archive.lookup_many(
                [record.self_pair.previous.sample_uid for record in batch]
            )
            oracle_metrics, dynamics = engine.compute(
                previous_rows,
                current_rows,
                [record.self_pair for record in batch],
            )
            del oracle_metrics
            classes.update(
                (archive.split, str(record.requested_lag), label)
                for record, label in zip(batch, dynamics)
            )
            alphas = np.asarray(
                [float(alpha_selection[record.requested_lag]["alpha"]) for record in batch],
                dtype=np.float32,
            )[:, None]
            selected_blend = current_pred + alphas * (previous_pred - current_pred)
            zero_blend = current_pred * (1.0 - alphas)

            lag_values = np.asarray([record.requested_lag for record in batch])
            shuffled_history = np.empty_like(previous_pred)
            for lag in config.controlled_lags:
                positions = np.flatnonzero(lag_values == int(lag))
                if not len(positions):
                    continue
                source_indices = shuffle_plan.take(int(lag), len(positions))
                shuffled_history[positions] = archive.lookup_artifact_indices(
                    source_indices
                )
            shuffled_blend = current_pred + alphas * (
                shuffled_history - current_pred
            )

            target_array = np.stack(current_rows).astype(np.float32, copy=False)
            rgb_frame_mse = np.square(current_pred - target_array).mean(axis=1)
            blend_frame_mse = np.square(selected_blend - target_array).mean(axis=1)
            oracle_uses_history = blend_frame_mse < rgb_frame_mse
            oracle_frame_gate = np.where(
                oracle_uses_history[:, None], selected_blend, current_pred
            )
            stable_uses_history = np.asarray(
                [label in {"empty_stable", "spatially_stable"} for label in dynamics],
                dtype=bool,
            )
            oracle_dynamics_gate = np.where(
                stable_uses_history[:, None], selected_blend, current_pred
            )
            oracle_decisions.update(
                (
                    archive.split,
                    str(record.requested_lag),
                    label,
                    "history" if selected else "rgb",
                )
                for record, label, selected in zip(
                    batch, dynamics, oracle_uses_history
                )
            )

            rgb_metrics = _add_candidate(
                table, "rgb_current", batch, dynamics, current_pred, current_rows, engine
            )
            _add_candidate(table, "same_previous_prediction", batch, dynamics, previous_pred, current_rows, engine)
            blend_metrics = _add_candidate(
                table,
                "selected_same_history_blend",
                batch,
                dynamics,
                selected_blend,
                current_rows,
                engine,
            )
            _add_candidate(table, "matched_zero_state_blend", batch, dynamics, zero_blend, current_rows, engine)
            shuffle_metrics = _add_candidate(
                table,
                "strict_cross_sequence_shuffle_blend",
                batch,
                dynamics,
                shuffled_blend,
                current_rows,
                engine,
            )
            _add_candidate(table, "oracle_previous_gt", batch, dynamics, np.stack(previous_rows), current_rows, engine)
            frame_oracle_metrics = _add_candidate(
                table,
                "oracle_frame_rmse_gate",
                batch,
                dynamics,
                oracle_frame_gate,
                current_rows,
                engine,
            )
            dynamics_oracle_metrics = _add_candidate(
                table,
                "oracle_dynamics_gate",
                batch,
                dynamics,
                oracle_dynamics_gate,
                current_rows,
                engine,
            )
            sequence_deltas.update(
                "selected_history_vs_rgb", batch, blend_metrics, rgb_metrics
            )
            sequence_deltas.update(
                "shuffle_history_vs_rgb", batch, shuffle_metrics, rgb_metrics
            )
            sequence_deltas.update(
                "selected_history_vs_shuffle",
                batch,
                blend_metrics,
                shuffle_metrics,
            )
            sequence_deltas.update(
                "oracle_frame_gate_vs_rgb",
                batch,
                frame_oracle_metrics,
                rgb_metrics,
            )
            sequence_deltas.update(
                "oracle_dynamics_gate_vs_rgb",
                batch,
                dynamics_oracle_metrics,
                rgb_metrics,
            )

            paired_positions = [
                index
                for index, record in enumerate(batch)
                if record.contralateral_previous is not None
            ]
            if paired_positions:
                paired_records = [batch[index] for index in paired_positions]
                paired_dynamics = [dynamics[index] for index in paired_positions]
                other_pred = archive.lookup_many(
                    [
                        record.contralateral_previous.sample_uid  # type: ignore[union-attr]
                        for record in paired_records
                    ]
                )
                paired_current = current_pred[paired_positions]
                paired_same = selected_blend[paired_positions]
                paired_alphas = alphas[paired_positions]
                other_blend = paired_current + paired_alphas * (
                    other_pred - paired_current
                )
                paired_targets = [current_rows[index] for index in paired_positions]
                paired_rgb_metrics = _add_candidate(
                    table,
                    "matched_bilateral_rgb_current",
                    paired_records,
                    paired_dynamics,
                    paired_current,
                    paired_targets,
                    engine,
                )
                paired_same_metrics = _add_candidate(
                    table,
                    "matched_bilateral_same_history_blend",
                    paired_records,
                    paired_dynamics,
                    paired_same,
                    paired_targets,
                    engine,
                )
                paired_other_metrics = _add_candidate(
                    table,
                    "matched_bilateral_contralateral_history_blend",
                    paired_records,
                    paired_dynamics,
                    other_blend,
                    paired_targets,
                    engine,
                )
                sequence_deltas.update(
                    "matched_same_history_vs_rgb",
                    paired_records,
                    paired_same_metrics,
                    paired_rgb_metrics,
                )
                sequence_deltas.update(
                    "matched_contralateral_history_vs_rgb",
                    paired_records,
                    paired_other_metrics,
                    paired_rgb_metrics,
                )
                sequence_deltas.update(
                    "matched_same_vs_contralateral_history",
                    paired_records,
                    paired_same_metrics,
                    paired_other_metrics,
                )

            processed += len(batch)
            if args.progress_every and processed // args.progress_every != (
                processed - len(batch)
            ) // args.progress_every:
                print(
                    f"[replay-{archive.split}] pairs={processed:,} "
                    f"rate={processed / max(time.time() - started, 1e-9):,.1f}/s",
                    flush=True,
                )
    finally:
        reader.close()
    shuffle_plan.assert_consumed()
    oracle_totals = Counter()
    for (split, lag, _dynamics, decision), count in oracle_decisions.items():
        oracle_totals[(split, lag, decision)] += count
    return table, classes, {
        "processed_pairs": processed,
        "elapsed_seconds": time.time() - started,
        "reader": reader.stats(),
        "strict_shuffle": shuffle_plan.metadata,
        "oracle_frame_gate": {
            f"{split}/lag{lag}/{decision}": int(count)
            for (split, lag, decision), count in sorted(oracle_totals.items())
        },
        "oracle_frame_gate_by_dynamics": {
            f"{split}/lag{lag}/{dynamics}/{decision}": int(count)
            for (split, lag, dynamics, decision), count in sorted(
                oracle_decisions.items()
            )
        },
    }


def evaluate_rollout(
    archive: PredictionArchive,
    config: AuditConfig,
    engine: BatchedPressureMetricEngine,
    alpha: float,
    palm_mask: np.ndarray,
    args: argparse.Namespace,
    sequence_deltas: SequenceDeltaAccumulator,
) -> tuple[BilateralMetricTable, Counter, dict[str, Any]]:
    table = BilateralMetricTable()
    reset_reasons: Counter = Counter()
    reader = PressureReader(
        config.max_open_hdf5,
        max_cached_rows=config.pressure_row_cache_size,
        palm_mask=palm_mask,
        cached_dtype=np.float32,
    )
    states: dict[tuple[str, str, str, str, int], np.ndarray] = {}
    previous_records: dict[tuple[str, str, str, str, int], SampleRecord] = {}
    current_sequence: Optional[tuple[str, str, str]] = None
    pending_records: list[BilateralPairRecord] = []
    pending_labels: list[str] = []
    pending_rgb: list[np.ndarray] = []
    pending_rollout: list[np.ndarray] = []
    pending_samples: list[SampleRecord] = []
    processed = 0
    continued = 0
    started = time.time()

    def flush() -> None:
        nonlocal processed
        if not pending_records:
            return
        targets = _require_pressure_rows(reader.read_many(pending_samples))
        rgb = np.stack(pending_rgb).astype(np.float32, copy=False)
        rollout = np.stack(pending_rollout).astype(np.float32, copy=False)
        rgb_metrics = _add_candidate(
            table,
            "rollout_rgb_current",
            pending_records,
            pending_labels,
            rgb,
            targets,
            engine,
        )
        rollout_metrics = _add_candidate(
            table,
            "ema_prediction_rollout",
            pending_records,
            pending_labels,
            rollout,
            targets,
            engine,
        )
        sequence_deltas.update(
            "ema_rollout_vs_rgb",
            pending_records,
            rollout_metrics,
            rgb_metrics,
        )
        processed += len(pending_records)
        pending_records.clear()
        pending_labels.clear()
        pending_rgb.clear()
        pending_rollout.clear()
        pending_samples.clear()

    try:
        for sample in iter_samples(config, pass_name=f"rollout-{archive.split}"):
            sequence = (sample.dataset, sample.split, sample.sequence_key)
            if current_sequence is not None and sequence != current_sequence:
                states.clear()
                previous_records.clear()
            current_sequence = sequence
            key = sample.group_key
            rgb = archive.lookup(sample.sample_uid)
            previous = previous_records.get(key)
            can_continue = previous is not None
            reason = "continued"
            pair: Optional[PairRecord] = None
            if previous is None:
                can_continue = False
                reason = "cold_start"
            else:
                pair = _pair_from_samples(previous, sample, config)
                if pair.frame_gap != 1:
                    can_continue = False
                    reason = "frame_gap"
                elif pair.time_gap is None or pair.time_gap > 0.05:
                    can_continue = False
                    reason = "time_gap"
                elif (
                    previous.bbox_association_id is not None
                    and sample.bbox_association_id is not None
                    and previous.bbox_association_id != sample.bbox_association_id
                ):
                    can_continue = False
                    reason = "association_change"
                elif not pair.bbox_stable:
                    can_continue = False
                    reason = "bbox_unstable"
            if can_continue:
                state = rgb + float(alpha) * (states[key].astype(np.float32) - rgb)
                continued += 1
            else:
                state = rgb.copy()
                reset_reasons[reason] += 1
            states[key] = state.astype(np.float16)
            previous_records[key] = sample
            if pair is None:
                pair = PairRecord(
                    previous=sample,
                    current=sample,
                    frame_gap=0,
                    time_gap=None,
                    bbox_iou=1.0,
                    bbox_center_jump=0.0,
                    bbox_abs_log_area_ratio=0.0,
                    bbox_stable=True,
                    temporal_eligible=False,
                    eligible=False,
                    transition="cold_start",
                )
            pending_records.append(
                BilateralPairRecord(pair, requested_lag=1, contralateral_previous=None)
            )
            pending_labels.append("state_continued" if can_continue else "state_reset")
            pending_rgb.append(rgb)
            pending_rollout.append(state)
            pending_samples.append(sample)
            if len(pending_records) >= int(args.pair_batch_size):
                flush()
                if args.progress_every and processed % args.progress_every < int(args.pair_batch_size):
                    print(
                        f"[rollout-{archive.split}] records={processed:,} "
                        f"rate={processed / max(time.time() - started, 1e-9):,.1f}/s",
                        flush=True,
                    )
        flush()
    finally:
        reader.close()
    return table, reset_reasons, {
        "processed_records": processed,
        "continued_records": continued,
        "continued_fraction": continued / max(processed, 1),
        "reset_reasons": dict(reset_reasons),
        "elapsed_seconds": time.time() - started,
        "reader": reader.stats(),
    }


def _merge_tables(target: BilateralMetricTable, source: BilateralMetricTable) -> None:
    for key, metrics in source.values.items():
        for name, running in metrics.items():
            payload = running.payload()
            if payload["count"]:
                target.values[key][name].merge_statistics(
                    int(payload["count"]),
                    running.total,
                    running.total_squared,
                    float(payload["min"]),
                    float(payload["max"]),
                )


def _class_rows(classes: Counter) -> list[dict[str, Any]]:
    totals = Counter()
    for (split, lag, _), count in classes.items():
        totals[(split, lag)] += count
    return [
        {
            "split": split,
            "lag": lag,
            "dynamics_class": dynamics,
            "count": count,
            "fraction": count / totals[(split, lag)],
        }
        for (split, lag, dynamics), count in sorted(classes.items())
    ]


def _summary_metrics(
    table: BilateralMetricTable, splits: Sequence[str], lags: Sequence[int]
) -> dict[str, Any]:
    relations = sorted({key[0] for key in table.values})
    result: dict[str, Any] = {}
    for split in splits:
        split_result = {}
        for lag in lags:
            lag_result = {}
            for relation in relations:
                values = {
                    metric: _safe_float(
                        table.mean(
                            relation,
                            metric,
                            dataset="TouchAnything",
                            split=split,
                            lag=str(lag),
                        )
                    )
                    for metric in (
                        "rmse",
                        "viou",
                        "distribution_viou",
                        "contact_iou",
                        "active_iou",
                        "pred_volume",
                        "gt_volume",
                        "false_high_count",
                        "false_high_rate",
                        "false_high_excess_volume",
                        "catastrophic_over",
                        "catastrophic_under",
                    )
                }
                if any(value is not None for value in values.values()):
                    lag_result[relation] = values
            split_result[str(lag)] = lag_result
        result[split] = split_result
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-root",
        action="append",
        required=True,
        help="Completed exact prediction export; repeat for val/seen/unseen.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--controlled-lags", type=_parse_lags, default=(1, 2, 4, 8))
    parser.add_argument("--metric-device", default="cuda")
    parser.add_argument("--pair-batch-size", type=int, default=4096)
    parser.add_argument("--contact-threshold", type=float, default=0.10)
    parser.add_argument("--active-threshold", type=float, default=0.05)
    parser.add_argument("--max-time-gap", type=float, default=0.5)
    parser.add_argument("--min-bbox-iou", type=float, default=0.05)
    parser.add_argument("--max-bbox-center-jump", type=float, default=0.5)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=2.0)
    parser.add_argument("--max-open-hdf5", type=int, default=4)
    parser.add_argument("--pressure-row-cache-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.pair_batch_size <= 0:
        raise ValueError("--pair-batch-size must be positive")
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence must be strictly between 0 and 1")
    if 1 not in args.controlled_lags:
        raise ValueError("--controlled-lags must include 1 for adjacent rollout")
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = [PredictionArchive(Path(path)) for path in args.prediction_root]
    by_split = {archive.split: archive for archive in archives}
    validation = by_split.get("val") or by_split.get("validation")
    if validation is None:
        raise RuntimeError("A validation prediction export is required for alpha selection")
    checkpoint_hashes = {str(archive.config["checkpoint_sha256"]) for archive in archives}
    vertex_hashes = {
        hashlib.sha256(np.ascontiguousarray(archive.vertex_indices).tobytes()).hexdigest()
        for archive in archives
    }
    if len(checkpoint_hashes) != 1 or len(vertex_hashes) != 1:
        raise RuntimeError("Prediction exports do not share one checkpoint and vertex layout")
    palm_mask, palm_provenance = _load_palm_mask()
    expected_vertices = np.flatnonzero(palm_mask).astype(np.int32)
    for archive in archives:
        if not np.array_equal(archive.vertex_indices, expected_vertices):
            raise RuntimeError(
                f"Prediction export is not the canonical palm layout: {archive.root}"
            )

    validation_config = _audit_config(
        validation, output_dir, args.controlled_lags, args
    )
    engine = BatchedPressureMetricEngine(validation_config)
    engine.validate_against_scalar(validation_config)
    alpha_selection = fit_validation_alphas(
        validation, validation_config, engine, palm_mask, args
    )
    selection_payload = {
        "schema": "predicted_tactile_history_alpha_selection_v1",
        "selection_split": validation.split,
        "objective": "vertex_weighted_rmse",
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "lags": {str(lag): payload for lag, payload in alpha_selection.items()},
    }
    write_json_atomic(output_dir / "alpha_selection.json", selection_payload)

    pair_table = BilateralMetricTable()
    rollout_table = BilateralMetricTable()
    all_classes: Counter = Counter()
    sequence_deltas = SequenceDeltaAccumulator()
    execution: dict[str, Any] = {}
    for archive in archives:
        config = _audit_config(archive, output_dir, args.controlled_lags, args)
        split_pair_table, split_classes, pair_stats = evaluate_pair_replay(
            archive,
            config,
            engine,
            alpha_selection,
            palm_mask,
            args,
            sequence_deltas,
        )
        _merge_tables(pair_table, split_pair_table)
        all_classes.update(split_classes)
        split_rollout_table, reset_reasons, rollout_stats = evaluate_rollout(
            archive,
            config,
            engine,
            float(alpha_selection[1]["alpha"]),
            palm_mask,
            args,
            sequence_deltas,
        )
        _merge_tables(rollout_table, split_rollout_table)
        execution[archive.split] = {
            "pair_replay": pair_stats,
            "rollout": rollout_stats,
            "reset_reasons": dict(reset_reasons),
        }

    _atomic_csv(output_dir / "pair_replay_metrics.csv", pair_table.rows())
    _atomic_csv(output_dir / "rollout_metrics.csv", rollout_table.rows())
    _atomic_csv(output_dir / "target_dynamics_classes.csv", _class_rows(all_classes))
    bootstrap_rows = sequence_deltas.rows(
        iterations=int(args.bootstrap_iterations),
        confidence=float(args.bootstrap_confidence),
        seed=int(args.seed),
    )
    _atomic_csv(output_dir / "sequence_bootstrap.csv", bootstrap_rows)
    splits = [archive.split for archive in archives]
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "checkpoint": str(validation.config["checkpoint"]),
        "controlled_lags": list(args.controlled_lags),
        "alpha_selection": selection_payload,
        "pair_summary": _summary_metrics(pair_table, splits, args.controlled_lags),
        "rollout_summary": _summary_metrics(rollout_table, splits, (1,)),
        "sequence_bootstrap": {
            "path": str(output_dir / "sequence_bootstrap.csv"),
            "row_count": len(bootstrap_rows),
            "iterations": int(args.bootstrap_iterations),
            "confidence": float(args.bootstrap_confidence),
        },
        "execution": execution,
        "prediction_exports": [archive.config for archive in archives],
        "palm": palm_provenance,
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "control_definitions": {
                "selected_same_history_blend": "rgb_t + alpha_lag * (rgb_t-lag - rgb_t)",
                "matched_bilateral_contralateral_history_blend": "same alpha with opposite anonymous query history on the exact subset having both hands",
                "matched_bilateral_same_history_blend": "same-hand history on that identical bilateral subset",
                "matched_bilateral_rgb_current": "RGB-only baseline on that identical bilateral subset",
                "matched_zero_state_blend": "same alpha with an all-zero history",
                "strict_cross_sequence_shuffle_blend": "same-lag previous predictions globally and deterministically reassigned with 100% cross-sequence provenance and no zero fallback",
                "oracle_frame_rmse_gate": "per-frame GT oracle choosing the lower-RMSE option between RGB-only and selected history",
                "oracle_dynamics_gate": "GT-derived dynamics oracle using history only for empty-stable or spatially-stable frames",
                "ema_prediction_rollout": "state_t = rgb_t + alpha_1 * (state_t-1 - rgb_t), reset on discontinuity",
            },
        },
    }
    write_json_atomic(output_dir / "summary.json", payload)

    lines = [
        "Predicted tactile history replay audit",
        f"checkpoint: {payload['checkpoint']}",
        f"splits: {', '.join(splits)}",
        "validation-selected alpha: "
        + ", ".join(
            f"lag{lag}={alpha_selection[lag]['alpha']:.4f}"
            for lag in args.controlled_lags
        ),
        f"sequence bootstrap: {args.bootstrap_confidence:.1%} CI, "
        f"{args.bootstrap_iterations} resamples",
    ]
    for split in splits:
        lines.append(f"{split}:")
        for lag in args.controlled_lags:
            metrics = payload["pair_summary"][split][str(lag)]
            rgb = metrics.get("rgb_current", {})
            fused = metrics.get("selected_same_history_blend", {})
            control = metrics.get("matched_zero_state_blend", {})
            if rgb and fused:
                lines.append(
                    f"  lag{lag} rgb/blend/zero RMSE="
                    f"{_format_metric(rgb.get('rmse'), 5)}/"
                    f"{_format_metric(fused.get('rmse'), 5)}/"
                    f"{_format_metric(control.get('rmse'), 5)}; V-IoU="
                    f"{_format_metric(rgb.get('viou'), 4)}/"
                    f"{_format_metric(fused.get('viou'), 4)}/"
                    f"{_format_metric(control.get('viou'), 4)}"
                )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Reports: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
