"""Offline RGB contact/ordinal sufficiency and complementarity audit.

The audit consumes exact per-vertex artifacts exported by eval_tactile_fast.py.
It fits only low-capacity logistic stackers on validation data and applies them
unchanged to test splits. It never modifies tactile pressure predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


ARTIFACT_SCHEMA = "tactile_selector_vertex_artifacts_v1"
AUDIT_SCHEMA = "rgb_contact_ordinal_sufficiency_v1"
SIGNALS = ("contact", "ordinal", "down", "down_control")
DEFAULT_SPLITS = ("val", "test_seen", "test_unseen")
PRECISION_LEVELS = (0.70, 0.80, 0.90, 0.95)
PROBABILITY_BINS = 4096
SEQUENCE_BINS = 128
EVALUATION_POINT_CHUNK = 1_000_000
EPSILON = 1e-7


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_write_text(path, "")
        return
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _finite(value):
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float32), EPSILON, 1.0 - EPSILON)
    return np.log(probability) - np.log1p(-probability)


def _threshold_index(thresholds: Sequence[float], target: float) -> int:
    values = np.asarray(thresholds, dtype=np.float64)
    matches = np.flatnonzero(np.isclose(values, float(target), atol=1e-7))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one selector threshold {target}, got {list(values)}"
        )
    return int(matches[0])


class ArtifactSplit:
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve(strict=True)
        config_path = self.root / "artifact_config.json"
        complete_path = self.root / "_COMPLETE"
        if not config_path.is_file() or not complete_path.is_file():
            raise RuntimeError(f"Incomplete selector artifact: {self.root}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        if self.config.get("schema") != ARTIFACT_SCHEMA:
            raise RuntimeError(
                f"Unsupported selector artifact schema at {config_path}: "
                f"{self.config.get('schema')!r}"
            )
        self.shards = sorted((self.root / "shards").glob("worker_*.npz"))
        if not self.shards:
            raise RuntimeError(f"Selector artifact has no worker shards: {self.root}")


class ArtifactBundle:
    def __init__(self, artifact_root: Path, split: str):
        self.root = Path(artifact_root).expanduser().resolve(strict=True)
        self.split = str(split)
        self.signals = {
            signal: ArtifactSplit(self.root / signal / self.split)
            for signal in SIGNALS
        }
        reference = self.signals["contact"]
        if not bool(reference.config.get("reference_payload")):
            raise RuntimeError("The contact artifact must contain the base/GT reference")
        for signal, artifact in self.signals.items():
            if str(artifact.config.get("split")) != self.split:
                raise RuntimeError(
                    f"Artifact split mismatch for {signal}: "
                    f"{artifact.config.get('split')!r} vs {self.split!r}"
                )
            for field in (
                "record_count",
                "valid_vertex_count",
                "vertex_indices_sha256",
                "input_resolution",
                "bbox_rescale_factor",
                "bbox_source_policy",
            ):
                if artifact.config.get(field) != reference.config.get(field):
                    raise RuntimeError(
                        f"Artifact contract differs for {signal}.{field}: "
                        f"{artifact.config.get(field)!r} vs "
                        f"{reference.config.get(field)!r}"
                    )
            if len(artifact.shards) != len(reference.shards):
                raise RuntimeError(
                    f"Artifact shard count differs for {signal}: "
                    f"{len(artifact.shards)} vs {len(reference.shards)}"
                )
        self.contact_index = _threshold_index(
            self.signals["contact"].config["selector_thresholds"], 0.10
        )
        self.ordinal_thresholds = tuple(
            float(value)
            for value in self.signals["ordinal"].config["selector_thresholds"]
        )
        self.ordinal_names = tuple(
            f"ordinal_{value:g}" for value in self.ordinal_thresholds
        )
        self.down_index = _threshold_index(
            self.signals["down"].config["selector_thresholds"], 0.10
        )
        self.down_control_index = _threshold_index(
            self.signals["down_control"].config["selector_thresholds"], 0.10
        )

    def iter_shards(self):
        paths_by_signal = {
            signal: artifact.shards for signal, artifact in self.signals.items()
        }
        for shard_index in range(len(paths_by_signal["contact"])):
            loaded = {}
            for signal in SIGNALS:
                path = paths_by_signal[signal][shard_index]
                with np.load(path, allow_pickle=False) as handle:
                    loaded[signal] = {
                        key: np.asarray(handle[key])
                        for key in handle.files
                    }
            reference = loaded["contact"]
            for signal in SIGNALS[1:]:
                current = loaded[signal]
                for field in (
                    "indices",
                    "sample_uids",
                    "sequence_keys",
                    "query_aliases",
                    "frame_indices",
                    "vertex_indices",
                ):
                    if not np.array_equal(current[field], reference[field]):
                        raise RuntimeError(
                            f"Artifact alignment differs for {signal}, shard {shard_index}, "
                            f"field {field}"
                        )
                if str(current["reference_sha256"].item()) != str(
                    reference["reference_sha256"].item()
                ):
                    raise RuntimeError(
                        f"Base/GT reference hash differs for {signal}, shard {shard_index}"
                    )
            if "base_predictions" not in reference or "targets" not in reference:
                raise RuntimeError("Contact reference shard lacks base predictions or GT")
            features = {
                "base": np.clip(
                    _logit(reference["base_predictions"]), -24.0, 24.0
                ).astype(np.float32),
                "contact": np.asarray(
                    reference["selector_logits"][:, self.contact_index],
                    dtype=np.float32,
                ),
                "down": np.asarray(
                    loaded["down"]["selector_logits"][:, self.down_index],
                    dtype=np.float32,
                ),
                "down_control": np.asarray(
                    loaded["down_control"]["selector_logits"][
                        :, self.down_control_index
                    ],
                    dtype=np.float32,
                ),
            }
            ordinal_logits = loaded["ordinal"]["selector_logits"]
            for index, name in enumerate(self.ordinal_names):
                features[name] = np.asarray(
                    ordinal_logits[:, index], dtype=np.float32
                )
            yield {
                "indices": np.asarray(reference["indices"], dtype=np.int64),
                "sample_uids": np.asarray(reference["sample_uids"], dtype=str),
                "sequence_keys": np.asarray(reference["sequence_keys"], dtype=str),
                "query_aliases": np.asarray(reference["query_aliases"], dtype=str),
                "frame_indices": np.asarray(reference["frame_indices"], dtype=np.int64),
                "base_predictions": np.asarray(
                    reference["base_predictions"], dtype=np.float32
                ),
                "targets": np.asarray(reference["targets"], dtype=np.float32),
                "features": features,
            }


@dataclass(frozen=True)
class TargetSpec:
    name: str
    kind: str
    threshold: float = 0.0

    @property
    def sequence_metrics(self) -> bool:
        return self.kind in {"contact", "false_high", "false_low"}

    def labels(
        self, base: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.kind == "contact":
            eligible = (target <= 0.02) | (target >= 0.10)
            labels = target >= 0.10
        elif self.kind == "false_high":
            clear = (target <= 0.02) | (target >= 0.10)
            eligible = clear & (base >= 0.10)
            labels = target <= 0.02
        elif self.kind == "false_low":
            clear = (target <= 0.02) | (target >= 0.10)
            eligible = clear & (base < 0.10)
            labels = target >= 0.10
        elif self.kind == "ordinal":
            eligible = np.ones(target.shape, dtype=bool)
            labels = target >= float(self.threshold)
        else:
            raise ValueError(f"Unsupported target kind: {self.kind}")
        return eligible, labels


@dataclass
class LogisticModel:
    feature_names: Tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    intercept: float
    coefficients: np.ndarray
    fit_count: int
    fit_positive_fraction: float

    def predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        value = np.full_like(
            np.asarray(features[self.feature_names[0]], dtype=np.float32),
            float(self.intercept),
            dtype=np.float32,
        )
        for index, name in enumerate(self.feature_names):
            normalized = (
                np.asarray(features[name], dtype=np.float32) - self.means[index]
            ) / self.scales[index]
            value += float(self.coefficients[index]) * normalized
        return _sigmoid(value).astype(np.float32)

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "means": [float(value) for value in self.means],
            "scales": [float(value) for value in self.scales],
            "intercept": float(self.intercept),
            "coefficients": [float(value) for value in self.coefficients],
            "fit_count": int(self.fit_count),
            "fit_positive_fraction": float(self.fit_positive_fraction),
        }


def _stacker_matrix(models: Mapping[str, LogisticModel]):
    model_names = tuple(models)
    feature_names = tuple(
        sorted(
            {
                feature_name
                for model in models.values()
                for feature_name in model.feature_names
            }
        )
    )
    feature_index = {name: index for index, name in enumerate(feature_names)}
    intercepts = np.empty(len(model_names), dtype=np.float64)
    weights = np.zeros(
        (len(model_names), len(feature_names)), dtype=np.float64
    )
    for model_index, model_name in enumerate(model_names):
        model = models[model_name]
        raw_coefficients = model.coefficients / model.scales
        intercepts[model_index] = model.intercept - float(
            np.dot(raw_coefficients, model.means)
        )
        for local_index, feature_name in enumerate(model.feature_names):
            weights[model_index, feature_index[feature_name]] = raw_coefficients[
                local_index
            ]
    return model_names, feature_names, intercepts, weights


def _model_specs(target: TargetSpec, ordinal_names: Sequence[str]):
    ordinal_names = tuple(ordinal_names)
    if target.kind == "false_high":
        return {
            "B": ("base",),
            "C": ("contact",),
            "O": ordinal_names,
            "D": ("down",),
            "Dctl": ("down_control",),
            "B+C+O": ("base", "contact", *ordinal_names),
            "B+C+O+D": ("base", "contact", *ordinal_names, "down"),
            "B+C+O+Dctl": (
                "base",
                "contact",
                *ordinal_names,
                "down_control",
            ),
        }
    if target.kind == "ordinal":
        ordinal_name = f"ordinal_{target.threshold:g}"
        return {
            "B": ("base",),
            "O_k": (ordinal_name,),
            "B+O": ("base", *ordinal_names),
            "B+C+O": ("base", "contact", *ordinal_names),
        }
    return {
        "B": ("base",),
        "C": ("contact",),
        "O": ordinal_names,
        "B+C": ("base", "contact"),
        "B+O": ("base", *ordinal_names),
        "B+C+O": ("base", "contact", *ordinal_names),
    }


def _collect_fit_samples(
    bundle: ArtifactBundle,
    targets: Sequence[TargetSpec],
    *,
    vertices_per_frame: int,
    maximum_rows: int,
    seed: int,
) -> Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]]:
    states = {
        target.name: {
            "target": target,
            "feature_rows": {},
            "labels": [],
            "sample_weights": [],
            "target_seed": zlib.crc32(target.name.encode("utf-8")) & 0xFFFFFFFF,
        }
        for target in targets
    }
    shard_count = len(bundle.signals["contact"].shards)
    for shard_index, shard in enumerate(bundle.iter_shards(), start=1):
        print(
            f"[audit] fit split={bundle.split} shard={shard_index}/{shard_count}",
            flush=True,
        )
        for state in states.values():
            target = state["target"]
            eligible, shard_labels = target.labels(
                shard["base_predictions"], shard["targets"]
            )
            for row_index, artifact_index in enumerate(shard["indices"]):
                candidates = np.flatnonzero(eligible[row_index])
                if len(candidates) == 0:
                    continue
                eligible_count = len(candidates)
                row_seed = (
                    int(seed)
                    ^ int(state["target_seed"])
                    ^ ((int(artifact_index) + 1) * 0x9E3779B1)
                ) & 0xFFFFFFFFFFFFFFFF
                rng = np.random.default_rng(row_seed)
                if len(candidates) > int(vertices_per_frame):
                    candidates = rng.choice(
                        candidates, size=int(vertices_per_frame), replace=False
                    )
                state["sample_weights"].append(
                    np.full(
                        len(candidates),
                        float(eligible_count) / len(candidates),
                        dtype=np.float64,
                    )
                )
                state["labels"].append(
                    shard_labels[row_index, candidates].astype(np.float32)
                )
                feature_rows = state["feature_rows"]
                for name, values in shard["features"].items():
                    feature_rows.setdefault(name, []).append(
                        values[row_index, candidates].astype(np.float32)
                    )
    output = {}
    for target_name, state in states.items():
        if not state["labels"]:
            raise RuntimeError(f"No fit rows were eligible for target {target_name}")
        y = np.concatenate(state["labels"])
        sample_weights = np.concatenate(state["sample_weights"])
        features = {
            name: np.concatenate(rows)
            for name, rows in state["feature_rows"].items()
        }
        if len(y) > int(maximum_rows):
            rng = np.random.default_rng(int(seed) ^ int(state["target_seed"]))
            keep = np.sort(
                rng.choice(len(y), size=int(maximum_rows), replace=False)
            )
            y = y[keep]
            sample_weights = sample_weights[keep]
            features = {name: values[keep] for name, values in features.items()}
        if float(y.min()) == float(y.max()):
            raise RuntimeError(f"Fit sample has one class for target {target_name}")
        output[target_name] = (features, y, sample_weights)
    return output


def _fit_logistic(
    features: Mapping[str, np.ndarray],
    labels: np.ndarray,
    feature_names: Sequence[str],
    *,
    l2: float,
    sample_weights: np.ndarray | None = None,
    iterations: int = 30,
) -> LogisticModel:
    names = tuple(feature_names)
    matrix = np.column_stack(
        [np.asarray(features[name], dtype=np.float64) for name in names]
    )
    labels = np.asarray(labels, dtype=np.float64)
    if sample_weights is None:
        sample_weights = np.ones(len(labels), dtype=np.float64)
    else:
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if sample_weights.shape != labels.shape or not np.isfinite(
        sample_weights
    ).all() or np.any(sample_weights <= 0.0):
        raise ValueError("Logistic sampling weights must be finite and positive")
    sample_weights = sample_weights / sample_weights.mean()
    weight_sum = float(sample_weights.sum())
    means = (matrix * sample_weights[:, None]).sum(axis=0) / weight_sum
    scales = np.sqrt(
        (
            np.square(matrix - means) * sample_weights[:, None]
        ).sum(axis=0)
        / weight_sum
    )
    scales = np.where(scales >= 1e-6, scales, 1.0)
    matrix = np.clip((matrix - means) / scales, -30.0, 30.0)
    prevalence = float(
        np.clip(
            np.dot(labels, sample_weights) / weight_sum,
            1e-6,
            1.0 - 1e-6,
        )
    )
    intercept = math.log(prevalence / (1.0 - prevalence))
    coefficients = np.zeros(matrix.shape[1], dtype=np.float64)

    def objective(current_intercept, current_coefficients):
        logits = np.clip(
            current_intercept + matrix @ current_coefficients, -30.0, 30.0
        )
        nll = np.logaddexp(0.0, logits) - labels * logits
        return float(
            np.dot(nll, sample_weights) / weight_sum
            + 0.5
            * float(l2)
            * np.dot(current_coefficients, current_coefficients)
        )

    previous = objective(intercept, coefficients)
    for _ in range(int(iterations)):
        logits = np.clip(intercept + matrix @ coefficients, -30.0, 30.0)
        probability = _sigmoid(logits)
        residual = (probability - labels) * sample_weights
        weight = (
            np.clip(probability * (1.0 - probability), 1e-6, None)
            * sample_weights
        )
        gradient = np.concatenate(
            (
                [residual.sum() / weight_sum],
                matrix.T @ residual / weight_sum + l2 * coefficients,
            )
        )
        hessian = np.empty((len(names) + 1, len(names) + 1), dtype=np.float64)
        hessian[0, 0] = weight.sum() / weight_sum
        hessian[0, 1:] = (matrix * weight[:, None]).sum(axis=0) / weight_sum
        hessian[1:, 0] = hessian[0, 1:]
        hessian[1:, 1:] = (
            matrix.T @ (matrix * weight[:, None]) / weight_sum
            + float(l2) * np.eye(len(names), dtype=np.float64)
        )
        hessian += 1e-8 * np.eye(len(names) + 1, dtype=np.float64)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        accepted = False
        step_scale = 1.0
        for _ in range(12):
            candidate_intercept = intercept - step_scale * float(step[0])
            candidate_coefficients = coefficients - step_scale * step[1:]
            candidate = objective(candidate_intercept, candidate_coefficients)
            if math.isfinite(candidate) and candidate <= previous + 1e-10:
                intercept = candidate_intercept
                coefficients = candidate_coefficients
                accepted = True
                break
            step_scale *= 0.5
        if not accepted:
            break
        if abs(previous - candidate) < 1e-9:
            previous = candidate
            break
        previous = candidate
    if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
        raise FloatingPointError("Logistic calibration produced non-finite parameters")
    return LogisticModel(
        feature_names=names,
        means=means.astype(np.float64),
        scales=scales.astype(np.float64),
        intercept=float(intercept),
        coefficients=coefficients,
        fit_count=len(labels),
        fit_positive_fraction=prevalence,
    )


def _histogram_metrics(
    positive: np.ndarray,
    negative: np.ndarray,
    probability_sum: np.ndarray,
    *,
    nll_sum: float,
    brier_sum: float,
) -> Dict[str, float]:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    count = positive + negative
    total = float(count.sum())
    positive_total = float(positive.sum())
    negative_total = float(negative.sum())
    true_positive = np.cumsum(positive[::-1])
    false_positive = np.cumsum(negative[::-1])
    recall = true_positive / max(positive_total, 1.0)
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall_delta = np.diff(np.concatenate(([0.0], recall)))
    average_precision = float(np.sum(np.maximum(recall_delta, 0.0) * precision))
    if positive_total > 0.0 and negative_total > 0.0:
        negative_below = np.cumsum(negative) - negative
        roc_auc = float(
            np.sum(positive * (negative_below + 0.5 * negative))
            / (positive_total * negative_total)
        )
    else:
        roc_auc = float("nan")
    nonempty = count > 0.0
    mean_probability = np.zeros_like(count)
    mean_probability[nonempty] = probability_sum[nonempty] / count[nonempty]
    empirical = np.zeros_like(count)
    empirical[nonempty] = positive[nonempty] / count[nonempty]
    ece = float(
        np.sum(count[nonempty] * np.abs(mean_probability[nonempty] - empirical[nonempty]))
        / max(total, 1.0)
    )
    return {
        "count": total,
        "positive_fraction": positive_total / max(total, 1.0),
        "average_precision": average_precision,
        "roc_auc": roc_auc,
        "nll": float(nll_sum) / max(total, 1.0),
        "brier": float(brier_sum) / max(total, 1.0),
        "ece": ece,
    }


def _threshold_for_precision(
    positive: np.ndarray, negative: np.ndarray, minimum_precision: float
) -> int:
    true_positive = np.cumsum(np.asarray(positive, dtype=np.float64)[::-1])
    false_positive = np.cumsum(np.asarray(negative, dtype=np.float64)[::-1])
    precision = true_positive / np.maximum(true_positive + false_positive, 1.0)
    selected = true_positive + false_positive
    candidates = np.flatnonzero(
        (selected > 0.0) & (precision >= float(minimum_precision))
    )
    if len(candidates) == 0:
        return len(positive)
    reverse_offset = int(candidates[-1])
    return len(positive) - 1 - reverse_offset


def _threshold_for_best_iou(positive: np.ndarray, negative: np.ndarray) -> int:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive_total = float(positive.sum())
    true_positive = np.cumsum(positive[::-1])
    false_positive = np.cumsum(negative[::-1])
    false_negative = positive_total - true_positive
    iou = true_positive / np.maximum(
        true_positive + false_positive + false_negative, 1.0
    )
    best_reverse_offset = int(np.argmax(iou))
    return len(positive) - 1 - best_reverse_offset


def _counts_at_threshold(
    positive: np.ndarray, negative: np.ndarray, threshold_index: int
) -> Dict[str, float]:
    threshold_index = min(max(int(threshold_index), 0), len(positive))
    true_positive = float(np.asarray(positive)[threshold_index:].sum())
    false_positive = float(np.asarray(negative)[threshold_index:].sum())
    positive_total = float(np.asarray(positive).sum())
    total = positive_total + float(np.asarray(negative).sum())
    selected = true_positive + false_positive
    false_negative = positive_total - true_positive
    return {
        "threshold_probability": float(threshold_index) / len(positive),
        "precision": true_positive / max(selected, 1.0),
        "error_recall": true_positive / max(positive_total, 1.0),
        "iou": true_positive
        / max(true_positive + false_positive + false_negative, 1.0),
        "f1": 2.0 * true_positive
        / max(2.0 * true_positive + false_positive + false_negative, 1.0),
        "action_coverage": selected / max(total, 1.0),
        "selected_count": selected,
        "positive_selected": true_positive,
    }


def _average_precision_from_small_hist(positive, negative):
    metrics = _histogram_metrics(
        positive,
        negative,
        np.zeros_like(np.asarray(positive, dtype=np.float64)),
        nll_sum=0.0,
        brier_sum=0.0,
    )
    return float(metrics["average_precision"])


def _bootstrap_mean(values: np.ndarray, seed: int, replicates: int):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) == 1 or int(replicates) <= 0:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        boot[index] = values[
            rng.integers(0, len(values), size=len(values))
        ].mean()
    return mean, float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _evaluate_targets(
    bundle: ArtifactBundle,
    targets: Sequence[TargetSpec],
    models_by_target: Mapping[str, Mapping[str, LogisticModel]],
    *,
    bootstrap_replicates: int,
    seed: int,
):
    states = {
        target.name: {
            model_name: {
                "positive": np.zeros(PROBABILITY_BINS, dtype=np.int64),
                "negative": np.zeros(PROBABILITY_BINS, dtype=np.int64),
                "probability_sum": np.zeros(PROBABILITY_BINS, dtype=np.float64),
                "nll_sum": 0.0,
                "brier_sum": 0.0,
                "sequences": {},
            }
            for model_name in models_by_target[target.name]
        }
        for target in targets
    }
    stackers = {
        target.name: _stacker_matrix(models_by_target[target.name])
        for target in targets
    }
    shard_count = len(bundle.signals["contact"].shards)
    for shard_index, shard in enumerate(bundle.iter_shards(), start=1):
        print(
            f"[audit] eval split={bundle.split} shard={shard_index}/{shard_count}",
            flush=True,
        )
        sequence_labels = np.char.add(
            np.char.add(shard["sequence_keys"].astype(str), "::"),
            shard["query_aliases"].astype(str),
        )
        for target in targets:
            eligible, labels = target.labels(
                shard["base_predictions"], shard["targets"]
            )
            eligible_flat = np.flatnonzero(eligible.reshape(-1))
            if len(eligible_flat) == 0:
                continue
            labels_flat = labels.reshape(-1)
            row_width = int(eligible.shape[1])
            model_names, feature_names, intercepts, weights = stackers[target.name]
            intercepts = intercepts.astype(np.float32, copy=False)
            weights = weights.astype(np.float32, copy=False)
            flattened_features = {
                name: shard["features"][name].reshape(-1)
                for name in feature_names
            }
            for chunk_start in range(
                0, len(eligible_flat), EVALUATION_POINT_CHUNK
            ):
                point_indices = eligible_flat[
                    chunk_start : chunk_start + EVALUATION_POINT_CHUNK
                ]
                selected_labels = labels_flat[point_indices]
                matrix = np.empty(
                    (len(point_indices), len(feature_names)), dtype=np.float32
                )
                for feature_index, feature_name in enumerate(feature_names):
                    matrix[:, feature_index] = flattened_features[feature_name][
                        point_indices
                    ]
                probabilities = matrix @ weights.T
                probabilities += intercepts[None]
                np.clip(probabilities, -30.0, 30.0, out=probabilities)
                np.negative(probabilities, out=probabilities)
                np.exp(probabilities, out=probabilities)
                probabilities += 1.0
                np.reciprocal(probabilities, out=probabilities)
                np.clip(
                    probabilities,
                    EPSILON,
                    1.0 - EPSILON,
                    out=probabilities,
                )

                for model_index, model_name in enumerate(model_names):
                    selected_probability = probabilities[:, model_index]
                    bins = np.minimum(
                        (
                            selected_probability * PROBABILITY_BINS
                        ).astype(np.int64),
                        PROBABILITY_BINS - 1,
                    )
                    state = states[target.name][model_name]
                    state["positive"] += np.bincount(
                        bins[selected_labels], minlength=PROBABILITY_BINS
                    )
                    state["negative"] += np.bincount(
                        bins[~selected_labels], minlength=PROBABILITY_BINS
                    )
                    state["probability_sum"] += np.bincount(
                        bins,
                        weights=selected_probability,
                        minlength=PROBABILITY_BINS,
                    )
                    state["nll_sum"] += float(
                        -np.log(
                            np.where(
                                selected_labels,
                                selected_probability,
                                1.0 - selected_probability,
                            )
                        ).sum()
                    )
                    state["brier_sum"] += float(
                        np.square(
                            selected_probability
                            - selected_labels.astype(np.float32)
                        ).sum()
                    )

                if not target.sequence_metrics:
                    continue
                row_indices = point_indices // row_width
                row_boundaries = np.flatnonzero(np.diff(row_indices)) + 1
                row_starts = np.concatenate(([0], row_boundaries))
                row_stops = np.concatenate((row_boundaries, [len(row_indices)]))
                for row_start, row_stop in zip(row_starts, row_stops):
                    row_index = int(row_indices[row_start])
                    row_labels = selected_labels[row_start:row_stop]
                    sequence_key = str(sequence_labels[row_index])
                    for model_index, model_name in enumerate(model_names):
                        row_probability = probabilities[
                            row_start:row_stop, model_index
                        ]
                        row_bins = np.minimum(
                            (row_probability * SEQUENCE_BINS).astype(np.int64),
                            SEQUENCE_BINS - 1,
                        )
                        sequence = states[target.name][model_name][
                            "sequences"
                        ].setdefault(
                            sequence_key,
                            {
                                "positive": np.zeros(
                                    SEQUENCE_BINS, dtype=np.int64
                                ),
                                "negative": np.zeros(
                                    SEQUENCE_BINS, dtype=np.int64
                                ),
                                "nll_sum": 0.0,
                                "brier_sum": 0.0,
                                "count": 0,
                            },
                        )
                        sequence["positive"] += np.bincount(
                            row_bins[row_labels], minlength=SEQUENCE_BINS
                        )
                        sequence["negative"] += np.bincount(
                            row_bins[~row_labels], minlength=SEQUENCE_BINS
                        )
                        sequence["nll_sum"] += float(
                            -np.log(
                                np.where(
                                    row_labels,
                                    row_probability,
                                    1.0 - row_probability,
                                )
                            ).sum()
                        )
                        sequence["brier_sum"] += float(
                            np.square(
                                row_probability
                                - row_labels.astype(np.float32)
                            ).sum()
                        )
                        sequence["count"] += int(len(row_labels))

    evaluated = {}
    sequence_rows = []
    for target in targets:
        summaries = {}
        for model_name, state in states[target.name].items():
            summary = _histogram_metrics(
                state["positive"],
                state["negative"],
                state["probability_sum"],
                nll_sum=state["nll_sum"],
                brier_sum=state["brier_sum"],
            )
            sequence_ap = []
            sequence_nll = []
            sequence_brier = []
            for sequence_key, sequence in sorted(state["sequences"].items()):
                ap = _average_precision_from_small_hist(
                    sequence["positive"], sequence["negative"]
                )
                nll = sequence["nll_sum"] / max(sequence["count"], 1)
                brier = sequence["brier_sum"] / max(sequence["count"], 1)
                sequence_ap.append(ap)
                sequence_nll.append(nll)
                sequence_brier.append(brier)
                sequence_rows.append(
                    {
                        "split": bundle.split,
                        "target": target.name,
                        "model": model_name,
                        "sequence_key": sequence_key,
                        "count": sequence["count"],
                        "average_precision": ap,
                        "nll": nll,
                        "brier": brier,
                    }
                )
            for metric_name, values, offset in (
                ("sequence_macro_ap", sequence_ap, 0),
                ("sequence_macro_nll", sequence_nll, 1),
                ("sequence_macro_brier", sequence_brier, 2),
            ):
                mean, low, high = _bootstrap_mean(
                    np.asarray(values),
                    seed=int(seed)
                    + zlib.crc32(
                        (
                            f"{bundle.split}:{target.name}:"
                            f"{model_name}:{offset}"
                        ).encode()
                    ),
                    replicates=bootstrap_replicates,
                )
                summary[metric_name] = mean
                summary[f"{metric_name}_ci_low"] = low
                summary[f"{metric_name}_ci_high"] = high
            summaries[model_name] = summary
        baseline_sequences = states[target.name]["B"]["sequences"]
        for model_name, state in states[target.name].items():
            common_sequences = sorted(
                set(baseline_sequences).intersection(state["sequences"])
            )
            for metric_name, offset in (
                ("ap", 10),
                ("nll", 11),
                ("brier", 12),
            ):
                deltas = []
                for sequence_key in common_sequences:
                    baseline_sequence = baseline_sequences[sequence_key]
                    current_sequence = state["sequences"][sequence_key]
                    if metric_name == "ap":
                        baseline_value = _average_precision_from_small_hist(
                            baseline_sequence["positive"],
                            baseline_sequence["negative"],
                        )
                        current_value = _average_precision_from_small_hist(
                            current_sequence["positive"],
                            current_sequence["negative"],
                        )
                    else:
                        baseline_value = baseline_sequence[
                            f"{metric_name}_sum"
                        ] / max(baseline_sequence["count"], 1)
                        current_value = current_sequence[
                            f"{metric_name}_sum"
                        ] / max(current_sequence["count"], 1)
                    deltas.append(current_value - baseline_value)
                mean, low, high = _bootstrap_mean(
                    np.asarray(deltas),
                    seed=int(seed)
                    + zlib.crc32(
                        (
                            f"{bundle.split}:{target.name}:{model_name}:"
                            f"paired:{offset}"
                        ).encode()
                    ),
                    replicates=bootstrap_replicates,
                )
                summaries[model_name][f"sequence_delta_{metric_name}_vs_B"] = mean
                summaries[model_name][
                    f"sequence_delta_{metric_name}_vs_B_ci_low"
                ] = low
                summaries[model_name][
                    f"sequence_delta_{metric_name}_vs_B_ci_high"
                ] = high
        for state in states[target.name].values():
            state["sequences"].clear()
        evaluated[target.name] = {
            "summary": summaries,
            "states": states[target.name],
        }
    return evaluated, sequence_rows


def _target_specs(ordinal_thresholds: Iterable[float]):
    targets = [
        TargetSpec("contact_clear_0.10", "contact"),
        TargetSpec("false_high_basepos", "false_high"),
        TargetSpec("false_low_baseneg", "false_low"),
    ]
    targets.extend(
        TargetSpec(f"ordinal_{threshold:g}", "ordinal", float(threshold))
        for threshold in ordinal_thresholds
    )
    return targets


def _score_correlation_rows(
    fit_samples: Mapping[
        str,
        Tuple[Mapping[str, np.ndarray], np.ndarray, np.ndarray],
    ]
) -> List[Dict[str, object]]:
    rows = []
    for target_name, (features, labels, sample_weights) in fit_samples.items():
        names = sorted(features)
        labels = np.asarray(labels, dtype=bool)
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        for first_index, first_name in enumerate(names):
            first = np.asarray(features[first_name], dtype=np.float64)
            for second_name in names[first_index + 1 :]:
                second = np.asarray(features[second_name], dtype=np.float64)
                weight_sum = float(sample_weights.sum())
                first_mean = float(
                    np.dot(first, sample_weights) / weight_sum
                )
                second_mean = float(
                    np.dot(second, sample_weights) / weight_sum
                )
                first_centered = first - first_mean
                second_centered = second - second_mean
                denominator = math.sqrt(
                    float(
                        np.dot(
                            sample_weights,
                            np.square(first_centered),
                        )
                    )
                    * float(
                        np.dot(
                            sample_weights,
                            np.square(second_centered),
                        )
                    )
                )
                correlation = (
                    float(
                        np.dot(
                            sample_weights,
                            first_centered * second_centered,
                        )
                    )
                    / denominator
                    if denominator > 0.0
                    else float("nan")
                )
                positive_weights = sample_weights[labels]
                negative_weights = sample_weights[~labels]
                rows.append(
                    {
                        "fit_split": "val",
                        "target": target_name,
                        "first_score": first_name,
                        "second_score": second_name,
                        "pearson_correlation": correlation,
                        "first_positive_mean": float(
                            np.dot(first[labels], positive_weights)
                            / positive_weights.sum()
                        ),
                        "first_negative_mean": float(
                            np.dot(first[~labels], negative_weights)
                            / negative_weights.sum()
                        ),
                        "second_positive_mean": float(
                            np.dot(second[labels], positive_weights)
                            / positive_weights.sum()
                        ),
                        "second_negative_mean": float(
                            np.dot(second[~labels], negative_weights)
                            / negative_weights.sum()
                        ),
                        "count": len(labels),
                        "sampling_weight_sum": weight_sum,
                    }
                )
    return rows


def run_audit(args) -> None:
    artifact_root = Path(args.artifact_root).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = {
        split: ArtifactBundle(artifact_root, split) for split in args.splits
    }
    if "val" not in bundles:
        raise ValueError("The audit requires a val artifact split")
    ordinal_thresholds = bundles["val"].ordinal_thresholds
    for split, bundle in bundles.items():
        if bundle.ordinal_thresholds != ordinal_thresholds:
            raise RuntimeError(
                f"Ordinal thresholds differ for split {split}: "
                f"{bundle.ordinal_thresholds} vs {ordinal_thresholds}"
            )
    targets = _target_specs(ordinal_thresholds)
    calibrators = {}
    print("[audit] fitting validation-only logistic stackers", flush=True)
    fit_samples = _collect_fit_samples(
        bundles["val"],
        targets,
        vertices_per_frame=args.fit_vertices_per_frame,
        maximum_rows=args.fit_max_rows,
        seed=args.seed,
    )
    correlation_rows = _score_correlation_rows(fit_samples)
    for target in targets:
        features, labels, sample_weights = fit_samples[target.name]
        specs = _model_specs(target, bundles["val"].ordinal_names)
        calibrators[target.name] = {
            model_name: _fit_logistic(
                features,
                labels,
                feature_names,
                l2=args.logistic_l2,
                sample_weights=sample_weights,
            )
            for model_name, feature_names in specs.items()
        }
        print(
            f"[audit] fitted target={target.name} rows={len(labels)} "
            f"models={len(specs)}",
            flush=True,
        )

    calibrator_payload = {
        "schema": AUDIT_SCHEMA,
        "fit_split": "val",
        "seed": int(args.seed),
        "fit_vertices_per_frame": int(args.fit_vertices_per_frame),
        "fit_max_rows": int(args.fit_max_rows),
        "fit_sampling_weight": (
            "eligible_vertices_in_frame / selected_vertices_in_frame"
        ),
        "class_weighting": False,
        "logistic_l2": float(args.logistic_l2),
        "ordinal_thresholds": list(ordinal_thresholds),
        "models": {
            target_name: {
                model_name: model.to_dict()
                for model_name, model in target_models.items()
            }
            for target_name, target_models in calibrators.items()
        },
    }
    _atomic_write_text(
        output_dir / "calibrators.json",
        json.dumps(_finite(calibrator_payload), indent=2, sort_keys=True) + "\n",
    )

    summary_rows = []
    sequence_rows = []
    precision_rows = []
    evaluated = {}
    print("[audit] evaluating locked models", flush=True)
    for split, bundle in bundles.items():
        evaluated[split], split_sequence_rows = _evaluate_targets(
            bundle,
            targets,
            calibrators,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        sequence_rows.extend(split_sequence_rows)
        for target in targets:
            summaries = evaluated[split][target.name]["summary"]
            baseline = summaries["B"]
            for model_name, summary in summaries.items():
                row = {
                    "split": split,
                    "target": target.name,
                    "model": model_name,
                }
                row.update(summary)
                row.update(
                    {
                        "delta_ap_vs_B": summary["average_precision"]
                        - baseline["average_precision"],
                        "delta_nll_vs_B": summary["nll"] - baseline["nll"],
                        "delta_brier_vs_B": summary["brier"] - baseline["brier"],
                    }
                )
                summary_rows.append(row)
            print(f"[audit] evaluated split={split} target={target.name}", flush=True)

    for target in targets:
        validation = evaluated["val"][target.name]
        for model_name, val_state in validation["states"].items():
            selection_rules = [
                (
                    "best_iou",
                    "",
                    _threshold_for_best_iou(
                        val_state["positive"], val_state["negative"]
                    ),
                )
            ]
            selection_rules.extend(
                (
                    "minimum_precision",
                    minimum_precision,
                    _threshold_for_precision(
                        val_state["positive"],
                        val_state["negative"],
                        minimum_precision,
                    ),
                )
                for minimum_precision in PRECISION_LEVELS
            )
            for selection_rule, minimum_precision, threshold_index in selection_rules:
                for split in args.splits:
                    state = evaluated[split][target.name]["states"][model_name]
                    counts = _counts_at_threshold(
                        state["positive"], state["negative"], threshold_index
                    )
                    precision_rows.append(
                        {
                            "fit_split": "val",
                            "eval_split": split,
                            "target": target.name,
                            "model": model_name,
                            "selection_rule": selection_rule,
                            "minimum_val_precision": minimum_precision,
                            **counts,
                        }
                    )

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "precision_coverage.csv", precision_rows)
    _write_csv(output_dir / "sequence_metrics.csv", sequence_rows)
    _write_csv(output_dir / "score_correlations.csv", correlation_rows)
    audit_config = {
        "schema": AUDIT_SCHEMA,
        "artifact_root": str(artifact_root),
        "output_dir": str(output_dir),
        "splits": list(args.splits),
        "signals": {
            split: {
                signal: bundle.signals[signal].config
                for signal in SIGNALS
            }
            for split, bundle in bundles.items()
        },
        "pressure_modified": False,
        "summary": str(output_dir / "summary.csv"),
        "precision_coverage": str(output_dir / "precision_coverage.csv"),
        "sequence_metrics": str(output_dir / "sequence_metrics.csv"),
        "score_correlations": str(output_dir / "score_correlations.csv"),
        "calibrators": str(output_dir / "calibrators.json"),
    }
    _atomic_write_text(
        output_dir / "audit_config.json",
        json.dumps(_finite(audit_config), indent=2, sort_keys=True) + "\n",
    )
    report_lines = [
        "# RGB Contact/Ordinal Sufficiency Audit",
        "",
        "This audit never modifies the frozen pressure prediction. Logistic stackers",
        "are fit on validation artifacts and applied unchanged to seen/unseen tests.",
        "",
        "## Key locked-test results",
        "",
        "| Split | Target | Locked model | AP | Delta vs B | NLL |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in args.splits:
        if split == "val":
            continue
        for target_name in (
            "contact_clear_0.10",
            "false_high_basepos",
            "false_low_baseneg",
        ):
            fixed_models = ["B", "B+C+O"]
            if target_name == "false_high_basepos":
                fixed_models.extend(("B+C+O+D", "B+C+O+Dctl"))
            for model_name in fixed_models:
                row = next(
                    row
                    for row in summary_rows
                    if row["split"] == split
                    and row["target"] == target_name
                    and row["model"] == model_name
                )
                report_lines.append(
                    f"| {split} | {target_name} | {model_name} | "
                    f"{row['average_precision']:.6f} | "
                    f"{row['delta_ap_vs_B']:+.6f} | {row['nll']:.6f} |"
                )
    report_lines.extend(
        [
            "",
            "Formal interpretation must use locked test metrics and val-selected",
            "precision thresholds. Candidate-restricted false-high metrics must not be",
            "compared directly with full-palm Contact AP.",
            "",
        ]
    )
    _atomic_write_text(output_dir / "REPORT.md", "\n".join(report_lines))
    print(f"[audit] complete: {output_dir}", flush=True)


def self_test() -> None:
    rng = np.random.default_rng(521)
    x = rng.normal(size=4000).astype(np.float32)
    probability = _sigmoid(1.2 * x - 0.3)
    labels = rng.random(len(x)) < probability
    model = _fit_logistic(
        {"x": x}, labels.astype(np.float32), ("x",), l2=1e-3
    )
    prediction = model.predict({"x": x})
    if not np.isfinite(prediction).all() or prediction.min() <= 0.0 or prediction.max() >= 1.0:
        raise AssertionError("Logistic self-test produced invalid probabilities")
    bins = np.minimum(
        (prediction * PROBABILITY_BINS).astype(np.int64), PROBABILITY_BINS - 1
    )
    positive = np.bincount(bins[labels], minlength=PROBABILITY_BINS)
    negative = np.bincount(bins[~labels], minlength=PROBABILITY_BINS)
    probability_sum = np.bincount(
        bins, weights=prediction, minlength=PROBABILITY_BINS
    )
    metrics = _histogram_metrics(
        positive,
        negative,
        probability_sum,
        nll_sum=float(
            -np.log(np.where(labels, prediction, 1.0 - prediction)).sum()
        ),
        brier_sum=float(np.square(prediction - labels.astype(np.float32)).sum()),
    )
    if metrics["average_precision"] <= metrics["positive_fraction"]:
        raise AssertionError("Logistic self-test did not learn useful ranking")
    second = LogisticModel(
        feature_names=("x",),
        means=model.means.copy(),
        scales=model.scales.copy(),
        intercept=model.intercept + 0.25,
        coefficients=model.coefficients.copy(),
        fit_count=model.fit_count,
        fit_positive_fraction=model.fit_positive_fraction,
    )
    names, feature_names, intercepts, weights = _stacker_matrix(
        {"first": model, "second": second}
    )
    matrix = np.column_stack([{"x": x}[name] for name in feature_names])
    stacked = _sigmoid(matrix @ weights.T + intercepts[None])
    expected = np.column_stack(
        [model.predict({"x": x}), second.predict({"x": x})]
    )
    if names != ("first", "second") or not np.allclose(
        stacked, expected, rtol=1e-5, atol=1e-6
    ):
        raise AssertionError("Vectorized stacker does not match scalar prediction")
    for precision in PRECISION_LEVELS:
        threshold = _threshold_for_precision(positive, negative, precision)
        counts = _counts_at_threshold(positive, negative, threshold)
        if counts["selected_count"] > 0 and counts["precision"] + 1e-9 < precision:
            raise AssertionError("Precision threshold self-test failed")
    print("Selector sufficiency tiny checks passed.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit RGB contact/ordinal sufficiency without changing pressure"
    )
    parser.add_argument("--artifact-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--splits", type=str, default=",".join(DEFAULT_SPLITS)
    )
    parser.add_argument("--fit-vertices-per-frame", type=int, default=64)
    parser.add_argument("--fit-max-rows", type=int, default=2000000)
    parser.add_argument("--logistic-l2", type=float, default=1e-3)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.artifact_root or not args.output_dir:
        parser.error("--artifact-root and --output-dir are required")
    args.splits = tuple(
        value.strip() for value in str(args.splits).split(",") if value.strip()
    )
    if "val" not in args.splits:
        parser.error("--splits must include val")
    if args.fit_vertices_per_frame < 1 or args.fit_max_rows < 100:
        parser.error("Fit sampling limits are too small")
    if args.logistic_l2 <= 0.0:
        parser.error("--logistic-l2 must be positive")
    run_audit(args)


if __name__ == "__main__":
    main()
