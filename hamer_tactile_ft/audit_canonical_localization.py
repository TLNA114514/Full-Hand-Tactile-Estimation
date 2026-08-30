#!/usr/bin/env python3
"""Audit where sensor-independent canonical tactile localization is lost.

This audit consumes the immutable FullGrid feature cache. It does not train a
model or expose ground-truth information to a deployable path. The outputs
separate pressure magnitude from normalized location, measure target topology,
test a continuous multiscale canonical surface basis, and look for visually
similar examples with incompatible pressure layouts. No basis anchor is derived
from a tactile sensor location or layout.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.audit_local_controllability import (
    CacheGroup,
    _available_fields,
    _base_logits_batch,
    _load_decoder,
    _palm_mask,
)
from hamer_tactile_ft.tactile_metrics import (
    location_distribution_stats,
    volumetric_iou_stats,
)
from tactile_input_priors.feature_cache import sha256_file

DEFAULT_MESH = (
    REPO_ROOT / "opentouch" / "preprocess" / "scratch" / "mano_right_neutral_subdiv.obj"
)
DEFAULT_PALM_FACES = (
    REPO_ROOT
    / "opentouch"
    / "preprocess"
    / "scratch"
    / "auto_calibrated_palm_subdiv_faces.json"
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _finite_mean(values: Iterable[float]) -> float:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(finite.mean()) if finite.size else float("nan")


def _finite_percentile(values: Iterable[float], percentile: float) -> float:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Expected positive comma-separated integers")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("Values must be strictly increasing")
    return values


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not values or any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("Expected finite comma-separated values")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("Values must be strictly increasing")
    return values


class MetricAccumulator:
    """Streaming point and frame metrics over the valid canonical palm."""

    def __init__(self, name: str):
        self.name = str(name)
        self.point_count = 0
        self.square_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.frame_rmse: list[float] = []
        self.frame_mae: list[float] = []
        self.support_iou: list[float] = []
        self.contact_iou: list[float] = []
        self.viou: list[float] = []
        self.distribution_viou: list[float] = []
        self.core_distribution_viou: list[float] = []
        self.pred_volume: list[float] = []
        self.gt_volume: list[float] = []
        self.volume_absolute_error: list[float] = []
        self.high_gt_prediction_sum = 0.0
        self.high_gt_prediction_count = 0
        self.false_high_excess: list[float] = []

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(prediction, dtype=np.float64)
        gt = np.asarray(target, dtype=np.float64)
        if pred.shape != gt.shape or pred.ndim != 2:
            raise ValueError(
                f"Expected matching [B,V] arrays, got {pred.shape}, {gt.shape}"
            )
        finite = np.isfinite(pred) & np.isfinite(gt)
        pred = np.where(finite, np.clip(pred, 0.0, 1.0), 0.0)
        gt = np.where(finite, np.clip(gt, 0.0, 1.0), 0.0)
        error = pred - gt
        self.point_count += int(finite.sum())
        self.square_error_sum += float(np.where(finite, error * error, 0.0).sum())
        self.absolute_error_sum += float(np.where(finite, np.abs(error), 0.0).sum())
        denominator = np.maximum(finite.sum(axis=1), 1)
        self.frame_rmse.extend(
            np.sqrt(
                np.where(finite, error * error, 0.0).sum(axis=1) / denominator
            ).tolist()
        )
        self.frame_mae.extend(
            (np.where(finite, np.abs(error), 0.0).sum(axis=1) / denominator).tolist()
        )
        for threshold, destination in (
            (0.05, self.support_iou),
            (0.10, self.contact_iou),
        ):
            pred_active = (pred >= threshold) & finite
            gt_active = (gt >= threshold) & finite
            intersection = np.logical_and(pred_active, gt_active).sum(axis=1)
            union = np.logical_or(pred_active, gt_active).sum(axis=1)
            destination.extend(
                np.divide(
                    intersection,
                    union,
                    out=np.ones_like(intersection, dtype=np.float64),
                    where=union > 0,
                ).tolist()
            )
        self.viou.extend(volumetric_iou_stats(pred, gt).per_frame.tolist())
        location = location_distribution_stats(pred, gt, min_gt_volume=1.0)
        core = location_distribution_stats(
            pred,
            gt,
            min_gt_volume=1.0,
            distribution_power=2.0,
            min_gt_peak=0.05,
        )
        self.distribution_viou.extend(location.distribution_viou.tolist())
        self.core_distribution_viou.extend(core.distribution_viou.tolist())
        pred_volume = pred.sum(axis=1)
        gt_volume = gt.sum(axis=1)
        self.pred_volume.extend(pred_volume.tolist())
        self.gt_volume.extend(gt_volume.tolist())
        self.volume_absolute_error.extend(np.abs(pred_volume - gt_volume).tolist())
        high = (gt >= 0.70) & finite
        self.high_gt_prediction_sum += float(pred[high].sum())
        self.high_gt_prediction_count += int(high.sum())
        low = (gt < 0.005) & finite
        self.false_high_excess.extend(
            np.where(low, np.maximum(pred - 0.005, 0.0), 0.0).sum(axis=1).tolist()
        )

    def summary(self) -> dict[str, Any]:
        has_points = self.point_count > 0
        return {
            "variant": self.name,
            "frame_count": len(self.frame_rmse),
            "point_count": self.point_count,
            "rmse_vertex_micro": (
                math.sqrt(self.square_error_sum / self.point_count)
                if has_points
                else float("nan")
            ),
            "mae_vertex_micro": (
                self.absolute_error_sum / self.point_count
                if has_points
                else float("nan")
            ),
            "rmse_frame_macro": _finite_mean(self.frame_rmse),
            "mae_frame_macro": _finite_mean(self.frame_mae),
            "support_iou_005_frame_macro": _finite_mean(self.support_iou),
            "contact_iou_010_frame_macro": _finite_mean(self.contact_iou),
            "volumetric_iou_frame_macro": _finite_mean(self.viou),
            "distribution_viou_frame_macro": _finite_mean(self.distribution_viou),
            "core_distribution_viou_frame_macro": _finite_mean(
                self.core_distribution_viou
            ),
            "pred_volume_mean": _finite_mean(self.pred_volume),
            "gt_volume_mean": _finite_mean(self.gt_volume),
            "volume_absolute_error_mean": _finite_mean(self.volume_absolute_error),
            "gt_ge_070_mean_prediction": (
                self.high_gt_prediction_sum / self.high_gt_prediction_count
                if self.high_gt_prediction_count
                else float("nan")
            ),
            "false_high_excess_mean": _finite_mean(self.false_high_excess),
        }


def _load_mesh_and_palm_graph(
    mesh_path: Path,
    palm_faces_path: Path,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    vertices: list[list[float]] = []
    with mesh_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    coordinates = np.asarray(vertices, dtype=np.float32)
    if coordinates.shape != (len(valid_mask), 3):
        raise ValueError(
            f"Mesh shape {coordinates.shape} does not match tactile dimension {len(valid_mask)}"
        )
    with palm_faces_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    faces = payload.get("group_negative", {}).get("face_triplets")
    if faces is None:
        raise KeyError(f"No group_negative.face_triplets in {palm_faces_path}")
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Palm face array must be [F,3], got {faces.shape}")
    valid_indices = np.flatnonzero(valid_mask)
    full_to_valid = np.full(len(valid_mask), -1, dtype=np.int64)
    full_to_valid[valid_indices] = np.arange(len(valid_indices), dtype=np.int64)
    neighbors = [set() for _ in range(len(valid_indices))]
    for face in faces:
        local = full_to_valid[face]
        if np.any(local < 0):
            continue
        a, b, c = (int(value) for value in local)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    adjacency = tuple(
        np.asarray(sorted(values), dtype=np.int32) for values in neighbors
    )
    return coordinates, valid_indices, adjacency


def _adjacency_csr(
    adjacency: Sequence[np.ndarray],
    coordinates: np.ndarray | None = None,
):
    rows = np.repeat(
        np.arange(len(adjacency), dtype=np.int64),
        np.asarray([len(values) for values in adjacency], dtype=np.int64),
    )
    columns = (
        np.concatenate(adjacency).astype(np.int64, copy=False)
        if rows.size
        else np.zeros(0, dtype=np.int64)
    )
    if coordinates is None:
        values = np.ones(len(rows), dtype=np.uint8)
    else:
        points = np.asarray(coordinates, dtype=np.float64)
        if points.shape != (len(adjacency), 3):
            raise ValueError(
                f"Expected [{len(adjacency)},3] coordinates, got {points.shape}"
            )
        values = np.linalg.norm(points[rows] - points[columns], axis=1)
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Palm graph contains a non-finite or zero-length edge")
    return coo_matrix(
        (values, (rows, columns)),
        shape=(len(adjacency), len(adjacency)),
    ).tocsr()


def _connected_component_count(adjacency: Sequence[np.ndarray]) -> int:
    remaining = np.ones(len(adjacency), dtype=bool)
    count = 0
    for start in range(len(adjacency)):
        if not remaining[start]:
            continue
        count += 1
        remaining[start] = False
        queue: deque[int] = deque((start,))
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                neighbor = int(neighbor)
                if remaining[neighbor]:
                    remaining[neighbor] = False
                    queue.append(neighbor)
    return count


def _geodesic_fps(
    coordinates: np.ndarray,
    adjacency: Sequence[np.ndarray],
    maximum_count: int,
    *,
    weighted: bool = False,
) -> np.ndarray:
    if not 1 <= int(maximum_count) <= len(adjacency):
        raise ValueError("maximum_count must fit the valid palm graph")
    centroid = coordinates.mean(axis=0, keepdims=True)
    first = int(np.square(coordinates - centroid).sum(axis=1).argmax())
    selected: list[int] = []
    selected_mask = np.zeros(len(adjacency), dtype=bool)
    minimum = np.full(len(adjacency), np.inf, dtype=np.float64)
    graph = _adjacency_csr(adjacency, coordinates if weighted else None)
    current = first
    for _ in range(int(maximum_count)):
        selected.append(current)
        selected_mask[current] = True
        distances = shortest_path(
            graph,
            directed=False,
            indices=current,
            unweighted=not weighted,
        )
        minimum = np.minimum(minimum, distances)
        candidate = minimum.copy()
        candidate[selected_mask] = -1
        current = int(candidate.argmax())
        if maximum_count >= 128 and (
            len(selected) == maximum_count or len(selected) % 128 == 0
        ):
            print(
                f"[canonical-basis] selected {len(selected)}/{maximum_count} "
                "geodesic anchors",
                flush=True,
            )
    return np.asarray(selected, dtype=np.int64)


def _multi_source_owner(
    adjacency: Sequence[np.ndarray], anchors: np.ndarray
) -> np.ndarray:
    owner = np.full(len(adjacency), -1, dtype=np.int32)
    distance = np.full(len(adjacency), np.iinfo(np.int32).max, dtype=np.int32)
    queue: deque[int] = deque()
    for anchor_index, vertex in enumerate(np.asarray(anchors, dtype=np.int64)):
        owner[int(vertex)] = int(anchor_index)
        distance[int(vertex)] = 0
        queue.append(int(vertex))
    while queue:
        vertex = queue.popleft()
        next_distance = int(distance[vertex]) + 1
        source_owner = int(owner[vertex])
        for neighbor in adjacency[vertex]:
            neighbor = int(neighbor)
            if next_distance < distance[neighbor]:
                distance[neighbor] = next_distance
                owner[neighbor] = source_owner
                queue.append(neighbor)
    if np.any(owner < 0):
        raise RuntimeError(
            "Palm graph has a component without an anchor; increase the smallest patch count"
        )
    return owner


class PatchPartition:
    def __init__(self, owner: np.ndarray):
        self.owner = np.asarray(owner, dtype=np.int64)
        self.patch_count = int(self.owner.max()) + 1
        self.order = np.argsort(self.owner, kind="stable")
        sorted_owner = self.owner[self.order]
        self.counts = np.bincount(sorted_owner, minlength=self.patch_count).astype(
            np.int64
        )
        if np.any(self.counts <= 0):
            raise RuntimeError("Canonical patch partition contains an empty patch")
        self.starts = np.r_[0, np.cumsum(self.counts)[:-1]].astype(np.int64)

    def sums(self, values: np.ndarray) -> np.ndarray:
        ordered = np.asarray(values)[:, self.order]
        return np.add.reduceat(ordered, self.starts, axis=1)

    def means(self, values: np.ndarray) -> np.ndarray:
        return self.sums(values) / self.counts[None]

    def reconstruct_means(self, values: np.ndarray) -> np.ndarray:
        return self.means(values)[:, self.owner]

    def active(self, values: np.ndarray, threshold: float) -> np.ndarray:
        active = (np.asarray(values)[:, self.order] >= float(threshold)).astype(
            np.int16
        )
        return np.add.reduceat(active, self.starts, axis=1) > 0


def _surface_basis_banks(
    coordinates: np.ndarray,
    adjacency: Sequence[np.ndarray],
    anchor_prefix: np.ndarray,
    anchor_counts: Sequence[int],
    *,
    bandwidth_scale: float,
    support_sigma: float,
    bandwidth_policy: str = "edge_floor",
    target_support_count: int = 6,
    diagnostics: dict[int, dict[str, Any]] | None = None,
) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Build nested, sensor-independent geodesic RBF banks.

    Each scale is a partition of unity over the canonical palm. Concatenating
    scales therefore gives a continuous multiresolution dictionary without
    assigning a hard patch or a sensor identity to any output location.
    """

    bandwidth_policy = str(bandwidth_policy).strip().lower()
    if bandwidth_policy not in {"edge_floor", "target_overlap"}:
        raise ValueError(
            "bandwidth_policy must be 'edge_floor' or 'target_overlap'"
        )
    if int(target_support_count) <= 0:
        raise ValueError("target_support_count must be positive")

    maximum_count = max(int(value) for value in anchor_counts)
    anchors = np.asarray(anchor_prefix[:maximum_count], dtype=np.int64)
    graph = _adjacency_csr(adjacency, coordinates)
    distances = shortest_path(
        graph,
        directed=False,
        indices=anchors,
        unweighted=False,
    )
    if distances.shape != (maximum_count, len(adjacency)):
        raise RuntimeError(f"Unexpected geodesic distance shape {distances.shape}")
    if not np.isfinite(distances).all():
        raise RuntimeError("Canonical palm graph is disconnected for surface basis")

    edge_lengths = np.asarray(graph.data, dtype=np.float64)
    minimum_bandwidth = float(np.median(edge_lengths))
    banks: dict[int, np.ndarray] = {}
    bandwidths: dict[int, float] = {}
    for count in anchor_counts:
        count = int(count)
        current = distances[:count]
        nearest = current.min(axis=0)
        target_distance_median = float("nan")
        coverage_radius = float("nan")
        if bandwidth_policy == "edge_floor":
            bandwidth = max(
                minimum_bandwidth,
                float(np.percentile(nearest, 90.0)) * float(bandwidth_scale),
            )
        else:
            support_rank = min(int(target_support_count), count) - 1
            support_distances = np.partition(
                current, support_rank, axis=0
            )[support_rank]
            target_distance_median = float(np.median(support_distances))
            coverage_radius = float(nearest.max()) * (1.0 + 1e-7)
            support_radius = max(
                coverage_radius,
                target_distance_median * float(bandwidth_scale),
            )
            bandwidth = max(
                support_radius / float(support_sigma),
                np.finfo(np.float64).eps,
            )
        normalized = current / bandwidth
        weights = np.exp(-0.5 * np.square(normalized))
        weights[normalized > float(support_sigma)] = 0.0
        weights = weights.T
        row_sum = weights.sum(axis=1, keepdims=True)
        missing = row_sum[:, 0] <= 1e-12
        natural_support_count = (~missing).astype(np.int64)
        if missing.any():
            closest = current[:, missing].argmin(axis=0)
            weights[missing] = 0.0
            weights[np.flatnonzero(missing), closest] = 1.0
            row_sum = weights.sum(axis=1, keepdims=True)
        banks[count] = (weights / row_sum).astype(np.float32)
        bandwidths[count] = bandwidth
        if diagnostics is not None:
            support_counts = (weights > 0.0).sum(axis=1)
            diagnostics[count] = {
                "anchor_count": count,
                "bandwidth": bandwidth,
                "bandwidth_policy": bandwidth_policy,
                "target_support_count": int(target_support_count),
                "target_support_distance_median": target_distance_median,
                "coverage_radius": coverage_radius,
                "minimum_edge_bandwidth": minimum_bandwidth,
                "fallback_vertex_count": int(missing.sum()),
                "natural_support_fraction": float(natural_support_count.mean()),
                "support_count_min": int(support_counts.min()),
                "support_count_median": float(np.median(support_counts)),
                "support_count_p90": float(np.percentile(support_counts, 90.0)),
                "partition_unity_max_error": float(
                    np.abs(banks[count].sum(axis=1) - 1.0).max()
                ),
            }
    return banks, bandwidths


def _basis_reconstruction_oracle(
    targets: np.ndarray,
    banks: Mapping[int, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    ridge: float,
) -> list[dict[str, Any]]:
    """Fit nested basis coefficients with a deterministic ridge projection."""

    target_array = np.asarray(targets, dtype=np.float32)
    if target_array.ndim != 2 or not len(target_array):
        return []
    rows: list[dict[str, Any]] = []
    cumulative: list[np.ndarray] = []
    for anchor_count in sorted(banks):
        cumulative.append(np.asarray(banks[anchor_count], dtype=np.float32))
        basis = np.concatenate(cumulative, axis=1)
        print(
            f"[canonical-basis] fitting samples={len(target_array)} "
            f"finest_anchors={anchor_count} coefficients={basis.shape[1]}",
            flush=True,
        )
        basis_tensor = torch.from_numpy(basis).to(device=device, dtype=torch.float32)
        gram = basis_tensor.T @ basis_tensor
        diagonal_mean = gram.diagonal().mean().clamp_min(1e-12)
        regularizer = float(ridge) * diagonal_mean
        gram.diagonal().add_(regularizer)
        chol, info = torch.linalg.cholesky_ex(gram)
        if int(info.max().item()) != 0:
            raise RuntimeError(
                f"Surface-basis ridge system is not positive definite at {anchor_count} anchors"
            )
        accumulators = {
            "all": MetricAccumulator(
                f"gt_multiscale_geodesic_ridge_{anchor_count}_all"
            ),
            "location_eligible": MetricAccumulator(
                f"gt_multiscale_geodesic_ridge_{anchor_count}_location_eligible"
            ),
        }
        coefficient_negative = 0
        coefficient_count = 0
        raw_below_zero = 0
        raw_above_one = 0
        raw_count = 0
        with torch.inference_mode():
            for start in range(0, len(target_array), int(batch_size)):
                target = torch.from_numpy(
                    target_array[start : start + int(batch_size)]
                ).to(device=device, dtype=torch.float32)
                right_hand_side = basis_tensor.T @ target.T
                coefficients = torch.cholesky_solve(right_hand_side, chol).T
                raw = coefficients @ basis_tensor.T
                prediction = raw.clamp(0.0, 1.0)
                prediction_np = prediction.cpu().numpy()
                target_np = target.cpu().numpy()
                accumulators["all"].update(prediction_np, target_np)
                eligible = target_np.sum(axis=1) >= 1.0
                if eligible.any():
                    accumulators["location_eligible"].update(
                        prediction_np[eligible], target_np[eligible]
                    )
                coefficient_negative += int((coefficients < 0.0).sum().item())
                coefficient_count += coefficients.numel()
                raw_below_zero += int((raw < 0.0).sum().item())
                raw_above_one += int((raw > 1.0).sum().item())
                raw_count += raw.numel()
        for population, accumulator in accumulators.items():
            row = accumulator.summary()
            row.update(
                {
                    "population": (
                        "all_basis_sampled_frames"
                        if population == "all"
                        else "basis_sampled_gt_volume_ge_1"
                    ),
                    "anchor_count_finest": int(anchor_count),
                    "scale_count": len(cumulative),
                    "coefficient_dimension": int(basis.shape[1]),
                    "coefficient_negative_fraction": (
                        coefficient_negative / coefficient_count
                        if coefficient_count
                        else float("nan")
                    ),
                    "raw_below_zero_fraction": (
                        raw_below_zero / raw_count if raw_count else float("nan")
                    ),
                    "raw_above_one_fraction": (
                        raw_above_one / raw_count if raw_count else float("nan")
                    ),
                    "ridge_relative_to_mean_gram_diagonal": float(ridge),
                    "ridge_absolute": float(regularizer.item()),
                }
            )
            rows.append(row)
        del basis_tensor, gram, chol
        print(
            f"[canonical-basis] completed finest_anchors={anchor_count}",
            flush=True,
        )
    return rows


def _active_components(
    values: np.ndarray,
    adjacency: Sequence[np.ndarray],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = np.asarray(values) >= float(threshold)
    remaining = active.copy()
    sizes: list[int] = []
    masses: list[float] = []
    for start in np.flatnonzero(active):
        start = int(start)
        if not remaining[start]:
            continue
        remaining[start] = False
        queue: deque[int] = deque((start,))
        size = 0
        mass = 0.0
        while queue:
            vertex = queue.popleft()
            size += 1
            mass += float(values[vertex])
            for neighbor in adjacency[vertex]:
                neighbor = int(neighbor)
                if remaining[neighbor]:
                    remaining[neighbor] = False
                    queue.append(neighbor)
        sizes.append(size)
        masses.append(mass)
    order = np.argsort(np.asarray(masses))[::-1]
    return np.asarray(sizes, dtype=np.int32)[order], np.asarray(masses)[order]


def _scale_distribution(
    values: np.ndarray, requested_mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    requested_mass = np.asarray(requested_mass, dtype=np.float64).reshape(-1)
    source_mass = values.sum(axis=1)
    scale = np.divide(
        requested_mass,
        source_mass,
        out=np.zeros_like(requested_mass),
        where=source_mass > 1e-12,
    )
    raw = values * scale[:, None]
    clipped = np.clip(raw, 0.0, 1.0)
    clipped_fraction = (raw > 1.0).mean(axis=1)
    return clipped.astype(np.float32), clipped_fraction


def _pooled_descriptor(grid: np.ndarray, projection: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim != 4:
        raise ValueError(f"Expected z_rgb [B,C,H,W], got {grid.shape}")
    batch, channels, height, width = grid.shape
    output_height = min(4, height)
    output_width = min(3, width)
    if height % output_height or width % output_width:
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            torch.from_numpy(grid), (output_height, output_width)
        ).numpy()
    else:
        pooled = grid.reshape(
            batch,
            channels,
            output_height,
            height // output_height,
            output_width,
            width // output_width,
        ).mean(axis=(3, 5))
    descriptor = pooled.reshape(batch, -1) @ projection
    norm = np.linalg.norm(descriptor, axis=1, keepdims=True)
    return (descriptor / np.maximum(norm, 1e-12)).astype(np.float32)


def _rank_summary(values: np.ndarray, representation: str) -> dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        return {"representation": representation, "sample_count": len(matrix)}
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    ratios = eigenvalues / total if total > 0.0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(ratios)

    def components_for(fraction: float) -> int:
        return (
            int(np.searchsorted(cumulative, fraction, side="left") + 1) if total else 0
        )

    positive = ratios[ratios > 0.0]
    effective_rank = (
        float(np.exp(-(positive * np.log(positive)).sum())) if positive.size else 0.0
    )
    return {
        "representation": representation,
        "sample_count": len(matrix),
        "dimension": matrix.shape[1],
        "components_90pct": components_for(0.90),
        "components_95pct": components_for(0.95),
        "components_99pct": components_for(0.99),
        "effective_rank": effective_rank,
        "first_component_fraction": float(ratios[0]) if ratios.size else 0.0,
        "first_8_fraction": float(ratios[:8].sum()),
        "first_16_fraction": float(ratios[:16].sum()),
        "first_32_fraction": float(ratios[:32].sum()),
    }


def _coactivation_summary(values: np.ndarray) -> dict[str, Any]:
    active = np.asarray(values, dtype=np.float64)
    prevalence = active.mean(axis=0)
    variable = (prevalence > 0.0) & (prevalence < 1.0)
    correlations: np.ndarray
    if variable.sum() >= 2:
        correlations = np.corrcoef(active[:, variable], rowvar=False)
        correlations = np.abs(correlations[np.triu_indices_from(correlations, k=1)])
        correlations = correlations[np.isfinite(correlations)]
    else:
        correlations = np.zeros(0, dtype=np.float64)
    return {
        "sample_count": len(active),
        "patch_count": active.shape[1] if active.ndim == 2 else 0,
        "active_patches_mean": float(active.sum(axis=1).mean()) if active.size else 0.0,
        "patch_prevalence_median": (
            float(np.median(prevalence)) if prevalence.size else 0.0
        ),
        "patch_prevalence_p90": (
            float(np.percentile(prevalence, 90)) if prevalence.size else 0.0
        ),
        "absolute_pair_correlation_median": (
            float(np.median(correlations)) if correlations.size else float("nan")
        ),
        "absolute_pair_correlation_p90": (
            float(np.percentile(correlations, 90))
            if correlations.size
            else float("nan")
        ),
        "absolute_pair_correlation_p99": (
            float(np.percentile(correlations, 99))
            if correlations.size
            else float("nan")
        ),
    }


def _distribution_viou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    intersection = np.minimum(left, right).sum(axis=1)
    union = np.maximum(left, right).sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.ones_like(intersection, dtype=np.float64),
        where=union > 1e-12,
    )


def _random_mass_matched_indices(
    masses: np.ndarray,
    sequence_ids: np.ndarray,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.argsort(masses)
    sorted_masses = masses[order]
    result = np.full(len(masses), -1, dtype=np.int64)
    for index, mass in enumerate(masses):
        tolerance = max(
            float(absolute_tolerance), float(relative_tolerance) * max(mass, 1.0)
        )
        left = int(np.searchsorted(sorted_masses, mass - tolerance, side="left"))
        right = int(np.searchsorted(sorted_masses, mass + tolerance, side="right"))
        candidates = order[left:right]
        candidates = candidates[
            (candidates != index) & (sequence_ids[candidates] != sequence_ids[index])
        ]
        if candidates.size:
            result[index] = int(candidates[int(rng.integers(0, len(candidates)))])
    return result


def _ambiguity_audit(
    descriptors: np.ndarray,
    pressure_distributions: np.ndarray,
    masses: np.ndarray,
    sequence_keys: Sequence[str],
    sample_ids: Sequence[str],
    *,
    device: torch.device,
    block_size: int,
    seed: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    high_similarity: float,
    low_pressure_viou: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    descriptors = np.asarray(descriptors, dtype=np.float32)
    pressure_distributions = np.asarray(pressure_distributions, dtype=np.float32)
    masses = np.asarray(masses, dtype=np.float32)
    unique_sequences = {
        value: index for index, value in enumerate(sorted(set(sequence_keys)))
    }
    sequence_ids = np.asarray(
        [unique_sequences[value] for value in sequence_keys], dtype=np.int64
    )
    random_indices = _random_mass_matched_indices(
        masses,
        sequence_ids,
        seed,
        relative_tolerance,
        absolute_tolerance,
    )
    descriptor_tensor = torch.from_numpy(descriptors).to(device)
    mass_tensor = torch.from_numpy(masses).to(device)
    sequence_tensor = torch.from_numpy(sequence_ids).to(device)
    nearest_indices = np.full(len(descriptors), -1, dtype=np.int64)
    nearest_similarity = np.full(len(descriptors), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(descriptors), int(block_size)):
            stop = min(start + int(block_size), len(descriptors))
            similarity = descriptor_tensor[start:stop] @ descriptor_tensor.T
            query_mass = mass_tensor[start:stop, None]
            tolerance = torch.maximum(
                torch.full_like(query_mass, float(absolute_tolerance)),
                float(relative_tolerance)
                * torch.maximum(query_mass, torch.ones_like(query_mass)),
            )
            eligible = (mass_tensor[None] - query_mass).abs() <= tolerance
            eligible &= sequence_tensor[None] != sequence_tensor[start:stop, None]
            similarity = similarity.masked_fill(~eligible, -torch.inf)
            values, indices = similarity.max(dim=1)
            finite = torch.isfinite(values).cpu().numpy()
            block_indices = nearest_indices[start:stop]
            block_similarity = nearest_similarity[start:stop]
            block_indices[finite] = indices.cpu().numpy()[finite]
            block_similarity[finite] = values.float().cpu().numpy()[finite]
    rows: list[dict[str, Any]] = []
    nearest_viou_values: list[float] = []
    random_viou_values: list[float] = []
    for index, nearest in enumerate(nearest_indices):
        if nearest < 0:
            continue
        nearest_viou = float(
            _distribution_viou(
                pressure_distributions[index : index + 1],
                pressure_distributions[nearest : nearest + 1],
            )[0]
        )
        random_index = int(random_indices[index])
        random_viou = (
            float(
                _distribution_viou(
                    pressure_distributions[index : index + 1],
                    pressure_distributions[random_index : random_index + 1],
                )[0]
            )
            if random_index >= 0
            else float("nan")
        )
        nearest_viou_values.append(nearest_viou)
        random_viou_values.append(random_viou)
        rows.append(
            {
                "sample_id": sample_ids[index],
                "neighbor_sample_id": sample_ids[nearest],
                "random_sample_id": (
                    sample_ids[random_index] if random_index >= 0 else ""
                ),
                "gt_volume": float(masses[index]),
                "neighbor_gt_volume": float(masses[nearest]),
                "visual_cosine": float(nearest_similarity[index]),
                "pressure_distribution_viou": nearest_viou,
                "random_pressure_distribution_viou": random_viou,
                "high_similarity_low_pressure_match": bool(
                    nearest_similarity[index] >= float(high_similarity)
                    and nearest_viou <= float(low_pressure_viou)
                ),
            }
        )
    high_ambiguity = [row for row in rows if row["high_similarity_low_pressure_match"]]
    summary = {
        "candidate_count": len(descriptors),
        "matched_count": len(rows),
        "different_sequence_constraint": True,
        "mass_relative_tolerance": relative_tolerance,
        "mass_absolute_tolerance": absolute_tolerance,
        "nearest_visual_cosine_mean": _finite_mean(
            row["visual_cosine"] for row in rows
        ),
        "nearest_pressure_distribution_viou_mean": _finite_mean(nearest_viou_values),
        "random_pressure_distribution_viou_mean": _finite_mean(random_viou_values),
        "nearest_minus_random_pressure_viou": (
            _finite_mean(nearest_viou_values) - _finite_mean(random_viou_values)
        ),
        "high_similarity_threshold": high_similarity,
        "low_pressure_viou_threshold": low_pressure_viou,
        "high_similarity_low_pressure_match_fraction": (
            len(high_ambiguity) / len(rows) if rows else float("nan")
        ),
    }
    return rows, summary


def _select_indices(sample_count: int, limit: int) -> np.ndarray:
    count = min(int(sample_count), int(limit))
    if count <= 0:
        raise ValueError("sample limit must be positive")
    if count == sample_count:
        return np.arange(sample_count, dtype=np.int64)
    return np.unique(np.linspace(0, sample_count - 1, num=count, dtype=np.int64))


def _stable_order_key(seed: int, *values: Any) -> bytes:
    payload = "\x1f".join((str(int(seed)), *(str(value) for value in values)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _sampling_identity(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    dataset = str(item.get("dataset", "") or "unknown")
    sequence = str(item.get("sequence_key", "") or item.get("sample_id", ""))
    parts = tuple(value for value in sequence.split("/") if value)
    scene = parts[0] if parts else "unknown"
    task = parts[1] if len(parts) > 1 else "unknown"
    return dataset, scene, task, sequence


def _balanced_sample_records(
    records: Sequence[Mapping[str, Any]],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministically balance samples over scene/task and sequence.

    Source caches are commonly sorted by scene and task. A prefix limit is
    therefore biased even when the parent audit uses evenly spaced indices.
    This sampler first interleaves sequences within each scene/task stratum,
    then interleaves strata, so every available group is represented before a
    second sample is taken from a well-populated group.
    """

    count = min(int(limit), len(records))
    if count <= 0:
        return []
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for raw in records:
        record = dict(raw)
        dataset, scene, task, sequence = _sampling_identity(record)
        record.update(
            {
                "dataset": dataset,
                "scene": scene,
                "task": task,
                "sequence_key": sequence,
            }
        )
        grouped[(dataset, scene, task)][sequence].append(record)

    stratum_queues: dict[tuple[str, str, str], deque[dict[str, Any]]] = {}
    for stratum, sequences in grouped.items():
        sequence_queues: dict[str, deque[dict[str, Any]]] = {}
        for sequence, values in sequences.items():
            ordered = sorted(
                values,
                key=lambda value: _stable_order_key(
                    seed,
                    sequence,
                    value.get("sample_id", ""),
                    value.get("cache_index", -1),
                ),
            )
            sequence_queues[sequence] = deque(ordered)
        sequence_order = sorted(
            sequence_queues,
            key=lambda value: _stable_order_key(seed, stratum, value),
        )
        interleaved: deque[dict[str, Any]] = deque()
        while any(sequence_queues[value] for value in sequence_order):
            for sequence in sequence_order:
                queue = sequence_queues[sequence]
                if queue:
                    interleaved.append(queue.popleft())
        stratum_queues[stratum] = interleaved

    stratum_order = sorted(
        stratum_queues,
        key=lambda value: _stable_order_key(seed, value),
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(
        stratum_queues[value] for value in stratum_order
    ):
        for stratum in stratum_order:
            queue = stratum_queues[stratum]
            if queue:
                selected.append(queue.popleft())
                if len(selected) == count:
                    break
    return selected


def _cache_sampling_records(
    cache_root: Path,
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    metadata_cache = CacheGroup(cache_root, ())
    try:
        records = []
        for raw_index in indices:
            cache_index = int(raw_index)
            item = metadata_cache[cache_index]
            records.append(
                {
                    "cache_index": cache_index,
                    "sample_id": str(item.get("sample_id", cache_index)),
                    "dataset": str(item.get("dataset", "")),
                    "sequence_key": str(item.get("sequence_key", "")),
                    "query_alias": str(item.get("query_alias", "")),
                    "frame_idx": int(item.get("frame_idx", -1)),
                }
            )
            if len(records) == len(indices) or len(records) % 10000 == 0:
                print(
                    f"[canonical-sampling] scanned metadata "
                    f"{len(records)}/{len(indices)}",
                    flush=True,
                )
        return records
    finally:
        metadata_cache.close()


def _sampling_summary_rows(
    candidates: Sequence[Mapping[str, Any]],
    selected_by_audit: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    candidate_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for item in candidates:
        dataset, scene, task, _ = _sampling_identity(item)
        candidate_counts[(dataset, scene, task)] += 1
    rows: list[dict[str, Any]] = []
    for audit_name, selected in selected_by_audit.items():
        selected_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        selected_sequences: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for item in selected:
            dataset, scene, task, sequence = _sampling_identity(item)
            key = (dataset, scene, task)
            selected_counts[key] += 1
            selected_sequences[key].add(sequence)
        for key in sorted(candidate_counts):
            rows.append(
                {
                    "audit": str(audit_name),
                    "dataset": key[0],
                    "scene": key[1],
                    "task": key[2],
                    "candidate_count": candidate_counts[key],
                    "selected_count": selected_counts.get(key, 0),
                    "selected_sequence_count": len(selected_sequences.get(key, ())),
                }
            )
    return rows


def _validate_checkpoint_provenance(cache: CacheGroup, checkpoint_path: Path) -> str:
    checkpoint_sha256 = sha256_file(checkpoint_path)
    for partition_index, provenance in enumerate(cache.provenances):
        expected = str(provenance.get("base_checkpoint_sha256", "") or "")
        if expected and expected != checkpoint_sha256:
            raise RuntimeError(
                "Feature cache checkpoint mismatch: "
                f"partition={partition_index}, cache={expected}, requested={checkpoint_sha256}"
            )
    return checkpoint_sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-limit", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--component-sample-limit", type=int, default=12000)
    parser.add_argument("--ambiguity-sample-limit", type=int, default=12000)
    parser.add_argument("--basis-sample-limit", type=int, default=2048)
    parser.add_argument("--basis-batch-size", type=int, default=64)
    parser.add_argument("--ambiguity-block-size", type=int, default=512)
    parser.add_argument("--descriptor-dim", type=int, default=128)
    parser.add_argument("--patch-counts", type=_parse_int_list, default=(32, 128, 512))
    parser.add_argument(
        "--basis-anchor-counts",
        type=_parse_int_list,
        default=(32, 128, 512, 1024),
    )
    parser.add_argument("--basis-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--basis-support-sigma", type=float, default=3.0)
    parser.add_argument("--basis-ridge", type=float, default=1e-3)
    parser.add_argument("--rank-patch-count", type=int, default=128)
    parser.add_argument(
        "--component-thresholds", type=_parse_float_list, default=(0.05, 0.10, 0.20)
    )
    parser.add_argument(
        "--component-top-k", type=_parse_int_list, default=(1, 2, 4, 8, 16)
    )
    parser.add_argument("--ambiguity-relative-mass-tolerance", type=float, default=0.10)
    parser.add_argument("--ambiguity-absolute-mass-tolerance", type=float, default=0.50)
    parser.add_argument("--ambiguity-high-similarity", type=float, default=0.95)
    parser.add_argument("--ambiguity-low-pressure-viou", type=float, default=0.30)
    parser.add_argument("--mesh", default=str(DEFAULT_MESH))
    parser.add_argument("--palm-faces", default=str(DEFAULT_PALM_FACES))
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--skip-ambiguity", action="store_true")
    parser.add_argument("--skip-basis-oracle", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    for name in ("feature_cache", "base_checkpoint", "output_dir"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} is required")
    for name in (
        "sample_limit",
        "batch_size",
        "component_sample_limit",
        "ambiguity_sample_limit",
        "basis_sample_limit",
        "basis_batch_size",
        "ambiguity_block_size",
        "descriptor_dim",
        "rank_patch_count",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.rank_patch_count not in args.patch_counts:
        raise ValueError("--rank-patch-count must be included in --patch-counts")
    if args.basis_bandwidth_scale <= 0.0:
        raise ValueError("--basis-bandwidth-scale must be positive")
    if args.basis_support_sigma <= 0.0:
        raise ValueError("--basis-support-sigma must be positive")
    if args.basis_ridge <= 0.0:
        raise ValueError("--basis-ridge must be positive")
    if not 0.0 <= args.ambiguity_relative_mass_tolerance <= 1.0:
        raise ValueError("ambiguity relative mass tolerance must lie in [0,1]")
    if args.ambiguity_absolute_mass_tolerance < 0.0:
        raise ValueError("ambiguity absolute mass tolerance must be nonnegative")


def _self_test() -> None:
    adjacency = (
        np.asarray([1], dtype=np.int32),
        np.asarray([0, 2], dtype=np.int32),
        np.asarray([1, 3], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
    )
    coordinates = np.arange(4, dtype=np.float32)[:, None]
    coordinates = np.pad(coordinates, ((0, 0), (0, 2)))
    anchors = _geodesic_fps(coordinates, adjacency, 2)
    owner = _multi_source_owner(adjacency, anchors)
    partition = PatchPartition(owner)
    values = np.asarray([[0.0, 0.2, 0.8, 1.0]], dtype=np.float32)
    reconstructed = partition.reconstruct_means(values)
    if not np.allclose(reconstructed.sum(axis=1), values.sum(axis=1)):
        raise AssertionError("Patch mean reconstruction must preserve total mass")
    sizes, masses = _active_components(values[0], adjacency, 0.1)
    if sizes.tolist() != [3] or not np.allclose(masses, [2.0]):
        raise AssertionError("Connected-component audit is inconsistent")
    scaled, _ = _scale_distribution(values, np.asarray([1.0]))
    if not np.isclose(scaled.sum(), 1.0):
        raise AssertionError("Distribution scaling failed")
    basis_banks, bandwidths = _surface_basis_banks(
        coordinates,
        adjacency,
        _geodesic_fps(coordinates, adjacency, 4),
        (2, 4),
        bandwidth_scale=1.0,
        support_sigma=3.0,
    )
    if set(basis_banks) != {2, 4} or any(value <= 0.0 for value in bandwidths.values()):
        raise AssertionError("Surface basis construction failed")
    for basis in basis_banks.values():
        if not np.allclose(basis.sum(axis=1), 1.0, atol=1e-6):
            raise AssertionError(
                "Each surface basis scale must form a partition of unity"
            )
    records = [
        {
            "cache_index": index,
            "sample_id": f"sample-{index}",
            "dataset": "d",
            "sequence_key": f"scene-{index % 2}/task-{index % 3}/seq-{index % 4}",
        }
        for index in range(24)
    ]
    selected = _balanced_sample_records(records, 12, 521)
    if len(selected) != 12 or len({row["scene"] for row in selected}) != 2:
        raise AssertionError("Balanced subaudit sampling failed")
    if selected != _balanced_sample_records(records, 12, 521):
        raise AssertionError("Balanced subaudit sampling must be deterministic")
    print("canonical localization self-test: OK")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.self_test:
        _self_test()
        return
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.feature_cache).expanduser().resolve(strict=True)
    checkpoint_path = Path(args.base_checkpoint).expanduser().resolve(strict=True)
    available = _available_fields(cache_root)
    required = {"z_rgb", "tactile_signal"}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"Feature cache lacks required fields: {missing}")
    fields = tuple(sorted(required | ({"has_tactile", "palm_mask"} & available)))
    cache = CacheGroup(cache_root, fields)
    try:
        checkpoint_sha256 = _validate_checkpoint_provenance(cache, checkpoint_path)
        first = cache[0]
        tactile_dim = int(np.asarray(first["tactile_signal"]).size)
        grid_size = tuple(int(value) for value in np.asarray(first["z_rgb"]).shape[-2:])
        valid_mask = _palm_mask(first, cache.provenance)
        coordinates, valid_indices, adjacency = _load_mesh_and_palm_graph(
            Path(args.mesh).expanduser().resolve(strict=True),
            Path(args.palm_faces).expanduser().resolve(strict=True),
            valid_mask,
        )
        graph_components = _connected_component_count(adjacency)
        if graph_components > min(args.patch_counts):
            raise RuntimeError(
                f"Valid palm graph has {graph_components} components, but the smallest "
                f"patch bank has only {min(args.patch_counts)} anchors"
            )
        maximum_anchor_count = max(
            max(args.patch_counts),
            0 if args.skip_basis_oracle else max(args.basis_anchor_counts),
        )
        if maximum_anchor_count > len(valid_indices):
            raise ValueError(
                f"Requested {maximum_anchor_count} anchors for only "
                f"{len(valid_indices)} valid palm vertices"
            )
        local_coordinates = coordinates[valid_indices]
        anchor_prefix = _geodesic_fps(
            local_coordinates, adjacency, maximum_anchor_count
        )
        partitions: dict[int, PatchPartition] = {}
        partition_payload: dict[str, np.ndarray] = {
            "valid_vertex_indices": valid_indices.astype(np.int64),
        }
        for patch_count in args.patch_counts:
            anchors = anchor_prefix[:patch_count]
            owner = _multi_source_owner(adjacency, anchors)
            partitions[int(patch_count)] = PatchPartition(owner)
            partition_payload[f"anchor_vertex_indices_{patch_count}"] = valid_indices[
                anchors
            ]
            partition_payload[f"owner_valid_{patch_count}"] = owner
        np.savez_compressed(
            output_dir / "canonical_patch_partitions.npz", **partition_payload
        )

        basis_banks: dict[int, np.ndarray] = {}
        basis_bandwidths: dict[int, float] = {}
        if not args.skip_basis_oracle:
            basis_banks, basis_bandwidths = _surface_basis_banks(
                local_coordinates,
                adjacency,
                anchor_prefix,
                args.basis_anchor_counts,
                bandwidth_scale=args.basis_bandwidth_scale,
                support_sigma=args.basis_support_sigma,
            )
            basis_payload: dict[str, np.ndarray] = {
                "valid_vertex_indices": valid_indices.astype(np.int64),
            }
            for anchor_count in args.basis_anchor_counts:
                basis_payload[f"anchor_vertex_indices_{anchor_count}"] = valid_indices[
                    anchor_prefix[:anchor_count]
                ]
                basis_payload[f"bandwidth_{anchor_count}"] = np.asarray(
                    basis_bandwidths[int(anchor_count)], dtype=np.float64
                )
            np.savez_compressed(
                output_dir / "canonical_surface_basis.npz", **basis_payload
            )

        decoder, model_config = _load_decoder(
            checkpoint_path,
            grid_size=grid_size,
            tactile_dim=tactile_dim,
            device=device,
        )
        indices = _select_indices(len(cache), args.sample_limit)
        sampling_records = _cache_sampling_records(cache_root, indices)
        component_selection = _balanced_sample_records(
            sampling_records, args.component_sample_limit, args.seed + 101
        )
        ambiguity_selection = (
            []
            if args.skip_ambiguity
            else _balanced_sample_records(
                sampling_records, args.ambiguity_sample_limit, args.seed + 211
            )
        )
        basis_selection = (
            []
            if args.skip_basis_oracle
            else _balanced_sample_records(
                sampling_records, args.basis_sample_limit, args.seed + 307
            )
        )
        selected_by_audit = {
            "components": component_selection,
            "ambiguity": ambiguity_selection,
            "surface_basis": basis_selection,
        }
        sampling_rows = _sampling_summary_rows(sampling_records, selected_by_audit)
        selected_sample_rows = [
            {"audit": audit_name, **dict(item)}
            for audit_name, selected in selected_by_audit.items()
            for item in selected
        ]
        component_indices = {int(item["cache_index"]) for item in component_selection}
        ambiguity_indices = {int(item["cache_index"]) for item in ambiguity_selection}
        basis_indices = {int(item["cache_index"]) for item in basis_selection}
        accumulators = {
            "base_all": MetricAccumulator("base_all"),
            "base_location_eligible": MetricAccumulator("base_location_eligible"),
            "base_distribution_gt_mass": MetricAccumulator("base_distribution_gt_mass"),
            "gt_distribution_base_mass": MetricAccumulator("gt_distribution_base_mass"),
        }
        patch_accumulators = {
            (patch_count, population): MetricAccumulator(
                f"gt_patch_mean_{patch_count}_{population}"
            )
            for patch_count in args.patch_counts
            for population in ("all", "location_eligible")
        }
        basis_base_accumulators = {
            "all": MetricAccumulator("base_basis_sampled_all"),
            "location_eligible": MetricAccumulator(
                "base_basis_sampled_location_eligible"
            ),
        }
        clipping = {"base_distribution_gt_mass": [], "gt_distribution_base_mass": []}
        component_rows: list[dict[str, Any]] = []
        rank_distribution_parts: list[np.ndarray] = []
        rank_binary_parts: list[np.ndarray] = []
        ambiguity_descriptors: list[np.ndarray] = []
        ambiguity_distributions: list[np.ndarray] = []
        ambiguity_masses: list[np.ndarray] = []
        ambiguity_sequence_keys: list[str] = []
        ambiguity_sample_ids: list[str] = []
        basis_target_parts: list[np.ndarray] = []
        basis_sample_ids: list[str] = []
        rng = np.random.default_rng(args.seed)
        projection: np.ndarray | None = None
        processed = 0
        for start in range(0, len(indices), args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            items = [cache[int(index)] for index in batch_indices]
            eligible_pairs = [
                (int(index), item)
                for index, item in zip(batch_indices, items)
                if "has_tactile" not in item
                or bool(np.asarray(item["has_tactile"]).reshape(-1)[0])
            ]
            eligible_items = [item for _, item in eligible_pairs]
            if not eligible_items:
                continue
            for item in eligible_items:
                if not np.array_equal(_palm_mask(item, cache.provenance), valid_mask):
                    raise RuntimeError(
                        f"Sample {item.get('sample_id')} has a non-canonical palm mask"
                    )
            target_full = np.stack(
                [
                    np.asarray(item["tactile_signal"], dtype=np.float32)
                    for item in eligible_items
                ]
            )
            target = np.nan_to_num(
                target_full[:, valid_indices], nan=0.0, posinf=1.0, neginf=0.0
            ).clip(0.0, 1.0)
            logits_full = _base_logits_batch(eligible_items, decoder, device)
            if not np.isfinite(logits_full).all():
                raise FloatingPointError("Frozen decoder produced non-finite logits")
            base = (
                1.0
                / (1.0 + np.exp(-np.clip(logits_full[:, valid_indices], -40.0, 40.0)))
            ).astype(np.float32)
            base_mass = base.sum(axis=1)
            gt_mass = target.sum(axis=1)
            base_gt_mass, base_clip = _scale_distribution(base, gt_mass)
            gt_base_mass, gt_clip = _scale_distribution(target, base_mass)
            location_eligible = gt_mass >= 1.0
            accumulators["base_all"].update(base, target)
            if location_eligible.any():
                accumulators["base_location_eligible"].update(
                    base[location_eligible], target[location_eligible]
                )
                accumulators["base_distribution_gt_mass"].update(
                    base_gt_mass[location_eligible], target[location_eligible]
                )
                accumulators["gt_distribution_base_mass"].update(
                    gt_base_mass[location_eligible], target[location_eligible]
                )
                clipping["base_distribution_gt_mass"].extend(
                    base_clip[location_eligible].tolist()
                )
                clipping["gt_distribution_base_mass"].extend(
                    gt_clip[location_eligible].tolist()
                )

            for patch_count, partition in partitions.items():
                reconstruction = partition.reconstruct_means(target).astype(np.float32)
                patch_accumulators[(patch_count, "all")].update(reconstruction, target)
                if location_eligible.any():
                    patch_accumulators[(patch_count, "location_eligible")].update(
                        reconstruction[location_eligible], target[location_eligible]
                    )

            rank_partition = partitions[args.rank_patch_count]
            patch_mass = rank_partition.sums(target).astype(np.float32)
            normalized_patch_mass = np.divide(
                patch_mass,
                patch_mass.sum(axis=1, keepdims=True),
                out=np.zeros_like(patch_mass),
                where=patch_mass.sum(axis=1, keepdims=True) > 1e-12,
            )
            rank_eligible = gt_mass >= 1.0
            if rank_eligible.any():
                rank_distribution_parts.append(normalized_patch_mass[rank_eligible])
                rank_binary_parts.append(
                    rank_partition.active(target[rank_eligible], 0.10).astype(
                        np.float32
                    )
                )

            component_local_indices = [
                local_index
                for local_index, (cache_index, _) in enumerate(eligible_pairs)
                if cache_index in component_indices
            ]
            for local_index in component_local_indices:
                values = target[local_index]
                item = eligible_items[local_index]
                dataset, scene, task, _ = _sampling_identity(item)
                for threshold in args.component_thresholds:
                    sizes, masses = _active_components(values, adjacency, threshold)
                    total_active_size = int(sizes.sum())
                    total_active_mass = float(masses.sum())
                    row: dict[str, Any] = {
                        "sample_id": str(item.get("sample_id", "")),
                        "dataset": dataset,
                        "scene": scene,
                        "task": task,
                        "sequence_key": str(item.get("sequence_key", "")),
                        "threshold": float(threshold),
                        "component_count": len(sizes),
                        "active_vertex_count": total_active_size,
                        "active_mass": total_active_mass,
                    }
                    for top_k in args.component_top_k:
                        row[f"top_{top_k}_mass_coverage"] = (
                            float(masses[:top_k].sum() / total_active_mass)
                            if total_active_mass > 0.0
                            else 1.0
                        )
                        row[f"top_{top_k}_vertex_coverage"] = (
                            float(sizes[:top_k].sum() / total_active_size)
                            if total_active_size > 0
                            else 1.0
                        )
                    component_rows.append(row)

            ambiguity_local_indices = [
                local_index
                for local_index, (cache_index, _) in enumerate(eligible_pairs)
                if cache_index in ambiguity_indices
            ]
            if ambiguity_local_indices and not args.skip_ambiguity:
                grids = np.stack(
                    [
                        np.asarray(
                            eligible_items[local_index]["z_rgb"], dtype=np.float32
                        )
                        for local_index in ambiguity_local_indices
                    ]
                )
                pooled_dimension = (
                    grids.shape[1] * min(4, grids.shape[2]) * min(3, grids.shape[3])
                )
                if projection is None:
                    projection = rng.standard_normal(
                        (pooled_dimension, args.descriptor_dim), dtype=np.float32
                    ) / math.sqrt(args.descriptor_dim)
                ambiguity_descriptors.append(
                    _pooled_descriptor(grids, projection).astype(np.float32)
                )
                ambiguity_distributions.append(
                    normalized_patch_mass[ambiguity_local_indices].astype(np.float32)
                )
                ambiguity_masses.append(
                    gt_mass[ambiguity_local_indices].astype(np.float32)
                )
                ambiguity_sequence_keys.extend(
                    str(eligible_items[local_index].get("sequence_key", ""))
                    for local_index in ambiguity_local_indices
                )
                ambiguity_sample_ids.extend(
                    str(eligible_items[local_index].get("sample_id", ""))
                    for local_index in ambiguity_local_indices
                )
            basis_local_indices = [
                local_index
                for local_index, (cache_index, _) in enumerate(eligible_pairs)
                if cache_index in basis_indices
            ]
            if basis_local_indices:
                basis_target_parts.append(
                    target[basis_local_indices].astype(np.float32)
                )
                selected_base = base[basis_local_indices]
                selected_target = target[basis_local_indices]
                basis_base_accumulators["all"].update(selected_base, selected_target)
                basis_eligible = location_eligible[basis_local_indices]
                if basis_eligible.any():
                    basis_base_accumulators["location_eligible"].update(
                        selected_base[basis_eligible], selected_target[basis_eligible]
                    )
                basis_sample_ids.extend(
                    str(eligible_items[local_index].get("sample_id", ""))
                    for local_index in basis_local_indices
                )
            processed += len(target)
            if (
                processed == len(indices)
                or processed % max(5000, args.batch_size) < args.batch_size
            ):
                print(
                    f"[canonical-localization] processed {processed}/{len(indices)}",
                    flush=True,
                )

        if processed == 0:
            raise RuntimeError("No tactile samples were eligible for the audit")
        mass_rows = [accumulator.summary() for accumulator in accumulators.values()]
        for row in mass_rows:
            values = clipping.get(str(row["variant"]))
            row["vertex_clipping_fraction_mean"] = (
                _finite_mean(values) if values is not None else 0.0
            )
            row["population"] = (
                "all_sampled_frames"
                if row["variant"] == "base_all"
                else "gt_volume_ge_1"
            )
        patch_rows = []
        for patch_count in args.patch_counts:
            for population in ("all", "location_eligible"):
                row = patch_accumulators[(patch_count, population)].summary()
                row["patch_count"] = patch_count
                row["population"] = (
                    "all_sampled_frames" if population == "all" else "gt_volume_ge_1"
                )
                patch_rows.append(row)

        component_summary_rows: list[dict[str, Any]] = []
        for threshold in args.component_thresholds:
            selected = [row for row in component_rows if row["threshold"] == threshold]
            active = [row for row in selected if row["active_vertex_count"] > 0]
            summary_row: dict[str, Any] = {
                "threshold": threshold,
                "sample_count": len(selected),
                "active_sample_count": len(active),
                "active_sample_fraction": (
                    len(active) / len(selected) if selected else float("nan")
                ),
                "component_count_mean_active": _finite_mean(
                    row["component_count"] for row in active
                ),
                "component_count_median_active": _finite_percentile(
                    (row["component_count"] for row in active), 50
                ),
                "component_count_p90_active": _finite_percentile(
                    (row["component_count"] for row in active), 90
                ),
            }
            for top_k in args.component_top_k:
                summary_row[f"representable_with_{top_k}_components_fraction"] = (
                    _finite_mean(row["component_count"] <= top_k for row in active)
                )
                summary_row[f"top_{top_k}_mass_coverage_mean"] = _finite_mean(
                    row[f"top_{top_k}_mass_coverage"] for row in active
                )
                summary_row[f"top_{top_k}_vertex_coverage_mean"] = _finite_mean(
                    row[f"top_{top_k}_vertex_coverage"] for row in active
                )
            component_summary_rows.append(summary_row)

        rank_distribution = (
            np.concatenate(rank_distribution_parts, axis=0)
            if rank_distribution_parts
            else np.zeros((0, args.rank_patch_count), dtype=np.float32)
        )
        rank_binary = (
            np.concatenate(rank_binary_parts, axis=0)
            if rank_binary_parts
            else np.zeros((0, args.rank_patch_count), dtype=np.float32)
        )
        rank_rows = [
            _rank_summary(rank_distribution, "normalized_patch_mass"),
            _rank_summary(rank_binary, "binary_patch_contact"),
        ]
        coactivation = _coactivation_summary(rank_binary)

        ambiguity_rows: list[dict[str, Any]] = []
        ambiguity_summary: dict[str, Any] = {"skipped": bool(args.skip_ambiguity)}
        if ambiguity_descriptors and not args.skip_ambiguity:
            descriptors = np.concatenate(ambiguity_descriptors, axis=0)
            distributions = np.concatenate(ambiguity_distributions, axis=0)
            masses = np.concatenate(ambiguity_masses, axis=0)
            eligible = masses >= 1.0
            ambiguity_rows, ambiguity_summary = _ambiguity_audit(
                descriptors[eligible],
                distributions[eligible],
                masses[eligible],
                np.asarray(ambiguity_sequence_keys)[eligible].tolist(),
                np.asarray(ambiguity_sample_ids)[eligible].tolist(),
                device=device,
                block_size=args.ambiguity_block_size,
                seed=args.seed,
                relative_tolerance=args.ambiguity_relative_mass_tolerance,
                absolute_tolerance=args.ambiguity_absolute_mass_tolerance,
                high_similarity=args.ambiguity_high_similarity,
                low_pressure_viou=args.ambiguity_low_pressure_viou,
            )

        basis_targets = (
            np.concatenate(basis_target_parts, axis=0)
            if basis_target_parts
            else np.zeros((0, len(valid_indices)), dtype=np.float32)
        )
        basis_rows: list[dict[str, Any]] = []
        if not args.skip_basis_oracle:
            for population, accumulator in basis_base_accumulators.items():
                row = accumulator.summary()
                row["population"] = (
                    "all_basis_sampled_frames"
                    if population == "all"
                    else "basis_sampled_gt_volume_ge_1"
                )
                row.update(
                    {
                        "anchor_count_finest": 0,
                        "scale_count": 0,
                        "coefficient_dimension": 0,
                    }
                )
                basis_rows.append(row)
            basis_rows.extend(
                _basis_reconstruction_oracle(
                    basis_targets,
                    basis_banks,
                    device=device,
                    batch_size=args.basis_batch_size,
                    ridge=args.basis_ridge,
                )
            )

        _write_csv(output_dir / "mass_distribution.csv", mass_rows)
        _write_csv(output_dir / "patch_reconstruction.csv", patch_rows)
        _write_csv(output_dir / "surface_basis_reconstruction.csv", basis_rows)
        _write_csv(output_dir / "component_per_sample.csv", component_rows)
        _write_csv(output_dir / "component_summary.csv", component_summary_rows)
        _write_csv(output_dir / "label_rank.csv", rank_rows)
        _write_csv(output_dir / "ambiguity_pairs.csv", ambiguity_rows)
        _write_csv(output_dir / "subaudit_sampling.csv", sampling_rows)
        _write_csv(output_dir / "subaudit_samples.csv", selected_sample_rows)
        run_config = {
            **vars(args),
            "feature_cache": str(cache_root),
            "base_checkpoint": str(checkpoint_path),
            "base_checkpoint_sha256": checkpoint_sha256,
            "cache_config_sha256s": list(cache.config_sha256s),
            "model_config": model_config,
            "tactile_dim": tactile_dim,
            "valid_vertex_count": len(valid_indices),
            "grid_size": list(grid_size),
            "palm_graph_component_count": graph_components,
            "mesh_sha256": sha256_file(
                Path(args.mesh).expanduser().resolve(strict=True)
            ),
            "palm_faces_sha256": sha256_file(
                Path(args.palm_faces).expanduser().resolve(strict=True)
            ),
            "patch_partition_method": "unweighted_mesh_hop_geodesic_fps_v1",
            "surface_basis_method": (
                "nested_weighted_geodesic_rbf_partition_of_unity_v1"
                if not args.skip_basis_oracle
                else "skipped"
            ),
            "surface_basis_bandwidths": {
                str(key): value for key, value in basis_bandwidths.items()
            },
            "canonical_patch_artifact_sha256": sha256_file(
                output_dir / "canonical_patch_partitions.npz"
            ),
            "canonical_surface_basis_artifact_sha256": (
                sha256_file(output_dir / "canonical_surface_basis.npz")
                if not args.skip_basis_oracle
                else ""
            ),
            "subaudit_samples_sha256": sha256_file(output_dir / "subaudit_samples.csv"),
            "subaudit_sampling_method": (
                "deterministic_scene_task_sequence_balanced_round_robin_v1"
            ),
            "component_selected_sample_count": len(
                {row["sample_id"] for row in component_rows}
            ),
            "ambiguity_selected_sample_count": len(ambiguity_sample_ids),
            "basis_selected_sample_count": len(basis_sample_ids),
            "processed_sample_count": processed,
        }
        summary = {
            "schema_version": 2,
            "purpose": (
                "sensor-independent canonical localization diagnosis; "
                "oracle results are not deployable"
            ),
            "run_config": run_config,
            "mass_distribution": mass_rows,
            "patch_reconstruction": patch_rows,
            "surface_basis_reconstruction": basis_rows,
            "contact_components": component_summary_rows,
            "label_rank": rank_rows,
            "patch_coactivation": coactivation,
            "matched_mass_visual_ambiguity": ambiguity_summary,
        }
        _write_json(output_dir / "run_config.json", run_config)
        _write_json(output_dir / "summary.json", summary)
        _write_json(
            output_dir / "AUDIT_DONE.json",
            {
                "schema_version": 2,
                "processed_sample_count": processed,
                "basis_sample_count": len(basis_targets),
                "summary": "summary.json",
            },
        )
        print(f"Canonical localization audit complete: {output_dir}", flush=True)
    finally:
        cache.close()


if __name__ == "__main__":
    main()
