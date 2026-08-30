#!/usr/bin/env python3
"""Clean up canonical surface-basis parameterizations before decoder training.

This Stage 0.3 audit reuses the exact Stage 0.2 balanced sample population. It
does not run DINO or train a model. The audit compares physical-geodesic RBF
families, output links, and a nonnegative control while keeping the canonical
surface independent of tactile sensor layout and count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.audit_canonical_localization import (
    DEFAULT_MESH,
    DEFAULT_PALM_FACES,
    MetricAccumulator,
    _available_fields,
    _geodesic_fps,
    _load_mesh_and_palm_graph,
    _parse_int_list,
    _surface_basis_banks,
    _write_csv,
    _write_json,
)
from hamer_tactile_ft.audit_local_controllability import CacheGroup
from tactile_input_priors.feature_cache import sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _resolve_path(
    explicit: str | None,
    inherited: str | None,
    *,
    name: str,
) -> Path:
    raw = explicit or inherited
    if not raw:
        raise ValueError(f"No {name} was provided or recorded by Stage 0.2")
    return Path(raw).expanduser().resolve(strict=True)


def _load_basis_population(
    cache_root: Path,
    sample_csv: Path,
    valid_indices: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], tuple[str, ...]]:
    records = [
        row for row in _read_csv(sample_csv) if row.get("audit") == "surface_basis"
    ]
    if not records:
        raise RuntimeError(f"No surface_basis records in {sample_csv}")
    records.sort(key=lambda row: int(row["cache_index"]))
    available = _available_fields(cache_root)
    if "tactile_signal" not in available:
        raise RuntimeError(f"Feature cache lacks tactile_signal: {cache_root}")
    fields = tuple(
        sorted({"tactile_signal"} | ({"has_tactile"} if "has_tactile" in available else set()))
    )
    cache = CacheGroup(cache_root, fields)
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    try:
        for row in records:
            cache_index = int(row["cache_index"])
            item = cache[cache_index]
            if "has_tactile" in item and not bool(
                np.asarray(item["has_tactile"]).reshape(-1)[0]
            ):
                raise RuntimeError(
                    f"Stage 0.2 selected non-tactile cache sample {cache_index}"
                )
            actual_id = str(item.get("sample_id", row.get("sample_id", "")))
            expected_id = str(row.get("sample_id", ""))
            if expected_id and actual_id and actual_id != expected_id:
                raise RuntimeError(
                    "Stage 0.2 sample/cache mismatch: "
                    f"index={cache_index}, expected={expected_id!r}, actual={actual_id!r}"
                )
            full = np.asarray(item["tactile_signal"], dtype=np.float32).reshape(-1)
            if int(valid_indices.max()) >= len(full):
                raise ValueError(
                    f"Cache target has {len(full)} vertices but basis needs "
                    f"index {int(valid_indices.max())}"
                )
            target = np.nan_to_num(
                full[valid_indices], nan=0.0, posinf=1.0, neginf=0.0
            ).clip(0.0, 1.0)
            targets.append(target)
            sample_ids.append(expected_id or actual_id)
    finally:
        provenances = tuple(cache.config_sha256s)
        cache.close()
    return np.stack(targets), records, provenances


def _hierarchical_detail_basis(
    banks: Mapping[int, np.ndarray],
    counts: Sequence[int],
    anchor_prefix: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Build local difference-of-RBF details with zero vertex-sum."""

    ordered = tuple(int(value) for value in counts)
    if not ordered or any(value not in banks for value in ordered):
        raise ValueError("Hierarchical counts must all have a constructed bank")
    parts = [np.asarray(banks[ordered[0]], dtype=np.float32)]
    diagnostics: list[dict[str, Any]] = [
        {
            "scale": ordered[0],
            "role": "coarse",
            "input_dimension": ordered[0],
            "output_dimension": ordered[0],
            "zero_sum_max_error": float("nan"),
        }
    ]
    previous_count = ordered[0]
    for count in ordered[1:]:
        fine = np.asarray(banks[count], dtype=np.float64)
        parent_bank = np.asarray(banks[previous_count], dtype=np.float64)
        parent_weights_at_anchors = parent_bank[anchor_prefix[:count]]
        parent_indices = parent_weights_at_anchors.argmax(axis=1)
        fine_mass = fine.sum(axis=0)
        parent_mass = parent_bank.sum(axis=0).clip(min=1e-12)
        parent_columns = parent_bank[:, parent_indices]
        details = fine - parent_columns * (
            fine_mass / parent_mass[parent_indices]
        )[None]
        # One scale-wise dependency is expected after the zero-mass conversion.
        details = details[:, :-1]
        parts.append(details.astype(np.float32))
        diagnostics.append(
            {
                "scale": count,
                "role": "signed_detail",
                "parent_scale": previous_count,
                "input_dimension": count,
                "output_dimension": count - 1,
                "zero_sum_max_error": float(np.abs(details.sum(axis=0)).max()),
            }
        )
        previous_count = count
    return np.concatenate(parts, axis=1), diagnostics


def _matrix_diagnostics(
    basis: torch.Tensor,
    gram: torch.Tensor,
    ridge: float,
) -> dict[str, Any]:
    eigenvalues = torch.linalg.eigvalsh(gram.float()).detach().cpu().numpy()
    maximum = float(max(eigenvalues[-1], 0.0))
    tolerance = max(maximum * 1e-6, 1e-10)
    positive = eigenvalues[eigenvalues > tolerance]
    numerical_rank = int(len(positive))
    diagonal_mean = float(gram.diagonal().mean().item())
    regularizer = float(ridge) * max(diagonal_mean, 1e-12)
    minimum_positive = float(positive[0]) if len(positive) else float("nan")
    return {
        "vertex_count": int(basis.shape[0]),
        "coefficient_dimension": int(basis.shape[1]),
        "numerical_rank": numerical_rank,
        "rank_deficiency": int(basis.shape[1] - numerical_rank),
        "positive_spectrum_condition": (
            maximum / minimum_positive if minimum_positive > 0.0 else float("inf")
        ),
        "ridge_absolute": regularizer,
        "ridge_relative_to_mean_gram_diagonal": float(ridge),
        "regularized_condition_upper": (
            (maximum + regularizer) / regularizer
            if regularizer > 0.0
            else float("inf")
        ),
        "basis_abs_max": float(basis.abs().max().item()),
        "basis_negative_fraction": float((basis < 0.0).float().mean().item()),
        "coefficient_fraction_of_valid_vertices": float(
            basis.shape[1] / max(int(basis.shape[0]), 1)
        ),
        "basis_storage_mib_float16": float(
            basis.numel() * 2 / (1024.0 * 1024.0)
        ),
        "basis_storage_mib_float32": float(
            basis.numel() * 4 / (1024.0 * 1024.0)
        ),
        "decode_macs_per_sample": int(basis.shape[0] * basis.shape[1]),
        "relative_decode_cost_vs_1536": float(basis.shape[1] / 1536.0),
    }


def _bootstrap_intervals(
    accumulator: MetricAccumulator,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    if repeats <= 0:
        return {}
    arrays = {
        "rmse_frame_macro": accumulator.frame_rmse,
        "contact_iou_010_frame_macro": accumulator.contact_iou,
        "volumetric_iou_frame_macro": accumulator.viou,
        "distribution_viou_frame_macro": accumulator.distribution_viou,
        "core_distribution_viou_frame_macro": accumulator.core_distribution_viou,
        "false_high_excess_mean": accumulator.false_high_excess,
    }
    output: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    for name, values in arrays.items():
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if not len(array):
            continue
        indices = rng.integers(0, len(array), size=(int(repeats), len(array)))
        estimates = array[indices].mean(axis=1)
        output[f"{name}_ci95_low"] = float(np.percentile(estimates, 2.5))
        output[f"{name}_ci95_high"] = float(np.percentile(estimates, 97.5))
    return output


def _prediction_rows(
    targets: np.ndarray,
    basis: torch.Tensor,
    coefficients_for_batch,
    *,
    variant: str,
    family: str,
    output_link: str,
    solver: str,
    device: torch.device,
    batch_size: int,
    bootstrap_repeats: int,
    seed: int,
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    accumulators = {
        "all": MetricAccumulator(variant),
        "location_eligible": MetricAccumulator(variant),
    }
    coefficient_negative = 0
    coefficient_count = 0
    coefficient_square_sum = 0.0
    prelink_negative = 0
    prelink_above_one = 0
    prelink_count = 0
    negative_volume: list[float] = []
    above_one_volume: list[float] = []
    output_near_zero = 0
    output_near_one = 0
    output_count = 0
    for start in range(0, len(targets), int(batch_size)):
        target = torch.from_numpy(targets[start : start + int(batch_size)]).to(
            device=device, dtype=torch.float32
        )
        coefficients = coefficients_for_batch(target)
        if not torch.isfinite(coefficients).all():
            raise FloatingPointError(f"Non-finite coefficients for {variant}")
        raw = coefficients @ basis.T
        if output_link == "hard_clamp":
            prediction = raw.clamp(0.0, 1.0)
        elif output_link == "sigmoid":
            prediction = torch.sigmoid(raw)
        elif output_link == "nonnegative_clip":
            prediction = raw.clamp_max(1.0)
        else:
            raise ValueError(f"Unsupported output link {output_link!r}")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Non-finite prediction for {variant}")
        prediction_np = prediction.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        accumulators["all"].update(prediction_np, target_np)
        eligible = target_np.sum(axis=1) >= 1.0
        if eligible.any():
            accumulators["location_eligible"].update(
                prediction_np[eligible], target_np[eligible]
            )
        coefficient_negative += int((coefficients < 0.0).sum().item())
        coefficient_count += coefficients.numel()
        coefficient_square_sum += float(coefficients.square().sum().item())
        prelink_negative += int((raw < 0.0).sum().item())
        prelink_above_one += int((raw > 1.0).sum().item())
        prelink_count += raw.numel()
        negative_volume.extend((-raw).clamp_min(0.0).sum(dim=1).cpu().tolist())
        above_one_volume.extend((raw - 1.0).clamp_min(0.0).sum(dim=1).cpu().tolist())
        output_near_zero += int((prediction <= 1e-3).sum().item())
        output_near_one += int((prediction >= 1.0 - 1e-3).sum().item())
        output_count += prediction.numel()
    shared = {
        **dict(common),
        "variant": variant,
        "basis_family": family,
        "output_link": output_link,
        "solver": solver,
        "coefficient_negative_fraction": (
            coefficient_negative / coefficient_count if coefficient_count else float("nan")
        ),
        "coefficient_rms": (
            math.sqrt(coefficient_square_sum / coefficient_count)
            if coefficient_count
            else float("nan")
        ),
        "prelink_negative_fraction": (
            prelink_negative / prelink_count if prelink_count else float("nan")
        ),
        "prelink_above_one_fraction": (
            prelink_above_one / prelink_count if prelink_count else float("nan")
        ),
        "prelink_negative_volume_mean": float(np.mean(negative_volume)),
        "prelink_above_one_volume_mean": float(np.mean(above_one_volume)),
        "output_near_zero_fraction": output_near_zero / max(output_count, 1),
        "output_near_one_fraction": output_near_one / max(output_count, 1),
    }
    rows: list[dict[str, Any]] = []
    for population, accumulator in accumulators.items():
        row = accumulator.summary()
        row.update(shared)
        row["population"] = (
            "all_basis_sampled_frames"
            if population == "all"
            else "basis_sampled_gt_volume_ge_1"
        )
        row.update(
            _bootstrap_intervals(
                accumulator,
                bootstrap_repeats,
                _stable_seed(f"{variant}:{population}", seed),
            )
        )
        rows.append(row)
    return rows


def _ridge_rows(
    targets: np.ndarray,
    basis_array: np.ndarray,
    *,
    family: str,
    anchor_count_finest: int,
    scale_count: int,
    device: torch.device,
    batch_size: int,
    ridge: float,
    logit_epsilon: float,
    bootstrap_repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], torch.Tensor, torch.Tensor]:
    basis = torch.from_numpy(np.asarray(basis_array, dtype=np.float32)).to(device)
    gram = basis.T @ basis
    matrix = _matrix_diagnostics(basis, gram, ridge)
    regularized = gram.clone()
    regularized.diagonal().add_(float(matrix["ridge_absolute"]))
    chol, info = torch.linalg.cholesky_ex(regularized)
    if int(info.max().item()) != 0:
        raise RuntimeError(f"Ridge matrix is not positive definite for {family}")
    common = {
        **matrix,
        "source": "stage0_3_cleanup",
        "anchor_count_finest": int(anchor_count_finest),
        "scale_count": int(scale_count),
    }

    def solve_pressure(target: torch.Tensor) -> torch.Tensor:
        rhs = basis.T @ target.T
        return torch.cholesky_solve(rhs, chol).T

    def solve_logits(target: torch.Tensor) -> torch.Tensor:
        bounded = target.clamp(float(logit_epsilon), 1.0 - float(logit_epsilon))
        logits = torch.logit(bounded)
        rhs = basis.T @ logits.T
        return torch.cholesky_solve(rhs, chol).T

    rows = _prediction_rows(
        targets,
        basis,
        solve_pressure,
        variant=f"{family}_pressure_ridge_clamp",
        family=family,
        output_link="hard_clamp",
        solver="ridge_pressure",
        device=device,
        batch_size=batch_size,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
        common=common,
    )
    rows.extend(
        _prediction_rows(
            targets,
            basis,
            solve_logits,
            variant=f"{family}_logit_ridge_sigmoid",
            family=family,
            output_link="sigmoid",
            solver="ridge_logit",
            device=device,
            batch_size=batch_size,
            bootstrap_repeats=bootstrap_repeats,
            seed=seed,
            common=common,
        )
    )
    return rows, {"basis_family": family, **matrix}, basis, gram


def _nnls_rows(
    targets: np.ndarray,
    basis: torch.Tensor,
    gram: torch.Tensor,
    matrix: Mapping[str, Any],
    *,
    family: str,
    anchor_count_finest: int,
    device: torch.device,
    batch_size: int,
    iterations: int,
    bootstrap_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    diagonal = gram.diagonal().clamp_min(1e-8)

    def solve(target: torch.Tensor) -> torch.Tensor:
        rhs = target @ basis
        coefficients = (rhs / diagonal[None]).clamp_min(1e-8)
        for _ in range(int(iterations)):
            denominator = (coefficients @ gram).clamp_min(1e-8)
            coefficients = coefficients * (rhs / denominator)
        return coefficients

    return _prediction_rows(
        targets,
        basis,
        solve,
        variant=f"{family}_pressure_nnls_clip",
        family=family,
        output_link="nonnegative_clip",
        solver=f"multiplicative_nnls_{iterations}",
        device=device,
        batch_size=batch_size,
        bootstrap_repeats=bootstrap_repeats,
        seed=seed,
        common={
            **dict(matrix),
            "source": "stage0_3_cleanup",
            "anchor_count_finest": int(anchor_count_finest),
            "scale_count": 1,
        },
    )


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    maximize = (
        "contact_iou_010_frame_macro",
        "volumetric_iou_frame_macro",
        "core_distribution_viou_frame_macro",
    )
    minimize = ("rmse_vertex_micro", "false_high_excess_mean")
    weakly_better = all(float(left[key]) >= float(right[key]) for key in maximize)
    weakly_better &= all(float(left[key]) <= float(right[key]) for key in minimize)
    strictly_better = any(float(left[key]) > float(right[key]) for key in maximize)
    strictly_better |= any(float(left[key]) < float(right[key]) for key in minimize)
    return bool(weakly_better and strictly_better)


def _tradeoff_rows(
    reconstruction_rows: Sequence[Mapping[str, Any]],
    base_row: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = [
        dict(row)
        for row in reconstruction_rows
        if row.get("population") == "all_basis_sampled_frames"
        and int(row.get("coefficient_dimension", 0)) > 0
    ]
    base_contact = float(base_row["contact_iou_010_frame_macro"])
    base_viou = float(base_row["volumetric_iou_frame_macro"])
    base_core = float(base_row["core_distribution_viou_frame_macro"])
    base_distribution = float(base_row["distribution_viou_frame_macro"])
    base_rmse = float(base_row["rmse_vertex_micro"])
    base_false_high = float(base_row["false_high_excess_mean"])
    base_peak = float(base_row["gt_ge_070_mean_prediction"])
    output: list[dict[str, Any]] = []
    for row in candidates:
        row["delta_contact"] = float(row["contact_iou_010_frame_macro"]) - base_contact
        row["delta_viou"] = float(row["volumetric_iou_frame_macro"]) - base_viou
        row["delta_coreloc"] = (
            float(row["core_distribution_viou_frame_macro"]) - base_core
        )
        row["delta_distribution_viou"] = (
            float(row["distribution_viou_frame_macro"]) - base_distribution
        )
        row["delta_rmse_vertex_micro"] = float(row["rmse_vertex_micro"]) - base_rmse
        row["false_high_ratio"] = (
            float(row["false_high_excess_mean"]) / max(base_false_high, 1e-12)
        )
        row["peak_prediction_ratio"] = (
            float(row["gt_ge_070_mean_prediction"]) / max(base_peak, 1e-12)
        )
        row["capacity_pass"] = bool(
            row["delta_contact"] >= 0.03
            and row["delta_viou"] >= 0.03
            and row["delta_coreloc"] >= 0.03
            and row["delta_distribution_viou"] >= -0.005
            and row["delta_rmse_vertex_micro"] <= 0.0
        )
        row["safety_pass"] = bool(
            row["false_high_ratio"] <= 1.10
            and row["peak_prediction_ratio"] >= 1.0
        )
        condition = float(row.get("regularized_condition_upper", float("inf")))
        fallback = int(row.get("fallback_vertex_count_finest", 0))
        rank_deficiency = int(row.get("rank_deficiency", 1))
        row["numerical_pass"] = bool(
            math.isfinite(condition)
            and condition <= 1e5
            and fallback == 0
            and rank_deficiency == 0
        )
        row["smooth_optimization_pass"] = bool(
            row.get("output_link") == "sigmoid"
        )
        row["strict_pass"] = bool(
            row["capacity_pass"]
            and row["safety_pass"]
            and row["numerical_pass"]
            and row["smooth_optimization_pass"]
        )
        output.append(row)
    deployable = [
        row
        for row in output
        if row["numerical_pass"] and row["smooth_optimization_pass"]
    ]
    for row in output:
        row["pareto_optimal_all"] = not any(
            _dominates(other, row) for other in output if other is not row
        )
        row["pareto_optimal"] = bool(
            any(item is row for item in deployable)
            and not any(
                _dominates(other, row) for other in deployable if other is not row
            )
        )
        if row["strict_pass"] and row["pareto_optimal"]:
            row["decision_tier"] = "strict_pareto_candidate"
        elif row["capacity_pass"] and not row["safety_pass"]:
            row["decision_tier"] = "capacity_with_safety_tradeoff"
        elif row["capacity_pass"]:
            row["decision_tier"] = "capacity_only"
        else:
            row["decision_tier"] = "insufficient_capacity"
    strict = [row["variant"] for row in output if row["strict_pass"]]
    pareto = [row["variant"] for row in output if row["pareto_optimal"]]
    diagnostic_pareto = [
        row["variant"] for row in output if row["pareto_optimal_all"]
    ]
    smooth_standalone = [
        row
        for row in output
        if row["strict_pass"]
        and str(row.get("basis_family", "")).startswith(
            ("weighted_single_", "weighted_target_")
        )
        and row.get("output_link") == "sigmoid"
    ]
    near_best_tolerances = {
        "contact_iou_010_frame_macro": 0.005,
        "volumetric_iou_frame_macro": 0.005,
        "distribution_viou_frame_macro": 0.005,
        "core_distribution_viou_frame_macro": 0.010,
        "rmse_vertex_micro": 0.0005,
        "gt_ge_070_mean_prediction": 0.020,
        "false_high_relative": 0.10,
    }
    near_best: list[dict[str, Any]] = []
    if smooth_standalone:
        maxima = {
            key: max(float(row[key]) for row in smooth_standalone)
            for key in (
                "contact_iou_010_frame_macro",
                "volumetric_iou_frame_macro",
                "distribution_viou_frame_macro",
                "core_distribution_viou_frame_macro",
                "gt_ge_070_mean_prediction",
            )
        }
        best_rmse = min(float(row["rmse_vertex_micro"]) for row in smooth_standalone)
        best_false_high = min(
            float(row["false_high_excess_mean"]) for row in smooth_standalone
        )
        for row in smooth_standalone:
            is_near = all(
                float(row[key]) >= maxima[key] - near_best_tolerances[key]
                for key in (
                    "contact_iou_010_frame_macro",
                    "volumetric_iou_frame_macro",
                    "distribution_viou_frame_macro",
                    "core_distribution_viou_frame_macro",
                    "gt_ge_070_mean_prediction",
                )
            )
            is_near &= (
                float(row["rmse_vertex_micro"])
                <= best_rmse + near_best_tolerances["rmse_vertex_micro"]
            )
            is_near &= (
                float(row["false_high_excess_mean"])
                <= best_false_high
                * (1.0 + near_best_tolerances["false_high_relative"])
            )
            row["complexity_near_best"] = bool(is_near)
            if is_near:
                near_best.append(row)
    smallest_near_best = min(
        near_best,
        key=lambda row: int(row["coefficient_dimension"]),
        default=None,
    )
    decision = {
        "selection_method": "guardrails_then_pareto; no weighted scalar score",
        "primary_capacity_guardrails": {
            "delta_contact_min": 0.03,
            "delta_viou_min": 0.03,
            "delta_coreloc_min": 0.03,
            "delta_distribution_viou_min": -0.005,
            "delta_rmse_vertex_micro_max": 0.0,
        },
        "safety_guardrails": {
            "false_high_ratio_max": 1.10,
            "peak_prediction_ratio_min": 1.0,
        },
        "trainability_guardrails": {
            "regularized_condition_upper_max": 1e5,
            "fallback_vertex_count_finest": 0,
            "hard_clamp_allowed": False,
        },
        "strict_candidates": strict,
        "pareto_front": pareto,
        "diagnostic_pareto_front_including_non_deployable": diagnostic_pareto,
        "complexity_selection": {
            "policy": "smallest smooth standalone strict candidate within per-metric near-best tolerances",
            "tolerances": near_best_tolerances,
            "eligible_variants": [row["variant"] for row in smooth_standalone],
            "near_best_variants": [row["variant"] for row in near_best],
            "smallest_near_best_variant": (
                smallest_near_best["variant"] if smallest_near_best else None
            ),
            "smallest_near_best_dimension": (
                int(smallest_near_best["coefficient_dimension"])
                if smallest_near_best
                else None
            ),
        },
        "recommendation": (
            "proceed to decoder memorization with every strict Pareto candidate"
            if strict
            else "do not select a training basis yet; inspect the reported tradeoff"
        ),
    }
    return output, decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--feature-cache")
    parser.add_argument("--mesh")
    parser.add_argument("--palm-faces")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--anchor-counts", type=_parse_int_list, default=(32, 128, 512, 768, 1024, 1536)
    )
    parser.add_argument(
        "--standalone-counts", type=_parse_int_list, default=(512, 768, 1024, 1536)
    )
    parser.add_argument(
        "--hierarchical-counts", type=_parse_int_list, default=(32, 128, 512, 1024)
    )
    parser.add_argument("--nnls-counts", type=_parse_int_list, default=(1024,))
    parser.add_argument(
        "--skip-multiscale-controls",
        action="store_true",
        help="Skip the already-diagnosed hierarchical and cumulative controls.",
    )
    parser.add_argument(
        "--skip-nnls",
        action="store_true",
        help="Skip the already-diagnosed nonnegative coefficient control.",
    )
    parser.add_argument("--basis-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--basis-support-sigma", type=float, default=3.0)
    parser.add_argument(
        "--basis-bandwidth-policy",
        choices=("edge_floor", "target_overlap"),
        default="edge_floor",
        help=(
            "edge_floor reproduces the historical mesh-edge bandwidth floor; "
            "target_overlap shrinks bandwidth with anchor density"
        ),
    )
    parser.add_argument("--basis-target-support-count", type=int, default=6)
    parser.add_argument("--result-source", default="stage0_3_cleanup")
    parser.add_argument("--basis-ridge", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nnls-iterations", type=int, default=100)
    parser.add_argument("--logit-epsilon", type=float, default=1e-3)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--self-test", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    if not args.stage1_dir or not args.output_dir:
        raise ValueError("--stage1-dir and --output-dir are required")
    all_counts = set(args.anchor_counts)
    count_fields = ["standalone_counts"]
    if not args.skip_multiscale_controls:
        count_fields.append("hierarchical_counts")
    if not args.skip_nnls:
        count_fields.append("nnls_counts")
    for name in count_fields:
        missing = set(getattr(args, name)) - all_counts
        if missing:
            raise ValueError(f"--{name.replace('_', '-')} contains unbuilt anchors: {sorted(missing)}")
    if args.batch_size <= 0 or args.nnls_iterations <= 0:
        raise ValueError("batch size and NNLS iterations must be positive")
    if args.bootstrap_repeats < 0:
        raise ValueError("--bootstrap-repeats must be nonnegative")
    if args.basis_bandwidth_scale <= 0.0 or args.basis_support_sigma <= 0.0:
        raise ValueError("basis bandwidth and support must be positive")
    if args.basis_target_support_count <= 0:
        raise ValueError("--basis-target-support-count must be positive")
    if not str(args.result_source).strip():
        raise ValueError("--result-source must be nonempty")
    if args.basis_ridge <= 0.0:
        raise ValueError("--basis-ridge must be positive")
    if not 0.0 < args.logit_epsilon < 0.5:
        raise ValueError("--logit-epsilon must lie in (0,0.5)")


def _self_test() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    adjacency = (
        np.asarray([1], dtype=np.int32),
        np.asarray([0, 2], dtype=np.int32),
        np.asarray([1, 3], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
    )
    anchors = _geodesic_fps(coordinates, adjacency, 4, weighted=True)
    diagnostics: dict[int, dict[str, Any]] = {}
    banks, _ = _surface_basis_banks(
        coordinates,
        adjacency,
        anchors,
        (2, 4),
        bandwidth_scale=1.0,
        support_sigma=3.0,
        diagnostics=diagnostics,
    )
    adaptive_diagnostics: dict[int, dict[str, Any]] = {}
    adaptive_banks, _ = _surface_basis_banks(
        coordinates,
        adjacency,
        anchors,
        (4,),
        bandwidth_scale=1.0,
        support_sigma=3.0,
        bandwidth_policy="target_overlap",
        target_support_count=2,
        diagnostics=adaptive_diagnostics,
    )
    if adaptive_diagnostics[4]["fallback_vertex_count"] != 0:
        raise AssertionError("Adaptive surface bank unexpectedly required fallback")
    if not np.allclose(adaptive_banks[4].sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("Adaptive surface bank is not a partition of unity")
    hierarchical, detail = _hierarchical_detail_basis(banks, (2, 4), anchors)
    if hierarchical.shape != (4, 5) or len(detail) != 2:
        raise AssertionError("Hierarchical detail construction has the wrong shape")
    if not np.allclose(banks[4].sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("Surface bank is not a partition of unity")
    target = np.asarray(
        [[0.0, 0.2, 0.8, 1.0], [0.0, 0.0, 0.3, 0.0]], dtype=np.float32
    )
    rows, matrix, basis, gram = _ridge_rows(
        target,
        banks[4],
        family="self_test",
        anchor_count_finest=4,
        scale_count=1,
        device=torch.device("cpu"),
        batch_size=2,
        ridge=1e-3,
        logit_epsilon=1e-3,
        bootstrap_repeats=10,
        seed=521,
    )
    if len(rows) != 4 or not math.isfinite(float(matrix["regularized_condition_upper"])):
        raise AssertionError("Surface cleanup oracle self-test failed")
    nnls = _nnls_rows(
        target,
        basis,
        gram,
        matrix,
        family="self_test",
        anchor_count_finest=4,
        device=torch.device("cpu"),
        batch_size=2,
        iterations=10,
        bootstrap_repeats=10,
        seed=521,
    )
    if len(nnls) != 2 or any(
        float(row["coefficient_negative_fraction"]) != 0.0 for row in nnls
    ):
        raise AssertionError("Nonnegative basis control failed")
    base = dict(rows[0])
    base.update(
        {
            "variant": "base_basis_sampled_all",
            "coefficient_dimension": 0,
            "population": "all_basis_sampled_frames",
        }
    )
    tradeoff, decision = _tradeoff_rows(rows + nnls, base)
    if len(tradeoff) != 3 or "selection_method" not in decision:
        raise AssertionError("Guardrail/Pareto comparison failed")
    print("surface basis cleanup self-test: OK")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    if args.self_test:
        _self_test()
        return
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    stage1_dir = Path(args.stage1_dir).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_summary = _read_json(stage1_dir / "summary.json")
    stage1_config = dict(stage1_summary.get("run_config", {}))
    cache_root = _resolve_path(
        args.feature_cache, stage1_config.get("feature_cache"), name="feature cache"
    )
    mesh_path = _resolve_path(
        args.mesh, stage1_config.get("mesh", str(DEFAULT_MESH)), name="mesh"
    )
    palm_faces_path = _resolve_path(
        args.palm_faces,
        stage1_config.get("palm_faces", str(DEFAULT_PALM_FACES)),
        name="palm faces",
    )
    with np.load(stage1_dir / "canonical_surface_basis.npz") as stage1_basis:
        valid_indices = np.asarray(
            stage1_basis["valid_vertex_indices"], dtype=np.int64
        )
    tactile_dim = int(stage1_config.get("tactile_dim", int(valid_indices.max()) + 1))
    valid_mask = np.zeros(tactile_dim, dtype=bool)
    valid_mask[valid_indices] = True
    coordinates, loaded_valid, adjacency = _load_mesh_and_palm_graph(
        mesh_path, palm_faces_path, valid_mask
    )
    if not np.array_equal(valid_indices, loaded_valid):
        raise RuntimeError("Stage 0.2 valid vertices do not match the canonical mesh")
    targets, sample_rows, cache_config_sha256s = _load_basis_population(
        cache_root, stage1_dir / "subaudit_samples.csv", valid_indices
    )
    expected_samples = int(stage1_config.get("basis_selected_sample_count", len(targets)))
    if len(targets) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} Stage 0.2 basis samples, found {len(targets)}"
        )
    if max(args.anchor_counts) > len(valid_indices):
        raise ValueError(
            f"Requested {max(args.anchor_counts)} anchors for only "
            f"{len(valid_indices)} valid canonical vertices"
        )
    local_coordinates = coordinates[valid_indices]
    anchor_prefix = _geodesic_fps(
        local_coordinates, adjacency, max(args.anchor_counts), weighted=True
    )
    basis_method = (
        "weighted_geodesic_rbf_partition_of_unity_v2"
        if args.basis_bandwidth_policy == "edge_floor"
        else "weighted_geodesic_rbf_target_overlap_v1"
    )
    bank_diagnostics: dict[int, dict[str, Any]] = {}
    banks, bandwidths = _surface_basis_banks(
        local_coordinates,
        adjacency,
        anchor_prefix,
        args.anchor_counts,
        bandwidth_scale=args.basis_bandwidth_scale,
        support_sigma=args.basis_support_sigma,
        bandwidth_policy=args.basis_bandwidth_policy,
        target_support_count=args.basis_target_support_count,
        diagnostics=bank_diagnostics,
    )
    basis_payload: dict[str, np.ndarray] = {
        "valid_vertex_indices": valid_indices,
        "anchor_prefix_valid_indices": anchor_prefix,
        "anchor_prefix_vertex_indices": valid_indices[anchor_prefix],
    }
    for count in args.anchor_counts:
        basis_payload[f"bandwidth_{count}"] = np.asarray(
            bandwidths[count], dtype=np.float64
        )
    np.savez_compressed(output_dir / "canonical_surface_basis_cleanup.npz", **basis_payload)

    stage1_summary_sha256 = sha256_file(stage1_dir / "summary.json")
    stage1_samples_sha256 = sha256_file(stage1_dir / "subaudit_samples.csv")
    mesh_sha256 = sha256_file(mesh_path)
    palm_faces_sha256 = sha256_file(palm_faces_path)
    implementation_sha256 = sha256_file(Path(__file__))
    basis_implementation_sha256 = sha256_file(
        REPO_ROOT / "hamer_tactile_ft" / "audit_canonical_localization.py"
    )
    partial_dir = output_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_rows = [
        {**dict(row), "source": "stage0_2_reference", "basis_family": "legacy"}
        for row in stage1_summary.get("surface_basis_reconstruction", [])
    ]
    matrix_rows: list[dict[str, Any]] = []
    family_cache: dict[int, tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]] = {}
    for count in args.standalone_counts:
        family = (
            f"weighted_single_{count}"
            if args.basis_bandwidth_policy == "edge_floor"
            else f"weighted_target_s{args.basis_target_support_count}_{count}"
        )
        partial_contract = {
            "schema_version": 1,
            "family": family,
            "implementation_sha256": implementation_sha256,
            "basis_implementation_sha256": basis_implementation_sha256,
            "stage1_summary_sha256": stage1_summary_sha256,
            "stage1_samples_sha256": stage1_samples_sha256,
            "mesh_sha256": mesh_sha256,
            "palm_faces_sha256": palm_faces_sha256,
            "sample_count": len(targets),
            "valid_vertex_count": len(valid_indices),
            "basis_method": basis_method,
            "basis_bandwidth_scale": float(args.basis_bandwidth_scale),
            "basis_support_sigma": float(args.basis_support_sigma),
            "basis_bandwidth_policy": args.basis_bandwidth_policy,
            "basis_target_support_count": int(args.basis_target_support_count),
            "basis_ridge": float(args.basis_ridge),
            "logit_epsilon": float(args.logit_epsilon),
            "bootstrap_repeats": int(args.bootstrap_repeats),
            "seed": int(args.seed),
        }
        partial_key = hashlib.sha256(
            json.dumps(partial_contract, sort_keys=True).encode("utf-8")
        ).hexdigest()
        partial_path = partial_dir / f"{family}.json"
        cached_partial = _read_json(partial_path) if partial_path.is_file() else {}
        needs_nnls_tensors = not args.skip_nnls and count in args.nnls_counts
        if (
            cached_partial.get("contract_sha256") == partial_key
            and not needs_nnls_tensors
        ):
            print(f"[basis-cleanup] reusing completed {family}", flush=True)
            rows = [dict(row) for row in cached_partial["reconstruction_rows"]]
            matrix = dict(cached_partial["matrix_diagnostics"])
            basis = gram = None
        else:
            print(
                f"[basis-cleanup] fitting {family}: vertices={len(valid_indices)} "
                f"coefficients={count} samples={len(targets)}",
                flush=True,
            )
            rows, matrix, basis, gram = _ridge_rows(
                targets,
                banks[count],
                family=family,
                anchor_count_finest=count,
                scale_count=1,
                device=device,
                batch_size=args.batch_size,
                ridge=args.basis_ridge,
                logit_epsilon=args.logit_epsilon,
                bootstrap_repeats=args.bootstrap_repeats,
                seed=args.seed,
            )
        fallback = int(bank_diagnostics[count]["fallback_vertex_count"])
        for row in rows:
            row["fallback_vertex_count_finest"] = fallback
            row["source"] = str(args.result_source)
            row["basis_bandwidth_policy"] = args.basis_bandwidth_policy
            row["basis_target_support_count"] = int(
                args.basis_target_support_count
            )
            row["basis_bandwidth"] = float(bandwidths[count])
            row["basis_support_count_median"] = float(
                bank_diagnostics[count]["support_count_median"]
            )
            row["basis_support_count_p90"] = float(
                bank_diagnostics[count]["support_count_p90"]
            )
        matrix["fallback_vertex_count_finest"] = fallback
        matrix["basis_bandwidth_policy"] = args.basis_bandwidth_policy
        matrix["basis_target_support_count"] = int(
            args.basis_target_support_count
        )
        matrix["basis_bandwidth"] = float(bandwidths[count])
        matrix["basis_support_count_median"] = float(
            bank_diagnostics[count]["support_count_median"]
        )
        matrix["basis_support_count_p90"] = float(
            bank_diagnostics[count]["support_count_p90"]
        )
        _write_json(
            partial_path,
            {
                "schema_version": 1,
                "contract_sha256": partial_key,
                "contract": partial_contract,
                "reconstruction_rows": rows,
                "matrix_diagnostics": matrix,
            },
        )
        reconstruction_rows.extend(rows)
        matrix_rows.append(matrix)
        if needs_nnls_tensors:
            if basis is None or gram is None:
                raise AssertionError("NNLS tensors were unexpectedly discarded")
            family_cache[count] = (basis, gram, matrix, family)
        else:
            del basis, gram

    hierarchical_scales: list[dict[str, Any]] = []
    if not args.skip_multiscale_controls:
        hierarchical, hierarchical_scales = _hierarchical_detail_basis(
            banks, args.hierarchical_counts, anchor_prefix
        )
        hierarchical_family = "weighted_hierarchical_" + "_".join(
            str(value) for value in args.hierarchical_counts
        )
        rows, matrix, _, _ = _ridge_rows(
            targets,
            hierarchical,
            family=hierarchical_family,
            anchor_count_finest=max(args.hierarchical_counts),
            scale_count=len(args.hierarchical_counts),
            device=device,
            batch_size=args.batch_size,
            ridge=args.basis_ridge,
            logit_epsilon=args.logit_epsilon,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        fallback = int(
            bank_diagnostics[max(args.hierarchical_counts)]["fallback_vertex_count"]
        )
        for row in rows:
            row["fallback_vertex_count_finest"] = fallback
        matrix["fallback_vertex_count_finest"] = fallback
        reconstruction_rows.extend(rows)
        matrix_rows.append(matrix)

        cumulative = np.concatenate(
            [banks[count] for count in args.hierarchical_counts], axis=1
        )
        cumulative_family = "weighted_cumulative_" + "_".join(
            str(value) for value in args.hierarchical_counts
        )
        rows, matrix, _, _ = _ridge_rows(
            targets,
            cumulative,
            family=cumulative_family,
            anchor_count_finest=max(args.hierarchical_counts),
            scale_count=len(args.hierarchical_counts),
            device=device,
            batch_size=args.batch_size,
            ridge=args.basis_ridge,
            logit_epsilon=args.logit_epsilon,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        for row in rows:
            row["fallback_vertex_count_finest"] = fallback
        matrix["fallback_vertex_count_finest"] = fallback
        reconstruction_rows.extend(rows)
        matrix_rows.append(matrix)

    for count in (() if args.skip_nnls else args.nnls_counts):
        basis, gram, matrix, family = family_cache[count]
        rows = _nnls_rows(
            targets,
            basis,
            gram,
            matrix,
            family=family,
            anchor_count_finest=count,
            device=device,
            batch_size=args.batch_size,
            iterations=args.nnls_iterations,
            bootstrap_repeats=args.bootstrap_repeats,
            seed=args.seed,
        )
        for row in rows:
            row["fallback_vertex_count_finest"] = int(
                bank_diagnostics[count]["fallback_vertex_count"]
            )
        reconstruction_rows.extend(rows)
    del family_cache

    base_rows = [
        row
        for row in reconstruction_rows
        if row.get("variant") == "base_basis_sampled_all"
    ]
    if len(base_rows) != 1:
        raise RuntimeError("Stage 0.2 summary lacks one base_basis_sampled_all row")
    tradeoff_rows, decision = _tradeoff_rows(reconstruction_rows, base_rows[0])
    capacity_rows = [
        row
        for row in tradeoff_rows
        if str(row.get("basis_family", "")).startswith(
            ("weighted_single_", "weighted_target_")
        )
    ]
    bank_rows = [bank_diagnostics[count] for count in args.anchor_counts]
    run_config = {
        **vars(args),
        "stage1_dir": str(stage1_dir),
        "stage1_summary_sha256": stage1_summary_sha256,
        "stage1_samples_sha256": stage1_samples_sha256,
        "feature_cache": str(cache_root),
        "cache_config_sha256s": list(cache_config_sha256s),
        "mesh": str(mesh_path),
        "mesh_sha256": mesh_sha256,
        "palm_faces": str(palm_faces_path),
        "palm_faces_sha256": palm_faces_sha256,
        "implementation_sha256": implementation_sha256,
        "basis_implementation_sha256": basis_implementation_sha256,
        "sample_count": len(targets),
        "valid_vertex_count": len(valid_indices),
        "anchor_method": "physical_weighted_geodesic_fps_v1",
        "basis_method": basis_method,
        "output_artifact_sha256": sha256_file(
            output_dir / "canonical_surface_basis_cleanup.npz"
        ),
    }
    summary = {
        "schema_version": 1,
        "purpose": str(args.result_source).replace("_", " "),
        "run_config": run_config,
        "decision": decision,
        "bank_diagnostics": bank_rows,
        "hierarchical_scale_diagnostics": hierarchical_scales,
        "matrix_diagnostics": matrix_rows,
        "reconstruction": reconstruction_rows,
        "tradeoff": tradeoff_rows,
    }
    _write_csv(output_dir / "basis_bank_diagnostics.csv", bank_rows)
    _write_csv(output_dir / "basis_matrix_diagnostics.csv", matrix_rows)
    _write_csv(output_dir / "basis_cleanup_reconstruction.csv", reconstruction_rows)
    _write_csv(output_dir / "basis_tradeoff.csv", tradeoff_rows)
    _write_csv(output_dir / "basis_capacity_curve.csv", capacity_rows)
    _write_csv(output_dir / "cleanup_samples.csv", sample_rows)
    _write_json(output_dir / "basis_tradeoff.json", decision)
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "AUDIT_DONE.json",
        {
            "schema_version": 1,
            "sample_count": len(targets),
            "strict_candidate_count": len(decision["strict_candidates"]),
            "smallest_near_best_dimension": decision["complexity_selection"][
                "smallest_near_best_dimension"
            ],
            "summary": "summary.json",
        },
    )
    print(f"Surface basis cleanup audit complete: {output_dir}", flush=True)
    print(decision["recommendation"], flush=True)


if __name__ == "__main__":
    main()
