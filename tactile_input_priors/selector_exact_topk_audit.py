"""Exact-budget causal audit for selector-guided pressure correction."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ExactTopKPolicy:
    name: str
    topk: int
    alpha: float
    target_floor: float


_RAW_FIELDS = (
    "frames",
    "values",
    "abs_sum",
    "sq_sum",
    "contact_iou_sum",
    "viou_sum",
    "pred_volume",
    "gt_volume",
    "temporal_correct",
    "core_viou_sum",
    "core_count",
    "false_high_excess",
    "cat_over",
    "cat_over_denom",
    "cat_under",
    "cat_under_denom",
    "candidate_count",
    "selected_count",
    "strict_selected",
    "protected_selected",
    "strict_candidate",
    "protected_candidate",
    "correction_total",
    "strict_removed",
    "protected_removed",
    "added_under",
    "added_under_protected",
    "strict_excess_before",
    "strict_excess_after",
    "corrected_crossings",
    "harmful_crossings",
    "high_gt_pred_sum",
    "high_gt_count",
    "low_gt_pred_sum",
    "low_gt_count",
)


def exact_topk_policies(
    topk_values: Sequence[int],
    *,
    alpha: float = 1.0,
    target_floor: float = 0.02,
) -> tuple[ExactTopKPolicy, ...]:
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("Exact-top-k alpha must lie in (0,1]")
    if not 0.0 <= float(target_floor) < 1.0:
        raise ValueError("Exact-top-k target floor must lie in [0,1)")
    values = sorted({int(value) for value in topk_values})
    if not values or values[0] < 0:
        raise ValueError("Exact-top-k values must be nonnegative and non-empty")
    if 0 not in values:
        values.insert(0, 0)
    return tuple(
        ExactTopKPolicy(
            name="base" if topk == 0 else f"top{topk:g}_a{alpha:g}_f{target_floor:g}",
            topk=topk,
            alpha=0.0 if topk == 0 else float(alpha),
            target_floor=float(target_floor),
        )
        for topk in values
    )


def exact_topk_policies_from_selection(
    path: os.PathLike[str] | str,
) -> tuple[ExactTopKPolicy, ...]:
    payload = json.loads(
        Path(path).expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    if payload.get("schema") != "tactile_selector_exact_topk_selection_v1":
        raise ValueError(f"Unsupported exact-top-k selection schema: {path}")
    policies = []
    seen: set[int] = set()
    for value in payload.get("selected_policies", ()):
        topk = int(value["topk"])
        if topk in seen:
            continue
        seen.add(topk)
        policies.append(
            ExactTopKPolicy(
                name=str(value["name"]),
                topk=topk,
                alpha=float(value["alpha"]),
                target_floor=float(value["target_floor"]),
            )
        )
    if 0 not in seen:
        config = dict(payload.get("audit_config", {}))
        policies.insert(
            0,
            ExactTopKPolicy(
                "base",
                0,
                0.0,
                float(config.get("target_floor", 0.02)),
            ),
        )
    policies.sort(key=lambda value: value.topk)
    if not policies:
        raise ValueError(f"Exact-top-k selection is empty: {path}")
    return tuple(policies)


def _stable_topk_mask(
    score: torch.Tensor,
    candidate: torch.Tensor,
    topk_values: Sequence[int],
) -> torch.Tensor:
    """Return [P,B,V] masks with exact per-frame counts and stable tie breaks."""

    if score.shape != candidate.shape or score.ndim != 2:
        raise ValueError("Exact-top-k score and candidate tensors must be matching [B,V]")
    batch_size, vertex_count = score.shape
    max_topk = min(max((int(value) for value in topk_values), default=0), vertex_count)
    selected = torch.zeros(
        (len(topk_values), batch_size, vertex_count),
        dtype=torch.bool,
        device=score.device,
    )
    if max_topk == 0 or not bool(candidate.any()):
        return selected

    if not bool(torch.isfinite(score[candidate]).all()):
        raise FloatingPointError("Exact-top-k candidate scores contain NaN or Inf")
    ranked = score.float().masked_fill(~candidate, -torch.inf)
    # Canonical vertices already have a fixed ascending order. Stable sorting
    # therefore gives ties a deterministic lowest-index-first contract.
    indices = torch.argsort(
        ranked,
        dim=1,
        descending=True,
        stable=True,
    )[:, :max_topk]
    available = candidate.sum(dim=1)
    positions = torch.arange(max_topk, device=score.device)[None]
    for policy_index, raw_topk in enumerate(topk_values):
        topk = min(int(raw_topk), max_topk)
        if topk <= 0:
            continue
        take = positions[:, :topk] < available[:, None]
        selected[policy_index].scatter_(1, indices[:, :topk], take)
    return selected


def _summary_from_raw(raw: Mapping[str, float]) -> dict[str, float]:
    frames = max(float(raw["frames"]), 1.0)
    values = max(float(raw["values"]), 1.0)
    selected = max(float(raw["selected_count"]), 1.0)
    strict_removed = float(raw["strict_removed"])
    utility = strict_removed - float(raw["protected_removed"]) - float(raw["added_under"])
    return {
        "frame_count": float(raw["frames"]),
        "candidate_count": float(raw["candidate_count"]),
        "action_count": float(raw["selected_count"]),
        "action_coverage_all_valid": float(raw["selected_count"]) / values,
        "action_coverage_candidates": float(raw["selected_count"])
        / max(float(raw["candidate_count"]), 1.0),
        "strict_precision_all_selected": float(raw["strict_selected"]) / selected,
        "strict_recall": float(raw["strict_selected"])
        / max(float(raw["strict_candidate"]), 1.0),
        "protected_selection_rate": float(raw["protected_selected"])
        / max(float(raw["protected_candidate"]), 1.0),
        "correction_total": float(raw["correction_total"]),
        "strict_false_high_volume_removed": strict_removed,
        "strict_false_high_volume_removed_fraction": strict_removed
        / max(float(raw["strict_excess_before"]), 1e-12),
        "strict_false_high_excess_after_fraction": float(raw["strict_excess_after"])
        / max(float(raw["strict_excess_before"]), 1e-12),
        "protected_contact_volume_removed": float(raw["protected_removed"]),
        "added_under_volume": float(raw["added_under"]),
        "added_under_protected_volume": float(raw["added_under_protected"]),
        "protected_removed_per_strict_removed": float(raw["protected_removed"])
        / max(strict_removed, 1e-12),
        "added_under_per_strict_removed": float(raw["added_under"])
        / max(strict_removed, 1e-12),
        "balanced_net_utility": utility,
        "balanced_net_utility_fraction": utility
        / max(float(raw["strict_excess_before"]), 1e-12),
        "threshold_crossing_correction_recall": float(raw["corrected_crossings"])
        / max(float(raw["strict_candidate"]), 1.0),
        "protected_harmful_crossing_rate": float(raw["harmful_crossings"])
        / max(float(raw["protected_candidate"]), 1.0),
        "mae": float(raw["abs_sum"]) / values,
        "rmse": math.sqrt(max(float(raw["sq_sum"]) / values, 0.0)),
        "contact_iou": float(raw["contact_iou_sum"]) / frames,
        "volumetric_iou": float(raw["viou_sum"]) / frames,
        "core_distribution_viou": float(raw["core_viou_sum"])
        / max(float(raw["core_count"]), 1.0),
        "pred_gt_volume_ratio": float(raw["pred_volume"])
        / max(float(raw["gt_volume"]), 1e-12),
        "temporal_accuracy_frame": float(raw["temporal_correct"]) / frames,
        "false_high_excess_fraction": float(raw["false_high_excess"])
        / max(float(raw["pred_volume"]), 1e-12),
        "catastrophic_over_rate": float(raw["cat_over"])
        / max(float(raw["cat_over_denom"]), 1.0),
        "catastrophic_under_rate": float(raw["cat_under"])
        / max(float(raw["cat_under_denom"]), 1.0),
        "high_gt_mean_prediction": float(raw["high_gt_pred_sum"])
        / max(float(raw["high_gt_count"]), 1.0),
        "low_gt_mean_prediction": float(raw["low_gt_pred_sum"])
        / max(float(raw["low_gt_count"]), 1.0),
    }


class ExactTopKAccumulator:
    """Accumulate pressure outcomes after exact per-frame selector ranking."""

    def __init__(
        self,
        policies: Sequence[ExactTopKPolicy],
        *,
        action_threshold: float = 0.10,
        no_contact_max: float = 0.02,
        contact_min: float = 0.10,
        chunk_size: int = 4,
    ):
        if not policies or policies[0].topk != 0:
            raise ValueError("Exact-top-k policies must start with topk=0")
        self.policies = tuple(policies)
        self.action_threshold = float(action_threshold)
        self.no_contact_max = float(no_contact_max)
        self.contact_min = float(contact_min)
        self.chunk_size = max(1, int(chunk_size))
        self.values = torch.zeros((len(policies), len(_RAW_FIELDS)), dtype=torch.float64)
        self.sequence_values: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(
        self,
        base_prediction: torch.Tensor,
        score: torch.Tensor,
        target: torch.Tensor,
        palm_mask: torch.Tensor,
        has_tactile: torch.Tensor,
        *,
        sequence_keys: Sequence[str],
    ) -> torch.Tensor:
        pred = base_prediction.detach().float()
        score = score.detach().float().to(pred.device)
        target = target.detach().float().to(pred.device)
        if pred.shape != score.shape or pred.shape != target.shape or pred.ndim != 2:
            raise ValueError("Exact-top-k pressure, score, and target must match [B,V]")
        if len(sequence_keys) != pred.shape[0]:
            raise ValueError("Exact-top-k sequence_key count does not match batch size")
        palm = palm_mask.detach().to(pred.device) > 0.5
        if palm.ndim == 1:
            palm = palm[None].expand_as(pred)
        elif palm.shape[0] == 1:
            palm = palm.expand_as(pred)
        if palm.shape != pred.shape:
            raise ValueError("Exact-top-k palm mask must match pressure")
        has = has_tactile.detach().reshape(-1).to(pred.device) > 0.5
        valid = palm & has[:, None]
        if not bool(valid.any()):
            return torch.zeros(
                (len(self.policies), *pred.shape),
                dtype=torch.bool,
                device=pred.device,
            )
        candidate = valid & (pred >= self.action_threshold)
        strict = candidate & (target <= self.no_contact_max)
        protected = candidate & (target >= self.contact_min)
        base_under = (target - pred).clamp_min(0.0)
        gt_volume = (target * valid).sum(dim=1)
        selected_all = _stable_topk_mask(
            score,
            candidate,
            [policy.topk for policy in self.policies],
        )
        expected_counts = torch.stack(
            [
                torch.minimum(
                    candidate.sum(dim=1),
                    candidate.new_full((pred.shape[0],), policy.topk, dtype=torch.long),
                )
                for policy in self.policies
            ]
        )
        actual_counts = selected_all.sum(dim=2)
        if not torch.equal(actual_counts, expected_counts):
            raise RuntimeError("Exact-top-k failed to preserve per-frame action counts")

        for start in range(0, len(self.policies), self.chunk_size):
            current = self.policies[start : start + self.chunk_size]
            selected = selected_all[start : start + len(current)]
            alphas = pred.new_tensor([policy.alpha for policy in current])[:, None, None]
            floors = pred.new_tensor([policy.target_floor for policy in current])[:, None, None]
            correction = alphas * selected * (pred[None] - floors).clamp_min(0.0)
            corrected = (pred[None] - correction).clamp(0.0, 1.0)
            valid_c = valid[None]
            difference = (corrected - target[None]) * valid_c
            pred_masked = corrected * valid_c
            target_masked = target[None] * valid_c
            pred_volume = pred_masked.sum(dim=2)
            pred_contact = (corrected >= self.contact_min) & valid_c
            gt_contact = (target >= self.contact_min) & valid
            intersection = (pred_contact & gt_contact[None]).sum(dim=2).double()
            union = (pred_contact | gt_contact[None]).sum(dim=2).double()
            contact_iou = torch.where(union > 0, intersection / union.clamp_min(1.0), 1.0)
            viou_min = torch.minimum(pred_masked, target_masked).sum(dim=2)
            viou_max = torch.maximum(pred_masked, target_masked).sum(dim=2)
            viou = torch.where(viou_max > 1e-12, viou_min / viou_max.clamp_min(1e-12), 1.0)
            eligible_core = (gt_volume >= 1.0) & ((target * valid).amax(dim=1) >= 0.05)
            pred_core = pred_masked.square()
            gt_core = target_masked.square()
            pred_dist = pred_core / pred_core.sum(dim=2, keepdim=True).clamp_min(1e-12)
            gt_dist = gt_core / gt_core.sum(dim=2, keepdim=True).clamp_min(1e-12)
            core_min = torch.minimum(pred_dist, gt_dist).sum(dim=2)
            core_max = torch.maximum(pred_dist, gt_dist).sum(dim=2)
            core_viou = core_min / core_max.clamp_min(1e-12)
            false_high = valid_c & (target[None] < 0.005) & (corrected >= 0.3)
            added_under = (
                ((target[None] - corrected).clamp_min(0.0) - base_under[None])
                .clamp_min(0.0)
                * valid_c
            )
            strict_before = ((pred - target).clamp_min(0.0) * strict).sum(dim=1)

            frame_values = torch.stack(
                (
                    has.float()[None].expand(len(current), -1),
                    valid.sum(dim=1).float()[None].expand(len(current), -1),
                    difference.abs().sum(dim=2),
                    difference.square().sum(dim=2),
                    contact_iou,
                    viou,
                    pred_volume,
                    gt_volume[None].expand(len(current), -1),
                    ((pred_contact.any(dim=2)) == gt_contact.any(dim=1)[None]).float(),
                    core_viou * eligible_core[None],
                    eligible_core.float()[None].expand(len(current), -1),
                    ((corrected - target[None]).clamp_min(0.0) * false_high).sum(dim=2),
                    ((gt_volume[None] < 10.0) & (pred_volume > 300.0)).float(),
                    ((gt_volume < 10.0) & has).float()[None].expand(len(current), -1),
                    ((gt_volume[None] >= 150.0) & (pred_volume < 50.0)).float(),
                    ((gt_volume >= 150.0) & has).float()[None].expand(len(current), -1),
                    candidate.sum(dim=1).float()[None].expand(len(current), -1),
                    selected.sum(dim=2).float(),
                    (selected & strict[None]).sum(dim=2).float(),
                    (selected & protected[None]).sum(dim=2).float(),
                    strict.sum(dim=1).float()[None].expand(len(current), -1),
                    protected.sum(dim=1).float()[None].expand(len(current), -1),
                    correction.sum(dim=2),
                    (correction * strict[None]).sum(dim=2),
                    (correction * protected[None]).sum(dim=2),
                    added_under.sum(dim=2),
                    (added_under * protected[None]).sum(dim=2),
                    strict_before[None].expand(len(current), -1),
                    ((corrected - target[None]).clamp_min(0.0) * strict[None]).sum(dim=2),
                    (strict[None] & (corrected < self.action_threshold)).sum(dim=2).float(),
                    (protected[None] & (corrected < self.action_threshold)).sum(dim=2).float(),
                    (corrected * (valid & (target >= 0.70))[None]).sum(dim=2),
                    (valid & (target >= 0.70)).sum(dim=1).float()[None].expand(len(current), -1),
                    (corrected * (valid & (target < 0.005))[None]).sum(dim=2),
                    (valid & (target < 0.005)).sum(dim=1).float()[None].expand(len(current), -1),
                ),
                dim=2,
            ).double().cpu()
            expected_shape = (len(current), pred.shape[0], len(_RAW_FIELDS))
            if tuple(frame_values.shape) != expected_shape:
                raise RuntimeError(
                    f"Exact-top-k metric layout mismatch: {tuple(frame_values.shape)} "
                    f"vs {expected_shape}"
                )
            self.values[start : start + len(current)] += frame_values.sum(dim=1)
            for batch_index, raw_key in enumerate(sequence_keys):
                if not bool(has[batch_index]):
                    continue
                key = str(raw_key)
                if key not in self.sequence_values:
                    self.sequence_values[key] = torch.zeros_like(self.values)
                self.sequence_values[key][start : start + len(current)] += frame_values[
                    :, batch_index
                ]
        return selected_all

    def synchronize(self, device: torch.device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        values = self.values.to(device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        self.values.copy_(values.cpu())
        rank = dist.get_rank()
        gathered: list[dict[str, torch.Tensor] | None] | None = (
            [None] * dist.get_world_size() if rank == 0 else None
        )
        dist.gather_object(self.sequence_values, gathered, dst=0)
        if rank == 0:
            merged: dict[str, torch.Tensor] = {}
            for shard in gathered or ():
                if shard is None:
                    continue
                for key, value in shard.items():
                    merged.setdefault(key, torch.zeros_like(value)).add_(value)
            self.sequence_values = merged
        else:
            self.sequence_values = {}

    def summaries(self) -> list[dict[str, Any]]:
        rows = []
        for policy, values in zip(self.policies, self.values):
            raw = {name: float(values[index]) for index, name in enumerate(_RAW_FIELDS)}
            rows.append({**asdict(policy), **_summary_from_raw(raw)})
        base = rows[0]
        for row in rows:
            for metric in (
                "rmse",
                "contact_iou",
                "volumetric_iou",
                "core_distribution_viou",
                "temporal_accuracy_frame",
                "false_high_excess_fraction",
                "high_gt_mean_prediction",
                "low_gt_mean_prediction",
            ):
                row[f"delta_{metric}"] = float(row[metric]) - float(base[metric])
        return rows


_PAIR_METRICS = (
    ("strict_precision_all_selected", 1.0),
    ("strict_recall", 1.0),
    ("strict_false_high_volume_removed_fraction", 1.0),
    ("protected_removed_per_strict_removed", -1.0),
    ("added_under_per_strict_removed", -1.0),
    ("balanced_net_utility_fraction", 1.0),
    ("contact_iou", 1.0),
    ("volumetric_iou", 1.0),
    ("core_distribution_viou", 1.0),
    ("rmse", -1.0),
)


def _metric_from_array(raw: np.ndarray, name: str) -> np.ndarray:
    index = {field: offset for offset, field in enumerate(_RAW_FIELDS)}
    denominator = lambda field, floor: np.maximum(raw[..., index[field]], floor)
    if name == "strict_precision_all_selected":
        return raw[..., index["strict_selected"]] / denominator("selected_count", 1.0)
    if name == "strict_recall":
        return raw[..., index["strict_selected"]] / denominator("strict_candidate", 1.0)
    if name == "strict_false_high_volume_removed_fraction":
        return raw[..., index["strict_removed"]] / denominator("strict_excess_before", 1e-12)
    if name == "protected_removed_per_strict_removed":
        return raw[..., index["protected_removed"]] / denominator("strict_removed", 1e-12)
    if name == "added_under_per_strict_removed":
        return raw[..., index["added_under"]] / denominator("strict_removed", 1e-12)
    if name == "balanced_net_utility_fraction":
        utility = (
            raw[..., index["strict_removed"]]
            - raw[..., index["protected_removed"]]
            - raw[..., index["added_under"]]
        )
        return utility / denominator("strict_excess_before", 1e-12)
    if name == "contact_iou":
        return raw[..., index["contact_iou_sum"]] / denominator("frames", 1.0)
    if name == "volumetric_iou":
        return raw[..., index["viou_sum"]] / denominator("frames", 1.0)
    if name == "core_distribution_viou":
        return raw[..., index["core_viou_sum"]] / denominator("core_count", 1.0)
    if name == "rmse":
        return np.sqrt(np.maximum(raw[..., index["sq_sum"]] / denominator("values", 1.0), 0.0))
    raise KeyError(name)


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"Cannot write an empty exact-top-k table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _select_validation_profiles(
    real_rows: Sequence[Mapping[str, Any]],
    *,
    target_removal_fraction: float,
) -> dict[str, Mapping[str, Any]]:
    rows = [dict(row) for row in real_rows]
    nonzero = [row for row in rows if int(row["topk"]) > 0]
    if not nonzero:
        raise ValueError("Exact-top-k validation sweep has no nonzero policies")
    balanced = max(
        rows,
        key=lambda row: (
            float(row["balanced_net_utility_fraction"]),
            -int(row["topk"]),
        ),
    )
    nonzero_balanced = max(
        nonzero,
        key=lambda row: (
            float(row["balanced_net_utility_fraction"]),
            -int(row["topk"]),
        ),
    )
    target = min(
        nonzero,
        key=lambda row: (
            abs(
                float(row["strict_false_high_volume_removed_fraction"])
                - float(target_removal_fraction)
            ),
            -float(row["balanced_net_utility_fraction"]),
            int(row["topk"]),
        ),
    )
    return {
        "best_balanced": balanced,
        "best_nonzero_balanced": nonzero_balanced,
        "target_removed_fraction": target,
    }


def write_exact_topk_outputs(
    output_dir: Path,
    *,
    accumulators: Mapping[str, ExactTopKAccumulator],
    audit_config: Mapping[str, Any],
    selection_source: os.PathLike[str] | str | None = None,
    bootstrap_iterations: int = 2000,
    bootstrap_seed: int = 521,
    target_removal_fraction: float = 0.03,
    action_pair_stats: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Write exact-count source comparisons and sequence-clustered intervals."""

    if "real" not in accumulators or "rgb_base" not in accumulators:
        raise ValueError("Exact-top-k audit requires real and rgb_base sources")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, accumulator in accumulators.items():
        if not accumulator.sequence_values:
            continue
        sequence_total = torch.stack(
            list(accumulator.sequence_values.values()), dim=0
        ).sum(dim=0)
        if not torch.allclose(sequence_total, accumulator.values, rtol=1e-10, atol=1e-8):
            maximum_gap = float((sequence_total - accumulator.values).abs().max())
            raise RuntimeError(
                f"Exact-top-k sequence/global totals differ for {source}: "
                f"max_gap={maximum_gap}"
            )
    source_rows = {source: accumulator.summaries() for source, accumulator in accumulators.items()}
    flattened = [
        {"source": source, **row}
        for source, rows in source_rows.items()
        for row in rows
    ]
    metrics_path = output_dir / "exact_topk_source_metrics.csv"
    _write_rows(metrics_path, flattened)

    real_rows = {int(row["topk"]): row for row in source_rows["real"]}
    pair_rows: list[dict[str, Any]] = []
    real_accumulator = accumulators["real"]
    sequence_keys = sorted(real_accumulator.sequence_values)
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_counts = None
    if bootstrap_iterations > 0 and len(sequence_keys) >= 2:
        draws = rng.integers(
            0,
            len(sequence_keys),
            size=(int(bootstrap_iterations), len(sequence_keys)),
            endpoint=False,
        )
        bootstrap_counts = np.zeros(
            (int(bootstrap_iterations), len(sequence_keys)), dtype=np.float64
        )
        np.add.at(
            bootstrap_counts,
            (
                np.repeat(np.arange(int(bootstrap_iterations)), len(sequence_keys)),
                draws.reshape(-1),
            ),
            1.0,
        )
    for source, accumulator in accumulators.items():
        if source == "real":
            continue
        source_by_topk = {int(row["topk"]): row for row in source_rows[source]}
        shared_sequences = [
            key for key in sequence_keys if key in accumulator.sequence_values
        ]
        if shared_sequences != sequence_keys:
            raise RuntimeError(
                f"Exact-top-k source {source!r} does not cover the same sequences as real"
            )
        for policy_index, policy in enumerate(real_accumulator.policies):
            topk = int(policy.topk)
            real_row = real_rows[topk]
            source_row = source_by_topk[topk]
            if float(real_row["action_count"]) != float(source_row["action_count"]):
                raise RuntimeError(
                    f"Exact-top-k action mismatch for {source}, k={topk}: "
                    f"{real_row['action_count']} vs {source_row['action_count']}"
                )
            row: dict[str, Any] = {
                "source": source,
                "name": policy.name,
                "topk": topk,
                "alpha": policy.alpha,
                "target_floor": policy.target_floor,
                "exact_action_count": float(real_row["action_count"]),
                "paired_sequence_count": len(sequence_keys),
            }
            if action_pair_stats is not None and source in action_pair_stats:
                intersection, union, disagreement = (
                    float(value)
                    for value in action_pair_stats[source][policy_index]
                )
                row.update(
                    {
                        "action_jaccard": intersection / union if union > 0 else 1.0,
                        "action_disagreement_fraction": (
                            disagreement / union if union > 0 else 0.0
                        ),
                    }
                )
            if bootstrap_counts is not None:
                real_sequence = np.stack(
                    [
                        real_accumulator.sequence_values[key][policy_index].numpy()
                        for key in sequence_keys
                    ]
                )
                source_sequence = np.stack(
                    [
                        accumulator.sequence_values[key][policy_index].numpy()
                        for key in sequence_keys
                    ]
                )
                real_boot = bootstrap_counts @ real_sequence
                source_boot = bootstrap_counts @ source_sequence
            else:
                real_boot = source_boot = None
            for metric, direction in _PAIR_METRICS:
                gap = float(real_row[metric]) - float(source_row[metric])
                row[f"real_{metric}"] = float(real_row[metric])
                row[f"source_{metric}"] = float(source_row[metric])
                row[f"real_minus_source_{metric}"] = gap
                if real_boot is not None:
                    boot_gap = _metric_from_array(real_boot, metric) - _metric_from_array(
                        source_boot, metric
                    )
                    low, high = np.quantile(boot_gap, (0.025, 0.975))
                    row[f"real_minus_source_{metric}_ci95_low"] = float(low)
                    row[f"real_minus_source_{metric}_ci95_high"] = float(high)
                    row[f"real_better_probability_{metric}"] = float(
                        np.mean(float(direction) * boot_gap > 0.0)
                    )
                else:
                    row[f"real_minus_source_{metric}_ci95_low"] = None
                    row[f"real_minus_source_{metric}_ci95_high"] = None
                    row[f"real_better_probability_{metric}"] = None
            pair_rows.append(row)
    pair_path = output_dir / "exact_topk_real_vs_controls.csv"
    _write_rows(pair_path, pair_rows)

    if selection_source is None:
        profiles = _select_validation_profiles(
            source_rows["real"],
            target_removal_fraction=target_removal_fraction,
        )
        selected_topks = {0, *(int(row["topk"]) for row in profiles.values())}
        selected_policies = [
            asdict(policy)
            for policy in real_accumulator.policies
            if policy.topk in selected_topks
        ]
        selection_scope = "validation_only"
        selection_source_value = None
    else:
        source_path = Path(selection_source).expanduser().resolve(strict=True)
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        profiles = dict(source_payload.get("profiles", {}))
        selected_policies = [asdict(policy) for policy in real_accumulator.policies]
        selection_scope = "fixed_from_validation"
        selection_source_value = str(source_path)

    selection_payload = {
        "schema": "tactile_selector_exact_topk_selection_v1",
        "selection_scope": selection_scope,
        "selection_source": selection_source_value,
        "audit_config": dict(audit_config),
        "profiles": {
            name: {
                key: value
                for key, value in dict(row).items()
                if key in {
                    "name",
                    "topk",
                    "alpha",
                    "target_floor",
                    "strict_precision_all_selected",
                    "strict_recall",
                    "strict_false_high_volume_removed_fraction",
                    "protected_removed_per_strict_removed",
                    "added_under_per_strict_removed",
                    "balanced_net_utility_fraction",
                }
            }
            for name, row in profiles.items()
        },
        "selected_policies": selected_policies,
    }
    selected_topks = {int(value["topk"]) for value in selected_policies}
    selected_pair_rows = [
        row for row in pair_rows if int(row["topk"]) in selected_topks
    ]
    selection_path = output_dir / "exact_topk_selection.json"
    temporary = selection_path.with_name(f".{selection_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, selection_path)

    payload = {
        "schema": "tactile_selector_exact_topk_causal_audit_v1",
        "budget_contract": (
            "per-frame exact min(k, RGB candidate count); shared candidate pool, "
            "alpha, floor, pressure, target, and palm mask"
        ),
        "score_sources": list(accumulators),
        "audit_config": dict(audit_config),
        "selection": selection_payload,
        "selected_pairwise_results": selected_pair_rows,
        "files": {
            "source_metrics_csv": str(metrics_path),
            "real_vs_controls_csv": str(pair_path),
            "selection_json": str(selection_path),
        },
    }
    summary_lines = [
        "Exact per-frame top-k selector causal audit",
        "All sources use identical candidate counts, alpha, floor, and frozen pressure.",
        f"Sources: {', '.join(accumulators)}",
        f"Sequence bootstrap iterations: {int(bootstrap_iterations)}",
        f"Selection scope: {selection_scope}",
    ]
    for profile, value in selection_payload["profiles"].items():
        summary_lines.append(
            f"{profile}: k={value.get('topk')}, "
            f"precision={float(value.get('strict_precision_all_selected', 0.0)):.6f}, "
            f"removed={float(value.get('strict_false_high_volume_removed_fraction', 0.0)):.6f}, "
            f"utility={float(value.get('balanced_net_utility_fraction', 0.0)):.6f}"
        )
    for row in selected_pair_rows:
        if int(row["topk"]) == 0:
            continue
        summary_lines.append(
            f"k={int(row['topk'])} real-vs-{row['source']}: "
            f"precision_gap={float(row['real_minus_source_strict_precision_all_selected']):+.6f}, "
            f"utility_gap={float(row['real_minus_source_balanced_net_utility_fraction']):+.6f}, "
            f"utility_ci=[{row['real_minus_source_balanced_net_utility_fraction_ci95_low']}, "
            f"{row['real_minus_source_balanced_net_utility_fraction_ci95_high']}]"
        )
    summary_path = output_dir / "exact_topk_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    payload["files"]["summary"] = str(summary_path)
    json_path = output_dir / "exact_topk_audit.json"
    payload["files"]["audit_json"] = str(json_path)
    temporary = json_path.with_name(f".{json_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, json_path)
    return payload


def run_exact_topk_tiny_checks() -> None:
    policies = exact_topk_policies((0, 1, 2, 8), alpha=1.0, target_floor=0.02)
    pred = torch.tensor([[0.20, 0.20, 0.20, 0.05], [0.20, 0.05, 0.05, 0.05]])
    target = torch.tensor([[0.00, 0.20, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]])
    candidate = pred >= 0.10
    tied = torch.ones_like(pred)
    masks = _stable_topk_mask(tied, candidate, [policy.topk for policy in policies])
    assert masks[:, 0].sum(dim=1).tolist() == [0, 1, 2, 3]
    assert masks[:, 1].sum(dim=1).tolist() == [0, 1, 1, 1]
    assert masks[1, 0, 0] and not masks[1, 0, 1]

    real = ExactTopKAccumulator(policies, chunk_size=2)
    control = ExactTopKAccumulator(policies, chunk_size=2)
    real.update(
        pred,
        torch.tensor([[4.0, 1.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]]),
        target,
        torch.ones_like(pred),
        torch.ones(2),
        sequence_keys=("a", "b"),
    )
    control.update(
        pred,
        torch.tensor([[1.0, 4.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]]),
        target,
        torch.ones_like(pred),
        torch.ones(2),
        sequence_keys=("a", "b"),
    )
    for real_row, control_row in zip(real.summaries(), control.summaries()):
        assert real_row["action_count"] == control_row["action_count"]
    assert real.summaries()[1]["strict_precision_all_selected"] > control.summaries()[1][
        "strict_precision_all_selected"
    ]
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        payload = write_exact_topk_outputs(
            directory,
            accumulators={"real": real, "rgb_base": control},
            audit_config={"tiny": True},
            bootstrap_iterations=20,
        )
        assert Path(payload["files"]["real_vs_controls_csv"]).is_file()
