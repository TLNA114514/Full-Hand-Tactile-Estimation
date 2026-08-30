"""Causal controls and pressure-policy metrics for prior-aware selectors."""

from __future__ import annotations

import csv
import hashlib
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
from torch.utils.data import Dataset


MAPPED_CONTROLS = ("cross_sequence", "same_sequence_far", "wrong_query")


def _stable_index(material: str, length: int, seed: int) -> int:
    if length <= 0:
        raise ValueError("Cannot select from an empty collection")
    digest = hashlib.sha256(f"{int(seed)}|{material}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % int(length)


def _record(row: Mapping[str, Any], fallback_index: int) -> dict[str, Any]:
    metadata = row.get("metadata")
    merged = dict(metadata) if isinstance(metadata, Mapping) else {}
    merged.update({key: value for key, value in row.items() if key != "metadata"})
    uid = str(
        merged.get("sample_uid")
        or merged.get("sample_id")
        or merged.get("sample_ref")
        or f"index:{int(fallback_index)}"
    )
    return {
        "sample_uid": uid,
        "sequence_key": str(merged.get("sequence_key", "")),
        "query_alias": str(merged.get("query_alias", merged.get("hand", "query"))),
        "frame_idx": int(merged.get("frame_idx", 0)),
    }


def _cache_rows(cache) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    entries = sorted(cache.shards, key=lambda item: int(item["source_start"]))
    for entry in entries:
        shard_index = int(entry["shard_index"])
        path = cache.cache_dir / "shards" / f"shard-{shard_index:06d}" / "samples.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid cache metadata at {path}:{line_number}") from exc
                rows.append(_record(value, len(rows)))
    if len(rows) != len(cache):
        raise RuntimeError(
            f"Feature-cache metadata count mismatch: rows={len(rows)}, cache={len(cache)}"
        )
    return rows


def dataset_audit_records(dataset: Dataset) -> list[dict[str, Any]]:
    """Read stable metadata without decoding images or copying feature arrays."""

    current = dataset
    visited: set[int] = set()
    while hasattr(current, "dataset") and id(current) not in visited:
        visited.add(id(current))
        samples = getattr(current, "samples", None)
        if samples is not None:
            break
        current = current.dataset

    samples = getattr(current, "samples", None)
    if samples is not None:
        records = [_record(samples[index], index) for index in range(len(samples))]
    elif hasattr(dataset, "base_group"):
        group = tuple(dataset.base_group)
        rows_by_partition = [_cache_rows(cache) for cache in group]
        records = []
        for index in range(len(dataset)):
            partition = index % len(group)
            local_index = index // len(group)
            if local_index >= len(rows_by_partition[partition]):
                raise RuntimeError(
                    "Partitioned feature-cache metadata cannot reproduce dataset order"
                )
            records.append(rows_by_partition[partition][local_index])
    else:
        # This fallback is intentionally explicit: it is correct, but may perform I/O.
        print(
            "[selector-audit] Warning: metadata-only path unavailable; scanning __getitem__.",
            flush=True,
        )
        records = [_record(dataset[index], index) for index in range(len(dataset))]

    if len(records) != len(dataset):
        raise RuntimeError(
            f"Audit metadata count differs from dataset: {len(records)} vs {len(dataset)}"
        )
    uids = [row["sample_uid"] for row in records]
    if len(set(uids)) != len(uids):
        raise RuntimeError("Audit metadata contains duplicate sample_uid values")
    return records


def build_control_mappings(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 521,
    minimum_far_frame_gap: int = 30,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Build deterministic controls whose relationship is known before model I/O."""

    count = len(records)
    mappings = {
        name: np.full(count, -1, dtype=np.int64) for name in MAPPED_CONTROLS
    }

    sequence_groups: dict[str, list[int]] = {}
    query_groups: dict[tuple[str, str], list[int]] = {}
    frame_groups: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(records):
        sequence = str(row.get("sequence_key", ""))
        query = str(row.get("query_alias", "query"))
        frame = int(row.get("frame_idx", 0))
        if sequence:
            sequence_groups.setdefault(sequence, []).append(index)
            query_groups.setdefault((sequence, query), []).append(index)
            frame_groups.setdefault((sequence, frame), []).append(index)

    sequence_keys = sorted(sequence_groups)
    if len(sequence_keys) >= 2:
        offset = 1 + _stable_index("cross-sequence-offset", len(sequence_keys) - 1, seed)
        for source_position, source_key in enumerate(sequence_keys):
            target_key = sequence_keys[(source_position + offset) % len(sequence_keys)]
            source_indices = sequence_groups[source_key]
            target_indices = sequence_groups[target_key]
            phase = _stable_index(source_key, len(target_indices), seed)
            for position, source_index in enumerate(source_indices):
                mappings["cross_sequence"][source_index] = target_indices[
                    (position + phase) % len(target_indices)
                ]

    for indices in query_groups.values():
        ordered = sorted(indices, key=lambda index: (int(records[index]["frame_idx"]), index))
        if len(ordered) < 2:
            continue
        first = ordered[0]
        last = ordered[-1]
        for index in ordered:
            frame = int(records[index]["frame_idx"])
            candidates = (first, last)
            alternate = max(
                candidates,
                key=lambda value: abs(int(records[value]["frame_idx"]) - frame),
            )
            gap = abs(int(records[alternate]["frame_idx"]) - frame)
            if alternate != index and gap >= int(minimum_far_frame_gap):
                mappings["same_sequence_far"][index] = alternate

    for indices in frame_groups.values():
        if len(indices) < 2:
            continue
        for index in indices:
            query = str(records[index]["query_alias"])
            candidates = [
                value for value in indices
                if value != index and str(records[value]["query_alias"]) != query
            ]
            if candidates:
                position = _stable_index(str(records[index]["sample_uid"]), len(candidates), seed)
                mappings["wrong_query"][index] = candidates[position]

    summaries: dict[str, dict[str, Any]] = {}
    for name, mapping in mappings.items():
        available = mapping >= 0
        same_sequence = 0
        same_query = 0
        frame_gaps: list[int] = []
        for source_index in np.flatnonzero(available):
            target_index = int(mapping[source_index])
            source = records[int(source_index)]
            target = records[target_index]
            same_sequence += int(source["sequence_key"] == target["sequence_key"])
            same_query += int(source["query_alias"] == target["query_alias"])
            frame_gaps.append(abs(int(source["frame_idx"]) - int(target["frame_idx"])))
        available_count = int(available.sum())
        summaries[name] = {
            "available_count": available_count,
            "unavailable_count": int(count - available_count),
            "available_fraction": available_count / max(count, 1),
            "same_sequence_fraction": same_sequence / max(available_count, 1),
            "same_query_fraction": same_query / max(available_count, 1),
            "frame_gap_median": (
                float(np.median(frame_gaps)) if frame_gaps else None
            ),
            "frame_gap_min": int(min(frame_gaps)) if frame_gaps else None,
            "frame_gap_max": int(max(frame_gaps)) if frame_gaps else None,
        }
    if summaries["cross_sequence"]["same_sequence_fraction"] != 0.0:
        raise AssertionError("Cross-sequence mapping contains a same-sequence pair")
    return mappings, summaries


class MappedPriorDataset(Dataset):
    """Attach alternate priors while avoiding duplicate RGB/base-feature reads."""

    def __init__(
        self,
        dataset: Dataset,
        records: Sequence[Mapping[str, Any]],
        mappings: Mapping[str, np.ndarray],
        *,
        prior_kind: str,
    ):
        self.dataset = dataset
        self.records = tuple(dict(row) for row in records)
        self.mappings = {name: np.asarray(value, dtype=np.int64) for name, value in mappings.items()}
        self.prior_kind = str(prior_kind)
        if self.prior_kind not in {"depth", "vlm"}:
            raise ValueError(f"Unsupported prior kind {self.prior_kind!r}")
        self.batch_key = "depth_prior" if self.prior_kind == "depth" else "vlm_prior"
        self.cache_field = "depth_grid" if self.prior_kind == "depth" else "vlm_embedding"
        self.available_key = (
            "depth_available" if self.prior_kind == "depth" else "vlm_available"
        )
        self._prior_cache_groups = self._find_prior_cache_groups(dataset)

    @staticmethod
    def _find_prior_cache_groups(dataset: Dataset):
        groups = getattr(dataset, "groups", None)
        if groups is None:
            groups = getattr(dataset, "cache_groups", None)
        if groups is None and hasattr(dataset, "dataset"):
            groups = getattr(dataset.dataset, "cache_groups", None)
        return tuple(groups or ())

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def _cached_prior(self, sample_uid: str, index: int):
        for group in self._prior_cache_groups:
            if not any(self.cache_field in cache.fields for cache in group):
                continue
            if group:
                cache = group[int(index) % len(group)]
                if self.cache_field in cache.fields:
                    local_index = int(index) // len(group)
                    try:
                        value = cache[local_index]
                    except (IndexError, KeyError):
                        value = None
                    if value is not None and str(value.get("sample_id", "")) == sample_uid:
                        prior = torch.from_numpy(
                            np.array(value[self.cache_field], copy=True)
                        ).float()
                        return {
                            self.batch_key: prior,
                            self.available_key: torch.tensor(True),
                        }
            for cache in group:
                if self.cache_field not in cache.fields:
                    continue
                try:
                    value = cache.get_by_id(sample_uid)
                except KeyError:
                    continue
                prior = torch.from_numpy(
                    np.array(value[self.cache_field], copy=True)
                ).float()
                return {self.batch_key: prior, self.available_key: torch.tensor(True)}
        return None

    def _prior_item(self, index: int) -> Mapping[str, Any]:
        uid = str(self.records[int(index)]["sample_uid"])
        cached = self._cached_prior(uid, int(index))
        if cached is not None:
            return cached
        return self.dataset[int(index)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[int(index)])
        if self.batch_key not in item:
            raise KeyError(f"Base audit item is missing {self.batch_key!r}")
        for control, mapping in self.mappings.items():
            alternate_index = int(mapping[int(index)])
            prefix = f"_audit_{control}"
            if alternate_index < 0:
                item[f"{prefix}_prior"] = torch.zeros_like(item[self.batch_key])
                item[f"{prefix}_available"] = torch.tensor(False)
                if self.prior_kind == "depth" and "depth_valid" in item:
                    item[f"{prefix}_valid"] = torch.zeros_like(item["depth_valid"])
                continue
            alternate = self._prior_item(alternate_index)
            prior = alternate.get(self.batch_key)
            if prior is None:
                raise KeyError(
                    f"Alternate sample {self.records[alternate_index]['sample_uid']!r} "
                    f"is missing {self.batch_key!r}"
                )
            if tuple(prior.shape) != tuple(item[self.batch_key].shape):
                raise ValueError(
                    f"Mapped prior shape differs for {control}: {tuple(prior.shape)} vs "
                    f"{tuple(item[self.batch_key].shape)}"
                )
            item[f"{prefix}_prior"] = prior
            item[f"{prefix}_available"] = torch.as_tensor(
                alternate.get(self.available_key, True), dtype=torch.bool
            ).reshape(())
            if self.prior_kind == "depth" and "depth_valid" in item:
                item[f"{prefix}_valid"] = alternate.get(
                    "depth_valid", torch.ones_like(item["depth_valid"])
                )
        return item


def mapped_control_batch(batch: Mapping[str, Any], control: str, *, is_depth: bool):
    prefix = f"_audit_{control}"
    result = dict(batch)
    prior_key = "depth_prior" if is_depth else "vlm_prior"
    available_key = "depth_available" if is_depth else "vlm_available"
    result[prior_key] = batch[f"{prefix}_prior"]
    result[available_key] = batch[f"{prefix}_available"]
    if is_depth and f"{prefix}_valid" in batch:
        result["depth_valid"] = batch[f"{prefix}_valid"]
    return result


@dataclass(frozen=True)
class PressurePolicy:
    name: str
    score_threshold: float
    alpha: float
    target_floor: float


_POLICY_FIELDS = (
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
    "selected_count",
    "strict_selected",
    "subthreshold_selected",
    "gray_selected",
    "protected_selected",
    "strict_candidate",
    "subthreshold_candidate",
    "gray_candidate",
    "protected_candidate",
    "correction_total",
    "correction_strict",
    "correction_subthreshold",
    "correction_gray",
    "correction_protected",
    "added_under_total",
    "added_under_protected",
    "corrected_crossings",
    "harmful_crossings",
    "strict_excess_before",
    "strict_excess_after",
    "high_gt_pred_sum",
    "high_gt_count",
    "low_gt_pred_sum",
    "low_gt_count",
)


def policy_grid(
    score_thresholds: Sequence[float],
    alphas: Sequence[float],
    target_floors: Sequence[float],
) -> tuple[PressurePolicy, ...]:
    policies = [PressurePolicy("base", 2.0, 0.0, 0.0)]
    for threshold in score_thresholds:
        for alpha in alphas:
            for floor in target_floors:
                if not 0.0 <= float(threshold) <= 1.0:
                    raise ValueError("Selector score thresholds must be in [0,1]")
                if not 0.0 < float(alpha) <= 1.0:
                    raise ValueError("Policy alpha values must be in (0,1]")
                if not 0.0 <= float(floor) < 1.0:
                    raise ValueError("Policy target floors must be in [0,1)")
                name = f"s{threshold:g}_a{alpha:g}_f{floor:g}"
                policies.append(
                    PressurePolicy(name, float(threshold), float(alpha), float(floor))
                )
    names = [policy.name for policy in policies]
    if len(set(names)) != len(names):
        raise ValueError("Pressure-policy grid contains duplicate configurations")
    return tuple(policies)


class PressurePolicyAccumulator:
    def __init__(
        self,
        policies: Sequence[PressurePolicy],
        *,
        action_threshold: float = 0.10,
        no_contact_max: float = 0.02,
        subthreshold_max: float = 0.08,
        contact_min: float = 0.10,
        chunk_size: int = 8,
    ):
        if not policies or policies[0].name != "base":
            raise ValueError("The first pressure policy must be the base policy")
        if not 0.0 <= no_contact_max < subthreshold_max < contact_min <= 1.0:
            raise ValueError(
                "Expected no_contact_max < subthreshold_max < contact_min within [0,1]"
            )
        self.policies = tuple(policies)
        self.action_threshold = float(action_threshold)
        self.no_contact_max = float(no_contact_max)
        self.subthreshold_max = float(subthreshold_max)
        self.contact_min = float(contact_min)
        self.chunk_size = max(1, int(chunk_size))
        self.values = torch.zeros(
            (len(self.policies), len(_POLICY_FIELDS)), dtype=torch.float64
        )

    @torch.no_grad()
    def update(
        self,
        base_prediction: torch.Tensor,
        contact_logits: torch.Tensor,
        target: torch.Tensor,
        palm_mask: torch.Tensor,
        has_tactile: torch.Tensor,
        *,
        false_high_logits: torch.Tensor | None = None,
    ) -> None:
        pred = base_prediction.detach().float()
        logits = contact_logits.detach().float().to(pred.device)
        target = target.detach().float().to(pred.device)
        palm = palm_mask.detach().to(pred.device) > 0.5
        if palm.ndim == 1:
            palm = palm[None].expand_as(pred)
        elif palm.shape[0] == 1:
            palm = palm.expand_as(pred)
        has = has_tactile.detach().reshape(-1).to(pred.device) > 0.5
        valid = palm & has[:, None]
        if not bool(valid.any()):
            return
        if false_high_logits is None:
            score = torch.sigmoid(-logits)
        else:
            false_high_logits = false_high_logits.detach().float().to(pred.device)
            if false_high_logits.shape != pred.shape:
                raise ValueError(
                    "Pressure-policy false-high logits must match the pressure shape"
                )
            score = torch.sigmoid(false_high_logits)
        candidate = valid & (pred >= self.action_threshold)
        strict = candidate & (target <= self.no_contact_max)
        subthreshold = candidate & (target <= self.subthreshold_max)
        gray = candidate & (target > self.no_contact_max) & (target < self.contact_min)
        protected = candidate & (target >= self.contact_min)
        base_under = (target - pred).clamp_min(0.0)

        for start in range(0, len(self.policies), self.chunk_size):
            current = self.policies[start : start + self.chunk_size]
            thresholds = pred.new_tensor([item.score_threshold for item in current])[:, None, None]
            alphas = pred.new_tensor([item.alpha for item in current])[:, None, None]
            floors = pred.new_tensor([item.target_floor for item in current])[:, None, None]
            selected = candidate[None] & (score[None] >= thresholds)
            correction = alphas * selected * (pred[None] - floors).clamp_min(0.0)
            corrected = (pred[None] - correction).clamp(0.0, 1.0)
            valid_c = valid[None]
            difference = (corrected - target[None]) * valid_c
            pred_masked = corrected * valid_c
            target_masked = target[None] * valid_c
            pred_volume = pred_masked.sum(dim=2)
            gt_volume = (target * valid).sum(dim=1)
            pred_contact = (corrected >= self.contact_min) & valid_c
            gt_contact = (target >= self.contact_min) & valid
            intersection = (pred_contact & gt_contact[None]).sum(dim=2).double()
            union = (pred_contact | gt_contact[None]).sum(dim=2).double()
            contact_iou = torch.where(union > 0, intersection / union.clamp_min(1.0), 1.0)
            viou_intersection = torch.minimum(pred_masked, target_masked).sum(dim=2)
            viou_union = torch.maximum(pred_masked, target_masked).sum(dim=2)
            viou = torch.where(
                viou_union > 1e-12,
                viou_intersection / viou_union.clamp_min(1e-12),
                1.0,
            )
            eligible_core = (gt_volume >= 1.0) & ((target * valid).amax(dim=1) >= 0.05)
            pred_core = pred_masked.square()
            gt_core = target_masked.square()
            pred_dist = pred_core / pred_core.sum(dim=2, keepdim=True).clamp_min(1e-12)
            gt_dist = gt_core / gt_core.sum(dim=2, keepdim=True).clamp_min(1e-12)
            core_intersection = torch.minimum(pred_dist, gt_dist).sum(dim=2)
            core_union = torch.maximum(pred_dist, gt_dist).sum(dim=2)
            core_viou = core_intersection / core_union.clamp_min(1e-12)
            false_high = valid_c & (target[None] < 0.005) & (corrected >= 0.3)
            added_under = ((target[None] - corrected).clamp_min(0.0) - base_under[None]).clamp_min(0.0)

            additions = torch.stack(
                (
                    pred.new_full((len(current),), float(has.sum())),
                    pred.new_full((len(current),), float(valid.sum())),
                    difference.abs().sum(dim=(1, 2)),
                    difference.square().sum(dim=(1, 2)),
                    contact_iou[:, has].sum(dim=1),
                    viou[:, has].sum(dim=1),
                    pred_volume[:, has].sum(dim=1),
                    gt_volume[has].sum().expand(len(current)),
                    ((pred_contact.any(dim=2)) == gt_contact.any(dim=1)[None])[:, has]
                    .float().sum(dim=1),
                    core_viou[:, eligible_core].sum(dim=1),
                    eligible_core.float().sum().expand(len(current)),
                    ((corrected - target[None]).clamp_min(0.0) * false_high).sum(dim=(1, 2)),
                    ((gt_volume[None] < 10.0) & (pred_volume > 300.0))[:, has].float().sum(dim=1),
                    ((gt_volume < 10.0) & has).float().sum().expand(len(current)),
                    ((gt_volume[None] >= 150.0) & (pred_volume < 50.0))[:, has].float().sum(dim=1),
                    ((gt_volume >= 150.0) & has).float().sum().expand(len(current)),
                    selected.sum(dim=(1, 2)),
                    (selected & strict[None]).sum(dim=(1, 2)),
                    (selected & subthreshold[None]).sum(dim=(1, 2)),
                    (selected & gray[None]).sum(dim=(1, 2)),
                    (selected & protected[None]).sum(dim=(1, 2)),
                    strict.sum().expand(len(current)),
                    subthreshold.sum().expand(len(current)),
                    gray.sum().expand(len(current)),
                    protected.sum().expand(len(current)),
                    correction.sum(dim=(1, 2)),
                    (correction * strict[None]).sum(dim=(1, 2)),
                    (correction * subthreshold[None]).sum(dim=(1, 2)),
                    (correction * gray[None]).sum(dim=(1, 2)),
                    (correction * protected[None]).sum(dim=(1, 2)),
                    (added_under * valid_c).sum(dim=(1, 2)),
                    (added_under * protected[None]).sum(dim=(1, 2)),
                    (subthreshold[None] & (corrected < self.action_threshold)).sum(dim=(1, 2)),
                    (protected[None] & (corrected < self.action_threshold)).sum(dim=(1, 2)),
                    (((pred - target).clamp_min(0.0) * strict).sum()).expand(len(current)),
                    (((corrected - target[None]).clamp_min(0.0) * strict[None]).sum(dim=(1, 2))),
                    (corrected * (valid & (target >= 0.70))[None]).sum(dim=(1, 2)),
                    (valid & (target >= 0.70)).sum().expand(len(current)),
                    (corrected * (valid & (target < 0.005))[None]).sum(dim=(1, 2)),
                    (valid & (target < 0.005)).sum().expand(len(current)),
                ),
                dim=1,
            ).double().cpu()
            expected_shape = (len(current), len(_POLICY_FIELDS))
            if tuple(additions.shape) != expected_shape:
                raise RuntimeError(
                    "Pressure-policy metric layout mismatch: "
                    f"expected={expected_shape}, actual={tuple(additions.shape)}"
                )
            self.values[start : start + len(current)] += additions

    def synchronize(self, device: torch.device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        values = self.values.to(device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        self.values.copy_(values.cpu())

    def summaries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for policy, values in zip(self.policies, self.values):
            raw = {name: float(values[index]) for index, name in enumerate(_POLICY_FIELDS)}
            frames = max(raw["frames"], 1.0)
            points = max(raw["values"], 1.0)
            selected = max(raw["selected_count"], 1.0)
            clear_selected = max(raw["strict_selected"] + raw["protected_selected"], 1.0)
            strict_removed = raw["correction_strict"]
            row = {
                **asdict(policy),
                "frame_count": raw["frames"],
                "mae": raw["abs_sum"] / points,
                "rmse": math.sqrt(raw["sq_sum"] / points),
                "contact_iou": raw["contact_iou_sum"] / frames,
                "volumetric_iou": raw["viou_sum"] / frames,
                "core_distribution_viou": raw["core_viou_sum"] / max(raw["core_count"], 1.0),
                "pred_gt_volume_ratio": raw["pred_volume"] / max(raw["gt_volume"], 1e-12),
                "temporal_accuracy_frame": raw["temporal_correct"] / frames,
                "false_high_excess_fraction": raw["false_high_excess"] / max(raw["pred_volume"], 1e-12),
                "catastrophic_over_rate": raw["cat_over"] / max(raw["cat_over_denom"], 1.0),
                "catastrophic_under_rate": raw["cat_under"] / max(raw["cat_under_denom"], 1.0),
                "action_coverage": raw["selected_count"] / points,
                "strict_precision_all_selected": raw["strict_selected"] / selected,
                "strict_precision_clear_only": raw["strict_selected"] / clear_selected,
                "strict_recall": raw["strict_selected"] / max(raw["strict_candidate"], 1.0),
                "subthreshold_recall": raw["subthreshold_selected"] / max(raw["subthreshold_candidate"], 1.0),
                "protected_selection_rate": raw["protected_selected"] / max(raw["protected_candidate"], 1.0),
                "gray_selection_rate": raw["gray_selected"] / max(raw["gray_candidate"], 1.0),
                "correction_total": raw["correction_total"],
                "strict_false_high_volume_removed": strict_removed,
                "strict_false_high_volume_removed_fraction": strict_removed / max(raw["strict_excess_before"], 1e-12),
                "strict_false_high_excess_after_fraction": raw["strict_excess_after"] / max(raw["strict_excess_before"], 1e-12),
                "subthreshold_volume_removed": raw["correction_subthreshold"],
                "gray_volume_removed": raw["correction_gray"],
                "protected_contact_volume_removed": raw["correction_protected"],
                "added_under_volume": raw["added_under_total"],
                "added_under_protected_volume": raw["added_under_protected"],
                "protected_removed_per_strict_removed": raw["correction_protected"] / max(strict_removed, 1e-12),
                "added_under_per_strict_removed": raw["added_under_total"] / max(strict_removed, 1e-12),
                "threshold_crossing_correction_recall": raw["corrected_crossings"] / max(raw["subthreshold_candidate"], 1.0),
                "protected_harmful_crossing_rate": raw["harmful_crossings"] / max(raw["protected_candidate"], 1.0),
                "high_gt_mean_prediction": raw["high_gt_pred_sum"] / max(raw["high_gt_count"], 1.0),
                "low_gt_mean_prediction": raw["low_gt_pred_sum"] / max(raw["low_gt_count"], 1.0),
                "balanced_net_utility": (
                    strict_removed
                    - raw["correction_protected"]
                    - raw["added_under_total"]
                ),
                "balanced_net_utility_fraction": (
                    strict_removed
                    - raw["correction_protected"]
                    - raw["added_under_total"]
                ) / max(raw["strict_excess_before"], 1e-12),
            }
            rows.append(row)
        base = rows[0]
        for row in rows:
            row.update(
                {
                    "delta_rmse": row["rmse"] - base["rmse"],
                    "delta_contact_iou": row["contact_iou"] - base["contact_iou"],
                    "delta_volumetric_iou": row["volumetric_iou"] - base["volumetric_iou"],
                    "delta_core_distribution_viou": row["core_distribution_viou"] - base["core_distribution_viou"],
                    "delta_temporal_accuracy": row["temporal_accuracy_frame"] - base["temporal_accuracy_frame"],
                    "delta_false_high_excess_fraction": row["false_high_excess_fraction"] - base["false_high_excess_fraction"],
                    "delta_high_gt_mean_prediction": row["high_gt_mean_prediction"] - base["high_gt_mean_prediction"],
                    "delta_low_gt_mean_prediction": row["low_gt_mean_prediction"] - base["low_gt_mean_prediction"],
                }
            )
        return rows


_POLICY_PAIR_FIELDS = (
    "eligible_frames",
    "eligible_vertices",
    "reference_selected",
    "control_selected",
    "selected_intersection",
    "selected_union",
    "selected_disagreement",
    "strict_removed_reference",
    "strict_removed_control",
    "protected_removed_reference",
    "protected_removed_control",
    "added_under_reference",
    "added_under_control",
    "net_utility_reference",
    "net_utility_control",
    "cat_over_reference",
    "cat_over_control",
)

_POLICY_PAIR_SEQUENCE_FIELDS = (
    "eligible_frames",
    "strict_removed_reference",
    "strict_removed_control",
    "protected_removed_reference",
    "protected_removed_control",
    "added_under_reference",
    "added_under_control",
    "net_utility_reference",
    "net_utility_control",
    "cat_over_reference",
    "cat_over_control",
)


class PressurePolicyPairAccumulator:
    """Compare one fixed policy under aligned and counterfactual selector scores.

    The pressure prediction, target, candidate set, and policy parameters are
    shared. Only the selector logits differ. This makes an aligned-control gap
    attributable to the prior evidence rather than to a different threshold.
    """

    def __init__(
        self,
        policies: Sequence[PressurePolicy],
        *,
        action_threshold: float = 0.10,
        no_contact_max: float = 0.02,
        contact_min: float = 0.10,
        chunk_size: int = 8,
        bootstrap_iterations: int = 2000,
        bootstrap_seed: int = 521,
        bootstrap_confidence: float = 0.95,
    ):
        if not policies or policies[0].name != "base":
            raise ValueError("The first pressure policy must be the base policy")
        self.policies = tuple(policies)
        self.action_threshold = float(action_threshold)
        self.no_contact_max = float(no_contact_max)
        self.contact_min = float(contact_min)
        self.chunk_size = max(1, int(chunk_size))
        self.bootstrap_iterations = max(0, int(bootstrap_iterations))
        self.bootstrap_seed = int(bootstrap_seed)
        self.bootstrap_confidence = float(bootstrap_confidence)
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("Policy-pair bootstrap confidence must lie in (0,1)")
        self.values = torch.zeros(
            (len(self.policies), len(_POLICY_PAIR_FIELDS)), dtype=torch.float64
        )
        self.sequence_values: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(
        self,
        base_prediction: torch.Tensor,
        reference_contact_logits: torch.Tensor,
        control_contact_logits: torch.Tensor,
        target: torch.Tensor,
        palm_mask: torch.Tensor,
        has_tactile: torch.Tensor,
        *,
        sequence_keys: Sequence[str],
        reference_false_high_logits: torch.Tensor | None = None,
        control_false_high_logits: torch.Tensor | None = None,
    ) -> None:
        pred = base_prediction.detach().float()
        reference_logits = reference_contact_logits.detach().float().to(pred.device)
        control_logits = control_contact_logits.detach().float().to(pred.device)
        target = target.detach().float().to(pred.device)
        if reference_logits.shape != pred.shape or control_logits.shape != pred.shape:
            raise ValueError("Policy-pair selector logits must match the pressure shape")
        if target.shape != pred.shape:
            raise ValueError("Policy-pair target must match the pressure shape")
        if len(sequence_keys) != pred.shape[0]:
            raise ValueError("sequence_key count does not match policy-pair batch size")

        palm = palm_mask.detach().to(pred.device) > 0.5
        if palm.ndim == 1:
            palm = palm[None].expand_as(pred)
        elif palm.shape[0] == 1:
            palm = palm.expand_as(pred)
        if palm.shape != pred.shape:
            raise ValueError("Policy-pair palm mask must match the pressure shape")
        has = has_tactile.detach().reshape(-1).to(pred.device) > 0.5
        valid = palm & has[:, None]
        if not bool(valid.any()):
            return

        candidate = valid & (pred >= self.action_threshold)
        strict = candidate & (target <= self.no_contact_max)
        protected = candidate & (target >= self.contact_min)
        base_under = (target - pred).clamp_min(0.0)
        gt_volume = (target * valid).sum(dim=1)
        if reference_false_high_logits is None:
            reference_score = torch.sigmoid(-reference_logits)
        else:
            reference_false_high_logits = (
                reference_false_high_logits.detach().float().to(pred.device)
            )
            if reference_false_high_logits.shape != pred.shape:
                raise ValueError(
                    "Reference false-high logits must match the pressure shape"
                )
            reference_score = torch.sigmoid(reference_false_high_logits)
        if control_false_high_logits is None:
            control_score = torch.sigmoid(-control_logits)
        else:
            control_false_high_logits = (
                control_false_high_logits.detach().float().to(pred.device)
            )
            if control_false_high_logits.shape != pred.shape:
                raise ValueError(
                    "Control false-high logits must match the pressure shape"
                )
            control_score = torch.sigmoid(control_false_high_logits)

        for start in range(0, len(self.policies), self.chunk_size):
            current = self.policies[start : start + self.chunk_size]
            thresholds = pred.new_tensor(
                [item.score_threshold for item in current]
            )[:, None, None]
            alphas = pred.new_tensor([item.alpha for item in current])[:, None, None]
            floors = pred.new_tensor(
                [item.target_floor for item in current]
            )[:, None, None]
            reference_selected = candidate[None] & (reference_score[None] >= thresholds)
            control_selected = candidate[None] & (control_score[None] >= thresholds)
            reference_correction = (
                alphas
                * reference_selected
                * (pred[None] - floors).clamp_min(0.0)
            )
            control_correction = (
                alphas
                * control_selected
                * (pred[None] - floors).clamp_min(0.0)
            )
            reference_corrected = (pred[None] - reference_correction).clamp(0.0, 1.0)
            control_corrected = (pred[None] - control_correction).clamp(0.0, 1.0)

            def policy_terms(correction, corrected):
                strict_removed = (correction * strict[None]).sum(dim=2)
                protected_removed = (correction * protected[None]).sum(dim=2)
                added_under = (
                    ((target[None] - corrected).clamp_min(0.0) - base_under[None])
                    .clamp_min(0.0)
                    * valid[None]
                ).sum(dim=2)
                net_utility = strict_removed - protected_removed - added_under
                pred_volume = (corrected * valid[None]).sum(dim=2)
                cat_over = (
                    (gt_volume[None] < 10.0) & (pred_volume > 300.0)
                ).float()
                return strict_removed, protected_removed, added_under, net_utility, cat_over

            reference_terms = policy_terms(reference_correction, reference_corrected)
            control_terms = policy_terms(control_correction, control_corrected)
            selected_intersection = reference_selected & control_selected
            selected_union = reference_selected | control_selected
            selected_disagreement = reference_selected ^ control_selected

            additions = torch.stack(
                (
                    has.float().sum().expand(len(current)),
                    valid.float().sum().expand(len(current)),
                    reference_selected.sum(dim=(1, 2)),
                    control_selected.sum(dim=(1, 2)),
                    selected_intersection.sum(dim=(1, 2)),
                    selected_union.sum(dim=(1, 2)),
                    selected_disagreement.sum(dim=(1, 2)),
                    reference_terms[0].sum(dim=1),
                    control_terms[0].sum(dim=1),
                    reference_terms[1].sum(dim=1),
                    control_terms[1].sum(dim=1),
                    reference_terms[2].sum(dim=1),
                    control_terms[2].sum(dim=1),
                    reference_terms[3].sum(dim=1),
                    control_terms[3].sum(dim=1),
                    reference_terms[4][:, has].sum(dim=1),
                    control_terms[4][:, has].sum(dim=1),
                ),
                dim=1,
            ).double().cpu()
            expected_shape = (len(current), len(_POLICY_PAIR_FIELDS))
            if tuple(additions.shape) != expected_shape:
                raise RuntimeError(
                    "Pressure-policy pair layout mismatch: "
                    f"expected={expected_shape}, actual={tuple(additions.shape)}"
                )
            self.values[start : start + len(current)] += additions

            sequence_additions = torch.stack(
                (
                    has.float()[None].expand(len(current), -1),
                    reference_terms[0],
                    control_terms[0],
                    reference_terms[1],
                    control_terms[1],
                    reference_terms[2],
                    control_terms[2],
                    reference_terms[3],
                    control_terms[3],
                    reference_terms[4],
                    control_terms[4],
                ),
                dim=2,
            ).double().cpu()
            for batch_index, raw_key in enumerate(sequence_keys):
                if not bool(has[batch_index]):
                    continue
                key = str(raw_key)
                if key not in self.sequence_values:
                    self.sequence_values[key] = torch.zeros(
                        (len(self.policies), len(_POLICY_PAIR_SEQUENCE_FIELDS)),
                        dtype=torch.float64,
                    )
                self.sequence_values[key][start : start + len(current)] += (
                    sequence_additions[:, batch_index]
                )

    def synchronize(self, device: torch.device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        values = self.values.to(device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        self.values.copy_(values.cpu())
        gathered: list[dict[str, torch.Tensor] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, self.sequence_values)
        merged: dict[str, torch.Tensor] = {}
        for shard in gathered:
            if shard is None:
                continue
            for key, sequence_value in shard.items():
                if key not in merged:
                    merged[key] = torch.zeros_like(sequence_value)
                merged[key].add_(sequence_value)
        self.sequence_values = merged

    def _bootstrap_interval(
        self,
        policy_index: int,
        reference_field: str,
        control_field: str,
    ) -> tuple[float, float, float, int]:
        reference_index = _POLICY_PAIR_SEQUENCE_FIELDS.index(reference_field)
        control_index = _POLICY_PAIR_SEQUENCE_FIELDS.index(control_field)
        values = []
        for key in sorted(self.sequence_values):
            sequence_value = self.sequence_values[key]
            if sequence_value[policy_index, 0] <= 0:
                continue
            values.append(
                float(
                    sequence_value[policy_index, reference_index]
                    - sequence_value[policy_index, control_index]
                )
            )
        if not values:
            return 0.0, 0.0, 0.0, 0
        array = np.asarray(values, dtype=np.float64)
        positive_fraction = float(np.mean(array > 0.0))
        if self.bootstrap_iterations <= 0 or len(array) == 1:
            total = float(array.sum())
            return total, total, positive_fraction, len(array)
        rng = np.random.default_rng(self.bootstrap_seed + 1009 * policy_index)
        draws = rng.integers(
            0,
            len(array),
            size=(self.bootstrap_iterations, len(array)),
            endpoint=False,
        )
        totals = array[draws].sum(axis=1)
        tail = (1.0 - self.bootstrap_confidence) * 0.5
        low, high = np.quantile(totals, (tail, 1.0 - tail))
        return float(low), float(high), positive_fraction, len(array)

    def _bootstrap_single_interval(
        self,
        policy_index: int,
        field: str,
    ) -> tuple[float, float, float, int]:
        field_index = _POLICY_PAIR_SEQUENCE_FIELDS.index(field)
        values = []
        for key in sorted(self.sequence_values):
            sequence_value = self.sequence_values[key]
            if sequence_value[policy_index, 0] <= 0:
                continue
            values.append(float(sequence_value[policy_index, field_index]))
        if not values:
            return 0.0, 0.0, 0.0, 0
        array = np.asarray(values, dtype=np.float64)
        positive_fraction = float(np.mean(array > 0.0))
        if self.bootstrap_iterations <= 0 or len(array) == 1:
            total = float(array.sum())
            return total, total, positive_fraction, len(array)
        rng = np.random.default_rng(
            self.bootstrap_seed + 1009 * policy_index + 7919 * (field_index + 1)
        )
        draws = rng.integers(
            0,
            len(array),
            size=(self.bootstrap_iterations, len(array)),
            endpoint=False,
        )
        totals = array[draws].sum(axis=1)
        tail = (1.0 - self.bootstrap_confidence) * 0.5
        low, high = np.quantile(totals, (tail, 1.0 - tail))
        return float(low), float(high), positive_fraction, len(array)

    def summaries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        field_index = {name: index for index, name in enumerate(_POLICY_PAIR_FIELDS)}
        for policy_index, (policy, values) in enumerate(zip(self.policies, self.values)):
            raw = {name: float(values[index]) for name, index in field_index.items()}
            union = raw["selected_union"]
            reference_selected = raw["reference_selected"]
            control_selected = raw["control_selected"]
            net_low, net_high, positive_fraction, sequence_count = self._bootstrap_interval(
                policy_index,
                "net_utility_reference",
                "net_utility_control",
            )
            (
                reference_net_low,
                reference_net_high,
                reference_positive_fraction,
                reference_sequence_count,
            ) = self._bootstrap_single_interval(
                policy_index, "net_utility_reference"
            )
            (
                control_net_low,
                control_net_high,
                control_positive_fraction,
                control_sequence_count,
            ) = self._bootstrap_single_interval(
                policy_index, "net_utility_control"
            )
            if reference_sequence_count != control_sequence_count:
                raise RuntimeError("Reference/control policy sequence coverage differs")
            rows.append(
                {
                    **asdict(policy),
                    "eligible_frame_count": raw["eligible_frames"],
                    "eligible_vertex_count": raw["eligible_vertices"],
                    "reference_action_count": reference_selected,
                    "control_action_count": control_selected,
                    "action_jaccard": (
                        raw["selected_intersection"] / union if union > 0.0 else 1.0
                    ),
                    "action_disagreement_fraction": (
                        raw["selected_disagreement"] / union if union > 0.0 else 0.0
                    ),
                    "reference_action_retained_by_control": (
                        raw["selected_intersection"] / reference_selected
                        if reference_selected > 0.0
                        else 1.0
                    ),
                    "control_action_supported_by_reference": (
                        raw["selected_intersection"] / control_selected
                        if control_selected > 0.0
                        else 1.0
                    ),
                    "aligned_minus_control_strict_removed": (
                        raw["strict_removed_reference"] - raw["strict_removed_control"]
                    ),
                    "aligned_minus_control_protected_removed": (
                        raw["protected_removed_reference"] - raw["protected_removed_control"]
                    ),
                    "aligned_minus_control_added_under": (
                        raw["added_under_reference"] - raw["added_under_control"]
                    ),
                    "aligned_minus_control_net_utility": (
                        raw["net_utility_reference"] - raw["net_utility_control"]
                    ),
                    "aligned_minus_control_catastrophic_over_count": (
                        raw["cat_over_reference"] - raw["cat_over_control"]
                    ),
                    "aligned_net_utility": raw["net_utility_reference"],
                    "control_net_utility": raw["net_utility_control"],
                    "aligned_sequence_positive_net_fraction": (
                        reference_positive_fraction
                    ),
                    "control_sequence_positive_net_fraction": (
                        control_positive_fraction
                    ),
                    "aligned_net_utility_ci95_low": reference_net_low,
                    "aligned_net_utility_ci95_high": reference_net_high,
                    "control_net_utility_ci95_low": control_net_low,
                    "control_net_utility_ci95_high": control_net_high,
                    "paired_sequence_count": sequence_count,
                    "paired_sequence_positive_net_fraction": positive_fraction,
                    "paired_sequence_net_utility_ci95_low": net_low,
                    "paired_sequence_net_utility_ci95_high": net_high,
                }
            )
        return rows


def pareto_and_recommendations(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    candidates = [dict(row) for row in rows if row["name"] != "base"]
    recommendation_candidates = [dict(row) for row in rows]
    pareto: list[str] = []
    objectives = (
        ("strict_false_high_volume_removed_fraction", 1.0),
        ("protected_contact_volume_removed", -1.0),
        ("added_under_volume", -1.0),
        ("delta_rmse", -1.0),
        ("delta_contact_iou", 1.0),
        ("delta_high_gt_mean_prediction", 1.0),
    )
    for row in candidates:
        dominated = False
        for other in candidates:
            if other["name"] == row["name"]:
                continue
            comparisons = [
                direction * float(other[key]) >= direction * float(row[key]) - 1e-12
                for key, direction in objectives
            ]
            strict = [
                direction * float(other[key]) > direction * float(row[key]) + 1e-12
                for key, direction in objectives
            ]
            if all(comparisons) and any(strict):
                dominated = True
                break
        if not dominated:
            pareto.append(str(row["name"]))

    recommendations: dict[str, Any] = {}
    for profile, harm_weight in (("aggressive", 0.5), ("balanced", 1.0), ("conservative", 2.0)):
        best = max(
            recommendation_candidates,
            key=lambda row: (
                float(row["strict_false_high_volume_removed"])
                - harm_weight
                * (
                    float(row["protected_contact_volume_removed"])
                    + float(row["added_under_volume"])
                ),
                float(row["delta_contact_iou"]),
                -float(row["delta_rmse"]),
            ),
        )
        recommendations[profile] = {
            "policy": {
                key: best[key]
                for key in ("name", "score_threshold", "alpha", "target_floor")
            },
            "harm_weight": harm_weight,
            "utility": (
                float(best["strict_false_high_volume_removed"])
                - harm_weight
                * (
                    float(best["protected_contact_volume_removed"])
                    + float(best["added_under_volume"])
                )
            ),
        }
    return sorted(pareto), recommendations


_MATCHED_POLICY_METRICS = (
    "action_coverage",
    "strict_precision_all_selected",
    "strict_precision_clear_only",
    "strict_recall",
    "strict_false_high_volume_removed_fraction",
    "protected_removed_per_strict_removed",
    "added_under_per_strict_removed",
    "balanced_net_utility_fraction",
    "delta_rmse",
    "delta_contact_iou",
    "delta_volumetric_iou",
    "delta_core_distribution_viou",
    "delta_temporal_accuracy",
    "delta_false_high_excess_fraction",
    "delta_high_gt_mean_prediction",
)


def _relative_budget_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-12)


def _policy_dominance(
    depth: Mapping[str, Any],
    rgb: Mapping[str, Any],
    *,
    axis: str,
) -> tuple[bool, bool]:
    if axis == "action_coverage":
        objectives = (
            ("strict_precision_all_selected", 1.0),
            ("strict_false_high_volume_removed_fraction", 1.0),
            ("protected_removed_per_strict_removed", -1.0),
            ("added_under_per_strict_removed", -1.0),
        )
    elif axis == "strict_false_high_volume_removed_fraction":
        objectives = (
            ("action_coverage", -1.0),
            ("strict_precision_all_selected", 1.0),
            ("protected_removed_per_strict_removed", -1.0),
            ("added_under_per_strict_removed", -1.0),
        )
    else:
        raise ValueError(f"Unsupported matched policy axis {axis!r}")

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        no_worse = []
        strictly_better = []
        for key, direction in objectives:
            lhs = direction * float(left[key])
            rhs = direction * float(right[key])
            no_worse.append(lhs >= rhs - 1e-12)
            strictly_better.append(lhs > rhs + 1e-12)
        return all(no_worse) and any(strictly_better)

    return dominates(depth, rgb), dominates(rgb, depth)


def matched_policy_comparisons(
    aligned_rows: Sequence[Mapping[str, Any]],
    rgb_rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
    relative_tolerance: float = 0.10,
) -> list[dict[str, Any]]:
    """Compare aligned and RGB policies at a matched intervention budget.

    Coverage matching preserves alpha and target floor, which isolates selector
    ranking rather than silently changing correction strength. Removed-volume
    matching allows all RGB policies because the achieved benefit is the budget.
    This routine is diagnostic and never selects a deployment policy.
    """

    supported_axes = {
        "action_coverage",
        "strict_false_high_volume_removed_fraction",
    }
    if axis not in supported_axes:
        raise ValueError(f"Unsupported matched policy axis {axis!r}")
    if not 0.0 <= float(relative_tolerance) <= 1.0:
        raise ValueError("Matched policy relative tolerance must be in [0,1]")
    aligned = [dict(row) for row in aligned_rows if str(row["name"]) != "base"]
    rgb = [dict(row) for row in rgb_rows if str(row["name"]) != "base"]
    if not aligned or not rgb:
        raise ValueError("Matched policy audit requires non-base policies from both sources")

    comparisons: list[dict[str, Any]] = []
    for depth in aligned:
        depth_budget = float(depth[axis])
        if depth_budget <= 0.0:
            continue
        candidates = rgb
        shape_locked = axis == "action_coverage"
        if shape_locked:
            candidates = [
                row
                for row in rgb
                if math.isclose(float(row["alpha"]), float(depth["alpha"]), abs_tol=1e-12)
                and math.isclose(
                    float(row["target_floor"]),
                    float(depth["target_floor"]),
                    abs_tol=1e-12,
                )
            ]
        candidates = [row for row in candidates if float(row[axis]) > 0.0]
        if not candidates:
            continue
        rgb_match = min(
            candidates,
            key=lambda row: (
                _relative_budget_error(depth_budget, float(row[axis])),
                abs(depth_budget - float(row[axis])),
                str(row["name"]),
            ),
        )
        rgb_budget = float(rgb_match[axis])
        relative_error = _relative_budget_error(depth_budget, rgb_budget)
        depth_dominates, rgb_dominates = _policy_dominance(
            depth, rgb_match, axis=axis
        )
        row: dict[str, Any] = {
            "match_axis": axis,
            "aligned_policy": str(depth["name"]),
            "rgb_policy": str(rgb_match["name"]),
            "aligned_score_threshold": float(depth["score_threshold"]),
            "rgb_score_threshold": float(rgb_match["score_threshold"]),
            "aligned_alpha": float(depth["alpha"]),
            "rgb_alpha": float(rgb_match["alpha"]),
            "aligned_target_floor": float(depth["target_floor"]),
            "rgb_target_floor": float(rgb_match["target_floor"]),
            "correction_shape_locked": shape_locked,
            "aligned_budget": depth_budget,
            "rgb_budget": rgb_budget,
            "absolute_budget_error": abs(depth_budget - rgb_budget),
            "relative_budget_error": relative_error,
            "matched_within_tolerance": relative_error <= float(relative_tolerance),
            "depth_dominates": depth_dominates,
            "rgb_dominates": rgb_dominates,
        }
        for key in _MATCHED_POLICY_METRICS:
            depth_value = float(depth[key])
            rgb_value = float(rgb_match[key])
            row[f"aligned_{key}"] = depth_value
            row[f"rgb_{key}"] = rgb_value
            row[f"depth_minus_rgb_{key}"] = depth_value - rgb_value
        comparisons.append(row)
    return comparisons


def _matched_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matched = [dict(row) for row in rows if bool(row["matched_within_tolerance"])]
    if not matched:
        return {
            "row_count": len(rows),
            "matched_row_count": 0,
            "matched_fraction": 0.0,
            "status": "insufficient_budget_overlap",
        }

    def fraction(predicate) -> float:
        return sum(bool(predicate(row)) for row in matched) / len(matched)

    def median_gap(key: str) -> float:
        return float(np.median([float(row[f"depth_minus_rgb_{key}"]) for row in matched]))

    depth_dominance = fraction(lambda row: row["depth_dominates"])
    rgb_dominance = fraction(lambda row: row["rgb_dominates"])
    if depth_dominance >= 0.60 and rgb_dominance <= 0.10:
        status = "depth_frontier_advantage"
    elif rgb_dominance >= 0.60 and depth_dominance <= 0.10:
        status = "rgb_frontier_advantage"
    else:
        status = "mixed_or_indistinguishable"
    return {
        "row_count": len(rows),
        "matched_row_count": len(matched),
        "matched_fraction": len(matched) / max(len(rows), 1),
        "status": status,
        "depth_dominance_fraction": depth_dominance,
        "rgb_dominance_fraction": rgb_dominance,
        "depth_higher_precision_fraction": fraction(
            lambda row: row["depth_minus_rgb_strict_precision_all_selected"] > 0.0
        ),
        "depth_lower_added_under_fraction": fraction(
            lambda row: row["depth_minus_rgb_added_under_per_strict_removed"] < 0.0
        ),
        "depth_lower_protected_removal_fraction": fraction(
            lambda row: row[
                "depth_minus_rgb_protected_removed_per_strict_removed"
            ] < 0.0
        ),
        "median_depth_minus_rgb_precision": median_gap(
            "strict_precision_all_selected"
        ),
        "median_depth_minus_rgb_added_under_per_removed": median_gap(
            "added_under_per_strict_removed"
        ),
        "median_depth_minus_rgb_protected_per_removed": median_gap(
            "protected_removed_per_strict_removed"
        ),
        "median_depth_minus_rgb_balanced_utility_fraction": median_gap(
            "balanced_net_utility_fraction"
        ),
        "median_depth_minus_rgb_contact_iou_delta": median_gap(
            "delta_contact_iou"
        ),
    }


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"Cannot write empty matched-policy table: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_matched_policy_outputs(
    output_dir: Path,
    *,
    aligned_rows: Sequence[Mapping[str, Any]],
    rgb_rows: Sequence[Mapping[str, Any]],
    coverage_relative_tolerance: float = 0.10,
    removal_relative_tolerance: float = 0.10,
) -> dict[str, Any]:
    """Write diagnostic, non-deployment matched-budget frontier comparisons."""

    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_rows = [dict(row) for row in aligned_rows]
    rgb_rows = [dict(row) for row in rgb_rows]
    coverage_rows = matched_policy_comparisons(
        aligned_rows,
        rgb_rows,
        axis="action_coverage",
        relative_tolerance=coverage_relative_tolerance,
    )
    removal_rows = matched_policy_comparisons(
        aligned_rows,
        rgb_rows,
        axis="strict_false_high_volume_removed_fraction",
        relative_tolerance=removal_relative_tolerance,
    )
    paths = {
        "aligned_sweep_csv": output_dir / "pressure_policy_aligned_sweep.csv",
        "rgb_sweep_csv": output_dir / "pressure_policy_rgb_base_sweep.csv",
        "coverage_csv": output_dir / "pressure_policy_matched_coverage.csv",
        "removal_csv": output_dir / "pressure_policy_matched_removal.csv",
    }
    _write_rows(paths["aligned_sweep_csv"], aligned_rows)
    _write_rows(paths["rgb_sweep_csv"], rgb_rows)
    _write_rows(paths["coverage_csv"], coverage_rows)
    _write_rows(paths["removal_csv"], removal_rows)
    payload = {
        "schema": "tactile_selector_matched_policy_pareto_v1",
        "scope": "diagnostic_only_no_policy_selection",
        "sources": {
            "aligned": "prior-aware selector score",
            "rgb_base": "frozen RGB selector score",
        },
        "matching": {
            "coverage": "nearest action coverage with alpha and target_floor fixed",
            "removal": (
                "nearest strict false-high removed fraction across the policy grid"
            ),
            "coverage_relative_tolerance": float(coverage_relative_tolerance),
            "removal_relative_tolerance": float(removal_relative_tolerance),
        },
        "coverage_summary": _matched_summary(coverage_rows),
        "removal_summary": _matched_summary(removal_rows),
        "files": {key: str(value) for key, value in paths.items()},
    }
    json_path = output_dir / "pressure_policy_matched_pareto.json"
    summary_path = output_dir / "pressure_policy_matched_summary.txt"
    payload["files"]["json"] = str(json_path)
    payload["files"]["summary"] = str(summary_path)
    temporary = json_path.with_name(f".{json_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, json_path)
    coverage = payload["coverage_summary"]
    removal = payload["removal_summary"]
    summary_path.write_text(
        "Matched selector pressure-policy Pareto audit\n"
        "This is diagnostic only; no test policy is selected.\n"
        f"Coverage matching: {coverage['status']}, "
        f"matched={coverage['matched_row_count']}/{coverage['row_count']}\n"
        f"Removed-volume matching: {removal['status']}, "
        f"matched={removal['matched_row_count']}/{removal['row_count']}\n",
        encoding="utf-8",
    )
    return payload


def policies_from_selection(path: os.PathLike[str] | str) -> tuple[PressurePolicy, ...]:
    payload = json.loads(Path(path).expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    recommendations = payload.get("recommendations", {})
    policies = [PressurePolicy("base", 2.0, 0.0, 0.0)]
    seen = {"base"}
    for profile in ("aggressive", "balanced", "conservative"):
        value = recommendations.get(profile, {}).get("policy")
        if not isinstance(value, Mapping):
            raise ValueError(f"Policy selection lacks recommendation {profile!r}")
        name = str(value["name"])
        if name in seen:
            continue
        seen.add(name)
        policies.append(
            PressurePolicy(
                name=name,
                score_threshold=float(value["score_threshold"]),
                alpha=float(value["alpha"]),
                target_floor=float(value["target_floor"]),
            )
        )
    return tuple(policies)


def write_policy_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    selection_source: str | None,
    audit_config: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    csv_path = output_dir / "pressure_policy_sweep.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)

    if selection_source is None:
        pareto, recommendations = pareto_and_recommendations(rows)
    else:
        source_path = Path(selection_source).expanduser().resolve(strict=True)
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if source_payload.get("schema") != "tactile_selector_pressure_policy_v1":
            raise ValueError(f"Unsupported policy selection schema: {source_path}")
        pareto = [str(value) for value in source_payload.get("pareto_policies", ())]
        recommendations = dict(source_payload.get("recommendations", {}))
    for row in rows:
        row["pareto"] = row["name"] in pareto
    pareto_path = output_dir / "pressure_policy_pareto.csv"
    temporary = pareto_path.with_name(f".{pareto_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([row for row in rows if row["pareto"]])
    os.replace(temporary, pareto_path)

    payload = {
        "schema": "tactile_selector_pressure_policy_v1",
        "selection_scope": "validation_only" if selection_source is None else "fixed_from_validation",
        "selection_source": (
            None
            if selection_source is None
            else str(Path(selection_source).expanduser().resolve(strict=True))
        ),
        "audit_config": dict(audit_config),
        "pareto_policies": pareto,
        "recommendations": recommendations,
    }
    selection_path = output_dir / "policy_selection.json"
    temporary = selection_path.with_name(f".{selection_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, selection_path)
    return payload


def write_policy_control_replay(
    output_dir: Path,
    *,
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    reference_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    pair_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    reference_scopes: Mapping[str, str],
    policy_profiles: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Write fixed-policy replay results without reselecting any policy."""

    output_dir.mkdir(parents=True, exist_ok=True)
    flattened: list[dict[str, Any]] = []
    nested: dict[str, Any] = {}
    identity_keys = {"name", "score_threshold", "alpha", "target_floor"}
    for source, raw_rows in source_rows.items():
        rows = [dict(row) for row in raw_rows]
        references = {
            str(row["name"]): dict(row) for row in reference_rows[source]
        }
        pairs = {
            str(row["name"]): dict(row) for row in pair_rows.get(source, ())
        }
        scope = str(reference_scopes[source])
        source_payload_rows: list[dict[str, Any]] = []
        for row in rows:
            name = str(row["name"])
            if name not in references:
                raise ValueError(
                    f"Policy replay source {source!r} lacks reference policy {name!r}"
                )
            reference = references[name]
            combined = {
                "source": str(source),
                "reference_scope": scope,
                "selection_profiles": ",".join(
                    (policy_profiles or {}).get(name, ())
                ),
                **row,
            }
            for key, value in row.items():
                if key in identity_keys or key not in reference:
                    continue
                if isinstance(value, (int, float)) and isinstance(
                    reference[key], (int, float)
                ):
                    combined[f"gap_vs_aligned_reference_{key}"] = (
                        float(value) - float(reference[key])
                    )
            pair = pairs.get(name)
            if pair is not None:
                combined.update(
                    {
                        key: value
                        for key, value in pair.items()
                        if key not in identity_keys
                    }
                )
            flattened.append(combined)
            source_payload_rows.append(combined)
        nested[str(source)] = {
            "reference_scope": scope,
            "rows": source_payload_rows,
        }

    if not flattened:
        raise ValueError("Policy control replay did not receive any source rows")
    fieldnames: list[str] = []
    for row in flattened:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = output_dir / "pressure_policy_control_replay.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)
    os.replace(temporary, csv_path)

    payload = {
        "schema": "tactile_selector_pressure_control_replay_v1",
        "comparison_signs": {
            "gap_vs_aligned_reference": (
                "source minus matched aligned reference; metric-specific direction applies"
            ),
            "aligned_minus_control_strict_removed": "positive favors aligned prior",
            "aligned_minus_control_protected_removed": "negative favors aligned prior",
            "aligned_minus_control_added_under": "negative favors aligned prior",
            "aligned_minus_control_net_utility": "positive favors aligned prior",
            "aligned_minus_control_catastrophic_over_count": (
                "negative favors aligned prior"
            ),
        },
        "sources": nested,
        "csv": str(csv_path),
    }
    json_path = output_dir / "pressure_policy_control_replay.json"
    temporary = json_path.with_name(f".{json_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, json_path)
    return payload


def run_tiny_checks() -> None:
    records = [
        {"sample_uid": "a0", "sequence_key": "a", "query_alias": "q0", "frame_idx": 0},
        {"sample_uid": "a1", "sequence_key": "a", "query_alias": "q0", "frame_idx": 50},
        {"sample_uid": "a2", "sequence_key": "a", "query_alias": "q1", "frame_idx": 0},
        {"sample_uid": "b0", "sequence_key": "b", "query_alias": "q0", "frame_idx": 0},
        {"sample_uid": "b1", "sequence_key": "b", "query_alias": "q0", "frame_idx": 50},
    ]
    mappings, summaries = build_control_mappings(records, minimum_far_frame_gap=30)
    assert summaries["cross_sequence"]["same_sequence_fraction"] == 0.0
    assert mappings["same_sequence_far"][0] == 1
    assert mappings["wrong_query"][0] == 2

    policies = policy_grid((0.5, 0.9), (0.5, 1.0), (0.02, 0.05))
    accumulator = PressurePolicyAccumulator(policies, chunk_size=3)
    pred = torch.tensor([[0.2, 0.2, 0.05]])
    logits = torch.tensor([[-4.0, 4.0, -4.0]])
    target = torch.tensor([[0.0, 0.2, 0.0]])
    accumulator.update(pred, logits, target, torch.ones_like(pred), torch.ones(1))
    rows = accumulator.summaries()
    corrected_rows = [row for row in rows if row["name"] != "base"]
    assert len(rows) > accumulator.chunk_size
    assert any(row["strict_false_high_volume_removed"] > 0.0 for row in corrected_rows)
    assert all(row["protected_contact_volume_removed"] == 0.0 for row in corrected_rows)
    assert min(row["rmse"] for row in corrected_rows) < rows[0]["rmse"]

    direct_accumulator = PressurePolicyAccumulator(policies, chunk_size=3)
    direct_accumulator.update(
        pred,
        logits,
        target,
        torch.ones_like(pred),
        torch.ones(1),
        false_high_logits=torch.tensor([[-4.0, 4.0, -4.0]]),
    )
    direct_rows = [
        row for row in direct_accumulator.summaries() if row["name"] != "base"
    ]
    assert all(row["strict_false_high_volume_removed"] == 0.0 for row in direct_rows)
    assert any(row["protected_contact_volume_removed"] > 0.0 for row in direct_rows)

    selected_policies = (
        PressurePolicy("base", 2.0, 0.0, 0.0),
        PressurePolicy("balanced", 0.5, 0.5, 0.05),
    )
    pair = PressurePolicyPairAccumulator(
        selected_policies,
        chunk_size=1,
        bootstrap_iterations=20,
    )
    pair.update(
        pred,
        logits,
        torch.full_like(logits, 4.0),
        target,
        torch.ones_like(pred),
        torch.ones(1),
        sequence_keys=("tiny",),
    )
    pair_rows = pair.summaries()
    assert pair_rows[0]["action_jaccard"] == 1.0
    assert pair_rows[1]["aligned_minus_control_net_utility"] > 0.0
    assert pair_rows[1]["paired_sequence_count"] == 1

    rgb_rows = [dict(row) for row in rows]
    aligned_rows = [dict(row) for row in rows]
    for row in aligned_rows:
        if row["name"] == "base":
            continue
        row["strict_precision_all_selected"] += 0.01
        row["protected_removed_per_strict_removed"] = max(
            0.0, row["protected_removed_per_strict_removed"] - 0.01
        )
    coverage_matches = matched_policy_comparisons(
        aligned_rows, rgb_rows, axis="action_coverage"
    )
    assert coverage_matches
    assert all(row["correction_shape_locked"] for row in coverage_matches)
    assert all(
        row["aligned_alpha"] == row["rgb_alpha"]
        and row["aligned_target_floor"] == row["rgb_target_floor"]
        for row in coverage_matches
    )
    removal_matches = matched_policy_comparisons(
        aligned_rows,
        rgb_rows,
        axis="strict_false_high_volume_removed_fraction",
    )
    assert removal_matches
    with tempfile.TemporaryDirectory(prefix="selector-matched-tiny-") as directory:
        payload = write_matched_policy_outputs(
            Path(directory),
            aligned_rows=aligned_rows,
            rgb_rows=rgb_rows,
        )
        assert payload["scope"] == "diagnostic_only_no_policy_selection"
        assert Path(payload["files"]["json"]).is_file()
        assert Path(payload["files"]["summary"]).is_file()
