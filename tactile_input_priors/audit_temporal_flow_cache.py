#!/usr/bin/env python3
"""Cache-only audit for temporal tactile action spaces, selectors, and gradients."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from hamer_tactile_ft.losses import compute_tactile_loss
from hamer_tactile_ft.process_lifecycle import (
    configure_supervised_process,
    initialize_worker_parent_death_signal,
)
from tactile_input_priors.prior_metrics import PriorMetricAccumulator
from tactile_input_priors.runtime import (
    file_sha256,
    load_torch_checkpoint,
    tactile_loss_config_from_checkpoint,
)
from tactile_input_priors.temporal_flow import (
    TEMPORAL_MODEL_FORMAT,
    PartitionedPalmCache,
    QueryAwareTemporalResidual,
    build_temporal_pair_index,
    history_quality_context,
    pair_context,
    strict_history_control_pair_indices,
    strict_lag_history_indices,
    strict_lag_history_metadata,
    temporal_manifest_key,
)


configure_supervised_process()
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

SCHEMA = "tactile_temporal_cache_audit_v2"
DYNAMICS_NAMES = (
    "empty_stable",
    "spatially_stable",
    "source_loading",
    "sink_release",
    "transport_candidate",
    "large_spatial_change",
)
PARAMETER_GROUPS = (
    "global_gate",
    "input_mlp",
    "graph_blocks",
    "coefficient_head",
    "transition_head",
    "history_gate_head",
)


class ExactRankSampler(Sampler[int]):
    """Stride-shard a finite audit without padding or duplicate samples."""

    def __init__(self, length: int, rank: int, world_size: int):
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return max(
            0,
            (self.length - self.rank + self.world_size - 1) // self.world_size,
        )


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not result or result[0] <= 0:
        raise argparse.ArgumentTypeError("lags must be positive integers")
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or not all(math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("expected a non-empty list of finite floats")
    return tuple(dict.fromkeys(result))


def _parse_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return tuple(dict.fromkeys(result))


def _parse_lag_masks(value: str) -> tuple[tuple[int, ...], ...] | str:
    value = str(value).strip().lower()
    if value in {"auto", "full"}:
        return value
    masks = []
    for raw_mask in value.split(";"):
        mask = tuple(
            sorted(
                {
                    int(item.strip())
                    for item in raw_mask.replace(",", "+").split("+")
                    if item.strip()
                }
            )
        )
        if not mask or mask[0] <= 0:
            raise argparse.ArgumentTypeError(
                "lag masks must be 'auto' or semicolon-separated values such as "
                "'1;2;4;1+2;1+4;2+4;1+2+4'"
            )
        if mask not in masks:
            masks.append(mask)
    return tuple(masks)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


class MultiLagCacheDataset(Dataset):
    def __init__(
        self,
        cache_root: str,
        pair_index: str,
        lags: Sequence[int],
        *,
        max_open_shards: int,
        max_pairs: int = 0,
        seed: int = 521,
        include_cross_sequence_control: bool = False,
    ):
        self.cache = PartitionedPalmCache(cache_root, max_open_shards=max_open_shards)
        self.pair_index_path = Path(pair_index).expanduser().resolve(strict=True)
        with np.load(self.pair_index_path, allow_pickle=False) as payload:
            self.arrays = {name: np.asarray(payload[name]) for name in payload.files}
        self.lags = tuple(int(lag) for lag in lags)
        current = np.asarray(self.arrays["current_index"], dtype=np.int64)
        previous = np.asarray(self.arrays["previous_index"], dtype=np.int64)
        history_metadata = strict_lag_history_metadata(
            len(self.cache),
            current,
            previous,
            self.lags,
            time_gap=self.arrays["time_gap"],
            bbox_iou=self.arrays["bbox_iou"],
            bbox_center_jump=self.arrays["bbox_center_jump"],
            bbox_abs_log_area_ratio=self.arrays["bbox_abs_log_area_ratio"],
            contralateral_previous_indices=self.arrays.get(
                "contralateral_previous_index"
            ),
        )
        histories = history_metadata.pop("history_indices")
        if include_cross_sequence_control:
            controls = strict_history_control_pair_indices(
                self.arrays["sequence_key"],
                self.arrays["side"],
                self.arrays["pressure_bin"],
                histories >= 0,
                seed=int(seed),
            )
            control_histories = histories[controls]
        else:
            controls = np.arange(len(histories), dtype=np.int64)
            control_histories = histories
        if max_pairs > 0:
            current = current[: int(max_pairs)]
            histories = histories[: int(max_pairs)]
            controls = controls[: int(max_pairs)]
            control_histories = control_histories[: int(max_pairs)]
            history_metadata = {
                key: value[: int(max_pairs)]
                for key, value in history_metadata.items()
            }
            self.arrays = {key: value[: int(max_pairs)] for key, value in self.arrays.items()}
        self.current_indices = current
        self.history_indices = histories
        self.history_metadata = history_metadata
        self.control_pair_indices = controls
        self.control_history_indices = control_histories
        self.sequence_ids = np.unique(
            np.asarray(self.arrays["sequence_key"], dtype=np.str_),
            return_inverse=True,
        )[1].astype(np.int64)

    def __len__(self) -> int:
        return len(self.current_indices)

    @staticmethod
    def _tensor(value, dtype=None):
        result = torch.from_numpy(np.array(value, copy=True))
        return result if dtype is None else result.to(dtype=dtype)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        current_index = int(self.current_indices[index])
        current = self.cache.values(current_index)
        current_logits = self._tensor(current["palm_base_logits"])
        current_target = self._tensor(current["palm_tactile_signal"], torch.float32)
        history_logits = []
        history_targets = []
        availability = []
        for history_index in self.history_indices[index]:
            available = int(history_index) >= 0
            history = self.cache.values(int(history_index)) if available else current
            history_logits.append(self._tensor(history["palm_base_logits"]))
            history_targets.append(
                self._tensor(history["palm_tactile_signal"], torch.float32)
            )
            availability.append(float(available))
        control_logits = []
        control_availability = []
        for history_index in self.control_history_indices[index]:
            available = int(history_index) >= 0
            history = self.cache.values(int(history_index)) if available else current
            control_logits.append(self._tensor(history["palm_base_logits"]))
            control_availability.append(float(available))
        contralateral_logits = []
        contralateral_availability = []
        for history_index in self.history_metadata[
            "contralateral_history_indices"
        ][index]:
            available = int(history_index) >= 0
            history = self.cache.values(int(history_index)) if available else current
            contralateral_logits.append(self._tensor(history["palm_base_logits"]))
            contralateral_availability.append(float(available))
        return {
            "current_logits": current_logits,
            "history_logits": torch.stack(history_logits),
            "tactile_signal": current_target,
            "history_tactile_signal": torch.stack(history_targets),
            "history_available": torch.tensor(availability, dtype=torch.float32),
            "control_history_logits": torch.stack(control_logits),
            "control_history_available": torch.tensor(
                control_availability, dtype=torch.float32
            ),
            "contralateral_history_logits": torch.stack(contralateral_logits),
            "contralateral_history_available": torch.tensor(
                contralateral_availability, dtype=torch.float32
            ),
            "has_tactile": self._tensor(
                current["has_tactile"], torch.float32
            ).reshape(()),
            "time_gap": torch.tensor(float(self.arrays["time_gap"][index])),
            "bbox_iou": torch.tensor(float(self.arrays["bbox_iou"][index])),
            "bbox_center_jump": torch.tensor(
                float(self.arrays["bbox_center_jump"][index])
            ),
            "bbox_abs_log_area_ratio": torch.tensor(
                float(self.arrays["bbox_abs_log_area_ratio"][index])
            ),
            "history_time_gap": self._tensor(
                self.history_metadata["history_time_gap"][index], torch.float32
            ),
            "history_min_bbox_iou": self._tensor(
                self.history_metadata["history_min_bbox_iou"][index], torch.float32
            ),
            "history_max_bbox_center_jump": self._tensor(
                self.history_metadata["history_max_bbox_center_jump"][index],
                torch.float32,
            ),
            "history_max_bbox_abs_log_area_ratio": self._tensor(
                self.history_metadata["history_max_bbox_abs_log_area_ratio"][index],
                torch.float32,
            ),
            "sequence_id": torch.tensor(int(self.sequence_ids[index])),
            "side": torch.tensor(int(self.arrays["side"][index])),
        }


def _history_reference(history_logits: torch.Tensor, columns: Sequence[int]) -> torch.Tensor:
    selected = history_logits[:, tuple(columns)]
    return selected[:, 0] if selected.shape[1] == 1 else selected.mean(dim=1)


def blend_history(
    current_logits: torch.Tensor,
    history_logits: torch.Tensor,
    alpha: float,
    space: str,
    max_logit_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_logits = current_logits.float()
    history_logits = history_logits.float()
    alpha = float(alpha)
    if space == "probability":
        current = torch.sigmoid(current_logits)
        history = torch.sigmoid(history_logits)
        pred = (current + alpha * (history - current)).clamp(1e-6, 1.0 - 1e-6)
        return torch.logit(pred), pred
    raw_delta = alpha * (history_logits - current_logits)
    if space == "logit_bounded":
        raw_delta = float(max_logit_delta) * torch.tanh(
            raw_delta / float(max_logit_delta)
        )
    elif space != "logit_linear":
        raise ValueError(f"Unsupported blend space: {space}")
    logits = current_logits + raw_delta
    return logits, torch.sigmoid(logits)


def _dynamics_labels(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    previous = previous.float().clamp_min(0.0)
    current = current.float().clamp_min(0.0)
    previous_volume = previous.sum(dim=1)
    current_volume = current.sum(dim=1)
    denominator = torch.maximum(previous_volume, current_volume).clamp_min(1.0)
    signed_change = (current_volume - previous_volume) / denominator
    previous_dist = previous / previous_volume[:, None].clamp_min(1e-12)
    current_dist = current / current_volume[:, None].clamp_min(1e-12)
    intersection = torch.minimum(previous_dist, current_dist).sum(dim=1)
    union = torch.maximum(previous_dist, current_dist).sum(dim=1)
    distribution_viou = intersection / union.clamp_min(1e-12)
    labels = torch.full_like(previous_volume, 5, dtype=torch.long)
    both_empty = (previous_volume <= 1e-12) & (current_volume <= 1e-12)
    source = (~both_empty) & (
        (previous_volume <= 1e-12) | (signed_change > 0.25)
    )
    sink = (~both_empty) & (~source) & (
        (current_volume <= 1e-12) | (signed_change < -0.25)
    )
    stable = (~both_empty) & (~source) & (~sink) & (distribution_viou >= 0.75)
    transport = (
        (~both_empty)
        & (~source)
        & (~sink)
        & (~stable)
        & (distribution_viou >= 0.25)
    )
    labels[both_empty] = 0
    labels[stable] = 1
    labels[source] = 2
    labels[sink] = 3
    labels[transport] = 4
    return labels


class BinaryHistogram:
    def __init__(self, bins: int = 4096):
        self.bins = int(bins)
        self.positive = torch.zeros(self.bins, dtype=torch.float64)
        self.negative = torch.zeros(self.bins, dtype=torch.float64)
        self.score_sum = 0.0
        self.brier_sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> None:
        scores = scores.detach().float()[mask].clamp(0.0, 1.0)
        labels = labels.detach().bool()[mask]
        if not scores.numel():
            return
        indices = torch.clamp((scores * (self.bins - 1)).long(), 0, self.bins - 1)
        self.positive += torch.bincount(
            indices[labels], minlength=self.bins
        ).double().cpu()
        self.negative += torch.bincount(
            indices[~labels], minlength=self.bins
        ).double().cpu()
        self.score_sum += float(scores.double().sum().cpu())
        self.brier_sum += float((scores - labels.float()).square().double().sum().cpu())
        self.count += int(scores.numel())

    def summary(self) -> dict[str, float]:
        positive = self.positive.flip(0)
        negative = self.negative.flip(0)
        total_positive = float(positive.sum())
        total_negative = float(negative.sum())
        tp = positive.cumsum(0)
        fp = negative.cumsum(0)
        precision = tp / (tp + fp).clamp_min(1.0)
        ap = float((precision * positive).sum() / max(total_positive, 1.0))
        # Integrate the descending-score ROC staircase using histogram bins.
        tpr = tp / max(total_positive, 1.0)
        fpr = fp / max(total_negative, 1.0)
        auroc = float(torch.trapz(tpr, fpr)) if total_positive and total_negative else math.nan
        counts = self.positive + self.negative
        centers = (torch.arange(self.bins, dtype=torch.float64) + 0.5) / self.bins
        observed = self.positive / counts.clamp_min(1.0)
        ece = float((counts * (observed - centers).abs()).sum() / max(self.count, 1))
        return {
            "count": float(self.count),
            "positive_fraction": total_positive / max(self.count, 1),
            "average_precision_histogram": ap,
            "auroc_histogram": auroc,
            "brier": self.brier_sum / max(self.count, 1),
            "ece_histogram": ece,
            "mean_score": self.score_sum / max(self.count, 1),
        }

    def synchronize(self, device: torch.device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        counts = torch.stack((self.positive, self.negative)).to(device=device)
        scalars = torch.tensor(
            (self.score_sum, self.brier_sum, float(self.count)),
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(counts)
        dist.all_reduce(scalars)
        self.positive = counts[0].cpu()
        self.negative = counts[1].cpu()
        self.score_sum = float(scalars[0].cpu())
        self.brier_sum = float(scalars[1].cpu())
        self.count = int(scalars[2].cpu())


class CandidateAccumulator:
    def __init__(self):
        self.metrics = PriorMetricAccumulator()
        self.batch_loss_sum = 0.0
        self.batch_loss_count = 0
        self.component_sums: dict[str, float] = defaultdict(float)
        self.exact_component_sums: dict[str, float] = defaultdict(float)
        self.exact_component_counts: dict[str, int] = defaultdict(int)

    def update(
        self,
        logits: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        loss_config,
    ) -> None:
        palm = torch.ones_like(target)
        self.metrics.update(pred, target, palm, valid)
        selected = valid > 0.5
        if not bool(selected.any()):
            return
        loss, components = compute_tactile_loss(
            pred=pred[selected],
            logits=logits[selected],
            target=target[selected],
            palm_mask=torch.ones_like(target[selected]),
            valid_mask=torch.ones(int(selected.sum()), device=target.device),
            dataset_batch="TouchAnything",
            config=loss_config,
            current_epoch=None,
            ramp_override=1.0,
            # Candidate eligibility differs between exact rank shards. Inner
            # DDP collectives would therefore be called in a different order;
            # accumulate rank-local terms here and merge them in synchronize().
            distributed_reduce=False,
        )
        count = int(selected.sum())
        self.batch_loss_sum += float(loss.detach().cpu()) * count
        self.batch_loss_count += count
        for name, value in components.items():
            if value.numel() == 1:
                self.component_sums[name] += float(value.cpu()) * count

        selected_target = target[selected].float()
        point_count = int(selected_target.numel())
        background_count = int(
            (selected_target <= float(loss_config.background_pressure_thr)).sum()
        )
        location_volume = selected_target.clamp_min(0.0).sum(dim=1)
        location_peak = selected_target.clamp_min(0.0).amax(dim=1)
        location_count = int(
            (
                (location_volume >= float(loss_config.location_gt_volume_thr))
                & (location_peak >= float(loss_config.location_min_gt_peak))
            ).sum()
        )
        exact_terms = {
            "loss_weighted_tactile": ("points", point_count),
            "loss_background": ("background", background_count),
            "loss_location_weighted": ("location", location_count),
            "loss_contact_weighted": ("contact", count),
        }
        for component_name, (group, denominator) in exact_terms.items():
            value = float(components[component_name].cpu())
            self.exact_component_sums[group] += value * denominator
            self.exact_component_counts[group] += denominator

    def row(self) -> dict[str, float]:
        row = self.metrics.summary()
        exact = {
            group: self.exact_component_sums[group]
            / max(self.exact_component_counts[group], 1)
            for group in ("points", "background", "location", "contact")
        }
        row["v2_loss_full_ramp"] = sum(exact.values())
        row["v2_loss_batch_weighted"] = self.batch_loss_sum / max(
            self.batch_loss_count, 1
        )
        row.update(
            {
                "v2_weighted_tactile_exact": exact["points"],
                "v2_background_exact": exact["background"],
                "v2_location_exact": exact["location"],
                "v2_contact_exact": exact["contact"],
            }
        )
        for name, total in self.component_sums.items():
            row[name] = total / max(self.batch_loss_count, 1)
        return row

    def synchronize(self, device: torch.device) -> None:
        self.metrics.synchronize(device)
        if not (dist.is_available() and dist.is_initialized()):
            return
        payload = {
            "batch_loss_sum": self.batch_loss_sum,
            "batch_loss_count": self.batch_loss_count,
            "component_sums": dict(self.component_sums),
            "exact_component_sums": dict(self.exact_component_sums),
            "exact_component_counts": dict(self.exact_component_counts),
        }
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, payload)
        self.batch_loss_sum = sum(float(item["batch_loss_sum"]) for item in gathered)
        self.batch_loss_count = sum(int(item["batch_loss_count"]) for item in gathered)
        self.component_sums = defaultdict(float)
        self.exact_component_sums = defaultdict(float)
        self.exact_component_counts = defaultdict(int)
        for item in gathered:
            for name, value in item["component_sums"].items():
                self.component_sums[name] += float(value)
            for name, value in item["exact_component_sums"].items():
                self.exact_component_sums[name] += float(value)
            for name, value in item["exact_component_counts"].items():
                self.exact_component_counts[name] += int(value)


class DirectionAccumulator:
    FIELDS = (
        "eligible_frames",
        "actionable_points",
        "sign_matches",
        "frame_cosine_sum",
        "frame_cosine_count",
        "false_high_points",
        "false_high_down_direction",
        "high_under_points",
        "high_under_up_direction",
        "true_contact_points",
        "true_contact_harmful_down",
    )

    def __init__(self) -> None:
        self.values = torch.zeros(len(self.FIELDS), dtype=torch.float64)

    @torch.no_grad()
    def update(
        self,
        current_prediction: torch.Tensor,
        history_prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        *,
        threshold: float,
    ) -> None:
        current = current_prediction.float()
        history = history_prediction.float()
        target = target.float()
        valid = valid.bool()
        desired = target - current
        direction = history - current
        point_valid = valid[:, None]
        actionable = point_valid & (desired.abs() >= float(threshold))
        sign_match = actionable & (desired * direction > 0.0)
        dot = (desired * direction).sum(dim=1)
        denominator = (
            torch.linalg.vector_norm(desired, dim=1)
            * torch.linalg.vector_norm(direction, dim=1)
        )
        cosine_valid = valid & (denominator > 1e-12)
        cosine = dot / denominator.clamp_min(1e-12)
        false_high = point_valid & (target < 0.005) & (current >= 0.3)
        high_under = point_valid & (target >= 0.7) & (current < 0.3)
        true_contact = point_valid & (target >= 0.10) & (current >= 0.10)
        additions = torch.stack(
            (
                valid.sum(),
                actionable.sum(),
                sign_match.sum(),
                cosine[cosine_valid].sum(),
                cosine_valid.sum(),
                false_high.sum(),
                (false_high & (direction < -float(threshold))).sum(),
                high_under.sum(),
                (high_under & (direction > float(threshold))).sum(),
                true_contact.sum(),
                (true_contact & (direction < -float(threshold))).sum(),
            )
        ).double()
        if self.values.device != additions.device:
            self.values = self.values.to(additions.device)
        self.values += additions

    def synchronize(self, device: torch.device) -> None:
        self.values = self.values.to(device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.values)

    def row(self) -> dict[str, float]:
        raw = self.values.detach().cpu().tolist()
        value = {name: float(raw[index]) for index, name in enumerate(self.FIELDS)}
        return {
            **value,
            "history_direction_sign_match_fraction": value["sign_matches"]
            / max(value["actionable_points"], 1.0),
            "history_direction_frame_cosine": value["frame_cosine_sum"]
            / max(value["frame_cosine_count"], 1.0),
            "false_high_down_direction_fraction": value[
                "false_high_down_direction"
            ]
            / max(value["false_high_points"], 1.0),
            "high_under_up_direction_fraction": value["high_under_up_direction"]
            / max(value["high_under_points"], 1.0),
            "true_contact_harmful_down_fraction": value[
                "true_contact_harmful_down"
            ]
            / max(value["true_contact_points"], 1.0),
        }


def _synchronize_counter(counter: Counter) -> Counter:
    if not (dist.is_available() and dist.is_initialized()):
        return counter
    gathered: list[dict[Any, int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, dict(counter))
    merged = Counter()
    for values in gathered:
        if values is not None:
            merged.update(values)
    return merged


def _distributed_runtime(args) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-rank temporal audit requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, world_size, torch.device("cuda", local_rank)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world_size, device


def _parameter_group(name: str) -> str | None:
    if name == "global_rezero_gate":
        return "global_gate"
    for group in PARAMETER_GROUPS[1:]:
        if name.startswith(group + "."):
            return group
    return None


def _balanced_mean(values: torch.Tensor, labels: torch.Tensor, classes: Iterable[Any]):
    terms = [values[labels == value].mean() for value in classes if bool((labels == value).any())]
    return torch.stack(terms).mean() if terms else values.sum() * 0.0


def _loss_terms(model, batch, loss_config, transition_threshold, history_margin):
    audit_lags = tuple(int(value) for value in batch.pop("_audit_lags"))
    model_columns = tuple(audit_lags.index(lag) for lag in model.history_lags)
    model_history = batch["history_logits"][:, model_columns]
    model_available = batch["history_available"][:, model_columns]
    model_quality = None
    if model.use_per_lag_quality:
        model_quality = history_quality_context(
            batch, audit_lags, nominal_fps=model.nominal_fps
        )[:, model_columns]
    output = _temporal_forward(
        model,
        batch["current_logits"],
        model_history,
        pair_context(batch),
        model_available,
        model_quality,
    )
    target = batch["tactile_signal"].float()
    tactile, _ = compute_tactile_loss(
        pred=output["pred_tactile"],
        logits=output["pred_logits"],
        target=target,
        palm_mask=torch.ones_like(target),
        valid_mask=batch["has_tactile"],
        dataset_batch="TouchAnything",
        config=loss_config,
        current_epoch=None,
        ramp_override=1.0,
        # Gradient scenarios are partitioned across ranks and must never enter
        # the training-time conditional-mean collective.
        distributed_reduce=False,
    )
    anchors = output["anchor_local_indices"]
    target_anchor = target[:, anchors]
    lag1_audit_column = audit_lags.index(1)
    lag1_model_column = model.history_lags.index(1)
    previous_target = batch["history_tactile_signal"][
        :, lag1_audit_column, anchors
    ].float()
    target_delta = target_anchor - previous_target
    transition_target = torch.zeros_like(target_delta, dtype=torch.long)
    transition_target[target_delta > transition_threshold] = 1
    transition_target[target_delta < -transition_threshold] = 2
    transition_logits = output["anchor_transition_logits"]
    history_gate_logits = output["anchor_history_gate_logits"]
    if model.architecture == "signed_additive":
        transition_logits = transition_logits[..., lag1_model_column, :]
        history_gate_logits = history_gate_logits[..., lag1_model_column]
    transition_values = F.cross_entropy(
        transition_logits.reshape(-1, 3),
        transition_target.reshape(-1),
        reduction="none",
    )
    transition = _balanced_mean(
        transition_values, transition_target.reshape(-1), (0, 1, 2)
    )
    current_error = (output["base_pred_tactile"][:, anchors] - target_anchor).abs()
    previous_error = (
        torch.sigmoid(
            batch["history_logits"][:, lag1_audit_column, anchors].float()
        )
        - target_anchor
    ).abs()
    advantage = current_error - previous_error
    clear = advantage.abs() >= history_margin
    history_target = advantage > 0.0
    history_values = F.binary_cross_entropy_with_logits(
        history_gate_logits, history_target.float(), reduction="none"
    )
    history = (
        _balanced_mean(history_values[clear], history_target[clear], (False, True))
        if bool(clear.any())
        else history_values.sum() * 0.0
    )
    fused_error = (output["pred_tactile"] - target).square().mean(dim=1)
    base_error = (output["base_pred_tactile"] - target).square().mean(dim=1)
    base_guard = F.relu(fused_error - base_error.detach()).mean()
    delta_l1 = output["bounded_logit_delta"].abs().mean()
    return {
        "pressure": 10.0 * tactile,
        "transition": transition,
        "history": history,
        "base_guard": base_guard,
        "delta_l1": delta_l1,
        "total": 10.0 * tactile + 0.1 * transition + 0.1 * history + 0.1 * base_guard + 0.001 * delta_l1,
    }


def gradient_audit(
    base_model: QueryAwareTemporalResidual,
    loader: DataLoader,
    device: torch.device,
    loss_config,
    *,
    batches: int,
    forced_gates: Sequence[float],
    transition_threshold: float,
    history_margin: float,
    audit_lags: Sequence[int],
    job_rank: int = 0,
    job_count: int = 1,
) -> list[dict[str, Any]]:
    if int(batches) <= 0:
        return []
    scenarios = [("learned", None)]
    if base_model.architecture == "legacy_product":
        scenarios.extend((("zero", 0.0),))
        scenarios.extend(
            (f"forced_{value:g}", float(value)) for value in forced_gates
        )
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    terms = ("pressure", "transition", "history", "base_guard", "delta_l1", "total")
    jobs = [
        (scenario_name, term)
        for scenario_name, _ in scenarios
        for term in terms
    ]
    selected_jobs = {
        job for index, job in enumerate(jobs) if index % int(job_count) == int(job_rank)
    }
    selected_scenarios = {scenario for scenario, _ in selected_jobs}
    models = []
    for name, gate in scenarios:
        if name not in selected_scenarios:
            continue
        model = copy.deepcopy(base_model).to(device).eval()
        if gate is not None:
            if not -0.999 < gate < 0.999:
                raise ValueError("forced effective gates must lie in (-.999,.999)")
            model.global_rezero_gate.data.fill_(math.atanh(gate))
        models.append((name, model))
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= batches:
            break
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        for scenario, model in models:
            for term in terms:
                if (scenario, term) not in selected_jobs:
                    continue
                model.zero_grad(set_to_none=True)
                loss_batch = dict(batch)
                loss_batch["_audit_lags"] = tuple(audit_lags)
                objective = _loss_terms(
                    model,
                    loss_batch,
                    loss_config,
                    transition_threshold,
                    history_margin,
                )[term]
                objective.backward()
                for group in PARAMETER_GROUPS:
                    gradients = []
                    for parameter_name, parameter in model.named_parameters():
                        if _parameter_group(parameter_name) == group and parameter.grad is not None:
                            gradients.append(parameter.grad.detach().float().reshape(-1).cpu())
                    gradient = torch.cat(gradients) if gradients else torch.zeros(1)
                    key = (scenario, term, group)
                    record = records.setdefault(
                        key,
                        {"norms": [], "sum": torch.zeros_like(gradient), "sum_norms": 0.0, "gate": []},
                    )
                    if record["sum"].numel() != gradient.numel():
                        raise RuntimeError("gradient group shape changed during audit")
                    norm = float(torch.linalg.vector_norm(gradient))
                    record["norms"].append(norm)
                    record["sum"].add_(gradient)
                    record["sum_norms"] += norm
                    if group == "global_gate":
                        record["gate"].append(float(gradient[0]))
    rows = []
    for (scenario, term, group), record in sorted(records.items()):
        norms = np.asarray(record["norms"], dtype=np.float64)
        summed_norm = float(torch.linalg.vector_norm(record["sum"]))
        row = {
            "scenario": scenario,
            "loss_term": term,
            "parameter_group": group,
            "batch_count": len(norms),
            "grad_norm_mean": float(norms.mean()) if len(norms) else 0.0,
            "grad_norm_median": float(np.median(norms)) if len(norms) else 0.0,
            "grad_norm_p10": float(np.quantile(norms, 0.1)) if len(norms) else 0.0,
            "grad_norm_p90": float(np.quantile(norms, 0.9)) if len(norms) else 0.0,
            "vector_cancellation_ratio": summed_norm / max(record["sum_norms"], 1e-30),
        }
        if record["gate"]:
            gates = np.asarray(record["gate"], dtype=np.float64)
            row.update(
                {
                    "positive_gradient_fraction": float(np.mean(gates > 0.0)),
                    "negative_gradient_fraction": float(np.mean(gates < 0.0)),
                    "scalar_cancellation_ratio": float(abs(gates.sum()) / max(np.abs(gates).sum(), 1e-30)),
                }
            )
        rows.append(row)
    return rows


def _best_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    objectives = {
        "v2_loss_full_ramp": min,
        "rmse": min,
        "contact_iou": max,
        "volumetric_iou": max,
        "core_distribution_viou": max,
    }
    result = []
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["relation"]), str(row["space"]))].append(row)
    for (relation, space), candidates in sorted(groups.items()):
        candidates = [
            row for row in candidates if float(row.get("frame_count", 0.0)) > 0.0
        ]
        if not candidates:
            continue
        for objective, selector in objectives.items():
            selected = selector(candidates, key=lambda row: float(row[objective]))
            result.append(
                {
                    "relation": relation,
                    "space": space,
                    "objective": objective,
                    "alpha": selected["alpha"],
                    "value": selected[objective],
                }
            )
    return result


def _automatic_lag_masks(lags: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    lags = tuple(int(value) for value in lags)
    if len(lags) > 8:
        raise ValueError("Automatic lag-mask expansion is limited to eight lags")
    masks = []
    for encoded in range(1, 1 << len(lags)):
        masks.append(
            tuple(lag for column, lag in enumerate(lags) if encoded & (1 << column))
        )
    return tuple(masks)


def _lag_mask_name(mask: Sequence[int]) -> str:
    return "+".join(str(value) for value in mask)


def _temporal_forward(
    model: QueryAwareTemporalResidual,
    current_logits: torch.Tensor,
    history_logits: torch.Tensor,
    context: torch.Tensor,
    history_available: torch.Tensor,
    history_quality: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if model.architecture == "legacy_product":
        previous = torch.where(
            history_available[:, :1] > 0.5,
            history_logits[:, :1],
            current_logits[:, None],
        )
        return model(current_logits, previous[:, 0], context)
    return model(
        current_logits,
        history_logits,
        context,
        history_available=history_available,
        history_quality=history_quality,
    )


def _prefix_relations(lags: Sequence[int]) -> dict[str, tuple[int, ...]]:
    result: dict[str, tuple[int, ...]] = {
        f"lag{lag}": (column,) for column, lag in enumerate(lags)
    }
    for stop in range(2, len(lags) + 1):
        prefix = tuple(range(stop))
        result["mean" + "".join(str(lag) for lag in lags[:stop])] = prefix
    return result


def _incremental_rows(
    best_rows: Sequence[Mapping[str, Any]], lags: Sequence[int]
) -> list[dict[str, Any]]:
    prefixes = [f"lag{lags[0]}"] + [
        "mean" + "".join(str(lag) for lag in lags[:stop])
        for stop in range(2, len(lags) + 1)
    ]
    by_key = {
        (str(row["relation"]), str(row["space"]), str(row["objective"])): row
        for row in best_rows
    }
    lower_is_better = {"v2_loss_full_ramp", "rmse"}
    rows = []
    for previous, current in zip(prefixes, prefixes[1:]):
        added_lag = int(lags[prefixes.index(current)])
        for space in sorted({str(row["space"]) for row in best_rows}):
            for objective in sorted({str(row["objective"]) for row in best_rows}):
                before = by_key.get((previous, space, objective))
                after = by_key.get((current, space, objective))
                if before is None or after is None:
                    continue
                before_value = float(before["value"])
                after_value = float(after["value"])
                improvement = (
                    before_value - after_value
                    if objective in lower_is_better
                    else after_value - before_value
                )
                rows.append(
                    {
                        "from_relation": previous,
                        "to_relation": current,
                        "added_lag": added_lag,
                        "space": space,
                        "objective": objective,
                        "before_value": before_value,
                        "after_value": after_value,
                        "incremental_improvement": improvement,
                        "before_alpha": before["alpha"],
                        "after_alpha": after["alpha"],
                    }
                )
    return rows


def _lag_metadata_rows(
    dataset: MultiLagCacheDataset, split: str
) -> list[dict[str, Any]]:
    rows = []
    available = dataset.history_indices >= 0
    fields = (
        "history_time_gap",
        "history_min_bbox_iou",
        "history_max_bbox_center_jump",
        "history_max_bbox_abs_log_area_ratio",
    )
    for column, lag in enumerate(dataset.lags):
        mask = available[:, column]
        row: dict[str, Any] = {
            "split": split,
            "lag": int(lag),
            "strict_chain_count": int(mask.sum()),
            "strict_chain_fraction": float(mask.mean()),
            "contralateral_chain_count": int(
                np.sum(
                    dataset.history_metadata["contralateral_history_indices"][:, column]
                    >= 0
                )
            ),
        }
        for field in fields:
            values = np.asarray(dataset.history_metadata[field][:, column])[mask]
            prefix = field.removeprefix("history_")
            row[f"{prefix}_mean"] = float(values.mean()) if len(values) else math.nan
            for quantile, suffix in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99")):
                row[f"{prefix}_{suffix}"] = (
                    float(np.quantile(values, quantile)) if len(values) else math.nan
                )
        rows.append(row)
    return rows


def _rescale_vertex_gate(
    values: torch.Tensor,
    target_abs_mean: float,
    maximum: float,
    *,
    signed: bool,
) -> torch.Tensor:
    denominator = values.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)
    result = values * (float(target_abs_mean) / denominator)
    if signed:
        return result.clamp(-float(maximum), float(maximum))
    return result.clamp(0.0, float(maximum))


def _apply_vertex_gate(
    current_logits: torch.Tensor,
    previous_logits: torch.Tensor,
    vertex_gate: torch.Tensor,
    maximum_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_delta = vertex_gate.float() * (
        previous_logits.float() - current_logits.float()
    )
    bounded = float(maximum_delta) * torch.tanh(raw_delta / float(maximum_delta))
    logits = current_logits.float() + bounded
    return logits, torch.sigmoid(logits)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default="")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--query-manifests", required=True)
    parser.add_argument("--pair-index-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lags", type=_parse_ints, default=(1, 2, 4))
    parser.add_argument(
        "--alphas",
        type=_parse_floats,
        default=(-0.10, -0.025, 0.0, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.75),
    )
    parser.add_argument(
        "--spaces",
        type=_parse_strings,
        default=("probability", "logit_bounded"),
    )
    parser.add_argument(
        "--evaluation-subset",
        choices=("matched_all_lags", "available"),
        default="matched_all_lags",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--max-open-shards", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-logit-delta", type=float, default=0.50)
    parser.add_argument(
        "--model-lag-masks",
        type=_parse_lag_masks,
        default="full",
        help=(
            "Lag subsets for the trained checkpoint, for example "
            "'1;2;4;1+2;1+4;2+4;1+2+4'."
        ),
    )
    parser.add_argument(
        "--residual-scales",
        type=_parse_floats,
        default=(1.0,),
    )
    parser.add_argument(
        "--model-controls",
        type=_parse_strings,
        default=("real",),
    )
    parser.add_argument("--direction-threshold", type=float, default=0.01)
    parser.add_argument("--gradient-batches", type=int, default=16)
    parser.add_argument("--gradient-batch-size", type=int, default=64)
    parser.add_argument("--forced-gates", type=_parse_floats, default=(0.05, 0.10))
    parser.add_argument(
        "--gate-target-means",
        type=_parse_floats,
        default=(0.025, 0.05, 0.10, 0.20, 0.40),
    )
    parser.add_argument("--transition-threshold", type=float, default=0.02)
    parser.add_argument("--history-margin", type=float, default=0.002)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--tiny-check", action="store_true")
    return parser


def tiny_check() -> None:
    distributed_indices = [
        list(ExactRankSampler(29, rank, 8)) for rank in range(8)
    ]
    flattened = [index for indices in distributed_indices for index in indices]
    if sorted(flattened) != list(range(29)) or len(flattened) != len(set(flattened)):
        raise AssertionError("Exact rank sampler omitted or duplicated audit samples")
    current = np.asarray([1, 2, 3, 4], dtype=np.int64)
    previous = np.asarray([0, 1, 2, 3], dtype=np.int64)
    histories = strict_lag_history_indices(5, current, previous, (1, 2, 4))
    expected = np.asarray([[0, -1, -1], [1, 0, -1], [2, 1, -1], [3, 2, 0]])
    if not np.array_equal(histories, expected):
        raise AssertionError((histories, expected))
    metadata = strict_lag_history_metadata(
        5,
        current,
        previous,
        (1, 2, 4),
        time_gap=np.full(4, 0.04),
        bbox_iou=np.asarray((0.9, 0.8, 0.7, 0.6)),
        bbox_center_jump=np.asarray((0.1, 0.2, 0.3, 0.4)),
        bbox_abs_log_area_ratio=np.asarray((0.01, 0.02, 0.03, 0.04)),
        contralateral_previous_indices=np.asarray((0, 1, 2, 3)),
    )
    if not np.allclose(metadata["history_time_gap"][-1], (0.04, 0.08, 0.16)):
        raise AssertionError("Long-lag cumulative time is incorrect")
    if not np.allclose(
        metadata["history_min_bbox_iou"][-1], (0.6, 0.6, 0.6)
    ):
        raise AssertionError("Long-lag minimum bbox IoU is incorrect")
    if metadata["contralateral_history_indices"][-1].tolist() != [3, 2, 0]:
        raise AssertionError("Contralateral long-lag chain is incorrect")
    if _automatic_lag_masks((1, 2, 4)) != (
        (1,),
        (2,),
        (1, 2),
        (4,),
        (1, 4),
        (2, 4),
        (1, 2, 4),
    ):
        raise AssertionError("Automatic lag masks changed ordering")
    logits = torch.tensor([[0.0, 1.0]])
    history = torch.tensor([[1.0, -1.0]])
    for space in ("probability", "logit_linear", "logit_bounded"):
        fused, pred = blend_history(logits, history, 0.0, space, 0.5)
        if not torch.equal(fused, logits) or not torch.equal(pred, torch.sigmoid(logits)):
            raise AssertionError(f"alpha=0 is not identity for {space}")
    _, probability = blend_history(logits, history, 1.0, "probability", 0.5)
    if not torch.allclose(probability, torch.sigmoid(history), atol=1e-7, rtol=0.0):
        raise AssertionError("probability alpha=1 does not recover history")
    bounded, _ = blend_history(logits, history, 100.0, "logit_bounded", 0.5)
    if float((bounded - logits).abs().max()) > 0.500001:
        raise AssertionError("bounded logit blend exceeded its budget")
    print("Temporal cache audit tiny checks passed.")


def main() -> None:
    if "--tiny-check" in sys.argv[1:]:
        tiny_check()
        return
    args = build_parser().parse_args()
    rank, world_size, device = _distributed_runtime(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if 1 not in args.lags:
        raise ValueError("--lags must include 1 for selector and gradient diagnostics")
    allowed_spaces = {"probability", "logit_linear", "logit_bounded"}
    if not set(args.spaces).issubset(allowed_spaces):
        raise ValueError(f"spaces must be drawn from {sorted(allowed_spaces)}")
    if args.batch_size <= 0 or args.gradient_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.residual_scales):
        raise ValueError("--residual-scales must lie in [0,1]")
    allowed_controls = {"real", "cross_sequence", "contralateral", "reset"}
    if not set(args.model_controls).issubset(allowed_controls):
        raise ValueError(
            f"model controls must be drawn from {sorted(allowed_controls)}"
        )
    if args.direction_threshold <= 0.0:
        raise ValueError("--direction-threshold must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    payload = load_torch_checkpoint(args.checkpoint)
    if payload.get("format") != TEMPORAL_MODEL_FORMAT:
        raise ValueError(f"Unsupported temporal checkpoint: {payload.get('format')!r}")
    model = QueryAwareTemporalResidual(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    missing_model_lags = sorted(set(model.history_lags) - set(args.lags))
    if missing_model_lags:
        raise ValueError(
            "--lags must include every lag used by the checkpoint; missing "
            f"{missing_model_lags}"
        )
    model_lag_masks = (
        _automatic_lag_masks(model.history_lags)
        if args.model_lag_masks == "auto"
        else (
            (tuple(model.history_lags),)
            if args.model_lag_masks == "full"
            else tuple(args.model_lag_masks)
        )
    )
    invalid_masks = [
        mask for mask in model_lag_masks if not set(mask).issubset(model.history_lags)
    ]
    if invalid_masks:
        raise ValueError(
            f"Model lag masks {invalid_masks} use lags outside {model.history_lags}"
        )
    cache = PartitionedPalmCache(args.cache, max_open_shards=args.max_open_shards)
    if cache.base_checkpoint_sha256 != str(payload.get("base_checkpoint_sha256") or ""):
        raise RuntimeError("Audit cache was built from a different RGB baseline")
    manifests = tuple(
        str(Path(value.strip()).expanduser().resolve(strict=True))
        for value in args.query_manifests.split(",")
        if value.strip()
    )
    key = temporal_manifest_key(manifests)
    pair_path = (
        Path(args.pair_index_root).expanduser().resolve(strict=False)
        / f"{args.split}-{cache.config_sha256[:12]}-{key}.npz"
    )
    if rank == 0:
        build_temporal_pair_index(cache, manifests, pair_path, seed=args.seed)
    if world_size > 1:
        dist.barrier()
    if not pair_path.is_file():
        raise FileNotFoundError(f"Temporal pair index was not created: {pair_path}")
    dataset = MultiLagCacheDataset(
        args.cache,
        str(pair_path),
        args.lags,
        max_open_shards=args.max_open_shards,
        max_pairs=args.max_pairs,
        seed=args.seed,
        include_cross_sequence_control="cross_sequence" in args.model_controls,
    )
    if rank == 0:
        print(
            f"[temporal-cache-audit:{args.split}] initialized {len(dataset):,} pairs; "
            f"lags={args.lags}; ranks={world_size}",
            flush=True,
        )
    base_checkpoint = args.base_checkpoint or str(payload.get("base_checkpoint") or "")
    if not base_checkpoint or not Path(base_checkpoint).expanduser().is_file():
        raise FileNotFoundError(
            "The RGB base checkpoint is required for exact V2 loss; pass --base-checkpoint"
        )
    if file_sha256(base_checkpoint) != str(payload.get("base_checkpoint_sha256") or ""):
        raise RuntimeError("--base-checkpoint does not match the temporal checkpoint")
    loss_config = tactile_loss_config_from_checkpoint(
        load_torch_checkpoint(base_checkpoint), full_ramp=True
    )
    sampler = ExactRankSampler(len(dataset), rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        worker_init_fn=initialize_worker_parent_death_signal if args.num_workers > 0 else None,
    )
    lag_columns = {lag: args.lags.index(lag) for lag in args.lags}
    model_columns = tuple(lag_columns[lag] for lag in model.history_lags)
    model_lag1_column = model.history_lags.index(1)
    relations = _prefix_relations(args.lags)
    candidates = {
        (relation, space, alpha): CandidateAccumulator()
        for relation in relations
        for space in args.spaces
        for alpha in args.alphas
    }
    gate_variants = {}
    if model.architecture == "legacy_product":
        gate_variants["learned_product"] = CandidateAccumulator()
        for target_mean in args.gate_target_means:
            for variant in (
                "product_rescaled",
                "history_only",
                "stable_only",
                "residual_history",
            ):
                gate_variants[f"{variant}_mean{target_mean:g}"] = CandidateAccumulator()
    full_model_mask = tuple(model.history_lags)
    trained_candidates = {
        (control, _lag_mask_name(mask), scale): CandidateAccumulator()
        for control in args.model_controls
        for mask in (
            model_lag_masks if control == "real" else (full_model_mask,)
        )
        for scale in args.residual_scales
    }
    oracle_metrics = {relation: PriorMetricAccumulator() for relation in relations}
    oracle_decisions: dict[str, Counter] = {relation: Counter() for relation in relations}
    selector_histogram = BinaryHistogram()
    selector_anchor_total = 0
    selector_clear_total = 0
    transition_confusion = torch.zeros((3, 3), dtype=torch.float64)
    dynamics_counts: dict[str, Counter] = {relation: Counter() for relation in relations}
    direction_metrics = {relation: DirectionAccumulator() for relation in relations}
    coverage = np.mean(dataset.history_indices >= 0, axis=0)
    common_coverage = float(np.mean(np.all(dataset.history_indices >= 0, axis=1)))
    persistent = Counter()
    started = time.time()
    if rank == 0:
        print(
            f"[temporal-cache-audit:{args.split}] starting candidate pass",
            flush=True,
        )
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            batch = {
                name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for name, value in raw_batch.items()
            }
            current_logits = batch["current_logits"].float()
            history_logits = batch["history_logits"].float()
            target = batch["tactile_signal"].float()
            history_target = batch["history_tactile_signal"].float()
            available = batch["history_available"] > 0.5
            common = available.all(dim=1)
            base_pred = torch.sigmoid(current_logits)
            current_context = pair_context(batch)
            checkpoint_history = history_logits[:, model_columns]
            checkpoint_available = available[:, model_columns].float()
            checkpoint_common = checkpoint_available.bool().all(dim=1)
            checkpoint_output = _temporal_forward(
                model,
                current_logits,
                checkpoint_history,
                current_context,
                checkpoint_available,
                history_quality_context(
                    batch, args.lags, nominal_fps=model.nominal_fps
                )[:, model_columns] if model.use_per_lag_quality else None,
            )
            anchors = checkpoint_output["anchor_local_indices"]
            current_error = (base_pred[:, anchors] - target[:, anchors]).abs()
            lag1_error = (
                torch.sigmoid(history_logits[:, lag_columns[1], anchors])
                - target[:, anchors]
            ).abs()
            advantage = current_error - lag1_error
            clear = advantage.abs() >= args.history_margin
            selector_anchor_total += int(clear.numel())
            selector_clear_total += int(clear.sum())
            checkpoint_history_score = checkpoint_output[
                "anchor_history_probability"
            ]
            checkpoint_transition = checkpoint_output["anchor_transition_logits"]
            if model.architecture == "signed_additive":
                checkpoint_history_score = checkpoint_history_score[
                    ..., model_lag1_column
                ]
                checkpoint_transition = checkpoint_transition[
                    ..., model_lag1_column, :
                ]
            selector_histogram.update(
                checkpoint_history_score, advantage > 0.0, clear
            )
            target_delta = target[:, anchors] - history_target[:, lag_columns[1], anchors]
            transition_target = torch.zeros_like(target_delta, dtype=torch.long)
            transition_target[target_delta > args.transition_threshold] = 1
            transition_target[target_delta < -args.transition_threshold] = 2
            transition_prediction = checkpoint_transition.argmax(dim=-1)
            encoded = transition_target.reshape(-1) * 3 + transition_prediction.reshape(-1)
            transition_confusion += torch.bincount(encoded, minlength=9).reshape(3, 3).double().cpu()

            lag1_valid = (available[:, lag_columns[1]] & (batch["has_tactile"] > 0.5)).float()
            if model.architecture == "legacy_product":
                gate_variants["learned_product"].update(
                    checkpoint_output["pred_logits"],
                    checkpoint_output["pred_tactile"],
                    target,
                    lag1_valid,
                    loss_config,
                )
                history_vertex = model._interpolate(checkpoint_history_score)
                stable_vertex = model._interpolate(
                    checkpoint_transition.softmax(dim=-1)[..., 0]
                )
                product_vertex = checkpoint_output["vertex_history_alpha"]
                previous_logits = history_logits[:, lag_columns[1]]
                for target_mean in args.gate_target_means:
                    rescaled_product = _rescale_vertex_gate(
                        product_vertex,
                        target_mean,
                        model.max_history_alpha,
                        signed=True,
                    )
                    history_only = _rescale_vertex_gate(
                        history_vertex,
                        target_mean,
                        model.max_history_alpha,
                        signed=False,
                    )
                    stable_only = _rescale_vertex_gate(
                        stable_vertex,
                        target_mean,
                        model.max_history_alpha,
                        signed=False,
                    )
                    residual_history = (
                        float(target_mean)
                        + float(target_mean) * (2.0 * history_vertex - 1.0)
                    ).clamp(0.0, model.max_history_alpha)
                    for variant, vertex_gate in (
                        ("product_rescaled", rescaled_product),
                        ("history_only", history_only),
                        ("stable_only", stable_only),
                        ("residual_history", residual_history),
                    ):
                        gate_logits, gate_pred = _apply_vertex_gate(
                            current_logits,
                            previous_logits,
                            vertex_gate,
                            model.max_logit_delta,
                        )
                        gate_variants[f"{variant}_mean{target_mean:g}"].update(
                            gate_logits,
                            gate_pred,
                            target,
                            lag1_valid,
                            loss_config,
                        )

            control_sources = {
                "real": (history_logits, available),
                "cross_sequence": (
                    batch["control_history_logits"].float(),
                    batch["control_history_available"] > 0.5,
                ),
                "contralateral": (
                    batch["contralateral_history_logits"].float(),
                    batch["contralateral_history_available"] > 0.5,
                ),
                "reset": (history_logits, torch.zeros_like(available)),
            }
            quality_sources = {}
            if model.use_per_lag_quality:
                real_quality = history_quality_context(
                    batch, args.lags, nominal_fps=model.nominal_fps
                )
                quality_sources = {
                    "real": real_quality,
                    "cross_sequence": history_quality_context(
                        batch,
                        args.lags,
                        prefix="control_",
                        nominal_fps=model.nominal_fps,
                    ),
                    "contralateral": history_quality_context(
                        batch,
                        args.lags,
                        availability=batch["contralateral_history_available"],
                        nominal_fps=model.nominal_fps,
                    ),
                    "reset": real_quality,
                }
            for control in args.model_controls:
                control_history_all, control_available_all = control_sources[control]
                model_history = control_history_all[:, model_columns]
                model_available = control_available_all[:, model_columns]
                model_quality = (
                    quality_sources[control][:, model_columns]
                    if model.use_per_lag_quality
                    else None
                )
                control_checkpoint_common = model_available.all(dim=1)
                control_masks = (
                    model_lag_masks if control == "real" else (full_model_mask,)
                )
                for mask in control_masks:
                    enabled = torch.tensor(
                        [lag in mask for lag in model.history_lags],
                        device=device,
                        dtype=torch.bool,
                    )[None]
                    masked_available = model_available & enabled
                    output = _temporal_forward(
                        model,
                        current_logits,
                        model_history,
                        current_context,
                        masked_available.float(),
                        model_quality,
                    )
                    if control == "reset" and not torch.equal(
                        output["pred_logits"], current_logits.float()
                    ):
                        maximum_reset_error = float(
                            (output["pred_logits"] - current_logits.float())
                            .abs()
                            .max()
                            .cpu()
                        )
                        raise RuntimeError(
                            "Temporal reset is not the exact RGB identity; "
                            f"max logit error={maximum_reset_error:.3e}"
                        )
                    selected_available = (
                        masked_available[:, enabled[0]].all(dim=1)
                        if bool(enabled.any())
                        else torch.zeros(len(target), device=device, dtype=torch.bool)
                    )
                    valid = (
                        checkpoint_common
                        if args.evaluation_subset == "matched_all_lags"
                        else selected_available
                    )
                    if control in {"cross_sequence", "contralateral"}:
                        valid = valid & (
                            control_checkpoint_common
                            if args.evaluation_subset == "matched_all_lags"
                            else selected_available
                        )
                    valid = valid & (batch["has_tactile"] > 0.5)
                    for scale in args.residual_scales:
                        scaled_logits = current_logits + float(scale) * output[
                            "bounded_logit_delta"
                        ].float()
                        trained_candidates[
                            (control, _lag_mask_name(mask), scale)
                        ].update(
                            scaled_logits,
                            torch.sigmoid(scaled_logits),
                            target,
                            valid.float(),
                            loss_config,
                        )

            if bool(common.any()):
                base_false_high = (target < 0.005) & (base_pred >= 0.3)
                history_pred = torch.sigmoid(history_logits)
                direction = (history_pred - base_pred[:, None]).abs().amax(dim=1)
                persistent["false_high_total"] += int(base_false_high[common].sum())
                persistent["false_high_no_history_direction"] += int(
                    (base_false_high & (direction < 0.01))[common].sum()
                )
                base_high_under = (target >= 0.7) & (base_pred < 0.3)
                persistent["high_under_total"] += int(base_high_under[common].sum())
                persistent["high_under_no_history_direction"] += int(
                    (base_high_under & (direction < 0.01))[common].sum()
                )

            for relation, columns in relations.items():
                relation_available = available[:, tuple(columns)].all(dim=1)
                valid = common if args.evaluation_subset == "matched_all_lags" else relation_available
                valid = valid & (batch["has_tactile"] > 0.5)
                if not bool(valid.any()):
                    continue
                reference_logits = _history_reference(history_logits, columns)
                reference_target = _history_reference(history_target, columns)
                direction_metrics[relation].update(
                    base_pred,
                    torch.sigmoid(reference_logits),
                    target,
                    valid,
                    threshold=args.direction_threshold,
                )
                labels = _dynamics_labels(reference_target[valid], target[valid])
                dynamics_counts[relation].update(
                    DYNAMICS_NAMES[int(value)] for value in labels.cpu().tolist()
                )
                best_error = (base_pred - target).square().mean(dim=1)
                best_prediction = base_pred.clone()
                best_choice = torch.full(
                    (len(target),), -1, dtype=torch.long, device=device
                )
                choice_index = 0
                for space in args.spaces:
                    for alpha in args.alphas:
                        logits, pred = blend_history(
                            current_logits,
                            reference_logits,
                            alpha,
                            space,
                            args.max_logit_delta,
                        )
                        candidates[(relation, space, alpha)].update(
                            logits, pred, target, valid.float(), loss_config
                        )
                        error = (pred - target).square().mean(dim=1)
                        improve = valid & (error < best_error)
                        best_error = torch.where(improve, error, best_error)
                        best_prediction = torch.where(improve[:, None], pred, best_prediction)
                        best_choice = torch.where(
                            improve, torch.full_like(best_choice, choice_index), best_choice
                        )
                        choice_index += 1
                oracle_metrics[relation].update(
                    best_prediction, target, torch.ones_like(target), valid.float()
                )
                choices, counts = torch.unique(best_choice[valid], return_counts=True)
                oracle_decisions[relation].update(
                    {int(choice): int(count) for choice, count in zip(choices.cpu(), counts.cpu())}
                )
            if (
                rank == 0
                and args.progress_every
                and (batch_index + 1) % args.progress_every == 0
            ):
                local_processed = min(
                    (batch_index + 1) * args.batch_size, len(sampler)
                )
                estimated_global = min(local_processed * world_size, len(dataset))
                print(
                    f"[temporal-cache-audit:{args.split}] ~{estimated_global:,}/"
                    f"{len(dataset):,} global pairs "
                    f"({estimated_global / max(time.time() - started, 1e-9):,.1f} pairs/s; "
                    f"{world_size} rank(s))",
                    flush=True,
                )

    if rank == 0:
        print(
            f"[temporal-cache-audit:{args.split}] candidate pass complete; "
            "synchronizing rank-local accumulators",
            flush=True,
        )
    for key in sorted(candidates):
        candidates[key].synchronize(device)
    for key in sorted(gate_variants):
        gate_variants[key].synchronize(device)
    for key in sorted(trained_candidates):
        trained_candidates[key].synchronize(device)
    for accumulator in direction_metrics.values():
        accumulator.synchronize(device)
    for accumulator in oracle_metrics.values():
        accumulator.synchronize(device)
    selector_histogram.synchronize(device)
    if world_size > 1:
        transition_confusion = transition_confusion.to(device=device)
        dist.all_reduce(transition_confusion)
        transition_confusion = transition_confusion.cpu()
        selector_totals = torch.tensor(
            (selector_anchor_total, selector_clear_total),
            device=device,
            dtype=torch.int64,
        )
        dist.all_reduce(selector_totals)
        selector_anchor_total, selector_clear_total = (
            int(value) for value in selector_totals.cpu().tolist()
        )
        persistent = _synchronize_counter(persistent)
        dynamics_counts = {
            relation: _synchronize_counter(dynamics_counts[relation])
            for relation in relations
        }
        oracle_decisions = {
            relation: _synchronize_counter(oracle_decisions[relation])
            for relation in relations
        }
    if rank == 0:
        print(
            f"[temporal-cache-audit:{args.split}] distributed synchronization complete; "
            "writing reports",
            flush=True,
        )

    alpha_rows = []
    for (relation, space, alpha), accumulator in sorted(candidates.items()):
        alpha_rows.append(
            {
                "split": args.split,
                "evaluation_subset": args.evaluation_subset,
                "relation": relation,
                "space": space,
                "alpha": alpha,
                **accumulator.row(),
            }
        )
    best_rows = _best_rows(alpha_rows)
    incremental_rows = _incremental_rows(best_rows, args.lags)
    trained_rows = [
        {
            "split": args.split,
            "control": control,
            "lag_mask": lag_mask,
            "residual_scale": scale,
            **accumulator.row(),
        }
        for (control, lag_mask, scale), accumulator in sorted(
            trained_candidates.items()
        )
    ]
    trained_best_rows = _best_rows(
        [
            {
                **row,
                "relation": row["control"],
                "space": row["lag_mask"],
                "alpha": row["residual_scale"],
            }
            for row in trained_rows
        ]
    )
    for row in trained_best_rows:
        row["control"] = row.pop("relation")
        row["lag_mask"] = row.pop("space")
        row["residual_scale"] = row.pop("alpha")
    direction_rows = [
        {
            "split": args.split,
            "relation": relation,
            "direction_threshold": args.direction_threshold,
            **accumulator.row(),
        }
        for relation, accumulator in sorted(direction_metrics.items())
    ]
    lag_metadata_rows = _lag_metadata_rows(dataset, args.split)
    dynamics_rows = []
    for relation, counts in sorted(dynamics_counts.items()):
        total = sum(int(value) for value in counts.values())
        for dynamics in DYNAMICS_NAMES:
            count = int(counts[dynamics])
            dynamics_rows.append(
                {
                    "split": args.split,
                    "relation": relation,
                    "dynamics": dynamics,
                    "count": count,
                    "fraction": count / max(total, 1),
                }
            )
    oracle_rows = []
    for relation, accumulator in oracle_metrics.items():
        oracle_rows.append(
            {
                "split": args.split,
                "relation": relation,
                "selection": "per_frame_rmse_oracle_over_spaces_and_alphas",
                **accumulator.summary(),
                "decision_counts": json.dumps(dict(oracle_decisions[relation]), sort_keys=True),
            }
        )
    transition_rows = []
    for class_index, class_name in enumerate(("stable", "source", "sink")):
        tp = float(transition_confusion[class_index, class_index])
        fp = float(transition_confusion[:, class_index].sum() - tp)
        fn = float(transition_confusion[class_index].sum() - tp)
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        transition_rows.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "support": float(transition_confusion[class_index].sum()),
            }
        )
    selector_summary = {
        "history_selector": selector_histogram.summary(),
        "history_clear_margin": args.history_margin,
        "history_clear_fraction": selector_clear_total / max(selector_anchor_total, 1),
        "transition_confusion": transition_confusion.tolist(),
        "transition_per_class": transition_rows,
        "transition_macro_f1": float(np.mean([row["f1"] for row in transition_rows])),
    }
    gradient_loader = DataLoader(
        dataset,
        batch_size=args.gradient_batch_size,
        shuffle=False,
        num_workers=0,
    )
    local_gradient_rows = gradient_audit(
        model,
        gradient_loader,
        device,
        loss_config,
        batches=args.gradient_batches,
        forced_gates=args.forced_gates,
        transition_threshold=args.transition_threshold,
        history_margin=args.history_margin,
        audit_lags=args.lags,
        job_rank=rank,
        job_count=world_size,
    )
    if world_size > 1:
        gathered_gradient_rows: list[list[dict[str, Any]] | None] = [
            None
        ] * world_size
        dist.all_gather_object(gathered_gradient_rows, local_gradient_rows)
        gradient_rows = sorted(
            (
                row
                for rank_rows in gathered_gradient_rows
                if rank_rows is not None
                for row in rank_rows
            ),
            key=lambda row: (
                str(row["scenario"]),
                str(row["loss_term"]),
                str(row["parameter_group"]),
            ),
        )
    else:
        gradient_rows = local_gradient_rows
    coverage_rows = [
        {
            "split": args.split,
            "lag": lag,
            "strict_chain_count": int(np.sum(dataset.history_indices[:, column] >= 0)),
            "strict_chain_fraction_of_lag1_pairs": float(coverage[column]),
        }
        for column, lag in enumerate(args.lags)
    ]
    coverage_rows.append(
        {
            "split": args.split,
            "lag": "matched_all",
            "strict_chain_count": int(np.sum(np.all(dataset.history_indices >= 0, axis=1))),
            "strict_chain_fraction_of_lag1_pairs": common_coverage,
        }
    )
    persistent_summary = {
        **dict(persistent),
        "false_high_no_history_direction_fraction": persistent["false_high_no_history_direction"]
        / max(persistent["false_high_total"], 1),
        "high_under_no_history_direction_fraction": persistent["high_under_no_history_direction"]
        / max(persistent["high_under_total"], 1),
        "history_direction_threshold": 0.01,
    }
    config = {
        "schema": SCHEMA,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve(strict=True)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "base_checkpoint": str(Path(base_checkpoint).expanduser().resolve(strict=True)),
        "cache": str(Path(args.cache).expanduser().resolve(strict=True)),
        "cache_config_sha256": cache.config_sha256,
        "pair_index": str(pair_path),
        "split": args.split,
        "lags": list(args.lags),
        "checkpoint_history_lags": list(model.history_lags),
        "alphas": list(args.alphas),
        "spaces": list(args.spaces),
        "evaluation_subset": args.evaluation_subset,
        "trained_model_evaluation_subset": (
            "matched_checkpoint_lags"
            if args.evaluation_subset == "matched_all_lags"
            else "selected_mask_available"
        ),
        "model_lag_masks": [list(mask) for mask in model_lag_masks],
        "residual_scales": list(args.residual_scales),
        "model_controls": list(args.model_controls),
        "sample_count": len(dataset),
        "elapsed_seconds": time.time() - started,
        "distributed_world_size": world_size,
        "sampling": "exact_stride_without_padding",
    }
    summary = {
        **config,
        "coverage": coverage_rows,
        "best_fixed_candidates": best_rows,
        "conditional_incremental_gain": incremental_rows,
        "trained_model_sweep_path": str(output_dir / "trained_model_sweep.csv"),
        "trained_model_best": trained_best_rows,
        "history_direction_path": str(output_dir / "history_direction.csv"),
        "lag_metadata_path": str(output_dir / "lag_metadata.csv"),
        "dynamics_by_horizon_path": str(output_dir / "dynamics_by_horizon.csv"),
        "oracle": oracle_rows,
        "selector": selector_summary,
        "dynamics_counts": {
            relation: dict(counts) for relation, counts in dynamics_counts.items()
        },
        "persistent_error_action_space": persistent_summary,
        "gate_algebra_path": str(output_dir / "gate_algebra_ablation.csv"),
        "gradient_audit_path": str(output_dir / "gradient_cancellation.csv"),
    }
    gate_rows = [
        {"split": args.split, "variant": name, **accumulator.row()}
        for name, accumulator in sorted(gate_variants.items())
    ]
    if rank != 0:
        dist.barrier()
        dist.destroy_process_group()
        return
    _write_csv(output_dir / "alpha_sweep.csv", alpha_rows)
    _write_csv(output_dir / "best_fixed_candidates.csv", best_rows)
    _write_csv(output_dir / "conditional_incremental_gain.csv", incremental_rows)
    _write_csv(output_dir / "trained_model_sweep.csv", trained_rows)
    _write_csv(output_dir / "trained_model_best.csv", trained_best_rows)
    _write_csv(output_dir / "history_direction.csv", direction_rows)
    _write_csv(output_dir / "lag_metadata.csv", lag_metadata_rows)
    _write_csv(output_dir / "dynamics_by_horizon.csv", dynamics_rows)
    _write_csv(output_dir / "oracle_selector.csv", oracle_rows)
    _write_csv(output_dir / "pair_coverage.csv", coverage_rows)
    _write_csv(output_dir / "gate_algebra_ablation.csv", gate_rows)
    _write_csv(output_dir / "gradient_cancellation.csv", gradient_rows)
    _write_json(output_dir / "selector_diagnostics.json", selector_summary)
    _write_json(output_dir / "summary.json", summary)
    lines = [
        f"Temporal cache audit: split={args.split}",
        f"Lag-1 pairs: {len(dataset):,}",
        f"Matched lag {','.join(map(str, args.lags))}: {common_coverage:.2%}",
        "Best fixed candidates:",
    ]
    for row in best_rows:
        lines.append(
            f"  {row['relation']}/{row['space']}/{row['objective']}: "
            f"alpha={float(row['alpha']):g} value={float(row['value']):.8f}"
        )
    lines.extend(
        (
            f"History selector AP: {selector_summary['history_selector']['average_precision_histogram']:.6f}",
            f"Transition macro-F1: {selector_summary['transition_macro_f1']:.6f}",
            "Persistent false-high without history direction: "
            f"{persistent_summary['false_high_no_history_direction_fraction']:.2%}",
            "Persistent high-under without history direction: "
            f"{persistent_summary['high_under_no_history_direction_fraction']:.2%}",
        )
    )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
