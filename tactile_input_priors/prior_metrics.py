"""Compact DDP-safe metrics for prior adapter training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist


METRIC_FIELDS = (
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
)

# Retain the private name for callers that may have imported it before the
# per-sequence bootstrap API was added.
_FIELDS = METRIC_FIELDS


@torch.no_grad()
def metric_contributions(
    pred: torch.Tensor,
    target: torch.Tensor,
    palm_mask: torch.Tensor,
    has_tactile: torch.Tensor,
    *,
    contact_threshold: float = 0.10,
) -> torch.Tensor:
    """Return additive per-frame sufficient statistics for all prior metrics."""

    pred = pred.detach().float()
    target = target.detach().float().to(pred.device)
    valid = (
        has_tactile.detach().reshape(-1).to(device=pred.device, dtype=pred.dtype)
        > 0.5
    ).to(pred.dtype)
    palm = palm_mask.detach().to(pred.device) > 0.5
    if palm.ndim == 1:
        palm = palm.unsqueeze(0).expand_as(pred)
    elif palm.shape != pred.shape:
        palm = palm.expand_as(pred)
    palm_float = palm.to(pred.dtype)
    valid_points = palm_float * valid[:, None]
    pred_palm = pred * palm_float
    target_palm = target * palm_float
    difference = (pred - target) * palm_float
    pred_volume = pred_palm.sum(dim=1)
    gt_volume = target_palm.sum(dim=1)
    pred_contact = (pred >= float(contact_threshold)) & palm
    gt_contact = (target >= float(contact_threshold)) & palm
    intersection = (pred_contact & gt_contact).sum(dim=1).double()
    union = (pred_contact | gt_contact).sum(dim=1).double()
    contact_iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
    vol_intersection = torch.minimum(pred_palm, target_palm).sum(dim=1).double()
    vol_union = torch.maximum(pred_palm, target_palm).sum(dim=1).double()
    viou = torch.where(
        vol_union > 1e-12,
        vol_intersection / vol_union.clamp_min(1e-12),
        1.0,
    )
    eligible = (valid > 0.5) & (gt_volume >= 1.0) & (
        target_palm.amax(dim=1) >= 0.05
    )
    pred_core = pred_palm.square()
    gt_core = target_palm.square()
    pred_dist = pred_core / pred_core.sum(dim=1, keepdim=True).clamp_min(1e-12)
    gt_dist = gt_core / gt_core.sum(dim=1, keepdim=True).clamp_min(1e-12)
    core_intersection = torch.minimum(pred_dist, gt_dist).sum(dim=1)
    core_union = torch.maximum(pred_dist, gt_dist).sum(dim=1)
    core_viou = core_intersection / core_union.clamp_min(1e-12)
    false_high = (
        (target < 0.005)
        & (pred >= 0.3)
        & palm
        & (valid[:, None] > 0.5)
    )
    temporal_correct = (
        pred_contact.any(dim=1) == gt_contact.any(dim=1)
    ).to(pred.dtype)
    return torch.stack(
        (
            valid,
            valid_points.sum(dim=1),
            (difference.abs() * valid_points).sum(dim=1),
            (difference.square() * valid_points).sum(dim=1),
            contact_iou * valid.double(),
            viou * valid.double(),
            pred_volume * valid,
            gt_volume * valid,
            temporal_correct * valid,
            core_viou * eligible.to(pred.dtype),
            eligible.to(pred.dtype),
            ((pred - target).clamp_min(0.0) * false_high).sum(dim=1),
            ((gt_volume < 10.0) & (pred_volume > 300.0)).to(pred.dtype) * valid,
            (gt_volume < 10.0).to(pred.dtype) * valid,
            ((gt_volume >= 150.0) & (pred_volume < 50.0)).to(pred.dtype) * valid,
            (gt_volume >= 150.0).to(pred.dtype) * valid,
        ),
        dim=1,
    ).double()


def summarize_metric_values(values: torch.Tensor) -> Dict[str, float]:
    """Reconstruct metrics from one vector of additive sufficient statistics."""

    raw = values.detach().double().cpu().reshape(-1).tolist()
    if len(raw) != len(METRIC_FIELDS):
        raise ValueError(
            f"Expected {len(METRIC_FIELDS)} metric values, got {len(raw)}"
        )
    value = {name: float(raw[index]) for index, name in enumerate(METRIC_FIELDS)}
    frames = max(value["frames"], 1.0)
    points = max(value["values"], 1.0)
    return {
        "frame_count": value["frames"],
        "mae": value["abs_sum"] / points,
        "rmse": (value["sq_sum"] / points) ** 0.5,
        "contact_iou": value["contact_iou_sum"] / frames,
        "volumetric_iou": value["viou_sum"] / frames,
        "core_distribution_viou": value["core_viou_sum"]
        / max(value["core_count"], 1.0),
        "pred_gt_volume_ratio": value["pred_volume"]
        / max(value["gt_volume"], 1e-12),
        "temporal_accuracy_frame": value["temporal_correct"] / frames,
        "false_high_excess_fraction": value["false_high_excess"]
        / max(value["pred_volume"], 1e-12),
        "catastrophic_over_rate": value["cat_over"]
        / max(value["cat_over_denom"], 1.0),
        "catastrophic_under_rate": value["cat_under"]
        / max(value["cat_under_denom"], 1.0),
    }


@dataclass
class PriorMetricAccumulator:
    contact_threshold: float = 0.10

    def __post_init__(self) -> None:
        self.values = torch.zeros(len(METRIC_FIELDS), dtype=torch.float64)

    def reset(self) -> None:
        self.values.zero_()

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        palm_mask: torch.Tensor,
        has_tactile: torch.Tensor,
    ) -> None:
        additions = metric_contributions(
            pred,
            target,
            palm_mask,
            has_tactile,
            contact_threshold=self.contact_threshold,
        )
        if self.values.device != additions.device:
            self.values = self.values.to(device=additions.device)
        self.values += additions.sum(dim=0)

    def synchronize(self, device: torch.device) -> None:
        if self.values.device != device:
            self.values = self.values.to(device=device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self.values, op=dist.ReduceOp.SUM)

    def summary(self) -> Dict[str, float]:
        return summarize_metric_values(self.values)
