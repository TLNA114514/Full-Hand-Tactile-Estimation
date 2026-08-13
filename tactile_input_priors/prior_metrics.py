"""Compact DDP-safe metrics for prior adapter training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist


_FIELDS = (
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


@dataclass
class PriorMetricAccumulator:
    contact_threshold: float = 0.10

    def __post_init__(self) -> None:
        self.values = torch.zeros(len(_FIELDS), dtype=torch.float64)

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
        pred = pred.detach().float()
        target = target.detach().float().to(pred.device)
        valid_frames = has_tactile.detach().reshape(-1).to(pred.device) > 0.5
        if not bool(valid_frames.any()):
            return
        pred = pred[valid_frames]
        target = target[valid_frames]
        palm = palm_mask.detach().to(pred.device) > 0.5
        if palm.ndim == 1:
            palm = palm.unsqueeze(0).expand_as(pred)
        else:
            palm = palm[valid_frames] if palm.shape[0] == valid_frames.shape[0] else palm
        pred = pred[palm].reshape(pred.shape[0], -1)
        target = target[palm].reshape(target.shape[0], -1)
        difference = pred - target
        pred_volume = pred.sum(dim=1)
        gt_volume = target.sum(dim=1)
        pred_contact = pred >= self.contact_threshold
        gt_contact = target >= self.contact_threshold
        intersection = (pred_contact & gt_contact).sum(dim=1).double()
        union = (pred_contact | gt_contact).sum(dim=1).double()
        contact_iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
        vol_intersection = torch.minimum(pred, target).sum(dim=1).double()
        vol_union = torch.maximum(pred, target).sum(dim=1).double()
        viou = torch.where(
            vol_union > 1e-12,
            vol_intersection / vol_union.clamp_min(1e-12),
            1.0,
        )
        eligible = (gt_volume >= 1.0) & (target.amax(dim=1) >= 0.05)
        pred_core = pred.square()
        gt_core = target.square()
        pred_dist = pred_core / pred_core.sum(dim=1, keepdim=True).clamp_min(1e-12)
        gt_dist = gt_core / gt_core.sum(dim=1, keepdim=True).clamp_min(1e-12)
        core_intersection = torch.minimum(pred_dist, gt_dist).sum(dim=1)
        core_union = torch.maximum(pred_dist, gt_dist).sum(dim=1)
        core_viou = core_intersection / core_union.clamp_min(1e-12)
        false_high = (target < 0.005) & (pred >= 0.3)

        additions = torch.stack(
            (
                pred.new_tensor(float(pred.shape[0])),
                pred.new_tensor(float(pred.numel())),
                difference.abs().sum(),
                difference.square().sum(),
                contact_iou.sum(),
                viou.sum(),
                pred_volume.sum(),
                gt_volume.sum(),
                ((pred_contact.any(dim=1)) == (gt_contact.any(dim=1))).float().sum(),
                core_viou[eligible].sum(),
                eligible.float().sum(),
                (pred - target).clamp_min(0.0)[false_high].sum(),
                ((gt_volume < 10.0) & (pred_volume > 300.0)).float().sum(),
                (gt_volume < 10.0).float().sum(),
                ((gt_volume >= 150.0) & (pred_volume < 50.0)).float().sum(),
                (gt_volume >= 150.0).float().sum(),
            )
        ).double().cpu()
        self.values += additions

    def synchronize(self, device: torch.device) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        value = self.values.to(device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        self.values.copy_(value.cpu())

    def summary(self) -> Dict[str, float]:
        value = {name: float(self.values[index]) for index, name in enumerate(_FIELDS)}
        frames = max(value["frames"], 1.0)
        points = max(value["values"], 1.0)
        return {
            "frame_count": value["frames"],
            "mae": value["abs_sum"] / points,
            "rmse": (value["sq_sum"] / points) ** 0.5,
            "contact_iou": value["contact_iou_sum"] / frames,
            "volumetric_iou": value["viou_sum"] / frames,
            "core_distribution_viou": value["core_viou_sum"] / max(value["core_count"], 1.0),
            "pred_gt_volume_ratio": value["pred_volume"] / max(value["gt_volume"], 1e-12),
            "temporal_accuracy_frame": value["temporal_correct"] / frames,
            "false_high_excess_fraction": value["false_high_excess"] / max(value["pred_volume"], 1e-12),
            "catastrophic_over_rate": value["cat_over"] / max(value["cat_over_denom"], 1.0),
            "catastrophic_under_rate": value["cat_under"] / max(value["cat_under_denom"], 1.0),
        }
