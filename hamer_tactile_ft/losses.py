from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class TactileLossConfig:
    active_pressure_thr: float = 0.05
    active_pressure_peak: float = 0.12
    active_pressure_high: float = 0.60
    background_pressure_thr: float = 0.02
    background_pred_margin: float = 0.02
    active_pressure_weight: float = 4.0
    active_pressure_gamma: float = 1.0
    background_loss_weight: float = 0.5
    volume_iou_loss_weight: float = 0.2
    opentouch_high_pressure_thr: float = 0.9
    opentouch_high_pressure_weight: float = 0.3
    loss_ramp_epochs: int = 5


def loss_ramp(config: TactileLossConfig, current_epoch: Optional[int]) -> float:
    if config.loss_ramp_epochs <= 0:
        return 1.0
    epoch = 0 if current_epoch is None else int(current_epoch)
    return min(1.0, max(0.0, float(epoch + 1) / float(config.loss_ramp_epochs)))


def _expand_palm_mask(palm_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    palm_mask = palm_mask.to(device=target.device, dtype=target.dtype)
    if palm_mask.shape == target.shape:
        return palm_mask
    if target.ndim == 2 and palm_mask.ndim == 1:
        return palm_mask.unsqueeze(0).expand_as(target)
    if target.ndim == 2 and palm_mask.ndim == 2:
        return palm_mask
    if target.ndim == 3 and palm_mask.ndim == 1:
        return palm_mask.view(1, 1, -1).expand_as(target)
    if target.ndim == 3 and palm_mask.ndim == 2:
        return palm_mask.unsqueeze(1).expand_as(target)
    return palm_mask.expand_as(target)


def _expand_valid_mask(valid_mask: Optional[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(target)
    valid = valid_mask.to(device=target.device, dtype=target.dtype)
    if valid.shape == target.shape:
        return valid
    return valid.unsqueeze(-1).expand_as(target)


def _canonical_dataset_names(dataset_batch: Any, batch_size: int, seq_len: Optional[int] = None):
    if dataset_batch is None:
        if seq_len is None:
            return [["OpenTouch"] for _ in range(batch_size)]
        return [[["OpenTouch"] for _ in range(seq_len)] for _ in range(batch_size)]
    if isinstance(dataset_batch, str):
        if seq_len is None:
            return [[dataset_batch] for _ in range(batch_size)]
        return [[[dataset_batch] for _ in range(seq_len)] for _ in range(batch_size)]
    if not isinstance(dataset_batch, (list, tuple)):
        if seq_len is None:
            return [[str(dataset_batch)] for _ in range(batch_size)]
        return [[[str(dataset_batch)] for _ in range(seq_len)] for _ in range(batch_size)]

    if seq_len is None:
        names = list(dataset_batch)
        if len(names) == batch_size:
            return [[name] for name in names]
        return [[names[0] if names else "OpenTouch"] for _ in range(batch_size)]

    # PyTorch default collate transposes list[str] sequences into list length T,
    # each item usually containing B dataset names.
    if len(dataset_batch) == seq_len:
        out = [[["OpenTouch"] for _ in range(seq_len)] for _ in range(batch_size)]
        for t, names_at_t in enumerate(dataset_batch):
            if isinstance(names_at_t, (list, tuple)):
                names = list(names_at_t)
            else:
                names = [names_at_t] * batch_size
            for b, name in enumerate(names[:batch_size]):
                out[b][t] = [name]
        return out

    if len(dataset_batch) == batch_size:
        out = []
        for item in dataset_batch:
            if isinstance(item, (list, tuple)) and len(item) == seq_len:
                out.append([[name] for name in item])
            else:
                out.append([[item] for _ in range(seq_len)])
        return out

    return [[["OpenTouch"] for _ in range(seq_len)] for _ in range(batch_size)]


def dataset_weight_like(target: torch.Tensor, dataset_batch: Any, config: TactileLossConfig) -> torch.Tensor:
    weights = torch.ones_like(target)
    if target.ndim == 2:
        batch_size = target.shape[0]
        names = _canonical_dataset_names(dataset_batch, batch_size)
        for b in range(batch_size):
            if str(names[b][0]).lower() == "opentouch":
                weights[b] = torch.where(
                    target[b] > config.opentouch_high_pressure_thr,
                    torch.as_tensor(config.opentouch_high_pressure_weight, dtype=target.dtype, device=target.device),
                    torch.ones((), dtype=target.dtype, device=target.device),
                )
        return weights

    batch_size, seq_len = target.shape[:2]
    names = _canonical_dataset_names(dataset_batch, batch_size, seq_len)
    for b in range(batch_size):
        for t in range(seq_len):
            if str(names[b][t][0]).lower() == "opentouch":
                weights[b, t] = torch.where(
                    target[b, t] > config.opentouch_high_pressure_thr,
                    torch.as_tensor(config.opentouch_high_pressure_weight, dtype=target.dtype, device=target.device),
                    torch.ones((), dtype=target.dtype, device=target.device),
                )
    return weights


def compute_tactile_loss(
    pred: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    palm_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    dataset_batch: Any,
    config: Optional[TactileLossConfig] = None,
    current_epoch: Optional[int] = None,
    sample_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    config = config or TactileLossConfig()
    ramp = loss_ramp(config, current_epoch)

    target = target.to(dtype=pred.dtype, device=pred.device)
    logits = logits.to(device=pred.device)
    palm = _expand_palm_mask(palm_mask, target)
    valid = _expand_valid_mask(valid_mask, target)
    if sample_weight is not None:
        valid = valid * _expand_valid_mask(sample_weight, target)

    smooth = F.smooth_l1_loss(pred, target, reduction="none")
    highway = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    base = smooth + 0.1 * highway

    active = (target >= config.active_pressure_thr).to(target.dtype)
    peak = max(config.active_pressure_peak, config.active_pressure_thr + 1e-6)
    high = max(config.active_pressure_high, peak + 1e-6)
    rising = (target - config.active_pressure_thr) / (peak - config.active_pressure_thr)
    falling = (high - target) / (high - peak)
    hump = torch.where(target <= peak, rising, falling)
    hump = torch.clamp(hump, min=0.0, max=1.0) * active
    active_strength = hump.pow(config.active_pressure_gamma)
    pressure_weight = 1.0 + ramp * config.active_pressure_weight * active_strength
    weights = pressure_weight * dataset_weight_like(target, dataset_batch, config)

    weighted = base * weights * palm * valid
    denom = (palm * valid).sum().clamp_min(1.0)
    weighted_tactile = weighted.sum() / denom
    base_tactile = (base * palm * valid).sum() / denom

    background_mask = ((target <= config.background_pressure_thr).to(target.dtype) * palm * valid)
    bg_denom = background_mask.sum().clamp_min(1.0)
    background = (F.relu(pred - config.background_pred_margin).pow(2) * background_mask).sum() / bg_denom
    background = background * (ramp * config.background_loss_weight)

    active_frames = (target.amax(dim=-1) >= config.active_pressure_thr).to(target.dtype)
    if valid_mask is not None:
        active_frames = active_frames * valid_mask.to(device=target.device, dtype=target.dtype)
    if sample_weight is not None:
        active_frames = active_frames * sample_weight.to(device=target.device, dtype=target.dtype)
    if active_frames.sum() > 0:
        vol_min = torch.minimum(pred, target) * palm
        vol_max = torch.maximum(pred, target) * palm
        vol_iou = vol_min.sum(dim=-1) / vol_max.sum(dim=-1).clamp_min(1e-6)
        volume_iou_loss = ((1.0 - vol_iou) * active_frames).sum() / active_frames.sum().clamp_min(1.0)
    else:
        volume_iou_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    volume_iou_loss = volume_iou_loss * (ramp * config.volume_iou_loss_weight)

    total = weighted_tactile + background + volume_iou_loss

    pred_volume = (pred * palm * valid).sum()
    gt_volume = (target * palm * valid).sum()
    ratio = pred_volume / gt_volume.clamp_min(1e-6)
    losses = {
        "loss_base_tactile": base_tactile.detach(),
        "loss_weighted_tactile": weighted_tactile.detach(),
        "loss_background": background.detach(),
        "loss_volume_iou": volume_iou_loss.detach(),
        "loss_tactile": total.detach(),
        "pred_volume": pred_volume.detach(),
        "gt_volume": gt_volume.detach(),
        "pred_gt_volume_ratio": ratio.detach(),
        "loss_ramp": torch.as_tensor(ramp, device=pred.device, dtype=pred.dtype),
    }
    return total, losses
