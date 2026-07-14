from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class TactileLossConfig:
    """Loss used by the original dense V2 tactile regressor."""

    loss_mode: str = "dense_v2"
    active_pressure_thr: float = 0.05
    active_pressure_peak: float = 0.10
    active_pressure_high: float = 0.30
    background_pressure_thr: float = 0.02
    background_pred_margin: float = 0.02
    active_pressure_weight: float = 1.0
    active_pressure_gamma: float = 1.0
    background_loss_weight: float = 1.0
    logit_bce_weight: float = 0.1
    loss_ramp_epochs: int = 10
    opentouch_high_pressure_thr: float = 0.9
    opentouch_high_pressure_weight: float = 0.3

    def __post_init__(self):
        if self.loss_mode != "dense_v2":
            raise ValueError("Only loss_mode=dense_v2 is supported")
        if not 0.0 <= self.active_pressure_thr < self.active_pressure_peak < self.active_pressure_high <= 1.0:
            raise ValueError("Expected active_pressure_thr < active_pressure_peak < active_pressure_high")
        if not 0.0 <= self.background_pressure_thr <= self.background_pred_margin <= 1.0:
            raise ValueError("Background thresholds must lie in [0, 1]")
        for name in ("active_pressure_weight", "active_pressure_gamma", "background_loss_weight", "logit_bce_weight"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 < self.opentouch_high_pressure_weight <= 1.0:
            raise ValueError("opentouch_high_pressure_weight must lie in (0, 1]")


def loss_ramp(config: TactileLossConfig, current_epoch: Optional[int]) -> float:
    """Match V2: epoch zero starts at 1 / ramp_epochs, not zero."""
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

    if len(dataset_batch) == seq_len:
        out = [[["OpenTouch"] for _ in range(seq_len)] for _ in range(batch_size)]
        for timestep, names_at_timestep in enumerate(dataset_batch):
            names = (
                list(names_at_timestep)
                if isinstance(names_at_timestep, (list, tuple))
                else [names_at_timestep] * batch_size
            )
            for batch_index, name in enumerate(names[:batch_size]):
                out[batch_index][timestep] = [name]
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
        names = _canonical_dataset_names(dataset_batch, target.shape[0])
        for batch_index in range(target.shape[0]):
            if str(names[batch_index][0]).lower() == "opentouch":
                weights[batch_index] = torch.where(
                    target[batch_index] > config.opentouch_high_pressure_thr,
                    target.new_tensor(config.opentouch_high_pressure_weight),
                    target.new_ones(()),
                )
        return weights

    batch_size, seq_len = target.shape[:2]
    names = _canonical_dataset_names(dataset_batch, batch_size, seq_len)
    for batch_index in range(batch_size):
        for timestep in range(seq_len):
            if str(names[batch_index][timestep][0]).lower() == "opentouch":
                weights[batch_index, timestep] = torch.where(
                    target[batch_index, timestep] > config.opentouch_high_pressure_thr,
                    target.new_tensor(config.opentouch_high_pressure_weight),
                    target.new_ones(()),
                )
    return weights


def pressure_weight_like(target: torch.Tensor, config: TactileLossConfig, ramp: float) -> torch.Tensor:
    active = (target >= config.active_pressure_thr).to(target.dtype)
    rising = (target - config.active_pressure_thr) / (config.active_pressure_peak - config.active_pressure_thr)
    falling = (config.active_pressure_high - target) / (config.active_pressure_high - config.active_pressure_peak)
    hump = torch.where(target <= config.active_pressure_peak, rising, falling)
    active_strength = torch.clamp(hump, min=0.0, max=1.0).pow(config.active_pressure_gamma) * active
    return 1.0 + float(ramp) * config.active_pressure_weight * active_strength


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
    logits = logits.to(device=pred.device, dtype=pred.dtype)
    palm = _expand_palm_mask(palm_mask, target)
    valid = _expand_valid_mask(valid_mask, target)
    if sample_weight is not None:
        valid = valid * _expand_valid_mask(sample_weight, target)

    smooth = F.smooth_l1_loss(pred, target, reduction="none")
    logit_bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    direct = smooth + config.logit_bce_weight * logit_bce

    pressure_weight = pressure_weight_like(target, config, ramp)
    weights = pressure_weight * dataset_weight_like(target, dataset_batch, config)
    mask = palm * valid
    denom = mask.sum().clamp_min(1.0)
    weighted_tactile = (direct * weights * mask).sum() / denom
    base_tactile = (direct * mask).sum() / denom
    smooth_l1_raw = (smooth * mask).sum() / denom
    logit_bce_raw = (logit_bce * mask).sum() / denom

    background_mask = (target <= config.background_pressure_thr).to(target.dtype) * mask
    background_denom = background_mask.sum().clamp_min(1.0)
    background = (
        F.relu(pred - config.background_pred_margin).pow(2) * background_mask
    ).sum() / background_denom
    background = background * (ramp * config.background_loss_weight)

    total = weighted_tactile + background
    return total, {
        "loss_smooth_l1_raw": smooth_l1_raw.detach(),
        "loss_logit_bce_raw": logit_bce_raw.detach(),
        "loss_base_tactile": base_tactile.detach(),
        "loss_weighted_tactile": weighted_tactile.detach(),
        "loss_background": background.detach(),
        "loss_tactile": total.detach(),
        "loss_ramp": pred.new_tensor(ramp),
    }
