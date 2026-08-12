from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
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
    pressure_weight_mode: str = "hump"
    active_pressure_tail_thr: float = 0.70
    active_pressure_tail_max: float = 3.0
    background_loss_weight: float = 1.0
    logit_bce_weight: float = 0.1
    loss_ramp_epochs: int = 5
    opentouch_high_pressure_thr: float = 0.9
    opentouch_high_pressure_weight: float = 0.3
    location_loss_weight: float = 0.0
    location_gt_volume_thr: float = 1.0
    location_distribution_power: float = 1.0
    location_min_gt_peak: float = 0.0
    contact_loss_type: str = "none"
    contact_loss_weight: float = 0.0
    contact_pressure_thr: float = 0.1
    contact_temperature: float = 0.025

    def __post_init__(self):
        if self.loss_mode != "dense_v2":
            raise ValueError("Only loss_mode=dense_v2 is supported")
        if not 0.0 <= self.active_pressure_thr < self.active_pressure_peak < self.active_pressure_high <= 1.0:
            raise ValueError("Expected active_pressure_thr < active_pressure_peak < active_pressure_high")
        if self.pressure_weight_mode not in {"hump", "plateau", "capped_linear"}:
            raise ValueError(
                "pressure_weight_mode must be one of: hump, plateau, capped_linear"
            )
        if not self.active_pressure_peak < self.active_pressure_tail_thr <= 1.0:
            raise ValueError(
                "active_pressure_tail_thr must lie in (active_pressure_peak, 1]"
            )
        peak_total_weight = 1.0 + float(self.active_pressure_weight)
        if not math.isfinite(float(self.active_pressure_tail_max)) or (
            float(self.active_pressure_tail_max) < peak_total_weight
        ):
            raise ValueError(
                "active_pressure_tail_max must be finite and at least "
                "1 + active_pressure_weight"
            )
        if not 0.0 <= self.background_pressure_thr <= self.background_pred_margin <= 1.0:
            raise ValueError("Background thresholds must lie in [0, 1]")
        for name in (
            "active_pressure_weight",
            "active_pressure_gamma",
            "background_loss_weight",
            "logit_bce_weight",
            "location_loss_weight",
            "contact_loss_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if self.location_gt_volume_thr < 0.0:
            raise ValueError("location_gt_volume_thr must be nonnegative")
        if not math.isfinite(float(self.location_distribution_power)) or not (
            1.0 <= float(self.location_distribution_power) <= 4.0
        ):
            raise ValueError("location_distribution_power must be finite and lie in [1, 4]")
        if not math.isfinite(float(self.location_min_gt_peak)) or not (
            0.0 <= float(self.location_min_gt_peak) <= 1.0
        ):
            raise ValueError("location_min_gt_peak must be finite and lie in [0, 1]")
        if self.contact_loss_type not in {"none", "soft_jaccard", "lovasz"}:
            raise ValueError(
                "contact_loss_type must be one of: none, soft_jaccard, lovasz"
            )
        if not math.isfinite(float(self.contact_loss_weight)):
            raise ValueError("contact_loss_weight must be finite and nonnegative")
        if not math.isfinite(float(self.contact_pressure_thr)) or not (
            0.0 <= float(self.contact_pressure_thr) <= 1.0
        ):
            raise ValueError("contact_pressure_thr must be finite and lie in [0, 1]")
        if not math.isfinite(float(self.contact_temperature)) or not (
            float(self.contact_temperature) > 0.0
        ):
            raise ValueError("contact_temperature must be finite and positive")
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
    rising_strength = torch.clamp(rising, min=0.0, max=1.0).pow(
        config.active_pressure_gamma
    ) * active
    peak_additive = float(config.active_pressure_weight)

    if config.pressure_weight_mode == "hump":
        falling = (config.active_pressure_high - target) / (
            config.active_pressure_high - config.active_pressure_peak
        )
        hump = torch.where(target <= config.active_pressure_peak, rising, falling)
        active_strength = (
            torch.clamp(hump, min=0.0, max=1.0).pow(config.active_pressure_gamma)
            * active
        )
        additive = peak_additive * active_strength
    elif config.pressure_weight_mode == "plateau":
        additive = peak_additive * rising_strength
    else:
        tail_progress = torch.clamp(
            (target - config.active_pressure_peak)
            / (config.active_pressure_tail_thr - config.active_pressure_peak),
            min=0.0,
            max=1.0,
        )
        tail_additive = peak_additive + tail_progress * (
            float(config.active_pressure_tail_max) - 1.0 - peak_additive
        )
        additive = torch.where(
            target <= config.active_pressure_peak,
            peak_additive * rising_strength,
            tail_additive * active,
        )
    return 1.0 + float(ramp) * additive


def global_conditional_mean(local_sum: torch.Tensor, local_count: torch.Tensor) -> torch.Tensor:
    """Return a DDP-gradient-correct mean over a distributed conditional mask."""
    global_count = local_count.detach().to(device=local_sum.device, dtype=torch.float32).clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    return local_sum * float(world_size) / global_count.clamp_min(1.0)


def _location_distribution_mass(values: torch.Tensor, power: float) -> torch.Tensor:
    values = values.float().clamp_min(0.0)
    return values if float(power) == 1.0 else values.pow(float(power))


def _lovasz_gradient(sorted_labels: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension with respect to sorted errors."""
    count = sorted_labels.numel()
    if count == 0:
        return sorted_labels
    total_positive = sorted_labels.sum()
    intersection = total_positive - sorted_labels.cumsum(dim=0)
    union = total_positive + (1.0 - sorted_labels).cumsum(dim=0)
    gradient = 1.0 - intersection / union.clamp_min(1e-12)
    if count > 1:
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


def _masked_lovasz_hinge(
    contact_logits: torch.Tensor,
    contact_labels: torch.Tensor,
    valid_vertices: torch.Tensor,
) -> torch.Tensor:
    logits = contact_logits[valid_vertices]
    labels = contact_labels[valid_vertices]
    if logits.numel() == 0:
        return contact_logits.sum() * 0.0
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    sorted_errors, permutation = torch.sort(errors, descending=True)
    sorted_labels = labels[permutation]
    return torch.dot(F.relu(sorted_errors), _lovasz_gradient(sorted_labels))


def _contact_loss_sum_and_count(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    config: TactileLossConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the sum of per-frame contact losses and eligible frame count."""
    pred_frames = pred.float().reshape(-1, pred.shape[-1])
    target_frames = target.float().reshape_as(pred_frames)
    valid_frames = (mask > 0.0).reshape_as(pred_frames)
    eligible = valid_frames.any(dim=-1)
    contact_logits = (
        pred_frames - float(config.contact_pressure_thr)
    ) / float(config.contact_temperature)
    # Match the TouchAnything protocol exactly: contact is strictly above 0.1.
    contact_labels = (target_frames > float(config.contact_pressure_thr)).float()

    if config.contact_loss_type == "soft_jaccard":
        probabilities = torch.sigmoid(contact_logits)
        valid_float = valid_frames.float()
        intersection = (probabilities * contact_labels * valid_float).sum(dim=-1)
        union = (
            (probabilities + contact_labels - probabilities * contact_labels)
            * valid_float
        ).sum(dim=-1)
        per_frame = 1.0 - (intersection + 1.0) / (union + 1.0)
    elif config.contact_loss_type == "lovasz":
        per_frame = (
            torch.stack(
                [
                    _masked_lovasz_hinge(frame_logits, frame_labels, frame_valid)
                    for frame_logits, frame_labels, frame_valid in zip(
                        contact_logits, contact_labels, valid_frames
                    )
                ]
            )
            if contact_logits.shape[0] > 0
            else contact_logits.new_empty((0,))
        )
    else:
        raise ValueError(f"Unsupported contact loss type: {config.contact_loss_type}")

    contact_sum = per_frame.masked_fill(~eligible, 0.0).sum()
    return contact_sum, eligible.float().sum()


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
    ramp_override: Optional[float] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    config = config or TactileLossConfig()
    ramp = loss_ramp(config, current_epoch) if ramp_override is None else float(ramp_override)
    if not 0.0 <= ramp <= 1.0:
        raise ValueError(f"ramp_override must lie in [0, 1], got {ramp}")

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
    full_ramp_weights = pressure_weight_like(target, config, 1.0) * dataset_weight_like(
        target, dataset_batch, config
    )
    mask = palm * valid
    denom = mask.sum().clamp_min(1.0)
    weighted_tactile = (direct * weights * mask).sum() / denom
    weighted_tactile_full_ramp = (direct * full_ramp_weights * mask).sum() / denom
    base_tactile = (direct * mask).sum() / denom
    pressure_weight_mean = (pressure_weight * mask).sum() / denom
    pressure_weight_max = (pressure_weight * mask).amax()
    pressure_weight_fraction_gt2 = (
        ((pressure_weight > 2.0).to(mask.dtype) * mask).sum() / denom
    )
    weighted_to_direct_loss_ratio = weighted_tactile / base_tactile.detach().clamp_min(
        1e-12
    )
    smooth_l1_raw = (smooth * mask).sum() / denom
    logit_bce_raw = (logit_bce * mask).sum() / denom

    background_mask = (target <= config.background_pressure_thr).to(target.dtype) * mask
    background_denom = background_mask.sum().clamp_min(1.0)
    background_raw = (
        F.relu(pred - config.background_pred_margin).pow(2) * background_mask
    ).sum() / background_denom
    background = background_raw * (ramp * config.background_loss_weight)

    total = weighted_tactile + background

    location_mask = palm.float() * (valid > 0.0).float()
    location_gt_raw = target.float().clamp_min(0.0) * location_mask
    location_gt = _location_distribution_mass(
        location_gt_raw,
        config.location_distribution_power,
    )
    location_gt_volume_raw = location_gt_raw.sum(dim=-1)
    location_gt_volume = location_gt.sum(dim=-1)
    location_gt_peak = location_gt_raw.amax(dim=-1)
    location_valid_frame = location_mask.sum(dim=-1) > 0.0
    location_eligible = location_valid_frame & (
        location_gt_volume_raw >= config.location_gt_volume_thr
    ) & (
        location_gt_peak >= config.location_min_gt_peak
    )
    location_count_local = location_eligible.float().sum()
    location_valid_count_local = location_valid_frame.float().sum()
    location_eligible_fraction = (
        location_count_local / location_valid_count_local.clamp_min(1.0)
    )

    if config.location_loss_weight > 0.0:
        location_pred = (
            _location_distribution_mass(pred, config.location_distribution_power)
            * location_mask
        )
        location_pred_volume = location_pred.sum(dim=-1)
        pred_dist = location_pred / location_pred_volume.unsqueeze(-1).clamp_min(1e-12)
        gt_dist = location_gt / location_gt_volume.unsqueeze(-1).clamp_min(1e-12)
        location_intersection = torch.minimum(pred_dist, gt_dist).sum(dim=-1)
        location_union = torch.maximum(pred_dist, gt_dist).sum(dim=-1)
        location_viou = torch.where(
            location_union > 1e-12,
            location_intersection / location_union.clamp_min(1e-12),
            torch.zeros_like(location_union),
        )
        location_sum = (1.0 - location_viou).masked_fill(~location_eligible, 0.0).sum()
        location_loss_raw = global_conditional_mean(location_sum, location_count_local)
        location_loss_weighted = location_loss_raw * (ramp * config.location_loss_weight)
        total = total + location_loss_weighted
    else:
        # Keep weight=0 numerically identical to the pre-location path and collective-free.
        location_loss_raw = pred.new_zeros(())
        location_loss_weighted = location_loss_raw

    if config.contact_loss_type != "none" and config.contact_loss_weight > 0.0:
        contact_sum, contact_count_local = _contact_loss_sum_and_count(
            pred,
            target,
            mask,
            config,
        )
        contact_loss_raw = global_conditional_mean(contact_sum, contact_count_local)
        contact_loss_weighted = contact_loss_raw * (ramp * config.contact_loss_weight)
        total = total + contact_loss_weighted
    else:
        # Preserve the baseline graph and avoid an unnecessary collective.
        contact_loss_raw = pred.new_zeros(())
        contact_loss_weighted = contact_loss_raw

    full_ramp_total = (
        weighted_tactile_full_ramp
        + background_raw * config.background_loss_weight
        + location_loss_raw * config.location_loss_weight
        + contact_loss_raw * config.contact_loss_weight
    )

    return total, {
        "loss_smooth_l1_raw": smooth_l1_raw.detach(),
        "loss_logit_bce_raw": logit_bce_raw.detach(),
        "loss_base_tactile": base_tactile.detach(),
        "loss_weighted_tactile": weighted_tactile.detach(),
        "diagnostics_pressure_weight_mean": pressure_weight_mean.detach(),
        "diagnostics_pressure_weight_max": pressure_weight_max.detach(),
        "diagnostics_pressure_weight_fraction_gt2": pressure_weight_fraction_gt2.detach(),
        "diagnostics_weighted_to_direct_loss_ratio": weighted_to_direct_loss_ratio.detach(),
        "loss_background": background.detach(),
        "loss_location_raw": location_loss_raw.detach(),
        "loss_location_weighted": location_loss_weighted.detach(),
        "diagnostics_location_eligible_fraction": location_eligible_fraction.detach(),
        "loss_contact_raw": contact_loss_raw.detach(),
        "loss_contact_weighted": contact_loss_weighted.detach(),
        "loss_full_ramp": full_ramp_total.detach(),
        "loss_tactile": total.detach(),
        "loss_ramp": pred.new_tensor(ramp),
    }
