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
    center_loss_weight: float = 0.0
    center_presence_loss_weight: float = 0.0
    center_aux_loss_weight: float = 0.0
    center_aux_presence_loss_weight: float = 0.0
    center_threshold_scale: float = 0.35
    center_threshold_min: float = 0.05
    center_threshold_max: float = 0.20
    center_target_power: float = 2.0
    center_presence_volume_thr: float = 1.0
    center_presence_peak_thr: float = 0.10
    center_presence_logit_scale: float = 4.0
    contact_loss_type: str = "none"
    contact_loss_weight: float = 0.0
    contact_pressure_thr: float = 0.1
    contact_temperature: float = 0.025

    def __post_init__(self):
        if self.loss_mode != "dense_v2":
            raise ValueError("Only loss_mode=dense_v2 is supported")
        if not 0.0 <= self.active_pressure_thr < self.active_pressure_peak < self.active_pressure_high <= 1.0:
            raise ValueError("Expected active_pressure_thr < active_pressure_peak < active_pressure_high")
        if self.pressure_weight_mode not in {
            "hump",
            "flat",
            "contact_step",
            "plateau",
            "capped_linear",
        }:
            raise ValueError(
                "pressure_weight_mode must be one of: hump, flat, "
                "contact_step, plateau, capped_linear"
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
            "center_loss_weight",
            "center_presence_loss_weight",
            "center_aux_loss_weight",
            "center_aux_presence_loss_weight",
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
        if not math.isfinite(float(self.center_threshold_scale)) or not (
            float(self.center_threshold_scale) > 0.0
        ):
            raise ValueError("center_threshold_scale must be finite and positive")
        if not (
            0.0 <= float(self.center_threshold_min)
            <= float(self.center_threshold_max)
            <= 1.0
        ):
            raise ValueError(
                "center thresholds must satisfy 0 <= min <= max <= 1"
            )
        if not math.isfinite(float(self.center_target_power)) or not (
            float(self.center_target_power) > 0.0
        ):
            raise ValueError("center_target_power must be finite and positive")
        if not math.isfinite(float(self.center_presence_volume_thr)) or not (
            float(self.center_presence_volume_thr) >= 0.0
        ):
            raise ValueError(
                "center_presence_volume_thr must be finite and nonnegative"
            )
        if not math.isfinite(float(self.center_presence_peak_thr)) or not (
            0.0 <= float(self.center_presence_peak_thr) <= 1.0
        ):
            raise ValueError(
                "center_presence_peak_thr must be finite and lie in [0, 1]"
            )
        if not math.isfinite(float(self.center_presence_logit_scale)) or not (
            float(self.center_presence_logit_scale) > 0.0
        ):
            raise ValueError(
                "center_presence_logit_scale must be finite and positive"
            )
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

    if config.pressure_weight_mode == "flat":
        additive = torch.zeros_like(target)
    elif config.pressure_weight_mode == "contact_step":
        contact = (target >= float(config.contact_pressure_thr)).to(target.dtype)
        additive = peak_additive * contact
    elif config.pressure_weight_mode == "hump":
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
    elif config.pressure_weight_mode == "capped_linear":
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
    else:
        raise RuntimeError(
            f"Unsupported pressure_weight_mode={config.pressure_weight_mode!r}"
        )
    return 1.0 + float(ramp) * additive


def global_conditional_mean(
    local_sum: torch.Tensor,
    local_count: torch.Tensor,
    *,
    distributed_reduce: bool = True,
) -> torch.Tensor:
    """Return a gradient-correct conditional mean.

    Training uses a global eligible count so that DDP ranks contribute to one
    common mean. Offline audits may disable that collective and merge their
    numerators/counts after processing; this is required when rank-local
    validity masks cause ranks to evaluate different candidates.
    """
    global_count = local_count.detach().to(device=local_sum.device, dtype=torch.float32).clone()
    world_size = 1
    if distributed_reduce and dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    return local_sum * float(world_size) / global_count.clamp_min(1.0)


def _location_distribution_mass(values: torch.Tensor, power: float) -> torch.Tensor:
    values = values.float().clamp_min(0.0)
    return values if float(power) == 1.0 else values.pow(float(power))


def compute_center_regularization(
    pred: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    vertex_mask: torch.Tensor,
    config: TactileLossConfig,
    *,
    distributed_reduce: bool = True,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regularize the existing pressure output without adding model parameters."""

    pred = pred.float()
    logits = logits.float()
    target = target.to(device=logits.device, dtype=torch.float32)
    if pred.shape != target.shape or logits.shape != target.shape:
        raise ValueError(
            "pressure prediction, logits, and target must have the same shape: "
            f"pred={tuple(pred.shape)}, logits={tuple(logits.shape)}, "
            f"target={tuple(target.shape)}"
        )
    vertex_mask = vertex_mask.to(device=target.device) > 0.0
    if vertex_mask.shape != target.shape:
        raise ValueError(
            f"center vertex mask must match target: {tuple(vertex_mask.shape)} "
            f"vs {tuple(target.shape)}"
        )
    frame_valid = vertex_mask.any(dim=-1)
    masked_target = target.clamp_min(0.0) * vertex_mask.float()
    target_peak = masked_target.amax(dim=-1)
    target_volume = masked_target.sum(dim=-1)
    threshold = torch.clamp(
        float(config.center_threshold_scale) * target_peak,
        min=float(config.center_threshold_min),
        max=float(config.center_threshold_max),
    )
    excess = torch.relu(masked_target - threshold[:, None]) * vertex_mask.float()
    excess_peak = excess.amax(dim=-1)
    heat = (
        excess / excess_peak[:, None].clamp_min(1e-12)
    ).pow(float(config.center_target_power))
    heat = heat * vertex_mask.float()
    heat_mass = heat.sum(dim=-1)
    center_eligible = frame_valid & (heat_mass > 1e-12)
    center_target = heat / heat_mass[:, None].clamp_min(1e-12)

    masked_center_logits = logits.masked_fill(
        ~vertex_mask,
        torch.finfo(logits.dtype).min,
    )
    center_log_prob = F.log_softmax(masked_center_logits, dim=-1)
    center_kl_per_frame = F.kl_div(
        center_log_prob,
        center_target,
        reduction="none",
    ).sum(dim=-1)
    center_count = center_eligible.float().sum()
    center_sum = center_kl_per_frame.masked_fill(~center_eligible, 0.0).sum()

    masked_pred = pred.clamp(0.0, 1.0) * vertex_mask.float()
    pred_volume = masked_pred.sum(dim=-1)
    pred_peak = masked_pred.amax(dim=-1)
    presence_target = (
        (target_volume >= float(config.center_presence_volume_thr))
        & (target_peak >= float(config.center_presence_peak_thr))
    ).float()
    volume_margin = (
        pred_volume - float(config.center_presence_volume_thr)
    ) / max(float(config.center_presence_volume_thr), 1e-6)
    peak_margin = (
        pred_peak - float(config.center_presence_peak_thr)
    ) / max(float(config.center_presence_peak_thr), 1e-6)
    margin_pair = torch.stack((volume_margin, peak_margin), dim=-1)
    smooth_and_margin = (
        -torch.logsumexp(-margin_pair, dim=-1) + math.log(2.0)
    )
    presence_logits = (
        float(config.center_presence_logit_scale) * smooth_and_margin
    )
    presence_per_frame = F.binary_cross_entropy_with_logits(
        presence_logits,
        presence_target,
        reduction="none",
    )
    presence_count = frame_valid.float().sum()
    presence_sum = presence_per_frame.masked_fill(~frame_valid, 0.0).sum()
    global_counts = torch.stack((center_count, presence_count)).detach().float()
    world_size = 1
    if distributed_reduce and dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    center_loss_raw = (
        center_sum * float(world_size) / global_counts[0].clamp_min(1.0)
    )
    presence_loss_raw = (
        presence_sum * float(world_size) / global_counts[1].clamp_min(1.0)
    )

    center_loss_weighted = center_loss_raw * float(config.center_loss_weight)
    presence_loss_weighted = (
        presence_loss_raw * float(config.center_presence_loss_weight)
    )
    total = center_loss_weighted + presence_loss_weighted
    valid_count = frame_valid.float().sum().clamp_min(1.0)
    eligible_threshold_mean = (
        threshold.masked_fill(~center_eligible, 0.0).sum()
        / center_count.clamp_min(1.0)
    )
    center_pred = center_log_prob.exp()
    center_intersection = torch.minimum(center_pred, center_target).sum(dim=-1)
    center_union = torch.maximum(center_pred, center_target).sum(dim=-1)
    center_viou = torch.where(
        center_union > 1e-12,
        center_intersection / center_union.clamp_min(1e-12),
        torch.zeros_like(center_union),
    )
    return total, {
        "loss_center_raw": center_loss_raw.detach(),
        "loss_center_weighted": center_loss_weighted.detach(),
        "loss_center_presence_raw": presence_loss_raw.detach(),
        "loss_center_presence_weighted": presence_loss_weighted.detach(),
        "loss_center_total": total.detach(),
        "diagnostics_center_eligible_fraction": (
            center_count / valid_count
        ).detach(),
        "diagnostics_center_threshold_mean": eligible_threshold_mean.detach(),
        "diagnostics_center_target_support_fraction": (
            ((heat > 0.0) & vertex_mask).float().sum()
            / vertex_mask.float().sum().clamp_min(1.0)
        ).detach(),
        "diagnostics_center_distribution_viou": (
            center_viou.masked_fill(~center_eligible, 0.0).sum()
            / center_count.clamp_min(1.0)
        ).detach(),
        "diagnostics_center_presence_positive_fraction": (
            (presence_target * frame_valid.float()).sum() / valid_count
        ).detach(),
        "diagnostics_center_presence_accuracy": (
            (
                (torch.sigmoid(presence_logits) >= 0.5)
                == (presence_target >= 0.5)
            ).float()
            * frame_valid.float()
        ).sum().div(valid_count).detach(),
        "diagnostics_center_presence_predicted_fraction": (
            ((presence_logits >= 0.0).float() * frame_valid.float()).sum()
            / valid_count
        ).detach(),
    }


def compute_center_auxiliary_loss(
    center_logits: torch.Tensor,
    presence_logits: torch.Tensor,
    target: torch.Tensor,
    palm_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    config: TactileLossConfig,
    current_epoch: Optional[int] = None,
    sample_weight: Optional[torch.Tensor] = None,
    *,
    distributed_reduce: bool = True,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Deep supervision for a center head that never alters pressure logits."""

    center_logits = center_logits.float()
    presence_logits = presence_logits.float().reshape(-1)
    target = target.to(device=center_logits.device, dtype=torch.float32)
    if center_logits.shape != target.shape:
        raise ValueError(
            "center auxiliary logits and target must have the same shape: "
            f"{tuple(center_logits.shape)} vs {tuple(target.shape)}"
        )
    if presence_logits.shape != target.shape[:1]:
        raise ValueError(
            "center presence logits must contain one value per frame: "
            f"{tuple(presence_logits.shape)} vs {tuple(target.shape[:1])}"
        )

    palm = _expand_palm_mask(palm_mask, target) > 0.0
    valid = _expand_valid_mask(valid_mask, target)
    if sample_weight is not None:
        valid = valid * _expand_valid_mask(sample_weight, target)
    vertex_mask = palm & (valid > 0.0)
    frame_valid = vertex_mask.any(dim=-1)
    mask_float = vertex_mask.float()
    masked_target = target.clamp_min(0.0) * mask_float
    target_peak = masked_target.amax(dim=-1)
    target_volume = masked_target.sum(dim=-1)
    threshold = torch.clamp(
        float(config.center_threshold_scale) * target_peak,
        min=float(config.center_threshold_min),
        max=float(config.center_threshold_max),
    )
    excess = torch.relu(masked_target - threshold[:, None]) * mask_float
    excess_peak = excess.amax(dim=-1)
    heat = (
        excess / excess_peak[:, None].clamp_min(1e-12)
    ).pow(float(config.center_target_power))
    heat = heat * mask_float
    heat_mass = heat.sum(dim=-1)
    center_eligible = frame_valid & (heat_mass > 1e-12)
    center_target = heat / heat_mass[:, None].clamp_min(1e-12)

    masked_logits = center_logits.masked_fill(
        ~vertex_mask,
        torch.finfo(center_logits.dtype).min,
    )
    center_log_prob = F.log_softmax(masked_logits, dim=-1)
    center_kl = F.kl_div(
        center_log_prob,
        center_target,
        reduction="none",
    ).sum(dim=-1)
    log_valid_vertices = vertex_mask.sum(dim=-1).clamp_min(2).float().log()
    center_kl = center_kl / log_valid_vertices
    center_sum = center_kl.masked_fill(~center_eligible, 0.0).sum()
    center_count = center_eligible.float().sum()

    presence_target = (
        frame_valid
        & (target_volume >= float(config.center_presence_volume_thr))
        & (target_peak >= float(config.center_presence_peak_thr))
    ).float()
    presence_per_frame = F.binary_cross_entropy_with_logits(
        presence_logits,
        presence_target,
        reduction="none",
    )
    presence_sum = presence_per_frame.masked_fill(~frame_valid, 0.0).sum()
    presence_count = frame_valid.float().sum()

    global_counts = torch.stack((center_count, presence_count)).detach().float()
    world_size = 1
    if distributed_reduce and dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    center_raw = center_sum * float(world_size) / global_counts[0].clamp_min(1.0)
    presence_raw = (
        presence_sum * float(world_size) / global_counts[1].clamp_min(1.0)
    )
    ramp = loss_ramp(config, current_epoch)
    center_weighted = (
        center_raw * ramp * float(config.center_aux_loss_weight)
    )
    presence_weighted = (
        presence_raw
        * ramp
        * float(config.center_aux_presence_loss_weight)
    )
    total = center_weighted + presence_weighted

    center_probability = center_log_prob.exp()
    center_intersection = torch.minimum(
        center_probability, center_target
    ).sum(dim=-1)
    center_union = torch.maximum(
        center_probability, center_target
    ).sum(dim=-1)
    center_viou = torch.where(
        center_union > 1e-12,
        center_intersection / center_union.clamp_min(1e-12),
        torch.zeros_like(center_union),
    )
    local_valid_count = frame_valid.float().sum().clamp_min(1.0)
    local_center_count = center_count.clamp_min(1.0)
    predicted_presence = torch.sigmoid(presence_logits) >= 0.5
    return total, {
        "loss_center_aux_raw": center_raw.detach(),
        "loss_center_aux_weighted": center_weighted.detach(),
        "loss_center_aux_presence_raw": presence_raw.detach(),
        "loss_center_aux_presence_weighted": presence_weighted.detach(),
        "loss_center_aux_total": total.detach(),
        "loss_center_aux_full_ramp": (
            center_raw * float(config.center_aux_loss_weight)
            + presence_raw * float(config.center_aux_presence_loss_weight)
        ).detach(),
        "diagnostics_center_aux_eligible_fraction": (
            center_count / local_valid_count
        ).detach(),
        "diagnostics_center_aux_threshold_mean": (
            threshold.masked_fill(~center_eligible, 0.0).sum()
            / local_center_count
        ).detach(),
        "diagnostics_center_aux_target_support_fraction": (
            ((heat > 0.0) & vertex_mask).float().sum()
            / vertex_mask.float().sum().clamp_min(1.0)
        ).detach(),
        "diagnostics_center_aux_distribution_viou": (
            center_viou.masked_fill(~center_eligible, 0.0).sum()
            / local_center_count
        ).detach(),
        "diagnostics_center_aux_presence_positive_fraction": (
            (presence_target * frame_valid.float()).sum()
            / local_valid_count
        ).detach(),
        "diagnostics_center_aux_presence_accuracy": (
            (
                predicted_presence
                == (presence_target >= 0.5)
            ).float()
            * frame_valid.float()
        ).sum().div(local_valid_count).detach(),
        "diagnostics_center_aux_presence_predicted_fraction": (
            (predicted_presence.float() * frame_valid.float()).sum()
            / local_valid_count
        ).detach(),
        "diagnostics_center_aux_ramp": center_logits.new_tensor(ramp),
    }


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
    distributed_reduce: bool = True,
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
        location_loss_raw = global_conditional_mean(
            location_sum,
            location_count_local,
            distributed_reduce=distributed_reduce,
        )
        location_loss_weighted = location_loss_raw * (ramp * config.location_loss_weight)
        total = total + location_loss_weighted
    else:
        # Keep weight=0 numerically identical to the pre-location path and collective-free.
        location_loss_raw = pred.new_zeros(())
        location_loss_weighted = location_loss_raw

    if (
        config.center_loss_weight > 0.0
        or config.center_presence_loss_weight > 0.0
    ):
        center_total, center_losses = compute_center_regularization(
            pred=pred,
            logits=logits,
            target=target,
            vertex_mask=location_mask,
            config=config,
            distributed_reduce=distributed_reduce,
        )
        total = total + center_total
        center_loss_raw = center_losses["loss_center_raw"]
        center_presence_loss_raw = center_losses["loss_center_presence_raw"]
    else:
        center_zero = pred.new_zeros(())
        center_loss_raw = center_zero
        center_presence_loss_raw = center_zero
        center_losses = {
            "loss_center_raw": center_zero,
            "loss_center_weighted": center_zero,
            "loss_center_presence_raw": center_zero,
            "loss_center_presence_weighted": center_zero,
            "loss_center_total": center_zero,
            "diagnostics_center_eligible_fraction": center_zero,
            "diagnostics_center_threshold_mean": center_zero,
            "diagnostics_center_target_support_fraction": center_zero,
            "diagnostics_center_distribution_viou": center_zero,
            "diagnostics_center_presence_positive_fraction": center_zero,
            "diagnostics_center_presence_accuracy": center_zero,
            "diagnostics_center_presence_predicted_fraction": center_zero,
        }

    if config.contact_loss_type != "none" and config.contact_loss_weight > 0.0:
        contact_sum, contact_count_local = _contact_loss_sum_and_count(
            pred,
            target,
            mask,
            config,
        )
        contact_loss_raw = global_conditional_mean(
            contact_sum,
            contact_count_local,
            distributed_reduce=distributed_reduce,
        )
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
        + center_loss_raw * config.center_loss_weight
        + center_presence_loss_raw * config.center_presence_loss_weight
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
        **center_losses,
        "loss_contact_raw": contact_loss_raw.detach(),
        "loss_contact_weighted": contact_loss_weighted.detach(),
        "loss_full_ramp": full_ramp_total.detach(),
        "loss_tactile": total.detach(),
        "loss_ramp": pred.new_tensor(ramp),
    }
