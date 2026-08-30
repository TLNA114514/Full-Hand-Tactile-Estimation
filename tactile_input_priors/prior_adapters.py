"""Feature-level adapters for frozen tactile models.

The adapters in this module deliberately do not predict tactile logits.  They
produce bounded changes in an existing RGB feature representation so a frozen
tactile decoder remains the only image-to-mesh mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_PRIOR_CONTROLS = (
    "real",
    "zero",
    "global_mean",
    "spatial_shuffle",
    "sample_shuffle",
    "context_shuffle",
    "wrong_frame",
    "wrong_query",
    "constant",
)


def _sample_rms(value: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, value.ndim))
    return value.float().pow(2).mean(dim=dimensions, keepdim=True).clamp_min(1e-24).sqrt()


def _broadcast_sample_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 1 or mask.shape[0] != value.shape[0]:
        raise ValueError(
            f"A sample mask must have shape [{value.shape[0]}], got {tuple(mask.shape)}"
        )
    return mask.reshape(mask.shape[0], *((1,) * (value.ndim - 1)))


def _fixed_spatial_permutation(length: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 104729 * int(length))
    return torch.randperm(length, generator=generator).to(device=device)


def per_sample_spatial_permutations(
    sample_uids: Sequence[str],
    length: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Create cheap, deterministic, sample-specific spatial permutations.

    Each sample receives a full random permutation seeded from its stable UID;
    this avoids the learnable fixed permutation used by the historical control.
    """

    if int(length) < 2:
        raise ValueError("A spatial permutation requires at least two positions")
    rows = []
    for sample_uid in sample_uids:
        digest = hashlib.sha256(
            f"{int(seed)}|{sample_uid}|{int(length)}".encode("utf-8")
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
        rows.append(torch.randperm(int(length), generator=generator))
    if not rows:
        return torch.empty((0, int(length)), device=device, dtype=torch.long)
    return torch.stack(rows, dim=0).to(device=device)


def apply_prior_control(
    prior: torch.Tensor,
    control: str = "real",
    *,
    alternate_prior: Optional[torch.Tensor] = None,
    control_index: Optional[torch.Tensor] = None,
    seed: int = 521,
) -> torch.Tensor:
    """Apply a parameter-free control while preserving the prior tensor shape.

    ``alternate_prior`` is required for wrong-frame/query controls.  Explicit
    indices can be supplied for paired, reproducible audits; otherwise spatial
    shuffle uses a fixed permutation and sample shuffle uses a cyclic shift.
    """

    control = str(control).strip().lower()
    if control not in SUPPORTED_PRIOR_CONTROLS:
        raise ValueError(
            f"Unsupported prior control {control!r}; choose one of {SUPPORTED_PRIOR_CONTROLS}"
        )
    if control == "real":
        return prior
    if control == "zero":
        return torch.zeros_like(prior)
    if control == "constant":
        dimensions = tuple(range(prior.ndim))
        return prior.mean(dim=dimensions, keepdim=True).expand_as(prior)
    if control == "global_mean":
        if prior.ndim < 3:
            return prior.mean(dim=0, keepdim=True).expand_as(prior)
        spatial_dimensions = tuple(range(2, prior.ndim))
        return prior.mean(dim=spatial_dimensions, keepdim=True).expand_as(prior)
    if control in ("wrong_frame", "wrong_query"):
        if alternate_prior is None:
            raise ValueError(f"prior control {control!r} requires alternate_prior")
        if alternate_prior.shape != prior.shape:
            raise ValueError(
                f"alternate_prior shape {tuple(alternate_prior.shape)} does not match "
                f"prior shape {tuple(prior.shape)}"
            )
        return alternate_prior
    if control in ("sample_shuffle", "context_shuffle"):
        if prior.shape[0] < 2:
            # A final validation batch may contain one sample. Using a zero
            # prior preserves the negative-control contract without crashing
            # or leaking the matching context into that sample.
            return torch.zeros_like(prior)
        if control_index is None:
            control_index = torch.roll(
                torch.arange(prior.shape[0], device=prior.device), shifts=1
            )
        control_index = control_index.to(device=prior.device, dtype=torch.long)
        if tuple(control_index.shape) != (prior.shape[0],):
            raise ValueError(
                f"sample control index must have shape [{prior.shape[0]}], "
                f"got {tuple(control_index.shape)}"
            )
        return prior.index_select(0, control_index)

    if prior.ndim != 4:
        raise ValueError(
            "spatial_shuffle requires a BCHW prior tensor, "
            f"got shape {tuple(prior.shape)}"
        )
    batch_size, channels, height, width = prior.shape
    flattened = prior.flatten(2)
    if control_index is None:
        control_index = _fixed_spatial_permutation(
            height * width, seed=seed, device=prior.device
        )
    control_index = control_index.to(device=prior.device, dtype=torch.long)
    if control_index.ndim == 1:
        if tuple(control_index.shape) != (height * width,):
            raise ValueError(
                f"spatial control index must have {height * width} entries, "
                f"got {tuple(control_index.shape)}"
            )
        shuffled = flattened.index_select(2, control_index)
    elif control_index.ndim == 2:
        if tuple(control_index.shape) != (batch_size, height * width):
            raise ValueError(
                "per-sample spatial control index must have shape "
                f"[{batch_size}, {height * width}], got {tuple(control_index.shape)}"
            )
        shuffled = torch.gather(
            flattened,
            2,
            control_index[:, None, :].expand(-1, channels, -1),
        )
    else:
        raise ValueError("spatial control index must be one- or two-dimensional")
    return shuffled.reshape_as(prior)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.permute(0, 2, 3, 1)
        value = self.norm(value)
        return value.permute(0, 3, 1, 2).contiguous()


class FeatureRMSClamp(nn.Module):
    """Per-sample RMS budget with a detached reference scale."""

    def __init__(self, budget: float):
        super().__init__()
        if not 0.0 < float(budget) <= 1.0:
            raise ValueError("feature RMS budget must lie in (0, 1]")
        self.budget = float(budget)

    def forward(
        self, delta: torch.Tensor, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        if delta.shape != reference.shape:
            raise ValueError(
                f"delta/reference shapes differ: {tuple(delta.shape)} vs {tuple(reference.shape)}"
            )
        reference_rms = _sample_rms(reference).detach()
        delta_rms_pre = _sample_rms(delta)
        allowed = self.budget * reference_rms
        scale = torch.clamp(allowed / delta_rms_pre.clamp_min(1e-12), max=1.0)
        bounded = delta * scale.to(dtype=delta.dtype)
        delta_rms_post = _sample_rms(bounded)
        diagnostics = {
            "feature_base_rms": reference_rms.mean(),
            "feature_delta_rms_pre": delta_rms_pre.mean(),
            "feature_delta_rms_post": delta_rms_post.mean(),
            "feature_delta_to_base_rms": (
                delta_rms_post / reference_rms.clamp_min(1e-12)
            ).mean(),
            "feature_budget_clip_rate": (scale < (1.0 - 1e-6)).float().mean(),
        }
        ratio = delta_rms_pre / reference_rms.clamp_min(1e-12)
        budget_penalty = (
            torch.relu(ratio - self.budget) / max(self.budget, 1e-12)
        ).square().mean()
        return bounded, diagnostics, budget_penalty


@dataclass(frozen=True)
class AdapterConfig:
    feature_rms_budget: float = 0.05
    prior_dropout: float = 0.10
    control_seed: int = 521

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.prior_dropout) < 1.0:
            raise ValueError("prior_dropout must lie in [0, 1)")
        if not 0.0 < float(self.feature_rms_budget) <= 1.0:
            raise ValueError("feature_rms_budget must lie in (0, 1]")


class _PriorAdapterBase(nn.Module):
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.rms_clamp = FeatureRMSClamp(config.feature_rms_budget)

    def _drop_prior(
        self,
        prior: torch.Tensor,
        availability: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = prior.shape[0]
        keep = torch.ones(batch_size, device=prior.device, dtype=torch.bool)
        if availability is not None:
            availability = availability.to(device=prior.device)
            if availability.ndim > 1:
                availability = availability.reshape(batch_size, -1).any(dim=1)
            if tuple(availability.shape) != (batch_size,):
                raise ValueError(
                    f"availability must reduce to [{batch_size}], got {tuple(availability.shape)}"
                )
            keep &= availability.bool()
        if self.training and self.config.prior_dropout > 0.0:
            keep &= torch.rand(batch_size, device=prior.device) >= self.config.prior_dropout
        mask = _broadcast_sample_mask(keep.to(dtype=prior.dtype), prior)
        return prior * mask, keep


class DepthSpatialRectificationAdapter(_PriorAdapterBase):
    """RGB-conditioned multiplicative rectification on an aligned depth grid."""

    def __init__(
        self,
        prior_channels: int,
        feature_channels: int = 256,
        hidden_channels: int = 128,
        modulation_max_scale: float = 0.10,
        config: AdapterConfig = AdapterConfig(),
    ):
        super().__init__(config)
        if int(prior_channels) <= 0:
            raise ValueError("prior_channels must be positive")
        if not 0.0 < float(modulation_max_scale) <= 1.0:
            raise ValueError("modulation_max_scale must lie in (0, 1]")
        self.prior_channels = int(prior_channels)
        self.feature_channels = int(feature_channels)
        self.modulation_max_scale = float(modulation_max_scale)
        self.rgb_norm = ChannelLayerNorm(self.feature_channels)
        self.prior_stem = nn.Sequential(
            nn.Conv2d(self.prior_channels, hidden_channels, kernel_size=3, padding=1),
            ChannelLayerNorm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            ChannelLayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.conditioner = nn.Sequential(
            nn.Conv2d(
                self.feature_channels + hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            ChannelLayerNorm(hidden_channels),
            nn.GELU(),
        )
        self.gamma_head = nn.Conv2d(hidden_channels, self.feature_channels, kernel_size=1)
        self.confidence_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        prior: torch.Tensor,
        *,
        valid: Optional[torch.Tensor] = None,
        availability: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        if rgb_feature.ndim != 4 or rgb_feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"rgb_feature must be [B,{self.feature_channels},H,W], "
                f"got {tuple(rgb_feature.shape)}"
            )
        if prior.ndim != 4 or prior.shape[1] != self.prior_channels:
            raise ValueError(
                f"depth prior must be [B,{self.prior_channels},H,W], got {tuple(prior.shape)}"
            )
        if prior.shape[0] != rgb_feature.shape[0]:
            raise ValueError("depth prior and RGB feature batch sizes differ")
        if prior.shape[-2:] != rgb_feature.shape[-2:]:
            prior = F.interpolate(
                prior.float(), size=rgb_feature.shape[-2:], mode="bilinear", align_corners=False
            ).to(dtype=rgb_feature.dtype)
            if valid is not None:
                valid = F.interpolate(
                    valid.float(), size=rgb_feature.shape[-2:], mode="nearest"
                )
        prior, keep = self._drop_prior(prior, availability)
        prior_present = prior.detach().float().abs().sum(dim=1, keepdim=True) > 0.0
        normalized_rgb = self.rgb_norm(rgb_feature)
        depth_feature = self.prior_stem(prior)
        conditioned = self.conditioner(torch.cat([normalized_rgb, depth_feature], dim=1))
        gamma_logits = self.gamma_head(conditioned)
        confidence = torch.sigmoid(self.confidence_head(conditioned))
        if valid is not None:
            if valid.ndim == 3:
                valid = valid[:, None]
            if valid.ndim != 4 or valid.shape[0] != prior.shape[0] or valid.shape[1] != 1:
                raise ValueError("depth valid mask must have shape [B,1,H,W] or [B,H,W]")
            if valid.shape[-2:] != rgb_feature.shape[-2:]:
                valid = F.interpolate(
                    valid.float(), size=rgb_feature.shape[-2:], mode="nearest"
                )
            confidence = confidence * valid.to(device=confidence.device, dtype=confidence.dtype)
        confidence = confidence * _broadcast_sample_mask(
            keep.to(dtype=confidence.dtype), confidence
        )
        confidence = confidence * prior_present.to(dtype=confidence.dtype)
        gamma = self.modulation_max_scale * torch.tanh(gamma_logits)
        raw_delta = confidence * gamma * normalized_rgb
        delta, diagnostics, budget_penalty = self.rms_clamp(raw_delta, rgb_feature)
        diagnostics.update(
            {
                "prior_keep_rate": keep.float().mean(),
                "modulation_abs_mean": gamma.detach().float().abs().mean(),
                "modulation_saturation": (
                    gamma_logits.detach().float().abs() > 3.0
                ).float().mean(),
                "confidence_mean": confidence.detach().float().mean(),
                "confidence_active_fraction": (
                    confidence.detach().float() > 0.5
                ).float().mean(),
            }
        )
        return rgb_feature + delta, diagnostics, {"feature_budget": budget_penalty}


class _DepthCausalAdapterBase(_PriorAdapterBase):
    """Common null-safe depth preprocessing for causal adapters."""

    def __init__(
        self,
        prior_channels: int,
        feature_channels: int,
        hidden_channels: int,
        config: AdapterConfig,
    ):
        super().__init__(config)
        if min(int(prior_channels), int(feature_channels), int(hidden_channels)) <= 0:
            raise ValueError("Depth adapter dimensions must be positive")
        self.prior_channels = int(prior_channels)
        self.feature_channels = int(feature_channels)
        self.hidden_channels = int(hidden_channels)
        self.rgb_norm = ChannelLayerNorm(self.feature_channels)
        self.prior_stem = nn.Sequential(
            nn.Conv2d(self.prior_channels, self.hidden_channels, 3, padding=1, bias=False),
            ChannelLayerNorm(self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1, bias=False),
            ChannelLayerNorm(self.hidden_channels),
            nn.GELU(),
        )

    def _prepare(
        self,
        rgb_feature: torch.Tensor,
        prior: torch.Tensor,
        valid: Optional[torch.Tensor],
        availability: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if rgb_feature.ndim != 4 or rgb_feature.shape[1] != self.feature_channels:
            raise ValueError(
                f"rgb_feature must be [B,{self.feature_channels},H,W], "
                f"got {tuple(rgb_feature.shape)}"
            )
        if prior.ndim != 4 or prior.shape[1] != self.prior_channels:
            raise ValueError(
                f"depth prior must be [B,{self.prior_channels},H,W], got {tuple(prior.shape)}"
            )
        if prior.shape[0] != rgb_feature.shape[0]:
            raise ValueError("depth prior and RGB feature batch sizes differ")
        if prior.shape[-2:] != rgb_feature.shape[-2:]:
            prior = F.interpolate(
                prior.float(), size=rgb_feature.shape[-2:], mode="bilinear", align_corners=False
            ).to(dtype=rgb_feature.dtype)
            if valid is not None:
                valid = F.interpolate(valid.float(), size=rgb_feature.shape[-2:], mode="nearest")
        if valid is None:
            valid = prior.detach().float().abs().sum(dim=1, keepdim=True) > 0.0
        elif valid.ndim == 3:
            valid = valid[:, None]
        if valid.ndim != 4 or valid.shape[:2] != (prior.shape[0], 1):
            raise ValueError("depth valid mask must have shape [B,1,H,W] or [B,H,W]")
        if valid.shape[-2:] != rgb_feature.shape[-2:]:
            valid = F.interpolate(valid.float(), size=rgb_feature.shape[-2:], mode="nearest")
        prior, keep = self._drop_prior(prior, availability)
        valid = valid.to(device=prior.device, dtype=prior.dtype)
        valid = valid * _broadcast_sample_mask(keep.to(dtype=prior.dtype), prior)
        prior = prior * valid
        depth_feature = self.prior_stem(prior) * valid
        return self.rgb_norm(rgb_feature), depth_feature, valid, keep

    @staticmethod
    def _common_diagnostics(
        keep: torch.Tensor,
        valid: torch.Tensor,
        raw_delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "prior_keep_rate": keep.float().mean(),
            "depth_valid_fraction": valid.detach().float().mean(),
            "depth_raw_correction_rms": _sample_rms(raw_delta).mean(),
        }


class DepthCausalFiLMAdapter(_DepthCausalAdapterBase):
    """Depth-only FiLM: RGB is modulated but cannot generate its condition."""

    def __init__(
        self,
        prior_channels: int,
        feature_channels: int = 256,
        hidden_channels: int = 128,
        modulation_max_scale: float = 0.10,
        config: AdapterConfig = AdapterConfig(),
    ):
        super().__init__(prior_channels, feature_channels, hidden_channels, config)
        if not 0.0 < float(modulation_max_scale) <= 1.0:
            raise ValueError("modulation_max_scale must lie in (0, 1]")
        self.modulation_max_scale = float(modulation_max_scale)
        self.affine_head = nn.Conv2d(
            self.hidden_channels, 2 * self.feature_channels, kernel_size=1, bias=False
        )
        self.confidence_head = nn.Conv2d(
            self.hidden_channels, 1, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.affine_head.weight)
        nn.init.zeros_(self.confidence_head.weight)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        prior: torch.Tensor,
        *,
        valid: Optional[torch.Tensor] = None,
        availability: Optional[torch.Tensor] = None,
    ):
        normalized_rgb, depth_feature, valid, keep = self._prepare(
            rgb_feature, prior, valid, availability
        )
        gamma_logits, beta_logits = self.affine_head(depth_feature).chunk(2, dim=1)
        gamma = self.modulation_max_scale * torch.tanh(gamma_logits)
        beta = self.modulation_max_scale * torch.tanh(beta_logits)
        confidence = torch.sigmoid(self.confidence_head(depth_feature)) * valid
        raw_delta = confidence * (gamma * normalized_rgb + beta)
        delta, diagnostics, budget_penalty = self.rms_clamp(raw_delta, rgb_feature)
        diagnostics.update(self._common_diagnostics(keep, valid, raw_delta))
        diagnostics.update(
            {
                "modulation_abs_mean": 0.5
                * (gamma.detach().float().abs().mean() + beta.detach().float().abs().mean()),
                "modulation_saturation": (
                    torch.maximum(gamma_logits.detach().float().abs(), beta_logits.detach().float().abs())
                    > 3.0
                ).float().mean(),
                "confidence_mean": confidence.detach().float().mean(),
                "confidence_spatial_std": confidence.detach().float().flatten(2).std(dim=2).mean(),
            }
        )
        return rgb_feature + delta, diagnostics, {"feature_budget": budget_penalty}


class DepthLocalCrossAttentionAdapter(_DepthCausalAdapterBase):
    """Local RGB-query/depth-value attention with no RGB-only output path."""

    def __init__(
        self,
        prior_channels: int,
        feature_channels: int = 256,
        hidden_channels: int = 128,
        attention_heads: int = 4,
        window_size: int = 5,
        config: AdapterConfig = AdapterConfig(),
    ):
        super().__init__(prior_channels, feature_channels, hidden_channels, config)
        if self.hidden_channels % int(attention_heads):
            raise ValueError("hidden_channels must be divisible by attention_heads")
        if int(window_size) < 1 or int(window_size) % 2 == 0:
            raise ValueError("Depth local-attention window must be a positive odd integer")
        self.attention_heads = int(attention_heads)
        self.window_size = int(window_size)
        self.head_dim = self.hidden_channels // self.attention_heads
        self.query = nn.Conv2d(self.feature_channels, self.hidden_channels, 1, bias=False)
        self.key = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, bias=False)
        self.value = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1, bias=False)
        self.output = nn.Conv2d(self.hidden_channels, self.feature_channels, 1, bias=False)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(self.attention_heads, self.window_size * self.window_size)
        )
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        prior: torch.Tensor,
        *,
        valid: Optional[torch.Tensor] = None,
        availability: Optional[torch.Tensor] = None,
    ):
        normalized_rgb, depth_feature, valid, keep = self._prepare(
            rgb_feature, prior, valid, availability
        )
        batch, _, height, width = normalized_rgb.shape
        positions = height * width
        window_area = self.window_size * self.window_size
        query = self.query(normalized_rgb).reshape(
            batch, self.attention_heads, self.head_dim, positions
        ).permute(0, 1, 3, 2)
        key = F.unfold(
            self.key(depth_feature), self.window_size, padding=self.window_size // 2
        ).reshape(batch, self.attention_heads, self.head_dim, window_area, positions)
        key = key.permute(0, 1, 4, 3, 2)
        value = F.unfold(
            self.value(depth_feature), self.window_size, padding=self.window_size // 2
        ).reshape(batch, self.attention_heads, self.head_dim, window_area, positions)
        value = value.permute(0, 1, 4, 3, 2)
        local_valid = F.unfold(
            valid.float(), self.window_size, padding=self.window_size // 2
        ).transpose(1, 2)[:, None, :, :, None] > 0.5
        scores = (query[:, :, :, None, :] * key).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores + self.relative_position_bias[None, :, None, :].to(dtype=scores.dtype)
        scores = scores.masked_fill(~local_valid.squeeze(-1), -1e4)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=value.dtype)
        weights = weights * local_valid.squeeze(-1).to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attended = (weights[..., None] * value).sum(dim=-2)
        attended = attended.permute(0, 1, 3, 2).reshape(
            batch, self.hidden_channels, height, width
        )
        raw_delta = self.output(attended) * valid
        delta, diagnostics, budget_penalty = self.rms_clamp(raw_delta, rgb_feature)
        diagnostics.update(self._common_diagnostics(keep, valid, raw_delta))
        entropy = -(weights.float().clamp_min(1e-12).log() * weights.float()).sum(dim=-1)
        normalizer = math.log(max(window_area, 2))
        diagnostics.update(
            {
                "attention_entropy": entropy.mean() / normalizer,
                "attention_peak": weights.detach().float().amax(dim=-1).mean(),
                "attention_valid_neighbors": local_valid.detach().float().sum(dim=-2).mean(),
            }
        )
        return rgb_feature + delta, diagnostics, {"feature_budget": budget_penalty}


class VLMLowRankModulationAdapter(_PriorAdapterBase):
    """Bounded multiplicative low-rank modulation of a 512-D bottleneck."""

    def __init__(
        self,
        prior_dim: int,
        feature_dim: int = 512,
        rank: int = 32,
        config: AdapterConfig = AdapterConfig(),
    ):
        super().__init__(config)
        if min(int(prior_dim), int(feature_dim), int(rank)) <= 0:
            raise ValueError("prior_dim, feature_dim, and rank must be positive")
        self.prior_dim = int(prior_dim)
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.prior_coefficients = nn.Sequential(
            nn.LayerNorm(self.prior_dim, elementwise_affine=False),
            nn.Linear(
                self.prior_dim, max(self.rank * 2, 64), bias=False
            ),
            nn.GELU(),
            nn.Linear(max(self.rank * 2, 64), self.rank, bias=False),
            nn.Tanh(),
        )
        self.feature_down = nn.Linear(self.feature_dim, self.rank, bias=False)
        self.feature_up = nn.Linear(self.rank, self.feature_dim, bias=False)
        nn.init.zeros_(self.feature_up.weight)

    def forward(
        self,
        rgb_feature: torch.Tensor,
        prior: torch.Tensor,
        *,
        availability: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        if rgb_feature.ndim != 2 or rgb_feature.shape[1] != self.feature_dim:
            raise ValueError(
                f"RGB bottleneck must be [B,{self.feature_dim}], got {tuple(rgb_feature.shape)}"
            )
        if prior.ndim != 2 or prior.shape != (rgb_feature.shape[0], self.prior_dim):
            raise ValueError(
                f"VLM prior must be [B,{self.prior_dim}], got {tuple(prior.shape)}"
            )
        prior, keep = self._drop_prior(prior, availability)
        coefficients = self.prior_coefficients(prior)
        coefficients = coefficients * keep[:, None].to(dtype=coefficients.dtype)
        factors = self.feature_down(self.feature_norm(rgb_feature))
        raw_delta = self.feature_up(coefficients * factors)
        delta, diagnostics, budget_penalty = self.rms_clamp(raw_delta, rgb_feature)
        diagnostics.update(
            {
                "prior_keep_rate": keep.float().mean(),
                "lowrank_coefficient_rms": coefficients.detach().float().pow(2).mean().sqrt(),
                "lowrank_factor_rms": factors.detach().float().pow(2).mean().sqrt(),
                "lowrank_coefficient_saturation": (
                    coefficients.detach().float().abs() > 0.95
                ).float().mean(),
            }
        )
        return rgb_feature + delta, diagnostics, {"feature_budget": budget_penalty}


def detached_diagnostics(values: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().float() for name, value in values.items()}
