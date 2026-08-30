"""Prior adapters that refine a frozen contact selector, never tactile pressure.

Depth is allowed to contribute spatial evidence.  VLM context is deliberately
restricted to a shared calibration function over already-local RGB evidence;
it cannot learn an independent per-vertex pressure or contact template.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prior_adapters import ChannelLayerNorm, FeatureRMSClamp


SUPPORTED_SELECTOR_PRIOR_ADAPTERS = (
    "depth_mapping_rectifier",
    "depth_anchor_residual",
    "depth_anchor_query",
    "vlm_global_calibrator",
)


def _sample_mask(
    availability: Optional[torch.Tensor], reference: torch.Tensor
) -> torch.Tensor:
    if availability is None:
        return reference.new_ones((reference.shape[0], 1, 1, 1))
    value = availability.to(device=reference.device, dtype=reference.dtype).reshape(-1)
    if value.shape[0] != reference.shape[0]:
        raise ValueError(
            f"Prior availability must have {reference.shape[0]} entries, got {value.shape[0]}"
        )
    return value[:, None, None, None]


def _depth_validity(
    prior: torch.Tensor, explicit_valid: Optional[torch.Tensor]
) -> torch.Tensor:
    if explicit_valid is not None:
        valid = explicit_valid
        if valid.ndim == 3:
            valid = valid[:, None]
        if valid.ndim != 4 or valid.shape[0] != prior.shape[0]:
            raise ValueError("depth_valid must be a B1HW/BHW tensor")
        valid = F.interpolate(valid.float(), prior.shape[-2:], mode="nearest")
        return valid.to(device=prior.device, dtype=prior.dtype).clamp(0.0, 1.0)
    # MoGe sidecars store validity in channel four.  Other teachers can pass
    # depth_valid explicitly and are not required to follow this convention.
    if prior.shape[1] >= 5:
        return prior[:, 4:5].clamp(0.0, 1.0)
    return prior.new_ones((prior.shape[0], 1, *prior.shape[-2:]))


class DepthStem(nn.Module):
    def __init__(self, prior_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                int(prior_channels), int(hidden_channels), 3, padding=1, bias=False
            ),
            ChannelLayerNorm(int(hidden_channels)),
            nn.GELU(),
            nn.Conv2d(
                int(hidden_channels), int(hidden_channels), 3, padding=1, bias=False
            ),
            ChannelLayerNorm(int(hidden_channels)),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
        )

    def forward(self, prior: torch.Tensor) -> torch.Tensor:
        return self.net(prior)


class DepthMappingRectifier(nn.Module):
    """Rectify the frozen RGB selector neck, then reuse its frozen mapping."""

    def __init__(
        self,
        *,
        prior_channels: int,
        neck_channels: int,
        hidden_channels: int = 128,
        feature_rms_budget: float = 0.05,
        modulation_max_scale: float = 0.10,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stem = DepthStem(prior_channels, hidden_channels, dropout)
        self.modulator = nn.Conv2d(
            hidden_channels, 2 * neck_channels + 1, 1, bias=False
        )
        nn.init.zeros_(self.modulator.weight)
        self.rgb_norm = ChannelLayerNorm(neck_channels)
        self.rms_clamp = FeatureRMSClamp(feature_rms_budget)
        self.modulation_max_scale = float(modulation_max_scale)

    def forward(
        self,
        base_neck: torch.Tensor,
        prior: torch.Tensor,
        *,
        valid: Optional[torch.Tensor] = None,
        availability: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if prior.ndim != 4:
            raise ValueError(f"Depth prior must be BCHW, got {tuple(prior.shape)}")
        if tuple(prior.shape[-2:]) != tuple(base_neck.shape[-2:]):
            prior = F.interpolate(prior, base_neck.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.stem(prior) - self.stem(torch.zeros_like(prior))
        gamma, beta, confidence_logit = torch.split(
            self.modulator(hidden), (base_neck.shape[1], base_neck.shape[1], 1), dim=1
        )
        validity = _depth_validity(prior, valid)
        confidence = torch.sigmoid(confidence_logit) * validity
        confidence = confidence * _sample_mask(availability, base_neck)
        delta = confidence * (
            self.modulation_max_scale * torch.tanh(gamma) * self.rgb_norm(base_neck)
            + beta
        )
        bounded, diagnostics, budget_penalty = self.rms_clamp(delta, base_neck)
        diagnostics.update(
            {
                "prior_confidence_mean": confidence.detach().float().mean(),
                "prior_valid_fraction": validity.detach().float().mean(),
                "prior_hidden_rms": hidden.detach().float().pow(2).mean().sqrt(),
            }
        )
        return base_neck + bounded, diagnostics, {"feature_budget": budget_penalty}


class DepthAnchorResidual(nn.Module):
    """Predict bounded anchor-logit corrections from RGB and Depth evidence."""

    def __init__(
        self,
        *,
        prior_channels: int,
        neck_channels: int,
        grid_size: Tuple[int, int],
        anchor_count: int,
        hidden_channels: int = 128,
        hidden_dim: int = 512,
        residual_max_logit: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stem = DepthStem(prior_channels, hidden_channels, dropout)
        self.fusion = nn.Sequential(
            nn.Conv2d(neck_channels + hidden_channels + 1, hidden_channels, 1),
            ChannelLayerNorm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            ChannelLayerNorm(hidden_channels),
            nn.GELU(),
        )
        flattened = hidden_channels * int(grid_size[0]) * int(grid_size[1])
        self.decoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(flattened, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(anchor_count)),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.residual_max_logit = float(residual_max_logit)

    def forward(
        self,
        base_neck: torch.Tensor,
        prior: torch.Tensor,
        *,
        valid: Optional[torch.Tensor] = None,
        availability: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if tuple(prior.shape[-2:]) != tuple(base_neck.shape[-2:]):
            prior = F.interpolate(prior, base_neck.shape[-2:], mode="bilinear", align_corners=False)
        depth = self.stem(prior) - self.stem(torch.zeros_like(prior))
        validity = _depth_validity(prior, valid)
        available = _sample_mask(availability, base_neck)
        fused = self.fusion(
            torch.cat((base_neck.detach(), depth, validity), dim=1)
        )
        baseline_fused = self.fusion(
            torch.cat(
                (
                    base_neck.detach(),
                    torch.zeros_like(depth),
                    torch.zeros_like(validity),
                ),
                dim=1,
            )
        )
        raw = self.decoder(fused) - self.decoder(baseline_fused)
        residual = self.residual_max_logit * torch.tanh(
            raw / max(self.residual_max_logit, 1e-6)
        )
        residual = residual * available.flatten(1)[:, :1]
        diagnostics = {
            "anchor_delta_rms": residual.detach().float().pow(2).mean().sqrt(),
            "anchor_delta_abs_max": residual.detach().float().abs().max(),
            "anchor_delta_saturation": (
                raw.detach().float().abs() > 3.0 * self.residual_max_logit
            ).float().mean(),
            "prior_valid_fraction": validity.detach().float().mean(),
        }
        return residual, diagnostics, {}


class _AnchorQueryBlock(nn.Module):
    """One canonical-query to aligned image-token attention block."""

    def __init__(self, dimension: int, heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(int(dimension))
        self.token_norm = nn.LayerNorm(int(dimension))
        self.attention = nn.MultiheadAttention(
            int(dimension),
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(int(dimension))
        self.ffn = nn.Sequential(
            nn.Linear(int(dimension), 2 * int(dimension)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * int(dimension), int(dimension)),
        )

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_tokens = self.token_norm(tokens)
        attended, _ = self.attention(
            normalized_queries,
            normalized_tokens,
            normalized_tokens,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class DepthAnchorQuerySelector(nn.Module):
    """Read aligned RGB/Depth tokens with fixed canonical anchor queries.

    The branch predicts contact-logit corrections and an independent false-high
    score at 512 canonical anchors.  It never emits pressure.  Canonical XYZ is
    a fixed output coordinate only; no per-frame MANO pose or image-to-MANO
    correspondence is consumed.
    """

    def __init__(
        self,
        *,
        prior_channels: int,
        neck_channels: int,
        anchor_coordinates: torch.Tensor,
        grid_size: Tuple[int, int],
        hidden_channels: int = 128,
        query_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        residual_max_logit: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        query_dim = int(query_dim)
        if query_dim <= 0 or int(heads) <= 0 or query_dim % int(heads):
            raise ValueError("query_dim must be positive and divisible by heads")
        if int(layers) < 1:
            raise ValueError("Depth anchor query selector needs at least one layer")
        coordinates = torch.as_tensor(anchor_coordinates, dtype=torch.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("anchor_coordinates must have shape [A,3]")
        centered = coordinates - coordinates.mean(dim=0, keepdim=True)
        centered = centered / centered.square().sum(dim=1).sqrt().amax().clamp_min(1e-6)
        self.register_buffer("anchor_coordinates", centered)
        self.anchor_count = int(centered.shape[0])
        self.residual_max_logit = float(residual_max_logit)

        self.depth_stem = DepthStem(prior_channels, hidden_channels, dropout)
        self.rgb_key = nn.Conv2d(neck_channels, query_dim, 1, bias=False)
        self.depth_key = nn.Conv2d(hidden_channels, query_dim, 1, bias=False)
        self.depth_value = nn.Conv2d(hidden_channels, query_dim, 1, bias=False)
        self.position = nn.Parameter(
            torch.empty(1, query_dim, int(grid_size[0]), int(grid_size[1]))
        )
        self.learned_queries = nn.Parameter(
            torch.empty(self.anchor_count, query_dim)
        )
        self.xyz_projection = nn.Sequential(
            nn.Linear(3, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.base_evidence_projection = nn.Linear(2, query_dim, bias=False)
        self.blocks = nn.ModuleList(
            _AnchorQueryBlock(query_dim, int(heads), float(dropout))
            for _ in range(int(layers))
        )
        self.contact_head = nn.Linear(query_dim, 1)
        self.false_high_head = nn.Linear(query_dim, 1)
        nn.init.normal_(self.learned_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.zeros_(self.contact_head.weight)
        nn.init.zeros_(self.contact_head.bias)
        nn.init.normal_(self.false_high_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.false_high_head.bias)

    @staticmethod
    def _flatten_tokens(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(2).transpose(1, 2).contiguous()

    def forward(
        self,
        base_neck: torch.Tensor,
        base_anchor_logits: torch.Tensor,
        base_anchor_pressure: torch.Tensor,
        prior: torch.Tensor,
        *,
        valid: Optional[torch.Tensor] = None,
        availability: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        if tuple(prior.shape[-2:]) != tuple(base_neck.shape[-2:]):
            prior = F.interpolate(
                prior, base_neck.shape[-2:], mode="bilinear", align_corners=False
            )
        validity = _depth_validity(prior, valid)
        available = _sample_mask(availability, base_neck)
        depth = self.depth_stem(prior) - self.depth_stem(torch.zeros_like(prior))
        depth = depth * validity * available

        # RGB determines where to look; Depth supplies the value being read.
        # With unavailable/zero Depth the value tokens are exactly zero, so the
        # branch cannot silently become another RGB-only global decoder.
        key_tokens = self._flatten_tokens(
            self.rgb_key(base_neck.detach())
            + self.depth_key(depth)
            + self.position
        )
        value_tokens = self._flatten_tokens(self.depth_value(depth))
        evidence = torch.stack(
            (
                base_anchor_logits.detach().clamp(-12.0, 12.0) / 12.0,
                base_anchor_pressure.detach().clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        queries = (
            self.learned_queries[None]
            + self.xyz_projection(self.anchor_coordinates)[None]
            + self.base_evidence_projection(evidence)
        )
        for block in self.blocks:
            normalized_queries = block.query_norm(queries)
            normalized_keys = block.token_norm(key_tokens)
            attended, _ = block.attention(
                normalized_queries,
                normalized_keys,
                value_tokens,
                need_weights=False,
            )
            queries = queries + attended
            queries = queries + block.ffn(block.ffn_norm(queries))

        raw_contact = self.contact_head(queries).squeeze(-1)
        contact_residual = self.residual_max_logit * torch.tanh(
            raw_contact / max(self.residual_max_logit, 1e-6)
        )
        false_high_logits = self.false_high_head(queries).squeeze(-1)
        sample_available = available.flatten(1)[:, :1]
        contact_residual = contact_residual * sample_available
        false_high_logits = false_high_logits * sample_available
        query_centered = queries.detach().float() - queries.detach().float().mean(
            dim=1, keepdim=True
        )
        diagnostics = {
            "anchor_delta_rms": contact_residual.detach().float().pow(2).mean().sqrt(),
            "anchor_delta_abs_max": contact_residual.detach().float().abs().max(),
            "anchor_delta_saturation": (
                raw_contact.detach().float().abs() > 3.0 * self.residual_max_logit
            ).float().mean(),
            "false_high_logit_rms": false_high_logits.detach().float().pow(2).mean().sqrt(),
            "query_diversity_rms": query_centered.pow(2).mean().sqrt(),
            "depth_value_rms": value_tokens.detach().float().pow(2).mean().sqrt(),
            "prior_valid_fraction": validity.detach().float().mean(),
        }
        return contact_residual, false_high_logits, diagnostics, {}


class VLMGlobalContactCalibrator(nn.Module):
    """Globally calibrate local RGB evidence without creating spatial identity."""

    def __init__(
        self,
        *,
        prior_dim: int,
        local_dim: int = 64,
        rank: int = 32,
        residual_max_logit: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.local_encoder = nn.Sequential(
            nn.Linear(2, local_dim),
            nn.LayerNorm(local_dim),
            nn.GELU(),
        )
        self.conditioner = nn.Sequential(
            nn.LayerNorm(int(prior_dim)),
            nn.Linear(int(prior_dim), int(rank), bias=False),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(rank), 2 * int(local_dim), bias=False),
        )
        self.output = nn.Linear(int(local_dim), 1)
        nn.init.zeros_(self.conditioner[-1].weight)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.output.bias)
        self.residual_max_logit = float(residual_max_logit)

    def forward(
        self,
        base_anchor_logits: torch.Tensor,
        base_anchor_pressure: torch.Tensor,
        prior: torch.Tensor,
        *,
        availability: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if prior.ndim != 2:
            raise ValueError(f"VLM prior must be [B,D], got {tuple(prior.shape)}")
        local = torch.stack(
            (
                base_anchor_logits.detach().clamp(-12.0, 12.0) / 12.0,
                base_anchor_pressure.detach().clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        hidden = self.local_encoder(local)
        gamma, beta = self.conditioner(prior).chunk(2, dim=-1)
        modulated = hidden * (1.0 + 0.1 * torch.tanh(gamma[:, None])) + beta[:, None]
        # The subtraction forbids an RGB-only residual: with a zero context
        # conditioner, the branch is exactly the frozen local selector while
        # the small output initialization still gives the conditioner gradient.
        raw = (
            self.output(F.gelu(modulated))
            - self.output(F.gelu(hidden))
        ).squeeze(-1)
        residual = self.residual_max_logit * torch.tanh(
            raw / max(self.residual_max_logit, 1e-6)
        )
        if availability is not None:
            residual = residual * availability.to(residual).reshape(-1, 1)
        diagnostics = {
            "anchor_delta_rms": residual.detach().float().pow(2).mean().sqrt(),
            "anchor_delta_abs_max": residual.detach().float().abs().max(),
            "vlm_gamma_rms": gamma.detach().float().pow(2).mean().sqrt(),
            "vlm_beta_rms": beta.detach().float().pow(2).mean().sqrt(),
        }
        return residual, diagnostics, {}
