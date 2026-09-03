"""Feature-grid temporal adapter for the frozen FullGrid tactile decoder."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from hamer_tactile_ft.hamer_tactile import (
    CANONICAL_MODEL_INITIALIZATION_ORDER,
    DinoTactileModel,
    _build_dense_decoder,
    parse_input_resolution,
)

from .runtime import _checkpoint_model_config, load_torch_checkpoint


TEMPORAL_GRID_FORMAT = "tactile_temporal_grid_adapter_v1"
TEMPORAL_GRID_RESUME_FORMAT = "tactile_temporal_grid_adapter_resume_v1"
TEMPORAL_MEMORY_FORMAT = "tactile_temporal_memory_fusion_v2"
TEMPORAL_MEMORY_RESUME_FORMAT = "tactile_temporal_memory_fusion_resume_v2"
TEMPORAL_TRUNK_FORMAT = "tactile_temporal_main_trunk_v3"
TEMPORAL_TRUNK_RESUME_FORMAT = "tactile_temporal_main_trunk_resume_v3"
TEMPORAL_CLIP_FORMAT = "tactile_temporal_causal_clip_trunk_v4"
TEMPORAL_CLIP_RESUME_FORMAT = "tactile_temporal_causal_clip_trunk_resume_v4"
TEMPORAL_FULLGRID_FORMAT = "tactile_temporal_fullgrid6144_trunk_v5"
TEMPORAL_FULLGRID_RESUME_FORMAT = "tactile_temporal_fullgrid6144_trunk_resume_v5"
TEMPORAL_ONLINEHMR_FORMAT = "tactile_temporal_onlinehmr_patch_kv_trunk_v6"
TEMPORAL_ONLINEHMR_RESUME_FORMAT = (
    "tactile_temporal_onlinehmr_patch_kv_trunk_resume_v6"
)
TEMPORAL_ARCHITECTURES = (
    "grid_difference_v1",
    "local_memory_v2",
    "hierarchical_memory_v3",
    "causal_clip_transformer_v4",
    "fullgrid6144_bidirectional_v5",
    "onlinehmr_patch_kv_v6",
)
TEMPORAL_GRID_SOURCES = (
    "rgb_reset",
    "real",
    "cross_sequence",
    "contralateral",
    "lag_reverse",
    "spatial_shuffle",
    "affine_perturb",
)


def module_state_sha256(module: nn.Module) -> str:
    """Hash module state using the canonical initialization audit encoding."""

    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_lags(history_lags: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in history_lags)
    if not values or any(value <= 0 for value in values):
        raise ValueError("history_lags must contain positive integers")
    if len(set(values)) != len(values):
        raise ValueError("history_lags must be unique")
    if tuple(sorted(values)) != values:
        raise ValueError("history_lags must be ordered from nearest to farthest")
    return values


def _decoder_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", {})
    if not isinstance(state, Mapping):
        raise ValueError("Base checkpoint state_dict is not a mapping")
    prefix = "tactile_head.decoder."
    result = {
        str(name)[len(prefix) :]: value
        for name, value in state.items()
        if str(name).startswith(prefix)
    }
    if not result:
        raise ValueError("Base checkpoint does not contain a FullGrid decoder")
    # Compact checkpoints predating the projection rename remain reproducible.
    for old_prefix, new_prefix in (("0.project.", "0.projection."),):
        for old_name in tuple(result):
            if not old_name.startswith(old_prefix):
                continue
            new_name = new_prefix + old_name[len(old_prefix) :]
            if new_name not in result:
                result[new_name] = result.pop(old_name)
    return result


def load_frozen_fullgrid_decoder(
    checkpoint_path: str | Path,
) -> tuple[nn.Sequential, dict[str, Any]]:
    """Rebuild only the cached-grid decoder without constructing frozen DINO."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    if checkpoint.get("format") != "tactile_trainable_v2":
        raise ValueError(
            "Temporal grid adapters require compact format=tactile_trainable_v2"
        )
    config = _checkpoint_model_config(checkpoint)
    if str(config.get("pool_layout", "")) != "fullgrid32":
        raise ValueError("Temporal grid adapters require a FullGrid32 base")
    input_resolution = parse_input_resolution(
        config.get("input_resolution", (256, 192))
    )
    grid_size = tuple(int(value) // 16 for value in input_resolution)
    state = _decoder_state(checkpoint)
    output_weight = state.get("7.weight")
    projection_weight = state.get("0.projection.0.weight")
    if output_weight is None or projection_weight is None:
        raise ValueError("Base checkpoint has an unsupported FullGrid decoder layout")
    tactile_dim = int(output_weight.shape[0])
    grid_channels = int(projection_weight.shape[1])
    decoder, _, _ = _build_dense_decoder(
        tactile_dim=tactile_dim,
        channels=grid_channels,
        pool_layout="fullgrid32",
        grid_size=grid_size,
        pool_output_channels=int(config.get("pool_output_channels", 32)),
        decoder_hidden_dim=int(config.get("decoder_hidden_dim", 512)),
        dropout_scale=float(config.get("decoder_dropout_scale", 1.0)),
    )
    decoder.load_state_dict(state, strict=True)
    decoder.requires_grad_(False)
    decoder.eval()
    metadata = {
        "tactile_dim": tactile_dim,
        "grid_channels": grid_channels,
        "grid_size": list(grid_size),
        "input_resolution": list(input_resolution),
        "pool_output_channels": int(config.get("pool_output_channels", 32)),
        "decoder_hidden_dim": int(config.get("decoder_hidden_dim", 512)),
    }
    return decoder, metadata


class TemporalGridAdapterV1(nn.Module):
    """Causal, zero-motion-exact residual on a fused DINO feature grid."""

    def __init__(
        self,
        *,
        grid_channels: int = 256,
        hidden_channels: int = 64,
        history_lags: Sequence[int] = (1, 2, 4, 8),
        nominal_fps: float = 30.0,
        temporal_kernel_size: int = 3,
        feature_rms_budget: float = 0.05,
    ):
        super().__init__()
        self.grid_channels = int(grid_channels)
        self.hidden_channels = int(hidden_channels)
        self.history_lags = _validate_lags(history_lags)
        self.nominal_fps = float(nominal_fps)
        self.temporal_kernel_size = min(
            int(temporal_kernel_size), len(self.history_lags)
        )
        self.feature_rms_budget = float(feature_rms_budget)
        if self.grid_channels <= 0 or self.hidden_channels <= 0:
            raise ValueError("grid_channels and hidden_channels must be positive")
        if self.nominal_fps <= 0.0:
            raise ValueError("nominal_fps must be positive")
        if self.temporal_kernel_size <= 0:
            raise ValueError("temporal_kernel_size must be positive")
        if self.feature_rms_budget <= 0.0:
            raise ValueError("feature_rms_budget must be positive")

        self.shared_projection = nn.Conv2d(
            self.grid_channels, self.hidden_channels, kernel_size=1, bias=False
        )
        pair_channels = self.hidden_channels * 3 + 2
        self.pair_mixer = nn.Conv2d(
            pair_channels, self.hidden_channels, kernel_size=1, bias=True
        )
        self.temporal_mixer = nn.Conv3d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=(self.temporal_kernel_size, 1, 1),
            groups=self.hidden_channels,
            bias=False,
        )
        self.spatial_mixer = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            groups=self.hidden_channels,
            bias=False,
        )
        self.output_projection = nn.Conv2d(
            self.hidden_channels, self.grid_channels, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.output_projection.weight)

    def config(self) -> dict[str, Any]:
        return {
            "grid_channels": self.grid_channels,
            "hidden_channels": self.hidden_channels,
            "history_lags": list(self.history_lags),
            "nominal_fps": self.nominal_fps,
            "temporal_kernel_size": self.temporal_kernel_size,
            "feature_rms_budget": self.feature_rms_budget,
        }

    def _validate_inputs(
        self,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_time_gap: torch.Tensor,
        history_available: torch.Tensor,
    ) -> None:
        if current_grid.ndim != 4 or current_grid.shape[1] != self.grid_channels:
            raise ValueError(
                f"current_grid must be [B,{self.grid_channels},H,W], got "
                f"{tuple(current_grid.shape)}"
            )
        expected = (
            current_grid.shape[0],
            len(self.history_lags),
            self.grid_channels,
            current_grid.shape[2],
            current_grid.shape[3],
        )
        if tuple(history_grids.shape) != expected:
            raise ValueError(
                f"history_grids must be {expected}, got {tuple(history_grids.shape)}"
            )
        scalar_shape = (current_grid.shape[0], len(self.history_lags))
        if tuple(history_time_gap.shape) != scalar_shape:
            raise ValueError("history_time_gap has an incompatible shape")
        if tuple(history_available.shape) != scalar_shape:
            raise ValueError("history_available has an incompatible shape")

    def forward(
        self,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_time_gap: torch.Tensor,
        history_available: torch.Tensor,
        history_crop_transform: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del history_crop_transform
        self._validate_inputs(
            current_grid, history_grids, history_time_gap, history_available
        )
        batch, lag_count, _, height, width = history_grids.shape
        current_projected = self.shared_projection(current_grid)
        history_projected = self.shared_projection(
            history_grids.reshape(batch * lag_count, self.grid_channels, height, width)
        ).reshape(batch, lag_count, self.hidden_channels, height, width)
        difference = history_projected - current_projected[:, None]
        absolute_difference = difference.abs()
        available = history_available.to(
            device=current_grid.device, dtype=current_projected.dtype
        ).clamp(0.0, 1.0)
        time_gap = history_time_gap.to(
            device=current_grid.device, dtype=current_projected.dtype
        ).clamp_min(0.0)
        maximum_gap = max(self.history_lags) / self.nominal_fps
        time_encoding = torch.log1p(time_gap * self.nominal_fps) / math.log1p(
            max(maximum_gap * self.nominal_fps, 1.0)
        )
        current_repeated = current_projected[:, None].expand(-1, lag_count, -1, -1, -1)
        scalar_shape = (batch, lag_count, 1, height, width)
        time_map = time_encoding[:, :, None, None, None].expand(scalar_shape)
        availability_map = available[:, :, None, None, None].expand(scalar_shape)
        real_pair = torch.cat(
            (
                current_repeated,
                difference,
                absolute_difference,
                time_map,
                availability_map,
            ),
            dim=2,
        )
        zero_pair = torch.cat(
            (
                current_repeated,
                torch.zeros_like(difference),
                torch.zeros_like(absolute_difference),
                time_map,
                availability_map,
            ),
            dim=2,
        )
        pair_shape = (batch * lag_count, real_pair.shape[2], height, width)
        encoded = F.gelu(self.pair_mixer(real_pair.reshape(pair_shape)))
        reference = F.gelu(self.pair_mixer(zero_pair.reshape(pair_shape)))
        encoded = (encoded - reference).reshape(
            batch, lag_count, self.hidden_channels, height, width
        )
        encoded = encoded * availability_map

        # Inputs arrive nearest-to-farthest. Reverse them so every temporal
        # convolution sees only older/equal evidence while moving toward L1.
        chronological = encoded.flip(1).permute(0, 2, 1, 3, 4)
        chronological = F.pad(
            chronological,
            (0, 0, 0, 0, self.temporal_kernel_size - 1, 0),
        )
        mixed = self.temporal_mixer(chronological)
        mixed = mixed.permute(0, 2, 1, 3, 4).flip(1)
        valid_count = available.sum(dim=1, keepdim=True).clamp_min(1.0)
        aggregated = (
            mixed * availability_map
        ).sum(dim=1) / valid_count[:, :, None, None]
        aggregated = F.gelu(self.spatial_mixer(aggregated))
        raw_delta = self.output_projection(aggregated)

        dimensions = tuple(range(1, current_grid.ndim))
        base_energy = current_grid.float().pow(2).mean(
            dim=dimensions, keepdim=True
        )
        raw_energy = raw_delta.float().pow(2).mean(
            dim=dimensions, keepdim=True
        )
        base_rms = torch.sqrt(base_energy + 1e-12)
        # The clamp is a safety constraint, not a learned normalization. At
        # zero-init, differentiating sqrt(mean(delta^2)) at delta=0 produces an
        # infinite derivative and poisons the first optimizer step. Use a
        # smooth denominator and keep the clamp coefficient outside autograd.
        raw_rms_for_scale = torch.sqrt(raw_energy + 1e-12)
        maximum_rms = self.feature_rms_budget * base_rms
        scale = torch.clamp(maximum_rms / raw_rms_for_scale, max=1.0).detach()
        bounded_delta = raw_delta * scale.to(dtype=raw_delta.dtype)
        fused_grid = current_grid + bounded_delta
        motion_rms = (
            difference.detach().float().pow(2).mean(dim=(2, 3, 4)).sqrt()
            * available.float()
        ).sum(dim=1) / available.float().sum(dim=1).clamp_min(1.0)
        diagnostics = {
            "feature_delta_rms_raw": raw_energy.detach()
            .sqrt()
            .flatten(1)
            .mean(dim=1),
            "feature_delta_rms": bounded_delta.detach().float()
            .pow(2)
            .mean(dim=dimensions)
            .sqrt(),
            "feature_base_rms": base_rms.detach().flatten(1).mean(dim=1),
            "feature_clamp_scale": scale.flatten(1).mean(dim=1),
            "history_available_fraction": available.detach().mean(dim=1),
            "history_motion_rms": motion_rms,
        }
        return fused_grid, diagnostics


class TemporalLocalMemoryFusionV2(nn.Module):
    """Affine-guided image-token memory fusion with an explicit null match.

    Matching is deliberately computed from frozen normalized DINO-ReZero
    features. The trainable path only decides how verified image-to-image
    evidence changes the current representation; it cannot move the matcher to
    exploit the pressure objective.
    """

    def __init__(
        self,
        *,
        grid_channels: int = 256,
        hidden_channels: int = 64,
        history_lags: Sequence[int] = (1, 2),
        nominal_fps: float = 30.0,
        patch_size: int = 16,
        search_window: int = 5,
        match_temperature: float = 0.07,
        null_similarity: float = 0.40,
        feature_rms_budget: float = 0.05,
    ):
        super().__init__()
        self.grid_channels = int(grid_channels)
        self.hidden_channels = int(hidden_channels)
        self.history_lags = _validate_lags(history_lags)
        self.nominal_fps = float(nominal_fps)
        self.patch_size = int(patch_size)
        self.search_window = int(search_window)
        self.match_temperature = float(match_temperature)
        self.null_similarity = float(null_similarity)
        self.feature_rms_budget = float(feature_rms_budget)
        if self.grid_channels <= 0 or self.hidden_channels <= 0:
            raise ValueError("grid_channels and hidden_channels must be positive")
        if self.nominal_fps <= 0.0 or self.patch_size <= 0:
            raise ValueError("nominal_fps and patch_size must be positive")
        if self.search_window <= 0 or self.search_window % 2 != 1:
            raise ValueError("search_window must be a positive odd integer")
        if self.match_temperature <= 0.0:
            raise ValueError("match_temperature must be positive")
        if self.feature_rms_budget <= 0.0:
            raise ValueError("feature_rms_budget must be positive")

        pair_channels = self.grid_channels * 3 + 6
        self.pair_mixer = nn.Conv2d(
            pair_channels, self.hidden_channels, kernel_size=1, bias=True
        )
        self.spatial_mixer = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            groups=self.hidden_channels,
            bias=False,
        )
        self.output_projection = nn.Conv2d(
            self.hidden_channels, self.grid_channels, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.output_projection.weight)

    def config(self) -> dict[str, Any]:
        return {
            "grid_channels": self.grid_channels,
            "hidden_channels": self.hidden_channels,
            "history_lags": list(self.history_lags),
            "nominal_fps": self.nominal_fps,
            "patch_size": self.patch_size,
            "search_window": self.search_window,
            "match_temperature": self.match_temperature,
            "null_similarity": self.null_similarity,
            "feature_rms_budget": self.feature_rms_budget,
        }

    def _validate_inputs(
        self,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_time_gap: torch.Tensor,
        history_available: torch.Tensor,
        history_crop_transform: torch.Tensor | None,
    ) -> torch.Tensor:
        if current_grid.ndim != 4 or current_grid.shape[1] != self.grid_channels:
            raise ValueError(
                f"current_grid must be [B,{self.grid_channels},H,W], got "
                f"{tuple(current_grid.shape)}"
            )
        expected = (
            current_grid.shape[0],
            len(self.history_lags),
            self.grid_channels,
            current_grid.shape[2],
            current_grid.shape[3],
        )
        if tuple(history_grids.shape) != expected:
            raise ValueError(
                f"history_grids must be {expected}, got {tuple(history_grids.shape)}"
            )
        scalar_shape = (current_grid.shape[0], len(self.history_lags))
        if tuple(history_time_gap.shape) != scalar_shape:
            raise ValueError("history_time_gap has an incompatible shape")
        if tuple(history_available.shape) != scalar_shape:
            raise ValueError("history_available has an incompatible shape")
        if history_crop_transform is None:
            raise ValueError("local_memory_v2 requires history_crop_transform")
        transform_shape = (*scalar_shape, 3, 3)
        if tuple(history_crop_transform.shape) != transform_shape:
            raise ValueError(
                f"history_crop_transform must be {transform_shape}, got "
                f"{tuple(history_crop_transform.shape)}"
            )
        return history_crop_transform

    def _local_matches(
        self,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_available: torch.Tensor,
        history_crop_transform: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, lag_count, channels, height, width = history_grids.shape
        device = current_grid.device
        available = history_available.float().clamp(0.0, 1.0)

        # grid_sample and cosine matching are fixed geometry, not a trainable
        # path. Float32 also avoids CUDA autocast dtype failures in grid_sample.
        with torch.autocast(device_type=device.type, enabled=False):
            current = F.normalize(current_grid.detach().float(), dim=1, eps=1e-6)
            history = F.normalize(history_grids.detach().float(), dim=2, eps=1e-6)
            transform = history_crop_transform.detach().float()
            y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5)
            x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5)
            y = y * float(self.patch_size) - 0.5
            x = x * float(self.patch_size) - 0.5
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            coordinates = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
            mapped = torch.einsum("blij,hwj->blihw", transform, coordinates)
            denominator = mapped[:, :, 2].clamp_min(1e-6)
            mapped_x = mapped[:, :, 0] / denominator
            mapped_y = mapped[:, :, 1] / denominator
            base_x = 2.0 * (mapped_x + 0.5) / float(width * self.patch_size) - 1.0
            base_y = 2.0 * (mapped_y + 0.5) / float(height * self.patch_size) - 1.0

            radius = self.search_window // 2
            offsets = torch.tensor(
                [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)],
                device=device,
                dtype=torch.float32,
            )
            search_grids = []
            search_valid = []
            for dx, dy in offsets:
                candidate_x = base_x + 2.0 * dx / float(width)
                candidate_y = base_y + 2.0 * dy / float(height)
                search_grids.append(torch.stack((candidate_x, candidate_y), dim=-1))
                search_valid.append(
                    (candidate_x >= -1.0)
                    & (candidate_x <= 1.0)
                    & (candidate_y >= -1.0)
                    & (candidate_y <= 1.0)
                )
            sample_grid = torch.cat(search_grids, dim=3).reshape(
                batch * lag_count, height, width * len(offsets), 2
            )
            sampled = F.grid_sample(
                history.reshape(batch * lag_count, channels, height, width),
                sample_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.reshape(
                batch, lag_count, channels, height, len(offsets), width
            ).permute(0, 1, 4, 2, 3, 5)
            valid = torch.stack(search_valid, dim=2)
            valid = valid & (available[:, :, None, None, None] > 0.5)
            similarity = (
                sampled * current[:, None, None]
            ).sum(dim=3).clamp(-1.0, 1.0)
            real_logits = (similarity - self.null_similarity) / self.match_temperature
            real_logits = real_logits.masked_fill(~valid, -1e4)
            null_logits = torch.zeros(
                batch, lag_count, 1, height, width, device=device
            )
            probability = torch.softmax(
                torch.cat((null_logits, real_logits), dim=2), dim=2
            )
            weights = probability[:, :, 1:]
            confidence = weights.sum(dim=2)
            confidence_safe = confidence.clamp_min(1e-6)
            matched = (weights[:, :, :, None] * sampled).sum(dim=2)
            matched = matched / confidence_safe[:, :, None]
            displacement = torch.einsum("blkhw,kd->bldhw", weights, offsets)
            displacement = displacement / confidence_safe[:, :, None]
            matched_similarity = (weights * similarity).sum(dim=2) / confidence_safe
            normalized_entropy = -(
                probability.clamp_min(1e-12).log() * probability
            ).sum(dim=2) / math.log(float(len(offsets) + 1))

        diagnostics = {
            "match_confidence": confidence,
            "match_null_probability": probability[:, :, 0],
            "match_entropy": normalized_entropy,
            "match_displacement": displacement,
            "match_similarity": matched_similarity,
        }
        return matched.to(dtype=current_grid.dtype), diagnostics

    def forward(
        self,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_time_gap: torch.Tensor,
        history_available: torch.Tensor,
        history_crop_transform: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        transform = self._validate_inputs(
            current_grid,
            history_grids,
            history_time_gap,
            history_available,
            history_crop_transform,
        )
        batch, lag_count, _, height, width = history_grids.shape
        matched, match = self._local_matches(
            current_grid, history_grids, history_available, transform
        )
        current_normalized = F.normalize(current_grid.float(), dim=1, eps=1e-6).to(
            dtype=current_grid.dtype
        )
        current_repeated = current_normalized[:, None].expand(
            -1, lag_count, -1, -1, -1
        )
        difference = matched - current_repeated
        absolute_difference = difference.abs()
        available = history_available.to(
            device=current_grid.device, dtype=current_grid.dtype
        ).clamp(0.0, 1.0)
        confidence = match["match_confidence"].to(dtype=current_grid.dtype)
        displacement = match["match_displacement"].to(dtype=current_grid.dtype)
        similarity = match["match_similarity"].to(dtype=current_grid.dtype)
        maximum_gap = max(self.history_lags) / self.nominal_fps
        time_gap = history_time_gap.to(
            device=current_grid.device, dtype=current_grid.dtype
        ).clamp_min(0.0)
        time_encoding = torch.log1p(time_gap * self.nominal_fps) / math.log1p(
            max(maximum_gap * self.nominal_fps, 1.0)
        )
        scalar_shape = (batch, lag_count, 1, height, width)
        time_map = time_encoding[:, :, None, None, None].expand(scalar_shape)
        available_map = available[:, :, None, None, None].expand(scalar_shape)
        confidence_map = confidence[:, :, None]
        similarity_map = similarity[:, :, None]
        real_pair = torch.cat(
            (
                current_repeated,
                difference,
                absolute_difference,
                displacement,
                confidence_map,
                similarity_map,
                time_map,
                available_map,
            ),
            dim=2,
        )
        reference_pair = torch.cat(
            (
                current_repeated,
                torch.zeros_like(difference),
                torch.zeros_like(absolute_difference),
                torch.zeros_like(displacement),
                confidence_map,
                similarity_map,
                time_map,
                available_map,
            ),
            dim=2,
        )
        pair_shape = (batch * lag_count, real_pair.shape[2], height, width)
        encoded = F.gelu(self.pair_mixer(real_pair.reshape(pair_shape)))
        reference = F.gelu(self.pair_mixer(reference_pair.reshape(pair_shape)))
        encoded = (encoded - reference).reshape(
            batch, lag_count, self.hidden_channels, height, width
        )
        match_weight = confidence[:, :, None] * available_map
        # Null matching must suppress the write, not merely change relative lag
        # weights. Dividing by confidence here would cancel that safety path.
        denominator = available_map.sum(dim=1).clamp_min(1.0)
        aggregated = (encoded * match_weight).sum(dim=1) / denominator
        aggregated = F.gelu(self.spatial_mixer(aggregated))
        raw_delta = self.output_projection(aggregated)

        dimensions = tuple(range(1, current_grid.ndim))
        base_energy = current_grid.float().pow(2).mean(dim=dimensions, keepdim=True)
        raw_energy = raw_delta.float().pow(2).mean(dim=dimensions, keepdim=True)
        base_rms = torch.sqrt(base_energy + 1e-12)
        raw_rms_for_scale = torch.sqrt(raw_energy + 1e-12)
        maximum_rms = self.feature_rms_budget * base_rms
        scale = torch.clamp(maximum_rms / raw_rms_for_scale, max=1.0).detach()
        bounded_delta = raw_delta * scale.to(dtype=raw_delta.dtype)
        fused_grid = current_grid + bounded_delta

        availability_pixels = available.sum(dim=1).clamp_min(1.0) * height * width
        confidence_sum = (confidence * available[:, :, None, None]).sum(
            dim=(1, 2, 3)
        )
        confidence_denominator = confidence_sum.clamp_min(1e-6)
        motion_rms = difference.detach().float().pow(2).mean(dim=(2, 3, 4)).sqrt()
        motion_rms = (motion_rms * available.float()).sum(dim=1) / available.float().sum(
            dim=1
        ).clamp_min(1.0)
        displacement_norm = match["match_displacement"].float().pow(2).sum(dim=2).sqrt()
        diagnostics = {
            "feature_delta_rms_raw": raw_energy.detach().sqrt().flatten(1).mean(dim=1),
            "feature_delta_rms": bounded_delta.detach().float().pow(2).mean(
                dim=dimensions
            ).sqrt(),
            "feature_base_rms": base_rms.detach().flatten(1).mean(dim=1),
            "feature_clamp_scale": scale.flatten(1).mean(dim=1),
            "history_available_fraction": available.detach().mean(dim=1),
            "history_motion_rms": motion_rms,
            "match_confidence": confidence_sum / availability_pixels,
            "match_null_fraction": 1.0 - confidence_sum / availability_pixels,
            "match_entropy": (
                match["match_entropy"].float()
                * available[:, :, None, None].float()
            ).sum(dim=(1, 2, 3)) / availability_pixels,
            "match_displacement": (
                displacement_norm * confidence.float()
            ).sum(dim=(1, 2, 3)) / confidence_denominator,
            "match_similarity": (
                match["match_similarity"].float() * confidence.float()
            ).sum(dim=(1, 2, 3)) / confidence_denominator,
        }
        return fused_grid, diagnostics


class HierarchicalTemporalMemoryFusionV3(nn.Module):
    """Independent short- and medium-horizon local writers.

    L1/L2 always use the fast writer. Optional L4/L8/L16 evidence is handled by
    a separately initialized writer, so adding long history cannot dilute or
    renormalize the short-history representation. Both writers are exact
    identities at construction and the combined write has one final RMS cap.
    """

    def __init__(
        self,
        *,
        grid_channels: int = 256,
        hidden_channels: int = 64,
        history_lags: Sequence[int] = (1, 2),
        nominal_fps: float = 30.0,
        patch_size: int = 16,
        search_window: int = 5,
        match_temperature: float = 0.07,
        null_similarity: float = 0.40,
        medium_null_similarity: float = 0.50,
        feature_rms_budget: float = 0.05,
        medium_feature_rms_budget: float = 0.025,
    ):
        super().__init__()
        self.history_lags = _validate_lags(history_lags)
        if self.history_lags[:2] != (1, 2):
            raise ValueError("hierarchical_memory_v3 requires leading lags 1,2")
        self.fast_lags = (1, 2)
        self.medium_lags = self.history_lags[2:]
        if self.medium_lags and self.medium_lags != (4, 8, 16):
            raise ValueError(
                "hierarchical_memory_v3 supports either 1,2 or 1,2,4,8,16"
            )
        self.grid_channels = int(grid_channels)
        self.hidden_channels = int(hidden_channels)
        self.nominal_fps = float(nominal_fps)
        self.patch_size = int(patch_size)
        self.search_window = int(search_window)
        self.match_temperature = float(match_temperature)
        self.null_similarity = float(null_similarity)
        self.medium_null_similarity = float(medium_null_similarity)
        self.feature_rms_budget = float(feature_rms_budget)
        self.medium_feature_rms_budget = float(medium_feature_rms_budget)
        if not 0.0 < self.medium_feature_rms_budget <= self.feature_rms_budget:
            raise ValueError(
                "medium_feature_rms_budget must lie in (0, feature_rms_budget]"
            )

        # Construction order is part of the experiment contract: both variants
        # build the same fast writer first; only the long model then appends the
        # medium writer.
        self.fast_writer = TemporalLocalMemoryFusionV2(
            grid_channels=self.grid_channels,
            hidden_channels=self.hidden_channels,
            history_lags=self.fast_lags,
            nominal_fps=self.nominal_fps,
            patch_size=self.patch_size,
            search_window=self.search_window,
            match_temperature=self.match_temperature,
            null_similarity=self.null_similarity,
            feature_rms_budget=self.feature_rms_budget,
        )
        self.medium_writer = None
        if self.medium_lags:
            self.medium_writer = TemporalLocalMemoryFusionV2(
                grid_channels=self.grid_channels,
                hidden_channels=self.hidden_channels,
                history_lags=self.medium_lags,
                nominal_fps=self.nominal_fps,
                patch_size=self.patch_size,
                search_window=self.search_window,
                match_temperature=self.match_temperature,
                null_similarity=self.medium_null_similarity,
                feature_rms_budget=self.medium_feature_rms_budget,
            )

    def config(self) -> dict[str, Any]:
        return {
            "grid_channels": self.grid_channels,
            "hidden_channels": self.hidden_channels,
            "history_lags": list(self.history_lags),
            "nominal_fps": self.nominal_fps,
            "patch_size": self.patch_size,
            "search_window": self.search_window,
            "match_temperature": self.match_temperature,
            "null_similarity": self.null_similarity,
            "medium_null_similarity": self.medium_null_similarity,
            "feature_rms_budget": self.feature_rms_budget,
            "medium_feature_rms_budget": self.medium_feature_rms_budget,
        }

    @staticmethod
    def _select_lags(value: torch.Tensor | None, start: int, stop: int):
        return None if value is None else value[:, start:stop]

    def forward(
        self,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_time_gap: torch.Tensor,
        history_available: torch.Tensor,
        history_crop_transform: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected_lags = len(self.history_lags)
        if history_grids.ndim != 5 or history_grids.shape[1] != expected_lags:
            raise ValueError(
                f"Expected {expected_lags} history grids, got {tuple(history_grids.shape)}"
            )
        fast_fused, fast = self.fast_writer(
            current_grid,
            history_grids[:, :2],
            history_time_gap[:, :2],
            history_available[:, :2],
            self._select_lags(history_crop_transform, 0, 2),
        )
        fast_delta = fast_fused - current_grid
        medium_delta = torch.zeros_like(fast_delta)
        medium: dict[str, torch.Tensor] = {}
        if self.medium_writer is not None:
            medium_fused, medium = self.medium_writer(
                current_grid,
                history_grids[:, 2:],
                history_time_gap[:, 2:],
                history_available[:, 2:],
                self._select_lags(history_crop_transform, 2, expected_lags),
            )
            medium_delta = medium_fused - current_grid

        raw_delta = fast_delta + medium_delta
        dimensions = tuple(range(1, current_grid.ndim))
        base_rms = torch.sqrt(
            current_grid.float().pow(2).mean(dim=dimensions, keepdim=True) + 1e-12
        )
        raw_rms = torch.sqrt(
            raw_delta.float().pow(2).mean(dim=dimensions, keepdim=True) + 1e-12
        )
        scale = torch.clamp(
            self.feature_rms_budget * base_rms / raw_rms, max=1.0
        ).detach()
        bounded_delta = raw_delta * scale.to(dtype=raw_delta.dtype)
        diagnostics = dict(fast)
        diagnostics.update(
            {
                "feature_delta_rms_raw": raw_rms.detach().flatten(1).mean(dim=1),
                "feature_delta_rms": bounded_delta.detach().float().pow(2).mean(
                    dim=dimensions
                ).sqrt(),
                "feature_base_rms": base_rms.detach().flatten(1).mean(dim=1),
                "feature_clamp_scale": scale.detach().flatten(1).mean(dim=1),
                "history_available_fraction": history_available.detach().float().mean(
                    dim=1
                ),
                "fast_feature_delta_rms": fast_delta.detach().float().pow(2).mean(
                    dim=dimensions
                ).sqrt(),
                "medium_feature_delta_rms": medium_delta.detach().float().pow(2).mean(
                    dim=dimensions
                ).sqrt(),
            }
        )
        for name, value in medium.items():
            diagnostics[f"medium_{name}"] = value
        return current_grid + bounded_delta, diagnostics


class TemporalMainTrunkV3(nn.Module):
    """Fresh FullGrid/ReZero trunk with frozen DINO and temporal grid memory."""

    def __init__(
        self,
        base_model: DinoTactileModel,
        palm_vertex_indices: Sequence[int],
        fusion: HierarchicalTemporalMemoryFusionV3,
        *,
        online_encoder_chunk_size: int = 128,
    ):
        super().__init__()
        if not isinstance(base_model, DinoTactileModel):
            raise TypeError("TemporalMainTrunkV3 requires a DinoTactileModel")
        if str(base_model.pool_layout) != "fullgrid32":
            raise ValueError("TemporalMainTrunkV3 requires FullGrid32")
        self.base_model = base_model
        self.fusion = fusion
        self.online_encoder_chunk_size = int(online_encoder_chunk_size)
        if self.online_encoder_chunk_size <= 0:
            raise ValueError("online_encoder_chunk_size must be positive")
        self.base_model.backbone.requires_grad_(False).eval()
        self.register_buffer(
            "palm_vertex_indices",
            torch.as_tensor(tuple(int(v) for v in palm_vertex_indices), dtype=torch.long),
            persistent=True,
        )

    @property
    def adapter(self):
        return self.fusion

    @property
    def decoder(self):
        return self.base_model.tactile_head.decoder

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.backbone.eval()
        self.base_model.tactile_head.train(mode)
        self.fusion.train(mode)
        return self

    def trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def trainable_named_parameters(self):
        return (
            (name, parameter)
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )

    def compact_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value
            for name, value in self.state_dict().items()
            if name.startswith("base_model.tactile_head.")
            or name.startswith("fusion.")
            or name == "palm_vertex_indices"
        }

    def load_compact_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_trainable = [
            name
            for name in incompatible.missing_keys
            if name.startswith("base_model.tactile_head.")
            or name.startswith("fusion.")
        ]
        if unexpected or missing_trainable:
            raise RuntimeError(
                "Temporal trunk state mismatch: "
                f"missing_trainable={missing_trainable}, unexpected={unexpected}"
            )

    def _encode_grid_group(self, images: torch.Tensor) -> torch.Tensor:
        grids = []
        for start in range(0, len(images), self.online_encoder_chunk_size):
            image_chunk = images[start : start + self.online_encoder_chunk_size]
            with torch.no_grad():
                levels = self.base_model._extract_tactile_features(image_chunk)
            grids.append(self.base_model.tactile_head._fuse(levels))
        return torch.cat(grids, dim=0)

    def materialize_online_features(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "current_grid" in batch:
            return dict(batch)
        result = dict(batch)
        current_image = batch["current_image"]
        history_images = batch["history_images"]
        batch_size, lag_count = history_images.shape[:2]
        current_grid = self._encode_grid_group(current_image)
        result["current_grid"] = current_grid

        for image_key, available_key, grid_key in (
            ("history_images", "history_available", "history_grids"),
            (
                "control_history_images",
                "control_history_available",
                "control_history_grids",
            ),
            (
                "contralateral_history_images",
                "contralateral_history_available",
                "contralateral_history_grids",
            ),
        ):
            images = batch.get(image_key)
            if images is None:
                continue
            available = batch[available_key].reshape(-1) > 0.5
            flat_images = images.reshape(-1, *images.shape[2:])
            flat_grids = current_grid[:, None].expand(
                -1, lag_count, -1, -1, -1
            ).reshape(-1, *current_grid.shape[1:]).clone()
            valid_indices = torch.nonzero(available, as_tuple=False).flatten()
            if valid_indices.numel() > 0:
                encoded = self._encode_grid_group(
                    flat_images.index_select(0, valid_indices)
                )
                flat_grids = flat_grids.index_copy(0, valid_indices, encoded)
            result[grid_key] = flat_grids.reshape(
                batch_size, lag_count, *current_grid.shape[1:]
            )
        return result

    def _decode_paired_grids(
        self,
        current_grid: torch.Tensor,
        fused_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a feature pair with identical training-time dropout masks.

        The temporal writers are zero initialized, so their first forward must
        be exactly equal to the RGB-only path while still retaining a gradient
        through the fused decode.  Replaying the decoder RNG state gives both
        paths the same dropout realization and advances the process RNG only
        once, just like a single baseline decoder call.
        """

        if not self.training:
            return self.decoder(current_grid), self.decoder(fused_grid)

        cpu_state_before = torch.random.get_rng_state()
        cuda_state_before = None
        if current_grid.is_cuda:
            cuda_state_before = torch.cuda.get_rng_state(current_grid.device)

        current_logits = self.decoder(current_grid)
        cpu_state_after = torch.random.get_rng_state()
        cuda_state_after = None
        if current_grid.is_cuda:
            cuda_state_after = torch.cuda.get_rng_state(current_grid.device)

        torch.random.set_rng_state(cpu_state_before)
        if cuda_state_before is not None:
            torch.cuda.set_rng_state(cuda_state_before, current_grid.device)
        try:
            fused_logits = self.decoder(fused_grid)
        finally:
            torch.random.set_rng_state(cpu_state_after)
            if cuda_state_after is not None:
                torch.cuda.set_rng_state(cuda_state_after, current_grid.device)
        return current_logits, fused_logits

    def forward(
        self,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_time_gap: torch.Tensor,
        history_available: torch.Tensor,
        *,
        history_crop_transform: torch.Tensor | None = None,
        cached_base_logits: torch.Tensor | None = None,
        decoded_current_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del cached_base_logits
        fused_grid, diagnostics = self.fusion(
            current_grid,
            history_grids,
            history_time_gap,
            history_available,
            history_crop_transform,
        )
        if decoded_current_logits is None:
            decoded_current_full, fused_full = self._decode_paired_grids(
                current_grid, fused_grid
            )
            decoded_current_logits = decoded_current_full.index_select(
                1, self.palm_vertex_indices
            )
            fused_logits = fused_full.index_select(1, self.palm_vertex_indices)
        else:
            fused_logits = self.decoder(fused_grid).index_select(
                1, self.palm_vertex_indices
            )
        output = {
            "pred_logits": fused_logits,
            "pred_tactile": torch.sigmoid(fused_logits),
            "base_pred_logits": decoded_current_logits,
            "base_pred_tactile": torch.sigmoid(decoded_current_logits),
            "decoded_current_logits": decoded_current_logits,
            "decoder_logit_delta": fused_logits.float()
            - decoded_current_logits.float(),
            "feature_delta": fused_grid - current_grid,
        }
        output.update(diagnostics)
        return output


class CausalClipTransformerBlock(nn.Module):
    """Factorized local-spatial and causal-temporal feature block."""

    def __init__(self, hidden_channels: int, heads: int, ffn_ratio: int = 2):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.heads = int(heads)
        if self.hidden_channels <= 0 or self.hidden_channels % self.heads:
            raise ValueError("hidden_channels must be positive and divisible by heads")
        self.spatial_norm = nn.LayerNorm(self.hidden_channels)
        self.spatial_depthwise = nn.Conv2d(
            self.hidden_channels,
            self.hidden_channels,
            kernel_size=3,
            padding=1,
            groups=self.hidden_channels,
            bias=False,
        )
        self.spatial_pointwise = nn.Conv2d(
            self.hidden_channels, self.hidden_channels, kernel_size=1, bias=True
        )
        self.temporal_norm = nn.LayerNorm(self.hidden_channels)
        self.qkv = nn.Linear(self.hidden_channels, self.hidden_channels * 3, bias=True)
        self.attention_output = nn.Linear(
            self.hidden_channels, self.hidden_channels, bias=True
        )
        self.ffn_norm = nn.LayerNorm(self.hidden_channels)
        ffn_channels = self.hidden_channels * int(ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_channels, ffn_channels),
            nn.GELU(),
            nn.Linear(ffn_channels, self.hidden_channels),
        )

    def _temporal_attention(
        self, tokens: torch.Tensor, valid: torch.Tensor, *, causal: bool
    ) -> torch.Tensor:
        batch_tokens, steps, channels = tokens.shape
        head_dim = channels // self.heads
        qkv = self.qkv(self.temporal_norm(tokens)).reshape(
            batch_tokens, steps, 3, self.heads, head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if causal:
            temporal = torch.ones(
                steps, steps, device=tokens.device, dtype=torch.bool
            ).tril()
        else:
            temporal = torch.eye(
                steps, device=tokens.device, dtype=torch.bool
            )
        allowed = temporal[None, None] & valid[:, None, None, :]
        invalid_query = ~valid[:, None, :, None]
        diagonal = torch.eye(
            steps, device=tokens.device, dtype=torch.bool
        )[None, None]
        allowed = allowed | (invalid_query & diagonal)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_tokens, steps, channels)
        attended = self.attention_output(attended)
        return attended * valid[:, :, None].to(dtype=attended.dtype)

    def forward(
        self,
        value: torch.Tensor,
        valid: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        batch, steps, channels, height, width = value.shape
        mask = valid[:, :, None, None, None].to(dtype=value.dtype)
        spatial = value.permute(0, 1, 3, 4, 2)
        spatial = self.spatial_norm(spatial).permute(0, 1, 4, 2, 3)
        spatial = spatial.reshape(batch * steps, channels, height, width)
        spatial = self.spatial_pointwise(F.gelu(self.spatial_depthwise(spatial)))
        value = value + spatial.reshape(batch, steps, channels, height, width) * mask

        tokens = value.permute(0, 3, 4, 1, 2).reshape(
            batch * height * width, steps, channels
        )
        token_valid = valid[:, None, None].expand(-1, height, width, -1).reshape(
            batch * height * width, steps
        )
        tokens = tokens + self._temporal_attention(
            tokens, token_valid, causal=causal
        )
        tokens = tokens + self.ffn(self.ffn_norm(tokens)) * token_valid[
            :, :, None
        ].to(dtype=tokens.dtype)
        return tokens.reshape(batch, height, width, steps, channels).permute(
            0, 3, 4, 1, 2
        )


class CausalClipTransformerFusionV4(nn.Module):
    """Eight-frame feature-grid Transformer with an exact RGB-only fallback."""

    def __init__(
        self,
        *,
        grid_channels: int = 256,
        hidden_channels: int = 128,
        clip_length: int = 8,
        grid_height: int = 16,
        grid_width: int = 12,
        layers: int = 2,
        heads: int = 4,
        ffn_ratio: int = 2,
        nominal_fps: float = 30.0,
        feature_rms_budget: float = 0.05,
    ):
        super().__init__()
        self.grid_channels = int(grid_channels)
        self.hidden_channels = int(hidden_channels)
        self.clip_length = int(clip_length)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.layers = int(layers)
        self.heads = int(heads)
        self.ffn_ratio = int(ffn_ratio)
        self.nominal_fps = float(nominal_fps)
        self.feature_rms_budget = float(feature_rms_budget)
        if self.clip_length < 2 or self.layers < 1:
            raise ValueError("clip_length must be >=2 and layers must be positive")
        if self.feature_rms_budget <= 0.0 or self.nominal_fps <= 0.0:
            raise ValueError("feature_rms_budget and nominal_fps must be positive")
        self.input_projection = nn.Conv2d(
            self.grid_channels, self.hidden_channels, kernel_size=1, bias=True
        )
        self.time_projection = nn.Linear(9, self.hidden_channels, bias=False)
        self.affine_projection = nn.Linear(6, self.hidden_channels, bias=False)
        self.blocks = nn.ModuleList(
            CausalClipTransformerBlock(
                self.hidden_channels, self.heads, self.ffn_ratio
            )
            for _ in range(self.layers)
        )
        self.output_norm = nn.LayerNorm(self.hidden_channels)
        self.output_projection = nn.Conv2d(
            self.hidden_channels, self.grid_channels, kernel_size=1, bias=False
        )
        nn.init.zeros_(self.output_projection.weight)
        self.register_buffer(
            "spatial_position",
            self._spatial_encoding(
                self.hidden_channels, self.grid_height, self.grid_width
            ),
            persistent=True,
        )

    @staticmethod
    def _spatial_encoding(channels: int, height: int, width: int) -> torch.Tensor:
        if channels % 4:
            raise ValueError("hidden_channels must be divisible by four")
        quarter = channels // 4
        frequencies = torch.exp(
            torch.arange(quarter, dtype=torch.float32)
            * (-math.log(10000.0) / max(quarter - 1, 1))
        )
        y = torch.linspace(0.0, 1.0, height, dtype=torch.float32)
        x = torch.linspace(0.0, 1.0, width, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.cat(
            (
                torch.sin(2.0 * math.pi * yy[..., None] * frequencies),
                torch.cos(2.0 * math.pi * yy[..., None] * frequencies),
                torch.sin(2.0 * math.pi * xx[..., None] * frequencies),
                torch.cos(2.0 * math.pi * xx[..., None] * frequencies),
            ),
            dim=-1,
        ).permute(2, 0, 1)[None, None]

    def config(self) -> dict[str, Any]:
        return {
            "grid_channels": self.grid_channels,
            "hidden_channels": self.hidden_channels,
            "clip_length": self.clip_length,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "layers": self.layers,
            "heads": self.heads,
            "ffn_ratio": self.ffn_ratio,
            "nominal_fps": self.nominal_fps,
            "feature_rms_budget": self.feature_rms_budget,
        }

    def _time_encoding(self, time: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        relative = (time - time[:, :1]).clamp_min(0.0) * self.nominal_fps
        scale = relative / float(max(self.clip_length - 1, 1))
        frequencies = scale.new_tensor((1.0, 2.0, 4.0, 8.0))
        phase = 2.0 * math.pi * scale[:, :, None] * frequencies
        features = torch.cat(
            (torch.sin(phase), torch.cos(phase), valid[:, :, None].float()), dim=2
        )
        return self.time_projection(features.to(dtype=self.time_projection.weight.dtype))

    def _affine_encoding(self, affine: torch.Tensor) -> torch.Tensor:
        batch, steps = affine.shape[:2]
        relative = torch.eye(
            3, device=affine.device, dtype=torch.float32
        ).reshape(1, 1, 3, 3).repeat(batch, steps, 1, 1)
        if steps > 1:
            with torch.autocast(device_type=affine.device.type, enabled=False):
                current = affine[:, 1:].detach().float()
                previous = affine[:, :-1].detach().float()
                relative[:, 1:] = previous @ torch.linalg.inv(current)
        features = torch.stack(
            (
                relative[:, :, 0, 0] - 1.0,
                relative[:, :, 0, 1],
                relative[:, :, 0, 2] / float(self.grid_width * 16),
                relative[:, :, 1, 0],
                relative[:, :, 1, 1] - 1.0,
                relative[:, :, 1, 2] / float(self.grid_height * 16),
            ),
            dim=2,
        )
        return self.affine_projection(
            features.to(dtype=self.affine_projection.weight.dtype)
        )

    def _encode(
        self,
        grids: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        causal: bool,
    ) -> torch.Tensor:
        batch, steps, _, height, width = grids.shape
        projected = self.input_projection(
            grids.reshape(batch * steps, self.grid_channels, height, width)
        ).reshape(batch, steps, self.hidden_channels, height, width)
        condition = self._time_encoding(time, valid) + self._affine_encoding(
            crop_affine
        )
        value = projected + self.spatial_position.to(dtype=projected.dtype)
        value = value + condition[:, :, :, None, None].to(dtype=projected.dtype)
        value = value * valid[:, :, None, None, None].to(dtype=value.dtype)
        for block in self.blocks:
            value = block(value, valid > 0.5, causal=causal)
        value = self.output_norm(value.permute(0, 1, 3, 4, 2)).permute(
            0, 1, 4, 2, 3
        )
        return value

    def forward(
        self,
        grids: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        force_reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = (
            grids.shape[0],
            self.clip_length,
            self.grid_channels,
            self.grid_height,
            self.grid_width,
        )
        if tuple(grids.shape) != expected:
            raise ValueError(f"clip grids must be {expected}, got {tuple(grids.shape)}")
        if tuple(valid.shape) != expected[:2] or tuple(time.shape) != expected[:2]:
            raise ValueError("clip time/valid tensors have incompatible shapes")
        if tuple(crop_affine.shape) != (*expected[:2], 3, 3):
            raise ValueError("clip crop affine tensor has an incompatible shape")
        causal_state = self._encode(grids, time, valid, crop_affine, causal=True)
        reset_state = self._encode(grids, time, valid, crop_affine, causal=False)
        temporal_state = causal_state - reset_state
        batch, steps = grids.shape[:2]
        raw_delta = self.output_projection(
            temporal_state.reshape(
                batch * steps,
                self.hidden_channels,
                self.grid_height,
                self.grid_width,
            )
        ).reshape_as(grids)
        frame_valid = valid[:, :, None, None, None].to(dtype=raw_delta.dtype)
        raw_delta = raw_delta * frame_valid
        if force_reset is not None:
            raw_delta = raw_delta * (~force_reset.bool())[:, None, None, None, None]
        dimensions = (2, 3, 4)
        base_energy = grids.float().pow(2).mean(dim=dimensions, keepdim=True)
        delta_energy = raw_delta.float().pow(2).mean(dim=dimensions, keepdim=True)
        base_rms = torch.sqrt(base_energy + 1e-12)
        raw_rms = torch.sqrt(delta_energy + 1e-12)
        scale = torch.clamp(
            self.feature_rms_budget * base_rms / raw_rms, max=1.0
        ).detach()
        bounded_delta = raw_delta * scale.to(dtype=raw_delta.dtype)
        previous = torch.roll(grids.detach().float(), shifts=1, dims=1)
        motion = (grids.detach().float() - previous).pow(2).mean(
            dim=dimensions
        ).sqrt()
        history_count = torch.cumsum(valid.float(), dim=1) - valid.float()
        history_fraction = history_count / torch.arange(
            self.clip_length, device=valid.device, dtype=torch.float32
        ).clamp_min(1.0)[None]
        diagnostics = {
            "feature_delta_rms_raw": raw_rms.flatten(2).mean(dim=2),
            "feature_delta_rms": bounded_delta.detach().float().pow(2).mean(
                dim=dimensions
            ).sqrt(),
            "feature_base_rms": base_rms.flatten(2).mean(dim=2),
            "feature_clamp_scale": scale.flatten(2).mean(dim=2),
            "history_available_fraction": history_fraction * valid.float(),
            "history_motion_rms": motion * valid.float(),
        }
        return grids + bounded_delta, diagnostics


class TemporalClipMainTrunkV4(TemporalMainTrunkV3):
    """Fresh FullGrid trunk that decodes every valid frame in a causal clip."""

    def __init__(
        self,
        base_model: DinoTactileModel,
        palm_vertex_indices: Sequence[int],
        fusion: CausalClipTransformerFusionV4,
        *,
        online_encoder_chunk_size: int = 128,
    ):
        super().__init__(
            base_model,
            palm_vertex_indices,
            fusion,
            online_encoder_chunk_size=online_encoder_chunk_size,
        )

    def materialize_online_features(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "clip_grids" in batch:
            return dict(batch)
        result = dict(batch)
        images = batch["clip_images"]
        valid = batch["clip_valid"].reshape(-1) > 0.5
        batch_size, steps = images.shape[:2]
        flat_images = images.reshape(-1, *images.shape[2:])
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise RuntimeError("A temporal clip batch contains no valid frames")
        encoded = self._encode_grid_group(flat_images.index_select(0, valid_indices))
        flat_grids = encoded.new_zeros(
            flat_images.shape[0], *encoded.shape[1:]
        )
        flat_grids = flat_grids.index_copy(0, valid_indices, encoded)
        result["clip_grids"] = flat_grids.reshape(
            batch_size, steps, *encoded.shape[1:]
        )
        # The RGB tensor is no longer needed after frozen DINO extraction.
        # Releasing it here matters for eight-frame online batches and for
        # evaluation batches that also carry a control clip.
        result.pop("clip_images", None)
        control_images = batch.get("control_clip_images")
        if control_images is not None:
            control_valid = batch["control_clip_valid"].reshape(-1) > 0.5
            control_flat = control_images.reshape(-1, *control_images.shape[2:])
            control_indices = torch.nonzero(control_valid, as_tuple=False).flatten()
            control_encoded = self._encode_grid_group(
                control_flat.index_select(0, control_indices)
            )
            control_grids = control_encoded.new_zeros(
                control_flat.shape[0], *control_encoded.shape[1:]
            ).index_copy(0, control_indices, control_encoded)
            result["control_clip_grids"] = control_grids.reshape(
                batch_size, steps, *control_encoded.shape[1:]
            )
            result.pop("control_clip_images", None)
        return result

    def forward_clip(
        self,
        grids: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        force_reset: torch.Tensor | None = None,
        decode_base: bool = True,
    ) -> dict[str, torch.Tensor]:
        fused, diagnostics = self.fusion(
            grids,
            time,
            valid,
            crop_affine,
            force_reset=force_reset,
        )
        batch, steps = grids.shape[:2]
        current_flat = grids.reshape(batch * steps, *grids.shape[2:])
        fused_flat = fused.reshape(batch * steps, *fused.shape[2:])
        if decode_base:
            base_full, fused_full = self._decode_paired_grids(
                current_flat, fused_flat
            )
            base_logits = base_full.index_select(
                1, self.palm_vertex_indices
            ).reshape(batch, steps, -1)
        else:
            # The first frame and randomly reset clips already traverse the
            # exact RGB-only path, so training needs only one decoder call.
            fused_full = self.decoder(fused_flat)
        fused_logits = fused_full.index_select(1, self.palm_vertex_indices).reshape(
            batch, steps, -1
        )
        if not decode_base:
            base_logits = fused_logits.detach()
        decoder_delta = (
            fused_logits.float() - base_logits.float()
            if decode_base
            else torch.zeros_like(fused_logits, dtype=torch.float32)
        )
        output = {
            "pred_logits": fused_logits,
            "pred_tactile": torch.sigmoid(fused_logits),
            "base_pred_logits": base_logits,
            "base_pred_tactile": torch.sigmoid(base_logits),
            "decoder_logit_delta": decoder_delta,
            "feature_delta": fused - grids,
            "base_decode_performed": decode_base,
        }
        output.update(diagnostics)
        return output

    def forward(self, batch: Mapping[str, torch.Tensor], **kwargs):
        materialized = self.materialize_online_features(batch)
        return self.forward_clip(
            materialized["clip_grids"],
            materialized["clip_time"],
            materialized["clip_valid"],
            materialized["clip_crop_affine"],
            **kwargs,
        )


def _rotary_frequencies(width: int, device: torch.device) -> torch.Tensor:
    if width <= 0 or width % 2:
        raise ValueError("RoPE width must be a positive even integer")
    return torch.exp(
        -math.log(10000.0)
        * torch.arange(0, width, 2, device=device, dtype=torch.float32)
        / float(width)
    )


def _apply_1d_rope(
    value: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Apply continuous-position RoPE to `[B,H,N,D]` query or key tensors."""

    width = value.shape[-1]
    frequencies = _rotary_frequencies(width, value.device)
    angles = positions.float()[..., None] * frequencies
    cosine = torch.cos(angles).repeat_interleave(2, dim=-1)[:, None]
    sine = torch.sin(angles).repeat_interleave(2, dim=-1)[:, None]
    paired = value.float().reshape(*value.shape[:-1], width // 2, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)
    return (value.float() * cosine + rotated * sine).to(dtype=value.dtype)


def _apply_2d_rope(
    value: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Apply axial 2D RoPE while preserving the full per-head width."""

    head_width = value.shape[-1]
    if head_width % 4:
        raise ValueError("2D RoPE requires head_dim divisible by four")
    if value.shape[-2] != int(height) * int(width):
        raise ValueError("2D RoPE token count does not match the grid")
    half = head_width // 2
    y = torch.arange(height, device=value.device, dtype=torch.float32)
    x = torch.arange(width, device=value.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    y_positions = yy.reshape(1, -1).expand(value.shape[0], -1)
    x_positions = xx.reshape(1, -1).expand(value.shape[0], -1)
    return torch.cat(
        (
            _apply_1d_rope(value[..., :half], y_positions),
            _apply_1d_rope(value[..., half:], x_positions),
        ),
        dim=-1,
    )


def _flash_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
) -> torch.Tensor:
    """Use Flash-SDPA on CUDA and fail instead of silently using math SDPA."""

    if query.is_cuda:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        ):
            return F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=causal,
            )
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=causal,
    )


class FullGridSpatialAttentionBlockV5(nn.Module):
    """Full 192-token spatial attention at the audited 32-channel width."""

    def __init__(self, channels: int, heads: int, ffn_ratio: int, dropout: float):
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        self.ffn_ratio = int(ffn_ratio)
        if self.channels <= 0 or self.channels % self.heads:
            raise ValueError("Spatial channels must be divisible by heads")
        if (self.channels // self.heads) % 4:
            raise ValueError("Spatial head_dim must be divisible by four for 2D RoPE")
        hidden = self.channels * self.ffn_ratio
        self.attention_norm = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, 3 * self.channels)
        self.output = nn.Linear(self.channels, self.channels)
        self.ffn_norm = nn.LayerNorm(self.channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.channels),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, count, channels = tokens.shape
        head_width = channels // self.heads
        qkv = self.qkv(self.attention_norm(tokens)).reshape(
            batch, count, 3, self.heads, head_width
        )
        query, key, value = (
            item.permute(0, 2, 1, 3) for item in qkv.unbind(dim=2)
        )
        query = _apply_2d_rope(query, height, width)
        key = _apply_2d_rope(key, height, width)
        attended = _flash_sdpa(query, key, value)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, count, channels)
        tokens = tokens + self.dropout(self.output(attended))
        return tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))


class FullWidthTemporalAttentionBlockV5(nn.Module):
    """Full-width temporal attention over complete 6144-D frame tokens."""

    MODES = ("bidirectional", "self_only", "past_only", "future_only")

    def __init__(
        self,
        channels: int,
        heads: int,
        ffn_ratio: int,
        dropout: float,
        layer_scale_init: float,
    ):
        super().__init__()
        self.channels = int(channels)
        self.heads = int(heads)
        self.ffn_ratio = int(ffn_ratio)
        if self.channels <= 0 or self.channels % self.heads:
            raise ValueError("Temporal channels must be divisible by heads")
        if (self.channels // self.heads) % 2:
            raise ValueError("Temporal head_dim must be even for RoPE")
        hidden = self.channels * self.ffn_ratio
        self.attention_norm = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, 3 * self.channels)
        self.output = nn.Linear(self.channels, self.channels)
        self.ffn_norm = nn.LayerNorm(self.channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.channels),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.attention_scale = nn.Parameter(
            torch.full((self.channels,), float(layer_scale_init))
        )
        self.ffn_scale = nn.Parameter(
            torch.full((self.channels,), float(layer_scale_init))
        )

    def _attend_rows(
        self,
        normalized: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        mode: str,
        force_reset: torch.Tensor,
    ) -> torch.Tensor:
        batch, steps, channels = normalized.shape
        head_width = channels // self.heads
        qkv = self.qkv(normalized).reshape(
            batch, steps, 3, self.heads, head_width
        )
        query, key, value = (
            item.permute(0, 2, 1, 3) for item in qkv.unbind(dim=2)
        )
        result = torch.zeros_like(value)
        lengths = valid.long().sum(dim=1)
        for length in torch.unique(lengths).tolist():
            length = int(length)
            if length <= 0:
                continue
            length_rows = lengths == length
            for reset_value in (False, True):
                rows = torch.nonzero(
                    length_rows & (force_reset == reset_value), as_tuple=False
                ).flatten()
                if rows.numel() == 0:
                    continue
                row_query = query.index_select(0, rows)[:, :, :length]
                row_key = key.index_select(0, rows)[:, :, :length]
                row_value = value.index_select(0, rows)[:, :, :length]
                row_positions = positions.index_select(0, rows)[:, :length]
                effective_mode = "self_only" if reset_value else mode
                if effective_mode == "self_only":
                    attended = row_value
                else:
                    row_query = _apply_1d_rope(row_query, row_positions)
                    row_key = _apply_1d_rope(row_key, row_positions)
                    if effective_mode == "future_only":
                        attended = _flash_sdpa(
                            row_query.flip(2),
                            row_key.flip(2),
                            row_value.flip(2),
                            causal=True,
                        ).flip(2)
                    else:
                        attended = _flash_sdpa(
                            row_query,
                            row_key,
                            row_value,
                            causal=effective_mode == "past_only",
                        )
                result[rows, :, :length] = attended
        return result.permute(0, 2, 1, 3).reshape(batch, steps, channels)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        *,
        mode: str,
        force_reset: torch.Tensor,
    ) -> torch.Tensor:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported temporal attention mode={mode!r}")
        attended = self._attend_rows(
            self.attention_norm(tokens), positions, valid, mode, force_reset
        )
        valid_float = valid[:, :, None].to(dtype=tokens.dtype)
        tokens = tokens + self.dropout(self.output(attended)) * self.attention_scale
        tokens = tokens + (
            self.dropout(self.ffn(self.ffn_norm(tokens)))
            * self.ffn_scale
            * valid_float
        )
        return tokens


class FullGrid6144SpatiotemporalFusionV5(nn.Module):
    """Spatial full attention followed by full-width bidirectional time attention."""

    def __init__(
        self,
        *,
        patch_channels: int = 32,
        clip_length: int = 8,
        grid_height: int = 16,
        grid_width: int = 12,
        spatial_layers: int = 1,
        spatial_heads: int = 4,
        spatial_ffn_ratio: int = 2,
        temporal_layers: int = 1,
        temporal_heads: int = 48,
        temporal_ffn_ratio: int = 2,
        nominal_fps: float = 30.0,
        residual_dropout: float = 0.10,
        layer_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.patch_channels = int(patch_channels)
        self.clip_length = int(clip_length)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.spatial_layers = int(spatial_layers)
        self.spatial_heads = int(spatial_heads)
        self.spatial_ffn_ratio = int(spatial_ffn_ratio)
        self.temporal_layers = int(temporal_layers)
        self.temporal_heads = int(temporal_heads)
        self.temporal_ffn_ratio = int(temporal_ffn_ratio)
        self.nominal_fps = float(nominal_fps)
        self.residual_dropout = float(residual_dropout)
        self.layer_scale_init = float(layer_scale_init)
        self.frame_channels = (
            self.patch_channels * self.grid_height * self.grid_width
        )
        if self.clip_length < 2:
            raise ValueError("FullGrid6144 clips require at least two frames")
        if self.spatial_layers < 1 or self.temporal_layers < 1:
            raise ValueError("Spatial and temporal layer counts must be positive")
        if self.nominal_fps <= 0.0:
            raise ValueError("nominal_fps must be positive")
        if not 0.0 <= self.residual_dropout < 1.0:
            raise ValueError("residual_dropout must lie in [0, 1)")
        if self.layer_scale_init <= 0.0:
            raise ValueError("layer_scale_init must be positive")
        self.spatial_blocks = nn.ModuleList(
            FullGridSpatialAttentionBlockV5(
                self.patch_channels,
                self.spatial_heads,
                self.spatial_ffn_ratio,
                self.residual_dropout,
            )
            for _ in range(self.spatial_layers)
        )
        self.affine_projection = nn.Linear(6, self.frame_channels, bias=False)
        nn.init.zeros_(self.affine_projection.weight)
        self.temporal_blocks = nn.ModuleList(
            FullWidthTemporalAttentionBlockV5(
                self.frame_channels,
                self.temporal_heads,
                self.temporal_ffn_ratio,
                self.residual_dropout,
                self.layer_scale_init,
            )
            for _ in range(self.temporal_layers)
        )
        self.output_norm = nn.LayerNorm(self.frame_channels)

    def config(self) -> dict[str, Any]:
        return {
            "patch_channels": self.patch_channels,
            "clip_length": self.clip_length,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "spatial_layers": self.spatial_layers,
            "spatial_heads": self.spatial_heads,
            "spatial_ffn_ratio": self.spatial_ffn_ratio,
            "temporal_layers": self.temporal_layers,
            "temporal_heads": self.temporal_heads,
            "temporal_ffn_ratio": self.temporal_ffn_ratio,
            "nominal_fps": self.nominal_fps,
            "residual_dropout": self.residual_dropout,
            "layer_scale_init": self.layer_scale_init,
        }

    def _affine_features(self, affine: torch.Tensor) -> torch.Tensor:
        batch, steps = affine.shape[:2]
        relative = torch.eye(
            3, device=affine.device, dtype=torch.float32
        ).reshape(1, 1, 3, 3).repeat(batch, steps, 1, 1)
        if steps > 1:
            with torch.autocast(device_type=affine.device.type, enabled=False):
                current = affine[:, 1:].detach().float()
                previous = affine[:, :-1].detach().float()
                relative[:, 1:] = previous @ torch.linalg.inv(current)
        return torch.stack(
            (
                relative[:, :, 0, 0] - 1.0,
                relative[:, :, 0, 1],
                relative[:, :, 0, 2] / float(self.grid_width * 16),
                relative[:, :, 1, 0],
                relative[:, :, 1, 1] - 1.0,
                relative[:, :, 1, 2] / float(self.grid_height * 16),
            ),
            dim=2,
        )

    def flatten_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Restore FullGrid's channel-major `[C,H,W]` flattening contract."""

        batch, steps, count, channels = tokens.shape
        if count != self.grid_height * self.grid_width or channels != self.patch_channels:
            raise ValueError("Patch-token shape does not match the FullGrid contract")
        return tokens.reshape(
            batch, steps, self.grid_height, self.grid_width, channels
        ).permute(0, 1, 4, 2, 3).contiguous().flatten(2)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        attention_mode: str = "bidirectional",
        force_reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, steps, count, channels = patch_tokens.shape
        expected = (
            batch,
            self.clip_length,
            self.grid_height * self.grid_width,
            self.patch_channels,
        )
        if tuple(patch_tokens.shape) != expected:
            raise ValueError(
                f"FullGrid6144 tokens must be {expected}, got {tuple(patch_tokens.shape)}"
            )
        if tuple(time.shape) != (batch, steps) or tuple(valid.shape) != (batch, steps):
            raise ValueError("FullGrid6144 time/valid shapes are incompatible")
        if tuple(crop_affine.shape) != (batch, steps, 3, 3):
            raise ValueError("FullGrid6144 crop affine shape is incompatible")
        reset = (
            torch.zeros(batch, device=patch_tokens.device, dtype=torch.bool)
            if force_reset is None
            else force_reset.bool()
        )
        if tuple(reset.shape) != (batch,):
            raise ValueError("force_reset must contain one flag per clip")
        spatial = patch_tokens.reshape(batch * steps, count, channels)
        for block in self.spatial_blocks:
            spatial = block(spatial, self.grid_height, self.grid_width)
        spatial = spatial.reshape(batch, steps, count, channels)
        base_frames = self.flatten_patch_tokens(patch_tokens)
        frames = self.flatten_patch_tokens(spatial)
        if attention_mode != "self_only":
            affine_features = self._affine_features(crop_affine)
            frames = frames + self.affine_projection(
                affine_features.to(dtype=self.affine_projection.weight.dtype)
            ).to(dtype=frames.dtype)
        positions = (time.float() - time[:, :1].float()) * self.nominal_fps
        temporal_input = frames
        for block in self.temporal_blocks:
            frames = block(
                frames,
                positions,
                valid > 0.5,
                mode=attention_mode,
                force_reset=reset,
            )
        frames = self.output_norm(frames)
        valid_float = valid[:, :, None].to(dtype=frames.dtype)
        spatial_delta = self.flatten_patch_tokens(spatial) - base_frames
        temporal_delta = frames - temporal_input
        full_delta = frames - base_frames
        previous = torch.roll(base_frames.detach().float(), shifts=1, dims=1)
        motion = (base_frames.detach().float() - previous).pow(2).mean(dim=2).sqrt()
        motion[:, 0] = 0.0
        valid_values = valid.float()
        if attention_mode == "bidirectional":
            history = (
                valid_values.sum(dim=1, keepdim=True) - 1.0
            ).clamp_min(0.0).expand(-1, steps)
        elif attention_mode == "past_only":
            history = (valid_values.cumsum(dim=1) - 1.0).clamp_min(0.0)
        elif attention_mode == "future_only":
            history = (
                valid_values.flip(1).cumsum(dim=1).flip(1) - 1.0
            ).clamp_min(0.0)
        else:
            history = torch.zeros_like(valid_values)
        history = history / float(max(self.clip_length - 1, 1))
        history[reset] = 0.0
        diagnostics = {
            "feature_delta_rms_raw": full_delta.detach().float().pow(2).mean(dim=2).sqrt(),
            "feature_delta_rms": full_delta.detach().float().pow(2).mean(dim=2).sqrt(),
            "feature_base_rms": base_frames.detach().float().pow(2).mean(dim=2).sqrt(),
            "feature_clamp_scale": torch.ones_like(valid.float()),
            "history_available_fraction": history * valid_values,
            "history_motion_rms": motion * valid.float(),
            "spatial_delta_rms": spatial_delta.detach().float().pow(2).mean(dim=2).sqrt(),
            "temporal_delta_rms": temporal_delta.detach().float().pow(2).mean(dim=2).sqrt(),
            "temporal_layer_scale": torch.stack(
                [block.attention_scale.detach().float().abs().mean() for block in self.temporal_blocks]
            ).mean(),
        }
        # Padded rows are excluded from every loss; retaining their finite
        # single-frame state avoids special-case decoder shapes.
        frames = frames * valid_float + base_frames * (1.0 - valid_float)
        return frames, diagnostics


class OnlineHMRPatchKVBlockV6(nn.Module):
    """OnlineHMR-style current-patch self attention and history cross attention."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_ratio: int,
        dropout: float,
        grid_height: int,
        grid_width: int,
        max_memory_frames: int,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.ffn_ratio = int(ffn_ratio)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.patch_count = self.grid_height * self.grid_width
        self.max_memory_frames = int(max_memory_frames)
        if self.hidden_dim <= 0 or self.hidden_dim % self.heads:
            raise ValueError("OnlineHMR hidden_dim must be divisible by heads")
        self.head_dim = self.hidden_dim // self.heads
        if self.head_dim % 8:
            raise ValueError(
                "OnlineHMR head_dim must be divisible by eight for split 2D/time RoPE"
            )
        if self.ffn_ratio < 1 or self.max_memory_frames < 1:
            raise ValueError("OnlineHMR FFN ratio and memory length must be positive")

        self.q_proj_self = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj_self = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj_self = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.q_proj_cross = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj_cross = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj_cross = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * self.ffn_ratio),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.ffn_ratio, self.hidden_dim),
        )
        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)
        self.ln3 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(
            value.shape[0], value.shape[1], self.heads, self.head_dim
        ).transpose(1, 2)

    def _spatiotemporal_rope(
        self,
        value: torch.Tensor,
        frame_positions: torch.Tensor,
        frame_count: int,
    ) -> torch.Tensor:
        batch, heads, tokens, width = value.shape
        frame_count = int(frame_count)
        if tokens != frame_count * self.patch_count:
            raise ValueError("OnlineHMR token count does not match frame_count")
        if tuple(frame_positions.shape) != (batch, frame_count):
            raise ValueError("OnlineHMR frame positions have an invalid shape")
        spatial_width = width // 2
        spatial, temporal = value.split(
            (spatial_width, width - spatial_width), dim=-1
        )
        spatial = spatial.reshape(
            batch, heads, frame_count, self.patch_count, spatial_width
        ).permute(0, 2, 1, 3, 4).reshape(
            batch * frame_count, heads, self.patch_count, spatial_width
        )
        spatial = _apply_2d_rope(
            spatial, self.grid_height, self.grid_width
        ).reshape(
            batch, frame_count, heads, self.patch_count, spatial_width
        ).permute(0, 2, 1, 3, 4).reshape(
            batch, heads, tokens, spatial_width
        )
        token_positions = frame_positions[:, :, None].expand(
            -1, -1, self.patch_count
        ).reshape(batch, tokens)
        temporal = _apply_1d_rope(temporal, token_positions)
        return torch.cat((spatial, temporal), dim=-1)

    def _self_attention(
        self, q_tokens: torch.Tensor, current_position: torch.Tensor
    ) -> torch.Tensor:
        positions = current_position[:, None]
        query = self._spatiotemporal_rope(
            self._heads(self.q_proj_self(q_tokens)), positions, 1
        )
        key = self._spatiotemporal_rope(
            self._heads(self.k_proj_self(q_tokens)), positions, 1
        )
        value = self._heads(self.v_proj_self(q_tokens))
        attended = _flash_sdpa(query, key, value)
        attended = attended.transpose(1, 2).reshape_as(q_tokens)
        return self.ln1(q_tokens + self.dropout(attended))

    def _cross_attention(
        self,
        q_tokens: torch.Tensor,
        memory: torch.Tensor,
        current_position: torch.Tensor,
        memory_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch, memory_frames, patches, channels = memory.shape
        if patches != self.patch_count or channels != self.hidden_dim:
            raise ValueError("OnlineHMR memory has an invalid shape")
        flat_memory = memory.reshape(
            batch, memory_frames * self.patch_count, self.hidden_dim
        )
        query = self._spatiotemporal_rope(
            self._heads(self.q_proj_cross(q_tokens)),
            current_position[:, None],
            1,
        )
        key = self._spatiotemporal_rope(
            self._heads(self.k_proj_cross(flat_memory)),
            memory_positions,
            memory_frames,
        )
        value = self._heads(self.v_proj_cross(flat_memory))
        attended = _flash_sdpa(query, key, value)
        attended = attended.transpose(1, 2).reshape_as(q_tokens)
        return self.ln2(q_tokens + self.dropout(attended))

    def forward(
        self,
        q_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_valid: torch.Tensor,
        current_position: torch.Tensor,
        memory_positions: torch.Tensor,
    ) -> torch.Tensor:
        q_tokens = self._self_attention(q_tokens, current_position)
        crossed = self.ln2(q_tokens)
        memory_counts = memory_valid.long().sum(dim=1)
        for count in torch.unique(memory_counts).tolist():
            count = int(count)
            if count <= 0:
                continue
            rows = torch.nonzero(memory_counts == count, as_tuple=False).flatten()
            row_output = self._cross_attention(
                q_tokens.index_select(0, rows),
                memory.index_select(0, rows)[:, -count:],
                current_position.index_select(0, rows),
                memory_positions.index_select(0, rows)[:, -count:],
            )
            crossed = crossed.index_copy(0, rows, row_output)
        return self.ln3(crossed + self.dropout(self.ffn(crossed)))

    def incremental_step(
        self,
        q_tokens: torch.Tensor,
        current_memory: torch.Tensor,
        current_position: torch.Tensor,
        cache: Mapping[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one causal step and append this frame's projected K/V to FIFO cache."""

        q_tokens = self._self_attention(q_tokens, current_position)
        cache = dict(cache or {})
        cached_key = cache.get("key")
        cached_value = cache.get("value")
        if cached_key is None:
            crossed = self.ln2(q_tokens)
        else:
            query = self._spatiotemporal_rope(
                self._heads(self.q_proj_cross(q_tokens)),
                current_position[:, None],
                1,
            )
            attended = _flash_sdpa(query, cached_key, cached_value)
            attended = attended.transpose(1, 2).reshape_as(q_tokens)
            crossed = self.ln2(q_tokens + self.dropout(attended))
        output = self.ln3(crossed + self.dropout(self.ffn(crossed)))

        current_key = self._spatiotemporal_rope(
            self._heads(self.k_proj_cross(current_memory)),
            current_position[:, None],
            1,
        )
        current_value = self._heads(self.v_proj_cross(current_memory))
        if cached_key is not None:
            current_key = torch.cat((cached_key, current_key), dim=2)
            current_value = torch.cat((cached_value, current_value), dim=2)
        maximum_tokens = self.max_memory_frames * self.patch_count
        new_cache = {
            "key": current_key[:, :, -maximum_tokens:].detach(),
            "value": current_value[:, :, -maximum_tokens:].detach(),
        }
        return output, new_cache


class OnlineHMRPatchKVFusionV6(nn.Module):
    """Causal patch-query/history-memory stack adapted from OnlineHMR."""

    MODES = ("causal", "past_only", "self_only", "memory1")

    def __init__(
        self,
        *,
        patch_channels: int = 32,
        hidden_dim: int = 512,
        clip_length: int = 8,
        grid_height: int = 16,
        grid_width: int = 12,
        layers: int = 4,
        heads: int = 4,
        ffn_ratio: int = 4,
        max_memory_frames: int = 2,
        nominal_fps: float = 30.0,
        residual_dropout: float = 0.10,
    ):
        super().__init__()
        self.patch_channels = int(patch_channels)
        self.hidden_dim = int(hidden_dim)
        self.clip_length = int(clip_length)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.layers = int(layers)
        self.heads = int(heads)
        self.ffn_ratio = int(ffn_ratio)
        self.max_memory_frames = int(max_memory_frames)
        self.nominal_fps = float(nominal_fps)
        self.residual_dropout = float(residual_dropout)
        self.patch_count = self.grid_height * self.grid_width
        self.frame_channels = self.patch_channels * self.patch_count
        if self.clip_length < 2 or self.layers < 1:
            raise ValueError("OnlineHMR clips and layer count must be positive")
        if self.nominal_fps <= 0.0:
            raise ValueError("OnlineHMR nominal_fps must be positive")
        if not 0.0 <= self.residual_dropout < 1.0:
            raise ValueError("OnlineHMR residual dropout must lie in [0, 1)")
        self.input_projection = nn.Linear(self.patch_channels, self.hidden_dim)
        self.blocks = nn.ModuleList(
            OnlineHMRPatchKVBlockV6(
                self.hidden_dim,
                self.heads,
                self.ffn_ratio,
                self.residual_dropout,
                self.grid_height,
                self.grid_width,
                self.max_memory_frames,
            )
            for _ in range(self.layers)
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_projection = nn.Linear(self.hidden_dim, self.patch_channels)
        self.dropout = nn.Dropout(self.residual_dropout)

    def config(self) -> dict[str, Any]:
        return {
            "patch_channels": self.patch_channels,
            "hidden_dim": self.hidden_dim,
            "clip_length": self.clip_length,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "layers": self.layers,
            "heads": self.heads,
            "ffn_ratio": self.ffn_ratio,
            "max_memory_frames": self.max_memory_frames,
            "nominal_fps": self.nominal_fps,
            "residual_dropout": self.residual_dropout,
        }

    def flatten_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, steps, count, channels = tokens.shape
        if count != self.patch_count or channels != self.patch_channels:
            raise ValueError("OnlineHMR patch-token shape violates FullGrid order")
        return tokens.reshape(
            batch, steps, self.grid_height, self.grid_width, channels
        ).permute(0, 1, 4, 2, 3).contiguous().flatten(2)

    def _history_memory(
        self,
        projected: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
        maximum_frames: int,
        force_reset: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, steps, patches, channels = projected.shape
        memory_frames = []
        memory_positions = []
        memory_valid = []
        for lag in range(self.max_memory_frames, 0, -1):
            indices = (torch.arange(steps, device=projected.device) - lag).clamp_min(0)
            memory_frames.append(projected.index_select(1, indices))
            memory_positions.append(positions.index_select(1, indices))
            slot_valid = valid.index_select(1, indices) & (
                torch.arange(steps, device=projected.device)[None] >= lag
            )
            memory_valid.append(slot_valid)
        memory = torch.stack(memory_frames, dim=2)
        memory_time = torch.stack(memory_positions, dim=2)
        available = torch.stack(memory_valid, dim=2)
        if maximum_frames < self.max_memory_frames:
            available[:, :, : self.max_memory_frames - maximum_frames] = False
        available[force_reset] = False
        memory = memory.reshape(
            batch * steps,
            self.max_memory_frames,
            patches,
            channels,
        )
        memory_time = memory_time.reshape(batch * steps, self.max_memory_frames)
        available = available.reshape(batch * steps, self.max_memory_frames)
        return memory, memory_time, available

    def forward(
        self,
        patch_tokens: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        attention_mode: str = "causal",
        force_reset: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del crop_affine
        if attention_mode not in self.MODES:
            raise ValueError(f"Unsupported OnlineHMR attention mode={attention_mode!r}")
        batch, steps, count, channels = patch_tokens.shape
        expected = (batch, self.clip_length, self.patch_count, self.patch_channels)
        if tuple(patch_tokens.shape) != expected:
            raise ValueError(
                f"OnlineHMR patch tokens must be {expected}, got {tuple(patch_tokens.shape)}"
            )
        if tuple(time.shape) != (batch, steps) or tuple(valid.shape) != (batch, steps):
            raise ValueError("OnlineHMR time/valid shapes are incompatible")
        reset = (
            torch.zeros(batch, device=patch_tokens.device, dtype=torch.bool)
            if force_reset is None
            else force_reset.bool()
        )
        if tuple(reset.shape) != (batch,):
            raise ValueError("OnlineHMR force_reset must contain one flag per clip")
        valid_bool = valid > 0.5
        positions = (time.float() - time[:, :1].float()) * self.nominal_fps
        projected = self.input_projection(patch_tokens)
        maximum_frames = (
            0
            if attention_mode == "self_only"
            else 1
            if attention_mode == "memory1"
            else self.max_memory_frames
        )
        memory, memory_time, memory_valid = self._history_memory(
            projected, positions, valid_bool, maximum_frames, reset
        )
        query = projected.reshape(batch * steps, count, self.hidden_dim)
        query_time = positions.reshape(batch * steps)
        for block in self.blocks:
            query = block(
                query, memory, memory_valid, query_time, memory_time
            )
        query = query.reshape(batch, steps, count, self.hidden_dim)
        fused_tokens = self.dropout(
            self.output_projection(self.output_norm(query))
        )
        base_frames = self.flatten_patch_tokens(patch_tokens)
        frames = self.flatten_patch_tokens(fused_tokens)
        valid_float = valid[:, :, None].to(dtype=frames.dtype)
        frames = frames * valid_float + base_frames * (1.0 - valid_float)
        feature_delta = frames - base_frames
        previous = torch.roll(base_frames.detach().float(), shifts=1, dims=1)
        motion = (base_frames.detach().float() - previous).pow(2).mean(dim=2).sqrt()
        motion[:, 0] = 0.0
        memory_count = memory_valid.reshape(batch, steps, -1).float().sum(dim=2)
        diagnostics = {
            "feature_delta_rms_raw": feature_delta.detach().float().pow(2).mean(dim=2).sqrt(),
            "feature_delta_rms": feature_delta.detach().float().pow(2).mean(dim=2).sqrt(),
            "feature_base_rms": base_frames.detach().float().pow(2).mean(dim=2).sqrt(),
            "feature_clamp_scale": torch.ones_like(valid.float()),
            "history_available_fraction": (
                memory_count / float(self.max_memory_frames)
            ) * valid.float(),
            "history_motion_rms": motion * valid.float(),
            "spatial_delta_rms": torch.zeros_like(valid.float()),
            "temporal_delta_rms": feature_delta.detach().float().pow(2).mean(dim=2).sqrt(),
            "temporal_layer_scale": frames.new_ones(()),
        }
        return frames, diagnostics

    def forward_incremental(
        self,
        patch_tokens: torch.Tensor,
        time_position: torch.Tensor,
        cache: Sequence[Mapping[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        """Decode one frame using only cached historical K/V tensors."""

        if tuple(patch_tokens.shape[1:]) != (
            self.patch_count,
            self.patch_channels,
        ):
            raise ValueError("OnlineHMR incremental patch shape is invalid")
        projected = self.input_projection(patch_tokens)
        query = projected
        cache = list(cache or ())
        new_cache = []
        for index, block in enumerate(self.blocks):
            layer_cache = cache[index] if index < len(cache) else None
            query, layer_cache = block.incremental_step(
                query, projected, time_position.float(), layer_cache
            )
            new_cache.append(layer_cache)
        fused = self.output_projection(self.output_norm(query))
        frame = self.flatten_patch_tokens(fused[:, None])[:, 0]
        return frame, new_cache

    @torch.no_grad()
    def incremental_equivalence_max_abs(
        self,
        patch_tokens: torch.Tensor,
        time: torch.Tensor,
    ) -> float:
        """Verify batched causal fusion matches frame-by-frame KV inference."""

        if self.training:
            raise RuntimeError("KV equivalence requires eval mode")
        batch, steps = patch_tokens.shape[:2]
        valid = torch.ones(batch, steps, device=patch_tokens.device)
        affine = torch.eye(
            3, device=patch_tokens.device, dtype=patch_tokens.dtype
        ).reshape(1, 1, 3, 3).expand(batch, steps, -1, -1)
        batched, _ = self(
            patch_tokens, time, valid, affine, attention_mode="causal"
        )
        positions = (time.float() - time[:, :1].float()) * self.nominal_fps
        cache = None
        incremental = []
        for step in range(steps):
            frame, cache = self.forward_incremental(
                patch_tokens[:, step], positions[:, step], cache
            )
            incremental.append(frame)
        incremental_frames = torch.stack(incremental, dim=1)
        return float((batched.float() - incremental_frames.float()).abs().max().item())


def _reset_module_parameters(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


class FullGrid6144TemporalMainTrunkV5(TemporalClipMainTrunkV4):
    """Fresh bidirectional FullGrid6144 trunk with independent dense twin heads."""

    def __init__(
        self,
        base_model: DinoTactileModel,
        palm_vertex_indices: Sequence[int],
        fusion: FullGrid6144SpatiotemporalFusionV5,
        *,
        online_encoder_chunk_size: int = 128,
    ):
        super().__init__(
            base_model,
            palm_vertex_indices,
            fusion,
            online_encoder_chunk_size=online_encoder_chunk_size,
        )
        pool = self.decoder[0]
        if not hasattr(pool, "projection") or int(pool.output_dim) != fusion.frame_channels:
            raise ValueError("FullGrid6144 fusion does not match the FullGrid projection")
        self.contact_head = copy.deepcopy(self.decoder[1:])
        self.contact_head.apply(_reset_module_parameters)

    def train(self, mode: bool = True):
        super().train(mode)
        self.contact_head.train(mode)
        return self

    def compact_state_dict(self) -> dict[str, torch.Tensor]:
        state = super().compact_state_dict()
        state.update(
            {
                name: value
                for name, value in self.state_dict().items()
                if name.startswith("contact_head.")
            }
        )
        return state

    def load_compact_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing_trainable = [
            name
            for name in incompatible.missing_keys
            if name.startswith("base_model.tactile_head.")
            or name.startswith("fusion.")
            or name.startswith("contact_head.")
        ]
        if unexpected or missing_trainable:
            raise RuntimeError(
                "FullGrid6144 trunk state mismatch: "
                f"missing_trainable={missing_trainable}, unexpected={unexpected}"
            )

    def _project_patch_tokens(self, grids: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = grids.shape
        projected = self.decoder[0].projection(
            grids.reshape(batch * steps, channels, height, width)
        )
        return projected.flatten(2).transpose(1, 2).reshape(
            batch, steps, height * width, projected.shape[1]
        )

    def _decode_frame_features(
        self, frame_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = frame_features.reshape(-1, frame_features.shape[-1])
        tactile = self.decoder[1:](flat)
        contact = self.contact_head(flat)
        return tactile, contact

    def forward_clip(
        self,
        grids: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        force_reset: torch.Tensor | None = None,
        decode_base: bool = True,
        attention_mode: str = "bidirectional",
    ) -> dict[str, torch.Tensor]:
        patch_tokens = self._project_patch_tokens(grids)
        frame_features, diagnostics = self.fusion(
            patch_tokens,
            time,
            valid,
            crop_affine,
            attention_mode=attention_mode,
            force_reset=force_reset,
        )
        batch, steps = valid.shape
        tactile_full, contact_full = self._decode_frame_features(frame_features)
        tactile_logits = tactile_full.index_select(
            1, self.palm_vertex_indices
        ).reshape(batch, steps, -1)
        contact_logits = contact_full.index_select(
            1, self.palm_vertex_indices
        ).reshape(batch, steps, -1)
        if decode_base:
            reset_features, _ = self.fusion(
                patch_tokens,
                time,
                valid,
                crop_affine,
                attention_mode="self_only",
                force_reset=torch.ones(batch, device=valid.device, dtype=torch.bool),
            )
            base_full, _ = self._decode_frame_features(reset_features)
            base_logits = base_full.index_select(
                1, self.palm_vertex_indices
            ).reshape(batch, steps, -1)
        else:
            base_logits = tactile_logits.detach()
        decoder_delta = (
            tactile_logits.float() - base_logits.float()
            if decode_base
            else torch.zeros_like(tactile_logits, dtype=torch.float32)
        )
        output = {
            "pred_logits": tactile_logits,
            "pred_tactile": torch.sigmoid(tactile_logits),
            "contact_logits": contact_logits,
            "contact_probability": torch.sigmoid(contact_logits),
            "base_pred_logits": base_logits,
            "base_pred_tactile": torch.sigmoid(base_logits),
            "decoder_logit_delta": decoder_delta,
            "feature_delta": frame_features
            - self.fusion.flatten_patch_tokens(patch_tokens),
            "shared_frame_features": frame_features,
            "base_decode_performed": decode_base,
        }
        output.update(diagnostics)
        return output


class OnlineHMRPatchKVTemporalMainTrunkV6(FullGrid6144TemporalMainTrunkV5):
    """Fresh causal patch-memory trunk with independent tactile/contact heads."""

    def __init__(
        self,
        base_model: DinoTactileModel,
        palm_vertex_indices: Sequence[int],
        fusion: OnlineHMRPatchKVFusionV6,
        *,
        online_encoder_chunk_size: int = 128,
    ):
        super().__init__(
            base_model,
            palm_vertex_indices,
            fusion,
            online_encoder_chunk_size=online_encoder_chunk_size,
        )

    def forward_clip(
        self,
        grids: torch.Tensor,
        time: torch.Tensor,
        valid: torch.Tensor,
        crop_affine: torch.Tensor,
        *,
        force_reset: torch.Tensor | None = None,
        decode_base: bool = True,
        attention_mode: str = "causal",
    ) -> dict[str, torch.Tensor]:
        return super().forward_clip(
            grids,
            time,
            valid,
            crop_affine,
            force_reset=force_reset,
            decode_base=decode_base,
            attention_mode=attention_mode,
        )


def build_fresh_temporal_base(
    dino_weights: str | Path,
    *,
    input_resolution: Sequence[int] = (256, 192),
    model_initialization_order: str = CANONICAL_MODEL_INITIALIZATION_ORDER,
) -> DinoTactileModel:
    """Construct the canonical FullGrid/ReZero RGB base without a tactile ckpt."""

    return DinoTactileModel(
        tactile_head_type="dense_v2_dino_rezero",
        backbone_feature_layers=(8, 16, 24, 32),
        visual_backbone="dinov3_hplus",
        dino_weights=str(Path(dino_weights).expanduser().resolve(strict=True)),
        dino_residual_max_scale=0.10,
        dino_residual_rms_budget=0.50,
        pool_layout="fullgrid32",
        decoder_dropout_scale=1.0,
        input_resolution=input_resolution,
        pool_output_channels=32,
        decoder_hidden_dim=512,
        model_initialization_order=model_initialization_order,
    )


class TemporalGridTactileModel(nn.Module):
    """Controlled probe for temporal fusion before the FullGrid decoder.

    The frozen decoder is an attribution instrument for V2, not the intended
    final deployment boundary. Successful fusion must later be moved into the
    jointly trained feature trunk.
    """

    def __init__(
        self,
        decoder: nn.Sequential,
        palm_vertex_indices: Sequence[int],
        adapter: TemporalGridAdapterV1,
        online_encoder: nn.Module | None = None,
        online_encoder_chunk_size: int = 128,
    ):
        super().__init__()
        self.decoder = decoder.requires_grad_(False).eval()
        self.adapter = adapter
        self.online_encoder = online_encoder
        self.online_encoder_chunk_size = int(online_encoder_chunk_size)
        if self.online_encoder_chunk_size <= 0:
            raise ValueError("online_encoder_chunk_size must be positive")
        if self.online_encoder is not None:
            self.online_encoder.requires_grad_(False)
            self.online_encoder.eval()
        self.register_buffer(
            "palm_vertex_indices",
            torch.as_tensor(tuple(int(value) for value in palm_vertex_indices), dtype=torch.long),
            persistent=True,
        )
        if self.palm_vertex_indices.numel() == 0:
            raise ValueError("palm_vertex_indices cannot be empty")

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        if self.online_encoder is not None:
            self.online_encoder.eval()
        self.adapter.train(mode)
        return self

    def trainable_parameters(self):
        return self.adapter.parameters()

    @property
    def fusion(self):
        return self.adapter

    def materialize_online_features(
        self, batch: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Encode current and historical RGB once for one online batch."""

        if "current_grid" in batch:
            return dict(batch)
        if self.online_encoder is None:
            raise RuntimeError(
                "The batch contains online RGB but the temporal model has no online encoder"
            )
        if "current_image" not in batch or "history_images" not in batch:
            raise KeyError("Online temporal batch needs current_image/history_images")
        result = dict(batch)
        current_image = batch["current_image"]
        history_images = batch["history_images"]
        if current_image.ndim != 4 or history_images.ndim != 5:
            raise ValueError("Online temporal images have incompatible dimensions")
        batch_size, lag_count = history_images.shape[:2]
        groups: list[tuple[str, torch.Tensor, tuple[int, ...]]] = [
            ("current_grid", current_image, (batch_size,)),
            (
                "history_grids",
                history_images.reshape(-1, *history_images.shape[2:]),
                (batch_size, lag_count),
            ),
        ]
        for image_key, grid_key in (
            ("control_history_images", "control_history_grids"),
            ("contralateral_history_images", "contralateral_history_grids"),
        ):
            images = batch.get(image_key)
            if images is None:
                continue
            if tuple(images.shape[:2]) != (batch_size, lag_count):
                raise ValueError(f"{image_key} has an incompatible shape")
            groups.append(
                (
                    grid_key,
                    images.reshape(-1, *images.shape[2:]),
                    (batch_size, lag_count),
                )
            )
        encoded_groups = []
        with torch.no_grad():
            for _, images, _ in groups:
                chunks = []
                for start in range(0, len(images), self.online_encoder_chunk_size):
                    chunks.append(
                        self.online_encoder(
                            images[start : start + self.online_encoder_chunk_size]
                        )
                    )
                encoded_groups.append(torch.cat(chunks, dim=0))
        for (grid_key, _, leading_shape), grids in zip(groups, encoded_groups):
            if len(leading_shape) == 1:
                result[grid_key] = grids
            else:
                result[grid_key] = grids.reshape(
                    *leading_shape, *grids.shape[1:]
                )
        return result

    def forward(
        self,
        current_grid: torch.Tensor,
        history_grids: torch.Tensor,
        history_time_gap: torch.Tensor,
        history_available: torch.Tensor,
        *,
        history_crop_transform: torch.Tensor | None = None,
        cached_base_logits: torch.Tensor | None = None,
        decoded_current_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        fused_grid, diagnostics = self.adapter(
            current_grid,
            history_grids,
            history_time_gap,
            history_available,
            history_crop_transform,
        )
        if decoded_current_logits is None:
            decoded_current_logits = self.decoder(current_grid).index_select(
                1, self.palm_vertex_indices
            )
        decoded_fused_logits = self.decoder(fused_grid).index_select(
            1, self.palm_vertex_indices
        )
        decoder_delta = decoded_fused_logits.float() - decoded_current_logits.float()
        base_logits = (
            decoded_current_logits.float()
            if cached_base_logits is None
            else cached_base_logits.float()
        )
        fused_logits = base_logits + decoder_delta
        output = {
            "pred_logits": fused_logits,
            "pred_tactile": torch.sigmoid(fused_logits),
            "base_pred_logits": base_logits,
            "base_pred_tactile": torch.sigmoid(base_logits),
            "decoded_current_logits": decoded_current_logits,
            "decoder_logit_delta": decoder_delta,
            "feature_delta": fused_grid - current_grid,
        }
        output.update(diagnostics)
        return output


class FrozenOnlineGridEncoder(nn.Module):
    """Run frozen DINO and ReZero fusion without retaining feature tensors."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        if not hasattr(base_model, "_extract_tactile_features"):
            raise TypeError("Online encoder requires a DinoTactileModel-compatible base")
        head = getattr(base_model, "tactile_head", None)
        if head is None or not hasattr(head, "_fuse"):
            raise TypeError("Online encoder requires a ReZero tactile head with _fuse")
        self.base_model = base_model.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.base_model.eval()
        return self

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.base_model.eval()
        levels = self.base_model._extract_tactile_features(image)
        return self.base_model.tactile_head._fuse(levels).detach()


def controlled_temporal_grid_inputs(
    batch: Mapping[str, torch.Tensor],
    source: str,
    *,
    shuffle_seed: int = 521,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Select one causal/counterfactual history without changing current RGB."""

    source = str(source).strip().lower()
    if source not in TEMPORAL_GRID_SOURCES:
        raise ValueError(
            f"Unsupported temporal grid source={source!r}; choose {TEMPORAL_GRID_SOURCES}"
        )
    current = batch["current_grid"]
    history = batch["history_grids"]
    time_gap = batch["history_time_gap"]
    available = batch["history_available"]
    transform = batch.get("history_crop_transform")
    if source == "rgb_reset":
        identity = None
        if transform is not None:
            identity = torch.eye(
                3, device=transform.device, dtype=transform.dtype
            ).reshape(1, 1, 3, 3).expand_as(transform)
        return (
            current[:, None].expand_as(history),
            torch.zeros_like(time_gap),
            torch.zeros_like(available),
            identity,
        )
    if source == "cross_sequence":
        history = batch["control_history_grids"]
        time_gap = batch["control_history_time_gap"]
        available = batch["control_history_available"]
        transform = batch.get("control_history_crop_transform")
    elif source == "contralateral":
        history = batch["contralateral_history_grids"]
        available = batch["contralateral_history_available"]
        transform = batch.get("contralateral_history_crop_transform")
    elif source == "lag_reverse":
        history = history.flip(1)
        time_gap = time_gap.flip(1)
        available = available.flip(1)
        if transform is not None:
            transform = transform.flip(1)
    elif source == "spatial_shuffle":
        batch_size, lag_count, channels, height, width = history.shape
        generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
        permutation = torch.randperm(height * width, generator=generator).to(
            device=history.device
        )
        history = history.flatten(-2).index_select(-1, permutation).reshape(
            batch_size, lag_count, channels, height, width
        )
    elif source == "affine_perturb":
        if transform is None:
            raise ValueError("affine_perturb requires history crop transforms")
        transform = transform.clone()
        # Move the local-search center by two DINO tokens without changing the
        # history tensor or any current-frame input.
        transform[:, :, 0, 2] += 32.0
    return history, time_gap, available, transform


def temporal_grid_tiny_check() -> dict[str, float]:
    """CPU check for identity, gradients, RMS limiting, and controls."""

    torch.manual_seed(7)
    adapter = TemporalGridAdapterV1(
        grid_channels=8,
        hidden_channels=4,
        history_lags=(1, 2, 4, 8),
        feature_rms_budget=0.05,
    )
    current = torch.randn(2, 8, 4, 3)
    history = torch.randn(2, 4, 8, 4, 3)
    gaps = torch.tensor(((0.03, 0.06, 0.12, 0.24),) * 2)
    available = torch.ones(2, 4)
    initial, _ = adapter(current, history, gaps, available)
    initial_error = float((initial - current).abs().max())
    loss = initial.square().mean()
    loss.backward()
    output_gradient = float(adapter.output_projection.weight.grad.abs().sum())
    if initial_error != 0.0:
        raise AssertionError(f"zero-init adapter drifted by {initial_error}")
    if not math.isfinite(output_gradient) or output_gradient <= 0.0:
        raise AssertionError("zero-init output projection did not receive a gradient")
    optimizer = torch.optim.SGD(adapter.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second, _ = adapter(current, history, gaps, available)
    second_loss = second.square().mean()
    second_loss.backward()
    all_finite_after_step = all(
        bool(torch.isfinite(parameter).all())
        and (
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all())
        )
        for parameter in adapter.parameters()
    )
    if not all_finite_after_step or not bool(torch.isfinite(second_loss)):
        raise AssertionError("adapter became non-finite after its first optimizer step")
    with torch.no_grad():
        adapter.output_projection.weight.normal_(std=0.1)
        repeated, _ = adapter(
            current, current[:, None].expand_as(history), gaps, available
        )
        reset, _ = adapter(
            current,
            current[:, None].expand_as(history),
            torch.zeros_like(gaps),
            torch.zeros_like(available),
        )
        fused, _ = adapter(current, history, gaps, available)
        base_rms = current.pow(2).mean(dim=(1, 2, 3)).sqrt()
        delta_rms = (fused - current).pow(2).mean(dim=(1, 2, 3)).sqrt()
        maximum_ratio = float((delta_rms / base_rms).max())
        repeated_error = float((repeated - current).abs().max())
        reset_error = float((reset - current).abs().max())
    if repeated_error > 1e-7 or reset_error > 1e-7:
        raise AssertionError("zero-motion/reset path is not an exact identity")
    if maximum_ratio > adapter.feature_rms_budget + 1e-5:
        raise AssertionError("feature residual exceeded its RMS budget")
    result = {
        "initial_identity_max_error": initial_error,
        "repeated_identity_max_error": repeated_error,
        "reset_identity_max_error": reset_error,
        "output_projection_gradient_l1": output_gradient,
        "second_step_loss": float(second_loss.detach()),
        "maximum_feature_rms_ratio": maximum_ratio,
    }
    memory = TemporalLocalMemoryFusionV2(
        grid_channels=8,
        hidden_channels=4,
        history_lags=(1, 2),
        patch_size=1,
        search_window=3,
        feature_rms_budget=0.05,
    )
    memory_history = history[:, :2]
    memory_gaps = gaps[:, :2]
    memory_available = available[:, :2]
    transforms = torch.eye(3).reshape(1, 1, 3, 3).expand(2, 2, -1, -1).clone()
    memory_initial, memory_diagnostics = memory(
        current,
        memory_history,
        memory_gaps,
        memory_available,
        transforms,
    )
    memory_initial_error = float((memory_initial - current).abs().max())
    memory_initial.square().mean().backward()
    memory_gradient = float(memory.output_projection.weight.grad.abs().sum())
    if memory_initial_error != 0.0:
        raise AssertionError(
            f"zero-init local-memory fusion drifted by {memory_initial_error}"
        )
    if not math.isfinite(memory_gradient) or memory_gradient <= 0.0:
        raise AssertionError("local-memory output projection did not receive a gradient")
    memory.zero_grad(set_to_none=True)
    counterfactual, _ = memory(
        current,
        memory_history,
        memory_gaps,
        memory_available,
        transforms,
    )
    counterfactual_delta = counterfactual.float() - current.float()
    counterfactual_energy = counterfactual_delta.pow(2).mean(dim=(1, 2, 3))
    base_energy = current.float().pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    counterfactual_identity = (
        counterfactual_energy
        / (memory.feature_rms_budget**2 * base_energy)
    ).mean()
    counterfactual_identity.backward()
    if not bool(torch.isfinite(counterfactual_identity)) or not all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in memory.parameters()
    ):
        raise AssertionError("zero-write counterfactual identity produced non-finite gradients")
    with torch.no_grad():
        memory.output_projection.weight.normal_(std=0.1)
        memory_reset, _ = memory(
            current,
            current[:, None].expand(-1, 2, -1, -1, -1),
            torch.zeros_like(memory_gaps),
            torch.zeros_like(memory_available),
            transforms,
        )
        memory_fused, _ = memory(
            current,
            memory_history,
            memory_gaps,
            memory_available,
            transforms,
        )
        memory_ratio = (
            (memory_fused - current).pow(2).mean(dim=(1, 2, 3)).sqrt()
            / current.pow(2).mean(dim=(1, 2, 3)).sqrt()
        ).max()
    memory_reset_error = float((memory_reset - current).abs().max())
    if memory_reset_error > 1e-7:
        raise AssertionError("local-memory reset path is not an exact identity")
    if float(memory_ratio) > memory.feature_rms_budget + 1e-5:
        raise AssertionError("local-memory residual exceeded its RMS budget")
    result.update(
        {
            "memory_initial_identity_max_error": memory_initial_error,
            "memory_reset_identity_max_error": memory_reset_error,
            "memory_output_projection_gradient_l1": memory_gradient,
            "memory_maximum_feature_rms_ratio": float(memory_ratio),
            "memory_match_confidence": float(
                memory_diagnostics["match_confidence"].mean()
            ),
            "memory_zero_write_identity_loss": float(counterfactual_identity.detach()),
        }
    )
    torch.manual_seed(19)
    short = HierarchicalTemporalMemoryFusionV3(
        grid_channels=8,
        hidden_channels=4,
        history_lags=(1, 2),
        patch_size=1,
        search_window=3,
    )
    torch.manual_seed(19)
    long = HierarchicalTemporalMemoryFusionV3(
        grid_channels=8,
        hidden_channels=4,
        history_lags=(1, 2, 4, 8, 16),
        patch_size=1,
        search_window=3,
    )
    fast_state_equal = all(
        torch.equal(value, long.fast_writer.state_dict()[name])
        for name, value in short.fast_writer.state_dict().items()
    )
    if not fast_state_equal:
        raise AssertionError("L12 and L124816 fast writers initialized differently")
    long_history = torch.randn(2, 5, 8, 4, 3)
    long_gaps = torch.tensor(((0.03, 0.06, 0.12, 0.24, 0.48),) * 2)
    long_available = torch.ones(2, 5)
    long_transforms = torch.eye(3).reshape(1, 1, 3, 3).expand(
        2, 5, -1, -1
    ).clone()
    long_initial, _ = long(
        current,
        long_history,
        long_gaps,
        long_available,
        long_transforms,
    )
    long_identity_error = float((long_initial - current).abs().max())
    long_initial.square().mean().backward()
    fast_gradient = float(
        long.fast_writer.output_projection.weight.grad.abs().sum()
    )
    medium_gradient = float(
        long.medium_writer.output_projection.weight.grad.abs().sum()
    )
    if long_identity_error != 0.0 or fast_gradient <= 0.0 or medium_gradient <= 0.0:
        raise AssertionError(
            "Hierarchical writer failed zero-init identity or first-step gradients"
        )
    result.update(
        {
            "hierarchical_initial_identity_max_error": long_identity_error,
            "hierarchical_fast_writer_state_equal": float(fast_state_equal),
            "hierarchical_fast_projection_gradient_l1": fast_gradient,
            "hierarchical_medium_projection_gradient_l1": medium_gradient,
        }
    )
    class _PairedDecoderHarness(nn.Module):
        _decode_paired_grids = TemporalMainTrunkV3._decode_paired_grids

        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(8, 4, bias=False),
            )

    paired_decoder = _PairedDecoderHarness().train()
    paired_current = torch.randn(32, 8)
    paired_fused = paired_current.detach().clone().requires_grad_(True)
    paired_base_output, paired_fused_output = paired_decoder._decode_paired_grids(
        paired_current, paired_fused
    )
    paired_decode_error = float(
        (paired_base_output - paired_fused_output).abs().max()
    )
    paired_fused_output.square().mean().backward()
    paired_gradient = float(paired_fused.grad.abs().sum())
    if paired_decode_error != 0.0 or not math.isfinite(paired_gradient) or paired_gradient <= 0.0:
        raise AssertionError(
            "Shared-dropout paired decoder lost identity or fused-input gradients"
        )
    result.update(
        {
            "paired_decoder_identity_max_error": paired_decode_error,
            "paired_decoder_fused_input_gradient_l1": paired_gradient,
        }
    )
    return result


def temporal_clip_tiny_check() -> dict[str, float]:
    """CPU audit for clip identity, causality, masks, gradients, and RMS cap."""

    torch.manual_seed(17)
    fusion = CausalClipTransformerFusionV4(
        grid_channels=8,
        hidden_channels=8,
        clip_length=8,
        grid_height=4,
        grid_width=3,
        layers=2,
        heads=2,
        ffn_ratio=2,
        feature_rms_budget=0.05,
    )
    grids = torch.randn(2, 8, 8, 4, 3)
    time = torch.arange(8, dtype=torch.float32)[None].expand(2, -1) / 30.0
    valid = torch.tensor(((1,) * 8, (1,) * 5 + (0,) * 3), dtype=torch.float32)
    affine = torch.eye(3).reshape(1, 1, 3, 3).expand(2, 8, -1, -1).clone()
    initial, _ = fusion(grids, time, valid, affine)
    initial_error = float((initial - grids).abs().max())
    initial.square().mean().backward()
    projection_gradient = float(fusion.output_projection.weight.grad.abs().sum())
    if initial_error != 0.0 or not math.isfinite(projection_gradient) or projection_gradient <= 0:
        raise AssertionError("Causal clip zero-init/gradient contract failed")
    with torch.no_grad():
        fusion.output_projection.weight.normal_(std=0.05)
        real, diagnostics = fusion(grids, time, valid, affine)
        reset, _ = fusion(
            grids,
            time,
            valid,
            affine,
            force_reset=torch.ones(2, dtype=torch.bool),
        )
        perturbed = grids.clone()
        perturbed[:, -1].add_(100.0)
        future, _ = fusion(perturbed, time, valid, affine)
    reset_error = float((reset - grids).abs().max())
    first_frame_error = float((real[:, 0] - grids[:, 0]).abs().max())
    padded_error = float((real[1, 5:] - grids[1, 5:]).abs().max())
    future_leakage = float((future[:, :-1] - real[:, :-1]).abs().max())
    base_rms = grids.float().pow(2).mean(dim=(2, 3, 4)).sqrt()
    delta_rms = (real - grids).float().pow(2).mean(dim=(2, 3, 4)).sqrt()
    maximum_ratio = float((delta_rms / base_rms.clamp_min(1e-12)).max())
    if reset_error > 1e-7 or first_frame_error > 1e-7 or padded_error > 1e-7:
        raise AssertionError("Causal clip reset/first/padding identity failed")
    if future_leakage > 1e-5:
        raise AssertionError(f"Causal clip leaked future information: {future_leakage}")
    if maximum_ratio > fusion.feature_rms_budget + 1e-5:
        raise AssertionError("Causal clip feature residual exceeded its RMS budget")
    return {
        "initial_identity_max_error": initial_error,
        "reset_identity_max_error": reset_error,
        "first_frame_identity_max_error": first_frame_error,
        "padding_identity_max_error": padded_error,
        "future_leakage_max_error": future_leakage,
        "output_projection_gradient_l1": projection_gradient,
        "maximum_feature_rms_ratio": maximum_ratio,
        "mean_history_available_fraction": float(
            diagnostics["history_available_fraction"].mean()
        ),
    }


def temporal_fullgrid_tiny_check() -> dict[str, float]:
    """Small CPU contract for FullGrid ordering, controls, and gradients."""

    torch.manual_seed(29)
    fusion = FullGrid6144SpatiotemporalFusionV5(
        patch_channels=8,
        clip_length=4,
        grid_height=2,
        grid_width=2,
        spatial_layers=1,
        spatial_heads=2,
        spatial_ffn_ratio=2,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ffn_ratio=2,
        residual_dropout=0.0,
        layer_scale_init=1e-3,
    )
    tokens = torch.randn(2, 4, 4, 8, requires_grad=True)
    time = torch.tensor(
        ((0.0, 0.04, 0.11, 0.21), (0.0, 0.03, 0.09, 0.09)),
        dtype=torch.float32,
    )
    valid = torch.tensor(((1, 1, 1, 1), (1, 1, 1, 0)), dtype=torch.float32)
    affine = torch.eye(3).reshape(1, 1, 3, 3).expand(2, 4, -1, -1).clone()

    ordered = torch.arange(32, dtype=torch.float32).reshape(1, 1, 4, 8)
    flattened = fusion.flatten_patch_tokens(ordered)
    expected = ordered.reshape(1, 1, 2, 2, 8).permute(
        0, 1, 4, 2, 3
    ).reshape(1, 1, 32)
    ordering_error = float((flattened - expected).abs().max())
    if ordering_error != 0.0:
        raise AssertionError("FullGrid token flattening changed channel-major order")

    fused, diagnostics = fusion(tokens, time, valid, affine)
    loss = (fused * valid[:, :, None]).square().mean()
    loss.backward()
    spatial_gradient = float(
        fusion.spatial_blocks[0].qkv.weight.grad.float().abs().sum()
    )
    temporal_gradient = float(
        fusion.temporal_blocks[0].qkv.weight.grad.float().abs().sum()
    )
    if not bool(torch.isfinite(fused).all()) or not (
        math.isfinite(spatial_gradient)
        and math.isfinite(temporal_gradient)
        and spatial_gradient > 0.0
        and temporal_gradient > 0.0
    ):
        raise AssertionError("FullGrid6144 attention produced invalid outputs or gradients")

    with torch.no_grad():
        clean_bi, _ = fusion(tokens.detach(), time, valid, affine)
        clean_past, _ = fusion(
            tokens.detach(), time, valid, affine, attention_mode="past_only"
        )
        perturbed = tokens.detach().clone()
        perturbed[:, -1].add_(10.0)
        changed_bi, _ = fusion(perturbed, time, valid, affine)
        changed_past, _ = fusion(
            perturbed, time, valid, affine, attention_mode="past_only"
        )
        bidirectional_future_effect = float(
            (changed_bi[0, :-1] - clean_bi[0, :-1]).abs().max()
        )
        past_future_leakage = float(
            (changed_past[0, :-1] - clean_past[0, :-1]).abs().max()
        )
        padded_error = float(
            (
                fused.detach()[1, -1]
                - fusion.flatten_patch_tokens(tokens.detach())[1, -1]
            ).abs().max()
        )
    if bidirectional_future_effect <= 0.0:
        raise AssertionError("Bidirectional FullGrid attention did not use a future frame")
    if past_future_leakage > 1e-5:
        raise AssertionError(
            f"Past-only FullGrid control leaked future input: {past_future_leakage}"
        )
    if padded_error > 1e-6:
        raise AssertionError("Padded FullGrid frame did not preserve its finite base state")
    return {
        "flatten_ordering_max_error": ordering_error,
        "spatial_qkv_gradient_l1": spatial_gradient,
        "temporal_qkv_gradient_l1": temporal_gradient,
        "bidirectional_future_effect": bidirectional_future_effect,
        "past_only_future_leakage": past_future_leakage,
        "padding_identity_max_error": padded_error,
        "feature_delta_rms": float(diagnostics["feature_delta_rms"].mean()),
    }


def temporal_onlinehmr_tiny_check() -> dict[str, float]:
    """CPU contract for causality, FIFO memory, gradients, and KV equivalence."""

    torch.manual_seed(41)
    fusion = OnlineHMRPatchKVFusionV6(
        patch_channels=8,
        hidden_dim=32,
        clip_length=4,
        grid_height=2,
        grid_width=2,
        layers=2,
        heads=4,
        ffn_ratio=2,
        max_memory_frames=2,
        residual_dropout=0.0,
    )
    tokens = torch.randn(2, 4, 4, 8, requires_grad=True)
    time = torch.tensor(
        ((0.0, 0.03, 0.08, 0.16), (0.0, 0.04, 0.10, 0.18)),
        dtype=torch.float32,
    )
    valid = torch.ones(2, 4, dtype=torch.float32)
    affine = torch.eye(3).reshape(1, 1, 3, 3).expand(2, 4, -1, -1).clone()
    fused, diagnostics = fusion(tokens, time, valid, affine)
    fused.square().mean().backward()
    self_gradient = float(
        fusion.blocks[0].q_proj_self.weight.grad.float().abs().sum()
    )
    cross_gradient = float(
        fusion.blocks[0].q_proj_cross.weight.grad.float().abs().sum()
    )
    if not bool(torch.isfinite(fused).all()) or min(
        self_gradient, cross_gradient
    ) <= 0.0:
        raise AssertionError("OnlineHMR patch attention lost finite gradients")

    fusion.eval()
    with torch.no_grad():
        clean, _ = fusion(tokens.detach(), time, valid, affine)
        future_tokens = tokens.detach().clone()
        future_tokens[:, -1].add_(10.0)
        changed_future, _ = fusion(future_tokens, time, valid, affine)
        future_leakage = float(
            (changed_future[:, :-1] - clean[:, :-1]).abs().max()
        )
        history_tokens = tokens.detach().clone()
        history_tokens[:, 1].add_(10.0)
        changed_history, _ = fusion(history_tokens, time, valid, affine)
        history_effect = float((changed_history[:, -1] - clean[:, -1]).abs().max())
        kv_error = fusion.incremental_equivalence_max_abs(
            tokens.detach(), time
        )
    if future_leakage > 1e-5:
        raise AssertionError(f"OnlineHMR leaked future input: {future_leakage}")
    if history_effect <= 0.0:
        raise AssertionError("OnlineHMR did not use an in-window history frame")
    if kv_error > 1e-5:
        raise AssertionError(f"OnlineHMR batched/KV paths differ: {kv_error}")
    return {
        "self_projection_gradient_l1": self_gradient,
        "cross_projection_gradient_l1": cross_gradient,
        "future_leakage_max_error": future_leakage,
        "history_effect_max_abs": history_effect,
        "kv_cache_equivalence_max_abs": kv_error,
        "feature_delta_rms": float(diagnostics["feature_delta_rms"].mean()),
    }
