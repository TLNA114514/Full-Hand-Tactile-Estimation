"""Minimal frozen-DINOv3 tactile runtime for compact FullGrid checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


SUPPORTED_RESOLUTIONS = ((256, 192), (320, 240), (384, 288))


def parse_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        pieces = value.lower().split("x")
        if len(pieces) != 2:
            raise ValueError(f"Invalid input_resolution: {value!r}")
        resolution = tuple(int(piece) for piece in pieces)
    elif isinstance(value, Sequence) and len(value) == 2:
        resolution = tuple(int(piece) for piece in value)
    else:
        raise ValueError(f"Invalid input_resolution: {value!r}")
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported input_resolution={resolution}; expected one of {SUPPORTED_RESOLUTIONS}"
        )
    return resolution


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_probability: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout_probability))
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.drop(self.act(self.norm1(self.fc1(features))))
        features = self.norm2(self.fc2(features))
        return self.act(features + residual)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.permute(0, 2, 3, 1)
        features = self.norm(features)
        return features.permute(0, 3, 1, 2).contiguous()


class FullGridSpatialPooling(nn.Module):
    def __init__(
        self,
        input_channels: int,
        grid_size: Sequence[int],
        output_channels: int,
    ):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        self.output_channels = int(output_channels)
        self.output_dim = self.output_channels * self.grid_size[0] * self.grid_size[1]
        self.projection = nn.Sequential(
            nn.Conv2d(int(input_channels), self.output_channels, kernel_size=1),
            ChannelLayerNorm(self.output_channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if tuple(features.shape[-2:]) != self.grid_size:
            raise ValueError(
                f"FullGrid expects grid={self.grid_size}, got {tuple(features.shape[-2:])}"
            )
        return self.projection(features).flatten(1)


def build_decoder(
    tactile_dim: int,
    grid_size: Sequence[int],
    pool_output_channels: int,
    dropout_scale: float,
) -> nn.Sequential:
    pool = FullGridSpatialPooling(256, grid_size, pool_output_channels)
    return nn.Sequential(
        pool,
        nn.Dropout(p=0.5 * float(dropout_scale)),
        nn.Linear(pool.output_dim, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(p=0.3 * float(dropout_scale)),
        ResidualBlock(512, dropout_probability=0.3 * float(dropout_scale)),
        nn.Linear(512, int(tactile_dim)),
    )


class DenseV2DinoReZeroHead(nn.Module):
    def __init__(
        self,
        tactile_dim: int,
        layer_indices: Sequence[int],
        grid_size: Sequence[int],
        pool_output_channels: int = 32,
        residual_max_scale: float = 0.1,
        residual_rms_budget: float = 0.5,
        dropout_scale: float = 1.0,
    ):
        super().__init__()
        self.layer_indices = tuple(int(layer) for layer in layer_indices)
        if len(self.layer_indices) < 2:
            raise ValueError("ReZero inference requires at least two feature layers")
        self.refinement_layer_indices = tuple(reversed(self.layer_indices[:-1]))
        self.residual_max_scale = float(residual_max_scale)
        self.residual_rms_budget = float(residual_rms_budget)
        self.base_projection = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.GELU(),
        )
        self.decoder = build_decoder(
            tactile_dim,
            grid_size,
            pool_output_channels,
            dropout_scale,
        )
        self.projections = nn.ModuleDict()
        self.refiners = nn.ModuleDict()
        for layer in self.refinement_layer_indices:
            key = str(layer)
            self.projections[key] = nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=1),
                ChannelLayerNorm(256),
                nn.GELU(),
            )
            self.refiners[key] = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                ChannelLayerNorm(256),
                nn.GELU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
            )
        self.level_logits = nn.Parameter(torch.zeros(len(self.refinement_layer_indices)))
        self.global_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _sample_rms(features: torch.Tensor) -> torch.Tensor:
        return (
            features.float()
            .pow(2)
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(1e-24)
            .sqrt()
        )

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_levels) != len(self.layer_indices):
            raise ValueError(
                f"Expected {len(self.layer_indices)} DINO levels, got {len(feature_levels)}"
            )
        by_layer = dict(zip(self.layer_indices, feature_levels))
        base = self.base_projection(by_layer[self.layer_indices[-1]])
        unit_residuals = []
        for layer in self.refinement_layer_indices:
            key = str(layer)
            projected = self.projections[key](by_layer[layer])
            if projected.shape[-2:] != base.shape[-2:]:
                projected = nn.functional.interpolate(
                    projected,
                    size=base.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            residual = self.refiners[key](torch.cat([base, projected], dim=1))
            unit_residuals.append(torch.tanh(residual))
        weights = torch.softmax(self.level_logits, dim=0).to(dtype=base.dtype)
        raw_delta = sum(
            weight * residual for weight, residual in zip(weights, unit_residuals)
        )
        gate = self.residual_max_scale * torch.tanh(self.global_gate)
        delta = gate.to(dtype=base.dtype) * raw_delta
        base_rms = self._sample_rms(base).detach()
        delta_rms = self._sample_rms(delta)
        scale = torch.clamp(
            self.residual_rms_budget * base_rms / delta_rms.clamp_min(1e-12),
            max=1.0,
        )
        return self.decoder(base + delta * scale.to(dtype=delta.dtype))


class DinoV3Backbone(nn.Module):
    MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"

    def __init__(self, weights_path: Path, image_size: Sequence[int]):
        super().__init__()
        import timm
        from timm.models._helpers import load_state_dict
        from timm.models.eva import checkpoint_filter_fn

        self.weights_path = weights_path.resolve()
        try:
            self.model = timm.create_model(
                self.MODEL_NAME,
                pretrained=False,
                img_size=tuple(image_size),
                dynamic_img_size=False,
                num_classes=0,
            )
            state = load_state_dict(str(self.weights_path), use_ema=False, device="cpu")
            state = checkpoint_filter_fn(state, self.model)
            self.model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to strictly load DINOv3 H+/16 from {self.weights_path}"
            ) from exc
        if int(self.model.num_features) != 1280:
            raise RuntimeError(f"Expected 1280 DINO channels, got {self.model.num_features}")
        if tuple(int(v) for v in self.model.patch_embed.patch_size) != (16, 16):
            raise RuntimeError("The tactile checkpoint requires a patch-16 DINO backbone")
        if int(self.model.num_prefix_tokens) != 5:
            raise RuntimeError(
                f"Expected 5 DINO prefix tokens, got {self.model.num_prefix_tokens}"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self, image: torch.Tensor, layer_indices: Sequence[int]
    ) -> Sequence[torch.Tensor]:
        feature_maps = self.model.forward_intermediates(
            image,
            indices=[int(layer) - 1 for layer in layer_indices],
            return_prefix_tokens=False,
            norm=True,
            stop_early=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        expected_grid = (
            int(image.shape[-2]) // 16,
            int(image.shape[-1]) // 16,
        )
        for feature_map in feature_maps:
            expected = (int(image.shape[0]), 1280, *expected_grid)
            if tuple(feature_map.shape) != expected:
                raise RuntimeError(
                    f"DINO returned {tuple(feature_map.shape)}, expected {expected}"
                )
        return feature_maps


class TactileRuntime(nn.Module):
    def __init__(
        self,
        dino_weights: Path,
        tactile_dim: int,
        input_resolution: Sequence[int],
        layer_indices: Sequence[int],
        pool_output_channels: int,
        residual_max_scale: float,
        residual_rms_budget: float,
        dropout_scale: float,
    ):
        super().__init__()
        self.input_resolution = parse_resolution(input_resolution)
        self.layer_indices = tuple(int(layer) for layer in layer_indices)
        if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
            raise ValueError("DINO feature layers must be unique and increasing")
        self.tactile_dim = int(tactile_dim)
        self.backbone = DinoV3Backbone(dino_weights, self.input_resolution)
        if self.layer_indices[-1] != len(self.backbone.model.blocks):
            raise ValueError(
                "backbone_feature_layers must end at the final DINO block"
            )
        self.tactile_head = DenseV2DinoReZeroHead(
            tactile_dim=self.tactile_dim,
            layer_indices=self.layer_indices,
            grid_size=(self.input_resolution[0] // 16, self.input_resolution[1] // 16),
            pool_output_channels=pool_output_channels,
            residual_max_scale=residual_max_scale,
            residual_rms_budget=residual_rms_budget,
            dropout_scale=dropout_scale,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) != self.input_resolution:
            raise ValueError(
                f"Expected input {self.input_resolution}, got {tuple(image.shape[-2:])}"
            )
        with torch.no_grad():
            feature_levels = self.backbone(image, self.layer_indices)
        return torch.sigmoid(self.tactile_head(feature_levels))


def _merged_metadata(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(checkpoint.get("model_config", {}) or {})
    for key in (
        "visual_backbone",
        "visual_backbone_model_name",
        "backbone_sha256",
        "tactile_head_type",
        "backbone_feature_layers",
        "dino_residual_max_scale",
        "dino_residual_rms_budget",
        "pool_layout",
        "input_resolution",
        "pool_grid_size",
        "pool_valid_tokens",
        "decoder_input_dim",
        "pool_output_channels",
        "decoder_dropout_scale",
        "bbox_rescale_factor",
        "bbox_source_policy",
        "dataset_filter",
    ):
        if checkpoint.get(key) not in (None, "", []):
            metadata[key] = checkpoint[key]
    return metadata


def _infer_tactile_dim(state: Mapping[str, torch.Tensor]) -> int:
    candidates = [
        tensor
        for name, tensor in state.items()
        if name.endswith(("decoder.7.weight", "decoder.8.weight")) and tensor.ndim == 2
    ]
    if len(candidates) != 1:
        raise ValueError("Could not infer tactile dimension from compact checkpoint")
    return int(candidates[0].shape[0])


def load_runtime_model(
    checkpoint_path: Path,
    dino_weights: Path,
    device: torch.device,
    *,
    verify_backbone_sha256: bool = False,
) -> tuple[TactileRuntime, dict[str, Any]]:
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=True)
    dino_weights = dino_weights.expanduser().resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "tactile_trainable_v2":
        raise ValueError("Only compact format=tactile_trainable_v2 checkpoints are supported")
    metadata = _merged_metadata(checkpoint)
    required_contract = {
        "visual_backbone": "dinov3_hplus",
        "tactile_head_type": "dense_v2_dino_rezero",
        "pool_layout": "fullgrid32",
    }
    for key, expected in required_contract.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(f"Checkpoint {key}={actual!r}, expected {expected!r}")
    resolution = parse_resolution(metadata.get("input_resolution", (256, 192)))
    if resolution != (256, 192):
        raise ValueError(
            f"This crop12 deployment expects 256x192, checkpoint uses {resolution}"
        )
    bbox_scale = float(metadata.get("bbox_rescale_factor", 1.2))
    if abs(bbox_scale - 1.2) > 1e-6:
        raise ValueError(
            f"This deployment is fixed to crop1.2, checkpoint records crop={bbox_scale}"
        )
    expected_hash = str(metadata.get("backbone_sha256", "") or "")
    if verify_backbone_sha256 and expected_hash:
        actual_hash = file_sha256(dino_weights)
        if actual_hash != expected_hash:
            raise ValueError(
                f"DINO backbone SHA256 mismatch: expected={expected_hash}, actual={actual_hash}"
            )
    state = checkpoint.get("state_dict", {})
    head_state = {
        str(name)[len("tactile_head.") :]: tensor
        for name, tensor in state.items()
        if str(name).startswith("tactile_head.")
    }
    if not head_state:
        raise ValueError("Compact checkpoint contains no tactile_head parameters")
    tactile_dim = _infer_tactile_dim(head_state)
    model = TactileRuntime(
        dino_weights=dino_weights,
        tactile_dim=tactile_dim,
        input_resolution=resolution,
        layer_indices=metadata.get("backbone_feature_layers", (8, 16, 24, 32)),
        pool_output_channels=int(metadata.get("pool_output_channels", 32)),
        residual_max_scale=float(metadata.get("dino_residual_max_scale", 0.1)),
        residual_rms_budget=float(metadata.get("dino_residual_rms_budget", 0.5)),
        dropout_scale=float(metadata.get("decoder_dropout_scale", 1.0)),
    )
    legacy_prefix = "decoder.0.project."
    for key in tuple(head_state):
        if key.startswith(legacy_prefix):
            replacement = "decoder.0.projection." + key[len(legacy_prefix) :]
            if replacement in head_state:
                raise ValueError(f"Checkpoint contains both {key} and {replacement}")
            head_state[replacement] = head_state.pop(key)
    model.tactile_head.load_state_dict(head_state, strict=True)
    model.to(device).eval()
    metadata.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "dino_weights": str(dino_weights),
            "tactile_dim": tactile_dim,
            "input_resolution": list(resolution),
            "backbone_feature_layers": list(model.layer_indices),
            "pool_grid_size": [resolution[0] // 16, resolution[1] // 16],
            "pool_valid_tokens": (resolution[0] // 16) * (resolution[1] // 16),
            "pool_output_channels": int(metadata.get("pool_output_channels", 32)),
            "bbox_rescale_factor": bbox_scale,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "checkpoint_monitor": checkpoint.get("monitor"),
            "checkpoint_score": checkpoint.get("score"),
        }
    )
    return model, metadata
