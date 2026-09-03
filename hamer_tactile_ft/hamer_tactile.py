import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .losses import (
        TactileLossConfig,
        compute_center_auxiliary_loss,
        compute_tactile_loss,
    )
else:
    from losses import (
        TactileLossConfig,
        compute_center_auxiliary_loss,
        compute_tactile_loss,
    )


def count_obj_vertices(obj_path: Path) -> int:
    with obj_path.open("r") as handle:
        return sum(1 for line in handle if line.startswith("v "))


def default_tactile_dim() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    subdiv_obj = (
        repo_root
        / "opentouch"
        / "preprocess"
        / "scratch"
        / "mano_right_neutral_subdiv.obj"
    )
    return count_obj_vertices(subdiv_obj)


SUPPORTED_INPUT_RESOLUTIONS = ((256, 192), (320, 240), (384, 288))
SUPPORTED_TACTILE_HEAD_TYPES = (
    "dense_v2",
    "dense_v2_dino_rezero",
    "dense_v2_dino_center_aux",
    "dense_v2_dino_local_residual",
    "dense_v2_dino_support_selector",
    "dense_v2_dino_surface_basis",
)
SUPPORTED_POOL_LAYOUTS = ("legacy5", "fullgrid32")
SUPPORTED_MODEL_INITIALIZATION_ORDERS = (
    "projection_first",
    "legacy_decoder_first",
)
# Project-wide canonical RNG assignment for every newly constructed tactile
# base. This is the order used by the July 2026 crop1.2 reference model.
CANONICAL_MODEL_INITIALIZATION_ORDER = "legacy_decoder_first"
SUPPORTED_SURFACE_COEFFICIENT_ARCHITECTURES = ("linear", "nonlinear")


def _canonical_mesh_assets() -> Tuple[torch.Tensor, torch.Tensor]:
    """Load fixed canonical vertices and the audited valid-palm mask."""

    repo_root = Path(__file__).resolve().parents[1]
    mesh_path = (
        repo_root
        / "opentouch"
        / "preprocess"
        / "scratch"
        / "mano_right_neutral_subdiv.obj"
    )
    palm_faces_path = (
        repo_root
        / "opentouch"
        / "preprocess"
        / "scratch"
        / "auto_calibrated_palm_subdiv_faces.json"
    )
    vertices = []
    with mesh_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    with palm_faces_path.open("r", encoding="utf-8") as handle:
        palm_faces = json.load(handle)["group_negative"]["face_triplets"]
    valid = torch.zeros(len(vertices), dtype=torch.bool)
    for face in palm_faces:
        valid[torch.as_tensor(face, dtype=torch.long)] = True
    return torch.tensor(vertices, dtype=torch.float32), valid


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _sparse_basis_sha256(
    support_indices: torch.Tensor, support_weights: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("indices", support_indices.detach().cpu().long().contiguous()),
        ("weights", support_weights.detach().cpu().float().contiguous()),
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _load_surface_basis_runtime(
    path: str,
    *,
    tactile_dim: int,
    coefficient_dim: int,
    target_support_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    artifact_path = Path(path).expanduser().resolve(strict=True)
    artifact = torch.load(artifact_path, map_location="cpu")
    if not isinstance(artifact, dict) or artifact.get("format") != (
        "canonical_surface_basis_v1"
    ):
        raise ValueError(
            f"Unsupported canonical surface basis artifact: {artifact_path}"
        )
    metadata = dict(artifact.get("metadata", {}))
    basis = artifact.get("basis_valid")
    support_indices = artifact.get("support_indices")
    support_weights = artifact.get("support_weights")
    valid_indices = artifact.get("valid_vertex_indices")
    if not isinstance(valid_indices, torch.Tensor):
        raise TypeError("Surface basis artifact lacks tensor payloads")
    valid_indices = valid_indices.detach().long().contiguous()
    expected_rows = int(valid_indices.numel())
    if isinstance(basis, torch.Tensor):
        basis = basis.detach().float().contiguous()
        expected_shape = (expected_rows, int(coefficient_dim))
        if tuple(basis.shape) != expected_shape:
            raise ValueError(
                f"Surface basis shape mismatch: artifact={tuple(basis.shape)}, "
                f"expected={expected_shape}"
            )
        support_counts = (basis > 0.0).sum(dim=1)
        maximum_support = int(support_counts.max().item())
        support_indices = torch.zeros(
            (expected_rows, maximum_support), dtype=torch.long
        )
        support_weights = torch.zeros(
            (expected_rows, maximum_support), dtype=torch.float32
        )
        for row_index in range(expected_rows):
            columns = torch.nonzero(
                basis[row_index] > 0.0, as_tuple=False
            ).flatten()
            count = int(columns.numel())
            support_indices[row_index, :count] = columns
            support_weights[row_index, :count] = basis[
                row_index, columns
            ]
    if not isinstance(support_indices, torch.Tensor) or not isinstance(
        support_weights, torch.Tensor
    ):
        raise TypeError("Surface basis artifact lacks sparse support tensors")
    support_indices = support_indices.detach().long().contiguous()
    support_weights = support_weights.detach().float().contiguous()
    if (
        support_indices.ndim != 2
        or support_weights.shape != support_indices.shape
        or support_indices.shape[0] != expected_rows
    ):
        raise ValueError(
            "Surface basis sparse support tensors have invalid shapes: "
            f"indices={tuple(support_indices.shape)}, "
            f"weights={tuple(support_weights.shape)}"
        )
    if int(metadata.get("tactile_dim", -1)) != int(tactile_dim):
        raise ValueError("Surface basis tactile dimension does not match the model")
    if int(metadata.get("coefficient_dim", -1)) != int(coefficient_dim):
        raise ValueError("Surface basis coefficient dimension does not match")
    if int(metadata.get("target_support_count", -1)) != int(
        target_support_count
    ):
        raise ValueError("Surface basis target support does not match")
    if valid_indices.ndim != 1 or bool(
        ((valid_indices < 0) | (valid_indices >= int(tactile_dim))).any().item()
    ):
        raise ValueError("Surface basis contains invalid canonical vertex indices")
    if int(torch.unique(valid_indices).numel()) != int(valid_indices.numel()):
        raise ValueError("Surface basis valid vertex indices are not unique")
    if not bool(torch.isfinite(support_weights).all().item()) or bool(
        (support_weights < 0).any().item()
    ):
        raise ValueError("Surface basis must be finite and nonnegative")
    active_support = support_weights > 0.0
    if bool(
        (
            (support_indices < 0)
            | (support_indices >= int(coefficient_dim))
        )[active_support].any().item()
    ):
        raise ValueError("Surface basis contains invalid coefficient indices")
    partition_error = float(
        (support_weights.sum(dim=1) - 1.0).abs().max().item()
    )
    if partition_error > 5e-6:
        raise ValueError(
            f"Surface basis violates partition of unity: error={partition_error}"
        )
    expected_tensor_sha = str(metadata.get("basis_sha256", "") or "")
    actual_tensor_sha = (
        _tensor_sha256(basis)
        if isinstance(basis, torch.Tensor)
        else expected_tensor_sha
    )
    if (
        expected_tensor_sha
        and actual_tensor_sha
        and expected_tensor_sha != actual_tensor_sha
    ):
        raise ValueError(
            "Surface basis tensor SHA256 mismatch: "
            f"expected={expected_tensor_sha}, actual={actual_tensor_sha}"
        )
    expected_sparse_sha = str(
        metadata.get("sparse_basis_sha256", "") or ""
    )
    actual_sparse_sha = _sparse_basis_sha256(
        support_indices, support_weights
    )
    if expected_sparse_sha and expected_sparse_sha != actual_sparse_sha:
        raise ValueError(
            "Sparse surface basis SHA256 mismatch: "
            f"expected={expected_sparse_sha}, actual={actual_sparse_sha}"
        )
    metadata.update(
        {
            "artifact": str(artifact_path),
            "artifact_sha256": _file_sha256(artifact_path),
            "basis_sha256": actual_tensor_sha,
            "sparse_basis_sha256": actual_sparse_sha,
            "partition_unity_max_error_runtime": partition_error,
        }
    )
    return support_indices, support_weights, valid_indices, metadata


def _canonical_rbf_assignment(
    *,
    tactile_dim: int,
    anchor_count: int,
    neighbor_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build deterministic local anchor interpolation on the canonical mesh."""

    vertices, valid_mask = _canonical_mesh_assets()
    if int(vertices.shape[0]) != int(tactile_dim):
        raise RuntimeError(
            f"Canonical mesh has {vertices.shape[0]} vertices, expected {tactile_dim}"
        )
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if not 1 <= int(anchor_count) <= int(valid_indices.numel()):
        raise ValueError(
            f"local_anchor_count must lie in [1, {valid_indices.numel()}]"
        )
    if not 1 <= int(neighbor_count) <= int(anchor_count):
        raise ValueError(
            "local_anchor_neighbors must lie in [1, local_anchor_count]"
        )

    coordinates = vertices[valid_indices]
    centroid = coordinates.mean(dim=0, keepdim=True)
    first = int((coordinates - centroid).square().sum(dim=1).argmax().item())
    selected = [first]
    minimum_distance = (coordinates - coordinates[first]).square().sum(dim=1)
    for _ in range(1, int(anchor_count)):
        next_index = int(minimum_distance.argmax().item())
        selected.append(next_index)
        distance = (coordinates - coordinates[next_index]).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)

    anchor_vertex_indices = valid_indices[torch.as_tensor(selected, dtype=torch.long)]
    anchor_coordinates = vertices[anchor_vertex_indices]
    distances = torch.cdist(coordinates, anchor_coordinates)
    nearest_distances, nearest_anchors = torch.topk(
        distances,
        k=int(neighbor_count),
        dim=1,
        largest=False,
        sorted=True,
    )
    nearest_scale = nearest_distances[:, 0]
    positive_scale = nearest_scale[nearest_scale > 1e-8]
    sigma = (
        positive_scale.median() * 2.0
        if positive_scale.numel()
        else distances.new_tensor(1.0)
    ).clamp_min(1e-6)
    weights = torch.exp(-0.5 * (nearest_distances / sigma).square())
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)

    vertex_anchor_indices = torch.zeros(
        (int(tactile_dim), int(neighbor_count)), dtype=torch.long
    )
    vertex_anchor_weights = torch.zeros(
        (int(tactile_dim), int(neighbor_count)), dtype=torch.float32
    )
    vertex_anchor_indices[valid_indices] = nearest_anchors
    vertex_anchor_weights[valid_indices] = weights
    return (
        anchor_vertex_indices,
        vertex_anchor_indices,
        vertex_anchor_weights,
        valid_mask,
    )


def parse_input_resolution(value) -> Tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError(
                "input_resolution must use HEIGHTxWIDTH, for example 320x240"
            )
        height, width = (int(part.strip()) for part in parts)
    elif isinstance(value, Sequence) and len(value) == 2:
        height, width = (int(part) for part in value)
    else:
        raise ValueError(
            "input_resolution must be HEIGHTxWIDTH or a two-value sequence"
        )
    resolution = (height, width)
    if resolution not in SUPPORTED_INPUT_RESOLUTIONS:
        allowed = ", ".join(
            f"{height}x{width}" for height, width in SUPPORTED_INPUT_RESOLUTIONS
        )
        raise ValueError(
            f"Unsupported input_resolution {height}x{width}; choose one of {allowed}"
        )
    if height % 16 or width % 16:
        raise ValueError(
            "input_resolution height and width must both be divisible by 16"
        )
    return resolution


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


class AnatomicalSpatialPooling(nn.Module):
    """Original V2 5x5 pooling with the four bottom-corner cells removed."""

    def __init__(self, input_channels: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        mask = torch.ones((5, 5), dtype=torch.bool)
        mask[4, 0] = False
        mask[4, 1] = False
        mask[4, 3] = False
        mask[4, 4] = False
        self.register_buffer("mask", mask)
        self.valid_token_count = int(mask.sum().item())
        self.output_dim = int(input_channels) * self.valid_token_count

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.pool(features)
        batch_size, channels, height, width = features.shape
        features = features.reshape(batch_size, channels, height * width)
        return features[:, :, self.mask.reshape(-1)].reshape(batch_size, -1)


class FullGrid32SpatialPooling(nn.Module):
    """Preserve the complete DINO grid while controlling decoder size."""

    def __init__(
        self,
        input_channels: int = 256,
        grid_size: Sequence[int] = (16, 12),
        output_channels: int = 32,
    ):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        self.output_channels = int(output_channels)
        self.valid_token_count = self.grid_size[0] * self.grid_size[1]
        self.output_dim = self.output_channels * self.valid_token_count
        self.projection = nn.Sequential(
            nn.Conv2d(int(input_channels), self.output_channels, kernel_size=1),
            ChannelLayerNorm(self.output_channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if tuple(features.shape[-2:]) != self.grid_size:
            raise ValueError(
                f"FullGrid32 expects a {self.grid_size} feature map, "
                f"got {tuple(features.shape[-2:])}"
            )
        return self.projection(features).flatten(1)


def _build_dense_decoder(
    *,
    tactile_dim: int,
    channels: int,
    pool_layout: str,
    grid_size: Sequence[int],
    pool_output_channels: int,
    decoder_hidden_dim: int,
    dropout_scale: float,
) -> Tuple[nn.Sequential, int, int]:
    decoder_hidden_dim = int(decoder_hidden_dim)
    if decoder_hidden_dim < 1:
        raise ValueError("decoder_hidden_dim must be positive")
    if pool_layout == "legacy5":
        pool = AnatomicalSpatialPooling(input_channels=channels)
    elif pool_layout == "fullgrid32":
        pool = FullGrid32SpatialPooling(
            input_channels=channels,
            grid_size=grid_size,
            output_channels=pool_output_channels,
        )
    else:
        raise ValueError(
            f"Unsupported pool_layout={pool_layout}; choose one of {SUPPORTED_POOL_LAYOUTS}"
        )
    decoder = nn.Sequential(
        pool,
        nn.Dropout(p=0.5 * dropout_scale),
        nn.Linear(pool.output_dim, decoder_hidden_dim),
        nn.LayerNorm(decoder_hidden_dim),
        nn.GELU(),
        nn.Dropout(p=0.3 * dropout_scale),
        ResidualBlock(
            decoder_hidden_dim,
            dropout_probability=0.3 * dropout_scale,
        ),
        nn.Linear(decoder_hidden_dim, int(tactile_dim)),
    )
    return decoder, int(pool.output_dim), int(pool.valid_token_count)


class DenseV2TactileHead(nn.Module):
    """Final-DINO-feature control using the canonical V2 dense decoder."""

    def __init__(
        self,
        tactile_dim: int,
        input_channels: int = 1280,
        channels: int = 256,
        pool_layout: str = "legacy5",
        decoder_dropout_scale: float = 1.0,
        grid_size: Sequence[int] = (16, 12),
        pool_output_channels: int = 32,
        decoder_hidden_dim: int = 512,
        model_initialization_order: str = CANONICAL_MODEL_INITIALIZATION_ORDER,
    ):
        super().__init__()
        dropout_scale = float(decoder_dropout_scale)
        if not 0.0 <= dropout_scale <= 1.0:
            raise ValueError("decoder_dropout_scale must lie in [0, 1]")
        self.pool_layout = str(pool_layout)
        self.grid_size = tuple(int(value) for value in grid_size)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.model_initialization_order = str(model_initialization_order)
        if self.model_initialization_order not in SUPPORTED_MODEL_INITIALIZATION_ORDERS:
            raise ValueError(
                "model_initialization_order must be one of "
                f"{SUPPORTED_MODEL_INITIALIZATION_ORDERS}"
            )

        def build_projection():
            return nn.Sequential(
                nn.Conv2d(int(input_channels), int(channels), kernel_size=1),
                nn.GELU(),
            )

        def build_decoder():
            return _build_dense_decoder(
                tactile_dim=tactile_dim,
                channels=channels,
                pool_layout=self.pool_layout,
                grid_size=self.grid_size,
                pool_output_channels=pool_output_channels,
                decoder_hidden_dim=self.decoder_hidden_dim,
                dropout_scale=dropout_scale,
            )

        # The July 2026 crop1.2 baseline consumed RNG for the decoder before
        # base_projection. Keep both assignments available for old experiment
        # replay without changing any state-dict key.
        if self.model_initialization_order == "legacy_decoder_first":
            self.decoder, self.decoder_input_dim, self.pool_valid_tokens = (
                build_decoder()
            )
            self.base_projection = build_projection()
        else:
            self.base_projection = build_projection()
            self.decoder, self.decoder_input_dim, self.pool_valid_tokens = (
                build_decoder()
            )
        self.refinement_layer_indices = ()

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, features) -> torch.Tensor:
        if isinstance(features, (tuple, list)):
            if not features:
                raise ValueError("dense_v2 received no DINO feature maps")
            features = features[-1]
        return self.decoder(self.base_projection(features))


class DenseV2DinoReZeroTactileHead(DenseV2TactileHead):
    """Fuse multilevel DINO maps through one bounded ReZero residual."""

    def __init__(
        self,
        tactile_dim: int,
        layer_indices: Sequence[int],
        residual_max_scale: float = 0.10,
        residual_rms_budget: float = 0.50,
        input_channels: int = 1280,
        channels: int = 256,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
        grid_size: Sequence[int] = (16, 12),
        pool_output_channels: int = 32,
        decoder_hidden_dim: int = 512,
        model_initialization_order: str = CANONICAL_MODEL_INITIALIZATION_ORDER,
    ):
        layer_indices = tuple(int(layer) for layer in layer_indices)
        if len(layer_indices) < 2 or tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError(
                "DINO ReZero layers must contain at least two unique increasing indices"
            )
        if not 0.0 < float(residual_max_scale) <= 1.0:
            raise ValueError("dino_residual_max_scale must lie in (0, 1]")
        if not 0.0 < float(residual_rms_budget) <= 1.0:
            raise ValueError("dino_residual_rms_budget must lie in (0, 1]")
        super().__init__(
            tactile_dim=tactile_dim,
            input_channels=input_channels,
            channels=channels,
            pool_layout=pool_layout,
            decoder_dropout_scale=decoder_dropout_scale,
            grid_size=grid_size,
            pool_output_channels=pool_output_channels,
            decoder_hidden_dim=decoder_hidden_dim,
            model_initialization_order=model_initialization_order,
        )
        self.layer_indices = layer_indices
        self.refinement_layer_indices = tuple(reversed(layer_indices[:-1]))
        self.residual_max_scale = float(residual_max_scale)
        self.residual_rms_budget = float(residual_rms_budget)

        self.projections = nn.ModuleDict()
        self.refiners = nn.ModuleDict()
        for layer in self.refinement_layer_indices:
            key = str(layer)
            self.projections[key] = nn.Sequential(
                nn.Conv2d(int(input_channels), channels, kernel_size=1),
                ChannelLayerNorm(channels),
                nn.GELU(),
            )
            refiner = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
                ChannelLayerNorm(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
            nn.init.normal_(refiner[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(refiner[-1].bias)
            self.refiners[key] = refiner

        self.level_logits = nn.Parameter(torch.zeros(len(self.refinement_layer_indices)))
        self.global_gate = nn.Parameter(torch.zeros(()))
        self._last_feature_diagnostics: Dict[str, torch.Tensor] = {}

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_feature_diagnostics

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.level_logits, dim=0)

    def effective_gate(self) -> torch.Tensor:
        return self.residual_max_scale * torch.tanh(self.global_gate)

    @staticmethod
    def _sample_rms(features: torch.Tensor) -> torch.Tensor:
        return (
            features.float()
            .pow(2)
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(1e-24)
            .sqrt()
        )

    def _fuse(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_levels) != len(self.layer_indices):
            raise ValueError(
                f"Expected {len(self.layer_indices)} DINO levels, got {len(feature_levels)}"
            )
        features_by_layer = dict(zip(self.layer_indices, feature_levels))
        base = self.base_projection(features_by_layer[self.layer_indices[-1]])
        projected_values = []
        residual_logits_values = []
        unit_residual_values = []
        for layer in self.refinement_layer_indices:
            key = str(layer)
            projected = self.projections[key](features_by_layer[layer])
            if projected.shape[-2:] != base.shape[-2:]:
                projected = nn.functional.interpolate(
                    projected,
                    size=base.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            residual_logits = self.refiners[key](
                torch.cat([base, projected], dim=1)
            )
            projected_values.append(projected)
            residual_logits_values.append(residual_logits)
            unit_residual_values.append(torch.tanh(residual_logits))

        weights = self.fusion_weights().to(dtype=base.dtype)
        raw_delta = sum(
            weight * residual
            for weight, residual in zip(weights, unit_residual_values)
        )
        delta_pre_budget = self.effective_gate().to(dtype=base.dtype) * raw_delta
        base_rms_per_sample = self._sample_rms(base).detach()
        delta_rms_pre_per_sample = self._sample_rms(delta_pre_budget)
        allowed_rms = self.residual_rms_budget * base_rms_per_sample
        budget_scale = torch.clamp(
            allowed_rms / delta_rms_pre_per_sample.clamp_min(1e-12),
            max=1.0,
        )
        delta = delta_pre_budget * budget_scale.to(dtype=delta_pre_budget.dtype)
        fused = base + delta

        if not self.training:
            unit_rms = torch.stack(
                [
                    value.detach().float().pow(2).mean().sqrt()
                    for value in unit_residual_values
                ]
            )
            weighted_rms = weights.detach().float() * unit_rms
            delta_rms_post_per_sample = self._sample_rms(delta)
            self._last_feature_diagnostics = {
                "gate_raw": self.global_gate.detach().float(),
                "gate_effective": self.effective_gate().detach().float(),
                "level_weight": weights.detach().float(),
                "projected_rms": torch.stack(
                    [
                        value.detach().float().pow(2).mean().sqrt()
                        for value in projected_values
                    ]
                ),
                "raw_residual_rms": unit_rms,
                "residual_saturation": torch.stack(
                    [
                        (value.detach().float().abs() > 3.0).float().mean()
                        for value in residual_logits_values
                    ]
                ),
                "effective_contribution": (
                    weighted_rms / weighted_rms.sum().clamp_min(1e-12)
                ),
                "delta_rms_pre_budget": delta_rms_pre_per_sample.detach().mean(),
                "delta_rms_post_budget": delta_rms_post_per_sample.detach().mean(),
                "delta_to_base_rms": (
                    delta_rms_post_per_sample.detach()
                    / base_rms_per_sample.clamp_min(1e-12)
                ).mean(),
                "budget_clip_rate": (
                    budget_scale.detach() < (1.0 - 1e-6)
                ).float().mean(),
                "base_rms": base.detach().float().pow(2).mean().sqrt(),
                "final_rms": fused.detach().float().pow(2).mean().sqrt(),
            }
        else:
            self._last_feature_diagnostics = {}
        return fused

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.decoder(self._fuse(feature_levels))


class DenseV2DinoCenterAuxTactileHead(DenseV2DinoReZeroTactileHead):
    """Train-only center supervision over the unchanged DenseV2 bottleneck.

    The pressure path is constructed first and remains byte-for-byte compatible
    with ``DenseV2DinoReZeroTactileHead`` under the same RNG state. Auxiliary
    logits never feed back into the pressure logits and can be discarded for
    deployment.
    """

    def __init__(
        self,
        tactile_dim: int,
        layer_indices: Sequence[int],
        *,
        center_aux_hidden_dim: int = 128,
        residual_max_scale: float = 0.10,
        residual_rms_budget: float = 0.50,
        input_channels: int = 1280,
        channels: int = 256,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
        grid_size: Sequence[int] = (16, 12),
        pool_output_channels: int = 32,
        decoder_hidden_dim: int = 512,
        model_initialization_order: str = CANONICAL_MODEL_INITIALIZATION_ORDER,
    ):
        super().__init__(
            tactile_dim=tactile_dim,
            layer_indices=layer_indices,
            residual_max_scale=residual_max_scale,
            residual_rms_budget=residual_rms_budget,
            input_channels=input_channels,
            channels=channels,
            pool_layout=pool_layout,
            decoder_dropout_scale=decoder_dropout_scale,
            grid_size=grid_size,
            pool_output_channels=pool_output_channels,
            decoder_hidden_dim=decoder_hidden_dim,
            model_initialization_order=model_initialization_order,
        )
        self.center_aux_hidden_dim = int(center_aux_hidden_dim)
        if self.center_aux_hidden_dim < 1:
            raise ValueError("center_aux_hidden_dim must be positive")
        self.center_aux_heatmap = nn.Sequential(
            nn.Linear(self.decoder_hidden_dim, self.center_aux_hidden_dim),
            nn.GELU(),
            nn.Linear(self.center_aux_hidden_dim, int(tactile_dim)),
        )
        self.center_aux_presence = nn.Linear(self.decoder_hidden_dim, 1)

    def forward_with_center_aux(
        self,
        feature_levels: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self._fuse(feature_levels)
        for layer in self.decoder[:-1]:
            value = layer(value)
        pressure_logits = self.decoder[-1](value)
        center_logits = self.center_aux_heatmap(value)
        presence_logits = self.center_aux_presence(value).squeeze(-1)
        return pressure_logits, center_logits, presence_logits

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.decoder(self._fuse(feature_levels))


class DenseV2DinoSurfaceBasisTactileHead(DenseV2DinoReZeroTactileHead):
    """Direct FullGrid coefficient decoder over a fixed canonical surface basis."""

    def __init__(
        self,
        tactile_dim: int,
        layer_indices: Sequence[int],
        *,
        surface_basis_path: str,
        surface_coefficient_dim: int,
        surface_coefficient_architecture: str = "linear",
        surface_coefficient_hidden_dim: int = 1024,
        surface_target_support_count: int = 4,
        surface_background_probability: float = 1e-3,
        freeze_surface_feature_extractor: bool = True,
        residual_max_scale: float = 0.10,
        residual_rms_budget: float = 0.50,
        input_channels: int = 1280,
        channels: int = 256,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
        grid_size: Sequence[int] = (16, 12),
        pool_output_channels: int = 32,
        decoder_hidden_dim: int = 512,
        model_initialization_order: str = CANONICAL_MODEL_INITIALIZATION_ORDER,
    ):
        if str(pool_layout) != "fullgrid32":
            raise ValueError("The surface-basis head requires pool_layout=fullgrid32")
        coefficient_dim = int(surface_coefficient_dim)
        if coefficient_dim <= 0:
            raise ValueError("surface_coefficient_dim must be positive")
        coefficient_architecture = str(
            surface_coefficient_architecture
        ).strip().lower()
        if coefficient_architecture not in SUPPORTED_SURFACE_COEFFICIENT_ARCHITECTURES:
            raise ValueError(
                "surface_coefficient_architecture must be one of "
                f"{SUPPORTED_SURFACE_COEFFICIENT_ARCHITECTURES}"
            )
        coefficient_hidden_dim = int(surface_coefficient_hidden_dim)
        if coefficient_hidden_dim <= 0:
            raise ValueError("surface_coefficient_hidden_dim must be positive")
        target_support = int(surface_target_support_count)
        if target_support != 4:
            raise ValueError("Stage 1 surface basis requires target support 4")
        background_probability = float(surface_background_probability)
        if not 0.0 < background_probability < 0.5:
            raise ValueError("surface_background_probability must lie in (0, 0.5)")
        if not str(surface_basis_path).strip():
            raise ValueError("surface_basis_path is required for surface-basis training")
        super().__init__(
            tactile_dim=tactile_dim,
            layer_indices=layer_indices,
            residual_max_scale=residual_max_scale,
            residual_rms_budget=residual_rms_budget,
            input_channels=input_channels,
            channels=channels,
            pool_layout=pool_layout,
            decoder_dropout_scale=decoder_dropout_scale,
            grid_size=grid_size,
            pool_output_channels=pool_output_channels,
            decoder_hidden_dim=decoder_hidden_dim,
            model_initialization_order=model_initialization_order,
        )
        # Keep the audited FullGrid32 projection but remove the old shared
        # 512-D dense decoder completely.
        self.fullgrid_pool = self.decoder[0]
        del self.decoder
        support_indices, support_weights, valid_indices, metadata = (
            _load_surface_basis_runtime(
                surface_basis_path,
                tactile_dim=int(tactile_dim),
                coefficient_dim=coefficient_dim,
                target_support_count=target_support,
            )
        )
        self.register_buffer(
            "surface_support_indices", support_indices, persistent=True
        )
        self.register_buffer(
            "surface_support_weights", support_weights, persistent=True
        )
        self.register_buffer(
            "surface_valid_vertex_indices", valid_indices, persistent=True
        )
        self.surface_basis_metadata = metadata
        self.surface_basis_path = str(metadata["artifact"])
        self.surface_basis_artifact_sha256 = str(metadata["artifact_sha256"])
        self.surface_basis_tensor_sha256 = str(metadata["basis_sha256"])
        self.surface_sparse_basis_sha256 = str(
            metadata["sparse_basis_sha256"]
        )
        self.surface_tactile_dim = int(tactile_dim)
        self.surface_valid_vertex_count = int(valid_indices.numel())
        self.surface_maximum_support_count = int(
            support_indices.shape[1]
        )
        self.surface_coefficient_dim = coefficient_dim
        self.surface_coefficient_architecture = coefficient_architecture
        self.surface_coefficient_hidden_dim = coefficient_hidden_dim
        self.surface_target_support_count = target_support
        self.surface_background_probability = background_probability
        self.freeze_surface_feature_extractor = bool(
            freeze_surface_feature_extractor
        )
        self.decoder_input_dim = int(self.fullgrid_pool.output_dim)
        self.pool_valid_tokens = int(self.fullgrid_pool.valid_token_count)
        dropout = 0.3 * float(decoder_dropout_scale)
        if coefficient_architecture == "linear":
            self.coefficient_head = nn.Sequential(
                nn.LayerNorm(self.decoder_input_dim),
                nn.Dropout(dropout),
                nn.Linear(self.decoder_input_dim, coefficient_dim),
            )
        else:
            self.coefficient_head = nn.Sequential(
                nn.LayerNorm(self.decoder_input_dim),
                nn.Linear(self.decoder_input_dim, coefficient_hidden_dim),
                nn.LayerNorm(coefficient_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(
                    coefficient_hidden_dim,
                    dropout_probability=dropout,
                ),
                ResidualBlock(
                    coefficient_hidden_dim,
                    dropout_probability=dropout,
                ),
                nn.Linear(coefficient_hidden_dim, coefficient_dim),
            )
        output_layer = self.coefficient_head[-1]
        nn.init.zeros_(output_layer.weight)
        background_logit = torch.logit(
            torch.tensor(background_probability, dtype=torch.float32)
        ).item()
        self.surface_background_logit = float(background_logit)
        nn.init.constant_(output_layer.bias, background_logit)
        self._last_surface_diagnostics: Dict[str, torch.Tensor] = {}
        if self.freeze_surface_feature_extractor:
            self.freeze_base_parameters()

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        diagnostics = dict(super().feature_diagnostics())
        diagnostics.update(self._last_surface_diagnostics)
        return diagnostics

    def base_checkpoint_mapping(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for key in self.state_dict():
            if key.startswith(("base_projection.", "projections.", "refiners.")):
                mapping[key] = key
            elif key in {"level_logits", "global_gate"}:
                mapping[key] = key
            elif key.startswith("fullgrid_pool."):
                mapping[key] = "decoder.0." + key[len("fullgrid_pool."):]
        return mapping

    def base_state_keys(self) -> Tuple[str, ...]:
        return tuple(self.base_checkpoint_mapping())

    def extension_state_keys(self) -> Tuple[str, ...]:
        base = set(self.base_state_keys())
        return tuple(key for key in self.state_dict() if key not in base)

    def freeze_base_parameters(self) -> None:
        for module in (
            self.base_projection,
            self.projections,
            self.refiners,
            self.fullgrid_pool,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.level_logits.requires_grad_(False)
        self.global_gate.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_surface_feature_extractor", False):
            for module in (
                self.base_projection,
                self.projections,
                self.refiners,
                self.fullgrid_pool,
            ):
                module.eval()
            self.coefficient_head.train(mode)
        return self

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        grid = self._fuse(feature_levels)
        fullgrid_features = self.fullgrid_pool(grid)
        coefficients = self.coefficient_head(fullgrid_features)
        supported_coefficients = coefficients[:, self.surface_support_indices]
        valid_logits = (
            supported_coefficients
            * self.surface_support_weights.to(dtype=coefficients.dtype)
        ).sum(dim=-1)
        # Autocast may accumulate the sparse weighted sum in FP32 even when
        # the coefficient projection is BF16. Keep the complete output in the
        # accumulator dtype so index_copy_ remains valid and does not discard
        # the more stable surface reconstruction.
        logits = valid_logits.new_full(
            (coefficients.shape[0], self.surface_tactile_dim),
            self.surface_background_logit,
        )
        logits.index_copy_(1, self.surface_valid_vertex_indices, valid_logits)
        if not self.training:
            self._last_surface_diagnostics = {
                "surface_coefficient_mean": coefficients.detach().float().mean(),
                "surface_coefficient_rms": coefficients.detach().float().pow(2).mean().sqrt(),
                "surface_coefficient_negative_fraction": (
                    coefficients.detach().float() < 0.0
                ).float().mean(),
                "surface_valid_logit_rms": valid_logits.detach().float().pow(2).mean().sqrt(),
            }
        else:
            self._last_surface_diagnostics = {}
        return logits


class CanonicalLocalResidualCarrier(nn.Module):
    """Map a complete RGB grid to bounded, locally supported mesh corrections."""

    def __init__(
        self,
        *,
        input_dim: int,
        tactile_dim: int,
        anchor_count: int = 512,
        anchor_neighbors: int = 4,
        maximum_logit_delta: float = 6.0,
        dropout: float = 0.10,
    ):
        super().__init__()
        if float(maximum_logit_delta) <= 0.0:
            raise ValueError("local_logit_delta_max must be positive")
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("local_residual_dropout must lie in [0, 1]")
        self.input_dim = int(input_dim)
        self.tactile_dim = int(tactile_dim)
        self.anchor_count = int(anchor_count)
        self.anchor_neighbors = int(anchor_neighbors)
        self.maximum_logit_delta = float(maximum_logit_delta)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.up_head = nn.Linear(self.input_dim, self.anchor_count)
        self.down_head = nn.Linear(self.input_dim, self.anchor_count)
        nn.init.zeros_(self.up_head.weight)
        nn.init.zeros_(self.up_head.bias)
        nn.init.zeros_(self.down_head.weight)
        nn.init.zeros_(self.down_head.bias)

        (
            anchor_vertex_indices,
            vertex_anchor_indices,
            vertex_anchor_weights,
            valid_palm_mask,
        ) = _canonical_rbf_assignment(
            tactile_dim=self.tactile_dim,
            anchor_count=self.anchor_count,
            neighbor_count=self.anchor_neighbors,
        )
        self.register_buffer("anchor_vertex_indices", anchor_vertex_indices)
        self.register_buffer("vertex_anchor_indices", vertex_anchor_indices)
        self.register_buffer("vertex_anchor_weights", vertex_anchor_weights)
        self.register_buffer("valid_palm_mask", valid_palm_mask)
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_diagnostics

    def forward(self, fullgrid_features: torch.Tensor) -> torch.Tensor:
        if fullgrid_features.ndim != 2 or fullgrid_features.shape[1] != self.input_dim:
            raise ValueError(
                f"Local carrier expects [B,{self.input_dim}], got "
                f"{tuple(fullgrid_features.shape)}"
            )
        normalized = self.dropout(self.input_norm(fullgrid_features))
        up_strength = torch.sigmoid(self.up_head(normalized))
        down_strength = torch.sigmoid(self.down_head(normalized))
        anchor_delta = self.maximum_logit_delta * (up_strength - down_strength)
        per_vertex = anchor_delta[:, self.vertex_anchor_indices]
        delta = (
            per_vertex
            * self.vertex_anchor_weights.to(dtype=per_vertex.dtype).unsqueeze(0)
        ).sum(dim=-1)
        delta = delta * self.valid_palm_mask.to(dtype=delta.dtype).unsqueeze(0)

        if not self.training:
            valid_delta = delta[:, self.valid_palm_mask]
            total_path = up_strength + down_strength
            net_path = (up_strength - down_strength).abs()
            self._last_diagnostics = {
                "local_logit_delta_rms": valid_delta.detach().float().pow(2).mean().sqrt(),
                "local_logit_delta_abs_max": valid_delta.detach().float().abs().max(),
                "local_logit_delta_saturation": (
                    valid_delta.detach().float().abs()
                    >= 0.95 * self.maximum_logit_delta
                ).float().mean(),
                "local_changed_vertex_fraction": (
                    valid_delta.detach().float().abs() > 0.05
                ).float().mean(),
                "local_anchor_active_fraction": (
                    anchor_delta.detach().float().abs() > 0.05
                ).float().mean(),
                "local_up_strength_mean": up_strength.detach().float().mean(),
                "local_down_strength_mean": down_strength.detach().float().mean(),
                "local_path_cancellation_ratio": (
                    1.0
                    - net_path.detach().float().sum()
                    / total_path.detach().float().sum().clamp_min(1e-12)
                ),
            }
        else:
            self._last_diagnostics = {}
        return delta


class DenseV2DinoLocalResidualTactileHead(DenseV2DinoReZeroTactileHead):
    """Frozen FullGrid baseline plus a canonical local logit correction path."""

    def __init__(
        self,
        tactile_dim: int,
        layer_indices: Sequence[int],
        residual_max_scale: float = 0.10,
        residual_rms_budget: float = 0.50,
        input_channels: int = 1280,
        channels: int = 256,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
        grid_size: Sequence[int] = (16, 12),
        pool_output_channels: int = 32,
        decoder_hidden_dim: int = 512,
        model_initialization_order: str = CANONICAL_MODEL_INITIALIZATION_ORDER,
        local_anchor_count: int = 512,
        local_anchor_neighbors: int = 4,
        local_logit_delta_max: float = 6.0,
        local_residual_dropout: float = 0.10,
        freeze_base: bool = True,
    ):
        if str(pool_layout) != "fullgrid32":
            raise ValueError("The local residual head requires pool_layout=fullgrid32")
        super().__init__(
            tactile_dim=tactile_dim,
            layer_indices=layer_indices,
            residual_max_scale=residual_max_scale,
            residual_rms_budget=residual_rms_budget,
            input_channels=input_channels,
            channels=channels,
            pool_layout=pool_layout,
            decoder_dropout_scale=decoder_dropout_scale,
            grid_size=grid_size,
            pool_output_channels=pool_output_channels,
            decoder_hidden_dim=decoder_hidden_dim,
            model_initialization_order=model_initialization_order,
        )
        self.local_residual = CanonicalLocalResidualCarrier(
            input_dim=self.decoder_input_dim,
            tactile_dim=tactile_dim,
            anchor_count=local_anchor_count,
            anchor_neighbors=local_anchor_neighbors,
            maximum_logit_delta=local_logit_delta_max,
            dropout=local_residual_dropout,
        )
        self.local_anchor_count = int(local_anchor_count)
        self.local_anchor_neighbors = int(local_anchor_neighbors)
        self.local_logit_delta_max = float(local_logit_delta_max)
        self.local_residual_dropout = float(local_residual_dropout)
        self.base_frozen = False
        if freeze_base:
            self.freeze_base_parameters()

    def _base_modules(self):
        return (
            self.base_projection,
            self.decoder,
            self.projections,
            self.refiners,
        )

    def freeze_base_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if not name.startswith("local_residual."):
                parameter.requires_grad_(False)
                parameter.grad = None
        for module in self._base_modules():
            module.eval()
        self.base_frozen = True

    def base_state_keys(self):
        return {
            key for key in self.state_dict() if not key.startswith("local_residual.")
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.base_frozen:
            for module in self._base_modules():
                module.eval()
            self.local_residual.train(mode)
        return self

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        diagnostics = dict(super().feature_diagnostics())
        diagnostics.update(self.local_residual.diagnostics())
        return diagnostics

    def forward_with_base(
        self, feature_levels: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = torch.no_grad() if self.base_frozen else torch.enable_grad()
        with context:
            grid = self._fuse(feature_levels)
            fullgrid_features = self.decoder[0](grid)
            base_logits = self.decoder[1:](fullgrid_features)
        if self.base_frozen:
            fullgrid_features = fullgrid_features.detach()
            base_logits = base_logits.detach()
        local_delta = self.local_residual(fullgrid_features)
        return base_logits + local_delta, base_logits, local_delta

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        fused_logits, _, _ = self.forward_with_base(feature_levels)
        return fused_logits


class CanonicalSupportSelector(nn.Module):
    """Predict cumulative contact probabilities on the canonical mesh."""

    def __init__(
        self,
        *,
        input_dim: int,
        tactile_dim: int,
        thresholds: Sequence[float],
        anchor_count: int = 512,
        anchor_neighbors: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()
        thresholds = tuple(float(value) for value in thresholds)
        if not thresholds:
            raise ValueError("support selector requires at least one threshold")
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError("support selector thresholds must lie in (0, 1)")
        if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("support selector thresholds must be strictly increasing")
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("support_selector_dropout must lie in [0, 1]")

        self.input_dim = int(input_dim)
        self.tactile_dim = int(tactile_dim)
        self.anchor_count = int(anchor_count)
        self.anchor_neighbors = int(anchor_neighbors)
        self.thresholds = thresholds
        self.output_count = len(thresholds)
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.anchor_head = nn.Linear(
            self.input_dim,
            self.anchor_count * self.output_count,
        )
        nn.init.zeros_(self.anchor_head.weight)
        nn.init.zeros_(self.anchor_head.bias)

        (
            anchor_vertex_indices,
            vertex_anchor_indices,
            vertex_anchor_weights,
            valid_palm_mask,
        ) = _canonical_rbf_assignment(
            tactile_dim=self.tactile_dim,
            anchor_count=self.anchor_count,
            neighbor_count=self.anchor_neighbors,
        )
        self.register_buffer("anchor_vertex_indices", anchor_vertex_indices)
        self.register_buffer("vertex_anchor_indices", vertex_anchor_indices)
        self.register_buffer("vertex_anchor_weights", vertex_anchor_weights)
        self.register_buffer("valid_palm_mask", valid_palm_mask)
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_diagnostics

    def forward(self, fullgrid_features: torch.Tensor) -> torch.Tensor:
        if fullgrid_features.ndim != 2 or fullgrid_features.shape[1] != self.input_dim:
            raise ValueError(
                f"Support selector expects [B,{self.input_dim}], got "
                f"{tuple(fullgrid_features.shape)}"
            )
        normalized = self.dropout(self.input_norm(fullgrid_features))
        anchor_logits = self.anchor_head(normalized).reshape(
            fullgrid_features.shape[0],
            self.anchor_count,
            self.output_count,
        )
        per_vertex = anchor_logits[:, self.vertex_anchor_indices, :]
        logits = (
            per_vertex
            * self.vertex_anchor_weights.to(dtype=per_vertex.dtype)[None, :, :, None]
        ).sum(dim=2)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits * self.valid_palm_mask.to(dtype=logits.dtype)[None, None]

        if not self.training:
            valid_logits = logits[:, :, self.valid_palm_mask]
            probabilities = torch.sigmoid(valid_logits.detach().float())
            self._last_diagnostics = {
                "selector_probability_mean": probabilities.mean(),
                "selector_probability_std": probabilities.std(unbiased=False),
                "selector_logit_rms": valid_logits.detach().float().pow(2).mean().sqrt(),
                "selector_monotonic_violation": (
                    (valid_logits[:, 1:] > valid_logits[:, :-1]).float().mean()
                    if self.output_count > 1
                    else valid_logits.new_zeros(())
                ),
            }
        else:
            self._last_diagnostics = {}
        return logits


class ContactSpatialResidualBlock(nn.Module):
    """A small contact-specific spatial refinement block."""

    def __init__(self, channels: int, dropout: float = 0.10):
        super().__init__()
        if not 0.0 <= float(dropout) <= 1.0:
            raise ValueError("support_selector_dropout must lie in [0, 1]")
        self.block = nn.Sequential(
            nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
            ChannelLayerNorm(int(channels)),
            nn.GELU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1),
            ChannelLayerNorm(int(channels)),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.block(features))


class RawDinoContactFusion(nn.Module):
    """Fuse frozen multilevel DINO maps without reusing tactile projections."""

    def __init__(
        self,
        *,
        layer_indices: Sequence[int],
        input_channels: int,
        output_channels: int,
        neck_channels: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_indices = tuple(int(value) for value in layer_indices)
        self.projections = nn.ModuleDict(
            {
                str(layer): nn.Sequential(
                    nn.Conv2d(int(input_channels), int(neck_channels), kernel_size=1),
                    ChannelLayerNorm(int(neck_channels)),
                    nn.GELU(),
                )
                for layer in self.layer_indices
            }
        )
        concatenated_channels = len(self.layer_indices) * int(neck_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(concatenated_channels, int(output_channels), kernel_size=1),
            ChannelLayerNorm(int(output_channels)),
            nn.GELU(),
            ContactSpatialResidualBlock(int(output_channels), dropout=dropout),
        )
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_diagnostics

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_levels) != len(self.layer_indices):
            raise ValueError(
                f"Raw-DINO selector expected {len(self.layer_indices)} levels, "
                f"got {len(feature_levels)}"
            )
        projected = []
        target_size = tuple(feature_levels[-1].shape[-2:])
        for layer, features in zip(self.layer_indices, feature_levels):
            value = self.projections[str(layer)](features)
            if tuple(value.shape[-2:]) != target_size:
                value = F.interpolate(
                    value,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)
        fused = self.fusion(torch.cat(projected, dim=1))
        if not self.training:
            self._last_diagnostics = {
                "selector_source_level_rms": torch.stack(
                    [
                        value.detach().float().pow(2).mean().sqrt()
                        for value in projected
                    ]
                ),
                "selector_source_fused_rms": (
                    fused.detach().float().pow(2).mean().sqrt()
                ),
            }
        else:
            self._last_diagnostics = {}
        return fused


class CanonicalSpatialSupportSelector(nn.Module):
    """Predict canonical support from a contact-specific spatial feature grid."""

    def __init__(
        self,
        *,
        input_channels: int,
        grid_size: Sequence[int],
        neck_channels: int,
        hidden_dim: int,
        tactile_dim: int,
        thresholds: Sequence[float],
        anchor_count: int,
        anchor_neighbors: int,
        dropout: float,
    ):
        super().__init__()
        thresholds = tuple(float(value) for value in thresholds)
        if not thresholds:
            raise ValueError("support selector requires at least one threshold")
        if any(not 0.0 < value < 1.0 for value in thresholds):
            raise ValueError("support selector thresholds must lie in (0, 1)")
        if any(left >= right for left, right in zip(thresholds, thresholds[1:])):
            raise ValueError("support selector thresholds must be strictly increasing")
        if int(neck_channels) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("support selector neck/hidden dimensions must be positive")

        self.grid_size = tuple(int(value) for value in grid_size)
        self.neck_channels = int(neck_channels)
        self.hidden_dim = int(hidden_dim)
        self.tactile_dim = int(tactile_dim)
        self.anchor_count = int(anchor_count)
        self.anchor_neighbors = int(anchor_neighbors)
        self.thresholds = thresholds
        self.output_count = len(thresholds)
        self.spatial_neck = nn.Sequential(
            nn.Conv2d(int(input_channels), self.neck_channels, kernel_size=1),
            ChannelLayerNorm(self.neck_channels),
            nn.GELU(),
            ContactSpatialResidualBlock(self.neck_channels, dropout=dropout),
        )
        flattened_dim = self.neck_channels * self.grid_size[0] * self.grid_size[1]
        self.anchor_decoder = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(flattened_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            ResidualBlock(self.hidden_dim, dropout_probability=float(dropout)),
            nn.Linear(self.hidden_dim, self.anchor_count * self.output_count),
        )
        nn.init.zeros_(self.anchor_decoder[-1].weight)
        nn.init.zeros_(self.anchor_decoder[-1].bias)

        (
            anchor_vertex_indices,
            vertex_anchor_indices,
            vertex_anchor_weights,
            valid_palm_mask,
        ) = _canonical_rbf_assignment(
            tactile_dim=self.tactile_dim,
            anchor_count=self.anchor_count,
            neighbor_count=self.anchor_neighbors,
        )
        self.register_buffer("anchor_vertex_indices", anchor_vertex_indices)
        self.register_buffer("vertex_anchor_indices", vertex_anchor_indices)
        self.register_buffer("vertex_anchor_weights", vertex_anchor_weights)
        self.register_buffer("valid_palm_mask", valid_palm_mask)
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_diagnostics

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        if spatial_features.ndim != 4:
            raise ValueError(
                "Spatial support selector expects a BCHW feature map, got "
                f"{tuple(spatial_features.shape)}"
            )
        if tuple(spatial_features.shape[-2:]) != self.grid_size:
            raise ValueError(
                f"Spatial support selector expects grid {self.grid_size}, got "
                f"{tuple(spatial_features.shape[-2:])}"
            )
        neck = self.spatial_neck(spatial_features)
        anchor_logits = self.anchor_decoder(neck.flatten(1)).reshape(
            spatial_features.shape[0],
            self.anchor_count,
            self.output_count,
        )
        per_vertex = anchor_logits[:, self.vertex_anchor_indices, :]
        logits = (
            per_vertex
            * self.vertex_anchor_weights.to(dtype=per_vertex.dtype)[None, :, :, None]
        ).sum(dim=2)
        logits = logits.permute(0, 2, 1).contiguous()
        logits = logits * self.valid_palm_mask.to(dtype=logits.dtype)[None, None]

        if not self.training:
            valid_logits = logits[:, :, self.valid_palm_mask]
            probabilities = torch.sigmoid(valid_logits.detach().float())
            self._last_diagnostics = {
                "selector_source_grid_rms": (
                    spatial_features.detach().float().pow(2).mean().sqrt()
                ),
                "selector_neck_rms": neck.detach().float().pow(2).mean().sqrt(),
                "selector_probability_mean": probabilities.mean(),
                "selector_probability_std": probabilities.std(unbiased=False),
                "selector_logit_rms": valid_logits.detach().float().pow(2).mean().sqrt(),
                "selector_monotonic_violation": (
                    (valid_logits[:, 1:] > valid_logits[:, :-1]).float().mean()
                    if self.output_count > 1
                    else valid_logits.new_zeros(())
                ),
            }
        else:
            self._last_diagnostics = {}
        return logits


class ContactSpecificSupportSelector(nn.Module):
    """Select contact from either the frozen ReZero grid or raw DINO levels."""

    def __init__(
        self,
        *,
        source: str,
        layer_indices: Sequence[int],
        input_channels: int,
        feature_channels: int,
        grid_size: Sequence[int],
        neck_channels: int,
        hidden_dim: int,
        tactile_dim: int,
        thresholds: Sequence[float],
        anchor_count: int,
        anchor_neighbors: int,
        dropout: float,
        base_conditioning: str = "none",
    ):
        super().__init__()
        source = str(source).strip().lower()
        if source not in {"rezero_grid", "raw_dino"}:
            raise ValueError(
                "support_selector_feature_source must be rezero_grid or raw_dino "
                "for architecture=spatial_mlp"
            )
        self.source = source
        self.base_conditioning = str(base_conditioning).strip().lower()
        if self.base_conditioning not in {"none", "real", "constant_control"}:
            raise ValueError(
                "support_selector_base_conditioning must be none, real, or "
                "constant_control"
            )
        # Build the shared selector first so both source controls receive the
        # same initialization under an identical seed.
        self.selector = CanonicalSpatialSupportSelector(
            input_channels=feature_channels,
            grid_size=grid_size,
            neck_channels=neck_channels,
            hidden_dim=hidden_dim,
            tactile_dim=tactile_dim,
            thresholds=thresholds,
            anchor_count=anchor_count,
            anchor_neighbors=anchor_neighbors,
            dropout=dropout,
        )
        self.raw_fusion = None
        if self.source == "raw_dino":
            self.raw_fusion = RawDinoContactFusion(
                layer_indices=layer_indices,
                input_channels=input_channels,
                output_channels=feature_channels,
                neck_channels=neck_channels,
                dropout=dropout,
            )
        self.base_conditioner = None
        if self.base_conditioning != "none":
            if len(tuple(thresholds)) != 1:
                raise ValueError(
                    "Base-conditioned support selection requires one output"
                )
            self.base_conditioner = nn.Sequential(
                nn.Linear(3, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(64, 1),
            )
            nn.init.normal_(self.base_conditioner[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.base_conditioner[-1].bias)
        self._last_diagnostics: Dict[str, torch.Tensor] = {}

    def diagnostics(self) -> Dict[str, torch.Tensor]:
        diagnostics = dict(self.selector.diagnostics())
        if self.raw_fusion is not None:
            diagnostics.update(self.raw_fusion.diagnostics())
        diagnostics.update(self._last_diagnostics)
        return diagnostics

    def forward(
        self,
        rezero_grid: torch.Tensor,
        raw_feature_levels: Sequence[torch.Tensor],
        base_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source_grid = rezero_grid
        if self.raw_fusion is not None:
            source_grid = self.raw_fusion(raw_feature_levels)
        selector_logits = self.selector(source_grid)
        if self.base_conditioner is None:
            return selector_logits
        if base_logits is None:
            raise ValueError(
                "Base-conditioned support selection requires frozen base logits"
            )
        if base_logits.ndim != 2 or base_logits.shape != (
            selector_logits.shape[0],
            selector_logits.shape[2],
        ):
            raise ValueError(
                "Base logits must match selector vertices, got "
                f"{tuple(base_logits.shape)} for {tuple(selector_logits.shape)}"
            )
        detached_base = base_logits.detach().to(dtype=selector_logits.dtype)
        bounded_base_logit = detached_base.clamp(-12.0, 12.0) / 12.0
        centered_base_probability = 2.0 * torch.sigmoid(detached_base) - 1.0
        if self.base_conditioning == "constant_control":
            bounded_base_logit = torch.zeros_like(bounded_base_logit)
            centered_base_probability = torch.zeros_like(
                centered_base_probability
            )
        conditioning = torch.stack(
            (
                selector_logits[:, 0],
                bounded_base_logit,
                centered_base_probability,
            ),
            dim=-1,
        )
        validity_logits = self.base_conditioner(conditioning).squeeze(-1)[:, None]
        validity_logits = validity_logits * self.selector.valid_palm_mask.to(
            dtype=validity_logits.dtype
        )[None, None]
        if not self.training:
            valid = self.selector.valid_palm_mask
            valid_logits = validity_logits[:, :, valid].detach().float()
            self._last_diagnostics = {
                "selector_probability_mean": torch.sigmoid(valid_logits).mean(),
                "selector_probability_std": torch.sigmoid(valid_logits).std(
                    unbiased=False
                ),
                "selector_logit_rms": valid_logits.pow(2).mean().sqrt(),
                "selector_base_probability_mean": torch.sigmoid(
                    detached_base[:, valid].detach().float()
                ).mean(),
                "selector_base_probability_std": torch.sigmoid(
                    detached_base[:, valid].detach().float()
                ).std(unbiased=False),
                "selector_condition_logit_rms": bounded_base_logit[
                    :, valid
                ].detach().float().pow(2).mean().sqrt(),
            }
        else:
            self._last_diagnostics = {}
        return validity_logits


class DenseV2DinoSupportSelectorTactileHead(DenseV2DinoReZeroTactileHead):
    """Frozen FullGrid pressure baseline with an independently supervised selector."""

    def __init__(
        self,
        tactile_dim: int,
        layer_indices: Sequence[int],
        residual_max_scale: float = 0.10,
        residual_rms_budget: float = 0.50,
        input_channels: int = 1280,
        channels: int = 256,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
        grid_size: Sequence[int] = (16, 12),
        pool_output_channels: int = 32,
        decoder_hidden_dim: int = 512,
        model_initialization_order: str = CANONICAL_MODEL_INITIALIZATION_ORDER,
        selector_thresholds: Sequence[float] = (0.10,),
        selector_anchor_count: int = 512,
        selector_anchor_neighbors: int = 4,
        selector_dropout: float = 0.10,
        selector_architecture: str = "linear",
        selector_feature_source: str = "fullgrid32",
        selector_neck_channels: int = 64,
        selector_hidden_dim: int = 512,
        selector_mode: str = "contact",
        selector_base_conditioning: str = "real",
    ):
        if str(pool_layout) != "fullgrid32":
            raise ValueError("The support selector requires pool_layout=fullgrid32")
        super().__init__(
            tactile_dim=tactile_dim,
            layer_indices=layer_indices,
            residual_max_scale=residual_max_scale,
            residual_rms_budget=residual_rms_budget,
            input_channels=input_channels,
            channels=channels,
            pool_layout=pool_layout,
            decoder_dropout_scale=decoder_dropout_scale,
            grid_size=grid_size,
            pool_output_channels=pool_output_channels,
            decoder_hidden_dim=decoder_hidden_dim,
            model_initialization_order=model_initialization_order,
        )
        self.selector_architecture = str(selector_architecture).strip().lower()
        self.selector_feature_source = str(selector_feature_source).strip().lower()
        self.selector_neck_channels = int(selector_neck_channels)
        self.selector_hidden_dim = int(selector_hidden_dim)
        self.selector_mode = str(selector_mode).strip().lower()
        if self.selector_architecture == "linear":
            if self.selector_feature_source != "fullgrid32":
                raise ValueError(
                    "The legacy linear selector requires "
                    "support_selector_feature_source=fullgrid32"
                )
            self.support_selector = CanonicalSupportSelector(
                input_dim=self.decoder_input_dim,
                tactile_dim=tactile_dim,
                thresholds=selector_thresholds,
                anchor_count=selector_anchor_count,
                anchor_neighbors=selector_anchor_neighbors,
                dropout=selector_dropout,
            )
        elif self.selector_architecture == "spatial_mlp":
            self.support_selector = ContactSpecificSupportSelector(
                source=self.selector_feature_source,
                layer_indices=layer_indices,
                input_channels=input_channels,
                feature_channels=channels,
                grid_size=grid_size,
                neck_channels=self.selector_neck_channels,
                hidden_dim=self.selector_hidden_dim,
                tactile_dim=tactile_dim,
                thresholds=selector_thresholds,
                anchor_count=selector_anchor_count,
                anchor_neighbors=selector_anchor_neighbors,
                dropout=selector_dropout,
                base_conditioning=(
                    selector_base_conditioning
                    if self.selector_mode == "down_error"
                    else "none"
                ),
            )
        else:
            raise ValueError(
                "support_selector_architecture must be linear or spatial_mlp"
            )
        self.selector_thresholds = tuple(float(value) for value in selector_thresholds)
        self.selector_anchor_count = int(selector_anchor_count)
        self.selector_anchor_neighbors = int(selector_anchor_neighbors)
        self.selector_dropout = float(selector_dropout)
        self.base_frozen = False
        self.freeze_base_parameters()

    def _base_modules(self):
        return (
            self.base_projection,
            self.decoder,
            self.projections,
            self.refiners,
        )

    def freeze_base_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if not name.startswith("support_selector."):
                parameter.requires_grad_(False)
                parameter.grad = None
        for module in self._base_modules():
            module.eval()
        self.base_frozen = True

    def base_state_keys(self):
        return {
            key for key in self.state_dict() if not key.startswith("support_selector.")
        }

    def extension_state_keys(self):
        return {
            key for key in self.state_dict() if key.startswith("support_selector.")
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.base_frozen:
            for module in self._base_modules():
                module.eval()
            self.support_selector.train(mode)
        return self

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        diagnostics = dict(super().feature_diagnostics())
        diagnostics.update(self.support_selector.diagnostics())
        return diagnostics

    def forward_with_selector(
        self, feature_levels: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            grid = self._fuse(feature_levels)
            fullgrid_features = self.decoder[0](grid)
            base_logits = self.decoder[1:](fullgrid_features)
        if self.selector_architecture == "linear":
            selector_logits = self.support_selector(fullgrid_features.detach())
        else:
            selector_logits = self.support_selector(
                grid.detach(),
                tuple(features.detach() for features in feature_levels),
                base_logits=base_logits.detach(),
            )
        return base_logits.detach(), selector_logits

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        base_logits, _ = self.forward_with_selector(feature_levels)
        return base_logits


class DinoV3BackboneAdapter(nn.Module):
    """Frozen local DINOv3 H+/16 backbone returning BCHW patch maps."""

    MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"

    def __init__(self, weights_path: str, image_size=(256, 192)):
        super().__init__()
        weights = Path(weights_path).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
        import timm
        from timm.models._helpers import load_state_dict
        from timm.models.eva import checkpoint_filter_fn

        self.weights_path = str(weights.resolve())
        try:
            self.model = timm.create_model(
                self.MODEL_NAME,
                pretrained=False,
                img_size=tuple(image_size),
                dynamic_img_size=False,
                num_classes=0,
            )
            state_dict = load_state_dict(
                self.weights_path, use_ema=False, device="cpu"
            )
            state_dict = checkpoint_filter_fn(state_dict, self.model)
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to strictly load local DINOv3 H+/16 weights from "
                f"{self.weights_path}"
            ) from exc
        if int(self.model.num_features) != 1280:
            raise RuntimeError(
                f"Expected 1280 DINOv3 feature channels, got {self.model.num_features}"
            )
        patch_size = tuple(int(value) for value in self.model.patch_embed.patch_size)
        if patch_size != (16, 16):
            raise RuntimeError(f"Expected DINOv3 patch size 16x16, got {patch_size}")
        self.num_prefix_tokens = int(self.model.num_prefix_tokens)
        if self.num_prefix_tokens != 5:
            raise RuntimeError(
                f"Expected 5 DINOv3 prefix tokens, got {self.num_prefix_tokens}"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.train(False)

    def get_num_layers(self) -> int:
        return len(self.model.blocks)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self, image: torch.Tensor, layer_indices: Sequence[int]
    ) -> Sequence[torch.Tensor]:
        layer_indices = tuple(int(layer) for layer in layer_indices)
        depth = self.get_num_layers()
        if not layer_indices or tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError("DINO layers must contain unique increasing indices")
        if min(layer_indices) < 1 or max(layer_indices) > depth:
            raise ValueError(
                f"DINO layers must lie in [1, {depth}], got {layer_indices}"
            )
        feature_maps = self.model.forward_intermediates(
            image,
            indices=[layer - 1 for layer in layer_indices],
            return_prefix_tokens=False,
            norm=True,
            stop_early=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        height, width = self.model.patch_embed.dynamic_feat_size(image.shape[-2:])
        for layer, feature_map in zip(layer_indices, feature_maps):
            expected_shape = (image.shape[0], 1280, height, width)
            if tuple(feature_map.shape) != expected_shape:
                raise RuntimeError(
                    f"DINO layer {layer} returned {tuple(feature_map.shape)}, "
                    f"expected {expected_shape}"
                )
        return feature_maps


class DinoTactileModel(pl.LightningModule):
    """Standalone frozen-DINO to canonical tactile model."""

    def __init__(
        self,
        cfg=None,
        init_renderer: bool = False,
        tactile_only_forward: bool = True,
        tactile_loss_scale: float = 10.0,
        tactile_head_type: str = "dense_v2_dino_rezero",
        backbone_feature_layers: Sequence[int] = (8, 16, 24, 32),
        visual_backbone: str = "dinov3_hplus",
        dino_weights: str = "",
        dino_residual_max_scale: float = 0.10,
        dino_residual_rms_budget: float = 0.50,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
        input_resolution: Sequence[int] = (256, 192),
        pool_output_channels: int = 32,
        decoder_hidden_dim: int = 512,
        center_aux_hidden_dim: int = 128,
        model_initialization_order: str = CANONICAL_MODEL_INITIALIZATION_ORDER,
        local_anchor_count: int = 512,
        local_anchor_neighbors: int = 4,
        local_logit_delta_max: float = 6.0,
        local_residual_dropout: float = 0.10,
        freeze_local_residual_base: bool = True,
        support_selector_mode: str = "contact",
        support_selector_thresholds: Sequence[float] = (0.02, 0.05, 0.10, 0.20, 0.50),
        support_selector_no_contact_max: float = 0.02,
        support_selector_contact_min: float = 0.10,
        support_selector_dropout: float = 0.10,
        support_selector_monotonicity_weight: float = 0.10,
        support_selector_architecture: str = "linear",
        support_selector_feature_source: str = "fullgrid32",
        support_selector_neck_channels: int = 64,
        support_selector_hidden_dim: int = 512,
        support_selector_base_conditioning: str = "real",
        surface_basis_path: str = "",
        surface_coefficient_dim: int = 4096,
        surface_coefficient_architecture: str = "linear",
        surface_coefficient_hidden_dim: int = 1024,
        surface_target_support_count: int = 4,
        surface_background_probability: float = 1e-3,
        freeze_surface_feature_extractor: bool = True,
    ):
        super().__init__()
        if not tactile_only_forward:
            raise ValueError(
                "The standalone tactile model only supports tactile_only_forward=True"
            )
        if tactile_head_type not in SUPPORTED_TACTILE_HEAD_TYPES:
            raise ValueError(
                f"Unsupported tactile_head_type={tactile_head_type}; "
                f"choose one of {SUPPORTED_TACTILE_HEAD_TYPES}"
            )
        if visual_backbone != "dinov3_hplus":
            raise ValueError("Only visual_backbone=dinov3_hplus is supported")
        if pool_layout not in SUPPORTED_POOL_LAYOUTS:
            raise ValueError(
                f"Unsupported pool_layout={pool_layout}; "
                f"choose one of {SUPPORTED_POOL_LAYOUTS}"
            )

        self.visual_backbone = str(visual_backbone)
        self.dino_weights = str(dino_weights)
        self.input_resolution = parse_input_resolution(input_resolution)
        self.backbone = DinoV3BackboneAdapter(
            self.dino_weights,
            image_size=self.input_resolution,
        )
        self.tactile_dim = default_tactile_dim()
        self.tactile_head_type = str(tactile_head_type)
        requested_layers = tuple(int(layer) for layer in backbone_feature_layers)
        if not requested_layers or requested_layers[-1] != self.backbone.get_num_layers():
            raise ValueError(
                "backbone_feature_layers must end at the final DINO block"
            )
        if self.tactile_head_type == "dense_v2":
            self.backbone_feature_layers = (requested_layers[-1],)
        else:
            if len(requested_layers) < 2:
                raise ValueError(
                    "A multilevel DINO tactile head requires at least two feature layers"
                )
            self.backbone_feature_layers = requested_layers

        self.dino_residual_max_scale = float(dino_residual_max_scale)
        self.dino_residual_rms_budget = float(dino_residual_rms_budget)
        self.pool_layout = str(pool_layout)
        self.feature_grid_size = (
            self.input_resolution[0] // 16,
            self.input_resolution[1] // 16,
        )
        self.pool_grid_size = self.feature_grid_size
        self.pool_output_channels = int(pool_output_channels)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.center_aux_hidden_dim = int(center_aux_hidden_dim)
        self.model_initialization_order = str(model_initialization_order)
        if self.model_initialization_order not in SUPPORTED_MODEL_INITIALIZATION_ORDERS:
            raise ValueError(
                "model_initialization_order must be one of "
                f"{SUPPORTED_MODEL_INITIALIZATION_ORDERS}"
            )
        if self.decoder_hidden_dim < 1:
            raise ValueError("decoder_hidden_dim must be positive")
        if self.center_aux_hidden_dim < 1:
            raise ValueError("center_aux_hidden_dim must be positive")
        self.decoder_dropout_scale = float(decoder_dropout_scale)
        self.local_anchor_count = int(local_anchor_count)
        self.local_anchor_neighbors = int(local_anchor_neighbors)
        self.local_logit_delta_max = float(local_logit_delta_max)
        self.local_residual_dropout = float(local_residual_dropout)
        self.freeze_local_residual_base = bool(freeze_local_residual_base)
        self.support_selector_mode = str(support_selector_mode).strip().lower()
        if self.support_selector_mode not in {"contact", "ordinal", "down_error"}:
            raise ValueError(
                "support_selector_mode must be contact, ordinal, or down_error"
            )
        parsed_selector_thresholds = tuple(
            float(value) for value in support_selector_thresholds
        )
        self.support_selector_no_contact_max = float(
            support_selector_no_contact_max
        )
        self.support_selector_contact_min = float(support_selector_contact_min)
        if not (
            0.0 <= self.support_selector_no_contact_max
            < self.support_selector_contact_min
            < 1.0
        ):
            raise ValueError(
                "support selector thresholds require 0 <= no_contact_max "
                "< contact_min < 1"
            )
        self.support_selector_thresholds = (
            (self.support_selector_contact_min,)
            if self.support_selector_mode in {"contact", "down_error"}
            else parsed_selector_thresholds
        )
        self.support_selector_dropout = float(support_selector_dropout)
        self.support_selector_monotonicity_weight = float(
            support_selector_monotonicity_weight
        )
        if self.support_selector_monotonicity_weight < 0.0:
            raise ValueError(
                "support_selector_monotonicity_weight must be nonnegative"
            )
        self.support_selector_architecture = str(
            support_selector_architecture
        ).strip().lower()
        self.support_selector_feature_source = str(
            support_selector_feature_source
        ).strip().lower()
        self.support_selector_neck_channels = int(
            support_selector_neck_channels
        )
        self.support_selector_hidden_dim = int(support_selector_hidden_dim)
        self.support_selector_base_conditioning = str(
            support_selector_base_conditioning
        ).strip().lower()
        self.surface_basis_path = str(surface_basis_path or "")
        self.surface_coefficient_dim = int(surface_coefficient_dim)
        self.surface_coefficient_architecture = str(
            surface_coefficient_architecture
        ).strip().lower()
        self.surface_coefficient_hidden_dim = int(
            surface_coefficient_hidden_dim
        )
        self.surface_target_support_count = int(surface_target_support_count)
        self.surface_background_probability = float(
            surface_background_probability
        )
        self.freeze_surface_feature_extractor = bool(
            freeze_surface_feature_extractor
        )
        self.surface_basis_artifact_sha256 = ""
        self.surface_basis_tensor_sha256 = ""
        self.surface_sparse_basis_sha256 = ""
        self.surface_valid_vertex_count = 0
        self.surface_maximum_support_count = 0
        if self.support_selector_base_conditioning not in {
            "real",
            "constant_control",
        }:
            raise ValueError(
                "support_selector_base_conditioning must be real or "
                "constant_control"
            )
        if self.support_selector_mode == "down_error" and (
            self.support_selector_architecture != "spatial_mlp"
            or self.support_selector_feature_source != "rezero_grid"
        ):
            raise ValueError(
                "down_error selector mode requires spatial_mlp on rezero_grid"
            )
        self.tactile_only_forward = True
        self.tactile_loss_scale = float(tactile_loss_scale)
        self.tactile_loss_config = TactileLossConfig()

        common_head_args = {
            "tactile_dim": self.tactile_dim,
            "pool_layout": self.pool_layout,
            "decoder_dropout_scale": self.decoder_dropout_scale,
            "grid_size": self.feature_grid_size,
            "pool_output_channels": self.pool_output_channels,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "model_initialization_order": self.model_initialization_order,
        }
        if self.tactile_head_type == "dense_v2":
            self.tactile_head = DenseV2TactileHead(**common_head_args)
        elif self.tactile_head_type == "dense_v2_dino_rezero":
            self.tactile_head = DenseV2DinoReZeroTactileHead(
                **common_head_args,
                layer_indices=self.backbone_feature_layers,
                residual_max_scale=self.dino_residual_max_scale,
                residual_rms_budget=self.dino_residual_rms_budget,
            )
        elif self.tactile_head_type == "dense_v2_dino_center_aux":
            self.tactile_head = DenseV2DinoCenterAuxTactileHead(
                **common_head_args,
                layer_indices=self.backbone_feature_layers,
                center_aux_hidden_dim=self.center_aux_hidden_dim,
                residual_max_scale=self.dino_residual_max_scale,
                residual_rms_budget=self.dino_residual_rms_budget,
            )
        elif self.tactile_head_type == "dense_v2_dino_local_residual":
            self.tactile_head = DenseV2DinoLocalResidualTactileHead(
                **common_head_args,
                layer_indices=self.backbone_feature_layers,
                residual_max_scale=self.dino_residual_max_scale,
                residual_rms_budget=self.dino_residual_rms_budget,
                local_anchor_count=self.local_anchor_count,
                local_anchor_neighbors=self.local_anchor_neighbors,
                local_logit_delta_max=self.local_logit_delta_max,
                local_residual_dropout=self.local_residual_dropout,
                freeze_base=self.freeze_local_residual_base,
            )
        elif self.tactile_head_type == "dense_v2_dino_support_selector":
            self.tactile_head = DenseV2DinoSupportSelectorTactileHead(
                **common_head_args,
                layer_indices=self.backbone_feature_layers,
                residual_max_scale=self.dino_residual_max_scale,
                residual_rms_budget=self.dino_residual_rms_budget,
                selector_thresholds=self.support_selector_thresholds,
                selector_anchor_count=self.local_anchor_count,
                selector_anchor_neighbors=self.local_anchor_neighbors,
                selector_dropout=self.support_selector_dropout,
                selector_architecture=self.support_selector_architecture,
                selector_feature_source=self.support_selector_feature_source,
                selector_neck_channels=self.support_selector_neck_channels,
                selector_hidden_dim=self.support_selector_hidden_dim,
                selector_mode=self.support_selector_mode,
                selector_base_conditioning=(
                    self.support_selector_base_conditioning
                ),
            )
        else:
            self.tactile_head = DenseV2DinoSurfaceBasisTactileHead(
                **common_head_args,
                layer_indices=self.backbone_feature_layers,
                residual_max_scale=self.dino_residual_max_scale,
                residual_rms_budget=self.dino_residual_rms_budget,
                surface_basis_path=self.surface_basis_path,
                surface_coefficient_dim=self.surface_coefficient_dim,
                surface_coefficient_architecture=(
                    self.surface_coefficient_architecture
                ),
                surface_coefficient_hidden_dim=(
                    self.surface_coefficient_hidden_dim
                ),
                surface_target_support_count=(
                    self.surface_target_support_count
                ),
                surface_background_probability=(
                    self.surface_background_probability
                ),
                freeze_surface_feature_extractor=(
                    self.freeze_surface_feature_extractor
                ),
            )
            self.surface_basis_path = self.tactile_head.surface_basis_path
            self.surface_basis_artifact_sha256 = (
                self.tactile_head.surface_basis_artifact_sha256
            )
            self.surface_basis_tensor_sha256 = (
                self.tactile_head.surface_basis_tensor_sha256
            )
            self.surface_sparse_basis_sha256 = (
                self.tactile_head.surface_sparse_basis_sha256
            )
            self.surface_valid_vertex_count = (
                self.tactile_head.surface_valid_vertex_count
            )
            self.surface_maximum_support_count = (
                self.tactile_head.surface_maximum_support_count
            )
            self.surface_coefficient_architecture = (
                self.tactile_head.surface_coefficient_architecture
            )
            self.surface_coefficient_hidden_dim = (
                self.tactile_head.surface_coefficient_hidden_dim
            )
        self.pool_valid_tokens = int(self.tactile_head.pool_valid_tokens)
        self.decoder_input_dim = int(self.tactile_head.decoder_input_dim)
        self.automatic_optimization = True

    def set_tactile_loss_config(self, config: TactileLossConfig) -> None:
        self.tactile_loss_config = config

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract_tactile_features(self, image: torch.Tensor):
        return self.backbone(image, self.backbone_feature_layers)

    def forward_step(
        self,
        batch: Dict,
        train: bool = False,
        *,
        compute_auxiliary: Optional[bool] = None,
    ) -> Dict:
        image = batch["img"]
        if tuple(image.shape[-2:]) != self.input_resolution:
            raise ValueError(
                f"Model expects input {self.input_resolution}, "
                f"got {tuple(image.shape[-2:])}"
            )
        with torch.no_grad():
            conditioning_features = self._extract_tactile_features(image)
        center_aux_logits = None
        center_aux_presence_logits = None
        if compute_auxiliary is None:
            compute_auxiliary = bool(train)
        if (
            compute_auxiliary
            and hasattr(self.tactile_head, "forward_with_center_aux")
        ):
            (
                pred_logits,
                center_aux_logits,
                center_aux_presence_logits,
            ) = self.tactile_head.forward_with_center_aux(
                conditioning_features
            )
            base_logits = None
            local_delta = None
        elif hasattr(self.tactile_head, "forward_with_selector"):
            pred_logits, selector_logits = (
                self.tactile_head.forward_with_selector(conditioning_features)
            )
            base_logits = None
            local_delta = None
        elif hasattr(self.tactile_head, "forward_with_base"):
            pred_logits, base_logits, local_delta = (
                self.tactile_head.forward_with_base(conditioning_features)
            )
        else:
            pred_logits = self.tactile_head(conditioning_features)
            base_logits = None
            local_delta = None
        output = {
            "losses": {},
            "pred_logits": pred_logits,
            "pred_tactile": torch.sigmoid(pred_logits),
        }
        if center_aux_logits is not None:
            output.update(
                {
                    "center_aux_logits": center_aux_logits,
                    "center_aux_presence_logits": (
                        center_aux_presence_logits
                    ),
                }
            )
        if hasattr(self.tactile_head, "forward_with_selector"):
            output.update({
                "support_selector_logits": selector_logits,
                "support_selector_thresholds": self.support_selector_thresholds,
            })
            if not train:
                output["support_selector_probabilities"] = torch.sigmoid(
                    selector_logits
                )
        if base_logits is not None:
            output.update(
                {
                    "base_pred_logits": base_logits,
                    "base_pred_tactile": torch.sigmoid(base_logits),
                    "local_logit_delta": local_delta,
                }
            )
        return output

    def compute_loss(
        self, batch: Dict, output: Dict, train: bool = True
    ) -> torch.Tensor:
        if "support_selector_logits" in output:
            selector_logits = output["support_selector_logits"].float()
            target = batch["tactile_signal"].float()
            palm = batch["palm_mask"].float() > 0.5
            has_tactile = batch["has_tactile"].float() > 0.5
            valid = palm & has_tactile[:, None]
            base_prediction = output["pred_tactile"].detach().float()
            clear = (
                (target <= self.support_selector_no_contact_max)
                | (target >= self.support_selector_contact_min)
            )
            base_contact = (
                base_prediction >= self.support_selector_contact_min
            )
            threshold_losses = []
            positive_fractions = []
            for threshold_index, threshold in enumerate(
                self.support_selector_thresholds
            ):
                labels = target >= self.support_selector_contact_min
                if self.support_selector_mode == "ordinal":
                    labels = target > float(threshold)
                eligible = valid
                if self.support_selector_mode == "contact":
                    eligible = valid & clear
                elif self.support_selector_mode == "down_error":
                    # The selector predicts whether an existing frozen-base
                    # contact should be retained. Low validity is the down-veto.
                    eligible = valid & clear & base_contact
                logits = selector_logits[:, threshold_index]
                element_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels.to(dtype=logits.dtype),
                    reduction="none",
                )
                positive = eligible & labels
                negative = eligible & ~labels
                class_losses = []
                if bool(positive.any().item()):
                    class_losses.append(element_loss[positive].mean())
                if bool(negative.any().item()):
                    class_losses.append(element_loss[negative].mean())
                if not class_losses:
                    if self.support_selector_mode == "down_error":
                        threshold_losses.append(logits.sum() * 0.0)
                        positive_fractions.append(logits.new_zeros(()))
                        continue
                    raise RuntimeError(
                        "Support selector batch has no eligible canonical vertices"
                    )
                threshold_losses.append(torch.stack(class_losses).mean())
                positive_fractions.append(
                    positive.sum().to(dtype=torch.float32)
                    / eligible.sum().clamp_min(1).to(dtype=torch.float32)
                )
            selector_bce = torch.stack(threshold_losses).mean()
            if selector_logits.shape[1] > 1:
                monotonic_values = F.relu(
                    selector_logits[:, 1:] - selector_logits[:, :-1]
                )
                monotonic_mask = valid[:, None].expand_as(monotonic_values)
                selector_monotonic = monotonic_values[monotonic_mask].mean()
            else:
                selector_monotonic = selector_bce.new_zeros(())
            total_loss = selector_bce + (
                self.support_selector_monotonicity_weight * selector_monotonic
            )
            output["losses"].update(
                {
                    "loss_selector_balanced_bce": selector_bce.detach(),
                    "loss_selector_monotonic": selector_monotonic.detach(),
                    "loss_selector_total": total_loss.detach(),
                    "diagnostics_selector_positive_fraction": torch.stack(
                        positive_fractions
                    ).mean().detach(),
                    "loss_total": total_loss.detach(),
                    "loss_current_ramp": total_loss.detach(),
                    "loss_direct_raw": selector_bce.detach(),
                    "loss_full_ramp_reference": total_loss.detach(),
                }
            )
            return total_loss

        tactile_loss, tactile_losses = compute_tactile_loss(
            pred=output["pred_tactile"],
            logits=output["pred_logits"],
            target=batch["tactile_signal"],
            palm_mask=batch["palm_mask"],
            valid_mask=batch["has_tactile"],
            dataset_batch=batch.get("dataset"),
            config=self.tactile_loss_config,
            current_epoch=getattr(self, "current_epoch", 0),
            sample_weight=batch.get("sample_weight"),
        )
        center_aux_loss = tactile_loss.new_zeros(())
        center_aux_losses = {}
        if "center_aux_logits" in output:
            center_aux_loss, center_aux_losses = (
                compute_center_auxiliary_loss(
                    center_logits=output["center_aux_logits"],
                    presence_logits=output[
                        "center_aux_presence_logits"
                    ],
                    target=batch["tactile_signal"],
                    palm_mask=batch["palm_mask"],
                    valid_mask=batch["has_tactile"],
                    config=self.tactile_loss_config,
                    current_epoch=getattr(self, "current_epoch", 0),
                    sample_weight=batch.get("sample_weight"),
                )
            )
        tactile_loss_with_aux = tactile_loss + center_aux_loss
        total_loss = self.tactile_loss_scale * tactile_loss_with_aux
        output["losses"].update(tactile_losses)
        output["losses"].update(center_aux_losses)
        output["losses"]["loss_tactile_with_aux"] = (
            tactile_loss_with_aux.detach()
        )
        output["losses"]["loss_total"] = total_loss.detach()
        output["losses"]["loss_current_ramp"] = total_loss.detach()
        output["losses"]["loss_direct_raw"] = tactile_losses[
            "loss_base_tactile"
        ]
        output["losses"]["loss_full_ramp_reference"] = (
            self.tactile_loss_scale * tactile_losses["loss_full_ramp"]
        ).detach()
        return total_loss
