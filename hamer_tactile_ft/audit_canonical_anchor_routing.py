#!/usr/bin/env python3
"""Train and evaluate local image-to-canonical anchor routing probes.

The probe reuses immutable Stage 0.7 FullGrid features. It keeps the formal
FullGrid prediction as a frozen base and learns a zero-initialized, bounded
surface-basis logit residual. Each K4096 coefficient can read only the routed
state of its nearest geometry-defined canonical anchor.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.csgraph import shortest_path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.audit_canonical_localization import (
    MetricAccumulator,
    _adjacency_csr,
    _load_mesh_and_palm_graph,
)
from hamer_tactile_ft.audit_local_controllability import CacheGroup
from hamer_tactile_ft.audit_surface_decoder_learnability import (
    _batch_array,
    _decode_coefficients,
    _finish_memmap,
    _load_runtime_basis,
    _loss_config,
    _open_atomic_memmap,
)
from hamer_tactile_ft.audit_surface_mapping_attribution import (
    SPLITS,
    _load_prepared_split,
)
from hamer_tactile_ft.losses import compute_tactile_loss
from tactile_input_priors.feature_cache import sha256_file


SCHEMA = "canonical_anchor_routing_v1"
ROUTING_MODES = ("competitive", "independent")
ROUTING_SOURCES = ("spatial", "global_control")
ROUTING_ARCHITECTURES = ("legacy", "evidence_only")
ROUTING_FEATURE_SOURCES = ("projected32", "rezero256")
EVAL_CONTROLS = ("configured", "real", "shuffle_spatial", "global_repeat")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    observed: set[str] = set()
    for row in rows:
        for key in row:
            if key not in observed:
                fields.append(key)
                observed.add(key)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _contract_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _load_prepared_root(
    prepared_root: Path,
) -> dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]]:
    prepared = {
        split: _load_prepared_split(prepared_root / split) for split in SPLITS
    }
    basis_hashes = {
        str(metadata["contract"]["surface_basis_sha256"])
        for metadata, _ in prepared.values()
    }
    base_hashes = {
        str(metadata["contract"]["base_checkpoint_sha256"])
        for metadata, _ in prepared.values()
    }
    feature_dims = {int(metadata["feature_dim"]) for metadata, _ in prepared.values()}
    grid_sizes = {
        tuple(int(value) for value in metadata["grid_size"])
        for metadata, _ in prepared.values()
    }
    if not (
        len(basis_hashes)
        == len(base_hashes)
        == len(feature_dims)
        == len(grid_sizes)
        == 1
    ):
        raise RuntimeError("Prepared routing splits do not share one model contract")
    return prepared


def _load_sample_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Invalid sample row at {path}:{line_number}")
            rows.append(dict(value))
    return rows


def _select_feature_source(
    prepared: Mapping[
        str, tuple[dict[str, Any], dict[str, np.ndarray]]
    ],
    *,
    feature_source: str,
    raw_prepared_root: str | Path | None,
) -> dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]]:
    if feature_source not in ROUTING_FEATURE_SOURCES:
        raise ValueError(f"Unsupported routing feature source {feature_source!r}")
    result: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}
    raw_root = None
    if feature_source == "rezero256":
        if not raw_prepared_root:
            raise ValueError("rezero256 routing requires --raw-prepared-root")
        raw_root = Path(raw_prepared_root).expanduser().resolve(strict=True)
    for split, (raw_metadata, raw_arrays) in prepared.items():
        metadata = dict(raw_metadata)
        arrays = dict(raw_arrays)
        if feature_source == "projected32":
            metadata["routing_feature_source"] = feature_source
            metadata["routing_feature_contract_sha256"] = str(
                metadata["contract_sha256"]
            )
        else:
            assert raw_root is not None
            split_root = raw_root / split
            done_path = split_root / "RAW_PREPARED.json"
            if not done_path.is_file():
                raise FileNotFoundError(
                    f"Aligned ReZero routing features are incomplete: {done_path}"
                )
            raw_contract = json.loads(done_path.read_text(encoding="utf-8"))
            if raw_contract.get("base_prepared_contract_sha256") != str(
                metadata["contract_sha256"]
            ):
                raise RuntimeError(
                    f"ReZero routing features for {split} use another Stage 0.7 sample set"
                )
            features = np.load(split_root / "features.npy", mmap_mode="r")
            if features.ndim != 2 or len(features) != int(metadata["sample_count"]):
                raise RuntimeError(
                    f"Malformed ReZero routing feature array for {split}: {features.shape}"
                )
            grid_size = tuple(int(value) for value in metadata["grid_size"])
            token_count = grid_size[0] * grid_size[1]
            if features.shape[1] % token_count:
                raise RuntimeError(
                    f"ReZero routing features for {split} are not token-factorable"
                )
            arrays["features"] = features
            metadata["feature_dim"] = int(features.shape[1])
            metadata["pool_output_channels"] = int(features.shape[1] // token_count)
            metadata["routing_feature_source"] = feature_source
            metadata["routing_feature_contract_sha256"] = str(
                raw_contract["contract_sha256"]
            )
        result[split] = (metadata, arrays)
    dimensions = {
        int(metadata["feature_dim"]) for metadata, _ in result.values()
    }
    if len(dimensions) != 1:
        raise RuntimeError("Selected routing features disagree across splits")
    return result


def _fixed_2d_sincos(height: int, width: int, dimension: int) -> torch.Tensor:
    if dimension % 4 != 0:
        raise ValueError("Routing dimension must be divisible by four")
    quarter = dimension // 4
    frequency = torch.exp(
        -math.log(10000.0) * torch.arange(quarter, dtype=torch.float32) / max(quarter, 1)
    )
    y = torch.arange(height, dtype=torch.float32)[:, None] * frequency[None]
    x = torch.arange(width, dtype=torch.float32)[:, None] * frequency[None]
    y_encoding = torch.cat((y.sin(), y.cos()), dim=1)
    x_encoding = torch.cat((x.sin(), x.cos()), dim=1)
    return torch.cat(
        (
            y_encoding[:, None].expand(-1, width, -1),
            x_encoding[None].expand(height, -1, -1),
        ),
        dim=2,
    ).reshape(height * width, dimension)


def _fixed_xyz_fourier(xyz: np.ndarray, bands: int = 4) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(xyz, dtype=np.float32))
    value = value - value.mean(dim=0, keepdim=True)
    radius = value.square().sum(dim=1).sqrt().amax().clamp_min(1e-6)
    value = value / radius
    features = [value]
    for band in range(int(bands)):
        phase = value * (math.pi * (2.0**band))
        features.extend((phase.sin(), phase.cos()))
    return torch.cat(features, dim=1)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if not len(values):
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    if cumulative[-1] <= 0.0:
        return float(values[-1])
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[min(index, len(values) - 1)])


def _build_routing_geometry(
    artifact: Mapping[str, Any], anchor_count: int
) -> dict[str, Any]:
    valid_indices = artifact["valid_vertex_indices"].detach().cpu().long().numpy()
    coefficient_anchor_rows = (
        artifact["anchor_prefix_valid_indices"].detach().cpu().long().numpy()
    )
    coefficient_dim = int(artifact["metadata"]["coefficient_dim"])
    if len(coefficient_anchor_rows) != coefficient_dim:
        raise RuntimeError("Runtime basis does not expose one anchor per coefficient")
    if not 1 <= int(anchor_count) <= coefficient_dim:
        raise ValueError(
            f"anchor_count must lie in [1,{coefficient_dim}], got {anchor_count}"
        )
    metadata = dict(artifact.get("metadata", {}))
    mesh = Path(str(metadata.get("mesh", ""))).expanduser().resolve(strict=True)
    palm_faces = (
        Path(str(metadata.get("palm_faces", ""))).expanduser().resolve(strict=True)
    )
    tactile_dim = int(metadata.get("tactile_dim", int(valid_indices.max()) + 1))
    valid_mask = np.zeros(tactile_dim, dtype=bool)
    valid_mask[valid_indices] = True
    coordinates, loaded_valid, adjacency = _load_mesh_and_palm_graph(
        mesh, palm_faces, valid_mask
    )
    if not np.array_equal(valid_indices, loaded_valid):
        raise RuntimeError("Routing geometry differs from the runtime surface basis")
    local_coordinates = coordinates[valid_indices]
    router_rows = coefficient_anchor_rows[: int(anchor_count)]
    graph = _adjacency_csr(adjacency, local_coordinates)
    distances = shortest_path(
        graph, directed=False, indices=router_rows, return_predecessors=False
    )
    if distances.shape != (int(anchor_count), len(valid_indices)):
        raise RuntimeError(f"Unexpected routing distance shape {distances.shape}")
    if not np.isfinite(distances).all():
        raise RuntimeError("Canonical routing graph is disconnected")
    coefficient_owner = np.argmin(
        distances[:, coefficient_anchor_rows], axis=0
    ).astype(np.int64)
    # The nested FPS prefix must retain its own canonical anchors.
    expected = np.arange(int(anchor_count), dtype=np.int64)
    if not np.array_equal(coefficient_owner[: int(anchor_count)], expected):
        raise RuntimeError("Nested routing anchors do not own their own coefficients")
    vertex_owner = np.argmin(distances, axis=0).astype(np.int64)
    support_indices = artifact["support_indices"].detach().cpu().long().numpy()
    support_weights = artifact["support_weights"].detach().cpu().float().numpy()
    rows: list[dict[str, Any]] = []
    for anchor in range(int(anchor_count)):
        coefficient_ids = np.flatnonzero(coefficient_owner == anchor)
        coefficient_distances = distances[
            anchor, coefficient_anchor_rows[coefficient_ids]
        ]
        effect_by_vertex = np.zeros(len(valid_indices), dtype=np.float64)
        for vertex in range(len(valid_indices)):
            owned = coefficient_owner[support_indices[vertex]] == anchor
            if owned.any():
                effect_by_vertex[vertex] = float(support_weights[vertex, owned].sum())
        affected = effect_by_vertex > 0.0
        total_effect = float(effect_by_vertex.sum())
        outside = float(effect_by_vertex[vertex_owner != anchor].sum())
        rows.append(
            {
                "routing_anchor": anchor,
                "valid_vertex_row": int(router_rows[anchor]),
                "canonical_vertex_index": int(valid_indices[router_rows[anchor]]),
                "owned_coefficient_count": int(len(coefficient_ids)),
                "owned_coefficient_distance_p90": float(
                    np.quantile(coefficient_distances, 0.9)
                ),
                "owned_coefficient_distance_max": float(coefficient_distances.max()),
                "affected_vertex_count": int(affected.sum()),
                "affected_vertex_fraction": float(affected.mean()),
                "effect_distance_mean": float(
                    (distances[anchor] * effect_by_vertex).sum()
                    / max(total_effect, 1e-12)
                ),
                "effect_distance_p90": _weighted_quantile(
                    distances[anchor, affected], effect_by_vertex[affected], 0.9
                ),
                "voronoi_leakage_fraction": outside / max(total_effect, 1e-12),
            }
        )
    group_sizes = np.bincount(coefficient_owner, minlength=int(anchor_count))
    locality = {
        "anchor_count": int(anchor_count),
        "coefficient_dim": coefficient_dim,
        "owned_coefficient_min": int(group_sizes.min()),
        "owned_coefficient_median": float(np.median(group_sizes)),
        "owned_coefficient_p90": float(np.quantile(group_sizes, 0.9)),
        "owned_coefficient_max": int(group_sizes.max()),
        "affected_vertex_fraction_mean": float(
            np.mean([row["affected_vertex_fraction"] for row in rows])
        ),
        "voronoi_leakage_fraction_mean": float(
            np.mean([row["voronoi_leakage_fraction"] for row in rows])
        ),
    }
    return {
        "coefficient_owner": torch.from_numpy(coefficient_owner),
        "router_rows": torch.from_numpy(router_rows.copy()),
        "router_xyz": _fixed_xyz_fourier(local_coordinates[router_rows]),
        "router_xyz_raw": torch.from_numpy(
            np.asarray(local_coordinates[router_rows], dtype=np.float32)
        ),
        "locality_rows": rows,
        "locality_summary": locality,
        "geometry_sha256": _tensor_sha256(torch.from_numpy(coefficient_owner)),
    }


def _save_geometry_cache(
    path: Path,
    geometry: Mapping[str, Any],
    *,
    basis_path: Path,
    anchor_count: int,
) -> None:
    _atomic_torch_save(
        path,
        {
            "format": f"{SCHEMA}_geometry",
            "surface_basis_sha256": sha256_file(basis_path),
            "anchor_count": int(anchor_count),
            "geometry": dict(geometry),
        },
    )


def _load_geometry_cache(
    path: Path,
    *,
    basis_path: Path,
    anchor_count: int,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != f"{SCHEMA}_geometry":
        raise ValueError(f"Unsupported routing geometry cache: {path}")
    if payload.get("surface_basis_sha256") != sha256_file(basis_path):
        raise RuntimeError("Routing geometry cache was built from another surface basis")
    if int(payload.get("anchor_count", -1)) != int(anchor_count):
        raise RuntimeError("Routing geometry cache has the wrong anchor count")
    geometry = dict(payload["geometry"])
    expected = _tensor_sha256(geometry["coefficient_owner"])
    if geometry.get("geometry_sha256") != expected:
        raise RuntimeError("Routing geometry cache checksum mismatch")
    return geometry


def _resolve_geometry(
    artifact: Mapping[str, Any],
    anchor_count: int,
    cache_path: str | Path | None,
    *,
    basis_path: Path,
) -> dict[str, Any]:
    if cache_path:
        path = Path(cache_path).expanduser().resolve(strict=True)
        return _load_geometry_cache(
            path, basis_path=basis_path, anchor_count=anchor_count
        )
    return _build_routing_geometry(artifact, anchor_count)


class RoutingBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        anchor_count: int,
        dropout: float,
        architecture: str,
    ):
        super().__init__()
        if dimension % heads != 0:
            raise ValueError("Routing dimension must be divisible by the head count")
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.head_dimension = self.dimension // self.heads
        self.architecture = str(architecture)
        if self.architecture not in ROUTING_ARCHITECTURES:
            raise ValueError(f"Unsupported routing architecture {architecture!r}")
        self.anchor_norm = nn.LayerNorm(self.dimension)
        self.token_norm = nn.LayerNorm(self.dimension)
        self.query = nn.Linear(self.dimension, self.dimension)
        self.key = nn.Linear(self.dimension, self.dimension)
        evidence_only = self.architecture == "evidence_only"
        self.value = nn.Linear(
            self.dimension, self.dimension, bias=not evidence_only
        )
        self.output = nn.Linear(
            self.dimension, self.dimension, bias=not evidence_only
        )
        self.null_score = nn.Linear(self.dimension, self.heads)
        if evidence_only:
            self.register_parameter("null_value", None)
        else:
            self.null_value = nn.Parameter(
                torch.zeros(int(anchor_count), self.heads, self.head_dimension)
            )
        self.dropout = nn.Dropout(float(dropout))
        self.ffn_norm = nn.LayerNorm(self.dimension)
        self.ffn = nn.Sequential(
            nn.Linear(self.dimension, 2 * self.dimension),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.dimension, self.dimension),
        )
        nn.init.zeros_(self.null_score.weight)
        nn.init.zeros_(self.null_score.bias)

    def forward(
        self,
        anchors: torch.Tensor,
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
        *,
        routing_mode: str,
        return_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, anchor_count, _ = anchors.shape
        token_count = key_tokens.shape[1]
        query = self.query(self.anchor_norm(anchors)).reshape(
            batch, anchor_count, self.heads, self.head_dimension
        ).permute(0, 2, 1, 3)
        key = self.key(self.token_norm(key_tokens)).reshape(
            batch, token_count, self.heads, self.head_dimension
        ).permute(0, 2, 1, 3)
        value = self.value(value_tokens).reshape(
            batch, token_count, self.heads, self.head_dimension
        ).permute(0, 2, 1, 3)
        if self.architecture == "evidence_only":
            # Preserve the exact identity contract even under finite-precision
            # projection: a spatially constant value field must remain zero.
            value = value - value.mean(dim=2, keepdim=True)
        logits = torch.einsum("bhad,bhtd->bhat", query, key)
        logits = logits.float() / math.sqrt(self.head_dimension)
        if routing_mode == "competitive":
            assignment = torch.softmax(logits, dim=2)
            weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(
                1e-8
            )
            anchor_load = assignment.sum(dim=-1)
        elif routing_mode == "independent":
            weights = torch.softmax(logits, dim=-1)
            anchor_load = weights.sum(dim=-1)
        else:
            raise ValueError(f"Unsupported routing_mode={routing_mode!r}")
        evidence = torch.logsumexp(logits, dim=-1) - math.log(token_count)
        null_score = self.null_score(anchors).permute(0, 2, 1).float()
        visibility = torch.sigmoid(evidence - null_score)
        attended = torch.einsum(
            "bhat,bhtd->bhad", weights.to(dtype=value.dtype), value
        )
        mixed = visibility[..., None].to(dtype=attended.dtype) * attended
        if self.architecture == "legacy":
            assert self.null_value is not None
            null_value = self.null_value.permute(1, 0, 2)[None].to(
                dtype=attended.dtype
            )
            mixed = mixed + (1.0 - visibility[..., None]).to(
                dtype=attended.dtype
            ) * null_value
        mixed = mixed.permute(0, 2, 1, 3).reshape(
            batch, anchor_count, self.dimension
        )
        routed_evidence = self.output(mixed)
        anchors = anchors + self.dropout(routed_evidence)
        anchors = anchors + self.dropout(self.ffn(self.ffn_norm(anchors)))
        diagnostics = {
            "visibility": visibility.mean(dim=1),
            "anchor_load": anchor_load.mean(dim=1),
        }
        if return_weights:
            diagnostics["routing_weights"] = weights.mean(dim=1)
        return anchors, routed_evidence, diagnostics


class CanonicalAnchorRouter(nn.Module):
    def __init__(
        self,
        *,
        grid_size: Sequence[int],
        token_channels: int,
        coefficient_owner: torch.Tensor,
        router_xyz: torch.Tensor,
        support_indices: torch.Tensor,
        support_weights: torch.Tensor,
        dimension: int,
        heads: int,
        layers: int,
        routing_mode: str,
        source: str,
        architecture: str,
        dropout: float,
        max_logit_delta: float,
        seed: int,
    ):
        super().__init__()
        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"Unsupported routing mode {routing_mode!r}")
        if source not in ROUTING_SOURCES:
            raise ValueError(f"Unsupported routing source {source!r}")
        if architecture not in ROUTING_ARCHITECTURES:
            raise ValueError(f"Unsupported routing architecture {architecture!r}")
        self.grid_size = tuple(int(value) for value in grid_size)
        self.token_count = self.grid_size[0] * self.grid_size[1]
        self.token_channels = int(token_channels)
        self.dimension = int(dimension)
        self.anchor_count = int(router_xyz.shape[0])
        self.coefficient_dim = int(coefficient_owner.numel())
        self.routing_mode = str(routing_mode)
        self.source = str(source)
        self.architecture = str(architecture)
        self.max_logit_delta = float(max_logit_delta)
        if self.max_logit_delta <= 0.0:
            raise ValueError("max_logit_delta must be positive")
        self.register_buffer(
            "coefficient_owner", coefficient_owner.detach().long(), persistent=True
        )
        self.register_buffer(
            "support_indices", support_indices.detach().long(), persistent=True
        )
        self.register_buffer(
            "support_weights", support_weights.detach().float(), persistent=True
        )
        self.register_buffer(
            "token_position",
            _fixed_2d_sincos(*self.grid_size, self.dimension),
            persistent=True,
        )
        self.register_buffer("router_xyz", router_xyz.detach().float(), persistent=True)
        generator = torch.Generator().manual_seed(int(seed) + 2903)
        self.register_buffer(
            "shuffle_index",
            torch.randperm(self.token_count, generator=generator),
            persistent=True,
        )
        self.token_norm = nn.LayerNorm(self.token_channels)
        if self.architecture == "evidence_only":
            # Input projections necessarily have different shapes for 32- and
            # 256-channel sources. Isolate their RNG consumption so all shared
            # anchor/router/readout parameters retain identical initialization
            # under the same experiment seed.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed) + 2905)
                self.token_projection = nn.Linear(
                    self.token_channels, self.dimension
                )
                self.value_projection = nn.Linear(
                    self.token_channels, self.dimension, bias=False
                )
        else:
            self.token_projection = nn.Linear(
                self.token_channels, self.dimension
            )
            self.value_projection = None
        self.anchor_embedding = nn.Parameter(
            torch.empty(self.anchor_count, self.dimension)
        )
        nn.init.normal_(self.anchor_embedding, std=0.02)
        self.xyz_projection = nn.Linear(int(router_xyz.shape[1]), self.dimension)
        self.blocks = nn.ModuleList(
            RoutingBlock(
                self.dimension,
                int(heads),
                self.anchor_count,
                float(dropout),
                self.architecture,
            )
            for _ in range(int(layers))
        )
        # Every coefficient has its own scalar readout, but can read only its
        # geodesically assigned routing anchor.
        self.coefficient_weight = nn.Parameter(
            torch.zeros(self.coefficient_dim, self.dimension)
        )
        if self.architecture == "evidence_only":
            self.register_parameter("coefficient_bias", None)
        else:
            self.coefficient_bias = nn.Parameter(torch.zeros(self.coefficient_dim))

    def _tokens(
        self, features: torch.Tensor, control: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.token_channels * self.token_count
        if features.ndim != 2 or features.shape[1] != expected:
            raise ValueError(
                f"Expected FullGrid features [B,{expected}], got {tuple(features.shape)}"
            )
        content = features.reshape(
            features.shape[0], self.token_channels, self.token_count
        ).transpose(1, 2)
        effective = control
        if control == "configured":
            effective = "real" if self.source == "spatial" else "global_repeat"
        if effective == "shuffle_spatial":
            content = content.index_select(1, self.shuffle_index)
        elif effective == "global_repeat":
            content = content.mean(dim=1, keepdim=True).expand(-1, self.token_count, -1)
        elif effective != "real":
            raise ValueError(f"Unsupported token control={control!r}")
        normalized = self.token_norm(content)
        key_tokens = self.token_projection(normalized) + self.token_position
        if self.architecture == "evidence_only":
            assert self.value_projection is not None
            centered = normalized - normalized.mean(dim=1, keepdim=True)
            value_tokens = self.value_projection(centered)
        else:
            value_tokens = key_tokens
        return key_tokens, value_tokens

    def forward(
        self,
        features: torch.Tensor,
        base_prediction: torch.Tensor,
        *,
        control: str = "configured",
        return_routing: bool = False,
    ) -> dict[str, torch.Tensor]:
        key_tokens, value_tokens = self._tokens(features, control)
        anchors = self.anchor_embedding + self.xyz_projection(self.router_xyz)
        anchors = anchors[None].expand(features.shape[0], -1, -1)
        block_diagnostics: dict[str, torch.Tensor] = {}
        routed_evidence = torch.zeros_like(anchors)
        for block in self.blocks:
            anchors, routed_evidence, block_diagnostics = block(
                anchors,
                key_tokens,
                value_tokens,
                routing_mode=self.routing_mode,
                return_weights=return_routing,
            )
        readout_state = (
            routed_evidence if self.architecture == "evidence_only" else anchors
        )
        owned_state = readout_state.index_select(1, self.coefficient_owner)
        raw_coefficients = torch.einsum(
            "bkd,kd->bk", owned_state, self.coefficient_weight
        )
        if self.coefficient_bias is not None:
            raw_coefficients = raw_coefficients + self.coefficient_bias
        coefficient_delta = self.max_logit_delta * torch.tanh(
            raw_coefficients / self.max_logit_delta
        )
        logit_delta = _decode_coefficients(
            coefficient_delta, self.support_indices, self.support_weights
        )
        effective_control = control
        if control == "configured":
            effective_control = (
                "real" if self.source == "spatial" else "global_repeat"
            )
        if (
            self.architecture == "evidence_only"
            and effective_control == "global_repeat"
            and bool((logit_delta.detach().float() != 0.0).any().item())
        ):
            raise RuntimeError(
                "Evidence-only global-repeat control violated its exact identity contract"
            )
        if base_prediction.shape != logit_delta.shape:
            raise ValueError(
                f"Base prediction shape {tuple(base_prediction.shape)} differs from "
                f"routing output {tuple(logit_delta.shape)}"
            )
        # Prepared probabilities are stored as FP16. Use only the minimum
        # finite FP32 clamp so zero residual remains numerically identical to
        # the frozen base except for exact FP16 endpoints.
        epsilon = torch.finfo(torch.float32).eps
        base_logits = torch.logit(
            base_prediction.float().clamp(epsilon, 1.0 - epsilon)
        )
        logits = base_logits + logit_delta.float()
        result = {
            "logits": logits,
            "base_logits": base_logits,
            "logit_delta": logit_delta,
            "coefficient_delta": coefficient_delta,
            "raw_coefficients": raw_coefficients,
            "visibility": block_diagnostics["visibility"],
            "anchor_load": block_diagnostics["anchor_load"],
        }
        if return_routing:
            result["routing_weights"] = block_diagnostics["routing_weights"]
        return result


def _model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "anchor_count": int(args.anchor_count),
        "dimension": int(args.dimension),
        "heads": int(args.heads),
        "layers": int(args.layers),
        "routing_mode": str(args.routing_mode),
        "source": str(args.source),
        "architecture": str(args.architecture),
        "feature_source": str(args.feature_source),
        "dropout": float(args.dropout),
        "max_logit_delta": float(args.max_logit_delta),
        "seed": int(args.seed),
    }


def _build_model(
    metadata: Mapping[str, Any],
    artifact: Mapping[str, Any],
    geometry: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> CanonicalAnchorRouter:
    feature_dim = int(metadata["feature_dim"])
    grid_size = tuple(int(value) for value in metadata["grid_size"])
    token_count = grid_size[0] * grid_size[1]
    if feature_dim % token_count:
        raise RuntimeError("Prepared FullGrid features are not token-factorable")
    return CanonicalAnchorRouter(
        grid_size=grid_size,
        token_channels=feature_dim // token_count,
        coefficient_owner=geometry["coefficient_owner"],
        router_xyz=geometry["router_xyz"],
        support_indices=artifact["support_indices"],
        support_weights=artifact["support_weights"],
        dimension=int(model_config["dimension"]),
        heads=int(model_config["heads"]),
        layers=int(model_config["layers"]),
        routing_mode=str(model_config["routing_mode"]),
        source=str(model_config["source"]),
        architecture=str(model_config.get("architecture", "legacy")),
        dropout=float(model_config["dropout"]),
        max_logit_delta=float(model_config["max_logit_delta"]),
        seed=int(model_config["seed"]),
    )


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.inference_mode()
def _evaluate(
    model: CanonicalAnchorRouter,
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
    control: str,
    bf16: bool,
    collect_attention: bool,
) -> dict[str, Any]:
    model.eval()
    metrics = MetricAccumulator("canonical_anchor_routing")
    base_metrics = MetricAccumulator("frozen_fullgrid_base")
    loss_config = _loss_config()
    valid_count = int(arrays["targets"].shape[1])
    ones = torch.ones(valid_count, device=device)
    pressure_loss_sum = 0.0
    base_loss_sum = 0.0
    sample_count = 0
    diagnostic_sums: dict[str, float] = {}
    attention_sum = torch.zeros(
        model.anchor_count, model.token_count, dtype=torch.float64
    )
    frame_attention_entropy_sum = 0.0
    frame_attention_effective_sum = 0.0
    frame_attention_anchor_cosine_sum = 0.0
    frame_top_token_unique_sum = 0.0
    for start in range(0, len(arrays["features"]), int(batch_size)):
        indices = np.arange(
            start, min(start + int(batch_size), len(arrays["features"])), dtype=np.int64
        )
        features = _batch_array(arrays["features"], indices, device)
        target = _batch_array(arrays["targets"], indices, device)
        base_prediction = _batch_array(arrays["base_predictions"], indices, device)
        with _autocast(device, bf16):
            output = model(
                features,
                base_prediction,
                control=control,
                return_routing=collect_attention,
            )
        prediction = torch.sigmoid(output["logits"].float())
        pressure_loss, _ = compute_tactile_loss(
            prediction,
            output["logits"].float(),
            target,
            ones,
            None,
            ["touchanything"] * len(indices),
            loss_config,
            ramp_override=1.0,
            distributed_reduce=False,
        )
        base_logits = output["base_logits"].float()
        base_loss, _ = compute_tactile_loss(
            base_prediction,
            base_logits,
            target,
            ones,
            None,
            ["touchanything"] * len(indices),
            loss_config,
            ramp_override=1.0,
            distributed_reduce=False,
        )
        count = len(indices)
        sample_count += count
        pressure_loss_sum += float(pressure_loss.item()) * count
        base_loss_sum += float(base_loss.item()) * count
        target_numpy = target.float().cpu().numpy()
        metrics.update(prediction.cpu().numpy(), target_numpy)
        base_metrics.update(base_prediction.float().cpu().numpy(), target_numpy)
        prediction_delta = prediction - base_prediction.float()
        scalar_diagnostics = {
            "coefficient_delta_rms": output["coefficient_delta"].float().square().mean().sqrt(),
            "coefficient_saturation": (
                output["raw_coefficients"].float().abs()
                > 3.0 * model.max_logit_delta
            ).float().mean(),
            "logit_delta_rms": output["logit_delta"].float().square().mean().sqrt(),
            "logit_delta_abs_max": output["logit_delta"].float().abs().amax(),
            "visibility_mean": output["visibility"].float().mean(),
            "null_fraction": (output["visibility"].float() < 0.5).float().mean(),
            "anchor_load_cv": output["anchor_load"].float().std(dim=1).mean()
            / output["anchor_load"].float().mean().clamp_min(1e-8),
            "output_delta_up_volume": prediction_delta.clamp_min(0.0).sum(dim=1).mean(),
            "output_delta_down_volume": (-prediction_delta).clamp_min(0.0).sum(dim=1).mean(),
            "output_delta_net_volume": prediction_delta.sum(dim=1).mean(),
        }
        for key, value in scalar_diagnostics.items():
            diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(
                value.item()
            ) * count
        if collect_attention:
            frame_attention = output["routing_weights"].detach().float()
            frame_normalized = frame_attention / frame_attention.sum(
                dim=2, keepdim=True
            ).clamp_min(1e-12)
            frame_entropy = -(
                frame_normalized * frame_normalized.clamp_min(1e-12).log()
            ).sum(dim=2)
            frame_attention_entropy_sum += float(frame_entropy.sum().item())
            frame_attention_effective_sum += float(frame_entropy.exp().sum().item())
            frame_unit = frame_normalized / frame_normalized.square().sum(
                dim=2, keepdim=True
            ).sqrt().clamp_min(1e-12)
            summed = frame_unit.sum(dim=1)
            pair_sum = summed.square().sum(dim=1) - float(model.anchor_count)
            pair_count = max(model.anchor_count * (model.anchor_count - 1), 1)
            frame_attention_anchor_cosine_sum += float(
                (pair_sum / pair_count).sum().item()
            )
            top_indices = frame_normalized.argmax(dim=2)
            top_presence = torch.zeros(
                top_indices.shape[0],
                model.token_count,
                device=top_indices.device,
                dtype=torch.float32,
            )
            top_presence.scatter_(1, top_indices, 1.0)
            frame_top_token_unique_sum += float(top_presence.sum().item())
            attention_sum += frame_attention.double().sum(dim=0).cpu()
    result = metrics.summary()
    base = base_metrics.summary()
    result.update(
        {
            "pressure_loss": pressure_loss_sum / max(sample_count, 1),
            "base_pressure_loss": base_loss_sum / max(sample_count, 1),
            "base": base,
            "control": control,
            "routing_diagnostics": {
                key: value / max(sample_count, 1)
                for key, value in diagnostic_sums.items()
            },
        }
    )
    if collect_attention:
        attention = attention_sum.float() / max(sample_count, 1)
        normalized = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=1)
        unit = normalized / normalized.square().sum(dim=1, keepdim=True).sqrt().clamp_min(
            1e-12
        )
        similarities = unit @ unit.T
        upper = torch.triu_indices(model.anchor_count, model.anchor_count, offset=1)
        pair_similarity = similarities[upper[0], upper[1]].cpu().numpy()
        xyz = model.router_xyz.detach().float().cpu().numpy()[:, :3]
        pair_distance = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)[
            upper[0].numpy(), upper[1].numpy()
        ]
        correlation = float("nan")
        if np.std(pair_similarity) > 0.0 and np.std(pair_distance) > 0.0:
            correlation = float(np.corrcoef(pair_distance, pair_similarity)[0, 1])
        top_tokens = normalized.argmax(dim=1).cpu().numpy()
        result["routing_diagnostics"].update(
            {
                "attention_frame_normalized_entropy_mean": float(
                    frame_attention_entropy_sum
                    / max(sample_count * model.anchor_count, 1)
                    / math.log(model.token_count)
                ),
                "attention_frame_effective_token_count_mean": float(
                    frame_attention_effective_sum
                    / max(sample_count * model.anchor_count, 1)
                ),
                "attention_frame_anchor_cosine_mean": float(
                    frame_attention_anchor_cosine_sum / max(sample_count, 1)
                ),
                "attention_frame_top_token_unique_mean": float(
                    frame_top_token_unique_sum / max(sample_count, 1)
                ),
                "attention_normalized_entropy_mean": float(
                    (entropy / math.log(model.token_count)).mean().item()
                ),
                "attention_effective_token_count_mean": float(entropy.exp().mean().item()),
                "attention_anchor_cosine_mean": float(np.mean(pair_similarity)),
                "canonical_distance_attention_correlation": correlation,
                "top_token_unique_count": int(np.unique(top_tokens).size),
                "top_token_unique_fraction": float(
                    np.unique(top_tokens).size / model.anchor_count
                ),
            }
        )
    return result


def _training_contract(
    prepared: Mapping[str, tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    basis_path: Path,
    geometry: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "prepared_contract_sha256s": {
            split: str(metadata["contract_sha256"])
            for split, (metadata, _) in prepared.items()
        },
        "routing_feature_contract_sha256s": {
            split: str(
                metadata.get(
                    "routing_feature_contract_sha256", metadata["contract_sha256"]
                )
            )
            for split, (metadata, _) in prepared.items()
        },
        "surface_basis": str(basis_path),
        "surface_basis_sha256": sha256_file(basis_path),
        "geometry_sha256": str(geometry["geometry_sha256"]),
        "model_config": dict(model_config),
        "selection": "official_val_pressure_loss",
        "base_mode": "frozen_fullgrid_logit_plus_local_surface_residual",
    }


def command_prepare_geometry(args: argparse.Namespace) -> None:
    basis_path = Path(args.surface_basis).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().resolve()
    if output.is_file() and not args.force:
        _load_geometry_cache(
            output, basis_path=basis_path, anchor_count=int(args.anchor_count)
        )
        print(f"[canonical-routing] reuse geometry: {output}")
        return
    artifact = _load_runtime_basis(basis_path)
    geometry = _build_routing_geometry(artifact, int(args.anchor_count))
    _save_geometry_cache(
        output,
        geometry,
        basis_path=basis_path,
        anchor_count=int(args.anchor_count),
    )
    _write_csv(output.with_suffix(".locality.csv"), geometry["locality_rows"])
    _write_json(output.with_suffix(".json"), geometry["locality_summary"])
    print(f"[canonical-routing] geometry complete: {output}")


def command_prepare_features(args: argparse.Namespace) -> None:
    prepared_split = Path(args.prepared_split).expanduser().resolve(strict=True)
    feature_cache = Path(args.feature_cache).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata, arrays = _load_prepared_split(prepared_split)
    samples_path = prepared_split / "samples.jsonl"
    rows = _load_sample_rows(samples_path)
    if len(rows) != int(metadata["sample_count"]):
        raise RuntimeError(
            f"Prepared samples/arrays disagree under {prepared_split}: "
            f"{len(rows)} vs {metadata['sample_count']}"
        )
    cache = CacheGroup(feature_cache, ("z_rgb",))
    try:
        first_index = int(rows[0]["cache_index"])
        first = cache[first_index]
        grid_shape = tuple(int(value) for value in np.asarray(first["z_rgb"]).shape)
        if len(grid_shape) != 3 or grid_shape[0] != 256:
            raise ValueError(
                f"Expected cached ReZero grid [256,H,W], got {grid_shape}"
            )
        if tuple(grid_shape[-2:]) != tuple(int(v) for v in metadata["grid_size"]):
            raise RuntimeError("Cached ReZero grid differs from Stage 0.7 grid size")
        contract = {
            "schema": f"{SCHEMA}_rezero_features",
            "base_prepared_contract_sha256": str(metadata["contract_sha256"]),
            "samples_sha256": sha256_file(samples_path),
            "feature_cache": str(feature_cache),
            "cache_config_sha256s": list(cache.config_sha256s),
            "sample_count": len(rows),
            "grid_shape": list(grid_shape),
            "dtype": "float16",
        }
        contract_sha = _contract_sha256(contract)
        done_path = output_dir / "RAW_PREPARED.json"
        destination = output_dir / "features.npy"
        if done_path.is_file() and destination.is_file() and not args.force:
            current = json.loads(done_path.read_text(encoding="utf-8"))
            if current.get("contract_sha256") == contract_sha:
                existing = np.load(destination, mmap_mode="r")
                expected_shape = (len(rows), int(np.prod(grid_shape)))
                if existing.shape != expected_shape:
                    raise RuntimeError(
                        f"Existing ReZero feature shape is {existing.shape}, "
                        f"expected {expected_shape}"
                    )
                print(f"[canonical-routing] reuse ReZero features: {output_dir}")
                return
            raise RuntimeError(
                f"Aligned ReZero feature contract changed under {output_dir}; use --force"
            )
        if args.force:
            done_path.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
        temporary, output = _open_atomic_memmap(
            destination,
            dtype=np.float16,
            shape=(len(rows), int(np.prod(grid_shape))),
        )
        try:
            batch_size = int(args.batch_size)
            for start in range(0, len(rows), batch_size):
                stop = min(start + batch_size, len(rows))
                values: list[np.ndarray] = []
                for row in rows[start:stop]:
                    cache_index = int(row["cache_index"])
                    item = cache[cache_index]
                    expected_id = str(row.get("sample_id", ""))
                    if expected_id and str(item.get("sample_id", "")) != expected_id:
                        raise RuntimeError(
                            "Stage 0.7/cache sample mismatch at cache index "
                            f"{cache_index}: expected={expected_id!r}, "
                            f"actual={item.get('sample_id')!r}"
                        )
                    grid = np.asarray(item["z_rgb"], dtype=np.float32)
                    if grid.shape != grid_shape or not np.isfinite(grid).all():
                        raise RuntimeError(
                            f"Invalid ReZero grid at cache index {cache_index}: {grid.shape}"
                        )
                    values.append(grid.reshape(-1))
                output[start:stop] = np.stack(values).astype(np.float16)
                if stop == len(rows) or stop % max(batch_size * 32, 1) == 0:
                    print(
                        f"[canonical-routing] prepared ReZero {stop:,}/{len(rows):,}",
                        flush=True,
                    )
            _finish_memmap(temporary, output, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        _write_json(
            done_path,
            {
                **contract,
                "contract_sha256": contract_sha,
                "feature_dim": int(np.prod(grid_shape)),
                "output": str(destination),
            },
        )
        # Keep the original arrays live until preparation finishes so their
        # contract cannot disappear halfway through a run.
        del arrays
        print(f"[canonical-routing] ReZero features complete: {output_dir}")
    finally:
        cache.close()


def command_train(args: argparse.Namespace) -> None:
    if args.architecture == "evidence_only" and args.source == "global_control":
        raise ValueError(
            "Evidence-only global_repeat is structurally fixed to the frozen base; "
            "evaluate it as a counterfactual control instead of training it"
        )
    prepared_root = Path(args.prepared_root).expanduser().resolve(strict=True)
    basis_path = Path(args.surface_basis).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = output_dir / "train_summary.json"
    existing_complete = None
    if complete_path.is_file() and not args.force:
        existing_complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if args.force:
        for name in (
            "train_summary.json",
            "summary.json",
            "best_loss.pt",
            "last.pt",
            "metrics.csv",
        ):
            (output_dir / name).unlink(missing_ok=True)

    prepared = _select_feature_source(
        _load_prepared_root(prepared_root),
        feature_source=str(args.feature_source),
        raw_prepared_root=args.raw_prepared_root,
    )
    train_metadata, train_arrays = prepared["train"]
    expected_basis_hash = str(train_metadata["contract"]["surface_basis_sha256"])
    if sha256_file(basis_path) != expected_basis_hash:
        raise RuntimeError("Routing basis differs from the prepared Stage 0.7 splits")
    artifact = _load_runtime_basis(basis_path)
    model_config = _model_config_from_args(args)
    geometry = _resolve_geometry(
        artifact,
        int(args.anchor_count),
        args.geometry_cache,
        basis_path=basis_path,
    )
    contract = _training_contract(
        prepared, basis_path, geometry, model_config
    )
    contract_sha = _contract_sha256(contract)
    if existing_complete is not None:
        if existing_complete.get("contract_sha256") == contract_sha:
            print(f"[canonical-routing] reuse completed training: {output_dir}")
            return
        raise RuntimeError(
            "Completed canonical-routing run has a different configuration; "
            "use --force or a new output directory"
        )
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = _build_model(train_metadata, artifact, geometry, model_config).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    epochs = int(args.epochs)
    warmup = max(0, int(args.warmup_epochs))

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return float(epoch + 1) / warmup
        progress = float(epoch - warmup) / max(epochs - warmup - 1, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    start_epoch = 0
    rng = np.random.default_rng(seed)
    last_path = output_dir / "last.pt"
    if last_path.is_file() and not args.no_resume and not args.force:
        resume = torch.load(last_path, map_location=device)
        if resume.get("contract_sha256") != contract_sha:
            raise RuntimeError("Canonical-routing resume configuration mismatch")
        model.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        rows = list(resume.get("metrics", ()))
        best_loss = float(resume.get("best_loss", float("inf")))
        best_epoch = int(resume.get("best_epoch", -1))
        start_epoch = int(resume["epoch"]) + 1
        rng.bit_generator.state = resume["numpy_rng_state"]
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        if device.type == "cuda" and resume.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resume["cuda_rng_state"].cpu(), device=device)
        print(f"[canonical-routing] resume epoch={start_epoch} from {last_path}")

    loss_config = _loss_config()
    valid_count = int(train_metadata["valid_vertex_count"])
    ones = torch.ones(valid_count, device=device)
    indices = np.arange(len(train_arrays["features"]), dtype=np.int64)
    best_path = output_dir / "best_loss.pt"
    for epoch in range(start_epoch, epochs):
        model.train()
        permutation = rng.permutation(indices)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(permutation), int(args.batch_size)):
            batch_indices = permutation[start : start + int(args.batch_size)]
            features = _batch_array(train_arrays["features"], batch_indices, device)
            target = _batch_array(train_arrays["targets"], batch_indices, device)
            base_prediction = _batch_array(
                train_arrays["base_predictions"], batch_indices, device
            )
            with _autocast(device, bool(args.bf16)):
                output = model(features, base_prediction)
            prediction = torch.sigmoid(output["logits"].float())
            loss, _ = compute_tactile_loss(
                prediction,
                output["logits"].float(),
                target,
                ones,
                None,
                ["touchanything"] * len(batch_indices),
                loss_config,
                ramp_override=1.0,
                distributed_reduce=False,
            )
            if not bool(torch.isfinite(loss).all().item()):
                raise FloatingPointError(
                    f"Non-finite routing loss at epoch={epoch}, start={start}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.gradient_clip)
            )
            if not bool(torch.isfinite(gradient_norm).all().item()):
                raise FloatingPointError(
                    f"Non-finite routing gradient at epoch={epoch}, start={start}"
                )
            optimizer.step()
            loss_sum += float(loss.detach().item()) * len(batch_indices)
            seen += len(batch_indices)
        validation = _evaluate(
            model,
            prepared["val"][1],
            device=device,
            batch_size=int(args.eval_batch_size),
            control="configured",
            bf16=bool(args.bf16),
            collect_attention=False,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_pressure_loss": loss_sum / max(seen, 1),
            **{
                f"val_{key}": value
                for key, value in validation.items()
                if key not in {"base", "routing_diagnostics"}
            },
            **{
                f"val_routing_{key}": value
                for key, value in validation["routing_diagnostics"].items()
            },
        }
        rows.append(row)
        if float(validation["pressure_loss"]) < best_loss:
            best_loss = float(validation["pressure_loss"])
            best_epoch = epoch
            _atomic_torch_save(
                best_path,
                {
                    "format": SCHEMA,
                    "contract": contract,
                    "contract_sha256": contract_sha,
                    "model_config": model_config,
                    "geometry_sha256": geometry["geometry_sha256"],
                    "parameter_count": parameter_count,
                    "epoch": epoch,
                    "val_metrics": validation,
                    "state_dict": model.state_dict(),
                },
            )
        scheduler.step()
        _write_csv(output_dir / "metrics.csv", rows)
        _atomic_torch_save(
            last_path,
            {
                "format": SCHEMA,
                "contract_sha256": contract_sha,
                "model_config": model_config,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "metrics": rows,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                ),
            },
        )
        print(
            f"[canonical-routing] epoch={epoch:03d} "
            f"mode={model.routing_mode} source={model.source} anchors={model.anchor_count} "
            f"val_loss={validation['pressure_loss']:.6f} "
            f"contact={validation['contact_iou_010_frame_macro']:.4f} "
            f"core={validation['core_distribution_viou_frame_macro']:.4f}",
            flush=True,
        )

    if not best_path.is_file():
        raise RuntimeError("Canonical-routing training produced no best checkpoint")
    _write_csv(output_dir / "locality_audit.csv", geometry["locality_rows"])
    summary = {
        "schema": SCHEMA,
        "contract": contract,
        "contract_sha256": contract_sha,
        "model_config": model_config,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "locality_summary": geometry["locality_summary"],
    }
    _write_json(complete_path, summary)
    print(f"[canonical-routing] training complete: {output_dir}")


def command_evaluate(args: argparse.Namespace) -> None:
    prepared_root = Path(args.prepared_root).expanduser().resolve(strict=True)
    basis_path = Path(args.surface_basis).expanduser().resolve(strict=True)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        if current.get("checkpoint_sha256") == sha256_file(checkpoint_path):
            print(f"[canonical-routing] reuse completed evaluation: {summary_path}")
            return
        print(
            "[canonical-routing] checkpoint changed; replacing stale evaluation "
            f"under {output_dir}",
            flush=True,
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != SCHEMA:
        raise ValueError(f"Unsupported routing checkpoint: {checkpoint_path}")
    model_config = dict(checkpoint["model_config"])
    prepared = _select_feature_source(
        _load_prepared_root(prepared_root),
        feature_source=str(model_config.get("feature_source", "projected32")),
        raw_prepared_root=args.raw_prepared_root,
    )
    artifact = _load_runtime_basis(basis_path)
    geometry = _resolve_geometry(
        artifact,
        int(model_config["anchor_count"]),
        args.geometry_cache,
        basis_path=basis_path,
    )
    contract = _training_contract(prepared, basis_path, geometry, model_config)
    if checkpoint.get("contract_sha256") != _contract_sha256(contract):
        raise RuntimeError("Routing checkpoint differs from evaluation inputs")
    device = torch.device(args.device)
    model = _build_model(prepared["train"][0], artifact, geometry, model_config).to(
        device
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    split_metrics: dict[str, Any] = {}
    control_metrics: dict[str, Any] = {}
    for split in SPLITS:
        collect_attention = split != "train"
        split_metrics[split] = _evaluate(
            model,
            prepared[split][1],
            device=device,
            batch_size=int(args.batch_size),
            control="configured",
            bf16=bool(args.bf16),
            collect_attention=collect_attention,
        )
        if model.source == "spatial" and split != "train":
            control_metrics[split] = {
                "real": {**split_metrics[split], "control": "real"}
            }
            for control in ("shuffle_spatial", "global_repeat"):
                control_metrics[split][control] = _evaluate(
                    model,
                    prepared[split][1],
                    device=device,
                    batch_size=int(args.batch_size),
                    control=control,
                    bf16=bool(args.bf16),
                    collect_attention=True,
                )
        print(f"[canonical-routing] evaluated split={split}", flush=True)
    _write_csv(output_dir / "locality_audit.csv", geometry["locality_rows"])
    summary = {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "selection": "official_val_pressure_loss",
        "model_config": model_config,
        "parameter_count": int(checkpoint["parameter_count"]),
        "split_metrics": split_metrics,
        "control_metrics": control_metrics,
        "locality_summary": geometry["locality_summary"],
        "geometry_sha256": geometry["geometry_sha256"],
    }
    _write_json(summary_path, summary)
    print(f"[canonical-routing] evaluation complete: {summary_path}")


def _delta_row(
    variant: str,
    model_config: Mapping[str, Any],
    split: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    base = metrics["base"]
    keys = (
        "rmse_vertex_micro",
        "contact_iou_010_frame_macro",
        "volumetric_iou_frame_macro",
        "distribution_viou_frame_macro",
        "core_distribution_viou_frame_macro",
        "false_high_excess_mean",
        "gt_ge_070_mean_prediction",
    )
    row: dict[str, Any] = {
        "variant": variant,
        **dict(model_config),
        "split": split,
        "pressure_loss": metrics["pressure_loss"],
        "base_pressure_loss": metrics["base_pressure_loss"],
        "delta_pressure_loss_vs_base": metrics["pressure_loss"]
        - metrics["base_pressure_loss"],
    }
    for key in keys:
        row[key] = metrics[key]
        row[f"delta_{key}_vs_base"] = metrics[key] - base[key]
    row.update(
        {
            f"routing_{key}": value
            for key, value in metrics["routing_diagnostics"].items()
        }
    )
    return row


def command_aggregate(args: argparse.Namespace) -> None:
    root = Path(args.input_root).expanduser().resolve(strict=True)
    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema") == SCHEMA:
            summaries[path.parent.name] = summary
    if not summaries:
        raise RuntimeError(f"No completed routing evaluations under {root}")
    parameter_counts: dict[tuple[Any, ...], set[int]] = {}
    for summary in summaries.values():
        config = summary["model_config"]
        group = (
            int(config["anchor_count"]),
            str(config.get("architecture", "legacy")),
            str(config.get("feature_source", "projected32")),
            int(config["dimension"]),
            int(config["heads"]),
            int(config["layers"]),
        )
        parameter_counts.setdefault(group, set()).add(
            int(summary["parameter_count"])
        )
    mismatched = {
        str(group): sorted(counts)
        for group, counts in parameter_counts.items()
        if len(counts) != 1
    }
    if mismatched:
        raise RuntimeError(
            "Routing controls are not parameter matched within one model family: "
            f"{mismatched}"
        )
    comparison_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    locality_rows: list[dict[str, Any]] = []
    for variant, summary in summaries.items():
        config = summary["model_config"]
        for split, metrics in summary["split_metrics"].items():
            comparison_rows.append(_delta_row(variant, config, split, metrics))
        for split, controls in summary.get("control_metrics", {}).items():
            real = controls["real"]
            for control, metrics in controls.items():
                row = _delta_row(variant, config, split, metrics)
                row["control"] = control
                row["delta_contact_vs_real"] = (
                    metrics["contact_iou_010_frame_macro"]
                    - real["contact_iou_010_frame_macro"]
                )
                row["delta_core_vs_real"] = (
                    metrics["core_distribution_viou_frame_macro"]
                    - real["core_distribution_viou_frame_macro"]
                )
                control_rows.append(row)
        locality_rows.append(
            {
                "variant": variant,
                **dict(config),
                **dict(summary["locality_summary"]),
            }
        )
    _write_csv(root / "comparison.csv", comparison_rows)
    _write_csv(root / "control_comparison.csv", control_rows)
    _write_csv(root / "locality_comparison.csv", locality_rows)

    def seed_aggregate(
        rows: Sequence[Mapping[str, Any]], *, include_control: bool
    ) -> list[dict[str, Any]]:
        group_fields = (
            "anchor_count",
            "architecture",
            "feature_source",
            "dimension",
            "heads",
            "layers",
            "routing_mode",
            "source",
            "split",
        ) + (("control",) if include_control else ())
        metric_fields = (
            "delta_pressure_loss_vs_base",
            "delta_rmse_vertex_micro_vs_base",
            "delta_contact_iou_010_frame_macro_vs_base",
            "delta_distribution_viou_frame_macro_vs_base",
            "delta_core_distribution_viou_frame_macro_vs_base",
            "delta_false_high_excess_mean_vs_base",
            "delta_gt_ge_070_mean_prediction_vs_base",
        ) + (
            ("delta_contact_vs_real", "delta_core_vs_real")
            if include_control
            else ()
        )
        grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for row in rows:
            defaults = {
                "architecture": "legacy",
                "feature_source": "projected32",
            }
            key = tuple(row.get(field, defaults.get(field, "")) for field in group_fields)
            grouped.setdefault(key, []).append(row)
        output: list[dict[str, Any]] = []
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
            row = dict(zip(group_fields, key))
            seeds = sorted({int(value.get("seed", 0)) for value in values})
            row["seed_count"] = len(seeds)
            row["seeds"] = ",".join(str(seed) for seed in seeds)
            for field in metric_fields:
                observed = np.asarray(
                    [float(value[field]) for value in values], dtype=np.float64
                )
                row[f"{field}_mean"] = float(observed.mean())
                row[f"{field}_std"] = float(
                    observed.std(ddof=1) if len(observed) > 1 else 0.0
                )
            output.append(row)
        return output

    seed_rows = seed_aggregate(comparison_rows, include_control=False)
    seed_control_rows = seed_aggregate(control_rows, include_control=True)
    _write_csv(root / "seed_comparison.csv", seed_rows)
    _write_csv(root / "seed_control_comparison.csv", seed_control_rows)
    _write_json(
        root / "interpretation.json",
        {
            "schema": SCHEMA,
            "formal_selection": "official val pressure loss",
            "questions": {
                "spatial_vs_global": "Does real image-token content outperform a parameter-identical global-repeat control?",
                "real_vs_shuffle": "Does destroying content-position alignment remove the spatial model's benefit?",
                "competitive_vs_independent": "Does token competition prevent canonical anchors from reading nearly identical global evidence?",
                "anchor_count": "Does 512-anchor routing improve locality enough to justify its extra state?",
                "feature_source": "Does the pre-projection 256-channel ReZero grid retain correspondence that FullGrid32 discarded?",
                "strict_identity": "Does evidence-only routing remain useful after query, null, bias, and global-repeat bypasses are removed?",
            },
            "warning": "This is a frozen-grid routing probe, not a formal FullGrid replacement until end-to-end confirmation.",
        },
    )
    _write_json(
        root / "AUDIT_DONE.json",
        {
            "schema": SCHEMA,
            "variants": sorted(summaries),
            "comparison_sha256": sha256_file(root / "comparison.csv"),
            "control_comparison_sha256": sha256_file(
                root / "control_comparison.csv"
            ),
            "locality_comparison_sha256": sha256_file(
                root / "locality_comparison.csv"
            ),
            "seed_comparison_sha256": sha256_file(root / "seed_comparison.csv"),
            "seed_control_comparison_sha256": sha256_file(
                root / "seed_control_comparison.csv"
            ),
        },
    )
    print(f"[canonical-routing] aggregate complete: {root}")


def command_self_test(_args: argparse.Namespace) -> None:
    coefficient_owner = torch.tensor([0, 0, 1, 1])
    support_indices = torch.tensor([[0, 1], [2, 3], [1, 3]])
    support_weights = torch.tensor([[0.5, 0.5], [0.25, 0.75], [0.6, 0.4]])
    xyz = _fixed_xyz_fourier(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    )
    for mode in ROUTING_MODES:
        for architecture in ROUTING_ARCHITECTURES:
            model = CanonicalAnchorRouter(
                grid_size=(2, 2),
                token_channels=3,
                coefficient_owner=coefficient_owner,
                router_xyz=xyz,
                support_indices=support_indices,
                support_weights=support_weights,
                dimension=16,
                heads=4,
                layers=2,
                routing_mode=mode,
                source="spatial",
                architecture=architecture,
                dropout=0.0,
                max_logit_delta=2.0,
                seed=521,
            )
            features = torch.randn(5, 12)
            base = torch.rand(5, 3).clamp(0.01, 0.99)
            output = model(features, base, return_routing=True)
            if output["logits"].shape != (5, 3):
                raise AssertionError("Routing output has the wrong shape")
            if not torch.allclose(
                torch.sigmoid(output["logits"]), base, atol=1e-6, rtol=1e-6
            ):
                raise AssertionError("Zero-initialized routing is not the exact base")
            if output["routing_weights"].shape != (5, 2, 4):
                raise AssertionError("Routing attention has the wrong shape")
            loss = model(features, base)["logits"].square().mean()
            loss.backward()
            if model.coefficient_weight.grad is None or not torch.isfinite(
                model.coefficient_weight.grad
            ).all():
                raise AssertionError("Routing coefficient head lacks finite gradients")
            global_output = model(features, base, control="global_repeat")
            shuffled_output = model(features, base, control="shuffle_spatial")
            if global_output["logits"].shape != shuffled_output["logits"].shape:
                raise AssertionError("Routing controls disagree on output shape")
            if architecture == "evidence_only":
                with torch.no_grad():
                    model.coefficient_weight.normal_(std=0.1)
                spatial_output = model(features, base, control="real")
                global_output = model(features, base, control="global_repeat")
                if not torch.equal(
                    global_output["logit_delta"],
                    torch.zeros_like(global_output["logit_delta"]),
                ):
                    raise AssertionError(
                        "Evidence-only global-repeat control is not exact identity"
                    )
                if bool((spatial_output["logit_delta"] == 0.0).all().item()):
                    raise AssertionError(
                        "Evidence-only spatial route ignores image values"
                    )
    shared_models = []
    for channels in (3, 7):
        torch.manual_seed(521)
        shared_models.append(
            CanonicalAnchorRouter(
                grid_size=(2, 2),
                token_channels=channels,
                coefficient_owner=coefficient_owner,
                router_xyz=xyz,
                support_indices=support_indices,
                support_weights=support_weights,
                dimension=16,
                heads=4,
                layers=2,
                routing_mode="competitive",
                source="spatial",
                architecture="evidence_only",
                dropout=0.0,
                max_logit_delta=2.0,
                seed=521,
            )
        )
    source_specific = (
        "token_norm.",
        "token_projection.",
        "value_projection.",
    )
    first_state = shared_models[0].state_dict()
    second_state = shared_models[1].state_dict()
    for name, first_value in first_state.items():
        if name.startswith(source_specific):
            continue
        second_value = second_state[name]
        if not torch.equal(first_value, second_value):
            raise AssertionError(
                f"Shared evidence-only initialization depends on token channels: {name}"
            )
    print("canonical anchor routing self-test: OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    geometry = subparsers.add_parser("prepare-geometry")
    geometry.add_argument("--surface-basis", required=True)
    geometry.add_argument("--anchor-count", type=int, choices=(256, 512), required=True)
    geometry.add_argument("--output", required=True)
    geometry.add_argument("--force", action="store_true")
    geometry.set_defaults(func=command_prepare_geometry)

    features = subparsers.add_parser("prepare-features")
    features.add_argument("--prepared-split", required=True)
    features.add_argument("--feature-cache", required=True)
    features.add_argument("--output-dir", required=True)
    features.add_argument("--batch-size", type=int, default=256)
    features.add_argument("--force", action="store_true")
    features.set_defaults(func=command_prepare_features)

    train = subparsers.add_parser("train")
    train.add_argument("--prepared-root", required=True)
    train.add_argument("--surface-basis", required=True)
    train.add_argument("--geometry-cache", default="")
    train.add_argument("--raw-prepared-root", default="")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--anchor-count", type=int, choices=(256, 512), required=True)
    train.add_argument("--dimension", type=int, default=128)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--routing-mode", choices=ROUTING_MODES, required=True)
    train.add_argument("--source", choices=ROUTING_SOURCES, required=True)
    train.add_argument(
        "--architecture", choices=ROUTING_ARCHITECTURES, default="legacy"
    )
    train.add_argument(
        "--feature-source", choices=ROUTING_FEATURE_SOURCES, default="projected32"
    )
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--max-logit-delta", type=float, default=2.0)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--eval-batch-size", type=int, default=1024)
    train.add_argument("--lr", type=float, default=4e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--warmup-epochs", type=int, default=3)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=521)
    train.add_argument("--bf16", action="store_true")
    train.add_argument("--no-resume", action="store_true")
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prepared-root", required=True)
    evaluate.add_argument("--surface-basis", required=True)
    evaluate.add_argument("--geometry-cache", default="")
    evaluate.add_argument("--raw-prepared-root", default="")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch-size", type=int, default=1024)
    evaluate.add_argument("--bf16", action="store_true")
    evaluate.add_argument("--force", action="store_true")
    evaluate.set_defaults(func=command_evaluate)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input-root", required=True)
    aggregate.set_defaults(func=command_aggregate)

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(func=command_self_test)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "dimension",
        "heads",
        "layers",
        "epochs",
        "batch_size",
        "eval_batch_size",
    ):
        if hasattr(args, name) and int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "dimension") and int(args.dimension) % int(args.heads):
        raise ValueError("--dimension must be divisible by --heads")
    if hasattr(args, "dropout") and not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError("--dropout must lie in [0,1)")
    if hasattr(args, "max_logit_delta") and float(args.max_logit_delta) <= 0.0:
        raise ValueError("--max-logit-delta must be positive")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
