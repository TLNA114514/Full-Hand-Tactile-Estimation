"""Frozen HaMeR spatial features and explicit patch-lattice utilities.

This module intentionally loads only the HaMeR image backbone. MANO, camera,
joint, and mesh outputs are outside the tactile feature-routing contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
HAMER_VIT_SOURCE = REPO_ROOT / "hamer" / "hamer" / "models" / "backbones" / "vit.py"
DEFAULT_HAMER_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "hamer_model_config.yaml"
)
HAMER_INPUT_RESOLUTION = (256, 192)
HAMER_FEATURE_LAYERS = (16, 24, 32)


def sha256_file(path: str | Path, chunk_size: int = 16 << 20) -> str:
    resolved = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_hamer_config_path(
    checkpoint_path: str | Path,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path and str(explicit_path).strip():
        return Path(explicit_path).expanduser().resolve(strict=True)
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    adjacent = checkpoint.parent.parent / "model_config.yaml"
    if adjacent.is_file():
        return adjacent.resolve()
    if DEFAULT_HAMER_CONFIG.is_file():
        return DEFAULT_HAMER_CONFIG.resolve()
    raise FileNotFoundError(
        "Could not find HaMeR model_config.yaml beside the checkpoint or at "
        f"the bundled fallback {DEFAULT_HAMER_CONFIG}"
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise TypeError(f"HaMeR config must be a mapping: {path}")
    return dict(value)


def validate_hamer_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    config = _load_yaml(resolved)
    model = config.get("MODEL")
    if not isinstance(model, Mapping):
        raise KeyError("HaMeR config is missing MODEL")
    backbone = model.get("BACKBONE")
    if not isinstance(backbone, Mapping) or str(backbone.get("TYPE", "")) != "vit":
        raise ValueError("HaMeR feature routing requires MODEL.BACKBONE.TYPE=vit")
    image_size = int(model.get("IMAGE_SIZE", 0))
    if image_size != HAMER_INPUT_RESOLUTION[0]:
        raise ValueError(
            f"Expected HaMeR IMAGE_SIZE=256, got {image_size} in {resolved}"
        )
    mean = tuple(float(value) for value in model.get("IMAGE_MEAN", ()))
    std = tuple(float(value) for value in model.get("IMAGE_STD", ()))
    expected_mean = (0.485, 0.456, 0.406)
    expected_std = (0.229, 0.224, 0.225)
    if mean != expected_mean or std != expected_std:
        raise ValueError(
            "HaMeR and tactile preprocessing must share ImageNet normalization; "
            f"got mean={mean}, std={std}"
        )
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "image_size": image_size,
        "input_resolution": list(HAMER_INPUT_RESOLUTION),
        "image_mean": list(mean),
        "image_std": list(std),
        "backbone_type": "vit",
    }


_VIT_MODULE: ModuleType | None = None


def _load_vit_source_module() -> ModuleType:
    """Load the backbone source without importing HaMeR's MANO model package."""

    global _VIT_MODULE
    if _VIT_MODULE is not None:
        return _VIT_MODULE
    if not HAMER_VIT_SOURCE.is_file():
        raise FileNotFoundError(f"HaMeR ViT source is missing: {HAMER_VIT_SOURCE}")
    module_name = "_tactile_hamer_backbone_vit"
    specification = importlib.util.spec_from_file_location(module_name, HAMER_VIT_SOURCE)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load HaMeR ViT source: {HAMER_VIT_SOURCE}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    _VIT_MODULE = module
    return module


def build_hamer_backbone() -> nn.Module:
    module = _load_vit_source_module()
    backbone = module.vit(None)
    if len(backbone.blocks) != 32 or int(backbone.embed_dim) != 1280:
        raise RuntimeError(
            "Unexpected HaMeR backbone architecture: "
            f"depth={len(backbone.blocks)}, width={backbone.embed_dim}"
        )
    convolution = backbone.patch_embed.proj
    if tuple(convolution.kernel_size) != (16, 16):
        raise RuntimeError(
            f"Expected HaMeR patch kernel 16x16, got {convolution.kernel_size}"
        )
    return backbone


def _load_torch_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise TypeError(f"HaMeR checkpoint is not a mapping: {path}")
    return value


def _checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    candidate = checkpoint.get("state_dict", checkpoint)
    if not isinstance(candidate, Mapping):
        raise TypeError("HaMeR checkpoint state_dict is not a mapping")
    state = {
        str(key): value
        for key, value in candidate.items()
        if isinstance(value, torch.Tensor)
    }
    if not state:
        raise ValueError("HaMeR checkpoint contains no tensor state")
    return state


def _select_backbone_prefix(
    state: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]
) -> tuple[str, dict[str, torch.Tensor], dict[str, Any]]:
    expected_keys = set(expected)
    prefixes = {
        "",
        "backbone.",
        "model.backbone.",
        "module.backbone.",
        "_forward_module.backbone.",
    }
    marker_keys = (
        "patch_embed.proj.weight",
        "pos_embed",
        "blocks.0.norm1.weight",
    )
    for raw_key in state:
        for marker in marker_keys:
            if raw_key.endswith(marker):
                prefixes.add(raw_key[: -len(marker)])

    candidates = []
    for prefix in sorted(prefixes):
        stripped = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        overlap = expected_keys.intersection(stripped)
        candidates.append((len(overlap), -len(stripped), prefix, stripped))
    overlap_count, _, prefix, stripped = max(candidates, key=lambda item: item[:3])
    selected = {key: stripped[key] for key in expected_keys.intersection(stripped)}
    missing = sorted(expected_keys - set(selected))
    unexpected = sorted(set(stripped) - expected_keys)
    shape_mismatches = []
    for key in sorted(expected_keys.intersection(selected)):
        actual_shape = tuple(selected[key].shape)
        expected_shape = tuple(expected[key].shape)
        if actual_shape != expected_shape:
            shape_mismatches.append(
                {"key": key, "checkpoint": list(actual_shape), "model": list(expected_shape)}
            )
    report = {
        "selected_prefix": prefix,
        "checkpoint_tensor_count": len(state),
        "prefixed_tensor_count": len(stripped),
        "expected_backbone_tensor_count": len(expected_keys),
        "matched_backbone_tensor_count": overlap_count,
        "missing_keys": missing,
        "unexpected_prefixed_keys": unexpected,
        "shape_mismatches": shape_mismatches,
    }
    return prefix, selected, report


class FrozenHamerSpatialExtractor(nn.Module):
    """A frozen rectangular HaMeR backbone with normalized intermediate maps."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.train(False)

    @property
    def depth(self) -> int:
        return len(self.backbone.blocks)

    def train(self, mode: bool = True):
        del mode
        super().train(False)
        self.backbone.eval()
        return self

    def forward(
        self,
        image: torch.Tensor,
        layers: Sequence[int] = HAMER_FEATURE_LAYERS,
    ) -> tuple[torch.Tensor, ...]:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, *HAMER_INPUT_RESOLUTION):
            raise ValueError(
                "HaMeR spatial extractor expects [B,3,256,192], got "
                f"{tuple(image.shape)}"
            )
        requested = tuple(int(layer) for layer in layers)
        if not requested or tuple(sorted(set(requested))) != requested:
            raise ValueError("HaMeR layers must be unique and increasing")
        if min(requested) < 1 or max(requested) > self.depth:
            raise ValueError(f"HaMeR layers must lie in [1,{self.depth}], got {requested}")
        maps = self.backbone(image, return_intermediate_layers=requested)
        normalized = []
        for layer, feature_map in zip(requested, maps):
            if tuple(feature_map.shape[1:]) != (1280, 16, 12):
                raise RuntimeError(
                    f"HaMeR layer {layer} returned {tuple(feature_map.shape)}, "
                    "expected [B,1280,16,12]"
                )
            if layer != self.depth:
                tokens = feature_map.flatten(2).transpose(1, 2)
                tokens = self.backbone.last_norm(tokens)
                feature_map = tokens.transpose(1, 2).reshape_as(feature_map).contiguous()
            normalized.append(feature_map)
        return tuple(normalized)


def load_frozen_hamer_spatial_extractor(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    minimum_checkpoint_bytes: int = 100_000_000,
) -> tuple[FrozenHamerSpatialExtractor, dict[str, Any]]:
    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    size = checkpoint.stat().st_size
    if size < int(minimum_checkpoint_bytes):
        raise RuntimeError(
            f"HaMeR checkpoint looks incomplete: {checkpoint} has {size:,} bytes, "
            f"minimum is {int(minimum_checkpoint_bytes):,}"
        )
    resolved_config = resolve_hamer_config_path(checkpoint, config_path)
    config_report = validate_hamer_config(resolved_config)
    backbone = build_hamer_backbone()
    expected = backbone.state_dict()
    payload = _load_torch_mapping(checkpoint)
    state = _checkpoint_state(payload)
    _, selected, load_report = _select_backbone_prefix(state, expected)
    if (
        load_report["missing_keys"]
        or load_report["unexpected_prefixed_keys"]
        or load_report["shape_mismatches"]
    ):
        raise RuntimeError(
            "HaMeR backbone checkpoint is not a strict architecture match: "
            f"missing={load_report['missing_keys'][:5]}, "
            f"unexpected={load_report['unexpected_prefixed_keys'][:5]}, "
            f"shape_mismatches={load_report['shape_mismatches'][:3]}"
        )
    backbone.load_state_dict(selected, strict=True)
    del payload, state, selected, expected
    gc.collect()
    extractor = FrozenHamerSpatialExtractor(backbone)
    report = {
        "schema": "frozen_hamer_spatial_extractor_v1",
        "checkpoint_path": str(checkpoint),
        "checkpoint_size_bytes": size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": config_report,
        "input_resolution": list(HAMER_INPUT_RESOLUTION),
        "output_grid": [16, 12],
        "feature_channels": 1280,
        "intermediate_normalization": "backbone.last_norm",
        "pose_outputs_loaded": False,
        "pose_outputs_consumed": False,
        "load": load_report,
    }
    return extractor, report


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, Sequence):
        return int(value[0]), int(value[1])
    return int(value), int(value)


@dataclass(frozen=True)
class PatchLattice:
    name: str
    input_hw: tuple[int, int]
    output_hw: tuple[int, int]
    kernel_hw: tuple[int, int]
    stride_hw: tuple[int, int]
    padding_hw: tuple[int, int]
    dilation_hw: tuple[int, int]
    extra_pad_bottom_right: tuple[int, int]
    center0_yx: tuple[float, float]
    centers_y: tuple[float, ...]
    centers_x: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def patch_lattice_from_conv(
    name: str,
    convolution: nn.Conv2d,
    input_hw: Sequence[int],
    *,
    extra_pad_bottom_right: Sequence[int] = (0, 0),
) -> PatchLattice:
    input_height, input_width = _pair(input_hw)
    kernel_height, kernel_width = _pair(convolution.kernel_size)
    stride_height, stride_width = _pair(convolution.stride)
    padding_height, padding_width = _pair(convolution.padding)
    dilation_height, dilation_width = _pair(convolution.dilation)
    extra_height, extra_width = _pair(extra_pad_bottom_right)
    effective_height = input_height + extra_height
    effective_width = input_width + extra_width
    output_height = (
        effective_height
        + 2 * padding_height
        - dilation_height * (kernel_height - 1)
        - 1
    ) // stride_height + 1
    output_width = (
        effective_width
        + 2 * padding_width
        - dilation_width * (kernel_width - 1)
        - 1
    ) // stride_width + 1
    center_y = -padding_height + dilation_height * (kernel_height - 1) / 2.0
    center_x = -padding_width + dilation_width * (kernel_width - 1) / 2.0
    centers_y = tuple(center_y + index * stride_height for index in range(output_height))
    centers_x = tuple(center_x + index * stride_width for index in range(output_width))
    return PatchLattice(
        name=str(name),
        input_hw=(input_height, input_width),
        output_hw=(output_height, output_width),
        kernel_hw=(kernel_height, kernel_width),
        stride_hw=(stride_height, stride_width),
        padding_hw=(padding_height, padding_width),
        dilation_hw=(dilation_height, dilation_width),
        extra_pad_bottom_right=(extra_height, extra_width),
        center0_yx=(center_y, center_x),
        centers_y=centers_y,
        centers_x=centers_x,
    )


def patch_embed_lattice(name: str, patch_embed: nn.Module, input_hw: Sequence[int]) -> PatchLattice:
    convolution = getattr(patch_embed, "proj", None)
    if not isinstance(convolution, nn.Conv2d):
        raise TypeError(f"{name} patch_embed does not expose a Conv2d projection")
    input_height, input_width = _pair(input_hw)
    extra_height = 0
    extra_width = 0
    if bool(getattr(patch_embed, "dynamic_img_pad", False)):
        patch_height, patch_width = _pair(getattr(patch_embed, "patch_size"))
        extra_height = (patch_height - input_height % patch_height) % patch_height
        extra_width = (patch_width - input_width % patch_width) % patch_width
    return patch_lattice_from_conv(
        name,
        convolution,
        (input_height, input_width),
        extra_pad_bottom_right=(extra_height, extra_width),
    )


def lattice_alignment(source: PatchLattice, target: PatchLattice) -> dict[str, Any]:
    if source.input_hw != target.input_hw:
        raise ValueError("Source and target lattices must describe the same input pixels")
    source_stride_y, source_stride_x = source.stride_hw
    target_stride_y, target_stride_x = target.stride_hw
    offset_y = (target.center0_yx[0] - source.center0_yx[0]) / source_stride_y
    offset_x = (target.center0_yx[1] - source.center0_yx[1]) / source_stride_x
    scale_y = target_stride_y / source_stride_y
    scale_x = target_stride_x / source_stride_x
    source_indices_y = tuple(offset_y + index * scale_y for index in range(target.output_hw[0]))
    source_indices_x = tuple(offset_x + index * scale_x for index in range(target.output_hw[1]))
    outside_y = sum(not (0.0 <= value <= source.output_hw[0] - 1) for value in source_indices_y)
    outside_x = sum(not (0.0 <= value <= source.output_hw[1] - 1) for value in source_indices_x)
    native_deltas_y = [
        target.centers_y[index] - source.centers_y[index]
        for index in range(min(len(source.centers_y), len(target.centers_y)))
    ]
    native_deltas_x = [
        target.centers_x[index] - source.centers_x[index]
        for index in range(min(len(source.centers_x), len(target.centers_x)))
    ]
    return {
        "schema": "patch_lattice_alignment_v1",
        "source": source.name,
        "target": target.name,
        "native_index_center_delta_y_pixels": native_deltas_y,
        "native_index_center_delta_x_pixels": native_deltas_x,
        "native_index_max_abs_center_delta_pixels": max(
            [abs(value) for value in (*native_deltas_y, *native_deltas_x)],
            default=0.0,
        ),
        "target_to_source_index": {
            "offset_y": offset_y,
            "offset_x": offset_x,
            "scale_y": scale_y,
            "scale_x": scale_x,
        },
        "target_rows_outside_source_center_range": outside_y,
        "target_columns_outside_source_center_range": outside_x,
        "fixed_resampling": {
            "mode": "bilinear",
            "padding_mode": "border",
            "align_corners": False,
        },
    }


def resample_feature_lattice(
    feature: torch.Tensor,
    source: PatchLattice,
    target: PatchLattice,
    *,
    padding_mode: str = "border",
) -> torch.Tensor:
    if feature.ndim != 4 or tuple(feature.shape[-2:]) != source.output_hw:
        raise ValueError(
            f"Expected source feature [B,C,{source.output_hw[0]},{source.output_hw[1]}], "
            f"got {tuple(feature.shape)}"
        )
    alignment = lattice_alignment(source, target)["target_to_source_index"]
    source_y = torch.arange(target.output_hw[0], device=feature.device, dtype=torch.float32)
    source_x = torch.arange(target.output_hw[1], device=feature.device, dtype=torch.float32)
    source_y = alignment["offset_y"] + source_y * alignment["scale_y"]
    source_x = alignment["offset_x"] + source_x * alignment["scale_x"]
    normalized_y = 2.0 * (source_y + 0.5) / source.output_hw[0] - 1.0
    normalized_x = 2.0 * (source_x + 0.5) / source.output_hw[1] - 1.0
    grid_y, grid_x = torch.meshgrid(normalized_y, normalized_x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(feature.shape[0], -1, -1, -1)
    with torch.autocast(device_type=feature.device.type, enabled=False):
        sampled = F.grid_sample(
            feature.float(),
            grid,
            mode="bilinear",
            padding_mode=str(padding_mode),
            align_corners=False,
        )
    return sampled.to(dtype=feature.dtype)
