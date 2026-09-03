"""Shared runtime utilities for feature-level tactile input priors.

This module is intentionally the only bridge to ``hamer_tactile_ft``.  New
prior adapters, checkpoints, training, and evaluation remain owned by
``tactile_input_priors`` while the mature tactile model is treated as a frozen
dependency.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "hamer_tactile_ft", REPO_ROOT / "hamer"):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from hamer.configs import get_config  # noqa: E402
from hamer_tactile_ft.dataset import OpenTouchTactileDataset  # noqa: E402
from hamer_tactile_ft.hamer_tactile import (  # noqa: E402
    CANONICAL_MODEL_INITIALIZATION_ORDER,
    DinoTactileModel,
)
from hamer_tactile_ft.hamer_config_assets import (  # noqa: E402
    resolve_hamer_model_config_path,
)
from hamer_tactile_ft.losses import TactileLossConfig  # noqa: E402
from hamer_tactile_ft.data.indexing import persistent_sha256_file  # noqa: E402

from tactile_input_priors.feature_cache import FeatureCacheDataset  # noqa: E402
from tactile_input_priors.prior_model import FrozenBasePriorModel  # noqa: E402
from tactile_input_priors.selector_prior_model import PriorSelectorModel  # noqa: E402


PRIOR_CHECKPOINT_FORMAT = "tactile_prior_adapter_v1"
SELECTOR_PRIOR_CHECKPOINT_FORMAT = "tactile_prior_selector_v1"
DATASET_ALIASES = {
    "opentouch": "OpenTouch",
    "ot": "OpenTouch",
    "touchanything": "TouchAnything",
    "ta": "TouchAnything",
    "egotouch": "TouchAnything",
}


def parse_csv(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        parts = value
    else:
        parts = str(value).split(",")
    return tuple(str(item).strip() for item in parts if str(item).strip())


def parse_resolution(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        height, width = (int(item) for item in value)
    else:
        parts = str(value).lower().replace("x", ",").split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected HEIGHTxWIDTH, got {value!r}")
        height, width = (int(item) for item in parts)
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise ValueError("Input height and width must be positive multiples of 16")
    return height, width


def canonical_datasets(value: Any) -> tuple[str, ...]:
    result = []
    for raw in parse_csv(value):
        canonical = DATASET_ALIASES.get(raw.casefold())
        if canonical is None:
            raise ValueError(f"Unsupported dataset {raw!r}")
        if canonical not in result:
            result.append(canonical)
    if not result:
        raise ValueError("At least one dataset is required")
    return tuple(result)


def file_sha256(path: os.PathLike[str] | str) -> str:
    cache_dir = os.environ.get(
        "TACTILE_PRIOR_HASH_CACHE",
        "/home/ma-user/work/cfzhao/input_prior_full/state",
    )
    return persistent_sha256_file(
        Path(path).expanduser().resolve(strict=True), cache_dir=cache_dir
    )


def atomic_torch_save(payload: Mapping[str, Any], path: os.PathLike[str] | str) -> None:
    destination = Path(path).expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_torch_checkpoint(path: os.PathLike[str] | str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a mapping: {resolved}")
    return checkpoint


def _checkpoint_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    nested = checkpoint.get("model_config")
    result = dict(nested) if isinstance(nested, Mapping) else {}
    for key in (
        "tactile_head_type",
        "visual_backbone",
        "backbone_feature_layers",
        "dino_residual_max_scale",
        "dino_residual_rms_budget",
        "pool_layout",
        "input_resolution",
        "pool_output_channels",
        "decoder_hidden_dim",
        "decoder_dropout_scale",
        "model_initialization_order",
        "bbox_rescale_factor",
        "bbox_source_policy",
        "dataset_filter",
        "local_anchor_count",
        "local_anchor_neighbors",
        "support_selector_mode",
        "support_selector_thresholds",
        "support_selector_no_contact_max",
        "support_selector_contact_min",
        "support_selector_dropout",
        "support_selector_monotonicity_weight",
        "support_selector_architecture",
        "support_selector_feature_source",
        "support_selector_neck_channels",
        "support_selector_hidden_dim",
        "support_selector_base_conditioning",
    ):
        if key not in result and key in checkpoint:
            result[key] = checkpoint[key]
    return result


def tactile_loss_config_from_checkpoint(
    checkpoint: Mapping[str, Any], *, full_ramp: bool = True
) -> TactileLossConfig:
    raw = checkpoint.get("loss_config", {})
    if not isinstance(raw, Mapping):
        raw = {}
    allowed = {field.name for field in dataclasses.fields(TactileLossConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    if full_ramp:
        values["loss_ramp_epochs"] = 0
    return TactileLossConfig(**values)


def build_frozen_base(
    checkpoint_path: os.PathLike[str] | str,
    dino_weights: os.PathLike[str] | str,
) -> tuple[DinoTactileModel, dict[str, Any], TactileLossConfig]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    dino_weights = Path(dino_weights).expanduser().resolve(strict=True)
    checkpoint = load_torch_checkpoint(checkpoint_path)
    if checkpoint.get("format") != "tactile_trainable_v2":
        raise ValueError(
            "Prior adapters require a compact format=tactile_trainable_v2 base checkpoint"
        )
    config = _checkpoint_model_config(checkpoint)
    if config.get("visual_backbone", "dinov3_hplus") != "dinov3_hplus":
        raise ValueError("Prior adapters currently require visual_backbone=dinov3_hplus")
    model = DinoTactileModel(
        tactile_head_type=str(config.get("tactile_head_type", "dense_v2_dino_rezero")),
        backbone_feature_layers=tuple(
            int(item) for item in config.get("backbone_feature_layers", (8, 16, 24, 32))
        ),
        visual_backbone="dinov3_hplus",
        dino_weights=str(dino_weights),
        dino_residual_max_scale=float(config.get("dino_residual_max_scale", 0.10)),
        dino_residual_rms_budget=float(config.get("dino_residual_rms_budget", 0.50)),
        pool_layout=str(config.get("pool_layout", "fullgrid32")),
        decoder_dropout_scale=float(config.get("decoder_dropout_scale", 1.0)),
        input_resolution=parse_resolution(config.get("input_resolution", (256, 192))),
        pool_output_channels=int(config.get("pool_output_channels", 32)),
        decoder_hidden_dim=int(config.get("decoder_hidden_dim", 512)),
        model_initialization_order=str(
            config.get(
                "model_initialization_order",
                CANONICAL_MODEL_INITIALIZATION_ORDER,
            )
        ),
        local_anchor_count=int(config.get("local_anchor_count", 512)),
        local_anchor_neighbors=int(config.get("local_anchor_neighbors", 4)),
        support_selector_mode=str(config.get("support_selector_mode", "contact")),
        support_selector_thresholds=tuple(
            float(item)
            for item in config.get("support_selector_thresholds", (0.10,))
        ),
        support_selector_no_contact_max=float(
            config.get("support_selector_no_contact_max", 0.02)
        ),
        support_selector_contact_min=float(
            config.get("support_selector_contact_min", 0.10)
        ),
        support_selector_dropout=float(config.get("support_selector_dropout", 0.10)),
        support_selector_monotonicity_weight=float(
            config.get("support_selector_monotonicity_weight", 0.10)
        ),
        support_selector_architecture=str(
            config.get("support_selector_architecture", "linear")
        ),
        support_selector_feature_source=str(
            config.get("support_selector_feature_source", "fullgrid32")
        ),
        support_selector_neck_channels=int(
            config.get("support_selector_neck_channels", 64)
        ),
        support_selector_hidden_dim=int(
            config.get("support_selector_hidden_dim", 512)
        ),
        support_selector_base_conditioning=str(
            config.get("support_selector_base_conditioning", "real")
        ),
    )
    expected_dino_sha = str(checkpoint.get("backbone_sha256", "") or "")
    actual_dino_sha = file_sha256(dino_weights)
    if expected_dino_sha and expected_dino_sha != actual_dino_sha:
        raise ValueError(
            f"DINO weight SHA256 mismatch: checkpoint={expected_dino_sha}, actual={actual_dino_sha}"
        )
    state = checkpoint.get("state_dict", {})
    head_state = {
        key[len("tactile_head.") :]: value
        for key, value in state.items()
        if str(key).startswith("tactile_head.")
    }
    if not head_state:
        raise ValueError("Base checkpoint has no tactile_head state")
    expected_state = model.tactile_head.state_dict()
    for old_prefix, new_prefix in (("decoder.0.project.", "decoder.0.projection."),):
        for old_key in tuple(head_state):
            if not old_key.startswith(old_prefix):
                continue
            new_key = new_prefix + old_key[len(old_prefix) :]
            if new_key in expected_state and new_key not in head_state:
                if tuple(head_state[old_key].shape) != tuple(expected_state[new_key].shape):
                    raise RuntimeError(
                        f"Cannot migrate tactile-head key {old_key!r} to {new_key!r}: shape mismatch"
                    )
                head_state[new_key] = head_state.pop(old_key)
    model.tactile_head.load_state_dict(head_state, strict=True)
    model.backbone_weights_path = str(dino_weights)
    model.backbone_weights_sha256 = actual_dino_sha
    model.base_checkpoint_path = str(checkpoint_path)
    model.base_checkpoint_sha256 = file_sha256(checkpoint_path)
    loss_config = tactile_loss_config_from_checkpoint(checkpoint, full_ramp=True)
    model.set_tactile_loss_config(loss_config)
    model.eval()
    return model, checkpoint, loss_config


def build_prior_model(
    *,
    base_checkpoint: os.PathLike[str] | str,
    dino_weights: os.PathLike[str] | str,
    adapter_type: str,
    prior_dim: int,
    feature_rms_budget: float = 0.05,
    logit_delta_max: float = 0.50,
    prior_dropout: float = 0.10,
    control_seed: int = 521,
    depth_hidden_channels: int = 128,
    depth_modulation_max_scale: float = 0.10,
    depth_attention_heads: int = 4,
    depth_attention_window: int = 5,
    zero_mean_logit_residual: bool = False,
    vlm_rank: int = 32,
    prior_control: str = "real",
    counterfactual_control: str = "",
    control_identity_weight: float = 0.0,
    feature_budget_penalty_weight: float = 0.0,
) -> tuple[FrozenBasePriorModel, dict[str, Any], TactileLossConfig]:
    del counterfactual_control, control_identity_weight, feature_budget_penalty_weight
    base, checkpoint, loss_config = build_frozen_base(base_checkpoint, dino_weights)
    model = FrozenBasePriorModel(
        base,
        adapter_type=adapter_type,
        prior_dim=prior_dim,
        feature_rms_budget=feature_rms_budget,
        logit_delta_max=logit_delta_max,
        prior_dropout=prior_dropout,
        control_seed=control_seed,
        depth_hidden_channels=depth_hidden_channels,
        depth_modulation_max_scale=depth_modulation_max_scale,
        depth_attention_heads=depth_attention_heads,
        depth_attention_window=depth_attention_window,
        zero_mean_logit_residual=zero_mean_logit_residual,
        vlm_rank=vlm_rank,
        default_control=prior_control,
    )
    return model, checkpoint, loss_config


def prior_checkpoint_payload(
    model: FrozenBasePriorModel,
    *,
    adapter_config: Mapping[str, Any],
    base_checkpoint: os.PathLike[str] | str,
    dino_weights: os.PathLike[str] | str,
    loss_config: TactileLossConfig,
    data_config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    monitor: str,
    score: Optional[float],
) -> dict[str, Any]:
    base_checkpoint = Path(base_checkpoint).expanduser().resolve(strict=True)
    dino_weights = Path(dino_weights).expanduser().resolve(strict=True)
    return {
        "format": PRIOR_CHECKPOINT_FORMAT,
        "adapter_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.prior_adapter.state_dict().items()
        },
        "adapter_config": dict(adapter_config),
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": file_sha256(base_checkpoint),
        "dino_weights": str(dino_weights),
        "dino_weights_sha256": file_sha256(dino_weights),
        "loss_config": dataclasses.asdict(loss_config),
        "data_config": dict(data_config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "monitor": str(monitor),
        "score": None if score is None else float(score),
    }


def load_prior_checkpoint(
    checkpoint_path: os.PathLike[str] | str,
    *,
    dino_weights_override: Optional[os.PathLike[str] | str] = None,
    base_checkpoint_override: Optional[os.PathLike[str] | str] = None,
) -> tuple[FrozenBasePriorModel, dict[str, Any], TactileLossConfig]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload = load_torch_checkpoint(checkpoint_path)
    if payload.get("format") != PRIOR_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported prior checkpoint format in {checkpoint_path}")
    config = dict(payload.get("adapter_config", {}))
    base_checkpoint = Path(
        base_checkpoint_override or payload.get("base_checkpoint", "")
    ).expanduser().resolve(strict=True)
    dino_weights = Path(
        dino_weights_override or payload.get("dino_weights", "")
    ).expanduser().resolve(strict=True)
    if payload.get("base_checkpoint_sha256") != file_sha256(base_checkpoint):
        raise ValueError("Frozen base checkpoint SHA256 differs from the adapter checkpoint")
    if payload.get("dino_weights_sha256") != file_sha256(dino_weights):
        raise ValueError("DINO weight SHA256 differs from the adapter checkpoint")
    model, _, loss_config = build_prior_model(
        base_checkpoint=base_checkpoint,
        dino_weights=dino_weights,
        **config,
    )
    model.prior_adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    model.eval()
    return model, payload, loss_config


def build_prior_selector_model(
    *,
    selector_checkpoint: os.PathLike[str] | str,
    dino_weights: os.PathLike[str] | str,
    adapter_type: str,
    prior_dim: int,
    prior_control: str = "real",
    control_seed: int = 521,
    feature_rms_budget: float = 0.05,
    prior_dropout: float = 0.0,
    depth_hidden_channels: int = 128,
    depth_modulation_max_scale: float = 0.10,
    anchor_residual_max_logit: float = 2.0,
    anchor_query_dim: int = 128,
    anchor_query_heads: int = 4,
    anchor_query_layers: int = 2,
    vlm_rank: int = 32,
    vlm_residual_max_logit: float = 1.0,
    **unused: Any,
) -> tuple[PriorSelectorModel, dict[str, Any]]:
    del unused
    base, checkpoint, _ = build_frozen_base(selector_checkpoint, dino_weights)
    model = PriorSelectorModel(
        base,
        adapter_type=adapter_type,
        prior_dim=prior_dim,
        prior_control=prior_control,
        control_seed=control_seed,
        feature_rms_budget=feature_rms_budget,
        prior_dropout=prior_dropout,
        depth_hidden_channels=depth_hidden_channels,
        depth_modulation_max_scale=depth_modulation_max_scale,
        anchor_residual_max_logit=anchor_residual_max_logit,
        anchor_query_dim=anchor_query_dim,
        anchor_query_heads=anchor_query_heads,
        anchor_query_layers=anchor_query_layers,
        vlm_rank=vlm_rank,
        vlm_residual_max_logit=vlm_residual_max_logit,
    )
    return model, checkpoint


def selector_prior_checkpoint_payload(
    model: PriorSelectorModel,
    *,
    adapter_config: Mapping[str, Any],
    selector_checkpoint: os.PathLike[str] | str,
    dino_weights: os.PathLike[str] | str,
    data_config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    monitor: str,
    score: Optional[float],
) -> dict[str, Any]:
    selector_checkpoint = Path(selector_checkpoint).expanduser().resolve(strict=True)
    dino_weights = Path(dino_weights).expanduser().resolve(strict=True)
    return {
        "format": SELECTOR_PRIOR_CHECKPOINT_FORMAT,
        "adapter_state_dict": {
            name: value.detach().cpu()
            for name, value in model.prior_adapter.state_dict().items()
        },
        "adapter_config": dict(adapter_config),
        "selector_checkpoint": str(selector_checkpoint),
        "selector_checkpoint_sha256": file_sha256(selector_checkpoint),
        "dino_weights": str(dino_weights),
        "dino_weights_sha256": file_sha256(dino_weights),
        "data_config": dict(data_config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "monitor": str(monitor),
        "score": None if score is None else float(score),
        "pressure_output_contract": "frozen_selector_checkpoint_exact",
    }


def load_prior_selector_checkpoint(
    checkpoint_path: os.PathLike[str] | str,
    *,
    dino_weights_override: Optional[os.PathLike[str] | str] = None,
    selector_checkpoint_override: Optional[os.PathLike[str] | str] = None,
) -> tuple[PriorSelectorModel, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload = load_torch_checkpoint(checkpoint_path)
    if payload.get("format") != SELECTOR_PRIOR_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported selector-prior checkpoint: {checkpoint_path}")
    selector_checkpoint = Path(
        selector_checkpoint_override or payload.get("selector_checkpoint", "")
    ).expanduser().resolve(strict=True)
    dino_weights = Path(
        dino_weights_override or payload.get("dino_weights", "")
    ).expanduser().resolve(strict=True)
    if payload.get("selector_checkpoint_sha256") != file_sha256(selector_checkpoint):
        raise ValueError("Frozen selector checkpoint SHA256 mismatch")
    if payload.get("dino_weights_sha256") != file_sha256(dino_weights):
        raise ValueError("DINO weight SHA256 mismatch")
    model, _ = build_prior_selector_model(
        selector_checkpoint=selector_checkpoint,
        dino_weights=dino_weights,
        **dict(payload.get("adapter_config", {})),
    )
    model.prior_adapter.load_state_dict(payload["adapter_state_dict"], strict=True)
    model.eval()
    return model, payload


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path.resolve()
    return None


def resolve_data_root(dataset: str) -> Path:
    canonical = canonical_datasets((dataset,))[0]
    if canonical == "TouchAnything":
        environment = os.environ.get("TOUCHANYTHING_DATA_ROOT")
        candidates = (
            Path(environment) if environment else Path("/__missing__"),
            REPO_ROOT.parent / "EgoTouch" / "extracted_frames",
            Path("/home/ma-user/work/cfzhao/EgoTouch/extracted_frames"),
        )
    else:
        environment = os.environ.get("OPENTOUCH_DATA_ROOT")
        candidates = (
            Path(environment) if environment else Path("/__missing__"),
            REPO_ROOT.parent / "OpenTouch Data" / "full_dataset",
            Path("/home/ma-user/work/cfzhao/OpenTouch Data/full_dataset"),
        )
    result = _first_existing(candidates)
    if result is None:
        raise FileNotFoundError(
            f"Could not auto-detect {canonical} root; set the matching *_DATA_ROOT"
        )
    return result


def discover_query_manifest(root: Path, dataset: str, split: str) -> Path:
    slug = "touchanything" if dataset == "TouchAnything" else "opentouch"
    aliases = [split]
    if split == "val":
        aliases.append("validation")
    candidates = []
    for alias in aliases:
        candidates.extend(
            (
                root / "manifests" / f"{slug}_{alias}.queries.jsonl",
                root / "manifests" / f"{alias}.queries.jsonl",
            )
        )
    result = _first_existing(candidates)
    if result is None:
        matches = sorted((root / "manifests").glob(f"*{split}*.queries.jsonl"))
        if len(matches) == 1:
            result = matches[0].resolve()
    if result is None:
        raise FileNotFoundError(
            f"No unique HDF5 query manifest for dataset={dataset}, split={split} under {root}"
        )
    return result


def default_bbox_manifest(dataset: str) -> Path:
    sam_root = Path(
        os.environ.get(
            "SAM3_RECONSTRUCTION_ROOT",
            REPO_ROOT / "sam3_bbox_reconstruction" / "outputs" / "full_reconstruction_flow",
        )
    ).expanduser()
    name = (
        "touchanything/manifests/touchanything_sam3_v1_highconf.jsonl"
        if dataset == "TouchAnything"
        else "opentouch/manifests/opentouch_sam3_v1.jsonl"
    )
    path = (sam_root / name).resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(
            f"Reviewed SAM3 bbox manifest not found: {path}; set SAM3_RECONSTRUCTION_ROOT"
        )
    return path


def hamer_config(input_resolution: Sequence[int]):
    height, width = parse_resolution(input_resolution)
    config_path = resolve_hamer_model_config_path(REPO_ROOT)
    config = get_config(str(config_path), update_cachedir=True)
    config.defrost()
    config.MODEL.IMAGE_SIZE = height
    config.MODEL.BBOX_SHAPE = [width, height]
    if "PRETRAINED_WEIGHTS" in config.MODEL.BACKBONE:
        config.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
    config.freeze()
    return config


def build_dataset(
    *,
    split: str,
    datasets: Any,
    input_resolution: Sequence[int],
    bbox_rescale_factor: float,
    train: bool,
    augmentation_enabled: bool,
    data_roots: Any = None,
    query_manifests: Any = None,
    bbox_manifests: Any = None,
    bbox_source_policy: str = "sam3_only",
    depth_sidecar_root: Optional[str] = None,
    depth_output_hw: Sequence[int] = (16, 12),
    hdf5_handle_cache_size: int = 4,
    hdf5_manifest_cache_dir: Optional[str] = None,
    crop_pipeline: str = "legacy_square_center",
    hdf5_sample_order: str = "legacy_sample_dir_hand",
    io_debug_enabled: bool = False,
    hdf5_batch_read_mode: str = "streaming",
) -> OpenTouchTactileDataset:
    canonical = canonical_datasets(datasets)
    roots = tuple(Path(path).expanduser().resolve(strict=True) for path in parse_csv(data_roots))
    if not roots:
        roots = tuple(resolve_data_root(dataset) for dataset in canonical)
    if len(roots) != len(canonical):
        raise ValueError(
            f"Expected one data root per dataset ({len(canonical)}), got {len(roots)}"
        )
    manifests = tuple(
        Path(path).expanduser().resolve(strict=True) for path in parse_csv(query_manifests)
    )
    if not manifests:
        manifests = tuple(
            discover_query_manifest(root, dataset, split)
            for root, dataset in zip(roots, canonical)
        )
    if len(manifests) != len(canonical):
        raise ValueError(
            f"Expected one query manifest per dataset ({len(canonical)}), got {len(manifests)}"
        )
    bboxes = tuple(
        Path(path).expanduser().resolve(strict=True) for path in parse_csv(bbox_manifests)
    )
    if not bboxes and bbox_source_policy == "sam3_only":
        bboxes = tuple(default_bbox_manifest(dataset) for dataset in canonical)
    return OpenTouchTactileDataset(
        cfg=hamer_config(input_resolution),
        split=split,
        data_dir=[str(path) for path in roots],
        train=train,
        tactile_only=True,
        input_resolution=parse_resolution(input_resolution),
        crop_pipeline=str(crop_pipeline),
        bbox_rescale_factor=float(bbox_rescale_factor),
        bbox_source_policy=bbox_source_policy,
        bbox_manifests=[str(path) for path in bboxes],
        augmentation_enabled=bool(augmentation_enabled),
        expected_datasets=canonical,
        data_backend="sequence_hdf5",
        query_manifests=[str(path) for path in manifests],
        hdf5_handle_cache_size=int(hdf5_handle_cache_size),
        hdf5_manifest_cache_dir=hdf5_manifest_cache_dir,
        hdf5_sample_order=str(hdf5_sample_order),
        depth_sidecar_root=depth_sidecar_root,
        depth_control="none",
        depth_output_hw=tuple(int(item) for item in depth_output_hw),
        lazy_index_records=True,
        io_debug_enabled=bool(io_debug_enabled),
        hdf5_batch_read_mode=str(hdf5_batch_read_mode),
    )


class CachedFeatureDataset(Dataset):
    """Attach one or more immutable feature caches by ``sample_uid``."""

    FIELD_TO_BATCH_KEY = {
        "z_rgb": "frozen_base_grid",
        "h_rgb": "frozen_base_bottleneck",
        "base_logits": "frozen_base_logits",
        "contact_neck": "frozen_contact_neck",
        "contact_anchor_logits": "frozen_contact_anchor_logits",
        "contact_logits": "frozen_contact_logits",
        "depth_grid": "depth_prior",
        "vlm_embedding": "vlm_prior",
        "tactile_signal": "tactile_signal",
        "palm_mask": "palm_mask",
        "has_tactile": "has_tactile",
    }

    def __init__(
        self,
        dataset: Dataset,
        cache_dirs: Sequence[os.PathLike[str] | str],
        *,
        require_fields: Sequence[str] = (),
    ):
        self.dataset = dataset
        cache_groups = []
        for raw_path in cache_dirs:
            path = Path(raw_path).expanduser().resolve(strict=True)
            if (path / "CACHE_DONE.json").is_file():
                cache_groups.append((FeatureCacheDataset(path, max_open_shards=2, copy_arrays=False),))
                continue
            partitions = sorted(
                child for child in path.glob("part-*-of-*")
                if (child / "CACHE_DONE.json").is_file()
            )
            if not partitions:
                raise FileNotFoundError(
                    f"No finalized feature cache or partition set under {path}"
                )
            match = re.fullmatch(r"part-(\d+)-of-(\d+)", partitions[0].name)
            expected_parts = int(match.group(2)) if match else len(partitions)
            if len(partitions) != expected_parts:
                raise RuntimeError(
                    f"Incomplete partitioned feature cache under {path}: "
                    f"found {len(partitions)}/{expected_parts} finalized partitions"
                )
            cache_groups.append(
                tuple(
                    FeatureCacheDataset(path, max_open_shards=2, copy_arrays=False)
                    for path in partitions
                )
            )
        self.cache_groups = tuple(cache_groups)
        self.caches = tuple(cache for group in self.cache_groups for cache in group)
        available = {field for cache in self.caches for field in cache.fields}
        missing = sorted(set(require_fields) - available)
        if missing:
            raise ValueError(f"Feature cache is missing required fields: {missing}")
        self.require_fields = tuple(require_fields)
        base_hashes = {
            str(cache.config.get("provenance", {}).get("base_checkpoint_sha256", ""))
            for cache in self.caches
            if any(
                field in cache.fields
                for field in (
                    "z_rgb",
                    "h_rgb",
                    "base_logits",
                    "contact_neck",
                    "contact_anchor_logits",
                )
            )
        }
        base_hashes.discard("")
        if len(base_hashes) > 1:
            raise ValueError(f"Feature caches mix multiple frozen bases: {sorted(base_hashes)}")
        self.base_checkpoint_sha256 = next(iter(base_hashes), "")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        sample_id = str(item.get("sample_uid", "")).strip()
        if not sample_id:
            raise KeyError("Dataset item is missing sample_uid required by feature cache")
        found_fields = set()
        for group in self.cache_groups:
            if len(group) > 1:
                cache = group[index % len(group)]
                try:
                    cached = cache[index // len(group)]
                except (IndexError, KeyError):
                    cached = cache.get_by_id(sample_id)
                if str(cached.get("sample_id")) != sample_id:
                    cached = cache.get_by_id(sample_id)
                candidates = ((cache, cached),)
            else:
                cache = group[0]
                try:
                    cached = cache.get_by_id(sample_id)
                except KeyError:
                    continue
                candidates = ((cache, cached),)
            for cache, cached in candidates:
                for field in cache.fields:
                    key = self.FIELD_TO_BATCH_KEY[field]
                    value = np.array(cached[field], copy=True)
                    item[key] = torch.from_numpy(value).float()
                    found_fields.add(field)
        if not found_fields:
            raise KeyError(f"Sample {sample_id!r} is absent from every configured feature cache")
        missing = sorted(set(self.require_fields) - found_fields)
        if missing:
            raise KeyError(
                f"Sample {sample_id!r} is missing required cached fields: {missing}"
            )
        if "depth_prior" in item:
            item["depth_available"] = torch.tensor(True)
        if "vlm_prior" in item:
            item["vlm_available"] = torch.tensor(True)
        return item

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getstate__(self):
        return self.__dict__


class FeatureOnlyTactileDataset(Dataset):
    """Train/evaluate entirely from immutable caches without JPEG/HDF5 reads."""

    REQUIRED_TRAIN_FIELDS = (
        "tactile_signal",
        "has_tactile",
    )

    def __init__(
        self,
        cache_dirs: Sequence[os.PathLike[str] | str],
        *,
        adapter_type: str,
    ):
        if not cache_dirs:
            raise ValueError("Feature-only mode requires at least one cache root")
        self.groups = []
        for raw_path in cache_dirs:
            path = Path(raw_path).expanduser().resolve(strict=True)
            if (path / "CACHE_DONE.json").is_file():
                paths = (path,)
            else:
                paths = tuple(
                    child
                    for child in sorted(path.glob("part-*-of-*"))
                    if (child / "CACHE_DONE.json").is_file()
                )
                if not paths:
                    raise FileNotFoundError(f"No finalized feature cache under {path}")
                match = re.fullmatch(r"part-(\d+)-of-(\d+)", paths[0].name)
                expected = int(match.group(2)) if match else len(paths)
                if len(paths) != expected:
                    raise RuntimeError(
                        f"Incomplete cache partition set under {path}: {len(paths)}/{expected}"
                    )
            self.groups.append(
                tuple(
                    FeatureCacheDataset(cache_path, max_open_shards=2, copy_arrays=False)
                    for cache_path in paths
                )
            )
        self.groups = tuple(self.groups)
        available = {field for group in self.groups for cache in group for field in cache.fields}
        is_vlm = adapter_type in {"vlm_lowrank", "vlm_global_calibrator"}
        is_selector = adapter_type in {
            "depth_mapping_rectifier",
            "depth_anchor_residual",
            "depth_anchor_query",
            "vlm_global_calibrator",
        }
        if is_selector:
            selector_pressure_field = next(
                (
                    field
                    for field in ("h_rgb", "base_logits")
                    if field in available
                ),
                None,
            )
            if selector_pressure_field is None:
                raise ValueError(
                    "Feature-only selector cache needs h_rgb or base_logits"
                )
            base_fields = (
                selector_pressure_field,
                "contact_neck",
                "contact_anchor_logits",
            )
        else:
            base_fields = ("h_rgb",) if adapter_type == "vlm_lowrank" else ("z_rgb",)
        prior_field = "vlm_embedding" if is_vlm else "depth_grid"
        required = set((*self.REQUIRED_TRAIN_FIELDS, *base_fields, prior_field))
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"Feature-only cache is missing fields: {missing}")
        # The base group defines global sample order. Partitioned caches were
        # built with global-index stride and can therefore be interleaved.
        base_candidates = [
            group for group in self.groups
            if any(
                set(base_fields).issubset(set(cache.fields))
                for cache in group
            )
        ]
        if len(base_candidates) != 1:
            raise ValueError("Feature-only mode requires exactly one base cache group")
        self.base_group = base_candidates[0]
        base_hashes = {
            str(cache.config.get("provenance", {}).get("base_checkpoint_sha256", ""))
            for cache in self.base_group
        }
        base_hashes.discard("")
        if len(base_hashes) != 1:
            raise ValueError(
                "Feature-only base cache must have one base_checkpoint_sha256"
            )
        self.base_checkpoint_sha256 = next(iter(base_hashes))
        self.sample_count = sum(len(cache) for cache in self.base_group)
        palm_values = self.base_group[0].config.get("provenance", {}).get("palm_mask")
        if not isinstance(palm_values, list) or not palm_values:
            raise ValueError(
                "Feature-only base cache provenance is missing the global palm_mask"
            )
        self.palm_mask = torch.tensor(palm_values, dtype=torch.float32)

    def __len__(self) -> int:
        return self.sample_count

    @staticmethod
    def _ordered_item(group, index: int):
        cache = group[index % len(group)]
        return cache[index // len(group)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        primary = self._ordered_item(self.base_group, index)
        sample_id = str(primary["sample_id"])
        merged = dict(primary)
        for group in self.groups:
            if group is self.base_group:
                continue
            if len(group) > 1:
                candidate = self._ordered_item(group, index)
                if str(candidate["sample_id"]) != sample_id:
                    candidate = group[index % len(group)].get_by_id(sample_id)
            else:
                candidate = group[0].get_by_id(sample_id)
            merged.update(candidate)
        result: dict[str, Any] = {
            "sample_uid": sample_id,
            "sample_ref": str(merged.get("sample_ref", sample_id)),
            "dataset": str(merged.get("dataset", "")),
            "sequence_key": str(merged.get("sequence_key", "")),
            "query_alias": str(merged.get("query_alias", "query")),
            "frame_idx": torch.tensor(int(merged.get("frame_idx", 0))),
        }
        for field, key in CachedFeatureDataset.FIELD_TO_BATCH_KEY.items():
            if field not in merged:
                continue
            value = np.array(merged[field], copy=True)
            tensor = torch.from_numpy(value).float()
            if field == "has_tactile":
                tensor = tensor.reshape(())
            result[key] = tensor
        if "depth_prior" in result:
            result["depth_available"] = torch.tensor(True)
        if "vlm_prior" in result:
            result["vlm_available"] = torch.tensor(True)
        # default_collate copies this immutable template into the batch; cloning
        # it once per sample here only doubles CPU memory traffic.
        result.setdefault("palm_mask", self.palm_mask)
        return result


def adapter_config_from_args(args: Any) -> dict[str, Any]:
    return {
        "adapter_type": str(args.adapter_type),
        "prior_dim": int(args.prior_dim),
        "feature_rms_budget": float(args.feature_rms_budget),
        "logit_delta_max": float(args.logit_delta_max),
        "prior_dropout": float(args.prior_dropout),
        "control_seed": int(args.seed),
        "depth_hidden_channels": int(args.depth_hidden_channels),
        "depth_modulation_max_scale": float(args.depth_modulation_max_scale),
        "depth_attention_heads": int(args.depth_attention_heads),
        "depth_attention_window": int(args.depth_attention_window),
        "zero_mean_logit_residual": bool(args.zero_mean_logit_residual),
        "vlm_rank": int(args.vlm_rank),
        "prior_control": str(args.prior_control),
        "counterfactual_control": str(args.counterfactual_control),
        "control_identity_weight": float(args.control_identity_weight),
        "feature_budget_penalty_weight": float(args.feature_budget_penalty_weight),
    }


def selector_prior_config_from_args(args: Any) -> dict[str, Any]:
    return {
        "adapter_type": str(args.adapter_type),
        "prior_dim": int(args.prior_dim),
        "prior_control": str(args.prior_control),
        "control_seed": int(args.seed),
        "feature_rms_budget": float(args.feature_rms_budget),
        "prior_dropout": float(args.prior_dropout),
        "depth_hidden_channels": int(args.depth_hidden_channels),
        "depth_modulation_max_scale": float(args.depth_modulation_max_scale),
        "anchor_residual_max_logit": float(args.anchor_residual_max_logit),
        "anchor_query_dim": int(args.anchor_query_dim),
        "anchor_query_heads": int(args.anchor_query_heads),
        "anchor_query_layers": int(args.anchor_query_layers),
        "vlm_rank": int(args.vlm_rank),
        "vlm_residual_max_logit": float(args.vlm_residual_max_logit),
    }
