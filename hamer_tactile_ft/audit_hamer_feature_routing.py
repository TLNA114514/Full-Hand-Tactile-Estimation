#!/usr/bin/env python3
"""Run the Stage H0 HaMeR feature-routing integrity and lattice audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "hamer_tactile_ft", REPO_ROOT / "hamer"):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from hamer_tactile_ft.hamer_feature_routing import (  # noqa: E402
    HAMER_FEATURE_LAYERS,
    HAMER_INPUT_RESOLUTION,
    lattice_alignment,
    load_frozen_hamer_spatial_extractor,
    patch_embed_lattice,
    patch_lattice_from_conv,
    resample_feature_lattice,
    resolve_hamer_config_path,
    sha256_file,
)
from tactile_input_priors.runtime import (  # noqa: E402
    build_dataset,
    build_frozen_base,
    parse_resolution,
)


SCHEMA = "hamer_feature_routing_h0_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "hamer_tactile_ft" / "reports" / "hamer_feature_routing" / "h0_integrity"


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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_layers(raw: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(raw, str):
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    else:
        values = tuple(int(value) for value in raw)
    if not values or tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("layers must be unique increasing integers")
    return values


def _stable_score(sample_uid: str, seed: int) -> int:
    payload = f"{int(seed)}:hamer-h0:{sample_uid}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _record_uid(record: Mapping[str, Any], index: int) -> str:
    for key in ("sample_uid", "sample_id", "sample_ref"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    raise KeyError(f"Dataset record {index} has no stable sample UID")


def _record_is_right(record: Mapping[str, Any]) -> bool:
    if "is_right" in record:
        return bool(int(record["is_right"]))
    alias = str(record.get("query_alias", record.get("hand", "right"))).casefold()
    return alias not in {"left", "lh", "l"}


def _select_sample_indices(dataset, count: int, seed: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("sample-count must be positive")
    candidates = []
    for index, record in enumerate(dataset.samples):
        uid = _record_uid(record, index)
        sequence = str(record.get("sequence_key", "") or uid)
        candidates.append(
            (_stable_score(uid, seed), index, sequence, _record_is_right(record))
        )
    candidates.sort()
    selected: list[int] = []
    used_sequences: set[str] = set()
    target_per_side = max(1, count // 2)
    for desired_side in (False, True):
        for _, index, sequence, is_right in candidates:
            if is_right != desired_side or sequence in used_sequences:
                continue
            selected.append(index)
            used_sequences.add(sequence)
            if sum(_record_is_right(dataset.samples[item]) == desired_side for item in selected) >= target_per_side:
                break
    for _, index, sequence, _ in candidates:
        if len(selected) >= count:
            break
        if index in selected or sequence in used_sequences:
            continue
        selected.append(index)
        used_sequences.add(sequence)
    for _, index, _, _ in candidates:
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError("No H0 dataset samples were selected")
    return tuple(selected[:count])


def tactile_crop_affine(
    bbox_xyxy: Sequence[float],
    *,
    bbox_scale: float,
    output_hw: Sequence[int],
) -> np.ndarray:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError(f"Invalid bbox {bbox_xyxy!r}")
    height, width = (int(value) for value in output_hw)
    center_x = float(bbox[0] + bbox[2]) * 0.5
    center_y = float(bbox[1] + bbox[3]) * 0.5
    box_size = float(bbox_scale) * float(np.max(bbox[2:4] - bbox[0:2]))
    if box_size <= 1.0:
        raise ValueError(f"Degenerate bbox {bbox.tolist()}")
    transform = np.zeros((2, 3), dtype=np.float32)
    transform[0, 0] = height / box_size
    transform[1, 1] = height / box_size
    transform[0, 2] = -height * center_x / box_size + width * 0.5
    transform[1, 2] = height * (-center_y / box_size + 0.5)
    return transform


def crop_equivalence_audit(seed: int = 521) -> dict[str, Any]:
    """Prove rectangular affine equals square affine plus a center crop."""

    rng = np.random.default_rng(int(seed))
    image = rng.integers(0, 256, size=(379, 517, 3), dtype=np.uint8)
    bboxes = (
        (53.0, 41.0, 311.0, 337.0),
        (0.0, 0.0, 205.0, 270.0),
        (289.0, 109.0, 516.0, 378.0),
    )
    rows = []
    for bbox in bboxes:
        square_affine = tactile_crop_affine(
            bbox, bbox_scale=1.2, output_hw=(256, 256)
        )
        rectangle_affine = tactile_crop_affine(
            bbox, bbox_scale=1.2, output_hw=HAMER_INPUT_RESOLUTION
        )
        square = cv2.warpAffine(
            image, square_affine, (256, 256), flags=cv2.INTER_LINEAR
        )
        rectangle = cv2.warpAffine(
            image, rectangle_affine, (192, 256), flags=cv2.INTER_LINEAR
        )
        legacy = square[:, 32:-32]
        difference = np.abs(legacy.astype(np.int16) - rectangle.astype(np.int16))
        rows.append(
            {
                "bbox_xyxy": list(bbox),
                "maximum_pixel_difference": int(difference.max(initial=0)),
                "mean_pixel_difference": float(difference.mean()),
                "different_scalar_count": int(np.count_nonzero(difference)),
                "affine_translation_delta_x": float(
                    square_affine[0, 2] - rectangle_affine[0, 2]
                ),
            }
        )
    maximum = max(row["maximum_pixel_difference"] for row in rows)
    if maximum > 1:
        raise RuntimeError(
            f"Rectangular crop is not pixel-equivalent to square-plus-crop: max diff={maximum}"
        )
    flip = cv2.flip(image, 1)
    flip_involution_max = int(
        np.abs(cv2.flip(flip, 1).astype(np.int16) - image.astype(np.int16)).max()
    )
    if flip_involution_max != 0:
        raise RuntimeError("Left-to-right image flip is not an exact involution")
    return {
        "schema": "rectangular_crop_equivalence_v1",
        "output_resolution": list(HAMER_INPUT_RESOLUTION),
        "legacy_square_resolution": [256, 256],
        "legacy_center_crop_columns": [32, 224],
        "bbox_scale": 1.2,
        "maximum_pixel_difference": maximum,
        "flip_involution_maximum_difference": flip_involution_max,
        "cases": rows,
    }


def _feature_rows(
    model_name: str,
    layers: Sequence[int],
    sample_uids: Sequence[str],
    outputs: Sequence[torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for layer, feature in zip(layers, outputs):
        feature = feature.detach().float()
        for sample_index, sample_uid in enumerate(sample_uids):
            value = feature[sample_index]
            token_mean = value.mean(dim=(1, 2), keepdim=True)
            spatial_delta = value - token_mean
            rms = value.square().mean().sqrt().clamp_min(1e-12)
            rows.append(
                {
                    "row_type": "feature",
                    "model": model_name,
                    "layer": int(layer),
                    "sample_uid": sample_uid,
                    "channels": int(value.shape[0]),
                    "grid_height": int(value.shape[1]),
                    "grid_width": int(value.shape[2]),
                    "finite": bool(torch.isfinite(value).all().item()),
                    "mean": float(value.mean().item()),
                    "std": float(value.std(unbiased=False).item()),
                    "rms": float(rms.item()),
                    "minimum": float(value.min().item()),
                    "maximum": float(value.max().item()),
                    "spatial_centered_rms_ratio": float(
                        (spatial_delta.square().mean().sqrt() / rms).item()
                    ),
                }
            )
    for left_index, left_layer in enumerate(layers):
        for right_index in range(left_index + 1, len(layers)):
            right_layer = layers[right_index]
            left = outputs[left_index].detach().float().flatten(1)
            right = outputs[right_index].detach().float().flatten(1)
            similarities = torch.nn.functional.cosine_similarity(left, right, dim=1)
            for sample_uid, similarity in zip(sample_uids, similarities):
                rows.append(
                    {
                        "row_type": "layer_pair",
                        "model": model_name,
                        "layer": f"{left_layer}:{right_layer}",
                        "sample_uid": sample_uid,
                        "cosine_similarity": float(similarity.item()),
                    }
                )
    return rows


def _extract_statistics(
    model_name: str,
    extractor,
    images: Sequence[torch.Tensor],
    sample_uids: Sequence[str],
    layers: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
    determinism_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    determinism: dict[str, float] = {}
    first_batch = True
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        batch = torch.stack(images[start:stop]).to(device, non_blocking=True)
        with torch.inference_mode():
            outputs = tuple(extractor(batch, layers))
        if first_batch:
            with torch.inference_mode():
                repeated = tuple(extractor(batch, layers))
            for layer, original, rerun in zip(layers, outputs, repeated):
                maximum = float(
                    (original.detach().float() - rerun.detach().float()).abs().max().item()
                )
                determinism[str(layer)] = maximum
                if maximum > float(determinism_tolerance):
                    raise RuntimeError(
                        f"{model_name} layer {layer} is nondeterministic: "
                        f"max diff={maximum} > {determinism_tolerance}"
                    )
            first_batch = False
        for layer, output in zip(layers, outputs):
            if not bool(torch.isfinite(output).all().item()):
                raise FloatingPointError(f"{model_name} layer {layer} contains non-finite values")
        rows.extend(
            _feature_rows(model_name, layers, sample_uids[start:stop], outputs)
        )
        del batch, outputs
    return rows, {
        "model": model_name,
        "maximum_absolute_repeat_difference_by_layer": determinism,
        "tolerance": float(determinism_tolerance),
        "passed": True,
    }


def _dataset_samples(args) -> tuple[Any, list[torch.Tensor], list[str], list[dict[str, Any]]]:
    dataset = build_dataset(
        split=args.split,
        datasets=args.datasets,
        input_resolution=args.input_resolution,
        bbox_rescale_factor=args.bbox_rescale_factor,
        train=False,
        augmentation_enabled=False,
        data_roots=args.data_roots,
        query_manifests=args.query_manifests,
        bbox_manifests=args.bbox_manifests,
        bbox_source_policy=args.bbox_source_policy,
        hdf5_handle_cache_size=args.hdf5_handle_cache_size,
        hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir or None,
    )
    indices = _select_sample_indices(dataset, args.sample_count, args.seed)
    images = []
    sample_uids = []
    rows = []
    for index in indices:
        item = dataset[index]
        image = item["img"].detach().cpu().float().contiguous()
        if tuple(image.shape) != (3, *HAMER_INPUT_RESOLUTION):
            raise ValueError(
                f"Dataset sample {index} has image shape {tuple(image.shape)}, "
                "expected [3,256,192]"
            )
        if not bool(torch.isfinite(image).all().item()):
            raise FloatingPointError(f"Dataset sample {index} image is non-finite")
        uid = str(item["sample_uid"])
        bbox = item["query_bbox"].detach().cpu().numpy().astype(np.float64)
        affine = tactile_crop_affine(
            bbox,
            bbox_scale=args.bbox_rescale_factor,
            output_hw=HAMER_INPUT_RESOLUTION,
        )
        images.append(image)
        sample_uids.append(uid)
        rows.append(
            {
                "sample_uid": uid,
                "dataset_index": int(index),
                "dataset": str(item.get("dataset", "")),
                "split": str(args.split),
                "sequence_key": str(item.get("sequence_key", "")),
                "query_alias": str(item.get("query_alias", item.get("hand", ""))),
                "is_right_source": bool(float(item["right"].item()) >= 0.5),
                "left_to_right_flip_applied": bool(float(item["right"].item()) < 0.5),
                "bbox_xyxy": bbox.tolist(),
                "image_height": int(item["image_height"].item()),
                "image_width": int(item["image_width"].item()),
                "crop_affine": affine.tolist(),
                "normalized_image_sha256": hashlib.sha256(
                    memoryview(image.numpy()).cast("B")
                ).hexdigest(),
            }
        )
    return dataset, images, sample_uids, rows


def _base_checkpoint_contract(checkpoint: Mapping[str, Any], args) -> dict[str, Any]:
    config = checkpoint.get("model_config")
    config = dict(config) if isinstance(config, Mapping) else {}
    for key in (
        "input_resolution",
        "pool_layout",
        "tactile_head_type",
        "bbox_rescale_factor",
        "bbox_source_policy",
        "backbone_feature_layers",
    ):
        if key not in config and key in checkpoint:
            config[key] = checkpoint[key]
    resolution = parse_resolution(config.get("input_resolution", HAMER_INPUT_RESOLUTION))
    if resolution != HAMER_INPUT_RESOLUTION:
        raise ValueError(f"Base checkpoint resolution is {resolution}, expected 256x192")
    if str(config.get("pool_layout", "fullgrid32")) != "fullgrid32":
        raise ValueError("H0 requires the crop1.2 FullGrid32 base checkpoint")
    checkpoint_scale = config.get("bbox_rescale_factor")
    if checkpoint_scale is not None and abs(float(checkpoint_scale) - args.bbox_rescale_factor) > 1e-9:
        raise ValueError(
            f"Base checkpoint bbox scale={checkpoint_scale} differs from H0 "
            f"scale={args.bbox_rescale_factor}"
        )
    return {
        "path": str(Path(args.base_checkpoint).expanduser().resolve(strict=True)),
        "sha256": sha256_file(args.base_checkpoint),
        "format": str(checkpoint.get("format", "")),
        "model_config": config,
    }


def run_integrity(args) -> None:
    resolution = parse_resolution(args.input_resolution)
    if resolution != HAMER_INPUT_RESOLUTION:
        raise ValueError("Stage H0 is fixed to input-resolution=256x192")
    if abs(float(args.bbox_rescale_factor) - 1.2) > 1e-9:
        raise ValueError("Stage H0 is fixed to bbox-rescale-factor=1.2")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_report = crop_equivalence_audit(args.seed)
    dataset, images, sample_uids, sample_rows = _dataset_samples(args)
    dataset_mean = tuple(float(value) for value in dataset.cfg.MODEL.IMAGE_MEAN)
    dataset_std = tuple(float(value) for value in dataset.cfg.MODEL.IMAGE_STD)
    if dataset_mean != (0.485, 0.456, 0.406) or dataset_std != (0.229, 0.224, 0.225):
        raise ValueError(
            f"Dataset normalization differs from HaMeR/DINO: {dataset_mean}/{dataset_std}"
        )

    config_path = resolve_hamer_config_path(args.hamer_checkpoint, args.hamer_config)
    hamer, hamer_report = load_frozen_hamer_spatial_extractor(
        args.hamer_checkpoint,
        config_path=config_path,
        minimum_checkpoint_bytes=args.minimum_hamer_checkpoint_bytes,
    )
    hamer = hamer.to(device).eval()
    if hamer.training or any(module.training for module in hamer.modules()):
        raise RuntimeError("Frozen HaMeR extractor contains a module in train mode")
    if any(parameter.requires_grad for parameter in hamer.parameters()):
        raise RuntimeError("Frozen HaMeR extractor contains trainable parameters")
    hamer_lattice = patch_embed_lattice(
        "hamer_vit_h16",
        hamer.backbone.patch_embed,
        HAMER_INPUT_RESOLUTION,
    )
    hamer_rows, hamer_determinism = _extract_statistics(
        "hamer",
        hamer,
        images,
        sample_uids,
        args.hamer_layers,
        device=device,
        batch_size=args.batch_size,
        determinism_tolerance=args.determinism_tolerance,
    )
    del hamer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    base, base_checkpoint, _ = build_frozen_base(
        args.base_checkpoint, args.dino_weights
    )
    base_contract = _base_checkpoint_contract(base_checkpoint, args)
    base = base.to(device).eval()
    dino_lattice = patch_embed_lattice(
        "dinov3_hplus_patch16",
        base.backbone.model.patch_embed,
        HAMER_INPUT_RESOLUTION,
    )

    def dino_extract(image, layers):
        return tuple(base.backbone(image, layers))

    dino_rows, dino_determinism = _extract_statistics(
        "dinov3",
        dino_extract,
        images,
        sample_uids,
        tuple(int(value) for value in base.backbone_feature_layers),
        device=device,
        batch_size=args.batch_size,
        determinism_tolerance=args.determinism_tolerance,
    )
    dino_contract = {
        "weights_path": str(Path(args.dino_weights).expanduser().resolve(strict=True)),
        "weights_sha256": str(base.backbone_weights_sha256),
        "layers": list(base.backbone_feature_layers),
        "feature_channels": 1280,
        "output_grid": list(base.feature_grid_size),
        "normalization": "DinoV3BackboneAdapter norm=True",
    }
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if hamer_lattice.output_hw != (16, 12) or dino_lattice.output_hw != (16, 12):
        raise RuntimeError(
            f"Unexpected patch grids: HaMeR={hamer_lattice.output_hw}, "
            f"DINO={dino_lattice.output_hw}"
        )
    alignment = lattice_alignment(hamer_lattice, dino_lattice)
    lattice_contract = {
        "schema": "hamer_dino_lattice_contract_v1",
        "input_resolution": list(HAMER_INPUT_RESOLUTION),
        "hamer": hamer_lattice.to_dict(),
        "dino": dino_lattice.to_dict(),
        "hamer_to_dino": alignment,
        "native_index_is_exactly_colocated": bool(
            alignment["native_index_max_abs_center_delta_pixels"] == 0.0
        ),
        "selected_fixed_alignment": "center_aligned_bilinear_border",
    }

    contract = {
        "schema": SCHEMA,
        "status": "pass",
        "created_unix_seconds": time.time(),
        "input_resolution": list(HAMER_INPUT_RESOLUTION),
        "bbox_rescale_factor": float(args.bbox_rescale_factor),
        "bbox_source_policy": str(args.bbox_source_policy),
        "shared_normalized_input_tensor": True,
        "dataset_normalization": {
            "mean": list(dataset_mean),
            "std": list(dataset_std),
        },
        "dataset": {
            "name": str(args.datasets),
            "split": str(args.split),
            "sample_count": len(sample_uids),
            "sample_uids_sha256": _json_sha256(sample_uids),
            "query_manifest_sha256": dict(getattr(dataset, "query_manifest_sha256", {})),
            "bbox_manifest_sha256": dict(getattr(dataset, "bbox_manifest_sha256", {})),
        },
        "crop": crop_report,
        "hamer": hamer_report,
        "dino": dino_contract,
        "base_checkpoint": base_contract,
        "lattice": lattice_contract,
        "determinism": [hamer_determinism, dino_determinism],
        "selected_hamer_layer_for_h1_default": 32,
        "pose_outputs_loaded": False,
        "pose_outputs_consumed": False,
    }
    contract["contract_sha256"] = _json_sha256(contract)
    _atomic_jsonl(output_dir / "sample_manifest.jsonl", sample_rows)
    _atomic_csv(output_dir / "feature_statistics.csv", [*hamer_rows, *dino_rows])
    _atomic_json(output_dir / "crop_equivalence.json", crop_report)
    _atomic_json(output_dir / "lattice_contract.json", lattice_contract)
    _atomic_json(output_dir / "integrity_contract.json", contract)
    _atomic_json(
        output_dir / "H0_DONE.json",
        {
            "schema": SCHEMA,
            "status": "pass",
            "contract_sha256": contract["contract_sha256"],
            "sample_count": len(sample_uids),
            "selected_hamer_layer_for_h1_default": 32,
            "output_dir": str(output_dir),
        },
    )
    print(f"HaMeR H0 integrity audit passed: {output_dir}")
    print(
        "Patch-center offset (native HaMeR -> DINO): "
        f"{alignment['native_index_max_abs_center_delta_pixels']:.3f} px"
    )


def run_self_test(args) -> None:
    crop = crop_equivalence_audit(args.seed)
    hamer_conv = torch.nn.Conv2d(3, 4, kernel_size=16, stride=16, padding=4)
    dino_conv = torch.nn.Conv2d(3, 4, kernel_size=16, stride=16, padding=0)
    hamer = patch_lattice_from_conv(
        "hamer_test", hamer_conv, HAMER_INPUT_RESOLUTION
    )
    dino = patch_lattice_from_conv(
        "dino_test", dino_conv, HAMER_INPUT_RESOLUTION
    )
    alignment = lattice_alignment(hamer, dino)
    if hamer.output_hw != (16, 12) or dino.output_hw != (16, 12):
        raise AssertionError("Synthetic patch grids are incorrect")
    if hamer.center0_yx != (3.5, 3.5) or dino.center0_yx != (7.5, 7.5):
        raise AssertionError("Synthetic patch centers are incorrect")
    if alignment["target_to_source_index"]["offset_y"] != 0.25:
        raise AssertionError("Expected a quarter-cell HaMeR-to-DINO offset")
    source_y = torch.tensor(hamer.centers_y)[:, None]
    source_x = torch.tensor(hamer.centers_x)[None, :]
    feature = (source_y + source_x)[None, None]
    aligned = resample_feature_lattice(feature, hamer, dino)
    expected_y = torch.tensor(dino.centers_y[:-1])[:, None]
    expected_x = torch.tensor(dino.centers_x[:-1])[None, :]
    interior_error = float(
        (aligned[0, 0, :-1, :-1] - (expected_y + expected_x)).abs().max().item()
    )
    if interior_error > 1e-5:
        raise AssertionError(f"Lattice resampling error is {interior_error}")
    result = {
        "schema": "hamer_feature_routing_h0_self_test_v1",
        "status": "pass",
        "crop": crop,
        "hamer_lattice": hamer.to_dict(),
        "dino_lattice": dino.to_dict(),
        "alignment": alignment,
        "interior_resampling_max_error": interior_error,
    }
    if args.output_dir:
        _atomic_json(
            Path(args.output_dir).expanduser().resolve(strict=False) / "SELF_TEST.json",
            result,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    self_test = subparsers.add_parser("self-test", help="Run synthetic crop/lattice checks")
    self_test.add_argument("--seed", type=int, default=521)
    self_test.add_argument("--output-dir", default="")
    self_test.set_defaults(func=run_self_test)

    integrity = subparsers.add_parser("integrity", help="Run the full Stage H0 audit")
    integrity.add_argument("--hamer-checkpoint", required=True)
    integrity.add_argument("--hamer-config", default="")
    integrity.add_argument("--base-checkpoint", required=True)
    integrity.add_argument("--dino-weights", required=True)
    integrity.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    integrity.add_argument("--datasets", default="touchanything")
    integrity.add_argument("--split", default="val")
    integrity.add_argument("--data-roots", default="")
    integrity.add_argument("--query-manifests", default="")
    integrity.add_argument("--bbox-manifests", default="")
    integrity.add_argument("--bbox-source-policy", default="sam3_only")
    integrity.add_argument("--bbox-rescale-factor", type=float, default=1.2)
    integrity.add_argument("--input-resolution", default="256x192")
    integrity.add_argument("--sample-count", type=int, default=4)
    integrity.add_argument("--batch-size", type=int, default=1)
    integrity.add_argument("--seed", type=int, default=521)
    integrity.add_argument("--device", default="cuda:0")
    integrity.add_argument(
        "--hamer-layers", type=_parse_layers, default=HAMER_FEATURE_LAYERS
    )
    integrity.add_argument("--determinism-tolerance", type=float, default=1e-6)
    integrity.add_argument(
        "--minimum-hamer-checkpoint-bytes", type=int, default=100_000_000
    )
    integrity.add_argument("--hdf5-handle-cache-size", type=int, default=2)
    integrity.add_argument("--hdf5-manifest-cache-dir", default="")
    integrity.set_defaults(func=run_integrity)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
