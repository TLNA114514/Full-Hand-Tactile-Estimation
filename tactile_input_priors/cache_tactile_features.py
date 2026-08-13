#!/usr/bin/env python3
"""Precompute frozen tactile-base features into an atomic mmap cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactile_input_priors.feature_cache import (
    FeatureCacheBuilder,
    FeatureSpec,
    atomic_write_json,
    canonical_json,
)
from tactile_input_priors.runtime import (
    build_dataset,
    build_frozen_base,
    file_sha256,
    parse_csv,
)


def _sample_uid(record: Mapping[str, Any], index: int) -> str:
    for key in ("sample_uid", "sample_id", "sample_ref"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    raise KeyError(f"Dataset record {index} has no stable sample UID")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _write_partition_manifest(dataset, indices: Sequence[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for local_index, dataset_index in enumerate(indices):
                record = dataset.samples[dataset_index]
                metadata = {
                    "dataset": str(record.get("dataset", "")),
                    "sequence_key": str(record.get("sequence_key", "")),
                    "query_alias": str(record.get("query_alias", record.get("hand", "query"))),
                    "frame_idx": _safe_int(record.get("frame_idx", 0)),
                    "sample_ref": str(record.get("sample_ref", "")),
                }
                row = {
                    "sample_uid": _sample_uid(record, dataset_index),
                    "dataset_index": int(dataset_index),
                    "partition_index": int(local_index),
                    "metadata": metadata,
                }
                handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class FrozenFeatureCallback:
    def __init__(
        self,
        dataset,
        source_indices: Sequence[int],
        model,
        fields: Sequence[str],
        *,
        batch_size: int,
        device: torch.device,
    ):
        self.dataset = dataset
        self.source_indices = tuple(int(index) for index in source_indices)
        self.model = model.to(device).eval()
        self.fields = tuple(fields)
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.start = -1
        self.stop = -1
        self.values: dict[str, np.ndarray] = {}

    def _fill(self, source_index: int) -> None:
        self.start = int(source_index)
        self.stop = min(self.start + self.batch_size, len(self.source_indices))
        items = [self.dataset[self.source_indices[index]] for index in range(self.start, self.stop)]
        batch = default_collate(items)
        image = batch["img"].to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            levels = self.model._extract_tactile_features(image)
            head = self.model.tactile_head
            grid = head._fuse(levels) if hasattr(head, "_fuse") else head.base_projection(levels[-1])
            bottleneck = None
            if "h_rgb" in self.fields or "base_logits" in self.fields:
                bottleneck = grid
                for layer in head.decoder[:5]:
                    bottleneck = layer(bottleneck)
            logits = (
                head.decoder[5:](bottleneck)
                if "base_logits" in self.fields
                else None
            )
        values = {}
        if "z_rgb" in self.fields:
            values["z_rgb"] = grid.detach().half().cpu().numpy()
        if "h_rgb" in self.fields:
            values["h_rgb"] = bottleneck.detach().half().cpu().numpy()
        if "base_logits" in self.fields:
            values["base_logits"] = logits.detach().half().cpu().numpy()
        if "depth_grid" in self.fields:
            if "depth_prior" not in batch:
                raise KeyError("depth_grid caching requires --depth-sidecar-root")
            values["depth_grid"] = batch["depth_prior"].half().cpu().numpy()
        if "tactile_signal" in self.fields:
            values["tactile_signal"] = batch["tactile_signal"].float().cpu().numpy()
        if "palm_mask" in self.fields:
            values["palm_mask"] = batch["palm_mask"].to(torch.uint8).cpu().numpy()
        if "has_tactile" in self.fields:
            values["has_tactile"] = batch["has_tactile"].reshape(-1, 1).to(torch.uint8).cpu().numpy()
        self.values = {name: value for name, value in values.items() if name in self.fields}

    def __call__(self, row: Mapping[str, Any], source_index: int):
        del row
        if not self.start <= source_index < self.stop:
            self._fill(source_index)
        offset = source_index - self.start
        return {name: value[offset] for name, value in self.values.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dino-weights", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--fields",
        default="z_rgb,tactile_signal,has_tactile",
    )
    parser.add_argument("--datasets", default="touchanything")
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-roots", default="")
    parser.add_argument("--query-manifests", default="")
    parser.add_argument("--bbox-manifests", default="")
    parser.add_argument("--bbox-source-policy", default="sam3_only")
    parser.add_argument("--bbox-rescale-factor", type=float, default=1.2)
    parser.add_argument("--input-resolution", default="256x192")
    parser.add_argument("--depth-sidecar-root", default="")
    parser.add_argument("--hdf5-handle-cache-size", type=int, default=4)
    parser.add_argument("--hdf5-manifest-cache-dir", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--max-new-shards", type=int)
    parser.add_argument("--repair-invalid-shards", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fields = parse_csv(args.fields)
    allowed = {
        "z_rgb",
        "h_rgb",
        "base_logits",
        "depth_grid",
        "tactile_signal",
        "palm_mask",
        "has_tactile",
    }
    unknown = sorted(set(fields) - allowed)
    if unknown or not fields:
        raise ValueError(f"Unsupported/empty --fields; unknown={unknown}")
    if not 0 <= args.partition_index < args.num_partitions:
        raise ValueError("partition-index must lie in [0, num-partitions)")
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
        depth_sidecar_root=args.depth_sidecar_root or None,
        depth_output_hw=(16, 12),
        hdf5_handle_cache_size=args.hdf5_handle_cache_size,
        hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir or None,
    )
    source_indices = tuple(range(args.partition_index, len(dataset), args.num_partitions))
    cache_root = Path(args.cache_dir).expanduser().resolve(strict=False)
    cache_dir = (
        cache_root / f"part-{args.partition_index:02d}-of-{args.num_partitions:02d}"
        if args.num_partitions > 1
        else cache_root
    )
    source_manifest = cache_dir.parent / "source_manifests" / (
        f"{args.split}.part-{args.partition_index:02d}-of-{args.num_partitions:02d}.jsonl"
    )
    _write_partition_manifest(dataset, source_indices, source_manifest)
    model, _, _ = build_frozen_base(args.base_checkpoint, args.dino_weights)
    grid_height, grid_width = model.feature_grid_size
    specs = []
    if "z_rgb" in fields:
        specs.append(FeatureSpec("z_rgb", (256, grid_height, grid_width), "float16"))
    if "h_rgb" in fields:
        specs.append(FeatureSpec("h_rgb", (512,), "float16"))
    if "base_logits" in fields:
        specs.append(FeatureSpec("base_logits", (model.tactile_dim,), "float16"))
    if "depth_grid" in fields:
        sample = dataset[source_indices[0]]
        specs.append(FeatureSpec("depth_grid", tuple(sample["depth_prior"].shape), "float16"))
    if "tactile_signal" in fields:
        specs.append(FeatureSpec("tactile_signal", (model.tactile_dim,), "float32"))
    if "palm_mask" in fields:
        specs.append(FeatureSpec("palm_mask", (model.tactile_dim,), "uint8"))
    if "has_tactile" in fields:
        specs.append(FeatureSpec("has_tactile", (1,), "uint8"))
    provenance = {
        "producer": "tactile_input_priors.cache_tactile_features",
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "dino_weights_sha256": file_sha256(args.dino_weights),
        "dataset": args.datasets,
        "split": args.split,
        "query_manifest_sha256": getattr(dataset, "query_manifest_sha256", {}),
        "bbox_manifest_sha256": getattr(dataset, "bbox_manifest_sha256", {}),
        "bbox_rescale_factor": args.bbox_rescale_factor,
        "input_resolution": list(model.input_resolution),
        "augmentation_enabled": False,
        "palm_mask": np.asarray(dataset.palm_mask, dtype=np.uint8).tolist(),
        "partition_index": args.partition_index,
        "num_partitions": args.num_partitions,
    }
    atomic_write_json(cache_dir / "cache_request.json", vars(args))
    callback = FrozenFeatureCallback(
        dataset,
        source_indices,
        model,
        fields,
        batch_size=args.batch_size,
        device=torch.device(args.device),
    )
    builder = FeatureCacheBuilder(
        cache_dir,
        source_manifest,
        specs,
        provenance=provenance,
        shard_size=args.shard_size,
        sample_id_key="sample_uid",
        repair_invalid_shards=args.repair_invalid_shards,
    )
    result = builder.build(callback, max_new_shards=args.max_new_shards)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
