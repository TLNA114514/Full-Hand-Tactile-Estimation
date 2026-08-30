#!/usr/bin/env python3
"""Precompute frozen tactile-base features into an atomic mmap cache."""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import heapq
import json
import os
import sys
import tempfile
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
    sha256_json,
)
from tactile_input_priors.runtime import (
    build_dataset,
    build_frozen_base,
    file_sha256,
    parse_csv,
    parse_resolution,
)


def _sample_uid(record: Mapping[str, Any], index: int) -> str:
    for key in ("sample_uid", "sample_id", "sample_ref"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    raise KeyError(f"Dataset record {index} has no stable sample UID")


def _stable_sample_score(
    record: Mapping[str, Any],
    index: int,
    *,
    seed: int,
    namespace: str,
) -> int:
    uid = _sample_uid(record, index)
    payload = f"{int(seed)}:{namespace}:{uid}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _selected_dataset_indices(
    dataset,
    *,
    sample_limit: int,
    max_samples_per_sequence: int,
    seed: int,
) -> tuple[int, ...]:
    """Select a stable, query-aware subset before any image or DINO read."""

    sample_limit = int(sample_limit)
    sequence_limit = int(max_samples_per_sequence)
    if sample_limit <= 0 and sequence_limit <= 0:
        return tuple(range(len(dataset)))

    def sequence_key(record: Mapping[str, Any], index: int) -> str:
        sequence = str(record.get("sequence_key", "")).strip()
        query = str(record.get("query_alias", record.get("hand", "query"))).strip()
        return f"{sequence or _sample_uid(record, index)}::{query or 'query'}"

    if sequence_limit > 0:
        # Each heap retains the smallest stable hashes without depending on
        # manifest order. Negative scores make heap[0] the current worst row.
        heaps: dict[str, list[tuple[int, int]]] = {}
        for index in range(len(dataset)):
            record = dataset.samples[index]
            score = _stable_sample_score(
                record, index, seed=seed, namespace="sequence-subset"
            )
            heap = heaps.setdefault(sequence_key(record, index), [])
            candidate = (-score, -int(index))
            if len(heap) < sequence_limit:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
        candidates = [
            -negative_index
            for heap in heaps.values()
            for _, negative_index in heap
        ]
    else:
        candidates = list(range(len(dataset)))

    if sample_limit > 0 and len(candidates) > sample_limit:
        scored = []
        for index in candidates:
            record = dataset.samples[index]
            scored.append(
                (
                    _stable_sample_score(
                        record, index, seed=seed, namespace="global-subset"
                    ),
                    int(index),
                )
            )
        candidates = [index for _, index in heapq.nsmallest(sample_limit, scored)]
    return tuple(sorted(int(index) for index in candidates))


def _selection_contract(args, dataset) -> dict[str, Any]:
    return {
        "schema": "tactile_cache_sample_selection_v1",
        "dataset": str(args.datasets),
        "split": str(args.split),
        "source_sample_count": int(len(dataset)),
        "sample_limit": int(args.sample_limit),
        "max_samples_per_sequence": int(args.max_samples_per_sequence),
        "sample_seed": int(args.sample_seed),
        "query_manifest_sha256": dict(
            getattr(dataset, "query_manifest_sha256", {})
        ),
        "bbox_manifest_sha256": dict(
            getattr(dataset, "bbox_manifest_sha256", {})
        ),
    }


def _selection_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _selection_digest(indices: Sequence[int]) -> str:
    array = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _write_selected_indices(
    path: Path,
    indices: Sequence[int],
    contract: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}-{time.time_ns()}")
    array = np.asarray(indices, dtype=np.int64)
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    atomic_write_json(
        _selection_metadata_path(path),
        {
            "contract": dict(contract),
            "contract_sha256": sha256_json(contract),
            "selected_sample_count": int(len(array)),
            "selected_indices_sha256": _selection_digest(array),
        },
    )


def _load_selected_indices(
    path: Path,
    dataset,
    contract: Mapping[str, Any],
) -> tuple[int, ...]:
    path = path.expanduser().resolve(strict=True)
    metadata_path = _selection_metadata_path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Sample selection metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("contract_sha256") != sha256_json(contract):
        raise RuntimeError(
            f"Sample selection contract differs for {path}; regenerate the selection"
        )
    indices_array = np.load(path, allow_pickle=False)
    if indices_array.ndim != 1 or not np.issubdtype(indices_array.dtype, np.integer):
        raise ValueError(f"Selected indices must be one integer vector: {path}")
    indices = tuple(int(value) for value in indices_array.tolist())
    if not indices:
        raise ValueError(f"Sample selection is empty: {path}")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError(f"Sample selection must be sorted and unique: {path}")
    if indices[0] < 0 or indices[-1] >= len(dataset):
        raise IndexError(f"Sample selection is outside dataset bounds: {path}")
    if metadata.get("selected_indices_sha256") != _selection_digest(indices):
        raise RuntimeError(f"Sample selection checksum mismatch: {path}")
    return indices


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


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
                    "source_frame_idx": _optional_int(record.get("source_frame_idx")),
                    "timestamp": _optional_float(record.get("timestamp")),
                    "is_right": _safe_int(
                        record.get(
                            "is_right",
                            str(record.get("query_alias", record.get("hand", ""))).lower()
                            == "right",
                        )
                    ),
                    "bbox_xyxy": list(record.get("bbox_xyxy", record.get("bbox", ()))),
                    "bbox_association_id": str(
                        (
                            record.get("bbox_source", {}).get("association_id", "")
                            if isinstance(record.get("bbox_source"), Mapping)
                            else record.get("bbox_association_id", "")
                        )
                        or ""
                    ),
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
        floating_dtype: str,
    ):
        self.dataset = dataset
        self.source_indices = tuple(int(index) for index in source_indices)
        self.model = model.to(device).eval()
        self.fields = tuple(fields)
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.floating_dtype = str(floating_dtype)
        if self.floating_dtype not in {"float16", "float32"}:
            raise ValueError("floating_dtype must be float16 or float32")
        self.start = -1
        self.stop = -1
        self.values: dict[str, np.ndarray] = {}

    def _floating_numpy(self, value: torch.Tensor) -> np.ndarray:
        value = value.detach().cpu()
        if self.floating_dtype == "float16":
            value = value.half()
        else:
            value = value.float()
        return value.numpy()

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
            if (
                "h_rgb" in self.fields
                or "base_logits" in self.fields
                or "palm_base_logits" in self.fields
            ):
                bottleneck = grid
                for layer in head.decoder[:5]:
                    bottleneck = layer(bottleneck)
            logits = (
                head.decoder[5:](bottleneck)
                if "base_logits" in self.fields or "palm_base_logits" in self.fields
                else None
            )
            contact_neck = None
            contact_anchor_logits = None
            contact_logits = None
            if any(
                field in self.fields
                for field in ("contact_neck", "contact_anchor_logits", "contact_logits")
            ):
                support = getattr(head, "support_selector", None)
                mapping = getattr(support, "selector", None)
                if mapping is None or getattr(support, "raw_fusion", None) is not None:
                    raise ValueError(
                        "Contact cache fields require the Binary Grid rezero selector checkpoint"
                    )
                contact_neck = mapping.spatial_neck(grid)
                contact_anchor_logits = mapping.anchor_decoder(
                    contact_neck.flatten(1)
                ).reshape(grid.shape[0], mapping.anchor_count)
                if "contact_logits" in self.fields:
                    per_vertex = contact_anchor_logits[:, mapping.vertex_anchor_indices]
                    contact_logits = (
                        per_vertex
                        * mapping.vertex_anchor_weights.to(per_vertex)[None]
                    ).sum(dim=2)
                    contact_logits = contact_logits * mapping.valid_palm_mask.to(
                        contact_logits
                    )[None]
        values = {}
        if "z_rgb" in self.fields:
            values["z_rgb"] = self._floating_numpy(grid)
        if "h_rgb" in self.fields:
            values["h_rgb"] = self._floating_numpy(bottleneck)
        if "base_logits" in self.fields:
            values["base_logits"] = self._floating_numpy(logits)
        if "palm_base_logits" in self.fields:
            palm = batch["palm_mask"][0].to(device=logits.device) > 0.5
            values["palm_base_logits"] = self._floating_numpy(logits[:, palm])
        if "contact_neck" in self.fields:
            values["contact_neck"] = self._floating_numpy(contact_neck)
        if "contact_anchor_logits" in self.fields:
            values["contact_anchor_logits"] = self._floating_numpy(
                contact_anchor_logits
            )
        if "contact_logits" in self.fields:
            values["contact_logits"] = self._floating_numpy(contact_logits)
        if "depth_grid" in self.fields:
            if "depth_prior" not in batch:
                raise KeyError("depth_grid caching requires --depth-sidecar-root")
            values["depth_grid"] = self._floating_numpy(batch["depth_prior"])
        if "tactile_signal" in self.fields:
            values["tactile_signal"] = batch["tactile_signal"].float().cpu().numpy()
        if "palm_tactile_signal" in self.fields:
            palm = batch["palm_mask"][0] > 0.5
            values["palm_tactile_signal"] = (
                batch["tactile_signal"][:, palm].float().cpu().numpy()
            )
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
    parser.add_argument("--cache-dir", default="")
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
    parser.add_argument(
        "--floating-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Storage dtype for frozen features and priors; targets remain float32.",
    )
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        help="Stable total sample cap applied before partitioning; 0 keeps all rows.",
    )
    parser.add_argument(
        "--max-samples-per-sequence",
        type=int,
        default=0,
        help="Stable per-sequence/query cap applied before --sample-limit; 0 disables it.",
    )
    parser.add_argument("--sample-seed", type=int, default=521)
    parser.add_argument(
        "--selection-output",
        default="",
        help="Build/reuse one contracted .npy sample index and exit before loading DINO.",
    )
    parser.add_argument(
        "--selected-indices-file",
        default="",
        help="Reuse a contracted sample index generated by --selection-output.",
    )
    parser.add_argument("--max-new-shards", type=int)
    parser.add_argument("--repair-invalid-shards", action="store_true")
    parser.add_argument(
        "--lock-timeout-seconds",
        type=float,
        default=21600.0,
        help="Maximum time to wait for another host building the same partition.",
    )
    parser.add_argument(
        "--break-stale-lock",
        action="store_true",
        help="Reclaim an existing cross-host build lock; use only after verifying its owner stopped.",
    )
    parser.add_argument(
        "--print-cache-key",
        action="store_true",
        help="Print the semantic cache key without loading DINO or writing files.",
    )
    return parser


def _cache_identity(
    args,
    fields,
    dataset,
    selected_count: int,
    selected_indices_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": "tactile_frozen_feature_identity_v2",
        "producer": "tactile_input_priors.cache_tactile_features",
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "dino_weights_sha256": file_sha256(args.dino_weights),
        "dataset": str(args.datasets),
        "split": str(args.split),
        "source_sample_count": int(len(dataset)),
        "sample_count": int(selected_count),
        "sample_limit": int(args.sample_limit),
        "max_samples_per_sequence": int(args.max_samples_per_sequence),
        "sample_seed": int(args.sample_seed),
        "selected_indices_sha256": str(selected_indices_sha256),
        "query_manifest_sha256": dict(
            getattr(dataset, "query_manifest_sha256", {})
        ),
        "bbox_manifest_sha256": dict(
            getattr(dataset, "bbox_manifest_sha256", {})
        ),
        "depth_sidecar_contract": dict(
            getattr(dataset, "depth_sidecar_contract", {})
        ),
        "bbox_rescale_factor": float(args.bbox_rescale_factor),
        "bbox_source_policy": str(args.bbox_source_policy),
        "input_resolution": list(parse_resolution(args.input_resolution)),
        "augmentation_enabled": False,
        "fields": sorted(str(field) for field in fields),
        "floating_dtype": str(args.floating_dtype),
        "num_partitions": int(args.num_partitions),
        "shard_size": int(args.shard_size),
    }


def main() -> None:
    args = build_parser().parse_args()
    fields = parse_csv(args.fields)
    allowed = {
        "z_rgb",
        "h_rgb",
        "base_logits",
        "palm_base_logits",
        "contact_neck",
        "contact_anchor_logits",
        "contact_logits",
        "depth_grid",
        "tactile_signal",
        "palm_tactile_signal",
        "palm_mask",
        "has_tactile",
    }
    unknown = sorted(set(fields) - allowed)
    if unknown or not fields:
        raise ValueError(f"Unsupported/empty --fields; unknown={unknown}")
    if not 0 <= args.partition_index < args.num_partitions:
        raise ValueError("partition-index must lie in [0, num-partitions)")
    if args.shard_size < 1:
        raise ValueError("--shard-size must be positive")
    if args.num_partitions < 1:
        raise ValueError("--num-partitions must be positive")
    if args.sample_limit < 0:
        raise ValueError("--sample-limit must be nonnegative")
    if args.max_samples_per_sequence < 0:
        raise ValueError("--max-samples-per-sequence must be nonnegative")
    redirect = (
        contextlib.redirect_stdout(sys.stderr)
        if args.print_cache_key
        else contextlib.nullcontext()
    )
    with redirect:
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
            depth_sidecar_root=(
                (args.depth_sidecar_root or None)
                if "depth_grid" in fields
                else None
            ),
            depth_output_hw=(16, 12),
            hdf5_handle_cache_size=args.hdf5_handle_cache_size,
            hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir or None,
        )
    selection_contract = _selection_contract(args, dataset)
    if args.selection_output:
        selection_path = Path(args.selection_output).expanduser().resolve()
        metadata_path = _selection_metadata_path(selection_path)
        if selection_path.is_file() and metadata_path.is_file():
            try:
                existing = _load_selected_indices(
                    selection_path, dataset, selection_contract
                )
            except (FileNotFoundError, RuntimeError, ValueError, IndexError):
                existing = ()
            if existing:
                print(
                    f"Reusing sample selection: {selection_path} "
                    f"({len(existing):,} rows)"
                )
                return
        selected_indices = _selected_dataset_indices(
            dataset,
            sample_limit=args.sample_limit,
            max_samples_per_sequence=args.max_samples_per_sequence,
            seed=args.sample_seed,
        )
        _write_selected_indices(selection_path, selected_indices, selection_contract)
        print(
            f"Sample selection ready: {selection_path} "
            f"({len(selected_indices):,}/{len(dataset):,} rows)"
        )
        return
    if args.selected_indices_file:
        selected_indices = _load_selected_indices(
            Path(args.selected_indices_file), dataset, selection_contract
        )
    else:
        selected_indices = _selected_dataset_indices(
            dataset,
            sample_limit=args.sample_limit,
            max_samples_per_sequence=args.max_samples_per_sequence,
            seed=args.sample_seed,
        )
    selected_digest = _selection_digest(selected_indices)
    identity = _cache_identity(
        args, fields, dataset, len(selected_indices), selected_digest
    )
    if args.print_cache_key:
        print(sha256_json(identity))
        return
    if not args.cache_dir:
        raise ValueError("--cache-dir is required unless --print-cache-key is used")
    source_indices = selected_indices[args.partition_index :: args.num_partitions]
    if not source_indices:
        raise ValueError(
            f"Partition {args.partition_index}/{args.num_partitions} has no samples"
        )
    cache_root = Path(args.cache_dir).expanduser().resolve(strict=False)
    cache_dir = (
        cache_root / f"part-{args.partition_index:02d}-of-{args.num_partitions:02d}"
        if args.num_partitions > 1
        else cache_root
    )
    published_source_manifest = cache_dir.parent / "source_manifests" / (
        f"{args.split}.part-{args.partition_index:02d}-of-{args.num_partitions:02d}.jsonl"
    )
    _write_partition_manifest(dataset, source_indices, published_source_manifest)
    manifest_tmp_dir = Path(
        os.environ.get("TACTILE_FEATURE_MANIFEST_TMPDIR", tempfile.gettempdir())
    ).expanduser().resolve(strict=False)
    manifest_tmp_dir.mkdir(parents=True, exist_ok=True)
    descriptor, private_manifest_name = tempfile.mkstemp(
        prefix=(
            f"tactile-feature-{args.split}-part-{args.partition_index:02d}-"
            f"of-{args.num_partitions:02d}-"
        ),
        suffix=".jsonl",
        dir=manifest_tmp_dir,
    )
    os.close(descriptor)
    source_manifest = Path(private_manifest_name)
    source_manifest.unlink(missing_ok=True)
    _write_partition_manifest(dataset, source_indices, source_manifest)

    def cleanup_private_manifest() -> None:
        source_manifest.unlink(missing_ok=True)

    atexit.register(cleanup_private_manifest)
    model, _, _ = build_frozen_base(args.base_checkpoint, args.dino_weights)
    if tuple(model.input_resolution) != tuple(parse_resolution(args.input_resolution)):
        raise ValueError(
            f"Checkpoint input resolution {tuple(model.input_resolution)} differs from "
            f"cache request {args.input_resolution}"
        )
    grid_height, grid_width = model.feature_grid_size
    floating_dtype = str(args.floating_dtype)
    specs = []
    if "z_rgb" in fields:
        specs.append(
            FeatureSpec("z_rgb", (256, grid_height, grid_width), floating_dtype)
        )
    if "h_rgb" in fields:
        specs.append(
            FeatureSpec(
                "h_rgb",
                (int(model.decoder_hidden_dim),),
                floating_dtype,
            )
        )
    if "base_logits" in fields:
        specs.append(
            FeatureSpec("base_logits", (model.tactile_dim,), floating_dtype)
        )
    palm_vertex_indices = np.flatnonzero(np.asarray(dataset.palm_mask) > 0.5).astype(
        np.int32
    )
    if "palm_base_logits" in fields:
        specs.append(
            FeatureSpec("palm_base_logits", (len(palm_vertex_indices),), floating_dtype)
        )
    if "contact_neck" in fields:
        specs.append(
            FeatureSpec(
                "contact_neck",
                (
                    int(model.support_selector_neck_channels),
                    grid_height,
                    grid_width,
                ),
                floating_dtype,
            )
        )
    if "contact_anchor_logits" in fields:
        specs.append(
            FeatureSpec(
                "contact_anchor_logits",
                (int(model.local_anchor_count),),
                floating_dtype,
            )
        )
    if "contact_logits" in fields:
        specs.append(
            FeatureSpec("contact_logits", (model.tactile_dim,), floating_dtype)
        )
    if "depth_grid" in fields:
        sample = dataset[source_indices[0]]
        specs.append(
            FeatureSpec(
                "depth_grid", tuple(sample["depth_prior"].shape), floating_dtype
            )
        )
    if "tactile_signal" in fields:
        specs.append(FeatureSpec("tactile_signal", (model.tactile_dim,), "float32"))
    if "palm_tactile_signal" in fields:
        specs.append(
            FeatureSpec("palm_tactile_signal", (len(palm_vertex_indices),), "float32")
        )
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
        "depth_sidecar_contract": getattr(dataset, "depth_sidecar_contract", {}),
        "bbox_rescale_factor": args.bbox_rescale_factor,
        "bbox_source_policy": args.bbox_source_policy,
        "input_resolution": list(model.input_resolution),
        "augmentation_enabled": False,
        "floating_dtype": floating_dtype,
        "cache_identity": identity,
        "cache_identity_sha256": sha256_json(identity),
        "palm_mask": np.asarray(dataset.palm_mask, dtype=np.uint8).tolist(),
        "palm_vertex_indices": palm_vertex_indices.tolist(),
        "partition_index": args.partition_index,
        "num_partitions": args.num_partitions,
        "source_sample_count": len(dataset),
        "selected_sample_count": len(selected_indices),
        "sample_limit": args.sample_limit,
        "max_samples_per_sequence": args.max_samples_per_sequence,
        "sample_seed": args.sample_seed,
    }
    atomic_write_json(cache_dir / "cache_request.json", vars(args))
    callback = FrozenFeatureCallback(
        dataset,
        source_indices,
        model,
        fields,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        floating_dtype=floating_dtype,
    )
    builder = FeatureCacheBuilder(
        cache_dir,
        source_manifest,
        specs,
        provenance=provenance,
        shard_size=args.shard_size,
        sample_id_key="sample_uid",
        lock_timeout_seconds=args.lock_timeout_seconds,
        break_stale_lock=args.break_stale_lock,
        repair_invalid_shards=args.repair_invalid_shards,
    )
    try:
        result = builder.build(callback, max_new_shards=args.max_new_shards)
    finally:
        cleanup_private_manifest()
        atexit.unregister(cleanup_private_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
