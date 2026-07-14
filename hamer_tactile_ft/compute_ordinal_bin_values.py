#!/usr/bin/env python
"""Compute train-split dense-pressure representatives for ordinal decode audits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

import numpy as np


FT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = FT_DIR.parent
sys.path.append(str(WORKSPACE_DIR / "hamer"))
sys.path.append(str(FT_DIR))

from hamer.configs import get_config
from dataset import OpenTouchTactileDataset
from train import resolve_data_dirs


_STATS_PALM_MASK = None
_STATS_THRESHOLDS = None
_STATS_TACTILE_DIM = None


def _init_stats_worker(palm_mask, thresholds, tactile_dim):
    """Keep immutable statistics inputs local to each worker process."""
    global _STATS_PALM_MASK, _STATS_THRESHOLDS, _STATS_TACTILE_DIM
    _STATS_PALM_MASK = np.asarray(palm_mask, dtype=bool)
    _STATS_THRESHOLDS = np.asarray(thresholds, dtype=np.float32)
    _STATS_TACTILE_DIM = int(tactile_dim)


def _pressure_from_record(sample_record):
    """Load only dense GT pressure; image decoding is unnecessary for this tool."""
    meta_path = os.path.join(sample_record["sample_dir"], "meta.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    dataset_name = sample_record.get("dataset", meta.get("dataset", "OpenTouch"))
    if dataset_name == "TouchAnything":
        hand = sample_record.get("hand")
        pressure = meta.get("hands", {}).get(hand, {}).get("gaussian_pressure")
    else:
        is_right = int(sample_record.get("is_right", meta.get("is_right", 1)))
        side = "right" if is_right else "left"
        original_data = meta.get("original_hdf5_data", {})
        pressure = original_data.get(f"{side}_pressure_continuous_subdiv")
        if pressure is None:
            pressure = meta.get("gaussian_pressure")

    if pressure is None:
        return None
    values = np.asarray(pressure, dtype=np.float32)
    if values.shape != (_STATS_TACTILE_DIM,):
        return None
    return values


def _summarize_record_chunk(sample_records):
    """Return small per-bin aggregates instead of transferring full pressure arrays."""
    bin_count = len(_STATS_THRESHOLDS) + 1
    counts = np.zeros(bin_count, dtype=np.int64)
    sums = np.zeros(bin_count, dtype=np.float64)
    valid_samples = 0
    skipped_samples = 0

    for sample_record in sample_records:
        pressure = _pressure_from_record(sample_record)
        if pressure is None:
            skipped_samples += 1
            continue
        values = np.clip(pressure[_STATS_PALM_MASK], 0.0, 1.0)
        if values.size == 0:
            skipped_samples += 1
            continue
        bin_idx = np.searchsorted(_STATS_THRESHOLDS, values, side="right")
        counts += np.bincount(bin_idx, minlength=bin_count)
        sums += np.bincount(bin_idx, weights=values, minlength=bin_count)
        valid_samples += 1
    return counts, sums, valid_samples, skipped_samples


def _chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def compute_bin_statistics(sample_records, palm_mask, thresholds, tactile_dim, workers, backend, chunksize):
    """Aggregate dense target bins without constructing image batches.

    The training/eval Dataset is intentionally not used for `__getitem__`: it
    decodes and warps an image for every sample, while this utility only needs
    the pressure array already stored in `meta.json`.
    """
    chunks = list(_chunked(sample_records, chunksize))
    bin_count = len(thresholds) + 1
    counts = np.zeros(bin_count, dtype=np.int64)
    sums = np.zeros(bin_count, dtype=np.float64)
    valid_samples = 0
    skipped_samples = 0

    _init_stats_worker(palm_mask, thresholds, tactile_dim)
    if workers <= 1:
        for chunk_index, chunk in enumerate(chunks, start=1):
            result = _summarize_record_chunk(chunk)
            counts += result[0]
            sums += result[1]
            valid_samples += result[2]
            skipped_samples += result[3]
            if chunk_index % 50 == 0 or chunk_index == len(chunks):
                print(f"Processed {chunk_index}/{len(chunks)} pressure chunks...", flush=True)
        return counts, sums, valid_samples, skipped_samples

    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    print(
        f"Computing pressure-bin statistics from {len(sample_records)} samples using "
        f"{workers} {backend} worker(s), chunksize={chunksize}...",
        flush=True,
    )
    with executor_cls(
        max_workers=workers,
        initializer=_init_stats_worker,
        initargs=(palm_mask, thresholds, tactile_dim),
    ) as executor:
        futures = [executor.submit(_summarize_record_chunk, chunk) for chunk in chunks]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts += result[0]
            sums += result[1]
            valid_samples += result[2]
            skipped_samples += result[3]
            if completed % 50 == 0 or completed == len(futures):
                print(f"Processed {completed}/{len(futures)} pressure chunks...", flush=True)
    return counts, sums, valid_samples, skipped_samples


def parse_float_csv(value, name):
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def load_model_cfg():
    model_cfg_path = WORKSPACE_DIR / "hamer" / "_DATA" / "hamer_ckpts" / "model_config.yaml"
    model_cfg = get_config(str(model_cfg_path), update_cachedir=True)
    if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        model_cfg.freeze()
    return model_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Compute dense-GT ordinal bin means from a training split."
    )
    parser.add_argument("--datasets", type=str, default="opentouch,touchanything")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ordinal_thresholds", type=str, default="0.005,0.02,0.05,0.1,0.2,0.4,0.7")
    parser.add_argument(
        "--output",
        type=str,
        default=str(FT_DIR / "ordinal_bin_values_train.json"),
        help="JSON output consumed by eval_tactile_fast.py --ordinal_bin_values_path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Deprecated compatibility option; statistics are aggregated from metadata chunks.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Workers for metadata-only pressure aggregation (not DataLoader workers).",
    )
    parser.add_argument("--stats_backend", type=str, default="process", choices=["process", "thread"])
    parser.add_argument("--stats_chunksize", type=int, default=256)
    parser.add_argument("--index_workers", type=int, default=8)
    parser.add_argument(
        "--max_index_workers",
        type=int,
        default=32,
        help="Safety cap for index scanning; shared storage is slower with hundreds of metadata processes.",
    )
    parser.add_argument("--index_backend", type=str, default="process", choices=["process", "thread"])
    parser.add_argument("--index_chunksize", type=int, default=512)
    parser.add_argument("--index_cache_dir", type=str, default=str(FT_DIR / "index_cache"))
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--index_cache_timeout", type=int, default=3600)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    thresholds = parse_float_csv(args.ordinal_thresholds, "ordinal_thresholds")
    if args.index_workers > args.max_index_workers:
        print(
            f"Capping --index_workers from {args.index_workers} to {args.max_index_workers}; "
            "the index scan is metadata-bound and more processes can stall a shared filesystem.",
            flush=True,
        )
        args.index_workers = args.max_index_workers
    args.num_workers = max(0, int(args.num_workers))
    args.stats_chunksize = max(1, int(args.stats_chunksize))
    data_dirs = resolve_data_dirs(args)
    model_cfg = load_model_cfg()
    dataset = OpenTouchTactileDataset(
        model_cfg,
        split=args.split,
        data_dir=data_dirs,
        train=False,
        index_workers=args.index_workers,
        index_chunksize=args.index_chunksize,
        index_backend=args.index_backend,
        index_cache_dir=args.index_cache_dir,
        rebuild_index=args.rebuild_index,
        index_cache_timeout=args.index_cache_timeout,
    )
    if args.max_samples is not None:
        dataset.samples = dataset.samples[: max(0, int(args.max_samples))]
    if len(dataset) == 0:
        raise RuntimeError("No samples found for ordinal bin statistics.")

    counts, sums, valid_samples, skipped_samples = compute_bin_statistics(
        sample_records=dataset.samples,
        palm_mask=dataset.palm_mask,
        thresholds=thresholds,
        tactile_dim=dataset.tactile_dim,
        workers=args.num_workers,
        backend=args.stats_backend,
        chunksize=args.stats_chunksize,
    )
    means = sums / np.maximum(counts, 1)
    payload = {
        "source_split": args.split,
        "resolved_data_dirs": data_dirs,
        "ordinal_thresholds": thresholds,
        "bin_edges": [0.0, *thresholds, 1.0],
        "bin_values": means.tolist(),
        "bin_counts": [int(value) for value in counts.tolist()],
        "valid_samples": int(valid_samples),
        "skipped_samples": int(skipped_samples),
        "statistics_backend": args.stats_backend,
        "statistics_workers": int(args.num_workers),
        "statistics_chunksize": int(args.stats_chunksize),
        "note": "Dense subdiv GT statistics only; no raw sensor-pressure map is used.",
    }
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = Path.cwd() / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Wrote ordinal bin values to: {output}")
    print("bin_values:", ",".join(f"{value:.8f}" for value in means.tolist()))


if __name__ == "__main__":
    main()
