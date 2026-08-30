#!/usr/bin/env python3
"""Audit runtime Depth validity after applying the current SAM3 crop."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

cv2.setNumThreads(0)

from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal
from tactile_input_priors.runtime import build_dataset


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _identity(dataset, args) -> dict:
    payload = {
        "schema": "runtime_depth_sam3_crop_coverage_v1",
        "datasets": str(args.datasets),
        "split": str(args.split),
        "input_resolution": str(args.input_resolution),
        "bbox_rescale_factor": float(args.bbox_rescale_factor),
        "bbox_source_policy": str(args.bbox_source_policy),
        "query_manifest_sha256": dict(
            getattr(dataset, "query_manifest_sha256", {})
        ),
        "bbox_manifest_sha256": dict(
            getattr(dataset, "bbox_manifest_sha256", {})
        ),
        "depth_sidecar_contract": dict(
            getattr(dataset, "depth_sidecar_contract", {})
        ),
        "sample_count": int(len(dataset)),
        "max_samples": int(args.max_samples),
        "seed": int(args.seed),
    }
    payload["identity_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("ascii")
    ).hexdigest()
    return payload


def _indices(length: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or maximum >= length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(length, size=int(maximum), replace=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="touchanything")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-roots", default="")
    parser.add_argument("--query-manifests", default="")
    parser.add_argument("--bbox-manifests", default="")
    parser.add_argument("--bbox-source-policy", default="sam3_only")
    parser.add_argument("--bbox-rescale-factor", type=float, default=1.2)
    parser.add_argument("--input-resolution", default="256x192")
    parser.add_argument("--depth-sidecar-root", required=True)
    parser.add_argument("--hdf5-manifest-cache-dir", default="")
    parser.add_argument("--hdf5-handle-cache-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--low-coverage-threshold", type=float, default=0.50)
    parser.add_argument("--max-low-coverage-rate", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-if-current", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.low_coverage_threshold <= 1.0:
        raise ValueError("--low-coverage-threshold must be in [0,1]")
    if not 0.0 <= args.max_low_coverage_rate <= 1.0:
        raise ValueError("--max-low-coverage-rate must be in [0,1]")
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
        depth_sidecar_root=args.depth_sidecar_root,
        depth_output_hw=(16, 12),
        hdf5_handle_cache_size=args.hdf5_handle_cache_size,
        hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir or None,
    )
    identity = _identity(dataset, args)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    lock_path = output_dir / ".audit.lock"
    lock_handle = lock_path.open("a+")
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    if args.reuse_if_current and summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("identity", {}).get("identity_sha256") == identity["identity_sha256"]:
            if not bool(previous.get("passed", False)):
                raise RuntimeError(
                    f"Reused Depth crop coverage audit is failing: {summary_path}"
                )
            print(f"[depth-coverage] reuse current audit: {summary_path}", flush=True)
            return

    selected = _indices(len(dataset), args.max_samples, args.seed)
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": False,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    if args.num_workers:
        loader_kwargs.update(prefetch_factor=1, persistent_workers=False)
    loader = DataLoader(Subset(dataset, selected.tolist()), **loader_kwargs)
    rows = []
    processed = 0
    for batch in loader:
        depth = batch["depth_prior"].float()
        if depth.ndim != 4 or depth.shape[1] < 5:
            raise ValueError(f"Expected Depth point-normal [B,8,H,W], got {tuple(depth.shape)}")
        valid = depth[:, 4].clamp(0.0, 1.0)
        border = torch.cat(
            (valid[:, 0], valid[:, -1], valid[:, 1:-1, 0], valid[:, 1:-1, -1]),
            dim=1,
        )
        coverage = valid.mean(dim=(1, 2))
        border_coverage = border.mean(dim=1)
        batch_size = int(valid.shape[0])
        for offset in range(batch_size):
            rows.append(
                {
                    "sample_uid": str(batch["sample_uid"][offset]),
                    "sequence_key": str(batch["sequence_key"][offset]),
                    "frame_idx": int(batch["frame_idx"][offset]),
                    "valid_fraction": float(coverage[offset]),
                    "border_valid_fraction": float(border_coverage[offset]),
                }
            )
        processed += batch_size
        if processed % 1024 < batch_size:
            print(
                f"[depth-coverage] {processed}/{len(selected)}",
                flush=True,
            )

    values = np.asarray([row["valid_fraction"] for row in rows], dtype=np.float64)
    low = values < float(args.low_coverage_threshold)
    low_rate = float(low.mean()) if values.size else 1.0
    passed = bool(values.size) and low_rate <= float(args.max_low_coverage_rate)
    summary = {
        "identity": identity,
        "evaluated_samples": int(values.size),
        "valid_fraction_mean": float(values.mean()) if values.size else float("nan"),
        "valid_fraction_p01": float(np.quantile(values, 0.01)) if values.size else float("nan"),
        "valid_fraction_p05": float(np.quantile(values, 0.05)) if values.size else float("nan"),
        "valid_fraction_min": float(values.min()) if values.size else float("nan"),
        "low_coverage_threshold": float(args.low_coverage_threshold),
        "low_coverage_count": int(low.sum()),
        "low_coverage_rate": low_rate,
        "max_low_coverage_rate": float(args.max_low_coverage_rate),
        "passed": passed,
        "interpretation": (
            "Validity is measured after current SAM3 crop reprojection and includes "
            "both teacher validity and crop coverage. Low values therefore require "
            "inspection before training."
        ),
    }
    csv_path = output_dir / "lowest_coverage_samples.csv"
    csv_tmp = output_dir / f".{csv_path.name}.{os.getpid()}.tmp"
    with csv_tmp.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["sample_uid"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["valid_fraction"])[:500])
    os.replace(csv_tmp, csv_path)
    summary_tmp = output_dir / f".{summary_path.name}.{os.getpid()}.tmp"
    summary_tmp.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(summary_tmp, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError(
            "Current SAM3 crops have insufficient aligned Depth coverage; "
            f"inspect {output_dir / 'lowest_coverage_samples.csv'}"
        )


if __name__ == "__main__":
    main()
