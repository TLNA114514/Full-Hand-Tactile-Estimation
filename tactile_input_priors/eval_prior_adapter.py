#!/usr/bin/env python3
"""Evaluate a tactile input-prior adapter against its frozen base."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    def record(function):
        return function

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactile_input_priors.runtime import (
    CachedFeatureDataset,
    FeatureOnlyTactileDataset,
    build_dataset,
    load_prior_checkpoint,
    parse_csv,
)
from tactile_input_priors.prior_metrics import PriorMetricAccumulator

from hamer_tactile_ft.tactile_metrics import (
    CompactTouchAnythingProtocolAccumulator,
    summarize_compact_touchanything_protocol,
    touchanything_protocol_frame_stats,
    touchanything_protocol_group_key,
)
from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal


faulthandler.enable(all_threads=True)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _evaluation_run_token(args) -> str:
    elastic_id = os.environ.get("TORCHELASTIC_RUN_ID", "single")
    restart = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    identity = "|".join(
        (
            elastic_id,
            restart,
            str(Path(args.checkpoint).expanduser().resolve(strict=False)),
            str(Path(args.output_dir).expanduser().resolve(strict=False)),
            str(args.split),
            str(args.prior_control),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", elastic_id).strip("._-")
    return f"{(readable or 'eval')[:48]}-{restart}-{digest}"


class EvaluationProgress:
    """Low-traffic rank status files plus one rank-zero console progress line."""

    def __init__(
        self,
        *,
        shard_dir: Path,
        rank: int,
        world_size: int,
        local_total: int,
        global_total: int,
        interval: float,
    ) -> None:
        self.shard_dir = Path(shard_dir)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.local_total = int(local_total)
        self.global_total = int(global_total)
        self.interval = max(float(interval), 0.5)
        self.started = time.monotonic()
        self.last_update = 0.0
        self.processed = 0
        self.batches = 0
        self.status_path = self.shard_dir / f"progress_rank_{self.rank:02d}.json"

    def _write_status(self, state: str) -> None:
        elapsed = max(time.monotonic() - self.started, 1e-6)
        _atomic_json(
            self.status_path,
            {
                "rank": self.rank,
                "state": state,
                "processed_samples": self.processed,
                "local_total": self.local_total,
                "batches": self.batches,
                "elapsed_seconds": elapsed,
                "samples_per_second": self.processed / elapsed,
                "updated_unix": time.time(),
            },
        )

    def _global_processed(self) -> tuple[int, int]:
        processed = 0
        visible = 0
        for current_rank in range(self.world_size):
            path = self.shard_dir / f"progress_rank_{current_rank:02d}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                processed += int(payload.get("processed_samples", 0))
                visible += 1
            except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if visible < self.world_size:
            processed = max(processed, min(self.global_total, self.processed * self.world_size))
        return min(processed, self.global_total), visible

    def update(self, batch_size: int, *, force: bool = False, state: str = "running") -> None:
        batch_size = int(batch_size)
        self.processed += batch_size
        if batch_size > 0:
            self.batches += 1
        now = time.monotonic()
        if not force and now - self.last_update < self.interval:
            return
        self.last_update = now
        self._write_status(state)
        if self.rank != 0:
            return
        global_processed, visible = self._global_processed()
        elapsed = max(now - self.started, 1e-6)
        rate = global_processed / elapsed
        remaining = max(self.global_total - global_processed, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        percent = 100.0 * global_processed / max(self.global_total, 1)
        eta_text = time.strftime("%H:%M:%S", time.gmtime(eta)) if np.isfinite(eta) else "--:--:--"
        print(
            f"\rEval {percent:6.2f}% | {global_processed}/{self.global_total} samples "
            f"| {rate:7.1f} samples/s | ETA {eta_text} | ranks {visible}/{self.world_size}",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        self.update(0, force=True, state="complete")
        if self.rank == 0:
            print(flush=True)


def _choose(cli_value, config, key, default):
    return cli_value if cli_value not in (None, "") else config.get(key, default)


def _build_eval_dataset(args, data_config, input_resolution, adapter_type):
    base_cache_value = args.base_feature_cache or (
        data_config.get("val_base_feature_cache", "")
        if args.split == "val"
        else data_config.get("base_feature_cache", "")
    )
    prior_cache_value = args.prior_feature_cache or (
        data_config.get("val_prior_feature_cache", "")
        if args.split == "val"
        else data_config.get("prior_feature_cache", "")
    )
    cache_only_requested = bool(
        data_config.get("cache_only", False)
        if args.cache_only is None
        else args.cache_only
    )
    cache_only = cache_only_requested and bool(parse_csv(base_cache_value))
    if cache_only:
        return FeatureOnlyTactileDataset(
            (*parse_csv(base_cache_value), *parse_csv(prior_cache_value)),
            adapter_type=adapter_type,
        )
    dataset = build_dataset(
        split=args.split,
        datasets=_choose(args.datasets, data_config, "datasets", "touchanything"),
        input_resolution=input_resolution,
        bbox_rescale_factor=float(
            _choose(args.bbox_rescale_factor, data_config, "bbox_rescale_factor", 1.2)
        ),
        train=False,
        augmentation_enabled=False,
        data_roots=_choose(args.data_roots, data_config, "data_roots", ""),
        query_manifests=args.query_manifests,
        bbox_manifests=_choose(args.bbox_manifests, data_config, "bbox_manifests", ""),
        bbox_source_policy=_choose(
            args.bbox_source_policy, data_config, "bbox_source_policy", "sam3_only"
        ),
        depth_sidecar_root=_choose(
            args.depth_sidecar_root, data_config, "depth_sidecar_root", ""
        )
        or None,
        depth_output_hw=(16, 12),
        hdf5_handle_cache_size=args.hdf5_handle_cache_size,
        hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir or None,
    )
    caches = (
        *parse_csv(base_cache_value),
        *parse_csv(prior_cache_value),
    )
    required = ("vlm_embedding",) if adapter_type == "vlm_lowrank" else ()
    if adapter_type.startswith("depth_") and not getattr(dataset, "depth_sidecar_root", None):
        required = ("depth_grid",)
    if parse_csv(base_cache_value):
        base_required = (
            ("h_rgb",)
            if adapter_type == "vlm_lowrank"
            else ("z_rgb",)
        )
        required = tuple(dict.fromkeys((*required, *base_required)))
    if caches:
        dataset = CachedFeatureDataset(dataset, caches, require_fields=required)
    elif required:
        raise ValueError(f"Missing required cache fields for adapter_type={adapter_type}")
    return dataset


def _valid_batch(batch, output):
    valid = batch["has_tactile"].detach().cpu().numpy().reshape(-1) > 0.5
    if not np.any(valid):
        return None
    prediction = output["pred_tactile"].detach().float().cpu().numpy()[valid]
    base = output["base_pred_tactile"].detach().float().cpu().numpy()[valid]
    target = batch["tactile_signal"].detach().float().cpu().numpy()[valid]
    records = []
    for index in np.flatnonzero(valid):
        records.append(
            {
                "dataset": batch.get("dataset", [""] * len(valid))[index],
                "sequence_key": batch.get("sequence_key", [""] * len(valid))[index],
                "query_alias": batch.get("query_alias", [""] * len(valid))[index],
                "frame_idx": int(batch["frame_idx"][index]),
                "sample_uid": batch.get("sample_uid", [""] * len(valid))[index],
            }
        )
    palm = batch["palm_mask"].detach().cpu().numpy()
    if palm.ndim == 2:
        palm = palm[0]
    return prediction, base, target, palm > 0.5, records


def _update_protocol(accumulator, prediction, target, palm, records):
    touch_indices = [
        index
        for index, record in enumerate(records)
        if str(record["dataset"]).casefold() in ("touchanything", "egotouch", "ta")
    ]
    if not touch_indices:
        return
    pred_palm = prediction[:, palm][touch_indices]
    target_palm = target[:, palm][touch_indices]
    frame_stats = touchanything_protocol_frame_stats(
        pred_palm,
        target_palm,
        value_axis=1,
        contact_threshold=0.10,
    )
    accumulator.add(
        [
            touchanything_protocol_group_key(
                records[index]["sequence_key"], records[index]["query_alias"]
            )
            for index in touch_indices
        ],
        [records[index]["frame_idx"] for index in touch_indices],
        frame_stats,
    )


def _format_summary(summary: dict[str, Any]) -> str:
    keys = (
        "mae",
        "rmse",
        "contact_iou",
        "volumetric_iou",
        "core_distribution_viou",
        "pred_gt_volume_ratio",
        "false_high_excess_fraction",
        "catastrophic_over_rate",
        "catastrophic_under_rate",
    )
    lines = []
    for key in keys:
        value = summary.get(key)
        if value is not None:
            lines.append(f"{key}: {value:.8f}")
    protocol = summary.get("touchanything_protocol", {})
    for key in ("contact_iou", "volumetric_iou", "temporal_accuracy"):
        if key in protocol:
            lines.append(f"touchanything_{key}: {protocol[key]:.8f}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--query-manifests",
        default="",
        help="Optional; auto-detected from the processed HDF5 root when omitted.",
    )
    parser.add_argument("--dino-weights", default="")
    parser.add_argument("--base-checkpoint", default="")
    parser.add_argument("--prior-control", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--data-roots", default="")
    parser.add_argument("--bbox-manifests", default="")
    parser.add_argument("--bbox-source-policy", default="")
    parser.add_argument("--bbox-rescale-factor", type=float)
    parser.add_argument("--depth-sidecar-root", default="")
    parser.add_argument("--base-feature-cache", default="")
    parser.add_argument("--prior-feature-cache", default="")
    parser.add_argument("--hdf5-handle-cache-size", type=int, default=4)
    parser.add_argument("--hdf5-manifest-cache-dir", default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-frame-csv", action="store_true")
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--rank-merge-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--cache-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _write_rank_result(
    shard_dir: Path,
    rank: int,
    fused_stats: PriorMetricAccumulator,
    base_stats: PriorMetricAccumulator,
    fused_protocol: CompactTouchAnythingProtocolAccumulator,
    base_protocol: CompactTouchAnythingProtocolAccumulator,
    diagnostic_sums: dict[str, float],
    diagnostic_count: int,
    frame_rows: list[dict[str, Any]],
    save_frame_csv: bool,
) -> None:
    fused_pack = fused_protocol.pack()
    base_pack = base_protocol.pack()
    prefix = shard_dir / f"rank_{rank:02d}"
    _atomic_npz(
        prefix.with_suffix(".npz"),
        fused_metric_values=fused_stats.values.numpy(),
        base_metric_values=base_stats.values.numpy(),
        fused_sequence_hashes=fused_pack["sequence_hashes"],
        fused_frame_indices=fused_pack["frame_indices"],
        fused_frame_stats=fused_pack["frame_stats"],
        base_sequence_hashes=base_pack["sequence_hashes"],
        base_frame_indices=base_pack["frame_indices"],
        base_frame_stats=base_pack["frame_stats"],
    )
    _atomic_json(
        prefix.with_suffix(".json"),
        {
            "rank": int(rank),
            "fused_sequence_keys": {
                str(key): value for key, value in fused_pack["sequence_keys"].items()
            },
            "base_sequence_keys": {
                str(key): value for key, value in base_pack["sequence_keys"].items()
            },
            "diagnostic_sums": diagnostic_sums,
            "diagnostic_count": int(diagnostic_count),
            "frame_row_count": len(frame_rows),
        },
    )
    if save_frame_csv:
        _atomic_csv(prefix.with_name(f"{prefix.name}_frames.csv"), frame_rows)
    _atomic_json(prefix.with_suffix(".done"), {"rank": int(rank), "complete": True})


def _wait_for_rank_results(shard_dir: Path, world_size: int, timeout: float) -> None:
    started = time.monotonic()
    next_message = 0.0
    while True:
        missing = [
            rank
            for rank in range(world_size)
            if not (shard_dir / f"rank_{rank:02d}.done").is_file()
        ]
        if not missing:
            return
        elapsed = time.monotonic() - started
        if elapsed >= float(timeout):
            raise TimeoutError(
                f"Timed out waiting for evaluation rank shards after {elapsed:.1f}s; "
                f"missing ranks={missing}, shard_dir={shard_dir}"
            )
        if elapsed >= next_message:
            print(
                f"Waiting for evaluation rank shards: missing={missing}, "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )
            next_message = elapsed + 10.0
        time.sleep(0.5)


def _read_rank_results(shard_dir: Path, world_size: int, save_frame_csv: bool):
    fused_values = None
    base_values = None
    fused_packs = []
    base_packs = []
    diagnostic_sums: dict[str, float] = {}
    diagnostic_count = 0
    frame_paths = []
    for rank in range(world_size):
        prefix = shard_dir / f"rank_{rank:02d}"
        metadata = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as arrays:
            current_fused = np.asarray(
                arrays["fused_metric_values"], dtype=np.float64
            ).copy()
            current_base = np.asarray(
                arrays["base_metric_values"], dtype=np.float64
            ).copy()
            fused_values = current_fused if fused_values is None else fused_values + current_fused
            base_values = current_base if base_values is None else base_values + current_base
            fused_packs.append(
                {
                    "sequence_hashes": np.asarray(arrays["fused_sequence_hashes"]).copy(),
                    "frame_indices": np.asarray(arrays["fused_frame_indices"]).copy(),
                    "frame_stats": np.asarray(arrays["fused_frame_stats"]).copy(),
                    "sequence_keys": metadata["fused_sequence_keys"],
                }
            )
            base_packs.append(
                {
                    "sequence_hashes": np.asarray(arrays["base_sequence_hashes"]).copy(),
                    "frame_indices": np.asarray(arrays["base_frame_indices"]).copy(),
                    "frame_stats": np.asarray(arrays["base_frame_stats"]).copy(),
                    "sequence_keys": metadata["base_sequence_keys"],
                }
            )
        for name, value in metadata.get("diagnostic_sums", {}).items():
            diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + float(value)
        diagnostic_count += int(metadata.get("diagnostic_count", 0))
        if save_frame_csv:
            frame_paths.append(prefix.with_name(f"{prefix.name}_frames.csv"))
    if fused_values is None or base_values is None:
        raise RuntimeError(f"No evaluation rank results were found under {shard_dir}")
    return fused_values, base_values, fused_packs, base_packs, diagnostic_sums, diagnostic_count, frame_paths


def _merge_frame_csvs(paths: list[Path], output_path: Path) -> None:
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    wrote_header = False
    with temporary.open("w", encoding="utf-8", newline="") as destination:
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                continue
            with path.open("r", encoding="utf-8", newline="") as source:
                header = source.readline()
                if not header:
                    continue
                if not wrote_header:
                    destination.write(header)
                    wrote_header = True
                shutil.copyfileobj(source, destination)
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, output_path)


@record
def main() -> None:
    args = build_parser().parse_args()
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1")
    if args.progress_interval <= 0 or args.rank_merge_timeout <= 0:
        raise ValueError("Progress interval and rank merge timeout must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested_device = torch.device(
        f"cuda:{local_rank}" if world_size > 1 else args.device
    )
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA evaluation requested but CUDA is unavailable: {requested_device}")
        torch.cuda.set_device(requested_device.index or 0)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / ".eval_shards" / _evaluation_run_token(args)
    shard_dir.mkdir(parents=True, exist_ok=True)
    model, payload, _ = load_prior_checkpoint(
        args.checkpoint,
        dino_weights_override=args.dino_weights or None,
        base_checkpoint_override=args.base_checkpoint or None,
    )
    if args.prior_control:
        model.default_control = args.prior_control
    data_config = dict(payload.get("data_config", {}))
    input_resolution = tuple(model.input_resolution)
    dataset = _build_eval_dataset(
        args, data_config, input_resolution, model.adapter_type
    )
    eval_base_cache = args.base_feature_cache or (
        data_config.get("val_base_feature_cache", "")
        if args.split == "val"
        else data_config.get("base_feature_cache", "")
    )
    if parse_csv(eval_base_cache):
        model.disable_online_backbone()
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    global_sample_count = len(dataset)
    if world_size > 1:
        dataset = torch.utils.data.Subset(dataset, range(rank, len(dataset), world_size))
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            multiprocessing_context="spawn",
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=initialize_worker_parent_death_signal,
        **loader_kwargs,
    )
    device = requested_device
    model.to(device).eval()
    fused_stats = PriorMetricAccumulator()
    base_stats = PriorMetricAccumulator()
    fused_protocol = CompactTouchAnythingProtocolAccumulator()
    base_protocol = CompactTouchAnythingProtocolAccumulator()
    diagnostic_sums: dict[str, float] = {}
    diagnostic_count = 0
    frame_rows = []
    progress = EvaluationProgress(
        shard_dir=shard_dir,
        rank=rank,
        world_size=world_size,
        local_total=len(dataset),
        global_total=global_sample_count,
        interval=args.progress_interval,
    )
    progress.update(0, force=True)
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(tensor_batch, train=False)
            progress.update(int(batch["tactile_signal"].shape[0]))
            fused_stats.update(
                output["pred_tactile"],
                tensor_batch["tactile_signal"],
                tensor_batch["palm_mask"],
                tensor_batch["has_tactile"],
            )
            base_stats.update(
                output["base_pred_tactile"],
                tensor_batch["tactile_signal"],
                tensor_batch["palm_mask"],
                tensor_batch["has_tactile"],
            )
            resolved = _valid_batch(batch, output)
            if resolved is None:
                continue
            prediction, base, target, palm, records = resolved
            _update_protocol(fused_protocol, prediction, target, palm, records)
            _update_protocol(base_protocol, base, target, palm, records)
            for name, value in output.get("prior_diagnostics", {}).items():
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + float(value)
            diagnostic_count += 1
            if args.save_frame_csv:
                for index, record in enumerate(records):
                    frame_rows.append(
                        {
                            **record,
                            "gt_volume": float(target[index, palm].sum()),
                            "base_volume": float(base[index, palm].sum()),
                            "fused_volume": float(prediction[index, palm].sum()),
                        }
                    )
    progress.finish()
    _write_rank_result(
        shard_dir,
        rank,
        fused_stats,
        base_stats,
        fused_protocol,
        base_protocol,
        diagnostic_sums,
        diagnostic_count,
        frame_rows,
        args.save_frame_csv,
    )
    if rank != 0:
        return
    _wait_for_rank_results(shard_dir, world_size, args.rank_merge_timeout)
    (
        fused_values,
        base_values,
        fused_packs,
        base_packs,
        diagnostic_sums,
        diagnostic_count,
        frame_paths,
    ) = _read_rank_results(shard_dir, world_size, args.save_frame_csv)
    fused_stats.values.copy_(torch.from_numpy(fused_values))
    base_stats.values.copy_(torch.from_numpy(base_values))
    fused_summary = fused_stats.summary()
    base_summary = base_stats.summary()
    if fused_summary["frame_count"] <= 0 or base_summary["frame_count"] <= 0:
        raise RuntimeError("Evaluation produced no valid tactile frames")
    fused_summary["touchanything_protocol"] = summarize_compact_touchanything_protocol(
        fused_packs, include_rows=False
    )
    base_summary["touchanything_protocol"] = summarize_compact_touchanything_protocol(
        base_packs, include_rows=False
    )
    differences = {
        key: float(fused_summary[key] - base_summary[key])
        for key in (
            "rmse",
            "contact_iou",
            "volumetric_iou",
            "core_distribution_viou",
            "false_high_excess_fraction",
            "catastrophic_over_rate",
            "catastrophic_under_rate",
        )
    }
    report = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "adapter_type": model.adapter_type,
        "prior_control": model.default_control,
        "split": args.split,
        "fused": fused_summary,
        "base": base_summary,
        "fused_minus_base": differences,
        "prior_diagnostics": {
            key: value / max(diagnostic_count, 1)
            for key, value in diagnostic_sums.items()
        },
        "adapter_config": payload.get("adapter_config", {}),
        "data_config": data_config,
        "evaluation_runtime": {
            "aggregation": "atomic_rank_shards",
            "world_size": world_size,
            "batch_size_per_rank": args.batch_size,
            "num_workers_per_rank": args.num_workers,
            "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else 0,
            "sample_count": global_sample_count,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "eval.txt").write_text(
        "FUSED\n" + _format_summary(fused_summary) + "\n\nBASE\n"
        + _format_summary(base_summary)
        + "\n\nFUSED_MINUS_BASE\n"
        + "\n".join(f"{key}: {value:+.8f}" for key, value in differences.items())
        + "\n",
        encoding="utf-8",
    )
    if args.save_frame_csv:
        _merge_frame_csvs(frame_paths, output_dir / "frame_metrics.csv")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    source_val_metrics = checkpoint_path.parent.parent / "val_metrics.csv"
    if source_val_metrics.is_file():
        shutil.copy2(source_val_metrics, output_dir / "val_metrics.csv")
    print(f"Prior evaluation report: {output_dir}")
    shutil.rmtree(shard_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
