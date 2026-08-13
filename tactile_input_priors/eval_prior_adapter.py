#!/usr/bin/env python3
"""Evaluate a tactile input-prior adapter against its frozen base."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

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
    if adapter_type == "depth_spatial" and not getattr(dataset, "depth_sidecar_root", None):
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
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-frame-csv", action="store_true")
    parser.add_argument(
        "--cache-only",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
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
    if world_size > 1:
        dataset = torch.utils.data.Subset(dataset, range(rank, len(dataset), world_size))
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs.update(prefetch_factor=2, multiprocessing_context="spawn")
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
    device = torch.device(f"cuda:{local_rank}" if world_size > 1 else args.device)
    model.to(device).eval()
    fused_stats = PriorMetricAccumulator()
    base_stats = PriorMetricAccumulator()
    fused_protocol = CompactTouchAnythingProtocolAccumulator()
    base_protocol = CompactTouchAnythingProtocolAccumulator()
    diagnostic_sums: dict[str, float] = {}
    diagnostic_count = 0
    frame_rows = []
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(tensor_batch, train=False)
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
    fused_stats.synchronize(device)
    base_stats.synchronize(device)
    if world_size > 1:
        fused_packs = [None for _ in range(world_size)]
        base_packs = [None for _ in range(world_size)]
        dist.all_gather_object(fused_packs, fused_protocol.pack())
        dist.all_gather_object(base_packs, base_protocol.pack())
        diagnostic_names = sorted(diagnostic_sums)
        all_names = [None for _ in range(world_size)]
        dist.all_gather_object(all_names, diagnostic_names)
        diagnostic_names = sorted({name for names in all_names for name in names})
        diagnostic_tensor = torch.tensor(
            [diagnostic_sums.get(name, 0.0) for name in diagnostic_names] + [diagnostic_count],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(diagnostic_tensor, op=dist.ReduceOp.SUM)
        diagnostic_count = int(diagnostic_tensor[-1].item())
        diagnostic_sums = {
            name: float(diagnostic_tensor[index].item())
            for index, name in enumerate(diagnostic_names)
        }
        if args.save_frame_csv:
            gathered_rows = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_rows, frame_rows)
            frame_rows = [row for rows in gathered_rows for row in rows]
    else:
        fused_packs = [fused_protocol.pack()]
        base_packs = [base_protocol.pack()]
    if rank != 0:
        dist.destroy_process_group()
        return
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
        )
    }
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    if frame_rows:
        with (output_dir / "frame_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    source_val_metrics = checkpoint_path.parent.parent / "val_metrics.csv"
    if source_val_metrics.is_file():
        shutil.copy2(source_val_metrics, output_dir / "val_metrics.csv")
    print(f"Prior evaluation report: {output_dir}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
