#!/usr/bin/env python3
"""Audit batch-to-batch gradient conflicts within one tactile dataset."""

import argparse
import csv
import json
import math
import os
import statistics
from dataclasses import fields
from itertools import combinations
from pathlib import Path

import audit_gradient_conflicts as base
import torch
from tqdm import tqdm
from losses import TactileLossConfig


FT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="mixed_zero_ordinal_residual_v19_condnll")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default=str(FT_DIR / "checkpoints"))
    parser.add_argument("--ckpt", choices=("rmse-best", "viou-best", "last", "best"), default="last")
    parser.add_argument("--tactile_head_type", default="zero_ordinal_residual")
    parser.add_argument("--domain", default="touchanything")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--gpu", "--gpus", dest="gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--index_workers", type=int, default=32)
    parser.add_argument("--index_chunksize", type=int, default=512)
    parser.add_argument("--index_backend", choices=("process", "thread"), default="process")
    parser.add_argument("--index_cache_dir", default=str(FT_DIR / "index_cache"))
    parser.add_argument("--index_cache_timeout", type=int, default=3600)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--sketch_size", type=int, default=262144)
    parser.add_argument("--strong_conflict_threshold", type=float, default=-0.2)
    parser.add_argument("--frame_low_volume_thr", type=float, default=30.0)
    parser.add_argument("--frame_high_volume_thr", type=float, default=150.0)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--progress_position", type=int, default=0)
    parser.add_argument("--disable_progress", action="store_true")
    return parser.parse_args()


def volume_bin(value, args):
    if value <= args.frame_low_volume_thr:
        return "low"
    if value < args.frame_high_volume_thr:
        return "mid"
    return "high"


def assign_terciles(records):
    ordered = sorted(records, key=lambda row: row["mean_gt_volume"])
    count = len(ordered)
    for rank, row in enumerate(ordered):
        row["volume_tercile"] = ("lower", "middle", "upper")[min(2, 3 * rank // max(count, 1))]


def gradient_signature(vector, indices):
    signature = vector[indices].float()
    finite = bool(torch.isfinite(signature).all().item())
    if not finite:
        return signature, False
    norm = torch.linalg.vector_norm(signature)
    if float(norm) > 0.0:
        signature = signature / norm
    return signature.to(torch.float16), True


def pair_relation(left, right, key):
    return "same" if left[key] == right[key] else "different"


def summarize(rows, threshold):
    grouped = {}
    for row in rows:
        keys = (
            (row["loss"], "all", "all"),
            (row["loss"], row["fixed_volume_relation"], "all"),
            (row["loss"], "all", row["tercile_relation"]),
            (row["loss"], row["fixed_volume_relation"], row["tercile_relation"]),
        )
        for key in keys:
            grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        values = [row["cosine"] for row in group if row["finite"] and math.isfinite(row["cosine"])]
        output.append({
            "loss": key[0],
            "fixed_volume_relation": key[1],
            "tercile_relation": key[2],
            "count": len(group),
            "finite_count": len(values),
            "nonfinite_rate": 1.0 - len(values) / max(len(group), 1),
            "cosine_mean": statistics.fmean(values) if values else math.nan,
            "cosine_median": statistics.median(values) if values else math.nan,
            "cosine_std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else math.nan,
            "cosine_min": min(values) if values else math.nan,
            "cosine_max": max(values) if values else math.nan,
            "negative_rate": sum(value < 0.0 for value in values) / max(len(values), 1),
            "strong_conflict_rate": sum(value < threshold for value in values) / max(len(values), 1),
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.num_batches < 2 or args.batch_size < 1:
        raise ValueError("--num_batches must be >= 2 and --batch_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This audit requires a CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    print(f"[1b 1/4] Loading {args.ckpt} checkpoint and model...", flush=True)
    checkpoint_path, exp_dir = base.resolve_checkpoint(args)
    model, loss_config, model_config = base.load_model(args, checkpoint_path, exp_dir, device)
    cfg = base.load_model_cfg()
    print(f"[1b 2/4] Building {args.domain}/{args.split} index and data loader...", flush=True)
    loader, data_dir, sample_count = base.make_domain_loader(
        args,
        cfg,
        args.domain,
        args.data_dir,
        args.seed,
    )
    print(
        f"[1b 3/4] Computing {args.num_batches} gradient batches on GPU {args.gpu}...",
        flush=True,
    )

    shared_count = sum(parameter.numel() for parameter in model.tactile_head.encoder.parameters())
    sketch_size = min(max(1, args.sketch_size), shared_count)
    generator = torch.Generator().manual_seed(args.seed + 99)
    sketch_indices = torch.randperm(shared_count, generator=generator)[:sketch_size]

    records = []
    iterator = iter(loader)
    progress = tqdm(
        range(args.num_batches),
        desc=f"1b {args.ckpt} {args.domain}",
        position=args.progress_position,
        dynamic_ncols=True,
        disable=args.disable_progress,
    )
    for batch_index in progress:
        gradients, diagnostics = base.domain_gradients(
            model,
            next(iterator),
            loss_config,
            args,
            device,
        )
        total = sum((gradients[name]["shared"] for name in base.LOSS_NAMES), start=torch.zeros(shared_count))
        signatures = {}
        finite = {}
        for name in (*base.LOSS_NAMES, "total"):
            vector = total if name == "total" else gradients[name]["shared"]
            signatures[name], finite[name] = gradient_signature(vector, sketch_indices)
        record = {
            "batch_index": batch_index,
            "mean_gt_volume": diagnostics["mean_gt_volume"],
            "mean_gt_pressure": diagnostics["mean_gt_pressure"],
            "positive_fraction": diagnostics["positive_fraction"],
            "active_fraction": diagnostics["active_fraction"],
            "fixed_volume_bin": volume_bin(diagnostics["mean_gt_volume"], args),
            "signatures": signatures,
            "finite": finite,
        }
        for name in base.LOSS_NAMES:
            record[f"{name}_loss"] = gradients[name]["loss"]
            record[f"{name}_grad_norm"] = base.vector_stats(gradients[name]["shared"])[1]
        record["total_grad_norm"] = base.vector_stats(total)[1]
        records.append(record)
        progress.set_postfix(
            volume=f"{diagnostics['mean_gt_volume']:.1f}",
            total_grad=f"{record['total_grad_norm']:.3g}",
            refresh=False,
        )
        del gradients, total
        torch.cuda.empty_cache()

    print("[1b 4/4] Computing pairwise cosines and writing results...", flush=True)
    assign_terciles(records)
    batch_rows = []
    for record in records:
        batch_rows.append({key: value for key, value in record.items() if key not in ("signatures", "finite")})

    pair_rows = []
    for left, right in combinations(records, 2):
        for loss_name in (*base.LOSS_NAMES, "total"):
            is_finite = left["finite"][loss_name] and right["finite"][loss_name]
            cosine = (
                float(torch.dot(left["signatures"][loss_name].float(), right["signatures"][loss_name].float()))
                if is_finite else math.nan
            )
            pair_rows.append({
                "left_batch": left["batch_index"],
                "right_batch": right["batch_index"],
                "loss": loss_name,
                "cosine": max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else cosine,
                "finite": is_finite,
                "left_volume": left["mean_gt_volume"],
                "right_volume": right["mean_gt_volume"],
                "left_fixed_volume_bin": left["fixed_volume_bin"],
                "right_fixed_volume_bin": right["fixed_volume_bin"],
                "fixed_volume_relation": pair_relation(left, right, "fixed_volume_bin"),
                "left_tercile": left["volume_tercile"],
                "right_tercile": right["volume_tercile"],
                "tercile_relation": pair_relation(left, right, "volume_tercile"),
                "left_positive_fraction": left["positive_fraction"],
                "right_positive_fraction": right["positive_fraction"],
            })

    summary_rows = summarize(pair_rows, args.strong_conflict_threshold)
    selector = "rmse-best" if args.ckpt == "best" else args.ckpt
    output_dir = Path(args.output_dir) if args.output_dir else (
        FT_DIR / "gradient_subdomain_audits" / f"{base.safe_name(args.exp_name)}_{selector}_{base.safe_name(args.domain)}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "batch_statistics.csv", batch_rows)
    write_csv(output_dir / "batch_gradient_pairs.csv", pair_rows)
    write_csv(output_dir / "batch_gradient_summary.csv", summary_rows)

    config = {
        "args": vars(args),
        "checkpoint": str(checkpoint_path),
        "data_dir": data_dir,
        "dataset_sample_count": sample_count,
        "shared_parameter_count": shared_count,
        "gradient_sketch_size": sketch_size,
        "gradient_sketch_fraction": sketch_size / shared_count,
        "loss_config": {field.name: getattr(loss_config, field.name) for field in fields(TactileLossConfig)},
        "model_config": model_config,
    }
    with (output_dir / "audit_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)

    print("\nBatch-to-batch gradient summary:")
    for row in summary_rows:
        if row["loss"] != "total":
            continue
        print(
            f"  fixed={row['fixed_volume_relation']:9s} tercile={row['tercile_relation']:9s} "
            f"cos={row['cosine_mean']:+.4f} neg={row['negative_rate']:.1%} "
            f"strong={row['strong_conflict_rate']:.1%}"
        )
    print(f"Wrote subdomain gradient audit to: {output_dir}")


if __name__ == "__main__":
    main()
