#!/usr/bin/env python3
"""Audit realistic-batch gradients across explicit pressure-volume strata."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import audit_gradient_conflicts as base
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import OpenTouchTactileDataset


FT_DIR = Path(__file__).resolve().parent
STRATA = ("empty", "low", "mid", "high")
VOLUME_PALM_MASK = None
VOLUME_TACTILE_DIM = None


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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batches_per_stratum", type=int, default=16)
    parser.add_argument("--aggregate_batches", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--index_workers", type=int, default=32)
    parser.add_argument("--index_chunksize", type=int, default=512)
    parser.add_argument("--index_backend", choices=("process", "thread"), default="process")
    parser.add_argument("--index_cache_dir", default=str(FT_DIR / "index_cache"))
    parser.add_argument("--index_cache_timeout", type=int, default=3600)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--volume_workers", type=int, default=32)
    parser.add_argument("--volume_backend", choices=("process", "thread"), default="process")
    parser.add_argument("--volume_scan_chunk_size", type=int, default=256)
    parser.add_argument("--volume_scan_round_size", type=int, default=16384)
    parser.add_argument("--empty_volume_max", type=float, default=1.0)
    parser.add_argument("--low_volume_max", type=float, default=30.0)
    parser.add_argument("--high_volume_min", type=float, default=150.0)
    parser.add_argument("--sketch_size", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--strong_conflict_threshold", type=float, default=-0.2)
    parser.add_argument("--strata_manifest", default=None)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--progress_position", type=int, default=0)
    parser.add_argument("--disable_progress", action="store_true")
    return parser.parse_args()


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def volume_stratum(volume, args):
    if volume <= args.empty_volume_max:
        return "empty"
    if volume <= args.low_volume_max:
        return "low"
    if volume < args.high_volume_min:
        return "mid"
    return "high"


def _init_volume_worker(palm_mask, tactile_dim):
    global VOLUME_PALM_MASK, VOLUME_TACTILE_DIM
    VOLUME_PALM_MASK = np.asarray(palm_mask, dtype=bool)
    VOLUME_TACTILE_DIM = int(tactile_dim)


def _pressure_from_record(record):
    try:
        with open(os.path.join(record["sample_dir"], "meta.json"), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    dataset_name = record.get("dataset", meta.get("dataset", "OpenTouch"))
    if dataset_name == "TouchAnything":
        pressure = meta.get("hands", {}).get(record.get("hand", ""), {}).get("gaussian_pressure")
    else:
        is_right = int(record.get("is_right", meta.get("is_right", 1)))
        side = "right" if is_right else "left"
        pressure = meta.get("original_hdf5_data", {}).get(f"{side}_pressure_continuous_subdiv")
        if pressure is None:
            pressure = meta.get("gaussian_pressure")
    if pressure is None:
        return None
    values = np.asarray(pressure, dtype=np.float32)
    if values.shape != (VOLUME_TACTILE_DIM,):
        return None
    return float(np.clip(values[VOLUME_PALM_MASK], 0.0, 1.0).sum(dtype=np.float64))


def _scan_volume_chunk(items):
    output = []
    for index, record in items:
        volume = _pressure_from_record(record)
        if volume is not None and math.isfinite(volume):
            output.append((index, volume))
    return output, len(items) - len(output)


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def default_manifest_path(args, data_dir):
    payload = {
        "domain": args.domain,
        "split": args.split,
        "data_dir": str(Path(data_dir).resolve()),
        "batch_size": args.batch_size,
        "batches_per_stratum": args.batches_per_stratum,
        "thresholds": [args.empty_volume_max, args.low_volume_max, args.high_volume_min],
        "seed": args.seed,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path(args.index_cache_dir) / f"gradient_strata_{args.domain}_{args.split}_{digest}.json"


def validate_manifest(payload, args, data_dir):
    expected = {
        "domain": args.domain,
        "split": args.split,
        "data_dir": str(Path(data_dir).resolve()),
        "batch_size": args.batch_size,
        "batches_per_stratum": args.batches_per_stratum,
        "empty_volume_max": args.empty_volume_max,
        "low_volume_max": args.low_volume_max,
        "high_volume_min": args.high_volume_min,
        "seed": args.seed,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Strata manifest mismatch for {key}: expected {value!r}, got {payload.get(key)!r}")
    quota = args.batch_size * args.batches_per_stratum
    for name in STRATA:
        if len(payload.get("strata", {}).get(name, [])) != quota:
            raise ValueError(f"Strata manifest {name!r} does not contain exactly {quota} records")


def build_manifest(dataset, args, data_dir, manifest_path):
    quota = args.batch_size * args.batches_per_stratum
    selected = {name: [] for name in STRATA}
    indices = list(range(len(dataset.samples)))
    random.Random(args.seed).shuffle(indices)
    scanned = 0
    invalid = 0
    progress = tqdm(
        total=quota * len(STRATA),
        desc=f"stratify {args.domain}",
        position=args.progress_position,
        dynamic_ncols=True,
        disable=args.disable_progress,
    )
    executor_cls = ProcessPoolExecutor if args.volume_backend == "process" else ThreadPoolExecutor
    with executor_cls(
        max_workers=max(1, args.volume_workers),
        initializer=_init_volume_worker,
        initargs=(dataset.palm_mask, dataset.tactile_dim),
    ) as executor:
        for round_start in range(0, len(indices), args.volume_scan_round_size):
            round_indices = indices[round_start:round_start + args.volume_scan_round_size]
            work = [(index, dataset.samples[index]) for index in round_indices]
            chunks = list(chunked(work, args.volume_scan_chunk_size))
            for result, invalid_count in executor.map(_scan_volume_chunk, chunks):
                scanned += len(result)
                invalid += invalid_count
                for index, volume in result:
                    name = volume_stratum(volume, args)
                    if len(selected[name]) < quota:
                        selected[name].append({"index": index, "volume": volume})
                        progress.update(1)
                progress.set_postfix(
                    scanned=scanned,
                    empty=len(selected["empty"]),
                    low=len(selected["low"]),
                    mid=len(selected["mid"]),
                    high=len(selected["high"]),
                    refresh=False,
                )
            if all(len(selected[name]) >= quota for name in STRATA):
                break
    progress.close()

    missing = {name: quota - len(rows) for name, rows in selected.items() if len(rows) < quota}
    if missing:
        raise RuntimeError(
            f"Unable to fill pressure strata after scanning {scanned} valid records; missing={missing}. "
            "Reduce --batches_per_stratum or inspect the frame-volume distribution."
        )
    payload = {
        "version": 1,
        "domain": args.domain,
        "split": args.split,
        "data_dir": str(Path(data_dir).resolve()),
        "dataset_size": len(dataset),
        "batch_size": args.batch_size,
        "batches_per_stratum": args.batches_per_stratum,
        "empty_volume_max": args.empty_volume_max,
        "low_volume_max": args.low_volume_max,
        "high_volume_min": args.high_volume_min,
        "seed": args.seed,
        "scanned_valid_records": scanned,
        "invalid_records": invalid,
        "strata": selected,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, manifest_path)
    return payload


def load_or_build_manifest(dataset, args, data_dir):
    manifest_path = Path(args.strata_manifest) if args.strata_manifest else default_manifest_path(args, data_dir)
    manifest_path = manifest_path.expanduser().resolve()
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        validate_manifest(payload, args, data_dir)
        print(f"Loaded pressure-strata manifest: {manifest_path}", flush=True)
        return payload, manifest_path
    print(f"Building pressure-strata manifest: {manifest_path}", flush=True)
    payload = build_manifest(dataset, args, data_dir, manifest_path)
    return payload, manifest_path


def make_dataset(args, cfg):
    data_dir = base.resolve_domain_dir(args.domain, args.data_dir)
    dataset = OpenTouchTactileDataset(
        cfg=cfg,
        split=args.split,
        data_dir=[data_dir],
        train=False,
        index_workers=args.index_workers,
        index_chunksize=args.index_chunksize,
        index_backend=args.index_backend,
        index_cache_dir=args.index_cache_dir,
        rebuild_index=args.rebuild_index,
        index_cache_timeout=args.index_cache_timeout,
    )
    if not dataset:
        raise RuntimeError(f"No samples found for {args.domain}/{args.split}")
    return dataset, data_dir


def make_stratum_loader(dataset, manifest, name, args):
    indices = [int(row["index"]) for row in manifest["strata"][name]]
    kwargs = {
        "dataset": Subset(dataset, indices),
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def normalized_signature(vector, indices):
    signature = vector[indices].float()
    finite = bool(torch.isfinite(signature).all().item())
    if not finite:
        return signature.to(torch.float16), False
    norm = torch.linalg.vector_norm(signature)
    if float(norm) > 0.0:
        signature = signature / norm
    return signature.to(torch.float16), True


def cosine(left, right, left_finite=True, right_finite=True):
    if not left_finite or not right_finite:
        return math.nan
    value = float(torch.dot(left.float(), right.float()))
    return max(-1.0, min(1.0, value))


def pair_rows(records, level):
    rows = []
    for left, right in combinations(records, 2):
        for loss_name in (*base.LOSS_NAMES, "total"):
            value = cosine(
                left["signatures"][loss_name],
                right["signatures"][loss_name],
                left["finite"][loss_name],
                right["finite"][loss_name],
            )
            pair = "|".join(sorted((left["stratum"], right["stratum"]), key=STRATA.index))
            rows.append({
                "level": level,
                "left_id": left["id"],
                "right_id": right["id"],
                "left_stratum": left["stratum"],
                "right_stratum": right["stratum"],
                "stratum_relation": "same" if left["stratum"] == right["stratum"] else "different",
                "stratum_pair": pair,
                "loss": loss_name,
                "cosine": value,
                "finite": math.isfinite(value),
            })
    return rows


def summarize_pairs(rows, threshold):
    grouped = {}
    for row in rows:
        for pair_name in ("all", row["stratum_pair"]):
            key = (row["level"], row["loss"], row["stratum_relation"], pair_name)
            grouped.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        values = [float(row["cosine"]) for row in group if row["finite"]]
        summaries.append({
            "level": key[0],
            "loss": key[1],
            "stratum_relation": key[2],
            "stratum_pair": key[3],
            "count": len(group),
            "finite_count": len(values),
            "cosine_mean": statistics.fmean(values) if values else math.nan,
            "cosine_median": statistics.median(values) if values else math.nan,
            "cosine_std": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else math.nan,
            "cosine_min": min(values) if values else math.nan,
            "cosine_max": max(values) if values else math.nan,
            "negative_rate": sum(value < 0.0 for value in values) / max(len(values), 1),
            "strong_conflict_rate": sum(value < threshold for value in values) / max(len(values), 1),
        })
    return summaries


def finalize_accumulator(accumulator):
    signatures = {}
    finite = {}
    for name, vector in accumulator.items():
        is_finite = bool(torch.isfinite(vector).all().item())
        norm = torch.linalg.vector_norm(vector) if is_finite else torch.tensor(math.inf)
        if is_finite and float(norm) > 0.0:
            vector = vector / norm
        signatures[name] = vector.to(torch.float16)
        finite[name] = is_finite
    return signatures, finite


def main():
    args = parse_args()
    if args.batch_size < 1 or args.batches_per_stratum < 1:
        raise ValueError("--batch_size and --batches_per_stratum must be positive")
    if args.aggregate_batches < 1 or args.batches_per_stratum % args.aggregate_batches != 0:
        raise ValueError("--aggregate_batches must evenly divide --batches_per_stratum")

    cfg = base.load_model_cfg()
    print(f"[1/5] Loading {args.domain}/{args.split} index...", flush=True)
    dataset, data_dir = make_dataset(args, cfg)
    print("[2/5] Preparing exact pressure-volume strata...", flush=True)
    manifest, manifest_path = load_or_build_manifest(dataset, args, data_dir)
    if args.prepare_only:
        print(f"Prepared manifest: {manifest_path}", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Gradient audit requires a CUDA GPU")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    print(f"[3/5] Loading {args.ckpt} checkpoint on GPU {args.gpu}...", flush=True)
    checkpoint_path, exp_dir = base.resolve_checkpoint(args)
    model, loss_config, model_config = base.load_model(args, checkpoint_path, exp_dir, device)

    shared_count = sum(parameter.numel() for parameter in model.tactile_head.encoder.parameters())
    sketch_size = min(max(1, args.sketch_size), shared_count)
    sketch_indices = torch.randperm(shared_count, generator=torch.Generator().manual_seed(args.seed + 99))[:sketch_size]
    loaders = {name: make_stratum_loader(dataset, manifest, name, args) for name in STRATA}
    batch_records = []
    aggregate_records = []
    net_records = []
    batch_statistics = []

    total_batches = args.batches_per_stratum * len(STRATA)
    progress = tqdm(
        total=total_batches,
        desc=f"strata {args.ckpt}",
        position=args.progress_position,
        dynamic_ncols=True,
        disable=args.disable_progress,
    )
    print(f"[4/5] Computing {total_batches} homogeneous batch gradients...", flush=True)
    for stratum in STRATA:
        net_accumulator = {name: torch.zeros(sketch_size) for name in (*base.LOSS_NAMES, "total")}
        group_accumulator = {name: torch.zeros(sketch_size) for name in (*base.LOSS_NAMES, "total")}
        group_index = 0
        for batch_index, batch in enumerate(loaders[stratum]):
            gradients, diagnostics = base.domain_gradients(model, batch, loss_config, args, device)
            total_vector = sum(
                (gradients[name]["shared"] for name in base.LOSS_NAMES),
                start=torch.zeros(shared_count),
            )
            vectors = {name: gradients[name]["shared"] for name in base.LOSS_NAMES}
            vectors["total"] = total_vector
            signatures = {}
            finite = {}
            for name, vector in vectors.items():
                raw = vector[sketch_indices].float()
                group_accumulator[name] += raw
                net_accumulator[name] += raw
                signatures[name], finite[name] = normalized_signature(vector, sketch_indices)

            batch_id = f"{stratum}_{batch_index:03d}"
            batch_records.append({
                "id": batch_id,
                "stratum": stratum,
                "signatures": signatures,
                "finite": finite,
            })
            row = {
                "batch_id": batch_id,
                "stratum": stratum,
                **diagnostics,
            }
            for name in base.LOSS_NAMES:
                row[f"{name}_loss"] = gradients[name]["loss"]
                row[f"{name}_grad_norm"] = base.vector_stats(gradients[name]["shared"])[1]
            row["total_grad_norm"] = base.vector_stats(total_vector)[1]
            batch_statistics.append(row)

            if (batch_index + 1) % args.aggregate_batches == 0:
                group_signatures, group_finite = finalize_accumulator(group_accumulator)
                aggregate_records.append({
                    "id": f"{stratum}_group_{group_index:03d}",
                    "stratum": stratum,
                    "signatures": group_signatures,
                    "finite": group_finite,
                })
                group_index += 1
                group_accumulator = {name: torch.zeros(sketch_size) for name in (*base.LOSS_NAMES, "total")}

            progress.set_postfix(
                stratum=stratum,
                volume=f"{diagnostics['mean_gt_volume']:.1f}",
                grad=f"{row['total_grad_norm']:.3g}",
                refresh=False,
            )
            progress.update(1)
            del gradients, vectors, total_vector
            torch.cuda.empty_cache()

        net_signatures, net_finite = finalize_accumulator(net_accumulator)
        net_records.append({
            "id": f"{stratum}_net",
            "stratum": stratum,
            "signatures": net_signatures,
            "finite": net_finite,
        })
    progress.close()

    print("[5/5] Computing batch/group/net cosine matrices and writing results...", flush=True)
    batch_pairs = pair_rows(batch_records, "batch")
    aggregate_pairs = pair_rows(aggregate_records, f"aggregate_{args.aggregate_batches}")
    net_pairs = pair_rows(net_records, "stratum_net")
    output_dir = Path(args.output_dir) if args.output_dir else (
        FT_DIR / "gradient_strata_audits" / f"{base.safe_name(args.exp_name)}_{args.ckpt}_{args.domain}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "batch_statistics.csv", batch_statistics)
    write_csv(output_dir / "batch_gradient_pairs.csv", batch_pairs)
    write_csv(output_dir / "batch_gradient_summary.csv", summarize_pairs(batch_pairs, args.strong_conflict_threshold))
    write_csv(output_dir / "aggregate_gradient_pairs.csv", aggregate_pairs)
    write_csv(output_dir / "aggregate_gradient_summary.csv", summarize_pairs(aggregate_pairs, args.strong_conflict_threshold))
    write_csv(output_dir / "stratum_net_gradient_cosines.csv", net_pairs)
    config = {
        "args": vars(args),
        "checkpoint": str(checkpoint_path),
        "data_dir": data_dir,
        "dataset_size": len(dataset),
        "manifest": str(manifest_path),
        "manifest_counts": {name: len(manifest["strata"][name]) for name in STRATA},
        "shared_parameter_count": shared_count,
        "sketch_size": sketch_size,
        "model_config": model_config,
    }
    with (output_dir / "audit_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    print(f"Wrote stratified gradient audit to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
