#!/usr/bin/env python3
"""Audit paired OpenTouch/TouchAnything gradients for Dense V2/ReZero heads."""

import argparse
import bisect
import csv
import json
import math
import os
import random
import statistics
import sys
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace


PROGRAM_PATH = sys.argv[0]


def _early_gpu_selection():
    for index, value in enumerate(sys.argv):
        if value in ("--gpu", "--gpus") and index + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]
            return


_early_gpu_selection()
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYRENDER_PLATFORM", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import torch
from torch.utils.data import DataLoader


FT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = FT_DIR.parent
sys.path.insert(0, str(WORKSPACE_DIR / "hamer"))
sys.path.insert(0, str(FT_DIR))

from dataset import OpenTouchTactileDataset
from eval_tactile_fast import (
    _load_model,
    _load_model_cfg,
    _resolve_experiment_model_metadata,
)
from losses import TactileLossConfig, compute_tactile_loss
from train import resolve_data_dirs

# train.py rewrites argv[0] during import.
sys.argv[0] = PROGRAM_PATH


PARAMETER_GROUPS = (
    "base_projection",
    "dense_decoder",
    "multilevel_residual_branches",
    "global_gate",
    "entire_tactile_head",
)


def parse_sampling_bin_edges(value, name):
    values = []
    for token in str(value).split(","):
        token = token.strip().lower()
        if not token:
            continue
        values.append(float("inf") if token == "inf" else float(token))
    if len(values) < 2 or values != sorted(set(values)):
        raise ValueError(f"{name} must contain at least two unique increasing edges")
    return tuple(values)


def safe_name(value):
    return "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(value).lower()
    ).strip("_")


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def resolve_checkpoint(args):
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path, path.parent
    if not args.exp_name:
        raise ValueError("Provide --exp_name or --checkpoint")
    experiment_dir = (Path(args.checkpoint_root).expanduser() / args.exp_name).resolve()
    names = {
        "loss-best": "best_loss.ckpt",
        "contact-best": "best_contact.ckpt",
        "last": "last.ckpt",
    }
    path = experiment_dir / names[args.ckpt]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint selector {args.ckpt!r} not found: {path}")
    return path, experiment_dir


def loss_config_from_json(payload):
    known = {field.name for field in fields(TactileLossConfig)}
    return TactileLossConfig(**{key: value for key, value in payload.items() if key in known})


def load_dense_model(args, checkpoint_path, experiment_dir, device):
    model_config = read_json(experiment_dir / "model_config.json")
    head_type = str(model_config.get("tactile_head_type", ""))
    if head_type != "dense_v2_dino_rezero":
        raise ValueError(f"Dense gradient audit does not support tactile_head_type={head_type!r}")
    eval_args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        dino_weights=args.dino_weights,
        bbox_rescale_factor=None,
        save_diagnostics=False,
        save_visualizations=False,
        model_metadata={},
    )
    _resolve_experiment_model_metadata(eval_args)
    model_cfg = _load_model_cfg()
    model = _load_model(eval_args, model_cfg, device)
    model.tactile_loss_scale = float(model_config.get("tactile_loss_scale", 10.0))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.tactile_head.parameters():
        parameter.requires_grad_(True)
    loss_config = loss_config_from_json(read_json(experiment_dir / "loss_config.json"))
    return model, model_cfg, loss_config, model_config


def resolve_domain_dir(domain, explicit_dir):
    if explicit_dir:
        return str(Path(explicit_dir).expanduser().resolve())
    roots = resolve_data_dirs(SimpleNamespace(datasets=domain, data_dir=None))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one root for {domain!r}, got {roots}")
    return roots[0]


def domain_manifest(args, domain):
    slug = "opentouch" if domain.lower() in {"opentouch", "ot"} else "touchanything"
    path = Path(args.manifest_root).expanduser() / slug / "data_integrity_samples.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Strict {slug} manifest is missing: {path}. Run audit_sequence_failures.py "
            "--mode export_manifests first."
        )
    return str(path.resolve()), slug


def make_domain_dataset(args, cfg, domain, explicit_dir):
    data_dir = resolve_domain_dir(domain, explicit_dir)
    manifest, slug = domain_manifest(args, domain)
    dataset = OpenTouchTactileDataset(
        cfg=cfg,
        split=args.split,
        data_dir=[data_dir],
        train=False,
        index_workers=args.index_workers,
        index_chunksize=args.index_chunksize,
        index_backend=args.index_backend,
        index_cache_dir=args.index_cache_dir,
        index_cache_timeout=args.index_cache_timeout,
        index_manifest=manifest,
        expected_datasets=[slug],
        tactile_only=True,
    )
    if not dataset:
        raise RuntimeError(f"No samples found for {domain!r} split={args.split!r}")
    return dataset, data_dir, manifest


def _bin_index(value, edges):
    index = bisect.bisect_right(edges, float(value)) - 1
    return max(0, min(index, len(edges) - 2))


def build_strata(dataset, pressure_bins, volume_bins):
    strata = {}
    for index, sample in enumerate(dataset.samples):
        if sample.get("max_pressure") is None or sample.get("target_volume") is None:
            raise ValueError(
                "Gradient matching requires max_pressure and target_volume in the compact index"
            )
        key = (
            _bin_index(sample["max_pressure"], pressure_bins),
            _bin_index(sample["target_volume"], volume_bins),
        )
        strata.setdefault(key, []).append(index)
    return strata


class _CyclingQueue:
    def __init__(self, values, seed):
        self.values = list(values)
        self.random = random.Random(seed)
        self.queue = []

    def draw(self, count):
        output = []
        while len(output) < count:
            if not self.queue:
                self.queue = list(self.values)
                self.random.shuffle(self.queue)
            take = min(count - len(output), len(self.queue))
            output.extend(self.queue[:take])
            del self.queue[:take]
        return output


def build_matched_batch_schedules(
    strata_a,
    strata_b,
    num_batches,
    batch_size,
    seed,
):
    common = sorted(set(strata_a).intersection(strata_b))
    if not common:
        raise RuntimeError("OT and TA have no shared pressure/volume strata")
    weights = [min(len(strata_a[key]), len(strata_b[key])) for key in common]
    chooser = random.Random(seed)
    selected = chooser.choices(common, weights=weights, k=num_batches)
    queues_a = {key: _CyclingQueue(strata_a[key], seed + 1009 * (index + 1)) for index, key in enumerate(common)}
    queues_b = {key: _CyclingQueue(strata_b[key], seed + 2003 * (index + 1)) for index, key in enumerate(common)}
    schedule_a = [queues_a[key].draw(batch_size) for key in selected]
    schedule_b = [queues_b[key].draw(batch_size) for key in selected]
    return schedule_a, schedule_b, selected


def make_loader(dataset, batch_schedule, args):
    kwargs = {
        "dataset": dataset,
        "batch_sampler": batch_schedule,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def autocast_context(args):
    if args.precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def parameter_group(name):
    if name.startswith("base_projection."):
        return "base_projection"
    if name.startswith("decoder."):
        return "dense_decoder"
    if name == "global_gate":
        return "global_gate"
    if name == "level_logits" or name.startswith("projections.") or name.startswith("refiners."):
        return "multilevel_residual_branches"
    return None


def flatten_group_gradients(named_parameters, gradients, group):
    chunks = []
    for (name, parameter), gradient in zip(named_parameters, gradients):
        if group != "entire_tactile_head" and parameter_group(name) != group:
            continue
        if gradient is None:
            chunks.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            chunks.append(gradient.detach().float().reshape(-1).cpu())
    return torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.float32)


def domain_gradients(model, batch, config, args, device):
    batch = move_to_device(batch, device)
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in model.tactile_head.named_parameters()
        if parameter.requires_grad
    )
    with autocast_context(args):
        output = model.forward_step(batch, train=False)
        loss, details = compute_tactile_loss(
            pred=output["pred_tactile"],
            logits=output["pred_logits"],
            target=batch["tactile_signal"],
            palm_mask=batch["palm_mask"],
            valid_mask=batch["has_tactile"],
            dataset_batch=batch.get("dataset"),
            config=config,
            current_epoch=args.epoch,
            ramp_override=1.0,
        )
        loss = loss * float(model.tactile_loss_scale)
    gradients = torch.autograd.grad(
        loss,
        tuple(parameter for _, parameter in named_parameters),
        allow_unused=True,
    )
    vectors = {
        group: flatten_group_gradients(named_parameters, gradients, group)
        for group in PARAMETER_GROUPS
    }
    target = batch["tactile_signal"].detach().float()
    palm = batch["palm_mask"].detach().float()
    valid = batch["has_tactile"].detach().float().unsqueeze(-1)
    mask = palm * valid
    denominator = mask.sum().clamp_min(1.0)
    diagnostics = {
        "loss": float(loss.detach().float().cpu()),
        "direct_raw": float(details["loss_base_tactile"].cpu()),
        "mean_gt_pressure": float(((target * mask).sum() / denominator).cpu()),
        "mean_gt_volume": float(((target * palm).sum(dim=1).mean()).cpu()),
        "mean_max_pressure": float(target.max(dim=1).values.mean().cpu()),
        "active_fraction": float((((target >= config.active_pressure_thr).float() * mask).sum() / denominator).cpu()),
    }
    return vectors, diagnostics


def vector_stats(vector):
    finite = bool(torch.isfinite(vector).all().item())
    norm = float(torch.linalg.vector_norm(vector).item()) if finite else math.inf
    return finite, norm


def cosine_row(batch_index, stratum, group, left_domain, right_domain, left, right):
    left_finite, left_norm = vector_stats(left)
    right_finite, right_norm = vector_stats(right)
    finite = left_finite and right_finite
    if finite and left_norm > 0.0 and right_norm > 0.0:
        dot = float(torch.dot(left, right).item())
        cosine = max(-1.0, min(1.0, dot / (left_norm * right_norm)))
    else:
        dot = math.nan
        cosine = math.nan
    return {
        "batch_index": batch_index,
        "pressure_bin": stratum[0],
        "volume_bin": stratum[1],
        "parameter_group": group,
        "left_domain": left_domain,
        "right_domain": right_domain,
        "cosine": cosine,
        "dot": dot,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "norm_ratio_max_over_min": (
            max(left_norm, right_norm) / max(min(left_norm, right_norm), 1e-30)
            if finite else math.inf
        ),
        "finite": finite,
    }


def summarize_rows(rows, threshold):
    output = []
    for group in PARAMETER_GROUPS:
        selected = [row for row in rows if row["parameter_group"] == group]
        finite = [row for row in selected if row["finite"] and math.isfinite(row["cosine"])]
        cosines = [row["cosine"] for row in finite]
        ratios = [row["norm_ratio_max_over_min"] for row in finite]
        output.append({
            "parameter_group": group,
            "count": len(selected),
            "finite_count": len(finite),
            "cosine_mean": statistics.fmean(cosines) if cosines else math.nan,
            "cosine_median": statistics.median(cosines) if cosines else math.nan,
            "cosine_min": min(cosines) if cosines else math.nan,
            "negative_rate": sum(value < 0.0 for value in cosines) / max(len(cosines), 1),
            "strong_conflict_rate": sum(value < threshold for value in cosines) / max(len(cosines), 1),
            "norm_ratio_median": statistics.median(ratios) if ratios else math.nan,
        })
    return output


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="mixed_dense_v2_dinov3_rezero_strictcontrol")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default=str(FT_DIR / "checkpoints"))
    parser.add_argument(
        "--ckpt",
        choices=("loss-best", "contact-best", "last"),
        default="loss-best",
    )
    parser.add_argument("--dino_weights", default=None)
    parser.add_argument("--domain_a", default="opentouch")
    parser.add_argument("--domain_b", default="touchanything")
    parser.add_argument("--domain_a_dir", default=None)
    parser.add_argument("--domain_b_dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--gpu", "--gpus", dest="gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--index_workers", type=int, default=32)
    parser.add_argument("--index_chunksize", type=int, default=512)
    parser.add_argument("--index_backend", choices=("process", "thread"), default="process")
    parser.add_argument("--index_cache_dir", default=str(FT_DIR / "index_cache"))
    parser.add_argument("--index_cache_timeout", type=int, default=3600)
    parser.add_argument(
        "--manifest_root",
        default=str(FT_DIR / "data_integrity_audits" / "mixed_v2_input" / "by_dataset"),
    )
    parser.add_argument("--pressure_bins", default="0,0.005,0.05,0.2,0.5,0.7,inf")
    parser.add_argument("--volume_bins", default="0,10,50,150,300,inf")
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--epoch", type=int, default=59)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--strong_conflict_threshold", type=float, default=-0.2)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_batches < 1:
        raise ValueError("--batch_size and --num_batches must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Dense gradient audit requires a CUDA GPU")
    args.pressure_bins = parse_sampling_bin_edges(args.pressure_bins, "--pressure_bins")
    args.volume_bins = parse_sampling_bin_edges(args.volume_bins, "--volume_bins")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    checkpoint_path, experiment_dir = resolve_checkpoint(args)
    output_dir = Path(args.output_dir) if args.output_dir else (
        FT_DIR / "gradient_audits" / f"{safe_name(args.exp_name)}_{safe_name(args.ckpt)}_dense_domains"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, loss_config, model_config = load_dense_model(
        args, checkpoint_path, experiment_dir, device
    )
    dataset_a, data_dir_a, manifest_a = make_domain_dataset(
        args, cfg, args.domain_a, args.domain_a_dir
    )
    dataset_b, data_dir_b, manifest_b = make_domain_dataset(
        args, cfg, args.domain_b, args.domain_b_dir
    )
    strata_a = build_strata(dataset_a, args.pressure_bins, args.volume_bins)
    strata_b = build_strata(dataset_b, args.pressure_bins, args.volume_bins)
    schedule_a, schedule_b, selected_strata = build_matched_batch_schedules(
        strata_a, strata_b, args.num_batches, args.batch_size, args.seed
    )
    loader_a = make_loader(dataset_a, schedule_a, args)
    loader_b = make_loader(dataset_b, schedule_b, args)

    pair_rows = []
    norm_rows = []
    for batch_index, (batch_a, batch_b, stratum) in enumerate(
        zip(loader_a, loader_b, selected_strata)
    ):
        results = {}
        for domain, batch in ((args.domain_a, batch_a), (args.domain_b, batch_b)):
            vectors, diagnostics = domain_gradients(model, batch, loss_config, args, device)
            results[domain] = vectors
            for group, vector in vectors.items():
                finite, norm = vector_stats(vector)
                norm_rows.append({
                    "batch_index": batch_index,
                    "pressure_bin": stratum[0],
                    "volume_bin": stratum[1],
                    "domain": domain,
                    "parameter_group": group,
                    "gradient_norm": norm,
                    "gradient_finite": finite,
                    **diagnostics,
                })
        for group in PARAMETER_GROUPS:
            pair_rows.append(cosine_row(
                batch_index,
                stratum,
                group,
                args.domain_a,
                args.domain_b,
                results[args.domain_a][group],
                results[args.domain_b][group],
            ))
        print(f"Completed matched batch {batch_index + 1}/{args.num_batches}", flush=True)

    summary_rows = summarize_rows(pair_rows, args.strong_conflict_threshold)
    write_csv(output_dir / "gradient_pairs_by_batch.csv", pair_rows)
    write_csv(output_dir / "gradient_pair_summary.csv", summary_rows)
    write_csv(output_dir / "loss_gradient_norms.csv", norm_rows)
    config = {
        "args": vars(args),
        "checkpoint": str(checkpoint_path),
        "experiment_dir": str(experiment_dir),
        "model_config": model_config,
        "loss_config": {field.name: getattr(loss_config, field.name) for field in fields(TactileLossConfig)},
        "domains": {
            args.domain_a: {"root": data_dir_a, "manifest": manifest_a, "samples": len(dataset_a)},
            args.domain_b: {"root": data_dir_b, "manifest": manifest_b, "samples": len(dataset_b)},
        },
        "parameter_counts": {
            group: int(sum(
                parameter.numel()
                for name, parameter in model.tactile_head.named_parameters()
                if group == "entire_tactile_head" or parameter_group(name) == group
            ))
            for group in PARAMETER_GROUPS
        },
        "matched_strata": [list(value) for value in selected_strata],
    }
    with (output_dir / "audit_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    with (output_dir / "gradient_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, sort_keys=True)

    print("\nDense OT/TA gradient summary:")
    for row in summary_rows:
        print(
            f"  {row['parameter_group']:34s} cos={row['cosine_mean']:+.4f} "
            f"neg={row['negative_rate']:.1%} strong={row['strong_conflict_rate']:.1%} "
            f"norm_ratio={row['norm_ratio_median']:.2f}"
        )
    print(f"\nWrote audit outputs to: {output_dir}")


if __name__ == "__main__":
    main()
