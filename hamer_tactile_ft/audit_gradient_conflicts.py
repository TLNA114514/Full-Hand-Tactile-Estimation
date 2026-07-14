#!/usr/bin/env python3
"""Audit loss and domain gradient conflicts for zero-inflated tactile heads."""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from contextlib import nullcontext
from dataclasses import fields
from itertools import combinations
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler


FT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = FT_DIR.parent
sys.path.insert(0, str(WORKSPACE_DIR / "hamer"))
sys.path.insert(0, str(FT_DIR))

from hamer.configs import get_config

from dataset import OpenTouchTactileDataset
from losses import (
    TactileLossConfig,
    dataset_weight_like,
    pressure_weight_like,
    ramp_value,
)
from train import OpenTouchHAMER_TactileWrapper, load_compatible_state_dict, resolve_data_dirs

# train.py rewrites argv[0] for its own launch path. Restore this script's name
# so argparse help and provenance remain accurate when train is imported.
sys.argv[0] = PROGRAM_PATH


LOSS_NAMES = ("final", "support", "positive_bin")


def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(value).lower()).strip("_")


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def resolve_checkpoint(args):
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path.resolve(), path.parent.resolve()

    if not args.exp_name:
        raise ValueError("Provide --exp_name or --checkpoint")
    root = Path(args.checkpoint_root).expanduser()
    if not root.is_absolute():
        root = WORKSPACE_DIR / root
    exp_dir = (root / args.exp_name).resolve()
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment checkpoint directory not found: {exp_dir}")

    selector = "rmse-best" if args.ckpt == "best" else args.ckpt
    names = {
        "rmse-best": "best_rmse.ckpt",
        "viou-best": "best_viou.ckpt",
        "last": "last.ckpt",
    }
    path = exp_dir / names[selector]
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint selector {selector!r} not found: {path}")
    return path, exp_dir


def load_model_cfg():
    path = WORKSPACE_DIR / "hamer" / "_DATA" / "hamer_ckpts" / "model_config.yaml"
    cfg = get_config(str(path), update_cachedir=True)
    if cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in cfg.MODEL:
        cfg.defrost()
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.freeze()
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.defrost()
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        cfg.freeze()
    return cfg


def loss_config_from_json(payload):
    known = {field.name for field in fields(TactileLossConfig)}
    values = {key: value for key, value in payload.items() if key in known}
    return TactileLossConfig(**values)


def model_value(model_config, key, default):
    value = model_config.get(key, default)
    return default if value is None else value


def load_model(args, checkpoint_path, exp_dir, device):
    model_config = read_json(exp_dir / "model_config.json")
    loss_payload = read_json(exp_dir / "loss_config.json")
    loss_config = loss_config_from_json(loss_payload)
    head_type = str(model_value(model_config, "tactile_head_type", args.tactile_head_type))
    if head_type not in {"zero_ordinal_residual", "region_source_ordinal"}:
        raise ValueError(
            "This audit requires a zero-inflated tactile head; "
            f"checkpoint config uses {head_type!r}"
        )

    positive_values = model_config.get("positive_bin_values")
    cfg = load_model_cfg()
    model = OpenTouchHAMER_TactileWrapper(
        cfg=cfg,
        tactile_only_forward=True,
        tactile_loss_scale=float(model_value(model_config, "tactile_loss_scale", 10.0)),
        tactile_loss_config=loss_config,
        tactile_head_type=head_type,
        contact_gate_floor=float(model_value(model_config, "contact_gate_floor", 0.1)),
        frame_volume_gate_mode=str(model_value(model_config, "frame_volume_gate_mode", "aux")),
        pool_layout=str(model_value(model_config, "pool_layout", "hand7")),
        pool_grid_size=int(model_value(model_config, "pool_grid_size", 7)),
        ordinal_thresholds=str(
            model_value(model_config, "ordinal_thresholds", loss_config.ordinal_thresholds)
        ),
        residual_max_scale=float(model_value(model_config, "residual_max_scale", 1.0)),
        positive_bin_values=positive_values,
        source_dim=int(model_value(model_config, "source_dim", 256)),
        source_slots=int(model_value(model_config, "source_slots", 38)),
        source_layers=int(model_value(model_config, "source_layers", 2)),
        source_heads=int(model_value(model_config, "source_heads", 4)),
        source_dropout=float(model_value(model_config, "source_dropout", 0.1)),
    )

    dummy = torch.zeros(1, 3, cfg.MODEL.IMAGE_SIZE, cfg.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        features = model.backbone(dummy[:, :, :, 32:-32])
        model.tactile_head(features)
    load_compatible_state_dict(model, str(checkpoint_path))
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.tactile_head.encoder.parameters():
        parameter.requires_grad_(True)

    return model, loss_config, model_config


def resolve_domain_dir(domain, explicit_dir):
    if explicit_dir:
        return str(Path(explicit_dir).expanduser().resolve())
    resolved = resolve_data_dirs(SimpleNamespace(datasets=domain, data_dir=None))
    if len(resolved) != 1:
        raise RuntimeError(f"Expected one data root for {domain!r}, got {resolved}")
    return resolved[0]


def make_domain_loader(args, cfg, domain, explicit_dir, seed):
    data_dir = resolve_domain_dir(domain, explicit_dir)
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
        raise RuntimeError(f"No samples found for domain {domain!r} split {args.split!r}")

    sample_count = args.num_batches * args.batch_size
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=sample_count,
        generator=generator,
    )
    loader_args = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers > 0:
        loader_args["persistent_workers"] = args.persistent_workers
        loader_args["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**loader_args), data_dir, len(dataset)


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


def expand_mask(mask, target):
    mask = mask.to(device=target.device, dtype=target.dtype)
    if mask.shape == target.shape:
        return mask
    if mask.ndim == 1:
        return mask.unsqueeze(0).expand_as(target)
    if mask.ndim == 2:
        return mask
    return mask.expand_as(target)


def valid_mask_like(valid, target):
    valid = valid.to(device=target.device, dtype=target.dtype)
    if valid.shape == target.shape:
        return valid
    return valid.unsqueeze(-1).expand_as(target)


def compute_loss_components(output, batch, config, current_epoch, tactile_loss_scale):
    pred = output["pred_tactile"]
    target = batch["tactile_signal"].to(device=pred.device, dtype=pred.dtype)
    palm = expand_mask(batch["palm_mask"], target)
    valid = valid_mask_like(batch["has_tactile"], target)
    mask = palm * valid
    denominator = mask.sum().clamp_min(1.0)

    pressure_ramp = ramp_value(
        config.pressure_loss_warmup_epochs,
        current_epoch,
        start=config.pressure_loss_warmup_start,
    )
    active_ramp = ramp_value(config.active_pressure_weight_warmup_epochs, current_epoch)
    pressure_weights = pressure_weight_like(target, config, active_ramp)
    dataset_weights = dataset_weight_like(target, batch.get("dataset"), config)
    final_raw = (
        F.smooth_l1_loss(pred, target, reduction="none")
        * pressure_weights
        * dataset_weights
        * mask
    ).sum() / denominator
    final = final_raw * pressure_ramp

    support_logits = output["support_logits"]
    positive_logits = output["positive_logits"]
    positive_target = (target >= config.zero_support_thr).to(dtype=pred.dtype)
    support_raw = (
        F.binary_cross_entropy_with_logits(support_logits, positive_target, reduction="none")
        * mask
    ).sum() / denominator
    support = support_raw * config.support_loss_weight

    thresholds = torch.as_tensor(
        [float(item) for item in config.ordinal_thresholds.split(",") if item.strip()],
        device=pred.device,
        dtype=pred.dtype,
    )
    positive_target_index = torch.bucketize(target, thresholds[1:], right=True)
    positive_target_index = positive_target_index.clamp(max=positive_logits.shape[-1] - 1)
    positive_mask = mask * positive_target
    positive_count = positive_mask.sum()
    positive_ce = F.cross_entropy(
        positive_logits.reshape(-1, positive_logits.shape[-1]),
        positive_target_index.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    positive_raw = (positive_ce * positive_mask).sum() / positive_count.clamp_min(1.0)
    positive_bin = positive_raw * config.positive_bin_loss_weight

    scale = float(tactile_loss_scale)
    losses = {
        "final": final * scale,
        "support": support * scale,
        "positive_bin": positive_bin * scale,
    }
    diagnostics = {
        "final_raw": float(final_raw.detach().cpu()),
        "support_raw": float(support_raw.detach().cpu()),
        "positive_bin_raw": float(positive_raw.detach().cpu()),
        "positive_fraction": float((positive_count / denominator).detach().cpu()),
        "mean_gt_pressure": float(((target * mask).sum() / denominator).detach().cpu()),
        "mean_gt_volume": float(((target * palm).sum(dim=-1).mean()).detach().cpu()),
        "active_fraction": float((((target >= config.active_pressure_thr).to(pred.dtype) * mask).sum() / denominator).detach().cpu()),
    }
    return losses, diagnostics


def autocast_context(args):
    if args.precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def flatten_gradients(gradients, parameters):
    chunks = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            chunks.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            chunks.append(gradient.detach().float().reshape(-1).cpu())
    return torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.float32)


def domain_gradients(model, batch, config, args, device):
    captured_features = []

    def capture_encoder_output(_module, _inputs, output):
        captured_features.append(output)

    hook = model.tactile_head.encoder.register_forward_hook(capture_encoder_output)
    batch = move_to_device(batch, device)
    try:
        with autocast_context(args):
            output = model.forward_step(batch, train=False)
            losses, diagnostics = compute_loss_components(
                output,
                batch,
                config,
                current_epoch=args.epoch,
                tactile_loss_scale=model.tactile_loss_scale,
            )
        feature = captured_features[-1]
        shared_parameters = tuple(model.tactile_head.encoder.parameters())
        requested = (feature, *shared_parameters)
        result = {}
        for loss_index, loss_name in enumerate(LOSS_NAMES):
            gradients = torch.autograd.grad(
                losses[loss_name],
                requested,
                retain_graph=loss_index < len(LOSS_NAMES) - 1,
                allow_unused=True,
            )
            result[loss_name] = {
                "loss": float(losses[loss_name].detach().float().cpu()),
                "feature": gradients[0].detach().float().reshape(-1).cpu(),
                "shared": flatten_gradients(gradients[1:], shared_parameters),
            }
        return result, diagnostics
    finally:
        hook.remove()


def vector_stats(vector):
    finite = bool(torch.isfinite(vector).all().item())
    norm = float(torch.linalg.vector_norm(vector).item()) if finite else math.inf
    return finite, norm


def cosine_row(batch_index, target, left_key, right_key, vectors):
    left = vectors[left_key]
    right = vectors[right_key]
    left_finite, left_norm = vector_stats(left)
    right_finite, right_norm = vector_stats(right)
    finite = left_finite and right_finite
    if finite and left_norm > 0.0 and right_norm > 0.0:
        dot = float(torch.dot(left, right).item())
        cosine = dot / (left_norm * right_norm)
        cosine = max(-1.0, min(1.0, cosine))
    else:
        dot = math.nan
        cosine = math.nan
    return {
        "batch_index": batch_index,
        "target": target,
        "left_domain": left_key[0],
        "left_loss": left_key[1],
        "right_domain": right_key[0],
        "right_loss": right_key[1],
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


def build_pair_rows(batch_index, domain_results):
    shared_vectors = {}
    feature_vectors = {}
    for domain, gradients in domain_results.items():
        for loss_name, values in gradients.items():
            shared_vectors[(domain, loss_name)] = values["shared"]
            feature_vectors[(domain, loss_name)] = values["feature"]

    rows = []
    for left_key, right_key in combinations(shared_vectors, 2):
        rows.append(cosine_row(batch_index, "shared_encoder", left_key, right_key, shared_vectors))
    for domain in domain_results:
        domain_keys = [(domain, loss_name) for loss_name in LOSS_NAMES]
        for left_key, right_key in combinations(domain_keys, 2):
            rows.append(cosine_row(batch_index, "encoder_feature", left_key, right_key, feature_vectors))
    return rows


def summarize_pair_rows(rows, strong_conflict_threshold):
    grouped = {}
    for row in rows:
        key = (
            row["target"],
            row["left_domain"],
            row["left_loss"],
            row["right_domain"],
            row["right_loss"],
        )
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group in sorted(grouped.items()):
        finite_rows = [row for row in group if row["finite"] and math.isfinite(row["cosine"])]
        cosines = [row["cosine"] for row in finite_rows]
        ratios = [row["norm_ratio_max_over_min"] for row in finite_rows]
        summary = {
            "target": key[0],
            "left_domain": key[1],
            "left_loss": key[2],
            "right_domain": key[3],
            "right_loss": key[4],
            "count": len(group),
            "finite_count": len(finite_rows),
            "nonfinite_rate": 1.0 - len(finite_rows) / max(len(group), 1),
            "cosine_mean": statistics.fmean(cosines) if cosines else math.nan,
            "cosine_median": statistics.median(cosines) if cosines else math.nan,
            "cosine_std": statistics.pstdev(cosines) if len(cosines) > 1 else 0.0 if cosines else math.nan,
            "cosine_min": min(cosines) if cosines else math.nan,
            "cosine_max": max(cosines) if cosines else math.nan,
            "negative_rate": sum(value < 0.0 for value in cosines) / max(len(cosines), 1),
            "strong_conflict_rate": sum(value < strong_conflict_threshold for value in cosines) / max(len(cosines), 1),
            "norm_ratio_median": statistics.median(ratios) if ratios else math.nan,
        }
        summaries.append(summary)
    return summaries


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp_name", default="mixed_zero_ordinal_residual_v19_condnll")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint_root", default=str(FT_DIR / "checkpoints"))
    parser.add_argument("--ckpt", choices=("rmse-best", "viou-best", "last", "best"), default="rmse-best")
    parser.add_argument("--tactile_head_type", default="zero_ordinal_residual")
    parser.add_argument("--domain_a", default="opentouch")
    parser.add_argument("--domain_b", default="touchanything")
    parser.add_argument("--domain_a_dir", default=None)
    parser.add_argument("--domain_b_dir", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--gpu", "--gpus", dest="gpu", default="0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--index_workers", type=int, default=32)
    parser.add_argument("--index_chunksize", type=int, default=512)
    parser.add_argument("--index_backend", choices=("process", "thread"), default="process")
    parser.add_argument("--index_cache_dir", default=str(FT_DIR / "index_cache"))
    parser.add_argument("--index_cache_timeout", type=int, default=3600)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epoch", type=int, default=0, help="Epoch used only for legacy loss schedules; v19 uses full weights at epoch 0.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--strong_conflict_threshold", type=float, default=-0.2)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.num_batches < 1:
        raise ValueError("--batch_size and --num_batches must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Gradient conflict audit requires a CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    checkpoint_path, exp_dir = resolve_checkpoint(args)
    output_dir = Path(args.output_dir) if args.output_dir else (
        FT_DIR / "gradient_audits" / f"{safe_name(args.exp_name or exp_dir.name)}_{safe_name(args.ckpt)}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Precision: {args.precision}; paired batches: {args.num_batches}; batch size/domain: {args.batch_size}")
    model, loss_config, model_config = load_model(args, checkpoint_path, exp_dir, device)
    cfg = load_model_cfg()
    loader_a, data_dir_a, count_a = make_domain_loader(
        args, cfg, args.domain_a, args.domain_a_dir, args.seed
    )
    loader_b, data_dir_b, count_b = make_domain_loader(
        args, cfg, args.domain_b, args.domain_b_dir, args.seed + 1
    )

    pair_rows = []
    loss_rows = []
    iterator_a = iter(loader_a)
    iterator_b = iter(loader_b)
    for batch_index in range(args.num_batches):
        domain_batches = {
            args.domain_a: next(iterator_a),
            args.domain_b: next(iterator_b),
        }
        domain_results = {}
        for domain, batch in domain_batches.items():
            gradients, diagnostics = domain_gradients(model, batch, loss_config, args, device)
            domain_results[domain] = gradients
            for loss_name in LOSS_NAMES:
                finite, norm = vector_stats(gradients[loss_name]["shared"])
                feature_finite, feature_norm = vector_stats(gradients[loss_name]["feature"])
                loss_rows.append({
                    "batch_index": batch_index,
                    "domain": domain,
                    "loss": loss_name,
                    "weighted_scaled_loss": gradients[loss_name]["loss"],
                    "shared_grad_norm": norm,
                    "feature_grad_norm": feature_norm,
                    "shared_grad_finite": finite,
                    "feature_grad_finite": feature_finite,
                    **diagnostics,
                })
        pair_rows.extend(build_pair_rows(batch_index, domain_results))
        print(f"Completed paired batch {batch_index + 1}/{args.num_batches}", flush=True)
        del domain_results
        torch.cuda.empty_cache()

    summary_rows = summarize_pair_rows(pair_rows, args.strong_conflict_threshold)
    write_csv(output_dir / "gradient_pairs_by_batch.csv", pair_rows)
    write_csv(output_dir / "gradient_pair_summary.csv", summary_rows)
    write_csv(output_dir / "loss_gradient_norms.csv", loss_rows)

    config = {
        "args": vars(args),
        "checkpoint": str(checkpoint_path),
        "experiment_dir": str(exp_dir),
        "output_dir": str(output_dir),
        "domain_data_dirs": {args.domain_a: data_dir_a, args.domain_b: data_dir_b},
        "domain_sample_counts": {args.domain_a: count_a, args.domain_b: count_b},
        "loss_config": {field.name: getattr(loss_config, field.name) for field in fields(TactileLossConfig)},
        "model_config": model_config,
        "shared_parameter_count": sum(parameter.numel() for parameter in model.tactile_head.encoder.parameters()),
    }
    with (output_dir / "audit_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    with (output_dir / "gradient_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2, sort_keys=True)

    print("\nGradient conflict summary (shared encoder):")
    for row in summary_rows:
        if row["target"] != "shared_encoder":
            continue
        left = f"{row['left_domain']}:{row['left_loss']}"
        right = f"{row['right_domain']}:{row['right_loss']}"
        print(
            f"  {left:34s} vs {right:34s} "
            f"cos={row['cosine_mean']:+.4f} "
            f"neg={row['negative_rate']:.1%} "
            f"strong={row['strong_conflict_rate']:.1%} "
            f"norm_ratio={row['norm_ratio_median']:.2f}"
        )
    print(f"\nWrote audit outputs to: {output_dir}")


if __name__ == "__main__":
    main()
