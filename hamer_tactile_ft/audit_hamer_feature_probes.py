#!/usr/bin/env python3
"""Probe HaMeR features for domain identity and tactile observability."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path

import audit_gradient_conflicts as base
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from dataset import OpenTouchTactileDataset


FT_DIR = Path(__file__).resolve().parent


class IndexedAuditSubset(Dataset):
    def __init__(self, dataset, indices, domain):
        self.dataset = dataset
        self.indices = list(indices)
        self.domain = str(domain)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        item = self.dataset[source_index]
        record = self.dataset.samples[source_index]
        item["audit_domain"] = self.domain
        item["audit_group"] = group_key(record, self.domain)
        return item


def group_key(record, domain):
    name = Path(record["sample_dir"]).name
    if str(domain).lower() in ("touchanything", "egotouch", "ta"):
        parts = name.split("__")
        group = "__".join(parts[:-1]) if len(parts) > 1 else name
    else:
        parts = name.rsplit("_", 2)
        group = parts[0] if len(parts) == 3 else name
    return f"{domain}:{group}"


def parse_int_csv(value):
    output = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not output:
        raise ValueError("--block_indices must contain at least one index")
    return output


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
    parser.add_argument("--samples_per_domain", type=int, default=2000)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--index_workers", type=int, default=32)
    parser.add_argument("--index_chunksize", type=int, default=512)
    parser.add_argument("--index_backend", choices=("process", "thread"), default="process")
    parser.add_argument("--index_cache_dir", default=str(FT_DIR / "index_cache"))
    parser.add_argument("--index_cache_timeout", type=int, default=3600)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--block_indices", default="7,15,23,31", help="Zero-based HaMeR ViT block indices.")
    parser.add_argument("--reduced_grid_channels", type=int, default=64)
    parser.add_argument("--probe_epochs", type=int, default=25)
    parser.add_argument("--probe_batch_size", type=int, default=512)
    parser.add_argument("--probe_lr", type=float, default=0.01)
    parser.add_argument("--probe_weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--zero_support_thr", type=float, default=0.005)
    parser.add_argument("--active_pressure_thr", type=float, default=0.05)
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--progress_position", type=int, default=0)
    parser.add_argument("--disable_progress", action="store_true")
    return parser.parse_args()


def make_base_dataset(args, cfg, domain, explicit_dir):
    data_dir = base.resolve_domain_dir(domain, explicit_dir)
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
    return dataset, data_dir


def sampled_indices(length, count, seed):
    count = min(max(1, count), length)
    return random.Random(seed).sample(range(length), count)


def make_loader(args, dataset_a, dataset_b):
    subset_a = IndexedAuditSubset(
        dataset_a,
        sampled_indices(len(dataset_a), args.samples_per_domain, args.seed),
        args.domain_a,
    )
    subset_b = IndexedAuditSubset(
        dataset_b,
        sampled_indices(len(dataset_b), args.samples_per_domain, args.seed + 1),
        args.domain_b,
    )
    kwargs = {
        "dataset": ConcatDataset([subset_a, subset_b]),
        "batch_size": args.feature_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = args.persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def stable_val_group(group, seed, val_fraction):
    digest = hashlib.sha1(f"{seed}:{group}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return bucket < val_fraction


def reduce_spatial_grid(feature_map, output_channels):
    pooled = F.adaptive_avg_pool2d(feature_map, (4, 3))
    batch, channels, height, width = pooled.shape
    output_channels = min(int(output_channels), channels)
    if channels % output_channels != 0:
        output_channels = math.gcd(channels, output_channels)
    grouped = pooled.reshape(batch, output_channels, channels // output_channels, height, width).mean(dim=2)
    return grouped.flatten(1)


def extract_features(model, loader, block_indices, args, device):
    captures = {}
    handles = []

    def hook_for(index):
        def hook(_module, _inputs, output):
            captures[index] = output
        return hook

    blocks = model.backbone.blocks
    for index in block_indices:
        if index < 0 or index >= len(blocks):
            raise ValueError(f"ViT block index {index} is outside [0, {len(blocks) - 1}]")
        handles.append(blocks[index].register_forward_hook(hook_for(index)))

    feature_parts = {}
    labels = {"domain": [], "volume": [], "support": [], "active": []}
    groups = []
    try:
        progress = tqdm(
            loader,
            desc=f"features {args.ckpt}",
            position=args.progress_position,
            dynamic_ncols=True,
            disable=args.disable_progress,
        )
        for batch in progress:
            images = batch["img"].to(device, non_blocking=True)
            captures.clear()
            with torch.no_grad():
                final_map = model.backbone(images[:, :, :, 32:-32])
                representations = {}
                for index in block_indices:
                    tokens = model.backbone.last_norm(captures[index])
                    representations[f"block_{index + 1:02d}_gap"] = tokens.mean(dim=1)
                flat = final_map.flatten(2)
                representations["final_gap"] = flat.mean(dim=2)
                representations["final_spatial_moments"] = torch.cat(
                    [flat.mean(dim=2), flat.std(dim=2, unbiased=False)],
                    dim=1,
                )
                representations["final_grid4x3"] = reduce_spatial_grid(
                    final_map,
                    args.reduced_grid_channels,
                )
                representations["tactile_encoder_512"] = model.tactile_head.encoder(final_map)

            for name, values in representations.items():
                feature_parts.setdefault(name, []).append(values.detach().float().cpu())

            target = batch["tactile_signal"].float()
            palm = batch["palm_mask"].float()
            palm_count = palm.sum(dim=-1).clamp_min(1.0)
            volume = (target * palm).sum(dim=-1)
            support = ((target >= args.zero_support_thr).float() * palm).sum(dim=-1) / palm_count
            active = ((target >= args.active_pressure_thr).float() * palm).sum(dim=-1) / palm_count
            domain_values = [0.0 if value == args.domain_a else 1.0 for value in batch["audit_domain"]]
            labels["domain"].append(torch.tensor(domain_values, dtype=torch.float32))
            labels["volume"].append(volume)
            labels["support"].append(support)
            labels["active"].append(active)
            groups.extend(batch["audit_group"])
    finally:
        for handle in handles:
            handle.remove()

    features = {name: torch.cat(parts, dim=0) for name, parts in feature_parts.items()}
    labels = {name: torch.cat(parts, dim=0) for name, parts in labels.items()}
    return features, labels, groups


def split_masks(groups, domains, args):
    val = torch.tensor(
        [stable_val_group(group, args.seed, args.val_fraction) for group in groups],
        dtype=torch.bool,
    )
    train = ~val
    for domain_value in (0.0, 1.0):
        if not ((domains == domain_value) & train).any() or not ((domains == domain_value) & val).any():
            raise RuntimeError("Group split produced an empty train/val domain; change --seed or --val_fraction")
    return train, val


def standardize_features(features, train_mask):
    train = features[train_mask]
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1e-5)
    return (features - mean) / std


def train_linear_probe(
    features,
    target,
    train_mask,
    val_mask,
    task,
    args,
    device,
    progress=None,
    representation=None,
):
    features = standardize_features(features, train_mask)
    if task == "domain":
        transformed_target = target
        target_mean = torch.tensor(0.0)
        target_std = torch.tensor(1.0)
    elif task == "volume":
        transformed_target = torch.log1p(target)
        target_mean = transformed_target[train_mask].mean()
        target_std = transformed_target[train_mask].std(unbiased=False).clamp_min(1e-5)
        transformed_target = (transformed_target - target_mean) / target_std
    else:
        target_mean = target[train_mask].mean()
        target_std = target[train_mask].std(unbiased=False).clamp_min(1e-5)
        transformed_target = (target - target_mean) / target_std

    model = nn.Linear(features.shape[1], 1).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
    )
    train_indices = torch.nonzero(train_mask, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(args.seed + features.shape[1] + len(task))
    model.train()
    for epoch in range(args.probe_epochs):
        order = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        epoch_loss = 0.0
        epoch_samples = 0
        for start in range(0, order.numel(), args.probe_batch_size):
            indices = order[start:start + args.probe_batch_size]
            x = features[indices].to(device, non_blocking=True)
            y = transformed_target[indices].to(device, non_blocking=True).unsqueeze(1)
            prediction = model(x)
            loss = F.binary_cross_entropy_with_logits(prediction, y) if task == "domain" else F.mse_loss(prediction, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * indices.numel()
            epoch_samples += indices.numel()
        if progress is not None:
            progress.set_postfix(
                representation=representation,
                task=task,
                epoch=f"{epoch + 1}/{args.probe_epochs}",
                loss=f"{epoch_loss / max(epoch_samples, 1):.4g}",
                refresh=False,
            )
            progress.update(1)

    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, features.shape[0], args.probe_batch_size):
            predictions.append(model(features[start:start + args.probe_batch_size].to(device)).squeeze(1).cpu())
    prediction = torch.cat(predictions)
    if task == "domain":
        prediction = torch.sigmoid(prediction)
    else:
        prediction = prediction * target_std + target_mean
        prediction = torch.expm1(prediction).clamp_min(0.0) if task == "volume" else prediction
    return prediction


def classification_metrics(prediction, target, mask):
    predicted = prediction[mask] >= 0.5
    truth = target[mask] >= 0.5
    accuracy = (predicted == truth).float().mean().item()
    recalls = []
    for value in (False, True):
        selected = truth == value
        recalls.append((predicted[selected] == truth[selected]).float().mean().item())
    return {"accuracy": accuracy, "balanced_accuracy": statistics.fmean(recalls)}


def regression_metrics(prediction, target, mask):
    prediction = prediction[mask]
    target = target[mask]
    residual = prediction - target
    denominator = ((target - target.mean()) ** 2).sum().clamp_min(1e-8)
    return {
        "mae": residual.abs().mean().item(),
        "rmse": residual.pow(2).mean().sqrt().item(),
        "r2": (1.0 - residual.pow(2).sum() / denominator).item(),
    }


def evaluate_probes(features, labels, train_mask, val_mask, args, device):
    rows = []
    progress = tqdm(
        total=len(features) * 4 * args.probe_epochs,
        desc=f"linear probes {args.ckpt}",
        position=args.progress_position,
        dynamic_ncols=True,
        disable=args.disable_progress,
    )
    for representation, values in features.items():
        for task in ("domain", "volume", "support", "active"):
            prediction = train_linear_probe(
                values,
                labels[task],
                train_mask,
                val_mask,
                task,
                args,
                device,
                progress=progress,
                representation=representation,
            )
            if task == "domain":
                metrics = classification_metrics(prediction, labels[task], val_mask)
                rows.append({
                    "representation": representation,
                    "dimension": values.shape[1],
                    "task": task,
                    "subset": "all",
                    **metrics,
                })
            else:
                subsets = {"all": val_mask}
                subsets[args.domain_a] = val_mask & (labels["domain"] == 0.0)
                subsets[args.domain_b] = val_mask & (labels["domain"] == 1.0)
                for subset, mask in subsets.items():
                    rows.append({
                        "representation": representation,
                        "dimension": values.shape[1],
                        "task": task,
                        "subset": subset,
                        **regression_metrics(prediction, labels[task], mask),
                    })
    progress.close()
    return rows


def write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val_fraction must be in (0, 1)")
    if args.samples_per_domain < 2:
        raise ValueError("--samples_per_domain must be >= 2")
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires a CUDA GPU")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    print(f"[probe 1/5] Loading {args.ckpt} checkpoint and HaMeR model...", flush=True)
    checkpoint_path, exp_dir = base.resolve_checkpoint(args)
    model, _loss_config, model_config = base.load_model(args, checkpoint_path, exp_dir, device)
    cfg = base.load_model_cfg()
    print(f"[probe 2/5] Building {args.domain_a}/{args.domain_b} indices...", flush=True)
    dataset_a, data_dir_a = make_base_dataset(args, cfg, args.domain_a, args.domain_a_dir)
    dataset_b, data_dir_b = make_base_dataset(args, cfg, args.domain_b, args.domain_b_dir)
    loader = make_loader(args, dataset_a, dataset_b)
    block_indices = parse_int_csv(args.block_indices)
    print(
        f"[probe 3/5] Extracting {len(loader.dataset)} samples on GPU {args.gpu}...",
        flush=True,
    )
    features, labels, groups = extract_features(model, loader, block_indices, args, device)
    train_mask, val_mask = split_masks(groups, labels["domain"], args)

    del model
    torch.cuda.empty_cache()
    print("[probe 4/5] Training domain and tactile linear probes...", flush=True)
    rows = evaluate_probes(features, labels, train_mask, val_mask, args, device)

    selector = "rmse-best" if args.ckpt == "best" else args.ckpt
    output_dir = Path(args.output_dir) if args.output_dir else (
        FT_DIR / "feature_probe_audits" / f"{base.safe_name(args.exp_name)}_{selector}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[probe 5/5] Writing probe metrics and provenance...", flush=True)
    write_csv(output_dir / "probe_metrics.csv", rows)

    split_summary = {
        "train_count": int(train_mask.sum()),
        "val_count": int(val_mask.sum()),
        "train_domain_a": int((train_mask & (labels["domain"] == 0.0)).sum()),
        "train_domain_b": int((train_mask & (labels["domain"] == 1.0)).sum()),
        "val_domain_a": int((val_mask & (labels["domain"] == 0.0)).sum()),
        "val_domain_b": int((val_mask & (labels["domain"] == 1.0)).sum()),
        "unique_groups": len(set(groups)),
    }
    config = {
        "args": vars(args),
        "checkpoint": str(checkpoint_path),
        "data_dirs": {args.domain_a: data_dir_a, args.domain_b: data_dir_b},
        "dataset_sample_counts": {args.domain_a: len(dataset_a), args.domain_b: len(dataset_b)},
        "representations": {name: list(values.shape) for name, values in features.items()},
        "split_summary": split_summary,
        "model_config": model_config,
    }
    with (output_dir / "audit_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    if args.save_features:
        torch.save(
            {"features": features, "labels": labels, "groups": groups, "train_mask": train_mask, "val_mask": val_mask},
            output_dir / "feature_cache.pt",
        )

    print("\nDomain probe accuracy:")
    for row in rows:
        if row["task"] == "domain":
            print(
                f"  {row['representation']:26s} dim={row['dimension']:4d} "
                f"acc={row['accuracy']:.4f} balanced={row['balanced_accuracy']:.4f}"
            )
    print(f"\nWrote feature probe audit to: {output_dir}")


if __name__ == "__main__":
    main()
