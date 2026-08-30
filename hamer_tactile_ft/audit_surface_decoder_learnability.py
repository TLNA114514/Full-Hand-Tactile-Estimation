#!/usr/bin/env python3
"""Layered learnability audit for the canonical surface-basis decoder.

This is deliberately separate from formal training.  It freezes one crop1.2
FullGrid feature cache, builds ridge-optimal coefficient targets once, and then
compares decoder capacity and coefficient supervision without changing DINO or
the tactile dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.audit_canonical_localization import (
    MetricAccumulator,
    _validate_checkpoint_provenance,
)
from hamer_tactile_ft.audit_local_controllability import (
    CacheGroup,
    _available_fields,
    _load_decoder,
    _palm_mask,
)
from hamer_tactile_ft.hamer_tactile import ResidualBlock
from hamer_tactile_ft.losses import TactileLossConfig, compute_tactile_loss
from hamer_tactile_ft.tactile_metrics import touchanything_protocol_group_key
from tactile_input_priors.feature_cache import sha256_file


DATASET_SCHEMA = "surface_decoder_learnability_v1"
BACKGROUND_PROBABILITY = 1e-3
VARIANTS = ("linear", "nonlinear")
OBJECTIVES = ("pressure", "coefficient")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _argument_config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key != "func"}


def _stable_hash(value: str, seed: int) -> int:
    payload = f"{int(seed)}:{value}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _load_runtime_basis(path: Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, Mapping) or artifact.get("format") != "canonical_surface_basis_v1":
        raise ValueError(f"Not a canonical_surface_basis_v1 artifact: {path}")
    result = dict(artifact)
    for key in ("support_indices", "support_weights", "valid_vertex_indices"):
        if key not in result or not torch.is_tensor(result[key]):
            raise ValueError(f"Surface basis artifact lacks tensor {key!r}")
    indices = result["support_indices"].long()
    weights = result["support_weights"].float()
    if indices.shape != weights.shape or indices.ndim != 2:
        raise ValueError("Surface support indices and weights must be matching [V,S] tensors")
    coefficient_dim = int(result.get("metadata", {}).get("coefficient_dim", 0))
    if coefficient_dim <= int(indices.max().item()):
        raise ValueError("Surface basis coefficient dimension does not contain all supports")
    result["coefficient_dim"] = coefficient_dim
    return result


def _dense_basis(artifact: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    indices = artifact["support_indices"].to(device=device, dtype=torch.long)
    weights = artifact["support_weights"].to(device=device, dtype=torch.float32)
    rows, support = indices.shape
    basis = torch.zeros(
        (rows, int(artifact["coefficient_dim"])), device=device, dtype=torch.float32
    )
    row_indices = torch.arange(rows, device=device)[:, None].expand(rows, support)
    basis.index_put_((row_indices, indices), weights, accumulate=True)
    return basis


def _open_atomic_memmap(path: Path, *, dtype: Any, shape: tuple[int, ...]):
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    value = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    return temporary, value


def _finish_memmap(temporary: Path, value: np.memmap, destination: Path) -> None:
    value.flush()
    del value
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _select_records(
    cache_root: Path,
    sample_limit: int,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    metadata_cache = CacheGroup(cache_root, ())
    try:
        sample_count = len(metadata_cache)
        count = min(int(sample_limit), sample_count)
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(sample_count, size=count, replace=False)).astype(np.int64)
        records: list[dict[str, Any]] = []
        sequence_counts: dict[str, int] = {}
        for ordinal, cache_index in enumerate(selected.tolist()):
            item = metadata_cache[int(cache_index)]
            raw_sequence = str(
                item.get("sequence_key", "") or item.get("sample_id", cache_index)
            )
            sequence = touchanything_protocol_group_key(
                raw_sequence,
                item.get("query_alias"),
            )
            sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1
            records.append(
                {
                    "ordinal": ordinal,
                    "cache_index": int(cache_index),
                    "sample_id": str(item.get("sample_id", "")),
                    "sequence_key": sequence,
                }
            )
    finally:
        metadata_cache.close()

    ordered_sequences = sorted(
        sequence_counts,
        key=lambda value: (_stable_hash(value, seed), value),
    )
    target_validation = max(1, int(round(len(selected) * float(validation_fraction))))
    validation_sequences: set[str] = set()
    validation_count = 0
    for sequence in ordered_sequences:
        if validation_count >= target_validation and validation_sequences:
            break
        validation_sequences.add(sequence)
        validation_count += sequence_counts[sequence]
    if len(validation_sequences) == len(ordered_sequences):
        validation_sequences.remove(ordered_sequences[-1])
    split = np.asarray(
        [1 if row["sequence_key"] in validation_sequences else 0 for row in records],
        dtype=np.uint8,
    )
    if not np.any(split == 0) or not np.any(split == 1):
        raise RuntimeError("Sequence-disjoint probe split produced an empty partition")
    for row, value in zip(records, split.tolist()):
        row["split"] = "validation" if value else "train"
    return selected, split, records


def _prepare_contract(args: argparse.Namespace, cache: CacheGroup, artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": DATASET_SCHEMA,
        "feature_cache": str(Path(args.feature_cache).expanduser().resolve()),
        "cache_config_sha256s": list(cache.config_sha256s),
        "base_checkpoint": str(Path(args.base_checkpoint).expanduser().resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "surface_basis": str(Path(args.surface_basis).expanduser().resolve()),
        "surface_basis_sha256": sha256_file(args.surface_basis),
        "surface_basis_tensor_sha256": str(artifact.get("metadata", {}).get("basis_sha256", "")),
        "sample_limit": int(args.sample_limit),
        "validation_fraction": float(args.validation_fraction),
        "ridge": float(args.ridge),
        "logit_epsilon": float(args.logit_epsilon),
        "seed": int(args.seed),
    }


def command_prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.feature_cache).expanduser().resolve(strict=True)
    checkpoint_path = Path(args.base_checkpoint).expanduser().resolve(strict=True)
    basis_path = Path(args.surface_basis).expanduser().resolve(strict=True)
    available = _available_fields(cache_root)
    missing = sorted({"z_rgb", "tactile_signal"} - available)
    if missing:
        raise RuntimeError(f"Feature cache lacks required fields: {missing}")
    fields = tuple(sorted({"z_rgb", "tactile_signal"} | ({"palm_mask"} & available)))
    cache = CacheGroup(cache_root, fields)
    artifact = _load_runtime_basis(basis_path)
    try:
        _validate_checkpoint_provenance(cache, checkpoint_path)
        contract = _prepare_contract(args, cache, artifact)
        contract_sha = hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode("utf-8")
        ).hexdigest()
        done_path = output_dir / "PREPARED.json"
        if done_path.is_file() and not args.force:
            existing = json.loads(done_path.read_text(encoding="utf-8"))
            if existing.get("contract_sha256") == contract_sha:
                print(f"[surface-learnability] reusing prepared dataset: {output_dir}")
                return
            raise RuntimeError(
                f"Prepared dataset contract differs under {output_dir}; use --force or a new directory"
            )
        if args.force:
            done_path.unlink(missing_ok=True)

        first = cache[0]
        tactile_dim = int(np.asarray(first["tactile_signal"]).size)
        grid_size = tuple(int(value) for value in np.asarray(first["z_rgb"]).shape[-2:])
        decoder, model_config = _load_decoder(
            checkpoint_path,
            grid_size=grid_size,
            tactile_dim=tactile_dim,
            device=torch.device(args.device),
        )
        valid_indices = artifact["valid_vertex_indices"].long().cpu().numpy()
        palm_mask = _palm_mask(first, cache.provenance)
        if not np.array_equal(np.flatnonzero(palm_mask), valid_indices):
            raise RuntimeError("Cache palm mask differs from the runtime surface basis")
        selected, split, records = _select_records(
            cache_root,
            args.sample_limit,
            args.validation_fraction,
            args.seed,
        )
        count = len(selected)
        coefficient_dim = int(artifact["coefficient_dim"])
        feature_dim = int(decoder[0].output_dim)
        valid_count = len(valid_indices)
        output_specs = {
            "features.npy": (np.float16, (count, feature_dim)),
            "targets_valid.npy": (np.float16, (count, valid_count)),
            "oracle_coefficients.npy": (np.float16, (count, coefficient_dim)),
            "base_predictions_valid.npy": (np.float16, (count, valid_count)),
            "split.npy": (np.uint8, (count,)),
        }
        partials: dict[str, Path] = {}
        arrays: dict[str, np.memmap] = {}
        for name, (dtype, shape) in output_specs.items():
            partials[name], arrays[name] = _open_atomic_memmap(
                output_dir / name, dtype=dtype, shape=shape
            )
        arrays["split.npy"][:] = split

        device = torch.device(args.device)
        basis = _dense_basis(artifact, device)
        gram = basis.T @ basis
        ridge_absolute = float(args.ridge) * max(float(gram.diagonal().mean().item()), 1e-12)
        gram.diagonal().add_(ridge_absolute)
        chol, info = torch.linalg.cholesky_ex(gram)
        if int(info.max().item()) != 0:
            raise RuntimeError("Surface-basis ridge matrix is not positive definite")
        pool = decoder[0]
        decoder.eval()
        read_cache = CacheGroup(cache_root, fields)
        coefficient_square_sum = 0.0
        coefficient_count = 0
        try:
            for start in range(0, count, int(args.batch_size)):
                stop = min(start + int(args.batch_size), count)
                items = [read_cache[int(value)] for value in selected[start:stop]]
                grid = torch.from_numpy(
                    np.stack([np.asarray(item["z_rgb"], dtype=np.float32) for item in items])
                ).to(device)
                target_full = np.stack(
                    [np.asarray(item["tactile_signal"], dtype=np.float32) for item in items]
                )
                target = torch.from_numpy(target_full[:, valid_indices]).to(device)
                with torch.inference_mode():
                    pooled = pool(grid)
                    base_prediction = torch.sigmoid(decoder(grid))[:, valid_indices]
                    bounded = target.clamp(float(args.logit_epsilon), 1.0 - float(args.logit_epsilon))
                    rhs = basis.T @ torch.logit(bounded).T
                    coefficients = torch.cholesky_solve(rhs, chol).T
                if not all(
                    bool(torch.isfinite(value).all().item())
                    for value in (pooled, base_prediction, coefficients)
                ):
                    raise FloatingPointError(f"Non-finite prepared values in rows {start}:{stop}")
                arrays["features.npy"][start:stop] = pooled.float().cpu().numpy().astype(np.float16)
                arrays["targets_valid.npy"][start:stop] = target.cpu().numpy().astype(np.float16)
                arrays["oracle_coefficients.npy"][start:stop] = (
                    coefficients.float().cpu().numpy().astype(np.float16)
                )
                arrays["base_predictions_valid.npy"][start:stop] = (
                    base_prediction.float().cpu().numpy().astype(np.float16)
                )
                coefficient_square_sum += float(coefficients.float().square().sum().item())
                coefficient_count += int(coefficients.numel())
                print(
                    f"[surface-learnability] prepared {stop:,}/{count:,}",
                    flush=True,
                )
        finally:
            read_cache.close()

        for name in output_specs:
            _finish_memmap(partials[name], arrays.pop(name), output_dir / name)
        samples_path = output_dir / "samples.jsonl"
        samples_temporary = samples_path.with_name(
            f".{samples_path.name}.partial.{os.getpid()}"
        )
        with samples_temporary.open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(samples_temporary, samples_path)
        metadata = {
            "schema": DATASET_SCHEMA,
            "contract": contract,
            "contract_sha256": contract_sha,
            "sample_count": count,
            "train_count": int((split == 0).sum()),
            "validation_count": int((split == 1).sum()),
            "sequence_disjoint": True,
            "feature_dim": feature_dim,
            "valid_vertex_count": valid_count,
            "tactile_dim": tactile_dim,
            "coefficient_dim": coefficient_dim,
            "maximum_support_count": int(artifact["support_indices"].shape[1]),
            "ridge_absolute": ridge_absolute,
            "oracle_coefficient_rms": math.sqrt(
                coefficient_square_sum / max(coefficient_count, 1)
            ),
            "model_config": model_config,
            "files": list(output_specs) + ["samples.jsonl"],
        }
        _write_json(done_path, metadata)
        print(f"[surface-learnability] prepared dataset complete: {output_dir}")
    finally:
        cache.close()


class CoefficientHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        coefficient_dim: int,
        *,
        architecture: str,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if architecture == "linear":
            self.network = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, coefficient_dim),
            )
        elif architecture == "nonlinear":
            self.network = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(hidden_dim, dropout_probability=dropout),
                ResidualBlock(hidden_dim, dropout_probability=dropout),
                nn.Linear(hidden_dim, coefficient_dim),
            )
        else:
            raise ValueError(f"Unsupported architecture={architecture!r}")
        output = self.network[-1]
        assert isinstance(output, nn.Linear)
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, float(torch.logit(torch.tensor(BACKGROUND_PROBABILITY))))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _decode_coefficients(
    coefficients: torch.Tensor,
    support_indices: torch.Tensor,
    support_weights: torch.Tensor,
) -> torch.Tensor:
    supported = coefficients[:, support_indices]
    return (supported * support_weights.to(dtype=supported.dtype)).sum(dim=-1)


def _batch_array(array: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    # Advanced indexing intentionally returns a contiguous writable batch.
    return torch.from_numpy(np.asarray(array[indices], dtype=np.float32)).to(
        device=device, non_blocking=True
    )


def _loss_config() -> TactileLossConfig:
    return TactileLossConfig(
        pressure_weight_mode="hump",
        loss_ramp_epochs=5,
        location_loss_weight=0.001,
        location_gt_volume_thr=1.0,
        location_distribution_power=2.0,
        location_min_gt_peak=0.05,
    )


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    support_indices: torch.Tensor,
    support_weights: torch.Tensor,
    coefficient_rms: float,
    loss_config: TactileLossConfig,
) -> dict[str, Any]:
    model.eval()
    prediction_metrics = MetricAccumulator("probe")
    base_metrics = MetricAccumulator("frozen_fullgrid_base")
    losses: list[float] = []
    coefficient_losses: list[float] = []
    coefficient_cosines: list[float] = []
    predicted_square_sum = 0.0
    predicted_count = 0
    ones = torch.ones(len(support_indices), device=device)
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        features = _batch_array(arrays["features"], batch_indices, device)
        target = _batch_array(arrays["targets"], batch_indices, device)
        oracle = _batch_array(arrays["coefficients"], batch_indices, device)
        coefficients = model(features)
        logits = _decode_coefficients(coefficients, support_indices, support_weights)
        prediction = torch.sigmoid(logits)
        pressure_loss, _ = compute_tactile_loss(
            prediction,
            logits,
            target,
            ones,
            None,
            ["touchanything"] * len(batch_indices),
            loss_config,
            ramp_override=1.0,
            distributed_reduce=False,
        )
        normalized_coefficient_loss = F.smooth_l1_loss(
            coefficients / coefficient_rms,
            oracle / coefficient_rms,
        )
        losses.append(float(pressure_loss.item()))
        coefficient_losses.append(float(normalized_coefficient_loss.item()))
        coefficient_cosines.extend(
            F.cosine_similarity(coefficients.float(), oracle.float(), dim=1).cpu().tolist()
        )
        predicted_square_sum += float(coefficients.float().square().sum().item())
        predicted_count += int(coefficients.numel())
        prediction_metrics.update(prediction.float().cpu().numpy(), target.float().cpu().numpy())
        base_metrics.update(
            np.asarray(arrays["base_predictions"][batch_indices], dtype=np.float32),
            target.float().cpu().numpy(),
        )
    output = prediction_metrics.summary()
    output.update(
        {
            "pressure_loss": float(np.mean(losses)),
            "normalized_coefficient_loss": float(np.mean(coefficient_losses)),
            "coefficient_cosine_mean": float(np.mean(coefficient_cosines)),
            "predicted_coefficient_rms": math.sqrt(
                predicted_square_sum / max(predicted_count, 1)
            ),
            "base": base_metrics.summary(),
        }
    )
    return output


def _load_prepared(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    done_path = path / "PREPARED.json"
    if not done_path.is_file():
        raise FileNotFoundError(f"Prepared probe dataset is incomplete: {done_path}")
    metadata = json.loads(done_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"Unsupported prepared dataset schema in {done_path}")
    arrays = {
        "features": np.load(path / "features.npy", mmap_mode="r"),
        "targets": np.load(path / "targets_valid.npy", mmap_mode="r"),
        "coefficients": np.load(path / "oracle_coefficients.npy", mmap_mode="r"),
        "base_predictions": np.load(path / "base_predictions_valid.npy", mmap_mode="r"),
        "split": np.load(path / "split.npy", mmap_mode="r"),
    }
    count = int(metadata["sample_count"])
    if any(len(value) != count for value in arrays.values()):
        raise RuntimeError("Prepared probe arrays disagree on sample count")
    return metadata, arrays


def command_run(args: argparse.Namespace) -> None:
    if args.architecture not in VARIANTS or args.objective not in OBJECTIVES:
        raise ValueError("Invalid probe variant")
    prepared_dir = Path(args.prepared_dir).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        print(f"[surface-learnability] reusing completed probe: {output_dir}")
        return
    if args.force:
        summary_path.unlink(missing_ok=True)
    metadata, arrays = _load_prepared(prepared_dir)
    basis_path = Path(args.surface_basis).expanduser().resolve(strict=True)
    artifact = _load_runtime_basis(basis_path)
    expected_basis_sha = str(
        metadata.get("contract", {}).get("surface_basis_sha256", "") or ""
    )
    if expected_basis_sha and sha256_file(basis_path) != expected_basis_sha:
        raise RuntimeError("Runtime surface basis differs from the prepared probe dataset")
    if int(artifact["coefficient_dim"]) != int(metadata["coefficient_dim"]):
        raise RuntimeError("Prepared coefficient dimension differs from runtime basis")

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    support_indices = artifact["support_indices"].to(device=device, dtype=torch.long)
    support_weights = artifact["support_weights"].to(device=device, dtype=torch.float32)
    train_indices = np.flatnonzero(np.asarray(arrays["split"]) == 0).astype(np.int64)
    validation_indices = np.flatnonzero(np.asarray(arrays["split"]) == 1).astype(np.int64)
    if args.memorization:
        rng = np.random.default_rng(seed)
        train_indices = np.sort(
            rng.choice(
                train_indices,
                size=min(int(args.memorization_samples), len(train_indices)),
                replace=False,
            )
        )
        validation_indices = train_indices.copy()

    dropout = 0.0 if args.memorization else float(args.dropout)
    model = CoefficientHead(
        int(metadata["feature_dim"]),
        int(metadata["coefficient_dim"]),
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    warmup = max(0, int(args.warmup_epochs))
    epochs = int(args.epochs)

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = float(epoch - warmup) / float(max(epochs - warmup - 1, 1))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_config = _loss_config()
    coefficient_rms = max(float(metadata["oracle_coefficient_rms"]), 1e-6)
    ones = torch.ones(len(support_indices), device=device)
    rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_metrics: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(train_indices)
        train_pressure: list[float] = []
        train_coefficient: list[float] = []
        for start in range(0, len(permutation), int(args.batch_size)):
            batch_indices = permutation[start : start + int(args.batch_size)]
            features = _batch_array(arrays["features"], batch_indices, device)
            target = _batch_array(arrays["targets"], batch_indices, device)
            oracle = _batch_array(arrays["coefficients"], batch_indices, device)
            coefficients = model(features)
            logits = _decode_coefficients(coefficients, support_indices, support_weights)
            prediction = torch.sigmoid(logits)
            pressure_loss, _ = compute_tactile_loss(
                prediction,
                logits,
                target,
                ones,
                None,
                ["touchanything"] * len(batch_indices),
                loss_config,
                ramp_override=1.0,
                distributed_reduce=False,
            )
            coefficient_loss = F.smooth_l1_loss(
                coefficients / coefficient_rms,
                oracle / coefficient_rms,
            )
            loss = pressure_loss
            if args.objective == "coefficient":
                loss = loss + float(args.coefficient_aux_weight) * coefficient_loss
            if not bool(torch.isfinite(loss.detach()).all().item()):
                raise FloatingPointError(
                    f"Non-finite probe loss at epoch={epoch}, batch_start={start}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
            optimizer.step()
            train_pressure.append(float(pressure_loss.detach().item()))
            train_coefficient.append(float(coefficient_loss.detach().item()))
        validation = _evaluate(
            model,
            arrays,
            validation_indices,
            device=device,
            batch_size=args.eval_batch_size,
            support_indices=support_indices,
            support_weights=support_weights,
            coefficient_rms=coefficient_rms,
            loss_config=loss_config,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_pressure_loss": float(np.mean(train_pressure)),
            "train_normalized_coefficient_loss": float(np.mean(train_coefficient)),
            **{f"val_{key}": value for key, value in validation.items() if key != "base"},
        }
        rows.append(row)
        if float(validation["pressure_loss"]) < best_loss:
            best_loss = float(validation["pressure_loss"])
            best_epoch = epoch
            best_metrics = validation
            torch.save(
                {
                    "format": "surface_decoder_learnability_probe_v1",
                    "architecture": args.architecture,
                    "objective": args.objective,
                    "memorization": bool(args.memorization),
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": validation,
                },
                output_dir / "best.pt",
            )
        scheduler.step()
        _write_csv(output_dir / "metrics.csv", rows)
        print(
            f"[surface-learnability] {output_dir.name} epoch={epoch:03d} "
            f"loss={validation['pressure_loss']:.6f} "
            f"contact={validation['contact_iou_010_frame_macro']:.4f} "
            f"core={validation['core_distribution_viou_frame_macro']:.4f}",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "architecture": args.architecture,
        "objective": args.objective,
        "memorization": bool(args.memorization),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "best_epoch": best_epoch,
        "best": best_metrics,
        "base": best_metrics.get("base", {}),
        "run_config": _argument_config(args),
        "prepared_contract_sha256": metadata["contract_sha256"],
    }
    _write_json(summary_path, summary)
    print(f"[surface-learnability] probe complete: {output_dir}")


def _flatten_summary(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    best = dict(value.get("best", {}))
    return {
        "probe": path.parent.name,
        "architecture": value.get("architecture"),
        "objective": value.get("objective"),
        "memorization": value.get("memorization"),
        "train_count": value.get("train_count"),
        "parameter_count": value.get("parameter_count"),
        "best_epoch": value.get("best_epoch"),
        "pressure_loss": best.get("pressure_loss"),
        "rmse_vertex_micro": best.get("rmse_vertex_micro"),
        "contact_iou": best.get("contact_iou_010_frame_macro"),
        "distribution_viou": best.get("distribution_viou_frame_macro"),
        "core_distribution_viou": best.get("core_distribution_viou_frame_macro"),
        "volumetric_iou": best.get("volumetric_iou_frame_macro"),
        "false_high_excess": best.get("false_high_excess_mean"),
        "coefficient_cosine": best.get("coefficient_cosine_mean"),
        "normalized_coefficient_loss": best.get("normalized_coefficient_loss"),
        "predicted_coefficient_rms": best.get("predicted_coefficient_rms"),
    }


def command_aggregate(args: argparse.Namespace) -> None:
    root = Path(args.input_root).expanduser().resolve(strict=True)
    rows: list[dict[str, Any]] = []
    raw: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*/summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        name = path.parent.name
        raw[name] = value
        rows.append(_flatten_summary(path, value))
    expected = {
        f"{population}_{architecture}_{objective}"
        for population in ("general", "memorize")
        for architecture in VARIANTS
        for objective in OBJECTIVES
    }
    missing = sorted(expected - set(raw))
    if missing:
        raise RuntimeError(f"Cannot aggregate incomplete probe matrix: {missing}")

    def metric(name: str, key: str) -> float:
        return float(raw[name]["best"][key])

    key_metrics = (
        "contact_iou_010_frame_macro",
        "core_distribution_viou_frame_macro",
        "distribution_viou_frame_macro",
        "rmse_vertex_micro",
    )
    contrasts: list[dict[str, Any]] = []
    for population in ("general", "memorize"):
        for objective in OBJECTIVES:
            left = f"{population}_linear_{objective}"
            right = f"{population}_nonlinear_{objective}"
            contrasts.append(
                {
                    "contrast": f"{population}_nonlinear_minus_linear_{objective}",
                    **{f"delta_{key}": metric(right, key) - metric(left, key) for key in key_metrics},
                }
            )
        for architecture in VARIANTS:
            left = f"{population}_{architecture}_pressure"
            right = f"{population}_{architecture}_coefficient"
            contrasts.append(
                {
                    "contrast": f"{population}_coefficient_minus_pressure_{architecture}",
                    **{f"delta_{key}": metric(right, key) - metric(left, key) for key in key_metrics},
                }
            )
    interpretation = {
        "schema_version": 1,
        "purpose": "separate decoder capacity, basis-loss conditioning, memorization, and generalization",
        "notes": [
            "Nonlinear-over-linear gains isolate coefficient predictor capacity on identical frozen FullGrid features.",
            "Coefficient-over-pressure gains isolate supervision/conditioning while preserving the pressure objective.",
            "Strong memorization with weak sequence-disjoint validation indicates observability or generalization rather than optimizer failure.",
            "Weak memorization in every variant indicates a decoder parameterization, scale, or optimization failure before correspondence is blamed.",
        ],
        "contrasts": contrasts,
    }
    _write_csv(root / "comparison.csv", rows)
    _write_csv(root / "contrasts.csv", contrasts)
    _write_json(root / "interpretation.json", interpretation)
    _write_json(
        root / "AUDIT_DONE.json",
        {"schema_version": 1, "probe_count": len(rows), "comparison": "comparison.csv"},
    )
    print(f"[surface-learnability] aggregate complete: {root}")


def command_self_test(_: argparse.Namespace) -> None:
    torch.manual_seed(521)
    artifact = {
        "coefficient_dim": 7,
        "support_indices": torch.tensor([[0, 1], [2, 3], [4, 6]]),
        "support_weights": torch.tensor([[0.5, 0.5], [0.25, 0.75], [1.0, 0.0]]),
    }
    coefficients = torch.randn(3, 7, requires_grad=True)
    logits = _decode_coefficients(
        coefficients, artifact["support_indices"], artifact["support_weights"]
    )
    dense = torch.zeros(3, 7)
    dense.scatter_add_(1, artifact["support_indices"], artifact["support_weights"])
    expected = coefficients @ dense.T
    torch.testing.assert_close(logits, expected)
    logits.square().mean().backward()
    if coefficients.grad is None or not torch.isfinite(coefficients.grad).all():
        raise AssertionError("Sparse surface decoder gradients are invalid")
    for architecture in VARIANTS:
        model = CoefficientHead(12, 7, architecture=architecture, hidden_dim=16, dropout=0.0)
        output = model(torch.randn(4, 12))
        if output.shape != (4, 7) or not torch.isfinite(output).all():
            raise AssertionError(f"Invalid {architecture} probe output")
    print("surface decoder learnability self-test: OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--feature-cache", required=True)
    prepare.add_argument("--base-checkpoint", required=True)
    prepare.add_argument("--surface-basis", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--sample-limit", type=int, default=32768)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    prepare.add_argument("--batch-size", type=int, default=128)
    prepare.add_argument("--ridge", type=float, default=0.001)
    prepare.add_argument("--logit-epsilon", type=float, default=0.001)
    prepare.add_argument("--seed", type=int, default=521)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=command_prepare)

    run = subparsers.add_parser("run")
    run.add_argument("--prepared-dir", required=True)
    run.add_argument("--surface-basis", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--architecture", choices=VARIANTS, required=True)
    run.add_argument("--objective", choices=OBJECTIVES, required=True)
    run.add_argument("--memorization", action="store_true")
    run.add_argument("--memorization-samples", type=int, default=1024)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--eval-batch-size", type=int, default=512)
    run.add_argument("--hidden-dim", type=int, default=1024)
    run.add_argument("--dropout", type=float, default=0.3)
    run.add_argument("--coefficient-aux-weight", type=float, default=0.25)
    run.add_argument("--lr", type=float, default=4e-4)
    run.add_argument("--weight-decay", type=float, default=1e-4)
    run.add_argument("--warmup-epochs", type=int, default=3)
    run.add_argument("--gradient-clip", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=521)
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=command_run)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input-root", required=True)
    aggregate.set_defaults(func=command_aggregate)

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(func=command_self_test)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("sample_limit", "batch_size", "epochs", "eval_batch_size", "hidden_dim"):
        if hasattr(args, name) and int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "validation_fraction") and not 0.0 < float(args.validation_fraction) < 1.0:
        raise ValueError("--validation-fraction must lie in (0, 1)")
    if hasattr(args, "ridge") and float(args.ridge) <= 0.0:
        raise ValueError("--ridge must be positive")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
