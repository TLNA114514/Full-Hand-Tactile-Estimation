#!/usr/bin/env python3
"""Attribute the K4096 surface failure to representation or RGB mapping.

The audit uses immutable frozen FullGrid caches.  It prepares official
train/val/test_seen/test_unseen splits, measures a ridge basis oracle on every
split, and trains parameter-matched nonlinear direct-valid and surface-basis
heads.  Checkpoints are selected only on official validation loss.
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
from hamer_tactile_ft.audit_surface_decoder_learnability import (
    BACKGROUND_PROBABILITY,
    CoefficientHead,
    _batch_array,
    _decode_coefficients,
    _dense_basis,
    _finish_memmap,
    _load_runtime_basis,
    _loss_config,
    _open_atomic_memmap,
)
from hamer_tactile_ft.losses import compute_tactile_loss
from tactile_input_priors.feature_cache import sha256_file


SCHEMA = "surface_mapping_attribution_v1"
SPLITS = ("train", "val", "test_seen", "test_unseen")
VARIANTS = ("basis", "direct")


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
    observed: set[str] = set()
    for row in rows:
        for key in row:
            if key not in observed:
                fields.append(key)
                observed.add(key)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _contract_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _select_cache_rows(cache: CacheGroup, sample_limit: int, seed: int) -> np.ndarray:
    count = len(cache)
    selected_count = min(count, int(sample_limit)) if int(sample_limit) > 0 else count
    if selected_count <= 0:
        raise ValueError("The prepared split must contain at least one sample")
    if selected_count == count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(count, size=selected_count, replace=False)).astype(np.int64)


def _prepare_contract(
    args: argparse.Namespace,
    cache: CacheGroup,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "split": str(args.split),
        "feature_cache": str(Path(args.feature_cache).expanduser().resolve()),
        "cache_config_sha256s": list(cache.config_sha256s),
        "base_checkpoint": str(Path(args.base_checkpoint).expanduser().resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "surface_basis": str(Path(args.surface_basis).expanduser().resolve()),
        "surface_basis_sha256": sha256_file(args.surface_basis),
        "surface_basis_tensor_sha256": str(
            artifact.get("metadata", {}).get("basis_sha256", "")
        ),
        "sample_limit": int(args.sample_limit),
        "oracle_sample_limit": int(args.oracle_sample_limit),
        "ridge": float(args.ridge),
        "logit_epsilon": float(args.logit_epsilon),
        "seed": int(args.seed),
    }


def command_prepare_split(args: argparse.Namespace) -> None:
    cache_root = Path(args.feature_cache).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    available = _available_fields(cache_root)
    missing = sorted({"z_rgb", "tactile_signal"} - available)
    if missing:
        raise RuntimeError(f"Feature cache lacks required fields: {missing}")
    fields = tuple(sorted({"z_rgb", "tactile_signal"} | ({"palm_mask"} & available)))
    cache = CacheGroup(cache_root, fields)
    artifact = _load_runtime_basis(
        Path(args.surface_basis).expanduser().resolve(strict=True)
    )
    try:
        checkpoint = Path(args.base_checkpoint).expanduser().resolve(strict=True)
        _validate_checkpoint_provenance(cache, checkpoint)
        contract = _prepare_contract(args, cache, artifact)
        contract_sha = _contract_sha256(contract)
        done_path = output_dir / "PREPARED.json"
        if done_path.is_file() and not args.force:
            current = json.loads(done_path.read_text(encoding="utf-8"))
            if current.get("contract_sha256") == contract_sha:
                print(f"[surface-attribution] reuse prepared split: {output_dir}")
                return
            raise RuntimeError(
                f"Prepared split contract differs under {output_dir}; use --force"
            )
        if args.force:
            done_path.unlink(missing_ok=True)

        selected = _select_cache_rows(cache, args.sample_limit, args.seed)
        first = cache[int(selected[0])]
        first_grid = np.asarray(first["z_rgb"])
        tactile_dim = int(np.asarray(first["tactile_signal"]).size)
        grid_size = tuple(int(value) for value in first_grid.shape[-2:])
        decoder, decoder_config = _load_decoder(
            checkpoint,
            grid_size=grid_size,
            tactile_dim=tactile_dim,
            device=torch.device(args.device),
        )
        valid_indices = artifact["valid_vertex_indices"].long().cpu().numpy()
        palm_mask = _palm_mask(first, cache.provenance)
        if not np.array_equal(np.flatnonzero(palm_mask), valid_indices):
            raise RuntimeError("Cache palm mask differs from the runtime surface basis")

        count = len(selected)
        oracle_count = min(count, int(args.oracle_sample_limit))
        oracle_rng = np.random.default_rng(int(args.seed) + 1701)
        oracle_ordinals = np.sort(
            oracle_rng.choice(count, size=oracle_count, replace=False)
        ).astype(np.int64)
        feature_dim = int(decoder[0].output_dim)
        output_specs = {
            "features.npy": (np.float16, (count, feature_dim)),
            "targets_valid.npy": (np.float16, (count, len(valid_indices))),
            "base_predictions_valid.npy": (
                np.float16,
                (count, len(valid_indices)),
            ),
        }
        partials: dict[str, Path] = {}
        arrays: dict[str, np.memmap] = {}
        for name, (dtype, shape) in output_specs.items():
            partials[name], arrays[name] = _open_atomic_memmap(
                output_dir / name, dtype=dtype, shape=shape
            )

        device = torch.device(args.device)
        basis = _dense_basis(artifact, device)
        gram = basis.T @ basis
        ridge_absolute = float(args.ridge) * max(
            float(gram.diagonal().mean().item()), 1e-12
        )
        gram.diagonal().add_(ridge_absolute)
        chol, info = torch.linalg.cholesky_ex(gram)
        if int(info.max().item()) != 0:
            raise RuntimeError("Surface-basis ridge matrix is not positive definite")
        pool = decoder[0]
        decoder.eval()
        oracle_metrics = MetricAccumulator(f"basis_oracle_{args.split}")
        base_metrics = MetricAccumulator(f"fullgrid_base_{args.split}")
        coefficient_square_sum = 0.0
        coefficient_count = 0
        sample_rows: list[dict[str, Any]] = []
        for start in range(0, count, int(args.batch_size)):
            stop = min(start + int(args.batch_size), count)
            batch_indices = selected[start:stop]
            items = [cache[int(value)] for value in batch_indices]
            grid = torch.from_numpy(
                np.stack(
                    [np.asarray(item["z_rgb"], dtype=np.float32) for item in items]
                )
            ).to(device=device, non_blocking=True)
            target_full = np.stack(
                [np.asarray(item["tactile_signal"], dtype=np.float32) for item in items]
            )
            target = torch.from_numpy(target_full[:, valid_indices]).to(
                device=device, non_blocking=True
            )
            with torch.inference_mode():
                features = pool(grid)
                base_prediction = torch.sigmoid(decoder[1:](features))[:, valid_indices]
            values = (features, base_prediction)
            if not all(bool(torch.isfinite(value).all().item()) for value in values):
                raise FloatingPointError(f"Non-finite prepared values at {start}:{stop}")
            arrays["features.npy"][start:stop] = (
                features.float().cpu().numpy().astype(np.float16)
            )
            arrays["targets_valid.npy"][start:stop] = (
                target.float().cpu().numpy().astype(np.float16)
            )
            arrays["base_predictions_valid.npy"][start:stop] = (
                base_prediction.float().cpu().numpy().astype(np.float16)
            )
            oracle_in_batch = oracle_ordinals[
                (oracle_ordinals >= start) & (oracle_ordinals < stop)
            ]
            if len(oracle_in_batch):
                local = torch.from_numpy(oracle_in_batch - start).to(
                    device=device, dtype=torch.long
                )
                oracle_target = target.index_select(0, local)
                with torch.inference_mode():
                    bounded = oracle_target.clamp(
                        float(args.logit_epsilon),
                        1.0 - float(args.logit_epsilon),
                    )
                    rhs = basis.T @ torch.logit(bounded).T
                    coefficients = torch.cholesky_solve(rhs, chol).T
                    oracle_prediction = torch.sigmoid(coefficients @ basis.T)
                if not all(
                    bool(torch.isfinite(value).all().item())
                    for value in (coefficients, oracle_prediction)
                ):
                    raise FloatingPointError(
                        f"Non-finite oracle values at {start}:{stop}"
                    )
                oracle_target_numpy = oracle_target.float().cpu().numpy()
                oracle_metrics.update(
                    oracle_prediction.float().cpu().numpy(), oracle_target_numpy
                )
                base_metrics.update(
                    base_prediction.index_select(0, local).float().cpu().numpy(),
                    oracle_target_numpy,
                )
                coefficient_square_sum += float(
                    coefficients.float().square().sum().item()
                )
                coefficient_count += int(coefficients.numel())
            for cache_index, item in zip(batch_indices.tolist(), items):
                sample_rows.append(
                    {
                        "cache_index": int(cache_index),
                        "sample_id": str(item.get("sample_id", "")),
                        "sequence_key": str(item.get("sequence_key", "")),
                        "query_alias": str(item.get("query_alias", "")),
                    }
                )
            print(
                f"[surface-attribution] split={args.split} prepared {stop:,}/{count:,}",
                flush=True,
            )

        for name in output_specs:
            _finish_memmap(partials[name], arrays.pop(name), output_dir / name)
        samples_path = output_dir / "samples.jsonl"
        temporary = samples_path.with_name(f".{samples_path.name}.partial.{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in sample_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, samples_path)

        oracle_summary = {
            "split": str(args.split),
            "sample_count": count,
            "oracle_sample_count": oracle_count,
            "ridge_absolute": ridge_absolute,
            "oracle_coefficient_rms": math.sqrt(
                coefficient_square_sum / max(coefficient_count, 1)
            ),
            "basis_oracle": oracle_metrics.summary(),
            "fullgrid_base": base_metrics.summary(),
        }
        _write_json(output_dir / "oracle_summary.json", oracle_summary)
        prepared = {
            "schema": SCHEMA,
            "contract": contract,
            "contract_sha256": contract_sha,
            "split": str(args.split),
            "sample_count": count,
            "feature_dim": feature_dim,
            "valid_vertex_count": len(valid_indices),
            "coefficient_dim": int(artifact["coefficient_dim"]),
            "grid_size": list(grid_size),
            "pool_output_channels": int(feature_dim // (grid_size[0] * grid_size[1])),
            "decoder_config": decoder_config,
            "oracle_summary_sha256": sha256_file(output_dir / "oracle_summary.json"),
            "samples_sha256": sha256_file(samples_path),
        }
        _write_json(done_path, prepared)
        print(f"[surface-attribution] prepared split complete: {output_dir}")
    finally:
        cache.close()


def _load_prepared_split(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    done_path = path / "PREPARED.json"
    if not done_path.is_file():
        raise FileNotFoundError(f"Prepared attribution split is incomplete: {done_path}")
    metadata = json.loads(done_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported prepared schema in {done_path}")
    arrays = {
        "features": np.load(path / "features.npy", mmap_mode="r"),
        "targets": np.load(path / "targets_valid.npy", mmap_mode="r"),
        "base_predictions": np.load(
            path / "base_predictions_valid.npy", mmap_mode="r"
        ),
    }
    count = int(metadata["sample_count"])
    if any(len(value) != count for value in arrays.values()):
        raise RuntimeError(f"Prepared arrays disagree on sample count under {path}")
    return metadata, arrays


def _parameter_count(feature_dim: int, hidden_dim: int, output_dim: int) -> int:
    # Exact count of CoefficientHead(..., architecture="nonlinear").
    return (
        2 * int(feature_dim)
        + 4 * int(hidden_dim) ** 2
        + int(hidden_dim) * (int(feature_dim) + int(output_dim) + 15)
        + int(output_dim)
    )


def _matched_direct_hidden_dim(
    feature_dim: int,
    coefficient_dim: int,
    valid_vertex_count: int,
    basis_hidden_dim: int,
) -> int:
    target = _parameter_count(feature_dim, basis_hidden_dim, coefficient_dim)
    candidates = range(64, max(65, 2 * int(basis_hidden_dim) + 1))
    return min(
        candidates,
        key=lambda hidden: abs(
            _parameter_count(feature_dim, hidden, valid_vertex_count) - target
        ),
    )


class AttributionHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        dropout: float,
        support_indices: torch.Tensor | None,
        support_weights: torch.Tensor | None,
    ):
        super().__init__()
        self.network = CoefficientHead(
            feature_dim,
            output_dim,
            architecture="nonlinear",
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        if support_indices is None:
            self.register_buffer("support_indices", torch.empty(0, dtype=torch.long))
            self.register_buffer("support_weights", torch.empty(0, dtype=torch.float32))
            self.uses_basis = False
        else:
            self.register_buffer("support_indices", support_indices.long())
            self.register_buffer("support_weights", support_weights.float())
            self.uses_basis = True

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.network(features)
        logits = (
            _decode_coefficients(latent, self.support_indices, self.support_weights)
            if self.uses_basis
            else latent
        )
        return logits, latent


@torch.inference_mode()
def _evaluate_model(
    model: AttributionHead,
    arrays: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    metrics = MetricAccumulator("surface_attribution")
    base_metrics = MetricAccumulator("frozen_fullgrid_base")
    losses: list[float] = []
    latent_square_sum = 0.0
    latent_count = 0
    ones = torch.ones(arrays["targets"].shape[1], device=device)
    loss_config = _loss_config()
    indices = np.arange(len(arrays["features"]), dtype=np.int64)
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        features = _batch_array(arrays["features"], batch_indices, device)
        target = _batch_array(arrays["targets"], batch_indices, device)
        logits, latent = model(features)
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
        losses.append(float(pressure_loss.item()))
        latent_square_sum += float(latent.float().square().sum().item())
        latent_count += int(latent.numel())
        target_numpy = target.float().cpu().numpy()
        metrics.update(prediction.float().cpu().numpy(), target_numpy)
        base_metrics.update(
            np.asarray(arrays["base_predictions"][batch_indices], dtype=np.float32),
            target_numpy,
        )
    result = metrics.summary()
    result.update(
        {
            "pressure_loss": float(np.mean(losses)),
            "latent_rms": math.sqrt(latent_square_sum / max(latent_count, 1)),
            "base": base_metrics.summary(),
        }
    )
    return result


def _load_mesh_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append(tuple(float(value) for value in parts[1:4]))
    result = np.asarray(vertices, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 3:
        raise RuntimeError(f"Could not read canonical vertices from {path}")
    return result


@torch.inference_mode()
def _token_influence_audit(
    model: AttributionHead,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    output_dir: Path,
    device: torch.device,
    sample_count: int,
    anchor_count: int,
    token_chunk: int,
    seed: int,
) -> dict[str, Any]:
    feature_dim = int(metadata["feature_dim"])
    grid_height, grid_width = (int(value) for value in metadata["grid_size"])
    token_count = grid_height * grid_width
    channels = feature_dim // token_count
    if channels * token_count != feature_dim:
        raise RuntimeError("Prepared FullGrid feature dimension is not token-factorable")
    rng = np.random.default_rng(int(seed))
    count = min(int(sample_count), len(arrays["features"]))
    selected = np.sort(
        rng.choice(len(arrays["features"]), size=count, replace=False)
    ).astype(np.int64)
    features = _batch_array(arrays["features"], selected, device)
    model.eval()
    base_logits, _ = model(features)
    if "anchor_prefix_valid_indices" not in artifact:
        raise KeyError("Runtime surface basis lacks canonical anchor indices")
    anchor_rows = artifact["anchor_prefix_valid_indices"].long()
    anchor_rows = anchor_rows[: min(int(anchor_count), len(anchor_rows))].to(device)
    influence = torch.zeros(
        (token_count, len(anchor_rows)), device=device, dtype=torch.float32
    )
    feature_grid = features.reshape(count, channels, token_count)
    channel_mean = feature_grid.mean(dim=2)
    for start in range(0, token_count, int(token_chunk)):
        token_ids = torch.arange(
            start, min(start + int(token_chunk), token_count), device=device
        )
        copies = features[:, None, :].expand(-1, len(token_ids), -1).clone()
        copies = copies.reshape(count, len(token_ids), channels, token_count)
        for local_index, token_index in enumerate(token_ids.tolist()):
            copies[:, local_index, :, token_index] = channel_mean
        perturbed_logits, _ = model(copies.reshape(-1, feature_dim))
        perturbed_logits = perturbed_logits.reshape(
            count, len(token_ids), -1
        )[:, :, anchor_rows]
        delta = (
            perturbed_logits.float()
            - base_logits[:, None, anchor_rows].float()
        ).abs().mean(dim=0)
        influence[token_ids] = delta

    anchor_influence = influence.T
    normalized = anchor_influence / anchor_influence.sum(dim=1, keepdim=True).clamp_min(
        1e-12
    )
    entropy = -(normalized * normalized.clamp_min(1e-12).log()).sum(dim=1)
    normalized_entropy = entropy / math.log(token_count)
    effective_tokens = entropy.exp()
    unit = normalized / normalized.square().sum(dim=1, keepdim=True).sqrt().clamp_min(
        1e-12
    )
    similarities = unit @ unit.T
    upper = torch.triu_indices(len(anchor_rows), len(anchor_rows), offset=1, device=device)
    pair_similarity = similarities[upper[0], upper[1]].cpu().numpy()

    valid_indices = artifact["valid_vertex_indices"].long()
    global_anchor_indices = valid_indices[anchor_rows.cpu()].numpy()
    mesh_path = Path(str(artifact.get("metadata", {}).get("mesh", ""))).expanduser()
    distance_correlation = float("nan")
    if mesh_path.is_file() and len(global_anchor_indices) > 1:
        xyz = _load_mesh_vertices(mesh_path)[global_anchor_indices]
        distances = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
        pair_distances = distances[
            upper[0].cpu().numpy(), upper[1].cpu().numpy()
        ]
        if np.std(pair_distances) > 0.0 and np.std(pair_similarity) > 0.0:
            distance_correlation = float(
                np.corrcoef(pair_distances, pair_similarity)[0, 1]
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "token_influence.npy",
        anchor_influence.float().cpu().numpy(),
    )
    anchor_rows_numpy = anchor_rows.cpu().numpy()
    top_tokens = normalized.argmax(dim=1).cpu().numpy()
    _write_csv(
        output_dir / "anchor_token_summary.csv",
        [
            {
                "anchor_ordinal": index,
                "valid_vertex_row": int(anchor_rows_numpy[index]),
                "canonical_vertex_index": int(global_anchor_indices[index]),
                "top_token": int(top_tokens[index]),
                "top_token_y": int(top_tokens[index] // grid_width),
                "top_token_x": int(top_tokens[index] % grid_width),
                "normalized_entropy": float(normalized_entropy[index].item()),
                "effective_token_count": float(effective_tokens[index].item()),
                "influence_sum": float(anchor_influence[index].sum().item()),
            }
            for index in range(len(anchor_rows))
        ],
    )
    return {
        "sample_count": count,
        "anchor_count": len(anchor_rows),
        "token_count": token_count,
        "normalized_entropy_mean": float(normalized_entropy.mean().item()),
        "effective_token_count_mean": float(effective_tokens.mean().item()),
        "top_token_unique_count": int(np.unique(top_tokens).size),
        "top_token_unique_fraction": float(np.unique(top_tokens).size / len(top_tokens)),
        "anchor_influence_cosine_mean": float(np.mean(pair_similarity)),
        "canonical_xyz_distance_influence_correlation": distance_correlation,
        "token_influence_sha256": sha256_file(output_dir / "token_influence.npy"),
    }


def command_train(args: argparse.Namespace) -> None:
    prepared_root = Path(args.prepared_root).expanduser().resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        print(f"[surface-attribution] reuse completed head: {output_dir}")
        return
    if args.force:
        summary_path.unlink(missing_ok=True)

    prepared: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {
        split: _load_prepared_split(prepared_root / split) for split in SPLITS
    }
    contracts = {
        str(value[0]["contract"]["surface_basis_sha256"])
        for value in prepared.values()
    }
    checkpoints = {
        str(value[0]["contract"]["base_checkpoint_sha256"])
        for value in prepared.values()
    }
    if len(contracts) != 1 or len(checkpoints) != 1:
        raise RuntimeError("Prepared splits do not share one basis/base contract")
    train_metadata, train_arrays = prepared["train"]
    feature_dim = int(train_metadata["feature_dim"])
    valid_count = int(train_metadata["valid_vertex_count"])
    coefficient_dim = int(train_metadata["coefficient_dim"])
    basis_path = Path(args.surface_basis).expanduser().resolve(strict=True)
    if sha256_file(basis_path) != next(iter(contracts)):
        raise RuntimeError("Runtime surface basis differs from prepared splits")
    artifact = _load_runtime_basis(basis_path)

    basis_hidden = int(args.basis_hidden_dim)
    direct_hidden = int(args.direct_hidden_dim)
    if direct_hidden <= 0:
        direct_hidden = _matched_direct_hidden_dim(
            feature_dim, coefficient_dim, valid_count, basis_hidden
        )
    variant = str(args.variant)
    hidden_dim = basis_hidden if variant == "basis" else direct_hidden
    output_dim = coefficient_dim if variant == "basis" else valid_count
    support_indices = artifact["support_indices"] if variant == "basis" else None
    support_weights = artifact["support_weights"] if variant == "basis" else None

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = AttributionHead(
        feature_dim,
        output_dim,
        hidden_dim=hidden_dim,
        dropout=float(args.dropout),
        support_indices=support_indices,
        support_weights=support_weights,
    ).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    target_parameter_count = _parameter_count(
        feature_dim, basis_hidden, coefficient_dim
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    epochs = int(args.epochs)
    warmup = max(0, int(args.warmup_epochs))

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = float(epoch - warmup) / float(max(epochs - warmup - 1, 1))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_indices = np.arange(len(train_arrays["features"]), dtype=np.int64)
    val_arrays = prepared["val"][1]
    loss_config = _loss_config()
    ones = torch.ones(valid_count, device=device)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = -1
    best_path = output_dir / "best_loss.pt"
    for epoch in range(epochs):
        model.train()
        permutation = rng.permutation(train_indices)
        train_losses: list[float] = []
        for start in range(0, len(permutation), int(args.batch_size)):
            batch_indices = permutation[start : start + int(args.batch_size)]
            features = _batch_array(train_arrays["features"], batch_indices, device)
            target = _batch_array(train_arrays["targets"], batch_indices, device)
            logits, _ = model(features)
            prediction = torch.sigmoid(logits)
            loss, _ = compute_tactile_loss(
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
            if not bool(torch.isfinite(loss).all().item()):
                raise FloatingPointError(
                    f"Non-finite attribution loss at epoch={epoch}, start={start}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.gradient_clip))
            optimizer.step()
            train_losses.append(float(loss.detach().item()))
        validation = _evaluate_model(
            model,
            val_arrays,
            device=device,
            batch_size=args.eval_batch_size,
        )
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_pressure_loss": float(np.mean(train_losses)),
            **{
                f"val_{key}": value
                for key, value in validation.items()
                if key != "base"
            },
        }
        rows.append(row)
        if float(validation["pressure_loss"]) < best_loss:
            best_loss = float(validation["pressure_loss"])
            best_epoch = epoch
            temporary = best_path.with_name(f".{best_path.name}.partial.{os.getpid()}")
            torch.save(
                {
                    "format": SCHEMA,
                    "variant": variant,
                    "hidden_dim": hidden_dim,
                    "parameter_count": parameter_count,
                    "epoch": epoch,
                    "val_metrics": validation,
                    "state_dict": model.state_dict(),
                },
                temporary,
            )
            os.replace(temporary, best_path)
        scheduler.step()
        _write_csv(output_dir / "metrics.csv", rows)
        print(
            f"[surface-attribution] {variant} epoch={epoch:03d} "
            f"val_loss={validation['pressure_loss']:.6f} "
            f"contact={validation['contact_iou_010_frame_macro']:.4f} "
            f"core={validation['core_distribution_viou_frame_macro']:.4f}",
            flush=True,
        )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    split_metrics = {
        split: _evaluate_model(
            model,
            arrays,
            device=device,
            batch_size=args.eval_batch_size,
        )
        for split, (_, arrays) in prepared.items()
    }
    influence = _token_influence_audit(
        model,
        prepared["val"][1],
        prepared["val"][0],
        artifact,
        output_dir=output_dir / "token_influence",
        device=device,
        sample_count=args.influence_samples,
        anchor_count=args.influence_anchors,
        token_chunk=args.influence_token_chunk,
        seed=seed,
    )
    summary = {
        "schema": SCHEMA,
        "variant": variant,
        "selection": "official_val_pressure_loss",
        "best_epoch": best_epoch,
        "hidden_dim": hidden_dim,
        "parameter_count": parameter_count,
        "target_basis_parameter_count": target_parameter_count,
        "parameter_count_delta": parameter_count - target_parameter_count,
        "parameter_count_relative_delta": (
            parameter_count / target_parameter_count - 1.0
        ),
        "split_metrics": split_metrics,
        "token_influence": influence,
        "prepared_contract_sha256s": {
            split: value[0]["contract_sha256"] for split, value in prepared.items()
        },
        "run_config": {
            key: value for key, value in vars(args).items() if key != "func"
        },
        "best_checkpoint_sha256": sha256_file(best_path),
    }
    _write_json(summary_path, summary)
    print(f"[surface-attribution] head complete: {output_dir}")


def command_aggregate(args: argparse.Namespace) -> None:
    root = Path(args.input_root).expanduser().resolve(strict=True)
    summaries = {
        variant: json.loads((root / variant / "summary.json").read_text(encoding="utf-8"))
        for variant in VARIANTS
    }
    comparison_rows: list[dict[str, Any]] = []
    metric_keys = (
        "pressure_loss",
        "rmse_vertex_micro",
        "contact_iou_010_frame_macro",
        "volumetric_iou_frame_macro",
        "distribution_viou_frame_macro",
        "core_distribution_viou_frame_macro",
        "false_high_excess_mean",
    )
    for split in SPLITS:
        basis = summaries["basis"]["split_metrics"][split]
        direct = summaries["direct"]["split_metrics"][split]
        base = basis["base"]
        for variant, metrics in (("fullgrid_base", base), ("basis", basis), ("direct", direct)):
            row = {"split": split, "variant": variant}
            for key in metric_keys:
                if key in metrics:
                    row[key] = metrics[key]
                    if key in base:
                        row[f"delta_{key}_vs_fullgrid"] = float(metrics[key]) - float(base[key])
            if variant in VARIANTS:
                for key in metric_keys:
                    if key in metrics and key in basis:
                        row[f"delta_{key}_vs_basis"] = float(metrics[key]) - float(basis[key])
            comparison_rows.append(row)

    oracle_rows = []
    for split in SPLITS:
        value = json.loads(
            (root / "prepared" / split / "oracle_summary.json").read_text(
                encoding="utf-8"
            )
        )
        for variant_key, variant_name in (
            ("fullgrid_base", "fullgrid_base"),
            ("basis_oracle", "basis_oracle"),
        ):
            oracle_rows.append(
                {
                    "split": split,
                    "variant": variant_name,
                    **value[variant_key],
                    "oracle_coefficient_rms": value["oracle_coefficient_rms"],
                }
            )

    influence_rows = [
        {"variant": variant, **summaries[variant]["token_influence"]}
        for variant in VARIANTS
    ]
    _write_csv(root / "comparison.csv", comparison_rows)
    _write_csv(root / "basis_oracle_by_split.csv", oracle_rows)
    _write_csv(root / "token_influence_comparison.csv", influence_rows)
    interpretation = {
        "schema": SCHEMA,
        "formal_checkpoint": "loss-best selected only by official val pressure loss",
        "questions": {
            "basis_oracle": "If unseen oracle remains strong, the fixed basis can represent unseen labels and RGB-to-canonical mapping is the bottleneck.",
            "direct_vs_basis": "Direct beating the parameter-matched basis head isolates harmful basis coupling; both failing implicates shared global mapping/generalization.",
            "token_influence": "High entropy, low top-token diversity, and near-identical anchor influence maps indicate global rather than selective token routing.",
        },
        "parameter_matching": {
            variant: {
                "hidden_dim": summaries[variant]["hidden_dim"],
                "parameter_count": summaries[variant]["parameter_count"],
                "relative_delta": summaries[variant]["parameter_count_relative_delta"],
            }
            for variant in VARIANTS
        },
    }
    _write_json(root / "interpretation.json", interpretation)
    _write_json(
        root / "AUDIT_DONE.json",
        {
            "schema": SCHEMA,
            "comparison_sha256": sha256_file(root / "comparison.csv"),
            "oracle_sha256": sha256_file(root / "basis_oracle_by_split.csv"),
            "influence_sha256": sha256_file(root / "token_influence_comparison.csv"),
        },
    )
    print(f"[surface-attribution] aggregate complete: {root}")


def command_self_test(_: argparse.Namespace) -> None:
    feature_dim = 24
    coefficient_dim = 7
    valid_count = 11
    basis_hidden = 128
    direct_hidden = _matched_direct_hidden_dim(
        feature_dim, coefficient_dim, valid_count, basis_hidden
    )
    target = _parameter_count(feature_dim, basis_hidden, coefficient_dim)
    candidates = {
        hidden: abs(_parameter_count(feature_dim, hidden, valid_count) - target)
        for hidden in range(64, 2 * basis_hidden + 1)
    }
    if direct_hidden != min(candidates, key=candidates.get):
        raise AssertionError("Parameter matching did not select the closest direct head")
    support_indices = torch.tensor([[0, 1], [2, 3], [4, 6]])
    support_weights = torch.tensor([[0.5, 0.5], [0.25, 0.75], [1.0, 0.0]])
    basis = AttributionHead(
        feature_dim,
        coefficient_dim,
        hidden_dim=basis_hidden,
        dropout=0.0,
        support_indices=support_indices,
        support_weights=support_weights,
    )
    direct = AttributionHead(
        feature_dim,
        valid_count,
        hidden_dim=direct_hidden,
        dropout=0.0,
        support_indices=None,
        support_weights=None,
    )
    features = torch.randn(4, feature_dim)
    if basis(features)[0].shape != (4, 3):
        raise AssertionError("Basis attribution head has the wrong output shape")
    if direct(features)[0].shape != (4, valid_count):
        raise AssertionError("Direct attribution head has the wrong output shape")
    print("surface mapping attribution self-test: OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-split")
    prepare.add_argument("--feature-cache", required=True)
    prepare.add_argument("--base-checkpoint", required=True)
    prepare.add_argument("--surface-basis", required=True)
    prepare.add_argument("--split", choices=SPLITS, required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--device", default="cuda:0")
    prepare.add_argument("--sample-limit", type=int, default=0)
    prepare.add_argument("--oracle-sample-limit", type=int, default=8192)
    prepare.add_argument("--batch-size", type=int, default=128)
    prepare.add_argument("--ridge", type=float, default=0.001)
    prepare.add_argument("--logit-epsilon", type=float, default=0.001)
    prepare.add_argument("--seed", type=int, default=521)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=command_prepare_split)

    train = subparsers.add_parser("train")
    train.add_argument("--prepared-root", required=True)
    train.add_argument("--surface-basis", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--variant", choices=VARIANTS, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--eval-batch-size", type=int, default=512)
    train.add_argument("--basis-hidden-dim", type=int, default=1024)
    train.add_argument(
        "--direct-hidden-dim",
        type=int,
        default=0,
        help="0 selects the closest parameter match to the basis head.",
    )
    train.add_argument("--dropout", type=float, default=0.3)
    train.add_argument("--lr", type=float, default=4e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--warmup-epochs", type=int, default=3)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--influence-samples", type=int, default=16)
    train.add_argument("--influence-anchors", type=int, default=256)
    train.add_argument("--influence-token-chunk", type=int, default=16)
    train.add_argument("--seed", type=int, default=521)
    train.add_argument("--force", action="store_true")
    train.set_defaults(func=command_train)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input-root", required=True)
    aggregate.set_defaults(func=command_aggregate)

    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(func=command_self_test)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size",
        "eval_batch_size",
        "basis_hidden_dim",
        "epochs",
        "influence_samples",
        "influence_anchors",
        "influence_token_chunk",
        "oracle_sample_limit",
    )
    for name in positive:
        if hasattr(args, name) and int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "sample_limit") and int(args.sample_limit) < 0:
        raise ValueError("--sample-limit must be nonnegative")
    if hasattr(args, "ridge") and float(args.ridge) <= 0.0:
        raise ValueError("--ridge must be positive")
    if hasattr(args, "direct_hidden_dim") and int(args.direct_hidden_dim) < 0:
        raise ValueError("--direct-hidden-dim must be nonnegative")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
