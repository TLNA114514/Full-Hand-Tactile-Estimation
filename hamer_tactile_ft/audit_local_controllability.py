#!/usr/bin/env python3
"""Audit local controllability of a frozen tactile decoder.

The audit operates on immutable feature caches.  It compares corrections that
must pass through the frozen FullGrid decoder with corrections applied directly
to output logits, then reports perfect-support and perfect-ordinal upper bounds.
No oracle output is suitable for model input or deployment.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import heapq
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

# Direct execution sets sys.path to hamer_tactile_ft rather than the repository
# root. Resolve it from this file so both direct and module execution work.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.hamer_tactile import _build_dense_decoder
from hamer_tactile_ft.tactile_metrics import (
    location_distribution_stats,
    volumetric_iou_stats,
)
from tactile_input_priors.feature_cache import FeatureCacheDataset, sha256_file


SUPPORTED_ORACLES = ("feature", "output", "output_exact", "support", "ordinal")
CATEGORY_NAMES = ("false_high", "false_low", "true_positive", "background")
ERROR_SCALE_NAMES = ("sparse", "medium", "broad")


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
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class CacheGroup:
    """Interleave one finalized cache or a stride-partitioned cache set."""

    def __init__(self, root: str | os.PathLike[str], fields: Sequence[str]):
        self.root = Path(root).expanduser().resolve(strict=True)
        paths = _cache_paths(self.root)
        self.caches = tuple(
            FeatureCacheDataset(
                path,
                fields=fields,
                max_open_shards=2,
                copy_arrays=True,
            )
            for path in paths
        )
        self.fields = tuple(fields)
        self.sample_count = sum(len(cache) for cache in self.caches)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cache = self.caches[index % len(self.caches)]
        return cache[index // len(self.caches)]

    @property
    def provenance(self) -> Mapping[str, Any]:
        return self.caches[0].config.get("provenance", {})

    @property
    def provenances(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(cache.config.get("provenance", {}) for cache in self.caches)

    @property
    def config_sha256s(self) -> tuple[str, ...]:
        return tuple(cache.config_sha256 for cache in self.caches)

    def close(self) -> None:
        for cache in self.caches:
            cache.close()


def _cache_paths(root: Path) -> tuple[Path, ...]:
    if (root / "CACHE_DONE.json").is_file():
        return (root,)
    paths = tuple(
        child
        for child in sorted(root.glob("part-*-of-*"))
        if (child / "CACHE_DONE.json").is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No finalized feature cache under {root}")
    parsed = [re.fullmatch(r"part-(\d+)-of-(\d+)", path.name) for path in paths]
    if any(match is None for match in parsed):
        raise RuntimeError(f"Malformed partition directory under {root}")
    expected_counts = {int(match.group(2)) for match in parsed if match is not None}
    indices = {int(match.group(1)) for match in parsed if match is not None}
    if len(expected_counts) != 1:
        raise RuntimeError(f"Partition counts disagree under {root}: {expected_counts}")
    expected = expected_counts.pop()
    if len(paths) != expected or indices != set(range(expected)):
        raise RuntimeError(
            f"Incomplete partitioned feature cache under {root}: "
            f"found indices={sorted(indices)}, expected=0..{expected - 1}"
        )
    return paths


def _available_fields(root: Path) -> set[str]:
    cache = FeatureCacheDataset(_cache_paths(root)[0], max_open_shards=1)
    try:
        return set(cache.fields)
    finally:
        cache.close()


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(checkpoint.get("model_config", {}) or {})
    for key in (
        "pool_layout",
        "pool_output_channels",
        "decoder_hidden_dim",
        "decoder_dropout_scale",
        "input_resolution",
    ):
        if key not in result and key in checkpoint:
            result[key] = checkpoint[key]
    return result


def _load_decoder(
    checkpoint_path: Path,
    *,
    grid_size: tuple[int, int],
    tactile_dim: int,
    device: torch.device,
) -> tuple[nn.Sequential, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint is not a mapping: {checkpoint_path}")
    if checkpoint.get("format") != "tactile_trainable_v2":
        raise ValueError(
            "Local controllability requires a compact format=tactile_trainable_v2 checkpoint"
        )
    config = _checkpoint_config(checkpoint)
    pool_layout = str(config.get("pool_layout", "legacy5"))
    if pool_layout != "fullgrid32":
        raise ValueError(
            f"This audit targets FullGrid32, but checkpoint pool_layout={pool_layout!r}"
        )
    decoder, _, _ = _build_dense_decoder(
        tactile_dim=int(tactile_dim),
        channels=256,
        pool_layout=pool_layout,
        grid_size=grid_size,
        pool_output_channels=int(config.get("pool_output_channels", 32)),
        decoder_hidden_dim=int(config.get("decoder_hidden_dim", 512)),
        dropout_scale=float(config.get("decoder_dropout_scale", 1.0)),
    )
    state = checkpoint.get("state_dict", {})
    decoder_state = {
        str(key)[len("tactile_head.decoder.") :]: value
        for key, value in state.items()
        if str(key).startswith("tactile_head.decoder.")
    }
    if not decoder_state:
        raise ValueError("Checkpoint has no tactile_head.decoder state")
    expected = decoder.state_dict()
    for old_prefix, new_prefix in (("0.project.", "0.projection."),):
        for old_key in tuple(decoder_state):
            if not old_key.startswith(old_prefix):
                continue
            new_key = new_prefix + old_key[len(old_prefix) :]
            if new_key in expected and new_key not in decoder_state:
                decoder_state[new_key] = decoder_state.pop(old_key)
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval().to(device=device, dtype=torch.float32)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    return decoder, config


def _palm_mask(item: Mapping[str, Any], provenance: Mapping[str, Any]) -> np.ndarray:
    raw = item.get("palm_mask")
    if raw is None:
        raw = provenance.get("palm_mask")
    if raw is None:
        return np.ones_like(np.asarray(item["tactile_signal"]), dtype=bool)
    mask = np.asarray(raw, dtype=bool).reshape(-1)
    if mask.shape != np.asarray(item["tactile_signal"]).shape:
        raise ValueError(
            f"Palm mask shape {mask.shape} differs from tactile signal "
            f"{np.asarray(item['tactile_signal']).shape}"
        )
    return mask


def _base_logits_batch(
    items: Sequence[Mapping[str, Any]],
    decoder: nn.Module,
    device: torch.device,
) -> np.ndarray:
    if not all("z_rgb" in item for item in items):
        raise ValueError("Cache needs z_rgb to reconstruct comparable base predictions")
    grid = torch.from_numpy(
        np.stack([np.asarray(item["z_rgb"], dtype=np.float32) for item in items])
    ).to(device)
    with torch.inference_mode():
        logits = decoder(grid)
    return logits.float().cpu().numpy()


def _candidate_scores(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    no_contact_threshold: float,
    contact_threshold: float,
) -> dict[str, float]:
    false_high = valid & (gt <= no_contact_threshold) & (pred >= contact_threshold)
    false_low = valid & (gt >= contact_threshold) & (pred <= no_contact_threshold)
    gt_contact = valid & (gt >= contact_threshold)
    pred_contact = valid & (pred >= contact_threshold)
    intersection = float(np.logical_and(gt_contact, pred_contact).sum())
    union = float(np.logical_or(gt_contact, pred_contact).sum())
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    return {
        "false_high": float(np.maximum(pred[false_high] - gt[false_high], 0.0).sum()),
        "false_low": float(np.maximum(gt[false_low] - pred[false_low], 0.0).sum()),
        "true_positive": intersection / union if intersection > 0.0 and union > 0.0 else 0.0,
        "background": (
            1.0 / (1.0 + float(gt_valid.sum()) + float(pred_valid.sum()))
            if gt_valid.size and float(gt_valid.max()) <= no_contact_threshold
            else 0.0
        ),
    }


def _error_scale(vertex_count: int) -> str:
    vertex_count = int(vertex_count)
    if vertex_count <= 32:
        return "sparse"
    if vertex_count <= 256:
        return "medium"
    return "broad"


def _push_candidate(
    heap: list[tuple[float, int]],
    score: float,
    index: int,
    capacity: int,
) -> None:
    if not math.isfinite(score) or score <= 0.0:
        return
    entry = (float(score), int(index))
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _scan_indices(sample_count: int, scan_limit: int) -> np.ndarray:
    count = min(int(sample_count), max(1, int(scan_limit)))
    if count == sample_count:
        return np.arange(sample_count, dtype=np.int64)
    return np.unique(np.linspace(0, sample_count - 1, num=count, dtype=np.int64))


def select_samples(
    cache: CacheGroup,
    decoder: nn.Module,
    device: torch.device,
    *,
    scan_limit: int,
    samples_per_category: int,
    samples_per_error_stratum: int,
    max_samples_per_sequence: int,
    scan_batch_size: int,
    no_contact_threshold: float,
    contact_threshold: float,
) -> list[dict[str, Any]]:
    heaps: dict[tuple[str, str], dict[str, list[tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    indices = _scan_indices(len(cache), scan_limit)
    for start in range(0, len(indices), int(scan_batch_size)):
        batch_indices = indices[start : start + int(scan_batch_size)]
        items = [cache[int(index)] for index in batch_indices]
        logits = _base_logits_batch(items, decoder, device)
        predictions = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        for index, item, pred in zip(batch_indices, items, predictions):
            if "has_tactile" in item and not bool(np.asarray(item["has_tactile"]).reshape(-1)[0]):
                continue
            gt = np.asarray(item["tactile_signal"], dtype=np.float32)
            valid = _palm_mask(item, cache.provenance)
            valid &= np.isfinite(gt) & np.isfinite(pred)
            gt = np.nan_to_num(gt, nan=0.0, posinf=1.0, neginf=0.0)
            false_high_mask = (
                valid
                & (gt <= no_contact_threshold)
                & (pred >= contact_threshold)
            )
            false_low_mask = (
                valid
                & (gt >= contact_threshold)
                & (pred <= no_contact_threshold)
            )
            scores = _candidate_scores(
                pred,
                gt,
                valid,
                no_contact_threshold=no_contact_threshold,
                contact_threshold=contact_threshold,
            )
            sequence_key = str(item.get("sequence_key", "") or item.get("sample_id", index))
            strata = {
                "false_high": _error_scale(int(false_high_mask.sum())),
                "false_low": _error_scale(int(false_low_mask.sum())),
                "true_positive": "all",
                "background": "all",
            }
            for name, score in scores.items():
                key = (name, strata[name])
                _push_candidate(
                    heaps[key][sequence_key],
                    score,
                    int(index),
                    max(1, int(max_samples_per_sequence)),
                )
        completed = min(start + scan_batch_size, len(indices))
        if completed == len(indices) or completed % max(5000, scan_batch_size) < scan_batch_size:
            print(f"[selection] scanned {completed}/{len(indices)}", flush=True)

    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    selection_groups = [
        *((category, scale) for category in ("false_high", "false_low") for scale in ERROR_SCALE_NAMES),
        ("true_positive", "all"),
        ("background", "all"),
    ]
    for category, error_scale in selection_groups:
        candidates = sorted(
            (
                entry
                for sequence_heap in heaps[(category, error_scale)].values()
                for entry in sequence_heap
            ),
            reverse=True,
        )
        quota = (
            int(samples_per_error_stratum)
            if error_scale != "all"
            else int(samples_per_category)
        )
        for score, index in candidates:
            if index in used:
                continue
            used.add(index)
            item = cache[index]
            selected.append(
                {
                    "index": index,
                    "category": category,
                    "error_scale": error_scale,
                    "selection_score": score,
                    "sample_id": str(item.get("sample_id", index)),
                    "dataset": str(item.get("dataset", "")),
                    "sequence_key": str(item.get("sequence_key", "")),
                    "query_alias": str(item.get("query_alias", "")),
                    "frame_idx": int(item.get("frame_idx", -1)),
                }
            )
            if sum(
                row["category"] == category and row["error_scale"] == error_scale
                for row in selected
            ) >= quota:
                break
    if not selected:
        raise RuntimeError("No eligible audit samples were selected")
    for category, error_scale in selection_groups:
        group = [
            row
            for row in selected
            if row["category"] == category and row["error_scale"] == error_scale
        ]
        print(
            f"[selection] {category}:{error_scale} samples={len(group)} "
            f"sequences={len({row['sequence_key'] for row in group})}",
            flush=True,
        )
    return selected


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    # Multiplication is not a valid NaN mask: NaN * 0 is still NaN.
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    return safe_values.sum() / mask.sum().to(values.dtype).clamp_min(1.0)


def _target_mask(
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    no_contact_threshold: float,
    contact_threshold: float,
) -> torch.Tensor:
    false_high = (target <= no_contact_threshold) & (base_prediction >= contact_threshold)
    false_low = (target >= contact_threshold) & (base_prediction <= no_contact_threshold)
    return valid & (false_high | false_low)


def _oracle_objective(
    prediction: torch.Tensor,
    base_prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    valid: torch.Tensor,
    *,
    off_target_weight: float,
) -> torch.Tensor:
    target_loss = _masked_mean(
        (prediction - target).square(), target_mask.to(prediction.dtype)
    )
    off_mask = valid & ~target_mask
    off_loss = _masked_mean(
        (prediction - base_prediction).square(), off_mask.to(prediction.dtype)
    )
    return target_loss + float(off_target_weight) * off_loss


def optimize_feature_oracle(
    grid: torch.Tensor,
    base_logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    decoder: nn.Module,
    *,
    steps: int,
    learning_rate: float,
    rms_budget: float,
    off_target_weight: float,
    gradient_clip: float,
    no_contact_threshold: float,
    contact_threshold: float,
) -> torch.Tensor:
    finite = torch.isfinite(target) & torch.isfinite(base_logits)
    valid = valid & finite
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    base_prediction = torch.sigmoid(base_logits).detach()
    mask = _target_mask(
        base_prediction,
        target,
        valid,
        no_contact_threshold=no_contact_threshold,
        contact_threshold=contact_threshold,
    )
    raw_delta = nn.Parameter(torch.zeros_like(grid))
    optimizer = torch.optim.Adam((raw_delta,), lr=float(learning_rate), eps=1e-6)
    base_rms = grid.detach().float().square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    limit = float(rms_budget) * base_rms
    if not mask.any():
        return base_prediction
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        candidate = torch.tanh(raw_delta)
        candidate_rms = (
            candidate.float().square().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
        ).sqrt()
        # This is a hard feasibility projection, not a learnable normalization.
        # Detaching also avoids the undefined zero-RMS derivative at step zero.
        scale = torch.clamp(limit / candidate_rms, max=1.0).detach()
        delta = candidate * scale
        prediction = torch.sigmoid(decoder(grid + delta))
        loss = _oracle_objective(
            prediction,
            base_prediction,
            target,
            mask,
            valid,
            off_target_weight=off_target_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Feature oracle produced a non-finite loss at step {step}"
            )
        loss.backward()
        if raw_delta.grad is None or not torch.isfinite(raw_delta.grad).all():
            raise FloatingPointError(
                f"Feature oracle produced non-finite gradients at step {step}"
            )
        torch.nn.utils.clip_grad_norm_((raw_delta,), float(gradient_clip))
        optimizer.step()
        if not torch.isfinite(raw_delta).all():
            raise FloatingPointError("Feature oracle parameter became non-finite")
    with torch.no_grad():
        candidate = torch.tanh(raw_delta)
        candidate_rms = (
            candidate.float().square().mean(dim=(1, 2, 3), keepdim=True) + 1e-12
        ).sqrt()
        scale = torch.clamp(limit / candidate_rms, max=1.0)
        return torch.sigmoid(decoder(grid + candidate * scale)).detach()


def optimize_output_oracle(
    base_logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    maximum_delta: float,
    off_target_weight: float,
    gradient_clip: float,
    no_contact_threshold: float,
    contact_threshold: float,
) -> torch.Tensor:
    finite = torch.isfinite(target) & torch.isfinite(base_logits)
    valid = valid & finite
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    base_prediction = torch.sigmoid(base_logits).detach()
    mask = _target_mask(
        base_prediction,
        target,
        valid,
        no_contact_threshold=no_contact_threshold,
        contact_threshold=contact_threshold,
    )
    raw_delta = nn.Parameter(torch.zeros_like(base_logits))
    optimizer = torch.optim.Adam((raw_delta,), lr=float(learning_rate), eps=1e-6)
    if not mask.any():
        return base_prediction
    for step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        bounded_delta = float(maximum_delta) * torch.tanh(
            raw_delta / float(maximum_delta)
        )
        prediction = torch.sigmoid(base_logits + bounded_delta)
        loss = _oracle_objective(
            prediction,
            base_prediction,
            target,
            mask,
            valid,
            off_target_weight=off_target_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Output oracle produced a non-finite loss at step {step}"
            )
        loss.backward()
        if raw_delta.grad is None or not torch.isfinite(raw_delta.grad).all():
            raise FloatingPointError(
                f"Output oracle produced non-finite gradients at step {step}"
            )
        torch.nn.utils.clip_grad_norm_((raw_delta,), float(gradient_clip))
        optimizer.step()
        if not torch.isfinite(raw_delta).all():
            raise FloatingPointError("Output oracle parameter became non-finite")
    with torch.no_grad():
        bounded_delta = float(maximum_delta) * torch.tanh(
            raw_delta / float(maximum_delta)
        )
        return torch.sigmoid(base_logits + bounded_delta).detach()


def exact_output_oracle(
    base_prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    no_contact_threshold: float,
    contact_threshold: float,
) -> np.ndarray:
    """Exact local upper bound with no change outside the diagnosed errors."""

    local = (
        valid
        & (
            ((target <= no_contact_threshold) & (base_prediction >= contact_threshold))
            | ((target >= contact_threshold) & (base_prediction <= no_contact_threshold))
        )
    )
    result = base_prediction.copy()
    result[local] = np.clip(target[local], 0.0, 1.0)
    return result


def support_oracle(
    base_prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    negative_threshold: float,
    positive_gt_threshold: float,
    positive_output_floor: float,
) -> np.ndarray:
    result = base_prediction.copy()
    result[valid & (target <= negative_threshold)] = np.minimum(
        result[valid & (target <= negative_threshold)], negative_threshold
    )
    result[valid & (target >= positive_gt_threshold)] = np.maximum(
        result[valid & (target >= positive_gt_threshold)], positive_output_floor
    )
    return result


def ordinal_oracle(
    base_prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    thresholds: Sequence[float],
) -> np.ndarray:
    thresholds = np.asarray(tuple(float(value) for value in thresholds), dtype=np.float32)
    boundaries = np.concatenate(([0.0], thresholds, [1.0])).astype(np.float32)
    bins = np.digitize(target, thresholds, right=False)
    lower = boundaries[bins]
    upper = boundaries[bins + 1]
    result = base_prediction.copy()
    result[valid] = np.clip(result[valid], lower[valid], upper[valid])
    return result


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def load_mesh_adjacency(
    path: str | os.PathLike[str] | None,
    *,
    face_key: str,
    vertex_count: int,
) -> tuple[np.ndarray, ...] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve(strict=True)
    loaded = np.load(resolved, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if face_key not in loaded.files:
                raise KeyError(
                    f"Mesh archive {resolved} has no {face_key!r}; keys={loaded.files}"
                )
            faces = np.asarray(loaded[face_key], dtype=np.int64)
        finally:
            loaded.close()
    else:
        faces = np.asarray(loaded, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError(f"Mesh faces must be [F,3+], got {faces.shape}")
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= vertex_count):
        raise ValueError(
            f"Mesh face index range [{faces.min()},{faces.max()}] does not fit "
            f"{vertex_count} tactile vertices"
        )
    neighbors = [set() for _ in range(vertex_count)]
    for face in faces:
        vertices = [int(value) for value in face]
        for offset, source in enumerate(vertices):
            for target in vertices[offset + 1 :]:
                neighbors[source].add(target)
                neighbors[target].add(source)
    return tuple(np.asarray(sorted(values), dtype=np.int64) for values in neighbors)


def mesh_hop_leakage(
    delta: np.ndarray,
    target_mask: np.ndarray,
    valid: np.ndarray,
    adjacency: tuple[np.ndarray, ...] | None,
    hops: int,
) -> float:
    if adjacency is None or not target_mask.any():
        return float("nan")
    neighborhood = np.asarray(target_mask, dtype=bool).copy()
    frontier = set(int(value) for value in np.flatnonzero(target_mask))
    for _ in range(max(0, int(hops))):
        next_frontier: set[int] = set()
        for vertex in frontier:
            next_frontier.update(int(value) for value in adjacency[vertex])
        next_frontier.difference_update(int(value) for value in np.flatnonzero(neighborhood))
        if not next_frontier:
            break
        neighborhood[np.fromiter(next_frontier, dtype=np.int64)] = True
        frontier = next_frontier
    absolute = np.abs(delta)
    total = float(absolute[valid].sum())
    return _safe_ratio(float(absolute[valid & ~neighborhood].sum()), total)


def per_sample_metrics(
    prediction: np.ndarray,
    base_prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    target_mask: np.ndarray,
    *,
    change_threshold: float,
    mesh_adjacency: tuple[np.ndarray, ...] | None,
    geodesic_hops: int,
) -> dict[str, float]:
    pred = prediction[valid].astype(np.float64)
    base = base_prediction[valid].astype(np.float64)
    gt = target[valid].astype(np.float64)
    local_target = target_mask[valid]
    delta = pred - base
    abs_delta = np.abs(delta)
    before = float(np.abs(base[local_target] - gt[local_target]).mean()) if local_target.any() else 0.0
    after = float(np.abs(pred[local_target] - gt[local_target]).mean()) if local_target.any() else 0.0
    target_reduction = _safe_ratio(before - after, before)
    total_delta = float(abs_delta.sum())
    off_target = float(abs_delta[~local_target].sum())
    up = delta > change_threshold
    down = delta < -change_threshold
    metrics = {
        "rmse": float(np.sqrt(np.mean((pred - gt) ** 2))),
        "mae": float(np.mean(np.abs(pred - gt))),
        "target_vertex_count": int(local_target.sum()),
        "target_error_before": before,
        "target_error_after": after,
        "target_error_reduction": target_reduction,
        "off_target_delta_ratio": _safe_ratio(off_target, total_delta),
        "changed_vertex_fraction": float((abs_delta > change_threshold).mean()),
        "up_correction_precision": (
            float((gt[up] > base[up]).mean()) if up.any() else float("nan")
        ),
        "down_correction_precision": (
            float((gt[down] < base[down]).mean()) if down.any() else float("nan")
        ),
        "delta_abs_sum": total_delta,
        "delta_net_sum": float(delta.sum()),
        "mesh_hop_leakage": mesh_hop_leakage(
            prediction - base_prediction,
            target_mask,
            valid,
            mesh_adjacency,
            geodesic_hops,
        ),
    }
    for name, threshold in (("support_iou_005", 0.05), ("contact_iou", 0.10)):
        pred_contact = pred >= threshold
        gt_contact = gt >= threshold
        intersection = float(np.logical_and(pred_contact, gt_contact).sum())
        union = float(np.logical_or(pred_contact, gt_contact).sum())
        metrics[name] = _safe_ratio(intersection, union, default=1.0)
    viou = volumetric_iou_stats(pred[None], gt[None])
    metrics["volumetric_iou"] = float(viou.frame_macro)
    location = location_distribution_stats(pred[None], gt[None])
    core = location_distribution_stats(
        pred[None],
        gt[None],
        distribution_power=2.0,
        min_gt_peak=0.05,
    )
    metrics["distribution_viou"] = float(location.distribution_viou[0])
    metrics["core_distribution_viou"] = float(core.distribution_viou[0])
    return metrics


def _finite_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    metric_names = (
        "rmse",
        "mae",
        "support_iou_005",
        "contact_iou",
        "volumetric_iou",
        "distribution_viou",
        "core_distribution_viou",
        "target_error_before",
        "target_error_after",
        "target_error_reduction",
        "off_target_delta_ratio",
        "changed_vertex_fraction",
        "up_correction_precision",
        "down_correction_precision",
        "delta_abs_sum",
        "delta_net_sum",
        "mesh_hop_leakage",
    )
    result = []
    oracle_names = list(dict.fromkeys(str(row["oracle"]) for row in rows))
    for oracle in oracle_names:
        oracle_rows = [row for row in rows if row["oracle"] == oracle]
        if not oracle_rows:
            continue
        groups: list[tuple[str, list[Mapping[str, Any]]]] = [("overall", oracle_rows)]
        for category in CATEGORY_NAMES:
            category_rows = [row for row in oracle_rows if row["category"] == category]
            groups.append((category, category_rows))
            if category in {"false_high", "false_low"}:
                groups.extend(
                    (
                        f"{category}:{scale}",
                        [row for row in category_rows if row.get("error_scale") == scale],
                    )
                    for scale in ERROR_SCALE_NAMES
                )
        for category, selected in groups:
            if not selected:
                continue
            summary: dict[str, Any] = {
                "oracle": oracle,
                "category": category,
                "sample_count": len(selected),
            }
            for name in metric_names:
                summary[name] = _finite_mean(row[name] for row in selected)
            result.append(summary)
    return result


def pca_first_component_ratio(deltas: np.ndarray, valid: np.ndarray) -> float:
    if deltas.ndim != 2 or len(deltas) < 2:
        return 0.0
    common_valid = np.asarray(valid, dtype=bool).all(axis=0)
    values = deltas[:, common_valid].astype(np.float64)
    if values.shape[1] == 0:
        return 0.0
    values -= values.mean(axis=0, keepdims=True)
    total = float(np.square(values).sum())
    if total <= 1e-20:
        return 0.0
    singular_values = np.linalg.svd(values, compute_uv=False, full_matrices=False)
    return float(singular_values[0] ** 2 / np.square(singular_values).sum())


def _parse_float_list(raw: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in str(raw).split(",") if item.strip())
    if not result or any(not math.isfinite(value) for value in result):
        raise argparse.ArgumentTypeError("Expected a comma-separated finite float list")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise argparse.ArgumentTypeError("Thresholds must be strictly increasing")
    return result


def _parse_oracles(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in str(raw).split(",") if item.strip())
    unknown = sorted(set(values) - set(SUPPORTED_ORACLES))
    if unknown or not values:
        raise argparse.ArgumentTypeError(
            f"Oracles must be chosen from {SUPPORTED_ORACLES}, got {unknown or values}"
        )
    return values


def _value_label(prefix: str, value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"{prefix}_{text}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--oracles", type=_parse_oracles, default=SUPPORTED_ORACLES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scan-limit", type=int, default=50000)
    parser.add_argument("--samples-per-category", type=int, default=32)
    parser.add_argument("--samples-per-error-stratum", type=int, default=16)
    parser.add_argument("--max-samples-per-sequence", type=int, default=4)
    parser.add_argument("--scan-batch-size", type=int, default=128)
    parser.add_argument("--oracle-batch-size", type=int, default=8)
    parser.add_argument("--oracle-steps", type=int, default=250)
    parser.add_argument("--feature-learning-rate", type=float, default=0.01)
    parser.add_argument("--output-learning-rate", type=float, default=0.10)
    parser.add_argument("--feature-rms-budget", type=float, default=0.10)
    parser.add_argument("--feature-rms-budgets", type=_parse_float_list)
    parser.add_argument("--output-logit-delta-max", type=float, default=1.0)
    parser.add_argument("--output-logit-delta-max-values", type=_parse_float_list)
    parser.add_argument("--off-target-weight", type=float, default=1.0)
    parser.add_argument("--oracle-gradient-clip", type=float, default=10.0)
    parser.add_argument("--no-contact-threshold", type=float, default=0.02)
    parser.add_argument("--support-positive-threshold", type=float, default=0.10)
    parser.add_argument("--contact-threshold", type=float, default=0.10)
    parser.add_argument(
        "--ordinal-thresholds",
        type=_parse_float_list,
        default=(0.02, 0.05, 0.10, 0.20, 0.50),
    )
    parser.add_argument("--change-threshold", type=float, default=0.005)
    parser.add_argument("--reselect-samples", action="store_true")
    parser.add_argument("--mesh-faces")
    parser.add_argument("--mesh-face-key", default="faces")
    parser.add_argument("--geodesic-hops", type=int, default=2)
    parser.add_argument("--save-outputs", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "scan_limit",
        "samples_per_category",
        "samples_per_error_stratum",
        "max_samples_per_sequence",
        "scan_batch_size",
        "oracle_batch_size",
        "oracle_steps",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "feature_rms_budget",
        "output_logit_delta_max",
        "change_threshold",
        "oracle_gradient_clip",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if float(args.off_target_weight) < 0.0:
        raise ValueError("--off-target-weight must be nonnegative")
    for name, values in (
        ("feature-rms-budgets", args.feature_rms_budgets),
        ("output-logit-delta-max-values", args.output_logit_delta_max_values),
    ):
        if values is not None and any(float(value) <= 0.0 for value in values):
            raise ValueError(f"--{name} values must be positive")
    if int(args.geodesic_hops) < 0:
        raise ValueError("--geodesic-hops must be nonnegative")
    if not (
        0.0 <= args.no_contact_threshold
        < args.support_positive_threshold
        <= args.contact_threshold
        <= 1.0
    ):
        raise ValueError(
            "Expected 0 <= no-contact < support-positive <= contact <= 1"
        )


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    torch.manual_seed(521)
    np.random.seed(521)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.feature_cache).expanduser().resolve(strict=True)
    checkpoint_path = Path(args.base_checkpoint).expanduser().resolve(strict=True)
    available = _available_fields(cache_root)
    required = {"tactile_signal"}
    required.add("z_rgb")
    optional = {"palm_mask", "has_tactile"} & available
    fields = tuple(sorted(required | optional))
    cache = CacheGroup(cache_root, fields)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    for partition_index, provenance in enumerate(cache.provenances):
        expected_checkpoint = str(provenance.get("base_checkpoint_sha256", "") or "")
        if expected_checkpoint and expected_checkpoint != checkpoint_sha256:
            raise RuntimeError(
                "Feature cache was produced by a different tactile checkpoint: "
                f"partition={partition_index}, cache={expected_checkpoint}, "
                f"requested={checkpoint_sha256}"
            )
    first = cache[0]
    tactile_dim = int(np.asarray(first["tactile_signal"]).size)
    if "z_rgb" in first:
        grid_size = tuple(int(value) for value in np.asarray(first["z_rgb"]).shape[-2:])
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        resolution = _checkpoint_config(checkpoint).get("input_resolution", (256, 192))
        if isinstance(resolution, str):
            height, width = (int(item) for item in resolution.lower().split("x"))
        else:
            height, width = (int(item) for item in resolution)
        grid_size = (height // 16, width // 16)
        del checkpoint
    decoder, model_config = _load_decoder(
        checkpoint_path,
        grid_size=grid_size,
        tactile_dim=tactile_dim,
        device=device,
    )
    mesh_adjacency = load_mesh_adjacency(
        args.mesh_faces,
        face_key=args.mesh_face_key,
        vertex_count=tactile_dim,
    )
    print(
        f"Loaded frozen decoder and cache: samples={len(cache)}, fields={fields}, "
        f"grid={grid_size}, device={device}",
        flush=True,
    )

    selection_signature = {
        "cache_config_sha256s": list(cache.config_sha256s),
        "checkpoint_sha256": checkpoint_sha256,
        "scan_limit": args.scan_limit,
        "samples_per_category": args.samples_per_category,
        "samples_per_error_stratum": args.samples_per_error_stratum,
        "max_samples_per_sequence": args.max_samples_per_sequence,
        "no_contact_threshold": args.no_contact_threshold,
        "contact_threshold": args.contact_threshold,
        "error_scale_vertex_bins": [1, 33, 257],
    }
    selection_path = output_dir / "selected_samples.json"
    selection_config_path = output_dir / "selection_config.json"
    saved_selection_config = None
    if selection_config_path.is_file():
        saved_selection_config = json.loads(
            selection_config_path.read_text(encoding="utf-8")
        )
    reuse_selection = (
        selection_path.is_file()
        and not args.reselect_samples
        and saved_selection_config == selection_signature
    )
    if reuse_selection:
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"Invalid saved sample selection: {selection_path}")
        for record in selected:
            if not isinstance(record, Mapping) or not 0 <= int(record["index"]) < len(cache):
                raise ValueError(f"Invalid saved sample selection row: {record!r}")
        print(
            f"[selection] Reusing {len(selected)} saved samples from {selection_path}",
            flush=True,
        )
    else:
        selected = select_samples(
            cache,
            decoder,
            device,
            scan_limit=args.scan_limit,
            samples_per_category=args.samples_per_category,
            samples_per_error_stratum=args.samples_per_error_stratum,
            max_samples_per_sequence=args.max_samples_per_sequence,
            scan_batch_size=args.scan_batch_size,
            no_contact_threshold=args.no_contact_threshold,
            contact_threshold=args.contact_threshold,
        )
        _write_json(selection_path, selected)
        _write_jsonl(output_dir / "selected_samples.jsonl", selected)
        _write_json(selection_config_path, selection_signature)

    all_rows: list[dict[str, Any]] = []
    outputs: dict[str, list[np.ndarray]] = defaultdict(list)
    feature_budgets = (
        tuple(args.feature_rms_budgets)
        if args.feature_rms_budgets is not None
        else (float(args.feature_rms_budget),)
    )
    output_limits = (
        tuple(args.output_logit_delta_max_values)
        if args.output_logit_delta_max_values is not None
        else (float(args.output_logit_delta_max),)
    )
    valid_parts: list[np.ndarray] = []
    for start in range(0, len(selected), args.oracle_batch_size):
        records = selected[start : start + args.oracle_batch_size]
        items = [cache[int(record["index"])] for record in records]
        target_np = np.stack(
            [np.asarray(item["tactile_signal"], dtype=np.float32) for item in items]
        )
        valid_np = np.stack([_palm_mask(item, cache.provenance) for item in items])
        logits_np = _base_logits_batch(items, decoder, device)
        valid_np &= np.isfinite(target_np) & np.isfinite(logits_np)
        target_np = np.nan_to_num(target_np, nan=0.0, posinf=1.0, neginf=0.0)
        if not np.isfinite(logits_np).all():
            bad = np.flatnonzero(~np.isfinite(logits_np).all(axis=1))
            bad_ids = [records[int(index)]["sample_id"] for index in bad[:8]]
            raise FloatingPointError(
                f"Frozen decoder produced non-finite base logits for samples {bad_ids}"
            )
        base_np = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -40.0, 40.0)))
        batch_outputs: dict[str, np.ndarray] = {"base": base_np}
        target = torch.from_numpy(target_np).to(device)
        valid = torch.from_numpy(valid_np).to(device=device, dtype=torch.bool)
        base_logits = torch.from_numpy(logits_np).to(device)

        if "feature" in args.oracles:
            grid = torch.from_numpy(
                np.stack([np.asarray(item["z_rgb"], dtype=np.float32) for item in items])
            ).to(device)
            for rms_budget in feature_budgets:
                oracle_name = (
                    "feature"
                    if len(feature_budgets) == 1
                    else _value_label("feature_rms", rms_budget)
                )
                batch_outputs[oracle_name] = optimize_feature_oracle(
                    grid,
                    base_logits,
                    target,
                    valid,
                    decoder,
                    steps=args.oracle_steps,
                    learning_rate=args.feature_learning_rate,
                    rms_budget=rms_budget,
                    off_target_weight=args.off_target_weight,
                    gradient_clip=args.oracle_gradient_clip,
                    no_contact_threshold=args.no_contact_threshold,
                    contact_threshold=args.contact_threshold,
                ).cpu().numpy()
        if "output" in args.oracles:
            for maximum_delta in output_limits:
                oracle_name = (
                    "output"
                    if len(output_limits) == 1
                    else _value_label("output_logit", maximum_delta)
                )
                batch_outputs[oracle_name] = optimize_output_oracle(
                    base_logits,
                    target,
                    valid,
                    steps=args.oracle_steps,
                    learning_rate=args.output_learning_rate,
                    maximum_delta=maximum_delta,
                    off_target_weight=args.off_target_weight,
                    gradient_clip=args.oracle_gradient_clip,
                    no_contact_threshold=args.no_contact_threshold,
                    contact_threshold=args.contact_threshold,
                ).cpu().numpy()
        if "output_exact" in args.oracles:
            batch_outputs["output_exact"] = exact_output_oracle(
                base_np,
                target_np,
                valid_np,
                no_contact_threshold=args.no_contact_threshold,
                contact_threshold=args.contact_threshold,
            )
        if "support" in args.oracles:
            batch_outputs["support"] = support_oracle(
                base_np,
                target_np,
                valid_np,
                negative_threshold=args.no_contact_threshold,
                positive_gt_threshold=args.support_positive_threshold,
                positive_output_floor=args.contact_threshold,
            )
        if "ordinal" in args.oracles:
            batch_outputs["ordinal"] = ordinal_oracle(
                base_np,
                target_np,
                valid_np,
                args.ordinal_thresholds,
            )

        local_target = (
            valid_np
            & (
                ((target_np <= args.no_contact_threshold) & (base_np >= args.contact_threshold))
                | ((target_np >= args.contact_threshold) & (base_np <= args.no_contact_threshold))
            )
        )
        valid_parts.append(valid_np)
        for oracle, predictions in batch_outputs.items():
            outputs[oracle].append(predictions.astype(np.float32, copy=False))
            for offset, (record, prediction) in enumerate(zip(records, predictions)):
                metrics = per_sample_metrics(
                    prediction,
                    base_np[offset],
                    target_np[offset],
                    valid_np[offset],
                    local_target[offset],
                    change_threshold=args.change_threshold,
                    mesh_adjacency=mesh_adjacency,
                    geodesic_hops=args.geodesic_hops,
                )
                all_rows.append({**record, "oracle": oracle, **metrics})
        print(
            f"[oracle] completed {min(start + args.oracle_batch_size, len(selected))}/"
            f"{len(selected)} samples",
            flush=True,
        )

    valid_all = np.concatenate(valid_parts, axis=0)
    pca: dict[str, dict[str, float]] = {}
    base_all = np.concatenate(outputs["base"], axis=0)
    for oracle, chunks in outputs.items():
        if oracle == "base" or not chunks:
            continue
        prediction = np.concatenate(chunks, axis=0)
        delta = prediction - base_all
        pca[oracle] = {
            "overall": pca_first_component_ratio(delta, valid_all)
        }
        for category in CATEGORY_NAMES:
            category_indices = np.asarray(
                [index for index, row in enumerate(selected) if row["category"] == category],
                dtype=np.int64,
            )
            if len(category_indices):
                pca[oracle][category] = pca_first_component_ratio(
                    delta[category_indices], valid_all[category_indices]
                )
            if category in {"false_high", "false_low"}:
                for scale in ERROR_SCALE_NAMES:
                    label = f"{category}:{scale}"
                    indices = np.asarray(
                        [
                            index
                            for index, row in enumerate(selected)
                            if row["category"] == category
                            and row.get("error_scale") == scale
                        ],
                        dtype=np.int64,
                    )
                    if len(indices):
                        pca[oracle][label] = pca_first_component_ratio(
                            delta[indices], valid_all[indices]
                        )

    summary_rows = summarize_rows(all_rows)
    for row in summary_rows:
        row["delta_pca_first_component_ratio"] = pca.get(
            row["oracle"], {}
        ).get(row["category"], 0.0)
    config = {
        **vars(args),
        "feature_cache": str(cache_root),
        "feature_cache_config_sha256s": list(cache.config_sha256s),
        "base_checkpoint": str(checkpoint_path),
        "base_checkpoint_sha256": checkpoint_sha256,
        "model_config": model_config,
        "cache_fields": list(fields),
        "grid_size": list(grid_size),
        "selected_sample_count": len(selected),
    }
    _write_json(output_dir / "run_config.json", config)
    _write_csv(output_dir / "sample_metrics.csv", all_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_json(
        output_dir / "summary.json",
        {
            "summary": summary_rows,
            "delta_pca_first_component_ratio": pca,
            "selected_by_category": {
                category: sum(row["category"] == category for row in selected)
                for category in CATEGORY_NAMES
            },
            "selected_by_stratum": {
                f"{category}:{scale}": sum(
                    row["category"] == category and row.get("error_scale") == scale
                    for row in selected
                )
                for category in ("false_high", "false_low")
                for scale in ERROR_SCALE_NAMES
            },
            "selected_sequence_count_by_group": {
                f"{category}:{scale}": len(
                    {
                        row["sequence_key"]
                        for row in selected
                        if row["category"] == category
                        and row.get("error_scale") == scale
                    }
                )
                for category, scales in (
                    ("false_high", ERROR_SCALE_NAMES),
                    ("false_low", ERROR_SCALE_NAMES),
                    ("true_positive", ("all",)),
                    ("background", ("all",)),
                )
                for scale in scales
            },
        },
    )
    if args.save_outputs:
        np.savez_compressed(
            output_dir / "oracle_outputs.npz",
            target=np.concatenate(
                [
                    np.stack(
                        [
                            np.asarray(cache[int(record["index"])]["tactile_signal"], dtype=np.float32)
                            for record in selected[start : start + args.oracle_batch_size]
                        ]
                    )
                    for start in range(0, len(selected), args.oracle_batch_size)
                ],
                axis=0,
            ),
            valid=valid_all,
            **{
                oracle: np.concatenate(chunks, axis=0)
                for oracle, chunks in outputs.items()
                if chunks
            },
        )
    cache.close()
    decoder.cpu()
    del decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Local controllability report written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
