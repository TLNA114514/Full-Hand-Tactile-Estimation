#!/usr/bin/env python3
"""Evaluate temporal image-grid fusion with strict causal history controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal
from tactile_input_priors.prior_metrics import (
    METRIC_FIELDS,
    metric_contributions,
    summarize_metric_values,
)
from tactile_input_priors.runtime import (
    build_dataset,
    build_frozen_base,
    file_sha256,
    load_torch_checkpoint,
    parse_csv,
    parse_resolution,
)
from tactile_input_priors.online_temporal import (
    OnlineTemporalDataset,
    OnlineTemporalRecordIndex,
    build_online_temporal_pair_index,
)
from tactile_input_priors.temporal_flow import (
    PartitionedPalmCache,
    TemporalReplayDataset,
    build_prediction_control_bins,
)
from tactile_input_priors.temporal_grid import (
    TEMPORAL_GRID_FORMAT,
    TEMPORAL_GRID_SOURCES,
    TEMPORAL_MEMORY_FORMAT,
    TEMPORAL_TRUNK_FORMAT,
    FrozenOnlineGridEncoder,
    HierarchicalTemporalMemoryFusionV3,
    TemporalMainTrunkV3,
    TemporalGridAdapterV1,
    TemporalGridTactileModel,
    TemporalLocalMemoryFusionV2,
    build_fresh_temporal_base,
    controlled_temporal_grid_inputs,
    load_frozen_fullgrid_decoder,
    module_state_sha256,
)
from tactile_input_priors.train_temporal_flow import _pair_index


SUBSETS = (
    "full",
    "available",
    "matched",
    "matched_rgb_exact",
    "onset",
    "stable",
    "release",
)
DIAGNOSTICS = (
    "feature_delta_rms",
    "feature_clamp_scale",
    "history_available_fraction",
    "history_motion_rms",
    "fast_feature_delta_rms",
    "medium_feature_delta_rms",
    "logit_delta_rms",
    "output_delta_up_volume",
    "output_delta_down_volume",
    "output_delta_net_volume",
    "feature_clamped_fraction",
    "match_confidence",
    "match_null_fraction",
    "match_entropy",
    "match_displacement",
    "match_similarity",
)


class ExactRankSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world_size: int):
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.length - self.rank + self.world_size - 1) // self.world_size)


def _metric_matrix(values: np.ndarray) -> dict[str, np.ndarray]:
    index = {name: column for column, name in enumerate(METRIC_FIELDS)}
    frames = np.maximum(values[:, index["frames"]], 1.0)
    points = np.maximum(values[:, index["values"]], 1.0)
    return {
        "rmse": np.sqrt(values[:, index["sq_sum"]] / points),
        "contact_iou": values[:, index["contact_iou_sum"]] / frames,
        "volumetric_iou": values[:, index["viou_sum"]] / frames,
        "core_distribution_viou": values[:, index["core_viou_sum"]]
        / np.maximum(values[:, index["core_count"]], 1.0),
        "temporal_accuracy_frame": values[:, index["temporal_correct"]] / frames,
        "false_high_excess_fraction": values[:, index["false_high_excess"]]
        / np.maximum(values[:, index["pred_volume"]], 1e-12),
        "catastrophic_over_rate": values[:, index["cat_over"]]
        / np.maximum(values[:, index["cat_over_denom"]], 1.0),
        "catastrophic_under_rate": values[:, index["cat_under"]]
        / np.maximum(values[:, index["cat_under_denom"]], 1.0),
    }


def _bootstrap_rows(
    sequence_values: Mapping[str, torch.Tensor],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    if iterations <= 0:
        return []
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must lie in (0, 1)")
    arrays = {
        source: values.detach().double().cpu().numpy()
        for source, values in sequence_values.items()
    }
    active = arrays["rgb_reset"][:, METRIC_FIELDS.index("frames")] > 0
    arrays = {source: values[active] for source, values in arrays.items()}
    sequence_count = int(active.sum())
    if sequence_count == 0:
        return []
    rng = np.random.default_rng(int(seed))
    draws: dict[str, dict[str, list[np.ndarray]]] = {
        source: {} for source in arrays
    }
    remaining = int(iterations)
    probability = np.full(sequence_count, 1.0 / sequence_count)
    while remaining:
        chunk = min(128, remaining)
        weights = rng.multinomial(sequence_count, probability, size=chunk).astype(
            np.float64, copy=False
        )
        for source, values in arrays.items():
            metrics = _metric_matrix(weights @ values)
            for metric, metric_values in metrics.items():
                draws[source].setdefault(metric, []).append(metric_values)
        remaining -= chunk
    complete = {
        source: {
            metric: np.concatenate(chunks)[:iterations]
            for metric, chunks in metrics.items()
        }
        for source, metrics in draws.items()
    }
    comparisons = (
        ("real", "rgb_reset"),
        ("real", "cross_sequence"),
        ("real", "contralateral"),
        ("spatial_shuffle", "real"),
        ("lag_reverse", "real"),
        ("affine_perturb", "real"),
    )
    alpha = (1.0 - float(confidence)) * 0.5
    rows = []
    for candidate, reference in comparisons:
        if candidate not in complete or reference not in complete:
            continue
        for metric in complete[candidate]:
            delta = complete[candidate][metric] - complete[reference][metric]
            rows.append(
                {
                    "candidate": candidate,
                    "reference": reference,
                    "metric": metric,
                    "sequence_count": sequence_count,
                    "iterations": int(iterations),
                    "mean_delta": float(delta.mean()),
                    "ci_low": float(np.quantile(delta, alpha)),
                    "ci_high": float(np.quantile(delta, 1.0 - alpha)),
                    "p_delta_gt_zero": float((delta > 0.0).mean()),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_model(
    payload,
    checkpoint: Path,
    data_source,
    *,
    online_base=None,
    dino_weights: str = "",
    online_encoder_chunk_size: int = 128,
) -> TemporalGridTactileModel:
    if payload.get("format") not in {
        TEMPORAL_GRID_FORMAT,
        TEMPORAL_MEMORY_FORMAT,
        TEMPORAL_TRUNK_FORMAT,
    }:
        raise ValueError(
            f"Unsupported temporal-grid checkpoint format={payload.get('format')!r}"
        )
    config = payload.get("model_config", {})
    architecture = str(
        config.get("temporal_architecture")
        or payload.get("temporal_architecture")
        or "grid_difference_v1"
    )
    fusion_config = dict(config.get("fusion") or config.get("adapter") or {})
    palm_indices = tuple(int(value) for value in config.get("palm_vertex_indices", ()))
    if palm_indices != tuple(
        int(value) for value in data_source.palm_vertex_indices
    ):
        raise RuntimeError("Evaluation palm vertices differ from the checkpoint")
    if payload.get("format") == TEMPORAL_TRUNK_FORMAT:
        if architecture != "hierarchical_memory_v3":
            raise RuntimeError("Temporal trunk checkpoint has an invalid architecture")
        if not dino_weights:
            raise ValueError("Temporal trunk evaluation requires DINO weights")
        base_config = dict(config.get("base_model") or {})
        seed = int(config.get("seed", 521))
        torch.manual_seed(seed)
        base_model = build_fresh_temporal_base(
            dino_weights,
            input_resolution=tuple(base_config.get("input_resolution", (256, 192))),
            model_initialization_order=str(
                base_config.get("model_initialization_order", "legacy_decoder_first")
            ),
        )
        actual_head_sha = module_state_sha256(base_model.tactile_head)
        expected_head_sha = str(config.get("initial_tactile_head_sha256") or "")
        if expected_head_sha and actual_head_sha != expected_head_sha:
            raise RuntimeError(
                "Fresh tactile-head initialization contract is not reproducible: "
                f"expected={expected_head_sha}, actual={actual_head_sha}"
            )
        fusion = HierarchicalTemporalMemoryFusionV3(**fusion_config)
        actual_fast_sha = module_state_sha256(fusion.fast_writer)
        expected_fast_sha = str(config.get("initial_fast_writer_sha256") or "")
        if expected_fast_sha and actual_fast_sha != expected_fast_sha:
            raise RuntimeError(
                "Fresh fast-writer initialization contract is not reproducible: "
                f"expected={expected_fast_sha}, actual={actual_fast_sha}"
            )
        model = TemporalMainTrunkV3(
            base_model,
            palm_indices,
            fusion,
            online_encoder_chunk_size=online_encoder_chunk_size,
        )
        state = payload.get("trunk_state_dict")
        if not isinstance(state, Mapping):
            raise ValueError("Temporal trunk checkpoint is missing trunk_state_dict")
        model.load_compact_state_dict(state)
        return model

    base_checkpoint = Path(str(payload.get("base_checkpoint", ""))).expanduser()
    if not base_checkpoint.is_file():
        raise FileNotFoundError(
            f"Temporal checkpoint references missing base checkpoint: {base_checkpoint}"
        )
    expected_sha = str(payload.get("base_checkpoint_sha256", ""))
    actual_sha = file_sha256(base_checkpoint)
    if expected_sha != actual_sha:
        raise RuntimeError(
            f"Base checkpoint SHA mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    if (
        hasattr(data_source, "base_checkpoint_sha256")
        and data_source.base_checkpoint_sha256 != actual_sha
    ):
        raise RuntimeError("Evaluation cache was built from a different RGB baseline")
    decoder, decoder_metadata = load_frozen_fullgrid_decoder(base_checkpoint)
    if dict(config.get("decoder", {})) != decoder_metadata:
        raise RuntimeError("Frozen decoder contract differs from the temporal checkpoint")
    if architecture == "local_memory_v2":
        adapter = TemporalLocalMemoryFusionV2(**fusion_config)
    elif architecture == "grid_difference_v1":
        adapter = TemporalGridAdapterV1(**fusion_config)
    else:
        raise ValueError(f"Unsupported temporal architecture={architecture!r}")
    state = payload.get("fusion_state_dict") or payload.get("adapter_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Temporal checkpoint is missing its fusion state")
    adapter.load_state_dict(state, strict=True)
    online_encoder = (
        FrozenOnlineGridEncoder(online_base) if online_base is not None else None
    )
    return TemporalGridTactileModel(
        decoder,
        palm_indices,
        adapter,
        online_encoder=online_encoder,
        online_encoder_chunk_size=online_encoder_chunk_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-mode", choices=("auto", "online", "cache"), default="auto")
    parser.add_argument("--cache", default="")
    parser.add_argument("--query-manifests", required=True)
    parser.add_argument("--pair-index-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dino-weights", default="")
    parser.add_argument("--data-roots", default="")
    parser.add_argument("--bbox-manifests", default="")
    parser.add_argument("--bbox-source-policy", default="sam3_only")
    parser.add_argument("--bbox-rescale-factor", type=float, default=1.2)
    parser.add_argument("--input-resolution", default="256x192")
    parser.add_argument(
        "--hdf5-manifest-index-dir",
        default=os.environ.get(
            "TEMPORAL_HDF5_INDEX_ROOT",
            "/home/ma-user/work/cfzhao/input_prior_full/state/hdf5_manifest_index",
        ),
    )
    parser.add_argument("--hdf5-handle-cache-size", type=int, default=8)
    parser.add_argument("--online-encoder-chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--max-open-shards", type=int, default=4)
    parser.add_argument("--sources", default=",".join(TEMPORAL_GRID_SOURCES))
    parser.add_argument(
        "--active-lags",
        default="",
        help="Optional comma-separated subset of checkpoint lags for mask attribution.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=521)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--phase-volume-threshold", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--copy-val-metrics-from", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    payload = load_torch_checkpoint(checkpoint)
    model_config = payload.get("model_config", {})
    architecture = str(
        model_config.get("temporal_architecture")
        or payload.get("temporal_architecture")
        or "grid_difference_v1"
    )
    fusion_config = model_config.get("fusion") or model_config.get("adapter") or {}
    history_lags = tuple(int(value) for value in fusion_config.get("history_lags", ()))
    if not history_lags:
        raise ValueError("Temporal checkpoint is missing history_lags")
    if history_lags[0] != 1:
        raise ValueError("Temporal fusion requires lag 1 for phase diagnostics")
    sources = tuple(parse_csv(args.sources))
    if architecture not in {"local_memory_v2", "hierarchical_memory_v3"}:
        sources = tuple(source for source in sources if source != "affine_perturb")
    invalid = sorted(set(sources) - set(TEMPORAL_GRID_SOURCES))
    if invalid or "rgb_reset" not in sources or "real" not in sources:
        raise ValueError(
            f"Sources must include rgb_reset/real and use {TEMPORAL_GRID_SOURCES}; "
            f"invalid={invalid}"
        )
    active_lags = (
        tuple(int(value) for value in parse_csv(args.active_lags))
        if args.active_lags
        else history_lags
    )
    if not active_lags or not set(active_lags).issubset(history_lags):
        raise ValueError(
            f"--active-lags must be a nonempty subset of {history_lags}, got {active_lags}"
        )
    if 1 not in active_lags:
        raise ValueError("--active-lags must retain lag 1 for phase diagnostics")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Temporal-grid evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    manifests = tuple(parse_csv(args.query_manifests))
    checkpoint_data_mode = str(
        model_config.get("data_mode")
        or payload.get("data_config", {}).get("mode")
        or "cache"
    )
    data_mode = checkpoint_data_mode if args.data_mode == "auto" else args.data_mode
    if data_mode != checkpoint_data_mode:
        raise RuntimeError(
            f"Evaluation data mode {data_mode!r} differs from checkpoint "
            f"mode {checkpoint_data_mode!r}"
        )
    if payload.get("format") == TEMPORAL_TRUNK_FORMAT:
        data_contract = dict(payload.get("data_config") or {})
        required_contract = {
            "base_initialization": "from_scratch",
            "model_initialization_order": "legacy_decoder_first",
            "worker_seed_mode": "lightning_legacy",
            "hdf5_sample_order": "legacy_sample_dir_hand",
            "crop_pipeline": "legacy_square_center",
            "optimizer_backend_mode": "legacy_default",
        }
        contract_mismatches = {
            name: {"expected": expected, "actual": model_config.get(name)}
            for name, expected in required_contract.items()
            if model_config.get(name) != expected
        }
        if contract_mismatches:
            raise RuntimeError(
                "Fresh temporal checkpoint violates the initialization contract: "
                f"{contract_mismatches}"
            )
        if payload.get("base_checkpoint") or payload.get("base_checkpoint_sha256"):
            raise RuntimeError(
                "Fresh temporal checkpoint unexpectedly depends on a tactile checkpoint"
            )
        recorded_scale = float(data_contract.get("bbox_rescale_factor", 1.2))
        if not math.isclose(
            float(args.bbox_rescale_factor), recorded_scale, rel_tol=0.0, abs_tol=1e-8
        ):
            raise RuntimeError(
                "Evaluation bbox scale differs from the temporal checkpoint: "
                f"requested={args.bbox_rescale_factor}, checkpoint={recorded_scale}"
            )
        recorded_policy = str(data_contract.get("bbox_source_policy", "sam3_only"))
        if args.bbox_source_policy != recorded_policy:
            raise RuntimeError(
                "Evaluation bbox source policy differs from the temporal checkpoint: "
                f"requested={args.bbox_source_policy}, checkpoint={recorded_policy}"
            )
    pair_root = Path(args.pair_index_root).expanduser().resolve(strict=False)
    online_base = None
    control_sidecar = None
    if data_mode == "online":
        dino_weights = str(args.dino_weights or payload.get("dino_weights") or "")
        if not dino_weights:
            raise ValueError("Online evaluation requires --dino-weights")
        dino_weights = str(Path(dino_weights).expanduser().resolve(strict=True))
        expected_dino_sha = str(payload.get("dino_weights_sha256") or "")
        if expected_dino_sha and file_sha256(dino_weights) != expected_dino_sha:
            raise RuntimeError("DINO weights differ from the temporal checkpoint")
        input_resolution = parse_resolution(args.input_resolution)
        recorded_resolution = tuple(
            int(value)
            for value in (
                model_config.get("base_model", {}).get("input_resolution")
                or payload.get("data_config", {}).get("input_resolution")
                or input_resolution
            )
        )
        if tuple(input_resolution) != recorded_resolution:
            raise RuntimeError(
                "Evaluation input resolution differs from the checkpoint: "
                f"requested={tuple(input_resolution)}, checkpoint={recorded_resolution}"
            )
        base_dataset = build_dataset(
            split=args.split,
            datasets="touchanything",
            input_resolution=input_resolution,
            bbox_rescale_factor=args.bbox_rescale_factor,
            train=False,
            augmentation_enabled=False,
            data_roots=args.data_roots or None,
            query_manifests=manifests,
            bbox_manifests=args.bbox_manifests or None,
            bbox_source_policy=args.bbox_source_policy,
            hdf5_handle_cache_size=args.hdf5_handle_cache_size,
            hdf5_manifest_cache_dir=args.hdf5_manifest_index_dir or None,
            hdf5_batch_read_mode="grouped",
        )
        record_index = OnlineTemporalRecordIndex(
            base_dataset,
            input_resolution=input_resolution,
            bbox_rescale_factor=args.bbox_rescale_factor,
        )
        pair_path = build_online_temporal_pair_index(
            record_index,
            manifests,
            pair_root,
            args.split,
            seed=args.seed,
        )
        palm_indices = tuple(
            int(value) for value in model_config.get("palm_vertex_indices", ())
        )
        dataset = OnlineTemporalDataset(
            base_dataset,
            pair_path,
            palm_vertex_indices=palm_indices,
            history_lags=history_lags,
            include_control=True,
            include_contralateral=True,
            pair_only=False,
        )
        control_bins = np.asarray(
            dataset.control_pressure_bins, dtype=np.int64
        )
        if payload.get("format") != TEMPORAL_TRUNK_FORMAT:
            base_checkpoint = Path(
                str(payload.get("base_checkpoint", ""))
            ).expanduser()
            online_base, _, _ = build_frozen_base(base_checkpoint, dino_weights)
        data_source = dataset
    else:
        if not args.cache:
            raise ValueError("Cache checkpoint evaluation requires --cache")
        pair_args = argparse.Namespace(
            max_open_shards=args.max_open_shards,
            seed=args.seed,
        )
        pair_path = _pair_index(
            args.cache,
            manifests,
            pair_root,
            args.split,
            pair_args,
        )
        control_sidecar = pair_path.with_name(
            f"{pair_path.stem}-rgbmax-control.npz"
        )
        if rank == 0:
            cache_for_control = PartitionedPalmCache(
                args.cache, max_open_shards=args.max_open_shards
            )
            build_prediction_control_bins(
                cache_for_control, pair_path, control_sidecar
            )
        if world_size > 1:
            dist.barrier()
        with np.load(control_sidecar, allow_pickle=False) as control_payload:
            control_bins = np.asarray(
                control_payload["prediction_pressure_bin"], dtype=np.int64
            )
        uses_affine_matching = architecture == "local_memory_v2"
        dataset = TemporalReplayDataset(
            args.cache,
            pair_path,
            include_control=True,
            include_dino_grid=True,
            include_crop_transform=uses_affine_matching,
            include_control_current_grid=False,
            history_lags=history_lags,
            max_open_shards=args.max_open_shards,
            control_pressure_bins=control_bins,
            control_crop_transform_from_current=uses_affine_matching,
        )
        data_source = dataset.cache
    exact_control_by_cache = torch.zeros(len(dataset), dtype=torch.bool)
    pair_current_indices = np.asarray(dataset.arrays["current_index"], dtype=np.int64)
    control_pairs = np.asarray(dataset.control_pair_indices, dtype=np.int64)
    exact_control_by_cache[torch.from_numpy(pair_current_indices)] = torch.from_numpy(
        control_bins == control_bins[control_pairs]
    )
    exact_control_by_cache = exact_control_by_cache.to(device=device)
    model = _load_model(
        payload,
        checkpoint,
        data_source,
        online_base=online_base,
        dino_weights=dino_weights if data_mode == "online" else "",
        online_encoder_chunk_size=args.online_encoder_chunk_size,
    ).to(device).eval()
    sampler = ExactRankSampler(len(dataset), rank, world_size)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    if args.num_workers:
        loader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True,
        )
    loader = DataLoader(dataset, **loader_kwargs)
    metric_values = torch.zeros(
        len(sources), len(SUBSETS), len(METRIC_FIELDS),
        dtype=torch.float64,
        device=device,
    )
    diagnostic_values = torch.zeros(
        len(sources), len(DIAGNOSTICS) + 1, dtype=torch.float64, device=device
    )
    reset_max_drift = torch.zeros((), dtype=torch.float64, device=device)
    sequence_values = torch.zeros(
        len(sources), dataset.sequence_count, len(METRIC_FIELDS),
        dtype=torch.float64,
        device=device,
    )
    processed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            batch = {
                name: value.to(device, non_blocking=True)
                if torch.is_tensor(value)
                else value
                for name, value in raw_batch.items()
            }
            target = batch["tactile_signal"].float()
            valid = batch["has_tactile"].float()
            palm = torch.ones_like(target)
            active_indices = torch.as_tensor(
                [history_lags.index(lag) for lag in active_lags],
                device=device,
                dtype=torch.long,
            )
            real_available = (
                batch["history_available"].index_select(1, active_indices) > 0.5
            ).all(dim=1)
            cross_available = (
                batch["control_history_available"].index_select(1, active_indices)
                > 0.5
            ).all(dim=1)
            contra_available = (
                batch["contralateral_history_available"].index_select(
                    1, active_indices
                )
                > 0.5
            ).all(dim=1)
            matched = real_available & cross_available & contra_available
            matched_rgb_exact = matched & exact_control_by_cache[
                batch["current_index"].long()
            ]
            current_volume = target.sum(dim=1)
            previous_volume = batch["history_tactile_signal"][:, 0].float().sum(dim=1)
            volume_delta = current_volume - previous_volume
            phase_valid = batch["history_available"][:, 0] > 0.5
            onset = phase_valid & (volume_delta > args.phase_volume_threshold)
            release = phase_valid & (volume_delta < -args.phase_volume_threshold)
            stable = phase_valid & ~onset & ~release
            subset_masks = {
                "full": torch.ones_like(real_available),
                "matched": matched,
                "matched_rgb_exact": matched_rgb_exact,
                "onset": onset,
                "stable": stable,
                "release": release,
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch = model.materialize_online_features(batch)
                decoded_current = model.decoder(batch["current_grid"]).index_select(
                    1, model.palm_vertex_indices
                )
                for source_index, source in enumerate(sources):
                    history, time_gap, available, crop_transform = controlled_temporal_grid_inputs(
                        batch, source, shuffle_seed=args.shuffle_seed
                    )
                    lag_mask = available.new_tensor(
                        [1.0 if lag in active_lags else 0.0 for lag in history_lags]
                    )
                    available = available * lag_mask[None]
                    output = model(
                        batch["current_grid"],
                        history,
                        time_gap,
                        available,
                        history_crop_transform=crop_transform,
                        cached_base_logits=batch.get("current_logits"),
                        decoded_current_logits=decoded_current,
                    )
                    if source in {
                        "rgb_reset",
                        "real",
                        "lag_reverse",
                        "spatial_shuffle",
                        "affine_perturb",
                    }:
                        source_available = real_available
                    elif source == "cross_sequence":
                        source_available = cross_available
                    else:
                        source_available = contra_available
                    subset_masks["available"] = source_available
                    contributions = metric_contributions(
                        output["pred_tactile"], target, palm, valid
                    )
                    for subset_index, subset in enumerate(SUBSETS):
                        mask = subset_masks[subset]
                        metric_values[source_index, subset_index] += contributions[
                            mask
                        ].sum(dim=0)
                    prediction_delta = (
                        output["pred_tactile"].float()
                        - output["base_pred_tactile"].float()
                    )
                    diagnostic_batch = torch.stack(
                        (
                            output["feature_delta_rms"].double(),
                            output["feature_clamp_scale"].double(),
                            output["history_available_fraction"].double(),
                            output["history_motion_rms"].double(),
                            output.get(
                                "fast_feature_delta_rms",
                                output["feature_delta_rms"].new_zeros(
                                    output["feature_delta_rms"].shape
                                ),
                            ).double(),
                            output.get(
                                "medium_feature_delta_rms",
                                output["feature_delta_rms"].new_zeros(
                                    output["feature_delta_rms"].shape
                                ),
                            ).double(),
                            output["decoder_logit_delta"].float()
                            .pow(2)
                            .mean(dim=1)
                            .sqrt()
                            .double(),
                            prediction_delta.clamp_min(0.0).sum(dim=1).double(),
                            (-prediction_delta).clamp_min(0.0).sum(dim=1).double(),
                            prediction_delta.sum(dim=1).double(),
                            (output["feature_clamp_scale"] < 0.999).double(),
                            output.get(
                                "match_confidence", output["feature_delta_rms"].new_zeros(
                                    output["feature_delta_rms"].shape
                                )
                            ).double(),
                            output.get(
                                "match_null_fraction", output["feature_delta_rms"].new_zeros(
                                    output["feature_delta_rms"].shape
                                )
                            ).double(),
                            output.get(
                                "match_entropy", output["feature_delta_rms"].new_zeros(
                                    output["feature_delta_rms"].shape
                                )
                            ).double(),
                            output.get(
                                "match_displacement", output["feature_delta_rms"].new_zeros(
                                    output["feature_delta_rms"].shape
                                )
                            ).double(),
                            output.get(
                                "match_similarity", output["feature_delta_rms"].new_zeros(
                                    output["feature_delta_rms"].shape
                                )
                            ).double(),
                        ),
                        dim=1,
                    )
                    diagnostic_values[source_index, :-1] += diagnostic_batch.sum(dim=0)
                    diagnostic_values[source_index, -1] += len(target)
                    if source == "rgb_reset":
                        reset_max_drift = torch.maximum(
                            reset_max_drift,
                            output["decoder_logit_delta"].double().abs().max(),
                        )
                    sequence_mask = matched & (batch["sequence_id"] >= 0)
                    if bool(sequence_mask.any()):
                        sequence_values[source_index].index_add_(
                            0,
                            batch["sequence_id"][sequence_mask].long(),
                            contributions[sequence_mask],
                        )
            processed += len(target)
            if rank == 0 and (
                (batch_index + 1) % max(args.progress_every, 1) == 0
                or processed >= len(sampler)
            ):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[temporal-grid:{args.split}] local={processed:,}/"
                    f"{len(sampler):,} rate={processed / elapsed:,.1f} samples/s",
                    flush=True,
                )
    if world_size > 1:
        dist.all_reduce(metric_values)
        dist.all_reduce(diagnostic_values)
        dist.all_reduce(sequence_values)
        dist.all_reduce(reset_max_drift, op=dist.ReduceOp.MAX)
    reset_drift = float(reset_max_drift.item())
    if reset_drift > 1e-6:
        if world_size > 1:
            dist.destroy_process_group()
        raise RuntimeError(
            f"RGB reset is not numerically identical; max logit drift={reset_drift}"
        )
    if rank == 0:
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        metric_rows = []
        summary_lookup: dict[str, dict[str, dict[str, float]]] = {}
        for source_index, source in enumerate(sources):
            summary_lookup[source] = {}
            for subset_index, subset in enumerate(SUBSETS):
                summary = summarize_metric_values(
                    metric_values[source_index, subset_index]
                )
                summary_lookup[source][subset] = summary
                metric_rows.append({"source": source, "subset": subset, **summary})
        _write_csv(output_dir / "temporal_grid_metrics.csv", metric_rows)
        diagnostic_rows = []
        for source_index, source in enumerate(sources):
            count = diagnostic_values[source_index, -1].clamp_min(1.0)
            diagnostic_rows.append(
                {
                    "source": source,
                    **{
                        name: float(
                            (diagnostic_values[source_index, index] / count).item()
                        )
                        for index, name in enumerate(DIAGNOSTICS)
                    },
                }
            )
        _write_csv(output_dir / "temporal_grid_diagnostics.csv", diagnostic_rows)
        sequence_mapping = {
            source: sequence_values[index]
            for index, source in enumerate(sources)
        }
        bootstrap_rows = _bootstrap_rows(
            sequence_mapping,
            iterations=args.bootstrap_iterations,
            confidence=args.bootstrap_confidence,
            seed=args.seed,
        )
        _write_csv(output_dir / "sequence_bootstrap.csv", bootstrap_rows)
        result = {
            "schema": "tactile_temporal_feature_fusion_eval_v3",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "split": args.split,
            "history_lags": list(history_lags),
            "active_lags": list(active_lags),
            "temporal_architecture": architecture,
            "initialization_contract": {
                "base_initialization": model_config.get("base_initialization", ""),
                "model_initialization_order": model_config.get(
                    "model_initialization_order", ""
                ),
                "initial_tactile_head_sha256": model_config.get(
                    "initial_tactile_head_sha256", ""
                ),
                "initial_fast_writer_sha256": model_config.get(
                    "initial_fast_writer_sha256", ""
                ),
                "worker_seed_mode": model_config.get("worker_seed_mode", ""),
                "hdf5_sample_order": model_config.get("hdf5_sample_order", ""),
                "crop_pipeline": model_config.get("crop_pipeline", ""),
                "optimizer_backend_mode": model_config.get(
                    "optimizer_backend_mode", ""
                ),
            },
            "sources": list(sources),
            "subsets": list(SUBSETS),
            "metrics": summary_lookup,
            "diagnostics": {
                row["source"]: row for row in diagnostic_rows
            },
            "reset_max_logit_drift": reset_drift,
            "pair_index": str(pair_path),
            "data_mode": data_mode,
            "control_bin_source": (
                "none_label_free" if data_mode == "online" else "frozen_rgb_prediction"
            ),
            "control_sidecar": (
                str(control_sidecar) if control_sidecar is not None else None
            ),
            "feature_cache": (
                str(Path(args.cache).resolve()) if data_mode == "cache" else None
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lines = [
            "Temporal Feature Fusion Evaluation",
            f"Checkpoint: {checkpoint}",
            f"Split: {args.split}",
            f"Lags: {history_lags}",
            f"Active lags: {active_lags}",
            f"Architecture: {architecture}",
            f"Reset max logit drift: {reset_drift:.3e}",
            "",
        ]
        for subset in (
            "full",
            "matched",
            "matched_rgb_exact",
            "onset",
            "stable",
            "release",
        ):
            lines.append(f"[{subset}]")
            for source in sources:
                value = summary_lookup[source][subset]
                lines.append(
                    f"{source:18s} n={value['frame_count']:.0f} "
                    f"RMSE={value['rmse']:.6f} Contact={value['contact_iou']:.6f} "
                    f"V-IoU={value['volumetric_iou']:.6f} "
                    f"CoreLoc={value['core_distribution_viou']:.6f} "
                    f"FalseHigh={value['false_high_excess_fraction']:.6f}"
                )
            lines.append("")
        (output_dir / "eval_temporal_grid.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        source_metrics = Path(args.copy_val_metrics_from).expanduser()
        if args.copy_val_metrics_from and source_metrics.is_file():
            shutil.copy2(source_metrics, output_dir / "training_val_metrics.csv")
        print(f"Temporal grid report: {output_dir}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
