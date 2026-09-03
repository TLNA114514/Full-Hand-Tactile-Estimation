#!/usr/bin/env python3
"""Evaluate temporal clip trunks with full-frame and endpoint controls."""

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
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal
from tactile_input_priors.online_temporal import (
    OnlineTemporalClipDataset,
    OnlineTemporalRecordIndex,
    build_online_temporal_pair_index,
)
from tactile_input_priors.prior_metrics import (
    METRIC_FIELDS,
    metric_contributions,
    summarize_metric_values,
)
from tactile_input_priors.runtime import (
    build_dataset,
    file_sha256,
    load_torch_checkpoint,
    parse_csv,
    parse_resolution,
)
from tactile_input_priors.temporal_grid import (
    TEMPORAL_CLIP_FORMAT,
    TEMPORAL_FULLGRID_FORMAT,
    TEMPORAL_ONLINEHMR_FORMAT,
    CausalClipTransformerFusionV4,
    FullGrid6144SpatiotemporalFusionV5,
    FullGrid6144TemporalMainTrunkV5,
    OnlineHMRPatchKVFusionV6,
    OnlineHMRPatchKVTemporalMainTrunkV6,
    TemporalClipMainTrunkV4,
    build_fresh_temporal_base,
    module_state_sha256,
)


SOURCES = (
    "rgb_reset",
    "real",
    "cross_sequence",
    "frame_shuffle",
    "lag_reverse",
    "spatial_shuffle",
    "past_only",
    "future_only",
    "repeat_current",
    "memory1",
)
SUBSETS = ("all_frames", "endpoints", "onset", "stable", "release")
FULL_FRAME_SOURCES = {
    "rgb_reset",
    "real",
    "past_only",
    "future_only",
    "memory1",
}
ONLINEHMR_ARCHITECTURE = "onlinehmr_patch_kv_v6"
DIAGNOSTICS = (
    "feature_delta_rms",
    "feature_clamp_scale",
    "history_available_fraction",
    "history_motion_rms",
    "spatial_delta_rms",
    "temporal_delta_rms",
    "temporal_layer_scale",
    "logit_delta_rms",
    "output_delta_up_volume",
    "output_delta_down_volume",
    "output_delta_net_volume",
)
CONTACT_BINS = 1000


class ExactRankSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world_size: int):
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.length - self.rank + self.world_size - 1) // self.world_size)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def _update_contact_counts(
    histogram: torch.Tensor,
    confusion: torch.Tensor,
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> None:
    if not bool(mask.any()):
        return
    probabilities = torch.sigmoid(logits.float())[mask].flatten()
    labels = (target.float()[mask].flatten() > float(threshold))
    indices = torch.clamp(
        (probabilities * CONTACT_BINS).long(), max=CONTACT_BINS - 1
    )
    histogram[1] += torch.bincount(
        indices[labels], minlength=CONTACT_BINS
    ).double()
    histogram[0] += torch.bincount(
        indices[~labels], minlength=CONTACT_BINS
    ).double()
    predicted = probabilities >= 0.5
    confusion[0] += (predicted & labels).sum()
    confusion[1] += (predicted & ~labels).sum()
    confusion[2] += (~predicted & labels).sum()
    confusion[3] += (~predicted & ~labels).sum()


def _summarize_contact_counts(
    histogram: torch.Tensor, confusion: torch.Tensor
) -> dict[str, float]:
    negative, positive = histogram
    tp_curve = positive.flip(0).cumsum(0)
    fp_curve = negative.flip(0).cumsum(0)
    precision_curve = tp_curve / (tp_curve + fp_curve).clamp_min(1.0)
    recall_increment = positive.flip(0) / positive.sum().clamp_min(1.0)
    average_precision = (precision_curve * recall_increment).sum()
    true_positive, false_positive, false_negative, _ = confusion
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    centers = (
        torch.arange(CONTACT_BINS, device=histogram.device, dtype=torch.float64)
        + 0.5
    ) / float(CONTACT_BINS)
    totals = positive + negative
    empirical = positive / totals.clamp_min(1.0)
    calibration_error = (
        (empirical - centers).abs() * totals
    ).sum() / totals.sum().clamp_min(1.0)
    brier = (
        positive * (1.0 - centers).square() + negative * centers.square()
    ).sum() / totals.sum().clamp_min(1.0)
    return {
        "contact_head_ap": float(average_precision.item()),
        "contact_head_precision": float(precision.item()),
        "contact_head_recall": float(recall.item()),
        "contact_head_f1": float(f1.item()),
        "contact_head_ece": float(calibration_error.item()),
        "contact_head_brier": float(brier.item()),
    }


def _load_model(payload, dino_weights: str, encoder_chunk: int):
    checkpoint_format = payload.get("format")
    if checkpoint_format not in {
        TEMPORAL_CLIP_FORMAT,
        TEMPORAL_FULLGRID_FORMAT,
        TEMPORAL_ONLINEHMR_FORMAT,
    }:
        raise ValueError(
            "Expected a temporal clip checkpoint, got "
            f"{checkpoint_format!r}"
        )
    config = dict(payload.get("model_config") or {})
    architecture = str(config.get("temporal_architecture") or "")
    if architecture not in {
        "causal_clip_transformer_v4",
        "fullgrid6144_bidirectional_v5",
        ONLINEHMR_ARCHITECTURE,
    }:
        raise RuntimeError("Checkpoint does not describe a supported clip trunk")
    seed = int(config.get("seed", 521))
    torch.manual_seed(seed)
    base_config = dict(config.get("base_model") or {})
    base_model = build_fresh_temporal_base(
        dino_weights,
        input_resolution=tuple(base_config.get("input_resolution", (256, 192))),
        model_initialization_order=str(
            base_config.get("model_initialization_order", "legacy_decoder_first")
        ),
    )
    actual_head = module_state_sha256(base_model.tactile_head)
    expected_head = str(config.get("initial_tactile_head_sha256") or "")
    if expected_head and actual_head != expected_head:
        raise RuntimeError(
            "Causal clip tactile-head initialization mismatch: "
            f"expected={expected_head}, actual={actual_head}"
        )
    fusion_class = (
        OnlineHMRPatchKVFusionV6
        if architecture == ONLINEHMR_ARCHITECTURE
        else FullGrid6144SpatiotemporalFusionV5
        if architecture == "fullgrid6144_bidirectional_v5"
        else CausalClipTransformerFusionV4
    )
    model_class = (
        OnlineHMRPatchKVTemporalMainTrunkV6
        if architecture == ONLINEHMR_ARCHITECTURE
        else FullGrid6144TemporalMainTrunkV5
        if architecture == "fullgrid6144_bidirectional_v5"
        else TemporalClipMainTrunkV4
    )
    fusion = fusion_class(**dict(config["fusion"]))
    actual_fusion = module_state_sha256(fusion)
    expected_fusion = str(
        config.get("initial_fusion_sha256")
        or config.get("initial_temporal_module_sha256")
        or config.get("initial_fast_writer_sha256")
        or ""
    )
    if expected_fusion and actual_fusion != expected_fusion:
        raise RuntimeError(
            "Causal clip Transformer initialization mismatch: "
            f"expected={expected_fusion}, actual={actual_fusion}"
        )
    if architecture == "fullgrid6144_bidirectional_v5":
        expected_spatial = str(config.get("initial_spatial_module_sha256") or "")
        expected_temporal = str(config.get("initial_temporal_module_sha256") or "")
        actual_spatial = module_state_sha256(fusion.spatial_blocks)
        actual_temporal = module_state_sha256(fusion.temporal_blocks)
        if expected_spatial and actual_spatial != expected_spatial:
            raise RuntimeError(
                "FullGrid6144 spatial initialization mismatch: "
                f"expected={expected_spatial}, actual={actual_spatial}"
            )
        if expected_temporal and actual_temporal != expected_temporal:
            raise RuntimeError(
                "FullGrid6144 temporal initialization mismatch: "
                f"expected={expected_temporal}, actual={actual_temporal}"
            )
    model = model_class(
        base_model,
        tuple(int(value) for value in config["palm_vertex_indices"]),
        fusion,
        online_encoder_chunk_size=encoder_chunk,
    )
    expected_contact = str(config.get("initial_contact_head_sha256") or "")
    if expected_contact:
        actual_contact = module_state_sha256(model.contact_head)
        if actual_contact != expected_contact:
            raise RuntimeError(
                "FullGrid6144 contact-head initialization mismatch: "
                f"expected={expected_contact}, actual={actual_contact}"
            )
    state = payload.get("trunk_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Causal clip checkpoint is missing trunk_state_dict")
    model.load_compact_state_dict(state)
    return model, config


def _endpoint_mask(valid: torch.Tensor) -> torch.Tensor:
    positions = valid.long().sum(dim=1).clamp_min(1) - 1
    result = torch.zeros_like(valid, dtype=torch.bool)
    result.scatter_(1, positions[:, None], True)
    return result & (valid > 0.5)


def _controlled_clip(batch, source: str, seed: int, architecture: str):
    grids = batch["clip_grids"].clone()
    valid = batch["clip_valid"].clone()
    time_values = batch["clip_time"]
    affines = batch["clip_crop_affine"].clone()
    lengths = valid.long().sum(dim=1)
    default_mode = "causal" if architecture == ONLINEHMR_ARCHITECTURE else "bidirectional"
    if source == "rgb_reset":
        if architecture == "causal_clip_transformer_v4":
            for row, length in enumerate(lengths.tolist()):
                valid[row, : max(length - 1, 0)] = 0.0
        return grids, time_values, valid, affines, "self_only"
    if source == "real":
        return grids, time_values, valid, affines, default_mode
    if source in {"past_only", "future_only"}:
        if architecture == ONLINEHMR_ARCHITECTURE and source == "past_only":
            return grids, time_values, valid, affines, "causal"
        return grids, time_values, valid, affines, source
    if source == "memory1":
        if architecture != ONLINEHMR_ARCHITECTURE:
            raise ValueError("memory1 requires an OnlineHMR patch-KV checkpoint")
        return grids, time_values, valid, affines, "memory1"
    if source == "cross_sequence":
        donor = batch["control_clip_grids"]
        donor_valid = batch["control_clip_valid"]
        for row, length in enumerate(lengths.tolist()):
            endpoint = max(length - 1, 0)
            grids[row, :endpoint] = donor[row, :endpoint]
            valid[row, :endpoint] = donor_valid[row, :endpoint]
        return grids, time_values, valid, affines, default_mode
    if source in {"frame_shuffle", "lag_reverse"}:
        for row, length in enumerate(lengths.tolist()):
            endpoint = max(length - 1, 0)
            if endpoint <= 1:
                continue
            if source == "lag_reverse":
                permutation = torch.arange(endpoint - 1, -1, -1, device=grids.device)
            else:
                generator = torch.Generator(device="cpu").manual_seed(
                    int(seed) + int(batch["clip_index"][row])
                )
                permutation = torch.randperm(endpoint, generator=generator).to(
                    grids.device
                )
            grids[row, :endpoint] = grids[row, :endpoint].index_select(0, permutation)
        return grids, time_values, valid, affines, default_mode
    if source == "spatial_shuffle":
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        permutation = torch.randperm(
            grids.shape[-2] * grids.shape[-1], generator=generator
        ).to(grids.device)
        for row, length in enumerate(lengths.tolist()):
            endpoint = max(length - 1, 0)
            if endpoint:
                history = grids[row, :endpoint].flatten(-2).index_select(
                    -1, permutation
                )
                grids[row, :endpoint] = history.reshape_as(grids[row, :endpoint])
        return grids, time_values, valid, affines, default_mode
    if source == "repeat_current":
        for row, length in enumerate(lengths.tolist()):
            endpoint = max(length - 1, 0)
            if endpoint:
                grids[row, :endpoint] = grids[row, endpoint]
                affines[row, :endpoint] = affines[row, endpoint]
        return grids, time_values, valid, affines, default_mode
    raise ValueError(f"Unsupported clip source={source!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--query-manifests", required=True)
    parser.add_argument("--pair-index-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dino-weights", required=True)
    parser.add_argument("--data-roots", default="")
    parser.add_argument("--bbox-manifests", default="")
    parser.add_argument("--bbox-source-policy", default="sam3_only")
    parser.add_argument("--bbox-rescale-factor", type=float, default=1.2)
    parser.add_argument("--input-resolution", default="256x192")
    parser.add_argument("--hdf5-manifest-index-dir", default="")
    parser.add_argument("--hdf5-handle-cache-size", type=int, default=8)
    parser.add_argument("--online-encoder-chunk-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--sources", default=",".join(SOURCES[:6]))
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--phase-volume-threshold", type=float, default=5.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--copy-val-metrics-from", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    payload = load_torch_checkpoint(checkpoint)
    dino_weights = str(Path(args.dino_weights).expanduser().resolve(strict=True))
    expected_dino = str(payload.get("dino_weights_sha256") or "")
    if expected_dino and file_sha256(dino_weights) != expected_dino:
        raise RuntimeError("Evaluation DINO weights differ from the checkpoint")
    sources = tuple(parse_csv(args.sources))
    invalid_sources = sorted(set(sources) - set(SOURCES))
    if invalid_sources or "real" not in sources or "rgb_reset" not in sources:
        raise ValueError(f"Invalid clip sources: {invalid_sources}")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Causal clip evaluation requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")

    model, model_config = _load_model(
        payload, dino_weights, args.online_encoder_chunk_size
    )
    architecture = str(model_config["temporal_architecture"])
    v5_only_sources = {"future_only"}
    if architecture != "fullgrid6144_bidirectional_v5" and set(sources) & v5_only_sources:
        raise ValueError(
            "future_only requires the bidirectional FullGrid6144 V5 checkpoint"
        )
    if "memory1" in sources and architecture != ONLINEHMR_ARCHITECTURE:
        raise ValueError("memory1 requires the OnlineHMR patch-KV checkpoint")
    model = model.to(device).eval()
    fusion_config = dict(model_config["fusion"])
    clip_length = int(fusion_config["clip_length"])
    input_resolution = parse_resolution(args.input_resolution)
    recorded_resolution = tuple(
        int(value) for value in model_config["base_model"]["input_resolution"]
    )
    if tuple(input_resolution) != recorded_resolution:
        raise RuntimeError("Evaluation resolution differs from clip checkpoint")
    data_config = dict(payload.get("data_config") or {})
    recorded_bbox_scale = float(data_config.get("bbox_rescale_factor", 1.2))
    if not math.isclose(
        float(args.bbox_rescale_factor), recorded_bbox_scale, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            "Evaluation bbox scale differs from clip checkpoint: "
            f"expected={recorded_bbox_scale}, actual={args.bbox_rescale_factor}"
        )
    recorded_bbox_policy = str(data_config.get("bbox_source_policy") or "sam3_only")
    if str(args.bbox_source_policy) != recorded_bbox_policy:
        raise RuntimeError(
            "Evaluation bbox source policy differs from clip checkpoint: "
            f"expected={recorded_bbox_policy!r}, actual={args.bbox_source_policy!r}"
        )
    manifests = tuple(parse_csv(args.query_manifests))
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
        args.pair_index_root,
        args.split,
        seed=args.seed,
    )
    dataset = OnlineTemporalClipDataset(
        base_dataset,
        pair_path,
        palm_vertex_indices=tuple(int(value) for value in model.palm_vertex_indices),
        clip_length=clip_length,
        include_control="cross_sequence" in sources,
        seed=args.seed,
    )
    expected_frame_count = len(base_dataset)
    partition_frame_count = int(dataset.clip_lengths.sum())
    if partition_frame_count != expected_frame_count:
        raise RuntimeError(
            "Temporal clip partition does not cover the evaluation split exactly: "
            f"dataset={expected_frame_count}, partition={partition_frame_count}"
        )
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
            prefetch_factor=args.prefetch_factor, persistent_workers=True
        )
    loader = DataLoader(dataset, **loader_kwargs)

    metric_values = torch.zeros(
        len(sources), len(SUBSETS), len(METRIC_FIELDS),
        dtype=torch.float64, device=device,
    )
    position_metric_values = torch.zeros(
        len(sources), clip_length, len(METRIC_FIELDS),
        dtype=torch.float64, device=device,
    )
    diagnostic_values = torch.zeros(
        len(sources), len(DIAGNOSTICS) + 1, dtype=torch.float64, device=device
    )
    contact_histograms = torch.zeros(
        len(sources),
        len(SUBSETS),
        2,
        CONTACT_BINS,
        dtype=torch.float64,
        device=device,
    )
    contact_confusions = torch.zeros(
        len(sources),
        len(SUBSETS),
        4,
        dtype=torch.float64,
        device=device,
    )
    contact_threshold = float(
        (model_config.get("contact_head") or {}).get("pressure_threshold", 0.10)
    )
    reset_max_drift = torch.zeros((), dtype=torch.float64, device=device)
    kv_equivalence_max_abs = torch.zeros((), dtype=torch.float64, device=device)
    kv_equivalence_check_count = torch.zeros((), dtype=torch.float64, device=device)
    # clips, non-padding source frames, source frames carrying tactile labels.
    coverage_values = torch.zeros(3, dtype=torch.float64, device=device)
    kv_equivalence_checked = False
    processed = 0
    started = time.monotonic()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            batch = {
                name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for name, value in raw_batch.items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch = model.materialize_online_features(batch)
                if architecture == ONLINEHMR_ARCHITECTURE and not kv_equivalence_checked:
                    full_rows = torch.nonzero(
                        batch["clip_valid"].long().sum(dim=1) == clip_length,
                        as_tuple=False,
                    ).flatten()
                    if full_rows.numel() > 0:
                        row = full_rows[:1]
                        patch_tokens = model._project_patch_tokens(
                            batch["clip_grids"].index_select(0, row)
                        )
                        kv_error = model.fusion.incremental_equivalence_max_abs(
                            patch_tokens,
                            batch["clip_time"].index_select(0, row).float(),
                        )
                        kv_equivalence_max_abs.fill_(kv_error)
                        kv_equivalence_check_count.add_(1.0)
                        kv_equivalence_checked = True
                        if kv_error > 2e-3:
                            raise RuntimeError(
                                "OnlineHMR batched and KV-cache inference differ: "
                                f"max_abs={kv_error:.6g}"
                            )
                endpoint = _endpoint_mask(batch["clip_valid"])
                target = batch["clip_tactile_signal"].float()
                has_tactile = batch["clip_has_tactile"].float() * batch["clip_valid"]
                coverage_values[0] += batch["clip_valid"].shape[0]
                coverage_values[1] += batch["clip_valid"].double().sum()
                coverage_values[2] += (has_tactile > 0.5).double().sum()
                previous_target = torch.roll(target, shifts=1, dims=1)
                volume_delta = target.sum(dim=2) - previous_target.sum(dim=2)
                phase_valid = endpoint & (
                    torch.arange(clip_length, device=device)[None] > 0
                )
                subset_masks = {
                    "all_frames": batch["clip_valid"] > 0.5,
                    "endpoints": endpoint,
                    "onset": phase_valid & (volume_delta > args.phase_volume_threshold),
                    "stable": phase_valid & (
                        volume_delta.abs() <= args.phase_volume_threshold
                    ),
                    "release": phase_valid & (volume_delta < -args.phase_volume_threshold),
                }
                controlled_outputs: dict[str, dict[str, torch.Tensor]] = {}
                reset_grids, reset_times, reset_valid, reset_affines, reset_mode = (
                    _controlled_clip(batch, "rgb_reset", args.seed, architecture)
                )
                if architecture in {
                    "fullgrid6144_bidirectional_v5",
                    ONLINEHMR_ARCHITECTURE,
                }:
                    reset_output = model.forward_clip(
                        reset_grids,
                        reset_times,
                        reset_valid,
                        reset_affines,
                        decode_base=False,
                        attention_mode=reset_mode,
                    )
                else:
                    reset_output = model.forward_clip(
                        reset_grids,
                        reset_times,
                        reset_valid,
                        reset_affines,
                        decode_base=False,
                    )
                controlled_outputs["rgb_reset"] = reset_output
                reset_prediction = reset_output["pred_tactile"].float()
                reset_logits = reset_output["pred_logits"].float()
                for source_index, source in enumerate(sources):
                    output = controlled_outputs.get(source)
                    if output is None:
                        grids, times, valid, affines, attention_mode = _controlled_clip(
                            batch, source, args.seed, architecture
                        )
                        if architecture in {
                            "fullgrid6144_bidirectional_v5",
                            ONLINEHMR_ARCHITECTURE,
                        }:
                            output = model.forward_clip(
                                grids,
                                times,
                                valid,
                                affines,
                                decode_base=False,
                                attention_mode=attention_mode,
                            )
                        else:
                            output = model.forward_clip(
                                grids,
                                times,
                                valid,
                                affines,
                                decode_base=False,
                            )
                        controlled_outputs[source] = output
                    prediction = output["pred_tactile"].float()
                    contributions = metric_contributions(
                        prediction.reshape(-1, prediction.shape[-1]),
                        target.reshape(-1, target.shape[-1]),
                        torch.ones_like(target).reshape(-1, target.shape[-1]),
                        has_tactile.reshape(-1),
                    ).reshape(target.shape[0], clip_length, -1)
                    for subset_index, subset in enumerate(SUBSETS):
                        if source not in FULL_FRAME_SOURCES and subset == "all_frames":
                            continue
                        mask = subset_masks[subset]
                        if source == "cross_sequence":
                            mask = mask & (
                                batch["control_is_cross_sequence"][:, None] > 0.5
                            )
                        metric_values[source_index, subset_index] += contributions[
                            mask
                        ].sum(dim=0)
                        if "contact_logits" in output:
                            _update_contact_counts(
                                contact_histograms[source_index, subset_index],
                                contact_confusions[source_index, subset_index],
                                output["contact_logits"],
                                target,
                                mask & (has_tactile > 0.5),
                                contact_threshold,
                            )
                    if source in FULL_FRAME_SOURCES:
                        for position in range(clip_length):
                            position_metric_values[source_index, position] += (
                                contributions[:, position].sum(dim=0)
                            )
                    frame_mask = (
                        batch["clip_valid"] > 0.5
                        if source in FULL_FRAME_SOURCES
                        else endpoint
                    )
                    prediction_delta = prediction - reset_prediction
                    logit_rms = (output["pred_logits"].float() - reset_logits).pow(
                        2
                    ).mean(dim=2).sqrt()

                    def frame_diagnostic(name: str) -> torch.Tensor:
                        value = output.get(name)
                        if value is None:
                            return torch.zeros_like(batch["clip_valid"])
                        value = value.float()
                        if value.ndim == 0:
                            return value.expand_as(batch["clip_valid"])
                        return value

                    diagnostic_batch = torch.stack(
                        (
                            frame_diagnostic("feature_delta_rms"),
                            frame_diagnostic("feature_clamp_scale"),
                            frame_diagnostic("history_available_fraction"),
                            frame_diagnostic("history_motion_rms"),
                            frame_diagnostic("spatial_delta_rms"),
                            frame_diagnostic("temporal_delta_rms"),
                            frame_diagnostic("temporal_layer_scale"),
                            logit_rms,
                            prediction_delta.clamp_min(0.0).sum(dim=2),
                            (-prediction_delta).clamp_min(0.0).sum(dim=2),
                            prediction_delta.sum(dim=2),
                        ),
                        dim=2,
                    ).double()
                    diagnostic_values[source_index, :-1] += diagnostic_batch[
                        frame_mask
                    ].sum(dim=0)
                    diagnostic_values[source_index, -1] += frame_mask.sum()
                    if source == "rgb_reset":
                        reset_max_drift = torch.maximum(
                            reset_max_drift,
                            (output["pred_logits"].float() - reset_logits)
                            .double()
                            .abs()
                            .max(),
                        )
            processed += len(batch["clip_valid"])
            if rank == 0 and (
                (batch_index + 1) % max(args.progress_every, 1) == 0
                or processed >= len(sampler)
            ):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[temporal-clip:{args.split}] clips={processed:,}/{len(sampler):,} "
                    f"rate={processed / elapsed:,.1f} clips/s",
                    flush=True,
                )
    if world_size > 1:
        dist.all_reduce(metric_values)
        dist.all_reduce(position_metric_values)
        dist.all_reduce(diagnostic_values)
        dist.all_reduce(contact_histograms)
        dist.all_reduce(contact_confusions)
        dist.all_reduce(reset_max_drift, op=dist.ReduceOp.MAX)
        dist.all_reduce(kv_equivalence_max_abs, op=dist.ReduceOp.MAX)
        dist.all_reduce(kv_equivalence_check_count)
        dist.all_reduce(coverage_values)
    evaluated_clip_count = int(round(float(coverage_values[0].item())))
    evaluated_frame_count = int(round(float(coverage_values[1].item())))
    evaluated_labeled_frame_count = int(round(float(coverage_values[2].item())))
    if evaluated_clip_count != len(dataset):
        raise RuntimeError(
            "Distributed temporal evaluation lost or duplicated clips: "
            f"expected={len(dataset)}, evaluated={evaluated_clip_count}"
        )
    if evaluated_frame_count != expected_frame_count:
        raise RuntimeError(
            "Distributed temporal evaluation did not cover every source frame exactly once: "
            f"expected={expected_frame_count}, evaluated={evaluated_frame_count}"
        )
    if "real" in sources:
        real_index = sources.index("real")
        all_frames_index = SUBSETS.index("all_frames")
        metric_frame_count = int(
            round(float(metric_values[real_index, all_frames_index, 0].item()))
        )
        if metric_frame_count != evaluated_labeled_frame_count:
            raise RuntimeError(
                "real/all_frames metrics do not cover every labeled frame exactly once: "
                f"expected={evaluated_labeled_frame_count}, metric={metric_frame_count}"
            )
    else:
        metric_frame_count = 0
    reset_drift = float(reset_max_drift.item())
    if reset_drift > 1e-6:
        raise RuntimeError(f"Clip RGB reset drifted by {reset_drift}")
    if (
        architecture == ONLINEHMR_ARCHITECTURE
        and float(kv_equivalence_check_count.item()) <= 0.0
    ):
        raise RuntimeError("No full clip was available for OnlineHMR KV verification")
    if rank == 0:
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        metric_rows = []
        contact_rows = []
        lookup: dict[str, dict[str, dict[str, float]]] = {}
        contact_lookup: dict[str, dict[str, dict[str, float]]] = {}
        for source_index, source in enumerate(sources):
            lookup[source] = {}
            contact_lookup[source] = {}
            for subset_index, subset in enumerate(SUBSETS):
                summary = summarize_metric_values(
                    metric_values[source_index, subset_index]
                )
                contact_summary = _summarize_contact_counts(
                    contact_histograms[source_index, subset_index],
                    contact_confusions[source_index, subset_index],
                )
                lookup[source][subset] = summary
                contact_lookup[source][subset] = contact_summary
                metric_rows.append({"source": source, "subset": subset, **summary})
                contact_rows.append(
                    {"source": source, "subset": subset, **contact_summary}
                )
        _write_csv(output_dir / "temporal_clip_metrics.csv", metric_rows)
        position_rows = []
        for source_index, source in enumerate(sources):
            if source not in FULL_FRAME_SOURCES:
                continue
            for position in range(clip_length):
                position_rows.append(
                    {
                        "source": source,
                        "frame_position": position,
                        **summarize_metric_values(
                            position_metric_values[source_index, position]
                        ),
                    }
                )
        _write_csv(
            output_dir / "temporal_clip_position_metrics.csv", position_rows
        )
        if hasattr(model, "contact_head"):
            _write_csv(output_dir / "temporal_clip_contact_metrics.csv", contact_rows)
        diagnostic_rows = []
        for source_index, source in enumerate(sources):
            count = diagnostic_values[source_index, -1].clamp_min(1.0)
            diagnostic_rows.append(
                {
                    "source": source,
                    **{
                        name: float((diagnostic_values[source_index, offset] / count).item())
                        for offset, name in enumerate(DIAGNOSTICS)
                    },
                }
            )
        _write_csv(output_dir / "temporal_clip_diagnostics.csv", diagnostic_rows)
        result = {
            "schema": "tactile_temporal_clip_eval_v2",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "split": args.split,
            "clip_length": clip_length,
            "sources": list(sources),
            "temporal_architecture": architecture,
            "control_scope": {
                "full_frames": sorted(FULL_FRAME_SOURCES),
                "endpoint_only": sorted(set(sources) - FULL_FRAME_SOURCES),
            },
            "metrics": lookup,
            "contact_head_metrics": contact_lookup if hasattr(model, "contact_head") else {},
            "contact_pressure_threshold": contact_threshold,
            "diagnostics": {row["source"]: row for row in diagnostic_rows},
            "reset_max_logit_drift": reset_drift,
            "kv_cache_equivalence_max_abs": float(
                kv_equivalence_max_abs.item()
            ),
            "pair_index": str(pair_path),
            "clip_contract_sha256": dataset.config_sha256,
            "evaluation_coverage": {
                "dataset_frame_count": expected_frame_count,
                "partition_frame_count": partition_frame_count,
                "evaluated_clip_count": evaluated_clip_count,
                "evaluated_frame_count": evaluated_frame_count,
                "evaluated_labeled_frame_count": evaluated_labeled_frame_count,
                "real_all_frames_metric_count": metric_frame_count,
                "source_frame_coverage": (
                    evaluated_frame_count / max(expected_frame_count, 1)
                ),
                "labeled_metric_coverage": (
                    metric_frame_count / max(evaluated_labeled_frame_count, 1)
                    if "real" in sources
                    else None
                ),
            },
        }
        (output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = [
            "Temporal Clip Transformer Evaluation",
            f"Checkpoint: {checkpoint}",
            f"Architecture: {architecture}",
            f"Split: {args.split}",
            f"Clip length: {clip_length}",
            "Frame coverage: "
            f"{evaluated_frame_count}/{expected_frame_count} source frames; "
            f"{metric_frame_count}/{evaluated_labeled_frame_count} labeled frames "
            "in real/all_frames",
            f"Reset max logit drift: {reset_drift:.3e}",
            "KV cache equivalence max abs: "
            f"{float(kv_equivalence_max_abs.item()):.3e}",
            "",
        ]
        for subset in ("all_frames", "endpoints", "onset", "stable", "release"):
            lines.append(f"[{subset}]")
            for source in sources:
                if subset == "all_frames" and source not in FULL_FRAME_SOURCES:
                    continue
                value = lookup[source][subset]
                lines.append(
                    f"{source:18s} n={value['frame_count']:.0f} "
                    f"RMSE={value['rmse']:.6f} Contact={value['contact_iou']:.6f} "
                    f"V-IoU={value['volumetric_iou']:.6f} "
                    f"CoreLoc={value['core_distribution_viou']:.6f} "
                    f"FalseHigh={value['false_high_excess_fraction']:.6f}"
                )
                if hasattr(model, "contact_head"):
                    contact_value = contact_lookup[source][subset]
                    lines.append(
                        f"{'':18s} ContactHead "
                        f"AP={contact_value['contact_head_ap']:.6f} "
                        f"P={contact_value['contact_head_precision']:.6f} "
                        f"R={contact_value['contact_head_recall']:.6f} "
                        f"F1={contact_value['contact_head_f1']:.6f} "
                        f"ECE={contact_value['contact_head_ece']:.6f}"
                    )
            lines.append("")
        (output_dir / "eval_temporal_clip.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        source_metrics = Path(args.copy_val_metrics_from).expanduser()
        if args.copy_val_metrics_from and source_metrics.is_file():
            shutil.copy2(source_metrics, output_dir / "training_val_metrics.csv")
        print(f"Temporal clip report: {output_dir}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
