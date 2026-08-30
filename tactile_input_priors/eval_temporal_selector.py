#!/usr/bin/env python3
"""Evaluate Temporal Selector V2 without applying any pressure correction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal
from tactile_input_priors.eval_temporal_flow import ExactRankSampler
from tactile_input_priors.runtime import file_sha256, load_torch_checkpoint
from tactile_input_priors.temporal_flow import (
    TEMPORAL_SELECTOR_FORMAT,
    PartitionedPalmCache,
    TemporalActionSelectorV2,
    TemporalReplayDataset,
    build_prediction_control_bins,
    build_temporal_pair_index,
    history_quality_context,
    temporal_action_targets,
    temporal_manifest_key,
)
from tactile_input_priors.temporal_selector_metrics import (
    ACTION_NAMES,
    BinaryScoreMetricAccumulator,
    SequenceBinaryScoreMetricAccumulator,
    TemporalActionMetricAccumulator,
)


BASE_SOURCES = ("real", "cross_sequence", "contralateral", "reset")
DINO_ISOLATED_SOURCES = (
    "dino_gate_zero",
    "dino_zero_motion",
    "dino_cross_history",
)
SUBSETS = ("full_split", "available", "matched")
DINO_DIAGNOSTICS = (
    "dino_effective_gate",
    "dino_residual_rms",
    "dino_valid_token_fraction",
    "dino_motion_rms",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--query-manifests", required=True)
    parser.add_argument("--pair-index-root", default=os.environ.get(
        "TEMPORAL_PAIR_ROOT", "/home/ma-user/work/cfzhao/input_prior_full/cache/temporal_pairs"
    ))
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--max-open-shards", type=int, default=4)
    parser.add_argument("--metric-bins", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--copy-val-metrics-from", default="")
    parser.add_argument(
        "--dino-isolated-controls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For DINO-history checkpoints, separately ablate the whole DINO "
            "branch, historical feature motion, and historical DINO content."
        ),
    )
    return parser


def _flatten_summary(source: str, subset: str, summary: dict) -> dict:
    row = {
        "source": source,
        "subset": subset,
        "sample_count": summary["sample_count"],
        "loss": summary["loss"],
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "macro_average_precision": summary["macro_average_precision"],
        "macro_auroc": summary["macro_auroc"],
        "ece": summary["ece"],
    }
    for name in ACTION_NAMES:
        for metric, value in summary["per_class"][name].items():
            row[f"{name}_{metric}"] = value
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _strict_clear_bootstrap_rows(
    accumulators: dict[str, SequenceBinaryScoreMetricAccumulator],
    sources: tuple[str, ...],
    *,
    iterations: int,
    seed: int,
) -> list[dict]:
    if iterations < 1:
        return []
    real_ap, real_eligible = accumulators["real"].sequence_average_precision()
    rows = []
    for source_index, source in enumerate(sources):
        if source == "real":
            continue
        control_ap, control_eligible = accumulators[source].sequence_average_precision()
        eligible = (real_eligible & control_eligible).numpy()
        sequence_indices = np.flatnonzero(eligible)
        if not len(sequence_indices):
            continue
        real_values = real_ap.numpy()[sequence_indices]
        control_values = control_ap.numpy()[sequence_indices]
        rng = np.random.default_rng(int(seed) + 1009 * source_index)
        deltas = []
        for _ in range(iterations):
            draw = rng.integers(0, len(sequence_indices), size=len(sequence_indices))
            deltas.append(float((real_values[draw] - control_values[draw]).mean()))
        low, high = _interval(deltas)
        estimate = float((real_values - control_values).mean())
        rows.append(
            {
                "control": source,
                "metric": "sequence_macro_average_precision",
                "real_minus_control": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "probability_gt_zero": float(np.mean(np.asarray(deltas) > 0.0)),
                "sequence_count": len(sequence_indices),
            }
        )
        for coverage in (0.001, 0.01):
            real_parts = accumulators["real"].budget_components(coverage)
            control_parts = accumulators[source].budget_components(coverage)
            count = accumulators["real"].sequence_count
            rng = np.random.default_rng(
                int(seed) + 1009 * source_index + int(coverage * 1_000_000)
            )
            metric_values = {"precision": [], "recall": []}
            for _ in range(iterations):
                draw = rng.integers(0, count, size=count)
                draw_tensor = torch.from_numpy(draw)
                for name, numerator_index, denominator in (
                    ("precision", 0, lambda parts: parts[0] + parts[1]),
                    ("recall", 0, lambda parts: parts[2]),
                ):
                    real_denominator = denominator(real_parts)[draw_tensor].sum().clamp_min(1.0)
                    control_denominator = denominator(control_parts)[draw_tensor].sum().clamp_min(1.0)
                    real_score = (
                        real_parts[numerator_index][draw_tensor].sum()
                        / real_denominator
                    )
                    control_score = (
                        control_parts[numerator_index][draw_tensor].sum()
                        / control_denominator
                    )
                    metric_values[name].append(float(real_score - control_score))
            for name, numerator_index, denominator in (
                ("precision", 0, lambda parts: parts[0] + parts[1]),
                ("recall", 0, lambda parts: parts[2]),
            ):
                values = metric_values[name]
                real_score = float(
                    real_parts[numerator_index].sum()
                    / denominator(real_parts).sum().clamp_min(1.0)
                )
                control_score = float(
                    control_parts[numerator_index].sum()
                    / denominator(control_parts).sum().clamp_min(1.0)
                )
                low, high = _interval(values)
                rows.append(
                    {
                        "control": source,
                        "metric": f"{name}_at_{coverage:g}",
                        "real_minus_control": real_score - control_score,
                        "ci95_low": low,
                        "ci95_high": high,
                        "probability_gt_zero": float(
                            np.mean(np.asarray(values) > 0.0)
                        ),
                        "sequence_count": count,
                    }
                )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-rank temporal selector evaluation requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    payload = load_torch_checkpoint(args.checkpoint)
    if payload.get("format") != TEMPORAL_SELECTOR_FORMAT:
        raise ValueError(
            f"Unsupported temporal selector checkpoint: {payload.get('format')!r}"
        )
    model = TemporalActionSelectorV2(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    cache = PartitionedPalmCache(
        args.cache,
        max_open_shards=args.max_open_shards,
        optional_fields=("z_rgb",) if model.uses_dino_history else (),
    )
    if cache.base_checkpoint_sha256 != str(payload.get("base_checkpoint_sha256") or ""):
        raise RuntimeError("Evaluation cache was built from a different RGB baseline")
    manifests = tuple(
        str(Path(value).expanduser().resolve(strict=True))
        for value in args.query_manifests.split(",")
        if value.strip()
    )
    key = temporal_manifest_key(manifests)
    pair_path = (
        Path(args.pair_index_root).expanduser().resolve(strict=False)
        / f"{args.split}-{cache.config_sha256[:12]}-{key}.npz"
    )
    if rank == 0:
        build_temporal_pair_index(cache, manifests, pair_path, seed=args.seed)
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        build_temporal_pair_index(cache, manifests, pair_path, seed=args.seed)
    control_sidecar = pair_path.with_name(f"{pair_path.stem}-rgbmax-control.npz")
    if rank == 0:
        build_prediction_control_bins(cache, pair_path, control_sidecar)
    if world_size > 1:
        dist.barrier()
    if not control_sidecar.is_file():
        raise FileNotFoundError(
            f"RGB prediction control sidecar was not materialized: {control_sidecar}"
        )
    with np.load(control_sidecar, allow_pickle=False) as control_payload:
        control_bins = np.asarray(
            control_payload["prediction_pressure_bin"], dtype=np.int64
        )
    model.to(device).eval()
    dataset = TemporalReplayDataset(
        args.cache,
        pair_path,
        include_control=True,
        include_dino_grid=model.uses_dino_history,
        history_lags=model.history_lags,
        max_open_shards=args.max_open_shards,
        control_pressure_bins=control_bins,
    )
    exact_control_by_cache = torch.zeros(len(dataset), dtype=torch.bool)
    pair_current_indices = np.asarray(dataset.arrays["current_index"], dtype=np.int64)
    control_pairs = np.asarray(dataset.control_pair_indices, dtype=np.int64)
    exact_control_by_cache[torch.from_numpy(pair_current_indices)] = torch.from_numpy(
        control_bins == control_bins[control_pairs]
    )
    exact_control_by_cache = exact_control_by_cache.to(device=device)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "sampler": ExactRankSampler(len(dataset), rank, world_size),
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    if args.num_workers:
        loader_kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=False)
    loader = DataLoader(dataset, **loader_kwargs)
    alternate_alignment = (
        "unwarped" if model.dino_alignment_mode == "aligned" else "aligned"
    )
    dino_sources = (
        (alternate_alignment, "spatial_shuffle")
        + (DINO_ISOLATED_SOURCES if args.dino_isolated_controls else ())
        if model.uses_dino_history
        else ()
    )
    sources = BASE_SOURCES + dino_sources
    accumulators = {
        subset: {
            source: TemporalActionMetricAccumulator(bins=args.metric_bins)
            for source in sources
        }
        for subset in SUBSETS
    }
    strict_clear = {
        source: BinaryScoreMetricAccumulator(bins=args.metric_bins)
        for source in sources
    }
    sequence_strict_clear = {
        source: SequenceBinaryScoreMetricAccumulator(
            dataset.sequence_count, bins=min(args.metric_bins, 512)
        )
        for source in sources
    }
    dino_diagnostic_totals = torch.zeros(
        len(sources), len(DINO_DIAGNOSTICS) + 1,
        device=device,
        dtype=torch.float64,
    )
    # sum(action_delta^2), action_count, flips, anchor_count,
    # sum(residual_delta^2), residual_count, cosine_sum, cosine_count
    dino_paired_totals = torch.zeros(
        len(sources), 8, device=device, dtype=torch.float64
    )
    processed = 0
    started = time.time()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            batch = {
                name: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for name, value in raw_batch.items()
            }
            real_available = batch["history_available"].float()
            cross_available = batch["control_history_available"].float()
            contra_available = batch["contralateral_history_available"].float()
            reset_available = torch.zeros_like(real_available)
            source_inputs = {
                "real": (
                    batch["history_logits"], real_available, "",
                    model.dino_alignment_mode, "full",
                ),
                "cross_sequence": (
                    batch["control_history_logits"], cross_available, "control_",
                    model.dino_alignment_mode, "full",
                ),
                "contralateral": (
                    batch["contralateral_history_logits"], contra_available, "",
                    model.dino_alignment_mode, "full",
                ),
                "reset": (
                    batch["current_logits"][:, None].expand_as(batch["history_logits"]),
                    reset_available,
                    "",
                    model.dino_alignment_mode, "full",
                ),
            }
            if model.uses_dino_history:
                source_inputs.update(
                    {
                        alternate_alignment: (
                            batch["history_logits"], real_available, "",
                            alternate_alignment, "full",
                        ),
                        "spatial_shuffle": (
                            batch["history_logits"], real_available, "",
                            "spatial_shuffle", "full",
                        ),
                    }
                )
                if args.dino_isolated_controls:
                    source_inputs.update(
                        {
                            "dino_gate_zero": (
                                batch["history_logits"], real_available, "",
                                model.dino_alignment_mode, "gate_zero",
                            ),
                            "dino_zero_motion": (
                                batch["history_logits"], real_available, "",
                                model.dino_alignment_mode, "zero_motion",
                            ),
                            "dino_cross_history": (
                                batch["history_logits"], real_available, "",
                                model.dino_alignment_mode, "full",
                            ),
                        }
                    )
            if tuple(source_inputs) != sources:
                raise RuntimeError(
                    "Temporal selector source construction drifted from metric setup: "
                    f"inputs={tuple(source_inputs)}, metrics={sources}"
                )
            has_tactile = batch["has_tactile"] > 0.5
            source_available = {
                "real": real_available.bool().all(dim=1),
                "cross_sequence": cross_available.bool().all(dim=1),
                "contralateral": contra_available.bool().all(dim=1),
                "reset": torch.ones_like(has_tactile),
            }
            if model.uses_dino_history:
                source_available[alternate_alignment] = source_available["real"]
                source_available["spatial_shuffle"] = source_available["real"]
                if args.dino_isolated_controls:
                    source_available["dino_gate_zero"] = source_available["real"]
                    source_available["dino_zero_motion"] = source_available["real"]
                    source_available["dino_cross_history"] = (
                        source_available["real"] & source_available["cross_sequence"]
                    )
            matched = (
                source_available["real"]
                & source_available["cross_sequence"]
                & source_available["contralateral"]
                & exact_control_by_cache[batch["current_index"].long()]
            )
            outputs = {}
            real_action_logits = None
            real_dino_residual = None
            for source, (
                history,
                available,
                prefix,
                dino_control,
                dino_residual_mode,
            ) in source_inputs.items():
                quality = None
                if model.use_per_lag_quality:
                    quality = history_quality_context(
                        batch,
                        model.history_lags,
                        prefix=prefix,
                        availability=available,
                        nominal_fps=model.nominal_fps,
                    )
                dino_kwargs = {}
                if model.uses_dino_history:
                    if source == "cross_sequence":
                        evidence_current = batch["control_current_grid"]
                        evidence_history = batch["control_history_grids"]
                        evidence_transform = batch["control_history_crop_transform"]
                    elif source == "contralateral":
                        evidence_current = batch["current_grid"]
                        evidence_history = batch["contralateral_history_grids"]
                        evidence_transform = batch[
                            "contralateral_history_crop_transform"
                        ]
                    elif source == "reset":
                        evidence_current = batch["current_grid"]
                        evidence_history = batch["current_grid"][:, None].expand_as(
                            batch["history_grids"]
                        )
                        evidence_transform = torch.eye(
                            3, device=device, dtype=torch.float32
                        )[None, None].expand(
                            batch["current_grid"].shape[0],
                            len(model.history_lags),
                            -1,
                            -1,
                        )
                    elif source == "dino_cross_history":
                        evidence_current = batch["current_grid"]
                        evidence_history = batch["control_history_grids"]
                        # Keep real geometry, lag availability, and pressure
                        # history fixed. Only historical DINO token content is
                        # replaced by the RGB-bin-matched wrong sequence.
                        evidence_transform = batch["history_crop_transform"]
                    else:
                        evidence_current = batch["current_grid"]
                        evidence_history = batch["history_grids"]
                        evidence_transform = batch["history_crop_transform"]
                    dino_kwargs = {
                        "current_grid": batch["current_grid"],
                        "history_grids": evidence_history,
                        "history_crop_transform": evidence_transform,
                        "evidence_current_grid": evidence_current,
                        "dino_control": dino_control,
                        "dino_residual_mode": dino_residual_mode,
                        "return_dino_residual": True,
                    }
                output = model(
                    batch["current_logits"],
                    history,
                    available,
                    quality,
                    apply_prior_correction=True,
                    **dino_kwargs,
                )
                outputs[source] = output
                if model.uses_dino_history:
                    source_index = sources.index(source)
                    batch_size = int(batch["current_logits"].shape[0])
                    for diagnostic_index, name in enumerate(DINO_DIAGNOSTICS):
                        dino_diagnostic_totals[source_index, diagnostic_index] += (
                            outputs[source][name].detach().double() * batch_size
                        )
                    dino_diagnostic_totals[source_index, -1] += batch_size
                    residual = output.pop("dino_hidden_residual").float()
                    if source == "real":
                        real_action_logits = output["action_logits"].float()
                        real_dino_residual = residual
                    if real_action_logits is None or real_dino_residual is None:
                        raise RuntimeError("Real DINO output must be evaluated first")
                    paired_valid = has_tactile & matched & source_available[source]
                    if bool(paired_valid.any()):
                        action_delta = output["action_logits"].float() - real_action_logits
                        action_mask = paired_valid[:, None, None]
                        flip = (
                            output["action_logits"].argmax(dim=-1)
                            != real_action_logits.argmax(dim=-1)
                        )
                        residual_delta = residual - real_dino_residual
                        residual_cosine = F.cosine_similarity(
                            residual,
                            real_dino_residual,
                            dim=-1,
                            eps=1e-8,
                        )
                        valid_count = paired_valid.double().sum()
                        dino_paired_totals[source_index, 0] += (
                            action_delta.square() * action_mask
                        ).double().sum()
                        dino_paired_totals[source_index, 1] += (
                            valid_count * action_delta.shape[1] * action_delta.shape[2]
                        )
                        dino_paired_totals[source_index, 2] += (
                            flip & paired_valid[:, None]
                        ).double().sum()
                        dino_paired_totals[source_index, 3] += (
                            valid_count * flip.shape[1]
                        )
                        dino_paired_totals[source_index, 4] += (
                            residual_delta.square() * action_mask
                        ).double().sum()
                        dino_paired_totals[source_index, 5] += (
                            valid_count
                            * residual_delta.shape[1]
                            * residual_delta.shape[2]
                        )
                        dino_paired_totals[source_index, 6] += (
                            residual_cosine * paired_valid[:, None]
                        ).double().sum()
                        dino_paired_totals[source_index, 7] += (
                            valid_count * residual_cosine.shape[1]
                        )
            target = temporal_action_targets(
                batch["current_logits"],
                batch["tactile_signal"],
                model.anchor_local_indices,
                margin=float(payload["action_margin"]),
            )
            for source, output in outputs.items():
                loss_element = F.cross_entropy(
                    output["action_logits"].reshape(-1, 3),
                    target.reshape(-1),
                    reduction="none",
                ).reshape_as(target)
                valid_masks = {
                    "full_split": has_tactile,
                    "available": has_tactile & source_available[source],
                    "matched": has_tactile & matched,
                }
                for subset, valid in valid_masks.items():
                    anchor_valid = valid[:, None].expand_as(target)
                    subset_loss = (
                        loss_element[anchor_valid].mean()
                        if bool(anchor_valid.any())
                        else loss_element.sum() * 0.0
                    )
                    accumulators[subset][source].update(
                        output["action_probability"], target, valid, loss=subset_loss
                    )
                anchor = output["anchor_local_indices"]
                prediction = torch.sigmoid(batch["current_logits"].float())[:, anchor]
                tactile = batch["tactile_signal"].float()[:, anchor]
                candidate = prediction >= 0.10
                positive = candidate & (tactile <= 0.02)
                protected = candidate & (tactile >= 0.10)
                strict_valid = (
                    (positive | protected)
                    & has_tactile[:, None]
                    & matched[:, None]
                )
                down_score = output["action_probability"][..., 0]
                strict_clear[source].update(down_score, positive, strict_valid)
                sequence_strict_clear[source].update(
                    down_score,
                    positive,
                    batch["sequence_id"],
                    strict_valid,
                )
            processed += int(batch["current_logits"].shape[0])
            if rank == 0 and (batch_index + 1) % 20 == 0:
                print(
                    f"[temporal-selector:{args.split}] local={processed:,}/"
                    f"{len(loader.sampler):,} rate="
                    f"{processed / max(time.time() - started, 1e-9):,.1f}/s",
                    flush=True,
                )
    for subset in SUBSETS:
        for source in sources:
            accumulators[subset][source].synchronize(device)
    for source in sources:
        strict_clear[source].synchronize(device)
        sequence_strict_clear[source].synchronize(device)
    if world_size > 1:
        dist.all_reduce(dino_diagnostic_totals)
        dist.all_reduce(dino_paired_totals)
    if rank == 0:
        summaries = {
            subset: {
                source: accumulators[subset][source].summary()
                for source in sources
            }
            for subset in SUBSETS
        }
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        metric_rows = [
            _flatten_summary(source, subset, summaries[subset][source])
            for subset in SUBSETS
            for source in sources
        ]
        coverage_rows = []
        pr_rows = []
        calibration_rows = []
        for subset in SUBSETS:
            for source in sources:
                for row in summaries[subset][source]["risk_coverage"]:
                    coverage_rows.append({"source": source, "subset": subset, **row})
                for row in accumulators[subset][source].pr_curve_rows():
                    pr_rows.append({"source": source, "subset": subset, **row})
                for row in accumulators[subset][source].calibration_curve_rows():
                    calibration_rows.append({"source": source, "subset": subset, **row})
        _write_csv(output_dir / "selector_metrics.csv", metric_rows)
        _write_csv(output_dir / "risk_coverage.csv", coverage_rows)
        _write_csv(output_dir / "pr_curves.csv", pr_rows)
        _write_csv(output_dir / "calibration_curves.csv", calibration_rows)
        strict_rows = []
        for source in sources:
            micro = strict_clear[source].summary()
            sequence_ap, sequence_eligible = sequence_strict_clear[
                source
            ].sequence_average_precision()
            row = {
                "source": source,
                "micro_average_precision": micro["average_precision"],
                "micro_auroc": micro["auroc"],
                "positive_fraction": micro["positive_fraction"],
                "sequence_macro_average_precision": float(
                    sequence_ap[sequence_eligible].mean()
                )
                if bool(sequence_eligible.any())
                else float("nan"),
                "eligible_sequence_count": int(sequence_eligible.sum()),
            }
            for coverage in (0.001, 0.01):
                tp, fp, total_positive, threshold, actual, _ = (
                    sequence_strict_clear[source].budget_components(coverage)
                )
                row[f"precision_at_{coverage:g}"] = float(
                    tp.sum() / (tp + fp).sum().clamp_min(1.0)
                )
                row[f"recall_at_{coverage:g}"] = float(
                    tp.sum() / total_positive.sum().clamp_min(1.0)
                )
                row[f"threshold_at_{coverage:g}"] = threshold
                row[f"actual_coverage_at_{coverage:g}"] = actual
            strict_rows.append(row)
        _write_csv(output_dir / "strict_clear_metrics.csv", strict_rows)
        bootstrap_rows = _strict_clear_bootstrap_rows(
            sequence_strict_clear,
            sources,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        _write_csv(output_dir / "strict_clear_paired_bootstrap.csv", bootstrap_rows)
        dino_rows = []
        dino_paired_rows = []
        if model.uses_dino_history:
            for source_index, source in enumerate(sources):
                denominator = dino_diagnostic_totals[source_index, -1].clamp_min(1.0)
                dino_rows.append(
                    {
                        "source": source,
                        **{
                            name: float(
                                dino_diagnostic_totals[source_index, diagnostic_index]
                                / denominator
                            )
                            for diagnostic_index, name in enumerate(DINO_DIAGNOSTICS)
                        },
                    }
                )
                paired = dino_paired_totals[source_index]
                dino_paired_rows.append(
                    {
                        "source": source,
                        "action_logit_delta_rms": float(
                            (paired[0] / paired[1].clamp_min(1.0)).sqrt()
                        ),
                        "action_argmax_flip_fraction": float(
                            paired[2] / paired[3].clamp_min(1.0)
                        ),
                        "dino_residual_delta_rms": float(
                            (paired[4] / paired[5].clamp_min(1.0)).sqrt()
                        ),
                        "dino_residual_cosine": float(
                            paired[6] / paired[7].clamp_min(1.0)
                        ),
                        "paired_anchor_count": int(paired[7]),
                    }
                )
            _write_csv(output_dir / "dino_diagnostics.csv", dino_rows)
            _write_csv(
                output_dir / "dino_paired_diagnostics.csv", dino_paired_rows
            )
        result = {
            "schema": "tactile_temporal_action_selector_eval_v3",
            "split": args.split,
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "cache": str(Path(args.cache).expanduser().resolve()),
            "cache_config_sha256": cache.config_sha256,
            "pair_index": str(pair_path),
            "cross_control_bin_source": "rgb_prediction",
            "cross_control_sidecar": str(control_sidecar),
            "cross_control_exact_match_fraction": float(
                np.mean(control_bins == control_bins[control_pairs])
            ),
            "sample_count": len(dataset),
            "history_lags": list(model.history_lags),
            "use_per_lag_quality": model.use_per_lag_quality,
            "class_prior": model.class_prior.cpu().tolist(),
            "action_margin": float(payload["action_margin"]),
            "pressure_correction_applied": False,
            "uses_dino_history": model.uses_dino_history,
            "dino_alignment_mode": model.dino_alignment_mode,
            "dino_grid_size": list(model.dino_grid_size),
            "dino_input_resolution": list(model.dino_input_resolution),
            "dino_isolated_controls": bool(args.dino_isolated_controls),
            "strict_clear": {row["source"]: row for row in strict_rows},
            "strict_clear_paired_bootstrap": bootstrap_rows,
            "dino_diagnostics": {row["source"]: row for row in dino_rows},
            "dino_paired_diagnostics": {
                row["source"]: row for row in dino_paired_rows
            },
            "subsets": summaries,
            "matched_real_minus_controls": {
                control: {
                    metric: summaries["matched"]["real"][metric]
                    - summaries["matched"][control][metric]
                    for metric in (
                        "macro_average_precision",
                        "macro_auroc",
                        "macro_f1",
                        "accuracy",
                    )
                } | {
                    f"{action}_average_precision":
                    summaries["matched"]["real"]["per_class"][action]["average_precision"]
                    - summaries["matched"][control]["per_class"][action]["average_precision"]
                    for action in ACTION_NAMES
                }
                for control in sources
                if control != "real"
            },
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        matched_summary = summaries["matched"]
        lines = [
            f"Temporal Selector diagnostic: {args.split}",
            f"Samples: {len(dataset):,}",
            f"History lags: {model.history_lags}",
            f"Per-lag quality: {model.use_per_lag_quality}",
            f"Historical DINO: {model.uses_dino_history} "
            f"alignment={model.dino_alignment_mode}",
            f"Class prior down/hold/up: {model.class_prior.cpu().tolist()}",
            "Pressure correction applied: no",
            "Matched real/cross/contralateral subset:",
        ]
        for source in sources:
            values = matched_summary[source]
            lines.append(
                f"  {source}: macro-AP={values['macro_average_precision']:.6f} "
                f"macro-F1={values['macro_f1']:.6f} ECE={values['ece']:.6f} "
                f"down-AP={values['per_class']['down']['average_precision']:.6f} "
                f"up-AP={values['per_class']['up']['average_precision']:.6f}"
            )
            strict = result["strict_clear"][source]
            lines.append(
                f"    strict-clear: micro-AP={strict['micro_average_precision']:.6f} "
                f"sequence-macro-AP={strict['sequence_macro_average_precision']:.6f} "
                f"P@1%={strict['precision_at_0.01']:.6f} "
                f"R@1%={strict['recall_at_0.01']:.6f}"
            )
            if model.uses_dino_history:
                paired = result["dino_paired_diagnostics"][source]
                lines.append(
                    f"    vs-real: action-logit-rms="
                    f"{paired['action_logit_delta_rms']:.6f} "
                    f"argmax-flip={paired['action_argmax_flip_fraction']:.6f} "
                    f"residual-delta-rms="
                    f"{paired['dino_residual_delta_rms']:.6f} "
                    f"residual-cosine={paired['dino_residual_cosine']:.6f}"
                )
        for control, values in result["matched_real_minus_controls"].items():
            lines.append(
                f"  real-minus-{control}: macro-AP="
                f"{values['macro_average_precision']:+.6f} macro-F1="
                f"{values['macro_f1']:+.6f} down-AP="
                f"{values['down_average_precision']:+.6f} up-AP="
                f"{values['up_average_precision']:+.6f}"
            )
        for row in bootstrap_rows:
            if row["metric"] != "sequence_macro_average_precision":
                continue
            lines.append(
                f"  strict-clear real-minus-{row['control']}: sequence-macro-AP="
                f"{row['real_minus_control']:+.6f} "
                f"CI95=[{row['ci95_low']:+.6f},{row['ci95_high']:+.6f}]"
            )
        (output_dir / "eval_selector.txt").write_text("\n".join(lines) + "\n")
        source = Path(args.copy_val_metrics_from).expanduser() if args.copy_val_metrics_from else None
        if source and source.is_file():
            shutil.copy2(source, output_dir / "val_metrics.csv")
        print("\n".join(lines), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
