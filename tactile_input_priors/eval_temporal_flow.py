#!/usr/bin/env python3
"""Evaluate a temporal residual against RGB and strict cross-sequence history."""

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
from typing import Any, Mapping, Sequence

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
    PriorMetricAccumulator,
    metric_contributions,
    summarize_metric_values,
)
from tactile_input_priors.runtime import file_sha256, load_torch_checkpoint
from tactile_input_priors.temporal_flow import (
    TEMPORAL_MODEL_FORMAT,
    PartitionedPalmCache,
    QueryAwareTemporalResidual,
    TemporalReplayDataset,
    build_temporal_pair_index,
    history_quality_context,
    pair_context,
    temporal_manifest_key,
)


class ExactRankSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world_size: int):
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self):
        return max(0, (self.length - self.rank + self.world_size - 1) // self.world_size)


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
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--copy-val-metrics-from", default="")
    parser.add_argument(
        "--confirmatory-suite",
        action="store_true",
        help="Evaluate the frozen Step-3 candidate set and paired sequence bootstrap.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=521)
    return parser


CONFIRMATORY_LAGS = (1, 2, 4, 8)
CONFIRMATORY_CANDIDATES = (
    "rgb_reset",
    "fixed_mean_l1248_probability_alpha0.10",
    "trained_real_l12_scale0.75",
    "trained_cross_sequence_l12_scale0.75",
    "trained_contralateral_l12_scale0.75",
    "trained_real_l124_scale0.50",
    "trained_cross_sequence_l124_scale0.50",
    "trained_contralateral_l124_scale0.50",
)


def _all_available(available: torch.Tensor, columns: Sequence[int]) -> torch.Tensor:
    return (available[:, tuple(columns)] > 0.5).all(dim=1)


def _scaled_temporal_prediction(
    model: QueryAwareTemporalResidual,
    current_logits: torch.Tensor,
    history_logits: torch.Tensor,
    history_available: torch.Tensor,
    context: torch.Tensor,
    history_quality: torch.Tensor | None = None,
    *,
    enabled_lags: Sequence[int],
    replay_lags: Sequence[int],
    scale: float,
) -> torch.Tensor:
    if model.architecture != "signed_additive":
        raise ValueError("The frozen confirmatory suite requires signed_additive")
    replay_columns = {int(lag): index for index, lag in enumerate(replay_lags)}
    try:
        columns = tuple(replay_columns[int(lag)] for lag in model.history_lags)
    except KeyError as error:
        raise ValueError(
            f"Checkpoint lag {error.args[0]} is absent from replay lags {replay_lags}"
        ) from error
    selected_history = history_logits[:, columns]
    selected_available = history_available[:, columns].clone()
    selected_quality = (
        history_quality[:, columns].clone() if history_quality is not None else None
    )
    enabled = {int(value) for value in enabled_lags}
    for column, lag in enumerate(model.history_lags):
        if int(lag) not in enabled:
            selected_available[:, column] = 0.0
    output = model(
        current_logits,
        selected_history,
        context,
        selected_available,
        selected_quality,
    )
    logits = current_logits.float() + float(scale) * output[
        "bounded_logit_delta"
    ].float()
    return torch.sigmoid(logits)


def _metric_matrix(values: np.ndarray) -> dict[str, np.ndarray]:
    index = {name: column for column, name in enumerate(METRIC_FIELDS)}
    frames = np.maximum(values[:, index["frames"]], 1.0)
    points = np.maximum(values[:, index["values"]], 1.0)
    return {
        "mae": values[:, index["abs_sum"]] / points,
        "rmse": np.sqrt(values[:, index["sq_sum"]] / points),
        "contact_iou": values[:, index["contact_iou_sum"]] / frames,
        "volumetric_iou": values[:, index["viou_sum"]] / frames,
        "core_distribution_viou": values[:, index["core_viou_sum"]]
        / np.maximum(values[:, index["core_count"]], 1.0),
        "pred_gt_volume_ratio": values[:, index["pred_volume"]]
        / np.maximum(values[:, index["gt_volume"]], 1e-12),
        "temporal_accuracy_frame": values[:, index["temporal_correct"]] / frames,
        "false_high_excess_fraction": values[:, index["false_high_excess"]]
        / np.maximum(values[:, index["pred_volume"]], 1e-12),
        "catastrophic_over_rate": values[:, index["cat_over"]]
        / np.maximum(values[:, index["cat_over_denom"]], 1.0),
        "catastrophic_under_rate": values[:, index["cat_under"]]
        / np.maximum(values[:, index["cat_under_denom"]], 1.0),
    }


def _sequence_bootstrap_rows(
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
        name: value.detach().double().cpu().numpy()
        for name, value in sequence_values.items()
    }
    active = arrays["rgb_reset"][:, METRIC_FIELDS.index("frames")] > 0
    arrays = {name: values[active] for name, values in arrays.items()}
    sequence_count = int(active.sum())
    if sequence_count == 0:
        return []
    rng = np.random.default_rng(int(seed))
    probabilities = np.full(sequence_count, 1.0 / sequence_count)
    samples: dict[str, dict[str, list[np.ndarray]]] = {
        name: {} for name in arrays
    }
    remaining = int(iterations)
    while remaining:
        chunk = min(128, remaining)
        weights = rng.multinomial(sequence_count, probabilities, size=chunk).astype(
            np.float64, copy=False
        )
        for name, values in arrays.items():
            metrics = _metric_matrix(weights @ values)
            for metric, metric_values in metrics.items():
                samples[name].setdefault(metric, []).append(metric_values)
        remaining -= chunk
    draws = {
        name: {
            metric: np.concatenate(chunks)[:iterations]
            for metric, chunks in metrics.items()
        }
        for name, metrics in samples.items()
    }
    references = {
        "fixed_mean_l1248_probability_alpha0.10": ("rgb_reset",),
        "trained_real_l12_scale0.75": (
            "rgb_reset",
            "trained_cross_sequence_l12_scale0.75",
            "trained_contralateral_l12_scale0.75",
        ),
        "trained_real_l124_scale0.50": (
            "rgb_reset",
            "trained_cross_sequence_l124_scale0.50",
            "trained_contralateral_l124_scale0.50",
        ),
    }
    lower_is_better = {
        "mae",
        "rmse",
        "false_high_excess_fraction",
        "catastrophic_over_rate",
        "catastrophic_under_rate",
    }
    higher_is_better = {
        "contact_iou",
        "volumetric_iou",
        "core_distribution_viou",
        "temporal_accuracy_frame",
    }
    tail = (1.0 - float(confidence)) * 0.5
    rows: list[dict[str, Any]] = []
    totals = {
        name: summarize_metric_values(torch.from_numpy(values.sum(axis=0)))
        for name, values in arrays.items()
    }
    for candidate, candidate_references in references.items():
        for reference in candidate_references:
            for metric, candidate_draws in draws[candidate].items():
                delta = candidate_draws - draws[reference][metric]
                if metric in lower_is_better:
                    probability_better: float | str = float(np.mean(delta < 0.0))
                elif metric in higher_is_better:
                    probability_better = float(np.mean(delta > 0.0))
                else:
                    probability_better = ""
                rows.append(
                    {
                        "candidate": candidate,
                        "reference": reference,
                        "metric": metric,
                        "estimate": totals[candidate][metric],
                        "reference_estimate": totals[reference][metric],
                        "delta_candidate_minus_reference": (
                            totals[candidate][metric] - totals[reference][metric]
                        ),
                        "ci_low": float(np.quantile(delta, tail)),
                        "ci_high": float(np.quantile(delta, 1.0 - tail)),
                        "probability_better": probability_better,
                        "sequence_count": sequence_count,
                        "iterations": int(iterations),
                        "confidence": float(confidence),
                    }
                )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _run_confirmatory_suite(
    args: argparse.Namespace,
    *,
    dataset: TemporalReplayDataset,
    loader: DataLoader,
    model: QueryAwareTemporalResidual,
    device: torch.device,
    rank: int,
    world_size: int,
    pair_path: Path,
) -> None:
    replay_lags = tuple(int(value) for value in dataset.history_lags)
    lag_column = {lag: column for column, lag in enumerate(replay_lags)}
    missing = sorted(set(CONFIRMATORY_LAGS) - set(replay_lags))
    if missing:
        raise ValueError(f"Confirmatory replay is missing lags: {missing}")
    if model.architecture != "signed_additive":
        raise ValueError("Confirmatory suite requires a signed_additive checkpoint")
    if not {1, 2, 4}.issubset(model.history_lags):
        raise ValueError(
            "Confirmatory suite requires a checkpoint trained with lags 1,2,4"
        )

    metadata = {
        "rgb_reset": {
            "source": "reset",
            "required_lags": "",
            "residual_scale": 0.0,
            "fixed_alpha": 0.0,
        },
        "fixed_mean_l1248_probability_alpha0.10": {
            "source": "real",
            "required_lags": "1,2,4,8",
            "residual_scale": 0.0,
            "fixed_alpha": 0.10,
        },
    }
    for source, label in (
        ("real", "real"),
        ("cross_sequence", "cross_sequence"),
        ("contralateral", "contralateral"),
    ):
        metadata[f"trained_{label}_l12_scale0.75"] = {
            "source": source,
            "required_lags": "1,2",
            "residual_scale": 0.75,
            "fixed_alpha": 0.0,
        }
        metadata[f"trained_{label}_l124_scale0.50"] = {
            "source": source,
            "required_lags": "1,2,4",
            "residual_scale": 0.50,
            "fixed_alpha": 0.0,
        }
    if set(metadata) != set(CONFIRMATORY_CANDIDATES):
        raise RuntimeError("Confirmatory candidates drifted from their frozen contract")

    accumulators = {
        subset: {
            name: PriorMetricAccumulator() for name in CONFIRMATORY_CANDIDATES
        }
        for subset in ("full_split", "available", "matched")
    }
    sequence_values = {
        name: torch.zeros(
            (dataset.sequence_count, len(METRIC_FIELDS)), dtype=torch.float64
        )
        for name in CONFIRMATORY_CANDIDATES
    }
    availability_counts = torch.zeros(
        len(CONFIRMATORY_CANDIDATES), device=device, dtype=torch.float64
    )
    matched_count = torch.zeros((), device=device, dtype=torch.float64)
    processed = 0
    started = time.time()

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            transferred = {
                key: value.to(device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            current_logits = transferred["current_logits"].float()
            base_prediction = torch.sigmoid(current_logits)
            target = transferred["tactile_signal"]
            has_tactile = transferred["has_tactile"]
            palm = torch.ones_like(target)
            context = pair_context(transferred)
            real_available = transferred["history_available"]
            cross_available = transferred["control_history_available"]
            contra_available = transferred["contralateral_history_available"]

            real_l1248 = _all_available(
                real_available, tuple(lag_column[lag] for lag in (1, 2, 4, 8))
            )
            cross_l124 = _all_available(
                cross_available, tuple(lag_column[lag] for lag in (1, 2, 4))
            )
            contra_l124 = _all_available(
                contra_available, tuple(lag_column[lag] for lag in (1, 2, 4))
            )
            matched = real_l1248 & cross_l124 & contra_l124
            matched_valid = has_tactile * matched.to(has_tactile.dtype)
            matched_count += matched_valid.double().sum()
            sequence_ids = transferred["sequence_id"].detach().long().cpu()

            def record_candidate(
                name: str,
                prediction: torch.Tensor,
                available: torch.Tensor,
            ) -> None:
                candidate_index = CONFIRMATORY_CANDIDATES.index(name)
                available = available.bool()
                availability_counts[candidate_index] += (
                    has_tactile * available.to(has_tactile.dtype)
                ).double().sum()
                full_prediction = torch.where(
                    available[:, None], prediction, base_prediction
                )
                accumulators["full_split"][name].update(
                    full_prediction, target, palm, has_tactile
                )
                accumulators["available"][name].update(
                    prediction,
                    target,
                    palm,
                    has_tactile * available.to(has_tactile.dtype),
                )
                accumulators["matched"][name].update(
                    prediction, target, palm, matched_valid
                )
                contributions = metric_contributions(
                    prediction, target, palm, matched_valid
                ).cpu()
                sequence_valid = sequence_ids >= 0
                if bool(sequence_valid.any()):
                    sequence_values[name].index_add_(
                        0,
                        sequence_ids[sequence_valid],
                        contributions[sequence_valid],
                    )

            all_rows = torch.ones(
                current_logits.shape[0], device=device, dtype=torch.bool
            )
            record_candidate("rgb_reset", base_prediction, all_rows)

            fixed_columns = tuple(lag_column[lag] for lag in (1, 2, 4, 8))
            fixed_reference_logits = transferred["history_logits"][
                :, fixed_columns
            ].float().mean(dim=1)
            fixed_reference = torch.sigmoid(fixed_reference_logits)
            fixed_prediction = (
                base_prediction + 0.10 * (fixed_reference - base_prediction)
            ).clamp(1e-6, 1.0 - 1e-6)
            record_candidate(
                "fixed_mean_l1248_probability_alpha0.10",
                fixed_prediction,
                real_l1248,
            )

            source_tensors = {
                "real": (
                    transferred["history_logits"],
                    real_available,
                    history_quality_context(
                        transferred, replay_lags, nominal_fps=model.nominal_fps
                    ) if model.use_per_lag_quality else None,
                ),
                "cross_sequence": (
                    transferred["control_history_logits"],
                    cross_available,
                    history_quality_context(
                        transferred,
                        replay_lags,
                        prefix="control_",
                        nominal_fps=model.nominal_fps,
                    ) if model.use_per_lag_quality else None,
                ),
                "contralateral": (
                    transferred["contralateral_history_logits"],
                    contra_available,
                    history_quality_context(
                        transferred,
                        replay_lags,
                        availability=contra_available,
                        nominal_fps=model.nominal_fps,
                    ) if model.use_per_lag_quality else None,
                ),
            }
            for enabled_lags, scale, suffix in (
                ((1, 2), 0.75, "l12_scale0.75"),
                ((1, 2, 4), 0.50, "l124_scale0.50"),
            ):
                columns = tuple(lag_column[lag] for lag in enabled_lags)
                for source, (history, available_matrix, quality) in source_tensors.items():
                    prediction = _scaled_temporal_prediction(
                        model,
                        current_logits,
                        history,
                        available_matrix,
                        context,
                        quality,
                        enabled_lags=enabled_lags,
                        replay_lags=replay_lags,
                        scale=scale,
                    )
                    record_candidate(
                        f"trained_{source}_{suffix}",
                        prediction,
                        _all_available(available_matrix, columns),
                    )

            processed += int(current_logits.shape[0])
            if rank == 0 and (batch_index + 1) % 20 == 0:
                print(
                    f"[temporal-confirm:{args.split}] local={processed:,}/"
                    f"{len(loader.sampler):,} rate="
                    f"{processed / max(time.time() - started, 1e-9):,.1f}/s",
                    flush=True,
                )

    for subset in ("full_split", "available", "matched"):
        for name in CONFIRMATORY_CANDIDATES:
            accumulators[subset][name].synchronize(device)
    if world_size > 1:
        dist.all_reduce(availability_counts)
        dist.all_reduce(matched_count)
    for name in CONFIRMATORY_CANDIDATES:
        values = sequence_values[name].to(device=device, non_blocking=True)
        if world_size > 1:
            dist.all_reduce(values)
        sequence_values[name] = values.cpu()

    if rank != 0:
        return
    rows: list[dict[str, Any]] = []
    for subset in ("full_split", "available", "matched"):
        for candidate_index, name in enumerate(CONFIRMATORY_CANDIDATES):
            row = {
                "candidate": name,
                "subset": subset,
                **metadata[name],
                "available_frame_count": float(availability_counts[candidate_index]),
                "matched_frame_count": float(matched_count),
                **accumulators[subset][name].summary(),
            }
            rows.append(row)
    bootstrap_rows = _sequence_bootstrap_rows(
        sequence_values,
        iterations=args.bootstrap_iterations,
        confidence=args.bootstrap_confidence,
        seed=args.bootstrap_seed,
    )
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "confirmatory_metrics.csv", rows)
    _write_csv(output_dir / "sequence_bootstrap.csv", bootstrap_rows)
    summary = {
        "schema": "tactile_temporal_confirmatory_eval_v1",
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "pair_index": str(pair_path),
        "sample_count": len(dataset),
        "matched_frame_count": int(matched_count),
        "matched_sequence_count": int(
            (
                sequence_values["rgb_reset"][:, METRIC_FIELDS.index("frames")] > 0
            ).sum()
        ),
        "replay_lags": list(replay_lags),
        "checkpoint_lags": list(model.history_lags),
        "candidates": list(CONFIRMATORY_CANDIDATES),
        "subsets": {
            subset: {
                name: accumulators[subset][name].summary()
                for name in CONFIRMATORY_CANDIDATES
            }
            for subset in ("full_split", "available", "matched")
        },
        "bootstrap": {
            "iterations": int(args.bootstrap_iterations),
            "confidence": float(args.bootstrap_confidence),
            "seed": int(args.bootstrap_seed),
            "row_count": len(bootstrap_rows),
        },
        "semantics": (
            "Frozen validation-selected candidates. Full split uses exact RGB "
            "fallback; available is candidate-specific; matched is one common "
            "record set. Bootstrap resamples sequence clusters with paired draws."
        ),
    }
    (output_dir / "confirmatory_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        f"Temporal confirmatory evaluation: {args.split}",
        f"Full split samples: {len(dataset):,}",
        f"Matched frames/sequences: {int(matched_count):,}/"
        f"{summary['matched_sequence_count']:,}",
        "Frozen candidates (matched subset):",
    ]
    for name in CONFIRMATORY_CANDIDATES:
        values = summary["subsets"]["matched"][name]
        lines.append(
            f"  {name}: RMSE={values['rmse']:.6f} "
            f"Contact={values['contact_iou']:.6f} "
            f"V-IoU={values['volumetric_iou']:.6f} "
            f"CoreLoc={values['core_distribution_viou']:.6f} "
            f"FalseHigh={values['false_high_excess_fraction']:.6f}"
        )
    (output_dir / "confirmatory_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    source = (
        Path(args.copy_val_metrics_from).expanduser()
        if args.copy_val_metrics_from
        else None
    )
    if source and source.is_file():
        shutil.copy2(source, output_dir / "val_metrics.csv")
    print("\n".join(lines), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    payload = load_torch_checkpoint(args.checkpoint)
    if payload.get("format") != TEMPORAL_MODEL_FORMAT:
        raise ValueError(f"Unsupported temporal checkpoint format: {payload.get('format')!r}")
    cache = PartitionedPalmCache(args.cache, max_open_shards=args.max_open_shards)
    if cache.base_checkpoint_sha256 != str(payload.get("base_checkpoint_sha256") or ""):
        raise RuntimeError("Evaluation cache was built from a different RGB baseline")
    manifests = tuple(
        str(Path(value).expanduser().resolve(strict=True))
        for value in args.query_manifests.split(",") if value.strip()
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
    checkpoint_lags = tuple(
        int(value)
        for value in payload.get("model_config", {}).get("history_lags", (1,))
    )
    replay_lags = (
        tuple(sorted(set(checkpoint_lags) | set(CONFIRMATORY_LAGS)))
        if args.confirmatory_suite
        else checkpoint_lags
    )
    dataset = TemporalReplayDataset(
        args.cache,
        pair_path,
        include_control=True,
        history_lags=replay_lags,
        max_open_shards=args.max_open_shards,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "sampler": ExactRankSampler(len(dataset), rank, world_size),
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    if args.num_workers:
        loader_kwargs.update(
            prefetch_factor=args.prefetch_factor, persistent_workers=False
        )
    loader = DataLoader(dataset, **loader_kwargs)
    model = QueryAwareTemporalResidual(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    if args.confirmatory_suite:
        _run_confirmatory_suite(
            args,
            dataset=dataset,
            loader=loader,
            model=model,
            device=device,
            rank=rank,
            world_size=world_size,
            pair_path=pair_path,
        )
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    fused = PriorMetricAccumulator()
    base = PriorMetricAccumulator()
    control = PriorMetricAccumulator()
    contralateral = PriorMetricAccumulator()
    eligible_fused = PriorMetricAccumulator()
    eligible_base = PriorMetricAccumulator()
    eligible_control = PriorMetricAccumulator()
    diagnostics = torch.zeros(8, device=device, dtype=torch.float64)
    lag_diagnostics = torch.zeros(
        (len(model.history_lags), 4), device=device, dtype=torch.float64
    )
    processed = 0
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            transferred = {
                key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            context = pair_context(transferred)
            if model.architecture == "signed_additive":
                real = model(
                    transferred["current_logits"],
                    transferred["history_logits"],
                    context,
                    transferred["history_available"],
                    history_quality_context(
                        transferred,
                        model.history_lags,
                        nominal_fps=model.nominal_fps,
                    ) if model.use_per_lag_quality else None,
                )
                wrong = model(
                    transferred["current_logits"],
                    transferred["control_history_logits"],
                    context,
                    transferred["control_history_available"],
                    history_quality_context(
                        transferred,
                        model.history_lags,
                        prefix="control_",
                        nominal_fps=model.nominal_fps,
                    ) if model.use_per_lag_quality else None,
                )
                opposite = model(
                    transferred["current_logits"],
                    transferred["contralateral_history_logits"],
                    context,
                    transferred["contralateral_history_available"],
                    history_quality_context(
                        transferred,
                        model.history_lags,
                        availability=transferred["contralateral_history_available"],
                        nominal_fps=model.nominal_fps,
                    ) if model.use_per_lag_quality else None,
                )
            else:
                real = model(
                    transferred["current_logits"],
                    transferred["previous_logits"],
                    context,
                )
                wrong = model(
                    transferred["current_logits"],
                    transferred["control_previous_logits"],
                    context,
                )
                opposite = model(
                    transferred["current_logits"],
                    transferred["contralateral_previous_logits"],
                    context,
                )
            palm = torch.ones_like(transferred["tactile_signal"])
            common = (
                transferred["tactile_signal"], palm, transferred["has_tactile"]
            )
            eligible_valid = (
                transferred["has_tactile"] * transferred["temporal_eligible"]
            )
            fused.update(real["pred_tactile"], *common)
            base.update(real["base_pred_tactile"], *common)
            control.update(wrong["pred_tactile"], *common)
            eligible_fused.update(
                real["pred_tactile"], transferred["tactile_signal"], palm, eligible_valid
            )
            eligible_base.update(
                real["base_pred_tactile"], transferred["tactile_signal"], palm, eligible_valid
            )
            eligible_control.update(
                wrong["pred_tactile"], transferred["tactile_signal"], palm, eligible_valid
            )
            contralateral.update(
                opposite["pred_tactile"],
                transferred["tactile_signal"],
                palm,
                transferred["has_tactile"]
                * transferred["contralateral_available"],
            )
            batch_size = int(transferred["tactile_signal"].shape[0])
            eligible = transferred["temporal_eligible"].float()
            eligible_count = eligible.sum()
            delta = real["bounded_logit_delta"].float()
            alpha = real["vertex_history_alpha"].float()
            delta_rms = delta.square().mean(dim=1).sqrt()
            delta_abs_max = delta.abs().amax(dim=1)
            saturation = (
                delta.abs() > 0.95 * model.max_logit_delta
            ).float().mean(dim=1)
            alpha_abs = alpha.abs().mean(dim=1)
            alpha_positive = (alpha > 0).float().mean(dim=1)
            if "vertex_history_alpha_per_lag" in real:
                alpha_per_lag = real["vertex_history_alpha_per_lag"].float()
                lag_available = transferred["history_available"].float()
            else:
                alpha_per_lag = alpha[:, None]
                lag_available = transferred["lag1_temporal_eligible"].float()[:, None]
            for column in range(len(model.history_lags)):
                available_column = lag_available[:, column]
                alpha_column = alpha_per_lag[:, column]
                lag_diagnostics[column] += torch.stack(
                    (
                        available_column.sum(),
                        (
                            alpha_column.abs().mean(dim=1) * available_column
                        ).sum(),
                        (
                            (alpha_column > 0).float().mean(dim=1)
                            * available_column
                        ).sum(),
                        (
                            (alpha_column < 0).float().mean(dim=1)
                            * available_column
                        ).sum(),
                    )
                ).to(dtype=torch.float64)
            real_control_gap = (
                real["pred_tactile"] - wrong["pred_tactile"]
            ).abs().mean(dim=1)
            diagnostics += torch.stack(
                (
                    eligible_count,
                    (delta_rms * eligible).sum(),
                    (delta_abs_max * eligible).sum(),
                    (saturation * eligible).sum(),
                    (alpha_abs * eligible).sum(),
                    (alpha_positive * eligible).sum(),
                    transferred["lag1_temporal_eligible"].float().sum(),
                    (real_control_gap * eligible).sum(),
                )
            ).to(dtype=torch.float64)
            processed += batch_size
            if rank == 0 and (batch_index + 1) % 20 == 0:
                print(
                    f"[temporal-eval:{args.split}] local={processed:,}/"
                    f"{len(loader.sampler):,} rate={processed / max(time.time()-started, 1e-9):,.1f}/s",
                    flush=True,
                )
    fused.synchronize(device)
    base.synchronize(device)
    control.synchronize(device)
    contralateral.synchronize(device)
    eligible_fused.synchronize(device)
    eligible_base.synchronize(device)
    eligible_control.synchronize(device)
    if world_size > 1:
        dist.all_reduce(diagnostics)
        dist.all_reduce(lag_diagnostics)
    if rank == 0:
        eligible_count = float(diagnostics[0])
        lag1_count = float(diagnostics[6])
        diagnostic_count = max(eligible_count, 1.0)
        fused_summary = fused.summary()
        base_summary = base.summary()
        control_summary = control.summary()
        eligible_fused_summary = eligible_fused.summary()
        eligible_base_summary = eligible_base.summary()
        eligible_control_summary = eligible_control.summary()
        result = {
            "schema": "tactile_temporal_flow_eval_v1",
            "split": args.split,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "pair_index": str(pair_path),
            "sample_count": len(dataset),
            "pair_count": int(eligible_count),
            "coverage_fraction": eligible_count / max(len(dataset), 1),
            "lag1_pair_count": int(lag1_count),
            "lag1_coverage_fraction": lag1_count / max(len(dataset), 1),
            "semantics": (
                "full-split replay; unavailable/reset lags are masked to exact "
                "RGB identity and every available history stores frozen RGB-base logits"
            ),
            "architecture": model.architecture,
            "history_lags": list(model.history_lags),
            "fused": fused_summary,
            "base": base_summary,
            "cross_sequence_control": control_summary,
            "contralateral_control": contralateral.summary(),
            "eligible_pairs": {
                "fused": eligible_fused_summary,
                "base": eligible_base_summary,
                "cross_sequence_control": eligible_control_summary,
            },
            "gain_vs_base": {
                key: fused_summary[key] - base_summary[key]
                for key in fused_summary
                if key != "frame_count"
            },
            "gain_vs_control": {
                key: fused_summary[key] - control_summary[key]
                for key in fused_summary
                if key != "frame_count"
            },
            "diagnostics": {
                "logit_delta_rms_sample_mean": float(diagnostics[1] / diagnostic_count),
                "logit_delta_abs_max_sample_mean": float(diagnostics[2] / diagnostic_count),
                "residual_saturation_fraction": float(diagnostics[3] / diagnostic_count),
                "history_alpha_abs_mean": float(diagnostics[4] / diagnostic_count),
                "history_alpha_positive_fraction": float(diagnostics[5] / diagnostic_count),
                "effective_global_gate": (
                    float(model.global_rezero_gate.tanh())
                    if hasattr(model, "global_rezero_gate")
                    else None
                ),
                "real_control_output_gap_mae": float(diagnostics[7] / diagnostic_count),
                "per_lag": {
                    f"lag{lag}": {
                        "available_count": int(lag_diagnostics[column, 0]),
                        "coverage_fraction": float(
                            lag_diagnostics[column, 0] / max(len(dataset), 1)
                        ),
                        "alpha_abs_mean": float(
                            lag_diagnostics[column, 1]
                            / lag_diagnostics[column, 0].clamp_min(1.0)
                        ),
                        "alpha_positive_fraction": float(
                            lag_diagnostics[column, 2]
                            / lag_diagnostics[column, 0].clamp_min(1.0)
                        ),
                        "alpha_negative_fraction": float(
                            lag_diagnostics[column, 3]
                            / lag_diagnostics[column, 0].clamp_min(1.0)
                        ),
                    }
                    for column, lag in enumerate(model.history_lags)
                },
            },
        }
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = [
            f"Temporal pair evaluation: {args.split}",
            f"Full split samples: {len(dataset):,}",
            f"Eligible lag 1 pairs: {int(lag1_count):,} "
            f"({lag1_count / max(len(dataset), 1):.2%})",
            f"Eligible lag {','.join(map(str, model.history_lags))} pairs: "
            f"{int(eligible_count):,} "
            f"({eligible_count / max(len(dataset), 1):.2%})",
        ]
        for lag, values in result["diagnostics"]["per_lag"].items():
            lines.append(
                f"{lag} alpha: abs={values['alpha_abs_mean']:.6f} "
                f"positive={values['alpha_positive_fraction']:.2%} "
                f"negative={values['alpha_negative_fraction']:.2%}"
            )
        for name in ("rmse", "contact_iou", "volumetric_iou", "core_distribution_viou", "false_high_excess_fraction"):
            lines.append(
                f"{name}: base={result['base'][name]:.6f} "
                f"fused={result['fused'][name]:.6f} "
                f"control={result['cross_sequence_control'][name]:.6f} "
                f"gain={result['gain_vs_base'][name]:+.6f}"
            )
        lines.append("Eligible-pair conditional metrics:")
        for name in (
            "rmse",
            "contact_iou",
            "volumetric_iou",
            "core_distribution_viou",
            "false_high_excess_fraction",
        ):
            lines.append(
                f"  {name}: base={eligible_base_summary[name]:.6f} "
                f"fused={eligible_fused_summary[name]:.6f} "
                f"control={eligible_control_summary[name]:.6f}"
            )
        (output_dir / "eval_temporal.txt").write_text("\n".join(lines) + "\n")
        source = Path(args.copy_val_metrics_from).expanduser() if args.copy_val_metrics_from else None
        if source and source.is_file():
            shutil.copy2(source, output_dir / "val_metrics.csv")
        print("\n".join(lines), flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
