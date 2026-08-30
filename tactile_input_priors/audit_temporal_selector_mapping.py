#!/usr/bin/env python3
"""Attribute Temporal Selector V2 errors to anchor-to-vertex score mapping."""

from __future__ import annotations

import argparse
import csv
import heapq
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
from torch.utils.data import DataLoader

from hamer_tactile_ft.hamer_tactile import _canonical_mesh_assets
from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal
from tactile_input_priors.eval_temporal_flow import ExactRankSampler
from tactile_input_priors.runtime import file_sha256, load_torch_checkpoint
from tactile_input_priors.selector_pressure_audit import (
    PressurePolicyAccumulator,
    PressurePolicyPairAccumulator,
    policy_grid,
)
from tactile_input_priors.temporal_flow import (
    TEMPORAL_SELECTOR_FORMAT,
    PartitionedPalmCache,
    TemporalActionSelectorV2,
    TemporalReplayDataset,
    build_prediction_control_bins,
    build_temporal_pair_index,
    history_quality_context,
    temporal_manifest_key,
)
from tactile_input_priors.temporal_selector_metrics import (
    BinaryScoreMetricAccumulator,
    SequenceBinaryScoreMetricAccumulator,
)


SOURCES = ("real", "cross_sequence", "contralateral", "reset")
MAPPING_MODES = (
    "rbf4",
    "euclidean_nearest",
    "geodesic_nearest",
    "anchor_only",
)
NATIVE_MAPPING = "anchor_native"
ORACLE_SOURCE = "oracle_anchor_gt"
SCORE_LABELS = (
    "down_action",
    "strict_false_high_all",
    "strict_false_high_clear",
    "formal_false_high",
)
DOWN_ACTION_INDEX = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--cache", default="")
    parser.add_argument("--query-manifests", default="")
    parser.add_argument(
        "--pair-index-root",
        default=os.environ.get(
            "TEMPORAL_PAIR_ROOT",
            "/home/ma-user/work/cfzhao/input_prior_full/cache/temporal_pairs",
        ),
    )
    parser.add_argument("--split", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--mapping-modes",
        default=",".join(MAPPING_MODES),
    )
    parser.add_argument(
        "--cross-control-bin-source",
        choices=("rgb_prediction", "target_gt"),
        default="rgb_prediction",
    )
    parser.add_argument("--metric-bins", type=int, default=4096)
    parser.add_argument("--budget-coverages", default="0.00001,0.0001,0.001,0.01")
    parser.add_argument("--bootstrap-bins", type=int, default=256)
    parser.add_argument("--bootstrap-coverages", default="0.001,0.01")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=521)
    parser.add_argument("--policy-score-thresholds", default="0.30,0.55,0.80")
    parser.add_argument("--policy-alphas", default="0.05,0.50")
    parser.add_argument("--policy-target-floor", type=float, default=0.02)
    parser.add_argument("--policy-action-threshold", type=float, default=0.10)
    parser.add_argument("--policy-no-contact-max", type=float, default=0.02)
    parser.add_argument("--policy-subthreshold-max", type=float, default=0.08)
    parser.add_argument("--policy-contact-min", type=float, default=0.10)
    parser.add_argument("--policy-chunk-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--max-open-shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--copy-val-metrics-from", default="")
    parser.add_argument("--tiny-check", action="store_true")
    return parser


def _float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated floating-point value")
    return values


def _string_csv(raw: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _probability_logit(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp(1e-6, 1.0 - 1e-6)
    return torch.log(probability) - torch.log1p(-probability)


def _paired_sequence_score_bootstrap_rows(
    accumulators: Mapping[
        tuple[str, str, str], SequenceBinaryScoreMetricAccumulator
    ],
    *,
    coverages: Sequence[float],
    iterations: int,
    confidence: float,
    seed: int,
) -> list[dict[str, Any]]:
    if int(iterations) < 0:
        raise ValueError("bootstrap iterations cannot be negative")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("bootstrap confidence must lie in (0,1)")
    pairs = sorted(
        (mapping, label)
        for source, mapping, label in accumulators
        if source == "real"
    )
    tail = (1.0 - float(confidence)) * 0.5
    rows: list[dict[str, Any]] = []
    for pair_index, (mapping, label) in enumerate(pairs):
        reference = accumulators[("real", mapping, label)]
        control = accumulators[("cross_sequence", mapping, label)]
        if not torch.equal(
            reference.histogram.sum(dim=2), control.histogram.sum(dim=2)
        ):
            raise RuntimeError(
                f"Paired sequence labels differ for mapping={mapping}, label={label}"
            )
        reference_ap, reference_eligible = reference.sequence_average_precision()
        control_ap, control_eligible = control.sequence_average_precision()
        eligible = reference_eligible & control_eligible
        active = torch.nonzero(eligible, as_tuple=False).flatten()
        if not active.numel():
            continue
        reference_ap_np = reference_ap[active].numpy()
        control_ap_np = control_ap[active].numpy()
        ap_delta = reference_ap_np - control_ap_np
        sequence_count = len(ap_delta)
        rng = np.random.default_rng(int(seed) + 1009 * pair_index)

        bootstrap_ap: list[np.ndarray] = []
        budget_payloads: list[dict[str, Any]] = []
        for coverage_index, coverage in enumerate(coverages):
            reference_components = reference.budget_components(float(coverage))
            control_components = control.budget_components(float(coverage))
            all_components = np.stack(
                [
                    reference_components[0].numpy(),
                    reference_components[1].numpy(),
                    reference_components[2].numpy(),
                    control_components[0].numpy(),
                    control_components[1].numpy(),
                    control_components[2].numpy(),
                ],
                axis=1,
            )
            budget_active = np.nonzero(
                (reference_components[5].numpy() > 0.0)
                | (control_components[5].numpy() > 0.0)
            )[0]
            component_matrix = all_components[budget_active]
            budget_payloads.append(
                {
                    "coverage": float(coverage),
                    "reference_threshold": float(reference_components[3]),
                    "reference_actual_coverage": float(reference_components[4]),
                    "control_threshold": float(control_components[3]),
                    "control_actual_coverage": float(control_components[4]),
                    "components": component_matrix,
                    "sequence_count": len(component_matrix),
                    "seed": int(seed) + 1009 * pair_index + 9176 * (coverage_index + 1),
                    "precision_draws": [],
                    "recall_draws": [],
                }
            )

        remaining = int(iterations)
        probabilities = np.full(sequence_count, 1.0 / sequence_count)
        while remaining > 0:
            chunk = min(128, remaining)
            weights = rng.multinomial(
                sequence_count, probabilities, size=chunk
            ).astype(np.float64, copy=False)
            bootstrap_ap.append(weights @ ap_delta / sequence_count)
            remaining -= chunk

        if bootstrap_ap:
            ap_draws = np.concatenate(bootstrap_ap)[: int(iterations)]
            ap_low, ap_high = np.quantile(ap_draws, (tail, 1.0 - tail))
            ap_better = float(np.mean(ap_draws > 0.0))
        else:
            ap_low = ap_high = float(ap_delta.mean())
            ap_better = float(np.mean(ap_delta > 0.0))
        rows.append(
            {
                "mapping": mapping,
                "label": label,
                "metric": "sequence_macro_average_precision",
                "requested_coverage": "",
                "paired_sequence_count": sequence_count,
                "reference_value": float(reference_ap_np.mean()),
                "control_value": float(control_ap_np.mean()),
                "reference_minus_control": float(ap_delta.mean()),
                "ci_low": float(ap_low),
                "ci_high": float(ap_high),
                "reference_better_probability": ap_better,
                "reference_micro_value": reference.micro_average_precision(),
                "control_micro_value": control.micro_average_precision(),
                "reference_threshold": "",
                "control_threshold": "",
                "reference_actual_coverage": "",
                "control_actual_coverage": "",
            }
        )
        for payload in budget_payloads:
            budget_sequence_count = int(payload["sequence_count"])
            if int(iterations) > 0 and budget_sequence_count > 0:
                budget_rng = np.random.default_rng(payload["seed"])
                probabilities = np.full(
                    budget_sequence_count, 1.0 / budget_sequence_count
                )
                remaining = int(iterations)
                while remaining > 0:
                    chunk = min(128, remaining)
                    weights = budget_rng.multinomial(
                        budget_sequence_count, probabilities, size=chunk
                    ).astype(np.float64, copy=False)
                    totals = weights @ payload["components"]
                    reference_precision_draw = totals[:, 0] / np.maximum(
                        totals[:, 0] + totals[:, 1], 1.0
                    )
                    control_precision_draw = totals[:, 3] / np.maximum(
                        totals[:, 3] + totals[:, 4], 1.0
                    )
                    reference_recall_draw = totals[:, 0] / np.maximum(
                        totals[:, 2], 1.0
                    )
                    control_recall_draw = totals[:, 3] / np.maximum(
                        totals[:, 5], 1.0
                    )
                    payload["precision_draws"].append(
                        reference_precision_draw - control_precision_draw
                    )
                    payload["recall_draws"].append(
                        reference_recall_draw - control_recall_draw
                    )
                    remaining -= chunk
            components = payload["components"].sum(axis=0)
            reference_precision = components[0] / max(
                components[0] + components[1], 1.0
            )
            control_precision = components[3] / max(
                components[3] + components[4], 1.0
            )
            reference_recall = components[0] / max(components[2], 1.0)
            control_recall = components[3] / max(components[5], 1.0)
            for metric, reference_value, control_value, draw_key in (
                (
                    "equal_budget_precision",
                    reference_precision,
                    control_precision,
                    "precision_draws",
                ),
                (
                    "equal_budget_recall",
                    reference_recall,
                    control_recall,
                    "recall_draws",
                ),
            ):
                if payload[draw_key]:
                    draws = np.concatenate(payload[draw_key])[: int(iterations)]
                    low, high = np.quantile(draws, (tail, 1.0 - tail))
                    better = float(np.mean(draws > 0.0))
                else:
                    low = high = float(reference_value - control_value)
                    better = float(reference_value > control_value)
                rows.append(
                    {
                        "mapping": mapping,
                        "label": label,
                        "metric": metric,
                        "requested_coverage": payload["coverage"],
                        "paired_sequence_count": budget_sequence_count,
                        "reference_value": float(reference_value),
                        "control_value": float(control_value),
                        "reference_minus_control": float(
                            reference_value - control_value
                        ),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "reference_better_probability": better,
                        "reference_micro_value": "",
                        "control_micro_value": "",
                        "reference_threshold": payload["reference_threshold"],
                        "control_threshold": payload["control_threshold"],
                        "reference_actual_coverage": payload[
                            "reference_actual_coverage"
                        ],
                        "control_actual_coverage": payload[
                            "control_actual_coverage"
                        ],
                    }
                )
    return rows


class VertexScoreProjector:
    """Deterministic controls for the 512-anchor to palm-vertex mapping."""

    def __init__(self, model: TemporalActionSelectorV2):
        self.model = model
        self.euclidean_owner = (
            model.vertex_anchor_indices[:, 0].detach().cpu().long()
        )
        self.geodesic_owner = self._build_geodesic_owner()

    def _build_geodesic_owner(self) -> torch.Tensor:
        vertices, valid_mask = _canonical_mesh_assets()
        palm_full = self.model.palm_vertex_indices.detach().cpu().long()
        if not torch.equal(torch.nonzero(valid_mask).flatten(), palm_full):
            raise RuntimeError("Selector and canonical palm vertex definitions differ")
        full_to_local = torch.full((len(vertices),), -1, dtype=torch.long)
        full_to_local[palm_full] = torch.arange(len(palm_full))
        faces_path = (
            REPO_ROOT
            / "opentouch"
            / "preprocess"
            / "scratch"
            / "auto_calibrated_palm_subdiv_faces.json"
        )
        faces = json.loads(faces_path.read_text(encoding="utf-8"))["group_negative"][
            "face_triplets"
        ]
        adjacency: list[dict[int, float]] = [dict() for _ in range(len(palm_full))]
        for face in faces:
            local = [int(full_to_local[int(value)]) for value in face]
            if any(value < 0 for value in local):
                continue
            edges = (
                (local[0], local[1]),
                (local[1], local[2]),
                (local[2], local[0]),
            )
            for first, second in edges:
                distance = float(
                    torch.linalg.vector_norm(
                        vertices[palm_full[first]] - vertices[palm_full[second]]
                    )
                )
                previous = adjacency[first].get(second)
                if previous is None or distance < previous:
                    adjacency[first][second] = distance
                    adjacency[second][first] = distance

        distance = [math.inf] * len(palm_full)
        owner = [-1] * len(palm_full)
        queue: list[tuple[float, int, int]] = []
        for anchor, vertex in enumerate(self.model.anchor_local_indices.detach().cpu().tolist()):
            distance[int(vertex)] = 0.0
            owner[int(vertex)] = int(anchor)
            heapq.heappush(queue, (0.0, int(anchor), int(vertex)))
        while queue:
            current_distance, current_owner, vertex = heapq.heappop(queue)
            if current_distance > distance[vertex] + 1e-12:
                continue
            if current_owner != owner[vertex] and math.isclose(
                current_distance, distance[vertex], abs_tol=1e-12
            ):
                continue
            for neighbor, edge in adjacency[vertex].items():
                candidate = current_distance + edge
                if candidate < distance[neighbor] - 1e-12 or (
                    math.isclose(candidate, distance[neighbor], abs_tol=1e-12)
                    and (owner[neighbor] < 0 or current_owner < owner[neighbor])
                ):
                    distance[neighbor] = candidate
                    owner[neighbor] = current_owner
                    heapq.heappush(queue, (candidate, current_owner, neighbor))
        unresolved = torch.tensor(owner, dtype=torch.long) < 0
        result = torch.tensor(owner, dtype=torch.long)
        result[unresolved] = self.euclidean_owner[unresolved]
        return result

    def project(self, anchor_score: torch.Tensor, mode: str) -> torch.Tensor:
        if anchor_score.ndim != 2 or anchor_score.shape[1] != self.model.anchor_count:
            raise ValueError(
                f"Expected anchor score [B,{self.model.anchor_count}], "
                f"got {tuple(anchor_score.shape)}"
            )
        mode = str(mode)
        if mode == "rbf4":
            indices = self.model.vertex_anchor_indices
            weights = self.model.vertex_anchor_weights.to(anchor_score)
            score = (anchor_score[:, indices] * weights[None]).sum(dim=-1)
        elif mode == "euclidean_nearest":
            score = anchor_score[:, self.euclidean_owner.to(anchor_score.device)]
        elif mode == "geodesic_nearest":
            score = anchor_score[:, self.geodesic_owner.to(anchor_score.device)]
        elif mode == "anchor_only":
            score = anchor_score.new_zeros(
                (anchor_score.shape[0], len(self.model.palm_vertex_indices))
            )
            score[:, self.model.anchor_local_indices] = anchor_score
        else:
            raise ValueError(f"Unsupported vertex score mapping: {mode}")
        return score.clamp(0.0, 1.0)


def _score_masks(
    prediction: torch.Tensor,
    target: torch.Tensor,
    has_tactile: torch.Tensor,
    *,
    action_margin: float,
    action_threshold: float,
    no_contact_max: float,
    contact_min: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    valid = has_tactile[:, None].expand_as(prediction)
    policy_candidate = valid & (prediction >= float(action_threshold))
    strict = policy_candidate & (target <= float(no_contact_max))
    protected = policy_candidate & (target >= float(contact_min))
    formal_candidate = valid & (prediction >= 0.30)
    return {
        "down_action": (
            target < prediction - float(action_margin),
            valid,
        ),
        "strict_false_high_all": (strict, policy_candidate),
        "strict_false_high_clear": (strict, strict | protected),
        "formal_false_high": (
            formal_candidate & (target < 0.005),
            formal_candidate,
        ),
    }


def _tiny_check() -> None:
    metric = BinaryScoreMetricAccumulator(bins=64)
    score = torch.tensor([[0.9, 0.8, 0.2, 0.1]])
    positive = torch.tensor([[True, True, False, False]])
    metric.update(score, positive)
    if metric.summary()["average_precision"] < 0.99:
        raise AssertionError("Binary score accumulator failed a separable ranking")
    if not metric.budget_rows((0.5,)):
        raise AssertionError("Binary score accumulator lost its budget summary")
    sequence = torch.arange(4, dtype=torch.long)
    sequence_positive = positive.expand(4, -1)
    reference_sequence_metric = SequenceBinaryScoreMetricAccumulator(4, bins=64)
    control_sequence_metric = SequenceBinaryScoreMetricAccumulator(4, bins=64)
    reference_sequence_metric.update(
        score.expand(4, -1), sequence_positive, sequence
    )
    control_sequence_metric.update(
        torch.tensor([[0.1, 0.2, 0.8, 0.9]]).expand(4, -1),
        sequence_positive,
        sequence,
    )
    bootstrap_rows = _paired_sequence_score_bootstrap_rows(
        {
            ("real", NATIVE_MAPPING, "down_action"): reference_sequence_metric,
            (
                "cross_sequence",
                NATIVE_MAPPING,
                "down_action",
            ): control_sequence_metric,
        },
        coverages=(0.5,),
        iterations=32,
        confidence=0.95,
        seed=521,
    )
    ap_row = next(
        row
        for row in bootstrap_rows
        if row["metric"] == "sequence_macro_average_precision"
    )
    if not (
        ap_row["reference_minus_control"] > 0.0 and ap_row["ci_low"] > 0.0
    ):
        raise AssertionError("Paired sequence AP bootstrap failed a separable case")
    vertices, valid = _canonical_mesh_assets()
    del vertices
    model = TemporalActionSelectorV2(
        torch.nonzero(valid).flatten(),
        anchor_count=8,
        anchor_neighbors=2,
        graph_neighbors=2,
        hidden_dim=8,
        graph_layers=1,
        history_lags=(1, 2),
        use_per_lag_quality=False,
    )
    projector = VertexScoreProjector(model)
    anchors = torch.ones(2, model.anchor_count)
    for mode in MAPPING_MODES:
        projected = projector.project(anchors, mode)
        if projected.shape != (2, len(model.palm_vertex_indices)):
            raise AssertionError(f"Mapping {mode} returned the wrong shape")
        if mode != "anchor_only" and not torch.allclose(
            projected, torch.ones_like(projected), atol=1e-6, rtol=0.0
        ):
            raise AssertionError(f"Mapping {mode} failed constant preservation")
    print("Temporal selector mapping tiny check passed", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.tiny_check:
        _tiny_check()
        return
    missing = [
        name
        for name in ("checkpoint", "cache", "query_manifests", "split", "output_dir")
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    mapping_modes = _string_csv(args.mapping_modes)
    unknown = sorted(set(mapping_modes) - set(MAPPING_MODES))
    if unknown:
        raise ValueError(f"Unsupported mapping mode(s): {', '.join(unknown)}")
    budget_coverages = _float_csv(args.budget_coverages)
    bootstrap_coverages = _float_csv(args.bootstrap_coverages)
    if args.bootstrap_bins < 32:
        raise ValueError("--bootstrap-bins must be at least 32")
    if args.bootstrap_iterations < 0:
        raise ValueError("--bootstrap-iterations cannot be negative")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("--bootstrap-confidence must lie in (0,1)")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-rank mapping attribution requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    checkpoint_sha256 = file_sha256(args.checkpoint)
    payload = load_torch_checkpoint(args.checkpoint)
    if payload.get("format") != TEMPORAL_SELECTOR_FORMAT:
        raise ValueError(f"Unsupported selector checkpoint: {payload.get('format')!r}")
    cache = PartitionedPalmCache(args.cache, max_open_shards=args.max_open_shards)
    if cache.base_checkpoint_sha256 != str(payload.get("base_checkpoint_sha256") or ""):
        raise RuntimeError("Mapping audit cache was built from a different RGB baseline")

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
    if not pair_path.is_file():
        raise FileNotFoundError(
            f"Rank 0 did not materialize the temporal pair index: {pair_path}"
        )

    control_bins = None
    control_sidecar = None
    if args.cross_control_bin_source == "rgb_prediction":
        control_sidecar = pair_path.with_name(
            f"{pair_path.stem}-rgbmax-control.npz"
        )
        if rank == 0:
            build_prediction_control_bins(cache, pair_path, control_sidecar)
        if world_size > 1:
            dist.barrier()
        if not control_sidecar.is_file():
            raise FileNotFoundError(
                "Rank 0 did not materialize the RGB prediction control sidecar: "
                f"{control_sidecar}"
            )
        with np.load(control_sidecar, allow_pickle=False) as control_payload:
            control_bins = np.asarray(
                control_payload["prediction_pressure_bin"], dtype=np.int64
            )

    model = TemporalActionSelectorV2(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    projector = VertexScoreProjector(model)
    dataset = TemporalReplayDataset(
        args.cache,
        pair_path,
        include_control=True,
        history_lags=model.history_lags,
        max_open_shards=args.max_open_shards,
        control_pressure_bins=control_bins,
    )
    exact_subset = (
        "matched_rgb_exact"
        if args.cross_control_bin_source == "rgb_prediction"
        else "matched_target_exact"
    )
    exact_control_by_cache = torch.zeros(len(dataset), dtype=torch.bool)
    if dataset.control_pair_indices is not None:
        pair_current_indices = np.asarray(
            dataset.arrays["current_index"], dtype=np.int64
        )
        control_pairs = np.asarray(dataset.control_pair_indices, dtype=np.int64)
        bins = np.asarray(dataset.control_pressure_bins, dtype=np.int64)
        exact_control_by_cache[torch.from_numpy(pair_current_indices)] = torch.from_numpy(
            bins == bins[control_pairs]
        )
    exact_control_by_cache = exact_control_by_cache.to(device=device)
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "sampler": ExactRankSampler(len(dataset), rank, world_size),
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    if args.num_workers:
        loader_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=False,
        )
    loader = DataLoader(dataset, **loader_kwargs)

    metric_keys: list[tuple[str, str, str, str]] = []
    for mapping in mapping_modes:
        for subset in ("full_split", "available"):
            for label in SCORE_LABELS:
                metric_keys.append((subset, "real", mapping, label))
                metric_keys.append((subset, ORACLE_SOURCE, mapping, label))
    for subset in ("full_split", "available"):
        for label in SCORE_LABELS:
            metric_keys.append((subset, "real", NATIVE_MAPPING, label))
    for source in SOURCES:
        for mapping in ("rbf4", NATIVE_MAPPING):
            for label in SCORE_LABELS:
                metric_keys.append(("matched", source, mapping, label))
    for source in ("real", "cross_sequence"):
        for mapping in ("rbf4", NATIVE_MAPPING):
            for label in SCORE_LABELS:
                metric_keys.append((exact_subset, source, mapping, label))
    score_metrics = {
        key: BinaryScoreMetricAccumulator(bins=args.metric_bins) for key in metric_keys
    }
    sequence_metric_labels = {
        "rbf4": (
            "strict_false_high_all",
            "strict_false_high_clear",
            "formal_false_high",
        ),
        NATIVE_MAPPING: SCORE_LABELS,
    }
    sequence_score_metrics = {
        (source, mapping, label): SequenceBinaryScoreMetricAccumulator(
            dataset.sequence_count,
            bins=args.bootstrap_bins,
        )
        for source in ("real", "cross_sequence")
        for mapping, labels in sequence_metric_labels.items()
        for label in labels
    }
    policies = policy_grid(
        _float_csv(args.policy_score_thresholds),
        _float_csv(args.policy_alphas),
        (float(args.policy_target_floor),),
    )
    policy_metrics = {
        mapping: PressurePolicyAccumulator(
            policies,
            action_threshold=args.policy_action_threshold,
            no_contact_max=args.policy_no_contact_max,
            subthreshold_max=args.policy_subthreshold_max,
            contact_min=args.policy_contact_min,
            chunk_size=args.policy_chunk_size,
        )
        for mapping in mapping_modes
    }
    policy_pair_metrics = PressurePolicyPairAccumulator(
        policies,
        action_threshold=args.policy_action_threshold,
        no_contact_max=args.policy_no_contact_max,
        contact_min=args.policy_contact_min,
        chunk_size=args.policy_chunk_size,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence=args.bootstrap_confidence,
    )
    subset_frame_counts = torch.zeros(2, dtype=torch.float64, device=device)

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
                "real": (batch["history_logits"], real_available, ""),
                "cross_sequence": (
                    batch["control_history_logits"],
                    cross_available,
                    "control_",
                ),
                "contralateral": (
                    batch["contralateral_history_logits"],
                    contra_available,
                    "",
                ),
                "reset": (
                    batch["current_logits"][:, None].expand_as(batch["history_logits"]),
                    reset_available,
                    "",
                ),
            }
            base_prediction = torch.sigmoid(batch["current_logits"].float())
            target = batch["tactile_signal"].float()
            has_tactile = batch["has_tactile"] > 0.5
            source_available = {
                "real": real_available.bool().all(dim=1),
                "cross_sequence": cross_available.bool().all(dim=1),
                "contralateral": contra_available.bool().all(dim=1),
                "reset": torch.ones_like(has_tactile),
            }
            matched = (
                source_available["real"]
                & source_available["cross_sequence"]
                & source_available["contralateral"]
            )
            exact_matched = (
                source_available["real"]
                & source_available["cross_sequence"]
                & exact_control_by_cache[batch["current_index"].long()]
            )
            subset_frame_counts[0] += (matched & has_tactile).sum().double()
            subset_frame_counts[1] += (exact_matched & has_tactile).sum().double()
            labels = _score_masks(
                base_prediction,
                target,
                has_tactile,
                action_margin=float(payload["action_margin"]),
                action_threshold=args.policy_action_threshold,
                no_contact_max=args.policy_no_contact_max,
                contact_min=args.policy_contact_min,
            )
            anchor_indices = model.anchor_local_indices
            anchor_labels = _score_masks(
                base_prediction[:, anchor_indices],
                target[:, anchor_indices],
                has_tactile,
                action_margin=float(payload["action_margin"]),
                action_threshold=args.policy_action_threshold,
                no_contact_max=args.policy_no_contact_max,
                contact_min=args.policy_contact_min,
            )

            anchor_scores: dict[str, torch.Tensor] = {}
            for source, (history, available, prefix) in source_inputs.items():
                quality = None
                if model.use_per_lag_quality:
                    quality = history_quality_context(
                        batch,
                        model.history_lags,
                        prefix=prefix,
                        availability=available,
                        nominal_fps=model.nominal_fps,
                    )
                output = model(
                    batch["current_logits"],
                    history,
                    available,
                    quality,
                    apply_prior_correction=True,
                )
                anchor_scores[source] = output["action_probability"][
                    ..., DOWN_ACTION_INDEX
                ]

            real_scores = {
                mapping: projector.project(anchor_scores["real"], mapping)
                * source_available["real"][:, None]
                for mapping in mapping_modes
            }
            for mapping, score in real_scores.items():
                for subset, frame_mask in (
                    ("full_split", has_tactile),
                    ("available", has_tactile & source_available["real"]),
                ):
                    for label, (positive, valid) in labels.items():
                        score_metrics[(subset, "real", mapping, label)].update(
                            score,
                            positive,
                            valid & frame_mask[:, None],
                        )
                policy_metrics[mapping].update(
                    base_prediction,
                    torch.zeros_like(base_prediction),
                    target,
                    torch.ones_like(base_prediction),
                    has_tactile,
                    false_high_logits=_probability_logit(score),
                )

            native_real_score = (
                anchor_scores["real"] * source_available["real"][:, None]
            )
            for subset, frame_mask in (
                ("full_split", has_tactile),
                ("available", has_tactile & source_available["real"]),
            ):
                for label, (positive, valid) in anchor_labels.items():
                    score_metrics[(subset, "real", NATIVE_MAPPING, label)].update(
                        native_real_score,
                        positive,
                        valid & frame_mask[:, None],
                    )

            for mapping in mapping_modes:
                for label, (anchor_positive, _) in anchor_labels.items():
                    oracle_score = projector.project(anchor_positive.float(), mapping)
                    positive, valid = labels[label]
                    for subset, frame_mask in (
                        ("full_split", has_tactile),
                        ("available", has_tactile & source_available["real"]),
                    ):
                        score_metrics[
                            (subset, ORACLE_SOURCE, mapping, label)
                        ].update(
                            oracle_score,
                            positive,
                            valid & frame_mask[:, None],
                        )

            source_rbf_scores: dict[str, torch.Tensor] = {}
            for source in SOURCES:
                vertex_score = projector.project(anchor_scores[source], "rbf4")
                native_score = anchor_scores[source]
                if source != "reset":
                    vertex_score = (
                        vertex_score * source_available[source][:, None]
                    )
                    native_score = native_score * source_available[source][:, None]
                source_rbf_scores[source] = vertex_score
                for label, (positive, valid) in labels.items():
                    score_metrics[("matched", source, "rbf4", label)].update(
                        vertex_score,
                        positive,
                        valid & matched[:, None],
                    )
                for label, (positive, valid) in anchor_labels.items():
                    score_metrics[
                        ("matched", source, NATIVE_MAPPING, label)
                    ].update(
                        native_score,
                        positive,
                        valid & matched[:, None],
                    )
                if source in ("real", "cross_sequence"):
                    for label, (positive, valid) in labels.items():
                        score_metrics[
                            (exact_subset, source, "rbf4", label)
                        ].update(
                            vertex_score,
                            positive,
                            valid & exact_matched[:, None],
                        )
                    for label, (positive, valid) in anchor_labels.items():
                        score_metrics[
                            (exact_subset, source, NATIVE_MAPPING, label)
                        ].update(
                            native_score,
                            positive,
                            valid & exact_matched[:, None],
                        )
                    sequence_id = batch["sequence_id"].long()
                    for label in sequence_metric_labels["rbf4"]:
                        positive, valid = labels[label]
                        sequence_score_metrics[(source, "rbf4", label)].update(
                            vertex_score,
                            positive,
                            sequence_id,
                            valid & exact_matched[:, None],
                        )
                    for label in sequence_metric_labels[NATIVE_MAPPING]:
                        positive, valid = anchor_labels[label]
                        sequence_score_metrics[
                            (source, NATIVE_MAPPING, label)
                        ].update(
                            native_score,
                            positive,
                            sequence_id,
                            valid & exact_matched[:, None],
                        )

            exact_has_tactile = has_tactile & exact_matched
            if bool(exact_has_tactile.any()):
                sequence_keys = [
                    f"{args.split}:{int(value)}"
                    for value in batch["sequence_id"].detach().cpu().tolist()
                ]
                zeros = torch.zeros_like(base_prediction)
                policy_pair_metrics.update(
                    base_prediction,
                    zeros,
                    zeros,
                    target,
                    torch.ones_like(base_prediction),
                    exact_has_tactile,
                    sequence_keys=sequence_keys,
                    reference_false_high_logits=_probability_logit(
                        source_rbf_scores["real"]
                    ),
                    control_false_high_logits=_probability_logit(
                        source_rbf_scores["cross_sequence"]
                    ),
                )

            processed += int(base_prediction.shape[0])
            if rank == 0 and (batch_index + 1) % 20 == 0:
                print(
                    f"[temporal-selector-mapping:{args.split}] local={processed:,}/"
                    f"{len(loader.sampler):,} rate="
                    f"{processed / max(time.time() - started, 1e-9):,.1f}/s",
                    flush=True,
                )

    for accumulator in score_metrics.values():
        accumulator.synchronize(device)
    for accumulator in sequence_score_metrics.values():
        accumulator.synchronize(device)
    for accumulator in policy_metrics.values():
        accumulator.synchronize(device)
    policy_pair_metrics.synchronize(device)
    if world_size > 1:
        dist.all_reduce(subset_frame_counts)
    subset_frame_counts = subset_frame_counts.cpu()

    if rank == 0:
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        sequence_bootstrap_rows = _paired_sequence_score_bootstrap_rows(
            sequence_score_metrics,
            coverages=bootstrap_coverages,
            iterations=args.bootstrap_iterations,
            confidence=args.bootstrap_confidence,
            seed=args.bootstrap_seed,
        )
        policy_pair_rows = policy_pair_metrics.summaries()
        score_rows = []
        pr_rows = []
        budget_rows = []
        for (subset, source, mapping, label), accumulator in score_metrics.items():
            score_rows.append(
                {
                    "subset": subset,
                    "source": source,
                    "mapping": mapping,
                    "label": label,
                    **accumulator.summary(),
                }
            )
            for row in accumulator.pr_curve_rows():
                pr_rows.append(
                    {
                        "subset": subset,
                        "source": source,
                        "mapping": mapping,
                        "label": label,
                        **row,
                    }
                )
            for row in accumulator.budget_rows(budget_coverages):
                budget_rows.append(
                    {
                        "subset": subset,
                        "source": source,
                        "mapping": mapping,
                        "label": label,
                        **row,
                    }
                )
        policy_rows = [
            {"mapping": mapping, **row}
            for mapping, accumulator in policy_metrics.items()
            for row in accumulator.summaries()
        ]
        _write_csv(output_dir / "vertex_score_metrics.csv", score_rows)
        _write_csv(output_dir / "vertex_score_pr_curves.csv", pr_rows)
        _write_csv(output_dir / "vertex_score_budget_points.csv", budget_rows)
        _write_csv(output_dir / "mapping_policy_sweep.csv", policy_rows)
        _write_csv(
            output_dir / "sequence_score_bootstrap.csv",
            sequence_bootstrap_rows,
        )
        _write_csv(
            output_dir / "exact_policy_real_vs_cross.csv",
            policy_pair_rows,
        )

        control_diagnostics: dict[str, Any] = {
            "bin_source": args.cross_control_bin_source,
            "sidecar": str(control_sidecar) if control_sidecar else None,
            "all_control_matched_tactile_frame_count": int(
                subset_frame_counts[0].item()
            ),
            "exact_real_cross_matched_tactile_frame_count": int(
                subset_frame_counts[1].item()
            ),
        }
        if dataset.control_pair_indices is not None:
            control_pairs = np.asarray(dataset.control_pair_indices, dtype=np.int64)
            bins = np.asarray(dataset.control_pressure_bins, dtype=np.int64)
            target_bins = np.asarray(dataset.arrays["pressure_bin"], dtype=np.int64)
            exact_control_match = bins == bins[control_pairs]
            control_diagnostics.update(
                {
                    "control_bin_match_count": int(exact_control_match.sum()),
                    "control_bin_match_fraction": float(np.mean(exact_control_match)),
                    "target_gt_bin_match_fraction": float(
                        np.mean(target_bins == target_bins[control_pairs])
                    ),
                    "cross_sequence_fraction": float(
                        np.mean(
                            np.asarray(dataset.arrays["sequence_key"], dtype=np.str_)
                            != np.asarray(dataset.arrays["sequence_key"], dtype=np.str_)[
                                control_pairs
                            ]
                        )
                    ),
                }
            )

        def score_row(subset: str, source: str, mapping: str, label: str):
            return next(
                row
                for row in score_rows
                if row["subset"] == subset
                and row["source"] == source
                and row["mapping"] == mapping
                and row["label"] == label
            )

        def bootstrap_row(mapping: str, label: str, metric: str):
            return next(
                row
                for row in sequence_bootstrap_rows
                if row["mapping"] == mapping
                and row["label"] == label
                and row["metric"] == metric
            )

        lines = [
            f"Temporal Selector mapping attribution: {args.split}",
            f"Cross control bins: {args.cross_control_bin_source}",
            "Real/full-split vertex score ranking:",
        ]
        for mapping in mapping_modes:
            strict = score_row(
                "full_split", "real", mapping, "strict_false_high_all"
            )
            formal = score_row("full_split", "real", mapping, "formal_false_high")
            down = score_row("full_split", "real", mapping, "down_action")
            best = max(
                (row for row in policy_rows if row["mapping"] == mapping),
                key=lambda row: (
                    float(row["balanced_net_utility"]),
                    float(row["delta_contact_iou"]),
                ),
            )
            lines.append(
                f"  {mapping}: strict_AP={strict['average_precision']:.6f} "
                f"formal_AP={formal['average_precision']:.6f} "
                f"down_AP={down['average_precision']:.6f} "
                f"best_policy={best['name']} utility={best['balanced_net_utility']:.1f}"
            )
        native_strict = score_row(
            "full_split", "real", NATIVE_MAPPING, "strict_false_high_all"
        )
        native_formal = score_row(
            "full_split", "real", NATIVE_MAPPING, "formal_false_high"
        )
        native_down = score_row("full_split", "real", NATIVE_MAPPING, "down_action")
        lines.extend(
            [
                "Native 512-anchor ranking before vertex mapping:",
                f"  strict_AP={native_strict['average_precision']:.6f} "
                f"formal_AP={native_formal['average_precision']:.6f} "
                f"down_AP={native_down['average_precision']:.6f}",
                "GT-anchor projection oracle diagnostic:",
            ]
        )
        for mapping in mapping_modes:
            strict = score_row(
                "full_split", ORACLE_SOURCE, mapping, "strict_false_high_all"
            )
            formal = score_row(
                "full_split", ORACLE_SOURCE, mapping, "formal_false_high"
            )
            down = score_row("full_split", ORACLE_SOURCE, mapping, "down_action")
            lines.append(
                f"  {mapping}: strict_AP={strict['average_precision']:.6f} "
                f"formal_AP={formal['average_precision']:.6f} "
                f"down_AP={down['average_precision']:.6f}"
            )
        lines.append("Matched history controls (strict false-high AP):")
        for mapping in ("rbf4", NATIVE_MAPPING):
            values = []
            for source in SOURCES:
                row = score_row(
                    "matched", source, mapping, "strict_false_high_all"
                )
                values.append(f"{source}={row['average_precision']:.6f}")
            lines.append(f"  {mapping}: " + " ".join(values))
        lines.append(
            "Sequence-clustered exact-control macro AP "
            "(real-cross, confidence interval):"
        )
        for mapping, labels in sequence_metric_labels.items():
            values = []
            for label in labels:
                row = bootstrap_row(
                    mapping, label, "sequence_macro_average_precision"
                )
                values.append(
                    f"{label}={row['reference_minus_control']:+.6f} "
                    f"[{row['ci_low']:+.6f},{row['ci_high']:+.6f}]"
                )
            lines.append(f"  {mapping}: " + " ".join(values))
        nonbase_policy_pairs = [
            row for row in policy_pair_rows if row["name"] != "base"
        ]
        if nonbase_policy_pairs:
            best_pair = max(
                nonbase_policy_pairs,
                key=lambda row: float(row["aligned_net_utility"]),
            )
            lines.append(
                "Best nonzero exact-control RBF policy by aligned utility: "
                f"{best_pair['name']} aligned={best_pair['aligned_net_utility']:.1f} "
                f"CI=[{best_pair['aligned_net_utility_ci95_low']:.1f},"
                f"{best_pair['aligned_net_utility_ci95_high']:.1f}] "
                f"aligned-cross={best_pair['aligned_minus_control_net_utility']:.1f} "
                f"paired_CI=[{best_pair['paired_sequence_net_utility_ci95_low']:.1f},"
                f"{best_pair['paired_sequence_net_utility_ci95_high']:.1f}]"
            )
        lines.append(
            f"Exact {args.cross_control_bin_source} control subset "
            "(real vs cross-sequence AP):"
        )
        for mapping in ("rbf4", NATIVE_MAPPING):
            values = []
            for label in (
                "strict_false_high_all",
                "formal_false_high",
                "down_action",
            ):
                real = score_row(exact_subset, "real", mapping, label)
                cross = score_row(exact_subset, "cross_sequence", mapping, label)
                values.append(
                    f"{label}=({real['average_precision']:.6f},"
                    f"{cross['average_precision']:.6f})"
                )
            lines.append(f"  {mapping}: " + " ".join(values))
        lines.append(
            "Decision: first separate native-anchor selector quality from the "
            "GT-anchor projection diagnostic; only then attribute lost AP to mapping. "
            "A deployable correction must still improve strict/formal AP and pressure "
            "utility under label-free exact-bin controls."
        )
        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = {
            "schema": "tactile_temporal_selector_mapping_attribution_v3",
            "split": args.split,
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "base_checkpoint_sha256": str(payload.get("base_checkpoint_sha256") or ""),
            "pair_index": str(pair_path),
            "sample_count": len(dataset),
            "mapping_modes": list(mapping_modes),
            "native_mapping": NATIVE_MAPPING,
            "oracle_source": ORACLE_SOURCE,
            "exact_control_subset": exact_subset,
            "sequence_bootstrap": {
                "bins": int(args.bootstrap_bins),
                "coverages": list(bootstrap_coverages),
                "iterations": int(args.bootstrap_iterations),
                "confidence": float(args.bootstrap_confidence),
                "seed": int(args.bootstrap_seed),
                "mapping_labels": {
                    mapping: list(labels)
                    for mapping, labels in sequence_metric_labels.items()
                },
            },
            "score_labels": list(SCORE_LABELS),
            "control_diagnostics": control_diagnostics,
            "policy_config": {
                "thresholds": list(_float_csv(args.policy_score_thresholds)),
                "alphas": list(_float_csv(args.policy_alphas)),
                "target_floor": args.policy_target_floor,
                "action_threshold": args.policy_action_threshold,
                "no_contact_max": args.policy_no_contact_max,
                "contact_min": args.policy_contact_min,
            },
            "files": {
                "score_metrics": str(output_dir / "vertex_score_metrics.csv"),
                "score_pr_curves": str(output_dir / "vertex_score_pr_curves.csv"),
                "score_budget_points": str(output_dir / "vertex_score_budget_points.csv"),
                "policy_sweep": str(output_dir / "mapping_policy_sweep.csv"),
                "sequence_score_bootstrap": str(
                    output_dir / "sequence_score_bootstrap.csv"
                ),
                "exact_policy_real_vs_cross": str(
                    output_dir / "exact_policy_real_vs_cross.csv"
                ),
                "summary": str(output_dir / "summary.txt"),
            },
        }
        _write_json(output_dir / "metrics.json", result)
        source = (
            Path(args.copy_val_metrics_from).expanduser()
            if args.copy_val_metrics_from
            else None
        )
        if source and source.is_file():
            shutil.copy2(source, output_dir / "val_metrics.csv")
        print("\n".join(lines), flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
