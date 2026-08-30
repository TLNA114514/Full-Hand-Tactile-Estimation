#!/usr/bin/env python3
"""Select and replay a down-only pressure policy behind Temporal Selector V2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal
from tactile_input_priors.eval_temporal_flow import ExactRankSampler
from tactile_input_priors.runtime import file_sha256, load_torch_checkpoint
from tactile_input_priors.selector_pressure_audit import (
    PressurePolicy,
    PressurePolicyAccumulator,
    PressurePolicyPairAccumulator,
    policy_grid,
    write_policy_outputs,
)
from tactile_input_priors.temporal_flow import (
    TEMPORAL_SELECTOR_FORMAT,
    PartitionedPalmCache,
    TemporalActionSelectorV2,
    TemporalReplayDataset,
    build_temporal_pair_index,
    history_quality_context,
    temporal_manifest_key,
)


SOURCES = ("real", "cross_sequence", "contralateral", "reset")
SUBSETS = ("full_split", "available", "matched")
DOWN_ACTION_INDEX = 0
SCORE_MAPPING = "canonical_rbf_weighted_down_probability"


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
        "--policy-selection-in",
        default="",
        help=(
            "Validation policy_selection.json. When absent, only aligned real "
            "history is swept on the full split; when present, its policies are "
            "replayed unchanged for every history control."
        ),
    )
    parser.add_argument(
        "--policy-score-thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
    )
    parser.add_argument("--policy-alphas", default="0.05,0.10,0.20,0.35,0.50")
    parser.add_argument("--policy-target-floors", default="0.02")
    parser.add_argument("--policy-action-threshold", type=float, default=0.10)
    parser.add_argument("--policy-no-contact-max", type=float, default=0.02)
    parser.add_argument("--policy-subthreshold-max", type=float, default=0.08)
    parser.add_argument("--policy-contact-min", type=float, default=0.10)
    parser.add_argument("--policy-chunk-size", type=int, default=4)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=521)
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


def _anchor_down_to_vertices(
    model: TemporalActionSelectorV2,
    anchor_probability: torch.Tensor,
) -> torch.Tensor:
    if anchor_probability.shape != (anchor_probability.shape[0], model.anchor_count):
        raise ValueError(
            f"Anchor down probability must be [B,{model.anchor_count}], "
            f"got {tuple(anchor_probability.shape)}"
        )
    indices = model.vertex_anchor_indices
    weights = model.vertex_anchor_weights.to(anchor_probability)
    vertex_probability = (
        anchor_probability[:, indices] * weights[None]
    ).sum(dim=-1)
    return vertex_probability.clamp(0.0, 1.0)


def _selection_policies(path: os.PathLike[str] | str) -> tuple[
    tuple[PressurePolicy, ...], dict[str, Any], dict[str, tuple[str, ...]]
]:
    selection_path = Path(path).expanduser().resolve(strict=True)
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "tactile_selector_pressure_policy_v1":
        raise ValueError(f"Unsupported pressure-policy selection: {selection_path}")
    policies = [PressurePolicy("base", 2.0, 0.0, 0.0)]
    names = {"base"}
    profiles: dict[str, list[str]] = {"base": []}
    for profile, item in payload.get("recommendations", {}).items():
        value = item.get("policy") if isinstance(item, Mapping) else None
        if not isinstance(value, Mapping):
            continue
        name = str(value["name"])
        profiles.setdefault(name, []).append(str(profile))
        if name in names:
            continue
        names.add(name)
        policies.append(
            PressurePolicy(
                name=name,
                score_threshold=float(value["score_threshold"]),
                alpha=float(value["alpha"]),
                target_floor=float(value["target_floor"]),
            )
        )
    return tuple(policies), payload, {
        name: tuple(values) for name, values in profiles.items()
    }


def _add_best_nonzero_selection(
    selection_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    candidates = [dict(row) for row in rows if str(row["name"]) != "base"]
    if candidates:
        best = max(
            candidates,
            key=lambda row: (
                float(row["strict_false_high_volume_removed"])
                - float(row["protected_contact_volume_removed"])
                - float(row["added_under_volume"]),
                float(row["delta_contact_iou"]),
                -float(row["delta_rmse"]),
            ),
        )
        utility = (
            float(best["strict_false_high_volume_removed"])
            - float(best["protected_contact_volume_removed"])
            - float(best["added_under_volume"])
        )
        payload.setdefault("recommendations", {})["best_nonzero_balanced"] = {
            "policy": {
                key: best[key]
                for key in ("name", "score_threshold", "alpha", "target_floor")
            },
            "harm_weight": 1.0,
            "utility": utility,
        }
        _write_json(selection_path, payload)
    return payload


def _new_accumulator(
    policies: Sequence[PressurePolicy], args: argparse.Namespace
) -> PressurePolicyAccumulator:
    return PressurePolicyAccumulator(
        policies,
        action_threshold=args.policy_action_threshold,
        no_contact_max=args.policy_no_contact_max,
        subthreshold_max=args.policy_subthreshold_max,
        contact_min=args.policy_contact_min,
        chunk_size=args.policy_chunk_size,
    )


def _new_pair_accumulator(
    policies: Sequence[PressurePolicy], args: argparse.Namespace
) -> PressurePolicyPairAccumulator:
    return PressurePolicyPairAccumulator(
        policies,
        action_threshold=args.policy_action_threshold,
        no_contact_max=args.policy_no_contact_max,
        contact_min=args.policy_contact_min,
        chunk_size=args.policy_chunk_size,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )


def _apply_selection_contract(
    args: argparse.Namespace,
    payload: Mapping[str, Any],
    checkpoint_sha256: str,
) -> None:
    config = dict(payload.get("audit_config", {}))
    expected = str(config.get("checkpoint_sha256", ""))
    if expected and expected != checkpoint_sha256:
        raise ValueError(
            "Validation pressure policy belongs to another temporal selector: "
            f"expected={expected}, actual={checkpoint_sha256}"
        )
    if config.get("score_mapping", SCORE_MAPPING) != SCORE_MAPPING:
        raise ValueError("Validation pressure policy uses another anchor mapping")
    for name in (
        "action_threshold",
        "no_contact_max",
        "subthreshold_max",
        "contact_min",
    ):
        if name in config:
            setattr(args, f"policy_{name}", float(config[name]))


def _tiny_check() -> None:
    probability = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
    recovered = torch.sigmoid(_probability_logit(probability))
    if not torch.allclose(probability, recovered, atol=1e-6, rtol=0.0):
        raise AssertionError("Probability/logit conversion is not reversible")
    policies = policy_grid((0.5,), (0.5,), (0.02,))
    accumulator = PressurePolicyAccumulator(policies, chunk_size=2)
    base = torch.tensor([[0.5, 0.5]])
    target = torch.tensor([[0.0, 0.5]])
    score = torch.tensor([[0.9, 0.1]])
    accumulator.update(
        base,
        torch.zeros_like(base),
        target,
        torch.ones_like(base),
        torch.ones(1),
        false_high_logits=_probability_logit(score),
    )
    rows = accumulator.summaries()
    if rows[1]["strict_false_high_volume_removed"] <= 0.0:
        raise AssertionError("Down-only policy failed to remove strict false-high volume")
    if rows[1]["protected_contact_volume_removed"] != 0.0:
        raise AssertionError("Down-only policy changed an unselected protected point")
    print("Temporal selector pressure-policy tiny check passed", flush=True)


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

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-rank pressure-policy audit requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    checkpoint_sha256 = file_sha256(args.checkpoint)
    payload = load_torch_checkpoint(args.checkpoint)
    if payload.get("format") != TEMPORAL_SELECTOR_FORMAT:
        raise ValueError(
            f"Unsupported temporal selector checkpoint: {payload.get('format')!r}"
        )
    cache = PartitionedPalmCache(args.cache, max_open_shards=args.max_open_shards)
    if cache.base_checkpoint_sha256 != str(payload.get("base_checkpoint_sha256") or ""):
        raise RuntimeError("Audit cache was built from a different RGB baseline")

    selection_payload: dict[str, Any] | None = None
    policy_profiles: dict[str, tuple[str, ...]] = {}
    if args.policy_selection_in:
        policies, selection_payload, policy_profiles = _selection_policies(
            args.policy_selection_in
        )
        _apply_selection_contract(args, selection_payload, checkpoint_sha256)
        mode = "replay"
    else:
        policies = policy_grid(
            _float_csv(args.policy_score_thresholds),
            _float_csv(args.policy_alphas),
            _float_csv(args.policy_target_floors),
        )
        mode = "selection"

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

    model = TemporalActionSelectorV2(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    dataset = TemporalReplayDataset(
        args.cache,
        pair_path,
        include_control=True,
        history_lags=model.history_lags,
        max_open_shards=args.max_open_shards,
    )
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

    if mode == "selection":
        selection_accumulator = _new_accumulator(policies, args)
        accumulators = None
        pair_accumulators = None
    else:
        selection_accumulator = None
        accumulators = {
            subset: {
                source: _new_accumulator(policies, args) for source in SOURCES
            }
            for subset in SUBSETS
        }
        pair_accumulators = {
            control: _new_pair_accumulator(policies, args)
            for control in ("cross_sequence", "contralateral", "reset")
        }

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
            palm = torch.ones_like(base_prediction)
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

            requested_sources = ("real",) if mode == "selection" else SOURCES
            vertex_scores: dict[str, torch.Tensor] = {}
            for source in requested_sources:
                history, available, prefix = source_inputs[source]
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
                anchor_down = output["action_probability"][..., DOWN_ACTION_INDEX]
                score = _anchor_down_to_vertices(model, anchor_down)
                if source != "reset":
                    score = score * source_available[source][:, None]
                vertex_scores[source] = score

            if mode == "selection":
                selection_accumulator.update(
                    base_prediction,
                    torch.zeros_like(base_prediction),
                    target,
                    palm,
                    has_tactile,
                    false_high_logits=_probability_logit(vertex_scores["real"]),
                )
            else:
                valid_masks = {
                    source: {
                        "full_split": has_tactile,
                        "available": has_tactile & source_available[source],
                        "matched": has_tactile & matched,
                    }
                    for source in SOURCES
                }
                for subset in SUBSETS:
                    for source in SOURCES:
                        accumulators[subset][source].update(
                            base_prediction,
                            torch.zeros_like(base_prediction),
                            target,
                            palm,
                            valid_masks[source][subset],
                            false_high_logits=_probability_logit(vertex_scores[source]),
                        )
                sequence_keys = [
                    str(int(value)) for value in batch["sequence_id"].detach().cpu()
                ]
                matched_tactile = has_tactile & matched
                reference_logit = _probability_logit(vertex_scores["real"])
                for control, accumulator in pair_accumulators.items():
                    accumulator.update(
                        base_prediction,
                        torch.zeros_like(base_prediction),
                        torch.zeros_like(base_prediction),
                        target,
                        palm,
                        matched_tactile,
                        sequence_keys=sequence_keys,
                        reference_false_high_logits=reference_logit,
                        control_false_high_logits=_probability_logit(
                            vertex_scores[control]
                        ),
                    )

            processed += int(base_prediction.shape[0])
            if rank == 0 and (batch_index + 1) % 20 == 0:
                print(
                    f"[temporal-selector-policy:{args.split}:{mode}] local="
                    f"{processed:,}/{len(loader.sampler):,} rate="
                    f"{processed / max(time.time() - started, 1e-9):,.1f}/s",
                    flush=True,
                )

    if mode == "selection":
        selection_accumulator.synchronize(device)
    else:
        for subset in SUBSETS:
            for source in SOURCES:
                accumulators[subset][source].synchronize(device)
        for accumulator in pair_accumulators.values():
            accumulator.synchronize(device)

    if rank == 0:
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        common_config = {
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "base_checkpoint_sha256": str(payload.get("base_checkpoint_sha256") or ""),
            "history_lags": list(model.history_lags),
            "use_per_lag_quality": bool(model.use_per_lag_quality),
            "score_action": "down",
            "score_mapping": SCORE_MAPPING,
            "upward_correction_enabled": False,
            "missing_history_behavior": "exact_rgb_no_action",
            "action_threshold": args.policy_action_threshold,
            "no_contact_max": args.policy_no_contact_max,
            "subthreshold_max": args.policy_subthreshold_max,
            "contact_min": args.policy_contact_min,
        }
        if mode == "selection":
            rows = selection_accumulator.summaries()
            selection = write_policy_outputs(
                output_dir,
                rows,
                selection_source=None,
                audit_config={
                    **common_config,
                    "selection_source": "real/full_split",
                    "score_thresholds": list(_float_csv(args.policy_score_thresholds)),
                    "alphas": list(_float_csv(args.policy_alphas)),
                    "target_floors": list(_float_csv(args.policy_target_floors)),
                    "policy_count": len(policies),
                },
            )
            selection = _add_best_nonzero_selection(
                output_dir / "policy_selection.json", rows
            )
            result = {
                "schema": "tactile_temporal_selector_pressure_selection_v1",
                "mode": mode,
                "split": args.split,
                "pair_index": str(pair_path),
                "sample_count": len(dataset),
                "selection": selection,
                "audit_config": common_config,
            }
            lines = [
                f"Temporal Selector down-only pressure selection: {args.split}",
                "Selection source: real/full_split",
                "Upward correction: disabled",
            ]
            for profile, value in selection.get("recommendations", {}).items():
                policy = value.get("policy", {})
                lines.append(
                    f"  {profile}: {policy.get('name')} "
                    f"threshold={policy.get('score_threshold')} "
                    f"alpha={policy.get('alpha')} floor={policy.get('target_floor')} "
                    f"utility={value.get('utility')}"
                )
        else:
            replay_rows = []
            nested: dict[str, Any] = {}
            for subset in SUBSETS:
                nested[subset] = {}
                for source in SOURCES:
                    rows = accumulators[subset][source].summaries()
                    nested[subset][source] = rows
                    for row in rows:
                        replay_rows.append(
                            {
                                "subset": subset,
                                "source": source,
                                "selection_profiles": ",".join(
                                    policy_profiles.get(str(row["name"]), ())
                                ),
                                **row,
                            }
                        )
            pair_rows = []
            nested_pairs: dict[str, Any] = {}
            for control, accumulator in pair_accumulators.items():
                rows = accumulator.summaries()
                nested_pairs[control] = rows
                for row in rows:
                    pair_rows.append({"control": control, **row})
            _write_csv(output_dir / "pressure_policy_replay.csv", replay_rows)
            _write_csv(output_dir / "pressure_policy_pairs.csv", pair_rows)
            result = {
                "schema": "tactile_temporal_selector_pressure_replay_v1",
                "mode": mode,
                "split": args.split,
                "pair_index": str(pair_path),
                "sample_count": len(dataset),
                "selection_source": str(
                    Path(args.policy_selection_in).expanduser().resolve(strict=True)
                ),
                "selection": selection_payload,
                "policy_profiles": policy_profiles,
                "audit_config": common_config,
                "subsets": nested,
                "matched_real_minus_controls": nested_pairs,
            }
            recommendations = selection_payload.get("recommendations", {})
            preferred = (
                recommendations.get("balanced", {})
                .get("policy", {})
                .get("name", "base")
            )
            lines = [
                f"Temporal Selector down-only fixed replay: {args.split}",
                f"Validation balanced policy: {preferred}",
                "Upward correction: disabled",
            ]
            reported_profiles = ("balanced", "best_nonzero_balanced")
            reported_names: set[str] = set()
            for profile in reported_profiles:
                name = (
                    recommendations.get(profile, {})
                    .get("policy", {})
                    .get("name")
                )
                if not name or str(name) in reported_names:
                    continue
                reported_names.add(str(name))
                lines.append(f"Matched real/control metrics for {profile}={name}:")
                for source in SOURCES:
                    row = next(
                        item
                        for item in nested["matched"][source]
                        if str(item["name"]) == str(name)
                    )
                    lines.append(
                        f"  {source}: dRMSE={row['delta_rmse']:+.6f} "
                        f"dContact={row['delta_contact_iou']:+.6f} "
                        f"dV-IoU={row['delta_volumetric_iou']:+.6f} "
                        f"dCoreLoc={row['delta_core_distribution_viou']:+.6f} "
                        f"removed={row['strict_false_high_volume_removed_fraction']:.4f} "
                        f"protected/strict={row['protected_removed_per_strict_removed']:.4f}"
                    )

        _write_json(output_dir / "metrics.json", result)
        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
