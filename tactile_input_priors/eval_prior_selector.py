#!/usr/bin/env python3
"""Evaluate a prior-aware contact selector and its counterfactual controls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader, Subset

cv2.setNumThreads(0)

from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal
from tactile_input_priors.runtime import (
    file_sha256,
    load_prior_selector_checkpoint,
    parse_csv,
)
from tactile_input_priors.selector_prior_metrics import (
    ContactMetricAccumulator,
    SequenceContactAPAccumulator,
)
from tactile_input_priors.selector_exact_topk_audit import (
    ExactTopKAccumulator,
    exact_topk_policies,
    exact_topk_policies_from_selection,
    run_exact_topk_tiny_checks,
    write_exact_topk_outputs,
)
from tactile_input_priors.selector_pressure_audit import (
    MAPPED_CONTROLS,
    MappedPriorDataset,
    PressurePolicyAccumulator,
    PressurePolicyPairAccumulator,
    build_control_mappings,
    dataset_audit_records,
    mapped_control_batch,
    policies_from_selection,
    policy_grid,
    run_tiny_checks,
    write_matched_policy_outputs,
    write_policy_control_replay,
    write_policy_outputs,
)
from tactile_input_priors.train_prior_adapter import _dataset_from_args, add_data_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dino-weights", default="")
    parser.add_argument("--selector-checkpoint", default="")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--controls",
        default=(
            "real,cross_sequence,same_sequence_far,wrong_query,"
            "spatial_shuffle,global_mean,zero"
        ),
    )
    parser.add_argument("--control-min-far-frame-gap", type=int, default=30)
    parser.add_argument(
        "--policy-sweep",
        action="store_true",
        help="Sweep bounded downward pressure policies; only valid on validation.",
    )
    parser.add_argument(
        "--policy-selection-in",
        default="",
        help="Validation policy_selection.json to apply unchanged on a test split.",
    )
    parser.add_argument(
        "--policy-matched-pareto",
        action="store_true",
        help=(
            "Diagnostic full-grid comparison of aligned and RGB selector policies "
            "at matched action coverage and false-high volume removed. This does "
            "not select a deployment policy."
        ),
    )
    parser.add_argument(
        "--policy-exact-topk",
        action="store_true",
        help=(
            "Compare RGB/aligned/control rankings under an identical per-frame "
            "top-k pressure-correction budget. Validation writes fixed replay choices."
        ),
    )
    parser.add_argument(
        "--exact-topk-selection-in",
        default="",
        help="Validation exact_topk_selection.json replayed unchanged on test.",
    )
    parser.add_argument("--exact-topk-values", default="0,1,2,4,8,16,32,64")
    parser.add_argument("--exact-topk-alpha", type=float, default=1.0)
    parser.add_argument("--exact-topk-target-floor", type=float, default=0.02)
    parser.add_argument(
        "--exact-topk-target-removal-fraction", type=float, default=0.03
    )
    parser.add_argument(
        "--matched-coverage-relative-tolerance",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--matched-removal-relative-tolerance",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--policy-score-thresholds",
        default="0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99",
    )
    parser.add_argument("--policy-alphas", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--policy-target-floors", default="0.02,0.05,0.08")
    parser.add_argument("--policy-action-threshold", type=float, default=0.10)
    parser.add_argument(
        "--policy-score-source",
        choices=("auto", "contact", "false_high"),
        default="auto",
        help=(
            "Score used by pressure-policy sweeps. auto uses the dedicated "
            "false-high head for depth_anchor_query and the complement of the "
            "contact score for legacy selector adapters."
        ),
    )
    parser.add_argument(
        "--false-high-score-source",
        choices=("auto", "contact", "head"),
        default="auto",
        help="Metric score source; auto restores the training checkpoint contract.",
    )
    parser.add_argument("--policy-no-contact-max", type=float, default=0.02)
    parser.add_argument("--policy-subthreshold-max", type=float, default=0.08)
    parser.add_argument("--policy-contact-min", type=float, default=0.10)
    parser.add_argument("--policy-chunk-size", type=int, default=8)
    parser.add_argument("--policy-bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--policy-bootstrap-seed", type=int, default=521)
    parser.add_argument(
        "--tiny-check",
        action="store_true",
        help="Run deterministic CPU checks and exit without loading a checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.set_defaults(
        adapter_type="depth_mapping_rectifier",
        no_train_augmentation=True,
        runtime_debug=False,
    )
    add_data_arguments(parser)
    return parser


def _distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    return rank, world_size, local_rank


def _float_csv(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in parse_csv(value))
    if not result:
        raise ValueError("Expected a non-empty comma-separated float list")
    return result


def _int_csv(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in parse_csv(value))
    if not result:
        raise ValueError("Expected a non-empty comma-separated integer list")
    return result


@record
def main() -> None:
    args = build_parser().parse_args()
    if args.tiny_check:
        run_tiny_checks()
        run_exact_topk_tiny_checks()
        print("Selector pressure audit tiny checks passed.", flush=True)
        return
    if args.policy_sweep and args.policy_selection_in:
        raise ValueError("Use either --policy-sweep or --policy-selection-in, not both")
    if args.policy_matched_pareto and args.policy_selection_in:
        raise ValueError(
            "Matched Pareto auditing requires a full policy grid, not a fixed "
            "--policy-selection-in"
        )
    if args.policy_exact_topk and args.exact_topk_selection_in:
        raise ValueError(
            "Use --policy-exact-topk on validation or --exact-topk-selection-in "
            "for replay, not both"
        )
    if (args.policy_exact_topk or args.exact_topk_selection_in) and (
        args.policy_sweep or args.policy_selection_in or args.policy_matched_pareto
    ):
        raise ValueError("Exact-top-k and threshold-policy audits cannot run together")
    normalized_split = str(args.split).strip().lower()
    if args.policy_sweep and normalized_split not in {"val", "validation"}:
        raise ValueError("Pressure-policy sweep is validation-only")
    if args.policy_exact_topk and normalized_split not in {"val", "validation"}:
        raise ValueError("Exact-top-k selection is validation-only")
    if args.exact_topk_selection_in and normalized_split in {"val", "validation"}:
        raise ValueError("Exact-top-k validation must select rather than replay")
    if args.control_min_far_frame_gap < 1:
        raise ValueError("--control-min-far-frame-gap must be positive")
    if args.policy_chunk_size < 1:
        raise ValueError("--policy-chunk-size must be positive")
    if args.policy_bootstrap_iterations < 0:
        raise ValueError("--policy-bootstrap-iterations cannot be negative")
    if not 0.0 <= float(args.exact_topk_target_removal_fraction) <= 1.0:
        raise ValueError("--exact-topk-target-removal-fraction must lie in [0,1]")
    for name in (
        "matched_coverage_relative_tolerance",
        "matched_removal_relative_tolerance",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    rank, world_size, local_rank = _distributed_context()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model, payload = load_prior_selector_checkpoint(
        args.checkpoint,
        dino_weights_override=args.dino_weights or None,
        selector_checkpoint_override=args.selector_checkpoint or None,
    )
    args.adapter_type = str(payload["adapter_config"]["adapter_type"])
    checkpoint_score_source = str(
        payload.get("training_config", {}).get("false_high_score_source", "") or ""
    )
    if args.false_high_score_source == "auto":
        args.false_high_score_source = checkpoint_score_source or (
            "head" if model.adapter_type == "depth_anchor_query" else "contact"
        )
    if args.false_high_score_source == "head" and model.adapter_type != "depth_anchor_query":
        raise ValueError(
            "false-high score source 'head' requires a depth_anchor_query checkpoint"
        )

    def metric_false_high_logits(output):
        if args.false_high_score_source == "contact":
            return None
        logits = output.get("false_high_logits")
        if logits is None:
            raise RuntimeError("Checkpoint metric contract requires false_high_logits")
        return logits

    data_config = dict(payload.get("data_config", {}))
    for name in (
        "datasets",
        "input_resolution",
        "bbox_rescale_factor",
        "bbox_source_policy",
        "depth_sidecar_root",
        "bbox_manifests",
    ):
        if name in data_config:
            current = getattr(args, name, None)
            if current in (None, "") or name in {
                "datasets",
                "input_resolution",
                "bbox_rescale_factor",
                "bbox_source_policy",
            }:
                setattr(args, name, data_config[name])
    controls = tuple(dict.fromkeys(parse_csv(args.controls)))
    if model.adapter_type == "vlm_global_calibrator" and "spatial_shuffle" in controls:
        controls = tuple(
            "context_shuffle" if value == "spatial_shuffle" else value
            for value in controls
        )
        controls = tuple(dict.fromkeys(controls))
    if "real" not in controls:
        controls = ("real", *controls)
    controls = tuple(dict.fromkeys(controls))
    if args.policy_exact_topk or args.exact_topk_selection_in:
        required_exact_controls = ("real", "spatial_shuffle", "global_mean", "zero")
        missing = [value for value in required_exact_controls if value not in controls]
        if missing:
            raise ValueError(
                "Exact-top-k causal audit requires controls "
                f"{required_exact_controls}; missing={missing}"
            )
    known_controls = {
        "real",
        "zero",
        "global_mean",
        "spatial_shuffle",
        "sample_shuffle",
        "context_shuffle",
        *MAPPED_CONTROLS,
    }
    unknown_controls = sorted(set(controls) - known_controls)
    if unknown_controls:
        raise ValueError(f"Unsupported audit controls: {unknown_controls}")

    dataset = _dataset_from_args(args, args.split, False)
    cached_sha = str(getattr(dataset, "base_checkpoint_sha256", "") or "")
    expected_sha = file_sha256(payload["selector_checkpoint"])
    if cached_sha and cached_sha != expected_sha:
        raise ValueError(
            "Evaluation feature cache was built from a different selector checkpoint"
        )
    if parse_csv(args.base_feature_cache or args.val_base_feature_cache):
        model.disable_online_backbone()
    mapped_names = tuple(name for name in controls if name in MAPPED_CONTROLS)
    control_mapping_summary = {}
    if mapped_names:
        records = dataset_audit_records(dataset)
        all_mappings, all_mapping_summaries = build_control_mappings(
            records,
            seed=int(payload.get("adapter_config", {}).get("control_seed", 521)),
            minimum_far_frame_gap=args.control_min_far_frame_gap,
        )
        dataset = MappedPriorDataset(
            dataset,
            records,
            {name: all_mappings[name] for name in mapped_names},
            prior_kind="depth" if model.is_depth else "vlm",
        )
        control_mapping_summary = {
            name: all_mapping_summaries[name] for name in mapped_names
        }
    subset = Subset(dataset, range(rank, len(dataset), world_size))
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    if args.num_workers:
        loader_kwargs.update(
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
        )
    loader = DataLoader(subset, **loader_kwargs)
    model = model.to(device).eval()
    policies = None
    selection_payload = None
    if args.policy_sweep or args.policy_matched_pareto:
        policies = policy_grid(
            _float_csv(args.policy_score_thresholds),
            _float_csv(args.policy_alphas),
            _float_csv(args.policy_target_floors),
        )
    elif args.policy_selection_in:
        selection_path = Path(args.policy_selection_in).expanduser().resolve(strict=True)
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_config = dict(selection_payload.get("audit_config", {}))
        expected_checkpoint_sha = str(selection_config.get("checkpoint_sha256", ""))
        actual_checkpoint_sha = file_sha256(args.checkpoint)
        if expected_checkpoint_sha and expected_checkpoint_sha != actual_checkpoint_sha:
            raise ValueError(
                "Validation pressure policy belongs to another selector-prior checkpoint: "
                f"expected={expected_checkpoint_sha}, actual={actual_checkpoint_sha}"
            )
        for name in (
            "action_threshold",
            "no_contact_max",
            "subthreshold_max",
            "contact_min",
        ):
            if name in selection_config:
                setattr(args, f"policy_{name}", float(selection_config[name]))
        selected_score_source = str(
            selection_config.get("policy_score_source", "") or ""
        )
        if selected_score_source:
            if (
                args.policy_score_source != "auto"
                and args.policy_score_source != selected_score_source
            ):
                raise ValueError(
                    "Pressure-policy score source differs from validation: "
                    f"validation={selected_score_source}, "
                    f"evaluation={args.policy_score_source}"
                )
            args.policy_score_source = selected_score_source
        elif args.policy_score_source == "auto":
            # Historical selections were always produced from
            # sigmoid(-contact_logits). Preserve that contract rather than
            # silently replaying an old threshold on the new direct head.
            args.policy_score_source = "contact"
        policies = policies_from_selection(args.policy_selection_in)
    use_false_high_policy_score = (
        args.policy_score_source == "false_high"
        or (
            args.policy_score_source == "auto"
            and args.false_high_score_source == "head"
        )
    )
    resolved_policy_score_source = (
        "false_high" if use_false_high_policy_score else "contact"
    )

    def policy_false_high_logits(output):
        if not use_false_high_policy_score:
            return None
        logits = output.get("false_high_logits")
        if logits is None:
            raise RuntimeError(
                "The selected pressure-policy score source requires false_high_logits"
            )
        return logits

    def exact_risk_score(output, *, rgb_base: bool = False):
        if rgb_base:
            # The frozen RGB selector has no dedicated false-high head.
            return -output["base_contact_logits"].float()
        if use_false_high_policy_score:
            logits = output.get("false_high_logits")
            if logits is None:
                raise RuntimeError(
                    "Exact-top-k score contract requires false_high_logits"
                )
            return logits.float()
        return -output["fused_contact_logits"].float()

    def new_policy_accumulator():
        return PressurePolicyAccumulator(
            policies,
            action_threshold=args.policy_action_threshold,
            no_contact_max=args.policy_no_contact_max,
            subthreshold_max=args.policy_subthreshold_max,
            contact_min=args.policy_contact_min,
            chunk_size=args.policy_chunk_size,
        )

    def new_policy_pair_accumulator():
        return PressurePolicyPairAccumulator(
            policies,
            action_threshold=args.policy_action_threshold,
            no_contact_max=args.policy_no_contact_max,
            contact_min=args.policy_contact_min,
            chunk_size=args.policy_chunk_size,
            bootstrap_iterations=args.policy_bootstrap_iterations,
            bootstrap_seed=args.policy_bootstrap_seed,
        )

    policy_accumulator = new_policy_accumulator() if policies is not None else None
    policy_base_accumulator = (
        new_policy_accumulator()
        if policies is not None and args.policy_matched_pareto
        else None
    )
    policy_replay_enabled = policies is not None and bool(args.policy_selection_in)
    policy_control_accumulators = (
        {
            source: new_policy_accumulator()
            for source in (
                "base_selector",
                *(control for control in controls if control != "real"),
            )
        }
        if policy_replay_enabled
        else {}
    )
    policy_reference_accumulators = (
        {
            control: new_policy_accumulator()
            for control in mapped_names
        }
        if policy_replay_enabled
        else {}
    )
    policy_pair_accumulators = (
        {
            source: new_policy_pair_accumulator()
            for source in policy_control_accumulators
        }
        if policy_replay_enabled
        else {}
    )
    exact_selection_payload = None
    exact_selection_path = None
    exact_policies = None
    if args.policy_exact_topk:
        exact_policies = exact_topk_policies(
            _int_csv(args.exact_topk_values),
            alpha=args.exact_topk_alpha,
            target_floor=args.exact_topk_target_floor,
        )
    elif args.exact_topk_selection_in:
        exact_selection_path = Path(args.exact_topk_selection_in).expanduser().resolve(
            strict=True
        )
        exact_selection_payload = json.loads(
            exact_selection_path.read_text(encoding="utf-8")
        )
        exact_config = dict(exact_selection_payload.get("audit_config", {}))
        expected_checkpoint_sha = str(exact_config.get("checkpoint_sha256", ""))
        actual_checkpoint_sha = file_sha256(args.checkpoint)
        if expected_checkpoint_sha and expected_checkpoint_sha != actual_checkpoint_sha:
            raise ValueError(
                "Exact-top-k selection belongs to another checkpoint: "
                f"expected={expected_checkpoint_sha}, actual={actual_checkpoint_sha}"
            )
        expected_score_source = str(exact_config.get("policy_score_source", ""))
        if expected_score_source and expected_score_source != resolved_policy_score_source:
            raise ValueError(
                "Exact-top-k score source differs from validation: "
                f"validation={expected_score_source}, evaluation={resolved_policy_score_source}"
            )
        for config_name, argument_name in (
            ("action_threshold", "policy_action_threshold"),
            ("no_contact_max", "policy_no_contact_max"),
            ("contact_min", "policy_contact_min"),
        ):
            if config_name in exact_config:
                setattr(args, argument_name, float(exact_config[config_name]))
        exact_policies = exact_topk_policies_from_selection(exact_selection_path)

    def new_exact_accumulator():
        return ExactTopKAccumulator(
            exact_policies,
            action_threshold=args.policy_action_threshold,
            no_contact_max=args.policy_no_contact_max,
            contact_min=args.policy_contact_min,
            chunk_size=args.policy_chunk_size,
        )

    exact_accumulators = (
        {
            source: new_exact_accumulator()
            for source in (
                "rgb_base",
                "real",
                "spatial_shuffle",
                "global_mean",
                "zero",
            )
        }
        if exact_policies is not None
        else {}
    )
    exact_action_pair_stats = {
        source: torch.zeros(
            (len(exact_policies), 3), device=device, dtype=torch.float64
        )
        for source in exact_accumulators
        if source != "real"
    }

    def update_exact_action_pair(
        source: str,
        real_selected: torch.Tensor,
        source_selected: torch.Tensor,
    ) -> None:
        if real_selected.shape != source_selected.shape:
            raise ValueError("Exact-top-k paired action masks differ in shape")
        exact_action_pair_stats[source] += torch.stack(
            (
                (real_selected & source_selected).sum(dim=(1, 2)),
                (real_selected | source_selected).sum(dim=(1, 2)),
                (real_selected ^ source_selected).sum(dim=(1, 2)),
            ),
            dim=1,
        ).double()
    accumulators = {
        control: ContactMetricAccumulator(
            no_contact_max=model.base_model.support_selector_no_contact_max,
            contact_min=model.base_model.support_selector_contact_min,
        )
        for control in controls
    }
    base_metrics = ContactMetricAccumulator(
        no_contact_max=model.base_model.support_selector_no_contact_max,
        contact_min=model.base_model.support_selector_contact_min,
    )
    sequence_accumulators = {
        control: SequenceContactAPAccumulator(
            no_contact_max=model.base_model.support_selector_no_contact_max,
            contact_min=model.base_model.support_selector_contact_min,
        )
        for control in controls
    }
    base_sequence_metrics = SequenceContactAPAccumulator(
        no_contact_max=model.base_model.support_selector_no_contact_max,
        contact_min=model.base_model.support_selector_contact_min,
    )
    control_references = {
        control: ContactMetricAccumulator(
            no_contact_max=model.base_model.support_selector_no_contact_max,
            contact_min=model.base_model.support_selector_contact_min,
        )
        for control in mapped_names
    }
    control_sequence_references = {
        control: SequenceContactAPAccumulator(
            no_contact_max=model.base_model.support_selector_no_contact_max,
            contact_min=model.base_model.support_selector_contact_min,
        )
        for control in mapped_names
    }
    delta_stats = {
        control: torch.zeros(4, device=device, dtype=torch.float64)
        for control in controls
    }
    progress_every = max(1, len(loader) // 100)
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = {
                key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value)
                for key, value in batch.items()
            }
            real = model.forward_step(batch, prior_control="real")
            if not torch.equal(real["pred_logits"], real["base_pressure_logits"]):
                raise RuntimeError("Prior selector changed pressure during evaluation")
            common = (
                batch["tactile_signal"],
                batch["palm_mask"],
                batch["has_tactile"],
            )
            pressure = real["base_pred_tactile"]
            raw_sequence_keys = [
                str(value) for value in batch.get("sequence_key", [])
            ]
            raw_datasets = [
                str(value)
                for value in batch.get("dataset", batch.get("dataset_name", []))
            ]
            if raw_sequence_keys and raw_datasets and len(raw_datasets) != len(
                raw_sequence_keys
            ):
                raise ValueError("dataset and sequence_key batch lengths differ")
            sequence_keys = [
                f"{raw_datasets[index]}/{value}" if raw_datasets else value
                for index, value in enumerate(raw_sequence_keys)
            ]
            if exact_accumulators and (
                len(sequence_keys) != pressure.shape[0]
                or any(not value for value in sequence_keys)
            ):
                raise RuntimeError(
                    "Exact-top-k sequence bootstrap requires one non-empty "
                    "dataset/sequence_key per sample"
                )
            if exact_accumulators:
                exact_rgb_selected = exact_accumulators["rgb_base"].update(
                    pressure,
                    exact_risk_score(real, rgb_base=True),
                    *common,
                    sequence_keys=sequence_keys,
                )
                exact_real_selected = exact_accumulators["real"].update(
                    pressure,
                    exact_risk_score(real),
                    *common,
                    sequence_keys=sequence_keys,
                )
                update_exact_action_pair(
                    "rgb_base", exact_real_selected, exact_rgb_selected
                )
            if policy_accumulator is not None:
                policy_accumulator.update(
                    pressure,
                    real["fused_contact_logits"],
                    *common,
                    false_high_logits=policy_false_high_logits(real),
                )
            if policy_base_accumulator is not None:
                policy_base_accumulator.update(
                    pressure,
                    real["base_contact_logits"],
                    *common,
                )
            if policy_replay_enabled:
                policy_control_accumulators["base_selector"].update(
                    pressure,
                    real["base_contact_logits"],
                    *common,
                )
                policy_pair_accumulators["base_selector"].update(
                    pressure,
                    real["fused_contact_logits"],
                    real["base_contact_logits"],
                    *common,
                    sequence_keys=sequence_keys,
                    reference_false_high_logits=policy_false_high_logits(real),
                )
            base_metrics.update(
                real["base_contact_logits"], *common, base_pressure=pressure
            )
            if sequence_keys:
                base_sequence_metrics.update(
                    real["base_contact_logits"],
                    *common,
                    sequence_keys=sequence_keys,
                )
            for control in controls:
                control_common = common
                if control in MAPPED_CONTROLS:
                    available = batch[f"_audit_{control}_available"].reshape(-1) > 0.5
                    control_has = batch["has_tactile"].reshape(-1) * available.to(
                        batch["has_tactile"].dtype
                    )
                    control_common = (
                        batch["tactile_signal"],
                        batch["palm_mask"],
                        control_has,
                    )
                    paired_batch = mapped_control_batch(
                        batch, control, is_depth=model.is_depth
                    )
                    output = model.forward_step(
                        paired_batch,
                        prior_control="real",
                        cached_evidence=real["_base_evidence"],
                    )
                    control_references[control].update(
                        real["fused_contact_logits"],
                        *control_common,
                        base_pressure=pressure,
                        false_high_logits=metric_false_high_logits(real),
                    )
                    if policy_replay_enabled:
                        policy_reference_accumulators[control].update(
                            pressure,
                            real["fused_contact_logits"],
                            *control_common,
                            false_high_logits=policy_false_high_logits(real),
                        )
                    if sequence_keys:
                        control_sequence_references[control].update(
                            real["fused_contact_logits"],
                            *control_common,
                            sequence_keys=sequence_keys,
                        )
                else:
                    output = real if control == "real" else model.forward_step(
                        batch,
                        prior_control=control,
                        cached_evidence=real["_base_evidence"],
                    )
                accumulators[control].update(
                    output["fused_contact_logits"],
                    *control_common,
                    base_pressure=pressure,
                    false_high_logits=metric_false_high_logits(output),
                )
                if policy_replay_enabled and control != "real":
                    policy_control_accumulators[control].update(
                        pressure,
                        output["fused_contact_logits"],
                        *control_common,
                        false_high_logits=policy_false_high_logits(output),
                    )
                    policy_pair_accumulators[control].update(
                        pressure,
                        real["fused_contact_logits"],
                        output["fused_contact_logits"],
                        *control_common,
                        sequence_keys=sequence_keys,
                        reference_false_high_logits=policy_false_high_logits(real),
                        control_false_high_logits=policy_false_high_logits(output),
                    )
                if exact_accumulators and control in exact_accumulators and control != "real":
                    exact_control_selected = exact_accumulators[control].update(
                        pressure,
                        exact_risk_score(output),
                        *control_common,
                        sequence_keys=sequence_keys,
                    )
                    update_exact_action_pair(
                        control, exact_real_selected, exact_control_selected
                    )
                if sequence_keys:
                    sequence_accumulators[control].update(
                        output["fused_contact_logits"],
                        *control_common,
                        sequence_keys=sequence_keys,
                    )
                valid = (batch["palm_mask"] > 0.5) & (
                    control_common[2].reshape(-1) > 0.5
                )[:, None]
                residual = output["prior_contact_residual"].double()[valid]
                delta_stats[control][0] += residual.sum()
                delta_stats[control][1] += residual.square().sum()
                if residual.numel():
                    delta_stats[control][2] = torch.maximum(
                        delta_stats[control][2], residual.abs().max()
                    )
                delta_stats[control][3] += residual.new_tensor(residual.numel())
            if rank == 0 and (batch_index % progress_every == 0 or batch_index + 1 == len(loader)):
                print(
                    f"[selector-eval] {batch_index + 1}/{len(loader)} "
                    f"({100.0 * (batch_index + 1) / max(len(loader), 1):.1f}%)",
                    flush=True,
                )
    base_metrics.synchronize(device)
    base_sequence_metrics.synchronize()
    for accumulator in accumulators.values():
        accumulator.synchronize(device)
    for accumulator in sequence_accumulators.values():
        accumulator.synchronize()
    for accumulator in control_references.values():
        accumulator.synchronize(device)
    for accumulator in control_sequence_references.values():
        accumulator.synchronize()
    if policy_accumulator is not None:
        policy_accumulator.synchronize(device)
    if policy_base_accumulator is not None:
        policy_base_accumulator.synchronize(device)
    for accumulator in policy_control_accumulators.values():
        accumulator.synchronize(device)
    for accumulator in policy_reference_accumulators.values():
        accumulator.synchronize(device)
    for accumulator in policy_pair_accumulators.values():
        accumulator.synchronize(device)
    for accumulator in exact_accumulators.values():
        accumulator.synchronize(device)
    if world_size > 1:
        for values in exact_action_pair_stats.values():
            torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    if world_size > 1:
        for values in delta_stats.values():
            torch.distributed.all_reduce(values[:2], op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(values[2:3], op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(values[3:], op=torch.distributed.ReduceOp.SUM)
    if rank == 0:
        base = base_metrics.summary()
        base.update(base_sequence_metrics.summary())
        results = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_monitor": payload.get("monitor"),
            "checkpoint_score": payload.get("score"),
            "split": args.split,
            "sample_count": len(dataset),
            "pressure_output_contract": payload.get("pressure_output_contract"),
            "policy_score_source": resolved_policy_score_source,
            "false_high_score_source": args.false_high_score_source,
            "control_mapping": control_mapping_summary,
            "legacy_control_notes": {
                "sample_shuffle": (
                    "Historical batch-local cyclic roll; ordered batches may contain "
                    "nearby frames. Use cross_sequence for causal interpretation."
                )
            },
            "base": base,
            "controls": {},
            "control_references": {},
        }
        for control, accumulator in accumulators.items():
            summary = accumulator.summary()
            summary.update(sequence_accumulators[control].summary())
            if control in control_references:
                reference = control_references[control].summary()
                reference.update(control_sequence_references[control].summary())
            else:
                reference = results["controls"].get("real")
                if reference is None:
                    reference = accumulators["real"].summary()
                    reference.update(sequence_accumulators["real"].summary())
            results["control_references"][control] = reference
            count = delta_stats[control][3].clamp_min(1.0)
            summary.update(
                {
                    "contact_ap_gain_vs_base": summary["contact_ap"] - base["contact_ap"],
                    "false_high_ap_gain_vs_base": (
                        summary["false_high_candidate_ap"]
                        - base["false_high_candidate_ap"]
                    ),
                    "contact_delta_mean": float(delta_stats[control][0] / count),
                    "contact_delta_rms": float((delta_stats[control][1] / count).sqrt()),
                    "contact_delta_abs_max": float(delta_stats[control][2]),
                    "evaluated_vertex_count": float(delta_stats[control][3]),
                    "contact_ap_gap_vs_aligned_real": (
                        summary["contact_ap"] - reference["contact_ap"]
                    ),
                    "false_high_ap_gap_vs_aligned_real": (
                        summary["false_high_candidate_ap"]
                        - reference["false_high_candidate_ap"]
                    ),
                }
            )
            results["controls"][control] = summary
        output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
        output_dir.mkdir(parents=True, exist_ok=True)
        pr_curve_path = output_dir / "false_high_pr_curve.csv"
        with pr_curve_path.open("w", encoding="utf-8", newline="") as handle:
            fieldnames = (
                "source",
                "score_source",
                "score_threshold",
                "precision",
                "recall",
                "selected_count",
                "selected_fraction",
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for source, accumulator in (
                ("base", base_metrics),
                *((name, accumulators[name]) for name in controls),
            ):
                score_source = (
                    "contact"
                    if source == "base"
                    else args.false_high_score_source
                )
                for curve_row in accumulator.false_high_pr_curve():
                    writer.writerow(
                        {
                            "source": source,
                            "score_source": score_source,
                            **curve_row,
                        }
                    )
        results["false_high_pr_curve_csv"] = str(pr_curve_path)
        results["aligned_control_gaps"] = {}
        for control, values in results["controls"].items():
            if control == "real":
                continue
            reference = results["control_references"][control]
            results["aligned_control_gaps"][control] = {
                "contact_ap": reference["contact_ap"] - values["contact_ap"],
                "false_high_candidate_ap": (
                    reference["false_high_candidate_ap"]
                    - values["false_high_candidate_ap"]
                ),
                "high_precision_false_high_coverage": (
                    reference["false_high_recall_at_precision_0.9"]
                    - values["false_high_recall_at_precision_0.9"]
                ),
                "reference_scope": (
                    "mapped_available_subset"
                    if control in MAPPED_CONTROLS
                    else "full_split"
                ),
            }
        policy_rows = None
        if policy_accumulator is not None:
            policy_rows = policy_accumulator.summaries()
            policy_payload = None
            if args.policy_sweep or args.policy_selection_in:
                policy_payload = write_policy_outputs(
                    output_dir,
                    policy_rows,
                    selection_source=args.policy_selection_in or None,
                    audit_config={
                        "checkpoint_sha256": file_sha256(args.checkpoint),
                        "selector_checkpoint_sha256": file_sha256(
                            payload["selector_checkpoint"]
                        ),
                        "action_threshold": args.policy_action_threshold,
                        "no_contact_max": args.policy_no_contact_max,
                        "subthreshold_max": args.policy_subthreshold_max,
                        "contact_min": args.policy_contact_min,
                        "score_thresholds": list(_float_csv(args.policy_score_thresholds)),
                        "alphas": list(_float_csv(args.policy_alphas)),
                        "target_floors": list(_float_csv(args.policy_target_floors)),
                        "policy_count": len(policies),
                        "bootstrap_iterations": args.policy_bootstrap_iterations,
                        "bootstrap_seed": args.policy_bootstrap_seed,
                        "policy_score_source": resolved_policy_score_source,
                    },
                )
                selected_names = {
                    "base",
                    *(
                        str(value.get("policy", {}).get("name", ""))
                        for value in policy_payload.get("recommendations", {}).values()
                    ),
                }
                results["pressure_policy_audit"] = {
                    **policy_payload,
                    "reported_rows": [
                        row for row in policy_rows if row["name"] in selected_names
                    ],
                    "sweep_csv": str(output_dir / "pressure_policy_sweep.csv"),
                    "pareto_csv": str(output_dir / "pressure_policy_pareto.csv"),
                }
            if args.policy_matched_pareto:
                if policy_base_accumulator is None:
                    raise RuntimeError("Matched Pareto audit lacks the RGB accumulator")
                matched_payload = write_matched_policy_outputs(
                    output_dir,
                    aligned_rows=policy_rows,
                    rgb_rows=policy_base_accumulator.summaries(),
                    coverage_relative_tolerance=(
                        args.matched_coverage_relative_tolerance
                    ),
                    removal_relative_tolerance=(
                        args.matched_removal_relative_tolerance
                    ),
                )
                results["matched_policy_pareto"] = matched_payload
            if policy_replay_enabled:
                source_rows = {
                    "real": policy_rows,
                    **{
                        source: accumulator.summaries()
                        for source, accumulator in policy_control_accumulators.items()
                    },
                }
                reference_rows = {"real": policy_rows}
                reference_scopes = {"real": "full_split_identity"}
                for source in policy_control_accumulators:
                    if source in policy_reference_accumulators:
                        reference_rows[source] = policy_reference_accumulators[
                            source
                        ].summaries()
                        reference_scopes[source] = "mapped_available_subset"
                    else:
                        reference_rows[source] = policy_rows
                        reference_scopes[source] = "full_split"
                pair_rows = {
                    source: accumulator.summaries()
                    for source, accumulator in policy_pair_accumulators.items()
                }
                policy_profiles = {}
                for profile, value in (selection_payload or {}).get(
                    "recommendations", {}
                ).items():
                    name = str(value.get("policy", {}).get("name", ""))
                    if name:
                        policy_profiles.setdefault(name, []).append(str(profile))
                replay_payload = write_policy_control_replay(
                    output_dir,
                    source_rows=source_rows,
                    reference_rows=reference_rows,
                    pair_rows=pair_rows,
                    reference_scopes=reference_scopes,
                    policy_profiles=policy_profiles,
                )
                results["pressure_policy_control_replay"] = replay_payload
        if exact_accumulators:
            nonzero_exact = next(
                (policy for policy in exact_policies if policy.topk > 0),
                exact_policies[0],
            )
            exact_payload = write_exact_topk_outputs(
                output_dir,
                accumulators=exact_accumulators,
                audit_config={
                    "checkpoint_sha256": file_sha256(args.checkpoint),
                    "selector_checkpoint_sha256": file_sha256(
                        payload["selector_checkpoint"]
                    ),
                    "cached_selector_checkpoint_sha256": cached_sha or None,
                    "index_manifest_sha256": str(
                        getattr(dataset, "index_manifest_sha256", "") or ""
                    ),
                    "query_manifest_sha256": dict(
                        getattr(dataset, "query_manifest_sha256", {}) or {}
                    ),
                    "bbox_manifest_sha256": dict(
                        getattr(dataset, "bbox_manifest_sha256", {}) or {}
                    ),
                    "depth_sidecar_contract": dict(
                        getattr(dataset, "depth_sidecar_contract", {}) or {}
                    ),
                    "base_feature_cache": list(
                        parse_csv(
                            args.base_feature_cache
                            or args.val_base_feature_cache
                            or ""
                        )
                    ),
                    "split": str(args.split),
                    "action_threshold": float(args.policy_action_threshold),
                    "no_contact_max": float(args.policy_no_contact_max),
                    "contact_min": float(args.policy_contact_min),
                    "topk_values": [int(policy.topk) for policy in exact_policies],
                    "alpha": float(nonzero_exact.alpha),
                    "target_floor": float(nonzero_exact.target_floor),
                    "policy_score_source": resolved_policy_score_source,
                    "rgb_score_source": "negative_base_contact_logit",
                    "candidate_contract": (
                        "valid palm and has_tactile and frozen RGB pressure "
                        ">= action_threshold"
                    ),
                    "budget_contract": "per-frame min(k, candidate_count)",
                    "tie_break_contract": (
                        "stable descending raw-risk sort; lower canonical vertex "
                        "index wins exact ties"
                    ),
                    "bootstrap_iterations": int(args.policy_bootstrap_iterations),
                    "bootstrap_seed": int(args.policy_bootstrap_seed),
                },
                selection_source=exact_selection_path,
                bootstrap_iterations=args.policy_bootstrap_iterations,
                bootstrap_seed=args.policy_bootstrap_seed,
                target_removal_fraction=args.exact_topk_target_removal_fraction,
                action_pair_stats={
                    source: values.cpu()
                    for source, values in exact_action_pair_stats.items()
                },
            )
            results["exact_topk_causal_audit"] = exact_payload
        (output_dir / "metrics.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        lines = [
            f"Selector-prior evaluation: split={args.split}",
            f"Samples: {len(dataset)}",
            f"Base contact AP: {base['contact_ap']:.6f}",
            f"Base false-high candidate AP: {base['false_high_candidate_ap']:.6f}",
        ]
        for control, values in results["controls"].items():
            lines.append(
                f"{control}: contact AP={values['contact_ap']:.6f}, "
                f"false-high AP={values['false_high_candidate_ap']:.6f}, "
                f"R@P70={values['false_high_recall_at_precision_0.7']:.6f}, "
                f"R@P90={values['false_high_recall_at_precision_0.9']:.6f}, "
                f"top16 precision={values['false_high_top16_precision']:.6f}, "
                f"delta RMS={values['contact_delta_rms']:.6f}"
            )
        if policy_rows is not None and (args.policy_sweep or args.policy_selection_in):
            lines.append(
                "Pressure policy audit: "
                f"{len(policy_rows) - 1} candidate policy/policies; "
                "see pressure_policy_sweep.csv and pressure_policy_pareto.csv"
            )
            if policy_replay_enabled:
                lines.append(
                    "Fixed policy replay: see pressure_policy_control_replay.csv "
                    "and pressure_policy_control_replay.json"
                )
        if args.policy_matched_pareto:
            matched = results["matched_policy_pareto"]
            lines.append(
                "Matched policy Pareto audit: "
                f"coverage={matched['coverage_summary']['status']}, "
                f"removed-volume={matched['removal_summary']['status']}; "
                "diagnostic only, no test policy selected"
            )
        if exact_accumulators:
            lines.append(
                "Exact-top-k causal audit: identical per-frame action counts for "
                "RGB/real/spatial-shuffle/global/zero; see exact_topk_summary.txt"
            )
        (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        experiment_dir = Path(args.checkpoint).expanduser().resolve().parent.parent
        validation_metrics = experiment_dir / "val_metrics.csv"
        if validation_metrics.is_file():
            shutil.copy2(validation_metrics, output_dir / "val_metrics.csv")
        print("\n".join(lines), flush=True)
    if world_size > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
