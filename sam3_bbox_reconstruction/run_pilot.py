#!/usr/bin/env python3
"""Prepare and run the 21-sequence SAM3 bbox pilot across visible GPUs."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import html
import json
import os
import queue
import shutil
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

try:
    from .defaults import (
        DEFAULT_OPENTOUCH_DATA_ROOT,
        DEFAULT_OPENTOUCH_SPLITS,
        DEFAULT_TOUCHANYTHING_ROOT,
        DEFAULT_TOUCHANYTHING_SPLIT_JSON,
        resolve_checkpoint,
    )
    from .pilot_manifest import (
        build_manifest,
        cleanup_opentouch_materialization,
        materialize_opentouch_record,
        parse_dataset_selection,
    )
    from .progress import progress
    from .track_video import (
        CHUNK_CONTINUITY_VERSION,
        load_prompt_preset,
        resolve_verifier_prompt_lists,
    )
except ImportError:  # Direct execution through run_pilot.sh.
    from defaults import (
        DEFAULT_OPENTOUCH_DATA_ROOT,
        DEFAULT_OPENTOUCH_SPLITS,
        DEFAULT_TOUCHANYTHING_ROOT,
        DEFAULT_TOUCHANYTHING_SPLIT_JSON,
        resolve_checkpoint,
    )
    from pilot_manifest import (
        build_manifest,
        cleanup_opentouch_materialization,
        materialize_opentouch_record,
        parse_dataset_selection,
    )
    from progress import progress
    from track_video import (
        CHUNK_CONTINUITY_VERSION,
        load_prompt_preset,
        resolve_verifier_prompt_lists,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="opentouch,touchanything",
        help="opentouch, touchanything, or a comma-separated combination.",
    )
    parser.add_argument(
        "--splits",
        default="auto",
        help="Comma-separated split names; auto selects every canonical split.",
    )
    parser.add_argument("--opentouch-data-root", type=Path, default=DEFAULT_OPENTOUCH_DATA_ROOT)
    parser.add_argument(
        "--opentouch-splits",
        type=Path,
        default=DEFAULT_OPENTOUCH_SPLITS,
    )
    parser.add_argument("--touchanything-root", type=Path, default=DEFAULT_TOUCHANYTHING_ROOT)
    parser.add_argument(
        "--touchanything-split-json",
        type=Path,
        default=DEFAULT_TOUCHANYTHING_SPLIT_JSON,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/pilot_<selected datasets>.",
    )
    parser.add_argument("--samples-per-split", type=int, default=3)
    parser.add_argument(
        "--samples-per-dataset",
        type=int,
        default=0,
        help=(
            "Randomly select this many sequences across all requested splits of each "
            "dataset. Zero keeps --samples-per-split behavior."
        ),
    )
    parser.add_argument(
        "--all-sequences",
        action="store_true",
        help="Run every available sequence in every selected split.",
    )
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help=(
            "Independent persistent sequence workers per physical GPU. Each worker "
            "loads one model copy; use 2 for OpenTouch on 80GB A800 GPUs."
        ),
    )
    parser.add_argument(
        "--cpu-threads-per-worker",
        type=int,
        default=1,
        help=(
            "CPU threads available to each persistent worker for OpenCV, PyTorch, "
            "OpenMP, BLAS, and NumExpr. Keep this at 1 when running many workers."
        ),
    )
    parser.add_argument("--sam-version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Local SAM checkpoint. By default sam3.pt or sam3.1_multiplex.pt is "
            "resolved from the shared repository _DATA directory."
        ),
    )
    parser.add_argument("--prompt-preset", choices=("gloved", "bare"), default="gloved")
    parser.add_argument("--prompt", help="Override the prompt preset primary phrase")
    parser.add_argument(
        "--propagation-direction",
        choices=("forward", "both"),
        default="forward",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs complete sequences")
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="Safety cap; 0 uses OpenTouch=1 and TouchAnything=2.",
    )
    parser.add_argument(
        "--sam-candidate-capacity",
        type=int,
        default=0,
        help="SAM3.1 candidate capacity; 0 resolves to at least four before query selection.",
    )
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.0005)
    parser.add_argument(
        "--min-relative-mask-area",
        type=float,
        default=0.05,
        help="Legacy compatibility option; not used for prompt-track identity selection.",
    )
    parser.add_argument("--min-prompt-score", type=float, default=0.5)
    parser.add_argument("--min-track-frames", type=int, default=2)
    parser.add_argument("--duplicate-track-iou-floor", type=float, default=0.80)
    parser.add_argument("--duplicate-track-overlap-fraction", type=float, default=0.60)
    parser.add_argument("--duplicate-track-match-fraction", type=float, default=0.80)
    parser.add_argument("--duplicate-track-centroid-ratio", type=float, default=0.18)
    parser.add_argument("--duplicate-track-area-ratio", type=float, default=1.50)
    parser.add_argument("--duplicate-track-min-frames", type=int, default=2)
    parser.add_argument(
        "--bare-verification-mode",
        choices=("off", "report", "filter"),
        default="filter",
        help=(
            "Legacy global semantic-verification fallback. It is used only when a "
            "dataset-specific semantic mode is set to inherit."
        ),
    )
    parser.add_argument(
        "--opentouch-semantic-verification-mode",
        choices=("inherit", "off", "report", "filter"),
        default="filter",
        help=(
            "OpenTouch default: require independent positive glove evidence. "
            "Use inherit to fall back to --bare-verification-mode."
        ),
    )
    parser.add_argument(
        "--touchanything-semantic-verification-mode",
        choices=("inherit", "off", "report", "filter"),
        default="off",
        help=(
            "TouchAnything default: disable glove/bare verifier replay because both "
            "query hands are gloved. Native prompt score and temporal filtering remain active."
        ),
    )
    parser.add_argument(
        "--glove-verification-prompts",
        default="auto",
    )
    parser.add_argument(
        "--bare-verification-prompts",
        default="auto",
    )
    parser.add_argument(
        "--bare-verification-prompt",
        default=None,
        help="Legacy additional negative verifier prompt.",
    )
    parser.add_argument("--bare-match-iou-floor", type=float, default=0.70)
    parser.add_argument("--min-glove-verifier-fraction", type=float, default=0.10)
    parser.add_argument("--semantic-match-centroid-ratio", type=float, default=0.25)
    parser.add_argument("--max-bare-evidence-fraction", type=float, default=0.0)
    parser.add_argument(
        "--bare-rejection-policy",
        choices=("hard", "bare_only", "off"),
        default="off",
        help="How bare verifier evidence is converted into rejection; off keeps bare votes diagnostic-only.",
    )
    parser.add_argument("--allow-missing-prompt-score", action="store_true")
    parser.add_argument("--temporal-max-frame-gap", type=int, default=1)
    parser.add_argument("--temporal-center-residual-ratio", type=float, default=0.75)
    parser.add_argument("--temporal-area-ratio", type=float, default=3.0)
    parser.add_argument("--temporal-neighbor-iou-floor", type=float, default=0.10)
    parser.add_argument("--temporal-return-excursion-frames", type=int, default=0)
    parser.add_argument(
        "--flow-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add conservative bidirectional LK validation after SAM track selection.",
    )
    parser.add_argument(
        "--flow-bridge-policy",
        choices=("off", "short_bridge"),
        default="off",
    )
    parser.add_argument("--flow-max-gap", type=int, default=5)
    parser.add_argument("--flow-fb-error", type=float, default=1.5)
    parser.add_argument("--flow-min-points", type=int, default=12)
    parser.add_argument("--flow-min-inlier-ratio", type=float, default=0.60)
    parser.add_argument("--flow-min-confidence", type=float, default=0.45)
    parser.add_argument("--flow-sam-iou-accept", type=float, default=0.50)
    parser.add_argument("--flow-conflict-iou", type=float, default=0.15)
    parser.add_argument("--flow-cache-frames", type=int, default=16)
    preview_group = parser.add_mutually_exclusive_group()
    preview_group.add_argument("--mask-previews", dest="mask_previews", action="store_true")
    preview_group.add_argument("--no-mask-previews", dest="mask_previews", action="store_false")
    parser.set_defaults(mask_previews=None)
    rgb_group = parser.add_mutually_exclusive_group()
    rgb_group.add_argument(
        "--input-rgb-samples",
        dest="input_rgb_samples",
        action="store_true",
    )
    rgb_group.add_argument(
        "--no-input-rgb-samples",
        dest="input_rgb_samples",
        action="store_false",
    )
    parser.set_defaults(input_rgb_samples=None)
    parser.add_argument(
        "--semantic-debug",
        action="store_true",
        help="Write per-observation match evidence and raw verifier previews for pilot review.",
    )
    parser.add_argument(
        "--reload-predictor-per-job",
        action="store_true",
        help="Fallback for a SAM build that cannot reuse a predictor across videos; disabled by default.",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep decoded source frames on CPU to reduce long-video VRAM growth.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        choices=("auto", "always", "never"),
        default="auto",
    )
    parser.add_argument("--long-video-offload-frames", type=int, default=256)
    parser.add_argument("--video-chunk-frames", type=int, default=256)
    parser.add_argument("--video-chunk-overlap", type=int, default=32)
    parser.add_argument(
        "--chunk-staging-root",
        default="auto",
        help="auto uses /dev/shm for bounded per-session frame chunks.",
    )
    parser.add_argument("--chunk-jpeg-quality", type=int, default=95)
    parser.add_argument("--chunk-encode-workers", type=int, default=4)
    parser.add_argument(
        "--cache-staged-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--empty-cache-between-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--chunk-continuity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--chunk-carry-min-score", type=float, default=0.60)
    parser.add_argument("--chunk-carry-sessions", type=int, default=2)
    parser.add_argument(
        "--chunk-fragment-reentry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continuous-state-memory",
        choices=("native", "bounded"),
        default="native",
        help="Bound input and inference history while preserving one continuous session.",
    )
    parser.add_argument("--continuous-state-retain-frames", type=int, default=64)
    parser.add_argument("--continuous-state-log-interval", type=int, default=256)
    parser.add_argument("--continuous-input-cache-frames", type=int, default=4)
    parser.add_argument("--opentouch-redetect-frames", type=int, default=96)
    parser.add_argument("--opentouch-redetect-overlap", type=int, default=24)
    parser.add_argument(
        "--opentouch-materialization",
        choices=("stream", "lazy", "eager"),
        default="stream",
        help=(
            "stream copies only the active HDF5 chunk to RAM; lazy/eager retain the "
            "legacy per-sequence JPEG materialization paths."
        ),
    )
    parser.add_argument(
        "--keep-materialized-opentouch",
        action="store_true",
        help="Keep disposable OpenTouch JPEG directories after tracking.",
    )
    parser.add_argument(
        "--materialization-min-free-gb",
        type=float,
        default=1.0,
        help="Abort and clean the active OpenTouch sequence before free space drops below this value.",
    )
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mask_previews_enabled(args: argparse.Namespace) -> bool:
    value = getattr(args, "mask_previews", None)
    if value is not None:
        return bool(value)
    return not bool(getattr(args, "no_mask_previews", False))


def input_rgb_samples_enabled(args: argparse.Namespace) -> bool:
    value = getattr(args, "input_rgb_samples", None)
    return True if value is None else bool(value)


def resolve_storage_policy(args: argparse.Namespace) -> argparse.Namespace:
    """Use review artifacts for pilots and compact artifacts for full runs."""

    if args.materialization_min_free_gb < 0:
        raise ValueError("--materialization-min-free-gb must be >= 0")
    if not 1 <= getattr(args, "chunk_jpeg_quality", 95) <= 100:
        raise ValueError("--chunk-jpeg-quality must lie in [1,100]")
    if getattr(args, "chunk_encode_workers", 4) < 1:
        raise ValueError("--chunk-encode-workers must be positive")
    if not 0 <= getattr(args, "chunk_carry_min_score", 0.60) <= 1:
        raise ValueError("--chunk-carry-min-score must lie in [0,1]")
    if getattr(args, "chunk_carry_sessions", 2) < 1:
        raise ValueError("--chunk-carry-sessions must be positive")
    if getattr(args, "continuous_state_retain_frames", 64) < 1:
        raise ValueError("--continuous-state-retain-frames must be positive")
    if getattr(args, "continuous_state_log_interval", 256) < 0:
        raise ValueError("--continuous-state-log-interval must be >=0")
    if getattr(args, "continuous_input_cache_frames", 4) < 1:
        raise ValueError("--continuous-input-cache-frames must be positive")
    flow_max_gap = int(getattr(args, "flow_max_gap", 5))
    flow_fb_error = float(getattr(args, "flow_fb_error", 1.5))
    flow_min_points = int(getattr(args, "flow_min_points", 12))
    flow_min_inlier_ratio = float(getattr(args, "flow_min_inlier_ratio", 0.60))
    flow_min_confidence = float(getattr(args, "flow_min_confidence", 0.45))
    flow_sam_iou_accept = float(getattr(args, "flow_sam_iou_accept", 0.50))
    flow_conflict_iou = float(getattr(args, "flow_conflict_iou", 0.15))
    flow_cache_frames = int(getattr(args, "flow_cache_frames", 16))
    flow_bridge_policy = str(getattr(args, "flow_bridge_policy", "off"))
    flow_assist = bool(getattr(args, "flow_assist", False))
    if flow_max_gap < 0:
        raise ValueError("--flow-max-gap must be >=0")
    if flow_fb_error <= 0:
        raise ValueError("--flow-fb-error must be positive")
    if flow_min_points < 3:
        raise ValueError("--flow-min-points must be >=3")
    if not 0 <= flow_min_inlier_ratio <= 1:
        raise ValueError("--flow-min-inlier-ratio must lie in [0,1]")
    if not 0 <= flow_min_confidence <= 1:
        raise ValueError("--flow-min-confidence must lie in [0,1]")
    if not 0 <= flow_conflict_iou < flow_sam_iou_accept <= 1:
        raise ValueError(
            "Require 0 <= --flow-conflict-iou < --flow-sam-iou-accept <= 1"
        )
    if flow_cache_frames < 2:
        raise ValueError("--flow-cache-frames must be >=2")
    if flow_bridge_policy != "off" and not flow_assist:
        raise ValueError("--flow-bridge-policy requires --flow-assist")
    if (
        getattr(args, "continuous_state_memory", "native") == "bounded"
        and int(getattr(args, "video_chunk_frames", 0)) == 0
        and args.propagation_direction != "forward"
    ):
        raise ValueError(
            "bounded continuous state currently supports forward propagation only"
        )
    if getattr(args, "workers_per_gpu", 1) < 1:
        raise ValueError("--workers-per-gpu must be positive")
    if getattr(args, "cpu_threads_per_worker", 1) < 1:
        raise ValueError("--cpu-threads-per-worker must be positive")
    if args.mask_previews is None:
        args.mask_previews = not args.all_sequences
    if args.input_rgb_samples is None:
        args.input_rgb_samples = not args.all_sequences
    return args


def resolved_max_objects(args: argparse.Namespace, row: dict) -> int:
    expected = int(row["expected_gloved_hands"])
    return expected if args.max_objects <= 0 else min(args.max_objects, expected)


def resolved_sam_candidate_capacity(args: argparse.Namespace, row: dict) -> int:
    max_objects = resolved_max_objects(args, row)
    return max(4, max_objects * 2) if args.sam_candidate_capacity <= 0 else args.sam_candidate_capacity


def resolved_semantic_verification_mode(args: argparse.Namespace, row: dict) -> str:
    """Resolve the per-dataset semantic gate passed to one tracking job.

    OpenTouch can contain a nearby bare hand, so its default keeps the positive
    glove verifier. TouchAnything's two expected query hands are both gloved;
    replaying glove/bare text prompts there turns an unreliable semantic vote
    into an avoidable recall gate. The result is a detector policy only -- it
    never enters the tactile model or assigns a left/right pressure target.
    """

    if args.prompt_preset != "gloved":
        return "off"
    dataset = str(row.get("dataset", ""))
    if dataset == "opentouch":
        configured = args.opentouch_semantic_verification_mode
    elif dataset == "touchanything":
        configured = args.touchanything_semantic_verification_mode
    else:
        raise ValueError(f"Unsupported pilot dataset for semantic policy: {dataset!r}")
    return args.bare_verification_mode if configured == "inherit" else configured


def resolved_verifier_prompts(
    args: argparse.Namespace,
    semantic_verification_mode: str | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve the exact verifier lists persisted by track_video.py."""

    mode = (
        args.bare_verification_mode
        if semantic_verification_mode is None
        else semantic_verification_mode
    )
    if args.prompt_preset != "gloved" or mode == "off":
        return [], []
    glove, bare = resolve_verifier_prompt_lists(
        load_prompt_preset(args.prompt_preset),
        glove_value=args.glove_verification_prompts,
        bare_value=args.bare_verification_prompts,
        legacy_bare_prompt=args.bare_verification_prompt,
    )
    return list(glove), list(bare)


def complete(job_dir: Path, args: argparse.Namespace, row: dict) -> bool:
    if (job_dir / ".in_progress").exists():
        return False
    summary_path = job_dir / "summary.json"
    required_outputs = (job_dir / "bboxes.jsonl", job_dir / "track_audit.json")
    if not summary_path.is_file() or any(not path.is_file() for path in required_outputs):
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary.get("run_config", {})
        semantic_verification_mode = resolved_semantic_verification_mode(args, row)
        glove_verifiers, bare_verifiers = resolved_verifier_prompts(
            args,
            semantic_verification_mode,
        )
        expected = {
            "sam_version": args.sam_version,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
            "dataset": row["dataset"],
            "prompt_preset": args.prompt_preset,
            "propagation_direction": args.propagation_direction,
            "max_frames": args.max_frames,
            "max_objects": resolved_max_objects(args, row),
            "sam_candidate_capacity": resolved_sam_candidate_capacity(args, row),
            "min_mask_area_ratio": args.min_mask_area_ratio,
            "min_relative_mask_area": args.min_relative_mask_area,
            "min_prompt_score": args.min_prompt_score,
            "min_track_frames": args.min_track_frames,
            "duplicate_track_iou_floor": args.duplicate_track_iou_floor,
            "duplicate_track_overlap_fraction": args.duplicate_track_overlap_fraction,
            "duplicate_track_match_fraction": args.duplicate_track_match_fraction,
            "duplicate_track_centroid_ratio": args.duplicate_track_centroid_ratio,
            "duplicate_track_area_ratio": args.duplicate_track_area_ratio,
            "duplicate_track_min_frames": args.duplicate_track_min_frames,
            "require_prompt_score": not args.allow_missing_prompt_score,
            "bare_verification_mode": semantic_verification_mode,
            "glove_verification_prompts": glove_verifiers,
            "bare_verification_prompts": bare_verifiers,
            "bare_match_iou_floor": args.bare_match_iou_floor,
            "semantic_match_centroid_ratio": args.semantic_match_centroid_ratio,
            "min_glove_verifier_fraction": args.min_glove_verifier_fraction,
            "max_bare_evidence_fraction": args.max_bare_evidence_fraction,
            "bare_rejection_policy": args.bare_rejection_policy,
            "temporal_max_frame_gap": args.temporal_max_frame_gap,
            "temporal_center_residual_ratio": args.temporal_center_residual_ratio,
            "temporal_area_ratio": args.temporal_area_ratio,
            "temporal_neighbor_iou_floor": args.temporal_neighbor_iou_floor,
            "temporal_return_excursion_frames": args.temporal_return_excursion_frames,
            "flow_assist": bool(getattr(args, "flow_assist", False)),
            "flow_bridge_policy": getattr(args, "flow_bridge_policy", "off"),
            "flow_max_gap": getattr(args, "flow_max_gap", 5),
            "flow_fb_error": getattr(args, "flow_fb_error", 1.5),
            "flow_min_points": getattr(args, "flow_min_points", 12),
            "flow_min_inlier_ratio": getattr(args, "flow_min_inlier_ratio", 0.60),
            "flow_min_confidence": getattr(args, "flow_min_confidence", 0.45),
            "flow_sam_iou_accept": getattr(args, "flow_sam_iou_accept", 0.50),
            "flow_conflict_iou": getattr(args, "flow_conflict_iou", 0.15),
            "flow_cache_frames": getattr(args, "flow_cache_frames", 16),
            "semantic_debug": bool(args.semantic_debug),
            "offload_video_to_cpu": bool(args.offload_video_to_cpu),
            "offload_state_to_cpu": args.offload_state_to_cpu,
            "long_video_offload_frames": args.long_video_offload_frames,
            "video_chunk_frames": args.video_chunk_frames,
            "video_chunk_overlap": args.video_chunk_overlap,
            "chunk_staging_root": getattr(args, "chunk_staging_root", "auto"),
            "chunk_jpeg_quality": getattr(args, "chunk_jpeg_quality", 95),
            "chunk_encode_workers": getattr(args, "chunk_encode_workers", 4),
            "cache_staged_chunks": bool(getattr(args, "cache_staged_chunks", True)),
            "empty_cache_between_chunks": bool(
                getattr(args, "empty_cache_between_chunks", True)
            ),
            "chunk_continuity": bool(getattr(args, "chunk_continuity", True)),
            "chunk_continuity_version": CHUNK_CONTINUITY_VERSION,
            "chunk_carry_min_score": getattr(args, "chunk_carry_min_score", 0.60),
            "chunk_carry_sessions": getattr(args, "chunk_carry_sessions", 2),
            "chunk_fragment_reentry": bool(
                getattr(args, "chunk_fragment_reentry", True)
            ),
            "continuous_state_memory": getattr(
                args, "continuous_state_memory", "native"
            ),
            "continuous_state_retain_frames": getattr(
                args, "continuous_state_retain_frames", 64
            ),
            "continuous_state_log_interval": getattr(
                args, "continuous_state_log_interval", 256
            ),
            "continuous_input_cache_frames": getattr(
                args, "continuous_input_cache_frames", 4
            ),
            "opentouch_redetect_frames": args.opentouch_redetect_frames,
            "opentouch_redetect_overlap": args.opentouch_redetect_overlap,
        }
        scientific_match = summary.get("status") == "complete" and all(
            config.get(key) == value for key, value in expected.items()
        ) and (args.prompt is None or config.get("prompt") == args.prompt)
        preview_match = (
            not mask_previews_enabled(args) or config.get("mask_previews", True)
        )
        rgb_match = (
            not input_rgb_samples_enabled(args)
            or config.get("input_rgb_samples", True)
        )
        return scientific_match and preview_match and rgb_match
    except (OSError, json.JSONDecodeError):
        return False


def track_video_argv(args: argparse.Namespace, row: dict, job_dir: Path) -> list[str]:
    """Build a track_video CLI payload without creating a subprocess."""

    semantic_verification_mode = resolved_semantic_verification_mode(args, row)
    command = [
        "--resource",
        row["resource_path"],
        "--output-dir",
        str(job_dir),
        "--sam-version",
        args.sam_version,
        "--dataset",
        row["dataset"],
        "--expected-gloved-hands",
        str(row["expected_gloved_hands"]),
        "--prompt-preset",
        args.prompt_preset,
        "--propagation-direction",
        args.propagation_direction,
        "--max-frames",
        str(args.max_frames),
        "--max-objects",
        str(resolved_max_objects(args, row)),
        "--sam-candidate-capacity",
        str(resolved_sam_candidate_capacity(args, row)),
        "--min-mask-area-ratio",
        str(args.min_mask_area_ratio),
        "--min-relative-mask-area",
        str(args.min_relative_mask_area),
        "--min-prompt-score",
        str(args.min_prompt_score),
        "--min-track-frames",
        str(args.min_track_frames),
        "--duplicate-track-iou-floor",
        str(args.duplicate_track_iou_floor),
        "--duplicate-track-overlap-fraction",
        str(args.duplicate_track_overlap_fraction),
        "--duplicate-track-match-fraction",
        str(args.duplicate_track_match_fraction),
        "--duplicate-track-centroid-ratio",
        str(args.duplicate_track_centroid_ratio),
        "--duplicate-track-area-ratio",
        str(args.duplicate_track_area_ratio),
        "--duplicate-track-min-frames",
        str(args.duplicate_track_min_frames),
        "--bare-verification-mode",
        semantic_verification_mode,
        "--glove-verification-prompts",
        args.glove_verification_prompts,
        "--bare-verification-prompts",
        args.bare_verification_prompts,
        "--bare-match-iou-floor",
        str(args.bare_match_iou_floor),
        "--min-glove-verifier-fraction",
        str(args.min_glove_verifier_fraction),
        "--semantic-match-centroid-ratio",
        str(args.semantic_match_centroid_ratio),
        "--max-bare-evidence-fraction",
        str(args.max_bare_evidence_fraction),
        "--bare-rejection-policy",
        args.bare_rejection_policy,
        "--temporal-max-frame-gap",
        str(args.temporal_max_frame_gap),
        "--temporal-center-residual-ratio",
        str(args.temporal_center_residual_ratio),
        "--temporal-area-ratio",
        str(args.temporal_area_ratio),
        "--temporal-neighbor-iou-floor",
        str(args.temporal_neighbor_iou_floor),
        "--temporal-return-excursion-frames",
        str(args.temporal_return_excursion_frames),
        "--flow-bridge-policy",
        str(getattr(args, "flow_bridge_policy", "off")),
        "--flow-max-gap",
        str(getattr(args, "flow_max_gap", 5)),
        "--flow-fb-error",
        str(getattr(args, "flow_fb_error", 1.5)),
        "--flow-min-points",
        str(getattr(args, "flow_min_points", 12)),
        "--flow-min-inlier-ratio",
        str(getattr(args, "flow_min_inlier_ratio", 0.60)),
        "--flow-min-confidence",
        str(getattr(args, "flow_min_confidence", 0.45)),
        "--flow-sam-iou-accept",
        str(getattr(args, "flow_sam_iou_accept", 0.50)),
        "--flow-conflict-iou",
        str(getattr(args, "flow_conflict_iou", 0.15)),
        "--flow-cache-frames",
        str(getattr(args, "flow_cache_frames", 16)),
        "--offload-state-to-cpu",
        args.offload_state_to_cpu,
        "--long-video-offload-frames",
        str(args.long_video_offload_frames),
        "--video-chunk-frames",
        str(args.video_chunk_frames),
        "--video-chunk-overlap",
        str(args.video_chunk_overlap),
        "--chunk-staging-root",
        str(getattr(args, "chunk_staging_root", "auto")),
        "--chunk-jpeg-quality",
        str(getattr(args, "chunk_jpeg_quality", 95)),
        "--chunk-encode-workers",
        str(getattr(args, "chunk_encode_workers", 4)),
        "--chunk-carry-min-score",
        str(getattr(args, "chunk_carry_min_score", 0.60)),
        "--chunk-carry-sessions",
        str(getattr(args, "chunk_carry_sessions", 2)),
        "--continuous-state-memory",
        str(getattr(args, "continuous_state_memory", "native")),
        "--continuous-state-retain-frames",
        str(getattr(args, "continuous_state_retain_frames", 64)),
        "--continuous-state-log-interval",
        str(getattr(args, "continuous_state_log_interval", 256)),
        "--continuous-input-cache-frames",
        str(getattr(args, "continuous_input_cache_frames", 4)),
        "--opentouch-redetect-frames",
        str(args.opentouch_redetect_frames),
        "--opentouch-redetect-overlap",
        str(args.opentouch_redetect_overlap),
    ]
    if args.checkpoint is not None:
        command.extend(["--checkpoint", str(args.checkpoint)])
    command.append(
        "--cache-staged-chunks"
        if getattr(args, "cache_staged_chunks", True)
        else "--no-cache-staged-chunks"
    )
    if (
        row["dataset"] == "opentouch"
        and getattr(args, "opentouch_materialization", "lazy") == "stream"
    ):
        command.extend(["--hdf5-source", str(row["source_path"])])
    if args.prompt:
        command.extend(["--prompt", args.prompt])
    if args.bare_verification_prompt:
        command.extend(["--bare-verification-prompt", args.bare_verification_prompt])
    if args.allow_missing_prompt_score:
        command.append("--allow-missing-prompt-score")
    command.append(
        "--flow-assist" if getattr(args, "flow_assist", False) else "--no-flow-assist"
    )
    if not mask_previews_enabled(args):
        command.append("--no-mask-previews")
    if not input_rgb_samples_enabled(args):
        command.append("--no-input-rgb-samples")
    if args.offload_video_to_cpu:
        command.append("--offload-video-to-cpu")
    else:
        command.append("--no-offload-video-to-cpu")
    if args.semantic_debug:
        command.append("--semantic-debug")
    if getattr(args, "empty_cache_between_chunks", True):
        command.append("--empty-cache-between-chunks")
    else:
        command.append("--no-empty-cache-between-chunks")
    if getattr(args, "chunk_continuity", True):
        command.append("--chunk-continuity")
    else:
        command.append("--no-chunk-continuity")
    if getattr(args, "chunk_fragment_reentry", True):
        command.append("--chunk-fragment-reentry")
    else:
        command.append("--no-chunk-fragment-reentry")
    if args.overwrite:
        command.append("--overwrite")
    return command


def prepare_job(args: argparse.Namespace, row: dict, results_root: Path) -> dict:
    job_dir = results_root / row["dataset"] / row["split"] / row["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate a previous completion before any output can be overwritten. If the
    # process is interrupted, the marker forces the next run to rebuild this job.
    (job_dir / "summary.json").unlink(missing_ok=True)
    marker_path = job_dir / ".in_progress"
    marker_tmp = job_dir / f".in_progress.{os.getpid()}.{time.time_ns()}.tmp"
    marker_tmp.write_text(
        json.dumps(
            {
                "status": "in_progress",
                "pid": os.getpid(),
                "started_unix": time.time(),
                "sequence_key": row["sequence_key"],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    os.replace(marker_tmp, marker_path)
    return {
        "argv": track_video_argv(args, row, job_dir),
        "job_dir": str(job_dir),
        "log_path": str(job_dir / "run.log"),
        "row": row,
        "started": time.time(),
        "reload_predictor_per_job": bool(args.reload_predictor_per_job),
        "opentouch_materialization": args.opentouch_materialization,
        "cleanup_materialized_opentouch": not args.keep_materialized_opentouch,
        "materialization_min_free_gb": args.materialization_min_free_gb,
        "max_frames": args.max_frames,
    }


def prune_job_review_artifacts(
    records: list[dict],
    results_root: Path,
    args: argparse.Namespace,
) -> int:
    """Remove old review-only files when resuming in compact storage mode."""

    removed_bytes = 0
    for row in records:
        job_dir = results_root / row["dataset"] / row["split"] / row["job_id"]
        paths: list[Path] = []
        if not mask_previews_enabled(args):
            paths.extend((job_dir / "raw_sam_preview.mp4", job_dir / "preview.mp4"))
        if not input_rgb_samples_enabled(args):
            paths.append(job_dir / "input_rgb_samples")
        if not args.semantic_debug:
            paths.append(job_dir / "semantic_debug")
        for path in paths:
            if path.is_dir():
                for child in path.rglob("*"):
                    if child.is_file():
                        removed_bytes += child.stat().st_size
                shutil.rmtree(path)
            elif path.is_file():
                removed_bytes += path.stat().st_size
                path.unlink()
    return removed_bytes


def gpu_worker(
    gpu: str,
    worker_key: str,
    task_queue,
    result_queue,
    cpu_threads_per_worker: int,
) -> None:
    """Run many video jobs on one GPU while retaining SAM model weights."""

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    thread_count = str(cpu_threads_per_worker)
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = thread_count
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from pilot_manifest import cleanup_opentouch_materialization, materialize_opentouch_record
    from track_video import (
        SamPredictorRuntime,
        cleanup_staged_chunk_cache,
        parse_args as parse_tracking_args,
        run_tracking,
    )

    try:
        import cv2

        cv2.setNumThreads(cpu_threads_per_worker)
    except (ImportError, AttributeError):
        pass
    try:
        import torch

        torch.set_num_threads(cpu_threads_per_worker)
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass

    predictor_runtime = SamPredictorRuntime()
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            row = task["row"]
            log_path = Path(task["log_path"])
            return_code = 0
            error = None
            started = time.time()
            removed_materialization_bytes = 0
            removed_staged_cache_bytes = 0
            session_cleanup: dict[str, Any] = {}
            tracking_args = None
            try:
                with log_path.open("w", encoding="utf-8") as log_handle:
                    with redirect_stdout(log_handle), redirect_stderr(log_handle):
                        print(
                            f"[worker {worker_key} GPU {gpu}] {row['dataset']}/{row['split']} "
                            f"{row['sequence_key']}",
                            flush=True,
                        )
                        if (
                            row["dataset"] == "opentouch"
                            and task["opentouch_materialization"] == "lazy"
                        ):
                            materialize_opentouch_record(
                                row,
                                int(task["max_frames"]),
                                min_free_space_gb=float(task["materialization_min_free_gb"]),
                            )
                        predictor_runtime.begin_job()
                        tracking_args = parse_tracking_args(task["argv"])
                        run_tracking(tracking_args, predictor_runtime=predictor_runtime)
            except Exception:
                return_code = 1
                error = traceback.format_exc()
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write(error)
                    if not error.endswith("\n"):
                        log_handle.write("\n")
            finally:
                if tracking_args is not None:
                    try:
                        removed_staged_cache_bytes = cleanup_staged_chunk_cache(
                            tracking_args
                        )
                    except Exception:
                        cleanup_error = traceback.format_exc()
                        with log_path.open("a", encoding="utf-8") as log_handle:
                            log_handle.write("\nFailed to clean staged chunk cache:\n")
                            log_handle.write(cleanup_error)
                try:
                    session_cleanup = predictor_runtime.end_job()
                    with log_path.open("a", encoding="utf-8") as log_handle:
                        log_handle.write(
                            "\n[sam-runtime] end_job "
                            + json.dumps(session_cleanup, separators=(",", ":"))
                            + "\n"
                        )
                except Exception:
                    cleanup_error = traceback.format_exc()
                    session_cleanup = {"error": cleanup_error}
                    with log_path.open("a", encoding="utf-8") as log_handle:
                        log_handle.write("\nSAM end_job cleanup failed:\n")
                        log_handle.write(cleanup_error)
                if task["cleanup_materialized_opentouch"] and row["dataset"] == "opentouch":
                    try:
                        removed_materialization_bytes = cleanup_opentouch_materialization(row)
                    except Exception:
                        cleanup_error = traceback.format_exc()
                        with log_path.open("a", encoding="utf-8") as log_handle:
                            log_handle.write("\nFailed to clean OpenTouch materialization:\n")
                            log_handle.write(cleanup_error)
                        if return_code == 0:
                            return_code = 1
                            error = cleanup_error
                # Preserve the fast resident path after successful jobs, but do not
                # carry a possibly corrupted low-level SAM state past a failed one.
                if (
                    task["reload_predictor_per_job"]
                    or return_code != 0
                    or bool(session_cleanup.get("recycle_recommended"))
                    or bool(session_cleanup.get("error"))
                ):
                    predictor_runtime.close()
            result_queue.put(
                {
                    "gpu": gpu,
                    "worker_key": worker_key,
                    "row": row,
                    "return_code": return_code,
                    "error": error,
                    "elapsed_seconds": time.time() - started,
                    "log_path": str(log_path),
                    "runtime": predictor_runtime.audit(),
                    "session_cleanup": session_cleanup,
                    "removed_materialization_bytes": removed_materialization_bytes,
                    "removed_staged_cache_bytes": removed_staged_cache_bytes,
                }
            )
    finally:
        predictor_runtime.close()


def write_gallery(output_dir: Path, records: list[dict], results_root: Path) -> Path:
    sections = []
    for row in records:
        job_dir = results_root / row["dataset"] / row["split"] / row["job_id"]
        summary_path = job_dir / "summary.json"
        status = "missing"
        detail = ""
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                status = summary.get("status", "unknown")
                detail = f"nonempty={summary.get('nonempty_rate', 0.0):.3f}"
            except (OSError, json.JSONDecodeError):
                status = "invalid summary"
        raw_preview = job_dir / "raw_sam_preview.mp4"
        preview = job_dir / "preview.mp4"
        preview_html = ""
        if raw_preview.is_file():
            relative = os.path.relpath(raw_preview, output_dir)
            preview_html += (
                "<h4>Raw gloved-prompt SAM masks</h4>"
                f'<video controls preload="metadata" src="{html.escape(relative)}"></video>'
            )
        if preview.is_file():
            relative = os.path.relpath(preview, output_dir)
            preview_html += (
                "<h4>Accepted prompt-validated SAM masks</h4>"
                f'<video controls preload="metadata" src="{html.escape(relative)}"></video>'
            )
        sections.append(
            "<article>"
            f"<h3>{html.escape(row['dataset'])} / {html.escape(row['split'])}</h3>"
            f"<p>{html.escape(row['sequence_key'])}</p>"
            f"<p class='status'>{html.escape(status)} {html.escape(detail)}</p>"
            f"{preview_html}"
            "</article>"
        )
    page = """<!doctype html>
<html><head><meta charset="utf-8"><title>SAM3 bbox pilot</title>
<style>
body{font-family:Arial,sans-serif;margin:24px;background:#f4f5f7;color:#17191c}
h1{margin-bottom:4px}.note{color:#555;max-width:900px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
article{background:white;border:1px solid #d9dde3;border-radius:6px;padding:12px}h3{margin:0 0 6px;font-size:16px}p{margin:4px 0;overflow-wrap:anywhere}.status{font-family:monospace;color:#444}video{width:100%;margin-top:8px;background:#111}
</style></head><body>
<h1>SAM3 anonymous-query bbox pilot</h1>
<p class="note">Three sequences per split. Track IDs are anonymous and are not left/right pressure assignments.</p>
<div class="grid">""" + "\n".join(sections) + "</div></body></html>"
    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def last_log_line(path: Path, max_bytes: int = 4096) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
        lines = [part.strip() for part in text.replace("\r", "\n").splitlines() if part.strip()]
        return lines[-1][-180:] if lines else "starting"
    except OSError:
        return "starting"


def main() -> int:
    args = resolve_storage_policy(parse_args())
    selected_datasets = parse_dataset_selection(args.datasets)
    if args.output_dir is None:
        suffix = "_".join(selected_datasets)
        args.output_dir = Path(__file__).resolve().parent / "outputs" / f"pilot_{suffix}"
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.manifest_only:
        args.checkpoint = resolve_checkpoint(args.sam_version, args.checkpoint)
        print(f"Resolved {args.sam_version} checkpoint: {args.checkpoint}", flush=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    worker_specs = [
        (f"{gpu}:{worker_index}", gpu)
        for gpu in gpus
        for worker_index in range(args.workers_per_gpu)
    ]
    print(
        "Storage policy: "
        f"OpenTouch={args.opentouch_materialization}, "
        f"cleanup={not args.keep_materialized_opentouch}, "
        f"mask_previews={args.mask_previews}, "
        f"input_rgb_samples={args.input_rgb_samples}",
        flush=True,
    )

    manifest_path = build_manifest(
        opentouch_data_root=args.opentouch_data_root,
        opentouch_splits=args.opentouch_splits,
        touchanything_root=args.touchanything_root,
        touchanything_split_json=args.touchanything_split_json,
        output_dir=args.output_dir,
        samples_per_split=args.samples_per_split,
        samples_per_dataset=args.samples_per_dataset,
        all_sequences=args.all_sequences,
        seed=args.seed,
        max_frames=args.max_frames,
        materialize_opentouch=args.opentouch_materialization == "eager",
        datasets=selected_datasets,
        splits=args.splits,
    )
    records = read_jsonl(manifest_path)
    if (
        args.opentouch_materialization in {"stream", "lazy"}
        and not args.keep_materialized_opentouch
    ):
        removed = sum(cleanup_opentouch_materialization(row) for row in records)
        if removed:
            print(f"Cleaned {removed / 1024**3:.2f} GiB of stale OpenTouch inputs.", flush=True)
    if args.manifest_only:
        return 0
    active_semantic_modes = {
        resolved_semantic_verification_mode(args, row) for row in records
    }
    for semantic_verification_mode in sorted(active_semantic_modes):
        if args.prompt_preset != "gloved" or semantic_verification_mode != "filter":
            continue
        glove_verifiers, _ = resolved_verifier_prompts(args, semantic_verification_mode)
        if not glove_verifiers:
            raise ValueError(
                "A filter-mode dataset requires at least one positive glove verifier; "
                "set --glove-verification-prompts or disable that dataset's semantic filter."
            )
    results_root = args.output_dir / "results"
    pruned = prune_job_review_artifacts(records, results_root, args)
    if pruned:
        print(f"Pruned {pruned / 1024**3:.2f} GiB of review-only artifacts.", flush=True)
    pending = []
    skipped_count = 0
    for row in records:
        job_dir = results_root / row["dataset"] / row["split"] / row["job_id"]
        if complete(job_dir, args, row) and not args.overwrite:
            skipped_count += 1
        else:
            pending.append(row)
    if skipped_count:
        print(f"Resume: {skipped_count}/{len(records)} completed sequence jobs found.", flush=True)

    running: dict[str, dict] = {}
    failures = []
    completed = len(records) - len(pending)
    last_status = time.time()
    interrupted = False
    worker_context = mp.get_context("spawn")
    result_queue = worker_context.Queue()
    task_queues: dict[str, object] = {}
    workers: dict[str, object] = {}
    sequence_progress = progress(
        total=len(records),
        initial=skipped_count,
        desc="SAM sequence jobs",
        unit="seq",
    )
    if pending:
        print(
            f"Starting {len(worker_specs)} persistent SAM worker(s) across {len(gpus)} "
            "GPU(s); each worker loads model weights once.",
            flush=True,
        )
        for worker_key, gpu in worker_specs:
            task_queue = worker_context.Queue()
            worker = worker_context.Process(
                target=gpu_worker,
                args=(
                    gpu,
                    worker_key,
                    task_queue,
                    result_queue,
                    args.cpu_threads_per_worker,
                ),
                name=f"sam3-pilot-worker-{worker_key.replace(':', '-')}",
            )
            worker.start()
            task_queues[worker_key] = task_queue
            workers[worker_key] = worker
    try:
        while pending or running:
            for worker_key, gpu in worker_specs:
                if not pending or worker_key in running:
                    continue
                worker = workers[worker_key]
                if not worker.is_alive():
                    raise RuntimeError(
                        f"Persistent worker {worker_key} for GPU {gpu} exited before a job "
                        f"(exit code {worker.exitcode})."
                    )
                row = pending.pop(0)
                state = prepare_job(args, row, results_root)
                task_queues[worker_key].put(state)
                state["gpu"] = gpu
                state["worker_key"] = worker_key
                running[worker_key] = state
                sequence_progress.write(
                    f"[launch {worker_key}] {row['dataset']}/{row['split']} "
                    f"{row['sequence_key']} ({completed}/{len(records)} complete)",
                )
            if not running:
                continue
            try:
                event = result_queue.get(timeout=1.0)
            except queue.Empty:
                event = None
            if event is not None:
                gpu = str(event["gpu"])
                worker_key = str(event["worker_key"])
                state = running.pop(worker_key, None)
                if state is not None:
                    row = state["row"]
                    elapsed = float(event["elapsed_seconds"])
                    return_code = int(event["return_code"])
                    sequence_progress.update(1)
                    if return_code == 0:
                        completed += 1
                        runtime = event.get("runtime", {})
                        cuda_memory = runtime.get("cuda_memory_mb", {})
                        peak_text = ""
                        if "peak_allocated" in cuda_memory:
                            peak_text = f"; peak_vram={cuda_memory['peak_allocated'] / 1024:.1f}GiB"
                        sequence_progress.write(
                            f"[done {worker_key}] {row['dataset']}/{row['split']} "
                            f"{row['sequence_key']} in {elapsed / 60:.1f} min "
                            f"({completed}/{len(records)}; model_loads={runtime.get('load_count', '?')}"
                            f"{peak_text})"
                        )
                    else:
                        failures.append((row, Path(event["log_path"]), return_code))
                        sequence_progress.write(
                            f"[failed {worker_key}] {row['sequence_key']} rc={return_code}; "
                            f"see {event['log_path']}"
                        )
            for worker_key, worker in workers.items():
                if not worker.is_alive() and worker.exitcode is not None:
                    state = running.pop(worker_key, None)
                    if state is not None:
                        failures.append((state["row"], Path(state["log_path"]), worker.exitcode))
                    raise RuntimeError(
                        f"Persistent worker {worker_key} exited unexpectedly "
                        f"with code {worker.exitcode}."
                    )
            if running and time.time() - last_status >= 30:
                sequence_progress.write(
                    f"[status] {completed}/{len(records)} successful; {len(running)} active"
                )
                for worker_key, state in sorted(running.items()):
                    row = state["row"]
                    elapsed = (time.time() - state["started"]) / 60.0
                    sequence_progress.write(
                        f"  worker {worker_key} {row['dataset']}/{row['split']} "
                        f"{elapsed:.1f} min | {last_log_line(Path(state['log_path']))}"
                    )
                last_status = time.time()
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopping persistent pilot workers...", file=sys.stderr)
    finally:
        sequence_progress.close()
        for task_queue in task_queues.values():
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        for worker in workers.values():
            worker.join(timeout=15)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
        if not args.keep_materialized_opentouch:
            removed = sum(cleanup_opentouch_materialization(row) for row in records)
            if removed:
                print(
                    f"Cleaned {removed / 1024**3:.2f} GiB of OpenTouch temporary inputs.",
                    flush=True,
                )
    if interrupted:
        return 130

    gallery = write_gallery(args.output_dir, records, results_root)
    print(f"Pilot gallery: {gallery}")
    if failures:
        print(f"{len(failures)} job(s) failed:", file=sys.stderr)
        for row, path, code in failures:
            print(f"  rc={code} {row['sequence_key']}: {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
