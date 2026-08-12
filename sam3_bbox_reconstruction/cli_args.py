"""Shared command-line arguments for SAM video tracking workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import MappingProxyType

try:
    from .track_selection import BARE_REJECTION_POLICIES
except ImportError:  # Direct script execution.
    from track_selection import BARE_REJECTION_POLICIES


# Defaults shared by the pilot orchestrator and the single-video tracker live
# here exactly once. Dataset orchestration defaults remain in run_pilot.py.
TRACKING_DEFAULTS = MappingProxyType(
    {
        "sam_version": "sam3",
        "prompt_preset": "gloved",
        "propagation_direction": "forward",
        "max_frames": 0,
        "max_objects": 0,
        "sam_candidate_capacity": 0,
        "min_mask_area_ratio": 0.0005,
        "min_relative_mask_area": 0.05,
        "min_prompt_score": 0.5,
        "min_track_frames": 2,
        "duplicate_track_iou_floor": 0.80,
        "duplicate_track_overlap_fraction": 0.60,
        "duplicate_track_match_fraction": 0.80,
        "duplicate_track_centroid_ratio": 0.18,
        "duplicate_track_area_ratio": 1.50,
        "duplicate_track_min_frames": 2,
        "bare_verification_mode": "filter",
        "glove_verification_prompts": "auto",
        "bare_verification_prompts": "auto",
        "bare_verification_prompt": None,
        "bare_match_iou_floor": 0.70,
        "min_glove_verifier_fraction": 0.10,
        "semantic_match_centroid_ratio": 0.25,
        "max_bare_evidence_fraction": 0.0,
        "bare_rejection_policy": "off",
        "temporal_max_frame_gap": 1,
        "temporal_center_residual_ratio": 0.75,
        "temporal_area_ratio": 3.0,
        "temporal_neighbor_iou_floor": 0.10,
        "temporal_return_excursion_frames": 0,
        "flow_assist": False,
        "flow_bridge_policy": "off",
        "flow_max_gap": 5,
        "flow_fb_error": 1.5,
        "flow_min_points": 12,
        "flow_min_inlier_ratio": 0.60,
        "flow_min_confidence": 0.45,
        "flow_sam_iou_accept": 0.50,
        "flow_conflict_iou": 0.15,
        "flow_cache_frames": 16,
        "offload_video_to_cpu": True,
        "offload_state_to_cpu": "auto",
        "long_video_offload_frames": 256,
        "video_chunk_frames": 256,
        "video_chunk_overlap": 32,
        "chunk_staging_root": "auto",
        "chunk_jpeg_quality": 95,
        "chunk_encode_workers": 4,
        "cache_staged_chunks": True,
        "empty_cache_between_chunks": True,
        "chunk_continuity": True,
        "chunk_carry_min_score": 0.60,
        "chunk_carry_sessions": 2,
        "chunk_fragment_reentry": True,
        "continuous_state_memory": "native",
        "continuous_state_retain_frames": 64,
        "continuous_state_log_interval": 256,
        "continuous_input_cache_frames": 4,
        "opentouch_redetect_frames": 96,
        "opentouch_redetect_overlap": 24,
        "mask_previews": True,
        "input_rgb_samples": True,
    }
)


def add_tracking_arguments(
    parser: argparse.ArgumentParser,
    *,
    preview_policy: str,
) -> None:
    """Add options shared by run_pilot.py and track_video.py.

    ``preview_policy='pilot'`` keeps the historical tri-state preview defaults:
    the orchestrator resolves them from ``--all-sequences``. ``track`` retains
    the direct single-video defaults of writing both preview artifact types.
    """

    if preview_policy not in {"pilot", "track"}:
        raise ValueError(f"Unsupported preview policy: {preview_policy!r}")
    defaults = TRACKING_DEFAULTS

    parser.add_argument(
        "--sam-version",
        choices=("sam3", "sam3.1"),
        default=defaults["sam_version"],
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "Local SAM checkpoint. If omitted, resolve it from the shared repository "
            "_DATA directory."
        ),
    )
    parser.add_argument(
        "--prompt-preset",
        choices=("gloved", "bare"),
        default=defaults["prompt_preset"],
    )
    parser.add_argument("--prompt", help="Override the prompt preset's primary phrase")
    parser.add_argument(
        "--propagation-direction",
        choices=("forward", "both"),
        default=defaults["propagation_direction"],
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=defaults["max_frames"],
        help="0 tracks the complete sequence.",
    )
    parser.add_argument("--max-objects", type=int, default=defaults["max_objects"])
    parser.add_argument(
        "--sam-candidate-capacity",
        type=int,
        default=defaults["sam_candidate_capacity"],
        help="SAM3.1 candidate capacity; 0 resolves a safe capacity automatically.",
    )
    parser.add_argument(
        "--min-mask-area-ratio",
        type=float,
        default=defaults["min_mask_area_ratio"],
    )
    parser.add_argument(
        "--min-relative-mask-area",
        type=float,
        default=defaults["min_relative_mask_area"],
        help="Legacy compatibility option; not used for prompt-track identity selection.",
    )
    parser.add_argument(
        "--min-prompt-score",
        type=float,
        default=defaults["min_prompt_score"],
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=defaults["min_track_frames"],
    )
    parser.add_argument(
        "--duplicate-track-iou-floor",
        type=float,
        default=defaults["duplicate_track_iou_floor"],
    )
    parser.add_argument(
        "--duplicate-track-overlap-fraction",
        type=float,
        default=defaults["duplicate_track_overlap_fraction"],
    )
    parser.add_argument(
        "--duplicate-track-match-fraction",
        type=float,
        default=defaults["duplicate_track_match_fraction"],
    )
    parser.add_argument(
        "--duplicate-track-centroid-ratio",
        type=float,
        default=defaults["duplicate_track_centroid_ratio"],
    )
    parser.add_argument(
        "--duplicate-track-area-ratio",
        type=float,
        default=defaults["duplicate_track_area_ratio"],
    )
    parser.add_argument(
        "--duplicate-track-min-frames",
        type=int,
        default=defaults["duplicate_track_min_frames"],
    )
    parser.add_argument(
        "--bare-verification-mode",
        choices=("off", "report", "filter"),
        default=defaults["bare_verification_mode"],
    )
    parser.add_argument(
        "--glove-verification-prompts",
        default=defaults["glove_verification_prompts"],
    )
    parser.add_argument(
        "--bare-verification-prompts",
        default=defaults["bare_verification_prompts"],
    )
    parser.add_argument(
        "--bare-verification-prompt",
        default=defaults["bare_verification_prompt"],
        help="Legacy additional negative verifier prompt.",
    )
    parser.add_argument(
        "--bare-match-iou-floor",
        type=float,
        default=defaults["bare_match_iou_floor"],
    )
    parser.add_argument(
        "--min-glove-verifier-fraction",
        type=float,
        default=defaults["min_glove_verifier_fraction"],
    )
    parser.add_argument(
        "--semantic-match-centroid-ratio",
        type=float,
        default=defaults["semantic_match_centroid_ratio"],
    )
    parser.add_argument(
        "--max-bare-evidence-fraction",
        type=float,
        default=defaults["max_bare_evidence_fraction"],
    )
    parser.add_argument(
        "--bare-rejection-policy",
        choices=BARE_REJECTION_POLICIES,
        default=defaults["bare_rejection_policy"],
    )
    parser.add_argument("--allow-missing-prompt-score", action="store_true")
    parser.add_argument(
        "--temporal-max-frame-gap",
        type=int,
        default=defaults["temporal_max_frame_gap"],
    )
    parser.add_argument(
        "--temporal-center-residual-ratio",
        type=float,
        default=defaults["temporal_center_residual_ratio"],
    )
    parser.add_argument(
        "--temporal-area-ratio",
        type=float,
        default=defaults["temporal_area_ratio"],
    )
    parser.add_argument(
        "--temporal-neighbor-iou-floor",
        type=float,
        default=defaults["temporal_neighbor_iou_floor"],
    )
    parser.add_argument(
        "--temporal-return-excursion-frames",
        type=int,
        default=defaults["temporal_return_excursion_frames"],
    )
    parser.add_argument(
        "--flow-assist",
        action=argparse.BooleanOptionalAction,
        default=defaults["flow_assist"],
    )
    parser.add_argument(
        "--flow-bridge-policy",
        choices=("off", "short_bridge"),
        default=defaults["flow_bridge_policy"],
    )
    parser.add_argument("--flow-max-gap", type=int, default=defaults["flow_max_gap"])
    parser.add_argument("--flow-fb-error", type=float, default=defaults["flow_fb_error"])
    parser.add_argument("--flow-min-points", type=int, default=defaults["flow_min_points"])
    parser.add_argument(
        "--flow-min-inlier-ratio",
        type=float,
        default=defaults["flow_min_inlier_ratio"],
    )
    parser.add_argument(
        "--flow-min-confidence",
        type=float,
        default=defaults["flow_min_confidence"],
    )
    parser.add_argument(
        "--flow-sam-iou-accept",
        type=float,
        default=defaults["flow_sam_iou_accept"],
    )
    parser.add_argument(
        "--flow-conflict-iou",
        type=float,
        default=defaults["flow_conflict_iou"],
    )
    parser.add_argument(
        "--flow-cache-frames",
        type=int,
        default=defaults["flow_cache_frames"],
    )

    if preview_policy == "pilot":
        preview_group = parser.add_mutually_exclusive_group()
        preview_group.add_argument("--mask-previews", dest="mask_previews", action="store_true")
        preview_group.add_argument(
            "--no-mask-previews", dest="mask_previews", action="store_false"
        )
        parser.set_defaults(mask_previews=None)
        rgb_group = parser.add_mutually_exclusive_group()
        rgb_group.add_argument(
            "--input-rgb-samples", dest="input_rgb_samples", action="store_true"
        )
        rgb_group.add_argument(
            "--no-input-rgb-samples", dest="input_rgb_samples", action="store_false"
        )
        parser.set_defaults(input_rgb_samples=None)
    else:
        parser.add_argument(
            "--no-mask-previews",
            action="store_true",
            default=not bool(defaults["mask_previews"]),
        )
        parser.add_argument(
            "--input-rgb-samples",
            action=argparse.BooleanOptionalAction,
            default=defaults["input_rgb_samples"],
        )

    parser.add_argument("--semantic-debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=defaults["offload_video_to_cpu"],
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        choices=("auto", "always", "never"),
        default=defaults["offload_state_to_cpu"],
    )
    parser.add_argument(
        "--long-video-offload-frames",
        type=int,
        default=defaults["long_video_offload_frames"],
    )
    parser.add_argument(
        "--video-chunk-frames",
        type=int,
        default=defaults["video_chunk_frames"],
    )
    parser.add_argument(
        "--video-chunk-overlap",
        type=int,
        default=defaults["video_chunk_overlap"],
    )
    parser.add_argument(
        "--chunk-staging-root",
        default=defaults["chunk_staging_root"],
    )
    parser.add_argument(
        "--chunk-jpeg-quality",
        type=int,
        default=defaults["chunk_jpeg_quality"],
    )
    parser.add_argument(
        "--chunk-encode-workers",
        type=int,
        default=defaults["chunk_encode_workers"],
    )
    parser.add_argument(
        "--cache-staged-chunks",
        action=argparse.BooleanOptionalAction,
        default=defaults["cache_staged_chunks"],
    )
    parser.add_argument(
        "--empty-cache-between-chunks",
        action=argparse.BooleanOptionalAction,
        default=defaults["empty_cache_between_chunks"],
    )
    parser.add_argument(
        "--chunk-continuity",
        action=argparse.BooleanOptionalAction,
        default=defaults["chunk_continuity"],
    )
    parser.add_argument(
        "--chunk-carry-min-score",
        type=float,
        default=defaults["chunk_carry_min_score"],
    )
    parser.add_argument(
        "--chunk-carry-sessions",
        type=int,
        default=defaults["chunk_carry_sessions"],
    )
    parser.add_argument(
        "--chunk-fragment-reentry",
        action=argparse.BooleanOptionalAction,
        default=defaults["chunk_fragment_reentry"],
    )
    parser.add_argument(
        "--continuous-state-memory",
        choices=("native", "bounded"),
        default=defaults["continuous_state_memory"],
    )
    parser.add_argument(
        "--continuous-state-retain-frames",
        type=int,
        default=defaults["continuous_state_retain_frames"],
    )
    parser.add_argument(
        "--continuous-state-log-interval",
        type=int,
        default=defaults["continuous_state_log_interval"],
    )
    parser.add_argument(
        "--continuous-input-cache-frames",
        type=int,
        default=defaults["continuous_input_cache_frames"],
    )
    parser.add_argument(
        "--opentouch-redetect-frames",
        type=int,
        default=defaults["opentouch_redetect_frames"],
    )
    parser.add_argument(
        "--opentouch-redetect-overlap",
        type=int,
        default=defaults["opentouch_redetect_overlap"],
    )
