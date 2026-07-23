#!/usr/bin/env python3
"""Run local SAM3/SAM3.1 native video tracking and render anonymous hand tracks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

try:
    from .flow_assist import FlowAssistConfig, apply_optical_flow_assist
    from .progress import progress as progress_bar
    from .track_selection import (
        BARE_REJECTION_POLICIES,
        TrackObservation,
        attach_semantic_prompt_votes,
        bbox_iou,
        consolidate_duplicate_track_observations,
        filter_selected_tracks,
        select_prompt_tracks,
        stitch_overlapping_chunk_tracks,
    )
except ImportError:  # Direct script execution from run_pilot.py.
    from flow_assist import FlowAssistConfig, apply_optical_flow_assist
    from progress import progress as progress_bar
    from track_selection import (
        BARE_REJECTION_POLICIES,
        TrackObservation,
        attach_semantic_prompt_votes,
        bbox_iou,
        consolidate_duplicate_track_observations,
        filter_selected_tracks,
        select_prompt_tracks,
        stitch_overlapping_chunk_tracks,
    )


COLORS = (
    (58, 196, 255),
    (92, 224, 126),
    (255, 126, 92),
    (196, 112, 255),
    (255, 216, 92),
    (112, 224, 224),
)
_PRUNED_STAGING_ROOTS: set[Path] = set()
CHUNK_CONTINUITY_VERSION = "semantic_box_isolation_v3_fragment_reentry"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", type=Path, required=True, help="MP4 or JPEG frame directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=("opentouch", "touchanything", "generic"),
        default="generic",
    )
    parser.add_argument("--expected-gloved-hands", type=int, default=0)
    parser.add_argument("--sam-version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt-preset", choices=("gloved", "bare"), default="gloved")
    parser.add_argument("--prompt", help="Override the preset's primary prompt")
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument(
        "--propagation-direction",
        choices=("forward", "both"),
        default="forward",
        help="Forward is sufficient for the default frame-0 anchor; use both for a deliberate mid-video anchor.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 tracks the complete sequence")
    parser.add_argument("--max-objects", type=int, default=0)
    parser.add_argument(
        "--sam-candidate-capacity",
        type=int,
        default=0,
        help="SAM3.1 internal candidate capacity; 0 keeps at least four candidates before prompt-track selection.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mask-area-ratio", type=float, default=0.0005)
    parser.add_argument(
        "--min-prompt-score",
        type=float,
        default=0.5,
        help="Minimum native SAM text-prompt confidence required for a kept frame.",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=2,
        help="Minimum number of prompt-conformant frames before a SAM ID becomes a query.",
    )
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
            "Run independent glove and bare verifier prompts. filter requires "
            "positive glove evidence; bare votes are controlled separately by "
            "--bare-rejection-policy. report only records their votes."
        ),
    )
    parser.add_argument(
        "--glove-verification-prompts",
        default="auto",
        help=(
            "Comma-separated independent positive verifier prompts, or auto to use "
            "the preset's curated verifier subset. The primary preset is unchanged."
        ),
    )
    parser.add_argument(
        "--bare-verification-prompts",
        default="auto",
        help=(
            "Comma-separated bare-hand diagnostic prompts, or auto to use the "
            "preset's diagnostic subset. These are not a calibrated glove classifier."
        ),
    )
    parser.add_argument(
        "--bare-verification-prompt",
        default=None,
        help="Legacy additional negative verifier prompt; appended without changing preset order.",
    )
    parser.add_argument(
        "--bare-match-iou-floor",
        type=float,
        default=0.70,
        help="Minimum same-frame bbox IoU used to associate gloved and bare prompt tracks.",
    )
    parser.add_argument(
        "--min-glove-verifier-fraction",
        type=float,
        default=0.10,
        help="Minimum primary-track frame fraction with independent positive glove evidence.",
    )
    parser.add_argument(
        "--semantic-match-centroid-ratio",
        type=float,
        default=0.25,
        help="Maximum verifier/primary centroid distance relative to the smaller bbox diagonal.",
    )
    parser.add_argument(
        "--max-bare-evidence-fraction",
        type=float,
        default=0.0,
        help="Reject a whole locked track when effective bare evidence covers more than this fraction.",
    )
    parser.add_argument(
        "--bare-rejection-policy",
        choices=BARE_REJECTION_POLICIES,
        default="off",
        help=(
            "hard rejects every matched bare vote; bare_only rejects only bare evidence "
            "without an independent glove vote; off keeps bare votes diagnostic-only."
        ),
    )
    parser.add_argument(
        "--allow-missing-prompt-score",
        action="store_true",
        help="Diagnostic-only escape hatch when a nonstandard SAM build omits out_probs.",
    )
    parser.add_argument(
        "--temporal-max-frame-gap",
        type=int,
        default=1,
        help="Only compare immediately adjacent observations separated by at most this many frames.",
    )
    parser.add_argument(
        "--temporal-center-residual-ratio",
        type=float,
        default=0.75,
        help="Reject a middle-frame return jump above this fraction of neighbouring hand size.",
    )
    parser.add_argument(
        "--temporal-area-ratio",
        type=float,
        default=3.0,
        help="Reject an isolated return-to-normal mask-area spike above this ratio.",
    )
    parser.add_argument(
        "--temporal-neighbor-iou-floor",
        type=float,
        default=0.10,
        help="Neighbours must agree at least this much before a middle frame is called a return jump.",
    )
    parser.add_argument(
        "--temporal-return-excursion-frames",
        type=int,
        default=0,
        help="Optional legacy return-excursion filter; 0 keeps it disabled.",
    )
    parser.add_argument(
        "--flow-assist",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run bidirectional pyramidal-LK motion validation after SAM semantic "
            "selection. SAM remains authoritative and flow never creates a new track."
        ),
    )
    parser.add_argument(
        "--flow-bridge-policy",
        choices=("off", "short_bridge"),
        default="off",
        help="Optionally fill only short gaps whose forward/backward flow projections agree.",
    )
    parser.add_argument("--flow-max-gap", type=int, default=5)
    parser.add_argument("--flow-fb-error", type=float, default=1.5)
    parser.add_argument("--flow-min-points", type=int, default=12)
    parser.add_argument("--flow-min-inlier-ratio", type=float, default=0.60)
    parser.add_argument("--flow-min-confidence", type=float, default=0.45)
    parser.add_argument("--flow-sam-iou-accept", type=float, default=0.50)
    parser.add_argument("--flow-conflict-iou", type=float, default=0.15)
    parser.add_argument("--flow-cache-frames", type=int, default=16)
    parser.add_argument(
        "--min-relative-mask-area",
        type=float,
        default=0.05,
        help="Legacy compatibility option; no longer used to choose an identity by area.",
    )
    parser.add_argument("--preview-fps", type=float, default=0.0, help="0 preserves source FPS")
    parser.add_argument(
        "--no-mask-previews",
        action="store_true",
        help="Skip raw/final SAM mask videos; bbox JSONL and track audit are still written.",
    )
    parser.add_argument(
        "--input-rgb-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist three decoded RGB audit JPEGs; the numeric color audit is always written.",
    )
    parser.add_argument(
        "--semantic-debug",
        action="store_true",
        help=(
            "Write per-observation semantic match evidence and raw verifier-mask videos. "
            "Use for pilot diagnosis; it adds verifier replay work."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
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
        help="Offload temporal state; auto enables it for sufficiently long videos.",
    )
    parser.add_argument(
        "--long-video-offload-frames",
        type=int,
        default=256,
        help="Frame threshold used by automatic temporal-state CPU offload.",
    )
    parser.add_argument(
        "--video-chunk-frames",
        type=int,
        default=256,
        help="Maximum frames physically staged into one independent SAM session; 0 disables chunking.",
    )
    parser.add_argument(
        "--video-chunk-overlap",
        type=int,
        default=32,
        help="Overlap between long-video sessions for retrospective track stitching.",
    )
    parser.add_argument(
        "--chunk-staging-root",
        default="auto",
        help=(
            "Temporary root for physical frame chunks. auto prefers /dev/shm so chunk "
            "JPEGs use RAM rather than filesystem capacity."
        ),
    )
    parser.add_argument(
        "--chunk-jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality used only when a source MP4 is staged into bounded RAM chunks.",
    )
    parser.add_argument(
        "--chunk-encode-workers",
        type=int,
        default=4,
        help="Bounded CPU JPEG-encoding workers per GPU process for MP4 chunk staging.",
    )
    parser.add_argument(
        "--cache-staged-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse byte-identical RAM-staged chunks across primary/verifier/preview "
            "passes within one sequence job, then delete them."
        ),
    )
    parser.add_argument(
        "--empty-cache-between-chunks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return closed-session CUDA cache between chunks while retaining model weights.",
    )
    parser.add_argument(
        "--chunk-continuity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Seed each new chunk with high-confidence boxes from the preceding overlap "
            "while retaining the text prompt."
        ),
    )
    parser.add_argument(
        "--chunk-carry-min-score",
        type=float,
        default=0.60,
        help="Minimum native text-prompt score for a box carried into the next chunk.",
    )
    parser.add_argument(
        "--chunk-carry-sessions",
        type=int,
        default=2,
        help=(
            "Maximum independent SAM sessions used to carry distinct boxes across a "
            "chunk boundary. SAM3 permits only one initial visual box per session."
        ),
    )
    parser.add_argument(
        "--chunk-fragment-reentry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retain later prompt-conformant track fragments when they fit unused "
            "per-frame hand slots. This recovers a hand rediscovered after a chunk "
            "boundary without allowing excess same-frame boxes."
        ),
    )
    parser.add_argument(
        "--continuous-state-memory",
        choices=("native", "bounded"),
        default="native",
        help=(
            "Memory policy for an unchunked continuous SAM session. bounded drops "
            "already-emitted full-resolution caches and tracker frames that are no "
            "longer reachable by SAM's finite memory bank."
        ),
    )
    parser.add_argument(
        "--continuous-state-retain-frames",
        type=int,
        default=64,
        help=(
            "Minimum recent tracker frames retained in bounded continuous mode; "
            "architecture-selected high-quality memory frames are retained in addition."
        ),
    )
    parser.add_argument(
        "--continuous-state-log-interval",
        type=int,
        default=256,
        help="Report rolling-state/RSS statistics every N emitted frames; 0 disables logs.",
    )
    parser.add_argument(
        "--continuous-input-cache-frames",
        type=int,
        default=4,
        help=(
            "Number of decoded input frames retained by the bounded unchunked loader. "
            "Frames are decoded lazily from the original video or image directory."
        ),
    )
    parser.add_argument(
        "--hdf5-source",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--opentouch-redetect-frames",
        type=int,
        default=96,
        help="OpenTouch-only fresh text re-detection session length; 0 disables it.",
    )
    parser.add_argument(
        "--opentouch-redetect-overlap",
        type=int,
        default=24,
        help="Overlap used to reconnect OpenTouch fresh-detection sessions.",
    )
    return parser.parse_args(argv)


def load_prompt_preset(preset_name: str) -> dict[str, Any]:
    preset_path = Path(__file__).with_name("prompt_presets.json")
    presets = json.loads(preset_path.read_text(encoding="utf-8"))
    try:
        preset = presets[preset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt preset: {preset_name!r}") from exc
    if not isinstance(preset, dict):
        raise ValueError(f"Prompt preset {preset_name!r} must be a JSON object")
    return preset


def load_prompt(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    preset = load_prompt_preset(args.prompt_preset)
    return str(args.prompt or preset["primary"]), preset


def parse_prompt_list(value: str, *, option_name: str) -> tuple[str, ...]:
    """Split a comma-separated verifier list while preserving user order."""

    prompts: list[str] = []
    seen: set[str] = set()
    for item in value.split(","):
        prompt = item.strip()
        if not prompt:
            continue
        normalized = prompt.casefold()
        if normalized not in seen:
            prompts.append(prompt)
            seen.add(normalized)
    if not prompts:
        raise ValueError(f"{option_name} must contain at least one prompt")
    return tuple(prompts)


def resolve_verifier_prompt_lists(
    preset: dict[str, Any],
    *,
    glove_value: str,
    bare_value: str,
    legacy_bare_prompt: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve explicit verifier CLI values or the preset's curated subsets.

    The verifier set intentionally differs from ``candidates``: candidate
    prompts include colour- and glove-style-specific fallbacks that are useful
    for a manual primary-prompt ablation but would make an automatic semantic
    vote less precise.  ``auto`` therefore selects only the verification
    phrases declared in the preset.  For the known black-glove datasets this
    includes black-specific variants, but it deliberately excludes the
    primary prompt itself and negated bare-hand wording.
    """

    verifiers = preset.get("verifiers", {})
    if not isinstance(verifiers, dict):
        raise ValueError("prompt preset field 'verifiers' must be a JSON object")

    def resolve(value: str, preset_key: str, option_name: str) -> tuple[str, ...]:
        if value.strip().casefold() != "auto":
            return parse_prompt_list(value, option_name=option_name)
        configured = verifiers.get(preset_key, ())
        if not isinstance(configured, list) or not all(
            isinstance(item, str) for item in configured
        ):
            raise ValueError(
                f"prompt preset verifier field {preset_key!r} must be a list of strings"
            )
        return parse_prompt_list(
            ",".join(configured),
            option_name=f"preset verifiers.{preset_key}",
        )

    positive_key = "positive" if "positive" in verifiers else "glove"
    negative_key = (
        "negative_diagnostic"
        if "negative_diagnostic" in verifiers
        else "bare_diagnostic"
    )
    glove = resolve(glove_value, positive_key, "--glove-verification-prompts")
    bare = resolve(bare_value, negative_key, "--bare-verification-prompts")
    if legacy_bare_prompt:
        legacy_prompt = legacy_bare_prompt.strip()
        if legacy_prompt and legacy_prompt.casefold() not in {
            item.casefold() for item in bare
        }:
            bare = (*bare, legacy_prompt)
    return glove, bare


def tensor_numpy(value: Any):
    import numpy as np
    import torch

    if value is None:
        return np.asarray([])
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def response_outputs(response: dict[str, Any]) -> tuple[list[int], Any, list[float | None], tuple[str, ...], str | None]:
    import numpy as np

    outputs = response.get("outputs", response)
    object_ids = outputs.get("out_obj_ids", outputs.get("object_ids", []))
    masks = outputs.get("out_binary_masks", outputs.get("masks"))
    object_ids_array = tensor_numpy(object_ids).reshape(-1)
    masks_array = tensor_numpy(masks)
    if masks_array.size == 0:
        return [], np.asarray([]), [], tuple(sorted(outputs)), None
    while masks_array.ndim > 3 and masks_array.shape[1] == 1:
        masks_array = masks_array[:, 0]
    if masks_array.ndim == 2:
        masks_array = masks_array[None]
    ids = [int(value) for value in object_ids_array.tolist()]
    if len(ids) != len(masks_array):
        raise RuntimeError(
            "SAM response has masks without stable one-to-one object IDs; "
            "refusing frame-wise anonymous IDs because they cannot support video tracking."
        )
    score_key = next(
        (
            key
            for key in ("out_probs", "out_scores", "object_scores", "scores")
            if key in outputs
        ),
        None,
    )
    scores: list[float | None]
    if score_key is None:
        scores = [None] * len(ids)
    else:
        score_array = tensor_numpy(outputs[score_key]).reshape(-1)
        if len(score_array) != len(ids):
            scores = [None] * len(ids)
            score_key = f"{score_key}:shape_mismatch"
        else:
            scores = [float(value) for value in score_array.tolist()]
    return ids, masks_array, scores, tuple(sorted(outputs)), score_key


def mask_bbox(mask) -> list[int] | None:
    import numpy as np

    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_centroid(mask) -> tuple[float, float] | None:
    import numpy as np

    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return float(xs.mean()), float(ys.mean())


def largest_connected_component(mask):
    import cv2
    import numpy as np

    binary = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if component_count <= 1:
        return binary.astype(bool)
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def resize_mask_to_frame(mask, width: int, height: int):
    import cv2
    import numpy as np

    binary = np.asarray(mask, dtype=bool)
    if binary.shape == (height, width):
        return binary
    return cv2.resize(
        binary.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def resolve_max_objects(dataset: str, expected: int, requested: int) -> int:
    contracts = {"opentouch": 1, "touchanything": 2, "generic": 2}
    contract = expected if expected > 0 else contracts[dataset]
    if contract < 1:
        raise ValueError("expected_gloved_hands must be positive")
    return contract if requested <= 0 else min(requested, contract)


class FrameReader:
    def __init__(self, resource: Path, *, hdf5_source: str | None = None):
        import cv2

        self.resource = resource
        self.paths: list[Path] | None = None
        self.capture = None
        self.hdf5_handle = None
        self.hdf5_images = None
        self.last_index = -1
        if hdf5_source is not None:
            try:
                import h5py
            except ImportError as exc:
                raise RuntimeError("h5py is required for streamed OpenTouch chunks") from exc
            hdf5_path, dataset_key = hdf5_source.split("::", 1)
            self.hdf5_handle = h5py.File(hdf5_path, "r")
            if dataset_key not in self.hdf5_handle:
                self.hdf5_handle.close()
                raise KeyError(f"Missing {dataset_key} in {hdf5_path}")
            self.hdf5_images = self.hdf5_handle[dataset_key]
            self.length = len(self.hdf5_images)
            first = self._decode_hdf5_frame(0)
            if first is None:
                self.close()
                raise RuntimeError(f"Could not decode frame 0 from {hdf5_source}")
            self.width, self.height = first.shape[1], first.shape[0]
            self.fps = 30.0
        elif resource.is_dir():
            paths = []
            for pattern in ("*.jpg", "*.jpeg", "*.png"):
                paths.extend(resource.glob(pattern))
            self.paths = sorted(paths)
            if not self.paths:
                raise RuntimeError(f"No image frames found under {resource}")
            first = cv2.imread(str(self.paths[0]))
            if first is None:
                raise RuntimeError(f"Could not decode {self.paths[0]}")
            self.width, self.height = first.shape[1], first.shape[0]
            self.fps = 30.0
            self.length = len(self.paths)
        else:
            self.capture = cv2.VideoCapture(str(resource))
            if not self.capture.isOpened():
                raise RuntimeError(f"Could not open video {resource}")
            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = float(self.capture.get(cv2.CAP_PROP_FPS)) or 30.0
            self.length = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def _hdf5_encoded_frame(self, frame_index: int) -> bytes:
        if self.hdf5_images is None:
            raise RuntimeError("FrameReader has no HDF5 source")
        encoded = self.hdf5_images[frame_index]
        return encoded.tobytes() if hasattr(encoded, "tobytes") else bytes(encoded)

    def _decode_hdf5_frame(self, frame_index: int):
        import cv2
        import numpy as np

        encoded = self._hdf5_encoded_frame(frame_index)
        return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)

    def get(self, frame_index: int):
        import cv2

        if self.hdf5_images is not None:
            if frame_index < 0 or frame_index >= self.length:
                return None
            return self._decode_hdf5_frame(frame_index)
        if self.paths is not None:
            if frame_index < 0 or frame_index >= len(self.paths):
                return None
            return cv2.imread(str(self.paths[frame_index]))
        assert self.capture is not None
        if frame_index != self.last_index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            return None
        self.last_index = frame_index
        return frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.hdf5_handle is not None:
            self.hdf5_handle.close()
            self.hdf5_handle = None
            self.hdf5_images = None


def open_frame_reader(args: argparse.Namespace) -> FrameReader:
    return FrameReader(args.resource, hdf5_source=getattr(args, "hdf5_source", None))


def audit_input_color(
    resource: Path,
    output_dir: Path,
    *,
    write_samples: bool = True,
    hdf5_source: str | None = None,
) -> dict[str, Any]:
    """Verify color decoding and optionally persist three unmasked RGB samples."""

    import cv2
    import numpy as np

    reader = FrameReader(resource, hdf5_source=hdf5_source)
    sample_dir = output_dir / "input_rgb_samples"
    if write_samples:
        sample_dir.mkdir(parents=True, exist_ok=True)
    else:
        shutil.rmtree(sample_dir, ignore_errors=True)
    indices = sorted({0, max(0, reader.length // 2), max(0, reader.length - 1)})
    samples = []
    try:
        for frame_index in indices:
            frame_bgr = reader.get(frame_index)
            if frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            sample_path = None
            if write_samples:
                sample_path = sample_dir / f"frame_{frame_index:08d}.jpg"
                if not cv2.imwrite(str(sample_path), frame_bgr):
                    raise RuntimeError(f"Could not write RGB audit sample: {sample_path}")
            pil_rgb_mae = None
            if reader.paths is not None:
                try:
                    from PIL import Image

                    pil_rgb = np.asarray(Image.open(reader.paths[frame_index]).convert("RGB"))
                    if pil_rgb.shape == frame_rgb.shape:
                        pil_rgb_mae = float(
                            np.abs(pil_rgb.astype(np.float32) - frame_rgb.astype(np.float32)).mean()
                        )
                except (ImportError, OSError):
                    pass
            samples.append(
                {
                    "frame_index": frame_index,
                    "sample_path": str(sample_path) if sample_path is not None else None,
                    "rgb_mean": [float(value) for value in frame_rgb.mean(axis=(0, 1))],
                    "rgb_std": [float(value) for value in frame_rgb.std(axis=(0, 1))],
                    "opencv_vs_pil_rgb_mae": pil_rgb_mae,
                }
            )
    finally:
        reader.close()
    audit = {
        "resource": str(resource),
        "frame_count": reader.length,
        "sample_images_written": bool(write_samples),
        "decode_contract": (
            "FrameReader/OpenCV yields BGR; SAM reads the original resource; preview writers "
            "receive BGR; RGB statistics explicitly use COLOR_BGR2RGB"
        ),
        "samples": samples,
    }
    audit_path = output_dir / "input_color_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit["audit_path"] = str(audit_path)
    return audit


def start_session_request(
    args: argparse.Namespace,
    reader: FrameReader,
    *,
    resource_path: Path | None = None,
) -> dict[str, Any]:
    if args.offload_state_to_cpu == "always":
        offload_state = True
    elif args.offload_state_to_cpu == "never":
        offload_state = False
    else:
        offload_state = reader.length >= args.long_video_offload_frames
    return {
        "type": "start_session",
        "resource_path": str(args.resource if resource_path is None else resource_path),
        "offload_video_to_cpu": bool(args.offload_video_to_cpu),
        "offload_state_to_cpu": bool(offload_state),
    }


def iter_propagation(
    predictor,
    session_id: str,
    propagation_direction: str,
    *,
    start_frame_index: int | None = None,
    max_frame_num_to_track: int | None = None,
    state_compactor: "ContinuousSessionStateCompactor | None" = None,
) -> Iterator[dict[str, Any]]:
    request = {
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": propagation_direction,
    }
    if start_frame_index is not None:
        request["start_frame_index"] = int(start_frame_index)
    if max_frame_num_to_track is not None:
        request["max_frame_num_to_track"] = int(max_frame_num_to_track)
    for response in predictor.handle_stream_request(request):
        if state_compactor is not None:
            state_compactor.compact_after_emit(int(response.get("frame_index", -1)))
        yield response


def process_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except (ImportError, OSError):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return float(resident_pages * page_size / (1024**2))
        except (OSError, ValueError, IndexError):
            return 0.0


class BoundedVideoFrameLoader:
    """Sequential lazy decoder with a small LRU instead of a dense T-frame tensor."""

    def __init__(
        self,
        video_path: str,
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean: tuple[float, float, float],
        img_std: tuple[float, float, float],
        *,
        cache_frames: int,
    ) -> None:
        import cv2
        import torch

        self.video_path = str(video_path)
        self.image_size = int(image_size)
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.cache_frames = max(1, int(cache_frames))
        self._cv2 = cv2
        self._torch = torch
        self._capture = cv2.VideoCapture(self.video_path)
        if not self._capture.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        self.video_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._length = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._length <= 0:
            self.close()
            raise RuntimeError(
                f"Bounded continuous loading requires a finite frame count: {self.video_path}"
            )
        self._mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        self._std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self._next_decode_index = 0
        self.decoded_frame_count = 0
        self.seek_count = 0
        self.max_cached_frames = 0
        # Session initialization and prompt insertion normally request frame zero.
        # Decode it now so dimensions and decoder validity fail before GPU work starts.
        self[0]

    def __len__(self) -> int:
        return self._length

    def _decode(self, index: int):
        cv2 = self._cv2
        if index != self._next_decode_index:
            if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise RuntimeError(
                    f"Could not seek {self.video_path} to frame {index}"
                )
            self._next_decode_index = index
            self.seek_count += 1
        ok, frame_bgr = self._capture.read()
        if not ok:
            raise RuntimeError(
                f"Could not decode frame {index}/{self._length} from {self.video_path}"
            )
        self._next_decode_index = index + 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(
            frame_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        tensor = self._torch.from_numpy(frame_rgb).permute(2, 0, 1)
        tensor = tensor.to(dtype=self._torch.float16).div_(255.0)
        tensor.sub_(self._mean).div_(self._std)
        if not self.offload_video_to_cpu:
            tensor = tensor.cuda(non_blocking=True)
        self.decoded_frame_count += 1
        return tensor

    def __getitem__(self, index: int):
        if hasattr(index, "item"):
            index = index.item()
        index = int(index)
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(index)
        cached = self._cache.pop(index, None)
        if cached is None:
            cached = self._decode(index)
        self._cache[index] = cached
        while len(self._cache) > self.cache_frames:
            self._cache.popitem(last=False)
        self.max_cached_frames = max(self.max_cached_frames, len(self._cache))
        return cached

    def audit(self) -> dict[str, Any]:
        return {
            "type": "bounded_cv2_video",
            "frame_count": self._length,
            "cache_limit": self.cache_frames,
            "cached_frames": list(self._cache),
            "max_cached_frames": self.max_cached_frames,
            "decoded_frame_count": self.decoded_frame_count,
            "seek_count": self.seek_count,
        }

    def close(self) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()
            self._capture = None
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.clear()

    def __del__(self) -> None:
        self.close()


class BoundedImageFrameLoader:
    """Lazy image-folder loader that never starts SAM's eager background preload."""

    def __init__(
        self,
        img_paths: list[str],
        image_size: int,
        offload_video_to_cpu: bool,
        img_mean,
        img_std,
        *,
        cache_frames: int,
    ) -> None:
        import cv2
        import torch

        self.img_paths = list(img_paths)
        self.image_size = int(image_size)
        self.offload_video_to_cpu = bool(offload_video_to_cpu)
        self.cache_frames = max(1, int(cache_frames))
        self._cv2 = cv2
        self._torch = torch
        self._mean = img_mean.to(dtype=torch.float16, device="cpu")
        self._std = img_std.to(dtype=torch.float16, device="cpu")
        self._cache: OrderedDict[int, Any] = OrderedDict()
        self.decoded_frame_count = 0
        self.max_cached_frames = 0
        first = cv2.imread(self.img_paths[0], cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Could not decode image: {self.img_paths[0]}")
        self.video_height, self.video_width = first.shape[:2]
        self[0]

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int):
        if hasattr(index, "item"):
            index = index.item()
        index = int(index)
        if index < 0:
            index += len(self.img_paths)
        if not 0 <= index < len(self.img_paths):
            raise IndexError(index)
        cached = self._cache.pop(index, None)
        if cached is None:
            frame_bgr = self._cv2.imread(self.img_paths[index], self._cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Could not decode image: {self.img_paths[index]}")
            frame_rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
            frame_rgb = self._cv2.resize(
                frame_rgb,
                (self.image_size, self.image_size),
                interpolation=self._cv2.INTER_CUBIC,
            )
            cached = self._torch.from_numpy(frame_rgb).permute(2, 0, 1)
            cached = cached.to(dtype=self._torch.float16).div_(255.0)
            cached.sub_(self._mean).div_(self._std)
            if not self.offload_video_to_cpu:
                cached = cached.cuda(non_blocking=True)
            self.decoded_frame_count += 1
        self._cache[index] = cached
        while len(self._cache) > self.cache_frames:
            self._cache.popitem(last=False)
        self.max_cached_frames = max(self.max_cached_frames, len(self._cache))
        return cached

    def audit(self) -> dict[str, Any]:
        return {
            "type": "bounded_image_folder",
            "frame_count": len(self.img_paths),
            "cache_limit": self.cache_frames,
            "cached_frames": list(self._cache),
            "max_cached_frames": self.max_cached_frames,
            "decoded_frame_count": self.decoded_frame_count,
        }


_SAM_NATIVE_INPUT_LOADERS: dict[str, Any] = {}


def configure_continuous_input_loader(
    args: argparse.Namespace,
    *,
    bounded: bool | None = None,
) -> None:
    """Select native or bounded SAM input loading for the next session."""

    if bounded is None:
        bounded = getattr(args, "continuous_state_memory", "native") == "bounded"
    if not bounded and not _SAM_NATIVE_INPUT_LOADERS:
        return
    from sam3.model import io_utils

    if not _SAM_NATIVE_INPUT_LOADERS:
        _SAM_NATIVE_INPUT_LOADERS.update(
            {
                "video": io_utils.load_video_frames_from_video_file_using_cv2,
                "images": io_utils.AsyncImageFrameLoader,
            }
        )
    if not bounded:
        io_utils.load_video_frames_from_video_file_using_cv2 = (
            _SAM_NATIVE_INPUT_LOADERS["video"]
        )
        io_utils.AsyncImageFrameLoader = _SAM_NATIVE_INPUT_LOADERS["images"]
        return

    cache_frames = max(1, int(getattr(args, "continuous_input_cache_frames", 4)))

    def load_bounded_video(
        video_path,
        image_size,
        img_mean=(0.5, 0.5, 0.5),
        img_std=(0.5, 0.5, 0.5),
        offload_video_to_cpu=False,
    ):
        loader = BoundedVideoFrameLoader(
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            cache_frames=cache_frames,
        )
        return loader, loader.video_height, loader.video_width

    class ConfiguredBoundedImageFrameLoader(BoundedImageFrameLoader):
        def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
            super().__init__(
                img_paths,
                image_size,
                offload_video_to_cpu,
                img_mean,
                img_std,
                cache_frames=cache_frames,
            )

    io_utils.load_video_frames_from_video_file_using_cv2 = load_bounded_video
    io_utils.AsyncImageFrameLoader = ConfiguredBoundedImageFrameLoader


class ContinuousSessionStateCompactor:
    """Bound one-pass SAM state without breaking a continuous video session.

    SAM retains full-resolution masks for later interactive fetches and every
    historical tracker output. Our reconstruction consumes each frame exactly
    once. After a response is converted to NumPy, those fetch caches can be
    dropped. Tracker outputs are retained only while they can still be selected
    by the model's finite spatial-memory/object-pointer windows.
    """

    def __init__(
        self,
        predictor,
        session_id: str,
        *,
        retain_frames: int,
        log_interval: int,
    ) -> None:
        self.predictor = predictor
        self.session_id = str(session_id)
        self.retain_frames = max(1, int(retain_frames))
        self.log_interval = max(0, int(log_interval))
        self.emitted_frames = 0
        self.cached_output_evictions = 0
        self.tracker_output_evictions = 0
        self.max_cached_outputs = 0
        self.max_tracker_outputs = 0
        self.last_snapshot: dict[str, Any] = {}

    def _state(self) -> dict[str, Any] | None:
        sessions = getattr(self.predictor, "_all_inference_states", None)
        if not isinstance(sessions, dict):
            return None
        session = sessions.get(self.session_id)
        if not isinstance(session, dict):
            return None
        state = session.get("state")
        return state if isinstance(state, dict) else None

    @staticmethod
    def _score(value: Any) -> float | None:
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "float"):
                value = value.float()
            if hasattr(value, "mean"):
                value = value.mean()
            if hasattr(value, "item"):
                value = value.item()
            result = float(value)
            return result if result == result else None
        except (TypeError, ValueError, RuntimeError):
            return None

    def _retained_noncond_frames(
        self,
        tracker,
        outputs: dict[int, dict[str, Any]],
    ) -> set[int]:
        frame_indices = sorted(int(index) for index in outputs)
        if not frame_indices:
            return set()
        max_obj_ptrs = max(1, int(getattr(tracker, "max_obj_ptrs_in_encoder", 16)))
        num_maskmem = max(1, int(getattr(tracker, "num_maskmem", 7)))
        stride = max(1, int(getattr(tracker, "memory_temporal_stride_for_eval", 1)))
        recent_count = max(
            self.retain_frames,
            max_obj_ptrs,
            num_maskmem * stride + 2,
        )
        retained = set(frame_indices[-recent_count:])
        if bool(getattr(tracker, "use_memory_selection", False)):
            threshold = float(getattr(tracker, "mf_threshold", 0.0))
            qualifying = []
            for frame_index in frame_indices:
                score = self._score(outputs[frame_index].get("eff_iou_score"))
                if score is not None and score > threshold:
                    qualifying.append(frame_index)
            retained.update(qualifying[-max_obj_ptrs:])
        return retained

    @staticmethod
    def _retained_cond_frames(
        tracker,
        outputs: dict[int, dict[str, Any]],
    ) -> set[int]:
        frame_indices = sorted(int(index) for index in outputs)
        if not frame_indices:
            return set()
        configured_limit = int(getattr(tracker, "max_cond_frames_in_attn", 1))
        if configured_limit < 0:
            return set(frame_indices)
        limit = max(1, configured_limit)
        retained = set(frame_indices[-limit:])
        if bool(getattr(tracker, "keep_first_cond_frame", False)):
            retained.add(frame_indices[0])
        return retained

    @staticmethod
    def _prune_frame_mapping(mapping: Any, retained: set[int]) -> int:
        if not isinstance(mapping, dict):
            return 0
        removed = 0
        for frame_index in list(mapping):
            if int(frame_index) not in retained:
                mapping.pop(frame_index, None)
                removed += 1
        return removed

    @classmethod
    def _prune_per_object_frame_mappings(
        cls,
        mappings: Any,
        retained: set[int],
    ) -> int:
        """Prune SAM point/mask inputs without violating preflight invariants."""

        if isinstance(mappings, dict):
            per_object = mappings.values()
        elif isinstance(mappings, (list, tuple)):
            per_object = mappings
        else:
            return 0
        removed = 0
        for frame_mapping in per_object:
            removed += cls._prune_frame_mapping(frame_mapping, retained)
        return removed

    @staticmethod
    def _per_object_input_frame_indices(tracker_state: dict[str, Any]) -> set[int]:
        result: set[int] = set()
        for key in ("point_inputs_per_obj", "mask_inputs_per_obj"):
            mappings = tracker_state.get(key)
            if isinstance(mappings, dict):
                per_object = mappings.values()
            elif isinstance(mappings, (list, tuple)):
                per_object = mappings
            else:
                continue
            for frame_mapping in per_object:
                if isinstance(frame_mapping, dict):
                    result.update(int(index) for index in frame_mapping)
        return result

    @staticmethod
    def _pending_temp_frame_indices(tracker_state: dict[str, Any]) -> set[int]:
        result: set[int] = set()
        temp_per_object = tracker_state.get("temp_output_dict_per_obj", {})
        if not isinstance(temp_per_object, dict):
            return result
        for object_outputs in temp_per_object.values():
            if not isinstance(object_outputs, dict):
                continue
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                frame_mapping = object_outputs.get(storage_key)
                if isinstance(frame_mapping, dict):
                    result.update(int(index) for index in frame_mapping)
        return result

    def compact_after_emit(self, emitted_frame_index: int) -> None:
        state = self._state()
        if state is None or emitted_frame_index < 0:
            return
        self.emitted_frames += 1

        cached_outputs = state.get("cached_frame_outputs")
        if isinstance(cached_outputs, dict):
            self.max_cached_outputs = max(self.max_cached_outputs, len(cached_outputs))
            for frame_index in list(cached_outputs):
                if int(frame_index) <= emitted_frame_index:
                    cached_outputs.pop(frame_index, None)
                    self.cached_output_evictions += 1

        tracker = getattr(getattr(self.predictor, "model", None), "tracker", None)
        tracker_states = state.get("tracker_inference_states", [])
        retained_total = 0
        for tracker_state in tracker_states if isinstance(tracker_states, list) else []:
            if not isinstance(tracker_state, dict):
                continue
            output_dict = tracker_state.get("output_dict", {})
            cond_outputs = output_dict.get("cond_frame_outputs", {})
            noncond_outputs = output_dict.get("non_cond_frame_outputs", {})
            retained_cond = self._retained_cond_frames(tracker, cond_outputs)
            retained_noncond = self._retained_noncond_frames(tracker, noncond_outputs)
            consolidated = tracker_state.get("consolidated_frame_inds", {})
            cond_set = (
                consolidated.get("cond_frame_outputs")
                if isinstance(consolidated, dict)
                else None
            )
            noncond_set = (
                consolidated.get("non_cond_frame_outputs")
                if isinstance(consolidated, dict)
                else None
            )
            # SAM requires the union of point/mask input frame indices to equal
            # the union of consolidated frame indices. Reconditioning creates
            # non-conditioning inputs during propagation, so pruning only the
            # output dictionaries eventually trips its preflight assertion.
            has_input_mappings = any(
                key in tracker_state
                for key in ("point_inputs_per_obj", "mask_inputs_per_obj")
            )
            pending_temp_frames = self._pending_temp_frame_indices(tracker_state)
            retained_input_frames = set()
            if has_input_mappings:
                if isinstance(cond_set, set):
                    retained_input_frames.update(
                        int(index) for index in cond_set & retained_cond
                    )
                if isinstance(noncond_set, set):
                    retained_input_frames.update(
                        int(index) for index in noncond_set & retained_noncond
                    )
                retained_input_frames.update(pending_temp_frames)
            self.tracker_output_evictions += self._prune_frame_mapping(
                cond_outputs, retained_cond
            )
            self.tracker_output_evictions += self._prune_frame_mapping(
                noncond_outputs, retained_noncond
            )
            retained_total += len(cond_outputs) + len(noncond_outputs)

            per_object = tracker_state.get("output_dict_per_obj", {})
            if isinstance(per_object, dict):
                for object_outputs in per_object.values():
                    if not isinstance(object_outputs, dict):
                        continue
                    self._prune_frame_mapping(
                        object_outputs.get("cond_frame_outputs"), retained_cond
                    )
                    self._prune_frame_mapping(
                        object_outputs.get("non_cond_frame_outputs"), retained_noncond
                    )
            if has_input_mappings:
                self.tracker_output_evictions += self._prune_per_object_frame_mappings(
                    tracker_state.get("point_inputs_per_obj"), retained_input_frames
                )
                self.tracker_output_evictions += self._prune_per_object_frame_mappings(
                    tracker_state.get("mask_inputs_per_obj"), retained_input_frames
                )
            if isinstance(consolidated, dict):
                if isinstance(cond_set, set):
                    cond_set.intersection_update(retained_cond)
                if isinstance(noncond_set, set):
                    noncond_set.intersection_update(retained_noncond)
            if has_input_mappings:
                input_frames_after = self._per_object_input_frame_indices(tracker_state)
                consolidated_after = set()
                if isinstance(cond_set, set):
                    consolidated_after.update(int(index) for index in cond_set)
                if isinstance(noncond_set, set):
                    consolidated_after.update(int(index) for index in noncond_set)
                expected_inputs = consolidated_after | pending_temp_frames
                if input_frames_after != expected_inputs:
                    raise RuntimeError(
                        "Bounded SAM state compaction would violate tracker preflight: "
                        f"input_only={sorted(input_frames_after - expected_inputs)[:8]}, "
                        "consolidated_only="
                        f"{sorted(expected_inputs - input_frames_after)[:8]}"
                    )
            frames_already_tracked = tracker_state.get("frames_already_tracked")
            if isinstance(frames_already_tracked, dict):
                for frame_index in list(frames_already_tracked):
                    if int(frame_index) not in retained_cond | retained_noncond:
                        frames_already_tracked.pop(frame_index, None)

        tracker_metadata = state.get("tracker_metadata", {})
        if isinstance(tracker_metadata, dict):
            frame_scores = tracker_metadata.get("obj_id_to_tracker_score_frame_wise")
            if isinstance(frame_scores, dict):
                for frame_index in list(frame_scores):
                    if int(frame_index) <= emitted_frame_index:
                        frame_scores.pop(frame_index, None)
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})
            if isinstance(rank0_metadata, dict):
                suppressed = rank0_metadata.get("suppressed_obj_ids")
                if isinstance(suppressed, dict):
                    for frame_index in list(suppressed):
                        if int(frame_index) <= emitted_frame_index:
                            suppressed.pop(frame_index, None)

        self.max_tracker_outputs = max(self.max_tracker_outputs, retained_total)
        self.last_snapshot = {
            "emitted_frame_index": emitted_frame_index,
            "cached_output_count": len(cached_outputs) if isinstance(cached_outputs, dict) else 0,
            "tracker_output_count": retained_total,
            "rss_mb": process_rss_mb(),
        }
        input_batch = state.get("input_batch")
        input_loader = getattr(input_batch, "img_batch", None)
        if hasattr(input_loader, "audit"):
            self.last_snapshot["input_loader"] = input_loader.audit()
        if self.log_interval and self.emitted_frames % self.log_interval == 0:
            print(
                "[continuous-state] "
                f"frame={emitted_frame_index} cached={self.last_snapshot['cached_output_count']} "
                f"tracker={retained_total} rss={self.last_snapshot['rss_mb']:.0f}MiB "
                f"evicted_cache={self.cached_output_evictions} "
                f"evicted_tracker={self.tracker_output_evictions}",
                flush=True,
            )

    def audit(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "policy": "bounded_one_pass_continuous",
            "retain_frames": self.retain_frames,
            "emitted_frames": self.emitted_frames,
            "cached_output_evictions": self.cached_output_evictions,
            "tracker_output_evictions": self.tracker_output_evictions,
            "max_cached_outputs_before_compaction": self.max_cached_outputs,
            "max_tracker_outputs_after_compaction": self.max_tracker_outputs,
            "last_snapshot": self.last_snapshot,
        }


def tracking_chunk_plan(args: argparse.Namespace, reader: FrameReader) -> list[dict[str, int | str]]:
    """Resolve bounded overlapping sessions while preserving global frame indices."""

    frame_limit = reader.length
    if args.max_frames > 0:
        frame_limit = min(frame_limit, args.max_frames)
    if frame_limit <= 0:
        return []
    chunk_frames = int(args.video_chunk_frames)
    overlap = int(args.video_chunk_overlap)
    opentouch_redetect_frames = int(getattr(args, "opentouch_redetect_frames", 0))
    if getattr(args, "dataset", "generic") == "opentouch" and opentouch_redetect_frames > 0:
        redetect_frames = opentouch_redetect_frames
        chunk_frames = (
            redetect_frames
            if chunk_frames <= 0
            else min(chunk_frames, redetect_frames)
        )
        overlap = int(getattr(args, "opentouch_redetect_overlap", 0))
    if chunk_frames <= 0 or frame_limit <= chunk_frames:
        return [
            {
                "chunk_index": 0,
                "start": 0,
                "end": frame_limit,
                "prompt_frame": min(max(0, int(args.prompt_frame)), frame_limit - 1),
                "direction": str(args.propagation_direction),
            }
        ]
    chunks: list[dict[str, int | str]] = []
    start = 0
    while start < frame_limit:
        end = min(frame_limit, start + chunk_frames)
        chunks.append(
            {
                "chunk_index": len(chunks),
                "start": start,
                "end": end,
                "prompt_frame": start,
                "direction": "forward",
            }
        )
        if end >= frame_limit:
            break
        start = end - overlap
    return chunks


def resolve_chunk_staging_root(requested: str) -> Path:
    if requested != "auto":
        root = Path(requested).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        cleanup_stale_chunk_resources(root)
        return root
    candidates = [Path("/dev/shm") / "sam3_bbox_chunks"]
    candidates.append(Path(tempfile.gettempdir()) / "sam3_bbox_chunks")
    errors = []
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / f".write_probe_{os.getpid()}"
            probe.write_bytes(b"")
            probe.unlink()
            cleanup_stale_chunk_resources(root)
            return root
        except OSError as exc:
            errors.append(f"{root}: {exc}")
    raise OSError("No writable chunk staging root: " + "; ".join(errors))


def cleanup_stale_chunk_resources(root: Path) -> None:
    """Reclaim RAM chunks left by a previously killed worker without touching live PIDs."""

    resolved = root.resolve()
    if resolved in _PRUNED_STAGING_ROOTS:
        return
    _PRUNED_STAGING_ROOTS.add(resolved)
    for candidate in resolved.glob("sam3_*_*_*"):
        if not candidate.is_dir():
            continue
        parts = candidate.name.split("_", 3)
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutil.rmtree(candidate, ignore_errors=True)
        except PermissionError:
            continue


@contextmanager
def staged_chunk_resource(
    args: argparse.Namespace,
    reader: FrameReader,
    chunk: dict[str, int | str],
) -> Iterator[Path]:
    """Expose only one chunk to SAM, preferably from a RAM-backed JPEG directory."""

    start = int(chunk["start"])
    end = int(chunk["end"])
    root = resolve_chunk_staging_root(str(getattr(args, "chunk_staging_root", "auto")))
    if bool(getattr(args, "cache_staged_chunks", True)):
        cache_root = getattr(args, "_staged_chunk_cache_root", None)
        if cache_root is None:
            cache_root = Path(
                tempfile.mkdtemp(prefix=f"sam3_{os.getpid()}_cache_", dir=root)
            )
            args._staged_chunk_cache_root = cache_root
        chunk_dir = cache_root / f"chunk_{int(chunk['chunk_index']):05d}_{start}_{end}"
        if len(list(chunk_dir.glob("*.jpg"))) != end - start:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            chunk_dir.mkdir(parents=True)
            populate_staged_chunk(args, reader, start=start, end=end, chunk_dir=chunk_dir)
        yield chunk_dir
        return
    with tempfile.TemporaryDirectory(
        prefix=f"sam3_{os.getpid()}_{int(chunk['chunk_index']):05d}_",
        dir=root,
    ) as temporary:
        chunk_dir = Path(temporary)
        populate_staged_chunk(args, reader, start=start, end=end, chunk_dir=chunk_dir)
        yield chunk_dir


def populate_staged_chunk(
    args: argparse.Namespace,
    reader: FrameReader,
    *,
    start: int,
    end: int,
    chunk_dir: Path,
) -> None:
    """Materialize one byte-identical bounded chunk for SAM's JPEG loader."""

    if reader.paths is not None:
        for local_index, frame_index in enumerate(range(start, end)):
            target = chunk_dir / f"{local_index:08d}.jpg"
            os.symlink(reader.paths[frame_index].resolve(), target)
        return
    if reader.hdf5_images is not None:
        for local_index, frame_index in enumerate(range(start, end)):
            target = chunk_dir / f"{local_index:08d}.jpg"
            target.write_bytes(reader._hdf5_encoded_frame(frame_index))
        return
    workers = max(1, int(getattr(args, "chunk_encode_workers", 4)))
    pending: list[tuple[Path, Future[bytes]]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for local_index, frame_index in enumerate(range(start, end)):
            target = chunk_dir / f"{local_index:08d}.jpg"
            frame = reader.get(frame_index)
            if frame is None:
                raise RuntimeError(
                    f"Could not decode source frame {frame_index} for chunk {start}:{end}"
                )
            pending.append(
                (
                    target,
                    executor.submit(
                        encode_jpeg,
                        frame,
                        int(getattr(args, "chunk_jpeg_quality", 95)),
                    ),
                )
            )
            if len(pending) >= workers * 2:
                path, future = pending.pop(0)
                path.write_bytes(future.result())
        for path, future in pending:
            path.write_bytes(future.result())


def cleanup_staged_chunk_cache(args: argparse.Namespace) -> int:
    """Delete the current sequence's transient RAM cache and return its bytes."""

    cache_root = getattr(args, "_staged_chunk_cache_root", None)
    if cache_root is None:
        return 0
    cache_root = Path(cache_root)
    removed_bytes = 0
    if cache_root.is_dir():
        removed_bytes = sum(
            path.stat().st_size for path in cache_root.rglob("*") if path.is_file()
        )
        shutil.rmtree(cache_root, ignore_errors=True)
    args._staged_chunk_cache_root = None
    return removed_bytes


def encode_jpeg(frame: Any, quality: int) -> bytes:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, int(quality)],
    )
    if not ok:
        raise RuntimeError("Could not encode source frame")
    return encoded.tobytes()


def continuity_boxes_from_response(
    response: dict[str, Any],
    *,
    reader: FrameReader,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Select prompt-conformant, nonduplicate boxes for the next chunk session."""

    import numpy as np

    object_ids, masks, scores, _, _ = response_outputs(response)
    minimum_score = max(
        float(getattr(args, "min_prompt_score", 0.0)),
        float(getattr(args, "chunk_carry_min_score", 0.60)),
    )
    minimum_area = float(getattr(args, "min_mask_area_ratio", 0.0)) * reader.height * reader.width
    candidates = []
    for object_id, mask, score in zip(object_ids, masks, scores):
        if score is None or not np.isfinite(score) or float(score) < minimum_score:
            continue
        clean_mask = largest_connected_component(
            resize_mask_to_frame(
                np.asarray(mask) > float(getattr(args, "mask_threshold", 0.5)),
                reader.width,
                reader.height,
            )
        )
        area = int(clean_mask.sum())
        if area < minimum_area:
            continue
        bbox = mask_bbox(clean_mask)
        if bbox is None:
            continue
        candidates.append(
            {
                "object_id": int(object_id),
                "bbox": bbox,
                "score": float(score),
                "area": area,
            }
        )
    candidates.sort(key=lambda row: (row["score"], row["area"]), reverse=True)
    selected = []
    duplicate_iou = float(getattr(args, "duplicate_track_iou_floor", 0.80))
    for candidate in candidates:
        if any(
            bbox_iou(tuple(candidate["bbox"]), tuple(previous["bbox"])) >= duplicate_iou
            for previous in selected
        ):
            continue
        x1, y1, x2, y2 = candidate["bbox"]
        candidate["normalized_xywh"] = [
            max(0.0, min(1.0, x1 / reader.width)),
            max(0.0, min(1.0, y1 / reader.height)),
            max(0.0, min(1.0, (x2 - x1 + 1) / reader.width)),
            max(0.0, min(1.0, (y2 - y1 + 1) / reader.height)),
        ]
        selected.append(candidate)
        if len(selected) >= int(getattr(args, "max_objects", 1)):
            break
    return selected


def merge_continuity_boxes(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """Deduplicate boxes recovered by independent carry sessions."""

    rows = sorted(rows, key=lambda row: (row["score"], row["area"]), reverse=True)
    selected: list[dict[str, Any]] = []
    duplicate_iou = float(getattr(args, "duplicate_track_iou_floor", 0.80))
    for row in rows:
        if any(
            bbox_iou(tuple(row["bbox"]), tuple(previous["bbox"])) >= duplicate_iou
            for previous in selected
        ):
            continue
        selected.append(row)
        if len(selected) >= int(getattr(args, "max_objects", 1)):
            break
    return selected


def merge_sam_session_responses(
    session_responses: tuple[dict[str, Any], ...],
    target_object_ids: tuple[int | None, ...] | None = None,
) -> dict[str, Any]:
    """Merge same-frame outputs while keeping object IDs unique per session."""

    if target_object_ids is None:
        target_object_ids = (None,) * len(session_responses)
    if len(target_object_ids) != len(session_responses):
        raise ValueError("target_object_ids must align with session_responses")
    if len(session_responses) == 1 and target_object_ids[0] is None:
        response = dict(session_responses[0])
        response["carry_session_count"] = 1
        return response

    import numpy as np

    frame_indices = {int(response["frame_index"]) for response in session_responses}
    if len(frame_indices) != 1:
        raise RuntimeError(
            "Independent SAM carry sessions returned misaligned frame indices: "
            f"{sorted(frame_indices)}"
        )
    merged_ids: list[int] = []
    merged_masks: list[Any] = []
    merged_scores: list[float | None] = []
    for session_slot, (response, target_object_id) in enumerate(
        zip(session_responses, target_object_ids)
    ):
        object_ids, masks, scores, _, _ = response_outputs(response)
        for object_id, mask, score in zip(object_ids, masks, scores):
            if target_object_id is not None and int(object_id) != target_object_id:
                continue
            # Downstream reserves one million IDs per physical chunk. Keep each
            # carry session in a stable sub-namespace inside that range.
            merged_ids.append(session_slot * 100_000 + int(object_id))
            merged_masks.append(np.asarray(mask))
            merged_scores.append(score)
    outputs: dict[str, Any] = {
        "out_obj_ids": np.asarray(merged_ids, dtype=np.int64),
        "out_binary_masks": (
            np.stack(merged_masks, axis=0) if merged_masks else np.asarray([])
        ),
    }
    if merged_scores and all(score is not None for score in merged_scores):
        outputs["out_probs"] = np.asarray(merged_scores, dtype=np.float32)
    return {
        "frame_index": frame_indices.pop(),
        "outputs": outputs,
        "carry_session_count": len(session_responses),
    }


def prompt_box_target_object(
    response: dict[str, Any],
    *,
    expected_bbox: list[int] | tuple[int, int, int, int],
    reader: FrameReader,
    mask_threshold: float,
) -> tuple[int | None, float]:
    """Associate one semantic-prompt object with its carry box."""

    import numpy as np

    object_ids, masks, scores, _, _ = response_outputs(response)
    ranked: list[tuple[float, float, int]] = []
    for object_id, mask, score in zip(object_ids, masks, scores):
        clean_mask = largest_connected_component(
            resize_mask_to_frame(
                np.asarray(mask) > mask_threshold,
                reader.width,
                reader.height,
            )
        )
        bbox = mask_bbox(clean_mask)
        if bbox is None:
            continue
        ranked.append(
            (
                bbox_iou(tuple(expected_bbox), tuple(bbox)),
                float(score) if score is not None else -1.0,
                int(object_id),
            )
        )
    if not ranked:
        return None, 0.0
    overlap, _, object_id = max(ranked)
    # Zero overlap cannot safely establish identity. In that case leave this
    # session unfiltered so text detection can still recover the hand.
    return (object_id, overlap) if overlap > 0 else (None, 0.0)


def iter_chunked_prompt_responses(
    args: argparse.Namespace,
    *,
    predictor,
    reader: FrameReader,
    prompt: str,
) -> Iterator[tuple[dict[str, int | str], dict[str, Any]]]:
    """Run bounded per-hand SAM sessions per chunk; model weights stay resident."""

    chunk_plan = tracking_chunk_plan(args, reader)
    continuous_unchunked = bool(
        len(chunk_plan) == 1
        and int(getattr(args, "video_chunk_frames", 0)) == 0
        and getattr(args, "hdf5_source", None) is None
    )
    configure_continuous_input_loader(args, bounded=continuous_unchunked and (
        getattr(args, "continuous_state_memory", "native") == "bounded"
    ))
    carry_boxes: list[dict[str, Any]] = []
    continuity_audit = getattr(args, "_chunk_continuity_audit", None)
    for chunk_position, chunk in enumerate(chunk_plan):
        session_ids: list[str] = []
        state_compactors: list[ContinuousSessionStateCompactor | None] = []
        chunk_start = int(chunk["start"])
        chunk_length = int(chunk["end"]) - chunk_start
        next_chunk_start = (
            int(chunk_plan[chunk_position + 1]["start"])
            if chunk_position + 1 < len(chunk_plan)
            else None
        )
        next_carry_candidates: list[dict[str, Any]] = []
        next_carry_boxes: list[dict[str, Any]] = []
        carry_enabled = bool(getattr(args, "chunk_continuity", True))
        carry_session_limit = max(1, int(getattr(args, "chunk_carry_sessions", 2)))
        if carry_enabled and len(carry_boxes) > 1 and carry_session_limit > 1:
            # SAM3 allows one initial visual box per session. Keep the original
            # semantic text, then lock each session to the prompt-frame object
            # that overlaps its own box instead of emitting both hands twice.
            session_specs = [
                {
                    "box": row,
                    "prompt": prompt,
                    "isolate_prompt_object": True,
                    "mode": "text_box_isolated",
                }
                for row in carry_boxes[:carry_session_limit]
            ]
        elif carry_enabled and carry_boxes:
            # With one surviving hand, retain text discovery so an occluded
            # second hand can re-enter when it becomes visible again.
            session_specs = [
                {
                    "box": carry_boxes[0],
                    "prompt": prompt,
                    "isolate_prompt_object": False,
                    "mode": "text_box_discovery",
                }
            ]
        else:
            session_specs = [
                {
                    "box": None,
                    "prompt": prompt,
                    "isolate_prompt_object": False,
                    "mode": "text_discovery",
                }
            ]
        target_object_ids: list[int | None] = []
        target_object_ious: list[float] = []
        try:
            resource_context = (
                nullcontext(Path(args.resource))
                if continuous_unchunked
                else staged_chunk_resource(args, reader, chunk)
            )
            with resource_context as chunk_resource:
                local_prompt_frame = int(chunk["prompt_frame"]) - chunk_start
                for spec in session_specs:
                    start = predictor.handle_request(
                        start_session_request(args, reader, resource_path=chunk_resource)
                    )
                    session_id = start["session_id"]
                    session_ids.append(session_id)
                    use_bounded_continuous_state = bool(
                        continuous_unchunked
                        and getattr(args, "continuous_state_memory", "native")
                        == "bounded"
                    )
                    state_compactors.append(
                        ContinuousSessionStateCompactor(
                            predictor,
                            session_id,
                            retain_frames=int(args.continuous_state_retain_frames),
                            log_interval=int(args.continuous_state_log_interval),
                        )
                        if use_bounded_continuous_state
                        else None
                    )
                    prompt_request = {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": local_prompt_frame,
                        "text": spec["prompt"],
                    }
                    if spec["box"] is not None:
                        prompt_request.update(
                            {
                                "bounding_boxes": [spec["box"]["normalized_xywh"]],
                                "bounding_box_labels": [1],
                                "rel_coordinates": True,
                            }
                        )
                    prompt_response = predictor.handle_request(prompt_request)
                    if spec["box"] is not None and spec["isolate_prompt_object"]:
                        target_object_id, target_iou = prompt_box_target_object(
                            prompt_response,
                            expected_bbox=spec["box"]["bbox"],
                            reader=reader,
                            mask_threshold=float(args.mask_threshold),
                        )
                    else:
                        target_object_id, target_iou = None, 0.0
                    target_object_ids.append(target_object_id)
                    target_object_ious.append(target_iou)
                streams = [
                    iter_propagation(
                        predictor,
                        session_id,
                        str(chunk["direction"]),
                        start_frame_index=0,
                        max_frame_num_to_track=chunk_length,
                        state_compactor=state_compactor,
                    )
                    for session_id, state_compactor in zip(
                        session_ids, state_compactors
                    )
                ]
                for response_group in zip_longest(*streams):
                    if any(response is None for response in response_group):
                        raise RuntimeError(
                            "Independent SAM carry sessions produced different frame counts"
                        )
                    merged_response = merge_sam_session_responses(
                        response_group,
                        tuple(target_object_ids),
                    )
                    local_frame_index = merged_response.get("frame_index")
                    if local_frame_index is None:
                        continue
                    local_frame_index = int(local_frame_index)
                    if 0 <= local_frame_index < chunk_length:
                        mapped_response = dict(merged_response)
                        mapped_response["frame_index"] = chunk_start + local_frame_index
                        if (
                            carry_enabled
                            and next_chunk_start is not None
                            and mapped_response["frame_index"] == next_chunk_start
                        ):
                            next_carry_candidates.extend(
                                continuity_boxes_from_response(
                                    mapped_response,
                                    reader=reader,
                                    args=args,
                                )
                            )
                        yield chunk, mapped_response
        finally:
            continuous_state_audit = getattr(args, "_continuous_state_audit", None)
            if isinstance(continuous_state_audit, list):
                continuous_state_audit.extend(
                    compactor.audit()
                    for compactor in state_compactors
                    if compactor is not None
                )
            for session_id in reversed(session_ids):
                _close_sam_session(
                    predictor,
                    session_id,
                    empty_cache=False,
                )
            if session_ids and bool(getattr(args, "empty_cache_between_chunks", True)):
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
            memory = cuda_memory_snapshot_mb()
            if memory is not None:
                print(
                    f"[chunk {int(chunk['chunk_index'])}] closed "
                    f"frames={chunk_start}:{chunk_start + chunk_length} "
                    f"allocated={memory['allocated']:.0f}MiB "
                    f"reserved={memory['reserved']:.0f}MiB "
                    f"free={memory['free']:.0f}MiB",
                    flush=True,
                )
        next_carry_boxes = merge_continuity_boxes(next_carry_candidates, args=args)
        if isinstance(continuity_audit, list):
            continuity_audit.append(
                {
                    "prompt": prompt,
                    "chunk_index": int(chunk["chunk_index"]),
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_start + chunk_length,
                    "carry_in_count": len(carry_boxes),
                    "carry_session_count": len(session_specs),
                    "carry_prompt_count": sum(
                        spec["box"] is not None for spec in session_specs
                    ),
                    "carry_prompt_dropped_count": max(
                        0, len(carry_boxes) - len(session_specs)
                    ),
                    "carry_prompt_modes": [spec["mode"] for spec in session_specs],
                    "carry_target_object_ids": target_object_ids,
                    "carry_target_object_ious": target_object_ious,
                    "carry_out_count": len(next_carry_boxes),
                    "carry_out_boxes": [
                        {
                            "bbox": row["bbox"],
                            "score": row["score"],
                        }
                        for row in next_carry_boxes
                    ],
                }
            )
        carry_boxes = next_carry_boxes


def build_predictor(args: argparse.Namespace):
    from sam3 import build_sam3_predictor

    if args.checkpoint is None:
        raise ValueError(
            "Offline SAM3 tracking requires --checkpoint PATH. Automatic Hugging Face "
            "downloads are disabled so a proxy outage cannot silently change the run."
        )
    kwargs: dict[str, Any] = {
        "version": args.sam_version,
        "compile": False,
        # Physical chunk directories make the lazy JPEG loader safe and avoid
        # allocating a dense [T,3,H,W] tensor at session start.
        "async_loading_frames": True,
    }
    kwargs["checkpoint_path"] = str(args.checkpoint)
    if args.sam_version == "sam3.1":
        # These switches are deliberate for the cu124 compatibility profile.
        kwargs.update(
            use_fa3=False,
            use_rope_real=False,
            max_num_objects=args.sam_candidate_capacity,
            multiplex_count=max(2, args.sam_candidate_capacity),
        )
    return build_sam3_predictor(**kwargs)


class SamPredictorRuntime:
    """Keep one SAM model resident while sequential video sessions change."""

    def __init__(self) -> None:
        self._predictor = None
        self._signature: tuple[str, str | None, int] | None = None
        self.load_count = 0
        self.acquire_count = 0
        self._post_job_allocated_floor_mb: float | None = None

    @staticmethod
    def _requested_signature(args: argparse.Namespace) -> tuple[str, str | None, int]:
        return (
            str(args.sam_version),
            str(args.checkpoint) if args.checkpoint is not None else None,
            int(args.sam_candidate_capacity),
        )

    def get(self, args: argparse.Namespace):
        requested = self._requested_signature(args)
        if self._predictor is None:
            print(
                "[sam-runtime] Loading SAM predictor once; subsequent prompt/video "
                "sessions on this GPU will reuse these weights.",
                flush=True,
            )
            self._predictor = build_predictor(args)
            self._signature = requested
            self.load_count += 1
        elif self._signature != requested:
            raise RuntimeError(
                "A persistent SAM worker cannot safely change model version, checkpoint, "
                "or SAM3.1 candidate capacity between jobs. Start a separate worker."
            )
        self.acquire_count += 1
        return self._predictor

    def begin_job(self) -> None:
        """Reset per-job CUDA peaks without unloading the resident predictor."""

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            pass

    def end_job(self) -> dict[str, Any]:
        """Ensure no video session survives a job and detect persistent CUDA growth."""

        if self._predictor is None:
            return {"active_before": 0, "active_after": 0, "forced_session_count": 0}
        states = getattr(self._predictor, "_all_inference_states", None)
        session_ids = list(states) if isinstance(states, dict) else []
        forced = 0
        close_errors = []
        for session_id in session_ids:
            result = _close_sam_session(
                self._predictor,
                str(session_id),
                empty_cache=True,
            )
            forced += int(bool(result.get("forced")))
            if result.get("error"):
                close_errors.append(str(result["error"]))
        states = getattr(self._predictor, "_all_inference_states", None)
        active_after = len(states) if isinstance(states, dict) else 0
        memory = cuda_memory_snapshot_mb()
        recycle = active_after > 0
        if memory is not None:
            allocated = float(memory["allocated"])
            if self._post_job_allocated_floor_mb is None:
                self._post_job_allocated_floor_mb = allocated
            else:
                recycle = recycle or allocated > self._post_job_allocated_floor_mb + 2048.0
                self._post_job_allocated_floor_mb = min(
                    self._post_job_allocated_floor_mb,
                    allocated,
                )
        return {
            "active_before": len(session_ids),
            "active_after": active_after,
            "forced_session_count": forced,
            "close_errors": close_errors,
            "cuda_memory_mb": memory,
            "recycle_recommended": recycle,
        }

    def audit(self) -> dict[str, Any]:
        states = (
            getattr(self._predictor, "_all_inference_states", None)
            if self._predictor is not None
            else None
        )
        audit = {
            "resident": self._predictor is not None,
            "load_count": self.load_count,
            "acquire_count": self.acquire_count,
            "signature": list(self._signature) if self._signature is not None else None,
            "active_session_count": len(states) if isinstance(states, dict) else 0,
        }
        try:
            import torch

            if torch.cuda.is_available():
                megabyte = 1024.0**2
                audit["cuda_memory_mb"] = {
                    "allocated": torch.cuda.memory_allocated() / megabyte,
                    "reserved": torch.cuda.memory_reserved() / megabyte,
                    "peak_allocated": torch.cuda.max_memory_allocated() / megabyte,
                    "peak_reserved": torch.cuda.max_memory_reserved() / megabyte,
                }
        except (ImportError, RuntimeError):
            pass
        return audit

    def close(self) -> None:
        if self._predictor is None:
            return
        try:
            import gc
            import torch

            shutdown = getattr(self._predictor, "shutdown", None)
            if callable(shutdown):
                shutdown()
            del self._predictor
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            self._predictor = None
            self._signature = None
            self._post_job_allocated_floor_mb = None


def draw_tracks(frame, tracks: list[dict[str, Any]]):
    import cv2
    import numpy as np

    overlay = frame.astype(np.float32).copy()
    for track in tracks:
        object_id = int(track["track_id"])
        bbox = [int(value) for value in track["bbox"]]
        color = COLORS[object_id % len(COLORS)]
        x1, y1, x2, y2 = bbox
        mask = track.get("mask")
        if mask is not None:
            binary_mask = np.asarray(mask, dtype=bool)
            if binary_mask.shape == overlay.shape[:2]:
                color_array = np.asarray(color, dtype=np.float32)
                overlay[binary_mask] = (
                    0.52 * overlay[binary_mask] + 0.48 * color_array
                )
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        score = track.get("prompt_score")
        suffix = f" {score:.2f}" if isinstance(score, (int, float)) else ""
        glove_votes = track.get("glove_verifier_prompts", ())
        bare_votes = track.get("bare_verifier_prompts", ())
        if glove_votes or bare_votes:
            suffix += f" | gV={len(glove_votes)} bV={len(bare_votes)}"
        label = str(track.get("label", f"query_{object_id}"))
        cv2.putText(
            overlay,
            f"{label}{suffix}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay.clip(0, 255).astype(np.uint8)


def observation_track_row(observation: TrackObservation, prompt: str) -> dict[str, Any]:
    if observation.glove_verifier_prompts and observation.bare_verifier_prompts:
        semantic_state = "both"
    elif observation.glove_verifier_prompts:
        semantic_state = "glove_only"
    elif observation.bare_verifier_prompts:
        semantic_state = "bare_only"
    else:
        semantic_state = "neither"
    return {
        "track_id": observation.object_id,
        "bbox": list(observation.bbox),
        "mask_area": observation.mask_area,
        "mask_centroid": list(observation.mask_centroid)
        if observation.mask_centroid is not None
        else None,
        "prompt": prompt,
        "prompt_score": observation.prompt_score,
        "glove_verifier_prompts": list(observation.glove_verifier_prompts),
        "bare_verifier_prompts": list(observation.bare_verifier_prompts),
        "semantic_state": semantic_state,
        "selection": "prompt_conformant_locked_track",
        "bbox_source": observation.bbox_source,
        "flow_confidence": observation.flow_confidence,
        "flow_bbox_iou": observation.flow_bbox_iou,
        "flow_anchor_frames": list(observation.flow_anchor_frames),
    }


def cuda_memory_snapshot_mb() -> dict[str, float] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        scale = 1024.0**2
        free_bytes, _ = torch.cuda.mem_get_info()
        return {
            "allocated": torch.cuda.memory_allocated() / scale,
            "reserved": torch.cuda.memory_reserved() / scale,
            "free": free_bytes / scale,
        }
    except (ImportError, RuntimeError):
        return None


def _close_sam_session(
    predictor,
    session_id: str | None,
    *,
    empty_cache: bool = False,
) -> dict[str, Any]:
    if predictor is None or session_id is None:
        return {"forced": False, "error": None}
    error = None
    forced = False
    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # A failed or incomplete close must not leave the heavy state reachable
        # from a persistent predictor. This is intentionally compatible with
        # both the stock and the patched SAM3 wrappers.
        states = getattr(predictor, "_all_inference_states", None)
        if isinstance(states, dict):
            stale = states.pop(session_id, None)
            if stale is not None:
                forced = True
                if isinstance(stale, dict):
                    state = stale.get("state")
                    if isinstance(state, dict):
                        state.clear()
                    stale.clear()
        import gc
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                under_pressure = total_bytes > 0 and free_bytes / total_bytes < 0.25
                if empty_cache or under_pressure:
                    torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    if error is not None:
        print(
            f"[sam-runtime] close_session({session_id}) failed; "
            f"forced_cleanup={forced}: {error}",
            flush=True,
        )
    return {"forced": forced, "error": error}


def collect_verification_observations(
    args: argparse.Namespace,
    *,
    predictor,
    prompt: str,
    require_prompt_score: bool,
) -> tuple[list[TrackObservation], dict[str, Any]]:
    """Run one independent text-prompt session for semantic verification."""

    import numpy as np
    import torch
    reader = open_frame_reader(args)
    rows_by_frame: dict[int, dict[int, TrackObservation]] = {}
    response_output_keys: set[str] = set()
    prompt_score_sources: set[str] = set()
    filter_stats = {
        "raw_instances": 0,
        "disconnected_pixels_removed": 0,
        "absolute_area_rejections": 0,
    }
    processed_response_count = 0
    chunk_plan = tracking_chunk_plan(args, reader)
    progress_total = sum(int(chunk["end"]) - int(chunk["start"]) for chunk in chunk_plan)
    amp_context = nullcontext() if args.no_amp else torch.autocast("cuda", dtype=torch.bfloat16)
    try:
        with torch.inference_mode(), amp_context:
            with progress_bar(
                total=progress_total or None,
                unit="frame",
                desc=f"{args.output_dir.name}:verify:{prompt[:18]}",
            ) as progress:
                for chunk, response in iter_chunked_prompt_responses(
                    args, predictor=predictor, reader=reader, prompt=prompt
                ):
                    frame_index = response.get("frame_index")
                    if frame_index is None:
                        continue
                    if args.max_frames > 0 and processed_response_count >= args.max_frames:
                        break
                    frame_index = int(frame_index)
                    object_ids, masks, scores, output_keys, score_source = response_outputs(response)
                    response_output_keys.update(output_keys)
                    if score_source is not None:
                        prompt_score_sources.add(score_source)
                    if (
                        object_ids
                        and require_prompt_score
                        and (score_source is None or score_source.endswith(":shape_mismatch"))
                    ):
                        available = ", ".join(sorted(response_output_keys))
                        raise RuntimeError(
                            "SAM semantic verifier lacks a finite per-object score "
                            f"({score_source!r}); available keys: {available}"
                        )
                    observations_by_id = rows_by_frame.setdefault(frame_index, {})
                    min_mask_area = args.min_mask_area_ratio * reader.height * reader.width
                    filter_stats["raw_instances"] += len(object_ids)
                    for object_id, mask, score in zip(object_ids, masks, scores):
                        raw_mask = resize_mask_to_frame(
                            np.asarray(mask) > args.mask_threshold,
                            reader.width,
                            reader.height,
                        )
                        clean_mask = largest_connected_component(raw_mask)
                        raw_area = int(raw_mask.sum())
                        clean_area = int(clean_mask.sum())
                        filter_stats["disconnected_pixels_removed"] += raw_area - clean_area
                        if clean_area < min_mask_area:
                            filter_stats["absolute_area_rejections"] += 1
                            continue
                        bbox = mask_bbox(clean_mask)
                        if bbox is None:
                            filter_stats["absolute_area_rejections"] += 1
                            continue
                        namespaced_object_id = (
                            int(chunk["chunk_index"]) * 1_000_000 + int(object_id)
                        )
                        observation = TrackObservation(
                            frame_index=frame_index,
                            object_id=namespaced_object_id,
                            bbox=tuple(bbox),
                            mask_area=clean_area,
                            prompt_score=score,
                            mask_centroid=mask_centroid(clean_mask),
                        )
                        previous = observations_by_id.get(observation.object_id)
                        if previous is None or (
                            observation.prompt_score is not None
                            and (
                                previous.prompt_score is None
                                or observation.prompt_score > previous.prompt_score
                            )
                        ):
                            observations_by_id[observation.object_id] = observation
                    processed_response_count += 1
                    progress.update(1)
    finally:
        reader.close()

    observations = [
        row
        for frame_index in sorted(rows_by_frame)
        for row in rows_by_frame[frame_index].values()
    ]
    observations, chunk_stitch_audit = stitch_overlapping_chunk_tracks(observations)
    observations = consolidate_duplicate_track_observations(observations, {})
    return observations, {
        "prompt": prompt,
        "require_prompt_score": require_prompt_score,
        "response_output_keys": sorted(response_output_keys),
        "prompt_score_sources": sorted(prompt_score_sources),
        "observation_count": len(observations),
        "frame_count": len(rows_by_frame),
        "filter_stats": filter_stats,
        "chunk_stitching": chunk_stitch_audit,
    }


def _match_replay_candidates(
    candidates: list[dict[str, Any]],
    accepted: list[TrackObservation],
) -> list[dict[str, Any]]:
    """Match replay masks to offline-selected observations without ID fallback."""

    matched: list[dict[str, Any]] = []
    used_candidates: set[int] = set()
    for observation in accepted:
        candidate_index = next(
            (
                index
                for index, candidate in enumerate(candidates)
                if index not in used_candidates
                and int(candidate["object_id"]) == observation.object_id
            ),
            None,
        )
        if candidate_index is None:
            compatible = [
                (bbox_iou(observation.bbox, tuple(candidate["bbox"])), index)
                for index, candidate in enumerate(candidates)
                if index not in used_candidates
            ]
            if compatible:
                best_iou, best_index = max(compatible)
                candidate_index = best_index if best_iou >= 0.85 else None
        if candidate_index is None:
            continue
        used_candidates.add(candidate_index)
        candidate = candidates[candidate_index]
        matched.append(
            {
                "track_id": observation.object_id,
                "bbox": list(observation.bbox),
                "mask": candidate["clean_mask"],
                "prompt_score": observation.prompt_score,
                "glove_verifier_prompts": observation.glove_verifier_prompts,
                "bare_verifier_prompts": observation.bare_verifier_prompts,
                "label": f"query_{observation.object_id}",
            }
        )
    return matched


def safe_file_token(value: str) -> str:
    """Make a stable, human-readable filename component from a prompt."""

    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()[:72] or "prompt"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_prompt_raw_mask_preview(
    args: argparse.Namespace,
    *,
    predictor,
    prompt: str,
    preview_path: Path,
    label: str,
) -> None:
    """Replay one verifier prompt for a visual semantic-debug artifact."""

    import cv2
    import numpy as np
    import torch
    reader = open_frame_reader(args)
    fps = args.preview_fps if args.preview_fps > 0 else reader.fps
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (reader.width, reader.height),
    )
    if not writer.isOpened():
        reader.close()
        raise RuntimeError(f"Could not create semantic debug preview: {preview_path}")
    seen_frames: set[int] = set()
    replayed = 0
    chunk_plan = tracking_chunk_plan(args, reader)
    progress_total = sum(int(chunk["end"]) - int(chunk["start"]) for chunk in chunk_plan)
    amp_context = nullcontext() if args.no_amp else torch.autocast("cuda", dtype=torch.bfloat16)
    try:
        with torch.inference_mode(), amp_context:
            with progress_bar(
                total=progress_total or None,
                unit="frame",
                desc=f"{args.output_dir.name}:debug:{prompt[:18]}",
            ) as progress:
                for _, response in iter_chunked_prompt_responses(
                    args, predictor=predictor, reader=reader, prompt=prompt
                ):
                    frame_index = response.get("frame_index")
                    if frame_index is None:
                        continue
                    frame_index = int(frame_index)
                    if frame_index in seen_frames:
                        continue
                    if args.max_frames > 0 and replayed >= args.max_frames:
                        break
                    seen_frames.add(frame_index)
                    frame = reader.get(frame_index)
                    if frame is None:
                        continue
                    object_ids, masks, scores, _, _ = response_outputs(response)
                    raw_tracks: list[dict[str, Any]] = []
                    for object_id, mask, score in zip(object_ids, masks, scores):
                        raw_mask = resize_mask_to_frame(
                            np.asarray(mask) > args.mask_threshold,
                            reader.width,
                            reader.height,
                        )
                        bbox = mask_bbox(raw_mask)
                        if bbox is None:
                            continue
                        raw_tracks.append(
                            {
                                "track_id": int(object_id),
                                "bbox": bbox,
                                "mask": raw_mask,
                                "prompt_score": score,
                                "label": f"raw_{int(object_id)}",
                            }
                        )
                    debug_frame = draw_tracks(frame, raw_tracks)
                    cv2.putText(
                        debug_frame,
                        f"{args.sam_version} | {label} | frame {frame_index}",
                        (18, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (245, 245, 245),
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(debug_frame)
                    replayed += 1
                    progress.update(1)
    finally:
        writer.release()
        reader.close()


def write_sam_mask_previews(
    args: argparse.Namespace,
    *,
    predictor,
    prompt: str,
    raw_preview_path: Path,
    preview_path: Path,
    accepted_by_frame: dict[int, list[TrackObservation]],
    rejected_by_frame: dict[int, list[dict]],
) -> None:
    """Replay the primary prompt so raw and final videos contain SAM masks.

    The first pass intentionally retains only compact metadata for selection.
    Replaying once avoids keeping a full video of boolean masks in RAM.  The
    replay is forward-only so both preview files remain chronological even when
    an analysis run intentionally used bidirectional propagation.
    """

    import cv2
    import numpy as np
    import torch
    reader = open_frame_reader(args)
    fps = args.preview_fps if args.preview_fps > 0 else reader.fps
    raw_writer = cv2.VideoWriter(
        str(raw_preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (reader.width, reader.height),
    )
    final_writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (reader.width, reader.height),
    )
    if not raw_writer.isOpened() or not final_writer.isOpened():
        raw_writer.release()
        final_writer.release()
        reader.close()
        raise RuntimeError(f"Could not create SAM mask preview videos under {args.output_dir}")
    seen_frames: set[int] = set()
    replayed = 0
    chunk_plan = tracking_chunk_plan(args, reader)
    progress_total = sum(int(chunk["end"]) - int(chunk["start"]) for chunk in chunk_plan)
    amp_context = nullcontext() if args.no_amp else torch.autocast("cuda", dtype=torch.bfloat16)
    try:
        with torch.inference_mode(), amp_context:
            with progress_bar(total=progress_total or None, unit="frame", desc=f"{args.output_dir.name}:preview") as progress:
                for _, response in iter_chunked_prompt_responses(
                    args, predictor=predictor, reader=reader, prompt=prompt
                ):
                    frame_index = response.get("frame_index")
                    if frame_index is None:
                        continue
                    frame_index = int(frame_index)
                    if frame_index in seen_frames:
                        continue
                    if args.max_frames > 0 and replayed >= args.max_frames:
                        break
                    seen_frames.add(frame_index)
                    frame = reader.get(frame_index)
                    if frame is None:
                        continue
                    object_ids, masks, scores, _, _ = response_outputs(response)
                    raw_tracks: list[dict[str, Any]] = []
                    cleaned_candidates: list[dict[str, Any]] = []
                    min_mask_area = args.min_mask_area_ratio * reader.height * reader.width
                    for object_id, mask, score in zip(object_ids, masks, scores):
                        raw_mask = resize_mask_to_frame(
                            np.asarray(mask) > args.mask_threshold,
                            reader.width,
                            reader.height,
                        )
                        raw_bbox = mask_bbox(raw_mask)
                        if raw_bbox is not None:
                            raw_tracks.append(
                                {
                                    "track_id": int(object_id),
                                    "bbox": raw_bbox,
                                    "mask": raw_mask,
                                    "prompt_score": score,
                                    "label": f"raw_{int(object_id)}",
                                }
                            )
                        clean_mask = largest_connected_component(raw_mask)
                        clean_area = int(clean_mask.sum())
                        if clean_area < min_mask_area:
                            continue
                        clean_bbox = mask_bbox(clean_mask)
                        if clean_bbox is None:
                            continue
                        cleaned_candidates.append(
                            {
                                "object_id": int(object_id),
                                "bbox": clean_bbox,
                                "clean_mask": clean_mask,
                                "prompt_score": score,
                            }
                        )
                    raw_debug = draw_tracks(frame, raw_tracks)
                    final_tracks = _match_replay_candidates(
                        cleaned_candidates,
                        accepted_by_frame.get(frame_index, []),
                    )
                    final_debug = draw_tracks(frame, final_tracks)
                    cv2.putText(
                        raw_debug,
                        f"{args.sam_version} | raw gloved-prompt SAM masks | frame {frame_index}",
                        (18, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (245, 245, 245),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        final_debug,
                        f"{args.sam_version} | accepted gloved-track SAM masks | frame {frame_index}",
                        (18, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (245, 245, 245),
                        2,
                        cv2.LINE_AA,
                    )
                    if rejected_by_frame.get(frame_index):
                        cv2.putText(
                            final_debug,
                            f"rejected={len(rejected_by_frame[frame_index])}",
                            (18, 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (70, 100, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    raw_writer.write(raw_debug)
                    final_writer.write(final_debug)
                    replayed += 1
                    progress.update(1)
    finally:
        raw_writer.release()
        final_writer.release()
        reader.close()


def run_tracking(args: argparse.Namespace, *, predictor_runtime: SamPredictorRuntime) -> int:
    """Track one video using a predictor that may be shared by a GPU worker."""

    args.resource = args.resource.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.hdf5_source is not None:
        hdf5_path, separator, dataset_key = args.hdf5_source.partition("::")
        if not separator or not dataset_key:
            raise ValueError("--hdf5-source must use FILE::DATASET_KEY syntax")
        resolved_hdf5 = Path(hdf5_path).expanduser().resolve()
        args.hdf5_source = f"{resolved_hdf5}::{dataset_key}"
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    args.max_objects = resolve_max_objects(
        args.dataset, args.expected_gloved_hands, args.max_objects
    )
    if args.sam_candidate_capacity <= 0:
        args.sam_candidate_capacity = max(4, args.max_objects * 2)
    if args.sam_candidate_capacity < args.max_objects:
        raise ValueError("--sam-candidate-capacity must be >= --max-objects")
    if args.hdf5_source is None and not args.resource.exists():
        raise FileNotFoundError(args.resource)
    if args.hdf5_source is not None and not Path(args.hdf5_source.split("::", 1)[0]).is_file():
        raise FileNotFoundError(args.hdf5_source.split("::", 1)[0])
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.max_frames < 0:
        raise ValueError("--max-frames must be >=0")
    if not 0 < args.mask_threshold < 1:
        raise ValueError("--mask-threshold must be between 0 and 1")
    if not 0 <= args.min_mask_area_ratio < 1:
        raise ValueError("--min-mask-area-ratio must lie in [0,1)")
    if not 0 <= args.min_prompt_score <= 1:
        raise ValueError("--min-prompt-score must lie in [0,1]")
    if args.min_track_frames < 1:
        raise ValueError("--min-track-frames must be positive")
    if args.temporal_max_frame_gap < 1:
        raise ValueError("--temporal-max-frame-gap must be positive")
    if args.temporal_center_residual_ratio <= 0:
        raise ValueError("--temporal-center-residual-ratio must be positive")
    if args.temporal_area_ratio <= 1:
        raise ValueError("--temporal-area-ratio must be >1")
    if not 0 <= args.temporal_neighbor_iou_floor <= 1:
        raise ValueError("--temporal-neighbor-iou-floor must lie in [0,1]")
    if args.temporal_return_excursion_frames < 0:
        raise ValueError("--temporal-return-excursion-frames must be >=0")
    if not 0 <= args.min_relative_mask_area <= 1:
        raise ValueError("--min-relative-mask-area must lie in [0,1]")
    if not 0 <= args.bare_match_iou_floor <= 1:
        raise ValueError("--bare-match-iou-floor must lie in [0,1]")
    if not 0 <= args.min_glove_verifier_fraction <= 1:
        raise ValueError("--min-glove-verifier-fraction must lie in [0,1]")
    if args.semantic_match_centroid_ratio <= 0:
        raise ValueError("--semantic-match-centroid-ratio must be positive")
    if not 0 <= args.max_bare_evidence_fraction <= 1:
        raise ValueError("--max-bare-evidence-fraction must lie in [0,1]")
    if args.long_video_offload_frames < 1:
        raise ValueError("--long-video-offload-frames must be positive")
    if args.video_chunk_frames < 0:
        raise ValueError("--video-chunk-frames must be >=0")
    if args.video_chunk_overlap < 0:
        raise ValueError("--video-chunk-overlap must be >=0")
    if args.video_chunk_frames > 0 and args.video_chunk_overlap >= args.video_chunk_frames:
        raise ValueError("--video-chunk-overlap must be smaller than --video-chunk-frames")
    if not 1 <= args.chunk_jpeg_quality <= 100:
        raise ValueError("--chunk-jpeg-quality must lie in [1,100]")
    if args.chunk_encode_workers < 1:
        raise ValueError("--chunk-encode-workers must be positive")
    if not 0 <= args.chunk_carry_min_score <= 1:
        raise ValueError("--chunk-carry-min-score must lie in [0,1]")
    if args.chunk_carry_sessions < 1:
        raise ValueError("--chunk-carry-sessions must be positive")
    if args.continuous_state_retain_frames < 1:
        raise ValueError("--continuous-state-retain-frames must be positive")
    if args.continuous_state_log_interval < 0:
        raise ValueError("--continuous-state-log-interval must be >=0")
    if args.continuous_input_cache_frames < 1:
        raise ValueError("--continuous-input-cache-frames must be positive")
    if args.flow_max_gap < 0:
        raise ValueError("--flow-max-gap must be >=0")
    if args.flow_fb_error <= 0:
        raise ValueError("--flow-fb-error must be positive")
    if args.flow_min_points < 3:
        raise ValueError("--flow-min-points must be >=3")
    if not 0 <= args.flow_min_inlier_ratio <= 1:
        raise ValueError("--flow-min-inlier-ratio must lie in [0,1]")
    if not 0 <= args.flow_min_confidence <= 1:
        raise ValueError("--flow-min-confidence must lie in [0,1]")
    if not 0 <= args.flow_conflict_iou < args.flow_sam_iou_accept <= 1:
        raise ValueError(
            "Require 0 <= --flow-conflict-iou < --flow-sam-iou-accept <= 1"
        )
    if args.flow_cache_frames < 2:
        raise ValueError("--flow-cache-frames must be >=2")
    if args.flow_bridge_policy != "off" and not args.flow_assist:
        raise ValueError("--flow-bridge-policy requires --flow-assist")
    if (
        args.continuous_state_memory == "bounded"
        and args.video_chunk_frames == 0
        and args.propagation_direction != "forward"
    ):
        raise ValueError(
            "bounded continuous state currently supports forward one-pass propagation only"
        )
    if args.opentouch_redetect_frames < 0:
        raise ValueError("--opentouch-redetect-frames must be >=0")
    if args.opentouch_redetect_overlap < 0:
        raise ValueError("--opentouch-redetect-overlap must be >=0")
    if (
        args.opentouch_redetect_frames > 0
        and args.opentouch_redetect_overlap >= args.opentouch_redetect_frames
    ):
        raise ValueError(
            "--opentouch-redetect-overlap must be smaller than --opentouch-redetect-frames"
        )

    prompt, preset = load_prompt(args)
    bare_verification_mode = args.bare_verification_mode
    glove_verifier_prompts: tuple[str, ...] = ()
    bare_verifier_prompts: tuple[str, ...] = ()
    if bare_verification_mode != "off":
        glove_verifier_prompts, bare_verifier_prompts = resolve_verifier_prompt_lists(
            preset,
            glove_value=args.glove_verification_prompts,
            bare_value=args.bare_verification_prompts,
            legacy_bare_prompt=args.bare_verification_prompt,
        )
    run_config = {
        "sam_version": args.sam_version,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "dataset": args.dataset,
        "prompt_preset": args.prompt_preset,
        "prompt": prompt,
        "prompt_frame": args.prompt_frame,
        "propagation_direction": args.propagation_direction,
        "max_frames": args.max_frames,
        "max_objects": args.max_objects,
        "sam_candidate_capacity": args.sam_candidate_capacity,
        "mask_threshold": args.mask_threshold,
        "min_mask_area_ratio": args.min_mask_area_ratio,
        "min_relative_mask_area": args.min_relative_mask_area,
        "legacy_relative_area_filter_disabled": True,
        "min_prompt_score": args.min_prompt_score,
        "min_track_frames": args.min_track_frames,
        "duplicate_track_iou_floor": args.duplicate_track_iou_floor,
        "duplicate_track_overlap_fraction": args.duplicate_track_overlap_fraction,
        "duplicate_track_match_fraction": args.duplicate_track_match_fraction,
        "duplicate_track_centroid_ratio": args.duplicate_track_centroid_ratio,
        "duplicate_track_area_ratio": args.duplicate_track_area_ratio,
        "duplicate_track_min_frames": args.duplicate_track_min_frames,
        "require_prompt_score": not args.allow_missing_prompt_score,
        "bare_verification_mode": bare_verification_mode,
        "glove_verification_prompts": list(glove_verifier_prompts),
        "bare_verification_prompts": list(bare_verifier_prompts),
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
        "flow_assist": bool(args.flow_assist),
        "flow_bridge_policy": args.flow_bridge_policy,
        "flow_max_gap": args.flow_max_gap,
        "flow_fb_error": args.flow_fb_error,
        "flow_min_points": args.flow_min_points,
        "flow_min_inlier_ratio": args.flow_min_inlier_ratio,
        "flow_min_confidence": args.flow_min_confidence,
        "flow_sam_iou_accept": args.flow_sam_iou_accept,
        "flow_conflict_iou": args.flow_conflict_iou,
        "flow_cache_frames": args.flow_cache_frames,
        "largest_component_only": True,
        "mask_previews": not args.no_mask_previews,
        "input_rgb_samples": bool(args.input_rgb_samples),
        "semantic_debug": bool(args.semantic_debug),
        "offload_video_to_cpu": bool(args.offload_video_to_cpu),
        "offload_state_to_cpu": args.offload_state_to_cpu,
        "long_video_offload_frames": args.long_video_offload_frames,
        "video_chunk_frames": args.video_chunk_frames,
        "video_chunk_overlap": args.video_chunk_overlap,
        "chunk_staging_root": args.chunk_staging_root,
        "chunk_jpeg_quality": args.chunk_jpeg_quality,
        "chunk_encode_workers": args.chunk_encode_workers,
        "cache_staged_chunks": bool(args.cache_staged_chunks),
        "empty_cache_between_chunks": bool(args.empty_cache_between_chunks),
        "chunk_continuity": bool(args.chunk_continuity),
        "chunk_continuity_version": CHUNK_CONTINUITY_VERSION,
        "chunk_carry_min_score": args.chunk_carry_min_score,
        "chunk_carry_sessions": args.chunk_carry_sessions,
        "chunk_fragment_reentry": bool(args.chunk_fragment_reentry),
        "continuous_state_memory": args.continuous_state_memory,
        "continuous_state_retain_frames": args.continuous_state_retain_frames,
        "continuous_state_log_interval": args.continuous_state_log_interval,
        "continuous_input_cache_frames": args.continuous_input_cache_frames,
        "physical_chunk_resources": bool(
            args.video_chunk_frames > 0
            or (args.dataset == "opentouch" and args.opentouch_redetect_frames > 0)
            or args.hdf5_source is not None
        ),
        "opentouch_redetect_frames": args.opentouch_redetect_frames,
        "opentouch_redetect_overlap": args.opentouch_redetect_overlap,
    }
    summary_path = args.output_dir / "summary.json"
    in_progress_path = args.output_dir / ".in_progress"
    if summary_path.is_file() and not in_progress_path.exists() and not args.overwrite:
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            previous_config = previous.get("run_config", {})
            if previous.get("status") == "complete" and previous_config == run_config:
                print(f"Already complete: {args.output_dir}")
                return 0
        except (OSError, json.JSONDecodeError):
            pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.unlink(missing_ok=True)
    atomic_write_text(
        in_progress_path,
        json.dumps(
            {
                "status": "in_progress",
                "pid": os.getpid(),
                "started_unix": time.time(),
                "resource": str(args.resource),
            },
            separators=(",", ":"),
        ),
    )
    if args.no_mask_previews:
        (args.output_dir / "raw_sam_preview.mp4").unlink(missing_ok=True)
        (args.output_dir / "preview.mp4").unlink(missing_ok=True)
    args._chunk_continuity_audit = []
    args._continuous_state_audit = []
    input_color_audit = audit_input_color(
        args.resource,
        args.output_dir,
        write_samples=args.input_rgb_samples,
        hdf5_source=args.hdf5_source,
    )

    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 video tracking requires a CUDA-visible GPU")
    predictor = predictor_runtime.get(args)
    reader = open_frame_reader(args)
    source_fps = reader.fps
    preview_path = args.output_dir / "preview.mp4"
    raw_preview_path = args.output_dir / "raw_sam_preview.mp4"
    raw_rows: dict[int, dict[str, Any]] = {}
    response_output_keys: set[str] = set()
    prompt_score_sources: set[str] = set()
    processed_response_count = 0
    filter_stats = {
        "raw_instances": 0,
        "disconnected_pixels_removed": 0,
        "absolute_area_rejections": 0,
        "relative_area_rejections": 0,
        "relative_area_filter_disabled": True,
        "cardinality_rejections": 0,
    }
    chunk_plan = tracking_chunk_plan(args, reader)
    progress_total = sum(int(chunk["end"]) - int(chunk["start"]) for chunk in chunk_plan)
    primary_session_request = start_session_request(args, reader)
    amp_context = nullcontext() if args.no_amp else torch.autocast("cuda", dtype=torch.bfloat16)
    try:
        with torch.inference_mode(), amp_context:
            with progress_bar(total=progress_total or None, unit="frame", desc=args.output_dir.name) as progress:
                for chunk, response in iter_chunked_prompt_responses(
                    args, predictor=predictor, reader=reader, prompt=prompt
                ):
                    frame_index = response.get("frame_index")
                    if frame_index is None:
                        continue
                    if args.max_frames > 0 and processed_response_count >= args.max_frames:
                        break
                    frame_index = int(frame_index)
                    object_ids, masks, scores, output_keys, score_source = response_outputs(response)
                    response_output_keys.update(output_keys)
                    if score_source is not None:
                        prompt_score_sources.add(score_source)
                    if (
                        object_ids
                        and not args.allow_missing_prompt_score
                        and (
                            score_source is None
                            or score_source.endswith(":shape_mismatch")
                        )
                    ):
                        available = ", ".join(sorted(response_output_keys))
                        raise RuntimeError(
                            "SAM response cannot prove prompt conformity because its "
                            f"per-object score is unavailable ({score_source!r}). "
                            f"Available SAM output keys: {available}"
                        )
                    frame_row = raw_rows.setdefault(
                        frame_index,
                        {
                            "frame_index": frame_index,
                            "timestamp_ms": int(round(frame_index / source_fps * 1000.0)),
                            "raw_response_candidate_count": 0,
                            "observations_by_id": {},
                        },
                    )
                    frame_row["raw_response_candidate_count"] += len(object_ids)
                    min_mask_area = args.min_mask_area_ratio * reader.height * reader.width
                    filter_stats["raw_instances"] += len(object_ids)
                    for object_id, mask, score in zip(object_ids, masks, scores):
                        raw_mask = resize_mask_to_frame(
                            np.asarray(mask) > args.mask_threshold,
                            reader.width,
                            reader.height,
                        )
                        clean_mask = largest_connected_component(raw_mask)
                        raw_area = int(raw_mask.sum())
                        clean_area = int(clean_mask.sum())
                        filter_stats["disconnected_pixels_removed"] += raw_area - clean_area
                        if clean_area < min_mask_area:
                            filter_stats["absolute_area_rejections"] += 1
                            continue
                        bbox = mask_bbox(clean_mask)
                        if bbox is None:
                            filter_stats["absolute_area_rejections"] += 1
                            continue
                        namespaced_object_id = (
                            int(chunk["chunk_index"]) * 1_000_000 + int(object_id)
                        )
                        observation = TrackObservation(
                            frame_index=frame_index,
                            object_id=namespaced_object_id,
                            bbox=tuple(bbox),
                            mask_area=clean_area,
                            prompt_score=score,
                            mask_centroid=mask_centroid(clean_mask),
                        )
                        previous = frame_row["observations_by_id"].get(observation.object_id)
                        if previous is None or (
                            observation.prompt_score is not None
                            and (
                                previous.prompt_score is None
                                or observation.prompt_score > previous.prompt_score
                            )
                        ):
                            frame_row["observations_by_id"][observation.object_id] = observation
                    processed_response_count += 1
                    progress.update(1)
    finally:
        reader.close()

    raw_frame_rows = []
    for frame_index in sorted(raw_rows):
        raw_row = raw_rows[frame_index]
        raw_row["observations"] = list(raw_row.pop("observations_by_id").values())
        raw_row["raw_candidate_count"] = len(raw_row["observations"])
        raw_frame_rows.append(raw_row)
    all_observations = [
        observation
        for row in raw_frame_rows
        for observation in row["observations"]
    ]
    all_observations, chunk_stitch_audit = stitch_overlapping_chunk_tracks(
        all_observations
    )
    all_observations = consolidate_duplicate_track_observations(all_observations, {})
    semantic_verification_audit: dict[str, Any] = {
        "mode": bare_verification_mode,
        "bare_rejection_policy": args.bare_rejection_policy,
        "primary_prompt": prompt,
        "glove_verifier_prompts": list(glove_verifier_prompts),
        "bare_verifier_prompts": list(bare_verifier_prompts),
        "policy": (
            "independent prompt sessions vote only after same-frame spatial association; "
            "out_probs are thresholded within each prompt and are never compared across prompts"
        ),
        "passes": {"glove": {}, "bare": {}},
    }
    semantic_debug_rows: list[dict[str, Any]] = []
    if bare_verification_mode != "off":
        glove_verifier_observations: dict[str, list[TrackObservation]] = {}
        bare_verifier_observations: dict[str, list[TrackObservation]] = {}
        require_glove_verifier_score = bare_verification_mode == "filter"
        # With the default diagnostic-only bare policy, a missing native score
        # on a bare prompt must not abort an otherwise usable positive glove
        # track.  Positive verifier scores remain fail-closed.
        require_bare_verifier_score = (
            bare_verification_mode == "filter"
            and args.bare_rejection_policy != "off"
        )
        for verifier_prompt in glove_verifier_prompts:
            rows, pass_audit = collect_verification_observations(
                args,
                predictor=predictor,
                prompt=verifier_prompt,
                require_prompt_score=require_glove_verifier_score,
            )
            glove_verifier_observations[verifier_prompt] = rows
            semantic_verification_audit["passes"]["glove"][verifier_prompt] = pass_audit
        for verifier_prompt in bare_verifier_prompts:
            rows, pass_audit = collect_verification_observations(
                args,
                predictor=predictor,
                prompt=verifier_prompt,
                require_prompt_score=require_bare_verifier_score,
            )
            bare_verifier_observations[verifier_prompt] = rows
            semantic_verification_audit["passes"]["bare"][verifier_prompt] = pass_audit
        all_observations, semantic_match_audit = attach_semantic_prompt_votes(
            all_observations,
            glove_verifier_observations,
            bare_verifier_observations,
            match_iou_floor=args.bare_match_iou_floor,
            min_verifier_score=args.min_prompt_score,
            max_centroid_distance_ratio=args.semantic_match_centroid_ratio,
            include_match_details=args.semantic_debug,
        )
        semantic_debug_rows = semantic_match_audit.pop("observation_match_details", [])
        semantic_verification_audit.update(
            {
                "match": semantic_match_audit,
                "min_glove_verifier_fraction": args.min_glove_verifier_fraction,
                "max_bare_evidence_fraction": args.max_bare_evidence_fraction,
            }
        )
    min_glove_verifier_fraction = (
        args.min_glove_verifier_fraction if bare_verification_mode == "filter" else 0.0
    )
    max_bare_evidence_fraction = (
        args.max_bare_evidence_fraction if bare_verification_mode == "filter" else 1.0
    )
    try:
        selected_track_ids, selection_audit = select_prompt_tracks(
            all_observations,
            total_frames=len(raw_frame_rows),
            max_tracks=args.max_objects,
            min_prompt_score=args.min_prompt_score,
            min_track_frames=args.min_track_frames,
            require_prompt_score=not args.allow_missing_prompt_score,
            min_glove_verifier_fraction=min_glove_verifier_fraction,
            max_bare_evidence_fraction=max_bare_evidence_fraction,
            bare_rejection_policy=args.bare_rejection_policy,
            duplicate_iou_floor=args.duplicate_track_iou_floor,
            duplicate_overlap_fraction=args.duplicate_track_overlap_fraction,
            duplicate_match_fraction=args.duplicate_track_match_fraction,
            duplicate_centroid_ratio=args.duplicate_track_centroid_ratio,
            duplicate_area_ratio=args.duplicate_track_area_ratio,
            duplicate_min_frames=args.duplicate_track_min_frames,
            allow_nonoverlapping_fragments=bool(args.chunk_fragment_reentry),
        )
    except RuntimeError as exc:
        available = ", ".join(sorted(response_output_keys)) or "<no output keys>"
        raise RuntimeError(f"{exc} Available SAM output keys: {available}") from exc
    all_observations = consolidate_duplicate_track_observations(
        all_observations,
        selection_audit.get("duplicate_track_aliases", {}),
    )
    accepted_by_frame, rejected_by_frame, temporal_audit = filter_selected_tracks(
        all_observations,
        selected_track_ids=selected_track_ids,
        min_prompt_score=args.min_prompt_score,
        require_prompt_score=not args.allow_missing_prompt_score,
        max_frame_gap=args.temporal_max_frame_gap,
        center_residual_ratio=args.temporal_center_residual_ratio,
        area_ratio=args.temporal_area_ratio,
        neighbor_iou_floor=args.temporal_neighbor_iou_floor,
        return_excursion_max_frames=args.temporal_return_excursion_frames,
        reject_unverified_return_excursions=(
            args.dataset == "opentouch" and bare_verification_mode == "filter"
        ),
        reject_bare_prompt_matches=bare_verification_mode == "filter",
        bare_rejection_policy=args.bare_rejection_policy,
    )
    flow_audit: dict[str, Any] = {
        "enabled": False,
        "bridge_policy": args.flow_bridge_policy,
    }
    if args.flow_assist:
        flow_reader = open_frame_reader(args)
        try:
            accepted_by_frame, flow_audit = apply_optical_flow_assist(
                accepted_by_frame,
                get_frame=flow_reader.get,
                width=flow_reader.width,
                height=flow_reader.height,
                config=FlowAssistConfig(
                    max_gap=args.flow_max_gap,
                    fb_error=args.flow_fb_error,
                    min_points=args.flow_min_points,
                    min_inlier_ratio=args.flow_min_inlier_ratio,
                    min_confidence=args.flow_min_confidence,
                    sam_iou_accept=args.flow_sam_iou_accept,
                    conflict_iou=args.flow_conflict_iou,
                    bridge_policy=args.flow_bridge_policy,
                    cache_frames=args.flow_cache_frames,
                    max_tracks_per_frame=args.max_objects,
                ),
            )
        finally:
            flow_reader.close()
    filter_stats["duplicate_track_suppressions"] = len(
        selection_audit.get("duplicate_track_aliases", {})
    )
    filter_stats["cardinality_rejections"] = max(
        0,
        len(selection_audit.get("duplicate_track_clusters", []))
        - len(selected_track_ids),
    )
    semantic_debug_paths: dict[str, Any] | None = None
    if args.semantic_debug:
        from collections import Counter

        debug_dir = args.output_dir / "semantic_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        selection_by_id = {
            int(summary["object_id"]): summary
            for summary in selection_audit["track_summaries"]
        }
        rejection_reasons_by_observation: dict[tuple[int, int], list[str]] = {}
        for frame_index, rows in rejected_by_frame.items():
            for row in rows:
                key = (int(frame_index), int(row["track_id"]))
                rejection_reasons_by_observation.setdefault(key, []).append(str(row["reason"]))
        for row in semantic_debug_rows:
            track_id = int(row["primary_object_id"])
            summary = selection_by_id.get(track_id, {})
            row["selected_track"] = bool(summary.get("selected", False))
            row["track_selection_reason"] = summary.get("selection_reason")
            row["effective_bare_evidence_fraction"] = summary.get(
                "effective_bare_evidence_fraction"
            )
            row["bare_rejection_policy"] = args.bare_rejection_policy
            row["frame_rejection_reasons"] = rejection_reasons_by_observation.get(
                (int(row["frame_index"]), track_id),
                [],
            )
        evidence_path = debug_dir / "semantic_match_evidence.jsonl"
        write_jsonl(evidence_path, semantic_debug_rows)
        debug_summary_path = debug_dir / "semantic_match_summary.json"
        debug_summary_path.write_text(
            json.dumps(
                {
                    "bare_rejection_policy": args.bare_rejection_policy,
                    "observation_count": len(semantic_debug_rows),
                    "semantic_state_counts": dict(
                        sorted(Counter(row["semantic_state"] for row in semantic_debug_rows).items())
                    ),
                    "selected_observation_count": sum(
                        bool(row["selected_track"]) for row in semantic_debug_rows
                    ),
                    "frame_rejection_reason_counts": dict(
                        sorted(
                            Counter(
                                reason
                                for row in semantic_debug_rows
                                for reason in row["frame_rejection_reasons"]
                            ).items()
                        )
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        semantic_debug_paths = {
            "evidence": str(evidence_path),
            "summary": str(debug_summary_path),
            "glove_verifier_raw_previews": [],
            "bare_verifier_raw_previews": [],
        }
    output_rows = []
    raw_frame_by_index = {
        int(raw_row["frame_index"]): raw_row for raw_row in raw_frame_rows
    }
    output_frame_indices = sorted(set(raw_frame_by_index) | set(accepted_by_frame))
    for frame_index in output_frame_indices:
        raw_row = raw_frame_by_index.get(
            frame_index,
            {
                "frame_index": frame_index,
                "timestamp_ms": int(round(frame_index / source_fps * 1000.0)),
                "raw_candidate_count": 0,
                "raw_response_candidate_count": 0,
            },
        )
        tracks = [
            observation_track_row(observation, prompt)
            for observation in accepted_by_frame.get(frame_index, [])
        ]
        output_rows.append(
            {
                "frame_index": frame_index,
                "timestamp_ms": raw_row["timestamp_ms"],
                "raw_candidate_count": raw_row["raw_candidate_count"],
                "raw_response_candidate_count": raw_row["raw_response_candidate_count"],
                "retained_candidate_count": len(tracks),
                "tracks": tracks,
                "rejected_tracks": rejected_by_frame.get(frame_index, []),
            }
        )
    nonempty_frames = sum(bool(row["tracks"]) for row in output_rows)

    jsonl_path = args.output_dir / "bboxes.jsonl"
    write_jsonl(jsonl_path, output_rows)
    audit_path = args.output_dir / "track_audit.json"
    audit_payload = {
        "prompt": prompt,
        "prompt_preset": args.prompt_preset,
        "response_output_keys": sorted(response_output_keys),
        "prompt_score_sources": sorted(prompt_score_sources),
        "selection": selection_audit,
        "chunk_stitching": chunk_stitch_audit,
        "chunk_continuity": args._chunk_continuity_audit,
        "continuous_state": args._continuous_state_audit,
        "semantic_prompt_verification": semantic_verification_audit,
        "temporal_filter": temporal_audit,
        "optical_flow_assist": flow_audit,
        "policy": (
            "global anonymous track lock with independent glove and bare semantic "
            "votes; no cross-prompt score comparison or per-frame area replacement; "
            "only policy-qualified bare evidence and temporally uncertain frames remain empty"
        ),
        "semantic_debug": semantic_debug_paths,
    }
    if not args.no_mask_previews:
        write_sam_mask_previews(
            args,
            predictor=predictor,
            prompt=prompt,
            raw_preview_path=raw_preview_path,
            preview_path=preview_path,
            accepted_by_frame=accepted_by_frame,
            rejected_by_frame=rejected_by_frame,
        )
    if semantic_debug_paths is not None:
        debug_dir = Path(semantic_debug_paths["evidence"]).parent
        if args.no_mask_previews:
            primary_debug_path = debug_dir / "primary_prompt_raw.mp4"
            write_prompt_raw_mask_preview(
                args,
                predictor=predictor,
                prompt=prompt,
                preview_path=primary_debug_path,
                label="primary gloved prompt raw masks",
            )
            semantic_debug_paths["primary_prompt_raw_preview"] = str(primary_debug_path)
        else:
            semantic_debug_paths["primary_prompt_raw_preview"] = str(raw_preview_path)
        for category, prompts in (
            ("glove", glove_verifier_prompts),
            ("bare", bare_verifier_prompts),
        ):
            preview_key = f"{category}_verifier_raw_previews"
            for index, verifier_prompt in enumerate(prompts):
                debug_path = debug_dir / (
                    f"{category}_{index:02d}_{safe_file_token(verifier_prompt)}.mp4"
                )
                write_prompt_raw_mask_preview(
                    args,
                    predictor=predictor,
                    prompt=verifier_prompt,
                    preview_path=debug_path,
                    label=f"{category} verifier: {verifier_prompt}",
                )
                semantic_debug_paths[preview_key].append(str(debug_path))
    atomic_write_text(
        audit_path,
        json.dumps(audit_payload, ensure_ascii=True, separators=(",", ":")),
    )
    summary = {
        "status": "complete",
        "resource": str(args.resource),
        "sam_version": args.sam_version,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "prompt_preset": args.prompt_preset,
        "prompt": prompt,
        "prompt_candidates": preset["candidates"],
        "run_config": run_config,
        "input_color_audit": input_color_audit,
        "resolved_session_policy": {
            "resource_frame_count": reader.length,
            "offload_video_to_cpu": primary_session_request["offload_video_to_cpu"],
            "offload_state_to_cpu": primary_session_request["offload_state_to_cpu"],
            "chunk_count": len(chunk_plan),
            "chunks": chunk_plan,
            "continuity_enabled": bool(args.chunk_continuity),
            "continuity": args._chunk_continuity_audit,
            "continuous_state_memory": args.continuous_state_memory,
            "continuous_state": args._continuous_state_audit,
        },
        "predictor_runtime": predictor_runtime.audit(),
        "tracked_frames": len(output_rows),
        "nonempty_frames": nonempty_frames,
        "nonempty_rate": nonempty_frames / len(output_rows) if output_rows else 0.0,
        "filter_stats": filter_stats,
        "optical_flow_assist": flow_audit,
        "selection_policy": (
            "semantic_prompt_vote_global_track_lock_with_bare_rejection_and_"
            "retrospective_temporal_filter_and_optional_bidirectional_lk"
        ),
        "anonymous_tracks": True,
        "left_right_assigned": False,
        "query_id_semantics": "SAM internal anonymous object ID; no handedness meaning",
        "preview_semantics": (
            "preview.mp4 overlays accepted prompt-validated SAM masks; "
            "raw_sam_preview.mp4 overlays all raw primary-prompt SAM masks; "
            "flow-only bridge boxes are recorded in bboxes.jsonl but have no synthetic mask"
            if not args.no_mask_previews
            else "SAM mask previews disabled by --no-mask-previews"
        ),
        "preview": str(preview_path) if not args.no_mask_previews else None,
        "raw_sam_preview": str(raw_preview_path) if not args.no_mask_previews else None,
        "bboxes": str(jsonl_path),
        "track_audit": str(audit_path),
        "semantic_debug": semantic_debug_paths,
    }
    atomic_write_text(summary_path, json.dumps(summary, indent=2))
    in_progress_path.unlink(missing_ok=True)
    if not args.no_mask_previews:
        print(f"Raw SAM masks: {raw_preview_path}")
        print(f"Accepted SAM masks: {preview_path}")
    print(f"Track audit: {audit_path}")
    if semantic_debug_paths is not None:
        print(f"Semantic debug evidence: {semantic_debug_paths['evidence']}")
    print(
        f"Tracked {len(output_rows)} frames; nonempty rate={summary['nonempty_rate']:.3f}; "
        f"locked IDs={selected_track_ids}"
    )
    return 0


def main() -> int:
    predictor_runtime = SamPredictorRuntime()
    try:
        return run_tracking(parse_args(), predictor_runtime=predictor_runtime)
    finally:
        predictor_runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
