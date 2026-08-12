#!/usr/bin/env python3
"""Evaluate the official TouchAnything model in the canonical tactile space.

The TouchAnything model is intentionally left on its native inference path:
raw chest/wrist temporal clips plus raw WiLoR pose input. Ego-only and all
official wrist-view combinations are supported without materializing official
HDF5. Only its predicted 21x21 pressure grids are converted to the 13,614-
vertex canonical representation used by ``eval_tactile_fast.py``.

The evaluated sample universe and ground truth come from the same processed
query manifest used by the tactile regressor. This keeps SAM3 query filtering,
frame/hand selection, and target construction identical across methods.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from collections import defaultdict
from importlib.machinery import PathFinder
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
TA_ROOT = REPO_ROOT / "TouchAnything"
TA_CORE = TA_ROOT / "scripts" / "core"
FT_ROOT = REPO_ROOT / "hamer_tactile_ft"
HAMER_ROOT = REPO_ROOT / "hamer"

# Keep local modules with generic names such as ``train`` and ``dataset`` ahead
# of hamer/train.py and TouchAnything/scripts/core/train.py. The repository root
# remains available for packages such as ``preprocess``.
import_paths = (FT_ROOT, TA_CORE, TA_ROOT, HAMER_ROOT, REPO_ROOT)
for path in reversed(import_paths):
    text = str(path)
    while text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

expected_modules = {
    "train": FT_ROOT / "train.py",
    "dataset": FT_ROOT / "dataset.py",
    "convert_to_hdf5": TA_CORE / "convert_to_hdf5.py",
    "src": TA_ROOT / "src/__init__.py",
    "hamer": HAMER_ROOT / "hamer/__init__.py",
    "preprocess": REPO_ROOT / "preprocess/__init__.py",
}
for module_name, expected_path in expected_modules.items():
    spec = PathFinder.find_spec(module_name, sys.path)
    origin = None if spec is None else spec.origin
    if origin is None or Path(origin).resolve() != expected_path.resolve():
        raise ImportError(
            f"Module {module_name!r} resolved to {origin!r}, expected {expected_path}"
        )

from convert_to_hdf5 import extract_video_frames, load_wilor_jsonl  # noqa: E402
from src.data import get_transforms  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import load_config_with_base  # noqa: E402

from dataset import OpenTouchTactileDataset  # noqa: E402
from hdf5_storage import open_readonly  # noqa: E402
from eval_tactile_fast import (  # noqa: E402
    _empty_diagnostics,
    _empty_eval_result,
    _empty_stats,
    _load_model_cfg,
    _merge_eval_results,
    _stats_summary,
    _trim_frame_diagnostics,
    _update_diagnostics,
    _update_stats,
    _write_diagnostic_outputs,
)
from process_lifecycle import initialize_worker_parent_death_signal  # noqa: E402
from tactile_metrics import (  # noqa: E402
    CompactTouchAnythingProtocolAccumulator,
    TOUCHANYTHING_CONTACT_THRESHOLD,
    TOUCHANYTHING_MIN_CONTACT_RATIO,
    summarize_compact_touchanything_protocol,
    touchanything_protocol_frame_stats,
    touchanything_protocol_group_key,
)


MAPPING_SIGMA = 0.005
MAPPING_SCHEMA = "touchanything_grid_to_subdiv_v1"


def _sha256_file(path: os.PathLike[str] | str, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_key(value: str) -> str:
    value = str(value).replace("\\", "/").strip().strip("/")
    if value.lower().endswith((".hdf5", ".h5")):
        value = value.rsplit(".", 1)[0]
    return "/".join(part for part in value.split("/") if part not in ("", "."))


def _resolve_config_path(value: str | None, *, required: bool = False) -> Path | None:
    if not value:
        if required:
            raise ValueError("A required path was not provided")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidate = TA_ROOT / path
        path = candidate if candidate.exists() else path
    path = path.resolve(strict=False)
    if required and not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_official_config(config_path: Path) -> dict:
    config = load_config_with_base(str(config_path))
    config = copy.deepcopy(config)
    for key in ("data_root", "split_file"):
        value = config.get("data", {}).get(key)
        if value:
            config["data"][key] = str(_resolve_config_path(value))
    pretrained = config.get("model", {}).get("vision_encoder", {}).get("pretrained_path")
    if pretrained:
        config["model"]["vision_encoder"]["pretrained_path"] = str(
            _resolve_config_path(pretrained)
        )
    return config


def _mapping_asset_paths() -> list[Path]:
    return [
        REPO_ROOT / "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj",
        REPO_ROOT / "opentouch/preprocess/scratch/auto_calibrated_palm_subdiv_faces.json",
        TA_ROOT / "scripts/tools/mano_visualization/ta_to_mano_mapping_left_visual.json",
        TA_ROOT / "scripts/tools/mano_visualization/ta_to_mano_mapping_right_visual.json",
    ]


def _mapping_cache_path(cache_dir: Path) -> Path:
    digest = hashlib.sha256()
    digest.update(MAPPING_SCHEMA.encode("ascii"))
    digest.update(repr(MAPPING_SIGMA).encode("ascii"))
    for path in _mapping_asset_paths():
        if not path.is_file():
            raise FileNotFoundError(f"Canonical mapping asset is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return cache_dir / f"ta_grid_to_subdiv_{digest.hexdigest()[:16]}.npz"


def _build_mapping_cache(cache_path: Path) -> None:
    """Build the exact geodesic-Gaussian operator used by TA preprocessing."""

    from preprocess.touchanything._gaussian import load_ta_mesh_and_compute_dist_cpu

    print(f"Building canonical TouchAnything mapping once: {cache_path}", flush=True)
    left = load_ta_mesh_and_compute_dist_cpu(str(REPO_ROOT), hand="left", sigma=MAPPING_SIGMA)
    right = load_ta_mesh_and_compute_dist_cpu(str(REPO_ROOT), hand="right", sigma=MAPPING_SIGMA)
    if int(left.V_total) != int(right.V_total):
        raise RuntimeError("Left/right mapping operators use different canonical meshes")
    if not torch.equal(left.palm_vertices_tensor_cpu, right.palm_vertices_tensor_cpu):
        raise RuntimeError("Left/right mapping operators use different palm masks")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(
        f".{cache_path.name}.tmp.{os.getpid()}.{time.time_ns()}.npz"
    )
    try:
        np.savez_compressed(
            temporary,
            schema=np.asarray(MAPPING_SCHEMA),
            sigma=np.asarray(MAPPING_SIGMA, dtype=np.float64),
            vertex_count=np.asarray(int(left.V_total), dtype=np.int64),
            palm_vertices=left.palm_vertices_tensor_cpu.numpy().astype(np.int64),
            left_rows=left.valid_rows_cpu.numpy().astype(np.int64),
            left_cols=left.valid_cols_cpu.numpy().astype(np.int64),
            left_weights=left.weights_tensor_cpu.numpy().astype(np.float32),
            right_rows=right.valid_rows_cpu.numpy().astype(np.int64),
            right_cols=right.valid_cols_cpu.numpy().astype(np.int64),
            right_weights=right.weights_tensor_cpu.numpy().astype(np.float32),
        )
        with np.load(temporary, allow_pickle=False) as payload:
            if str(payload["schema"].item()) != MAPPING_SCHEMA:
                raise RuntimeError("New mapping cache failed validation")
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_mapping_cache(cache_dir: Path) -> Path:
    cache_path = _mapping_cache_path(cache_dir)
    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                valid = (
                    str(payload["schema"].item()) == MAPPING_SCHEMA
                    and math.isclose(float(payload["sigma"]), MAPPING_SIGMA)
                    and int(payload["vertex_count"]) == 13614
                )
            if valid:
                return cache_path
        except Exception:
            pass
        raise RuntimeError(
            f"Existing canonical mapping cache is invalid; remove it and retry: {cache_path}"
        )
    _build_mapping_cache(cache_path)
    return cache_path


class GridToCanonicalMapper:
    def __init__(self, cache_path: str, device: torch.device, batch_size: int):
        with np.load(cache_path, allow_pickle=False) as payload:
            self.vertex_count = int(payload["vertex_count"])
            palm_vertices = torch.from_numpy(payload["palm_vertices"].astype(np.int64))
            self.palm_vertices = palm_vertices.to(device=device)
            self.palm_mask = np.zeros((self.vertex_count,), dtype=bool)
            self.palm_mask[palm_vertices.numpy()] = True
            self.operators = {}
            for hand in ("left", "right"):
                self.operators[hand] = (
                    torch.from_numpy(payload[f"{hand}_rows"].astype(np.int64)).to(device),
                    torch.from_numpy(payload[f"{hand}_cols"].astype(np.int64)).to(device),
                    torch.from_numpy(payload[f"{hand}_weights"].astype(np.float32)).to(device),
                )
        self.device = device
        self.batch_size = max(1, int(batch_size))

    @torch.no_grad()
    def __call__(self, grids: np.ndarray, hand: str) -> np.ndarray:
        grids = np.asarray(grids, dtype=np.float32)
        if grids.ndim != 3:
            raise ValueError(f"Expected pressure grids [N,H,W], got {grids.shape}")
        if grids.shape[1:] != (21, 21):
            tensor = torch.from_numpy(np.nan_to_num(grids, nan=0.0)).unsqueeze(1)
            tensor = F.interpolate(tensor, size=(21, 21), mode="bilinear", align_corners=False)
            grids = tensor[:, 0].numpy()
        rows, cols, weights = self.operators[hand]
        outputs = []
        for start in range(0, len(grids), self.batch_size):
            grid = torch.as_tensor(
                np.nan_to_num(grids[start : start + self.batch_size], nan=0.0),
                dtype=torch.float32,
                device=self.device,
            ).clamp_(0.0, 1.0)
            active = grid[:, rows, cols]
            dense = torch.amax(active.unsqueeze(2) * weights.unsqueeze(0), dim=1)
            masked = torch.zeros_like(dense)
            masked[:, self.palm_vertices] = dense[:, self.palm_vertices]
            outputs.append(masked.clamp_(0.0, 1.0).cpu().numpy())
        if not outputs:
            return np.zeros((0, self.vertex_count), dtype=np.float32)
        return np.concatenate(outputs, axis=0)


def _load_checkpoint(model, checkpoint_path: str, allow_non_strict: bool) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise TypeError("TouchAnything checkpoint has no state_dict")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        message = (
            f"Checkpoint/config mismatch: missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
        if not allow_non_strict:
            raise RuntimeError(message)
        print(f"WARNING: {message}", flush=True)
    return {
        "epoch": checkpoint.get("epoch"),
        "loss": checkpoint.get("loss"),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _query_source_sequence_key(record: dict) -> str:
    """Reduce a query-scoped key to the official Scene/Task/Trajectory key."""

    value = _canonical_key(record.get("sequence_key", ""))
    parts = [part for part in value.split("/") if part]
    if parts and parts[0].lower().replace("_", "") == "touchanything":
        parts.pop(0)
    if parts and parts[0].lower() in {
        "train",
        "val",
        "test",
        "test_seen",
        "test_unseen",
    }:
        parts.pop(0)

    query_alias = str(
        record.get("query_alias", record.get("hand", ""))
    ).strip().lower()
    if len(parts) > 3 and parts[-1].lower() == query_alias:
        parts.pop()
    if len(parts) < 3:
        raise ValueError(
            f"Cannot derive Scene/Task/Trajectory from query key {value!r}"
        )
    return _canonical_key("/".join(parts[-3:]))


def _resolve_raw_sequence_dir(raw_root: Path, split: str, sequence_key: str) -> Path:
    relative = Path(sequence_key)
    candidates = (raw_root / relative, raw_root / split / relative)
    required = (
        "chest.mp4",
        "left.mp4",
        "right.mp4",
        "pressure_grids.npz",
        "wilor_hands.json",
    )
    complete = [
        candidate.resolve()
        for candidate in candidates
        if candidate.is_dir()
        and all((candidate / filename).is_file() for filename in required)
    ]
    if len(complete) == 1:
        return complete[0]
    if len(complete) > 1 and len(set(complete)) == 1:
        return complete[0]
    missing_details = []
    for candidate in candidates:
        missing = [name for name in required if not (candidate / name).is_file()]
        missing_details.append(f"{candidate}: missing={missing}")
    raise FileNotFoundError(
        f"Cannot locate raw TouchAnything sequence {sequence_key!r}. "
        + " | ".join(missing_details)
    )


def _match_query_groups(records: list[dict], raw_root: Path, split: str):
    groups = defaultdict(list)
    for record in records:
        groups[_query_source_sequence_key(record)].append(record)
    matched = []
    for query_key, query_records in sorted(groups.items()):
        raw_dir = _resolve_raw_sequence_dir(raw_root, split, query_key)
        matched.append((query_key, str(raw_dir), query_records))
    return matched


def _load_eval_records(args) -> tuple[list[dict], np.ndarray, dict]:
    model_cfg = _load_model_cfg((256, 192))
    bbox_manifests = [item for item in args.bbox_manifests.split(",") if item.strip()]
    dataset = OpenTouchTactileDataset(
        model_cfg,
        split=args.split,
        data_dir=str(args.processed_root),
        train=False,
        index_workers=1,
        tactile_only=True,
        input_resolution=(256, 192),
        bbox_rescale_factor=1.2,
        bbox_source_policy=args.bbox_source_policy,
        bbox_manifests=bbox_manifests,
        data_backend="sequence_hdf5",
        query_manifests=[
            {
                "path": str(args.query_manifest),
                "root": str(args.processed_root),
                "dataset": "touchanything",
            }
        ],
        hdf5_manifest_cache_dir=str(args.manifest_cache_dir),
        lazy_index_records=False,
    )
    records = list(dataset.samples)
    palm_mask = np.asarray(dataset.palm_mask, dtype=np.float32) > 0.5
    metadata = {
        "query_manifest_sha256": _sha256_file(args.query_manifest),
        "bbox_manifest_sha256": {
            str(Path(path).resolve()): _sha256_file(path) for path in bbox_manifests
        },
        "selected_query_count": len(records),
    }
    dataset._close_hdf5_handles()
    return records, palm_mask, metadata


def _read_targets(records: list[dict]) -> np.ndarray:
    by_file = defaultdict(list)
    for output_index, record in enumerate(records):
        by_file[str(record["h5_path"])].append((output_index, int(record["query_row"])))
    output = [None] * len(records)
    for path, indexed_rows in by_file.items():
        with open_readonly(path) as handle:
            target = handle["targets/pressure"]
            for output_index, row in indexed_rows:
                output[output_index] = np.asarray(target[row], dtype=np.float32)
    if any(value is None for value in output):
        raise RuntimeError("Failed to read one or more canonical ground-truth rows")
    return np.stack(output, axis=0)


def _video_frame_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return count if count > 0 else None


def _load_raw_pressure_grids(raw_dir: Path) -> dict[str, np.ndarray]:
    path = raw_dir / "pressure_grids.npz"
    with np.load(path, allow_pickle=False) as payload:
        grids = {
            "left": np.asarray(payload["left_pressure_grid"], dtype=np.float32),
            "right": np.asarray(payload["right_pressure_grid"], dtype=np.float32),
        }
    for hand, value in grids.items():
        if value.ndim != 3 or value.shape[1:] != (21, 21):
            raise ValueError(f"{path}: invalid {hand} pressure grid shape {value.shape}")
    return grids


@torch.no_grad()
def _infer_raw_trajectory_batched(
    raw_dir: str,
    model,
    device: torch.device,
    config: dict,
    *,
    batch_size: int,
    decode_gpu_id: int,
    view_config: str,
):
    """Official temporal/view inference without materializing official HDF5."""

    raw_dir = Path(raw_dir)
    requested_streams = {"ego": "chest.mp4"}
    if view_config in ("ego+left", "all"):
        requested_streams["wrist_left"] = "left.mp4"
    if view_config in ("ego+right", "all"):
        requested_streams["wrist_right"] = "right.mp4"
    frames_by_view = {}
    for view_name, filename in requested_streams.items():
        frames = extract_video_frames(raw_dir / filename, gpu_id=decode_gpu_id)
        if not frames:
            raise RuntimeError(f"Could not decode raw video: {raw_dir / filename}")
        frames_by_view[view_name] = frames
    pressure_grids = _load_raw_pressure_grids(raw_dir)

    frame_limits = [
        *(len(frames) for frames in frames_by_view.values()),
        len(pressure_grids["left"]),
        len(pressure_grids["right"]),
    ]
    for filename in ("left.mp4", "right.mp4"):
        count = _video_frame_count(raw_dir / filename)
        if count is not None:
            frame_limits.append(count)
    frame_count = min(frame_limits)
    if frame_count <= 0:
        raise RuntimeError(f"Raw trajectory has no aligned frames: {raw_dir}")
    frames_by_view = {
        view_name: frames[:frame_count]
        for view_name, frames in frames_by_view.items()
    }
    pressure_grids = {
        hand: value[:frame_count] for hand, value in pressure_grids.items()
    }

    left_xyz, right_xyz, left_valid, right_valid = load_wilor_jsonl(
        raw_dir / "wilor_hands.json", frame_count
    )
    if not (np.any(left_valid) or np.any(right_valid)):
        raise RuntimeError(f"Raw WiLoR pose file has no valid hand poses: {raw_dir}")

    transform = get_transforms(config, is_training=False)
    image_size = tuple(config["data"].get("image_size", [224, 224]))
    clip_length = int(config["data"].get("clip_length", 8))
    frame_interval = int(config["data"].get("frame_interval", 2))
    tactile_size = int(config["data"].get("tactile_size", 21))
    span = (clip_length - 1) * frame_interval + 1
    clip_starts = list(range(0, frame_count - span + 1))
    pred_sum = np.zeros(
        (frame_count, 2, tactile_size, tactile_size), dtype=np.float32
    )
    pred_count = np.zeros((frame_count,), dtype=np.int32)

    def prepare_images(frames: list[np.ndarray], starts: list[int]) -> torch.Tensor:
        clips = []
        for start in starts:
            indices = [start + index * frame_interval for index in range(clip_length)]
            resized = np.stack(
                [
                    cv2.resize(
                        np.asarray(frames[index])[..., :3],
                        (image_size[1], image_size[0]),
                    )
                    for index in indices
                ],
                axis=0,
            )
            clips.append(transform(resized))
        return torch.stack(clips, dim=0).to(device, non_blocking=True)

    invalid_pose_value = -10.0
    for batch_start in range(0, len(clip_starts), max(1, int(batch_size))):
        starts = clip_starts[batch_start : batch_start + max(1, int(batch_size))]
        view_tensors = {
            view_name: prepare_images(frames, starts)
            for view_name, frames in frames_by_view.items()
        }
        poses = []
        for start in starts:
            indices = [start + index * frame_interval for index in range(clip_length)]
            left = left_xyz[indices].copy()
            right = right_xyz[indices].copy()
            left[np.asarray(left_valid[indices]) == 0] = invalid_pose_value
            right[np.asarray(right_valid[indices]) == 0] = invalid_pose_value
            pose = np.concatenate([left, right], axis=1).astype(np.float32)
            pose[:, :, 0:2] = np.clip(pose[:, :, 0:2], -10.0, 10.0)
            pose[:, :, 2] = np.clip(pose[:, :, 2], -10.0, 100.0)
            poses.append(torch.from_numpy(pose))
        pose_tensor = torch.stack(poses, dim=0).to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            if view_config == "ego":
                output = model(frames=view_tensors["ego"], poses=pose_tensor)
            else:
                output = model(poses=pose_tensor, views=view_tensors)
        prediction = output["tactile"].float().cpu().numpy()
        for batch_index, start in enumerate(starts):
            for clip_index in range(clip_length):
                frame_index = start + clip_index * frame_interval
                pred_sum[frame_index] += prediction[batch_index, clip_index]
                pred_count[frame_index] += 1

    pred_maps = np.full_like(pred_sum, np.nan)
    valid = pred_count > 0
    pred_maps[valid] = pred_sum[valid] / pred_count[valid, None, None, None]
    return pred_maps, pressure_grids


def _finish_worker_diagnostics(diagnostics, max_frames: int, worker_rank: int) -> None:
    for key in diagnostics["frame"]:
        arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in diagnostics["frame"][key]]
        diagnostics["frame"][key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
    for key in diagnostics["frame_provenance"]:
        arrays = [np.asarray(item, dtype=object).reshape(-1) for item in diagnostics["frame_provenance"][key]]
        diagnostics["frame_provenance"][key] = (
            np.concatenate(arrays) if arrays else np.zeros(0, dtype=object)
        )
    _trim_frame_diagnostics(diagnostics, max_frames, seed=2026 + worker_rank)


def _worker_main(worker_rank, device_id, args_dict, tasks, result_queue):
    try:
        initialize_worker_parent_death_signal()
        args = SimpleNamespace(**args_dict)
        if args.dinov2_repo:
            os.environ["TOUCHANYTHING_DINOV2_REPO"] = args.dinov2_repo
        device = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(device)
        config = _load_official_config(Path(args.config))
        if args.pose_source:
            config["data"]["pose_source"] = args.pose_source
        pose_source = config["data"].get("pose_source")
        if pose_source != "wilor":
            raise ValueError(
                "The raw-data adapter currently requires pose_source='wilor' to match "
                f"the supplied checkpoint config, got {pose_source!r}"
            )

        model = build_model(config)
        checkpoint_metadata = _load_checkpoint(
            model, args.checkpoint, args.allow_non_strict_checkpoint
        )
        model.to(device).eval()
        mapper = GridToCanonicalMapper(args.mapping_cache, device, args.mapping_batch_size)
        evaluation_palm_mask = np.asarray(
            args.evaluation_palm_mask, dtype=bool
        )
        if evaluation_palm_mask.shape != (mapper.vertex_count,):
            raise ValueError(
                "Evaluation palm mask has shape "
                f"{evaluation_palm_mask.shape}, expected {(mapper.vertex_count,)}"
            )

        stats = _empty_stats()
        diagnostics = _empty_diagnostics()
        touch_stats = CompactTouchAnythingProtocolAccumulator()
        parity_checked = 0
        parity_max_abs = 0.0
        parity_all_vertices_max_abs = 0.0
        predicted_queries = 0
        missing_prediction_queries = 0

        for task_index, (sequence_key, raw_dir, records) in enumerate(tasks):
            print(
                f"[GPU {device_id}] {task_index + 1}/{len(tasks)} {sequence_key} "
                f"({len(records)} queries)",
                flush=True,
            )
            pred_maps, source_grids = _infer_raw_trajectory_batched(
                raw_dir,
                model,
                device,
                config,
                batch_size=args.inference_batch_size,
                decode_gpu_id=device_id,
                view_config=args.views,
            )

            valid_records = []
            selected_grids = {"left": [], "right": []}
            selected_positions = {"left": [], "right": []}
            for record in records:
                prediction_index = int(record.get("frame_idx", 0))
                pressure_grid_index = prediction_index
                jq_pressure_index = record.get("source_frame_idx")
                hand = str(record.get("hand", record.get("query_alias", ""))).lower()
                if hand not in ("left", "right"):
                    raise ValueError(f"Unsupported query hand {hand!r} in {sequence_key}")
                hand_index = 0 if hand == "left" else 1
                if not 0 <= prediction_index < len(pred_maps):
                    raise IndexError(
                        f"Query RGB frame {prediction_index} is outside raw trajectory "
                        f"length {len(pred_maps)} for {sequence_key}"
                    )
                if not 0 <= pressure_grid_index < len(source_grids[hand]):
                    raise IndexError(
                        f"Query pressure frame {pressure_grid_index} is outside raw pressure "
                        f"length {len(source_grids[hand])} for {sequence_key}"
                    )
                source_grid = source_grids[hand][pressure_grid_index]
                grid = pred_maps[prediction_index, hand_index].copy()
                grid[np.isnan(source_grid)] = np.nan
                if not np.isfinite(grid).any():
                    missing_prediction_queries += 1
                    continue
                selected_positions[hand].append(len(valid_records))
                selected_grids[hand].append(grid)
                copied = dict(record)
                copied["dataset"] = "TouchAnything"
                copied["sample_dir"] = (
                    f"{raw_dir}#rgb_npz_frame={prediction_index}"
                    f"&jq_frame={jq_pressure_index}&hand={hand}"
                )
                copied["sample_ref"] = copied["sample_dir"]
                copied["worker_rank"] = worker_rank
                valid_records.append(copied)

            if not valid_records:
                continue
            pred_canonical = np.zeros((len(valid_records), mapper.vertex_count), dtype=np.float32)
            for hand in ("left", "right"):
                if selected_grids[hand]:
                    mapped = mapper(np.stack(selected_grids[hand], axis=0), hand)
                    pred_canonical[np.asarray(selected_positions[hand], dtype=np.int64)] = mapped
            target = _read_targets(valid_records)

            remaining = max(0, args.verify_target_parity_samples - parity_checked)
            if remaining:
                count = min(remaining, len(valid_records))
                for index in range(count):
                    record = valid_records[index]
                    hand = str(record["hand"]).lower()
                    pressure_grid_index = int(record.get("frame_idx", 0))
                    mapped_gt = mapper(
                        source_grids[hand][pressure_grid_index][None, ...], hand
                    )[0]
                    absolute_error = np.abs(mapped_gt - target[index])
                    parity_max_abs = max(
                        parity_max_abs,
                        float(np.max(absolute_error[evaluation_palm_mask])),
                    )
                    parity_all_vertices_max_abs = max(
                        parity_all_vertices_max_abs,
                        float(np.max(absolute_error)),
                    )
                parity_checked += count
                if parity_max_abs > args.target_parity_atol:
                    raise RuntimeError(
                        "Official source-grid to canonical-GT parity failed on the "
                        "current evaluation palm mask: "
                        f"max_abs={parity_max_abs:.8g} > atol={args.target_parity_atol:.8g}. "
                        "The raw pressure grid, normalization, frame alignment, or mapping differs "
                        "from the evaluated tactile target."
                    )

            _update_stats(
                stats,
                pred_canonical,
                target,
                evaluation_palm_mask,
                args.contact_thr,
                active_thr=args.active_pressure_thr,
                background_thr=args.background_pressure_thr,
            )
            pred_palm = pred_canonical[:, evaluation_palm_mask]
            target_palm = target[:, evaluation_palm_mask]
            frame_stats = touchanything_protocol_frame_stats(
                pred_palm,
                target_palm,
                value_axis=1,
                contact_threshold=args.touchanything_contact_thr,
            )
            touch_stats.add(
                [
                    touchanything_protocol_group_key(
                        record["sequence_key"],
                        record.get("query_alias", record.get("hand", "")),
                    )
                    for record in valid_records
                ],
                [int(record.get("frame_idx", 0)) for record in valid_records],
                frame_stats,
            )
            if args.save_diagnostics:
                _update_diagnostics(
                    diagnostics,
                    pred_canonical,
                    target,
                    evaluation_palm_mask,
                    args.contact_thr,
                    args.active_pressure_thr,
                    args.diagnostic_max_frames,
                    valid_records,
                    worker_rank,
                )
            predicted_queries += len(valid_records)

        if args.save_diagnostics:
            _finish_worker_diagnostics(
                diagnostics, args.diagnostic_max_frames, worker_rank
            )
        result = _empty_eval_result()
        result.update(
            {
                "stats": stats,
                "diagnostics": diagnostics,
                "touchanything_protocol_stats": touch_stats.pack(),
            }
        )
        worker_metadata = {
            "worker_rank": worker_rank,
            "device_id": device_id,
            "trajectory_count": len(tasks),
            "predicted_query_count": predicted_queries,
            "missing_prediction_query_count": missing_prediction_queries,
            "target_parity_checked": parity_checked,
            "target_parity_max_abs": parity_max_abs,
            "target_parity_all_vertices_max_abs": parity_all_vertices_max_abs,
            "checkpoint": checkpoint_metadata,
        }
        result_queue.put((worker_rank, result, worker_metadata, None))
    except BaseException:
        result_queue.put((worker_rank, None, None, traceback.format_exc()))


def _format_report(args, stats, touch_payload, provenance) -> str:
    summary = _stats_summary(
        stats,
        touch_payload,
        touchanything_min_contact_ratio=args.touchanything_min_contact_ratio,
    )
    if summary is None:
        raise RuntimeError("Official baseline produced no valid evaluation queries")
    touch = summarize_compact_touchanything_protocol(
        [touch_payload],
        min_contact_ratio=args.touchanything_min_contact_ratio,
    )
    active_mae = stats["active_abs_sum"] / max(stats["active_count"], 1)
    background_mae = stats["background_abs_sum"] / max(stats["background_count"], 1)
    return "\n".join(
        [
            "TouchAnything Official Baseline -> Canonical Mesh Evaluation",
            "=" * 68,
            f" Split              : {args.split}",
            f" Checkpoint         : {args.checkpoint}",
            f" Official config    : {args.config}",
            f" Official inputs    : raw videos, views={args.views}, pose={provenance['pose_source']}, "
            f"clip={provenance['clip_length']}x{provenance['frame_interval']}",
            " Query/crop policy   : current tactile query manifest + SAM3 filter; "
            "bbox is not passed to TouchAnything",
            f" Output conversion  : 21x21 grid -> 13,614 subdiv vertices, "
            f"geodesic Gaussian sigma={MAPPING_SIGMA:g}",
            f" Evaluation mask    : current canonical palm "
            f"({provenance['palm_masks']['evaluation_count']} vertices); "
            f"legacy mapping support={provenance['palm_masks']['mapping_count']}",
            f" Valid query frames : {stats['total_frames']}",
            f" Overall MAE        : {summary['mae']:.6f}",
            f" Overall RMSE       : {summary['rmse']:.6f}",
            f" Contact IoU Frame-Macro: {summary['contact_iou']:.6f}",
            f" V-IoU Frame-Macro : {summary['volumetric_iou_frame_macro']:.6f}",
            f" V-IoU Split-Micro : {summary['volumetric_iou_split_micro']:.6f}",
            f" Distribution V-IoU: {summary['distribution_viou']:.6f}",
            f" Core Distribution V-IoU: {summary['core_distribution_viou']:.6f}",
            f" TA Temporal Acc    : {touch['temporal_accuracy']:.6f}",
            f" TA Contact IoU     : {touch['contact_iou']:.6f} (source-trajectory macro)",
            f" TA V-IoU           : {touch['volumetric_iou']:.6f} (source-trajectory macro)",
            f" TA MAE             : {touch['mae']:.6f} (source-trajectory macro)",
            f" TA Temporal F1     : {touch['temporal_f1']:.6f}",
            f" Pred/GT Volume     : {summary['pred_gt_volume_ratio']:.6f}",
            f" Active MAE         : {active_mae:.6f}",
            f" Background MAE     : {background_mae:.6f}",
            f" Active Recall      : {summary['active_recall']:.6f}",
            f" BG False Positive  : {summary['bg_false_positive']:.6f}",
            f" False-high excess  : "
            f"{summary['false_high_gt005_pred03_excess_volume_fraction']:.6f}",
            f" Catastrophic Over  : {summary['catastrophic_over_rate']:.6f}",
            f" GT>=0.7 mapping note: inspect diagnostics/pressure_bins.csv",
            f" Target parity      : n={provenance['target_parity_checked']}, "
            f"eval-mask max_abs={provenance['target_parity_max_abs']:.8g}, "
            f"all-vertex max_abs="
            f"{provenance['target_parity_all_vertices_max_abs']:.8g}",
            "=" * 68,
            "Fairness note: TouchAnything retains its official temporal/pose/view "
            "inputs; only sample selection, output representation, GT, and metrics "
            "are shared with the tactile-regressor evaluation.",
        ]
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run official TouchAnything inference and evaluate canonical mesh outputs"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        default=str(TA_ROOT / "configs/touchanything_with_glove_aug_wilor.yaml"),
        help="The exact model config used to create the checkpoint.",
    )
    parser.add_argument(
        "--dinov2_repo",
        default=os.environ.get("TOUCHANYTHING_DINOV2_REPO", ""),
        help=(
            "Local clone of the official facebookresearch/dinov2 source repository. "
            "The directory must contain hubconf.py; no online torch.hub download is used."
        ),
    )
    parser.add_argument("--split", required=True, choices=("test_seen", "test_unseen", "val", "train"))
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Raw EgoTouch root containing Scene/Task/Clip directories.",
    )
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--query_manifest", required=True)
    parser.add_argument("--bbox_manifests", default="")
    parser.add_argument("--bbox_source_policy", choices=("any", "sam3_only"), default="sam3_only")
    parser.add_argument(
        "--views",
        choices=("ego", "ego+left", "ego+right", "all"),
        default="ego",
    )
    parser.add_argument("--pose_source", choices=("wilor",), default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num_workers", type=int, default=0, help="0 means one process per listed GPU")
    parser.add_argument("--inference_batch_size", type=int, default=16)
    parser.add_argument("--mapping_batch_size", type=int, default=8)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mapping_cache_dir", default=str(REPO_ROOT / "hamer_tactile_ft/mapping_cache"))
    parser.add_argument("--manifest_cache_dir", default=str(REPO_ROOT / "hamer_tactile_ft/hdf5_manifest_cache"))
    parser.add_argument("--verify_target_parity_samples", type=int, default=128)
    parser.add_argument("--target_parity_atol", type=float, default=2e-5)
    parser.add_argument("--contact_thr", type=float, default=0.05)
    parser.add_argument("--active_pressure_thr", type=float, default=0.05)
    parser.add_argument("--background_pressure_thr", type=float, default=0.02)
    parser.add_argument("--touchanything_contact_thr", type=float, default=TOUCHANYTHING_CONTACT_THRESHOLD)
    parser.add_argument("--touchanything_min_contact_ratio", type=float, default=TOUCHANYTHING_MIN_CONTACT_RATIO)
    parser.add_argument("--diagnostic_max_frames", type=int, default=20000)
    parser.add_argument("--save_diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_non_strict_checkpoint", action="store_true")
    parser.add_argument("--max_trajectories", type=int, default=0)
    return parser.parse_args()


def main():
    args = _parse_args()
    args.checkpoint = str(Path(args.checkpoint).expanduser().resolve(strict=True))
    args.config = str(Path(args.config).expanduser().resolve(strict=True))
    if args.dinov2_repo:
        dinov2_repo = Path(args.dinov2_repo).expanduser().resolve(strict=True)
        if not (dinov2_repo / "hubconf.py").is_file():
            raise RuntimeError(
                f"DINOv2 source repository has no hubconf.py: {dinov2_repo}"
            )
        args.dinov2_repo = str(dinov2_repo)
        os.environ["TOUCHANYTHING_DINOV2_REPO"] = args.dinov2_repo
    args.raw_root = Path(args.raw_root).expanduser().resolve(strict=True)
    args.processed_root = Path(args.processed_root).expanduser().resolve(strict=True)
    args.query_manifest = Path(args.query_manifest).expanduser().resolve(strict=True)
    args.output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.mapping_cache_dir = Path(args.mapping_cache_dir).expanduser().resolve(strict=False)
    args.manifest_cache_dir = Path(args.manifest_cache_dir).expanduser().resolve(strict=False)
    args.manifest_cache_dir.mkdir(parents=True, exist_ok=True)

    config = _load_official_config(Path(args.config))
    pose_source = args.pose_source or config["data"].get("pose_source")

    records, palm_mask, selection_metadata = _load_eval_records(args)
    tasks = _match_query_groups(records, args.raw_root, args.split)
    if args.max_trajectories > 0:
        tasks = tasks[: args.max_trajectories]
    if not tasks:
        raise RuntimeError("No matched TouchAnything trajectories were selected")
    selected_query_count = sum(len(item[2]) for item in tasks)
    print(
        f"Selected {len(tasks)} official trajectories / {selected_query_count} canonical queries "
        f"for split={args.split}",
        flush=True,
    )

    mapping_cache = ensure_mapping_cache(args.mapping_cache_dir)
    with np.load(mapping_cache, allow_pickle=False) as payload:
        mapping_palm_mask = np.zeros((int(payload["vertex_count"]),), dtype=bool)
        mapping_palm_mask[payload["palm_vertices"].astype(np.int64)] = True
    if palm_mask.shape != mapping_palm_mask.shape:
        raise RuntimeError(
            f"Palm mask shape mismatch: evaluation={palm_mask.shape}, "
            f"mapping={mapping_palm_mask.shape}"
        )
    palm_mask_metadata = {
        "evaluation_count": int(palm_mask.sum()),
        "mapping_count": int(mapping_palm_mask.sum()),
        "intersection_count": int(np.sum(palm_mask & mapping_palm_mask)),
        "evaluation_only_count": int(np.sum(palm_mask & ~mapping_palm_mask)),
        "mapping_only_count": int(np.sum(mapping_palm_mask & ~palm_mask)),
        "policy": (
            "replicate legacy grid-to-mesh mapping, then evaluate with the "
            "current dataset palm mask"
        ),
    }
    if not np.array_equal(palm_mask, mapping_palm_mask):
        print(
            "Palm-mask note: legacy TouchAnything mapping support differs from "
            f"the current evaluation mask: {palm_mask_metadata}",
            flush=True,
        )
    gpu_ids = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must list at least one CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("Official TouchAnything baseline evaluation requires CUDA")
    worker_count = args.num_workers if args.num_workers > 0 else len(gpu_ids)
    worker_count = min(worker_count, len(gpu_ids), len(tasks))
    shards = [tasks[index::worker_count] for index in range(worker_count)]

    worker_args = vars(args).copy()
    worker_args.update(
        {
            "processed_root": str(args.processed_root),
            "raw_root": str(args.raw_root),
            "query_manifest": str(args.query_manifest),
            "output_dir": str(args.output_dir),
            "mapping_cache": str(mapping_cache),
            "mapping_cache_dir": str(args.mapping_cache_dir),
            "manifest_cache_dir": str(args.manifest_cache_dir),
            "evaluation_palm_mask": palm_mask.astype(bool).tolist(),
        }
    )
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for rank, shard in enumerate(shards):
        process = ctx.Process(
            target=_worker_main,
            args=(rank, gpu_ids[rank], worker_args, shard, result_queue),
        )
        process.start()
        processes.append(process)

    worker_results = []
    worker_metadata = []
    errors = []
    pending = set(range(len(processes)))
    try:
        while pending:
            try:
                rank, result, metadata, error = result_queue.get(timeout=1.0)
            except queue.Empty:
                for rank in list(pending):
                    if processes[rank].exitcode is not None:
                        errors.append(
                            f"Worker {rank} exited without a result (exit={processes[rank].exitcode})"
                        )
                        pending.remove(rank)
                if errors:
                    break
                continue
            pending.discard(rank)
            if error:
                errors.append(f"Worker {rank} failed:\n{error}")
            else:
                worker_results.append(result)
                worker_metadata.append(metadata)
            if errors:
                break
    except KeyboardInterrupt:
        errors.append("Interrupted")
    finally:
        if errors:
            for process in processes:
                if process.is_alive():
                    process.terminate()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        result_queue.close()
        result_queue.join_thread()
    if errors:
        raise RuntimeError("\n".join(errors))

    result = _merge_eval_results(worker_results, args.diagnostic_max_frames)
    predicted_queries = sum(item["predicted_query_count"] for item in worker_metadata)
    missing_predictions = sum(item["missing_prediction_query_count"] for item in worker_metadata)
    coverage = predicted_queries / max(selected_query_count, 1)
    if coverage < 0.995:
        raise RuntimeError(
            f"Official inference covered only {predicted_queries}/{selected_query_count} "
            f"queries ({coverage:.3%}); expected at least 99.5%"
        )

    provenance = {
        "schema": "touchanything_official_raw_adapter_canonical_eval_v1",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "config": args.config,
        "config_sha256": _sha256_file(args.config),
        "dinov2_repo": args.dinov2_repo or None,
        "dinov2_hubconf_sha256": (
            _sha256_file(Path(args.dinov2_repo) / "hubconf.py")
            if args.dinov2_repo
            else None
        ),
        "raw_root": str(args.raw_root),
        "processed_root": str(args.processed_root),
        "query_manifest": str(args.query_manifest),
        "mapping_cache": str(mapping_cache),
        "mapping_cache_sha256": _sha256_file(mapping_cache),
        "mapping_assets": {
            str(path): _sha256_file(path) for path in _mapping_asset_paths()
        },
        "mapping_sigma": MAPPING_SIGMA,
        "palm_masks": palm_mask_metadata,
        "split": args.split,
        "views": args.views,
        "pose_source": pose_source,
        "clip_length": int(config["data"].get("clip_length", 8)),
        "frame_interval": int(config["data"].get("frame_interval", 2)),
        "bbox_source_policy": args.bbox_source_policy,
        "trajectory_count": len(tasks),
        "selected_query_count": selected_query_count,
        "predicted_query_count": predicted_queries,
        "missing_prediction_query_count": missing_predictions,
        "prediction_coverage": coverage,
        "target_parity_checked": sum(item["target_parity_checked"] for item in worker_metadata),
        "target_parity_max_abs": max(
            (item["target_parity_max_abs"] for item in worker_metadata), default=0.0
        ),
        "target_parity_all_vertices_max_abs": max(
            (
                item["target_parity_all_vertices_max_abs"]
                for item in worker_metadata
            ),
            default=0.0,
        ),
        "selection": selection_metadata,
        "workers": worker_metadata,
    }
    provenance_path = args.output_dir / "official_baseline_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = _format_report(args, result["stats"], result["touchanything_protocol_stats"], provenance)
    report_path = args.output_dir / f"eval_touchanything_{args.split}.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print("\n" + report, flush=True)
    print(f"Report: {report_path}", flush=True)

    if args.save_diagnostics:
        diagnostic_args = SimpleNamespace(
            save_diagnostics=True,
            save_visualizations=False,
            diagnostics_dir=str(
                args.output_dir / f"eval_touchanything_{args.split}_diagnostics"
            ),
            report_dir=str(args.output_dir),
            report_name=report_path.name,
            active_pressure_thr=args.active_pressure_thr,
            background_pressure_thr=args.background_pressure_thr,
            touchanything_min_contact_ratio=args.touchanything_min_contact_ratio,
        )
        diagnostics_dir = _write_diagnostic_outputs(diagnostic_args, result)
        print(f"Diagnostics: {diagnostics_dir}", flush=True)


if __name__ == "__main__":
    main()
