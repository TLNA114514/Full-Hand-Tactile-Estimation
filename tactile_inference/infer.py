#!/usr/bin/env python3
"""Portable image/video to canonical tactile inference pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from .media import (
        assign_track_sides,
        assign_track_sides_interactively,
        load_obj,
        load_palm_support,
        load_sam_tracks,
        render_query_output,
        stage_input,
        tactile_crop,
        write_track_preview,
    )
    from .runtime_model import load_runtime_model
except ImportError:  # Direct execution through run.sh.
    from media import (
        assign_track_sides,
        assign_track_sides_interactively,
        load_obj,
        load_palm_support,
        load_sam_tracks,
        render_query_output,
        stage_input,
        tactile_crop,
        write_track_preview,
    )
    from runtime_model import load_runtime_model


PATH_FIELDS = (
    ("tactile", "checkpoint"),
    ("tactile", "dino_weights"),
    ("tactile", "mesh_obj"),
    ("tactile", "palm_faces"),
    ("sam3", "python"),
    ("sam3", "tracker"),
    ("sam3", "checkpoint"),
)
_ACTIVE_CHILD: subprocess.Popen | None = None


def _resolve_config_path(value: str, config_dir: Path) -> str:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve(strict=False))


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The inference config must be a JSON object")
    for section, key in PATH_FIELDS:
        value = config.get(section, {}).get(key)
        if value:
            config[section][key] = _resolve_config_path(value, path.parent)
    config["_config_path"] = str(path)
    return config


def _required_path(config: dict[str, Any], section: str, key: str) -> Path:
    value = config.get(section, {}).get(key)
    if not value:
        raise ValueError(f"Missing config field {section}.{key}")
    path = Path(value).expanduser().resolve(strict=False)
    if not path.exists():
        raise FileNotFoundError(f"Config path does not exist: {section}.{key}={path}")
    return path


def doctor(config: dict[str, Any]) -> None:
    print(f"Config: {config['_config_path']}")
    for section, key in PATH_FIELDS:
        path = _required_path(config, section, key)
        print(f"[ok] {section}.{key}: {path}")
    checkpoint = torch.load(
        _required_path(config, "tactile", "checkpoint"),
        map_location="cpu",
    )
    if checkpoint.get("format") != "tactile_trainable_v2":
        raise ValueError("Configured tactile checkpoint is not tactile_trainable_v2")
    model_config = dict(checkpoint.get("model_config", {}) or {})
    for key in ("tactile_head_type", "pool_layout", "input_resolution", "bbox_rescale_factor"):
        value = checkpoint.get(key, model_config.get(key))
        print(f"[checkpoint] {key}: {value}")
    tracker = _required_path(config, "sam3", "tracker")
    preset_path = tracker.with_name("prompt_presets.json")
    if not preset_path.is_file():
        raise FileNotFoundError(
            f"SAM3 tracker prompt presets are missing beside the tracker: {preset_path}"
        )
    presets = json.loads(preset_path.read_text(encoding="utf-8"))
    preset_name = str(config.get("sam3", {}).get("prompt_preset", "gloved"))
    if preset_name not in presets:
        raise ValueError(f"Unknown configured SAM3 prompt preset: {preset_name!r}")
    primary_prompt = str(presets[preset_name].get("primary", ""))
    prompt_override = config.get("sam3", {}).get("prompt")
    print(f"[ok] sam3.prompt_presets: {preset_path}")
    print(
        f"[sam3] preset={preset_name}, primary={primary_prompt!r}, "
        f"override={prompt_override!r}"
    )
    if not torch.cuda.is_available():
        print("[warning] CUDA is unavailable in the tactile environment")
    else:
        print(f"[ok] tactile CUDA devices: {torch.cuda.device_count()}")
    sam_python = _required_path(config, "sam3", "python")
    probe = subprocess.run(
        [str(sam_python), "-c", "import torch, sam3; print(torch.cuda.is_available())"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"[ok] SAM3 environment CUDA: {probe.stdout.strip()}")


def _terminate_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def _handle_termination(signum, _frame) -> None:
    if _ACTIVE_CHILD is not None:
        _terminate_process_group(_ACTIVE_CHILD)
    raise SystemExit(128 + int(signum))


def run_sam3(
    config: dict[str, Any],
    frame_dir: Path,
    output_dir: Path,
    *,
    is_image: bool,
    prompt_preset: str | None,
    prompt_override: str | None,
) -> Path:
    sam = config["sam3"]
    sam_python = _required_path(config, "sam3", "python")
    tracker = _required_path(config, "sam3", "tracker")
    checkpoint = _required_path(config, "sam3", "checkpoint")
    max_hands = int(sam.get("max_hands", 2))
    if max_hands not in (1, 2):
        raise ValueError("sam3.max_hands must be 1 or 2")
    command = [
        str(sam_python),
        "-u",
        str(tracker),
        "--resource",
        str(frame_dir),
        "--output-dir",
        str(output_dir),
        "--dataset",
        "generic",
        "--expected-gloved-hands",
        str(max_hands),
        "--max-objects",
        str(max_hands),
        "--checkpoint",
        str(checkpoint),
        "--sam-version",
        str(sam.get("version", "sam3")),
        "--prompt-preset",
        str(prompt_preset or sam.get("prompt_preset", "gloved")),
        "--min-track-frames",
        "1" if is_image else str(int(sam.get("min_track_frames", 2))),
        "--video-chunk-frames",
        str(int(sam.get("video_chunk_frames", 256))),
        "--video-chunk-overlap",
        str(int(sam.get("video_chunk_overlap", 32))),
        "--bare-verification-mode",
        str(sam.get("bare_verification_mode", "filter")),
        "--overwrite",
        "--no-input-rgb-samples",
    ]
    if not bool(sam.get("save_mask_previews", True)):
        command.append("--no-mask-previews")
    resolved_prompt = prompt_override or sam.get("prompt")
    if resolved_prompt:
        command.extend(["--prompt", str(resolved_prompt)])
    if bool(sam.get("flow_assist", False)):
        command.extend(["--flow-assist", "--flow-bridge-policy", "short_bridge"])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(sam.get("gpu", "0"))
    environment["PYTHONUNBUFFERED"] = "1"
    print("SAM3 command:", " ".join(command), flush=True)
    global _ACTIVE_CHILD
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
    )
    _ACTIVE_CHILD = process
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        _terminate_process_group(process)
        raise
    finally:
        _ACTIVE_CHILD = None
    if return_code != 0:
        raise RuntimeError(f"SAM3 tracking failed with exit code {return_code}")
    bbox_path = output_dir / "bboxes.jsonl"
    if not bbox_path.is_file():
        raise FileNotFoundError(f"SAM3 did not create {bbox_path}")
    return bbox_path


def _parse_render_size(value: Any) -> tuple[int, int]:
    if isinstance(value, str):
        pieces = value.lower().split("x")
    else:
        pieces = list(value)
    if len(pieces) != 2:
        raise ValueError("render_size must use WIDTHxHEIGHT")
    size = tuple(int(piece) for piece in pieces)
    if min(size) < 128:
        raise ValueError("render_size dimensions must be at least 128")
    return size


def _selected_track_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(piece.strip()) for piece in value.split(",") if piece.strip()}


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def infer_queries(
    config: dict[str, Any],
    frames,
    query_specs: list[dict[str, Any]],
    output_dir: Path,
    *,
    checkpoint_override: str | None,
    dino_override: str | None,
    no_render: bool,
) -> dict[str, Any]:
    tactile = config["tactile"]
    checkpoint = Path(checkpoint_override or tactile["checkpoint"])
    dino_weights = Path(dino_override or tactile["dino_weights"])
    device = torch.device(str(tactile.get("device", "cuda:0")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DINOv3 H+ tactile model")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Loading tactile model on {device}...", flush=True)
    model, metadata = load_runtime_model(
        checkpoint,
        dino_weights,
        device,
        verify_backbone_sha256=bool(tactile.get("verify_backbone_sha256", False)),
    )
    bbox_scale = float(metadata["bbox_rescale_factor"])
    resolution = tuple(int(value) for value in metadata["input_resolution"])
    batch_size = int(tactile.get("batch_size", 8))
    if batch_size < 1:
        raise ValueError("tactile.batch_size must be positive")
    precision = str(tactile.get("precision", "bf16"))
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("tactile.precision must be bf16, fp16, or fp32")
    query_outputs = {}
    jobs = []
    for query_index, spec in enumerate(query_specs):
        boxes = np.asarray(spec["bboxes"], dtype=np.float32)
        detected = np.isfinite(boxes).all(axis=1)
        query_dir = output_dir / spec["name"]
        query_dir.mkdir(parents=True, exist_ok=True)
        pressure = np.lib.format.open_memmap(
            query_dir / "pressure_raw.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(frames.paths), model.tactile_dim),
        )
        pressure[:] = 0.0
        crops = np.full_like(boxes, np.nan)
        query_outputs[spec["name"]] = {
            "spec": spec,
            "pressure": pressure,
            "detected": detected,
            "crop_boxes": crops,
            "query_dir": query_dir,
        }
        for frame_index in np.flatnonzero(detected):
            frame_index = int(frame_index)
            jobs.append((query_index, spec["name"], frame_index))
    print(f"Running {len(jobs)} query-frame predictions in batches of {batch_size}...", flush=True)
    with torch.inference_mode():
        for start in range(0, len(jobs), batch_size):
            batch_jobs = jobs[start : start + batch_size]
            patches = []
            for _query_index, query_name, frame_index in batch_jobs:
                frame = cv2.imread(str(frames.paths[frame_index]), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f"Could not decode staged frame: {frames.paths[frame_index]}")
                values = query_outputs[query_name]
                patch, crop = tactile_crop(
                    frame,
                    values["spec"]["bboxes"][frame_index],
                    values["spec"]["side"],
                    resolution,
                    bbox_scale,
                )
                values["crop_boxes"][frame_index] = crop
                patches.append(patch)
            images = torch.from_numpy(np.stack(patches)).to(
                device=device,
                non_blocking=True,
            )
            with _autocast(device, precision):
                prediction = model(images)
            prediction = prediction.float().cpu().numpy()
            for job, values in zip(batch_jobs, prediction):
                query_outputs[job[1]]["pressure"][job[2]] = values
            print(f"  tactile {min(start + len(batch_jobs), len(jobs))}/{len(jobs)}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mesh_path = _required_path(config, "tactile", "mesh_obj")
    palm_path = _required_path(config, "tactile", "palm_faces")
    vertices, faces = load_obj(mesh_path)
    palm_mask, palm_faces = load_palm_support(palm_path, len(vertices))
    if len(vertices) != int(metadata["tactile_dim"]):
        raise ValueError(
            f"Mesh has {len(vertices)} vertices, model predicts {metadata['tactile_dim']}"
        )
    render_config = config.get("render", {})
    render_size = _parse_render_size(render_config.get("size", "720x1280"))
    summaries = []
    for spec in query_specs:
        values = query_outputs[spec["name"]]
        raw_pressure = values["pressure"]
        raw_pressure.flush()
        query_dir = values["query_dir"]
        masked_pressure = np.lib.format.open_memmap(
            query_dir / "pressure_palm_masked.npy",
            mode="w+",
            dtype=np.float32,
            shape=raw_pressure.shape,
        )
        for start in range(0, len(raw_pressure), 1024):
            masked_pressure[start : start + 1024] = (
                raw_pressure[start : start + 1024] * palm_mask[None]
            )
        masked_pressure.flush()
        np.save(query_dir / "bbox_tight.npy", np.asarray(spec["bboxes"], np.float32))
        np.save(query_dir / "bbox_crop12.npy", values["crop_boxes"])
        np.save(query_dir / "detected.npy", values["detected"])
        if not no_render:
            render_query_output(
                query_dir=query_dir,
                frames=frames,
                pressure=masked_pressure,
                bboxes=np.asarray(spec["bboxes"], np.float32),
                crop_boxes=values["crop_boxes"],
                detected=values["detected"],
                side=spec["side"],
                label=spec["name"],
                vertices=vertices,
                faces=faces,
                palm_faces=palm_faces,
                render_size=render_size,
                display_floor=float(render_config.get("display_floor", 0.05)),
                temporal_alpha=float(render_config.get("temporal_alpha", 0.0)),
            )
        summaries.append(
            {
                "name": spec["name"],
                "track_id": int(spec["track_id"]),
                "side": spec["side"],
                "side_source": spec["side_source"],
                "detected_frames": int(values["detected"].sum()),
                "frame_count": len(frames.paths),
                "mean_predicted_volume_detected": float(
                    masked_pressure[values["detected"]].sum(axis=1).mean()
                ) if values["detected"].any() else 0.0,
                "output_dir": str(query_dir),
            }
        )
    return {"model": metadata, "queries": summaries}


def _prepare_output(path: Path, overwrite: bool) -> Path:
    path = path.expanduser().resolve(strict=False)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite: {path}")
        if path == Path(path.anchor):
            raise ValueError("Refusing to overwrite a filesystem root")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 hand queries and crop1.2 tactile inference on an image or video"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dino-weights")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-bboxes", type=Path, help="Reuse an existing SAM3 bboxes.jsonl")
    parser.add_argument("--prompt-preset", choices=("gloved", "bare"))
    parser.add_argument("--sam-prompt", help="Override the SAM3 preset's primary text prompt")
    parser.add_argument(
        "--handedness",
        choices=("auto", "left", "right", "both", "interactive"),
        default="auto",
        help=(
            "Canonical crop orientation. 'both' emits both orientations for every query; "
            "'interactive' asks for each detected track after SAM3 finishes."
        ),
    )
    parser.add_argument("--track-ids", help="Optional comma-separated SAM track IDs")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)
    args = parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config["tactile"]["checkpoint"] = str(Path(args.checkpoint).expanduser().resolve(strict=False))
    if args.dino_weights:
        config["tactile"]["dino_weights"] = str(Path(args.dino_weights).expanduser().resolve(strict=False))
    if args.sam_checkpoint:
        config["sam3"]["checkpoint"] = str(Path(args.sam_checkpoint).expanduser().resolve(strict=False))
    if args.doctor:
        doctor(config)
        return
    if args.input is None or args.output is None:
        raise ValueError("--input and --output are required unless --doctor is used")
    input_path = args.input.expanduser().resolve(strict=True)
    output_candidate = args.output.expanduser().resolve(strict=False)
    if input_path == output_candidate or input_path.is_relative_to(output_candidate):
        raise ValueError("Output must not be the input path or one of its parents")
    output_dir = _prepare_output(args.output, args.overwrite)
    started = time.time()
    frames = stage_input(
        input_path,
        output_dir / "input_frames",
        jpeg_quality=int(config.get("input", {}).get("frame_jpeg_quality", 95)),
    )
    if args.sam_bboxes:
        source_bbox_path = args.sam_bboxes.expanduser().resolve(strict=True)
        reused_dir = output_dir / "sam3"
        reused_dir.mkdir(parents=True, exist_ok=True)
        bbox_path = reused_dir / "bboxes.jsonl"
        shutil.copy2(source_bbox_path, bbox_path)
        sam_source = "existing_jsonl"
    else:
        bbox_path = run_sam3(
            config,
            output_dir / "input_frames",
            output_dir / "sam3",
            is_image=frames.is_image,
            prompt_preset=args.prompt_preset,
            prompt_override=args.sam_prompt,
        )
        sam_source = "pipeline"
    tracks = load_sam_tracks(bbox_path, len(frames.paths))
    selected_ids = _selected_track_ids(args.track_ids)
    if selected_ids is not None:
        missing = selected_ids - set(tracks)
        if missing:
            raise ValueError(f"Requested track IDs are absent: {sorted(missing)}")
        tracks = {track_id: boxes for track_id, boxes in tracks.items() if track_id in selected_ids}
    if args.handedness == "interactive":
        preview_path = write_track_preview(
            frames,
            tracks,
            output_dir / "sam3" / "handedness_preview.jpg",
        )
        query_specs = assign_track_sides_interactively(tracks, preview_path)
    else:
        query_specs = assign_track_sides(
            tracks,
            args.handedness,
            str(config.get("tactile", {}).get("single_hand_default", "right")),
        )
    result = infer_queries(
        config,
        frames,
        query_specs,
        output_dir,
        checkpoint_override=args.checkpoint,
        dino_override=args.dino_weights,
        no_render=args.no_render,
    )
    manifest = {
        "schema": "standalone_sam3_crop12_tactile_v1",
        "config": config["_config_path"],
        "resolved_config": {
            key: value for key, value in config.items() if not key.startswith("_")
        },
        "input": str(frames.source),
        "input_type": "image" if frames.is_image else "video_or_frame_directory",
        "frame_count": len(frames.paths),
        "fps": frames.fps,
        "frame_size": [frames.width, frames.height],
        "display_rotation": frames.rotation,
        "sam3_bboxes": str(bbox_path),
        "sam3_source": sam_source,
        "sam3_prompt_preset": args.prompt_preset or config.get("sam3", {}).get("prompt_preset"),
        "sam3_prompt_override": args.sam_prompt or config.get("sam3", {}).get("prompt"),
        "handedness_mode": args.handedness,
        "elapsed_seconds": time.time() - started,
        **result,
    }
    (output_dir / "inference_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Inference complete: {output_dir}")
    for query in result["queries"]:
        print(
            f"  {query['name']}: side={query['side']} "
            f"detected={query['detected_frames']}/{query['frame_count']} "
            f"output={query['output_dir']}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
