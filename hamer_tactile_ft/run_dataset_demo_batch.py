#!/usr/bin/env python3
"""Randomly select dataset sequences and render them in parallel across GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def canonical_dataset_name(value):
    aliases = {
        "opentouch": "OpenTouch",
        "open_touch": "OpenTouch",
        "ot": "OpenTouch",
        "touchanything": "TouchAnything",
        "touch_anything": "TouchAnything",
        "egotouch": "TouchAnything",
        "ego_touch": "TouchAnything",
        "ta": "TouchAnything",
    }
    raw = str(value or "OpenTouch")
    return aliases.get(raw.lower(), raw)


def valid_bbox(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(item) for item in values)
        and values[2] - values[0] > 1.0
        and values[3] - values[1] > 1.0
    )


def sequence_candidate(sample_dir, split, hand):
    meta_path = sample_dir / "meta.json"
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    dataset = canonical_dataset_name(meta.get("dataset"))
    if dataset == "TouchAnything":
        image_name = meta.get("views", {}).get("chest", "chest.jpg")
        requested_hands = ("left", "right") if hand == "auto" else (hand,)
        hands = []
        for hand_side in requested_hands:
            hand_meta = meta.get("hands", {}).get(hand_side, {})
            if valid_bbox(hand_meta.get("bbox_chest")) and hand_meta.get(
                "gaussian_pressure"
            ) is not None:
                hands.append(hand_side)
        identity = (
            dataset,
            str(meta.get("split", split)),
            str(meta.get("scene", "")),
            str(meta.get("task", "")),
            str(meta.get("clip", meta.get("rel_clip", ""))),
        )
    else:
        is_right = int(meta.get("is_right", 1))
        metadata_hand = "right" if is_right else "left"
        if hand != "auto" and hand != metadata_hand:
            return None
        bbox = meta.get("bbox")
        side = metadata_hand
        pressure = meta.get("original_hdf5_data", {}).get(
            f"{side}_pressure_continuous_subdiv"
        )
        if pressure is None:
            pressure = meta.get("gaussian_pressure")
        image_name = meta.get("image", "image.jpg")
        identity = (
            dataset,
            split,
            str(meta.get("scene", "")),
            str(meta.get("demo", "")),
        )
        hands = [metadata_hand] if valid_bbox(bbox) and pressure is not None else []
    if not hands:
        return None
    if not (sample_dir / str(image_name)).is_file():
        return None
    if any(not value for value in identity[2:]):
        return None
    return {
        "dataset": dataset,
        "split": split,
        "hands": hands,
        "sequence_key": "/".join(identity),
        "representative_sample": str(sample_dir.resolve()),
        "frame_idx": int(meta.get("frame_idx", 0) or 0),
    }


def sequence_prefix_from_directory_name(name):
    """Derive the extraction-time sequence prefix without opening meta.json."""
    if "__" in name:
        return name.rsplit("__", 1)[0]
    if name.count("_") >= 2:
        return name.rsplit("_", 2)[0]
    return name


def _reservoir_add(reservoirs, key, path, rng, samples_per_sequence):
    state = reservoirs.setdefault(key, {"seen": 0, "paths": []})
    state["seen"] += 1
    paths = state["paths"]
    if len(paths) < samples_per_sequence:
        paths.append(path)
        return
    replacement = rng.randrange(state["seen"])
    if replacement < samples_per_sequence:
        paths[replacement] = path


def _flatten_reservoirs(reservoirs, rng):
    representatives = [
        path
        for state in reservoirs.values()
        for path in state["paths"]
    ]
    rng.shuffle(representatives)
    return representatives


def scan_sequence_representatives(split_dir, rng, samples_per_sequence=3):
    """Stream a frame-directory split while retaining only a tiny reservoir per sequence."""
    started = time.monotonic()
    scanned = 0
    reservoirs = {}
    print(f"[{split_dir.name}] Scanning frame directories under {split_dir}...", flush=True)
    with os.scandir(split_dir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            scanned += 1
            prefix = sequence_prefix_from_directory_name(entry.name)
            _reservoir_add(
                reservoirs,
                prefix,
                Path(entry.path),
                rng,
                samples_per_sequence,
            )
            if scanned % 25000 == 0:
                print(
                    f"[{split_dir.name}] scanned {scanned:,} frame directories, "
                    f"found {len(reservoirs):,} sequence prefixes "
                    f"({time.monotonic() - started:.1f}s)",
                    flush=True,
                )
    representatives = _flatten_reservoirs(reservoirs, rng)
    print(
        f"[{split_dir.name}] scan complete: {scanned:,} frame directories, "
        f"{len(reservoirs):,} sequences, {len(representatives):,} candidate frames "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return representatives


def _cached_sample_dir(raw_path, dataset_root, split):
    raw_text = str(raw_path or "").strip()
    if not raw_text:
        return None
    raw_path = Path(os.path.abspath(os.path.expanduser(raw_text)))
    split_root = Path(os.path.abspath(dataset_root / split))
    try:
        if raw_path.is_relative_to(split_root):
            return raw_path
    except (OSError, RuntimeError, ValueError):
        pass
    return split_root / raw_path.name


def infer_dataset_name(split_dir):
    with os.scandir(split_dir) as entries:
        for entry in entries:
            if not entry.is_dir():
                continue
            meta_path = Path(entry.path) / "meta.json"
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    return canonical_dataset_name(json.load(handle).get("dataset"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
    return None


def index_cache_sequence_representatives(
    dataset_root,
    split,
    hand,
    rng,
    index_cache_dir,
    minimum_sequences,
    samples_per_sequence=3,
):
    index_cache_dir = Path(index_cache_dir).expanduser().resolve()
    if not index_cache_dir.is_dir():
        print(f"[{split}] Index cache directory not found; falling back: {index_cache_dir}", flush=True)
        return []
    cache_paths = [
        path
        for path in index_cache_dir.glob(f"{split}_*.jsonl")
        if path.is_file() and path.stat().st_size > 0 and ".integrity." not in path.name
    ]
    cache_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    if not cache_paths:
        print(f"[{split}] No non-empty index cache found; falling back to directory scan.", flush=True)
        return []

    reservoirs = {}
    usable_rows = 0
    started = time.monotonic()
    expected_dataset = infer_dataset_name(dataset_root / split)
    if expected_dataset:
        print(f"[{split}] Index cache dataset filter: {expected_dataset}", flush=True)

    def add_cache_row(row):
        nonlocal usable_rows
        row_dataset = canonical_dataset_name(row.get("dataset"))
        if expected_dataset and row_dataset != expected_dataset:
            return
        row_hand = str(row.get("hand") or row.get("query_alias") or "")
        if hand != "auto" and row_hand and row_hand != hand:
            return
        sample_dir = _cached_sample_dir(row.get("sample_dir", ""), dataset_root, split)
        if sample_dir is None:
            return
        sequence_key = str(row.get("sequence_key") or "")
        if sequence_key.endswith("/left") or sequence_key.endswith("/right"):
            sequence_key = sequence_key.rsplit("/", 1)[0]
        if not sequence_key:
            sequence_key = sequence_prefix_from_directory_name(sample_dir.name)
        _reservoir_add(
            reservoirs,
            sequence_key,
            sample_dir,
            rng,
            samples_per_sequence,
        )
        usable_rows += 1

    target_sequence_count = max(minimum_sequences * 8, 24)
    probe_limit = max(2048, minimum_sequences * 512)
    for cache_path in cache_paths:
        print(f"[{split}] Random-probing compact index cache: {cache_path}", flush=True)
        try:
            file_size = cache_path.stat().st_size
            with cache_path.open("rb") as handle:
                for _probe_index in range(probe_limit):
                    offset = rng.randrange(max(1, file_size))
                    handle.seek(offset)
                    if offset:
                        handle.readline()
                    line = handle.readline()
                    if not line:
                        handle.seek(0)
                        line = handle.readline()
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    add_cache_row(row)
                    if len(reservoirs) >= target_sequence_count:
                        break
        except OSError as exc:
            print(f"[{split}] Warning: failed to read {cache_path}: {exc}", flush=True)
            continue
        if len(reservoirs) >= minimum_sequences:
            representatives = _flatten_reservoirs(reservoirs, rng)
            print(
                f"[{split}] Using index cache random access: {usable_rows:,} matching probes, "
                f"{len(reservoirs):,} sequences in {time.monotonic() - started:.1f}s",
                flush=True,
            )
            return representatives
    print(
        f"[{split}] Index caches had only {len(reservoirs)} matching sequences; "
        "falling back to directory scan.",
        flush=True,
    )
    return []


def random_sequences(dataset_root, split, count, hand, rng, index_cache_dir=None):
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    cached_sample_dirs = (
        index_cache_sequence_representatives(
            dataset_root,
            split,
            hand,
            rng,
            index_cache_dir,
            minimum_sequences=count,
        )
        if index_cache_dir
        else []
    )
    sample_sources = []
    if cached_sample_dirs:
        sample_sources.append(("index cache", cached_sample_dirs))
    else:
        sample_sources.append(("directory scan", scan_sequence_representatives(split_dir, rng)))

    selected = []
    seen_keys = set()
    checked = 0
    for source_name, sample_dirs in sample_sources:
        for sample_dir in sample_dirs:
            if not (sample_dir / "meta.json").is_file():
                continue
            checked += 1
            candidate = sequence_candidate(sample_dir, split, hand)
            if candidate is None or candidate["sequence_key"] in seen_keys:
                continue
            seen_keys.add(candidate["sequence_key"])
            selected.append(candidate)
            print(
                f"[{split}] selected {len(selected)}/{count}: {candidate['sequence_key']} "
                f"hands={','.join(candidate['hands'])}",
                flush=True,
            )
            if len(selected) == count:
                break
        if len(selected) == count:
            break
        if source_name == "index cache":
            print(
                f"[{split}] Cache candidates produced only {len(selected)}/{count} valid "
                "sequences; falling back to frame-directory scan.",
                flush=True,
            )
            sample_sources.append(("directory scan", scan_sequence_representatives(split_dir, rng)))
    if len(selected) != count:
        raise RuntimeError(
            f"Only found {len(selected)}/{count} distinct valid {hand} hand-selection sequences "
            f"in {split_dir} after checking {checked} frame directories"
        )
    return selected


def safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.") or "task"


def parse_csv(value):
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return values


def build_child_command(args, task, gpu, demo_script):
    command = [
        sys.executable,
        "-u",
        str(demo_script),
        "--checkpoint",
        str(args.checkpoint),
        "--dataset_sequence",
        task["representative_sample"],
        "--out_dir",
        task["output_root"],
        "--gpu",
        str(gpu),
        "--hand",
        task["hand"],
        "--dataset_fps",
        str(args.dataset_fps),
        "--dataset_stride",
        str(args.dataset_stride),
        "--tactile_render_size",
        args.tactile_render_size,
        "--display_floor",
        str(args.display_floor),
        "--temporal_alpha",
        str(args.temporal_alpha),
        "--render_platform",
        args.render_platform,
    ]
    if args.dataset_max_frames is not None:
        command.extend(["--dataset_max_frames", str(args.dataset_max_frames)])
    if args.dino_weights:
        command.extend(["--dino_weights", str(args.dino_weights)])
    if args.bbox_rescale_factor is not None:
        command.extend(["--bbox_rescale_factor", str(args.bbox_rescale_factor)])
    return command


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def tail_log(path, line_count=20):
    try:
        return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-line_count:])
    except OSError:
        return ""


def terminate_children(running):
    for state in running.values():
        process = state["process"]
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(
        state["process"].poll() is None for state in running.values()
    ):
        time.sleep(0.2)
    for state in running.values():
        process = state["process"]
        if process.poll() is None:
            process.kill()


def run_tasks(args, tasks, GPUs, run_dir, demo_script):
    pending = deque(tasks)
    available_gpus = deque(GPUs)
    running = {}
    completed = []
    total = len(tasks)
    last_heartbeat = time.monotonic()
    try:
        while pending or running:
            while pending and available_gpus:
                task = pending.popleft()
                gpu = available_gpus.popleft()
                task_index = int(task["task_index"])
                log_name = f"{task_index:03d}_{safe_component(task['sequence_key'])}.log"
                log_path = run_dir / "logs" / log_name
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("w", encoding="utf-8")
                command = build_child_command(args, task, gpu, demo_script)
                environment = os.environ.copy()
                environment["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    command,
                    cwd=str(demo_script.parent.parent),
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                running[task_index] = {
                    "process": process,
                    "gpu": gpu,
                    "task": task,
                    "command": command,
                    "log_path": log_path,
                    "log_handle": log_handle,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "started_monotonic": time.monotonic(),
                }
                print(
                    f"[START {task_index + 1}/{total}] GPU {gpu}: "
                    f"{task['sequence_key']} (log: {log_path})",
                    flush=True,
                )

            finished = []
            for task_index, state in running.items():
                return_code = state["process"].poll()
                if return_code is None:
                    continue
                state["log_handle"].close()
                available_gpus.append(state["gpu"])
                result = {
                    **state["task"],
                    "gpu": state["gpu"],
                    "return_code": int(return_code),
                    "log_path": str(state["log_path"]),
                    "started_at": state["started_at"],
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                }
                completed.append(result)
                status = "DONE" if return_code == 0 else "FAIL"
                print(
                    f"[{status} {len(completed)}/{total}] GPU {state['gpu']}: "
                    f"{state['task']['sequence_key']}",
                    flush=True,
                )
                if return_code != 0:
                    print(tail_log(state["log_path"]), file=sys.stderr, flush=True)
                finished.append(task_index)
            for task_index in finished:
                del running[task_index]
            now = time.monotonic()
            if running and now - last_heartbeat >= 30.0:
                states = []
                for state in running.values():
                    elapsed = int(now - state["started_monotonic"])
                    states.append(
                        f"GPU {state['gpu']} {state['task']['sequence_key']} {elapsed}s"
                    )
                print("[RUNNING] " + " | ".join(states), flush=True)
                last_heartbeat = now
            if running and not finished:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: terminating active dataset demo workers...", flush=True)
        terminate_children(running)
        raise
    finally:
        for state in running.values():
            state["log_handle"].close()
    return completed


def build_parser():
    parser = argparse.ArgumentParser(
        description="Randomly select N sequences per split and render dataset demos across GPUs."
    )
    parser.add_argument(
        "--dataset_root",
        action="append",
        required=True,
        help="Extracted dataset root containing train/val/test. Repeat for multiple datasets.",
    )
    parser.add_argument("--checkpoint", required=True, help="Tactile compact checkpoint")
    parser.add_argument("--dino_weights", default=None, help="Optional local DINOv3 weights")
    parser.add_argument("--out_dir", default="./demo_output", help="Shared ignored output root")
    parser.add_argument("--splits", type=parse_csv, default=parse_csv("train,val,test"))
    parser.add_argument("--sequences_per_split", type=int, default=3)
    parser.add_argument("--gpus", type=parse_csv, default=parse_csv("0"))
    parser.add_argument(
        "--index_cache_dir",
        default=str(Path(__file__).resolve().parent / "index_cache"),
        help="Compact JSONL cache directory used before any frame-directory scan.",
    )
    parser.add_argument(
        "--no_index_cache",
        action="store_true",
        help="Ignore compact index caches and stream the frame directories directly.",
    )
    parser.add_argument(
        "--hand",
        choices=["auto", "left", "right"],
        default="auto",
        help=(
            "Dataset hand association. auto uses OpenTouch is_right and renders one "
            "aligned two-row task for each TouchAnything sequence."
        ),
    )
    parser.add_argument("--dataset_fps", type=float, default=30.0)
    parser.add_argument("--dataset_stride", type=int, default=1)
    parser.add_argument("--dataset_max_frames", type=int, default=None)
    parser.add_argument("--tactile_render_size", default="720x1280")
    parser.add_argument("--display_floor", type=float, default=0.05)
    parser.add_argument("--temporal_alpha", type=float, default=0.4)
    parser.add_argument("--bbox_rescale_factor", type=float, default=None)
    parser.add_argument(
        "--render_platform",
        choices=["software", "egl", "osmesa", "auto"],
        default="software",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Select and write the random manifest without launching model processes.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.sequences_per_split < 1:
        raise ValueError("--sequences_per_split must be positive")
    if args.dataset_stride < 1:
        raise ValueError("--dataset_stride must be positive")
    if args.dataset_max_frames is not None and args.dataset_max_frames < 1:
        raise ValueError("--dataset_max_frames must be positive")
    if not math.isfinite(args.dataset_fps) or args.dataset_fps <= 0:
        raise ValueError("--dataset_fps must be finite and positive")

    roots = [Path(value).expanduser().resolve() for value in args.dataset_root]
    missing_roots = [str(root) for root in roots if not root.is_dir()]
    if missing_roots:
        raise FileNotFoundError("Dataset root(s) not found: " + ", ".join(missing_roots))
    args.checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.dino_weights:
        args.dino_weights = Path(args.dino_weights).expanduser().resolve()
        if not args.dino_weights.is_file():
            raise FileNotFoundError(f"DINO weights not found: {args.dino_weights}")
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError(f"--gpus contains duplicates: {args.gpus}")
    args.index_cache_dir = (
        None
        if args.no_index_cache
        else Path(args.index_cache_dir).expanduser().resolve()
    )

    # Seed once from OS entropy, then shuffle locally. SystemRandom.shuffle on
    # a 100k+ frame directory makes one entropy call per swap and is needlessly
    # slow; this remains non-deterministic without exposing a fixed seed.
    rng = random.Random(secrets.randbits(256))
    tasks = []
    for root_index, root in enumerate(roots):
        for split in args.splits:
            selected = random_sequences(
                root,
                split,
                args.sequences_per_split,
                args.hand,
                rng,
                index_cache_dir=args.index_cache_dir,
            )
            for sequence in selected:
                hands = sequence.pop("hands")
                task_hands = (
                    [("auto", "both")]
                    if sequence["dataset"] == "TouchAnything" and args.hand == "auto"
                    else [(hand_side, hand_side) for hand_side in hands]
                )
                for hand_side, key_suffix in task_hands:
                    task = {
                        **sequence,
                        "hand": hand_side,
                        "sequence_key": f"{sequence['sequence_key']}/{key_suffix}",
                        "dataset_root": str(root),
                        "dataset_root_index": root_index,
                        "task_index": len(tasks),
                    }
                    tasks.append(task)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / "dataset_batch_runs" / timestamp
    for task in tasks:
        hand_label = "both" if task["hand"] == "auto" else task["hand"]
        task_dir_name = "_".join(
            (
                f"{int(task['task_index']):03d}",
                safe_component(task["dataset"]),
                safe_component(task["split"]),
                safe_component(hand_label),
            )
        )
        task["output_root"] = str(
            run_dir / "outputs" / task_dir_name
        )
    selection_path = run_dir / "selection.json"
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_source": "random.Random seeded once from OS entropy; no fixed seed",
        "dataset_roots": [str(root) for root in roots],
        "splits": args.splits,
        "sequences_per_split_per_root": args.sequences_per_split,
        "hand_selection": args.hand,
        "task_expansion": (
            "TouchAnything auto mode uses one task with aligned left/right rows; "
            "OpenTouch uses its metadata is_right hand"
        ),
        "gpus": args.gpus,
        "index_cache_dir": str(args.index_cache_dir) if args.index_cache_dir else None,
        "tasks": tasks,
    }
    write_json(selection_path, selection)
    print(f"Random selection saved to: {selection_path}")
    for task in tasks:
        print(f"  [{task['task_index']:02d}] {task['sequence_key']}")
    if args.dry_run:
        print("Dry run complete; no model process was launched.")
        return 0

    demo_script = Path(__file__).resolve().parent / "demo_tactile_video.py"
    try:
        completed = run_tasks(args, tasks, args.gpus, run_dir, demo_script)
    except KeyboardInterrupt:
        return 130
    summary = {
        "selection_manifest": str(selection_path),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "total": len(tasks),
        "succeeded": sum(item["return_code"] == 0 for item in completed),
        "failed": sum(item["return_code"] != 0 for item in completed),
        "results": sorted(completed, key=lambda item: item["task_index"]),
    }
    write_json(run_dir / "summary.json", summary)
    print(
        f"Batch complete: {summary['succeeded']}/{summary['total']} succeeded; "
        f"summary={run_dir / 'summary.json'}"
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
