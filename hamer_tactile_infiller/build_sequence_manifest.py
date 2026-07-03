#!/usr/bin/env python
import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from .common import (
        canonical_dataset_name,
        resolve_data_dirs,
        split_roots,
        valid_bbox,
        write_jsonl,
    )
except ImportError:
    from common import (
        canonical_dataset_name,
        resolve_data_dirs,
        split_roots,
        valid_bbox,
        write_jsonl,
    )


def _get_path(data, dotted_key):
    cur = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _touchanything_pressure(meta, hand):
    hand_meta = meta.get("hands", {}).get(hand, {})
    if hand_meta.get("gaussian_pressure") is not None:
        return "hands.%s.gaussian_pressure" % hand
    return None


def _single_hand_pressure(meta, is_right):
    side = "right" if int(is_right) == 1 else "left"
    key = f"original_hdf5_data.{side}_pressure_continuous_subdiv"
    if _get_path(meta, key) is not None:
        return key
    if meta.get("gaussian_pressure") is not None:
        return "gaussian_pressure"
    return None


def _sequence_key_for_opentouch(meta, split, is_right):
    scene = meta.get("scene", "unknown_scene")
    clip = meta.get("demo", meta.get("clip", meta.get("clip_id", "unknown_clip")))
    return f"OpenTouch/{split}/{scene}/{clip}/{int(is_right)}"


def _sequence_key_for_touchanything(meta, split, hand):
    scene = meta.get("scene", "unknown_scene")
    task = meta.get("task", "unknown_task")
    clip = meta.get("clip", meta.get("clip_id", "unknown_clip"))
    return f"TouchAnything/{split}/{scene}/{task}/{clip}/{hand}"


def _sequence_key_for_egotactile(meta, split, hand):
    rel_seq = meta.get("rel_seq", meta.get("seq_dir", "unknown_seq"))
    return f"EgoTactile/{split}/{rel_seq}/{hand}"


def _sample_record(sample_dir, image_name, bbox, bbox_valid, tactile_key, tactile_valid, frame_idx):
    return {
        "frame_idx": int(frame_idx),
        "sample_dir": str(sample_dir),
        "image": image_name,
        "bbox": bbox,
        "bbox_valid": bool(bbox_valid),
        "tactile_key": tactile_key,
        "tactile_valid": bool(tactile_valid),
    }


def iter_sample_records(root, split_name, sample_dir):
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return

    dataset_name = canonical_dataset_name(meta.get("dataset", "OpenTouch"))
    meta_split = str(meta.get("split", split_name))
    split = meta_split or split_name
    frame_idx = int(meta.get("frame_idx", 0))

    if dataset_name == "TouchAnything":
        image_name = meta.get("views", {}).get("chest", "chest.jpg")
        for hand in ("left", "right"):
            hand_meta = meta.get("hands", {}).get(hand, {})
            bbox = hand_meta.get("bbox_chest")
            tactile_key = _touchanything_pressure(meta, hand)
            is_right = int(hand_meta.get("is_right", 1 if hand == "right" else 0))
            yield (
                _sequence_key_for_touchanything(meta, split, hand),
                {
                    "dataset": "TouchAnything",
                    "split": split,
                    "hand": hand,
                    "is_right": is_right,
                },
                _sample_record(
                    sample_dir,
                    image_name,
                    bbox,
                    valid_bbox(bbox),
                    tactile_key,
                    tactile_key is not None,
                    frame_idx,
                ),
            )
        return

    hand = str(meta.get("hand", "right" if int(meta.get("is_right", 1)) else "left"))
    is_right = int(meta.get("is_right", 1 if hand == "right" else 0))
    bbox = meta.get("bbox")
    tactile_key = _single_hand_pressure(meta, is_right)
    image_name = meta.get("image", "image.jpg")

    if dataset_name == "EgoTactile":
        seq_key = _sequence_key_for_egotactile(meta, split, hand)
    else:
        dataset_name = "OpenTouch"
        seq_key = _sequence_key_for_opentouch(meta, split, is_right)

    yield (
        seq_key,
        {
            "dataset": dataset_name,
            "split": split,
            "hand": hand,
            "is_right": is_right,
        },
        _sample_record(
            sample_dir,
            image_name,
            bbox,
            valid_bbox(bbox),
            tactile_key,
            tactile_key is not None,
            frame_idx,
        ),
    )


def _scan_sample_task(args):
    root, split, sample_dir = args
    records = []
    for item in iter_sample_records(Path(root), split, Path(sample_dir)) or []:
        records.append(item)
    return records


def _scan_sample_batch(batch):
    records = []
    for task in batch:
        records.extend(_scan_sample_task(task))
    return records


def _registry_paths_for_root(root):
    root = Path(root)
    return [
        root / "egotactile_frames_registry.json",
        root / "touchanything_frames_registry.json",
        root / "dataset_frames_registry.json",
    ]


def _split_from_sample_dir(root, sample_dir):
    try:
        rel = Path(sample_dir).relative_to(root)
    except ValueError:
        return "train"
    if not rel.parts:
        return "train"
    first = rel.parts[0]
    if first in {"train", "val", "test", "test_seen", "test_unseen", "bare_hand", "gloved_hand"}:
        return first
    return "train"


def _sample_tasks_from_registry(root):
    root = Path(root)
    for registry_path in _registry_paths_for_root(root):
        if not registry_path.exists():
            continue
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  Warning: failed to read registry {registry_path}: {exc}")
            continue
        tasks = []
        for item in registry:
            sample_dir = item.get("sample_dir") or item.get("path")
            if not sample_dir:
                continue
            sample_dir = Path(sample_dir)
            if not sample_dir.is_dir():
                continue
            split = str(item.get("split") or _split_from_sample_dir(root, sample_dir))
            tasks.append((str(root), split, str(sample_dir)))
        if tasks:
            return tasks, registry_path
    return [], None


def _sample_tasks_from_walk(root):
    tasks = []
    root = Path(root)
    for meta_path in sorted(root.rglob("meta.json")):
        sample_dir = meta_path.parent
        split = _split_from_sample_dir(root, sample_dir)
        tasks.append((str(root), split, str(sample_dir)))
    return tasks


def _chunked(items, batch_size):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_tasks_for_root(root_path):
    root_tasks, registry_path = _sample_tasks_from_registry(root_path)
    if not root_tasks:
        root_tasks = _sample_tasks_from_walk(root_path)
    return root_tasks, registry_path


def build_extracted_sequences(data_roots, workers=1):
    grouped = {}
    scan_counts = defaultdict(int)
    registry_sources = []
    missing_roots = []
    total_tasks = 0
    root_tasks_by_path = []
    for root in data_roots:
        root_path = Path(root)
        if not root_path.exists():
            missing_roots.append(str(root))
            continue
        root_tasks, registry_path = _load_tasks_for_root(root_path)
        if registry_path is not None:
            registry_sources.append((str(root_path), str(registry_path), len(root_tasks)))
        root_tasks_by_path.append((root_path, root_tasks))
        total_tasks += len(root_tasks)
        for _, split, _ in root_tasks:
            scan_counts[(str(root_path), split)] += 1

    print("Manifest scan plan:")
    if missing_roots:
        print("  Missing roots:")
        for root in missing_roots:
            print(f"    - {root}")
    if registry_sources:
        print("  Registry sources:")
        for root, registry_path, count in registry_sources:
            print(f"    - {count:8d} sample dirs from {registry_path} ({root})")
    for (root, split), count in sorted(scan_counts.items()):
        print(f"  {split:11s} {count:8d} sample dirs from {root}")
    print(f"  total       {total_tasks:8d} sample dirs")

    def add_record(seq_key, seq_meta, frame):
        if seq_key not in grouped:
            grouped[seq_key] = {
                "dataset": seq_meta["dataset"],
                "split": seq_meta["split"],
                "sequence_id": seq_key,
                "hand": seq_meta["hand"],
                "is_right": seq_meta["is_right"],
                "frames": [],
            }
        grouped[seq_key]["frames"].append(frame)

    workers = max(1, int(workers))
    batch_size = max(1, min(1024, workers * 64))
    if workers == 1:
        for _, root_tasks in root_tasks_by_path:
            for batch in _chunked(root_tasks, batch_size):
                for seq_key, seq_meta, frame in _scan_sample_batch(batch):
                    add_record(seq_key, seq_meta, frame)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batch_count = 0
            for _, root_tasks in root_tasks_by_path:
                for records in executor.map(_scan_sample_batch, _chunked(root_tasks, batch_size), chunksize=1):
                    for seq_key, seq_meta, frame in records:
                        add_record(seq_key, seq_meta, frame)
                    batch_count += 1
                    if batch_count % 100 == 0:
                        print(f"Scanned {batch_count} batches...")

    rows = []
    for row in grouped.values():
        row["frames"].sort(key=lambda x: x["frame_idx"])
        rows.append(row)
    rows.sort(key=lambda x: x["sequence_id"])
    summary = defaultdict(lambda: {"sequences": 0, "frames": 0, "bbox_valid": 0, "tactile_valid": 0})
    for row in rows:
        key = (row["split"], row["dataset"])
        summary[key]["sequences"] += 1
        summary[key]["frames"] += len(row["frames"])
        summary[key]["bbox_valid"] += sum(1 for frame in row["frames"] if frame.get("bbox_valid"))
        summary[key]["tactile_valid"] += sum(1 for frame in row["frames"] if frame.get("tactile_valid"))
    print("Manifest grouping summary:")
    for (split, dataset), stats in sorted(summary.items()):
        print(
            f"  {split:11s} {dataset:14s} "
            f"sequences={stats['sequences']:6d} "
            f"frames={stats['frames']:8d} "
            f"bbox_valid={stats['bbox_valid']:8d} "
            f"tactile_valid={stats['tactile_valid']:8d}"
        )
    return rows


def parse_motion_name(path):
    name = Path(path).name
    match = re.match(r"(?P<subject>p\d+)-(?P<object>.+)-repeat(?P<repeat>\d+)$", name, flags=re.IGNORECASE)
    if not match:
        return {"subject": None, "object": None, "repeat": None}
    return {
        "subject": match.group("subject").lower(),
        "object": match.group("object"),
        "repeat": int(match.group("repeat")),
    }


def split_for_motion(path, mode):
    if mode == "motion_list":
        return "train"
    parts = parse_motion_name(path)
    repeat = parts.get("repeat")
    subject = parts.get("subject")
    if repeat is not None and repeat % 10 == 9:
        return "test"
    if subject in {"p012"}:
        return "val"
    return "train"


def build_motion_list_sequences(motion_list, mode):
    rows = []
    with Path(motion_list).open("r", encoding="utf-8") as f:
        for line in f:
            seq_path = line.strip()
            if not seq_path:
                continue
            parts = parse_motion_name(seq_path)
            split = split_for_motion(seq_path, mode)
            for hand in ("left", "right"):
                rows.append(
                    {
                        "dataset": "EgoTactile",
                        "split": split,
                        "sequence_id": f"EgoTactile/{split}/{Path(seq_path).as_posix()}/{hand}",
                        "hand": hand,
                        "is_right": 1 if hand == "right" else 0,
                        "motion_dir": seq_path,
                        "subject": parts.get("subject"),
                        "object": parts.get("object"),
                        "repeat": parts.get("repeat"),
                        "frames": [],
                    }
                )
    return rows


def write_split_manifests(rows, output_dir, prefix=None):
    by_split = defaultdict(list)
    for row in rows:
        by_split[row.get("split", "train")].append(row)
    output_dir = Path(output_dir)
    prefix = str(prefix or "").strip()
    for split, split_rows in sorted(by_split.items()):
        if prefix:
            path = output_dir / f"sequence_manifest_{prefix}_{split}.jsonl"
        else:
            path = output_dir / f"sequence_manifest_{split}.jsonl"
        write_jsonl(path, split_rows)
        print(f"Wrote {len(split_rows)} sequences: {path}")
    return by_split


def parse_args():
    parser = argparse.ArgumentParser(description="Build hand-level sequence manifests for tactile infiller training.")
    parser.add_argument("--datasets", default=None, help="Dataset names/aliases, comma-separated.")
    parser.add_argument("--data_dir", default=None, help="Explicit extracted roots, comma-separated.")
    parser.add_argument("--output_dir", default="hamer_tactile_infiller/manifests")
    parser.add_argument(
        "--manifest_prefix",
        default=None,
        help=(
            "Optional filename prefix to avoid overwriting per-dataset manifests. "
            "Example: --manifest_prefix touchanything writes sequence_manifest_touchanything_train.jsonl."
        ),
    )
    parser.add_argument("--manifest_workers", type=int, default=1, help="Processes for scanning meta.json files.")
    parser.add_argument(
        "--egotactile_split_source",
        default="extracted",
        choices=["extracted", "motion_list", "derived"],
        help="How to handle EgoTactile/EgoPressureDiff splits.",
    )
    parser.add_argument(
        "--egopressure_motion_list",
        default="EgoPressureDiff/V2P_data/motion_list.txt",
        help="Motion list used when --egotactile_split_source is motion_list or derived.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    if args.egotactile_split_source in {"motion_list", "derived"}:
        rows.extend(build_motion_list_sequences(args.egopressure_motion_list, args.egotactile_split_source))
    else:
        roots = resolve_data_dirs(args.datasets, args.data_dir)
        print("Scanning extracted roots:")
        for root in roots:
            print(f"  - {root}")
        rows.extend(build_extracted_sequences(roots, workers=args.manifest_workers))

    if not rows:
        raise RuntimeError("No sequences were found.")
    write_split_manifests(rows, args.output_dir, prefix=args.manifest_prefix)


if __name__ == "__main__":
    main()
