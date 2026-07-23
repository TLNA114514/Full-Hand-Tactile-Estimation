#!/usr/bin/env python3
"""Select and materialize deterministic SAM3 bbox pilots by dataset/split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .defaults import (
        DEFAULT_OPENTOUCH_DATA_ROOT,
        DEFAULT_OPENTOUCH_SPLITS,
        DEFAULT_TOUCHANYTHING_ROOT,
        DEFAULT_TOUCHANYTHING_SPLIT_JSON,
    )
    from .progress import progress
except ImportError:
    from defaults import (
        DEFAULT_OPENTOUCH_DATA_ROOT,
        DEFAULT_OPENTOUCH_SPLITS,
        DEFAULT_TOUCHANYTHING_ROOT,
        DEFAULT_TOUCHANYTHING_SPLIT_JSON,
    )
    from progress import progress


OPEN_TOUCH_SPLITS = ("train", "val", "test")
TOUCH_ANYTHING_SPLITS = ("train", "val", "test_seen", "test_unseen")
DATASET_SPLITS = {
    "opentouch": OPEN_TOUCH_SPLITS,
    "touchanything": TOUCH_ANYTHING_SPLITS,
}


@dataclass(frozen=True)
class PilotRecord:
    job_id: str
    dataset: str
    split: str
    sequence_key: str
    resource_path: str
    resource_type: str
    source_path: str
    expected_gloved_hands: int
    prompt_preset: str = "gloved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pick three raw sequences per OpenTouch/TouchAnything split."
    )
    parser.add_argument("--datasets", default="opentouch,touchanything")
    parser.add_argument(
        "--splits",
        default="auto",
        help="Comma-separated splits for the selected dataset(s), or auto for all canonical splits.",
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
    parser.add_argument("--output-dir", type=Path, required=True)
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
        help="Select every available sequence in each requested split.",
    )
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit materialized OpenTouch frames; 0 keeps the complete sequence.",
    )
    return parser.parse_args()


def parse_dataset_selection(value: str | Iterable[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    selected: list[str] = []
    aliases = {
        "ot": "opentouch",
        "opentouch": "opentouch",
        "ta": "touchanything",
        "touchanything": "touchanything",
        "egotouch": "touchanything",
    }
    for item in raw:
        token = str(item).strip().lower()
        if not token:
            continue
        if token not in aliases:
            raise ValueError(f"Unsupported dataset {item!r}; choose opentouch or touchanything")
        canonical = aliases[token]
        if canonical not in selected:
            selected.append(canonical)
    if not selected:
        raise ValueError("At least one dataset must be selected")
    return tuple(selected)


def resolve_split_selection(
    datasets: Iterable[str],
    value: str | Iterable[str],
) -> dict[str, tuple[str, ...]]:
    datasets = tuple(datasets)
    raw = value.split(",") if isinstance(value, str) else list(value)
    requested = tuple(str(item).strip() for item in raw if str(item).strip())
    if not requested or requested == ("auto",):
        return {dataset: DATASET_SPLITS[dataset] for dataset in datasets}
    if "auto" in requested:
        raise ValueError("--splits cannot combine auto with explicit split names")
    result: dict[str, tuple[str, ...]] = {}
    recognized: set[str] = set()
    for dataset in datasets:
        selected = tuple(split for split in requested if split in DATASET_SPLITS[dataset])
        if selected:
            result[dataset] = selected
            recognized.update(selected)
    unknown = sorted(set(requested) - recognized)
    if unknown:
        raise ValueError(
            f"Splits {unknown} are not valid for the selected datasets {list(datasets)}"
        )
    missing = [dataset for dataset in datasets if dataset not in result]
    if missing:
        raise ValueError(
            f"Explicit --splits selected no split for datasets {missing}; run the domains "
            "separately when their split names differ"
        )
    return result


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sequence"


def stable_order(records: Iterable[PilotRecord], seed: int) -> list[PilotRecord]:
    def key(record: PilotRecord) -> tuple[str, str]:
        token = f"{seed}\0{record.dataset}\0{record.split}\0{record.sequence_key}"
        return hashlib.sha256(token.encode("utf-8")).hexdigest(), record.sequence_key

    return sorted(records, key=key)


def hdf5_path_for_scene(root: Path, scene: str) -> Path | None:
    for suffix in (".hdf5", ".h5"):
        candidate = root / f"{scene}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _record_value(record: PilotRecord | Mapping[str, Any], key: str) -> Any:
    return getattr(record, key) if isinstance(record, PilotRecord) else record[key]


def materialize_opentouch_record(
    record: PilotRecord | Mapping[str, Any],
    max_frames: int,
    *,
    min_free_space_gb: float = 1.0,
) -> Path:
    """Materialize one HDF5 JPEG sequence for one active tracking worker."""

    if _record_value(record, "dataset") != "opentouch":
        raise ValueError("Only OpenTouch records require JPEG materialization")
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to materialize OpenTouch frames") from exc

    source_path = str(_record_value(record, "source_path"))
    hdf5_file_text, dataset_key = source_path.split("::", 1)
    frames_dir = Path(str(_record_value(record, "resource_path")))
    done_path = frames_dir / ".materialized.json"
    requested = {"source": source_path, "max_frames": int(max_frames)}
    if done_path.is_file():
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if all(done.get(key) == value for key, value in requested.items()):
                return frames_dir
        except (OSError, json.JSONDecodeError):
            pass

    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.jpg"):
        stale.unlink()
    done_path.unlink(missing_ok=True)
    min_free_bytes = max(0, int(float(min_free_space_gb) * 1024**3))
    try:
        with h5py.File(hdf5_file_text, "r") as handle:
            if dataset_key not in handle:
                raise KeyError(f"Missing {dataset_key} in {hdf5_file_text}")
            images = handle[dataset_key]
            total = len(images) if max_frames <= 0 else min(len(images), max_frames)
            sample_indices = {0, max(0, total // 2), max(0, total - 1)}
            sample_checksums = []
            materialized_bytes = 0
            for frame_index in progress(
                range(total),
                desc=f"materialize {_record_value(record, 'sequence_key')}",
                unit="frame",
                leave=False,
            ):
                if frame_index % 64 == 0 and min_free_bytes:
                    free_bytes = shutil.disk_usage(frames_dir).free
                    if free_bytes < min_free_bytes:
                        raise OSError(
                            "OpenTouch materialization stopped before filling the filesystem: "
                            f"{free_bytes / 1024**3:.2f} GiB free, "
                            f"{min_free_space_gb:.2f} GiB required"
                        )
                encoded = images[frame_index]
                if hasattr(encoded, "tobytes"):
                    encoded = encoded.tobytes()
                else:
                    encoded = bytes(encoded)
                frame_path = frames_dir / f"{frame_index:08d}.jpg"
                frame_path.write_bytes(encoded)
                materialized_bytes += len(encoded)
                if frame_index in sample_indices:
                    source_sha = hashlib.sha256(encoded).hexdigest()
                    materialized_sha = hashlib.sha256(frame_path.read_bytes()).hexdigest()
                    sample_checksums.append(
                        {
                            "frame_index": frame_index,
                            "source_jpeg_sha256": source_sha,
                            "materialized_jpeg_sha256": materialized_sha,
                            "byte_identical": source_sha == materialized_sha,
                        }
                    )
        materialization_audit = {
            **requested,
            "frame_count": total,
            "materialized_bytes": materialized_bytes,
            "jpeg_passthrough": True,
            "sample_checksums": sample_checksums,
        }
        done_path.write_text(
            json.dumps(materialization_audit, indent=2), encoding="utf-8"
        )
        return frames_dir
    except Exception:
        cleanup_opentouch_materialization(record)
        raise


def cleanup_opentouch_materialization(
    record: PilotRecord | Mapping[str, Any],
) -> int:
    """Remove only the disposable JPEG directory declared by an OpenTouch record."""

    if _record_value(record, "dataset") != "opentouch":
        return 0
    if _record_value(record, "resource_type") != "jpeg_directory":
        raise ValueError("Refusing to clean a non-JPEG OpenTouch resource")
    source_path = str(_record_value(record, "source_path"))
    if "::" not in source_path:
        raise ValueError("Refusing to clean an OpenTouch resource without an HDF5 source")
    frames_dir = Path(str(_record_value(record, "resource_path")))
    if not frames_dir.exists():
        return 0
    removed_bytes = 0
    done_path = frames_dir / ".materialized.json"
    if done_path.is_file():
        try:
            removed_bytes = int(
                json.loads(done_path.read_text(encoding="utf-8")).get(
                    "materialized_bytes", 0
                )
            )
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    shutil.rmtree(frames_dir)
    return removed_bytes


def select_opentouch(
    data_root: Path,
    split_path: Path,
    output_dir: Path,
    count: int | None,
    seed: int,
    max_frames: int,
    materialize: bool = True,
    splits: Iterable[str] = OPEN_TOUCH_SPLITS,
    count_per_dataset: bool = False,
) -> list[PilotRecord]:
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    selected: list[PilotRecord] = []
    dataset_candidates: list[PilotRecord] = []
    for split in splits:
        candidates: list[PilotRecord] = []
        for scene, clip_id in split_data.get(split, []):
            source = hdf5_path_for_scene(data_root, str(scene))
            if source is None:
                continue
            sequence_key = f"{scene}/{clip_id}"
            frames_dir = output_dir / "inputs" / "opentouch" / split / safe_component(sequence_key)
            candidates.append(
                PilotRecord(
                    job_id=f"opentouch__{split}__{safe_component(sequence_key)}",
                    dataset="opentouch",
                    split=split,
                    sequence_key=sequence_key,
                    resource_path=str(frames_dir.resolve()),
                    resource_type="jpeg_directory",
                    source_path=f"{source.resolve()}::data/{clip_id}/rgb_images_jpeg",
                    expected_gloved_hands=1,
                )
            )
        if count_per_dataset:
            dataset_candidates.extend(candidates)
            continue
        ordered = stable_order(candidates, seed)
        chosen = ordered if count is None else ordered[:count]
        if count is not None and len(chosen) != count:
            raise RuntimeError(
                f"OpenTouch {split} has only {len(chosen)} available sequences; expected {count}"
            )
        for record in progress(
            chosen,
            desc=f"OpenTouch {split}: sequences",
            unit="seq",
        ):
            if materialize:
                materialize_opentouch_record(record, max_frames)
            selected.append(record)
    if count_per_dataset:
        ordered = stable_order(dataset_candidates, seed)
        chosen = ordered if count is None else ordered[:count]
        if count is not None and len(chosen) != count:
            raise RuntimeError(
                f"OpenTouch has only {len(chosen)} available sequences across the "
                f"requested splits; expected {count}"
            )
        for record in progress(chosen, desc="OpenTouch: random dataset sample", unit="seq"):
            if materialize:
                materialize_opentouch_record(record, max_frames)
            selected.append(record)
    return selected


def extract_split_path(entry: Any) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (list, tuple)) and entry:
        return extract_split_path(entry[0])
    if isinstance(entry, dict):
        for key in (
            "hdf5_path",
            "file_path",
            "path",
            "hdf5",
            "file",
            "trajectory",
            "traj",
            "clip_path",
            "clip",
        ):
            if key in entry:
                return extract_split_path(entry[key])
    raise ValueError(f"Unsupported TouchAnything split entry: {entry!r}")


def build_touchanything_index(raw_root: Path) -> dict[tuple[str, ...], Path]:
    index: dict[tuple[str, ...], Path] = {}
    skip = {"extracted_frames", "metadata", "__pycache__"}
    scene_dirs = sorted(raw_root.iterdir())
    for scene_dir in progress(
        scene_dirs,
        desc="TouchAnything: index raw videos",
        unit="scene",
    ):
        if not scene_dir.is_dir() or scene_dir.name in skip:
            continue
        for task_dir in sorted(scene_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name in skip:
                continue
            for clip_dir in sorted(task_dir.iterdir()):
                video = clip_dir / "chest.mp4"
                if not clip_dir.is_dir() or not video.is_file():
                    continue
                scene, task, clip = scene_dir.name, task_dir.name, clip_dir.name
                index[("scene_task_clip", scene, task, clip)] = clip_dir
                index[("task_clip", task, clip)] = clip_dir
                index.setdefault(("clip", clip), clip_dir)
    return index


def resolve_touchanything_entry(entry: Any, raw_root: Path, index: dict[tuple[str, ...], Path]) -> Path:
    split_path = Path(extract_split_path(entry))
    parts = split_path.parts
    clip = split_path.stem
    candidates: list[Path | None] = []
    if len(parts) >= 3:
        scene, task = parts[-3], parts[-2]
        candidates.extend(
            [
                raw_root / scene / task / clip,
                index.get(("scene_task_clip", scene, task, clip)),
            ]
        )
    if len(parts) >= 2:
        candidates.append(index.get(("task_clip", parts[-2], clip)))
    candidates.append(index.get(("clip", clip)))
    for candidate in candidates:
        if candidate is not None and (candidate / "chest.mp4").is_file():
            return candidate
    raise FileNotFoundError(f"Could not map TouchAnything split entry to chest.mp4: {entry!r}")


def select_touchanything(
    raw_root: Path,
    split_path: Path,
    count: int | None,
    seed: int,
    splits: Iterable[str] = TOUCH_ANYTHING_SPLITS,
    count_per_dataset: bool = False,
) -> list[PilotRecord]:
    split_data = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split_data, dict):
        raise ValueError("TouchAnything split JSON must be a split-name dictionary")
    raw_index = build_touchanything_index(raw_root)
    selected: list[PilotRecord] = []
    dataset_candidates: list[PilotRecord] = []
    missing: list[str] = []
    for split in splits:
        entries = split_data.get(split, [])
        if isinstance(entries, dict):
            entries = list(entries.values())
        candidates: list[PilotRecord] = []
        for entry in progress(
            entries,
            desc=f"TouchAnything {split}: map sequences",
            unit="seq",
        ):
            try:
                clip_dir = resolve_touchanything_entry(entry, raw_root, raw_index)
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            rel = clip_dir.relative_to(raw_root)
            sequence_key = "/".join(rel.parts[-3:])
            video = clip_dir / "chest.mp4"
            candidates.append(
                PilotRecord(
                    job_id=f"touchanything__{split}__{safe_component(sequence_key)}",
                    dataset="touchanything",
                    split=split,
                    sequence_key=sequence_key,
                    resource_path=str(video.resolve()),
                    resource_type="video",
                    source_path=str(video.resolve()),
                    expected_gloved_hands=2,
                )
            )
        if count_per_dataset:
            dataset_candidates.extend(candidates)
            continue
        ordered = stable_order(candidates, seed)
        chosen = ordered if count is None else ordered[:count]
        if count is not None and len(chosen) != count:
            example = f" Example mapping failure: {missing[0]}" if missing else ""
            raise RuntimeError(
                f"TouchAnything {split} has only {len(chosen)} available sequences; "
                f"expected {count}.{example}"
            )
        selected.extend(chosen)
    if count_per_dataset:
        ordered = stable_order(dataset_candidates, seed)
        chosen = ordered if count is None else ordered[:count]
        if count is not None and len(chosen) != count:
            example = f" Example mapping failure: {missing[0]}" if missing else ""
            raise RuntimeError(
                f"TouchAnything has only {len(chosen)} available sequences across the "
                f"requested splits; expected {count}.{example}"
            )
        selected.extend(chosen)
    return selected


def build_manifest(
    *,
    opentouch_data_root: Path,
    opentouch_splits: Path,
    touchanything_root: Path,
    touchanything_split_json: Path,
    output_dir: Path,
    samples_per_split: int,
    samples_per_dataset: int = 0,
    all_sequences: bool = False,
    seed: int,
    max_frames: int,
    materialize_opentouch: bool = True,
    datasets: str | Iterable[str] = ("opentouch", "touchanything"),
    splits: str | Iterable[str] = "auto",
) -> Path:
    if samples_per_dataset < 0:
        raise ValueError("samples_per_dataset must be >= 0")
    if samples_per_dataset and all_sequences:
        raise ValueError("--samples-per-dataset and --all-sequences are mutually exclusive")
    if samples_per_split < 1 and not all_sequences and not samples_per_dataset:
        raise ValueError("samples_per_split must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_datasets = parse_dataset_selection(datasets)
    selected_splits = resolve_split_selection(selected_datasets, splits)
    records: list[PilotRecord] = []
    count_per_dataset = samples_per_dataset > 0
    selection_count = (
        None if all_sequences else samples_per_dataset if count_per_dataset else samples_per_split
    )
    if "opentouch" in selected_splits:
        records.extend(
            select_opentouch(
                opentouch_data_root.expanduser().resolve(),
                opentouch_splits.expanduser().resolve(),
                output_dir.resolve(),
                selection_count,
                seed,
                max_frames,
                materialize_opentouch,
                selected_splits["opentouch"],
                count_per_dataset=count_per_dataset,
            )
        )
    if "touchanything" in selected_splits:
        records.extend(
            select_touchanything(
                touchanything_root.expanduser().resolve(),
                touchanything_split_json.expanduser().resolve(),
                selection_count,
                seed,
                selected_splits["touchanything"],
                count_per_dataset=count_per_dataset,
            )
        )
    records.sort(key=lambda item: (item.dataset, item.split, item.sequence_key))
    if not all_sequences:
        expected = (
            samples_per_dataset * len(selected_splits)
            if count_per_dataset
            else samples_per_split * sum(len(value) for value in selected_splits.values())
        )
        if len(records) != expected:
            raise RuntimeError(f"Pilot manifest has {len(records)} rows, expected {expected}")

    manifest_path = output_dir / "pilot_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(asdict(record), ensure_ascii=True, separators=(",", ":")) + "\n"
            )
    summary = {
        "seed": seed,
        "samples_per_split": samples_per_split,
        "samples_per_dataset": samples_per_dataset,
        "all_sequences": all_sequences,
        "record_count": len(records),
        "max_frames": max_frames,
        "opentouch_materialization": "eager" if materialize_opentouch else "lazy",
        "datasets": list(selected_datasets),
        "splits": {key: list(value) for key, value in selected_splits.items()},
        "sources": {
            "opentouch_data_root": (
                str(opentouch_data_root.expanduser().resolve())
                if "opentouch" in selected_splits
                else None
            ),
            "opentouch_split_file": (
                str(opentouch_splits.expanduser().resolve())
                if "opentouch" in selected_splits
                else None
            ),
            "touchanything_root": (
                str(touchanything_root.expanduser().resolve())
                if "touchanything" in selected_splits
                else None
            ),
            "touchanything_split_file": (
                str(touchanything_split_json.expanduser().resolve())
                if "touchanything" in selected_splits
                else None
            ),
        },
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (output_dir / "pilot_manifest.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Pilot manifest: {manifest_path}")
    split_count = sum(len(value) for value in selected_splits.values())
    print(f"Selected {len(records)} sequences across {split_count} dataset split(s).")
    return manifest_path


def main() -> int:
    args = parse_args()
    build_manifest(
        opentouch_data_root=args.opentouch_data_root,
        opentouch_splits=args.opentouch_splits,
        touchanything_root=args.touchanything_root,
        touchanything_split_json=args.touchanything_split_json,
        output_dir=args.output_dir.expanduser().resolve(),
        samples_per_split=args.samples_per_split,
        samples_per_dataset=args.samples_per_dataset,
        all_sequences=args.all_sequences,
        seed=args.seed,
        max_frames=args.max_frames,
        datasets=args.datasets,
        splits=args.splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
