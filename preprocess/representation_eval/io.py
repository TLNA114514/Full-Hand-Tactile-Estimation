from __future__ import annotations

import json
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


DATASET_NAMES = {"opentouch", "egotactile", "touchanything"}
DEFAULT_SCAN_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "outputs",
    "metadata",
    "artifacts",
    "touchanything_bboxes_cache",
    "full_bboxes_cache",
    "extracted_frames",
}


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expand_path_list(value: str | Sequence[str] | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [item.strip() for item in str(value).split(",") if item.strip()]
    paths: list[Path] = []
    for item in raw_items:
        if any(ch in item for ch in "*?[]"):
            paths.extend(Path(match) for match in sorted(glob.glob(item)))
        else:
            paths.append(Path(item))
    deduped = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def read_sequence_manifests(manifest_paths: str | Sequence[str], datasets: Sequence[str] | None = None) -> list[dict]:
    requested = {d.lower() for d in datasets or []}
    rows = []
    for path in expand_path_list(manifest_paths):
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        for row in read_jsonl(path):
            dataset = str(row.get("dataset", "")).lower()
            if requested and dataset not in requested:
                continue
            for frame in row.get("frames", []):
                if "meta_path" not in frame and frame.get("sample_dir"):
                    frame["meta_path"] = str(Path(frame["sample_dir"]) / "meta.json")
            rows.append(row)
    rows.sort(key=lambda x: str(x.get("sequence_id", "")))
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _dataset_from_meta(meta: dict) -> str | None:
    value = str(meta.get("dataset", "")).strip().lower()
    aliases = {
        "opentouch": "opentouch",
        "open_touch": "opentouch",
        "egotactile": "egotactile",
        "ego_tactile": "egotactile",
        "touchanything": "touchanything",
        "ego_touch": "touchanything",
        "egotouch": "touchanything",
    }
    if value in aliases:
        return aliases[value]
    if "original_hdf5_data" in meta and "scene" in meta and "demo" in meta:
        return "opentouch"
    return None


def _sequence_id(meta: dict, sample_dir: Path, dataset: str) -> str:
    split = str(meta.get("split", "train"))
    hand = str(meta.get("hand", "right" if int(meta.get("is_right", 1)) else "left"))
    if dataset == "opentouch":
        scene = meta.get("scene", "unknown_scene")
        clip = meta.get("demo", meta.get("clip", meta.get("clip_id", "unknown_clip")))
        return f"OpenTouch/{split}/{scene}/{clip}/{hand}"
    if dataset == "egotactile":
        rel_seq = meta.get("rel_seq") or meta.get("seq_dir") or sample_dir.parent.as_posix()
        return f"EgoTactile/{split}/{rel_seq}/{hand}"
    scene = meta.get("scene", "unknown_scene")
    task = meta.get("task", "unknown_task")
    clip = meta.get("clip", meta.get("clip_id", "unknown_clip"))
    return f"TouchAnything/{split}/{scene}/{task}/{clip}/{hand}"


def _scan_meta_path(meta_path: Path, datasets: set[str]) -> tuple[str, dict, dict] | None:
    try:
        meta = load_json(meta_path)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    dataset = _dataset_from_meta(meta)
    if dataset not in datasets:
        return None
    sample_dir = meta_path.parent
    frame_idx = int(meta.get("frame_idx", 0))
    if dataset == "touchanything":
        out = []
        for hand in ("left", "right"):
            seq_meta = {
                "dataset": dataset,
                "split": str(meta.get("split", "train")),
                "hand": hand,
                "is_right": 1 if hand == "right" else 0,
            }
            seq_key = _sequence_id({**meta, "hand": hand, "is_right": seq_meta["is_right"]}, sample_dir, dataset)
            frame = {
                "frame_idx": frame_idx,
                "sample_dir": str(sample_dir),
                "meta_path": str(meta_path),
                "hand": hand,
            }
            out.append((seq_key, seq_meta, frame))
        return ("__multi__", {}, {"items": out})
    hand = str(meta.get("hand", "right" if int(meta.get("is_right", 1)) else "left"))
    seq_meta = {
        "dataset": dataset,
        "split": str(meta.get("split", "train")),
        "hand": hand,
        "is_right": int(meta.get("is_right", 1 if hand == "right" else 0)),
    }
    seq_key = _sequence_id(meta, sample_dir, dataset)
    frame = {
        "frame_idx": frame_idx,
        "sample_dir": str(sample_dir),
        "meta_path": str(meta_path),
        "hand": hand,
    }
    return seq_key, seq_meta, frame


def _registry_tasks(root: Path) -> list[Path]:
    names = [
        "dataset_frames_registry.json",
        "egotactile_frames_registry.json",
        "touchanything_frames_registry.json",
    ]
    for name in names:
        registry_path = root / name
        if not registry_path.exists():
            continue
        try:
            registry = load_json(registry_path)
        except Exception:
            continue
        paths = []
        for item in registry if isinstance(registry, list) else []:
            sample_dir = item.get("sample_dir") or item.get("path")
            if not sample_dir:
                continue
            meta_path = Path(sample_dir) / "meta.json"
            if meta_path.exists():
                paths.append(meta_path)
        if paths:
            return paths
    return []


def discover_sequences(dataset_roots: Dict[str, str | None], datasets: Sequence[str], check_workers: int) -> list[dict]:
    requested = {d.lower() for d in datasets}
    unknown = requested - DATASET_NAMES
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")

    meta_paths: list[Path] = []
    for dataset, root in dataset_roots.items():
        if dataset not in requested or not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        registry_meta = _registry_tasks(root_path)
        if registry_meta:
            meta_paths.extend(registry_meta)
        else:
            meta_paths.extend(root_path.rglob("meta.json"))

    grouped: dict[str, dict] = {}
    workers = max(1, int(check_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in executor.map(lambda p: _scan_meta_path(p, requested), meta_paths, chunksize=128):
            if result is None:
                continue
            if result[0] == "__multi__":
                items = result[2]["items"]
            else:
                items = [result]
            for seq_key, seq_meta, frame in items:
                row = grouped.setdefault(
                    seq_key,
                    {
                        "sequence_id": seq_key,
                        "dataset": seq_meta["dataset"],
                        "split": seq_meta["split"],
                        "hand": seq_meta["hand"],
                        "is_right": seq_meta["is_right"],
                        "frames": [],
                    },
                )
                row["frames"].append(frame)

    rows = list(grouped.values())
    for row in rows:
        row["frames"].sort(key=lambda x: x["frame_idx"])
    rows.sort(key=lambda x: x["sequence_id"])
    return rows


def _normalized_depth(value: int | None) -> int | None:
    if value is None or int(value) < 0:
        return None
    return int(value)


def _rel_depth(path: str, root: str) -> int:
    rel = os.path.relpath(path, root)
    if rel == ".":
        return 0
    return len(rel.split(os.sep))


def _collect_scan_roots(root: str, split_depth: int, exclude_dirs: set[str]) -> list[str]:
    root = os.path.abspath(root)
    if split_depth <= 0:
        return [root]

    current = [root]
    for _ in range(split_depth):
        next_dirs = []
        for parent in current:
            try:
                with os.scandir(parent) as it:
                    for entry in it:
                        if entry.name in exclude_dirs:
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            next_dirs.append(entry.path)
            except OSError:
                continue
        if not next_dirs:
            return current
        current = next_dirs
    return current


def _walk_named_files(start_dir: str, root: str, filename: str, exclude_dirs: set[str], max_depth: int | None) -> list[str]:
    matches = []
    for dirpath, dirnames, filenames in os.walk(start_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        depth = _rel_depth(dirpath, root)
        if max_depth is not None and depth > max_depth:
            dirnames[:] = []
            continue
        if filename in filenames:
            matches.append(os.path.join(dirpath, filename))
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
    return matches


def find_named_files(
    root: str | Path,
    filename: str,
    check_workers: int,
    exclude_dirs: Iterable[str] | None = None,
    max_depth: int | None = None,
    split_depth: int = 0,
) -> list[str]:
    root = os.path.abspath(str(root))
    if not os.path.isdir(root):
        return []
    exclude = set(exclude_dirs or DEFAULT_SCAN_EXCLUDE_DIRS)
    scan_roots = _collect_scan_roots(root, int(split_depth), exclude)
    root_file = os.path.join(root, filename)
    matches = [root_file] if os.path.isfile(root_file) else []
    if len(scan_roots) == 1:
        matches.extend(_walk_named_files(scan_roots[0], root, filename, exclude, _normalized_depth(max_depth)))
        return sorted(set(matches))

    workers = max(1, min(int(check_workers), len(scan_roots)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_walk_named_files, scan_root, root, filename, exclude, _normalized_depth(max_depth))
            for scan_root in scan_roots
        ]
        for future in futures:
            matches.extend(future.result())
    return sorted(set(matches))


def _find_hdf5_files(root: str | Path) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    files = []
    try:
        for entry in root.iterdir():
            if entry.is_file() and entry.suffix.lower() in {".h5", ".hdf5"}:
                files.append(str(entry))
    except OSError:
        return []
    return sorted(files)


def _npz_frame_count(data, key: str) -> int:
    if key not in data:
        return 0
    try:
        shape = data[key].shape
    except Exception:
        return 0
    return int(shape[0]) if shape else 0


def _npz_array_readable(data, key: str) -> bool:
    if key not in data:
        return False
    try:
        arr = data[key]
        _ = arr[()] if arr.shape == () else arr[:]
        return True
    except Exception:
        return False


def _npz_hand_has_direct_data(data, dataset: str, hand: str) -> bool:
    if dataset == "egotactile":
        raw_key = f"{hand}_sensor_256_norm"
        gaussian_key = f"{hand}_pressure_continuous_subdiv"
        raw_frames = _npz_frame_count(data, raw_key)
        gaussian_frames = _npz_frame_count(data, gaussian_key)
        if raw_frames <= 0:
            return False
        if gaussian_frames <= 0:
            return False
        if not _npz_array_readable(data, raw_key) or not _npz_array_readable(data, gaussian_key):
            return False
        valid_key = f"{hand}_sensor_valid"
        if valid_key in data:
            try:
                valid = np.asarray(data[valid_key]).reshape(-1)
            except Exception:
                return False
            if valid.size == 0 or not bool(np.any(valid)):
                return False
        return True

    if dataset == "touchanything":
        raw_key = f"{hand}_pressure_grid"
        gaussian_key = f"{hand}_pressure_continuous_subdiv"
        raw_frames = _npz_frame_count(data, raw_key)
        gaussian_frames = _npz_frame_count(data, gaussian_key)
        return (
            raw_frames > 0
            and gaussian_frames > 0
            and _npz_array_readable(data, raw_key)
            and _npz_array_readable(data, gaussian_key)
        )

    return True


def _npz_source_rows(dataset: str, npz_files: Sequence[str], root: str | Path, name_prefix: str) -> list[dict]:
    rows = []
    root_path = Path(root)
    for npz_path in npz_files:
        path = Path(npz_path)
        try:
            with np.load(path, allow_pickle=False) as data:
                hands = [hand for hand in ("left", "right") if _npz_hand_has_direct_data(data, dataset, hand)]
        except Exception:
            continue
        if not hands:
            continue
        try:
            rel = path.parent.relative_to(root_path).as_posix()
        except ValueError:
            rel = path.parent.as_posix()
        for hand in hands:
            rows.append(
                {
                    "sequence_id": f"{name_prefix}/{rel}/{hand}",
                    "dataset": dataset,
                    "split": "all",
                    "hand": hand,
                    "is_right": 1 if hand == "right" else 0,
                    "source_type": "npz",
                    "source_path": str(path),
                    "frames": [],
                }
            )
    return rows


def _opentouch_hdf5_rows(hdf5_files: Sequence[str], root: str | Path) -> list[dict]:
    rows = []
    root_path = Path(root)
    try:
        import h5py
    except Exception:
        return rows

    for h5_path in hdf5_files:
        path = Path(h5_path)
        scene = path.stem
        try:
            rel_scene = path.relative_to(root_path).with_suffix("").as_posix()
        except ValueError:
            rel_scene = scene
        try:
            with h5py.File(path, "r") as f:
                if "data" not in f:
                    continue
                for demo_name in sorted(f["data"].keys()):
                    demo = f["data"][demo_name]
                    for hand in ("left", "right"):
                        raw_key = f"{hand}_pressure"
                        gaussian_key = f"{hand}_pressure_continuous_subdiv"
                        if raw_key not in demo and gaussian_key not in demo:
                            continue
                        rows.append(
                            {
                                "sequence_id": f"OpenTouch/all/{rel_scene}/{demo_name}/{hand}",
                                "dataset": "opentouch",
                                "split": "all",
                                "hand": hand,
                                "is_right": 1 if hand == "right" else 0,
                                "source_type": "hdf5",
                                "source_path": str(path),
                                "hdf5_demo": str(demo_name),
                                "scene": scene,
                                "frames": [],
                            }
                        )
        except Exception:
            continue
    return rows


def discover_pressure_sources(
    dataset_roots: Dict[str, str | None],
    datasets: Sequence[str],
    check_workers: int,
    egotactile_npz_name: str = "pressure_grids_egotactile.npz",
    scan_exclude_dirs: Iterable[str] | None = None,
    touchanything_scan_depth: int = 3,
    egotactile_scan_depth: int = 4,
    touchanything_scan_split_depth: int = 2,
    egotactile_scan_split_depth: int = 3,
) -> list[dict]:
    requested = {d.lower() for d in datasets}
    unknown = requested - DATASET_NAMES
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    exclude = set(scan_exclude_dirs or DEFAULT_SCAN_EXCLUDE_DIRS)
    rows: list[dict] = []

    if "opentouch" in requested and dataset_roots.get("opentouch"):
        hdf5_files = _find_hdf5_files(dataset_roots["opentouch"])
        rows.extend(_opentouch_hdf5_rows(hdf5_files, dataset_roots["opentouch"]))

    if "touchanything" in requested and dataset_roots.get("touchanything"):
        files = find_named_files(
            dataset_roots["touchanything"],
            "pressure_grids.npz",
            check_workers=check_workers,
            exclude_dirs=exclude,
            max_depth=touchanything_scan_depth,
            split_depth=touchanything_scan_split_depth,
        )
        rows.extend(_npz_source_rows("touchanything", files, dataset_roots["touchanything"], "TouchAnything/all"))

    if "egotactile" in requested and dataset_roots.get("egotactile"):
        files = find_named_files(
            dataset_roots["egotactile"],
            egotactile_npz_name,
            check_workers=check_workers,
            exclude_dirs=exclude,
            max_depth=egotactile_scan_depth,
            split_depth=egotactile_scan_split_depth,
        )
        rows.extend(_npz_source_rows("egotactile", files, dataset_roots["egotactile"], "EgoTactile/all"))

    rows.sort(key=lambda x: x["sequence_id"])
    return rows


def atomic_touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o644)
    os.close(fd)
