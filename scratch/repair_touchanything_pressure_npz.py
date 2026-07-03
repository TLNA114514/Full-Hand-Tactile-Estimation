import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    np = None


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_mapping(path):
    with Path(path).open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    parsed = []
    for key, sensor_idx in mapping.items():
        row, col = [int(x) for x in key.split(",")]
        parsed.append((row, col, int(sensor_idx)))
    return parsed


def normalize_sensor(values, mode):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size != 256:
        raise ValueError(f"Expected 256 sensor values, got {values.size}")

    if mode == "none":
        return values
    if mode == "clip01":
        return np.clip(values, 0.0, 1.0)
    if mode == "max":
        max_value = float(np.nanmax(values)) if values.size else 0.0
        if max_value > 1.0:
            values = values / max_value
        return np.clip(values, 0.0, 1.0)
    raise ValueError(f"Unknown normalization mode: {mode}")


def sensors_to_grid(records, sensor_key, mapping, normalization):
    grids = np.full((len(records), 21, 21), np.nan, dtype=np.float32)
    for frame_idx, record in enumerate(records):
        if sensor_key not in record or record[sensor_key] is None:
            continue
        sensor = normalize_sensor(record[sensor_key], normalization)
        for row, col, sensor_idx in mapping:
            if 0 <= row < 21 and 0 <= col < 21 and 0 <= sensor_idx < sensor.size:
                grids[frame_idx, row, col] = sensor[sensor_idx]
    return grids


def npz_is_valid(path):
    try:
        data = np.load(path)
        try:
            return "left_pressure_grid" in data and "right_pressure_grid" in data
        finally:
            data.close()
    except Exception:
        return False


def discover_clip_dirs_under_task(task_dir):
    clip_dirs = []
    try:
        with os.scandir(task_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                clip_dir = Path(entry.path)
                if (clip_dir / "jq_pressure.json").exists():
                    clip_dirs.append(clip_dir)
    except OSError:
        pass
    return clip_dirs


def discover_clip_dirs(root, scan_workers=32):
    root = Path(root)
    task_dirs = []
    excluded = {"extracted_frames", "metadata", "__pycache__"}
    try:
        with os.scandir(root) as scenes:
            for scene in scenes:
                if scene.name in excluded or not scene.is_dir(follow_symlinks=False):
                    continue
                try:
                    with os.scandir(scene.path) as tasks:
                        for task in tasks:
                            if task.name in excluded or not task.is_dir(follow_symlinks=False):
                                continue
                            task_dirs.append(task.path)
                except OSError:
                    continue
    except OSError:
        return []

    if not task_dirs:
        return []

    clip_dirs = []
    scan_workers = max(1, int(scan_workers))
    with ThreadPoolExecutor(max_workers=min(scan_workers, len(task_dirs))) as executor:
        futures = [executor.submit(discover_clip_dirs_under_task, task_dir) for task_dir in task_dirs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Discovering TouchAnything clips"):
            clip_dirs.extend(future.result())
    return sorted(clip_dirs)


def repair_clip(clip_dir, left_mapping, right_mapping, normalization, backup_bad):
    clip_dir = Path(clip_dir)
    jq_path = clip_dir / "jq_pressure.json"
    npz_path = clip_dir / "pressure_grids.npz"
    for stale_tmp in (
        npz_path.with_name(npz_path.name + ".tmp"),
        npz_path.with_name(npz_path.name + ".tmp.npz"),
    ):
        if stale_tmp.exists():
            stale_tmp.unlink()
    if not jq_path.exists():
        return "missing_jq_pressure"

    records = load_jsonl(jq_path)
    if not records:
        return "empty_jq_pressure"

    left_grids = sensors_to_grid(records, "sensor_left", left_mapping, normalization)
    right_grids = sensors_to_grid(records, "sensor_right", right_mapping, normalization)

    if npz_path.exists() and backup_bad:
        backup_path = npz_path.with_suffix(npz_path.suffix + ".bad")
        if not backup_path.exists():
            os.replace(npz_path, backup_path)

    tmp_path = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        left_pressure_grid=left_grids,
        right_pressure_grid=right_grids,
        grid_size=np.asarray(21, dtype=np.int32),
        num_frames=np.asarray(len(records), dtype=np.int32),
        separate_normalization=np.asarray(False),
        repaired_from_jq_pressure=np.asarray(True),
        repair_normalization=np.asarray(normalization),
    )
    os.replace(tmp_path, npz_path)
    return "repaired"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Repair missing/corrupted TouchAnything pressure_grids.npz from jq_pressure.json sensor arrays."
    )
    parser.add_argument("--root", default="/data1/jiangrui/EgoTouch", help="TouchAnything/EgoTouch root.")
    parser.add_argument("--clip_dir", action="append", default=[], help="Specific clip dir. Can be passed multiple times.")
    parser.add_argument(
        "--normalization",
        choices=["clip01", "max", "none"],
        default="clip01",
        help="How to normalize jq_pressure sensor_left/right before writing grids.",
    )
    parser.add_argument(
        "--only_bad",
        action="store_true",
        help="Only repair clips where pressure_grids.npz is missing/corrupted or lacks required keys.",
    )
    parser.add_argument(
        "--backup_bad",
        action="store_true",
        help="Rename an existing bad pressure_grids.npz to pressure_grids.npz.bad before writing repaired npz.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--check_workers", type=int, default=32, help="Threads for checking/repairing clip npz files.")
    args = parser.parse_args()

    if np is None:
        raise ImportError("numpy is required to repair TouchAnything pressure_grids.npz")

    left_mapping = load_mapping(repo_root / "TouchAnything/configs/pressure_position_mapping_left.json")
    right_mapping = load_mapping(repo_root / "TouchAnything/configs/pressure_position_mapping_right.json")

    if args.clip_dir:
        clip_dirs = [Path(p) for p in args.clip_dir]
    else:
        clip_dirs = discover_clip_dirs(args.root, scan_workers=args.check_workers)
    print(f"Discovered {len(clip_dirs)} clip dirs with jq_pressure.json under {args.root}")
    repaired = 0
    skipped = 0
    failed = 0
    examples = []

    def process_one(clip_dir):
        npz_path = clip_dir / "pressure_grids.npz"
        valid = npz_path.exists() and npz_is_valid(npz_path)
        if args.only_bad and valid:
            return "skipped", clip_dir, "valid"
        if args.dry_run:
            status = "would_repair" if not valid or not args.only_bad else "skip_valid"
        else:
            try:
                status = repair_clip(
                    clip_dir,
                    left_mapping,
                    right_mapping,
                    args.normalization,
                    args.backup_bad,
                )
            except Exception as exc:
                status = f"failed: {exc}"
        if status in ("repaired", "would_repair"):
            return "repaired", clip_dir, status
        if status.startswith("failed") or status.startswith("missing") or status.startswith("empty"):
            return "failed", clip_dir, status
        return "skipped", clip_dir, status

    check_workers = max(1, int(args.check_workers))
    if check_workers == 1:
        results = (process_one(clip_dir) for clip_dir in clip_dirs)
        iterator = tqdm(results, total=len(clip_dirs), desc="Repairing pressure npz")
        for kind, clip_dir, status in iterator:
            if kind == "repaired":
                repaired += 1
            elif kind == "failed":
                failed += 1
                if len(examples) < 10:
                    examples.append(f"{clip_dir}: {status}")
            else:
                skipped += 1
    else:
        with ThreadPoolExecutor(max_workers=check_workers) as executor:
            futures = [executor.submit(process_one, clip_dir) for clip_dir in clip_dirs]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Repairing pressure npz"):
                kind, clip_dir, status = future.result()
                if kind == "repaired":
                    repaired += 1
                elif kind == "failed":
                    failed += 1
                    if len(examples) < 10:
                        examples.append(f"{clip_dir}: {status}")
                else:
                    skipped += 1

    print("TouchAnything pressure npz repair finished.")
    print(f"  repaired/would repair: {repaired}")
    print(f"  skipped: {skipped}")
    print(f"  failed: {failed}")
    if examples:
        print("  first failures:")
        for item in examples:
            print(f"    {item}")


if __name__ == "__main__":
    main()
