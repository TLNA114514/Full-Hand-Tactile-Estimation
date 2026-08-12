#!/usr/bin/env python3
"""Safely remove unused wrist JPEGs from processed TouchAnything samples.

Dry-run is the default. The tool never follows symlinks, never matches videos,
and refuses Hugging Face/cache paths. By default it also requires the
corresponding sequence HDF5 to exist and pass structural verification.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from convert_sequence_hdf5 import (
        discover_splits,
        bounded_ordered_map,
        load_json_bytes,
        metadata_dataset_name,
        output_h5_path,
        sequence_identity,
        suspicious_raw_or_hf_path,
    )
    from hdf5_storage import verify_sequence_hdf5, write_json_atomic
except ImportError:
    from .convert_sequence_hdf5 import (
        discover_splits,
        bounded_ordered_map,
        load_json_bytes,
        metadata_dataset_name,
        output_h5_path,
        sequence_identity,
        suspicious_raw_or_hf_path,
    )
    from .hdf5_storage import verify_sequence_hdf5, write_json_atomic


WRIST_NAMES = frozenset(("left.jpg", "right.jpg"))


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def split_sample_dirs(processed_root: Path, split: str) -> list[Path]:
    split_root = processed_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(split_root)
    result = []
    with os.scandir(split_root) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            sample_dir = Path(entry.path)
            if (sample_dir / "meta.json").is_file():
                result.append(sample_dir)
    result.sort()
    return result


def inspect_sample(args) -> dict[str, Any]:
    sample_dir_text, processed_root_text, split = args
    sample_dir = Path(sample_dir_text)
    processed_root = Path(processed_root_text)
    try:
        if sample_dir.is_symlink():
            raise RuntimeError("sample directory is a symlink")
        meta, _ = load_json_bytes(sample_dir / "meta.json")
        dataset = metadata_dataset_name(meta, "touchanything")
        if dataset != "touchanything":
            raise RuntimeError(f"metadata belongs to {dataset!r}, not TouchAnything")
        sequence_key, sequence_parts = sequence_identity(meta, "touchanything")
        chest_name = str(meta.get("views", {}).get("chest", "chest.jpg"))
        if chest_name != "chest.jpg" or not (sample_dir / chest_name).is_file():
            raise RuntimeError("processed chest.jpg is missing")

        candidates = []
        for name in sorted(WRIST_NAMES):
            path = sample_dir / name
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"{name} is not a regular non-symlink file")
            resolved = path.resolve()
            if not is_within(resolved, processed_root):
                raise RuntimeError(f"{name} resolves outside processed root")
            if resolved.parent != sample_dir.resolve() or resolved.name not in WRIST_NAMES:
                raise RuntimeError(f"unsafe wrist candidate path: {resolved}")
            candidates.append(
                {
                    "path": str(resolved),
                    "relpath": resolved.relative_to(processed_root).as_posix(),
                    "size": int(resolved.stat().st_size),
                }
            )
        h5_path = output_h5_path(processed_root, split, sequence_parts).resolve()
        return {
            "status": "ok",
            "sample_dir": str(sample_dir),
            "sequence_key": sequence_key,
            "h5_path": str(h5_path),
            "candidates": candidates,
        }
    except Exception as exc:
        return {
            "status": "error",
            "sample_dir": str(sample_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }


def unlink_candidate(args) -> dict[str, Any]:
    path_text, processed_root_text = args
    path = Path(path_text)
    processed_root = Path(processed_root_text)
    try:
        if path.name not in WRIST_NAMES:
            raise RuntimeError(f"refusing unexpected filename {path.name!r}")
        if path.suffix.lower() != ".jpg":
            raise RuntimeError(f"refusing non-JPEG path {path}")
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("candidate is no longer a regular non-symlink file")
        resolved = path.resolve()
        if not is_within(resolved, processed_root):
            raise RuntimeError("candidate resolves outside processed root")
        size = int(resolved.stat().st_size)
        resolved.unlink()
        return {"status": "deleted", "path": str(resolved), "size": size}
    except FileNotFoundError:
        return {"status": "missing", "path": str(path), "size": 0}
    except Exception as exc:
        return {
            "status": "error",
            "path": str(path),
            "size": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def parse_splits(value: str, processed_root: Path) -> list[str]:
    if value.strip().lower() == "auto":
        splits = discover_splits(processed_root)
    else:
        splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("No processed splits selected or discovered")
    return sorted(set(splits))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run or delete only left.jpg/right.jpg from an extracted "
            "TouchAnything processed root."
        )
    )
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--splits", default="auto")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually unlink validated wrist JPEGs. Without this flag nothing is changed.",
    )
    parser.add_argument(
        "--allow-without-hdf5",
        action="store_true",
        help=(
            "Allow deletion before a corresponding verified sequence HDF5 exists. "
            "This weakens the default safety gate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    processed_root = Path(args.processed_root).expanduser().resolve()
    if suspicious_raw_or_hf_path(processed_root):
        raise RuntimeError(
            f"Refusing Hugging Face/cache/raw-looking path: {processed_root}"
        )
    if not processed_root.is_dir():
        raise FileNotFoundError(processed_root)
    registry_path = processed_root / "touchanything_frames_registry.json"
    if not registry_path.is_file():
        raise RuntimeError(
            "Refusing cleanup because the processed TouchAnything registry is "
            f"missing: {registry_path}"
        )
    splits = parse_splits(args.splits, processed_root)
    print(
        f"TouchAnything wrist JPEG prune {'EXECUTE' if args.execute else 'DRY RUN'}",
        flush=True,
    )
    print(f"  processed root: {processed_root}", flush=True)
    print(f"  splits:         {','.join(splits)}", flush=True)
    print("  allowed names:  left.jpg, right.jpg", flush=True)
    print("  videos/raw/HF:  never matched", flush=True)

    inspections = []
    for split in splits:
        sample_dirs = split_sample_dirs(processed_root, split)
        print(f"[{split}] Inspecting {len(sample_dirs)} processed folders...", flush=True)
        scan_args = (
            (str(path), str(processed_root), split) for path in sample_dirs
        )
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in bounded_ordered_map(
                executor,
                inspect_sample,
                scan_args,
                max_pending=args.workers * 4,
            ):
                inspections.append(result)
    errors = [row for row in inspections if row["status"] == "error"]
    if errors:
        preview = "\n".join(
            f"  - {row['sample_dir']}: {row['error']}" for row in errors[:10]
        )
        raise RuntimeError(
            f"Refusing cleanup because {len(errors)} processed folders failed "
            f"preflight:\n{preview}"
        )

    verified_h5: dict[str, bool] = {}
    if args.execute and not args.allow_without_hdf5:
        for h5_path_text in sorted({row["h5_path"] for row in inspections}):
            h5_path = Path(h5_path_text)
            verify_sequence_hdf5(h5_path, deep=False)
            verified_h5[h5_path_text] = True

    candidates = [
        candidate
        for row in inspections
        if (
            not args.execute
            or args.allow_without_hdf5
            or verified_h5.get(row["h5_path"], False)
        )
        for candidate in row["candidates"]
    ]
    total_bytes = sum(row["size"] for row in candidates)
    print(
        f"Validated {len(candidates)} wrist JPEG(s), "
        f"{total_bytes / (1024 ** 3):.3f} GiB.",
        flush=True,
    )
    if not args.execute:
        print(
            "Dry run complete; no files were deleted. Re-run with --execute after review.",
            flush=True,
        )
        return

    started = time.monotonic()
    results = []
    delete_args = ((row["path"], str(processed_root)) for row in candidates)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, result in enumerate(
            bounded_ordered_map(
                executor,
                unlink_candidate,
                delete_args,
                max_pending=args.workers * 4,
            ),
            start=1,
        ):
            results.append(result)
            if completed % 10000 == 0 or completed == len(candidates):
                print(
                    f"Deleted/checked {completed}/{len(candidates)} wrist JPEG(s)...",
                    flush=True,
                )
    status_counts = Counter(row["status"] for row in results)
    failures = [row for row in results if row["status"] == "error"]
    report = {
        "schema": "touchanything_processed_wrist_prune_v1",
        "processed_root": str(processed_root),
        "splits": splits,
        "allowed_filenames": sorted(WRIST_NAMES),
        "required_verified_hdf5": not args.allow_without_hdf5,
        "candidate_count": len(candidates),
        "candidate_bytes": total_bytes,
        "status_counts": dict(status_counts),
        "deleted_bytes": sum(
            row["size"] for row in results if row["status"] == "deleted"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "failures": failures[:100],
    }
    report_path = processed_root / "manifests/touchanything_wrist_prune_report.json"
    write_json_atomic(report_path, report)
    if failures:
        raise RuntimeError(
            f"{len(failures)} wrist JPEG(s) failed deletion; inspect {report_path}"
        )
    print(
        f"Cleanup complete: {status_counts.get('deleted', 0)} JPEG(s), "
        f"{report['deleted_bytes'] / (1024 ** 3):.3f} GiB; report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
