#!/usr/bin/env python3
"""Discover or atomically rebuild authoritative HDF5 query manifests.

Existing published manifests are preferred. A missing manifest is rebuilt only
from finalized sequence HDF5 metadata; legacy sample directories and raw source
datasets are never scanned or modified.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactile_input_priors.hdf5_manifest import (  # noqa: E402
    AtomicJsonlWriter,
    manifest_rows_from_hdf5,
    sequence_manifest_row,
    sha256_file,
    verify_sequence_hdf5,
    write_json_atomic,
)


DATASET_NAMES = {
    "opentouch": "OpenTouch",
    "touchanything": "TouchAnything",
}
DEFAULT_ROOTS = {
    "opentouch": (
        "/home/ma-user/work/cfzhao/OpenTouch Data/full_dataset",
        "/data1/jiangrui/OpenTouch Data/full_dataset",
    ),
    "touchanything": (
        "/home/ma-user/work/cfzhao/EgoTouch/extracted_frames",
        "/data1/jiangrui/EgoTouch/extracted_frames",
    ),
}
PREFERRED_SPLIT_ORDER = ("train", "val", "test_seen", "test_unseen", "test")


class ManifestLock:
    def __init__(self, path: Path, timeout: float):
        self.path = path
        self.timeout = float(timeout)
        self.acquired = False

    def __enter__(self):
        started = time.monotonic()
        while True:
            try:
                self.path.mkdir(parents=False)
                self.acquired = True
                owner = {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "created_unix": time.time(),
                }
                (self.path / "owner.json").write_text(
                    json.dumps(owner, sort_keys=True) + "\n", encoding="utf-8"
                )
                return self
            except FileExistsError:
                if time.monotonic() - started > self.timeout:
                    owner_path = self.path / "owner.json"
                    owner = (
                        owner_path.read_text(encoding="utf-8").strip()
                        if owner_path.is_file()
                        else "unavailable"
                    )
                    raise TimeoutError(
                        f"Timed out waiting for manifest lock {self.path}; owner={owner}"
                    )
                time.sleep(2.0)

    def __exit__(self, exc_type, exc, traceback):
        if self.acquired:
            try:
                (self.path / "owner.json").unlink(missing_ok=True)
                self.path.rmdir()
            finally:
                self.acquired = False
        return False


def _resolve_root(dataset: str, value: str) -> Path:
    if value:
        root = Path(value).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError(root)
        return root
    env_name = f"{dataset.upper()}_DATA_ROOT"
    candidates = []
    if os.environ.get(env_name):
        candidates.append(os.environ[env_name])
    candidates.extend(DEFAULT_ROOTS[dataset])
    existing = []
    seen = set()
    for item in candidates:
        candidate = Path(item).expanduser()
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        key = os.path.normcase(os.fspath(resolved))
        if key in seen:
            continue
        seen.add(key)
        existing.append(resolved)
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise FileNotFoundError(
            f"Could not auto-detect {dataset} processed HDF5 root. Set {env_name} "
            "or pass --processed-root. Checked: " + ", ".join(candidates)
        )
    raise RuntimeError(
        f"Multiple {dataset} roots exist; pass --processed-root explicitly: {existing}"
    )


def _manifest_paths(root: Path, dataset: str, split: str) -> tuple[Path, Path, Path]:
    prefix = root / "manifests" / f"{dataset}_{split}"
    return (
        Path(f"{prefix}.queries.jsonl"),
        Path(f"{prefix}.sequences.jsonl"),
        Path(f"{prefix}.summary.json"),
    )


def _split_hdf5_files(root: Path, dataset: str, split: str) -> list[Path]:
    split_root = root / split
    if not split_root.is_dir():
        return []
    expected_dataset = DATASET_NAMES[dataset]
    accepted = []
    for path in sorted(split_root.rglob("*.h5")):
        try:
            summary = verify_sequence_hdf5(path, deep=False)
        except Exception as exc:
            raise RuntimeError(f"Invalid finalized sequence HDF5 {path}: {exc}") from exc
        if summary.get("dataset") != expected_dataset or summary.get("split") != split:
            continue
        accepted.append(path.resolve())
    return accepted


def _discover_splits(root: Path, dataset: str) -> list[str]:
    splits = set()
    manifest_dir = root / "manifests"
    prefix = f"{dataset}_"
    suffix = ".queries.jsonl"
    if manifest_dir.is_dir():
        for path in manifest_dir.glob(f"{dataset}_*.queries.jsonl"):
            splits.add(path.name[len(prefix) : -len(suffix)])
    # Limit implicit filesystem discovery to the protocol splits. Walking every
    # top-level directory can accidentally traverse a legacy frame tree with
    # millions of files. Custom HDF5-only splits remain available via --splits.
    for split in PREFERRED_SPLIT_ORDER:
        split_root = root / split
        if split_root.is_dir() and next(split_root.rglob("*.h5"), None):
            splits.add(split)
    order = {name: index for index, name in enumerate(PREFERRED_SPLIT_ORDER)}
    return sorted(splits, key=lambda value: (order.get(value, len(order)), value))


def _parse_splits(value: str, root: Path, dataset: str) -> list[str]:
    if str(value).strip().lower() == "auto":
        splits = _discover_splits(root, dataset)
    else:
        splits = [item.strip() for item in str(value).split(",") if item.strip()]
    if not splits:
        raise RuntimeError(f"No sequence-HDF5 splits found under {root}")
    if len(splits) != len(set(splits)):
        raise ValueError(f"Duplicate split names: {splits}")
    return splits


def _write_manifests(root: Path, dataset: str, split: str, h5_paths: Iterable[Path]):
    query_path, sequence_path, summary_path = _manifest_paths(root, dataset, split)
    query_count = 0
    sequence_count = 0
    with AtomicJsonlWriter(query_path) as query_writer, AtomicJsonlWriter(sequence_path) as sequence_writer:
        for h5_path in h5_paths:
            for row in manifest_rows_from_hdf5(h5_path, root):
                query_writer.write(row)
                query_count += 1
            sequence_writer.write(sequence_manifest_row(h5_path, root))
            sequence_count += 1
    summary = {
        "schema": "tactile_depth_manifest_rebuild_v1",
        "dataset": DATASET_NAMES[dataset],
        "dataset_key": dataset,
        "split": split,
        "processed_root": str(root),
        "sequence_count": sequence_count,
        "query_count": query_count,
        "query_manifest": query_path.relative_to(root).as_posix(),
        "query_manifest_sha256": sha256_file(query_path),
        "sequence_manifest": sequence_path.relative_to(root).as_posix(),
        "sequence_manifest_sha256": sha256_file(sequence_path),
        "source": "finalized_sequence_hdf5_only",
    }
    write_json_atomic(summary_path, summary)
    return query_path, summary


def resolve_manifests(args, *, emit_paths: bool = True) -> list[Path]:
    root = _resolve_root(args.dataset, args.processed_root)
    splits = _parse_splits(args.splits, root, args.dataset)
    resolved = []
    for split in splits:
        query_path, _, _ = _manifest_paths(root, args.dataset, split)
        if query_path.is_file():
            if not args.print_paths:
                print(f"[depth-manifest] reuse split={split}: {query_path}", file=sys.stderr)
            resolved.append(query_path.resolve())
            continue
        if not args.create_missing:
            raise FileNotFoundError(
                f"Missing query manifest for split={split}: {query_path}. "
                "Pass --create-missing to rebuild it from finalized HDF5."
            )
        lock_path = query_path.parent / f".{args.dataset}_{split}.manifest.lock"
        query_path.parent.mkdir(parents=True, exist_ok=True)
        with ManifestLock(lock_path, args.lock_timeout):
            if query_path.is_file():
                resolved.append(query_path.resolve())
                continue
            h5_paths = _split_hdf5_files(root, args.dataset, split)
            if not h5_paths:
                raise FileNotFoundError(
                    f"No finalized {DATASET_NAMES[args.dataset]}/{split} HDF5 files "
                    f"were found under {root / split}"
                )
            if not args.print_paths:
                print(
                    f"[depth-manifest] rebuilding split={split} from "
                    f"{len(h5_paths)} finalized sequence files",
                    file=sys.stderr,
                )
            rebuilt_path, _ = _write_manifests(root, args.dataset, split, h5_paths)
            resolved.append(rebuilt_path.resolve())
    if emit_paths:
        for path in resolved:
            print(path)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_NAMES), default="touchanything")
    parser.add_argument("--processed-root", default="")
    parser.add_argument("--splits", default="auto")
    parser.add_argument(
        "--create-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Atomically rebuild missing manifests from finalized sequence HDF5.",
    )
    parser.add_argument("--lock-timeout", type=float, default=3600.0)
    parser.add_argument("--print-paths", action="store_true")
    return parser


def main() -> None:
    resolve_manifests(build_parser().parse_args())


if __name__ == "__main__":
    main()
