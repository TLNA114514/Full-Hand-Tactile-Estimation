#!/usr/bin/env python3
"""Adapt checked-in local absolute paths to the ModelArts filesystem.

The deployment sync invokes this script on the remote checkout after rsync.
It is intentionally deterministic and idempotent: running it repeatedly does
not modify files after the first successful pass.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile


LOCAL_WORKSPACE = "/code/users/jiangrui/Full-Hand-Tactile-Estimation"
LOCAL_DATA_ROOT = "/data1/jiangrui"
DEFAULT_REMOTE_ROOT = "/home/ma-user/work/cfzhao"

SOURCE_SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml"}
SKIP_DIRS = {
    ".git",
    ".agents",
    ".codex",
    "__pycache__",
    "wandb",
    "logs",
    "lightning_logs",
    "checkpoints",
    "eval_reports",
    "index_cache",
    "third_party",
    "outputs",
    "results",
    "reports",
    "demo_output",
    "data_integrity_audits",
    "input_prior_audits",
    "amp_audits",
    "memorization",
    "hdf5_manifest_cache",
    "index_cache",
    "sidecars",
    "models",
    "envs",
    "pre_dialog",
}

SKIP_DIR_PREFIXES = ("eval_reports",)


def _path_mappings(remote_root: str) -> dict[str, str]:
    remote_root = remote_root.rstrip("/")
    return {
        LOCAL_WORKSPACE: f"{remote_root}/Full-Hand-Tactile-Estimation",
        f"{LOCAL_DATA_ROOT}/OpenTouch Data": f"{remote_root}/OpenTouch Data",
        f"{LOCAL_DATA_ROOT}/EgoTouch": f"{remote_root}/EgoTouch",
        f"{LOCAL_DATA_ROOT}/EgoTactile": f"{remote_root}/EgoTactile",
    }


def _iter_source_files(root: Path):
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [
            name
            for name in dirs
            if name not in SKIP_DIRS
            and not any(name.startswith(prefix) for prefix in SKIP_DIR_PREFIXES)
        ]
        current = Path(current_root)
        for name in files:
            path = current / name
            if path.name == Path(__file__).name or path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            yield path


def _atomic_write_text(path: Path, content: str) -> None:
    mode = path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.adapt.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def adapt_paths(root: Path, remote_root: str, dry_run: bool = False) -> int:
    mappings = sorted(
        _path_mappings(remote_root).items(), key=lambda item: len(item[0]), reverse=True
    )
    changed = 0
    for path in _iter_source_files(root):
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        adapted = content
        for local_path, remote_path in mappings:
            adapted = adapted.replace(local_path, remote_path)
        if adapted == content:
            continue
        changed += 1
        print(f"[path-adapt] {path.relative_to(root)}")
        if not dry_run:
            _atomic_write_text(path, adapted)
    action = "would update" if dry_run else "updated"
    print(f"[path-adapt] {action} {changed} source file(s).")
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rewrite local absolute source paths for ModelArts deployment."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Checkout root to scan (defaults to this script's directory).",
    )
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("REMOTE_WORK_ROOT", DEFAULT_REMOTE_ROOT),
        help="Remote parent containing the project and datasets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Checkout root does not exist: {root}")
    adapt_paths(root, args.remote_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
