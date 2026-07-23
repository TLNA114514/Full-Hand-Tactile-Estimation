#!/usr/bin/env python3
"""Check the isolated SAM3 bbox environment and optionally build a model."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("compat-cu124", "official"), required=True)
    parser.add_argument("--sam3-root", type=Path, required=True)
    parser.add_argument("--model-version", choices=("sam3", "sam3.1"), default="sam3")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def version_tuple(value: str) -> tuple[int, ...]:
    parts = value.split("+", 1)[0].split(".")
    return tuple(int(part) for part in parts[:3])


def main() -> int:
    args = parse_args()
    args.sam3_root = args.sam3_root.expanduser().resolve()
    sys.path.insert(0, str(args.sam3_root))

    import torch
    import h5py

    print(f"[doctor] Python: {platform.python_version()}")
    print(f"[doctor] PyTorch: {torch.__version__}")
    print(f"[doctor] h5py/HDF5: {h5py.__version__}/{h5py.version.hdf5_version}")
    print(f"[doctor] Torch CUDA runtime: {torch.version.cuda}")
    print(f"[doctor] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[doctor] GPU: {torch.cuda.get_device_name(0)}")

    from sam3.model_builder import build_sam3_predictor

    print(f"[doctor] SAM3 import: {args.sam3_root}")
    if args.profile == "compat-cu124" and version_tuple(torch.__version__) < (2, 7):
        print(
            "[doctor] WARNING: this is a compatibility profile below the current "
            "upstream PyTorch>=2.7 requirement. Use compile=False, use_fa3=False, "
            "and treat the real-video smoke test as mandatory."
        )

    if args.checkpoint is None:
        print("[doctor] Model build skipped because --checkpoint was not supplied.")
        return 0
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-visible GPU is required for the SAM3 video predictor.")

    kwargs = {
        "checkpoint_path": str(checkpoint),
        "version": args.model_version,
        "compile": False,
        "async_loading_frames": False,
    }
    if args.model_version == "sam3.1":
        kwargs.update(use_fa3=False, use_rope_real=False, max_num_objects=4, multiplex_count=4)
    model = build_sam3_predictor(**kwargs)
    print(f"[doctor] {args.model_version} checkpoint build succeeded.")
    del model
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
