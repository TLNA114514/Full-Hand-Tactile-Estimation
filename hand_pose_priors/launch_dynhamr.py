#!/usr/bin/env python3
"""Launch pinned Dyn-HaMR on a prepared TouchAnything trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _git_commit(root: Path) -> str:
    snapshot_marker = root / ".dynhamr_commit"
    if snapshot_marker.is_file():
        commit = snapshot_marker.read_text(encoding="utf-8").strip()
        if commit:
            return commit
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def launch(args: argparse.Namespace) -> None:
    checkout = Path(args.checkout).expanduser().resolve(strict=True)
    trial_root = Path(args.trial_root).expanduser().resolve(strict=True)
    preparation = json.loads(
        (trial_root / "PREPARE_DONE.json").read_text(encoding="utf-8")
    )
    sequence_name = str(preparation["request"]["sequence_name"])
    source_dir = checkout / "dyn-hamr"
    entrypoint = source_dir / "run_opt.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(entrypoint)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else trial_root / "outputs" / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dir = trial_root / "dynhamr/cameras" / sequence_name / "shot-0"
    command = [
        sys.executable,
        "-u",
        str(entrypoint),
        "data=video_vipe",
        f"data.root={trial_root}",
        f"data.seq={sequence_name}",
        f"data.src_path={trial_root / 'videos' / (sequence_name + '.mp4')}",
        f"data.sources.images={trial_root / 'images' / sequence_name}",
        f"data.sources.cameras={camera_dir}",
        f"data.sources.tracks={trial_root / 'dynhamr/track_preds' / sequence_name}",
        f"data.sources.shots={trial_root / 'dynhamr/shot_idcs' / (sequence_name + '.json')}",
        "data.use_cams=true",
        "data.use_vipe=false",
        "data.track_ids=all",
        "data.shot_idx=0",
        "data.start_idx=0",
        "data.end_idx=-1",
        "data.split_cameras=true",
        "is_static=false",
        "model.opt_scale=false",
        "model.opt_cams=false",
        "run_prior=false",
        "run_vis=false",
        f"run_opt={'false' if args.validate_only else 'true'}",
        "gpu=0",
        f"+seed={args.seed}",
        f"fps={float(preparation['request']['fps']):g}",
        f"optim.root.num_iters={args.root_iters}",
        f"optim.smooth.num_iters={args.smooth_iters}",
        f"optim.options.save_every={args.save_every}",
        "optim.options.vis_every=-1",
        "optim.options.save_meshes=false",
        f"hydra.run.dir={output_dir}",
    ]
    metadata = {
        "schema": "touchanything_dynhamr_launch",
        "schema_version": "1.0.0",
        "checkout": str(checkout),
        "commit": _git_commit(checkout),
        "trial_root": str(trial_root),
        "output_dir": str(output_dir),
        "validate_only": bool(args.validate_only),
        "command": command,
    }
    _write_json(output_dir / "launch_config.json", metadata)
    print("[dynhamr-run] " + " ".join(command), flush=True)
    environment = dict(os.environ)
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    process = subprocess.Popen(
        command,
        cwd=source_dir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with (output_dir / "launch.log").open("a", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)
    _write_json(
        output_dir / ("VALIDATE_DONE.json" if args.validate_only else "OPTIMIZE_DONE.json"),
        metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--trial-root", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--run-name", default="static_focal_v1")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--root-iters", type=int, default=50)
    parser.add_argument("--smooth-iters", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--validate-only", action="store_true")
    return parser


if __name__ == "__main__":
    launch(build_parser().parse_args())
