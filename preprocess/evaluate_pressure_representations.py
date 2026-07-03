#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from preprocess.common import ARTIFACT_ROOT, REPO_ROOT
from preprocess.representation_eval.io import (
    append_jsonl,
    discover_pressure_sources,
    discover_sequences,
    read_sequence_manifests,
    write_csv,
)


METHOD_CHOICES = [
    "egotactile_heatmap",
    "ot_centered_mano",
    "ot_discrete_heatmap",
    "ot_raw_heatmap",
    "preprocess_gaussian",
]


WORKER_MESH = None
WORKER_SENSOR_GEOMS = {}
WORKER_ARGS = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate pressure representations with spatial, temporal, centroid, peak, and EMD metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["opentouch", "egotactile"],
        choices=["opentouch", "egotactile", "touchanything"],
        help="Datasets to scan/evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ot_raw_heatmap", "ot_discrete_heatmap", "ot_centered_mano", "egotactile_heatmap", "preprocess_gaussian"],
        choices=METHOD_CHOICES,
        help="Representation methods to evaluate. Inapplicable methods are skipped per dataset.",
    )
    parser.add_argument("--dataset_root_opentouch", default="/data1/jiangrui/OpenTouch Data/full_dataset")
    parser.add_argument("--dataset_root_egotactile", default="/data1/jiangrui/EgoTactile/Raw_data/extracted_frames")
    parser.add_argument("--dataset_root_touchanything", default="/data1/jiangrui/EgoTouch/extracted_frames")
    parser.add_argument(
        "--source_mode",
        choices=["auto", "direct", "meta", "manifest"],
        default="auto",
        help="direct reads sequence-level NPZ/HDF5 pressure arrays; meta reads extracted per-frame meta.json files.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest jsonl path(s), comma-separated list, or glob. Used when --source_mode manifest.",
    )
    parser.add_argument("--dataset_raw_root_opentouch", default="/data1/jiangrui/OpenTouch Data/data")
    parser.add_argument("--dataset_raw_root_egotactile", default="/data1/jiangrui/EgoTactile/Raw_data")
    parser.add_argument("--dataset_raw_root_touchanything", default="/data1/jiangrui/EgoTouch")
    parser.add_argument("--egotactile_npz_name", default="pressure_grids_egotactile.npz")
    parser.add_argument("--scan_exclude_dirs", nargs="*", default=None)
    parser.add_argument("--touchanything_scan_depth", type=int, default=3)
    parser.add_argument("--egotactile_scan_depth", type=int, default=4)
    parser.add_argument("--touchanything_scan_split_depth", type=int, default=2)
    parser.add_argument("--egotactile_scan_split_depth", type=int, default=3)
    parser.add_argument("--output_dir", default="outputs/representation_eval")
    parser.add_argument("--repo_root", default=str(REPO_ROOT))
    parser.add_argument("--cache_dir", default=str(ARTIFACT_ROOT / "representation_eval/cache"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check_workers", type=int, default=32)
    parser.add_argument("--gpu", default="", help="Comma-separated GPU ids. Empty means CPU/default torch device.")
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--limit_sequences", type=int, default=0)
    parser.add_argument("--discrete_levels", type=int, default=5)
    parser.add_argument("--emd_mode", choices=["full", "active", "topk"], default="full")
    parser.add_argument("--emd_solver", choices=["sinkhorn_log_gpu", "exact"], default="sinkhorn_log_gpu")
    parser.add_argument("--sinkhorn_iters", type=int, default=100)
    parser.add_argument("--sinkhorn_epsilon", default="auto", help="'auto' or a positive float.")
    parser.add_argument("--emd_batch_frames", type=int, default=64, help="Reserved for future batched EMD kernels.")
    parser.add_argument("--topk_raw", type=int, default=128)
    parser.add_argument("--topk_new", type=int, default=512)
    parser.add_argument("--contact_threshold", type=float, default=1e-8)
    parser.add_argument(
        "--native_heatmap_full_pixels",
        action="store_true",
        help="Reserved flag for native heatmap full-pixel EMD. Current main EMD is common-MANO.",
    )
    parser.add_argument("--no_frame_metrics", action="store_true", help="Do not write frame_metrics.jsonl.")
    parser.add_argument("--max_errors", type=int, default=20)
    return parser.parse_args()


def _parse_epsilon(value):
    if value is None or str(value).lower() == "auto":
        return None
    epsilon = float(value)
    if epsilon <= 0:
        raise ValueError("--sinkhorn_epsilon must be 'auto' or a positive float")
    return epsilon


def _parse_gpus(value):
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _init_worker(args_dict, gpu_queue=None):
    global WORKER_MESH, WORKER_SENSOR_GEOMS, WORKER_ARGS
    from preprocess.representation_eval.geometry import load_mesh_geometry

    WORKER_ARGS = args_dict
    if gpu_queue is not None:
        try:
            gpu_id = gpu_queue.get_nowait()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        except Exception:
            pass
    WORKER_MESH = load_mesh_geometry(Path(args_dict["repo_root"]))
    WORKER_SENSOR_GEOMS = {}


def _sensor_geom_for(dataset, hand):
    from preprocess.representation_eval.geometry import build_sensor_geometry

    key = (dataset, hand)
    if key not in WORKER_SENSOR_GEOMS:
        WORKER_SENSOR_GEOMS[key] = build_sensor_geometry(
            dataset=dataset,
            hand=hand,
            mesh=WORKER_MESH,
            repo_root=Path(WORKER_ARGS["repo_root"]),
            cache_dir=Path(WORKER_ARGS["cache_dir"]),
        )
    return WORKER_SENSOR_GEOMS[key]


def _method_applicable(dataset, method):
    if method.startswith("ot_"):
        return dataset == "opentouch"
    if method == "egotactile_heatmap":
        return dataset == "egotactile"
    if method == "preprocess_gaussian":
        return True
    return False


def _evaluate_sequence_worker(sequence):
    from preprocess.representation_eval.metrics import evaluate_method_sequence
    from preprocess.representation_eval.representations import load_sequence_representations

    try:
        dataset = sequence["dataset"]
        hand = sequence["hand"]
        methods = [m for m in WORKER_ARGS["methods"] if _method_applicable(dataset, m)]
        if not methods:
            return {"sequence_id": sequence["sequence_id"], "frame_rows": [], "sequence_rows": [], "error": None}
        sensor_geom = _sensor_geom_for(dataset, hand)
        reps = load_sequence_representations(
            sequence,
            methods=methods,
            mesh=WORKER_MESH,
            sensor_geom=sensor_geom,
            discrete_levels=WORKER_ARGS["discrete_levels"],
            repo_root=Path(WORKER_ARGS["repo_root"]),
        )
        frame_rows = []
        sequence_rows = []
        for method, frames in reps.items():
            rows, summary = evaluate_method_sequence(
                sequence=sequence,
                method=method,
                frames=frames,
                mesh=WORKER_MESH,
                sensor_geom=sensor_geom,
                emd_mode=WORKER_ARGS["emd_mode"],
                emd_solver=WORKER_ARGS["emd_solver"],
                contact_threshold=WORKER_ARGS["contact_threshold"],
                topk_raw=WORKER_ARGS["topk_raw"],
                topk_new=WORKER_ARGS["topk_new"],
                sinkhorn_iters=WORKER_ARGS["sinkhorn_iters"],
                sinkhorn_epsilon=WORKER_ARGS["sinkhorn_epsilon"],
            )
            if int(summary.get("frames_evaluated") or 0) <= 0:
                continue
            if WORKER_ARGS["write_frame_metrics"]:
                frame_rows.extend(rows)
            sequence_rows.append(summary)
        return {"sequence_id": sequence["sequence_id"], "frame_rows": frame_rows, "sequence_rows": sequence_rows, "error": None}
    except Exception as exc:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8))
        source = sequence.get("source_path") or sequence.get("frames", [{}])[0].get("meta_path")
        if source:
            tb = f"source_path={source}\n{tb}"
        return {"sequence_id": sequence.get("sequence_id", "unknown"), "frame_rows": [], "sequence_rows": [], "error": tb}


def aggregate_summary(sequence_rows):
    grouped = defaultdict(list)
    for row in sequence_rows:
        grouped[(row["dataset"], row["split"], row["hand"], row["method"])].append(row)
    out = []
    numeric_keys = [
        "frames_evaluated",
        "spatial_laplacian_mean",
        "spatial_laplacian_p50",
        "spatial_laplacian_p90",
        "native_spatial_laplacian_mean",
        "temp_1st_mean",
        "temp_2nd_mean",
        "centroid_error_l2_mean",
        "peak_abs_error_mean",
        "peak_overshoot_rate",
        "emd_mean",
        "emd_p50",
        "emd_p90",
    ]
    for (dataset, split, hand, method), rows in sorted(grouped.items()):
        item = {
            "dataset": dataset,
            "split": split,
            "hand": hand,
            "method": method,
            "sequences": len(rows),
            "emd_mode": rows[0].get("emd_mode"),
            "emd_solver": rows[0].get("emd_solver"),
        }
        for key in numeric_keys:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            if key == "frames_evaluated":
                item[key] = int(sum(vals)) if vals else 0
            else:
                item[key] = float(sum(vals) / len(vals)) if vals else None
        out.append(item)
    return out


def write_summary_md(path, summary_rows, errors):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pressure Representation Evaluation", ""]
    if not summary_rows:
        lines.append("No evaluated rows.")
    else:
        columns = [
            "dataset",
            "split",
            "hand",
            "method",
            "sequences",
            "frames_evaluated",
            "spatial_laplacian_mean",
            "temp_1st_mean",
            "temp_2nd_mean",
            "centroid_error_l2_mean",
            "peak_abs_error_mean",
            "peak_overshoot_rate",
            "emd_mean",
        ]
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in summary_rows:
            vals = []
            for col in columns:
                val = row.get(col)
                if isinstance(val, float):
                    vals.append(f"{val:.6g}")
                else:
                    vals.append("" if val is None else str(val))
            lines.append("| " + " | ".join(vals) + " |")
    if errors:
        lines.extend(["", "## First Errors", ""])
        for seq, err in errors[:20]:
            lines.append(f"- `{seq}`: `{err}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_roots = {
        "opentouch": args.dataset_root_opentouch,
        "egotactile": args.dataset_root_egotactile,
        "touchanything": args.dataset_root_touchanything,
    }
    raw_dataset_roots = {
        "opentouch": args.dataset_raw_root_opentouch,
        "egotactile": args.dataset_raw_root_egotactile,
        "touchanything": args.dataset_raw_root_touchanything,
    }
    sequences = []
    discovery_mode = args.source_mode
    if args.source_mode == "manifest":
        if not args.manifest:
            raise ValueError("--manifest is required when --source_mode manifest")
        print("Loading sequence manifests...")
        sequences = read_sequence_manifests(args.manifest, args.datasets)
        discovery_mode = "manifest"

    if args.source_mode in {"auto", "direct"}:
        print("Discovering direct pressure sources...")
        sequences = discover_pressure_sources(
            raw_dataset_roots,
            args.datasets,
            check_workers=args.check_workers,
            egotactile_npz_name=args.egotactile_npz_name,
            scan_exclude_dirs=args.scan_exclude_dirs,
            touchanything_scan_depth=args.touchanything_scan_depth,
            egotactile_scan_depth=args.egotactile_scan_depth,
            touchanything_scan_split_depth=args.touchanything_scan_split_depth,
            egotactile_scan_split_depth=args.egotactile_scan_split_depth,
        )
        print(f"Discovered {len(sequences)} direct pressure sequences.")
        if args.source_mode == "direct" and not sequences:
            raise RuntimeError("No direct pressure sources discovered. Check raw dataset roots and source file names.")
        if sequences:
            discovery_mode = "direct"

    if args.source_mode == "meta" or (args.source_mode == "auto" and not sequences):
        print("Discovering extracted meta sequences...")
        sequences = discover_sequences(dataset_roots, args.datasets, args.check_workers)
        discovery_mode = "meta"
    if args.limit_sequences and args.limit_sequences > 0:
        sequences = sequences[: args.limit_sequences]
    print(f"Using {discovery_mode} source mode with {len(sequences)} sequences.")
    if not sequences:
        raise RuntimeError("No sequences discovered. Check dataset roots and source mode.")

    frame_path = output_dir / "frame_metrics.jsonl"
    seq_path = output_dir / "sequence_metrics.jsonl"
    if frame_path.exists():
        frame_path.unlink()
    if seq_path.exists():
        seq_path.unlink()

    worker_args = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "methods": args.methods,
        "discrete_levels": args.discrete_levels,
        "emd_mode": args.emd_mode,
        "emd_solver": args.emd_solver,
        "contact_threshold": args.contact_threshold,
        "topk_raw": args.topk_raw,
        "topk_new": args.topk_new,
        "sinkhorn_iters": args.sinkhorn_iters,
        "sinkhorn_epsilon": _parse_epsilon(args.sinkhorn_epsilon),
        "write_frame_metrics": not args.no_frame_metrics,
    }

    errors = []
    all_sequence_rows = []
    workers = max(1, int(args.workers))
    gpus = _parse_gpus(args.gpu)
    ctx = multiprocessing.get_context("spawn")
    manager = None
    gpu_queue = None
    max_workers = workers
    if gpus:
        manager = ctx.Manager()
        gpu_queue = manager.Queue()
        max_workers = min(workers, max(1, len(gpus) * max(1, args.workers_per_gpu)))
        for idx in range(max_workers):
            gpu_queue.put(gpus[idx % len(gpus)])

    if max_workers == 1:
        _init_worker(worker_args, None)
        iterator = tqdm(sequences, desc="Evaluating sequences")
        for seq in iterator:
            result = _evaluate_sequence_worker(seq)
            if result["error"]:
                errors.append((result["sequence_id"], result["error"]))
            if result["sequence_rows"]:
                all_sequence_rows.extend(result["sequence_rows"])
                append_jsonl(seq_path, result["sequence_rows"])
            if result["frame_rows"] and not args.no_frame_metrics:
                append_jsonl(frame_path, result["frame_rows"])
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(worker_args, gpu_queue),
        ) as executor:
            futures = [executor.submit(_evaluate_sequence_worker, seq) for seq in sequences]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating sequences"):
                result = future.result()
                if result["error"]:
                    errors.append((result["sequence_id"], result["error"]))
                    if len(errors) <= args.max_errors:
                        print(f"Warning: {result['sequence_id']}: {result['error']}")
                    continue
                if result["sequence_rows"]:
                    all_sequence_rows.extend(result["sequence_rows"])
                    append_jsonl(seq_path, result["sequence_rows"])
                if result["frame_rows"] and not args.no_frame_metrics:
                    append_jsonl(frame_path, result["frame_rows"])

    summary_rows = aggregate_summary(all_sequence_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_summary_md(output_dir / "summary.md", summary_rows, errors)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Pressure representation evaluation finished.")
    print(f"  sequences discovered: {len(sequences)}")
    print(f"  sequence metric rows: {len(all_sequence_rows)}")
    print(f"  summary rows: {len(summary_rows)}")
    print(f"  errors: {len(errors)}")
    print(f"  output_dir: {output_dir}")
    if errors:
        print("  first errors:")
        for seq, err in errors[: args.max_errors]:
            print(f"    {seq}: {err}")


if __name__ == "__main__":
    main()
