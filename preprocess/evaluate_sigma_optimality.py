#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


np = None
WORKER_MESH = None
WORKER_SENSOR_GEOMS = {}
WORKER_ARGS: dict | None = None
WORKER_WEIGHT_CACHE = {}


def _ensure_numpy():
    global np
    if np is None:
        import numpy as _np

        np = _np
    return np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep Dijkstra-Gaussian sigma and evaluate full-sensor LOOCV optimality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["opentouch", "egotactile", "touchanything"],
        choices=["opentouch", "egotactile", "touchanything"],
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
    parser.add_argument("--sigma_values", default=None, help="Comma-separated explicit sigma values. Overrides --alpha_values.")
    parser.add_argument("--alpha_values", default="0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--output_dir", default="outputs/sigma_optimality")
    parser.add_argument("--repo_root", default=str(REPO_ROOT))
    parser.add_argument("--cache_dir", default=str(ARTIFACT_ROOT / "sigma_optimality/cache"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check_workers", type=int, default=32)
    parser.add_argument("--gpu", default="", help="Comma-separated GPU ids. Used to set CUDA_VISIBLE_DEVICES per worker.")
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--limit_sequences", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames_per_sequence", type=int, default=0)
    parser.add_argument("--contact_threshold", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_frame_metrics", action="store_true")
    parser.add_argument("--max_errors", type=int, default=20)
    return parser.parse_args()


def _parse_float_list(value: str | None) -> list[float]:
    if value is None or str(value).strip() == "":
        return []
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if val <= 0:
            raise ValueError(f"Expected positive float, got {item}")
        out.append(val)
    return sorted(set(out))


def _parse_gpus(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _finite_mean(values) -> float | None:
    _ensure_numpy()
    arr = np.asarray([x for x in values if x is not None and math.isfinite(float(x))], dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def _init_worker(args_dict, gpu_queue=None):
    global WORKER_MESH, WORKER_SENSOR_GEOMS, WORKER_ARGS, WORKER_WEIGHT_CACHE
    from preprocess.representation_eval.geometry import load_mesh_geometry

    _ensure_numpy()
    WORKER_ARGS = args_dict
    if gpu_queue is not None:
        try:
            gpu_id = gpu_queue.get_nowait()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        except Exception:
            pass
    WORKER_MESH = load_mesh_geometry(Path(args_dict["repo_root"]))
    WORKER_SENSOR_GEOMS = {}
    WORKER_WEIGHT_CACHE = {}


def _sensor_geom_for(dataset: str, hand: str):
    from preprocess.representation_eval.geometry import build_sensor_geometry

    key = (dataset, hand)
    if key not in WORKER_SENSOR_GEOMS:
        WORKER_SENSOR_GEOMS[key] = build_sensor_geometry(
            dataset=dataset,
            hand=hand,
            mesh=WORKER_MESH,
            repo_root=Path(WORKER_ARGS["repo_root"]),
            cache_dir=Path(WORKER_ARGS["representation_cache_dir"]),
        )
    return WORKER_SENSOR_GEOMS[key]


def _sensor_hash(sensor_geom) -> str:
    _ensure_numpy()
    h = hashlib.sha1()
    h.update(sensor_geom.sensor_ids.astype(np.int32).tobytes())
    h.update(np.asarray(sensor_geom.sensor_to_vertex_cost.shape, dtype=np.int64).tobytes())
    h.update(sensor_geom.sensor_to_vertex_cost.astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def _weights_for(dataset: str, hand: str, sigma: float, sensor_geom):
    _ensure_numpy()
    key = (dataset, hand, float(sigma))
    if key in WORKER_WEIGHT_CACHE:
        return WORKER_WEIGHT_CACHE[key]

    cache_dir = Path(WORKER_ARGS["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = _sensor_hash(sensor_geom)
    sigma_tag = f"{float(sigma):.8g}".replace(".", "p")
    cache_path = cache_dir / f"weights_{dataset}_{hand}_sigma_{sigma_tag}_{digest}.npy"
    if cache_path.exists():
        try:
            weights = np.load(cache_path, mmap_mode=None).astype(np.float32, copy=False)
            WORKER_WEIGHT_CACHE[key] = weights
            return weights
        except Exception:
            try:
                cache_path.unlink()
            except OSError:
                pass

    dist = sensor_geom.sensor_to_vertex_cost.astype(np.float32, copy=False)
    weights = np.exp(-(dist * dist) / max(2.0 * float(sigma) * float(sigma), 1e-12)).astype(np.float32)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp_path, weights)
        os.replace(tmp_path, cache_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    WORKER_WEIGHT_CACHE[key] = weights
    return weights


def _sensor_nearest_neighbor_stats(sensor_geom) -> dict:
    _ensure_numpy()
    n = len(sensor_geom.sensor_ids)
    if n <= 1:
        return {"d_min_mean": None, "d_min_median": None, "d_min_p10": None, "d_min_p90": None}
    pair = np.full((n, n), np.inf, dtype=np.float32)
    for i in range(n):
        row_cost = sensor_geom.sensor_to_vertex_cost[i]
        for j, vids in enumerate(sensor_geom.sensor_vertices):
            if i == j or len(vids) == 0:
                continue
            valid_vids = np.asarray(vids, dtype=np.int32)
            valid_vids = valid_vids[(0 <= valid_vids) & (valid_vids < row_cost.size)]
            if valid_vids.size:
                pair[i, j] = float(np.mean(row_cost[valid_vids]))
    nearest = np.min(pair, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return {"d_min_mean": None, "d_min_median": None, "d_min_p10": None, "d_min_p90": None}
    return {
        "d_min_mean": float(np.mean(nearest)),
        "d_min_median": float(np.median(nearest)),
        "d_min_p10": float(np.percentile(nearest, 10)),
        "d_min_p90": float(np.percentile(nearest, 90)),
    }


def _sigmas_for_geom(sensor_geom) -> tuple[list[float], dict]:
    d_stats = _sensor_nearest_neighbor_stats(sensor_geom)
    explicit = WORKER_ARGS["sigma_values"]
    if explicit:
        return explicit, d_stats
    d_mean = d_stats.get("d_min_mean")
    if d_mean is None or not math.isfinite(float(d_mean)) or d_mean <= 0:
        raise ValueError("Cannot derive sigma from alpha_values because d_min_mean is invalid.")
    sigmas = sorted(set(float(alpha) * float(d_mean) for alpha in WORKER_ARGS["alpha_values"]))
    return sigmas, d_stats


def _target_pressure(raw_rows, weights, num_vertices: int):
    _ensure_numpy()
    if raw_rows.size == 0 or weights.size == 0:
        return np.zeros((num_vertices,), dtype=np.float32)
    contrib = raw_rows[:, None].astype(np.float32) * weights
    return np.clip(np.max(contrib, axis=0), 0.0, 1.0).astype(np.float32)


def _loocv_metrics(raw_rows, weights, sensor_geom, valid_geom_rows, threshold: float) -> dict:
    _ensure_numpy()
    m = raw_rows.size
    if m <= 1:
        return {
            "loocv_mse_all": None,
            "loocv_mae_all": None,
            "loocv_mse_active": None,
            "loocv_mse_inactive": None,
            "loocv_sensor_count": int(m),
            "loocv_active_sensor_count": int(np.sum(raw_rows > threshold)),
            "loocv_inactive_sensor_count": int(np.sum(raw_rows <= threshold)),
        }

    contrib = raw_rows[:, None].astype(np.float32) * weights
    if m == 2:
        max1 = np.max(contrib, axis=0)
        arg1 = np.argmax(contrib, axis=0).astype(np.int32)
        max2 = np.min(contrib, axis=0)
    else:
        part = np.partition(contrib, -2, axis=0)
        max2 = part[-2]
        max1 = part[-1]
        arg1 = np.argmax(contrib, axis=0).astype(np.int32)

    pred = np.zeros((m,), dtype=np.float32)
    for local_i, geom_row in enumerate(valid_geom_rows.tolist()):
        vids = np.asarray(sensor_geom.sensor_vertices[int(geom_row)], dtype=np.int32)
        vids = vids[(0 <= vids) & (vids < max1.size)]
        if vids.size == 0:
            pred[local_i] = 0.0
            continue
        held_values = np.where(arg1[vids] == local_i, max2[vids], max1[vids])
        pred[local_i] = float(np.mean(held_values))

    err = pred - raw_rows
    sq = err * err
    active = raw_rows > threshold
    inactive = ~active
    return {
        "loocv_mse_all": float(np.mean(sq)),
        "loocv_mae_all": float(np.mean(np.abs(err))),
        "loocv_mse_active": float(np.mean(sq[active])) if np.any(active) else None,
        "loocv_mse_inactive": float(np.mean(sq[inactive])) if np.any(inactive) else None,
        "loocv_sensor_count": int(m),
        "loocv_active_sensor_count": int(np.sum(active)),
        "loocv_inactive_sensor_count": int(np.sum(inactive)),
    }


def _entropy(values, vertex_area) -> tuple[float | None, float | None, int]:
    _ensure_numpy()
    mass = np.clip(values.astype(np.float64), 0.0, None) * vertex_area[: values.size].astype(np.float64)
    total = float(np.sum(mass))
    if total <= 1e-12:
        return None, None, 0
    p = mass[mass > 0] / total
    h = float(-np.sum(p * np.log(np.maximum(p, 1e-300))))
    norm = float(h / math.log(p.size)) if p.size > 1 else 0.0
    return h, norm, int(p.size)


def _raw_rows_for_sensor_geom(raw, sensor_geom):
    _ensure_numpy()
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    rows = []
    vals = []
    for row_idx, sid in enumerate(sensor_geom.sensor_ids.tolist()):
        if 0 <= int(sid) < values.size:
            rows.append(row_idx)
            vals.append(float(values[int(sid)]))
    if not rows:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    return np.asarray(rows, dtype=np.int32), np.clip(np.asarray(vals, dtype=np.float32), 0.0, 1.0)


def _ta_grid_mapping(repo_root: Path, hand: str) -> dict[tuple[int, int], int]:
    from preprocess.representation_eval.io import load_json

    path = repo_root / f"TouchAnything/configs/pressure_position_mapping_{hand}.json"
    if not path.exists():
        return {}
    out = {}
    for key, value in load_json(path).items():
        try:
            row, col = [int(x) for x in key.split(",")]
            out[(row, col)] = int(value)
        except Exception:
            continue
    return out


def _ta_grid_to_sensor256(grid, repo_root: Path, hand: str):
    _ensure_numpy()
    if grid is None:
        return None
    arr = np.nan_to_num(np.asarray(grid, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim != 2:
        flat = arr.reshape(-1)
        return flat.astype(np.float32) if flat.size == 256 else None
    mapping = _ta_grid_mapping(repo_root, hand)
    if not mapping:
        return None
    out = np.zeros((256,), dtype=np.float32)
    h, w = arr.shape
    for (row, col), sid in mapping.items():
        if 0 <= row < h and 0 <= col < w and 0 <= sid < out.size:
            out[sid] = float(arr[row, col])
    return out


def _normalize_opentouch_raw(value):
    _ensure_numpy()
    if value is None:
        return None
    arr = np.nan_to_num(np.asarray(value, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    if arr.size == 0:
        return None
    if arr.max(initial=0.0) > 1.5:
        arr = np.clip((3072.0 - arr) / 3072.0, 0.0, 1.0)
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _load_npz_raw_frames(sequence: dict, repo_root: Path):
    _ensure_numpy()
    dataset = sequence["dataset"]
    hand = sequence["hand"]
    path = Path(sequence["source_path"])
    raw_frames = []
    with np.load(path, allow_pickle=False) as data:
        if dataset == "egotactile":
            key = f"{hand}_sensor_256_norm"
            if key not in data:
                return []
            raw_arr = data[key][:]
            valid_arr = data[f"{hand}_sensor_valid"][:] if f"{hand}_sensor_valid" in data else None
            n = int(raw_arr.shape[0]) if raw_arr.ndim >= 2 else 0
            valid_flat = np.asarray(valid_arr).reshape(-1) if valid_arr is not None else None
            for i in range(n):
                if valid_flat is not None and (i >= valid_flat.size or not bool(valid_flat[i])):
                    raw_frames.append(None)
                    continue
                raw = np.nan_to_num(np.asarray(raw_arr[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
                raw_frames.append(np.clip(raw, 0.0, 1.0).astype(np.float32))
        elif dataset == "touchanything":
            key = f"{hand}_pressure_grid"
            if key not in data:
                return []
            raw_arr = data[key][:]
            n = int(raw_arr.shape[0]) if raw_arr.ndim >= 2 else 0
            for i in range(n):
                raw_frames.append(_ta_grid_to_sensor256(raw_arr[i], repo_root, hand))
    return raw_frames


def _load_hdf5_raw_frames(sequence: dict):
    import h5py

    hand = sequence["hand"]
    path = Path(sequence["source_path"])
    demo_name = sequence.get("hdf5_demo")
    raw_frames = []
    with h5py.File(path, "r") as f:
        if not demo_name or "data" not in f or demo_name not in f["data"]:
            return []
        demo = f["data"][demo_name]
        key = f"{hand}_pressure"
        if key not in demo:
            return []
        raw_arr = demo[key][:]
        n = int(raw_arr.shape[0]) if raw_arr.ndim >= 2 else 0
        for i in range(n):
            raw_frames.append(_normalize_opentouch_raw(raw_arr[i]))
    return raw_frames


def _load_raw_frames(sequence: dict):
    repo_root = Path(WORKER_ARGS["repo_root"])
    if sequence.get("source_type") == "npz":
        raw = _load_npz_raw_frames(sequence, repo_root)
    elif sequence.get("source_type") == "hdf5":
        raw = _load_hdf5_raw_frames(sequence)
    else:
        raw = []
    frame_indices = list(range(len(raw)))
    stride = max(1, int(WORKER_ARGS["frame_stride"]))
    max_frames = int(WORKER_ARGS["max_frames_per_sequence"])
    selected_raw = []
    selected_idx = []
    for pos, item in enumerate(raw):
        if pos % stride != 0:
            continue
        selected_raw.append(item)
        selected_idx.append(frame_indices[pos])
        if max_frames > 0 and len(selected_raw) >= max_frames:
            break
    return selected_raw, selected_idx


def _evaluate_sequence_worker(sequence: dict) -> dict:
    try:
        from preprocess.representation_eval.metrics import laplacian_energy

        dataset = sequence["dataset"]
        hand = sequence["hand"]
        sensor_geom = _sensor_geom_for(dataset, hand)
        sigmas, d_stats = _sigmas_for_geom(sensor_geom)
        raw_frames, frame_indices = _load_raw_frames(sequence)
        frame_rows = []
        sequence_rows = []
        valid_frame_count = 0

        for sigma in sigmas:
            weights_all = _weights_for(dataset, hand, sigma, sensor_geom)
            sigma_frame_rows = []
            for raw, frame_idx in zip(raw_frames, frame_indices):
                if raw is None:
                    continue
                valid_geom_rows, raw_rows = _raw_rows_for_sensor_geom(raw, sensor_geom)
                if raw_rows.size <= 1:
                    continue
                weights = weights_all[valid_geom_rows]
                target = _target_pressure(raw_rows, weights, WORKER_MESH.vertices.shape[0])
                loocv = _loocv_metrics(
                    raw_rows=raw_rows,
                    weights=weights,
                    sensor_geom=sensor_geom,
                    valid_geom_rows=valid_geom_rows,
                    threshold=WORKER_ARGS["contact_threshold"],
                )
                entropy, norm_entropy, active_vertices = _entropy(target, WORKER_MESH.vertex_area)
                peak_raw = float(np.max(raw_rows)) if raw_rows.size else 0.0
                peak_new = float(np.max(target)) if target.size else 0.0
                spatial = laplacian_energy(
                    target,
                    WORKER_MESH.adjacency,
                    np.arange(min(target.size, WORKER_MESH.vertices.shape[0]), dtype=np.int32),
                )
                row = {
                    "dataset": dataset,
                    "split": sequence.get("split", "all"),
                    "sequence_id": sequence["sequence_id"],
                    "hand": hand,
                    "frame_idx": int(frame_idx),
                    "sigma": float(sigma),
                    **d_stats,
                    "sigma_over_dmin": float(sigma / d_stats["d_min_mean"]) if d_stats.get("d_min_mean") else None,
                    **loocv,
                    "spatial_entropy": entropy,
                    "normalized_entropy": norm_entropy,
                    "entropy_active_vertices": active_vertices,
                    "spatial_laplacian": spatial,
                    "peak_raw": peak_raw,
                    "peak_new": peak_new,
                    "peak_abs_error": abs(peak_new - peak_raw),
                    "peak_ratio": float(peak_new / max(peak_raw, 1e-12)),
                    "overshoot": bool(peak_new > peak_raw + 1e-4),
                }
                sigma_frame_rows.append(row)
            if sigma_frame_rows:
                valid_frame_count = max(valid_frame_count, len(sigma_frame_rows))
                sequence_rows.append(_summarize_rows(sigma_frame_rows, sequence, sigma, d_stats))
                if WORKER_ARGS["write_frame_metrics"]:
                    frame_rows.extend(sigma_frame_rows)

        return {
            "sequence_id": sequence["sequence_id"],
            "frame_rows": frame_rows,
            "sequence_rows": sequence_rows,
            "frames_evaluated": valid_frame_count,
            "error": None,
        }
    except Exception as exc:
        return {
            "sequence_id": sequence.get("sequence_id", "unknown"),
            "frame_rows": [],
            "sequence_rows": [],
            "frames_evaluated": 0,
            "error": f"{repr(exc)}\n{traceback.format_exc()}",
        }


def _summarize_rows(rows: list[dict], sequence: dict, sigma: float, d_stats: dict) -> dict:
    sensor_count = int(max((r.get("loocv_sensor_count", 0) for r in rows), default=0))
    active_sensor_count = int(round(_finite_mean(r.get("loocv_active_sensor_count") for r in rows) or 0))
    inactive_sensor_count = int(round(_finite_mean(r.get("loocv_inactive_sensor_count") for r in rows) or 0))
    return {
        "dataset": sequence["dataset"],
        "split": sequence.get("split", "all"),
        "sequence_id": sequence["sequence_id"],
        "hand": sequence["hand"],
        "sigma": float(sigma),
        **d_stats,
        "sigma_over_dmin": float(sigma / d_stats["d_min_mean"]) if d_stats.get("d_min_mean") else None,
        "frames_evaluated": len(rows),
        "sensor_count": sensor_count,
        "active_sensor_count_mean": active_sensor_count,
        "inactive_sensor_count_mean": inactive_sensor_count,
        "loocv_mse_all": _finite_mean(r.get("loocv_mse_all") for r in rows),
        "loocv_mae_all": _finite_mean(r.get("loocv_mae_all") for r in rows),
        "loocv_mse_active": _finite_mean(r.get("loocv_mse_active") for r in rows),
        "loocv_mse_inactive": _finite_mean(r.get("loocv_mse_inactive") for r in rows),
        "spatial_entropy": _finite_mean(r.get("spatial_entropy") for r in rows),
        "normalized_entropy": _finite_mean(r.get("normalized_entropy") for r in rows),
        "spatial_laplacian": _finite_mean(r.get("spatial_laplacian") for r in rows),
        "peak_abs_error": _finite_mean(r.get("peak_abs_error") for r in rows),
        "peak_ratio": _finite_mean(r.get("peak_ratio") for r in rows),
        "overshoot_rate": _finite_mean(float(r.get("overshoot", False)) for r in rows),
    }


def aggregate_summary(sequence_rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in sequence_rows:
        grouped[(row["dataset"], row["split"], row["hand"], float(row["sigma"]))].append(row)
    out = []
    metric_keys = [
        "loocv_mse_all",
        "loocv_mae_all",
        "loocv_mse_active",
        "loocv_mse_inactive",
        "spatial_entropy",
        "normalized_entropy",
        "spatial_laplacian",
        "peak_abs_error",
        "peak_ratio",
        "overshoot_rate",
        "sigma_over_dmin",
        "d_min_mean",
        "d_min_median",
        "d_min_p10",
        "d_min_p90",
    ]
    for (dataset, split, hand, sigma), rows in sorted(grouped.items()):
        item = {
            "dataset": dataset,
            "split": split,
            "hand": hand,
            "sigma": sigma,
            "sequences": len(rows),
            "frames_evaluated": int(sum(int(r.get("frames_evaluated", 0)) for r in rows)),
            "sensor_count": int(max((r.get("sensor_count", 0) for r in rows), default=0)),
        }
        for key in metric_keys:
            vals = [r.get(key) for r in rows if r.get(key) is not None]
            item[key] = float(sum(vals) / len(vals)) if vals else None
        out.append(item)

    best_by_group = {}
    for row in out:
        key = (row["dataset"], row["split"], row["hand"])
        val = row.get("loocv_mse_all")
        if val is None:
            continue
        if key not in best_by_group or val < best_by_group[key]["loocv_mse_all"]:
            best_by_group[key] = row
    for row in out:
        key = (row["dataset"], row["split"], row["hand"])
        best = best_by_group.get(key)
        row["is_best_sigma"] = bool(best is row)
        row["best_sigma_by_loocv_mse_all"] = best["sigma"] if best else None
    return out


def write_summary_md(path: Path, summary_rows: list[dict], errors: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Sigma Optimality Evaluation", ""]
    if not summary_rows:
        lines.append("No evaluated rows.")
    else:
        best = [r for r in summary_rows if r.get("is_best_sigma")]
        lines.extend(["## Best Sigma By Full-Sensor LOOCV", ""])
        cols = ["dataset", "split", "hand", "sigma", "loocv_mse_all", "loocv_mse_active", "loocv_mse_inactive", "sigma_over_dmin"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in best:
            lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")

        lines.extend(["", "## All Results", ""])
        cols = [
            "dataset",
            "split",
            "hand",
            "sigma",
            "sequences",
            "frames_evaluated",
            "loocv_mse_all",
            "loocv_mse_active",
            "loocv_mse_inactive",
            "normalized_entropy",
            "peak_abs_error",
            "spatial_laplacian",
            "is_best_sigma",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in summary_rows:
            lines.append("| " + " | ".join(_fmt(row.get(c)) for c in cols) + " |")
    if errors:
        lines.extend(["", "## First Errors", ""])
        for seq, err in errors[:20]:
            first = str(err).splitlines()[0]
            lines.append(f"- `{seq}`: `{first}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def plot_curves(path: Path, summary_rows: list[dict]) -> None:
    if not summary_rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Warning: matplotlib unavailable; skipping curves: {exc}")
        return

    groups = defaultdict(list)
    for row in summary_rows:
        groups[(row["dataset"], row["hand"])].append(row)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("loocv_mse_all", "Full-sensor LOOCV MSE"),
        ("normalized_entropy", "Normalized Entropy"),
        ("peak_abs_error", "Peak Absolute Error"),
        ("spatial_laplacian", "Spatial Laplacian"),
    ]
    for ax, (metric, title) in zip(axes.reshape(-1), metrics):
        for (dataset, hand), rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda x: x["sigma"])
            xs = [r["sigma"] for r in rows if r.get(metric) is not None]
            ys = [r[metric] for r in rows if r.get(metric) is not None]
            if xs and ys:
                ax.plot(xs, ys, marker="o", label=f"{dataset}-{hand}")
        ax.set_title(title)
        ax.set_xlabel("sigma")
        ax.grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    from preprocess.representation_eval.io import append_jsonl, discover_pressure_sources, write_csv

    _ensure_numpy()
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_roots = {
        "opentouch": args.dataset_raw_root_opentouch,
        "egotactile": args.dataset_raw_root_egotactile,
        "touchanything": args.dataset_raw_root_touchanything,
    }
    print("Discovering direct pressure sources...")
    sequences = discover_pressure_sources(
        raw_roots,
        args.datasets,
        check_workers=args.check_workers,
        egotactile_npz_name=args.egotactile_npz_name,
        scan_exclude_dirs=args.scan_exclude_dirs,
        touchanything_scan_depth=args.touchanything_scan_depth,
        egotactile_scan_depth=args.egotactile_scan_depth,
        touchanything_scan_split_depth=args.touchanything_scan_split_depth,
        egotactile_scan_split_depth=args.egotactile_scan_split_depth,
    )
    if args.limit_sequences and args.limit_sequences > 0:
        sequences = sequences[: args.limit_sequences]
    print(f"Discovered {len(sequences)} direct pressure sequences.")
    if not sequences:
        raise RuntimeError("No direct pressure sources discovered. Check raw roots.")

    frame_path = output_dir / "sigma_frame_metrics.jsonl"
    seq_path = output_dir / "sigma_sequence_metrics.jsonl"
    for path in (frame_path, seq_path):
        if path.exists():
            path.unlink()

    worker_args = {
        "repo_root": str(Path(args.repo_root).resolve()),
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "representation_cache_dir": str(ARTIFACT_ROOT / "representation_eval/cache"),
        "sigma_values": _parse_float_list(args.sigma_values),
        "alpha_values": _parse_float_list(args.alpha_values),
        "frame_stride": max(1, int(args.frame_stride)),
        "max_frames_per_sequence": max(0, int(args.max_frames_per_sequence)),
        "contact_threshold": float(args.contact_threshold),
        "write_frame_metrics": not args.no_frame_metrics,
    }
    if not worker_args["sigma_values"] and not worker_args["alpha_values"]:
        raise ValueError("Provide --sigma_values or --alpha_values.")

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
        max_workers = min(workers, max(1, len(gpus) * max(1, int(args.workers_per_gpu))))
        for idx in range(max_workers):
            gpu_queue.put(gpus[idx % len(gpus)])

    if max_workers == 1:
        _init_worker(worker_args, None)
        iterator = tqdm(sequences, desc="Evaluating sigma")
        for seq in iterator:
            result = _evaluate_sequence_worker(seq)
            if result["error"]:
                errors.append((result["sequence_id"], result["error"]))
            if result["sequence_rows"]:
                all_sequence_rows.extend(result["sequence_rows"])
                append_jsonl(seq_path, result["sequence_rows"])
            if result["frame_rows"]:
                append_jsonl(frame_path, result["frame_rows"])
    else:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(worker_args, gpu_queue),
        ) as executor:
            futures = [executor.submit(_evaluate_sequence_worker, seq) for seq in sequences]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating sigma"):
                result = future.result()
                if result["error"]:
                    errors.append((result["sequence_id"], result["error"]))
                    if len(errors) <= args.max_errors:
                        print(f"Warning: {result['sequence_id']}: {str(result['error']).splitlines()[0]}")
                    continue
                if result["sequence_rows"]:
                    all_sequence_rows.extend(result["sequence_rows"])
                    append_jsonl(seq_path, result["sequence_rows"])
                if result["frame_rows"]:
                    append_jsonl(frame_path, result["frame_rows"])

    summary_rows = aggregate_summary(all_sequence_rows)
    write_csv(output_dir / "sigma_summary.csv", summary_rows)
    write_summary_md(output_dir / "sigma_summary.md", summary_rows, errors)
    plot_curves(output_dir / "sigma_curves.png", summary_rows)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Sigma optimality evaluation finished.")
    print(f"  sequences discovered: {len(sequences)}")
    print(f"  sequence metric rows: {len(all_sequence_rows)}")
    print(f"  summary rows: {len(summary_rows)}")
    print(f"  errors: {len(errors)}")
    print(f"  output_dir: {output_dir}")
    if errors:
        print("  first errors:")
        for seq, err in errors[: args.max_errors]:
            print(f"    {seq}: {str(err).splitlines()[0]}")


if __name__ == "__main__":
    main()
