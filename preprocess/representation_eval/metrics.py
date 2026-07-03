from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .geometry import MeshGeometry, SensorGeometry
from .representations import FrameRepresentation


def _finite_mean(values: Iterable[float]) -> float | None:
    arr = np.asarray([x for x in values if x is not None and math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _percentile(values: Iterable[float], q: float) -> float | None:
    arr = np.asarray([x for x in values if x is not None and math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def laplacian_energy(values: np.ndarray, adjacency: list[np.ndarray], valid_vertices: np.ndarray | None = None) -> float:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if valid_vertices is None:
        valid_vertices = np.arange(x.size, dtype=np.int32)
    vals = []
    for i in valid_vertices.astype(np.int32).tolist():
        if i >= x.size:
            continue
        nbrs = adjacency[i]
        nbrs = nbrs[nbrs < x.size]
        if nbrs.size == 0:
            continue
        diff = float(x[i] - x[nbrs].mean())
        vals.append(diff * diff)
    return float(np.mean(vals)) if vals else 0.0


def grid_laplacian_energy(values: np.ndarray, mask: np.ndarray | None) -> float:
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2:
        return 0.0
    if mask is None:
        mask = np.ones_like(x, dtype=bool)
    vals = []
    h, w = x.shape
    for y in range(h):
        for z in range(w):
            if not mask[y, z]:
                continue
            nbr = []
            for yy, xx in ((y - 1, z), (y + 1, z), (y, z - 1), (y, z + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    nbr.append(x[yy, xx])
            if nbr:
                diff = float(x[y, z] - np.mean(nbr))
                vals.append(diff * diff)
    return float(np.mean(vals)) if vals else 0.0


def support_distribution(values: np.ndarray, mode: str, threshold: float, topk: int | None = None):
    x = np.nan_to_num(np.asarray(values, dtype=np.float64).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)
    if mode == "active":
        idx = np.where(x > threshold)[0]
    elif mode == "topk":
        k = min(int(topk or 512), x.size)
        idx = np.argpartition(x, -k)[-k:] if k > 0 else np.zeros(0, dtype=np.int64)
        idx = idx[x[idx] > threshold]
    else:
        idx = np.where(x > 0)[0]
    if idx.size == 0:
        return idx.astype(np.int32), np.zeros((0,), dtype=np.float64)
    mass = x[idx]
    s = float(mass.sum())
    if s <= 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float64)
    return idx.astype(np.int32), (mass / s).astype(np.float64)


def sinkhorn_distance(a: np.ndarray, b: np.ndarray, cost: np.ndarray, epsilon: float | None, iters: int) -> tuple[float | None, float]:
    if a.size == 0 or b.size == 0:
        return None, float(epsilon or 0.0)
    c = np.asarray(cost, dtype=np.float64)
    finite = c[np.isfinite(c) & (c > 0)]
    eps = float(epsilon) if epsilon and epsilon > 0 else float(np.median(finite) * 0.02 if finite.size else 1e-2)
    eps = max(eps, 1e-8)
    try:
        import torch

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_a = torch.log(torch.tensor(a, dtype=torch.float64, device=device).clamp_min(1e-300))
        log_b = torch.log(torch.tensor(b, dtype=torch.float64, device=device).clamp_min(1e-300))
        ct = torch.tensor(c, dtype=torch.float64, device=device)
        log_k = -ct / eps
        u = torch.zeros_like(log_a)
        v = torch.zeros_like(log_b)
        for _ in range(max(1, int(iters))):
            u = log_a - torch.logsumexp(log_k + v[None, :], dim=1)
            v = log_b - torch.logsumexp(log_k + u[:, None], dim=0)
        pi = torch.exp(log_k + u[:, None] + v[None, :])
        return float(torch.sum(pi * ct).detach().cpu().item()), eps
    except Exception:
        k = np.exp(-c / eps)
        u = np.ones_like(a)
        v = np.ones_like(b)
        for _ in range(max(1, int(iters))):
            u = a / np.maximum(k @ v, 1e-300)
            v = b / np.maximum(k.T @ u, 1e-300)
        pi = (u[:, None] * k) * v[None, :]
        return float(np.sum(pi * c)), eps


def exact_emd(a: np.ndarray, b: np.ndarray, cost: np.ndarray, max_vars: int = 2500) -> float | None:
    if a.size * b.size > max_vars:
        return None
    try:
        from scipy.optimize import linprog

        m, n = cost.shape
        c = cost.reshape(-1)
        a_eq = []
        b_eq = []
        for i in range(m):
            row = np.zeros((m, n), dtype=np.float64)
            row[i, :] = 1.0
            a_eq.append(row.reshape(-1))
            b_eq.append(a[i])
        for j in range(n):
            row = np.zeros((m, n), dtype=np.float64)
            row[:, j] = 1.0
            a_eq.append(row.reshape(-1))
            b_eq.append(b[j])
        res = linprog(c, A_eq=np.vstack(a_eq), b_eq=np.asarray(b_eq), bounds=(0, None), method="highs")
        if not res.success:
            return None
        return float(res.fun)
    except Exception:
        return None


def centroid_error(raw: np.ndarray, target: np.ndarray, sensor_geom: SensorGeometry, mesh: MeshGeometry) -> float | None:
    raw_idx, raw_mass = support_distribution(raw, "full", 0.0)
    new_idx, new_mass = support_distribution(target * mesh.vertex_area[: target.size], "full", 0.0)
    if raw_idx.size == 0 or new_idx.size == 0:
        return None
    sid_to_row = {int(sid): i for i, sid in enumerate(sensor_geom.sensor_ids.tolist())}
    rows = [sid_to_row.get(int(i)) for i in raw_idx.tolist()]
    keep = np.asarray([r is not None for r in rows], dtype=bool)
    if not keep.any():
        return None
    coords_raw = sensor_geom.sensor_centers[np.asarray([r for r in rows if r is not None], dtype=np.int32)]
    raw_mass = raw_mass[keep]
    raw_mass = raw_mass / max(raw_mass.sum(), 1e-12)
    c_raw = np.sum(coords_raw * raw_mass[:, None], axis=0)
    coords_new = mesh.vertices[new_idx]
    c_new = np.sum(coords_new * new_mass[:, None], axis=0)
    return float(np.linalg.norm(c_raw - c_new))


def emd_for_frame(
    raw: np.ndarray,
    target: np.ndarray,
    sensor_geom: SensorGeometry,
    mesh: MeshGeometry,
    mode: str,
    solver: str,
    threshold: float,
    topk_raw: int,
    topk_new: int,
    sinkhorn_iters: int,
    sinkhorn_epsilon: float | None,
) -> dict:
    raw_idx, raw_mass = support_distribution(raw, mode, threshold, topk=topk_raw)
    target_mass_values = target * mesh.vertex_area[: target.size]
    new_idx, new_mass = support_distribution(target_mass_values, mode, threshold, topk=topk_new)
    sid_to_row = {int(sid): i for i, sid in enumerate(sensor_geom.sensor_ids.tolist())}
    raw_rows = np.asarray([sid_to_row.get(int(i), -1) for i in raw_idx.tolist()], dtype=np.int32)
    keep = raw_rows >= 0
    raw_rows = raw_rows[keep]
    raw_mass = raw_mass[keep]
    if raw_mass.size:
        raw_mass = raw_mass / raw_mass.sum()
    result = {
        "emd": None,
        "emd_mode": mode,
        "emd_solver": solver,
        "num_raw_support": int(raw_mass.size),
        "num_new_support": int(new_mass.size),
        "cost_matrix_shape": [int(raw_mass.size), int(new_mass.size)],
        "sinkhorn_epsilon": None,
        "sinkhorn_iters": int(sinkhorn_iters),
    }
    if raw_mass.size == 0 or new_mass.size == 0:
        return result
    cost = sensor_geom.sensor_to_vertex_cost[raw_rows][:, new_idx]
    if solver == "exact":
        value = exact_emd(raw_mass, new_mass, cost)
        if value is not None:
            result["emd"] = value
            return result
    value, eps = sinkhorn_distance(raw_mass, new_mass, cost, sinkhorn_epsilon, sinkhorn_iters)
    result["emd"] = value
    result["emd_solver"] = "sinkhorn_log_gpu"
    result["sinkhorn_epsilon"] = eps
    return result


def evaluate_method_sequence(
    sequence: dict,
    method: str,
    frames: Sequence[FrameRepresentation],
    mesh: MeshGeometry,
    sensor_geom: SensorGeometry,
    emd_mode: str,
    emd_solver: str,
    contact_threshold: float,
    topk_raw: int,
    topk_new: int,
    sinkhorn_iters: int,
    sinkhorn_epsilon: float | None,
) -> tuple[list[dict], dict]:
    frame_rows = []
    targets = []
    raw_values = []
    for i, rep in enumerate(frames):
        raw = rep.raw_sensor
        target = rep.target
        if raw is None or target is None:
            continue
        raw = np.clip(np.asarray(raw, dtype=np.float32).reshape(-1), 0.0, 1.0)
        target = np.clip(np.asarray(target, dtype=np.float32).reshape(-1), 0.0, 1.0)
        if raw.sum() <= 1e-8 and target.sum() <= 1e-8:
            continue
        spatial = laplacian_energy(target, mesh.adjacency, np.arange(min(target.size, mesh.vertices.shape[0]), dtype=np.int32))
        native_spatial = None
        if rep.native_target is not None:
            native_spatial = grid_laplacian_energy(rep.native_target, rep.native_mask)
        centroid = centroid_error(raw, target, sensor_geom, mesh)
        peak_raw = float(np.max(raw)) if raw.size else 0.0
        peak_new = float(np.max(target)) if target.size else 0.0
        emd = emd_for_frame(
            raw,
            target,
            sensor_geom,
            mesh,
            mode=emd_mode,
            solver=emd_solver,
            threshold=contact_threshold,
            topk_raw=topk_raw,
            topk_new=topk_new,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_epsilon=sinkhorn_epsilon,
        )
        row = {
            "dataset": sequence["dataset"],
            "split": sequence["split"],
            "sequence_id": sequence["sequence_id"],
            "hand": sequence["hand"],
            "method": method,
            "frame_idx": int(sequence["frames"][i]["frame_idx"]),
            "spatial_laplacian": spatial,
            "native_spatial_laplacian": native_spatial,
            "centroid_error_l2": centroid,
            "peak_raw": peak_raw,
            "peak_new": peak_new,
            "peak_error": peak_new - peak_raw,
            "peak_abs_error": abs(peak_new - peak_raw),
            "peak_overshoot": bool(peak_new > peak_raw + 1e-4),
            **emd,
        }
        frame_rows.append(row)
        targets.append(target)
        raw_values.append(raw)

    if targets:
        tmat = np.stack(targets, axis=0)
        temp1 = np.mean(np.diff(tmat, axis=0) ** 2, axis=1) if tmat.shape[0] > 1 else np.asarray([], dtype=np.float32)
        temp2 = (
            np.mean((tmat[2:] - 2 * tmat[1:-1] + tmat[:-2]) ** 2, axis=1)
            if tmat.shape[0] > 2
            else np.asarray([], dtype=np.float32)
        )
    else:
        temp1 = temp2 = np.asarray([], dtype=np.float32)

    summary = {
        "dataset": sequence["dataset"],
        "split": sequence["split"],
        "sequence_id": sequence["sequence_id"],
        "hand": sequence["hand"],
        "method": method,
        "frames_evaluated": len(frame_rows),
        "spatial_laplacian_mean": _finite_mean(r["spatial_laplacian"] for r in frame_rows),
        "spatial_laplacian_p50": _percentile((r["spatial_laplacian"] for r in frame_rows), 50),
        "spatial_laplacian_p90": _percentile((r["spatial_laplacian"] for r in frame_rows), 90),
        "native_spatial_laplacian_mean": _finite_mean(r["native_spatial_laplacian"] for r in frame_rows),
        "temp_1st_mean": float(temp1.mean()) if temp1.size else None,
        "temp_2nd_mean": float(temp2.mean()) if temp2.size else None,
        "centroid_error_l2_mean": _finite_mean(r["centroid_error_l2"] for r in frame_rows),
        "peak_abs_error_mean": _finite_mean(r["peak_abs_error"] for r in frame_rows),
        "peak_overshoot_rate": _finite_mean(float(r["peak_overshoot"]) for r in frame_rows),
        "emd_mean": _finite_mean(r["emd"] for r in frame_rows),
        "emd_p50": _percentile((r["emd"] for r in frame_rows), 50),
        "emd_p90": _percentile((r["emd"] for r in frame_rows), 90),
        "emd_mode": emd_mode,
        "emd_solver": emd_solver,
    }
    return frame_rows, summary

