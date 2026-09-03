from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .geometry import MeshGeometry, SensorGeometry
from .representations import FrameRepresentation


_DIRECT_OPERATOR_CACHE = {}
_DENSE_OPERATOR_CACHE = {}


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
    if solver == "none":
        return result
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


def _evaluate_method_sequence_generic(
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
    include_frame_rows: bool = True,
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


def _dense_laplacian_operator(mesh: MeshGeometry):
    key = id(mesh)
    cached = _DENSE_OPERATOR_CACHE.get(key)
    if cached is not None:
        return cached
    from scipy.sparse import coo_matrix, eye

    vertex_count = int(mesh.vertices.shape[0])
    rows = []
    columns = []
    values = []
    for vertex_id, neighbors in enumerate(mesh.adjacency):
        valid = np.asarray(neighbors, dtype=np.int64)
        valid = valid[(valid >= 0) & (valid < vertex_count)]
        if valid.size == 0:
            continue
        rows.extend([vertex_id] * int(valid.size))
        columns.extend(valid.tolist())
        values.extend([-1.0 / float(valid.size)] * int(valid.size))
    operator = coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(columns, dtype=np.int32)),
        ),
        shape=(vertex_count, vertex_count),
        dtype=np.float32,
    ).tocsr()
    operator = operator + eye(vertex_count, dtype=np.float32, format="csr")
    _DENSE_OPERATOR_CACHE[key] = operator
    return operator


def _evaluate_ema_gaussian_sequence(
    sequence: dict,
    method: str,
    frames: Sequence[FrameRepresentation],
    mesh: MeshGeometry,
    sensor_geom: SensorGeometry,
    emd_mode: str,
    emd_solver: str,
    include_frame_rows: bool,
) -> tuple[list[dict], dict]:
    if emd_solver != "none":
        raise ValueError(
            f"{method} uses the batched structural metric path; set --emd_solver none"
        )

    indexed = []
    for frame_index, frame in enumerate(frames):
        if frame.raw_sensor is None or frame.target is None:
            continue
        raw = np.clip(
            np.nan_to_num(np.asarray(frame.raw_sensor, dtype=np.float32).reshape(-1)),
            0.0,
            1.0,
        )
        target = np.clip(
            np.nan_to_num(np.asarray(frame.target, dtype=np.float32).reshape(-1)),
            0.0,
            1.0,
        )
        if raw.sum() <= 1e-8 and target.sum() <= 1e-8:
            continue
        indexed.append((frame_index, raw, target))

    if indexed:
        raw_values = np.stack([item[1] for item in indexed], axis=0)
        targets = np.stack([item[2] for item in indexed], axis=0)
        source_indices = np.asarray([item[0] for item in indexed], dtype=np.int64)
    else:
        raw_values = np.zeros((0, 0), dtype=np.float32)
        targets = np.zeros((0, mesh.vertices.shape[0]), dtype=np.float32)
        source_indices = np.zeros((0,), dtype=np.int64)

    spatial = np.zeros((len(targets),), dtype=np.float64)
    laplacian = _dense_laplacian_operator(mesh)
    for start in range(0, len(targets), 256):
        stop = min(start + 256, len(targets))
        differences = (laplacian @ targets[start:stop].T).T
        spatial[start:stop] = np.mean(
            np.asarray(differences, dtype=np.float64) ** 2, axis=1
        )

    if len(targets):
        peak_raw = raw_values.max(axis=1)
        peak_new = targets.max(axis=1)
        if len(targets) > 1:
            temp1 = np.mean(np.diff(targets, axis=0) ** 2, axis=1)
        else:
            temp1 = np.zeros((0,), dtype=np.float32)
        if len(targets) > 2:
            temp2 = np.mean(
                (targets[2:] - 2.0 * targets[1:-1] + targets[:-2]) ** 2,
                axis=1,
            )
        else:
            temp2 = np.zeros((0,), dtype=np.float32)

        valid_rows = np.asarray(
            [
                row_index
                for row_index, sensor_id in enumerate(sensor_geom.sensor_ids.tolist())
                if 0 <= int(sensor_id) < raw_values.shape[1]
            ],
            dtype=np.int64,
        )
        valid_ids = np.asarray(
            [
                int(sensor_id)
                for sensor_id in sensor_geom.sensor_ids.tolist()
                if 0 <= int(sensor_id) < raw_values.shape[1]
            ],
            dtype=np.int64,
        )
        source_values = raw_values[:, valid_ids].astype(np.float64)
        source_mass = source_values.sum(axis=1)
        source_centroid = source_values @ sensor_geom.sensor_centers[valid_rows]
        source_centroid /= np.maximum(source_mass[:, None], 1e-12)

        vertex_area = mesh.vertex_area[: targets.shape[1]].astype(np.float64)
        area_coordinates = (
            vertex_area[:, None]
            * mesh.vertices[: targets.shape[1]].astype(np.float64)
        )
        target_mass = np.zeros((len(targets),), dtype=np.float64)
        target_centroid = np.zeros((len(targets), 3), dtype=np.float64)
        for start in range(0, len(targets), 256):
            stop = min(start + 256, len(targets))
            target_chunk = targets[start:stop].astype(np.float64)
            target_mass[start:stop] = target_chunk @ vertex_area
            target_centroid[start:stop] = target_chunk @ area_coordinates
        target_centroid /= np.maximum(target_mass[:, None], 1e-12)
        centroid = np.linalg.norm(source_centroid - target_centroid, axis=1)
        centroid[(source_mass <= 1e-12) | (target_mass <= 1e-12)] = np.nan
    else:
        peak_raw = peak_new = centroid = np.zeros((0,), dtype=np.float64)
        temp1 = temp2 = np.zeros((0,), dtype=np.float32)

    peak_abs_error = np.abs(peak_new - peak_raw)
    peak_overshoot = peak_new > peak_raw + 1e-4
    frame_rows = []
    if include_frame_rows:
        for row_index, source_index in enumerate(source_indices.tolist()):
            frame_rows.append(
                {
                    "dataset": sequence["dataset"],
                    "split": sequence["split"],
                    "sequence_id": sequence["sequence_id"],
                    "hand": sequence["hand"],
                    "method": method,
                    "frame_idx": int(sequence["frames"][source_index]["frame_idx"]),
                    "spatial_laplacian": float(spatial[row_index]),
                    "native_spatial_laplacian": None,
                    "centroid_error_l2": (
                        float(centroid[row_index])
                        if math.isfinite(float(centroid[row_index]))
                        else None
                    ),
                    "peak_raw": float(peak_raw[row_index]),
                    "peak_new": float(peak_new[row_index]),
                    "peak_error": float(peak_new[row_index] - peak_raw[row_index]),
                    "peak_abs_error": float(peak_abs_error[row_index]),
                    "peak_overshoot": bool(peak_overshoot[row_index]),
                    "emd": None,
                    "emd_mode": emd_mode,
                    "emd_solver": emd_solver,
                    "num_raw_support": None,
                    "num_new_support": None,
                    "cost_matrix_shape": None,
                    "sinkhorn_epsilon": None,
                    "sinkhorn_iters": 0,
                }
            )
    summary = {
        "dataset": sequence["dataset"],
        "split": sequence["split"],
        "sequence_id": sequence["sequence_id"],
        "hand": sequence["hand"],
        "method": method,
        "frames_evaluated": int(len(targets)),
        "spatial_laplacian_mean": _finite_mean(spatial.tolist()),
        "spatial_laplacian_p50": _percentile(spatial.tolist(), 50),
        "spatial_laplacian_p90": _percentile(spatial.tolist(), 90),
        "native_spatial_laplacian_mean": None,
        "temp_1st_mean": _finite_mean(temp1.tolist()),
        "temp_2nd_mean": _finite_mean(temp2.tolist()),
        "centroid_error_l2_mean": _finite_mean(centroid.tolist()),
        "peak_abs_error_mean": _finite_mean(peak_abs_error.tolist()),
        "peak_overshoot_rate": _finite_mean(
            peak_overshoot.astype(np.float64).tolist()
        ),
        "emd_mean": None,
        "emd_p50": None,
        "emd_p90": None,
        "emd_mode": emd_mode,
        "emd_solver": emd_solver,
    }
    return frame_rows, summary


def _direct_operators(
    sensor_geom: SensorGeometry,
    mesh: MeshGeometry,
    raw_dim: int,
):
    key = (id(sensor_geom), id(mesh), int(raw_dim))
    cached = _DIRECT_OPERATOR_CACHE.get(key)
    if cached is not None:
        return cached

    from scipy.sparse import coo_matrix, diags, eye

    vertex_count = int(mesh.vertices.shape[0])
    rows = []
    cols = []
    for sensor_id, vertices in zip(sensor_geom.sensor_ids.tolist(), sensor_geom.sensor_vertices):
        sensor_id = int(sensor_id)
        if not 0 <= sensor_id < raw_dim:
            continue
        valid_vertices = np.asarray(vertices, dtype=np.int64)
        valid_vertices = valid_vertices[
            (valid_vertices >= 0) & (valid_vertices < vertex_count)
        ]
        rows.extend([sensor_id] * int(valid_vertices.size))
        cols.extend(valid_vertices.tolist())
    projection = coo_matrix(
        (
            np.ones((len(rows),), dtype=np.float64),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32)),
        ),
        shape=(raw_dim, vertex_count),
        dtype=np.float64,
    ).tocsr()
    vertex_counts = np.asarray(projection.sum(axis=0)).reshape(-1)
    inverse_counts = np.zeros_like(vertex_counts)
    assigned = vertex_counts > 0
    inverse_counts[assigned] = 1.0 / vertex_counts[assigned]
    projection = (projection @ diags(inverse_counts)).tocsr()

    adjacency_rows = []
    adjacency_cols = []
    adjacency_values = []
    for vertex_id, neighbors in enumerate(mesh.adjacency):
        valid_neighbors = np.asarray(neighbors, dtype=np.int64)
        valid_neighbors = valid_neighbors[
            (valid_neighbors >= 0) & (valid_neighbors < vertex_count)
        ]
        if valid_neighbors.size == 0:
            continue
        adjacency_rows.extend([vertex_id] * int(valid_neighbors.size))
        adjacency_cols.extend(valid_neighbors.tolist())
        adjacency_values.extend(
            [1.0 / float(valid_neighbors.size)] * int(valid_neighbors.size)
        )
    neighbor_mean = coo_matrix(
        (
            np.asarray(adjacency_values, dtype=np.float64),
            (
                np.asarray(adjacency_rows, dtype=np.int32),
                np.asarray(adjacency_cols, dtype=np.int32),
            ),
        ),
        shape=(vertex_count, vertex_count),
        dtype=np.float64,
    ).tocsr()
    laplacian = eye(vertex_count, dtype=np.float64, format="csr") - neighbor_mean
    projected_laplacian = (projection @ laplacian.T).tocsr()
    spatial_quadratic = (
        projected_laplacian @ projected_laplacian.T
    ).tocsr() / float(vertex_count)
    temporal_quadratic = (
        projection @ projection.T
    ).tocsr() / float(vertex_count)
    vertex_area = np.asarray(mesh.vertex_area[:vertex_count], dtype=np.float64)
    target_mass = np.asarray(projection @ vertex_area).reshape(-1)
    target_coordinates = np.asarray(
        projection @ (vertex_area[:, None] * mesh.vertices[:vertex_count])
    )
    target_sum = np.asarray(
        projection @ np.ones((vertex_count,), dtype=np.float64)
    ).reshape(-1)
    cached = {
        "projection": projection,
        "spatial_quadratic": spatial_quadratic,
        "temporal_quadratic": temporal_quadratic,
        "target_mass": target_mass,
        "target_coordinates": target_coordinates,
        "target_sum": target_sum,
    }
    _DIRECT_OPERATOR_CACHE[key] = cached
    return cached


def _quadratic_values(values: np.ndarray, operator) -> np.ndarray:
    return np.sum(np.asarray(values @ operator) * values, axis=1)


def _direct_target_peak(
    raw_values: np.ndarray,
    projection,
    chunk_size: int = 1024,
) -> np.ndarray:
    peaks = np.zeros((raw_values.shape[0],), dtype=np.float64)
    for start in range(0, raw_values.shape[0], chunk_size):
        stop = min(start + chunk_size, raw_values.shape[0])
        target = np.asarray(raw_values[start:stop] @ projection)
        peaks[start:stop] = target.max(axis=1) if target.shape[1] else 0.0
    return peaks


def _evaluate_direct_method_sequence(
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
    include_frame_rows: bool,
) -> tuple[list[dict], dict]:
    del contact_threshold, topk_raw, topk_new, sinkhorn_iters, sinkhorn_epsilon
    indexed_raw = []
    raw_dim = 0
    for frame_index, frame in enumerate(frames):
        if frame.raw_sensor is None:
            continue
        raw = np.clip(
            np.asarray(frame.raw_sensor, dtype=np.float64).reshape(-1), 0.0, 1.0
        )
        raw_dim = max(raw_dim, int(raw.size))
        indexed_raw.append((frame_index, raw))
    if not indexed_raw:
        raw_values = np.zeros((0, 0), dtype=np.float64)
    else:
        raw_values = np.zeros((len(indexed_raw), raw_dim), dtype=np.float64)
        for row_index, (_, raw) in enumerate(indexed_raw):
            raw_values[row_index, : raw.size] = raw

    operators = _direct_operators(sensor_geom, mesh, raw_dim) if raw_dim else None
    if raw_values.size:
        raw_sum = raw_values.sum(axis=1)
        direct_sum = raw_values @ operators["target_sum"]
        eligible = (raw_sum > 1e-8) | (direct_sum > 1e-8)
        raw_values = raw_values[eligible]
        source_indices = np.asarray(
            [frame_index for frame_index, _ in indexed_raw], dtype=np.int64
        )[eligible]
    else:
        source_indices = np.zeros((0,), dtype=np.int64)

    if raw_values.size:
        spatial = np.maximum(
            _quadratic_values(raw_values, operators["spatial_quadratic"]), 0.0
        )
        peak_raw = raw_values.max(axis=1)
        peak_new = _direct_target_peak(raw_values, operators["projection"])

        valid_sensor_rows = np.asarray(
            [
                row
                for row, sensor_id in enumerate(sensor_geom.sensor_ids.tolist())
                if 0 <= int(sensor_id) < raw_dim
            ],
            dtype=np.int64,
        )
        valid_sensor_ids = np.asarray(
            [
                int(sensor_id)
                for sensor_id in sensor_geom.sensor_ids.tolist()
                if 0 <= int(sensor_id) < raw_dim
            ],
            dtype=np.int64,
        )
        source_values = raw_values[:, valid_sensor_ids]
        source_mass = source_values.sum(axis=1)
        source_coordinates = (
            source_values @ sensor_geom.sensor_centers[valid_sensor_rows]
        )
        source_coordinates = source_coordinates / np.maximum(source_mass[:, None], 1e-12)

        target_mass = raw_values @ operators["target_mass"]
        target_coordinates = raw_values @ operators["target_coordinates"]
        target_coordinates = target_coordinates / np.maximum(target_mass[:, None], 1e-12)
        centroid = np.linalg.norm(source_coordinates - target_coordinates, axis=1)
        centroid[(source_mass <= 1e-12) | (target_mass <= 1e-12)] = np.nan

        if raw_values.shape[0] > 1:
            first_difference = np.diff(raw_values, axis=0)
            temp1 = np.maximum(
                _quadratic_values(
                    first_difference, operators["temporal_quadratic"]
                ),
                0.0,
            )
        else:
            temp1 = np.zeros((0,), dtype=np.float64)
        if raw_values.shape[0] > 2:
            second_difference = (
                raw_values[2:] - 2.0 * raw_values[1:-1] + raw_values[:-2]
            )
            temp2 = np.maximum(
                _quadratic_values(
                    second_difference, operators["temporal_quadratic"]
                ),
                0.0,
            )
        else:
            temp2 = np.zeros((0,), dtype=np.float64)
    else:
        spatial = peak_raw = peak_new = centroid = np.zeros((0,), dtype=np.float64)
        temp1 = temp2 = np.zeros((0,), dtype=np.float64)

    peak_abs_error = np.abs(peak_new - peak_raw)
    peak_overshoot = peak_new > peak_raw + 1e-4
    frame_rows = []
    if include_frame_rows:
        for row_index, source_index in enumerate(source_indices.tolist()):
            frame_rows.append(
                {
                    "dataset": sequence["dataset"],
                    "split": sequence["split"],
                    "sequence_id": sequence["sequence_id"],
                    "hand": sequence["hand"],
                    "method": method,
                    "frame_idx": int(sequence["frames"][source_index]["frame_idx"]),
                    "spatial_laplacian": float(spatial[row_index]),
                    "native_spatial_laplacian": None,
                    "centroid_error_l2": (
                        float(centroid[row_index])
                        if math.isfinite(float(centroid[row_index]))
                        else None
                    ),
                    "peak_raw": float(peak_raw[row_index]),
                    "peak_new": float(peak_new[row_index]),
                    "peak_error": float(peak_new[row_index] - peak_raw[row_index]),
                    "peak_abs_error": float(peak_abs_error[row_index]),
                    "peak_overshoot": bool(peak_overshoot[row_index]),
                    "emd": None,
                    "emd_mode": emd_mode,
                    "emd_solver": emd_solver,
                    "num_raw_support": None,
                    "num_new_support": None,
                    "cost_matrix_shape": None,
                    "sinkhorn_epsilon": None,
                    "sinkhorn_iters": 0,
                }
            )
    summary = {
        "dataset": sequence["dataset"],
        "split": sequence["split"],
        "sequence_id": sequence["sequence_id"],
        "hand": sequence["hand"],
        "method": method,
        "frames_evaluated": int(raw_values.shape[0]),
        "spatial_laplacian_mean": _finite_mean(spatial.tolist()),
        "spatial_laplacian_p50": _percentile(spatial.tolist(), 50),
        "spatial_laplacian_p90": _percentile(spatial.tolist(), 90),
        "native_spatial_laplacian_mean": None,
        "temp_1st_mean": _finite_mean(temp1.tolist()),
        "temp_2nd_mean": _finite_mean(temp2.tolist()),
        "centroid_error_l2_mean": _finite_mean(centroid.tolist()),
        "peak_abs_error_mean": _finite_mean(peak_abs_error.tolist()),
        "peak_overshoot_rate": _finite_mean(peak_overshoot.astype(np.float64).tolist()),
        "emd_mean": None,
        "emd_p50": None,
        "emd_p90": None,
        "emd_mode": emd_mode,
        "emd_solver": emd_solver,
    }
    return frame_rows, summary


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
    include_frame_rows: bool = True,
) -> tuple[list[dict], dict]:
    if method == "ema_preprocess_gaussian":
        return _evaluate_ema_gaussian_sequence(
            sequence,
            method,
            frames,
            mesh,
            sensor_geom,
            emd_mode,
            emd_solver,
            include_frame_rows,
        )
    if method == "preprocess_gaussian" and emd_solver == "none":
        return _evaluate_ema_gaussian_sequence(
            sequence,
            method,
            frames,
            mesh,
            sensor_geom,
            emd_mode,
            emd_solver,
            include_frame_rows,
        )
    if method == "raw_to_mano_direct":
        if emd_solver != "none":
            raise ValueError(
                "raw_to_mano_direct uses the sparse structural fast path; "
                "set --emd_solver none to match the historical OpenTouch direct baseline"
            )
        return _evaluate_direct_method_sequence(
            sequence,
            method,
            frames,
            mesh,
            sensor_geom,
            emd_mode,
            emd_solver,
            contact_threshold,
            topk_raw,
            topk_new,
            sinkhorn_iters,
            sinkhorn_epsilon,
            include_frame_rows,
        )
    return _evaluate_method_sequence_generic(
        sequence,
        method,
        frames,
        mesh,
        sensor_geom,
        emd_mode,
        emd_solver,
        contact_threshold,
        topk_raw,
        topk_new,
        sinkhorn_iters,
        sinkhorn_epsilon,
        include_frame_rows,
    )
