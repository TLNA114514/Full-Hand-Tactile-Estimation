from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class MeshGeometry:
    vertices: np.ndarray
    faces: np.ndarray
    adjacency: list[np.ndarray]
    vertex_area: np.ndarray
    valid_vertices: np.ndarray


@dataclass
class SensorGeometry:
    sensor_ids: np.ndarray
    sensor_vertices: list[np.ndarray]
    sensor_centers: np.ndarray
    sensor_to_vertex_cost: np.ndarray
    target_vertices: np.ndarray


def repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def sha1_files(paths: Iterable[Path]) -> str:
    h = hashlib.sha1()
    for path in paths:
        path = Path(path)
        h.update(str(path).encode("utf-8"))
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                face = []
                for item in line.split()[1:4]:
                    face.append(int(item.split("/")[0]) - 1)
                faces.append(face)
    if not vertices or not faces:
        raise ValueError(f"Failed to parse mesh OBJ: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def build_adjacency(num_vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    neighbors = [set() for _ in range(num_vertices)]
    for a, b, c in faces.astype(np.int64):
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.asarray(sorted(x), dtype=np.int32) for x in neighbors]


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    area = np.zeros((vertices.shape[0],), dtype=np.float32)
    tri = vertices[faces]
    tri_area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    for idx in range(3):
        np.add.at(area, faces[:, idx], tri_area / 3.0)
    area[area <= 0] = 1.0
    return area


def load_mesh_geometry(repo_root: Path | None = None) -> MeshGeometry:
    repo_root = repo_root or repo_root_from_file()
    obj_path = repo_root / "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj"
    vertices, faces = load_obj_mesh(obj_path)
    adjacency = build_adjacency(vertices.shape[0], faces)
    area = vertex_areas(vertices, faces)
    valid = np.arange(vertices.shape[0], dtype=np.int32)
    return MeshGeometry(vertices=vertices, faces=faces, adjacency=adjacency, vertex_area=area, valid_vertices=valid)


def graph_distances_from_sources(mesh: MeshGeometry, source_vertices: np.ndarray) -> np.ndarray:
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        rows, cols, data = [], [], []
        for i, nbrs in enumerate(mesh.adjacency):
            for j in nbrs.tolist():
                rows.append(i)
                cols.append(j)
                data.append(float(np.linalg.norm(mesh.vertices[i] - mesh.vertices[j])))
        graph = csr_matrix((data, (rows, cols)), shape=(mesh.vertices.shape[0], mesh.vertices.shape[0]))
        dist = shortest_path(graph, directed=False, indices=source_vertices.astype(np.int32))
        return np.asarray(dist, dtype=np.float32)
    except Exception:
        src = mesh.vertices[source_vertices.astype(np.int32)]
        return np.linalg.norm(src[:, None, :] - mesh.vertices[None, :, :], axis=2).astype(np.float32)


def _center_for_vids(vertices: np.ndarray, vids: Iterable[int]) -> tuple[np.ndarray, np.ndarray] | None:
    arr = np.asarray([int(v) for v in vids if 0 <= int(v) < vertices.shape[0]], dtype=np.int32)
    if arr.size == 0:
        return None
    return arr, vertices[arr].mean(axis=0)


def opentouch_sensor_vertices(repo_root: Path, hand: str, mesh: MeshGeometry) -> Dict[int, np.ndarray]:
    layout_path = repo_root / "opentouch/preprocess/scratch/handLayoutNewest_meshid.json"
    layout = load_json(layout_path)["positions"]
    out: dict[int, np.ndarray] = {}
    for key, info in layout.items():
        sep = "-" if "-" in key else ","
        row, col = [int(x) for x in key.split(sep)]
        sid = row * 16 + col
        item = _center_for_vids(mesh.vertices, info.get("mano_vid", []))
        if item is not None:
            out[sid] = item[0]
    return out


def ta_sensor_vertices(repo_root: Path, hand: str, mesh: MeshGeometry) -> Dict[int, np.ndarray]:
    mapping_path = repo_root / f"TouchAnything/configs/pressure_position_mapping_{hand}.json"
    visual_path = repo_root / f"TouchAnything/scripts/tools/mano_visualization/ta_to_mano_mapping_{hand}_visual.json"
    grid_to_sensor = {}
    if mapping_path.exists():
        for key, value in load_json(mapping_path).items():
            row, col = [int(x) for x in key.split(",")]
            grid_to_sensor[(row, col)] = int(value)
    layout = load_json(visual_path)["positions"] if visual_path.exists() else {}
    out: dict[int, np.ndarray] = {}
    for key, info in layout.items():
        row, col = [int(x) for x in key.split(",")]
        sid = grid_to_sensor.get((row, col))
        if sid is None:
            continue
        item = _center_for_vids(mesh.vertices, info.get("mano_vid", []))
        if item is not None:
            out[int(sid)] = item[0]
    return out


def egotactile_sensor_vertices(
    repo_root: Path,
    hand: str,
    mesh: MeshGeometry,
) -> tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    mapping_path = repo_root / f"preprocess/egotactile/egotactile_mapping_{hand}.json"
    mapping = load_json(mapping_path)
    vertices_by_sensor: Dict[int, np.ndarray] = {}
    centers_by_sensor: Dict[int, np.ndarray] = {}
    for item in mapping.get("pressure_sensors", {}).values():
        sensor_id = int(item["raw_id_0based"])
        mapped = np.asarray(
            [
                int(vertex_id)
                for vertex_id in item.get("mano_vid", [])
                if 0 <= int(vertex_id) < mesh.vertices.shape[0]
            ],
            dtype=np.int32,
        )
        if mapped.size == 0:
            continue
        vertices_by_sensor[sensor_id] = mapped
        center = np.asarray(item.get("center_xyz", []), dtype=np.float32)
        centers_by_sensor[sensor_id] = (
            center if center.shape == (3,) else mesh.vertices[mapped].mean(axis=0)
        )
    return vertices_by_sensor, centers_by_sensor


def _sensor_geometry_from_cache(cache_path: Path) -> SensorGeometry:
    with np.load(cache_path, allow_pickle=True) as data:
        return SensorGeometry(
            sensor_ids=data["sensor_ids"].astype(np.int32),
            sensor_vertices=[x.astype(np.int32) for x in data["sensor_vertices"]],
            sensor_centers=data["sensor_centers"].astype(np.float32),
            sensor_to_vertex_cost=data["sensor_to_vertex_cost"].astype(np.float32),
            target_vertices=data["target_vertices"].astype(np.int32),
        )


def _write_sensor_geometry_cache(
    cache_path: Path,
    sensor_ids: np.ndarray,
    sensor_vertices: list[np.ndarray],
    sensor_centers: np.ndarray,
    sensor_to_vertex_cost: np.ndarray,
    target_vertices: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp_path,
            sensor_ids=sensor_ids,
            sensor_vertices=np.asarray(sensor_vertices, dtype=object),
            sensor_centers=sensor_centers,
            sensor_to_vertex_cost=sensor_to_vertex_cost,
            target_vertices=target_vertices.astype(np.int32),
        )
        # npz loading is lazy; read every array before publishing the cache.
        _sensor_geometry_from_cache(tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def build_sensor_geometry(
    dataset: str,
    hand: str,
    mesh: MeshGeometry,
    repo_root: Path | None = None,
    cache_dir: Path | None = None,
) -> SensorGeometry:
    repo_root = repo_root or repo_root_from_file()
    cache_dir = cache_dir or (repo_root / "preprocess/artifacts/representation_eval/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_key = dataset.lower()
    if dataset_key == "opentouch":
        mapping_files = [repo_root / "opentouch/preprocess/scratch/handLayoutNewest_meshid.json"]
        sensor_map = opentouch_sensor_vertices(repo_root, hand, mesh)
        explicit_centers = {}
    elif dataset_key == "egotactile":
        mapping_files = [
            repo_root / f"preprocess/egotactile/egotactile_mapping_{hand}.json"
        ]
        sensor_map, explicit_centers = egotactile_sensor_vertices(
            repo_root, hand, mesh
        )
    else:
        mapping_files = [
            repo_root / f"TouchAnything/configs/pressure_position_mapping_{hand}.json",
            repo_root / f"TouchAnything/scripts/tools/mano_visualization/ta_to_mano_mapping_{hand}_visual.json",
        ]
        sensor_map = ta_sensor_vertices(repo_root, hand, mesh)
        explicit_centers = {}
    mesh_file = repo_root / "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj"
    key = sha1_files([mesh_file, *mapping_files])
    cache_path = cache_dir / f"sensor_geom_{dataset_key}_{hand}_{key}.npz"
    if cache_path.exists():
        try:
            return _sensor_geometry_from_cache(cache_path)
        except Exception as exc:
            print(f"Warning: bad sensor geometry cache {cache_path}: {exc}; rebuilding.")
            try:
                cache_path.unlink()
            except OSError:
                pass

    sensor_ids = np.asarray(sorted(sensor_map), dtype=np.int32)
    sensor_vertices = [np.asarray(sensor_map[int(sid)], dtype=np.int32) for sid in sensor_ids]
    centers = np.stack(
        [
            explicit_centers.get(int(sensor_id), mesh.vertices[vertices].mean(axis=0))
            for sensor_id, vertices in zip(sensor_ids.tolist(), sensor_vertices)
        ],
        axis=0,
    ).astype(np.float32)
    target_vertices = np.unique(np.concatenate(sensor_vertices)) if sensor_vertices else np.zeros(0, dtype=np.int32)
    if target_vertices.size == 0:
        target_vertices = mesh.valid_vertices
    cost_rows = []
    for vids, center in zip(sensor_vertices, centers):
        dist = graph_distances_from_sources(mesh, vids)
        jump = np.linalg.norm(mesh.vertices[vids] - center[None, :], axis=1)
        cost_rows.append(np.min(dist + jump[:, None], axis=0))
    cost = np.stack(cost_rows, axis=0).astype(np.float32) if cost_rows else np.zeros((0, mesh.vertices.shape[0]), dtype=np.float32)
    _write_sensor_geometry_cache(
        cache_path,
        sensor_ids=sensor_ids,
        sensor_vertices=sensor_vertices,
        sensor_centers=centers,
        sensor_to_vertex_cost=cost,
        target_vertices=target_vertices,
    )
    return SensorGeometry(
        sensor_ids=sensor_ids,
        sensor_vertices=sensor_vertices,
        sensor_centers=centers,
        sensor_to_vertex_cost=cost,
        target_vertices=target_vertices.astype(np.int32),
    )


def map_sensor_vector_to_vertices(sensor_values: np.ndarray, geom: SensorGeometry, num_vertices: int) -> np.ndarray:
    out = np.zeros((num_vertices,), dtype=np.float32)
    counts = np.zeros((num_vertices,), dtype=np.float32)
    values = np.asarray(sensor_values, dtype=np.float32).reshape(-1)
    for sid, vids in zip(geom.sensor_ids.tolist(), geom.sensor_vertices):
        if 0 <= sid < values.size:
            out[vids] += float(values[sid])
            counts[vids] += 1.0
    mask = counts > 0
    out[mask] /= counts[mask]
    return out


def gaussian_smooth_vertices(values: np.ndarray, adjacency: list[np.ndarray], vertices: np.ndarray, sigma: float = 0.005, iters: int = 2) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if sigma <= 0 or iters <= 0:
        return out
    two_sig2 = 2.0 * sigma * sigma
    for _ in range(iters):
        new = out.copy()
        for i, nbrs in enumerate(adjacency):
            if nbrs.size == 0:
                continue
            d2 = np.sum((vertices[nbrs] - vertices[i]) ** 2, axis=1)
            w = np.exp(-d2 / two_sig2).astype(np.float32)
            denom = float(1.0 + w.sum())
            new[i] = (out[i] + float(np.dot(w, out[nbrs]))) / max(denom, 1e-8)
        out = new
    return out
