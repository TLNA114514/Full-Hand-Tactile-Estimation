from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .geometry import (
    MeshGeometry,
    SensorGeometry,
    gaussian_smooth_vertices,
    map_sensor_vector_to_vertices,
)
from .io import load_json


METHODS = {
    "ot_raw_heatmap",
    "ot_discrete_heatmap",
    "ot_centered_mano",
    "egotactile_heatmap",
    "preprocess_gaussian",
}


@dataclass
class FrameRepresentation:
    raw_sensor: np.ndarray | None
    target: np.ndarray | None
    target_space: str
    native_target: np.ndarray | None = None
    native_mask: np.ndarray | None = None


def _as_float_array(value, shape=None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        return None
    if shape is not None and arr.shape != tuple(shape):
        return None
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _clip01(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    return np.clip(np.asarray(arr, dtype=np.float32), 0.0, 1.0)


def normalize_opentouch_pressure(value) -> np.ndarray | None:
    arr = _as_float_array(value)
    if arr is None:
        return None
    if arr.ndim == 2:
        flat = arr.reshape(-1)
    else:
        flat = arr.reshape(-1)
    if flat.size < 1:
        return None
    if flat.max(initial=0.0) > 1.5:
        flat = np.clip(flat, 0.0, 3072.0) / 3072.0
    return _clip01(flat)


def normalize_opentouch_pressure_direct(value) -> np.ndarray | None:
    arr = _as_float_array(value)
    if arr is None:
        return None
    flat = arr.reshape(-1)
    if flat.size < 1:
        return None
    if flat.max(initial=0.0) > 1.5:
        flat = np.clip((3072.0 - flat) / 3072.0, 0.0, 1.0)
    return _clip01(flat)


def discretize(values: np.ndarray, levels: int = 5) -> np.ndarray:
    values = _clip01(values)
    if levels <= 1:
        return values
    idx = np.round(values * (levels - 1)).astype(np.int32)
    bins = np.linspace(0.0, 1.0, levels, dtype=np.float32)
    return bins[np.clip(idx, 0, levels - 1)]


def _left_right_key(hand: str, suffix: str) -> str:
    return f"{hand}_{suffix}"


def extract_raw_sensor(meta: dict, dataset: str, hand: str) -> np.ndarray | None:
    if dataset == "opentouch":
        original = meta.get("original_hdf5_data", {})
        pressure = original.get(_left_right_key(hand, "pressure"))
        return normalize_opentouch_pressure(pressure)
    if dataset == "egotactile":
        return _clip01(_as_float_array(meta.get("normalized_sensor_256"), shape=(256,)))
    if dataset == "touchanything":
        hand_meta = meta.get("hands", {}).get(hand, {})
        grid = _as_float_array(hand_meta.get("normalized_pressure_grid"))
        if grid is not None:
            return _clip01(grid.reshape(-1))
        raw = _as_float_array(hand_meta.get("raw_pressure"))
        if raw is not None:
            if raw.max(initial=0.0) > 1.5:
                raw = np.clip(raw, 0.0, 200.0) / 200.0
            return _clip01(raw.reshape(-1))
    return None


def extract_preprocess_gaussian(meta: dict, dataset: str, hand: str) -> np.ndarray | None:
    if dataset == "touchanything":
        return _clip01(_as_float_array(meta.get("hands", {}).get(hand, {}).get("gaussian_pressure")))
    value = meta.get("gaussian_pressure")
    if value is None:
        value = meta.get("original_hdf5_data", {}).get(_left_right_key(hand, "pressure_continuous_subdiv"))
    return _clip01(_as_float_array(value))


def _repo_root_default() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_ta_grid_mapping(repo_root: Path, hand: str) -> dict[tuple[int, int], int]:
    mapping_path = repo_root / f"TouchAnything/configs/pressure_position_mapping_{hand}.json"
    if not mapping_path.exists():
        return {}
    out = {}
    for key, value in load_json(mapping_path).items():
        try:
            row, col = [int(x) for x in key.split(",")]
            out[(row, col)] = int(value)
        except Exception:
            continue
    return out


def _ta_grid_to_sensor256(grid: np.ndarray | None, repo_root: Path, hand: str) -> np.ndarray | None:
    arr = _clip01(_as_float_array(grid))
    if arr is None:
        return None
    if arr.ndim != 2:
        arr = arr.reshape(-1)
        if arr.size == 256:
            return arr.astype(np.float32)
        return None
    mapping = _load_ta_grid_mapping(repo_root, hand)
    if not mapping:
        return None
    out = np.zeros((256,), dtype=np.float32)
    h, w = arr.shape
    for (row, col), sid in mapping.items():
        if 0 <= row < h and 0 <= col < w and 0 <= sid < out.size:
            out[sid] = float(arr[row, col])
    return out


def _ensure_frame_index(sequence: dict, count: int) -> None:
    frames = sequence.get("frames")
    if isinstance(frames, list) and len(frames) == count:
        return
    sequence["frames"] = [{"frame_idx": i} for i in range(max(0, int(count)))]


def _array_frame(arr, frame_idx: int):
    if arr is None:
        return None
    try:
        if frame_idx >= arr.shape[0]:
            return None
        return arr[frame_idx]
    except Exception:
        return None


def _num_frames_from_arrays(*arrays) -> int:
    counts = []
    for arr in arrays:
        if arr is not None:
            try:
                counts.append(int(arr.shape[0]))
            except Exception:
                pass
    return min(counts) if counts else 0


def build_ot_raw(raw_sensor: np.ndarray | None, sensor_geom: SensorGeometry, mesh: MeshGeometry) -> np.ndarray | None:
    if raw_sensor is None:
        return None
    return map_sensor_vector_to_vertices(raw_sensor, sensor_geom, mesh.vertices.shape[0])


def build_ot_discrete(raw_sensor: np.ndarray | None, sensor_geom: SensorGeometry, mesh: MeshGeometry, levels: int = 5) -> np.ndarray | None:
    if raw_sensor is None:
        return None
    return map_sensor_vector_to_vertices(discretize(raw_sensor, levels=levels), sensor_geom, mesh.vertices.shape[0])


def build_centered_sequence(
    raw_sensors: list[np.ndarray | None],
    sensor_geom: SensorGeometry,
    mesh: MeshGeometry,
    smooth_sigma: float = 0.005,
) -> list[np.ndarray | None]:
    valid = [x for x in raw_sensors if x is not None and x.size > 0]
    if not valid:
        return [None for _ in raw_sensors]
    normed = []
    for arr in raw_sensors:
        if arr is None:
            normed.append(None)
            continue
        x = np.asarray(arr, dtype=np.float32)
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx > mn:
            normed.append((x - mn) / (mx - mn))
        else:
            normed.append(np.zeros_like(x, dtype=np.float32))
    mean = np.mean([x for x in normed if x is not None], axis=0).astype(np.float32)
    centered = [(x - mean) if x is not None else None for x in normed]
    valid_centered = [x for x in centered if x is not None]
    vmin = float(np.min([np.min(x) for x in valid_centered]))
    vmax = float(np.max([np.max(x) for x in valid_centered]))
    out = []
    for x in centered:
        if x is None:
            out.append(None)
            continue
        if vmax > vmin:
            y = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            y = np.ones_like(x, dtype=np.float32)
        vertices = map_sensor_vector_to_vertices(y, sensor_geom, mesh.vertices.shape[0])
        vertices = gaussian_smooth_vertices(vertices, mesh.adjacency, mesh.vertices, sigma=smooth_sigma, iters=2)
        out.append(_clip01(vertices))
    return out


def _load_npz_source_representations(
    sequence: dict,
    methods: set[str],
    mesh: MeshGeometry,
    sensor_geom: SensorGeometry,
    discrete_levels: int,
    repo_root: Path,
) -> dict[str, list[FrameRepresentation]]:
    dataset = sequence["dataset"]
    hand = sequence["hand"]
    source_path = Path(sequence["source_path"])
    out = {method: [] for method in methods}
    with np.load(source_path, allow_pickle=False) as data:
        if dataset == "egotactile":
            raw_arr = data[f"{hand}_sensor_256_norm"][:] if f"{hand}_sensor_256_norm" in data else None
            valid_arr = data[f"{hand}_sensor_valid"][:] if f"{hand}_sensor_valid" in data else None
            gaussian_arr = data[f"{hand}_pressure_continuous_subdiv"][:] if f"{hand}_pressure_continuous_subdiv" in data else None
            n = _num_frames_from_arrays(raw_arr, gaussian_arr, valid_arr)
            _ensure_frame_index(sequence, n)
            raw_sensors = []
            for i in range(n):
                valid = True if valid_arr is None else bool(np.asarray(valid_arr).reshape(-1)[i])
                raw = _clip01(_as_float_array(raw_arr[i], shape=(256,))) if valid and raw_arr is not None else None
                raw_sensors.append(raw)
            for i, raw in enumerate(raw_sensors):
                if "egotactile_heatmap" in methods:
                    native, native_mask, sampled = egotactile_heatmap_native(raw, hand)
                    mano = map_sensor_vector_to_vertices(sampled, sensor_geom, mesh.vertices.shape[0]) if sampled is not None else None
                    out["egotactile_heatmap"].append(FrameRepresentation(raw, mano, "mano", native, native_mask))
                if "preprocess_gaussian" in methods:
                    target = _clip01(_as_float_array(_array_frame(gaussian_arr, i)))
                    out["preprocess_gaussian"].append(FrameRepresentation(raw, target, "mano"))

        elif dataset == "touchanything":
            raw_arr = data[f"{hand}_pressure_grid"][:] if f"{hand}_pressure_grid" in data else None
            gaussian_arr = data[f"{hand}_pressure_continuous_subdiv"][:] if f"{hand}_pressure_continuous_subdiv" in data else None
            n = _num_frames_from_arrays(raw_arr, gaussian_arr)
            _ensure_frame_index(sequence, n)
            for i in range(n):
                raw = _ta_grid_to_sensor256(_array_frame(raw_arr, i), repo_root, hand)
                if "preprocess_gaussian" in methods:
                    target = _clip01(_as_float_array(_array_frame(gaussian_arr, i)))
                    out["preprocess_gaussian"].append(FrameRepresentation(raw, target, "mano"))
    return {key: value for key, value in out.items() if value}


def _load_hdf5_source_representations(
    sequence: dict,
    methods: set[str],
    mesh: MeshGeometry,
    sensor_geom: SensorGeometry,
    discrete_levels: int,
) -> dict[str, list[FrameRepresentation]]:
    import h5py

    hand = sequence["hand"]
    source_path = Path(sequence["source_path"])
    demo_name = sequence.get("hdf5_demo")
    out = {method: [] for method in methods}
    with h5py.File(source_path, "r") as f:
        if not demo_name or "data" not in f or demo_name not in f["data"]:
            return {}
        demo = f["data"][demo_name]
        raw_arr = demo[f"{hand}_pressure"][:] if f"{hand}_pressure" in demo else None
        gaussian_arr = demo[f"{hand}_pressure_continuous_subdiv"][:] if f"{hand}_pressure_continuous_subdiv" in demo else None
        n = _num_frames_from_arrays(raw_arr, gaussian_arr)
        _ensure_frame_index(sequence, n)
        raw_sensors = [normalize_opentouch_pressure_direct(_array_frame(raw_arr, i)) for i in range(n)]
        centered = None
        if "ot_centered_mano" in methods:
            centered = build_centered_sequence(raw_sensors, sensor_geom, mesh)
        for i, raw in enumerate(raw_sensors):
            if "ot_raw_heatmap" in methods:
                out["ot_raw_heatmap"].append(FrameRepresentation(raw, build_ot_raw(raw, sensor_geom, mesh), "mano"))
            if "ot_discrete_heatmap" in methods:
                out["ot_discrete_heatmap"].append(
                    FrameRepresentation(raw, build_ot_discrete(raw, sensor_geom, mesh, levels=discrete_levels), "mano")
                )
            if "ot_centered_mano" in methods:
                out["ot_centered_mano"].append(FrameRepresentation(raw, centered[i] if centered else None, "mano"))
            if "preprocess_gaussian" in methods:
                target = _clip01(_as_float_array(_array_frame(gaussian_arr, i)))
                out["preprocess_gaussian"].append(FrameRepresentation(raw, target, "mano"))
    return {key: value for key, value in out.items() if value}


def egotactile_heatmap_native(raw_sensor: np.ndarray | None, hand: str, square_size: int = 256, sigma_pix: float = 1.0) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if raw_sensor is None:
        return None, None, None
    try:
        from EgoPressureDiff.scripts.raw_to_training import make_side_by_side_videos  # noqa: F401
    except Exception:
        pass
    # Lightweight reimplementation of the EgoPressureDiff 17x19 mask layouts.
    if hand == "left":
        mask = np.array([
            [31,30,29,0,28,27,26,0,25,24,23,0,22,21,20,0,19,18,17],
            [15,14,13,0,12,11,10,0,9,8,7,0,6,5,4,0,3,2,1],
            [255,254,253,0,252,251,250,0,249,248,247,0,246,245,244,0,243,242,241],
            [239,238,237,0,236,235,234,0,233,232,231,0,230,229,228,0,227,226,225],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,222,0,0,0,219,0,0,0,216,0,0,0,213,0,0,0,0,210],
            [0,222,0,0,0,219,0,0,0,216,0,0,0,213,0,0,0,0,210],
            [0,222,0,0,0,219,0,0,0,216,0,0,0,213,0,0,0,210,0],
            [0,222,0,0,0,219,0,0,0,216,0,0,0,213,0,0,0,210,0],
            [0,222,0,0,0,219,0,0,0,216,0,0,0,213,0,0,210,0,0],
            [0,222,0,0,0,219,0,0,0,216,0,0,0,213,0,0,210,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [207,206,205,204,203,202,201,200,199,198,197,196,0,0,0,0,0,0,0],
            [191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,0,0,0,0],
            [175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,0,0,0,0],
            [159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,0,0,0,0],
            [143,142,141,140,139,138,137,136,135,134,133,132,131,130,129,0,0,0,0],
        ], dtype=np.int32)
    else:
        mask = np.array([
            [240,239,238,0,237,236,235,0,234,233,232,0,231,230,229,0,228,227,226],
            [256,255,254,0,253,252,251,0,250,249,248,0,247,246,245,0,244,243,242],
            [16,15,14,0,13,12,11,0,10,9,8,0,7,6,5,0,4,3,2],
            [32,31,30,0,29,28,27,0,26,25,24,0,23,22,21,0,20,19,18],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [47,0,0,0,0,44,0,0,0,41,0,0,0,38,0,0,0,35,0],
            [47,0,0,0,0,44,0,0,0,41,0,0,0,38,0,0,0,35,0],
            [0,47,0,0,0,44,0,0,0,41,0,0,0,38,0,0,0,35,0],
            [0,47,0,0,0,44,0,0,0,41,0,0,0,38,0,0,0,35,0],
            [0,0,47,0,0,44,0,0,0,41,0,0,0,38,0,0,0,35,0],
            [0,0,47,0,0,44,0,0,0,41,0,0,0,38,0,0,0,35,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,61,60,59,58,57,56,55,54,53,52,51,50],
            [0,0,0,0,80,79,78,77,76,75,74,73,72,71,70,69,68,67,66],
            [0,0,0,0,96,95,94,93,92,91,90,89,88,87,86,85,84,83,82],
            [0,0,0,0,112,111,110,109,108,107,106,105,104,103,102,101,100,99,98],
            [0,0,0,0,128,127,126,125,124,123,122,121,120,119,118,117,116,115,114],
        ], dtype=np.int32)
    valid = mask > 0
    h, w = mask.shape
    scale = min(square_size / float(h), square_size / float(w))
    th, tw = int(round(h * scale)), int(round(w * scale))
    top, left = (square_size - th) // 2, (square_size - tw) // 2
    ys, xs = np.where(valid)
    centers = np.stack([(ys + 0.5) * (th / h) - 0.5, (xs + 0.5) * (tw / w) - 0.5], axis=1).astype(np.float32)
    pix_y, pix_x = np.where(np.ones((th, tw), dtype=bool))
    pix = np.stack([pix_y, pix_x], axis=1).astype(np.float32)
    idx1 = mask[ys, xs].astype(np.int32)
    vals = np.asarray([raw_sensor[i - 1] if 0 <= i - 1 < raw_sensor.size else 0.0 for i in idx1], dtype=np.float32)
    d2 = (pix[:, [0]] - centers[:, 0][None, :]) ** 2 + (pix[:, [1]] - centers[:, 1][None, :]) ** 2
    weights = np.exp(-0.5 * d2 / max(sigma_pix * sigma_pix, 1e-8)).astype(np.float32)
    heat_small = (weights @ vals) / np.maximum(weights.sum(axis=1), 1e-8)
    heat = np.zeros((square_size, square_size), dtype=np.float32)
    mask_out = np.zeros((square_size, square_size), dtype=bool)
    heat[top:top + th, left:left + tw] = heat_small.reshape(th, tw)
    mask_out[top:top + th, left:left + tw] = True
    sampled = np.zeros((256,), dtype=np.float32)
    counts = np.zeros((256,), dtype=np.float32)
    for idx, (cy, cx) in zip(idx1.tolist(), centers):
        yy = int(round(float(cy))) + top
        xx = int(round(float(cx))) + left
        if 0 <= yy < square_size and 0 <= xx < square_size and 1 <= idx <= 256:
            sampled[idx - 1] += float(heat[yy, xx])
            counts[idx - 1] += 1.0
    valid_counts = counts > 0
    sampled[valid_counts] /= counts[valid_counts]
    return _clip01(heat), mask_out, _clip01(sampled)


def load_sequence_representations(
    sequence: dict,
    methods: Iterable[str],
    mesh: MeshGeometry,
    sensor_geom: SensorGeometry,
    discrete_levels: int = 5,
    repo_root: Path | None = None,
) -> dict[str, list[FrameRepresentation]]:
    dataset = sequence["dataset"]
    hand = sequence["hand"]
    methods = set(methods)
    repo_root = Path(repo_root) if repo_root is not None else _repo_root_default()
    if sequence.get("source_type") == "npz":
        return _load_npz_source_representations(sequence, methods, mesh, sensor_geom, discrete_levels, repo_root)
    if sequence.get("source_type") == "hdf5":
        return _load_hdf5_source_representations(sequence, methods, mesh, sensor_geom, discrete_levels)
    metas = [load_json(Path(frame["meta_path"])) for frame in sequence["frames"]]
    raw_sensors = [extract_raw_sensor(meta, dataset, hand) for meta in metas]
    out = {method: [] for method in methods}
    centered = None
    if "ot_centered_mano" in methods and dataset == "opentouch":
        centered = build_centered_sequence(raw_sensors, sensor_geom, mesh)
    for idx, meta in enumerate(metas):
        raw = raw_sensors[idx]
        if "ot_raw_heatmap" in methods and dataset == "opentouch":
            out["ot_raw_heatmap"].append(FrameRepresentation(raw, build_ot_raw(raw, sensor_geom, mesh), "mano"))
        if "ot_discrete_heatmap" in methods and dataset == "opentouch":
            out["ot_discrete_heatmap"].append(
                FrameRepresentation(raw, build_ot_discrete(raw, sensor_geom, mesh, levels=discrete_levels), "mano")
            )
        if "ot_centered_mano" in methods and dataset == "opentouch":
            out["ot_centered_mano"].append(FrameRepresentation(raw, centered[idx] if centered else None, "mano"))
        if "egotactile_heatmap" in methods and dataset == "egotactile":
            native, native_mask, sampled = egotactile_heatmap_native(raw, hand)
            mano = map_sensor_vector_to_vertices(sampled, sensor_geom, mesh.vertices.shape[0]) if sampled is not None else None
            out["egotactile_heatmap"].append(FrameRepresentation(raw, mano, "mano", native, native_mask))
        if "preprocess_gaussian" in methods:
            target = extract_preprocess_gaussian(meta, dataset, hand)
            out["preprocess_gaussian"].append(FrameRepresentation(raw, target, "mano"))
    return {key: value for key, value in out.items() if value}
