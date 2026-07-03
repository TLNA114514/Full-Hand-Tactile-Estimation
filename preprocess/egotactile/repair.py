import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm


worker_device = None
worker_deps_left = None
worker_deps_right = None
worker_args = None

HAND_TO_JSON_KEY = {
    "left": "LH",
    "right": "RH",
}

DEFAULT_SCAN_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "extracted_frames",
    "metadata",
    "artifacts",
}


class DepContainer:
    pass


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_frame_list(path):
    with open(path, "r") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        frames = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                frames.append(json.loads(line))
        return frames


def normalize_sensor(sensor, pmin=5.0, pmax=200.0):
    values = np.asarray(sensor, dtype=np.float32).reshape(-1)
    if values.size != 256:
        raise ValueError(f"Expected sensor_256 with 256 values, got {values.size}")
    if pmax <= pmin:
        raise ValueError(f"pmax must be greater than pmin, got pmin={pmin}, pmax={pmax}")
    values = np.clip(values, pmin, pmax)
    values = (values - pmin) / (pmax - pmin)
    return values


def load_pressure_position_mapping(mapping_path):
    mapping = load_json(mapping_path)
    parsed = []
    for key, sensor_idx in mapping.items():
        row, col = [int(x) for x in key.split(",")]
        parsed.append((row, col, int(sensor_idx)))
    return parsed


def sensors_to_21_grid(sensor_norm, pressure_mapping):
    grid = np.full((sensor_norm.shape[0], 21, 21), np.nan, dtype=np.float32)
    for row, col, sensor_idx in pressure_mapping:
        if 0 <= row < 21 and 0 <= col < 21 and 0 <= sensor_idx < sensor_norm.shape[1]:
            grid[:, row, col] = sensor_norm[:, sensor_idx]
    return grid


def transform_grid_for_mano(grid, transform):
    if transform == "none":
        return grid
    if transform == "flip_lr":
        return np.flip(grid, axis=2).copy()
    if transform == "flip_ud":
        return np.flip(grid, axis=1).copy()
    if transform == "rot180":
        return np.flip(np.flip(grid, axis=1), axis=2).copy()
    raise ValueError(f"Unknown grid transform: {transform}")


def load_ta_mesh_and_compute_dist_cpu(repo_root, hand="left", sigma=0.005):
    obj_path = os.path.join(repo_root, "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj")
    palm_faces_path = os.path.join(repo_root, "opentouch/preprocess/scratch/auto_calibrated_palm_subdiv_faces.json")
    mapping_path = os.path.join(
        repo_root,
        f"TouchAnything/scripts/tools/mano_visualization/ta_to_mano_mapping_{hand}_visual.json",
    )

    mesh = trimesh.load(obj_path, process=False)
    mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mano_faces = np.asarray(mesh.faces, dtype=np.int32)
    v_total = mano_vertices.shape[0]

    layout = load_json(mapping_path)["positions"]
    valid_nodes = {}
    for nid, info in layout.items():
        vids = [int(v) for v in info.get("mano_vid", []) if int(v) < v_total]
        if vids:
            center = np.mean(mano_vertices[vids], axis=0)
            valid_nodes[nid] = {"center": center, "vids": vids}

    edges = set()
    for a, b, c in mano_faces.astype(np.int64):
        edges.update({(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))})

    rows, cols, data = [], [], []
    for i, j in edges:
        dist = np.linalg.norm(mano_vertices[i] - mano_vertices[j])
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([dist, dist])
    graph = csr_matrix((data, (rows, cols)), shape=(v_total, v_total))

    node_keys = list(valid_nodes.keys())
    dist_matrix = np.zeros((len(node_keys), v_total), dtype=np.float32)
    for k, nid in enumerate(node_keys):
        center = valid_nodes[nid]["center"]
        vids = valid_nodes[nid]["vids"]
        d_vids = shortest_path(graph, directed=False, indices=vids)
        jump_dists = np.linalg.norm(mano_vertices[vids] - center, axis=1)
        dist_matrix[k, :] = np.min(d_vids + jump_dists[:, np.newaxis], axis=0)

    weights = np.exp(-(dist_matrix**2) / (2.0 * sigma * sigma))

    palm_vertices_set = set()
    palm_data = load_json(palm_faces_path)
    for fid in palm_data["group_negative"]["face_indices"]:
        if fid < len(mano_faces):
            for vid in mano_faces[fid]:
                if vid < v_total:
                    palm_vertices_set.add(int(vid))

    grid_rows = []
    grid_cols = []
    for nid in node_keys:
        row, col = [int(x) for x in nid.split(",")]
        grid_rows.append(row)
        grid_cols.append(col)

    deps = DepContainer()
    deps.v_total = v_total
    deps.valid_rows_cpu = torch.tensor(grid_rows, dtype=torch.long)
    deps.valid_cols_cpu = torch.tensor(grid_cols, dtype=torch.long)
    deps.weights_tensor_cpu = torch.tensor(weights, dtype=torch.float32)
    deps.palm_vertices_tensor_cpu = torch.tensor(sorted(palm_vertices_set), dtype=torch.long)
    return deps


def move_deps_to_device(deps, device):
    moved = DepContainer()
    moved.v_total = deps.v_total
    moved.valid_rows = deps.valid_rows_cpu.to(device)
    moved.valid_cols = deps.valid_cols_cpu.to(device)
    moved.weights_tensor = deps.weights_tensor_cpu.to(device)
    moved.palm_vertices_tensor = deps.palm_vertices_tensor_cpu.to(device)
    return moved


def init_worker(gpu_queue, deps_left_cpu, deps_right_cpu, args_dict):
    global worker_device, worker_deps_left, worker_deps_right, worker_args
    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    worker_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    worker_deps_left = move_deps_to_device(deps_left_cpu, worker_device)
    worker_deps_right = move_deps_to_device(deps_right_cpu, worker_device)
    worker_args = args_dict


def process_sequence_gpu(grid, deps):
    grid = np.nan_to_num(grid, nan=0.0).astype(np.float32)
    frames = grid.shape[0]
    if frames == 0:
        return np.zeros((0, deps.v_total), dtype=np.float32)

    with torch.no_grad():
        p_tensor = torch.tensor(grid, dtype=torch.float32, device=worker_device)
        p_tensor = torch.clamp(p_tensor, 0.0, 1.0)
        active = p_tensor[:, deps.valid_rows, deps.valid_cols]

        batch_size = worker_args["batch_size"]
        out = torch.zeros((frames, deps.v_total), dtype=torch.float32, device=worker_device)
        for start in range(0, frames, batch_size):
            active_b = active[start:start + batch_size].unsqueeze(2)
            weights_b = deps.weights_tensor.unsqueeze(0)
            palm_vals, _ = torch.max(active_b * weights_b, dim=1)
            masked = torch.zeros_like(palm_vals)
            masked[:, deps.palm_vertices_tensor] = palm_vals[:, deps.palm_vertices_tensor]
            out[start:start + batch_size] = torch.clamp(masked, 0.0, 1.0)
        return out.cpu().numpy()


def extract_hand_sensor(frames, hand, pmin, pmax):
    key = HAND_TO_JSON_KEY[hand]
    values = []
    valid_mask = []
    for frame in frames:
        if key in frame and "sensor_256" in frame[key]:
            try:
                values.append(normalize_sensor(frame[key]["sensor_256"], pmin=pmin, pmax=pmax))
                valid_mask.append(True)
            except ValueError:
                values.append(np.zeros(256, dtype=np.float32))
                valid_mask.append(False)
        else:
            values.append(np.zeros(256, dtype=np.float32))
            valid_mask.append(False)
    return np.stack(values, axis=0).astype(np.float32), np.asarray(valid_mask, dtype=bool)


def process_data_json_worker(data_json_path):
    try:
        frames = load_frame_list(data_json_path)
        if not frames:
            return {"path": data_json_path, "status": "empty"}

        repo_root = worker_args["repo_root"]
        output_name = worker_args["output_name"]
        force = worker_args["force"]
        pmin = worker_args["pmin"]
        pmax = worker_args["pmax"]

        out_path = os.path.join(os.path.dirname(data_json_path), output_name)
        if os.path.exists(out_path) and not force:
            return {"path": data_json_path, "status": "exists"}

        result = {
            "pmin": np.asarray(pmin, dtype=np.float32),
            "pmax": np.asarray(pmax, dtype=np.float32),
            "normalization": np.asarray(f"linear_{pmin:g}_{pmax:g}_to_0_1"),
            "frame_count": np.asarray(len(frames), dtype=np.int32),
        }
        processed_any = False

        for hand in ("left", "right"):
            pressure_mapping = load_pressure_position_mapping(
                os.path.join(repo_root, f"TouchAnything/configs/pressure_position_mapping_{hand}.json")
            )
            sensor_norm, valid_mask = extract_hand_sensor(frames, hand, pmin, pmax)
            grid = sensors_to_21_grid(sensor_norm, pressure_mapping)
            transform = worker_args[f"{hand}_grid_transform"]
            mano_grid = transform_grid_for_mano(grid, transform)
            deps = worker_deps_left if hand == "left" else worker_deps_right
            continuous = process_sequence_gpu(mano_grid, deps)

            result[f"{hand}_sensor_256_norm"] = sensor_norm
            result[f"{hand}_sensor_valid"] = valid_mask
            result[f"{hand}_pressure_grid"] = grid
            result[f"{hand}_pressure_grid_mano"] = mano_grid
            result[f"{hand}_pressure_continuous_subdiv"] = continuous
            result[f"{hand}_grid_transform"] = np.asarray(transform)
            processed_any = processed_any or bool(np.any(valid_mask))

        if not processed_any:
            return {"path": data_json_path, "status": "no_sensor"}

        tmp_path = out_path + ".tmp.npz"
        np.savez_compressed(tmp_path, **result)

        # Full readback check: np.load on npz is lazy, so opening alone is not enough.
        with np.load(tmp_path, allow_pickle=False) as check_data:
            for key in (
                "left_sensor_256_norm",
                "right_sensor_256_norm",
                "left_sensor_valid",
                "right_sensor_valid",
                "left_pressure_grid",
                "right_pressure_grid",
                "left_pressure_grid_mano",
                "right_pressure_grid_mano",
                "left_pressure_continuous_subdiv",
                "right_pressure_continuous_subdiv",
            ):
                if key in check_data:
                    arr = check_data[key]
                    _ = arr[()] if arr.shape == () else arr[:]

        os.replace(tmp_path, out_path)
        return {"path": data_json_path, "status": "ok"}
    except Exception as exc:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return {"path": data_json_path, "status": "error", "error": str(exc)}


def find_data_json_files(root, exclude_dirs):
    root = os.path.abspath(root)
    excluded = set(exclude_dirs or [])
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in excluded]
        if "data.json" in filenames and "video.mp4" in filenames:
            files.append(os.path.join(dirpath, "data.json"))
    return sorted(files)


def parse_args():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    parser = argparse.ArgumentParser(description="Normalize EgoTactile sensor_256 and generate MANO subdiv Gaussian pressure.")
    parser.add_argument("--egotactile_dir", default="/data1/jiangrui/EgoTactile/Raw_data")
    parser.add_argument("--output_name", default="pressure_grids_egotactile.npz")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--workers_per_gpu", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--pmin", type=float, default=5.0)
    parser.add_argument("--pmax", type=float, default=200.0)
    parser.add_argument("--sigma", type=float, default=0.005)
    parser.add_argument("--left_grid_transform", choices=["none", "flip_lr", "flip_ud", "rot180"], default="none")
    parser.add_argument("--right_grid_transform", choices=["none", "flip_lr", "flip_ud", "rot180"], default="flip_lr")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output npz files.")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: process at most this many data.json files.")
    parser.add_argument("--repo_root", default=repo_root)
    parser.add_argument(
        "--scan_exclude_dirs",
        nargs="*",
        default=sorted(DEFAULT_SCAN_EXCLUDE_DIRS),
        help="Directory names to prune while discovering EgoTactile data.json files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_json_files = find_data_json_files(args.egotactile_dir, args.scan_exclude_dirs)
    if args.limit > 0:
        data_json_files = data_json_files[:args.limit]

    gpus = [int(g.strip()) for g in args.gpu.split(",") if g.strip()]
    num_workers = max(1, len(gpus) * args.workers_per_gpu)

    print("🚀 EgoTactile normalization + MANO Gaussian generation")
    print(f"   data root: {args.egotactile_dir}")
    print(f"   data.json files: {len(data_json_files)}")
    print(f"   output name: {args.output_name}")
    print(f"   GPUs: {gpus}")
    print(f"   workers: {num_workers}")
    print(f"   pmin/pmax: {args.pmin}/{args.pmax}")
    print(f"   left/right transforms: {args.left_grid_transform}/{args.right_grid_transform}")

    print("⏳ Precomputing LEFT topology...")
    deps_left_cpu = load_ta_mesh_and_compute_dist_cpu(args.repo_root, hand="left", sigma=args.sigma)
    print("⏳ Precomputing RIGHT topology...")
    deps_right_cpu = load_ta_mesh_and_compute_dist_cpu(args.repo_root, hand="right", sigma=args.sigma)

    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    gpu_queue = manager.Queue()
    for gpu_id in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_queue.put(gpu_id)

    args_dict = {
        "repo_root": args.repo_root,
        "output_name": args.output_name,
        "force": args.force,
        "pmin": args.pmin,
        "pmax": args.pmax,
        "batch_size": args.batch_size,
        "left_grid_transform": args.left_grid_transform,
        "right_grid_transform": args.right_grid_transform,
    }

    counts = {}
    errors = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(gpu_queue, deps_left_cpu, deps_right_cpu, args_dict),
    ) as executor:
        futures = [executor.submit(process_data_json_worker, path) for path in data_json_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing EgoTactile"):
            item = future.result()
            status = item["status"]
            counts[status] = counts.get(status, 0) + 1
            if status == "error" and len(errors) < 10:
                errors.append(item)

    print("✅ Done.")
    print("Summary:", counts)
    if errors:
        print("First errors:")
        for item in errors:
            print(f"  {item['path']}: {item['error']}")


if __name__ == "__main__":
    main()
