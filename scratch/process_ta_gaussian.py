import os
import json
import glob
import numpy as np
import trimesh
import torch
import argparse
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# ----------------- 全局的 Worker 本地变量 -----------------
worker_device = None
worker_deps_left = None
worker_deps_right = None
worker_left_grid_transform = "none"
worker_right_grid_transform = "none"

class DepContainer:
    pass

def transform_pressure_grid_for_mano(p_numpy, transform):
    if transform == "none":
        return p_numpy
    if transform == "flip_lr":
        return np.flip(p_numpy, axis=2).copy()
    if transform == "flip_ud":
        return np.flip(p_numpy, axis=1).copy()
    if transform == "rot180":
        return np.flip(np.flip(p_numpy, axis=1), axis=2).copy()
    raise ValueError(f"Unknown grid transform: {transform}")

def load_ta_mesh_and_compute_dist_cpu(repo_root, hand='left', sigma=0.005):
    """仅在主进程计算一次 CPU 缓存版本的拓扑，节省启动时间和内存"""
    obj_path = os.path.join(repo_root, "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj")
    palm_faces_path = os.path.join(repo_root, "opentouch/preprocess/scratch/auto_calibrated_palm_subdiv_faces.json")
    mapping_path = os.path.join(repo_root, f"TouchAnything/scripts/tools/mano_visualization/ta_to_mano_mapping_{hand}_visual.json")

    mesh = trimesh.load(obj_path, process=False)
    mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mano_faces = np.asarray(mesh.faces, dtype=np.int32)
    V_total = mano_vertices.shape[0]

    with open(mapping_path, "r") as f:
        mapping_data = json.load(f)

    layout = mapping_data["positions"]

    valid_nodes = {}
    for nid, info in layout.items():
        vids = info.get("mano_vid", [])
        vids = [v for v in vids if v < V_total]
        if len(vids) > 0:
            center = np.mean(mano_vertices[vids], axis=0)
            valid_nodes[nid] = {"center": center, "vids": vids}

    V = V_total
    edges = set()
    for a, b, c in mano_faces.astype(np.int64):
        edges.update({(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))})

    rows, cols, data = [], [], []
    for i, j in edges:
        d = np.linalg.norm(mano_vertices[i] - mano_vertices[j])
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([d, d])

    pure_graph = csr_matrix((data, (rows, cols)), shape=(V, V))

    node_keys = list(valid_nodes.keys())
    K = len(node_keys)
    dist_matrix = np.zeros((K, V), dtype=np.float32)

    for k, nid in enumerate(node_keys):
        center = valid_nodes[nid]["center"]
        vids = valid_nodes[nid]["vids"]
        D_vids = shortest_path(pure_graph, directed=False, indices=vids)
        jump_dists = np.linalg.norm(mano_vertices[vids] - center, axis=1)
        D_k = np.min(D_vids + jump_dists[:, np.newaxis], axis=0)
        dist_matrix[k, :] = D_k

    two_sig2 = 2.0 * (sigma * sigma)
    weights = np.exp(-(dist_matrix**2) / two_sig2)

    palm_vertices_set = set()
    with open(palm_faces_path, "r") as f:
        palm_data = json.load(f)
    for fid in palm_data["group_negative"]["face_indices"]:
        if fid < len(mano_faces):
            for vid in mano_faces[fid]:
                if vid < V_total:
                    palm_vertices_set.add(vid)
    palm_vertices = list(palm_vertices_set)

    deps = DepContainer()
    deps.V_total = V_total

    grid_r = []
    grid_c = []
    for nid in node_keys:
        r, c = map(int, nid.split(','))
        grid_r.append(r)
        grid_c.append(c)

    deps.valid_rows_cpu = torch.tensor(grid_r, dtype=torch.long)
    deps.valid_cols_cpu = torch.tensor(grid_c, dtype=torch.long)
    deps.weights_tensor_cpu = torch.tensor(weights, dtype=torch.float32)
    deps.palm_vertices_tensor_cpu = torch.tensor(palm_vertices, dtype=torch.long)

    return deps

def move_deps_to_device(deps, device):
    """将 CPU 依赖加载到当前 worker 的专属 GPU 上"""
    d = DepContainer()
    d.V_total = deps.V_total
    d.valid_rows = deps.valid_rows_cpu.to(device)
    d.valid_cols = deps.valid_cols_cpu.to(device)
    d.weights_tensor = deps.weights_tensor_cpu.to(device)
    d.palm_vertices_tensor = deps.palm_vertices_tensor_cpu.to(device)
    return d

def init_worker_queue(gpu_queue, deps_left_cpu, deps_right_cpu, left_grid_transform, right_grid_transform):
    """子进程初始化函数：领取一个 GPU，并设置孤立环境"""
    global worker_device, worker_deps_left, worker_deps_right
    global worker_left_grid_transform, worker_right_grid_transform
    gpu_id = gpu_queue.get()

    # 强制隔离：这个子进程只看得见分配给它的那张卡，所以即使它调 cuda:0，也是落在那张卡上！
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    worker_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    worker_deps_left = move_deps_to_device(deps_left_cpu, worker_device)
    worker_deps_right = move_deps_to_device(deps_right_cpu, worker_device)
    worker_left_grid_transform = left_grid_transform
    worker_right_grid_transform = right_grid_transform

def process_sequence_gpu(p_numpy, deps):
    T = p_numpy.shape[0]
    V = deps.V_total
    if T == 0:
        return np.zeros((0, V), dtype=np.float32)

    with torch.no_grad():
        p_tensor = torch.tensor(p_numpy, dtype=torch.float32, device=worker_device)
        p_tensor = torch.clamp(p_tensor, 0.0, 1.0)

        active = p_tensor[:, deps.valid_rows, deps.valid_cols]

        # 减小 Batch Size，防止 OOM (2000 会占用单进程 6GB 显存，多进程必然炸显存)
        B = 500
        out = torch.zeros((T, V), dtype=torch.float32, device=worker_device)

        for i in range(0, T, B):
            active_b = active[i:i+B].unsqueeze(2)
            w_b = deps.weights_tensor.unsqueeze(0)

            palm_vals, _ = torch.max(active_b * w_b, dim=1)

            masked = torch.zeros_like(palm_vals)
            masked[:, deps.palm_vertices_tensor] = palm_vals[:, deps.palm_vertices_tensor]
            out[i:i+B] = torch.clamp(masked, 0.0, 1.0)

        return out.cpu().numpy()

def process_npz_worker(npz_path):
    try:
        data = np.load(npz_path)
        out_dict = dict(data)
        modified = False

        if 'left_pressure_grid' in data:
            p_l = data['left_pressure_grid']
            p_l[np.isnan(p_l)] = 0.0
            p_l = transform_pressure_grid_for_mano(p_l, worker_left_grid_transform)
            c_l = process_sequence_gpu(p_l, worker_deps_left)
            out_dict['left_pressure_continuous_subdiv'] = c_l
            modified = True

        if 'right_pressure_grid' in data:
            p_r = data['right_pressure_grid']
            p_r[np.isnan(p_r)] = 0.0
            p_r = transform_pressure_grid_for_mano(p_r, worker_right_grid_transform)
            c_r = process_sequence_gpu(p_r, worker_deps_right)
            out_dict['right_pressure_continuous_subdiv'] = c_r
            modified = True

        if modified:
            np.savez_compressed(npz_path, **out_dict)

    except Exception as e:
        pass


def needs_continuous_subdiv(npz_path):
    try:
        data = np.load(npz_path)
        try:
            has_left_grid = "left_pressure_grid" in data
            has_right_grid = "right_pressure_grid" in data
            missing_left = has_left_grid and "left_pressure_continuous_subdiv" not in data
            missing_right = has_right_grid and "right_pressure_continuous_subdiv" not in data
            return missing_left or missing_right
        finally:
            data.close()
    except Exception:
        return False


def discover_npz_under_task(task_dir):
    npz_files = []
    try:
        with os.scandir(task_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                npz_path = os.path.join(entry.path, "pressure_grids.npz")
                if os.path.exists(npz_path):
                    npz_files.append(npz_path)
    except OSError:
        pass
    return npz_files


def discover_touchanything_npz(root, scan_workers=32):
    task_dirs = []
    excluded = {"extracted_frames", "metadata", "__pycache__"}
    try:
        with os.scandir(root) as scenes:
            for scene in scenes:
                if scene.name in excluded or not scene.is_dir(follow_symlinks=False):
                    continue
                try:
                    with os.scandir(scene.path) as tasks:
                        for task in tasks:
                            if task.name in excluded or not task.is_dir(follow_symlinks=False):
                                continue
                            task_dirs.append(task.path)
                except OSError:
                    continue
    except OSError:
        return []

    if not task_dirs:
        return []

    npz_files = []
    scan_workers = max(1, int(scan_workers))
    with ThreadPoolExecutor(max_workers=min(scan_workers, len(task_dirs))) as executor:
        futures = [executor.submit(discover_npz_under_task, task_dir) for task_dir in task_dirs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Discovering TouchAnything NPZ"):
            npz_files.extend(future.result())
    return sorted(npz_files)

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Process TouchAnything NPZ with multiple GPUs")
    parser.add_argument("--gpu", type=str, default="0", help="指定多卡的 ID，用逗号分隔，例如 '0,1,2,3'")
    parser.add_argument("--workers_per_gpu", type=int, default=12, help="每张卡挂载的进程数 (掩盖 I/O 延迟)")
    parser.add_argument("--ta_dir", default="/data1/jiangrui/EgoTouch/", help="TouchAnything/EgoTouch root directory.")
    parser.add_argument("--clip_dir", action="append", default=[], help="Specific clip dir to process. Can be passed multiple times.")
    parser.add_argument(
        "--only_missing_continuous",
        action="store_true",
        help="Only process npz files missing left/right_pressure_continuous_subdiv.",
    )
    parser.add_argument("--check_workers", type=int, default=32, help="Threads for filtering/checking npz files.")
    parser.add_argument("--repo_root", default=repo_root, help="Repository root for mapping and mesh assets.")
    parser.add_argument(
        "--left_grid_transform",
        choices=["none", "flip_lr", "flip_ud", "rot180"],
        default="none",
        help="Transform left_pressure_grid before TA->MANO visual mapping.",
    )
    parser.add_argument(
        "--right_grid_transform",
        choices=["none", "flip_lr", "flip_ud", "rot180"],
        default="none",
        help="Transform right_pressure_grid before TA->MANO visual mapping.",
    )
    args = parser.parse_args()

    gpus = [int(g.strip()) for g in args.gpu.split(',')]
    num_workers = len(gpus) * args.workers_per_gpu

    print(f"🚀 启动 TouchAnything 多卡拓扑重映射引擎")
    print(f"   => 挂载的 GPU 列表: {gpus}")
    print(f"   => 进程池规模: {num_workers} 个高并发 Worker ({args.workers_per_gpu} 进程/GPU)")
    print(f"   => LEFT grid transform: {args.left_grid_transform}")
    print(f"   => RIGHT grid transform: {args.right_grid_transform}")
    print(f"   => TouchAnything root: {args.ta_dir}")
    print(f"   => Repo root: {args.repo_root}")

    print("\n⏳ [主进程] 预计算 LEFT hand 拓扑网络及距离权重 (建立 CPU 缓存)...")
    deps_left_cpu = load_ta_mesh_and_compute_dist_cpu(args.repo_root, hand='left')

    print("⏳ [主进程] 预计算 RIGHT hand 拓扑网络及距离权重 (建立 CPU 缓存)...")
    deps_right_cpu = load_ta_mesh_and_compute_dist_cpu(args.repo_root, hand='right')

    if args.clip_dir:
        npz_files = [os.path.join(clip_dir, "pressure_grids.npz") for clip_dir in args.clip_dir]
        npz_files = [path for path in npz_files if os.path.exists(path)]
    else:
        npz_files = discover_touchanything_npz(args.ta_dir, scan_workers=args.check_workers)
    print(f"   => discovered NPZ files: {len(npz_files)}")
    if args.only_missing_continuous:
        before_filter = len(npz_files)
        check_workers = max(1, int(args.check_workers))
        if check_workers == 1:
            npz_files = [path for path in npz_files if needs_continuous_subdiv(path)]
        else:
            keep = []
            with ThreadPoolExecutor(max_workers=check_workers) as executor:
                futures = {executor.submit(needs_continuous_subdiv, path): path for path in npz_files}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Checking NPZ"):
                    if future.result():
                        keep.append(futures[future])
            npz_files = keep
        print(f"   => only_missing_continuous: {before_filter} -> {len(npz_files)} npz files")

    print(f"\n✅ 图拓扑计算完成！准备分发任务给子进程，并行处理 {len(npz_files)} 个样本...")

    # 使用多进程队列派发 GPU 身份牌
    ctx = multiprocessing.get_context('spawn')
    m = ctx.Manager()
    gpu_queue = m.Queue()
    for gpu_id in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_queue.put(gpu_id)

    with ProcessPoolExecutor(max_workers=num_workers,
                             mp_context=ctx,
                             initializer=init_worker_queue,
                             initargs=(
                                 gpu_queue,
                                 deps_left_cpu,
                                 deps_right_cpu,
                                 args.left_grid_transform,
                                 args.right_grid_transform,
                             )) as executor:
        futures = [executor.submit(process_npz_worker, f) for f in npz_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing NPZ"):
            pass

    print("✅ 全量数据处理完毕！可以运行 compare_pressure_dist.py 检查新的 Subdiv 数据了！")

if __name__ == '__main__':
    main()
