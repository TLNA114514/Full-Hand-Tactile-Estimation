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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ----------------- 全局的 Worker 本地变量 -----------------
worker_device = None
worker_deps_left = None
worker_deps_right = None

class DepContainer:
    pass

def load_ta_mesh_and_compute_dist_cpu(hand='left', sigma=0.005):
    """仅在主进程计算一次 CPU 缓存版本的拓扑，节省启动时间和内存"""
    base_dir = "/code/users/jiangrui/Full-Hand-Tactile-Estimation"
    obj_path = os.path.join(base_dir, "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj")
    palm_faces_path = os.path.join(base_dir, "opentouch/preprocess/scratch/auto_calibrated_palm_subdiv_faces.json")
    mapping_path = os.path.join(base_dir, f"TouchAnything/scripts/tools/mano_visualization/ta_to_mano_mapping_{hand}_visual.json")
    
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

def init_worker_queue(gpu_queue, deps_left_cpu, deps_right_cpu):
    """子进程初始化函数：领取一个 GPU，并设置孤立环境"""
    global worker_device, worker_deps_left, worker_deps_right
    gpu_id = gpu_queue.get()
    
    # 强制隔离：这个子进程只看得见分配给它的那张卡，所以即使它调 cuda:0，也是落在那张卡上！
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    worker_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    worker_deps_left = move_deps_to_device(deps_left_cpu, worker_device)
    worker_deps_right = move_deps_to_device(deps_right_cpu, worker_device)

def process_sequence_gpu(p_numpy, deps):
    T = p_numpy.shape[0]
    V = deps.V_total
    if T == 0:
        return np.zeros((0, V), dtype=np.float32)
        
    with torch.no_grad():
        p_tensor = torch.tensor(p_numpy, dtype=torch.float32, device=worker_device)
        p_tensor = torch.clamp(p_tensor, 0.0, 1.0)
        
        active = p_tensor[:, deps.valid_rows, deps.valid_cols] 
        
        B = 2000
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
            c_l = process_sequence_gpu(p_l, worker_deps_left)
            out_dict['left_pressure_continuous_subdiv'] = c_l
            modified = True
            
        if 'right_pressure_grid' in data:
            p_r = data['right_pressure_grid']
            p_r[np.isnan(p_r)] = 0.0
            c_r = process_sequence_gpu(p_r, worker_deps_right)
            out_dict['right_pressure_continuous_subdiv'] = c_r
            modified = True
            
        if modified:
            np.savez_compressed(npz_path, **out_dict)
            
    except Exception as e:
        pass

def main():
    parser = argparse.ArgumentParser(description="Process TouchAnything NPZ with multiple GPUs")
    parser.add_argument("--gpu", type=str, default="0", help="指定多卡的 ID，用逗号分隔，例如 '0,1,2,3'")
    parser.add_argument("--workers_per_gpu", type=int, default=12, help="每张卡挂载的进程数 (掩盖 I/O 延迟)")
    args = parser.parse_args()
    
    gpus = [int(g.strip()) for g in args.gpu.split(',')]
    num_workers = len(gpus) * args.workers_per_gpu
    
    print(f"🚀 启动 TouchAnything 多卡拓扑重映射引擎")
    print(f"   => 挂载的 GPU 列表: {gpus}")
    print(f"   => 进程池规模: {num_workers} 个高并发 Worker ({args.workers_per_gpu} 进程/GPU)")
    
    print("\n⏳ [主进程] 预计算 LEFT hand 拓扑网络及距离权重 (建立 CPU 缓存)...")
    deps_left_cpu = load_ta_mesh_and_compute_dist_cpu(hand='left')
    
    print("⏳ [主进程] 预计算 RIGHT hand 拓扑网络及距离权重 (建立 CPU 缓存)...")
    deps_right_cpu = load_ta_mesh_and_compute_dist_cpu(hand='right')
    
    ta_dir = "/data/jiangrui/EgoTouch/"
    npz_files = glob.glob(os.path.join(ta_dir, "**", "pressure_grids.npz"), recursive=True)
    
    print(f"\n✅ 图拓扑计算完成！准备分发任务给子进程，并行处理 {len(npz_files)} 个样本...")
    
    # 使用多进程队列派发 GPU 身份牌
    m = multiprocessing.Manager()
    gpu_queue = m.Queue()
    for gpu_id in gpus:
        for _ in range(args.workers_per_gpu):
            gpu_queue.put(gpu_id)
            
    with ProcessPoolExecutor(max_workers=num_workers, 
                             initializer=init_worker_queue, 
                             initargs=(gpu_queue, deps_left_cpu, deps_right_cpu)) as executor:
        futures = [executor.submit(process_npz_worker, f) for f in npz_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing NPZ"):
            pass
            
    print("✅ 全量数据处理完毕！可以运行 compare_pressure_dist.py 检查新的 Subdiv 数据了！")

if __name__ == '__main__':
    main()
