import os
import json
import glob
import numpy as np
import trimesh
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from concurrent.futures import ThreadPoolExecutor, as_completed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DepContainer:
    pass

def load_ta_mesh_and_compute_dist(hand='left', sigma=0.005):
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
            
    # Dijkstra graph
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
    
    # Palm vertices
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
    deps.mano_vertices = mano_vertices
    deps.palm_vertices = palm_vertices
    deps.dist_matrix = dist_matrix
    deps.node_keys = node_keys
    deps.weights = weights
    deps.V_total = V_total
    
    grid_r = []
    grid_c = []
    for nid in node_keys:
        r, c = map(int, nid.split(','))
        grid_r.append(r)
        grid_c.append(c)
        
    deps.valid_rows = torch.tensor(grid_r, dtype=torch.long, device=device)
    deps.valid_cols = torch.tensor(grid_c, dtype=torch.long, device=device)
    deps.weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    deps.palm_vertices_tensor = torch.tensor(palm_vertices, dtype=torch.long, device=device)
    
    return deps

def process_sequence_gpu(p_numpy, deps):
    T = p_numpy.shape[0]
    V = deps.V_total
    if T == 0:
        return np.zeros((0, V), dtype=np.float32)
        
    with torch.no_grad():
        # TouchAnything 的压力已经是 [0, 1] 的形式，直接读取即可！
        p_tensor = torch.tensor(p_numpy, dtype=torch.float32, device=device)
        p_tensor = torch.clamp(p_tensor, 0.0, 1.0)
        
        # 按照 valid_rows 和 valid_cols 抽取有效的像素点 => shape: (T, K)
        active = p_tensor[:, deps.valid_rows, deps.valid_cols] 
        
        B = 2000
        out = torch.zeros((T, V), dtype=torch.float32, device=device)
        
        for i in range(0, T, B):
            active_b = active[i:i+B].unsqueeze(2) # (B, K, 1)
            w_b = deps.weights_tensor.unsqueeze(0) # (1, K, V)
            
            palm_vals, _ = torch.max(active_b * w_b, dim=1) # (B, V)
            
            masked = torch.zeros_like(palm_vals)
            masked[:, deps.palm_vertices_tensor] = palm_vals[:, deps.palm_vertices_tensor]
            out[i:i+B] = torch.clamp(masked, 0.0, 1.0)
            
        return out.cpu().numpy()

def process_npz(npz_path, deps_left, deps_right):
    try:
        data = np.load(npz_path)
        out_dict = dict(data)
        modified = False
        
        if 'left_pressure_grid' in data:
            p_l = data['left_pressure_grid']
            p_l[np.isnan(p_l)] = 0.0
            c_l = process_sequence_gpu(p_l, deps_left)
            out_dict['left_pressure_continuous_subdiv'] = c_l
            modified = True
            
        if 'right_pressure_grid' in data:
            p_r = data['right_pressure_grid']
            p_r[np.isnan(p_r)] = 0.0
            c_r = process_sequence_gpu(p_r, deps_right)
            out_dict['right_pressure_continuous_subdiv'] = c_r
            modified = True
            
        if modified:
            np.savez_compressed(npz_path, **out_dict)
            
    except Exception as e:
        print(f"Error processing {npz_path}: {e}")

def main():
    print("🚀 启动 TouchAnything (21x21 -> MANO Subdiv) Gaussian 平滑融合...")
    
    print("⏳ 预计算 LEFT hand 拓扑网络及距离权重...")
    deps_left = load_ta_mesh_and_compute_dist(hand='left')
    
    print("⏳ 预计算 RIGHT hand 拓扑网络及距离权重...")
    deps_right = load_ta_mesh_and_compute_dist(hand='right')
    
    ta_dir = "/data/jiangrui/EgoTouch/"
    npz_files = glob.glob(os.path.join(ta_dir, "**", "pressure_grids.npz"), recursive=True)
    
    print(f"✅ 图拓扑计算完成！准备对 {len(npz_files)} 个 TouchAnything 样本注入 continuous_subdiv 数据。")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_npz, f, deps_left, deps_right) for f in npz_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing NPZ"):
            pass

if __name__ == '__main__':
    main()
