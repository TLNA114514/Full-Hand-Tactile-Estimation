import os
import json
import glob
import numpy as np
import h5py
import trimesh
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[{device.type.upper()}] PyTorch 硬件加速已就绪" + ("!" if device.type == "cuda" else " (回退到 CPU 模式)"))

def load_mesh_and_compute_dist(res_type, sigma=0.005):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if res_type == "low_res":
        obj_path = os.path.join(base_dir, "opentouch/preprocess/scratch/mano_right_neutral.obj")
        palm_faces_path = os.path.join(base_dir, "opentouch/preprocess/scratch/auto_calibrated_palm_faces.json")
        layout_path = os.path.join(base_dir, "opentouch/preprocess/scratch/handLayoutNewest_meshid_lowres.json")
    else:
        obj_path = os.path.join(base_dir, "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj")
        palm_faces_path = os.path.join(base_dir, "opentouch/preprocess/scratch/auto_calibrated_palm_subdiv_faces.json")
        layout_path = os.path.join(base_dir, "opentouch/preprocess/scratch/handLayoutNewest_meshid.json")
        
    mesh = trimesh.load(obj_path, process=False)
    mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mano_faces = np.asarray(mesh.faces, dtype=np.int32)
    V_total = mano_vertices.shape[0]

    if not os.path.exists(layout_path):
        layout_path = os.path.join(base_dir, "opentouch/preprocess/scratch/handLayoutNewest_meshid.json")
        
    with open(layout_path, "r") as f:
        layout_data = json.load(f)
    layout = layout_data["positions"]
    erased_nodes = set(layout_data.get("erasedNodes", []))

    valid_nodes = {}
    for nid, info in layout.items():
        if nid in erased_nodes:
            continue
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

    # Calculate Weights ahead of time
    two_sig2 = 2.0 * (sigma * sigma)
    weights = np.exp(-(dist_matrix**2) / two_sig2)

    # Load palm vertices
    palm_vertices_set = set()
    if res_type == "low_res":
        with open(palm_faces_path, "r") as f:
            palm_data = json.load(f)
        for triplet in palm_data["group_negative"]["face_triplets"]:
            for vid in triplet:
                if vid < V_total:
                    palm_vertices_set.add(vid)
    else:
        with open(palm_faces_path, "r") as f:
            palm_data = json.load(f)
        for fid in palm_data["group_negative"]["face_indices"]:
            if fid < len(mano_faces):
                for vid in mano_faces[fid]:
                    if vid < V_total:
                        palm_vertices_set.add(vid)
    palm_vertices = list(palm_vertices_set)

    return mano_vertices, palm_vertices, dist_matrix, node_keys, weights

class DepContainer:
    pass

def prepare_gpu_deps(deps):
    rows, cols = [], []
    valid_k = []
    for k, nid in enumerate(deps.node_keys):
        r, c = map(int, nid.split('-'))
        if r < 16 and c < 16:
            rows.append(r)
            cols.append(c)
            valid_k.append(k)
            
    deps.valid_rows = torch.tensor(rows, dtype=torch.long, device=device)
    deps.valid_cols = torch.tensor(cols, dtype=torch.long, device=device)
    deps.weights_tensor = torch.tensor(deps.weights[valid_k], dtype=torch.float32, device=device)
    deps.palm_vertices_tensor = torch.tensor(deps.palm_vertices, dtype=torch.long, device=device)
    deps.V_total = deps.mano_vertices.shape[0]

def process_sequence_gpu(p_numpy, deps):
    T = p_numpy.shape[0]
    V = deps.V_total
    if T == 0:
        return np.zeros((0, V), dtype=np.float32)
        
    with torch.no_grad():
        p_tensor = torch.tensor(p_numpy, dtype=torch.float32, device=device)
        p_norm = torch.clamp((3072.0 - p_tensor) / 3072.0, 0.0, 1.0)
        
        active = p_norm[:, deps.valid_rows, deps.valid_cols] # (T, K_valid)
        
        # Batch over T to prevent massive memory spikes
        B = 2000
        out = torch.zeros((T, V), dtype=torch.float32, device=device)
        
        for i in range(0, T, B):
            active_b = active[i:i+B].unsqueeze(2) # (B, K_valid, 1)
            w_b = deps.weights_tensor.unsqueeze(0) # (1, K_valid, V)
            
            # Broadcast multiply and take max over sensors (dim 1)
            palm_vals, _ = torch.max(active_b * w_b, dim=1) # (B, V)
            
            # Mask out non-palm vertices
            masked = torch.zeros_like(palm_vals)
            masked[:, deps.palm_vertices_tensor] = palm_vals[:, deps.palm_vertices_tensor]
            out[i:i+B] = torch.clamp(masked, 0.0, 1.0)
            
        return out.cpu().numpy()

def process_h5_file(filepath, deps_low, deps_sub):
    try:
        with h5py.File(filepath, "r+") as f:
            if "data" not in f:
                return
            data_group = f["data"]
            
            needs_processing = False
            for demo_name in data_group.keys():
                demo = data_group[demo_name]
                if "right_pressure" in demo or "left_pressure" in demo:
                    needs_processing = True
                    break
                    
            if not needs_processing:
                return
                
            for demo_name in data_group.keys():
                demo = data_group[demo_name]
                
                for side in ["right", "left"]:
                    key = f"{side}_pressure"
                    if key in demo:
                        p = demo[key][:]
                        p_cont_low = process_sequence_gpu(p, deps_low)
                        p_cont_sub = process_sequence_gpu(p, deps_sub)
                        
                        cont_key = f"{key}_continuous"
                        sub_key = f"{key}_continuous_subdiv"
                        
                        if cont_key in demo:
                            del demo[cont_key]
                        demo.create_dataset(cont_key, data=p_cont_low, compression="gzip")
                        
                        if sub_key in demo:
                            del demo[sub_key]
                        demo.create_dataset(sub_key, data=p_cont_sub, compression="gzip")
                        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_meta(mf, deps_low, deps_sub):
    try:
        with open(mf, "r") as f:
            data = json.load(f)
            
        hdf5_data = data.get("original_hdf5_data", {})
        modified = False
        
        for key in ["right_pressure", "left_pressure"]:
            if key in hdf5_data:
                pressure = np.array(hdf5_data[key])
                
                if pressure.ndim == 2:
                    p = pressure[np.newaxis, ...] # (1, 16, 16)
                    c_low = process_sequence_gpu(p, deps_low)[0]
                    c_sub = process_sequence_gpu(p, deps_sub)[0]
                    hdf5_data[f"{key}_continuous"] = c_low.tolist()
                    hdf5_data[f"{key}_continuous_subdiv"] = c_sub.tolist()
                    modified = True
                elif pressure.ndim == 3:
                    c_low = process_sequence_gpu(pressure, deps_low)
                    c_sub = process_sequence_gpu(pressure, deps_sub)
                    hdf5_data[f"{key}_continuous"] = c_low.tolist()
                    hdf5_data[f"{key}_continuous_subdiv"] = c_sub.tolist()
                    modified = True
                
        if modified:
            with open(mf, "w") as f:
                json.dump(data, f)
    except Exception as e:
        print(f"Error processing {mf}: {e}")

def process_extracted_dataset(dataset_dir, deps_low, deps_sub):
    meta_files = glob.glob(os.path.join(dataset_dir, "*", "*", "meta.json"))
    
    # 采用多线程并行处理海量小文件，极大打破单线程 I/O 瓶颈
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_single_meta, mf, deps_low, deps_sub) for mf in meta_files]
        for _ in tqdm(as_completed(futures), total=len(meta_files), desc="Processing meta.json (Multi-threaded)"):
            pass

def load_all_deps():
    deps_low = DepContainer()
    deps_sub = DepContainer()
    
    print("⏳ [Low_Res] 正在预计算 Dijkstra 距离矩阵与权重场...")
    (deps_low.mano_vertices, deps_low.palm_vertices, 
     deps_low.dist_matrix, deps_low.node_keys, deps_low.weights) = load_mesh_and_compute_dist("low_res")
     
    print("⏳ [Subdiv] 正在预计算 Dijkstra 距离矩阵与权重场...")
    (deps_sub.mano_vertices, deps_sub.palm_vertices,
     deps_sub.dist_matrix, deps_sub.node_keys, deps_sub.weights) = load_mesh_and_compute_dist("subdiv")
     
    prepare_gpu_deps(deps_low)
    prepare_gpu_deps(deps_sub)
    
    print("✅ 双分辨率图拓扑及 GPU 张量预备完毕！")
    return deps_low, deps_sub

def main():
    print("🚀 启动完美双轨道数据注入流 (PyTorch GPU 加速版)...")
    deps_low, deps_sub = load_all_deps()
    
    # 1. /data/jiangrui/OpenTouch Data/data/
    h5_dir = "/data/jiangrui/OpenTouch Data/data/"
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5")) + glob.glob(os.path.join(h5_dir, "*.hdf5"))
    if len(h5_files) > 0:
        for h5f in tqdm(h5_files, desc="Processing HDF5 data files"):
            process_h5_file(h5f, deps_low, deps_sub)
            
    # 2. /data/jiangrui/OpenTouch Data/extracted_dataset/
    ext_dir = "/data/jiangrui/OpenTouch Data/extracted_dataset/"
    if os.path.exists(ext_dir):
        process_extracted_dataset(ext_dir, deps_low, deps_sub)

if __name__ == "__main__":
    main()
