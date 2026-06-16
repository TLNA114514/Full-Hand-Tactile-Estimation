import os
import json
import glob
import numpy as np
import h5py
import trimesh
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def load_mesh_and_compute_dist(res_type):
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

    # Load palm vertices
    palm_vertices_set = set()
    if res_type == "low_res":
        with open(palm_faces_path, "r") as f:
            palm_data = json.load(f)
        for triplet in palm_data["group_positive"]["face_triplets"]:
            for vid in triplet:
                if vid < V_total:
                    palm_vertices_set.add(vid)
    else:
        with open(palm_faces_path, "r") as f:
            palm_data = json.load(f)
        for fid in palm_data["group_positive"]["face_indices"]:
            if fid < len(mano_faces):
                for vid in mano_faces[fid]:
                    if vid < V_total:
                        palm_vertices_set.add(vid)
    palm_vertices = list(palm_vertices_set)

    return mano_vertices, palm_vertices, valid_nodes, dist_matrix, node_keys

def compute_continuous_pressure(pressure16, valid_nodes, mano_vertices, palm_vertices, dist_matrix, node_keys, sigma=0.005):
    p_norm_matrix = np.clip((3072.0 - pressure16) / 3072.0, 0.0, 1.0)
    
    active_pressures = []
    for nid in node_keys:
        r, c = map(int, nid.split('-'))
        if r < pressure16.shape[0] and c < pressure16.shape[1]:
            active_pressures.append(p_norm_matrix[r, c])
        else:
            active_pressures.append(0.0)
            
    active_pressures = np.array(active_pressures, dtype=np.float32)
    two_sig2 = 2.0 * (sigma * sigma)
    
    weights = np.exp(-(dist_matrix**2) / two_sig2) # (K, V)
    palm_vals = np.max(weights * active_pressures[:, np.newaxis], axis=0) # (V,)
    
    vert_vals = np.zeros(mano_vertices.shape[0], dtype=np.float32)
    vert_vals[palm_vertices] = palm_vals[palm_vertices]
    
    return np.clip(vert_vals, 0.0, 1.0)

class DepContainer:
    pass

def process_h5_file(filepath, deps_low, deps_sub, sigma=0.005):
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
                        p_cont_low = []
                        p_cont_sub = []
                        for i in range(p.shape[0]):
                            c_low = compute_continuous_pressure(p[i], deps_low.valid_nodes, deps_low.mano_vertices, deps_low.palm_vertices, deps_low.dist_matrix, deps_low.node_keys, sigma)
                            c_sub = compute_continuous_pressure(p[i], deps_sub.valid_nodes, deps_sub.mano_vertices, deps_sub.palm_vertices, deps_sub.dist_matrix, deps_sub.node_keys, sigma)
                            p_cont_low.append(c_low)
                            p_cont_sub.append(c_sub)
                            
                        p_cont_low = np.array(p_cont_low, dtype=np.float32)
                        p_cont_sub = np.array(p_cont_sub, dtype=np.float32)
                        
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

def process_extracted_dataset(dataset_dir, deps_low, deps_sub, sigma=0.005):
    meta_files = glob.glob(os.path.join(dataset_dir, "*", "*", "meta.json"))
    for mf in tqdm(meta_files, desc="Processing extracted dataset"):
        try:
            with open(mf, "r") as f:
                data = json.load(f)
                
            hdf5_data = data.get("original_hdf5_data", {})
            modified = False
            has_pressure = False
            
            for key in ["right_pressure", "left_pressure"]:
                if key in hdf5_data:
                    has_pressure = True
                    pressure = np.array(hdf5_data[key])
                    
                    if pressure.ndim == 2:
                        c_low = compute_continuous_pressure(pressure, deps_low.valid_nodes, deps_low.mano_vertices, deps_low.palm_vertices, deps_low.dist_matrix, deps_low.node_keys, sigma)
                        c_sub = compute_continuous_pressure(pressure, deps_sub.valid_nodes, deps_sub.mano_vertices, deps_sub.palm_vertices, deps_sub.dist_matrix, deps_sub.node_keys, sigma)
                        hdf5_data[f"{key}_continuous"] = c_low.tolist()
                        hdf5_data[f"{key}_continuous_subdiv"] = c_sub.tolist()
                        modified = True
                    elif pressure.ndim == 3:
                        c_low_list = []
                        c_sub_list = []
                        for p in pressure:
                            c_low_list.append(compute_continuous_pressure(p, deps_low.valid_nodes, deps_low.mano_vertices, deps_low.palm_vertices, deps_low.dist_matrix, deps_low.node_keys, sigma).tolist())
                            c_sub_list.append(compute_continuous_pressure(p, deps_sub.valid_nodes, deps_sub.mano_vertices, deps_sub.palm_vertices, deps_sub.dist_matrix, deps_sub.node_keys, sigma).tolist())
                        hdf5_data[f"{key}_continuous"] = c_low_list
                        hdf5_data[f"{key}_continuous_subdiv"] = c_sub_list
                        modified = True
                    
            if modified:
                with open(mf, "w") as f:
                    json.dump(data, f)
        except Exception as e:
            print(f"Error processing {mf}: {e}")

def load_all_deps():
    deps_low = DepContainer()
    deps_sub = DepContainer()
    
    print("⏳ [Low_Res] 正在预计算 Dijkstra 距离矩阵...")
    (deps_low.mano_vertices, deps_low.palm_vertices, deps_low.valid_nodes, 
     deps_low.dist_matrix, deps_low.node_keys) = load_mesh_and_compute_dist("low_res")
     
    print("⏳ [Subdiv] 正在预计算 Dijkstra 距离矩阵...")
    (deps_sub.mano_vertices, deps_sub.palm_vertices, deps_sub.valid_nodes, 
     deps_sub.dist_matrix, deps_sub.node_keys) = load_mesh_and_compute_dist("subdiv")
     
    print("✅ 双分辨率图拓扑准备完毕！")
    return deps_low, deps_sub

def main():
    print("🚀 启动完美双轨道(Low_Res & Subdiv)数据注入流...")
    deps_low, deps_sub = load_all_deps()
    
    # 1. /data/jiangrui/OpenTouch Data/data/
    h5_dir = "/data/jiangrui/OpenTouch Data/data/"
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5")) + glob.glob(os.path.join(h5_dir, "*.hdf5"))
    if len(h5_files) > 0:
        for h5f in tqdm(h5_files, desc="Processing HDF5 data files"):
            process_h5_file(h5f, deps_low, deps_sub, sigma=0.005)
            
    # 2. /data/jiangrui/OpenTouch Data/extracted_dataset/
    ext_dir = "/data/jiangrui/OpenTouch Data/extracted_dataset/"
    if os.path.exists(ext_dir):
        process_extracted_dataset(ext_dir, deps_low, deps_sub, sigma=0.005)

if __name__ == "__main__":
    main()
