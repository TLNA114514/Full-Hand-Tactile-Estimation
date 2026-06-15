import os
import json
import glob
import numpy as np
import h5py
import trimesh
from tqdm import tqdm

def load_dependencies():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 1. Load MANO mesh
    mesh_path = os.path.join(base_dir, "opentouch", "preprocess", "scratch", "mano_right_neutral.obj")
    mesh = trimesh.load(mesh_path, process=False)
    mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)

    # 2. Load palm faces/vertices
    palm_faces_path = os.path.join(base_dir, "opentouch", "preprocess", "scratch", "auto_calibrated_palm_faces.json")
    with open(palm_faces_path, "r") as f:
        palm_data = json.load(f)
        
    palm_vertices_set = set()
    for triplet in palm_data["group_positive"]["face_triplets"]:
        for vid in triplet:
            if vid <= 777:
                palm_vertices_set.add(vid)
    palm_vertices = list(palm_vertices_set)
    
    # 3. Load layout
    layout_path = os.path.join(base_dir, "opentouch", "preprocess", "scratch", "handLayoutNewest_meshid_lowres.json")
    if not os.path.exists(layout_path):
        layout_path = os.path.join(base_dir, "opentouch", "preprocess", "scratch", "handLayoutNewest_meshid.json")
        
    with open(layout_path, "r") as f:
        layout_data = json.load(f)
    layout = layout_data["positions"]
    erased_nodes = set(layout_data.get("erasedNodes", []))

    valid_nodes = {}
    for nid, info in layout.items():
        if nid in erased_nodes:
            continue
        vids = info.get("mano_vid", [])
        vids = [v for v in vids if v <= 777]
        if len(vids) > 0:
            center = np.mean(mano_vertices[vids], axis=0)
            valid_nodes[nid] = center

    return mano_vertices, palm_vertices, valid_nodes

def compute_continuous_pressure(pressure16, valid_nodes, mano_vertices, palm_vertices, sigma=0.005):
    n_verts = mano_vertices.shape[0] 
    vert_vals = np.zeros(n_verts, dtype=np.float32)
    two_sig2 = 2.0 * (sigma * sigma)
    
    centers = []
    pressures = []
    for nid, center in valid_nodes.items():
        r, c = map(int, nid.split('-'))
        # 确保不会因为 pressure16 尺寸不匹配越界
        if r < pressure16.shape[0] and c < pressure16.shape[1]:
            # 核心修正：在扩散源头将压力约束为 0.0 ~ 1.0 的归一化百分比！
            p_val = float(pressure16[r, c])
            p_norm = np.clip((3072.0 - p_val) / 3072.0, 0.0, 1.0)
            centers.append(center)
            pressures.append(p_norm)
        
    if len(centers) == 0:
        return vert_vals
        
    centers = np.array(centers, dtype=np.float32) # (K, 3)
    pressures = np.array(pressures, dtype=np.float32) # (K,)
    
    palm_coords = mano_vertices[palm_vertices] # (P, 3)
    
    diff = palm_coords[:, np.newaxis, :] - centers[np.newaxis, :, :] # (P, K, 3)
    dist2 = np.sum(diff**2, axis=2) # (P, K)
    weights = np.exp(-dist2 / two_sig2) # (P, K)
    
    W_sum = np.sum(weights, axis=1) # (P,)
    # 将归一化后的 Taxel 百分比扩散
    palm_vals = np.sum(weights * pressures[np.newaxis, :], axis=1) # (P,)
    
    # 核心修正：除以权重和，得到真正的归一化场，完美解决密集重叠区域超界和基线漂移问题
    palm_vals = palm_vals / (W_sum + 1e-8)
    
    vert_vals[palm_vertices] = palm_vals
    return vert_vals

def process_h5_file(filepath, valid_nodes, mano_vertices, palm_vertices, sigma=0.005):
    try:
        with h5py.File(filepath, "r+") as f:
            if "data" not in f:
                print(f"\n[Skipped] 'data' group not found in {filepath}")
                return
            data_group = f["data"]
            
            needs_processing = False
            for demo_name in data_group.keys():
                demo = data_group[demo_name]
                if "right_pressure" in demo: needs_processing = True
                if "left_pressure" in demo: needs_processing = True
                    
            if not needs_processing:
                return
                
            for demo_name in data_group.keys():
                demo = data_group[demo_name]
                
                if "right_pressure" in demo:
                    rp = demo["right_pressure"][:]
                    rp_cont = []
                    for i in range(rp.shape[0]):
                        cont = compute_continuous_pressure(rp[i], valid_nodes, mano_vertices, palm_vertices, sigma)
                        rp_cont.append(cont)
                    rp_cont = np.array(rp_cont, dtype=np.float32)
                    
                    if "right_pressure_continuous" in demo:
                        del demo["right_pressure_continuous"]
                    demo.create_dataset("right_pressure_continuous", data=rp_cont, compression="gzip")
                    
                if "left_pressure" in demo:
                    lp = demo["left_pressure"][:]
                    lp_cont = []
                    for i in range(lp.shape[0]):
                        cont = compute_continuous_pressure(lp[i], valid_nodes, mano_vertices, palm_vertices, sigma)
                        lp_cont.append(cont)
                    lp_cont = np.array(lp_cont, dtype=np.float32)
                    
                    if "left_pressure_continuous" in demo:
                        del demo["left_pressure_continuous"]
                    demo.create_dataset("left_pressure_continuous", data=lp_cont, compression="gzip")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def process_extracted_dataset(dataset_dir, valid_nodes, mano_vertices, palm_vertices, sigma=0.005):
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
                    # 无条件覆盖旧的表征，因为我们现在使用了 fixed 版本
                    pressure = np.array(hdf5_data[key])
                    if pressure.ndim == 2:
                        cont = compute_continuous_pressure(pressure, valid_nodes, mano_vertices, palm_vertices, sigma)
                        hdf5_data[f"{key}_continuous"] = cont.tolist()
                        modified = True
                    elif pressure.ndim == 3:
                        conts = [compute_continuous_pressure(p, valid_nodes, mano_vertices, palm_vertices, sigma).tolist() for p in pressure]
                        hdf5_data[f"{key}_continuous"] = conts
                        modified = True
                    
            if modified:
                with open(mf, "w") as f:
                    json.dump(data, f)
            elif not has_pressure:
                # 只有当原始数据里既没有right_pressure也没有left_pressure时才打印
                # 避免刷屏
                pass
        except Exception as e:
            print(f"Error processing {mf}: {e}")

def main():
    print("🚀 正在加载依赖，准备重新生成完美的 0.0~1.0 连续表征场...")
    mano_vertices, palm_vertices, valid_nodes = load_dependencies()
    print(f"Loaded {len(palm_vertices)} valid palm vertices out of {mano_vertices.shape[0]}.")
    print(f"Loaded {len(valid_nodes)} valid active taxels.")
    
    # 1. 覆盖修改 /data/jiangrui/OpenTouch Data/data/ 下的 .h5 / .hdf5
    h5_dir = "/data/jiangrui/OpenTouch Data/data/"
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5")) + glob.glob(os.path.join(h5_dir, "*.hdf5"))
    if len(h5_files) > 0:
        print(f"Found {len(h5_files)} HDF5 files in {h5_dir}.")
        for h5f in tqdm(h5_files, desc="Processing HDF5 data files"):
            process_h5_file(h5f, valid_nodes, mano_vertices, palm_vertices, sigma=0.005)
    else:
        print(f"No HDF5 files found in {h5_dir}.")
        
    # 2. 覆盖修改 /data/jiangrui/OpenTouch Data/extracted_dataset/ 下的文件
    ext_dir = "/data/jiangrui/OpenTouch Data/extracted_dataset/"
    if os.path.exists(ext_dir):
        print(f"Found extracted_dataset directory at {ext_dir}.")
        process_extracted_dataset(ext_dir, valid_nodes, mano_vertices, palm_vertices, sigma=0.005)
    else:
        print(f"Directory not found: {ext_dir}.")

if __name__ == "__main__":
    main()
