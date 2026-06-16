#!/usr/bin/env python3
import sys
import os
import io
import json
import argparse
import h5py
import cv2
import numpy as np
import random
import glob
from matplotlib import cm
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# Monkey patch pyrender to force a dark background
import pyrender
orig_scene_init = pyrender.Scene.__init__
def new_scene_init(self, *args, **kwargs):
    kwargs['bg_color'] = [30, 30, 30, 0] # Dark grey background
    orig_scene_init(self, *args, **kwargs)
pyrender.Scene.__init__ = new_scene_init

# Try OSMesa if EGL fails, but let's default to egl
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYRENDER_PLATFORM'] = 'egl'
base_dir = "/code/users/jiangrui/Full-Hand-Tactile-Estimation"
sys.path.append(os.path.join(base_dir, "opentouch/preprocess"))
from pyrenderer import ManoRenderer

class Cache:
    def __init__(self):
        self.deps = {}

cache = Cache()

def precompute_deps(res_type):
    if res_type in cache.deps:
        return cache.deps[res_type]
        
    if res_type == "low_res":
        obj_path = os.path.join(base_dir, "opentouch/preprocess/scratch/mano_right_neutral.obj")
        layout_path = os.path.join(base_dir, "opentouch/preprocess/scratch/handLayoutNewest_meshid_lowres.json")
    else:
        obj_path = os.path.join(base_dir, "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj")
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
            valid_nodes[nid] = center

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
    
    node_keys = []
    dist_rows = []
    for nid, center in valid_nodes.items():
        vids = layout[nid].get("mano_vid", [])
        vids = [v for v in vids if v < V_total]
        if len(vids) > 0:
            node_keys.append(nid)
            D_vids = shortest_path(pure_graph, directed=False, indices=vids)
            jump_dists = np.linalg.norm(mano_vertices[vids] - center, axis=1)
            D_k = np.min(D_vids + jump_dists[:, np.newaxis], axis=0)
            dist_rows.append(D_k)
            
    dist_matrix = np.vstack(dist_rows) if dist_rows else np.zeros((0, V), dtype=np.float32)
    
    # Renderer
    width, height = 800, 600
    renderer = ManoRenderer(image_size=(width, height), focal_length=8000.0*(width/1280.0), mano_vertices=mano_vertices, mano_faces=mano_faces)

    cache.deps[res_type] = {
        'V_total': V_total,
        'valid_nodes': valid_nodes,
        'node_keys': node_keys,
        'layout': layout,
        'dist_matrix': dist_matrix,
        'renderer': renderer
    }
    return cache.deps[res_type]

def render_resolution(res_type, pressure16, p_norm_matrix):
    deps = precompute_deps(res_type)
    V_total = deps['V_total']
    valid_nodes = deps['valid_nodes']
    node_keys = deps['node_keys']
    layout = deps['layout']
    dist_matrix = deps['dist_matrix']
    renderer = deps['renderer']

    # Method 1: RAW
    vert_vals_raw = np.zeros(V_total, dtype=np.float32)
    for nid, center in valid_nodes.items():
        r, c = map(int, nid.split('-'))
        if r < pressure16.shape[0] and c < pressure16.shape[1]:
            p_norm = p_norm_matrix[r, c]
            for vid in layout[nid].get("mano_vid", []):
                if vid < V_total:
                    vert_vals_raw[vid] = max(vert_vals_raw[vid], p_norm)

    # Method 3: Dijkstra
    active_pressures = []
    for nid in node_keys:
        r, c = map(int, nid.split('-'))
        if r < pressure16.shape[0] and c < pressure16.shape[1]:
            active_pressures.append(p_norm_matrix[r, c])
        else:
            active_pressures.append(0.0)
            
    K = len(active_pressures)
    if K == 0:
        vert_vals_graph = np.zeros(V_total, dtype=np.float32)
    else:
        active_pressures = np.array(active_pressures, dtype=np.float32)
        sigma = 0.005
        two_sig2 = 2.0 * (sigma * sigma)
        weights = np.exp(-(dist_matrix**2) / two_sig2)
        vert_vals_graph = np.max(weights * active_pressures[:, np.newaxis], axis=0)
        vert_vals_graph = np.clip(vert_vals_graph, 0.0, 1.0)

    colormap_fn = lambda x: np.array(cm.gnuplot2(x))
    
    color_raw = 1.0 - vert_vals_raw
    img_raw_bgr = renderer.render(vertex_colors=colormap_fn(color_raw), colormap_fn=colormap_fn, smooth=True)
    
    color_new = 1.0 - vert_vals_graph
    img_new_bgr = renderer.render(vertex_colors=colormap_fn(color_new), colormap_fn=colormap_fn, smooth=True)
    
    alpha = 1.2
    beta = 0.1 * 255
    img_raw_adj = cv2.convertScaleAbs(img_raw_bgr, alpha=alpha, beta=beta)
    img_new_adj = cv2.convertScaleAbs(img_new_bgr, alpha=alpha, beta=beta)
    
    cv2.putText(img_raw_adj, f"RAW ({res_type})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_new_adj, f"NEW Dijkstra ({res_type})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return cv2.hconcat([img_raw_adj, img_new_adj])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", action="store_true", help="Generate a GIF of a full clip instead of a single frame.")
    args = parser.parse_args()

    h5_dir = "/data/jiangrui/OpenTouch Data/data/"
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5")) + glob.glob(os.path.join(h5_dir, "*.hdf5"))
    random.shuffle(h5_files)

    selected_clip = None
    selected_h5 = None
    for sample_h5 in h5_files:
        try:
            with h5py.File(sample_h5, "r") as f:
                data_group = f["data"]
                for demo_name in data_group.keys():
                    demo = data_group[demo_name]
                    if "right_pressure" in demo:
                        rp = demo["right_pressure"][:]
                        if np.max(np.clip((3072.0 - rp) / 3072.0, 0.0, 1.0)) > 0.15:
                            selected_clip = rp
                            selected_h5 = sample_h5
                            break
            if selected_clip is not None:
                break
        except Exception as e:
            continue

    if selected_clip is None:
        print("Could not find a clip with active pressure.")
        exit(1)

    print(f"Sampled clip from {selected_h5} with {selected_clip.shape[0]} frames.")

    if args.gif:
        import imageio
        from tqdm import tqdm
        
        # Precompute to avoid overhead during loop
        precompute_deps("low_res")
        precompute_deps("subdiv")
        
        frames_bgr = []
        for i in tqdm(range(selected_clip.shape[0]), desc="Rendering GIF frames"):
            pressure16 = selected_clip[i]
            p_norm_matrix = np.clip((3072.0 - pressure16) / 3072.0, 0.0, 1.0)
            
            img_low_res = render_resolution("low_res", pressure16, p_norm_matrix)
            img_subdiv = render_resolution("subdiv", pressure16, p_norm_matrix)
            img_final = cv2.vconcat([img_low_res, img_subdiv])
            
            # Convert BGR to RGB for imageio
            frames_bgr.append(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
            
        save_path = "/code/users/jiangrui/.gemini/antigravity-ide/brain/e9f42487-aff5-47bd-a82a-70ffa4859767/artifacts/gt_comparison.gif"
        imageio.mimsave(save_path, frames_bgr, fps=10)
        print(f"GIF saved to {save_path}")
        
    else:
        # Just render the frame with max pressure
        max_idx = np.argmax([np.max(np.clip((3072.0 - p) / 3072.0, 0.0, 1.0)) for p in selected_clip])
        pressure16 = selected_clip[max_idx]
        p_norm_matrix = np.clip((3072.0 - pressure16) / 3072.0, 0.0, 1.0)
        
        img_low_res = render_resolution("low_res", pressure16, p_norm_matrix)
        img_subdiv = render_resolution("subdiv", pressure16, p_norm_matrix)
        
        img_final = cv2.vconcat([img_low_res, img_subdiv])

        save_path = "/code/users/jiangrui/.gemini/antigravity-ide/brain/e9f42487-aff5-47bd-a82a-70ffa4859767/artifacts/gt_comparison_svg.png"
        cv2.imwrite(save_path, img_final)
        print(f"Visualization saved to {save_path}")
