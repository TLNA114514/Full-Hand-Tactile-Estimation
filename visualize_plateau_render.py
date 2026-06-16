#!/usr/bin/env python3
import sys
import os
import io
import json
import argparse
import h5py
import cv2
import numpy as np
import torch
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

def render_resolution(res_type, pressure16, p_norm_matrix):
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
            valid_nodes[nid] = center

    # Method 1: RAW
    vert_vals_raw = np.zeros(V_total, dtype=np.float32)
    for nid, center in valid_nodes.items():
        r, c = map(int, nid.split('-'))
        if r < pressure16.shape[0] and c < pressure16.shape[1]:
            p_norm = p_norm_matrix[r, c]
            for vid in layout[nid].get("mano_vid", []):
                if vid < V_total:
                    vert_vals_raw[vid] = max(vert_vals_raw[vid], p_norm)

    # Method 3: Dijkstra Geodesic (No Wormholes)
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
    
    active_pressures = []
    dist_rows = []
    for nid, center in valid_nodes.items():
        r, c = map(int, nid.split('-'))
        if r < pressure16.shape[0] and c < pressure16.shape[1]:
            p_norm = p_norm_matrix[r, c]
            vids = layout[nid].get("mano_vid", [])
            vids = [v for v in vids if v < V_total]
            if len(vids) > 0:
                active_pressures.append(p_norm)
                # Compute shortest paths from all anchor vertices to all vertices
                D_vids = shortest_path(pure_graph, directed=False, indices=vids) # (len(vids), V)
                # Jump distance from virtual center to each anchor vertex
                jump_dists = np.linalg.norm(mano_vertices[vids] - center, axis=1) # (len(vids),)
                # D_k is the minimum sum of jump_distance + geodesic_distance
                D_k = np.min(D_vids + jump_dists[:, np.newaxis], axis=0) # (V,)
                dist_rows.append(D_k)
                    
    K = len(active_pressures)
    if K == 0:
        vert_vals_graph = np.zeros(V, dtype=np.float32)
    else:
        dist_matrix = np.vstack(dist_rows) # (K, V)
        active_pressures = np.array(active_pressures, dtype=np.float32)
        sigma = 0.005
        two_sig2 = 2.0 * (sigma * sigma)
        weights = np.exp(-(dist_matrix**2) / two_sig2)
        vert_vals_graph = np.max(weights * active_pressures[:, np.newaxis], axis=0)
        vert_vals_graph = np.clip(vert_vals_graph, 0.0, 1.0)

    # Render
    width, height = 800, 600
    renderer = ManoRenderer(image_size=(width, height), focal_length=8000.0*(width/1280.0), mano_vertices=mano_vertices, mano_faces=mano_faces)
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
    h5_dir = "/data/jiangrui/OpenTouch Data/data/"
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5")) + glob.glob(os.path.join(h5_dir, "*.hdf5"))
    random.shuffle(h5_files)

    pressure16 = None
    for sample_h5 in h5_files:
        try:
            with h5py.File(sample_h5, "r") as f:
                data_group = f["data"]
                valid_frames = []
                for demo_name in data_group.keys():
                    demo = data_group[demo_name]
                    if "right_pressure" in demo:
                        rp = demo["right_pressure"][:]
                        for i in range(rp.shape[0]):
                            p_frame = rp[i]
                            p_norm_matrix = np.clip((3072.0 - p_frame) / 3072.0, 0.0, 1.0)
                            if np.max(p_norm_matrix) > 0.15:
                                valid_frames.append(p_frame)
                if valid_frames:
                    pressure16 = random.choice(valid_frames)
                    print(f"Sampled frame from {sample_h5}")
                    break
        except Exception as e:
            continue

    if pressure16 is None:
        print("Could not find a frame with active pressure.")
        exit(1)

    p_norm_matrix = np.clip((3072.0 - pressure16) / 3072.0, 0.0, 1.0)
    
    img_low_res = render_resolution("low_res", pressure16, p_norm_matrix)
    img_subdiv = render_resolution("subdiv", pressure16, p_norm_matrix)
    
    img_final = cv2.vconcat([img_low_res, img_subdiv])

    save_path = "/code/users/jiangrui/.gemini/antigravity-ide/brain/e9f42487-aff5-47bd-a82a-70ffa4859767/artifacts/gt_comparison_svg.png"
    cv2.imwrite(save_path, img_final)
    print(f"Visualization saved to {save_path}")
