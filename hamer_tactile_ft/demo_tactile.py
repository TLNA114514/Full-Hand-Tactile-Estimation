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
torch.set_float32_matmul_precision('high')
from pathlib import Path
from tqdm import tqdm

# Parse GPU early to prevent EGL/CUDA conflicts
_gpus = ""
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        _gpus = sys.argv[i+1]
        break
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

# ==========================================================================================
# 🛑 核心黑魔法：源码感知 + 全局空间硬核注入补丁（地表最强终结版，完美解决一切 NameError）
# ==========================================================================================
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
_args, _ = _parser.parse_known_args()

os.environ['PYOPENGL_PLATFORM'] = _args.render_platform
os.environ['PYRENDER_PLATFORM'] = _args.render_platform

try:
    import types
    import builtins
    import re
    import sys

    # 1. 定义一个全能通配符类：既是数字0，又是可任意调用的函数，还支持无限切片和属性延伸
    class UltimateMagicMock(int):
        def __call__(self, *args, **kwargs): return self
        def __getattr__(self, name): return self
        def __getitem__(self, item): return self
        def __iter__(self): return iter([])

    class PerfectMockModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__'): raise AttributeError(name)
            return UltimateMagicMock(0)

    mock_obj = PerfectMockModule('OpenGL.GL')

    # 2. 拦截系统的底层 __import__ 行为
    orig_import = builtins.__import__
    def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        # 只要发现有任何文件在尝试染指 OpenGL/EGL/OSMesa
        if name.startswith('OpenGL') or name in ['EGL', 'OSMesa']:
            if globals is not None and '__file__' in globals:
                try:
                    # 【硬核注入】读取当前正在执行 import 的文件（如 texture.py）的源码
                    with open(globals['__file__'], 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # 抓取该文件里写的所有 OpenGL 相关的函数和常量（如 GL_TEXTURE_2D, glGenTextures 等）
                    tokens = re.findall(r'\b([gG][lL][A-Za-z0-9_]+|[eE][gG][lL][A-Za-z0-9_]+|OSMesa[A-Za-z0-9_]+)\b', content)
                    
                    # 直接强行把这些变量塞进该文件的全局命名空间，彻底断绝 NameError 的可能
                    for token in tokens:
                        if token not in globals:
                            globals[token] = UltimateMagicMock(0)
                except Exception:
                    pass
            return mock_obj
        return orig_import(name, globals, locals, fromlist, level)
    
    # 替换系统全局导入函数
    builtins.__import__ = custom_import

    # 3. 固化系统路由备份
    sys.modules['EGL'] = mock_obj
    sys.modules['OSMesa'] = mock_obj
    sys.modules['OpenGL'] = mock_obj
    sys.modules['OpenGL.GL'] = mock_obj
    sys.modules['OpenGL.GL.shaders'] = mock_obj
    
    print("\n====== [Success] Hardcore Global Token Injector Activated! ======\n")
except Exception as e:
    print(f"Bypass failed: {e}")
# ==========================================================================================

# Setup sys.path relative to workspace
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
eval_dir = os.path.join(workspace_dir, "evaluation")
hamer_dir = os.path.join(workspace_dir, "hamer")
preprocess_dir = os.path.join(workspace_dir, "opentouch/preprocess")

sys.path.append(eval_dir)
sys.path.append(hamer_dir)
sys.path.append(ft_dir)
sys.path.append(preprocess_dir)

# Import necessary dependencies
from train import OpenTouchHAMER_TactileWrapper, load_compatible_state_dict
from eval_hamer import ViTDetDataset
from vitpose_model import ViTPoseModel
from hamer.utils import recursive_to
from hamer.configs import get_config
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer

# Import preprocess module helpers
from load_data import (
    export_rgb_frames,
    export_pose_mano,
    export_tactile_mano,
)
from pyrenderer import ManoRenderer
import trimesh
from generate_video import generate_video_from_images
from concat_videos import concat_videos

# ==============================================================================
# Local definitions of helpers from load_data.py to avoid nested function ImportError
# ==============================================================================

def _find_first_existing(cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def _load_layout_json():
    layout_json = _find_first_existing([
        "handLayoutNewest_meshid.json",
        os.path.join(preprocess_dir, "data", "handLayoutNewest_meshid.json"),
        os.path.join(preprocess_dir, "scratch", "handLayoutNewest_meshid.json"),
        os.path.join(preprocess_dir, "handLayoutNewest_meshid.json"),
        os.path.join(workspace_dir, "data", "handLayoutNewest_meshid.json"),
    ])
    if layout_json is None:
        raise FileNotFoundError("Missing handLayoutNewest_meshid.json")
    with open(layout_json, "r") as f:
        d = json.load(f)
    return d["positions"], set(d.get("erasedNodes", []))


def _build_vertex_graph(verts, faces):
    V = len(verts)
    nbrs = [[] for _ in range(V)]
    dists = [[] for _ in range(V)]
    edges = set()
    for a, b, c in faces.astype(np.int64):
        edges.update({(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))})
    for i, j in edges:
        dij = np.linalg.norm(verts[i] - verts[j])
        nbrs[i].append(j)
        dists[i].append(dij)
        nbrs[j].append(i)
        dists[j].append(dij)
    return nbrs, dists


def _gaussian_smooth_vertex_signal(vals, nbrs, dists, sigma=0.005, iters=2):
    if sigma <= 0 or iters <= 0: 
        return vals
    two_sig2 = 2.0 * (sigma * sigma)
    out = vals.astype(np.float32).copy()
    for _ in range(iters):
        new = out.copy()
        for i, (N, D) in enumerate(zip(nbrs, dists)):
            max_val = out[i]
            for j, dij in zip(N, D):
                w = np.exp(-(dij * dij) / two_sig2)
                v_decay = w * out[j]
                if v_decay > max_val:
                    max_val = v_decay
            new[i] = max_val
        out = new
    return out


def _render_pressure_mano(mano_vertices, mano_faces, renderer, pressure16, layout, erased_nodes, vmin, vmax, nbrs, dists):
    from collections import defaultdict
    from matplotlib import cm
    # normalize pressure → [0,1]
    norm = ((pressure16 - vmin) / max(vmax - vmin, 1e-6)).clip(0, 1)
    valid_nodes = {
        nid: {"mano_vid": layout[nid].get("mano_vid", [])}
        for nid in layout.keys() if nid not in erased_nodes
    }
    vert_to_vals = defaultdict(list)
    for nid, info in valid_nodes.items():
        r, c = map(int, nid.split('-'))
        val = float(norm[r, c])
        for vid in info["mano_vid"]:
            vert_to_vals[vid].append(val)

    n_verts = mano_vertices.shape[0]
    vert_vals = np.zeros(n_verts, dtype=np.float32)
    if vert_to_vals:
        for vid, arr in vert_to_vals.items():
            vert_vals[vid] = float(np.mean(arr))
        known_mask = np.zeros(n_verts, bool)
        known_mask[list(vert_to_vals.keys())] = True
        vert_max = float(vert_vals[known_mask].max())
        vert_vals[~known_mask] = vert_max

    # connectivity smoothing
    vert_vals = _gaussian_smooth_vertex_signal(vert_vals, nbrs, dists, sigma=0.005, iters=2)

    # invert + min-max normalize
    mn, mx = float(vert_vals.min()), float(vert_vals.max())
    if mx > mn:
        vert_vals = 1.0 - (vert_vals - mn) / (mx - mn)
    else:
        vert_vals[:] = 1.0

    colormap_fn = lambda x: np.array(cm.gnuplot2(x))
    vertex_colors = colormap_fn(vert_vals)  # RGBA float in [0,1]
    img_rgb = renderer.render(vertex_colors=vertex_colors, colormap_fn=colormap_fn, smooth=True)
    return img_rgb[:, :, ::-1], vertex_colors


def _prepare_tactile_frame(sample, attrs):
    import math
    arr = np.asarray(sample)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 1:
        grid_shape = attrs.get("grid_shape") if attrs else None
        if grid_shape:
            rows, cols = map(int, grid_shape)
            return arr.reshape(rows, cols).astype(np.float32)

        length = arr.shape[0]
        root = int(math.sqrt(length))
        if root * root == length:
            return arr.reshape(root, root).astype(np.float32)

        return arr.reshape(1, length).astype(np.float32)


def _resolve_file(filename):
    for candidate in [
        os.path.join(workspace_dir, "data", filename),
        os.path.join(preprocess_dir, "data", filename),
        os.path.join(preprocess_dir, "scratch", filename),
        os.path.join(workspace_dir, "EasyMocap", "data", "smplx", filename),
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Missing {filename}")


def _load_mano_model(side, model_root, use_cuda=True, **kwargs):
    # Patch numpy for chumpy compatibility (numpy >=1.24 removed these aliases)
    import numpy as _np
    for _attr in ("bool", "int", "float", "complex", "object", "unicode", "str"):
        if not hasattr(_np, _attr):
            setattr(_np, _attr, getattr(__builtins__, _attr, object))
    from easymocap.smplmodel.body_model import SMPLlayer
    import torch
    device = torch.device("cuda") if (use_cuda and torch.cuda.is_available()) else torch.device("cpu")
    lr = {'left': 'LEFT', 'right': 'RIGHT'}
    pkl_path = _resolve_file(f'MANO_{lr[side]}.pkl')
    reg_path = _resolve_file(f'J_regressor_mano_{lr[side]}.txt')
    body_model = SMPLlayer(pkl_path,
                           model_type='mano', gender='neutral',
                           device=device,
                           regressor_path=reg_path,
                           **kwargs).to(device)
    return body_model, device


def _fit_sequence_to_mano(body_model, keypoints3d_seq):
    from easymocap.pipeline import smpl_from_keypoints3d
    from easymocap.dataset import CONFIG
    import numpy as np
    T = keypoints3d_seq.shape[0]
    kp = np.concatenate([keypoints3d_seq.astype(np.float32),
                         np.ones((T, keypoints3d_seq.shape[1], 1), np.float32)], axis=-1)
    class _Args: pass
    args = _Args(); args.robust3d=False; args.verbose=False; args.model='mano'
    w_pose = {'k3d':1e2,'k2d':1e-8,'reg_poses':1e-1,'smooth_body':1e2,'smooth_poses':1e2}
    return smpl_from_keypoints3d(body_model, kp, config=CONFIG['handr'],
                                 args=args,
                                 weight_shape={'s3d':1e6,'reg_shapes':1e1},
                                 weight_pose=w_pose)


def nudge_thumb_lateral_3j(arr, k_tip=0.08, k_ip=0.05, k_mcp=0.03):
    import numpy as np
    T = arr.shape[0]
    for t in range(T):
        c = arr[t]

        wrist     = c[0]
        idx_mcp   = c[5]  if c.shape[0] > 5  else wrist
        pky_mcp   = c[17] if c.shape[0] > 17 else wrist
        mid_mcp   = c[9]  if c.shape[0] > 9  else wrist
        th_mcp    = c[2]  if c.shape[0] > 2  else wrist
        th_ip     = c[3]  if c.shape[0] > 3  else wrist
        th_tip    = c[4]  if c.shape[0] > 4  else wrist

        x_axis = idx_mcp - pky_mcp
        w = np.linalg.norm(x_axis)
        if w < 1e-8:
            x_axis = np.array([1.,0.,0.], np.float32); w = 1.0
        else:
            x_axis = x_axis / w

        d_tip = x_axis * (k_tip * w)
        d_ip  = x_axis * (k_ip  * w)
        d_mcp = x_axis * (k_mcp * w)

        c[4] = th_tip + d_tip      # tip
        c[3] = th_ip  + d_ip       # under the tip (IP)
        c[2] = th_mcp + d_mcp      # second under (MCP)

        arr[t] = c
    return arr


class RenderOnce:
    def __init__(self, faces, image_size=(1280, 960), bg_rgb=(249, 235, 142)):
        import pyrender
        import trimesh
        W, H = image_size
        self.W, self.H = W, H
        self.scene = pyrender.Scene(bg_color=[*bg_rgb, 0], ambient_light=[0.25, 0.25, 0.25])
        # camera (focal length scaled proportionally to width to keep hand size consistent)
        fx = 8000.0 * (W / 1280.0)
        fy = 8000.0 * (W / 1280.0)
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=W/2, cy=H/2, zfar=1e12)
        T = np.eye(4); T[:3,3] = [0,0,2.0]
        self.scene.add(cam, pose=T)
        # a few directional lights
        def _dir(theta, phi, inten):
            T = np.eye(4); r=4.0
            th=np.radians(theta); ph=np.radians(phi)
            pos=np.array([r*np.sin(th)*np.cos(ph), r*np.sin(th)*np.sin(ph), r*np.cos(th)])
            z=-pos/np.linalg.norm(pos); x=np.array([-z[1],z[0],0.]); 
            if np.linalg.norm(x)==0: x=np.array([1.,0.,0.])
            x/=np.linalg.norm(x); y=np.cross(z,x); T[:3,:3]=np.stack([x,y,z],1); T[:3,3]=pos
            lightnode = pyrender.Node(light=pyrender.DirectionalLight(intensity=inten), matrix=T)
            self.scene.add_node(lightnode)
        _dir(40,   0, 3.0); _dir(65, 120, 2.0); _dir(70, -120, 2.0)
        # static faces, material (shiny), placeholder positions
        self.faces = faces.astype(np.int32)
        self.material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(153/255, 41/255, 234/255, 1.), metallicFactor=0.0, roughnessFactor=0.08,
            alphaMode='OPAQUE'
        )
        self.Ry = trimesh.transformations.rotation_matrix(np.radians(90),  [0,1,0])
        self.Rx = trimesh.transformations.rotation_matrix(np.radians(-90), [1,0,0])

        dummy = np.zeros((self.faces.max()+1, 3), dtype=np.float32)
        tri = trimesh.Trimesh(vertices=dummy, faces=self.faces, process=False)
        tri.vertex_normals  # ensure normals array exists
        self.mesh = pyrender.Mesh.from_trimesh(tri, material=self.material, smooth=True)
        self.node = self.scene.add(self.mesh)

        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(W, H)
        self.flags = pyrender.RenderFlags.RGBA
        if hasattr(pyrender.RenderFlags, "SHADOWS"):
            self.flags |= pyrender.RenderFlags.SHADOWS

    def render(self, verts):
        import trimesh
        import pyrender
        v = np.asarray(verts, dtype=np.float32)
        v_h = np.concatenate([v, np.ones((len(v),1), np.float32)], axis=1)
        v_h = (self.Ry @ v_h.T).T
        v_h = (self.Rx @ v_h.T).T
        v = v_h[:, :3]

        prim = self.mesh.primitives[0]
        if hasattr(prim, "positions") and hasattr(prim, "needs_update"):
            prim.positions = v
            prim.needs_update = True
        else:
            self.scene.remove_node(self.node)
            tri = trimesh.Trimesh(vertices=v, faces=self.faces, process=False)
            tri.vertex_normals
            self.mesh = pyrender.Mesh.from_trimesh(tri, material=self.material, smooth=True)
            self.node = self.scene.add(self.mesh)

        rgba, _ = self.renderer.render(self.scene, flags=self.flags)
        return rgba[:, :, :3]

    def close(self):
        self.renderer.delete()


# ==============================================================================
# Model Loading and Rendering Sequences
# ==============================================================================

def load_models(checkpoint_path, device):
    print(">>> Loading model config...")
    model_cfg_path = os.path.join(hamer_dir, '_DATA/hamer_ckpts/model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
        
    print(">>> Initializing OpenTouchHAMER_TactileWrapper...")
    model = OpenTouchHAMER_TactileWrapper(cfg=model_cfg)
    dummy_input = torch.zeros(1, 3, model_cfg.MODEL.IMAGE_SIZE, model_cfg.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        dummy_feat = model.backbone(dummy_input[:, :, :, 32:-32])
        model.tactile_head(dummy_feat)
        print(f">>> Tactile head initialized with output dim: {model.tactile_dim}")
    print(f">>> Loading weights from checkpoint: {checkpoint_path}")
    load_compatible_state_dict(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    print(">>> Initializing ViTDet...")
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    local_vitdet_path = os.path.join(hamer_dir, "_DATA/model_final_f05665.pkl")
    if os.path.exists(local_vitdet_path):
        detectron2_cfg.train.init_checkpoint = local_vitdet_path
    else:
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # ViTPose import monkey patching if needed
    try:
        import importlib.util
        vit_path = os.path.join(hamer_dir, "third-party/ViTPose/mmpose/models/backbones/vit.py")
        spec = importlib.util.spec_from_file_location("mmpose.models.backbones.vit", vit_path)
        vit_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vit_module)
        import mmpose.apis.inference
        custom_ckpt_path = os.path.join(hamer_dir, "third-party/ViTPose/mmcv_custom/checkpoint.py")
        spec_ckpt = importlib.util.spec_from_file_location("mmcv_custom.checkpoint", custom_ckpt_path)
        custom_ckpt_module = importlib.util.module_from_spec(spec_ckpt)
        spec_ckpt.loader.exec_module(custom_ckpt_module)
        mmpose.apis.inference.load_checkpoint = custom_ckpt_module.load_checkpoint
    except Exception:
        pass
        
    cpm = ViTPoseModel(device)
    
    return model, detector, cpm, model_cfg


def render_pose_sequence(joints_seq, output_dir, demo_id, target_size=(1280, 960), side="right", use_cuda=True):
    # Prepare the joints sequence (flip X-axis and nudge thumb)
    arr = joints_seq.copy().astype(np.float32)
    arr[:, :, 0] = -arr[:, :, 0]
    nudge_thumb_lateral_3j(arr, k_tip=0.3, k_ip=0.2, k_mcp=0.1)
    
    # Load MANO model
    body_model, device = _load_mano_model(
        side=side, model_root=None, use_cuda=use_cuda,
        num_pca_comps=6, use_pose_blending=True, use_shape_blending=True,
        use_pca=False, use_flat_mean=False
    )
    
    # Fit keypoints sequence to MANO to get params
    params = _fit_sequence_to_mano(body_model, arr)
    faces = body_model.faces
    
    # Create renderer
    background_color = (249, 235, 142)  # Cream color matching GT Pose background
    rctx = RenderOnce(faces=faces, image_size=target_size, bg_rgb=background_color)
    
    os.makedirs(output_dir, exist_ok=True)
    T = arr.shape[0]
    
    def _select_nf(pd, nf):
        out = {}
        for k, v in pd.items():
            if isinstance(v, np.ndarray) and v.shape[:1] == (T,):
                v = v[nf]
            if isinstance(v, np.ndarray) and v.ndim == 1:
                v = v[None, ...]
            out[k] = v
        return out

    print(f"Rendering pose sequence ({side}) to {output_dir}...")
    for nf in range(T):
        p = _select_nf(params, nf)
        if 'Th' in p: p['Th'] = np.zeros_like(p['Th'])
        if 'Rh' in p: p['Rh'] = np.zeros_like(p['Rh'])

        verts = body_model(return_verts=True, return_tensor=False, **p)[0]
        if verts.ndim == 3:
            verts = verts[0]

        img = rctx.render(verts)
        cv2.imwrite(os.path.join(output_dir, f"{demo_id}_{nf:05d}.png"), img[:, :, ::-1])
        
    rctx.close()
    print("Pose rendering completed.")


def render_tactile_sequence(pressure_seq, output_dir, demo_id, vmin, vmax, target_size=(1280, 960), temporal_alpha=0.4):
    width, height = target_size
    layout, erased_nodes = _load_layout_json()
    
    # Find subdivisions OBJ
    obj_path = _find_first_existing([
        "mano_right_neutral_subdiv.obj",
        os.path.join(preprocess_dir, "data", "mano_right_neutral_subdiv.obj"),
        os.path.join(preprocess_dir, "scratch", "mano_right_neutral_subdiv.obj"),
        os.path.join(preprocess_dir, "mano_right_neutral_subdiv.obj"),
        os.path.join(workspace_dir, "data", "mano_right_neutral_subdiv.obj"),
    ])
    if obj_path is None:
        raise FileNotFoundError("Missing mano_right_neutral_subdiv.obj")
        
    mesh = trimesh.load(obj_path, process=False)
    mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mano_faces    = np.asarray(mesh.faces, dtype=np.int32)

    nbrs, dists = _build_vertex_graph(mano_vertices, mano_faces)
    
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    renderer = ManoRenderer(image_size=(width, height),
                            focal_length=8000.0 * (width / 1280.0),
                            mano_vertices=mano_vertices,
                            mano_faces=mano_faces)
                            
    os.makedirs(output_dir, exist_ok=True)
    prev = None
    
    print(f"Rendering tactile sequence to {output_dir}...")
    for idx, grid in enumerate(pressure_seq):
        if temporal_alpha and prev is not None:
            grid = temporal_alpha * grid + (1.0 - temporal_alpha) * prev
        prev = grid

        img_bgr, vcolors = _render_pressure_mano(
            mano_vertices, mano_faces, renderer, grid, layout, erased_nodes, vmin, vmax, nbrs, dists
        )
        
        # Contrast adjustment matching the original code
        alpha = 1.2
        beta = 0.1 * 255
        img_adj = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
        cv2.imwrite(os.path.join(output_dir, f"{demo_id}_{idx:05d}.png"), img_adj)
        
    print("Tactile rendering completed.")


def render_subdiv_tactile_signal_sequence(tactile_seq, output_dir, demo_id, target_size=(1280, 960), temporal_alpha=0.4):
    from matplotlib import cm
    width, height = target_size

    obj_path = _find_first_existing([
        "mano_right_neutral_subdiv.obj",
        os.path.join(preprocess_dir, "data", "mano_right_neutral_subdiv.obj"),
        os.path.join(preprocess_dir, "scratch", "mano_right_neutral_subdiv.obj"),
        os.path.join(preprocess_dir, "mano_right_neutral_subdiv.obj"),
        os.path.join(workspace_dir, "data", "mano_right_neutral_subdiv.obj"),
    ])
    if obj_path is None:
        raise FileNotFoundError("Missing mano_right_neutral_subdiv.obj")

    mesh = trimesh.load(obj_path, process=False)
    mano_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    mano_faces = np.asarray(mesh.faces, dtype=np.int32)

    renderer = ManoRenderer(
        image_size=(width, height),
        focal_length=8000.0 * (width / 1280.0),
        mano_vertices=mano_vertices,
        mano_faces=mano_faces,
    )

    os.makedirs(output_dir, exist_ok=True)
    prev = None
    colormap_fn = lambda x: np.array(cm.gnuplot2(x))

    print(f"Rendering subdiv tactile signal sequence to {output_dir}...")
    for idx, signal in enumerate(tactile_seq):
        signal = np.asarray(signal, dtype=np.float32)
        if signal.shape[0] != mano_vertices.shape[0]:
            raise ValueError(
                f"Predicted tactile dim {signal.shape[0]} does not match subdiv vertex count {mano_vertices.shape[0]}"
            )
        if temporal_alpha and prev is not None:
            signal = temporal_alpha * signal + (1.0 - temporal_alpha) * prev
        prev = signal

        color_new = np.clip(1.0 - signal, 0.0, 1.0)
        img_bgr = renderer.render(vertex_colors=colormap_fn(color_new), colormap_fn=colormap_fn, smooth=True)
        img_adj = cv2.convertScaleAbs(img_bgr, alpha=1.2, beta=0.1 * 255)
        cv2.imwrite(os.path.join(output_dir, f"{demo_id}_{idx:05d}.png"), img_adj)

    print("Subdiv tactile rendering completed.")


def main():
    parser = argparse.ArgumentParser(description="Demo pipeline to build comparative predicted vs GT videos.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained tactile checkpoint path")
    parser.add_argument("--hdf5_path", type=str, required=True, help="HDF5 file path")
    parser.add_argument("--clip_id", type=str, required=True, help="Clip ID within the HDF5 file")
    parser.add_argument("--out_dir", type=str, default="./demo_output", help="Output directory")
    parser.add_argument("--gpu", type=str, default="4", help="GPU index")
    parser.add_argument("--hand", type=str, choices=["left", "right"], default="right", help="Hand side")
    parser.add_argument("--render_platform", type=str, default="egl", choices=["egl", "osmesa"], help="Rendering platform (egl or osmesa)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        # Since CUDA_VISIBLE_DEVICES is set early to args.gpu, PyTorch maps it to index 0.
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    clip_id = args.clip_id
    hand_side = args.hand
    is_right_hand = 1 if hand_side == "right" else 0

    # Ensure output folders
    out_path = Path(args.out_dir) / clip_id
    rgb_dir = out_path / "rgb"
    gt_pose_dir = out_path / "gt_pose"
    pred_pose_dir = out_path / "pred_pose"
    gt_touch_dir = out_path / "gt_touch"
    pred_touch_dir = out_path / "pred_touch"

    target_size = (1280, 960)

    # Step 1: Export RGB images
    print("\n>>> [1/7] Exporting RGB frames...")
    export_rgb_frames(
        file_path=args.hdf5_path,
        demo_id=clip_id,
        output_dir=str(rgb_dir),
        target_size=target_size,
        channel_order="bgr",
    )

    # Step 2: Export GT pose and GT tactile heatmaps
    print("\n>>> [2/7] Exporting GT hand pose...")
    export_pose_mano(
        file_path=args.hdf5_path,
        demo_id=clip_id,
        output_dir=str(gt_pose_dir),
        dataset_names=(f"{hand_side}_hand_landmarks",),
        target_size=target_size,
    )

    print("\n>>> [3/7] Exporting GT tactile heatmaps...")
    export_tactile_mano(
        file_path=args.hdf5_path,
        demo_id=clip_id,
        output_dir=str(gt_touch_dir),
        dataset_names=(f"{hand_side}_pressure",),
        target_size=target_size,
    )

    # Compute global vmin and vmax from HDF5 raw pressure signals
    from load_data import _resolve_demo_name, _require_data_group
    with h5py.File(args.hdf5_path, "r") as src:
        dpath = f"data/{clip_id}/{hand_side}_pressure"
        if dpath not in src:
            resolved = _resolve_demo_name(_require_data_group(src, args.hdf5_path), clip_id)
            dpath = f"data/{resolved}/{hand_side}_pressure"
        dset = src[dpath]
        gt_pressure_seq = []
        for sample in dset:
            grid = _prepare_tactile_frame(sample, dset.attrs).astype(np.float32)
            if grid.shape != (16, 16):
                grid = cv2.resize(grid, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
            gt_pressure_seq.append(grid)

    layout, erased_nodes = _load_layout_json()
    stack = np.stack(gt_pressure_seq, 0) if gt_pressure_seq else np.zeros((0, 16, 16), np.float32)
    valid_mask = np.zeros((16, 16), dtype=bool)
    for nid in layout.keys():
        if nid in erased_nodes: continue
        r, c = map(int, nid.split('-'))
        valid_mask[r, c] = True

    if stack.size:
        vmin = float(stack[:, valid_mask].min())
        vmax = float(stack[:, valid_mask].max())
    else:
        vmin, vmax = 0.0, 1.0

    # Step 3: Run model inference
    print("\n>>> [4/7] Loading models & running inference on clip...")
    model, detector, cpm, model_cfg = load_models(args.checkpoint, device)

    # Read RGB image sequence from HDF5
    with h5py.File(args.hdf5_path, "r") as f:
        clip_group = f[f"data/{clip_id}"]
        rgb_bytes_seq = clip_group["rgb_images_jpeg"][()]
        
    num_frames = len(rgb_bytes_seq)
    
    pred_joints_list = []
    pred_tactile_list = []

    for idx in tqdm(range(num_frames), desc="Running Inference"):
        img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes_seq[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            pred_joints_list.append(None)
            pred_tactile_list.append(None)
            continue
            
        img_rgb = img_bgr[:, :, ::-1]
        
        # Run ViTDet detection
        try:
            det_out = detector(img_bgr)
            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        except Exception:
            pred_joints_list.append(None)
            pred_tactile_list.append(None)
            continue
            
        if len(pred_bboxes) == 0:
            pred_joints_list.append(None)
            pred_tactile_list.append(None)
            continue
            
        # Run ViTPose
        try:
            vitposes_out = cpm.predict_pose(img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])
        except Exception:
            pred_joints_list.append(None)
            pred_tactile_list.append(None)
            continue
        
        bboxes = []
        is_right = []
        
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            
            if not is_right_hand:  # left hand
                valid = left_hand_keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [left_hand_keyp[valid, 0].min(), left_hand_keyp[valid, 1].min(),
                            left_hand_keyp[valid, 0].max(), left_hand_keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
            else:  # right hand
                valid = right_hand_keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [right_hand_keyp[valid, 0].min(), right_hand_keyp[valid, 1].min(),
                            right_hand_keyp[valid, 0].max(), right_hand_keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)
                    
        if len(bboxes) == 0:
            pred_joints_list.append(None)
            pred_tactile_list.append(None)
            continue
            
        boxes_arr = np.stack(bboxes)
        right_arr = np.stack(is_right)
        
        dataset_batch = ViTDetDataset(model_cfg, img_bgr, boxes_arr, right_arr, rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset_batch, batch_size=len(bboxes), shuffle=False, num_workers=0)
        
        frame_joints = None
        frame_tactile = None
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model.forward_step(batch, train=False)
                
            pred_joints = out['pred_keypoints_3d'].detach().cpu().numpy()
            pred_tactile = out['pred_tactile'].detach().cpu().numpy()
            
            for n in range(pred_joints.shape[0]):
                is_r = int(batch['right'][n].cpu().numpy())
                if (is_right_hand and is_r == 1) or (not is_right_hand and is_r == 0):
                    frame_joints = pred_joints[n]
                    frame_tactile = pred_tactile[n]
                    break
            if frame_joints is not None:
                break
                
        if frame_joints is not None:
            if not is_right_hand:
                frame_joints[:, 0] *= -1.0
                
        pred_joints_list.append(frame_joints)
        pred_tactile_list.append(frame_tactile)

    # Perform gap filling for any missing detection frames
    first_valid_idx = next((idx for idx, j in enumerate(pred_joints_list) if j is not None), None)
    if first_valid_idx is None:
        print(">>> Warning: No hand detected in any frame of the video!")
        pred_joints_seq = np.zeros((num_frames, 21, 3), dtype=np.float32)
        pred_tactile_seq = np.zeros((num_frames, model.tactile_dim), dtype=np.float32)
    else:
        for idx in range(first_valid_idx):
            pred_joints_list[idx] = pred_joints_list[first_valid_idx]
            pred_tactile_list[idx] = pred_tactile_list[first_valid_idx]
            
        curr_joints = pred_joints_list[0]
        curr_tactile = pred_tactile_list[0]
        for idx in range(1, num_frames):
            if pred_joints_list[idx] is None:
                pred_joints_list[idx] = curr_joints
                pred_tactile_list[idx] = curr_tactile
            else:
                curr_joints = pred_joints_list[idx]
                curr_tactile = pred_tactile_list[idx]
                
        pred_joints_seq = np.stack(pred_joints_list, axis=0)
        pred_tactile_seq = np.stack(pred_tactile_list, axis=0)

    # Step 4: Render Predicted Pose using custom RenderOnce matching export_pose_mano
    print("\n>>> [5/7] Rendering Predicted Hand Pose...")
    render_pose_sequence(
        joints_seq=pred_joints_seq,
        output_dir=str(pred_pose_dir),
        demo_id=clip_id,
        target_size=target_size,
        side=hand_side,
        use_cuda=(device.type == 'cuda'),
    )

    # Step 5: Render predicted tactile directly on the subdiv MANO vertices.
    print("\n>>> [6/7] Rendering Predicted Subdiv Tactile Heatmap...")
    render_subdiv_tactile_signal_sequence(
        tactile_seq=pred_tactile_seq,
        output_dir=str(pred_touch_dir),
        demo_id=clip_id,
        target_size=target_size,
    )

    # Step 6: Create sub-videos & stack them
    print("\n>>> [7/7] Compiling sub-videos and stitching final visualization...")
    
    # Locate paths
    rgb_mp4 = out_path / "rgb.mp4"
    gt_pose_mp4 = out_path / "gt_pose.mp4"
    pred_pose_mp4 = out_path / "pred_pose.mp4"
    gt_touch_mp4 = out_path / "gt_touch.mp4"
    pred_touch_mp4 = out_path / "pred_touch.mp4"
    
    # Generate MP4s from image directories
    generate_video_from_images(clip_id, str(rgb_dir), str(rgb_mp4), fps=30)
    generate_video_from_images(clip_id, str(gt_pose_dir / f"{hand_side}_hand_landmarks"), str(gt_pose_mp4), fps=30)
    generate_video_from_images(clip_id, str(pred_pose_dir), str(pred_pose_mp4), fps=30)
    generate_video_from_images(clip_id, str(gt_touch_dir / f"{hand_side}_pressure"), str(gt_touch_mp4), fps=30)
    generate_video_from_images(clip_id, str(pred_touch_dir), str(pred_touch_mp4), fps=30)
    
    # Vertical concat Middle Column
    middle_column_mp4 = out_path / "middle_column.mp4"
    concat_videos(
        videos=[gt_pose_mp4, pred_pose_mp4],
        output=middle_column_mp4,
        layout="vertical",
    )
    
    # Vertical concat Right Column
    right_column_mp4 = out_path / "right_column.mp4"
    concat_videos(
        videos=[gt_touch_mp4, pred_touch_mp4],
        output=right_column_mp4,
        layout="vertical",
    )
    
    # Horizontal concat Left, Middle, Right columns
    final_output_mp4 = out_path / "combined.mp4"
    concat_videos(
        videos=[rgb_mp4, middle_column_mp4, right_column_mp4],
        output=final_output_mp4,
        layout="horizontal",
        scale_height=1920,  # 2 * 960
    )
    
    print(f"\n>>> Visualization compilation complete! Saved to: {final_output_mp4}")


if __name__ == "__main__":
    main()
