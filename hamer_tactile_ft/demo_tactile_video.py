import sys
import os

# OpenGL backends must be selected before importing hamer/pyrender. The
# canonical tactile renderer below is pure OpenCV/NumPy and needs neither EGL
# nor OSMesa, but the explicit choices keep legacy pose rendering usable on
# headless machines.
_render_platform = "software"
for i, arg in enumerate(sys.argv):
    if arg == "--render_platform" and i + 1 < len(sys.argv):
        _render_platform = sys.argv[i + 1].lower()
        break
if _render_platform in {"egl", "osmesa"}:
    os.environ["PYOPENGL_PLATFORM"] = _render_platform
if _render_platform == "egl":
    # EGL indexes CUDA_VISIBLE_DEVICES rather than the original physical ID.
    os.environ["EGL_DEVICE_ID"] = "0"

import argparse
import json
import re
import subprocess
import cv2
import numpy as np
import torch
import trimesh
# Parse GPU early
_gpus = ""
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        _gpus = sys.argv[i+1]
        break
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

torch.set_float32_matmul_precision('high')
from pathlib import Path
from tqdm import tqdm

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
from vitpose_model import ViTPoseModel
from hamer.utils import recursive_to
from hamer.configs import get_config
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer
from train import (
    TactileTrainingModule,
    _load_checkpoint,
    file_sha256,
    load_compatible_state_dict,
)

class ViTDetDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, img_cv2, boxes, right, rescale_factor=2.0):
        self.cfg = cfg
        self.img_cv2 = img_cv2
        self.boxes = boxes
        self.right = right
        self.rescale_factor = rescale_factor

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        bbox = self.boxes[idx]
        is_right = self.right[idx]
        
        center = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        scale_pixels = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        bbox_size = scale_pixels * self.rescale_factor
        
        res = self.cfg.MODEL.IMAGE_SIZE
        t = np.zeros((2, 3), dtype=np.float32)
        t[0, 0] = float(res) / bbox_size
        t[1, 1] = float(res) / bbox_size
        t[0, 2] = res * (-float(center[0]) / bbox_size + 0.5)
        t[1, 2] = res * (-float(center[1]) / bbox_size + 0.5)
        
        img_rgb = cv2.cvtColor(self.img_cv2, cv2.COLOR_BGR2RGB)
        img_patch = cv2.warpAffine(img_rgb, t, (res, res), flags=cv2.INTER_LINEAR)
        img_patch = img_patch.astype(np.float32) / 255.0
        
        if is_right == 0:
            img_patch = cv2.flip(img_patch, 1)
            
        img_patch = (img_patch - self.cfg.MODEL.IMAGE_MEAN) / self.cfg.MODEL.IMAGE_STD
        img_patch = img_patch.transpose(2, 0, 1)
        
        return {
            'img': torch.from_numpy(img_patch).float(),
            'right': torch.tensor(is_right, dtype=torch.float32),
            'box_center': torch.from_numpy(center).float(),
            'box_size': torch.tensor(bbox_size).float(),
            'img_size': torch.tensor([self.img_cv2.shape[1], self.img_cv2.shape[0]], dtype=torch.float32)
        }


# Import preprocess module helpers
import trimesh
from generate_video import generate_video_from_images
from concat_videos import concat_videos


DEMO_RENDERER_VERSION = "canonical-software-rasterizer-palm-view-opentouch-bbox-v6"
DEFAULT_TACTILE_RENDER_SIZE = (720, 1280)

# ==============================================================================
# Helpers
# ==============================================================================

def _find_first_existing(cands):
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None

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

def _resolve_checkpoint_relative_path(value, checkpoint_path):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(checkpoint_path).expanduser().resolve().parent / path
    return path.resolve(strict=False)


def _load_tactile_model_metadata(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    config_path = checkpoint_path.parent / "model_config.json"
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    checkpoint = _load_checkpoint(checkpoint_path)
    if isinstance(checkpoint, dict) and checkpoint.get("format") == "tactile_trainable_v2":
        metadata = dict(checkpoint.get("model_config", {}) or {})
        for key in (
            "visual_backbone",
            "visual_backbone_model_name",
            "backbone_weights",
            "backbone_sha256",
            "tactile_head_type",
            "backbone_feature_layers",
            "dino_rezero_source",
            "dino_residual_max_scale",
            "dino_residual_rms_budget",
            "pool_layout",
            "decoder_dropout_scale",
            "bbox_rescale_factor",
        ):
            if checkpoint.get(key) not in (None, "", []):
                metadata[key] = checkpoint[key]
        return metadata
    return {}


def load_models(
    checkpoint_path,
    device,
    dino_weights=None,
    bbox_rescale_factor=None,
    load_detectors=True,
):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
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

    metadata = _load_tactile_model_metadata(checkpoint_path)
    tactile_head_type = str(metadata.get("tactile_head_type", "dense_v2_dino_rezero"))
    visual_backbone = str(metadata.get("visual_backbone", "dinov3_hplus"))
    backbone_feature_layers = tuple(int(layer) for layer in metadata.get("backbone_feature_layers", [8, 16, 24, 32]))
    dino_residual_max_scale = float(metadata.get("dino_residual_max_scale", 0.10))
    dino_residual_rms_budget = float(metadata.get("dino_residual_rms_budget", 0.50))
    pool_layout = str(metadata.get("pool_layout", "fullgrid32"))
    decoder_dropout_scale = float(metadata.get("decoder_dropout_scale", 1.0))
    resolved_bbox_rescale_factor = float(
        bbox_rescale_factor
        if bbox_rescale_factor is not None
        else metadata.get("bbox_rescale_factor", 2.0)
    )
    if not 1.0 <= resolved_bbox_rescale_factor <= 4.0:
        raise ValueError("bbox_rescale_factor must lie in [1.0, 4.0]")

    if visual_backbone != "dinov3_hplus":
        raise ValueError(f"Unsupported visual_backbone in checkpoint metadata: {visual_backbone}")
    local_dino_weights = Path(workspace_dir) / "_DATA" / (
        "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    )
    weight_candidates = []
    for value in (dino_weights, metadata.get("backbone_weights"), local_dino_weights):
        if not value:
            continue
        candidate = _resolve_checkpoint_relative_path(value, checkpoint_path)
        if candidate not in weight_candidates:
            weight_candidates.append(candidate)
    backbone_weights = next((path for path in weight_candidates if path.is_file()), None)
    if backbone_weights is None:
        raise FileNotFoundError(
            "DINOv3 frozen weights were not found. Tried: "
            f"{[str(path) for path in weight_candidates]}"
        )
    if not backbone_weights.is_file():
        raise FileNotFoundError(f"Frozen backbone weights not found: {backbone_weights}")

    print(
        ">>> Initializing tactile model: "
        f"head={tactile_head_type}, backbone={visual_backbone}, "
        f"layers={backbone_feature_layers}, pool={pool_layout}, "
        f"bbox_rescale={resolved_bbox_rescale_factor:g}"
    )
    model = TactileTrainingModule(
        cfg=model_cfg,
        tactile_only_forward=True,
        tactile_head_type=tactile_head_type,
        backbone_feature_layers=backbone_feature_layers,
        visual_backbone=visual_backbone,
        dino_weights=str(backbone_weights),
        dino_rezero_source="multilevel",
        dino_residual_max_scale=dino_residual_max_scale,
        dino_residual_rms_budget=dino_residual_rms_budget,
        pool_layout=pool_layout,
        decoder_dropout_scale=decoder_dropout_scale,
        bbox_rescale_factor=resolved_bbox_rescale_factor,
    )
    model.visual_backbone_model_name = str(metadata.get("visual_backbone_model_name", ""))
    model.backbone_weights_path = str(backbone_weights)
    expected_hash = str(metadata.get("backbone_sha256", "") or "")
    model.backbone_weights_sha256 = file_sha256(backbone_weights) if expected_hash else ""

    dummy_input = torch.zeros(1, 3, model_cfg.MODEL.IMAGE_SIZE, model_cfg.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        dummy_feat = model._extract_tactile_features(dummy_input[:, :, :, 32:-32])
        model.tactile_head(dummy_feat)
        print(f">>> Tactile head initialized with output dim: {model.tactile_dim}")
    print(f">>> Loading weights from checkpoint: {checkpoint_path}")
    load_compatible_state_dict(
        model,
        checkpoint_path,
        load_backbone=False,
    )
    model = model.to(device)
    model.eval()

    if not load_detectors:
        print(">>> Using external bbox source; skipping ViTDet and ViTPose initialization.")
        return model, None, None, model_cfg

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


def _load_subdiv_palm_support(tactile_dim):
    palm_faces_path = os.path.join(
        workspace_dir,
        "opentouch",
        "preprocess",
        "scratch",
        "auto_calibrated_palm_subdiv_faces.json",
    )
    with open(palm_faces_path, "r", encoding="utf-8") as handle:
        palm_data = json.load(handle)

    palm_vertices = {
        vertex_id
        for triplet in palm_data["group_negative"]["face_triplets"]
        for vertex_id in triplet
        if 0 <= vertex_id < tactile_dim
    }
    palm_mask = np.zeros(tactile_dim, dtype=np.float32)
    palm_mask[list(palm_vertices)] = 1.0
    palm_face_indices = np.asarray(
        palm_data["group_negative"]["face_indices"], dtype=np.int32
    )
    return palm_mask, palm_face_indices


def _load_subdiv_palm_mask(tactile_dim):
    return _load_subdiv_palm_support(tactile_dim)[0]


def _opentouch_pressure_rgb(pressure):
    """Match OpenTouch's subdiv tactile-signal convention exactly."""
    from matplotlib import cm

    normalized_pressure = np.clip(np.asarray(pressure, dtype=np.float32), 0.0, 1.0)
    return np.asarray(cm.gnuplot2(1.0 - normalized_pressure)[..., :3], dtype=np.float32)


def _apply_display_pressure_floor(pressure, display_floor):
    """Suppress low-confidence pressure only in the rendered visualization."""
    if not 0.0 <= display_floor <= 1.0:
        raise ValueError("display_floor must lie in [0, 1]")
    pressure = np.asarray(pressure, dtype=np.float32)
    return np.where(pressure < display_floor, 0.0, pressure)


def _pressure_vertex_colors(pressure, display_floor, display_gamma):
    del display_gamma
    pressure = _apply_display_pressure_floor(pressure, display_floor)
    colors_rgb = _opentouch_pressure_rgb(pressure)
    colors = np.concatenate(
        [np.round(colors_rgb * 255.0).astype(np.uint8), np.full((len(pressure), 1), 255, dtype=np.uint8)],
        axis=1,
    )
    return colors


def _draw_pressure_colorbar(image_rgb, display_floor, display_gamma):
    """Draw the OpenTouch pressure mapping with the visualization cutoff."""
    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    height, width = image.shape[:2]
    bar_width = max(18, width // 36)
    margin_right = max(20, width // 32)
    label_margin = max(12, width // 60)
    x1 = width - margin_right
    x0 = x1 - bar_width
    y0 = max(50, height // 12)
    y1 = height - y0
    del display_gamma
    physical_pressure = np.linspace(1.0, 0.0, y1 - y0 + 1, dtype=np.float32)
    displayed_pressure = _apply_display_pressure_floor(physical_pressure, display_floor)
    bar_rgb = _opentouch_pressure_rgb(displayed_pressure)
    image[y0 : y1 + 1, x0:x1] = np.round(bar_rgb[:, None, :] * 255.0).astype(np.uint8)
    cv2.rectangle(image, (x0 - 1, y0 - 1), (x1, y1 + 1), (230, 230, 230), 1)

    label_x = max(4, x0 - 6 * label_margin)
    label_color = (235, 235, 235)
    font_scale = max(0.38, min(0.65, height / 1600.0))
    labels = [(1.0, y0 + 6), (0.5, (y0 + y1) // 2 + 6)]
    if 0.0 < display_floor < 0.5:
        floor_y = y0 + (1.0 - display_floor) * (y1 - y0)
        labels.append((display_floor, int(floor_y)))
    labels.append((0.0, y1))
    for value, y in labels:
        text = f"{value:.2f}" if value < 0.1 else f"{value:.1f}"
        cv2.putText(
            image,
            text,
            (label_x, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            label_color,
            1,
            cv2.LINE_AA,
        )
    return image


def _draw_panel_title(image, title):
    if not title:
        return image
    image = np.asarray(image, dtype=np.uint8).copy()
    title = str(title).upper()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.65, min(1.15, image.shape[0] / 900.0))
    thickness = max(2, int(round(font_scale * 2.0)))
    (text_width, text_height), baseline = cv2.getTextSize(
        title, font, font_scale, thickness
    )
    x0, y0 = 18, 18
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + text_width + 24, y0 + text_height + baseline + 20),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + text_width + 24, y0 + text_height + baseline + 20),
        (32, 32, 32),
        2,
    )
    cv2.putText(
        image,
        title,
        (x0 + 12, y0 + text_height + 8),
        font,
        font_scale,
        (24, 24, 24),
        thickness,
        cv2.LINE_AA,
    )
    return image


def _draw_panel_notice(image, notice):
    if not notice:
        return image
    image = np.asarray(image, dtype=np.uint8).copy()
    lines = [line.strip().upper() for line in str(notice).splitlines() if line.strip()]
    if not lines:
        return image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(0.85, image.shape[0] / 1200.0))
    thickness = max(1, int(round(font_scale * 2.0)))
    measurements = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_height = max(height for _width, height in measurements) + max(10, image.shape[0] // 90)
    box_width = max(width for width, _height in measurements) + 36
    box_height = line_height * len(lines) + 24
    x0 = max(12, (image.shape[1] - box_width) // 2)
    y0 = image.shape[0] - box_height - max(20, image.shape[0] // 28)
    cv2.rectangle(image, (x0, y0), (x0 + box_width, y0 + box_height), (255, 255, 255), -1)
    cv2.rectangle(image, (x0, y0), (x0 + box_width, y0 + box_height), (210, 40, 40), 3)
    for line_index, (line, (text_width, text_height)) in enumerate(zip(lines, measurements)):
        text_x = x0 + (box_width - text_width) // 2
        text_y = y0 + 16 + text_height + line_index * line_height
        cv2.putText(
            image,
            line,
            (text_x, text_y),
            font,
            font_scale,
            (185, 25, 25),
            thickness,
            cv2.LINE_AA,
        )
    return image


def _parse_render_size(value):
    try:
        width_text, height_text = str(value).lower().split("x", maxsplit=1)
        width, height = int(width_text), int(height_text)
    except (TypeError, ValueError) as error:
        raise ValueError("tactile render size must use WIDTHxHEIGHT, for example 720x1280") from error
    if width < 128 or height < 128:
        raise ValueError("tactile render width and height must each be at least 128")
    return width, height


def _probe_video_rotation(video_path):
    """Read the display rotation metadata without relying on OpenCV defaults."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(result.stdout).get("streams", [{}])[0]
        rotate_tag = stream.get("tags", {}).get("rotate")
        side_rotation = next(
            (
                entry["rotation"]
                for entry in stream.get("side_data_list", [])
                if entry.get("rotation") is not None
            ),
            None,
        )
        if rotate_tag is not None:
            rotation = int(round(float(rotate_tag)))
        elif side_rotation is not None:
            # FFprobe reports the display-matrix angle with the opposite sign
            # from cv2.ROTATE_*'s clockwise convention.
            rotation = -int(round(float(side_rotation)))
        else:
            rotation = 0
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, TypeError, IndexError):
        rotation = 0
    rotation %= 360
    return rotation if rotation in {0, 90, 180, 270} else 0


def _resolve_video_rotation(value, video_path):
    return _probe_video_rotation(video_path) if value == "auto" else int(value)


def _rotate_bgr_frame(frame, rotation):
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _numbered_frame_paths(frame_dir, demo_id):
    frame_dir = Path(frame_dir)
    paths = sorted(frame_dir.glob(f"{demo_id}_*.png"))
    if not paths:
        paths = sorted(frame_dir.glob(f"{demo_id}_*.jpg"))
    return paths


def _clear_numbered_frames(frame_dir, demo_id):
    frame_dir = Path(frame_dir)
    for suffix in ("png", "jpg"):
        for path in frame_dir.glob(f"{demo_id}_*.{suffix}"):
            path.unlink()


def _video_fps(video_path, fallback_video=None):
    for candidate in (video_path, fallback_video):
        if candidate is None or not Path(candidate).is_file():
            continue
        capture = cv2.VideoCapture(str(candidate))
        fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0.0
        capture.release()
        if fps and np.isfinite(fps):
            return float(fps)
    return 30.0


def _prepare_rgb_frames(video_path, rgb_dir, demo_id, video_rotation, reuse_existing=False):
    """Extract display-oriented frames or validate and reuse an existing sequence."""
    rgb_dir = Path(rgb_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if reuse_existing:
        frame_paths = _numbered_frame_paths(rgb_dir, demo_id)
        if not frame_paths:
            raise FileNotFoundError(
                f"--skip_frame_extraction requires existing frames in: {rgb_dir}"
            )
        first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first_frame is None:
            raise ValueError(f"Failed to read existing RGB frame: {frame_paths[0]}")
        height, width = first_frame.shape[:2]
        fps = _video_fps(video_path, rgb_dir.parent / "rgb.mp4")
        print(f"Reusing {len(frame_paths)} RGB frames at {width}x{height} from {rgb_dir}")
        return frame_paths, fps, (width, height)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps):
        fps = 30.0
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    encoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    encoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_rotation in {90, 270}:
        target_size = (encoded_height, encoded_width)
    else:
        target_size = (encoded_width, encoded_height)
    print(
        f"Source display rotation: {video_rotation} degrees; "
        f"RGB frame size: {target_size[0]}x{target_size[1]}"
    )

    _clear_numbered_frames(rgb_dir, demo_id)
    frame_paths = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = _rotate_bgr_frame(frame, video_rotation)
        if (frame.shape[1], frame.shape[0]) != target_size:
            frame = cv2.resize(frame, target_size)
        frame_path = rgb_dir / f"{demo_id}_{frame_index:05d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Failed to write RGB frame: {frame_path}")
        frame_paths.append(frame_path)
        frame_index += 1
    capture.release()
    if not frame_paths:
        raise ValueError(f"No frames could be read from video: {video_path}")
    print(f"Extracted {len(frame_paths)} RGB frames.")
    return frame_paths, float(fps), target_size


def _model_crop_bbox(tight_bbox, rescale_factor):
    tight_bbox = np.asarray(tight_bbox, dtype=np.float32)
    center_x = 0.5 * float(tight_bbox[0] + tight_bbox[2])
    center_y = 0.5 * float(tight_bbox[1] + tight_bbox[3])
    side = max(float(tight_bbox[2] - tight_bbox[0]), float(tight_bbox[3] - tight_bbox[1]))
    half_side = 0.5 * side * float(rescale_factor)
    return np.asarray(
        [center_x - half_side, center_y - half_side, center_x + half_side, center_y + half_side],
        dtype=np.float32,
    )


def _save_hand_bbox_data(
    output_path,
    tight_bboxes,
    crop_bboxes,
    rescale_factor,
    source="vitpose",
    track_id=-1,
):
    output_path = Path(output_path)
    np.savez_compressed(
        output_path,
        tight=np.asarray(tight_bboxes, dtype=np.float32),
        crop=np.asarray(crop_bboxes, dtype=np.float32),
        detected=np.isfinite(np.asarray(tight_bboxes, dtype=np.float32)).all(axis=1),
        rescale_factor=np.asarray(float(rescale_factor), dtype=np.float32),
        source=np.asarray(str(source)),
        track_id=np.asarray(int(track_id), dtype=np.int64),
    )
    print(f">>> Saved hand-box provenance: {output_path}")


def _load_hand_bbox_data(path, expected_frames=None):
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path) as data:
        tight = np.asarray(data["tight"], dtype=np.float32)
        crop = np.asarray(data["crop"], dtype=np.float32)
        detected = np.asarray(
            data["detected"] if "detected" in data.files else np.isfinite(tight).all(axis=1),
            dtype=bool,
        )
        rescale_factor = float(
            np.asarray(data["rescale_factor"] if "rescale_factor" in data.files else 1.0).reshape(())
        )
        source = str(np.asarray(data["source"]).reshape(())) if "source" in data.files else "unknown"
        track_id = int(np.asarray(data["track_id"]).reshape(())) if "track_id" in data.files else -1
    if tight.ndim != 2 or tight.shape[1] != 4 or crop.shape != tight.shape:
        raise ValueError(f"Invalid hand-box arrays in {path}: tight={tight.shape}, crop={crop.shape}")
    if detected.shape != (len(tight),):
        raise ValueError(f"Invalid detected mask in {path}: {detected.shape}")
    if expected_frames is not None and len(tight) != int(expected_frames):
        raise ValueError(
            f"Hand-box frame count mismatch in {path}: {len(tight)} vs {expected_frames} RGB frames"
        )
    return {
        "tight": tight,
        "crop": crop,
        "detected": detected,
        "rescale_factor": rescale_factor,
        "source": source,
        "track_id": track_id,
    }


def _load_sam3_bbox_jsonl(path, expected_frames, rescale_factor, track_id=None):
    """Load one anonymous SAM3 track from the environment-neutral JSONL output."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SAM3 bbox JSONL not found: {path}")

    rows_by_frame = {}
    track_counts = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            frame_index = int(row.get("frame_index", -1))
            if not 0 <= frame_index < int(expected_frames):
                raise ValueError(
                    f"SAM3 frame_index {frame_index} at {path}:{line_number} lies outside "
                    f"the {expected_frames}-frame RGB sequence"
                )
            if frame_index in rows_by_frame:
                raise ValueError(f"Duplicate SAM3 frame_index {frame_index} in {path}")
            tracks = row.get("tracks", [])
            if not isinstance(tracks, list):
                raise ValueError(f"SAM3 tracks must be a list at {path}:{line_number}")
            rows_by_frame[frame_index] = tracks
            for track in tracks:
                candidate_id = int(track["track_id"])
                track_counts[candidate_id] = track_counts.get(candidate_id, 0) + 1

    if not track_counts:
        raise ValueError(
            f"SAM3 produced no accepted hand tracks in {path}; inspect preview.mp4 and track_audit.json"
        )
    available_ids = sorted(track_counts)
    if track_id is None:
        if len(available_ids) != 1:
            counts = ", ".join(f"{candidate}:{track_counts[candidate]}" for candidate in available_ids)
            raise ValueError(
                "SAM3 returned multiple anonymous tracks. Select one explicitly with "
                f"--sam3_track_id. Available track_id:frame_count = {counts}"
            )
        selected_track_id = available_ids[0]
    else:
        selected_track_id = int(track_id)
        if selected_track_id not in track_counts:
            raise ValueError(
                f"SAM3 track {selected_track_id} is absent from {path}; available IDs: {available_ids}"
            )

    tight_bboxes = np.full((int(expected_frames), 4), np.nan, dtype=np.float32)
    crop_bboxes = np.full((int(expected_frames), 4), np.nan, dtype=np.float32)
    for frame_index, tracks in rows_by_frame.items():
        candidates = [track for track in tracks if int(track["track_id"]) == selected_track_id]
        if not candidates:
            continue
        if len(candidates) > 1:
            candidates.sort(
                key=lambda track: float(track.get("prompt_score") or float("-inf")),
                reverse=True,
            )
        bbox = np.asarray(candidates[0].get("bbox", []), dtype=np.float32)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError(f"Invalid SAM3 bbox for frame {frame_index}, track {selected_track_id}: {bbox}")
        if not bbox[2] > bbox[0] or not bbox[3] > bbox[1]:
            raise ValueError(f"Degenerate SAM3 bbox for frame {frame_index}: {bbox.tolist()}")
        tight_bboxes[frame_index] = bbox
        crop_bboxes[frame_index] = _model_crop_bbox(bbox, rescale_factor)

    detected = np.isfinite(tight_bboxes).all(axis=1)
    print(
        f">>> Loaded SAM3 track {selected_track_id} from {path}: "
        f"{int(detected.sum())}/{expected_frames} accepted frames"
    )
    return {
        "tight": tight_bboxes,
        "crop": crop_bboxes,
        "detected": detected,
        "rescale_factor": float(rescale_factor),
        "source": "sam3",
        "track_id": selected_track_id,
    }


def _draw_bbox_rectangle(image, bbox, color, thickness, label):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    clipped_x0 = min(max(x0, 0), width - 1)
    clipped_y0 = min(max(y0, 0), height - 1)
    clipped_x1 = min(max(x1, 0), width - 1)
    clipped_y1 = min(max(y1, 0), height - 1)
    cv2.rectangle(image, (clipped_x0, clipped_y0), (clipped_x1, clipped_y1), color, thickness)
    label_y = max(24, clipped_y0 - 8)
    cv2.putText(
        image,
        label,
        (clipped_x0 + 4, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def render_hand_bbox_sequence(frame_paths, bbox_data, output_dir, demo_id):
    """Render the tight hand box and exact square crop consumed by the model."""
    if bbox_data is None:
        print(">>> Hand-box data is unavailable; skipping bbox.mp4")
        return False
    if len(frame_paths) != len(bbox_data["tight"]):
        raise ValueError("RGB and hand-box frame counts must match")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_numbered_frames(output_dir, demo_id)
    detected_count = int(np.count_nonzero(bbox_data["detected"]))
    print(
        f">>> Target hand detected in {detected_count}/{len(frame_paths)} frames "
        f"({100.0 * detected_count / max(len(frame_paths), 1):.1f}%)"
    )
    for frame_index, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read RGB frame: {frame_path}")
        source_label = str(bbox_data.get("source", "unknown")).upper()
        if int(bbox_data.get("track_id", -1)) >= 0:
            source_label += f" TRACK {int(bbox_data['track_id'])}"
        cv2.putText(
            image,
            source_label,
            (24, image.shape[0] - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        if bbox_data["detected"][frame_index]:
            _draw_bbox_rectangle(
                image,
                bbox_data["crop"][frame_index],
                (0, 165, 255),
                3,
                f"model crop x{bbox_data['rescale_factor']:g}",
            )
            _draw_bbox_rectangle(
                image,
                bbox_data["tight"][frame_index],
                (60, 220, 80),
                2,
                "hand bbox",
            )
        else:
            cv2.putText(
                image,
                "TARGET HAND NOT DETECTED",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (40, 40, 240),
                2,
                cv2.LINE_AA,
            )
        output_path = output_dir / f"{demo_id}_{frame_index:05d}.png"
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to write bbox frame: {output_path}")
    return True


class CanonicalSubdivRasterizer:
    """Software rasterizer for the static canonical subdiv hand mesh.

    The geometry is fixed for a demo, so face visibility and barycentric weights
    are built once. Each subsequent frame only interpolates vertex colors.
    """

    def __init__(
        self,
        vertices,
        faces,
        image_size,
        palm_face_indices=None,
        view_mode="palm",
        background_rgb=(24, 28, 32),
        mirror_horizontal=False,
    ):
        if view_mode not in {"palm", "legacy"}:
            raise ValueError("view_mode must be 'palm' or 'legacy'")
        width, height = image_size
        self.width = int(width)
        self.height = int(height)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.vertex_count = int(len(vertices))
        self.background_rgb = np.asarray(background_rgb, dtype=np.uint8)
        self.mirror_horizontal = bool(mirror_horizontal)

        rotation = (
            trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
            @ trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        )
        vertices_h = np.concatenate(
            [np.asarray(vertices, dtype=np.float32), np.ones((len(vertices), 1), dtype=np.float32)],
            axis=1,
        )
        base_camera_vertices = (rotation @ vertices_h.T).T[:, :3]
        candidate_vertices = [("legacy", base_camera_vertices)]
        if view_mode == "palm":
            opposite_camera_vertices = base_camera_vertices.copy()
            # A 180-degree camera-frame yaw looks at the opposite hand surface
            # while retaining an upright canonical hand presentation.
            opposite_camera_vertices[:, (0, 2)] *= -1.0
            candidate_vertices.append(("opposite", opposite_camera_vertices))

        palm_face_indices = np.asarray(
            palm_face_indices if palm_face_indices is not None else [], dtype=np.int32
        )
        palm_face_indices = palm_face_indices[
            (palm_face_indices >= 0) & (palm_face_indices < len(self.faces))
        ]

        candidates = []
        for name, camera_vertices in candidate_vertices:
            camera_vertices = camera_vertices.copy()
            if self.mirror_horizontal:
                # Mirror in camera space so handedness changes left/right on
                # screen without flipping the canonical hand upside down.
                camera_vertices[:, 0] *= -1.0
            camera_vertices[:, 2] += 2.0
            screen_vertices = self._project(camera_vertices)
            encoded_ids = self._rasterize_face_ids(screen_vertices, camera_vertices)
            if len(palm_face_indices):
                visible_face_indices = encoded_ids[encoded_ids >= 0]
                palm_pixel_count = int(
                    np.count_nonzero(np.isin(visible_face_indices, palm_face_indices))
                )
            else:
                palm_pixel_count = 0
            candidates.append((palm_pixel_count, name, screen_vertices, encoded_ids))

        if view_mode == "palm" and len(palm_face_indices):
            _, selected_name, screen_vertices, encoded_ids = max(
                candidates, key=lambda item: item[0]
            )
        else:
            _, selected_name, screen_vertices, encoded_ids = candidates[0]
        candidate_scores = {name: score for score, name, _, _ in candidates}
        self.view_name = selected_name
        print(
            "Canonical tactile view: "
            f"{selected_name} (visible palm pixels={candidate_scores})"
        )

        self._build_barycentric_lookup(screen_vertices, encoded_ids)

    def _project(self, camera_vertices):
        focal_length = 8000.0 * (self.width / 1280.0)
        screen_vertices = np.empty((len(camera_vertices), 2), dtype=np.float32)
        screen_vertices[:, 0] = focal_length * camera_vertices[:, 0] / camera_vertices[:, 2] + self.width / 2.0
        screen_vertices[:, 1] = -focal_length * camera_vertices[:, 1] / camera_vertices[:, 2] + self.height / 2.0
        return screen_vertices

    def _rasterize_face_ids(self, screen_vertices, camera_vertices):
        face_depth = camera_vertices[self.faces, 2].mean(axis=1)
        id_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Painter's algorithm is sufficient for this non-self-intersecting mesh.
        for face_index in np.argsort(face_depth)[::-1]:
            triangle = screen_vertices[self.faces[face_index]]
            if not np.isfinite(triangle).all():
                continue
            encoded_id = int(face_index) + 1
            cv2.fillConvexPoly(
                id_image,
                np.rint(triangle).astype(np.int32),
                (encoded_id & 255, (encoded_id >> 8) & 255, (encoded_id >> 16) & 255),
                lineType=cv2.LINE_8,
            )

        return (
            id_image[:, :, 0].astype(np.int32)
            + (id_image[:, :, 1].astype(np.int32) << 8)
            + (id_image[:, :, 2].astype(np.int32) << 16)
            - 1
        )

    def _build_barycentric_lookup(self, screen_vertices, encoded_ids):
        rows, cols = np.nonzero(encoded_ids >= 0)
        if not len(rows):
            raise RuntimeError("Canonical mesh rasterization produced no visible pixels")
        face_indices = encoded_ids[rows, cols]
        pixel_faces = self.faces[face_indices]
        triangles = screen_vertices[pixel_faces]
        points = np.stack([cols, rows], axis=1).astype(np.float32)
        denominator = (
            (triangles[:, 1, 1] - triangles[:, 2, 1]) * (triangles[:, 0, 0] - triangles[:, 2, 0])
            + (triangles[:, 2, 0] - triangles[:, 1, 0]) * (triangles[:, 0, 1] - triangles[:, 2, 1])
        )
        valid = np.abs(denominator) > 1e-8
        triangles = triangles[valid]
        pixel_faces = pixel_faces[valid]
        points = points[valid]
        rows = rows[valid]
        cols = cols[valid]
        denominator = denominator[valid]
        w0 = (
            (triangles[:, 1, 1] - triangles[:, 2, 1]) * (points[:, 0] - triangles[:, 2, 0])
            + (triangles[:, 2, 0] - triangles[:, 1, 0]) * (points[:, 1] - triangles[:, 2, 1])
        ) / denominator
        w1 = (
            (triangles[:, 2, 1] - triangles[:, 0, 1]) * (points[:, 0] - triangles[:, 2, 0])
            + (triangles[:, 0, 0] - triangles[:, 2, 0]) * (points[:, 1] - triangles[:, 2, 1])
        ) / denominator
        self.rows = rows
        self.cols = cols
        self.pixel_vertices = pixel_faces
        self.barycentric_weights = np.stack([w0, w1, 1.0 - w0 - w1], axis=1).astype(np.float32)

    def render(self, vertex_colors):
        vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)
        if vertex_colors.shape != (self.vertex_count, 4):
            raise ValueError("Vertex color shape does not match the canonical subdiv mesh")
        sampled = vertex_colors[self.pixel_vertices, :3].astype(np.float32)
        pixel_colors = np.einsum("nij,ni->nj", sampled, self.barycentric_weights)
        image = np.broadcast_to(
            self.background_rgb,
            (self.height, self.width, 3),
        ).copy()
        image[self.rows, self.cols] = np.clip(pixel_colors, 0, 255).astype(np.uint8)
        return image


def render_subdiv_tactile_sequence(
    pressure_seq,
    output_dir,
    demo_id,
    target_size=DEFAULT_TACTILE_RENDER_SIZE,
    temporal_alpha=0.4,
    display_floor=0.05,
    display_gamma=0.65,
    hand_side="right",
    canonical_view="palm",
    valid_frames=None,
    panel_label=None,
    invalid_frame_notices=None,
):
    if not 0.0 <= temporal_alpha <= 1.0:
        raise ValueError("temporal_alpha must lie in [0, 1]")
    if valid_frames is not None:
        valid_frames = np.asarray(valid_frames, dtype=bool)
        if valid_frames.shape != (len(pressure_seq),):
            raise ValueError(
                f"valid_frames must have shape ({len(pressure_seq)},), got {valid_frames.shape}"
            )
    if invalid_frame_notices is not None and len(invalid_frame_notices) != len(pressure_seq):
        raise ValueError(
            "invalid_frame_notices must have the same length as pressure_seq"
        )
    width, height = target_size
    
    # Find subdiv OBJ.
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

    _, palm_face_indices = _load_subdiv_palm_support(len(mano_vertices))
    renderer = CanonicalSubdivRasterizer(
        vertices=mano_vertices,
        faces=mano_faces,
        image_size=(width, height),
        palm_face_indices=palm_face_indices,
        view_mode=canonical_view,
        mirror_horizontal=hand_side == "left",
    )
    os.makedirs(output_dir, exist_ok=True)
    _clear_numbered_frames(output_dir, demo_id)
    previous = None

    print(f"Rendering canonical subdiv tactile sequence to {output_dir} (software rasterizer)...")
    for idx, pressure in enumerate(pressure_seq):
        pressure = np.asarray(pressure, dtype=np.float32)
        if pressure.shape != (mano_vertices.shape[0],):
            raise ValueError(
                f"Predicted tactile shape {pressure.shape} does not match "
                f"subdiv vertex count {mano_vertices.shape[0]}"
            )
        if valid_frames is not None and not valid_frames[idx]:
            previous = None
            image_rgb = renderer.render(
                _pressure_vertex_colors(
                    np.zeros_like(pressure),
                    display_floor=display_floor,
                    display_gamma=display_gamma,
                )
            )
            image_rgb = _draw_pressure_colorbar(
                image_rgb,
                display_floor=display_floor,
                display_gamma=display_gamma,
            )
            image_rgb = _draw_panel_title(image_rgb, panel_label)
            notice = (
                invalid_frame_notices[idx]
                if invalid_frame_notices is not None
                else "NO SAM3 BBOX\nPRESSURE SUPPRESSED"
            )
            image_rgb = _draw_panel_notice(image_rgb, notice)
            cv2.imwrite(
                os.path.join(output_dir, f"{demo_id}_{idx:05d}.png"),
                image_rgb[:, :, ::-1],
            )
            continue
        if previous is not None and temporal_alpha:
            pressure = temporal_alpha * pressure + (1.0 - temporal_alpha) * previous
        previous = pressure

        image_rgb = renderer.render(
            _pressure_vertex_colors(
                pressure,
                display_floor=display_floor,
                display_gamma=display_gamma,
            )
        )
        image_rgb = _draw_pressure_colorbar(
            image_rgb,
            display_floor=display_floor,
            display_gamma=display_gamma,
        )
        image_rgb = _draw_panel_title(image_rgb, panel_label)
        cv2.imwrite(
            os.path.join(output_dir, f"{demo_id}_{idx:05d}.png"),
            image_rgb[:, :, ::-1],
        )
    print("Canonical tactile rendering completed.")


def _compose_demo_videos(
    output_path,
    demo_id,
    fps,
    rgb_size,
    tactile_size,
    combined_layout="horizontal",
):
    rgb_video = output_path / "rgb.mp4"
    pred_touch_video = output_path / "pred_touch.mp4"
    combined_video = output_path / "combined.mp4"
    generate_video_from_images(demo_id, str(output_path / "pred_touch"), str(pred_touch_video), fps=fps)

    rgb_width, rgb_height = rgb_size
    tactile_width, tactile_height = tactile_size
    if combined_layout not in {"auto", "horizontal", "vertical"}:
        raise ValueError("combined_layout must be 'auto', 'horizontal', or 'vertical'")
    # Keep the source orientation intact and place RGB/tactile panels side by
    # side by default. ``auto`` remains a legacy alias for this layout.
    layout = "horizontal" if combined_layout == "auto" else combined_layout
    if layout == "vertical":
        # Preserve portrait viewing: one complete panel above the other.
        concat_videos(
            videos=[rgb_video, pred_touch_video],
            output=combined_video,
            layout="vertical",
            scale_width=tactile_width,
        )
    else:
        concat_videos(
            videos=[rgb_video, pred_touch_video],
            output=combined_video,
            layout="horizontal",
            scale_height=tactile_height,
        )
    print(f">>> Combined video saved to: {combined_video}")


def rerender_saved_tactile_output(
    output_dir,
    hand_side="right",
    temporal_alpha=0.4,
    display_floor=0.05,
    display_gamma=0.65,
    canonical_view="palm",
    tactile_size=DEFAULT_TACTILE_RENDER_SIZE,
    combined_layout="horizontal",
):
    """Re-render an existing demo from its saved masked tactile predictions."""
    output_path = Path(output_dir).expanduser().resolve()
    pressure_path = output_path / "pred_tactile_palm_masked.npy"
    bbox_path = output_path / "pred_hand_bboxes.npz"
    rgb_dir = output_path / "rgb"
    rgb_images = sorted(rgb_dir.glob("*.png"))
    if not pressure_path.is_file():
        raise FileNotFoundError(f"Missing saved masked tactile predictions: {pressure_path}")
    if not rgb_images:
        raise FileNotFoundError(f"Missing RGB frames required for output size: {rgb_dir}")

    reference_frame = cv2.imread(str(rgb_images[0]), cv2.IMREAD_COLOR)
    if reference_frame is None:
        raise ValueError(f"Failed to read RGB frame: {rgb_images[0]}")
    height, width = reference_frame.shape[:2]
    frame_stem = rgb_images[0].stem
    prefix, _, suffix = frame_stem.rpartition("_")
    demo_id = prefix if suffix.isdecimal() else frame_stem
    pressure_seq = np.load(pressure_path)
    pred_touch_dir = output_path / "pred_touch"
    bbox_data = _load_hand_bbox_data(bbox_path, expected_frames=len(rgb_images))
    valid_frames = (
        bbox_data["detected"]
        if bbox_data is not None and bbox_data.get("source") == "sam3"
        else None
    )

    print(f">>> Re-rendering saved tactile predictions from: {pressure_path}")
    render_subdiv_tactile_sequence(
        pressure_seq=pressure_seq,
        output_dir=str(pred_touch_dir),
        demo_id=demo_id,
        target_size=tactile_size,
        temporal_alpha=temporal_alpha,
        display_floor=display_floor,
        display_gamma=display_gamma,
        hand_side=hand_side,
        canonical_view=canonical_view,
        valid_frames=valid_frames,
    )

    rgb_video = output_path / "rgb.mp4"
    if not rgb_video.is_file():
        print(f">>> Skipping MP4 rebuild because RGB video is missing: {rgb_video}")
        return
    capture = cv2.VideoCapture(str(rgb_video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    fps = fps if fps and np.isfinite(fps) else 30.0
    bbox_video = output_path / "bbox.mp4"
    if render_hand_bbox_sequence(
        frame_paths=rgb_images,
        bbox_data=bbox_data,
        output_dir=output_path / "bbox",
        demo_id=demo_id,
    ):
        generate_video_from_images(
            demo_id,
            str(output_path / "bbox"),
            str(bbox_video),
            fps=fps,
        )
    elif bbox_video.is_file():
        bbox_video.unlink()
    _compose_demo_videos(
        output_path=output_path,
        demo_id=demo_id,
        fps=fps,
        rgb_size=(width, height),
        tactile_size=tactile_size,
        combined_layout=combined_layout,
    )
    print(f">>> Re-rendered tactile and combined video in: {output_path}")


def _canonical_dataset_name(value):
    aliases = {
        "opentouch": "OpenTouch",
        "open_touch": "OpenTouch",
        "ot": "OpenTouch",
        "touchanything": "TouchAnything",
        "touch_anything": "TouchAnything",
        "egotouch": "TouchAnything",
        "ego_touch": "TouchAnything",
        "ta": "TouchAnything",
    }
    raw = str(value or "OpenTouch")
    return aliases.get(raw.lower(), raw)


def _dataset_sequence_identity(meta):
    dataset_name = _canonical_dataset_name(meta.get("dataset"))
    if dataset_name == "TouchAnything":
        values = (
            meta.get("split", ""),
            meta.get("scene", ""),
            meta.get("task", ""),
            meta.get("clip", meta.get("rel_clip", "")),
        )
    else:
        values = (meta.get("scene", ""), meta.get("demo", ""))
    return dataset_name, tuple(str(value) for value in values)


def _safe_output_component(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return value or "sequence"


def _valid_dataset_bbox(value):
    try:
        bbox = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return (
        bbox.shape == (4,)
        and bool(np.isfinite(bbox).all())
        and bool(np.all(bbox[2:4] - bbox[0:2] > 1.0))
    )


def _dataset_frame_record(sample_dir, meta, expected_identity, hand_side):
    dataset_name, identity = _dataset_sequence_identity(meta)
    if (dataset_name, identity) != expected_identity:
        return None
    is_right = 1 if hand_side == "right" else 0
    if dataset_name == "TouchAnything":
        hand_meta = meta.get("hands", {}).get(hand_side, {})
        bbox = hand_meta.get("bbox_chest")
        pressure = hand_meta.get("gaussian_pressure")
        image_name = meta.get("views", {}).get("chest", "chest.jpg")
    else:
        meta_is_right = int(meta.get("is_right", 1))
        if meta_is_right != is_right:
            return None
        bbox = meta.get("bbox")
        side = "right" if meta_is_right else "left"
        pressure = meta.get("original_hdf5_data", {}).get(
            f"{side}_pressure_continuous_subdiv"
        )
        if pressure is None:
            pressure = meta.get("gaussian_pressure")
        image_name = meta.get("image", "image.jpg")
    image_path = Path(sample_dir) / str(image_name)
    if not image_path.is_file():
        return None
    bbox_valid = _valid_dataset_bbox(bbox)
    bbox_array = np.asarray(bbox, dtype=np.float32) if bbox_valid else None
    if pressure is not None:
        pressure = np.asarray(pressure, dtype=np.float32)
        if pressure.ndim != 1 or not bool(np.isfinite(pressure).all()):
            pressure = None
    return {
        "sample_dir": str(Path(sample_dir).resolve()),
        "dataset": dataset_name,
        "hand": hand_side,
        "is_right": is_right,
        "frame_idx": int(meta.get("frame_idx", 0) or 0),
        "image_path": str(image_path.resolve()),
        "bbox": bbox_array,
        "bbox_valid": bool(bbox_valid),
        "pressure": None if pressure is None else np.clip(pressure, 0.0, 1.0),
    }


def _load_dataset_sequence(sample_path, hand_side, stride=1, max_frames=None):
    sample_path = Path(sample_path).expanduser().resolve()
    if sample_path.is_file() and sample_path.name == "meta.json":
        sample_path = sample_path.parent
    representative_meta_path = sample_path / "meta.json"
    if not representative_meta_path.is_file():
        raise FileNotFoundError(
            "--dataset_sequence must point to one extracted frame directory containing "
            f"meta.json: {sample_path}"
        )
    with representative_meta_path.open("r", encoding="utf-8") as handle:
        representative_meta = json.load(handle)
    expected_identity = _dataset_sequence_identity(representative_meta)

    dataset_name, _identity = expected_identity
    if dataset_name == "TouchAnything" and "__" in sample_path.name:
        directory_prefix = sample_path.name.rsplit("__", 1)[0] + "__"
    elif dataset_name == "OpenTouch" and sample_path.name.count("_") >= 2:
        directory_prefix = sample_path.name.rsplit("_", 2)[0] + "_"
    else:
        directory_prefix = ""
    candidates = (
        sample_path.parent.glob(f"{directory_prefix}*")
        if directory_prefix
        else sample_path.parent.iterdir()
    )

    records = []
    for candidate in candidates:
        meta_path = candidate / "meta.json"
        if not candidate.is_dir() or not meta_path.is_file():
            continue
        try:
            with meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            record = _dataset_frame_record(candidate, meta, expected_identity, hand_side)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if record is not None:
            records.append(record)
    records.sort(key=lambda item: (item["frame_idx"], item["sample_dir"]))
    if not records:
        dataset_name, identity = expected_identity
        raise ValueError(
            f"No {hand_side}-hand timeline frames were found for "
            f"{dataset_name} sequence {identity} under {sample_path.parent}"
        )
    duplicate_indices = [
        records[index]["frame_idx"]
        for index in range(1, len(records))
        if records[index]["frame_idx"] == records[index - 1]["frame_idx"]
    ]
    if duplicate_indices:
        raise ValueError(
            "Dataset sequence contains duplicate frame indices for the selected hand: "
            f"{sorted(set(duplicate_indices))[:10]}"
        )
    dataset_name, identity = expected_identity
    tactile_dims = {
        int(record["pressure"].shape[0])
        for record in records
        if record["pressure"] is not None
    }
    if not tactile_dims:
        raise ValueError(
            f"No {hand_side}-hand pressure GT was found for {dataset_name} sequence {identity}"
        )
    if len(tactile_dims) != 1:
        raise ValueError(f"Dataset sequence has inconsistent GT tactile dimensions: {tactile_dims}")
    records = records[:: int(stride)]
    if max_frames is not None:
        records = records[: int(max_frames)]
    sequence_parts = [dataset_name, *identity, hand_side]
    demo_id = _safe_output_component("__".join(part for part in sequence_parts if part))
    print(
        f">>> Dataset sequence: {dataset_name} {identity}, hand={hand_side}, "
        f"timeline frames={len(records)}, bbox+GT frames="
        f"{sum(record['bbox_valid'] and record['pressure'] is not None for record in records)}, "
        f"source={sample_path.parent}"
    )
    return records, demo_id


def _dataset_manifest_rows(records):
    return [
        {
            "sample_dir": record["sample_dir"],
            "dataset": record["dataset"],
            "hand": record["hand"],
            "frame_idx": record["frame_idx"],
            "image_path": record["image_path"],
            "bbox": None if record["bbox"] is None else record["bbox"].tolist(),
            "bbox_valid": bool(record["bbox_valid"]),
        }
        for record in records
    ]


def _prepare_dataset_rgb_frames(records, rgb_dir, demo_id, reuse_existing=False):
    rgb_dir = Path(rgb_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rgb_dir.parent / "dataset_sequence_manifest.json"
    manifest_rows = _dataset_manifest_rows(records)
    if reuse_existing:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"--skip_frame_extraction requires dataset manifest: {manifest_path}"
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing_rows = json.load(handle)
        if existing_rows != manifest_rows:
            raise ValueError(
                "Saved RGB frames belong to a different dataset sequence or frame selection; "
                "rerun without --skip_frame_extraction"
            )
        paths = _numbered_frame_paths(rgb_dir, demo_id)
        if len(paths) != len(records):
            raise ValueError(
                f"Saved RGB frame count mismatch: {len(paths)} vs {len(records)} manifest rows"
            )
        first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise ValueError(f"Failed to read saved RGB frame: {paths[0]}")
        return paths, (first.shape[1], first.shape[0])

    _clear_numbered_frames(rgb_dir, demo_id)
    output_paths = []
    target_size = None
    for output_index, record in enumerate(records):
        image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode dataset image: {record['image_path']}")
        if target_size is None:
            target_size = (image.shape[1], image.shape[0])
        elif (image.shape[1], image.shape[0]) != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        image = _draw_panel_title(image, "RGB")
        cv2.putText(
            image,
            f"frame {record['frame_idx']}",
            (20, image.shape[0] - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output_path = rgb_dir / f"{demo_id}_{output_index:05d}.png"
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to write dataset RGB frame: {output_path}")
        output_paths.append(output_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_rows, handle, indent=2)
        handle.write("\n")
    return output_paths, target_size


def _predict_dataset_sequence(model, model_cfg, records, device):
    predictions = []
    for record in tqdm(records, desc="Dataset tactile inference"):
        if not record["bbox_valid"]:
            predictions.append(np.zeros(model.tactile_dim, dtype=np.float32))
            continue
        image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not decode dataset image: {record['image_path']}")
        crop_dataset = ViTDetDataset(
            model_cfg,
            image,
            np.asarray([record["bbox"]], dtype=np.float32),
            np.asarray([record["is_right"]], dtype=np.int64),
            rescale_factor=model.bbox_rescale_factor,
        )
        batch = next(iter(torch.utils.data.DataLoader(crop_dataset, batch_size=1, num_workers=0)))
        batch = recursive_to(batch, device)
        with torch.no_grad():
            output = model.forward_step(batch, train=False)
        predictions.append(output["pred_tactile"][0].detach().float().cpu().numpy())
    return np.asarray(predictions, dtype=np.float32)


def _run_dataset_sequence_hand_demo(
    args,
    tactile_size,
    records,
    demo_id,
    hand_side,
    model_bundle=None,
):
    if hand_side not in {"left", "right"}:
        raise ValueError(f"Dataset hand must be concrete, got {hand_side!r}")
    output_path = Path(args.out_dir).expanduser().resolve() / "dataset_sequences" / demo_id
    output_path.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_path / "rgb"
    rgb_paths, rgb_size = _prepare_dataset_rgb_frames(
        records,
        rgb_dir,
        demo_id,
        reuse_existing=args.skip_frame_extraction,
    )
    fps = float(args.dataset_fps)
    generate_video_from_images(
        demo_id, str(rgb_dir), str(output_path / "rgb.mp4"), fps=fps
    )
    if args.prepare_frames_only:
        print(f">>> Dataset RGB preparation complete: {output_path}")
        return

    pressure_values = [record["pressure"] for record in records if record["pressure"] is not None]
    tactile_dim = int(pressure_values[0].shape[0])
    gt_raw = np.zeros((len(records), tactile_dim), dtype=np.float32)
    for frame_index, record in enumerate(records):
        if record["pressure"] is not None:
            gt_raw[frame_index] = record["pressure"]
    bbox_valid_frames = np.asarray(
        [record["bbox_valid"] for record in records],
        dtype=bool,
    )
    pressure_valid_frames = np.asarray(
        [record["pressure"] is not None for record in records],
        dtype=bool,
    )
    palm_mask = _load_subdiv_palm_mask(tactile_dim)
    gt_masked = gt_raw * palm_mask[None, :]
    np.save(output_path / "gt_tactile_raw.npy", gt_raw)
    np.save(output_path / "gt_tactile_palm_masked.npy", gt_masked)
    bbox_path = output_path / "pred_hand_bboxes.npz"

    if args.skip_inference:
        pred_path = output_path / "pred_tactile_palm_masked.npy"
        if not pred_path.is_file():
            raise FileNotFoundError(f"--skip_inference requires predictions: {pred_path}")
        pred_masked = np.asarray(np.load(pred_path), dtype=np.float32)
        if pred_masked.shape != gt_masked.shape:
            raise ValueError(
                f"Saved prediction shape mismatch: {pred_masked.shape} vs GT {gt_masked.shape}"
            )
        bbox_data = _load_hand_bbox_data(bbox_path, expected_frames=len(records))
        if bbox_data is None:
            raise FileNotFoundError(f"--skip_inference requires saved bbox data: {bbox_path}")
    else:
        if model_bundle is None:
            raise RuntimeError("Dataset inference model bundle was not initialized")
        model, model_cfg, device = model_bundle
        pred_raw = _predict_dataset_sequence(model, model_cfg, records, device)
        if pred_raw.shape != gt_raw.shape:
            raise ValueError(
                f"Model prediction shape mismatch: {pred_raw.shape} vs GT {gt_raw.shape}"
            )
        np.save(output_path / "pred_tactile_raw.npy", pred_raw)
        pred_masked = pred_raw * palm_mask[None, :]
        np.save(output_path / "pred_tactile_palm_masked.npy", pred_masked)
        tight = np.full((len(records), 4), np.nan, dtype=np.float32)
        crops = np.full((len(records), 4), np.nan, dtype=np.float32)
        for frame_index, record in enumerate(records):
            if not bbox_valid_frames[frame_index]:
                continue
            tight[frame_index] = record["bbox"]
            crops[frame_index] = _model_crop_bbox(
                record["bbox"], model.bbox_rescale_factor
            )
        _save_hand_bbox_data(
            bbox_path,
            tight_bboxes=tight,
            crop_bboxes=crops,
            rescale_factor=model.bbox_rescale_factor,
            source="dataset",
        )
        bbox_data = _load_hand_bbox_data(bbox_path, expected_frames=len(records))

    bbox_valid_frames = bbox_valid_frames & np.asarray(bbox_data["detected"], dtype=bool)
    pred_valid_frames = bbox_valid_frames
    gt_valid_frames = bbox_valid_frames & pressure_valid_frames
    pred_invalid_notices = [
        None if is_valid else "NO SAM3 BBOX\nPRESSURE SUPPRESSED"
        for is_valid in pred_valid_frames
    ]
    gt_invalid_notices = []
    for bbox_valid, pressure_valid in zip(bbox_valid_frames, pressure_valid_frames):
        if not bbox_valid:
            gt_invalid_notices.append("NO SAM3 BBOX\nGT HIDDEN")
        elif not pressure_valid:
            gt_invalid_notices.append("NO PRESSURE GT")
        else:
            gt_invalid_notices.append(None)
    print(
        f">>> {hand_side} bbox-gated tactile frames: "
        f"{int(bbox_valid_frames.sum())}/{len(bbox_valid_frames)}; "
        "missing frames render a zero-pressure mesh with a reason label"
    )

    bbox_dir = output_path / "bbox"
    if render_hand_bbox_sequence(rgb_paths, bbox_data, bbox_dir, demo_id):
        generate_video_from_images(
            demo_id, str(bbox_dir), str(output_path / "bbox.mp4"), fps=fps
        )
    render_subdiv_tactile_sequence(
        pressure_seq=pred_masked,
        output_dir=str(output_path / "pred_touch"),
        demo_id=demo_id,
        target_size=tactile_size,
        temporal_alpha=args.temporal_alpha,
        display_floor=args.display_floor,
        display_gamma=args.display_gamma,
        hand_side=hand_side,
        canonical_view=args.canonical_view,
        valid_frames=pred_valid_frames,
        panel_label=f"{hand_side.upper()} PRED",
        invalid_frame_notices=pred_invalid_notices,
    )
    render_subdiv_tactile_sequence(
        pressure_seq=gt_masked,
        output_dir=str(output_path / "gt_touch"),
        demo_id=demo_id,
        target_size=tactile_size,
        temporal_alpha=args.temporal_alpha,
        display_floor=args.display_floor,
        display_gamma=args.display_gamma,
        hand_side=hand_side,
        canonical_view=args.canonical_view,
        valid_frames=gt_valid_frames,
        panel_label=f"{hand_side.upper()} GT",
        invalid_frame_notices=gt_invalid_notices,
    )
    pred_video = output_path / "pred_touch.mp4"
    gt_video = output_path / "gt_touch.mp4"
    generate_video_from_images(
        demo_id, str(output_path / "pred_touch"), str(pred_video), fps=fps
    )
    generate_video_from_images(
        demo_id, str(output_path / "gt_touch"), str(gt_video), fps=fps
    )
    concat_videos(
        videos=[output_path / "rgb.mp4", pred_video, gt_video],
        output=output_path / "combined.mp4",
        layout="horizontal",
        scale_height=tactile_size[1],
    )
    print(f">>> Dataset RGB | PRED | GT comparison saved to: {output_path / 'combined.mp4'}")
    return output_path


def _run_dataset_sequence_demo(args, tactile_size):
    requested_hands = (args.hand,) if args.hand != "auto" else ("left", "right")
    available = []
    errors = []
    for hand_side in requested_hands:
        try:
            records, demo_id = _load_dataset_sequence(
                args.dataset_sequence,
                hand_side=hand_side,
                stride=args.dataset_stride,
                max_frames=args.dataset_max_frames,
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        available.append((hand_side, records, demo_id))

    if not available:
        requested = ", ".join(requested_hands)
        details = "\n  - ".join(errors)
        raise ValueError(
            f"No metadata-bound pressure hand was available for requested side(s): {requested}."
            + (f"\n  - {details}" if details else "")
        )

    if args.hand == "auto":
        print(
            ">>> Auto hand association from stored dataset metadata: "
            + ", ".join(hand_side for hand_side, _records, _demo_id in available)
        )
    model_bundle = None
    if not args.prepare_frames_only and not args.skip_inference:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        model, _detector, _cpm, model_cfg = load_models(
            args.checkpoint,
            device,
            dino_weights=args.dino_weights,
            bbox_rescale_factor=args.bbox_rescale_factor,
            load_detectors=False,
        )
        model_bundle = (model, model_cfg, device)
    rendered = {}
    for hand_side, records, demo_id in available:
        output_path = _run_dataset_sequence_hand_demo(
            args,
            tactile_size,
            records,
            demo_id,
            hand_side,
            model_bundle=model_bundle,
        )
        if output_path is not None:
            rendered[hand_side] = (output_path, records)

    is_touchanything = bool(available and available[0][1][0]["dataset"] == "TouchAnything")
    if args.hand == "auto" and is_touchanything and set(rendered) == {"left", "right"}:
        left_path, left_records = rendered["left"]
        right_path, right_records = rendered["right"]
        left_indices = [record["frame_idx"] for record in left_records]
        right_indices = [record["frame_idx"] for record in right_records]
        if left_indices != right_indices:
            raise ValueError(
                "TouchAnything left/right timelines differ after loading; refusing to "
                "create a visually misaligned two-hand video"
            )
        base_demo_id = left_path.name.removesuffix("__left")
        combined_dir = left_path.parent / f"{base_demo_id}__both_hands"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / "combined.mp4"
        concat_videos(
            videos=[left_path / "combined.mp4", right_path / "combined.mp4"],
            output=combined_path,
            layout="vertical",
            preset="fast",
        )
        with (combined_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "layout": ["LEFT: RGB | PRED | GT", "RIGHT: RGB | PRED | GT"],
                    "frame_indices": left_indices,
                    "left_output": str(left_path),
                    "right_output": str(right_path),
                    "combined_output": str(combined_path),
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        print(f">>> TouchAnything two-hand comparison saved to: {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render tactile predictions from a video or compare prediction/GT on an extracted dataset sequence."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained tactile checkpoint path")
    parser.add_argument("--video_path", type=str, default=None, help="Input video path")
    parser.add_argument(
        "--dataset_sequence",
        type=str,
        default=None,
        help=(
            "Extracted dataset frame directory containing meta.json. All sibling frames with the "
            "same OpenTouch scene/demo or TouchAnything scene/task/clip are compared as RGB | PRED | GT."
        ),
    )
    parser.add_argument(
        "--dataset_fps",
        type=float,
        default=30.0,
        help="Output FPS for dataset-sequence comparison videos.",
    )
    parser.add_argument(
        "--dataset_stride",
        type=int,
        default=1,
        help="Keep every Nth valid dataset frame after sorting by frame_idx.",
    )
    parser.add_argument(
        "--dataset_max_frames",
        type=int,
        default=None,
        help="Optional maximum number of selected dataset frames, useful for a quick preview.",
    )
    parser.add_argument("--out_dir", type=str, default="./demo_output", help="Output directory")
    parser.add_argument("--gpu", type=str, default="4", help="GPU index")
    parser.add_argument(
        "--hand",
        type=str,
        choices=["auto", "left", "right"],
        default="auto",
        help=(
            "Hand side. Dataset mode defaults to metadata-bound auto association; "
            "ordinary videos require left or right."
        ),
    )
    parser.add_argument(
        "--dino_weights",
        type=str,
        default=None,
        help="Optional local DINOv3 weight override. Compact checkpoint metadata is used by default.",
    )
    parser.add_argument(
        "--bbox_rescale_factor",
        type=float,
        default=None,
        help="Optional crop-scale override. Checkpoint metadata is used by default.",
    )
    parser.add_argument(
        "--display_floor",
        type=float,
        default=0.05,
        help="Visualization-only cutoff: predicted pressure below this value is rendered as zero.",
    )
    parser.add_argument(
        "--display_gamma",
        type=float,
        default=0.65,
        help="Legacy compatibility argument; ignored by the OpenTouch raw-pressure color mapping.",
    )
    parser.add_argument(
        "--temporal_alpha",
        type=float,
        default=0.4,
        help="Current-frame weight of the causal tactile display EMA in [0, 1].",
    )
    parser.add_argument(
        "--canonical_view",
        choices=["palm", "legacy"],
        default="palm",
        help="Canonical mesh view. 'palm' automatically selects the side exposing the trained palm faces.",
    )
    parser.add_argument(
        "--tactile_render_size",
        type=str,
        default="720x1280",
        help="Fixed tactile panel size as WIDTHxHEIGHT. This does not change RGB inference resolution.",
    )
    parser.add_argument(
        "--video_rotation",
        choices=["auto", "0", "90", "180", "270"],
        default="auto",
        help="Display rotation applied to source RGB frames before inference. 'auto' reads FFmpeg rotation metadata.",
    )
    parser.add_argument(
        "--combined_layout",
        choices=["auto", "horizontal", "vertical"],
        default="horizontal",
        help="Panel stacking direction. 'auto' is retained as an alias for the default side-by-side layout.",
    )
    parser.add_argument(
        "--rerender_tactile_dir",
        type=str,
        default=None,
        help="Re-render saved pred_tactile_palm_masked.npy in this demo directory without detection or model inference.",
    )
    parser.add_argument(
        "--skip_frame_extraction",
        action="store_true",
        help="Reuse existing out_dir/VIDEO_STEM/rgb frames instead of decoding the source video again.",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Reuse saved tactile predictions and hand boxes instead of loading detection/tactile models.",
    )
    parser.add_argument(
        "--prepare_frames_only",
        action="store_true",
        help="Extract orientation-correct RGB frames and rgb.mp4, then stop before loading any model.",
    )
    parser.add_argument(
        "--bbox_source",
        choices=["vitpose", "sam3"],
        default="vitpose",
        help="Generate per-frame hand boxes with ViTPose or consume an offline SAM3 bboxes.jsonl.",
    )
    parser.add_argument(
        "--sam3_bbox_jsonl",
        type=str,
        default=None,
        help="SAM3 track_video.py bboxes.jsonl; required when --bbox_source sam3.",
    )
    parser.add_argument(
        "--sam3_track_id",
        type=int,
        default=None,
        help="Anonymous SAM3 track to use. May be omitted only when JSONL contains one accepted track.",
    )
    parser.add_argument(
        "--missing_bbox_policy",
        choices=["auto", "zero", "hold"],
        default="auto",
        help="Tactile behavior on missing-box frames: auto uses zero for SAM3 and hold for ViTPose.",
    )
    parser.add_argument(
        "--render_platform",
        choices=["software", "egl", "osmesa", "auto"],
        default="software",
        help=(
            "Rendering backend selected before importing pyrender. Canonical tactile rendering uses "
            "the software rasterizer and needs no OpenGL; use osmesa for legacy headless pyrender paths."
        ),
    )
    args = parser.parse_args()

    if args.dataset_stride < 1:
        parser.error("--dataset_stride must be at least 1")
    if args.dataset_max_frames is not None and args.dataset_max_frames < 1:
        parser.error("--dataset_max_frames must be positive")
    if not np.isfinite(args.dataset_fps) or args.dataset_fps <= 0:
        parser.error("--dataset_fps must be finite and positive")
    if args.video_path and args.dataset_sequence:
        parser.error("Use either --video_path or --dataset_sequence, not both")

    print(
        f">>> Demo entry: {Path(__file__).resolve()} "
        f"(renderer={DEMO_RENDERER_VERSION}, platform={args.render_platform})"
    )
    tactile_size = _parse_render_size(args.tactile_render_size)

    if args.rerender_tactile_dir:
        if args.hand == "auto":
            parser.error("--rerender_tactile_dir requires --hand left or --hand right")
        rerender_saved_tactile_output(
            output_dir=args.rerender_tactile_dir,
            hand_side=args.hand,
            temporal_alpha=args.temporal_alpha,
            display_floor=args.display_floor,
            display_gamma=args.display_gamma,
            canonical_view=args.canonical_view,
            tactile_size=tactile_size,
            combined_layout=args.combined_layout,
        )
        return
    if not args.video_path and not args.dataset_sequence:
        parser.error(
            "--video_path or --dataset_sequence is required unless --rerender_tactile_dir is used"
        )
    if args.prepare_frames_only and args.skip_inference:
        parser.error("--prepare_frames_only and --skip_inference are separate terminal modes")
    if not args.prepare_frames_only and not args.skip_inference and not args.checkpoint:
        parser.error("--checkpoint is required unless --skip_inference or --rerender_tactile_dir is used")
    if (
        not args.prepare_frames_only
        and not args.skip_inference
        and args.bbox_source == "sam3"
        and not args.sam3_bbox_jsonl
    ):
        parser.error("--sam3_bbox_jsonl is required when --bbox_source sam3")
    if args.dataset_sequence:
        _run_dataset_sequence_demo(args, tactile_size)
        return

    if not args.prepare_frames_only and args.hand == "auto":
        parser.error("Ordinary video inference requires --hand left or --hand right")

    video_path = Path(args.video_path)
    demo_id = video_path.stem
    hand_side = args.hand
    is_right_hand = 1 if hand_side == "right" else 0
    video_rotation = _resolve_video_rotation(args.video_rotation, video_path)

    # Ensure output folders
    out_path = Path(args.out_dir) / demo_id
    rgb_dir = out_path / "rgb"
    bbox_dir = out_path / "bbox"
    pred_touch_dir = out_path / "pred_touch"

    action = "Reusing" if args.skip_frame_extraction else "Extracting"
    print(f"\n>>> [1/5] {action} RGB frames for {video_path}...")
    frame_paths, fps, target_size = _prepare_rgb_frames(
        video_path=video_path,
        rgb_dir=rgb_dir,
        demo_id=demo_id,
        video_rotation=video_rotation,
        reuse_existing=args.skip_frame_extraction,
    )
    num_frames = len(frame_paths)
    bbox_path = out_path / "pred_hand_bboxes.npz"

    if args.prepare_frames_only:
        rgb_video = out_path / "rgb.mp4"
        generate_video_from_images(demo_id, str(rgb_dir), str(rgb_video), fps=fps)
        print(f">>> Frame preparation complete: {rgb_dir}")
        print(f">>> RGB preview saved to: {rgb_video}")
        return

    if not args.skip_inference:
        if torch.cuda.is_available():
            # CUDA_VISIBLE_DEVICES maps the selected physical GPU to local index 0.
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

    if args.skip_inference:
        print("\n>>> [2/5] Reusing saved tactile inference and hand boxes...")
        pressure_path = out_path / "pred_tactile_palm_masked.npy"
        if not pressure_path.is_file():
            raise FileNotFoundError(
                f"--skip_inference requires saved masked predictions: {pressure_path}"
            )
        pred_tactile_seq = np.asarray(np.load(pressure_path), dtype=np.float32)
        if pred_tactile_seq.ndim != 2 or len(pred_tactile_seq) != num_frames:
            raise ValueError(
                f"Saved tactile frame count mismatch: {pred_tactile_seq.shape} vs {num_frames} RGB frames"
            )
        bbox_data = _load_hand_bbox_data(bbox_path, expected_frames=num_frames)
    else:
        print("\n>>> [2/5] Loading models & running tactile inference on video frames...")
        model, detector, cpm, model_cfg = load_models(
            args.checkpoint,
            device,
            dino_weights=args.dino_weights,
            bbox_rescale_factor=args.bbox_rescale_factor,
            load_detectors=args.bbox_source == "vitpose",
        )
        pred_tactile_list = []
        if args.bbox_source == "sam3":
            bbox_data = _load_sam3_bbox_jsonl(
                args.sam3_bbox_jsonl,
                expected_frames=num_frames,
                rescale_factor=model.bbox_rescale_factor,
                track_id=args.sam3_track_id,
            )
            tight_bboxes = bbox_data["tight"].copy()
            crop_bboxes = bbox_data["crop"].copy()
        else:
            bbox_data = None
            tight_bboxes = np.full((num_frames, 4), np.nan, dtype=np.float32)
            crop_bboxes = np.full((num_frames, 4), np.nan, dtype=np.float32)

        for frame_index, frame_path in enumerate(tqdm(frame_paths, desc="Running Inference")):
            img_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                pred_tactile_list.append(None)
                continue
            if args.bbox_source == "sam3":
                if not bbox_data["detected"][frame_index]:
                    pred_tactile_list.append(None)
                    continue
                boxes_arr = tight_bboxes[frame_index : frame_index + 1]
                right_arr = np.asarray([is_right_hand], dtype=np.int64)
            else:
                img_rgb = img_bgr[:, :, ::-1]
                try:
                    det_out = detector(img_bgr)
                    det_instances = det_out["instances"]
                    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
                    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
                except Exception:
                    pred_tactile_list.append(None)
                    continue
                if len(pred_bboxes) == 0:
                    pred_tactile_list.append(None)
                    continue
                try:
                    vitposes_out = cpm.predict_pose(
                        img_rgb,
                        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
                    )
                except Exception:
                    pred_tactile_list.append(None)
                    continue

                bboxes = []
                is_right = []
                for vitposes in vitposes_out:
                    left_hand_keyp = vitposes["keypoints"][-42:-21]
                    right_hand_keyp = vitposes["keypoints"][-21:]
                    if not is_right_hand:
                        valid = left_hand_keyp[:, 2] > 0.5
                        if sum(valid) > 3:
                            bboxes.append(
                                [
                                    left_hand_keyp[valid, 0].min(),
                                    left_hand_keyp[valid, 1].min(),
                                    left_hand_keyp[valid, 0].max(),
                                    left_hand_keyp[valid, 1].max(),
                                ]
                            )
                            is_right.append(0)
                    else:
                        valid = right_hand_keyp[:, 2] > 0.5
                        if sum(valid) > 3:
                            bboxes.append(
                                [
                                    right_hand_keyp[valid, 0].min(),
                                    right_hand_keyp[valid, 1].min(),
                                    right_hand_keyp[valid, 0].max(),
                                    right_hand_keyp[valid, 1].max(),
                                ]
                            )
                            is_right.append(1)
                if not bboxes:
                    pred_tactile_list.append(None)
                    continue
                boxes_arr = np.asarray(bboxes, dtype=np.float32)
                right_arr = np.asarray(is_right, dtype=np.int64)
            dataset_batch = ViTDetDataset(
                model_cfg,
                img_bgr,
                boxes_arr,
                right_arr,
                rescale_factor=model.bbox_rescale_factor,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset_batch,
                batch_size=len(boxes_arr),
                shuffle=False,
                num_workers=0,
            )
            frame_tactile = None
            selected_index = None
            for batch in dataloader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model.forward_step(batch, train=False)
                pred_tactile = out["pred_tactile"].detach().cpu().numpy()
                for candidate_index in range(pred_tactile.shape[0]):
                    candidate_is_right = int(batch["right"][candidate_index].cpu().numpy())
                    if (is_right_hand and candidate_is_right == 1) or (
                        not is_right_hand and candidate_is_right == 0
                    ):
                        frame_tactile = pred_tactile[candidate_index]
                        selected_index = candidate_index
                        break
                if frame_tactile is not None:
                    break
            if selected_index is not None:
                tight_bboxes[frame_index] = boxes_arr[selected_index]
                crop_bboxes[frame_index] = _model_crop_bbox(
                    boxes_arr[selected_index],
                    model.bbox_rescale_factor,
                )
            pred_tactile_list.append(frame_tactile)

        _save_hand_bbox_data(
            bbox_path,
            tight_bboxes=tight_bboxes,
            crop_bboxes=crop_bboxes,
            rescale_factor=model.bbox_rescale_factor,
            source=args.bbox_source,
            track_id=bbox_data["track_id"] if bbox_data is not None else -1,
        )
        bbox_data = _load_hand_bbox_data(bbox_path, expected_frames=num_frames)

        first_valid_index = next(
            (index for index, tactile in enumerate(pred_tactile_list) if tactile is not None),
            None,
        )
        missing_bbox_policy = args.missing_bbox_policy
        if missing_bbox_policy == "auto":
            missing_bbox_policy = "zero" if args.bbox_source == "sam3" else "hold"
        print(f">>> Missing-bbox tactile policy: {missing_bbox_policy}")
        if first_valid_index is None:
            print(">>> Warning: No hand detected in any frame of the video!")
            pred_tactile_seq = np.zeros((num_frames, model.tactile_dim), dtype=np.float32)
        elif missing_bbox_policy == "zero":
            zero_tactile = np.zeros(model.tactile_dim, dtype=np.float32)
            pred_tactile_seq = np.stack(
                [tactile if tactile is not None else zero_tactile for tactile in pred_tactile_list],
                axis=0,
            )
        else:
            for frame_index in range(first_valid_index):
                pred_tactile_list[frame_index] = pred_tactile_list[first_valid_index]
            current_tactile = pred_tactile_list[0]
            for frame_index in range(1, num_frames):
                if pred_tactile_list[frame_index] is None:
                    pred_tactile_list[frame_index] = current_tactile
                else:
                    current_tactile = pred_tactile_list[frame_index]
            pred_tactile_seq = np.stack(pred_tactile_list, axis=0)

        np.save(out_path / "pred_tactile_raw.npy", pred_tactile_seq)
        pred_tactile_seq = pred_tactile_seq * _load_subdiv_palm_mask(model.tactile_dim)
        np.save(out_path / "pred_tactile_palm_masked.npy", pred_tactile_seq)

    print("\n>>> [3/5] Rendering hand-box audit video...")
    bbox_video = out_path / "bbox.mp4"
    if render_hand_bbox_sequence(
        frame_paths=frame_paths,
        bbox_data=bbox_data,
        output_dir=bbox_dir,
        demo_id=demo_id,
    ):
        generate_video_from_images(
            demo_id,
            str(bbox_dir),
            str(bbox_video),
            fps=fps,
        )
    elif bbox_video.is_file():
        bbox_video.unlink()

    print("\n>>> [4/5] Rendering canonical subdiv tactile heatmap...")
    render_bbox_source = bbox_data.get("source", args.bbox_source) if bbox_data is not None else args.bbox_source
    render_missing_policy = args.missing_bbox_policy
    if render_missing_policy == "auto":
        render_missing_policy = "zero" if render_bbox_source == "sam3" else "hold"
    valid_frames = (
        bbox_data["detected"]
        if bbox_data is not None and render_missing_policy == "zero"
        else None
    )
    render_subdiv_tactile_sequence(
        pressure_seq=pred_tactile_seq,
        output_dir=str(pred_touch_dir),
        demo_id=demo_id,
        target_size=tactile_size,
        temporal_alpha=args.temporal_alpha,
        display_floor=args.display_floor,
        display_gamma=args.display_gamma,
        hand_side=hand_side,
        canonical_view=args.canonical_view,
        valid_frames=valid_frames,
    )

    print("\n>>> [5/5] Compiling RGB and canonical tactile visualization...")
    
    # Locate paths
    rgb_mp4 = out_path / "rgb.mp4"
    # Generate the RGB video, then compose with the fixed-size tactile panel.
    generate_video_from_images(demo_id, str(rgb_dir), str(rgb_mp4), fps=fps)
    _compose_demo_videos(
        output_path=out_path,
        demo_id=demo_id,
        fps=fps,
        rgb_size=target_size,
        tactile_size=tactile_size,
        combined_layout=args.combined_layout,
    )
    print(f"\n>>> Visualization compilation complete! Saved to: {out_path / 'combined.mp4'}")


if __name__ == "__main__":
    main()
