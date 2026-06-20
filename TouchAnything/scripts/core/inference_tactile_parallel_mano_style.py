"""
Tactile pressure-map inference and visualization, parallel optimized version

Optimizations:
1. Batched inference: process multiple clips per model call to reduce overhead
2. Multiprocessing: process multiple trajectory files in parallel
3. Data preloading: improve I/O efficiency

Layout (1440x630, aligned with visualize_trajectory.py):
  ┌───────────────────────────────────────────┐  270px
  │  Chest Cam  │  Left Wrist  │ Right Wrist  │
  ├─────────────┴──────┬───────┴──────────────┤  360px
  │   GT Pressure      │   Pred Pressure      │
  │   L hand | R hand  │   L hand | R hand    │
  └────────────────────┴──────────────────────┘
                                          1440px
"""
import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import multiprocessing as mp
# Force spawn mode to avoid CUDA fork deadlocks
mp.set_start_method('spawn', force=True)

import argparse
import numpy as np
import torch
import h5py
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans SC', 'WenQuanYi Micro Hei',
                                           'AR PL UMing CN', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.models import build_model
from src.utils import load_config_with_base
from src.data import get_transforms
from src.utils.vis_pressure import (fig_to_bgr, render_pressure_panel,
                                    PRESS_W, PRESS_H, pred_to_21)
from src.utils.metrics import compute_tactile_metrics

# ---------------------------------------------------------------------------
# Import visualization functions from visualize_hdf5.py
# ---------------------------------------------------------------------------
def put_label(img, text, pos=(10, 24), font_scale=0.65, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def put_text_cn(img_bgr, text, pos, font_size=14, color=(180, 180, 180)):
    """Render Unicode text on a BGR numpy image through PIL to bypass cv2 Unicode limitations."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Candidate CJK font paths in priority order
    _CN_FONT_CANDIDATES = [
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
    ]
    
    def _get_cn_font(size=14):
        for path in _CN_FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return ImageFont.load_default()
    
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=_get_cn_font(font_size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _load_bend_sensor_positions():
    """
    Load bend-sensor positions from the config file
    
    Returns:
        dict: {'left': set of (row, col), 'right': set of (row, col)}
    """
    import json
    from pathlib import Path
    
    # Find the config file
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'hand_joint_positions.json'
    
    if not config_path.exists():
        # Return empty sets if the config file is missing
        print(f"[WARNING] Bend-sensor config file does not exist: {config_path}")
        return {'left': set(), 'right': set()}
    
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    positions = {'left': set(), 'right': set()}
    
    # Read left- and right-hand joint positions
    for hand in ['left', 'right']:
        if hand in data:
            for joint in data[hand]:
                row = joint['row']
                col = joint['col']
                positions[hand].add((row, col))
    
    # Debug information, printed only once
    if not hasattr(_load_bend_sensor_positions, '_printed'):
        print(f"[INFO] Bend-sensor positions loaded: left hand{len(positions['left'])} points, right hand{len(positions['right'])} points")
        _load_bend_sensor_positions._printed = True
    
    return positions


def _create_bend_sensor_mask(size=21, hand='both'):
    """
    Create a bend-sensor mask that marks finger-joint bend-sensor positions
    
    Read exact joint positions directly from configs/hand_joint_positions.json
    
    Args:
        size: grid size, default 21
        hand: 'left', 'right', or 'both' (default)
    
    Returns:
        mask: (size, size) bool array where True marks bend sensors that should be excluded
    """
    mask = np.zeros((size, size), dtype=bool)
    
    # Load bend-sensor positions
    positions = _load_bend_sensor_positions()
    
    # Select positions to mark based on the hand argument
    if hand == 'both':
        # Merge left- and right-hand positions
        all_positions = positions['left'] | positions['right']
    elif hand in positions:
        all_positions = positions[hand]
    else:
        all_positions = set()
    
    # Mark bend-sensor positions
    for row, col in all_positions:
        if 0 <= row < size and 0 <= col < size:
            mask[row, col] = True
    
    return mask


def check_contact_strict(pressure_map, threshold=0.1, min_contact_ratio=0.05, exclude_bend_sensors=True):
    """
    Strict contact detection:
    1. Exclude finger bend-sensor regions, optional
    2. Pressure values exceed the threshold
    3. The ratio of above-threshold points over valid tactile sensors reaches the required percentage
    
    Args:
        pressure_map: (2, H, W) or (H, W) pressure map
                     For (2, H, W), dim 0 is the left hand and dim 1 is the right hand
        threshold: pressure threshold
        min_contact_ratio: minimum ratio of above-threshold points over the valid area, default 5%
        exclude_bend_sensors: whether to exclude bend-sensor regions, default True
    
    Returns:
        bool: whether contact is detected
    """
    if np.all(np.isnan(pressure_map)):
        return False
    
    # Get pressure-map size
    if pressure_map.ndim == 3:  # (2, H, W) - two hands
        size = pressure_map.shape[-1]
    else:  # (H, W) - single hand
        size = pressure_map.shape[-1]
    
    # Create bend-sensor masks
    if exclude_bend_sensors:
        if pressure_map.ndim == 3:  # (2, H, W) - two hands, apply left and right masks separately
            left_mask = _create_bend_sensor_mask(size, hand='left')
            right_mask = _create_bend_sensor_mask(size, hand='right')
            bend_mask = np.stack([left_mask, right_mask], axis=0)
        else:  # (H, W) - single hand, use the merged mask
            bend_mask = _create_bend_sensor_mask(size, hand='both')
    else:
        bend_mask = np.zeros_like(pressure_map, dtype=bool)
    
    # Compute valid tactile sensor regions, excluding NaNs and bend sensors
    valid_mask = ~np.isnan(pressure_map) & ~bend_mask
    num_valid_points = np.sum(valid_mask)
    
    if num_valid_points == 0:
        return False
    
    # Count above-threshold points only in valid tactile sensor regions
    above_threshold = (pressure_map > threshold) & valid_mask
    num_contact_points = np.sum(above_threshold)
    
    # Compute the ratio
    contact_ratio = num_contact_points / num_valid_points
    
    # Decision: require above-threshold points and the required ratio
    is_contact = contact_ratio >= min_contact_ratio
    
    return is_contact

# ── Layout constants, aligned with visualize_trajectory.py──
CAM_W,   CAM_H   = 480, 270
VIDEO_W = CAM_W * 3      # 1440
VIDEO_H = CAM_H + PRESS_H # 630


def _read_pose_source_from_hdf5(f: h5py.File, source: str):
    """Read full trajectory pose arrays from selected source."""
    if source == 'rokoko':
        left_xyz = f['hands/left_joint_xyz'][:]
        right_xyz = f['hands/right_joint_xyz'][:]
        left_valid = np.ones((left_xyz.shape[0],), dtype=np.uint8)
        right_valid = np.ones((right_xyz.shape[0],), dtype=np.uint8)
        return left_xyz, right_xyz, left_valid, right_valid
    
    elif source == 'wilor':
        wilor_keys = [
            'hands/wilor_left_joint_xyz',
            'hands/wilor_right_joint_xyz',
            'hands/wilor_left_valid',
            'hands/wilor_right_valid',
        ]
        missing = [k for k in wilor_keys if k not in f]
        if missing:
            raise KeyError(f"WiLoR datasets missing: {missing}")
        
        left_xyz = f['hands/wilor_left_joint_xyz'][:]
        right_xyz = f['hands/wilor_right_joint_xyz'][:]
        left_valid = f['hands/wilor_left_valid'][:].astype(np.uint8)
        right_valid = f['hands/wilor_right_valid'][:].astype(np.uint8)
        return left_xyz, right_xyz, left_valid, right_valid
    
    elif source == 'hamer':
        hamer_keys = [
            'hands/hamer_left_joint_xyz',
            'hands/hamer_right_joint_xyz',
            'hands/hamer_left_valid',
            'hands/hamer_right_valid',
        ]
        missing = [k for k in hamer_keys if k not in f]
        if missing:
            raise KeyError(f"HaMeR datasets missing: {missing}")
        
        left_xyz = f['hands/hamer_left_joint_xyz'][:]
        right_xyz = f['hands/hamer_right_joint_xyz'][:]
        left_valid = f['hands/hamer_left_valid'][:].astype(np.uint8)
        right_valid = f['hands/hamer_right_valid'][:].astype(np.uint8)
        return left_xyz, right_xyz, left_valid, right_valid
    
    else:
        raise ValueError(f"Unknown pose source: {source}. Must be 'rokoko', 'wilor', or 'hamer'")


def _load_pose_arrays_with_fallback(f: h5py.File, pose_source: str, fallback_pose_source: str = None):
    """Load selected pose source and optionally fill invalid frames from fallback source."""
    try:
        left_xyz, right_xyz, left_valid, right_valid = _read_pose_source_from_hdf5(f, pose_source)
    except KeyError:
        if fallback_pose_source is None or fallback_pose_source == pose_source:
            raise
        return _read_pose_source_from_hdf5(f, fallback_pose_source)

    if fallback_pose_source is not None and fallback_pose_source != pose_source:
        try:
            fb_left, fb_right, fb_left_valid, fb_right_valid = _read_pose_source_from_hdf5(
                f, fallback_pose_source
            )
            miss_l = left_valid == 0
            miss_r = right_valid == 0
            if np.any(miss_l):
                left_xyz[miss_l] = fb_left[miss_l]
                left_valid[miss_l] = fb_left_valid[miss_l]
            if np.any(miss_r):
                right_xyz[miss_r] = fb_right[miss_r]
                right_valid[miss_r] = fb_right_valid[miss_r]
        except KeyError:
            pass

    return left_xyz, right_xyz, left_valid, right_valid


# ──────────────────────────────────────────────
# Run batched inference for one HDF5 trajectory, optimized version
# ──────────────────────────────────────────────
@torch.no_grad()
def infer_trajectory_batched(hdf5_path: str, model, device, cfg,
                             clip_length: int = 8, frame_interval: int = 2,
                             batch_size: int = 16, view_config: str = 'all',
                             pose_source: str = 'wilor',
                             fallback_pose_source: str = None):
    """
    Batched inference optimization: process multiple clips per model call
    
    Args:
        batch_size: number of clips per batch, default 16
        view_config: view configuration, one of 'ego', 'ego+left', 'ego+right', or 'all' (default)
        pose_source: pose source, 'rokoko' or 'hamer'
        fallback_pose_source: fallback pose source, optional
    
    Returns:
        ego_frames, left_frames, right_frames, pred_maps, gt_raw, vmax, pose_frame_valid
    """
    transform = get_transforms(cfg, is_training=False)
    tactile_size = cfg['data'].get('tactile_size', 16)
    image_size   = tuple(cfg['data'].get('image_size', [224, 224]))

    with h5py.File(hdf5_path, 'r') as f:
        num_frames         = int(f['metadata'].attrs['num_frames'])
        chest_imgs         = f['images/chest_color'][:]
        left_xyz, right_xyz, left_valid, right_valid = _load_pose_arrays_with_fallback(
            f, pose_source=pose_source, fallback_pose_source=fallback_pose_source
        )
        left_grids         = f['pressure/left_pressure_grid'][:]
        right_grids        = f['pressure/right_pressure_grid'][:]
        left_imgs          = f['images/left_color'][:]
        right_imgs         = f['images/right_color'][:]
        timestamps         = f['timestamps'][:] if 'timestamps' in f else None
        already_normalized = bool(f['pressure'].attrs.get('normalized', False))

    # Pose-valid frame: both hands valid after fallback
    pose_frame_valid = (left_valid > 0) & (right_valid > 0)  # (T,)

    T = num_frames
    span = (clip_length - 1) * frame_interval + 1
    S = tactile_size
    pred_maps_sum = np.zeros((T, 2, S, S), dtype=np.float32)  # accumulated predictions
    pred_counts = np.zeros(T, dtype=np.int32)  # number of predictions per frame

    def _prep_clip_images(arr, start):
        """(T_clip, H, W, 3) uint8 → tensor (1, T, 3, H, W)"""
        idxs = [start + i * frame_interval for i in range(clip_length)]
        frames = np.stack([arr[i] for i in idxs], axis=0)
        frames_rgb = frames[..., :3]
        resized = np.stack([
            cv2.resize(f, (image_size[1], image_size[0])) for f in frames_rgb
        ])
        t = transform(resized)
        return t.unsqueeze(0)  # (1, T, 3, H, W)

    # ---- Batched inference ----
    clip_starts = list(range(0, T - span + 1))
    num_batches = (len(clip_starts) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc='  Batched inference', leave=False):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(clip_starts))
        batch_starts = clip_starts[start_idx:end_idx]
        
        # Prepare batched data
        ego_batch = []
        left_batch = []
        right_batch = []
        poses_batch = []
        last_frame_idxs = []
        
        for start in batch_starts:
            idxs = list(range(start, start + span, frame_interval))[:clip_length]
            
            ego_batch.append(_prep_clip_images(chest_imgs, start))
            left_batch.append(_prep_clip_images(left_imgs, start))
            right_batch.append(_prep_clip_images(right_imgs, start))
            
            lxyz = left_xyz[idxs].copy()
            rxyz = right_xyz[idxs].copy()
            lvalid = left_valid[idxs]
            rvalid = right_valid[idxs]
            
            # Mark invalid poses with a special value BEFORE clipping
            # (consistent with training data processing in touchanything_dataset.py)
            INVALID_POSE_VALUE = -10.0
            for t in range(clip_length):
                if lvalid[t] == 0:
                    lxyz[t, :, :] = INVALID_POSE_VALUE
                if rvalid[t] == 0:
                    rxyz[t, :, :] = INVALID_POSE_VALUE
            
            poses_np = np.concatenate([lxyz, rxyz], axis=1).astype(np.float32)
            
            # Clip abnormal pose coordinates to avoid extreme values from failed predictions
            # (consistent with training data processing in touchanything_dataset.py)
            poses_np[:, :, 0] = np.clip(poses_np[:, :, 0], -10.0, 10.0)   # x: horizontal offset
            poses_np[:, :, 1] = np.clip(poses_np[:, :, 1], -10.0, 10.0)   # y: vertical offset
            poses_np[:, :, 2] = np.clip(poses_np[:, :, 2], -10.0, 100.0)  # z: depth value
            
            poses_batch.append(torch.from_numpy(poses_np).unsqueeze(0))
            
            last_frame_idxs.append(idxs[-1])
        
        # Merge into batched tensors
        ego_t = torch.cat(ego_batch, dim=0).to(device)      # (B, T, 3, H, W)
        left_t = torch.cat(left_batch, dim=0).to(device)
        right_t = torch.cat(right_batch, dim=0).to(device)
        poses_t = torch.cat(poses_batch, dim=0).to(device)  # (B, T, 42, 3) - 21 joints x 2 hands
        
        # Build the views dictionary from view_config
        views = {'ego': ego_t}
        if view_config in ['ego+left', 'all']:
            views['wrist_left'] = left_t
        if view_config in ['ego+right', 'all']:
            views['wrist_right'] = right_t
        
        # Batched inference
        # Note: use the frames argument for ego-only input, otherwise use views
        # with torch.amp.autocast('cuda'):
        #     if view_config == 'ego':
        #         # Single view: use the frames argument
        #         out = model(frames=ego_t, poses=poses_t)
        #     else:
        #         # Multi-view: use only the views argument and omit frames
        #         out = model(poses=poses_t, views=views)
        with torch.amp.autocast('cuda'):
            if getattr(model, 'multi_view_enabled', False):
                out = model(poses=poses_t, views=views)
            else:
                # Use frames for single-view models without multi_view_encoder
                out = model(frames=ego_t, poses=poses_t)
        pred = out['tactile'].float().cpu().numpy()  # (B, T, 2, S, S)
        
        # Write all frames from each clip, accumulating sums for later averaging
        for i, start in enumerate(batch_starts):
            for t in range(clip_length):
                frame_idx = start + t * frame_interval
                if frame_idx < T:
                    pred_maps_sum[frame_idx] += pred[i, t]
                    pred_counts[frame_idx] += 1

    # ---- Average repeated predictions for overlapping frames----
    pred_maps_all = np.full((T, 2, S, S), np.nan, dtype=np.float32)
    for t in range(T):
        if pred_counts[t] > 0:
            pred_maps_all[t] = pred_maps_sum[t] / pred_counts[t]
    
    # ---- GT: normalize consistently to [0, 1] while preserving NaNs ----
    if already_normalized:
        left_grids_display  = left_grids
        right_grids_display = right_grids
    else:
        raw_vmax = max(float(np.nanmax(left_grids)), float(np.nanmax(right_grids)), 1.0)
        left_grids_display  = np.where(np.isnan(left_grids),  np.nan, np.clip(left_grids  / raw_vmax, 0.0, 1.0))
        right_grids_display = np.where(np.isnan(right_grids), np.nan, np.clip(right_grids / raw_vmax, 0.0, 1.0))
    vmax = 1.0  # visualization range fixed to [0, 1]
    gt_raw = np.stack([left_grids_display, right_grids_display], axis=1)
    
    # Copy no-sensor GT positions (NaN) to predictions to prevent invalid outputs from affecting decisions
    pred_maps_all[np.isnan(gt_raw)] = np.nan
    
    ego_display   = chest_imgs[:, :, :, :3]
    left_display  = left_imgs[:, :, :, :3]
    right_display = right_imgs[:, :, :, :3]

    return ego_display, left_display, right_display, pred_maps_all, gt_raw, vmax, pose_frame_valid, timestamps


def make_comparison_video(ego_frames, left_frames, right_frames,
                          pred_maps, gt_raw, timestamps, task_name, traj_name,
                          output_path: str, fps: int = 10,
                          vmax: float = 1.0, pressure_style: str = '2d',
                          use_mano_3d: bool = False):
    """Compose comparison video supporting both 2D heatmap and MANO 3D styles
    
    Args:
        pressure_style: '2d' or 'mano_3d' - Pressure display style
        use_mano_3d: whether to use MANO 3D rendering, requiring pregenerated images
    """
    T = ego_frames.shape[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Use ffmpeg instead of cv2.VideoWriter for better codec support
    import subprocess
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{VIDEO_W}x{VIDEO_H}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264', '-preset', 'medium',
        '-crf', '23', '-pix_fmt', 'yuv420p', str(output_path)
    ]
    # Avoid deadlocks by redirecting stderr to DEVNULL so the pipe buffer does not fill up
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # NaN mask: from GT frame 0 (fixed across all frames)
    nan_mask_l = np.isnan(gt_raw[0, 0])
    nan_mask_r = np.isnan(gt_raw[0, 1])

    # MANO 3D variables
    mano_left_dir = None
    mano_right_dir = None
    tactile_left_dir = None
    tactile_right_dir = None
    
    if use_mano_3d and pressure_style == 'mano_3d':
        print("\nGenerating MANO 3D visualization with OpenTouch...")
        import tempfile
        import shutil
        
        temp_dir = Path(tempfile.mkdtemp(prefix='mano_vis_'))
        
        try:
            # Import local visualization functions
            script_dir = Path(__file__).resolve().parent
            sys.path.insert(0, str(script_dir))
            
            # Configure headless rendering
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            
            # Create a temporary OpenTouch-format HDF5 file
            temp_opentouch_hdf5 = temp_dir / 'opentouch_format.hdf5'
            
            print("  Converting to OpenTouch format...")
            # Create temporary data for MANO rendering
            with h5py.File(temp_opentouch_hdf5, 'w') as dst:
                data_group = dst.create_group('data')
                demo_group = data_group.create_group('demo_0')
                
                # Copy pressure data for tactile MANO
                left_pressure = np.nan_to_num(gt_raw[:, 0], nan=0.0)
                right_pressure = np.nan_to_num(gt_raw[:, 1], nan=0.0)
                demo_group.create_dataset('left_pressure', data=left_pressure)
                demo_group.create_dataset('right_pressure', data=right_pressure)
            
            # Call export_tactile_mano_ta
            from load_data import export_tactile_mano_ta
            
            print("  Generating left- and right-hand tactile MANO renders...")
            tactile_output_dir = temp_dir / 'tactile_output'
            
            # Get mapping file paths
            project_root = script_dir.parent
            mapping_left = project_root / 'ref' / 'opentouch' / 'preprocess' / 'ta_to_mano_mapping_left_visual.json'
            mapping_right = project_root / 'ref' / 'opentouch' / 'preprocess' / 'ta_to_mano_mapping_right_visual.json'
            
            # Generate left-hand tactile MANO
            if mapping_left.exists():
                export_tactile_mano_ta(
                    file_path=str(temp_opentouch_hdf5),
                    demo_id='demo_0',
                    output_dir=str(tactile_output_dir),
                    dataset_names=('left_pressure',),
                    target_size=(640, 640),
                    max_value=1.0,
                    temporal_alpha=0.4,
                    layout_json_path=str(mapping_left)
                )
                tactile_left_dir = tactile_output_dir / 'left_pressure'
            
            # Generate right-hand tactile MANO
            if mapping_right.exists():
                export_tactile_mano_ta(
                    file_path=str(temp_opentouch_hdf5),
                    demo_id='demo_0',
                    output_dir=str(tactile_output_dir),
                    dataset_names=('right_pressure',),
                    target_size=(640, 640),
                    max_value=1.0,
                    temporal_alpha=0.4,
                    layout_json_path=str(mapping_right)
                )
                tactile_right_dir = tactile_output_dir / 'right_pressure'
            
            if tactile_left_dir and tactile_right_dir:
                print(f"  ✓ Tactile MANO images generated")
            else:
                print(f"  ⚠️  Tactile MANO generation failed; using 2D heatmaps")
                pressure_style = '2d'
                
        except Exception as e:
            print(f"  ⚠️  MANO 3D generation failed: {e}")
            print(f"  Falling back to 2D heatmap mode")
            pressure_style = '2d'
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
                temp_dir = None

    # Compute per-frame contact status for visualization
    contact_threshold = 0.1  # contact detection threshold
    min_contact_ratio = 0.05  # minimum ratio of above-threshold points over the valid area (3%)
    gt_contact_status = []
    pred_contact_status = []
    
    for t in range(T):
        # GT contact status: use strict detection (threshold + ratio)
        gt_has_contact = check_contact_strict(gt_raw[t], contact_threshold, min_contact_ratio)
        gt_contact_status.append(gt_has_contact)
        
        # Predicted contact status: use strict detection (threshold + ratio)
        pred_has_contact = check_contact_strict(pred_maps[t], contact_threshold, min_contact_ratio)
        pred_contact_status.append(pred_has_contact)
    
    for t in range(T):
        # ── Three camera images, RGB to BGR──────────
        def prep_cam(img_rgb, label):
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.resize(img_bgr, (CAM_W, CAM_H))
            put_label(img_bgr, label)
            return img_bgr

        cam_chest = prep_cam(ego_frames[t], 'Head')
        cam_left = prep_cam(left_frames[t],  'Left Wrist')
        cam_right = prep_cam(right_frames[t], 'Right Wrist')
        row_top = np.hstack([cam_chest, cam_left, cam_right])

        # ── Bottom two panels ──────────────────────────────────────────
        # Bottom-left: GT pressure map
        if pressure_style == 'mano_3d' and use_mano_3d:
            # Use MANO 3D rendering from pregenerated images
            gt_left_file = tactile_left_dir / f"demo_0_{t:05d}.png" if tactile_left_dir else None
            gt_right_file = tactile_right_dir / f"demo_0_{t:05d}.png" if tactile_right_dir else None
            gt_left_img = cv2.imread(str(gt_left_file)) if gt_left_file and gt_left_file.exists() else None
            gt_right_img = cv2.imread(str(gt_right_file)) if gt_right_file and gt_right_file.exists() else None
            gt_panel = render_mano_pressure_panel(gt_left_img, gt_right_img, title_prefix="Ground Truth")
        else:
            # Use 2D heatmaps
            gt_panel = render_pressure_panel(
                gt_raw[t, 0], gt_raw[t, 1],
                title='Ground Truth Pressure',
                w=PRESS_W, h=PRESS_H, vmax=vmax, global_max=vmax,
            )
        
        # Bottom-right: predicted pressure map
        if np.all(np.isnan(pred_maps[t])):
            pred_panel = np.zeros((PRESS_H, PRESS_W, 3), dtype=np.uint8)
            cv2.putText(pred_panel, 'No Prediction', (PRESS_W//2 - 80, PRESS_H//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        else:
            if pressure_style == 'mano_3d' and use_mano_3d:
                # Use MANO 3D rendering for predicted pressure maps
                # Need to generate MANO 3D images for predictions
                try:
                    # Create temporary prediction data for MANO rendering
                    temp_pred_hdf5 = temp_dir / 'pred_opentouch_format.hdf5'
                    with h5py.File(temp_pred_hdf5, 'w') as dst:
                        data_group = dst.create_group('data')
                        demo_group = data_group.create_group('demo_0')
                        
                        # Use prediction data for the current frame
                        pred_left_pressure = np.nan_to_num(pred_maps[t, 0], nan=0.0)
                        pred_right_pressure = np.nan_to_num(pred_maps[t, 1], nan=0.0)
                        
                        # Expand to a time sequence because MANO requires a time dimension
                        pred_left_seq = np.expand_dims(pred_left_pressure, axis=0)
                        pred_right_seq = np.expand_dims(pred_right_pressure, axis=0)
                        
                        demo_group.create_dataset('left_pressure', data=pred_left_seq)
                        demo_group.create_dataset('right_pressure', data=pred_right_seq)
                    
                    # Generate predicted MANO 3D images
                    pred_output_dir = temp_dir / 'pred_output'
                    
                    # left hand
                    if mapping_left.exists():
                        export_tactile_mano_ta(
                            file_path=str(temp_pred_hdf5),
                            demo_id='demo_0',
                            output_dir=str(pred_output_dir),
                            dataset_names=('left_pressure',),
                            target_size=(640, 640),
                            max_value=1.0,
                            temporal_alpha=0.4,
                            layout_json_path=str(mapping_left)
                        )
                    
                    # right hand
                    if mapping_right.exists():
                        export_tactile_mano_ta(
                            file_path=str(temp_pred_hdf5),
                            demo_id='demo_0',
                            output_dir=str(pred_output_dir),
                            dataset_names=('right_pressure',),
                            target_size=(640, 640),
                            max_value=1.0,
                            temporal_alpha=0.4,
                            layout_json_path=str(mapping_right)
                        )
                    
                    # Read generated MANO images
                    pred_left_dir = pred_output_dir / 'left_pressure'
                    pred_right_dir = pred_output_dir / 'right_pressure'
                    pred_left_file = pred_left_dir / "demo_0_00000.png" if pred_left_dir else None
                    pred_right_file = pred_right_dir / "demo_0_00000.png" if pred_right_dir else None
                    pred_left_img = cv2.imread(str(pred_left_file)) if pred_left_file and pred_left_file.exists() else None
                    pred_right_img = cv2.imread(str(pred_right_file)) if pred_right_file and pred_right_file.exists() else None
                    
                    pred_panel = render_mano_pressure_panel(pred_left_img, pred_right_img, title_prefix="Predicted")
                    
                except Exception as e:
                    # Fall back to 2D heatmaps if MANO 3D generation fails
                    pred_l21 = pred_to_21(pred_maps[t, 0], nan_mask_l)
                    pred_r21 = pred_to_21(pred_maps[t, 1], nan_mask_r)
                    pred_panel = render_pressure_panel(
                        pred_l21, pred_r21,
                        title='Predicted Pressure',
                        w=PRESS_W, h=PRESS_H, vmax=vmax, global_max=vmax,
                    )
            else:
                # Use 2D heatmaps
                pred_l21 = pred_to_21(pred_maps[t, 0], nan_mask_l)
                pred_r21 = pred_to_21(pred_maps[t, 1], nan_mask_r)
                pred_panel = render_pressure_panel(
                    pred_l21, pred_r21,
                    title='Predicted Pressure',
                    w=PRESS_W, h=PRESS_H, vmax=vmax, global_max=vmax,
                )

        row_bottom = np.hstack([gt_panel, pred_panel])
        
        # ── Merge frames ────────────────────────────────────────────────────
        frame_out = np.vstack([row_top, row_bottom])

        # ── Add contact-status indicator at the top right────────────────────────────
        # GT contact status
        gt_contact = gt_contact_status[t]
        pred_contact = pred_contact_status[t]
        
        # Compute current-frame max pressure and contact-decision details for debugging
        # Only compute the max over valid tactile sensor regions, excluding bend sensors
        # This keeps the display consistent with contact detection
        
        # Compute contact-decision details; GT and prediction should use the same valid mask
        # Create a shared bend-sensor mask
        size = gt_raw[t].shape[-1]
        if gt_raw[t].ndim == 3:  # (2, H, W)
            left_mask = _create_bend_sensor_mask(size, hand='left')
            right_mask = _create_bend_sensor_mask(size, hand='right')
            bend_mask = np.stack([left_mask, right_mask], axis=0)
        else:
            bend_mask = _create_bend_sensor_mask(size, hand='both')
        
        # Shared valid mask based on GT NaNs, since GT defines which positions have sensors
        # GT and prediction must use the same valid mask
        valid_mask = ~np.isnan(gt_raw[t]) & ~bend_mask
        num_valid = np.sum(valid_mask)
        
        # Compute max pressure over the valid area for display
        if num_valid > 0:
            gt_max_val = np.nanmax(gt_raw[t][valid_mask])
            pred_max_val = np.nanmax(pred_maps[t][valid_mask])
        else:
            gt_max_val = 0.0
            pred_max_val = 0.0
        
        # Count above-threshold points
        if num_valid > 0:
            # GT
            gt_above = (gt_raw[t] > contact_threshold) & valid_mask
            gt_contact_pts = np.sum(gt_above)
            gt_ratio = gt_contact_pts / num_valid
            
            # Prediction, using the same valid mask
            pred_above = (pred_maps[t] > contact_threshold) & valid_mask
            pred_contact_pts = np.sum(pred_above)
            pred_ratio = pred_contact_pts / num_valid
        else:
            gt_contact_pts = 0
            gt_ratio = 0.0
            pred_contact_pts = 0
            pred_ratio = 0.0
        
        gt_valid = num_valid
        pred_valid = num_valid  # force the same value
        
        # Determine correctness
        is_correct = (gt_contact == pred_contact)
        
        # Draw the contact-status box, with extra height for details
        status_x = VIDEO_W - 320
        status_y = 10
        box_width = 300
        box_height = 130  # increase height to show more details
        
        # Background box
        overlay = frame_out.copy()
        cv2.rectangle(overlay, (status_x, status_y), 
                     (status_x + box_width, status_y + box_height),
                     (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame_out, 0.3, 0, frame_out)
        
        # Border color: green for correct, red for incorrect
        border_color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.rectangle(frame_out, (status_x, status_y), 
                     (status_x + box_width, status_y + box_height),
                     border_color, 2)
        
        # Title
        cv2.putText(frame_out, "Contact Status", 
                   (status_x + 10, status_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # GT status
        gt_text = "GT:   " + ("CONTACT" if gt_contact else "NO CONTACT")
        gt_color = (0, 255, 0) if gt_contact else (128, 128, 128)
        cv2.putText(frame_out, gt_text, 
                   (status_x + 10, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, gt_color, 1, cv2.LINE_AA)
        
        # Predicted status
        pred_text = "Pred: " + ("CONTACT" if pred_contact else "NO CONTACT")
        pred_color = (0, 255, 0) if pred_contact else (128, 128, 128)
        cv2.putText(frame_out, pred_text, 
                   (status_x + 10, status_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, pred_color, 1, cv2.LINE_AA)
        
        # Show max pressure
        max_text = f"Max: GT={gt_max_val:.3f} Pred={pred_max_val:.3f}"
        cv2.putText(frame_out, max_text, 
                   (status_x + 10, status_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Show GT contact-decision details
        gt_detail = f"GT:  {gt_contact_pts}/{gt_valid} ({gt_ratio*100:.1f}%)"
        cv2.putText(frame_out, gt_detail, 
                   (status_x + 10, status_y + 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Show predicted contact-decision details
        pred_detail = f"Pred: {pred_contact_pts}/{pred_valid} ({pred_ratio*100:.1f}%)"
        cv2.putText(frame_out, pred_detail, 
                   (status_x + 10, status_y + 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Correctness marker, using circles instead of Unicode symbols
        marker_center = (status_x + box_width - 20, status_y + 45)
        if is_correct:
            # Solid green circle means correct
            cv2.circle(frame_out, marker_center, 12, (0, 255, 0), -1)
            cv2.circle(frame_out, marker_center, 12, (255, 255, 255), 2)
        else:
            # Red cross means incorrect
            cv2.line(frame_out, 
                    (marker_center[0] - 8, marker_center[1] - 8),
                    (marker_center[0] + 8, marker_center[1] + 8),
                    (0, 0, 255), 3)
            cv2.line(frame_out, 
                    (marker_center[0] + 8, marker_center[1] - 8),
                    (marker_center[0] - 8, marker_center[1] + 8),
                    (0, 0, 255), 3)

        # ── Bottom information row rendered with PIL for Unicode task names
        ts_sec = (int(timestamps[t]) - int(timestamps[0])) / 1000.0 if timestamps is not None else 0
        style_text = "MANO 3D" if pressure_style == 'mano_3d' else "2D Heatmap"
        info_line = (f"Frame {t+1:03d}/{T}   t={ts_sec:6.2f}s"
                     f"   [{task_name}/{traj_name}]   ({style_text})")
        frame_out = put_text_cn(
            frame_out, info_line,
            pos=(10, VIDEO_H - 22), font_size=14, color=(180, 180, 180))

        # Style watermark at the top right
        style_color = (0, 120, 200) if pressure_style == 'mano_3d' else (0, 200, 120)
        cv2.putText(frame_out, style_text,
                    (VIDEO_W - 120, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, style_color, 2, cv2.LINE_AA)
        
        # ── Add running contact-status accuracy at the top left────────────────────────────
        # Compute accuracy up to the current frame
        correct_count = sum(1 for i in range(t+1) if gt_contact_status[i] == pred_contact_status[i])
        current_accuracy = correct_count / (t + 1) * 100
        
        acc_text = f"Contact Acc: {current_accuracy:.1f}% ({correct_count}/{t+1})"
        acc_color = (0, 255, 0) if current_accuracy >= 95 else (0, 165, 255) if current_accuracy >= 80 else (0, 0, 255)
        
        # Translucent background
        acc_overlay = frame_out.copy()
        cv2.rectangle(acc_overlay, (5, 5), (300, 35), (40, 40, 40), -1)
        cv2.addWeighted(acc_overlay, 0.6, frame_out, 0.4, 0, frame_out)
        
        cv2.putText(frame_out, acc_text,
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, acc_color, 2, cv2.LINE_AA)

        # ── Write to the ffmpeg pipe ────────────────────────────────────────
        ffmpeg_proc.stdin.write(frame_out.tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    
    # Clean up the temporary directory
    if use_mano_3d and temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)


def render_mano_pressure_panel(mano_left_img, mano_right_img, w=PRESS_W, h=PRESS_H, title_prefix=""):
    """MANO 3D pressure panel: horizontally concatenate left- and right-hand MANO visualizations
    
    Args:
        title_prefix: title prefix, such as "Ground Truth" or "Predicted"
    """
    # Target size: 720x360, each hand 360x360
    target_hand_size = (360, 360)
    
    # Resize the left-hand image
    if mano_left_img is not None:
        left_resized = cv2.resize(mano_left_img, target_hand_size)
    else:
        left_resized = np.zeros((360, 360, 3), dtype=np.uint8)
    
    # Resize the right-hand image
    if mano_right_img is not None:
        right_resized = cv2.resize(mano_right_img, target_hand_size)
    else:
        right_resized = np.zeros((360, 360, 3), dtype=np.uint8)
    
    # Concatenate horizontally
    panel = np.hstack([left_resized, right_resized])
    
    # Add labels with the title prefix
    left_title = f'{title_prefix} Left Hand' if title_prefix else 'Left Hand'
    right_title = f'{title_prefix} Right Hand' if title_prefix else 'Right Hand'
    put_label(panel, left_title, pos=(10, 24))
    put_label(panel, right_title, pos=(370, 24))
    
    return panel


# ──────────────────────────────────────────────
# Worker-local model for spawn mode and Pool initializer
# Each worker loads the model once at startup and reuses it across trajectories
# ──────────────────────────────────────────────
_WORKER_MODEL = None
_WORKER_CFG = None
_WORKER_DEVICE = None

def _worker_init(checkpoint_path, config_path, gpu_ids):
    """
    Pool initializer: called once when each worker process starts.
    Use spawn mode for full CUDA compatibility.
    """
    global _WORKER_MODEL, _WORKER_CFG, _WORKER_DEVICE
    # Assign GPU by worker index
    worker_id = mp.current_process()._identity
    worker_idx = (worker_id[0] - 1) if worker_id else 0
    device_id = gpu_ids[worker_idx % len(gpu_ids)]
    _WORKER_DEVICE = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    print(f'[Worker {worker_idx}] initializing model on {_WORKER_DEVICE}')
    _WORKER_CFG = load_config_with_base(config_path)
    _WORKER_MODEL = build_model(_WORKER_CFG)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    _WORKER_MODEL.load_state_dict(state, strict=False)  # allow extra keys while loading
    _WORKER_MODEL.to(_WORKER_DEVICE).eval()

# ──────────────────────────────────────────────
# Single-trajectory processing function for multiprocessing
# ──────────────────────────────────────────────
def process_single_trajectory(args):
    """Single-trajectory processing function for multiprocessing"""
    hdf5_path, output_dir, fps, batch_size, views, pose_source, fallback_pose_source, pressure_style, use_mano_3d, skip_video, skip_hdf5 = args
    
    # Use the current worker model and device, loaded in _worker_init
    model = _WORKER_MODEL
    device = _WORKER_DEVICE
    cfg = _WORKER_CFG
    
    # Inference
    task_name = Path(hdf5_path).parent.name
    traj_name = Path(hdf5_path).stem
    
    ego_frames, left_frames, right_frames, pred_maps, gt_raw, vmax, pose_frame_valid, timestamps = infer_trajectory_batched(
        hdf5_path, model, device, cfg,
        clip_length=cfg['data'].get('clip_length', 8),
        frame_interval=cfg['data'].get('frame_interval', 2),  # default matches the training config
        batch_size=batch_size,
        view_config=views,
        pose_source=pose_source,
        fallback_pose_source=fallback_pose_source
    )
    
    # Compute two metric sets:
    # 1) all_frames: all frames with predictions
    # 2) pose_valid_frames: frames with predictions and valid poses
    valid_pred_frames = np.array([not np.all(np.isnan(pred_maps[t])) for t in range(len(pred_maps))])
    eval_all_mask = valid_pred_frames
    eval_pose_mask = valid_pred_frames & pose_frame_valid

    metrics_all = {}
    metrics_pose = {}

    if eval_all_mask.sum() > 0:
        # Resize GT to match the model output size
        s = pred_maps.shape[-1]
        gt_resized = gt_raw.copy()
        if s != 21:
            gt_resized = np.stack([
                np.stack([cv2.resize(gt_raw[t, h], (s, s), interpolation=cv2.INTER_LINEAR) 
                         for h in range(2)])
                for t in range(len(gt_raw))
            ])  # (T, 2, S, S)
        
        metrics_all = compute_tactile_metrics(
            pred_maps[eval_all_mask],      # (N, 2, S, S)
            gt_resized[eval_all_mask],     # (N, 2, S, S)
            threshold=0.1,                 # contact threshold
            min_contact_ratio=0.05,        # required ratio of above-threshold points (5%)
            exclude_bend_sensors=True      # exclude finger bend-sensor regions
        )

        if eval_pose_mask.sum() > 0:
            metrics_pose = compute_tactile_metrics(
                pred_maps[eval_pose_mask],
                gt_resized[eval_pose_mask],
                threshold=0.1,                 # contact threshold
                min_contact_ratio=0.05,        # required ratio of above-threshold points (5%)
                exclude_bend_sensors=True      # exclude finger bend-sensor regions
            )
        else:
            metrics_pose = {
                'temporal_accuracy': np.nan,
                'contact_iou': np.nan,
                'volumetric_iou': np.nan,
                'mae': np.nan,
                'mae_left': np.nan,
                'mae_right': np.nan,
            }
    else:
        # Return NaN when no valid prediction frames exist
        metrics_all = {
            'temporal_accuracy': np.nan,
            'contact_iou': np.nan,
            'volumetric_iou': np.nan,
            'mae': np.nan,
            'mae_left': np.nan,
            'mae_right': np.nan,
        }
        metrics_pose = dict(metrics_all)
    
    # Optional: save inference results to HDF5 for later batch visualization
    if not skip_hdf5:
        result_hdf5_path = os.path.join(output_dir, f'{task_name}_{traj_name}_result.hdf5')
        with h5py.File(result_hdf5_path, 'w') as f:
            f.create_dataset('ego_frames', data=ego_frames, compression='gzip')
            f.create_dataset('left_frames', data=left_frames, compression='gzip')
            f.create_dataset('right_frames', data=right_frames, compression='gzip')
            f.create_dataset('pred_maps', data=pred_maps, compression='gzip')
            f.create_dataset('gt_raw', data=gt_raw, compression='gzip')
            f.create_dataset('pose_frame_valid', data=pose_frame_valid)
            if timestamps is not None:
                f.create_dataset('timestamps', data=timestamps)
            f.attrs['task_name'] = task_name
            f.attrs['traj_name'] = traj_name
            f.attrs['vmax'] = vmax
            f.attrs['fps'] = fps
            f.attrs['pressure_style'] = pressure_style
            f.attrs['use_mano_3d'] = use_mano_3d
    
    # Optional: also generate a video unless --skip_video is specified
    if not skip_video:
        out_path = os.path.join(output_dir, f'{task_name}_{traj_name}_tactile.mp4')
        make_comparison_video(
            ego_frames, left_frames, right_frames,
            pred_maps, gt_raw, timestamps, task_name, traj_name,
            out_path, fps=fps, vmax=vmax,
            pressure_style=pressure_style,
            use_mano_3d=use_mano_3d
        )
    
    return (
        task_name,
        traj_name,
        metrics_all,
        metrics_pose,
        int(valid_pred_frames.sum()),
        int(len(valid_pred_frames)),
        int(pose_frame_valid.sum()),
        int(eval_pose_mask.sum()),
    )


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint .pth path')
    parser.add_argument('--config', type=str,
                        default='configs/touchanything_train.yaml')
    parser.add_argument('--hdf5', type=str, default=None,
                        help='single HDF5 file; if omitted, automatically use the first N val trajectories')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/tactile_inference')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test',"test_seen","test_unseen"])
    parser.add_argument('--num_traj', type=int, default=3,
                        help='number of trajectories to process in automatic mode')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batched inference batch size, default 16')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of parallel worker processes, default 4')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use, comma-separated, e.g. "0,1,2"')
    parser.add_argument('--views', type=str, default='all',
                        choices=['ego', 'ego+left', 'ego+right', 'all'],
                        help='camera-view configuration, default all')
    parser.add_argument('--pose_source', type=str, default=None,
                        choices=['rokoko', 'hamer'],
                        help='pose source, defaults to config data.pose_source')
    parser.add_argument('--fallback_pose_source', type=str, default=None,
                        choices=['rokoko', 'hamer'],
                        help='fallback pose source, defaults to config data.fallback_pose_source')
    parser.add_argument('--pressure_style', type=str, default='2d',
                        choices=['2d', 'mano_3d'],
                        help='pressure display style: 2d=2D heatmap, mano_3d=MANO 3D rendering')
    parser.add_argument('--use_mano_3d', action='store_true',
                        help='whether to use MANO 3D rendering, requires --pressure_style mano_3d')
    parser.add_argument('--skip_video', action='store_true',
                        help='skip video generation and only save HDF5 inference results, recommended for large-scale inference')
    parser.add_argument('--skip_hdf5', action='store_true',
                        help='skip saving HDF5 results and only generate videos and metrics to save storage')
    parser.add_argument('--trajectory_list', type=str, default=None,
                        help='trajectory list file, one HDF5 path per line, for lite inference')
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)

    pose_source = args.pose_source or cfg['data'].get('pose_source', 'rokoko')
    fallback_pose_source = args.fallback_pose_source
    if fallback_pose_source is None:
        fallback_pose_source = cfg['data'].get('fallback_pose_source', None)
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    # ---- Determine the HDF5 file list ----
    if args.trajectory_list:
        # Read from a trajectory list file for lite inference
        print(f'Reading from trajectory list file: {args.trajectory_list}')
        with open(args.trajectory_list, 'r', encoding='utf-8') as f:
            hdf5_list = [line.strip() for line in f if line.strip()]
        print(f'Lite inference mode: total {len(hdf5_list)} samples')
    elif args.hdf5:
        hdf5_list = [args.hdf5]
    else:
        split_file = cfg['data'].get('split_file')
        data_root  = Path(cfg['data']['data_root'])
        if split_file and Path(split_file).exists():
            with open(split_file) as fp:
                split_data = json.load(fp)
            hdf5_list = split_data.get(args.split, [])[:args.num_traj]
        else:
            all_hdf5  = sorted(data_root.rglob('*.hdf5'))
            n = len(all_hdf5)
            tr_end = int(n * 0.8)
            val_end = tr_end + int(n * 0.1)
            if args.split == 'val':
                hdf5_list = [str(p) for p in all_hdf5[tr_end:val_end][:args.num_traj]]
            elif args.split == 'test':
                hdf5_list = [str(p) for p in all_hdf5[val_end:][:args.num_traj]]
            else:
                hdf5_list = [str(p) for p in all_hdf5[:tr_end][:args.num_traj]]

    print(f'Trajectories to infer: {len(hdf5_list)} ')
    print(f'Batched inference batch_size: {args.batch_size}')
    print(f'number of worker processes: {args.num_workers}')
    print(f'Using GPUs: {gpu_ids}')
    print(f'View configuration: {args.views}')
    print(f'Pose source: {pose_source}')
    print(f'Fallback pose source: {fallback_pose_source}')
    print(f'Save HDF5: {"No" if args.skip_hdf5 else "Yes"}')
    print(f'Generate video: {"No" if args.skip_video else "Yes"}')
    if not args.skip_video:
        print(f'Pressure display style: {args.pressure_style}')
        if args.pressure_style == 'mano_3d':
            print(f'MANO 3D rendering: {"enabled" if args.use_mano_3d else "disabled"}')
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Multiprocess parallel inference ----
    # Use a Pool initializer so each worker loads the model once and reuses it across trajectories
    task_args = [
        (
            hdf5_path,
            args.output_dir,
            args.fps,
            args.batch_size,
            args.views,
            pose_source,
            fallback_pose_source,
            args.pressure_style,
            args.use_mano_3d,
            args.skip_video,
            args.skip_hdf5,
        )
        for hdf5_path in hdf5_list
    ]
    
    results = []
    
    # Serial mode vs. parallel mode
    if args.num_workers == 1:
        # Serial mode: process directly in the main process to avoid multiprocessing overhead and deadlocks
        print('Using serial mode (num_workers=1)')
        _worker_init(args.checkpoint, args.config, gpu_ids)
        
        with tqdm(total=len(task_args), desc='Overall progress') as pbar:
            for task_arg in task_args:
                try:
                    result = process_single_trajectory(task_arg)
                    (task_name, traj_name, metrics_all, metrics_pose,
                     valid_pred, total, pose_valid_cnt, eval_pose_cnt) = result
                    results.append(result)
                    print(
                        f'\n[{task_name}/{traj_name}] '
                        f'(valid prediction frames={valid_pred}/{total}, valid pose frames={pose_valid_cnt}/{total}, '
                        f'evaluable valid-pose frames={eval_pose_cnt})'
                    )
                    print(f'  [ALL]  Temporal Acc: {metrics_all["temporal_accuracy"]:.4f}  '
                          f'Contact IoU: {metrics_all["contact_iou"]:.4f}  '
                          f'Vol IoU: {metrics_all["volumetric_iou"]:.4f}  '
                          f'MAE: {metrics_all["mae"]:.4f}')
                    print(f'  [POSE] Temporal Acc: {metrics_pose["temporal_accuracy"]:.4f}  '
                          f'Contact IoU: {metrics_pose["contact_iou"]:.4f}  '
                          f'Vol IoU: {metrics_pose["volumetric_iou"]:.4f}  '
                          f'MAE: {metrics_pose["mae"]:.4f}')
                except Exception as e:
                    import traceback
                    print(f'\nError: {e}')
                    traceback.print_exc()
                pbar.update(1)
    else:
        # Parallel mode: use multiprocessing Pool
        print(f'Using parallel mode (num_workers={args.num_workers})')
        with Pool(
            processes=args.num_workers,
            initializer=_worker_init,
            initargs=(args.checkpoint, args.config, gpu_ids),
        ) as pool:
            with tqdm(total=len(task_args), desc='Overall progress') as pbar:
                for result in pool.imap_unordered(process_single_trajectory, task_args):
                    try:
                        (task_name, traj_name, metrics_all, metrics_pose,
                         valid_pred, total, pose_valid_cnt, eval_pose_cnt) = result
                        results.append(result)
                        print(
                            f'\n[{task_name}/{traj_name}] '
                            f'(valid prediction frames={valid_pred}/{total}, valid pose frames={pose_valid_cnt}/{total}, '
                            f'evaluable valid-pose frames={eval_pose_cnt})'
                        )
                        print(f'  [ALL]  Temporal Acc: {metrics_all["temporal_accuracy"]:.4f}  '
                              f'Contact IoU: {metrics_all["contact_iou"]:.4f}  '
                              f'Vol IoU: {metrics_all["volumetric_iou"]:.4f}  '
                              f'MAE: {metrics_all["mae"]:.4f}')
                        print(f'  [POSE] Temporal Acc: {metrics_pose["temporal_accuracy"]:.4f}  '
                              f'Contact IoU: {metrics_pose["contact_iou"]:.4f}  '
                              f'Vol IoU: {metrics_pose["volumetric_iou"]:.4f}  '
                              f'MAE: {metrics_pose["mae"]:.4f}')
                    except Exception as e:
                        import traceback
                        print(f'\nError: {e}')
                        traceback.print_exc()
                    pbar.update(1)
    
    # ---- Summary Statistics ----
    if results:
        # Aggregate both metric sets
        all_frame_metrics = [r[2] for r in results]
        pose_valid_metrics = [r[3] for r in results]
        avg_all = {
            'temporal_accuracy': np.nanmean([m['temporal_accuracy'] for m in all_frame_metrics]),
            'contact_iou': np.nanmean([m['contact_iou'] for m in all_frame_metrics]),
            'volumetric_iou': np.nanmean([m['volumetric_iou'] for m in all_frame_metrics]),
            'mae': np.nanmean([m['mae'] for m in all_frame_metrics]),
            'mae_left': np.nanmean([m['mae_left'] for m in all_frame_metrics]),
            'mae_right': np.nanmean([m['mae_right'] for m in all_frame_metrics]),
        }
        avg_pose = {
            'temporal_accuracy': np.nanmean([m['temporal_accuracy'] for m in pose_valid_metrics]),
            'contact_iou': np.nanmean([m['contact_iou'] for m in pose_valid_metrics]),
            'volumetric_iou': np.nanmean([m['volumetric_iou'] for m in pose_valid_metrics]),
            'mae': np.nanmean([m['mae'] for m in pose_valid_metrics]),
            'mae_left': np.nanmean([m['mae_left'] for m in pose_valid_metrics]),
            'mae_right': np.nanmean([m['mae_right'] for m in pose_valid_metrics]),
        }

        total_valid_pred = int(sum(r[4] for r in results))
        total_frames = int(sum(r[5] for r in results))
        total_pose_valid = int(sum(r[6] for r in results))
        total_pose_eval = int(sum(r[7] for r in results))
        
        print(f'\n========== Summary Statistics ==========')
        print(f'Processed trajectories: {len(results)}')
        print(f'Total valid prediction frames: {total_valid_pred}/{total_frames}')
        print(f'Total valid pose frames: {total_pose_valid}/{total_frames}')
        print(f'Total evaluable valid-pose frames: {total_pose_eval}/{total_frames}')
        print(f'\nPaper metrics (PressureVision) - ALL-frame protocol:')
        print(f'  Temporal Accuracy ↑: {avg_all["temporal_accuracy"]:.4f}')
        print(f'  Contact IoU ↑:       {avg_all["contact_iou"]:.4f}')
        print(f'  Volumetric IoU ↑:    {avg_all["volumetric_iou"]:.4f}')
        print(f'  MAE ↓:               {avg_all["mae"]:.4f}')
        print(f'  MAE Left ↓:          {avg_all["mae_left"]:.4f}')
        print(f'  MAE Right ↓:         {avg_all["mae_right"]:.4f}')
        print(f'\nPaper metrics (PressureVision) - POSE_VALID-frame protocol:')
        print(f'  Temporal Accuracy ↑: {avg_pose["temporal_accuracy"]:.4f}')
        print(f'  Contact IoU ↑:       {avg_pose["contact_iou"]:.4f}')
        print(f'  Volumetric IoU ↑:    {avg_pose["volumetric_iou"]:.4f}')
        print(f'  MAE ↓:               {avg_pose["mae"]:.4f}')
        print(f'  MAE Left ↓:          {avg_pose["mae_left"]:.4f}')
        print(f'  MAE Right ↓:         {avg_pose["mae_right"]:.4f}')
        print(f'\nOutput directory: {args.output_dir}')
        
        # Save results to a txt file
        results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('Tactile pressure prediction evaluation results\n')
            f.write('=' * 80 + '\n\n')
            
            f.write(f'Model checkpoint: {args.checkpoint}\n')
            f.write(f'Config file: {args.config}\n')
            f.write(f'Dataset splits: {args.split}\n')
            f.write(f'View configuration: {args.views}\n')
            f.write(f'Pose source: {pose_source}\n')
            f.write(f'Fallback pose source: {fallback_pose_source}\n')
            f.write(f'Processed trajectories: {len(results)}\n\n')
            f.write(f'Total valid prediction frames: {total_valid_pred}/{total_frames}\n')
            f.write(f'Total valid pose frames: {total_pose_valid}/{total_frames}\n')
            f.write(f'Total evaluable valid-pose frames: {total_pose_eval}/{total_frames}\n\n')
            
            f.write('-' * 80 + '\n')
            f.write('Per-trajectory detailed results:\n')
            f.write('-' * 80 + '\n\n')
            
            for (task_name, traj_name, metrics_all, metrics_pose,
                 valid_pred, total, pose_valid_cnt, eval_pose_cnt) in results:
                f.write(f'[{task_name}/{traj_name}]\n')
                f.write(f'  valid prediction frames: {valid_pred}/{total}\n')
                f.write(f'  valid pose frames: {pose_valid_cnt}/{total}\n')
                f.write(f'  evaluable valid-pose frames: {eval_pose_cnt}/{total}\n')
                f.write(f'  [ALL] Temporal Accuracy: {metrics_all["temporal_accuracy"]:.4f}\n')
                f.write(f'  [ALL] Contact IoU:       {metrics_all["contact_iou"]:.4f}\n')
                f.write(f'  [ALL] Volumetric IoU:    {metrics_all["volumetric_iou"]:.4f}\n')
                f.write(f'  [ALL] MAE:               {metrics_all["mae"]:.4f}\n')
                f.write(f'  [ALL] MAE Left:          {metrics_all["mae_left"]:.4f}\n')
                f.write(f'  [ALL] MAE Right:         {metrics_all["mae_right"]:.4f}\n')
                f.write(f'  [POSE] Temporal Accuracy:{metrics_pose["temporal_accuracy"]:.4f}\n')
                f.write(f'  [POSE] Contact IoU:      {metrics_pose["contact_iou"]:.4f}\n')
                f.write(f'  [POSE] Volumetric IoU:   {metrics_pose["volumetric_iou"]:.4f}\n')
                f.write(f'  [POSE] MAE:              {metrics_pose["mae"]:.4f}\n')
                f.write(f'  [POSE] MAE Left:         {metrics_pose["mae_left"]:.4f}\n')
                f.write(f'  [POSE] MAE Right:        {metrics_pose["mae_right"]:.4f}\n')
                f.write('\n')
            
            f.write('=' * 80 + '\n')
            f.write('Average metrics (ALL-frame protocol):\n')
            f.write('=' * 80 + '\n\n')
            f.write(f'Temporal Accuracy ↑: {avg_all["temporal_accuracy"]:.4f}\n')
            f.write(f'Contact IoU ↑:       {avg_all["contact_iou"]:.4f}\n')
            f.write(f'Volumetric IoU ↑:    {avg_all["volumetric_iou"]:.4f}\n')
            f.write(f'MAE ↓:               {avg_all["mae"]:.4f}\n')
            f.write(f'MAE Left ↓:          {avg_all["mae_left"]:.4f}\n')
            f.write(f'MAE Right ↓:         {avg_all["mae_right"]:.4f}\n\n')

            f.write('=' * 80 + '\n')
            f.write('Average metrics (POSE_VALID-frame protocol):\n')
            f.write('=' * 80 + '\n\n')
            f.write(f'Temporal Accuracy ↑: {avg_pose["temporal_accuracy"]:.4f}\n')
            f.write(f'Contact IoU ↑:       {avg_pose["contact_iou"]:.4f}\n')
            f.write(f'Volumetric IoU ↑:    {avg_pose["volumetric_iou"]:.4f}\n')
            f.write(f'MAE ↓:               {avg_pose["mae"]:.4f}\n')
            f.write(f'MAE Left ↓:          {avg_pose["mae_left"]:.4f}\n')
            f.write(f'MAE Right ↓:         {avg_pose["mae_right"]:.4f}\n')
        
        print(f'Evaluation results saved to: {results_file}')
    
    print('\nInference complete!')


if __name__ == '__main__':
    # spawn mode: child processes use fresh processes and are fully CUDA-compatible
    # The model is loaded through the Pool initializer, once per worker
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
