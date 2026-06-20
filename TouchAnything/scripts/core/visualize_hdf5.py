#!/usr/bin/env python3
"""
TouchAnything HDF5 visualization entry point.

This script supports two HDF5 schemas:

1. Converted dataset HDF5 files produced by ``convert_to_hdf5.py``.
2. Inference-result HDF5 files produced by
   ``inference_tactile_parallel_mano_style.py``.

For converted dataset HDF5 files, the script renders a verification video with
the same layout as ``visualize_trajectory.py``. For inference-result HDF5
files, it renders the tactile comparison video from the stored prediction
outputs.

Output layout (1440 x 630):
  ┌─────────────┬─────────────┬─────────────┐  270px
  │  Head Cam   │  Left Cam   │  Right Cam  │
  ├─────────────┴──────┬──────┴─────────────┤  360px
  │   3D Hand Skeleton │  Pressure Heatmaps │
  └────────────────────┴────────────────────┘

Pressure values are read directly from the preprocessed
``pressure/left_pressure_grid`` and ``pressure/right_pressure_grid`` datasets
stored in HDF5. No additional baseline correction or interpolation is needed.

Usage:
    # Visualize a single HDF5 file
    python scripts/core/visualize_hdf5.py --hdf5 datasets/TouchAnything_hdf5/pick_up_bottle/xxx.hdf5

    # Visualize an entire directory in parallel
    python scripts/core/visualize_hdf5.py --root datasets/TouchAnything_hdf5 --batch

    # Quick test (first 60 frames)
    python scripts/core/visualize_hdf5.py --hdf5 ... --max_frames 60

    # Save 2D heatmap-mode frame images without writing video
    python scripts/core/visualize_hdf5.py --hdf5 ... --use-2d-heatmap --no-video --frames-output outputs/tactile_frames

    # Adjust the number of workers
    python scripts/core/visualize_hdf5.py --root ... --batch --workers 8
"""

import os
import sys
import argparse

import h5py
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import tempfile
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fonts / colormap (kept aligned with visualize_trajectory.py)
# ---------------------------------------------------------------------------
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans SC', 'WenQuanYi Micro Hei',
                                           'AR PL UMing CN', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

PRESSURE_CMAP = LinearSegmentedColormap.from_list(
    'pressure', ['blue', 'cyan', 'yellow', 'red'])

HAND_MATRIX_ROWS, HAND_MATRIX_COLS = 21, 21

# ---------------------------------------------------------------------------
# ROKOKO 21-joint connectivity (aligned with visualize_trajectory.py)
# ---------------------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

FINGER_JOINT_GROUPS = [
    ([0, 1, 2, 3, 4],    '#FF6B6B'),
    ([0, 5, 6, 7, 8],    '#4ECDC4'),
    ([0, 9, 10, 11, 12], '#45B7D1'),
    ([0, 13, 14, 15, 16],'#96CEB4'),
    ([0, 17, 18, 19, 20],'#FFEAA7'),
]

# ---------------------------------------------------------------------------
# Video layout constants (aligned with visualize_trajectory.py)
# ---------------------------------------------------------------------------
CAM_W, CAM_H     = 480, 270
SKEL_W, SKEL_H   = 720, 360
PRESS_W, PRESS_H = 720, 360
VIDEO_W = CAM_W * 3    # 1440
VIDEO_H = CAM_H + SKEL_H  # 630

BG_COLOR = (26, 26, 46)


# ---------------------------------------------------------------------------
# Helper utilities (aligned with visualize_trajectory.py)
# ---------------------------------------------------------------------------

def put_label(img, text, pos=(10, 24), font_scale=0.65, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# Candidate CJK font paths in priority order.
_CN_FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
]
_CN_FONT_CACHE = {}


def _get_cn_font(size=14):
    if size in _CN_FONT_CACHE:
        return _CN_FONT_CACHE[size]
    for path in _CN_FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                fnt = ImageFont.truetype(path, size)
                _CN_FONT_CACHE[size] = fnt
                return fnt
            except Exception:
                continue
    fnt = ImageFont.load_default()
    _CN_FONT_CACHE[size] = fnt
    return fnt


def put_text_cn(img_bgr, text, pos, font_size=14, color=(180, 180, 180)):
    """Render Unicode text on a BGR NumPy image via PIL."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=_get_cn_font(font_size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def fig_to_bgr(fig, target_w, target_h):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    if (w, h) != (target_w, target_h):
        bgr = cv2.resize(bgr, (target_w, target_h))
    return bgr


# ---------------------------------------------------------------------------
# Skeleton-panel rendering (aligned with visualize_trajectory.py)
# ---------------------------------------------------------------------------

def render_mano_pose_panel(mano_left_img, mano_right_img, w=SKEL_W, h=SKEL_H):
    """Render a MANO 3D pose panel by concatenating left/right images horizontally."""
    # Target size: 720x360, each hand 360x360.
    target_hand_size = (360, 360)
    
    # Resize the left-hand image.
    if mano_left_img is not None:
        left_resized = cv2.resize(mano_left_img, target_hand_size)
    else:
        left_resized = np.zeros((360, 360, 3), dtype=np.uint8)
    
    # Resize the right-hand image.
    if mano_right_img is not None:
        right_resized = cv2.resize(mano_right_img, target_hand_size)
    else:
        right_resized = np.zeros((360, 360, 3), dtype=np.uint8)
    
    # Concatenate horizontally.
    panel = np.hstack([left_resized, right_resized])
    
    # Add labels.
    put_label(panel, 'Left Hand Pose', pos=(10, 24))
    put_label(panel, 'Right Hand Pose', pos=(370, 24))
    
    return panel

def render_skeleton_panel(joints_l, joints_r, w=SKEL_W, h=SKEL_H,
                           view_elev=20, view_azim=-60):
    """Render the 3D skeleton panel as a fallback when MANO rendering is unavailable."""
    dpi = 100
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='#888888', labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#333355')
    ax.grid(False)

    def draw_hand(joints, hand_color):
        if joints is None:
            return
        j = np.array(joints)
        for indices, fc in FINGER_JOINT_GROUPS:
            pts = j[indices]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=fc, linewidth=1.5, alpha=0.9)
        palm_idx = [5, 9, 13, 17]
        for a, b in zip(palm_idx[:-1], palm_idx[1:]):
            ax.plot([j[a, 0], j[b, 0]], [j[a, 1], j[b, 1]], [j[a, 2], j[b, 2]],
                    color='#888888', linewidth=1.0, alpha=0.6)
        ax.scatter(j[:, 0], j[:, 1], j[:, 2],
                   c=hand_color, s=18, depthshade=True, alpha=0.95, zorder=5)

    draw_hand(joints_l, '#00BFFF')
    draw_hand(joints_r, '#FF4500')

    all_pts = [j for j in [joints_l, joints_r] if j is not None]
    if all_pts:
        combined = np.vstack(all_pts)
        center = combined.mean(axis=0)
        radius = max(np.ptp(combined, axis=0)) * 0.55 + 0.05
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel('X', fontsize=7, color='#888888')
    ax.set_ylabel('Y', fontsize=7, color='#888888')
    ax.set_zlabel('Z', fontsize=7, color='#888888')
    ax.set_title('Hand Skeleton (3D)  ■Left  ■Right',
                 fontsize=9, color='#cccccc', pad=4)
    ax.view_init(elev=view_elev, azim=view_azim)
    fig.tight_layout(pad=0.4)

    img = fig_to_bgr(fig, w, h)
    plt.close(fig)
    return img


def render_pose_placeholder_panel(w=SKEL_W, h=SKEL_H):
    """Return a lightweight pose panel used when 2D heatmap mode is selected."""
    panel = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)
    put_label(panel, 'Hand Pose (disabled in 2D heatmap mode)', pos=(10, 24))
    return panel


# ---------------------------------------------------------------------------
# Pressure-panel rendering (aligned with visualize_trajectory.py, with HDF5 labels)
# ---------------------------------------------------------------------------

def render_pressure_panel(grid_l, grid_r, w=PRESS_W, h=PRESS_H,
                          vmax=1.0, global_max=None, from_hdf5=True):
    """Render the original 2D heatmap-style pressure panel."""
    dpi = 100
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')
    PRESSURE_CMAP.set_bad(color='#111122')

    panels = [
        (fig.add_subplot(1, 2, 1), grid_l, 'Left Hand Pressure'),
        (fig.add_subplot(1, 2, 2), grid_r, 'Right Hand Pressure'),
    ]
    for ax, grid, title in panels:
        ax.set_facecolor('#111122')
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap=PRESSURE_CMAP, vmin=0, vmax=vmax,
                       interpolation='nearest', aspect='auto', origin='upper')
        ax.set_title(title, fontsize=9, color='#cccccc', pad=4)
        ax.axis('off')
        label_str = f'Pressure (0-{vmax:.2f})' if vmax <= 1.0 else f'Pressure (0-{vmax:.0f})'
        cbar = fig.colorbar(im, ax=ax, label=label_str,
                            shrink=0.7, fraction=0.04)
        cbar.ax.tick_params(labelsize=6, colors='#aaaaaa')
        cbar.ax.yaxis.label.set_size(7)
        cbar.ax.yaxis.label.set_color('#aaaaaa')
        valid = grid[~np.isnan(grid)] if np.any(np.isnan(grid)) else grid.ravel()
        peak_thresh = vmax * 0.05  # Only mark peaks above 5% of vmax.
        if len(valid) > 0 and valid.max() > peak_thresh:
            pos = np.unravel_index(np.nanargmax(grid), grid.shape)
            ax.plot(pos[1], pos[0], 'w+', markersize=7,
                    markeredgewidth=1.5, alpha=0.9)

    fmt = '.2f' if (global_max is not None and global_max <= 1.0) else '.1f'
    footer = f'baseline-corrected  |  max={global_max:{fmt}}'
    if from_hdf5:
        footer += '  [HDF5 normalized]' if (global_max is not None and global_max <= 1.0) else '  [HDF5]'
    if global_max is not None:
        fig.text(0.98, 0.02, footer,
                 ha='right', va='bottom', fontsize=7, color='#888888')

    fig.subplots_adjust(left=0.03, right=0.92, top=0.92, bottom=0.05,
                        wspace=0.15)
    img = fig_to_bgr(fig, w, h)
    plt.close(fig)
    return img


def render_mano_pressure_panel(mano_left_img, mano_right_img, w=PRESS_W, h=PRESS_H):
    """Render a MANO 3D pressure panel by concatenating left/right hand images."""
    # Target size: 720x360, each hand 360x360.
    target_hand_size = (360, 360)
    
    # Resize the left-hand image.
    if mano_left_img is not None:
        left_resized = cv2.resize(mano_left_img, target_hand_size)
    else:
        left_resized = np.zeros((360, 360, 3), dtype=np.uint8)
    
    # Resize the right-hand image.
    if mano_right_img is not None:
        right_resized = cv2.resize(mano_right_img, target_hand_size)
    else:
        right_resized = np.zeros((360, 360, 3), dtype=np.uint8)
    
    # Concatenate horizontally.
    panel = np.hstack([left_resized, right_resized])
    
    # Add labels.
    put_label(panel, 'Left Hand Pressure', pos=(10, 24))
    put_label(panel, 'Right Hand Pressure', pos=(370, 24))
    
    return panel


# ---------------------------------------------------------------------------
# Core: visualize a single HDF5 file
# ---------------------------------------------------------------------------

def _visualize_inference_result_hdf5(
    hdf5_path,
    output_path,
    max_frames=None,
    fps=15,
    use_mano_3d=True,
    save_video=True,
):
    """Visualize an inference-result HDF5 file via the shared comparison renderer."""
    if not save_video:
        raise ValueError(
            "--no-video is only supported for converted dataset HDF5 files; "
            "inference-result HDF5 files currently render through make_comparison_video."
        )

    with h5py.File(hdf5_path, 'r') as f:
        ego_frames = f['ego_frames'][...]
        left_frames = f['left_frames'][...]
        right_frames = f['right_frames'][...]
        pred_maps = f['pred_maps'][...]
        gt_raw = f['gt_raw'][...]
        timestamps = f['timestamps'][...] if 'timestamps' in f else None
        task_name = str(f.attrs.get('task_name', 'unknown_task'))
        traj_name = str(f.attrs.get('traj_name', Path(hdf5_path).stem))
        vmax = float(f.attrs.get('vmax', 1.0))
        pressure_style = str(f.attrs.get('pressure_style', '2d'))
        use_mano_3d = bool(f.attrs.get('use_mano_3d', use_mano_3d))
        fps = int(f.attrs.get('fps', fps))

    if max_frames:
        ego_frames = ego_frames[:max_frames]
        left_frames = left_frames[:max_frames]
        right_frames = right_frames[:max_frames]
        pred_maps = pred_maps[:max_frames]
        gt_raw = gt_raw[:max_frames]
        if timestamps is not None:
            timestamps = timestamps[:max_frames]

    from scripts.core.inference_tactile_parallel_mano_style import make_comparison_video

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Inference-result HDF5 detected: {hdf5_path}")
    print(f"Rendering comparison video via visualize_hdf5 -> {output_path}")
    make_comparison_video(
        ego_frames,
        left_frames,
        right_frames,
        pred_maps,
        gt_raw,
        timestamps,
        task_name,
        traj_name,
        str(output_path),
        fps=fps,
        vmax=vmax,
        pressure_style=pressure_style,
        use_mano_3d=use_mano_3d,
    )


def visualize_hdf5(
    hdf5_path,
    output_path,
    max_frames=None,
    fps=15,
    use_mano_3d=True,
    save_video=True,
    save_frames=True,
    frames_output_dir=None,
):
    """
    Generate a visualization video from an HDF5 file.

    Pressure values are read directly from the preprocessed left/right
    pressure grids stored in the HDF5 file. The script supports either
    2D heatmaps or MANO 3D rendering.

    Args:
        hdf5_path: Path to the HDF5 file
        output_path: Output video path
        max_frames: Optional maximum frame count
        fps: Output video frame rate
        use_mano_3d: Whether to use MANO 3D visualization (default: True)
        save_video: Whether to write an MP4 visualization video
        save_frames: Whether to save per-frame PNG panels
        frames_output_dir: Optional base directory for per-frame PNG panels
    """
    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)

    with h5py.File(hdf5_path, 'r') as f:
        if all(k in f for k in ('ego_frames', 'left_frames', 'right_frames', 'pred_maps', 'gt_raw')):
            return _visualize_inference_result_hdf5(
                hdf5_path,
                output_path,
                max_frames=max_frames,
                fps=fps,
                use_mano_3d=use_mano_3d,
                save_video=save_video,
            )

        num_frames   = int(f['metadata'].attrs['num_frames'])
        traj_id      = str(f['metadata'].attrs['trajectory_id'])
        task_name    = str(f['metadata'].attrs['task_name'])
        timestamps   = f['timestamps'][...]

        # Load all pressure grids so the script can compute a global max.
        left_grids  = f['pressure/left_pressure_grid'][...]   # (T, 21, 21)
        right_grids = f['pressure/right_pressure_grid'][...]  # (T, 21, 21)
        f_attrs     = dict(f['pressure'].attrs)  # Normalization metadata.

    if max_frames:
        num_frames = min(num_frames, max_frames)
        timestamps  = timestamps[:num_frames]
        left_grids  = left_grids[:num_frames]
        right_grids = right_grids[:num_frames]

    already_normalized = bool(f_attrs.get('normalized', False))
    if already_normalized:
        global_max = 1.0
    else:
        global_max = max(float(np.nanmax(left_grids)), float(np.nanmax(right_grids)), 1.0)
    
    vis_mode = "MANO 3D" if use_mano_3d else "2D Heatmap"
    requested_mano_3d = use_mano_3d
    print(f"Trajectory: {task_name}/{traj_id}  |  Frames: {num_frames}  |  Max pressure: {global_max:.2f}{'  (normalized)' if already_normalized else ''}")
    print(f"Visualization mode: {vis_mode}")
    if save_video:
        print(f"Video output: {output_path}")
    if save_frames:
        frames_base_dir = Path(frames_output_dir) if frames_output_dir else output_path.parent / "frames"
        print(f"Frame output: {frames_base_dir / traj_id}")

    # If MANO 3D is enabled, generate the MANO visualization assets first.
    mano_left_dir = None
    mano_right_dir = None
    tactile_left_dir = None
    tactile_right_dir = None
    temp_dir = None
    
    if use_mano_3d:
        print("\nGenerating MANO 3D visualization with OpenTouch...")
        temp_dir = Path(tempfile.mkdtemp(prefix='mano_vis_'))
        
        # Import local visualization helpers.
        script_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(script_dir))
        
        try:
            # Enable headless rendering.
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
            
            # Create a temporary OpenTouch-format HDF5 file.
            temp_opentouch_hdf5 = temp_dir / 'opentouch_format.hdf5'
            
            print("  Converting data to OpenTouch format...")
            with h5py.File(hdf5_path, 'r') as src:
                with h5py.File(temp_opentouch_hdf5, 'w') as dst:
                    # Build the structure expected by OpenTouch.
                    data_group = dst.create_group('data')
                    demo_group = data_group.create_group('demo_0')
                    
                    # Copy hand keypoints for pose visualization.
                    if 'hands/left_joint_xyz' in src:
                        demo_group.create_dataset(
                            'left_hand_landmarks',
                            data=src['hands/left_joint_xyz'][:num_frames]
                        )
                    if 'hands/right_joint_xyz' in src:
                        demo_group.create_dataset(
                            'right_hand_landmarks',
                            data=src['hands/right_joint_xyz'][:num_frames]
                        )
                    
                    # Copy pressure data for tactile-MANO rendering.
                    if 'pressure/left_pressure_grid' in src:
                        left_pressure = src['pressure/left_pressure_grid'][:num_frames]
                        left_pressure = np.nan_to_num(left_pressure, nan=0.0)
                        demo_group.create_dataset('left_pressure', data=left_pressure)
                    if 'pressure/right_pressure_grid' in src:
                        right_pressure = src['pressure/right_pressure_grid'][:num_frames]
                        right_pressure = np.nan_to_num(right_pressure, nan=0.0)
                        demo_group.create_dataset('right_pressure', data=right_pressure)
            
            # Use helpers from scripts/core/load_data.py.
            from load_data import export_pose_mano, export_tactile_mano_ta
            
            print("  Rendering MANO 3D pose images for both hands...")
            mano_output_dir = temp_dir / 'mano_output'
            results = export_pose_mano(
                file_path=str(temp_opentouch_hdf5),
                demo_id='demo_0',
                output_dir=str(mano_output_dir),
                dataset_names=('left_hand_landmarks', 'right_hand_landmarks'),
                target_size=(480, 480),
                background_color=(249, 235, 142),
                mano_side='auto',  # _pick_side infers the side from the dataset name.
                use_cuda=True,
            )
            
            mano_left_dir = mano_output_dir / 'left_hand_landmarks'
            mano_right_dir = mano_output_dir / 'right_hand_landmarks'
            
            if results.get('left_hand_landmarks') and results.get('right_hand_landmarks'):
                print(f"  ✓ MANO 3D pose images generated")
            else:
                raise Exception("MANO pose rendering did not complete successfully")
            
            # Generate tactile-MANO visualization by mapping pressure to MANO.
            print("  Rendering tactile MANO images for both hands...")
            tactile_output_dir = temp_dir / 'tactile_output'
            
            # Locate mapping files, preferring the in-repo MANO visualization assets.
            project_root = script_dir.parent.parent
            mapping_roots = [
                project_root / 'scripts' / 'tools' / 'mano_visualization',
                project_root / 'third_party' / 'opentouch_visualization_subset' / 'opentouch' / 'preprocess',
                project_root / 'ref' / 'opentouch' / 'preprocess',
            ]
            mapping_root = next((p for p in mapping_roots if p.exists()), mapping_roots[0])
            mapping_left = mapping_root / 'ta_to_mano_mapping_left_visual.json'
            mapping_right = mapping_root / 'ta_to_mano_mapping_right_visual.json'
            
            # Render left-hand tactile MANO with a larger target size to avoid cropping.
            if mapping_left.exists():
                export_tactile_mano_ta(
                    file_path=str(temp_opentouch_hdf5),
                    demo_id='demo_0',
                    output_dir=str(tactile_output_dir),
                    dataset_names=('left_pressure',),
                    target_size=(640, 640),  # Larger target size to avoid cropping.
                    max_value=1.0,
                    temporal_alpha=0.4,
                    layout_json_path=str(mapping_left)
                )
                tactile_left_dir = tactile_output_dir / 'left_pressure'
            
            # Render right-hand tactile MANO with a larger target size to avoid cropping.
            if mapping_right.exists():
                export_tactile_mano_ta(
                    file_path=str(temp_opentouch_hdf5),
                    demo_id='demo_0',
                    output_dir=str(tactile_output_dir),
                    dataset_names=('right_pressure',),
                    target_size=(640, 640),  # Larger target size to avoid cropping.
                    max_value=1.0,
                    temporal_alpha=0.4,
                    layout_json_path=str(mapping_right)
                )
                tactile_right_dir = tactile_output_dir / 'right_pressure'
            
            if tactile_left_dir and tactile_right_dir:
                print(f"  ✓ Tactile MANO images generated")
            else:
                print(f"  ⚠️  Failed to generate tactile MANO images; falling back to 2D heatmaps")
            
        except Exception as e:
            print(f"  ⚠️  MANO 3D generation failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Falling back to 2D heatmap mode")
            use_mano_3d = False
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
                temp_dir = None

    if save_video:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    ffmpeg_proc = None
    if save_video:
        # Use ffmpeg instead of cv2.VideoWriter for better encoding support.
        import subprocess
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{VIDEO_W}x{VIDEO_H}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-an', '-vcodec', 'libx264', '-preset', 'medium',
            '-crf', '23', '-pix_fmt', 'yuv420p', str(output_path)
        ]
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    black_cam = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    frames_dir = None
    if save_frames:
        frames_base_dir = Path(frames_output_dir) if frames_output_dir else output_path.parent / "frames"
        frames_dir = frames_base_dir / traj_id
        frames_dir.mkdir(parents=True, exist_ok=True)
    pose_placeholder_panel = None
    if not requested_mano_3d:
        pose_placeholder_panel = render_pose_placeholder_panel()

    # Process frames one by one and load images on demand from HDF5.
    with h5py.File(hdf5_path, 'r') as f:
        for fi in range(num_frames):
            # Prepare the three camera views (HDF5 stores RGB; cv2 uses BGR).
            def prep_cam(dataset_key, label):
                img_rgb = f[dataset_key][fi]           # (H, W, 3) uint8 RGB
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                img_bgr = cv2.resize(img_bgr, (CAM_W, CAM_H))
                put_label(img_bgr, label)
                return img_bgr

            cam_chest = prep_cam('images/chest_color', 'Head')
            cam_left = prep_cam('images/left_color',  'Left Wrist')
            cam_right = prep_cam('images/right_color', 'Right Wrist')
            row_top = np.hstack([cam_chest, cam_left, cam_right])

            # Pressure grids (read directly from preprocessed HDF5 data).
            grid_l = left_grids[fi]    # (21, 21) float32
            grid_r = right_grids[fi]   # (21, 21) float32

            # Render the bottom panels.
            # Bottom-left: pose panel.
            if use_mano_3d and mano_left_dir and mano_right_dir:
                # Use MANO 3D pose visualization.
                pose_left_file = mano_left_dir / f"demo_0_{fi:05d}.png"
                pose_right_file = mano_right_dir / f"demo_0_{fi:05d}.png"
                
                pose_left_img = cv2.imread(str(pose_left_file)) if pose_left_file.exists() else None
                pose_right_img = cv2.imread(str(pose_right_file)) if pose_right_file.exists() else None
                
                skel_panel = render_mano_pose_panel(pose_left_img, pose_right_img)
            elif requested_mano_3d:
                # Use 3D skeleton visualization.
                joints_l = f['hands/left_joint_xyz'][fi]   # (21, 3)
                joints_r = f['hands/right_joint_xyz'][fi]  # (21, 3)
                skel_panel = render_skeleton_panel(joints_l, joints_r)
            else:
                # In 2D heatmap mode, keep the output layout/files but skip the
                # expensive per-frame 3D pose rendering.
                skel_panel = pose_placeholder_panel
            
            # Bottom-right: pressure panel (tactile MANO or 2D heatmap fallback).
            if use_mano_3d and tactile_left_dir and tactile_right_dir:
                # Use tactile MANO visualization.
                tactile_left_file = tactile_left_dir / f"demo_0_{fi:05d}.png"
                tactile_right_file = tactile_right_dir / f"demo_0_{fi:05d}.png"
                
                tactile_left_img = cv2.imread(str(tactile_left_file)) if tactile_left_file.exists() else None
                tactile_right_img = cv2.imread(str(tactile_right_file)) if tactile_right_file.exists() else None
                
                press_panel = render_mano_pressure_panel(tactile_left_img, tactile_right_img)
            else:
                # Use 2D heatmaps.
                press_panel = render_pressure_panel(
                    grid_l, grid_r,
                    vmax=global_max, global_max=global_max,
                    from_hdf5=True,
                )
            
            row_bottom = np.hstack([skel_panel, press_panel])

            # Merge the frame.
            frame_out = np.vstack([row_top, row_bottom])

            # Bottom info line. Use PIL so task names can remain Unicode.
            ts_sec = (int(timestamps[fi]) - int(timestamps[0])) / 1000.0
            info_line = (f"Frame {fi+1:03d}/{num_frames}   t={ts_sec:6.2f}s"
                         f"   ts={timestamps[fi]}   [{task_name}]")
            frame_out = put_text_cn(
                frame_out, info_line,
                pos=(10, VIDEO_H - 22), font_size=14, color=(180, 180, 180))

            # HDF5 watermark (top-right, ASCII only).
            cv2.putText(frame_out, 'HDF5',
                        (VIDEO_W - 70, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 120), 2, cv2.LINE_AA)

            if save_frames:
                # Save individual panels as separate images.
                frame_prefix = f"frame_{fi:05d}"
                cv2.imwrite(str(frames_dir / f"{frame_prefix}_cam_chest.png"), cam_chest)
                cv2.imwrite(str(frames_dir / f"{frame_prefix}_cam_left.png"), cam_left)
                cv2.imwrite(str(frames_dir / f"{frame_prefix}_cam_right.png"), cam_right)
                cv2.imwrite(str(frames_dir / f"{frame_prefix}_pose_panel.png"), skel_panel)
                cv2.imwrite(str(frames_dir / f"{frame_prefix}_pressure_panel.png"), press_panel)
                cv2.imwrite(str(frames_dir / f"{frame_prefix}_combined.png"), frame_out)
                
                # If MANO 3D is enabled, also save the raw MANO images.
                if use_mano_3d:
                    if 'pose_left_img' in locals() and pose_left_img is not None:
                        cv2.imwrite(str(frames_dir / f"{frame_prefix}_mano_left_pose.png"), pose_left_img)
                    if 'pose_right_img' in locals() and pose_right_img is not None:
                        cv2.imwrite(str(frames_dir / f"{frame_prefix}_mano_right_pose.png"), pose_right_img)
                    if 'tactile_left_img' in locals() and tactile_left_img is not None:
                        cv2.imwrite(str(frames_dir / f"{frame_prefix}_mano_left_pressure.png"), tactile_left_img)
                    if 'tactile_right_img' in locals() and tactile_right_img is not None:
                        cv2.imwrite(str(frames_dir / f"{frame_prefix}_mano_right_pressure.png"), tactile_right_img)

            if save_video:
                ffmpeg_proc.stdin.write(frame_out.tobytes())

            if (fi + 1) % 20 == 0 or fi == num_frames - 1:
                print(f"  [{fi+1:3d}/{num_frames}] {100*(fi+1)//num_frames:3d}%  t={ts_sec:.2f}s")

    if save_video:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(f"✓ Saved video to: {output_path}")
    
    # Report where the individual panel images were saved.
    if save_frames and frames_dir and frames_dir.exists():
        print(f"✓ Saved panel images to: {frames_dir}")
        print(f"  - frame_XXXXX_cam_chest.png (head camera)")
        print(f"  - frame_XXXXX_cam_left.png (left-wrist camera)")
        print(f"  - frame_XXXXX_cam_right.png (right-wrist camera)")
        print(f"  - frame_XXXXX_pose_panel.png (hand-pose panel)")
        print(f"  - frame_XXXXX_pressure_panel.png (pressure panel)")
        if use_mano_3d:
            print(f"  - frame_XXXXX_mano_left_pose.png (left-hand MANO pose)")
            print(f"  - frame_XXXXX_mano_right_pose.png (right-hand MANO pose)")
            print(f"  - frame_XXXXX_mano_left_pressure.png (left-hand MANO pressure)")
            print(f"  - frame_XXXXX_mano_right_pressure.png (right-hand MANO pressure)")
        print(f"  - frame_XXXXX_combined.png (composited frame)\n")
    
    # Clean up temporary MANO files.
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"✓ Removed temporary files")


# ---------------------------------------------------------------------------
# Batch visualization
# ---------------------------------------------------------------------------

def _vis_worker(args):
    """Worker function used by ProcessPoolExecutor."""
    hdf5_path, output_path, max_frames, fps, use_mano_3d, save_video, save_frames, frames_output_dir = args
    try:
        visualize_hdf5(
            hdf5_path,
            output_path,
            max_frames=max_frames,
            fps=fps,
            use_mano_3d=use_mano_3d,
            save_video=save_video,
            save_frames=save_frames,
            frames_output_dir=frames_output_dir,
        )
        return (str(hdf5_path), None)
    except Exception as e:
        import traceback
        return (str(hdf5_path), traceback.format_exc())


def batch_visualize(
    root_dir,
    output_dir,
    workers=4,
    max_frames=None,
    fps=15,
    use_mano_3d=True,
    max_trajectories=None,
    save_video=True,
    save_frames=True,
    frames_output_dir=None,
):
    """
    Batch-visualize all HDF5 files under ``root_dir`` in parallel.

    Args:
        root_dir: HDF5 dataset root directory generated by ``convert_to_hdf5.py``
        output_dir: Video output directory, preserving the original subdirectory layout
        workers: Number of worker processes
        max_frames: Per-trajectory frame cap (None = all frames)
        fps: Output frame rate
        max_trajectories: Max trajectories per scene (None = all)
        save_video: Whether to write MP4 visualization videos
        save_frames: Whether to save per-frame PNG panels
        frames_output_dir: Optional base directory for per-frame PNG panels
    """
    root_dir   = Path(root_dir)
    output_dir = Path(output_dir)

    all_hdf5 = sorted(root_dir.rglob('*.hdf5'))
    if not all_hdf5:
        print(f"Error: no HDF5 files found under: {root_dir}")
        return

    # If max_trajectories is set, group by scene and cap each group.
    if max_trajectories:
        from collections import defaultdict
        scene_groups = defaultdict(list)
        for hdf5_file in all_hdf5:
            scene = hdf5_file.parent.name  # Scene category (parent directory name).
            scene_groups[scene].append(hdf5_file)
        
        # Keep at most max_trajectories per scene.
        selected_hdf5 = []
        for scene, files in sorted(scene_groups.items()):
            selected = files[:max_trajectories]
            selected_hdf5.extend(selected)
            if len(files) > max_trajectories:
                print(f"Scene '{scene}': selected {len(selected)}/{len(files)} trajectories")
        all_hdf5 = selected_hdf5

    print(f"==========================================")
    print(f"Batch HDF5 visualization")
    print(f"==========================================")
    print(f"Dataset directory: {root_dir}")
    print(f"Output directory: {output_dir}")
    if frames_output_dir:
        print(f"Frame output directory: {frames_output_dir}")
    print(f"Total files: {len(all_hdf5)}")
    print(f"Worker processes: {workers}")
    if max_frames:
        print(f"Max frames per trajectory: {max_frames}")
    if max_trajectories:
        print(f"Max trajectories per scene: {max_trajectories}")
    print(f"==========================================\n")

    # Build the task list while preserving the subdirectory structure.
    tasks = []
    for hdf5_file in all_hdf5:
        rel = hdf5_file.relative_to(root_dir)
        out_path = output_dir / rel.with_suffix('.mp4')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        task_frames_output_dir = None
        if frames_output_dir:
            task_frames_output_dir = Path(frames_output_dir) / rel.parent

        tasks.append((
            str(hdf5_file),
            str(out_path),
            max_frames,
            fps,
            use_mano_3d,
            save_video,
            save_frames,
            str(task_frames_output_dir) if task_frames_output_dir else None,
        ))

    success, fail = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_vis_worker, t): t for t in tasks}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Batch visualization")
        for future in pbar:
            hdf5_path_str, error = future.result()
            name = Path(hdf5_path_str).name
            if error is None:
                success += 1
                pbar.set_postfix(ok=success, fail=fail, last=name)
            else:
                fail += 1
                tqdm.write(f"✗ Failed: {name}\n{error}")

    print(f"\n==========================================")
    print(f"Done  success: {success}  failed: {fail}")
    print(f"Output: {output_dir}")
    print(f"==========================================")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    _script_dir  = Path(__file__).resolve().parent
    _project_root = _script_dir.parents[1]

    parser = argparse.ArgumentParser(description='TouchAnything HDF5 visualization (conversion validation)')
    parser.add_argument('--hdf5', type=str, default=None,
                        help='Path to a single HDF5 file')
    parser.add_argument('--root', type=str, default=None,
                        help='Root directory of the HDF5 dataset (batch mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (single-file mode) or output directory (batch mode)')
    parser.add_argument('--batch', action='store_true',
                        help='Batch mode: visualize all HDF5 files under --root')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames per trajectory (useful for quick tests)')
    parser.add_argument('--max_trajectories', type=int, default=None,
                        help='Maximum trajectories per scene category (batch mode)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Output frame rate (default: 30)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker processes in batch mode (default: 4)')
    parser.add_argument('--use-mano-3d', action='store_true', default=True,
                        help='Use MANO 3D visualization (default: enabled)')
    parser.add_argument('--use-2d-heatmap', dest='use_mano_3d', action='store_false',
                        help='Use 2D heatmaps instead of MANO 3D')
    parser.add_argument('--no-video', dest='save_video', action='store_false',
                        help='Do not write an MP4; save frame images only')
    parser.add_argument('--no-frames', dest='save_frames', action='store_false',
                        help='Do not save per-frame PNG panel images')
    parser.add_argument('--frames-output', type=str, default=None,
                        help='Base output directory for per-frame PNG panel images')
    args = parser.parse_args()
    if not args.save_video and not args.save_frames:
        parser.error('At least one output must be enabled; remove --no-video or --no-frames.')

    if args.batch:
        root = args.root or str(_project_root / 'datasets' / 'TouchAnything_hdf5')
        out  = args.output or str(_project_root / 'outputs' / 'hdf5_visualization')
        batch_visualize(root, out,
                        workers=args.workers,
                        max_frames=args.max_frames,
                        fps=args.fps,
                        use_mano_3d=args.use_mano_3d,
                        max_trajectories=args.max_trajectories,
                        save_video=args.save_video,
                        save_frames=args.save_frames,
                        frames_output_dir=args.frames_output)
    else:
        # Single-file mode: use the first available HDF5 if none is specified.
        hdf5 = args.hdf5
        if hdf5 is None:
            candidates = sorted(
                (_project_root / 'datasets').rglob('*.hdf5'))
            if not candidates:
                sys.exit("Error: no HDF5 file found. Please pass one via --hdf5")
            hdf5 = str(candidates[0])
            print(f"[INFO] Using default HDF5: {hdf5}")

        if args.output is None:
            name = Path(hdf5).stem
            out = str(_project_root / 'outputs' / 'hdf5_visualization' / f'vis_{name}.mp4')
        else:
            out = args.output

        visualize_hdf5(
            hdf5,
            out,
            max_frames=args.max_frames,
            fps=args.fps,
            use_mano_3d=args.use_mano_3d,
            save_video=args.save_video,
            save_frames=args.save_frames,
            frames_output_dir=args.frames_output,
        )


if __name__ == '__main__':
    main()
