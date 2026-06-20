#!/usr/bin/env python3
"""
TouchAnything cleaned-data visualization tool

Generate visualization videos from the cleaned dataset (datasets/TouchAnything_Datasets_opensource).
Supports incremental mode: only missing or stale trajectory visualizations are generated.

Output video layout (1440 x 630):
  ┌─────────────┬─────────────┬─────────────┐  270px
  │  Head Cam   │  Left Cam   │  Right Cam  │
  ├─────────────┴──────┬──────┴─────────────┤  360px
  │  WiLoR Annotated   │  Pressure Heatmaps │
  └────────────────────┴────────────────────┘

Data sources:
- Video: chest.mp4, left.mp4, right.mp4
- Pose: wilor_annotated.mp4 (if present)
- Pressure: pressure_grids.npz (if present; otherwise read from jq_pressure.json)

Usage:
    # Visualize one trajectory
    python visualize_cleaned_data.py --traj datasets/TouchAnything_Datasets_opensource/Home/task_name/trajectory_id

    # Batch-visualize the full dataset in incremental mode
    python visualize_cleaned_data.py --root datasets/TouchAnything_Datasets_opensource --batch

    # Force regeneration of all visualizations
    python visualize_cleaned_data.py --root datasets/TouchAnything_Datasets_opensource --batch --force

    # Adjust the number of worker processes
    python visualize_cleaned_data.py --root datasets/TouchAnything_Datasets_opensource --batch --workers 8
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import subprocess

# Video layout constants
CAM_W, CAM_H = 480, 270
PANEL_W, PANEL_H = 720, 360
VIDEO_W = CAM_W * 3  # 1440
VIDEO_H = CAM_H + PANEL_H  # 630

# Pressure visualization constants
HAND_GRID_SIZE = 21
PRESSURE_CMAP = LinearSegmentedColormap.from_list(
    'pressure', ['blue', 'cyan', 'yellow', 'red'])

# Font settings
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans SC', 'WenQuanYi Micro Hei',
                                           'AR PL UMing CN', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Candidate CJK font paths
_CN_FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
]
_CN_FONT_CACHE = {}


def _get_cn_font(size=14):
    """Get a CJK-capable font"""
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
    """Render Unicode text on a BGR numpy image"""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=_get_cn_font(font_size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def put_label(img, text, pos=(10, 24), font_scale=0.65, thickness=2):
    """Add label"""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def fig_to_bgr(fig, target_w, target_h):
    """Convert a matplotlib figure to a BGR image"""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    if (w, h) != (target_w, target_h):
        bgr = cv2.resize(bgr, (target_w, target_h))
    return bgr


def render_pressure_panel(grid_l, grid_r, w=PANEL_W, h=PANEL_H, vmax=1.0, global_max=None):
    """Render pressure heatmap panel"""
    dpi = 100
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.patch.set_facecolor('#1a1a2e')
    PRESSURE_CMAP.set_bad(color='#111122')

    panels = [
        (fig.add_subplot(1, 2, 1), grid_l, 'Left-Hand Pressure'),
        (fig.add_subplot(1, 2, 2), grid_r, 'Right-Hand Pressure'),
    ]
    
    for ax, grid, title in panels:
        ax.set_facecolor('#111122')
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap=PRESSURE_CMAP, vmin=0, vmax=vmax,
                       interpolation='nearest', aspect='auto', origin='upper')
        ax.set_title(title, fontsize=9, color='#cccccc', pad=4)
        ax.axis('off')
        
        label_str = f'Pressure value (0-{vmax:.2f})' if vmax <= 1.0 else f'Pressure value (0-{vmax:.0f})'
        cbar = fig.colorbar(im, ax=ax, label=label_str,
                            shrink=0.7, fraction=0.04)
        cbar.ax.tick_params(labelsize=6, colors='#aaaaaa')
        cbar.ax.yaxis.label.set_size(7)
        cbar.ax.yaxis.label.set_color('#aaaaaa')
        
        valid = grid[~np.isnan(grid)] if np.any(np.isnan(grid)) else grid.ravel()
        peak_thresh = vmax * 0.05
        if len(valid) > 0 and valid.max() > peak_thresh:
            pos = np.unravel_index(np.nanargmax(grid), grid.shape)
            ax.plot(pos[1], pos[0], 'w+', markersize=7,
                    markeredgewidth=1.5, alpha=0.9)

    fmt = '.2f' if (global_max is not None and global_max <= 1.0) else '.1f'
    footer = f'baseline-corrected  |  max={global_max:{fmt}}'
    if global_max is not None:
        fig.text(0.98, 0.02, footer,
                 ha='right', va='bottom', fontsize=7, color='#888888')

    fig.subplots_adjust(left=0.03, right=0.92, top=0.92, bottom=0.05, wspace=0.15)
    img = fig_to_bgr(fig, w, h)
    plt.close(fig)
    return img


def visualize_trajectory(traj_dir, output_path=None, fps=30, force=False, max_frames=None):
    """
    Generate a visualization video for a single trajectory
    
    Args:
        traj_dir: trajectory directory path
        output_path: output video path; defaults to visualization.mp4 under the trajectory directory
        fps: video frame rate
        force: whether to force regeneration
        max_frames: maximum number of frames to visualize; None means all frames
    """
    traj_dir = Path(traj_dir)
    
    # If no output path is specified, save under the trajectory directory
    if output_path is None:
        output_path = traj_dir / 'visualization.mp4'
    else:
        output_path = Path(output_path)
    
    # Check whether generation is needed
    if not force and output_path.exists():
        return True, "already exists"
    
    # Check required files
    chest_video = traj_dir / 'chest.mp4'
    left_video = traj_dir / 'left.mp4'
    right_video = traj_dir / 'right.mp4'
    
    if not all([chest_video.exists(), left_video.exists(), right_video.exists()]):
        return False, "missing video files"
    
    # Open video files
    cap_chest = cv2.VideoCapture(str(chest_video))
    cap_left = cv2.VideoCapture(str(left_video))
    cap_right = cv2.VideoCapture(str(right_video))
    
    if not all([cap_chest.isOpened(), cap_left.isOpened(), cap_right.isOpened()]):
        return False, "failed to open video"
    
    total_frames = int(cap_chest.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(total_frames, max_frames) if max_frames else total_frames
    
    # Check wilor_annotated.mp4
    wilor_video_path = traj_dir / 'wilor_annotated.mp4'
    has_wilor = wilor_video_path.exists()
    cap_wilor = cv2.VideoCapture(str(wilor_video_path)) if has_wilor else None
    
    # Load pressure data
    pressure_grids_file = traj_dir / 'pressure_grids.npz'
    has_pressure = False
    left_grids = None
    right_grids = None
    global_max = 1.0
    
    if pressure_grids_file.exists():
        try:
            data = np.load(pressure_grids_file)
            left_grids = data['left_pressure_grid']
            right_grids = data['right_pressure_grid']
            has_pressure = True
            # Separately normalized values are in [0, 1], so use 1.0 as the display range
            global_max = 1.0
        except Exception as e:
            print(f"  Warning: failed to load pressure data: {e}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Encode with ffmpeg
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{VIDEO_W}x{VIDEO_H}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264', '-preset', 'medium',
        '-crf', '23', '-pix_fmt', 'yuv420p', str(output_path)
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                                   stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    
    # Process frame by frame
    for fi in range(num_frames):
        # Read the three camera streams
        ret_chest, frame_chest = cap_chest.read()
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not all([ret_chest, ret_left, ret_right]):
            break
        
        # Resize and add labels
        frame_chest = cv2.resize(frame_chest, (CAM_W, CAM_H))
        frame_left = cv2.resize(frame_left, (CAM_W, CAM_H))
        frame_right = cv2.resize(frame_right, (CAM_W, CAM_H))
        
        put_label(frame_chest, 'Head')
        put_label(frame_left, 'Left Wrist')
        put_label(frame_right, 'Right Wrist')
        
        row_top = np.hstack([frame_chest, frame_left, frame_right])
        
        # Bottom-left: WiLoR pose panel
        if has_wilor and cap_wilor.isOpened():
            ret_wilor, frame_wilor = cap_wilor.read()
            if ret_wilor:
                wilor_panel = cv2.resize(frame_wilor, (PANEL_W, PANEL_H))
            else:
                wilor_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                put_text_cn(wilor_panel, 'Pose data unavailable', (PANEL_W//2 - 80, PANEL_H//2), 
                           font_size=20, color=(100, 100, 100))
        else:
            wilor_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            put_text_cn(wilor_panel, 'Pose data not generated', (PANEL_W//2 - 80, PANEL_H//2), 
                       font_size=20, color=(100, 100, 100))
        
        # Bottom-right: pressure panel
        if has_pressure and fi < len(left_grids):
            grid_l = left_grids[fi]
            grid_r = right_grids[fi]
            pressure_panel = render_pressure_panel(grid_l, grid_r, vmax=global_max, global_max=global_max)
        else:
            pressure_panel = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
            put_text_cn(pressure_panel, 'Pressure data not generated', (PANEL_W//2 - 80, PANEL_H//2), 
                       font_size=20, color=(100, 100, 100))
        
        row_bottom = np.hstack([wilor_panel, pressure_panel])
        
        # Merge frames
        frame_out = np.vstack([row_top, row_bottom])
        
        # Add information line
        scene = traj_dir.parent.parent.name
        task = traj_dir.parent.name
        traj_id = traj_dir.name
        frame_info = f"{fi+1:03d}/{total_frames}" if max_frames else f"{fi+1:03d}/{num_frames}"
        info_line = f"Frame {frame_info}   [{scene}/{task}/{traj_id}]"
        if max_frames and fi+1 >= max_frames:
            info_line += f"  (first {max_frames} frames)"
        frame_out = put_text_cn(frame_out, info_line, pos=(10, VIDEO_H - 22), 
                               font_size=14, color=(180, 180, 180))
        
        ffmpeg_proc.stdin.write(frame_out.tobytes())
    
    # Cleanup
    cap_chest.release()
    cap_left.release()
    cap_right.release()
    if cap_wilor:
        cap_wilor.release()
    
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    
    return True, f"success ({num_frames} frames)"


def _vis_worker(args):
    """ProcessPoolExecutor worker function"""
    traj_dir, output_path, fps, force, max_frames = args
    try:
        success, msg = visualize_trajectory(traj_dir, output_path, fps=fps, force=force, max_frames=max_frames)
        return (str(traj_dir), success, msg)
    except Exception as e:
        import traceback
        return (str(traj_dir), False, traceback.format_exc())


def batch_visualize(root_dir, workers=4, fps=30, force=False, max_frames=None):
    """
    Batch-visualize the full dataset (videos are saved under each trajectory folder)
    
    Args:
        root_dir: cleaned dataset root directory
        workers: number of worker processes
        fps: video frame rate
        force: whether to force regeneration
        max_frames: maximum number of frames to visualize; None means all frames
    """
    root_dir = Path(root_dir)
    
    # Scan all trajectories
    all_trajs = []
    for scene_dir in root_dir.iterdir():
        if not scene_dir.is_dir() or scene_dir.name.startswith('.'):
            continue
        for task_dir in scene_dir.iterdir():
            if not task_dir.is_dir():
                continue
            for traj_dir in task_dir.iterdir():
                if not traj_dir.is_dir():
                    continue
                # Check for chest.mp4
                if (traj_dir / 'chest.mp4').exists():
                    all_trajs.append(traj_dir)
    
    if not all_trajs:
        print(f"Error: no valid trajectories found: {root_dir}")
        return
    
    print(f"=" * 80)
    print(f"Batch visualization for cleaned dataset")
    print(f"=" * 80)
    print(f"Data directory: {root_dir}")
    print(f"Output location: visualization.mp4 under each trajectory folder")
    print(f"Total trajectories: {len(all_trajs)}")
    print(f"Worker processes: {workers}")
    print(f"Force regeneration: {force}")
    if max_frames:
        print(f"Maximum frames to visualize: {max_frames}")
    print(f"=" * 80 + "\n")
    
    # Build task list; output path is None to use the default path
    tasks = []
    for traj_dir in all_trajs:
        tasks.append((str(traj_dir), None, fps, force, max_frames))
    
    success, skip, fail = 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_vis_worker, t): t for t in tasks}
        pbar = tqdm(as_completed(futures), total=len(futures), desc="Visualization progress")
        for future in pbar:
            traj_path_str, ok, msg = future.result()
            name = Path(traj_path_str).name
            if ok:
                if msg == "already exists":
                    skip += 1
                else:
                    success += 1
                pbar.set_postfix(ok=success, skip=skip, fail=fail, last=name)
            else:
                fail += 1
                tqdm.write(f"✗ failed: {name} - {msg}")
    
    print(f"\n" + "=" * 80)
    print(f"Complete - success: {success}  skipped: {skip}  failed: {fail}")
    print(f"=" * 80)


def main():
    parser = argparse.ArgumentParser(description='TouchAnything cleaned-data visualization')
    parser.add_argument('--traj', type=str, default=None,
                        help='single trajectory directory path')
    parser.add_argument('--root', type=str, default=None,
                        help='dataset root directory for batch mode')
    parser.add_argument('--output', type=str, default=None,
                        help='output video path, single-trajectory mode only; defaults to visualization.mp4 under the trajectory directory')
    parser.add_argument('--batch', action='store_true',
                        help='batch mode: visualize the full dataset')
    parser.add_argument('--fps', type=int, default=30,
                        help='output frame rate, default 30')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of worker processes for batch mode, default 4')
    parser.add_argument('--force', action='store_true',
                        help='force regeneration of all visualizations')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='maximum number of frames to visualize for quick preview; default all')
    args = parser.parse_args()
    
    if args.batch:
        root = args.root or 'datasets/TouchAnything_Datasets_opensource'
        batch_visualize(root, workers=args.workers, fps=args.fps, force=args.force, max_frames=args.max_frames)
    else:
        if args.traj is None:
            print("Error: please specify --traj or use --batch mode")
            return
        
        traj_dir = Path(args.traj)
        if args.output is None:
            out = traj_dir / 'visualization.mp4'
        else:
            out = Path(args.output)
        
        success, msg = visualize_trajectory(traj_dir, out, fps=args.fps, force=args.force, max_frames=args.max_frames)
        if success:
            print(f"✓ {msg}: {out}")
        else:
            print(f"✗ failed: {msg}")


if __name__ == '__main__':
    main()
