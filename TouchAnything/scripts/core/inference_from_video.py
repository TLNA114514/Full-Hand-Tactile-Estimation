#!/usr/bin/env python3
"""
Infer tactile pressure maps directly from an MP4 video.

- Supports a single egocentric video input
- Uses HaMeR to estimate hand poses automatically
- Produces a tactile-pressure prediction video
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
import tempfile
import shutil

# Add the project root and HaMeR paths.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    # Keep the repo on sys.path so `third_party.*` imports work, while
    # avoiding higher precedence than site-packages for vendored deps.
    sys.path.append(str(PROJECT_ROOT))


def _resolve_hamer_dir() -> Path:
    candidates = [
        PROJECT_ROOT / 'third_party' / 'hamer',
        PROJECT_ROOT / 'hamer',  # backward compatibility
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


HAMER_DIR = _resolve_hamer_dir()
sys.path.insert(0, str(HAMER_DIR))

from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full
from third_party.hamer.vitpose_model import ViTPoseModel

from src.models import build_model
from src.utils.config import load_config_with_base
from src.data.transforms import VideoTransform
from src.utils.vis_pressure import (render_pressure_panel, PRESS_W, PRESS_H, pred_to_21,
                                    SENSOR_NAN_MASK_L, SENSOR_NAN_MASK_R)

# Video layout constants.
CAM_W, CAM_H = 480, 270
VIDEO_W, VIDEO_H = 1440, 630


def detect_both_hands_boxes(img, detector, cpm):
    """Detect left- and right-hand bounding boxes in an image."""
    det_out = detector(img)
    img_rgb = img[:, :, ::-1].copy()

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)

    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    if len(pred_bboxes) == 0:
        return None, None

    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right_list = []

    for vitposes in vitposes_out:
        keypoints = vitposes['keypoints']

        left_hand_keyp = keypoints[-42:-21]
        left_valid = left_hand_keyp[:, 2] > 0.5
        if np.sum(left_valid) > 3:
            left_bbox = [
                left_hand_keyp[left_valid, 0].min(),
                left_hand_keyp[left_valid, 1].min(),
                left_hand_keyp[left_valid, 0].max(),
                left_hand_keyp[left_valid, 1].max()
            ]
            bboxes.append(left_bbox)
            is_right_list.append(0)

        right_hand_keyp = keypoints[-21:]
        right_valid = right_hand_keyp[:, 2] > 0.5
        if np.sum(right_valid) > 3:
            right_bbox = [
                right_hand_keyp[right_valid, 0].min(),
                right_hand_keyp[right_valid, 1].min(),
                right_hand_keyp[right_valid, 0].max(),
                right_hand_keyp[right_valid, 1].max()
            ]
            bboxes.append(right_bbox)
            is_right_list.append(1)

    if len(bboxes) == 0:
        return None, None

    return np.array(bboxes), np.array(is_right_list)


def project_points(points_3d, focal_length, img_w, img_h):
    """Project 3D points into 2D image coordinates."""
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    eps = 1e-9
    
    u = focal_length * (x / (z + eps)) + img_w / 2.0
    v = focal_length * (y / (z + eps)) + img_h / 2.0
    return np.stack([u, v], axis=1)


def extract_hamer_poses_from_video(
    video_path,
    detector,
    cpm,
    hamer_model,
    hamer_cfg,
    device,
    batch_size=8,
    max_frames=None
):
    """
    Extract HaMeR hand poses from a video.

    Returns:
        poses: list of dicts with keys ``left_pos`` and ``right_pos`` (21x3 arrays)
        frames: list of RGB frames with shape (H, W, 3)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    poses = []
    frames = []
    
    print(f"Extracting hand poses from video: {total_frames} frames")
    
    with tqdm(total=total_frames, desc="Pose extraction") as pbar:
        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            img_h, img_w = frame.shape[:2]
            left_pos = None
            right_pos = None
            
            # Detect hand bounding boxes.
            boxes, right = detect_both_hands_boxes(frame, detector, cpm)
            
            if boxes is not None:
                dataset = ViTDetDataset(
                    hamer_cfg,
                    frame,
                    boxes,
                    right,
                    rescale_factor=2.0
                )
                
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )
                
                for batch in dataloader:
                    batch = recursive_to(batch, device)
                    
                    with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                        out = hamer_model(batch)
                    
                    pred_joints_3d = out['pred_keypoints_3d']
                    multiplier = (2 * batch['right'] - 1)
                    pred_cam = out['pred_cam'].clone()
                    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                    
                    box_center = batch["box_center"].float()
                    box_size = batch["box_size"].float()
                    img_size = batch["img_size"].float()
                    
                    scaled_focal_length = (
                        hamer_cfg.EXTRA.FOCAL_LENGTH / hamer_cfg.MODEL.IMAGE_SIZE * img_size.max()
                    )
                    
                    pred_cam_t_full = cam_crop_to_full(
                        pred_cam,
                        box_center,
                        box_size,
                        img_size,
                        scaled_focal_length
                    ).detach().cpu().numpy()
                    
                    bs = batch['img'].shape[0]
                    for n in range(bs):
                        joints_3d = pred_joints_3d[n].detach().cpu().numpy()
                        is_right_hand = int(batch['right'][n].detach().cpu().item())
                        
                        joints_3d[:, 0] = (2 * is_right_hand - 1) * joints_3d[:, 0]
                        joints_3d_cam = joints_3d + pred_cam_t_full[n][None, :]
                        
                        if is_right_hand:
                            right_pos = joints_3d_cam
                        else:
                            left_pos = joints_3d_cam
            
            poses.append({
                'left_pos': left_pos if left_pos is not None else np.zeros((21, 3)),
                'right_pos': right_pos if right_pos is not None else np.zeros((21, 3))
            })
            
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    return poses, frames


def infer_tactile_from_video(
    video_path,
    model,
    device,
    cfg,
    hamer_model,
    hamer_cfg,
    detector,
    cpm,
    clip_length=8,
    frame_interval=2,  # Must match training config
    batch_size=16,
    hamer_batch_size=8,
    max_frames=None,
):
    """
    Infer tactile pressure maps from a video.

    Returns:
        frames: (T, H, W, 3) RGB frames
        pred_maps: (T, 2, S, S) predicted pressure maps
    """
    # 1. Extract HaMeR poses.
    poses, frames = extract_hamer_poses_from_video(
        video_path,
        detector,
        cpm,
        hamer_model,
        hamer_cfg,
        device,
        batch_size=hamer_batch_size,
        max_frames=max_frames,
    )
    
    T = len(frames)
    print(f"Extracted {T} frames and pose estimates")
    
    # 2. Prepare inputs for tactile inference.
    image_size = cfg['data']['image_size']
    transform = VideoTransform(
        image_size=image_size,
        normalize_mean=cfg['data'].get('normalize_mean', [0.485, 0.456, 0.406]),
        normalize_std=cfg['data'].get('normalize_std', [0.229, 0.224, 0.225]),
        use_augmentation=False,
    )
    
    # Build clips with dense sliding window + averaging (aligned with batch inference)
    span = (clip_length - 1) * frame_interval + 1
    tactile_size = cfg['data'].get('tactile_size', 21)
    
    # Use dense sliding window (step=1) for robust predictions
    clip_starts = list(range(0, T - span + 1))
    num_clips = len(clip_starts)
    
    # Accumulate predictions for averaging
    pred_maps_sum = np.zeros((T, 2, tactile_size, tactile_size), dtype=np.float32)
    pred_counts = np.zeros(T, dtype=np.int32)
    
    print(f"Tactile inference: {num_clips} clips (span={span}, interval={frame_interval}, dense sliding)")
    
    with torch.no_grad():
        for batch_start in tqdm(range(0, num_clips, batch_size), desc="Tactile inference"):
            batch_clip_starts = clip_starts[batch_start:batch_start + batch_size]
            batch_frames = []
            batch_poses = []
            batch_indices = []

            for start_idx in batch_clip_starts:
                frame_indices = [start_idx + i * frame_interval for i in range(clip_length)]
                clip_frames = np.stack([frames[i] for i in frame_indices])
                batch_frames.append(transform(clip_frames))

                left_poses = np.stack([poses[i]['left_pos'] for i in frame_indices])
                right_poses = np.stack([poses[i]['right_pos'] for i in frame_indices])
                left_valid = np.array([poses[i].get('left_valid', True) for i in frame_indices])
                right_valid = np.array([poses[i].get('right_valid', True) for i in frame_indices])
                poses_np = np.concatenate([left_poses, right_poses], axis=1).astype(np.float32)

                # Mark invalid poses with special value (consistent with training)
                INVALID_POSE_VALUE = -10.0
                for t in range(clip_length):
                    if not left_valid[t]:
                        poses_np[t, :21, :] = INVALID_POSE_VALUE
                    if not right_valid[t]:
                        poses_np[t, 21:, :] = INVALID_POSE_VALUE

                # Clip abnormal pose coordinates (aligned with training data processing)
                poses_np[:, :, 0] = np.clip(poses_np[:, :, 0], -10.0, 10.0)   # x: horizontal offset
                poses_np[:, :, 1] = np.clip(poses_np[:, :, 1], -10.0, 10.0)   # y: vertical offset
                poses_np[:, :, 2] = np.clip(poses_np[:, :, 2], -10.0, 100.0)  # z: depth value

                batch_poses.append(torch.from_numpy(poses_np))
                batch_indices.append(frame_indices)

            frames_t = torch.stack(batch_frames, dim=0).to(device)
            poses_t = torch.stack(batch_poses, dim=0).to(device)

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                outputs = model(frames=frames_t, poses=poses_t)

            pred_pressure = outputs['tactile'].float().cpu().numpy()

            for clip_idx, frame_indices in enumerate(batch_indices):
                for t_idx, frame_idx in enumerate(frame_indices):
                    if frame_idx < T:
                        pred_maps_sum[frame_idx] += pred_pressure[clip_idx, t_idx]
                        pred_counts[frame_idx] += 1
    
    # Average multiple predictions for each frame
    pred_maps_all = np.full((T, 2, tactile_size, tactile_size), np.nan, dtype=np.float32)
    for t in range(T):
        if pred_counts[t] > 0:
            pred_maps_all[t] = pred_maps_sum[t] / pred_counts[t]
    
    print(f"Prediction coverage: {np.sum(pred_counts > 0)}/{T} frames, avg predictions per frame: {pred_counts.mean():.1f}")
    return np.array(frames), pred_maps_all


def make_video_with_prediction(
    frames,
    pred_maps,
    output_path,
    fps=10,
    vmax=1.0
):
    """Generate a video that includes the tactile predictions."""
    import subprocess
    
    T = frames.shape[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{VIDEO_W}x{VIDEO_H}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264', '-preset', 'medium',
        '-crf', '23', '-pix_fmt', 'yuv420p', str(output_path)
    ]
    # Avoid deadlocks by redirecting stderr to DEVNULL so the pipe buffer does not fill up
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    # Use physical sensor NaN masks (True = no sensor at that position)
    nan_mask_l = SENSOR_NAN_MASK_L
    nan_mask_r = SENSOR_NAN_MASK_R
    
    for t in range(T):
        # Prepare the camera image.
        cam_frame = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        cam_frame = cv2.resize(cam_frame, (CAM_W, CAM_H))
        
        # Create the top row (single camera input).
        row_top = np.zeros((CAM_H, VIDEO_W, 3), dtype=np.uint8)
        row_top[:, :CAM_W] = cam_frame
        
        # Create the bottom row (predicted pressure map).
        if t < len(pred_maps) and not np.all(np.isnan(pred_maps[t])):
            pred_l21 = pred_to_21(pred_maps[t, 0], nan_mask_l)
            pred_r21 = pred_to_21(pred_maps[t, 1], nan_mask_r)
            pred_panel = render_pressure_panel(
                pred_l21, pred_r21,
                title='Predicted Pressure',
                w=PRESS_W, h=PRESS_H, vmax=vmax, global_max=vmax
            )
        else:
            pred_panel = np.zeros((PRESS_H, PRESS_W, 3), dtype=np.uint8)
            cv2.putText(pred_panel, 'No Prediction', (PRESS_W//2 - 80, PRESS_H//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Blank panel.
        empty_panel = np.zeros((PRESS_H, PRESS_W, 3), dtype=np.uint8)
        row_bottom = np.hstack([empty_panel, pred_panel])
        
        # Merge panels.
        frame_out = np.vstack([row_top, row_bottom])
        
        # Add overlay text.
        cv2.putText(frame_out, f'Frame {t+1}/{T}',
                   (10, VIDEO_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(frame_out, 'Video Input',
                   (VIDEO_W - 120, 20), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 200, 120), 2, cv2.LINE_AA)
        
        ffmpeg_proc.stdin.write(frame_out.tobytes())
    
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f"Saved video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Infer tactile pressure maps from an MP4 video')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video path (MP4)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Tactile-model checkpoint path')
    parser.add_argument('--config', type=str, required=True,
                       help='Tactile-model config path')
    parser.add_argument('--hamer_checkpoint', type=str,
                       default=str(HAMER_DIR / '_DATA' / 'hamer_ckpts' / 'checkpoints' / 'hamer.ckpt'),
                       help='HaMeR checkpoint path')
    parser.add_argument('--output', type=str, default='output_tactile.mp4',
                       help='Output video path')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for tactile inference')
    parser.add_argument('--hamer_batch_size', type=int, default=8,
                       help='Batch size for HaMeR inference')
    parser.add_argument('--fps', type=int, default=10,
                       help='Output video frame rate')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process (for testing)')
    
    args = parser.parse_args()

    if not HAMER_DIR.exists():
        raise FileNotFoundError(
            f"HaMeR directory not found. Expected one of: "
            f"{PROJECT_ROOT / 'third_party' / 'hamer'} or {PROJECT_ROOT / 'hamer'}"
        )
    
    # Switch to the HaMeR directory to avoid chumpy circular-import issues.
    original_cwd = os.getcwd()
    os.chdir(HAMER_DIR)
    
    # Resolve relative paths before changing execution context.
    video_path = Path(original_cwd) / args.video if not Path(args.video).is_absolute() else Path(args.video)
    checkpoint_path = Path(original_cwd) / args.checkpoint if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    config_path = Path(original_cwd) / args.config if not Path(args.config).is_absolute() else Path(args.config)
    hamer_checkpoint_path = (
        Path(original_cwd) / args.hamer_checkpoint
        if not Path(args.hamer_checkpoint).is_absolute()
        else Path(args.hamer_checkpoint)
    )
    output_path = Path(original_cwd) / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    args.video = str(video_path)
    args.checkpoint = str(checkpoint_path)
    args.config = str(config_path)
    args.hamer_checkpoint = str(hamer_checkpoint_path)
    args.output = str(output_path)
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the tactile model.
    print("Loading tactile prediction model...")
    cfg = load_config_with_base(args.config)
    vision_cfg = cfg.get('model', {}).get('vision_encoder', {})
    pretrained_path = vision_cfg.get('pretrained_path')
    if pretrained_path and not Path(pretrained_path).is_absolute():
        resolved_pretrained = (PROJECT_ROOT / pretrained_path).resolve()
        vision_cfg['pretrained_path'] = str(resolved_pretrained)
    model = build_model(cfg)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print("✓ Tactile model loaded")
    
    # 2. Load the HaMeR model.
    print("Loading HaMeR pose-estimation model...")
    project_related = {
        str(PROJECT_ROOT.resolve()),
        str((PROJECT_ROOT / 'scripts').resolve()),
    }
    original_sys_path = list(sys.path)
    sys.path = [
        p for p in sys.path
        if (p and str(Path(p).resolve()) not in project_related)
    ]
    hamer_model, hamer_cfg = load_hamer(args.hamer_checkpoint, init_renderer=False)
    sys.path = original_sys_path
    hamer_model = hamer_model.to(device)
    hamer_model.eval()
    print("✓ HaMeR model loaded")
    
    # 3. Load the detector.
    print("Loading hand detector...")
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    
    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    local_detectron2_ckpt = HAMER_DIR / '_DATA' / 'detectron2_models' / 'model_final_f05665.pkl'
    if local_detectron2_ckpt.exists():
        detectron2_cfg.train.init_checkpoint = str(local_detectron2_ckpt)
        print(f"Using local Detectron2 weights: {local_detectron2_ckpt}")
    else:
        raise FileNotFoundError(
            "Local Detectron2 weights not found. Expected: "
            f"{local_detectron2_ckpt}. Refusing to fall back to an implicit online download."
        )
    
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    print("✓ Detector loaded")
    
    # 4. Load ViTPose.
    print("Loading ViTPose keypoint detector...")
    cpm = ViTPoseModel(device)
    print("✓ ViTPose loaded")
    
    # 5. Process the video.
    print(f"\nProcessing video: {args.video}")
    frames, pred_maps = infer_tactile_from_video(
        args.video,
        model,
        device,
        cfg,
        hamer_model,
        hamer_cfg,
        detector,
        cpm,
        clip_length=cfg['data'].get('clip_length', 8),
        frame_interval=cfg['data'].get('frame_interval', 2),  # Default matches training config
        batch_size=args.batch_size,
        hamer_batch_size=args.hamer_batch_size,
        max_frames=args.max_frames,
    )
    
    # 6. Render the output video.
    print(f"\nRendering output video...")
    make_video_with_prediction(
        frames,
        pred_maps,
        args.output,
        fps=args.fps,
        vmax=1.0
    )
    
    print(f"\n✓ Done. Output video: {args.output}")


if __name__ == '__main__':
    main()
