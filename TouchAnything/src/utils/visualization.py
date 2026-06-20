"""Visualization utilities for hand pose predictions.
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional


# Per-finger color scheme (BGR) matching EgoDex official style
FINGER_COLORS_BGR = {
    'thumb':  (238, 130, 238),  # violet
    'index':  (71, 99, 255),    # tomato (BGR)
    'middle': (250, 245, 230),  # pale turquoise (BGR)
    'ring':   (47, 255, 173),   # green-yellow (BGR)
    'little': (191, 152, 0),    # light blue (BGR)
}

# Finger joint index ranges within 24-joint hand
# Thumb: joints 0-3, Index: 4-8, Middle: 9-13, Ring: 14-18, Little: 19-23
FINGER_CONNECTIONS = {
    'thumb':  [(0, 1), (1, 2), (2, 3)],
    'index':  [(4, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(9, 10), (10, 11), (11, 12), (12, 13)],
    'ring':   [(14, 15), (15, 16), (16, 17), (17, 18)],
    'little': [(19, 20), (20, 21), (21, 22), (22, 23)],
}

# Finger joint indices (for coloring each joint by finger)
FINGER_JOINT_INDICES = {
    'thumb':  [0, 1, 2, 3],
    'index':  [4, 5, 6, 7, 8],
    'middle': [9, 10, 11, 12, 13],
    'ring':   [14, 15, 16, 17, 18],
    'little': [19, 20, 21, 22, 23],
}


def project_3d_to_2d(points_3d: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates using camera intrinsics.
    
    Args:
        points_3d: (N, 3) array of 3D points in camera frame
        intrinsic: (3, 3) camera intrinsic matrix
    
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
    """
    # Homogeneous coordinates
    points_2d_homo = intrinsic @ points_3d.T  # (3, N)
    
    # Normalize by depth, avoid division by zero
    depth = points_2d_homo[2:3]  # (1, N)
    depth = np.where(depth > 1e-6, depth, 1e-6)
    points_2d = points_2d_homo[:2] / depth  # (2, N)
    
    return points_2d.T  # (N, 2)


def _pt_in_bounds(pt, W, H, margin=50):
    """Check if a 2D point is roughly within image bounds."""
    return -margin <= pt[0] < W + margin and -margin <= pt[1] < H + margin


def draw_hand_skeleton(
    image: np.ndarray,
    joints_2d: np.ndarray,
    thickness: int = 4,
    radius: int = 8,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Draw hand skeleton with per-finger colors (reference EgoDex style).
    
    Args:
        image: (H, W, 3) BGR image
        joints_2d: (24, 2) array of 2D joint coordinates
        thickness: line thickness
        radius: joint circle radius
        alpha: overlay opacity (1.0 = opaque)
    
    Returns:
        image: image with skeleton drawn
    """
    H, W = image.shape[:2]
    overlay = image.copy()
    
    # Draw bones per finger (back to front: little -> thumb so thumb is on top)
    for finger in ['little', 'ring', 'middle', 'index', 'thumb']:
        color = FINGER_COLORS_BGR[finger]
        for start_idx, end_idx in FINGER_CONNECTIONS[finger]:
            pt1 = joints_2d[start_idx]
            pt2 = joints_2d[end_idx]
            if _pt_in_bounds(pt1, W, H) and _pt_in_bounds(pt2, W, H):
                cv2.line(overlay, tuple(pt1.astype(int)), tuple(pt2.astype(int)),
                         color, thickness, cv2.LINE_AA)
    
    # Draw joints per finger (on top of bones)
    for finger in ['little', 'ring', 'middle', 'index', 'thumb']:
        color = FINGER_COLORS_BGR[finger]
        for idx in FINGER_JOINT_INDICES[finger]:
            pt = joints_2d[idx]
            if _pt_in_bounds(pt, W, H):
                pt_int = tuple(pt.astype(int))
                # Dark outline for contrast
                cv2.circle(overlay, pt_int, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
                # Colored fill
                cv2.circle(overlay, pt_int, radius, color, -1, cv2.LINE_AA)
    
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        return image
    return overlay


def draw_hand_skeleton_gt(
    image: np.ndarray,
    joints_2d: np.ndarray,
    thickness: int = 3,
    radius: int = 6,
) -> np.ndarray:
    """
    Draw ground truth hand skeleton with distinct dashed-style and red tones.
    Uses hollow circles + thinner lines to distinguish from predictions.
    """
    H, W = image.shape[:2]
    overlay = image.copy()
    color = (80, 80, 255)  # light red
    
    for finger in ['little', 'ring', 'middle', 'index', 'thumb']:
        for start_idx, end_idx in FINGER_CONNECTIONS[finger]:
            pt1 = joints_2d[start_idx]
            pt2 = joints_2d[end_idx]
            if _pt_in_bounds(pt1, W, H) and _pt_in_bounds(pt2, W, H):
                cv2.line(overlay, tuple(pt1.astype(int)), tuple(pt2.astype(int)),
                         color, thickness, cv2.LINE_AA)
    
    for finger in ['little', 'ring', 'middle', 'index', 'thumb']:
        for idx in FINGER_JOINT_INDICES[finger]:
            pt = joints_2d[idx]
            if _pt_in_bounds(pt, W, H):
                pt_int = tuple(pt.astype(int))
                # Hollow circle (ring) for GT
                cv2.circle(overlay, pt_int, radius, color, 2, cv2.LINE_AA)
    
    return overlay


def draw_legend(image: np.ndarray, has_gt: bool = False) -> np.ndarray:
    """Draw a semi-transparent legend panel in the top-left corner."""
    H, W = image.shape[:2]
    overlay = image.copy()
    
    # Panel size
    entries = [('Prediction', FINGER_COLORS_BGR)]
    if has_gt:
        entries.append(('Ground Truth', None))
    
    panel_h = 40 + len(entries) * 35 + 10
    panel_w = 280
    
    # Semi-transparent dark background
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, overlay)
    
    y = 35
    for label, colors in entries:
        if colors is not None:
            # Prediction legend: show finger color dots
            cv2.putText(overlay, label, (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
            x = 25
            for finger in ['thumb', 'index', 'middle', 'ring', 'little']:
                c = colors[finger]
                cv2.circle(overlay, (x, y - 5), 7, c, -1, cv2.LINE_AA)
                x += 50
            y += 18
        else:
            # GT legend: hollow red circle + text
            cv2.circle(overlay, (25, y - 5), 7, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, label, (42, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 255), 2, cv2.LINE_AA)
            y += 35
    
    return overlay


def visualize_predictions(
    image: np.ndarray,
    pred_joints_3d: np.ndarray,
    gt_joints_3d: Optional[np.ndarray] = None,
    intrinsic: Optional[np.ndarray] = None,
    show_left: bool = True,
    show_right: bool = True
) -> np.ndarray:
    """
    Visualize predicted (and optionally ground truth) hand poses on image.
    
    Args:
        image: (H, W, 3) BGR image
        pred_joints_3d: (48, 3) predicted 3D joints in camera frame
        gt_joints_3d: (48, 3) optional ground truth 3D joints
        intrinsic: (3, 3) camera intrinsic matrix
        show_left: whether to show left hand
        show_right: whether to show right hand
    
    Returns:
        vis_image: image with visualizations
    """
    if intrinsic is None:
        # Use EgoDex default intrinsics
        intrinsic = np.array([
            [736.6339, 0., 960.],
            [0., 736.6339, 540.],
            [0., 0., 1.]
        ], dtype=np.float32)
    
    vis_image = image.copy()
    
    # Split into left and right hands (24 joints each)
    pred_left = pred_joints_3d[:24]
    pred_right = pred_joints_3d[24:]
    
    # Draw ground truth first (underneath predictions)
    if gt_joints_3d is not None:
        gt_left = gt_joints_3d[:24]
        gt_right = gt_joints_3d[24:]
        
        if show_left:
            gt_left_2d = project_3d_to_2d(gt_left, intrinsic)
            vis_image = draw_hand_skeleton_gt(vis_image, gt_left_2d)
        if show_right:
            gt_right_2d = project_3d_to_2d(gt_right, intrinsic)
            vis_image = draw_hand_skeleton_gt(vis_image, gt_right_2d)
    
    # Draw predictions on top
    if show_left:
        pred_left_2d = project_3d_to_2d(pred_left, intrinsic)
        vis_image = draw_hand_skeleton(vis_image, pred_left_2d)
    if show_right:
        pred_right_2d = project_3d_to_2d(pred_right, intrinsic)
        vis_image = draw_hand_skeleton(vis_image, pred_right_2d)
    
    return vis_image


def create_video_with_predictions(
    video_path: str,
    predictions: np.ndarray,
    output_path: str,
    ground_truth: Optional[np.ndarray] = None,
    intrinsic: Optional[np.ndarray] = None,
    fps: Optional[int] = None,
    show_left: bool = True,
    show_right: bool = True
):
    """
    Create a video with hand pose predictions overlaid.
    
    Args:
        video_path: path to input video
        predictions: (T, 48, 3) predicted 3D joints
        output_path: path to save output video
        ground_truth: (T, 48, 3) optional ground truth joints
        intrinsic: (3, 3) camera intrinsic matrix
        fps: output video FPS (if None, use input video FPS)
        show_left: whether to show left hand
        show_right: whether to show right hand
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    if fps is None:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = len(predictions)
    frame_idx = 0
    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get predictions for this frame
        pred_joints = predictions[frame_idx]
        gt_joints = ground_truth[frame_idx] if ground_truth is not None else None
        
        # Visualize
        vis_frame = visualize_predictions(
            frame,
            pred_joints,
            gt_joints,
            intrinsic,
            show_left,
            show_right
        )
        
        # Add legend panel
        vis_frame = draw_legend(vis_frame, has_gt=(ground_truth is not None))
        
        # Frame counter (bottom-left, with shadow for readability)
        frame_text = f"Frame {frame_idx + 1}/{total_frames}"
        text_y = height - 18
        cv2.putText(vis_frame, frame_text, (12, text_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis_frame, frame_text, (12, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        out.write(vis_frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"Visualization saved to: {output_path}")
