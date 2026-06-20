"""
Pseudo pressure map generation from hand pose data.
Used as proxy target when real tactile data is unavailable.

The 16x16 grid maps anatomically to the hand surface:
- Each cell corresponds to a fixed location on the palm/fingers
- Fingers are arranged as columns, palm fills the lower region
- Pressure is derived from joint flexion (how curled each finger is)

Hand layout on 16x16 grid (row 0 = top = fingertips):
  col:  0-1   3-4   5-7   9-10  12-13
        Thumb Index Middle Ring  Little
  row 0-3:   fingertip segments
  row 4-7:   intermediate segments
  row 8-11:  knuckle / metacarpal segments
  row 12-15: palm region
"""
import numpy as np
import torch


# ============================================================
# Anatomical hand layout: joint index → (row, col) on 16x16 grid
# ============================================================
# EgoDex 24-joint layout per hand:
#   Thumb:  0=Knuckle, 1=IntermBase, 2=IntermTip, 3=Tip
#   Index:  4=Meta, 5=Knuckle, 6=IntermBase, 7=IntermTip, 8=Tip
#   Middle: 9=Meta, 10=Knuckle, 11=IntermBase, 12=IntermTip, 13=Tip
#   Ring:   14=Meta, 15=Knuckle, 16=IntermBase, 17=IntermTip, 18=Tip
#   Little: 19=Meta, 20=Knuckle, 21=IntermBase, 22=IntermTip, 23=Tip

# Each joint maps to a center (row, col) on the 16x16 grid
# Organized anatomically: fingertips at top, palm at bottom
#
# Finger columns (2 cols each, with 1-col gaps):
#   Thumb:  cols 0-1
#   [gap]:  col 2
#   Index:  cols 3-4
#   [gap]:  col 5
#   Middle: cols 6-7
#   [gap]:  col 8
#   Ring:   cols 9-10
#   [gap]:  col 11
#   Little: cols 12-13
#   [edge]: cols 14-15
JOINT_GRID_POS = {
    # Thumb (cols 0-1, slightly angled)
    3:  (1.5, 0.5),   # Tip
    2:  (3.5, 0.8),   # IntermediateTip
    1:  (6.0, 1.0),   # IntermediateBase
    0:  (8.5, 1.2),   # Knuckle
    # Index (cols 3-4)
    8:  (0.5, 3.5),   # Tip
    7:  (2.5, 3.5),   # IntermediateTip
    6:  (5.0, 3.5),   # IntermediateBase
    5:  (7.5, 3.5),   # Knuckle
    4:  (10.0, 3.5),  # Metacarpal
    # Middle (cols 6-7)
    13: (0.5, 6.5),   # Tip
    12: (2.5, 6.5),   # IntermediateTip
    11: (5.0, 6.5),   # IntermediateBase
    10: (7.5, 6.5),   # Knuckle
    9:  (10.0, 6.5),  # Metacarpal
    # Ring (cols 9-10)
    18: (0.5, 9.5),   # Tip
    17: (2.5, 9.5),   # IntermediateTip
    16: (5.0, 9.5),   # IntermediateBase
    15: (7.5, 9.5),   # Knuckle
    14: (10.0, 9.5),  # Metacarpal
    # Little (cols 12-13)
    23: (1.0, 12.5),  # Tip
    22: (3.0, 12.5),  # IntermediateTip
    21: (5.5, 12.5),  # IntermediateBase
    20: (8.0, 12.5),  # Knuckle
    19: (10.5, 12.5), # Metacarpal
}

# Finger segment pairs: (parent_joint, child_joint)
# Used to compute flexion-based pressure
FINGER_SEGMENTS = {
    'thumb':  [(0, 1), (1, 2), (2, 3)],
    'index':  [(4, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(9, 10), (10, 11), (11, 12), (12, 13)],
    'ring':   [(14, 15), (15, 16), (16, 17), (17, 18)],
    'little': [(19, 20), (20, 21), (21, 22), (22, 23)],
}

# Palm region: extra grid positions not tied to specific joints
# These represent the center of the palm (below finger region)
PALM_CENTERS = [
    (12.0, 3.5), (12.0, 6.5), (12.0, 9.5),     # upper palm
    (13.5, 3.0), (13.5, 6.5), (13.5, 10.0),    # mid palm
    (14.5, 5.0), (14.5, 8.0),                    # lower palm
]


def _hand_mask_16x16() -> np.ndarray:
    """
    Generate a binary mask of hand shape on the 16x16 grid.
    Each finger occupies 2 columns with 1-column gaps between them.
    Returns: (16, 16) bool array, True = hand region
    """
    mask = np.zeros((16, 16), dtype=bool)
    
    # Finger regions (rows 0-10): 2 cols each, 1-col gap
    # Thumb: cols 0-1, rows 0-9 (shorter, angled start)
    for r in range(0, 10):
        mask[r, 0:2] = True
    
    # Index: cols 3-4, rows 0-10
    for r in range(0, 11):
        mask[r, 3:5] = True
    
    # Middle: cols 6-7, rows 0-10
    for r in range(0, 11):
        mask[r, 6:8] = True
    
    # Ring: cols 9-10, rows 0-10
    for r in range(0, 11):
        mask[r, 9:11] = True
    
    # Little: cols 12-13, rows 1-10 (shorter)
    for r in range(1, 11):
        mask[r, 12:14] = True
    
    # Palm region (rows 10-15): fill across all finger bases, tapering at bottom
    for r in range(10, 16):
        width_shrink = (r - 10) * 0.5
        c_lo = int(0 + width_shrink)
        c_hi = int(14 - width_shrink)
        for c in range(c_lo, c_hi + 1):
            mask[r, c] = True
    
    return mask


# Pre-computed hand mask (cached)
HAND_MASK = _hand_mask_16x16()


def _compute_joint_flexion(joints_3d: np.ndarray, parent: int, child: int) -> float:
    """
    Compute flexion (bending) of a finger segment.
    Returns a value in [0, 1]: 0 = straight, 1 = fully bent.
    
    Uses the angle between the bone vector and the reference direction
    (metacarpal-to-knuckle vector extended).
    """
    bone = joints_3d[child] - joints_3d[parent]
    bone_len = np.linalg.norm(bone)
    if bone_len < 1e-8:
        return 0.0
    return 1.0  # default: all segments contribute


def _compute_finger_pressure(joints_3d: np.ndarray) -> dict:
    """
    Compute per-joint pressure from finger flexion.
    
    Strategy: measure how much each finger is curled by comparing
    the fingertip-to-metacarpal distance vs the sum of bone lengths
    (straight finger = ratio~1, curled = ratio < 1).
    
    Returns:
        dict: joint_index → pressure value [0, 1]
    """
    pressures = {}
    
    finger_joints = {
        'thumb':  [0, 1, 2, 3],
        'index':  [4, 5, 6, 7, 8],
        'middle': [9, 10, 11, 12, 13],
        'ring':   [14, 15, 16, 17, 18],
        'little': [19, 20, 21, 22, 23],
    }
    
    for finger, joint_ids in finger_joints.items():
        # Sum of bone lengths (max extension)
        total_bone_len = 0.0
        for i in range(len(joint_ids) - 1):
            bone = joints_3d[joint_ids[i + 1]] - joints_3d[joint_ids[i]]
            total_bone_len += np.linalg.norm(bone)
        
        if total_bone_len < 1e-8:
            for jid in joint_ids:
                pressures[jid] = 0.0
            continue
        
        # Direct distance from base to tip
        base = joints_3d[joint_ids[0]]
        tip = joints_3d[joint_ids[-1]]
        direct_dist = np.linalg.norm(tip - base)
        
        # Curl ratio: 1 = straight (no pressure), 0 = fully curled (max pressure)
        curl_ratio = direct_dist / (total_bone_len + 1e-8)
        curl_ratio = np.clip(curl_ratio, 0.0, 1.0)
        
        # Pressure = 1 - curl_ratio (more curl = more pressure)
        finger_pressure = 1.0 - curl_ratio
        
        # Distribute pressure along finger: tips get more, base gets less
        n = len(joint_ids)
        for k, jid in enumerate(joint_ids):
            # Weight: tip=1.0, base=0.3
            weight = 0.3 + 0.7 * (k / (n - 1))
            pressures[jid] = finger_pressure * weight
    
    return pressures


def generate_pseudo_pressure_map(
    joints_3d: np.ndarray,
    map_size: int = 16,
    sigma: float = 1.2,
) -> np.ndarray:
    """
    Generate an anatomically-mapped pseudo pressure map from 3D hand pose.
    
    The 16x16 grid maps to the hand surface. Each joint has a fixed
    grid position. Pressure is computed from finger curl (flexion).
    
    Args:
        joints_3d: (24, 3) single hand 3D joint positions (meters)
        map_size: output map resolution (default 16)
        sigma: Gaussian blob spread in grid units
    
    Returns:
        pressure: (map_size, map_size) pressure map in [0, 1]
    """
    # Compute per-joint pressure from finger flexion
    joint_pressures = _compute_finger_pressure(joints_3d)
    
    # Generate pressure map: place Gaussian blobs at anatomical positions
    yy, xx = np.mgrid[0:map_size, 0:map_size].astype(np.float32)
    pressure = np.zeros((map_size, map_size), dtype=np.float32)
    
    for jid, (row, col) in JOINT_GRID_POS.items():
        amp = joint_pressures.get(jid, 0.0)
        if amp > 0.01:
            blob = np.exp(-((yy - row)**2 + (xx - col)**2) / (2 * sigma**2))
            pressure += amp * blob
    
    # Add palm pressure (proportional to metacarpal curl)
    # Use higher coefficient than individual fingers to ensure visibility
    meta_pressure = np.mean([joint_pressures.get(j, 0.0) for j in [4, 9, 14, 19]])
    palm_pressure = meta_pressure * 1.2  # palm pressure slightly higher than metacarpals
    
    for (pr, pc) in PALM_CENTERS:
        if palm_pressure > 1e-6:
            blob = np.exp(-((yy - pr)**2 + (xx - pc)**2) / (2 * (sigma * 1.5)**2))
            pressure += palm_pressure * blob
    
    # Apply hand mask
    pressure = pressure * HAND_MASK.astype(np.float32)
    
    # Return absolute pressure values (no normalization)
    # This preserves the relative magnitude between different hands/frames
    return pressure


def generate_pseudo_pressure_maps_batch(
    poses_3d: np.ndarray,
    map_size: int = 16,
    sigma: float = 1.2,
) -> np.ndarray:
    """
    Generate pseudo pressure maps for a batch of frames with left+right hands.
    
    Args:
        poses_3d: (T, J, 3) where J = 48 (24 left + 24 right)
        map_size: output map resolution
        sigma: Gaussian blob spread
    
    Returns:
        pressure_maps: (T, 2, map_size, map_size) - [left, right]
    """
    T = poses_3d.shape[0]
    maps = np.zeros((T, 2, map_size, map_size), dtype=np.float32)
    
    for t in range(T):
        left_joints = poses_3d[t, :24]   # (24, 3)
        right_joints = poses_3d[t, 24:]  # (24, 3)
        
        maps[t, 0] = generate_pseudo_pressure_map(left_joints, map_size, sigma)
        maps[t, 1] = generate_pseudo_pressure_map(right_joints, map_size, sigma)
    
    return maps


def get_hand_mask(map_size: int = 16) -> np.ndarray:
    """Return the pre-computed hand-shaped mask. Useful for visualization."""
    if map_size == 16:
        return HAND_MASK.copy()
    return _hand_mask_16x16()  # regenerate if non-default size
