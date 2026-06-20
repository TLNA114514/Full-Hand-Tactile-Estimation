#!/usr/bin/env python3
"""
TouchAnything dataset conversion: raw trajectory format -> HDF5.

This script packs scattered txt/png files into a single HDF5 file so they are
easier to load during training and inference.

Usage:
    # Convert a single trajectory (including pressure preprocessing)
    python scripts/core/convert_to_hdf5.py --traj_path /path/to/trajectory

    # Batch conversion
    python scripts/core/convert_to_hdf5.py --root_path /path/to/dataset --batch

    # Specify mapping files explicitly
    python scripts/core/convert_to_hdf5.py --traj_path /path/to/trajectory \\
        --mapping_left /path/to/mapping_left.json \\
        --mapping_right /path/to/mapping_right.json
"""

import os
import sys
import json
import argparse
import h5py
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


# Default mapping-file paths relative to the project root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_DEFAULT_MAPPING_LEFT  = str(_PROJECT_ROOT / 'configs' / 'pressure_position_mapping_left.json')
_DEFAULT_MAPPING_RIGHT = str(_PROJECT_ROOT / 'configs' / 'pressure_position_mapping_right.json')
_DEFAULT_HDF5_OUTPUT_ROOT = _PROJECT_ROOT / 'datasets' / 'TouchAnything_hdf5_wilor'

HAND_GRID_SIZE = 21  # Sensor-grid size.

# Baseline-noise threshold: subtract the first frame only when its mean sensor
# value is low enough to be treated as unloaded noise.
# Raw uint8 values are in [0, 255].
BASELINE_NOISE_THRESHOLD = 15.0


# Duplicate frame threshold for quality check
DUPLICATE_FRAME_THRESHOLD = 0.20  # 20%


def _parse_bad_files_list(bad_files_path: Path, reason_filter: str = None):
    bad_files_path = Path(bad_files_path)
    if not bad_files_path.exists():
        raise FileNotFoundError(f"bad_files_list not found: {bad_files_path}")

    allowed_reasons = None
    if reason_filter is not None:
        allowed_reasons = {r.strip() for r in reason_filter.split(',') if r.strip()}
        if not allowed_reasons:
            allowed_reasons = None

    rel_trajs = []
    with open(bad_files_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('='):
                continue

            if '|' in line:
                left, right = line.split('|', 1)
                rel_hdf5 = left.strip()
                reason = right.strip()
            else:
                rel_hdf5 = line
                reason = ''

            if allowed_reasons is not None:
                reason_tag = reason.split('(')[0].strip() if reason else ''
                if reason_tag not in allowed_reasons:
                    continue

            rel_path = Path(rel_hdf5)
            if rel_path.suffix == '.hdf5':
                rel_path = rel_path.with_suffix('')
            rel_trajs.append(rel_path)

    return rel_trajs


def validate_required_files(traj_dir):
    """
    Validate that all required files exist for conversion.
    
    Required files:
    - chest.mp4
    - left.mp4
    - right.mp4
    - pressure_grids.npz
    - wilor_hands.json
    
    Args:
        traj_dir: Path to trajectory directory
        
    Returns:
        (is_valid, missing_files): Tuple of validation result and list of missing files
    """
    traj_dir = Path(traj_dir)
    
    required_files = [
        'chest.mp4',
        'left.mp4',
        'right.mp4',
        'pressure_grids.npz',
        'wilor_hands.json',
    ]
    
    missing_files = []
    for filename in required_files:
        if not (traj_dir / filename).exists():
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files


def check_quality_issues(traj_dir):
    """
    Check quality_issues.json for duplicate frame ratios.
    
    Args:
        traj_dir: Path to trajectory directory
        
    Returns:
        dict with keys:
        - left_duplicate_ratio: float or None
        - right_duplicate_ratio: float or None
        - left_has_issue: bool (True if > 20% duplicates)
        - right_has_issue: bool (True if > 20% duplicates)
    """
    quality_file = traj_dir / 'quality_issues.json'
    
    result = {
        'left_duplicate_ratio': None,
        'right_duplicate_ratio': None,
        'left_has_issue': False,
        'right_has_issue': False,
    }
    
    if not quality_file.exists():
        return result
    
    try:
        with open(quality_file, 'r') as f:
            quality_data = json.load(f)
        
        # Check duplicate frame ratios
        if 'duplicate_frames' in quality_data:
            dup_data = quality_data['duplicate_frames']
            
            # Left wrist
            if 'left' in dup_data:
                ratio = dup_data['left'].get('ratio', 0.0)
                result['left_duplicate_ratio'] = ratio
                result['left_has_issue'] = ratio > DUPLICATE_FRAME_THRESHOLD
            
            # Right wrist
            if 'right' in dup_data:
                ratio = dup_data['right'].get('ratio', 0.0)
                result['right_duplicate_ratio'] = ratio
                result['right_has_issue'] = ratio > DUPLICATE_FRAME_THRESHOLD
    
    except Exception as e:
        print(f"    Warning: Failed to read quality_issues.json: {e}")
    
    return result


def find_mask_file(traj_dir):
    """
    Find the mask NPZ file in the trajectory directory.
    
    Args:
        traj_dir: Path to trajectory directory
        
    Returns:
        Path to masks.npz file, or None if not found
    """
    traj_dir = Path(traj_dir)
    
    # Look for masks.npz (single prompt mode - preferred)
    mask_file = traj_dir / "masks.npz"
    if mask_file.exists():
        return mask_file
    
    # Look for masks_left_hand.npz and masks_right_hand.npz (separate mode)
    left_file = traj_dir / "masks_left_hand.npz"
    right_file = traj_dir / "masks_right_hand.npz"
    if left_file.exists() and right_file.exists():
        return (left_file, right_file)
    
    return None


def load_mask_data(mask_file, num_frames, img_height, img_width):
    """
    Load mask data from NPZ file(s).
    
    Args:
        mask_file: Path to mask file or tuple of (left, right) paths
        num_frames: Expected number of frames
        img_height: Image height
        img_width: Image width
        
    Returns:
        Dictionary with 'masks', 'obj_ids', 'valid_frames', or None if no mask data
    """
    if mask_file is None:
        # No mask data - return empty placeholders
        return {
            'masks': np.zeros((num_frames, 0, img_height, img_width), dtype=np.uint8),
            'obj_ids': np.zeros((num_frames, 0), dtype=np.int32),
            'valid_frames': np.zeros(num_frames, dtype=bool),
        }
    
    try:
        if isinstance(mask_file, tuple):
            # Separate left/right mode
            left_file, right_file = mask_file
            left_data = np.load(left_file)
            right_data = np.load(right_file)
            
            # Combine left and right masks
            masks = np.concatenate([left_data['masks'], right_data['masks']], axis=1)
            obj_ids = np.concatenate([left_data['obj_ids'], right_data['obj_ids']], axis=1)
            valid_frames = left_data['valid_frames'] | right_data['valid_frames']
        else:
            # Single prompt mode
            data = np.load(mask_file)
            masks = data['masks']
            obj_ids = data['obj_ids']
            valid_frames = data['valid_frames']
        
        # Verify frame count
        if len(masks) != num_frames:
            print(f"    Warning: mask frame count mismatch (expected {num_frames}, got {len(masks)}), using empty masks")
            return {
                'masks': np.zeros((num_frames, 0, img_height, img_width), dtype=np.uint8),
                'obj_ids': np.zeros((num_frames, 0), dtype=np.int32),
                'valid_frames': np.zeros(num_frames, dtype=bool),
            }
        
        return {
            'masks': masks,
            'obj_ids': obj_ids,
            'valid_frames': valid_frames,
        }
    
    except Exception as e:
        print(f"    Warning: failed to load mask data: {e}, using empty masks")
        return {
            'masks': np.zeros((num_frames, 0, img_height, img_width), dtype=np.uint8),
            'obj_ids': np.zeros((num_frames, 0), dtype=np.int32),
            'valid_frames': np.zeros(num_frames, dtype=bool),
        }


def is_hdf5_valid(hdf5_path):
    """
    Check if an HDF5 file is complete and valid.
    
    Args:
        hdf5_path: Path to the HDF5 file.
    
    Returns:
        bool: True if the file is valid and complete, False otherwise.
    """
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # Check essential groups exist.
            required_groups = ['metadata', 'images', 'poses', 'hands', 'pressure']
            for group in required_groups:
                if group not in f:
                    return False
            
            # Check metadata.
            meta = f['metadata']
            num_frames = meta.attrs.get('num_frames', 0)
            if num_frames <= 0:
                return False
            
            # Check essential datasets have correct shape.
            if 'timestamps' not in f or len(f['timestamps']) != num_frames:
                return False
            
            # Check image datasets.
            img_group = f['images']
            for key in ['chest_color', 'left_color', 'right_color']:
                if key not in img_group or img_group[key].shape[0] != num_frames:
                    return False
            
            # Check pose datasets.
            pose_group = f['poses']
            for key in ['chest_pose', 'left_pose', 'right_pose']:
                if key not in pose_group or pose_group[key].shape[0] != num_frames:
                    return False
            
            # Check hand datasets.
            hand_group = f['hands']
            for key in ['left_joint_xyz', 'right_joint_xyz']:
                if key not in hand_group or hand_group[key].shape[0] != num_frames:
                    return False
            
            # Check pressure datasets (only grids, raw sensor data was removed to save space).
            pressure_group = f['pressure']
            for key in ['left_pressure_grid', 'right_pressure_grid']:
                if key not in pressure_group or pressure_group[key].shape[0] != num_frames:
                    return False
            
            return True
    except Exception as e:
        # File is corrupted or cannot be read.
        return False



def load_pose(txt_path):
    """Load a 7D pose vector (xyz + quaternion)."""
    if not os.path.exists(txt_path):
        return np.zeros(7, dtype=np.float32)
    try:
        vals = list(map(float, open(txt_path).read().split()))
        if len(vals) == 7:
            return np.array(vals, dtype=np.float32)
    except Exception:
        pass
    return np.zeros(7, dtype=np.float32)


# Depth images removed - not needed for training


def load_json_data(json_path):
    """Load a JSONL file (one JSON object per line)."""
    json_path = Path(json_path)
    if not json_path.exists():
        return []
    data = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def load_wilor_jsonl(wilor_json_path, num_frames):
    """
    Read wilor_hands.json (JSONL) and return fixed-length arrays.
    
    Filters out invalid data:
    - NaN values (WiLoR prediction failures)
    - Extreme values (|x| > 1000 or |y| > 1000 or z > 1000)
    - These indicate WiLoR detection failures and should be marked as invalid

    Returns:
        left_xyz:   [T, 21, 3] float32
        right_xyz:  [T, 21, 3] float32
        left_valid: [T] uint8 (0 = invalid, 1 = valid)
        right_valid:[T] uint8
    """
    left_xyz = np.zeros((num_frames, 21, 3), dtype=np.float32)
    right_xyz = np.zeros((num_frames, 21, 3), dtype=np.float32)
    left_valid = np.zeros((num_frames,), dtype=np.uint8)
    right_valid = np.zeros((num_frames,), dtype=np.uint8)

    wilor_json_path = Path(wilor_json_path)
    if not wilor_json_path.exists():
        return left_xyz, right_xyz, left_valid, right_valid

    def is_valid_pose(pose_data):
        """Check if pose data contains valid coordinates (no NaN or extreme values)."""
        if not isinstance(pose_data, list) or len(pose_data) != 21:
            return False
        
        try:
            pose_array = np.asarray(pose_data, dtype=np.float32)
            # Check for NaN
            if np.isnan(pose_array).any():
                return False
            # Check for inf
            if np.isinf(pose_array).any():
                return False
            # Check for extreme values (WiLoR failures often produce values > 1e10)
            # Normal range: x/y in [-5, 5], z in [0, 100]
            # Allow some margin but reject clearly abnormal values
            if np.abs(pose_array[:, :2]).max() > 1000:  # x, y should be < 1000
                return False
            if pose_array[:, 2].max() > 1000:  # z should be < 1000
                return False
            return True
        except (ValueError, TypeError):
            return False

    with open(wilor_json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_frames:
                break
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue

            left_pos = frame.get('left_pos', [])
            right_pos = frame.get('right_pos', [])

            # Only mark as valid if data passes sanity checks
            if is_valid_pose(left_pos):
                left_xyz[i] = np.asarray(left_pos, dtype=np.float32)
                left_valid[i] = 1

            if is_valid_pose(right_pos):
                right_xyz[i] = np.asarray(right_pos, dtype=np.float32)
                right_valid[i] = 1

    return left_xyz, right_xyz, left_valid, right_valid


def default_output_hdf5_path(traj_path: Path) -> Path:
    """Write to a new directory under project datasets by default to avoid overwriting raw data."""
    task_name = traj_path.parent.name
    return _DEFAULT_HDF5_OUTPUT_ROOT / task_name / f"{traj_path.name}.hdf5"


def extract_video_frames(video_path, gpu_id=None):
    """
    Extract all frames from a video using GPU-accelerated FFmpeg decoding.
    Returns RGB frames as numpy arrays (same format as before).
    
    Args:
        video_path: Path to video file
        gpu_id: GPU device ID to use (0-7). If None, auto-select based on process ID.
    
    Falls back to OpenCV if FFmpeg fails.
    """
    import subprocess
    import os
    
    video_path = str(video_path)
    
    # Auto-select GPU based on process ID if not specified
    if gpu_id is None:
        # Distribute across 8 GPUs (A100 x8)
        gpu_id = os.getpid() % 8
    
    # Try GPU-accelerated FFmpeg first
    try:
        # Get video info first
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=width,height,nb_read_packets',
            '-of', 'csv=p=0',
            video_path
        ]
        probe_output = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL).decode().strip()
        width, height, num_frames = probe_output.split(',')
        width, height, num_frames = int(width), int(height), int(num_frames)
        
        # Use FFmpeg with CUDA hardware decoding to extract RGB frames
        # Distribute load across GPUs for better parallelism
        ffmpeg_cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',                    # Use CUDA hardware acceleration
            '-hwaccel_device', str(gpu_id),        # Use assigned GPU
            '-c:v', 'h264_cuvid',                  # Use CUDA H.264 decoder explicitly
            '-i', video_path,
            '-f', 'rawvideo',                      # Output raw video
            '-pix_fmt', 'rgb24',                   # Direct RGB output (no BGR conversion!)
            'pipe:1'                               # Output to stdout
        ]
        
        # Run FFmpeg and capture output
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )
        
        # Read all frames at once
        raw_data = process.stdout.read()
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError("FFmpeg failed")
        
        # Convert raw RGB data to numpy array
        frame_size = height * width * 3
        expected_size = frame_size * num_frames
        
        if len(raw_data) != expected_size:
            # Frame count mismatch, recalculate
            num_frames = len(raw_data) // frame_size
        
        # Reshape to list of frames
        frames_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(num_frames, height, width, 3)
        frames = [frames_array[i] for i in range(num_frames)]
        
        return frames
        
    except Exception as e:
        # Fallback to OpenCV if FFmpeg fails
        # print(f"  FFmpeg GPU decode failed ({e}), falling back to OpenCV...")
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return frames


def detect_data_format(traj_path):
    """Detect data format: 'new' (MP4+JSON) or 'old' (data/timestamps/)."""
    traj_path = Path(traj_path)
    
    # Check marker files for the new format.
    has_mp4 = (traj_path / 'chest.mp4').exists()
    has_json = (traj_path / 'jq_pressure.json').exists()
    
    if has_mp4 and has_json:
        return 'new'
    
    # Check the legacy format.
    data_dir = traj_path / 'data'
    if data_dir.exists() and any(d.is_dir() for d in data_dir.iterdir()):
        return 'old'
    
    return 'unknown'




def convert_trajectory_to_hdf5(
    traj_path,
    output_path=None,
    compression='gzip',
    compression_opts=9,
    skip_existing=False,
):
    """
    Convert a single trajectory to HDF5.
    
    Only processes trajectories with all required files:
    - chest.mp4, left.mp4, right.mp4
    - pressure_grids.npz
    - wilor_hands.json
    
    Args:
        traj_path:          Trajectory root directory.
        output_path:        Output HDF5 path; defaults to traj_path.hdf5.
        compression:        Compression algorithm ('gzip', 'lzf', None).
        compression_opts:   Compression level (1-9 for gzip).
        skip_existing:      Skip if valid HDF5 already exists.
    
    Returns:
        output_path if successful, None if skipped/failed
    """
    traj_path = Path(traj_path)
    
    # Only support new format (MP4 + JSON + NPZ)
    data_format = detect_data_format(traj_path)
    
    if data_format == 'new':
        return _convert_new_format(traj_path, output_path, compression, compression_opts, skip_existing)
    else:
        raise ValueError(f"Only new format (MP4+JSON+NPZ) is supported. Got: {data_format}")


def load_pressure_grids_from_npz(npz_path, num_frames):
    """
    Load pressure grids from existing pressure_grids.npz file.
    
    Args:
        npz_path: Path to pressure_grids.npz
        num_frames: Expected number of frames
        
    Returns:
        (left_grids, right_grids, metadata) or None if failed
    """
    try:
        data = np.load(npz_path)
        left_grids = data['left_pressure_grid']
        right_grids = data['right_pressure_grid']
        
        # Verify frame count
        if len(left_grids) != num_frames or len(right_grids) != num_frames:
            print(f"    Warning: pressure grid frame count mismatch (expected {num_frames}, got L:{len(left_grids)} R:{len(right_grids)})")
            return None
        
        # Extract metadata
        metadata = {
            'grid_size': int(data.get('grid_size', 21)),
            'num_frames': int(data.get('num_frames', num_frames)),
            'separate_normalization': bool(data.get('separate_normalization', False)),
        }
        
        # Add normalization info if available
        if metadata['separate_normalization']:
            metadata['tactile_max_left'] = float(data.get('tactile_max_left', 0))
            metadata['tactile_max_right'] = float(data.get('tactile_max_right', 0))
            metadata['bend_max_left'] = float(data.get('bend_max_left', 0))
            metadata['bend_max_right'] = float(data.get('bend_max_right', 0))
        
        return left_grids, right_grids, metadata
    
    except Exception as e:
        print(f"    Error loading pressure_grids.npz: {e}")
        return None


def _convert_new_format(
    traj_path,
    output_path,
    compression,
    compression_opts,
    skip_existing,
):
    """Convert new-format data (MP4 + JSON + NPZ)."""
    traj_path = Path(traj_path)
    
    # Resolve output path.
    if output_path is None:
        output_path = default_output_hdf5_path(traj_path)
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and output_path.exists():
        if is_hdf5_valid(output_path):
            print(f"  [Skip] valid HDF5 already exists: {output_path.name}")
            return output_path
        else:
            print(f"  [Warning] Existing HDF5 is corrupted, will re-convert: {output_path.name}")
            try:
                output_path.unlink()
            except OSError:
                pass

    # Validate required files
    is_valid, missing_files = validate_required_files(traj_path)
    if not is_valid:
        raise ValueError(f"Missing required files: {', '.join(missing_files)}")
    
    # Check quality issues
    quality_info = check_quality_issues(traj_path)
    
    # Load JSON data (silent)
    rokoko_data = load_json_data(traj_path / 'rokoko_hands.json')
    vive_data = load_json_data(traj_path / 'vive_poses.json')
    
    # Extract video frames (silent)
    chest_video = traj_path / 'chest.mp4'
    left_video = traj_path / 'left.mp4'
    right_video = traj_path / 'right.mp4'

    chest_frames = extract_video_frames(chest_video)
    left_frames = extract_video_frames(left_video)
    right_frames = extract_video_frames(right_video)

    # Diagnose decode failures early to avoid confusing downstream errors.
    empty_videos = []
    if len(chest_frames) == 0:
        empty_videos.append('chest.mp4')
    if len(left_frames) == 0:
        empty_videos.append('left.mp4')
    if len(right_frames) == 0:
        empty_videos.append('right.mp4')
    if empty_videos:
        raise ValueError(
            "video_decode_failed: empty frames from " + ",".join(empty_videos)
        )

    # Determine frame count from videos
    num_frames = min(len(chest_frames), len(left_frames), len(right_frames))
    if num_frames <= 0:
        raise ValueError(
            f"video_decode_failed: num_frames={num_frames} (lens: chest={len(chest_frames)}, left={len(left_frames)}, right={len(right_frames)})"
        )
    
    # Only show essential info: trajectory name and frame count
    warnings = []
    if quality_info['left_has_issue']:
        warnings.append(f"L:{quality_info['left_duplicate_ratio']*100:.0f}%dup")
    if quality_info['right_has_issue']:
        warnings.append(f"R:{quality_info['right_duplicate_ratio']*100:.0f}%dup")
    
    warning_str = f" [{', '.join(warnings)}]" if warnings else ""
    print(f"[{traj_path.name}] {num_frames} frames{warning_str}")

    # Safety: ensure first frame is valid
    if chest_frames[0] is None or not hasattr(chest_frames[0], 'shape'):
        raise ValueError("video_decode_failed: chest.mp4 first frame invalid")

    img_h, img_w = chest_frames[0].shape[:2]
    
    # Generate timestamps from frame indices (30 FPS)
    timestamps = np.arange(num_frames, dtype=np.int64) * (1000000 // 30)  # microseconds
    
    # Create HDF5 file with retry mechanism for file locking issues
    max_retries = 3
    retry_delay = 1.0  # seconds
    
    for attempt in range(max_retries):
        try:
            f = h5py.File(output_path, 'w')
            break
        except OSError as e:
            if attempt < max_retries - 1 and 'unable to lock file' in str(e):
                import time
                print(f"  File lock conflict, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise
    
    with f:
        # ========== Metadata ==========
        meta = f.create_group('metadata')
        meta.attrs['trajectory_id'] = traj_path.name
        meta.attrs['task_name'] = traj_path.parent.name
        meta.attrs['num_frames'] = num_frames
        meta.attrs['fps'] = 30
        meta.attrs['camera_resolution'] = [img_h, img_w]
        meta.attrs['data_format'] = 'new'
        
        # Quality metadata
        meta.attrs['left_wrist_duplicate_ratio'] = quality_info['left_duplicate_ratio'] if quality_info['left_duplicate_ratio'] is not None else -1.0
        meta.attrs['right_wrist_duplicate_ratio'] = quality_info['right_duplicate_ratio'] if quality_info['right_duplicate_ratio'] is not None else -1.0
        meta.attrs['left_wrist_has_issue'] = quality_info['left_has_issue']
        meta.attrs['right_wrist_has_issue'] = quality_info['right_has_issue']
        
        # ========== Timestamps ==========
        f.create_dataset('timestamps', data=np.array(timestamps, dtype=np.int64))
        
        # ========== Image Data ==========
        img_group = f.create_group('images')
        
        # Write image data (silent)
        try:
            img_group.create_dataset('chest_color',
                data=np.stack(chest_frames[:num_frames]), dtype=np.uint8,
                compression=compression, compression_opts=compression_opts)
            img_group.create_dataset('left_color',
                data=np.stack(left_frames[:num_frames]), dtype=np.uint8,
                compression=compression, compression_opts=compression_opts)
            img_group.create_dataset('right_color',
                data=np.stack(right_frames[:num_frames]), dtype=np.uint8,
                compression=compression, compression_opts=compression_opts)
        except Exception as e:
            raise ValueError(f"image_stack_failed: {e}")
        
        # New format has no depth images; skip depth data to save space
        # (Removed: empty depth arrays were wasting ~2GB per file)
        
        # ========== Pose Data ==========
        pose_group = f.create_group('poses')
        
        # Process pose data (silent)
        if vive_data:
            chest_poses = []
            left_poses = []
            right_poses = []
            
            for vd in vive_data[:num_frames]:
                try:
                    poses = vd['poses']
                    chest_poses.append(poses['chest']['trans'] + poses['chest']['rot'])
                    left_poses.append(poses['left_wrist']['trans'] + poses['left_wrist']['rot'])
                    right_poses.append(poses['right_wrist']['trans'] + poses['right_wrist']['rot'])
                except (KeyError, TypeError):
                    chest_poses.append([0.0]*7)
                    left_poses.append([0.0]*7)
                    right_poses.append([0.0]*7)
            
            pose_group.create_dataset('chest_pose', data=np.array(chest_poses, dtype=np.float32))
            pose_group.create_dataset('left_pose', data=np.array(left_poses, dtype=np.float32))
            pose_group.create_dataset('right_pose', data=np.array(right_poses, dtype=np.float32))
        else:
            empty_pose = np.zeros((num_frames, 7), dtype=np.float32)
            pose_group.create_dataset('chest_pose', data=empty_pose)
            pose_group.create_dataset('left_pose', data=empty_pose)
            pose_group.create_dataset('right_pose', data=empty_pose)
        
        # ========== Hand Data ==========
        hand_group = f.create_group('hands')
        # Process hand-joint data (silent)
        if rokoko_data:
            left_xyz_list = []
            right_xyz_list = []
            
            for rd in rokoko_data[:num_frames]:
                try:
                    left_xyz_list.append(np.array(rd['left_pos'], dtype=np.float32))
                    right_xyz_list.append(np.array(rd['right_pos'], dtype=np.float32))
                except (KeyError, TypeError):
                    left_xyz_list.append(np.zeros((21, 3), dtype=np.float32))
                    right_xyz_list.append(np.zeros((21, 3), dtype=np.float32))
            
            hand_group.create_dataset('left_joint_xyz', data=np.stack(left_xyz_list), dtype=np.float32)
            hand_group.create_dataset('right_joint_xyz', data=np.stack(right_xyz_list), dtype=np.float32)
        else:
            # rokoko_hands.json is empty; filling zeros (silent)
            empty_xyz = np.zeros((num_frames, 21, 3), dtype=np.float32)
            hand_group.create_dataset('left_joint_xyz', data=empty_xyz, dtype=np.float32)
            hand_group.create_dataset('right_joint_xyz', data=empty_xyz, dtype=np.float32)
        
        # WiLoR joints: write if available; otherwise write zeros and valid=0 to keep a stable schema.
        wilor_json_path = traj_path / 'wilor_hands.json'
        w_lxyz, w_rxyz, w_lvalid, w_rvalid = load_wilor_jsonl(wilor_json_path, num_frames)
        hand_group.create_dataset('wilor_left_joint_xyz', data=w_lxyz, dtype=np.float32)
        hand_group.create_dataset('wilor_right_joint_xyz', data=w_rxyz, dtype=np.float32)
        hand_group.create_dataset('wilor_left_valid', data=w_lvalid, dtype=np.uint8)
        hand_group.create_dataset('wilor_right_valid', data=w_rvalid, dtype=np.uint8)
        hand_group.attrs['wilor_available'] = bool(wilor_json_path.exists())
        hand_group.attrs['wilor_source_file'] = str(wilor_json_path) if wilor_json_path.exists() else ''
        hand_group.attrs['wilor_left_valid_frames'] = int(w_lvalid.sum())
        hand_group.attrs['wilor_right_valid_frames'] = int(w_rvalid.sum())
        
        # ========== Pressure Data ==========
        pressure_group = f.create_group('pressure')
        
        # Load pressure grids from pressure_grids.npz (silent)
        pressure_npz_path = traj_path / 'pressure_grids.npz'
        pressure_result = load_pressure_grids_from_npz(pressure_npz_path, num_frames)
        
        if pressure_result is None:
            raise ValueError(f"Failed to load pressure_grids.npz: {pressure_npz_path}")
        
        left_grids, right_grids, pressure_metadata = pressure_result
        
        pressure_group.create_dataset('left_pressure_grid', data=left_grids, dtype=np.float32,
            compression=compression, compression_opts=compression_opts)
        pressure_group.create_dataset('right_pressure_grid', data=right_grids, dtype=np.float32,
            compression=compression, compression_opts=compression_opts)
        
        # Record pressure metadata from npz file
        pressure_group.attrs['grid_size'] = pressure_metadata['grid_size']
        pressure_group.attrs['separate_normalization'] = pressure_metadata['separate_normalization']
        
        if pressure_metadata['separate_normalization']:
            pressure_group.attrs['tactile_max_left'] = pressure_metadata['tactile_max_left']
            pressure_group.attrs['tactile_max_right'] = pressure_metadata['tactile_max_right']
            pressure_group.attrs['bend_max_left'] = pressure_metadata['bend_max_left']
            pressure_group.attrs['bend_max_right'] = pressure_metadata['bend_max_right']
        
        # ========== Mask Data ==========
        masks_group = f.create_group('masks')
        
        # Load glove mask data (silent)
        mask_file = find_mask_file(traj_path)
        mask_data = load_mask_data(mask_file, num_frames, img_h, img_w)
        
        masks_group.create_dataset('glove_masks', 
            data=mask_data['masks'], dtype=np.uint8,
            compression=compression, compression_opts=compression_opts)
        masks_group.create_dataset('glove_obj_ids',
            data=mask_data['obj_ids'], dtype=np.int32)
        masks_group.create_dataset('glove_valid_frames',
            data=mask_data['valid_frames'], dtype=np.bool_)
        
        masks_group.attrs['num_objects'] = mask_data['masks'].shape[1] if len(mask_data['masks'].shape) > 1 else 0
        masks_group.attrs['mask_available'] = mask_file is not None
        masks_group.attrs['valid_frame_count'] = int(mask_data['valid_frames'].sum())
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Conversion complete: {output_path.name} ({file_size_mb:.1f} MB)")
    
    return output_path




def check_trajectory_quality(hdf5_path):
    """
    Check data quality of one HDF5 trajectory and return detected issues.
    
    Checks:
    1. Whether vive_poses stays constant.
    2. Whether rokoko_hands stays constant.
    3. Whether pressure values stay constant.
    4. Whether one hand's pressure is always 0.
    5. Whether baseline correction was skipped (high first-frame pressure).
    
    Args:
        hdf5_path: HDF5 file path.
    Returns:
        dict: {
            'traj_name': str,
            'task_name': str,
            'num_frames': int,
            'issues': [str, ...],        # Issue list
            'warnings': [str, ...],      # Warning list
        }
    """
    hdf5_path = Path(hdf5_path)
    result = {
        'traj_name': hdf5_path.stem,
        'task_name': '',
        'num_frames': 0,
        'issues': [],
        'warnings': [],
    }
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            meta = f.get('metadata')
            if meta:
                result['task_name'] = meta.attrs.get('task_name', '')
                result['num_frames'] = int(meta.attrs.get('num_frames', 0))
            
            num_frames = result['num_frames']
            if num_frames < 2:
                result['issues'].append(f"Too few frames: {num_frames}")
                return result
            
            # ---------- Check poses ----------
            if 'poses' in f:
                pose_group = f['poses']
                for key in ['chest_pose', 'left_pose', 'right_pose']:
                    if key in pose_group:
                        data = pose_group[key][...]  # [T, 7]
                        if np.all(data == data[0]):
                            result['issues'].append(f"Pose {key} is exactly constant across all frames")
                        elif np.allclose(data, 0.0):
                            result['warnings'].append(f"Pose {key} is all zeros (data may be missing)")
            else:
                result['issues'].append("Missing poses group")
            
            # ---------- Check hands ----------
            if 'hands' in f:
                hand_group = f['hands']
                for key in ['left_joint_xyz', 'right_joint_xyz']:
                    if key in hand_group:
                        data = hand_group[key][...]  # [T, 21, 3]
                        if np.all(data == data[0]):
                            result['issues'].append(f"Hand joints {key} are exactly constant across all frames")
                        elif np.allclose(data, 0.0):
                            result['warnings'].append(f"Hand joints {key} are all zeros (data may be missing)")
            else:
                result['issues'].append("Missing hands group")
            
            # ---------- Check pressure ----------
            if 'pressure' in f:
                pressure_group = f['pressure']
                
                # Check raw pressure.
                for side, key in [('Left hand', 'left_sensor_raw'), ('Right hand', 'right_sensor_raw')]:
                    if key in pressure_group:
                        raw = pressure_group[key][...]  # [T, 256] uint8
                        if np.all(raw == raw[0]):
                            result['issues'].append(f"{side} raw pressure ({key}) is exactly constant across all frames")
                        if np.all(raw == 0):
                            result['issues'].append(f"{side} raw pressure ({key}) is all zeros")
                
                # Check processed pressure grids.
                for side, key in [('Left hand', 'left_pressure_grid'), ('Right hand', 'right_pressure_grid')]:
                    if key in pressure_group:
                        grid = pressure_group[key][...]  # [T, 21, 21]
                        valid = grid[~np.isnan(grid)]
                        if len(valid) > 0 and np.all(valid == 0):
                            result['issues'].append(f"{side} pressure grid ({key}) has all valid values equal to 0")
                
                # Check baseline-correction status.
                bl_l = pressure_group.attrs.get('baseline_corrected_left', None)
                bl_r = pressure_group.attrs.get('baseline_corrected_right', None)
                if bl_l is False:
                    result['warnings'].append("Left-hand first-frame pressure is high; baseline correction was skipped (contact may already exist at start)")
                if bl_r is False:
                    result['warnings'].append("Right-hand first-frame pressure is high; baseline correction was skipped (contact may already exist at start)")
            else:
                result['issues'].append("Missing pressure group")
    
    except Exception as e:
        result['issues'].append(f"Failed to read HDF5: {str(e)}")
    
    return result


def generate_quality_report(output_dir, quality_results):
    """
    Generate a data-quality report and save it to the output directory.
    
    Args:
        output_dir: Output directory.
        quality_results: List of results returned by check_trajectory_quality.
    """
    output_dir = Path(output_dir)
    report_path = output_dir / 'quality_report.txt'
    
    # Aggregate statistics.
    total = len(quality_results)
    with_issues = [r for r in quality_results if r['issues']]
    with_warnings = [r for r in quality_results if r['warnings']]
    clean = [r for r in quality_results if not r['issues'] and not r['warnings']]
    
    # Group by task.
    by_task = {}
    for r in quality_results:
        task = r['task_name'] or 'Unknown task'
        by_task.setdefault(task, []).append(r)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TouchAnything Data Quality Report\n")
        f.write(f"Generated at: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total trajectories: {total}\n")
        f.write(f"  With issues: {len(with_issues)}\n")
        f.write(f"  With warnings: {len(with_warnings)}\n")
        f.write(f"  Clean: {len(clean)}\n\n")
        
        # ========== Issue Summary ==========
        if with_issues:
            f.write("=" * 80 + "\n")
            f.write("[ERROR] Issue Summary (data may have defects)\n")
            f.write("=" * 80 + "\n\n")
            
            # Group by issue type.
            issue_categories = {}
            for r in with_issues:
                for issue in r['issues']:
                    issue_categories.setdefault(issue, []).append(r['traj_name'])
            
            for issue_desc, trajs in sorted(issue_categories.items(), key=lambda x: -len(x[1])):
                f.write(f"[{issue_desc}] ({len(trajs)} trajectories)\n")
                for tn in trajs[:20]:
                    f.write(f"    - {tn}\n")
                if len(trajs) > 20:
                    f.write(f"    ... and {len(trajs)-20} more\n")
                f.write("\n")
        
        # ========== Warning Summary ==========
        if with_warnings:
            f.write("=" * 80 + "\n")
            f.write("[WARN] Warning Summary (may be normal but needs attention)\n")
            f.write("=" * 80 + "\n\n")
            
            warning_categories = {}
            for r in with_warnings:
                for w in r['warnings']:
                    warning_categories.setdefault(w, []).append(r['traj_name'])
            
            for warn_desc, trajs in sorted(warning_categories.items(), key=lambda x: -len(x[1])):
                f.write(f"[{warn_desc}] ({len(trajs)} trajectories)\n")
                for tn in trajs[:20]:
                    f.write(f"    - {tn}\n")
                if len(trajs) > 20:
                    f.write(f"    ... and {len(trajs)-20} more\n")
                f.write("\n")
        
        # ========== Per-Task Details ==========
        f.write("=" * 80 + "\n")
        f.write("Detailed Report by Task\n")
        f.write("=" * 80 + "\n\n")
        
        for task_name in sorted(by_task.keys()):
            results = by_task[task_name]
            task_issues = [r for r in results if r['issues']]
            task_warnings = [r for r in results if r['warnings']]
            f.write(f"[{task_name}] {len(results)} trajectories")
            if task_issues:
                f.write(f", {len(task_issues)} with issues")
            if task_warnings:
                f.write(f", {len(task_warnings)} with warnings")
            f.write("\n")
            
            for r in results:
                if r['issues'] or r['warnings']:
                    f.write(f"  {r['traj_name']} ({r['num_frames']} frames)\n")
                    for issue in r['issues']:
                        f.write(f"    [ERROR] {issue}\n")
                    for w in r['warnings']:
                        f.write(f"    [WARN] {w}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of report\n")
        f.write("=" * 80 + "\n")
    
    # Print summary.
    print(f"\nData quality report saved to: {report_path}")
    print(f"  Total trajectories: {total}, issues: {len(with_issues)}, warnings: {len(with_warnings)}, clean: {len(clean)}")
    if with_issues:
        print(f"  [ERROR] Trajectories with issues:")
        shown = 0
        for r in with_issues[:5]:
            for issue in r['issues']:
                print(f"     {r['traj_name']}: {issue}")
                shown += 1
        if len(with_issues) > 5:
            print(f"     ... see {report_path} for details")


def _batch_worker(args):
    """ProcessPoolExecutor worker function (top-level so it can be pickled)."""
    traj_dir, output_path, compression, compression_opts, skip_existing = args
    try:
        # Validate trajectory directory
        if not traj_dir.exists():
            return {'path': str(traj_dir), 'status': 'error', 'reason': 'Directory does not exist', 'duration': 0}
        
        # Check required files
        is_valid, missing_files = validate_required_files(traj_dir)
        if not is_valid:
            return {
                'path': str(traj_dir),
                'status': 'skipped',
                'reason': f'Missing required files: {", ".join(missing_files)}',
                'duration': 0
            }
        
        # Convert the trajectory
        result = convert_trajectory_to_hdf5(
            traj_dir,
            output_path=output_path,
            compression=compression,
            compression_opts=compression_opts,
            skip_existing=skip_existing,
        )
        
        # Get trajectory duration from metadata
        duration = 0
        try:
            with h5py.File(result, 'r') as f:
                num_frames = f['metadata'].attrs['num_frames']
                fps = f['metadata'].attrs.get('fps', 30)
                duration = num_frames / fps
        except:
            pass
        
        return {
            'path': str(traj_dir),
            'status': 'success',
            'output': str(result),
            'duration': duration
        }
    
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            'path': str(traj_dir),
            'status': 'error',
            'reason': str(e),
            'traceback': tb,
            'duration': 0
        }


def _generate_statistics_report(root_path, output_dir, task_stats, successful_trajs, 
                                success_count, skip_count, fail_count):
    """
    Generate dataset-conversion statistics and save them to conversion_summary.txt.
    
    Args:
        root_path: Raw dataset root directory.
        output_dir: HDF5 output directory.
        task_stats: Raw-data stats {task_name: {'trajs': [(traj_name, frame_count), ...]}}.
        successful_trajs: Successfully converted trajectories [(task_name, traj_name, frame_count), ...].
        success_count: Number of successful conversions.
        skip_count: Number of skipped trajectories.
        fail_count: Number of failed conversions.
    """
    summary_file = output_dir / "conversion_summary.txt"
    
    # Compute raw-data statistics.
    total_raw_tasks = len(task_stats)
    total_raw_trajs = sum(len(info['trajs']) for info in task_stats.values())
    total_raw_frames = sum(fc for info in task_stats.values() for _, fc in info['trajs'])
    total_raw_duration = total_raw_frames / 30.0  # 30fps
    
    # Raw-data statistics by task.
    raw_task_details = {}
    for task_name, info in task_stats.items():
        trajs = info['trajs']
        num_trajs = len(trajs)
        total_frames = sum(fc for _, fc in trajs)
        total_duration = total_frames / 30.0
        avg_duration = total_duration / num_trajs if num_trajs > 0 else 0.0
        raw_task_details[task_name] = {
            'num_trajs': num_trajs,
            'total_frames': total_frames,
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'trajs': trajs
        }
    
    # Compute post-conversion effective-data statistics.
    converted_tasks = {}
    for task_name, traj_name, frame_count in successful_trajs:
        if task_name not in converted_tasks:
            converted_tasks[task_name] = []
        converted_tasks[task_name].append((traj_name, frame_count))
    
    total_converted_tasks = len(converted_tasks)
    total_converted_trajs = len(successful_trajs)
    total_converted_frames = sum(fc for _, _, fc in successful_trajs)
    total_converted_duration = total_converted_frames / 30.0
    
    # Post-conversion statistics by task.
    converted_task_details = {}
    for task_name, trajs in converted_tasks.items():
        num_trajs = len(trajs)
        total_frames = sum(fc for _, fc in trajs)
        total_duration = total_frames / 30.0
        avg_duration = total_duration / num_trajs if num_trajs > 0 else 0.0
        converted_task_details[task_name] = {
            'num_trajs': num_trajs,
            'total_frames': total_frames,
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'trajs': trajs
        }
    
    # Write statistics report.
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TouchAnything Dataset Conversion Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Raw data path: {root_path}\n")
        f.write(f"Converted output path: {output_dir}\n")
        f.write(f"Conversion time: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Conversion summary.
        f.write("-" * 80 + "\n")
        f.write("Conversion Summary\n")
        f.write("-" * 80 + "\n")
        f.write(f"Successfully converted: {success_count} trajectories\n")
        f.write(f"Skipped: {skip_count} trajectories (already exist)\n")
        f.write(f"Failed: {fail_count} trajectories\n\n")
        
        # Raw-data statistics.
        f.write("=" * 80 + "\n")
        f.write("Raw Data Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tasks: {total_raw_tasks}\n")
        f.write(f"Total trajectories: {total_raw_trajs}\n")
        f.write(f"Total frames: {total_raw_frames}\n")
        f.write(f"Total duration: {total_raw_duration:.2f} s ({total_raw_duration/60:.2f} min)\n")
        f.write(f"Average per trajectory: {total_raw_duration/total_raw_trajs:.2f} s\n\n")
        
        # Raw-data details by task.
        f.write("-" * 80 + "\n")
        f.write("Raw Data - Per-Task Details\n")
        f.write("-" * 80 + "\n\n")
        for task_name in sorted(raw_task_details.keys()):
            details = raw_task_details[task_name]
            f.write(f"[{task_name}]\n")
            f.write(f"  Trajectories: {details['num_trajs']}\n")
            f.write(f"  Total frames: {details['total_frames']}\n")
            f.write(f"  Total duration: {details['total_duration']:.2f} s ({details['total_duration']/60:.2f} min)\n")
            f.write(f"  Average per trajectory: {details['avg_duration']:.2f} s\n")
            f.write(f"  Trajectory list:\n")
            for i, (traj_name, frame_count) in enumerate(sorted(details['trajs']), 1):
                duration = frame_count / 30.0
                f.write(f"    {i}. {traj_name} - {frame_count} frames ({duration:.2f} s)\n")
            f.write("\n")
        
        # Post-conversion effective-data statistics.
        f.write("=" * 80 + "\n")
        f.write("Post-Conversion Effective Data Statistics\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total tasks: {total_converted_tasks}\n")
        f.write(f"Total trajectories: {total_converted_trajs}\n")
        f.write(f"Total frames: {total_converted_frames}\n")
        f.write(f"Total duration: {total_converted_duration:.2f} s ({total_converted_duration/60:.2f} min)\n")
        avg_per_traj = total_converted_duration/total_converted_trajs if total_converted_trajs > 0 else 0.0
        f.write(f"Average per trajectory: {avg_per_traj:.2f} s\n\n")
        
        # Post-conversion details by task.
        f.write("-" * 80 + "\n")
        f.write("Post-Conversion Effective Data - Per-Task Details\n")
        f.write("-" * 80 + "\n\n")
        for task_name in sorted(converted_task_details.keys()):
            details = converted_task_details[task_name]
            f.write(f"[{task_name}]\n")
            f.write(f"  Trajectories: {details['num_trajs']}\n")
            f.write(f"  Total frames: {details['total_frames']}\n")
            f.write(f"  Total duration: {details['total_duration']:.2f} s ({details['total_duration']/60:.2f} min)\n")
            f.write(f"  Average per trajectory: {details['avg_duration']:.2f} s\n")
            f.write(f"  Trajectory list:\n")
            for i, (traj_name, frame_count) in enumerate(sorted(details['trajs']), 1):
                duration = frame_count / 30.0
                f.write(f"    {i}. {traj_name} - {frame_count} frames ({duration:.2f} s)\n")
            f.write("\n")
        
        # Data-integrity comparison.
        f.write("=" * 80 + "\n")
        f.write("Data Integrity Comparison\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Task conversion rate: {total_converted_tasks}/{total_raw_tasks} ({100*total_converted_tasks/total_raw_tasks if total_raw_tasks > 0 else 0:.1f}%)\n")
        f.write(f"Trajectory conversion rate: {total_converted_trajs}/{total_raw_trajs} ({100*total_converted_trajs/total_raw_trajs if total_raw_trajs > 0 else 0:.1f}%)\n")
        f.write(f"Frame conversion rate: {total_converted_frames}/{total_raw_frames} ({100*total_converted_frames/total_raw_frames if total_raw_frames > 0 else 0:.1f}%)\n")
        f.write(f"Duration conversion rate: {total_converted_duration:.2f}/{total_raw_duration:.2f} s ({100*total_converted_duration/total_raw_duration if total_raw_duration > 0 else 0:.1f}%)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("End of report\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nStatistics report saved to: {summary_file}")
    print(f"\nDataset statistics summary:")
    print(f"  Raw data: {total_raw_tasks} tasks, {total_raw_trajs} trajectories, {total_raw_duration:.1f} s")
    print(f"  Converted: {total_converted_tasks} tasks, {total_converted_trajs} trajectories, {total_converted_duration:.1f} s")
    print(f"  Conversion rate: {100*total_converted_trajs/total_raw_trajs if total_raw_trajs > 0 else 0:.1f}%")


def batch_convert(
    root_path,
    output_dir=None,
    compression='gzip',
    compression_opts=9,
    traj_workers=32,
    skip_existing=False,
    limit_to_trajs=None,
    delete_before_regen=False,
):
    """
    Batch-convert all trajectories in a dataset (multi-process parallelism).
    
    Only converts trajectories with all required files:
    - chest.mp4, left.mp4, right.mp4
    - pressure_grids.npz
    - wilor_hands.json

    Args:
        root_path:          Dataset root directory.
        output_dir:         Output directory; defaults to datasets/<root_name>_hdf5.
        compression:        Compression algorithm.
        compression_opts:   Compression level.
        traj_workers:       Number of worker processes for trajectory conversion.
        skip_existing:      Skip if valid HDF5 already exists.
        limit_to_trajs:     Optional iterable of trajectory paths relative to root_path (Scene/Task/Trajectory).
        delete_before_regen: If True, delete existing output HDF5 before conversion (for regeneration).
    """
    root_path = Path(root_path)
    
    if output_dir is None:
        output_dir = _PROJECT_ROOT / 'datasets' / f"{root_path.name}_hdf5"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all trajectory directories and gather raw-data statistics.
    # Supports two formats:
    # Legacy: root/task_category/task_name/trajectory_id/data/(frame subdirs)
    # New:    root/Scene/Task/trajectory_id/jq_pressure.json + *.mp4
    trajectories = []  # [(traj_dir, relative_path, frame_count), ...]
    task_stats = {}  # {relative_path: {'trajs': [(traj_name, frame_count), ...]}}
    
    limit_set = None
    if limit_to_trajs is not None:
        limit_set = {Path(p) for p in limit_to_trajs}

    def scan_for_trajectories(parent_dir, depth=0):
        """Recursively scan directories and locate trajectory folders (new format only)."""
        if depth > 4:
            return []
        found = []
        for item in parent_dir.iterdir():
            if not item.is_dir():
                continue
            
            # New format: contains chest.mp4 (use this as marker for trajectory directory)
            if (item / 'chest.mp4').exists():
                # Just mark as found, frame count will be determined during conversion
                # Use 0 as placeholder since we'll get actual count from video during conversion
                if limit_set is not None:
                    try:
                        rel_path = item.relative_to(root_path)
                    except Exception:
                        rel_path = None
                    if rel_path is None or rel_path not in limit_set:
                        continue
                found.append((item, 0))
            else:
                # No match here; recurse into child directories
                found.extend(scan_for_trajectories(item, depth + 1))
        return found
    
    # Scan all trajectories while keeping full relative paths.
    all_trajs = scan_for_trajectories(root_path)
    for traj_dir, frame_count in all_trajs:
        # Compute path relative to root (preserve Scene/Task structure).
        rel_path = traj_dir.relative_to(root_path)
        # Use parent path (Scene/Task) as the grouping key.
        parent_rel_path = rel_path.parent
        
        trajectories.append((traj_dir, parent_rel_path, frame_count))
        
        # Group statistics by parent relative path.
        if parent_rel_path not in task_stats:
            task_stats[parent_rel_path] = {'trajs': []}
        task_stats[parent_rel_path]['trajs'].append((traj_dir.name, frame_count))
    
    if len(trajectories) == 0:
        print(f"No trajectory data found: {root_path}")
        return
    
    print(f"===========================================")
    print(f"Batch HDF5 conversion")
    print(f"===========================================")
    print(f"Dataset root: {root_path}")
    print(f"Output directory: {output_dir}")
    print(f"Total trajectories found: {len(trajectories)}")
    print(f"Compression: {compression} (level {compression_opts})")
    print(f"Parallel workers: {traj_workers}")
    print(f"Skip existing files: {'yes' if skip_existing else 'no'}")
    print(f"===========================================\n")

    # Pre-create output directories and build task argument list
    task_args = []
    for traj_dir, parent_path, frame_count in trajectories:
        # Preserve full relative path structure (Scene/Task/Trajectory.hdf5)
        task_output_dir = output_dir / parent_path
        task_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_output_dir / f"{traj_dir.name}.hdf5"

        if delete_before_regen and output_path.exists():
            try:
                output_path.unlink()
            except Exception as e:
                print(f"Warning: failed to delete existing HDF5 before regen: {output_path} ({e})")
        task_args.append((
            traj_dir, output_path,
            compression, compression_opts,
            skip_existing,
        ))

    success_count = 0
    skip_count = 0
    fail_count = 0
    converted_duration = 0.0
    skip_reasons = {}  # {reason: count}
    successful_trajs = []  # Successfully converted trajectories
    failed_trajs = []  # Failed trajectories with reasons

    # Process conversions
    if task_args:
        with ProcessPoolExecutor(max_workers=traj_workers) as executor:
            futures = {executor.submit(_batch_worker, arg): arg for arg in task_args}
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Converting")
            for future in pbar:
                result = future.result()
                traj_path = Path(result['path'])
                traj_name = traj_path.name
                
                if result['status'] == 'success':
                    success_count += 1
                    converted_duration += result['duration']
                    parent_path = traj_path.parent.relative_to(root_path)
                    successful_trajs.append((str(parent_path), traj_name, result['duration']))
                    pbar.set_postfix(ok=success_count, skip=skip_count, fail=fail_count, last=traj_name)
                
                elif result['status'] == 'skipped':
                    skip_count += 1
                    reason = result['reason']
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                    pbar.set_postfix(ok=success_count, skip=skip_count, fail=fail_count)
                
                else:  # error
                    fail_count += 1
                    reason = result.get('reason', 'Unknown error')
                    failed_trajs.append((traj_name, reason))
                    tqdm.write(f"✗ {traj_name}: {reason}")
                    pbar.set_postfix(ok=success_count, skip=skip_count, fail=fail_count)

    # Generate conversion report
    report_path = output_dir / "conversion_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HDF5 Conversion Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset root: {root_path}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Total trajectories found: {len(trajectories)}\n\n")
        
        f.write("Conversion Summary:\n")
        f.write(f"  Successfully converted: {success_count} trajectories ({converted_duration/3600:.2f} hours)\n")
        f.write(f"  Skipped: {skip_count} trajectories\n")
        f.write(f"  Failed: {fail_count} trajectories\n\n")
        
        if skip_reasons:
            f.write("Skip Reasons:\n")
            for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
                f.write(f"  - {reason}: {count} trajectories\n")
            f.write("\n")
        
        if failed_trajs:
            f.write("Failed Trajectories:\n")
            for traj_name, reason in failed_trajs[:50]:  # Limit to first 50
                f.write(f"  - {traj_name}: {reason}\n")
            if len(failed_trajs) > 50:
                f.write(f"  ... and {len(failed_trajs) - 50} more\n")
            f.write("\n")
        
        f.write("Successfully Converted Trajectories:\n")
        for parent_path, traj_name, duration in successful_trajs[:100]:  # Limit to first 100
            f.write(f"  - {parent_path}/{traj_name} ({duration:.1f}s)\n")
        if len(successful_trajs) > 100:
            f.write(f"  ... and {len(successful_trajs) - 100} more\n")
    
    print(f"\n============================================")
    print(f"Batch conversion finished")
    print(f"============================================")
    print(f"Successfully converted: {success_count} trajectories ({converted_duration/3600:.2f} hours)")
    print(f"Skipped: {skip_count} trajectories")
    if skip_reasons:
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1])[:3]:
            print(f"  - {reason}: {count}")
    print(f"Failed: {fail_count} trajectories")
    print(f"Output directory: {output_dir}")
    print(f"Conversion report: {report_path}")
    print(f"============================================")
    
    # ---- Data-quality check ----
    print(f"\n==========================================")
    print(f"Data-quality check")
    print(f"==========================================")
    
    # Collect all HDF5 files, including previously converted outputs.
    all_hdf5 = sorted(output_dir.rglob('*.hdf5'))
    if all_hdf5:
        print(f"Checking {len(all_hdf5)} HDF5 files...")
        quality_results = []
        for hdf5_path in tqdm(all_hdf5, desc="Quality check"):
            quality_results.append(check_trajectory_quality(hdf5_path))
        generate_quality_report(output_dir, quality_results)
    else:
        print("No HDF5 files found; skipping the quality check.")


def main():
    parser = argparse.ArgumentParser(description='Convert the TouchAnything dataset to HDF5')
    parser.add_argument('--traj_path', type=str, help='Path to a single trajectory')
    parser.add_argument('--root_path', type=str, help='Dataset root directory (batch mode)')
    parser.add_argument('--output', type=str,
                       help='Output path (single trajectory) or output directory (batch mode); defaults to a new directory under datasets/')
    parser.add_argument('--batch', action='store_true', help='Enable batch-conversion mode')
    parser.add_argument('--bad_files_list', type=str, default=None,
                       help='Path to bad_files.txt (format: rel_path.hdf5 | reason). If set, only regenerate these trajectories (batch mode only).')
    parser.add_argument('--bad_reason_filter', type=str, default=None,
                       help='Comma-separated reason tags to regenerate from bad_files_list, e.g. read_error,too_short')
    parser.add_argument('--delete_before_regen', action='store_true',
                       help='When using --bad_files_list, delete existing output HDF5 before regenerating')
    parser.add_argument('--compression', type=str, default='gzip',
                       choices=['gzip', 'lzf', 'none'], help='Compression algorithm')
    parser.add_argument('--compression_level', type=int, default=9,
                       help='Compression level (1-9 for gzip, default 9 for best compression)')
    parser.add_argument('--traj_workers', type=int, default=32,
                       help='Batch mode: number of worker processes for trajectory conversion (default: 32)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip HDF5 files that already exist in the output directory')

    args = parser.parse_args()

    compression = None if args.compression == 'none' else args.compression

    if args.batch:
        if not args.root_path:
            print("Error: batch mode requires --root_path")
            sys.exit(1)

        limit_to_trajs = None
        if args.bad_files_list is not None:
            limit_to_trajs = _parse_bad_files_list(Path(args.bad_files_list), args.bad_reason_filter)
            if len(limit_to_trajs) == 0:
                print("No trajectories selected from bad_files_list; exiting.")
                return
        batch_convert(
            args.root_path, args.output,
            compression, args.compression_level,
            traj_workers=args.traj_workers,
            skip_existing=args.skip_existing,
            limit_to_trajs=limit_to_trajs,
            delete_before_regen=bool(args.delete_before_regen),
        )
    else:
        if not args.traj_path:
            print("Error: single-trajectory mode requires --traj_path")
            sys.exit(1)
        
        convert_trajectory_to_hdf5(
            args.traj_path, args.output,
            compression, args.compression_level,
            skip_existing=args.skip_existing,
        )


if __name__ == '__main__':
    main()
