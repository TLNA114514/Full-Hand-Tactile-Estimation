import torch
import numpy as np
from typing import Optional


def _load_bend_sensor_positions():
    """
    Load bend-sensor positions from the config file
    
    Returns:
        dict: {'left': set of (row, col), 'right': set of (row, col)}
    """
    import json
    from pathlib import Path
    
    # Find the config file by walking two levels up from src/utils to the project root
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'hand_joint_positions.json'
    
    if not config_path.exists():
        # Return empty sets if the config file is missing
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


def compute_mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Mean Per Joint Position Error (MPJPE).
    
    Args:
        pred: (B, T, J, 3) predicted poses
        target: (B, T, J, 3) ground truth poses
        mask: (B, T, J) optional mask for valid joints
    
    Returns:
        mpjpe: mean error in meters
    """
    # Compute L2 distance
    error = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # (B, T, J)
    
    if mask is not None:
        error = error * mask
        mpjpe = error.sum() / mask.sum()
    else:
        mpjpe = error.mean()
    
    return mpjpe.item()


def compute_pa_mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Procrustes Aligned MPJPE.
    Aligns prediction to target using optimal rotation, translation, and scale.
    
    Args:
        pred: (B, T, J, 3) predicted poses
        target: (B, T, J, 3) ground truth poses
        mask: (B, T, J) optional mask for valid joints
    
    Returns:
        pa_mpjpe: aligned mean error in meters
    """
    B, T, J, _ = pred.shape
    
    total_error = 0.0
    total_count = 0
    
    for b in range(B):
        for t in range(T):
            pred_frame = pred[b, t].cpu().numpy()  # (J, 3)
            target_frame = target[b, t].cpu().numpy()  # (J, 3)
            
            # Apply mask: only use valid joints for alignment and error
            if mask is not None:
                valid = mask[b, t].cpu().numpy() > 0.5
                if valid.sum() < 3:  # Need at least 3 points for Procrustes
                    continue
                pred_valid = pred_frame[valid]
                target_valid = target_frame[valid]
            else:
                pred_valid = pred_frame
                target_valid = target_frame
            
            # Align using Procrustes
            aligned_pred = procrustes_align(pred_valid, target_valid)
            
            # Compute error on valid joints only
            error = np.sqrt(((aligned_pred - target_valid) ** 2).sum(axis=-1))  # (J_valid,)
            total_error += error.sum()
            total_count += len(error)
    
    return total_error / max(total_count, 1)


def procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align prediction to target using Procrustes analysis (rotation + translation + scale).
    
    Args:
        pred: (J, 3) predicted poses
        target: (J, 3) ground truth poses
    
    Returns:
        aligned_pred: (J, 3) aligned prediction
    """
    # Center the points
    mu_pred = pred.mean(axis=0)
    mu_target = target.mean(axis=0)
    pred_centered = pred - mu_pred
    target_centered = target - mu_target
    
    # Compute optimal rotation using SVD
    H = pred_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute optimal scale
    pred_var = (pred_centered ** 2).sum()
    if pred_var > 1e-10:
        scale = np.trace(np.diag(S) @ Vt @ U.T) / pred_var
        # Clamp scale to reasonable range to avoid degenerate cases
        scale = np.clip(scale, 0.1, 10.0)
    else:
        scale = 1.0
    
    # Apply scale, rotation, and translation
    aligned_pred = scale * (pred_centered @ R) + mu_target
    
    return aligned_pred


def compute_pck(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.05, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute Percentage of Correct Keypoints (PCK).
    
    Args:
        pred: (B, T, J, 3) predicted poses
        target: (B, T, J, 3) ground truth poses
        threshold: distance threshold in meters (default: 5cm)
        mask: (B, T, J) optional mask
    
    Returns:
        pck: percentage of correct keypoints
    """
    # Compute L2 distance
    error = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # (B, T, J)
    
    # Check if within threshold
    correct = (error < threshold).float()
    
    if mask is not None:
        correct = correct * mask
        pck = correct.sum() / mask.sum()
    else:
        pck = correct.mean()
    
    return pck.item()


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, confidences: Optional[torch.Tensor] = None, threshold: float = 0.05) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        pred: (B, T, J, 3) predicted poses
        target: (B, T, J, 3) ground truth poses
        confidences: (B, T, J) confidence scores
        threshold: PCK threshold
    
    Returns:
        metrics: dict of all metrics
    """
    # Use confidence as mask if provided
    mask = None
    if confidences is not None:
        mask = (confidences > 0.5).float()
    
    metrics = {
        'mpjpe': compute_mpjpe(pred, target, mask),
        'pa_mpjpe': compute_pa_mpjpe(pred, target, mask),
        'pck': compute_pck(pred, target, threshold, mask),
    }
    
    return metrics


# ============================================================
# Tactile Pressure Metrics (following PressureVision paper)
# ============================================================

def temporal_accuracy(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.05, min_contact_ratio: float = 0.05, exclude_bend_sensors: bool = True) -> float:
    """
    Temporal Accuracy: Evaluates temporal accuracy of contact onset and termination.
    Frame is marked as in contact if: (1) pressure > threshold AND (2) contact points ratio > min_contact_ratio.
    Frame is correct if presence of contact is consistent.
    
    WARNING: This metric may be inflated if data is imbalanced (e.g., 95% frames have contact).
    Consider using temporal_f1_score() for a more balanced evaluation.
    
    Args:
        pred: (T, H, W) or (T, 2, H, W) predicted pressure maps
        gt: (T, H, W) or (T, 2, H, W) ground truth pressure maps
        threshold: pressure threshold to determine contact (default: 0.05)
        min_contact_ratio: minimum ratio of contact points to valid points (default: 0.05 = 5%)
        exclude_bend_sensors: whether to exclude finger bend sensors (default: True)
    
    Returns:
        accuracy: fraction of frames with correct contact presence [0, 1]
    """
    def check_contact_strict_frame(frame):
        """Strict contact detection: exclude bend sensors + threshold + ratio"""
        if np.all(np.isnan(frame)):
            return False
        
        # Get pressure-map size
        size = frame.shape[-1]
        
        # Create bend-sensor masks
        if exclude_bend_sensors:
            if frame.ndim == 3:  # (2, H, W) - two hands, apply left and right masks separately
                left_mask = _create_bend_sensor_mask(size, hand='left')
                right_mask = _create_bend_sensor_mask(size, hand='right')
                bend_mask = np.stack([left_mask, right_mask], axis=0)
            else:  # (H, W) - single hand, use the merged mask
                bend_mask = _create_bend_sensor_mask(size, hand='both')
        else:
            bend_mask = np.zeros_like(frame, dtype=bool)
        
        # Compute valid tactile sensor regions, excluding NaNs and bend sensors
        valid_mask = ~np.isnan(frame) & ~bend_mask
        num_valid = np.sum(valid_mask)
        
        if num_valid == 0:
            return False
        
        # Count above-threshold points only in valid tactile sensor regions
        above_threshold = (frame > threshold) & valid_mask
        num_contact = np.sum(above_threshold)
        contact_ratio = num_contact / num_valid
        return contact_ratio >= min_contact_ratio
    
    # Handle both single-hand (T, H, W) and bilateral (T, 2, H, W) formats
    pred_contact = np.array([check_contact_strict_frame(pred[t]) for t in range(len(pred))])
    gt_contact = np.array([check_contact_strict_frame(gt[t]) for t in range(len(gt))])
    
    # Frames are correct if contact presence matches
    correct = (pred_contact == gt_contact)
    
    # Only count frames where we have valid GT (not all NaN)
    if pred.ndim == 4:
        valid = ~np.all(np.isnan(gt), axis=(1, 2, 3))
    else:
        valid = ~np.all(np.isnan(gt), axis=(1, 2))
    
    if valid.sum() == 0:
        return np.nan
    
    return correct[valid].mean()


def temporal_f1_score(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.05, min_contact_ratio: float = 0.05, exclude_bend_sensors: bool = True) -> dict:
    """
    Temporal F1 Score: More balanced metric for contact presence detection.
    Computes precision, recall, and F1 for binary contact classification.
    
    This metric is less affected by class imbalance than temporal_accuracy.
    
    Args:
        pred: (T, H, W) or (T, 2, H, W) predicted pressure maps
        gt: (T, H, W) or (T, 2, H, W) ground truth pressure maps
        threshold: pressure threshold to determine contact
        min_contact_ratio: minimum ratio of contact points to valid points (default: 0.05 = 5%)
        exclude_bend_sensors: whether to exclude finger bend sensors (default: True)
    
    Returns:
        dict with keys: precision, recall, f1_score, support (number of contact frames in GT)
    """
    def check_contact_strict_frame(frame):
        """Strict contact detection: exclude bend sensors + threshold + ratio"""
        if np.all(np.isnan(frame)):
            return False
        
        size = frame.shape[-1]
        
        if exclude_bend_sensors:
            if frame.ndim == 3:  # (2, H, W) - two hands, apply left and right masks separately
                left_mask = _create_bend_sensor_mask(size, hand='left')
                right_mask = _create_bend_sensor_mask(size, hand='right')
                bend_mask = np.stack([left_mask, right_mask], axis=0)
            else:  # (H, W) - single hand, use the merged mask
                bend_mask = _create_bend_sensor_mask(size, hand='both')
        else:
            bend_mask = np.zeros_like(frame, dtype=bool)
        
        valid_mask = ~np.isnan(frame) & ~bend_mask
        num_valid = np.sum(valid_mask)
        if num_valid == 0:
            return False
        above_threshold = (frame > threshold) & valid_mask
        num_contact = np.sum(above_threshold)
        contact_ratio = num_contact / num_valid
        return contact_ratio >= min_contact_ratio
    
    # Get binary contact labels using strict check
    pred_contact = np.array([check_contact_strict_frame(pred[t]) for t in range(len(pred))])
    gt_contact = np.array([check_contact_strict_frame(gt[t]) for t in range(len(gt))])
    
    if pred.ndim == 4:
        valid = ~np.all(np.isnan(gt), axis=(1, 2, 3))
    else:
        valid = ~np.all(np.isnan(gt), axis=(1, 2))
    
    if valid.sum() == 0:
        return {'precision': np.nan, 'recall': np.nan, 'f1_score': np.nan, 'support': 0}
    
    pred_contact = pred_contact[valid]
    gt_contact = gt_contact[valid]
    
    # Compute TP, FP, FN
    tp = np.sum((pred_contact == True) & (gt_contact == True))
    fp = np.sum((pred_contact == True) & (gt_contact == False))
    fn = np.sum((pred_contact == False) & (gt_contact == True))
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': int(gt_contact.sum()),  # Number of contact frames in GT
    }


def temporal_onset_offset_accuracy(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.05, tolerance: int = 2, min_contact_ratio: float = 0.05, exclude_bend_sensors: bool = True) -> dict:
    """
    Temporal Onset/Offset Detection Accuracy: Stricter metric for temporal modeling.
    Detects contact onset (no-contact → contact) and offset (contact → no-contact) events.
    
    An onset/offset is considered correct if detected within ±tolerance frames.
    
    Args:
        pred: (T, H, W) or (T, 2, H, W) predicted pressure maps
        gt: (T, H, W) or (T, 2, H, W) ground truth pressure maps
        threshold: pressure threshold to determine contact
        tolerance: allowed frame deviation for onset/offset detection (default: 2 frames)
        min_contact_ratio: minimum ratio of contact points to valid points (default: 0.05 = 5%)
        exclude_bend_sensors: whether to exclude finger bend sensors (default: True)
    
    Returns:
        dict with keys:
            - onset_precision: fraction of predicted onsets that are correct
            - onset_recall: fraction of GT onsets that are detected
            - offset_precision: fraction of predicted offsets that are correct
            - offset_recall: fraction of GT offsets that are detected
            - num_gt_onsets: number of onset events in GT
            - num_gt_offsets: number of offset events in GT
    """
    def check_contact_strict_frame(frame):
        """Strict contact detection: exclude bend sensors + threshold + ratio"""
        if np.all(np.isnan(frame)):
            return False
        
        size = frame.shape[-1]
        
        if exclude_bend_sensors:
            if frame.ndim == 3:  # (2, H, W) - two hands, apply left and right masks separately
                left_mask = _create_bend_sensor_mask(size, hand='left')
                right_mask = _create_bend_sensor_mask(size, hand='right')
                bend_mask = np.stack([left_mask, right_mask], axis=0)
            else:  # (H, W) - single hand, use the merged mask
                bend_mask = _create_bend_sensor_mask(size, hand='both')
        else:
            bend_mask = np.zeros_like(frame, dtype=bool)
        
        valid_mask = ~np.isnan(frame) & ~bend_mask
        num_valid = np.sum(valid_mask)
        if num_valid == 0:
            return False
        above_threshold = (frame > threshold) & valid_mask
        num_contact = np.sum(above_threshold)
        contact_ratio = num_contact / num_valid
        return contact_ratio >= min_contact_ratio
    
    # Get binary contact labels using strict check
    pred_contact = np.array([check_contact_strict_frame(pred[t]) for t in range(len(pred))])
    gt_contact = np.array([check_contact_strict_frame(gt[t]) for t in range(len(gt))])
    
    if pred.ndim == 4:
        valid = ~np.all(np.isnan(gt), axis=(1, 2, 3))
    else:
        valid = ~np.all(np.isnan(gt), axis=(1, 2))
    
    if valid.sum() < 2:  # Need at least 2 frames for transitions
        return {
            'onset_precision': np.nan, 'onset_recall': np.nan,
            'offset_precision': np.nan, 'offset_recall': np.nan,
            'num_gt_onsets': 0, 'num_gt_offsets': 0
        }
    
    pred_contact = pred_contact[valid]
    gt_contact = gt_contact[valid]
    
    # Detect onset (False → True) and offset (True → False) events
    def detect_transitions(contact_seq):
        onsets = []
        offsets = []
        for t in range(1, len(contact_seq)):
            if not contact_seq[t-1] and contact_seq[t]:
                onsets.append(t)
            elif contact_seq[t-1] and not contact_seq[t]:
                offsets.append(t)
        return np.array(onsets), np.array(offsets)
    
    gt_onsets, gt_offsets = detect_transitions(gt_contact)
    pred_onsets, pred_offsets = detect_transitions(pred_contact)
    
    # Match predicted events to GT events within tolerance
    def match_events(pred_events, gt_events, tolerance):
        if len(gt_events) == 0:
            return 0, 0  # No GT events
        if len(pred_events) == 0:
            return 0, len(gt_events)  # No predictions
        
        matched_gt = set()
        matched_pred = 0
        
        for pred_t in pred_events:
            # Check if within tolerance of any GT event
            for gt_t in gt_events:
                if abs(pred_t - gt_t) <= tolerance and gt_t not in matched_gt:
                    matched_gt.add(gt_t)
                    matched_pred += 1
                    break
        
        return matched_pred, len(gt_events)
    
    # Onset metrics
    onset_matched, num_gt_onsets = match_events(pred_onsets, gt_onsets, tolerance)
    onset_precision = onset_matched / len(pred_onsets) if len(pred_onsets) > 0 else 0.0
    onset_recall = onset_matched / num_gt_onsets if num_gt_onsets > 0 else 0.0
    
    # Offset metrics
    offset_matched, num_gt_offsets = match_events(pred_offsets, gt_offsets, tolerance)
    offset_precision = offset_matched / len(pred_offsets) if len(pred_offsets) > 0 else 0.0
    offset_recall = offset_matched / num_gt_offsets if num_gt_offsets > 0 else 0.0
    
    return {
        'onset_precision': onset_precision,
        'onset_recall': onset_recall,
        'offset_precision': offset_precision,
        'offset_recall': offset_recall,
        'num_gt_onsets': int(num_gt_onsets),
        'num_gt_offsets': int(num_gt_offsets),
    }


def contact_iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.05) -> float:
    """
    Contact IoU: Spatial and temporal accuracy via IoU of binary contact maps.
    Does not consider pressure magnitude.
    
    Args:
        pred: (T, H, W) or (T, 2, H, W) predicted pressure maps
        gt: (T, H, W) or (T, 2, H, W) ground truth pressure maps
        threshold: pressure threshold for binarization
    
    Returns:
        iou: intersection over union [0, 1]
    """
    # Binarize
    pred_binary = (pred > threshold).astype(np.float32)
    gt_binary = (gt > threshold).astype(np.float32)
    
    # Mask out NaN regions in GT
    valid_mask = ~np.isnan(gt)
    pred_binary = pred_binary * valid_mask
    gt_binary = gt_binary * valid_mask
    
    # Compute IoU
    intersection = (pred_binary * gt_binary).sum()
    union = ((pred_binary + gt_binary) > 0).sum()
    
    if union == 0:
        return np.nan
    
    return intersection / union


def volumetric_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Volumetric IoU: Extends Contact IoU to evaluate pressure magnitudes.
    Each 2D pressure image is treated as a 3D volume where height = pressure.
    
    Formula: IoU_vol = sum(min(P, P_hat)) / sum(max(P, P_hat))
    
    Args:
        pred: (T, H, W) or (T, 2, H, W) predicted pressure maps
        gt: (T, H, W) or (T, 2, H, W) ground truth pressure maps
    
    Returns:
        iou_vol: volumetric intersection over union [0, 1]
    """
    # Mask out NaN regions
    valid_mask = ~np.isnan(gt)
    pred_valid = np.where(valid_mask, pred, 0)
    gt_valid = np.where(valid_mask, gt, 0)
    
    # Compute volumetric IoU
    intersection = np.minimum(pred_valid, gt_valid).sum()
    union = np.maximum(pred_valid, gt_valid).sum()
    
    if union == 0:
        return np.nan
    
    return intersection / union


def pressure_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Mean Absolute Error: Evaluate accuracy of pressure in physical units.
    Computed over valid (non-NaN) pixels only.
    
    Args:
        pred: (T, H, W) or (T, 2, H, W) predicted pressure maps
        gt: (T, H, W) or (T, 2, H, W) ground truth pressure maps
    
    Returns:
        mae: mean absolute error
    """
    valid_mask = ~np.isnan(gt)
    
    if valid_mask.sum() == 0:
        return np.nan
    
    errors = np.abs(pred[valid_mask] - gt[valid_mask])
    return errors.mean()


def compute_tactile_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.05, min_contact_ratio: float = 0.05, exclude_bend_sensors: bool = True) -> dict:
    """
    Compute all tactile pressure metrics following the paper.
    
    Args:
        pred: (T, 2, H, W) predicted bilateral pressure maps [left, right]
        gt: (T, 2, H, W) ground truth bilateral pressure maps
        threshold: contact threshold for binary metrics
        min_contact_ratio: minimum ratio of contact points to valid points (default: 0.05 = 5%)
        exclude_bend_sensors: whether to exclude finger bend sensors from contact detection (default: True)
    
    Returns:
        metrics: dict with keys:
            - temporal_accuracy: temporal contact accuracy (WARNING: may be inflated)
            - temporal_f1: F1 score for contact detection (more balanced)
            - temporal_precision: precision for contact detection
            - temporal_recall: recall for contact detection
            - onset_precision: precision for contact onset detection
            - onset_recall: recall for contact onset detection
            - offset_precision: precision for contact offset detection
            - offset_recall: recall for contact offset detection
            - contact_iou: spatial/temporal IoU of binary contact
            - volumetric_iou: IoU considering pressure magnitude
            - mae: mean absolute error in pressure units
            - mae_left: MAE for left hand only
            - mae_right: MAE for right hand only
    """
    # Basic metrics
    metrics = {
        'temporal_accuracy': temporal_accuracy(pred, gt, threshold, min_contact_ratio, exclude_bend_sensors),
        'contact_iou': contact_iou(pred, gt, threshold),
        'volumetric_iou': volumetric_iou(pred, gt),
        'mae': pressure_mae(pred, gt),
    }
    
    # Temporal F1 score (more balanced than accuracy)
    f1_results = temporal_f1_score(pred, gt, threshold, min_contact_ratio, exclude_bend_sensors)
    metrics['temporal_f1'] = f1_results['f1_score']
    metrics['temporal_precision'] = f1_results['precision']
    metrics['temporal_recall'] = f1_results['recall']
    metrics['temporal_support'] = f1_results['support']
    
    # Onset/Offset detection (stricter temporal metric)
    onset_offset_results = temporal_onset_offset_accuracy(pred, gt, threshold, tolerance=2, min_contact_ratio=min_contact_ratio, exclude_bend_sensors=exclude_bend_sensors)
    metrics['onset_precision'] = onset_offset_results['onset_precision']
    metrics['onset_recall'] = onset_offset_results['onset_recall']
    metrics['offset_precision'] = onset_offset_results['offset_precision']
    metrics['offset_recall'] = onset_offset_results['offset_recall']
    metrics['num_gt_onsets'] = onset_offset_results['num_gt_onsets']
    metrics['num_gt_offsets'] = onset_offset_results['num_gt_offsets']
    
    # Per-hand MAE
    if pred.ndim == 4 and pred.shape[1] == 2:
        metrics['mae_left'] = pressure_mae(pred[:, 0], gt[:, 0])
        metrics['mae_right'] = pressure_mae(pred[:, 1], gt[:, 1])
    
    return metrics
