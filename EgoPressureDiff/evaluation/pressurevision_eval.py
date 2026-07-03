#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Sequence, Optional

import numpy as np


right_mask_indexed = np.array([
        [240, 239, 238,   0, 237, 236, 235,   0, 234, 233, 232,   0, 231, 230, 229,   0, 228, 227, 226],
        [256, 255, 254,   0, 253, 252, 251,   0, 250, 249, 248,   0, 247, 246, 245,   0, 244, 243, 242],
        [ 16,  15,  14,   0,  13,  12,  11,   0,  10,   9,   8,   0,   7,   6,   5,   0,   4,   3,   2],
        [ 32,  31,  30,   0,  29,  28,  27,   0,  26,  25,  24,   0,  23,  22,  21,   0,  20,  19,  18],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [ 47,   0,   0,   0,   0,  44,   0,   0,   0,  41,   0,   0,   0,  38,   0,   0,   0,  35,   0],
        [ 47,   0,   0,   0,   0,  44,   0,   0,   0,  41,   0,   0,   0,  38,   0,   0,   0,  35,   0],
        [  0,  47,   0,   0,   0,  44,   0,   0,   0,  41,   0,   0,   0,  38,   0,   0,   0,  35,   0],
        [  0,  47,   0,   0,   0,  44,   0,   0,   0,  41,   0,   0,   0,  38,   0,   0,   0,  35,   0],
        [  0,   0,  47,   0,   0,  44,   0,   0,   0,  41,   0,   0,   0,  38,   0,   0,   0,  35,   0],
        [  0,   0,  47,   0,   0,  44,   0,   0,   0,  41,   0,   0,   0,  38,   0,   0,   0,  35,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,  61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50],
        [  0,   0,   0,   0,  80,  79,  78,  77,  76,  75,  74,  73,  72,  71,  70,  69,  68,  67,  66],
        [  0,   0,   0,   0,  96,  95,  94,  93,  92,  91,  90,  89,  88,  87,  86,  85,  84,  83,  82],
        [  0,   0,   0,   0, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100,  99,  98],
        [  0,   0,   0,   0, 128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114],
    ], dtype=np.int32)

left_mask_indexed = np.array([
        [ 31,  30,  29,   0,  28,  27,  26,   0,  25,  24,  23,   0,  22,  21,  20,   0,  19,  18,  17],
        [ 15,  14,  13,   0,  12,  11,  10,   0,   9,   8,   7,   0,   6,   5,   4,   0,   3,   2,   1],
        [255, 254, 253,   0, 252, 251, 250,   0, 249, 248, 247,   0, 246, 245, 244,   0, 243, 242, 241],
        [239, 238, 237,   0, 236, 235, 234,   0, 233, 232, 231,   0, 230, 229, 228,   0, 227, 226, 225],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0, 222,   0,   0,   0, 219,   0,   0,   0, 216,   0,   0,   0, 213,   0,   0,   0,   0, 210],
        [  0, 222,   0,   0,   0, 219,   0,   0,   0, 216,   0,   0,   0, 213,   0,   0,   0,   0, 210],
        [  0, 222,   0,   0,   0, 219,   0,   0,   0, 216,   0,   0,   0, 213,   0,   0,   0, 210,   0],
        [  0, 222,   0,   0,   0, 219,   0,   0,   0, 216,   0,   0,   0, 213,   0,   0,   0, 210,   0],
        [  0, 222,   0,   0,   0, 219,   0,   0,   0, 216,   0,   0,   0, 213,   0,   0, 210,   0,   0],
        [  0, 222,   0,   0,   0, 219,   0,   0,   0, 216,   0,   0,   0, 213,   0,   0, 210,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196,   0,   0,   0,   0,   0,   0,   0],
        [191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177,   0,   0,   0,   0],
        [175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161,   0,   0,   0,   0],
        [159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145,   0,   0,   0,   0],
        [143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129,   0,   0,   0,   0],
    ], dtype=np.int32)


def get_valid_indices_from_mask(mask_indexed: np.ndarray) -> np.ndarray:
    used_1based = np.unique(mask_indexed[mask_indexed > 0])
    if used_1based.size == 0:
        raise ValueError("No valid indices found in mask_indexed (all zeros?).")
    used_0based = used_1based.astype(np.int32) - 1
    used_0based = used_0based[(used_0based >= 0) & (used_0based < 256)]
    used_0based = np.unique(used_0based)
    return used_0based


LH_FINGERS_12 = {
    "little": [31,30,29, 15,14,13, 255,254,253, 239,238,237],
    "ring"  : [28,27,26, 12,11,10, 252,251,250, 236,235,234],
    "middle": [25,24,23,  9, 8, 7, 249,248,247, 233,232,231],
    "index" : [22,21,20,  6, 5, 4, 246,245,244, 230,229,228],
    "thumb" : [19,18,17,  3, 2, 1, 243,242,241, 227,226,225],
}
LH_BENDS_5 = {
    "little": [222,222,222,222,222,222],
    "ring"  : [219,219,219,219,219,219],
    "middle": [216,216,216,216,216,216],
    "index" : [213,213,213,213,213,213],
    "thumb" : [210,210,210,210,210,210],
}

RH_FINGERS_12 = {
    "little": [228,227,226, 244,243,242,  4, 3, 2, 20,19,18],
    "ring"  : [231,230,229, 247,246,245,  7, 6, 5, 23,22,21],
    "middle": [234,233,232, 250,249,248, 10, 9, 8, 26,25,24],
    "thumb" : [240,239,238, 256,255,254, 16,15,14, 32,31,30],
    "index" : [237,236,235, 253,252,251, 13,12,11, 29,28,27],
}
RH_BENDS_5 = {
    "little": [35,35,35,35,35,35],
    "ring"  : [38,38,38,38,38,38],
    "middle": [41,41,41,41,41,41],
    "index" : [44,44,44,44,44,44],
    "thumb" : [47,47,47,47,47,47],
}


def _to_zero_based(idxs_1based: Sequence[int]) -> List[int]:
    return sorted({int(i) - 1 for i in idxs_1based if 1 <= int(i) <= 256})


def build_finger_index_map(mask_side: str) -> Dict[str, List[int]]:
    if mask_side == "right":
        fingers_12 = RH_FINGERS_12
        bends_5 = RH_BENDS_5
    elif mask_side == "left":
        fingers_12 = LH_FINGERS_12
        bends_5 = LH_BENDS_5
    else:
        raise ValueError("--mask_side must be 'right' or 'left'")

    out: Dict[str, List[int]] = {}
    for finger in ["thumb", "index", "middle", "ring", "little"]:
        out[finger] = _to_zero_based(list(fingers_12[finger]) + list(bends_5[finger]))
    return out


def apply_hand_contact_rule_per_frame(
    seq_t256: np.ndarray,
    finger_index_map: Dict[str, List[int]],
    value_threshold: float = 10.0,
    finger_contact_min_points: int = 3,
    total_contact_min_points: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    seq_t256 = np.asarray(seq_t256, dtype=np.float64)
    if seq_t256.ndim != 2 or seq_t256.shape[1] != 256:
        raise ValueError(f"Expected shape (T,256), got {seq_t256.shape}")

    T = seq_t256.shape[0]
    out = seq_t256.copy()
    flags = np.zeros((T,), dtype=bool)

    finger_names = ["thumb", "index", "middle", "ring", "little"]

    for t in range(T):
        frame = out[t]
        finger_counts = []
        for finger in finger_names:
            idxs = finger_index_map[finger]
            cnt = int(np.sum(frame[idxs] > value_threshold))
            finger_counts.append(cnt)

        cond1 = any(cnt >= finger_contact_min_points for cnt in finger_counts)
        total_cnt = int(np.sum(finger_counts))
        cond2 = total_cnt >= total_contact_min_points
        hand_in_contact = cond1 or cond2

        flags[t] = hand_in_contact
        if not hand_in_contact:
            out[t, :] = 0.0

    return out, flags


def metric_mae(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def metric_vol_iou(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float]:
    min_sum = float(np.sum(np.minimum(gt, pred)))
    max_sum = float(np.sum(np.maximum(gt, pred)))
    if max_sum == 0.0:
        return 1.0, min_sum, max_sum
    return (min_sum / max_sum), min_sum, max_sum


def metric_contact_iou(gt: np.ndarray, pred: np.ndarray, contact_thresh: float) -> Tuple[float, int, int]:
    gt_c = gt > contact_thresh
    pr_c = pred > contact_thresh
    inter = int(np.sum(gt_c & pr_c))
    union = int(np.sum(gt_c | pr_c))
    if union == 0:
        return 1.0, inter, union
    return (inter / union), inter, union


def metric_temporal_accuracy_hand_contact(gt_in: np.ndarray, pred_in: np.ndarray) -> Tuple[float, int]:
    if gt_in.shape != pred_in.shape:
        raise ValueError("gt_in and pred_in must have the same shape")
    T = gt_in.shape[0]
    correct = int(np.sum(gt_in == pred_in))
    acc = correct / T if T > 0 else 0.0
    return float(acc), correct


def load_t256_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path} does not contain a numpy.ndarray")
    if arr.ndim != 2 or arr.shape[1] != 256:
        raise ValueError(f"{path} must have shape (T, 256), got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def select_mask(mask_side: str) -> np.ndarray:
    if mask_side == "right":
        return right_mask_indexed
    if mask_side == "left":
        return left_mask_indexed
    raise ValueError("--mask_side must be 'right' or 'left'")


def evaluate_single_pair(
    gt_path: str,
    pred_path: str,
    mask_side: str,
    metrics: List[str],
    contact_thresh: float,
    hand_contact_value_threshold: float,
    finger_contact_min_points: int,
    total_contact_min_points: int,
) -> Dict[str, Any]:
    gt_full = load_t256_npy(gt_path)
    pred_full = load_t256_npy(pred_path)

    if gt_full.shape[0] != pred_full.shape[0]:
        raise ValueError(f"T mismatch: gt has {gt_full.shape[0]} frames, pred has {pred_full.shape[0]} frames")

    finger_map = build_finger_index_map(mask_side=mask_side)
    gt_full_filtered, gt_in = apply_hand_contact_rule_per_frame(
        gt_full,
        finger_index_map=finger_map,
        value_threshold=hand_contact_value_threshold,
        finger_contact_min_points=finger_contact_min_points,
        total_contact_min_points=total_contact_min_points,
    )
    pred_full_filtered, pred_in = apply_hand_contact_rule_per_frame(
        pred_full,
        finger_index_map=finger_map,
        value_threshold=hand_contact_value_threshold,
        finger_contact_min_points=finger_contact_min_points,
        total_contact_min_points=total_contact_min_points,
    )

    mask_indexed = select_mask(mask_side)
    valid_idx = get_valid_indices_from_mask(mask_indexed)

    gt = gt_full_filtered[:, valid_idx]
    pred = pred_full_filtered[:, valid_idx]

    out: Dict[str, Any] = {}
    metrics_set = set(metrics)
    if "all" in metrics_set:
        metrics_set = {"mae", "vol_iou", "contact_iou", "temporal_accuracy"}

    if "mae" in metrics_set:
        out["mae"] = metric_mae(gt, pred)

    if "vol_iou" in metrics_set:
        vol_iou, min_sum, max_sum = metric_vol_iou(gt, pred)
        out["vol_iou"] = vol_iou
        out["vol_iou_min_sum"] = min_sum
        out["vol_iou_max_sum"] = max_sum

    if "contact_iou" in metrics_set:
        c_iou, inter, union = metric_contact_iou(gt, pred, contact_thresh=contact_thresh)
        out["contact_iou"] = c_iou
        out["contact_iou_inter_sum"] = inter
        out["contact_iou_union_sum"] = union

    if "temporal_accuracy" in metrics_set:
        t_acc, correct = metric_temporal_accuracy_hand_contact(gt_in, pred_in)
        out["temporal_accuracy"] = t_acc
        out["temporal_correct_frames"] = correct

    out["meta"] = {
        "gt_path": os.path.abspath(gt_path),
        "pred_path": os.path.abspath(pred_path),
        "frames_T": int(gt_full.shape[0]),
        "valid_channels": int(valid_idx.size),
        "mask_side": mask_side,
        "contact_thresh": float(contact_thresh),
        "hand_contact_value_threshold": float(hand_contact_value_threshold),
        "finger_contact_min_points": int(finger_contact_min_points),
        "total_contact_min_points": int(total_contact_min_points),
        "stats_gt_contact_frames": int(np.sum(gt_in)),
        "stats_pred_contact_frames": int(np.sum(pred_in)),
    }
    return out


def list_npy_files(root: str, recursive: bool = True) -> List[str]:
    root = os.path.abspath(root)
    out: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".npy"):
                    out.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(".npy"):
                out.append(os.path.join(root, fn))
    out.sort()
    return out


def build_gt_basename_index(gt_dir: str) -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for p in list_npy_files(gt_dir, recursive=True):
        b = os.path.basename(p)
        idx.setdefault(b, []).append(p)
    return idx


def match_gt_path(gt_dir: str, pred_dir: str, pred_path: str, gt_basename_index: Dict[str, List[str]]) -> Optional[str]:
    rel = os.path.relpath(pred_path, start=os.path.abspath(pred_dir))
    cand = os.path.join(os.path.abspath(gt_dir), rel)
    if os.path.isfile(cand):
        return cand

    b = os.path.basename(pred_path)
    hits = gt_basename_index.get(b, [])
    if len(hits) == 1:
        return hits[0]
    return None


def robust_z_scores(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scaled = 1.4826 * mad
    if scaled <= 1e-12:
        return np.zeros_like(values), med, scaled
    z = (values - med) / scaled
    return z, med, scaled


@dataclass
class PairResult:
    pred_path: str
    gt_path: str
    metrics: Dict[str, float]


def mean_metrics(pairs: List[PairResult], keys: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in keys:
        vals = [p.metrics[k] for p in pairs if k in p.metrics and np.isfinite(p.metrics[k])]
        out[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    return out


def format_metrics_line(d: Dict[str, float], keys: List[str]) -> str:
    parts = []
    for k in keys:
        v = d.get(k, float("nan"))
        if np.isnan(v):
            parts.append(f"{k}=nan")
        else:
            parts.append(f"{k}={v:.6f}")
    return "  " + " | ".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluate T×256 pressure sequences from pred folder against gt folder (PressureVision-style metrics)."
    )

    p.add_argument("--gt", default="", help="GT folder path (contains .npy).")
    p.add_argument("--pred", default="", help="Pred folder path (contains .npy).")

    p.add_argument("--recursive", action="store_true", help="Recursively scan pred/gt folders for .npy (default: True).")
    p.set_defaults(recursive=False)

    p.add_argument("--mask_side", choices=["right", "left"], default="right")
    p.add_argument("--metrics", nargs="+", default=["all"],
                   help="mae vol_iou contact_iou temporal_accuracy or 'all' (default: all).")

    p.add_argument("--contact_thresh", type=float, default=10.0)

    p.add_argument("--hand_contact_value_threshold", type=float, default=10.0)
    p.add_argument("--finger_contact_min_points", type=int, default=3)
    p.add_argument("--total_contact_min_points", type=int, default=5)

    p.add_argument("--outlier_z", type=float, default=4.0,
                   help="Robust z-score threshold for outlier detection (default: 4.0).")
    p.add_argument("--max_print_outliers", type=int, default=50,
                   help="Max outlier pairs to print (default: 50).")

    p.add_argument("--save_json", type=str, default="",
                   help="If set, save per-pair results and summaries to this json path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if os.path.isfile(args.pred) and args.pred.lower().endswith(".npy"):
        if not (os.path.isfile(args.gt) and args.gt.lower().endswith(".npy")):
            raise ValueError("If --pred is a file, --gt must also be a file (.npy).")
        res = evaluate_single_pair(
            gt_path=args.gt,
            pred_path=args.pred,
            mask_side=args.mask_side,
            metrics=args.metrics,
            contact_thresh=args.contact_thresh,
            hand_contact_value_threshold=args.hand_contact_value_threshold,
            finger_contact_min_points=args.finger_contact_min_points,
            total_contact_min_points=args.total_contact_min_points,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    gt_dir = os.path.abspath(args.gt)
    pred_dir = os.path.abspath(args.pred)
    if not os.path.isdir(gt_dir):
        raise NotADirectoryError(f"--gt must be a directory: {gt_dir}")
    if not os.path.isdir(pred_dir):
        raise NotADirectoryError(f"--pred must be a directory: {pred_dir}")

    pred_files = list_npy_files(pred_dir, recursive=args.recursive)
    gt_basename_index = build_gt_basename_index(gt_dir)

    metrics_set = set(args.metrics)
    if "all" in metrics_set:
        metric_keys = ["mae", "vol_iou", "contact_iou", "temporal_accuracy"]
    else:
        metric_keys = [k for k in ["mae", "vol_iou", "contact_iou", "temporal_accuracy"] if k in metrics_set]
        if not metric_keys:
            raise ValueError(f"No valid metrics selected: {args.metrics}")

    paired: List[PairResult] = []
    missing: List[str] = []
    ambiguous: List[Tuple[str, str]] = []  # (pred, basename)

    for pred_path in pred_files:
        gt_path = match_gt_path(gt_dir, pred_dir, pred_path, gt_basename_index)
        if gt_path is None:
            b = os.path.basename(pred_path)
            hits = gt_basename_index.get(b, [])
            if len(hits) == 0:
                missing.append(pred_path)
            else:
                ambiguous.append((pred_path, b))
            continue

        try:
            res = evaluate_single_pair(
                gt_path=gt_path,
                pred_path=pred_path,
                mask_side=args.mask_side,
                metrics=list(metric_keys),
                contact_thresh=args.contact_thresh,
                hand_contact_value_threshold=args.hand_contact_value_threshold,
                finger_contact_min_points=args.finger_contact_min_points,
                total_contact_min_points=args.total_contact_min_points,
            )
            m = {k: float(res[k]) for k in metric_keys if k in res}
            paired.append(PairResult(pred_path=pred_path, gt_path=gt_path, metrics=m))
        except Exception as e:
            missing.append(f"{pred_path}  [EVAL_ERROR: {type(e).__name__}: {e}]")

    print("=" * 80)
    print("Batch Evaluation Summary")
    print("- pred_dir:", pred_dir)
    print("- gt_dir  :", gt_dir)
    print(f"- pred .npy found: {len(pred_files)}")
    print(f"- paired success : {len(paired)}")
    print(f"- missing/failed : {len(missing)}")
    print(f"- ambiguous match: {len(ambiguous)}")
    print("=" * 80)

    if len(paired) == 0:
        print("No paired samples to evaluate. Please check folder structure / filenames.")
        if missing:
            print("\nExamples of missing/failed:")
            for x in missing[:10]:
                print("  -", x)
        if ambiguous:
            print("\nExamples of ambiguous basename matches (multiple GT with same name):")
            for pred_path, b in ambiguous[:10]:
                print(f"  - pred: {pred_path} | basename: {b} | gt_candidates={len(gt_basename_index.get(b, []))}")
        return

    overall_mean = mean_metrics(paired, metric_keys)
    print("\nOverall mean (all paired):")
    print(format_metrics_line(overall_mean, metric_keys))

    values_by_key: Dict[str, np.ndarray] = {}
    for k in metric_keys:
        values_by_key[k] = np.array([p.metrics.get(k, np.nan) for p in paired], dtype=np.float64)

    outlier_flags = np.zeros((len(paired),), dtype=bool)
    outlier_reasons: List[List[str]] = [[] for _ in range(len(paired))]

    for k in metric_keys:
        vals = values_by_key[k]
        finite_mask = np.isfinite(vals)
        if np.sum(finite_mask) < 5:
            continue

        z, med, scaled_mad = robust_z_scores(vals[finite_mask])
        z_full = np.zeros_like(vals)
        z_full[finite_mask] = z

        if k == "mae":
            bad = z_full > float(args.outlier_z)
        else:
            bad = z_full < -float(args.outlier_z)

        for i, is_bad in enumerate(bad):
            if bool(is_bad):
                outlier_flags[i] = True
                outlier_reasons[i].append(f"{k} (robust_z={z_full[i]:+.2f})")

    outlier_indices = np.where(outlier_flags)[0].tolist()
    inlier_pairs = [p for i, p in enumerate(paired) if not outlier_flags[i]]

    if len(outlier_indices) > 0:
        print("\nDetected outliers (potential abnormal pairs):")
        to_print = outlier_indices[: int(args.max_print_outliers)]
        for i in to_print:
            p = paired[i]
            reasons = ", ".join(outlier_reasons[i]) if outlier_reasons[i] else "flagged"
            print(f"- {os.path.basename(p.pred_path)}")
            print(f"    pred: {p.pred_path}")
            print(f"    gt  : {p.gt_path}")
            print(f"    metrics: {format_metrics_line(p.metrics, metric_keys).strip()}")
            print(f"    reason : {reasons}")
        if len(outlier_indices) > len(to_print):
            print(f"... and {len(outlier_indices) - len(to_print)} more outliers not shown (use --max_print_outliers).")

        if len(inlier_pairs) > 0:
            inlier_mean = mean_metrics(inlier_pairs, metric_keys)
            print("\nMean after removing outliers:")
            print(format_metrics_line(inlier_mean, metric_keys))
        else:
            print("\nAll samples are flagged as outliers under current threshold. "
                  "Consider increasing --outlier_z.")
    else:
        print("\nNo outliers detected under current threshold (robust MAD z-score).")

    if missing:
        print("\nMissing/failed examples (first 10):")
        for x in missing[:10]:
            print("  -", x)

    if ambiguous:
        print("\nAmbiguous basename matches examples (first 10):")
        for pred_path, b in ambiguous[:10]:
            print(f"  - pred: {pred_path} | basename: {b} | gt_candidates={len(gt_basename_index.get(b, []))}")

    if args.save_json:
        payload = {
            "pred_dir": pred_dir,
            "gt_dir": gt_dir,
            "counts": {
                "pred_found": len(pred_files),
                "paired_success": len(paired),
                "missing_or_failed": len(missing),
                "ambiguous": len(ambiguous),
                "outliers": len(outlier_indices),
                "inliers": len(inlier_pairs),
            },
            "metric_keys": metric_keys,
            "overall_mean": overall_mean,
            "outlier_indices": outlier_indices,
            "mean_after_outlier_removal": mean_metrics(inlier_pairs, metric_keys) if len(inlier_pairs) > 0 else None,
            "pairs": [
                {
                    "pred_path": p.pred_path,
                    "gt_path": p.gt_path,
                    "metrics": p.metrics,
                    "is_outlier": bool(outlier_flags[i]),
                    "outlier_reasons": outlier_reasons[i],
                }
                for i, p in enumerate(paired)
            ],
            "missing": missing,
            "ambiguous": [{"pred_path": pp, "basename": b, "gt_candidates": gt_basename_index.get(b, [])} for pp, b in ambiguous],
            "args": vars(args),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved detailed results to: {os.path.abspath(args.save_json)}")

    print("\nDone.")
    print("=" * 80)


if __name__ == "__main__":
    main()
