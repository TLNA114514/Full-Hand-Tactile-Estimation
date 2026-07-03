#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np


try:
    import pressurevision_eval as pv
except Exception as e:
    raise ImportError(f"{repr(e)}")


def load_t256_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path} does not contain a numpy.ndarray")
    if arr.ndim != 2 or arr.shape[1] != 256:
        raise ValueError(f"{path} must have shape (T, 256), got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def list_npy_files(folder: str) -> List[str]:
    files = sorted(glob(os.path.join(folder, "*.npy")))
    return files


def select_mask(mask_side: str) -> np.ndarray:
    if mask_side == "right":
        return pv.right_mask_indexed
    if mask_side == "left":
        return pv.left_mask_indexed
    raise ValueError("--mask_side must be 'right' or 'left'")


def build_sensor_coords_from_mask(mask_indexed: np.ndarray, coord_space: str = "grid") -> np.ndarray:
    H, W = mask_indexed.shape
    coords = np.full((256, 2), np.nan, dtype=np.float64)

    ys, xs = np.where(mask_indexed > 0)
    ids_1based = mask_indexed[ys, xs].astype(np.int32)

    buckets: Dict[int, List[Tuple[float, float]]] = {}
    for y, x, sid in zip(ys, xs, ids_1based):
        m = int(sid) - 1
        if 0 <= m < 256:
            buckets.setdefault(m, []).append((float(x), float(y)))

    for m, pts in buckets.items():
        arr = np.array(pts, dtype=np.float64)
        mean_xy = arr.mean(axis=0)
        coords[m] = mean_xy

    if coord_space == "normalized":
        denom_x = max(W - 1, 1)
        denom_y = max(H - 1, 1)
        coords[:, 0] = coords[:, 0] / denom_x
        coords[:, 1] = coords[:, 1] / denom_y
    elif coord_space == "grid":
        pass
    else:
        raise ValueError("--coord_space must be 'grid' or 'normalized'")

    return coords


# ---------------------- part index sets：5 fingertips + palm ---------------------- #
def _to_zero_based(idxs_1based: List[int]) -> List[int]:
    return sorted({int(i) - 1 for i in idxxs_1based if 1 <= int(i) <= 256})


def build_part_index_sets(mask_side: str, mask_indexed: np.ndarray) -> Dict[str, np.ndarray]:
    if mask_side == "right":
        fingers_12 = pv.RH_FINGERS_12
    elif mask_side == "left":
        fingers_12 = pv.LH_FINGERS_12
    else:
        raise ValueError("--mask_side must be 'right' or 'left'")

    tip_1based = {
        "thumb_tip": list(fingers_12["thumb"][:3]),
        "index_tip": list(fingers_12["index"][:3]),
        "middle_tip": list(fingers_12["middle"][:3]),
        "ring_tip": list(fingers_12["ring"][:3]),
        "little_tip": list(fingers_12["little"][:3]),
    }

    parts: Dict[str, np.ndarray] = {}
    tip_union: set[int] = set()
    for name, idxs1 in tip_1based.items():
        idxs0 = sorted({int(i) - 1 for i in idxs1 if 1 <= int(i) <= 256})
        parts[name] = np.asarray(idxs0, dtype=np.int32)
        tip_union.update(idxs0)

    valid_idx = pv.get_valid_indices_from_mask(mask_indexed).tolist()
    palm0 = sorted([i for i in valid_idx if i not in tip_union])
    parts["palm"] = np.asarray(palm0, dtype=np.int32)

    return parts


@dataclass
class CopPartResult:
    err: float                # Err_CoP^(r)
    valid_frames: int         # |T_r|
    mean_dist: float          


def cop_error_partwise(
    gt: np.ndarray,           # (T,256)
    pred: np.ndarray,         # (T,256)
    coords: np.ndarray,       # (256,2)
    part_indices: np.ndarray, # (K,)
    tau: float,
    eps: float = 1e-6,
) -> CopPartResult:
    if part_indices.size == 0:
        return CopPartResult(err=float("nan"), valid_frames=0, mean_dist=float("nan"))

    # (T,K)
    gt_p = gt[:, part_indices]
    pr_p = pred[:, part_indices]

    # (T,)
    sum_gt = gt_p.sum(axis=1)
    sum_pr = pr_p.sum(axis=1)

    # Tr: sum_gt > tau OR sum_pr > tau
    valid = (sum_gt > tau) | (sum_pr > tau)
    idx_t = np.where(valid)[0]
    if idx_t.size == 0:
        return CopPartResult(err=float("nan"), valid_frames=0, mean_dist=float("nan"))

    # K×2
    c = coords[part_indices] 
    if np.any(~np.isfinite(c)):
        good = np.all(np.isfinite(c), axis=1)
        c = c[good]
        gt_p = gt_p[:, good]
        pr_p = pr_p[:, good]
        sum_gt = gt_p.sum(axis=1)
        sum_pr = pr_p.sum(axis=1)
        valid = (sum_gt > tau) | (sum_pr > tau)
        idx_t = np.where(valid)[0]
        if idx_t.size == 0 or c.shape[0] == 0:
            return CopPartResult(err=float("nan"), valid_frames=0, mean_dist=float("nan"))

    gt_num = gt_p @ c
    pr_num = pr_p @ c
    gt_den = (gt_p.sum(axis=1, keepdims=True) + eps)
    pr_den = (pr_p.sum(axis=1, keepdims=True) + eps)
    g = gt_num / gt_den
    gb = pr_num / pr_den

    d = np.linalg.norm(gb[idx_t] - g[idx_t], axis=1)
    err = float(d.mean()) if d.size > 0 else float("nan")
    return CopPartResult(err=err, valid_frames=int(idx_t.size), mean_dist=err)


def cop_error_system(
    gt: np.ndarray,
    pred: np.ndarray,
    coords: np.ndarray,
    parts: Dict[str, np.ndarray],
    tau: float,
) -> Dict[str, object]:
    
    per_part: Dict[str, Dict[str, object]] = {}

    errs: List[float] = []
    errs_fill0: List[float] = []
    for name in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip", "palm"]:
        r = cop_error_partwise(gt, pred, coords, parts[name], tau=tau)
        per_part[name] = {"err": r.err, "valid_frames": r.valid_frames}
        errs_fill0.append(0.0 if not np.isfinite(r.err) else float(r.err))
        if np.isfinite(r.err):
            errs.append(float(r.err))

    err_cop = float(np.mean(errs)) if len(errs) > 0 else float("nan")
    err_cop_fill0 = float(np.mean(errs_fill0)) if len(errs_fill0) == 6 else float("nan")

    return {
        "per_part": per_part,
        "err_cop": err_cop,
        "err_cop_fill0": err_cop_fill0,
    }


def detect_outliers_mad(values: List[float], z: float = 3.5, side: str = "low") -> Tuple[np.ndarray, float, float]:
    x = np.asarray(values, dtype=np.float64)
    med = np.nanmedian(x)
    abs_dev = np.abs(x - med)
    mad = np.nanmedian(abs_dev)

    if not np.isfinite(mad) or mad < 1e-12:
        if side == "low":
            thr = np.nanpercentile(x, 1.0)
            mask = x < thr
        elif side == "high":
            thr = np.nanpercentile(x, 99.0)
            mask = x > thr
        else:
            thr_lo = np.nanpercentile(x, 1.0)
            thr_hi = np.nanpercentile(x, 99.0)
            mask = (x < thr_lo) | (x > thr_hi)
        return mask.astype(bool), float(med), float(mad)

    robust_z = 0.6745 * (x - med) / mad
    if side == "low":
        mask = robust_z < -z
    elif side == "high":
        mask = robust_z > z
    elif side == "both":
        mask = np.abs(robust_z) > z
    else:
        raise ValueError("--outlier_side must be 'low', 'high', or 'both'")

    return mask.astype(bool), float(med), float(mad)


def evaluate_pair(
    gt_path: str,
    pred_path: str,
    mask_side: str,
    cop_contact_tau: float,
    coord_space: str,
    use_hand_contact_rule: bool,
    hand_contact_value_threshold: float,
    finger_contact_min_points: int,
    total_contact_min_points: int,
) -> Dict[str, object]:
    gt_full = load_t256_npy(gt_path)
    pred_full = load_t256_npy(pred_path)
    if gt_full.shape[0] != pred_full.shape[0]:
        raise ValueError(f"T mismatch: gt={gt_full.shape[0]} pred={pred_full.shape[0]} for {os.path.basename(pred_path)}")

    mask_indexed = select_mask(mask_side)
    coords = build_sensor_coords_from_mask(mask_indexed, coord_space=coord_space)
    parts = build_part_index_sets(mask_side=mask_side, mask_indexed=mask_indexed)

    if use_hand_contact_rule:
        finger_map = pv.build_finger_index_map(mask_side=mask_side)
        gt_full, _ = pv.apply_hand_contact_rule_per_frame(
            gt_full,
            finger_index_map=finger_map,
            value_threshold=hand_contact_value_threshold,
            finger_contact_min_points=finger_contact_min_points,
            total_contact_min_points=total_contact_min_points,
        )
        pred_full, _ = pv.apply_hand_contact_rule_per_frame(
            pred_full,
            finger_index_map=finger_map,
            value_threshold=hand_contact_value_threshold,
            finger_contact_min_points=finger_contact_min_points,
            total_contact_min_points=total_contact_min_points,
        )

    cop = cop_error_system(
        gt=gt_full,
        pred=pred_full,
        coords=coords,
        parts=parts,
        tau=cop_contact_tau,
    )

    return {
        "file": os.path.basename(pred_path),
        "gt_path": os.path.abspath(gt_path),
        "pred_path": os.path.abspath(pred_path),
        "frames_T": int(gt_full.shape[0]),
        "cop_contact_tau": float(cop_contact_tau),
        "coord_space": coord_space,
        "use_hand_contact_rule": bool(use_hand_contact_rule),
        "err_cop": float(cop["err_cop"]),
        "err_cop_fill0": float(cop["err_cop_fill0"]),
        "per_part": cop["per_part"],
    }


def aggregate_means(results: List[Dict[str, object]]) -> Dict[str, float]:
    vals = [float(r["err_cop"]) for r in results if np.isfinite(float(r["err_cop"]))]
    mean_err = float(np.mean(vals)) if len(vals) else float("nan")

    vals0 = [float(r["err_cop_fill0"]) for r in results if np.isfinite(float(r["err_cop_fill0"]))]
    mean_err0 = float(np.mean(vals0)) if len(vals0) else float("nan")

    part_names = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "little_tip", "palm"]
    per_part_mean: Dict[str, float] = {}
    for p in part_names:
        pv_list = []
        for r in results:
            e = r["per_part"][p]["err"]
            if e is None:
                continue
            e = float(e)
            if np.isfinite(e):
                pv_list.append(e)
        per_part_mean[p] = float(np.mean(pv_list)) if len(pv_list) else float("nan")

    out = {"mean_err_cop": mean_err, "mean_err_cop_fill0": mean_err0}
    out.update({f"mean_{k}": v for k, v in per_part_mean.items()})
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch evaluate Part-wise CoP Error (5 fingertips + palm).")
    p.add_argument("--gt_dir", default="", help="GT folder path (contains .npy).")
    p.add_argument("--pred_dir", default="", help="Pred folder path (contains .npy).")
    p.add_argument("--mask_side", choices=["right", "left"], default="right")

    p.add_argument("--cop_contact_tau", type=float, default=10.0, help="τ in Appendix H (part contact threshold).")

    p.add_argument("--coord_space", choices=["grid", "normalized"], default="grid",
                   help="Sensor coordinate space for CoP distance.")

    p.add_argument("--disable_hand_contact_rule", action="store_true",
                   help="Disable frame-level hand-contact filtering (do NOT zero-out non-contact frames).")
    p.add_argument("--hand_contact_value_threshold", type=float, default=10.0)
    p.add_argument("--finger_contact_min_points", type=int, default=3)
    p.add_argument("--total_contact_min_points", type=int, default=5)

    # outlier
    p.add_argument("--outlier_method", choices=["mad"], default="mad")
    p.add_argument("--outlier_z", type=float, default=3.5, help="MAD robust z threshold.")
    p.add_argument("--outlier_side", choices=["low", "high", "both"], default="low",
                   help="Detect outliers on which side. Default: low.")

    p.add_argument("--save_json", default="", help="Optional: save per-file results to a JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gt_dir = args.gt_dir
    pred_dir = args.pred_dir

    pred_files = list_npy_files(pred_dir)
    if len(pred_files) == 0:
        print(f"[ERROR] No .npy found in pred_dir: {pred_dir}", file=sys.stderr)
        sys.exit(2)

    mask_side = args.mask_side
    use_hand_contact_rule = (not args.disable_hand_contact_rule)

    results: List[Dict[str, object]] = []
    skipped: List[str] = []

    for pred_path in pred_files:
        name = os.path.basename(pred_path)
        gt_path = os.path.join(gt_dir, name)
        if not os.path.isfile(gt_path):
            skipped.append(name)
            continue
        try:
            r = evaluate_pair(
                gt_path=gt_path,
                pred_path=pred_path,
                mask_side=mask_side,
                cop_contact_tau=args.cop_contact_tau,
                coord_space=args.coord_space,
                use_hand_contact_rule=use_hand_contact_rule,
                hand_contact_value_threshold=args.hand_contact_value_threshold,
                finger_contact_min_points=args.finger_contact_min_points,
                total_contact_min_points=args.total_contact_min_points,
            )
            results.append(r)
        except Exception as e:
            skipped.append(f"{name} (error: {repr(e)})")

    if len(results) == 0:
        print("[ERROR] No valid GT-Pred pairs evaluated. Check folder paths / filenames.", file=sys.stderr)
        if skipped:
            print("[INFO] Skipped:", file=sys.stderr)
            for s in skipped[:50]:
                print("  -", s, file=sys.stderr)
        sys.exit(2)

    agg_all = aggregate_means(results)

    err_list = [float(r["err_cop"]) for r in results]
    mask_out, med, mad = detect_outliers_mad(err_list, z=args.outlier_z, side=args.outlier_side)

    outliers = [results[i] for i, m in enumerate(mask_out) if bool(m)]
    kept = [results[i] for i, m in enumerate(mask_out) if not bool(m)]

    agg_kept = aggregate_means(kept) if len(kept) > 0 else {}

    print("=" * 80)
    print("Batch Part-wise CoP Error Evaluation")
    print("- pred_dir:", os.path.abspath(pred_dir))
    print("- gt_dir  :", os.path.abspath(gt_dir))
    print(f"- mask_side={mask_side}, coord_space={args.coord_space}, cop_contact_tau={args.cop_contact_tau}")
    print(f"- use_hand_contact_rule={use_hand_contact_rule}")
    print(f"- evaluated_pairs={len(results)}, skipped={len(skipped)}")
    if skipped:
        print("  (first 20 skipped):")
        for s in skipped[:20]:
            print("   -", s)

    print("-" * 80)
    print("Mean metrics over ALL evaluated pairs:")
    print(json.dumps(agg_all, ensure_ascii=False, indent=2))

    print("-" * 80)
    print(f"Outlier detection: method=MAD, side={args.outlier_side}, z={args.outlier_z}")
    print(f"  median={med:.6g}, mad={mad:.6g}, outliers={len(outliers)}/{len(results)}")
    if outliers:
        print("Outlier files (name -> err_cop):")
        for r in outliers:
            print(f"  - {r['file']} -> {float(r['err_cop']):.6g}")

    if kept and outliers:
        print("-" * 80)
        print("Mean metrics AFTER removing outliers:")
        print(json.dumps(agg_kept, ensure_ascii=False, indent=2))

    print("=" * 80)

    if args.save_json:
        payload = {
            "meta": {
                "pred_dir": os.path.abspath(pred_dir),
                "gt_dir": os.path.abspath(gt_dir),
                "mask_side": mask_side,
                "coord_space": args.coord_space,
                "cop_contact_tau": float(args.cop_contact_tau),
                "use_hand_contact_rule": bool(use_hand_contact_rule),
                "outlier_method": args.outlier_method,
                "outlier_side": args.outlier_side,
                "outlier_z": float(args.outlier_z),
            },
            "summary_all": agg_all,
            "summary_after_outlier_removal": agg_kept if kept else None,
            "outliers": [{"file": r["file"], "err_cop": float(r["err_cop"])} for r in outliers],
            "results": results,
            "skipped": skipped,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {os.path.abspath(args.save_json)}")


if __name__ == "__main__":
    main()
