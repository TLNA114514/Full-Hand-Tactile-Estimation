# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Literal, List, Optional

import numpy as np
import cv2

try:
    # moviepy 1.x
    from moviepy.editor import ImageSequenceClip
except ModuleNotFoundError:
    # moviepy 2.x
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


def str2bool(v):
    """
    argparse friendly bool parser.
    Accepts: true/false, 1/0, yes/no, y/n, on/off (case-insensitive).
    Also accepts python bool directly.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    if not isinstance(v, str):
        raise argparse.ArgumentTypeError(f"Invalid boolean value type: {type(v)}")
    s = v.strip().lower()
    if s in {'1','true','t','yes','y','on'}:
        return True
    if s in {'0','false','f','no','n','off'}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{v}'")



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


def get_mask_indexed(hand: str) -> np.ndarray:
    hand = hand.lower().strip()
    if hand == "right":
        return right_mask_indexed
    if hand == "left":
        return left_mask_indexed
    raise ValueError(f"Unknown hand='{hand}', expected 'right' or 'left'")


def load_image_any(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def is_w3h1(img: np.ndarray, tol: float = 0.02) -> bool:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return False
    ratio = w / float(h)
    return abs(ratio - 3.0) <= tol


def split_long_image_keep_right_square(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    start = max(0, w - h)
    left_part = img[:, :start].copy()
    right_square = img[:, start:w].copy()
    return left_part, right_square


def to_bgr3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return out
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape for to_bgr3: {img.shape}")


def ensure_square_and_resize_to_256(img: np.ndarray, target: int = 256) -> np.ndarray:
    h, w = img.shape[:2]
    if h != w:
        raise ValueError(
            f"Input image is not square: H={h}, W={w}. "
            f"Reason: The pipeline assumes a square heatmap canvas."
        )
    if h == target:
        return img

    print(f"[WARNING] Input image size is {h}x{w}, not {target}x{target}. Resizing to {target}x{target}.",
          file=sys.stderr)

    if h < target:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_AREA
    return cv2.resize(img, (target, target), interpolation=interp)


def compute_fit_layout(mask_indexed: np.ndarray, square_size: int = 256) -> Tuple[int, int, int, int]:
    H, W = mask_indexed.shape
    scale = min(square_size / float(W), square_size / float(H))
    th = max(1, int(round(H * scale)))
    tw = max(1, int(round(W * scale)))
    top = (square_size - th) // 2
    left = (square_size - tw) // 2
    return th, tw, top, left


def _resize_mask_nn(mask: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    oh, ow = out_size
    return cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)


def _scale_positions(src_positions, src_shape, dst_shape) -> np.ndarray:
    Hs, Ws = src_shape
    Hd, Wd = dst_shape
    sy = Hd / Hs
    sx = Wd / Ws
    out = []
    for (y, x) in src_positions:
        yd = (y + 0.5) * sy - 0.5
        xd = (x + 0.5) * sx - 0.5
        out.append((yd, xd))
    return np.array(out, dtype=np.float32)


def _build_sensor_mapping_keep_dups(mask_indexed: np.ndarray):
    H, W = mask_indexed.shape
    hand_mask = (mask_indexed > 0).astype(np.uint8)
    sensors = []
    for y in range(H):
        for x in range(W):
            idx1 = int(mask_indexed[y, x])
            if idx1 > 0:
                sensors.append((idx1, y, x))
    sensors.sort(key=lambda t: t[0])
    return sensors, hand_mask


GrayMethod = Literal["pca1", "luma", "hsv_v", "max_rgb", "mean_rgb"]

def force_to_grayscale(img_bgr: np.ndarray, method: GrayMethod = "pca1") -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr.astype(np.float32)
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {img_bgr.shape}")

    x = img_bgr.astype(np.float32)

    if method == "luma":
        b, g, r = x[..., 0], x[..., 1], x[..., 2]
        return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)

    if method == "hsv_v":
        xu8 = np.clip(x, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(xu8, cv2.COLOR_BGR2HSV)
        return hsv[..., 2].astype(np.float32)

    if method == "max_rgb":
        return np.max(x, axis=2).astype(np.float32)

    if method == "mean_rgb":
        return np.mean(x, axis=2).astype(np.float32)

    if method == "pca1":
        flat = x.reshape(-1, 3)
        flat_centered = flat - np.mean(flat, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(flat_centered, full_matrices=False)
        w = vt[0]
        proj = (flat_centered @ w).reshape(x.shape[0], x.shape[1]).astype(np.float32)

        luma = force_to_grayscale(img_bgr, method="luma")
        c = np.corrcoef(proj.reshape(-1), luma.reshape(-1))[0, 1]
        if np.isnan(c) or c < 0:
            proj = -proj
        return proj

    raise ValueError(f"Unknown grayscale method: {method}")


def maybe_scale_to_01(s: np.ndarray) -> np.ndarray:
    x = s.astype(np.float32)
    mx = float(np.max(x)) if x.size else 0.0
    if mx > 1.5:
        x = x / 255.0
    return x



def robust_normalize_inside_mask(scalar01: np.ndarray,
                                 valid_mask: np.ndarray,
                                 low_q: float = 1.0,
                                 high_q: float = 99.0,
                                 min_range: float = 1e-3,
                                 fallback: str = "minmax") -> np.ndarray:
    m = valid_mask.astype(bool)
    out = np.zeros_like(scalar01, dtype=np.float32)
    if not np.any(m):
        return out

    v = scalar01[m].astype(np.float32)
    lo = float(np.percentile(v, low_q))
    hi = float(np.percentile(v, high_q))
    rng = hi - lo

    if rng >= min_range:
        vv = np.clip(v, lo, hi)
        vv = (vv - lo) / (hi - lo)
        out[m] = vv
        return out

    print(f"[WARNING] Percentile normalization degenerates inside mask (hi-lo={rng:.6f} < {min_range}). "
          f"Using fallback='{fallback}'.", file=sys.stderr)

    if fallback == "minmax":
        vmin = float(np.min(v))
        vmax = float(np.max(v))
        rr = vmax - vmin
        if rr < 1e-8:
            out[m] = 0.0
            return out
        out[m] = np.clip((v - vmin) / rr, 0.0, 1.0)
        return out

    if fallback == "zscore":
        mu = float(np.mean(v))
        sd = float(np.std(v))
        if sd < 1e-6:
            out[m] = 0.0
            return out
        z = (v - mu) / sd
        out[m] = np.clip((z + 2.0) / 4.0, 0.0, 1.0)
        return out

    raise ValueError(f"Unknown fallback method: {fallback}")



@dataclass
class APack:
    A_u: np.ndarray
    active_idx1_u: np.ndarray
    pix_coords: np.ndarray
    hand_mask_up: np.ndarray
    centers_occ_up: np.ndarray
    idx1_occ: np.ndarray


def build_A_matrix_v2_compressed(mask_indexed: np.ndarray,
                                 out_size: Tuple[int, int],
                                 sigma_pix: float,
                                 eps: float = 1e-8,
                                 chunk: int = 4096) -> APack:
    Hs, Ws = mask_indexed.shape
    oh, ow = out_size

    sensors, hand_mask = _build_sensor_mapping_keep_dups(mask_indexed)
    idx1_occ = np.array([s[0] for s in sensors], dtype=np.int32)

    hand_mask_up = _resize_mask_nn(hand_mask, out_size)
    src_centers = [(y, x) for (_, y, x) in sensors]
    centers_occ_up = _scale_positions(src_centers, (Hs, Ws), (oh, ow))

    ys, xs = np.where(hand_mask_up > 0)
    pix_coords = np.stack([ys, xs], axis=1).astype(np.int32)
    P = pix_coords.shape[0]
    Mocc = idx1_occ.shape[0]

    A_occ = np.zeros((P, Mocc), dtype=np.float32)
    pc = pix_coords.astype(np.float32)
    for i in range(0, P, chunk):
        j = min(P, i + chunk)
        p = pc[i:j]
        d2 = (p[:, [0]] - centers_occ_up[:, 0][None, :])**2 + (p[:, [1]] - centers_occ_up[:, 1][None, :])**2
        W = np.exp(-0.5 * d2 / (sigma_pix**2)).astype(np.float32)
        denom = np.sum(W, axis=1, keepdims=True) + eps
        A_occ[i:j, :] = W / denom

    idx1_u = np.unique(idx1_occ)
    Ku = idx1_u.shape[0]
    A_u = np.zeros((P, Ku), dtype=np.float32)
    for k, idx in enumerate(idx1_u.tolist()):
        cols = np.where(idx1_occ == idx)[0]
        A_u[:, k] = np.sum(A_occ[:, cols], axis=1)

    return APack(
        A_u=A_u,
        active_idx1_u=idx1_u.astype(np.int32),
        pix_coords=pix_coords,
        hand_mask_up=(hand_mask_up > 0).astype(np.uint8),
        centers_occ_up=centers_occ_up,
        idx1_occ=idx1_occ
    )


AggMethod = Literal["mean", "median", "trimmed_mean"]

def sample_disk_values(gray01_th_tw: np.ndarray,
                       centers_occ_up: np.ndarray,
                       radius: float = 3.0,
                       agg: AggMethod = "mean",
                       trim_ratio: float = 0.2) -> np.ndarray:
    H, W = gray01_th_tw.shape
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    disk = (yy*yy + xx*xx) <= (radius*radius)

    s_list = []
    for (cy, cx) in centers_occ_up:
        iy = int(np.round(cy))
        ix = int(np.round(cx))

        y0 = max(0, iy - r); y1 = min(H, iy + r + 1)
        x0 = max(0, ix - r); x1 = min(W, ix + r + 1)

        patch = gray01_th_tw[y0:y1, x0:x1]
        dy0 = y0 - (iy - r); dy1 = dy0 + (y1 - y0)
        dx0 = x0 - (ix - r); dx1 = dx0 + (x1 - x0)
        m = disk[dy0:dy1, dx0:dx1]

        vals = patch[m].astype(np.float32)
        if vals.size == 0:
            s_list.append(0.0)
            continue

        if agg == "mean":
            s_list.append(float(np.mean(vals)))
        elif agg == "median":
            s_list.append(float(np.median(vals)))
        elif agg == "trimmed_mean":
            vv = np.sort(vals)
            k = int(np.floor(trim_ratio * vv.size))
            if vv.size - 2 * k <= 1:
                s_list.append(float(np.mean(vv)))
            else:
                s_list.append(float(np.mean(vv[k:-k])))
        else:
            raise ValueError(f"Unknown agg: {agg}")

    return np.array(s_list, dtype=np.float32)


def occ_to_v256(idx1_occ: np.ndarray,
                s_occ: np.ndarray,
                agg: AggMethod = "median",
                trim_ratio: float = 0.2) -> np.ndarray:
    v256 = np.zeros((256,), dtype=np.float32)
    for idx in np.unique(idx1_occ):
        vals = s_occ[idx1_occ == idx].astype(np.float32)
        if vals.size == 0:
            continue
        if agg == "mean":
            v = float(np.mean(vals))
        elif agg == "median":
            v = float(np.median(vals))
        elif agg == "trimmed_mean":
            vv = np.sort(vals)
            k = int(np.floor(trim_ratio * vv.size))
            if vv.size - 2 * k <= 1:
                v = float(np.mean(vv))
            else:
                v = float(np.mean(vv[k:-k]))
        else:
            raise ValueError(f"Unknown agg: {agg}")
        v256[idx - 1] = v
    return v256



def map_to_uint8_fixed_range(heatmap_float: np.ndarray,
                             valid_mask_uint8: np.ndarray,
                             vmin: float,
                             vmax: float) -> np.ndarray:
    if vmax <= vmin:
        raise ValueError("value_max must be greater than value_min")
    m = valid_mask_uint8.astype(bool)
    gray = np.zeros_like(heatmap_float, dtype=np.uint8)
    if m.any():
        hm = heatmap_float.astype(np.float32).copy()
        hm = np.clip(hm, vmin, vmax, out=hm)
        hm = (hm - vmin) / (vmax - vmin)
        gray[m] = (hm[m] * 255.0).clip(0, 255).astype(np.uint8)
    return gray



@dataclass
class ProjectConfig:
    square_size: int = 256
    sigma_pix: float = 1.0
    gray_method: GrayMethod = "pca1"
    value_min: float = 0.0
    value_max: float = 350.0
    use_percentile_normalize: bool = True
    low_q: float = 1.0
    high_q: float = 99.0
    sample_radius: float = 3.0
    occ_agg: AggMethod = "mean"
    idx_agg: AggMethod = "median"
    trim_ratio: float = 0.2
    intensity_scale: float = 1.0


def project_to_standard_heatmap(img_256: np.ndarray,
                                mask_indexed: np.ndarray,
                                cfg: ProjectConfig) -> Dict[str, Any]:
    th, tw, top, left = compute_fit_layout(mask_indexed, square_size=cfg.square_size)
    A_pack = build_A_matrix_v2_compressed(mask_indexed, out_size=(th, tw), sigma_pix=cfg.sigma_pix)

    gray_sq = force_to_grayscale(img_256, method=cfg.gray_method)
    gray01_sq = maybe_scale_to_01(gray_sq)
    gray01_th_tw = gray01_sq[top:top + th, left:left + tw].copy()

    if cfg.use_percentile_normalize:
        gray01_th_tw = robust_normalize_inside_mask(
            gray01_th_tw, A_pack.hand_mask_up, low_q=cfg.low_q, high_q=cfg.high_q
        )
    else:
        gray01_th_tw = np.clip(gray01_th_tw, 0.0, 1.0).astype(np.float32) * A_pack.hand_mask_up.astype(np.float32)
        
    if cfg.intensity_scale != 1.0:
        gray01_th_tw = np.clip(gray01_th_tw * cfg.intensity_scale, 0.0, 1.0).astype(np.float32)
        gray01_th_tw = gray01_th_tw * A_pack.hand_mask_up.astype(np.float32)

    s_occ = sample_disk_values(
        gray01_th_tw, A_pack.centers_occ_up,
        radius=cfg.sample_radius, agg=cfg.occ_agg, trim_ratio=cfg.trim_ratio
    )

    v256_intensity = occ_to_v256(A_pack.idx1_occ, s_occ, agg=cfg.idx_agg, trim_ratio=cfg.trim_ratio)
    v256_physical = cfg.value_min + v256_intensity * (cfg.value_max - cfg.value_min)

    v_u = np.array([v256_physical[idx1 - 1] for idx1 in A_pack.active_idx1_u.tolist()], dtype=np.float32)
    h_vec = (A_pack.A_u @ v_u).astype(np.float32)

    heat_std_th_tw = np.zeros((th, tw), dtype=np.float32)
    pc = A_pack.pix_coords
    heat_std_th_tw[pc[:, 0], pc[:, 1]] = h_vec

    heat_sq = np.zeros((cfg.square_size, cfg.square_size), dtype=np.float32)
    mask_sq = np.zeros((cfg.square_size, cfg.square_size), dtype=np.uint8)
    heat_sq[top:top + th, left:left + tw] = heat_std_th_tw
    mask_sq[top:top + th, left:left + tw] = A_pack.hand_mask_up.astype(np.uint8)

    standardized_u8 = map_to_uint8_fixed_range(heat_sq, mask_sq, cfg.value_min, cfg.value_max)

    return {
        "standardized_u8": standardized_u8,
        "heat_std_th_tw": heat_std_th_tw,
        "A_pack": A_pack,
        "layout": (th, tw, top, left),
        "v256_from_sampling": v256_physical,
    }


PixelStrategy = Literal["all_mask_pixels", "near_centers"]

def select_pixels(pix_coords: np.ndarray,
                  centers_occ_up: np.ndarray,
                  strategy: PixelStrategy = "near_centers",
                  radius_pix: float = 6.0) -> np.ndarray:
    if strategy == "all_mask_pixels":
        return np.arange(pix_coords.shape[0], dtype=np.int32)

    pc = pix_coords.astype(np.float32)
    P = pc.shape[0]
    min_d2 = np.full((P,), np.inf, dtype=np.float32)

    chunk = 4096
    for i in range(0, P, chunk):
        j = min(P, i + chunk)
        p = pc[i:j]
        d2 = (p[:, [0]] - centers_occ_up[:, 0][None, :])**2 + (p[:, [1]] - centers_occ_up[:, 1][None, :])**2
        min_d2[i:j] = np.min(d2, axis=1)

    sel = min_d2 <= (radius_pix * radius_pix)
    idx = np.where(sel)[0].astype(np.int32)
    if idx.size < 64:
        return np.arange(pix_coords.shape[0], dtype=np.int32)
    return idx


def _undo_fixed_range(gray01: np.ndarray,
                      valid_mask_uint8: np.ndarray,
                      vmin: float,
                      vmax: float) -> np.ndarray:
    heat = np.zeros_like(gray01, dtype=np.float32)
    m = valid_mask_uint8.astype(bool)
    if m.any():
        heat[m] = gray01[m].astype(np.float32) * (vmax - vmin) + vmin
    return heat


def _solve_v(A: np.ndarray,
             h: np.ndarray,
             ridge_lambda: float = 0.0,
             nonneg: bool = False) -> np.ndarray:
    A = A.astype(np.float32)
    h = h.astype(np.float32).reshape(-1)

    if ridge_lambda and ridge_lambda > 0:
        M = A.shape[1]
        lhs = A.T @ A + float(ridge_lambda) * np.eye(M, dtype=np.float32)
        rhs = A.T @ h
        v = np.linalg.solve(lhs, rhs).astype(np.float32)
    else:
        v, *_ = np.linalg.lstsq(A, h, rcond=None)
        v = v.astype(np.float32)

    if nonneg:
        v = np.maximum(v, 0.0)
    return v


@dataclass
class InvertConfig:
    pixel_strategy: PixelStrategy = "near_centers"
    center_radius_pix: float = 6.0

    ridge_lambda: float = 1e-3

    huber_k: float = 1.345
    max_iter: int = 30
    nonneg: bool = True


def invert_from_standard_u8(standardized_u8_sq: np.ndarray,
                            A_pack: APack,
                            layout: Tuple[int, int, int, int],
                            value_min: float,
                            value_max: float,
                            cfg: InvertConfig) -> np.ndarray:
    th, tw, top, left = layout
    gray01_sq = standardized_u8_sq.astype(np.float32) / 255.0
    gray01_th_tw = gray01_sq[top:top + th, left:left + tw].copy()

    heat_th_tw = _undo_fixed_range(gray01_th_tw, A_pack.hand_mask_up, value_min, value_max)

    idx_sel = select_pixels(
        A_pack.pix_coords, A_pack.centers_occ_up,
        strategy=cfg.pixel_strategy, radius_pix=cfg.center_radius_pix
    )

    pix = A_pack.pix_coords[idx_sel]
    A_sel = A_pack.A_u[idx_sel, :]
    h_sel = heat_th_tw[pix[:, 0], pix[:, 1]].astype(np.float32)

    v_u = _solve_v(A_sel, h_sel, ridge_lambda=cfg.ridge_lambda, nonneg=cfg.nonneg)

    v256 = np.zeros((256,), dtype=np.float32)
    for idx1, val in zip(A_pack.active_idx1_u.tolist(), v_u.tolist()):
        v256[idx1 - 1] = float(val)
    return v256


def invert_from_standard_heat(heat_std_th_tw: np.ndarray,
                              A_pack: APack,
                              cfg: InvertConfig) -> np.ndarray:
    idx_sel = select_pixels(
        A_pack.pix_coords, A_pack.centers_occ_up,
        strategy=cfg.pixel_strategy, radius_pix=cfg.center_radius_pix
    )
    pix = A_pack.pix_coords[idx_sel]
    A_sel = A_pack.A_u[idx_sel, :]
    h_sel = heat_std_th_tw[pix[:, 0], pix[:, 1]].astype(np.float32)

    v_u = _solve_v(A_sel, h_sel, ridge_lambda=cfg.ridge_lambda, nonneg=cfg.nonneg)

    v256 = np.zeros((256,), dtype=np.float32)
    for idx1, val in zip(A_pack.active_idx1_u.tolist(), v_u.tolist()):
        v256[idx1 - 1] = float(val)
    return v256


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def find_animated_images_dirs(root: Path) -> List[Path]:
    out = []
    for dirpath, _, _ in os.walk(root):
        if Path(dirpath).name == "animated_images":
            out.append(Path(dirpath))
    return out

def list_images_in_dir(d: Path) -> List[Path]:
    imgs = [p for p in d.iterdir() if is_image_file(p)]
    imgs.sort(key=lambda x: x.name)
    return imgs

def ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def concat_frames_to_video_moviepy(frames: List[Path], video_path: Path, fps: int = 15) -> None:
    if not frames:
        return
    if ImageSequenceClip is None:
        raise ImportError(f"moviepy import failed: {_MOVIEPY_IMPORT_ERR}")

    images = [str(p) for p in frames]

    video_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path.exists():
        try:
            video_path.unlink()
        except Exception:
            pass

    clip = ImageSequenceClip(images, fps=fps)
    clip.write_videofile(str(video_path), codec="libx264", audio=False)
    print(f"[OK] Video saved: {video_path}")


def process_one_image(
    img_path: Path,
    out_heatmap_path: Path,
    out_npy_path: Path,
    mask_indexed: np.ndarray,
    pcfg: ProjectConfig,
    icfg: InvertConfig,
    save_heatmap: bool,
    enable_concat_video: bool,
) -> Tuple[np.ndarray, Optional[Path]]:
    img_full = load_image_any(str(img_path))

    long_flag = is_w3h1(img_full, tol=0.02)
    left_part = None
    right_square = None
    if long_flag:
        left_part, right_square = split_long_image_keep_right_square(img_full)
        img_for_pipeline = right_square
    else:
        img_for_pipeline = img_full

    img_for_pipeline = ensure_square_and_resize_to_256(img_for_pipeline, target=256)

    proj = project_to_standard_heatmap(img_for_pipeline, mask_indexed, pcfg)
    standardized_u8 = proj["standardized_u8"]        
    heat_std_th_tw = proj["heat_std_th_tw"]
    A_pack = proj["A_pack"]

    layout = proj["layout"]
    v256 = invert_from_standard_u8(standardized_u8, A_pack, layout, pcfg.value_min, pcfg.value_max, icfg)

    v256_2dp = np.round(v256.astype(np.float32), 2)
    ensure_parent_dir(out_npy_path)
    np.save(str(out_npy_path), v256_2dp)

    if save_heatmap:
        ensure_parent_dir(out_heatmap_path)
        ok = cv2.imwrite(str(out_heatmap_path), standardized_u8)
        if not ok:
            raise IOError(f"Failed to save standardized heatmap: {out_heatmap_path}")
    concat_path: Optional[Path] = None
    if enable_concat_video and long_flag and left_part is not None and right_square is not None:
        h0, w0 = right_square.shape[:2] 
        heat_resized = cv2.resize(standardized_u8, (w0, h0), interpolation=cv2.INTER_LINEAR)
        heat_bgr = cv2.cvtColor(heat_resized, cv2.COLOR_GRAY2BGR)

        left_bgr = to_bgr3(left_part)

        if left_bgr.shape[0] != heat_bgr.shape[0]:
            left_bgr = cv2.resize(left_bgr, (left_bgr.shape[1], heat_bgr.shape[0]), interpolation=cv2.INTER_AREA)

        concat_img = np.concatenate([left_bgr, heat_bgr], axis=1)

        concat_path = out_heatmap_path.with_name(out_heatmap_path.stem + "_concat.png")
        ensure_parent_dir(concat_path)
        ok2 = cv2.imwrite(str(concat_path), concat_img)
        if not ok2:
            raise IOError(f"Failed to save concat frame: {concat_path}")

    return v256_2dp, concat_path



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input_dir", default="", help="root folder to search recursively")
    p.add_argument("--output_dir", default="", help="output root folder")
    p.add_argument("--intensity_scale", type=float, default=1)
    
    

    p.add_argument("--hand", default="right", choices=["right", "left"], help="mask choice (default: right)")
    p.add_argument("--save_heatmap", type=str2bool, default=True,
                   help="save standardized heatmap images (True/False, default: True)")
    p.add_argument("--aggregate_npy", type=str2bool, nargs='?', const=True, default=True,
                   help="save v256_all.npy per animated_images folder (True/False, default: True)")
    p.add_argument("--save_concat_video", type=str2bool, default=True,
                   help="save concat frames and merge video per animated_images folder (True/False, default: True)")

    p.add_argument("--sigma_pix", type=float, default=1.0, help="sigma in pixels for gaussian kernel")
    p.add_argument("--gray_method", default="mean_rgb",
                   choices=["pca1", "luma", "hsv_v", "max_rgb", "mean_rgb"])
    p.add_argument("--value_min", type=float, default=0.0)
    p.add_argument("--value_max", type=float, default=200.0)

    p.add_argument("--use_percentile_normalize", action="store_true",
                   help="normalize inside mask with percentiles (recommended)")
    p.add_argument("--low_q", type=float, default=1.0)
    p.add_argument("--high_q", type=float, default=99.0)

    p.add_argument("--sample_radius", type=float, default=3.0, help="disk radius for sampling centers (stage1)")
    p.add_argument("--occ_agg", default="mean", choices=["mean", "median", "trimmed_mean"],
                   help="aggregation inside each center disk (stage1)")
    p.add_argument("--idx_agg", default="median", choices=["mean", "median", "trimmed_mean"],
                   help="aggregation across duplicate indices (stage1)")
    p.add_argument("--trim_ratio", type=float, default=0.2, help="for trimmed_mean")

    p.add_argument("--pixel_strategy", default="near_centers", choices=["near_centers", "all_mask_pixels"])
    p.add_argument("--center_radius_pix", type=float, default=6.0)
    p.add_argument("--ridge_lambda", type=float, default=1e-3)
    p.add_argument("--huber_k", type=float, default=1.345)
    p.add_argument("--max_iter", type=int, default=30)
    p.add_argument("--nonneg", action="store_true", help="enforce v>=0 (recommended)")

    return p.parse_args()


def main():
    args = parse_args()

    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise ValueError(f"--input_dir is not a valid directory: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    mask_indexed = get_mask_indexed(args.hand)

    pcfg = ProjectConfig(
        square_size=256,
        sigma_pix=float(args.sigma_pix),
        gray_method=args.gray_method,
        value_min=float(args.value_min),
        value_max=float(args.value_max),
        use_percentile_normalize=bool(args.use_percentile_normalize),
        low_q=float(args.low_q),
        high_q=float(args.high_q),
        sample_radius=float(args.sample_radius),
        occ_agg=args.occ_agg,
        idx_agg=args.idx_agg,
        trim_ratio=float(args.trim_ratio),
        intensity_scale=args.intensity_scale,
    )

    icfg = InvertConfig(
        pixel_strategy=args.pixel_strategy,
        center_radius_pix=float(args.center_radius_pix),
        ridge_lambda=float(args.ridge_lambda),
        huber_k=float(args.huber_k),
        max_iter=int(args.max_iter),
        nonneg=bool(args.nonneg),
    )

    animated_dirs = find_animated_images_dirs(input_root)
    if not animated_dirs:
        print(f"[WARN] No 'animated_images' folder found under: {input_root}")
        return

    total_imgs = 0
    total_dirs = 0

    for adir in sorted(animated_dirs, key=lambda x: str(x)):
        imgs = list_images_in_dir(adir)
        if not imgs:
            continue

        total_dirs += 1
        total_imgs += len(imgs)

        rel_dir = adir.relative_to(input_root)   # .../animated_images
        out_dir = output_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        v_list = []  # for aggregation
        concat_frames: List[Path] = [] 

        for img_path in imgs:
            rel_img = img_path.relative_to(input_root)

            out_heatmap_path = output_root / rel_img

            out_npy_path = (output_root / rel_img).with_suffix(".npy")

            try:
                v256_2dp, concat_path = process_one_image(
                    img_path=img_path,
                    out_heatmap_path=out_heatmap_path,
                    out_npy_path=out_npy_path,
                    mask_indexed=mask_indexed,
                    pcfg=pcfg,
                    icfg=icfg,
                    save_heatmap=bool(args.save_heatmap),
                    enable_concat_video=bool(args.save_concat_video),
                )
                v_list.append(v256_2dp)

                if concat_path is not None:
                    concat_frames.append(concat_path)

                print(f"[OK] {img_path} -> {out_npy_path}" + (f", {out_heatmap_path}" if args.save_heatmap else ""))
            except Exception as e:
                print(f"[ERR] Failed on {img_path}: {e}", file=sys.stderr)
        if args.aggregate_npy and v_list:
            v_all = np.stack(v_list, axis=0).astype(np.float32)
            v_all = np.round(v_all, 2) 
            out_all_path = out_dir.parent.with_suffix(".npy")
            np.save(str(out_all_path), v_all)
            print(f"[OK] Saved aggregated v256 to: {out_all_path}")
        if args.save_concat_video and concat_frames:
            video_path = out_dir.parent / "animated_images.mp4"
            try:
                concat_frames_to_video_moviepy(concat_frames, video_path, fps=5)
                for p in concat_frames:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                print(f"[OK] Deleted concat frames under: {out_dir}")
            except Exception as e:
                print(f"[ERR] Failed to create video for {out_dir}: {e}", file=sys.stderr)

    print(f"[DONE] processed animated_images folders: {total_dirs}, images: {total_imgs}")


if __name__ == "__main__":
    main()
