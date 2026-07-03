#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Iterator, Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
import argparse
import moviepy

from denoise import denoise_sensor_256


def make_side_by_side_videos(
    root_dir: str,
    output_dir: Optional[str] = None,
    object_info_json: str = "object_info.json",
    subject_info_json: Optional[str] = None,
    crop_x_offset: int = 100,
    crop_y_offset: int = 0,
    mirror_RGB_video: bool = False,
    square_size: int = 256,
    value_min: float = 0.0,
    value_max: float = 250.0,
    threshold: float = 5.0,
    heatmap_mode: str = "color",  # "color" | "gray"
    allow_frame_mismatch: bool = False,
) -> None:

    heatmap_mode = str(heatmap_mode).lower().strip()
    if heatmap_mode not in ("color", "gray"):
        raise ValueError(f"heatmap_mode must be 'color' or 'gray', got: {heatmap_mode}")

    S = int(square_size)

    LEFT_MASK_INDEXED = np.array([
        [ 31, 30, 29, 0, 28, 27, 26, 0, 25, 24, 23, 0, 22, 21, 20, 0, 19, 18, 17],
        [ 15, 14, 13, 0, 12, 11, 10, 0,  9,  8,  7, 0,  6,  5,  4, 0,  3,  2,  1],
        [255,254,253, 0,252,251,250, 0,249,248,247, 0,246,245,244, 0,243,242,241],
        [239,238,237, 0,236,235,234, 0,233,232,231, 0,230,229,228, 0,227,226,225],
        [  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0],
        [  0,222,  0, 0,  0,219,  0, 0,  0,216,  0, 0,  0,213,  0, 0,  0,  0,210],
        [  0,222,  0, 0,  0,219,  0, 0,  0,216,  0, 0,  0,213,  0, 0,  0,  0,210],
        [  0,222,  0, 0,  0,219,  0, 0,  0,216,  0, 0,  0,213,  0, 0,  0,210,  0],
        [  0,222,  0, 0,  0,219,  0, 0,  0,216,  0, 0,  0,213,  0, 0,  0,210,  0],
        [  0,222,  0, 0,  0,219,  0, 0,  0,216,  0, 0,  0,213,  0, 0, 210,  0,  0],
        [  0,222,  0, 0,  0,219,  0, 0,  0,216,  0, 0,  0,213,  0, 0, 210,  0,  0],
        [  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0],
        [207,206,205,204,203,202,201,200,199,198,197,196,  0,  0,  0, 0,  0,  0,  0],
        [191,190,189,188,187,186,185,184,183,182,181,180,179,178,177, 0,  0,  0,  0],
        [175,174,173,172,171,170,169,168,167,166,165,164,163,162,161, 0,  0,  0,  0],
        [159,158,157,156,155,154,153,152,151,150,149,148,147,146,145, 0,  0,  0,  0],
        [143,142,141,140,139,138,137,136,135,134,133,132,131,130,129, 0,  0,  0,  0],
    ], dtype=np.int32)

    RIGHT_MASK_INDEXED = np.array([
        [240,239,238, 0,237,236,235, 0,234,233,232, 0,231,230,229, 0,228,227,226],
        [256,255,254, 0,253,252,251, 0,250,249,248, 0,247,246,245, 0,244,243,242],
        [ 16, 15, 14, 0, 13, 12, 11, 0, 10,  9,  8, 0,  7,  6,  5, 0,  4,  3,  2],
        [ 32, 31, 30, 0, 29, 28, 27, 0, 26, 25, 24, 0, 23, 22, 21, 0, 20, 19, 18],
        [  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0],
        [ 47,  0,  0, 0,  0, 44,  0, 0,  0, 41,  0, 0,  0, 38,  0, 0,  0, 35,  0],
        [ 47,  0,  0, 0,  0, 44,  0, 0,  0, 41,  0, 0,  0, 38,  0, 0,  0, 35,  0],
        [  0, 47,  0, 0,  0, 44,  0, 0,  0, 41,  0, 0,  0, 38,  0, 0,  0, 35,  0],
        [  0, 47,  0, 0,  0, 44,  0, 0,  0, 41,  0, 0,  0, 38,  0, 0,  0, 35,  0],
        [  0,  0, 47, 0,  0, 44,  0, 0,  0, 41,  0, 0,  0, 38,  0, 0,  0, 35,  0],
        [  0,  0, 47, 0,  0, 44,  0, 0,  0, 41,  0, 0,  0, 38,  0, 0,  0, 35,  0],
        [  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0],
        [  0,  0,  0, 0,  0,  0,  0,61, 60, 59, 58,57, 56, 55, 54,53, 52, 51, 50],
        [  0,  0,  0, 0, 80, 79, 78,77, 76, 75, 74,73, 72, 71, 70,69, 68, 67, 66],
        [  0,  0,  0, 0, 96, 95, 94,93, 92, 91, 90,89, 88, 87, 86,85, 84, 83, 82],
        [  0,  0,  0, 0,112,111,110,109,108,107,106,105,104,103,102,101,100, 99, 98],
        [  0,  0,  0, 0,128,127,126,125,124,123,122,121,120,119,118,117,116,115,114],
    ], dtype=np.int32)

    def _build_sensor_mapping(mask_indexed: np.ndarray):
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

    def _resize_mask_nn(mask: np.ndarray, out_size) -> np.ndarray:
        oh, ow = out_size
        return cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

    def _scale_positions(src_points, src_shape, out_shape):
        Hs, Ws = src_shape
        oh, ow = out_shape
        sy = oh / float(Hs)
        sx = ow / float(Ws)
        out = []
        for (y, x) in src_points:
            yd = (y + 0.5) * sy - 0.5
            xd = (x + 0.5) * sx - 0.5
            out.append((yd, xd))
        return np.array(out, dtype=np.float32)

    def build_A_matrix(mask_indexed: np.ndarray, out_size, sigma_pix: float = 4.0, eps: float = 1e-8):
        Hs, Ws = mask_indexed.shape
        oh, ow = out_size
        sensors, hand_mask = _build_sensor_mapping(mask_indexed)
        active_idx1 = [s[0] for s in sensors]

        hand_mask_up = _resize_mask_nn(hand_mask, out_size)
        src_centers = [(y, x) for (_, y, x) in sensors]
        centers_up = _scale_positions(src_centers, (Hs, Ws), (oh, ow))

        ys, xs = np.where(hand_mask_up > 0)
        pix_coords = np.stack([ys, xs], axis=1).astype(np.float32)
        P = pix_coords.shape[0]
        M = len(active_idx1)

        A = np.zeros((P, M), dtype=np.float32)
        chunk = 4096
        for i in range(0, P, chunk):
            j = min(P, i + chunk)
            pc = pix_coords[i:j]
            d2 = (pc[:, [0]] - centers_up[:, 0][None, :])**2 + (pc[:, [1]] - centers_up[:, 1][None, :])**2
            W = np.exp(-0.5 * d2 / (sigma_pix**2)).astype(np.float32)
            denom = np.sum(W, axis=1, keepdims=True) + eps
            A[i:j, :] = W / denom

        return dict(
            A=A,
            active_idx1=active_idx1,
            pix_coords=pix_coords.astype(np.int32),
            out_size=(oh, ow),
            hand_mask_up=hand_mask_up,
            centers_up=centers_up,
            src_shape=(Hs, Ws),
        )

    def render_heatmap(A_pack, v_256: np.ndarray) -> np.ndarray:
        A = A_pack["A"]
        active_idx1 = A_pack["active_idx1"]
        oh, ow = A_pack["out_size"]
        pix_coords = A_pack["pix_coords"].astype(np.int32)
        v_active = np.array([float(v_256[idx1 - 1]) for idx1 in active_idx1], dtype=np.float32)
        h_flat = A.dot(v_active)
        heatmap = np.zeros((oh, ow), dtype=np.float32)
        heatmap[pix_coords[:, 0], pix_coords[:, 1]] = h_flat
        return heatmap

    def _map_to_uint8_fixed_range(heatmap_float: np.ndarray, valid_mask_uint8: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax <= vmin:
            raise ValueError("value_max must be greater than value_min")
        m = valid_mask_uint8.astype(bool)
        gray = np.zeros_like(heatmap_float, dtype=np.uint8)
        if m.any():
            hm = heatmap_float.copy()
            hm = np.clip(hm, vmin, vmax, out=hm)
            hm = (hm - vmin) / (vmax - vmin)
            gray_m = (hm[m] * 255.0).clip(0, 255).astype(np.uint8)
            gray[m] = gray_m
        return gray

    class _Renderer:
        def __init__(self, square_size: int, sigma_pix: float, value_min: float, value_max: float, threshold: float):
            self.square_size = int(square_size)
            self.sigma_pix = float(sigma_pix)
            self.value_min = float(value_min)
            self.value_max = float(value_max)
            self.threshold = float(threshold)
            self._cache: Dict[str, Dict[str, Any]] = {}

        def _apply_special_div6(self, hand_key: str, v256_proc: np.ndarray):
            if hand_key == "LH":
                idx_1 = [222, 219, 216, 213, 210]
            else:
                idx_1 = [47, 44, 41, 38, 35]
            for k in idx_1:
                z = k - 1
                if 0 <= z < v256_proc.shape[0]:
                    v256_proc[z] = v256_proc[z] / 6.0

        def _build_or_get_cache(self, hand_key: str, mask_indexed: np.ndarray):
            if hand_key in self._cache:
                return self._cache[hand_key]

            H, W = mask_indexed.shape
            S0 = self.square_size
            if H >= W:
                th, tw = S0, int(round(S0 * (W / H)))
            else:
                tw, th = S0, int(round(S0 * (H / W)))
            top = (S0 - th) // 2
            left = (S0 - tw) // 2

            A_pack = build_A_matrix(mask_indexed, out_size=(th, tw), sigma_pix=self.sigma_pix)
            valid_th_tw = (A_pack["hand_mask_up"] > 0).astype(np.uint8) * 255
            mask_sq = np.zeros((S0, S0), dtype=np.uint8)
            mask_sq[top:top + th, left:left + tw] = valid_th_tw

            cache = dict(A_pack=A_pack, th=th, tw=tw, top=top, left=left, mask_sq=mask_sq)
            self._cache[hand_key] = cache
            return cache

        def render_one_gray_u8(self, seq_256: np.ndarray, mask_indexed: np.ndarray, hand_key: str) -> Tuple[np.ndarray, np.ndarray]:
            v256_orig = np.asarray(seq_256, dtype=np.float32).copy()
            v256_proc = v256_orig.copy()
            self._apply_special_div6(hand_key, v256_proc)
            v256_proc[v256_proc < float(self.threshold)] = 0.0

            cache = self._build_or_get_cache(hand_key, mask_indexed)
            A_pack = cache["A_pack"]
            th, tw, top, left = cache["th"], cache["tw"], cache["top"], cache["left"]
            mask_sq = cache["mask_sq"]

            heat_th_tw = render_heatmap(A_pack, v256_proc)
            heat_sq = np.zeros((self.square_size, self.square_size), dtype=np.float32)
            heat_sq[top:top + th, left:left + tw] = heat_th_tw

            gray = _map_to_uint8_fixed_range(heat_sq, mask_sq, self.value_min, self.value_max)
            gray[mask_sq == 0] = 0
            return gray, mask_sq

        def render_one_color_bgr(self, seq_256: np.ndarray, mask_indexed: np.ndarray, hand_key: str) -> np.ndarray:
            gray, mask_sq = self.render_one_gray_u8(seq_256, mask_indexed, hand_key)
            color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            color[mask_sq == 0] = 0
            return color  # S×S BGR

    def _yield_old_jsonl(fp) -> Iterator[Tuple[str, List[float], float, str]]:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") != "full_frame":
                continue
            sensor_name = obj.get("sensor_name", "LH")
            seq_256 = obj.get("sensor_256", None)

            if denoise_sensor_256 is not None and seq_256 is not None:
                seq_256 = denoise_sensor_256(seq_256, sensor_name)

            ts = obj.get("timestamp", None)
            if seq_256 is None or ts is None:
                continue
            yield (sensor_name, seq_256, ts, "none")

    def _yield_new_json_array(records: list) -> Iterator[Tuple[str, List[float], float, str]]:
        for rec in records:
            top_ts = rec.get("timestamp", None)
            for hand_key in ("RH", "LH"):
                sub = rec.get(hand_key)
                if not isinstance(sub, dict):
                    continue
                seq = sub.get("sensor_256", None)

                if denoise_sensor_256 is not None and seq is not None:
                    seq = denoise_sensor_256(seq, hand_key)

                if seq is None:
                    continue
                anomaly = sub.get("anomaly", "none")
                yield (hand_key, seq, top_ts, anomaly)

    def _iter_records_from_json(json_path: str) -> Iterator[Tuple[str, List[float], float, str]]:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and (len(data) == 0 or isinstance(data[0], dict)):
                yield from _yield_new_json_array(data)
                return
        except Exception:
            pass
        with open(json_path, "r", encoding="utf-8") as f:
            yield from _yield_old_jsonl(f)

    def _load_object_info_map(path: str) -> Dict[str, Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_subject_info_map(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _extract_subject_key_from_string(s: str) -> Optional[str]:
        if not s:
            return None
        import re
        m = re.search(r"\b(p\d{3})\b", s, flags=re.IGNORECASE)
        return m.group(1).lower() if m else None

    def _extract_object_name(json_path: str) -> str:
        obj_name = "unknown"
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict) and ("object_name" in rec):
                        v = rec.get("object_name")
                        if v:
                            obj_name = str(v)
                            break
        except Exception:
            pass
        return obj_name

    def _extract_object_info_from_map(object_name: str, info_map: Dict[str, Dict[str, Any]]) -> Tuple[Any, str, Any]:
        info = info_map.get(object_name, {}) if isinstance(info_map, dict) else {}
        weight_g = info.get("weight_g", "unknown")
        surface_material = info.get("surface_material", "unknown")
        fill_state = info.get("fill_state", "unknown")
        return weight_g, str(surface_material), fill_state

    def _write_description_txt(
        txt_path: str,
        json_path: str,
        object_map: Dict[str, Dict[str, Any]],
        subject_map: Dict[str, Dict[str, Any]],
        subject_key: Optional[str] = None,
    ) -> None:
        name = _extract_object_name(json_path)
        weight_g, material, fill_state = _extract_object_info_from_map(name, object_map)

        if not subject_key:
            subject_key = _extract_subject_key_from_string(txt_path)
        if not subject_key:
            subject_key = _extract_subject_key_from_string(os.path.dirname(txt_path))

        subj_info = subject_map.get(subject_key, {}) if (subject_key and isinstance(subject_map, dict)) else {}
        gender = subj_info.get("gender", "unknown")
        subj_weight = subj_info.get("weight", "unknown")  # kg
        bodyfat = subj_info.get("bodyfat", "unknown")
        age = subj_info.get("age", "unknown")
        dominant_hand = subj_info.get("dominant_hand", "unknown")
        hand_length = subj_info.get("hand_length", "unknown")  # cm

        if isinstance(weight_g, (int, float)):
            weight_str = f"{weight_g}g"
        else:
            weight_str = str(weight_g)

        line = (
            f"This video shows the action of picking up {name}. The weight of {name} is {weight_str}, and its surface material is {material}. "
        )

        if subj_info:
            if isinstance(subj_weight, (int, float)):
                subj_weight_str = f"{subj_weight}kg"
            else:
                subj_weight_str = str(subj_weight)

            if isinstance(bodyfat, (int, float)):
                bodyfat_str = f"{bodyfat}%"
            else:
                bodyfat_str = str(bodyfat)

            if isinstance(hand_length, (int, float)):
                hand_length_str = f"{hand_length}cm"
            else:
                hand_length_str = str(hand_length)

            if gender == "F":
                full_gender = "female"
            elif gender == "M":
                full_gender = "male"
            line += (
                f"The person performing the action is {full_gender}, {age} years old, weighing {subj_weight_str} with {bodyfat_str} body fat."
            )
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(line + "\n")

    object_info_map = _load_object_info_map(object_info_json)
    subject_info_map = _load_subject_info_map(subject_info_json)

    def _iter_pressure_frames(json_path: str) -> Iterator[np.ndarray]:
        renderer = _Renderer(square_size=S, sigma_pix=1.0, value_min=value_min, value_max=value_max, threshold=threshold)

        cur_ts = None
        cur_pair_color: Dict[str, Optional[np.ndarray]] = {"RH": None, "LH": None}
        cur_pair_gray: Dict[str, Optional[np.ndarray]] = {"RH": None, "LH": None}
        cur_anomaly = {"RH": "none", "LH": "none"}

        def _flush_pair():
            if cur_pair_color["RH"] is None and cur_pair_color["LH"] is None and cur_pair_gray["RH"] is None and cur_pair_gray["LH"] is None:
                return None

            if heatmap_mode == "color":
                canvas = np.zeros((S, S, 3), dtype=np.uint8)
                for hk, _mask_idx in (("LH", LEFT_MASK_INDEXED), ("RH", RIGHT_MASK_INDEXED)):
                    img = cur_pair_color[hk]
                    if img is None:
                        continue
                    m = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0)
                    canvas[m] = img[m]

                text_lines = []
                for hk in ("RH", "LH"):
                    anom = cur_anomaly.get(hk, "none")
                    if anom and anom != "none":
                        text_lines.append(f"{hk}: {anom}")
                if text_lines:
                    y0 = 28
                    for line in text_lines:
                        cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                        y0 += 26
                return canvas

            else:
                # gray
                canvas_g = np.zeros((S, S), dtype=np.uint8)
                for hk in ("LH", "RH"):
                    g = cur_pair_gray[hk]
                    if g is None:
                        continue
                    canvas_g = np.maximum(canvas_g, g)

                text_lines = []
                for hk in ("RH", "LH"):
                    anom = cur_anomaly.get(hk, "none")
                    if anom and anom != "none":
                        text_lines.append(f"{hk}: {anom}")
                if text_lines:
                    bgr = cv2.cvtColor(canvas_g, cv2.COLOR_GRAY2BGR)
                    y0 = 28
                    for line in text_lines:
                        cv2.putText(bgr, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        y0 += 26
                    canvas_g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                return canvas_g

        for hand_key, seq_256, ts, anomaly in _iter_records_from_json(json_path):
            if cur_ts is None:
                cur_ts = ts
            if ts != cur_ts:
                out = _flush_pair()
                if out is not None:
                    yield out
                cur_ts = ts
                cur_pair_color = {"RH": None, "LH": None}
                cur_pair_gray = {"RH": None, "LH": None}
                cur_anomaly = {"RH": "none", "LH": "none"}

            mask_idx = RIGHT_MASK_INDEXED if hand_key == "RH" else LEFT_MASK_INDEXED
            if heatmap_mode == "color":
                img = renderer.render_one_color_bgr(seq_256, mask_idx, hand_key)
                if img.shape[0] != S or img.shape[1] != S:
                    img = cv2.resize(img, (S, S), interpolation=cv2.INTER_NEAREST)
                cur_pair_color[hand_key] = img
            else:
                g, _ = renderer.render_one_gray_u8(seq_256, mask_idx, hand_key)
                if g.shape[0] != S or g.shape[1] != S:
                    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_NEAREST)
                cur_pair_gray[hand_key] = g

            cur_anomaly[hand_key] = anomaly

        out = _flush_pair()
        if out is not None:
            yield out

    def _center_crop_square(frame_bgr: np.ndarray, x_offset: int = 100, y_offset: int = 0) -> np.ndarray:
        h, w = frame_bgr.shape[:2]

        if w == h:
            cx = w // 2 + x_offset
            cy = h // 2 + y_offset

            half = w // 2
            x0 = max(0, min(w - w, cx - half)) 
            y0 = max(0, min(h - h, cy - half))
            return frame_bgr[y0:y0 + h, x0:x0 + w]

        if w > h:
            cx = w // 2 + x_offset
            cy = h // 2 + y_offset

            half = h // 2
            x0 = cx - half
            x0 = max(0, min(w - h, x0))
            return frame_bgr[:, x0:x0 + h]
        else:
            cx = w // 2 + x_offset
            cy = h // 2 + y_offset

            half = w // 2
            y0 = cy - half
            y0 = max(0, min(h - w, y0))
            return frame_bgr[y0:y0 + w, :]

    def _crop_resize_to_S(frame_bgr: np.ndarray) -> np.ndarray:
        sq = _center_crop_square(frame_bgr, x_offset=int(crop_x_offset), y_offset=int(crop_y_offset))
        if sq.shape[0] != S or sq.shape[1] != S:
            interp = cv2.INTER_AREA if (sq.shape[0] > S or sq.shape[1] > S) else cv2.INTER_LINEAR
            sq = cv2.resize(sq, (S, S), interpolation=interp)
        return sq

    leaf_dirs: List[str] = []
    for cur, _, files in os.walk(root_dir):
        if "video.mp4" in files and "data.json" in files:
            leaf_dirs.append(cur)

    if not leaf_dirs:
        print("[INFO] No leaf directory contains both video.mp4 and data.json.")
        return

    for leaf in sorted(leaf_dirs):
        video_path = os.path.join(leaf, "video.mp4")
        json_path = os.path.join(leaf, "data.json")

        grandparent = os.path.basename(os.path.dirname(os.path.dirname(leaf)))
        parent = os.path.basename(os.path.dirname(leaf))
        leafname = os.path.basename(leaf)
        out_name = f"{grandparent}-{parent}-{leafname}.mp4"

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, out_name)
        else:
            out_path = os.path.join(os.path.dirname(leaf), out_name)

        out_dir = os.path.dirname(out_path)
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(leaf)))
        parent = os.path.basename(os.path.dirname(leaf))
        leafname = os.path.basename(leaf)
        base = f"{grandparent}-{parent}-{leafname}"
        frame_dir = os.path.join(out_dir, base)
        os.makedirs(frame_dir, exist_ok=True)

        txt_path = os.path.join(out_dir, base + ".txt")
        subj_key = _extract_subject_key_from_string(base) or _extract_subject_key_from_string(leaf)
        _write_description_txt(txt_path, json_path, object_info_map, subject_info_map, subject_key=subj_key)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open video: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        pressure_iter = _iter_pressure_frames(json_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w, out_h = 2 * S, S
        os.makedirs(out_dir, exist_ok=True)
        writer = cv2.VideoWriter(out_path, fourcc, float(fps), (out_w, out_h), True)

        i = 0
        written = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    pressure = next(pressure_iter)
                except StopIteration:
                    if allow_frame_mismatch:
                        break
                    else:
                        raise ValueError(f"[ERROR] Pressure frames ended early: i={i}, video_frames={total_frames_video} @ {leaf}")

                rgb_sq = _crop_resize_to_S(frame)

                if pressure.ndim == 2:
                    pressure_bgr = cv2.cvtColor(pressure, cv2.COLOR_GRAY2BGR)
                else:
                    pressure_bgr = pressure  # BGR

                if bool(mirror_RGB_video):
                    rgb_out = cv2.flip(rgb_sq, 1)
                else:
                    rgb_out = rgb_sq
                pressure_out = pressure_bgr

                combo_video = np.hstack([rgb_out, pressure_out])

                text = f"frame {i+1}/{total_frames_video}"
                overlay = combo_video.copy()
                cv2.rectangle(overlay, (8, 8), (8 + 260, 44), (60, 60, 60), thickness=-1)
                alpha = 0.5
                cv2.addWeighted(overlay, alpha, combo_video, 1 - alpha, 0, combo_video)
                cv2.putText(combo_video, text, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                writer.write(combo_video)
                written += 1
                rgb_path = os.path.join(frame_dir, f"{i:05d}_rgb.png")
                pr_path = os.path.join(frame_dir, f"{i:05d}_pressure.png")
                cv2.imwrite(rgb_path, rgb_out)
                cv2.imwrite(pr_path, pressure_out)
                i += 1

            if not allow_frame_mismatch:
                try:
                    _ = next(pressure_iter)
                    raise ValueError(f"[ERROR] Pressure frames longer than video: video_frames={total_frames_video}, written={written} @ {leaf}")
                except StopIteration:
                    pass

        finally:
            writer.release()
            cap.release()

        if not allow_frame_mismatch and total_frames_video != 0 and written != total_frames_video:
            raise ValueError(f"[ERROR] Written frames mismatch: expected={total_frames_video}, got={written} @ {leaf}")

        print(f"[OK] video: {out_path}")
        print(f"[OK] frames: {frame_dir}")
        print(f"[OK] txt:   {txt_path}")


def _build_argparser():
    p = argparse.ArgumentParser(description="Create side-by-side video (cropped RGB + pressure heatmap) and save frames/txt.")
    p.add_argument("--root_dir", type=str,
                   default="",
                   help="Root directory to search leaf folders containing video.mp4 and data.json.")
    p.add_argument("--output_dir", type=str,
                   default="",
                   help="Output directory (optional). If not set, output to leaf's parent directory.")
    p.add_argument("--object_info_json", type=str, default="./object_info.json",
                   help="Path to object_info.json (used to map weight/surface_material/fill_state by object_name).")
    p.add_argument("--subject_info_json", type=str, default="./subject_info.json", help="Optional path to subject_info.json (keyed by p001/p002/...). If provided, subject info will be appended into the prompt txt.",
    )
    p.add_argument("--crop_x_offset", type=int, default=100,
                   help="X offset (pixels) added to the crop center when center-cropping to a square. Default matches previous behavior.")
    p.add_argument("--crop_y_offset", type=int, default=0,
                   help="Y offset (pixels) added to the crop center when center-cropping to a square. Default matches previous behavior.")
    p.add_argument("--mirror_RGB_video", default=False,
                   help="If set, mirror the saved mp4 and frame PNGs left-right.")
    p.add_argument("--square_size", type=int, default=256, help="Target square size for cropped RGB and heatmap.")
    p.add_argument("--value_min", type=float, default=0.0)
    p.add_argument("--value_max", type=float, default=200.0)
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--heatmap_mode", type=str, default="gray", choices=["color", "gray"], help="Save heatmap as colored or grayscale.")
    p.add_argument("--allow_frame_mismatch", action="store_true", help="If set, use min(video_frames, pressure_frames) without raising.")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    if args.output_dir is None:
        args.output_dir=args.root_dir + "-visualization"
    make_side_by_side_videos(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        object_info_json=args.object_info_json,
        subject_info_json=args.subject_info_json,
        crop_x_offset=args.crop_x_offset,
        crop_y_offset=args.crop_y_offset,
        mirror_RGB_video=args.mirror_RGB_video,
        square_size=args.square_size,
        value_min=args.value_min,
        value_max=args.value_max,
        threshold=args.threshold,
        heatmap_mode=args.heatmap_mode,
        allow_frame_mismatch=args.allow_frame_mismatch,
    )

    def _reencode_videos_in_place(target_root: str) -> None:
        for cur, _, files in os.walk(target_root):
            mp4_files = [f for f in files if f.lower().endswith(".mp4")]
            for fname in mp4_files:
                src = os.path.join(cur, fname)
                tmp = os.path.join(cur, f"tmp_reencode_{fname}")
                try:
                    clip = moviepy.VideoFileClip(src)
                    clip.write_videofile(
                        tmp,
                        codec="libx264",
                        audio=False,
                        logger=None,
                    )
                    clip.close()
                except Exception as e:
                    print(f"[WARN] Failed to re-encode {src}: {e}")
                    continue
                try:
                    os.remove(src)
                    os.rename(tmp, src)
                except Exception as e:
                    print(f"[WARN] Failed to replace original video {src}: {e}")
                    if os.path.exists(tmp):
                        os.remove(tmp)

    _reencode_videos_in_place(args.output_dir)