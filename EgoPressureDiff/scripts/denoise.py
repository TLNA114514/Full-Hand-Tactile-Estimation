
# -*- coding: utf-8 -*-

from typing import Dict, List, Sequence
import numpy as np

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
LH_PALM_72 = [
    207,206,205,204,203,202,201,200,199,198,197,196,
    191,190,189,188,187,186,185,184,183,182,181,180,179,178,177,
    175,174,173,172,171,170,169,168,167,166,165,164,163,162,161,
    159,158,157,156,155,154,153,152,151,150,149,148,147,146,145,
    143,142,141,140,139,138,137,136,135,134,133,132,131,130,129
]

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
RH_PALM_72 = [
     61,60,59,58,57,56,55,54,53,52,51,50,
     80,79,78,77,76,75,74,73,72,71,70,69,68,67,66,
     96,95,94,93,92,91,90,89,88,87,86,85,84,83,82,
     112,111,110,109,108,107,106,105,104,103,102,101,100,99,98,
    128,127,126,125,124,123,122,121,120,119,118,117,116,115,114
]



def _to_zero_based(unique_1_based: Sequence[int]) -> List[int]:
    return sorted({idx - 1 for idx in unique_1_based})


def _build_hand_index_maps() -> Dict[str, Dict[str, List[int]]]:
    hand_maps: Dict[str, Dict[str, List[int]]] = {}

    # Left hand
    lh_map: Dict[str, List[int]] = {}
    for finger in LH_FINGERS_12.keys():
        finger_idxs = LH_FINGERS_12[finger]
        bend_idxs   = LH_BENDS_5[finger]
        lh_map[finger] = _to_zero_based(list(finger_idxs) + list(bend_idxs))
    lh_map["palm"] = _to_zero_based(LH_PALM_72)
    hand_maps["LH"] = lh_map

    # Right hand
    rh_map: Dict[str, List[int]] = {}
    for finger in RH_FINGERS_12.keys():
        finger_idxs = RH_FINGERS_12[finger]
        bend_idxs   = RH_BENDS_5[finger]
        rh_map[finger] = _to_zero_based(list(finger_idxs) + list(bend_idxs))
    rh_map["palm"] = _to_zero_based(RH_PALM_72)
    hand_maps["RH"] = rh_map

    return hand_maps


_HAND_INDEX_MAPS = _build_hand_index_maps()


def _normalize_hand_name(hand: str) -> str:
    h = hand.strip().lower()
    if h in ("lh", "left", "l"):
        return "LH"
    if h in ("rh", "right", "r"):
        return "RH"
    raise ValueError(f"Unknown hand type: {hand!r}, expected LH/RH/left/right/l/r.")



def denoise_sensor_256(
    sensor_256: Sequence[float],
    hand: str,
    value_threshold: float = 15.0,
    finger_contact_min_points: int = 3,   
    total_contact_min_points: int = 6,   
    per_finger_min_points: int = 2,      
    palm_min_points: int = 5              
) -> np.ndarray:

    sensor_256 = np.asarray(sensor_256, dtype=float)
    if sensor_256.shape[0] != 256:
        raise ValueError(f"sensor_256 must have length 256, got {sensor_256.shape[0]}")

    hand_norm = _normalize_hand_name(hand)
    index_map = _HAND_INDEX_MAPS[hand_norm]

    finger_names = ["thumb", "index", "middle", "ring", "little"]


    finger_counts: Dict[str, int] = {}
    for finger in finger_names:
        idxs = index_map[finger]
        vals = sensor_256[idxs]
        cnt = int(np.sum(vals > value_threshold))
        finger_counts[finger] = cnt

    cond1 = any(cnt >= finger_contact_min_points for cnt in finger_counts.values())

    total_cnt = sum(finger_counts.values())
    cond2 = total_cnt >= total_contact_min_points

    hand_in_contact = cond1 or cond2

    if not hand_in_contact:
        return np.zeros_like(sensor_256)

    clean = sensor_256.copy()

    for finger in finger_names:
        idxs = index_map[finger]
        vals = clean[idxs]
        cnt = int(np.sum(vals > value_threshold))

        if cnt < per_finger_min_points:
            clean[idxs] = 0.0

    palm_idxs = index_map["palm"]
    palm_vals = clean[palm_idxs]
    palm_cnt = int(np.sum(palm_vals > value_threshold))

    if palm_cnt < palm_min_points:
        clean[palm_idxs] = 0.0

    return clean

