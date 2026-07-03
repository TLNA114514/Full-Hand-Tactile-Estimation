import json
import os
from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
CANONICAL_SPLITS = ("train", "val", "test", "test_seen", "test_unseen")

SUBDIV_OBJ_PATH = WORKSPACE_DIR / "opentouch" / "preprocess" / "scratch" / "mano_right_neutral_subdiv.obj"
SUBDIV_PALM_FACES_PATH = (
    WORKSPACE_DIR
    / "opentouch"
    / "preprocess"
    / "scratch"
    / "auto_calibrated_palm_subdiv_faces.json"
)

DATASET_ROOTS = {
    "opentouch": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "open_touch": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "ot": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "touchanything": "/data1/jiangrui/EgoTouch/extracted_frames",
    "touch_anything": "/data1/jiangrui/EgoTouch/extracted_frames",
    "egotouch": "/data1/jiangrui/EgoTouch/extracted_frames",
    "ego_touch": "/data1/jiangrui/EgoTouch/extracted_frames",
    "ta": "/data1/jiangrui/EgoTouch/extracted_frames",
    "egotactile": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
    "ego_tactile": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
    "ego": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
}


def split_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_data_dirs(datasets=None, data_dir=None):
    roots = []
    for name in split_csv(datasets):
        key = name.lower()
        if key not in DATASET_ROOTS:
            known = ", ".join(sorted(set(DATASET_ROOTS.keys())))
            raise ValueError(f"Unknown dataset name '{name}'. Known aliases: {known}")
        roots.append(DATASET_ROOTS[key])
    roots.extend(split_csv(data_dir))
    if not roots:
        roots.append(DATASET_ROOTS["opentouch"])

    deduped = []
    seen = set()
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)
    return deduped


def count_obj_vertices(obj_path=SUBDIV_OBJ_PATH):
    count = 0
    with Path(obj_path).open("r") as f:
        for line in f:
            if line.startswith("v "):
                count += 1
    return count


def load_palm_mask(tactile_dim=None):
    tactile_dim = tactile_dim or count_obj_vertices()
    with SUBDIV_PALM_FACES_PATH.open("r") as f:
        palm_data = json.load(f)

    mask = [0.0] * tactile_dim
    for triplet in palm_data["group_negative"]["face_triplets"]:
        for vid in triplet:
            if 0 <= int(vid) < tactile_dim:
                mask[int(vid)] = 1.0
    return mask


def canonical_dataset_name(value):
    raw_name = str(value or "OpenTouch")
    aliases = {
        "opentouch": "OpenTouch",
        "open_touch": "OpenTouch",
        "touchanything": "TouchAnything",
        "touch_anything": "TouchAnything",
        "egotouch": "TouchAnything",
        "ego_touch": "TouchAnything",
        "egotactile": "EgoTactile",
        "ego_tactile": "EgoTactile",
    }
    return aliases.get(raw_name.lower(), raw_name)


def split_roots(root):
    root = Path(root)
    existing = [name for name in CANONICAL_SPLITS if (root / name).is_dir()]
    if existing:
        return [(name, root / name) for name in existing]
    all_dir = root / "all"
    if all_dir.is_dir():
        return [("train", all_dir)]
    return [("train", root)]


def valid_bbox(bbox):
    if bbox is None or bbox == "null":
        return False
    try:
        vals = [float(x) for x in bbox]
    except Exception:
        return False
    if len(vals) != 4:
        return False
    return all(v == v and abs(v) != float("inf") for v in vals) and max(vals[2] - vals[0], vals[3] - vals[1]) > 1.0


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
