import json
import math
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def file_nonempty(path):
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0


def infer_shape(value, max_depth=4):
    shape = []
    current = value
    for _ in range(max_depth):
        if not isinstance(current, list):
            break
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


def iter_numbers(value, limit=10000):
    stack = [value]
    seen = 0
    while stack and seen < limit:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            seen += 1
            yield float(item)


def numeric_stats(value, limit=10000):
    count = 0
    finite = 0
    min_value = math.inf
    max_value = -math.inf
    for number in iter_numbers(value, limit=limit):
        count += 1
        if math.isfinite(number):
            finite += 1
            min_value = min(min_value, number)
            max_value = max(max_value, number)
    return {
        "count": count,
        "finite": finite,
        "min": None if finite == 0 else min_value,
        "max": None if finite == 0 else max_value,
    }


def check_required_fields(meta, required_fields):
    return [field for field in required_fields if field not in meta]


def check_array_field(meta, field, expected_shape=None, min_value=None, max_value=None, allow_none=True):
    if field not in meta:
        return f"missing field: {field}"
    value = meta[field]
    if value is None:
        return None if allow_none else f"null field: {field}"
    if expected_shape is not None:
        shape = infer_shape(value)
        if shape[: len(expected_shape)] != tuple(expected_shape):
            return f"{field} shape {shape}, expected prefix {tuple(expected_shape)}"
    stats = numeric_stats(value)
    if stats["count"] == 0:
        return f"{field} has no numeric values"
    if stats["finite"] != stats["count"]:
        return f"{field} has non-finite values"
    if min_value is not None and stats["min"] is not None and stats["min"] < min_value:
        return f"{field} min {stats['min']:.6g} < {min_value}"
    if max_value is not None and stats["max"] is not None and stats["max"] > max_value:
        return f"{field} max {stats['max']:.6g} > {max_value}"
    return None


def resolve_sample_dir_from_opentouch(item, output_dir):
    sample_dir = item.get("sample_dir") or item.get("path")
    if sample_dir:
        return Path(sample_dir)
    if output_dir is None:
        return None
    try:
        split = item["split"]
        scene = item["scene"]
        clip = item["clip"]
        frame_idx = int(item["frame_idx"])
        is_right = int(item["is_right"])
    except Exception:
        return None
    return Path(output_dir) / split / f"{scene}_{clip}_{frame_idx:04d}_{is_right}"

