import os
import sys
import json
import csv
import cv2
import numpy as np
import torch
import hashlib
import time
import socket
import gc
import mmap
import ctypes
from functools import lru_cache
from array import array
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from torch.utils.data import Dataset
from yacs.config import CfgNode

try:
    import orjson
except ImportError:
    orjson = None

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))

# Global keypoint permutation for hand flipping (MediaPipe format)
FLIP_KEYPOINT_PERMUTATION = list(range(21))
CANONICAL_SPLITS = ("train", "val", "test")
INDEX_CACHE_VERSION = 7

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

SUBDIV_OBJ_PATH = os.path.join(
    workspace_dir,
    "opentouch",
    "preprocess",
    "scratch",
    "mano_right_neutral_subdiv.obj",
)
SUBDIV_PALM_FACES_PATH = os.path.join(
    workspace_dir,
    "opentouch",
    "preprocess",
    "scratch",
    "auto_calibrated_palm_subdiv_faces.json",
)


def count_obj_vertices(obj_path):
    count = 0
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                count += 1
    return count


EXPECTED_TACTILE_DIM = count_obj_vertices(SUBDIV_OBJ_PATH)


def load_subdiv_palm_mask(tactile_dim=EXPECTED_TACTILE_DIM):
    with open(SUBDIV_PALM_FACES_PATH, "r", encoding="utf-8") as handle:
        palm_data = json.load(handle)
    palm_mask = np.zeros(int(tactile_dim), dtype=np.float32)
    for triplet in palm_data["group_negative"]["face_triplets"]:
        for vertex_id in triplet:
            if 0 <= vertex_id < tactile_dim:
                palm_mask[vertex_id] = 1.0
    return palm_mask


INDEX_PALM_MASK = load_subdiv_palm_mask()
INDEX_PALM_VERTEX_MASK = INDEX_PALM_MASK > 0.5


def load_json_file(path):
    if orjson is not None:
        with open(path, "rb") as handle:
            return orjson.loads(handle.read())
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def canonical_dataset_filter(values):
    if values is None:
        return ()
    if isinstance(values, str):
        values = values.split(",")
    return tuple(sorted({
        canonical_dataset_name(value)
        for value in values
        if str(value).strip()
    }))


@lru_cache(maxsize=32)
def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def valid_bbox(bbox):
    if bbox is None or (isinstance(bbox, str) and bbox == "null"):
        return False
    try:
        arr = np.array(bbox, dtype=np.float32)
    except Exception:
        return False
    return arr.shape == (4,) and np.isfinite(arr).all() and np.all(arr[2:4] - arr[0:2] > 1.0)


def legacy_valid_bbox(bbox):
    """Reproduce the v2 index condition for data-integrity comparisons only."""
    if bbox is None or (isinstance(bbox, str) and bbox == "null"):
        return False
    try:
        arr = np.asarray(bbox, dtype=np.float32)
    except Exception:
        return False
    return arr.shape == (4,) and np.isfinite(arr).all() and np.max(arr[2:4] - arr[0:2]) > 1.0


def pressure_for_query(meta, dataset_name, hand=None, is_right=None):
    if dataset_name == "TouchAnything":
        return meta.get("hands", {}).get(hand or "", {}).get("gaussian_pressure")
    if is_right is None:
        is_right = int(meta.get("is_right", 1))
    side = "right" if int(is_right) == 1 else "left"
    pressure = meta.get("original_hdf5_data", {}).get(f"{side}_pressure_continuous_subdiv")
    return pressure if pressure is not None else meta.get("gaussian_pressure")


def pressure_array_or_none(pressure, tactile_dim=EXPECTED_TACTILE_DIM):
    if pressure is None:
        return None
    try:
        value = np.asarray(pressure, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if value.shape != (int(tactile_dim),) or not bool(np.isfinite(value).all()):
        return None
    return value


def valid_pressure(pressure, tactile_dim=EXPECTED_TACTILE_DIM):
    return pressure_array_or_none(pressure, tactile_dim=tactile_dim) is not None


def has_pressure(meta, dataset_name, hand=None, is_right=None):
    return valid_pressure(pressure_for_query(meta, dataset_name, hand=hand, is_right=is_right))


def query_sequence_key(meta, dataset_name, hand, sample_dir):
    query_alias = str(hand or ("right" if int(meta.get("is_right", 1)) else "left"))
    if dataset_name == "TouchAnything":
        parts = (
            dataset_name,
            meta.get("split", ""),
            meta.get("scene", ""),
            meta.get("task", ""),
            meta.get("clip", meta.get("rel_clip", "")),
            query_alias,
        )
    else:
        parts = (
            dataset_name,
            meta.get("split", ""),
            meta.get("scene", ""),
            meta.get("demo", ""),
            query_alias,
        )
    normalized = [str(value).strip() for value in parts if str(value).strip()]
    if len(normalized) <= 2:
        normalized.insert(-1, os.path.basename(sample_dir))
    return "/".join(normalized)


def sample_provenance(meta, dataset_name, hand, is_right, sample_dir, pressure):
    hand_meta = meta.get("hands", {}).get(hand or "", {}) if dataset_name == "TouchAnything" else {}
    if dataset_name == "TouchAnything":
        pressure_source_key = hand_meta.get("gaussian_pressure_key") or "gaussian_pressure"
        bbox_score = hand_meta.get("bbox_score", 0.0)
    else:
        side = "right" if int(is_right) == 1 else "left"
        original = meta.get("original_hdf5_data", {})
        pressure_source_key = (
            f"{side}_pressure_continuous_subdiv"
            if original.get(f"{side}_pressure_continuous_subdiv") is not None
            else "gaussian_pressure"
        )
        bbox_score = meta.get("bbox_score", 0.0)
    pressure_array = pressure if isinstance(pressure, np.ndarray) else np.asarray(pressure, dtype=np.float32)
    palm_pressure = np.clip(pressure_array[INDEX_PALM_VERTEX_MASK], 0.0, 1.0)
    return {
        "sequence_key": query_sequence_key(meta, dataset_name, hand, sample_dir),
        "frame_idx": int(meta.get("frame_idx", 0) or 0),
        "query_alias": str(hand or ("right" if int(is_right) else "left")),
        "bbox_score": float(bbox_score or 0.0),
        "pressure_source_key": str(pressure_source_key),
        "source_frame_idx": meta.get("jq_pressure_frame_index"),
        "timestamp": meta.get("timestamp"),
        "max_pressure": float(palm_pressure.max()),
        "target_volume": float(palm_pressure.sum()),
        "target_active_count": int(np.count_nonzero(palm_pressure >= 0.05)),
    }


SAM3_BBOX_SOURCE_SCHEMA = "sam3_bbox_source_v1"
BBOX_SOURCE_POLICIES = ("any", "sam3_only")


def bbox_source_for_query(meta, dataset_name, hand=None):
    if dataset_name == "TouchAnything":
        return meta.get("hands", {}).get(hand or "", {}).get("bbox_source")
    return meta.get("bbox_source")


def bbox_source_allowed(
    meta,
    dataset_name,
    hand=None,
    policy="any",
    allowed_manifest_sha256=None,
):
    policy = str(policy or "any").lower()
    if policy not in BBOX_SOURCE_POLICIES:
        raise ValueError(
            f"Unsupported bbox_source_policy={policy!r}; choose one of {BBOX_SOURCE_POLICIES}"
        )
    if policy == "any":
        return True
    source = bbox_source_for_query(meta, dataset_name, hand=hand)
    if not isinstance(source, dict) or source.get("schema") != SAM3_BBOX_SOURCE_SCHEMA:
        return False
    allowed_hashes = set(allowed_manifest_sha256 or ())
    return not allowed_hashes or source.get("source_manifest_sha256") in allowed_hashes


def _strict_samples_from_meta(
    sample_dir,
    meta,
    bbox_source_policy="any",
    allowed_manifest_sha256=None,
):
    samples = []
    dataset_name = canonical_dataset_name(meta.get("dataset", "OpenTouch"))
    if dataset_name == "TouchAnything":
        image_name = meta.get("views", {}).get("chest", "chest.jpg")
        if not os.path.isfile(os.path.join(sample_dir, image_name)):
            return []
        for hand in ("left", "right"):
            hand_meta = meta.get("hands", {}).get(hand, {})
            bbox = hand_meta.get("bbox_chest")
            is_right = int(hand_meta.get("is_right", 1 if hand == "right" else 0))
            pressure = pressure_array_or_none(pressure_for_query(meta, dataset_name, hand=hand))
            if (
                valid_bbox(bbox)
                and pressure is not None
                and bbox_source_allowed(
                    meta,
                    dataset_name,
                    hand=hand,
                    policy=bbox_source_policy,
                    allowed_manifest_sha256=allowed_manifest_sha256,
                )
            ):
                record = {
                    "sample_dir": sample_dir,
                    "dataset": dataset_name,
                    "hand": hand,
                    "is_right": is_right,
                    "bbox_source_policy": str(bbox_source_policy),
                }
                record.update(sample_provenance(meta, dataset_name, hand, is_right, sample_dir, pressure))
                samples.append(record)
    else:
        is_right = int(meta.get("is_right", 1))
        hand = "right" if is_right else "left"
        image_name = meta.get("image", "image.jpg")
        pressure = pressure_array_or_none(pressure_for_query(meta, dataset_name, is_right=is_right))
        if (
            os.path.isfile(os.path.join(sample_dir, image_name))
            and valid_bbox(meta.get("bbox"))
            and pressure is not None
            and bbox_source_allowed(
                meta,
                dataset_name,
                hand=hand,
                policy=bbox_source_policy,
                allowed_manifest_sha256=allowed_manifest_sha256,
            )
        ):
            record = {
                "sample_dir": sample_dir,
                "dataset": dataset_name,
                "hand": hand,
                "is_right": is_right,
                "bbox_source_policy": str(bbox_source_policy),
            }
            record.update(sample_provenance(meta, dataset_name, hand, is_right, sample_dir, pressure))
            samples.append(record)
    return samples


def scan_sample_dir(
    sample_dir,
    bbox_source_policy="any",
    allowed_manifest_sha256=None,
):
    if not os.path.isdir(sample_dir):
        return []
    meta_path = os.path.join(sample_dir, "meta.json")
    if not os.path.exists(meta_path):
        return []
    try:
        meta = load_json_file(meta_path)
    except Exception:
        return []
    return _strict_samples_from_meta(
        sample_dir,
        meta,
        bbox_source_policy,
        allowed_manifest_sha256=allowed_manifest_sha256,
    )


def scan_sample_dirs_batch(
    sample_dirs,
    bbox_source_policy="any",
    allowed_manifest_sha256=None,
):
    """Build compact training-index records without data-integrity sidecars."""
    samples = []
    for sample_dir in sample_dirs:
        samples.extend(
            scan_sample_dir(
                sample_dir,
                bbox_source_policy,
                allowed_manifest_sha256=allowed_manifest_sha256,
            )
        )
    return {
        "sample_dir_count": len(sample_dirs),
        "samples": samples,
    }


def sample_dirs_from_bbox_manifests(
    paths,
    *,
    split,
    expected_datasets=None,
    progress_every=100000,
    progress_callback=None,
):
    """Read candidate sample directories directly from reviewed SAM3 manifests."""

    expected = set(canonical_dataset_filter(expected_datasets))
    selected_dirs = set()
    selected_datasets = set()
    rows_seen = 0
    started = time.monotonic()
    for path in paths:
        with open(path, "rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = orjson.loads(line) if orjson is not None else json.loads(line)
                except Exception as exc:
                    raise ValueError(f"Invalid SAM3 manifest row {path}:{line_number}: {exc}") from exc
                rows_seen += 1
                if rows_seen % int(progress_every) == 0:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"[{split}] SAM3 manifests read {rows_seen} rows, selected "
                        f"{len(selected_dirs)} sample dirs ({rows_seen / elapsed:.0f} rows/s)...",
                        flush=True,
                    )
                    if progress_callback:
                        progress_callback(rows_seen)
                if str(row.get("split", "")) != str(split):
                    continue
                dataset_name = canonical_dataset_name(row.get("dataset"))
                if expected and dataset_name not in expected:
                    continue
                sample_dir = str(row.get("sample_dir") or "").strip()
                if not sample_dir:
                    continue
                selected_dirs.add(os.path.realpath(os.path.abspath(os.path.expanduser(sample_dir))))
                selected_datasets.add(dataset_name)
    missing = sorted(expected - selected_datasets)
    if missing:
        raise ValueError(
            f"SAM3 manifests contain no split={split!r} samples for expected datasets {missing}"
        )
    print(
        f"[{split}] SAM3 manifests selected {len(selected_dirs)} unique sample dirs "
        f"from {rows_seen} rows.",
        flush=True,
    )
    if progress_callback:
        progress_callback(rows_seen)
    return sorted(selected_dirs)


def _audit_record_from_sample(sample, meta):
    dataset_name = sample["dataset"]
    hand = sample["hand"]
    is_right = int(sample["is_right"])
    pressure = pressure_for_query(meta, dataset_name, hand=hand, is_right=is_right)
    pressure_array = np.clip(np.asarray(pressure, dtype=np.float32), 0.0, 1.0)
    masked_pressure = pressure_array * INDEX_PALM_MASK
    if dataset_name == "TouchAnything":
        hand_meta = meta.get("hands", {}).get(hand, {})
        bbox = hand_meta.get("bbox_chest")
        image_name = meta.get("views", {}).get("chest", "chest.jpg")
        other_hand = "right" if hand == "left" else "left"
        other_meta = meta.get("hands", {}).get(other_hand, {})
        other_bbox = other_meta.get("bbox_chest") if valid_bbox(other_meta.get("bbox_chest")) else None
        other_pressure = other_meta.get("gaussian_pressure")
        other_volume = 0.0
        if valid_pressure(other_pressure):
            other_array = np.clip(np.asarray(other_pressure, dtype=np.float32), 0.0, 1.0)
            other_volume = float((other_array * INDEX_PALM_MASK).sum())
        required_fields = ("frame_idx", "scene", "task", "clip")
    else:
        bbox = meta.get("bbox")
        image_name = meta.get("image", "image.jpg")
        other_bbox = None
        other_volume = 0.0
        required_fields = ("frame_idx", "scene", "demo")
    bbox_array = np.asarray(bbox, dtype=np.float32)
    missing_fields = [name for name in required_fields if meta.get(name) in (None, "")]
    return {
        "sample_dir": sample["sample_dir"],
        "dataset": dataset_name,
        "hand": hand,
        "is_right": is_right,
        "sequence_key": sample["sequence_key"],
        "query_alias": sample["query_alias"],
        "frame_idx": sample["frame_idx"],
        "bbox_score": sample["bbox_score"],
        "pressure_source_key": sample["pressure_source_key"],
        "source_frame_idx": sample.get("source_frame_idx"),
        "timestamp": sample.get("timestamp"),
        "query_bbox": bbox_array.tolist(),
        "co_visible_bbox": None if other_bbox is None else np.asarray(other_bbox, dtype=np.float32).tolist(),
        "co_visible_gt_volume": other_volume,
        "image_name": image_name,
        "target_checksum": hashlib.sha256(np.ascontiguousarray(masked_pressure).tobytes()).hexdigest(),
        "target_volume": float(masked_pressure.sum()),
        "target_active_count": int((masked_pressure >= 0.05).sum()),
        "max_pressure": float(masked_pressure.max()),
        "metadata_missing_fields": missing_fields,
        "pressure_npz_frame_index": meta.get("pressure_npz_frame_index", meta.get("npz_array_index")),
        "rgb_timestamp": meta.get("rgb_timestamp", meta.get("frame_timestamp")),
        "pressure_timestamp": meta.get("pressure_timestamp"),
    }


def scan_sample_dir_integrity(sample_dir):
    """Report records admitted by the old index but rejected by the strict index."""
    empty_result = {
        "samples": [],
        "audit_rows": [],
        "legacy_candidate_count": 0,
        "strict_sample_count": 0,
        "rejections": [],
    }
    meta_path = os.path.join(sample_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return empty_result
    try:
        meta = load_json_file(meta_path)
    except Exception:
        return empty_result

    dataset_name = canonical_dataset_name(meta.get("dataset", "OpenTouch"))
    strict_samples = _strict_samples_from_meta(sample_dir, meta)
    strict_keys = {(sample["hand"], int(sample["is_right"])) for sample in strict_samples}
    candidates = []
    if dataset_name == "TouchAnything":
        image_name = meta.get("views", {}).get("chest", "chest.jpg")
        for hand in ("left", "right"):
            hand_meta = meta.get("hands", {}).get(hand, {})
            is_right = int(hand_meta.get("is_right", 1 if hand == "right" else 0))
            pressure = hand_meta.get("gaussian_pressure")
            bbox = hand_meta.get("bbox_chest")
            if legacy_valid_bbox(bbox) and pressure is not None:
                candidates.append((hand, is_right, bbox, pressure, image_name))
    else:
        is_right = int(meta.get("is_right", 1))
        hand = "right" if is_right else "left"
        pressure = pressure_for_query(meta, dataset_name, is_right=is_right)
        bbox = meta.get("bbox")
        if legacy_valid_bbox(bbox) and pressure is not None:
            candidates.append((hand, is_right, bbox, pressure, meta.get("image", "image.jpg")))

    rejections = []
    for hand, is_right, bbox, pressure, image_name in candidates:
        if (hand, is_right) in strict_keys:
            continue
        reasons = []
        if not os.path.isfile(os.path.join(sample_dir, image_name)):
            reasons.append("image_missing")
        if not valid_bbox(bbox):
            reasons.append("bbox_invalid_strict")
        if not valid_pressure(pressure):
            reasons.append("pressure_invalid")
        rejections.append({
            "sample_dir": sample_dir,
            "dataset": dataset_name,
            "hand": hand,
            "is_right": is_right,
            "reason": ",".join(reasons) or "unknown_strict_rejection",
        })
    return {
        "samples": strict_samples,
        "audit_rows": [_audit_record_from_sample(sample, meta) for sample in strict_samples],
        "legacy_candidate_count": len(candidates),
        "strict_sample_count": len(strict_samples),
        "rejections": rejections,
    }


def scan_sample_dirs_integrity_batch(sample_dirs):
    """Scan and serialize a directory batch inside one worker process."""
    samples = []
    audit_lines = []
    legacy_candidate_count = 0
    rejections = []
    for sample_dir in sample_dirs:
        result = scan_sample_dir_integrity(sample_dir)
        samples.extend(result["samples"])
        legacy_candidate_count += int(result["legacy_candidate_count"])
        rejections.extend(result["rejections"])
        audit_lines.extend(
            json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n"
            for row in result["audit_rows"]
        )
    return {
        "sample_dir_count": len(sample_dirs),
        "samples": samples,
        "audit_jsonl": "".join(audit_lines),
        "legacy_candidate_count": legacy_candidate_count,
        "strict_sample_count": len(samples),
        "rejections": rejections,
    }


def filter_sample_groups_by_bbox_source_batch(payload):
    """Filter compact-index groups while reading each sample meta.json only once."""

    sample_groups, policy = payload
    kept = []
    rejected = 0
    for group in sample_groups:
        if not group:
            continue
        meta_path = os.path.join(group[0]["sample_dir"], "meta.json")
        try:
            meta = load_json_file(meta_path)
        except Exception:
            rejected += len(group)
            continue
        for sample in group:
            dataset_name = canonical_dataset_name(
                sample.get("dataset", meta.get("dataset", "OpenTouch"))
            )
            if bbox_source_allowed(
                meta,
                dataset_name,
                hand=sample.get("hand"),
                policy=policy,
            ):
                kept.append({**sample, "bbox_source_policy": policy})
            else:
                rejected += 1
    return {"samples": kept, "rejected": rejected}


COMPACT_INDEX_FIELDS = (
    "sample_dir",
    "dataset",
    "hand",
    "is_right",
    "sequence_key",
    "frame_idx",
    "query_alias",
    "bbox_score",
    "pressure_source_key",
    "source_frame_idx",
    "timestamp",
    "max_pressure",
    "target_volume",
    "target_active_count",
)


def _optional_manifest_value(value):
    value = str(value or "").strip()
    return value if value else None


def read_compact_index_from_audit_csv(
    path,
    split,
    data_roots,
    expected_datasets=None,
    progress_label=None,
    progress_every=100000,
    progress_callback=None,
):
    """Convert a completed data-integrity CSV into the compact training index."""
    path = os.path.abspath(path)
    stem_summary_path = os.path.splitext(path)[0] + ".summary.json"
    summary_path = (
        stem_summary_path
        if os.path.isfile(stem_summary_path)
        else os.path.join(os.path.dirname(path), "summary.json")
    )
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Audit manifest summary is missing: {summary_path}")
    summary = load_json_file(summary_path)
    blocking_reasons = summary.get("blocking_reasons", [])
    if blocking_reasons:
        raise ValueError(f"Audit manifest has blocking reasons: {blocking_reasons}")
    for name in (
        "target_mismatch_count",
        "indexed_invalid_bbox_count",
        "indexed_sample_failure_count",
        "jpeg_decode_failure_count",
    ):
        if int(summary.get(name, 0)) != 0:
            raise ValueError(f"Audit manifest is not clean: {name}={summary.get(name)}")

    split_summary = next(
        (item for item in summary.get("split_summaries", []) if item.get("split") == split),
        None,
    )
    if split_summary is None:
        raise ValueError(f"Audit manifest does not contain split={split!r}")
    expected_count = int(split_summary.get("indexed_samples", split_summary.get("audited_samples", -1)))
    if expected_count < 0:
        raise ValueError(f"Audit manifest has no indexed sample count for split={split!r}")

    expected_dataset_filter = canonical_dataset_filter(expected_datasets)
    manifest_dataset_filter = canonical_dataset_filter(summary.get("dataset_filter"))
    dataset_counts = summary.get("dataset_counts")
    manifest_dataset_coverage = manifest_dataset_filter
    if not manifest_dataset_coverage and isinstance(dataset_counts, dict):
        try:
            manifest_dataset_coverage = canonical_dataset_filter(
                dataset
                for dataset, count in dataset_counts.items()
                if int(count) > 0
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Audit manifest has invalid dataset_counts: {dataset_counts!r}") from exc
    if expected_dataset_filter and not manifest_dataset_coverage:
        raise ValueError(
            "Audit manifest does not declare dataset_filter or usable dataset_counts; "
            f"cannot verify requested filter {expected_dataset_filter}"
        )
    if expected_dataset_filter and manifest_dataset_coverage != expected_dataset_filter:
        raise ValueError(
            "Audit manifest dataset filter does not match requested filter: "
            f"manifest={manifest_dataset_coverage}, requested={expected_dataset_filter}"
        )

    expected_roots = {os.path.realpath(os.path.abspath(root)) for root in data_roots}
    manifest_roots = {
        os.path.realpath(os.path.abspath(root)) for root in split_summary.get("roots", [])
    }
    root_remap = None
    if manifest_roots != expected_roots and (
        len(manifest_roots) == 1
        and len(expected_roots) == 1
        and len(manifest_dataset_coverage) == 1
    ):
        root_remap = (next(iter(manifest_roots)), next(iter(expected_roots)))
        print(
            f"[{split}] Relocating verified {manifest_dataset_coverage[0]} manifest root: "
            f"{root_remap[0]} -> {root_remap[1]}",
            flush=True,
        )
    elif manifest_roots != expected_roots:
        raise ValueError(
            "Audit manifest roots do not match requested data roots: "
            f"manifest={sorted(manifest_roots)}, requested={sorted(expected_roots)}"
        )

    samples = []
    rows_seen = 0
    start = time.monotonic()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_dir",
            "dataset",
            "split",
            "sequence_key",
            "query_alias",
            "is_right",
            "frame_idx",
            "bbox_score",
            "pressure_source_key",
            "target_volume",
            "target_active_count",
            "max_pressure",
        }
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"Audit manifest is missing required columns: {missing}")
        for row in reader:
            rows_seen += 1
            row_dataset = canonical_dataset_name(row["dataset"])
            if expected_dataset_filter and row_dataset not in expected_dataset_filter:
                raise ValueError(
                    f"Audit manifest contains unexpected dataset {row_dataset!r}; "
                    f"expected only {expected_dataset_filter}"
                )
            if row["split"] == split:
                query_alias = str(row["query_alias"] or "query")
                sample_dir = os.path.realpath(os.path.abspath(row["sample_dir"]))
                if root_remap is not None:
                    source_root, target_root = root_remap
                    relative = os.path.relpath(sample_dir, source_root)
                    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
                        raise ValueError(
                            f"Manifest sample is outside declared root {source_root}: {sample_dir}"
                        )
                    sample_dir = os.path.join(target_root, relative)
                samples.append({
                    "sample_dir": sample_dir,
                    "dataset": row_dataset,
                    "hand": query_alias,
                    "is_right": int(row["is_right"]),
                    "sequence_key": row["sequence_key"],
                    "frame_idx": int(row["frame_idx"]),
                    "query_alias": query_alias,
                    "bbox_score": float(row["bbox_score"] or 0.0),
                    "pressure_source_key": row["pressure_source_key"],
                    "source_frame_idx": _optional_manifest_value(row.get("jq_pressure_frame_idx")),
                    "timestamp": _optional_manifest_value(row.get("timestamp")),
                    "max_pressure": float(row["max_pressure"]),
                    "target_volume": float(row["target_volume"]),
                    "target_active_count": int(row["target_active_count"]),
                })
            if rows_seen % int(progress_every) == 0:
                if progress_label:
                    elapsed = max(time.monotonic() - start, 1e-6)
                    print(
                        f"{progress_label} read {rows_seen} audit rows, selected {len(samples)} "
                        f"({rows_seen / elapsed:.0f} rows/s)...",
                        flush=True,
                    )
                if progress_callback:
                    progress_callback(rows_seen)
    if len(samples) != expected_count:
        raise ValueError(
            f"Audit manifest selected {len(samples)} samples for split={split}, expected {expected_count}"
        )
    return samples


def read_compact_index_from_integrity_sidecar(
    path,
    progress_label=None,
    progress_every=100000,
    progress_callback=None,
):
    samples = []
    start = time.monotonic()
    with open(path, "rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = orjson.loads(line) if orjson is not None else json.loads(line)
            missing = [field for field in COMPACT_INDEX_FIELDS if field not in row]
            if missing:
                raise ValueError(
                    f"Integrity sidecar row {line_number} is missing compact index fields: {missing}"
                )
            samples.append({field: row[field] for field in COMPACT_INDEX_FIELDS})
            if progress_label and line_number % int(progress_every) == 0:
                elapsed = max(time.monotonic() - start, 1e-6)
                print(
                    f"{progress_label} converted {line_number} records "
                    f"({line_number / elapsed:.0f} records/s)...",
                    flush=True,
                )
            if progress_callback and line_number % int(progress_every) == 0:
                progress_callback(line_number)
    return samples


def write_jsonl_atomic(
    path,
    rows,
    progress_label=None,
    progress_every=100000,
    progress_callback=None,
):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    start = time.monotonic()
    open_kwargs = {} if orjson is not None else {"encoding": "utf-8"}
    mode = "wb" if orjson is not None else "w"
    with open(tmp_path, mode, **open_kwargs) as f:
        for index, row in enumerate(rows, start=1):
            if orjson is not None:
                f.write(orjson.dumps(row) + b"\n")
            else:
                f.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
            if progress_label and index % int(progress_every) == 0:
                elapsed = max(time.monotonic() - start, 1e-6)
                print(
                    f"{progress_label} wrote {index} records ({index / elapsed:.0f} records/s)...",
                    flush=True,
                )
            if progress_callback and index % int(progress_every) == 0:
                progress_callback(index)
    os.replace(tmp_path, path)


def write_json_atomic(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def read_jsonl(path, progress_label=None, progress_every=100000):
    rows = []
    start = time.monotonic()
    with open(path, "rb") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                rows.append(orjson.loads(line) if orjson is not None else json.loads(line))
            if progress_label and line_number % int(progress_every) == 0:
                elapsed = max(time.monotonic() - start, 1e-6)
                print(
                    f"{progress_label} loaded {line_number} records "
                    f"({line_number / elapsed:.0f} records/s)...",
                    flush=True,
                )
    return rows


class MMapJsonlRecords:
    """Random-access JSONL records without a per-worker Python object graph."""

    def __init__(self, path):
        self.path = os.path.realpath(os.path.abspath(path))
        self._mapping = None
        self._offsets = None
        self._open()

    def _open(self):
        file_size = os.path.getsize(self.path)
        offsets = array("Q", [0])
        if file_size == 0:
            self._mapping = None
            self._offsets = offsets
            return
        with open(self.path, "rb") as handle:
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        position = 0
        while True:
            newline = mapping.find(b"\n", position)
            if newline < 0:
                if position < file_size:
                    offsets.append(file_size)
                break
            position = newline + 1
            offsets.append(position)
            if position >= file_size:
                break
        self._mapping = mapping
        self._offsets = offsets

    def __len__(self):
        return max(0, len(self._offsets) - 1)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        raw = self._mapping[start:end].strip()
        return orjson.loads(raw) if orjson is not None else json.loads(raw)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getstate__(self):
        return {"path": self.path}

    def __setstate__(self, state):
        self.path = state["path"]
        self._mapping = None
        self._offsets = None
        self._open()

    def close(self):
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def release_unused_python_heap():
    """Return freed eager-index arenas before DataLoader workers are forked."""

    gc.collect()
    try:
        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim(0)
    except Exception:
        pass


def ddp_global_rank():
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            local_rank = int(local_rank)
            node_rank = int(os.environ.get("NODE_RANK", os.environ.get("GROUP_RANK", "0")))
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
            return node_rank * local_world_size + local_rank
        except ValueError:
            pass
    return 0


def wait_for_file(path, timeout_sec=3600, poll_sec=5, progress_label=None):
    start = time.time()
    last_report = start
    while True:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        now = time.time()
        if now - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for index cache: {path}")
        if progress_label and now - last_report >= 30.0:
            print(f"{progress_label} still waiting ({now - start:.0f}s elapsed)...", flush=True)
            last_report = now
        time.sleep(poll_sec)


def wait_for_shared_cache(
    cache_path,
    done_path,
    lock_dir,
    timeout_sec=3600,
    poll_sec=5,
    stale_heartbeat_sec=600,
):
    start = time.time()
    last_report = start
    while True:
        builder_active = os.path.isdir(lock_dir)
        cache_ready = os.path.isfile(cache_path) and os.path.isfile(done_path)
        if not builder_active and cache_ready:
            return
        now = time.time()
        if now - last_report >= 30.0:
            owner = SharedCacheBuildLock(lock_dir).owner_description()
            print(
                f"Waiting for shared index cache ({now - start:.0f}s elapsed; {owner})...",
                flush=True,
            )
            last_report = now
        owner = SharedCacheBuildLock.read_owner(lock_dir)
        heartbeat = owner.get("heartbeat_unix", owner.get("created_unix")) if owner else None
        if builder_active and heartbeat is not None and now - float(heartbeat) > stale_heartbeat_sec:
            raise RuntimeError(
                f"Shared index builder heartbeat is stale for {now - float(heartbeat):.0f}s: "
                f"{lock_dir} ({SharedCacheBuildLock(lock_dir).owner_description()}). "
                "Verify that the builder process has stopped, then remove this stale lock directory."
            )
        if now - start > timeout_sec:
            raise TimeoutError(
                f"Timed out waiting for shared index cache: {cache_path} "
                f"(builder_active={builder_active})"
            )
        time.sleep(poll_sec)


class SharedCacheBuildLock:
    """Cross-host cache-build lock backed by an atomic shared-filesystem mkdir."""

    def __init__(self, lock_dir, timeout_sec=3600, poll_sec=5):
        self.lock_dir = os.path.abspath(lock_dir)
        self.timeout_sec = int(timeout_sec)
        self.poll_sec = float(poll_sec)
        self.acquired = False
        self._owner = None

    def try_acquire(self):
        try:
            os.makedirs(self.lock_dir)
        except FileExistsError:
            if not self._reclaim_dead_local_owner():
                return False
            try:
                os.makedirs(self.lock_dir)
            except FileExistsError:
                return False
        self.acquired = True
        self._owner = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_unix": time.time(),
            "heartbeat_unix": time.time(),
            "stage": "acquired",
        }
        try:
            self._write_owner()
        except Exception:
            self.release()
            raise
        return True

    @staticmethod
    def read_owner(lock_dir):
        owner_path = os.path.join(os.path.abspath(lock_dir), "owner.json")
        try:
            with open(owner_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    @staticmethod
    def _pid_is_alive(pid):
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except (PermissionError, ValueError, TypeError):
            return True
        return True

    def _reclaim_dead_local_owner(self):
        owner = self.read_owner(self.lock_dir)
        if not owner or owner.get("hostname") != socket.gethostname():
            return False
        pid = owner.get("pid")
        if self._pid_is_alive(pid):
            return False
        stale_dir = f"{self.lock_dir}.stale.{int(time.time())}.{pid}"
        try:
            os.replace(self.lock_dir, stale_dir)
        except (FileNotFoundError, FileExistsError, OSError):
            return False
        try:
            os.remove(os.path.join(stale_dir, "owner.json"))
        except FileNotFoundError:
            pass
        try:
            os.rmdir(stale_dir)
        except OSError:
            pass
        print(
            f"Reclaimed stale local index lock from dead pid={pid}: {self.lock_dir}",
            flush=True,
        )
        return True

    def _write_owner(self):
        owner_path = os.path.join(self.lock_dir, "owner.json")
        tmp_path = f"{owner_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._owner, handle, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, owner_path)

    def heartbeat(self, stage, completed=None, total=None):
        if not self.acquired:
            return
        self._owner["heartbeat_unix"] = time.time()
        self._owner["stage"] = str(stage)
        if completed is not None:
            self._owner["completed"] = int(completed)
        if total is not None:
            self._owner["total"] = int(total)
        self._write_owner()

    def release(self):
        if not self.acquired:
            return
        owner_path = os.path.join(self.lock_dir, "owner.json")
        try:
            os.remove(owner_path)
        except FileNotFoundError:
            pass
        try:
            for name in os.listdir(self.lock_dir):
                if name.startswith(f"owner.json.tmp.{os.getpid()}"):
                    os.remove(os.path.join(self.lock_dir, name))
        except FileNotFoundError:
            pass
        try:
            os.rmdir(self.lock_dir)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"Warning: could not remove index lock directory {self.lock_dir}: {exc}", flush=True)
        self.acquired = False
        self._owner = None

    def owner_description(self):
        try:
            owner = self.read_owner(self.lock_dir)
            if owner is None:
                raise ValueError("owner metadata unavailable")
            heartbeat = owner.get("heartbeat_unix", owner.get("created_unix"))
            age = time.time() - float(heartbeat) if heartbeat is not None else float("nan")
            return (
                f"host={owner.get('hostname')}, pid={owner.get('pid')}, "
                f"stage={owner.get('stage', 'unknown')}, heartbeat_age={age:.0f}s"
            )
        except Exception:
            return "owner metadata unavailable"


class OpenTouchTactileDataset(Dataset):
    def __init__(self, cfg: CfgNode, split: str = "train", 
                 data_dir: str = None, train: bool = True, index_workers: int = 1,
                 index_chunksize: int = 256, index_cache_dir: str = None,
                 rebuild_index: bool = False, index_cache_timeout: int = 3600,
                 index_backend: str = "process",
                 sample_records=None, tactile_only: bool = False,
                 bbox_rescale_factor: float = 2.0,
                 bbox_source_policy: str = "any",
                 bbox_manifests=None,
                 lazy_index_records: bool = False,
                 augmentation_enabled: bool = True,
                 index_process_worker_cap: int = 64,
                 index_manifest: str = None,
                 expected_datasets=None):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.train = train
        self.augmentation_enabled = bool(augmentation_enabled)
        self.tactile_only = bool(tactile_only)
        self.index_workers = max(1, int(index_workers))
        self.index_chunksize = max(1, int(index_chunksize))
        self.index_backend = str(index_backend or "process").lower()
        if self.index_backend not in ("process", "thread"):
            raise ValueError(f"Unsupported index_backend: {index_backend!r}. Use 'process' or 'thread'.")
        self.index_cache_dir = index_cache_dir
        self.index_process_worker_cap = max(0, int(index_process_worker_cap))
        self.index_manifest = (
            os.path.abspath(os.path.expanduser(str(index_manifest)))
            if index_manifest and str(index_manifest).strip()
            else None
        )
        self.expected_datasets = canonical_dataset_filter(expected_datasets)
        self.index_manifest_sha256 = (
            sha256_file(self.index_manifest)
            if self.index_manifest and os.path.isfile(self.index_manifest)
            else ""
        )
        self.rebuild_index = bool(rebuild_index)
        self.index_cache_timeout = int(index_cache_timeout)
        self._active_cache_lock = None
        self.lazy_index_records = bool(lazy_index_records)
        
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.rescale_factor = float(bbox_rescale_factor)
        if not 1.0 <= self.rescale_factor <= 4.0:
            raise ValueError("bbox_rescale_factor must lie in [1.0, 4.0]")
        self.bbox_source_policy = str(bbox_source_policy or "any").lower()
        if self.bbox_source_policy not in BBOX_SOURCE_POLICIES:
            raise ValueError(
                f"bbox_source_policy must be one of {BBOX_SOURCE_POLICIES}, "
                f"got {bbox_source_policy!r}"
            )
        if isinstance(bbox_manifests, str):
            bbox_manifests = bbox_manifests.split(",")
        self.bbox_manifests = tuple(
            os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
            for path in (bbox_manifests or ())
            if str(path).strip()
        )
        missing_bbox_manifests = [
            path for path in self.bbox_manifests if not os.path.isfile(path)
        ]
        if missing_bbox_manifests:
            raise FileNotFoundError(
                "SAM3 bbox manifest(s) are missing: " + ", ".join(missing_bbox_manifests)
            )
        self.bbox_manifest_sha256 = {
            path: sha256_file(path) for path in self.bbox_manifests
        }
        
        if data_dir is None:
            data_dirs = ["/data1/jiangrui/OpenTouch Data/extracted_dataset"]
        elif isinstance(data_dir, (list, tuple)):
            data_dirs = [str(d) for d in data_dir if str(d).strip()]
        else:
            data_dirs = [d.strip() for d in str(data_dir).split(",") if d.strip()]
        self.data_dirs = data_dirs
            
        self.tactile_dim = count_obj_vertices(SUBDIV_OBJ_PATH)
        print(f"[{split}] Loading subdiv palm mask for evaluation and loss masking...")
        self.palm_mask = self._load_palm_mask()
        
        if sample_records is None:
            self.samples = self._load_or_build_index()
        else:
            self.samples = list(sample_records)
            if self.bbox_source_policy != "any" and not all(
                sample.get("bbox_source_policy") == self.bbox_source_policy
                for sample in self.samples
            ):
                self.samples = self._filter_samples_by_bbox_source(self.samples)
        self._validate_dataset_filter()
        source_counts = {}
        if isinstance(self.samples, MMapJsonlRecords):
            source_counts = {"mmap_records": len(self.samples)}
        else:
            for sample in self.samples:
                source_counts[sample["dataset"]] = source_counts.get(sample["dataset"], 0) + 1
        print(f"[{split}] Loaded {len(self.samples)} hand samples from {len(self.data_dirs)} root(s): {source_counts}")

    def _validate_dataset_filter(self):
        if not self.expected_datasets:
            return
        if isinstance(self.samples, MMapJsonlRecords):
            if len(self.samples) == 0:
                raise RuntimeError(
                    f"[{self.split}] Dataset filter validation failed: mmap index is empty"
                )
            # The cache key binds the expected filter and construction validates
            # every manifest row before this read-only view is created.
            return
        observed = {canonical_dataset_name(sample.get("dataset")) for sample in self.samples}
        expected = set(self.expected_datasets)
        unexpected = sorted(observed - expected)
        missing = sorted(expected - observed)
        if unexpected or missing:
            raise RuntimeError(
                f"[{self.split}] Dataset filter validation failed: expected={sorted(expected)}, "
                f"observed={sorted(observed)}, unexpected={unexpected}, missing={missing}"
            )

    def _load_palm_mask(self):
        return load_subdiv_palm_mask(self.tactile_dim)

    def _has_sample_dirs(self, path):
        if not os.path.isdir(path):
            return False
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir() and os.path.exists(os.path.join(entry.path, "meta.json")):
                    return True
        return False

    def _split_dir(self, root):
        split_path = os.path.join(root, self.split)
        if os.path.isdir(split_path):
            return split_path

        has_any_split = any(os.path.isdir(os.path.join(root, name)) for name in CANONICAL_SPLITS)
        if has_any_split:
            return None

        if self.split != "train":
            return None

        all_path = os.path.join(root, "all")
        if self._has_sample_dirs(all_path):
            print(f"[{self.split}] No train/val/test under {root}; using {all_path} as train split.")
            return all_path

        if os.path.isdir(root):
            print(f"[{self.split}] No train/val/test under {root}; using the full root as train split.")
            return root

        return None

    def _infer_dataset_name(self, meta):
        return canonical_dataset_name(meta.get("dataset", "OpenTouch"))

    def _valid_bbox(self, bbox):
        return valid_bbox(bbox)

    def _has_pressure(self, meta, dataset_name, hand=None, is_right=None):
        return has_pressure(meta, dataset_name, hand=hand, is_right=is_right)

    def _cache_path(self):
        if not self.index_cache_dir:
            return None
        key_data = {
            "split": self.split,
            "data_dirs": [os.path.abspath(path) for path in self.data_dirs],
            "version": INDEX_CACHE_VERSION,
            "manifest_sha256": self.index_manifest_sha256,
            "dataset_filter": list(self.expected_datasets),
            "bbox_source_policy": self.bbox_source_policy,
            "bbox_manifest_sha256": getattr(self, "bbox_manifest_sha256", {}),
        }
        digest = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.index_cache_dir, f"{self.split}_{digest}.jsonl")

    def index_cache_metadata(self):
        cache_path = self._cache_path()
        cache_key = os.path.splitext(os.path.basename(cache_path))[0] if cache_path else "uncached"
        return {
            "index_schema_version": INDEX_CACHE_VERSION,
            "index_cache_key": cache_key,
            "indexed_sample_count": len(self.samples),
            "index_manifest_sha256": self.index_manifest_sha256,
            "dataset_filter": list(self.expected_datasets),
            "bbox_source_policy": self.bbox_source_policy,
            "bbox_manifest_sha256": dict(getattr(self, "bbox_manifest_sha256", {})),
            "lazy_index_records": bool(getattr(self, "lazy_index_records", False)),
        }

    def _read_index_cache(self, cache_path, progress_label):
        if bool(getattr(self, "lazy_index_records", False)):
            print(
                f"[{self.split}] Opening memory-mapped index cache: {cache_path}",
                flush=True,
            )
            return MMapJsonlRecords(cache_path)
        return read_jsonl(cache_path, progress_label=progress_label)

    def _filter_samples_by_bbox_source(self, samples):
        if self.bbox_source_policy == "any" or not samples:
            return samples
        grouped = {}
        for sample in samples:
            grouped.setdefault(sample["sample_dir"], []).append(sample)
        groups = [grouped[path] for path in sorted(grouped)]
        batch_size = max(1, self.index_chunksize)
        batches = [groups[start : start + batch_size] for start in range(0, len(groups), batch_size)]
        workers = min(max(1, self.index_workers), 64)
        kept = []
        rejected = 0
        print(
            f"[{self.split}] Enforcing bbox_source_policy={self.bbox_source_policy} "
            f"on {len(samples)} compact index records...",
            flush=True,
        )
        if workers == 1:
            results = [
                filter_sample_groups_by_bbox_source_batch((batch, self.bbox_source_policy))
                for batch in batches
            ]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(
                    filter_sample_groups_by_bbox_source_batch,
                    ((batch, self.bbox_source_policy) for batch in batches),
                )
        for result in results:
            kept.extend(result["samples"])
            rejected += int(result["rejected"])
        print(
            f"[{self.split}] SAM3 bbox provenance retained {len(kept)}/{len(samples)} "
            f"records; rejected {rejected} legacy/unmatched records.",
            flush=True,
        )
        return kept

    def integrity_cache_path(self):
        cache_path = self._cache_path()
        return f"{cache_path}.integrity.jsonl" if cache_path else None

    def integrity_summary_path(self):
        cache_path = self._cache_path()
        return f"{cache_path}.integrity.json" if cache_path else None

    def _load_or_build_index(self):
        cache_path = self._cache_path()
        if cache_path is None:
            return self._build_index()

        done_path = f"{cache_path}.done"
        rank = ddp_global_rank()
        if os.path.exists(cache_path) and os.path.exists(done_path) and not self.rebuild_index:
            print(f"[{self.split}] Loading index cache: {cache_path}")
            return self._read_index_cache(cache_path, f"[{self.split}] Index cache")

        if rank == 0:
            lock = SharedCacheBuildLock(f"{cache_path}.lock", timeout_sec=self.index_cache_timeout)
            if lock.try_acquire():
                print(f"[{self.split}] Building index cache under shared lock: {cache_path}")
                try:
                    if self.rebuild_index:
                        for path in (
                            cache_path,
                            done_path,
                        ):
                            try:
                                os.remove(path)
                            except FileNotFoundError:
                                pass
                    elif os.path.exists(cache_path) and os.path.exists(done_path):
                        return self._read_index_cache(cache_path, f"[{self.split}] Index cache")
                    self._active_cache_lock = lock
                    samples = None
                    if self.index_manifest and os.path.isfile(self.index_manifest):
                        try:
                            print(
                                f"[{self.split}] Building compact index from verified audit manifest: "
                                f"{self.index_manifest}",
                                flush=True,
                            )
                            samples = read_compact_index_from_audit_csv(
                                self.index_manifest,
                                split=self.split,
                                data_roots=self.data_dirs,
                                expected_datasets=self.expected_datasets,
                                progress_label=f"[{self.split}] Audit manifest",
                                progress_callback=lambda count: lock.heartbeat(
                                    "converting_audit_manifest",
                                    completed=count,
                                ),
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"[{self.split}] Verified audit manifest cannot be used: "
                                f"{self.index_manifest}: {exc}"
                            ) from exc
                    elif self.index_manifest and self.expected_datasets:
                        raise FileNotFoundError(
                            f"[{self.split}] Required single-domain audit manifest not found: "
                            f"{self.index_manifest}"
                        )
                    elif self.index_manifest:
                        print(
                            f"[{self.split}] Audit manifest not found at {self.index_manifest}; "
                            "falling back to an integrity sidecar or compact metadata scan.",
                            flush=True,
                        )
                    integrity_path = self.integrity_cache_path()
                    integrity_summary_path = self.integrity_summary_path()
                    if (
                        samples is None
                        and os.path.isfile(integrity_path)
                        and os.path.isfile(integrity_summary_path)
                    ):
                        try:
                            integrity_summary = load_json_file(integrity_summary_path)
                            expected_count = int(integrity_summary["strict_sample_count"])
                            print(
                                f"[{self.split}] Reusing completed integrity sidecar to build compact index: "
                                f"{integrity_path}",
                                flush=True,
                            )
                            samples = read_compact_index_from_integrity_sidecar(
                                integrity_path,
                                progress_label=f"[{self.split}] Integrity sidecar",
                                progress_callback=lambda count: lock.heartbeat(
                                    "converting_integrity_sidecar",
                                    completed=count,
                                    total=expected_count,
                                ),
                            )
                            if len(samples) != expected_count:
                                raise ValueError(
                                    f"sidecar contains {len(samples)} records, expected {expected_count}"
                                )
                        except Exception as exc:
                            print(
                                f"[{self.split}] Integrity sidecar reuse failed ({exc}); "
                                "falling back to a compact metadata scan.",
                                flush=True,
                            )
                            samples = None
                    if samples is None:
                        samples = self._build_index()
                    else:
                        samples = self._filter_samples_by_bbox_source(samples)
                        lock.heartbeat("sorting", completed=len(samples), total=len(samples))
                        print(
                            f"[{self.split}] Sorting {len(samples)} compact index records...",
                            flush=True,
                        )
                        samples.sort(key=lambda item: (item["sample_dir"], item["hand"]))
                    lock.heartbeat("writing_cache", completed=0, total=len(samples))
                    print(
                        f"[{self.split}] Writing {len(samples)} compact index records to {cache_path}...",
                        flush=True,
                    )
                    write_jsonl_atomic(
                        cache_path,
                        samples,
                        progress_label=f"[{self.split}] Index cache",
                        progress_callback=lambda count: lock.heartbeat(
                            "writing_cache",
                            completed=count,
                            total=len(samples),
                        ),
                    )
                    write_jsonl_atomic(done_path, [{"complete": True, "num_samples": len(samples)}])
                    lock.heartbeat("complete", completed=len(samples), total=len(samples))
                    print(f"[{self.split}] Wrote index cache: {cache_path}", flush=True)
                    if self.lazy_index_records:
                        records = MMapJsonlRecords(cache_path)
                        del samples
                        release_unused_python_heap()
                        return records
                    return samples
                finally:
                    self._active_cache_lock = None
                    lock.release()

            print(
                f"[{self.split}] Another host is building the shared index cache "
                f"({lock.owner_description()}); waiting for {done_path}"
            )
            wait_for_shared_cache(
                cache_path,
                done_path,
                lock.lock_dir,
                timeout_sec=self.index_cache_timeout,
            )
            return self._read_index_cache(cache_path, f"[{self.split}] Index cache")

        print(f"[{self.split}] Rank {rank} waiting for index cache: {cache_path}")
        wait_for_shared_cache(
            cache_path,
            done_path,
            f"{cache_path}.lock",
            timeout_sec=self.index_cache_timeout,
        )
        return self._read_index_cache(cache_path, f"[{self.split}] Index cache")

    def _build_index(self):
        lock = getattr(self, "_active_cache_lock", None)
        bbox_manifests = tuple(getattr(self, "bbox_manifests", ()))
        if bbox_manifests:
            print(
                f"[{self.split}] Loading sample directories from {len(bbox_manifests)} "
                "SAM3 bbox manifest(s); filesystem discovery is disabled.",
                flush=True,
            )
            sample_dirs = sample_dirs_from_bbox_manifests(
                bbox_manifests,
                split=self.split,
                expected_datasets=self.expected_datasets,
                progress_callback=(
                    (lambda count: lock.heartbeat("reading_bbox_manifests", completed=count))
                    if lock is not None
                    else None
                ),
            )
        else:
            sample_dirs = []
            print(f"[{self.split}] Discovering sample directories...", flush=True)
            for root in self.data_dirs:
                split_path = self._split_dir(root)
                if split_path is None:
                    print(f"[{self.split}] Warning: split directory not found under {root}")
                    continue
                root_count = 0
                with os.scandir(split_path) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            sample_dirs.append(entry.path)
                            root_count += 1
                            if root_count % 100000 == 0:
                                print(
                                    f"[{self.split}] Discovered {root_count} sample dirs under {split_path}...",
                                    flush=True,
                                )
                                if lock is not None:
                                    lock.heartbeat("discovering", completed=len(sample_dirs))
                print(
                    f"[{self.split}] Discovered {root_count} sample dirs under {split_path}.",
                    flush=True,
                )
            sample_dirs.sort()
        if lock is not None:
            lock.heartbeat("discovered", completed=len(sample_dirs), total=len(sample_dirs))

        effective_workers = self.index_workers
        allowed_manifest_sha256 = tuple(
            getattr(self, "bbox_manifest_sha256", {}).values()
        )
        if (
            self.index_backend == "process"
            and self.index_process_worker_cap > 0
            and effective_workers > self.index_process_worker_cap
        ):
            effective_workers = self.index_process_worker_cap
            print(
                f"[{self.split}] Capping process index workers from {self.index_workers} to "
                f"{effective_workers}; higher process counts cause shared-filesystem I/O and IPC stalls. "
                "Set --index_process_worker_cap 0 to disable the cap.",
                flush=True,
            )
        print(
            f"[{self.split}] Index scan: {len(sample_dirs)} sample dirs with "
            f"{effective_workers} {self.index_backend} worker(s), chunksize={self.index_chunksize}",
            flush=True,
        )
        samples = []

        def consume_result(result):
            samples.extend(result["samples"])

        done = 0
        last_progress_report = 0
        scan_start = time.monotonic()
        batch_size = max(1, self.index_chunksize)
        sample_dir_batches = [
            sample_dirs[start : start + batch_size]
            for start in range(0, len(sample_dirs), batch_size)
        ]
        try:
            if effective_workers == 1:
                for batch in sample_dir_batches:
                    result = scan_sample_dirs_batch(
                        batch,
                        getattr(self, "bbox_source_policy", "any"),
                        allowed_manifest_sha256,
                    )
                    consume_result(result)
                    done += int(result["sample_dir_count"])
                    if done - last_progress_report >= 10000:
                        elapsed = max(time.monotonic() - scan_start, 1e-6)
                        print(
                            f"[{self.split}] Indexed {done}/{len(sample_dirs)} sample dirs "
                            f"({done / elapsed:.0f} dirs/s)...",
                            flush=True,
                        )
                        lock = getattr(self, "_active_cache_lock", None)
                        if lock is not None:
                            lock.heartbeat("scanning", completed=done, total=len(sample_dirs))
                        last_progress_report = done
            else:
                executor_cls = ProcessPoolExecutor if self.index_backend == "process" else ThreadPoolExecutor
                with executor_cls(max_workers=effective_workers) as executor:
                    batch_iterator = iter(sample_dir_batches)
                    pending = set()
                    max_pending = max(1, effective_workers * 2)

                    def submit_next():
                        try:
                            batch = next(batch_iterator)
                        except StopIteration:
                            return False
                        pending.add(
                            executor.submit(
                                scan_sample_dirs_batch,
                                batch,
                                getattr(self, "bbox_source_policy", "any"),
                                allowed_manifest_sha256,
                            )
                        )
                        return True

                    for _ in range(min(max_pending, len(sample_dir_batches))):
                        submit_next()
                    while pending:
                        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            result = future.result()
                            consume_result(result)
                            done += int(result["sample_dir_count"])
                            submit_next()
                        if done - last_progress_report >= 10000:
                            elapsed = max(time.monotonic() - scan_start, 1e-6)
                            print(
                                f"[{self.split}] Indexed {done}/{len(sample_dirs)} sample dirs "
                                f"({done / elapsed:.0f} dirs/s)...",
                                flush=True,
                            )
                            lock = getattr(self, "_active_cache_lock", None)
                            if lock is not None:
                                lock.heartbeat("scanning", completed=done, total=len(sample_dirs))
                            last_progress_report = done
            if done != len(sample_dirs):
                print(f"[{self.split}] Indexed {done}/{len(sample_dirs)} sample dirs...", flush=True)
            print(
                f"[{self.split}] Index scan complete in {time.monotonic() - scan_start:.1f}s; "
                f"collected {len(samples)} hand samples.",
                flush=True,
            )
        except Exception:
            raise

        lock = getattr(self, "_active_cache_lock", None)
        if lock is not None:
            lock.heartbeat("sorting", completed=len(samples), total=len(samples))
        print(f"[{self.split}] Sorting {len(samples)} compact index records...", flush=True)
        samples.sort(key=lambda item: (item["sample_dir"], item["hand"]))
        return samples

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _sample_error(sample_dir, reason):
        raise RuntimeError(f"Invalid indexed tactile sample at {sample_dir}: {reason}")

    def __getitem__(self, idx):
        sample_record = self.samples[idx]
        sample_dir = sample_record["sample_dir"]

        meta_path = os.path.join(sample_dir, "meta.json")
        
        # 1. Check if files exist (in case extraction is not finished)
        if not os.path.exists(meta_path):
            self._sample_error(sample_dir, "meta.json disappeared after index construction")
            
        # 2. Load pre-computed metadata
        meta = load_json_file(meta_path)

        dataset_name = sample_record.get("dataset", self._infer_dataset_name(meta))
        hand = sample_record.get("hand")
        is_right = int(sample_record.get("is_right", meta.get("is_right", 1)))
        if not bbox_source_allowed(
            meta,
            dataset_name,
            hand=hand,
            policy=self.bbox_source_policy,
        ):
            self._sample_error(
                sample_dir,
                f"bbox source no longer satisfies policy={self.bbox_source_policy}",
            )

        if dataset_name == "TouchAnything":
            hand_meta = meta.get("hands", {}).get(hand, {})
            image_name = meta.get("views", {}).get("chest", "chest.jpg")
            bbox = np.array(hand_meta["bbox_chest"], dtype=np.float32)
            pressure_data = hand_meta.get("gaussian_pressure")
            landmarks_cam = np.zeros((21, 3), dtype=np.float32)
            valid_mask = np.zeros(21, dtype=bool)
        else:
            image_name = meta.get("image", "image.jpg")
            bbox = np.array(meta["bbox"], dtype=np.float32)
            landmarks_cam = np.array(meta.get("keypoints_3d_cam", np.zeros((21, 3))), dtype=np.float32)
            valid_mask = np.array(meta.get("valid_mask", np.zeros(21, dtype=bool)), dtype=bool)
            side = "right" if is_right else "left"
            tactile_key = f"{side}_pressure_continuous_subdiv"
            pressure_data = meta.get("original_hdf5_data", {}).get(tactile_key)
            if pressure_data is None:
                pressure_data = meta.get("gaussian_pressure")

        img_path = os.path.join(sample_dir, image_name)
        if not os.path.exists(img_path):
            self._sample_error(sample_dir, f"image disappeared after index construction: {img_path}")

        # 3. Load image using OpenCV
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            self._sample_error(sample_dir, f"OpenCV could not decode image: {img_path}")
        
        # Extract tactile pressure signal on the subdiv MANO mesh.
        tactile_signal = np.zeros(self.tactile_dim, dtype=np.float32)
        has_tactile = 0.0
        if pressure_data is not None:
            raw_signal = np.array(pressure_data, dtype=np.float32)
            if raw_signal.shape == (self.tactile_dim,) and np.isfinite(raw_signal).all():
                tactile_signal = np.clip(raw_signal, 0.0, 1.0)
                has_tactile = 1.0
            else:
                self._sample_error(
                    sample_dir,
                    "tactile signal must be finite with shape "
                    f"({self.tactile_dim},), got shape={raw_signal.shape}",
                )
        else:
            self._sample_error(sample_dir, "pressure target disappeared after index construction")

        if not self.tactile_only:
            keypoints_3d = np.zeros((21, 4), dtype=np.float32)
            keypoints_3d[valid_mask, :3] = landmarks_cam[valid_mask]
            keypoints_3d[valid_mask, 3] = 1.0
            keypoints_2d = np.zeros((21, 3), dtype=np.float32)
            num_pose = 3 * (self.cfg.MANO.NUM_HAND_JOINTS + 1)
            mano_params = {
                'global_orient': np.zeros(3, dtype=np.float32),
                'hand_pose': np.zeros(num_pose - 3, dtype=np.float32),
                'betas': np.zeros(10, dtype=np.float32)
            }
            has_mano_params = {k: 0.0 for k in mano_params.keys()}
            mano_params_is_axis_angle = {'global_orient': True, 'hand_pose': True, 'betas': False}

        # Calculate bounding box parameters.
        if np.isnan(bbox).any() or len(bbox) < 4:
            self._sample_error(sample_dir, "bbox is missing or non-finite")
            
        center = (bbox[2:4] + bbox[0:2]) / 2.0
        center_x, center_y = center[0], center[1]
        
        scale_pixels = np.max(bbox[2:4] - bbox[0:2])
        if not valid_bbox(bbox) or np.isnan(scale_pixels) or scale_pixels <= 1.0:
            self._sample_error(sample_dir, f"bbox is invalid: {bbox.tolist()}")
            
        bbox_size = self.rescale_factor * scale_pixels
        
        # Add basic augmentation during training
        if self.train and self.augmentation_enabled:
            augm_config = self.cfg.DATASETS.CONFIG
            scale_aug = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.SCALE_FACTOR + 1.0
            tx = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.TRANS_FACTOR * bbox_size
            ty = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.TRANS_FACTOR * bbox_size
            
            bbox_size = bbox_size * scale_aug
            center_x += tx
            center_y += ty
            
        # Crop and resize image using affine transform
        res = self.img_size
        t = np.zeros((2, 3), dtype=np.float32)
        t[0, 0] = float(res) / bbox_size
        t[1, 1] = float(res) / bbox_size
        t[0, 2] = res * (-float(center_x) / bbox_size + 0.5)
        t[1, 2] = res * (-float(center_y) / bbox_size + 0.5)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_patch = cv2.warpAffine(img_rgb, t, (res, res), flags=cv2.INTER_LINEAR)
        
        # Normalize and convert to CHW
        img_patch = img_patch.astype(np.float32) / 255.0
        
        if is_right == 0:
            # Flip left hands to right hands for Hamer
            img_patch = cv2.flip(img_patch, 1)
            if not self.tactile_only:
                keypoints_3d[:, 0] = -keypoints_3d[:, 0]
            # Continuous pressure is already generated on the canonical MANO topology.
            
        # Standard mean/std normalization
        img_patch = (img_patch - self.cfg.MODEL.IMAGE_MEAN) / self.cfg.MODEL.IMAGE_STD
        img_patch = img_patch.transpose(2, 0, 1)
        
        item = {
            'dataset': dataset_name,
            'sample_dir': sample_dir,
            'hand': str(hand),
            'sequence_key': str(sample_record.get('sequence_key', '')),
            'query_alias': str(sample_record.get('query_alias', hand or 'query')),
            'frame_idx': torch.tensor(int(sample_record.get('frame_idx', meta.get('frame_idx', 0) or 0))),
            'bbox_score': torch.tensor(float(sample_record.get('bbox_score', 0.0))).float(),
            'pressure_source_key': str(sample_record.get('pressure_source_key', '')),
            'query_bbox': torch.from_numpy(bbox.copy()).float(),
            'image_width': torch.tensor(int(img_bgr.shape[1])),
            'image_height': torch.tensor(int(img_bgr.shape[0])),
            'img': torch.from_numpy(img_patch).float(),
            'tactile_signal': torch.from_numpy(tactile_signal).float(),
            'has_tactile': torch.tensor(has_tactile).float(),
            'palm_mask': torch.from_numpy(self.palm_mask).float(),
            'right': torch.tensor(float(is_right)).float(),
        }
        if not self.tactile_only:
            img_size_array = np.array([img_bgr.shape[1], img_bgr.shape[0]])
            item.update({
                'keypoints_3d': torch.from_numpy(keypoints_3d).float(),
                'keypoints_2d': torch.from_numpy(keypoints_2d).float(),
                'box_center': torch.tensor([center_x, center_y]).float(),
                'box_size': torch.tensor(bbox_size).float(),
                'img_size': torch.from_numpy(img_size_array).float(),
                'mano_params': {k: torch.from_numpy(v).float() for k, v in mano_params.items()},
                'has_mano_params': {k: torch.tensor(float(v)).float() for k, v in has_mano_params.items()},
                'mano_params_is_axis_angle': {k: torch.tensor(v).bool() for k, v in mano_params_is_axis_angle.items()},
            })
        return item
