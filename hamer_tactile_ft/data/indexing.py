"""Index, manifest, hashing, and shared-cache infrastructure for tactile data."""

import csv
import ctypes
import gc
import hashlib
import json
import mmap
import os
import socket
import time
from array import array
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from functools import lru_cache

import numpy as np

try:
    from ..process_lifecycle import initialize_worker_parent_death_signal
except (ImportError, ValueError):
    from process_lifecycle import initialize_worker_parent_death_signal

try:
    import orjson
except ImportError:
    orjson = None

try:
    from . import hdf5_backend as _hdf5_storage
except ImportError:
    _hdf5_storage = None


ft_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))

CANONICAL_SPLITS = ("train", "val", "test")
INDEX_CACHE_VERSION = 7
HDF5_STORAGE_SCHEMA_VERSION = str(
    getattr(
        _hdf5_storage,
        "HDF5_SCHEMA_VERSION",
        getattr(_hdf5_storage, "SCHEMA_VERSION", "tactile_sequence_hdf5_v1"),
    )
)

__all__ = [
    "BBOX_SOURCE_POLICIES",
    "CANONICAL_SPLITS",
    "COMPACT_INDEX_FIELDS",
    "DatasetIndexingMixin",
    "EXPECTED_TACTILE_DIM",
    "INDEX_CACHE_VERSION",
    "INDEX_PALM_MASK",
    "INDEX_PALM_VERTEX_MASK",
    "MMapJsonlRecords",
    "SAM3_BBOX_SOURCE_SCHEMA",
    "SUBDIV_OBJ_PATH",
    "SUBDIV_PALM_FACES_PATH",
    "SharedCacheBuildLock",
    "bbox_source_allowed",
    "bbox_source_for_query",
    "canonical_dataset_filter",
    "canonical_dataset_name",
    "count_obj_vertices",
    "ddp_global_rank",
    "filter_sample_groups_by_bbox_source_batch",
    "has_pressure",
    "legacy_valid_bbox",
    "load_json_file",
    "load_subdiv_palm_mask",
    "persistent_sha256_file",
    "pressure_array_or_none",
    "pressure_for_query",
    "query_sequence_key",
    "read_compact_index_from_audit_csv",
    "read_compact_index_from_integrity_sidecar",
    "read_jsonl",
    "release_unused_python_heap",
    "sample_dirs_from_bbox_manifests",
    "sample_provenance",
    "scan_sample_dir",
    "scan_sample_dir_integrity",
    "scan_sample_dirs_batch",
    "scan_sample_dirs_integrity_batch",
    "sha256_file",
    "valid_bbox",
    "valid_pressure",
    "wait_for_file",
    "wait_for_shared_cache",
    "write_json_atomic",
    "write_jsonl_atomic",
]


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
        "acedata": "AceData",
        "ace_data": "AceData",
        "ace": "AceData",
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


@lru_cache(maxsize=64)
def _sha256_file_for_stat(path, size, mtime_ns, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path, chunk_size=1024 * 1024):
    normalized = os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
    stat = os.stat(normalized)
    return _sha256_file_for_stat(
        normalized,
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(chunk_size),
    )


def persistent_sha256_file(path, cache_dir=None, chunk_size=1024 * 1024):
    """Reuse a verified file digest without rereading a large manifest each run."""
    normalized = os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
    stat = os.stat(normalized)
    fingerprint = {
        "path": normalized,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }
    if not cache_dir:
        return sha256_file(normalized, chunk_size=chunk_size)

    checksum_dir = os.path.join(
        os.path.realpath(os.path.abspath(os.path.expanduser(str(cache_dir)))),
        "checksums",
    )
    cache_name = hashlib.sha1(normalized.encode("utf-8")).hexdigest() + ".json"
    cache_path = os.path.join(checksum_dir, cache_name)
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if (
            all(cached.get(key) == value for key, value in fingerprint.items())
            and isinstance(cached.get("sha256"), str)
            and len(cached["sha256"]) == 64
        ):
            return cached["sha256"]
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        pass

    digest = sha256_file(normalized, chunk_size=chunk_size)
    os.makedirs(checksum_dir, exist_ok=True)
    payload = dict(fingerprint)
    payload["sha256"] = digest
    tmp_path = (
        f"{cache_path}.tmp.{socket.gethostname()}.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, cache_path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
    return digest


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

    OFFSETS_VERSION = 1

    def __init__(self, path):
        self.path = os.path.realpath(os.path.abspath(path))
        self._mapping = None
        self._offsets = None
        self._open()

    def _offset_paths(self):
        return f"{self.path}.offsets.u64", f"{self.path}.offsets.json"

    def _source_fingerprint(self):
        stat = os.stat(self.path)
        return {
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "source_ctime_ns": int(stat.st_ctime_ns),
            "offsets_version": self.OFFSETS_VERSION,
        }

    def _read_persistent_offsets(self):
        offsets_path, metadata_path = self._offset_paths()
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            fingerprint = self._source_fingerprint()
            if any(metadata.get(key) != value for key, value in fingerprint.items()):
                return None
            offsets_size = os.path.getsize(offsets_path)
            if offsets_size < 8 or offsets_size % 8:
                return None
            offsets = array("Q")
            with open(offsets_path, "rb") as handle:
                offsets.fromfile(handle, offsets_size // 8)
            if (
                len(offsets) != int(metadata.get("offset_count", -1))
                or int(offsets[-1]) != fingerprint["source_size"]
            ):
                return None
            return offsets
        except (FileNotFoundError, OSError, ValueError, TypeError, EOFError):
            return None

    def _scan_offsets(self, mapping, file_size):
        offsets = array("Q", [0])
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
        return offsets

    def _write_persistent_offsets(self, offsets):
        offsets_path, metadata_path = self._offset_paths()
        os.makedirs(os.path.dirname(offsets_path), exist_ok=True)
        unique_suffix = (
            f"{socket.gethostname()}.{os.getpid()}.{time.time_ns()}"
        )
        offsets_tmp = f"{offsets_path}.tmp.{unique_suffix}"
        metadata_tmp = f"{metadata_path}.tmp.{unique_suffix}"
        metadata = self._source_fingerprint()
        metadata["offset_count"] = len(offsets)
        try:
            with open(offsets_tmp, "wb") as handle:
                offsets.tofile(handle)
            with open(metadata_tmp, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, sort_keys=True)
                handle.write("\n")
            os.replace(offsets_tmp, offsets_path)
            os.replace(metadata_tmp, metadata_path)
        finally:
            for temporary in (offsets_tmp, metadata_tmp):
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    def _load_or_build_offsets(self, mapping, file_size):
        offsets = self._read_persistent_offsets()
        if offsets is not None:
            return offsets

        offsets_path, metadata_path = self._offset_paths()
        lock = SharedCacheBuildLock(
            f"{offsets_path}.lock",
            timeout_sec=3600,
        )
        if lock.try_acquire():
            try:
                offsets = self._read_persistent_offsets()
                if offsets is None:
                    if ddp_global_rank() == 0:
                        print(
                            f"Building persistent mmap row offsets once: {offsets_path}",
                            flush=True,
                        )
                    offsets = self._scan_offsets(mapping, file_size)
                    self._write_persistent_offsets(offsets)
                return offsets
            finally:
                lock.release()

        wait_for_file(
            metadata_path,
            timeout_sec=3600,
            progress_label=f"Waiting for mmap row offsets {metadata_path}",
        )
        offsets = self._read_persistent_offsets()
        if offsets is None:
            raise RuntimeError(
                f"Persistent mmap row offsets are incomplete or stale: {offsets_path}"
            )
        return offsets

    def _open(self):
        file_size = os.path.getsize(self.path)
        if file_size == 0:
            self._mapping = None
            self._offsets = array("Q", [0])
            return
        with open(self.path, "rb") as handle:
            mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        offsets = self._load_or_build_offsets(mapping, file_size)
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


class DatasetIndexingMixin:
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

    def _normalize_query_manifest_specs(self, query_manifests):
        if query_manifests is None or (
            isinstance(query_manifests, str) and not query_manifests.strip()
        ):
            return []

        raw_specs = []
        if isinstance(query_manifests, dict):
            for key, value in query_manifests.items():
                values = value if isinstance(value, (list, tuple)) else [value]
                for entry in values:
                    if isinstance(entry, dict):
                        raw_specs.append({
                            "path": entry.get("path") or entry.get("manifest"),
                            "root": entry.get("root") or entry.get("data_root"),
                            "dataset": entry.get("dataset") or key,
                        })
                    else:
                        raw_specs.append({"path": entry, "dataset": key})
        else:
            values = (
                query_manifests.split(",")
                if isinstance(query_manifests, str)
                else (
                    [query_manifests]
                    if isinstance(query_manifests, os.PathLike)
                    else list(query_manifests)
                )
            )
            for entry in values:
                if not isinstance(entry, dict) and not str(entry).strip():
                    continue
                if isinstance(entry, dict):
                    raw_specs.append({
                        "path": entry.get("path") or entry.get("manifest"),
                        "root": entry.get("root") or entry.get("data_root"),
                        "dataset": entry.get("dataset"),
                    })
                else:
                    raw_specs.append({"path": entry})

        specs = []
        for index, raw_spec in enumerate(raw_specs):
            raw_path = raw_spec.get("path")
            if not raw_path or not str(raw_path).strip():
                raise ValueError(f"query_manifests entry {index} has no manifest path")
            path = os.path.realpath(
                os.path.abspath(os.path.expanduser(str(raw_path).strip()))
            )
            if not os.path.isfile(path):
                raise FileNotFoundError(f"HDF5 query manifest does not exist: {path}")
            root = raw_spec.get("root")
            if root:
                root = os.path.realpath(
                    os.path.abspath(os.path.expanduser(str(root).strip()))
                )
            elif len(raw_specs) == len(self.data_dirs):
                root = self.data_dirs[index]
            dataset_hint = raw_spec.get("dataset")
            specs.append({
                "path": path,
                "root": root,
                "dataset": (
                    canonical_dataset_name(dataset_hint)
                    if dataset_hint and str(dataset_hint).strip()
                    else None
                ),
            })
        return specs

    def _discover_query_manifest_specs(self):
        """Find one split manifest under every processed dataset root."""
        discovered = []
        missing_roots = []
        expected_by_root = (
            [self.expected_datasets[0]]
            if len(self.expected_datasets) == 1 and len(self.data_dirs) == 1
            else [None] * len(self.data_dirs)
        )
        for root_index, root in enumerate(self.data_dirs):
            manifest_dir = os.path.join(root, "manifests")
            dataset_hint = expected_by_root[root_index]
            dataset_slug = (
                str(dataset_hint).lower().replace("_", "")
                if dataset_hint
                else ""
            )
            candidates = []
            if dataset_slug:
                candidates.extend((
                    os.path.join(
                        manifest_dir,
                        f"{dataset_slug}_{self.split}.queries.jsonl",
                    ),
                    os.path.join(
                        manifest_dir,
                        f"{dataset_slug}_{self.split}.jsonl",
                    ),
                ))
            candidates.extend((
                os.path.join(manifest_dir, f"{self.split}.queries.jsonl"),
                os.path.join(manifest_dir, f"{self.split}.jsonl"),
            ))
            manifest_path = next(
                (path for path in candidates if os.path.isfile(path)),
                None,
            )
            if manifest_path is None and os.path.isdir(manifest_dir):
                suffix = f"_{self.split}.queries.jsonl"
                prefixed = sorted(
                    os.path.join(manifest_dir, name)
                    for name in os.listdir(manifest_dir)
                    if name.endswith(suffix)
                    and os.path.isfile(os.path.join(manifest_dir, name))
                )
                if len(prefixed) > 1:
                    raise RuntimeError(
                        f"Ambiguous sequence-HDF5 manifests for root={root}, "
                        f"split={self.split}: {prefixed}"
                    )
                if prefixed:
                    manifest_path = prefixed[0]
            if manifest_path is None:
                missing_roots.append(root)
                continue
            discovered.append({
                "path": os.path.realpath(os.path.abspath(manifest_path)),
                "root": os.path.realpath(os.path.abspath(root)),
                "dataset": dataset_hint,
            })

        if discovered and missing_roots:
            raise RuntimeError(
                "Only part of the requested processed roots have sequence-HDF5 "
                f"manifests for split={self.split!r}. Migrated roots="
                f"{[spec['root'] for spec in discovered]}, missing={missing_roots}. "
                "Finish conversion, pass --query_manifests explicitly, or select "
                "--data_backend legacy_dirs."
            )
        if discovered:
            print(
                f"[{self.split}] Auto-discovered {len(discovered)} sequence-HDF5 "
                "query manifest(s).",
                flush=True,
            )
        return discovered

    def _resolve_hdf5_path(self, record, manifest_path, root_hint):
        raw_path = record.get("h5_path")
        relative_path = record.get("h5_relpath")
        if raw_path is None and relative_path is None:
            raise ValueError("record must contain h5_path or h5_relpath")

        path_value = os.path.expanduser(str(raw_path or relative_path))
        record_root = record.get("data_root") or record.get("hdf5_root")
        cache_key = (
            path_value,
            str(record_root or ""),
            str(root_hint or ""),
            str(manifest_path or ""),
        )
        cached = self._resolved_hdf5_paths.get(cache_key)
        if cached is not None:
            return cached
        if os.path.isabs(path_value):
            candidates = [path_value]
        else:
            candidates = []
            if record_root:
                candidates.append(os.path.join(os.path.expanduser(str(record_root)), path_value))
            if root_hint:
                candidates.append(os.path.join(root_hint, path_value))
            if not record_root and not root_hint:
                candidates.extend(os.path.join(root, path_value) for root in self.data_dirs)
            if manifest_path and not record_root:
                manifest_dir = os.path.dirname(manifest_path)
                candidates.extend((
                    os.path.join(manifest_dir, path_value),
                    os.path.join(os.path.dirname(manifest_dir), path_value),
                ))

        unique_candidates = []
        seen = set()
        for candidate in candidates:
            normalized = os.path.realpath(os.path.abspath(candidate))
            if normalized not in seen:
                seen.add(normalized)
                unique_candidates.append(normalized)
        existing = [path for path in unique_candidates if os.path.isfile(path)]
        if not existing:
            preview = ", ".join(unique_candidates[:5]) or path_value
            raise FileNotFoundError(
                f"HDF5 container does not exist for {path_value!r}; checked: {preview}"
            )
        if len(existing) > 1:
            raise RuntimeError(
                f"Ambiguous h5_relpath={path_value!r}; it resolves under multiple roots: "
                + ", ".join(existing)
            )
        self._resolved_hdf5_paths[cache_key] = existing[0]
        return existing[0]

    def _manifest_bbox_source_allowed(self, record):
        if self.bbox_source_policy == "any":
            return True
        source = record.get("bbox_source")
        if not isinstance(source, dict):
            source = {
                "schema": record.get("bbox_source_schema"),
                "source_manifest_sha256": record.get("bbox_manifest_sha256"),
            }
        if not isinstance(source, dict) or source.get("schema") != SAM3_BBOX_SOURCE_SCHEMA:
            return False
        dataset_name = canonical_dataset_name(record.get("dataset"))
        if dataset_name == "EgoTactile":
            return bool(
                source.get("association_policy")
                == "egotactile_task_hand_single_track"
                and source.get("source_bbox_jsonl_sha256")
                and source.get("source_manifest_sha256")
            )
        if dataset_name == "AceData":
            return bool(
                source.get("association_policy") == "initial_screen_order"
                and source.get("source_bbox_jsonl_sha256")
                and source.get("source_manifest_sha256")
                and source.get("source_materialized_bbox_sha256")
            )
        allowed_hashes = set(getattr(self, "bbox_manifest_sha256", {}).values())
        return not allowed_hashes or source.get("source_manifest_sha256") in allowed_hashes

    @staticmethod
    def _normalized_query_alias(dataset_name, value):
        if dataset_name == "OpenTouch":
            return "*"
        alias = str(value or "").strip().lower()
        if alias in ("l", "left"):
            return "left"
        if alias in ("r", "right"):
            return "right"
        return alias

    def _portable_sample_relpath(self, value):
        if value is None or not str(value).strip():
            return ""
        raw = os.path.expanduser(str(value).strip())
        normalized = os.path.normpath(raw).replace("\\", "/")
        if os.path.isabs(raw):
            absolute = os.path.abspath(raw)
            for root in self.data_dirs:
                try:
                    relative = os.path.relpath(absolute, os.path.abspath(root))
                except ValueError:
                    continue
                if relative != ".." and not relative.startswith(f"..{os.sep}"):
                    return os.path.normpath(relative).replace("\\", "/")
            split_marker = f"/{self.split}/"
            normalized_absolute = absolute.replace("\\", "/")
            if split_marker in normalized_absolute:
                return self.split + "/" + normalized_absolute.rsplit(split_marker, 1)[1]
        return normalized[2:] if normalized.startswith("./") else normalized

    @staticmethod
    def _normalized_sequence_key(value):
        return str(value or "").strip().replace("\\", "/").strip("/")

    def _bbox_overlay_keys(
        self,
        *,
        dataset_name,
        split,
        sequence_key,
        frame_idx,
        query_alias,
        sample_path,
    ):
        alias = self._normalized_query_alias(dataset_name, query_alias)
        keys = []
        relative_path = self._portable_sample_relpath(sample_path)
        if relative_path:
            keys.append(("sample", dataset_name, relative_path, alias))
        normalized_sequence = self._normalized_sequence_key(sequence_key)
        if normalized_sequence and frame_idx is not None:
            keys.append((
                "frame",
                dataset_name,
                str(split or self.split),
                normalized_sequence,
                int(frame_idx),
                alias,
            ))
        return keys

    def _load_bbox_manifest_overlay_index(self):
        if self._bbox_manifest_overlay_index is not None:
            return self._bbox_manifest_overlay_index

        overlay_index = {}
        row_count = 0
        selected_count = 0
        required_keys = self._bbox_manifest_overlay_required_keys
        expected = set(self.expected_datasets)
        for manifest_path in self.bbox_manifests:
            manifest_sha256 = self.bbox_manifest_sha256[manifest_path]
            with open(manifest_path, "rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    row_count += 1
                    try:
                        row = (
                            orjson.loads(raw_line)
                            if orjson is not None
                            else json.loads(raw_line)
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"Invalid SAM3 bbox manifest JSON at "
                            f"{manifest_path}:{line_number}: {exc}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise TypeError(
                            f"{manifest_path}:{line_number}: bbox manifest row "
                            "must be a JSON object"
                        )
                    dataset_value = row.get("dataset")
                    if not dataset_value:
                        raise ValueError(
                            f"{manifest_path}:{line_number}: bbox manifest row "
                            "is missing dataset"
                        )
                    dataset_name = canonical_dataset_name(dataset_value)
                    if expected and dataset_name not in expected:
                        continue
                    row_split = str(row.get("split") or self.split)
                    if row_split != self.split:
                        continue
                    bbox = row.get("bbox_xyxy", row.get("bbox"))
                    if not valid_bbox(bbox):
                        raise ValueError(
                            f"{manifest_path}:{line_number}: invalid SAM3 bbox "
                            f"{bbox!r}"
                        )
                    query_alias = row.get(
                        "target_hand",
                        row.get("query_alias", row.get("hand")),
                    )
                    if dataset_name == "TouchAnything" and self._normalized_query_alias(
                        dataset_name, query_alias
                    ) not in ("left", "right"):
                        raise ValueError(
                            f"{manifest_path}:{line_number}: TouchAnything SAM3 "
                            "row is missing target_hand=left|right"
                        )
                    association_evidence = row.get("association_evidence")
                    if not isinstance(association_evidence, dict):
                        association_evidence = {}
                    source = {
                        "schema": SAM3_BBOX_SOURCE_SCHEMA,
                        "association_policy": (
                            "single_gloved_query"
                            if dataset_name == "OpenTouch"
                            else association_evidence.get(
                                "assignment_policy", "legacy_anchor"
                            )
                        ),
                        "association_confidence": row.get("association_confidence"),
                        "raw_track_id": row.get("raw_track_id"),
                        "association_id": row.get("association_id"),
                        "source_manifest": manifest_path,
                        "source_manifest_sha256": manifest_sha256,
                    }
                    tracking_source = row.get("bbox_source")
                    if not isinstance(tracking_source, dict):
                        source.update({
                            "tracking_bbox_source": tracking_source or "sam3_native",
                            "flow_confidence": row.get("flow_confidence"),
                            "flow_bbox_iou": row.get("flow_bbox_iou"),
                            "flow_anchor_frames": list(
                                row.get("flow_anchor_frames") or ()
                            ),
                        })
                    payload = {
                        "bbox_xyxy": [float(value) for value in bbox],
                        "bbox_score": float(
                            row.get("bbox_score", row.get("prompt_score", 0.0)) or 0.0
                        ),
                        "bbox_source": source,
                        "_bbox_overlay_manifest": manifest_path,
                        "_bbox_overlay_line": line_number,
                    }
                    keys = self._bbox_overlay_keys(
                        dataset_name=dataset_name,
                        split=row_split,
                        sequence_key=row.get("sequence_key"),
                        frame_idx=row.get("frame_idx"),
                        query_alias=query_alias,
                        sample_path=row.get(
                            "source_sample_relpath",
                            row.get("sample_dir"),
                        ),
                    )
                    if not keys:
                        raise ValueError(
                            f"{manifest_path}:{line_number}: SAM3 bbox row has "
                            "neither a portable sample path nor sequence/frame identity"
                        )
                    row_selected = False
                    for key in keys:
                        if required_keys is not None and key not in required_keys:
                            continue
                        previous = overlay_index.get(key)
                        if previous is not None and (
                            previous["bbox_xyxy"] != payload["bbox_xyxy"]
                            or previous["bbox_source"]["source_manifest_sha256"]
                            != manifest_sha256
                        ):
                            raise RuntimeError(
                                "Conflicting SAM3 bbox rows resolve to the same HDF5 "
                                f"query key {key!r}: "
                                f"{previous['_bbox_overlay_manifest']}:"
                                f"{previous['_bbox_overlay_line']} and "
                                f"{manifest_path}:{line_number}"
                            )
                        overlay_index[key] = payload
                        row_selected = True
                    selected_count += int(row_selected)
        self._bbox_manifest_overlay_index = overlay_index
        print(
            f"[{self.split}] Loaded {selected_count}/{row_count} current SAM3 bbox "
            f"rows as {len(overlay_index)} HDF5 overlay key(s).",
            flush=True,
        )
        return overlay_index

    def _collect_required_hdf5_bbox_overlay_keys(self):
        """Collect only stale query keys before scanning large bbox manifests."""

        if not self.bbox_manifests:
            return None
        required = set()
        expected = set(self.expected_datasets)
        stale_records = 0
        for spec in self.query_manifest_specs:
            path = spec["path"]
            with open(path, "rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        record = (
                            orjson.loads(raw_line)
                            if orjson is not None
                            else json.loads(raw_line)
                        )
                        if not isinstance(record, dict):
                            raise TypeError("query manifest row must be a JSON object")
                        record_split = record.get("split")
                        if record_split is not None and str(record_split) != self.split:
                            continue
                        dataset_value = record.get("dataset") or spec.get("dataset")
                        if not dataset_value:
                            raise ValueError("query manifest row is missing dataset")
                        dataset_name = canonical_dataset_name(dataset_value)
                        if expected and dataset_name not in expected:
                            continue
                        if self._manifest_bbox_source_allowed(record):
                            continue
                        stale_records += 1
                        required.update(self._bbox_overlay_keys(
                            dataset_name=dataset_name,
                            split=record.get(
                                "source_split", record.get("split", self.split)
                            ),
                            sequence_key=record.get("sequence_key"),
                            frame_idx=record.get("frame_idx"),
                            query_alias=record.get(
                                "target_hand",
                                record.get("query_alias", record.get("hand")),
                            ),
                            sample_path=record.get("source_sample_relpath"),
                        ))
                    except Exception as exc:
                        raise RuntimeError(
                            f"Invalid HDF5 query manifest row while preparing the "
                            f"SAM3 overlay at {path}:{line_number}: {exc}"
                        ) from exc
        print(
            f"[{self.split}] SAM3 overlay prefilter: {stale_records} stale "
            f"query row(s), {len(required)} required key(s).",
            flush=True,
        )
        return required

    def _apply_hdf5_bbox_manifest_overlay(self, record, dataset_name):
        if self._manifest_bbox_source_allowed(record):
            return record
        if not self.bbox_manifests:
            return None
        query_alias = record.get(
            "query_alias",
            record.get("hand"),
        )
        keys = self._bbox_overlay_keys(
            dataset_name=dataset_name,
            split=record.get("source_split", record.get("split", self.split)),
            sequence_key=record.get("sequence_key"),
            frame_idx=record.get("frame_idx"),
            query_alias=query_alias,
            sample_path=record.get("source_sample_relpath"),
        )
        overlay_index = self._load_bbox_manifest_overlay_index()
        matches = [overlay_index[key] for key in keys if key in overlay_index]
        if not matches:
            return None
        overlay = matches[0]
        if any(
            match["bbox_xyxy"] != overlay["bbox_xyxy"]
            or match["bbox_source"]["source_manifest_sha256"]
            != overlay["bbox_source"]["source_manifest_sha256"]
            for match in matches[1:]
        ):
            raise RuntimeError(
                "SAM3 bbox path and sequence/frame keys resolve to conflicting rows "
                f"for HDF5 query {keys!r}"
            )
        updated = dict(record)
        updated.update(overlay)
        return updated

    def _normalize_hdf5_manifest_record(
        self,
        record,
        manifest_path,
        root_hint,
        dataset_hint,
        line_number,
        allow_bbox_policy_skip=False,
    ):
        location = (
            f"{manifest_path}:{line_number}" if manifest_path else f"sample_records[{line_number - 1}]"
        )
        if not isinstance(record, dict):
            raise TypeError(f"{location}: query manifest row must be a JSON object")

        dataset_value = record.get("dataset") or dataset_hint
        if not dataset_value:
            raise ValueError(f"{location}: query manifest row is missing dataset")
        dataset_name = canonical_dataset_name(dataset_value)
        if dataset_hint and dataset_name != canonical_dataset_name(dataset_hint):
            raise ValueError(
                f"{location}: dataset={dataset_name!r} conflicts with manifest "
                f"dataset hint={dataset_hint!r}"
            )

        try:
            frame_row = int(record["frame_row"])
            query_row = int(record["query_row"])
        except KeyError as exc:
            raise ValueError(f"{location}: missing required field {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location}: frame_row/query_row must be integers") from exc
        if frame_row < 0 or query_row < 0:
            raise ValueError(f"{location}: frame_row/query_row must be non-negative")

        hand = record.get("hand", record.get("query_alias"))
        is_right = record.get("is_right")
        if is_right is None:
            normalized_hand = str(hand or "").strip().lower()
            if normalized_hand in ("right", "r"):
                is_right = 1
            elif normalized_hand in ("left", "l"):
                is_right = 0
            else:
                raise ValueError(
                    f"{location}: row must provide is_right when hand is not left/right"
                )
        try:
            is_right = int(is_right)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location}: is_right must be 0 or 1") from exc
        if is_right not in (0, 1):
            raise ValueError(f"{location}: is_right must be 0 or 1, got {is_right}")
        hand = str(hand or ("right" if is_right else "left"))

        overlaid_record = self._apply_hdf5_bbox_manifest_overlay(record, dataset_name)
        if overlaid_record is None:
            if allow_bbox_policy_skip:
                return None
            raise ValueError(
                f"{location}: bbox provenance does not satisfy "
                f"bbox_source_policy={self.bbox_source_policy}, and no matching "
                "row exists in the current SAM3 bbox manifest(s)"
            )
        record = overlaid_record
        bbox = record.get("bbox_xyxy", record.get("bbox"))
        if bbox is not None and not valid_bbox(bbox):
            raise ValueError(f"{location}: bbox_xyxy is invalid: {bbox!r}")
        if not self._manifest_bbox_source_allowed(record):
            raise ValueError(
                f"{location}: bbox provenance does not satisfy "
                f"bbox_source_policy={self.bbox_source_policy}"
            )

        h5_path = self._resolve_hdf5_path(record, manifest_path, root_hint)
        frame_idx = int(record.get("frame_idx", frame_row) or 0)
        sequence_key = str(record.get("sequence_key") or os.path.splitext(
            os.path.basename(h5_path)
        )[0])
        query_alias = str(record.get("query_alias") or hand or f"query_{query_row}")
        sample_uid = str(
            record.get("sample_uid")
            or f"{dataset_name}/{self.split}/{sequence_key}/{frame_idx}/{query_alias}"
        )
        sample_ref = str(
            record.get("sample_ref")
            or f"{h5_path}#frame={frame_row}&query={query_row}"
        )
        explicit_schema_version = (
            record.get("hdf5_schema_version") or record.get("schema_version")
        )
        schema_version = str(explicit_schema_version or HDF5_STORAGE_SCHEMA_VERSION)
        self.hdf5_schema_versions.add(schema_version)

        normalized = dict(record)
        normalized.update({
            "dataset": dataset_name,
            "hand": hand,
            "is_right": is_right,
            "h5_path": h5_path,
            "frame_row": frame_row,
            "query_row": query_row,
            "frame_idx": frame_idx,
            "sequence_key": sequence_key,
            "query_alias": query_alias,
            "sample_uid": sample_uid,
            "sample_ref": sample_ref,
            "sample_dir": sample_ref,
            "hdf5_schema_version": schema_version,
            "_expected_hdf5_schema_version": (
                str(explicit_schema_version) if explicit_schema_version else ""
            ),
        })
        if bbox is not None:
            normalized["bbox_xyxy"] = [float(value) for value in bbox]
        return normalized

    def _hdf5_manifest_cache_path(self):
        if not self.hdf5_manifest_cache_dir:
            return None
        key_data = {
            "backend": "sequence_hdf5",
            "split": self.split,
            "data_dirs": [os.path.realpath(os.path.abspath(path)) for path in self.data_dirs],
            "query_manifest_sha256": self.query_manifest_sha256,
            "dataset_filter": list(self.expected_datasets),
            "bbox_source_policy": self.bbox_source_policy,
            "bbox_manifest_sha256": self.bbox_manifest_sha256,
            # Version 4 recognizes train-only AceData SAM3 provenance in
            # addition to EgoTactile and the reviewed OT/TA overlays.
            "bbox_hdf5_overlay_schema_version": 4,
            "storage_schema_version": HDF5_STORAGE_SCHEMA_VERSION,
        }
        digest = hashlib.sha1(
            json.dumps(key_data, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(
            self.hdf5_manifest_cache_dir,
            f"{self.split}_hdf5_{digest}.jsonl",
        )

    def _parse_hdf5_query_manifests(self):
        samples = []
        bbox_policy_skipped = 0
        bbox_overlay_applied = 0
        expected = set(self.expected_datasets)
        self._bbox_manifest_overlay_required_keys = (
            self._collect_required_hdf5_bbox_overlay_keys()
        )
        try:
            for spec in self.query_manifest_specs:
                path = spec["path"]
                print(f"[{self.split}] Loading HDF5 query manifest: {path}", flush=True)
                with open(path, "rb") as handle:
                    for line_number, raw_line in enumerate(handle, start=1):
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            raw_record = (
                                orjson.loads(raw_line)
                                if orjson is not None
                                else json.loads(raw_line)
                            )
                            record_split = raw_record.get("split") if isinstance(raw_record, dict) else None
                            if record_split is not None and str(record_split) != self.split:
                                continue
                            if not isinstance(raw_record, dict):
                                raise TypeError("query manifest row must be a JSON object")
                            dataset_value = raw_record.get("dataset") or spec.get("dataset")
                            if not dataset_value:
                                raise ValueError("query manifest row is missing dataset")
                            if expected and canonical_dataset_name(dataset_value) not in expected:
                                continue
                            normalized = self._normalize_hdf5_manifest_record(
                                raw_record,
                                manifest_path=path,
                                root_hint=spec.get("root"),
                                dataset_hint=spec.get("dataset"),
                                line_number=line_number,
                                allow_bbox_policy_skip=True,
                            )
                            if normalized is None:
                                bbox_policy_skipped += 1
                                continue
                            if normalized.get("_bbox_overlay_manifest"):
                                bbox_overlay_applied += 1
                            samples.append(normalized)
                        except Exception as exc:
                            raise RuntimeError(
                                f"Invalid HDF5 query manifest row at {path}:{line_number}: {exc}"
                            ) from exc
        finally:
            self._bbox_manifest_overlay_index = None
            self._bbox_manifest_overlay_required_keys = None
            release_unused_python_heap()
        if bbox_overlay_applied or bbox_policy_skipped:
            print(
                f"[{self.split}] HDF5 SAM3 bbox adaptation: "
                f"overlaid={bbox_overlay_applied}, "
                f"filtered_not_in_current_manifest={bbox_policy_skipped}.",
                flush=True,
            )
        if not samples:
            raise RuntimeError(
                f"[{self.split}] HDF5 query manifests produced no samples after "
                "split/dataset/SAM3 bbox filtering "
                f"(expected_datasets={list(self.expected_datasets)}, "
                f"bbox_policy_skipped={bbox_policy_skipped})"
            )
        if expected:
            observed = {sample["dataset"] for sample in samples}
            if observed != expected:
                raise RuntimeError(
                    f"[{self.split}] HDF5 manifest dataset contract failed: "
                    f"expected={sorted(expected)}, observed={sorted(observed)}"
                )
        return samples

    def _load_hdf5_query_manifests(self):
        if not self.lazy_index_records:
            return self._parse_hdf5_query_manifests()

        cache_path = self._hdf5_manifest_cache_path()
        if cache_path is None:
            print(
                f"[{self.split}] --lazy_index_records requested without "
                "--hdf5_manifest_cache_dir; retaining normalized HDF5 manifest "
                "rows in memory.",
                flush=True,
            )
            return self._parse_hdf5_query_manifests()

        done_path = f"{cache_path}.done"
        if os.path.isfile(cache_path) and os.path.isfile(done_path):
            if ddp_global_rank() == 0:
                print(
                    f"[{self.split}] Reusing persistent normalized HDF5 manifest "
                    f"cache: {cache_path}",
                    flush=True,
                )
            self.hdf5_schema_versions.add(HDF5_STORAGE_SCHEMA_VERSION)
            return MMapJsonlRecords(cache_path)

        lock = SharedCacheBuildLock(
            f"{cache_path}.lock",
            timeout_sec=self.index_cache_timeout,
        )
        if lock.try_acquire():
            try:
                if os.path.isfile(cache_path) and os.path.isfile(done_path):
                    if ddp_global_rank() == 0:
                        print(
                            f"[{self.split}] Reusing persistent normalized HDF5 "
                            f"manifest cache: {cache_path}",
                            flush=True,
                        )
                    self.hdf5_schema_versions.add(HDF5_STORAGE_SCHEMA_VERSION)
                    return MMapJsonlRecords(cache_path)
                lock.heartbeat("normalizing_hdf5_manifest")
                samples = self._parse_hdf5_query_manifests()
                lock.heartbeat("writing_hdf5_manifest_cache", 0, len(samples))
                write_jsonl_atomic(
                    cache_path,
                    samples,
                    progress_label=f"[{self.split}] HDF5 manifest cache",
                    progress_callback=lambda count: lock.heartbeat(
                        "writing_hdf5_manifest_cache",
                        count,
                        len(samples),
                    ),
                )
                write_json_atomic(
                    done_path,
                    {
                        "complete": True,
                        "num_samples": len(samples),
                        "query_manifest_sha256": self.query_manifest_sha256,
                        "bbox_manifest_sha256": self.bbox_manifest_sha256,
                        "bbox_hdf5_overlay_schema_version": 4,
                    },
                )
                records = MMapJsonlRecords(cache_path)
                del samples
                release_unused_python_heap()
                return records
            finally:
                lock.release()

        print(
            f"[{self.split}] Another process is normalizing the HDF5 query "
            f"manifest ({lock.owner_description()}); waiting for {done_path}.",
            flush=True,
        )
        wait_for_shared_cache(
            cache_path,
            done_path,
            lock.lock_dir,
            timeout_sec=self.index_cache_timeout,
        )
        self.hdf5_schema_versions.add(HDF5_STORAGE_SCHEMA_VERSION)
        return MMapJsonlRecords(cache_path)

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
        if self.data_backend == "sequence_hdf5":
            return None
        if not self.index_cache_dir:
            return None
        key_data = {
            "data_backend": self.data_backend,
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
        if self.data_backend == "sequence_hdf5":
            manifest_digest = hashlib.sha1(
                json.dumps(self.query_manifest_sha256, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            normalized_cache = self._hdf5_manifest_cache_path()
            return {
                "data_backend": self.data_backend,
                "storage_schema_version": sorted(self.hdf5_schema_versions),
                "index_schema_version": INDEX_CACHE_VERSION,
                "index_cache_key": f"hdf5_manifest_{manifest_digest}",
                "hdf5_manifest_cache_key": (
                    os.path.splitext(os.path.basename(normalized_cache))[0]
                    if normalized_cache
                    else "uncached"
                ),
                "indexed_sample_count": len(self.samples),
                "query_manifest_sha256": dict(self.query_manifest_sha256),
                "dataset_filter": list(self.expected_datasets),
                "bbox_source_policy": self.bbox_source_policy,
                "bbox_manifest_sha256": dict(getattr(self, "bbox_manifest_sha256", {})),
                "legacy_index_cache_used": False,
                "hdf5_manifest_cache_dir": str(self.hdf5_manifest_cache_dir or ""),
                "depth_sidecar_contract": dict(self.depth_sidecar_contract),
            }
        cache_path = self._cache_path()
        cache_key = os.path.splitext(os.path.basename(cache_path))[0] if cache_path else "uncached"
        return {
            "data_backend": self.data_backend,
            "storage_schema_version": "legacy_dirs",
            "index_schema_version": INDEX_CACHE_VERSION,
            "index_cache_key": cache_key,
            "indexed_sample_count": len(self.samples),
            "index_manifest_sha256": self.index_manifest_sha256,
            "dataset_filter": list(self.expected_datasets),
            "bbox_source_policy": self.bbox_source_policy,
            "bbox_manifest_sha256": dict(getattr(self, "bbox_manifest_sha256", {})),
            "lazy_index_records": bool(getattr(self, "lazy_index_records", False)),
            "depth_sidecar_contract": dict(self.depth_sidecar_contract),
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
            if not self.rebuild_index:
                missing_parts = []
                if not os.path.exists(cache_path):
                    missing_parts.append("cache")
                if not os.path.exists(done_path):
                    missing_parts.append("completion marker")
                print(
                    f"[{self.split}] No complete compatible index cache "
                    f"({', '.join(missing_parts) or 'unknown cache state'}); "
                    "building this cache key once. This is not a forced rebuild. "
                    f"Path: {cache_path}",
                    flush=True,
                )
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
                executor_kwargs = {"max_workers": effective_workers}
                if executor_cls is ProcessPoolExecutor:
                    executor_kwargs["initializer"] = initialize_worker_parent_death_signal
                with executor_cls(**executor_kwargs) as executor:
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
