#!/usr/bin/env python3
"""Build atomic, resumable HaMeR sidecars for TouchAnything query manifests.

The cache intentionally stores right-canonical MANO parameters plus optional
source-camera 778-vertex geometry. Full-image UV, tactile-crop UV, and any
dense 13,614-vertex representation are derived later instead of duplicated for
every query.
"""

from __future__ import annotations

import argparse
from array import array
import bisect
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import cv2
import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler


try:
    import orjson
except ImportError:  # pragma: no cover - depends on the runtime environment.
    orjson = None


REPO_ROOT = Path(__file__).resolve().parents[1]
HAMER_ROOT = REPO_ROOT / "hamer"
DEFAULT_CHECKPOINT = HAMER_ROOT / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
DEFAULT_MANO = HAMER_ROOT / "_DATA/data/mano/MANO_RIGHT.pkl"
DEFAULT_BBOX_MANIFEST = (
    REPO_ROOT
    / "sam3_bbox_reconstruction/outputs/full_reconstruction_flow/touchanything/manifests"
    / "touchanything_sam3_v1_highconf.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/home/ma-user/work/cfzhao/hand_pose_sidecars/touchanything_hamer_v1"
)
DEFAULT_PROCESSED_ROOTS = (
    Path("/home/ma-user/work/cfzhao/EgoTouch/extracted_frames"),
    REPO_ROOT / "EgoTouch/extracted_frames",
)
DEFAULT_SPLITS = ("train", "val", "test_seen", "test_unseen")
SCHEMA_NAME = "touchanything_hamer_pose_sidecar"
SCHEMA_VERSION = "1.1.0"
SHARD_SCHEMA = "touchanything_hamer_pose_shard"
DONE_SCHEMA = "touchanything_hamer_pose_shard_done"
CONFIG_FILE = "sidecar_config.json"
MANIFEST_FILE = "sidecar_manifest.json"
ROOT_DONE_FILE = "SIDECAR_DONE.json"
PROGRESS_FILE = "progress.json"
STATUS_VALID = 1
STATUS_NONFINITE = 2
STATUS_MISSING_SAM3_BBOX = 3
BBOX_MISSING_SAM3 = 0
BBOX_MANIFEST_SAM3 = 1
BBOX_CURRENT_SAM3_OVERLAY = 2
BBOX_UNVERIFIED_FALLBACK = 3


def _json_loads(raw: bytes | str) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    return json.loads(raw)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _file_identity(path: Path, *, digest: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def _cached_sha256(path: Path) -> str:
    cache_path = path.with_name(f".{path.name}.sha256.json")
    stat = path.stat()
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                int(cached.get("size_bytes", -1)) == int(stat.st_size)
                and int(cached.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
                and re.fullmatch(r"[0-9a-f]{64}", str(cached.get("sha256", "")))
            ):
                return str(cached["sha256"])
        except (OSError, ValueError, TypeError):
            pass
    print(f"[hamer-sidecar] hashing model asset: {path}", flush=True)
    digest = _sha256_file(path)
    try:
        _atomic_json(
            cache_path,
            {
                "path": str(path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": digest,
            },
        )
    except OSError:
        pass
    return digest


def _resolve_processed_root(raw: str | None) -> Path:
    candidates: list[Path] = []
    if raw:
        candidates.append(Path(raw).expanduser())
    env_value = os.environ.get("TOUCHANYTHING_PROCESSED_ROOT", "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(DEFAULT_PROCESSED_ROOTS)
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    for candidate in unique:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find TouchAnything processed HDF5 root; pass --processed-root. "
        f"Checked: {[str(value) for value in unique]}"
    )


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in str(value).split(",") if item.strip())
    unknown = [item for item in splits if item not in DEFAULT_SPLITS]
    if not splits or unknown or len(set(splits)) != len(splits):
        raise ValueError(
            f"--splits must be unique values from {DEFAULT_SPLITS}; got {splits}"
        )
    return splits


def _infer_split(path: Path) -> str:
    match = re.search(r"touchanything_(train|val|test_seen|test_unseen)\.queries\.jsonl$", path.name)
    if not match:
        raise ValueError(
            f"Cannot infer split from {path}; use the SPLIT=/path form for --manifest"
        )
    return match.group(1)


def _resolve_manifests(
    processed_root: Path,
    splits: Sequence[str],
    raw_manifests: Sequence[str],
) -> list[tuple[str, Path]]:
    if raw_manifests:
        values = []
        for raw in raw_manifests:
            if "=" in raw:
                split, path_text = raw.split("=", 1)
                split = split.strip()
                path = Path(path_text).expanduser().resolve(strict=True)
            else:
                path = Path(raw).expanduser().resolve(strict=True)
                split = _infer_split(path)
            if split not in DEFAULT_SPLITS:
                raise ValueError(f"Unsupported manifest split {split!r}")
            values.append((split, path))
    else:
        values = [
            (
                split,
                (processed_root / f"manifests/touchanything_{split}.queries.jsonl").resolve(
                    strict=True
                ),
            )
            for split in splits
        ]
    names = [split for split, _ in values]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate source split(s): {names}")
    return values


def _request_contract(args: argparse.Namespace) -> dict[str, Any]:
    processed_root = _resolve_processed_root(args.processed_root)
    splits = _parse_splits(args.splits)
    manifests = _resolve_manifests(processed_root, splits, args.manifest)
    bbox_manifest = (
        None
        if not args.bbox_manifest
        else str(Path(args.bbox_manifest).expanduser().resolve(strict=True))
    )
    unresolved_bbox_policy = str(args.unresolved_bbox_policy)
    if args.allow_unverified_bbox_fallback:
        if unresolved_bbox_policy not in ("skip", "manifest_fallback"):
            raise ValueError(
                "--allow-unverified-bbox-fallback conflicts with "
                f"--unresolved-bbox-policy={unresolved_bbox_policy}"
            )
        unresolved_bbox_policy = "manifest_fallback"
    return {
        "processed_root": str(processed_root),
        "manifests": [{"split": split, "path": str(path)} for split, path in manifests],
        "bbox_manifest": bbox_manifest,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve(strict=True)),
        "mano": str(Path(args.mano).expanduser().resolve(strict=True)),
        "shard_size": int(args.shard_size),
        "max_samples": None if args.max_samples is None else int(args.max_samples),
        "hamer_bbox_scale": float(args.hamer_bbox_scale),
        "precision": str(args.precision),
        "compression": str(args.compression),
        "store_camera_vertices": bool(args.store_camera_vertices),
        "unresolved_bbox_policy": unresolved_bbox_policy,
    }


def _bbox_valid(value: Any) -> bool:
    try:
        bbox = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return bool(
        bbox.shape == (4,)
        and np.isfinite(bbox).all()
        and bbox[2] > bbox[0]
        and bbox[3] > bbox[1]
    )


def _float_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _normalized_alias(value: Any) -> str:
    alias = str(value or "").strip().lower()
    if alias in ("l", "left"):
        return "left"
    if alias in ("r", "right"):
        return "right"
    return alias


def _bbox_key(row: Mapping[str, Any], split_hint: str) -> tuple[str, str, int, str] | None:
    split = str(row.get("source_split", row.get("split", split_hint)) or split_hint)
    sequence = str(row.get("sequence_key") or "").strip().replace("\\", "/").strip("/")
    frame = row.get("frame_idx")
    alias = _normalized_alias(
        row.get("target_hand", row.get("query_alias", row.get("hand")))
    )
    if not sequence or frame is None or alias not in ("left", "right"):
        return None
    try:
        frame_index = int(frame)
    except (TypeError, ValueError, OverflowError):
        return None
    return split, sequence, frame_index, alias


def _is_sam3_source(row: Mapping[str, Any]) -> bool:
    source = row.get("bbox_source")
    return isinstance(source, Mapping) and str(source.get("schema")) == "sam3_bbox_source_v1"


def _save_source_indices(
    output_dir: Path,
    source_states: Sequence[dict[str, Any]],
) -> None:
    index_root = output_dir / "source_index"
    for state in source_states:
        split = state["split"]
        paths = {
            "offsets": index_root / f"{split}.offsets.npy",
            "bbox_xyxy": index_root / f"{split}.bbox_xyxy.npy",
            "bbox_score": index_root / f"{split}.bbox_score.npy",
            "bbox_source_code": index_root / f"{split}.bbox_source_code.npy",
        }
        arrays = {
            "offsets": np.asarray(state["offsets"], dtype=np.int64),
            "bbox_xyxy": np.asarray(state["bboxes"], dtype=np.float32).reshape(-1, 4),
            "bbox_score": np.asarray(state["scores"], dtype=np.float32),
            "bbox_source_code": np.frombuffer(state["codes"], dtype=np.uint8).copy(),
        }
        for key, path in paths.items():
            _atomic_npy(path, arrays[key])
        state["index_files"] = {
            key: {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256_file(path),
                "shape": list(arrays[key].shape),
                "dtype": arrays[key].dtype.str,
            }
            for key, path in paths.items()
        }


def _prepare_new_config(args: argparse.Namespace, request: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    source_states: list[dict[str, Any]] = []
    required_overlays: dict[tuple[str, str, int, str], list[tuple[int, int]]] = {}
    unresolved_examples: list[str] = []
    remaining = request["max_samples"]
    global_start = 0

    for source_index, source in enumerate(request["manifests"]):
        split = str(source["split"])
        path = Path(source["path"])
        offsets = array("q")
        bboxes = array("f")
        scores = array("f")
        codes = bytearray()
        digest = hashlib.sha256()
        ids_digest = hashlib.sha256()
        embedded_sam3_hashes: set[str] = set()
        count = 0
        with path.open("rb") as handle:
            while remaining is None or remaining > 0:
                byte_offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                digest.update(raw)
                if not raw.strip():
                    continue
                row = _json_loads(raw)
                if not isinstance(row, Mapping):
                    raise ValueError(f"{path}: source row {count} is not a JSON object")
                uid = str(row.get("sample_uid") or "").strip()
                bbox = row.get("bbox_xyxy", row.get("bbox"))
                if not uid:
                    raise ValueError(f"{path}: source row {count} has no sample_uid")
                for required in ("h5_relpath", "frame_row", "query_row", "is_right"):
                    if row.get(required) is None:
                        raise ValueError(f"{path}: source row {count} lacks {required!r}")
                offsets.append(byte_offset)
                manifest_bbox_valid = _bbox_valid(bbox)
                bboxes.extend(
                    (float(value) for value in bbox)
                    if manifest_bbox_valid
                    else (float("nan"),) * 4
                )
                scores.append(_float_or_nan(row.get("bbox_score")))
                ids_digest.update(uid.encode("utf-8"))
                ids_digest.update(b"\n")
                if manifest_bbox_valid and _is_sam3_source(row):
                    codes.append(BBOX_MANIFEST_SAM3)
                    source_meta = row.get("bbox_source")
                    source_hash = str(source_meta.get("source_manifest_sha256") or "")
                    if source_hash:
                        embedded_sam3_hashes.add(source_hash)
                else:
                    codes.append(BBOX_MISSING_SAM3)
                    key = _bbox_key(row, split)
                    if key is None:
                        if len(unresolved_examples) < 20:
                            unresolved_examples.append(uid)
                    else:
                        required_overlays.setdefault(key, []).append((source_index, count))
                count += 1
                if remaining is not None:
                    remaining -= 1
        state = {
            "split": split,
            "path": str(path),
            "count": count,
            "global_start": global_start,
            "global_stop": global_start + count,
            "offsets": offsets,
            "bboxes": bboxes,
            "scores": scores,
            "codes": codes,
            "consumed_sha256": digest.hexdigest(),
            "sample_uids_sha256": ids_digest.hexdigest(),
            "embedded_sam3_manifest_sha256": sorted(embedded_sam3_hashes),
        }
        source_states.append(state)
        global_start += count
        print(
            f"[hamer-sidecar] indexed {split}: {count:,} rows; "
            f"SAM3 overlay still required={sum(len(v) for v in required_overlays.values()):,}",
            flush=True,
        )
        if remaining == 0:
            break

    bbox_identity = None
    raw_bbox_manifest = request.get("bbox_manifest")
    if required_overlays and raw_bbox_manifest:
        bbox_manifest = Path(str(raw_bbox_manifest))
        bbox_digest = hashlib.sha256()
        selected = 0
        invalid = 0
        scanned = 0
        with bbox_manifest.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                bbox_digest.update(raw)
                if not raw.strip():
                    continue
                scanned += 1
                row = _json_loads(raw)
                if not isinstance(row, Mapping):
                    raise ValueError(f"{bbox_manifest}:{line_number}: expected a JSON object")
                key = _bbox_key(row, str(row.get("split") or ""))
                locations = required_overlays.get(key) if key is not None else None
                if not locations:
                    continue
                bbox = row.get("bbox_xyxy", row.get("bbox"))
                if not _bbox_valid(bbox):
                    invalid += len(locations)
                    continue
                score = _float_or_nan(row.get("bbox_score", row.get("prompt_score")))
                for source_index, row_index in locations:
                    state = source_states[source_index]
                    start = row_index * 4
                    state["bboxes"][start : start + 4] = array(
                        "f", (float(value) for value in bbox)
                    )
                    state["scores"][row_index] = score
                    state["codes"][row_index] = BBOX_CURRENT_SAM3_OVERLAY
                    selected += 1
                del required_overlays[key]
        bbox_identity = _file_identity(bbox_manifest, digest=bbox_digest.hexdigest())
        print(
            f"[hamer-sidecar] scanned {scanned:,} current SAM3 rows; "
            f"resolved {selected:,} query bbox(es); "
            f"matched rows with invalid bbox={invalid:,}",
            flush=True,
        )
    elif required_overlays:
        print(
            "[hamer-sidecar] no current SAM3 bbox manifest supplied; "
            f"{sum(len(v) for v in required_overlays.values()):,} candidate row(s) "
            "remain unresolved",
            flush=True,
        )

    unresolved_count = sum(state["codes"].count(BBOX_MISSING_SAM3) for state in source_states)
    if unresolved_count:
        unresolved_examples.extend(
            "/".join((split, sequence, str(frame), alias))
            for split, sequence, frame, alias in list(required_overlays)[:20]
        )
        policy = str(request["unresolved_bbox_policy"])
        if policy == "error":
            raise RuntimeError(
                f"{unresolved_count:,} query rows have no matching SAM3 bbox. "
                "Use a complete bbox manifest or choose --unresolved-bbox-policy=skip. "
                f"Examples: {unresolved_examples[:5]}"
            )
        if policy == "manifest_fallback":
            fallback_count = 0
            for state in source_states:
                for row_index, code in enumerate(state["codes"]):
                    if code != BBOX_MISSING_SAM3:
                        continue
                    start = row_index * 4
                    bbox = state["bboxes"][start : start + 4]
                    if _bbox_valid(bbox):
                        state["codes"][row_index] = BBOX_UNVERIFIED_FALLBACK
                        fallback_count += 1
            unresolved_count -= fallback_count
            print(
                f"[hamer-sidecar] WARNING: using {fallback_count:,} unverified manifest "
                f"bbox(es); {unresolved_count:,} rows still have no usable bbox",
                flush=True,
            )
        else:
            print(
                f"[hamer-sidecar] retaining {unresolved_count:,} rows as "
                "missing_sam3_bbox; inference will skip them",
                flush=True,
            )

    bbox_code_counts: dict[str, int] = {}
    for state in source_states:
        state["bbox_code_counts"] = {
            str(code): int(state["codes"].count(code))
            for code in (
                BBOX_MISSING_SAM3,
                BBOX_MANIFEST_SAM3,
                BBOX_CURRENT_SAM3_OVERLAY,
                BBOX_UNVERIFIED_FALLBACK,
            )
        }
        for code, count in state["bbox_code_counts"].items():
            bbox_code_counts[code] = bbox_code_counts.get(code, 0) + count

    _save_source_indices(output_dir, source_states)
    checkpoint = Path(request["checkpoint"])
    mano = Path(request["mano"])
    model_config = checkpoint.parent.parent / "model_config.yaml"
    asset_identities = {
        "checkpoint": _file_identity(checkpoint, digest=_cached_sha256(checkpoint)),
        "mano": _file_identity(mano, digest=_cached_sha256(mano)),
        "model_config": _file_identity(model_config, digest=_sha256_file(model_config)),
    }
    sources = []
    work_items = []
    global_work_index = 0
    for state in source_states:
        source_entry = {
            key: state[key]
            for key in (
                "split",
                "path",
                "count",
                "global_start",
                "global_stop",
                "consumed_sha256",
                "sample_uids_sha256",
                "embedded_sam3_manifest_sha256",
                "bbox_code_counts",
                "index_files",
            )
        }
        source_entry["file"] = _file_identity(Path(state["path"]))
        sources.append(source_entry)
        shard_count = math.ceil(int(state["count"]) / int(request["shard_size"]))
        for shard_index in range(shard_count):
            start = shard_index * int(request["shard_size"])
            stop = min(start + int(request["shard_size"]), int(state["count"]))
            work_items.append(
                {
                    "global_work_index": global_work_index,
                    "source_index": len(sources) - 1,
                    "split": state["split"],
                    "shard_index": shard_index,
                    "source_start": start,
                    "source_stop": stop,
                    "sample_count": stop - start,
                    "path": f"shards/{state['split']}/shard-{shard_index:06d}.h5",
                }
            )
            global_work_index += 1

    total_count = sum(int(source["count"]) for source in sources)
    bytes_per_record = 724 + (778 * 3 * 2 if request["store_camera_vertices"] else 0)
    config_body = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "request": dict(request),
        "assets": asset_identities,
        "bbox_manifest_used": bbox_identity,
        "bbox_source_codes": {
            str(BBOX_MISSING_SAM3): "missing_sam3_bbox",
            str(BBOX_MANIFEST_SAM3): "manifest_sam3",
            str(BBOX_CURRENT_SAM3_OVERLAY): "current_sam3_overlay",
            str(BBOX_UNVERIFIED_FALLBACK): "unverified_manifest_fallback",
        },
        "bbox_resolution_summary": bbox_code_counts,
        "unresolved_bbox_examples": unresolved_examples[:20],
        "mano_parameter_space": "right_hand_canonical",
        "source_camera_geometry": "left hands mirrored into original image camera",
        "mano_vertex_count": 778,
        "sources": sources,
        "work_items": work_items,
        "sample_count": total_count,
        "shard_count": len(work_items),
        "estimated_uncompressed_bytes": total_count * bytes_per_record,
    }
    config = dict(config_body)
    config["config_sha256"] = _sha256_json(config_body)
    _atomic_json(output_dir / CONFIG_FILE, config)
    free_bytes = shutil.disk_usage(output_dir).free
    print(
        f"[hamer-sidecar] prepared {total_count:,} samples / {len(work_items):,} shards; "
        f"uncompressed estimate={config['estimated_uncompressed_bytes'] / 1e9:.1f} GB; "
        f"free={free_bytes / 1e9:.1f} GB",
        flush=True,
    )
    return config


def _validate_source_identity(record: Mapping[str, Any]) -> None:
    identity = record["file"]
    path = Path(identity["path"])
    stat = path.stat()
    if int(stat.st_size) != int(identity["size_bytes"]):
        raise RuntimeError(f"Source manifest size changed since cache preparation: {path}")
    if int(stat.st_mtime_ns) != int(identity["mtime_ns"]):
        digest = hashlib.sha256()
        consumed_rows = 0
        with path.open("rb") as handle:
            while consumed_rows < int(record["count"]):
                raw = handle.readline()
                if not raw:
                    break
                digest.update(raw)
                consumed_rows += int(bool(raw.strip()))
        consumed = digest.hexdigest()
        if consumed != record["consumed_sha256"]:
            raise RuntimeError(f"Source manifest content changed: {path}")


def _validate_prepared_indices(output_dir: Path, source: Mapping[str, Any]) -> None:
    for name, spec in source["index_files"].items():
        path = output_dir / str(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Prepared source index is missing: {path}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = tuple(int(item) for item in spec["shape"])
        expected_dtype = np.dtype(spec["dtype"])
        if value.shape != expected_shape or value.dtype != expected_dtype:
            raise RuntimeError(
                f"Prepared index contract differs for {path}: "
                f"shape={value.shape}, dtype={value.dtype}"
            )
        del value


def _prepare_or_reuse(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = _request_contract(args)
    config_path = output_dir / CONFIG_FILE
    if not config_path.is_file():
        return _prepare_new_config(args, request)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema") != SCHEMA_NAME
        or config.get("schema_version") != SCHEMA_VERSION
    ):
        raise RuntimeError(
            f"Existing sidecar uses schema {config.get('schema_version')!r}, but this "
            f"exporter writes {SCHEMA_VERSION}. Use a new output directory."
        )
    body = {key: value for key, value in config.items() if key != "config_sha256"}
    if config.get("config_sha256") != _sha256_json(body):
        raise RuntimeError(f"Sidecar config checksum is invalid: {config_path}")
    if config.get("request") != request:
        raise RuntimeError(
            f"Existing sidecar configuration differs under {output_dir}; use a new output "
            "directory or rerun with the original semantic arguments"
        )
    for source in config["sources"]:
        _validate_source_identity(source)
        _validate_prepared_indices(output_dir, source)
    print(
        f"[hamer-sidecar] reusing prepared config: {config['sample_count']:,} samples, "
        f"{config['shard_count']:,} shards",
        flush=True,
    )
    return config


def _open_hdf5_readonly(path: Path) -> h5py.File:
    attempts = (
        {"mode": "r", "libver": "latest", "swmr": True, "locking": False},
        {"mode": "r", "libver": "latest", "locking": False},
        {"mode": "r", "libver": "latest", "swmr": True},
        {"mode": "r", "libver": "latest"},
    )
    errors = []
    for kwargs in attempts:
        try:
            return h5py.File(path, **kwargs)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{kwargs}: {exc}")
    raise OSError(f"Could not open HDF5 {path}: {' | '.join(errors)}")


class _IndexedCropDataset(Dataset):
    def __init__(
        self,
        output_dir: Path,
        config: Mapping[str, Any],
        work_items: Sequence[Mapping[str, Any]],
        *,
        image_size: int,
        image_mean: Sequence[float],
        image_std: Sequence[float],
        frame_cache_size: int,
        max_hdf5_handles: int = 8,
    ) -> None:
        self.output_dir = str(output_dir)
        self.processed_root = str(config["request"]["processed_root"])
        self.sources = list(config["sources"])
        self.work_items = [dict(item) for item in work_items]
        self.image_size = int(image_size)
        self.image_mean = np.asarray(image_mean, dtype=np.float32)
        self.image_std = np.asarray(image_std, dtype=np.float32)
        self.hamer_bbox_scale = float(config["request"]["hamer_bbox_scale"])
        self.frame_cache_size = max(int(frame_cache_size), 0)
        self.max_hdf5_handles = max(int(max_hdf5_handles), 1)
        self.prefix = [0]
        for item in self.work_items:
            self.prefix.append(self.prefix[-1] + int(item["sample_count"]))
        self._manifest_handles: dict[int, Any] = {}
        self._index_arrays: dict[tuple[int, str], np.ndarray] = {}
        self._hdf5_handles: OrderedDict[str, h5py.File] = OrderedDict()
        self._frame_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return self.prefix[-1]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_manifest_handles"] = {}
        state["_index_arrays"] = {}
        state["_hdf5_handles"] = OrderedDict()
        state["_frame_cache"] = OrderedDict()
        return state

    def _source_array(self, source_index: int, name: str) -> np.ndarray:
        key = (source_index, name)
        value = self._index_arrays.get(key)
        if value is None:
            relative = self.sources[source_index]["index_files"][name]["path"]
            value = np.load(
                Path(self.output_dir) / relative,
                mmap_mode="r",
                allow_pickle=False,
            )
            self._index_arrays[key] = value
        return value

    def _source_row(self, source_index: int, row_index: int) -> Mapping[str, Any]:
        handle = self._manifest_handles.get(source_index)
        if handle is None:
            handle = open(self.sources[source_index]["path"], "rb")
            self._manifest_handles[source_index] = handle
        offsets = self._source_array(source_index, "offsets")
        handle.seek(int(offsets[row_index]))
        row = _json_loads(handle.readline())
        if not isinstance(row, Mapping):
            raise ValueError(f"Manifest source row {source_index}:{row_index} is invalid")
        return row

    def _hdf5(self, path: Path) -> h5py.File:
        key = str(path)
        handle = self._hdf5_handles.pop(key, None)
        if handle is None or not handle.id.valid:
            handle = _open_hdf5_readonly(path)
        self._hdf5_handles[key] = handle
        while len(self._hdf5_handles) > self.max_hdf5_handles:
            _, old = self._hdf5_handles.popitem(last=False)
            old.close()
        return handle

    def _decode_frame(self, path: Path, frame_row: int) -> np.ndarray:
        key = (str(path), int(frame_row))
        cached = self._frame_cache.pop(key, None)
        if cached is not None:
            self._frame_cache[key] = cached
            return cached
        handle = self._hdf5(path)
        offsets = handle["images/rgb/jpeg_offsets"]
        start, stop = np.asarray(offsets[frame_row : frame_row + 2], dtype=np.uint64)
        encoded = np.asarray(
            handle["images/rgb/jpeg_data"][int(start) : int(stop)], dtype=np.uint8
        )
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode frame {frame_row} from {path}")
        if self.frame_cache_size:
            self._frame_cache[key] = image
            while len(self._frame_cache) > self.frame_cache_size:
                self._frame_cache.popitem(last=False)
        return image

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        work_position = bisect.bisect_right(self.prefix, int(index)) - 1
        item = self.work_items[work_position]
        local_index = int(index) - self.prefix[work_position]
        source_index = int(item["source_index"])
        source_row = int(item["source_start"]) + local_index
        row = self._source_row(source_index, source_row)
        bbox = np.asarray(
            self._source_array(source_index, "bbox_xyxy")[source_row], dtype=np.float32
        )
        bbox_score = float(self._source_array(source_index, "bbox_score")[source_row])
        bbox_source_code = int(
            self._source_array(source_index, "bbox_source_code")[source_row]
        )
        is_right = int(row["is_right"])
        frame_row = int(row["frame_row"])
        common = {
            "work_position": work_position,
            "source_row": source_row,
            "query_row": int(row["query_row"]),
            "frame_row": frame_row,
            "sample_uid": str(row["sample_uid"]),
            "bbox_xyxy": torch.from_numpy(bbox.copy()),
            "bbox_score": bbox_score,
            "bbox_source_code": bbox_source_code,
        }
        if is_right not in (0, 1):
            raise ValueError(f"Invalid handedness for {row.get('sample_uid')!r}")
        if bbox_source_code == BBOX_MISSING_SAM3:
            return {
                **common,
                "img": torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32),
                "right": torch.tensor(float(is_right), dtype=torch.float32),
                "box_center": torch.zeros(2, dtype=torch.float32),
                "box_size": torch.tensor(1.0, dtype=torch.float32),
                "img_size": torch.zeros(2, dtype=torch.float32),
            }
        relative = str(row["h5_relpath"])
        h5_path = (Path(self.processed_root) / relative).resolve(strict=True)
        try:
            h5_path.relative_to(Path(self.processed_root))
        except ValueError as exc:
            raise ValueError(f"h5_relpath escapes processed root: {relative!r}") from exc
        image = self._decode_frame(h5_path, frame_row)
        center = (bbox[:2] + bbox[2:]) * 0.5
        box_size = float(np.max(bbox[2:] - bbox[:2])) * self.hamer_bbox_scale
        if not math.isfinite(box_size) or box_size <= 1.0:
            raise ValueError(f"Invalid hand query {row.get('sample_uid')!r}")
        transform = np.zeros((2, 3), dtype=np.float32)
        transform[0, 0] = self.image_size / box_size
        transform[1, 1] = self.image_size / box_size
        transform[0, 2] = self.image_size * (-float(center[0]) / box_size + 0.5)
        transform[1, 2] = self.image_size * (-float(center[1]) / box_size + 0.5)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        patch = cv2.warpAffine(
            rgb,
            transform,
            (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR,
        ).astype(np.float32) / 255.0
        if not is_right:
            patch = cv2.flip(patch, 1)
        patch = ((patch - self.image_mean) / self.image_std).transpose(2, 0, 1).copy()
        return {
            **common,
            "img": torch.from_numpy(patch),
            "right": torch.tensor(float(is_right), dtype=torch.float32),
            "box_center": torch.from_numpy(center.copy()),
            "box_size": torch.tensor(box_size, dtype=torch.float32),
            "img_size": torch.tensor([image.shape[1], image.shape[0]], dtype=torch.float32),
        }


class _ShardBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: _IndexedCropDataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch size must be positive")

    def __iter__(self):
        for start, stop in zip(self.dataset.prefix[:-1], self.dataset.prefix[1:]):
            for batch_start in range(start, stop, self.batch_size):
                yield list(range(batch_start, min(batch_start + self.batch_size, stop)))

    def __len__(self) -> int:
        return sum(
            math.ceil((stop - start) / self.batch_size)
            for start, stop in zip(self.dataset.prefix[:-1], self.dataset.prefix[1:])
        )


def _worker_init(_: int) -> None:
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def _compression_kwargs(name: str) -> dict[str, Any]:
    if name == "none":
        return {}
    if name == "lzf":
        return {"compression": "lzf", "shuffle": True}
    if name == "gzip1":
        return {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    raise ValueError(f"Unknown compression: {name}")


class _PoseShardWriter:
    def __init__(
        self,
        output_dir: Path,
        item: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> None:
        self.output_dir = output_dir
        self.item = dict(item)
        self.config = config
        self.path = output_dir / str(item["path"])
        self.temporary = self.path.with_name(
            f".{self.path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
        )
        self.count = int(item["sample_count"])
        self.position = 0
        self.uid_digest = hashlib.sha256()
        self.valid_count = 0
        self.nonfinite_count = 0
        self.missing_bbox_count = 0
        self.handle: h5py.File | None = None
        self.datasets: dict[str, h5py.Dataset] = {}

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.temporary, "w", libver="latest")
        attrs = self.handle.attrs
        attrs["schema_name"] = SHARD_SCHEMA
        attrs["schema_version"] = SCHEMA_VERSION
        attrs["complete"] = np.uint8(0)
        attrs["config_sha256"] = self.config["config_sha256"]
        attrs["split"] = self.item["split"]
        attrs["shard_index"] = int(self.item["shard_index"])
        attrs["source_start"] = int(self.item["source_start"])
        attrs["source_stop"] = int(self.item["source_stop"])
        attrs["record_count"] = self.count
        attrs["mano_parameter_space"] = "right_hand_canonical"
        attrs["camera_geometry_space"] = "source_image_camera"
        attrs["status_codes_json"] = _canonical_json(
            {
                str(STATUS_VALID): "valid",
                str(STATUS_NONFINITE): "nonfinite",
                str(STATUS_MISSING_SAM3_BBOX): "missing_sam3_bbox",
            }
        )
        query = self.handle.create_group("queries")
        camera = self.handle.create_group("camera")
        mano = self.handle.create_group("mano")
        quality = self.handle.create_group("quality")
        chunk = min(max(self.count, 1), 256)
        compression = _compression_kwargs(self.config["request"]["compression"])

        def create(group, name, shape, dtype, *, compressed=True, row_chunk=chunk):
            kwargs: dict[str, Any] = {}
            if shape[0] > 0:
                kwargs["chunks"] = (min(row_chunk, shape[0]), *shape[1:])
                if compressed:
                    kwargs.update(compression)
            value = group.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
            self.datasets[f"{group.name}/{name}"] = value
            return value

        create(
            query,
            "sample_uid",
            (self.count,),
            h5py.string_dtype(encoding="utf-8"),
            compressed=False,
        )
        for name, dtype in (("source_row", "<i8"), ("query_row", "<i8"), ("frame_row", "<i8")):
            create(query, name, (self.count,), dtype)
        create(query, "is_right", (self.count,), "u1")
        create(query, "bbox_source_code", (self.count,), "u1")
        create(query, "bbox_xyxy", (self.count, 4), "<f4")
        create(query, "bbox_score", (self.count,), "<f4")
        create(camera, "image_wh", (self.count, 2), "<u2")
        create(camera, "focal_length", (self.count,), "<f4")
        create(camera, "translation", (self.count, 3), "<f4")
        create(mano, "global_orient", (self.count, 1, 3, 3), "<f4")
        create(mano, "hand_pose", (self.count, 15, 3, 3), "<f4")
        create(mano, "betas", (self.count, 10), "<f4")
        if self.config["request"]["store_camera_vertices"]:
            create(
                self.handle.create_group("geometry"),
                "vertices_camera",
                (self.count, 778, 3),
                "<f2",
                row_chunk=32,
            )
        create(quality, "status", (self.count,), "u1")
        create(quality, "positive_depth_fraction", (self.count,), "<f2")
        create(quality, "in_frame_fraction", (self.count,), "<f2")
        return self

    def append(self, batch: Mapping[str, Any], output: Mapping[str, np.ndarray]) -> None:
        count = len(batch["sample_uid"])
        start, stop = self.position, self.position + count
        if stop > self.count:
            raise RuntimeError("Shard writer received too many records")
        expected_rows = np.arange(
            int(self.item["source_start"]) + start,
            int(self.item["source_start"]) + stop,
            dtype=np.int64,
        )
        source_rows = np.asarray(batch["source_row"], dtype=np.int64)
        if not np.array_equal(source_rows, expected_rows):
            raise RuntimeError(
                f"Shard rows are not contiguous: expected {expected_rows[:3]}, got {source_rows[:3]}"
            )
        uids = [str(value) for value in batch["sample_uid"]]
        self.datasets["/queries/sample_uid"][start:stop] = np.asarray(uids, dtype=object)
        for uid in uids:
            self.uid_digest.update(uid.encode("utf-8"))
            self.uid_digest.update(b"\n")
        assignments = {
            "/queries/source_row": source_rows,
            "/queries/query_row": np.asarray(batch["query_row"], dtype=np.int64),
            "/queries/frame_row": np.asarray(batch["frame_row"], dtype=np.int64),
            "/queries/is_right": np.asarray(batch["right"], dtype=np.uint8),
            "/queries/bbox_source_code": np.asarray(batch["bbox_source_code"], dtype=np.uint8),
            "/queries/bbox_xyxy": np.asarray(batch["bbox_xyxy"], dtype=np.float32),
            "/queries/bbox_score": np.asarray(batch["bbox_score"], dtype=np.float32),
            "/camera/image_wh": output["image_wh"],
            "/camera/focal_length": output["focal_length"],
            "/camera/translation": output["camera_translation"],
            "/mano/global_orient": output["global_orient"],
            "/mano/hand_pose": output["hand_pose"],
            "/mano/betas": output["betas"],
            "/quality/status": output["status"],
            "/quality/positive_depth_fraction": output["positive_depth_fraction"],
            "/quality/in_frame_fraction": output["in_frame_fraction"],
        }
        if "/geometry/vertices_camera" in self.datasets:
            assignments["/geometry/vertices_camera"] = output["vertices_camera"]
        for name, value in assignments.items():
            self.datasets[name][start:stop] = value
        status = np.asarray(output["status"])
        self.valid_count += int((status == STATUS_VALID).sum())
        self.nonfinite_count += int((status == STATUS_NONFINITE).sum())
        self.missing_bbox_count += int((status == STATUS_MISSING_SAM3_BBOX).sum())
        self.position = stop

    def _abort(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.temporary.unlink(missing_ok=True)

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self._abort()
            return False
        try:
            if self.position != self.count:
                raise RuntimeError(f"Shard expected {self.count} rows, wrote {self.position}")
            assert self.handle is not None
            self.handle.attrs["valid_count"] = self.valid_count
            self.handle.attrs["nonfinite_count"] = self.nonfinite_count
            self.handle.attrs["missing_bbox_count"] = self.missing_bbox_count
            self.handle.attrs["sample_uids_sha256"] = self.uid_digest.hexdigest()
            self.handle.attrs["complete"] = np.uint8(1)
            self.handle.flush()
            self.handle.close()
            self.handle = None
            descriptor = os.open(self.temporary, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            digest = _sha256_file(self.temporary)
            os.replace(self.temporary, self.path)
            _fsync_directory(self.path.parent)
            _atomic_json(
                self.path.with_suffix(".done.json"),
                {
                    "schema": DONE_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "config_sha256": self.config["config_sha256"],
                    "split": self.item["split"],
                    "shard_index": int(self.item["shard_index"]),
                    "record_count": self.count,
                    "valid_count": self.valid_count,
                    "nonfinite_count": self.nonfinite_count,
                    "missing_bbox_count": self.missing_bbox_count,
                    "sample_uids_sha256": self.uid_digest.hexdigest(),
                    "hdf5_sha256": digest,
                    "size_bytes": self.path.stat().st_size,
                },
            )
        except BaseException:
            self._abort()
            raise
        return False


def _validate_shard(
    output_dir: Path,
    item: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    deep: bool,
    recover_done: bool = True,
) -> dict[str, Any]:
    path = output_dir / str(item["path"])
    done_path = path.with_suffix(".done.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r", libver="latest") as handle:
        if str(handle.attrs.get("schema_name", "")) != SHARD_SCHEMA:
            raise RuntimeError(f"Invalid shard schema: {path}")
        if str(handle.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            raise RuntimeError(f"Invalid shard schema version: {path}")
        if int(handle.attrs.get("complete", 0)) != 1:
            raise RuntimeError(f"Shard is incomplete: {path}")
        if str(handle.attrs.get("config_sha256", "")) != config["config_sha256"]:
            raise RuntimeError(f"Shard config differs: {path}")
        count = int(handle.attrs.get("record_count", -1))
        if count != int(item["sample_count"]):
            raise RuntimeError(f"Shard row count differs: {path}")
        expected_shapes = {
            "queries/source_row": (count,),
            "queries/sample_uid": (count,),
            "mano/hand_pose": (count, 15, 3, 3),
            "quality/status": (count,),
        }
        if config["request"]["store_camera_vertices"]:
            expected_shapes["geometry/vertices_camera"] = (count, 778, 3)
        for name, shape in expected_shapes.items():
            if name not in handle or handle[name].shape != shape:
                raise RuntimeError(f"Shard field contract differs for {path}:{name}")
        embedded = {
            "valid_count": int(handle.attrs.get("valid_count", 0)),
            "nonfinite_count": int(handle.attrs.get("nonfinite_count", 0)),
            "missing_bbox_count": int(handle.attrs.get("missing_bbox_count", 0)),
            "sample_uids_sha256": str(handle.attrs.get("sample_uids_sha256", "")),
        }
    done = None
    if done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if (
            done.get("schema") != DONE_SCHEMA
            or done.get("schema_version") != SCHEMA_VERSION
            or done.get("config_sha256") != config["config_sha256"]
            or int(done.get("record_count", -1)) != int(item["sample_count"])
            or done.get("sample_uids_sha256") != embedded["sample_uids_sha256"]
            or int(done.get("valid_count", -1)) != embedded["valid_count"]
            or int(done.get("nonfinite_count", -1)) != embedded["nonfinite_count"]
            or int(done.get("missing_bbox_count", 0)) != embedded["missing_bbox_count"]
            or int(done.get("size_bytes", -1)) != path.stat().st_size
        ):
            raise RuntimeError(f"Shard completion marker differs: {done_path}")
        if deep and _sha256_file(path) != done.get("hdf5_sha256"):
            raise RuntimeError(f"Shard checksum differs: {path}")
    elif recover_done:
        digest = _sha256_file(path)
        done = {
            "schema": DONE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config["config_sha256"],
            "split": item["split"],
            "shard_index": int(item["shard_index"]),
            "record_count": int(item["sample_count"]),
            **embedded,
            "hdf5_sha256": digest,
            "size_bytes": path.stat().st_size,
        }
        _atomic_json(done_path, done)
    else:
        raise FileNotFoundError(done_path)
    return done


def _shard_reusable(
    output_dir: Path,
    item: Mapping[str, Any],
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> bool:
    path = output_dir / str(item["path"])
    for partial in path.parent.glob(f".{path.name}.partial.*"):
        partial.unlink(missing_ok=True)
    if not path.exists() and not path.with_suffix(".done.json").exists():
        return False
    try:
        _validate_shard(
            output_dir,
            item,
            config,
            deep=bool(args.deep_verify_existing),
        )
        return True
    except (OSError, ValueError, RuntimeError, FileNotFoundError):
        if not args.repair_invalid_shards:
            raise
        path.unlink(missing_ok=True)
        path.with_suffix(".done.json").unlink(missing_ok=True)
        return False


def _add_hamer_to_path() -> None:
    value = str(HAMER_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)


def _crop_camera_to_full(
    crop_camera: torch.Tensor,
    box_center: torch.Tensor,
    box_size: torch.Tensor,
    image_size: torch.Tensor,
    focal_length: torch.Tensor,
) -> torch.Tensor:
    width, height = image_size[:, 0], image_size[:, 1]
    scale = box_size * crop_camera[:, 0] + 1e-9
    return torch.stack(
        (
            2.0 * (box_center[:, 0] - width / 2.0) / scale + crop_camera[:, 1],
            2.0 * (box_center[:, 1] - height / 2.0) / scale + crop_camera[:, 2],
            2.0 * focal_length / scale,
        ),
        dim=-1,
    )


def _numpy_output(
    prediction: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    model_config: Any,
) -> dict[str, np.ndarray]:
    right = batch["right"].float()
    hand_multiplier = 2.0 * right - 1.0
    crop_camera = prediction["pred_cam"].detach().float().clone()
    crop_camera[:, 1] *= hand_multiplier
    image_wh = batch["img_size"].float()
    focal = (
        float(model_config.EXTRA.FOCAL_LENGTH)
        / float(model_config.MODEL.IMAGE_SIZE)
        * image_wh.max(dim=1).values
    )
    camera = _crop_camera_to_full(
        crop_camera,
        batch["box_center"].float(),
        batch["box_size"].float(),
        image_wh,
        focal,
    )
    vertices = prediction["pred_vertices"].detach().float().clone()
    vertices[:, :, 0] *= hand_multiplier[:, None]
    vertices_camera = vertices + camera[:, None]
    params = prediction["pred_mano_params"]
    global_orient = params["global_orient"].detach().float()
    hand_pose = params["hand_pose"].detach().float()
    betas = params["betas"].detach().float()
    finite = (
        torch.isfinite(camera).all(dim=1)
        & torch.isfinite(vertices_camera).flatten(1).all(dim=1)
        & torch.isfinite(global_orient).flatten(1).all(dim=1)
        & torch.isfinite(hand_pose).flatten(1).all(dim=1)
        & torch.isfinite(betas).all(dim=1)
    )
    depth_valid = torch.isfinite(vertices_camera).all(dim=2) & (vertices_camera[:, :, 2] > 1e-6)
    safe_depth = vertices_camera[:, :, 2].clamp_min(1e-6)
    u = focal[:, None] * vertices_camera[:, :, 0] / safe_depth + image_wh[:, 0, None] / 2.0
    v = focal[:, None] * vertices_camera[:, :, 1] / safe_depth + image_wh[:, 1, None] / 2.0
    in_frame = (
        depth_valid
        & (u >= 0.0)
        & (u < image_wh[:, 0, None])
        & (v >= 0.0)
        & (v < image_wh[:, 1, None])
    )
    status = torch.where(
        finite,
        torch.full_like(finite, STATUS_VALID, dtype=torch.uint8),
        torch.full_like(finite, STATUS_NONFINITE, dtype=torch.uint8),
    )
    valid_float = finite[:, None].float()
    camera = torch.nan_to_num(camera) * valid_float
    vertices_camera = torch.nan_to_num(vertices_camera) * valid_float[:, :, None]
    global_orient = torch.nan_to_num(global_orient) * valid_float[:, :, None, None]
    hand_pose = torch.nan_to_num(hand_pose) * valid_float[:, :, None, None]
    betas = torch.nan_to_num(betas) * valid_float
    output = {
        "image_wh": image_wh.detach().cpu().numpy().astype(np.uint16),
        "focal_length": focal.detach().cpu().numpy().astype(np.float32),
        "camera_translation": camera.cpu().numpy().astype(np.float32),
        "global_orient": global_orient.cpu().numpy().astype(np.float32),
        "hand_pose": hand_pose.cpu().numpy().astype(np.float32),
        "betas": betas.cpu().numpy().astype(np.float32),
        "status": status.cpu().numpy().astype(np.uint8),
        "positive_depth_fraction": depth_valid.float().mean(dim=1).cpu().numpy().astype(np.float16),
        "in_frame_fraction": in_frame.float().mean(dim=1).cpu().numpy().astype(np.float16),
    }
    if config["request"]["store_camera_vertices"]:
        output["vertices_camera"] = vertices_camera.cpu().numpy().astype(np.float16)
    return output


def _missing_bbox_output(
    count: int, config: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    output = {
        "image_wh": np.zeros((count, 2), dtype=np.uint16),
        "focal_length": np.zeros(count, dtype=np.float32),
        "camera_translation": np.zeros((count, 3), dtype=np.float32),
        "global_orient": np.zeros((count, 1, 3, 3), dtype=np.float32),
        "hand_pose": np.zeros((count, 15, 3, 3), dtype=np.float32),
        "betas": np.zeros((count, 10), dtype=np.float32),
        "status": np.full(count, STATUS_MISSING_SAM3_BBOX, dtype=np.uint8),
        "positive_depth_fraction": np.zeros(count, dtype=np.float16),
        "in_frame_fraction": np.zeros(count, dtype=np.float16),
    }
    if config["request"]["store_camera_vertices"]:
        output["vertices_camera"] = np.zeros((count, 778, 3), dtype=np.float16)
    return output


def _scatter_output(
    destination: dict[str, np.ndarray],
    indices: np.ndarray,
    source: Mapping[str, np.ndarray],
) -> None:
    for name, value in source.items():
        if name not in destination:
            raise KeyError(f"Unexpected HaMeR output field {name!r}")
        destination[name][indices] = value


def _load_model(config: Mapping[str, Any], device: torch.device):
    _add_hamer_to_path()
    from hamer.models import load_hamer

    model, model_config = load_hamer(
        config["request"]["checkpoint"], init_renderer=False
    )
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, model_config


def _distributed_context() -> tuple[int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("gloo")
        initialized_here = True
    return rank, world_size, local_rank, initialized_here


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _acquire_build_lock(output_dir: Path) -> Path:
    lock = output_dir / ".build.lock"
    try:
        lock.mkdir(parents=True)
    except FileExistsError:
        owner_path = lock / "owner.json"
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            owner = {}
        stale = False
        if owner.get("hostname") == socket.gethostname():
            try:
                os.kill(int(owner["pid"]), 0)
            except (ProcessLookupError, KeyError, TypeError, ValueError):
                stale = True
            except PermissionError:
                pass
        if not stale:
            raise RuntimeError(f"Another sidecar build owns {lock}: {owner}") from None
        shutil.rmtree(lock)
        lock.mkdir(parents=True)
    _atomic_json(
        lock / "owner.json",
        {"hostname": socket.gethostname(), "pid": os.getpid(), "created_unix": time.time()},
    )
    return lock


def _finalize_root(output_dir: Path, config: Mapping[str, Any], *, deep: bool = False) -> dict[str, Any]:
    entries = []
    incomplete = []
    valid_count = 0
    nonfinite_count = 0
    missing_bbox_count = 0
    size_bytes = 0
    for item in config["work_items"]:
        try:
            done = _validate_shard(output_dir, item, config, deep=deep)
        except (OSError, ValueError, RuntimeError, FileNotFoundError):
            incomplete.append(str(item["path"]))
            continue
        done_path = (output_dir / str(item["path"])).with_suffix(".done.json")
        entries.append(
            {
                **dict(item),
                "done_path": str(done_path.relative_to(output_dir)),
                "done_sha256": _sha256_file(done_path),
                "hdf5_sha256": done["hdf5_sha256"],
                "size_bytes": int(done["size_bytes"]),
                "valid_count": int(done["valid_count"]),
                "nonfinite_count": int(done["nonfinite_count"]),
                "missing_bbox_count": int(done.get("missing_bbox_count", 0)),
            }
        )
        valid_count += int(done["valid_count"])
        nonfinite_count += int(done["nonfinite_count"])
        missing_bbox_count += int(done.get("missing_bbox_count", 0))
        size_bytes += int(done["size_bytes"])
    progress = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config["config_sha256"],
        "complete": not incomplete,
        "completed_shards": len(entries),
        "shard_count": len(config["work_items"]),
        "completed_records": valid_count + nonfinite_count + missing_bbox_count,
        "sample_count": int(config["sample_count"]),
        "valid_count": valid_count,
        "nonfinite_count": nonfinite_count,
        "missing_bbox_count": missing_bbox_count,
        "size_bytes": size_bytes,
        "incomplete_examples": incomplete[:20],
    }
    _atomic_json(output_dir / PROGRESS_FILE, progress)
    if incomplete:
        (output_dir / ROOT_DONE_FILE).unlink(missing_ok=True)
        print(
            f"[hamer-sidecar] partial: {len(entries)}/{len(config['work_items'])} shards complete",
            flush=True,
        )
        return progress
    manifest = {**progress, "shards": entries}
    _atomic_json(output_dir / MANIFEST_FILE, manifest)
    _atomic_json(
        output_dir / ROOT_DONE_FILE,
        {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "config_sha256": config["config_sha256"],
            "manifest_sha256": _sha256_file(output_dir / MANIFEST_FILE),
            "sample_count": int(config["sample_count"]),
            "valid_count": valid_count,
            "nonfinite_count": nonfinite_count,
            "missing_bbox_count": missing_bbox_count,
            "size_bytes": size_bytes,
        },
    )
    print(
        f"[hamer-sidecar] COMPLETE: {len(entries)} shards, {valid_count:,} valid, "
        f"{nonfinite_count:,} nonfinite, {missing_bbox_count:,} missing bbox, "
        f"{size_bytes / 1e9:.2f} GB",
        flush=True,
    )
    return progress


def command_build(args: argparse.Namespace) -> None:
    rank, world_size, local_rank, initialized_here = _distributed_context()
    output_dir = Path(args.output_dir).expanduser().resolve()
    lock: Path | None = None
    try:
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
            lock = _acquire_build_lock(output_dir)
            config = _prepare_or_reuse(args)
        _barrier(world_size)
        if rank != 0:
            config = json.loads((output_dir / CONFIG_FILE).read_text(encoding="utf-8"))
        assigned = [
            item
            for item in config["work_items"]
            if int(item["global_work_index"]) % world_size == rank
        ]
        missing = []
        for item in assigned:
            if not _shard_reusable(output_dir, item, config, args):
                missing.append(item)
        if args.max_new_shards_per_rank is not None:
            missing = missing[: int(args.max_new_shards_per_rank)]
        print(
            f"[hamer-sidecar rank {rank}] assigned={len(assigned)} missing={len(missing)}",
            flush=True,
        )
        if missing:
            if not torch.cuda.is_available():
                raise RuntimeError("HaMeR sidecar export requires CUDA")
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if args.stagger_load_seconds > 0:
                time.sleep(local_rank * float(args.stagger_load_seconds))
            model, model_config = _load_model(config, device)
            dataset = _IndexedCropDataset(
                output_dir,
                config,
                missing,
                image_size=int(model_config.MODEL.IMAGE_SIZE),
                image_mean=model_config.MODEL.IMAGE_MEAN,
                image_std=model_config.MODEL.IMAGE_STD,
                frame_cache_size=args.frame_cache_size,
            )
            loader_kwargs: dict[str, Any] = {
                "dataset": dataset,
                "batch_sampler": _ShardBatchSampler(dataset, args.batch_size),
                "num_workers": int(args.workers),
                "pin_memory": True,
                "worker_init_fn": _worker_init,
            }
            if args.workers:
                loader_kwargs.update(
                    {
                        "persistent_workers": True,
                        "prefetch_factor": int(args.prefetch_factor),
                        "multiprocessing_context": "spawn",
                    }
                )
            loader = DataLoader(**loader_kwargs)
            current_position = None
            writer = None
            samples_done = 0
            started = time.monotonic()
            try:
                for batch_index, batch in enumerate(loader):
                    work_positions = np.asarray(batch["work_position"], dtype=np.int64)
                    if not np.all(work_positions == work_positions[0]):
                        raise RuntimeError("A loader batch crossed a shard boundary")
                    work_position = int(work_positions[0])
                    if current_position != work_position:
                        if writer is not None:
                            writer.__exit__(None, None, None)
                        current_position = work_position
                        writer = _PoseShardWriter(
                            output_dir, missing[work_position], config
                        )
                        writer.__enter__()
                    batch_count = len(batch["sample_uid"])
                    eligible = batch["bbox_source_code"].ne(BBOX_MISSING_SAM3)
                    eligible_indices = eligible.nonzero(as_tuple=False).flatten()
                    numeric = _missing_bbox_output(batch_count, config)
                    if len(eligible_indices):
                        device_batch = {
                            key: batch[key]
                            .index_select(0, eligible_indices)
                            .to(device, non_blocking=True)
                            for key in ("img", "right", "box_center", "box_size", "img_size")
                        }
                        precision = config["request"]["precision"]
                        if precision == "fp16":
                            amp = torch.autocast("cuda", dtype=torch.float16)
                        elif precision == "bf16":
                            amp = torch.autocast("cuda", dtype=torch.bfloat16)
                        else:
                            amp = nullcontext()
                        with torch.inference_mode(), amp:
                            prediction = model(device_batch)
                        selected = _numpy_output(
                            prediction, device_batch, config, model_config
                        )
                        _scatter_output(
                            numeric,
                            eligible_indices.numpy().astype(np.int64, copy=False),
                            selected,
                        )
                    writer.append(batch, numeric)
                    samples_done += batch_count
                    if (batch_index + 1) % int(args.progress_every) == 0:
                        elapsed = max(time.monotonic() - started, 1e-6)
                        print(
                            f"[hamer-sidecar rank {rank}] {samples_done:,}/{len(dataset):,} "
                            f"samples ({samples_done / elapsed:.1f}/s)",
                            flush=True,
                        )
                if writer is not None:
                    writer.__exit__(None, None, None)
                    writer = None
            except BaseException:
                if writer is not None:
                    writer.__exit__(*sys.exc_info())
                raise
            del loader, dataset, model
            torch.cuda.empty_cache()
        _barrier(world_size)
        if rank == 0:
            _finalize_root(output_dir, config, deep=False)
        _barrier(world_size)
    finally:
        if rank == 0 and lock is not None:
            shutil.rmtree(lock, ignore_errors=True)
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def command_verify(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=True)
    config = json.loads((output_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    result = _finalize_root(output_dir, config, deep=bool(args.deep))
    if not result["complete"]:
        raise RuntimeError(
            f"Sidecar is incomplete: {result['completed_shards']}/{result['shard_count']} shards"
        )


def command_inspect(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=True)
    config_path = output_dir / CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    completed = 0
    size_bytes = 0
    for item in config["work_items"]:
        path = output_dir / item["path"]
        if path.is_file() and path.with_suffix(".done.json").is_file():
            completed += 1
            size_bytes += path.stat().st_size
    bbox_summary = dict(config.get("bbox_resolution_summary", {}))
    expected_missing = int(bbox_summary.get(str(BBOX_MISSING_SAM3), 0))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_count": config["sample_count"],
                "bbox_resolution_summary": bbox_summary,
                "expected_missing_bbox_count": expected_missing,
                "expected_hamer_inference_count": int(config["sample_count"])
                - expected_missing,
                "completed_shards": completed,
                "shard_count": config["shard_count"],
                "size_bytes": size_bytes,
                "size_gb": size_bytes / 1e9,
                "complete": (output_dir / ROOT_DONE_FILE).is_file(),
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_self_test(_: argparse.Namespace) -> None:
    crop = torch.tensor([[2.0, 0.1, -0.2]])
    center = torch.tensor([[320.0, 240.0]])
    size = torch.tensor([200.0])
    image = torch.tensor([[640.0, 480.0]])
    focal = torch.tensor([5000.0])
    full = _crop_camera_to_full(crop, center, size, image, focal)
    if not torch.allclose(full, torch.tensor([[0.1, -0.2, 25.0]])):
        raise AssertionError(full)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        item = {
            "path": "shards/train/shard-000000.h5",
            "split": "train",
            "shard_index": 0,
            "source_start": 0,
            "source_stop": 2,
            "sample_count": 2,
        }
        config = {
            "config_sha256": "a" * 64,
            "request": {"compression": "lzf", "store_camera_vertices": True},
        }
        batch = {
            "sample_uid": ["a", "b"],
            "source_row": torch.tensor([0, 1]),
            "query_row": torch.tensor([0, 1]),
            "frame_row": torch.tensor([0, 0]),
            "right": torch.tensor([0.0, 1.0]),
            "bbox_source_code": torch.tensor([1, 0]),
            "bbox_xyxy": torch.ones(2, 4),
            "bbox_score": torch.ones(2),
        }
        output = {
            "image_wh": np.full((2, 2), 256, np.uint16),
            "focal_length": np.ones(2, np.float32),
            "camera_translation": np.ones((2, 3), np.float32),
            "global_orient": np.ones((2, 1, 3, 3), np.float32),
            "hand_pose": np.ones((2, 15, 3, 3), np.float32),
            "betas": np.ones((2, 10), np.float32),
            "vertices_camera": np.ones((2, 778, 3), np.float16),
            "status": np.asarray(
                [STATUS_VALID, STATUS_MISSING_SAM3_BBOX], dtype=np.uint8
            ),
            "positive_depth_fraction": np.ones(2, np.float16),
            "in_frame_fraction": np.ones(2, np.float16),
        }
        with _PoseShardWriter(root, item, config) as writer:
            writer.append(batch, output)
        done = _validate_shard(root, item, config, deep=True)
        if int(done["record_count"]) != 2 or int(done["missing_bbox_count"]) != 1:
            raise AssertionError(done)

        assets = root / "assets"
        checkpoint = assets / "checkpoints" / "model.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (assets / "model_config.yaml").write_text("model: test\n", encoding="utf-8")
        mano = assets / "mano.pkl"
        mano.write_bytes(b"mano")
        source = root / "touchanything_train.queries.jsonl"
        rows = [
            {
                "sample_uid": "missing-invalid",
                "h5_relpath": "does-not-exist.h5",
                "frame_row": 0,
                "query_row": 0,
                "is_right": 1,
                "sequence_key": "sequence",
                "frame_idx": 0,
                "target_hand": "right",
                "bbox_xyxy": None,
            },
            {
                "sample_uid": "fallback-valid",
                "h5_relpath": "also-does-not-exist.h5",
                "frame_row": 1,
                "query_row": 1,
                "is_right": 0,
                "sequence_key": "sequence",
                "frame_idx": 1,
                "target_hand": "left",
                "bbox_xyxy": [1.0, 2.0, 10.0, 20.0],
                "bbox_score": None,
            },
        ]
        source.write_text(
            "".join(_canonical_json(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        def prepare(
            policy: str,
            *,
            bbox_manifest: Optional[Path] = None,
            suffix: str = "",
        ) -> tuple[Path, dict[str, Any]]:
            fixture_root = root / f"policy-{policy}{suffix}"
            request = {
                "processed_root": str(root),
                "manifests": [{"split": "train", "path": str(source)}],
                "bbox_manifest": None if bbox_manifest is None else str(bbox_manifest),
                "checkpoint": str(checkpoint),
                "mano": str(mano),
                "shard_size": 8,
                "max_samples": None,
                "hamer_bbox_scale": 2.0,
                "precision": "fp16",
                "compression": "lzf",
                "store_camera_vertices": True,
                "unresolved_bbox_policy": policy,
            }
            return fixture_root, _prepare_new_config(
                argparse.Namespace(output_dir=str(fixture_root)), request
            )

        skip_root, skip_config = prepare("skip")
        if skip_config["bbox_resolution_summary"] != {
            "0": 2,
            "1": 0,
            "2": 0,
            "3": 0,
        }:
            raise AssertionError(skip_config["bbox_resolution_summary"])
        dataset = _IndexedCropDataset(
            skip_root,
            skip_config,
            skip_config["work_items"],
            image_size=256,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            frame_cache_size=0,
        )
        if any(
            int(dataset[index]["bbox_source_code"]) != BBOX_MISSING_SAM3
            or bool(dataset[index]["img"].count_nonzero())
            for index in range(2)
        ):
            raise AssertionError("Skipped bbox rows performed image I/O or changed status")

        invalid_overlay = root / "invalid-sam3.jsonl"
        invalid_overlay.write_text(
            "".join(
                _canonical_json(
                    {
                        "split": "train",
                        "sequence_key": "sequence",
                        "frame_idx": index,
                        "target_hand": "right" if index == 0 else "left",
                        "bbox_xyxy": None,
                    }
                )
                + "\n"
                for index in range(2)
            ),
            encoding="utf-8",
        )
        _, invalid_overlay_config = prepare(
            "skip", bbox_manifest=invalid_overlay, suffix="-invalid-overlay"
        )
        if invalid_overlay_config["bbox_resolution_summary"]["0"] != 2:
            raise AssertionError(invalid_overlay_config["bbox_resolution_summary"])

        _, fallback_config = prepare("manifest_fallback")
        if fallback_config["bbox_resolution_summary"] != {
            "0": 1,
            "1": 0,
            "2": 0,
            "3": 1,
        }:
            raise AssertionError(fallback_config["bbox_resolution_summary"])

        try:
            prepare("error")
        except RuntimeError as exc:
            if "no matching SAM3 bbox" not in str(exc):
                raise
        else:
            raise AssertionError("Strict unresolved-bbox policy did not fail")
    print("HAMER_SIDECAR_SELF_TEST_OK", flush=True)


def _add_common_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--processed-root")
    parser.add_argument("--manifest", action="append", default=[])
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--bbox-manifest", default=str(DEFAULT_BBOX_MANIFEST))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--mano", default=str(DEFAULT_MANO))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--frame-cache-size", type=int, default=8)
    parser.add_argument("--hamer-bbox-scale", type=float, default=2.0)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--compression", choices=("none", "lzf", "gzip1"), default="lzf")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-shards-per-rank", type=int)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--stagger-load-seconds", type=float, default=1.0)
    parser.add_argument("--repair-invalid-shards", action="store_true")
    parser.add_argument("--deep-verify-existing", action="store_true")
    parser.add_argument(
        "--unresolved-bbox-policy",
        choices=("skip", "error", "manifest_fallback"),
        default="skip",
        help=(
            "How to handle rows without a verifiable SAM3 box. The default keeps "
            "their source-row position, writes status=missing_sam3_bbox, and skips "
            "HaMeR inference."
        ),
    )
    parser.add_argument(
        "--allow-unverified-bbox-fallback",
        action="store_true",
        help="Deprecated alias for --unresolved-bbox-policy=manifest_fallback.",
    )
    parser.add_argument(
        "--store-camera-vertices",
        action=argparse.BooleanOptionalAction,
        default=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Prepare and export all missing shards")
    _add_common_build_arguments(build)
    build.set_defaults(func=command_build)
    for name, function in (("verify", command_verify), ("inspect", command_inspect)):
        child = subparsers.add_parser(name)
        child.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
        if name == "verify":
            child.add_argument("--deep", action="store_true")
        child.set_defaults(func=function)
    self_test = subparsers.add_parser("self-test")
    self_test.set_defaults(func=command_self_test)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "shard_size") and args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if hasattr(args, "workers") and args.workers < 0:
        raise ValueError("--workers cannot be negative")
    args.func(args)


if __name__ == "__main__":
    main()
