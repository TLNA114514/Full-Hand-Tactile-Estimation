#!/usr/bin/env python3
"""Convert processed tactile sample folders into sequence-level HDF5 files.

This tool only accepts an already extracted/processed root. It never reads,
writes, deletes, or repairs the original Hugging Face/raw dataset. Legacy
processed sample folders are retained unless the explicit verified-prune flag
is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import h5py

try:
    import orjson
except ImportError:
    orjson = None

try:
    from process_lifecycle import initialize_worker_parent_death_signal
except ImportError:
    try:
        from .process_lifecycle import initialize_worker_parent_death_signal
    except ImportError:
        initialize_worker_parent_death_signal = None

try:
    from hdf5_storage import (
        AtomicJsonlWriter,
        SCHEMA_VERSION,
        TACTILE_DIM,
        SequenceHDF5Writer,
        canonical_json,
        finite_optional_float,
        jsonl_bytes,
        manifest_rows_from_hdf5,
        sequence_manifest_row,
        sha256_file,
        verify_sequence_hdf5,
        write_json_atomic,
    )
except ImportError:
    from .hdf5_storage import (
        AtomicJsonlWriter,
        SCHEMA_VERSION,
        TACTILE_DIM,
        SequenceHDF5Writer,
        canonical_json,
        finite_optional_float,
        jsonl_bytes,
        manifest_rows_from_hdf5,
        sequence_manifest_row,
        sha256_file,
        verify_sequence_hdf5,
        write_json_atomic,
    )


DATASETS = ("opentouch", "touchanything", "egotactile")
HAND_ALIASES = ("left", "right")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
LEGACY_SAMPLE_FILES = {
    "opentouch": frozenset(("image.jpg", "meta.json")),
    "touchanything": frozenset(("chest.jpg", "left.jpg", "right.jpg", "meta.json")),
    "egotactile": frozenset(("image.jpg", "meta.json")),
}


@dataclass(frozen=True)
class SampleDescriptor:
    sample_dir: str
    sample_relpath: str
    meta_sha256: str
    image_path: str
    image_relpath: str
    image_size: int
    image_mtime_ns: int
    dataset: str
    split: str
    sequence_key: str
    sequence_parts: tuple[str, ...]
    frame_idx: int
    query_aliases: tuple[str, ...]
    trainable_query_aliases: tuple[str, ...]
    meta_size: int


@dataclass(frozen=True)
class SequenceTask:
    dataset: str
    dataset_name: str
    split: str
    sequence_key: str
    sequence_parts: tuple[str, ...]
    processed_root: str
    h5_path: str
    source_fingerprint: str
    descriptors: tuple[SampleDescriptor, ...]
    pressure_compression: str
    mode: str
    deep_verify: bool
    palm_vertex_mask: np.ndarray
    mesh_sha256: str
    palm_faces_sha256: str


def bounded_ordered_map(executor, function, iterable, max_pending: int):
    """Yield executor results in input order without submitting the full corpus."""

    iterator = iter(iterable)
    pending = {}
    buffered = {}
    submitted = 0
    next_yield = 0

    def submit_one() -> bool:
        nonlocal submitted
        try:
            item = next(iterator)
        except StopIteration:
            return False
        future = executor.submit(function, item)
        pending[future] = submitted
        submitted += 1
        return True

    for _ in range(max(1, int(max_pending))):
        if not submit_one():
            break
    while pending:
        completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
        for future in completed:
            index = pending.pop(future)
            buffered[index] = future.result()
        while next_yield in buffered:
            yield buffered.pop(next_yield)
            next_yield += 1
        while len(pending) + len(buffered) < max(1, int(max_pending)):
            if not submit_one():
                break


def load_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    value = orjson.loads(payload) if orjson is not None else json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: meta.json root must be an object")
    return value, digest


def canonical_dataset_name(dataset: str) -> str:
    return {
        "opentouch": "OpenTouch",
        "touchanything": "TouchAnything",
        "egotactile": "EgoTactile",
    }[dataset]


def metadata_dataset_name(meta: dict[str, Any], requested: str) -> str:
    raw = str(meta.get("dataset", "")).strip().lower().replace("_", "")
    aliases = {
        "opentouch": "opentouch",
        "touchanything": "touchanything",
        "egotouch": "touchanything",
        "egotactile": "egotactile",
    }
    if not raw:
        # Historical OpenTouch metadata did not include a dataset field.
        return requested
    return aliases.get(raw, raw)


def valid_bbox(value: Any) -> bool:
    if value is None or value == "null":
        return False
    try:
        bbox = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return (
        bbox.shape == (4,)
        and bool(np.isfinite(bbox).all())
        and bool(np.all(bbox[2:4] - bbox[0:2] > 1.0))
    )


def effective_pressure(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        pressure = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if pressure.shape != (TACTILE_DIM,) or not bool(np.isfinite(pressure).all()):
        return None
    # This exactly matches the effective pressure tensor created by dataset.py.
    return np.clip(pressure, 0.0, 1.0).astype(np.float32, copy=False)


def pressure_for_query(
    meta: dict[str, Any], dataset: str, query_alias: str
) -> tuple[np.ndarray | None, str]:
    if dataset == "touchanything":
        hand_meta = meta.get("hands", {}).get(query_alias, {})
        key = hand_meta.get("gaussian_pressure_key") or "gaussian_pressure"
        return effective_pressure(hand_meta.get("gaussian_pressure")), str(key)
    side = query_alias
    key = f"{side}_pressure_continuous_subdiv"
    original = meta.get("original_hdf5_data", {})
    pressure = original.get(key)
    if pressure is not None:
        return effective_pressure(pressure), key
    return effective_pressure(meta.get("gaussian_pressure")), "gaussian_pressure"


def bbox_for_query(
    meta: dict[str, Any], dataset: str, query_alias: str
) -> tuple[Any, float, Any]:
    if dataset == "touchanything":
        hand_meta = meta.get("hands", {}).get(query_alias, {})
        return (
            hand_meta.get("bbox_chest"),
            float(hand_meta.get("bbox_score", 0.0) or 0.0),
            hand_meta.get("bbox_source"),
        )
    return (
        meta.get("bbox"),
        float(meta.get("bbox_score", 0.0) or 0.0),
        meta.get("bbox_source"),
    )


def sequence_identity(
    meta: dict[str, Any], dataset: str
) -> tuple[str, tuple[str, ...]]:
    if dataset == "touchanything":
        parts = (
            str(meta.get("scene", "")).strip(),
            str(meta.get("task", "")).strip(),
            str(meta.get("clip", meta.get("rel_clip", ""))).strip(),
        )
    elif dataset == "opentouch":
        parts = (
            str(meta.get("scene", "")).strip(),
            str(meta.get("demo", "")).strip(),
        )
    else:
        rel_seq = str(meta.get("rel_seq", "")).strip().replace("\\", "/")
        parts = tuple(part for part in Path(rel_seq).parts if part not in ("", "."))
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(
            f"Cannot derive a safe {dataset} sequence path from processed metadata"
        )
    if Path(parts[0]).is_absolute():
        raise ValueError("Sequence identity must be relative")
    return "/".join(parts), tuple(parts)


def image_name_for_meta(meta: dict[str, Any], dataset: str) -> str:
    if dataset == "touchanything":
        return str(meta.get("views", {}).get("chest", "chest.jpg"))
    return str(meta.get("image", "image.jpg"))


def query_aliases_for_meta(
    meta: dict[str, Any], dataset: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    archive_aliases: list[str] = []
    trainable_aliases: list[str] = []
    if dataset == "touchanything":
        candidates = HAND_ALIASES
    else:
        is_right = int(meta.get("is_right", 1))
        candidates = ("right" if is_right else "left",)
    for alias in candidates:
        bbox, _, _ = bbox_for_query(meta, dataset, alias)
        pressure, _ = pressure_for_query(meta, dataset, alias)
        if pressure is not None:
            archive_aliases.append(alias)
            if valid_bbox(bbox):
                trainable_aliases.append(alias)
    return tuple(archive_aliases), tuple(trainable_aliases)


def _scan_sample(args) -> dict[str, Any]:
    sample_dir_text, dataset, split, processed_root_text = args
    sample_dir = Path(sample_dir_text)
    processed_root = Path(processed_root_text)
    meta_path = sample_dir / "meta.json"
    try:
        meta, meta_sha256 = load_json_bytes(meta_path)
        found_dataset = metadata_dataset_name(meta, dataset)
        if found_dataset != dataset:
            raise ValueError(
                f"metadata dataset={found_dataset!r}, requested={dataset!r}"
            )
        meta_split = str(meta.get("split", split))
        if meta_split and meta_split != split:
            raise ValueError(f"metadata split={meta_split!r}, directory split={split!r}")
        sequence_key, sequence_parts = sequence_identity(meta, dataset)
        frame_idx = int(meta.get("frame_idx", 0))
        query_aliases, trainable_query_aliases = query_aliases_for_meta(
            meta, dataset
        )
        image_name = image_name_for_meta(meta, dataset)
        image_path = sample_dir / image_name
        if image_name not in ("chest.jpg", "image.jpg"):
            raise ValueError(f"unexpected processed image name {image_name!r}")
        if not image_path.is_file() or image_path.is_symlink():
            raise FileNotFoundError(f"processed JPEG is missing or a symlink: {image_path}")
        stat = image_path.stat()
        descriptor = SampleDescriptor(
            sample_dir=str(sample_dir),
            sample_relpath=sample_dir.relative_to(processed_root).as_posix(),
            meta_sha256=meta_sha256,
            image_path=str(image_path),
            image_relpath=image_path.relative_to(processed_root).as_posix(),
            image_size=int(stat.st_size),
            image_mtime_ns=int(stat.st_mtime_ns),
            dataset=dataset,
            split=split,
            sequence_key=sequence_key,
            sequence_parts=sequence_parts,
            frame_idx=frame_idx,
            query_aliases=query_aliases,
            trainable_query_aliases=trainable_query_aliases,
            meta_size=int(meta_path.stat().st_size),
        )
        return {
            "status": "valid",
            "descriptor": descriptor,
            "has_trainable_query": bool(trainable_query_aliases),
        }
    except Exception as exc:
        return {
            "status": "error",
            "sample_dir": str(sample_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }


def safe_component(value: str) -> str:
    if SAFE_COMPONENT.fullmatch(value) and value not in (".", ".."):
        return value
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "sequence"
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned}__{suffix}"


def output_h5_path(
    processed_root: Path,
    split: str,
    sequence_parts: tuple[str, ...],
) -> Path:
    safe_parts = [safe_component(part) for part in sequence_parts]
    directory = processed_root / split
    for part in safe_parts[:-1]:
        directory = directory / part
    return directory / f"{safe_parts[-1]}.h5"


def sequence_source_fingerprint(descriptors: Iterable[SampleDescriptor]) -> str:
    digest = hashlib.sha256()
    for descriptor in sorted(
        descriptors, key=lambda item: (item.frame_idx, item.sample_relpath)
    ):
        record = {
            "sample_relpath": descriptor.sample_relpath,
            "meta_sha256": descriptor.meta_sha256,
            "image_relpath": descriptor.image_relpath,
            "image_size": descriptor.image_size,
            "image_mtime_ns": descriptor.image_mtime_ns,
            "frame_idx": descriptor.frame_idx,
            "query_aliases": descriptor.query_aliases,
            "trainable_query_aliases": descriptor.trainable_query_aliases,
        }
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_jpeg_hw(payload: bytes) -> tuple[int, int]:
    """Read JPEG dimensions without decoding or changing encoded bytes."""

    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise ValueError("input is not a JPEG stream")
    offset = 2
    while offset + 4 <= len(payload):
        while offset < len(payload) and payload[offset] != 0xFF:
            offset += 1
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(payload):
            break
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(payload):
            break
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if segment_length < 7:
                break
            height = int.from_bytes(payload[offset + 3 : offset + 5], "big")
            width = int.from_bytes(payload[offset + 5 : offset + 7], "big")
            if height <= 0 or width <= 0:
                break
            return height, width
        offset += segment_length
    raise ValueError("JPEG dimensions could not be parsed")


def load_query(
    descriptor: SampleDescriptor,
    query_alias: str,
    expected_dataset: str,
    *,
    require_bbox: bool = True,
) -> dict[str, Any]:
    meta_path = Path(descriptor.sample_dir) / "meta.json"
    meta, digest = load_json_bytes(meta_path)
    if digest != descriptor.meta_sha256:
        raise RuntimeError(f"{meta_path} changed after source discovery")
    pressure, pressure_source_key = pressure_for_query(meta, expected_dataset, query_alias)
    bbox, bbox_score, bbox_source = bbox_for_query(
        meta, expected_dataset, query_alias
    )
    if pressure is None or (require_bbox and not valid_bbox(bbox)):
        raise RuntimeError(
            f"{meta_path}: query {query_alias} became invalid after discovery"
        )
    if expected_dataset == "touchanything":
        keypoints = np.zeros((21, 3), dtype=np.float32)
        keypoints_valid = np.zeros(21, dtype=np.uint8)
    else:
        keypoints = np.asarray(
            meta.get("keypoints_3d_cam", np.zeros((21, 3))), dtype=np.float32
        )
        keypoints_valid = np.asarray(
            meta.get("valid_mask", np.zeros(21)), dtype=np.uint8
        )
        if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
            raise RuntimeError(f"{meta_path}: invalid keypoints_3d_cam")
        if keypoints_valid.shape != (21,):
            raise RuntimeError(f"{meta_path}: invalid keypoint valid_mask")
    return {
        "meta": meta,
        "pressure": pressure,
        "pressure_source_key": pressure_source_key,
        "bbox": (
            np.asarray(bbox, dtype=np.float32)
            if valid_bbox(bbox)
            else np.full(4, np.nan, dtype=np.float32)
        ),
        "bbox_score": bbox_score,
        "bbox_source": bbox_source,
        "keypoints_3d_cam": keypoints,
        "keypoints_valid": keypoints_valid,
    }


def frame_metadata(
    descriptor: SampleDescriptor, dataset: str
) -> tuple[int | None, float | None, int]:
    meta, digest = load_json_bytes(Path(descriptor.sample_dir) / "meta.json")
    if digest != descriptor.meta_sha256:
        raise RuntimeError(f"{descriptor.sample_dir}/meta.json changed after discovery")
    if dataset == "touchanything":
        source_frame_idx = meta.get("jq_pressure_frame_index")
        timestamp = finite_optional_float(meta.get("timestamp"))
        return (
            int(source_frame_idx) if source_frame_idx is not None else None,
            timestamp,
            1 if timestamp is not None else 0,
        )
    timestamp = finite_optional_float(meta.get("timestamp"))
    return descriptor.frame_idx, timestamp, 1 if timestamp is not None else 0


def query_uid(task: SequenceTask, frame_idx: int, query_alias: str) -> str:
    return (
        f"{task.dataset_name}/{task.split}/{task.sequence_key}/"
        f"{frame_idx:08d}/{query_alias}"
    )


def load_existing_train_query_overlay(h5_path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    if not h5_path.is_file():
        return {}
    overlays = {}
    with h5py.File(h5_path, "r") as handle:
        queries = handle["queries"]
        targets = handle["targets"]
        trainable = (
            np.asarray(queries["is_trainable"][:], dtype=bool)
            if "is_trainable" in queries
            else np.ones(int(handle.attrs["query_count"]), dtype=bool)
        )
        for row in np.flatnonzero(trainable):
            alias = queries["query_alias"][row]
            if isinstance(alias, bytes):
                alias = alias.decode("utf-8")
            bbox_source_text = queries["bbox_source_json"][row]
            if isinstance(bbox_source_text, bytes):
                bbox_source_text = bbox_source_text.decode("utf-8")
            overlays[(int(queries["frame_idx"][row]), str(alias))] = {
                "bbox": np.asarray(queries["bbox_xyxy"][row], dtype=np.float32),
                "bbox_score": float(queries["bbox_score"][row]),
                "bbox_source": (
                    json.loads(bbox_source_text) if bbox_source_text else None
                ),
                "pressure": np.asarray(targets["pressure"][row], dtype=np.float32),
                "keypoints_3d_cam": np.asarray(
                    queries["keypoints_3d_cam"][row], dtype=np.float32
                ),
                "keypoints_valid": np.asarray(
                    queries["keypoints_valid"][row], dtype=np.uint8
                ),
                "pressure_source_key": (
                    queries["pressure_source_key"][row].decode("utf-8")
                    if isinstance(queries["pressure_source_key"][row], bytes)
                    else str(queries["pressure_source_key"][row])
                ),
            }
    return overlays


def convert_sequence(task: SequenceTask) -> dict[str, Any]:
    h5_path = Path(task.h5_path)
    processed_root = Path(task.processed_root)
    existing_train_overlay = (
        load_existing_train_query_overlay(h5_path)
        if h5_path.exists() and task.mode == "overwrite"
        else {}
    )
    if h5_path.exists() and task.mode in ("resume", "verify"):
        verify_sequence_hdf5(
            h5_path,
            expected_source_fingerprint=task.source_fingerprint,
            expected_pressure_compression=task.pressure_compression,
            deep=task.deep_verify,
        )
        rows = manifest_rows_from_hdf5(h5_path, processed_root)
        return {
            "status": "verified" if task.mode == "verify" else "resumed",
            "query_rows": rows,
            "sequence_row": sequence_manifest_row(h5_path, processed_root),
        }
    if task.mode == "verify":
        raise FileNotFoundError(f"Expected converted sequence is missing: {h5_path}")
    if h5_path.exists() and task.mode == "create":
        raise FileExistsError(
            f"{h5_path} already exists; use --resume, --overwrite, or --verify-only"
        )

    descriptors = sorted(
        task.descriptors, key=lambda item: (item.frame_idx, item.sample_relpath)
    )
    by_frame: dict[int, list[SampleDescriptor]] = defaultdict(list)
    for descriptor in descriptors:
        by_frame[descriptor.frame_idx].append(descriptor)
    frame_indices = sorted(by_frame)
    expected_frame_indices = list(range(frame_indices[-1] + 1))
    if frame_indices != expected_frame_indices:
        missing = sorted(set(expected_frame_indices) - set(frame_indices))
        raise RuntimeError(
            f"{task.sequence_key}: processed sequence is incomplete; "
            f"observed={len(frame_indices)} frames over 0..{frame_indices[-1]}, "
            f"missing={missing[:20]}. Regenerate the processed sequence before "
            "writing the complete archive."
        )
    frame_row_by_idx = {frame_idx: row for row, frame_idx in enumerate(frame_indices)}

    query_specs: list[tuple[SampleDescriptor, str]] = []
    seen_queries: set[tuple[int, str]] = set()
    for descriptor in descriptors:
        for alias in descriptor.query_aliases:
            key = (descriptor.frame_idx, alias)
            if key in seen_queries:
                raise RuntimeError(
                    f"{task.sequence_key}: duplicate processed query frame={key[0]}, hand={key[1]}"
                )
            seen_queries.add(key)
            query_specs.append((descriptor, alias))
    query_specs.sort(key=lambda item: (item[0].frame_idx, item[1]))
    metadata_specs = sorted(
        descriptors, key=lambda item: (item.frame_idx, item.sample_relpath)
    )

    extra_attrs = {
        "mesh_sha256": task.mesh_sha256,
        "palm_faces_sha256": task.palm_faces_sha256,
        "target_construction": "legacy_dataset_float32_clip_0_1",
        "image_storage": "original_jpeg_bytes_concat_offsets",
        "archive_scope": "all_processed_sequence_frames",
        "archive_frame_index_contract": "contiguous_zero_based",
        "archive_sequence_complete": True,
        "metadata_storage": "exact_meta_json_bytes_concat_offsets",
    }
    with SequenceHDF5Writer(
        h5_path,
        dataset=task.dataset_name,
        split=task.split,
        sequence_key=task.sequence_key,
        frame_count=len(frame_indices),
        query_count=len(query_specs),
        jpeg_total_bytes=sum(
            int(by_frame[frame_idx][0].image_size) for frame_idx in frame_indices
        ),
        metadata_record_count=len(metadata_specs),
        metadata_total_bytes=sum(item.meta_size for item in metadata_specs),
        source_fingerprint=task.source_fingerprint,
        pressure_compression=task.pressure_compression,
        extra_attrs=extra_attrs,
    ) as writer:
        for frame_row, frame_idx in enumerate(frame_indices):
            candidates = by_frame[frame_idx]
            selected_bytes = Path(candidates[0].image_path).read_bytes()
            selected_hash = hashlib.sha256(selected_bytes).hexdigest()
            for duplicate in candidates[1:]:
                duplicate_bytes = Path(duplicate.image_path).read_bytes()
                if (
                    len(duplicate_bytes) != len(selected_bytes)
                    or hashlib.sha256(duplicate_bytes).hexdigest() != selected_hash
                ):
                    raise RuntimeError(
                        f"{task.sequence_key} frame {frame_idx}: left/right processed "
                        "samples do not share identical RGB JPEG bytes"
                    )
            source_frame_idx, timestamp, timestamp_kind = frame_metadata(
                candidates[0], task.dataset
            )
            writer.write_frame(
                frame_row,
                frame_idx=frame_idx,
                source_frame_idx=source_frame_idx,
                timestamp=timestamp,
                timestamp_kind=timestamp_kind,
                image_hw=parse_jpeg_hw(selected_bytes),
                jpeg_bytes=selected_bytes,
            )

        for metadata_row, descriptor in enumerate(metadata_specs):
            meta_path = Path(descriptor.sample_dir) / "meta.json"
            meta_bytes = meta_path.read_bytes()
            if (
                len(meta_bytes) != descriptor.meta_size
                or hashlib.sha256(meta_bytes).hexdigest() != descriptor.meta_sha256
            ):
                raise RuntimeError(f"{meta_path} changed after source discovery")
            query_alias = (
                descriptor.query_aliases[0]
                if len(descriptor.query_aliases) == 1
                else ""
            )
            writer.write_metadata_record(
                metadata_row,
                frame_row=frame_row_by_idx[descriptor.frame_idx],
                query_alias=query_alias,
                source_sample_relpath=descriptor.sample_relpath,
                meta_json_bytes=meta_bytes,
            )

        for row, (descriptor, alias) in enumerate(query_specs):
            overlay = existing_train_overlay.get((descriptor.frame_idx, alias))
            is_trainable = (
                alias in descriptor.trainable_query_aliases or overlay is not None
            )
            loaded = load_query(
                descriptor,
                alias,
                task.dataset,
                require_bbox=is_trainable and overlay is None,
            )
            if overlay is not None:
                loaded.update(overlay)
            pressure = loaded["pressure"]
            palm_pressure = pressure[task.palm_vertex_mask]
            uid = query_uid(task, descriptor.frame_idx, alias)
            writer.write_query(
                row,
                frame_row=frame_row_by_idx[descriptor.frame_idx],
                frame_idx=descriptor.frame_idx,
                hand_code=1 if alias == "right" else 0,
                is_right=1 if alias == "right" else 0,
                bbox_xyxy=loaded["bbox"],
                bbox_score=loaded["bbox_score"],
                keypoints_3d_cam=loaded["keypoints_3d_cam"],
                keypoints_valid=loaded["keypoints_valid"],
                query_uid=uid,
                query_alias=alias,
                source_sample_relpath=descriptor.sample_relpath,
                bbox_source=loaded["bbox_source"],
                pressure_source_key=loaded["pressure_source_key"],
                pressure=pressure,
                max_pressure=float(palm_pressure.max()),
                volume=float(palm_pressure.sum()),
                active_count=int(np.count_nonzero(palm_pressure >= 0.05)),
                is_trainable=is_trainable,
            )

    verify_sequence_hdf5(
        h5_path,
        expected_source_fingerprint=task.source_fingerprint,
        expected_pressure_compression=task.pressure_compression,
        deep=task.deep_verify,
    )
    rows = manifest_rows_from_hdf5(h5_path, processed_root)
    return {
        "status": "converted" if task.mode != "overwrite" else "overwritten",
        "query_rows": rows,
        "sequence_row": sequence_manifest_row(h5_path, processed_root),
    }


def suspicious_raw_or_hf_path(path: Path) -> bool:
    normalized = path.resolve().as_posix().lower()
    markers = (
        "/.cache/huggingface/",
        "/huggingface/hub/",
        "/snapshots/",
        "/datasets--",
    )
    return any(marker in normalized for marker in markers)


def split_sample_dirs(processed_root: Path, split: str) -> list[Path]:
    split_root = processed_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Processed split directory does not exist: {split_root}")
    samples: list[Path] = []
    with os.scandir(split_root) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                candidate = Path(entry.path)
                if (candidate / "meta.json").is_file():
                    samples.append(candidate)
    samples.sort()
    return samples


def discover_splits(processed_root: Path) -> list[str]:
    splits = set()
    with os.scandir(processed_root) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False) or entry.name == "manifests":
                continue
            try:
                with os.scandir(entry.path) as children:
                    if any(
                        child.is_dir(follow_symlinks=False)
                        and (Path(child.path) / "meta.json").is_file()
                        for child in children
                    ):
                        splits.add(entry.name)
            except OSError:
                continue
    manifest_dir = processed_root / "manifests"
    if manifest_dir.is_dir():
        for name in os.listdir(manifest_dir):
            for dataset in DATASETS:
                prefix = f"{dataset}_"
                suffix = ".sequences.jsonl"
                if name.startswith(prefix) and name.endswith(suffix):
                    splits.add(name[len(prefix) : -len(suffix)])
    return sorted(splits)


def scan_split(
    processed_root: Path,
    dataset: str,
    split: str,
    workers: int,
    backend: str,
) -> tuple[list[SampleDescriptor], dict[str, Any], list[Path]]:
    sample_dirs = split_sample_dirs(processed_root, split)
    if not sample_dirs:
        raise RuntimeError(
            f"{processed_root / split} contains no direct processed sample folders "
            "with meta.json; refusing to treat it as a processed root"
        )
    print(
        f"[{split}] Scanning {len(sample_dirs)} processed sample folders "
        f"with {workers} {backend} worker(s)...",
        flush=True,
    )
    executor_cls = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    scan_args = (
        (str(path), dataset, split, str(processed_root)) for path in sample_dirs
    )
    descriptors: list[SampleDescriptor] = []
    errors: list[dict[str, Any]] = []
    no_query = 0
    started = time.monotonic()
    executor_kwargs = {"max_workers": workers}
    if backend == "process" and initialize_worker_parent_death_signal is not None:
        executor_kwargs["initializer"] = initialize_worker_parent_death_signal
    with executor_cls(**executor_kwargs) as executor:
        results = bounded_ordered_map(
            executor, _scan_sample, scan_args, max_pending=workers * 4
        )
        for completed, result in enumerate(results, start=1):
            if result["status"] == "valid":
                descriptors.append(result["descriptor"])
                if not result["has_trainable_query"]:
                    no_query += 1
            else:
                errors.append(result)
            if completed % 10000 == 0 or completed == len(sample_dirs):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[{split}] Scanned {completed}/{len(sample_dirs)} "
                    f"({completed / elapsed:.0f} dirs/s), valid={len(descriptors)}, "
                    f"no-query={no_query}, errors={len(errors)}",
                    flush=True,
                )
    if errors:
        preview = "\n".join(
            f"  - {row['sample_dir']}: {row['error']}" for row in errors[:10]
        )
        raise RuntimeError(
            f"[{split}] {len(errors)} processed sample folders could not be read. "
            f"First errors:\n{preview}"
        )
    if not descriptors:
        raise RuntimeError(f"[{split}] No readable processed frames were found")
    if not any(descriptor.trainable_query_aliases for descriptor in descriptors):
        raise RuntimeError(f"[{split}] No valid training queries were found")
    return (
        descriptors,
        {
            "sample_dir_count": len(sample_dirs),
            "archived_sample_dir_count": len(descriptors),
            "trainable_sample_dir_count": sum(
                bool(descriptor.trainable_query_aliases)
                for descriptor in descriptors
            ),
            "no_valid_query_count": no_query,
            "error_count": len(errors),
        },
        sample_dirs,
    )


def load_palm_assets() -> tuple[np.ndarray, str, str]:
    workspace = Path(__file__).resolve().parent.parent
    obj_path = workspace / "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj"
    faces_path = (
        workspace
        / "opentouch/preprocess/scratch/auto_calibrated_palm_subdiv_faces.json"
    )
    if not obj_path.is_file() or not faces_path.is_file():
        raise FileNotFoundError(
            "Current loader palm assets are missing; cannot reproduce sampler fields"
        )
    vertex_count = 0
    with obj_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertex_count += 1
    if vertex_count != TACTILE_DIM:
        raise ValueError(
            f"Mesh has {vertex_count} vertices, expected converter dimension {TACTILE_DIM}"
        )
    faces = json.loads(faces_path.read_text(encoding="utf-8"))
    mask = np.zeros(TACTILE_DIM, dtype=bool)
    for triplet in faces["group_negative"]["face_triplets"]:
        for vertex_id in triplet:
            if 0 <= int(vertex_id) < TACTILE_DIM:
                mask[int(vertex_id)] = True
    if not mask.any():
        raise ValueError("Palm mask is empty")
    return mask, sha256_file(obj_path), sha256_file(faces_path)


def build_tasks(
    descriptors: list[SampleDescriptor],
    *,
    processed_root: Path,
    dataset: str,
    split: str,
    pressure_compression: str,
    mode: str,
    deep_verify: bool,
    palm_vertex_mask: np.ndarray,
    mesh_sha256: str,
    palm_faces_sha256: str,
) -> list[SequenceTask]:
    grouped: dict[str, list[SampleDescriptor]] = defaultdict(list)
    parts_by_key: dict[str, tuple[str, ...]] = {}
    for descriptor in descriptors:
        grouped[descriptor.sequence_key].append(descriptor)
        previous = parts_by_key.setdefault(
            descriptor.sequence_key, descriptor.sequence_parts
        )
        if previous != descriptor.sequence_parts:
            raise RuntimeError(
                f"Sequence key collision for {descriptor.sequence_key}: "
                f"{previous} versus {descriptor.sequence_parts}"
            )
    tasks = []
    output_paths: dict[Path, str] = {}
    for sequence_key in sorted(grouped):
        sequence_descriptors = tuple(
            sorted(
                grouped[sequence_key],
                key=lambda item: (item.frame_idx, item.sample_relpath),
            )
        )
        h5_path = output_h5_path(
            processed_root, split, parts_by_key[sequence_key]
        ).resolve()
        collision = output_paths.setdefault(h5_path, sequence_key)
        if collision != sequence_key:
            raise RuntimeError(
                f"Output path collision: {collision!r} and {sequence_key!r} -> {h5_path}"
            )
        tasks.append(
            SequenceTask(
                dataset=dataset,
                dataset_name=canonical_dataset_name(dataset),
                split=split,
                sequence_key=sequence_key,
                sequence_parts=parts_by_key[sequence_key],
                processed_root=str(processed_root),
                h5_path=str(h5_path),
                source_fingerprint=sequence_source_fingerprint(
                    sequence_descriptors
                ),
                descriptors=sequence_descriptors,
                pressure_compression=pressure_compression,
                mode=mode,
                deep_verify=deep_verify,
                palm_vertex_mask=palm_vertex_mask,
                mesh_sha256=mesh_sha256,
                palm_faces_sha256=palm_faces_sha256,
            )
        )
    return tasks


def _manifest_paths(
    processed_root: Path, dataset: str, split: str
) -> tuple[Path, Path, Path]:
    directory = processed_root / "manifests"
    prefix = f"{dataset}_{split}"
    return (
        directory / f"{prefix}.queries.jsonl",
        directory / f"{prefix}.sequences.jsonl",
        directory / f"{prefix}.summary.json",
    )


def _manifest_digest(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sha256_file(path)


def process_split(
    tasks: list[SequenceTask],
    *,
    workers: int,
    processed_root: Path,
    dataset: str,
    split: str,
    verify_only: bool,
    scan_summary: dict[str, Any],
) -> dict[str, Any]:
    query_manifest, sequence_manifest, summary_path = _manifest_paths(
        processed_root, dataset, split
    )
    expected_query_digest = hashlib.sha256()
    expected_sequence_digest = hashlib.sha256()
    counts = defaultdict(int)
    query_count = 0
    started = time.monotonic()

    query_writer = None if verify_only else AtomicJsonlWriter(query_manifest)
    sequence_writer = None if verify_only else AtomicJsonlWriter(sequence_manifest)
    if query_writer is not None:
        query_writer.__enter__()
        sequence_writer.__enter__()
    try:
        executor_kwargs = {"max_workers": workers}
        if initialize_worker_parent_death_signal is not None:
            executor_kwargs["initializer"] = initialize_worker_parent_death_signal
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            for completed, result in enumerate(
                bounded_ordered_map(
                    executor,
                    convert_sequence,
                    tasks,
                    max_pending=workers * 2,
                ),
                start=1,
            ):
                counts[result["status"]] += 1
                for row in result["query_rows"]:
                    payload = jsonl_bytes(row)
                    expected_query_digest.update(payload)
                    if query_writer is not None:
                        query_writer.write(row)
                    query_count += 1
                sequence_row = result["sequence_row"]
                expected_sequence_digest.update(jsonl_bytes(sequence_row))
                if sequence_writer is not None:
                    sequence_writer.write(sequence_row)
                if completed % 10 == 0 or completed == len(tasks):
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"[{split}] Sequences {completed}/{len(tasks)} "
                        f"({completed / elapsed:.2f} seq/s), queries={query_count}, "
                        f"status={dict(counts)}",
                        flush=True,
                    )
    except BaseException:
        if query_writer is not None:
            query_writer.__exit__(*sys.exc_info())
            sequence_writer.__exit__(*sys.exc_info())
        raise
    else:
        if query_writer is not None:
            query_writer.__exit__(None, None, None)
            sequence_writer.__exit__(None, None, None)

    expected_query_sha = expected_query_digest.hexdigest()
    expected_sequence_sha = expected_sequence_digest.hexdigest()
    if verify_only:
        actual_query_sha = _manifest_digest(query_manifest)
        actual_sequence_sha = _manifest_digest(sequence_manifest)
        if actual_query_sha != expected_query_sha:
            raise RuntimeError(
                f"{query_manifest}: manifest does not match verified HDF5 content"
            )
        if actual_sequence_sha != expected_sequence_sha:
            raise RuntimeError(
                f"{sequence_manifest}: manifest does not match verified HDF5 content"
            )
    else:
        actual_query_sha = _manifest_digest(query_manifest)
        actual_sequence_sha = _manifest_digest(sequence_manifest)

    summary = {
        "schema": "tactile_hdf5_conversion_summary_v1",
        "schema_version": SCHEMA_VERSION,
        "dataset": canonical_dataset_name(dataset),
        "dataset_key": dataset,
        "split": split,
        "processed_root": str(processed_root),
        "sequence_count": len(tasks),
        "query_count": query_count,
        "status_counts": dict(counts),
        "scan": scan_summary,
        "query_manifest": query_manifest.relative_to(processed_root).as_posix(),
        "query_manifest_sha256": actual_query_sha,
        "sequence_manifest": sequence_manifest.relative_to(processed_root).as_posix(),
        "sequence_manifest_sha256": actual_sequence_sha,
        "verify_only": bool(verify_only),
        "elapsed_seconds": time.monotonic() - started,
    }
    if not verify_only:
        write_json_atomic(summary_path, summary)
    print(
        f"[{split}] Complete: {len(tasks)} sequences, {query_count} queries, "
        f"query manifest={query_manifest}",
        flush=True,
    )
    return summary


def _inspect_legacy_sample_for_prune(args) -> dict[str, Any]:
    sample_dir_text, processed_root_text, split, dataset = args
    sample_dir = Path(sample_dir_text)
    processed_root = Path(processed_root_text)
    split_root = (processed_root / split).resolve()
    try:
        if sample_dir.is_symlink():
            raise RuntimeError("sample directory is a symlink")
        resolved = sample_dir.resolve(strict=True)
        if resolved.parent != split_root:
            raise RuntimeError(
                f"sample directory is not an immediate child of {split_root}"
            )
        allowed = LEGACY_SAMPLE_FILES[dataset]
        entries = list(os.scandir(resolved))
        names = {entry.name for entry in entries}
        unexpected = sorted(names - allowed)
        if unexpected:
            raise RuntimeError(f"unexpected entries prevent deletion: {unexpected}")
        required = {"meta.json", "chest.jpg" if dataset == "touchanything" else "image.jpg"}
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"required processed files are missing: {missing}")
        total_bytes = 0
        for entry in entries:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise RuntimeError(f"entry is not a regular file: {entry.name}")
            total_bytes += int(entry.stat(follow_symlinks=False).st_size)
        return {
            "status": "ready",
            "sample_dir": str(resolved),
            "file_count": len(entries),
            "size_bytes": total_bytes,
        }
    except Exception as exc:
        return {
            "status": "error",
            "sample_dir": str(sample_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _delete_verified_legacy_sample(args) -> dict[str, Any]:
    sample_dir_text, processed_root_text, split, dataset = args
    sample_dir = Path(sample_dir_text)
    try:
        rechecked = _inspect_legacy_sample_for_prune(
            (sample_dir_text, processed_root_text, split, dataset)
        )
        if rechecked["status"] != "ready":
            raise RuntimeError(
                f"folder changed after preflight: {rechecked.get('error', 'unknown error')}"
            )
        deleted_bytes = 0
        for entry in os.scandir(sample_dir):
            path = Path(entry.path)
            deleted_bytes += int(entry.stat(follow_symlinks=False).st_size)
            path.unlink()
        sample_dir.rmdir()
        return {
            "status": "deleted",
            "sample_dir": str(sample_dir),
            "deleted_bytes": deleted_bytes,
        }
    except Exception as exc:
        return {
            "status": "error",
            "sample_dir": str(sample_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }


def prune_verified_legacy_samples(
    descriptors: list[SampleDescriptor],
    legacy_sample_dirs: list[Path],
    tasks: list[SequenceTask],
    *,
    processed_root: Path,
    dataset: str,
    split: str,
    workers: int,
) -> dict[str, Any]:
    """Delete only verified, allowlisted processed sample folders."""

    print(
        f"[{split}] Deep-verifying {len(tasks)} HDF5 sequence files before "
        "legacy-folder deletion...",
        flush=True,
    )
    for completed, task in enumerate(tasks, start=1):
        verify_sequence_hdf5(
            task.h5_path,
            expected_source_fingerprint=task.source_fingerprint,
            expected_pressure_compression=task.pressure_compression,
            deep=True,
        )
        if completed % 10 == 0 or completed == len(tasks):
            print(
                f"[{split}] Deep-verified {completed}/{len(tasks)} HDF5 files.",
                flush=True,
            )

    represented_dirs = {str(Path(descriptor.sample_dir).resolve()) for descriptor in descriptors}
    unique_dirs = sorted({str(path.resolve()) for path in legacy_sample_dirs})
    inspect_args = (
        (path, str(processed_root), split, dataset) for path in unique_dirs
    )
    inspections = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspections.extend(
            bounded_ordered_map(
                executor,
                _inspect_legacy_sample_for_prune,
                inspect_args,
                max_pending=workers * 4,
            )
        )
    failures = [row for row in inspections if row["status"] != "ready"]
    if failures:
        preview = "\n".join(
            f"  - {row['sample_dir']}: {row['error']}" for row in failures[:10]
        )
        raise RuntimeError(
            f"[{split}] Refusing legacy-folder deletion because "
            f"{len(failures)} folders failed preflight:\n{preview}"
        )

    print(
        f"[{split}] Deleting {len(inspections)} verified processed sample folders...",
        flush=True,
    )
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(
            bounded_ordered_map(
                executor,
                _delete_verified_legacy_sample,
                (
                    (
                        row["sample_dir"],
                        str(processed_root),
                        split,
                        dataset,
                    )
                    for row in inspections
                ),
                max_pending=workers * 4,
            ),
            start=1,
        ):
            results.append(result)
            if completed % 10000 == 0 or completed == len(inspections):
                print(
                    f"[{split}] Deleted {completed}/{len(inspections)} folders.",
                    flush=True,
                )
    delete_failures = [row for row in results if row["status"] != "deleted"]
    report = {
        "schema": "tactile_legacy_processed_prune_v1",
        "dataset": canonical_dataset_name(dataset),
        "split": split,
        "processed_root": str(processed_root),
        "deep_verified_sequence_count": len(tasks),
        "candidate_folder_count": len(inspections),
        "represented_folder_count": len(represented_dirs),
        "excluded_from_training_folder_count": len(
            set(unique_dirs) - represented_dirs
        ),
        "candidate_bytes": sum(row["size_bytes"] for row in inspections),
        "deleted_folder_count": sum(row["status"] == "deleted" for row in results),
        "deleted_bytes": sum(row.get("deleted_bytes", 0) for row in results),
        "failures": delete_failures[:100],
    }
    report_path = (
        processed_root
        / "manifests"
        / f"{dataset}_{split}.legacy_prune.summary.json"
    )
    write_json_atomic(report_path, report)
    if delete_failures:
        raise RuntimeError(
            f"[{split}] {len(delete_failures)} verified folders failed deletion; "
            f"inspect {report_path}"
        )
    print(
        f"[{split}] Legacy processed folders removed; report={report_path}",
        flush=True,
    )
    return report


def verify_published_archive(
    *,
    processed_root: Path,
    dataset: str,
    split: str,
    require_complete_schema: bool = True,
) -> list[dict[str, Any]]:
    query_manifest, sequence_manifest, summary_path = _manifest_paths(
        processed_root, dataset, split
    )
    for path in (query_manifest, sequence_manifest, summary_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"[{split}] Cannot clean remaining folders without verified "
                f"conversion metadata: {path}"
            )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset_key") != dataset or summary.get("split") != split:
        raise RuntimeError(
            f"[{split}] Conversion summary dataset/split does not match cleanup request"
        )
    if _manifest_digest(query_manifest) != summary.get("query_manifest_sha256"):
        raise RuntimeError(f"[{split}] Query manifest checksum does not match summary")
    if _manifest_digest(sequence_manifest) != summary.get("sequence_manifest_sha256"):
        raise RuntimeError(f"[{split}] Sequence manifest checksum does not match summary")

    sequence_rows = []
    with sequence_manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("dataset") != canonical_dataset_name(dataset)
                or row.get("split") != split
            ):
                raise RuntimeError(
                    f"{sequence_manifest}:{line_number}: dataset/split mismatch"
                )
            if (
                require_complete_schema
                and row.get("hdf5_schema_version") != SCHEMA_VERSION
            ):
                raise RuntimeError(
                    f"{sequence_manifest}:{line_number}: archive schema is "
                    f"{row.get('hdf5_schema_version', 'legacy-v1')!r}; regenerate "
                    f"with --overwrite to obtain complete schema {SCHEMA_VERSION}"
                )
            h5_path = (processed_root / str(row["h5_relpath"])).resolve()
            try:
                h5_path.relative_to(processed_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"{sequence_manifest}:{line_number}: HDF5 path escapes processed root"
                ) from exc
            verify_sequence_hdf5(
                h5_path,
                expected_source_fingerprint=row.get("source_fingerprint"),
                deep=True,
            )
            sequence_rows.append(row)
            if len(sequence_rows) % 10 == 0:
                print(
                    f"[{split}] Deep-verified {len(sequence_rows)} sequence files "
                    "from the published manifest.",
                    flush=True,
                )
    if not sequence_rows:
        raise RuntimeError(f"[{split}] Sequence manifest is empty")
    print(
        f"[{split}] Published archive verified: {len(sequence_rows)} "
        f"{'complete ' if require_complete_schema else ''}sequence files.",
        flush=True,
    )
    return sequence_rows


def prune_remaining_legacy_samples(
    *,
    processed_root: Path,
    dataset: str,
    split: str,
    workers: int,
) -> dict[str, Any]:
    """Remove legacy folders left behind by an older manifest-only prune."""

    sequence_rows = verify_published_archive(
        processed_root=processed_root,
        dataset=dataset,
        split=split,
        require_complete_schema=False,
    )

    legacy_sample_dirs = split_sample_dirs(processed_root, split)
    inspections = []
    inspect_args = (
        (str(path), str(processed_root), split, dataset)
        for path in legacy_sample_dirs
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspections.extend(
            bounded_ordered_map(
                executor,
                _inspect_legacy_sample_for_prune,
                inspect_args,
                max_pending=workers * 4,
            )
        )
    failures = [row for row in inspections if row["status"] != "ready"]
    if failures:
        preview = "\n".join(
            f"  - {row['sample_dir']}: {row['error']}" for row in failures[:10]
        )
        raise RuntimeError(
            f"[{split}] Refusing remaining-folder cleanup because "
            f"{len(failures)} folders failed preflight:\n{preview}"
        )

    results = []
    delete_args = (
        (row["sample_dir"], str(processed_root), split, dataset)
        for row in inspections
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, result in enumerate(
            bounded_ordered_map(
                executor,
                _delete_verified_legacy_sample,
                delete_args,
                max_pending=workers * 4,
            ),
            start=1,
        ):
            results.append(result)
            if completed % 10000 == 0 or completed == len(inspections):
                print(
                    f"[{split}] Removed remaining legacy folders "
                    f"{completed}/{len(inspections)}.",
                    flush=True,
                )
    delete_failures = [row for row in results if row["status"] != "deleted"]
    report = {
        "schema": "tactile_remaining_legacy_prune_v1",
        "dataset": canonical_dataset_name(dataset),
        "split": split,
        "processed_root": str(processed_root),
        "deep_verified_sequence_count": len(sequence_rows),
        "remaining_folder_count": len(inspections),
        "deleted_folder_count": sum(row["status"] == "deleted" for row in results),
        "deleted_bytes": sum(row.get("deleted_bytes", 0) for row in results),
        "failures": delete_failures[:100],
    }
    report_path = (
        processed_root
        / "manifests"
        / f"{dataset}_{split}.remaining_legacy_prune.summary.json"
    )
    write_json_atomic(report_path, report)
    if delete_failures:
        raise RuntimeError(
            f"[{split}] {len(delete_failures)} remaining folders failed deletion; "
            f"inspect {report_path}"
        )
    print(
        f"[{split}] Remaining legacy cleanup complete: "
        f"{report['deleted_folder_count']} folders; report={report_path}",
        flush=True,
    )
    return report


def parse_splits(value: str, processed_root: Path) -> list[str]:
    if value.strip().lower() == "auto":
        splits = discover_splits(processed_root)
    else:
        splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("No processed splits were selected or discovered")
    if len(splits) != len(set(splits)):
        raise ValueError(f"Duplicate splits are not allowed: {splits}")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one processed tactile dataset at a time from per-sample folders "
            "to one lossless HDF5 per visual sequence."
        )
    )
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--processed-root",
        required=True,
        help="Existing extracted/processed root. Raw/Hugging Face roots are rejected.",
    )
    parser.add_argument(
        "--splits",
        default="auto",
        help="Comma-separated processed splits, or auto (default).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Parallel sequence writer processes.",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=0,
        help="Metadata scan workers; 0 uses --workers.",
    )
    parser.add_argument(
        "--scan-backend",
        choices=("thread", "process"),
        default="thread",
        help="Threads reduce IPC; processes can parse very large JSON faster.",
    )
    parser.add_argument(
        "--compression",
        choices=("lzf", "gzip1"),
        default="lzf",
        help="Lossless pressure compression. JPEG bytes are never recompressed.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Verify and reuse matching HDF5 files; convert missing sequences.",
    )
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing sequence HDF5 files.",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Deep-verify HDF5 files and manifests without writing anything.",
    )
    mode.add_argument(
        "--cleanup-remaining-legacy-folders",
        action="store_true",
        help=(
            "Verify already-published manifests/HDF5 and delete processed sample "
            "folders left by an older query-only cleanup. No conversion is run."
        ),
    )
    mode.add_argument(
        "--verify-published-archive",
        action="store_true",
        help=(
            "Deep-verify published manifests and sequence HDF5 files without "
            "requiring or modifying legacy processed folders."
        ),
    )
    parser.add_argument(
        "--deep-verify-after-write",
        action="store_true",
        help="Rehash every JPEG and scan every pressure row after conversion.",
    )
    parser.add_argument(
        "--delete-legacy-folders-after-verify",
        action="store_true",
        help=(
            "After manifests are published, deep-verify every sequence HDF5 and "
            "delete only allowlisted per-sample folders under the processed split. "
            "Raw/Hugging Face data is never touched."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    scan_workers = args.scan_workers or args.workers
    if scan_workers <= 0:
        raise ValueError("--scan-workers must be positive")
    processed_root = Path(args.processed_root).expanduser().resolve()
    if suspicious_raw_or_hf_path(processed_root):
        raise RuntimeError(
            f"Refusing Hugging Face/cache/raw-looking path: {processed_root}"
        )
    if not processed_root.is_dir():
        raise FileNotFoundError(processed_root)

    splits = parse_splits(args.splits, processed_root)
    if args.verify_published_archive:
        for split in splits:
            verify_published_archive(
                processed_root=processed_root,
                dataset=args.dataset,
                split=split,
            )
        return
    if args.cleanup_remaining_legacy_folders:
        for split in splits:
            prune_remaining_legacy_samples(
                processed_root=processed_root,
                dataset=args.dataset,
                split=split,
                workers=scan_workers,
            )
        return

    palm_mask, mesh_sha256, palm_faces_sha256 = load_palm_assets()
    if args.verify_only:
        mode = "verify"
    elif args.overwrite:
        mode = "overwrite"
    elif args.resume:
        mode = "resume"
    else:
        mode = "create"
    if args.verify_only and args.delete_legacy_folders_after_verify:
        raise ValueError(
            "--verify-only and --delete-legacy-folders-after-verify are mutually exclusive"
        )

    print("Sequence HDF5 conversion", flush=True)
    print(f"  dataset:       {args.dataset}", flush=True)
    print(f"  processed root:{processed_root}", flush=True)
    print(f"  splits:        {','.join(splits)}", flush=True)
    print(f"  mode:          {mode}", flush=True)
    print(f"  workers:       {args.workers} sequence / {scan_workers} scan", flush=True)
    print(
        "  source policy: processed files only; raw/Hugging Face data is untouched",
        flush=True,
    )

    all_summaries = []
    for split in splits:
        descriptors, scan_summary, legacy_sample_dirs = scan_split(
            processed_root,
            args.dataset,
            split,
            scan_workers,
            args.scan_backend,
        )
        tasks = build_tasks(
            descriptors,
            processed_root=processed_root,
            dataset=args.dataset,
            split=split,
            pressure_compression=args.compression,
            mode=mode,
            deep_verify=args.verify_only or args.deep_verify_after_write,
            palm_vertex_mask=palm_mask,
            mesh_sha256=mesh_sha256,
            palm_faces_sha256=palm_faces_sha256,
        )
        print(
            f"[{split}] Grouped {len(descriptors)} source folders into "
            f"{len(tasks)} sequence file(s).",
            flush=True,
        )
        summary = process_split(
            tasks,
            workers=args.workers,
            processed_root=processed_root,
            dataset=args.dataset,
            split=split,
            verify_only=args.verify_only,
            scan_summary=scan_summary,
        )
        if args.delete_legacy_folders_after_verify:
            summary["legacy_prune"] = prune_verified_legacy_samples(
                descriptors,
                legacy_sample_dirs,
                tasks,
                processed_root=processed_root,
                dataset=args.dataset,
                split=split,
                workers=scan_workers,
            )
        all_summaries.append(summary)
    total_sequences = sum(row["sequence_count"] for row in all_summaries)
    total_queries = sum(row["query_count"] for row in all_summaries)
    print(
        f"Finished {args.dataset}: {total_sequences} sequence(s), "
        f"{total_queries} query row(s).",
        flush=True,
    )


if __name__ == "__main__":
    main()
