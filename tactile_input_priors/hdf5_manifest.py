#!/usr/bin/env python3
"""Self-contained HDF5 readers and manifest utilities for depth sidecars.

This module intentionally has no dependency on ``hamer_tactile_ft``.  It owns
the small subset of the sequence-HDF5 contract needed to discover manifests,
decode source RGB frames, verify finalized containers, and hash/publish
manifest files atomically.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np


SEQUENCE_SCHEMA_NAME = "tactile_sequence_hdf5"
SUPPORTED_SEQUENCE_SCHEMA_VERSIONS = ("1.0.0", "2.0.0")
MANIFEST_SCHEMA = "tactile_query_manifest_v1"
SEQUENCE_MANIFEST_SCHEMA = "tactile_sequence_manifest_v1"
SUPPORTED_PRESSURE_COMPRESSIONS = ("lzf", "gzip1")
DEPTH_QUERY_SCHEMA = "tactile_depth_query_record_v1"


def _decoded_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _jsonl_bytes(row: Mapping[str, Any]) -> bytes:
    return (canonical_json(dict(row)) + "\n").encode("utf-8")


def sha256_file(
    path: os.PathLike[str] | str,
    chunk_size: int = 4 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: os.PathLike[str] | str) -> None:
    descriptor = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_sibling(path: Path, label: str = "tmp") -> Path:
    return path.with_name(f".{path.name}.{label}.{os.getpid()}.{uuid.uuid4().hex}")


class AtomicJsonlWriter:
    """Write deterministic JSONL and publish it with an atomic rename."""

    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)
        self.temp_path = _temporary_sibling(self.path)
        self._handle = None
        self.count = 0
        self.digest = hashlib.sha256()

    def __enter__(self) -> "AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temp_path.open("wb")
        return self

    def write(self, row: Mapping[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("AtomicJsonlWriter is not open")
        payload = _jsonl_bytes(row)
        self._handle.write(payload)
        self.digest.update(payload)
        self.count += 1

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if self._handle is not None:
                if exc_type is None:
                    self._handle.flush()
                    os.fsync(self._handle.fileno())
                self._handle.close()
        finally:
            self._handle = None
        if exc_type is not None:
            self.temp_path.unlink(missing_ok=True)
            return False
        os.replace(self.temp_path, self.path)
        _fsync_directory(self.path.parent)
        return False


def write_json_atomic(path: os.PathLike[str] | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_sibling(path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def open_readonly(
    path: os.PathLike[str] | str,
    *,
    raw_chunk_cache_bytes: int = 256 * 1024,
    raw_chunk_cache_slots: int = 521,
) -> h5py.File:
    """Open a finalized sequence HDF5 without shared-filesystem write locks."""

    base_kwargs = {
        "mode": "r",
        "libver": "latest",
        "rdcc_nbytes": max(int(raw_chunk_cache_bytes), 64 * 1024),
        "rdcc_nslots": max(int(raw_chunk_cache_slots), 17),
        "rdcc_w0": 0.0,
    }
    attempts = (
        {**base_kwargs, "swmr": True, "locking": False},
        {**base_kwargs, "locking": False},
        {**base_kwargs, "swmr": True},
        base_kwargs,
    )
    errors = []
    for kwargs in attempts:
        try:
            return h5py.File(path, **kwargs)
        except (OSError, TypeError, ValueError) as exc:
            errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
    raise OSError(f"Could not open immutable HDF5 container {path}: " + " | ".join(errors))


class HDF5ImageReader:
    """Bounded LRU reader for source JPEGs and pressure rows."""

    def __init__(self, max_handles: int = 4):
        self.max_handles = max(1, int(max_handles))
        self.handles: OrderedDict[str, h5py.File] = OrderedDict()

    def close(self) -> None:
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self.handles.clear()

    def _handle(self, path: os.PathLike[str] | str) -> h5py.File:
        key = os.fspath(path)
        handle = self.handles.pop(key, None)
        if handle is None or not handle.id.valid:
            handle = open_readonly(key, raw_chunk_cache_bytes=1024 * 1024)
        self.handles[key] = handle
        while len(self.handles) > self.max_handles:
            _, old = self.handles.popitem(last=False)
            old.close()
        return handle

    def read_bgr(self, row: Mapping[str, Any]) -> np.ndarray:
        import cv2

        handle = self._handle(row["h5_path"])
        data = handle["images/rgb/jpeg_data"]
        offsets = handle["images/rgb/jpeg_offsets"]
        frame_row = int(row["frame_row"])
        start, end = np.asarray(offsets[frame_row : frame_row + 2], dtype=np.uint64)
        encoded = np.asarray(data[int(start) : int(end)], dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(
                f"Could not decode {row['sample_uid']} from {row['h5_path']}"
            )
        return image

    def read_pressure(self, row: Mapping[str, Any]) -> np.ndarray:
        handle = self._handle(row["h5_path"])
        query_row = int(row["query_row"])
        for name in (
            "targets/pressure",
            "queries/pressure/gaussian_subdiv",
            "tactile/pressure",
        ):
            if name not in handle:
                continue
            value = np.asarray(handle[name][query_row], dtype=np.float32)
            if value.ndim == 1 and np.isfinite(value).all():
                return np.clip(value, 0.0, 1.0)
        raise KeyError(
            f"No pressure target for query row {query_row} in {row['h5_path']}"
        )


def canonical_query_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a query manifest row to the fields required by the builder."""

    bbox = record.get("bbox_xyxy", record.get("bbox"))
    required = (
        "sample_uid",
        "dataset",
        "split",
        "sequence_key",
        "frame_idx",
        "query_alias",
        "is_right",
        "h5_path",
        "frame_row",
        "query_row",
        "max_pressure",
        "target_volume",
        "target_active_count",
    )
    missing = [name for name in required if record.get(name) is None]
    if bbox is None:
        missing.append("bbox_xyxy")
    if missing:
        raise ValueError(
            f"Normalized depth query record is missing {sorted(set(missing))}"
        )
    return {
        "schema": DEPTH_QUERY_SCHEMA,
        "sample_uid": str(record["sample_uid"]),
        "dataset": str(record["dataset"]),
        "split": str(record["split"]),
        "sequence_key": str(record["sequence_key"]),
        "frame_idx": int(record["frame_idx"]),
        "query_alias": str(record["query_alias"]),
        "is_right": int(record["is_right"]),
        "bbox_xyxy": [float(value) for value in bbox],
        "bbox_score": float(record.get("bbox_score", float("nan"))),
        "h5_path": str(record["h5_path"]),
        "frame_row": int(record["frame_row"]),
        "query_row": int(record["query_row"]),
        "max_pressure": float(record["max_pressure"]),
        "target_volume": float(record["target_volume"]),
        "target_active_count": int(record["target_active_count"]),
    }


def manifest_rows_from_hdf5(
    h5_path: os.PathLike[str] | str,
    processed_root: os.PathLike[str] | str,
) -> list[dict[str, Any]]:
    h5_path = Path(h5_path).resolve()
    processed_root = Path(processed_root).resolve()
    h5_relpath = h5_path.relative_to(processed_root).as_posix()
    rows: list[dict[str, Any]] = []
    with open_readonly(h5_path) as handle:
        dataset = _decoded_string(handle.attrs["dataset"])
        split = _decoded_string(handle.attrs["split"])
        sequence_key = _decoded_string(handle.attrs["sequence_key"])
        schema_version = _decoded_string(handle.attrs["schema_version"])
        frames = handle["frames"]
        queries = handle["queries"]
        targets = handle["targets"]
        trainable = (
            np.asarray(queries["is_trainable"][:], dtype=bool)
            if "is_trainable" in queries
            else np.ones(int(handle.attrs["query_count"]), dtype=bool)
        )
        for query_row in range(int(handle.attrs["query_count"])):
            if not trainable[query_row]:
                continue
            bbox_source_text = _decoded_string(queries["bbox_source_json"][query_row])
            bbox_source = json.loads(bbox_source_text) if bbox_source_text else None
            frame_row = int(queries["frame_row"][query_row])
            source_frame_idx = int(frames["source_frame_idx"][frame_row])
            timestamp = float(frames["timestamp"][frame_row])
            rows.append(
                {
                    "schema": MANIFEST_SCHEMA,
                    "hdf5_schema_version": schema_version,
                    "sample_uid": _decoded_string(queries["query_uid"][query_row]),
                    "dataset": dataset,
                    "split": split,
                    "sequence_key": sequence_key,
                    "h5_relpath": h5_relpath,
                    "frame_row": frame_row,
                    "query_row": query_row,
                    "frame_idx": int(queries["frame_idx"][query_row]),
                    "query_alias": _decoded_string(queries["query_alias"][query_row]),
                    "hand": _decoded_string(queries["query_alias"][query_row]),
                    "is_right": int(queries["is_right"][query_row]),
                    "bbox_xyxy": [
                        float(value) for value in queries["bbox_xyxy"][query_row]
                    ],
                    "bbox_score": float(queries["bbox_score"][query_row]),
                    "bbox_source": bbox_source,
                    "source_sample_relpath": _decoded_string(
                        queries["source_sample_relpath"][query_row]
                    ),
                    "pressure_source_key": _decoded_string(
                        queries["pressure_source_key"][query_row]
                    ),
                    "source_frame_idx": None if source_frame_idx < 0 else source_frame_idx,
                    "timestamp": timestamp if math.isfinite(timestamp) else None,
                    "max_pressure": float(targets["max_pressure"][query_row]),
                    "target_volume": float(targets["volume"][query_row]),
                    "target_active_count": int(targets["active_count"][query_row]),
                }
            )
    return rows


def sequence_manifest_row(
    h5_path: os.PathLike[str] | str,
    processed_root: os.PathLike[str] | str,
) -> dict[str, Any]:
    h5_path = Path(h5_path).resolve()
    processed_root = Path(processed_root).resolve()
    with open_readonly(h5_path) as handle:
        return {
            "schema": SEQUENCE_MANIFEST_SCHEMA,
            "hdf5_schema_version": _decoded_string(handle.attrs["schema_version"]),
            "dataset": _decoded_string(handle.attrs["dataset"]),
            "split": _decoded_string(handle.attrs["split"]),
            "sequence_key": _decoded_string(handle.attrs["sequence_key"]),
            "h5_relpath": h5_path.relative_to(processed_root).as_posix(),
            "frame_count": int(handle.attrs["frame_count"]),
            "query_count": int(handle.attrs["query_count"]),
            "archive_query_count": int(handle.attrs["query_count"]),
            "train_query_count": int(
                handle.attrs.get("train_query_count", handle.attrs["query_count"])
            ),
            "metadata_record_count": int(handle.attrs.get("metadata_record_count", 0)),
            "pressure_dim": int(handle.attrs["pressure_dim"]),
            "source_fingerprint": _decoded_string(handle.attrs["source_fingerprint"]),
            "file_size_bytes": int(h5_path.stat().st_size),
        }


def verify_sequence_hdf5(
    h5_path: os.PathLike[str] | str,
    *,
    expected_source_fingerprint: str | None = None,
    expected_pressure_compression: str | None = None,
    deep: bool = False,
) -> dict[str, Any]:
    """Validate sequence schema and optionally every JPEG and pressure row."""

    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open_readonly(path) as handle:
        if _decoded_string(handle.attrs.get("schema_name", "")) != SEQUENCE_SCHEMA_NAME:
            raise ValueError(f"{path}: unsupported schema_name")
        schema_version = _decoded_string(handle.attrs.get("schema_version", ""))
        if schema_version not in SUPPORTED_SEQUENCE_SCHEMA_VERSIONS:
            raise ValueError(f"{path}: unsupported schema_version {schema_version!r}")
        frame_count = int(handle.attrs["frame_count"])
        query_count = int(handle.attrs["query_count"])
        pressure_dim = int(handle.attrs["pressure_dim"])
        source_fingerprint = _decoded_string(handle.attrs["source_fingerprint"])
        pressure_compression = _decoded_string(
            handle.attrs.get("pressure_compression", "")
        )
        if (
            expected_source_fingerprint is not None
            and source_fingerprint != expected_source_fingerprint
        ):
            raise ValueError(
                f"{path}: source fingerprint changed "
                f"({source_fingerprint} != {expected_source_fingerprint})"
            )
        if (
            expected_pressure_compression is not None
            and pressure_compression != expected_pressure_compression
        ):
            raise ValueError(
                f"{path}: pressure compression changed "
                f"({pressure_compression} != {expected_pressure_compression})"
            )
        required = (
            "frames/frame_idx",
            "frames/source_frame_idx",
            "frames/timestamp",
            "frames/image_hw",
            "images/rgb/jpeg_data",
            "images/rgb/jpeg_offsets",
            "images/rgb/jpeg_sha256",
            "queries/frame_row",
            "queries/bbox_xyxy",
            "queries/keypoints_3d_cam",
            "queries/keypoints_valid",
            "queries/query_uid",
            "targets/pressure",
            "targets/pressure_sha256",
            "targets/max_pressure",
            "targets/volume",
            "targets/active_count",
        )
        if schema_version == "2.0.0":
            required += (
                "queries/is_trainable",
                "archive/meta_json/data",
                "archive/meta_json/offsets",
                "archive/meta_json/sha256",
                "archive/meta_json/frame_row",
                "archive/meta_json/query_alias",
                "archive/meta_json/source_sample_relpath",
            )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path}: missing HDF5 datasets: {missing}")

        offsets = np.asarray(handle["images/rgb/jpeg_offsets"][:], dtype=np.uint64)
        jpeg_size = int(handle["images/rgb/jpeg_data"].shape[0])
        if offsets.shape != (frame_count + 1,):
            raise ValueError(f"{path}: invalid JPEG offsets shape {offsets.shape}")
        if offsets[0] != 0 or offsets[-1] != jpeg_size or np.any(np.diff(offsets) <= 0):
            raise ValueError(f"{path}: JPEG offsets are not strictly increasing")
        frame_indices = np.asarray(handle["frames/frame_idx"][:], dtype=np.int64)
        if schema_version == "2.0.0" and not np.array_equal(
            frame_indices, np.arange(frame_count, dtype=np.int64)
        ):
            raise ValueError(
                f"{path}: v2 archive frame indices must be contiguous and zero-based"
            )

        pressure = handle["targets/pressure"]
        if pressure.shape != (query_count, pressure_dim):
            raise ValueError(f"{path}: invalid pressure shape {pressure.shape}")
        if pressure.dtype != np.dtype("<f4"):
            raise ValueError(f"{path}: pressure dtype must be float32, got {pressure.dtype}")
        if pressure.chunks != (1, pressure_dim):
            raise ValueError(
                f"{path}: pressure chunks must be (1,{pressure_dim}), got {pressure.chunks}"
            )
        if pressure_compression == "lzf" and pressure.compression != "lzf":
            raise ValueError(f"{path}: pressure dataset is not LZF-compressed")
        if pressure_compression == "gzip1" and not (
            pressure.compression == "gzip" and pressure.compression_opts == 1
        ):
            raise ValueError(f"{path}: pressure dataset is not gzip level 1")
        if pressure_compression not in SUPPORTED_PRESSURE_COMPRESSIONS:
            raise ValueError(
                f"{path}: unsupported stored pressure compression "
                f"{pressure_compression!r}"
            )

        frame_rows = np.asarray(handle["queries/frame_row"][:], dtype=np.int64)
        if frame_rows.shape != (query_count,) or np.any(frame_rows < 0) or np.any(
            frame_rows >= frame_count
        ):
            raise ValueError(f"{path}: query frame rows are out of range")
        bbox = np.asarray(handle["queries/bbox_xyxy"][:], dtype=np.float32)
        if bbox.shape != (query_count, 4):
            raise ValueError(f"{path}: invalid query bbox array")
        trainable = (
            np.asarray(handle["queries/is_trainable"][:], dtype=bool)
            if "queries/is_trainable" in handle
            else np.ones(query_count, dtype=bool)
        )
        if np.any(~np.isfinite(bbox[trainable])):
            raise ValueError(f"{path}: trainable query contains a non-finite bbox")
        keypoints = np.asarray(handle["queries/keypoints_3d_cam"][:], dtype=np.float32)
        if keypoints.shape != (query_count, 21, 3) or not np.isfinite(keypoints).all():
            raise ValueError(f"{path}: invalid query keypoint array")

        if deep:
            stored_hashes = handle["images/rgb/jpeg_sha256"]
            jpeg_data = handle["images/rgb/jpeg_data"]
            for row in range(frame_count):
                payload = bytes(jpeg_data[int(offsets[row]) : int(offsets[row + 1])])
                digest = hashlib.sha256(payload).hexdigest()
                if digest != _decoded_string(stored_hashes[row]):
                    raise ValueError(f"{path}: JPEG checksum mismatch at frame row {row}")
            for row in range(query_count):
                values = np.asarray(pressure[row], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: non-finite pressure at query row {row}")
                if np.any(values < 0.0) or np.any(values > 1.0):
                    raise ValueError(f"{path}: unclipped pressure at query row {row}")
                expected_digest = _decoded_string(handle["targets/pressure_sha256"][row])
                actual_digest = hashlib.sha256(
                    np.ascontiguousarray(values, dtype="<f4").tobytes()
                ).hexdigest()
                if actual_digest != expected_digest:
                    raise ValueError(
                        f"{path}: pressure checksum mismatch at query row {row}"
                    )
            if schema_version == "2.0.0":
                archive = handle["archive/meta_json"]
                metadata_count = int(handle.attrs.get("metadata_record_count", 0))
                metadata_offsets = np.asarray(archive["offsets"][:], dtype=np.uint64)
                metadata_size = int(archive["data"].shape[0])
                if (
                    metadata_offsets.shape != (metadata_count + 1,)
                    or metadata_offsets[0] != 0
                    or metadata_offsets[-1] != metadata_size
                    or np.any(np.diff(metadata_offsets) <= 0)
                ):
                    raise ValueError(f"{path}: invalid archived metadata offsets")
                for row in range(metadata_count):
                    payload = bytes(
                        archive["data"][
                            int(metadata_offsets[row]) : int(metadata_offsets[row + 1])
                        ]
                    )
                    if hashlib.sha256(payload).hexdigest() != _decoded_string(
                        archive["sha256"][row]
                    ):
                        raise ValueError(
                            f"{path}: archived meta.json checksum mismatch at row {row}"
                        )
                    json.loads(payload)

        return {
            "path": str(path),
            "dataset": _decoded_string(handle.attrs["dataset"]),
            "split": _decoded_string(handle.attrs["split"]),
            "sequence_key": _decoded_string(handle.attrs["sequence_key"]),
            "frame_count": frame_count,
            "query_count": query_count,
            "train_query_count": int(trainable.sum()),
            "metadata_record_count": int(handle.attrs.get("metadata_record_count", 0)),
            "pressure_dim": pressure_dim,
            "source_fingerprint": source_fingerprint,
            "deep": bool(deep),
        }


def jsonl_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_jsonl_bytes(row))
    return digest.hexdigest()
