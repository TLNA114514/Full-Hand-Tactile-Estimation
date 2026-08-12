#!/usr/bin/env python3
"""Shared sequence-level HDF5 primitives for processed tactile datasets.

The format is intentionally independent of the training Dataset.  One HDF5
file stores one visual sequence, while one JSONL manifest row continues to
represent one hand/query sample.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


SCHEMA_NAME = "tactile_sequence_hdf5"
SCHEMA_VERSION = "2.0.0"
SUPPORTED_SCHEMA_VERSIONS = ("1.0.0", SCHEMA_VERSION)
TACTILE_DIM = 13614
MANIFEST_SCHEMA = "tactile_query_manifest_v1"
SEQUENCE_MANIFEST_SCHEMA = "tactile_sequence_manifest_v1"
SUPPORTED_COMPRESSIONS = ("lzf", "gzip1")


def open_readonly(
    path: os.PathLike[str] | str,
    *,
    raw_chunk_cache_bytes: int = 256 * 1024,
    raw_chunk_cache_slots: int = 521,
) -> h5py.File:
    """Open an immutable training container without shared-filesystem locks.

    Training performs sparse, mostly one-shot row reads. A small raw chunk cache
    avoids multiplying HDF5's default cache footprint across every DDP worker,
    while disabled file locking removes needless lock-manager traffic for
    finalized read-only files.
    """

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
    raise OSError(
        f"Could not open immutable HDF5 container {path}: "
        + " | ".join(errors)
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def jsonl_bytes(row: dict[str, Any]) -> bytes:
    return (canonical_json(row) + "\n").encode("utf-8")


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: os.PathLike[str] | str) -> None:
    directory = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def temporary_sibling(path: Path, label: str = "tmp") -> Path:
    return path.with_name(f".{path.name}.{label}.{os.getpid()}.{uuid.uuid4().hex}")


class AtomicJsonlWriter:
    """Write deterministic JSONL and publish it with an atomic rename."""

    def __init__(self, path: os.PathLike[str] | str):
        self.path = Path(path)
        self.temp_path = temporary_sibling(self.path)
        self._handle = None
        self.count = 0
        self.digest = hashlib.sha256()

    def __enter__(self) -> "AtomicJsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.temp_path.open("wb")
        return self

    def write(self, row: dict[str, Any]) -> None:
        if self._handle is None:
            raise RuntimeError("AtomicJsonlWriter is not open")
        payload = jsonl_bytes(row)
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
        fsync_directory(self.path.parent)
        return False


def write_json_atomic(path: os.PathLike[str] | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temporary_sibling(path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        fsync_directory(path.parent)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _compression_args(name: str) -> dict[str, Any]:
    if name == "lzf":
        return {"compression": "lzf", "shuffle": True}
    if name == "gzip1":
        return {"compression": "gzip", "compression_opts": 1, "shuffle": True}
    raise ValueError(
        f"Unsupported pressure compression {name!r}; choose {SUPPORTED_COMPRESSIONS}"
    )


class SequenceHDF5Writer:
    """Single-writer, atomic sequence HDF5 builder.

    JPEG data is appended as untouched encoded bytes. Pressure is written one
    query at a time as losslessly compressed float32 chunks.
    """

    def __init__(
        self,
        target_path: os.PathLike[str] | str,
        *,
        dataset: str,
        split: str,
        sequence_key: str,
        frame_count: int,
        query_count: int,
        jpeg_total_bytes: int | None,
        metadata_record_count: int = 0,
        metadata_total_bytes: int = 0,
        source_fingerprint: str,
        pressure_compression: str = "lzf",
        tactile_dim: int = TACTILE_DIM,
        extra_attrs: dict[str, Any] | None = None,
    ):
        if frame_count <= 0:
            raise ValueError("A sequence HDF5 must contain at least one frame")
        if query_count < 0:
            raise ValueError("A sequence HDF5 query count cannot be negative")
        if jpeg_total_bytes is not None and jpeg_total_bytes <= 0:
            raise ValueError("A sequence HDF5 must contain non-empty JPEG payloads")
        self.target_path = Path(target_path)
        self.temp_path = temporary_sibling(self.target_path)
        self.dataset = str(dataset)
        self.split = str(split)
        self.sequence_key = str(sequence_key)
        self.frame_count = int(frame_count)
        self.query_count = int(query_count)
        self.jpeg_total_bytes = (
            None if jpeg_total_bytes is None else int(jpeg_total_bytes)
        )
        self._dynamic_jpeg = self.jpeg_total_bytes is None
        self.metadata_record_count = int(metadata_record_count)
        self.metadata_total_bytes = int(metadata_total_bytes)
        if self.metadata_record_count < 0 or self.metadata_total_bytes < 0:
            raise ValueError("Archive metadata counts cannot be negative")
        if bool(self.metadata_record_count) != bool(self.metadata_total_bytes):
            raise ValueError(
                "Archive metadata record count and byte count must both be zero or nonzero"
            )
        self.source_fingerprint = str(source_fingerprint)
        self.pressure_compression = str(pressure_compression)
        self.tactile_dim = int(tactile_dim)
        self.extra_attrs = dict(extra_attrs or {})
        self.file: h5py.File | None = None
        self._jpeg_size = 0
        self._jpeg_capacity = 0
        self._frames_written = 0
        self._queries_written = 0
        self._train_queries_written = 0
        self._metadata_size = 0
        self._metadata_written = 0

    def __enter__(self) -> "SequenceHDF5Writer":
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.temp_path, "w", libver="latest")
        attrs = self.file.attrs
        attrs["schema_name"] = SCHEMA_NAME
        attrs["schema_version"] = SCHEMA_VERSION
        attrs["dataset"] = self.dataset
        attrs["split"] = self.split
        attrs["sequence_key"] = self.sequence_key
        attrs["frame_count"] = self.frame_count
        attrs["query_count"] = self.query_count
        attrs["archive_query_count"] = self.query_count
        attrs["metadata_record_count"] = self.metadata_record_count
        attrs["jpeg_total_bytes"] = (
            -1 if self.jpeg_total_bytes is None else self.jpeg_total_bytes
        )
        attrs["pressure_dim"] = self.tactile_dim
        attrs["source_fingerprint"] = self.source_fingerprint
        attrs["pressure_compression"] = self.pressure_compression
        for key, value in self.extra_attrs.items():
            attrs[str(key)] = value

        frames = self.file.create_group("frames")
        frames.create_dataset("frame_idx", (self.frame_count,), dtype="<i8")
        frames.create_dataset("source_frame_idx", (self.frame_count,), dtype="<i8")
        frames.create_dataset("timestamp", (self.frame_count,), dtype="<f8")
        frames.create_dataset("timestamp_kind", (self.frame_count,), dtype="u1")
        frames.create_dataset("image_hw", (self.frame_count, 2), dtype="<u4")

        rgb = self.file.create_group("images").create_group("rgb")
        if self._dynamic_jpeg:
            rgb.create_dataset(
                "jpeg_data",
                shape=(0,),
                maxshape=(None,),
                chunks=(4 * 1024 * 1024,),
                dtype="u1",
            )
        else:
            rgb.create_dataset(
                "jpeg_data",
                shape=(self.jpeg_total_bytes,),
                dtype="u1",
            )
        rgb.create_dataset("jpeg_offsets", (self.frame_count + 1,), dtype="<u8")
        rgb["jpeg_offsets"][0] = 0
        rgb.create_dataset("jpeg_sha256", (self.frame_count,), dtype="S64")

        queries = self.file.create_group("queries")
        queries.create_dataset("frame_row", (self.query_count,), dtype="<i8")
        queries.create_dataset("frame_idx", (self.query_count,), dtype="<i8")
        queries.create_dataset("hand_code", (self.query_count,), dtype="i1")
        queries.create_dataset("is_right", (self.query_count,), dtype="i1")
        queries.create_dataset("is_trainable", (self.query_count,), dtype="u1")
        queries.create_dataset("bbox_xyxy", (self.query_count, 4), dtype="<f4")
        queries.create_dataset("bbox_score", (self.query_count,), dtype="<f4")
        queries.create_dataset(
            "keypoints_3d_cam", (self.query_count, 21, 3), dtype="<f4"
        )
        queries.create_dataset(
            "keypoints_valid", (self.query_count, 21), dtype="u1"
        )
        queries.create_dataset("query_uid", (self.query_count,), dtype=_string_dtype())
        queries.create_dataset("query_alias", (self.query_count,), dtype=_string_dtype())
        queries.create_dataset(
            "source_sample_relpath", (self.query_count,), dtype=_string_dtype()
        )
        queries.create_dataset(
            "bbox_source_json", (self.query_count,), dtype=_string_dtype()
        )
        queries.create_dataset(
            "pressure_source_key", (self.query_count,), dtype=_string_dtype()
        )

        targets = self.file.create_group("targets")
        targets.create_dataset(
            "pressure",
            (self.query_count, self.tactile_dim),
            dtype="<f4",
            chunks=(1, self.tactile_dim),
            maxshape=(None, self.tactile_dim),
            **_compression_args(self.pressure_compression),
        )
        targets.create_dataset("pressure_sha256", (self.query_count,), dtype="S64")
        targets.create_dataset("max_pressure", (self.query_count,), dtype="<f4")
        targets.create_dataset("volume", (self.query_count,), dtype="<f4")
        targets.create_dataset("active_count", (self.query_count,), dtype="<i4")

        if self.metadata_record_count:
            archive = self.file.create_group("archive").create_group("meta_json")
            archive.create_dataset(
                "data",
                (self.metadata_total_bytes,),
                dtype="u1",
                compression="lzf",
                shuffle=True,
            )
            archive.create_dataset(
                "offsets", (self.metadata_record_count + 1,), dtype="<u8"
            )
            archive["offsets"][0] = 0
            archive.create_dataset(
                "sha256", (self.metadata_record_count,), dtype="S64"
            )
            archive.create_dataset(
                "frame_row", (self.metadata_record_count,), dtype="<i8"
            )
            archive.create_dataset(
                "query_alias", (self.metadata_record_count,), dtype=_string_dtype()
            )
            archive.create_dataset(
                "source_sample_relpath",
                (self.metadata_record_count,),
                dtype=_string_dtype(),
            )
        return self

    def write_metadata_record(
        self,
        row: int,
        *,
        frame_row: int,
        query_alias: str,
        source_sample_relpath: str,
        meta_json_bytes: bytes,
    ) -> None:
        if self.file is None:
            raise RuntimeError("SequenceHDF5Writer is not open")
        if row != self._metadata_written or not 0 <= row < self.metadata_record_count:
            raise ValueError(f"Metadata records must be written in row order; got row={row}")
        if not meta_json_bytes:
            raise ValueError("Archived meta.json payload cannot be empty")
        payload = np.frombuffer(meta_json_bytes, dtype=np.uint8)
        end = self._metadata_size + int(payload.size)
        if end > self.metadata_total_bytes:
            raise ValueError("Archived metadata exceeds declared byte count")
        archive = self.file["archive/meta_json"]
        archive["data"][self._metadata_size : end] = payload
        archive["offsets"][row + 1] = end
        archive["sha256"][row] = hashlib.sha256(meta_json_bytes).hexdigest().encode(
            "ascii"
        )
        archive["frame_row"][row] = int(frame_row)
        archive["query_alias"][row] = str(query_alias)
        archive["source_sample_relpath"][row] = str(source_sample_relpath)
        self._metadata_size = end
        self._metadata_written += 1

    def write_frame(
        self,
        row: int,
        *,
        frame_idx: int,
        source_frame_idx: int | None,
        timestamp: float | None,
        timestamp_kind: int,
        image_hw: tuple[int, int],
        jpeg_bytes: bytes,
    ) -> None:
        if self.file is None:
            raise RuntimeError("SequenceHDF5Writer is not open")
        if row != self._frames_written or not 0 <= row < self.frame_count:
            raise ValueError(f"Frames must be written in row order; got row={row}")
        if not jpeg_bytes:
            raise ValueError(f"Frame {frame_idx} has empty JPEG bytes")
        data = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        rgb = self.file["images/rgb"]
        end = self._jpeg_size + int(data.size)
        if not self._dynamic_jpeg and end > self.jpeg_total_bytes:
            raise ValueError(
                f"JPEG payload exceeds declared total bytes "
                f"({end} > {self.jpeg_total_bytes})"
            )
        if self._dynamic_jpeg and end > rgb["jpeg_data"].shape[0]:
            current_capacity = int(rgb["jpeg_data"].shape[0])
            new_capacity = max(
                end,
                64 * 1024 * 1024,
                current_capacity * 2,
            )
            rgb["jpeg_data"].resize((new_capacity,))
            self._jpeg_capacity = new_capacity
        rgb["jpeg_data"][self._jpeg_size : end] = data
        rgb["jpeg_offsets"][row + 1] = end
        rgb["jpeg_sha256"][row] = hashlib.sha256(jpeg_bytes).hexdigest().encode("ascii")

        frames = self.file["frames"]
        frames["frame_idx"][row] = int(frame_idx)
        frames["source_frame_idx"][row] = (
            -1 if source_frame_idx is None else int(source_frame_idx)
        )
        frames["timestamp"][row] = np.nan if timestamp is None else float(timestamp)
        frames["timestamp_kind"][row] = int(timestamp_kind)
        frames["image_hw"][row] = np.asarray(image_hw, dtype=np.uint32)
        self._jpeg_size = end
        self._frames_written += 1

    def write_query(
        self,
        row: int,
        *,
        frame_row: int,
        frame_idx: int,
        hand_code: int,
        is_right: int,
        bbox_xyxy: np.ndarray,
        bbox_score: float,
        keypoints_3d_cam: np.ndarray,
        keypoints_valid: np.ndarray,
        query_uid: str,
        query_alias: str,
        source_sample_relpath: str,
        bbox_source: Any,
        pressure_source_key: str,
        pressure: np.ndarray,
        max_pressure: float,
        volume: float,
        active_count: int,
        is_trainable: bool = True,
    ) -> None:
        if self.file is None:
            raise RuntimeError("SequenceHDF5Writer is not open")
        if row != self._queries_written or not 0 <= row < self.query_count:
            raise ValueError(f"Queries must be written in row order; got row={row}")
        pressure = np.asarray(pressure, dtype=np.float32)
        bbox = np.asarray(bbox_xyxy, dtype=np.float32)
        keypoints = np.asarray(keypoints_3d_cam, dtype=np.float32)
        keypoint_validity = np.asarray(keypoints_valid, dtype=np.uint8)
        if pressure.shape != (self.tactile_dim,) or not np.isfinite(pressure).all():
            raise ValueError(
                f"Query {query_uid} pressure must be finite [{self.tactile_dim}] float32"
            )
        if bbox.shape != (4,):
            raise ValueError(f"Query {query_uid} bbox must have shape [4]")
        if is_trainable and not np.isfinite(bbox).all():
            raise ValueError(f"Trainable query {query_uid} bbox must be finite")
        if keypoints.shape != (21, 3) or not np.isfinite(keypoints).all():
            raise ValueError(f"Query {query_uid} keypoints must be finite [21,3]")
        if keypoint_validity.shape != (21,):
            raise ValueError(f"Query {query_uid} keypoint validity must be [21]")
        queries = self.file["queries"]
        queries["frame_row"][row] = int(frame_row)
        queries["frame_idx"][row] = int(frame_idx)
        queries["hand_code"][row] = int(hand_code)
        queries["is_right"][row] = int(is_right)
        queries["is_trainable"][row] = int(bool(is_trainable))
        self._train_queries_written += int(bool(is_trainable))
        queries["bbox_xyxy"][row] = bbox
        queries["bbox_score"][row] = float(bbox_score)
        queries["keypoints_3d_cam"][row] = keypoints
        queries["keypoints_valid"][row] = keypoint_validity
        queries["query_uid"][row] = str(query_uid)
        queries["query_alias"][row] = str(query_alias)
        queries["source_sample_relpath"][row] = str(source_sample_relpath)
        queries["bbox_source_json"][row] = (
            "" if bbox_source is None else canonical_json(bbox_source)
        )
        queries["pressure_source_key"][row] = str(pressure_source_key)

        targets = self.file["targets"]
        targets["pressure"][row] = pressure
        targets["pressure_sha256"][row] = hashlib.sha256(
            np.ascontiguousarray(pressure, dtype="<f4").tobytes()
        ).hexdigest().encode("ascii")
        targets["max_pressure"][row] = float(max_pressure)
        targets["volume"][row] = float(volume)
        targets["active_count"][row] = int(active_count)
        self._queries_written += 1

    def _publish(self) -> None:
        if self.file is None:
            raise RuntimeError("SequenceHDF5Writer is not open")
        if self._frames_written != self.frame_count:
            raise RuntimeError(
                f"Wrote {self._frames_written}/{self.frame_count} HDF5 frames"
            )
        if self._queries_written != self.query_count:
            raise RuntimeError(
                f"Wrote {self._queries_written}/{self.query_count} HDF5 queries"
            )
        if self._metadata_written != self.metadata_record_count:
            raise RuntimeError(
                f"Wrote {self._metadata_written}/{self.metadata_record_count} "
                "archive metadata records"
            )
        if self._metadata_size != self.metadata_total_bytes:
            raise RuntimeError(
                f"Wrote {self._metadata_size}/{self.metadata_total_bytes} "
                "archive metadata bytes"
            )
        self.file.attrs["train_query_count"] = self._train_queries_written
        if not self._dynamic_jpeg and self._jpeg_size != self.jpeg_total_bytes:
            raise RuntimeError(
                f"Wrote {self._jpeg_size}/{self.jpeg_total_bytes} JPEG bytes"
            )
        if self._dynamic_jpeg:
            self.file["images/rgb/jpeg_data"].resize((self._jpeg_size,))
            self.file.attrs["jpeg_total_bytes"] = self._jpeg_size
        self.file.flush()
        self.file.close()
        self.file = None
        with self.temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self.temp_path, self.target_path)
        fsync_directory(self.target_path.parent)

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None:
            if self.file is not None:
                self.file.close()
                self.file = None
            self.temp_path.unlink(missing_ok=True)
            return False
        try:
            self._publish()
        except BaseException:
            if self.file is not None:
                self.file.close()
                self.file = None
            self.temp_path.unlink(missing_ok=True)
            raise
        return False


def _decoded_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def manifest_rows_from_hdf5(
    h5_path: os.PathLike[str] | str,
    processed_root: os.PathLike[str] | str,
) -> list[dict[str, Any]]:
    h5_path = Path(h5_path).resolve()
    processed_root = Path(processed_root).resolve()
    h5_relpath = h5_path.relative_to(processed_root).as_posix()
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle:
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
            row = {
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
                "source_frame_idx": (
                    None if source_frame_idx < 0 else source_frame_idx
                ),
                "timestamp": timestamp if math.isfinite(timestamp) else None,
                "max_pressure": float(targets["max_pressure"][query_row]),
                "target_volume": float(targets["volume"][query_row]),
                "target_active_count": int(targets["active_count"][query_row]),
            }
            rows.append(row)
    return rows


def sequence_manifest_row(
    h5_path: os.PathLike[str] | str,
    processed_root: os.PathLike[str] | str,
) -> dict[str, Any]:
    h5_path = Path(h5_path).resolve()
    processed_root = Path(processed_root).resolve()
    with h5py.File(h5_path, "r") as handle:
        return {
            "schema": SEQUENCE_MANIFEST_SCHEMA,
            "hdf5_schema_version": _decoded_string(
                handle.attrs["schema_version"]
            ),
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
            "metadata_record_count": int(
                handle.attrs.get("metadata_record_count", 0)
            ),
            "pressure_dim": int(handle.attrs["pressure_dim"]),
            "source_fingerprint": _decoded_string(
                handle.attrs["source_fingerprint"]
            ),
            "file_size_bytes": int(h5_path.stat().st_size),
        }


def verify_sequence_hdf5(
    h5_path: os.PathLike[str] | str,
    *,
    expected_source_fingerprint: str | None = None,
    expected_pressure_compression: str | None = None,
    deep: bool = False,
) -> dict[str, Any]:
    """Validate schema, shapes, chunks and optionally every JPEG/pressure row."""

    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        if _decoded_string(handle.attrs.get("schema_name", "")) != SCHEMA_NAME:
            raise ValueError(f"{path}: unsupported schema_name")
        schema_version = _decoded_string(handle.attrs.get("schema_version", ""))
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
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
        if pressure_compression not in SUPPORTED_COMPRESSIONS:
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
        keypoints = np.asarray(
            handle["queries/keypoints_3d_cam"][:], dtype=np.float32
        )
        if keypoints.shape != (query_count, 21, 3) or not np.isfinite(
            keypoints
        ).all():
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
                expected_digest = _decoded_string(
                    handle["targets/pressure_sha256"][row]
                )
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
            "metadata_record_count": int(
                handle.attrs.get("metadata_record_count", 0)
            ),
            "pressure_dim": pressure_dim,
            "source_fingerprint": source_fingerprint,
            "deep": bool(deep),
        }


def jsonl_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(jsonl_bytes(row))
    return digest.hexdigest()


def file_sha256_or_none(path: os.PathLike[str] | str) -> str | None:
    path = Path(path)
    return sha256_file(path) if path.is_file() else None


def finite_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

