"""MoGe geometry sidecars for tactile input priors.

This module deliberately depends only on NumPy, OpenCV, and h5py. It is safe to
import in DataLoader workers without importing torch or initializing CUDA.

Affine convention
-----------------
``teacher_affine`` maps source-image pixel coordinates to the stored teacher
grid. ``rgb_affine`` maps the same source-image coordinates to the RGB crop.
The runtime transform composes these two mappings and optionally rescales the
RGB crop to a lower-resolution geometry grid with pixel-center correction.

Point-normal channel order
--------------------------
The returned array has shape ``[8, H, W]`` and channels::

    centered_x, centered_y, centered_z, radius, valid,
    normal_x, normal_y, normal_z

Invalid locations are zero in every channel. Spatial shuffle, when requested,
is applied after warp, normalization, and canonical left-to-right flip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import h5py
import numpy as np


SCHEMA_NAME = "tactile_moge_geometry_sidecar"
SCHEMA_VERSION = "1.0"
AFFINE_CONVENTION = "source_pixel_to_grid_pixel"
POINTNORMAL_CHANNELS = (
    "centered_x",
    "centered_y",
    "centered_z",
    "radius",
    "valid",
    "normal_x",
    "normal_y",
    "normal_z",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(name: str, value: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256")
    return normalized


def _validate_hw(name: str, value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain (height, width)")
    height, width = (int(item) for item in value)
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must be positive, got {(height, width)}")
    return height, width


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class DepthSidecarConfig:
    """Configuration shared by all sequence shards in one sidecar collection."""

    teacher_model: str
    model_sha256: str
    manifest_sha256: str
    teacher_input_hw: tuple[int, int] = (512, 384)
    stored_grid_hw: tuple[int, int] = (64, 48)
    teacher_bbox_scale: float = 1.65
    coordinate_convention: str = AFFINE_CONVENTION
    extra: Mapping[str, Any] = field(default_factory=dict)
    config_sha256: str = ""

    def __post_init__(self) -> None:
        if not str(self.teacher_model).strip():
            raise ValueError("teacher_model must be non-empty")
        object.__setattr__(self, "model_sha256", _validate_sha256("model_sha256", self.model_sha256))
        object.__setattr__(
            self,
            "manifest_sha256",
            _validate_sha256("manifest_sha256", self.manifest_sha256),
        )
        object.__setattr__(
            self,
            "teacher_input_hw",
            _validate_hw("teacher_input_hw", self.teacher_input_hw),
        )
        object.__setattr__(
            self,
            "stored_grid_hw",
            _validate_hw("stored_grid_hw", self.stored_grid_hw),
        )
        if not np.isfinite(self.teacher_bbox_scale) or self.teacher_bbox_scale <= 0:
            raise ValueError("teacher_bbox_scale must be finite and positive")
        if self.coordinate_convention != AFFINE_CONVENTION:
            raise ValueError(
                f"Unsupported coordinate convention {self.coordinate_convention!r}; "
                f"expected {AFFINE_CONVENTION!r}"
            )
        normalized_extra = json.loads(_canonical_json(dict(self.extra)))
        object.__setattr__(self, "extra", normalized_extra)
        computed = _sha256_json(self.hash_payload())
        if self.config_sha256:
            supplied = _validate_sha256("config_sha256", self.config_sha256)
            if supplied != computed:
                raise ValueError(
                    "config_sha256 does not match the canonical sidecar configuration: "
                    f"supplied={supplied}, computed={computed}"
                )
        object.__setattr__(self, "config_sha256", computed)

    def hash_payload(self) -> dict[str, Any]:
        return {
            "teacher_model": self.teacher_model,
            "model_sha256": self.model_sha256,
            "teacher_input_hw": list(self.teacher_input_hw),
            "stored_grid_hw": list(self.stored_grid_hw),
            "teacher_bbox_scale": float(self.teacher_bbox_scale),
            "coordinate_convention": self.coordinate_convention,
            "extra": dict(self.extra),
        }

    def semantic_hash_payload(self) -> dict[str, Any]:
        """Return extraction semantics without split-specific provenance."""
        payload = self.hash_payload()
        extra = dict(payload["extra"])
        extra.pop("manifest_name", None)
        payload["extra"] = extra
        return payload

    @property
    def semantic_config_sha256(self) -> str:
        return _sha256_json(self.semantic_hash_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["teacher_input_hw"] = list(self.teacher_input_hw)
        payload["stored_grid_hw"] = list(self.stored_grid_hw)
        payload["extra"] = dict(self.extra)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DepthSidecarConfig":
        return cls(
            teacher_model=str(payload["teacher_model"]),
            model_sha256=str(payload["model_sha256"]),
            manifest_sha256=str(payload["manifest_sha256"]),
            teacher_input_hw=tuple(payload.get("teacher_input_hw", (512, 384))),
            stored_grid_hw=tuple(payload.get("stored_grid_hw", (64, 48))),
            teacher_bbox_scale=float(payload.get("teacher_bbox_scale", 1.65)),
            coordinate_convention=str(
                payload.get("coordinate_convention", AFFINE_CONVENTION)
            ),
            extra=dict(payload.get("extra", {})),
            config_sha256=str(payload.get("config_sha256", "")),
        )


@dataclass(frozen=True)
class GeometryRecord:
    sample_uid: str
    query_row: int
    point: np.ndarray
    normal: np.ndarray
    valid: np.ndarray
    teacher_affine: np.ndarray


@dataclass
class CoverageReport:
    shard_count: int = 0
    record_count: int = 0
    expected_count: int = 0
    missing_count: int = 0
    unexpected_count: int = 0
    duplicate_uid_count: int = 0
    duplicate_sequence_count: int = 0
    query_row_mismatch_count: int = 0
    invalid_record_count: int = 0
    model_sha256: str | None = None
    config_sha256: str | None = None
    manifest_sha256: str | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.missing_count,
                self.unexpected_count,
                self.duplicate_uid_count,
                self.duplicate_sequence_count,
                self.query_row_mismatch_count,
                self.invalid_record_count,
                len(self.issues),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        return payload


def sequence_sidecar_filename(sequence_key: str) -> str:
    """Return a stable, collision-resistant filename for a sequence key."""

    key = str(sequence_key).strip()
    if not key:
        raise ValueError("sequence_key must be non-empty")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("._") or "sequence"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:96]}.{digest}.depth.h5"


class SequenceSidecarWriter:
    """Atomically write one sequence-sharded geometry HDF5 file."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        sequence_key: str,
        config: DepthSidecarConfig,
        expected_count: int | None = None,
        compression: str | None = "lzf",
        overwrite: bool = False,
        growth_records: int = 64,
    ) -> None:
        self.path = Path(path)
        self.sequence_key = str(sequence_key).strip()
        if not self.sequence_key:
            raise ValueError("sequence_key must be non-empty")
        self.config = config
        self.expected_count = None if expected_count is None else int(expected_count)
        if self.expected_count is not None and self.expected_count < 0:
            raise ValueError("expected_count must be non-negative")
        if compression not in (None, "lzf", "gzip"):
            raise ValueError("compression must be None, 'lzf', or 'gzip'")
        self.compression = compression
        self.overwrite = bool(overwrite)
        self.growth_records = max(int(growth_records), 1)
        self._file: h5py.File | None = None
        self._temp_path: Path | None = None
        self._count = 0
        self._capacity = 0
        self._datasets: tuple[h5py.Dataset, ...] = ()
        self._uids: set[str] = set()
        self._query_rows: set[int] = set()

    def __enter__(self) -> "SequenceSidecarWriter":
        if self._file is not None:
            raise RuntimeError("Writer is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not self.overwrite:
            raise FileExistsError(self.path)
        token = uuid.uuid4().hex
        self._temp_path = self.path.parent / f".{self.path.name}.{os.getpid()}.{token}.tmp"
        self._capacity = max(self.expected_count or 0, 1)
        self._file = h5py.File(self._temp_path, "w", libver="latest")
        self._initialize_file()
        return self

    def _initialize_file(self) -> None:
        assert self._file is not None
        handle = self._file
        handle.attrs["schema_name"] = SCHEMA_NAME
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["complete"] = np.uint8(0)
        handle.attrs["sequence_key"] = self.sequence_key
        handle.attrs["record_count"] = np.int64(0)
        handle.attrs["config_json"] = _canonical_json(self.config.to_dict())
        handle.attrs["config_sha256"] = self.config.config_sha256
        handle.attrs["model_sha256"] = self.config.model_sha256
        handle.attrs["manifest_sha256"] = self.config.manifest_sha256
        handle.attrs["affine_convention"] = AFFINE_CONVENTION
        handle.attrs["pointnormal_channels"] = _canonical_json(POINTNORMAL_CHANNELS)

        query_group = handle.create_group("queries")
        geometry_group = handle.create_group("geometry")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        row_chunk = min(max(self.growth_records, 1), max(self._capacity, 1))
        sample_uid = query_group.create_dataset(
            "sample_uid",
            shape=(self._capacity,),
            maxshape=(None,),
            chunks=(row_chunk,),
            dtype=string_dtype,
        )
        query_row = query_group.create_dataset(
            "query_row",
            shape=(self._capacity,),
            maxshape=(None,),
            chunks=(row_chunk,),
            dtype=np.int64,
        )
        height, width = self.config.stored_grid_hw
        geometry_kwargs: dict[str, Any] = {}
        if self.compression is not None:
            geometry_kwargs["compression"] = self.compression
            if self.compression == "gzip":
                geometry_kwargs["compression_opts"] = 1
        point = geometry_group.create_dataset(
            "point",
            shape=(self._capacity, height, width, 3),
            maxshape=(None, height, width, 3),
            chunks=(1, height, width, 3),
            dtype=np.float16,
            **geometry_kwargs,
        )
        normal = geometry_group.create_dataset(
            "normal",
            shape=(self._capacity, height, width, 3),
            maxshape=(None, height, width, 3),
            chunks=(1, height, width, 3),
            dtype=np.float16,
            **geometry_kwargs,
        )
        valid = geometry_group.create_dataset(
            "valid",
            shape=(self._capacity, height, width),
            maxshape=(None, height, width),
            chunks=(1, height, width),
            dtype=np.uint8,
            **geometry_kwargs,
        )
        teacher_affine = geometry_group.create_dataset(
            "teacher_affine",
            shape=(self._capacity, 2, 3),
            maxshape=(None, 2, 3),
            chunks=(1, 2, 3),
            dtype=np.float32,
        )
        self._datasets = (sample_uid, query_row, point, normal, valid, teacher_affine)

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        new_capacity = max(required, self._capacity + self.growth_records, self._capacity * 2)
        for dataset in self._datasets:
            dataset.resize((new_capacity,) + dataset.shape[1:])
        self._capacity = new_capacity

    def append(
        self,
        sample_uid: str,
        query_row: int,
        point: np.ndarray,
        normal: np.ndarray,
        valid: np.ndarray,
        teacher_affine: np.ndarray,
    ) -> None:
        if self._file is None:
            raise RuntimeError("Writer must be used as a context manager")
        uid = str(sample_uid).strip()
        row = int(query_row)
        if not uid:
            raise ValueError("sample_uid must be non-empty")
        if row < 0:
            raise ValueError("query_row must be non-negative")
        if uid in self._uids:
            raise ValueError(f"Duplicate sample_uid in sequence shard: {uid}")
        if row in self._query_rows:
            raise ValueError(f"Duplicate query_row in sequence shard: {row}")
        height, width = self.config.stored_grid_hw
        point_array = np.asarray(point, dtype=np.float32)
        normal_array = np.asarray(normal, dtype=np.float32)
        valid_array = np.asarray(valid, dtype=bool)
        affine_array = _as_affine(teacher_affine, "teacher_affine")
        expected_vector_shape = (height, width, 3)
        if point_array.shape != expected_vector_shape:
            raise ValueError(f"point must have shape {expected_vector_shape}, got {point_array.shape}")
        if normal_array.shape != expected_vector_shape:
            raise ValueError(
                f"normal must have shape {expected_vector_shape}, got {normal_array.shape}"
            )
        if valid_array.shape != (height, width):
            raise ValueError(f"valid must have shape {(height, width)}, got {valid_array.shape}")

        finite = np.isfinite(point_array).all(axis=-1) & np.isfinite(normal_array).all(axis=-1)
        valid_array &= finite
        point_array = point_array.copy()
        normal_array = normal_array.copy()
        point_array[~valid_array] = 0.0
        normal_array[~valid_array] = 0.0
        normal_norm = np.linalg.norm(normal_array, axis=-1, keepdims=True)
        normalized = normal_norm[..., 0] > 1e-8
        normal_array[normalized] /= normal_norm[normalized]
        valid_array &= normalized
        point_array[~valid_array] = 0.0
        normal_array[~valid_array] = 0.0

        self._ensure_capacity(self._count + 1)
        sample_ds, row_ds, point_ds, normal_ds, valid_ds, affine_ds = self._datasets
        sample_ds[self._count] = uid
        row_ds[self._count] = row
        point_ds[self._count] = point_array.astype(np.float16)
        normal_ds[self._count] = normal_array.astype(np.float16)
        valid_ds[self._count] = valid_array.astype(np.uint8)
        affine_ds[self._count] = affine_array
        self._uids.add(uid)
        self._query_rows.add(row)
        self._count += 1

    def _finalize(self) -> None:
        assert self._file is not None
        if self.expected_count is not None and self._count != self.expected_count:
            raise RuntimeError(
                f"Expected {self.expected_count} records, wrote {self._count}"
            )
        for dataset in self._datasets:
            dataset.resize((self._count,) + dataset.shape[1:])

        query_rows = np.asarray(self._datasets[1][:], dtype=np.int64)
        order = np.argsort(query_rows, kind="stable")
        lookup = self._file.create_group("lookup")
        lookup.create_dataset("query_row_sorted", data=query_rows[order], dtype=np.int64)
        lookup.create_dataset("storage_row_by_query", data=order.astype(np.int64))
        uid_digest = np.empty((self._count, 32), dtype=np.uint8)
        for index, uid in enumerate(self._datasets[0].asstr()[:]):
            uid_digest[index] = np.frombuffer(
                hashlib.sha256(uid.encode("utf-8")).digest(), dtype=np.uint8
            )
        lookup.create_dataset("sample_uid_sha256", data=uid_digest, dtype=np.uint8)
        self._file.attrs["record_count"] = np.int64(self._count)
        self._file.attrs["complete"] = np.uint8(1)
        self._file.flush()

    def _abort(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._temp_path is not None:
            self._temp_path.unlink(missing_ok=True)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is not None:
            self._abort()
            return False
        try:
            self._finalize()
            assert self._file is not None and self._temp_path is not None
            self._file.close()
            self._file = None
            with self._temp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            if self.path.exists() and not self.overwrite:
                raise FileExistsError(self.path)
            os.replace(self._temp_path, self.path)
            _fsync_directory(self.path.parent)
        except BaseException:
            self._abort()
            raise
        return False


class SequenceSidecarReader:
    """Read one sequence shard with fork-safe lazy HDF5 reopening."""

    def __init__(
        self,
        path: os.PathLike[str] | str,
        expected_model_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.expected_model_sha256 = expected_model_sha256
        self.expected_config_sha256 = expected_config_sha256
        self.expected_manifest_sha256 = expected_manifest_sha256
        self._file: h5py.File | None = None
        self._pid: int | None = None
        self._uid_to_storage: dict[str, int] | None = None
        self._query_to_storage: dict[int, int] | None = None
        self.config: DepthSidecarConfig
        self.sequence_key: str
        self.record_count: int
        self._open()

    def _open(self) -> None:
        self.close()
        handle = h5py.File(self.path, "r", libver="latest", swmr=True)
        try:
            if _decode_string(handle.attrs.get("schema_name", "")) != SCHEMA_NAME:
                raise ValueError(f"Not a {SCHEMA_NAME} file: {self.path}")
            if _decode_string(handle.attrs.get("schema_version", "")) != SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported sidecar schema at {self.path}: "
                    f"{handle.attrs.get('schema_version')!r}"
                )
            if int(handle.attrs.get("complete", 0)) != 1:
                raise ValueError(f"Sidecar was not atomically completed: {self.path}")
            config = DepthSidecarConfig.from_dict(
                json.loads(_decode_string(handle.attrs["config_json"]))
            )
            self._check_expected_hash(
                "model_sha256", config.model_sha256, self.expected_model_sha256
            )
            self._check_expected_hash(
                "config_sha256", config.config_sha256, self.expected_config_sha256
            )
            self._check_expected_hash(
                "manifest_sha256", config.manifest_sha256, self.expected_manifest_sha256
            )
            record_count = int(handle.attrs["record_count"])
            for dataset_name in (
                "queries/sample_uid",
                "queries/query_row",
                "geometry/point",
                "geometry/normal",
                "geometry/valid",
                "geometry/teacher_affine",
            ):
                if dataset_name not in handle:
                    raise ValueError(f"Missing dataset {dataset_name!r} in {self.path}")
                if len(handle[dataset_name]) != record_count:
                    raise ValueError(
                        f"Dataset {dataset_name!r} length does not match record_count "
                        f"in {self.path}"
                    )
            height, width = config.stored_grid_hw
            expected_shapes = {
                "geometry/point": (record_count, height, width, 3),
                "geometry/normal": (record_count, height, width, 3),
                "geometry/valid": (record_count, height, width),
                "geometry/teacher_affine": (record_count, 2, 3),
            }
            for name, shape in expected_shapes.items():
                if handle[name].shape != shape:
                    raise ValueError(
                        f"Dataset {name!r} has shape {handle[name].shape}, expected {shape}"
                    )
        except BaseException:
            handle.close()
            raise
        self._file = handle
        self._pid = os.getpid()
        self._uid_to_storage = None
        self._query_to_storage = None
        self.config = config
        self.sequence_key = _decode_string(handle.attrs["sequence_key"])
        self.record_count = record_count

    @staticmethod
    def _check_expected_hash(name: str, actual: str, expected: str | None) -> None:
        if expected is None:
            return
        normalized = _validate_sha256(name, expected)
        if actual != normalized:
            raise ValueError(f"{name} mismatch: expected={normalized}, actual={actual}")

    def _ensure_open(self) -> h5py.File:
        if self._file is None or self._pid != os.getpid():
            self._open()
        assert self._file is not None
        return self._file

    def _build_lookup(self) -> None:
        handle = self._ensure_open()
        uids = handle["queries/sample_uid"].asstr()[:]
        rows = np.asarray(handle["queries/query_row"][:], dtype=np.int64)
        uid_lookup: dict[str, int] = {}
        row_lookup: dict[int, int] = {}
        for storage_row, (uid, query_row) in enumerate(zip(uids, rows)):
            uid_text = str(uid)
            row_value = int(query_row)
            if uid_text in uid_lookup:
                raise ValueError(f"Duplicate sample_uid {uid_text!r} in {self.path}")
            if row_value in row_lookup:
                raise ValueError(f"Duplicate query_row {row_value} in {self.path}")
            uid_lookup[uid_text] = storage_row
            row_lookup[row_value] = storage_row
        self._uid_to_storage = uid_lookup
        self._query_to_storage = row_lookup

    def storage_row(
        self,
        sample_uid: str | None = None,
        query_row: int | None = None,
    ) -> int:
        if sample_uid is None and query_row is None:
            raise ValueError("At least one of sample_uid or query_row is required")
        if self._uid_to_storage is None or self._query_to_storage is None:
            self._build_lookup()
        assert self._uid_to_storage is not None and self._query_to_storage is not None
        uid_position = None
        query_position = None
        if sample_uid is not None:
            uid = str(sample_uid)
            if uid not in self._uid_to_storage:
                raise KeyError(f"sample_uid not found in {self.path}: {uid}")
            uid_position = self._uid_to_storage[uid]
        if query_row is not None:
            row = int(query_row)
            if row not in self._query_to_storage:
                raise KeyError(f"query_row not found in {self.path}: {row}")
            query_position = self._query_to_storage[row]
        if uid_position is not None and query_position is not None and uid_position != query_position:
            raise KeyError(
                f"sample_uid={sample_uid!r} and query_row={query_row} identify different records"
            )
        return int(uid_position if uid_position is not None else query_position)

    def read(
        self,
        sample_uid: str | None = None,
        query_row: int | None = None,
    ) -> GeometryRecord:
        handle = self._ensure_open()
        storage_row = self.storage_row(sample_uid=sample_uid, query_row=query_row)
        uid = _decode_string(handle["queries/sample_uid"][storage_row])
        row = int(handle["queries/query_row"][storage_row])
        return GeometryRecord(
            sample_uid=uid,
            query_row=row,
            point=np.asarray(handle["geometry/point"][storage_row], dtype=np.float32),
            normal=np.asarray(handle["geometry/normal"][storage_row], dtype=np.float32),
            valid=np.asarray(handle["geometry/valid"][storage_row], dtype=bool),
            teacher_affine=np.asarray(
                handle["geometry/teacher_affine"][storage_row], dtype=np.float32
            ),
        )

    def iter_identity(self) -> Iterator[tuple[str, int]]:
        handle = self._ensure_open()
        uids = handle["queries/sample_uid"].asstr()
        rows = handle["queries/query_row"]
        for index in range(self.record_count):
            yield str(uids[index]), int(rows[index])

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._pid = None
        self._uid_to_storage = None
        self._query_to_storage = None

    def __enter__(self) -> "SequenceSidecarReader":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.close()
        return False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        state["_uid_to_storage"] = None
        state["_query_to_storage"] = None
        return state


def _as_affine(value: np.ndarray, name: str) -> np.ndarray:
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape == (3, 3):
        affine = affine[:2]
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        raise ValueError(f"{name} must be finite with shape (2, 3) or (3, 3)")
    homogeneous = np.vstack((affine, np.asarray((0.0, 0.0, 1.0))))
    if abs(float(np.linalg.det(homogeneous))) < 1e-12:
        raise ValueError(f"{name} is singular")
    return affine.astype(np.float32)


def _homogeneous(affine: np.ndarray, name: str) -> np.ndarray:
    affine_2x3 = _as_affine(affine, name).astype(np.float64)
    return np.vstack((affine_2x3, np.asarray((0.0, 0.0, 1.0))))


def compose_teacher_to_output_affine(
    teacher_affine: np.ndarray,
    rgb_affine: np.ndarray,
    rgb_output_hw: Sequence[int],
    output_hw: Sequence[int],
) -> np.ndarray:
    """Compose teacher-grid to runtime-grid affine with pixel-center scaling."""

    rgb_height, rgb_width = _validate_hw("rgb_output_hw", rgb_output_hw)
    output_height, output_width = _validate_hw("output_hw", output_hw)
    teacher_matrix = _homogeneous(teacher_affine, "teacher_affine")
    rgb_matrix = _homogeneous(rgb_affine, "rgb_affine")
    scale_x = output_width / float(rgb_width)
    scale_y = output_height / float(rgb_height)
    rgb_to_output = np.asarray(
        (
            (scale_x, 0.0, 0.5 * scale_x - 0.5),
            (0.0, scale_y, 0.5 * scale_y - 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    composed = rgb_to_output @ rgb_matrix @ np.linalg.inv(teacher_matrix)
    return composed[:2].astype(np.float32)


def _validity_aware_warp(
    values: np.ndarray,
    valid: np.ndarray,
    affine: np.ndarray,
    output_hw: tuple[int, int],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = output_hw
    valid_float = np.asarray(valid, dtype=np.float32)
    weight = cv2.warpAffine(
        valid_float,
        affine,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        values = values[..., None]
    warped = np.empty((height, width, values.shape[-1]), dtype=np.float32)
    denominator = np.maximum(weight, float(epsilon))
    for channel in range(values.shape[-1]):
        numerator = cv2.warpAffine(
            values[..., channel] * valid_float,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        warped[..., channel] = numerator / denominator
    warped[weight <= epsilon] = 0.0
    return warped, weight


def deterministic_spatial_permutation(
    sample_uid: str,
    seed: int,
    token_count: int,
) -> np.ndarray:
    """Return a stable permutation independent of Python hash randomization."""

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    material = f"{int(seed)}\0{sample_uid}".encode("utf-8")
    rng_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return np.random.default_rng(rng_seed).permutation(token_count)


def warp_pointnormal(
    point: np.ndarray,
    normal: np.ndarray,
    valid: np.ndarray,
    teacher_affine: np.ndarray,
    rgb_affine: np.ndarray,
    rgb_output_hw: Sequence[int],
    output_hw: Sequence[int] = (32, 24),
    flip_left_to_right: bool = False,
    sample_uid: str | None = None,
    spatial_shuffle_seed: int | None = None,
    validity_threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Warp one geometry record into an aligned point-normal feature grid.

    ``flip_left_to_right`` must match the canonical flip applied to the RGB crop.
    Supplying ``spatial_shuffle_seed`` enables deterministic post-warp shuffle and
    requires ``sample_uid``.
    """

    output_height, output_width = _validate_hw("output_hw", output_hw)
    point_array = np.asarray(point, dtype=np.float32)
    normal_array = np.asarray(normal, dtype=np.float32)
    valid_array = np.asarray(valid, dtype=bool)
    if point_array.ndim != 3 or point_array.shape[-1] != 3:
        raise ValueError(f"point must have shape [H,W,3], got {point_array.shape}")
    if normal_array.shape != point_array.shape:
        raise ValueError(
            f"normal must match point shape {point_array.shape}, got {normal_array.shape}"
        )
    if valid_array.shape != point_array.shape[:2]:
        raise ValueError(
            f"valid must have shape {point_array.shape[:2]}, got {valid_array.shape}"
        )
    if not 0.0 <= validity_threshold <= 1.0:
        raise ValueError("validity_threshold must be in [0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    finite = np.isfinite(point_array).all(axis=-1) & np.isfinite(normal_array).all(axis=-1)
    source_valid = valid_array & finite
    safe_point = np.where(source_valid[..., None], point_array, 0.0)
    safe_normal = np.where(source_valid[..., None], normal_array, 0.0)
    affine = compose_teacher_to_output_affine(
        teacher_affine=teacher_affine,
        rgb_affine=rgb_affine,
        rgb_output_hw=rgb_output_hw,
        output_hw=(output_height, output_width),
    )
    warped_point, point_weight = _validity_aware_warp(
        safe_point,
        source_valid,
        affine,
        (output_height, output_width),
        epsilon,
    )
    warped_normal, normal_weight = _validity_aware_warp(
        safe_normal,
        source_valid,
        affine,
        (output_height, output_width),
        epsilon,
    )
    output_valid = (
        (point_weight >= validity_threshold)
        & (normal_weight >= validity_threshold)
        & np.isfinite(warped_point).all(axis=-1)
        & np.isfinite(warped_normal).all(axis=-1)
    )

    normal_norm = np.linalg.norm(warped_normal, axis=-1, keepdims=True)
    output_valid &= normal_norm[..., 0] > epsilon
    warped_normal = warped_normal / np.maximum(normal_norm, epsilon)

    centered = warped_point.copy()
    if output_valid.any():
        center = np.median(centered[output_valid], axis=0)
        centered -= center
        radius_values = np.linalg.norm(centered[output_valid], axis=-1)
        scale = max(float(np.percentile(radius_values, 95)), epsilon)
        centered /= scale
    else:
        centered.fill(0.0)
    centered[~output_valid] = 0.0
    warped_normal[~output_valid] = 0.0

    if flip_left_to_right:
        centered = centered[:, ::-1].copy()
        warped_normal = warped_normal[:, ::-1].copy()
        output_valid = output_valid[:, ::-1].copy()
        centered[..., 0] *= -1.0
        warped_normal[..., 0] *= -1.0

    radius = np.linalg.norm(centered, axis=-1)
    features = np.concatenate(
        (
            centered,
            radius[..., None],
            output_valid.astype(np.float32)[..., None],
            warped_normal,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    features[~output_valid] = 0.0

    if spatial_shuffle_seed is not None:
        if sample_uid is None or not str(sample_uid):
            raise ValueError("sample_uid is required when spatial shuffle is enabled")
        permutation = deterministic_spatial_permutation(
            str(sample_uid), int(spatial_shuffle_seed), output_height * output_width
        )
        features = features.reshape(-1, len(POINTNORMAL_CHANNELS))[permutation].reshape(
            output_height, output_width, len(POINTNORMAL_CHANNELS)
        )
    return np.moveaxis(features, -1, 0).copy()


def warp_record_pointnormal(
    record: GeometryRecord,
    rgb_affine: np.ndarray,
    rgb_output_hw: Sequence[int],
    output_hw: Sequence[int] = (32, 24),
    flip_left_to_right: bool = False,
    spatial_shuffle_seed: int | None = None,
    validity_threshold: float = 0.5,
) -> np.ndarray:
    return warp_pointnormal(
        point=record.point,
        normal=record.normal,
        valid=record.valid,
        teacher_affine=record.teacher_affine,
        rgb_affine=rgb_affine,
        rgb_output_hw=rgb_output_hw,
        output_hw=output_hw,
        flip_left_to_right=flip_left_to_right,
        sample_uid=record.sample_uid,
        spatial_shuffle_seed=spatial_shuffle_seed,
        validity_threshold=validity_threshold,
    )


def discover_sidecars(paths: Iterable[os.PathLike[str] | str]) -> list[Path]:
    discovered: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            discovered.update(candidate for candidate in path.rglob("*.h5") if candidate.is_file())
        elif path.is_file():
            discovered.add(path)
        else:
            raise FileNotFoundError(path)
    return sorted(discovered)


def load_expected_manifest(path: os.PathLike[str] | str) -> Iterator[dict[str, Any]]:
    manifest_path = Path(path)
    if manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest_path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {manifest_path}:{line_number}")
            yield value


def validate_coverage(
    sidecar_paths: Iterable[os.PathLike[str] | str],
    expected_records: Iterable[Mapping[str, Any]] | None = None,
    expected_model_sha256: str | None = None,
    expected_config_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    deep: bool = False,
    max_issue_examples: int = 100,
) -> CoverageReport:
    """Validate shard metadata, identities, optional manifest coverage, and arrays."""

    report = CoverageReport()
    expected_rows = (
        None if expected_records is None else [dict(row) for row in expected_records]
    )
    expected_sequences = {
        str(row.get("sequence_key", "")).strip()
        for row in (expected_rows or ())
        if str(row.get("sequence_key", "")).strip()
    }

    def issue(message: str) -> None:
        if len(report.issues) < max_issue_examples:
            report.issues.append(message)

    actual: dict[str, tuple[str, int, Path]] = {}
    seen_sequences: set[str] = set()
    common_hashes: dict[str, str] = {}
    for path in discover_sidecars(sidecar_paths):
        try:
            reader = SequenceSidecarReader(
                path,
                expected_model_sha256=expected_model_sha256,
                expected_config_sha256=expected_config_sha256,
            )
        except Exception as exc:
            report.invalid_record_count += 1
            issue(f"{path}: {exc}")
            continue
        with reader:
            # One shared root may hold train/val/test shards. When a manifest is
            # supplied, validate only the sequences owned by that manifest.
            if expected_sequences and reader.sequence_key not in expected_sequences:
                continue
            if (
                expected_manifest_sha256 is not None
                and reader.config.manifest_sha256
                != _validate_sha256("manifest_sha256", expected_manifest_sha256)
            ):
                report.invalid_record_count += 1
                issue(
                    f"{path}: manifest_sha256 mismatch: "
                    f"expected={expected_manifest_sha256}, "
                    f"actual={reader.config.manifest_sha256}"
                )
                continue
            report.shard_count += 1
            report.record_count += reader.record_count
            if reader.sequence_key in seen_sequences:
                report.duplicate_sequence_count += 1
                issue(f"Duplicate sequence shard: {reader.sequence_key}")
            seen_sequences.add(reader.sequence_key)
            for name, value in (
                ("model_sha256", reader.config.model_sha256),
                ("config_sha256", reader.config.config_sha256),
                ("manifest_sha256", reader.config.manifest_sha256),
            ):
                if name in common_hashes and common_hashes[name] != value:
                    issue(
                        f"Inconsistent {name}: {path} has {value}, "
                        f"expected {common_hashes[name]}"
                    )
                common_hashes.setdefault(name, value)
            for uid, query_row in reader.iter_identity():
                if uid in actual:
                    report.duplicate_uid_count += 1
                    issue(f"Duplicate sample_uid across shards: {uid}")
                else:
                    actual[uid] = (reader.sequence_key, query_row, path)
                if deep:
                    try:
                        record = reader.read(sample_uid=uid, query_row=query_row)
                        finite_valid = (
                            np.isfinite(record.point[record.valid]).all()
                            and np.isfinite(record.normal[record.valid]).all()
                            and np.isfinite(record.teacher_affine).all()
                        )
                        if not finite_valid:
                            raise ValueError("non-finite geometry in valid cells")
                    except Exception as exc:
                        report.invalid_record_count += 1
                        issue(f"Invalid record {uid} in {path}: {exc}")

    report.model_sha256 = common_hashes.get("model_sha256")
    report.config_sha256 = common_hashes.get("config_sha256")
    report.manifest_sha256 = common_hashes.get("manifest_sha256")

    if expected_rows is not None:
        expected: dict[str, tuple[str | None, int | None]] = {}
        for row in expected_rows:
            uid = str(row.get("sample_uid", "")).strip()
            if not uid:
                report.invalid_record_count += 1
                issue("Expected manifest row is missing sample_uid")
                continue
            sequence_key = row.get("sequence_key")
            query_row = row.get("query_row")
            identity = (
                None if sequence_key in (None, "") else str(sequence_key),
                None if query_row in (None, "") else int(query_row),
            )
            if uid in expected:
                report.duplicate_uid_count += 1
                issue(f"Duplicate sample_uid in expected manifest: {uid}")
            expected[uid] = identity
        report.expected_count = len(expected)
        missing = expected.keys() - actual.keys()
        unexpected = actual.keys() - expected.keys()
        report.missing_count = len(missing)
        report.unexpected_count = len(unexpected)
        for uid in sorted(missing)[:max_issue_examples]:
            issue(f"Missing sidecar record: {uid}")
        for uid in sorted(unexpected)[:max_issue_examples]:
            issue(f"Unexpected sidecar record: {uid}")
        for uid in expected.keys() & actual.keys():
            expected_sequence, expected_query_row = expected[uid]
            actual_sequence, actual_query_row, _ = actual[uid]
            if expected_sequence is not None and expected_sequence != actual_sequence:
                report.query_row_mismatch_count += 1
                issue(
                    f"Sequence mismatch for {uid}: expected={expected_sequence}, "
                    f"actual={actual_sequence}"
                )
            if expected_query_row is not None and expected_query_row != actual_query_row:
                report.query_row_mismatch_count += 1
                issue(
                    f"query_row mismatch for {uid}: expected={expected_query_row}, "
                    f"actual={actual_query_row}"
                )
    return report


def _record_summary(record: GeometryRecord) -> dict[str, Any]:
    valid_count = int(record.valid.sum())
    result: dict[str, Any] = {
        "sample_uid": record.sample_uid,
        "query_row": record.query_row,
        "point_shape": list(record.point.shape),
        "normal_shape": list(record.normal.shape),
        "valid_count": valid_count,
        "valid_fraction": float(record.valid.mean()),
        "teacher_affine": record.teacher_affine.tolist(),
    }
    if valid_count:
        point = record.point[record.valid]
        normal = record.normal[record.valid]
        result.update(
            {
                "point_min": point.min(axis=0).tolist(),
                "point_max": point.max(axis=0).tolist(),
                "normal_norm_mean": float(np.linalg.norm(normal, axis=-1).mean()),
            }
        )
    return result


def _command_validate(args: argparse.Namespace) -> int:
    expected = None if args.manifest is None else load_expected_manifest(args.manifest)
    expected_manifest_sha256 = args.manifest_sha256
    if args.manifest is not None and expected_manifest_sha256 is None:
        expected_manifest_sha256 = _sha256_file(args.manifest)
    report = validate_coverage(
        args.paths,
        expected_records=expected,
        expected_model_sha256=args.model_sha256,
        expected_config_sha256=args.config_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        deep=args.deep,
        max_issue_examples=args.max_issues,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


def _command_inspect(args: argparse.Namespace) -> int:
    with SequenceSidecarReader(args.path) as reader:
        payload: dict[str, Any] = {
            "path": str(reader.path),
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "sequence_key": reader.sequence_key,
            "record_count": reader.record_count,
            "config": reader.config.to_dict(),
        }
        if args.sample_uid is not None or args.query_row is not None:
            payload["record"] = _record_summary(
                reader.read(sample_uid=args.sample_uid, query_row=args.query_row)
            )
        elif reader.record_count:
            first_uid, first_query_row = next(reader.iter_identity())
            payload["first_record"] = _record_summary(
                reader.read(sample_uid=first_uid, query_row=first_query_row)
            )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="validate sidecar shards and optional manifest coverage"
    )
    validate_parser.add_argument("paths", nargs="+", help="sidecar HDF5 files or roots")
    validate_parser.add_argument("--manifest", help="expected JSONL or CSV sample manifest")
    validate_parser.add_argument("--model-sha256")
    validate_parser.add_argument("--config-sha256")
    validate_parser.add_argument("--manifest-sha256")
    validate_parser.add_argument("--deep", action="store_true", help="read and check every record")
    validate_parser.add_argument("--max-issues", type=int, default=100)
    validate_parser.set_defaults(func=_command_validate)

    inspect_parser = subparsers.add_parser("inspect", help="inspect one sidecar shard")
    inspect_parser.add_argument("path")
    selector = inspect_parser.add_mutually_exclusive_group()
    selector.add_argument("--sample-uid")
    selector.add_argument("--query-row", type=int)
    inspect_parser.set_defaults(func=_command_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
