#!/usr/bin/env python3
"""Atomic, resumable, fixed-shape feature caches for tactile input priors.

The cache deliberately uses plain ``.npy`` files so workers can mmap fixed
shape arrays without HDF5 locks or decompression.  A shard becomes visible in
three steps: write a private ``.partial`` directory, atomically rename it, then
publish ``DONE.json``.  Readers accept only shards referenced by a finalized
root manifest.

This module is model-agnostic.  A caller supplies a callback mapping one source
manifest row to any configured subset of the supported feature fields.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import time
import uuid
from urllib.parse import quote
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


CACHE_SCHEMA = "tactile_feature_cache_v1"
CACHE_SCHEMA_VERSION = "1.0.0"
SHARD_SCHEMA = "tactile_feature_cache_shard_v1"
ALLOWED_FEATURE_FIELDS = (
    "z_rgb",
    "h_rgb",
    "base_logits",
    "palm_base_logits",
    "contact_neck",
    "contact_anchor_logits",
    "contact_logits",
    "depth_grid",
    "vlm_embedding",
    "tactile_signal",
    "palm_tactile_signal",
    "palm_mask",
    "has_tactile",
)
CONFIG_FILE = "cache_config.json"
CONFIG_SHA_FILE = "cache_config.sha256"
CACHE_MANIFEST_FILE = "cache_manifest.json"
CACHE_DONE_FILE = "CACHE_DONE.json"
SAMPLE_INDEX_FILE = "sample_index.sqlite3"


class FeatureCacheError(RuntimeError):
    """Base exception for malformed, incompatible, or incomplete caches."""


class FeatureCacheMismatchError(FeatureCacheError):
    """Raised when semantic cache provenance differs from the expectation."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 8 * 1024 * 1024) -> str:
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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> None:
    _atomic_write_text(
        Path(path),
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
    )


def _normalize_dtype(dtype: Any) -> np.dtype:
    result = np.dtype(dtype)
    if result.hasobject or result.kind in {"O", "S", "U", "V"}:
        raise ValueError(f"Feature cache dtype must be numeric/bool, got {result}")
    if result.byteorder not in {"=", "|"}:
        result = result.newbyteorder("=")
    return result


@dataclass(frozen=True)
class FeatureSpec:
    """One fixed-shape field stored once per sample."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype

    def __init__(self, name: str, shape: Sequence[int], dtype: Any):
        clean_name = str(name).strip()
        if clean_name not in ALLOWED_FEATURE_FIELDS:
            raise ValueError(
                f"Unsupported feature field {clean_name!r}; allowed={ALLOWED_FEATURE_FIELDS}"
            )
        clean_shape = tuple(int(item) for item in shape)
        if not clean_shape or any(item <= 0 for item in clean_shape):
            raise ValueError(f"Feature {clean_name!r} needs a positive non-empty shape")
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(self, "shape", clean_shape)
        object.__setattr__(self, "dtype", _normalize_dtype(dtype))

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype.str,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "FeatureSpec":
        return cls(value["name"], value["shape"], value["dtype"])


def normalize_feature_specs(specs: Iterable[FeatureSpec | Mapping[str, Any]]) -> tuple[FeatureSpec, ...]:
    normalized = tuple(
        item if isinstance(item, FeatureSpec) else FeatureSpec.from_json(item)
        for item in specs
    )
    if not normalized:
        raise ValueError("At least one feature field is required")
    names = [item.name for item in normalized]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate feature fields: {names}")
    return tuple(sorted(normalized, key=lambda item: item.name))


def _sample_id(value: Any, *, location: str) -> str:
    if value is None:
        raise ValueError(f"{location}: sample ID is missing")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{location}: sample ID is empty")
    if "\x00" in result:
        raise ValueError(f"{location}: sample ID contains NUL")
    return result


@dataclass(frozen=True)
class SourceManifestInfo:
    path: Path
    sha256: str
    sample_count: int
    sample_ids_sha256: str
    sample_id_key: str

    def semantic_json(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "sample_count": self.sample_count,
            "sample_ids_sha256": self.sample_ids_sha256,
            "sample_id_key": self.sample_id_key,
        }


def iter_jsonl(path: os.PathLike[str] | str) -> Iterator[tuple[int, dict[str, Any]]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, row


def inspect_source_manifest(
    path: os.PathLike[str] | str,
    sample_id_key: str = "sample_id",
) -> SourceManifestInfo:
    path = Path(path).expanduser().resolve(strict=True)
    ids_digest = hashlib.sha256()
    count = 0
    for line_number, row in iter_jsonl(path):
        sample_id = _sample_id(
            row.get(sample_id_key),
            location=f"{path}:{line_number}:{sample_id_key}",
        )
        ids_digest.update(sample_id.encode("utf-8"))
        ids_digest.update(b"\n")
        count += 1
    if count == 0:
        raise ValueError(f"Source manifest is empty: {path}")
    return SourceManifestInfo(
        path=path,
        sha256=sha256_file(path),
        sample_count=count,
        sample_ids_sha256=ids_digest.hexdigest(),
        sample_id_key=str(sample_id_key),
    )


def _semantic_config(
    source: SourceManifestInfo,
    specs: Sequence[FeatureSpec],
    shard_size: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = json.loads(canonical_json(dict(provenance)))
    return {
        "schema": CACHE_SCHEMA,
        "schema_version": CACHE_SCHEMA_VERSION,
        "source_manifest": source.semantic_json(),
        "features": [item.to_json() for item in specs],
        "shard_size": int(shard_size),
        "provenance": provenance,
        "provenance_sha256": sha256_json(provenance),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureCacheError(f"Could not read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FeatureCacheError(f"Expected a JSON object in {path}")
    return value


def _read_and_verify_config(cache_dir: Path) -> tuple[dict[str, Any], str]:
    config_path = cache_dir / CONFIG_FILE
    sha_path = cache_dir / CONFIG_SHA_FILE
    if not config_path.is_file() or not sha_path.is_file():
        raise FeatureCacheError(f"Incomplete cache configuration under {cache_dir}")
    actual_sha = sha256_file(config_path)
    expected_sha = sha_path.read_text(encoding="ascii").strip()
    if actual_sha != expected_sha:
        raise FeatureCacheError(
            f"Cache config SHA mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    config = _read_json(config_path)
    if config.get("schema") != CACHE_SCHEMA or config.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise FeatureCacheError(
            f"Unsupported feature cache schema: {config.get('schema')!r} "
            f"version={config.get('schema_version')!r}"
        )
    if sha256_json(config.get("provenance", {})) != config.get("provenance_sha256"):
        raise FeatureCacheError("Embedded provenance SHA does not match cache config")
    return config, actual_sha


class CacheBuildLock:
    """Atomic directory lock suitable for a shared filesystem.

    Dead locks from the same host are reclaimed only when their owner PID no
    longer exists.  Cross-host locks are never silently removed.
    """

    def __init__(
        self,
        cache_dir: Path,
        timeout_seconds: float = 600.0,
        poll_seconds: float = 2.0,
        break_stale_lock: bool = False,
    ):
        self.path = cache_dir / ".build.lock"
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self.poll_seconds = max(0.05, float(poll_seconds))
        self.break_stale_lock = bool(break_stale_lock)
        self.acquired = False

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _may_reclaim(self) -> bool:
        owner_path = self.path / "owner.json"
        try:
            owner = _read_json(owner_path)
        except FeatureCacheError:
            return self.break_stale_lock
        if owner.get("hostname") == socket.gethostname():
            try:
                return not self._pid_alive(int(owner["pid"]))
            except (KeyError, TypeError, ValueError):
                return self.break_stale_lock
        return self.break_stale_lock

    def __enter__(self) -> "CacheBuildLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self.path.mkdir()
            except FileExistsError:
                if self._may_reclaim():
                    try:
                        shutil.rmtree(self.path)
                    except FileNotFoundError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    owner = self.path / "owner.json"
                    try:
                        detail = owner.read_text(encoding="utf-8")
                    except OSError:
                        detail = "unknown (lock changed while reporting timeout)"
                    raise FeatureCacheError(
                        f"Timed out waiting for feature-cache build lock {self.path}; "
                        f"owner={detail.strip()}"
                    ) from None
                time.sleep(self.poll_seconds)
                continue
            owner = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_unix": time.time(),
                "token": uuid.uuid4().hex,
            }
            atomic_write_json(self.path / "owner.json", owner)
            self.acquired = True
            return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)
            self.acquired = False
        return False


FeatureCallback = Callable[[Mapping[str, Any], int], Mapping[str, Any]]


def _as_numpy(value: Any, spec: FeatureSpec, *, location: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    array = np.asarray(value)
    if array.shape != spec.shape:
        raise ValueError(
            f"{location}: feature {spec.name!r} shape={array.shape}, expected={spec.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise ValueError(f"{location}: feature {spec.name!r} is not numeric")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{location}: feature {spec.name!r} contains non-finite values")
    return np.asarray(array, dtype=spec.dtype, order="C")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            payload = (canonical_json(dict(row)) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count, digest.hexdigest()


def _shard_name(index: int) -> str:
    return f"shard-{int(index):06d}"


def _shard_paths(cache_dir: Path, index: int) -> tuple[Path, Path]:
    final = cache_dir / "shards" / _shard_name(index)
    done = final / "DONE.json"
    return final, done


def _expected_npy_shape(spec: FeatureSpec, count: int) -> tuple[int, ...]:
    return (int(count), *spec.shape)


def validate_shard(
    cache_dir: os.PathLike[str] | str,
    shard_index: int,
    *,
    config_sha256: str,
    specs: Sequence[FeatureSpec],
    expected_count: int | None = None,
    deep: bool = False,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    shard_dir, done_path = _shard_paths(cache_dir, shard_index)
    if not done_path.is_file():
        raise FeatureCacheError(f"Shard is not finalized: {shard_dir}")
    done = _read_json(done_path)
    if done.get("schema") != SHARD_SCHEMA:
        raise FeatureCacheError(f"Invalid shard schema in {done_path}")
    if done.get("config_sha256") != config_sha256:
        raise FeatureCacheMismatchError(f"Shard config mismatch in {done_path}")
    count = int(done.get("sample_count", -1))
    if count <= 0 or (expected_count is not None and count != int(expected_count)):
        raise FeatureCacheError(
            f"Shard sample count mismatch in {done_path}: {count} vs {expected_count}"
        )
    samples_path = shard_dir / "samples.jsonl"
    metadata_path = shard_dir / "shard.json"
    if sha256_file(samples_path) != done.get("samples_sha256"):
        raise FeatureCacheError(f"Sample table SHA mismatch in {shard_dir}")
    if sha256_file(metadata_path) != done.get("metadata_sha256"):
        raise FeatureCacheError(f"Shard metadata SHA mismatch in {shard_dir}")
    field_hashes = done.get("field_sha256", {})
    for spec in specs:
        path = shard_dir / f"{spec.name}.npy"
        if not path.is_file():
            raise FeatureCacheError(f"Missing feature field {path}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise FeatureCacheError(f"Could not mmap {path}: {exc}") from exc
        if array.shape != _expected_npy_shape(spec, count) or array.dtype != spec.dtype:
            raise FeatureCacheError(
                f"Field contract mismatch for {path}: shape={array.shape}, dtype={array.dtype}; "
                f"expected={_expected_npy_shape(spec, count)}, {spec.dtype}"
            )
        del array
        if deep and sha256_file(path) != field_hashes.get(spec.name):
            raise FeatureCacheError(f"Feature data SHA mismatch for {path}")
    return done


class FeatureCacheBuilder:
    """Build or resume an atomic sharded feature cache."""

    def __init__(
        self,
        cache_dir: os.PathLike[str] | str,
        source_manifest: os.PathLike[str] | str,
        feature_specs: Iterable[FeatureSpec | Mapping[str, Any]],
        *,
        provenance: Mapping[str, Any] | None = None,
        shard_size: int = 4096,
        sample_id_key: str = "sample_id",
        lock_timeout_seconds: float = 600.0,
        break_stale_lock: bool = False,
        deep_verify_existing: bool = False,
        repair_invalid_shards: bool = False,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve(strict=False)
        self.source = inspect_source_manifest(source_manifest, sample_id_key)
        self.specs = normalize_feature_specs(feature_specs)
        self.shard_size = int(shard_size)
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.provenance = dict(provenance or {})
        self.config = _semantic_config(
            self.source,
            self.specs,
            self.shard_size,
            self.provenance,
        )
        self.config_sha256 = sha256_bytes(
            (json.dumps(self.config, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.break_stale_lock = bool(break_stale_lock)
        self.deep_verify_existing = bool(deep_verify_existing)
        self.repair_invalid_shards = bool(repair_invalid_shards)

    @property
    def shard_count(self) -> int:
        return (self.source.sample_count + self.shard_size - 1) // self.shard_size

    def _expected_shard_count(self, shard_index: int) -> int:
        start = shard_index * self.shard_size
        return min(self.shard_size, self.source.sample_count - start)

    def _ensure_config(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.cache_dir / CONFIG_FILE
        sha_path = self.cache_dir / CONFIG_SHA_FILE
        expected_text = (
            json.dumps(self.config, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        )
        expected_sha = sha256_bytes(expected_text.encode("utf-8"))
        if config_path.exists():
            existing = _read_json(config_path)
            if canonical_json(existing) != canonical_json(self.config):
                raise FeatureCacheMismatchError(
                    f"Cache configuration/provenance differs under {self.cache_dir}. "
                    "Use a new cache directory."
                )
            actual_sha = sha256_file(config_path)
            if actual_sha != expected_sha:
                raise FeatureCacheError(
                    f"Existing config bytes are non-canonical or damaged: {config_path}"
                )
            if sha_path.exists() and sha_path.read_text(encoding="ascii").strip() != actual_sha:
                raise FeatureCacheError(f"Existing config SHA sidecar is invalid: {sha_path}")
            if not sha_path.exists():
                _atomic_write_text(sha_path, actual_sha + "\n")
        else:
            _atomic_write_text(config_path, expected_text)
            _atomic_write_text(sha_path, expected_sha + "\n")
        self.config_sha256 = expected_sha

    def _clean_orphan_partials(self) -> None:
        shards_dir = self.cache_dir / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        for path in shards_dir.glob("*.partial.*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

    def _reuse_shard(self, shard_index: int) -> bool:
        final_dir, done_path = _shard_paths(self.cache_dir, shard_index)
        if not final_dir.exists():
            return False
        if not done_path.is_file():
            shutil.rmtree(final_dir)
            return False
        try:
            validate_shard(
                self.cache_dir,
                shard_index,
                config_sha256=self.config_sha256,
                specs=self.specs,
                expected_count=self._expected_shard_count(shard_index),
                deep=self.deep_verify_existing,
            )
        except FeatureCacheError:
            if not self.repair_invalid_shards:
                raise
            shutil.rmtree(final_dir)
            return False
        return True

    def _write_shard(
        self,
        shard_index: int,
        rows: Sequence[tuple[int, int, Mapping[str, Any]]],
        callback: FeatureCallback,
    ) -> None:
        final_dir, _ = _shard_paths(self.cache_dir, shard_index)
        partial_dir = final_dir.with_name(
            f"{final_dir.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
        )
        partial_dir.mkdir(parents=False)
        arrays: dict[str, np.memmap] = {}
        try:
            for spec in self.specs:
                arrays[spec.name] = np.lib.format.open_memmap(
                    partial_dir / f"{spec.name}.npy",
                    mode="w+",
                    dtype=spec.dtype,
                    shape=_expected_npy_shape(spec, len(rows)),
                )
            sample_rows = []
            for shard_row, (source_index, line_number, row) in enumerate(rows):
                sample_id = _sample_id(
                    row.get(self.source.sample_id_key),
                    location=f"{self.source.path}:{line_number}:{self.source.sample_id_key}",
                )
                produced = callback(row, source_index)
                if not isinstance(produced, Mapping):
                    raise TypeError(
                        f"Feature callback for sample {sample_id!r} returned "
                        f"{type(produced).__name__}, expected a mapping"
                    )
                for spec in self.specs:
                    if spec.name not in produced:
                        raise KeyError(
                            f"Feature callback omitted {spec.name!r} for sample {sample_id!r}"
                        )
                    arrays[spec.name][shard_row] = _as_numpy(
                        produced[spec.name],
                        spec,
                        location=f"sample={sample_id}",
                    )
                sample_entry = {
                    "sample_id": sample_id,
                    "source_index": source_index,
                    "source_line": line_number,
                }
                metadata = row.get("metadata")
                if metadata is not None:
                    if not isinstance(metadata, Mapping):
                        raise TypeError(
                            f"sample={sample_id}: source metadata must be a mapping"
                        )
                    sample_entry["metadata"] = json.loads(canonical_json(dict(metadata)))
                sample_rows.append(sample_entry)
            for name, array in list(arrays.items()):
                array.flush()
                del arrays[name]
                path = partial_dir / f"{name}.npy"
                descriptor = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            samples_count, samples_sha = _write_jsonl(
                partial_dir / "samples.jsonl", sample_rows
            )
            if samples_count != len(rows):
                raise AssertionError("Shard sample table count changed while writing")
            metadata = {
                "schema": SHARD_SCHEMA,
                "shard_index": shard_index,
                "sample_count": len(rows),
                "source_start": rows[0][0],
                "source_stop": rows[-1][0] + 1,
                "config_sha256": self.config_sha256,
                "features": [item.to_json() for item in self.specs],
            }
            metadata_path = partial_dir / "shard.json"
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            descriptor = os.open(metadata_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            field_hashes = {
                spec.name: sha256_file(partial_dir / f"{spec.name}.npy")
                for spec in self.specs
            }
            done = {
                "schema": SHARD_SCHEMA,
                "shard_index": shard_index,
                "sample_count": len(rows),
                "config_sha256": self.config_sha256,
                "samples_sha256": samples_sha,
                "metadata_sha256": sha256_file(metadata_path),
                "field_sha256": field_hashes,
            }
            _fsync_directory(partial_dir)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(partial_dir, final_dir)
            _fsync_directory(final_dir.parent)
            atomic_write_json(final_dir / "DONE.json", done)
        except BaseException:
            for name in list(arrays):
                try:
                    arrays[name].flush()
                except Exception:
                    pass
                del arrays[name]
            shutil.rmtree(partial_dir, ignore_errors=True)
            raise

    def _build_sample_index(self, shard_entries: Sequence[Mapping[str, Any]]) -> str:
        final_path = self.cache_dir / SAMPLE_INDEX_FILE
        partial_path = final_path.with_name(
            f".{final_path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}"
        )
        connection = sqlite3.connect(partial_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=MEMORY")
            connection.execute(
                "CREATE TABLE samples ("
                "sample_id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL UNIQUE, "
                "shard INTEGER NOT NULL, row_in_shard INTEGER NOT NULL, "
                "sample_ref TEXT NOT NULL, dataset TEXT NOT NULL, "
                "sequence_key TEXT NOT NULL, query_alias TEXT NOT NULL, "
                "frame_idx INTEGER NOT NULL, source_frame_idx INTEGER, "
                "timestamp REAL, is_right INTEGER NOT NULL, bbox_xyxy TEXT NOT NULL, "
                "bbox_association_id TEXT NOT NULL"
                ") WITHOUT ROWID"
            )
            batch = []
            for entry in shard_entries:
                shard_index = int(entry["shard_index"])
                samples_path = (
                    self.cache_dir / "shards" / _shard_name(shard_index) / "samples.jsonl"
                )
                for _, row in iter_jsonl(samples_path):
                    metadata = row.get("metadata")
                    metadata = metadata if isinstance(metadata, Mapping) else {}
                    batch.append(
                        (
                            _sample_id(row.get("sample_id"), location=str(samples_path)),
                            int(row["source_index"]),
                            shard_index,
                            int(row["source_index"]) - int(entry["source_start"]),
                            str(metadata.get("sample_ref", "")),
                            str(metadata.get("dataset", "")),
                            str(metadata.get("sequence_key", "")),
                            str(metadata.get("query_alias", "query")),
                            int(metadata.get("frame_idx", 0)),
                            (
                                None
                                if metadata.get("source_frame_idx") is None
                                else int(metadata["source_frame_idx"])
                            ),
                            (
                                None
                                if metadata.get("timestamp") is None
                                else float(metadata["timestamp"])
                            ),
                            int(metadata.get("is_right", 0)),
                            canonical_json(metadata.get("bbox_xyxy", ())),
                            str(metadata.get("bbox_association_id", "")),
                        )
                    )
                    if len(batch) >= 8192:
                        connection.executemany(
                            "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            batch,
                        )
                        batch.clear()
            if batch:
                connection.executemany(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", batch
                )
            count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            if count != self.source.sample_count:
                raise FeatureCacheError(
                    f"Global sample index has {count} rows, expected {self.source.sample_count}"
                )
            connection.commit()
        except sqlite3.IntegrityError as exc:
            raise FeatureCacheError(
                "Source manifest contains duplicate sample IDs or ordinals"
            ) from exc
        finally:
            connection.close()
        descriptor = os.open(partial_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(partial_path, final_path)
        _fsync_directory(final_path.parent)
        return sha256_file(final_path)

    def _finalize(self) -> dict[str, Any]:
        shard_entries = []
        for shard_index in range(self.shard_count):
            expected_count = self._expected_shard_count(shard_index)
            done = validate_shard(
                self.cache_dir,
                shard_index,
                config_sha256=self.config_sha256,
                specs=self.specs,
                expected_count=expected_count,
                deep=False,
            )
            source_start = shard_index * self.shard_size
            shard_entries.append(
                {
                    "shard_index": shard_index,
                    "sample_count": expected_count,
                    "source_start": source_start,
                    "source_stop": source_start + expected_count,
                    "done_sha256": sha256_file(
                        self.cache_dir
                        / "shards"
                        / _shard_name(shard_index)
                        / "DONE.json"
                    ),
                    "field_sha256": dict(done["field_sha256"]),
                }
            )
        sample_index_sha = self._build_sample_index(shard_entries)
        manifest = {
            "schema": CACHE_SCHEMA,
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_sha256": self.config_sha256,
            "sample_count": self.source.sample_count,
            "shard_count": self.shard_count,
            "sample_index": SAMPLE_INDEX_FILE,
            "sample_index_sha256": sample_index_sha,
            "shards": shard_entries,
        }
        manifest_path = self.cache_dir / CACHE_MANIFEST_FILE
        atomic_write_json(manifest_path, manifest)
        done = {
            "schema": CACHE_SCHEMA,
            "schema_version": CACHE_SCHEMA_VERSION,
            "config_sha256": self.config_sha256,
            "cache_manifest_sha256": sha256_file(manifest_path),
            "sample_count": self.source.sample_count,
        }
        atomic_write_json(self.cache_dir / CACHE_DONE_FILE, done)
        return manifest

    def build(
        self,
        callback: FeatureCallback,
        *,
        max_new_shards: int | None = None,
    ) -> dict[str, Any]:
        """Build missing shards and finalize the cache when all are present.

        ``max_new_shards`` exists mainly for bounded jobs and recovery tests.  A
        partial run returns a summary without publishing the root completion
        marker; a later call resumes from finalized shards.
        """

        max_new_shards = None if max_new_shards is None else int(max_new_shards)
        if max_new_shards is not None and max_new_shards < 0:
            raise ValueError("max_new_shards must be non-negative")
        with CacheBuildLock(
            self.cache_dir,
            timeout_seconds=self.lock_timeout_seconds,
            break_stale_lock=self.break_stale_lock,
        ):
            self._ensure_config()
            self._clean_orphan_partials()
            new_shards = 0
            completed = 0
            current_rows: list[tuple[int, int, Mapping[str, Any]]] = []
            current_shard = 0
            for source_index, (line_number, row) in enumerate(
                iter_jsonl(self.source.path)
            ):
                shard_index = source_index // self.shard_size
                if shard_index != current_shard:
                    raise AssertionError("Manifest shard grouping became non-contiguous")
                current_rows.append((source_index, line_number, row))
                is_end = len(current_rows) == self.shard_size
                is_last = source_index + 1 == self.source.sample_count
                if not (is_end or is_last):
                    continue
                if self._reuse_shard(current_shard):
                    completed += 1
                elif max_new_shards is None or new_shards < max_new_shards:
                    self._write_shard(current_shard, current_rows, callback)
                    new_shards += 1
                    completed += 1
                current_rows = []
                current_shard += 1
            if current_shard != self.shard_count:
                raise AssertionError(
                    f"Manifest produced {current_shard} shards, expected {self.shard_count}"
                )
            if completed == self.shard_count:
                manifest = self._finalize()
                return {
                    "complete": True,
                    "new_shards": new_shards,
                    "shard_count": self.shard_count,
                    "sample_count": self.source.sample_count,
                    "cache_manifest": manifest,
                }
            return {
                "complete": False,
                "new_shards": new_shards,
                "completed_shards": completed,
                "shard_count": self.shard_count,
                "sample_count": self.source.sample_count,
            }


class FeatureCacheDataset:
    """Read-only mmap dataset with integer and sample-ID lookup."""

    def __init__(
        self,
        cache_dir: os.PathLike[str] | str,
        *,
        fields: Sequence[str] | None = None,
        expected_manifest_sha256: str | None = None,
        expected_manifest_path: os.PathLike[str] | str | None = None,
        expected_provenance: Mapping[str, Any] | None = None,
        expected_provenance_sha256: str | None = None,
        expected_config_sha256: str | None = None,
        max_open_shards: int = 2,
        copy_arrays: bool = False,
        deep_verify: bool = False,
    ):
        self.cache_dir = Path(cache_dir).expanduser().resolve(strict=True)
        self.config, self.config_sha256 = _read_and_verify_config(self.cache_dir)
        if expected_config_sha256 and self.config_sha256 != expected_config_sha256:
            raise FeatureCacheMismatchError(
                f"Cache config SHA differs: {self.config_sha256} != {expected_config_sha256}"
            )
        source = self.config["source_manifest"]
        expected_manifest_sha256 = expected_manifest_sha256 or (
            sha256_file(Path(expected_manifest_path).expanduser().resolve(strict=True))
            if expected_manifest_path is not None
            else None
        )
        if expected_manifest_sha256 and source.get("sha256") != expected_manifest_sha256:
            raise FeatureCacheMismatchError(
                f"Source manifest SHA differs: cache={source.get('sha256')}, "
                f"expected={expected_manifest_sha256}"
            )
        if expected_provenance is not None:
            expected_sha = sha256_json(dict(expected_provenance))
            if expected_sha != self.config.get("provenance_sha256"):
                raise FeatureCacheMismatchError(
                    f"Provenance differs: cache={self.config.get('provenance_sha256')}, "
                    f"expected={expected_sha}"
                )
        if (
            expected_provenance_sha256
            and expected_provenance_sha256 != self.config.get("provenance_sha256")
        ):
            raise FeatureCacheMismatchError("Expected provenance SHA does not match cache")
        self.specs = normalize_feature_specs(self.config["features"])
        spec_by_name = {item.name: item for item in self.specs}
        self.fields = tuple(fields) if fields is not None else tuple(spec_by_name)
        unknown = sorted(set(self.fields) - set(spec_by_name))
        if unknown:
            raise FeatureCacheMismatchError(f"Requested fields are absent: {unknown}")
        self.spec_by_name = spec_by_name
        done_path = self.cache_dir / CACHE_DONE_FILE
        manifest_path = self.cache_dir / CACHE_MANIFEST_FILE
        if not done_path.is_file() or not manifest_path.is_file():
            raise FeatureCacheError(f"Feature cache is not finalized: {self.cache_dir}")
        done = _read_json(done_path)
        if done.get("config_sha256") != self.config_sha256:
            raise FeatureCacheMismatchError("Root completion marker config SHA differs")
        if sha256_file(manifest_path) != done.get("cache_manifest_sha256"):
            raise FeatureCacheError("Root cache manifest SHA mismatch")
        self.manifest = _read_json(manifest_path)
        if self.manifest.get("config_sha256") != self.config_sha256:
            raise FeatureCacheMismatchError("Root cache manifest config SHA differs")
        self.sample_count = int(self.manifest["sample_count"])
        self.shards = tuple(self.manifest["shards"])
        if sum(int(item["sample_count"]) for item in self.shards) != self.sample_count:
            raise FeatureCacheError("Shard counts do not sum to cache sample count")
        self.shard_stops = tuple(int(item["source_stop"]) for item in self.shards)
        sample_index = self.cache_dir / self.manifest["sample_index"]
        if not sample_index.is_file():
            raise FeatureCacheError(f"Missing sample ID index: {sample_index}")
        if sha256_file(sample_index) != self.manifest.get("sample_index_sha256"):
            raise FeatureCacheError("Sample ID index SHA mismatch")
        self.sample_index_path = sample_index
        self.max_open_shards = max(1, int(max_open_shards))
        self.copy_arrays = bool(copy_arrays)
        self._open_shards: OrderedDict[
            int, tuple[dict[str, np.ndarray], list[dict[str, Any]] | None]
        ] = OrderedDict()
        self._index_connection: sqlite3.Connection | None = None
        self._index_has_metadata_cache: bool | None = None
        self._index_has_temporal_metadata_cache: bool | None = None
        for entry in self.shards:
            shard_index = int(entry["shard_index"])
            done_path = (
                self.cache_dir / "shards" / _shard_name(shard_index) / "DONE.json"
            )
            if sha256_file(done_path) != entry.get("done_sha256"):
                raise FeatureCacheError(
                    f"Root manifest shard marker SHA mismatch: {done_path}"
                )
            shard_done = validate_shard(
                self.cache_dir,
                shard_index,
                config_sha256=self.config_sha256,
                specs=self.specs,
                expected_count=int(entry["sample_count"]),
                deep=deep_verify,
            )
            if dict(shard_done.get("field_sha256", {})) != dict(
                entry.get("field_sha256", {})
            ):
                raise FeatureCacheError(
                    f"Root manifest field hashes differ for shard {shard_index}"
                )

    def __len__(self) -> int:
        return self.sample_count

    def _connection(self) -> sqlite3.Connection:
        if self._index_connection is None:
            encoded_path = quote(self.sample_index_path.as_posix(), safe="/")
            uri = f"file:{encoded_path}?mode=ro&immutable=1"
            self._index_connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._index_connection

    def _index_has_metadata(self) -> bool:
        if self._index_has_metadata_cache is None:
            columns = {
                str(row[1])
                for row in self._connection().execute("PRAGMA table_info(samples)")
            }
            self._index_has_metadata_cache = {
                "sample_ref",
                "dataset",
                "sequence_key",
                "query_alias",
                "frame_idx",
            }.issubset(columns)
        return self._index_has_metadata_cache

    def _metadata_at(self, ordinal: int) -> dict[str, Any]:
        if self._index_has_temporal_metadata_cache is None:
            columns = {
                str(row[1])
                for row in self._connection().execute("PRAGMA table_info(samples)")
            }
            self._index_has_temporal_metadata_cache = "source_frame_idx" in columns
        extended = self._index_has_temporal_metadata_cache
        select = (
            "sample_id, sample_ref, dataset, sequence_key, query_alias, frame_idx, "
            "source_frame_idx, timestamp, is_right, bbox_xyxy, bbox_association_id"
            if extended
            else "sample_id, sample_ref, dataset, sequence_key, query_alias, frame_idx"
        )
        row = self._connection().execute(
            f"SELECT {select} FROM samples WHERE ordinal = ?", (int(ordinal),)
        ).fetchone()
        if row is None:
            raise IndexError(ordinal)
        metadata = {
            "sample_ref": str(row[1]),
            "dataset": str(row[2]),
            "sequence_key": str(row[3]),
            "query_alias": str(row[4]),
            "frame_idx": int(row[5]),
        }
        if extended:
            metadata.update(
                {
                    "source_frame_idx": None if row[6] is None else int(row[6]),
                    "timestamp": None if row[7] is None else float(row[7]),
                    "is_right": int(row[8]),
                    "bbox_xyxy": json.loads(str(row[9])),
                    "bbox_association_id": str(row[10]),
                }
            )
        return {
            "sample_id": str(row[0]),
            "metadata": metadata,
        }

    def field_values(
        self, index: int, fields: Sequence[str] | None = None
    ) -> dict[str, np.ndarray]:
        """Read arrays without paying for a per-sample SQLite metadata query."""

        ordinal = int(index)
        if ordinal < 0:
            ordinal += self.sample_count
        if ordinal < 0 or ordinal >= self.sample_count:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.shard_stops, ordinal)
        entry = self.shards[shard_index]
        row_in_shard = ordinal - int(entry["source_start"])
        arrays, _ = self._load_shard(shard_index)
        requested = self.fields if fields is None else tuple(fields)
        unknown = sorted(set(requested) - set(arrays))
        if unknown:
            raise KeyError(f"Cache fields are unavailable: {unknown}")
        return {name: arrays[name][row_in_shard] for name in requested}

    def location(self, sample_id: str) -> tuple[int, int, int]:
        sample_id = _sample_id(sample_id, location="sample_id lookup")
        row = self._connection().execute(
            "SELECT ordinal, shard, row_in_shard FROM samples WHERE sample_id = ?",
            (sample_id,),
        ).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return int(row[0]), int(row[1]), int(row[2])

    def _load_shard(
        self, shard_index: int
    ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]] | None]:
        cached = self._open_shards.pop(shard_index, None)
        if cached is not None:
            self._open_shards[shard_index] = cached
            return cached
        shard_dir = self.cache_dir / "shards" / _shard_name(shard_index)
        arrays = {
            name: np.load(shard_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in self.fields
        }
        sample_rows = None
        if not self._index_has_metadata():
            sample_rows = []
            for _, row in iter_jsonl(shard_dir / "samples.jsonl"):
                row = dict(row)
                row["sample_id"] = _sample_id(
                    row.get("sample_id"), location=str(shard_dir / "samples.jsonl")
                )
                sample_rows.append(row)
        cached = (arrays, sample_rows)
        self._open_shards[shard_index] = cached
        while len(self._open_shards) > self.max_open_shards:
            self._open_shards.popitem(last=False)
        return cached

    def _item_at(self, ordinal: int, shard_index: int, row_in_shard: int) -> dict[str, Any]:
        arrays, sample_rows = self._load_shard(shard_index)
        if row_in_shard < 0 or row_in_shard >= int(self.shards[shard_index]["sample_count"]):
            raise IndexError(
                f"Invalid row {row_in_shard} for shard {shard_index}"
            )
        if sample_rows is None:
            sample_row = self._metadata_at(ordinal)
        else:
            sample_row = sample_rows[row_in_shard]
        result: dict[str, Any] = {
            "sample_id": sample_row["sample_id"],
            "ordinal": ordinal,
            "shard": shard_index,
            "row_in_shard": row_in_shard,
        }
        metadata = sample_row.get("metadata")
        if isinstance(metadata, Mapping):
            result.update(metadata)
        for name, array in arrays.items():
            value = array[row_in_shard]
            result[name] = np.array(value, copy=True) if self.copy_arrays else value
        return result

    def __getitem__(self, index: int | str) -> dict[str, Any]:
        if isinstance(index, str):
            ordinal, shard_index, row_in_shard = self.location(index)
            return self._item_at(ordinal, shard_index, row_in_shard)
        ordinal = int(index)
        if ordinal < 0:
            ordinal += self.sample_count
        if ordinal < 0 or ordinal >= self.sample_count:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.shard_stops, ordinal)
        entry = self.shards[shard_index]
        row_in_shard = ordinal - int(entry["source_start"])
        return self._item_at(ordinal, shard_index, row_in_shard)

    def get_by_id(self, sample_id: str) -> dict[str, Any]:
        return self[sample_id]

    def close(self) -> None:
        self._open_shards.clear()
        if self._index_connection is not None:
            self._index_connection.close()
            self._index_connection = None

    def __enter__(self) -> "FeatureCacheDataset":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_open_shards"] = OrderedDict()
        state["_index_connection"] = None
        state["_index_has_metadata_cache"] = None
        state["_index_has_temporal_metadata_cache"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def verify_feature_cache(
    cache_dir: os.PathLike[str] | str,
    *,
    deep: bool = False,
    expected_manifest_path: os.PathLike[str] | str | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    with FeatureCacheDataset(
        cache_dir,
        expected_manifest_path=expected_manifest_path,
        expected_provenance=expected_provenance,
        deep_verify=deep,
    ) as dataset:
        return {
            "cache_dir": str(dataset.cache_dir),
            "config_sha256": dataset.config_sha256,
            "provenance_sha256": dataset.config["provenance_sha256"],
            "sample_count": len(dataset),
            "shard_count": len(dataset.shards),
            "fields": [dataset.spec_by_name[name].to_json() for name in dataset.fields],
            "deep_verified": bool(deep),
        }
