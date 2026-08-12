#!/usr/bin/env python3
"""Build sequence-sharded MoGe-2 geometry sidecars.

The authoritative query manifest is grouped first by sequence and then by
source HDF5 frame. MoGe is evaluated exactly once for each source frame; every
query on that frame receives its own bbox-aligned point/normal/valid crop.

Existing complete sequence shards are reused only when their sequence,
configuration, manifest SHA, and query identities all match. Missing geometry
is always fatal: this extractor never writes zero-filled records and has no
online fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactile_input_priors.depth_teacher import MogeTeacher
from tactile_input_priors.hdf5_manifest import (
    HDF5ImageReader,
    canonical_query_record,
    sha256_file,
)
from tactile_input_priors.depth_sidecar import (
    DepthSidecarConfig,
    SequenceSidecarWriter,
    sequence_sidecar_filename,
)


EXTRACTOR_SCHEMA = "tactile_moge_sidecar_extractor_v1"


def _parse_hw(value: str) -> tuple[int, int]:
    text = str(value).strip().lower().replace("*", "x")
    parts = text.split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must be HEIGHTxWIDTH")
    try:
        height, width = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution must be HEIGHTxWIDTH") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return height, width


def _load_authoritative_manifest(
    path: Path,
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("row must be a JSON object")
                if raw.get("h5_path") is None and raw.get("h5_relpath") is not None:
                    root_value = raw.get("data_root") or raw.get("hdf5_root")
                    root = (
                        Path(str(root_value)).expanduser().resolve(strict=False)
                        if root_value
                        else data_root
                    )
                    if root is None:
                        # Published manifests normally live at ROOT/manifests/.
                        inferred_root = path.parent.parent.resolve(strict=False)
                        candidate = inferred_root / str(raw["h5_relpath"])
                        if candidate.is_file():
                            root = inferred_root
                    if root is None:
                        raise ValueError(
                            "row contains h5_relpath but no resolvable data root; "
                            "pass --data-root"
                        )
                    root = root.resolve(strict=True)
                    h5_path = (root / str(raw["h5_relpath"])).resolve(strict=True)
                    try:
                        h5_path.relative_to(root)
                    except ValueError as exc:
                        raise ValueError(
                            f"h5_relpath escapes data root {root}: {raw['h5_relpath']!r}"
                        ) from exc
                    raw["h5_path"] = str(h5_path)
                row = canonical_query_record(raw)
            except Exception as exc:
                raise RuntimeError(
                    f"Invalid authoritative manifest row at {path}:{line_number}: {exc}"
                ) from exc
            bbox = np.asarray(row["bbox_xyxy"], dtype=np.float64)
            if (
                bbox.shape != (4,)
                or not np.isfinite(bbox).all()
                or bbox[2] <= bbox[0] + 1.0
                or bbox[3] <= bbox[1] + 1.0
            ):
                raise RuntimeError(
                    f"Invalid bbox at {path}:{line_number}: {row['bbox_xyxy']!r}"
                )
            uid = str(row["sample_uid"])
            if uid in seen_uids:
                raise RuntimeError(f"Duplicate sample_uid in manifest: {uid!r}")
            seen_uids.add(uid)
            row["_manifest_line"] = line_number
            rows.append(row)
    if not rows:
        raise RuntimeError(f"Authoritative query manifest is empty: {path}")
    return rows


def _group_by_sequence(
    rows: Iterable[Mapping[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in rows:
        row = dict(value)
        sequence_key = str(row["sequence_key"]).strip()
        if not sequence_key:
            raise RuntimeError(f"Empty sequence_key for sample {row['sample_uid']!r}")
        grouped[sequence_key].append(row)
    output: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for sequence_key in sorted(grouped):
        sequence_rows = sorted(
            grouped[sequence_key],
            key=lambda row: (
                str(row["h5_path"]),
                int(row["frame_row"]),
                int(row["query_row"]),
                str(row["sample_uid"]),
            ),
        )
        query_rows = [int(row["query_row"]) for row in sequence_rows]
        if len(query_rows) != len(set(query_rows)):
            raise RuntimeError(
                f"Sequence {sequence_key!r} contains duplicate query_row values"
            )
        output[sequence_key] = sequence_rows
    return output


def _resolve_model_checkpoint(model: Path) -> Path:
    path = model.expanduser().resolve(strict=False)
    if path.is_dir():
        path = path / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"MoGe-2 checkpoint is missing: {path}")
    return path.resolve()


def _validate_sha256(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256")
    return normalized


def _source_to_grid_affine(
    bbox_xyxy: Sequence[float],
    bbox_scale: float,
    output_hw: tuple[int, int],
) -> np.ndarray:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    center = 0.5 * (bbox[:2] + bbox[2:])
    side = float(np.max(bbox[2:] - bbox[:2])) * float(bbox_scale)
    if not math.isfinite(side) or side <= 1.0:
        raise ValueError(f"Invalid scaled bbox side {side} for bbox {bbox.tolist()}")
    height, width = output_hw
    transform = np.zeros((2, 3), dtype=np.float64)
    transform[0, 0] = height / side
    transform[1, 1] = height / side
    transform[0, 2] = -height * float(center[0]) / side + width * 0.5
    transform[1, 2] = height * (-float(center[1]) / side + 0.5)
    return transform.astype(np.float32)


def _homogeneous(affine: np.ndarray) -> np.ndarray:
    value = np.asarray(affine, dtype=np.float64)
    if value.shape != (2, 3) or not np.isfinite(value).all():
        raise ValueError(f"Expected finite 2x3 affine, got {value.shape}")
    return np.vstack((value, np.asarray((0.0, 0.0, 1.0), dtype=np.float64)))


def _source_to_geometry_affine(
    source_hw: tuple[int, int], geometry_hw: tuple[int, int]
) -> np.ndarray:
    """Map source-image pixel centers to geometry-map pixel centers."""

    source_height, source_width = source_hw
    geometry_height, geometry_width = geometry_hw
    scale_x = geometry_width / float(source_width)
    scale_y = geometry_height / float(source_height)
    return np.asarray(
        (
            (scale_x, 0.0, 0.5 * scale_x - 0.5),
            (0.0, scale_y, 0.5 * scale_y - 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def _canonical_vector_map(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise RuntimeError(f"MoGe {name} must be a 3D map, got {array.shape}")
    if array.shape[-1] in (3, 4):
        array = array[..., :3]
    elif array.shape[0] in (3, 4):
        array = np.moveaxis(array[:3], 0, -1)
    else:
        raise RuntimeError(f"MoGe {name} has no XYZ channels: {array.shape}")
    if array.ndim != 3 or array.shape[-1] != 3 or min(array.shape[:2]) <= 0:
        raise RuntimeError(f"Could not canonicalize MoGe {name}: {array.shape}")
    return np.ascontiguousarray(array, dtype=np.float32)


def _canonical_valid_map(value: Any, expected_hw: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"MoGe mask must be a 2D map, got {array.shape}")
    if array.shape != expected_hw:
        raise RuntimeError(
            f"MoGe mask shape {array.shape} does not match geometry shape {expected_hw}"
        )
    if np.issubdtype(array.dtype, np.bool_):
        return np.ascontiguousarray(array, dtype=bool)
    array = np.asarray(array, dtype=np.float32)
    return np.ascontiguousarray(np.isfinite(array) & (array > 0.5), dtype=bool)


def _extract_geometry(output: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_value = output.get("points")
    if point_value is None:
        point_value = output.get("point_map")
    if point_value is None:
        raise RuntimeError("MoGe output is missing points/point_map")
    if output.get("normal") is None:
        raise RuntimeError("MoGe output is missing normal")
    if output.get("mask") is None:
        raise RuntimeError("MoGe output is missing mask/valid geometry")
    point = _canonical_vector_map(point_value, "point map")
    normal = _canonical_vector_map(output["normal"], "normal map")
    if normal.shape != point.shape:
        raise RuntimeError(
            f"MoGe point/normal shapes differ: point={point.shape}, normal={normal.shape}"
        )
    valid = _canonical_valid_map(output["mask"], point.shape[:2])
    valid &= np.isfinite(point).all(axis=-1) & np.isfinite(normal).all(axis=-1)
    normal_norm = np.linalg.norm(normal, axis=-1)
    valid &= normal_norm > 1e-8
    if not valid.any():
        raise RuntimeError("MoGe produced no finite valid geometry for the source frame")
    return point, normal, valid


def _validity_aware_crop(
    point: np.ndarray,
    normal: np.ndarray,
    valid: np.ndarray,
    geometry_to_grid: np.ndarray,
    output_hw: tuple[int, int],
    validity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = output_hw
    valid_float = np.asarray(valid, dtype=np.float32)
    weights = cv2.warpAffine(
        valid_float,
        geometry_to_grid,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    def warp_vectors(values: np.ndarray) -> np.ndarray:
        result = np.empty((height, width, 3), dtype=np.float32)
        denominator = np.maximum(weights, 1e-6)
        for channel in range(3):
            numerator = cv2.warpAffine(
                np.asarray(values[..., channel] * valid_float, dtype=np.float32),
                geometry_to_grid,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
            result[..., channel] = numerator / denominator
        result[weights <= 1e-6] = 0.0
        return result

    cropped_point = warp_vectors(point)
    cropped_normal = warp_vectors(normal)
    cropped_valid = (
        (weights >= float(validity_threshold))
        & np.isfinite(cropped_point).all(axis=-1)
        & np.isfinite(cropped_normal).all(axis=-1)
    )
    normal_norm = np.linalg.norm(cropped_normal, axis=-1, keepdims=True)
    cropped_valid &= normal_norm[..., 0] > 1e-8
    cropped_normal /= np.maximum(normal_norm, 1e-8)
    cropped_point[~cropped_valid] = 0.0
    cropped_normal[~cropped_valid] = 0.0
    if not cropped_valid.any():
        raise RuntimeError("Teacher bbox crop contains no valid MoGe geometry")
    return cropped_point, cropped_normal, cropped_valid


def _crop_query_geometry(
    point: np.ndarray,
    normal: np.ndarray,
    valid: np.ndarray,
    source_hw: tuple[int, int],
    bbox_xyxy: Sequence[float],
    bbox_scale: float,
    stored_grid_hw: tuple[int, int],
    validity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    teacher_affine = _source_to_grid_affine(
        bbox_xyxy=bbox_xyxy,
        bbox_scale=bbox_scale,
        output_hw=stored_grid_hw,
    )
    source_to_geometry = _source_to_geometry_affine(source_hw, point.shape[:2])
    geometry_to_grid = (
        _homogeneous(teacher_affine) @ np.linalg.inv(source_to_geometry)
    )[:2].astype(np.float32)
    cropped_point, cropped_normal, cropped_valid = _validity_aware_crop(
        point=point,
        normal=normal,
        valid=valid,
        geometry_to_grid=geometry_to_grid,
        output_hw=stored_grid_hw,
        validity_threshold=validity_threshold,
    )
    return cropped_point, cropped_normal, cropped_valid, teacher_affine


def _expected_identities(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    return sorted((str(row["sample_uid"]), int(row["query_row"])) for row in rows)


def _complete_shard_matches(
    path: Path,
    sequence_key: str,
    rows: Sequence[Mapping[str, Any]],
    config: DepthSidecarConfig,
) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    try:
        with h5py.File(path, "r", libver="latest", swmr=True) as handle:
            if int(handle.attrs.get("complete", 0)) != 1:
                return False, "incomplete"
            actual_sequence = handle.attrs.get("sequence_key", "")
            if isinstance(actual_sequence, bytes):
                actual_sequence = actual_sequence.decode("utf-8")
            if str(actual_sequence) != sequence_key:
                return False, "sequence mismatch"
            expected_attrs = {
                "model_sha256": config.model_sha256,
                "manifest_sha256": config.manifest_sha256,
                "config_sha256": config.config_sha256,
            }
            for name, expected in expected_attrs.items():
                actual = handle.attrs.get(name, "")
                if isinstance(actual, bytes):
                    actual = actual.decode("utf-8")
                if str(actual) != expected:
                    return False, f"{name} mismatch"
            if int(handle.attrs.get("record_count", -1)) != len(rows):
                return False, "record_count mismatch"
            if "queries/sample_uid" not in handle or "queries/query_row" not in handle:
                return False, "missing query identity datasets"
            actual_identities = sorted(
                zip(
                    (str(value) for value in handle["queries/sample_uid"].asstr()[:]),
                    (int(value) for value in handle["queries/query_row"][:]),
                )
            )
            if actual_identities != _expected_identities(rows):
                return False, "query identity mismatch"
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return False, f"unreadable: {exc}"
    return True, "complete"


def _frame_groups(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[tuple[str, int], list[Mapping[str, Any]]]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(Path(str(row["h5_path"])).expanduser()), int(row["frame_row"]))
        grouped[key].append(row)
    return sorted(
        (
            key,
            sorted(
                values,
                key=lambda row: (int(row["query_row"]), str(row["sample_uid"])),
            ),
        )
        for key, values in grouped.items()
    )


def _build_config(
    args: argparse.Namespace,
    model_path: Path,
    model_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> DepthSidecarConfig:
    return DepthSidecarConfig(
        teacher_model=f"moge2:{model_path.name}",
        model_sha256=model_sha256,
        manifest_sha256=manifest_sha256,
        teacher_input_hw=args.teacher_input_hw,
        stored_grid_hw=args.stored_grid_hw,
        teacher_bbox_scale=float(args.teacher_bbox_scale),
        extra={
            "extractor_schema": EXTRACTOR_SCHEMA,
            "teacher_backend": "moge2",
            "teacher_inference_view": "full_source_frame_native",
            "teacher_crop_semantics": "rectangular_affine_height_anchors_square_bbox",
            "teacher_input_hw_semantics": "nominal_teacher_resolution_provenance",
            "point_key_priority": ["points", "point_map"],
            "normal_key": "normal",
            "valid_key": "mask",
            "interpolation": "validity_aware_bilinear",
            "manifest_name": manifest_path.name,
            "model_name": model_path.name,
        },
    )


def build_sidecars(args: argparse.Namespace) -> None:
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"--shard-index must be in [0,{args.num_shards}), got {args.shard_index}"
        )
    if not math.isfinite(args.teacher_bbox_scale) or args.teacher_bbox_scale <= 0:
        raise ValueError("--teacher-bbox-scale must be finite and positive")
    if not 0.0 < args.validity_threshold <= 1.0:
        raise ValueError("--validity-threshold must be in (0,1]")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")

    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    model_path = _resolve_model_checkpoint(Path(args.model))
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = sha256_file(manifest_path)
    model_sha256 = (
        _validate_sha256(args.model_sha256, "--model-sha256")
        if args.model_sha256
        else sha256_file(model_path)
    )
    data_root = (
        Path(args.data_root).expanduser().resolve(strict=True)
        if args.data_root
        else None
    )
    rows = _load_authoritative_manifest(manifest_path, data_root=data_root)
    sequences = _group_by_sequence(rows)
    selected_keys = list(sequences)[args.shard_index :: args.num_shards]
    config = _build_config(
        args,
        model_path=model_path,
        model_sha256=model_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )

    pending: list[tuple[str, list[dict[str, Any]], Path]] = []
    reused_sequences = 0
    reused_records = 0
    for sequence_key in selected_keys:
        sequence_rows = sequences[sequence_key]
        path = output_dir / sequence_sidecar_filename(sequence_key)
        matches, reason = _complete_shard_matches(
            path, sequence_key, sequence_rows, config
        )
        if matches:
            reused_sequences += 1
            reused_records += len(sequence_rows)
            continue
        if path.exists() and not args.overwrite_incompatible:
            raise RuntimeError(
                f"Existing sidecar is not reusable ({reason}): {path}. "
                "Inspect it, or pass --overwrite-incompatible to replace it atomically."
            )
        pending.append((sequence_key, sequence_rows, path))

    total_pending_records = sum(len(value) for _, value, _ in pending)
    total_pending_frames = sum(len(_frame_groups(value)) for _, value, _ in pending)
    print(
        f"[depth-sidecar:{args.shard_index}/{args.num_shards}] "
        f"manifest_rows={len(rows)} selected_sequences={len(selected_keys)} "
        f"reused_sequences={reused_sequences} reused_records={reused_records} "
        f"pending_sequences={len(pending)} pending_frames={total_pending_frames} "
        f"pending_records={total_pending_records}",
        flush=True,
    )
    print(
        f"[depth-sidecar] model_sha256={model_sha256} "
        f"manifest_sha256={manifest_sha256} config_sha256={config.config_sha256}",
        flush=True,
    )
    if not pending:
        print("All selected sequence sidecars are already complete and compatible.", flush=True)
        return

    teacher = MogeTeacher(model_path, device=args.device)
    reader = HDF5ImageReader(args.hdf5_handle_cache_size)
    completed_frames = 0
    completed_records = 0
    completed_sequences = 0
    started_at = time.monotonic()
    try:
        for sequence_key, sequence_rows, output_path in pending:
            frame_groups = _frame_groups(sequence_rows)
            with SequenceSidecarWriter(
                output_path,
                sequence_key=sequence_key,
                config=config,
                expected_count=len(sequence_rows),
                compression=args.compression,
                overwrite=args.overwrite_incompatible,
            ) as writer:
                for (_h5_path, _frame_row), frame_rows in frame_groups:
                    representative = frame_rows[0]
                    bgr = reader.read_bgr(representative)
                    source_hw = (int(bgr.shape[0]), int(bgr.shape[1]))
                    output = teacher.infer(bgr)
                    point, normal, valid = _extract_geometry(output)
                    for row in frame_rows:
                        try:
                            point_crop, normal_crop, valid_crop, teacher_affine = (
                                _crop_query_geometry(
                                    point=point,
                                    normal=normal,
                                    valid=valid,
                                    source_hw=source_hw,
                                    bbox_xyxy=row["bbox_xyxy"],
                                    bbox_scale=args.teacher_bbox_scale,
                                    stored_grid_hw=args.stored_grid_hw,
                                    validity_threshold=args.validity_threshold,
                                )
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"Geometry crop failed for sample={row['sample_uid']!r}, "
                                f"sequence={sequence_key!r}, frame_row={row['frame_row']}, "
                                f"query_row={row['query_row']}: {exc}"
                            ) from exc
                        writer.append(
                            sample_uid=str(row["sample_uid"]),
                            query_row=int(row["query_row"]),
                            point=point_crop,
                            normal=normal_crop,
                            valid=valid_crop,
                            teacher_affine=teacher_affine,
                        )
                        completed_records += 1
                    completed_frames += 1
                    if completed_frames % args.progress_every == 0:
                        elapsed = max(time.monotonic() - started_at, 1e-6)
                        print(
                            f"[depth-sidecar:{args.shard_index}/{args.num_shards}] "
                            f"frames={completed_frames}/{total_pending_frames} "
                            f"records={completed_records}/{total_pending_records} "
                            f"frames_per_s={completed_frames / elapsed:.2f} "
                            f"current_sequence={sequence_key}",
                            flush=True,
                        )
            completed_sequences += 1
            print(
                f"[depth-sidecar:{args.shard_index}/{args.num_shards}] committed "
                f"sequence={sequence_key!r} records={len(sequence_rows)} "
                f"path={output_path}",
                flush=True,
            )
    finally:
        reader.close()

    elapsed = max(time.monotonic() - started_at, 1e-6)
    print(
        f"Completed shard {args.shard_index}/{args.num_shards}: "
        f"sequences={completed_sequences}, frames={completed_frames}, "
        f"records={completed_records}, elapsed_s={elapsed:.1f}, "
        f"frames_per_s={completed_frames / elapsed:.2f}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract full-frame MoGe-2 geometry into atomic sequence sidecars."
    )
    parser.add_argument("--manifest", required=True, help="Authoritative query JSONL manifest")
    parser.add_argument(
        "--data-root",
        default="",
        help=(
            "Processed HDF5 root used to resolve h5_relpath rows. When omitted, "
            "ROOT/manifests/MANIFEST is detected automatically."
        ),
    )
    parser.add_argument("--model", required=True, help="MoGe-2 model.pt or its directory")
    parser.add_argument("--output-dir", required=True, help="Sequence sidecar output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--teacher-bbox-scale", type=float, default=1.65)
    parser.add_argument(
        "--teacher-input-hw",
        type=_parse_hw,
        default=(512, 384),
        metavar="HEIGHTxWIDTH",
        help="Nominal MoGe teacher resolution recorded in sidecar provenance",
    )
    parser.add_argument(
        "--stored-grid-hw",
        type=_parse_hw,
        default=(64, 48),
        metavar="HEIGHTxWIDTH",
    )
    parser.add_argument("--validity-threshold", type=float, default=0.5)
    parser.add_argument("--hdf5-handle-cache-size", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument(
        "--model-sha256",
        default="",
        help="Known model SHA256; avoids rehashing a large shared checkpoint",
    )
    parser.add_argument(
        "--overwrite-incompatible",
        action="store_true",
        help="Atomically replace existing incomplete or incompatible sequence shards",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.compression == "none":
        args.compression = None
    build_sidecars(args)


if __name__ == "__main__":
    main()
