#!/usr/bin/env python3
"""Aggregate frame diagnostics into anonymous-query sequence failure reports."""

import argparse
import csv
import hashlib
import json
import os
import struct
import sys
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from functools import lru_cache
from pathlib import Path

_EXPORT_ONLY = any(
    value == "--mode=export_manifests"
    or value == "export_manifests"
    for index, value in enumerate(sys.argv)
    if value == "--mode=export_manifests"
    or (index > 0 and sys.argv[index - 1] == "--mode")
)

if not _EXPORT_ONLY:
    import cv2
    import matplotlib

    cv2.setNumThreads(0)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
else:
    cv2 = None
    plt = None
    np = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **_kwargs):
        return iterable if iterable is not None else range(0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("predictions", "data_integrity", "export_manifests"),
        default="predictions",
    )
    parser.add_argument("--diagnostics_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--source_manifest",
        default=None,
        help="Verified mixed data_integrity_samples.csv used by export_manifests.",
    )
    parser.add_argument("--top_sequences", type=int, default=20)
    parser.add_argument("--plot_top", type=int, default=10)
    parser.add_argument("--peaks_per_sequence", type=int, default=3)
    parser.add_argument("--neighbor_radius", type=int, default=2)
    parser.add_argument("--query_overlay_count", type=int, default=50)
    parser.add_argument("--bbox_rescale_factor", type=float, default=None)
    parser.add_argument(
        "--max_metadata_integrity_error_rate",
        "--max_pairing_error_rate",
        dest="max_metadata_integrity_error_rate",
        type=float,
        default=0.005,
    )
    parser.add_argument("--max_source_frame_mismatch_rate", type=float, default=0.005)
    parser.add_argument("--datasets", default="opentouch,touchanything")
    parser.add_argument("--data_dir", action="append", default=[])
    parser.add_argument("--splits", default="auto")
    parser.add_argument("--index_workers", type=int, default=32)
    parser.add_argument(
        "--integrity_workers",
        type=int,
        default=32,
        help="Bounded workers used only when an old index lacks the integrity sidecar.",
    )
    parser.add_argument(
        "--decode_samples",
        type=int,
        default=4096,
        help="Stratified JPEGs decoded by the fast path; -1 checks every indexed image.",
    )
    parser.add_argument("--decode_workers", type=int, default=64)
    parser.add_argument("--decode_batch_size", type=int, default=256)
    parser.add_argument("--jpeg_backend", choices=("process", "thread"), default="process")
    parser.add_argument(
        "--jpeg_check_mode",
        choices=("header", "decode"),
        default="header",
        help="Header mode checks every image cheaply; decode mode fully decompresses every selected JPEG.",
    )
    parser.add_argument(
        "--getitem_samples",
        type=int,
        default=4096,
        help="Stratified samples passed through the complete dataset __getitem__; -1 checks all samples.",
    )
    parser.add_argument("--getitem_workers", type=int, default=16)
    parser.add_argument(
        "--write_full_integrity_csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write the very large per-query CSV; the integrity sidecar already contains all records.",
    )
    parser.add_argument("--index_backend", choices=("process", "thread"), default="process")
    parser.add_argument("--index_chunksize", type=int, default=512)
    parser.add_argument("--index_cache_dir", default=None)
    parser.add_argument("--index_cache_timeout", type=int, default=3600)
    parser.add_argument("--rebuild_index", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--focus_keywords",
        default="swing_hammer,pick_up_dairy_product",
        help="Comma-separated keywords forced into timeline outputs when present.",
    )
    return parser.parse_args()


STRICT_DOMAIN_COUNTS = {
    "OpenTouch": {"train": 127326, "val": 16566},
    "TouchAnything": {"train": 429967, "val": 38845},
}


def _canonical_audit_dataset(value):
    aliases = {
        "opentouch": "OpenTouch",
        "open_touch": "OpenTouch",
        "ot": "OpenTouch",
        "touchanything": "TouchAnything",
        "touch_anything": "TouchAnything",
        "egotouch": "TouchAnything",
        "ego_touch": "TouchAnything",
        "ta": "TouchAnything",
    }
    raw = str(value or "").strip()
    return aliases.get(raw.lower(), raw)


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic_local(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _dataset_root_from_sample(sample_dir, split):
    normalized = os.path.abspath(os.path.expanduser(str(sample_dir)))
    marker = f"{os.sep}{split}{os.sep}"
    if marker not in normalized:
        raise ValueError(
            f"Cannot infer dataset root from sample_dir={sample_dir!r}, split={split!r}"
        )
    return normalized.split(marker, 1)[0]


def export_verified_domain_manifests(args):
    default_source = (
        Path(__file__).resolve().parent
        / "data_integrity_audits"
        / "mixed_v2_input"
        / "data_integrity_samples.csv"
    )
    source_manifest = Path(args.source_manifest or default_source).expanduser().resolve()
    source_summary_path = source_manifest.parent / "summary.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Verified mixed manifest is missing: {source_manifest}")
    if not source_summary_path.is_file():
        raise FileNotFoundError(f"Verified mixed manifest summary is missing: {source_summary_path}")

    with source_summary_path.open("r", encoding="utf-8") as handle:
        source_summary = json.load(handle)
    if source_summary.get("blocking_reasons"):
        raise ValueError(
            f"Cannot export blocked data-integrity audit: {source_summary['blocking_reasons']}"
        )
    for field in (
        "target_mismatch_count",
        "indexed_invalid_bbox_count",
        "indexed_sample_failure_count",
        "jpeg_decode_failure_count",
        "source_frame_mismatch_count",
    ):
        if int(source_summary.get(field, 0)) != 0:
            raise ValueError(f"Cannot export non-clean audit: {field}={source_summary.get(field)}")

    output_root = Path(
        args.output_dir or source_manifest.parent / "by_dataset"
    ).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_sha256 = _sha256_file(source_manifest)
    names = {"OpenTouch": "opentouch", "TouchAnything": "touchanything"}
    handles = {}
    writers = {}
    temporary_paths = {}
    counts = {dataset: Counter() for dataset in names}
    roots = {dataset: defaultdict(set) for dataset in names}

    try:
        with source_manifest.open("r", newline="", encoding="utf-8") as source_handle:
            reader = csv.DictReader(source_handle)
            if not reader.fieldnames:
                raise ValueError(f"Manifest has no header: {source_manifest}")
            required = {"sample_dir", "dataset", "split"}
            missing = sorted(required.difference(reader.fieldnames))
            if missing:
                raise ValueError(f"Manifest is missing required columns: {missing}")
            for dataset, slug in names.items():
                destination_dir = output_root / slug
                destination_dir.mkdir(parents=True, exist_ok=True)
                temporary = destination_dir / f".data_integrity_samples.csv.tmp-{os.getpid()}"
                handle = temporary.open("w", newline="", encoding="utf-8")
                handles[dataset] = handle
                temporary_paths[dataset] = temporary
                writer = csv.DictWriter(handle, fieldnames=reader.fieldnames, extrasaction="ignore")
                writer.writeheader()
                writers[dataset] = writer

            for row in tqdm(reader, total=int(source_summary.get("audited_samples", 0)), desc="Export manifests"):
                dataset = _canonical_audit_dataset(row.get("dataset"))
                if dataset not in writers:
                    continue
                split = str(row.get("split", "")).strip()
                writers[dataset].writerow(row)
                counts[dataset][split] += 1
                roots[dataset][split].add(_dataset_root_from_sample(row["sample_dir"], split))
    finally:
        for handle in handles.values():
            handle.close()

    for dataset, required_counts in STRICT_DOMAIN_COUNTS.items():
        actual = counts[dataset]
        mismatches = {
            split: (actual.get(split, 0), expected)
            for split, expected in required_counts.items()
            if actual.get(split, 0) != expected
        }
        if mismatches:
            for temporary in temporary_paths.values():
                temporary.unlink(missing_ok=True)
            raise ValueError(f"Strict {dataset} manifest count mismatch: {mismatches}")

    exported = {}
    inherited_fields = (
        "target_mismatch_count",
        "target_mismatch_rate",
        "indexed_invalid_bbox_count",
        "indexed_sample_failure_count",
        "jpeg_decode_failure_count",
        "source_frame_checked_count",
        "source_frame_mismatch_count",
        "source_frame_mismatch_rate",
        "metadata_integrity_error_count",
        "metadata_integrity_error_rate",
    )
    for dataset, slug in names.items():
        destination_dir = output_root / slug
        destination = destination_dir / "data_integrity_samples.csv"
        os.replace(temporary_paths[dataset], destination)
        manifest_sha256 = _sha256_file(destination)
        split_summaries = [
            {
                "split": split,
                "roots": sorted(roots[dataset][split]),
                "indexed_samples": int(count),
                "audited_samples": int(count),
            }
            for split, count in sorted(counts[dataset].items())
        ]
        summary = {
            "status": "ok",
            "blocking_reasons": [],
            "dataset_filter": [dataset],
            "dataset_counts": {dataset: int(sum(counts[dataset].values()))},
            "audited_samples": int(sum(counts[dataset].values())),
            "strict_expected_counts": STRICT_DOMAIN_COUNTS[dataset],
            "source_audit_status": source_summary.get("status", "unknown"),
            "source_summary": str(source_summary_path),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": source_sha256,
            "manifest_sha256": manifest_sha256,
            "split_summaries": split_summaries,
            "identity_policy": "dataset and query metadata are provenance only and never model inputs",
        }
        for field in inherited_fields:
            summary[field] = source_summary.get(field, 0)
        _write_json_atomic_local(destination_dir / "summary.json", summary)
        exported[slug] = {
            "manifest": str(destination),
            "manifest_sha256": manifest_sha256,
            "counts": dict(sorted(counts[dataset].items())),
        }

    export_summary = {
        "status": "ok",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_sha256,
        "exports": exported,
    }
    _write_json_atomic_local(output_root / "summary.json", export_summary)
    print(json.dumps(export_summary, indent=2, sort_keys=True))
    print(f"Strict per-domain manifests: {output_root}")


def sequence_name(sample_dir, dataset):
    name = Path(sample_dir).name
    if str(dataset).lower() in {"touchanything", "egotouch", "ta"}:
        return name.rsplit("__", 1)[0] if "__" in name else name
    parts = name.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else name


def frame_index(sample_dir):
    name = Path(sample_dir).name
    parts = name.rsplit("_", 2)
    token = name.rsplit("__", 1)[-1] if "__" in name else (parts[-2] if len(parts) >= 2 else name)
    try:
        return int(token)
    except ValueError:
        return 0


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fieldnames=None):
    if not rows and not fieldnames:
        return
    path = Path(path)
    keys = list(fieldnames or rows[0])
    for attempt in range(2):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{attempt}")
        try:
            with temp_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            os.replace(temp_path, path)
            return
        except FileNotFoundError:
            if attempt == 1:
                raise
        finally:
            if temp_path.exists():
                temp_path.unlink()


def _read_image_header_size(path):
    """Return image width and height without decoding full pixel data."""
    with open(path, "rb") as handle:
        header = handle.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        if len(header) >= 10 and header[:6] in (b"GIF87a", b"GIF89a"):
            width, height = struct.unpack("<HH", header[6:10])
            return int(width), int(height)
        if len(header) < 2 or header[:2] != b"\xff\xd8":
            raise ValueError("unsupported or invalid image header")

        handle.seek(2)
        sof_markers = {
            0xC0, 0xC1, 0xC2, 0xC3,
            0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB,
            0xCD, 0xCE, 0xCF,
        }
        while True:
            prefix = handle.read(1)
            if not prefix:
                break
            if prefix != b"\xff":
                continue
            marker_byte = handle.read(1)
            while marker_byte == b"\xff":
                marker_byte = handle.read(1)
            if not marker_byte:
                break
            marker = marker_byte[0]
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                break
            if marker in sof_markers:
                dimensions = handle.read(5)
                if len(dimensions) != 5:
                    break
                height = int.from_bytes(dimensions[1:3], "big")
                width = int.from_bytes(dimensions[3:5], "big")
                if width > 0 and height > 0:
                    return width, height
                break
            handle.seek(segment_length - 2, os.SEEK_CUR)
    raise ValueError("could not read image dimensions")


def _bounded_thread_results(function, items, max_workers, prefetch_factor=4):
    """Yield results without creating one Future per dataset sample."""
    max_workers = max(1, int(max_workers))
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = set()
        for _ in range(max_workers * max(1, int(prefetch_factor))):
            try:
                pending.add(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                try:
                    pending.add(executor.submit(function, next(iterator)))
                except StopIteration:
                    pass


def numeric(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def safe_name(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def pearson(x, y):
    if len(x) < 2 or np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def row_failure_score(row):
    return (
        numeric(row, "false_high_gt005_pred03_excess_volume")
        + numeric(row, "false_high_gt05_pred03_excess_volume")
        + 300.0 * (numeric(row, "gt_volume") < 10.0 and numeric(row, "pred_volume") > 300.0)
    )


def dedupe_query_rows(rows):
    deduped = {}
    for row in sorted(rows, key=lambda item: (frame_index(item["sample_dir"]), item["sample_dir"])):
        index = frame_index(row["sample_dir"])
        if index not in deduped or row_failure_score(row) > row_failure_score(deduped[index]):
            deduped[index] = row
    return [deduped[index] for index in sorted(deduped)]


def contiguous_segments(rows):
    rows = dedupe_query_rows(rows)
    if not rows:
        return []
    segments = [[rows[0]]]
    for row in rows[1:]:
        if frame_index(row["sample_dir"]) != frame_index(segments[-1][-1]["sample_dir"]) + 1:
            segments.append([])
        segments[-1].append(row)
    return segments


def summarize_group(name, rows, query_instance=""):
    gt = np.asarray([numeric(row, "gt_volume") for row in rows], dtype=np.float64)
    pred = np.asarray([numeric(row, "pred_volume") for row in rows], dtype=np.float64)
    excess_005 = sum(numeric(row, "false_high_gt005_pred03_excess_volume") for row in rows)
    excess_05 = sum(numeric(row, "false_high_gt05_pred03_excess_volume") for row in rows)
    catastrophic_over = int(np.sum((gt < 10.0) & (pred > 300.0)))
    catastrophic_under = int(np.sum((gt >= 150.0) & (pred < 50.0)))
    base_available = [row for row in rows if numeric(row, "base_prediction_available") > 0.5]
    summary = {
        "sequence": name,
        "query_instance": query_instance,
        "dataset": rows[0].get("dataset", ""),
        "frame_count": len(rows),
        "mean_gt_volume": float(gt.mean()),
        "mean_pred_volume": float(pred.mean()),
        "frame_volume_corr": pearson(gt, pred),
        "max_pred_volume": float(pred.max()),
        "max_pred_gt_ratio": float(np.max(pred / np.maximum(gt, 1e-6))),
        "false_high_gt005_pred03_count": int(
            sum(numeric(row, "false_high_gt005_pred03_count") for row in rows)
        ),
        "false_high_gt005_pred03_excess_volume": excess_005,
        "false_high_gt05_pred03_count": int(
            sum(numeric(row, "false_high_gt05_pred03_count") for row in rows)
        ),
        "false_high_gt05_pred03_excess_volume": excess_05,
        "catastrophic_over_count": catastrophic_over,
        "catastrophic_under_count": catastrophic_under,
        "base_prediction_frame_count": len(base_available),
        "mean_base_pred_volume": (
            float(np.mean([numeric(row, "base_pred_volume") for row in base_available]))
            if base_available else -1.0
        ),
        "base_catastrophic_over_count": int(
            sum(numeric(row, "base_catastrophic_over") for row in base_available)
        ),
        "residual_created_catastrophic_over_count": int(
            sum(numeric(row, "residual_created_catastrophic_over") for row in base_available)
        ),
        "residual_corrected_catastrophic_over_count": int(
            sum(numeric(row, "residual_corrected_catastrophic_over") for row in base_available)
        ),
        "failure_score": excess_005 + excess_05 + 300.0 * catastrophic_over + 100.0 * catastrophic_under,
    }
    return summary


@lru_cache(maxsize=4096)
def load_metadata(sample_dir):
    path = Path(sample_dir) / "meta.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pressure_volume(value):
    if value is None:
        return None
    try:
        signal = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if signal.size == 0 or not np.isfinite(signal).all():
        return None
    return float(np.clip(signal, 0.0, 1.0).sum())


def valid_bbox(value):
    try:
        bbox = np.asarray(value, dtype=np.float32).reshape(-1)[:4]
    except (TypeError, ValueError):
        return None
    if bbox.size != 4 or not np.isfinite(bbox).all() or np.any(bbox[2:] <= bbox[:2]):
        return None
    return bbox


def bbox_intersection_fraction(bbox, region):
    if bbox is None or region is None:
        return 0.0
    intersection_min = np.maximum(bbox[:2], region[:2])
    intersection_max = np.minimum(bbox[2:], region[2:])
    intersection_size = np.maximum(intersection_max - intersection_min, 0.0)
    intersection = float(intersection_size[0] * intersection_size[1])
    bbox_size = bbox[2:] - bbox[:2]
    area = float(bbox_size[0] * bbox_size[1])
    return intersection / max(area, 1e-12)


def query_fov_bbox(query_bbox, bbox_rescale_factor):
    if query_bbox is None:
        return None
    center = 0.5 * (query_bbox[:2] + query_bbox[2:])
    square_size = float(bbox_rescale_factor) * float(np.max(query_bbox[2:] - query_bbox[:2]))
    # The model receives columns 32:224 from the 256x256 square crop.
    return np.asarray(
        [
            center[0] - 0.375 * square_size,
            center[1] - 0.5 * square_size,
            center[0] + 0.375 * square_size,
            center[1] + 0.5 * square_size,
        ],
        dtype=np.float32,
    )


def metadata_pressure_for_single_query(metadata):
    original = metadata.get("original_hdf5_data", {})
    is_right = int(metadata.get("is_right", 1))
    side = "right" if is_right else "left"
    value = original.get(f"{side}_pressure_continuous_subdiv")
    if value is None:
        value = metadata.get("gaussian_pressure")
    return value


def enrich_query_metadata(row, bbox_rescale_factor):
    metadata = load_metadata(row["sample_dir"])
    result = {
        "metadata_integrity_checked": 0,
        "metadata_integrity_error": 0,
        "metadata_integrity_status": "metadata_unavailable",
        "query_raw_key": row.get("query_instance_key") or row.get("hand") or "query",
        "query_bbox": None,
        "query_fov_bbox": None,
        "query_flip": 0,
        "query_metadata_gt_volume": -1.0,
        "co_visible_supported": 0,
        "co_visible_in_fov": 0,
        "co_visible_bbox": None,
        "co_visible_bbox_fraction": 0.0,
        "co_visible_gt_volume": 0.0,
    }
    if metadata is None:
        return result

    hands = metadata.get("hands")
    if isinstance(hands, dict):
        result["metadata_integrity_checked"] = 1
        result["co_visible_supported"] = 1
        target_alias = str(row.get("hand", ""))
        target = hands.get(target_alias)
        if not isinstance(target, dict):
            result.update(metadata_integrity_error=1, metadata_integrity_status="query_alias_missing")
            return result
        result["query_raw_key"] = str(
            target.get("track_id") or target.get("instance_id") or target.get("hand_id") or target_alias or "query"
        )
        result["query_flip"] = int(target.get("is_right", 1 if target_alias == "right" else 0)) == 0
        query_bbox = valid_bbox(target.get("bbox_chest"))
        query_pressure = pressure_volume(target.get("gaussian_pressure"))
        if query_bbox is None or query_pressure is None:
            result.update(
                metadata_integrity_error=1,
                metadata_integrity_status="query_bbox_or_pressure_missing",
            )
            return result
        result["metadata_integrity_status"] = "ok"
        result["query_bbox"] = query_bbox
        result["query_fov_bbox"] = query_fov_bbox(query_bbox, bbox_rescale_factor)
        result["query_metadata_gt_volume"] = float(query_pressure)

        visible_candidates = []
        for alias, other in hands.items():
            if alias == target_alias or not isinstance(other, dict):
                continue
            other_bbox = valid_bbox(other.get("bbox_chest"))
            other_volume = pressure_volume(other.get("gaussian_pressure"))
            if other_bbox is None or other_volume is None:
                continue
            fraction = bbox_intersection_fraction(other_bbox, result["query_fov_bbox"])
            if fraction > 0.0:
                visible_candidates.append((other_volume, fraction, other_bbox))
        if visible_candidates:
            other_volume, fraction, other_bbox = max(visible_candidates, key=lambda item: (item[0], item[1]))
            result["co_visible_in_fov"] = 1
            result["co_visible_bbox"] = other_bbox
            result["co_visible_bbox_fraction"] = float(fraction)
            result["co_visible_gt_volume"] = float(other_volume)
        return result

    result["metadata_integrity_checked"] = 1
    result["query_flip"] = int(metadata.get("is_right", 1)) == 0
    query_bbox = valid_bbox(metadata.get("bbox"))
    query_pressure = pressure_volume(metadata_pressure_for_single_query(metadata))
    if query_bbox is None or query_pressure is None:
        result.update(
            metadata_integrity_error=1,
            metadata_integrity_status="query_bbox_or_pressure_missing",
        )
        return result
    result["metadata_integrity_status"] = "ok"
    result["query_bbox"] = query_bbox
    result["query_fov_bbox"] = query_fov_bbox(query_bbox, bbox_rescale_factor)
    result["query_metadata_gt_volume"] = float(query_pressure)
    return result


def bbox_columns(prefix, bbox):
    if bbox is None:
        return {f"{prefix}_{axis}": "" for axis in ("x1", "y1", "x2", "y2")}
    return {
        f"{prefix}_x1": float(bbox[0]),
        f"{prefix}_y1": float(bbox[1]),
        f"{prefix}_x2": float(bbox[2]),
        f"{prefix}_y2": float(bbox[3]),
    }


def resolve_bbox_rescale_factor(args, diagnostics_dir):
    if args.bbox_rescale_factor is not None:
        value = float(args.bbox_rescale_factor)
    else:
        value = 2.0
        config_path = diagnostics_dir.parent / "eval_config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                value = float(
                    config.get("args", {}).get(
                        "bbox_rescale_factor",
                        config.get("model_config", {}).get("bbox_rescale_factor", 2.0),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                value = 2.0
    if not 1.0 <= value <= 4.0:
        raise ValueError("bbox_rescale_factor must lie in [1.0, 4.0]")
    return value


def image_path(sample_dir):
    sample_dir = Path(sample_dir)
    metadata = load_metadata(sample_dir)
    if metadata is None:
        return None
    image_name = metadata.get("views", {}).get("chest") or metadata.get("image")
    if not image_name:
        image_name = "chest.jpg" if "hands" in metadata else "image.jpg"
    path = sample_dir / image_name
    return path if path.is_file() else None


def write_timeline(output_dir, label, rows):
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=140, sharex=True)
    first_segment = True
    for segment in contiguous_segments(rows):
        x = np.asarray([frame_index(row["sample_dir"]) for row in segment])
        gt = np.asarray([numeric(row, "gt_volume") for row in segment])
        pred = np.asarray([numeric(row, "pred_volume") for row in segment])
        false_high = np.asarray([
            numeric(row, "false_high_gt005_pred03_excess_volume")
            + numeric(row, "false_high_gt05_pred03_excess_volume")
            for row in segment
        ])
        axes[0].plot(x, gt, label="GT volume" if first_segment else None, linewidth=1.2)
        axes[0].plot(x, pred, label="Pred volume" if first_segment else None, linewidth=1.2)
        if numeric(segment[0], "base_prediction_available") > 0.5:
            base = np.asarray([numeric(row, "base_pred_volume") for row in segment])
            axes[0].plot(x, base, label="Base-only volume" if first_segment else None, linewidth=0.9)
        axes[1].plot(x, false_high, color="tab:red", linewidth=1.0)
        first_segment = False
    axes[0].axhline(300.0, color="tab:red", linestyle="--", linewidth=0.8)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("frame volume")
    axes[1].set_ylabel("false-high excess")
    axes[1].set_xlabel("frame index")
    figure.suptitle(label)
    figure.tight_layout()
    figure.savefig(output_dir / f"timeline_{safe_name(label)}.png")
    plt.close(figure)


def write_peak_neighbors(output_dir, label, rows, peaks_per_sequence, radius):
    rows = dedupe_query_rows(rows)
    scores = np.asarray([row_failure_score(row) for row in rows])
    if not np.any(scores > 0):
        return
    for peak_rank, peak_index in enumerate(np.argsort(-scores)[:peaks_per_sequence], start=1):
        selected = range(max(0, peak_index - radius), min(len(rows), peak_index + radius + 1))
        panels = []
        for index in selected:
            row = rows[index]
            path = image_path(row["sample_dir"])
            if path is None:
                continue
            image = cv2.imread(str(path))
            if image is None:
                continue
            target_width = 320
            scale = target_width / max(image.shape[1], 1)
            image = cv2.resize(image, (target_width, max(1, int(image.shape[0] * scale))))
            text = f"f={frame_index(row['sample_dir'])} GT={numeric(row, 'gt_volume'):.1f} Pred={numeric(row, 'pred_volume'):.1f}"
            cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), thickness=-1)
            cv2.putText(image, text, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            panels.append(image)
        if panels:
            height = max(panel.shape[0] for panel in panels)
            padded = [
                cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                for panel in panels
            ]
            cv2.imwrite(
                str(output_dir / f"peak_{safe_name(label)}_{peak_rank}.jpg"),
                np.concatenate(padded, axis=1),
            )


def draw_bbox(image, bbox, color, label):
    if bbox is None:
        return
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
    cv2.putText(image, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def write_query_overlay(output_dir, row, rank, bbox_rescale_factor):
    path = image_path(row["sample_dir"])
    query_bbox = row.get("_query_bbox")
    query_fov = row.get("_query_fov_bbox")
    if path is None or query_bbox is None or query_fov is None:
        return
    image = cv2.imread(str(path))
    if image is None:
        return
    overlay = image.copy()
    draw_bbox(overlay, query_bbox, (0, 255, 0), "query bbox")
    draw_bbox(overlay, query_fov, (255, 255, 0), "model FOV")
    draw_bbox(overlay, row.get("_co_visible_bbox"), (0, 0, 255), "co-visible")
    stem = f"{rank:03d}_{safe_name(row['_sequence_name'])}_{safe_name(row['_query_instance'])}_{frame_index(row['sample_dir'])}"
    cv2.imwrite(str(output_dir / f"{stem}_overlay.jpg"), overlay)

    center = 0.5 * (query_bbox[:2] + query_bbox[2:])
    bbox_size = bbox_rescale_factor * float(np.max(query_bbox[2:] - query_bbox[:2]))
    resolution = 256
    transform = np.zeros((2, 3), dtype=np.float32)
    transform[0, 0] = resolution / bbox_size
    transform[1, 1] = resolution / bbox_size
    transform[0, 2] = resolution * (-float(center[0]) / bbox_size + 0.5)
    transform[1, 2] = resolution * (-float(center[1]) / bbox_size + 0.5)
    square_crop = cv2.warpAffine(image, transform, (resolution, resolution), flags=cv2.INTER_LINEAR)
    if row.get("_query_flip"):
        square_crop = cv2.flip(square_crop, 1)
    model_crop = square_crop[:, 32:-32]
    cv2.imwrite(str(output_dir / f"{stem}_model_crop.jpg"), model_crop)


def purity_summary(label, rows):
    count = len(rows)
    catastrophic = sum(
        numeric(row, "gt_volume") < 10.0 and numeric(row, "pred_volume") > 300.0
        for row in rows
    )
    return {
        "condition": label,
        "frame_count": count,
        "mean_query_gt_volume": float(np.mean([numeric(row, "gt_volume") for row in rows])) if rows else 0.0,
        "mean_co_visible_gt_volume": float(np.mean([numeric(row, "_co_visible_gt_volume") for row in rows])) if rows else 0.0,
        "mean_pred_volume": float(np.mean([numeric(row, "pred_volume") for row in rows])) if rows else 0.0,
        "catastrophic_over_count": int(catastrophic),
        "catastrophic_over_rate": float(catastrophic / max(count, 1)),
        "false_high_excess_volume": float(sum(
            numeric(row, "false_high_gt005_pred03_excess_volume")
            + numeric(row, "false_high_gt05_pred03_excess_volume")
            for row in rows
        )),
        "residual_created_catastrophic_over_count": int(sum(
            numeric(row, "residual_created_catastrophic_over") for row in rows
        )),
        "residual_corrected_catastrophic_over_count": int(sum(
            numeric(row, "residual_corrected_catastrophic_over") for row in rows
        )),
    }


def run_prediction_audit(args):
    if not args.diagnostics_dir:
        raise ValueError("--diagnostics_dir is required in predictions mode")
    diagnostics_dir = Path(args.diagnostics_dir).expanduser().resolve()
    frame_path = diagnostics_dir / "frame_metrics_sample.csv"
    if not frame_path.is_file():
        raise FileNotFoundError(f"Missing frame diagnostics: {frame_path}")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else diagnostics_dir / "sequence_failure_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(frame_path)
    required = {"sample_dir", "dataset", "gt_volume", "pred_volume"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"frame_metrics_sample.csv lacks provenance columns: {sorted(missing)}")
    bbox_rescale_factor = resolve_bbox_rescale_factor(args, diagnostics_dir)

    sequence_raw_queries = defaultdict(set)
    for row in tqdm(rows, desc="Auditing anonymous queries"):
        dataset = row.get("dataset", "")
        row["_sequence_name"] = f"{dataset}:{sequence_name(row['sample_dir'], dataset)}"
        metadata_info = enrich_query_metadata(row, bbox_rescale_factor)
        row.update({f"_{key}": value for key, value in metadata_info.items()})
        sequence_raw_queries[row["_sequence_name"]].add(str(row["_query_raw_key"]))

    anonymous_maps = {
        sequence: {raw: f"query_{index}" for index, raw in enumerate(sorted(raw_values))}
        for sequence, raw_values in sequence_raw_queries.items()
    }
    for row in rows:
        row["_query_instance"] = anonymous_maps[row["_sequence_name"]][str(row["_query_raw_key"])]

    query_groups = defaultdict(list)
    for row in rows:
        query_groups[(row["_sequence_name"], row["_query_instance"])].append(row)
    query_groups = {key: dedupe_query_rows(value) for key, value in query_groups.items()}

    sequence_groups = defaultdict(list)
    for (sequence, _query), query_rows in query_groups.items():
        sequence_groups[sequence].extend(query_rows)

    sequence_summaries = [summarize_group(name, group) for name, group in sequence_groups.items()]
    sequence_summaries.sort(key=lambda row: (-row["failure_score"], row["sequence"]))
    query_summaries = [
        summarize_group(sequence, group, query_instance=query)
        for (sequence, query), group in query_groups.items()
    ]
    query_summaries.sort(key=lambda row: (-row["failure_score"], row["sequence"], row["query_instance"]))
    write_rows(output_dir / "sequence_failure_summary.csv", sequence_summaries)
    write_rows(output_dir / "top_sequence_failures.csv", sequence_summaries[: max(0, args.top_sequences)])
    write_rows(output_dir / "query_failure_summary.csv", query_summaries)
    write_rows(output_dir / "top_query_failures.csv", query_summaries[: max(0, args.top_sequences)])

    selected_queries = [
        (row["sequence"], row["query_instance"])
        for row in query_summaries[: max(0, args.plot_top)]
    ]
    focus_keywords = [value.strip().lower() for value in args.focus_keywords.split(",") if value.strip()]
    for key in query_groups:
        if any(keyword in key[0].lower() for keyword in focus_keywords) and key not in selected_queries:
            selected_queries.append(key)
    for sequence, query in selected_queries:
        label = f"{sequence}:{query}"
        write_timeline(output_dir, label, query_groups[(sequence, query)])
        write_peak_neighbors(
            output_dir,
            label,
            query_groups[(sequence, query)],
            peaks_per_sequence=args.peaks_per_sequence,
            radius=args.neighbor_radius,
        )

    query_low = [row for row in rows if numeric(row, "gt_volume") < 10.0 and row["_co_visible_supported"]]
    co_visible_high = [
        row for row in query_low
        if row["_co_visible_in_fov"] and numeric(row, "_co_visible_gt_volume") >= 150.0
    ]
    co_visible_low = [
        row for row in query_low
        if not row["_co_visible_in_fov"] or numeric(row, "_co_visible_gt_volume") < 50.0
    ]
    co_visible_high_ids = {id(row) for row in co_visible_high}
    co_visible_low_ids = {id(row) for row in co_visible_low}
    write_rows(
        output_dir / "query_purity_summary.csv",
        [
            purity_summary("query_low_co_visible_high", co_visible_high),
            purity_summary("query_low_co_visible_low", co_visible_low),
        ],
    )

    metadata_checked = sum(int(row["_metadata_integrity_checked"]) for row in rows)
    metadata_errors = sum(int(row["_metadata_integrity_error"]) for row in rows)
    metadata_error_rate = metadata_errors / max(metadata_checked, 1)
    write_rows(
        output_dir / "query_metadata_integrity_summary.csv",
        [{
            "checked_frames": metadata_checked,
            "error_frames": metadata_errors,
            "error_rate": metadata_error_rate,
            "threshold": args.max_metadata_integrity_error_rate,
            "status": "review_required" if metadata_error_rate > args.max_metadata_integrity_error_rate else "ok",
        }],
    )

    diagnostic_rows = []
    for row in rows:
        condition = ""
        if id(row) in co_visible_high_ids:
            condition = "query_low_co_visible_high"
        elif id(row) in co_visible_low_ids:
            condition = "query_low_co_visible_low"
        if row_failure_score(row) <= 0.0 and not condition:
            continue
        record = {
            key: value
            for key, value in row.items()
            if not key.startswith("_") and key != "hand"
        }
        record.update(
            {
                "sequence": row["_sequence_name"],
                "query_instance": row["_query_instance"],
                "condition": condition,
                "metadata_integrity_status": row["_metadata_integrity_status"],
                "query_crop_flipped": int(row["_query_flip"]),
                "query_metadata_gt_volume": row["_query_metadata_gt_volume"],
                "co_visible_supported": row["_co_visible_supported"],
                "co_visible_in_fov": row["_co_visible_in_fov"],
                "co_visible_bbox_fraction": row["_co_visible_bbox_fraction"],
                "co_visible_gt_volume": row["_co_visible_gt_volume"],
            }
        )
        record.update(bbox_columns("query_bbox", row["_query_bbox"]))
        record.update(bbox_columns("model_fov", row["_query_fov_bbox"]))
        record.update(bbox_columns("co_visible_bbox", row["_co_visible_bbox"]))
        diagnostic_rows.append(record)
    purity_fields = [key for key in rows[0] if not key.startswith("_") and key != "hand"] + [
        "sequence",
        "query_instance",
        "condition",
        "metadata_integrity_status",
        "query_crop_flipped",
        "query_metadata_gt_volume",
        "co_visible_supported",
        "co_visible_in_fov",
        "co_visible_bbox_fraction",
        "co_visible_gt_volume",
    ]
    for prefix in ("query_bbox", "model_fov", "co_visible_bbox"):
        purity_fields.extend(f"{prefix}_{axis}" for axis in ("x1", "y1", "x2", "y2"))
    write_rows(output_dir / "query_purity_frames.csv", diagnostic_rows, fieldnames=purity_fields)

    overlay_dir = output_dir / "query_overlays"
    overlay_dir.mkdir(exist_ok=True)
    ranked_rows = sorted(rows, key=row_failure_score, reverse=True)
    for rank, row in enumerate(ranked_rows[: max(0, args.query_overlay_count)], start=1):
        write_query_overlay(overlay_dir, row, rank, bbox_rescale_factor)

    (output_dir / "audit_config.json").write_text(
        json.dumps(
            {
                "bbox_rescale_factor": bbox_rescale_factor,
                "max_metadata_integrity_error_rate": args.max_metadata_integrity_error_rate,
                "metadata_integrity_error_rate": metadata_error_rate,
                "identity_policy": "hand/track metadata is used only to group offline anonymous queries",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(sequence_summaries)} sequence and {len(query_summaries)} anonymous-query "
        f"summaries to {output_dir} (metadata_integrity_error_rate={metadata_error_rate:.6f})"
    )
    if metadata_error_rate > args.max_metadata_integrity_error_rate:
        print(
            "WARNING: query bbox/pressure metadata integrity errors exceed the configured threshold; "
            "review query_metadata_integrity_summary.csv before interpreting crop experiments."
        )


def _split_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = str(value).split(",")
    return [str(item).strip() for item in values if str(item).strip()]


def _audit_imports():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(workspace_dir / "hamer"))
    from dataset import (  # pylint: disable=import-outside-toplevel
        DATASET_ROOTS,
        OpenTouchTactileDataset,
        SharedCacheBuildLock,
        canonical_dataset_name,
        pressure_for_query,
        scan_sample_dirs_integrity_batch,
        valid_bbox as dataset_valid_bbox,
        wait_for_file,
        write_json_atomic,
    )
    from hamer.configs import get_config  # pylint: disable=import-outside-toplevel
    from hamer_config_assets import (  # pylint: disable=import-outside-toplevel
        resolve_hamer_model_config_path,
    )

    return {
        "script_dir": script_dir,
        "workspace_dir": workspace_dir,
        "dataset_roots": DATASET_ROOTS,
        "dataset_class": OpenTouchTactileDataset,
        "cache_build_lock": SharedCacheBuildLock,
        "canonical_dataset_name": canonical_dataset_name,
        "pressure_for_query": pressure_for_query,
        "scan_sample_dirs_integrity_batch": scan_sample_dirs_integrity_batch,
        "valid_bbox": dataset_valid_bbox,
        "wait_for_file": wait_for_file,
        "write_json_atomic": write_json_atomic,
        "get_config": get_config,
        "resolve_hamer_model_config_path": resolve_hamer_model_config_path,
    }


def _resolve_audit_roots(args, dataset_roots):
    roots = []
    for name in _split_csv(args.datasets):
        key = name.lower()
        if key not in dataset_roots:
            raise ValueError(f"Unknown dataset alias for integrity audit: {name}")
        roots.append(str(Path(dataset_roots[key]).expanduser()))
    for value in args.data_dir:
        roots.extend(_split_csv(value))
    deduped = []
    for root in roots:
        absolute = str(Path(root).expanduser().resolve(strict=False))
        if absolute not in deduped:
            deduped.append(absolute)
    return deduped


def _root_has_samples(path):
    path = Path(path)
    if not path.is_dir():
        return False
    return any(child.is_dir() and (child / "meta.json").is_file() for child in path.iterdir())


def _resolve_audit_splits(root, requested):
    if requested != "auto":
        return _split_csv(requested)
    root_path = Path(root)
    candidates = ("train", "val", "test", "test_seen", "test_unseen")
    splits = [name for name in candidates if _root_has_samples(root_path / name)]
    if splits:
        return splits
    if _root_has_samples(root_path / "all") or _root_has_samples(root_path):
        return ["train"]
    return []


def _metadata_query(meta, dataset_name, hand):
    if dataset_name == "TouchAnything":
        hand_meta = meta.get("hands", {}).get(hand or "", {})
        return hand_meta.get("bbox_chest"), hand_meta
    return meta.get("bbox"), meta


def _array_checksum(value):
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _compare_masked_targets(expected, actual, palm, atol=1e-6):
    expected = np.clip(np.asarray(expected, dtype=np.float32), 0.0, 1.0)
    actual = np.asarray(actual, dtype=np.float32)
    palm = np.asarray(palm, dtype=np.float32)
    expected_masked = expected * palm if expected.shape == palm.shape else expected
    actual_masked = actual * palm if actual.shape == palm.shape else actual
    mismatch = expected_masked.shape != actual_masked.shape or not np.allclose(
        expected_masked,
        actual_masked,
        rtol=0.0,
        atol=float(atol),
    )
    max_abs_diff = (
        float(np.max(np.abs(expected_masked - actual_masked)))
        if expected_masked.shape == actual_masked.shape
        else -1.0
    )
    return expected_masked, actual_masked, bool(mismatch), max_abs_diff


def _optional_int(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or not number.is_integer():
        return None
    return int(number)


def _optional_float(value):
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _sample_dir_frame_index(sample_dir):
    name = Path(sample_dir).name
    if "__" in name:
        return _optional_int(name.rsplit("__", 1)[-1])
    parts = name.rsplit("_", 2)
    return _optional_int(parts[-2]) if len(parts) >= 2 else None


def _bbox_iou(first, second):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    lower = np.maximum(first[:2], second[:2])
    upper = np.minimum(first[2:], second[2:])
    intersection = float(np.maximum(upper - lower, 0.0).prod())
    first_area = float(np.maximum(first[2:] - first[:2], 0.0).prod())
    second_area = float(np.maximum(second[2:] - second[:2], 0.0).prod())
    return intersection / max(first_area + second_area - intersection, 1e-12)


def _write_integrity_overlay(output_dir, row, rank):
    metadata = load_metadata(row["sample_dir"])
    if metadata is None:
        return
    image_name = (
        metadata.get("views", {}).get("chest", "chest.jpg")
        if row["dataset"] == "TouchAnything"
        else metadata.get("image", "image.jpg")
    )
    image = cv2.imread(str(Path(row["sample_dir"]) / image_name))
    if image is None:
        return
    for key, color in (("query_bbox", (0, 255, 0)), ("query_fov", (255, 180, 0)), ("co_visible_bbox", (0, 0, 255))):
        values = [row.get(f"{key}_{axis}", "") for axis in ("x1", "y1", "x2", "y2")]
        if any(value in (None, "") for value in values):
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in values]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = (
        f"{row['dataset']} {row['sequence_key']} f={row['frame_idx']} "
        f"iou={float(row.get('previous_bbox_iou', -1.0)):.3f}"
    )
    cv2.putText(image, label[:140], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{rank:04d}_{safe_name(row['sequence_key'])}_{row['frame_idx']}.jpg"), image)


def _ensure_integrity_sidecar(dataset, job_roots, split, args, imports):
    sidecar_path = Path(dataset.integrity_cache_path())
    summary_path = Path(dataset.integrity_summary_path())
    if sidecar_path.is_file() and summary_path.is_file():
        return sidecar_path, json.loads(summary_path.read_text(encoding="utf-8"))

    raw_sample_dirs = []
    for job_root in job_roots:
        split_dir = dataset._split_dir(job_root)
        if split_dir is not None:
            raw_sample_dirs.extend(str(path) for path in sorted(Path(split_dir).iterdir()) if path.is_dir())
    raw_sample_dirs.sort()

    workers = max(1, min(int(args.integrity_workers), 64))
    batch_size = max(1, min(int(args.index_chunksize), 64))
    raw_sample_dir_batches = [
        raw_sample_dirs[start : start + batch_size]
        for start in range(0, len(raw_sample_dirs), batch_size)
    ]
    executor_class = ProcessPoolExecutor if args.index_backend == "process" else ThreadPoolExecutor
    temp_path = sidecar_path.with_name(f".{sidecar_path.name}.tmp-{os.getpid()}")
    legacy_candidate_count = 0
    strict_sample_count = 0
    legacy_rejections = []
    lock = imports["cache_build_lock"](
        f"{sidecar_path}.lock",
        timeout_sec=args.index_cache_timeout,
    )
    if not lock.try_acquire():
        print(
            f"[{split}] Another host is building the integrity sidecar "
            f"({lock.owner_description()}); waiting for completion."
        )
        imports["wait_for_file"](summary_path, timeout_sec=args.index_cache_timeout)
        imports["wait_for_file"](sidecar_path, timeout_sec=args.index_cache_timeout)
        return sidecar_path, json.loads(summary_path.read_text(encoding="utf-8"))

    print(
        f"[{split}] Integrity sidecar missing; one metadata-only pass over "
        f"{len(raw_sample_dirs)} sample dirs with {workers} {args.index_backend} workers "
        f"in {len(raw_sample_dir_batches)} batches of at most {batch_size}."
    )
    try:
        if sidecar_path.is_file() and summary_path.is_file():
            return sidecar_path, json.loads(summary_path.read_text(encoding="utf-8"))
        with temp_path.open("w", encoding="utf-8") as handle:
            with executor_class(max_workers=workers) as executor:
                results = executor.map(
                    imports["scan_sample_dirs_integrity_batch"],
                    raw_sample_dir_batches,
                    chunksize=1,
                )
                with tqdm(total=len(raw_sample_dirs), desc=f"Integrity sidecar/{split}") as progress:
                    for result in results:
                        legacy_candidate_count += int(result["legacy_candidate_count"])
                        strict_sample_count += int(result["strict_sample_count"])
                        legacy_rejections.extend(result["rejections"])
                        handle.write(result["audit_jsonl"])
                        progress.update(int(result["sample_dir_count"]))
        os.replace(temp_path, sidecar_path)
        summary = {
            "split": split,
            "sample_dir_count": len(raw_sample_dirs),
            "strict_sample_count": strict_sample_count,
            "legacy_candidate_count": legacy_candidate_count,
            "legacy_index_rejection_count": len(legacy_rejections),
            "legacy_index_rejections": legacy_rejections,
        }
        imports["write_json_atomic"](summary_path, summary)
    finally:
        if temp_path.exists():
            temp_path.unlink()
        lock.release()

    return sidecar_path, summary


def _pressure_stratum(value):
    value = float(value)
    for index, upper in enumerate((0.005, 0.05, 0.2, 0.5, 0.7)):
        if value < upper:
            return index
    return 5


def _stratified_decode_indices(rows, budget):
    if budget < 0 or budget >= len(rows):
        return list(range(len(rows)))
    if budget == 0 or not rows:
        return []
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["dataset"], _pressure_stratum(row["max_pressure"]))].append(index)
    selected = set()
    quota = max(1, budget // max(len(groups), 1))
    for indices in groups.values():
        count = min(quota, len(indices))
        positions = np.linspace(0, len(indices) - 1, num=count, dtype=np.int64)
        selected.update(indices[int(position)] for position in positions)
    if len(selected) < budget:
        remaining = (index for index in range(len(rows)) if index not in selected)
        for index in remaining:
            selected.add(index)
            if len(selected) >= budget:
                break
    return sorted(selected)


def _decode_jpeg_batch(tasks):
    """Check image headers or decode pixels without running the dataset transform."""
    cv2.setNumThreads(0)
    results = []
    for index, image_path, check_mode in tasks:
        try:
            if check_mode == "header":
                width, height = _read_image_header_size(image_path)
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise RuntimeError("OpenCV could not decode image")
                height, width = image.shape[:2]
            results.append({
                "index": index,
                "width": int(width),
                "height": int(height),
                "error": None,
            })
        except Exception as error:
            results.append({
                "index": index,
                "error": f"{type(error).__name__}: {error}",
            })
    return results


def _job_budget(total_budget, job_index, job_count):
    if total_budget < 0:
        return -1
    budget = total_budget // max(job_count, 1)
    return budget + int(job_index < (total_budget % max(job_count, 1)))


def _row_from_integrity_record(record, split, bbox_scale):
    bbox = np.asarray(record["query_bbox"], dtype=np.float32)
    fov = query_fov_bbox(bbox, bbox_scale)
    other_bbox = valid_bbox(record.get("co_visible_bbox"))
    other_volume = float(record.get("co_visible_gt_volume", 0.0) or 0.0)
    frame_idx_value = int(record.get("frame_idx", 0) or 0)
    jq_source_frame = _optional_int(record.get("source_frame_idx"))
    rgb_frame = _sample_dir_frame_index(record["sample_dir"])
    explicit_npz_frame = _optional_int(record.get("pressure_npz_frame_index"))
    checked_sources = [
        (name, value)
        for name, value in (
            ("jq_pressure", jq_source_frame),
            ("rgb_sample_dir", rgb_frame),
            ("pressure_npz", explicit_npz_frame),
        )
        if value is not None
    ]
    mismatched_sources = [name for name, value in checked_sources if value != frame_idx_value]
    rgb_timestamp = _optional_float(record.get("rgb_timestamp"))
    pressure_timestamp = _optional_float(record.get("pressure_timestamp"))
    if pressure_timestamp is None and rgb_timestamp is not None:
        pressure_timestamp = _optional_float(record.get("timestamp"))
    timestamp_offset = (
        pressure_timestamp - rgb_timestamp
        if pressure_timestamp is not None and rgb_timestamp is not None
        else None
    )
    co_visible_high = int(
        other_bbox is not None
        and bbox_intersection_fraction(other_bbox, fov) > 0.0
        and other_volume >= 150.0
    )
    return {
        "sample_dir": record["sample_dir"],
        "dataset": record["dataset"],
        "split": split,
        "sequence_key": record.get("sequence_key", ""),
        "query_alias": record.get("query_alias", record.get("hand", "query")),
        "is_right": int(record.get("is_right", 1)),
        "frame_idx": frame_idx_value,
        "bbox_score": float(record.get("bbox_score", 0.0) or 0.0),
        "pressure_source_key": record.get("pressure_source_key", ""),
        "image_name": record.get("image_name", "chest.jpg" if record["dataset"] == "TouchAnything" else "image.jpg"),
        "target_checksum": record["target_checksum"],
        "target_volume": float(record["target_volume"]),
        "target_active_count": int(record["target_active_count"]),
        "max_pressure": float(record["max_pressure"]),
        "metadata_missing_fields": ",".join(record.get("metadata_missing_fields", [])),
        "source_frame_check_count": len(checked_sources),
        "source_frame_mismatch": int(bool(mismatched_sources)),
        "source_frame_mismatch_count": len(mismatched_sources),
        "source_frame_mismatch_fields": ",".join(mismatched_sources),
        "jq_pressure_frame_idx": "" if jq_source_frame is None else jq_source_frame,
        "rgb_frame_idx": "" if rgb_frame is None else rgb_frame,
        "npz_array_index": frame_idx_value if explicit_npz_frame is None else explicit_npz_frame,
        "npz_array_index_status": "implicit_extractor_frame_idx" if explicit_npz_frame is None else "explicit",
        "timestamp": record.get("timestamp", ""),
        "rgb_timestamp": "" if rgb_timestamp is None else rgb_timestamp,
        "pressure_timestamp": "" if pressure_timestamp is None else pressure_timestamp,
        "cross_modal_timestamp_offset": "" if timestamp_offset is None else timestamp_offset,
        "fov_image_coverage": "",
        "decoded_jpeg": 0,
        "decoded_getitem": 0,
        "co_visible_gt_volume": other_volume,
        "query_low_co_visible_high": int(float(record["target_volume"]) < 10.0 and co_visible_high),
        "previous_bbox_iou": -1.0,
        "center_jump_normalized": -1.0,
        **bbox_columns("query_bbox", bbox),
        **bbox_columns("query_fov", fov),
        **bbox_columns("co_visible_bbox", other_bbox),
    }


def run_data_integrity_audit(args):
    imports = _audit_imports()
    roots = _resolve_audit_roots(args, imports["dataset_roots"])
    if not roots:
        raise ValueError("No dataset roots resolved for data_integrity audit")
    output_dir = Path(args.output_dir or imports["script_dir"] / "data_integrity_audits" / "mixed_v2_input")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_cache_dir = Path(args.index_cache_dir or imports["script_dir"] / "index_cache").expanduser().resolve()
    bbox_scale = float(args.bbox_rescale_factor if args.bbox_rescale_factor is not None else 2.0)

    model_cfg_path = imports["resolve_hamer_model_config_path"](
        imports["workspace_dir"]
    )
    model_cfg = imports["get_config"](str(model_cfg_path), update_cachedir=True)
    if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    rows = []
    sample_failures = []
    metadata_errors = []
    target_mismatches = []
    split_summaries = []
    legacy_index_rejections = []
    legacy_candidate_count = 0
    strict_index_count = 0
    indexed_invalid_bbox_count = 0
    decoded_jpeg_count = 0
    jpeg_decode_failure_count = 0
    decoded_sample_count = 0
    split_roots = defaultdict(list)
    for root in roots:
        splits = _resolve_audit_splits(root, args.splits)
        if not splits:
            sample_failures.append({"root": root, "reason": "no_supported_splits"})
            continue
        for split in splits:
            split_roots[split].append(root)

    split_order = {name: index for index, name in enumerate(("train", "val", "test", "test_seen", "test_unseen"))}
    audit_jobs = sorted(split_roots.items(), key=lambda item: (split_order.get(item[0], 99), item[0]))
    for job_index, (split, job_roots) in enumerate(audit_jobs):
        dataset = imports["dataset_class"](
            cfg=model_cfg,
            split=split,
            data_dir=job_roots,
            train=False,
            index_workers=args.index_workers,
            index_chunksize=args.index_chunksize,
            index_backend=args.index_backend,
            index_cache_dir=str(index_cache_dir),
            rebuild_index=args.rebuild_index,
            index_cache_timeout=args.index_cache_timeout,
            tactile_only=True,
            bbox_rescale_factor=bbox_scale,
        )
        sidecar_path, sidecar_summary = _ensure_integrity_sidecar(
            dataset,
            job_roots,
            split,
            args,
            imports,
        )
        legacy_candidate_count += int(sidecar_summary.get("legacy_candidate_count", 0))
        strict_index_count += int(sidecar_summary.get("strict_sample_count", 0))
        legacy_index_rejections.extend(sidecar_summary.get("legacy_index_rejections", []))

        local_rows = []
        limit = len(dataset) if args.max_samples < 0 else min(len(dataset), args.max_samples)
        print(f"[{split}] Reading integrity sidecar sequentially: {sidecar_path}")
        with sidecar_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(tqdm(handle, total=limit, desc=f"Integrity records/{split}")):
                if index >= limit:
                    break
                record = json.loads(line)
                indexed_record = dataset.samples[index]
                if (
                    record["sample_dir"] != indexed_record["sample_dir"]
                    or record["hand"] != indexed_record["hand"]
                ):
                    sample_failures.append({
                        "sample_dir": record.get("sample_dir", ""),
                        "dataset": record.get("dataset", ""),
                        "hand": record.get("hand", ""),
                        "reason": "integrity_sidecar_order_mismatch",
                    })
                    continue
                row = _row_from_integrity_record(record, split, bbox_scale)
                if row["metadata_missing_fields"]:
                    metadata_errors.append({
                        "sample_dir": row["sample_dir"],
                        "dataset": row["dataset"],
                        "hand": record["hand"],
                        "reason": "missing:" + row["metadata_missing_fields"],
                    })
                bbox_value = [row[f"query_bbox_{axis}"] for axis in ("x1", "y1", "x2", "y2")]
                if not imports["valid_bbox"](bbox_value):
                    indexed_invalid_bbox_count += 1
                local_rows.append(row)

        if len(local_rows) != limit:
            sample_failures.append({
                "root": ",".join(job_roots),
                "reason": f"integrity_sidecar_count_mismatch:{len(local_rows)}!={limit}",
            })

        jpeg_budget = _job_budget(args.decode_samples, job_index, len(audit_jobs))
        jpeg_indices = _stratified_decode_indices(local_rows, jpeg_budget)
        jpeg_workers = max(1, min(int(args.decode_workers), 64))
        jpeg_batch_size = max(1, min(int(args.decode_batch_size), 512))
        jpeg_tasks = [
            (
                index,
                str(Path(local_rows[index]["sample_dir"]) / local_rows[index]["image_name"]),
                args.jpeg_check_mode,
            )
            for index in jpeg_indices
        ]
        jpeg_batches = [
            jpeg_tasks[start : start + jpeg_batch_size]
            for start in range(0, len(jpeg_tasks), jpeg_batch_size)
        ]
        print(
            f"[{split}] Running image {args.jpeg_check_mode} audit on "
            f"{len(jpeg_indices)}/{len(local_rows)} "
            f"samples with {jpeg_workers} {args.jpeg_backend} workers in "
            f"{len(jpeg_batches)} batches."
        )

        if jpeg_batches:
            jpeg_executor = ProcessPoolExecutor if args.jpeg_backend == "process" else ThreadPoolExecutor
            with jpeg_executor(max_workers=jpeg_workers) as executor:
                futures = [executor.submit(_decode_jpeg_batch, batch) for batch in jpeg_batches]
                with tqdm(total=len(jpeg_indices), desc=f"JPEG audit/{split}") as progress:
                    for future in as_completed(futures):
                        results = future.result()
                        for result in results:
                            row = local_rows[result["index"]]
                            if result["error"] is not None:
                                jpeg_decode_failure_count += 1
                                sample_failures.append({
                                    "sample_dir": row["sample_dir"],
                                    "dataset": row["dataset"],
                                    "hand": row["query_alias"],
                                    "reason": f"image_{args.jpeg_check_mode}:{result['error']}",
                                })
                                continue
                            decoded_jpeg_count += 1
                            row["decoded_jpeg"] = 1
                            row["image_check_mode"] = args.jpeg_check_mode
                            image_bounds = np.asarray(
                                [0.0, 0.0, float(result["width"]), float(result["height"])],
                                dtype=np.float32,
                            )
                            fov = np.asarray(
                                [row[f"query_fov_{axis}"] for axis in ("x1", "y1", "x2", "y2")],
                                dtype=np.float32,
                            )
                            row["fov_image_coverage"] = bbox_intersection_fraction(fov, image_bounds)
                        progress.update(len(results))

        getitem_budget = _job_budget(args.getitem_samples, job_index, len(audit_jobs))
        getitem_indices = _stratified_decode_indices(local_rows, getitem_budget)
        getitem_workers = max(1, min(int(args.getitem_workers), 32))
        print(
            f"[{split}] Running complete __getitem__/target audit on "
            f"{len(getitem_indices)}/{len(local_rows)} samples with {getitem_workers} threads."
        )

        def getitem_one(index):
            row = local_rows[index]
            try:
                item = dataset[index]
                actual = item["tactile_signal"].detach().cpu().numpy().astype(np.float32, copy=False)
                actual_masked = actual * dataset.palm_mask.astype(np.float32, copy=False)
                actual_checksum = _array_checksum(actual_masked)
                image_bounds = np.asarray(
                    [0.0, 0.0, float(item["image_width"]), float(item["image_height"])],
                    dtype=np.float32,
                )
                fov = np.asarray(
                    [row[f"query_fov_{axis}"] for axis in ("x1", "y1", "x2", "y2")],
                    dtype=np.float32,
                )
                return {
                    "index": index,
                    "actual_shape": list(actual.shape),
                    "actual_checksum": actual_checksum,
                    "mismatch": actual_checksum != row["target_checksum"],
                    "fov_image_coverage": bbox_intersection_fraction(fov, image_bounds),
                    "error": None,
                }
            except Exception as error:
                return {"index": index, "error": f"{type(error).__name__}: {error}"}

        if getitem_indices:
            results = _bounded_thread_results(getitem_one, getitem_indices, getitem_workers)
            for result in tqdm(results, total=len(getitem_indices), desc=f"__getitem__ audit/{split}"):
                row = local_rows[result["index"]]
                if result["error"] is not None:
                    sample_failures.append({
                        "sample_dir": row["sample_dir"],
                        "dataset": row["dataset"],
                        "hand": row["query_alias"],
                        "reason": result["error"],
                    })
                    continue
                decoded_sample_count += 1
                row["decoded_getitem"] = 1
                row["fov_image_coverage"] = result["fov_image_coverage"]
                if result["mismatch"]:
                    target_mismatches.append({
                        "sample_dir": row["sample_dir"],
                        "dataset": row["dataset"],
                        "hand": row["query_alias"],
                        "expected_shape": [int(dataset.tactile_dim)],
                        "actual_shape": result["actual_shape"],
                        "expected_checksum": row["target_checksum"],
                        "actual_checksum": result["actual_checksum"],
                        "max_abs_diff": -1.0,
                    })

        rows.extend(local_rows)
        split_summaries.append({
            "roots": job_roots,
            "split": split,
            "indexed_samples": len(dataset),
            "audited_samples": len(local_rows),
            "requested_jpeg_decode_samples": len(jpeg_indices),
            "requested_getitem_samples": len(getitem_indices),
            "image_check_mode": args.jpeg_check_mode,
            "integrity_sidecar": str(sidecar_path),
        })

    sequence_groups = defaultdict(list)
    for row in rows:
        sequence_groups[(row["dataset"], row["split"], row["sequence_key"])].append(row)
    discontinuities = []
    sequence_rows = []
    for (dataset_name, split, key), group in sequence_groups.items():
        group.sort(key=lambda row: (int(row["frame_idx"]), row["sample_dir"]))
        timestamp_deltas = []
        timestamp_offsets = []
        side_switch_count = 0
        for row in group:
            offset = _optional_float(row.get("cross_modal_timestamp_offset"))
            if offset is not None:
                timestamp_offsets.append(offset)
        for previous, current in zip(group, group[1:]):
            if int(current["frame_idx"]) != int(previous["frame_idx"]) + 1:
                continue
            previous_bbox = np.asarray([previous[f"query_bbox_{axis}"] for axis in ("x1", "y1", "x2", "y2")])
            current_bbox = np.asarray([current[f"query_bbox_{axis}"] for axis in ("x1", "y1", "x2", "y2")])
            iou = _bbox_iou(previous_bbox, current_bbox)
            previous_center = 0.5 * (previous_bbox[:2] + previous_bbox[2:])
            current_center = 0.5 * (current_bbox[:2] + current_bbox[2:])
            scale = max(float(np.max(previous_bbox[2:] - previous_bbox[:2])), 1.0)
            jump = float(np.linalg.norm(current_center - previous_center) / scale)
            current["previous_bbox_iou"] = iou
            current["center_jump_normalized"] = jump
            current["bbox_discontinuity"] = int(iou < 0.05 or jump > 0.75)
            if current["bbox_discontinuity"]:
                discontinuities.append(current)
            try:
                timestamp_delta = float(current["timestamp"]) - float(previous["timestamp"])
            except (TypeError, ValueError):
                timestamp_delta = None
            if timestamp_delta is not None and np.isfinite(timestamp_delta):
                timestamp_deltas.append(timestamp_delta)
            side_switch_count += int(current["is_right"] != previous["is_right"])
        positive_timestamp_deltas = [value for value in timestamp_deltas if value > 0.0]
        timestamp_period = float(np.median(positive_timestamp_deltas)) if positive_timestamp_deltas else None
        timestamp_offset_median = float(np.median(timestamp_offsets)) if timestamp_offsets else None
        timestamp_offset_mad = (
            float(np.median(np.abs(np.asarray(timestamp_offsets) - timestamp_offset_median)))
            if timestamp_offsets
            else None
        )
        stable_nonzero_offset = bool(
            len(timestamp_offsets) >= 3
            and timestamp_period is not None
            and timestamp_period > 0.0
            and abs(timestamp_offset_median) >= 0.5 * timestamp_period
            and timestamp_offset_mad <= 0.1 * timestamp_period
        )
        sequence_rows.append({
            "dataset": dataset_name,
            "split": split,
            "sequence_key": key,
            "frame_count": len(group),
            "mean_target_volume": float(np.mean([row["target_volume"] for row in group])),
            "frame_fraction_gt02": float(np.mean([row["max_pressure"] >= 0.2 for row in group])),
            "frame_fraction_gt07": float(np.mean([row["max_pressure"] >= 0.7 for row in group])),
            "bbox_discontinuity_count": sum(int(row.get("bbox_discontinuity", 0)) for row in group),
            "timestamp_delta_median": float(np.median(timestamp_deltas)) if timestamp_deltas else "",
            "timestamp_nonpositive_count": sum(value <= 0.0 for value in timestamp_deltas),
            "cross_modal_timestamp_checked_count": len(timestamp_offsets),
            "cross_modal_timestamp_offset_median": "" if timestamp_offset_median is None else timestamp_offset_median,
            "cross_modal_timestamp_offset_mad": "" if timestamp_offset_mad is None else timestamp_offset_mad,
            "stable_nonzero_timestamp_offset": int(stable_nonzero_offset),
            "side_switch_count": side_switch_count,
        })

    source_checked = sum(int(row["source_frame_check_count"]) for row in rows)
    source_mismatches = sum(int(row["source_frame_mismatch_count"]) for row in rows)
    source_mismatch_rate = source_mismatches / max(source_checked, 1)
    metadata_rate = len(metadata_errors) / max(len(rows) + len(metadata_errors), 1)
    dataset_counts = Counter(row["dataset"] for row in rows)
    blocking_reasons = []
    if target_mismatches:
        blocking_reasons.append("target_mismatch")
    if indexed_invalid_bbox_count:
        blocking_reasons.append("indexed_invalid_bbox")
    if sample_failures:
        blocking_reasons.append("indexed_sample_failure")
    if source_mismatch_rate > args.max_source_frame_mismatch_rate:
        blocking_reasons.append("source_frame_mismatch_rate")
    stable_timestamp_offsets = sum(int(row["stable_nonzero_timestamp_offset"]) for row in sequence_rows)
    if stable_timestamp_offsets:
        blocking_reasons.append("stable_nonzero_timestamp_offset")
    baseline_retrain_required = bool(legacy_index_rejections)
    pressure_histogram_edges = (0.0, 0.005, 0.05, 0.2, 0.5, 0.7, float("inf"))
    pressure_histogram = {}
    for lower, upper in zip(pressure_histogram_edges, pressure_histogram_edges[1:]):
        label = f"{lower:g}_{'inf' if np.isinf(upper) else f'{upper:g}'}"
        pressure_histogram[label] = sum(lower <= row["max_pressure"] < upper for row in rows)
    summary = {
        "status": (
            "blocked"
            if blocking_reasons
            else "baseline_retrain_required"
            if baseline_retrain_required
            else "ok"
        ),
        "blocking_reasons": blocking_reasons,
        "audited_samples": len(rows),
        "dataset_filter": sorted(dataset_counts),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "sequence_count": len(sequence_groups),
        "target_mismatch_count": len(target_mismatches),
        "target_mismatch_rate": len(target_mismatches) / max(decoded_sample_count, 1),
        "decoded_jpeg_sample_count": decoded_jpeg_count,
        "decoded_jpeg_fraction": decoded_jpeg_count / max(len(rows), 1),
        "jpeg_decode_failure_count": jpeg_decode_failure_count,
        "jpeg_decode_policy": "full" if args.decode_samples < 0 else "deterministic_stratified",
        "jpeg_decode_backend": args.jpeg_backend,
        "image_check_mode": args.jpeg_check_mode,
        "image_check_failure_count": jpeg_decode_failure_count,
        "decoded_getitem_sample_count": decoded_sample_count,
        "decoded_getitem_fraction": decoded_sample_count / max(len(rows), 1),
        "getitem_policy": "full" if args.getitem_samples < 0 else "deterministic_stratified",
        "decode_policy": f"image_{args.jpeg_check_mode}_plus_bounded_getitem",
        "full_integrity_csv_written": bool(args.write_full_integrity_csv),
        "indexed_invalid_bbox_count": indexed_invalid_bbox_count,
        "indexed_sample_failure_count": len(sample_failures),
        "legacy_index_candidate_count": legacy_candidate_count,
        "strict_index_sample_count": strict_index_count,
        "legacy_index_rejection_count": len(legacy_index_rejections),
        "baseline_retrain_required": baseline_retrain_required,
        "metadata_integrity_error_count": len(metadata_errors),
        "metadata_integrity_error_rate": metadata_rate,
        "metadata_review_required": metadata_rate > args.max_metadata_integrity_error_rate,
        "source_frame_checked_count": source_checked,
        "source_frame_mismatch_count": source_mismatches,
        "source_frame_mismatch_rate": source_mismatch_rate,
        "source_frame_mismatch_threshold": args.max_source_frame_mismatch_rate,
        "stable_nonzero_timestamp_offset_sequence_count": stable_timestamp_offsets,
        "max_pressure_histogram": pressure_histogram,
        "bbox_discontinuity_count": len(discontinuities),
        "identity_policy": "query aliases are offline provenance only and never model inputs",
        "split_summaries": split_summaries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    imports["write_json_atomic"](output_dir / "summary.json", summary)
    write_rows(output_dir / "dataset_sequence_stats.csv", sorted(sequence_rows, key=lambda row: (-row["frame_count"], row["sequence_key"])))
    write_rows(output_dir / "target_mismatches.csv", target_mismatches, fieldnames=(
        "sample_dir", "dataset", "hand", "expected_shape", "actual_shape",
        "expected_checksum", "actual_checksum", "max_abs_diff",
    ))
    write_rows(output_dir / "sample_failures.csv", sample_failures, fieldnames=("sample_dir", "dataset", "hand", "root", "reason"))
    write_rows(
        output_dir / "legacy_index_rejections.csv",
        legacy_index_rejections,
        fieldnames=("sample_dir", "dataset", "hand", "is_right", "reason"),
    )
    write_rows(output_dir / "metadata_integrity_errors.csv", metadata_errors, fieldnames=("sample_dir", "dataset", "hand", "reason"))
    write_rows(output_dir / "bbox_discontinuities.csv", discontinuities)
    if args.write_full_integrity_csv:
        write_rows(output_dir / "data_integrity_samples.csv", rows)
    else:
        write_rows(
            output_dir / "decoded_integrity_samples.csv",
            [row for row in rows if int(row.get("decoded_getitem", 0))],
        )

    overlay_candidates = sorted(
        rows,
        key=lambda row: (
            -int(row["query_low_co_visible_high"]),
            float(row["previous_bbox_iou"]),
            -float(row["center_jump_normalized"]),
        ),
    )
    overlay_dir = output_dir / "query_overlays"
    for rank, row in enumerate(overlay_candidates[: max(0, args.query_overlay_count)], start=1):
        _write_integrity_overlay(overlay_dir, row, rank)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Data integrity audit results: {output_dir}")
    if blocking_reasons:
        raise SystemExit(2)
    if baseline_retrain_required:
        raise SystemExit(3)


def main():
    args = parse_args()
    if args.mode == "data_integrity":
        run_data_integrity_audit(args)
    elif args.mode == "export_manifests":
        export_verified_domain_manifests(args)
    else:
        run_prediction_audit(args)


if __name__ == "__main__":
    main()
