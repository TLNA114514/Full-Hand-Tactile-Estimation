#!/usr/bin/env python3
"""Safely apply reviewed SAM bbox manifests to extracted tactile metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, TypeVar

try:
    import orjson
except ImportError:  # Optional fast path; the standard library remains supported.
    orjson = None

try:
    from .progress import progress
except ImportError:
    from progress import progress


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("rb", buffering=4 * 1024 * 1024) as handle:
        for line_number, line in enumerate(
            progress(handle, desc=f"Read {path.name}", unit="row"),
            1,
        ):
            if not line.strip():
                continue
            try:
                row = orjson.loads(line) if orjson is not None else json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            row["_source_manifest"] = str(path)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb", buffering=4 * 1024 * 1024) as handle:
        for row in rows:
            handle.write(encode_json(row))
            handle.write(b"\n")
            count += 1
    return count


def load_json(path: Path) -> dict:
    payload = path.read_bytes()
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def encode_json(value: dict) -> bytes:
    if orjson is not None:
        return orjson.dumps(value)
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def lexical_absolute_path(path: str | os.PathLike[str]) -> Path:
    """Normalize manifest paths without filesystem-resolving every row."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


T = TypeVar("T")
R = TypeVar("R")


def atomic_write_json(path: Path, value: dict, *, fsync_file: bool = True) -> None:
    temporary = path.with_name(f".{path.name}.sam3bbox.{os.getpid()}.tmp")
    replaced = False
    try:
        payload = encode_json(value)
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            if fsync_file:
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        replaced = True
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)


def bounded_thread_map(
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    desc: str,
    unit: str,
    total: int | None = None,
    result_callback: Callable[[R], None] | None = None,
    progress_weight: Callable[[R], int] | None = None,
    collect_results: bool = True,
) -> list[R]:
    """Map small-file I/O concurrently without queuing the full dataset at once."""

    if total is None and hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]
    weight = progress_weight or (lambda _result: 1)
    results: list[R] = []
    if workers <= 1:
        bar = progress(total=total, desc=desc, unit=unit)
        try:
            for item in items:
                result = function(item)
                if result_callback is not None:
                    result_callback(result)
                if collect_results:
                    results.append(result)
                bar.update(weight(result))
        finally:
            bar.close()
        return results
    batch_size = max(256, workers * 8)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bbox-writeback") as pool:
        bar = progress(total=total, desc=desc, unit=unit)
        iterator = iter(items)
        try:
            while True:
                batch = list(islice(iterator, batch_size))
                if not batch:
                    break
                for result in pool.map(function, batch):
                    if result_callback is not None:
                        result_callback(result)
                    if collect_results:
                        results.append(result)
                    bar.update(weight(result))
        finally:
            bar.close()
    return results


def valid_bbox(value) -> list[float] | None:
    try:
        box = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if len(box) != 4 or not all(math.isfinite(item) for item in box):
        return None
    if box[2] <= box[0] + 1 or box[3] <= box[1] + 1:
        return None
    return box


def manifest_fingerprint(paths: Iterable[Path]) -> dict[str, str]:
    fingerprints = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        fingerprints[str(path)] = digest.hexdigest()
    return fingerprints


def canonical_target_key(row: dict) -> tuple[str, str]:
    dataset = str(row.get("dataset", "")).lower()
    if dataset == "touchanything":
        target = str(row.get("target_hand", ""))
        if target not in {"left", "right"}:
            raise ValueError(f"Missing/invalid target_hand: {target!r}")
    elif dataset == "opentouch":
        target = "bbox"
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    sample_dir = row.get("sample_dir")
    if sample_dir:
        sample_key = str(lexical_absolute_path(sample_dir))
    else:
        sample_key = "unmaterialized:" + "/".join(
            str(row.get(key, ""))
            for key in ("split", "sequence_key", "frame_idx")
        )
    return sample_key, target


def select_rows(rows: Iterable[dict], *, confidence: str) -> tuple[list[dict], list[dict]]:
    def opentouch_evidence_rank(row: dict) -> tuple[float, float, float, float, int]:
        source_rank = {
            "sam3_flow_agreed": 3.0,
            "sam3_native": 2.0,
            "flow_short_bridge": 1.0,
            "semantic_motion_conflict": -3.0,
        }.get(str(row.get("bbox_source", "sam3_native")), 0.0)
        return (
            source_rank,
            float(row.get("prompt_score") or -math.inf),
            float(row.get("flow_confidence") or -math.inf),
            float(row.get("flow_bbox_iou") or -math.inf),
            -int(row.get("raw_track_id") or 0),
        )

    selected: dict[tuple[str, str], dict] = {}
    skipped = []
    for row in progress(rows, desc="Select manifest rows", unit="row"):
        row_confidence = row.get("association_confidence")
        accepted = (
            confidence == "any"
            or row_confidence == confidence
            or (
                confidence == "eligible"
                and row_confidence in {"high", "single_gloved_query"}
            )
        )
        if not accepted:
            skipped.append({"reason": "confidence", **row})
            continue
        try:
            key = canonical_target_key(row)
        except ValueError as exc:
            skipped.append({"reason": str(exc), **row})
            continue
        bbox = valid_bbox(row.get("bbox"))
        if bbox is None:
            skipped.append({"reason": "invalid_bbox", **row})
            continue
        row = {**row, "bbox": bbox}
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue
        if previous["bbox"] != bbox:
            if str(row.get("dataset", "")).lower() != "opentouch" or str(
                previous.get("dataset", "")
            ).lower() != "opentouch":
                raise RuntimeError(
                    "Conflicting boxes for the same extracted target "
                    f"{key}: {previous['bbox']} versus {bbox}"
                )
            loser, winner = sorted(
                (previous, row),
                key=opentouch_evidence_rank,
            )
            selected[key] = winner
            skipped.append(
                {
                    "reason": "opentouch_single_slot_conflict_lower_evidence",
                    "canonical_target": list(key),
                    "selected_bbox": winner["bbox"],
                    "selected_track_id": winner.get("raw_track_id"),
                    **loser,
                }
            )
            continue
        previous_score = float(previous.get("prompt_score") or -math.inf)
        current_score = float(row.get("prompt_score") or -math.inf)
        if current_score > previous_score:
            selected[key] = row
    return [selected[key] for key in sorted(selected)], skipped


def expected_sequence(meta: dict) -> str:
    return "/".join(str(meta.get(key, "")) for key in ("scene", "task", "clip"))


def expected_opentouch_sequence(meta: dict) -> str:
    return "/".join(str(meta.get(key, "")) for key in ("scene", "demo"))


def expected_bbox_source(row: dict, fingerprints: dict[str, str]) -> dict:
    association_policy = (
        "single_gloved_query"
        if str(row["dataset"]).lower() == "opentouch"
        else (row.get("association_evidence") or {}).get(
            "assignment_policy", "legacy_anchor"
        )
    )
    source = {
        "schema": "sam3_bbox_source_v1",
        "association_policy": association_policy,
        "association_confidence": row.get("association_confidence"),
        "raw_track_id": row.get("raw_track_id"),
        "association_id": row.get("association_id"),
        "source_manifest": row["_source_manifest"],
        "source_manifest_sha256": fingerprints[row["_source_manifest"]],
    }
    if any(
        key in row
        for key in (
            "bbox_source",
            "flow_confidence",
            "flow_bbox_iou",
            "flow_anchor_frames",
        )
    ):
        source.update(
            {
                "tracking_bbox_source": row.get("bbox_source", "sam3_native"),
                "flow_confidence": row.get("flow_confidence"),
                "flow_bbox_iou": row.get("flow_bbox_iou"),
                "flow_anchor_frames": list(row.get("flow_anchor_frames", ())),
            }
        )
    return source


def _preflight_row(
    row: dict,
    fingerprints: dict[str, str] | None = None,
    *,
    loaded_meta: dict | None = None,
    loaded_meta_path: Path | None = None,
) -> tuple[dict | None, dict | None]:
    try:
        if not row.get("sample_dir"):
            return None, {"reason": "missing_sample_dir", **row}
        sample_dir = lexical_absolute_path(row["sample_dir"])
        meta_path = loaded_meta_path or (sample_dir / "meta.json")
        if loaded_meta is None:
            if not meta_path.is_file():
                return None, {"reason": "missing_meta", **row}
            try:
                meta = load_json(meta_path)
            except (OSError, ValueError) as exc:
                return None, {"reason": f"invalid_meta:{exc}", **row}
        else:
            meta = loaded_meta
        if int(meta.get("frame_idx", -1)) != int(row["frame_idx"]):
            raise RuntimeError(f"Frame mismatch for {meta_path}")
        dataset = str(row["dataset"]).lower()
        if dataset == "touchanything":
            if str(meta.get("dataset", "")).lower() != "touchanything":
                raise RuntimeError(f"Expected TouchAnything metadata: {meta_path}")
            sequence = expected_sequence(meta)
            target_hand = row["target_hand"]
            target_meta = meta.get("hands", {}).get(target_hand)
            if not isinstance(target_meta, dict):
                raise RuntimeError(f"Missing hands.{target_hand} metadata in {meta_path}")
            old_bbox = target_meta.get("bbox_chest")
            old_score = target_meta.get("bbox_score")
            old_source = target_meta.get("bbox_source")
        elif dataset == "opentouch":
            sequence = expected_opentouch_sequence(meta)
            target_hand = None
            old_bbox = meta.get("bbox")
            old_score = meta.get("bbox_score")
            old_source = meta.get("bbox_source")
        else:
            raise RuntimeError(f"Unsupported dataset {dataset!r}")
        if sequence != str(row["sequence_key"]):
            raise RuntimeError(
                f"Sequence mismatch for {meta_path}: {sequence!r} != {row['sequence_key']!r}"
            )
        planned = {
            "schema": "sam3_bbox_writeback_plan_v1",
            "meta_path": str(meta_path),
            "sample_dir": str(sample_dir),
            "dataset": dataset,
            "split": row["split"],
            "sequence_key": row["sequence_key"],
            "frame_idx": int(row["frame_idx"]),
            "target_hand": target_hand,
            "old_bbox": old_bbox,
            "old_bbox_score": old_score,
            "old_bbox_source": old_source,
            "new_bbox": row["bbox"],
            "new_bbox_score": row.get("prompt_score"),
            "raw_track_id": row.get("raw_track_id"),
            "association_id": row.get("association_id"),
            "association_confidence": row.get("association_confidence"),
            "association_evidence": row.get("association_evidence"),
            "tracking_bbox_source": row.get("bbox_source", "sam3_native"),
            "flow_confidence": row.get("flow_confidence"),
            "flow_bbox_iou": row.get("flow_bbox_iou"),
            "flow_anchor_frames": list(row.get("flow_anchor_frames", ())),
            "source_manifest": row["_source_manifest"],
        }
        if fingerprints is not None:
            new_source = expected_bbox_source(row, fingerprints)
            planned["new_bbox_source"] = new_source
            planned["needs_update"] = bool(
                old_bbox != row["bbox"]
                or old_score != row.get("prompt_score")
                or old_source != new_source
            )
        return planned, None
    except Exception as exc:
        # Keep integrity errors fatal while adding the sample path to their context.
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Failed to preflight {row.get('sample_dir')}: {exc}") from exc


def build_update_plan(
    rows: Iterable[dict],
    *,
    workers: int = 1,
    fingerprints: dict[str, str] | None = None,
) -> tuple[list[dict], list[dict]]:
    row_list = list(rows)

    def grouped_rows() -> Iterable[list[dict]]:
        current_key: str | None = None
        current_rows: list[dict] = []
        for ordinal, row in enumerate(row_list):
            sample_dir = row.get("sample_dir")
            key = (
                str(lexical_absolute_path(sample_dir) / "meta.json")
                if sample_dir
                else f"\0missing:{ordinal}"
            )
            if current_rows and key != current_key:
                yield current_rows
                current_rows = []
            current_key = key
            current_rows.append(row)
        if current_rows:
            yield current_rows

    def preflight_group(group: list[dict]) -> list[tuple[dict | None, dict | None]]:
        first = group[0]
        if not first.get("sample_dir"):
            return [_preflight_row(first, fingerprints)]
        sample_dir = lexical_absolute_path(first["sample_dir"])
        meta_path = sample_dir / "meta.json"
        if not meta_path.is_file():
            return [(None, {"reason": "missing_meta", **row}) for row in group]
        try:
            meta = load_json(meta_path)
        except (OSError, ValueError) as exc:
            return [
                (None, {"reason": f"invalid_meta:{exc}", **row}) for row in group
            ]
        return [
            _preflight_row(
                row,
                fingerprints,
                loaded_meta=meta,
                loaded_meta_path=meta_path,
            )
            for row in group
        ]

    plan: list[dict] = []
    unavailable: list[dict] = []

    def collect_group(
        group_results: list[tuple[dict | None, dict | None]],
    ) -> None:
        for planned, missing in group_results:
            if planned is not None:
                plan.append(planned)
            if missing is not None:
                unavailable.append(missing)

    bounded_thread_map(
        preflight_group,
        grouped_rows(),
        workers=workers,
        desc="Preflight meta.json",
        unit="target",
        total=len(row_list),
        result_callback=collect_group,
        progress_weight=len,
        collect_results=False,
    )
    return plan, unavailable


def backup_rows_from_plan(plan: Iterable[dict]) -> list[dict]:
    return [
        {
            "schema": "sam3_bbox_writeback_backup_v1",
            "meta_path": row["meta_path"],
            "dataset": row["dataset"],
            "target_hand": row["target_hand"],
            "bbox": row["old_bbox"],
            "bbox_score": row["old_bbox_score"],
            "bbox_source": row["old_bbox_source"],
        }
        for row in plan
    ]


def _apply_meta_group(
    payload: tuple[list[dict], dict[str, str], bool]
) -> int:
    rows, fingerprints, fsync_each_file = payload
    meta_path = Path(rows[0]["meta_path"])
    meta = load_json(meta_path)
    changed = False
    for row in rows:
        if not row.get("needs_update", True):
            continue
        if row["dataset"] == "touchanything":
            target_meta = meta["hands"][row["target_hand"]]
            bbox_key = "bbox_chest"
        else:
            target_meta = meta
            bbox_key = "bbox"
        bbox_source = row.get("new_bbox_source")
        if bbox_source is None:
            source_row = {**row, "_source_manifest": row["source_manifest"]}
            bbox_source = expected_bbox_source(source_row, fingerprints)
        if (
            target_meta.get(bbox_key) != row["new_bbox"]
            or target_meta.get("bbox_score") != row["new_bbox_score"]
            or target_meta.get("bbox_source") != bbox_source
        ):
            target_meta[bbox_key] = row["new_bbox"]
            target_meta["bbox_score"] = row["new_bbox_score"]
            target_meta["bbox_source"] = bbox_source
            changed = True
    if changed:
        atomic_write_json(meta_path, meta, fsync_file=fsync_each_file)
    return len(rows)


def apply_plan(
    plan: Iterable[dict],
    fingerprints: dict[str, str],
    *,
    workers: int = 1,
    fsync_each_file: bool = True,
    return_backups: bool = True,
) -> list[dict]:
    plan_list = list(plan)

    def payloads() -> Iterable[tuple[list[dict], dict[str, str], bool]]:
        current_path: str | None = None
        current_rows: list[dict] = []
        for row in plan_list:
            if not row.get("needs_update", True):
                continue
            meta_path = row["meta_path"]
            if current_rows and meta_path != current_path:
                yield current_rows, fingerprints, fsync_each_file
                current_rows = []
            current_path = meta_path
            current_rows.append(row)
        if current_rows:
            yield current_rows, fingerprints, fsync_each_file

    pending_count = sum(row.get("needs_update", True) for row in plan_list)
    bounded_thread_map(
        _apply_meta_group,
        payloads(),
        workers=workers,
        desc="Apply bbox updates",
        unit="target",
        total=pending_count,
        progress_weight=lambda target_count: target_count,
        collect_results=False,
    )
    return backup_rows_from_plan(plan_list) if return_backups else []


def restore_backup(path: Path) -> int:
    rows = read_jsonl(path)
    for row in progress(rows, desc="Restore bbox backup", unit="meta"):
        meta_path = Path(row["meta_path"])
        meta = load_json(meta_path)
        if row.get("dataset", "touchanything") == "touchanything":
            target_meta = meta["hands"][row["target_hand"]]
            bbox_key = "bbox_chest"
        else:
            target_meta = meta
            bbox_key = "bbox"
        target_meta[bbox_key] = row.get("bbox")
        target_meta["bbox_score"] = row.get("bbox_score")
        if row.get("bbox_source") is None:
            target_meta.pop("bbox_source", None)
        else:
            target_meta["bbox_source"] = row["bbox_source"]
        atomic_write_json(meta_path, meta)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--confidence",
        choices=("eligible", "high", "single_gloved_query", "any"),
        default="eligible",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-missing-samples", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Legacy default for both preflight and writeback worker counts.",
    )
    parser.add_argument(
        "--preflight-workers",
        type=int,
        help="Concurrent meta.json readers; overrides --workers for preflight.",
    )
    parser.add_argument(
        "--apply-workers",
        type=int,
        help="Concurrent atomic writers; overrides --workers for apply.",
    )
    parser.add_argument(
        "--fsync-each-file",
        action="store_true",
        help=(
            "Call fsync for every meta.json before atomic rename. This provides stronger "
            "power-loss durability but is substantially slower on shared storage."
        ),
    )
    parser.add_argument("--restore-backup", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    preflight_workers = (
        args.preflight_workers
        if args.preflight_workers is not None
        else args.workers
    )
    apply_workers = (
        args.apply_workers if args.apply_workers is not None else args.workers
    )
    if preflight_workers < 1 or apply_workers < 1:
        raise ValueError("--preflight-workers and --apply-workers must be positive")
    if args.restore_backup is not None:
        count = restore_backup(args.restore_backup.expanduser().resolve())
        print(f"Restored {count} bbox target(s) from {args.restore_backup}")
        return 0
    if not args.manifest:
        raise ValueError("Pass at least one --manifest, or use --restore-backup")
    manifests = [path.expanduser().resolve() for path in args.manifest]
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else manifests[0].parent / "writeback"
    )
    started_at = time.perf_counter()
    phase_seconds: dict[str, float] = {}

    phase_started = time.perf_counter()
    fingerprints = manifest_fingerprint(manifests)
    phase_seconds["manifest_fingerprint"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    all_rows = [row for path in manifests for row in read_jsonl(path)]
    phase_seconds["manifest_read"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    selected, skipped = select_rows(all_rows, confidence=args.confidence)
    phase_seconds["row_selection"] = time.perf_counter() - phase_started

    phase_started = time.perf_counter()
    plan, unavailable = build_update_plan(
        selected,
        workers=preflight_workers,
        fingerprints=fingerprints,
    )
    phase_seconds["metadata_preflight"] = time.perf_counter() - phase_started
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "bbox_writeback_plan.jsonl"
    skipped_path = output_dir / "bbox_writeback_skipped.jsonl"
    pending_rows = [row for row in plan if row.get("needs_update", True)]
    pending_meta_files = len({row["meta_path"] for row in pending_rows})
    phase_started = time.perf_counter()
    write_jsonl(plan_path, pending_rows)
    write_jsonl(skipped_path, [*skipped, *unavailable])
    phase_seconds["plan_artifact_write"] = time.perf_counter() - phase_started
    summary = {
        "schema": "sam3_bbox_writeback_summary_v1",
        "manifest_sha256": fingerprints,
        "selected_rows": len(selected),
        "valid_targets": len(plan),
        "ready_updates": len(pending_rows),
        "skipped_rows": len(skipped),
        "unavailable_samples": len(unavailable),
        "pending_target_updates": len(pending_rows),
        "pending_meta_files": pending_meta_files,
        "already_current_targets": len(plan) - len(pending_rows),
        "applied": False,
        "plan": str(plan_path),
        "skipped": str(skipped_path),
    }
    if args.apply:
        if unavailable and not args.allow_missing_samples:
            raise RuntimeError(
                f"Refusing partial writeback: {len(unavailable)} selected rows have no valid "
                "extracted meta.json. Inspect the skipped file or pass --allow-missing-samples."
            )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = output_dir / f"bbox_writeback_backup_{timestamp}.jsonl"
        # Persist every state that will change before mutating the first metadata file.
        backups = backup_rows_from_plan(pending_rows)
        phase_started = time.perf_counter()
        write_jsonl(backup_path, backups)
        phase_seconds["backup_write"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        apply_plan(
            plan,
            fingerprints,
            workers=apply_workers,
            fsync_each_file=args.fsync_each_file,
            return_backups=False,
        )
        phase_seconds["metadata_apply"] = time.perf_counter() - phase_started
        summary.update({"applied": True, "backup": str(backup_path)})
    phase_seconds["total"] = time.perf_counter() - started_at
    summary["phase_seconds"] = {
        key: round(value, 3) for key, value in phase_seconds.items()
    }
    preflight_seconds = phase_seconds.get("metadata_preflight", 0.0)
    apply_seconds = phase_seconds.get("metadata_apply", 0.0)
    summary["phase_throughput_targets_per_second"] = {
        "metadata_preflight": round(len(selected) / preflight_seconds, 1)
        if preflight_seconds > 0
        else None,
        "metadata_apply": round(len(pending_rows) / apply_seconds, 1)
        if apply_seconds > 0
        else None,
    }
    summary["io_backend"] = "orjson" if orjson is not None else "stdlib_json"
    summary["preflight_workers"] = preflight_workers
    summary["apply_workers"] = apply_workers
    summary_path = output_dir / "bbox_writeback_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
