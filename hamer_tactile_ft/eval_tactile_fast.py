import sys
import os
import argparse
import csv
import json
import queue as queue_module
import traceback
import multiprocessing as mp
import hashlib

from process_lifecycle import (
    configure_supervised_process,
    initialize_worker_parent_death_signal,
)

configure_supervised_process()

import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from pathlib import Path
from tqdm import tqdm

# Parse GPU early
_gpus = ""
for i, arg in enumerate(sys.argv):
    if arg in ('--gpu', '--gpus') and i + 1 < len(sys.argv):
        _gpus = sys.argv[i+1]
        break
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'hamer'))
sys.path.append(os.path.join(base_dir, 'evaluation'))
sys.path.append(os.path.join(base_dir, 'hamer_tactile_ft'))

from hamer.configs import get_config
from train import TactileTrainingModule
from train import _load_checkpoint
from train import file_sha256
from train import load_compatible_state_dict
from train import resolve_data_dirs
from hamer_tactile import (
    CANONICAL_MODEL_INITIALIZATION_ORDER,
    parse_input_resolution,
)
from hamer_config_assets import resolve_hamer_model_config_path
from dataset import OpenTouchTactileDataset, canonical_dataset_filter
from data.indexing import write_jsonl_atomic
from tactile_metrics import (
    CompactTouchAnythingProtocolAccumulator,
    TOUCHANYTHING_CONTACT_THRESHOLD,
    TOUCHANYTHING_MIN_CONTACT_RATIO,
    TOUCHANYTHING_SCENE_CATEGORIES,
    location_distribution_stats,
    merge_compact_touchanything_protocol_stats,
    summarize_compact_touchanything_protocol,
    touchanything_protocol_group_key,
    touchanything_protocol_frame_stats,
    volumetric_iou_stats,
)
from selector_calibration import (
    SELECTOR_CORRECTION_MIN_PRECISION,
    SELECTOR_HISTOGRAM_BINS,
    SELECTOR_LOGIT_MAX,
    SELECTOR_LOGIT_MIN,
    calibrated_correction_counts,
    calibrated_counts,
    selector_histogram_layout,
    selector_histogram_rows,
    selector_threshold_curve,
    summarize_selector_histograms,
)


def recursive_to(value, target):
    if isinstance(value, dict):
        return {key: recursive_to(item, target) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(target)
    if isinstance(value, list):
        return [recursive_to(item, target) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to(item, target) for item in value)
    return value


DIAG_VALUE_BINS = np.linspace(0.0, 1.0, 101, dtype=np.float32)
DIAG_ERROR_BINS = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
DIAG_PRESSURE_BINS = np.array(
    [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.000001],
    dtype=np.float32,
)
DIAG_PRED_TAIL_THRESHOLDS = np.array(
    [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70],
    dtype=np.float32,
)
CORE_LOCATION_DISTRIBUTION_POWER = 2.0
CORE_LOCATION_MIN_GT_PEAK = 0.05
FRAME_DIAG_KEYS = (
    "gt_volume",
    "pred_volume",
    "volume_ratio",
    "volumetric_iou",
    "contact_iou",
    "gt_active_vertices",
    "pred_active_vertices",
    "frame_mae",
    "distribution_viou",
    "volume_matched_viou",
    "pred_mass_on_gt_support",
    "gt_mass_in_pred_support",
    "location_eligible",
    "core_distribution_viou",
    "core_pred_mass_on_gt_support",
    "core_gt_mass_in_pred_support",
    "core_location_eligible",
    "false_high_gt005_pred03_count",
    "false_high_gt005_pred03_excess_volume",
    "false_high_gt005_pred05_count",
    "false_high_gt005_pred05_excess_volume",
    "false_high_gt05_pred03_count",
    "false_high_gt05_pred03_excess_volume",
)
FRAME_PROVENANCE_KEYS = (
    "sample_dir",
    "sample_ref",
    "sequence_key",
    "frame_idx",
    "query_alias",
    "h5_path",
    "frame_row",
    "query_row",
    "dataset",
    "hand",
    "worker_rank",
)
CATASTROPHIC_OVER_GT_MAX = 10.0
CATASTROPHIC_OVER_PRED_MIN = 300.0
CATASTROPHIC_UNDER_GT_MIN = 150.0
CATASTROPHIC_UNDER_PRED_MAX = 50.0
CHECKPOINT_FILENAMES = {
    "loss-best": "best_loss.ckpt",
    "contact-best": "best_contact.ckpt",
    "selector-best": "best_selector.ckpt",
    "last": "last.ckpt",
}


def _safe_name(value):
    text = str(value or "default").strip().lower()
    chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        elif ch in (",", "+", "/", "\\", " "):
            chars.append("_")
    name = "".join(chars).strip("_")
    return name or "default"


def _finite_json_value(value):
    if isinstance(value, dict):
        return {key: _finite_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _report_path(args):
    if args.report_dir:
        report_dir = args.report_dir
    elif args.exp_name:
        report_dir = os.path.join(
            base_dir,
            "hamer_tactile_ft",
            f"eval_reports_{_safe_name(args.exp_name)}_{_safe_name(args.ckpt)}",
        )
    else:
        report_dir = os.path.join(base_dir, "hamer_tactile_ft", "eval_reports")
    os.makedirs(report_dir, exist_ok=True)
    if args.report_name:
        filename = args.report_name
    else:
        dataset_label = args.datasets or args.data_dir or "resolved"
        filename = f"eval_{_safe_name(dataset_label)}_{_safe_name(args.split)}.txt"
    if not filename.endswith(".txt"):
        filename += ".txt"
    return os.path.join(report_dir, filename)


def _prepare_prediction_export(args):
    if not args.prediction_output_dir:
        return
    output_root = Path(args.prediction_output_dir)
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for path in shard_root.glob("worker_*.npz"):
        path.unlink()
    for name in (
        "prediction_config.json",
        "prediction_vertex_indices.npy",
        "sample_records.jsonl",
        "_COMPLETE",
    ):
        path = output_root / name
        if path.exists():
            path.unlink()


def _materialize_prediction_sample_records(args, sample_records):
    """Freeze the post-policy dataset universe used by prediction workers.

    Query manifests may contain rows outside the current reviewed SAM3
    universe. Dataset construction performs that overlay/filter first. The
    artifact must bind the resulting records rather than the unfiltered source
    manifest, otherwise replay would request predictions that were never
    evaluated.
    """

    output_root = Path(args.prediction_output_dir)
    records = []
    for artifact_index, source in enumerate(sample_records):
        record = dict(source)
        sample_uid = str(record.get("sample_uid", ""))
        if not sample_uid:
            raise ValueError(
                "Prediction export requires sample_uid in every normalized "
                f"record; missing at index {artifact_index}"
            )
        record["_artifact_index"] = artifact_index
        records.append(record)
    if not records:
        raise RuntimeError("Prediction export has no records after dataset filtering")
    manifest_path = output_root / "sample_records.jsonl"
    write_jsonl_atomic(
        manifest_path,
        records,
        progress_label="[prediction-export] materializing exact manifest",
    )
    args.sample_records_jsonl = str(manifest_path.resolve(strict=True))
    print(
        "[prediction-export] frozen post-policy sample universe: "
        f"{len(records)} records -> {args.sample_records_jsonl}",
        flush=True,
    )
    return records


def _finalize_prediction_export(args, sample_records):
    if not args.prediction_output_dir:
        return None
    output_root = Path(args.prediction_output_dir)
    shard_paths = sorted((output_root / "shards").glob("worker_*.npz"))
    if not shard_paths:
        raise RuntimeError(f"Prediction export produced no shards under {output_root}")

    observed_indices = []
    observed_uids = []
    prediction_width = None
    vertex_indices = None
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            indices = np.asarray(shard["indices"], dtype=np.int64)
            predictions = np.asarray(shard["predictions"])
            sample_uids = np.asarray(shard["sample_uids"], dtype=str)
            shard_vertex_indices = np.asarray(
                shard["vertex_indices"], dtype=np.int32
            )
        if predictions.ndim != 2 or len(indices) != len(predictions):
            raise RuntimeError(f"Malformed prediction shard: {shard_path}")
        if len(sample_uids) != len(indices):
            raise RuntimeError(f"UID count differs from predictions: {shard_path}")
        if prediction_width is None:
            prediction_width = int(predictions.shape[1])
            vertex_indices = shard_vertex_indices
        elif prediction_width != int(predictions.shape[1]):
            raise RuntimeError("Prediction width differs between worker shards")
        elif not np.array_equal(vertex_indices, shard_vertex_indices):
            raise RuntimeError("Prediction vertex indices differ between worker shards")
        if len(shard_vertex_indices) != int(predictions.shape[1]):
            raise RuntimeError(
                f"Prediction vertex index count differs from payload width: {shard_path}"
            )
        observed_indices.append(indices)
        observed_uids.append(sample_uids)

    indices = np.concatenate(observed_indices)
    sample_uids = np.concatenate(observed_uids)
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    sample_uids = sample_uids[order]
    expected_indices = np.arange(len(sample_records), dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError(
            "Prediction export does not cover the exact sample manifest once: "
            f"expected={len(expected_indices)}, observed={len(indices)}"
        )
    expected_uids = np.asarray(
        [str(record.get("sample_uid", "")) for record in sample_records], dtype=str
    )
    if not np.array_equal(sample_uids, expected_uids):
        mismatch = int(np.flatnonzero(sample_uids != expected_uids)[0])
        raise RuntimeError(
            "Prediction export UID order differs from the exact sample manifest at "
            f"index {mismatch}: observed={sample_uids[mismatch]!r}, "
            f"expected={expected_uids[mismatch]!r}"
        )

    if vertex_indices is None:
        raise RuntimeError("Prediction export did not resolve vertex indices")
    vertex_target = output_root / "prediction_vertex_indices.npy"
    vertex_temporary = output_root / f".prediction_vertex_indices.{os.getpid()}.tmp.npy"
    np.save(vertex_temporary, vertex_indices)
    os.replace(vertex_temporary, vertex_target)
    manifest_path = Path(args.sample_records_jsonl).resolve(strict=True)
    config = {
        "schema": "tactile_exact_prediction_shards_v2",
        "status": "complete",
        "sample_records": str(manifest_path),
        "sample_records_sha256": file_sha256(manifest_path),
        "record_count": len(sample_records),
        "prediction_width": prediction_width,
        "prediction_dtype": "float16",
        "prediction_palm_only": bool(args.prediction_palm_only),
        "vertex_indices": str(vertex_target),
        "vertex_index_count": int(len(vertex_indices)),
        "vertex_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(vertex_indices).tobytes()
        ).hexdigest(),
        "checkpoint": str(Path(args.checkpoint).resolve(strict=True)),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "source_query_manifests": [
            str(Path(path.strip()).expanduser().resolve(strict=True))
            for path in str(args.query_manifests or "").split(",")
            if path.strip()
        ],
        "source_query_manifest_sha256": {
            str(Path(path.strip()).expanduser().resolve(strict=True)): file_sha256(
                path.strip()
            )
            for path in str(args.query_manifests or "").split(",")
            if path.strip()
        },
        "bbox_source_policy": str(args.bbox_source_policy),
        "bbox_manifest_sha256": dict(
            getattr(args, "active_bbox_manifest_sha256", None)
            or args.model_metadata.get("bbox_manifest_sha256")
            or {}
        ),
        "shards": [str(path) for path in shard_paths],
    }
    config_path = output_root / "prediction_config.json"
    temporary = output_root / f".prediction_config.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, config_path)
    complete_path = output_root / "_COMPLETE"
    temporary = output_root / f"._COMPLETE.{os.getpid()}.tmp"
    temporary.write_text(config["sample_records_sha256"] + "\n", encoding="utf-8")
    os.replace(temporary, complete_path)
    print(
        f"[prediction-export] complete records={len(sample_records)} root={output_root}",
        flush=True,
    )
    return config


def _prepare_selector_artifact_export(args):
    if not args.selector_artifact_output_dir:
        return
    output_root = Path(args.selector_artifact_output_dir)
    shard_root = output_root / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for path in shard_root.glob("worker_*.npz"):
        path.unlink()
    for name in ("artifact_config.json", "_COMPLETE"):
        path = output_root / name
        if path.exists():
            path.unlink()


def _finalize_selector_artifact_export(args, sample_records):
    if not args.selector_artifact_output_dir:
        return None
    output_root = Path(args.selector_artifact_output_dir)
    shard_paths = sorted((output_root / "shards").glob("worker_*.npz"))
    if not shard_paths:
        raise RuntimeError(
            f"Selector artifact export produced no shards under {output_root}"
        )

    observed_indices = []
    observed_uids = []
    selector_shape = None
    vertex_indices = None
    thresholds = None
    selector_mode = None
    reference_hashes = []
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            indices = np.asarray(shard["indices"], dtype=np.int64)
            sample_uids = np.asarray(shard["sample_uids"], dtype=str)
            logits = np.asarray(shard["selector_logits"])
            shard_vertices = np.asarray(shard["vertex_indices"], dtype=np.int32)
            shard_thresholds = np.asarray(
                shard["selector_thresholds"], dtype=np.float64
            )
            shard_mode = str(np.asarray(shard["selector_mode"]).item())
            reference_hash = str(
                np.asarray(shard["reference_sha256"]).item()
            )
            has_reference = "base_predictions" in shard and "targets" in shard
            if bool(args.selector_artifact_include_reference) != has_reference:
                raise RuntimeError(
                    f"Selector artifact reference payload differs from CLI: {shard_path}"
                )
            if has_reference:
                base_predictions = np.asarray(shard["base_predictions"])
                targets = np.asarray(shard["targets"])
                if base_predictions.shape != targets.shape or base_predictions.shape != (
                    len(indices),
                    len(shard_vertices),
                ):
                    raise RuntimeError(
                        f"Malformed selector reference arrays: {shard_path}"
                    )
        if logits.ndim != 3 or logits.shape[0] != len(indices):
            raise RuntimeError(f"Malformed selector logit shard: {shard_path}")
        if logits.shape[1] != len(shard_thresholds) or logits.shape[2] != len(
            shard_vertices
        ):
            raise RuntimeError(
                f"Selector thresholds/vertices do not match logits: {shard_path}"
            )
        if len(sample_uids) != len(indices):
            raise RuntimeError(f"UID count differs from selector logits: {shard_path}")
        current_shape = (int(logits.shape[1]), int(logits.shape[2]))
        if selector_shape is None:
            selector_shape = current_shape
            vertex_indices = shard_vertices
            thresholds = shard_thresholds
            selector_mode = shard_mode
        elif (
            selector_shape != current_shape
            or selector_mode != shard_mode
            or not np.array_equal(vertex_indices, shard_vertices)
            or not np.array_equal(thresholds, shard_thresholds)
        ):
            raise RuntimeError("Selector artifact schema differs between worker shards")
        observed_indices.append(indices)
        observed_uids.append(sample_uids)
        reference_hashes.append(reference_hash)

    indices = np.concatenate(observed_indices)
    sample_uids = np.concatenate(observed_uids)
    order = np.argsort(indices, kind="stable")
    indices = indices[order]
    sample_uids = sample_uids[order]
    expected_indices = np.arange(len(sample_records), dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError(
            "Selector artifact does not cover the dataset exactly once: "
            f"expected={len(expected_indices)}, observed={len(indices)}"
        )
    expected_uids = np.asarray(
        [str(record.get("sample_uid", "")) for record in sample_records], dtype=str
    )
    if not np.array_equal(sample_uids, expected_uids):
        mismatch = int(np.flatnonzero(sample_uids != expected_uids)[0])
        raise RuntimeError(
            "Selector artifact UID order differs from the dataset at "
            f"index {mismatch}: observed={sample_uids[mismatch]!r}, "
            f"expected={expected_uids[mismatch]!r}"
        )

    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    config = {
        "schema": "tactile_selector_vertex_artifacts_v1",
        "status": "complete",
        "record_count": len(sample_records),
        "split": str(args.split),
        "datasets": str(args.datasets),
        "selector_mode": selector_mode,
        "selector_thresholds": [float(value) for value in thresholds],
        "selector_output_count": int(selector_shape[0]),
        "valid_vertex_count": int(selector_shape[1]),
        "vertex_indices_sha256": hashlib.sha256(
            np.ascontiguousarray(vertex_indices).tobytes()
        ).hexdigest(),
        "selector_dtype": "float16",
        "reference_payload": bool(args.selector_artifact_include_reference),
        "reference_dtype": (
            "float16" if args.selector_artifact_include_reference else None
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "exp_name": str(args.exp_name),
        "ckpt": str(args.ckpt),
        "input_resolution": list(args.input_resolution),
        "bbox_rescale_factor": float(args.bbox_rescale_factor),
        "bbox_source_policy": str(args.bbox_source_policy),
        "shards": [str(path.resolve()) for path in shard_paths],
        "shard_reference_sha256": reference_hashes,
    }
    config_path = output_root / "artifact_config.json"
    temporary = output_root / f".artifact_config.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, config_path)
    complete_path = output_root / "_COMPLETE"
    temporary = output_root / f"._COMPLETE.{os.getpid()}.tmp"
    temporary.write_text(config["checkpoint_sha256"] + "\n", encoding="utf-8")
    os.replace(temporary, complete_path)
    print(
        "[selector-artifact] complete "
        f"records={len(sample_records)} outputs={selector_shape[0]} "
        f"vertices={selector_shape[1]} root={output_root}",
        flush=True,
    )
    return config


def _archive_training_val_metrics(args):
    output_root = Path(
        args.eval_output_root or args.report_dir or os.path.dirname(_report_path(args))
    ).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve(strict=False)
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path(base_dir) / checkpoint_root
    candidates = [checkpoint_path.parent / "val_metrics.txt"]
    if args.exp_name:
        candidates.append(
            checkpoint_root.expanduser().resolve(strict=False)
            / args.exp_name
            / "val_metrics.txt"
        )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(
            "Training val_metrics.txt is required for evaluation provenance; "
            "checked: " + ", ".join(str(path) for path in candidates)
        )

    source = source.resolve()
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    destination = output_root / "val_metrics.txt"
    if destination.exists():
        existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if existing_digest != digest:
            raise RuntimeError(
                "Eval output root already contains val_metrics.txt from a different "
                f"training run: destination={destination}, existing_sha256="
                f"{existing_digest}, source_sha256={digest}"
            )
    elif destination.resolve(strict=False) != source:
        temporary = output_root / f".val_metrics.txt.tmp.{os.getpid()}"
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)

    record = {
        "source": str(source),
        "destination": str(destination),
        "sha256": digest,
        "size_bytes": len(payload),
    }
    provenance_path = output_root / "val_metrics.provenance.json"
    provenance_payload = (
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing.get("sha256") != digest:
            raise RuntimeError(
                "Existing val_metrics provenance conflicts with the selected "
                f"training run: {provenance_path}"
            )
    else:
        temporary = output_root / f".val_metrics.provenance.json.tmp.{os.getpid()}"
        with temporary.open("wb") as handle:
            handle.write(provenance_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, provenance_path)
    record["provenance"] = str(provenance_path)
    print(
        "Archived training validation metrics: "
        f"{source} -> {destination} (sha256={digest[:12]}...)"
    )
    return record


def _gpu_ids(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _checkpoint_exists(path: Path) -> bool:
    # Path.exists() is false for a broken symlink, but lexists lets us report
    # the link target clearly instead of silently ignoring last.ckpt.
    return os.path.lexists(path)


def _canonical_checkpoint_selector(selector: str) -> str:
    selector = str(selector).strip().lower()
    if selector not in CHECKPOINT_FILENAMES:
        choices = ", ".join(CHECKPOINT_FILENAMES)
        raise ValueError(f"Unsupported checkpoint selector {selector!r}; choose one of: {choices}")
    return selector


def _validate_checkpoint_candidate(path: Path, label: str) -> str:
    if not _checkpoint_exists(path):
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    if path.is_symlink():
        link_target = os.readlink(path)
        resolved = path.resolve(strict=False)
        print(f"{label} checkpoint is symlink: {path} -> {link_target}")
        if not resolved.is_file():
            raise FileNotFoundError(
                f"{label} checkpoint symlink is broken or does not point to a file: "
                f"{path} -> {link_target} (resolved: {resolved})"
            )
        print(f"{label} checkpoint resolved target: {resolved}")
        return str(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint is not a file: {path}")
    return str(path)


def _resolve_checkpoint_path(args):
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        ckpt_path = ckpt_path.resolve(strict=False) if not ckpt_path.is_symlink() else ckpt_path
        selected = _validate_checkpoint_candidate(ckpt_path, "Explicit")
        print(f"Checkpoint selector overridden by --checkpoint: {selected}")
        return selected

    if not args.exp_name:
        raise ValueError(
            "Please provide either --checkpoint /path/to/model.ckpt or "
            "--exp_name <experiment_name> with --ckpt "
            "loss-best|contact-best|selector-best|last."
        )

    checkpoint_root = Path(args.checkpoint_root).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path(base_dir) / checkpoint_root
    exp_dir = (checkpoint_root / args.exp_name).resolve()
    if not exp_dir.is_dir():
        existing = sorted(path.name for path in checkpoint_root.glob("*") if path.is_dir()) if checkpoint_root.is_dir() else []
        suffix = f" Available experiments: {', '.join(existing[:20])}" if existing else ""
        raise FileNotFoundError(f"Checkpoint experiment directory not found: {exp_dir}.{suffix}")

    selector = _canonical_checkpoint_selector(args.ckpt)
    canonical_path = exp_dir / CHECKPOINT_FILENAMES[selector]
    if _checkpoint_exists(canonical_path):
        selected = _validate_checkpoint_candidate(canonical_path, selector)
        print(f"Checkpoint selector: --ckpt {selector}; selected canonical checkpoint: {selected}")
        return selected

    available = sorted(path.name for path in exp_dir.glob("*.ckpt"))
    suffix = f" Available checkpoints: {', '.join(available)}" if available else " No .ckpt files found."
    raise FileNotFoundError(f"Could not resolve --ckpt {selector!r} under {exp_dir}.{suffix}")


def _load_model_cfg(input_resolution=(256, 192)):
    model_cfg_path = resolve_hamer_model_config_path(base_dir)
    model_cfg = get_config(str(model_cfg_path), update_cachedir=True)
    height, width = parse_input_resolution(input_resolution)
    model_cfg.defrost()
    model_cfg.MODEL.IMAGE_SIZE = height
    model_cfg.MODEL.BBOX_SHAPE = [width, height]
    model_cfg.freeze()
    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
    return model_cfg


def _materialize_embedded_surface_basis(
    checkpoint, metadata, checkpoint_path
):
    if str(metadata.get("tactile_head_type", "")) != (
        "dense_v2_dino_surface_basis"
    ):
        return
    recorded_value = str(metadata.get("surface_basis_path", "") or "")
    recorded_path = (
        Path(recorded_value).expanduser() if recorded_value else None
    )
    if recorded_path is not None and recorded_path.is_file():
        metadata["surface_basis_path"] = str(recorded_path.resolve())
        return
    state_dict = checkpoint.get("state_dict", {})
    basis = state_dict.get("tactile_head.surface_basis_valid")
    support_indices = state_dict.get(
        "tactile_head.surface_support_indices"
    )
    support_weights = state_dict.get(
        "tactile_head.surface_support_weights"
    )
    valid_indices = state_dict.get(
        "tactile_head.surface_valid_vertex_indices"
    )
    sparse_payload = isinstance(support_indices, torch.Tensor) and isinstance(
        support_weights, torch.Tensor
    )
    dense_payload = isinstance(basis, torch.Tensor)
    if not isinstance(valid_indices, torch.Tensor) or not (
        sparse_payload or dense_payload
    ):
        raise FileNotFoundError(
            "Surface basis path is unavailable and the compact checkpoint "
            "does not contain embedded basis buffers: "
            f"{recorded_value or '<empty>'}"
        )
    valid_indices = valid_indices.detach().cpu().long().contiguous()
    expected_basis_sha = str(
        metadata.get("surface_basis_tensor_sha256", "") or ""
    )
    artifact_payload = {
        "format": "canonical_surface_basis_v1",
        "valid_vertex_indices": valid_indices,
    }
    if sparse_payload:
        support_indices = (
            support_indices.detach().cpu().long().contiguous()
        )
        support_weights = (
            support_weights.detach().cpu().float().contiguous()
        )
        if (
            support_indices.ndim != 2
            or support_weights.shape != support_indices.shape
            or support_indices.shape[0] != valid_indices.numel()
        ):
            raise RuntimeError(
                "Embedded sparse surface basis has invalid shapes: "
                f"indices={tuple(support_indices.shape)}, "
                f"weights={tuple(support_weights.shape)}"
            )
        coefficient_dim = int(
            metadata.get(
                "surface_coefficient_dim",
                int(support_indices.max().item()) + 1,
            )
        )
        digest = hashlib.sha256()
        for name, value in (
            ("indices", support_indices),
            ("weights", support_weights),
        ):
            digest.update(name.encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(memoryview(value.numpy()).cast("B"))
        sparse_sha = digest.hexdigest()
        expected_sparse_sha = str(
            metadata.get("surface_sparse_basis_sha256", "") or ""
        )
        if expected_sparse_sha and sparse_sha != expected_sparse_sha:
            raise RuntimeError(
                "Embedded sparse surface basis SHA256 mismatch: "
                f"expected={expected_sparse_sha}, actual={sparse_sha}"
            )
        artifact_payload["support_indices"] = support_indices
        artifact_payload["support_weights"] = support_weights
        basis_shape = [int(valid_indices.numel()), coefficient_dim]
        runtime_identity = sparse_sha
    else:
        basis = basis.detach().cpu().float().contiguous()
        coefficient_dim = int(
            metadata.get("surface_coefficient_dim", basis.shape[1])
        )
        if tuple(basis.shape) != (
            int(valid_indices.numel()),
            coefficient_dim,
        ):
            raise RuntimeError(
                f"Embedded surface basis has invalid shape {tuple(basis.shape)}"
            )
        actual_basis_sha = hashlib.sha256(
            memoryview(basis.numpy()).cast("B")
        ).hexdigest()
        if expected_basis_sha and actual_basis_sha != expected_basis_sha:
            raise RuntimeError(
                "Embedded surface basis SHA256 mismatch: "
                f"expected={expected_basis_sha}, actual={actual_basis_sha}"
            )
        sparse_sha = ""
        artifact_payload["basis_valid"] = basis
        basis_shape = list(basis.shape)
        runtime_identity = actual_basis_sha
    runtime_path = Path("/tmp") / (
        f"tactile_surface_basis_{runtime_identity[:24]}_k{coefficient_dim}.pt"
    )
    artifact_metadata = {
        "schema_version": 1,
        "basis_method": "weighted_geodesic_rbf_target_overlap_v1",
        "coefficient_dim": coefficient_dim,
        "target_support_count": int(
            metadata.get("surface_target_support_count", 4)
        ),
        "valid_vertex_count": int(valid_indices.numel()),
        "tactile_dim": int(metadata.get("tactile_dim", 13614)),
        "basis_shape": basis_shape,
        "basis_dtype": "torch.float32",
        "basis_sha256": expected_basis_sha,
        "sparse_basis_sha256": sparse_sha,
        "rehydrated_from_checkpoint": str(Path(checkpoint_path).resolve()),
    }
    temporary_path = runtime_path.with_name(
        f".{runtime_path.name}.partial.{os.getpid()}"
    )
    artifact_payload["metadata"] = artifact_metadata
    torch.save(artifact_payload, temporary_path)
    os.replace(temporary_path, runtime_path)
    metadata["surface_basis_source_path"] = recorded_value
    metadata["surface_basis_path"] = str(runtime_path)
    print(
        "Rehydrated embedded canonical surface basis for evaluation: "
        f"{runtime_path}"
    )


def _load_model(args, model_cfg, device):
    print(f"🚀 初始化模型 (使用设备: {device})...")
    experiment_model_config = dict(getattr(args, "model_metadata", {}) or {})
    tactile_head_type = experiment_model_config.get("tactile_head_type", "dense_v2_dino_rezero")
    backbone_feature_layers = experiment_model_config.get("backbone_feature_layers", [8, 16, 24, 32])
    dino_residual_max_scale = float(experiment_model_config.get("dino_residual_max_scale", 0.10))
    dino_residual_rms_budget = float(experiment_model_config.get("dino_residual_rms_budget", 0.50))
    pool_layout = str(experiment_model_config.get("pool_layout", "fullgrid32"))
    decoder_dropout_scale = float(experiment_model_config.get("decoder_dropout_scale", 1.0))
    input_resolution = parse_input_resolution(
        experiment_model_config.get("input_resolution", (256, 192))
    )
    pool_output_channels = int(experiment_model_config.get("pool_output_channels", 32))
    decoder_hidden_dim = int(experiment_model_config.get("decoder_hidden_dim", 512))
    center_aux_hidden_dim = int(
        experiment_model_config.get("center_aux_hidden_dim", 128)
    )
    model_initialization_order = str(
        experiment_model_config.get(
            "model_initialization_order", CANONICAL_MODEL_INITIALIZATION_ORDER
        )
    )
    local_anchor_count = int(experiment_model_config.get("local_anchor_count", 512))
    local_anchor_neighbors = int(
        experiment_model_config.get("local_anchor_neighbors", 4)
    )
    local_logit_delta_max = float(
        experiment_model_config.get("local_logit_delta_max", 6.0)
    )
    local_residual_dropout = float(
        experiment_model_config.get("local_residual_dropout", 0.10)
    )
    freeze_local_residual_base = bool(
        experiment_model_config.get("freeze_local_residual_base", True)
    )
    surface_basis_path = str(
        experiment_model_config.get("surface_basis_path", "") or ""
    )
    surface_coefficient_dim = int(
        experiment_model_config.get("surface_coefficient_dim", 4096)
    )
    surface_coefficient_architecture = str(
        experiment_model_config.get(
            "surface_coefficient_architecture", "linear"
        )
    )
    surface_coefficient_hidden_dim = int(
        experiment_model_config.get(
            "surface_coefficient_hidden_dim", 1024
        )
    )
    surface_target_support_count = int(
        experiment_model_config.get("surface_target_support_count", 4)
    )
    surface_background_probability = float(
        experiment_model_config.get(
            "surface_background_probability", 1e-3
        )
    )
    freeze_surface_feature_extractor = bool(
        experiment_model_config.get(
            "freeze_surface_feature_extractor", True
        )
    )
    support_selector_mode = str(
        experiment_model_config.get("support_selector_mode", "contact")
    )
    support_selector_thresholds = tuple(
        float(value)
        for value in experiment_model_config.get(
            "support_selector_thresholds", (0.10,)
        )
    )
    support_selector_no_contact_max = float(
        experiment_model_config.get("support_selector_no_contact_max", 0.02)
    )
    support_selector_contact_min = float(
        experiment_model_config.get("support_selector_contact_min", 0.10)
    )
    support_selector_dropout = float(
        experiment_model_config.get("support_selector_dropout", 0.10)
    )
    support_selector_monotonicity_weight = float(
        experiment_model_config.get(
            "support_selector_monotonicity_weight", 0.10
        )
    )
    support_selector_architecture = str(
        experiment_model_config.get("support_selector_architecture", "linear")
    )
    support_selector_feature_source = str(
        experiment_model_config.get(
            "support_selector_feature_source", "fullgrid32"
        )
    )
    support_selector_neck_channels = int(
        experiment_model_config.get("support_selector_neck_channels", 64)
    )
    support_selector_hidden_dim = int(
        experiment_model_config.get("support_selector_hidden_dim", 512)
    )
    support_selector_base_conditioning = str(
        experiment_model_config.get(
            "support_selector_base_conditioning", "real"
        )
    )
    support_selector_correction_min_precision = float(
        experiment_model_config.get(
            "support_selector_correction_min_precision",
            SELECTOR_CORRECTION_MIN_PRECISION,
        )
    )
    init_tactile_checkpoint = str(
        experiment_model_config.get("init_tactile_checkpoint", "") or ""
    )
    visual_backbone = experiment_model_config.get("visual_backbone", "dinov3_hplus")
    dino_weights = getattr(args, "resolved_backbone_weights", "")
    print(
        f"Tactile config: head={tactile_head_type}, visual_backbone={visual_backbone}, "
        f"backbone_feature_layers={backbone_feature_layers}, "
        "fusion=multilevel_rezero"
    )
    model = TactileTrainingModule(
        cfg=model_cfg,
        tactile_head_type=tactile_head_type,
        backbone_feature_layers=backbone_feature_layers,
        visual_backbone=visual_backbone,
        dino_weights=dino_weights,
        dino_residual_max_scale=dino_residual_max_scale,
        dino_residual_rms_budget=dino_residual_rms_budget,
        pool_layout=pool_layout,
        decoder_dropout_scale=decoder_dropout_scale,
        input_resolution=input_resolution,
        pool_output_channels=pool_output_channels,
        decoder_hidden_dim=decoder_hidden_dim,
        center_aux_hidden_dim=center_aux_hidden_dim,
        model_initialization_order=model_initialization_order,
        local_anchor_count=local_anchor_count,
        local_anchor_neighbors=local_anchor_neighbors,
        local_logit_delta_max=local_logit_delta_max,
        local_residual_dropout=local_residual_dropout,
        freeze_local_residual_base=freeze_local_residual_base,
        surface_basis_path=surface_basis_path,
        surface_coefficient_dim=surface_coefficient_dim,
        surface_coefficient_architecture=(
            surface_coefficient_architecture
        ),
        surface_coefficient_hidden_dim=surface_coefficient_hidden_dim,
        surface_target_support_count=surface_target_support_count,
        surface_background_probability=surface_background_probability,
        freeze_surface_feature_extractor=(
            freeze_surface_feature_extractor
        ),
        support_selector_mode=support_selector_mode,
        support_selector_thresholds=support_selector_thresholds,
        support_selector_no_contact_max=support_selector_no_contact_max,
        support_selector_contact_min=support_selector_contact_min,
        support_selector_dropout=support_selector_dropout,
        support_selector_monotonicity_weight=(
            support_selector_monotonicity_weight
        ),
        support_selector_architecture=support_selector_architecture,
        support_selector_feature_source=support_selector_feature_source,
        support_selector_neck_channels=support_selector_neck_channels,
        support_selector_hidden_dim=support_selector_hidden_dim,
        support_selector_base_conditioning=(
            support_selector_base_conditioning
        ),
        support_selector_correction_min_precision=(
            support_selector_correction_min_precision
        ),
        init_tactile_checkpoint=init_tactile_checkpoint,
    )
    model.visual_backbone_model_name = experiment_model_config.get("visual_backbone_model_name", "")
    model.backbone_weights_path = getattr(args, "resolved_backbone_weights", "")
    model.backbone_weights_sha256 = getattr(args, "resolved_backbone_sha256", "")
    dummy_input = torch.zeros(1, 3, *input_resolution)
    with torch.no_grad():
        dummy_feat = model._extract_tactile_features(dummy_input)
        model.tactile_head(dummy_feat)
        print(f"Tactile head initialized with output dim: {model.tactile_dim}")

    print(f"📦 Loading checkpoint from: {args.checkpoint}")
    load_compatible_state_dict(model, args.checkpoint, load_backbone=False)
    calibration = experiment_model_config.get("support_selector_calibration")
    if isinstance(calibration, dict):
        model.support_selector_calibration = calibration
    model = model.to(device)
    model.eval()
    return model


def _resolve_experiment_model_metadata(args):
    metadata = {}
    model_config_path = Path(args.checkpoint).parent / "model_config.json"
    if model_config_path.is_file():
        with model_config_path.open("r", encoding="utf-8") as config_file:
            metadata.update(json.load(config_file))

    checkpoint = _load_checkpoint(args.checkpoint)
    if isinstance(checkpoint, dict) and checkpoint.get("format") == "tactile_trainable_v2":
        metadata.update(checkpoint.get("model_config", {}))
        for key in (
                "visual_backbone",
                "visual_backbone_model_name",
                "backbone_weights",
                "backbone_sha256",
                "tactile_head_type",
                "backbone_feature_layers",
                "dino_residual_max_scale",
                "dino_residual_rms_budget",
                "pool_layout",
                "input_resolution",
                "pool_grid_size",
                "pool_valid_tokens",
                "decoder_input_dim",
                "pool_output_channels",
                "decoder_hidden_dim",
                "center_aux_hidden_dim",
                "model_initialization_order",
                "worker_seed_mode",
                "hdf5_sample_order",
                "hdf5_ordered_sample_sha256",
                "val_hdf5_ordered_sample_sha256",
                "hdf5_sample_set_sha256",
                "crop_pipeline",
                "replay_profile",
                "initial_tactile_head_sha256",
                "optimizer_backend_mode",
                "decoder_dropout_scale",
                "local_anchor_count",
                "local_anchor_neighbors",
                "local_logit_delta_max",
                "local_residual_dropout",
                "freeze_local_residual_base",
                "surface_basis_path",
                "surface_basis_artifact_sha256",
                "surface_basis_tensor_sha256",
                "surface_sparse_basis_sha256",
                "surface_valid_vertex_count",
                "surface_maximum_support_count",
                "surface_coefficient_dim",
                "surface_coefficient_architecture",
                "surface_coefficient_hidden_dim",
                "surface_target_support_count",
                "surface_background_probability",
                "freeze_surface_feature_extractor",
                "support_selector_mode",
                "support_selector_thresholds",
                "support_selector_no_contact_max",
                "support_selector_contact_min",
                "support_selector_dropout",
                "support_selector_monotonicity_weight",
                "support_selector_architecture",
                "support_selector_feature_source",
                "support_selector_neck_channels",
                "support_selector_hidden_dim",
                "support_selector_base_conditioning",
                "support_selector_correction_min_precision",
                "support_selector_calibration",
                "init_tactile_checkpoint",
                "init_tactile_checkpoint_sha256",
                "index_schema_version",
                "index_cache_key",
                "indexed_sample_count",
                "index_manifest_sha256",
                "data_backend",
                "query_manifest_sha256",
                "hdf5_schema_version",
                "hdf5_handle_cache_size",
                "hdf5_manifest_cache_dir",
                "hdf5_manifest_cache_key",
                "bbox_manifest_sha256",
                "dataset_filter",
                "bbox_rescale_factor",
                "bbox_source_policy",
                "train_augmentation",
        ):
            if checkpoint.get(key) not in (None, "", []):
                metadata[key] = checkpoint[key]
        _materialize_embedded_surface_basis(
            checkpoint, metadata, args.checkpoint
        )

    visual_backbone = str(metadata.get("visual_backbone", "dinov3_hplus"))
    if visual_backbone != "dinov3_hplus":
        raise ValueError("Only visual_backbone=dinov3_hplus is supported")
    weights_value = args.dino_weights or metadata.get("backbone_weights")
    if not weights_value:
        raise ValueError(
            "DINOv3 evaluation requires --dino_weights or backbone_weights in the compact checkpoint"
        )

    weights_path = Path(weights_value).expanduser()
    if not weights_path.is_absolute():
        weights_path = Path(args.checkpoint).parent / weights_path
    weights_path = weights_path.resolve(strict=False)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Frozen backbone weights not found: {weights_path}")

    expected_hash = str(metadata.get("backbone_sha256", "") or "")
    actual_hash = file_sha256(weights_path) if expected_hash else ""
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"Backbone SHA256 mismatch: expected={expected_hash}, actual={actual_hash}")

    metadata["visual_backbone"] = visual_backbone
    metadata["backbone_weights"] = str(weights_path)
    checkpoint_resolution = parse_input_resolution(
        metadata.get("input_resolution", (256, 192))
    )
    if args.input_resolution is not None:
        requested_resolution = parse_input_resolution(args.input_resolution)
        if requested_resolution != checkpoint_resolution:
            raise ValueError(
                "Explicit --input_resolution conflicts with checkpoint metadata: "
                f"requested={requested_resolution}, checkpoint={checkpoint_resolution}"
            )
    args.input_resolution = checkpoint_resolution
    metadata["input_resolution"] = list(checkpoint_resolution)
    checkpoint_crop_pipeline = str(
        metadata.get("crop_pipeline", "legacy_square_center")
    )
    if args.crop_pipeline is not None and args.crop_pipeline != checkpoint_crop_pipeline:
        raise ValueError(
            "Explicit --crop_pipeline conflicts with checkpoint metadata: "
            f"requested={args.crop_pipeline}, checkpoint={checkpoint_crop_pipeline}"
        )
    args.crop_pipeline = checkpoint_crop_pipeline
    metadata["crop_pipeline"] = checkpoint_crop_pipeline
    checkpoint_sample_order = str(
        metadata.get("hdf5_sample_order", "legacy_sample_dir_hand")
    )
    if (
        args.hdf5_sample_order is not None
        and args.hdf5_sample_order != checkpoint_sample_order
    ):
        raise ValueError(
            "Explicit --hdf5_sample_order conflicts with checkpoint metadata: "
            f"requested={args.hdf5_sample_order}, checkpoint={checkpoint_sample_order}"
        )
    args.hdf5_sample_order = checkpoint_sample_order
    metadata["hdf5_sample_order"] = checkpoint_sample_order
    bbox_rescale_factor = (
        args.bbox_rescale_factor
        if args.bbox_rescale_factor is not None
        else metadata.get("bbox_rescale_factor", 2.0)
    )
    bbox_rescale_factor = float(bbox_rescale_factor)
    if not 1.0 <= bbox_rescale_factor <= 4.0:
        raise ValueError("bbox_rescale_factor must lie in [1.0, 4.0]")
    args.bbox_rescale_factor = bbox_rescale_factor
    metadata["bbox_rescale_factor"] = bbox_rescale_factor
    bbox_source_policy = (
        args.bbox_source_policy
        if args.bbox_source_policy is not None
        else metadata.get("bbox_source_policy", "any")
    )
    bbox_source_policy = str(bbox_source_policy)
    if bbox_source_policy not in ("any", "sam3_only"):
        raise ValueError(f"Unsupported bbox_source_policy: {bbox_source_policy!r}")
    args.bbox_source_policy = bbox_source_policy
    metadata["bbox_source_policy"] = bbox_source_policy
    bbox_manifest_hashes = metadata.get("bbox_manifest_sha256") or {}
    if args.bbox_manifests is None and isinstance(bbox_manifest_hashes, dict):
        args.bbox_manifests = ",".join(str(path) for path in bbox_manifest_hashes)
    if args.bbox_manifests:
        resolved_manifests = [
            str(Path(path).expanduser().resolve(strict=False))
            for path in str(args.bbox_manifests).split(",")
            if path.strip()
        ]
        missing = [path for path in resolved_manifests if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                "Evaluation SAM3 bbox manifest(s) are missing: " + ", ".join(missing)
            )
        args.bbox_manifests = ",".join(resolved_manifests)
        # The current SAM3 manifests define the evaluated sample universe. An old
        # audit CSV must not silently override it.
        args.index_manifest = None
    args.model_metadata = metadata
    args.resolved_backbone_weights = str(weights_path)
    args.resolved_backbone_sha256 = actual_hash or expected_hash


def _selector_contact_index(thresholds, contact_min):
    matches = [
        index
        for index, value in enumerate(thresholds)
        if np.isclose(float(value), float(contact_min), rtol=0.0, atol=1e-8)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Support selector thresholds must contain contact_min exactly once"
        )
    return matches[0]


def _resolve_selector_calibration(args):
    is_selector = (
        str(args.model_metadata.get("tactile_head_type", ""))
        == "dense_v2_dino_support_selector"
    )
    if not is_selector:
        if args.selector_calibration_input or args.selector_calibration_output:
            raise ValueError(
                "Selector calibration options require a support-selector checkpoint"
            )
        args.selector_checkpoint_sha256 = ""
        args.selector_calibration = None
        args.selector_calibration_source = ""
        return
    args.selector_checkpoint_sha256 = file_sha256(args.checkpoint)
    calibration = None
    calibration_source = ""
    if args.selector_calibration_input:
        path = Path(args.selector_calibration_input)
        with path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
        expected_sha = str(artifact.get("checkpoint_sha256", "") or "")
        if expected_sha and expected_sha != args.selector_checkpoint_sha256:
            raise ValueError(
                "Selector calibration checkpoint SHA256 mismatch: "
                f"calibration={expected_sha}, checkpoint="
                f"{args.selector_checkpoint_sha256}"
            )
        calibration = artifact.get("calibration", artifact)
        calibration_source = str(path)
    else:
        embedded = args.model_metadata.get("support_selector_calibration")
        if isinstance(embedded, dict) and embedded:
            calibration = embedded
            calibration_source = "compact_checkpoint"

    if calibration is not None:
        configured_thresholds = tuple(
            float(value)
            for value in args.model_metadata.get(
                "support_selector_thresholds", (0.10,)
            )
        )
        calibrated_thresholds = tuple(
            float(value) for value in calibration.get("thresholds", ())
        )
        if calibrated_thresholds != configured_thresholds:
            raise ValueError(
                "Selector calibration output thresholds do not match checkpoint: "
                f"calibration={calibrated_thresholds}, "
                f"checkpoint={configured_thresholds}"
            )
        expected_mode = str(
            args.model_metadata.get("support_selector_mode", "contact")
        )
        calibrated_mode = str(calibration.get("selector_mode", expected_mode))
        if calibrated_mode != expected_mode:
            raise ValueError(
                "Selector calibration mode does not match checkpoint: "
                f"calibration={calibrated_mode}, checkpoint={expected_mode}"
            )
    args.selector_calibration = calibration
    args.selector_calibration_source = calibration_source


def _empty_stats():
    return {
        "total_frames": 0,
        "total_values": 0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "pcc_sum": 0.0,
        "pcc_count": 0,
        "temporal_correct": 0,
        "contact_iou_sum": 0.0,
        "vol_iou_sum": 0.0,
        "vol_intersection_sum": 0.0,
        "vol_union_sum": 0.0,
        "tactile_dim": 0,
        "pred_volume": 0.0,
        "gt_volume": 0.0,
        "active_abs_sum": 0.0,
        "active_count": 0,
        "background_abs_sum": 0.0,
        "background_count": 0,
        "active_true_positive": 0,
        "active_gt_count": 0,
        "background_false_positive": 0,
        "catastrophic_over_count": 0,
        "catastrophic_over_denominator": 0,
        "catastrophic_under_count": 0,
        "catastrophic_under_denominator": 0,
        "false_high_gt005_pred03_count": 0,
        "false_high_gt005_pred03_denominator": 0,
        "false_high_gt005_pred03_excess_volume": 0.0,
        "distribution_viou_sum": 0.0,
        "pred_mass_on_gt_support_sum": 0.0,
        "gt_mass_in_pred_support_sum": 0.0,
        "nonempty_location_frame_count": 0,
        "core_distribution_viou_sum": 0.0,
        "core_pred_mass_on_gt_support_sum": 0.0,
        "core_gt_mass_in_pred_support_sum": 0.0,
        "core_location_frame_count": 0,
    }


def _stats_summary(
    stats,
    touchanything_protocol_stats=None,
    touchanything_min_contact_ratio=TOUCHANYTHING_MIN_CONTACT_RATIO,
):
    if stats["total_frames"] == 0 or stats["total_values"] == 0:
        return None
    total_values = max(stats["total_values"], 1)
    total_frames = max(stats["total_frames"], 1)
    touch_summary = summarize_compact_touchanything_protocol(
        [touchanything_protocol_stats or {}],
        min_contact_ratio=touchanything_min_contact_ratio,
    )
    return {
        "mae": stats["abs_sum"] / total_values,
        "rmse": float(np.sqrt(stats["sq_sum"] / total_values)),
        "pcc": stats["pcc_sum"] / max(stats["pcc_count"], 1),
        "contact_iou": stats["contact_iou_sum"] / total_frames,
        "volumetric_iou": stats["vol_iou_sum"] / total_frames,
        "volumetric_iou_frame_macro": stats["vol_iou_sum"] / total_frames,
        "volumetric_iou_split_micro": (
            stats["vol_intersection_sum"] / stats["vol_union_sum"]
            if stats["vol_union_sum"] > 1e-12
            else 1.0
        ),
        "volumetric_iou_touchanything_trajectory": touch_summary["volumetric_iou"],
        "touchanything_protocol": {key: value for key, value in touch_summary.items() if key != "rows"},
        "pred_gt_volume_ratio": stats["pred_volume"] / max(stats["gt_volume"], 1e-6),
        "active_recall": stats["active_true_positive"] / max(stats["active_gt_count"], 1),
        "bg_false_positive": stats["background_false_positive"] / max(stats["background_count"], 1),
        "false_high_gt005_pred03_rate": (
            stats["false_high_gt005_pred03_count"]
            / max(stats["false_high_gt005_pred03_denominator"], 1)
        ),
        "false_high_gt005_pred03_excess_volume_fraction": (
            stats["false_high_gt005_pred03_excess_volume"]
            / max(stats["pred_volume"], 1e-6)
        ),
        "catastrophic_over_rate": (
            stats["catastrophic_over_count"]
            / max(stats["catastrophic_over_denominator"], 1)
        ),
        "distribution_viou": (
            stats["distribution_viou_sum"] / max(stats["nonempty_location_frame_count"], 1)
        ),
        "volume_matched_viou": (
            stats["distribution_viou_sum"] / max(stats["nonempty_location_frame_count"], 1)
        ),
        "pred_mass_on_gt_support": (
            stats["pred_mass_on_gt_support_sum"] / max(stats["nonempty_location_frame_count"], 1)
        ),
        "gt_mass_in_pred_support": (
            stats["gt_mass_in_pred_support_sum"] / max(stats["nonempty_location_frame_count"], 1)
        ),
        "nonempty_location_frame_count": stats["nonempty_location_frame_count"],
        "core_distribution_viou": (
            stats["core_distribution_viou_sum"] / max(stats["core_location_frame_count"], 1)
        ),
        "core_pred_mass_on_gt_support": (
            stats["core_pred_mass_on_gt_support_sum"]
            / max(stats["core_location_frame_count"], 1)
        ),
        "core_gt_mass_in_pred_support": (
            stats["core_gt_mass_in_pred_support_sum"]
            / max(stats["core_location_frame_count"], 1)
        ),
        "core_location_frame_count": stats["core_location_frame_count"],
    }


def _resolve_invocation_path(path):
    """Resolve user-supplied paths before worker processes are spawned."""
    if not path:
        return None
    return str(Path(path).expanduser().resolve(strict=False))


def _empty_diagnostics():
    n_value_bins = len(DIAG_VALUE_BINS) - 1
    n_error_bins = len(DIAG_ERROR_BINS) - 1
    n_pressure_bins = len(DIAG_PRESSURE_BINS) - 1
    tail_shape = (n_pressure_bins, len(DIAG_PRED_TAIL_THRESHOLDS))
    return {
        "gt_hist": np.zeros(n_value_bins, dtype=np.float64),
        "pred_hist": np.zeros(n_value_bins, dtype=np.float64),
        "error_hist": np.zeros(n_error_bins, dtype=np.float64),
        "pressure_count": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_gt_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_pred_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_err_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_abs_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_over_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_under_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_pred_active": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_gt_active": np.zeros(n_pressure_bins, dtype=np.float64),
        "pred_tail_count": np.zeros(tail_shape, dtype=np.float64),
        "pred_tail_pred_sum": np.zeros(tail_shape, dtype=np.float64),
        "pred_tail_gt_sum": np.zeros(tail_shape, dtype=np.float64),
        "frame": {key: [] for key in FRAME_DIAG_KEYS},
        "frame_provenance": {key: [] for key in FRAME_PROVENANCE_KEYS},
    }


def _empty_support_selector_stats(thresholds=()):
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds:
        return {}
    return {
        "thresholds": thresholds,
        "selector_all": np.zeros(4, dtype=np.float64),
        "base_all": np.zeros(4, dtype=np.float64),
        "selector_clear": np.zeros(4, dtype=np.float64),
        "base_clear": np.zeros(4, dtype=np.float64),
        "false_high_detected": 0.0,
        "false_high_count": 0.0,
        "false_low_recovered": 0.0,
        "false_low_count": 0.0,
        "disagreement_count": 0.0,
        "valid_count": 0.0,
        "ordinal_abs_error": 0.0,
        "ordinal_exact": 0.0,
        "ordinal_count": 0.0,
        "monotonic_violation": 0.0,
        "monotonic_count": 0.0,
        "cumulative": np.zeros((len(thresholds), 4), dtype=np.float64),
        "calibration_histogram": np.zeros(
            (
                selector_histogram_rows(len(thresholds)),
                SELECTOR_HISTOGRAM_BINS,
            ),
            dtype=np.float64,
        ),
    }


def _binary_counts_numpy(prediction, target, mask):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    return np.asarray(
        [
            np.sum(prediction & target & mask),
            np.sum(prediction & ~target & mask),
            np.sum(~prediction & target & mask),
            np.sum(~prediction & ~target & mask),
        ],
        dtype=np.float64,
    )


def _update_support_selector_stats(
    stats,
    selector_logits,
    selector_probabilities,
    base_prediction,
    target,
    palm_mask,
    *,
    no_contact_max,
    contact_min,
    selector_mode,
):
    thresholds = tuple(float(value) for value in stats["thresholds"])
    contact_matches = [
        index
        for index, value in enumerate(thresholds)
        if np.isclose(value, float(contact_min), rtol=0.0, atol=1e-8)
    ]
    if len(contact_matches) != 1:
        raise RuntimeError(
            "Support selector thresholds must contain contact_min exactly once"
        )
    palm = np.asarray(palm_mask, dtype=bool)[None, :]
    palm = np.broadcast_to(palm, target.shape)
    selector_contact = selector_probabilities[:, contact_matches[0]] >= 0.5
    base_contact = base_prediction >= float(contact_min)
    gt_contact = target >= float(contact_min)
    clear = palm & ((target <= float(no_contact_max)) | gt_contact)
    selector_eligible = palm
    if str(selector_mode) == "down_error":
        selector_eligible = clear & base_contact
    stats["selector_all"] += _binary_counts_numpy(
        selector_contact, gt_contact, selector_eligible
    )
    stats["base_all"] += _binary_counts_numpy(
        base_contact, gt_contact, selector_eligible
    )
    stats["selector_clear"] += _binary_counts_numpy(
        selector_contact, gt_contact, clear & selector_eligible
    )
    stats["base_clear"] += _binary_counts_numpy(
        base_contact, gt_contact, clear & selector_eligible
    )
    false_high = clear & ~gt_contact & base_contact
    false_low = clear & gt_contact & ~base_contact
    stats["false_high_detected"] += float(
        np.sum(false_high & ~selector_contact)
    )
    stats["false_high_count"] += float(np.sum(false_high))
    if str(selector_mode) != "down_error":
        stats["false_low_recovered"] += float(
            np.sum(false_low & selector_contact)
        )
    stats["false_low_count"] += float(np.sum(false_low))
    stats["disagreement_count"] += float(
        np.sum((selector_contact != base_contact) & selector_eligible)
    )
    stats["valid_count"] += float(np.sum(selector_eligible))

    gt_bin = np.stack([target > threshold for threshold in thresholds], axis=1).sum(axis=1)
    if str(selector_mode) == "down_error":
        gt_bin = gt_contact.astype(gt_bin.dtype, copy=False)
    pred_bin = (selector_probabilities >= 0.5).sum(axis=1)
    stats["ordinal_abs_error"] += float(
        np.sum(np.abs(pred_bin - gt_bin) * selector_eligible)
    )
    stats["ordinal_exact"] += float(
        np.sum((pred_bin == gt_bin) & selector_eligible)
    )
    stats["ordinal_count"] += float(np.sum(selector_eligible))
    if len(thresholds) > 1:
        expanded_palm = np.broadcast_to(
            selector_eligible[:, None, :],
            (palm.shape[0], len(thresholds) - 1, palm.shape[1]),
        )
        stats["monotonic_violation"] += float(
            np.sum(
                (selector_probabilities[:, 1:] > selector_probabilities[:, :-1])
                & expanded_palm
            )
        )
        stats["monotonic_count"] += float(np.sum(expanded_palm))
    for index, threshold in enumerate(thresholds):
        cumulative_target = target > threshold
        if str(selector_mode) == "down_error":
            cumulative_target = gt_contact
        stats["cumulative"][index] += _binary_counts_numpy(
            selector_probabilities[:, index] >= 0.5,
            cumulative_target,
            selector_eligible,
        )

    logits = np.asarray(selector_logits, dtype=np.float32)
    if logits.shape != selector_probabilities.shape:
        raise ValueError("Selector logits/probabilities have different shapes")
    if not np.isfinite(logits).all():
        raise FloatingPointError("Support selector produced non-finite eval logits")
    scale = SELECTOR_HISTOGRAM_BINS / (
        SELECTOR_LOGIT_MAX - SELECTOR_LOGIT_MIN
    )
    bin_indices = np.floor(
        (np.clip(logits, SELECTOR_LOGIT_MIN, SELECTOR_LOGIT_MAX)
         - SELECTOR_LOGIT_MIN)
        * scale
    ).astype(np.int64)
    np.clip(bin_indices, 0, SELECTOR_HISTOGRAM_BINS - 1, out=bin_indices)
    histogram = stats["calibration_histogram"]
    layout = selector_histogram_layout(len(thresholds))

    def add_histogram(row, indices, mask):
        selected = indices[np.asarray(mask, dtype=bool)]
        if selected.size:
            histogram[int(row)] += np.bincount(
                selected,
                minlength=SELECTOR_HISTOGRAM_BINS,
            )

    for output_index, threshold in enumerate(thresholds):
        labels = (
            target >= float(contact_min)
            if str(selector_mode) in {"contact", "down_error"}
            else target > threshold
        )
        pair = layout["cumulative"][output_index]
        add_histogram(
            pair["positive"],
            bin_indices[:, output_index],
            selector_eligible & labels,
        )
        add_histogram(
            pair["negative"],
            bin_indices[:, output_index],
            selector_eligible & ~labels,
        )
    contact_bins = bin_indices[:, contact_matches[0]]
    add_histogram(
        layout["clear_positive"],
        contact_bins,
        clear & selector_eligible & gt_contact,
    )
    add_histogram(
        layout["clear_negative"],
        contact_bins,
        clear & selector_eligible & ~gt_contact,
    )
    add_histogram(layout["false_high"], contact_bins, false_high)
    add_histogram(
        layout["base_true_positive"],
        contact_bins,
        clear & gt_contact & base_contact,
    )
    if str(selector_mode) != "down_error":
        add_histogram(layout["false_low"], contact_bins, false_low)
        add_histogram(
            layout["base_true_negative"],
            contact_bins,
            clear & ~gt_contact & ~base_contact,
        )


def _merge_support_selector_stats(items):
    items = [item for item in items if item]
    if not items:
        return {}
    thresholds = tuple(float(value) for value in items[0]["thresholds"])
    merged = _empty_support_selector_stats(thresholds)
    for item in items:
        if tuple(float(value) for value in item["thresholds"]) != thresholds:
            raise RuntimeError("Evaluation workers used different selector thresholds")
        for key in (
            "selector_all",
            "base_all",
            "selector_clear",
            "base_clear",
            "cumulative",
            "calibration_histogram",
        ):
            merged[key] += np.asarray(item[key], dtype=np.float64)
        for key in (
            "false_high_detected",
            "false_high_count",
            "false_low_recovered",
            "false_low_count",
            "disagreement_count",
            "valid_count",
            "ordinal_abs_error",
            "ordinal_exact",
            "ordinal_count",
            "monotonic_violation",
            "monotonic_count",
        ):
            merged[key] += float(item[key])
    return merged


def _empty_eval_result():
    return {
        "stats": _empty_stats(),
        "base_stats": _empty_stats(),
        "diagnostics": _empty_diagnostics(),
        "touchanything_protocol_stats": {},
        "base_touchanything_protocol_stats": {},
        "support_selector_stats": {},
    }


def _merge_stats(items):
    merged = _empty_stats()
    for stats in items:
        for key in merged:
            if key == "tactile_dim":
                merged[key] = max(merged[key], int(stats.get(key, 0)))
            else:
                merged[key] += stats.get(key, 0)
    return merged


def _merge_diagnostics(items, max_frames):
    merged = _empty_diagnostics()
    for diag in items:
        for key in (
            "gt_hist",
            "pred_hist",
            "error_hist",
            "pressure_count",
            "pressure_gt_sum",
            "pressure_pred_sum",
            "pressure_err_sum",
            "pressure_abs_sum",
            "pressure_over_sum",
            "pressure_under_sum",
            "pressure_pred_active",
            "pressure_gt_active",
            "pred_tail_count",
            "pred_tail_pred_sum",
            "pred_tail_gt_sum",
        ):
            merged[key] += diag.get(key, 0)
        frame = diag.get("frame", {})
        for key in FRAME_DIAG_KEYS:
            values = frame.get(key, [])
            if isinstance(values, list):
                merged["frame"][key].extend(values)
            elif values is not None:
                merged["frame"][key].append(values)
        provenance = diag.get("frame_provenance", {})
        for key in FRAME_PROVENANCE_KEYS:
            values = provenance.get(key, [])
            if isinstance(values, list):
                merged["frame_provenance"][key].extend(values)
            elif values is not None:
                merged["frame_provenance"][key].append(values)

    for key in FRAME_DIAG_KEYS:
        arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in merged["frame"][key]]
        merged["frame"][key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
    for key in FRAME_PROVENANCE_KEYS:
        arrays = [np.asarray(item, dtype=object).reshape(-1) for item in merged["frame_provenance"][key]]
        merged["frame_provenance"][key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=object)

    _trim_frame_diagnostics(merged, max_frames)
    return merged


def _trim_frame_diagnostics(diag, max_frames, seed=2026):
    max_frames = max(0, int(max_frames))
    frame = diag["frame"]
    for key in FRAME_DIAG_KEYS:
        if isinstance(frame[key], list):
            arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in frame[key]]
            frame[key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
    provenance = diag["frame_provenance"]
    for key in FRAME_PROVENANCE_KEYS:
        if isinstance(provenance[key], list):
            arrays = [np.asarray(item, dtype=object).reshape(-1) for item in provenance[key]]
            provenance[key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=object)
    if max_frames <= 0:
        for key in FRAME_DIAG_KEYS:
            frame[key] = np.zeros(0, dtype=np.float32)
        for key in FRAME_PROVENANCE_KEYS:
            provenance[key] = np.zeros(0, dtype=object)
        return
    total = len(np.asarray(frame["gt_volume"]).reshape(-1))
    if total <= max_frames:
        return
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(total, size=max_frames, replace=False))
    for key in FRAME_DIAG_KEYS:
        frame[key] = np.asarray(frame[key], dtype=np.float32)[keep]
    for key in FRAME_PROVENANCE_KEYS:
        provenance[key] = np.asarray(provenance[key], dtype=object)[keep]


def _merge_eval_results(items, max_frames):
    stats = _merge_stats([item["stats"] for item in items])
    base_stats = _merge_stats([item.get("base_stats", {}) for item in items])
    diagnostics = _merge_diagnostics([item["diagnostics"] for item in items], max_frames)
    touchanything_protocol_stats = merge_compact_touchanything_protocol_stats(
        [item.get("touchanything_protocol_stats", {}) for item in items]
    )
    base_touchanything_protocol_stats = merge_compact_touchanything_protocol_stats(
        [item.get("base_touchanything_protocol_stats", {}) for item in items]
    )
    support_selector_stats = _merge_support_selector_stats(
        [item.get("support_selector_stats", {}) for item in items]
    )
    return {
        "stats": stats,
        "base_stats": base_stats,
        "diagnostics": diagnostics,
        "touchanything_protocol_stats": touchanything_protocol_stats,
        "base_touchanything_protocol_stats": base_touchanything_protocol_stats,
        "support_selector_stats": support_selector_stats,
    }


def _update_stats(stats, pred_tactile, gt_tactile, palm_mask, contact_thr, active_thr=0.05, background_thr=0.02):
    if pred_tactile.shape[0] == 0:
        return

    stats["tactile_dim"] = max(stats["tactile_dim"], int(pred_tactile.shape[1]))
    if palm_mask is not None:
        pred = pred_tactile[:, palm_mask]
        gt = gt_tactile[:, palm_mask]
    else:
        pred = pred_tactile
        gt = gt_tactile

    diff = pred - gt
    pred_volume_per_frame = pred.sum(axis=1)
    gt_volume_per_frame = gt.sum(axis=1)
    catastrophic_over_base = gt_volume_per_frame < CATASTROPHIC_OVER_GT_MAX
    catastrophic_under_base = gt_volume_per_frame >= CATASTROPHIC_UNDER_GT_MIN
    stats["catastrophic_over_count"] += int(
        np.sum(catastrophic_over_base & (pred_volume_per_frame > CATASTROPHIC_OVER_PRED_MIN))
    )
    stats["catastrophic_over_denominator"] += int(np.sum(catastrophic_over_base))
    stats["catastrophic_under_count"] += int(
        np.sum(catastrophic_under_base & (pred_volume_per_frame < CATASTROPHIC_UNDER_PRED_MAX))
    )
    stats["catastrophic_under_denominator"] += int(np.sum(catastrophic_under_base))
    false_high_mask = (gt < 0.005) & (pred >= 0.3)
    stats["false_high_gt005_pred03_count"] += int(false_high_mask.sum())
    stats["false_high_gt005_pred03_denominator"] += int(np.sum(gt < 0.005))
    stats["false_high_gt005_pred03_excess_volume"] += float(
        np.maximum(pred - gt, 0.0)[false_high_mask].sum()
    )
    stats["total_frames"] += int(pred.shape[0])
    stats["total_values"] += int(pred.size)
    stats["abs_sum"] += float(np.abs(diff).sum())
    stats["sq_sum"] += float((diff ** 2).sum())
    stats["pred_volume"] += float(pred.sum())
    stats["gt_volume"] += float(gt.sum())

    active_mask = gt > active_thr
    background_mask = gt <= background_thr
    if np.any(active_mask):
        stats["active_abs_sum"] += float(np.abs(diff[active_mask]).sum())
        stats["active_count"] += int(active_mask.sum())
        stats["active_true_positive"] += int(np.sum((pred > active_thr) & active_mask))
        stats["active_gt_count"] += int(active_mask.sum())
    if np.any(background_mask):
        stats["background_abs_sum"] += float(np.abs(diff[background_mask]).sum())
        stats["background_count"] += int(background_mask.sum())
        stats["background_false_positive"] += int(np.sum((pred > active_thr) & background_mask))

    for p, g in zip(pred, gt):
        if np.std(p) > 1e-6 and np.std(g) > 1e-6:
            pcc = np.corrcoef(p, g)[0, 1]
            if not np.isnan(pcc):
                stats["pcc_sum"] += float(pcc)
                stats["pcc_count"] += 1

    pred_bin = pred > contact_thr
    gt_bin = gt > contact_thr
    pred_frame_contact = np.any(pred_bin, axis=1)
    gt_frame_contact = np.any(gt_bin, axis=1)
    stats["temporal_correct"] += int(np.sum(pred_frame_contact == gt_frame_contact))

    intersection = np.sum(pred_bin & gt_bin, axis=1)
    union = np.sum(pred_bin | gt_bin, axis=1)
    contact_iou_per_frame = np.ones(len(union), dtype=np.float32)
    non_zero_mask = union != 0
    contact_iou_per_frame[non_zero_mask] = intersection[non_zero_mask] / union[non_zero_mask]
    stats["contact_iou_sum"] += float(contact_iou_per_frame.sum())

    vol_iou = volumetric_iou_stats(pred, gt, value_axis=1)
    stats["vol_iou_sum"] += float(vol_iou.per_frame.sum())
    stats["vol_intersection_sum"] += vol_iou.intersection_sum
    stats["vol_union_sum"] += vol_iou.union_sum
    location = location_distribution_stats(
        pred,
        gt,
        value_axis=1,
        support_threshold=active_thr,
        min_gt_volume=1.0,
    )
    eligible = location.eligible
    stats["nonempty_location_frame_count"] += int(eligible.sum())
    if np.any(eligible):
        stats["distribution_viou_sum"] += float(np.nansum(location.distribution_viou))
        stats["pred_mass_on_gt_support_sum"] += float(
            np.nansum(location.pred_mass_on_gt_support)
        )
        stats["gt_mass_in_pred_support_sum"] += float(
            np.nansum(location.gt_mass_in_pred_support)
        )
    core_location = location_distribution_stats(
        pred,
        gt,
        value_axis=1,
        support_threshold=CORE_LOCATION_MIN_GT_PEAK,
        min_gt_volume=1.0,
        distribution_power=CORE_LOCATION_DISTRIBUTION_POWER,
        min_gt_peak=CORE_LOCATION_MIN_GT_PEAK,
    )
    core_eligible = core_location.eligible
    stats["core_location_frame_count"] += int(core_eligible.sum())
    if np.any(core_eligible):
        stats["core_distribution_viou_sum"] += float(
            np.nansum(core_location.distribution_viou)
        )
        stats["core_pred_mass_on_gt_support_sum"] += float(
            np.nansum(core_location.pred_mass_on_gt_support)
        )
        stats["core_gt_mass_in_pred_support_sum"] += float(
            np.nansum(core_location.gt_mass_in_pred_support)
        )


def _frame_metrics(pred, gt, contact_thr, active_thr=0.05):
    pred_bin = pred > contact_thr
    gt_bin = gt > contact_thr
    intersection = float(np.sum(pred_bin & gt_bin))
    union = float(np.sum(pred_bin | gt_bin))
    contact_iou = 1.0 if union == 0 else intersection / union

    vol_intersection = float(np.minimum(pred, gt).sum())
    vol_union = float(np.maximum(pred, gt).sum())
    vol_iou = 1.0 if vol_union <= 1e-12 else vol_intersection / vol_union
    pred_volume = float(pred.sum())
    gt_volume = float(gt.sum())
    volume_ratio = pred_volume / max(gt_volume, 1e-12)
    metrics = {
        "contact_iou": contact_iou,
        "volumetric_iou": vol_iou,
        "pred_volume": pred_volume,
        "gt_volume": gt_volume,
        "volume_ratio": volume_ratio,
        "gt_active_vertices": float(np.sum(gt > active_thr)),
        "pred_active_vertices": float(np.sum(pred > active_thr)),
        "frame_mae": float(np.mean(np.abs(pred - gt))) if pred.size else 0.0,
    }
    location = location_distribution_stats(
        pred[None, :],
        gt[None, :],
        value_axis=1,
        support_threshold=active_thr,
        min_gt_volume=1.0,
    )
    metrics.update({
        "distribution_viou": float(location.distribution_viou[0]),
        "volume_matched_viou": float(location.distribution_viou[0]),
        "pred_mass_on_gt_support": float(location.pred_mass_on_gt_support[0]),
        "gt_mass_in_pred_support": float(location.gt_mass_in_pred_support[0]),
        "location_eligible": float(location.eligible[0]),
    })
    core_location = location_distribution_stats(
        pred[None, :],
        gt[None, :],
        value_axis=1,
        support_threshold=CORE_LOCATION_MIN_GT_PEAK,
        min_gt_volume=1.0,
        distribution_power=CORE_LOCATION_DISTRIBUTION_POWER,
        min_gt_peak=CORE_LOCATION_MIN_GT_PEAK,
    )
    metrics.update({
        "core_distribution_viou": float(core_location.distribution_viou[0]),
        "core_pred_mass_on_gt_support": float(core_location.pred_mass_on_gt_support[0]),
        "core_gt_mass_in_pred_support": float(core_location.gt_mass_in_pred_support[0]),
        "core_location_eligible": float(core_location.eligible[0]),
    })
    for label, gt_max, pred_min in (
        ("gt005_pred03", 0.005, 0.3),
        ("gt005_pred05", 0.005, 0.5),
        ("gt05_pred03", 0.05, 0.3),
    ):
        mask = (gt < gt_max) & (pred >= pred_min)
        metrics[f"false_high_{label}_count"] = float(np.sum(mask))
        metrics[f"false_high_{label}_excess_volume"] = float(np.maximum(pred - gt, 0.0)[mask].sum())
    return metrics


def _update_diagnostics(
    diagnostics,
    pred_tactile,
    gt_tactile,
    palm_mask,
    contact_thr,
    active_thr,
    max_frames,
    frame_records,
    worker_rank,
):
    if pred_tactile.shape[0] == 0:
        return

    pred = pred_tactile[:, palm_mask] if palm_mask is not None else pred_tactile
    gt = gt_tactile[:, palm_mask] if palm_mask is not None else gt_tactile
    flat_pred = np.clip(pred.reshape(-1).astype(np.float32), 0.0, 1.0)
    flat_gt = np.clip(gt.reshape(-1).astype(np.float32), 0.0, 1.0)
    flat_err = flat_pred - flat_gt

    diagnostics["gt_hist"] += np.histogram(flat_gt, bins=DIAG_VALUE_BINS)[0]
    diagnostics["pred_hist"] += np.histogram(flat_pred, bins=DIAG_VALUE_BINS)[0]
    diagnostics["error_hist"] += np.histogram(flat_err, bins=DIAG_ERROR_BINS)[0]

    bin_idx = np.digitize(flat_gt, DIAG_PRESSURE_BINS, right=False) - 1
    valid_bin = (bin_idx >= 0) & (bin_idx < len(DIAG_PRESSURE_BINS) - 1)
    bin_idx = bin_idx[valid_bin]
    if bin_idx.size:
        gt_values = flat_gt[valid_bin]
        pred_values = flat_pred[valid_bin]
        err_values = pred_values - gt_values
        n_bins = len(DIAG_PRESSURE_BINS) - 1
        diagnostics["pressure_count"] += np.bincount(bin_idx, minlength=n_bins)
        diagnostics["pressure_gt_sum"] += np.bincount(bin_idx, weights=gt_values, minlength=n_bins)
        diagnostics["pressure_pred_sum"] += np.bincount(bin_idx, weights=pred_values, minlength=n_bins)
        diagnostics["pressure_err_sum"] += np.bincount(bin_idx, weights=err_values, minlength=n_bins)
        diagnostics["pressure_abs_sum"] += np.bincount(bin_idx, weights=np.abs(err_values), minlength=n_bins)
        diagnostics["pressure_over_sum"] += np.bincount(bin_idx, weights=np.maximum(err_values, 0.0), minlength=n_bins)
        diagnostics["pressure_under_sum"] += np.bincount(bin_idx, weights=np.maximum(-err_values, 0.0), minlength=n_bins)
        diagnostics["pressure_pred_active"] += np.bincount(
            bin_idx, weights=(pred_values > active_thr).astype(np.float32), minlength=n_bins
        )
        diagnostics["pressure_gt_active"] += np.bincount(
            bin_idx, weights=(gt_values > active_thr).astype(np.float32), minlength=n_bins
        )
        for threshold_index, threshold in enumerate(DIAG_PRED_TAIL_THRESHOLDS):
            tail = pred_values >= threshold
            if not np.any(tail):
                continue
            diagnostics["pred_tail_count"][:, threshold_index] += np.bincount(
                bin_idx[tail], minlength=n_bins
            )
            diagnostics["pred_tail_pred_sum"][:, threshold_index] += np.bincount(
                bin_idx[tail], weights=pred_values[tail], minlength=n_bins
            )
            diagnostics["pred_tail_gt_sum"][:, threshold_index] += np.bincount(
                bin_idx[tail], weights=gt_values[tail], minlength=n_bins
            )

    frame_metrics = [
        _frame_metrics(p, g, contact_thr, active_thr=active_thr)
        for p, g in zip(pred, gt)
    ]
    for key in FRAME_DIAG_KEYS:
        diagnostics["frame"][key].append(np.asarray([item[key] for item in frame_metrics], dtype=np.float32))
    if len(frame_records) != len(frame_metrics):
        raise RuntimeError(f"Frame provenance mismatch: {len(frame_records)} records for {len(frame_metrics)} metrics")
    provenance_values = {
        "sample_dir": [str(record.get("sample_dir", "")) for record in frame_records],
        "sample_ref": [
            str(record.get("sample_ref", record.get("sample_dir", "")))
            for record in frame_records
        ],
        "sequence_key": [str(record.get("sequence_key", "")) for record in frame_records],
        "frame_idx": [int(record.get("frame_idx", 0) or 0) for record in frame_records],
        "query_alias": [
            str(record.get("query_alias", record.get("hand", "")))
            for record in frame_records
        ],
        "h5_path": [str(record.get("h5_path", "")) for record in frame_records],
        "frame_row": [int(record.get("frame_row", -1)) for record in frame_records],
        "query_row": [int(record.get("query_row", -1)) for record in frame_records],
        "dataset": [str(record.get("dataset", "")) for record in frame_records],
        "hand": [str(record.get("hand", "")) for record in frame_records],
        "worker_rank": [int(worker_rank)] * len(frame_records),
    }
    for key in FRAME_PROVENANCE_KEYS:
        diagnostics["frame_provenance"][key].append(np.asarray(provenance_values[key], dtype=object))

    # Keep worker-side memory bounded during very large eval runs.
    current_frames = sum(len(np.asarray(item).reshape(-1)) for item in diagnostics["frame"]["gt_volume"])
    if current_frames > max_frames * 2:
        for key in FRAME_DIAG_KEYS:
            diagnostics["frame"][key] = [np.concatenate(diagnostics["frame"][key])]
        for key in FRAME_PROVENANCE_KEYS:
            diagnostics["frame_provenance"][key] = [
                np.concatenate(diagnostics["frame_provenance"][key])
            ]
        _trim_frame_diagnostics(diagnostics, max_frames)
        for key in FRAME_DIAG_KEYS:
            diagnostics["frame"][key] = [diagnostics["frame"][key]]
        for key in FRAME_PROVENANCE_KEYS:
            diagnostics["frame_provenance"][key] = [diagnostics["frame_provenance"][key]]


def _diagnostic_dir(args):
    if args.diagnostics_dir:
        return args.diagnostics_dir
    report_path = _report_path(args)
    base_report = os.path.splitext(os.path.basename(report_path))[0]
    report_dir = os.path.dirname(report_path)
    return os.path.join(report_dir, f"{base_report}_diagnostics")


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_diagnostic_outputs(args, result):
    if not (args.save_diagnostics or args.save_visualizations):
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _diagnostic_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    diag = result["diagnostics"]
    touch_summary = summarize_compact_touchanything_protocol(
        [result.get("touchanything_protocol_stats", {})],
        min_contact_ratio=args.touchanything_min_contact_ratio,
    )
    touch_rows = touch_summary["rows"]
    _write_csv(
        os.path.join(out_dir, "touchanything_protocol_sequence_metrics.csv"),
        [
            "sequence_key",
            "scene_category",
            "frame_count",
            "mae",
            "volumetric_iou",
            "contact_iou",
            "temporal_accuracy",
            "temporal_precision",
            "temporal_recall",
            "temporal_f1",
        ],
        [
            [
                row["sequence_key"],
                row["scene_category"],
                row["frame_count"],
                row["mae"],
                row["volumetric_iou"],
                row["contact_iou"],
                row["temporal_accuracy"],
                row["temporal_precision"],
                row["temporal_recall"],
                row["temporal_f1"],
            ]
            for row in touch_rows
        ],
    )
    _write_csv(
        os.path.join(out_dir, "touchanything_protocol_scene_metrics.csv"),
        [
            "scene_category",
            "sequence_count",
            "frame_count",
            "T.Acc",
            "C.IoU",
            "V.IoU",
            "MAE",
        ],
        [
            [
                scene,
                touch_summary["by_scene"][scene]["sequence_count"],
                touch_summary["by_scene"][scene]["frame_count"],
                touch_summary["by_scene"][scene]["temporal_accuracy"],
                touch_summary["by_scene"][scene]["contact_iou"],
                touch_summary["by_scene"][scene]["volumetric_iou"],
                touch_summary["by_scene"][scene]["mae"],
            ]
            for scene in TOUCHANYTHING_SCENE_CATEGORIES
        ],
    )

    pressure_rows = []
    denom = np.maximum(diag["pressure_count"], 1.0)
    for i in range(len(DIAG_PRESSURE_BINS) - 1):
        pressure_rows.append([
            float(DIAG_PRESSURE_BINS[i]),
            float(DIAG_PRESSURE_BINS[i + 1]),
            int(diag["pressure_count"][i]),
            float(diag["pressure_gt_sum"][i] / denom[i]),
            float(diag["pressure_pred_sum"][i] / denom[i]),
            float(diag["pressure_err_sum"][i] / denom[i]),
            float(diag["pressure_abs_sum"][i] / denom[i]),
            float(diag["pressure_over_sum"][i] / denom[i]),
            float(diag["pressure_under_sum"][i] / denom[i]),
            float(diag["pressure_pred_active"][i] / denom[i]),
            float(diag["pressure_gt_active"][i] / denom[i]),
        ])
    _write_csv(
        os.path.join(out_dir, "pressure_bins.csv"),
        [
            "gt_bin_low",
            "gt_bin_high",
            "count",
            "mean_gt",
            "mean_pred",
            "mean_error_pred_minus_gt",
            "mae",
            "mean_over_prediction",
            "mean_under_prediction",
            "pred_active_rate",
            "gt_active_rate",
        ],
        pressure_rows,
    )

    tail_rows = []
    for bin_index in range(len(DIAG_PRESSURE_BINS) - 1):
        gt_count = int(diag["pressure_count"][bin_index])
        for threshold_index, threshold in enumerate(DIAG_PRED_TAIL_THRESHOLDS):
            tail_count = int(diag["pred_tail_count"][bin_index, threshold_index])
            tail_denom = max(tail_count, 1)
            mean_tail_pred = float(
                diag["pred_tail_pred_sum"][bin_index, threshold_index] / tail_denom
            )
            mean_tail_gt = float(
                diag["pred_tail_gt_sum"][bin_index, threshold_index] / tail_denom
            )
            tail_rows.append([
                float(DIAG_PRESSURE_BINS[bin_index]),
                float(DIAG_PRESSURE_BINS[bin_index + 1]),
                gt_count,
                float(threshold),
                tail_count,
                float(tail_count / max(gt_count, 1)),
                mean_tail_gt,
                mean_tail_pred,
                mean_tail_pred - mean_tail_gt,
            ])
    _write_csv(
        os.path.join(out_dir, "pointwise_pressure_tails.csv"),
        [
            "gt_bin_low",
            "gt_bin_high",
            "gt_count",
            "pred_threshold",
            "pred_at_or_above_count",
            "pred_at_or_above_rate",
            "mean_gt_in_tail",
            "mean_pred_in_tail",
            "mean_overprediction_in_tail",
        ],
        tail_rows,
    )

    false_high_rows = []
    total_pred_volume = max(float(result["stats"].get("pred_volume", 0.0)), 1e-12)
    for gt_max in (0.005, 0.02, 0.05, 0.10, 0.20):
        gt_bins = np.flatnonzero(DIAG_PRESSURE_BINS[1:] <= gt_max + 1e-8)
        gt_count = int(diag["pressure_count"][gt_bins].sum())
        for threshold_index, threshold in enumerate(DIAG_PRED_TAIL_THRESHOLDS):
            if threshold <= gt_max:
                continue
            tail_count = int(diag["pred_tail_count"][gt_bins, threshold_index].sum())
            tail_denom = max(tail_count, 1)
            mean_tail_pred = float(
                diag["pred_tail_pred_sum"][gt_bins, threshold_index].sum() / tail_denom
            )
            mean_tail_gt = float(
                diag["pred_tail_gt_sum"][gt_bins, threshold_index].sum() / tail_denom
            )
            false_high_pred_volume = float(
                diag["pred_tail_pred_sum"][gt_bins, threshold_index].sum()
            )
            false_high_gt_volume = float(
                diag["pred_tail_gt_sum"][gt_bins, threshold_index].sum()
            )
            false_high_rows.append([
                float(gt_max),
                float(threshold),
                gt_count,
                tail_count,
                float(tail_count / max(gt_count, 1)),
                mean_tail_gt,
                mean_tail_pred,
                mean_tail_pred - mean_tail_gt,
                false_high_pred_volume,
                false_high_pred_volume - false_high_gt_volume,
                false_high_pred_volume / total_pred_volume,
            ])
    _write_csv(
        os.path.join(out_dir, "false_high_pressure_summary.csv"),
        [
            "gt_below",
            "pred_at_or_above",
            "gt_count",
            "false_high_count",
            "false_high_rate",
            "mean_gt_in_false_high",
            "mean_pred_in_false_high",
            "mean_overprediction_in_false_high",
            "false_high_pred_volume",
            "false_high_excess_volume",
            "fraction_of_total_pred_volume",
        ],
        false_high_rows,
    )

    frame = {key: np.asarray(diag["frame"][key], dtype=np.float32) for key in FRAME_DIAG_KEYS}
    provenance = {
        key: np.asarray(diag["frame_provenance"][key], dtype=object)
        for key in FRAME_PROVENANCE_KEYS
    }
    frame_rows = zip(
        *(provenance[key] for key in FRAME_PROVENANCE_KEYS),
        *(frame[key] for key in FRAME_DIAG_KEYS),
    )
    _write_csv(
        os.path.join(out_dir, "frame_metrics_sample.csv"),
        list(FRAME_PROVENANCE_KEYS) + list(FRAME_DIAG_KEYS),
        frame_rows,
    )

    value_centers = 0.5 * (DIAG_VALUE_BINS[:-1] + DIAG_VALUE_BINS[1:])
    error_centers = 0.5 * (DIAG_ERROR_BINS[:-1] + DIAG_ERROR_BINS[1:])
    pressure_centers = 0.5 * (DIAG_PRESSURE_BINS[:-1] + DIAG_PRESSURE_BINS[1:])
    count = diag["pressure_count"]
    valid = count > 0

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    ax.plot(value_centers, diag["gt_hist"], label="GT", linewidth=1.8)
    ax.plot(value_centers, diag["pred_hist"], label="Pred", linewidth=1.8)
    ax.axvline(args.background_pressure_thr, color="gray", linestyle="--", linewidth=1.0, label="background thr")
    ax.axvline(args.active_pressure_thr, color="black", linestyle="--", linewidth=1.0, label="active thr")
    ax.set_yscale("log")
    ax.set_xlabel("pressure value")
    ax.set_ylabel("vertex count, log scale")
    ax.set_title("Point Pressure Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pressure_distribution.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    ax.bar(error_centers, diag["error_hist"], width=np.diff(DIAG_ERROR_BINS), align="center")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("pred - gt")
    ax.set_ylabel("vertex count, log scale")
    ax.set_title("Point Error Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_distribution.png"))
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    mean_gt = np.divide(diag["pressure_gt_sum"], denom)
    mean_pred = np.divide(diag["pressure_pred_sum"], denom)
    mean_err = np.divide(diag["pressure_err_sum"], denom)
    ax1.plot(pressure_centers[valid], mean_gt[valid], marker="o", label="mean GT")
    ax1.plot(pressure_centers[valid], mean_pred[valid], marker="o", label="mean Pred")
    ax1.plot(pressure_centers[valid], mean_err[valid], marker="o", label="mean Pred-GT")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xlabel("GT pressure bin center")
    ax1.set_ylabel("pressure / error")
    ax2 = ax1.twinx()
    ax2.bar(pressure_centers[valid], count[valid], width=np.diff(DIAG_PRESSURE_BINS)[valid] * 0.75, alpha=0.18, color="gray", label="count")
    ax2.set_yscale("log")
    ax2.set_ylabel("vertex count, log scale")
    ax1.set_title("Calibration By GT Pressure")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_by_gt_pressure.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    mean_over = np.divide(diag["pressure_over_sum"], denom)
    mean_under = np.divide(diag["pressure_under_sum"], denom)
    ax.plot(pressure_centers[valid], mean_over[valid], marker="o", label="over: max(pred-gt, 0)")
    ax.plot(pressure_centers[valid], mean_under[valid], marker="o", label="under: max(gt-pred, 0)")
    ax.plot(pressure_centers[valid], np.divide(diag["pressure_abs_sum"], denom)[valid], marker="o", label="MAE")
    ax.set_xlabel("GT pressure bin center")
    ax.set_ylabel("mean error magnitude")
    ax.set_title("Over / Under Prediction By GT Pressure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "over_under_by_gt_pressure.png"))
    plt.close(fig)

    if len(frame["gt_volume"]) > 0:
        fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=160)
        scatter = ax.scatter(
            frame["gt_volume"],
            frame["pred_volume"],
            c=frame["volumetric_iou"],
            s=4,
            alpha=0.35,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        upper = float(np.percentile(np.concatenate([frame["gt_volume"], frame["pred_volume"]]), 99.5))
        upper = max(upper, 1e-6)
        ax.plot([0, upper], [0, upper], color="black", linewidth=1.0, linestyle="--")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("GT volume per frame")
        ax.set_ylabel("Pred volume per frame")
        ax.set_title("Frame Volume: Pred vs GT")
        fig.colorbar(scatter, ax=ax, label="Volumetric IoU")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "frame_volume_pred_vs_gt.png"))
        plt.close(fig)

        _write_frame_binned_outputs(out_dir, frame, provenance, args)

    return out_dir


def _write_frame_binned_outputs(out_dir, frame, provenance, args):
    import matplotlib.pyplot as plt

    gt_volume = frame["gt_volume"]
    positive = gt_volume > 1e-8
    if len(gt_volume) == 0:
        return

    rows = []
    centers = []
    viou_means = []
    ratio_medians = []
    contact_means = []
    frame_mae_means = []
    active_ratio_medians = []
    distribution_viou_means = []
    pred_mass_support_means = []
    gt_mass_support_means = []
    core_distribution_viou_means = []
    core_pred_mass_support_means = []
    core_gt_mass_support_means = []
    empty_bins = []

    def add_frame_bin(low, high, mask):
        if not np.any(mask):
            return
        centers.append(float(0.5 * (low + high)))
        empty_bins.append(bool(high <= 1e-8))
        viou_means.append(float(np.mean(frame["volumetric_iou"][mask])))
        ratio_medians.append(float(np.median(frame["volume_ratio"][mask])))
        contact_means.append(float(np.mean(frame["contact_iou"][mask])))
        frame_mae_means.append(float(np.mean(frame["frame_mae"][mask])))
        active_ratio = frame["pred_active_vertices"][mask] / np.maximum(frame["gt_active_vertices"][mask], 1.0)
        active_ratio_medians.append(float(np.median(active_ratio)))
        location_mask = mask & (frame["location_eligible"] > 0.5)
        location_count = int(np.sum(location_mask))
        distribution_viou_means.append(
            float(np.nanmean(frame["distribution_viou"][location_mask]))
            if location_count else float("nan")
        )
        pred_mass_support_means.append(
            float(np.nanmean(frame["pred_mass_on_gt_support"][location_mask]))
            if location_count else float("nan")
        )
        gt_mass_support_means.append(
            float(np.nanmean(frame["gt_mass_in_pred_support"][location_mask]))
            if location_count else float("nan")
        )
        core_location_mask = mask & (frame["core_location_eligible"] > 0.5)
        core_location_count = int(np.sum(core_location_mask))
        core_distribution_viou_means.append(
            float(np.nanmean(frame["core_distribution_viou"][core_location_mask]))
            if core_location_count else float("nan")
        )
        core_pred_mass_support_means.append(
            float(np.nanmean(frame["core_pred_mass_on_gt_support"][core_location_mask]))
            if core_location_count else float("nan")
        )
        core_gt_mass_support_means.append(
            float(np.nanmean(frame["core_gt_mass_in_pred_support"][core_location_mask]))
            if core_location_count else float("nan")
        )
        rows.append([
            float(low),
            float(high),
            int(np.sum(mask)),
            float(np.mean(frame["gt_volume"][mask])),
            float(np.mean(frame["pred_volume"][mask])),
            ratio_medians[-1],
            viou_means[-1],
            contact_means[-1],
            frame_mae_means[-1],
            active_ratio_medians[-1],
            location_count,
            distribution_viou_means[-1],
            pred_mass_support_means[-1],
            gt_mass_support_means[-1],
            core_location_count,
            core_distribution_viou_means[-1],
            core_pred_mass_support_means[-1],
            core_gt_mass_support_means[-1],
        ])

    zero_mask = gt_volume <= 1e-8
    add_frame_bin(0.0, 0.0, zero_mask)

    if np.sum(positive) >= 2:
        quantiles = np.linspace(0.0, 1.0, 11)
        edges = np.unique(np.quantile(gt_volume[positive], quantiles))
        if len(edges) >= 3:
            edges[0] = max(0.0, edges[0] - 1e-8)
            edges[-1] = edges[-1] + 1e-8
            bin_idx = np.digitize(gt_volume, edges, right=False) - 1
            for i in range(len(edges) - 1):
                add_frame_bin(edges[i], edges[i + 1], bin_idx == i)

    if not rows:
        return

    _write_csv(
        os.path.join(out_dir, "frame_bins_by_gt_volume.csv"),
        [
            "gt_volume_bin_low",
            "gt_volume_bin_high",
            "frame_count",
            "mean_gt_volume",
            "mean_pred_volume",
            "median_pred_gt_volume_ratio",
            "mean_volumetric_iou",
            "mean_contact_iou",
            "mean_frame_mae",
            "median_pred_gt_active_vertex_ratio",
            "location_frame_count",
            "mean_distribution_viou",
            "mean_pred_mass_on_gt_support",
            "mean_gt_mass_in_pred_support",
            "core_location_frame_count",
            "mean_core_distribution_viou",
            "mean_core_pred_mass_on_gt_support",
            "mean_core_gt_mass_in_pred_support",
        ],
        rows,
    )

    dataset_values = np.asarray(provenance.get("dataset", [""] * len(gt_volume)), dtype=object)
    domain_rows = []
    domain_masks = [("overall", np.ones(len(gt_volume), dtype=bool))]
    domain_masks.extend(
        (str(dataset), dataset_values == dataset)
        for dataset in sorted(set(dataset_values.tolist()))
        if str(dataset)
    )
    for dataset, domain_mask in domain_masks:
        eligible = domain_mask & (frame["location_eligible"] > 0.5)
        eligible_count = int(np.sum(eligible))
        core_eligible = domain_mask & (frame["core_location_eligible"] > 0.5)
        core_eligible_count = int(np.sum(core_eligible))
        domain_rows.append([
            dataset,
            int(np.sum(domain_mask)),
            eligible_count,
            float(np.nanmean(frame["distribution_viou"][eligible]))
            if eligible_count else float("nan"),
            float(np.nanmean(frame["pred_mass_on_gt_support"][eligible]))
            if eligible_count else float("nan"),
            float(np.nanmean(frame["gt_mass_in_pred_support"][eligible]))
            if eligible_count else float("nan"),
            core_eligible_count,
            float(np.nanmean(frame["core_distribution_viou"][core_eligible]))
            if core_eligible_count else float("nan"),
            float(np.nanmean(frame["core_pred_mass_on_gt_support"][core_eligible]))
            if core_eligible_count else float("nan"),
            float(np.nanmean(frame["core_gt_mass_in_pred_support"][core_eligible]))
            if core_eligible_count else float("nan"),
        ])
    _write_csv(
        os.path.join(out_dir, "location_metrics_by_dataset.csv"),
        [
            "dataset",
            "frame_count",
            "location_frame_count",
            "mean_distribution_viou",
            "mean_pred_mass_on_gt_support",
            "mean_gt_mass_in_pred_support",
            "core_location_frame_count",
            "mean_core_distribution_viou",
            "mean_core_pred_mass_on_gt_support",
            "mean_core_gt_mass_in_pred_support",
        ],
        domain_rows,
    )

    catastrophic_over_base = gt_volume < CATASTROPHIC_OVER_GT_MAX
    catastrophic_under_base = gt_volume >= CATASTROPHIC_UNDER_GT_MIN
    catastrophic_rows = [
        [
            "over",
            f"gt_volume < {CATASTROPHIC_OVER_GT_MAX:g} and pred_volume > {CATASTROPHIC_OVER_PRED_MIN:g}",
            int(np.sum(catastrophic_over_base & (frame["pred_volume"] > CATASTROPHIC_OVER_PRED_MIN))),
            int(np.sum(catastrophic_over_base)),
            float(
                np.sum(catastrophic_over_base & (frame["pred_volume"] > CATASTROPHIC_OVER_PRED_MIN))
                / max(np.sum(catastrophic_over_base), 1)
            ),
        ],
        [
            "under",
            f"gt_volume >= {CATASTROPHIC_UNDER_GT_MIN:g} and pred_volume < {CATASTROPHIC_UNDER_PRED_MAX:g}",
            int(np.sum(catastrophic_under_base & (frame["pred_volume"] < CATASTROPHIC_UNDER_PRED_MAX))),
            int(np.sum(catastrophic_under_base)),
            float(
                np.sum(catastrophic_under_base & (frame["pred_volume"] < CATASTROPHIC_UNDER_PRED_MAX))
                / max(np.sum(catastrophic_under_base), 1)
            ),
        ],
    ]
    _write_csv(
        os.path.join(out_dir, "catastrophic_frame_diagnostics.csv"),
        ["type", "condition", "count", "denominator", "rate"],
        catastrophic_rows,
    )

    over_mask = catastrophic_over_base & (frame["pred_volume"] > CATASTROPHIC_OVER_PRED_MIN)
    under_mask = catastrophic_under_base & (frame["pred_volume"] < CATASTROPHIC_UNDER_PRED_MAX)
    sample_header = ["type"] + list(FRAME_PROVENANCE_KEYS) + list(FRAME_DIAG_KEYS)
    sample_rows = []
    for condition, mask in (("over", over_mask), ("under", under_mask)):
        for index in np.flatnonzero(mask):
            sample_rows.append(
                [condition]
                + [provenance[key][index] for key in FRAME_PROVENANCE_KEYS]
                + [frame[key][index] for key in FRAME_DIAG_KEYS]
            )
    _write_csv(
        os.path.join(out_dir, "catastrophic_frame_samples.csv"),
        sample_header,
        sample_rows,
    )

    false_high_score = (
        frame["false_high_gt005_pred03_excess_volume"]
        + frame["false_high_gt05_pred03_excess_volume"]
    )
    ranked = np.argsort(-false_high_score)
    ranked = ranked[false_high_score[ranked] > 0][:500]
    top_rows = [
        [rank]
        + [provenance[key][index] for key in FRAME_PROVENANCE_KEYS]
        + [frame[key][index] for key in FRAME_DIAG_KEYS]
        for rank, index in enumerate(ranked, start=1)
    ]
    _write_csv(
        os.path.join(out_dir, "top_false_high_frames.csv"),
        ["rank"] + list(FRAME_PROVENANCE_KEYS) + list(FRAME_DIAG_KEYS),
        top_rows,
    )

    if np.any(zero_mask):
        empty_pred_volume = float(np.mean(frame["pred_volume"][zero_mask]))
        empty_pred_active = float(np.mean(frame["pred_active_vertices"][zero_mask]))
        _write_csv(
            os.path.join(out_dir, "empty_frame_diagnostics.csv"),
            ["frame_count", "mean_pred_volume", "mean_pred_active_vertices"],
            [[int(np.sum(zero_mask)), empty_pred_volume, empty_pred_active]],
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), dpi=160)
        axes[0].bar(["empty frames"], [empty_pred_volume], color="tab:red")
        axes[0].set_ylabel("mean predicted volume")
        axes[1].bar(["empty frames"], [empty_pred_active], color="tab:purple")
        axes[1].set_ylabel("mean predicted active vertices")
        fig.suptitle(f"Empty Frame Diagnostics (n={int(np.sum(zero_mask))})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "empty_frame_diagnostics.png"))
        plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    ax1.plot(centers, viou_means, marker="o", label="mean V-IoU")
    ax1.plot(centers, contact_means, marker="o", label="mean Contact IoU")
    ax1.set_xlabel("GT volume bin center")
    ax1.set_ylabel("IoU")
    ax1.set_ylim(0.0, 1.0)
    ax2 = ax1.twinx()
    ratio_plot = np.asarray(ratio_medians, dtype=np.float64)
    active_ratio_plot = np.asarray(active_ratio_medians, dtype=np.float64)
    empty_bins = np.asarray(empty_bins, dtype=bool)
    ratio_plot[empty_bins] = np.nan
    active_ratio_plot[empty_bins] = np.nan
    ax2.plot(centers, ratio_plot, marker="o", color="tab:red", label="median Pred/GT volume")
    ax2.plot(centers, active_ratio_plot, marker="o", color="tab:purple", label="median Pred/GT active vertices")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax2.set_ylabel("ratio")
    ax1.set_title("Frame Metrics By GT Volume")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frame_metrics_by_gt_volume.png"))
    plt.close(fig)


def _managed_dataloader_batches(dataloader, *, description, position, show_progress):
    data_iterator = iter(dataloader)
    progress = (
        tqdm(data_iterator, desc=description, position=position)
        if show_progress
        else data_iterator
    )
    try:
        yield from progress
    finally:
        if show_progress:
            progress.close()
        shutdown_workers = getattr(data_iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()


def _evaluate_sample_records(
    args,
    data_dirs,
    sample_records,
    worker_rank=0,
    show_progress=True,
    selector_artifact_indices=None,
):
    if len(sample_records) == 0:
        return _empty_eval_result()
    if args.selector_artifact_output_dir:
        if selector_artifact_indices is None:
            raise RuntimeError(
                "Selector artifact export requires explicit global artifact indices"
            )
        selector_artifact_indices = np.asarray(
            selector_artifact_indices, dtype=np.int64
        )
        if selector_artifact_indices.shape != (len(sample_records),):
            raise ValueError(
                "Selector artifact index count differs from the worker sample shard: "
                f"indices={len(selector_artifact_indices)}, "
                f"samples={len(sample_records)}"
            )

    device = torch.device(f'cuda:{worker_rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(worker_rank)

    model_cfg = _load_model_cfg(args.input_resolution)
    model = _load_model(args, model_cfg, device)
    dataset = OpenTouchTactileDataset(
        model_cfg,
        split=args.split,
        data_dir=data_dirs,
        train=False,
        sample_records=sample_records,
        tactile_only=True,
        input_resolution=args.input_resolution,
        crop_pipeline=args.crop_pipeline,
        bbox_rescale_factor=args.bbox_rescale_factor,
        bbox_source_policy=args.bbox_source_policy,
        data_backend=args.data_backend,
        query_manifests=args.query_manifests,
        hdf5_handle_cache_size=args.hdf5_handle_cache_size,
        hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir,
        # sample_records already carry the globally assigned order for this worker.
        hdf5_sample_order="manifest",
        lazy_index_records=(args.data_backend != "legacy_dirs"),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=initialize_worker_parent_death_signal,
        drop_last=False,
    )

    stats = _empty_stats()
    base_stats = _empty_stats()
    diagnostics = _empty_diagnostics()
    touchanything_protocol_stats = CompactTouchAnythingProtocolAccumulator()
    base_touchanything_protocol_stats = CompactTouchAnythingProtocolAccumulator()
    support_selector_stats = {}
    palm_mask = None
    sample_cursor = 0
    exported_indices = []
    exported_predictions = []
    exported_sample_uids = []
    exported_prediction_vertex_indices = None
    exported_selector_indices = []
    exported_selector_logits = []
    exported_selector_sample_uids = []
    exported_selector_sequence_keys = []
    exported_selector_query_aliases = []
    exported_selector_frame_indices = []
    exported_base_predictions = []
    exported_targets = []
    selector_artifact_vertex_indices = None
    selector_artifact_thresholds = None
    selector_artifact_mode = None
    selector_reference_hasher = hashlib.sha256()
    iterator = _managed_dataloader_batches(
        dataloader,
        description=f"GPU {worker_rank} Evaluating",
        position=worker_rank,
        show_progress=show_progress,
    )
    for batch in iterator:
        raw_batch_size = len(batch["dataset"]) if isinstance(batch.get("dataset"), (list, tuple)) else int(batch["img"].shape[0])
        batch_start = sample_cursor
        batch_stop = sample_cursor + raw_batch_size
        batch_records = sample_records[batch_start:batch_stop]
        batch_selector_indices = (
            selector_artifact_indices[batch_start:batch_stop]
            if selector_artifact_indices is not None
            else None
        )
        sample_cursor += raw_batch_size

        batch = recursive_to(batch, device)
        valid_tactile_mask = batch['has_tactile'].detach().cpu().numpy() > 0.5
        if not np.any(valid_tactile_mask):
            continue

        if palm_mask is None:
            palm_mask = batch['palm_mask'][0].detach().cpu().numpy() > 0.5

        with torch.no_grad():
            out = model.forward_step(batch, train=False)

        pred_tactile = out['pred_tactile'].detach().cpu().numpy()[valid_tactile_mask]
        base_prediction = out.get("base_pred_tactile")
        if base_prediction is not None:
            base_prediction = (
                base_prediction.detach().cpu().numpy()[valid_tactile_mask]
            )
        gt_tactile = batch['tactile_signal'].detach().cpu().numpy()[valid_tactile_mask]
        selector_probabilities = out.get("support_selector_probabilities")
        selector_logits_numpy = None
        if selector_probabilities is not None:
            selector_logits = (
                out["support_selector_logits"]
                .detach()
                .float()
                .cpu()
                .numpy()[valid_tactile_mask]
            )
            selector_logits_numpy = selector_logits
            selector_probabilities = (
                selector_probabilities.detach().cpu().numpy()[valid_tactile_mask]
            )
            thresholds = tuple(
                float(value)
                for value in out.get(
                    "support_selector_thresholds",
                    getattr(model, "support_selector_thresholds", ()),
                )
            )
            if not support_selector_stats:
                support_selector_stats = _empty_support_selector_stats(thresholds)
            _update_support_selector_stats(
                support_selector_stats,
                selector_logits,
                selector_probabilities,
                pred_tactile,
                gt_tactile,
                palm_mask,
                no_contact_max=float(model.support_selector_no_contact_max),
                contact_min=float(model.support_selector_contact_min),
                selector_mode=str(model.support_selector_mode),
            )
        valid_records = [
            record for record, is_valid in zip(batch_records, valid_tactile_mask) if is_valid
        ]
        if args.selector_artifact_output_dir:
            if selector_logits_numpy is None:
                raise RuntimeError(
                    "--selector_artifact_output_dir requires a support-selector checkpoint"
                )
            if batch_selector_indices is None:
                raise RuntimeError("Selector artifact indices were not propagated")
            if palm_mask is None:
                raise RuntimeError("Selector artifact export did not resolve a palm mask")
            current_vertices = np.flatnonzero(palm_mask).astype(np.int32)
            current_thresholds = np.asarray(thresholds, dtype=np.float64)
            current_mode = str(model.support_selector_mode)
            if selector_artifact_vertex_indices is None:
                selector_artifact_vertex_indices = current_vertices
                selector_artifact_thresholds = current_thresholds
                selector_artifact_mode = current_mode
            elif (
                selector_artifact_mode != current_mode
                or not np.array_equal(
                    selector_artifact_vertex_indices, current_vertices
                )
                or not np.array_equal(
                    selector_artifact_thresholds, current_thresholds
                )
            ):
                raise RuntimeError("Selector artifact schema changed within one worker")
            artifact_indices = batch_selector_indices[valid_tactile_mask]
            artifact_base = (
                base_prediction if base_prediction is not None else pred_tactile
            )[:, palm_mask]
            artifact_target = gt_tactile[:, palm_mask]
            artifact_logits = selector_logits_numpy[:, :, palm_mask]
            artifact_base = np.ascontiguousarray(
                artifact_base.astype(np.float16, copy=False)
            )
            artifact_target = np.ascontiguousarray(
                artifact_target.astype(np.float16, copy=False)
            )
            artifact_logits = np.ascontiguousarray(
                artifact_logits.astype(np.float16, copy=False)
            )
            selector_reference_hasher.update(artifact_base.tobytes())
            selector_reference_hasher.update(artifact_target.tobytes())
            exported_selector_indices.append(artifact_indices)
            exported_selector_logits.append(artifact_logits)
            exported_selector_sample_uids.extend(
                str(record.get("sample_uid", "")) for record in valid_records
            )
            exported_selector_sequence_keys.extend(
                str(record.get("sequence_key", "")) for record in valid_records
            )
            exported_selector_query_aliases.extend(
                str(record.get("query_alias", record.get("hand", "")))
                for record in valid_records
            )
            exported_selector_frame_indices.extend(
                int(record.get("frame_idx", -1)) for record in valid_records
            )
            if args.selector_artifact_include_reference:
                exported_base_predictions.append(artifact_base)
                exported_targets.append(artifact_target)
        if args.prediction_output_dir:
            missing_indices = [
                record.get("sample_uid", "")
                for record in valid_records
                if record.get("_artifact_index") is None
            ]
            if missing_indices:
                raise RuntimeError(
                    "--prediction_output_dir requires _artifact_index in every exact "
                    f"sample record; first missing sample={missing_indices[0]!r}"
                )
            exported_indices.append(
                np.asarray(
                    [int(record["_artifact_index"]) for record in valid_records],
                    dtype=np.int64,
                )
            )
            current_prediction_vertices = (
                np.flatnonzero(palm_mask).astype(np.int32)
                if args.prediction_palm_only
                else np.arange(pred_tactile.shape[1], dtype=np.int32)
            )
            if exported_prediction_vertex_indices is None:
                exported_prediction_vertex_indices = current_prediction_vertices
            elif not np.array_equal(
                exported_prediction_vertex_indices, current_prediction_vertices
            ):
                raise RuntimeError(
                    "Prediction export vertex indices changed within one worker"
                )
            exported_predictions.append(
                pred_tactile[:, current_prediction_vertices].astype(
                    np.float16, copy=False
                )
            )
            exported_sample_uids.extend(
                str(record.get("sample_uid", "")) for record in valid_records
            )
        _update_stats(
            stats,
            pred_tactile,
            gt_tactile,
            palm_mask,
            args.contact_thr,
            active_thr=args.active_pressure_thr,
            background_thr=args.background_pressure_thr,
        )
        if base_prediction is not None:
            _update_stats(
                base_stats,
                base_prediction,
                gt_tactile,
                palm_mask,
                args.contact_thr,
                active_thr=args.active_pressure_thr,
                background_thr=args.background_pressure_thr,
            )
        pred_palm = pred_tactile[:, palm_mask] if palm_mask is not None else pred_tactile
        gt_palm = gt_tactile[:, palm_mask] if palm_mask is not None else gt_tactile
        touch_indices = [
            index
            for index, record in enumerate(valid_records)
            if str(record.get("dataset", "")).casefold() in ("touchanything", "egotouch", "ta")
        ]
        if touch_indices:
            touch_frame_stats = touchanything_protocol_frame_stats(
                pred_palm[touch_indices],
                gt_palm[touch_indices],
                value_axis=1,
                contact_threshold=args.touchanything_contact_thr,
            )
            touchanything_protocol_stats.add(
                [
                    touchanything_protocol_group_key(
                        valid_records[index].get("sequence_key", ""),
                        valid_records[index].get(
                            "query_alias", valid_records[index].get("hand", "")
                        ),
                    )
                    for index in touch_indices
                ],
                [valid_records[index].get("frame_idx", 0) for index in touch_indices],
                touch_frame_stats,
            )
            if base_prediction is not None:
                base_palm = (
                    base_prediction[:, palm_mask]
                    if palm_mask is not None
                    else base_prediction
                )
                base_touch_frame_stats = touchanything_protocol_frame_stats(
                    base_palm[touch_indices],
                    gt_palm[touch_indices],
                    value_axis=1,
                    contact_threshold=args.touchanything_contact_thr,
                )
                base_touchanything_protocol_stats.add(
                    [
                        touchanything_protocol_group_key(
                            valid_records[index].get("sequence_key", ""),
                            valid_records[index].get(
                                "query_alias", valid_records[index].get("hand", "")
                            ),
                        )
                        for index in touch_indices
                    ],
                    [valid_records[index].get("frame_idx", 0) for index in touch_indices],
                    base_touch_frame_stats,
                )
        if args.save_diagnostics or args.save_visualizations:
            _update_diagnostics(
                diagnostics,
                pred_tactile,
                gt_tactile,
                palm_mask,
                args.contact_thr,
                args.active_pressure_thr,
                args.diagnostic_max_frames,
                valid_records,
                worker_rank,
            )
    if args.save_diagnostics or args.save_visualizations:
        for key in FRAME_DIAG_KEYS:
            arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in diagnostics["frame"][key]]
            diagnostics["frame"][key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
        for key in FRAME_PROVENANCE_KEYS:
            arrays = [
                np.asarray(item, dtype=object).reshape(-1)
                for item in diagnostics["frame_provenance"][key]
            ]
            diagnostics["frame_provenance"][key] = (
                np.concatenate(arrays) if arrays else np.zeros(0, dtype=object)
            )
        _trim_frame_diagnostics(diagnostics, args.diagnostic_max_frames, seed=2026 + int(worker_rank))

    if args.prediction_output_dir:
        if exported_prediction_vertex_indices is None:
            raise RuntimeError("Prediction export worker produced no tactile records")
        output_root = Path(args.prediction_output_dir)
        shard_root = output_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        indices = (
            np.concatenate(exported_indices)
            if exported_indices
            else np.zeros(0, dtype=np.int64)
        )
        predictions = (
            np.concatenate(exported_predictions, axis=0)
            if exported_predictions
            else np.zeros(
                (0, len(exported_prediction_vertex_indices)), dtype=np.float16
            )
        )
        if len(indices) != len(predictions) or len(indices) != len(exported_sample_uids):
            raise RuntimeError("Prediction artifact arrays have inconsistent lengths")
        order = np.argsort(indices, kind="stable")
        target = shard_root / f"worker_{int(worker_rank):02d}.npz"
        temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.npz")
        np.savez(
            temporary,
            indices=indices[order],
            predictions=predictions[order],
            sample_uids=np.asarray(exported_sample_uids, dtype=str)[order],
            vertex_indices=exported_prediction_vertex_indices,
        )
        os.replace(temporary, target)
        print(
            f"[prediction-export] worker={worker_rank} records={len(indices)} path={target}",
            flush=True,
        )

    if args.selector_artifact_output_dir:
        if selector_artifact_vertex_indices is None:
            raise RuntimeError("Selector artifact worker produced no tactile records")
        output_root = Path(args.selector_artifact_output_dir)
        shard_root = output_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        indices = np.concatenate(exported_selector_indices)
        logits = np.concatenate(exported_selector_logits, axis=0)
        if len(indices) != len(logits) or len(indices) != len(
            exported_selector_sample_uids
        ):
            raise RuntimeError("Selector artifact arrays have inconsistent lengths")
        order = np.argsort(indices, kind="stable")
        payload = {
            "indices": indices[order],
            "sample_uids": np.asarray(
                exported_selector_sample_uids, dtype=str
            )[order],
            "sequence_keys": np.asarray(
                exported_selector_sequence_keys, dtype=str
            )[order],
            "query_aliases": np.asarray(
                exported_selector_query_aliases, dtype=str
            )[order],
            "frame_indices": np.asarray(
                exported_selector_frame_indices, dtype=np.int64
            )[order],
            "selector_logits": logits[order],
            "vertex_indices": selector_artifact_vertex_indices,
            "selector_thresholds": selector_artifact_thresholds,
            "selector_mode": np.asarray(selector_artifact_mode),
            "reference_sha256": np.asarray(
                selector_reference_hasher.hexdigest()
            ),
        }
        if args.selector_artifact_include_reference:
            payload["base_predictions"] = np.concatenate(
                exported_base_predictions, axis=0
            )[order]
            payload["targets"] = np.concatenate(exported_targets, axis=0)[order]
        target = shard_root / f"worker_{int(worker_rank):02d}.npz"
        temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.npz")
        np.savez(temporary, **payload)
        os.replace(temporary, target)
        print(
            f"[selector-artifact] worker={worker_rank} records={len(indices)} "
            f"path={target}",
            flush=True,
        )

    return {
        "stats": stats,
        "base_stats": base_stats,
        "diagnostics": diagnostics,
        "touchanything_protocol_stats": touchanything_protocol_stats.pack(),
        "base_touchanything_protocol_stats": (
            base_touchanything_protocol_stats.pack()
        ),
        "support_selector_stats": support_selector_stats,
    }


def _eval_worker(
    rank,
    args,
    data_dirs,
    sample_records,
    selector_artifact_indices,
    queue,
):
    initialize_worker_parent_death_signal()
    try:
        result = _evaluate_sample_records(
            args,
            data_dirs,
            sample_records,
            worker_rank=rank,
            show_progress=True,
            selector_artifact_indices=selector_artifact_indices,
        )
        queue.put((rank, result, None))
    except Exception:
        queue.put((rank, None, traceback.format_exc()))


def _loss_config_summary(args):
    if not args.exp_name:
        return "N/A"
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path(base_dir) / checkpoint_root
    loss_path = checkpoint_root / args.exp_name / "loss_config.json"
    if loss_path.is_file():
        try:
            with loss_path.open("r", encoding="utf-8") as f:
                loss_config = json.load(f)
        except Exception as exc:
            return f"unreadable ({loss_path}: {exc})"
        source = str(loss_path)
    else:
        checkpoint = _load_checkpoint(args.checkpoint)
        loss_config = checkpoint.get("loss_config") if isinstance(checkpoint, dict) else None
        if not isinstance(loss_config, dict):
            return f"missing ({loss_path}; no compact loss_config)"
        source = f"compact:{args.checkpoint}"
    compact = json.dumps(loss_config, sort_keys=True, ensure_ascii=False)
    return f"{source} | {compact}"


def _format_report(
    args,
    stats,
    touchanything_protocol_stats,
):
    if stats["total_frames"] == 0 or stats["total_values"] == 0:
        return None

    mae = stats["abs_sum"] / stats["total_values"]
    rmse = np.sqrt(stats["sq_sum"] / stats["total_values"])
    avg_pcc = stats["pcc_sum"] / stats["pcc_count"] if stats["pcc_count"] > 0 else 0.0
    temporal_acc = stats["temporal_correct"] / stats["total_frames"]
    contact_iou = stats["contact_iou_sum"] / stats["total_frames"]
    volumetric_iou_frame_macro = stats["vol_iou_sum"] / stats["total_frames"]
    volumetric_iou_split_micro = (
        stats["vol_intersection_sum"] / stats["vol_union_sum"]
        if stats["vol_union_sum"] > 1e-12
        else 1.0
    )
    touch_summary = summarize_compact_touchanything_protocol(
        [touchanything_protocol_stats],
        min_contact_ratio=args.touchanything_min_contact_ratio,
    )
    volume_ratio = stats["pred_volume"] / max(stats["gt_volume"], 1e-6)
    active_mae = stats["active_abs_sum"] / stats["active_count"] if stats["active_count"] > 0 else 0.0
    background_mae = stats["background_abs_sum"] / stats["background_count"] if stats["background_count"] > 0 else 0.0
    active_recall = stats["active_true_positive"] / stats["active_gt_count"] if stats["active_gt_count"] > 0 else 0.0
    false_positive_rate = (
        stats["background_false_positive"] / stats["background_count"] if stats["background_count"] > 0 else 0.0
    )
    location_count = max(int(stats["nonempty_location_frame_count"]), 1)
    distribution_viou = stats["distribution_viou_sum"] / location_count
    pred_mass_on_gt_support = stats["pred_mass_on_gt_support_sum"] / location_count
    gt_mass_in_pred_support = stats["gt_mass_in_pred_support_sum"] / location_count
    core_location_count = max(int(stats["core_location_frame_count"]), 1)
    core_distribution_viou = stats["core_distribution_viou_sum"] / core_location_count
    core_pred_mass_on_gt_support = (
        stats["core_pred_mass_on_gt_support_sum"] / core_location_count
    )
    core_gt_mass_in_pred_support = (
        stats["core_gt_mass_in_pred_support_sum"] / core_location_count
    )
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_symlink():
        checkpoint_target = os.readlink(checkpoint_path)
    else:
        checkpoint_target = ""

    model_metadata = getattr(args, "model_metadata", {}) or {}
    report_lines = [
        f"🎉 Tactile Fast Evaluation 最终评估结果 🎉",
        "="*55,
        f" 评测数据集    : {args.datasets or args.data_dir or 'resolved-default'}",
        f" 评测划分集    : {args.split}",
        f" 实验名        : {args.exp_name or 'explicit-checkpoint'}",
        f" Checkpoint    : {args.ckpt}",
        f" Resolved Ckpt : {args.checkpoint}",
        f" Symlink Target: {checkpoint_target or 'N/A'}",
        f" Head Type     : {model_metadata.get('tactile_head_type', 'dense_v2_dino_rezero')}",
        f" Input HxW     : {args.input_resolution[0]}x{args.input_resolution[1]}",
        f" Patch Grid    : {model_metadata.get('pool_grid_size', 'legacy-default')}",
        f" Spatial Tokens: {model_metadata.get('pool_valid_tokens', 'legacy-default')}",
        f" BBox Rescale  : {args.bbox_rescale_factor:g}",
        f" BBox Source   : {args.bbox_source_policy}",
        f" Pool Layout   : {model_metadata.get('pool_layout', 'fullgrid32')}",
        f" Loss Config   : {_loss_config_summary(args)}",
        f" 总有效评估帧数: {stats['total_frames']}",
        f" 触觉输出维度  : {stats['tactile_dim']} (subdiv MANO vertices)",
        f" 整体 MAE      : {mae:.4f} (归一化区间 [0,1])",
        f" 整体 RMSE     : {rmse:.4f} (归一化区间 [0,1])",
        f" 整体 PCC      : {avg_pcc:.4f} (皮尔逊相关系数)",
        f" Temporal Acc  : {temporal_acc:.4f} (Contact Thr = {args.contact_thr})",
        f" Contact IoU Frame-Macro: {contact_iou:.4f} (Contact Thr = {args.contact_thr})",
        f" V-IoU Frame-Macro : {volumetric_iou_frame_macro:.4f} (逐帧计算后等权平均)",
        (
            f" V-IoU TA-Trajectory: {touch_summary['volumetric_iou']:.4f} "
            f"(source-trajectory micro, trajectory macro; n={touch_summary['sequence_count']})"
        ),
        f" V-IoU Split-Micro  : {volumetric_iou_split_micro:.4f} (整个 split pressure mass 聚合)",
        f" Distribution V-IoU : {distribution_viou:.4f} (GT volume >= 1, scale invariant)",
        f" Volume-Matched V-IoU: {distribution_viou:.4f} (same normalized distribution metric)",
        f" Pred Mass on GT Support: {pred_mass_on_gt_support:.4f} (GT >= {args.active_pressure_thr:g})",
        f" GT Mass in Pred Support: {gt_mass_in_pred_support:.4f} (Pred >= {args.active_pressure_thr:g})",
        f" Location Frames: {stats['nonempty_location_frame_count']} (GT volume >= 1)",
        (
            f" Core Distribution V-IoU: {core_distribution_viou:.4f} "
            f"(power={CORE_LOCATION_DISTRIBUTION_POWER:g}, GT max >= "
            f"{CORE_LOCATION_MIN_GT_PEAK:g})"
        ),
        (
            f" Core Pred Mass on GT Support: {core_pred_mass_on_gt_support:.4f} "
            f"(GT >= {CORE_LOCATION_MIN_GT_PEAK:g})"
        ),
        (
            f" Core GT Mass in Pred Support: {core_gt_mass_in_pred_support:.4f} "
            f"(Pred >= {CORE_LOCATION_MIN_GT_PEAK:g})"
        ),
        f" Core Location Frames: {stats['core_location_frame_count']}",
        (
            f" TA Temporal Acc    : {touch_summary['temporal_accuracy']:.4f} "
            f"(Thr={args.touchanything_contact_thr:g}, min ratio={args.touchanything_min_contact_ratio:g})"
        ),
        f" TA Contact IoU     : {touch_summary['contact_iou']:.4f} (source-trajectory macro)",
        f" TA MAE             : {touch_summary['mae']:.4f} (source-trajectory macro)",
        f" TA Temporal F1     : {touch_summary['temporal_f1']:.4f} (source-trajectory macro)",
        " TA Protocol Note   : canonical palm query; no sensor-grid bend-mask equivalent",
        f" Pred/GT Volume: {volume_ratio:.4f}",
        f" Active MAE    : {active_mae:.4f} (GT > {args.active_pressure_thr})",
        f" Background MAE: {background_mae:.4f} (GT <= {args.background_pressure_thr})",
        f" Active Recall : {active_recall:.4f} (Thr = {args.active_pressure_thr})",
        f" BG False Pos  : {false_positive_rate:.4f} (Pred > {args.active_pressure_thr})",
        (
            " Catastrophic Over: "
            f"{stats['catastrophic_over_count']}/{stats['catastrophic_over_denominator']} "
            f"({stats['catastrophic_over_count'] / max(stats['catastrophic_over_denominator'], 1):.6f}; "
            f"GT volume < {CATASTROPHIC_OVER_GT_MAX:g}, "
            f"Pred volume > {CATASTROPHIC_OVER_PRED_MIN:g})"
        ),
        (
            " Catastrophic Under: "
            f"{stats['catastrophic_under_count']}/{stats['catastrophic_under_denominator']} "
            f"({stats['catastrophic_under_count'] / max(stats['catastrophic_under_denominator'], 1):.6f}; "
            f"GT volume >= {CATASTROPHIC_UNDER_GT_MIN:g}, "
            f"Pred volume < {CATASTROPHIC_UNDER_PRED_MAX:g})"
        ),
        "="*55
    ]
    if touch_summary["sequence_count"] > 0:
        scene_lines = [" TouchAnything Scene Metrics (sequence-macro):"]
        for scene in TOUCHANYTHING_SCENE_CATEGORIES:
            values = touch_summary["by_scene"][scene]
            scene_lines.append(
                f"  {scene:<9} T.Acc={values['temporal_accuracy']:.4f} "
                f"C.IoU={values['contact_iou']:.4f} "
                f"V.IoU={values['volumetric_iou']:.4f} MAE={values['mae']:.4f} "
                f"(n={values['sequence_count']})"
            )
        report_lines[-1:-1] = scene_lines
    return "\n".join(report_lines)


def _write_local_base_vs_fused_summary(args, report_path, result):
    base_stats = result.get("base_stats", {})
    if int(base_stats.get("total_frames", 0)) <= 0:
        return None, None
    fused_summary = _stats_summary(
        result["stats"],
        result.get("touchanything_protocol_stats", {}),
        touchanything_min_contact_ratio=args.touchanything_min_contact_ratio,
    )
    base_summary = _stats_summary(
        base_stats,
        result.get("base_touchanything_protocol_stats", {}),
        touchanything_min_contact_ratio=args.touchanything_min_contact_ratio,
    )
    if fused_summary is None or base_summary is None:
        return None, None

    def selected(summary):
        touch = summary.get("touchanything_protocol", {})
        return {
            "mae": summary["mae"],
            "rmse": summary["rmse"],
            "frame_pcc": summary["pcc"],
            "contact_iou": summary["contact_iou"],
            "volumetric_iou": summary["volumetric_iou_frame_macro"],
            "core_distribution_viou": summary["core_distribution_viou"],
            "pred_gt_volume_ratio": summary["pred_gt_volume_ratio"],
            "false_high_excess_volume_fraction": summary[
                "false_high_gt005_pred03_excess_volume_fraction"
            ],
            "catastrophic_over_rate": summary["catastrophic_over_rate"],
            "ta_contact_iou": touch.get("contact_iou", float("nan")),
            "ta_volumetric_iou": touch.get("volumetric_iou", float("nan")),
            "ta_temporal_accuracy": touch.get("temporal_accuracy", float("nan")),
        }

    base_values = selected(base_summary)
    fused_values = selected(fused_summary)
    output_path = os.path.join(
        os.path.dirname(report_path), "local_base_vs_fused_summary.csv"
    )
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "base", "fused", "fused_minus_base"))
        for metric in base_values:
            base_value = float(base_values[metric])
            fused_value = float(fused_values[metric])
            writer.writerow(
                (metric, base_value, fused_value, fused_value - base_value)
            )
    return output_path, {
        "base": base_values,
        "fused": fused_values,
        "fused_minus_base": {
            metric: float(fused_values[metric]) - float(base_values[metric])
            for metric in base_values
        },
    }


def _write_support_selector_summary(args, report_path, result):
    stats = result.get("support_selector_stats", {})
    if not stats:
        if args.selector_calibration_output:
            raise RuntimeError(
                "--selector_calibration_output requires a support-selector checkpoint"
            )
        return None, None

    rows = []
    summary = {}
    selector_mode = str(
        args.model_metadata.get("support_selector_mode", "contact")
    )
    summary["selector_mode"] = selector_mode
    summary["selector_base_conditioning"] = str(
        args.model_metadata.get("support_selector_base_conditioning", "real")
    )

    def add_binary(scope, counts, threshold=""):
        tp, fp, fn, tn = (float(value) for value in counts)
        values = {
            "iou": tp / max(tp + fp + fn, 1.0),
            "precision": tp / max(tp + fp, 1.0),
            "recall": tp / max(tp + fn, 1.0),
            "f1": 2.0 * tp / max(2.0 * tp + fp + fn, 1.0),
            "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1.0),
        }
        summary[scope] = values
        denominators = {
            "iou": tp + fp + fn,
            "precision": tp + fp,
            "recall": tp + fn,
            "f1": 2.0 * tp + fp + fn,
            "accuracy": tp + fp + fn + tn,
        }
        numerators = {
            "iou": tp,
            "precision": tp,
            "recall": tp,
            "f1": 2.0 * tp,
            "accuracy": tp + tn,
        }
        for metric, value in values.items():
            rows.append(
                [
                    scope,
                    threshold,
                    metric,
                    value,
                    numerators[metric],
                    denominators[metric],
                ]
            )

    add_binary("selector_all", stats["selector_all"])
    add_binary("base_pressure_all", stats["base_all"])
    add_binary("selector_clear", stats["selector_clear"])
    add_binary("base_pressure_clear", stats["base_clear"])
    for index, threshold in enumerate(stats["thresholds"]):
        add_binary(
            f"selector_cumulative_{float(threshold):g}",
            stats["cumulative"][index],
            threshold=float(threshold),
        )

    scalar_metrics = {
        "contact_iou_gain_vs_base": (
            summary["selector_all"]["iou"]
            - summary["base_pressure_all"]["iou"]
        ),
        "clear_iou_gain_vs_base": (
            summary["selector_clear"]["iou"]
            - summary["base_pressure_clear"]["iou"]
        ),
        "false_high_detect_rate": float(stats["false_high_detected"])
        / max(float(stats["false_high_count"]), 1.0),
        "false_low_recovery_rate": float(stats["false_low_recovered"])
        / max(float(stats["false_low_count"]), 1.0),
        "disagreement_rate": float(stats["disagreement_count"])
        / max(float(stats["valid_count"]), 1.0),
        "ordinal_bin_mae": float(stats["ordinal_abs_error"])
        / max(float(stats["ordinal_count"]), 1.0),
        "ordinal_bin_accuracy": float(stats["ordinal_exact"])
        / max(float(stats["ordinal_count"]), 1.0),
        "monotonic_violation_rate": float(stats["monotonic_violation"])
        / max(float(stats["monotonic_count"]), 1.0),
    }
    scalar_counts = {
        "false_high_detect_rate": (
            stats["false_high_detected"], stats["false_high_count"]
        ),
        "false_low_recovery_rate": (
            stats["false_low_recovered"], stats["false_low_count"]
        ),
        "disagreement_rate": (
            stats["disagreement_count"], stats["valid_count"]
        ),
        "ordinal_bin_mae": (
            stats["ordinal_abs_error"], stats["ordinal_count"]
        ),
        "ordinal_bin_accuracy": (
            stats["ordinal_exact"], stats["ordinal_count"]
        ),
        "monotonic_violation_rate": (
            stats["monotonic_violation"], stats["monotonic_count"]
        ),
    }
    for metric, value in scalar_metrics.items():
        numerator, denominator = scalar_counts.get(metric, ("", ""))
        rows.append(
            ["comparison", "", metric, value, numerator, denominator]
        )
    summary["comparison"] = scalar_metrics

    thresholds = tuple(float(value) for value in stats["thresholds"])
    contact_min = float(
        args.model_metadata.get("support_selector_contact_min", 0.10)
    )
    contact_index = _selector_contact_index(thresholds, contact_min)
    histogram = np.asarray(stats["calibration_histogram"], dtype=np.float64)
    minimum_precision = float(
        args.model_metadata.get(
            "support_selector_correction_min_precision",
            SELECTOR_CORRECTION_MIN_PRECISION,
        )
    )
    oracle = summarize_selector_histograms(
        histogram.tolist(),
        thresholds,
        contact_index=contact_index,
        minimum_correction_precision=minimum_precision,
    )
    split_is_validation = str(args.split).strip().lower() in {"val", "validation"}
    oracle_scope = "validation_fit" if split_is_validation else "test_oracle"
    for metric, value in oracle["contact_clear"]["metrics"].items():
        rows.append(
            [
                f"{oracle_scope}_selector_clear",
                oracle["contact_clear"]["threshold_probability"],
                metric,
                value,
                "",
                "",
            ]
        )
    for metric in ("average_precision", "roc_auc"):
        value = float(oracle["contact_clear"][metric])
        if np.isfinite(value):
            rows.append(
                [
                    f"{oracle_scope}_selector_clear",
                    oracle["contact_clear"]["threshold_probability"],
                    metric,
                    value,
                    "",
                    "",
                ]
            )
    for item in oracle["cumulative"]:
        scope = f"{oracle_scope}_cumulative_{float(item['target_threshold']):g}"
        for metric, value in item["metrics"].items():
            rows.append(
                [scope, item["threshold_probability"], metric, value, "", ""]
            )
        for metric in ("average_precision", "roc_auc"):
            value = float(item[metric])
            if np.isfinite(value):
                rows.append(
                    [scope, item["threshold_probability"], metric, value, "", ""]
                )
    for direction, values in oracle["correction"].items():
        for metric, value in values.items():
            if isinstance(value, (int, float)):
                rows.append(
                    [
                        f"{oracle_scope}_{direction}_correction",
                        values["threshold_probability"],
                        metric,
                        value,
                        "",
                        "",
                    ]
                )

    calibrated = None
    calibrated_correction = None
    calibration = getattr(args, "selector_calibration", None)
    if calibration:
        calibrated = calibrated_counts(
            histogram.tolist(),
            thresholds,
            calibration["probability_thresholds"],
            contact_index=contact_index,
            threshold_indices=[
                int(item["threshold_index"])
                for item in calibration["cumulative"]
            ],
        )
        calibrated_correction = calibrated_correction_counts(
            histogram.tolist(),
            thresholds,
            calibration["correction"],
        )
        for metric, value in calibrated["contact_clear"]["metrics"].items():
            rows.append(
                [
                    "validation_calibrated_selector_clear",
                    calibrated["contact_clear"]["threshold_probability"],
                    metric,
                    value,
                    "",
                    "",
                ]
            )
        calibrated_clear_iou = float(
            calibrated["contact_clear"]["metrics"]["iou"]
        )
        calibrated_gain = calibrated_clear_iou - float(
            summary["base_pressure_clear"]["iou"]
        )
        rows.append(
            [
                "validation_calibrated_comparison",
                calibrated["contact_clear"]["threshold_probability"],
                "clear_iou_gain_vs_base",
                calibrated_gain,
                "",
                "",
            ]
        )
        for item in calibrated["cumulative"]:
            scope = (
                "validation_calibrated_cumulative_"
                f"{float(item['target_threshold']):g}"
            )
            for metric, value in item["metrics"].items():
                rows.append(
                    [scope, item["threshold_probability"], metric, value, "", ""]
                )
        for direction, values in calibrated_correction.items():
            for metric, value in values.items():
                rows.append(
                    [
                        f"validation_calibrated_{direction}_correction",
                        values["threshold_probability"],
                        metric,
                        value,
                        "",
                        "",
                    ]
                )
        if selector_mode == "down_error":
            down = calibrated_correction["down"]
            for metric in (
                "precision",
                "false_high_coverage",
                "selected_count",
            ):
                rows.append(
                    [
                        "validation_calibrated_down_error",
                        down["threshold_probability"],
                        metric,
                        down[metric],
                        "",
                        "",
                    ]
                )

    curve = selector_threshold_curve(
        histogram.tolist(),
        thresholds,
        contact_index=contact_index,
    )
    curve_path = os.path.join(
        os.path.dirname(report_path), "support_selector_threshold_curve.csv"
    )
    curve_fields = list(curve[0])
    _write_csv(
        curve_path,
        curve_fields,
        [[item[field] for field in curve_fields] for item in curve],
    )

    calibration_output = None
    if args.selector_calibration_output:
        if not split_is_validation:
            raise ValueError(
                "Selector calibration may only be fitted on --split val or validation"
            )
        validation_calibration = dict(oracle)
        validation_calibration.update(
            {
                "source": "validation_eval",
                "selector_mode": str(
                    args.model_metadata.get("support_selector_mode", "contact")
                ),
                "no_contact_max": float(
                    args.model_metadata.get(
                        "support_selector_no_contact_max", 0.02
                    )
                ),
                "contact_min": contact_min,
                "dataset": str(args.datasets or args.data_dir or ""),
                "split": str(args.split),
            }
        )
        artifact = _finite_json_value({
            "format": "support_selector_calibration_v1",
            "checkpoint_sha256": args.selector_checkpoint_sha256,
            "checkpoint": str(args.checkpoint),
            "calibration": validation_calibration,
        })
        calibration_path = Path(args.selector_calibration_output)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = calibration_path.with_name(
            f".{calibration_path.name}.tmp-{os.getpid()}"
        )
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, calibration_path)
        calibration_output = str(calibration_path)

    output_path = os.path.join(
        os.path.dirname(report_path), "support_selector_summary.csv"
    )
    _write_csv(
        output_path,
        ["scope", "threshold", "metric", "value", "numerator", "denominator"],
        rows,
    )
    summary.update(
        {
            "validation_calibrated": calibrated,
            "validation_calibrated_correction": calibrated_correction,
            "calibration_source": getattr(
                args, "selector_calibration_source", ""
            ),
            "calibration_output": calibration_output,
            "threshold_curve": curve_path,
            f"{oracle_scope}": oracle,
        }
    )
    return output_path, summary


def main():
    parser = argparse.ArgumentParser(description='Fast evaluation for the DINO tactile regressor')
    parser.add_argument('--checkpoint', type=str, default=None, help='Explicit trained tactile checkpoint path. Overrides --exp_name/--ckpt.')
    parser.add_argument(
        '--dino_weights',
        type=str,
        default=None,
        help='Optional local DINOv3 weights override; compact checkpoint metadata is used by default.',
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='mixed_dense_v2_dinov3_rezero_fullgrid32',
        help='Experiment name under --checkpoint_root.',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='loss-best',
        choices=[
            'loss-best',
            'contact-best',
            'selector-best',
            'last',
        ],
        help='Checkpoint selector used with --exp_name.',
    )
    parser.add_argument(
        '--checkpoint_root',
        type=str,
        default=os.path.join(base_dir, "hamer_tactile_ft", "checkpoints"),
        help='Root directory containing experiment checkpoint folders.',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Explicit extracted dataset root(s), comma-separated. Appended after --datasets if both are provided.',
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default=None,
        help=(
            'Dataset names/aliases, comma-separated: opentouch/ot, '
            'touchanything/egotouch/ta, egotactile/ego, acedata/ace.'
        ),
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Extracted split directory to evaluate, e.g. train/val/test/test_seen/test_unseen.',
    )
    parser.add_argument('--gpu', '--gpus', dest='gpu', type=str, default='4')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument(
        '--input_resolution',
        type=str,
        default=None,
        help='Optional HEIGHTxWIDTH assertion; must match checkpoint metadata.',
    )
    parser.add_argument(
        '--crop_pipeline',
        choices=('direct_rectangle', 'legacy_square_center'),
        default=None,
        help='Optional crop-pipeline assertion; checkpoint metadata is authoritative.',
    )
    parser.add_argument(
        '--hdf5_sample_order',
        choices=('manifest', 'legacy_sample_dir_hand'),
        default=None,
        help='Optional HDF5-order assertion; checkpoint metadata is authoritative.',
    )
    parser.add_argument(
        '--bbox_rescale_factor',
        type=float,
        default=None,
        help='Optional eval crop override; compact checkpoint metadata is used by default.',
    )
    parser.add_argument(
        '--bbox_source_policy',
        choices=('any', 'sam3_only'),
        default=None,
        help='Optional bbox provenance override; compact checkpoint metadata is used by default.',
    )
    parser.add_argument(
        '--bbox_manifests',
        type=str,
        default=None,
        help='Optional comma-separated reviewed SAM3 bbox manifests; checkpoint metadata is used by default.',
    )
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument(
        '--data_backend',
        choices=('auto', 'legacy_dirs', 'sequence_hdf5'),
        default='auto',
        help='Storage backend; auto prefers sequence-HDF5 manifests under each processed root.',
    )
    parser.add_argument(
        '--query_manifests',
        type=str,
        default='',
        help='Optional comma-separated sequence-HDF5 query manifests.',
    )
    parser.add_argument(
        '--hdf5_handle_cache_size',
        type=int,
        default=4,
        help='Maximum read-only sequence HDF5 handles retained by each DataLoader worker.',
    )
    parser.add_argument(
        '--hdf5_manifest_cache_dir',
        type=str,
        default=os.path.join(base_dir, 'hamer_tactile_ft', 'hdf5_manifest_cache'),
        help='Shared mmap cache for normalized HDF5 query-manifest rows.',
    )
    parser.add_argument('--contact_thr', type=float, default=0.05, help='Threshold for defining contact (0-1)')
    parser.add_argument('--active_pressure_thr', type=float, default=0.05)
    parser.add_argument('--background_pressure_thr', type=float, default=0.02)
    parser.add_argument(
        '--touchanything_contact_thr',
        type=float,
        default=TOUCHANYTHING_CONTACT_THRESHOLD,
        help='TouchAnything inference protocol contact threshold (official caller uses 0.1).',
    )
    parser.add_argument(
        '--touchanything_min_contact_ratio',
        type=float,
        default=TOUCHANYTHING_MIN_CONTACT_RATIO,
        help='Minimum active-point ratio for TouchAnything temporal contact (official caller uses 0.05).',
    )
    parser.add_argument('--index_workers', type=int, default=128)
    parser.add_argument('--index_backend', type=str, default='process', choices=['process', 'thread'])
    parser.add_argument('--index_chunksize', type=int, default=512)
    parser.add_argument(
        '--index_process_worker_cap',
        type=int,
        default=64,
        help='Maximum process workers for shared-filesystem index scans; 0 disables the cap.',
    )
    parser.add_argument(
        '--index_manifest',
        type=str,
        default=os.path.join(
            base_dir,
            'hamer_tactile_ft',
            'data_integrity_audits',
            'mixed_v2_input',
            'data_integrity_samples.csv',
        ),
        help='Verified audit CSV used to build compact indexes without rescanning sample dirs.',
    )
    parser.add_argument(
        '--sample_records_jsonl',
        type=str,
        default='',
        help=(
            'Optional exact normalized sample-record subset. This bypasses full index/query '
            'manifest enumeration and is intended for deterministic train-fit probes.'
        ),
    )
    parser.add_argument(
        '--prediction_output_dir',
        type=str,
        default='',
        help=(
            'Optional atomic NPZ shard output for predictions. The post-policy '
            'dataset universe is frozen beside the shards as an exact manifest.'
        ),
    )
    parser.add_argument(
        '--prediction_palm_only',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Store only valid palm vertices in prediction shards and bind the '
            'selected vertex indices in export provenance.'
        ),
    )
    parser.add_argument(
        '--selector_artifact_output_dir',
        type=str,
        default='',
        help=(
            'Optional atomic shard output for per-vertex selector logits used by '
            'the offline sufficiency audit.'
        ),
    )
    parser.add_argument(
        '--selector_artifact_include_reference',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'Store frozen-base predictions and GT beside selector logits. Enable '
            'this for exactly one reference artifact to avoid duplicated storage.'
        ),
    )
    parser.add_argument('--index_cache_dir', type=str, default=os.path.join(base_dir, "hamer_tactile_ft", "index_cache"))
    parser.add_argument(
        '--rebuild_index',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Force index-cache reconstruction; disabled by default.',
    )
    parser.add_argument('--index_cache_timeout', type=int, default=3600)
    parser.add_argument(
        '--report_dir',
        type=str,
        default=None,
        help='Directory for evaluation reports. Defaults to eval_reports_{exp_name}_{ckpt} when --exp_name is set.',
    )
    parser.add_argument(
        '--eval_output_root',
        type=str,
        default=None,
        help=(
            'Root shared by all tasks/checkpoints in one eval matrix. Training '
            'val_metrics.txt is archived here atomically.'
        ),
    )
    parser.add_argument(
        '--report_name',
        type=str,
        default=None,
        help='Optional report filename. Defaults to eval_{datasets}_{split}.txt.',
    )
    parser.add_argument(
        '--selector_calibration_input',
        type=str,
        default='',
        help=(
            'Validation calibration JSON to apply. It must be bound to the '
            'selected checkpoint SHA256.'
        ),
    )
    parser.add_argument(
        '--selector_calibration_output',
        type=str,
        default='',
        help=(
            'Write selector calibration fitted on this split. Only val or '
            'validation splits are accepted.'
        ),
    )
    parser.add_argument('--save_diagnostics', action='store_true', help='Save dataset-level distribution diagnostics and plots.')
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Deprecated alias for --save_diagnostics. Saves dataset-level diagnostic plots, not per-frame MANO projections.',
    )
    parser.add_argument('--diagnostic_max_frames', type=int, default=200000, help='Maximum frame-level samples kept for diagnostic scatter plots/CSV.')
    parser.add_argument('--diagnostics_dir', type=str, default=None, help='Directory for diagnostic CSV/PNG outputs. Defaults next to the eval report.')
    args = parser.parse_args()
    if not 0.0 <= args.touchanything_contact_thr <= 1.0:
        parser.error("--touchanything_contact_thr must lie in [0, 1]")
    if not 0.0 <= args.touchanything_min_contact_ratio <= 1.0:
        parser.error("--touchanything_min_contact_ratio must lie in [0, 1]")

    # Resolve every user-facing filesystem input before worker processes spawn.
    for name in (
        "dino_weights",
        "report_dir",
        "eval_output_root",
        "diagnostics_dir",
        "index_cache_dir",
        "hdf5_manifest_cache_dir",
        "index_manifest",
        "prediction_output_dir",
        "selector_artifact_output_dir",
        "selector_calibration_input",
        "selector_calibration_output",
    ):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, _resolve_invocation_path(value))
    if args.query_manifests:
        args.query_manifests = ",".join(
            _resolve_invocation_path(path.strip())
            for path in str(args.query_manifests).split(",")
            if path.strip()
        )
    if int(args.hdf5_handle_cache_size) < 1:
        parser.error("--hdf5_handle_cache_size must be at least 1")
    args.checkpoint = _resolve_checkpoint_path(args)
    _resolve_experiment_model_metadata(args)
    _resolve_selector_calibration(args)
    args.training_val_metrics_archive = _archive_training_val_metrics(args)
    print(f"Resolved checkpoint: {args.checkpoint}")

    data_dirs = resolve_data_dirs(args)
    print("Resolved evaluation data roots:")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")

    gpu_ids = _gpu_ids(args.gpu)
    world_size = len(gpu_ids) if torch.cuda.is_available() and len(gpu_ids) > 0 else 1

    print(f"📦 加载 {args.split} 划分集...")
    model_cfg = _load_model_cfg(args.input_resolution)
    sample_records = None
    if args.sample_records_jsonl:
        sample_manifest = Path(args.sample_records_jsonl).expanduser().resolve(strict=True)
        sample_records = []
        with sample_manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{sample_manifest}:{line_number}: sample record must be a JSON object"
                    )
                record_split = str(record.get("split", args.split))
                if record_split != args.split:
                    raise ValueError(
                        f"{sample_manifest}:{line_number}: split={record_split!r} does not "
                        f"match --split={args.split!r}"
                    )
                sample_records.append(record)
        if not sample_records:
            raise RuntimeError(f"Exact sample manifest is empty: {sample_manifest}")
        print(
            f"Using exact sample subset: {sample_manifest} ({len(sample_records)} records)",
            flush=True,
        )

    dataset = OpenTouchTactileDataset(
        model_cfg,
        split=args.split,
        data_dir=data_dirs,
        train=False,
        index_workers=args.index_workers,
        index_chunksize=args.index_chunksize,
        index_backend=args.index_backend,
        sample_records=sample_records,
        index_process_worker_cap=args.index_process_worker_cap,
        index_manifest=args.index_manifest,
        index_cache_dir=args.index_cache_dir,
        rebuild_index=args.rebuild_index,
        index_cache_timeout=args.index_cache_timeout,
        input_resolution=args.input_resolution,
        crop_pipeline=args.crop_pipeline,
        bbox_rescale_factor=args.bbox_rescale_factor,
        bbox_source_policy=args.bbox_source_policy,
        bbox_manifests=args.bbox_manifests,
        expected_datasets=canonical_dataset_filter(args.datasets),
        data_backend=args.data_backend,
        query_manifests=args.query_manifests,
        hdf5_handle_cache_size=args.hdf5_handle_cache_size,
        hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir,
        hdf5_sample_order=args.hdf5_sample_order,
        lazy_index_records=(args.data_backend != "legacy_dirs"),
    )

    if len(dataset) == 0:
        print("❌ 评估集为空。请检查 --datasets/--data_dir 和 --split。")
        return

    if args.prediction_output_dir:
        _prepare_prediction_export(args)
        dataset.samples = _materialize_prediction_sample_records(
            args, dataset.samples
        )
        args.active_bbox_manifest_sha256 = dict(dataset.bbox_manifest_sha256)
    if args.selector_artifact_output_dir:
        if str(args.model_metadata.get("tactile_head_type", "")) != (
            "dense_v2_dino_support_selector"
        ):
            parser.error(
                "--selector_artifact_output_dir requires a support-selector checkpoint"
            )
        for artifact_index, record in enumerate(dataset.samples):
            if not str(record.get("sample_uid", "")):
                parser.error(
                    "Selector artifact export requires sample_uid in every record; "
                    f"missing at index {artifact_index}"
                )
        _prepare_selector_artifact_export(args)

    selector_artifact_indices = (
        np.arange(len(dataset.samples), dtype=np.int64)
        if args.selector_artifact_output_dir
        else None
    )

    print(f"🔔 开始极速评估推理 | samples={len(dataset)} | GPUs={world_size} | batch_size/GPU={args.batch_size}")
    if world_size <= 1:
        result = _evaluate_sample_records(
            args,
            data_dirs,
            dataset.samples,
            worker_rank=0,
            show_progress=True,
            selector_artifact_indices=selector_artifact_indices,
        )
    else:
        shards = [dataset.samples[rank::world_size] for rank in range(world_size)]
        selector_index_shards = (
            [
                selector_artifact_indices[rank::world_size]
                for rank in range(world_size)
            ]
            if selector_artifact_indices is not None
            else [None] * world_size
        )
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        procs = []
        for rank, (shard, selector_index_shard) in enumerate(
            zip(shards, selector_index_shards)
        ):
            proc = ctx.Process(
                target=_eval_worker,
                args=(
                    rank,
                    args,
                    data_dirs,
                    shard,
                    selector_index_shard,
                    result_queue,
                ),
            )
            proc.start()
            procs.append(proc)

        worker_results = []
        errors = []
        pending_ranks = set(range(len(procs)))
        try:
            while pending_ranks:
                try:
                    rank, result_item, error = result_queue.get(timeout=1.0)
                except queue_module.Empty:
                    failed_without_result = [
                        rank
                        for rank in pending_ranks
                        if procs[rank].exitcode is not None
                    ]
                    for rank in failed_without_result:
                        errors.append(
                            (
                                rank,
                                f"Worker exited without returning a result "
                                f"(exitcode={procs[rank].exitcode}).",
                            )
                        )
                        pending_ranks.remove(rank)
                    if errors:
                        break
                    continue
                pending_ranks.discard(rank)
                if error:
                    errors.append((rank, error))
                else:
                    worker_results.append(result_item)
                if errors:
                    break
        finally:
            if errors:
                for proc in procs:
                    if proc.is_alive():
                        proc.terminate()
            for proc in procs:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2.0)
            result_queue.close()
            result_queue.join_thread()

        if errors:
            for rank, error in errors:
                print(f"❌ Worker {rank} failed:\n{error}")
            raise RuntimeError("One or more evaluation workers failed.")
        result = _merge_eval_results(worker_results, args.diagnostic_max_frames)

    prediction_export = _finalize_prediction_export(args, dataset.samples)
    selector_artifact_export = _finalize_selector_artifact_export(
        args, dataset.samples
    )

    stats = result["stats"]

    report_text = _format_report(
        args,
        stats,
        result.get("touchanything_protocol_stats", {}),
    )
    if report_text is None:
        print("❌ 未产生任何有效的评估指标！可能数据集中 has_tactile 都是 0。")
        return

    print("\n🧮 推理完成，指标如下：")
    print("\n" + report_text)
    
    report_path = _report_path(args)
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_text + "\n")
    print(f"📝 最终评测报告已保存至: {report_path}")

    local_comparison_path, local_comparison = (
        _write_local_base_vs_fused_summary(args, report_path, result)
    )
    if local_comparison_path:
        print(f"🧭 Local base/fused 对照已保存至: {local_comparison_path}")

    selector_summary_path, selector_summary = _write_support_selector_summary(
        args, report_path, result
    )
    if selector_summary_path:
        print(f"Selector 独立对照已保存至: {selector_summary_path}")

    diagnostics_dir = _write_diagnostic_outputs(args, result)
    if diagnostics_dir:
        print(f"📊 Volumetric IoU / prediction distribution diagnostics 已保存至: {diagnostics_dir}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path(base_dir) / checkpoint_root
    exp_dir = checkpoint_root / args.exp_name if args.exp_name else None

    def read_json_if_exists(path):
        if path is None or not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    loss_config = read_json_if_exists(exp_dir / "loss_config.json") if exp_dir is not None else None
    model_config = read_json_if_exists(exp_dir / "model_config.json") if exp_dir is not None else None
    if loss_config is None:
        compact_checkpoint = _load_checkpoint(args.checkpoint)
        if isinstance(compact_checkpoint, dict):
            compact_loss_config = compact_checkpoint.get("loss_config")
            if isinstance(compact_loss_config, dict):
                loss_config = compact_loss_config
    if model_config is None:
        model_config = dict(getattr(args, "model_metadata", {}) or {})
    eval_config = {
        "args": vars(args),
        "exp_name": args.exp_name,
        "ckpt": args.ckpt,
        "ckpt_canonical": _canonical_checkpoint_selector(args.ckpt),
        "resolved_checkpoint": args.checkpoint,
        "checkpoint_symlink_target": os.readlink(checkpoint_path) if checkpoint_path.is_symlink() else None,
        "head_type": model_config.get("tactile_head_type", "dense_v2") if model_config else "dense_v2",
        "loss_config": loss_config,
        "model_config": model_config,
        "report_path": report_path,
        "local_base_vs_fused_summary": local_comparison_path,
        "local_base_vs_fused_metrics": local_comparison,
        "support_selector_summary": selector_summary_path,
        "support_selector_metrics": selector_summary,
        "diagnostics_dir": diagnostics_dir,
        "training_val_metrics_archive": args.training_val_metrics_archive,
        "prediction_export": prediction_export,
        "selector_artifact_export": selector_artifact_export,
        "metrics": _stats_summary(
            stats,
            result.get("touchanything_protocol_stats", {}),
            touchanything_min_contact_ratio=args.touchanything_min_contact_ratio,
        ),
    }
    with open(os.path.join(os.path.dirname(report_path), "eval_config.json"), "w", encoding="utf-8") as f_cfg:
        json.dump(eval_config, f_cfg, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"🧾 Eval config 已保存至: {os.path.join(os.path.dirname(report_path), 'eval_config.json')}")

if __name__ == '__main__':
    main()
