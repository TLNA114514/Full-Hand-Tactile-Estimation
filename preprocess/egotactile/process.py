import sys
import os

# Parse --gpu early so CUDA_VISIBLE_DEVICES is set before torch/model imports.
_gpus = ""
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        _gpus = sys.argv[i + 1]
        break
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

import argparse
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

try:
    import numpy as np
except ImportError:
    np = None


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(base_dir, "hamer"))
sys.path.append(os.path.join(base_dir, "evaluation"))

HAND_TO_JSON_KEY = {
    "left": "LH",
    "right": "RH",
}

DEFAULT_SCAN_EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "extracted_frames",
    "metadata",
    "artifacts",
}


def initialize_models(device):
    import torch
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer

    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))

    local_vitdet_path = os.path.join(base_dir, "hamer/_DATA/model_final_f05665.pkl")
    if os.path.exists(local_vitdet_path):
        detectron2_cfg.train.init_checkpoint = local_vitdet_path
    else:
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )

    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    try:
        import importlib.util

        vit_path = os.path.join(base_dir, "hamer/third-party/ViTPose/mmpose/models/backbones/vit.py")
        spec = importlib.util.spec_from_file_location("mmpose.models.backbones.vit", vit_path)
        vit_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vit_module)

        import mmpose.apis.inference

        custom_ckpt_path = os.path.join(base_dir, "hamer/third-party/ViTPose/mmcv_custom/checkpoint.py")
        spec_ckpt = importlib.util.spec_from_file_location("mmcv_custom.checkpoint", custom_ckpt_path)
        custom_ckpt_module = importlib.util.module_from_spec(spec_ckpt)
        spec_ckpt.loader.exec_module(custom_ckpt_module)
        mmpose.apis.inference.load_checkpoint = custom_ckpt_module.load_checkpoint
    except Exception as e:
        print(f"Warning: ViTPose dynamic registration failed: {e}")

    from vitpose_model import ViTPoseModel

    cpm = ViTPoseModel(device)
    return detector, cpm


def sanitize_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def sequence_cache_name(seq):
    return "__".join(sanitize_component(x) for x in seq["rel_parts"]) + ".json"


def load_frame_list(path):
    with Path(path).open("r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        frames = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                frames.append(json.loads(line))
        return frames


def normalize_sensor(sensor, pmin=5.0, pmax=200.0):
    if np is None:
        raise ImportError("numpy is required for EgoTactile extraction")
    values = np.asarray(sensor, dtype=np.float32).reshape(-1)
    if values.size != 256:
        raise ValueError(f"Expected sensor_256 with 256 values, got {values.size}")
    if pmax <= pmin:
        raise ValueError(f"pmax must be greater than pmin, got pmin={pmin}, pmax={pmax}")
    values = np.clip(values, pmin, pmax)
    return (values - pmin) / (pmax - pmin)


def np_to_json(value):
    if np is None:
        return value
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return np_to_json(value.tolist())
    if isinstance(value, np.generic):
        return np_to_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [np_to_json(v) for v in value]
    if isinstance(value, tuple):
        return [np_to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): np_to_json(v) for k, v in value.items()}
    return value


def write_json_atomic(path, data, indent=None):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        if indent is None:
            json.dump(data, f, separators=(",", ":"), allow_nan=False)
        else:
            json.dump(data, f, indent=indent, allow_nan=False)
    os.replace(tmp_path, path)


def iter_data_json_files(raw_root, exclude_dirs=None):
    raw_root = Path(raw_root)
    excluded = set(exclude_dirs or DEFAULT_SCAN_EXCLUDE_DIRS)
    for dirpath, dirnames, filenames in os.walk(raw_root):
        dirnames[:] = [name for name in dirnames if name not in excluded]
        if "data.json" in filenames and "video.mp4" in filenames:
            yield Path(dirpath) / "data.json"


def discover_sequences(raw_root, gaussian_npz_name, max_sequences=None, split_name=None, scan_exclude_dirs=None):
    raw_root = Path(raw_root)
    sequences = []
    for data_json in sorted(iter_data_json_files(raw_root, scan_exclude_dirs)):
        seq_dir = data_json.parent
        video_path = seq_dir / "video.mp4"
        if not video_path.exists():
            continue
        rel = seq_dir.relative_to(raw_root)
        rel_parts = rel.parts
        split = split_name or (rel_parts[0] if rel_parts else "all")
        sequences.append(
            {
                "split": split,
                "rel_seq": str(rel),
                "rel_parts": rel_parts,
                "seq_dir": str(seq_dir),
                "data_json": str(data_json),
                "video_path": str(video_path),
                "gaussian_npz": str(seq_dir / gaussian_npz_name),
            }
        )
        if max_sequences is not None and len(sequences) >= max_sequences:
            break
    return sequences


def hand_bbox_from_keypoints(keypoints, conf_thresh):
    valid = keypoints[:, 2] > conf_thresh
    if int(valid.sum()) <= 3:
        return None
    xy = keypoints[valid, :2]
    return {
        "bbox": [
            float(xy[:, 0].min()),
            float(xy[:, 1].min()),
            float(xy[:, 0].max()),
            float(xy[:, 1].max()),
        ],
        "score": float(keypoints[valid, 2].mean()),
    }


def choose_better_bbox(current, candidate):
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if candidate.get("score", 0.0) > current.get("score", 0.0) else current


def extract_bbox_worker(
    worker_idx,
    gpu_id,
    sequences_chunk,
    cache_dir,
    person_score_thresh,
    hand_conf_thresh,
    max_frames,
):
    import cv2
    import numpy as np
    import torch

    try:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        print(
            f"[BBox worker {worker_idx} GPU {gpu_id}] initializing models for "
            f"{len(sequences_chunk)} EgoTactile sequences"
        )
        detector, cpm = initialize_models(device)

        for seq in sequences_chunk:
            cache_path = Path(cache_dir) / sequence_cache_name(seq)
            if cache_path.exists():
                continue

            cap = cv2.VideoCapture(seq["video_path"])
            if not cap.isOpened():
                print(f"[BBox worker {worker_idx}] cannot open {seq['video_path']}")
                write_json_atomic(cache_path, {})
                continue

            try:
                frames = load_frame_list(seq["data_json"])
                frame_count = len(frames)
            except Exception:
                frame_count = 0
            video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if video_count > 0:
                frame_count = min(frame_count, video_count) if frame_count > 0 else video_count
            if max_frames is not None:
                frame_count = min(frame_count, max_frames) if frame_count > 0 else max_frames

            clip_bboxes = {}
            pbar_total = frame_count if frame_count > 0 else None
            desc = f"[BBox {worker_idx}/GPU {gpu_id}] {seq['rel_seq']}"
            frame_idx = 0
            with tqdm(total=pbar_total, desc=desc, leave=False, position=worker_idx) as pbar:
                while True:
                    if max_frames is not None and frame_idx >= max_frames:
                        break
                    if frame_count > 0 and frame_idx >= frame_count:
                        break

                    ret, img_bgr = cap.read()
                    if not ret:
                        break

                    frame_record = {
                        "left": {"bbox": None, "score": 0.0},
                        "right": {"bbox": None, "score": 0.0},
                    }

                    try:
                        det_out = detector(img_bgr)
                        det_instances = det_out["instances"]
                        valid_idx = (
                            (det_instances.pred_classes == 0)
                            & (det_instances.scores > person_score_thresh)
                        )
                        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
                    except Exception:
                        pred_bboxes = []
                        pred_scores = []

                    if len(pred_bboxes) > 0:
                        img_rgb = img_bgr[:, :, ::-1]
                        try:
                            vitposes_out = cpm.predict_pose(
                                img_rgb,
                                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
                            )
                            best_left = None
                            best_right = None
                            for vitposes in vitposes_out:
                                left_hand_keyp = vitposes["keypoints"][-42:-21]
                                right_hand_keyp = vitposes["keypoints"][-21:]
                                best_left = choose_better_bbox(
                                    best_left,
                                    hand_bbox_from_keypoints(left_hand_keyp, hand_conf_thresh),
                                )
                                best_right = choose_better_bbox(
                                    best_right,
                                    hand_bbox_from_keypoints(right_hand_keyp, hand_conf_thresh),
                                )
                            if best_left is not None:
                                frame_record["left"] = best_left
                            if best_right is not None:
                                frame_record["right"] = best_right
                        except Exception:
                            pass

                    clip_bboxes[str(frame_idx)] = frame_record
                    frame_idx += 1
                    pbar.update(1)

            cap.release()
            write_json_atomic(cache_path, clip_bboxes)

    except Exception as e:
        import traceback

        print(f"[BBox worker {worker_idx} GPU {gpu_id}] fatal error: {e}")
        traceback.print_exc()


def extract_egotactile_bboxes_multigpu(
    logical_gpus,
    sequences,
    cache_dir,
    bbox_json_path,
    person_score_thresh,
    hand_conf_thresh,
    max_frames=None,
    workers_per_gpu=1,
):
    import torch.multiprocessing as mp

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not sequences:
        print("No EgoTactile sequences found for bbox extraction.")
        return None

    worker_gpu_ids = []
    for gpu_id in logical_gpus:
        for _ in range(max(1, int(workers_per_gpu))):
            worker_gpu_ids.append(int(gpu_id))

    chunk_size = math.ceil(len(sequences) / len(worker_gpu_ids))
    chunks = [sequences[i : i + chunk_size] for i in range(0, len(sequences), chunk_size)]

    pool_args = []
    for worker_idx, gpu_id in enumerate(worker_gpu_ids):
        if worker_idx < len(chunks) and chunks[worker_idx]:
            pool_args.append(
                (
                    worker_idx,
                    gpu_id,
                    chunks[worker_idx],
                    str(cache_dir),
                    person_score_thresh,
                    hand_conf_thresh,
                    max_frames,
                )
            )

    print(f"Starting EgoTactile bbox extraction for {len(sequences)} sequences")
    print(f"  bbox workers: {len(pool_args)} ({workers_per_gpu} per GPU)")
    print(f"  bbox cache: {cache_dir}")

    if len(pool_args) > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        with mp.Pool(len(pool_args)) as pool:
            pool.starmap(extract_bbox_worker, pool_args)
    elif pool_args:
        extract_bbox_worker(*pool_args[0])

    merged = {}
    total_frames = 0
    for seq in sequences:
        cache_path = cache_dir / sequence_cache_name(seq)
        if not cache_path.exists():
            continue
        with cache_path.open("r", encoding="utf-8") as f:
            seq_bboxes = json.load(f)
        key = f"{seq['split']}/{seq['rel_seq']}"
        merged[key] = seq_bboxes
        total_frames += len(seq_bboxes)

    bbox_json_path = Path(bbox_json_path)
    bbox_json_path.parent.mkdir(parents=True, exist_ok=True)
    with bbox_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved merged EgoTactile bboxes for {total_frames} frames: {bbox_json_path}")
    return str(bbox_json_path)


def sample_is_complete(sample_dir):
    sample_dir = Path(sample_dir)
    meta_path = sample_dir / "meta.json"
    image_path = sample_dir / "image.jpg"
    if not image_path.exists() or image_path.stat().st_size <= 0:
        return False
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False
    return True


def get_npz_value(npz, key, frame_idx):
    if npz is None or key not in npz:
        return None
    arr = npz[key]
    if frame_idx >= arr.shape[0]:
        return None
    return arr[frame_idx]


def get_npz_valid(npz, key, frame_idx):
    if npz is None or key not in npz:
        return None
    arr = npz[key]
    if frame_idx >= arr.shape[0]:
        return None
    return bool(arr[frame_idx])


def frame_hand_sensor(frame, hand):
    key = HAND_TO_JSON_KEY[hand]
    if key not in frame or "sensor_256" not in frame[key]:
        return None
    return frame[key]["sensor_256"]


def make_frame_hand_meta(seq, frame_idx, hand, frame, bbox_record, gaussian_npz, pmin, pmax):
    raw_sensor = frame_hand_sensor(frame, hand)
    normalized_sensor = None
    if raw_sensor is not None:
        try:
            normalized_sensor = normalize_sensor(raw_sensor, pmin=pmin, pmax=pmax)
        except ValueError:
            normalized_sensor = None

    hand_bbox = (bbox_record or {}).get(hand, {})
    grid_key = f"{hand}_pressure_grid"
    mano_grid_key = f"{hand}_pressure_grid_mano"
    continuous_key = f"{hand}_pressure_continuous_subdiv"
    sensor_norm_key = f"{hand}_sensor_256_norm"
    sensor_valid_key = f"{hand}_sensor_valid"

    npz_sensor_norm = get_npz_value(gaussian_npz, sensor_norm_key, frame_idx)
    if npz_sensor_norm is not None:
        normalized_sensor = npz_sensor_norm

    gaussian = get_npz_value(gaussian_npz, continuous_key, frame_idx)
    pressure_grid = get_npz_value(gaussian_npz, grid_key, frame_idx)
    pressure_grid_mano = get_npz_value(gaussian_npz, mano_grid_key, frame_idx)
    sensor_valid = get_npz_valid(gaussian_npz, sensor_valid_key, frame_idx)
    if sensor_valid is None:
        sensor_valid = raw_sensor is not None

    original_data = {
        f"{hand}_pressure": normalized_sensor,
        f"{hand}_pressure_grid": pressure_grid,
        f"{hand}_pressure_grid_mano": pressure_grid_mano,
        f"{hand}_pressure_continuous_subdiv": gaussian,
    }

    return {
        "dataset": "EgoTactile",
        "split": seq["split"],
        "rel_seq": seq["rel_seq"],
        "seq_dir": seq["seq_dir"],
        "data_json": seq["data_json"],
        "video_path": seq["video_path"],
        "gaussian_npz": seq["gaussian_npz"],
        "frame_idx": frame_idx,
        "task_hand": frame.get("task_hand"),
        "hand": hand,
        "is_right": 1 if hand == "right" else 0,
        "sensor_valid": bool(sensor_valid),
        "bbox": hand_bbox.get("bbox"),
        "bbox_score": float(hand_bbox.get("score", 0.0)),
        "image": "image.jpg",
        "raw_sensor_256": raw_sensor,
        "normalized_sensor_256": normalized_sensor,
        "normalized_pressure_grid": pressure_grid,
        "normalized_pressure_grid_mano": pressure_grid_mano,
        "gaussian_pressure": gaussian,
        "pressure_keys": {
            "raw_sensor": f"{HAND_TO_JSON_KEY[hand]}.sensor_256",
            "normalized_sensor": sensor_norm_key if npz_sensor_norm is not None else "computed_from_raw_sensor_256",
            "normalized_pressure_grid": grid_key,
            "normalized_pressure_grid_mano": mano_grid_key,
            "gaussian_pressure": continuous_key,
        },
        "normalization": {
            "type": "linear_clip",
            "pmin": float(pmin),
            "pmax": float(pmax),
            "formula": "(clip(raw,pmin,pmax)-pmin)/(pmax-pmin)",
        },
        "original_frame_record": frame,
        "original_hdf5_data": original_data,
        "keypoints_3d_cam": np.zeros((21, 3), dtype=np.float32),
        "valid_mask": np.zeros(21, dtype=bool),
    }


def extract_sequence_to_disk_worker(args):
    (
        seq,
        seq_bboxes,
        existing_samples,
        target_samples,
        output_dir,
        gaussian_npz_name,
        max_frames,
        meta_indent,
        pmin,
        pmax,
        keep_no_bbox,
        require_gaussian,
        trust_registry,
    ) = args
    if np is None:
        raise ImportError("numpy is required for EgoTactile frame extraction")
    import cv2

    output_dir = Path(output_dir)
    written = 0
    skipped = 0
    failed = 0
    registry_entries = []
    gaussian_npz = None
    cap = None

    try:
        frames = load_frame_list(seq["data_json"])
        if not frames:
            return {
                "written": written,
                "skipped": skipped,
                "failed": 1,
                "registry": registry_entries,
                "error": f"empty data.json: {seq['data_json']}",
            }

        npz_path = Path(seq["seq_dir"]) / gaussian_npz_name
        if npz_path.exists():
            gaussian_npz = np.load(npz_path)
        elif require_gaussian:
            return {
                "written": written,
                "skipped": skipped,
                "failed": 1,
                "registry": registry_entries,
                "error": f"missing gaussian npz: {npz_path}",
            }

        cap = cv2.VideoCapture(seq["video_path"])
        if not cap.isOpened():
            return {
                "written": written,
                "skipped": skipped,
                "failed": 1,
                "registry": registry_entries,
                "error": f"unreadable video: {seq['video_path']}",
            }

        video_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        cap = None
        frame_count = len(frames)
        if video_count > 0:
            frame_count = min(frame_count, video_count)
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)

        existing_samples = set(tuple(x) for x in (existing_samples or []))
        target_mode = target_samples is not None
        if target_mode:
            expected_samples = [tuple(x) for x in target_samples]
            if expected_samples:
                frame_count = min(frame_count, max(frame_idx for frame_idx, _ in expected_samples) + 1)
        else:
            expected_samples = []
            for frame_idx in range(frame_count):
                frame = frames[frame_idx]
                bbox_record = seq_bboxes.get(str(frame_idx), {})
                for hand in ("left", "right"):
                    raw_sensor = frame_hand_sensor(frame, hand)
                    npz_valid = get_npz_valid(gaussian_npz, f"{hand}_sensor_valid", frame_idx)
                    has_sensor = raw_sensor is not None if npz_valid is None else bool(npz_valid)
                    if not has_sensor:
                        continue
                    hand_bbox = (bbox_record or {}).get(hand, {})
                    has_bbox = hand_bbox.get("bbox") is not None
                    if not has_bbox and not keep_no_bbox:
                        continue
                    expected_samples.append((frame_idx, hand))

        if trust_registry and expected_samples and all(sample in existing_samples for sample in expected_samples):
            return {
                "written": written,
                "skipped": len(expected_samples),
                "failed": failed,
                "registry": registry_entries,
                "error": None,
            }

        expected_samples_set = set(expected_samples)
        cap = cv2.VideoCapture(seq["video_path"])
        if not cap.isOpened():
            return {
                "written": written,
                "skipped": skipped,
                "failed": 1,
                "registry": registry_entries,
                "error": f"unreadable video: {seq['video_path']}",
            }

        for frame_idx in range(frame_count):
            frame_needs_image = any(
                (frame_idx, hand) in expected_samples_set
                and (not trust_registry or (frame_idx, hand) not in existing_samples)
                for hand in ("left", "right")
            )

            ok, frame_bgr = cap.read() if frame_needs_image else (cap.grab(), None)
            if not ok:
                failed += 1
                break

            frame = frames[frame_idx]
            bbox_record = seq_bboxes.get(str(frame_idx), {})

            for hand in ("left", "right"):
                raw_sensor = frame_hand_sensor(frame, hand)
                npz_valid = get_npz_valid(gaussian_npz, f"{hand}_sensor_valid", frame_idx)
                has_sensor = raw_sensor is not None if npz_valid is None else bool(npz_valid)
                if not has_sensor:
                    continue

                hand_bbox = (bbox_record or {}).get(hand, {})
                has_bbox = hand_bbox.get("bbox") is not None
                if not has_bbox and not keep_no_bbox:
                    skipped += 1
                    continue

                if trust_registry and (frame_idx, hand) in existing_samples:
                    skipped += 1
                    continue

                folder_name = "__".join(
                    sanitize_component(x)
                    for x in [*seq["rel_parts"], f"{frame_idx:06d}", hand]
                )
                sample_dir = output_dir / seq["split"] / folder_name
                meta_path = sample_dir / "meta.json"

                if not target_mode and sample_is_complete(sample_dir):
                    skipped += 1
                    continue
                if meta_path.exists():
                    try:
                        meta_path.unlink()
                    except OSError:
                        pass

                sample_dir.mkdir(parents=True, exist_ok=True)
                image_path = sample_dir / "image.jpg"
                cv2.imwrite(str(image_path), frame_bgr)

                meta = make_frame_hand_meta(
                    seq,
                    frame_idx,
                    hand,
                    frame,
                    bbox_record,
                    gaussian_npz,
                    pmin,
                    pmax,
                )
                write_json_atomic(meta_path, np_to_json(meta), indent=meta_indent)

                registry_entries.append(
                    {
                        "split": seq["split"],
                        "rel_seq": seq["rel_seq"],
                        "frame_idx": frame_idx,
                        "hand": hand,
                        "is_right": 1 if hand == "right" else 0,
                        "sample_dir": str(sample_dir),
                        "has_bbox": has_bbox,
                        "has_gaussian": meta["gaussian_pressure"] is not None,
                    }
                )
                written += 1

    except Exception as exc:
        failed += 1
        return {
            "written": written,
            "skipped": skipped,
            "failed": failed,
            "registry": registry_entries,
            "error": f"{seq['rel_seq']}: {exc}",
        }
    finally:
        if cap is not None:
            cap.release()
        if gaussian_npz is not None:
            gaussian_npz.close()

    return {
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "registry": registry_entries,
        "error": None,
    }


def merge_extract_result(result, registry, existing, written, skipped, failed, errors):
    written += result.get("written", 0)
    skipped += result.get("skipped", 0)
    failed += result.get("failed", 0)
    if result.get("error") and len(errors) < 10:
        errors.append(result["error"])

    for item in result.get("registry", []):
        reg_key = f"{item['split']}/{item['rel_seq']}/{item['frame_idx']}/{item['hand']}"
        if reg_key not in existing:
            registry.append(item)
            existing.add(reg_key)
    return written, skipped, failed


def registry_key(item):
    return f"{item['split']}/{item['rel_seq']}/{item['frame_idx']}/{item['hand']}"


def sample_files_exist(sample_dir):
    sample_dir = Path(sample_dir)
    image_path = sample_dir / "image.jpg"
    meta_path = sample_dir / "meta.json"
    return (
        image_path.exists()
        and image_path.stat().st_size > 0
        and meta_path.exists()
        and meta_path.stat().st_size > 0
    )


def validate_egotactile_registry(registry, check_workers, strict=False):
    def check_item(item):
        sample_dir = item.get("sample_dir")
        if not sample_dir:
            return None
        ok = sample_is_complete(sample_dir) if strict else sample_files_exist(sample_dir)
        return registry_key(item) if ok else None

    check_workers = max(1, int(check_workers))
    if check_workers == 1:
        results = (check_item(item) for item in registry)
        iterator = tqdm(results, total=len(registry), desc="Checking EgoTactile registry")
        return {key for key in iterator if key is not None}

    existing = set()
    with ThreadPoolExecutor(max_workers=check_workers) as executor:
        for key in tqdm(
            executor.map(check_item, registry),
            total=len(registry),
            desc="Checking EgoTactile registry",
        ):
            if key is not None:
                existing.add(key)
    return existing


def expected_samples_for_sequence(
    seq,
    seq_bboxes,
    existing_samples,
    gaussian_npz_name,
    max_frames,
    keep_no_bbox,
    require_gaussian,
):
    gaussian_npz = None
    try:
        frames = load_frame_list(seq["data_json"])
        if not frames:
            return {
                "seq": seq,
                "missing": [],
                "total_expected": 0,
                "skipped_existing": 0,
                "error": f"empty data.json: {seq['data_json']}",
            }

        npz_path = Path(seq["seq_dir"]) / gaussian_npz_name
        if npz_path.exists():
            gaussian_npz = np.load(npz_path)
        elif require_gaussian:
            return {
                "seq": seq,
                "missing": [],
                "total_expected": 0,
                "skipped_existing": 0,
                "error": f"missing gaussian npz: {npz_path}",
            }

        frame_count = len(frames)
        if gaussian_npz is not None and "frame_count" in gaussian_npz:
            try:
                frame_count = min(frame_count, int(gaussian_npz["frame_count"]))
            except Exception:
                pass
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)

        existing_samples = set(existing_samples or [])
        expected = []
        for frame_idx in range(frame_count):
            frame = frames[frame_idx]
            bbox_record = seq_bboxes.get(str(frame_idx), {})
            for hand in ("left", "right"):
                raw_sensor = frame_hand_sensor(frame, hand)
                npz_valid = get_npz_valid(gaussian_npz, f"{hand}_sensor_valid", frame_idx)
                has_sensor = raw_sensor is not None if npz_valid is None else bool(npz_valid)
                if not has_sensor:
                    continue
                hand_bbox = (bbox_record or {}).get(hand, {})
                has_bbox = hand_bbox.get("bbox") is not None
                if not has_bbox and not keep_no_bbox:
                    continue
                expected.append((frame_idx, hand))

        missing = [sample for sample in expected if sample not in existing_samples]
        return {
            "seq": seq,
            "missing": missing,
            "total_expected": len(expected),
            "skipped_existing": len(expected) - len(missing),
            "error": None,
        }
    except Exception as exc:
        return {
            "seq": seq,
            "missing": [],
            "total_expected": 0,
            "skipped_existing": 0,
            "error": f"{seq['rel_seq']}: {exc}",
        }
    finally:
        if gaussian_npz is not None:
            gaussian_npz.close()


def extract_egotactile_to_disk(
    sequences,
    bbox_json_path,
    output_dir,
    gaussian_npz_name,
    max_frames=None,
    meta_indent=None,
    extract_workers=1,
    pmin=5.0,
    pmax=200.0,
    keep_no_bbox=False,
    require_gaussian=False,
    trust_registry=True,
    check_workers=32,
    strict_check_registry=False,
    prefilter_workers=32,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_json_path = Path(bbox_json_path) if bbox_json_path else None
    if bbox_json_path is not None and bbox_json_path.exists():
        with bbox_json_path.open("r", encoding="utf-8") as f:
            all_bboxes = json.load(f)
    elif keep_no_bbox:
        print("Warning: bbox JSON not found; continuing with empty bboxes because --keep_no_bbox is set.")
        all_bboxes = {}
    else:
        raise FileNotFoundError(f"Bbox JSON not found: {bbox_json_path}")

    registry_path = output_dir / "egotactile_frames_registry.json"
    registry = []
    existing = set()
    if registry_path.exists():
        try:
            print(f"Loading EgoTactile registry: {registry_path}")
            with registry_path.open("r", encoding="utf-8") as f:
                registry = json.load(f)
            print(f"Loaded {len(registry)} EgoTactile registry entries.")
            if trust_registry:
                print("Using EgoTactile registry as fast skip cache.")
                existing = {registry_key(r) for r in registry}
            else:
                existing = validate_egotactile_registry(registry, check_workers, strict=strict_check_registry)
                trust_registry = True
        except Exception:
            registry = []
            existing = set()
    registry_original_len = len(registry)

    existing_by_seq = defaultdict(set)
    if existing:
        print("Indexing EgoTactile existing samples by sequence...")
    for r in registry:
        key = registry_key(r)
        if key in existing:
            existing_by_seq[f"{r['split']}/{r['rel_seq']}"].add((int(r["frame_idx"]), r["hand"]))

    print(f"Prefiltering EgoTactile sequences for missing samples ({len(sequences)} sequences)...")
    prefilter_workers = max(1, int(prefilter_workers))
    prefilter_tasks = []
    for seq in sequences:
        bbox_key = f"{seq['split']}/{seq['rel_seq']}"
        prefilter_tasks.append(
            (seq, all_bboxes.get(bbox_key, {}), existing_by_seq.get(bbox_key, set()))
        )

    written = 0
    skipped = 0
    failed = 0
    errors = []

    tasks = []
    if prefilter_workers == 1:
        results_iter = (
            expected_samples_for_sequence(
                seq,
                seq_bboxes,
                existing_samples,
                gaussian_npz_name,
                max_frames,
                keep_no_bbox,
                require_gaussian,
            )
            for seq, seq_bboxes, existing_samples in prefilter_tasks
        )
        iterator = tqdm(results_iter, total=len(prefilter_tasks), desc="Prefiltering EgoTactile")
        prefilter_results = list(iterator)
    else:
        prefilter_results = []
        with ThreadPoolExecutor(max_workers=prefilter_workers) as executor:
            futures = [
                executor.submit(
                    expected_samples_for_sequence,
                    seq,
                    seq_bboxes,
                    existing_samples,
                    gaussian_npz_name,
                    max_frames,
                    keep_no_bbox,
                    require_gaussian,
                )
                for seq, seq_bboxes, existing_samples in prefilter_tasks
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Prefiltering EgoTactile"):
                prefilter_results.append(future.result())

    for result in prefilter_results:
        if result.get("error"):
            failed += 1
            if len(errors) < 10:
                errors.append(result["error"])
            continue
        skipped += result.get("skipped_existing", 0)
        missing = result.get("missing", [])
        if not missing:
            continue
        seq = result["seq"]
        bbox_key = f"{seq['split']}/{seq['rel_seq']}"
        tasks.append(
            (
                seq,
                all_bboxes.get(bbox_key, {}),
                existing_by_seq.get(bbox_key, set()),
                missing,
                str(output_dir),
                gaussian_npz_name,
                max_frames,
                meta_indent,
                pmin,
                pmax,
                keep_no_bbox,
                require_gaussian,
                trust_registry,
            )
        )

    print(f"Submitting {len(tasks)} EgoTactile sequences with missing samples to extraction workers.")

    extract_workers = max(1, int(extract_workers))
    if extract_workers == 1:
        results_iter = (extract_sequence_to_disk_worker(task) for task in tasks)
        iterator = tqdm(results_iter, total=len(tasks), desc="Extracting EgoTactile frames")
        for result in iterator:
            written, skipped, failed = merge_extract_result(
                result, registry, existing, written, skipped, failed, errors
            )
    else:
        with ProcessPoolExecutor(max_workers=extract_workers) as executor:
            futures = [executor.submit(extract_sequence_to_disk_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting EgoTactile frames"):
                result = future.result()
                written, skipped, failed = merge_extract_result(
                    result, registry, existing, written, skipped, failed, errors
                )

    if len(registry) != registry_original_len:
        with registry_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    else:
        print("Registry unchanged; skipped rewriting registry JSON.")

    print("EgoTactile frame extraction finished.")
    print(f"  new samples: {written}")
    print(f"  skipped samples: {skipped}")
    print(f"  failed/skipped sequences or frames: {failed}")
    print(f"  output_dir: {output_dir}")
    print(f"  registry: {registry_path}")
    if errors:
        print("  first errors:")
        for err in errors[:10]:
            print(f"    {err}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract EgoTactile video frames, hand bboxes, and pressure metadata into per-hand samples."
    )
    parser.add_argument("--gpu", type=str, default="0", help="Visible GPU ids, e.g. 0 or 0,1,2,3")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="/data1/jiangrui/EgoTactile/Raw_data",
        help="EgoTactile Raw_data root containing */*/ */repeat*/data.json and video.mp4.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root for extracted per-hand samples. Defaults to <raw_root>/extracted_frames.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join(base_dir, "preprocess/artifacts/egotactile/bboxes_cache"),
        help="Per-sequence bbox cache directory.",
    )
    parser.add_argument(
        "--bbox_json",
        type=str,
        default=os.path.join(base_dir, "preprocess/artifacts/egotactile/egotactile_all_bboxes.json"),
        help="Merged bbox JSON path.",
    )
    parser.add_argument("--gaussian_npz_name", default="pressure_grids_egotactile.npz")
    parser.add_argument("--split_name", default=None, help="Override split name. Default uses first path component.")
    parser.add_argument("--pmin", type=float, default=5.0)
    parser.add_argument("--pmax", type=float, default=200.0)
    parser.add_argument("--person_score_thresh", type=float, default=0.5)
    parser.add_argument("--hand_conf_thresh", type=float, default=0.5)
    parser.add_argument("--max_sequences", type=int, default=None, help="Debug limit.")
    parser.add_argument("--max_frames", type=int, default=None, help="Debug limit per sequence.")
    parser.add_argument(
        "--bbox_workers_per_gpu",
        type=int,
        default=1,
        help="Number of independent bbox extractor processes per visible GPU.",
    )
    parser.add_argument(
        "--extract_workers",
        type=int,
        default=1,
        help="CPU worker processes for video decoding/JPEG/meta writing.",
    )
    parser.add_argument(
        "--check_workers",
        type=int,
        default=32,
        help="Threads for validating registry entries when --no_trust_registry is used.",
    )
    parser.add_argument(
        "--prefilter_workers",
        type=int,
        default=32,
        help="Threads for finding EgoTactile sequences/samples that still need extraction.",
    )
    parser.add_argument(
        "--meta_indent",
        type=int,
        default=None,
        help="Indent meta.json. Default writes compact JSON for speed and smaller files.",
    )
    parser.add_argument("--skip_bbox", action="store_true", help="Use an existing --bbox_json and only write samples.")
    parser.add_argument("--bbox_only", action="store_true", help="Only extract/merge bboxes.")
    parser.add_argument(
        "--keep_no_bbox",
        action="store_true",
        help="Write pressure/RGB samples even when the hand bbox is missing. Default skips them.",
    )
    parser.add_argument(
        "--require_gaussian",
        action="store_true",
        help="Skip sequences without --gaussian_npz_name. Default still writes raw/normalized sensor metadata.",
    )
    parser.add_argument(
        "--no_trust_registry",
        action="store_true",
        help="Do not use the registry as a fast skip cache; re-check sample folders/meta.json on disk.",
    )
    parser.add_argument(
        "--strict_check_registry",
        action="store_true",
        help="When --no_trust_registry is used, parse meta.json instead of only checking image/meta file existence.",
    )
    parser.add_argument(
        "--scan_exclude_dirs",
        nargs="*",
        default=sorted(DEFAULT_SCAN_EXCLUDE_DIRS),
        help="Directory names to prune while discovering EgoTactile data.json files.",
    )
    args = parser.parse_args()

    os.chdir(base_dir)

    raw_root = Path(args.raw_root)
    if args.output_dir is None:
        args.output_dir = str(raw_root / "extracted_frames")

    sequences = discover_sequences(
        raw_root,
        args.gaussian_npz_name,
        max_sequences=args.max_sequences,
        split_name=args.split_name,
        scan_exclude_dirs=args.scan_exclude_dirs,
    )
    print(f"Resolved {len(sequences)} EgoTactile sequences under {raw_root}")
    if not sequences:
        raise RuntimeError("No EgoTactile sequences found. Check --raw_root.")

    gpu_list = [g.strip() for g in args.gpu.split(",") if g.strip()] or ["0"]
    logical_gpus = list(range(len(gpu_list)))

    bbox_path = args.bbox_json
    if not args.skip_bbox:
        bbox_path = extract_egotactile_bboxes_multigpu(
            logical_gpus,
            sequences,
            args.cache_dir,
            args.bbox_json,
            args.person_score_thresh,
            args.hand_conf_thresh,
            max_frames=args.max_frames,
            workers_per_gpu=args.bbox_workers_per_gpu,
        )

    if args.bbox_only:
        return
    if (not bbox_path or not Path(bbox_path).exists()) and not args.keep_no_bbox:
        raise FileNotFoundError(f"Bbox JSON not found: {bbox_path}")

    extract_egotactile_to_disk(
        sequences,
        bbox_path,
        args.output_dir,
        args.gaussian_npz_name,
        max_frames=args.max_frames,
        meta_indent=args.meta_indent,
        extract_workers=args.extract_workers,
        pmin=args.pmin,
        pmax=args.pmax,
        keep_no_bbox=args.keep_no_bbox,
        require_gaussian=args.require_gaussian,
        trust_registry=not args.no_trust_registry,
        check_workers=args.check_workers,
        strict_check_registry=args.strict_check_registry,
        prefilter_workers=args.prefilter_workers,
    )


if __name__ == "__main__":
    main()
