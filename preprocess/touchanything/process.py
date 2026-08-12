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


def cache_name_for_clip(clip):
    return "__".join(
        sanitize_component(x)
        for x in [clip["split"], clip["scene"], clip["task"], clip["clip_id"]]
    ) + ".json"


def extract_path_from_split_entry(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (list, tuple)) and entry:
        return extract_path_from_split_entry(entry[0])
    if isinstance(entry, dict):
        for key in (
            "hdf5_path",
            "file_path",
            "path",
            "hdf5",
            "file",
            "trajectory",
            "traj",
            "clip_path",
            "clip",
        ):
            if key in entry:
                return extract_path_from_split_entry(entry[key])
    raise ValueError(f"Unsupported split entry: {entry!r}")


def build_raw_clip_index(raw_root):
    raw_root = Path(raw_root)
    index = {}
    skip_names = {"extracted_frames", "metadata", "__pycache__"}

    for scene_dir in raw_root.iterdir():
        if not scene_dir.is_dir() or scene_dir.name in skip_names:
            continue
        for task_dir in scene_dir.iterdir():
            if not task_dir.is_dir() or task_dir.name in skip_names:
                continue
            for clip_dir in task_dir.iterdir():
                if not clip_dir.is_dir() or clip_dir.name in skip_names:
                    continue
                if not (
                    (clip_dir / "pressure_grids.npz").exists()
                    or (clip_dir / "jq_pressure.json").exists()
                    or (clip_dir / "chest.mp4").exists()
                ):
                    continue
                try:
                    rel = clip_dir.relative_to(raw_root)
                except ValueError:
                    continue
                if len(rel.parts) < 3:
                    continue
                scene, task, clip_id = rel.parts[-3], rel.parts[-2], rel.parts[-1]
                index[("scene_task_clip", scene, task, clip_id)] = clip_dir
                index[("task_clip", task, clip_id)] = clip_dir
                index.setdefault(("clip", clip_id), clip_dir)
    return index


def split_entry_to_raw_clip(entry, raw_root, raw_index):
    hdf5_path_str = extract_path_from_split_entry(entry)
    hdf5_path = Path(hdf5_path_str)
    parts = list(hdf5_path.parts)
    clip_id = hdf5_path.stem

    candidates = []
    if len(parts) >= 3:
        scene, task = parts[-3], parts[-2]
        candidates.append(Path(raw_root) / scene / task / clip_id)
        candidates.append(raw_index.get(("scene_task_clip", scene, task, clip_id)))
    if len(parts) >= 2:
        task = parts[-2]
        candidates.append(raw_index.get(("task_clip", task, clip_id)))
    candidates.append(raw_index.get(("clip", clip_id)))

    for candidate in candidates:
        if candidate is not None and Path(candidate).exists():
            clip_dir = Path(candidate)
            rel = clip_dir.relative_to(raw_root)
            scene, task, resolved_clip_id = rel.parts[-3], rel.parts[-2], rel.parts[-1]
            return {
                "hdf5_path": hdf5_path_str,
                "raw_clip_dir": str(clip_dir),
                "scene": scene,
                "task": task,
                "clip_id": resolved_clip_id,
                "rel_clip": str(Path(scene) / task / resolved_clip_id),
            }

    expected = None
    if len(parts) >= 3:
        expected = Path(raw_root) / parts[-3] / parts[-2] / clip_id
    suffix = f" (expected raw clip like: {expected})" if expected is not None else ""
    raise FileNotFoundError(f"Cannot map split entry to raw clip: {hdf5_path_str}{suffix}")


def load_split_clips(split_json_path, raw_root, requested_splits=None, max_clips=None):
    split_json_path = Path(split_json_path)
    raw_root = Path(raw_root)
    with split_json_path.open("r") as f:
        split_data = json.load(f)

    if requested_splits is not None:
        requested_splits = {s.strip() for s in requested_splits if s.strip()}

    raw_index = build_raw_clip_index(raw_root)
    raw_clip_count = sum(1 for key in raw_index if key[0] == "scene_task_clip")
    clips = []
    missing = []

    if isinstance(split_data, dict):
        split_items = split_data.items()
    elif isinstance(split_data, list):
        split_items = [("all", split_data)]
    else:
        raise ValueError(f"Unsupported split JSON root type: {type(split_data).__name__}")

    for split_name, entries in split_items:
        if requested_splits is not None and split_name not in requested_splits:
            continue
        if isinstance(entries, dict):
            entries = list(entries.values())
        for entry in entries:
            try:
                clip = split_entry_to_raw_clip(entry, raw_root, raw_index)
            except Exception as e:
                missing.append(str(e))
                continue
            clip["split"] = split_name
            clips.append(clip)
            if max_clips is not None and len(clips) >= max_clips:
                return clips, missing, raw_clip_count

    return clips, missing, raw_clip_count


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


def extract_bbox_worker(gpu_id, clips_chunk, cache_dir, person_score_thresh, hand_conf_thresh, max_frames):
    import cv2
    import numpy as np
    import torch

    try:
        device = torch.device(f"cuda:{gpu_id}")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        print(f"[Worker GPU {gpu_id}] initializing models for {len(clips_chunk)} TouchAnything clips")
        detector, cpm = initialize_models(device)

        for clip in clips_chunk:
            cache_path = Path(cache_dir) / cache_name_for_clip(clip)
            if cache_path.exists():
                continue

            video_path = Path(clip["raw_clip_dir"]) / "chest.mp4"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[Worker GPU {gpu_id}] cannot open {video_path}")
                cache_path.write_text("{}", encoding="utf-8")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames is not None:
                frame_count = min(frame_count, max_frames) if frame_count > 0 else max_frames

            clip_bboxes = {}
            pbar_total = frame_count if frame_count > 0 else None
            desc = f"[GPU {gpu_id}] {clip['scene']}/{clip['task']}/{clip['clip_id']}"
            frame_idx = 0
            with tqdm(total=pbar_total, desc=desc, leave=False, position=gpu_id) as pbar:
                while True:
                    if max_frames is not None and frame_idx >= max_frames:
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
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(clip_bboxes, f)

    except Exception as e:
        import traceback

        print(f"[Worker GPU {gpu_id}] fatal error: {e}")
        traceback.print_exc()


def extract_touchanything_bboxes_multigpu(
    logical_gpus,
    clips,
    cache_dir,
    bbox_json_path,
    person_score_thresh,
    hand_conf_thresh,
    max_frames=None,
):
    import torch.multiprocessing as mp

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not clips:
        print("No TouchAnything clips found for bbox extraction.")
        return None

    print(f"Starting chest bbox extraction for {len(clips)} TouchAnything clips")
    print(f"Bbox cache: {cache_dir}")

    num_gpus = len(logical_gpus)
    chunk_size = math.ceil(len(clips) / num_gpus)
    chunks = [clips[i : i + chunk_size] for i in range(0, len(clips), chunk_size)]

    pool_args = []
    for i, gpu_id in enumerate(logical_gpus):
        if i < len(chunks) and chunks[i]:
            pool_args.append(
                (
                    int(gpu_id),
                    chunks[i],
                    str(cache_dir),
                    person_score_thresh,
                    hand_conf_thresh,
                    max_frames,
                )
            )

    if len(pool_args) > 1:
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        with mp.Pool(len(pool_args)) as pool:
            pool.starmap(extract_bbox_worker, pool_args)
    else:
        extract_bbox_worker(*pool_args[0])

    merged = {}
    total_frames = 0
    for clip in clips:
        cache_path = cache_dir / cache_name_for_clip(clip)
        if not cache_path.exists():
            continue
        with cache_path.open("r", encoding="utf-8") as f:
            clip_bboxes = json.load(f)
        key = f"{clip['split']}/{clip['scene']}/{clip['task']}/{clip['clip_id']}"
        merged[key] = clip_bboxes
        total_frames += len(clip_bboxes)

    bbox_json_path = Path(bbox_json_path)
    bbox_json_path.parent.mkdir(parents=True, exist_ok=True)
    with bbox_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Saved merged TouchAnything chest bboxes for {total_frames} frames: {bbox_json_path}")
    return str(bbox_json_path)


def load_jsonl_by_frame(jsonl_path):
    records = {}
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            frame_idx = int(item.get("frame_index", line_idx))
            records[frame_idx] = item
    return records


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
        if math.isfinite(value):
            return value
        return None
    if isinstance(value, list):
        return [np_to_json(v) for v in value]
    if isinstance(value, tuple):
        return [np_to_json(v) for v in value]
    if isinstance(value, dict):
        return {str(k): np_to_json(v) for k, v in value.items()}
    return value


def read_video_counts(clip_dir):
    import cv2

    counts = {}
    for view in ("chest", "left", "right"):
        cap = cv2.VideoCapture(str(Path(clip_dir) / f"{view}.mp4"))
        counts[view] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
    return counts


def count_jsonl_records(jsonl_path):
    count = 0
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_frame_images(caps, sample_dir):
    import cv2

    frames = {}
    for view, cap in caps.items():
        ret, frame = cap.read()
        if not ret:
            return False
        frames[view] = frame
    # Only the chest view is consumed by tactile training/evaluation. Keep
    # advancing all source streams for alignment, but do not materialize the
    # two wrist views in the processed dataset.
    return bool(cv2.imwrite(str(sample_dir / "chest.jpg"), frames["chest"]))


def sample_is_complete(sample_dir):
    sample_dir = Path(sample_dir)
    meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return False
    image_path = sample_dir / "chest.jpg"
    if not image_path.exists() or image_path.stat().st_size <= 0:
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False
    return True


def write_json_atomic(path, data, indent=None):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        if indent is None:
            json.dump(data, f, separators=(",", ":"), allow_nan=False)
        else:
            json.dump(data, f, indent=indent, allow_nan=False)
    os.replace(tmp_path, path)


def make_frame_meta(clip, frame_idx, bbox_record, pressure_records, pressure_npz, pressure_npz_error=None):
    raw_pressure = pressure_records.get(frame_idx, {})

    hands = {}
    for hand in ("left", "right"):
        raw_key = f"sensor_{hand}"
        grid_key = f"{hand}_pressure_grid"
        continuous_key = f"{hand}_pressure_continuous_subdiv"
        alt_continuous_key = f"{hand}_pressure_continuous"

        gaussian = None
        gaussian_key_used = None
        if pressure_npz is not None and continuous_key in pressure_npz and frame_idx < pressure_npz[continuous_key].shape[0]:
            gaussian = pressure_npz[continuous_key][frame_idx]
            gaussian_key_used = continuous_key
        elif (
            pressure_npz is not None
            and alt_continuous_key in pressure_npz
            and frame_idx < pressure_npz[alt_continuous_key].shape[0]
        ):
            gaussian = pressure_npz[alt_continuous_key][frame_idx]
            gaussian_key_used = alt_continuous_key

        normalized_grid = None
        if pressure_npz is not None and grid_key in pressure_npz and frame_idx < pressure_npz[grid_key].shape[0]:
            normalized_grid = pressure_npz[grid_key][frame_idx]

        hand_bbox = (bbox_record or {}).get(hand, {})
        hands[hand] = {
            "is_right": 1 if hand == "right" else 0,
            "bbox_chest": hand_bbox.get("bbox"),
            "bbox_score": float(hand_bbox.get("score", 0.0)),
            "raw_pressure": raw_pressure.get(raw_key),
            "raw_pressure_key": raw_key,
            "normalized_pressure_grid": normalized_grid,
            "normalized_pressure_grid_key": grid_key,
            "gaussian_pressure": gaussian,
            "gaussian_pressure_key": gaussian_key_used,
        }

    return {
        "dataset": "TouchAnything",
        "split": clip["split"],
        "scene": clip["scene"],
        "task": clip["task"],
        "clip": clip["clip_id"],
        "rel_clip": clip["rel_clip"],
        "raw_clip_dir": clip["raw_clip_dir"],
        "source_hdf5_path_from_split": clip["hdf5_path"],
        "frame_idx": frame_idx,
        "timestamp": raw_pressure.get("ts"),
        "jq_pressure_frame_index": raw_pressure.get("frame_index"),
        "views": {
            "chest": "chest.jpg",
        },
        "bbox_view": "chest",
        "hands": hands,
        "pressure_npz_available": pressure_npz is not None,
        "pressure_npz_error": pressure_npz_error,
        "original_jq_pressure": raw_pressure,
    }


def extract_clip_to_disk_worker(args):
    (
        clip,
        clip_bboxes,
        target_frames,
        output_dir,
        max_frames,
        meta_indent,
        allow_bad_npz,
        existing_frames,
        force_rewrite_frames,
        trust_registry,
    ) = args
    if np is None:
        raise ImportError("numpy is required for TouchAnything frame extraction")
    import cv2

    output_dir = Path(output_dir)
    written = 0
    skipped = 0
    failed = 0
    registry_entries = []

    clip_dir = Path(clip["raw_clip_dir"])
    required = [
        clip_dir / "chest.mp4",
        clip_dir / "left.mp4",
        clip_dir / "right.mp4",
        clip_dir / "jq_pressure.json",
    ]
    if not allow_bad_npz:
        required.append(clip_dir / "pressure_grids.npz")
    if any(not p.exists() for p in required):
        return {
            "written": written,
            "skipped": skipped,
            "failed": 1,
            "registry": registry_entries,
            "error": f"missing required files: {clip_dir}",
        }

    pressure_npz = None
    pressure_npz_error = None
    caps = {}
    try:
        pressure_records = load_jsonl_by_frame(clip_dir / "jq_pressure.json")
        pressure_npz_path = clip_dir / "pressure_grids.npz"
        if pressure_npz_path.exists():
            try:
                pressure_npz = np.load(pressure_npz_path)
            except Exception as exc:
                pressure_npz_error = str(exc)
                if not allow_bad_npz:
                    raise
        elif not allow_bad_npz:
            raise FileNotFoundError(f"missing pressure_grids.npz: {pressure_npz_path}")

        video_counts = read_video_counts(clip_dir)
        frame_count_candidates = [c for c in video_counts.values() if c > 0]
        frame_count_candidates.append(len(pressure_records))
        if pressure_npz is not None:
            for key in ("left_pressure_grid", "right_pressure_grid"):
                if key in pressure_npz:
                    frame_count_candidates.append(pressure_npz[key].shape[0])
        frame_count = min(frame_count_candidates) if frame_count_candidates else 0
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)

        existing_frames = set(existing_frames or [])
        force_rewrite_frames = set(force_rewrite_frames or [])
        target_mode = target_frames is not None
        target_frames = set(int(x) for x in (target_frames or []))
        if target_mode and target_frames:
            frame_count = min(frame_count, max(target_frames) + 1)

        if not target_mode and trust_registry and frame_count > 0 and len(existing_frames) >= frame_count:
            if all(frame_idx in existing_frames for frame_idx in range(frame_count)) and not force_rewrite_frames:
                return {
                    "written": written,
                    "skipped": frame_count,
                    "failed": failed,
                    "registry": registry_entries,
                    "error": None,
                }

        caps = {
            view: cv2.VideoCapture(str(clip_dir / f"{view}.mp4"))
            for view in ("chest", "left", "right")
        }
        if any(not cap.isOpened() for cap in caps.values()):
            return {
                "written": written,
                "skipped": skipped,
                "failed": 1,
                "registry": registry_entries,
                "error": f"unreadable video: {clip_dir}",
            }

        for frame_idx in range(frame_count):
            frame_needs_write = (
                frame_idx in target_frames
                if target_mode
                else (not trust_registry or frame_idx not in existing_frames or frame_idx in force_rewrite_frames)
            )

            if not frame_needs_write:
                if not target_mode:
                    skipped += 1
                for cap in caps.values():
                    cap.grab()
                continue

            folder_name = "__".join(
                sanitize_component(x)
                for x in [clip["scene"], clip["task"], clip["clip_id"], f"{frame_idx:06d}"]
            )
            sample_dir = output_dir / clip["split"] / folder_name
            meta_path = sample_dir / "meta.json"

            if trust_registry and frame_idx in existing_frames and frame_idx not in force_rewrite_frames:
                skipped += 1
                for cap in caps.values():
                    cap.grab()
                continue

            if not target_mode and frame_idx not in force_rewrite_frames and sample_is_complete(sample_dir):
                skipped += 1
                for cap in caps.values():
                    cap.grab()
                continue
            if meta_path.exists():
                try:
                    meta_path.unlink()
                except OSError:
                    pass

            sample_dir.mkdir(parents=True, exist_ok=True)
            ok = write_frame_images(caps, sample_dir)
            if not ok:
                failed += 1
                break

            meta = make_frame_meta(
                clip,
                frame_idx,
                clip_bboxes.get(str(frame_idx)),
                pressure_records,
                pressure_npz,
                pressure_npz_error,
            )
            write_json_atomic(meta_path, np_to_json(meta), indent=meta_indent)

            registry_entries.append(
                {
                    "split": clip["split"],
                    "scene": clip["scene"],
                    "task": clip["task"],
                    "clip": clip["clip_id"],
                    "frame_idx": frame_idx,
                    "sample_dir": str(sample_dir),
                    "has_left_bbox": meta["hands"]["left"]["bbox_chest"] is not None,
                    "has_right_bbox": meta["hands"]["right"]["bbox_chest"] is not None,
                    "pressure_npz_available": meta["pressure_npz_available"],
                    "pressure_npz_error": meta["pressure_npz_error"],
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
            "error": f"{clip_dir}: {exc}",
        }
    finally:
        for cap in caps.values():
            cap.release()
        if pressure_npz is not None:
            pressure_npz.close()

    return {
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "registry": registry_entries,
        "error": None,
    }


def extract_touchanything_to_disk(
    clips,
    bbox_json_path,
    output_dir,
    max_frames=None,
    meta_indent=None,
    extract_workers=1,
    allow_bad_npz=False,
    trust_registry=True,
    check_workers=32,
    prefilter_workers=32,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with Path(bbox_json_path).open("r", encoding="utf-8") as f:
        all_bboxes = json.load(f)

    registry_path = output_dir / "touchanything_frames_registry.json"
    registry = []
    existing = set()
    force_rewrite = set()
    registry_changed = False
    if registry_path.exists():
        try:
            print(f"Loading TouchAnything registry: {registry_path}")
            with registry_path.open("r", encoding="utf-8") as f:
                registry = json.load(f)
            print(f"Loaded {len(registry)} TouchAnything registry entries.")
            if not allow_bad_npz:
                kept_registry = []
                for r in registry:
                    if r.get("pressure_npz_available") is False:
                        force_rewrite.add(registry_key(r))
                    else:
                        kept_registry.append(r)
                registry_changed = len(kept_registry) != len(registry)
                registry = kept_registry
            if trust_registry:
                print("Using TouchAnything registry as fast skip cache.")
                existing = {registry_key(r) for r in registry}
            else:
                existing = validate_touchanything_registry(registry, allow_bad_npz, check_workers)
                trust_registry = True
        except Exception:
            registry = []
            existing = set()
            force_rewrite = set()
    registry_original_len = len(registry)

    existing_by_clip = defaultdict(set)
    if existing:
        print("Indexing TouchAnything existing frames by clip...")
    for key in existing:
        split, scene, task, clip_id, frame_idx = key.rsplit("/", 4)
        existing_by_clip[f"{split}/{scene}/{task}/{clip_id}"].add(int(frame_idx))
    force_rewrite_by_clip = defaultdict(set)
    for key in force_rewrite:
        split, scene, task, clip_id, frame_idx = key.rsplit("/", 4)
        force_rewrite_by_clip[f"{split}/{scene}/{task}/{clip_id}"].add(int(frame_idx))

    print(f"Prefiltering TouchAnything clips for missing frames ({len(clips)} clips)...")
    prefilter_workers = max(1, int(prefilter_workers))
    prefilter_tasks = []
    for clip in clips:
        bbox_key = f"{clip['split']}/{clip['scene']}/{clip['task']}/{clip['clip_id']}"
        prefilter_tasks.append(
            (
                clip,
                existing_by_clip.get(bbox_key, set()),
                force_rewrite_by_clip.get(bbox_key, set()),
            )
        )

    written = 0
    skipped = 0
    failed = 0
    errors = []

    if prefilter_workers == 1:
        results_iter = (
            expected_frames_for_clip(
                clip,
                existing_frames,
                force_rewrite_frames,
                allow_bad_npz,
                max_frames,
            )
            for clip, existing_frames, force_rewrite_frames in prefilter_tasks
        )
        iterator = tqdm(results_iter, total=len(prefilter_tasks), desc="Prefiltering TouchAnything")
        prefilter_results = list(iterator)
    else:
        prefilter_results = []
        with ThreadPoolExecutor(max_workers=prefilter_workers) as executor:
            futures = [
                executor.submit(
                    expected_frames_for_clip,
                    clip,
                    existing_frames,
                    force_rewrite_frames,
                    allow_bad_npz,
                    max_frames,
                )
                for clip, existing_frames, force_rewrite_frames in prefilter_tasks
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Prefiltering TouchAnything"):
                prefilter_results.append(future.result())

    tasks = []
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
        clip = result["clip"]
        bbox_key = f"{clip['split']}/{clip['scene']}/{clip['task']}/{clip['clip_id']}"
        tasks.append(
            (
                clip,
                all_bboxes.get(bbox_key, {}),
                missing,
                str(output_dir),
                max_frames,
                meta_indent,
                allow_bad_npz,
                existing_by_clip.get(bbox_key, set()),
                force_rewrite_by_clip.get(bbox_key, set()),
                trust_registry,
            )
        )

    print(f"Submitting {len(tasks)} TouchAnything clips with missing frames to extraction workers.")

    extract_workers = max(1, int(extract_workers))
    if extract_workers == 1:
        results_iter = (extract_clip_to_disk_worker(task) for task in tasks)
        iterator = tqdm(results_iter, total=len(tasks), desc="Extracting TouchAnything frames")
        for result in iterator:
            written, skipped, failed = merge_extract_result(
                result, registry, existing, written, skipped, failed, errors
            )
    else:
        with ProcessPoolExecutor(max_workers=extract_workers) as executor:
            futures = [executor.submit(extract_clip_to_disk_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting TouchAnything frames"):
                result = future.result()
                written, skipped, failed = merge_extract_result(
                    result, registry, existing, written, skipped, failed, errors
                )

    if registry_changed or len(registry) != registry_original_len:
        with registry_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)
    else:
        print("Registry unchanged; skipped rewriting registry JSON.")

    print("TouchAnything frame extraction finished.")
    print(f"  new frames: {written}")
    print(f"  skipped existing frames: {skipped}")
    print(f"  failed/skipped clips or frames: {failed}")
    print(f"  output_dir: {output_dir}")
    print(f"  registry: {registry_path}")
    if errors:
        print("  first errors:")
        for err in errors[:10]:
            print(f"    {err}")


def merge_extract_result(result, registry, existing, written, skipped, failed, errors):
    written += result.get("written", 0)
    skipped += result.get("skipped", 0)
    failed += result.get("failed", 0)
    if result.get("error") and len(errors) < 10:
        errors.append(result["error"])

    for item in result.get("registry", []):
        reg_key = f"{item['split']}/{item['scene']}/{item['task']}/{item['clip']}/{item['frame_idx']}"
        if reg_key not in existing:
            registry.append(item)
            existing.add(reg_key)
    return written, skipped, failed


def registry_key(item):
    return f"{item['split']}/{item['scene']}/{item['task']}/{item['clip']}/{item['frame_idx']}"


def validate_touchanything_registry(registry, allow_bad_npz, check_workers):
    def check_item(item):
        if not allow_bad_npz and item.get("pressure_npz_available") is False:
            return None
        sample_dir = item.get("sample_dir")
        if not sample_dir:
            return None
        return registry_key(item) if sample_is_complete(sample_dir) else None

    check_workers = max(1, int(check_workers))
    if check_workers == 1:
        results = (check_item(item) for item in registry)
        iterator = tqdm(results, total=len(registry), desc="Checking TouchAnything registry")
        return {key for key in iterator if key is not None}

    existing = set()
    with ThreadPoolExecutor(max_workers=check_workers) as executor:
        for key in tqdm(
            executor.map(check_item, registry),
            total=len(registry),
            desc="Checking TouchAnything registry",
        ):
            if key is not None:
                existing.add(key)
    return existing


def expected_frames_for_clip(clip, existing_frames, force_rewrite_frames, allow_bad_npz, max_frames):
    clip_dir = Path(clip["raw_clip_dir"])
    pressure_npz = None
    try:
        required = [
            clip_dir / "chest.mp4",
            clip_dir / "left.mp4",
            clip_dir / "right.mp4",
            clip_dir / "jq_pressure.json",
        ]
        if not allow_bad_npz:
            required.append(clip_dir / "pressure_grids.npz")
        missing_required = [str(p) for p in required if not p.exists()]
        if missing_required:
            return {
                "clip": clip,
                "missing": [],
                "total_expected": 0,
                "skipped_existing": 0,
                "error": f"missing required files: {clip_dir}",
            }

        frame_count_candidates = []
        video_counts = read_video_counts(clip_dir)
        frame_count_candidates.extend(c for c in video_counts.values() if c > 0)
        frame_count_candidates.append(count_jsonl_records(clip_dir / "jq_pressure.json"))

        pressure_npz_path = clip_dir / "pressure_grids.npz"
        if pressure_npz_path.exists():
            try:
                pressure_npz = np.load(pressure_npz_path)
                for key in ("left_pressure_grid", "right_pressure_grid"):
                    if key in pressure_npz:
                        frame_count_candidates.append(pressure_npz[key].shape[0])
            except Exception as exc:
                if not allow_bad_npz:
                    raise exc
        elif not allow_bad_npz:
            raise FileNotFoundError(f"missing pressure_grids.npz: {pressure_npz_path}")

        frame_count_candidates = [int(c) for c in frame_count_candidates if int(c) > 0]
        frame_count = min(frame_count_candidates) if frame_count_candidates else 0
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)

        existing_frames = set(int(x) for x in (existing_frames or []))
        force_rewrite_frames = set(int(x) for x in (force_rewrite_frames or []))
        expected = list(range(frame_count))
        missing = [
            frame_idx
            for frame_idx in expected
            if frame_idx not in existing_frames or frame_idx in force_rewrite_frames
        ]
        return {
            "clip": clip,
            "missing": missing,
            "total_expected": len(expected),
            "skipped_existing": len(expected) - len(missing),
            "error": None,
        }
    except Exception as exc:
        return {
            "clip": clip,
            "missing": [],
            "total_expected": 0,
            "skipped_existing": 0,
            "error": f"{clip_dir}: {exc}",
        }
    finally:
        if pressure_npz is not None:
            pressure_npz.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract TouchAnything raw clips into per-frame folders with chest bboxes and dual-hand pressure."
    )
    parser.add_argument("--gpu", type=str, default="0", help="Visible GPU ids, e.g. 0 or 0,1,2,3")
    parser.add_argument(
        "--raw_root",
        type=str,
        default=None,
        help="Raw TouchAnything/EgoTouch root. Defaults to the directory containing --split_json.",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="TouchAnything split.json. Entries may point to HDF5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root for extracted per-frame folders. Defaults to <raw_root>/extracted_frames.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.join(base_dir, "preprocess/artifacts/touchanything/bboxes_cache"),
        help="Per-clip bbox cache directory.",
    )
    parser.add_argument(
        "--bbox_json",
        type=str,
        default=os.path.join(base_dir, "preprocess/artifacts/touchanything/touchanything_all_bboxes.json"),
        help="Merged chest bbox JSON path.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test_seen,test_unseen",
        help="Comma-separated split names to extract.",
    )
    parser.add_argument("--person_score_thresh", type=float, default=0.5)
    parser.add_argument("--hand_conf_thresh", type=float, default=0.5)
    parser.add_argument("--max_clips", type=int, default=None, help="Debug limit after split mapping.")
    parser.add_argument("--max_frames", type=int, default=None, help="Debug limit per clip.")
    parser.add_argument(
        "--meta_indent",
        type=int,
        default=None,
        help="Indent meta.json. Default writes compact JSON for speed and smaller files.",
    )
    parser.add_argument(
        "--extract_workers",
        type=int,
        default=1,
        help="CPU worker processes for frame decoding/JPEG/meta writing. Increase for faster extraction.",
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
        help="Threads for prefiltering clips to find missing frames before extraction.",
    )
    parser.add_argument("--skip_bbox", action="store_true", help="Use an existing --bbox_json and only write frames.")
    parser.add_argument("--bbox_only", action="store_true", help="Only extract/merge chest bboxes.")
    parser.add_argument(
        "--allow_bad_npz",
        action="store_true",
        help=(
            "Continue extracting RGB/bboxes/raw jq_pressure when pressure_grids.npz is missing "
            "or corrupted. Normalized grid and Gaussian pressure will be null in those meta.json files."
        ),
    )
    parser.add_argument(
        "--no_trust_registry",
        action="store_true",
        help="Do not use the registry as a fast skip cache; re-check sample folders/meta.json on disk.",
    )
    args = parser.parse_args()

    os.chdir(base_dir)

    if args.split_json is None:
        candidates = [
            Path.cwd() / "split.json",
            Path("/data1/jiangrui/EgoTouch/split.json"),
            Path("/home/ma-user/work/cfzhao/EgoTouch/split.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                args.split_json = str(candidate)
                break
        if args.split_json is None:
            raise FileNotFoundError("Please pass --split_json; no default split.json was found.")

    if args.raw_root is None:
        args.raw_root = str(Path(args.split_json).resolve().parent)

    if args.output_dir is None:
        args.output_dir = str(Path(args.raw_root) / "extracted_frames")

    requested_splits = args.splits.split(",") if args.splits else None
    clips, missing, raw_clip_count = load_split_clips(
        args.split_json,
        args.raw_root,
        requested_splits=requested_splits,
        max_clips=args.max_clips,
    )
    print(f"Resolved {len(clips)} TouchAnything clips from {args.split_json}")
    print(f"Indexed {raw_clip_count} raw clips under {args.raw_root}")
    if missing:
        print(f"Warning: {len(missing)} split entries could not be mapped. First 5:")
        for msg in missing[:5]:
            print(f"  {msg}")
    if not clips:
        raise RuntimeError(
            "No clips resolved from split.json. Check that --raw_root points to the raw "
            "TouchAnything/EgoTouch directory containing clip folders with chest.mp4, "
            "left.mp4, right.mp4, jq_pressure.json, and pressure_grids.npz."
        )

    gpu_list = [g.strip() for g in args.gpu.split(",") if g.strip()] or ["0"]
    logical_gpus = list(range(len(gpu_list)))

    bbox_path = args.bbox_json
    if not args.skip_bbox:
        bbox_path = extract_touchanything_bboxes_multigpu(
            logical_gpus,
            clips,
            args.cache_dir,
            args.bbox_json,
            args.person_score_thresh,
            args.hand_conf_thresh,
            max_frames=args.max_frames,
        )

    if args.bbox_only:
        return
    if not bbox_path or not Path(bbox_path).exists():
        raise FileNotFoundError(f"Bbox JSON not found: {bbox_path}")

    extract_touchanything_to_disk(
        clips,
        bbox_path,
        args.output_dir,
        max_frames=args.max_frames,
        meta_indent=args.meta_indent,
        extract_workers=args.extract_workers,
        allow_bad_npz=args.allow_bad_npz,
        trust_registry=not args.no_trust_registry,
        check_workers=args.check_workers,
        prefilter_workers=args.prefilter_workers,
    )


if __name__ == "__main__":
    main()
