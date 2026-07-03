#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
import torchvision
import supervision as sv

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

import multiprocessing as mp


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _mkdir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _is_image_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in IMG_EXTS


def _safe_imread(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return None
    return img


def _parse_classes(s: str) -> List[str]:
    if s is None:
        return []
    s = str(s).strip()
    if s == "":
        return []
    items = [x.strip() for x in s.split(",") if x.strip()]
    if not items:
        return []
    return items


def _infer_classes_from_rgb_path(rgb_path: str) -> List[str]:
    folder = os.path.basename(os.path.dirname(os.path.abspath(rgb_path)))
    parts = [p.strip() for p in folder.split("-")]
    obj = ""
    if len(parts) > 1:
        obj = parts[1]
    if not obj:
        # Robust fallback: still return a valid pair.
        obj = "object"
    return ["hand", obj]


def _put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def _color_for_class(cid0: int) -> Tuple[int, int, int]:
    # stable vivid colors via HSV hue sampling
    golden_ratio = 0.618033988749895
    h = (cid0 * golden_ratio) % 1.0
    s = 0.90
    v = 0.95
    hsv = np.uint8([[[int(h * 179), int(s * 255), int(v * 255)]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def _parse_gpu_ids(s: str) -> List[int]:
    """
    Accept:
      "0,1,3" or "0 1 3" or "0, 1, 3"
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    parts = []
    for token in s.replace(",", " ").split():
        token = token.strip()
        if token == "":
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid gpu id: {token}. Example: --gpu_ids 0,1,3")
        parts.append(int(token))
    # de-dup but keep order
    seen = set()
    out = []
    for x in parts:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


@dataclass
class InferenceConfig:
    device: str
    classes: List[str]
    box_threshold: float
    text_threshold: float
    nms_threshold: float
    max_det: int
    mask_alpha: float


class GroundedSAMPipeline:
    def __init__(
        self,
        grounding_dino_config_path: str,
        grounding_dino_checkpoint_path: str,
        sam_encoder_version: str,
        sam_checkpoint_path: str,
        device: Optional[str] = None,
    ):
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        for p in [grounding_dino_config_path, grounding_dino_checkpoint_path, sam_checkpoint_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"File not found: {p}")

        # GroundingDINO wrapper (kept as in your original logic)
        self.grounding_dino_model = Model(
            model_config_path=grounding_dino_config_path,
            model_checkpoint_path=grounding_dino_checkpoint_path
        )

        if sam_encoder_version not in sam_model_registry:
            raise ValueError(
                f"Unknown SAM encoder version: {sam_encoder_version}. "
                f"Available: {list(sam_model_registry.keys())}"
            )
        sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    @torch.no_grad()
    def _sam_segment_boxes(self, image_bgr: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        if xyxy is None or len(xyxy) == 0:
            return np.zeros((0, image_bgr.shape[0], image_bgr.shape[1]), dtype=bool)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)

        boxes_t = torch.as_tensor(xyxy, dtype=torch.float32, device=self.device)
        boxes_t = self.sam_predictor.transform.apply_boxes_torch(boxes_t, image_rgb.shape[:2])

        masks, scores, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes_t,
            multimask_output=True
        )
        # masks: [N, 3, H, W], scores: [N, 3]
        best_idx = torch.argmax(scores, dim=1)  # [N]
        n = masks.shape[0]
        best_masks = masks[torch.arange(n, device=self.device), best_idx]  # [N, H, W]
        return best_masks.detach().cpu().numpy().astype(bool)

    def process_image(self, image_bgr: np.ndarray, cfg: InferenceConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
        H, W = image_bgr.shape[:2]
        meta = {"num_det_before_nms": 0, "num_det_after_nms": 0}

        detections = self.grounding_dino_model.predict_with_classes(
            image_bgr, cfg.classes, cfg.box_threshold, cfg.text_threshold
        )

        label_dtype = np.uint8 if len(cfg.classes) <= 255 else np.uint16

        if detections is None or len(detections) == 0 or detections.xyxy is None or len(detections.xyxy) == 0:
            label_mask = np.zeros((H, W), dtype=label_dtype)
            vis = image_bgr.copy()
            _put_text(vis, "No detections", (10, 30))
            return vis, label_mask, meta

        meta["num_det_before_nms"] = int(len(detections.xyxy))

        # NMS (batched by class)
        boxes = torch.from_numpy(detections.xyxy).float()
        scores = torch.from_numpy(detections.confidence).float()
        class_ids = torch.from_numpy(detections.class_id).long()

        keep = torchvision.ops.batched_nms(boxes, scores, class_ids, cfg.nms_threshold)
        keep = keep.cpu().numpy().tolist()
        if cfg.max_det > 0:
            keep = keep[: cfg.max_det]

        detections.xyxy = detections.xyxy[keep]
        detections.confidence = detections.confidence[keep]
        detections.class_id = detections.class_id[keep]

        meta["num_det_after_nms"] = int(len(detections.xyxy))

        masks_bool = self._sam_segment_boxes(image_bgr, detections.xyxy)  # [N,H,W]

        # Merge to label mask by confidence
        label_mask = np.zeros((H, W), dtype=label_dtype)
        score_map = np.zeros((H, W), dtype=np.float32)

        order = np.argsort(-detections.confidence)
        for idx in order:
            cid0 = int(detections.class_id[idx])
            conf = float(detections.confidence[idx])
            m = masks_bool[idx]
            if m is None or m.size == 0:
                continue
            update = m & (conf > score_map)
            if np.any(update):
                label_mask[update] = np.array(cid0 + 1, dtype=label_dtype)
                score_map[update] = conf

        # Visualization
        vis = image_bgr.copy()
        vis = np.zeros_like(vis, dtype=np.uint8)
        alpha = float(np.clip(cfg.mask_alpha, 0.0, 1.0))

        for idx in order:
            cid0 = int(detections.class_id[idx])
            conf = float(detections.confidence[idx])
            x1, y1, x2, y2 = detections.xyxy[idx].astype(int).tolist()
            cls_name = cfg.classes[cid0] if 0 <= cid0 < len(cfg.classes) else f"class_{cid0}"
            color = _color_for_class(cid0)

            m = masks_bool[idx]
            if m is not None and m.size > 0 and alpha > 0:
                overlay = np.zeros_like(vis, dtype=np.uint8)
                overlay[m] = color
                vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0.0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            _put_text(vis, f"{cls_name} {conf:.2f}", (x1, max(20, y1 - 5)), scale=0.6)

        _put_text(vis, f"det: {meta['num_det_after_nms']} (NMS)", (10, H - 10), scale=0.6)
        return vis, label_mask, meta


def _iter_rgb_images(root_dir: str, keyword: str = "_rgb", recursive: bool = True) -> List[str]:
    paths: List[str] = []
    if recursive:
        for r, _, files in os.walk(root_dir):
            for fn in files:
                if keyword in fn and _is_image_file(fn):
                    paths.append(os.path.join(r, fn))
    else:
        for fn in os.listdir(root_dir):
            p = os.path.join(root_dir, fn)
            if os.path.isfile(p) and keyword in fn and _is_image_file(p):
                paths.append(p)
    paths.sort()
    return paths


def _mask_path_from_rgb_path(rgb_path: str, src_key: str = "_rgb", dst_key: str = "_mask") -> str:
    replaced = rgb_path.replace(src_key, dst_key)
    stem, _ = os.path.splitext(replaced)
    return stem + ".png"


def _save_mask(mask_path: str, mask: np.ndarray) -> None:
    _mkdir(os.path.dirname(mask_path))
    ok = cv2.imwrite(mask_path, mask)
    if not ok:
        raise RuntimeError(f"Failed to write mask: {mask_path}")


def _save_vis(vis_path: str, vis_bgr: np.ndarray) -> None:
    _mkdir(os.path.dirname(vis_path))
    ok = cv2.imwrite(vis_path, vis_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write visualization: {vis_path}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("grounded_sam_folder_rgb")

    ap.add_argument("--input_dir", type=str, default="", help="Input folder to scan for '*_rgb*' images")
    ap.add_argument("--recursive", type=bool, default=True, help="Recursively traverse input_dir (default: False)")
    ap.add_argument("--keyword", type=str, default="_rgb", help="Keyword to filter images (default: _rgb)")
    
    ap.add_argument("--classes", type=str, default="hand, object", help="Comma-separated classes, e.g. 'hand,bottle'")
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--text_threshold", type=float, default=0.25)
    ap.add_argument("--nms_threshold", type=float, default=0.4)
    ap.add_argument("--max_det", type=int, default=3, help="0 means no limit")
    ap.add_argument("--mask_alpha", type=float, default=0.6)

    ap.add_argument("--device", type=str, default="cuda", help="auto/cuda/cpu (single-process fallback)")
    ap.add_argument("--gpu_ids", type=str, default="0",
                    help="Multi-GPU mode: comma/space separated GPU ids, e.g. '0,1,3'. "
                         "If provided, will spawn one process per GPU.")
    
    # Model paths (default values follow your current script; please adjust if needed)
    ap.add_argument("--grounding_dino_config", type=str, default="./Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    ap.add_argument("--grounding_dino_ckpt", type=str, default="./Grounded-Segment-Anything/groundingdino_swint_ogc.pth")
    ap.add_argument("--sam_encoder", type=str, default="vit_h", help="SAM encoder version: vit_h / vit_l / vit_b")
    ap.add_argument("--sam_ckpt", type=str, default="./Grounded-Segment-Anything/sam_vit_h_4b8939.pth")

    ap.add_argument("--save_vis", default=False, help="If set, save visualization results")
    ap.add_argument("--vis_dir", type=str, default="", help="Folder to save visualizations (separate folder)")

    return ap

def _process_subset(
    worker_rank: int,
    num_workers: int,
    gpu_id: Optional[int],
    rgb_paths: List[str],
    input_dir: str,
    args: argparse.Namespace,
):
    try:
        if gpu_id is not None:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # After this, this process sees exactly one GPU as cuda:0
            device_for_worker = "cuda" if torch.cuda.is_available() else "cpu"
            if device_for_worker == "cuda":
                torch.cuda.set_device(0)
        else:
            device_for_worker = args.device

        base_classes = _parse_classes(args.classes)

        pipeline = GroundedSAMPipeline(
            grounding_dino_config_path=args.grounding_dino_config,
            grounding_dino_checkpoint_path=args.grounding_dino_ckpt,
            sam_encoder_version=args.sam_encoder,
            sam_checkpoint_path=args.sam_ckpt,
            device=device_for_worker,
        )

        # Build a template config; classes may be overridden per-image
        cfg_template = dict(
            device=str(pipeline.device),
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            nms_threshold=float(args.nms_threshold),
            max_det=int(args.max_det),
            mask_alpha=float(args.mask_alpha),
        )

        vis_dir = ""
        if args.save_vis:
            vis_dir = args.vis_dir.strip()
            if not vis_dir:
                vis_dir = os.path.join(input_dir, "_grounded_sam_vis")
            vis_dir = os.path.abspath(vis_dir)
            _mkdir(vis_dir)

        total = len(rgb_paths)
        processed = 0
        skipped = 0
        t0 = time.time()

        for idx, rgb_path in enumerate(rgb_paths):
            if idx % num_workers != worker_rank:
                continue
            
            mask_path = _mask_path_from_rgb_path(rgb_path, src_key=args.keyword, dst_key="_mask")
            # If mask already exists and is readable, skip this image
            if os.path.exists(mask_path):
                existing = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if existing is not None and existing.size > 0:
                    # print(f"[W{worker_rank}] [SKIP] mask exists: {mask_path}")
                    pass
            else:
                img = _safe_imread(rgb_path)
                if img is None:
                    print(f"[W{worker_rank}] [WARN] Unreadable image, skip: {rgb_path}")
                    skipped += 1
                    continue

                # If --classes is an empty string, infer per-image classes from path.
                # Otherwise, keep the original fixed class list.
                classes = base_classes if base_classes else _infer_classes_from_rgb_path(rgb_path)
                cfg = InferenceConfig(classes=classes, **cfg_template)
                
                vis, mask, _meta = pipeline.process_image(img, cfg)
                _save_mask(mask_path, mask)

                if args.save_vis:
                    rel = os.path.relpath(rgb_path, input_dir)
                    rel_stem, _ = os.path.splitext(rel)
                    vis_path = os.path.join(vis_dir, rel_stem + "_vis.jpg")
                    _save_vis(vis_path, vis)

            processed += 1
            if processed % 50 == 0:
                ginfo = f"GPU{gpu_id}" if gpu_id is not None else str(device_for_worker)
                total_this_worker = (total - worker_rank + num_workers - 1) // num_workers
                print(f"[W{worker_rank}] [{ginfo}] processed={processed}/{total_this_worker} skipped={skipped}")

        t1 = time.time()
        ginfo = f"GPU{gpu_id}" if gpu_id is not None else str(device_for_worker)
        print(f"[W{worker_rank}] [{ginfo}] DONE processed={processed} skipped={skipped} time={t1 - t0:.2f}s")

    except Exception:
        ginfo = f"GPU{gpu_id}" if gpu_id is not None else "device"
        print(f"[W{worker_rank}] [{ginfo}] ERROR:\n{traceback.format_exc()}")


def main():
    args = build_argparser().parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise ValueError(f"--input_dir is not a folder: {input_dir}")

    rgb_paths = _iter_rgb_images(input_dir, keyword=args.keyword, recursive=bool(args.recursive))
    if not rgb_paths:
        raise RuntimeError(f"No images containing '{args.keyword}' found under: {input_dir}")

    gpu_ids = _parse_gpu_ids(args.gpu_ids)

    # Single-process fallback (keep behavior)
    if not gpu_ids:
        _process_subset(
            worker_rank=0,
            num_workers=1,
            gpu_id=None,
            rgb_paths=rgb_paths,
            input_dir=input_dir,
            args=args,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("You specified --gpu_ids but CUDA is not available in this environment.")

    # Multi-process: one process per GPU id
    num_workers = len(gpu_ids)
    print(f"[INFO] Multi-GPU enabled. gpu_ids={gpu_ids}, num_workers={num_workers}, total_images={len(rgb_paths)}")

    # Use spawn to avoid CUDA/fork issues
    ctx = mp.get_context("spawn")
    procs: List[mp.Process] = []

    for rank, gid in enumerate(gpu_ids):
        p = ctx.Process(
            target=_process_subset,
            args=(rank, num_workers, gid, rgb_paths, input_dir, args),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # If any worker exits non-zero, surface it
    bad = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    if bad:
        raise RuntimeError(f"Some worker processes failed. Exit codes: {[p.exitcode for p in procs]}")


if __name__ == "__main__":
    main()
