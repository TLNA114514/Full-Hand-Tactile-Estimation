#!/usr/bin/env python
import os
import sys
import json
import argparse
import h5py
import cv2
import numpy as np
import torch
import glob
from pathlib import Path
from tqdm import tqdm

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))
sys.path.append(os.path.join(workspace_dir, "evaluation"))

from vitpose_model import ViTPoseModel

def parse_args():
    parser = argparse.ArgumentParser(description="Extract hand bounding boxes from OpenTouch dataset (Multi-GPU)")
    parser.add_argument("--split_json", type=str, default=os.path.join(workspace_dir, "evaluation/opentouch_splits.json"), help="Path to splits JSON")
    parser.add_argument("--data_dir", type=str, default=os.path.join(workspace_dir, "opentouch/data"), help="Path to OpenTouch HDF5 datasets")
    parser.add_argument("--output_json", type=str, default=os.path.join(ft_dir, "opentouch_train_val_bboxes.json"), help="Output JSON path")
    parser.add_argument("--gpu", type=str, default="4", help="GPU index to use")
    
    # Multi-GPU parameters
    parser.add_argument("--gpu_idx", type=int, default=0, help="Index of current GPU in parallel extraction")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs in parallel extraction")
    parser.add_argument("--merge", action="store_true", help="Merge all temporary GPU JSON files into the final JSON and exit")
    
    parser.add_argument("--sample_only", action="store_true", help="Only run on 1 sample clip to test script functionality")
    return parser.parse_args()

def merge_gpu_files(output_json_path):
    print("\n>>> Merging all GPU temporary JSON files...")
    final_data = {}
    
    # 1. Load existing main JSON if it exists
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                final_data = json.load(f)
            print(f"Loaded existing base JSON with {len(final_data)} clips.")
        except Exception as e:
            print(f"Warning: Failed to load existing output file: {e}. Starting fresh merge.")

    # 2. Search for temporary GPU files
    gpu_files = sorted(glob.glob(f"{output_json_path}.gpu_*"))
    if not gpu_files:
        print("No temporary GPU JSON files found to merge.")
        return

    print(f"Found {len(gpu_files)} temporary GPU files to merge: {gpu_files}")
    
    # 3. Merge data from all GPU files
    merged_count = 0
    for gpu_file in gpu_files:
        try:
            with open(gpu_file, "r", encoding="utf-8") as f:
                gpu_data = json.load(f)
            for k, v in gpu_data.items():
                if k not in final_data or len(gpu_data[k]) > len(final_data.get(k, {})):
                    final_data[k] = v
                    merged_count += 1
        except Exception as e:
            print(f"Error loading {gpu_file}: {e}")

    # 4. Save merged final JSON
    with open(output_json_path, "w", encoding="utf-8") as out_f:
        json.dump(final_data, out_f, indent=2, ensure_ascii=False)
    print(f"Successfully merged {merged_count} clips into: {output_json_path}")
    
    # 5. Clean up temporary files
    print("Cleaning up temporary GPU files...")
    for gpu_file in gpu_files:
        try:
            os.remove(gpu_file)
        except Exception as e:
            print(f"Failed to remove temporary file {gpu_file}: {e}")
    print("Cleanup completed.")

def main():
    args = parse_args()
    
    # If merge flag is set, merge files and exit
    if args.merge:
        merge_gpu_files(args.output_json)
        return
        
    # Set CWD to hamer root for relative paths within hamer package
    hamer_root = os.path.join(workspace_dir, "hamer")
    os.chdir(hamer_root)
    
    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[GPU {args.gpu_idx}/{args.num_gpus}] Using GPU: cuda:{args.gpu} for bounding box extraction")
    
    # Initialize Detectron2 detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    
    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    
    # Initialize ViTPose model
    cpm = ViTPoseModel(device)
    
    # Load splits
    if not os.path.exists(args.split_json):
        raise FileNotFoundError(f"Splits JSON not found at: {args.split_json}")
    with open(args.split_json, "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    # Gather target clips (train & val)
    target_clips = []
    for split_name in ["train", "val"]:
        target_clips.extend(splits.get(split_name, []))
        
    if args.sample_only:
        print("Running in sample mode. Only extracting bboxes for 1 clip.")
        target_clips = target_clips[:1]
        args.num_gpus = 1
        args.gpu_idx = 0
        
    print(f"Total dataset clips: {len(target_clips)}")
    
    # Load existing GPU-specific cache to support incremental resume
    gpu_output_json = f"{args.output_json}.gpu_{args.gpu_idx}"
    bbox_cache = {}
    if os.path.exists(gpu_output_json):
        try:
            with open(gpu_output_json, "r", encoding="utf-8") as f:
                bbox_cache = json.load(f)
            print(f"[GPU {args.gpu_idx}] Loaded local cache. {len(bbox_cache)} clips already processed by this GPU.")
        except Exception as e:
            print(f"Warning: Failed to load local cache file {gpu_output_json}: {e}. Starting fresh.")
            
    # Also read master JSON to see what clips are completed globally
    completed_globally = set()
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, "r", encoding="utf-8") as f:
                master_data = json.load(f)
                completed_globally = set(master_data.keys())
            print(f"[GPU {args.gpu_idx}] Found {len(completed_globally)} globally completed clips.")
        except:
            pass
            
    # Filter out clips already processed globally or locally on this GPU
    remaining_clips = []
    for scene, clip in target_clips:
        clip_key = f"{scene}/{clip}"
        if clip_key in completed_globally or clip_key in bbox_cache:
            continue
        remaining_clips.append((scene, clip))
        
    # Distribute remaining clips using modulo partition
    my_clips = [clip for idx, clip in enumerate(remaining_clips) if idx % args.num_gpus == args.gpu_idx]
    
    print(f"[GPU {args.gpu_idx}/{args.num_gpus}] Remaining clips in dataset: {len(remaining_clips)} | Clips assigned to this GPU: {len(my_clips)}")
    
    if len(my_clips) == 0:
        print(f"[GPU {args.gpu_idx}] No clips left to process. Exiting.")
        # Ensure we write a dummy JSON if not exist to mark completion
        if not os.path.exists(gpu_output_json):
            with open(gpu_output_json, "w", encoding="utf-8") as out_f:
                json.dump({}, out_f)
        return
        
    # Group assigned clips by scene to optimize HDF5 open operations
    from collections import defaultdict
    scene_to_clips = defaultdict(list)
    for scene, clip in my_clips:
        scene_to_clips[scene].append(clip)
        
    for scene_name, clip_ids in scene_to_clips.items():
        hdf5_path = os.path.join(args.data_dir, f"{scene_name}.hdf5")
        if not os.path.exists(hdf5_path):
            print(f"Warning: HDF5 file {hdf5_path} does not exist. Skipping.")
            continue
            
        print(f"\n[GPU {args.gpu_idx}] Processing scene: {scene_name} | Assigned Clips: {len(clip_ids)}")
        with h5py.File(hdf5_path, "r") as f:
            for clip_id in clip_ids:
                clip_key = f"{scene_name}/{clip_id}"
                if clip_id not in f["data"]:
                    print(f"Warning: Clip {clip_id} not found in {scene_name}.hdf5. Skipping.")
                    continue
                    
                clip_group = f[f"data/{clip_id}"]
                rgb_bytes_seq = clip_group["rgb_images_jpeg"][()]
                
                gt_right = "right_hand_landmarks" in clip_group
                gt_left = "left_hand_landmarks" in clip_group
                
                num_frames = len(rgb_bytes_seq)
                clip_bboxes = {}
                
                print(f"[GPU {args.gpu_idx}] Extracting bboxes for {clip_key} ({num_frames} frames)...")
                for i in tqdm(range(num_frames), desc=f"GPU {args.gpu_idx} Running detector"):
                    img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes_seq[i], dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue
                    img_rgb = img_bgr[:, :, ::-1]
                    
                    # Human detection
                    det_out = detector(img_bgr)
                    det_instances = det_out["instances"]
                    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
                    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                    pred_scores = det_instances.scores[valid_idx].cpu().numpy()
                    
                    if len(pred_bboxes) == 0:
                        continue
                        
                    # Hand pose keypoints
                    vitposes_out = cpm.predict_pose(
                        img_rgb,
                        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
                    )
                    
                    frame_bboxes = []
                    for vitposes in vitposes_out:
                        left_hand_keyp = vitposes["keypoints"][-42:-21]
                        right_hand_keyp = vitposes["keypoints"][-21:]
                        
                        if gt_left:
                            valid = left_hand_keyp[:, 2] > 0.5
                            if sum(valid) > 3:
                                bbox = [
                                    float(left_hand_keyp[valid, 0].min()),
                                    float(left_hand_keyp[valid, 1].min()),
                                    float(left_hand_keyp[valid, 0].max()),
                                    float(left_hand_keyp[valid, 1].max())
                                ]
                                frame_bboxes.append({"bbox": bbox, "is_right": 0})
                                
                        if gt_right:
                            valid = right_hand_keyp[:, 2] > 0.5
                            if sum(valid) > 3:
                                bbox = [
                                    float(right_hand_keyp[valid, 0].min()),
                                    float(right_hand_keyp[valid, 1].min()),
                                    float(right_hand_keyp[valid, 0].max()),
                                    float(right_hand_keyp[valid, 1].max())
                                ]
                                frame_bboxes.append({"bbox": bbox, "is_right": 1})
                                
                    if frame_bboxes:
                        clip_bboxes[str(i)] = frame_bboxes
                        
                bbox_cache[clip_key] = clip_bboxes
                
                # Save progress incrementally to GPU-specific file
                with open(gpu_output_json, "w", encoding="utf-8") as out_f:
                    json.dump(bbox_cache, out_f, indent=2, ensure_ascii=False)
                    
    print(f"\n[GPU {args.gpu_idx}] Successfully completed assigned extraction and saved local cache to: {gpu_output_json}")

if __name__ == "__main__":
    main()
