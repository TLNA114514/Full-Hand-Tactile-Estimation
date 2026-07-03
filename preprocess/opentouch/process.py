import sys
import os
import json
import argparse
import h5py
import cv2
import numpy as np
import torch
import math
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp

# ======================================================================
# 提前解析 --gpu 参数并设置 CUDA_VISIBLE_DEVICES
# ======================================================================
_gpus = ""
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        _gpus = sys.argv[i+1]
        break
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(base_dir, 'hamer'))
sys.path.append(os.path.join(base_dir, 'evaluation'))

def initialize_models(device):
    # 1. 初始化 ViTDet
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    
    local_vitdet_path = os.path.join(base_dir, "hamer/_DATA/model_final_f05665.pkl")
    if os.path.exists(local_vitdet_path):
        detectron2_cfg.train.init_checkpoint = local_vitdet_path
    else:
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    
    # 2. 动态注册 ViTPose
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
        print(f"⚠️ 动态注册 ViT 发生异常: {e}")
        
    from vitpose_model import ViTPoseModel
    cpm = ViTPoseModel(device)
    return detector, cpm


def extract_worker(gpu_id, clips_chunk, cache_dir, data_dir):
    """单卡子进程：负责对自己被分配到的 clips_chunk 提取 Bbox (含 null 回退) 并将结果刷入 cache_dir"""
    try:
        device = torch.device(f'cuda:{gpu_id}')
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            
        print(f"[Worker GPU {gpu_id}] 初始化模型... 分配了 {len(clips_chunk)} 个 clip。")
        detector, cpm = initialize_models(device)
        
        # 按照 split 和 scene 进行分组，减少切换 HDF5 的开销
        from collections import defaultdict
        scene_to_clips = defaultdict(list)
        for split, scene, clip_id in clips_chunk:
            scene_to_clips[(split, scene)].append(clip_id)
            
        for (split, scene), clips in scene_to_clips.items():
            hdf5_path = os.path.join(data_dir, f"{scene}.hdf5")
            if not os.path.exists(hdf5_path):
                hdf5_path = os.path.join(data_dir, f"{scene}.h5")
                if not os.path.exists(hdf5_path):
                    continue
                    
            with h5py.File(hdf5_path, "r") as f:
                for clip_id in clips:
                    # 断点重续核心逻辑：单个 clip 提取完直接写入 json
                    cache_path = os.path.join(cache_dir, f"{split}_{scene}_{clip_id}.json")
                    if os.path.exists(cache_path):
                        continue
                        
                    if f"data/{clip_id}" not in f:
                        continue
                        
                    clip_group = f[f"data/{clip_id}"]
                    rgb_bytes_seq = clip_group["rgb_images_jpeg"][()]
                    
                    # 检查手部数据是否存在，以此为依据决定是否在缺失 bbox 时填充 null
                    has_right = "right_pressure" in clip_group or "right_hand_landmarks" in clip_group
                    has_left = "left_pressure" in clip_group or "left_hand_landmarks" in clip_group
                    
                    num_frames = len(rgb_bytes_seq)
                    clip_frames_dict = {}
                    
                    for i in tqdm(range(num_frames), desc=f"[GPU {gpu_id}] {split}/{clip_id}", leave=False, position=gpu_id):
                        img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes_seq[i], dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_bgr is None: continue
                        img_rgb = img_bgr[:, :, ::-1]
                        
                        bboxes = []
                        found_right = False
                        found_left = False
                        
                        try:
                            det_out = detector(img_bgr)
                            det_instances = det_out['instances']
                            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
                            pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                            pred_scores = det_instances.scores[valid_idx].cpu().numpy()
                            
                            if len(pred_bboxes) > 0:
                                vitposes_out = cpm.predict_pose(img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])
                                for vitposes in vitposes_out:
                                    left_hand_keyp = vitposes['keypoints'][-42:-21]
                                    right_hand_keyp = vitposes['keypoints'][-21:]
                                    
                                    if has_left:
                                        valid = left_hand_keyp[:, 2] > 0.5
                                        if sum(valid) > 3:
                                            bbox = [float(left_hand_keyp[valid, 0].min()), float(left_hand_keyp[valid, 1].min()),
                                                    float(left_hand_keyp[valid, 0].max()), float(left_hand_keyp[valid, 1].max())]
                                            bboxes.append({"bbox": bbox, "is_right": 0})
                                            found_left = True
                                            
                                    if has_right:
                                        valid = right_hand_keyp[:, 2] > 0.5
                                        if sum(valid) > 3:
                                            bbox = [float(right_hand_keyp[valid, 0].min()), float(right_hand_keyp[valid, 1].min()),
                                                    float(right_hand_keyp[valid, 0].max()), float(right_hand_keyp[valid, 1].max())]
                                            bboxes.append({"bbox": bbox, "is_right": 1})
                                            found_right = True
                        except Exception:
                            pass
                            
                        # 如果确实物理存在该手的数据，但是没检测出 Bbox，强制填充 fallback: "null"
                        if has_right and not found_right:
                            bboxes.append({"bbox": "null", "is_right": 1})
                        if has_left and not found_left:
                            bboxes.append({"bbox": "null", "is_right": 0})
                            
                        if len(bboxes) > 0:
                            clip_frames_dict[str(i)] = bboxes
                            
                    with open(cache_path, 'w') as cf:
                        json.dump(clip_frames_dict, cf)
                        
    except Exception as e:
        import traceback
        print(f"[Worker GPU {gpu_id}] 发生严重错误: {e}")
        traceback.print_exc()


def extract_full_bboxes_multigpu(logical_gpus, data_dir, cache_dir=None, output_bbox_json=None):
    """主进程：分配 train/val/test 任务，等待结束，然后合并大 JSON"""
    splits_json_path = os.path.join(base_dir, "evaluation/opentouch_splits.json")
    if output_bbox_json is None:
        output_bbox_json = os.path.join(base_dir, "preprocess/artifacts/opentouch/opentouch_all_bboxes.json")
    
    if cache_dir is None:
        cache_dir = os.path.join(base_dir, "preprocess/artifacts/opentouch/full_bboxes_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    with open(splits_json_path, 'r') as f:
        splits = json.load(f)
        
    all_clips = []
    for split_name in ["train", "val", "test"]:
        for scene, clip_id in splits.get(split_name, []):
            all_clips.append((split_name, scene, clip_id))
            
    if not all_clips:
        print("未找到任何划分数据。")
        return None
        
    print(f"✅ 开始为全量集(train/val/test)提框/打底，总计包含 {len(all_clips)} 个 clip...")
    print(f"✅ 断点重续机制已激活，将在 {cache_dir} 记录进度。")
    
    num_gpus = len(logical_gpus)
    chunk_size = math.ceil(len(all_clips) / num_gpus)
    chunks = [all_clips[i:i + chunk_size] for i in range(0, len(all_clips), chunk_size)]
    
    pool_args = []
    for i, gpu_id in enumerate(logical_gpus):
        if i < len(chunks) and len(chunks[i]) > 0:
            pool_args.append((int(gpu_id), chunks[i], cache_dir, data_dir))
            
    if num_gpus > 1:
        print(f"🚀 启动 {len(pool_args)} 卡并行提框进程池！")
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        with mp.Pool(len(pool_args)) as pool:
            pool.starmap(extract_worker, pool_args)
    else:
        print(f"🚀 启动单进程提框模式！")
        extract_worker(*pool_args[0])
        
    samples_dict = {}
    total_valid_frames = 0
    print("🔄 所有 Worker 已完成，正在合并局部 JSON 结果...")
    
    for split, scene, clip_id in all_clips:
        cache_path = os.path.join(cache_dir, f"{split}_{scene}_{clip_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as cf:
                clip_frames_dict = json.load(cf)
                if clip_frames_dict:
                    samples_dict[f"{split}/{scene}/{clip_id}"] = clip_frames_dict
                    total_valid_frames += len(clip_frames_dict)
                    
    with open(output_bbox_json, 'w') as f:
        json.dump(samples_dict, f, indent=2)
        
    print(f"🎉 Bbox(含null兜底) 全量预提取完成！成功提取了 {total_valid_frames} 个有效帧。")
    print(f"大 JSON 已保存至: {output_bbox_json}")
    
    return output_bbox_json


def extract_full_to_disk(bbox_json_path, data_dir, output_dir, registry_path=None):
    """第二阶段：读取全量大 JSON，从 HDF5 中解包图像和 meta，并记录 Registry"""
    print("\n📦 开始将全量数据集图片及 meta.json 写入磁盘，并生成 Registry...")
    if registry_path is None:
        registry_path = os.path.join(base_dir, "preprocess/artifacts/opentouch/dataset_frames_registry.json")
    
    with open(bbox_json_path, 'r') as f:
        samples_dict = json.load(f)
        
    h5_files = {}
    count = 0
    skipped_count = 0
    registry_entries = []
    
    # 因为存在断点重试可能，尝试读取已存在的 registry 并更新
    if os.path.exists(registry_path):
        try:
            with open(registry_path, 'r') as f:
                registry_entries = json.load(f)
        except Exception:
            pass
    # 建立一个去重用的 set
    existing_registry = set(f"{r['split']}/{r['scene']}/{r['clip']}/{r['frame_idx']}/{r['is_right']}" for r in registry_entries)
    
    for split_scene_clip, frames in tqdm(samples_dict.items(), desc="Extracting to Disk"):
        parts = split_scene_clip.split("/")
        split_name = parts[0]
        scene = parts[1]
        clip_id = parts[2]
        
        h5_path = os.path.join(data_dir, f"{scene}.hdf5")
        if not os.path.exists(h5_path):
            h5_path = os.path.join(data_dir, f"{scene}.h5")
        if h5_path not in h5_files:
            if not os.path.exists(h5_path):
                continue
            h5_files[h5_path] = h5py.File(h5_path, 'r', swmr=True)
            
        f = h5_files[h5_path]
        if f"data/{clip_id}" not in f: continue
        clip_group = f[f"data/{clip_id}"]
        rgb_bytes_seq = clip_group["rgb_images_jpeg"]
        
        for frame_idx_str, boxes in frames.items():
            frame_idx = int(frame_idx_str)
            if frame_idx >= len(rgb_bytes_seq): continue
            
            raw_bytes = rgb_bytes_seq[frame_idx]
            byte_data = raw_bytes.tobytes() if isinstance(raw_bytes, np.ndarray) else raw_bytes
            
            for sample in boxes:
                is_right = sample.get("is_right", 1)
                bbox = sample["bbox"]
                
                sample_folder_name = f"{scene}_{clip_id}_{frame_idx:04d}_{is_right}"
                sample_dir = os.path.join(output_dir, split_name, sample_folder_name)
                os.makedirs(sample_dir, exist_ok=True)
                
                # 记录 Registry
                reg_key = f"{split_name}/{scene}/{clip_id}/{frame_idx}/{is_right}"
                if reg_key not in existing_registry:
                    registry_entries.append({
                        "split": split_name,
                        "scene": scene,
                        "clip": clip_id,
                        "frame_idx": frame_idx,
                        "is_right": is_right,
                        "has_bbox": bbox != "null",
                        "sample_dir": sample_dir
                    })
                    existing_registry.add(reg_key)
                
                meta_path = os.path.join(sample_dir, "meta.json")
                if os.path.exists(meta_path):
                    skipped_count += 1
                    continue
                
                image_path = os.path.join(sample_dir, "image.jpg")
                with open(image_path, "wb") as img_file:
                    img_file.write(byte_data)
                    
                # 处理 Landmarks
                if is_right == 1:
                    landmarks = clip_group["right_hand_landmarks"][frame_idx] if "right_hand_landmarks" in clip_group else np.full((21,3), np.nan)
                else:
                    landmarks = clip_group["left_hand_landmarks"][frame_idx] if "left_hand_landmarks" in clip_group else np.full((21,3), np.nan)
                        
                camera_pose = clip_group["camera_poses"][frame_idx]
                rgb_calib = f["calibration/rgb"]
                T_device_camera = rgb_calib["T_device_camera"][:]
                focal_length = rgb_calib["focal_length"][()]
                image_size = rgb_calib["image_size"][:]
                principal_point = rgb_calib["principal_point"][:]
                
                original_data = {}
                for key in clip_group.keys():
                    if key == "rgb_images_jpeg":
                        continue
                    val = clip_group[key]
                    if len(val.shape) > 0 and val.shape[0] > frame_idx:
                        try:
                            frame_data = val[frame_idx]
                            if isinstance(frame_data, np.ndarray):
                                original_data[key] = frame_data.tolist()
                            else:
                                original_data[key] = frame_data.item() if hasattr(frame_data, "item") else frame_data
                        except Exception:
                            pass
                            
                original_calibration = {}
                for k in rgb_calib.keys():
                    val = rgb_calib[k][...]
                    if isinstance(val, np.ndarray):
                        original_calibration[k] = val.tolist()
                    else:
                        original_calibration[k] = val.item() if hasattr(val, "item") else val
                        
                T_world_to_slam = np.linalg.inv(camera_pose)
                if not np.isnan(landmarks).any() and len(landmarks) == 21:
                    landmarks_homo = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
                    landmarks_device = (T_world_to_slam @ landmarks_homo.T).T[:, :3]
                    landmarks_device_homo = np.concatenate([landmarks_device, np.ones((landmarks_device.shape[0], 1))], axis=1)
                    landmarks_cam = (T_device_camera @ landmarks_device_homo.T).T[:, :3]
                    valid_mask = [True] * 21
                else:
                    landmarks_cam = np.zeros((21, 3))
                    valid_mask = [False] * 21
                    
                meta_data = {
                    "scene": scene,
                    "demo": clip_id,
                    "frame_idx": frame_idx,
                    "is_right": is_right,
                    "keypoints_3d_cam": landmarks_cam.tolist(),
                    "valid_mask": valid_mask,
                    "bbox": bbox, # 这里如果是 "null"，JSON 里也是 "null" 字符串；如果需要 None 可以写成 bbox if bbox != "null" else None，但兼容性考虑保留字符串或 null 都可以。这里尊重前面写入的。如果是字符串 "null"，我们就写入 "null"。 
                    "camera_intrinsics": {
                        "focal_length": float(focal_length),
                        "principal_point": principal_point.tolist(),
                        "image_size": image_size.tolist(),
                        "T_device_camera": T_device_camera.tolist(),
                        "camera_pose": camera_pose.tolist()
                    },
                    "original_hdf5_data": original_data,
                    "original_calibration": original_calibration
                }
                
                with open(meta_path, "w") as meta_file:
                    json.dump(meta_data, meta_file, indent=2)
                    
                count += 1
                
                # 每写入 5000 次，保存一次 registry，防止中断导致全损
                if count % 5000 == 0:
                    with open(registry_path, "w") as rf:
                        json.dump(registry_entries, rf, indent=2)

    for f in h5_files.values():
        f.close()
        
    # 最后统一保存 registry
    with open(registry_path, "w") as rf:
        json.dump(registry_entries, rf, indent=2)
        
    print(f"全量磁盘碎片化写入及 Registry 记录完成！")
    print(f"共全新提取: {count} 个样本。")
    print(f"跳过已存在样本: {skipped_count} 个。")
    print(f"Registry 保存在: {registry_path}")
    print(f"输出根目录: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract full dataset with multi-GPU and fallback null bboxes")
    parser.add_argument('--gpu', type=str, default='0', help='使用的 GPU 编号 (例: 0,1,2,3)')
    parser.add_argument('--data_dir', type=str, default='/data1/jiangrui/OpenTouch Data/data', help='原始 HDF5 数据目录')
    parser.add_argument('--output_dir', type=str, default='/data1/jiangrui/OpenTouch Data/full_dataset', help='输出的全量数据集目录')
    parser.add_argument('--cache_dir', type=str, default=None, help='BBox cache directory')
    parser.add_argument('--bbox_json', type=str, default=None, help='Merged bbox JSON path')
    parser.add_argument('--registry_json', type=str, default=None, help='Extracted frames registry JSON path')
    args = parser.parse_args()
    
    os.chdir(base_dir)
    
    gpu_list = args.gpu.split(',') if args.gpu else ['0']
    logical_gpus = list(range(len(gpu_list)))
    
    bbox_path = extract_full_bboxes_multigpu(
        logical_gpus,
        args.data_dir,
        cache_dir=args.cache_dir,
        output_bbox_json=args.bbox_json,
    )
    if bbox_path:
        extract_full_to_disk(
            bbox_path,
            args.data_dir,
            args.output_dir,
            registry_path=args.registry_json,
        )
