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

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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


def extract_worker(gpu_id, clips_chunk, cache_dir):
    """单卡子进程：负责对自己被分配到的 clips_chunk 提取 Bbox 并将结果刷入 cache_dir"""
    try:
        device = torch.device(f'cuda:{gpu_id}')
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            
        print(f"[Worker GPU {gpu_id}] 初始化模型... 分配了 {len(clips_chunk)} 个 clip。")
        detector, cpm = initialize_models(device)
        
        data_dir = "/data1/jiangrui/OpenTouch Data/data"
        
        # 按照 scene 进行分组，减少切换 HDF5 的开销
        from collections import defaultdict
        scene_to_clips = defaultdict(list)
        for scene, clip_id in clips_chunk:
            scene_to_clips[scene].append(clip_id)
            
        for scene, clips in scene_to_clips.items():
            hdf5_path = os.path.join(data_dir, f"{scene}.hdf5")
            if not os.path.exists(hdf5_path):
                hdf5_path = os.path.join(data_dir, f"{scene}.h5")
                if not os.path.exists(hdf5_path):
                    continue
                    
            with h5py.File(hdf5_path, "r") as f:
                for clip_id in clips:
                    # ==========================================
                    # 断点重续核心逻辑：单个 clip 提取完直接写入 json
                    # 启动时若检查到 json 已存在，直接秒跳过
                    # ==========================================
                    cache_path = os.path.join(cache_dir, f"{scene}_{clip_id}.json")
                    if os.path.exists(cache_path):
                        # 如果已存在，直接跳过当前 clip，不进行推理
                        continue
                        
                    if f"data/{clip_id}" not in f:
                        continue
                        
                    clip_group = f[f"data/{clip_id}"]
                    rgb_bytes_seq = clip_group["rgb_images_jpeg"][()]
                    
                    gt_right_landmarks = clip_group["right_hand_landmarks"][()] if "right_hand_landmarks" in clip_group else None
                    gt_left_landmarks = clip_group["left_hand_landmarks"][()] if "left_hand_landmarks" in clip_group else None
                    
                    num_frames = len(rgb_bytes_seq)
                    clip_frames_dict = {}
                    
                    # 留存进程号，使得不同 GPU 的 tqdm 尽量不要互相覆盖
                    for i in tqdm(range(num_frames), desc=f"[GPU {gpu_id}] {clip_id}", leave=False, position=gpu_id):
                        img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes_seq[i], dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_bgr is None: continue
                        img_rgb = img_bgr[:, :, ::-1]
                        
                        try:
                            det_out = detector(img_bgr)
                        except Exception:
                            continue
                            
                        det_instances = det_out['instances']
                        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
                        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
                        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
                        
                        if len(pred_bboxes) == 0: continue
                        
                        try:
                            vitposes_out = cpm.predict_pose(img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])
                        except Exception:
                            continue
                            
                        bboxes = []
                        for vitposes in vitposes_out:
                            left_hand_keyp = vitposes['keypoints'][-42:-21]
                            right_hand_keyp = vitposes['keypoints'][-21:]
                            
                            if gt_left_landmarks is not None:
                                valid = left_hand_keyp[:, 2] > 0.5
                                if sum(valid) > 3:
                                    bbox = [float(left_hand_keyp[valid, 0].min()), float(left_hand_keyp[valid, 1].min()),
                                            float(left_hand_keyp[valid, 0].max()), float(left_hand_keyp[valid, 1].max())]
                                    bboxes.append({"bbox": bbox, "is_right": 0})
                                    
                            if gt_right_landmarks is not None:
                                valid = right_hand_keyp[:, 2] > 0.5
                                if sum(valid) > 3:
                                    bbox = [float(right_hand_keyp[valid, 0].min()), float(right_hand_keyp[valid, 1].min()),
                                            float(right_hand_keyp[valid, 0].max()), float(right_hand_keyp[valid, 1].max())]
                                    bboxes.append({"bbox": bbox, "is_right": 1})
                                    
                        if len(bboxes) > 0:
                            clip_frames_dict[str(i)] = bboxes
                            
                    # 将这个 clip 的结果落地写入临时 json（无论是否有框，写了就代表跑过了）
                    with open(cache_path, 'w') as cf:
                        json.dump(clip_frames_dict, cf)
                        
    except Exception as e:
        import traceback
        print(f"[Worker GPU {gpu_id}] 发生严重错误: {e}")
        traceback.print_exc()


def extract_test_bboxes_multigpu(logical_gpus):
    """主进程：分配任务，等待结束，然后合并所有缓存文件为一个大 JSON"""
    splits_json_path = os.path.join(base_dir, "evaluation/opentouch_splits.json")
    output_bbox_json = os.path.join(base_dir, "hamer_tactile_ft/opentouch_test_bboxes.json")
    
    # 建立用于断点重续的临时缓存文件夹
    cache_dir = os.path.join(base_dir, "hamer_tactile_ft/test_bboxes_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    with open(splits_json_path, 'r') as f:
        splits = json.load(f)
        
    test_clips = splits.get("test", [])
    if not test_clips:
        print("未在划分文件中找到 test 集。")
        return None
        
    print(f"✅ 开始为 test 集提框，总计包含 {len(test_clips)} 个 clip...")
    print(f"✅ 断点重续机制已激活，将在 {cache_dir} 记录进度。")
    
    # 根据指定的 logical GPU 数量对剪辑进行分块
    num_gpus = len(logical_gpus)
    chunk_size = math.ceil(len(test_clips) / num_gpus)
    chunks = [test_clips[i:i + chunk_size] for i in range(0, len(test_clips), chunk_size)]
    
    pool_args = []
    for i, gpu_id in enumerate(logical_gpus):
        if i < len(chunks) and len(chunks[i]) > 0:
            pool_args.append((int(gpu_id), chunks[i], cache_dir))
            
    if num_gpus > 1:
        print(f"🚀 启动 {len(pool_args)} 卡并行提框进程池！")
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        with mp.Pool(len(pool_args)) as pool:
            pool.starmap(extract_worker, pool_args)
    else:
        # 如果只有 1 张卡，走单进程调用，防止 spawn 导致的奇奇怪怪错误
        print(f"🚀 启动单进程提框模式！")
        extract_worker(*pool_args[0])
        
    # 所有进程跑完后，将缓存目录下所有的 json 合并成大 JSON
    samples_dict = {}
    total_valid_frames = 0
    print("🔄 所有 Worker 已完成，正在合并局部 JSON 结果...")
    
    for scene, clip_id in test_clips:
        cache_path = os.path.join(cache_dir, f"{scene}_{clip_id}.json")
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as cf:
                clip_frames_dict = json.load(cf)
                # 只有非空的字典才加进去
                if clip_frames_dict:
                    samples_dict[f"{scene}/{clip_id}"] = clip_frames_dict
                    total_valid_frames += len(clip_frames_dict)
                    
    with open(output_bbox_json, 'w') as f:
        json.dump(samples_dict, f, indent=2)
        
    print(f"🎉 Bbox 预提取完成！总共成功提取并保存了 {total_valid_frames} 个 test 集有效帧。")
    print(f"Bbox 大 JSON 已保存至: {output_bbox_json}")
    
    return output_bbox_json


def extract_test_to_disk(bbox_json_path):
    """第二阶段：读取大 JSON，从 HDF5 中解包图像和 meta"""
    print("\n📦 开始将数据集图片及 meta.json 写入磁盘...")
    data_dir = "/data1/jiangrui/OpenTouch Data/data"
    output_dir = "/data1/jiangrui/OpenTouch Data/extracted_dataset"
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    with open(bbox_json_path, 'r') as f:
        samples_dict = json.load(f)
        
    h5_files = {}
    count = 0
    skipped_count = 0
    
    for scene_clip, frames in tqdm(samples_dict.items(), desc="Extracting to Disk"):
        scene, clip_id = scene_clip.split("/")
        
        # 延迟打开 HDF5
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
                sample_dir = os.path.join(output_dir, "test", sample_folder_name)
                os.makedirs(sample_dir, exist_ok=True)
                
                # ==============================================================
                # 磁盘提取断点重续：如果目标 meta.json 已经存在，则说明之前已经跑过该图
                # ==============================================================
                meta_path = os.path.join(sample_dir, "meta.json")
                if os.path.exists(meta_path):
                    skipped_count += 1
                    continue
                
                image_path = os.path.join(sample_dir, "image.jpg")
                with open(image_path, "wb") as img_file:
                    img_file.write(byte_data)
                    
                # 处理 Landmarks
                if is_right == 1:
                    landmarks = clip_group["right_hand_landmarks"][frame_idx]
                else:
                    if "left_hand_landmarks" in clip_group:
                        landmarks = clip_group["left_hand_landmarks"][frame_idx]
                    else:
                        landmarks = np.zeros((21, 3))
                        landmarks[:] = np.nan
                        
                camera_pose = clip_group["camera_poses"][frame_idx]
                rgb_calib = f["calibration/rgb"]
                T_device_camera = rgb_calib["T_device_camera"][:]
                focal_length = rgb_calib["focal_length"][()]
                image_size = rgb_calib["image_size"][:]
                principal_point = rgb_calib["principal_point"][:]
                
                # 收集原始 HDF5 所有关键数据 (包含刚刚生成的 _continuous 数据)
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
                    "bbox": bbox,
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

    for f in h5_files.values():
        f.close()
        
    print(f"磁盘碎片化写入完成！")
    print(f"共全新提取: {count} 个 test 样本。")
    print(f"跳过已存在样本: {skipped_count} 个。")
    print(f"输出目录: {os.path.join(output_dir, 'test')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract test set bboxes and write dataset to disk")
    parser.add_argument('--gpu', type=str, default='0', help='使用的 GPU 编号 (例: 3,4,5,7)')
    args = parser.parse_args()
    
    # 强制切换工作目录到 hamer_tactile_ft 上级
    os.chdir(base_dir)
    
    # 解析逻辑 GPU (比如传了 3,4,5,7，在可见设备里他们分别是 0,1,2,3)
    gpu_list = args.gpu.split(',') if args.gpu else ['0']
    logical_gpus = list(range(len(gpu_list)))
    
    bbox_path = extract_test_bboxes_multigpu(logical_gpus)
    if bbox_path:
        extract_test_to_disk(bbox_path)
