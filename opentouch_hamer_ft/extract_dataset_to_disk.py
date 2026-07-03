import os
import json
import h5py
import numpy as np
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def extract_dataset():
    # 路径配置
    bbox_json_path = "/code/users/jiangrui/Full-Hand-Tactile-Estimation/opentouch_hamer_ft/opentouch_train_val_bboxes.json"
    data_dir = "/data1/jiangrui/OpenTouch Data/data"
    output_dir = "/data1/jiangrui/OpenTouch Data/extracted_dataset"
    
    # 划分文件配置
    splits_json_path = "/code/users/jiangrui/Full-Hand-Tactile-Estimation/evaluation/opentouch_splits.json"
    
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    
    # 读取划分列表
    with open(splits_json_path, 'r') as f:
        splits = json.load(f)
        
    train_clips = set(f"{item[0]}/{item[1]}" for item in splits.get("train", []))
    val_clips = set(f"{item[0]}/{item[1]}" for item in splits.get("val", []))
    
    print(f"Loading bbox JSON: {bbox_json_path}")
    with open(bbox_json_path, 'r') as f:
        samples_dict = json.load(f)
        
    # 我们用一个字典来缓存打开的 HDF5 文件，避免频繁开关
    h5_files = {}
    
    count = 0
    # 由于 samples_dict 的结构是: {"scene/clip_id": {"frame_idx": [{"bbox": [], "is_right": 1}]}}
    for scene_clip, frames in tqdm(samples_dict.items(), desc="Processing clips"):
        scene, clip_id = scene_clip.split("/")
        
        # 判断属于哪个集
        if scene_clip in train_clips:
            split_folder = "train"
        elif scene_clip in val_clips:
            split_folder = "val"
        else:
            # 如果都不在，暂时跳过
            continue
            
        for frame_idx_str, boxes in frames.items():
            frame_idx = int(frame_idx_str)
            
            for box_idx, sample in enumerate(boxes):
                is_right = sample.get("is_right", 1)
                bbox = sample["bbox"]
                
                # 2. 读取对应的数据 (提取到前面来做越界检查)
                h5_path = os.path.join(data_dir, f"{scene}.hdf5")
                if h5_path not in h5_files:
                    if not os.path.exists(h5_path):
                        print(f"Warning: {h5_path} does not exist. Skipping.")
                        continue
                    h5_files[h5_path] = h5py.File(h5_path, 'r', swmr=True)
                    
                f = h5_files[h5_path]
                if f"data/{clip_id}" not in f:
                    continue
                clip_group = f[f"data/{clip_id}"]
                
                # 读取图片序列并做越界检查
                rgb_bytes_seq = clip_group["rgb_images_jpeg"]
                if frame_idx >= len(rgb_bytes_seq):
                    # 数据集不一致导致帧索引越界，跳过该样本
                    continue
                
                # 为了防止同一帧里有两只手导致文件夹重名，在名字末尾加上 is_right 标识
                sample_folder_name = f"{scene}_{clip_id}_{frame_idx:04d}_{is_right}"
                sample_dir = os.path.join(output_dir, split_folder, sample_folder_name)
                os.makedirs(sample_dir, exist_ok=True)
                
                raw_bytes = rgb_bytes_seq[frame_idx]
                byte_data = raw_bytes.tobytes() if isinstance(raw_bytes, np.ndarray) else raw_bytes
        
                # 写入图片文件
                image_path = os.path.join(sample_dir, "image.jpg")
                with open(image_path, "wb") as img_file:
                    img_file.write(byte_data)
                    
                # 读取并处理 3D landmarks
                if is_right == 1:
                    landmarks = clip_group["right_hand_landmarks"][frame_idx]
                else:
                    if "left_hand_landmarks" in clip_group:
                        landmarks = clip_group["left_hand_landmarks"][frame_idx]
                    else:
                        landmarks = np.zeros((21, 3))
                        landmarks[:] = np.nan
                    
                camera_pose = clip_group["camera_poses"][frame_idx]
                
                # 读取相机内参
                rgb_calib = f["calibration/rgb"]
                T_device_camera = rgb_calib["T_device_camera"][:]
                focal_length = rgb_calib["focal_length"][()]
                image_size = rgb_calib["image_size"][:]
                principal_point = rgb_calib["principal_point"][:]
                
                # 提取当前帧的所有原始数据，以便未来备用
                original_data = {}
                for key in clip_group.keys():
                    if key == "rgb_images_jpeg":
                        continue
                    
                    val = clip_group[key]
                    # 对时间戳、姿态、压力等各个字段提取当前帧
                    if len(val.shape) > 0 and val.shape[0] > frame_idx:
                        try:
                            frame_data = val[frame_idx]
                            if isinstance(frame_data, np.ndarray):
                                original_data[key] = frame_data.tolist()
                            else:
                                # Convert numpy scalars to Python scalars
                                original_data[key] = frame_data.item() if hasattr(frame_data, "item") else frame_data
                        except Exception:
                            pass
                
                # 将原始 calibration 写入
                original_calibration = {}
                for k in rgb_calib.keys():
                    val = rgb_calib[k][...]
                    if isinstance(val, np.ndarray):
                        original_calibration[k] = val.tolist()
                    else:
                        original_calibration[k] = val.item() if hasattr(val, "item") else val
        
                # 坐标转换逻辑: World -> SLAM Device -> RGB Camera
                T_world_to_slam = np.linalg.inv(camera_pose)
                
                # NaN 检查与处理
                if not np.isnan(landmarks).any() and len(landmarks) == 21:
                    landmarks_homo = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
                    landmarks_device = (T_world_to_slam @ landmarks_homo.T).T[:, :3]
                    landmarks_device_homo = np.concatenate([landmarks_device, np.ones((landmarks_device.shape[0], 1))], axis=1)
                    landmarks_cam = (T_device_camera @ landmarks_device_homo.T).T[:, :3]
                    valid_mask = [True] * 21
                else:
                    landmarks_cam = np.zeros((21, 3))
                    valid_mask = [False] * 21
                    
                # 构造元数据
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
                
                # 写入元数据 JSON 文件
                meta_path = os.path.join(sample_dir, "meta.json")
                with open(meta_path, "w") as meta_file:
                    json.dump(meta_data, meta_file, indent=2)
                    
                count += 1
        
    # 关闭所有 HDF5 句柄
    for f in h5_files.values():
        f.close()
        
    print(f"Extraction complete! Successfully processed {count} samples.")
    print(f"Extracted dataset is saved at: {output_dir}")

if __name__ == "__main__":
    extract_dataset()
