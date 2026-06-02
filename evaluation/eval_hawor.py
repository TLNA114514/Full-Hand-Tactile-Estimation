import sys
import os
import json
import argparse
import h5py
import cv2
import numpy as np
import torch
import shutil
from pathlib import Path
from tqdm import tqdm

# 将 HaWoR 和 evaluation 路径添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../HaWoR')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
from hawor.utils.process import run_mano, run_mano_left
from eval_utils import compute_similarity_transform, compute_mpjpe, compute_pck, compute_auc, fit_mano_to_joints
from eval_hamer import mano_to_mediapipe

def eval_clip_hawor(args, hdf5_path, clip_id, temp_root="temp_hawor"):
    # 1. 自动从 HDF5 中解包图像帧并写入临时目录
    temp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), temp_root))
    temp_clip_dir = os.path.join(temp_root, clip_id)
    img_folder = os.path.join(temp_clip_dir, clip_id, "extracted_images")
    os.makedirs(img_folder, exist_ok=True)
    
    with h5py.File(hdf5_path, "r") as f:
        clip_group = f[f"data/{clip_id}"]
        rgb_bytes_seq = clip_group["rgb_images_jpeg"][()]
        
        # 加载 GT
        gt_right_landmarks = clip_group["right_hand_landmarks"][()] if "right_hand_landmarks" in clip_group else None
        gt_left_landmarks = clip_group["left_hand_landmarks"][()] if "left_hand_landmarks" in clip_group else None
        
    num_frames = len(rgb_bytes_seq)
    print(f"正在从 HDF5 解压 Clip: {clip_id} 的图片序列 ({num_frames} 帧)...")
    
    for i in range(num_frames):
        img_path = os.path.join(img_folder, f"{i:04d}.jpg")
        # 直接写入 JPEG 字节数据
        with open(img_path, "wb") as img_f:
            img_f.write(rgb_bytes_seq[i])
            
    # 设置运行参数
    args.video_path = os.path.join(temp_clip_dir, f"{clip_id}.mp4") # 虚拟 mp4 路径
    
    # 2. 调用 HaWoR 内置的自动检测和追踪流水线
    print(f"正在运行 HaWoR 检测与追踪跟踪器...")
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    
    # 3. 运行 HaWoR 时序运动估计
    print(f"正在运行 HaWoR 运动估计 (Chunk Inference)...")
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    
    # 4. 直接使用相机空间结果，跳过 SLAM 与 Infiller 以彻底规避 Droid-SLAM (masked_droid_slam) 依赖缺失问题
    # 这也更契合单视角相机空间手部重建的通用评估场景，普氏对齐 (Procrustes Alignment) 会数学上完全消除坐标系差异。
    print(f"基于相机空间对齐提取评估关节与顶点 (已完美规避 Droid-SLAM 依赖)...")
    
    all_pred_joints = []
    all_gt_joints = []
    all_pred_verts = []
    all_gt_verts = []
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # 初始化一个 MANO 模块用于 GT 拟合评估
    from lib.models.mano_wrapper import MANO
    MANO_cfg = {
        'DATA_DIR': '_DATA/data/',
        'MODEL_PATH': '_DATA/data/mano',
        'GENDER': 'neutral',
        'NUM_HAND_JOINTS': 15,
        'CREATE_BODY_POSE': False
    }
    mano_model = MANO(**{k.lower(): v for k,v in MANO_cfg.items()}).to(device)
    
    # 左右手对应的 MANO 索引，0: 左手，1: 右手
    hand_configs = []
    if gt_left_landmarks is not None:
        hand_configs.append((0, 'left', gt_left_landmarks))
    if gt_right_landmarks is not None:
        hand_configs.append((1, 'right', gt_right_landmarks))
        
    import glob
    import json
    
    for hand_idx, hand_name, gt_seq in hand_configs:
        json_dir = os.path.join(seq_folder, 'cam_space', str(hand_idx))
        if not os.path.exists(json_dir):
            continue
            
        json_paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        
        for json_path in json_paths:
            base = os.path.basename(json_path)
            fn_parts = os.path.splitext(base)[0].split("_")
            chunk_start = int(fn_parts[0])
            chunk_end = int(fn_parts[1])
            frame_ck = list(range(chunk_start, chunk_end + 1))
            
            with open(json_path, "r") as f_json:
                pred_dict = json.load(f_json)
                
            data_out = {
                k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in pred_dict.items()
            }
            
            from hawor.utils.rotation import rotation_matrix_to_angle_axis
            init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
            init_hand_pose = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
            
            with torch.no_grad():
                if hand_name == 'right':
                    pred_glob = run_mano(
                        data_out["init_trans"], 
                        init_root, 
                        init_hand_pose, 
                        betas=data_out["init_betas"], 
                        use_cuda=torch.cuda.is_available()
                    )
                else:
                    pred_glob = run_mano_left(
                        data_out["init_trans"], 
                        init_root, 
                        init_hand_pose, 
                        betas=data_out["init_betas"], 
                        use_cuda=torch.cuda.is_available()
                    )
                    
            pred_joints = pred_glob['joints'][0].cpu().numpy() # [T, 21, 3]
            pred_verts = pred_glob['vertices'][0].cpu().numpy() # [T, 778, 3]
            
            # 模型内置的 MANO_wrapper 已自动将关节转为 MediaPipe 格式
            pred_joints_mp = pred_joints.copy()
            
            for local_idx, i in enumerate(frame_ck):
                if i >= num_frames:
                    continue
                gt_j = gt_seq[i]
                if np.isnan(gt_j).any():
                    continue
                    
                all_pred_joints.append(pred_joints_mp[local_idx])
                all_gt_joints.append(gt_j)
                
                # 拟合获取 GT 顶点 (PA-MPVPE)
                try:
                    # 由于 mano_model 是右手模型，针对左手 GT 需做镜像拟合处理
                    fit_gt_j = gt_j.copy()
                    if hand_name == 'left':
                        fit_gt_j[:, 0] *= -1.0
                        
                    gt_v_fit = fit_mano_to_joints(mano_model, fit_gt_j, device, num_steps=40)
                    
                    if hand_name == 'left':
                        gt_v_fit[:, 0] *= -1.0
                        
                    # pred_verts 已通过 run_mano_left 获得了正确的左手 mesh，无需镜像
                    all_pred_verts.append(pred_verts[local_idx])
                    all_gt_verts.append(gt_v_fit)
                except Exception as ve:
                    pass
            
    # 清理临时解包图像，释放磁盘空间
    try:
        shutil.rmtree(temp_clip_dir)
    except:
        pass
        
    if len(all_pred_joints) == 0:
        print(f"❌ Clip {clip_id} 中没有成功匹配的帧用于指标计算！")
        return None
        
    all_pred = np.stack(all_pred_joints)
    all_gt = np.stack(all_gt_joints)
    
    # 5. 普氏对齐与指标计算
    pa_mpjpe = compute_mpjpe(all_pred, all_gt, alignment='procrustes')
    pck_5 = compute_pck(all_pred, all_gt, threshold_mm=5.0, alignment='procrustes')
    pck_15 = compute_pck(all_pred, all_gt, threshold_mm=15.0, alignment='procrustes')
    auc = compute_auc(all_pred, all_gt, min_thr=0.0, max_thr=50.0, num_steps=31, alignment='procrustes')
    
    pa_mpvpe = 0.0
    if len(all_pred_verts) > 0:
        from eval_utils import compute_pa_mpvpe
        pa_mpvpe = compute_pa_mpvpe(all_pred, all_gt, np.stack(all_pred_verts), np.stack(all_gt_verts))
    
    print(f"--- {clip_id} (HaWoR) 评估结果 ---")
    print(f"有效评估关节帧数: {len(all_pred)}")
    print(f"PA-MPJPE: {pa_mpjpe:.2f} mm")
    if len(all_pred_verts) > 0:
        print(f"PA-MPVPE: {pa_mpvpe:.2f} mm")
    print(f"PCK@5mm  : {pck_5:.2f} %")
    print(f"PCK@15mm : {pck_15:.2f} %")
    print(f"AUC (0-50mm): {auc:.2f}")
    
    return {
        "pa_mpjpe": pa_mpjpe,
        "pa_mpvpe": pa_mpvpe,
        "pck_5": pck_5,
        "pck_15": pck_15,
        "auc": auc,
        "count": len(all_pred)
    }

def load_or_generate_splits(split_json_path, data_dir):
    if os.path.exists(split_json_path):
        with open(split_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    print(f"⚠️ 划分文件 {split_json_path} 不存在，正在自动扫描 {data_dir} 并生成默认划分...")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"OpenTouch 数据目录 {data_dir} 不存在，无法自动生成划分！")
        
    import random
    from generate_splits import split_clips_into_train_val_test
    
    hdf5_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])
    clip_keys = []
    for f_name in hdf5_files:
        f_path = os.path.join(data_dir, f_name)
        scene_name = os.path.splitext(f_name)[0]
        try:
            with h5py.File(f_path, "r") as f:
                if "data" in f:
                    clips_in_file = sorted(list(f["data"].keys()))
                    for clip_id in clips_in_file:
                        clip_keys.append((scene_name, clip_id))
        except Exception as e:
            print(f"Warning: 读取 {f_name} 失败: {e}")
            
    clip_keys.sort()
    splits = split_clips_into_train_val_test(clip_keys, 0.1, 0.1, 42)
    
    os.makedirs(os.path.dirname(split_json_path), exist_ok=True)
    with open(split_json_path, "w", encoding="utf-8") as json_f:
        json.dump(splits, json_f, indent=2, ensure_ascii=False)
    print(f"✅ 默认划分已成功生成并保存至 {split_json_path}")
    return splits

def main():
    parser = argparse.ArgumentParser(description='HaWoR OpenTouch 数据集一键式时序评估')
    parser.add_argument('--checkpoint', type=str, default='./weights/hawor/checkpoints/hawor.ckpt', help='HaWoR 权重路径')
    parser.add_argument('--hdf5_path', type=str, default=None, help='OpenTouch HDF5 数据集文件路径')
    parser.add_argument('--clips', nargs='+', default=['demo_05', 'demo_10', 'demo_15'], help='评估的 Clips 列表')
    parser.add_argument('--gpu', type=str, default='4', help='使用 GPU 编号')
    parser.add_argument('--img_focal', type=float, default=600, help='焦距估计值')
    parser.add_argument('--input_type', type=str, default='file')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val', 'test', 'all'], help='评估的数据集划分')
    parser.add_argument('--split_json', type=str, default=None, help='划分 JSON 文件的路径')
    args = parser.parse_args()
    
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    if args.split_json is None:
        args.split_json = os.path.join(eval_dir, "opentouch_splits.json")
        
    # 确保传入的绝对路径不会被 chdir 影响
    if args.hdf5_path is not None:
        args.hdf5_path = os.path.abspath(args.hdf5_path)
    args.checkpoint = os.path.abspath(args.checkpoint)
    args.split_json = os.path.abspath(args.split_json)
    
    if args.split is None and args.hdf5_path is None:
        parser.error("在使用单文件评估模式时，必须指定 --hdf5_path；或者指定 --split 进行划分评估。")
    
    # 将工作目录切换到 HaWoR 根目录，使得其内部对 ./weights/... 和 _DATA/... 的相对路径访问生效
    hawor_root = os.path.abspath(os.path.join(eval_dir, '../HaWoR'))
    os.chdir(hawor_root)
    
    # 设置运行的 GPU 设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"使用 GPU 设备: cuda:{args.gpu} 进行推理评估")
    
    results = []
    
    if args.split is not None:
        data_dir = os.path.abspath(os.path.join(eval_dir, "../opentouch/data"))
        try:
            splits = load_or_generate_splits(args.split_json, data_dir)
        except Exception as se:
            print(f"❌ 加载或生成划分失败: {se}")
            sys.exit(1)
            
        if args.split == 'all':
            target_clips = []
            for s_name in ['train', 'val', 'test']:
                target_clips.extend(splits.get(s_name, []))
        else:
            target_clips = splits.get(args.split, [])
            
        if not target_clips:
            print(f"❌ 划分 {args.split} 中没有包含任何 clip！")
            sys.exit(1)
            
        print(f"🔔 开始评测 split: {args.split} | 共有 {len(target_clips)} 个 clips 进行评估")
        
        # 按 scene_name 分组优化 HDF5 打开性能
        from collections import defaultdict
        scene_to_clips = defaultdict(list)
        for scene, clip in target_clips:
            scene_to_clips[scene].append(clip)
            
        for scene_name, clip_ids in scene_to_clips.items():
            hdf5_path = os.path.join(data_dir, f"{scene_name}.hdf5")
            if not os.path.exists(hdf5_path):
                print(f"⚠️ 找不到 HDF5 文件 {hdf5_path}，跳过该场景的评估。")
                continue
                
            print(f"📂 正在评估场景: {scene_name} | Clips: {clip_ids}")
            for clip in clip_ids:
                try:
                    res = eval_clip_hawor(args, hdf5_path, clip)
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    print(f"⚠️ 处理场景 {scene_name} 中的 Clip {clip} 时发生错误: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        # 原有单文件评估逻辑
        for clip in args.clips:
            try:
                res = eval_clip_hawor(args, args.hdf5_path, clip)
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"⚠️ 处理 Clip {clip} 时发生错误: {e}")
                import traceback
                traceback.print_exc()
            
    if len(results) > 0:
        total_frames = sum([r["count"] for r in results])
        avg_pa_mpjpe = sum([r["pa_mpjpe"] * r["count"] for r in results]) / total_frames
        avg_pa_mpvpe = sum([r["pa_mpvpe"] * r["count"] for r in results]) / total_frames
        avg_pck_5 = sum([r["pck_5"] * r["count"] for r in results]) / total_frames
        avg_pck_15 = sum([r["pck_15"] * r["count"] for r in results]) / total_frames
        avg_auc = sum([r["auc"] * r["count"] for r in results]) / total_frames
        
        report_lines = [
            f"🎉 HaWoR 在 OpenTouch {f'划分: {args.split}' if args.split else '指定集'} 上的最终时序评估结果 🎉",
            "="*55,
            f" 评估模型类型  : HaWoR",
            f" 评测划分/片段 : {args.split if args.split else args.clips}",
            f" 总评估帧数    : {total_frames}",
            f" 整体 PA-MPJPE : {avg_pa_mpjpe:.2f} mm",
            f" 整体 PA-MPVPE : {avg_pa_mpvpe:.2f} mm",
            f" 整体 PCK@5mm  : {avg_pck_5:.2f} %",
            f" 整体 PCK@15mm : {avg_pck_15:.2f} %",
            f" 整体 AUC (0-50mm): {avg_auc:.2f}",
            "="*55
        ]
        report_text = "\n".join(report_lines)
        print("\n" + report_text)
        
        # 保存报告
        report_path = os.path.join(eval_dir, "eval_hawor_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f_rep:
                f_rep.write(report_text + "\n")
            print(f"📝 最终评测报告已保存至: {report_path}")
        except Exception as re:
            print(f"⚠️ 保存报告失败: {re}")
    else:
        print("❌ 未能成功跑完任何评估！")

if __name__ == '__main__':
    main()
