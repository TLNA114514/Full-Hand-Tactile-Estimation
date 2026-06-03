import sys
import os
import json
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['PYRENDER_PLATFORM'] = 'osmesa'
import argparse
import h5py
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# 将 hamer 和 evaluation 路径添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../hamer')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from vitpose_model import ViTPoseModel
from eval_utils import compute_similarity_transform, compute_mpjpe, compute_pck, compute_auc, fit_mano_to_joints

# 定义 MANO 到 MediaPipe 的索引映射
def mano_to_mediapipe(mano_joints):
    """
    将 [B, 21, 3] 或 [21, 3] 的 MANO 格式关节重映射到 MediaPipe 21 关节标准
    """
    is_batched = mano_joints.ndim == 3
    if not is_batched:
        mano = mano_joints[None, ...]
    else:
        mano = mano_joints
        
    B = mano.shape[0]
    mp_joints = np.zeros((B, 21, 3), dtype=mano.dtype)
    
    # 0: Wrist
    mp_joints[:, 0] = mano[:, 0]
    
    # 大拇指: MANO 13, 14, 15, 20 -> MediaPipe 1, 2, 3, 4
    mp_joints[:, 1] = mano[:, 13]
    mp_joints[:, 2] = mano[:, 14]
    mp_joints[:, 3] = mano[:, 15]
    mp_joints[:, 4] = mano[:, 20]
    
    # 食指: MANO 1, 2, 3, 16 -> MediaPipe 5, 6, 7, 8
    mp_joints[:, 5] = mano[:, 1]
    mp_joints[:, 6] = mano[:, 2]
    mp_joints[:, 7] = mano[:, 3]
    mp_joints[:, 8] = mano[:, 16]
    
    # 中指: MANO 4, 5, 6, 17 -> MediaPipe 9, 10, 11, 12
    mp_joints[:, 9] = mano[:, 4]
    mp_joints[:, 10] = mano[:, 5]
    mp_joints[:, 11] = mano[:, 6]
    mp_joints[:, 12] = mano[:, 17]
    
    # 无名指: MANO 10, 11, 12, 19 -> MediaPipe 13, 14, 15, 16
    mp_joints[:, 13] = mano[:, 10]
    mp_joints[:, 14] = mano[:, 11]
    mp_joints[:, 15] = mano[:, 12]
    mp_joints[:, 16] = mano[:, 19]
    
    # 小拇指: MANO 7, 8, 9, 18 -> MediaPipe 17, 18, 19, 20
    mp_joints[:, 17] = mano[:, 7]
    mp_joints[:, 18] = mano[:, 8]
    mp_joints[:, 19] = mano[:, 9]
    mp_joints[:, 20] = mano[:, 18]
    
    return mp_joints if is_batched else mp_joints[0]

def eval_clip(model, detector, cpm, model_cfg, hdf5_path, clip_id, device, rescale_factor=2.0, disable_tqdm=False):
    with h5py.File(hdf5_path, "r") as f:
        clip_group = f[f"data/{clip_id}"]
        rgb_bytes_seq = clip_group["rgb_images_jpeg"][()]
        
        # 加载 GT 关节点 (MediaPipe 21关节格式)
        gt_right_landmarks = clip_group["right_hand_landmarks"][()] if "right_hand_landmarks" in clip_group else None
        gt_left_landmarks = clip_group["left_hand_landmarks"][()] if "left_hand_landmarks" in clip_group else None
        
    num_frames = len(rgb_bytes_seq)
    print(f"正在处理 Clip: {clip_id} | 帧数: {num_frames}")
    
    all_pred_joints = []
    all_gt_joints = []
    all_pred_verts = []
    all_gt_verts = []
    
    # 逐帧解码与推理
    for i in tqdm(range(num_frames), desc=f"Clip {clip_id}", disable=disable_tqdm):
        # 1. 图像解码
        img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes_seq[i], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
            
        img_rgb = img_bgr[:, :, ::-1]
        
        # 2. 调用内置目标检测器定位人体
        det_out = detector(img_bgr)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        if len(pred_bboxes) == 0:
            continue
            
        # 3. 调用 ViTPose 提取手部位置并生成检测框
        vitposes_out = cpm.predict_pose(
            img_rgb,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        
        bboxes = []
        is_right = []
        
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            
            # 过滤置信度低于 0.5 的关键点，生成手部检测框
            # 仅在检测到右手且有 GT 右手时才记录（或者左右手分别对齐评估）
            if gt_left_landmarks is not None:
                valid = left_hand_keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [left_hand_keyp[valid, 0].min(), left_hand_keyp[valid, 1].min(),
                            left_hand_keyp[valid, 0].max(), left_hand_keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
            
            if gt_right_landmarks is not None:
                valid = right_hand_keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [right_hand_keyp[valid, 0].min(), right_hand_keyp[valid, 1].min(),
                            right_hand_keyp[valid, 0].max(), right_hand_keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)
                    
        if len(bboxes) == 0:
            continue
            
        boxes_arr = np.stack(bboxes)
        right_arr = np.stack(is_right)
        
        # 4. Hamer 三维 Mesh 推理
        dataset = ViTDetDataset(model_cfg, img_bgr, boxes_arr, right_arr, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
                
            pred_joints = out['pred_keypoints_3d'].detach().cpu().numpy()
            
            # 模型内置的 MANO_wrapper 已自动将关节转为 MediaPipe 格式
            pred_joints_mp = pred_joints.copy()
            
            # 逐个对齐评估
            for n in range(pred_joints_mp.shape[0]):
                is_r = batch['right'][n].cpu().numpy()
                
                # 获取该帧对应的 Ground Truth
                if is_r == 1 and gt_right_landmarks is not None:
                    gt_j = gt_right_landmarks[i]
                elif is_r == 0 and gt_left_landmarks is not None:
                    gt_j = gt_left_landmarks[i]
                else:
                    continue
                    
                # 排除缺失的关节点
                if np.isnan(gt_j).any():
                    continue
                    
                # 【重要修复】Hamer 永远输出右手。如果是左手，必须将预测的 X 轴翻转回去！
                if is_r == 0:
                    pred_joints_mp[n, :, 0] *= -1.0
                    
                all_pred_joints.append(pred_joints_mp[n])
                all_gt_joints.append(gt_j)
                
                # 拟合获取 GT 顶点，并保存预测/GT 顶点用于计算 PA-MPVPE
                try:
                    # 对于左手，由于我们仅有一个右手 MANO 模型，需先将 GT 镜像为右手，拟合后再镜像回左手
                    fit_gt_j = gt_j.copy()
                    if is_r == 0:
                        fit_gt_j[:, 0] *= -1.0
                        
                    gt_v_fit = fit_mano_to_joints(model.mano, fit_gt_j, device, num_steps=40)
                    
                    if is_r == 0:
                        gt_v_fit[:, 0] *= -1.0
                        
                    pred_v = out['pred_vertices'][n].detach().cpu().numpy()
                    
                    # 同样，Hamer 预测的 mesh 永远是右手，如果是左手也需要镜像翻转 X 轴
                    if is_r == 0:
                        pred_v[:, 0] *= -1.0
                        
                    all_pred_verts.append(pred_v)
                    all_gt_verts.append(gt_v_fit)
                except Exception as ve:
                    pass
                
    if len(all_pred_joints) == 0:
        print(f"❌ Clip {clip_id} 中没有成功匹配的帧用于指标计算！")
        return None
        
    all_pred = np.stack(all_pred_joints)
    all_gt = np.stack(all_gt_joints)
    
    # 5. 计算当前 Clip 的所有指标
    pa_mpjpe = compute_mpjpe(all_pred, all_gt, alignment='procrustes')
    pck_5 = compute_pck(all_pred, all_gt, threshold_mm=5.0, alignment='procrustes')
    pck_15 = compute_pck(all_pred, all_gt, threshold_mm=15.0, alignment='procrustes')
    auc = compute_auc(all_pred, all_gt, min_thr=0.0, max_thr=50.0, num_steps=31, alignment='procrustes')
    
    pa_mpvpe = 0.0
    if len(all_pred_verts) > 0:
        from eval_utils import compute_pa_mpvpe
        pa_mpvpe = compute_pa_mpvpe(all_pred, all_gt, np.stack(all_pred_verts), np.stack(all_gt_verts))
    
    print(f"--- {clip_id} 评估结果 ---")
    print(f"有效评估帧数: {len(all_pred)}")
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

def eval_worker(gpu_id, clips_chunk, args_checkpoint, args_model_cfg, data_dir, hdf5_path_arg):
    import traceback
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"[Worker GPU {gpu_id}] 正在初始化模型... 分配了 {len(clips_chunk)} 个片段")
        
        if args_model_cfg is not None:
            from hamer.configs import get_config
            model_cfg_path = os.path.abspath(args_model_cfg)
            model_cfg = get_config(model_cfg_path, update_cachedir=True)
            if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
                model_cfg.defrost()
                model_cfg.MODEL.BBOX_SHAPE = [192, 256]
                model_cfg.freeze()
            if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
                model_cfg.defrost()
                model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
                model_cfg.freeze()
            model = HAMER.load_from_checkpoint(args_checkpoint, strict=False, cfg=model_cfg)
        else:
            model, model_cfg = load_hamer(args_checkpoint)
            
        model = model.to(device)
        model.eval()
        
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        cpm = ViTPoseModel(device)
        
        results = []
        
        if data_dir is not None:
            from collections import defaultdict
            scene_to_clips = defaultdict(list)
            for scene, clip in clips_chunk:
                scene_to_clips[scene].append(clip)
                
            for scene_name, clip_ids in scene_to_clips.items():
                hdf5_path = os.path.join(data_dir, f"{scene_name}.hdf5")
                if not os.path.exists(hdf5_path):
                    continue
                for clip in clip_ids:
                    try:
                        res = eval_clip(model, detector, cpm, model_cfg, hdf5_path, clip, device, disable_tqdm=True)
                        if res is not None:
                            results.append(res)
                            print(f"[Worker GPU {gpu_id}] ✅ 完成片段: {clip}")
                    except Exception as e:
                        print(f"[Worker GPU {gpu_id}] ⚠️ 处理 {scene_name}/{clip} 失败: {e}")
        else:
            for clip in clips_chunk:
                try:
                    res = eval_clip(model, detector, cpm, model_cfg, hdf5_path_arg, clip, device, disable_tqdm=True)
                    if res is not None:
                        results.append(res)
                        print(f"[Worker GPU {gpu_id}] ✅ 完成片段: {clip}")
                except Exception as e:
                    print(f"[Worker GPU {gpu_id}] ⚠️ 处理 Clip {clip} 失败: {e}")
                    
        return results
    except Exception as e:
        print(f"[Worker GPU {gpu_id}] ❌ 初始化/评估时发生严重错误: {e}")
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser(description='Hamer OpenTouch 数据集一键式评估')
    parser.add_argument('--checkpoint', type=str, default="../hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt", help='Hamer 模型 Checkpoint 路径')
    parser.add_argument('--model_cfg', type=str, default=None, help='模型配置文件路径 (用于微调后的模型)')
    parser.add_argument('--hdf5_path', type=str, default=None, help='OpenTouch HDF5 数据集文件路径')
    parser.add_argument('--clips', nargs='*', default=None, help='需要评估的 Clips 列表 (可与 --split 结合使用来过滤特定 Clip)')
    parser.add_argument('--gpu', type=str, default='4', help='使用的 GPU 编号')
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
    
    # 切换 CWD 到 hamer 根目录，以便 Hamer 内部使用相对路径加载文件（如 ./_DATA/...）
    hamer_root = os.path.abspath(os.path.join(eval_dir, '../hamer'))
    os.chdir(hamer_root)
    
    gpus = [g.strip() for g in args.gpu.split(',')]
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
            
        # 允许用户在指定 Split 的同时，再单独提取某几个 Clip
        if args.clips is not None and len(args.clips) > 0:
            target_clips = [tc for tc in target_clips if tc[1] in args.clips]
            
        if not target_clips:
            print(f"❌ 划分 {args.split} 中没有包含任何指定的 clip！")
            sys.exit(1)
    else:
        target_clips = args.clips
        data_dir = None
        
    print(f"🔔 开始评测 | 共有 {len(target_clips)} 个 clips 进行评估 | 使用 GPUs: {args.gpu}")
    
    if len(gpus) > 1:
        import torch.multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        # Split targets into len(gpus) chunks
        import math
        chunk_size = math.ceil(len(target_clips) / len(gpus))
        chunks = [target_clips[i:i + chunk_size] for i in range(0, len(target_clips), chunk_size)]
        
        pool_args = []
        for i, gpu_id in enumerate(gpus):
            if i < len(chunks) and len(chunks[i]) > 0:
                pool_args.append((
                    gpu_id, 
                    chunks[i], 
                    args.checkpoint, 
                    args.model_cfg, 
                    data_dir, 
                    args.hdf5_path
                ))
                
        print(f"🚀 启动 {len(pool_args)} 卡并行评测进程池！")
        with mp.Pool(len(pool_args)) as pool:
            multi_results = pool.starmap(eval_worker, pool_args)
            
        for r in multi_results:
            results.extend(r)
    else:
        # 单卡模式
        gpu_id = gpus[0]
        print(f"🚀 单卡模式启动，GPU: {gpu_id}")
        results = eval_worker(gpu_id, target_clips, args.checkpoint, args.model_cfg, data_dir, args.hdf5_path)
                
    if len(results) > 0:
        # 计算加权平均指标
        total_frames = sum([r["count"] for r in results])
        avg_pa_mpjpe = sum([r["pa_mpjpe"] * r["count"] for r in results]) / total_frames
        avg_pa_mpvpe = sum([r["pa_mpvpe"] * r["count"] for r in results]) / total_frames
        avg_pck_5 = sum([r["pck_5"] * r["count"] for r in results]) / total_frames
        avg_pck_15 = sum([r["pck_15"] * r["count"] for r in results]) / total_frames
        avg_auc = sum([r["auc"] * r["count"] for r in results]) / total_frames
        
        report_lines = [
            f"🎉 Hamer 在 OpenTouch {f'划分: {args.split}' if args.split else '指定集'} 上的最终加权评估结果 🎉",
            "="*55,
            f" 评估模型类型  : Hamer",
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
        report_path = os.path.join(eval_dir, "eval_hamer_report.txt")
        try:
            with open(report_path, "w", encoding="utf-8") as f_rep:
                f_rep.write(report_text + "\n")
            print(f"📝 最终评测报告已保存至: {report_path}")
        except Exception as re:
            print(f"⚠️ 保存报告失败: {re}")
    else:
        print("❌ 未产生任何有效的评估指标！")

if __name__ == '__main__':
    main()
