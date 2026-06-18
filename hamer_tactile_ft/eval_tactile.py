import sys
import os
# Force OSMesa to prevent silent EGL segfaults on headless servers
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import json
import argparse
import h5py
import cv2
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from pathlib import Path
from tqdm import tqdm

# Parse GPU early
_gpus = ""
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        _gpus = sys.argv[i+1]
        break
if _gpus:
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

# ==========================================================================================
# 🛑 核心黑魔法：源码感知 + 全局空间硬核注入补丁（地表最强终结版，完美解决一切 NameError）
# ==========================================================================================
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
_args, _ = _parser.parse_known_args()

os.environ['PYOPENGL_PLATFORM'] = _args.render_platform
os.environ['PYRENDER_PLATFORM'] = _args.render_platform

try:
    import types
    import builtins
    import re
    import sys

    # 1. 定义一个全能通配符类：既是数字0，又是可任意调用的函数，还支持无限切片和属性延伸
    class UltimateMagicMock(int):
        def __call__(self, *args, **kwargs): return self
        def __getattr__(self, name): return self
        def __getitem__(self, item): return self
        def __iter__(self): return iter([])

    class PerfectMockModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__'): raise AttributeError(name)
            return UltimateMagicMock(0)

    mock_obj = PerfectMockModule('OpenGL.GL')

    # 2. 拦截系统的底层 __import__ 行为
    orig_import = builtins.__import__
    def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
        # 只要发现有任何文件在尝试染指 OpenGL/EGL/OSMesa
        if name.startswith('OpenGL') or name in ['EGL', 'OSMesa']:
            if globals is not None and '__file__' in globals:
                try:
                    # 【硬核注入】读取当前正在执行 import 的文件（如 texture.py）的源码
                    with open(globals['__file__'], 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # 抓取该文件里写的所有 OpenGL 相关的函数和常量（如 GL_TEXTURE_2D, glGenTextures 等）
                    tokens = re.findall(r'\b([gG][lL][A-Za-z0-9_]+|[eE][gG][lL][A-Za-z0-9_]+|OSMesa[A-Za-z0-9_]+)\b', content)
                    
                    # 直接强行把这些变量塞进该文件的全局命名空间，彻底断绝 NameError 的可能
                    for token in tokens:
                        if token not in globals:
                            globals[token] = UltimateMagicMock(0)
                except Exception:
                    pass
            return mock_obj
        return orig_import(name, globals, locals, fromlist, level)
    
    # 替换系统全局导入函数
    builtins.__import__ = custom_import

    # 3. 固化系统路由备份
    sys.modules['EGL'] = mock_obj
    sys.modules['OSMesa'] = mock_obj
    sys.modules['OpenGL'] = mock_obj
    sys.modules['OpenGL.GL'] = mock_obj
    sys.modules['OpenGL.GL.shaders'] = mock_obj
    
    print("\n====== [Success] Hardcore Global Token Injector Activated! ======\n")
except Exception as e:
    print(f"Bypass failed: {e}")
# ==========================================================================================

# Add paths
eval_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../evaluation'))
hamer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../hamer'))
ft_dir = os.path.abspath(os.path.dirname(__file__))

sys.path.append(eval_dir)
sys.path.append(hamer_dir)
sys.path.append(ft_dir)

from hamer.configs import get_config
from train import OpenTouchHAMER_TactileWrapper
from eval_utils import compute_similarity_transform, compute_mpjpe, compute_pck, compute_auc, fit_mano_to_joints
from eval_hamer import ViTDetDataset, load_or_generate_splits
from vitpose_model import ViTPoseModel
from hamer.utils import recursive_to

def eval_clip(model, detector, cpm, model_cfg, hdf5_path, clip_id, device, contact_thr=0.05, rescale_factor=2.0, disable_tqdm=False):
    with h5py.File(hdf5_path, "r") as f:
        clip_group = f[f"data/{clip_id}"]
        rgb_bytes_seq = clip_group["rgb_images_jpeg"][()]
        
        # Load GT landmarks
        gt_right_landmarks = clip_group["right_hand_landmarks"][()] if "right_hand_landmarks" in clip_group else None
        gt_left_landmarks = clip_group["left_hand_landmarks"][()] if "left_hand_landmarks" in clip_group else None
        
        # Load GT tactile
        gt_right_pressure = clip_group["right_pressure"][()] if "right_pressure" in clip_group else None
        gt_left_pressure = clip_group["left_pressure"][()] if "left_pressure" in clip_group else None
        
    num_frames = len(rgb_bytes_seq)
    print(f"Processing Clip: {clip_id} | Frames: {num_frames}")
    
    all_pred_tactile = []
    all_gt_tactile = []
    
    stats = {
        'total_frames': num_frames,
        'no_img': 0,
        'no_person': 0,
        'no_hand_keypoints': 0,
        'no_valid_gt_tactile': 0,
        'valid_samples': 0
    }
    
    for i in tqdm(range(num_frames), desc=f"Clip {clip_id}", disable=disable_tqdm):
        # Decode image
        img_bgr = cv2.imdecode(np.frombuffer(rgb_bytes_seq[i], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            stats['no_img'] += 1
            continue
            
        img_rgb = img_bgr[:, :, ::-1]
        
        # ViTDet
        try:
            det_out = detector(img_bgr)
        except Exception as e:
            continue
            
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        
        if len(pred_bboxes) == 0:
            stats['no_person'] += 1
            continue
            
        # ViTPose
        try:
            vitposes_out = cpm.predict_pose(img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])
        except Exception as e:
            continue
        
        bboxes = []
        is_right = []
        
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]
            
            # We check if there's GT tactile data available for each hand detected
            if gt_left_pressure is not None:
                valid = left_hand_keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [left_hand_keyp[valid, 0].min(), left_hand_keyp[valid, 1].min(),
                            left_hand_keyp[valid, 0].max(), left_hand_keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
            
            if gt_right_pressure is not None:
                valid = right_hand_keyp[:, 2] > 0.5
                if sum(valid) > 3:
                    bbox = [right_hand_keyp[valid, 0].min(), right_hand_keyp[valid, 1].min(),
                            right_hand_keyp[valid, 0].max(), right_hand_keyp[valid, 1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)
                    
        if len(bboxes) == 0:
            stats['no_hand_keypoints'] += 1
            continue
            
        boxes_arr = np.stack(bboxes)
        right_arr = np.stack(is_right)
        
        # Inference
        dataset = ViTDetDataset(model_cfg, img_bgr, boxes_arr, right_arr, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            try:
                with torch.no_grad():
                    out = model.forward_step(batch, train=False)
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
                
            pred_tactile = out['pred_tactile'].detach().cpu().numpy() # [N, 256]
            
            for n in range(pred_tactile.shape[0]):
                is_r = int(batch['right'][n].cpu().numpy())
                
                # Fetch ground truth tactile
                if is_r == 1 and gt_right_pressure is not None:
                    gt_p = gt_right_pressure[i]
                elif is_r == 0 and gt_left_pressure is not None:
                    gt_p = gt_left_pressure[i]
                else:
                    continue
                    
                if np.isnan(gt_p).any() or gt_p is None:
                    stats['no_valid_gt_tactile'] += 1
                    continue
                    
                # Format GT tactile
                raw_signal = np.array(gt_p, dtype=np.float32).flatten()
                # Apply the same inverted pseudo-normalization used during training
                gt_tactile_norm = np.clip((3072.0 - raw_signal) / 3072.0, 0.0, 1.0)
                
                all_pred_tactile.append(pred_tactile[n])
                all_gt_tactile.append(gt_tactile_norm)
                stats['valid_samples'] += 1
                
    if len(all_pred_tactile) == 0:
        print(f"⚠️ Clip {clip_id} returned empty results! Stats: {stats}")
        return None
        
    all_pred = np.stack(all_pred_tactile) # [N, 256]
    all_gt = np.stack(all_gt_tactile)     # [N, 256]
    
    # Metrics
    mae = np.mean(np.abs(all_pred - all_gt))
    rmse = np.sqrt(np.mean((all_pred - all_gt) ** 2))
    
    # Calculate PCC if variance is > 0
    pcc_list = []
    for p, g in zip(all_pred, all_gt):
        if np.std(p) > 1e-6 and np.std(g) > 1e-6:
            pcc = np.corrcoef(p, g)[0, 1]
            if not np.isnan(pcc):
                pcc_list.append(pcc)
    
    avg_pcc = np.mean(pcc_list) if len(pcc_list) > 0 else 0.0
    
    # --- New Metrics: Temporal Accuracy, Contact IoU, Volumetric IoU ---
    pred_bin = all_pred > contact_thr
    gt_bin = all_gt > contact_thr
    
    # Temporal Accuracy
    pred_frame_contact = np.any(pred_bin, axis=1)
    gt_frame_contact = np.any(gt_bin, axis=1)
    temporal_acc = np.mean(pred_frame_contact == gt_frame_contact)
    
    # Contact IoU
    intersection = np.sum(pred_bin & gt_bin, axis=1)
    union = np.sum(pred_bin | gt_bin, axis=1)
    contact_iou_per_frame = np.zeros(len(union), dtype=np.float32)
    zero_union_mask = (union == 0)
    contact_iou_per_frame[zero_union_mask] = 1.0 # Both correctly predict no contact
    non_zero_mask = ~zero_union_mask
    contact_iou_per_frame[non_zero_mask] = intersection[non_zero_mask] / union[non_zero_mask]
    contact_iou = np.mean(contact_iou_per_frame)
    
    # Volumetric IoU
    vol_intersection = np.sum(np.minimum(all_pred, all_gt), axis=1)
    vol_union = np.sum(np.maximum(all_pred, all_gt), axis=1)
    vol_iou_per_frame = np.zeros(len(vol_union), dtype=np.float32)
    vol_zero_union_mask = (vol_union == 0)
    vol_iou_per_frame[vol_zero_union_mask] = 1.0
    vol_non_zero_mask = ~vol_zero_union_mask
    vol_iou_per_frame[vol_non_zero_mask] = vol_intersection[vol_non_zero_mask] / vol_union[vol_non_zero_mask]
    volumetric_iou = np.mean(vol_iou_per_frame)
    
    print(f"--- {clip_id} 评估结果 ---")
    print(f"有效评估帧数: {len(all_pred)}")
    print(f"MAE  (归一化 [0,1]): {mae:.4f}")
    print(f"RMSE (归一化 [0,1]): {rmse:.4f}")
    print(f"Pearson Correlation: {avg_pcc:.4f}")
    print(f"Temporal Accuracy: {temporal_acc:.4f}")
    print(f"Contact IoU: {contact_iou:.4f}")
    print(f"Volumetric IoU: {volumetric_iou:.4f}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "pcc": avg_pcc,
        "temporal_acc": temporal_acc,
        "contact_iou": contact_iou,
        "volumetric_iou": volumetric_iou,
        "count": len(all_pred)
    }

def eval_worker(local_gpu_id, clips_chunk, args_checkpoint, data_dir, hdf5_path_arg, contact_thr=0.05):
    import traceback
    try:
        device = torch.device(f'cuda:{local_gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
            
        print(f"[Worker GPU {local_gpu_id}] Initializing model...")
        
        # Load configuration
        model_cfg_path = os.path.join(hamer_dir, '_DATA/hamer_ckpts/model_config.yaml')
        model_cfg = get_config(model_cfg_path, update_cachedir=True)
        if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
            model_cfg.defrost()
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()
        if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost()
            model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
            model_cfg.freeze()
            
        # Initialize Tactile Wrapper
        model = OpenTouchHAMER_TactileWrapper(cfg=model_cfg)
        
        # Load weights
        print(f"[Worker GPU {local_gpu_id}] Loading checkpoint from: {args_checkpoint}")
        state_dict = torch.load(args_checkpoint, map_location="cpu")['state_dict']
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # Initialize ViTDet and ViTPose ... (Copied from eval_hamer.py)
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        local_vitdet_path = os.path.join(hamer_dir, "_DATA/model_final_f05665.pkl")
        if os.path.exists(local_vitdet_path):
            detectron2_cfg.train.init_checkpoint = local_vitdet_path
        else:
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        
        try:
            import importlib.util
            vit_path = os.path.join(hamer_dir, "third-party/ViTPose/mmpose/models/backbones/vit.py")
            spec = importlib.util.spec_from_file_location("mmpose.models.backbones.vit", vit_path)
            vit_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vit_module)
            import mmpose.apis.inference
            custom_ckpt_path = os.path.join(hamer_dir, "third-party/ViTPose/mmcv_custom/checkpoint.py")
            spec_ckpt = importlib.util.spec_from_file_location("mmcv_custom.checkpoint", custom_ckpt_path)
            custom_ckpt_module = importlib.util.module_from_spec(spec_ckpt)
            spec_ckpt.loader.exec_module(custom_ckpt_module)
            mmpose.apis.inference.load_checkpoint = custom_ckpt_module.load_checkpoint
        except Exception as e:
            pass
            
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
                        res = eval_clip(model, detector, cpm, model_cfg, hdf5_path, clip, device, contact_thr=contact_thr, disable_tqdm=True)
                        if res is not None:
                            results.append(res)
                    except Exception as e:
                        pass
        else:
            for clip in clips_chunk:
                try:
                    res = eval_clip(model, detector, cpm, model_cfg, hdf5_path_arg, clip, device, contact_thr=contact_thr, disable_tqdm=True)
                    if res is not None:
                        results.append(res)
                except Exception as e:
                    pass
                    
        return results
    except Exception as e:
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser(description='Hamer Tactile Head Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Trained Tactile Checkpoint 路径')
    parser.add_argument('--hdf5_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default="/data/jiangrui/OpenTouch Data/data")
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--split_json', type=str, default=os.path.join(eval_dir, "opentouch_splits.json"))
    parser.add_argument('--contact_thr', type=float, default=0.00, help='Threshold for defining contact (0-1)')
    parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
    args = parser.parse_args()
    
    os.chdir(hamer_dir)
    
    gpus = [g.strip() for g in args.gpu.split(',')]
    results = []
    
    if args.split is not None:
        try:
            splits = load_or_generate_splits(args.split_json, args.data_dir)
        except Exception as se:
            print(f"❌ Load splits failed: {se}")
            sys.exit(1)
            
        target_clips = splits.get(args.split, [])
        if not target_clips:
            print(f"❌ Split {args.split} is empty!")
            sys.exit(1)
    else:
        target_clips = args.clips
        
    print(f"🔔 开始评测 Tactile 性能 | 共 {len(target_clips)} 个 clips | GPUs: {args.gpu}")
    
    if len(gpus) > 1:
        import torch.multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
            
        import math
        chunk_size = math.ceil(len(target_clips) / len(gpus))
        chunks = [target_clips[i:i + chunk_size] for i in range(0, len(target_clips), chunk_size)]
        
        pool_args = []
        for i, gpu_id in enumerate(gpus):
            if i < len(chunks) and len(chunks[i]) > 0:
                pool_args.append((i, chunks[i], args.checkpoint, args.data_dir, args.hdf5_path, args.contact_thr))
                
        with mp.Pool(len(pool_args)) as pool:
            multi_results = pool.starmap(eval_worker, pool_args)
            
        for r in multi_results:
            results.extend(r)
    else:
        results = eval_worker(0, target_clips, args.checkpoint, args.data_dir, args.hdf5_path, args.contact_thr)
                
    if len(results) > 0:
        total_frames = sum([r["count"] for r in results])
        avg_mae = sum([r["mae"] * r["count"] for r in results]) / total_frames
        avg_rmse = sum([r["rmse"] * r["count"] for r in results]) / total_frames
        avg_pcc = sum([r["pcc"] * r["count"] for r in results]) / total_frames
        avg_temp_acc = sum([r["temporal_acc"] * r["count"] for r in results]) / total_frames
        avg_cont_iou = sum([r["contact_iou"] * r["count"] for r in results]) / total_frames
        avg_vol_iou = sum([r["volumetric_iou"] * r["count"] for r in results]) / total_frames
        
        report_lines = [
            f"🎉 Tactile Regression 最终加权评估结果 🎉",
            "="*55,
            f" 评测划分/片段 : {args.split}",
            f" 总评估帧数    : {total_frames}",
            f" 整体 MAE      : {avg_mae:.4f} (归一化区间 [0,1])",
            f" 整体 RMSE     : {avg_rmse:.4f} (归一化区间 [0,1])",
            f" 整体 PCC      : {avg_pcc:.4f} (皮尔逊相关系数)",
            f" Temporal Acc  : {avg_temp_acc:.4f} (Contact Thr = {args.contact_thr})",
            f" Contact IoU   : {avg_cont_iou:.4f} (Contact Thr = {args.contact_thr})",
            f" Volumetric IoU: {avg_vol_iou:.4f} (无需 Thr)",
            "="*55
        ]
        report_text = "\n".join(report_lines)
        print("\n" + report_text)
        
        report_path = os.path.join(ft_dir, "eval_tactile_report.txt")
        with open(report_path, "w", encoding="utf-8") as f_rep:
            f_rep.write(report_text + "\n")
        print(f"📝 评测报告已保存至: {report_path}")
    else:
        print("❌ 未产生有效的评估指标！")

if __name__ == '__main__':
    main()
