import sys
import os
import argparse
import numpy as np
import torch
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

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
_args, _ = _parser.parse_known_args()

os.environ['PYOPENGL_PLATFORM'] = _args.render_platform
os.environ['PYRENDER_PLATFORM'] = _args.render_platform

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(base_dir, 'hamer'))
sys.path.append(os.path.join(base_dir, 'evaluation'))
sys.path.append(os.path.join(base_dir, 'hamer_tactile_ft'))

from hamer.configs import get_config
from train import OpenTouchHAMER_TactileWrapper
from dataset import OpenTouchTactileDataset
from hamer.utils import recursive_to

def main():
    parser = argparse.ArgumentParser(description='Hamer Tactile Fast Evaluation using Extracted Dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Trained Tactile Checkpoint 路径')
    parser.add_argument('--data_dir', type=str, default="/data/jiangrui/OpenTouch Data/extracted_dataset")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--contact_thr', type=float, default=0.00, help='Threshold for defining contact (0-1)')
    parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
    args = parser.parse_args()
    
    os.chdir(os.path.join(base_dir, 'hamer'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 初始化模型 (使用设备: {device})...")
    model_cfg_path = os.path.join(base_dir, 'hamer/_DATA/hamer_ckpts/model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
        
    model = OpenTouchHAMER_TactileWrapper(cfg=model_cfg)
    print(f"📦 Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"📦 加载 {args.split} 划分集...")
    dataset = OpenTouchTactileDataset(model_cfg, split=args.split, data_dir=args.data_dir, train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )
    
    all_pred_tactile = []
    all_gt_tactile = []
    
    print("🔔 开始极速评估推理...")
    for batch in tqdm(dataloader, desc="Evaluating Batches"):
        batch = recursive_to(batch, device)
        
        # 跳过没有 GT 触觉信号的样本
        valid_tactile_mask = batch['has_tactile'].cpu().numpy() > 0.5
        if not np.any(valid_tactile_mask):
            continue
            
        try:
            with torch.no_grad():
                out = model.forward_step(batch, train=False)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
            
        pred_tactile = out['pred_tactile'].detach().cpu().numpy() # [B, 778]
        gt_tactile = batch['tactile_signal'].detach().cpu().numpy() # [B, 778] 已经被归一化过
        
        for n in range(pred_tactile.shape[0]):
            if valid_tactile_mask[n]:
                all_pred_tactile.append(pred_tactile[n])
                all_gt_tactile.append(gt_tactile[n])
                
    if len(all_pred_tactile) == 0:
        print("❌ 未产生任何有效的评估指标！可能数据集中 has_tactile 都是 0。")
        return
        
    all_pred = np.stack(all_pred_tactile) # [N, 778]
    all_gt = np.stack(all_gt_tactile)     # [N, 778]
    
    print("\n🧮 推理完成，正在计算指标...")
    
    mae = np.mean(np.abs(all_pred - all_gt))
    rmse = np.sqrt(np.mean((all_pred - all_gt) ** 2))
    
    pcc_list = []
    for p, g in zip(all_pred, all_gt):
        if np.std(p) > 1e-6 and np.std(g) > 1e-6:
            pcc = np.corrcoef(p, g)[0, 1]
            if not np.isnan(pcc):
                pcc_list.append(pcc)
    avg_pcc = np.mean(pcc_list) if len(pcc_list) > 0 else 0.0
    
    pred_bin = all_pred > args.contact_thr
    gt_bin = all_gt > args.contact_thr
    
    pred_frame_contact = np.any(pred_bin, axis=1)
    gt_frame_contact = np.any(gt_bin, axis=1)
    temporal_acc = np.mean(pred_frame_contact == gt_frame_contact)
    
    intersection = np.sum(pred_bin & gt_bin, axis=1)
    union = np.sum(pred_bin | gt_bin, axis=1)
    contact_iou_per_frame = np.zeros(len(union), dtype=np.float32)
    zero_union_mask = (union == 0)
    contact_iou_per_frame[zero_union_mask] = 1.0
    non_zero_mask = ~zero_union_mask
    contact_iou_per_frame[non_zero_mask] = intersection[non_zero_mask] / union[non_zero_mask]
    contact_iou = np.mean(contact_iou_per_frame)
    
    vol_intersection = np.sum(np.minimum(all_pred, all_gt), axis=1)
    vol_union = np.sum(np.maximum(all_pred, all_gt), axis=1)
    vol_iou_per_frame = np.zeros(len(vol_union), dtype=np.float32)
    vol_zero_union_mask = (vol_union == 0)
    vol_iou_per_frame[vol_zero_union_mask] = 1.0
    vol_non_zero_mask = ~vol_zero_union_mask
    vol_iou_per_frame[vol_non_zero_mask] = vol_intersection[vol_non_zero_mask] / vol_union[vol_non_zero_mask]
    volumetric_iou = np.mean(vol_iou_per_frame)
    
    total_frames = len(all_pred)
    
    report_lines = [
        f"🎉 Tactile Fast Evaluation 最终评估结果 🎉",
        "="*55,
        f" 评测划分集    : {args.split}",
        f" 总有效评估帧数: {total_frames}",
        f" 整体 MAE      : {mae:.4f} (归一化区间 [0,1])",
        f" 整体 RMSE     : {rmse:.4f} (归一化区间 [0,1])",
        f" 整体 PCC      : {avg_pcc:.4f} (皮尔逊相关系数)",
        f" Temporal Acc  : {temporal_acc:.4f} (Contact Thr = {args.contact_thr})",
        f" Contact IoU   : {contact_iou:.4f} (Contact Thr = {args.contact_thr})",
        f" Volumetric IoU: {volumetric_iou:.4f} (无需 Thr)",
        "="*55
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    report_path = os.path.join(base_dir, "hamer_tactile_ft", "eval_tactile_fast_report.txt")
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_text + "\n")
    print(f"📝 最终评测报告已保存至: {report_path}")

if __name__ == '__main__':
    main()
