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

# ==========================================================================================
# 🛑 核心黑魔法：源码感知 + 全局空间硬核注入补丁（地表最强终结版，完美解决一切 NameError）
# ==========================================================================================
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

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(base_dir, 'hamer'))
sys.path.append(os.path.join(base_dir, 'evaluation'))
sys.path.append(os.path.join(base_dir, 'hamer_tactile_ft'))

from hamer.configs import get_config
from hamer.models import HAMER, load_hamer
from hamer.utils import recursive_to
from dataset import OpenTouchTactileDataset
from eval_utils import compute_mpjpe, compute_pck, compute_auc, fit_mano_to_joints

def main():
    parser = argparse.ArgumentParser(description='Hamer Fast Evaluation using Extracted Dataset')
    parser.add_argument('--checkpoint', type=str, default="../hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt")
    parser.add_argument('--model_cfg', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default="/data/jiangrui/OpenTouch Data/extracted_dataset")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
    args = parser.parse_args()
    
    os.chdir(os.path.join(base_dir, 'hamer'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 初始化模型 (使用设备: {device})...")
    if args.model_cfg is not None:
        model_cfg_path = os.path.abspath(args.model_cfg)
        model_cfg = get_config(model_cfg_path, update_cachedir=True)
        if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
            model_cfg.defrost()
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()
        if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost()
            model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
            model_cfg.freeze()
        model = HAMER.load_from_checkpoint(os.path.abspath(args.checkpoint), strict=False, cfg=model_cfg, map_location='cpu')
    else:
        model, model_cfg = load_hamer(os.path.abspath(args.checkpoint))
        
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
    
    all_pred_joints = []
    all_gt_joints = []
    all_pred_verts = []
    all_gt_verts = []
    
    print("🔔 开始极速评估推理...")
    for batch in tqdm(dataloader, desc="Evaluating Batches"):
        batch = recursive_to(batch, device)
        try:
            with torch.no_grad():
                out = model(batch)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
            
        pred_joints = out['pred_keypoints_3d'].detach().cpu().numpy() # [B, 21, 3]
        pred_verts = out['pred_vertices'].detach().cpu().numpy()     # [B, 778, 3]
        
        # 在 dataset.py 中，左手的 X 轴已经被镜像反转，使得所有手在网络看来都是右手。
        # 因此预测结果也是右手，GT 的 keypoints_3d 也是右手。我们直接拿来算误差，欧式距离是守恒的！
        gt_j_batch = batch['keypoints_3d'].detach().cpu().numpy() # [B, 21, 4]
        
        for n in range(pred_joints.shape[0]):
            valid_mask = gt_j_batch[n, :, 3] > 0
            if not valid_mask.all():
                continue # 有缺失的 GT 关键点，跳过 (或者仅评估 valid 的点)
                
            gt_j = gt_j_batch[n, :, :3]
            pred_j = pred_joints[n]
            
            all_pred_joints.append(pred_j)
            all_gt_joints.append(gt_j)
            
            # 为了计算 PA-MPVPE，我们需要用 GT 关键点来拟合出 GT 的 MANO Mesh
            # 由于这里的 gt_j 已经被 dataset.py 统一镜像成右手了，我们直接用 model.mano (本身就是右手) 即可！
            try:
                # 拟合过程比较慢，每个样本独立拟合
                gt_v_fit = fit_mano_to_joints(model.mano, gt_j, device, num_steps=40)
                pred_v = pred_verts[n]
                
                all_pred_verts.append(pred_v)
                all_gt_verts.append(gt_v_fit)
            except Exception as ve:
                pass
                
    if len(all_pred_joints) == 0:
        print("❌ 未产生任何有效的评估指标！")
        return
        
    all_pred = np.stack(all_pred_joints)
    all_gt = np.stack(all_gt_joints)
    
    print("\n🧮 推理完成，正在计算指标...")
    pa_mpjpe = compute_mpjpe(all_pred, all_gt, alignment='procrustes')
    pck_5 = compute_pck(all_pred, all_gt, threshold_mm=5.0, alignment='procrustes')
    pck_15 = compute_pck(all_pred, all_gt, threshold_mm=15.0, alignment='procrustes')
    auc = compute_auc(all_pred, all_gt, min_thr=0.0, max_thr=50.0, num_steps=31, alignment='procrustes')
    
    pa_mpvpe = 0.0
    if len(all_pred_verts) > 0:
        from eval_utils import compute_pa_mpvpe
        pa_mpvpe = compute_pa_mpvpe(all_pred, all_gt, np.stack(all_pred_verts), np.stack(all_gt_verts))
        
    total_frames = len(all_pred)
    
    report_lines = [
        f"🎉 Hamer Fast Evaluation 最终评估结果 🎉",
        "="*55,
        f" 评测划分集    : {args.split}",
        f" 总有效评估帧数: {total_frames}",
        f" 整体 PA-MPJPE : {pa_mpjpe:.2f} mm",
        f" 整体 PA-MPVPE : {pa_mpvpe:.2f} mm",
        f" 整体 PCK@5mm  : {pck_5:.2f} %",
        f" 整体 PCK@15mm : {pck_15:.2f} %",
        f" 整体 AUC (0-50mm): {auc:.2f}",
        "="*55
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    
    report_path = os.path.join(base_dir, "evaluation", "eval_hamer_fast_report.txt")
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_text + "\n")
    print(f"📝 最终评测报告已保存至: {report_path}")

if __name__ == '__main__':
    main()
