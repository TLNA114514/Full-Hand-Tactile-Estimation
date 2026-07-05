import sys
import os
import argparse
import traceback
import multiprocessing as mp
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from pathlib import Path
from tqdm import tqdm

# Parse GPU early
_gpus = ""
for i, arg in enumerate(sys.argv):
    if arg in ('--gpu', '--gpus') and i + 1 < len(sys.argv):
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
from train import OpenTouchHAMER_TactileWrapper
from train import load_compatible_state_dict
from train import resolve_data_dirs
from dataset import OpenTouchTactileDataset
from hamer.utils import recursive_to


def _safe_name(value):
    text = str(value or "default").strip().lower()
    chars = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        elif ch in (",", "+", "/", "\\", " "):
            chars.append("_")
    name = "".join(chars).strip("_")
    return name or "default"


def _report_path(args):
    report_dir = args.report_dir or os.path.join(base_dir, "hamer_tactile_ft", "eval_reports")
    os.makedirs(report_dir, exist_ok=True)
    if args.report_name:
        filename = args.report_name
    else:
        dataset_label = args.datasets or args.data_dir or "resolved"
        filename = f"eval_{_safe_name(dataset_label)}_{_safe_name(args.split)}.txt"
    if not filename.endswith(".txt"):
        filename += ".txt"
    return os.path.join(report_dir, filename)


def _gpu_ids(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _load_model_cfg():
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
    return model_cfg


def _load_model(args, model_cfg, device):
    print(f"🚀 初始化模型 (使用设备: {device})...")
    model = OpenTouchHAMER_TactileWrapper(cfg=model_cfg)
    dummy_input = torch.zeros(1, 3, model_cfg.MODEL.IMAGE_SIZE, model_cfg.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        dummy_feat = model.backbone(dummy_input[:, :, :, 32:-32])
        model.tactile_head(dummy_feat)
        print(f"Tactile head initialized with output dim: {model.tactile_dim}")

    print(f"📦 Loading checkpoint from: {args.checkpoint}")
    load_compatible_state_dict(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    return model


def _empty_stats():
    return {
        "total_frames": 0,
        "total_values": 0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "pcc_sum": 0.0,
        "pcc_count": 0,
        "temporal_correct": 0,
        "contact_iou_sum": 0.0,
        "vol_iou_sum": 0.0,
        "tactile_dim": 0,
        "pred_volume": 0.0,
        "gt_volume": 0.0,
        "active_abs_sum": 0.0,
        "active_count": 0,
        "background_abs_sum": 0.0,
        "background_count": 0,
        "active_true_positive": 0,
        "active_gt_count": 0,
        "background_false_positive": 0,
    }


def _merge_stats(items):
    merged = _empty_stats()
    for stats in items:
        for key in merged:
            if key == "tactile_dim":
                merged[key] = max(merged[key], int(stats.get(key, 0)))
            else:
                merged[key] += stats.get(key, 0)
    return merged


def _update_stats(stats, pred_tactile, gt_tactile, palm_mask, contact_thr, active_thr=0.05, background_thr=0.02):
    if pred_tactile.shape[0] == 0:
        return

    stats["tactile_dim"] = max(stats["tactile_dim"], int(pred_tactile.shape[1]))
    if palm_mask is not None:
        pred = pred_tactile[:, palm_mask]
        gt = gt_tactile[:, palm_mask]
    else:
        pred = pred_tactile
        gt = gt_tactile

    diff = pred - gt
    stats["total_frames"] += int(pred.shape[0])
    stats["total_values"] += int(pred.size)
    stats["abs_sum"] += float(np.abs(diff).sum())
    stats["sq_sum"] += float((diff ** 2).sum())
    stats["pred_volume"] += float(pred.sum())
    stats["gt_volume"] += float(gt.sum())

    active_mask = gt > active_thr
    background_mask = gt <= background_thr
    if np.any(active_mask):
        stats["active_abs_sum"] += float(np.abs(diff[active_mask]).sum())
        stats["active_count"] += int(active_mask.sum())
        stats["active_true_positive"] += int(np.sum((pred > active_thr) & active_mask))
        stats["active_gt_count"] += int(active_mask.sum())
    if np.any(background_mask):
        stats["background_abs_sum"] += float(np.abs(diff[background_mask]).sum())
        stats["background_count"] += int(background_mask.sum())
        stats["background_false_positive"] += int(np.sum((pred > active_thr) & background_mask))

    for p, g in zip(pred, gt):
        if np.std(p) > 1e-6 and np.std(g) > 1e-6:
            pcc = np.corrcoef(p, g)[0, 1]
            if not np.isnan(pcc):
                stats["pcc_sum"] += float(pcc)
                stats["pcc_count"] += 1

    pred_bin = pred > contact_thr
    gt_bin = gt > contact_thr
    pred_frame_contact = np.any(pred_bin, axis=1)
    gt_frame_contact = np.any(gt_bin, axis=1)
    stats["temporal_correct"] += int(np.sum(pred_frame_contact == gt_frame_contact))

    intersection = np.sum(pred_bin & gt_bin, axis=1)
    union = np.sum(pred_bin | gt_bin, axis=1)
    contact_iou_per_frame = np.ones(len(union), dtype=np.float32)
    non_zero_mask = union != 0
    contact_iou_per_frame[non_zero_mask] = intersection[non_zero_mask] / union[non_zero_mask]
    stats["contact_iou_sum"] += float(contact_iou_per_frame.sum())

    vol_intersection = np.sum(np.minimum(pred, gt), axis=1)
    vol_union = np.sum(np.maximum(pred, gt), axis=1)
    vol_iou_per_frame = np.ones(len(vol_union), dtype=np.float32)
    vol_non_zero_mask = vol_union != 0
    vol_iou_per_frame[vol_non_zero_mask] = vol_intersection[vol_non_zero_mask] / vol_union[vol_non_zero_mask]
    stats["vol_iou_sum"] += float(vol_iou_per_frame.sum())


def _evaluate_sample_records(args, data_dirs, sample_records, worker_rank=0, show_progress=True):
    if len(sample_records) == 0:
        return _empty_stats()

    os.chdir(os.path.join(base_dir, 'hamer'))
    device = torch.device(f'cuda:{worker_rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(worker_rank)

    model_cfg = _load_model_cfg()
    model = _load_model(args, model_cfg, device)
    dataset = OpenTouchTactileDataset(
        model_cfg,
        split=args.split,
        data_dir=data_dirs,
        train=False,
        sample_records=sample_records,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    stats = _empty_stats()
    palm_mask = None
    iterator = tqdm(dataloader, desc=f"GPU {worker_rank} Evaluating", position=worker_rank) if show_progress else dataloader
    for batch in iterator:
        batch = recursive_to(batch, device)
        valid_tactile_mask = batch['has_tactile'].detach().cpu().numpy() > 0.5
        if not np.any(valid_tactile_mask):
            continue

        if palm_mask is None:
            palm_mask = batch['palm_mask'][0].detach().cpu().numpy() > 0.5

        with torch.no_grad():
            out = model.forward_step(batch, train=False)

        pred_tactile = out['pred_tactile'].detach().cpu().numpy()[valid_tactile_mask]
        gt_tactile = batch['tactile_signal'].detach().cpu().numpy()[valid_tactile_mask]
        _update_stats(
            stats,
            pred_tactile,
            gt_tactile,
            palm_mask,
            args.contact_thr,
            active_thr=args.active_pressure_thr,
            background_thr=args.background_pressure_thr,
        )

    return stats


def _eval_worker(rank, args, data_dirs, sample_records, queue):
    try:
        stats = _evaluate_sample_records(
            args,
            data_dirs,
            sample_records,
            worker_rank=rank,
            show_progress=True,
        )
        queue.put((rank, stats, None))
    except Exception:
        queue.put((rank, None, traceback.format_exc()))


def _format_report(args, stats):
    if stats["total_frames"] == 0 or stats["total_values"] == 0:
        return None

    mae = stats["abs_sum"] / stats["total_values"]
    rmse = np.sqrt(stats["sq_sum"] / stats["total_values"])
    avg_pcc = stats["pcc_sum"] / stats["pcc_count"] if stats["pcc_count"] > 0 else 0.0
    temporal_acc = stats["temporal_correct"] / stats["total_frames"]
    contact_iou = stats["contact_iou_sum"] / stats["total_frames"]
    volumetric_iou = stats["vol_iou_sum"] / stats["total_frames"]
    volume_ratio = stats["pred_volume"] / max(stats["gt_volume"], 1e-6)
    active_mae = stats["active_abs_sum"] / stats["active_count"] if stats["active_count"] > 0 else 0.0
    background_mae = stats["background_abs_sum"] / stats["background_count"] if stats["background_count"] > 0 else 0.0
    active_recall = stats["active_true_positive"] / stats["active_gt_count"] if stats["active_gt_count"] > 0 else 0.0
    false_positive_rate = (
        stats["background_false_positive"] / stats["background_count"] if stats["background_count"] > 0 else 0.0
    )

    report_lines = [
        f"🎉 Tactile Fast Evaluation 最终评估结果 🎉",
        "="*55,
        f" 评测数据集    : {args.datasets or args.data_dir or 'resolved-default'}",
        f" 评测划分集    : {args.split}",
        f" 总有效评估帧数: {stats['total_frames']}",
        f" 触觉输出维度  : {stats['tactile_dim']} (subdiv MANO vertices)",
        f" 整体 MAE      : {mae:.4f} (归一化区间 [0,1])",
        f" 整体 RMSE     : {rmse:.4f} (归一化区间 [0,1])",
        f" 整体 PCC      : {avg_pcc:.4f} (皮尔逊相关系数)",
        f" Temporal Acc  : {temporal_acc:.4f} (Contact Thr = {args.contact_thr})",
        f" Contact IoU   : {contact_iou:.4f} (Contact Thr = {args.contact_thr})",
        f" Volumetric IoU: {volumetric_iou:.4f} (无需 Thr)",
        f" Pred/GT Volume: {volume_ratio:.4f}",
        f" Active MAE    : {active_mae:.4f} (GT > {args.active_pressure_thr})",
        f" Background MAE: {background_mae:.4f} (GT <= {args.background_pressure_thr})",
        f" Active Recall : {active_recall:.4f} (Thr = {args.active_pressure_thr})",
        f" BG False Pos  : {false_positive_rate:.4f} (Pred > {args.active_pressure_thr})",
        "="*55
    ]
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description='Hamer Tactile Fast Evaluation using Extracted Dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Trained Tactile Checkpoint 路径')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Explicit extracted dataset root(s), comma-separated. Appended after --datasets if both are provided.',
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default=None,
        help='Dataset names/aliases, comma-separated: opentouch/ot, touchanything/egotouch/ta, egotactile/ego.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Extracted split directory to evaluate, e.g. train/val/test/test_seen/test_unseen.',
    )
    parser.add_argument('--gpu', '--gpus', dest='gpu', type=str, default='4')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--contact_thr', type=float, default=0.00, help='Threshold for defining contact (0-1)')
    parser.add_argument('--active_pressure_thr', type=float, default=0.05)
    parser.add_argument('--background_pressure_thr', type=float, default=0.02)
    parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
    parser.add_argument('--index_workers', type=int, default=8)
    parser.add_argument('--index_chunksize', type=int, default=512)
    parser.add_argument('--index_cache_dir', type=str, default=os.path.join(base_dir, "hamer_tactile_ft", "index_cache"))
    parser.add_argument('--rebuild_index', action='store_true')
    parser.add_argument('--index_cache_timeout', type=int, default=3600)
    parser.add_argument(
        '--report_dir',
        type=str,
        default=None,
        help='Directory for evaluation reports. Defaults to hamer_tactile_ft/eval_reports.',
    )
    parser.add_argument(
        '--report_name',
        type=str,
        default=None,
        help='Optional report filename. Defaults to eval_{datasets}_{split}.txt.',
    )
    args = parser.parse_args()
    data_dirs = resolve_data_dirs(args)
    print("Resolved evaluation data roots:")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")

    gpu_ids = _gpu_ids(args.gpu)
    world_size = len(gpu_ids) if torch.cuda.is_available() and len(gpu_ids) > 0 else 1

    os.chdir(os.path.join(base_dir, 'hamer'))
    print(f"📦 加载 {args.split} 划分集...")
    model_cfg = _load_model_cfg()
    dataset = OpenTouchTactileDataset(
        model_cfg,
        split=args.split,
        data_dir=data_dirs,
        train=False,
        index_workers=args.index_workers,
        index_chunksize=args.index_chunksize,
        index_cache_dir=args.index_cache_dir,
        rebuild_index=args.rebuild_index,
        index_cache_timeout=args.index_cache_timeout,
    )

    if len(dataset) == 0:
        print("❌ 评估集为空。请检查 --datasets/--data_dir 和 --split。")
        return

    print(f"🔔 开始极速评估推理 | samples={len(dataset)} | GPUs={world_size} | batch_size/GPU={args.batch_size}")
    if world_size <= 1:
        stats = _evaluate_sample_records(args, data_dirs, dataset.samples, worker_rank=0, show_progress=True)
    else:
        shards = [dataset.samples[rank::world_size] for rank in range(world_size)]
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = []
        for rank, shard in enumerate(shards):
            proc = ctx.Process(target=_eval_worker, args=(rank, args, data_dirs, shard, queue))
            proc.start()
            procs.append(proc)

        worker_stats = []
        errors = []
        for _ in procs:
            rank, stats_item, error = queue.get()
            if error:
                errors.append((rank, error))
            else:
                worker_stats.append(stats_item)

        for proc in procs:
            proc.join()

        if errors:
            for rank, error in errors:
                print(f"❌ Worker {rank} failed:\n{error}")
            raise RuntimeError("One or more evaluation workers failed.")
        stats = _merge_stats(worker_stats)

    report_text = _format_report(args, stats)
    if report_text is None:
        print("❌ 未产生任何有效的评估指标！可能数据集中 has_tactile 都是 0。")
        return

    print("\n🧮 推理完成，指标如下：")
    print("\n" + report_text)
    
    report_path = _report_path(args)
    with open(report_path, "w", encoding="utf-8") as f_rep:
        f_rep.write(report_text + "\n")
    print(f"📝 最终评测报告已保存至: {report_path}")

if __name__ == '__main__':
    main()
