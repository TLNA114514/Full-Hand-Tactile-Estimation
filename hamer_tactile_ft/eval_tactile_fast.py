import sys
import os
import argparse
import csv
import json
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
from train import _load_checkpoint
from train import file_sha256
from train import load_compatible_state_dict
from train import resolve_data_dirs
from dataset import OpenTouchTactileDataset
from hamer.utils import recursive_to


DIAG_VALUE_BINS = np.linspace(0.0, 1.0, 101, dtype=np.float32)
DIAG_ERROR_BINS = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
DIAG_PRESSURE_BINS = np.array(
    [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.000001],
    dtype=np.float32,
)
DIAG_PRED_TAIL_THRESHOLDS = np.array(
    [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70],
    dtype=np.float32,
)
FRAME_DIAG_KEYS = (
    "gt_volume",
    "pred_volume",
    "volume_ratio",
    "volumetric_iou",
    "contact_iou",
    "gt_active_vertices",
    "pred_active_vertices",
    "frame_mae",
    "false_high_gt005_pred03_count",
    "false_high_gt005_pred03_excess_volume",
    "false_high_gt005_pred05_count",
    "false_high_gt005_pred05_excess_volume",
    "false_high_gt05_pred03_count",
    "false_high_gt05_pred03_excess_volume",
)
FRAME_PROVENANCE_KEYS = ("sample_dir", "dataset", "hand", "worker_rank")
CATASTROPHIC_OVER_GT_MAX = 10.0
CATASTROPHIC_OVER_PRED_MIN = 300.0
CATASTROPHIC_UNDER_GT_MIN = 150.0
CATASTROPHIC_UNDER_PRED_MAX = 50.0


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
    if args.report_dir:
        report_dir = args.report_dir
    elif args.exp_name:
        report_dir = os.path.join(
            base_dir,
            "hamer_tactile_ft",
            f"eval_reports_{_safe_name(args.exp_name)}_{_safe_name(args.ckpt)}",
        )
    else:
        report_dir = os.path.join(base_dir, "hamer_tactile_ft", "eval_reports")
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


def _checkpoint_exists(path: Path) -> bool:
    # Path.exists() is false for a broken symlink, but lexists lets us report
    # the link target clearly instead of silently ignoring last.ckpt.
    return os.path.lexists(path)


def _validate_checkpoint_candidate(path: Path, label: str) -> str:
    if not _checkpoint_exists(path):
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    if path.is_symlink():
        link_target = os.readlink(path)
        resolved = path.resolve(strict=False)
        print(f"{label} checkpoint is symlink: {path} -> {link_target}")
        if not resolved.is_file():
            raise FileNotFoundError(
                f"{label} checkpoint symlink is broken or does not point to a file: "
                f"{path} -> {link_target} (resolved: {resolved})"
            )
        print(f"{label} checkpoint resolved target: {resolved}")
        return str(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint is not a file: {path}")
    return str(path)


def _raise_if_last_aliases_best(exp_dir: Path, last_ckpt: Path):
    """Reject ambiguous historical runs where `last.ckpt` is the best file."""
    if not last_ckpt.is_symlink():
        return

    last_target = last_ckpt.resolve(strict=False)
    best_candidates = []
    for pattern in ("best_rmse.ckpt", "best_viou.ckpt", "best_tactile_model*.ckpt", "*best*.ckpt"):
        best_candidates.extend(exp_dir.glob(pattern))
    for best_ckpt in best_candidates:
        if _checkpoint_exists(best_ckpt) and best_ckpt.resolve(strict=False) == last_target:
            raise RuntimeError(
                "--ckpt last is not a final-epoch checkpoint for this experiment: "
                f"{last_ckpt} is a symlink to best checkpoint {best_ckpt}. "
                "Refusing to create a misleading 'last' report. Retrain with the current "
                "train.py final-checkpoint fix, or pass an independently saved final checkpoint "
                "via --checkpoint."
            )


def _canonical_checkpoint_selector(selector):
    return "rmse-best" if selector == "best" else selector


def _select_unique_legacy_checkpoint(exp_dir: Path, patterns, label):
    candidates = []
    seen = set()
    for pattern in patterns:
        for candidate in sorted(exp_dir.glob(pattern)):
            candidate_key = candidate.resolve(strict=False) if candidate.is_symlink() else candidate
            if candidate.is_file() and candidate_key not in seen:
                seen.add(candidate_key)
                candidates.append(candidate)
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous legacy {label} checkpoints under {exp_dir}: "
            f"{[path.name for path in candidates]}. Pass --checkpoint explicitly."
        )
    return candidates[0] if candidates else None


def _resolve_checkpoint_path(args):
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        ckpt_path = ckpt_path.resolve(strict=False) if not ckpt_path.is_symlink() else ckpt_path
        selected = _validate_checkpoint_candidate(ckpt_path, "Explicit")
        print(f"Checkpoint selector overridden by --checkpoint: {selected}")
        return selected

    if not args.exp_name:
        raise ValueError(
            "Please provide either --checkpoint /path/to/model.ckpt or "
            "--exp_name <experiment_name> with --ckpt rmse-best|viou-best|last."
        )

    checkpoint_root = Path(args.checkpoint_root).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path(base_dir) / checkpoint_root
    exp_dir = (checkpoint_root / args.exp_name).resolve()
    if not exp_dir.is_dir():
        existing = sorted(path.name for path in checkpoint_root.glob("*") if path.is_dir()) if checkpoint_root.is_dir() else []
        suffix = f" Available experiments: {', '.join(existing[:20])}" if existing else ""
        raise FileNotFoundError(f"Checkpoint experiment directory not found: {exp_dir}.{suffix}")

    selector = _canonical_checkpoint_selector(args.ckpt)
    if args.ckpt == "best":
        print("Checkpoint selector --ckpt best is a compatibility alias for --ckpt rmse-best.")

    canonical_names = {
        "rmse-best": "best_rmse.ckpt",
        "viou-best": "best_viou.ckpt",
        "last": "last.ckpt",
    }
    canonical_path = exp_dir / canonical_names[selector]
    if _checkpoint_exists(canonical_path):
        if selector == "last":
            _raise_if_last_aliases_best(exp_dir, canonical_path)
        selected = _validate_checkpoint_candidate(canonical_path, selector)
        print(f"Checkpoint selector: --ckpt {args.ckpt}; selected canonical checkpoint: {selected}")
        return selected

    if selector == "last":
        legacy = _select_unique_legacy_checkpoint(exp_dir, ("*last*.ckpt",), "last")
    elif selector == "rmse-best":
        legacy = _select_unique_legacy_checkpoint(
            exp_dir,
            ("best_tactile_model.ckpt", "best_tactile_model*.ckpt"),
            "RMSE-best",
        )
    else:
        legacy = None

    if legacy is not None:
        if selector == "last":
            _raise_if_last_aliases_best(exp_dir, legacy)
        selected = _validate_checkpoint_candidate(legacy, f"Legacy {selector}")
        print(f"Checkpoint selector: --ckpt {args.ckpt}; selected legacy checkpoint: {selected}")
        return selected

    available = sorted(path.name for path in exp_dir.glob("*.ckpt"))
    suffix = f" Available checkpoints: {', '.join(available)}" if available else " No .ckpt files found."
    raise FileNotFoundError(f"Could not resolve --ckpt {args.ckpt!r} under {exp_dir}.{suffix}")


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
    experiment_model_config = dict(getattr(args, "model_metadata", {}) or {})
    tactile_head_type = experiment_model_config.get("tactile_head_type", "dense_v2")
    backbone_feature_layers = experiment_model_config.get("backbone_feature_layers", [16, 24, 32])
    visual_backbone = experiment_model_config.get("visual_backbone", "hamer")
    dino_weights = getattr(args, "resolved_backbone_weights", "") if visual_backbone == "dinov3_hplus" else ""
    print(
        f"Tactile config: head={tactile_head_type}, visual_backbone={visual_backbone}, "
        f"backbone_feature_layers={backbone_feature_layers}"
    )
    model = OpenTouchHAMER_TactileWrapper(
        cfg=model_cfg,
        tactile_head_type=tactile_head_type,
        backbone_feature_layers=backbone_feature_layers,
        visual_backbone=visual_backbone,
        dino_weights=dino_weights,
    )
    model.visual_backbone_model_name = experiment_model_config.get("visual_backbone_model_name", "")
    model.backbone_weights_path = getattr(args, "resolved_backbone_weights", "")
    model.backbone_weights_sha256 = getattr(args, "resolved_backbone_sha256", "")
    dummy_input = torch.zeros(1, 3, model_cfg.MODEL.IMAGE_SIZE, model_cfg.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        dummy_feat = model._extract_tactile_features(dummy_input[:, :, :, 32:-32])
        model.tactile_head(dummy_feat)
        print(f"Tactile head initialized with output dim: {model.tactile_dim}")

    print(f"📦 Loading checkpoint from: {args.checkpoint}")
    load_compatible_state_dict(model, args.checkpoint, load_backbone=visual_backbone == "hamer")
    model = model.to(device)
    model.eval()
    return model


def _resolve_experiment_model_metadata(args):
    metadata = {}
    model_config_path = Path(args.checkpoint).parent / "model_config.json"
    if model_config_path.is_file():
        with model_config_path.open("r", encoding="utf-8") as config_file:
            metadata.update(json.load(config_file))

    if not model_config_path.is_file():
        checkpoint = _load_checkpoint(args.checkpoint)
        if isinstance(checkpoint, dict) and checkpoint.get("format") == "tactile_trainable_v2":
            metadata.update(checkpoint.get("model_config", {}))
            for key in (
                "visual_backbone",
                "visual_backbone_model_name",
                "backbone_weights",
                "backbone_sha256",
                "tactile_head_type",
                "backbone_feature_layers",
            ):
                if checkpoint.get(key) not in (None, "", []):
                    metadata[key] = checkpoint[key]

    visual_backbone = str(metadata.get("visual_backbone", "hamer"))
    if visual_backbone == "dinov3_hplus":
        weights_value = args.dino_weights or metadata.get("backbone_weights")
        if not weights_value:
            raise ValueError(
                "DINOv3 evaluation requires --dino_weights or backbone_weights in the compact checkpoint"
            )
    else:
        weights_value = metadata.get("backbone_weights") or os.path.join(
            base_dir, "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
        )

    weights_path = Path(weights_value).expanduser()
    if not weights_path.is_absolute():
        weights_path = Path(args.checkpoint).parent / weights_path
    weights_path = weights_path.resolve(strict=False)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Frozen backbone weights not found: {weights_path}")

    expected_hash = str(metadata.get("backbone_sha256", "") or "")
    actual_hash = file_sha256(weights_path) if expected_hash else ""
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(f"Backbone SHA256 mismatch: expected={expected_hash}, actual={actual_hash}")

    metadata["visual_backbone"] = visual_backbone
    metadata["backbone_weights"] = str(weights_path)
    args.model_metadata = metadata
    args.resolved_backbone_weights = str(weights_path)
    args.resolved_backbone_sha256 = actual_hash or expected_hash


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
        "catastrophic_over_count": 0,
        "catastrophic_over_denominator": 0,
        "catastrophic_under_count": 0,
        "catastrophic_under_denominator": 0,
    }


def _stats_summary(stats):
    if stats["total_frames"] == 0 or stats["total_values"] == 0:
        return None
    total_values = max(stats["total_values"], 1)
    total_frames = max(stats["total_frames"], 1)
    return {
        "mae": stats["abs_sum"] / total_values,
        "rmse": float(np.sqrt(stats["sq_sum"] / total_values)),
        "pcc": stats["pcc_sum"] / max(stats["pcc_count"], 1),
        "contact_iou": stats["contact_iou_sum"] / total_frames,
        "volumetric_iou": stats["vol_iou_sum"] / total_frames,
        "pred_gt_volume_ratio": stats["pred_volume"] / max(stats["gt_volume"], 1e-6),
        "active_recall": stats["active_true_positive"] / max(stats["active_gt_count"], 1),
        "bg_false_positive": stats["background_false_positive"] / max(stats["background_count"], 1),
    }


def _resolve_invocation_path(path):
    """Resolve user-supplied paths before eval changes into the HaMeR directory."""
    if not path:
        return None
    return str(Path(path).expanduser().resolve(strict=False))


def _empty_diagnostics():
    n_value_bins = len(DIAG_VALUE_BINS) - 1
    n_error_bins = len(DIAG_ERROR_BINS) - 1
    n_pressure_bins = len(DIAG_PRESSURE_BINS) - 1
    tail_shape = (n_pressure_bins, len(DIAG_PRED_TAIL_THRESHOLDS))
    return {
        "gt_hist": np.zeros(n_value_bins, dtype=np.float64),
        "pred_hist": np.zeros(n_value_bins, dtype=np.float64),
        "error_hist": np.zeros(n_error_bins, dtype=np.float64),
        "pressure_count": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_gt_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_pred_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_err_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_abs_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_over_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_under_sum": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_pred_active": np.zeros(n_pressure_bins, dtype=np.float64),
        "pressure_gt_active": np.zeros(n_pressure_bins, dtype=np.float64),
        "pred_tail_count": np.zeros(tail_shape, dtype=np.float64),
        "pred_tail_pred_sum": np.zeros(tail_shape, dtype=np.float64),
        "pred_tail_gt_sum": np.zeros(tail_shape, dtype=np.float64),
        "frame": {key: [] for key in FRAME_DIAG_KEYS},
        "frame_provenance": {key: [] for key in FRAME_PROVENANCE_KEYS},
    }


def _empty_eval_result():
    return {
        "stats": _empty_stats(),
        "diagnostics": _empty_diagnostics(),
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


def _merge_diagnostics(items, max_frames):
    merged = _empty_diagnostics()
    for diag in items:
        for key in (
            "gt_hist",
            "pred_hist",
            "error_hist",
            "pressure_count",
            "pressure_gt_sum",
            "pressure_pred_sum",
            "pressure_err_sum",
            "pressure_abs_sum",
            "pressure_over_sum",
            "pressure_under_sum",
            "pressure_pred_active",
            "pressure_gt_active",
            "pred_tail_count",
            "pred_tail_pred_sum",
            "pred_tail_gt_sum",
        ):
            merged[key] += diag.get(key, 0)
        frame = diag.get("frame", {})
        for key in FRAME_DIAG_KEYS:
            values = frame.get(key, [])
            if isinstance(values, list):
                merged["frame"][key].extend(values)
            elif values is not None:
                merged["frame"][key].append(values)
        provenance = diag.get("frame_provenance", {})
        for key in FRAME_PROVENANCE_KEYS:
            values = provenance.get(key, [])
            if isinstance(values, list):
                merged["frame_provenance"][key].extend(values)
            elif values is not None:
                merged["frame_provenance"][key].append(values)

    for key in FRAME_DIAG_KEYS:
        arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in merged["frame"][key]]
        merged["frame"][key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
    for key in FRAME_PROVENANCE_KEYS:
        arrays = [np.asarray(item, dtype=object).reshape(-1) for item in merged["frame_provenance"][key]]
        merged["frame_provenance"][key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=object)

    _trim_frame_diagnostics(merged, max_frames)
    return merged


def _trim_frame_diagnostics(diag, max_frames, seed=2026):
    max_frames = max(0, int(max_frames))
    frame = diag["frame"]
    for key in FRAME_DIAG_KEYS:
        if isinstance(frame[key], list):
            arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in frame[key]]
            frame[key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
    provenance = diag["frame_provenance"]
    for key in FRAME_PROVENANCE_KEYS:
        if isinstance(provenance[key], list):
            arrays = [np.asarray(item, dtype=object).reshape(-1) for item in provenance[key]]
            provenance[key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=object)
    if max_frames <= 0:
        for key in FRAME_DIAG_KEYS:
            frame[key] = np.zeros(0, dtype=np.float32)
        for key in FRAME_PROVENANCE_KEYS:
            provenance[key] = np.zeros(0, dtype=object)
        return
    total = len(np.asarray(frame["gt_volume"]).reshape(-1))
    if total <= max_frames:
        return
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(total, size=max_frames, replace=False))
    for key in FRAME_DIAG_KEYS:
        frame[key] = np.asarray(frame[key], dtype=np.float32)[keep]
    for key in FRAME_PROVENANCE_KEYS:
        provenance[key] = np.asarray(provenance[key], dtype=object)[keep]


def _merge_eval_results(items, max_frames):
    stats = _merge_stats([item["stats"] for item in items])
    diagnostics = _merge_diagnostics([item["diagnostics"] for item in items], max_frames)
    return {"stats": stats, "diagnostics": diagnostics}


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
    pred_volume_per_frame = pred.sum(axis=1)
    gt_volume_per_frame = gt.sum(axis=1)
    catastrophic_over_base = gt_volume_per_frame < CATASTROPHIC_OVER_GT_MAX
    catastrophic_under_base = gt_volume_per_frame >= CATASTROPHIC_UNDER_GT_MIN
    stats["catastrophic_over_count"] += int(
        np.sum(catastrophic_over_base & (pred_volume_per_frame > CATASTROPHIC_OVER_PRED_MIN))
    )
    stats["catastrophic_over_denominator"] += int(np.sum(catastrophic_over_base))
    stats["catastrophic_under_count"] += int(
        np.sum(catastrophic_under_base & (pred_volume_per_frame < CATASTROPHIC_UNDER_PRED_MAX))
    )
    stats["catastrophic_under_denominator"] += int(np.sum(catastrophic_under_base))
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


def _frame_metrics(pred, gt, contact_thr, active_thr=0.05):
    pred_bin = pred > contact_thr
    gt_bin = gt > contact_thr
    intersection = float(np.sum(pred_bin & gt_bin))
    union = float(np.sum(pred_bin | gt_bin))
    contact_iou = 1.0 if union == 0 else intersection / union

    vol_intersection = float(np.minimum(pred, gt).sum())
    vol_union = float(np.maximum(pred, gt).sum())
    vol_iou = 1.0 if vol_union <= 1e-12 else vol_intersection / vol_union
    pred_volume = float(pred.sum())
    gt_volume = float(gt.sum())
    volume_ratio = pred_volume / max(gt_volume, 1e-12)
    metrics = {
        "contact_iou": contact_iou,
        "volumetric_iou": vol_iou,
        "pred_volume": pred_volume,
        "gt_volume": gt_volume,
        "volume_ratio": volume_ratio,
        "gt_active_vertices": float(np.sum(gt > active_thr)),
        "pred_active_vertices": float(np.sum(pred > active_thr)),
        "frame_mae": float(np.mean(np.abs(pred - gt))) if pred.size else 0.0,
    }
    for label, gt_max, pred_min in (
        ("gt005_pred03", 0.005, 0.3),
        ("gt005_pred05", 0.005, 0.5),
        ("gt05_pred03", 0.05, 0.3),
    ):
        mask = (gt < gt_max) & (pred >= pred_min)
        metrics[f"false_high_{label}_count"] = float(np.sum(mask))
        metrics[f"false_high_{label}_excess_volume"] = float(np.maximum(pred - gt, 0.0)[mask].sum())
    return metrics


def _update_diagnostics(
    diagnostics,
    pred_tactile,
    gt_tactile,
    palm_mask,
    contact_thr,
    active_thr,
    max_frames,
    frame_records,
    worker_rank,
):
    if pred_tactile.shape[0] == 0:
        return

    pred = pred_tactile[:, palm_mask] if palm_mask is not None else pred_tactile
    gt = gt_tactile[:, palm_mask] if palm_mask is not None else gt_tactile
    flat_pred = np.clip(pred.reshape(-1).astype(np.float32), 0.0, 1.0)
    flat_gt = np.clip(gt.reshape(-1).astype(np.float32), 0.0, 1.0)
    flat_err = flat_pred - flat_gt

    diagnostics["gt_hist"] += np.histogram(flat_gt, bins=DIAG_VALUE_BINS)[0]
    diagnostics["pred_hist"] += np.histogram(flat_pred, bins=DIAG_VALUE_BINS)[0]
    diagnostics["error_hist"] += np.histogram(flat_err, bins=DIAG_ERROR_BINS)[0]

    bin_idx = np.digitize(flat_gt, DIAG_PRESSURE_BINS, right=False) - 1
    valid_bin = (bin_idx >= 0) & (bin_idx < len(DIAG_PRESSURE_BINS) - 1)
    bin_idx = bin_idx[valid_bin]
    if bin_idx.size:
        gt_values = flat_gt[valid_bin]
        pred_values = flat_pred[valid_bin]
        err_values = pred_values - gt_values
        n_bins = len(DIAG_PRESSURE_BINS) - 1
        diagnostics["pressure_count"] += np.bincount(bin_idx, minlength=n_bins)
        diagnostics["pressure_gt_sum"] += np.bincount(bin_idx, weights=gt_values, minlength=n_bins)
        diagnostics["pressure_pred_sum"] += np.bincount(bin_idx, weights=pred_values, minlength=n_bins)
        diagnostics["pressure_err_sum"] += np.bincount(bin_idx, weights=err_values, minlength=n_bins)
        diagnostics["pressure_abs_sum"] += np.bincount(bin_idx, weights=np.abs(err_values), minlength=n_bins)
        diagnostics["pressure_over_sum"] += np.bincount(bin_idx, weights=np.maximum(err_values, 0.0), minlength=n_bins)
        diagnostics["pressure_under_sum"] += np.bincount(bin_idx, weights=np.maximum(-err_values, 0.0), minlength=n_bins)
        diagnostics["pressure_pred_active"] += np.bincount(
            bin_idx, weights=(pred_values > active_thr).astype(np.float32), minlength=n_bins
        )
        diagnostics["pressure_gt_active"] += np.bincount(
            bin_idx, weights=(gt_values > active_thr).astype(np.float32), minlength=n_bins
        )
        for threshold_index, threshold in enumerate(DIAG_PRED_TAIL_THRESHOLDS):
            tail = pred_values >= threshold
            if not np.any(tail):
                continue
            diagnostics["pred_tail_count"][:, threshold_index] += np.bincount(
                bin_idx[tail], minlength=n_bins
            )
            diagnostics["pred_tail_pred_sum"][:, threshold_index] += np.bincount(
                bin_idx[tail], weights=pred_values[tail], minlength=n_bins
            )
            diagnostics["pred_tail_gt_sum"][:, threshold_index] += np.bincount(
                bin_idx[tail], weights=gt_values[tail], minlength=n_bins
            )

    frame_metrics = [_frame_metrics(p, g, contact_thr, active_thr=active_thr) for p, g in zip(pred, gt)]
    for key in FRAME_DIAG_KEYS:
        diagnostics["frame"][key].append(np.asarray([item[key] for item in frame_metrics], dtype=np.float32))
    if len(frame_records) != len(frame_metrics):
        raise RuntimeError(f"Frame provenance mismatch: {len(frame_records)} records for {len(frame_metrics)} metrics")
    provenance_values = {
        "sample_dir": [str(record.get("sample_dir", "")) for record in frame_records],
        "dataset": [str(record.get("dataset", "")) for record in frame_records],
        "hand": [str(record.get("hand", "")) for record in frame_records],
        "worker_rank": [int(worker_rank)] * len(frame_records),
    }
    for key in FRAME_PROVENANCE_KEYS:
        diagnostics["frame_provenance"][key].append(np.asarray(provenance_values[key], dtype=object))

    # Keep worker-side memory bounded during very large eval runs.
    current_frames = sum(len(np.asarray(item).reshape(-1)) for item in diagnostics["frame"]["gt_volume"])
    if current_frames > max_frames * 2:
        for key in FRAME_DIAG_KEYS:
            diagnostics["frame"][key] = [np.concatenate(diagnostics["frame"][key])]
        for key in FRAME_PROVENANCE_KEYS:
            diagnostics["frame_provenance"][key] = [
                np.concatenate(diagnostics["frame_provenance"][key])
            ]
        _trim_frame_diagnostics(diagnostics, max_frames)
        for key in FRAME_DIAG_KEYS:
            diagnostics["frame"][key] = [diagnostics["frame"][key]]
        for key in FRAME_PROVENANCE_KEYS:
            diagnostics["frame_provenance"][key] = [diagnostics["frame_provenance"][key]]


def _diagnostic_dir(args):
    if args.diagnostics_dir:
        return args.diagnostics_dir
    report_path = _report_path(args)
    base_report = os.path.splitext(os.path.basename(report_path))[0]
    report_dir = os.path.dirname(report_path)
    return os.path.join(report_dir, f"{base_report}_diagnostics")


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_diagnostic_outputs(args, result):
    if not (args.save_diagnostics or args.save_visualizations):
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _diagnostic_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    diag = result["diagnostics"]

    pressure_rows = []
    denom = np.maximum(diag["pressure_count"], 1.0)
    for i in range(len(DIAG_PRESSURE_BINS) - 1):
        pressure_rows.append([
            float(DIAG_PRESSURE_BINS[i]),
            float(DIAG_PRESSURE_BINS[i + 1]),
            int(diag["pressure_count"][i]),
            float(diag["pressure_gt_sum"][i] / denom[i]),
            float(diag["pressure_pred_sum"][i] / denom[i]),
            float(diag["pressure_err_sum"][i] / denom[i]),
            float(diag["pressure_abs_sum"][i] / denom[i]),
            float(diag["pressure_over_sum"][i] / denom[i]),
            float(diag["pressure_under_sum"][i] / denom[i]),
            float(diag["pressure_pred_active"][i] / denom[i]),
            float(diag["pressure_gt_active"][i] / denom[i]),
        ])
    _write_csv(
        os.path.join(out_dir, "pressure_bins.csv"),
        [
            "gt_bin_low",
            "gt_bin_high",
            "count",
            "mean_gt",
            "mean_pred",
            "mean_error_pred_minus_gt",
            "mae",
            "mean_over_prediction",
            "mean_under_prediction",
            "pred_active_rate",
            "gt_active_rate",
        ],
        pressure_rows,
    )

    tail_rows = []
    for bin_index in range(len(DIAG_PRESSURE_BINS) - 1):
        gt_count = int(diag["pressure_count"][bin_index])
        for threshold_index, threshold in enumerate(DIAG_PRED_TAIL_THRESHOLDS):
            tail_count = int(diag["pred_tail_count"][bin_index, threshold_index])
            tail_denom = max(tail_count, 1)
            mean_tail_pred = float(
                diag["pred_tail_pred_sum"][bin_index, threshold_index] / tail_denom
            )
            mean_tail_gt = float(
                diag["pred_tail_gt_sum"][bin_index, threshold_index] / tail_denom
            )
            tail_rows.append([
                float(DIAG_PRESSURE_BINS[bin_index]),
                float(DIAG_PRESSURE_BINS[bin_index + 1]),
                gt_count,
                float(threshold),
                tail_count,
                float(tail_count / max(gt_count, 1)),
                mean_tail_gt,
                mean_tail_pred,
                mean_tail_pred - mean_tail_gt,
            ])
    _write_csv(
        os.path.join(out_dir, "pointwise_pressure_tails.csv"),
        [
            "gt_bin_low",
            "gt_bin_high",
            "gt_count",
            "pred_threshold",
            "pred_at_or_above_count",
            "pred_at_or_above_rate",
            "mean_gt_in_tail",
            "mean_pred_in_tail",
            "mean_overprediction_in_tail",
        ],
        tail_rows,
    )

    false_high_rows = []
    total_pred_volume = max(float(result["stats"].get("pred_volume", 0.0)), 1e-12)
    for gt_max in (0.005, 0.02, 0.05, 0.10, 0.20):
        gt_bins = np.flatnonzero(DIAG_PRESSURE_BINS[1:] <= gt_max + 1e-8)
        gt_count = int(diag["pressure_count"][gt_bins].sum())
        for threshold_index, threshold in enumerate(DIAG_PRED_TAIL_THRESHOLDS):
            if threshold <= gt_max:
                continue
            tail_count = int(diag["pred_tail_count"][gt_bins, threshold_index].sum())
            tail_denom = max(tail_count, 1)
            mean_tail_pred = float(
                diag["pred_tail_pred_sum"][gt_bins, threshold_index].sum() / tail_denom
            )
            mean_tail_gt = float(
                diag["pred_tail_gt_sum"][gt_bins, threshold_index].sum() / tail_denom
            )
            false_high_pred_volume = float(
                diag["pred_tail_pred_sum"][gt_bins, threshold_index].sum()
            )
            false_high_gt_volume = float(
                diag["pred_tail_gt_sum"][gt_bins, threshold_index].sum()
            )
            false_high_rows.append([
                float(gt_max),
                float(threshold),
                gt_count,
                tail_count,
                float(tail_count / max(gt_count, 1)),
                mean_tail_gt,
                mean_tail_pred,
                mean_tail_pred - mean_tail_gt,
                false_high_pred_volume,
                false_high_pred_volume - false_high_gt_volume,
                false_high_pred_volume / total_pred_volume,
            ])
    _write_csv(
        os.path.join(out_dir, "false_high_pressure_summary.csv"),
        [
            "gt_below",
            "pred_at_or_above",
            "gt_count",
            "false_high_count",
            "false_high_rate",
            "mean_gt_in_false_high",
            "mean_pred_in_false_high",
            "mean_overprediction_in_false_high",
            "false_high_pred_volume",
            "false_high_excess_volume",
            "fraction_of_total_pred_volume",
        ],
        false_high_rows,
    )

    frame = {key: np.asarray(diag["frame"][key], dtype=np.float32) for key in FRAME_DIAG_KEYS}
    provenance = {
        key: np.asarray(diag["frame_provenance"][key], dtype=object)
        for key in FRAME_PROVENANCE_KEYS
    }
    frame_rows = zip(
        *(provenance[key] for key in FRAME_PROVENANCE_KEYS),
        *(frame[key] for key in FRAME_DIAG_KEYS),
    )
    _write_csv(
        os.path.join(out_dir, "frame_metrics_sample.csv"),
        list(FRAME_PROVENANCE_KEYS) + list(FRAME_DIAG_KEYS),
        frame_rows,
    )

    value_centers = 0.5 * (DIAG_VALUE_BINS[:-1] + DIAG_VALUE_BINS[1:])
    error_centers = 0.5 * (DIAG_ERROR_BINS[:-1] + DIAG_ERROR_BINS[1:])
    pressure_centers = 0.5 * (DIAG_PRESSURE_BINS[:-1] + DIAG_PRESSURE_BINS[1:])
    count = diag["pressure_count"]
    valid = count > 0

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    ax.plot(value_centers, diag["gt_hist"], label="GT", linewidth=1.8)
    ax.plot(value_centers, diag["pred_hist"], label="Pred", linewidth=1.8)
    ax.axvline(args.background_pressure_thr, color="gray", linestyle="--", linewidth=1.0, label="background thr")
    ax.axvline(args.active_pressure_thr, color="black", linestyle="--", linewidth=1.0, label="active thr")
    ax.set_yscale("log")
    ax.set_xlabel("pressure value")
    ax.set_ylabel("vertex count, log scale")
    ax.set_title("Point Pressure Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pressure_distribution.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=160)
    ax.bar(error_centers, diag["error_hist"], width=np.diff(DIAG_ERROR_BINS), align="center")
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("pred - gt")
    ax.set_ylabel("vertex count, log scale")
    ax.set_title("Point Error Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_distribution.png"))
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    mean_gt = np.divide(diag["pressure_gt_sum"], denom)
    mean_pred = np.divide(diag["pressure_pred_sum"], denom)
    mean_err = np.divide(diag["pressure_err_sum"], denom)
    ax1.plot(pressure_centers[valid], mean_gt[valid], marker="o", label="mean GT")
    ax1.plot(pressure_centers[valid], mean_pred[valid], marker="o", label="mean Pred")
    ax1.plot(pressure_centers[valid], mean_err[valid], marker="o", label="mean Pred-GT")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_xlabel("GT pressure bin center")
    ax1.set_ylabel("pressure / error")
    ax2 = ax1.twinx()
    ax2.bar(pressure_centers[valid], count[valid], width=np.diff(DIAG_PRESSURE_BINS)[valid] * 0.75, alpha=0.18, color="gray", label="count")
    ax2.set_yscale("log")
    ax2.set_ylabel("vertex count, log scale")
    ax1.set_title("Calibration By GT Pressure")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_by_gt_pressure.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    mean_over = np.divide(diag["pressure_over_sum"], denom)
    mean_under = np.divide(diag["pressure_under_sum"], denom)
    ax.plot(pressure_centers[valid], mean_over[valid], marker="o", label="over: max(pred-gt, 0)")
    ax.plot(pressure_centers[valid], mean_under[valid], marker="o", label="under: max(gt-pred, 0)")
    ax.plot(pressure_centers[valid], np.divide(diag["pressure_abs_sum"], denom)[valid], marker="o", label="MAE")
    ax.set_xlabel("GT pressure bin center")
    ax.set_ylabel("mean error magnitude")
    ax.set_title("Over / Under Prediction By GT Pressure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "over_under_by_gt_pressure.png"))
    plt.close(fig)

    if len(frame["gt_volume"]) > 0:
        fig, ax = plt.subplots(figsize=(6.5, 6.0), dpi=160)
        scatter = ax.scatter(
            frame["gt_volume"],
            frame["pred_volume"],
            c=frame["volumetric_iou"],
            s=4,
            alpha=0.35,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )
        upper = float(np.percentile(np.concatenate([frame["gt_volume"], frame["pred_volume"]]), 99.5))
        upper = max(upper, 1e-6)
        ax.plot([0, upper], [0, upper], color="black", linewidth=1.0, linestyle="--")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("GT volume per frame")
        ax.set_ylabel("Pred volume per frame")
        ax.set_title("Frame Volume: Pred vs GT")
        fig.colorbar(scatter, ax=ax, label="Volumetric IoU")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "frame_volume_pred_vs_gt.png"))
        plt.close(fig)

        _write_frame_binned_outputs(out_dir, frame, provenance, args)

    return out_dir


def _write_frame_binned_outputs(out_dir, frame, provenance, args):
    import matplotlib.pyplot as plt

    gt_volume = frame["gt_volume"]
    positive = gt_volume > 1e-8
    if len(gt_volume) == 0:
        return

    rows = []
    centers = []
    viou_means = []
    ratio_medians = []
    contact_means = []
    frame_mae_means = []
    active_ratio_medians = []
    empty_bins = []

    def add_frame_bin(low, high, mask):
        if not np.any(mask):
            return
        centers.append(float(0.5 * (low + high)))
        empty_bins.append(bool(high <= 1e-8))
        viou_means.append(float(np.mean(frame["volumetric_iou"][mask])))
        ratio_medians.append(float(np.median(frame["volume_ratio"][mask])))
        contact_means.append(float(np.mean(frame["contact_iou"][mask])))
        frame_mae_means.append(float(np.mean(frame["frame_mae"][mask])))
        active_ratio = frame["pred_active_vertices"][mask] / np.maximum(frame["gt_active_vertices"][mask], 1.0)
        active_ratio_medians.append(float(np.median(active_ratio)))
        rows.append([
            float(low),
            float(high),
            int(np.sum(mask)),
            float(np.mean(frame["gt_volume"][mask])),
            float(np.mean(frame["pred_volume"][mask])),
            ratio_medians[-1],
            viou_means[-1],
            contact_means[-1],
            frame_mae_means[-1],
            active_ratio_medians[-1],
        ])

    zero_mask = gt_volume <= 1e-8
    add_frame_bin(0.0, 0.0, zero_mask)

    if np.sum(positive) >= 2:
        quantiles = np.linspace(0.0, 1.0, 11)
        edges = np.unique(np.quantile(gt_volume[positive], quantiles))
        if len(edges) >= 3:
            edges[0] = max(0.0, edges[0] - 1e-8)
            edges[-1] = edges[-1] + 1e-8
            bin_idx = np.digitize(gt_volume, edges, right=False) - 1
            for i in range(len(edges) - 1):
                add_frame_bin(edges[i], edges[i + 1], bin_idx == i)

    if not rows:
        return

    _write_csv(
        os.path.join(out_dir, "frame_bins_by_gt_volume.csv"),
        [
            "gt_volume_bin_low",
            "gt_volume_bin_high",
            "frame_count",
            "mean_gt_volume",
            "mean_pred_volume",
            "median_pred_gt_volume_ratio",
            "mean_volumetric_iou",
            "mean_contact_iou",
            "mean_frame_mae",
            "median_pred_gt_active_vertex_ratio",
        ],
        rows,
    )

    catastrophic_over_base = gt_volume < CATASTROPHIC_OVER_GT_MAX
    catastrophic_under_base = gt_volume >= CATASTROPHIC_UNDER_GT_MIN
    catastrophic_rows = [
        [
            "over",
            f"gt_volume < {CATASTROPHIC_OVER_GT_MAX:g} and pred_volume > {CATASTROPHIC_OVER_PRED_MIN:g}",
            int(np.sum(catastrophic_over_base & (frame["pred_volume"] > CATASTROPHIC_OVER_PRED_MIN))),
            int(np.sum(catastrophic_over_base)),
            float(
                np.sum(catastrophic_over_base & (frame["pred_volume"] > CATASTROPHIC_OVER_PRED_MIN))
                / max(np.sum(catastrophic_over_base), 1)
            ),
        ],
        [
            "under",
            f"gt_volume >= {CATASTROPHIC_UNDER_GT_MIN:g} and pred_volume < {CATASTROPHIC_UNDER_PRED_MAX:g}",
            int(np.sum(catastrophic_under_base & (frame["pred_volume"] < CATASTROPHIC_UNDER_PRED_MAX))),
            int(np.sum(catastrophic_under_base)),
            float(
                np.sum(catastrophic_under_base & (frame["pred_volume"] < CATASTROPHIC_UNDER_PRED_MAX))
                / max(np.sum(catastrophic_under_base), 1)
            ),
        ],
    ]
    _write_csv(
        os.path.join(out_dir, "catastrophic_frame_diagnostics.csv"),
        ["type", "condition", "count", "denominator", "rate"],
        catastrophic_rows,
    )

    over_mask = catastrophic_over_base & (frame["pred_volume"] > CATASTROPHIC_OVER_PRED_MIN)
    under_mask = catastrophic_under_base & (frame["pred_volume"] < CATASTROPHIC_UNDER_PRED_MAX)
    sample_header = ["type"] + list(FRAME_PROVENANCE_KEYS) + list(FRAME_DIAG_KEYS)
    sample_rows = []
    for condition, mask in (("over", over_mask), ("under", under_mask)):
        for index in np.flatnonzero(mask):
            sample_rows.append(
                [condition]
                + [provenance[key][index] for key in FRAME_PROVENANCE_KEYS]
                + [frame[key][index] for key in FRAME_DIAG_KEYS]
            )
    _write_csv(
        os.path.join(out_dir, "catastrophic_frame_samples.csv"),
        sample_header,
        sample_rows,
    )

    false_high_score = (
        frame["false_high_gt005_pred03_excess_volume"]
        + frame["false_high_gt05_pred03_excess_volume"]
    )
    ranked = np.argsort(-false_high_score)
    ranked = ranked[false_high_score[ranked] > 0][:500]
    top_rows = [
        [rank]
        + [provenance[key][index] for key in FRAME_PROVENANCE_KEYS]
        + [frame[key][index] for key in FRAME_DIAG_KEYS]
        for rank, index in enumerate(ranked, start=1)
    ]
    _write_csv(
        os.path.join(out_dir, "top_false_high_frames.csv"),
        ["rank"] + list(FRAME_PROVENANCE_KEYS) + list(FRAME_DIAG_KEYS),
        top_rows,
    )

    if np.any(zero_mask):
        empty_pred_volume = float(np.mean(frame["pred_volume"][zero_mask]))
        empty_pred_active = float(np.mean(frame["pred_active_vertices"][zero_mask]))
        _write_csv(
            os.path.join(out_dir, "empty_frame_diagnostics.csv"),
            ["frame_count", "mean_pred_volume", "mean_pred_active_vertices"],
            [[int(np.sum(zero_mask)), empty_pred_volume, empty_pred_active]],
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), dpi=160)
        axes[0].bar(["empty frames"], [empty_pred_volume], color="tab:red")
        axes[0].set_ylabel("mean predicted volume")
        axes[1].bar(["empty frames"], [empty_pred_active], color="tab:purple")
        axes[1].set_ylabel("mean predicted active vertices")
        fig.suptitle(f"Empty Frame Diagnostics (n={int(np.sum(zero_mask))})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "empty_frame_diagnostics.png"))
        plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9.0, 5.2), dpi=160)
    ax1.plot(centers, viou_means, marker="o", label="mean V-IoU")
    ax1.plot(centers, contact_means, marker="o", label="mean Contact IoU")
    ax1.set_xlabel("GT volume bin center")
    ax1.set_ylabel("IoU")
    ax1.set_ylim(0.0, 1.0)
    ax2 = ax1.twinx()
    ratio_plot = np.asarray(ratio_medians, dtype=np.float64)
    active_ratio_plot = np.asarray(active_ratio_medians, dtype=np.float64)
    empty_bins = np.asarray(empty_bins, dtype=bool)
    ratio_plot[empty_bins] = np.nan
    active_ratio_plot[empty_bins] = np.nan
    ax2.plot(centers, ratio_plot, marker="o", color="tab:red", label="median Pred/GT volume")
    ax2.plot(centers, active_ratio_plot, marker="o", color="tab:purple", label="median Pred/GT active vertices")
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax2.set_ylabel("ratio")
    ax1.set_title("Frame Metrics By GT Volume")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frame_metrics_by_gt_volume.png"))
    plt.close(fig)


def _evaluate_sample_records(args, data_dirs, sample_records, worker_rank=0, show_progress=True):
    if len(sample_records) == 0:
        return _empty_eval_result()

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
        tactile_only=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    stats = _empty_stats()
    diagnostics = _empty_diagnostics()
    palm_mask = None
    sample_cursor = 0
    iterator = tqdm(dataloader, desc=f"GPU {worker_rank} Evaluating", position=worker_rank) if show_progress else dataloader
    for batch in iterator:
        raw_batch_size = len(batch["dataset"]) if isinstance(batch.get("dataset"), (list, tuple)) else int(batch["img"].shape[0])
        batch_records = sample_records[sample_cursor:sample_cursor + raw_batch_size]
        sample_cursor += raw_batch_size

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
        valid_records = [
            record for record, is_valid in zip(batch_records, valid_tactile_mask) if is_valid
        ]
        _update_stats(
            stats,
            pred_tactile,
            gt_tactile,
            palm_mask,
            args.contact_thr,
            active_thr=args.active_pressure_thr,
            background_thr=args.background_pressure_thr,
        )
        if args.save_diagnostics or args.save_visualizations:
            _update_diagnostics(
                diagnostics,
                pred_tactile,
                gt_tactile,
                palm_mask,
                args.contact_thr,
                args.active_pressure_thr,
                args.diagnostic_max_frames,
                valid_records,
                worker_rank,
            )
    if args.save_diagnostics or args.save_visualizations:
        for key in FRAME_DIAG_KEYS:
            arrays = [np.asarray(item, dtype=np.float32).reshape(-1) for item in diagnostics["frame"][key]]
            diagnostics["frame"][key] = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
        for key in FRAME_PROVENANCE_KEYS:
            arrays = [
                np.asarray(item, dtype=object).reshape(-1)
                for item in diagnostics["frame_provenance"][key]
            ]
            diagnostics["frame_provenance"][key] = (
                np.concatenate(arrays) if arrays else np.zeros(0, dtype=object)
            )
        _trim_frame_diagnostics(diagnostics, args.diagnostic_max_frames, seed=2026 + int(worker_rank))

    return {"stats": stats, "diagnostics": diagnostics}


def _eval_worker(rank, args, data_dirs, sample_records, queue):
    try:
        result = _evaluate_sample_records(
            args,
            data_dirs,
            sample_records,
            worker_rank=rank,
            show_progress=True,
        )
        queue.put((rank, result, None))
    except Exception:
        queue.put((rank, None, traceback.format_exc()))


def _loss_config_summary(args):
    if not args.exp_name:
        return "N/A"
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path(base_dir) / checkpoint_root
    loss_path = checkpoint_root / args.exp_name / "loss_config.json"
    if not loss_path.is_file():
        return f"missing ({loss_path})"
    try:
        with loss_path.open("r", encoding="utf-8") as f:
            loss_config = json.load(f)
    except Exception as exc:
        return f"unreadable ({loss_path}: {exc})"
    compact = json.dumps(loss_config, sort_keys=True, ensure_ascii=False)
    return f"{loss_path} | {compact}"


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
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.is_symlink():
        checkpoint_target = os.readlink(checkpoint_path)
    else:
        checkpoint_target = ""

    report_lines = [
        f"🎉 Tactile Fast Evaluation 最终评估结果 🎉",
        "="*55,
        f" 评测数据集    : {args.datasets or args.data_dir or 'resolved-default'}",
        f" 评测划分集    : {args.split}",
        f" 实验名        : {args.exp_name or 'explicit-checkpoint'}",
        f" Checkpoint    : {args.ckpt}",
        f" Resolved Ckpt : {args.checkpoint}",
        f" Symlink Target: {checkpoint_target or 'N/A'}",
        f" Head Type     : {(getattr(args, 'model_metadata', {}) or {}).get('tactile_head_type', 'dense_v2')}",
        " Pool Layout   : legacy5 / 5x5 (21 retained cells)",
        f" Loss Config   : {_loss_config_summary(args)}",
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
        (
            " Catastrophic Over: "
            f"{stats['catastrophic_over_count']}/{stats['catastrophic_over_denominator']} "
            f"({stats['catastrophic_over_count'] / max(stats['catastrophic_over_denominator'], 1):.6f}; "
            f"GT volume < {CATASTROPHIC_OVER_GT_MAX:g}, "
            f"Pred volume > {CATASTROPHIC_OVER_PRED_MIN:g})"
        ),
        (
            " Catastrophic Under: "
            f"{stats['catastrophic_under_count']}/{stats['catastrophic_under_denominator']} "
            f"({stats['catastrophic_under_count'] / max(stats['catastrophic_under_denominator'], 1):.6f}; "
            f"GT volume >= {CATASTROPHIC_UNDER_GT_MIN:g}, "
            f"Pred volume < {CATASTROPHIC_UNDER_PRED_MAX:g})"
        ),
        "="*55
    ]
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description='Hamer Tactile Fast Evaluation using Extracted Dataset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Explicit trained tactile checkpoint path. Overrides --exp_name/--ckpt.')
    parser.add_argument(
        '--dino_weights',
        type=str,
        default=None,
        help='Optional local DINOv3 weights override; compact checkpoint metadata is used by default.',
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='mixed_dense_v2_repro',
        help='Experiment name under --checkpoint_root.',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='rmse-best',
        choices=['rmse-best', 'viou-best', 'last', 'best'],
        help='Checkpoint selector used with --exp_name. "best" is a compatibility alias for "rmse-best".',
    )
    parser.add_argument(
        '--checkpoint_root',
        type=str,
        default=os.path.join(base_dir, "hamer_tactile_ft", "checkpoints"),
        help='Root directory containing experiment checkpoint folders.',
    )
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
    parser.add_argument('--contact_thr', type=float, default=0.05, help='Threshold for defining contact (0-1)')
    parser.add_argument('--active_pressure_thr', type=float, default=0.05)
    parser.add_argument('--background_pressure_thr', type=float, default=0.02)
    parser.add_argument('--render_platform', type=str, default='egl', choices=['egl', 'osmesa'], help='Rendering platform (egl or osmesa)')
    parser.add_argument('--index_workers', type=int, default=128)
    parser.add_argument('--index_backend', type=str, default='process', choices=['process', 'thread'])
    parser.add_argument('--index_chunksize', type=int, default=512)
    parser.add_argument('--index_cache_dir', type=str, default=os.path.join(base_dir, "hamer_tactile_ft", "index_cache"))
    parser.add_argument('--rebuild_index', action='store_true')
    parser.add_argument('--index_cache_timeout', type=int, default=3600)
    parser.add_argument(
        '--report_dir',
        type=str,
        default=None,
        help='Directory for evaluation reports. Defaults to eval_reports_{exp_name}_{ckpt} when --exp_name is set.',
    )
    parser.add_argument(
        '--report_name',
        type=str,
        default=None,
        help='Optional report filename. Defaults to eval_{datasets}_{split}.txt.',
    )
    parser.add_argument('--save_diagnostics', action='store_true', help='Save dataset-level distribution diagnostics and plots.')
    parser.add_argument(
        '--save_visualizations',
        action='store_true',
        help='Deprecated alias for --save_diagnostics. Saves dataset-level diagnostic plots, not per-frame MANO projections.',
    )
    parser.add_argument('--diagnostic_max_frames', type=int, default=200000, help='Maximum frame-level samples kept for diagnostic scatter plots/CSV.')
    parser.add_argument('--diagnostics_dir', type=str, default=None, help='Directory for diagnostic CSV/PNG outputs. Defaults next to the eval report.')
    args = parser.parse_args()

    # Worker processes call os.chdir(<workspace>/hamer), so resolve every
    # user-facing filesystem input while the launch directory is still intact.
    for name in ("dino_weights", "report_dir", "diagnostics_dir", "index_cache_dir"):
        value = getattr(args, name, None)
        if value:
            setattr(args, name, _resolve_invocation_path(value))
    args.checkpoint = _resolve_checkpoint_path(args)
    _resolve_experiment_model_metadata(args)
    print(f"Resolved checkpoint: {args.checkpoint}")

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
        index_backend=args.index_backend,
        index_cache_dir=args.index_cache_dir,
        rebuild_index=args.rebuild_index,
        index_cache_timeout=args.index_cache_timeout,
    )

    if len(dataset) == 0:
        print("❌ 评估集为空。请检查 --datasets/--data_dir 和 --split。")
        return

    print(f"🔔 开始极速评估推理 | samples={len(dataset)} | GPUs={world_size} | batch_size/GPU={args.batch_size}")
    if world_size <= 1:
        result = _evaluate_sample_records(args, data_dirs, dataset.samples, worker_rank=0, show_progress=True)
    else:
        shards = [dataset.samples[rank::world_size] for rank in range(world_size)]
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = []
        for rank, shard in enumerate(shards):
            proc = ctx.Process(target=_eval_worker, args=(rank, args, data_dirs, shard, queue))
            proc.start()
            procs.append(proc)

        worker_results = []
        errors = []
        for _ in procs:
            rank, result_item, error = queue.get()
            if error:
                errors.append((rank, error))
            else:
                worker_results.append(result_item)

        for proc in procs:
            proc.join()

        if errors:
            for rank, error in errors:
                print(f"❌ Worker {rank} failed:\n{error}")
            raise RuntimeError("One or more evaluation workers failed.")
        result = _merge_eval_results(worker_results, args.diagnostic_max_frames)

    stats = result["stats"]

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

    diagnostics_dir = _write_diagnostic_outputs(args, result)
    if diagnostics_dir:
        print(f"📊 Volumetric IoU / prediction distribution diagnostics 已保存至: {diagnostics_dir}")

    checkpoint_path = Path(args.checkpoint)
    checkpoint_root = Path(args.checkpoint_root).expanduser()
    if not checkpoint_root.is_absolute():
        checkpoint_root = Path(base_dir) / checkpoint_root
    exp_dir = checkpoint_root / args.exp_name if args.exp_name else None

    def read_json_if_exists(path):
        if path is None or not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    loss_config = read_json_if_exists(exp_dir / "loss_config.json") if exp_dir is not None else None
    model_config = read_json_if_exists(exp_dir / "model_config.json") if exp_dir is not None else None
    eval_config = {
        "args": vars(args),
        "exp_name": args.exp_name,
        "ckpt": args.ckpt,
        "ckpt_canonical": _canonical_checkpoint_selector(args.ckpt),
        "resolved_checkpoint": args.checkpoint,
        "checkpoint_symlink_target": os.readlink(checkpoint_path) if checkpoint_path.is_symlink() else None,
        "head_type": model_config.get("tactile_head_type", "dense_v2") if model_config else "dense_v2",
        "loss_config": loss_config,
        "model_config": model_config,
        "report_path": report_path,
        "diagnostics_dir": diagnostics_dir,
    }
    with open(os.path.join(os.path.dirname(report_path), "eval_config.json"), "w", encoding="utf-8") as f_cfg:
        json.dump(eval_config, f_cfg, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"🧾 Eval config 已保存至: {os.path.join(os.path.dirname(report_path), 'eval_config.json')}")

if __name__ == '__main__':
    main()
