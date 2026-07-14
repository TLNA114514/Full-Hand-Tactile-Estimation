#!/usr/bin/env python
import os
import sys
import argparse
import hashlib
import json
import math
import shlex
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

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

sys.argv[0] = os.path.abspath(__file__)

import faulthandler
faulthandler.enable()

import argparse
import pytorch_lightning as pl
import cv2

cv2.setNumThreads(0)

from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))
sys.path.append(ft_dir)

from dataset import OpenTouchTactileDataset
from hamer_tactile import HAMER_Tactile
from losses import TactileLossConfig

DATASET_ROOTS = {
    "opentouch": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "open_touch": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "ot": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "touchanything": "/data1/jiangrui/EgoTouch/extracted_frames",
    "touch_anything": "/data1/jiangrui/EgoTouch/extracted_frames",
    "egotouch": "/data1/jiangrui/EgoTouch/extracted_frames",
    "ego_touch": "/data1/jiangrui/EgoTouch/extracted_frames",
    "ta": "/data1/jiangrui/EgoTouch/extracted_frames",
    "egotactile": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
    "ego_tactile": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
    "ego": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
}

def linear_scaled_learning_rate(base_lr, num_gpus):
    if int(num_gpus) < 1:
        raise ValueError("num_gpus must be at least 1")
    return float(base_lr) * int(num_gpus)


def gradient_clip_triggered(grad_norm, clip_value):
    return float(not math.isfinite(float(grad_norm)) or (clip_value > 0.0 and grad_norm > clip_value))


class ValidationMetricsTextLogger(Callback):
    TRAIN_EPOCH_KEYS = {
        "train/epoch_grad_norm_mean",
        "train/epoch_grad_norm_max",
        "train/grad_clip_trigger_rate_epoch",
        "train/nonfinite_grad_rate_epoch",
        "train/effective_lr_epoch_end",
    }

    def __init__(self, output_path, config_record=None):
        super().__init__()
        self.output_path = output_path
        self.config_record = config_record

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            if self.config_record is not None:
                record = {"record_type": "config", **self.config_record}
                f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if not str(name).startswith("val/") and str(name) not in self.TRAIN_EPOCH_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().cpu().item()
            elif isinstance(value, (int, float)):
                value = float(value)
            else:
                continue
            metrics[str(name)] = value

        for name, value in getattr(pl_module, "_train_epoch_summary", {}).items():
            metrics[str(name)] = float(value)

        if not metrics:
            return

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        record = {
            "record_type": "metrics",
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "metrics": dict(sorted(metrics.items())),
        }
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def _split_csv(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_data_dirs(args):
    data_dirs = []

    for dataset_name in _split_csv(args.datasets):
        key = dataset_name.lower()
        if key not in DATASET_ROOTS:
            known = ", ".join(sorted(set(DATASET_ROOTS.keys())))
            raise ValueError(f"Unknown dataset name '{dataset_name}'. Known names/aliases: {known}")
        data_dirs.append(DATASET_ROOTS[key])

    data_dirs.extend(_split_csv(args.data_dir))

    if not data_dirs:
        data_dirs.append(DATASET_ROOTS["opentouch"])

    deduped = []
    seen = set()
    for path in data_dirs:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


class OpenTouchHAMER_TactileWrapper(HAMER_Tactile):
    def __init__(
        self,
        cfg,
        learning_rate=1e-4,
        tactile_loss_config=None,
        tactile_only_forward=True,
        tactile_loss_scale=10.0,
        lr_warmup_epochs=0,
        frame_low_volume_thr=30.0,
        frame_high_volume_thr=150.0,
        sync_train_logs=False,
        audit_nonfinite_grads=False,
        nonfinite_audit_dir=None,
        audit_checkpoint_path=None,
        audit_seed=2029,
        tactile_head_type="dense_v2",
        backbone_feature_layers=(16, 24, 32),
        visual_backbone="hamer",
        dino_weights="",
    ):
        # Initialize without loading rendering to save GPU memory
        super().__init__(
            cfg,
            init_renderer=False,
            tactile_only_forward=tactile_only_forward,
            tactile_loss_scale=tactile_loss_scale,
            tactile_head_type=tactile_head_type,
            backbone_feature_layers=backbone_feature_layers,
            visual_backbone=visual_backbone,
            dino_weights=dino_weights,
        )
        self.learning_rate = learning_rate
        self.lr_warmup_epochs = int(lr_warmup_epochs)
        self.sync_train_logs = bool(sync_train_logs)
        self.frame_low_volume_thr = float(frame_low_volume_thr)
        self.frame_high_volume_thr = float(frame_high_volume_thr)
        self.audit_nonfinite_grads = bool(audit_nonfinite_grads)
        self.nonfinite_audit_dir = str(nonfinite_audit_dir) if nonfinite_audit_dir else None
        self.audit_checkpoint_path = str(audit_checkpoint_path) if audit_checkpoint_path else None
        self.audit_seed = int(audit_seed)
        self.visual_backbone_model_name = (
            self.backbone.MODEL_NAME if self.visual_backbone == "dinov3_hplus" else "hamer_vit_h"
        )
        self.backbone_weights_path = (
            str(dino_weights) if self.visual_backbone == "dinov3_hplus" else str(audit_checkpoint_path or "")
        )
        self.backbone_weights_sha256 = ""
        self._amp_audit_batch = None
        self._amp_audit_replay_batch = None
        self._amp_audit_cpu_rng_state = None
        self._amp_audit_cuda_rng_state = None
        self._pending_nonfinite_event = None
        self._nonfinite_event_count = 0
        if tactile_loss_config is not None:
            self.set_tactile_loss_config(tactile_loss_config)
        
        self.automatic_optimization = True
        
        print("Freezing all non-tactile modules. Only training the tactile head...")
        # HAMER also owns modules such as the adversarial discriminator. In
        # tactile-only training they are intentionally skipped, so freeze the
        # full model first to keep DDP from seeing unused trainable parameters.
        for param in self.parameters():
            param.requires_grad = False

        # Keep the tactile head trainable
        for param in self.tactile_head.parameters():
            param.requires_grad = True

        self._val_eval_stats = None
        self._grad_clip_trigger_count = 0
        self._grad_clip_step_count = 0
        self._grad_norm_finite_step_count = 0
        self._nonfinite_grad_step_count = 0
        self._grad_norm_sum = 0.0
        self._grad_norm_max = 0.0
        self._effective_lr_epoch_end = float(self.learning_rate)
        self._train_epoch_summary = {}

    def on_train_epoch_start(self):
        self._grad_clip_trigger_count = 0
        self._grad_clip_step_count = 0
        self._grad_norm_finite_step_count = 0
        self._nonfinite_grad_step_count = 0
        self._grad_norm_sum = 0.0
        self._grad_norm_max = 0.0

    def _amp_scaler_scale(self):
        trainer = getattr(self, "_trainer", None)
        plugin = getattr(trainer, "precision_plugin", None) if trainer is not None else None
        scaler = getattr(plugin, "scaler", None)
        if scaler is None:
            return None
        try:
            return float(scaler.get_scale())
        except Exception:
            return None

    @staticmethod
    def _audit_tensor_stats(value):
        if not isinstance(value, torch.Tensor):
            return None
        tensor = value.detach().float()
        finite = torch.isfinite(tensor)
        finite_values = tensor[finite]
        return {
            "shape": list(tensor.shape),
            "nonfinite_count": int((~finite).sum().item()),
            "min": float(finite_values.min().item()) if finite_values.numel() else None,
            "max": float(finite_values.max().item()) if finite_values.numel() else None,
            "abs_max": float(finite_values.abs().max().item()) if finite_values.numel() else None,
        }

    def _capture_amp_audit_batch(self, batch, output, batch_idx):
        target = batch["tactile_signal"].detach().float()
        palm = batch["palm_mask"].detach().float()
        valid = batch["has_tactile"].detach().float()
        palm_count = palm.sum(dim=-1).clamp_min(1.0)
        volume = (target * palm).sum(dim=-1)
        positive = (((target >= 0.005).float() * palm).sum(dim=-1) / palm_count)
        active = (((target >= self.tactile_loss_config.active_pressure_thr).float() * palm).sum(dim=-1) / palm_count)
        right = batch.get("right")
        right_values = right.detach().cpu().tolist() if isinstance(right, torch.Tensor) else [None] * len(volume)
        sample_dirs = list(batch.get("sample_dir", [None] * len(volume)))
        hands = list(batch.get("hand", [None] * len(volume)))
        datasets = list(batch.get("dataset", [None] * len(volume)))
        samples = []
        for index in range(len(volume)):
            samples.append({
                "sample_dir": sample_dirs[index],
                "hand": hands[index],
                "dataset": datasets[index],
                "is_right": int(round(float(right_values[index]))) if right_values[index] is not None else None,
                "gt_volume": float(volume[index].cpu()),
                "positive_fraction": float(positive[index].cpu()),
                "active_fraction": float(active[index].cpu()),
                "has_tactile": float(valid[index].cpu()),
            })
        loss_values = {}
        for name, value in output.get("losses", {}).items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                loss_values[name] = float(value.detach().float().cpu())
        tensor_stats = {
            name: self._audit_tensor_stats(output.get(name))
            for name in ("pred_logits", "pred_tactile")
            if output.get(name) is not None
        }
        self._amp_audit_batch = {
            "batch_idx": int(batch_idx),
            "samples": samples,
            "batch_mean_gt_volume": float(volume.mean().cpu()),
            "batch_positive_fraction": float(positive.mean().cpu()),
            "batch_active_fraction": float(active.mean().cpu()),
            "losses": loss_values,
            "tensor_stats": tensor_stats,
        }
        replay_keys = ("img", "tactile_signal", "has_tactile", "palm_mask")
        self._amp_audit_replay_batch = {
            key: batch[key].detach().cpu()
            for key in replay_keys
        }
        self._amp_audit_replay_batch["dataset"] = datasets
        self._amp_audit_replay_batch["sample_dir"] = sample_dirs
        self._amp_audit_replay_batch["hand"] = hands
        self._amp_audit_replay_batch["right"] = (
            right.detach().cpu() if isinstance(right, torch.Tensor) else right_values
        )
        self._amp_audit_replay_batch["cpu_rng_state"] = self._amp_audit_cpu_rng_state
        self._amp_audit_replay_batch["cuda_rng_state"] = self._amp_audit_cuda_rng_state

    def _nonfinite_audit_path(self, filename):
        if not self.nonfinite_audit_dir:
            return None
        path = Path(self.nonfinite_audit_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path / filename

    def _write_nonfinite_event(self, event):
        rank = int(getattr(self, "global_rank", 0))
        path = self._nonfinite_audit_path(f"nonfinite_grad_rank{rank}.jsonl")
        if path is None:
            return
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")

    def on_train_start(self):
        if not self.audit_nonfinite_grads:
            return
        rank = int(getattr(self, "global_rank", 0))
        path = self._nonfinite_audit_path(f"capture_config_rank{rank}.json")
        if path is None:
            return
        payload = {
            "rank": rank,
            "world_size": int(getattr(self.trainer, "world_size", 1)),
            "checkpoint": self.audit_checkpoint_path,
            "seed": self.audit_seed,
            "precision": str(getattr(self.trainer, "precision", "unknown")),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def on_before_optimizer_step(self, optimizer):
        squared_norm = torch.zeros((), device=self.device)
        branch_squared = {}
        first_nonfinite_parameter = None
        nonfinite_parameter_count = 0
        for name, parameter in self.tactile_head.named_parameters():
            if parameter.grad is not None:
                gradient = parameter.grad.detach().float()
                gradient_squared = gradient.pow(2).sum()
                squared_norm = squared_norm + gradient_squared
                gradient_is_finite = bool(torch.isfinite(gradient).all().item())
                if not gradient_is_finite:
                    nonfinite_parameter_count += 1
                    if first_nonfinite_parameter is None:
                        first_nonfinite_parameter = name
                if self.audit_nonfinite_grads:
                    branch = name.split(".", 1)[0]
                    branch_squared[branch] = branch_squared.get(
                        branch,
                        torch.zeros((), device=self.device),
                    ) + gradient_squared
        grad_norm = torch.sqrt(squared_norm)
        clip_value = float(getattr(self.trainer, "gradient_clip_val", 0.0) or 0.0)
        clip_trigger = gradient_clip_triggered(grad_norm.item(), clip_value)
        self._grad_clip_trigger_count += int(clip_trigger)
        self._grad_clip_step_count += 1
        grad_norm_value = float(grad_norm.item())
        if math.isfinite(grad_norm_value):
            self._grad_norm_finite_step_count += 1
            self._grad_norm_sum += grad_norm_value
            self._grad_norm_max = max(self._grad_norm_max, grad_norm_value)
        else:
            self._nonfinite_grad_step_count += 1
        self._effective_lr_epoch_end = float(optimizer.param_groups[0]["lr"])
        if self.audit_nonfinite_grads and first_nonfinite_parameter is not None:
            self._nonfinite_event_count += 1
            event = {
                "event_index": self._nonfinite_event_count - 1,
                "epoch": int(self.current_epoch),
                "global_step": int(self.global_step),
                "rank": int(getattr(self, "global_rank", 0)),
                "world_size": int(getattr(self.trainer, "world_size", 1)),
                "checkpoint": self.audit_checkpoint_path,
                "seed": self.audit_seed,
                "scaler_scale_before": self._amp_scaler_scale(),
                "grad_norm_pre_clip": None,
                "first_nonfinite_parameter": first_nonfinite_parameter,
                "nonfinite_parameter_count": nonfinite_parameter_count,
                "branch_grad_norms": {},
                "batch": self._amp_audit_batch,
            }
            rank = int(getattr(self, "global_rank", 0))
            payload_path = self._nonfinite_audit_path(
                f"nonfinite_batch_rank{rank}_step{int(self.global_step)}.pt"
            )
            if payload_path is not None and self._amp_audit_replay_batch is not None:
                torch.save(self._amp_audit_replay_batch, payload_path)
                event["replay_batch_path"] = str(payload_path)
            if math.isfinite(grad_norm_value):
                event["grad_norm_pre_clip"] = grad_norm_value
            for branch, value in branch_squared.items():
                norm = float(torch.sqrt(value).item())
                event["branch_grad_norms"][branch] = norm if math.isfinite(norm) else None
            self._pending_nonfinite_event = event
        self.log(
            "train/grad_norm_pre_clip",
            grad_norm,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/grad_clip_trigger",
            clip_trigger,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/nonfinite_grad_step",
            float(first_nonfinite_parameter is not None),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/effective_lr",
            float(optimizer.param_groups[0]["lr"]),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
        if first_nonfinite_parameter is not None and not self.audit_nonfinite_grads:
            raise FloatingPointError(
                "Non-finite tactile-head gradient detected before optimizer step: "
                f"parameter={first_nonfinite_parameter}, epoch={self.current_epoch}, "
                f"global_step={self.global_step}, precision={self.trainer.precision}. "
                "The optimizer step was aborted; rerun with --audit_nonfinite_grads "
                "to capture the exact batch."
            )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.audit_nonfinite_grads or self._pending_nonfinite_event is None:
            return
        event = self._pending_nonfinite_event
        scale_after = self._amp_scaler_scale()
        scale_before = event.get("scaler_scale_before")
        event["scaler_scale_after"] = scale_after
        event["optimizer_step_skipped"] = bool(
            scale_before is not None and scale_after is not None and scale_after < scale_before
        )
        self._write_nonfinite_event(event)
        self._pending_nonfinite_event = None

    def on_train_end(self):
        if not self.audit_nonfinite_grads:
            return
        rank = int(getattr(self, "global_rank", 0))
        path = self._nonfinite_audit_path(f"capture_summary_rank{rank}.json")
        if path is None:
            return
        payload = {
            "rank": rank,
            "events": int(self._nonfinite_event_count),
            "checkpoint": self.audit_checkpoint_path,
            "seed": self.audit_seed,
            "final_global_step": int(self.global_step),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def on_train_epoch_end(self):
        summary = self._current_train_epoch_summary()
        self.log(
            "train/epoch_grad_norm_mean",
            summary["train/epoch_grad_norm_mean"],
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            reduce_fx="mean",
        )
        self.log(
            "train/epoch_grad_norm_max",
            summary["train/epoch_grad_norm_max"],
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            reduce_fx="max",
        )
        self.log(
            "train/grad_clip_trigger_rate_epoch",
            summary["train/grad_clip_trigger_rate_epoch"],
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/nonfinite_grad_rate_epoch",
            summary["train/nonfinite_grad_rate_epoch"],
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/effective_lr_epoch_end",
            summary["train/effective_lr_epoch_end"],
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )

    def _current_train_epoch_summary(self):
        steps = max(self._grad_clip_step_count, 1)
        finite_steps = max(self._grad_norm_finite_step_count, 1)
        return {
            "train/epoch_grad_norm_mean": self._grad_norm_sum / finite_steps,
            "train/epoch_grad_norm_max": self._grad_norm_max,
            "train/grad_clip_trigger_rate_epoch": self._grad_clip_trigger_count / steps,
            "train/nonfinite_grad_rate_epoch": self._nonfinite_grad_step_count / steps,
            "train/effective_lr_epoch_end": self._effective_lr_epoch_end,
        }

    def _global_train_epoch_summary(self):
        local = torch.tensor(
            [
                self._grad_norm_sum,
                float(self._grad_norm_finite_step_count),
                self._grad_norm_max,
                float(self._grad_clip_trigger_count),
                self._effective_lr_epoch_end,
                float(self._nonfinite_grad_step_count),
                float(self._grad_clip_step_count),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if getattr(self.trainer, "world_size", 1) > 1:
            gathered = self.all_gather(local).reshape(-1, local.numel())
        else:
            gathered = local.unsqueeze(0)
        finite_steps = gathered[:, 1].sum().clamp_min(1.0)
        total_steps = gathered[:, 6].sum().clamp_min(1.0)
        return {
            "train/epoch_grad_norm_mean": float((gathered[:, 0].sum() / finite_steps).item()),
            "train/epoch_grad_norm_max": float(gathered[:, 2].max().item()),
            "train/grad_clip_trigger_rate_epoch": float((gathered[:, 3].sum() / total_steps).item()),
            "train/nonfinite_grad_rate_epoch": float((gathered[:, 5].sum() / total_steps).item()),
            "train/effective_lr_epoch_end": float(gathered[:, 4].mean().item()),
        }

    def train(self, mode=True):
        super().train(mode)
        if self.tactile_only_forward:
            self.backbone.eval()
            self.mano_head.eval()
            if hasattr(self, "discriminator"):
                self.discriminator.eval()
        return self
                
    def training_step(self, batch, batch_idx):
        if self.audit_nonfinite_grads:
            self._amp_audit_cpu_rng_state = torch.get_rng_state().cpu()
            self._amp_audit_cuda_rng_state = torch.cuda.get_rng_state(self.device).cpu()
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError(
                "Non-finite training loss detected before backward: "
                f"epoch={self.current_epoch}, global_step={self.global_step}, "
                f"batch_idx={batch_idx}, precision={self.trainer.precision}."
            )
        if self.audit_nonfinite_grads:
            self._capture_amp_audit_batch(batch, output, batch_idx)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_train_logs)
        self._log_tactile_loss_breakdown("train", output, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        if not bool(torch.isfinite(loss.detach()).all().item()):
            raise FloatingPointError(
                "Non-finite validation loss detected: "
                f"epoch={self.current_epoch}, global_step={self.global_step}, "
                f"batch_idx={batch_idx}, precision={self.trainer.precision}."
            )
        
        pred_tactile = output['pred_tactile']
        gt_tactile = batch['tactile_signal']
        has_tactile = batch['has_tactile']
        palm_mask = batch['palm_mask']

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self._log_tactile_loss_breakdown("val", output, on_step=False)
        if self.tactile_head_type == "dense_v2_multilevel_concat":
            diagnostics = self.tactile_head.feature_diagnostics()
            projected_rms = diagnostics.get("projected_rms")
            if projected_rms is not None:
                for layer, value in zip(self.backbone_feature_layers, projected_rms):
                    self.log(
                        f"val/feature_projected_rms_layer_{layer}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        sync_dist=True,
                    )
            fusion_rms = diagnostics.get("fusion_rms")
            if fusion_rms is not None:
                self.log(
                    "val/feature_fusion_rms",
                    fusion_rms,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )
        self._accumulate_val_eval_stats(
            pred_tactile,
            gt_tactile,
            has_tactile,
            palm_mask,
        )
        return loss

    def on_validation_epoch_start(self):
        self._train_epoch_summary = self._global_train_epoch_summary()
        self._val_eval_stats = None

    def _accumulate_val_eval_stats(
        self,
        pred_tactile,
        gt_tactile,
        has_tactile,
        palm_mask,
    ):
        valid = has_tactile > 0.5
        if valid.sum() == 0:
            return

        pred = pred_tactile[valid].detach().float()
        gt = gt_tactile[valid].detach().float()
        palm = (palm_mask[valid].detach().float() > 0.5).float()
        raw_contact = pred
        contact = pred
        pressure = pred
        active_thr = float(self.tactile_loss_config.active_pressure_thr)
        background_thr = 0.02

        diff = (pred - gt) * palm
        palm_values = palm.sum()
        active_mask = ((gt > active_thr).float() * palm)
        background_mask = ((gt <= background_thr).float() * palm)
        pred_active = ((pred > active_thr).float() * palm)
        contact_eval_mask = ((((gt >= active_thr) | (gt <= background_thr)).float()) * palm)
        contact_pred = ((contact > 0.5).float() * contact_eval_mask)
        contact_tp = (contact_pred * active_mask).sum()
        contact_pred_count = contact_pred.sum()

        pred_bin = pred_active
        gt_bin = active_mask
        intersection = (pred_bin * gt_bin).sum(dim=1)
        union = torch.clamp(((pred_bin + gt_bin) > 0).float().sum(dim=1), min=1.0)
        contact_iou_sum = (intersection / union).sum()

        vol_intersection = (torch.minimum(pred, gt) * palm).sum(dim=1)
        vol_union = torch.clamp((torch.maximum(pred, gt) * palm).sum(dim=1), min=1e-6)
        volumetric_iou = vol_intersection / vol_union
        volumetric_iou_sum = volumetric_iou.sum()
        pred_volume_frame = (pred * palm).sum(dim=1)
        gt_volume_frame = (gt * palm).sum(dim=1)
        frame_volume_norm = 150.0
        frame_volume_target = torch.clamp(gt_volume_frame / frame_volume_norm, min=0.0, max=1.0)
        frame_volume_pred = torch.clamp(
            pred_volume_frame.detach() / frame_volume_norm,
            min=0.0,
            max=1.0,
        )
        low_volume_mask = (gt_volume_frame <= self.frame_low_volume_thr).to(pred.dtype)
        high_volume_mask = (gt_volume_frame >= self.frame_high_volume_thr).to(pred.dtype)
        empty_frame_mask = (gt_volume_frame <= 1e-8).to(pred.dtype)
        pred_active_vertices_frame = pred_active.sum(dim=1)

        near_zero_thr = 0.005
        low_pressure_mask = ((gt < near_zero_thr).float() * palm)
        weak_pressure_mask = (((gt >= 0.02) & (gt < active_thr)).float() * palm)
        mid_pressure_mask = (((gt >= 0.1) & (gt < 0.3)).float() * palm)
        high_pressure_mask = ((gt >= 0.3).float() * palm)

        stats = torch.stack([
            valid.sum().float(),
            palm_values,
            diff.abs().sum(),
            diff.pow(2).sum(),
            (pred * palm).sum(),
            (gt * palm).sum(),
            (diff.abs() * active_mask).sum(),
            active_mask.sum(),
            (diff.abs() * background_mask).sum(),
            background_mask.sum(),
            (pred_active * active_mask).sum(),
            active_mask.sum(),
            (pred_active * background_mask).sum(),
            contact_iou_sum,
            volumetric_iou_sum,
            (pred * low_pressure_mask).sum(),
            low_pressure_mask.sum(),
            (pred * weak_pressure_mask).sum(),
            weak_pressure_mask.sum(),
            (pred * mid_pressure_mask).sum(),
            mid_pressure_mask.sum(),
            (pred * high_pressure_mask).sum(),
            high_pressure_mask.sum(),
            contact_tp,
            contact_pred_count,
            (raw_contact * active_mask).sum(),
            (raw_contact * background_mask).sum(),
            (pressure * low_pressure_mask).sum(),
            (pressure * weak_pressure_mask).sum(),
            (pressure * mid_pressure_mask).sum(),
            (pressure * high_pressure_mask).sum(),
            (pressure * active_mask).sum(),
            (pressure * background_mask).sum(),
            gt_volume_frame.pow(2).sum(),
            pred_volume_frame.pow(2).sum(),
            (gt_volume_frame * pred_volume_frame).sum(),
            (pred_volume_frame * low_volume_mask).sum(),
            (gt_volume_frame * low_volume_mask).sum(),
            (volumetric_iou * low_volume_mask).sum(),
            low_volume_mask.sum(),
            (pred_volume_frame * high_volume_mask).sum(),
            (gt_volume_frame * high_volume_mask).sum(),
            (volumetric_iou * high_volume_mask).sum(),
            high_volume_mask.sum(),
            (pred_volume_frame * empty_frame_mask).sum(),
            (pred_active_vertices_frame * empty_frame_mask).sum(),
            empty_frame_mask.sum(),
            (pred_active * weak_pressure_mask).sum(),
            (contact * active_mask).sum(),
            (contact * background_mask).sum(),
            frame_volume_pred.sum(),
            frame_volume_target.sum(),
            (frame_volume_pred - frame_volume_target).abs().sum(),
            (pred_active * low_pressure_mask).sum(),
            ((gt_volume_frame < 10.0) & (pred_volume_frame > 300.0)).sum().float(),
            (gt_volume_frame < 10.0).sum().float(),
            ((gt_volume_frame >= 150.0) & (pred_volume_frame < 50.0)).sum().float(),
            (gt_volume_frame >= 150.0).sum().float(),
            (((gt < 0.005) & (pred >= 0.3)).float() * palm).sum(),
            (((gt < 0.005).float()) * palm).sum(),
            (torch.clamp(pred - gt, min=0.0) * ((gt < 0.005) & (pred >= 0.3)).float() * palm).sum(),
            (((gt < 0.005) & (pred >= 0.5)).float() * palm).sum(),
            (torch.clamp(pred - gt, min=0.0) * ((gt < 0.005) & (pred >= 0.5)).float() * palm).sum(),
            (((gt < 0.05) & (pred >= 0.3)).float() * palm).sum(),
            (((gt < 0.05).float()) * palm).sum(),
            (torch.clamp(pred - gt, min=0.0) * ((gt < 0.05) & (pred >= 0.3)).float() * palm).sum(),
        ]).to(pred_tactile.device)

        if self._val_eval_stats is None:
            self._val_eval_stats = stats
        else:
            self._val_eval_stats = self._val_eval_stats + stats

    def on_validation_epoch_end(self):
        if self._val_eval_stats is None:
            return
        stats = self._val_eval_stats
        if self.trainer.world_size > 1:
            gathered = self.all_gather(stats)
            stats = gathered.reshape(-1, stats.numel()).sum(dim=0)

        frames = stats[0].clamp_min(1.0)
        values = stats[1].clamp_min(1.0)
        gt_volume = stats[5].clamp_min(1e-6)
        active_count = stats[7].clamp_min(1.0)
        background_count = stats[9].clamp_min(1.0)
        active_gt_count = stats[11].clamp_min(1.0)
        low_pressure_count = stats[16].clamp_min(1.0)
        weak_pressure_count = stats[18].clamp_min(1.0)
        mid_pressure_count = stats[20].clamp_min(1.0)
        high_pressure_count = stats[22].clamp_min(1.0)
        contact_tp = stats[23]
        contact_pred_count = stats[24].clamp_min(1.0)
        contact_precision = contact_tp / contact_pred_count
        contact_recall = contact_tp / active_gt_count
        contact_f1 = 2.0 * contact_precision * contact_recall / (contact_precision + contact_recall).clamp_min(1e-6)
        corr_n = frames
        corr_num = corr_n * stats[35] - stats[5] * stats[4]
        corr_gt_var = (corr_n * stats[33] - stats[5].pow(2)).clamp_min(0.0)
        corr_pred_var = (corr_n * stats[34] - stats[4].pow(2)).clamp_min(0.0)
        corr_den = torch.sqrt(corr_gt_var * corr_pred_var).clamp_min(1e-6)
        frame_volume_corr = torch.clamp(corr_num / corr_den, min=-1.0, max=1.0)
        low_volume_count = stats[39].clamp_min(1.0)
        high_volume_count = stats[43].clamp_min(1.0)
        empty_frame_count = stats[46].clamp_min(1.0)

        eval_metrics = {
            "val/eval_mae": stats[2] / values,
            "val/eval_rmse": torch.sqrt(stats[3] / values),
            "val/eval_pred_gt_volume_ratio": stats[4] / gt_volume,
            "val/eval_frame_volume_corr": frame_volume_corr,
            "val/eval_low_volume_pred_gt_ratio": stats[36] / stats[37].clamp_min(1e-6),
            "val/eval_high_volume_pred_gt_ratio": stats[40] / stats[41].clamp_min(1e-6),
            "val/eval_low_volume_pred_volume": stats[36] / low_volume_count,
            "val/eval_high_volume_pred_volume": stats[40] / high_volume_count,
            "val/eval_low_volume_volumetric_iou": stats[38] / low_volume_count,
            "val/eval_high_volume_volumetric_iou": stats[42] / high_volume_count,
            "val/eval_empty_frame_pred_volume": stats[44] / empty_frame_count,
            "val/eval_empty_frame_pred_active_vertices": stats[45] / empty_frame_count,
            "val/eval_weak_zone_pred_active_rate": stats[47] / weak_pressure_count,
            "val/eval_active_mae": stats[6] / active_count,
            "val/eval_background_mae": stats[8] / background_count,
            "val/eval_active_recall": stats[10] / active_gt_count,
            "val/eval_bg_false_positive": stats[12] / background_count,
            "val/eval_contact_iou_active_thr": stats[13] / frames,
            "val/eval_volumetric_iou": stats[14] / frames,
            "val/eval_contact_precision": contact_precision,
            "val/eval_contact_recall": contact_recall,
            "val/eval_contact_f1": contact_f1,
            "val/eval_low_pressure_mean_pred": stats[15] / low_pressure_count,
            "val/eval_weak_pressure_mean_pred": stats[17] / weak_pressure_count,
            "val/eval_mid_pressure_mean_pred": stats[19] / mid_pressure_count,
            "val/eval_high_pressure_mean_pred": stats[21] / high_pressure_count,
            "val/eval_low_pressure_pred_active_rate": stats[53] / low_pressure_count,
            "val/eval_catastrophic_over_rate": stats[54] / stats[55].clamp_min(1.0),
            "val/eval_catastrophic_under_rate": stats[56] / stats[57].clamp_min(1.0),
            "val/eval_false_high_gt005_pred03_rate": stats[58] / stats[59].clamp_min(1.0),
            "val/eval_false_high_gt005_pred03_excess_volume_fraction": stats[60] / stats[4].clamp_min(1e-6),
            "val/eval_false_high_gt005_pred05_rate": stats[61] / stats[59].clamp_min(1.0),
            "val/eval_false_high_gt05_pred03_rate": stats[63] / stats[64].clamp_min(1.0),
            "val/eval_false_high_gt05_pred03_excess_volume_fraction": stats[65] / stats[4].clamp_min(1e-6),
        }
        for name, value in eval_metrics.items():
            self.log(
                name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"val/eval_rmse", "val/eval_pred_gt_volume_ratio"},
                logger=True,
                sync_dist=True,
            )
        if self.tactile_head_type == "dense_v2_multilevel":
            fusion_weights = self.tactile_head.fusion_weights()
            fusion_entropy = -(fusion_weights * fusion_weights.clamp_min(1e-8).log()).sum()
            for layer, weight in zip(self.backbone_feature_layers, fusion_weights):
                self.log(
                    f"val/feature_fusion_weight_layer_{layer}",
                    weight,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=False,
                )
            self.log(
                "val/feature_fusion_entropy",
                fusion_entropy,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
            )
        elif self.tactile_head_type == "dense_v2_multilevel_concat":
            for layer, value in zip(
                self.backbone_feature_layers,
                self.tactile_head.fusion_group_contributions(),
            ):
                self.log(
                    f"val/feature_fusion_kernel_contribution_layer_{layer}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=False,
                )

    def _log_tactile_loss_breakdown(self, prefix, output, on_step):
        mapping = {
            "loss_smooth_l1_raw": "loss/smooth_l1_raw",
            "loss_logit_bce_raw": "loss/logit_bce_raw",
            "loss_base_tactile": "loss/base_tactile",
            "loss_weighted_tactile": "loss/weighted_tactile",
            "loss_background": "loss/background",
            "loss_tactile": "loss/total",
            "loss_ramp": "schedule/loss_ramp",
        }
        for key, name in mapping.items():
            if key in output["losses"]:
                self.log(
                    f"{prefix}/{name}",
                    output["losses"][key],
                    on_step=on_step,
                    on_epoch=True,
                    logger=True,
                    sync_dist=self.sync_train_logs if prefix == "train" else True,
                )
        
    def tensorboard_logging(self, *args, **kwargs):
        # Override to be a no-op to prevent WebGL/OpenGL/Renderer crashes during training
        pass

    def configure_optimizers(self):
        # Separate parameters into decay and no_decay groups
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ["bias", "norm", "LayerNorm"]):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
                    
        optim_groups = [
            {"params": decay_params, "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
                    
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.learning_rate
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        print(f"Total training steps for LR Scheduler: {total_steps}")
        
        max_epochs = max(int(getattr(self.trainer, "max_epochs", 1)), 1)
        warmup_steps = int(round(total_steps * max(self.lr_warmup_epochs, 0) / max_epochs))
        if warmup_steps > 0:
            warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / float(max(warmup_steps, 1)),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1),
                eta_min=self.learning_rate * 0.01,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
            print(
                f"LR schedule: linear warmup for {warmup_steps} steps "
                f"({self.lr_warmup_epochs} epoch(s)), then cosine for {total_steps - warmup_steps} steps"
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.learning_rate * 0.01,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def _install_legacy_lightning_attribute_dict():
    """Restore a class path used by checkpoints from older Lightning releases."""
    import lightning_fabric.utilities.data as fabric_data

    if hasattr(fabric_data, "AttributeDict"):
        return

    try:
        from pytorch_lightning.utilities.parsing import AttributeDict
    except ImportError:
        class AttributeDict(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError as error:
                    raise AttributeError(key) from error

            def __setattr__(self, key, value):
                self[key] = value

    fabric_data.AttributeDict = AttributeDict
    print("Installed legacy Lightning AttributeDict checkpoint compatibility shim.")


def _load_checkpoint(checkpoint_path):
    _install_legacy_lightning_attribute_dict()
    return torch.load(checkpoint_path, map_location="cpu")


def load_compatible_state_dict(model, checkpoint_path, load_backbone=True):
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    compact_format = checkpoint.get("format") if isinstance(checkpoint, dict) else None
    if compact_format in {"tactile_head_only_v1", "tactile_trainable_v2"}:
        checkpoint_visual_backbone = str(checkpoint.get("visual_backbone", "hamer") or "hamer")
        model_visual_backbone = str(getattr(model, "visual_backbone", "hamer"))
        if checkpoint_visual_backbone != model_visual_backbone:
            raise ValueError(
                f"Checkpoint visual_backbone={checkpoint_visual_backbone} does not match model={model_visual_backbone}"
            )
        checkpoint_head = str(checkpoint.get("tactile_head_type", "") or "")
        if checkpoint_head and checkpoint_head != str(getattr(model, "tactile_head_type", "")):
            raise ValueError(
                f"Checkpoint tactile_head_type={checkpoint_head} does not match model={model.tactile_head_type}"
            )
        expected_hash = str(checkpoint.get("backbone_sha256", "") or "")
        actual_hash = str(getattr(model, "backbone_weights_sha256", "") or "")
        if expected_hash and actual_hash and expected_hash != actual_hash:
            raise ValueError(
                f"Backbone SHA256 mismatch: checkpoint={expected_hash}, actual={actual_hash}"
            )

        raw_base_path = str(
            checkpoint.get("backbone_weights", "")
            or checkpoint.get("base_checkpoint", "")
            or ""
        )
        base_candidates = []
        if raw_base_path and checkpoint_visual_backbone == "hamer":
            base_path = Path(raw_base_path).expanduser()
            if not base_path.is_absolute():
                base_path = Path(checkpoint_path).expanduser().resolve().parent / base_path
            base_candidates.append(base_path)
        if checkpoint_visual_backbone == "hamer":
            base_candidates.append(
                Path(workspace_dir) / "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"
            )
            base_path = next((path for path in base_candidates if path.is_file()), None)
            if base_path is None:
                raise FileNotFoundError(
                    "Compact tactile checkpoint requires its frozen HaMeR base checkpoint. "
                    f"Tried: {[str(path) for path in base_candidates]}"
                )
            base_checkpoint = _load_checkpoint(base_path)
            base_state = (
                base_checkpoint["state_dict"]
                if isinstance(base_checkpoint, dict) and "state_dict" in base_checkpoint
                else base_checkpoint
            )
            state_dict = dict(base_state)
            state_dict.update(checkpoint["state_dict"])
            if hasattr(model, "audit_checkpoint_path"):
                model.audit_checkpoint_path = str(base_path)
        else:
            base_path = Path(raw_base_path) if raw_base_path else None
            state_dict = dict(checkpoint["state_dict"])
        print(
            "Loading compact tactile checkpoint: "
            f"backbone={checkpoint_visual_backbone}:{base_path}, head={checkpoint_path}, "
            f"epoch={checkpoint.get('epoch')}, global_step={checkpoint.get('global_step')}"
        )
    model_state = model.state_dict()

    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
        if not load_backbone and (key.startswith("backbone.") or key.startswith("model.backbone.")):
            continue
        if key not in model_state:
            compatible_state[key] = value
            continue
        if tuple(model_state[key].shape) == tuple(value.shape):
            compatible_state[key] = value
        else:
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if skipped:
        print("Skipped incompatible checkpoint tensors:")
        for key, old_shape, new_shape in skipped:
            print(f"  {key}: checkpoint {old_shape} -> model {new_shape}")
    return missing, unexpected


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Hamer Tactile Head")
    # Default to the fine-tuned checkpoint!
    parser.add_argument("--checkpoint", type=str, default=os.path.join(workspace_dir, "opentouch_hamer_ft/checkpoints/regression_wocam_60/best_ft_model.ckpt"), help="Path to fine-tuned Hamer checkpoint")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "One or more explicit extracted dataset roots. Use comma-separated paths for mixed training. "
            "If omitted, --datasets is resolved to the default processed roots."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Dataset names/aliases to train on, comma-separated. Supported: "
            "opentouch/ot, touchanything/egotouch/ta, egotactile/ego. "
            "Explicit --data_dir paths are appended after these resolved roots. "
            "If both --datasets and --data_dir are omitted, defaults to opentouch."
        ),
    )
    
    parser.add_argument("--gpus", type=str, default="4", help="GPU indices (comma-separated, e.g. 4,5)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (per GPU)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--val_num_workers", type=int, default=None, help="Workers for validation DataLoader; defaults to --num_workers")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True, help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor when num_workers > 0")
    parser.add_argument("--index_workers", type=int, default=1, help="Workers for initial meta.json index scanning")
    parser.add_argument("--index_backend", type=str, default="process", choices=["process", "thread"], help="Parallel backend for initial index scanning")
    parser.add_argument("--index_chunksize", type=int, default=256, help="Chunk size for parallel index scanning")
    parser.add_argument("--index_cache_dir", type=str, default=os.path.join(ft_dir, "index_cache"), help="Shared JSONL cache for scanned dataset indices")
    parser.add_argument("--rebuild_index", action="store_true", help="Force rank 0 to rebuild the index cache")
    parser.add_argument("--index_cache_timeout", type=int, default=3600, help="Seconds nonzero ranks wait for index cache")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--exp_name", type=str, default="mixed_dense_v2_repro", help="Experiment name")
    parser.add_argument("--checkpoint_monitor", type=str, default="val/eval_rmse", help="Legacy provenance field; training always saves mixed-val RMSE-best and V-IoU-best checkpoints")
    parser.add_argument("--checkpoint_mode", type=str, default="min", choices=["min", "max"], help="Legacy provenance field retained for command compatibility")
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test training")
    parser.add_argument("--render_platform", type=str, default="egl", choices=["egl", "osmesa"], help="Rendering platform (egl or osmesa)")
    parser.add_argument("--tactile_only_forward", action=argparse.BooleanOptionalAction, default=True, help="Skip MANO/base forward and train only the tactile path")
    parser.add_argument(
        "--tactile_head_type",
        choices=("dense_v2", "dense_v2_multilevel", "dense_v2_multilevel_concat"),
        default="dense_v2",
    )
    parser.add_argument(
        "--backbone_feature_layers",
        type=str,
        default="16,24,32",
        help="Comma-separated 1-based ViT block indices used by dense_v2_multilevel.",
    )
    parser.add_argument(
        "--visual_backbone",
        choices=("hamer", "dinov3_hplus"),
        default="hamer",
    )
    parser.add_argument(
        "--dino_weights",
        type=str,
        default=None,
        help="Required local DINOv3 H+/16 checkpoint when --visual_backbone=dinov3_hplus.",
    )
    parser.add_argument("--tactile_loss_scale", type=float, default=10.0, help="Multiplier for tactile loss")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Norm clipping applied by Lightning after grad-norm logging")
    parser.add_argument("--trainer_precision", choices=("16-mixed", "bf16-mixed", "32-true"), default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--lr_warmup_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--skip_validation", action="store_true")
    parser.add_argument("--skip_checkpointing", action="store_true")
    parser.add_argument("--audit_nonfinite_grads", action="store_true")
    parser.add_argument("--nonfinite_audit_dir", type=str, default=None)
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Use DDP unused-parameter detection")
    parser.add_argument("--sync_train_logs", action="store_true", help="Synchronize train logs across distributed workers")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validation frequency in epochs")
    parser.add_argument("--active_pressure_thr", type=float, default=0.05)
    parser.add_argument("--active_pressure_peak", type=float, default=0.10)
    parser.add_argument("--active_pressure_high", type=float, default=0.30)
    parser.add_argument("--background_pressure_thr", type=float, default=0.02)
    parser.add_argument("--background_pred_margin", type=float, default=0.02)
    parser.add_argument("--active_pressure_weight", type=float, default=1.0)
    parser.add_argument("--active_pressure_gamma", type=float, default=1.0)
    parser.add_argument("--background_loss_weight", type=float, default=1.0)
    parser.add_argument("--logit_bce_weight", type=float, default=0.1)
    parser.add_argument("--loss_ramp_epochs", type=int, default=10)
    parser.add_argument("--frame_low_volume_thr", type=float, default=30.0, help="GT frame volume threshold for low-volume validation diagnostics")
    parser.add_argument("--frame_high_volume_thr", type=float, default=150.0, help="GT frame volume threshold for high-volume validation diagnostics")
    parser.add_argument("--opentouch_high_pressure_thr", type=float, default=0.9)
    parser.add_argument("--opentouch_high_pressure_weight", type=float, default=0.3)
    return parser.parse_args()


def tactile_loss_config_from_args(args):
    return TactileLossConfig(
        loss_mode="dense_v2",
        active_pressure_thr=args.active_pressure_thr,
        active_pressure_peak=args.active_pressure_peak,
        active_pressure_high=args.active_pressure_high,
        background_pressure_thr=args.background_pressure_thr,
        background_pred_margin=args.background_pred_margin,
        active_pressure_weight=args.active_pressure_weight,
        active_pressure_gamma=args.active_pressure_gamma,
        background_loss_weight=args.background_loss_weight,
        logit_bce_weight=args.logit_bce_weight,
        loss_ramp_epochs=args.loss_ramp_epochs,
        opentouch_high_pressure_thr=args.opentouch_high_pressure_thr,
        opentouch_high_pressure_weight=args.opentouch_high_pressure_weight,
    )


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _run_git_command(args):
    try:
        return subprocess.check_output(args, cwd=workspace_dir, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_snapshot():
    return {
        "commit": _run_git_command(["git", "rev-parse", "HEAD"]),
        "branch": _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_files": _run_git_command(["git", "status", "--short"]).splitlines(),
        "diff_stat": _run_git_command(["git", "diff", "--stat"]).splitlines(),
    }


def _count_params(module, trainable_only=False):
    params = list(module.parameters())
    if trainable_only:
        params = [p for p in params if p.requires_grad]
    return int(sum(p.numel() for p in params))


def model_summary(model):
    sections = {
        "total": model,
        "backbone": model.backbone,
        "mano_head": model.mano_head,
        "tactile_head": model.tactile_head,
    }
    lines = [
        "HAMER tactile fine-tuning model summary",
        f"tactile_dim: {model.tactile_dim}",
        f"tactile_head_type: {model.tactile_head_type}",
        f"pool_layout: {getattr(model, 'pool_layout', None)}",
        f"pool_grid_size: {getattr(model, 'pool_grid_size', None)}",
        f"pool_valid_tokens: {getattr(model, 'pool_valid_tokens', None)}",
        f"frame_low_volume_thr: {getattr(model, 'frame_low_volume_thr', None)}",
        f"frame_high_volume_thr: {getattr(model, 'frame_high_volume_thr', None)}",
        f"tactile_only_forward: {model.tactile_only_forward}",
        f"tactile_loss_scale: {model.tactile_loss_scale}",
        "",
    ]
    summary = {}
    for name, module in sections.items():
        total = _count_params(module, trainable_only=False)
        trainable = _count_params(module, trainable_only=True)
        frozen = total - trainable
        summary[name] = {"total": total, "trainable": trainable, "frozen": frozen}
        lines.append(f"{name}: total={total:,}, trainable={trainable:,}, frozen={frozen:,}")
    return summary, "\n".join(lines) + "\n"


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)


def write_run_provenance(args, data_dirs, tactile_loss_config, model, ckpt_dir, num_gpus, lr_scaled):
    os.makedirs(ckpt_dir, exist_ok=True)
    git_info = git_snapshot()
    model_counts, model_summary_text = model_summary(model)
    run_config = {
        "args": vars(args),
        "resolved_data_dirs": data_dirs,
        "checkpoint": args.checkpoint,
        "visual_backbone": model.visual_backbone,
        "backbone_weights": model.backbone_weights_path,
        "backbone_sha256": model.backbone_weights_sha256,
        "num_gpus": num_gpus,
        "base_lr": args.lr,
        "effective_lr": lr_scaled,
        "lr_scaled": lr_scaled,
        "checkpoint_monitor": args.checkpoint_monitor,
        "checkpoint_mode": args.checkpoint_mode,
        "checkpoint_monitors": {
            "rmse-best": {"metric": "val/eval_rmse", "mode": "min"},
            "viou-best": {"metric": "val/eval_volumetric_iou", "mode": "max"},
        },
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
    }
    loss_config = asdict(tactile_loss_config)
    model_config = {
        "tactile_dim": int(model.tactile_dim),
        "tactile_only_forward": bool(model.tactile_only_forward),
        "tactile_head_type": str(model.tactile_head_type),
        "visual_backbone": str(model.visual_backbone),
        "visual_backbone_model_name": str(getattr(model, "visual_backbone_model_name", "")),
        "backbone_weights": str(model.backbone_weights_path),
        "backbone_sha256": str(model.backbone_weights_sha256),
        "pool_layout": str(getattr(model, "pool_layout", "")),
        "pool_grid_size": int(getattr(model, "pool_grid_size", 0)),
        "pool_valid_tokens": int(getattr(model, "pool_valid_tokens", 0)),
        "backbone_feature_layers": list(getattr(model, "backbone_feature_layers", ())),
        "lr_warmup_epochs": int(getattr(model, "lr_warmup_epochs", 0)),
        "frame_low_volume_thr": float(getattr(model, "frame_low_volume_thr", 0.0)),
        "frame_high_volume_thr": float(getattr(model, "frame_high_volume_thr", 0.0)),
        "tactile_loss_scale": float(model.tactile_loss_scale),
        "skip_mano_base_loss": bool(model.tactile_only_forward),
        "parameter_counts": model_counts,
    }
    write_json(os.path.join(ckpt_dir, "run_config.json"), run_config)
    write_json(os.path.join(ckpt_dir, "loss_config.json"), loss_config)
    write_json(os.path.join(ckpt_dir, "model_config.json"), model_config)
    write_json(os.path.join(ckpt_dir, "git_snapshot.json"), git_info)
    with open(os.path.join(ckpt_dir, "run_command.txt"), "w", encoding="utf-8") as f:
        f.write(run_config["command"] + "\n")
    with open(os.path.join(ckpt_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write(model_summary_text)
    return {
        "run_config": run_config,
        "loss_config": loss_config,
        "model_config": model_config,
        "git_snapshot": git_info,
    }


def make_dataloader(dataset, batch_size, shuffle, num_workers, persistent_workers, prefetch_factor):
    num_workers = int(num_workers)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def save_final_last_checkpoint(trainer, ckpt_dir, checkpoint_callbacks=None):
    return save_materialized_last_checkpoint(
        trainer,
        ckpt_dir,
        reason="fit_return_fallback",
        checkpoint_callbacks=checkpoint_callbacks,
    )


def _fsync_dir(path):
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_json(path, payload):
    path = Path(path)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)
    _fsync_dir(str(path.parent))


def _checkpoint_callback_metadata(checkpoint_callbacks):
    metadata = {}
    for selector, callback in (checkpoint_callbacks or {}).items():
        score = getattr(callback, "best_model_score", None)
        if isinstance(score, torch.Tensor):
            score = float(score.detach().cpu().item())
        elif score is not None:
            score = float(score)
        metadata[str(selector)] = {
            "monitor": str(getattr(callback, "monitor", "") or ""),
            "mode": str(getattr(callback, "mode", "") or ""),
            "path": str(getattr(callback, "best_model_path", "") or ""),
            "score": score,
        }
    return metadata


def _compact_checkpoint_payload(module, epoch, global_step, reason, monitor=None, score=None):
    head_state = {
        f"tactile_head.{name}": tensor.detach().cpu()
        for name, tensor in module.tactile_head.state_dict().items()
    }
    return {
        "format": "tactile_trainable_v2",
        "state_dict": head_state,
        "visual_backbone": str(getattr(module, "visual_backbone", "hamer")),
        "visual_backbone_model_name": str(getattr(module, "visual_backbone_model_name", "")),
        "backbone_weights": str(getattr(module, "backbone_weights_path", "") or ""),
        "backbone_sha256": str(getattr(module, "backbone_weights_sha256", "") or ""),
        "base_checkpoint": str(getattr(module, "audit_checkpoint_path", "") or ""),
        "tactile_head_type": str(getattr(module, "tactile_head_type", "")),
        "backbone_feature_layers": list(getattr(module, "backbone_feature_layers", ())),
        "loss_config": asdict(module.tactile_loss_config),
        "model_config": {
            "tactile_head_type": str(getattr(module, "tactile_head_type", "")),
            "visual_backbone": str(getattr(module, "visual_backbone", "hamer")),
            "visual_backbone_model_name": str(getattr(module, "visual_backbone_model_name", "")),
            "backbone_feature_layers": list(getattr(module, "backbone_feature_layers", ())),
            "backbone_weights": str(getattr(module, "backbone_weights_path", "") or ""),
            "backbone_sha256": str(getattr(module, "backbone_weights_sha256", "") or ""),
            "tactile_dim": int(getattr(module, "tactile_dim", 0)),
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "reason": str(reason),
        "monitor": str(monitor or ""),
        "score": None if score is None else float(score),
    }


def _atomic_torch_save(payload, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}")
    torch.save(payload, temp_path)
    os.replace(temp_path, destination)
    _fsync_dir(str(destination.parent))
    if destination.is_symlink() or not destination.is_file():
        raise RuntimeError(f"Failed to persist compact checkpoint: {destination}")


class CompactBestCheckpoint(Callback):
    def __init__(self, dirpath, filename, monitor, mode):
        super().__init__()
        self.dirpath = str(dirpath)
        self.filename = str(filename)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.best_model_path = ""
        self.best_model_score = None

    def _improved(self, score):
        if self.best_model_score is None:
            return True
        if self.mode == "min":
            return score < self.best_model_score
        return score > self.best_model_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        score = float(value.detach().float().cpu().item()) if isinstance(value, torch.Tensor) else float(value)
        if not math.isfinite(score) or not self._improved(score):
            return
        path = Path(self.dirpath) / f"{self.filename}.ckpt"
        payload = _compact_checkpoint_payload(
            pl_module,
            trainer.current_epoch,
            trainer.global_step,
            reason=f"best_{self.monitor}",
            monitor=self.monitor,
            score=score,
        )
        _atomic_torch_save(payload, path)
        self.best_model_score = score
        self.best_model_path = str(path)
        print(f"Compact best checkpoint: {self.monitor}={score:.8g} -> {path}", flush=True)


def save_materialized_last_checkpoint(
    trainer,
    ckpt_dir,
    reason="fit_end",
    checkpoint_callbacks=None,
    epoch=None,
):
    """Persist a compact tactile-head last.ckpt without DDP checkpoint collectives."""
    last_path = Path(ckpt_dir) / "last.ckpt"
    module = trainer.lightning_module
    checkpoint_epoch = int(getattr(trainer, "current_epoch", -1) if epoch is None else epoch)
    checkpoint = _compact_checkpoint_payload(
        module,
        checkpoint_epoch,
        getattr(trainer, "global_step", -1),
        reason=reason,
    )
    _atomic_torch_save(checkpoint, last_path)

    metadata = {
        "reason": str(reason),
        "epoch": checkpoint_epoch,
        "global_step": int(getattr(trainer, "global_step", -1)),
        "time_unix": time.time(),
        "last_checkpoint": str(last_path),
        "last_checkpoint_is_symlink": bool(last_path.is_symlink()),
        "checkpoint_format": checkpoint["format"],
        "base_checkpoint": checkpoint["base_checkpoint"],
        "visual_backbone": checkpoint["visual_backbone"],
        "backbone_weights": checkpoint["backbone_weights"],
        "backbone_sha256": checkpoint["backbone_sha256"],
        "best_checkpoints": _checkpoint_callback_metadata(checkpoint_callbacks),
        "pid": int(os.getpid()),
    }
    _atomic_write_json(last_path.parent / "last_checkpoint_info.json", metadata)
    return last_path


class MaterializedLastCheckpointCallback(Callback):
    """Save a concrete last.ckpt on normal finish and on a single Ctrl+C."""

    def __init__(self, ckpt_dir, checkpoint_callbacks=None):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.checkpoint_callbacks = checkpoint_callbacks or {}
        self.last_checkpoint_path = None
        self._save_in_progress = False
        self._saved = False
        self._last_started_epoch = -1

    def on_train_epoch_start(self, trainer, pl_module):
        self._last_started_epoch = int(trainer.current_epoch)

    def _save(self, trainer, reason, raise_errors):
        if not trainer.is_global_zero or self._saved or self._save_in_progress:
            return
        self._save_in_progress = True
        checkpoint_epoch = (
            self._last_started_epoch
            if self._last_started_epoch >= 0
            else int(getattr(trainer, "current_epoch", -1))
        )
        print(
            f"Saving materialized last checkpoint ({reason}, "
            f"epoch={checkpoint_epoch}, global_step={trainer.global_step})...",
            flush=True,
        )
        try:
            self.last_checkpoint_path = save_materialized_last_checkpoint(
                trainer,
                self.ckpt_dir,
                reason=reason,
                checkpoint_callbacks=self.checkpoint_callbacks,
                epoch=checkpoint_epoch,
            )
            print(
                f"Materialized last checkpoint ({reason}): "
                f"{self.last_checkpoint_path} "
                f"(epoch={checkpoint_epoch}, global_step={trainer.global_step})"
            )
            self._saved = True
        except Exception as exc:
            message = f"Warning: failed to materialize last checkpoint during {reason}: {exc}"
            if raise_errors:
                raise RuntimeError(message) from exc
            print(message)
        finally:
            self._save_in_progress = False

    def on_fit_end(self, trainer, pl_module):
        self._save(trainer, reason="fit_end", raise_errors=True)

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            self._save(trainer, reason="keyboard_interrupt", raise_errors=False)

    def on_keyboard_interrupt(self, trainer, pl_module):
        self._save(trainer, reason="keyboard_interrupt", raise_errors=False)


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    args.checkpoint = str(Path(args.checkpoint).expanduser().resolve(strict=False))
    if args.visual_backbone == "dinov3_hplus":
        if not args.dino_weights:
            raise ValueError("--dino_weights is required for --visual_backbone=dinov3_hplus")
        args.dino_weights = str(Path(args.dino_weights).expanduser().resolve(strict=False))
        if not Path(args.dino_weights).is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {args.dino_weights}")
    elif args.dino_weights:
        args.dino_weights = str(Path(args.dino_weights).expanduser().resolve(strict=False))
    if args.audit_nonfinite_grads and not args.nonfinite_audit_dir:
        args.nonfinite_audit_dir = os.path.join(ft_dir, "amp_audits", args.exp_name, "capture")
    if args.nonfinite_audit_dir:
        args.nonfinite_audit_dir = os.path.abspath(os.path.expanduser(args.nonfinite_audit_dir))
    data_dirs = resolve_data_dirs(args)
    print("Resolved training data roots:")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")
    
    hamer_root = os.path.join(workspace_dir, "hamer")
    os.chdir(hamer_root)
    
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    num_gpus = len(gpu_ids)
    
    lr_scaled = linear_scaled_learning_rate(args.lr, num_gpus)
    print(
        "\n>>> Learning-rate configuration: "
        f"base_lr={args.lr}, num_gpus={num_gpus}, effective_lr={lr_scaled} "
        "(linear scaling)"
    )
    
    print(f"Loading finetuned Hamer weights from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    from hamer.configs import get_config
    
    # We can use the original model_config.yaml from hamer
    model_cfg_path = os.path.join(workspace_dir, 'hamer/_DATA/hamer_ckpts/model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
        
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()
        
    print("Initializing HAMER_Tactile Wrapper...")
    tactile_loss_config = tactile_loss_config_from_args(args)
    backbone_feature_layers = tuple(int(layer) for layer in _split_csv(args.backbone_feature_layers))
    model = OpenTouchHAMER_TactileWrapper(
        cfg=model_cfg,
        learning_rate=lr_scaled,
        tactile_loss_config=tactile_loss_config,
        tactile_only_forward=args.tactile_only_forward,
        tactile_loss_scale=args.tactile_loss_scale,
        lr_warmup_epochs=args.lr_warmup_epochs,
        frame_low_volume_thr=args.frame_low_volume_thr,
        frame_high_volume_thr=args.frame_high_volume_thr,
        sync_train_logs=args.sync_train_logs,
        audit_nonfinite_grads=args.audit_nonfinite_grads,
        nonfinite_audit_dir=args.nonfinite_audit_dir,
        audit_checkpoint_path=args.checkpoint,
        audit_seed=args.seed,
        tactile_head_type=args.tactile_head_type,
        backbone_feature_layers=backbone_feature_layers,
        visual_backbone=args.visual_backbone,
        dino_weights=args.dino_weights or "",
    )
    model.visual_backbone_model_name = (
        model.backbone.MODEL_NAME if args.visual_backbone == "dinov3_hplus" else "hamer_vit_h"
    )
    model.backbone_weights_path = args.dino_weights if args.visual_backbone == "dinov3_hplus" else args.checkpoint
    print(f"Computing backbone SHA256: {model.backbone_weights_path}")
    model.backbone_weights_sha256 = file_sha256(model.backbone_weights_path)
    
    # Do a dummy forward pass to initialize lazy layers before shape-aware loading.
    dummy_input = torch.zeros(1, 3, model_cfg.MODEL.IMAGE_SIZE, model_cfg.MODEL.IMAGE_SIZE)
    with torch.no_grad():
        dummy_feat = model._extract_tactile_features(dummy_input[:, :, :, 32:-32])
        model.tactile_head(dummy_feat)
        print(f"Tactile head initialized with output dim: {model.tactile_dim}")

    print("Loading compatible weights from fine-tuned model...")
    missing, unexpected = load_compatible_state_dict(
        model,
        args.checkpoint,
        load_backbone=args.visual_backbone == "hamer",
    )
    print(f"Missing keys during load: {len(missing)}")
        
    train_dataset = OpenTouchTactileDataset(
        cfg=model_cfg,
        split="train",
        data_dir=data_dirs,
        train=True,
        index_workers=args.index_workers,
        index_chunksize=args.index_chunksize,
        index_backend=args.index_backend,
        index_cache_dir=args.index_cache_dir,
        rebuild_index=args.rebuild_index,
        index_cache_timeout=args.index_cache_timeout,
        tactile_only=args.tactile_only_forward,
    )
    
    val_dataset = None
    if not args.skip_validation:
        val_dataset = OpenTouchTactileDataset(
            cfg=model_cfg,
            split="val",
            data_dir=data_dirs,
            train=False,
            index_workers=args.index_workers,
            index_chunksize=args.index_chunksize,
            index_backend=args.index_backend,
            index_cache_dir=args.index_cache_dir,
            rebuild_index=args.rebuild_index,
            index_cache_timeout=args.index_cache_timeout,
            tactile_only=args.tactile_only_forward,
        )
    
    if args.quick_test:
        train_dataset.samples = train_dataset.samples[:64]
        if val_dataset is not None:
            val_dataset.samples = val_dataset.samples[:32]
        args.epochs = 1
        args.num_workers = 0
        args.val_num_workers = 0

    if len(train_dataset) == 0:
        raise RuntimeError(
            "Training dataset is empty after scanning resolved roots. "
            "Check --datasets/--data_dir and the extracted meta.json structure."
        )

    val_num_workers = args.num_workers if args.val_num_workers is None else args.val_num_workers
    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = make_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=val_num_workers,
            # Validation workers otherwise stay alive throughout every train
            # epoch and make DDP interrupt cleanup unnecessarily expensive.
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
        )
    else:
        print("Validation dataset is empty; training will run without validation metrics/checkpoint monitoring.")
    
    ckpt_dir = os.path.join(ft_dir, "checkpoints", args.exp_name) if not args.quick_test else os.path.join(ft_dir, "checkpoints_test")
    provenance = write_run_provenance(args, data_dirs, tactile_loss_config, model, ckpt_dir, num_gpus, lr_scaled)
    val_metrics_text_logger = ValidationMetricsTextLogger(os.path.join(ckpt_dir, "val_metrics.txt"), config_record=provenance)
    checkpoint_callbacks = {}
    if val_loader is not None and not args.skip_checkpointing:
        print("Best checkpoint monitors: val/eval_rmse (min), val/eval_volumetric_iou (max)")
        checkpoint_callbacks["rmse-best"] = CompactBestCheckpoint(
            dirpath=ckpt_dir,
            filename="best_rmse",
            monitor="val/eval_rmse",
            mode="min",
        )
        checkpoint_callbacks["viou-best"] = CompactBestCheckpoint(
            dirpath=ckpt_dir,
            filename="best_viou",
            monitor="val/eval_volumetric_iou",
            mode="max",
        )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    materialized_last_callback = None
    if not args.skip_checkpointing:
        materialized_last_callback = MaterializedLastCheckpointCallback(
            ckpt_dir=ckpt_dir,
            checkpoint_callbacks=checkpoint_callbacks,
        )
    
    if num_gpus > 1:
        strategy = "ddp_find_unused_parameters_true" if args.ddp_find_unused_parameters else "ddp_find_unused_parameters_false"
    else:
        strategy = "auto"
    
    if args.use_wandb:
        # Pass config through WandbLogger/wandb.init instead of depending on the
        # version-specific shape of logger.experiment.config.
        logger = WandbLogger(
            project="opentouch-hamer-tactile-ft",
            name=args.exp_name,
            config=provenance,
        )
    else:
        logger = True
    
    callbacks = [*checkpoint_callbacks.values(), lr_monitor, val_metrics_text_logger]
    if materialized_last_callback is not None:
        callbacks.append(materialized_last_callback)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        precision=args.trainer_precision,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )
    
    if val_loader is None:
        trainer.fit(model, train_dataloaders=train_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_last_path = materialized_last_callback.last_checkpoint_path if materialized_last_callback else None
    if not args.skip_checkpointing and trainer.is_global_zero and final_last_path is None:
        final_last_path = save_final_last_checkpoint(trainer, ckpt_dir, checkpoint_callbacks)
    if trainer.is_global_zero:
        print("\nFine-tuning completed.")
        for selector, callback in checkpoint_callbacks.items():
            print(f"{selector}: {callback.best_model_path} (score={callback.best_model_score})")
        if not args.skip_checkpointing:
            print(f"Final checkpoint: {final_last_path}")

if __name__ == "__main__":
    main()
