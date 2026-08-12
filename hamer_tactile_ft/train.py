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
import uuid
from dataclasses import asdict
from pathlib import Path

from process_lifecycle import (
    configure_supervised_process,
    initialize_worker_parent_death_signal,
)

configure_supervised_process()

sys.argv[0] = os.path.abspath(__file__)

import faulthandler
faulthandler.enable()

import pytorch_lightning as pl
import cv2

cv2.setNumThreads(0)

from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import torch
torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(workspace_dir)
sys.path.append(os.path.join(workspace_dir, "hamer"))
sys.path.append(ft_dir)

from dataset import (
    DATASET_ROOTS,
    INDEX_CACHE_VERSION,
    OpenTouchTactileDataset,
    canonical_dataset_name,
    canonical_dataset_filter,
    ddp_global_rank,
    release_unused_python_heap,
)
from hamer_tactile import DinoTactileModel, parse_input_resolution
from losses import TactileLossConfig
from tactile_metrics import (
    CompactTouchAnythingProtocolAccumulator,
    TOUCHANYTHING_CONTACT_THRESHOLD,
    TOUCHANYTHING_MIN_CONTACT_RATIO,
    summarize_compact_touchanything_protocol,
    touchanything_protocol_group_key,
)

CORE_LOCATION_DISTRIBUTION_POWER = 2.0
CORE_LOCATION_MIN_GT_PEAK = 0.05

def linear_scaled_learning_rate(base_lr, num_gpus):
    if int(num_gpus) < 1:
        raise ValueError("num_gpus must be at least 1")
    return float(base_lr) * int(num_gpus)


def gradient_clip_triggered(grad_norm, clip_value):
    return float(not math.isfinite(float(grad_norm)) or (clip_value > 0.0 and grad_norm > clip_value))


class ValidationMetricsTextLogger(Callback):
    TRAIN_EPOCH_KEYS = {
        "train/loss_epoch_global",
        "train/epoch_grad_norm_mean",
        "train/epoch_grad_norm_max",
        "train/grad_clip_trigger_rate_epoch",
        "train/nonfinite_grad_rate_epoch",
        "train/effective_lr_epoch_end",
        "train/data_wait_fraction_epoch",
        "train/data_wait_mean_ms_epoch",
        "train/host_step_mean_ms_epoch",
    }

    def __init__(self, output_path, config_record=None, append=False):
        super().__init__()
        self.output_path = output_path
        self.config_record = config_record
        self.append = bool(append)

    def on_fit_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        mode = "a" if self.append and os.path.isfile(self.output_path) else "w"
        with open(self.output_path, mode, encoding="utf-8") as f:
            if self.config_record is not None:
                record = {
                    "record_type": "resume_config" if mode == "a" else "config",
                    **self.config_record,
                }
                f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if (
                not str(name).startswith("val/")
                and not str(name).startswith("train/loss_")
                and str(name) not in self.TRAIN_EPOCH_KEYS
            ):
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


def optimizer_parameter_groups(named_parameters, weight_decay):
    weight_decay = float(weight_decay)
    decay_params = []
    no_decay_params = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if any(no_decay_name in name for no_decay_name in ("bias", "norm", "LayerNorm")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def lr_warmup_step_count(total_steps, max_epochs, warmup_epochs):
    total_steps = int(total_steps)
    max_epochs = max(int(max_epochs), 1)
    warmup_steps = int(round(total_steps * max(int(warmup_epochs), 0) / max_epochs))
    if warmup_steps > 0:
        warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
    return warmup_steps


def resolve_lr_decay_milestones(value, total_steps):
    total_steps = int(total_steps)
    if total_steps <= 1:
        raise ValueError("Multistep scheduling requires at least two total training steps")
    tokens = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not tokens:
        raise ValueError("--lr_decay_milestones must contain at least one fraction or step")

    milestones = []
    for token in tokens:
        try:
            numeric = float(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --lr_decay_milestones value {token!r}; expected fractions or steps"
            ) from exc
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("--lr_decay_milestones values must be finite and positive")
        if numeric < 1.0:
            step = int(round(numeric * total_steps))
        elif numeric.is_integer():
            step = int(numeric)
        else:
            raise ValueError(
                f"Absolute LR milestone {token!r} must be an integer number of steps"
            )
        if not 0 < step < total_steps:
            raise ValueError(
                f"Resolved LR milestone {step} must lie strictly between 0 and {total_steps}"
            )
        milestones.append(step)

    if milestones != sorted(set(milestones)):
        raise ValueError("--lr_decay_milestones must resolve to unique increasing steps")
    return milestones


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


class TactileTrainingModule(DinoTactileModel):
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
        tactile_head_type="dense_v2_dino_rezero",
        backbone_feature_layers=(8, 16, 24, 32),
        visual_backbone="dinov3_hplus",
        dino_weights="",
        dino_residual_max_scale=0.10,
        dino_residual_rms_budget=0.50,
        bbox_rescale_factor=2.0,
        bbox_source_policy="sam3_only",
        pool_layout="fullgrid32",
        decoder_dropout_scale=1.0,
        input_resolution=(256, 192),
        pool_output_channels=32,
        optimizer_weight_decay=1e-4,
        lr_scheduler="cosine",
        lr_decay_milestones="0.5,0.75",
        lr_decay_gamma=0.1,
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
            dino_residual_max_scale=dino_residual_max_scale,
            dino_residual_rms_budget=dino_residual_rms_budget,
            pool_layout=pool_layout,
            decoder_dropout_scale=decoder_dropout_scale,
            input_resolution=input_resolution,
            pool_output_channels=pool_output_channels,
        )
        self.learning_rate = learning_rate
        self.optimizer_weight_decay = float(optimizer_weight_decay)
        self.lr_scheduler_name = str(lr_scheduler)
        self.lr_decay_milestones = str(lr_decay_milestones)
        self.lr_decay_milestones_resolved = []
        self.lr_decay_gamma = float(lr_decay_gamma)
        self.bbox_rescale_factor = float(bbox_rescale_factor)
        self.bbox_source_policy = str(bbox_source_policy)
        self.lr_warmup_epochs = int(lr_warmup_epochs)
        self.sync_train_logs = bool(sync_train_logs)
        self.frame_low_volume_thr = float(frame_low_volume_thr)
        self.frame_high_volume_thr = float(frame_high_volume_thr)
        self.visual_backbone_model_name = self.backbone.MODEL_NAME
        self.backbone_weights_path = str(dino_weights)
        self.backbone_weights_sha256 = ""
        if tactile_loss_config is not None:
            self.set_tactile_loss_config(tactile_loss_config)
        
        self.automatic_optimization = True
        
        print("Freezing all non-tactile modules. Only training the tactile head...")
        # Freeze the DINO backbone and keep only the tactile head trainable.
        for param in self.parameters():
            param.requires_grad = False

        for param in self.tactile_head.parameters():
            param.requires_grad = True

        self._val_eval_stats = None
        self._val_touchanything_protocol_stats = CompactTouchAnythingProtocolAccumulator()
        self._grad_clip_trigger_count = 0
        self._grad_clip_step_count = 0
        self._grad_norm_finite_step_count = 0
        self._nonfinite_grad_step_count = 0
        self._grad_norm_sum = 0.0
        self._grad_norm_max = 0.0
        self._effective_lr_epoch_end = float(self.learning_rate)
        self._train_epoch_summary = {}
        self._train_metric_sums = {}
        self._train_metric_weight = 0
        self._global_train_summary_epoch = None
        self._global_train_summary_cache = None
        self._throughput_started_at = None
        self._last_train_batch_end_at = None
        self._train_batch_started_at = None
        self._data_wait_seconds = 0.0
        self._host_step_seconds = 0.0
        self._timed_train_batches = 0
        self._loading_tactile_resume = False

    def _resume_contract(self):
        return {
            "tactile_head_type": str(self.tactile_head_type),
            "input_resolution": list(self.input_resolution),
            "backbone_feature_layers": list(self.backbone_feature_layers),
            "pool_layout": str(self.pool_layout),
            "pool_output_channels": int(self.pool_output_channels),
            "dataset_filter": list(getattr(self, "dataset_filter", ())),
            "bbox_rescale_factor": float(self.bbox_rescale_factor),
            "bbox_source_policy": str(self.bbox_source_policy),
            "loss_config": asdict(self.tactile_loss_config),
        }

    def on_save_checkpoint(self, checkpoint):
        """Keep exact trainer state while rebuilding the frozen DINO from disk."""
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            name: value
            for name, value in state_dict.items()
            if not name.startswith("backbone.")
        }
        checkpoint["format"] = "tactile_resume_v1"
        checkpoint["backbone_weights"] = str(self.backbone_weights_path or "")
        checkpoint["backbone_sha256"] = str(self.backbone_weights_sha256 or "")
        checkpoint["tactile_head_type"] = str(self.tactile_head_type)
        checkpoint["input_resolution"] = list(self.input_resolution)
        checkpoint["dataset_filter"] = list(getattr(self, "dataset_filter", ()))
        checkpoint["wandb_run_id"] = str(
            getattr(self, "wandb_run_id", "") or ""
        )
        checkpoint["resume_contract"] = self._resume_contract()

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("format") != "tactile_resume_v1":
            return
        checkpoint_wandb_run_id = str(
            checkpoint.get("wandb_run_id", "") or ""
        )
        configured_wandb_run_id = str(
            getattr(self, "wandb_run_id", "") or ""
        )
        if (
            checkpoint_wandb_run_id
            and configured_wandb_run_id
            and checkpoint_wandb_run_id != configured_wandb_run_id
        ):
            raise RuntimeError(
                "Resume checkpoint WandB run ID mismatch: "
                f"checkpoint={checkpoint_wandb_run_id}, "
                f"current={configured_wandb_run_id}"
            )
        if checkpoint_wandb_run_id:
            self.wandb_run_id = checkpoint_wandb_run_id
        expected_sha = str(getattr(self, "backbone_weights_sha256", "") or "")
        checkpoint_sha = str(checkpoint.get("backbone_sha256", "") or "")
        if expected_sha and checkpoint_sha and expected_sha != checkpoint_sha:
            raise RuntimeError(
                "Resume checkpoint DINO SHA256 mismatch: "
                f"checkpoint={checkpoint_sha}, current={expected_sha}"
            )
        if str(checkpoint.get("tactile_head_type", "")) != self.tactile_head_type:
            raise RuntimeError(
                "Resume checkpoint tactile head mismatch: "
                f"checkpoint={checkpoint.get('tactile_head_type')}, "
                f"current={self.tactile_head_type}"
            )
        checkpoint_resolution = tuple(
            int(value) for value in checkpoint.get("input_resolution", ())
        )
        if checkpoint_resolution and checkpoint_resolution != self.input_resolution:
            raise RuntimeError(
                "Resume checkpoint input resolution mismatch: "
                f"checkpoint={checkpoint_resolution}, current={self.input_resolution}"
            )
        checkpoint_contract = checkpoint.get("resume_contract")
        if checkpoint_contract is not None:
            current_contract = self._resume_contract()
            mismatches = {
                key: {
                    "checkpoint": checkpoint_contract.get(key),
                    "current": current_contract.get(key),
                }
                for key in current_contract
                if checkpoint_contract.get(key) != current_contract.get(key)
            }
            if mismatches:
                raise RuntimeError(
                    "Resume checkpoint configuration mismatch: "
                    + json.dumps(mismatches, sort_keys=True)
                )
        self._loading_tactile_resume = True

    def load_state_dict(self, state_dict, strict=True):
        if not self._loading_tactile_resume:
            return super().load_state_dict(state_dict, strict=strict)
        incompatible = super().load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if not name.startswith("backbone.")
        ]
        self._loading_tactile_resume = False
        if unexpected or missing:
            raise RuntimeError(
                "Resume checkpoint state mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return incompatible

    def on_train_start(self):
        self._throughput_started_at = time.perf_counter()
        self._last_train_batch_end_at = self._throughput_started_at
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_end(self):
        if self._throughput_started_at is None:
            return
        elapsed = max(time.perf_counter() - self._throughput_started_at, 1e-6)
        world_size = int(getattr(self.trainer, "world_size", 1))
        accumulation = int(getattr(self.trainer, "accumulate_grad_batches", 1))
        batch_size = int(getattr(self, "batch_size_config", 0))
        estimated_samples = int(self.global_step) * batch_size * world_size * accumulation
        samples_per_second = estimated_samples / elapsed
        tokens_per_second = samples_per_second * int(self.pool_valid_tokens)
        train_dataset = getattr(self.trainer.train_dataloader, "dataset", None)
        dataset_size = len(train_dataset) if train_dataset is not None else 0
        estimated_epoch_seconds = dataset_size / max(samples_per_second, 1e-12)
        peak_gib = (
            torch.cuda.max_memory_reserved(self.device) / (1024.0 ** 3)
            if torch.cuda.is_available()
            else 0.0
        )
        if self.trainer.is_global_zero:
            print(
                "Training performance: "
                f"peak_reserved={peak_gib:.2f} GiB/GPU, "
                f"samples_per_second={samples_per_second:.2f}, "
                f"spatial_tokens_per_second={tokens_per_second:.2f}, "
                f"estimated_epoch_seconds={estimated_epoch_seconds:.1f}, "
                f"elapsed={elapsed:.1f}s"
            )
            if self.logger is not None:
                self.logger.log_metrics(
                    {
                        "performance/peak_reserved_gib": peak_gib,
                        "performance/samples_per_second": samples_per_second,
                        "performance/spatial_tokens_per_second": tokens_per_second,
                        "performance/estimated_epoch_seconds": estimated_epoch_seconds,
                    },
                    step=int(self.global_step),
                )

    def on_train_epoch_start(self):
        self._grad_clip_trigger_count = 0
        self._grad_clip_step_count = 0
        self._grad_norm_finite_step_count = 0
        self._nonfinite_grad_step_count = 0
        self._grad_norm_sum = 0.0
        self._grad_norm_max = 0.0
        self._train_metric_sums = {}
        self._train_metric_weight = 0
        self._global_train_summary_epoch = None
        self._global_train_summary_cache = None
        self._data_wait_seconds = 0.0
        self._host_step_seconds = 0.0
        self._timed_train_batches = 0
        self._last_train_batch_end_at = time.perf_counter()
        self._train_batch_started_at = None

    def on_train_batch_start(self, batch, batch_idx):
        del batch, batch_idx
        now = time.perf_counter()
        if self._last_train_batch_end_at is not None:
            self._data_wait_seconds += max(
                now - self._last_train_batch_end_at,
                0.0,
            )
        self._train_batch_started_at = now

    def on_train_batch_end(self, outputs, batch, batch_idx):
        del outputs, batch, batch_idx
        now = time.perf_counter()
        if self._train_batch_started_at is not None:
            self._host_step_seconds += max(
                now - self._train_batch_started_at,
                0.0,
            )
            self._timed_train_batches += 1
        self._last_train_batch_end_at = now

    def on_before_optimizer_step(self, optimizer):
        gradient_names = []
        gradients = []
        for name, parameter in self.tactile_head.named_parameters():
            if parameter.grad is not None:
                gradient_names.append(name)
                gradients.append(parameter.grad.detach())
        if gradients:
            try:
                parameter_norms = torch._foreach_norm(gradients, 2.0)
            except (AttributeError, RuntimeError):
                parameter_norms = [
                    torch.linalg.vector_norm(gradient) for gradient in gradients
                ]
            stacked_norms = torch.stack(
                [norm.to(device=self.device, dtype=torch.float32) for norm in parameter_norms]
            )
            grad_norm = torch.linalg.vector_norm(stacked_norms)
        else:
            stacked_norms = torch.zeros(0, device=self.device, dtype=torch.float32)
            grad_norm = torch.zeros((), device=self.device, dtype=torch.float32)
        grad_norm_value = float(grad_norm.item())
        first_nonfinite_parameter = None
        if not math.isfinite(grad_norm_value):
            for name, norm in zip(gradient_names, stacked_norms):
                if not math.isfinite(float(norm.item())):
                    first_nonfinite_parameter = name
                    break
            if first_nonfinite_parameter is None:
                first_nonfinite_parameter = "<aggregate_gradient_norm>"
        clip_value = float(getattr(self.trainer, "gradient_clip_val", 0.0) or 0.0)
        clip_trigger = gradient_clip_triggered(grad_norm_value, clip_value)
        self._grad_clip_trigger_count += int(clip_trigger)
        self._grad_clip_step_count += 1
        if math.isfinite(grad_norm_value):
            self._grad_norm_finite_step_count += 1
            self._grad_norm_sum += grad_norm_value
            self._grad_norm_max = max(self._grad_norm_max, grad_norm_value)
        else:
            self._nonfinite_grad_step_count += 1
        self._effective_lr_epoch_end = float(optimizer.param_groups[0]["lr"])
        self.log(
            "train/grad_norm_pre_clip",
            grad_norm,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/grad_clip_trigger",
            clip_trigger,
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/nonfinite_grad_step",
            float(first_nonfinite_parameter is not None),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train/effective_lr",
            float(optimizer.param_groups[0]["lr"]),
            on_step=True,
            on_epoch=False,
            logger=True,
            sync_dist=False,
        )
        if first_nonfinite_parameter is not None:
            raise FloatingPointError(
                "Non-finite tactile-head gradient detected before optimizer step: "
                f"parameter={first_nonfinite_parameter}, epoch={self.current_epoch}, "
                f"global_step={self.global_step}, precision={self.trainer.precision}."
            )

    def on_train_epoch_end(self):
        summary = self._current_train_epoch_summary()
        summary_was_cached = (
            self._global_train_summary_epoch == int(getattr(self, "current_epoch", 0))
            and self._global_train_summary_cache is not None
        )
        global_summary = self._cached_global_train_epoch_summary()
        self._train_epoch_summary = global_summary
        if not summary_was_cached:
            for name, value in global_summary.items():
                if not name.startswith("train/loss_"):
                    continue
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=False,
                )
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
        for name in (
            "train/data_wait_fraction_epoch",
            "train/data_wait_mean_ms_epoch",
            "train/host_step_mean_ms_epoch",
        ):
            self.log(
                name,
                global_summary[name],
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
        metric_names = tuple(sorted(self._train_metric_sums))
        metric_sums = [
            self._train_metric_sums[name].to(device=self.device, dtype=torch.float64)
            for name in metric_names
        ]
        local = torch.tensor(
            [
                self._grad_norm_sum,
                float(self._grad_norm_finite_step_count),
                self._grad_norm_max,
                float(self._grad_clip_trigger_count),
                self._effective_lr_epoch_end,
                float(self._nonfinite_grad_step_count),
                float(self._grad_clip_step_count),
                float(self._train_metric_weight),
                self._data_wait_seconds,
                self._host_step_seconds,
                float(self._timed_train_batches),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        if metric_sums:
            local = torch.cat([local, torch.stack(metric_sums)])
        if getattr(self.trainer, "world_size", 1) > 1:
            gathered = self.all_gather(local).reshape(-1, local.numel())
        else:
            gathered = local.unsqueeze(0)
        finite_steps = gathered[:, 1].sum().clamp_min(1.0)
        total_steps = gathered[:, 6].sum().clamp_min(1.0)
        train_metric_weight = gathered[:, 7].sum().clamp_min(1.0)
        data_wait_seconds = gathered[:, 8].sum()
        host_step_seconds = gathered[:, 9].sum()
        timed_batches = gathered[:, 10].sum().clamp_min(1.0)
        summary = {
            "train/epoch_grad_norm_mean": float((gathered[:, 0].sum() / finite_steps).item()),
            "train/epoch_grad_norm_max": float(gathered[:, 2].max().item()),
            "train/grad_clip_trigger_rate_epoch": float((gathered[:, 3].sum() / total_steps).item()),
            "train/nonfinite_grad_rate_epoch": float((gathered[:, 5].sum() / total_steps).item()),
            "train/effective_lr_epoch_end": float(gathered[:, 4].mean().item()),
            "train/data_wait_fraction_epoch": float(
                (
                    data_wait_seconds
                    / (data_wait_seconds + host_step_seconds).clamp_min(1e-12)
                ).item()
            ),
            "train/data_wait_mean_ms_epoch": float(
                (1000.0 * data_wait_seconds / timed_batches).item()
            ),
            "train/host_step_mean_ms_epoch": float(
                (1000.0 * host_step_seconds / timed_batches).item()
            ),
        }
        for index, name in enumerate(metric_names, start=11):
            summary[name] = float((gathered[:, index].sum() / train_metric_weight).item())
        # Compatibility alias for existing dashboards and report readers.
        current_name = "train/loss_current_ramp_epoch_global"
        if current_name in summary:
            summary["train/loss_epoch_global"] = summary[current_name]
        return summary

    def _cached_global_train_epoch_summary(self):
        epoch = int(getattr(self, "current_epoch", 0))
        if self._global_train_summary_epoch != epoch or self._global_train_summary_cache is None:
            self._global_train_summary_cache = self._global_train_epoch_summary()
            self._global_train_summary_epoch = epoch
        return dict(self._global_train_summary_cache)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self
                
    def training_step(self, batch, batch_idx):
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)
        if not bool(torch.isfinite(loss.detach()).all().item()):
            nonfinite_outputs = [
                name
                for name in ("pred_logits", "pred_tactile")
                if name in output
                and not bool(torch.isfinite(output[name].detach()).all().item())
            ]
            nonfinite_components = [
                name
                for name, value in output.get("losses", {}).items()
                if torch.is_tensor(value)
                and not bool(torch.isfinite(value.detach()).all().item())
            ]
            nonfinite_inputs = [
                name
                for name in ("img", "tactile_signal", "palm_mask")
                if name in batch
                and torch.is_tensor(batch[name])
                and not bool(torch.isfinite(batch[name].detach()).all().item())
            ]
            sample_ids = list(batch.get("sample_uid", ()))[:8]
            raise FloatingPointError(
                "Non-finite training loss detected before backward: "
                f"epoch={self.current_epoch}, global_step={self.global_step}, "
                f"batch_idx={batch_idx}, precision={self.trainer.precision}, "
                f"outputs={nonfinite_outputs}, components={nonfinite_components}, "
                f"inputs={nonfinite_inputs}, sample_uids={sample_ids}."
            )
        batch_size = int(batch["img"].shape[0])
        metric_values = {
            "train/loss_current_ramp_epoch_global": loss.detach(),
            "train/loss_full_ramp_reference_epoch_global": output["losses"].get(
                "loss_full_ramp_reference", loss.detach()
            ),
            "train/loss_direct_raw_epoch_global": output["losses"].get(
                "loss_direct_raw", loss.detach()
            ),
            "train/loss_smooth_l1_raw_epoch_global": output["losses"].get(
                "loss_smooth_l1_raw", loss.detach().new_zeros(())
            ),
            "train/loss_logit_bce_raw_epoch_global": output["losses"].get(
                "loss_logit_bce_raw", loss.detach().new_zeros(())
            ),
            "train/loss_location_raw_epoch_global": output["losses"].get(
                "loss_location_raw", loss.detach().new_zeros(())
            ),
            "train/loss_location_weighted_epoch_global": output["losses"].get(
                "loss_location_weighted", loss.detach().new_zeros(())
            ),
            "train/loss_contact_raw_epoch_global": output["losses"].get(
                "loss_contact_raw", loss.detach().new_zeros(())
            ),
            "train/loss_contact_weighted_epoch_global": output["losses"].get(
                "loss_contact_weighted", loss.detach().new_zeros(())
            ),
        }
        for name, value in metric_values.items():
            weighted = value.detach().to(dtype=torch.float64) * float(batch_size)
            if name not in self._train_metric_sums:
                self._train_metric_sums[name] = weighted.clone()
            else:
                self._train_metric_sums[name].add_(weighted)
        self._train_metric_weight += batch_size
        
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self.sync_train_logs,
            batch_size=batch_size,
        )
        self._log_tactile_loss_breakdown(
            "train",
            output,
            on_step=True,
            batch_size=batch_size,
        )
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

        batch_size = int(batch["img"].shape[0])
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self._log_tactile_loss_breakdown(
            "val",
            output,
            on_step=False,
            batch_size=batch_size,
        )
        diagnostics = self.tactile_head.feature_diagnostics()
        layer_indices = tuple(
            getattr(self.tactile_head, "refinement_layer_indices", ())
        )
        for diagnostic_key, metric_name in (
            ("level_weight", "rezero_level_weight"),
            ("projected_rms", "rezero_projected_rms"),
            ("raw_residual_rms", "rezero_raw_residual_rms"),
            ("residual_saturation", "rezero_residual_saturation"),
            ("effective_contribution", "rezero_effective_contribution"),
        ):
            values = diagnostics.get(diagnostic_key)
            if values is None:
                continue
            for layer, value in zip(layer_indices, values):
                self.log(
                    f"val/{metric_name}_{layer}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        for diagnostic_key, metric_name in (
            ("gate_raw", "rezero_gate_raw"),
            ("gate_effective", "rezero_gate_effective"),
            ("delta_rms_pre_budget", "rezero_delta_rms_pre_budget"),
            ("delta_rms_post_budget", "rezero_delta_rms_post_budget"),
            ("delta_to_base_rms", "rezero_delta_to_base_rms"),
            ("budget_clip_rate", "rezero_budget_clip_rate"),
            ("base_rms", "rezero_base_rms"),
            ("final_rms", "rezero_final_rms"),
        ):
            value = diagnostics.get(diagnostic_key)
            if value is not None:
                self.log(
                    f"val/{metric_name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
        self._accumulate_val_eval_stats(
            pred_tactile,
            gt_tactile,
            has_tactile,
            palm_mask,
            sequence_keys=batch.get("sequence_key"),
            query_aliases=batch.get("query_alias"),
            frame_indices=batch.get("frame_idx"),
            datasets=batch.get("dataset"),
        )
        return loss

    def on_validation_epoch_start(self):
        if self.trainer.sanity_checking:
            self._train_epoch_summary = {}
        else:
            self._train_epoch_summary = self._cached_global_train_epoch_summary()
            for name, value in self._train_epoch_summary.items():
                if not name.startswith("train/loss_"):
                    continue
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    sync_dist=False,
                )
        self._val_eval_stats = None
        self._val_eval_stats_by_domain = {}
        self._val_touchanything_protocol_stats = CompactTouchAnythingProtocolAccumulator()

    def _accumulate_val_eval_stats(
        self,
        pred_tactile,
        gt_tactile,
        has_tactile,
        palm_mask,
        sequence_keys=None,
        query_aliases=None,
        frame_indices=None,
        datasets=None,
        accumulator_name=None,
        touchanything_protocol=False,
    ):
        if datasets is not None and accumulator_name is None:
            self._accumulate_val_eval_stats(
                pred_tactile,
                gt_tactile,
                has_tactile,
                palm_mask,
                sequence_keys=sequence_keys,
                query_aliases=query_aliases,
                frame_indices=frame_indices,
                datasets=None,
                accumulator_name="__overall__",
                touchanything_protocol=False,
            )
            canonical_domains = [canonical_dataset_name(value) for value in datasets]
            for domain_name, metric_name in (
                ("OpenTouch", "opentouch"),
                ("TouchAnything", "touchanything"),
            ):
                domain_mask = torch.as_tensor(
                    [value == domain_name for value in canonical_domains],
                    dtype=has_tactile.dtype,
                    device=has_tactile.device,
                )
                if bool((domain_mask > 0.5).any().item()):
                    self._accumulate_val_eval_stats(
                        pred_tactile,
                        gt_tactile,
                        has_tactile * domain_mask,
                        palm_mask,
                        sequence_keys=sequence_keys,
                        query_aliases=query_aliases,
                        frame_indices=frame_indices,
                        datasets=None,
                        accumulator_name=metric_name,
                        touchanything_protocol=metric_name == "touchanything",
                    )
            return

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

        if touchanything_protocol and (
            sequence_keys is None or query_aliases is None or frame_indices is None
        ):
            raise ValueError(
                "TouchAnything-compatible validation metrics require sequence_key, query_alias, and frame_idx"
            )

        diff = (pred - gt) * palm
        palm_values = palm.sum()
        active_mask = ((gt > active_thr).float() * palm)
        background_mask = ((gt <= background_thr).float() * palm)
        pred_active = ((pred > active_thr).float() * palm)
        contact_eval_mask = ((((gt >= active_thr) | (gt <= background_thr)).float()) * palm)
        contact_pred = ((contact > active_thr).float() * contact_eval_mask)
        contact_tp = (contact_pred * active_mask).sum()
        contact_pred_count = contact_pred.sum()

        pred_bin = pred_active
        gt_bin = active_mask
        intersection = (pred_bin * gt_bin).sum(dim=1)
        union = ((pred_bin + gt_bin) > 0).float().sum(dim=1)
        contact_iou = torch.where(
            union > 0,
            intersection / union.clamp_min(1.0),
            torch.ones_like(union),
        )
        contact_iou_sum = contact_iou.sum()

        if touchanything_protocol:
            valid_flags = valid.detach().cpu().tolist()
            valid_sequence_keys = [
                touchanything_protocol_group_key(key, alias)
                for key, alias, is_valid in zip(sequence_keys, query_aliases, valid_flags)
                if is_valid
            ]
            touch_pred_binary = ((pred > TOUCHANYTHING_CONTACT_THRESHOLD).float() * palm)
            touch_gt_binary = ((gt > TOUCHANYTHING_CONTACT_THRESHOLD).float() * palm)
            touch_frame_stats = torch.stack(
                (
                    diff.abs().sum(dim=1),
                    palm.sum(dim=1),
                    (torch.minimum(pred, gt) * palm).sum(dim=1),
                    (torch.maximum(pred, gt) * palm).sum(dim=1),
                    (touch_pred_binary * touch_gt_binary).sum(dim=1),
                    ((touch_pred_binary + touch_gt_binary) > 0).float().sum(dim=1),
                    touch_pred_binary.sum(dim=1),
                    touch_gt_binary.sum(dim=1),
                ),
                dim=1,
            )
            self._val_touchanything_protocol_stats.add(
                valid_sequence_keys,
                [
                    int(index)
                    for index, is_valid in zip(frame_indices.detach().cpu().tolist(), valid_flags)
                    if is_valid
                ],
                touch_frame_stats.detach().cpu().numpy(),
            )

        vol_intersection = (torch.minimum(pred, gt) * palm).sum(dim=1)
        vol_union = (torch.maximum(pred, gt) * palm).sum(dim=1)
        volumetric_iou = torch.where(
            vol_union > 1e-6,
            vol_intersection / vol_union.clamp_min(1e-6),
            torch.ones_like(vol_union),
        )
        volumetric_iou_sum = volumetric_iou.sum()
        pred_volume_frame = (pred * palm).sum(dim=1)
        gt_volume_frame = (gt * palm).sum(dim=1)
        location_eligible = gt_volume_frame >= 1.0
        pred_dist = (pred * palm) / pred_volume_frame.unsqueeze(1).clamp_min(1e-12)
        gt_dist = (gt * palm) / gt_volume_frame.unsqueeze(1).clamp_min(1e-12)
        distribution_intersection = torch.minimum(pred_dist, gt_dist).sum(dim=1)
        distribution_union = torch.maximum(pred_dist, gt_dist).sum(dim=1)
        distribution_viou = torch.where(
            distribution_union > 1e-12,
            distribution_intersection / distribution_union.clamp_min(1e-12),
            torch.zeros_like(distribution_union),
        )
        gt_location_support = ((gt >= 0.05).float() * palm)
        pred_location_support = ((pred >= 0.05).float() * palm)
        pred_mass_on_gt_support = (
            pred * gt_location_support
        ).sum(dim=1) / pred_volume_frame.clamp_min(1e-12)
        gt_mass_in_pred_support = (
            gt * pred_location_support
        ).sum(dim=1) / gt_volume_frame.clamp_min(1e-12)
        core_location_eligible = location_eligible & (
            (gt * palm).amax(dim=1) >= CORE_LOCATION_MIN_GT_PEAK
        )
        pred_core_mass = pred.clamp_min(0.0).pow(CORE_LOCATION_DISTRIBUTION_POWER) * palm
        gt_core_mass = gt.clamp_min(0.0).pow(CORE_LOCATION_DISTRIBUTION_POWER) * palm
        pred_core_volume = pred_core_mass.sum(dim=1)
        gt_core_volume = gt_core_mass.sum(dim=1)
        pred_core_dist = pred_core_mass / pred_core_volume.unsqueeze(1).clamp_min(1e-12)
        gt_core_dist = gt_core_mass / gt_core_volume.unsqueeze(1).clamp_min(1e-12)
        core_distribution_intersection = torch.minimum(pred_core_dist, gt_core_dist).sum(dim=1)
        core_distribution_union = torch.maximum(pred_core_dist, gt_core_dist).sum(dim=1)
        core_distribution_viou = torch.where(
            core_distribution_union > 1e-12,
            core_distribution_intersection / core_distribution_union.clamp_min(1e-12),
            torch.zeros_like(core_distribution_union),
        )
        gt_core_support = ((gt >= CORE_LOCATION_MIN_GT_PEAK).float() * palm)
        pred_core_support = ((pred >= CORE_LOCATION_MIN_GT_PEAK).float() * palm)
        core_pred_mass_on_gt_support = (
            pred_core_mass * gt_core_support
        ).sum(dim=1) / pred_core_volume.clamp_min(1e-12)
        core_gt_mass_in_pred_support = (
            gt_core_mass * pred_core_support
        ).sum(dim=1) / gt_core_volume.clamp_min(1e-12)
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
        tail_masks = [((gt >= threshold).float() * palm) for threshold in (0.2, 0.3, 0.5, 0.7)]

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
            *[
                value
                for tail_mask in tail_masks
                for value in (
                    (diff.abs() * tail_mask).sum(),
                    (pred * tail_mask).sum(),
                    tail_mask.sum(),
                )
            ],
            vol_intersection.sum(),
            vol_union.sum(),
            (distribution_viou * location_eligible).sum(),
            (pred_mass_on_gt_support * location_eligible).sum(),
            (gt_mass_in_pred_support * location_eligible).sum(),
            location_eligible.sum().float(),
            (core_distribution_viou * core_location_eligible).sum(),
            (core_pred_mass_on_gt_support * core_location_eligible).sum(),
            (core_gt_mass_in_pred_support * core_location_eligible).sum(),
            core_location_eligible.sum().float(),
        ]).to(pred_tactile.device)

        if accumulator_name in (None, "__overall__"):
            current = self._val_eval_stats
            self._val_eval_stats = stats if current is None else current + stats
        else:
            current = self._val_eval_stats_by_domain.get(accumulator_name)
            self._val_eval_stats_by_domain[accumulator_name] = (
                stats if current is None else current + stats
            )

    def _metrics_from_eval_stats(self, stats, prefix, touch_summary=None):
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
        contact_precision = stats[23] / stats[24].clamp_min(1.0)
        contact_recall = stats[23] / active_gt_count
        contact_f1 = (
            2.0 * contact_precision * contact_recall
            / (contact_precision + contact_recall).clamp_min(1e-6)
        )
        corr_num = frames * stats[35] - stats[5] * stats[4]
        corr_gt_var = (frames * stats[33] - stats[5].pow(2)).clamp_min(0.0)
        corr_pred_var = (frames * stats[34] - stats[4].pow(2)).clamp_min(0.0)
        frame_volume_corr = torch.clamp(
            corr_num / torch.sqrt(corr_gt_var * corr_pred_var).clamp_min(1e-6),
            min=-1.0,
            max=1.0,
        )
        low_volume_count = stats[39].clamp_min(1.0)
        high_volume_count = stats[43].clamp_min(1.0)
        empty_frame_count = stats[46].clamp_min(1.0)
        volumetric_iou_frame_macro = stats[14] / frames
        volumetric_iou_split_micro = torch.where(
            stats[79] > 1e-6,
            stats[78] / stats[79].clamp_min(1e-6),
            stats.new_tensor(1.0),
        )
        location_frame_count = stats[83].clamp_min(1.0)
        core_location_frame_count = stats[87].clamp_min(1.0)
        metrics = {
            f"{prefix}/eval_mae": stats[2] / values,
            f"{prefix}/eval_rmse": torch.sqrt(stats[3] / values),
            f"{prefix}/eval_pred_volume_mean": stats[4] / frames,
            f"{prefix}/eval_gt_volume_mean": stats[5] / frames,
            f"{prefix}/eval_pred_gt_volume_ratio": stats[4] / gt_volume,
            f"{prefix}/eval_frame_volume_corr": frame_volume_corr,
            f"{prefix}/eval_low_volume_pred_gt_ratio": stats[36] / stats[37].clamp_min(1e-6),
            f"{prefix}/eval_high_volume_pred_gt_ratio": stats[40] / stats[41].clamp_min(1e-6),
            f"{prefix}/eval_low_volume_pred_volume": stats[36] / low_volume_count,
            f"{prefix}/eval_high_volume_pred_volume": stats[40] / high_volume_count,
            f"{prefix}/eval_low_volume_volumetric_iou": stats[38] / low_volume_count,
            f"{prefix}/eval_high_volume_volumetric_iou": stats[42] / high_volume_count,
            f"{prefix}/eval_empty_frame_pred_volume": stats[44] / empty_frame_count,
            f"{prefix}/eval_empty_frame_pred_active_vertices": stats[45] / empty_frame_count,
            f"{prefix}/eval_weak_zone_pred_active_rate": stats[47] / weak_pressure_count,
            f"{prefix}/eval_active_mae": stats[6] / active_count,
            f"{prefix}/eval_background_mae": stats[8] / background_count,
            f"{prefix}/eval_active_recall": stats[10] / active_gt_count,
            f"{prefix}/eval_bg_false_positive": stats[12] / background_count,
            f"{prefix}/eval_contact_iou_active_thr": stats[13] / frames,
            f"{prefix}/eval_volumetric_iou": volumetric_iou_frame_macro,
            f"{prefix}/eval_volumetric_iou_frame_macro": volumetric_iou_frame_macro,
            f"{prefix}/eval_volumetric_iou_split_micro": volumetric_iou_split_micro,
            f"{prefix}/eval_distribution_viou": stats[80] / location_frame_count,
            f"{prefix}/eval_volume_matched_viou": stats[80] / location_frame_count,
            f"{prefix}/eval_pred_mass_on_gt_support": stats[81] / location_frame_count,
            f"{prefix}/eval_gt_mass_in_pred_support": stats[82] / location_frame_count,
            f"{prefix}/eval_nonempty_location_frame_count": stats[83],
            f"{prefix}/eval_core_distribution_viou": stats[84] / core_location_frame_count,
            f"{prefix}/eval_core_pred_mass_on_gt_support": stats[85] / core_location_frame_count,
            f"{prefix}/eval_core_gt_mass_in_pred_support": stats[86] / core_location_frame_count,
            f"{prefix}/eval_core_location_frame_count": stats[87],
            f"{prefix}/eval_contact_precision": contact_precision,
            f"{prefix}/eval_contact_recall": contact_recall,
            f"{prefix}/eval_contact_f1": contact_f1,
            f"{prefix}/eval_low_pressure_mean_pred": stats[15] / low_pressure_count,
            f"{prefix}/eval_weak_pressure_mean_pred": stats[17] / weak_pressure_count,
            f"{prefix}/eval_mid_pressure_mean_pred": stats[19] / mid_pressure_count,
            f"{prefix}/eval_high_pressure_mean_pred": stats[21] / high_pressure_count,
            f"{prefix}/eval_low_pressure_pred_active_rate": stats[53] / low_pressure_count,
            f"{prefix}/eval_catastrophic_over_rate": stats[54] / stats[55].clamp_min(1.0),
            f"{prefix}/eval_catastrophic_under_rate": stats[56] / stats[57].clamp_min(1.0),
            f"{prefix}/eval_false_high_gt005_pred03_rate": stats[58] / stats[59].clamp_min(1.0),
            f"{prefix}/eval_false_high_gt005_pred03_excess_volume_fraction": stats[60] / stats[4].clamp_min(1e-6),
            f"{prefix}/eval_false_high_gt005_pred05_rate": stats[61] / stats[59].clamp_min(1.0),
            f"{prefix}/eval_false_high_gt05_pred03_rate": stats[63] / stats[64].clamp_min(1.0),
            f"{prefix}/eval_false_high_gt05_pred03_excess_volume_fraction": stats[65] / stats[4].clamp_min(1e-6),
        }
        for offset, label in enumerate(("02", "03", "05", "07")):
            base_index = 66 + offset * 3
            tail_count = stats[base_index + 2].clamp_min(1.0)
            metrics[f"{prefix}/eval_tail_mae_gt{label}"] = stats[base_index] / tail_count
            metrics[f"{prefix}/eval_tail_mean_pred_gt{label}"] = stats[base_index + 1] / tail_count
        if touch_summary is not None:
            metrics.update({
                f"{prefix}/touchanything_protocol_volumetric_iou": stats.new_tensor(
                    touch_summary["volumetric_iou"]
                ),
                f"{prefix}/touchanything_protocol_mae": stats.new_tensor(touch_summary["mae"]),
                f"{prefix}/touchanything_protocol_contact_iou": stats.new_tensor(
                    touch_summary["contact_iou"]
                ),
                f"{prefix}/touchanything_protocol_temporal_accuracy": stats.new_tensor(
                    touch_summary["temporal_accuracy"]
                ),
                f"{prefix}/touchanything_protocol_temporal_f1": stats.new_tensor(
                    touch_summary["temporal_f1"]
                ),
                f"{prefix}/touchanything_protocol_sequence_count": stats.new_tensor(
                    float(touch_summary["sequence_count"])
                ),
            })
        return metrics

    def on_validation_epoch_end(self):
        if self._val_eval_stats is None:
            return
        stats = self._val_eval_stats
        if self.trainer.world_size > 1:
            gathered = self.all_gather(stats)
            stats = gathered.reshape(-1, stats.numel()).sum(dim=0)

        local_touch_stats = self._val_touchanything_protocol_stats.pack()
        if self.trainer.world_size > 1:
            touch_items = (
                [None for _ in range(self.trainer.world_size)]
                if self.global_rank == 0
                else None
            )
            torch.distributed.gather_object(local_touch_stats, touch_items, dst=0)
            summary_box = [
                summarize_compact_touchanything_protocol(touch_items, include_rows=False)
                if self.global_rank == 0
                else None
            ]
            torch.distributed.broadcast_object_list(summary_box, src=0)
            touch_summary = summary_box[0]
        else:
            touch_summary = summarize_compact_touchanything_protocol(
                [local_touch_stats], include_rows=False
            )
        self._val_touchanything_protocol_stats = CompactTouchAnythingProtocolAccumulator()
        del local_touch_stats
        if self.global_rank == 0 and self.trainer.world_size > 1:
            del touch_items
        release_unused_python_heap()
        eval_metrics = self._metrics_from_eval_stats(stats, "val", touch_summary=touch_summary)
        domain_metrics = {}
        domain_core_metrics = []
        for domain_name in ("opentouch", "touchanything"):
            domain_stats = self._val_eval_stats_by_domain.get(domain_name)
            if domain_stats is None:
                domain_stats = torch.zeros_like(stats)
            if self.trainer.world_size > 1:
                gathered = self.all_gather(domain_stats)
                domain_stats = gathered.reshape(-1, domain_stats.numel()).sum(dim=0)
            if float(domain_stats[0].item()) <= 0.0:
                continue
            metrics = self._metrics_from_eval_stats(
                domain_stats,
                f"val/{domain_name}",
                touch_summary=touch_summary if domain_name == "touchanything" else None,
            )
            domain_metrics.update(metrics)
            domain_core_metrics.append((
                metrics[f"val/{domain_name}/eval_rmse"],
                metrics[f"val/{domain_name}/eval_volumetric_iou"],
                metrics[f"val/{domain_name}/eval_core_distribution_viou"],
            ))
        if domain_core_metrics:
            eval_metrics["val/domain_macro/eval_rmse"] = torch.stack(
                [item[0] for item in domain_core_metrics]
            ).mean()
            eval_metrics["val/domain_macro/eval_volumetric_iou"] = torch.stack(
                [item[1] for item in domain_core_metrics]
            ).mean()
            eval_metrics["val/domain_macro/eval_core_distribution_viou"] = torch.stack(
                [item[2] for item in domain_core_metrics]
            ).mean()
        eval_metrics.update(domain_metrics)

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

    def _log_tactile_loss_breakdown(
        self,
        prefix,
        output,
        on_step,
        batch_size,
    ):
        mapping = {
            "loss_smooth_l1_raw": "loss/smooth_l1_raw",
            "loss_logit_bce_raw": "loss/logit_bce_raw",
            "loss_base_tactile": "loss/base_tactile",
            "loss_weighted_tactile": "loss/weighted_tactile",
            "loss_background": "loss/background",
            "loss_location_raw": "loss/location_raw",
            "loss_location_weighted": "loss/location_weighted",
            "loss_contact_raw": "loss/contact_raw",
            "loss_contact_weighted": "loss/contact_weighted",
            "diagnostics_location_eligible_fraction": "diagnostics/location_eligible_fraction",
            "diagnostics_pressure_weight_mean": "pressure_weight_mean",
            "diagnostics_pressure_weight_max": "pressure_weight_max",
            "diagnostics_pressure_weight_fraction_gt2": "pressure_weight_fraction_gt2",
            "diagnostics_weighted_to_direct_loss_ratio": "weighted_to_direct_loss_ratio",
            "loss_tactile": "loss/total",
            "loss_ramp": "schedule/loss_ramp",
            "loss_full_ramp_reference": "loss_full_ramp_reference",
        }
        if prefix == "val":
            mapping.update({
                "loss_current_ramp": "loss_current_ramp",
                "loss_direct_raw": "loss_direct_raw",
            })
        for key, name in mapping.items():
            if key in output["losses"]:
                self.log(
                    f"{prefix}/{name}",
                    output["losses"][key],
                    on_step=on_step,
                    on_epoch=True,
                    logger=True,
                    sync_dist=self.sync_train_logs if prefix == "train" else True,
                    batch_size=int(batch_size),
                )

    def configure_optimizers(self):
        optim_groups = optimizer_parameter_groups(
            self.named_parameters(),
            self.optimizer_weight_decay,
        )

        gradient_clip_val = float(
            getattr(self.trainer, "gradient_clip_val", 0.0) or 0.0
        )
        optimizer_kwargs = {
            "lr": self.learning_rate,
        }
        optimizer_backend = "default"
        if torch.cuda.is_available() and gradient_clip_val <= 0.0:
            optimizer_kwargs["fused"] = True
            optimizer_backend = "fused"
        elif torch.cuda.is_available():
            # Lightning 2.1 cannot externally unscale and clip gradients for
            # fused AdamW because that optimizer performs AMP unscaling
            # internally. Foreach keeps the batched CUDA update without that
            # conflict and preserves the requested clipping semantics.
            optimizer_kwargs["foreach"] = True
            optimizer_backend = "foreach"
        try:
            optimizer = torch.optim.AdamW(optim_groups, **optimizer_kwargs)
        except (TypeError, RuntimeError) as exc:
            accelerated_flag = (
                optimizer_kwargs.pop("fused", False)
                or optimizer_kwargs.pop("foreach", False)
            )
            if not accelerated_flag:
                raise
            print(
                f"{optimizer_backend} AdamW unavailable; "
                f"falling back to default AdamW: {exc}",
                flush=True,
            )
            optimizer_backend = "default"
            optimizer = torch.optim.AdamW(optim_groups, **optimizer_kwargs)
        self.optimizer_backend = optimizer_backend
        print(
            "AdamW backend: "
            f"{optimizer_backend} (gradient_clip_val={gradient_clip_val:g})",
            flush=True,
        )

        total_steps = self.trainer.estimated_stepping_batches
        print(f"Total training steps for LR Scheduler: {total_steps}")
        
        max_epochs = max(int(getattr(self.trainer, "max_epochs", 1)), 1)
        warmup_steps = lr_warmup_step_count(
            total_steps,
            max_epochs,
            self.lr_warmup_epochs,
        )
        warmup = None
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / float(max(warmup_steps, 1)),
                end_factor=1.0,
                total_iters=warmup_steps,
            )

        if self.lr_scheduler_name == "cosine":
            self.lr_decay_milestones_resolved = []
            decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_steps - warmup_steps, 1) if warmup_steps > 0 else total_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif self.lr_scheduler_name == "constant":
            self.lr_decay_milestones_resolved = []
            decay_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda _step: 1.0,
            )
        elif self.lr_scheduler_name == "multistep":
            milestones = resolve_lr_decay_milestones(
                self.lr_decay_milestones,
                total_steps,
            )
            self.lr_decay_milestones_resolved = list(milestones)
            scheduler_milestones = [step - warmup_steps for step in milestones]
            if any(step <= 0 for step in scheduler_milestones):
                raise ValueError(
                    "Resolved --lr_decay_milestones must occur after LR warmup; "
                    f"warmup ends at step {warmup_steps}, milestones={milestones}"
                )
            decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_milestones,
                gamma=self.lr_decay_gamma,
            )
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler_name}")

        if warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, decay_scheduler],
                milestones=[warmup_steps],
            )
            print(
                f"LR schedule: linear warmup for {warmup_steps} steps "
                f"({self.lr_warmup_epochs} epoch(s)), then {self.lr_scheduler_name} "
                f"for {total_steps - warmup_steps} steps"
            )
        else:
            scheduler = decay_scheduler
            print(f"LR schedule: {self.lr_scheduler_name} for {total_steps} steps")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def _load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path, map_location="cpu")


def _migrate_tactile_head_state_keys(head_state, expected_state):
    """Apply narrowly scoped parameter renames while preserving strict loading."""
    migrated = dict(head_state)
    renamed = []
    prefix_aliases = (
        ("decoder.0.project.", "decoder.0.projection."),
    )
    for old_prefix, new_prefix in prefix_aliases:
        for old_key in tuple(migrated):
            if not old_key.startswith(old_prefix):
                continue
            new_key = new_prefix + old_key[len(old_prefix):]
            if new_key not in expected_state:
                continue
            if new_key in migrated:
                raise RuntimeError(
                    f"Checkpoint contains both legacy and current tactile-head keys: "
                    f"{old_key!r}, {new_key!r}"
                )
            old_shape = tuple(migrated[old_key].shape)
            expected_shape = tuple(expected_state[new_key].shape)
            if old_shape != expected_shape:
                raise RuntimeError(
                    f"Cannot migrate tactile-head key {old_key!r} to {new_key!r}: "
                    f"checkpoint shape={old_shape}, expected shape={expected_shape}"
                )
            migrated[new_key] = migrated.pop(old_key)
            renamed.append((old_key, new_key))
    return migrated, renamed


def load_compatible_state_dict(model, checkpoint_path, load_backbone=False):
    checkpoint = _load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "tactile_trainable_v2":
        raise ValueError("Only compact format=tactile_trainable_v2 checkpoints are supported")
    if checkpoint.get("visual_backbone") != "dinov3_hplus":
        raise ValueError("Checkpoint must use visual_backbone=dinov3_hplus")
    checkpoint_head_type = checkpoint.get("tactile_head_type")
    if checkpoint_head_type not in {"dense_v2", "dense_v2_dino_rezero"}:
        raise ValueError("Checkpoint uses an unsupported tactile head type")
    if checkpoint_head_type != getattr(model, "tactile_head_type", None):
        raise ValueError(
            f"Tactile head mismatch: checkpoint={checkpoint_head_type}, "
            f"model={getattr(model, 'tactile_head_type', None)}"
        )
    checkpoint_resolution = parse_input_resolution(
        checkpoint.get("input_resolution", (256, 192))
    )
    if checkpoint_resolution != tuple(getattr(model, "input_resolution", (256, 192))):
        raise ValueError(
            f"Input resolution mismatch: checkpoint={checkpoint_resolution}, "
            f"model={getattr(model, 'input_resolution', None)}"
        )
    expected_hash = str(checkpoint.get("backbone_sha256", "") or "")
    actual_hash = str(getattr(model, "backbone_weights_sha256", "") or "")
    if expected_hash and actual_hash and expected_hash != actual_hash:
        raise ValueError(
            f"Backbone SHA256 mismatch: checkpoint={expected_hash}, actual={actual_hash}"
        )
    prefix = "tactile_head."
    head_state = {
        key[len(prefix):]: value
        for key, value in checkpoint.get("state_dict", {}).items()
        if key.startswith(prefix)
    }
    if not head_state:
        raise ValueError("Compact checkpoint does not contain tactile_head parameters")
    head_state, renamed = _migrate_tactile_head_state_keys(
        head_state,
        model.tactile_head.state_dict(),
    )
    if renamed:
        print(
            "Migrated legacy tactile-head checkpoint keys: "
            + ", ".join(f"{old} -> {new}" for old, new in renamed)
        )
    model.tactile_head.load_state_dict(head_state, strict=True)
    print(
        f"Loaded compact tactile checkpoint: {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch')}, global_step={checkpoint.get('global_step')})"
    )
    return [], []


def parse_args():
    parser = argparse.ArgumentParser(description="Train the frozen-DINO tactile regressor")
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
    parser.add_argument(
        "--optimizer_weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay for non-bias/non-normalization parameters.",
    )
    parser.add_argument(
        "--lr_scheduler",
        choices=("cosine", "constant", "multistep"),
        default="cosine",
    )
    parser.add_argument(
        "--lr_decay_milestones",
        type=str,
        default="0.5,0.75",
        help="Comma-separated LR decay points: values below 1 are total-step fractions; integers are steps.",
    )
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--val_num_workers", type=int, default=None, help="Workers for validation DataLoader; defaults to --num_workers")
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep DataLoader workers alive between epochs; disabled by default to bound worker memory growth.",
    )
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor when num_workers > 0")
    parser.add_argument(
        "--data_backend",
        choices=("auto", "legacy_dirs", "sequence_hdf5"),
        default="auto",
        help=(
            "Dataset storage backend. auto prefers <processed_root>/manifests/"
            "<split>.queries.jsonl when present and otherwise keeps the legacy directory reader."
        ),
    )
    parser.add_argument(
        "--query_manifests",
        type=str,
        default="",
        help=(
            "Optional comma-separated sequence-HDF5 query manifests. Each manifest row "
            "represents one frame/hand query and replaces filesystem index discovery."
        ),
    )
    parser.add_argument(
        "--hdf5_handle_cache_size",
        type=int,
        default=4,
        help="Maximum read-only sequence HDF5 handles retained by each DataLoader worker.",
    )
    parser.add_argument(
        "--hdf5_manifest_cache_dir",
        type=str,
        default=os.path.join(ft_dir, "hdf5_manifest_cache"),
        help=(
            "Shared mmap cache for normalized HDF5 query-manifest rows. This is "
            "separate from the legacy directory index cache."
        ),
    )
    parser.add_argument("--index_workers", type=int, default=1, help="Workers for initial meta.json index scanning")
    parser.add_argument("--index_backend", type=str, default="process", choices=["process", "thread"], help="Parallel backend for initial index scanning")
    parser.add_argument("--index_chunksize", type=int, default=256, help="Chunk size for parallel index scanning")
    parser.add_argument(
        "--index_process_worker_cap",
        type=int,
        default=64,
        help="Maximum process workers used for shared-filesystem index scans; 0 disables the cap.",
    )
    parser.add_argument(
        "--index_manifest",
        type=str,
        default=os.path.join(
            ft_dir,
            "data_integrity_audits",
            "mixed_v2_input",
            "data_integrity_samples.csv",
        ),
        help="Verified data-integrity CSV used to build compact indexes without rescanning sample dirs.",
    )
    parser.add_argument(
        "--expected_datasets",
        type=str,
        default="",
        help=(
            "Hard dataset-content contract for strict/single-domain runs. The dataset constructor "
            "fails if an unexpected domain is present or an expected domain is absent."
        ),
    )
    parser.add_argument("--index_cache_dir", type=str, default=os.path.join(ft_dir, "index_cache"), help="Shared JSONL cache for scanned dataset indices")
    parser.add_argument(
        "--bbox_source_policy",
        choices=("any", "sam3_only"),
        default="sam3_only",
        help=(
            "sam3_only admits only per-query bboxes carrying schema=sam3_bbox_source_v1; "
            "legacy boxes are never used as fallback."
        ),
    )
    parser.add_argument(
        "--bbox_manifests",
        type=str,
        default="",
        help=(
            "Comma-separated reviewed SAM3 bbox JSONL manifests. When set, index "
            "construction reads candidate sample_dir values from these manifests and "
            "skips expensive flat-directory discovery."
        ),
    )
    parser.add_argument(
        "--lazy_index_records",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Memory-map the compact JSONL index instead of retaining millions of "
            "Python dictionaries in every DataLoader worker. Intended for frame-uniform "
            "training on very large datasets."
        ),
    )
    parser.add_argument(
        "--rebuild_index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force rank 0 to rebuild the index cache; disabled by default.",
    )
    parser.add_argument("--index_cache_timeout", type=int, default=3600, help="Seconds nonzero ranks wait for index cache")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb_mode",
        choices=("async", "online", "offline", "disabled"),
        default="async",
        help=(
            "async/online uploads durable epoch snapshots from an isolated "
            "process; offline queues snapshots without uploading; CSV logging "
            "is always enabled."
        ),
    )
    parser.add_argument(
        "--wandb_sync_on_finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch one final asynchronous upload attempt when fitting ends.",
    )
    parser.add_argument("--wandb_sync_retries", type=int, default=24)
    parser.add_argument("--wandb_sync_interval", type=int, default=300)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Exact tactile_resume_v1 checkpoint to restore.",
    )
    parser.add_argument(
        "--auto_resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume checkpoints/<exp_name>/resume.ckpt when it exists.",
    )
    parser.add_argument(
        "--resume_save_every_n_epochs",
        type=int,
        default=1,
        help="Frequency for atomic optimizer/scheduler/loop resume checkpoints.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="mixed_dense_v2_dinov3_rezero_fullgrid32",
        help="Experiment name",
    )
    parser.add_argument("--quick_test", action="store_true", help="Run a quick test training")
    parser.add_argument(
        "--tactile_head_type",
        choices=("dense_v2", "dense_v2_dino_rezero"),
        default="dense_v2_dino_rezero",
    )
    parser.add_argument(
        "--backbone_feature_layers",
        type=str,
        default="8,16,24,32",
        help="Comma-separated 1-based DINO block indices used by ReZero fusion.",
    )
    parser.add_argument("--dino_residual_max_scale", type=float, default=0.10)
    parser.add_argument("--dino_residual_rms_budget", type=float, default=0.50)
    parser.add_argument(
        "--pool_layout",
        choices=("legacy5", "fullgrid32"),
        default="fullgrid32",
        help="Use original V2 5x5/21-cell pooling or preserve the full DINO grid.",
    )
    parser.add_argument(
        "--input_resolution",
        type=str,
        default="256x192",
        help="DINO input in HEIGHTxWIDTH form: 256x192, 320x240, or 384x288.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches accumulated before each optimizer step.",
    )
    parser.add_argument("--pool_output_channels", type=int, default=32)
    parser.add_argument(
        "--decoder_dropout_scale",
        type=float,
        default=1.0,
        help="Multiply all three Dense V2 decoder dropout probabilities by this value.",
    )
    parser.add_argument(
        "--train_augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bbox scale/translation augmentation for training samples.",
    )
    parser.add_argument(
        "--bbox_rescale_factor",
        type=float,
        default=2.0,
        help=(
            "Crop height relative to the query bbox longest side; the crop width is "
            "75% of its height. Must lie in [1.0, 4.0]."
        ),
    )
    parser.add_argument(
        "--visual_backbone",
        choices=("dinov3_hplus",),
        default="dinov3_hplus",
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
    parser.add_argument(
        "--save_contact_best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the optional TouchAnything contact-best compact checkpoint.",
    )
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
    parser.add_argument(
        "--pressure_weight_mode",
        choices=("hump", "plateau", "capped_linear"),
        default="hump",
    )
    parser.add_argument("--active_pressure_tail_thr", type=float, default=0.70)
    parser.add_argument("--active_pressure_tail_max", type=float, default=3.0)
    parser.add_argument("--background_loss_weight", type=float, default=1.0)
    parser.add_argument("--logit_bce_weight", type=float, default=0.1)
    parser.add_argument("--loss_ramp_epochs", type=int, default=5)
    parser.add_argument("--frame_low_volume_thr", type=float, default=30.0, help="GT frame volume threshold for low-volume validation diagnostics")
    parser.add_argument("--frame_high_volume_thr", type=float, default=150.0, help="GT frame volume threshold for high-volume validation diagnostics")
    parser.add_argument("--opentouch_high_pressure_thr", type=float, default=0.9)
    parser.add_argument("--opentouch_high_pressure_weight", type=float, default=0.3)
    parser.add_argument("--location_loss_weight", type=float, default=0.0)
    parser.add_argument("--location_gt_volume_thr", type=float, default=1.0)
    parser.add_argument("--location_distribution_power", type=float, default=1.0)
    parser.add_argument("--location_min_gt_peak", type=float, default=0.0)
    parser.add_argument(
        "--contact_loss_type",
        choices=("none", "soft_jaccard", "lovasz"),
        default="none",
    )
    parser.add_argument("--contact_loss_weight", type=float, default=0.0)
    parser.add_argument("--contact_pressure_thr", type=float, default=0.1)
    parser.add_argument("--contact_temperature", type=float, default=0.025)
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
        pressure_weight_mode=args.pressure_weight_mode,
        active_pressure_tail_thr=args.active_pressure_tail_thr,
        active_pressure_tail_max=args.active_pressure_tail_max,
        background_loss_weight=args.background_loss_weight,
        logit_bce_weight=args.logit_bce_weight,
        loss_ramp_epochs=args.loss_ramp_epochs,
        opentouch_high_pressure_thr=args.opentouch_high_pressure_thr,
        opentouch_high_pressure_weight=args.opentouch_high_pressure_weight,
        location_loss_weight=args.location_loss_weight,
        location_gt_volume_thr=args.location_gt_volume_thr,
        location_distribution_power=args.location_distribution_power,
        location_min_gt_peak=args.location_min_gt_peak,
        contact_loss_type=args.contact_loss_type,
        contact_loss_weight=args.contact_loss_weight,
        contact_pressure_thr=args.contact_pressure_thr,
        contact_temperature=args.contact_temperature,
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
        "tactile_head": model.tactile_head,
    }
    lines = [
        "DINO tactile model summary",
        f"tactile_dim: {model.tactile_dim}",
        f"tactile_head_type: {model.tactile_head_type}",
        f"pool_layout: {getattr(model, 'pool_layout', None)}",
        f"pool_grid_size: {getattr(model, 'pool_grid_size', None)}",
        f"pool_valid_tokens: {getattr(model, 'pool_valid_tokens', None)}",
        f"input_resolution: {getattr(model, 'input_resolution', None)}",
        f"decoder_input_dim: {getattr(model, 'decoder_input_dim', None)}",
        f"pool_output_channels: {getattr(model, 'pool_output_channels', None)}",
        f"decoder_dropout_scale: {getattr(model, 'decoder_dropout_scale', None)}",
        f"dino_residual_max_scale: {getattr(model, 'dino_residual_max_scale', None)}",
        f"dino_residual_rms_budget: {getattr(model, 'dino_residual_rms_budget', None)}",
        f"bbox_rescale_factor: {getattr(model, 'bbox_rescale_factor', None)}",
        f"bbox_source_policy: {getattr(model, 'bbox_source_policy', None)}",
        f"frame_low_volume_thr: {getattr(model, 'frame_low_volume_thr', None)}",
        f"frame_high_volume_thr: {getattr(model, 'frame_high_volume_thr', None)}",
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
    args_config = dict(vars(args))
    checkpoint_monitors = {
        "loss-best": {"metric": "val/loss_full_ramp_reference", "mode": "min"},
    }
    requested_datasets = set(getattr(model, "dataset_filter", ()))
    if args.save_contact_best and "TouchAnything" in requested_datasets:
        checkpoint_monitors["contact-best"] = {
            "metric": "val/touchanything/touchanything_protocol_contact_iou",
            "mode": "max",
        }
    run_config = {
        "args": args_config,
        "resolved_data_dirs": data_dirs,
        "visual_backbone": model.visual_backbone,
        "backbone_weights": model.backbone_weights_path,
        "backbone_sha256": model.backbone_weights_sha256,
        "num_gpus": num_gpus,
        "base_lr": args.lr,
        "effective_lr": lr_scaled,
        "lr_scaled": lr_scaled,
        "optimizer_config": {
            "name": "AdamW",
            "backend": (
                "foreach"
                if num_gpus > 0 and float(args.gradient_clip_val) > 0.0
                else "fused_if_cuda"
            ),
            "weight_decay": float(args.optimizer_weight_decay),
            "no_decay_weight_decay": 0.0,
            "gradient_clip_val": float(args.gradient_clip_val),
        },
        "lr_scheduler_config": {
            "name": str(args.lr_scheduler),
            "warmup_epochs": int(args.lr_warmup_epochs),
            "decay_milestones": str(args.lr_decay_milestones),
            "decay_gamma": float(args.lr_decay_gamma),
        },
        "checkpoint_monitors": checkpoint_monitors,
        "metric_protocols": {
            "volumetric_iou_frame_macro": "per-frame min/max ratio, then frame mean",
            "volumetric_iou_split_micro": "min/max pressure mass over the complete validation split",
            "touchanything_canonical_query": {
                "aggregation": "source-trajectory micro across canonical queries, then unweighted trajectory mean",
                "contact_threshold": TOUCHANYTHING_CONTACT_THRESHOLD,
                "min_contact_ratio": TOUCHANYTHING_MIN_CONTACT_RATIO,
                "bend_sensor_mask": "not_applicable_to_canonical_mesh",
            },
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
        "pool_grid_size": (
            list(getattr(model, "pool_grid_size", ()))
            if isinstance(getattr(model, "pool_grid_size", 0), (tuple, list))
            else int(getattr(model, "pool_grid_size", 0))
        ),
        "pool_valid_tokens": int(getattr(model, "pool_valid_tokens", 0)),
        "input_resolution": list(getattr(model, "input_resolution", (256, 192))),
        "decoder_input_dim": int(getattr(model, "decoder_input_dim", 6144)),
        "pool_output_channels": int(getattr(model, "pool_output_channels", 32)),
        "decoder_dropout_scale": float(getattr(model, "decoder_dropout_scale", 1.0)),
        "backbone_feature_layers": list(getattr(model, "backbone_feature_layers", ())),
        "dino_residual_max_scale": float(getattr(model, "dino_residual_max_scale", 0.10)),
        "dino_residual_rms_budget": float(getattr(model, "dino_residual_rms_budget", 0.50)),
        "accumulate_grad_batches": int(args.accumulate_grad_batches),
        "index_schema_version": int(getattr(model, "index_schema_version", INDEX_CACHE_VERSION)),
        "index_cache_key": str(getattr(model, "index_cache_key", "")),
        "indexed_sample_count": int(getattr(model, "indexed_sample_count", 0)),
        "index_manifest_sha256": str(getattr(model, "index_manifest_sha256", "")),
        "data_backend": str(getattr(model, "data_backend", "legacy_dirs")),
        "query_manifest_sha256": dict(getattr(model, "query_manifest_sha256", {})),
        "hdf5_schema_version": str(getattr(model, "hdf5_schema_version", "")),
        "hdf5_handle_cache_size": int(getattr(model, "hdf5_handle_cache_size", 4)),
        "hdf5_manifest_cache_dir": str(
            getattr(model, "hdf5_manifest_cache_dir", "")
        ),
        "hdf5_manifest_cache_key": str(
            getattr(model, "hdf5_manifest_cache_key", "")
        ),
        "bbox_manifest_sha256": dict(getattr(model, "bbox_manifest_sha256", {})),
        "lazy_index_records": bool(getattr(model, "lazy_index_records", False)),
        "dataset_filter": list(getattr(model, "dataset_filter", ())),
        "train_augmentation": bool(getattr(model, "train_augmentation", True)),
        "bbox_rescale_factor": float(getattr(model, "bbox_rescale_factor", 2.0)),
        "bbox_source_policy": str(getattr(model, "bbox_source_policy", "any")),
        "location_loss_weight": float(tactile_loss_config.location_loss_weight),
        "location_gt_volume_thr": float(tactile_loss_config.location_gt_volume_thr),
        "location_distribution_power": float(tactile_loss_config.location_distribution_power),
        "location_min_gt_peak": float(tactile_loss_config.location_min_gt_peak),
        "contact_loss_type": str(tactile_loss_config.contact_loss_type),
        "contact_loss_weight": float(tactile_loss_config.contact_loss_weight),
        "contact_pressure_thr": float(tactile_loss_config.contact_pressure_thr),
        "contact_temperature": float(tactile_loss_config.contact_temperature),
        "optimizer_weight_decay": float(getattr(model, "optimizer_weight_decay", 1e-4)),
        "lr_scheduler": str(getattr(model, "lr_scheduler_name", "cosine")),
        "lr_decay_milestones": str(getattr(model, "lr_decay_milestones", "0.5,0.75")),
        "lr_decay_gamma": float(getattr(model, "lr_decay_gamma", 0.1)),
        "lr_warmup_epochs": int(getattr(model, "lr_warmup_epochs", 0)),
        "frame_low_volume_thr": float(getattr(model, "frame_low_volume_thr", 0.0)),
        "frame_high_volume_thr": float(getattr(model, "frame_high_volume_thr", 0.0)),
        "tactile_loss_scale": float(model.tactile_loss_scale),
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


def make_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    persistent_workers,
    prefetch_factor,
    sampler=None,
):
    num_workers = int(num_workers)
    kwargs = {
        "batch_size": batch_size,
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
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
        "visual_backbone": str(getattr(module, "visual_backbone", "dinov3_hplus")),
        "visual_backbone_model_name": str(getattr(module, "visual_backbone_model_name", "")),
        "backbone_weights": str(getattr(module, "backbone_weights_path", "") or ""),
        "backbone_sha256": str(getattr(module, "backbone_weights_sha256", "") or ""),
        "tactile_head_type": str(getattr(module, "tactile_head_type", "")),
        "backbone_feature_layers": list(getattr(module, "backbone_feature_layers", ())),
        "dino_residual_max_scale": float(getattr(module, "dino_residual_max_scale", 0.10)),
        "dino_residual_rms_budget": float(getattr(module, "dino_residual_rms_budget", 0.50)),
        "pool_layout": str(getattr(module, "pool_layout", "fullgrid32")),
        "input_resolution": list(getattr(module, "input_resolution", (256, 192))),
        "pool_grid_size": list(getattr(module, "pool_grid_size", (16, 12))),
        "pool_valid_tokens": int(getattr(module, "pool_valid_tokens", 192)),
        "decoder_input_dim": int(getattr(module, "decoder_input_dim", 6144)),
        "pool_output_channels": int(getattr(module, "pool_output_channels", 32)),
        "decoder_dropout_scale": float(getattr(module, "decoder_dropout_scale", 1.0)),
        "accumulate_grad_batches": int(getattr(module, "accumulate_grad_batches_config", 1)),
        "index_schema_version": int(getattr(module, "index_schema_version", INDEX_CACHE_VERSION)),
        "index_cache_key": str(getattr(module, "index_cache_key", "")),
        "indexed_sample_count": int(getattr(module, "indexed_sample_count", 0)),
        "index_manifest_sha256": str(getattr(module, "index_manifest_sha256", "")),
        "data_backend": str(getattr(module, "data_backend", "legacy_dirs")),
        "query_manifest_sha256": dict(getattr(module, "query_manifest_sha256", {})),
        "hdf5_schema_version": str(getattr(module, "hdf5_schema_version", "")),
        "hdf5_handle_cache_size": int(getattr(module, "hdf5_handle_cache_size", 4)),
        "hdf5_manifest_cache_dir": str(
            getattr(module, "hdf5_manifest_cache_dir", "")
        ),
        "hdf5_manifest_cache_key": str(
            getattr(module, "hdf5_manifest_cache_key", "")
        ),
        "bbox_manifest_sha256": dict(getattr(module, "bbox_manifest_sha256", {})),
        "lazy_index_records": bool(getattr(module, "lazy_index_records", False)),
        "dataset_filter": list(getattr(module, "dataset_filter", ())),
        "train_augmentation": bool(getattr(module, "train_augmentation", True)),
        "bbox_rescale_factor": float(getattr(module, "bbox_rescale_factor", 2.0)),
        "bbox_source_policy": str(getattr(module, "bbox_source_policy", "any")),
        "pressure_weight_mode": str(
            module.tactile_loss_config.pressure_weight_mode
        ),
        "active_pressure_tail_thr": float(
            module.tactile_loss_config.active_pressure_tail_thr
        ),
        "active_pressure_tail_max": float(
            module.tactile_loss_config.active_pressure_tail_max
        ),
        "location_loss_weight": float(module.tactile_loss_config.location_loss_weight),
        "location_gt_volume_thr": float(module.tactile_loss_config.location_gt_volume_thr),
        "location_distribution_power": float(
            module.tactile_loss_config.location_distribution_power
        ),
        "location_min_gt_peak": float(module.tactile_loss_config.location_min_gt_peak),
        "contact_loss_type": str(module.tactile_loss_config.contact_loss_type),
        "contact_loss_weight": float(module.tactile_loss_config.contact_loss_weight),
        "contact_pressure_thr": float(module.tactile_loss_config.contact_pressure_thr),
        "contact_temperature": float(module.tactile_loss_config.contact_temperature),
        "optimizer_weight_decay": float(getattr(module, "optimizer_weight_decay", 1e-4)),
        "lr_scheduler": str(getattr(module, "lr_scheduler_name", "cosine")),
        "lr_decay_milestones": str(getattr(module, "lr_decay_milestones", "0.5,0.75")),
        "lr_decay_milestones_resolved": list(
            getattr(module, "lr_decay_milestones_resolved", ())
        ),
        "lr_decay_gamma": float(getattr(module, "lr_decay_gamma", 0.1)),
        "lr_warmup_epochs": int(getattr(module, "lr_warmup_epochs", 0)),
        "loss_config": asdict(module.tactile_loss_config),
        "model_config": {
            "tactile_head_type": str(getattr(module, "tactile_head_type", "")),
            "visual_backbone": str(getattr(module, "visual_backbone", "dinov3_hplus")),
            "visual_backbone_model_name": str(getattr(module, "visual_backbone_model_name", "")),
            "backbone_feature_layers": list(getattr(module, "backbone_feature_layers", ())),
            "dino_residual_max_scale": float(getattr(module, "dino_residual_max_scale", 0.10)),
            "dino_residual_rms_budget": float(getattr(module, "dino_residual_rms_budget", 0.50)),
            "pool_layout": str(getattr(module, "pool_layout", "fullgrid32")),
            "input_resolution": list(getattr(module, "input_resolution", (256, 192))),
            "pool_grid_size": list(getattr(module, "pool_grid_size", (16, 12))),
            "pool_valid_tokens": int(getattr(module, "pool_valid_tokens", 192)),
            "decoder_input_dim": int(getattr(module, "decoder_input_dim", 6144)),
            "pool_output_channels": int(getattr(module, "pool_output_channels", 32)),
            "decoder_dropout_scale": float(getattr(module, "decoder_dropout_scale", 1.0)),
            "accumulate_grad_batches": int(
                getattr(module, "accumulate_grad_batches_config", 1)
            ),
            "index_schema_version": int(getattr(module, "index_schema_version", INDEX_CACHE_VERSION)),
            "index_cache_key": str(getattr(module, "index_cache_key", "")),
            "indexed_sample_count": int(getattr(module, "indexed_sample_count", 0)),
            "index_manifest_sha256": str(getattr(module, "index_manifest_sha256", "")),
            "data_backend": str(getattr(module, "data_backend", "legacy_dirs")),
            "query_manifest_sha256": dict(getattr(module, "query_manifest_sha256", {})),
            "hdf5_schema_version": str(getattr(module, "hdf5_schema_version", "")),
            "hdf5_handle_cache_size": int(getattr(module, "hdf5_handle_cache_size", 4)),
            "hdf5_manifest_cache_dir": str(
                getattr(module, "hdf5_manifest_cache_dir", "")
            ),
            "hdf5_manifest_cache_key": str(
                getattr(module, "hdf5_manifest_cache_key", "")
            ),
            "bbox_manifest_sha256": dict(getattr(module, "bbox_manifest_sha256", {})),
            "lazy_index_records": bool(getattr(module, "lazy_index_records", False)),
            "dataset_filter": list(getattr(module, "dataset_filter", ())),
            "train_augmentation": bool(getattr(module, "train_augmentation", True)),
            "bbox_rescale_factor": float(getattr(module, "bbox_rescale_factor", 2.0)),
            "bbox_source_policy": str(getattr(module, "bbox_source_policy", "any")),
            "location_loss_weight": float(module.tactile_loss_config.location_loss_weight),
            "location_gt_volume_thr": float(module.tactile_loss_config.location_gt_volume_thr),
            "location_distribution_power": float(
                module.tactile_loss_config.location_distribution_power
            ),
            "location_min_gt_peak": float(module.tactile_loss_config.location_min_gt_peak),
            "contact_loss_type": str(module.tactile_loss_config.contact_loss_type),
            "contact_loss_weight": float(module.tactile_loss_config.contact_loss_weight),
            "contact_pressure_thr": float(module.tactile_loss_config.contact_pressure_thr),
            "contact_temperature": float(module.tactile_loss_config.contact_temperature),
            "optimizer_weight_decay": float(
                getattr(module, "optimizer_weight_decay", 1e-4)
            ),
            "lr_scheduler": str(getattr(module, "lr_scheduler_name", "cosine")),
            "lr_decay_milestones": str(
                getattr(module, "lr_decay_milestones", "0.5,0.75")
            ),
            "lr_decay_milestones_resolved": list(
                getattr(module, "lr_decay_milestones_resolved", ())
            ),
            "lr_decay_gamma": float(getattr(module, "lr_decay_gamma", 0.1)),
            "lr_warmup_epochs": int(getattr(module, "lr_warmup_epochs", 0)),
            "backbone_weights": str(getattr(module, "backbone_weights_path", "") or ""),
            "backbone_sha256": str(getattr(module, "backbone_weights_sha256", "") or ""),
            "tactile_dim": int(getattr(module, "tactile_dim", 0)),
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "reason": str(reason),
        "monitor": str(monitor or ""),
        "score": None if score is None else float(score),
        "wandb_run_id": str(getattr(module, "wandb_run_id", "") or ""),
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


def _scalar_metric_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    elif not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class WandbEpochSyncCallback(Callback):
    """Durably queue epoch metrics and upload them outside the trainer process."""

    def __init__(
        self,
        queue_dir,
        run_id,
        project,
        run_name,
        config_path,
        mode,
        retries,
        interval,
        sync_on_finish=True,
        local_logger=None,
    ):
        super().__init__()
        self.queue_dir = Path(queue_dir)
        self.run_id = str(run_id)
        self.project = str(project)
        self.run_name = str(run_name)
        self.config_path = Path(config_path)
        self.mode = "async" if str(mode) == "online" else str(mode)
        self.retries = max(int(retries), 1)
        self.interval = max(int(interval), 1)
        self.sync_on_finish = bool(sync_on_finish)
        self.local_logger = local_logger
        self._last_payload_key = None
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    @property
    def uploads_enabled(self):
        return self.mode == "async"

    def _collect_metrics(self, trainer, pl_module):
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            scalar = _scalar_metric_value(value)
            if scalar is not None:
                metrics[str(name)] = scalar
        for name, value in getattr(pl_module, "_train_epoch_summary", {}).items():
            scalar = _scalar_metric_value(value)
            if scalar is not None:
                metrics[str(name)] = scalar
        metrics["trainer/epoch"] = float(trainer.current_epoch)
        metrics["trainer/global_step"] = float(trainer.global_step)
        return dict(sorted(metrics.items()))

    def _enqueue(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return False
        key = (int(trainer.current_epoch), int(trainer.global_step))
        if key == self._last_payload_key:
            return False
        if self.local_logger is not None:
            try:
                self.local_logger.save()
            except Exception as exc:
                print(f"Warning: failed to flush local CSV logger: {exc}", flush=True)
        payload_path = self.queue_dir / (
            f"epoch_{key[0]:04d}_step_{key[1]:012d}.json"
        )
        _atomic_write_json(
            payload_path,
            {
                "format": "tactile_wandb_epoch_v1",
                "run_id": self.run_id,
                "epoch": key[0],
                "global_step": key[1],
                "created_unix": time.time(),
                "metrics": self._collect_metrics(trainer, pl_module),
            },
        )
        self._last_payload_key = key
        return True

    def _launch_uploader(self):
        if not self.uploads_enabled:
            return
        log_path = self.queue_dir / "upload.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "wandb_epoch_sync.py"),
            "--queue-dir",
            str(self.queue_dir),
            "--run-id",
            self.run_id,
            "--project",
            self.project,
            "--name",
            self.run_name,
            "--config",
            str(self.config_path),
            "--retries",
            str(self.retries),
            "--interval",
            str(self.interval),
        ]
        with log_path.open("ab", buffering=0) as log_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )
        print(
            f"Queued WandB epoch upload via pid={process.pid}; "
            f"run_id={self.run_id}, log={log_path}",
            flush=True,
        )

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._launch_uploader()

    def on_validation_end(self, trainer, pl_module):
        if self._enqueue(trainer, pl_module):
            self._launch_uploader()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._enqueue(trainer, pl_module):
            self._launch_uploader()

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        self._enqueue(trainer, pl_module)
        if self.sync_on_finish:
            self._launch_uploader()


class CompactBestCheckpoint(Callback):
    def __init__(self, dirpath, filename, monitor, mode):
        super().__init__()
        self.dirpath = str(dirpath)
        self.filename = str(filename)
        self.monitor = str(monitor)
        self.mode = str(mode)
        self.best_model_path = ""
        self.best_model_score = None

    @property
    def state_key(self):
        return (
            f"{self.__class__.__qualname__}:"
            f"{self.filename}:{self.monitor}:{self.mode}"
        )

    def state_dict(self):
        return {
            "best_model_path": self.best_model_path,
            "best_model_score": self.best_model_score,
        }

    def load_state_dict(self, state_dict):
        self.best_model_path = str(state_dict.get("best_model_path", "") or "")
        score = state_dict.get("best_model_score")
        if isinstance(score, torch.Tensor):
            score = float(score.detach().cpu().item())
        self.best_model_score = None if score is None else float(score)

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


class AtomicResumeCheckpoint(Callback):
    """Persist an exact, backbone-stripped Lightning resume checkpoint."""

    def __init__(self, dirpath, every_n_epochs=1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.every_n_epochs = max(int(every_n_epochs), 1)
        self.last_saved_step = -1

    @property
    def state_key(self):
        return f"{self.__class__.__qualname__}:{self.dirpath}"

    def state_dict(self):
        return {"last_saved_step": int(self.last_saved_step)}

    def load_state_dict(self, state_dict):
        self.last_saved_step = int(state_dict.get("last_saved_step", -1))

    def _save(self, trainer):
        if int(trainer.global_step) == self.last_saved_step:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        destination = self.dirpath / "resume.ckpt"
        temporary = self.dirpath / ".resume.ckpt.tmp"
        trainer.save_checkpoint(str(temporary), weights_only=False)
        trainer.strategy.barrier("tactile_resume_checkpoint_written")
        if trainer.is_global_zero:
            os.replace(temporary, destination)
            _fsync_dir(str(self.dirpath))
            _atomic_write_json(
                self.dirpath / "resume_checkpoint_info.json",
                {
                    "format": "tactile_resume_v1",
                    "epoch": int(trainer.current_epoch),
                    "global_step": int(trainer.global_step),
                    "path": str(destination),
                    "wandb_run_id": str(
                        getattr(trainer.lightning_module, "wandb_run_id", "")
                        or ""
                    ),
                    "time_unix": time.time(),
                },
            )
            print(
                f"Atomic resume checkpoint: epoch={trainer.current_epoch}, "
                f"global_step={trainer.global_step} -> {destination}",
                flush=True,
            )
        trainer.strategy.barrier("tactile_resume_checkpoint_materialized")
        self.last_saved_step = int(trainer.global_step)

    def on_train_epoch_end(self, trainer, pl_module):
        if (int(trainer.current_epoch) + 1) % self.every_n_epochs == 0:
            self._save(trainer)

    def on_fit_end(self, trainer, pl_module):
        self._save(trainer)


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
    if float(args.optimizer_weight_decay) < 0.0:
        raise ValueError("--optimizer_weight_decay must be nonnegative")
    if float(args.lr_decay_gamma) <= 0.0:
        raise ValueError("--lr_decay_gamma must be positive")
    if not 0.0 <= float(args.decoder_dropout_scale) <= 1.0:
        raise ValueError("--decoder_dropout_scale must lie in [0, 1]")
    if not 1.0 <= float(args.bbox_rescale_factor) <= 4.0:
        raise ValueError("--bbox_rescale_factor must lie in [1.0, 4.0]")
    args.input_resolution = parse_input_resolution(args.input_resolution)
    if int(args.accumulate_grad_batches) < 1:
        raise ValueError("--accumulate_grad_batches must be at least 1")
    if int(args.resume_save_every_n_epochs) < 1:
        raise ValueError("--resume_save_every_n_epochs must be at least 1")
    if int(args.wandb_sync_retries) < 1:
        raise ValueError("--wandb_sync_retries must be at least 1")
    if int(args.wandb_sync_interval) < 1:
        raise ValueError("--wandb_sync_interval must be at least 1")
    if int(args.pool_output_channels) < 1:
        raise ValueError("--pool_output_channels must be positive")
    if args.index_manifest:
        args.index_manifest = str(Path(args.index_manifest).expanduser().resolve(strict=False))
    if args.query_manifests:
        args.query_manifests = ",".join(
            str(Path(path).expanduser().resolve(strict=False))
            for path in _split_csv(args.query_manifests)
        )
    if args.hdf5_manifest_cache_dir:
        args.hdf5_manifest_cache_dir = str(
            Path(args.hdf5_manifest_cache_dir).expanduser().resolve(strict=False)
        )
    if int(args.hdf5_handle_cache_size) < 1:
        raise ValueError("--hdf5_handle_cache_size must be at least 1")
    args.expected_datasets = list(canonical_dataset_filter(args.expected_datasets))
    if not args.dino_weights:
        raise ValueError("--dino_weights is required")
    args.dino_weights = str(Path(args.dino_weights).expanduser().resolve(strict=False))
    if not Path(args.dino_weights).is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {args.dino_weights}")
    data_dirs = resolve_data_dirs(args)
    print("Resolved training data roots:")
    for data_dir in data_dirs:
        print(f"  - {data_dir}")
    
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
    
    from hamer.configs import get_config
    
    # We can use the original model_config.yaml from hamer
    model_cfg_path = os.path.join(workspace_dir, 'hamer/_DATA/hamer_ckpts/model_config.yaml')
    model_cfg = get_config(model_cfg_path, update_cachedir=True)
    
    model_cfg.defrost()
    model_cfg.MODEL.IMAGE_SIZE = int(args.input_resolution[0])
    model_cfg.MODEL.BBOX_SHAPE = [
        int(args.input_resolution[1]),
        int(args.input_resolution[0]),
    ]
    model_cfg.freeze()
        
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    # Build/load the filesystem index before constructing the large DINO model.
    # Forking an already multithreaded torch/timm process is prone to process-pool stalls.
    train_dataset = OpenTouchTactileDataset(
        cfg=model_cfg,
        split="train",
        data_dir=data_dirs,
        train=True,
        index_workers=args.index_workers,
        index_chunksize=args.index_chunksize,
        index_backend=args.index_backend,
        index_process_worker_cap=args.index_process_worker_cap,
        index_manifest=args.index_manifest,
        expected_datasets=args.expected_datasets,
        index_cache_dir=args.index_cache_dir,
        rebuild_index=args.rebuild_index,
        index_cache_timeout=args.index_cache_timeout,
        tactile_only=True,
        input_resolution=args.input_resolution,
        bbox_rescale_factor=args.bbox_rescale_factor,
        bbox_source_policy=args.bbox_source_policy,
        bbox_manifests=args.bbox_manifests,
        lazy_index_records=args.lazy_index_records,
        augmentation_enabled=args.train_augmentation,
        data_backend=args.data_backend,
        query_manifests=args.query_manifests,
        hdf5_handle_cache_size=args.hdf5_handle_cache_size,
        hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir,
    )
    index_metadata = train_dataset.index_cache_metadata()

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
            index_process_worker_cap=args.index_process_worker_cap,
            index_manifest=args.index_manifest,
            expected_datasets=args.expected_datasets,
            index_cache_dir=args.index_cache_dir,
            rebuild_index=args.rebuild_index,
            index_cache_timeout=args.index_cache_timeout,
            tactile_only=True,
            input_resolution=args.input_resolution,
            bbox_rescale_factor=args.bbox_rescale_factor,
            bbox_source_policy=args.bbox_source_policy,
            bbox_manifests=args.bbox_manifests,
            lazy_index_records=args.lazy_index_records,
            augmentation_enabled=False,
            data_backend=args.data_backend,
            query_manifests=args.query_manifests,
            hdf5_handle_cache_size=args.hdf5_handle_cache_size,
            hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir,
        )

    print("Initializing standalone DINO tactile model...")
    tactile_loss_config = tactile_loss_config_from_args(args)
    backbone_feature_layers = tuple(int(layer) for layer in _split_csv(args.backbone_feature_layers))
    model = TactileTrainingModule(
        cfg=model_cfg,
        learning_rate=lr_scaled,
        tactile_loss_config=tactile_loss_config,
        tactile_only_forward=True,
        tactile_loss_scale=args.tactile_loss_scale,
        lr_warmup_epochs=args.lr_warmup_epochs,
        frame_low_volume_thr=args.frame_low_volume_thr,
        frame_high_volume_thr=args.frame_high_volume_thr,
        sync_train_logs=args.sync_train_logs,
        tactile_head_type=args.tactile_head_type,
        backbone_feature_layers=backbone_feature_layers,
        visual_backbone=args.visual_backbone,
        dino_weights=args.dino_weights or "",
        dino_residual_max_scale=args.dino_residual_max_scale,
        dino_residual_rms_budget=args.dino_residual_rms_budget,
        bbox_rescale_factor=args.bbox_rescale_factor,
        bbox_source_policy=args.bbox_source_policy,
        pool_layout=args.pool_layout,
        decoder_dropout_scale=args.decoder_dropout_scale,
        input_resolution=args.input_resolution,
        pool_output_channels=args.pool_output_channels,
        optimizer_weight_decay=args.optimizer_weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_decay_milestones=args.lr_decay_milestones,
        lr_decay_gamma=args.lr_decay_gamma,
    )
    model.visual_backbone_model_name = model.backbone.MODEL_NAME
    model.backbone_weights_path = args.dino_weights
    print(f"Computing backbone SHA256: {model.backbone_weights_path}")
    model.backbone_weights_sha256 = file_sha256(model.backbone_weights_path)
    model.accumulate_grad_batches_config = int(args.accumulate_grad_batches)
    model.batch_size_config = int(args.batch_size)
    model.index_schema_version = int(index_metadata["index_schema_version"])
    model.bbox_source_policy = str(index_metadata["bbox_source_policy"])
    model.index_cache_key = str(index_metadata["index_cache_key"])
    model.indexed_sample_count = int(index_metadata["indexed_sample_count"])
    model.index_manifest_sha256 = str(index_metadata.get("index_manifest_sha256", ""))
    model.data_backend = str(index_metadata.get("data_backend", args.data_backend))
    model.query_manifest_sha256 = dict(index_metadata.get("query_manifest_sha256", {}))
    storage_schema_version = index_metadata.get("storage_schema_version", "")
    if isinstance(storage_schema_version, (list, tuple)):
        storage_schema_version = ",".join(str(value) for value in storage_schema_version)
    model.hdf5_schema_version = str(storage_schema_version)
    model.hdf5_handle_cache_size = int(args.hdf5_handle_cache_size)
    model.hdf5_manifest_cache_dir = str(args.hdf5_manifest_cache_dir or "")
    model.hdf5_manifest_cache_key = str(
        index_metadata.get("hdf5_manifest_cache_key", "")
    )
    model.bbox_manifest_sha256 = dict(index_metadata.get("bbox_manifest_sha256", {}))
    model.lazy_index_records = bool(index_metadata.get("lazy_index_records", False))
    model.dataset_filter = tuple(index_metadata.get("dataset_filter", ()))
    model.train_augmentation = bool(args.train_augmentation)
    # Validate the complete feature/head shape before allocating DataLoader workers.
    dummy_input = torch.zeros(1, 3, *args.input_resolution)
    with torch.no_grad():
        dummy_feat = model._extract_tactile_features(dummy_input)
        model.tactile_head(dummy_feat)
        print(
            "Tactile head initialized: "
            f"output_dim={model.tactile_dim}, input={model.input_resolution}, "
            f"grid={model.pool_grid_size}, tokens={model.pool_valid_tokens}, "
            f"decoder_input={model.decoder_input_dim}"
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
    total_loader_workers = num_gpus * (int(args.num_workers) + int(val_num_workers))
    if total_loader_workers > 256 and ddp_global_rank() == 0:
        print(
            "WARNING: requested DataLoader worker count is very high: "
            f"train={args.num_workers}/rank, val={val_num_workers}/rank, "
            f"aggregate={total_loader_workers}. The dataset owns millions of Python "
            "index records; forked workers progressively copy those pages as records "
            "are accessed, so host RAM can grow throughout the first epoch. Prefer "
            "16 train and 8 val workers per rank before increasing this further.",
            flush=True,
        )
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
    resume_checkpoint = ""
    resume_wandb_run_id = ""
    if args.resume_from_checkpoint:
        resume_checkpoint = str(
            Path(args.resume_from_checkpoint).expanduser().resolve(strict=False)
        )
    elif args.auto_resume:
        candidate = Path(ckpt_dir) / "resume.ckpt"
        if candidate.is_file():
            resume_checkpoint = str(candidate.resolve())
    if resume_checkpoint:
        if not Path(resume_checkpoint).is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_checkpoint}"
            )
        resume_header = torch.load(resume_checkpoint, map_location="cpu")
        if resume_header.get("format") != "tactile_resume_v1":
            raise ValueError(
                "--resume_from_checkpoint requires format=tactile_resume_v1; "
                "compact loss-best/last checkpoints cannot restore optimizer state"
            )
        resume_wandb_run_id = str(
            resume_header.get("wandb_run_id", "") or ""
        )
        del resume_header
        print(f"Exact training resume enabled: {resume_checkpoint}", flush=True)
    args.resolved_resume_checkpoint = resume_checkpoint
    wandb_enabled = args.use_wandb and args.wandb_mode != "disabled"
    wandb_run_id = ""
    wandb_root = Path(ckpt_dir) / "wandb"
    run_id_path = Path(ckpt_dir) / "wandb_run_id.txt"
    if wandb_enabled:
        wandb_root.mkdir(parents=True, exist_ok=True)
        if resume_checkpoint:
            wandb_run_id = resume_wandb_run_id
            if not wandb_run_id and run_id_path.is_file():
                wandb_run_id = run_id_path.read_text(
                    encoding="utf-8"
                ).strip()
        if not wandb_run_id:
            wandb_run_id = uuid.uuid4().hex[:8]
        run_id_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_run_id_path = run_id_path.with_name(
            f".{run_id_path.name}.tmp-{os.getpid()}"
        )
        temporary_run_id_path.write_text(
            wandb_run_id + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_run_id_path, run_id_path)
        _fsync_dir(str(run_id_path.parent))
        print(
            f"WandB epoch queue: mode={args.wandb_mode}, "
            f"run_id={wandb_run_id}, resume={bool(resume_checkpoint)}",
            flush=True,
        )
    args.wandb_run_id = wandb_run_id
    model.wandb_run_id = wandb_run_id
    provenance = write_run_provenance(args, data_dirs, tactile_loss_config, model, ckpt_dir, num_gpus, lr_scaled)
    val_metrics_text_logger = ValidationMetricsTextLogger(
        os.path.join(ckpt_dir, "val_metrics.txt"),
        config_record=provenance,
        append=bool(resume_checkpoint),
    )
    checkpoint_callbacks = {}
    if val_loader is not None and not args.skip_checkpointing:
        print("Primary checkpoint monitor: val/loss_full_ramp_reference (min)")
        checkpoint_callbacks["loss-best"] = CompactBestCheckpoint(
            dirpath=ckpt_dir,
            filename="best_loss",
            monitor="val/loss_full_ramp_reference",
            mode="min",
        )
        requested_datasets = set(model.dataset_filter)
        if args.save_contact_best and "TouchAnything" in requested_datasets:
            print(
                "Secondary checkpoint monitor: "
                "val/touchanything/touchanything_protocol_contact_iou (max)"
            )
            checkpoint_callbacks["contact-best"] = CompactBestCheckpoint(
                dirpath=ckpt_dir,
                filename="best_contact",
                monitor="val/touchanything/touchanything_protocol_contact_iou",
                mode="max",
            )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    materialized_last_callback = None
    resume_checkpoint_callback = None
    if not args.skip_checkpointing:
        materialized_last_callback = MaterializedLastCheckpointCallback(
            ckpt_dir=ckpt_dir,
            checkpoint_callbacks=checkpoint_callbacks,
        )
        resume_checkpoint_callback = AtomicResumeCheckpoint(
            dirpath=ckpt_dir,
            every_n_epochs=args.resume_save_every_n_epochs,
        )
    
    if num_gpus > 1:
        strategy = "ddp_find_unused_parameters_true" if args.ddp_find_unused_parameters else "ddp_find_unused_parameters_false"
    else:
        strategy = "auto"
    
    local_logger = CSVLogger(
        save_dir=os.path.join(ckpt_dir, "local_logs"),
        name="metrics",
    )
    wandb_epoch_callback = None
    if wandb_enabled:
        queue_dir = (
            wandb_root / "epoch_queue" / wandb_run_id
        )
        wandb_epoch_callback = WandbEpochSyncCallback(
            queue_dir=queue_dir,
            run_id=wandb_run_id,
            project="opentouch-hamer-tactile-ft",
            run_name=args.exp_name,
            config_path=Path(ckpt_dir) / "run_config.json",
            mode=args.wandb_mode,
            retries=args.wandb_sync_retries,
            interval=args.wandb_sync_interval,
            sync_on_finish=args.wandb_sync_on_finish,
            local_logger=local_logger,
        )

    callbacks = [*checkpoint_callbacks.values(), lr_monitor, val_metrics_text_logger]
    if wandb_epoch_callback is not None:
        callbacks.append(wandb_epoch_callback)
    if resume_checkpoint_callback is not None:
        callbacks.append(resume_checkpoint_callback)
    if materialized_last_callback is not None:
        callbacks.append(materialized_last_callback)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        precision=args.trainer_precision,
        logger=local_logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.accumulate_grad_batches,
        use_distributed_sampler=True,
    )
    
    if val_loader is None:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            ckpt_path=resume_checkpoint or None,
        )
    else:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_checkpoint or None,
        )

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
