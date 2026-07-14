#!/usr/bin/env python
import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--render_platform", type=str, default="egl", choices=["egl", "osmesa"])
_early_args, _ = _parser.parse_known_args()
os.environ["PYOPENGL_PLATFORM"] = _early_args.render_platform
os.environ["PYRENDER_PLATFORM"] = _early_args.render_platform

try:
    from .opengl_guard import install_opengl_guard
except ImportError:
    from opengl_guard import install_opengl_guard

install_opengl_guard()

import cv2
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

cv2.setNumThreads(0)
torch.set_float32_matmul_precision("high")

INFILLER_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = INFILLER_DIR.parent
HAMER_DIR = WORKSPACE_DIR / "hamer"
FT_DIR = WORKSPACE_DIR / "hamer_tactile_ft"
sys.path.append(str(HAMER_DIR))
sys.path.append(str(FT_DIR))
sys.path.append(str(INFILLER_DIR))

try:
    from .build_sequence_manifest import build_extracted_sequences, build_motion_list_sequences, write_split_manifests
    from .common import resolve_data_dirs
    from .dataset import TactileSequenceDataset
    from .model import TactileInfiller, infiller_loss, metrics
except ImportError:
    from build_sequence_manifest import build_extracted_sequences, build_motion_list_sequences, write_split_manifests
    from common import resolve_data_dirs
    from dataset import TactileSequenceDataset
    from model import TactileInfiller, infiller_loss, metrics
from hamer.configs import get_config
from losses import TactileLossConfig


class ValidationMetricsTextLogger(Callback):
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
                f.write(json.dumps({"record_type": "config", **self.config_record}, sort_keys=True, ensure_ascii=False) + "\n")

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = {}
        for name, value in trainer.callback_metrics.items():
            if not str(name).startswith("val/"):
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


def load_compatible_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_state = model.state_dict()
    compatible_state = {}
    skipped = []
    for key, value in state_dict.items():
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


class InfillerLightningModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        checkpoint=None,
        learning_rate=1e-4,
        temporal_smooth_weight=0.05,
        joint_finetune=False,
        tactile_loss_config=None,
        sync_train_logs=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "tactile_loss_config"])
        self.tactile_loss_config = tactile_loss_config or TactileLossConfig()
        self.sync_train_logs = bool(sync_train_logs)
        self.model = TactileInfiller(cfg)
        self.model.initialize_lazy_layers()
        if checkpoint:
            print(f"Loading frame tactile checkpoint: {checkpoint}")
            load_compatible_state_dict(self.model.frame_model, checkpoint)
        self.model.freeze_frame_model()
        if joint_finetune:
            self.model.unfreeze_tactile_head()
        self._val_eval_stats = None

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss, losses = infiller_loss(
            batch,
            output,
            self.hparams.temporal_smooth_weight,
            tactile_loss_config=self.tactile_loss_config,
            current_epoch=getattr(self, "current_epoch", 0),
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_train_logs)
        self.log("train/loss_tactile", losses["loss_tactile"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_train_logs)
        self.log("train/loss_temporal", losses["loss_temporal"], on_step=True, on_epoch=True, logger=True, sync_dist=self.sync_train_logs)
        self._log_tactile_loss_breakdown("train", losses, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model(batch)
        loss, losses = infiller_loss(
            batch,
            output,
            self.hparams.temporal_smooth_weight,
            tactile_loss_config=self.tactile_loss_config,
            current_epoch=getattr(self, "current_epoch", 0),
        )
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/loss_tactile", losses["loss_tactile"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/loss_temporal", losses["loss_temporal"], on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self._log_tactile_loss_breakdown("val", losses, on_step=False)
        for name, value in metrics(batch, output).items():
            self.log(f"val/{name}", value, on_step=False, on_epoch=True, prog_bar=name == "missing_bbox_mae", logger=True, sync_dist=True)
        self._accumulate_val_eval_stats(batch, output)
        return loss

    def on_validation_epoch_start(self):
        self._val_eval_stats = None

    def _empty_eval_stats(self, device):
        return torch.zeros(42, dtype=torch.float32, device=device)

    def _add_subset_stats(self, stats, offset, pred, target, palm, frame_mask):
        valid = frame_mask > 0.5
        frame_count = valid.sum()
        if frame_count <= 0:
            return
        mask = valid.unsqueeze(-1).float() * palm
        diff = (pred - target) * mask
        values = mask.sum()
        active_thr = float(self.tactile_loss_config.active_pressure_thr)
        background_thr = 0.02

        active_mask = ((target > active_thr).float() * mask)
        background_mask = ((target <= background_thr).float() * mask)
        pred_active = ((pred > active_thr).float() * mask)

        vol_intersection = (torch.minimum(pred, target) * mask).sum(dim=-1)
        vol_union = torch.clamp((torch.maximum(pred, target) * mask).sum(dim=-1), min=1e-6)
        vol_iou = (vol_intersection / vol_union) * valid.float()

        stats[offset + 0] += frame_count.float()
        stats[offset + 1] += values
        stats[offset + 2] += diff.abs().sum()
        stats[offset + 3] += diff.pow(2).sum()
        stats[offset + 4] += (pred * mask).sum()
        stats[offset + 5] += (target * mask).sum()
        stats[offset + 6] += (diff.abs() * active_mask).sum()
        stats[offset + 7] += active_mask.sum()
        stats[offset + 8] += (diff.abs() * background_mask).sum()
        stats[offset + 9] += background_mask.sum()
        stats[offset + 10] += (pred_active * active_mask).sum()
        stats[offset + 11] += active_mask.sum()
        stats[offset + 12] += (pred_active * background_mask).sum()
        stats[offset + 13] += vol_iou.sum()

    def _accumulate_val_eval_stats(self, batch, output):
        pred = output["pred_tactile"].detach().float()
        target = batch["tactile_signal"].detach().float()
        palm = (batch["palm_mask"].detach().float() > 0.5).float()
        target_mask = batch["target_mask"].detach().float()
        bbox_valid = batch["bbox_valid"].detach().float()

        if self._val_eval_stats is None:
            self._val_eval_stats = self._empty_eval_stats(pred.device)

        self._add_subset_stats(self._val_eval_stats, 0, pred, target, palm, target_mask)
        self._add_subset_stats(self._val_eval_stats, 14, pred, target, palm, target_mask * bbox_valid)
        self._add_subset_stats(self._val_eval_stats, 28, pred, target, palm, target_mask * (1.0 - bbox_valid))

    def on_validation_epoch_end(self):
        if self._val_eval_stats is None:
            return
        stats = self._val_eval_stats
        if self.trainer.world_size > 1:
            gathered = self.all_gather(stats)
            stats = gathered.reshape(-1, stats.numel()).sum(dim=0)

        subset_names = {
            0: "all",
            14: "observed_bbox",
            28: "missing_bbox",
        }
        for offset, subset in subset_names.items():
            frames = stats[offset + 0]
            if frames <= 0:
                continue
            values = stats[offset + 1].clamp_min(1.0)
            gt_volume = stats[offset + 5].clamp_min(1e-6)
            active_count = stats[offset + 7].clamp_min(1.0)
            background_count = stats[offset + 9].clamp_min(1.0)
            active_gt_count = stats[offset + 11].clamp_min(1.0)
            metrics_to_log = {
                f"val/eval_{subset}_mae": stats[offset + 2] / values,
                f"val/eval_{subset}_rmse": torch.sqrt(stats[offset + 3] / values),
                f"val/eval_{subset}_pred_gt_volume_ratio": stats[offset + 4] / gt_volume,
                f"val/eval_{subset}_active_mae": stats[offset + 6] / active_count,
                f"val/eval_{subset}_background_mae": stats[offset + 8] / background_count,
                f"val/eval_{subset}_active_recall": stats[offset + 10] / active_gt_count,
                f"val/eval_{subset}_bg_false_positive": stats[offset + 12] / background_count,
                f"val/eval_{subset}_volumetric_iou": stats[offset + 13] / frames.clamp_min(1.0),
            }
            for name, value in metrics_to_log.items():
                self.log(
                    name,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=name in {"val/eval_missing_bbox_mae", "val/eval_missing_bbox_pred_gt_volume_ratio"},
                    logger=True,
                    rank_zero_only=True,
                )

    def _log_tactile_loss_breakdown(self, prefix, losses, on_step):
        mapping = {
            "loss_base_pressure": "loss/base_pressure",
            "loss_final_pressure": "loss/final_pressure",
            "loss_ordinal": "loss/ordinal",
            "loss_residual_reg": "loss/residual_reg",
            "loss_tactile": "loss/total",
            "schedule_pressure_loss": "schedule/pressure_loss",
            "schedule_active_pressure_weight": "schedule/active_pressure_weight",
        }
        for key, name in mapping.items():
            if key in losses:
                self.log(
                    f"{prefix}/{name}",
                    losses[key],
                    on_step=on_step,
                    on_epoch=True,
                    logger=True,
                    sync_dist=self.sync_train_logs if prefix == "train" else True,
                )

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "norm", "LayerNorm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 1e-4},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, self.trainer.estimated_stepping_batches),
            eta_min=self.hparams.learning_rate * 0.01,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def prepare_model_cfg():
    cfg_path = WORKSPACE_DIR / "hamer" / "_DATA" / "hamer_ckpts" / "model_config.yaml"
    cfg = get_config(str(cfg_path), update_cachedir=True)
    if (cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in cfg.MODEL):
        cfg.defrost()
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.freeze()
    if "PRETRAINED_WEIGHTS" in cfg.MODEL.BACKBONE:
        cfg.defrost()
        cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
        cfg.freeze()
    return cfg


def ensure_manifests(args):
    output_dir = Path(args.manifest_dir)
    train_manifest = output_dir / "sequence_manifest_train.jsonl"
    if args.manifest_prefix:
        prefixes = [x.strip() for x in args.manifest_prefix.split(",") if x.strip()]
        if len(prefixes) > 1 and args.rebuild_manifest:
            raise ValueError(
                "--rebuild_manifest with multiple --manifest_prefix values is ambiguous. "
                "Build each dataset manifest separately, then train without --rebuild_manifest."
            )
        first_prefix = prefixes[0]
        train_manifest = output_dir / f"sequence_manifest_{first_prefix}_train.jsonl"
    if train_manifest.exists() and not args.rebuild_manifest:
        return output_dir

    print("Building sequence manifests...")
    if args.egotactile_split_source in {"motion_list", "derived"}:
        rows = build_motion_list_sequences(args.egopressure_motion_list, args.egotactile_split_source)
    else:
        rows = build_extracted_sequences(resolve_data_dirs(args.datasets, args.data_dir), workers=args.manifest_workers)
    if not rows:
        raise RuntimeError("No sequence rows found while building manifests.")
    write_split_manifests(rows, output_dir, prefix=args.manifest_prefix)
    return output_dir


def manifest_paths_for_split(manifest_dir, split, manifest_prefix=None):
    manifest_dir = Path(manifest_dir)
    prefixes = [x.strip() for x in str(manifest_prefix or "").split(",") if x.strip()]
    if prefixes:
        paths = [manifest_dir / f"sequence_manifest_{prefix}_{split}.jsonl" for prefix in prefixes]
        return [path for path in paths if path.exists()]

    standard = manifest_dir / f"sequence_manifest_{split}.jsonl"
    if standard.exists():
        return [standard]

    return sorted(manifest_dir.glob(f"sequence_manifest_*_{split}.jsonl"))


def make_dataset(cfg, manifest_dir, split, args, train):
    manifests = manifest_paths_for_split(manifest_dir, split, args.manifest_prefix)
    if not manifests:
        return None
    return TactileSequenceDataset(
        cfg=cfg,
        manifest_path=manifests,
        split=split,
        train=train,
        seq_len=args.seq_len,
        seq_stride=args.seq_stride if train else args.eval_seq_stride,
        sample_frame_rate=args.sample_frame_rate,
        min_observed_bbox=args.min_observed_bbox,
        allow_missing_bbox=args.allow_missing_bbox,
        mask_prob=args.mask_prob if train else 0.0,
        target_policy=args.target_policy,
        missing_bbox_weight=args.missing_bbox_weight,
        observed_bbox_weight=args.observed_bbox_weight,
        pressure_key_priority=[x.strip() for x in args.pressure_key_priority.split(">") if x.strip()],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train sequence-level tactile infiller.")
    parser.add_argument("--checkpoint", default=str(WORKSPACE_DIR / "hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt"))
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--manifest_dir", default=str(INFILLER_DIR / "manifests"))
    parser.add_argument(
        "--manifest_prefix",
        default=None,
        help=(
            "Comma-separated per-dataset manifest prefixes to read/write. "
            "Example: opentouch,touchanything,egotactile reads sequence_manifest_opentouch_train.jsonl etc."
        ),
    )
    parser.add_argument("--rebuild_manifest", action="store_true")
    parser.add_argument("--egotactile_split_source", default="extracted", choices=["extracted", "motion_list", "derived"])
    parser.add_argument("--egopressure_motion_list", default=str(WORKSPACE_DIR / "EgoPressureDiff/V2P_data/motion_list.txt"))
    parser.add_argument("--manifest_workers", type=int, default=1)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_num_workers", type=int, default=None)
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--exp_name", default="tactile_infiller")
    parser.add_argument("--checkpoint_monitor", default="val/eval_missing_bbox_mae")
    parser.add_argument("--checkpoint_mode", default="min", choices=["min", "max"])
    parser.add_argument("--ddp_find_unused_parameters", action="store_true")
    parser.add_argument("--sync_train_logs", action="store_true")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--quick_test", action="store_true")
    parser.add_argument("--joint_finetune", action="store_true")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--seq_stride", type=int, default=8)
    parser.add_argument("--eval_seq_stride", type=int, default=16)
    parser.add_argument("--sample_frame_rate", type=int, default=1)
    parser.add_argument("--min_observed_bbox", type=int, default=1)
    parser.add_argument("--allow_missing_bbox", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mask_prob", type=float, default=0.5)
    parser.add_argument("--target_policy", default="has_tactile")
    parser.add_argument("--missing_bbox_weight", type=float, default=1.0)
    parser.add_argument("--observed_bbox_weight", type=float, default=0.5)
    parser.add_argument("--pressure_key_priority", default="continuous_subdiv>gaussian_pressure>original_hdf5_data")
    parser.add_argument("--temporal_smooth_weight", type=float, default=0.05)
    parser.add_argument("--active_pressure_thr", type=float, default=0.05)
    parser.add_argument("--active_pressure_weight", type=float, default=1.5)
    parser.add_argument("--active_pressure_gamma", type=float, default=0.7)
    parser.add_argument("--active_pressure_max_weight", type=float, default=2.5)
    parser.add_argument("--active_pressure_weight_warmup_epochs", type=int, default=10)
    parser.add_argument("--pressure_loss_warmup_epochs", type=int, default=0)
    parser.add_argument("--pressure_loss_warmup_start", type=float, default=1.0)
    parser.add_argument("--opentouch_high_pressure_thr", type=float, default=0.9)
    parser.add_argument("--opentouch_high_pressure_weight", type=float, default=0.3)
    parser.add_argument("--render_platform", default="egl", choices=["egl", "osmesa"])
    return parser.parse_args()


def tactile_loss_config_from_args(args):
    return TactileLossConfig(
        active_pressure_thr=args.active_pressure_thr,
        active_pressure_weight=args.active_pressure_weight,
        active_pressure_gamma=args.active_pressure_gamma,
        active_pressure_max_weight=args.active_pressure_max_weight,
        active_pressure_weight_warmup_epochs=args.active_pressure_weight_warmup_epochs,
        pressure_loss_warmup_epochs=args.pressure_loss_warmup_epochs,
        pressure_loss_warmup_start=args.pressure_loss_warmup_start,
        ordinal_loss_weight=0.0,
        ordinal_thresholds="0.005,0.02,0.05,0.1,0.2,0.4,0.7",
        ordinal_threshold_weights="0.25,0.5,1.0,1.0,1.5,2.0,3.0",
        ordinal_pos_weights="1,1,2,3,4,6,8",
        residual_reg_loss_weight=0.0,
        residual_warmup_epochs=0,
        residual_max_scale=0.0,
        opentouch_high_pressure_thr=args.opentouch_high_pressure_thr,
        opentouch_high_pressure_weight=args.opentouch_high_pressure_weight,
    )


def _run_git_command(args):
    try:
        return subprocess.check_output(args, cwd=str(WORKSPACE_DIR), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_snapshot():
    return {
        "commit": _run_git_command(["git", "rev-parse", "HEAD"]),
        "branch": _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty_files": _run_git_command(["git", "status", "--short"]).splitlines(),
        "diff_stat": _run_git_command(["git", "diff", "--stat"]).splitlines(),
    }


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)


def write_run_provenance(args, tactile_loss_config, model, ckpt_dir, num_gpus, val_split):
    os.makedirs(ckpt_dir, exist_ok=True)
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total = int(sum(p.numel() for p in model.parameters()))
    provenance = {
        "run_config": {
            "args": vars(args),
            "num_gpus": num_gpus,
            "val_split": val_split,
            "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        },
        "loss_config": asdict(tactile_loss_config),
        "model_config": {
            "tactile_dim": int(model.model.tactile_dim),
            "parameter_counts": {
                "total": total,
                "trainable": trainable,
                "frozen": total - trainable,
            },
        },
        "git_snapshot": git_snapshot(),
    }
    write_json(str(Path(ckpt_dir) / "run_config.json"), provenance["run_config"])
    write_json(str(Path(ckpt_dir) / "loss_config.json"), provenance["loss_config"])
    write_json(str(Path(ckpt_dir) / "model_config.json"), provenance["model_config"])
    write_json(str(Path(ckpt_dir) / "git_snapshot.json"), provenance["git_snapshot"])
    with open(Path(ckpt_dir) / "run_command.txt", "w", encoding="utf-8") as f:
        f.write(provenance["run_config"]["command"] + "\n")
    with open(Path(ckpt_dir) / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(
            "Tactile infiller model summary\n"
            f"tactile_dim: {model.model.tactile_dim}\n"
            f"total_params: {total:,}\n"
            f"trainable_params: {trainable:,}\n"
            f"frozen_params: {total - trainable:,}\n"
        )
    return provenance


def make_dataloader(dataset, batch_size, shuffle, num_workers, persistent_workers, prefetch_factor):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    num_gpus = len([x for x in args.gpus.split(",") if x.strip()])

    cfg = prepare_model_cfg()
    manifest_dir = ensure_manifests(args)
    train_dataset = make_dataset(cfg, manifest_dir, "train", args, train=True)
    val_dataset = None
    val_split = None
    for candidate in ("val", "test_seen", "test_unseen", "test"):
        candidate_dataset = make_dataset(cfg, manifest_dir, candidate, args, train=False)
        if candidate_dataset is not None and len(candidate_dataset) > 0:
            val_dataset = candidate_dataset
            val_split = candidate
            break
    if train_dataset is None or len(train_dataset) == 0:
        raise RuntimeError("Infiller train dataset is empty.")

    if args.quick_test:
        train_dataset.windows = train_dataset.windows[:8]
        if val_dataset is not None:
            val_dataset.windows = val_dataset.windows[:4]
        args.epochs = 1
        args.num_workers = 0
        args.val_num_workers = 0

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
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
        print(f"Using '{val_split}' manifest for validation.")
    else:
        print("No val/test manifest produced non-empty windows; training without val monitor.")

    tactile_loss_config = tactile_loss_config_from_args(args)
    model = InfillerLightningModule(
        cfg=cfg,
        checkpoint=args.checkpoint if args.checkpoint else None,
        learning_rate=args.lr * max(1, num_gpus),
        temporal_smooth_weight=args.temporal_smooth_weight,
        joint_finetune=args.joint_finetune,
        tactile_loss_config=tactile_loss_config,
        sync_train_logs=args.sync_train_logs,
    )

    ckpt_dir = INFILLER_DIR / "checkpoints" / args.exp_name
    provenance = write_run_provenance(args, tactile_loss_config, model, str(ckpt_dir), num_gpus, val_split)
    val_metrics_text_logger = ValidationMetricsTextLogger(str(ckpt_dir / "val_metrics.txt"), config_record=provenance)
    if val_loader is None:
        checkpoint_callback = ModelCheckpoint(dirpath=str(ckpt_dir), filename="infiller", save_top_k=0, save_last=True)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best_infiller",
            monitor=args.checkpoint_monitor,
            mode=args.checkpoint_mode,
            save_top_k=1,
            save_last=True,
        )
    logger = (
        WandbLogger(
            project="tactile-infiller",
            name=args.exp_name,
            config=provenance,
        )
        if args.use_wandb
        else True
    )
    strategy = "auto"
    if num_gpus > 1:
        strategy = "ddp_find_unused_parameters_true" if args.ddp_find_unused_parameters else "ddp_find_unused_parameters_false"
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step"), val_metrics_text_logger],
        log_every_n_steps=10,
        enable_progress_bar=True,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    if val_loader is None:
        trainer.fit(model, train_dataloaders=train_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Done. Checkpoint: {checkpoint_callback.best_model_path or checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()
