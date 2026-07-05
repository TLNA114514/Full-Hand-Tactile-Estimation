#!/usr/bin/env python
import argparse
import os
import sys
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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "tactile_loss_config"])
        self.tactile_loss_config = tactile_loss_config or TactileLossConfig()
        self.model = TactileInfiller(cfg)
        self.model.initialize_lazy_layers()
        if checkpoint:
            print(f"Loading frame tactile checkpoint: {checkpoint}")
            load_compatible_state_dict(self.model.frame_model, checkpoint)
        self.model.freeze_frame_model()
        if joint_finetune:
            self.model.unfreeze_tactile_head()

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        loss, losses = infiller_loss(
            batch,
            output,
            self.hparams.temporal_smooth_weight,
            tactile_loss_config=self.tactile_loss_config,
            current_epoch=getattr(self, "current_epoch", 0),
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss_tactile", losses["loss_tactile"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/loss_temporal", losses["loss_temporal"], on_step=True, on_epoch=True, logger=True, sync_dist=True)
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
        return loss

    def _log_tactile_loss_breakdown(self, prefix, losses, on_step):
        mapping = {
            "loss_base_tactile": "loss/base_tactile",
            "loss_weighted_tactile": "loss/weighted_tactile",
            "loss_background": "loss/background",
            "loss_volume_iou": "loss/volume_iou",
            "pred_volume": "stats/pred_volume",
            "gt_volume": "stats/gt_volume",
            "pred_gt_volume_ratio": "stats/pred_gt_volume_ratio",
        }
        for key, name in mapping.items():
            if key in losses:
                self.log(f"{prefix}/{name}", losses[key], on_step=on_step, on_epoch=True, logger=True, sync_dist=True)

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
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--exp_name", default="tactile_infiller")
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
    parser.add_argument("--active_pressure_peak", type=float, default=0.12)
    parser.add_argument("--active_pressure_high", type=float, default=0.60)
    parser.add_argument("--background_pressure_thr", type=float, default=0.02)
    parser.add_argument("--background_pred_margin", type=float, default=0.02)
    parser.add_argument("--active_pressure_weight", type=float, default=4.0)
    parser.add_argument("--active_pressure_gamma", type=float, default=1.0)
    parser.add_argument("--background_loss_weight", type=float, default=0.5)
    parser.add_argument("--volume_iou_loss_weight", type=float, default=0.2)
    parser.add_argument("--opentouch_high_pressure_thr", type=float, default=0.9)
    parser.add_argument("--opentouch_high_pressure_weight", type=float, default=0.3)
    parser.add_argument("--loss_ramp_epochs", type=int, default=5)
    parser.add_argument("--render_platform", default="egl", choices=["egl", "osmesa"])
    return parser.parse_args()


def tactile_loss_config_from_args(args):
    return TactileLossConfig(
        active_pressure_thr=args.active_pressure_thr,
        active_pressure_peak=args.active_pressure_peak,
        active_pressure_high=args.active_pressure_high,
        background_pressure_thr=args.background_pressure_thr,
        background_pred_margin=args.background_pred_margin,
        active_pressure_weight=args.active_pressure_weight,
        active_pressure_gamma=args.active_pressure_gamma,
        background_loss_weight=args.background_loss_weight,
        volume_iou_loss_weight=args.volume_iou_loss_weight,
        opentouch_high_pressure_thr=args.opentouch_high_pressure_thr,
        opentouch_high_pressure_weight=args.opentouch_high_pressure_weight,
        loss_ramp_epochs=args.loss_ramp_epochs,
    )


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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        print(f"Using '{val_split}' manifest for validation.")
    else:
        print("No val/test manifest produced non-empty windows; training without val monitor.")

    model = InfillerLightningModule(
        cfg=cfg,
        checkpoint=args.checkpoint if args.checkpoint else None,
        learning_rate=args.lr * max(1, num_gpus),
        temporal_smooth_weight=args.temporal_smooth_weight,
        joint_finetune=args.joint_finetune,
        tactile_loss_config=tactile_loss_config_from_args(args),
    )

    ckpt_dir = INFILLER_DIR / "checkpoints" / args.exp_name
    if val_loader is None:
        checkpoint_callback = ModelCheckpoint(dirpath=str(ckpt_dir), filename="infiller", save_top_k=0, save_last=True)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best_infiller",
            monitor="val/missing_bbox_mae",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    logger = WandbLogger(project="tactile-infiller", name=args.exp_name) if args.use_wandb else True
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy="ddp_find_unused_parameters_true" if num_gpus > 1 else "auto",
        precision="16-mixed",
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    if val_loader is None:
        trainer.fit(model, train_dataloaders=train_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"Done. Checkpoint: {checkpoint_callback.best_model_path or checkpoint_callback.last_model_path}")


if __name__ == "__main__":
    main()
