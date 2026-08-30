#!/usr/bin/env python3
"""Train a pressure-inert temporal down/hold/up selector from frozen caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hamer_tactile_ft.process_lifecycle import (
    configure_supervised_process,
    initialize_worker_parent_death_signal,
)
from tactile_input_priors.runtime import (
    atomic_torch_save,
    file_sha256,
    parse_csv,
)
from tactile_input_priors.temporal_flow import (
    TemporalActionSelectorV2,
    TemporalPairDataset,
    history_quality_context,
    temporal_action_targets,
    temporal_selector_checkpoint_payload,
)
from tactile_input_priors.temporal_selector_metrics import (
    ACTION_NAMES,
    BinaryScoreMetricAccumulator,
    TemporalActionMetricAccumulator,
)
from tactile_input_priors.train_prior_adapter import (
    AtomicPriorResumeCheckpoint,
    DurableWandbEpochCallback,
    RuntimeBatchTimingCallback,
    RuntimeSystemMonitorCallback,
)
from tactile_input_priors.train_temporal_flow import (
    DistributedPairBlockSampler,
    _pair_index,
    _parse_history_lags,
)


configure_supervised_process()
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


def _balanced_action_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    element = F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1), reduction="none")
    labels = target.reshape(-1)
    mask = valid[:, None].expand_as(target).reshape(-1).bool()
    terms = [
        element[mask & (labels == action)].mean()
        for action in range(3)
        if bool((mask & (labels == action)).any())
    ]
    return torch.stack(terms).mean() if terms else element.sum() * 0.0


class TemporalSelectorTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: TemporalActionSelectorV2,
        *,
        model_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        base_checkpoint: str,
        output_dir: str,
        action_margin: float,
        learning_rate: float,
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        wandb_run_id: str,
    ):
        super().__init__()
        self.model = model
        self.model_config = dict(model_config)
        self.data_config = dict(data_config)
        self.base_checkpoint = str(base_checkpoint)
        self.output_dir = Path(output_dir)
        self.action_margin = float(action_margin)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs_config = int(max_epochs)
        self.wandb_run_id = str(wandb_run_id)
        self.best_loss = math.inf
        self.best_macro_ap = -math.inf
        self.best_strict_clear_ap = -math.inf
        self._loading_resume = False
        self._class_counts = torch.zeros(3, dtype=torch.float64)
        self._class_prior_updated = False
        self._train_sum = torch.zeros((), dtype=torch.float64)
        self._train_count = torch.zeros((), dtype=torch.float64)
        self._val_sum = torch.zeros((), dtype=torch.float64)
        self._val_count = torch.zeros((), dtype=torch.float64)
        self._val_dino_sum = torch.zeros(4, dtype=torch.float64)
        self.train_loss_epoch_global = math.nan
        self.val_metrics = TemporalActionMetricAccumulator()
        self.val_strict_clear = BinaryScoreMetricAccumulator(bins=2048)

    def _resume_contract(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "data_config": self.data_config,
            "base_checkpoint_sha256": file_sha256(self.base_checkpoint),
            "action_margin": self.action_margin,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs_config,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"] = {
            name: value
            for name, value in checkpoint.get("state_dict", {}).items()
            if name.startswith("model.")
        }
        checkpoint["format"] = "tactile_temporal_action_selector_resume_v2"
        checkpoint["resume_contract"] = self._resume_contract()
        checkpoint["wandb_run_id"] = self.wandb_run_id
        checkpoint["best_loss"] = self.best_loss
        checkpoint["best_macro_ap"] = self.best_macro_ap
        checkpoint["best_strict_clear_ap"] = self.best_strict_clear_ap

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != "tactile_temporal_action_selector_resume_v2":
            return
        if checkpoint.get("resume_contract") != self._resume_contract():
            raise RuntimeError("Temporal-selector resume configuration mismatch")
        self.wandb_run_id = str(checkpoint.get("wandb_run_id") or self.wandb_run_id)
        self.best_loss = float(checkpoint.get("best_loss", math.inf))
        self.best_macro_ap = float(checkpoint.get("best_macro_ap", -math.inf))
        self.best_strict_clear_ap = float(
            checkpoint.get("best_strict_clear_ap", -math.inf)
        )
        self._loading_resume = True

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self._loading_resume:
            return super().load_state_dict(state_dict, strict=strict)
        result = super().load_state_dict(state_dict, strict=False)
        self._loading_resume = False
        missing = [name for name in result.missing_keys if name.startswith("model.")]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"Temporal selector resume state mismatch: missing={missing}, "
                f"unexpected={result.unexpected_keys}"
            )
        return result

    def _forward(self, batch, *, corrected: bool):
        quality = (
            history_quality_context(
                batch,
                self.model.history_lags,
                nominal_fps=self.model.nominal_fps,
            )
            if self.model.use_per_lag_quality
            else None
        )
        dino_kwargs = {}
        if self.model.uses_dino_history:
            dino_kwargs = {
                "current_grid": batch["current_grid"],
                "history_grids": batch["history_grids"],
                "history_crop_transform": batch["history_crop_transform"],
            }
        output = self.model(
            batch["current_logits"],
            batch["history_logits"],
            batch["history_available"],
            quality,
            apply_prior_correction=corrected,
            **dino_kwargs,
        )
        target = temporal_action_targets(
            batch["current_logits"],
            batch["tactile_signal"],
            output["anchor_local_indices"],
            margin=self.action_margin,
        )
        valid = batch["has_tactile"].float()
        loss = _balanced_action_loss(
            output["balanced_action_logits"], target, valid
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite temporal selector loss at epoch={self.current_epoch}, "
                f"step={self.global_step}"
            )
        return output, target, valid, loss

    def on_train_epoch_start(self) -> None:
        sampler = getattr(self.trainer.train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(self.current_epoch))
        self._class_counts = torch.zeros(3, device=self.device, dtype=torch.float64)
        self._class_prior_updated = False
        self._train_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._train_count = torch.zeros((), device=self.device, dtype=torch.float64)

    def training_step(self, batch, batch_idx):
        del batch_idx
        output, target, valid, loss = self._forward(batch, corrected=False)
        anchor_valid = valid[:, None].expand_as(target).bool()
        labels = target[anchor_valid]
        self._class_counts += torch.bincount(labels, minlength=3).double()
        batch_size = int(batch["tactile_signal"].shape[0])
        self._train_sum += loss.detach().double() * batch_size
        self._train_count += batch_size
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        probability = output["balanced_action_logits"].softmax(dim=-1)
        self.log(
            "train/action_accuracy",
            (probability.argmax(dim=-1)[anchor_valid] == labels).float().mean(),
            on_epoch=True,
            batch_size=batch_size,
        )
        if "dino_effective_gate" in output:
            self.log(
                "train/dino_effective_gate",
                output["dino_effective_gate"],
                on_step=True,
                on_epoch=True,
                batch_size=batch_size,
            )
            self.log(
                "train/dino_residual_rms",
                output["dino_residual_rms"],
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def on_train_epoch_end(self) -> None:
        values = torch.stack((self._train_sum, self._train_count))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        self.train_loss_epoch_global = float(values[0] / values[1].clamp_min(1.0))
        self.log("train/loss_epoch_global", self.train_loss_epoch_global, sync_dist=False)

    def _update_class_prior(self) -> None:
        if self._class_prior_updated or float(self._class_counts.sum()) <= 0.0:
            return
        counts = self._class_counts.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts)
        self.model.set_class_prior(counts)
        self._class_prior_updated = True
        for action, name in enumerate(ACTION_NAMES):
            self.log(f"train/{name}_prior", self.model.class_prior[action], sync_dist=False)

    def on_validation_epoch_start(self) -> None:
        if not self.trainer.sanity_checking:
            self._update_class_prior()
        self.val_metrics.reset()
        self.val_strict_clear = BinaryScoreMetricAccumulator(bins=2048)
        self._val_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_count = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_dino_sum = torch.zeros(4, device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        del batch_idx
        output, target, valid, loss = self._forward(batch, corrected=True)
        batch_size = int(batch["tactile_signal"].shape[0])
        self._val_sum += loss.detach().double() * batch_size
        self._val_count += batch_size
        if self.model.uses_dino_history:
            self._val_dino_sum += torch.stack(
                (
                    output["dino_effective_gate"],
                    output["dino_residual_rms"],
                    output["dino_valid_token_fraction"],
                    output["dino_motion_rms"],
                )
            ).detach().double() * batch_size
        self.val_metrics.update(
            output["action_probability"], target, valid > 0.5, loss=loss
        )
        anchor = output["anchor_local_indices"]
        prediction = torch.sigmoid(batch["current_logits"].float())[:, anchor]
        tactile = batch["tactile_signal"].float()[:, anchor]
        candidate = prediction >= 0.10
        strict = candidate & (tactile <= 0.02)
        protected = candidate & (tactile >= 0.10)
        strict_valid = (strict | protected) & (valid[:, None] > 0.5)
        self.val_strict_clear.update(
            output["action_probability"][..., 0], strict, strict_valid
        )
        return loss

    def _compact(self, monitor: str, score: float):
        return temporal_selector_checkpoint_payload(
            self.model,
            model_config=self.model_config,
            data_config=self.data_config,
            base_checkpoint=self.base_checkpoint,
            action_margin=self.action_margin,
            epoch=self.current_epoch,
            global_step=self.global_step,
            monitor=monitor,
            score=score,
        )

    def on_validation_epoch_end(self) -> None:
        self.val_metrics.synchronize(self.device)
        self.val_strict_clear.synchronize(self.device)
        totals = torch.cat(
            (
                torch.stack((self._val_sum, self._val_count)),
                self._val_dino_sum,
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        val_loss = float(totals[0] / totals[1].clamp_min(1.0))
        summary = self.val_metrics.summary()
        macro_ap = float(summary["macro_average_precision"])
        strict_clear = self.val_strict_clear.summary()
        strict_clear_ap = float(strict_clear["average_precision"])
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=False)
        self.log("val/macro_ap", macro_ap, prog_bar=True, sync_dist=False)
        self.log("val/strict_clear_ap", strict_clear_ap, prog_bar=True, sync_dist=False)
        self.log("val/macro_f1", float(summary["macro_f1"]), sync_dist=False)
        self.log("val/ece", float(summary["ece"]), sync_dist=False)
        dino_values = None
        if self.model.uses_dino_history:
            dino_values = totals[2:] / totals[1].clamp_min(1.0)
            for index, name in enumerate(
                (
                    "effective_gate",
                    "residual_rms",
                    "valid_token_fraction",
                    "motion_rms",
                )
            ):
                self.log(f"val/dino_{name}", dino_values[index], sync_dist=False)
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        row = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "train_loss_epoch_global": self.train_loss_epoch_global,
            "val_loss": val_loss,
            "macro_average_precision": macro_ap,
            "strict_clear_average_precision": strict_clear_ap,
            "strict_clear_auroc": strict_clear["auroc"],
            "strict_clear_positive_fraction": strict_clear["positive_fraction"],
            "macro_auroc": summary["macro_auroc"],
            "macro_f1": summary["macro_f1"],
            "accuracy": summary["accuracy"],
            "ece": summary["ece"],
        }
        if self.model.uses_dino_history:
            row.update(
                {
                    "dino_effective_gate": float(dino_values[0]),
                    "dino_residual_rms": float(dino_values[1]),
                    "dino_valid_token_fraction": float(dino_values[2]),
                    "dino_motion_rms": float(dino_values[3]),
                }
            )
        for name in ACTION_NAMES:
            for metric in ("average_precision", "auroc", "precision", "recall", "f1", "count"):
                row[f"{name}_{metric}"] = summary["per_class"][name][metric]
        path = self.output_dir / "val_metrics.csv"
        write_header = not path.is_file()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        checkpoint_dir = self.output_dir / "checkpoints"
        atomic_torch_save(self._compact("last", macro_ap), checkpoint_dir / "last.ckpt")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            atomic_torch_save(
                self._compact("val/loss", val_loss), checkpoint_dir / "best_loss.ckpt"
            )
        if macro_ap > self.best_macro_ap:
            self.best_macro_ap = macro_ap
            atomic_torch_save(
                self._compact("val/macro_ap", macro_ap), checkpoint_dir / "selector-best.ckpt"
            )
        if strict_clear_ap > self.best_strict_clear_ap:
            self.best_strict_clear_ap = strict_clear_ap
            atomic_torch_save(
                self._compact("val/strict_clear_ap", strict_clear_ap),
                checkpoint_dir / "strict-clear-best.ckpt",
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=False,
        )

        def schedule(step: int) -> float:
            total = max(1, int(self.trainer.estimated_stepping_batches))
            per_epoch = max(1, total // max(self.max_epochs_config, 1))
            warmup = self.warmup_epochs * per_epoch
            if step < warmup:
                return max((step + 1) / max(warmup, 1), 1e-8)
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
                "interval": "step",
            },
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--train-query-manifests", required=True)
    parser.add_argument("--val-query-manifests", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--exp-name", default="ta_tsel_l12_q_r256")
    parser.add_argument("--output-root", default=os.environ.get(
        "TEMPORAL_EXPERIMENT_ROOT", "/home/ma-user/work/cfzhao/input_prior_full/temporal_experiments"
    ))
    parser.add_argument("--pair-index-root", default=os.environ.get(
        "TEMPORAL_PAIR_ROOT", "/home/ma-user/work/cfzhao/input_prior_full/cache/temporal_pairs"
    ))
    parser.add_argument("--history-lags", type=_parse_history_lags, default=(1, 2))
    parser.add_argument("--use-per-lag-quality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--nominal-fps", type=float, default=30.0)
    parser.add_argument(
        "--use-dino-history", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--dino-attention-heads", type=int, default=4)
    parser.add_argument(
        "--dino-alignment-mode", choices=("aligned", "unwarped"), default="aligned"
    )
    parser.add_argument("--dino-shuffle-seed", type=int, default=521)
    parser.add_argument("--action-margin", type=float, default=0.02)
    parser.add_argument("--anchor-count", type=int, default=512)
    parser.add_argument("--anchor-neighbors", type=int, default=4)
    parser.add_argument("--graph-neighbors", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-open-shards", type=int, default=4)
    parser.add_argument("--cache-shuffle-block-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--resume-save-every-n-epochs", type=int, default=1)
    parser.add_argument("--wandb-project", default="tactile-priors-v2")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-sync-retries", type=int, default=24)
    parser.add_argument("--wandb-sync-interval", type=int, default=300)
    parser.add_argument("--runtime-debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-debug-interval", type=float, default=2.0)
    parser.add_argument("--runtime-debug-flush-steps", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if 1 not in args.history_lags:
        raise ValueError("--history-lags must include lag 1")
    pl.seed_everything(args.seed, workers=True)
    gpu_ids = parse_csv(args.gpus)
    if not gpu_ids:
        raise ValueError("--gpus cannot be empty")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    output_dir = Path(args.output_root).expanduser().resolve(strict=False) / args.exp_name
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pair_root = Path(args.pair_index_root).expanduser().resolve(strict=False)
    train_manifests = parse_csv(args.train_query_manifests)
    val_manifests = parse_csv(args.val_query_manifests)
    train_pair = _pair_index(args.train_cache, train_manifests, pair_root, "train", args)
    val_pair = _pair_index(args.val_cache, val_manifests, pair_root, "val", args)
    train_dataset = TemporalPairDataset(
        args.train_cache, train_pair, history_lags=args.history_lags,
        include_dino_grid=args.use_dino_history,
        max_open_shards=args.max_open_shards,
    )
    val_dataset = TemporalPairDataset(
        args.val_cache, val_pair, history_lags=args.history_lags,
        include_dino_grid=args.use_dino_history,
        max_open_shards=args.max_open_shards,
    )
    expected_sha = file_sha256(args.base_checkpoint)
    for name, dataset in (("train", train_dataset), ("val", val_dataset)):
        if dataset.cache.base_checkpoint_sha256 != expected_sha:
            raise RuntimeError(f"{name} cache was built from a different RGB checkpoint")
    model_config = {
        "palm_vertex_indices": train_dataset.cache.palm_vertex_indices.tolist(),
        "anchor_count": args.anchor_count,
        "anchor_neighbors": args.anchor_neighbors,
        "graph_neighbors": args.graph_neighbors,
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "dropout": args.dropout,
        "history_lags": list(args.history_lags),
        "use_per_lag_quality": args.use_per_lag_quality,
        "nominal_fps": args.nominal_fps,
        "dino_grid_channels": 0,
    }
    if args.use_dino_history:
        train_spec = train_dataset.cache.parts[0].spec_by_name["z_rgb"]
        val_spec = val_dataset.cache.parts[0].spec_by_name["z_rgb"]
        if train_spec.shape != val_spec.shape or len(train_spec.shape) != 3:
            raise RuntimeError(
                f"Train/val DINO grid cache mismatch: {train_spec.shape} vs {val_spec.shape}"
            )
        model_config.update(
            {
                "dino_grid_channels": int(train_spec.shape[0]),
                "dino_grid_size": list(train_spec.shape[1:]),
                "dino_input_resolution": list(train_dataset.cache.input_resolution),
                "dino_attention_heads": args.dino_attention_heads,
                "dino_alignment_mode": args.dino_alignment_mode,
                "dino_shuffle_seed": args.dino_shuffle_seed,
            }
        )
    model = TemporalActionSelectorV2(**model_config)
    data_config = {
        "train_cache": str(Path(args.train_cache).resolve()),
        "val_cache": str(Path(args.val_cache).resolve()),
        "train_cache_sha256": train_dataset.cache.config_sha256,
        "val_cache_sha256": val_dataset.cache.config_sha256,
        "train_pair_index": str(train_pair),
        "val_pair_index": str(val_pair),
        "train_pair_sha256": file_sha256(train_pair),
        "val_pair_sha256": file_sha256(val_pair),
        "history_lags": list(args.history_lags),
        "use_dino_history": bool(args.use_dino_history),
        "dino_alignment_mode": args.dino_alignment_mode,
    }
    resume_path = args.resume_from_checkpoint
    if not resume_path and args.auto_resume and (checkpoint_dir / "resume.ckpt").is_file():
        resume_path = str(checkpoint_dir / "resume.ckpt")
    run_id_file = checkpoint_dir / "wandb_run_id.txt"
    resume_run = ""
    if resume_path:
        header = torch.load(resume_path, map_location="cpu")
        resume_run = str(header.get("wandb_run_id") or "")
    wandb_run_id = resume_run or (
        run_id_file.read_text().strip() if run_id_file.is_file() else uuid.uuid4().hex[:8]
    )
    run_id_file.write_text(wandb_run_id + "\n", encoding="utf-8")
    module = TemporalSelectorTrainingModule(
        model,
        model_config=model_config,
        data_config=data_config,
        base_checkpoint=args.base_checkpoint,
        output_dir=str(output_dir),
        action_margin=args.action_margin,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        wandb_run_id=wandb_run_id,
    )
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    train_sampler = DistributedPairBlockSampler(
        train_dataset, replicas=len(gpu_ids), rank=rank,
        block=args.cache_shuffle_block_size, seed=args.seed,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=len(gpu_ids), rank=rank,
        shuffle=False, drop_last=False,
    )
    common = {
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    train_kwargs = dict(common, batch_size=args.batch_size, sampler=train_sampler,
                        drop_last=True, num_workers=args.num_workers)
    val_kwargs = dict(common, batch_size=args.val_batch_size, sampler=val_sampler,
                      num_workers=args.val_num_workers)
    if args.num_workers:
        train_kwargs.update(prefetch_factor=args.prefetch_factor,
                            persistent_workers=args.persistent_workers)
    if args.val_num_workers:
        val_kwargs.update(prefetch_factor=args.prefetch_factor,
                          persistent_workers=args.persistent_workers)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    run_config = output_dir / "run_config.json"
    run_config.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    logger = CSVLogger(str(output_dir), name="lightning")
    callbacks = [
        AtomicPriorResumeCheckpoint(checkpoint_dir, every_n_epochs=args.resume_save_every_n_epochs)
    ]
    if args.runtime_debug:
        debug = output_dir / "runtime_debug"
        callbacks.extend((
            RuntimeBatchTimingCallback(debug, args.runtime_debug_flush_steps),
            RuntimeSystemMonitorCallback(debug, args.runtime_debug_interval),
        ))
    if not args.no_wandb:
        callbacks.append(DurableWandbEpochCallback(
            queue_dir=checkpoint_dir / "wandb" / "epoch_queue" / wandb_run_id,
            run_id=wandb_run_id,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.exp_name,
            config_path=run_config,
            retries=args.wandb_sync_retries,
            interval=args.wandb_sync_interval,
            local_logger=logger,
        ))
    if rank == 0:
        print(
            f"Temporal Selector V2: train_pairs={len(train_dataset):,}, "
            f"val_pairs={len(val_dataset):,}, lags={args.history_lags}, "
            f"quality={args.use_per_lag_quality}, action_margin={args.action_margin}, "
            f"dino_history={args.use_dino_history}, "
            f"dino_alignment={args.dino_alignment_mode}, "
            "diagnostic only (no pressure residual)",
            flush=True,
        )
    trainer = pl.Trainer(
        accelerator="gpu", devices=len(gpu_ids),
        strategy="ddp" if len(gpu_ids) > 1 else "auto",
        precision=args.precision, max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger, callbacks=callbacks,
        enable_checkpointing=False, use_distributed_sampler=False,
        num_sanity_val_steps=2, log_every_n_steps=20,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_path or None)


if __name__ == "__main__":
    main()
