#!/usr/bin/env python3
"""Train a query-aware lag-1 residual entirely from frozen tactile caches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from hamer_tactile_ft.losses import compute_tactile_loss
from hamer_tactile_ft.process_lifecycle import (
    configure_supervised_process,
    initialize_worker_parent_death_signal,
)
from tactile_input_priors.prior_metrics import PriorMetricAccumulator
from tactile_input_priors.runtime import (
    atomic_torch_save,
    file_sha256,
    load_torch_checkpoint,
    parse_csv,
    tactile_loss_config_from_checkpoint,
)
from tactile_input_priors.temporal_flow import (
    QueryAwareTemporalResidual,
    TemporalPairDataset,
    build_temporal_pair_index,
    history_quality_context,
    pair_context,
    temporal_manifest_key,
    temporal_checkpoint_payload,
)
from tactile_input_priors.train_prior_adapter import (
    AtomicPriorResumeCheckpoint,
    DurableWandbEpochCallback,
    RuntimeBatchTimingCallback,
    RuntimeSystemMonitorCallback,
)


configure_supervised_process()
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


class DistributedPairBlockSampler(Sampler[int]):
    """Shuffle large contiguous pair blocks while preserving mmap locality."""

    def __init__(self, dataset, *, replicas: int, rank: int, block: int, seed: int):
        self.dataset = dataset
        self.replicas = int(replicas)
        self.rank = int(rank)
        self.block = int(block)
        self.seed = int(seed)
        self.epoch = 0
        current = np.asarray(dataset.arrays["current_index"], dtype=np.int64)
        cache = getattr(dataset, "cache", None)
        partition_count = len(getattr(cache, "parts", ()))
        self.partition_aligned = partition_count == self.replicas
        if self.partition_aligned:
            counts = np.bincount(current % self.replicas, minlength=self.replicas)
            self.num_samples = int(counts.min())
            self.candidates = np.flatnonzero(current % self.replicas == self.rank)
        else:
            self.num_samples = int(math.floor(len(dataset) / self.replicas))
            self.candidates = np.arange(len(dataset), dtype=np.int64)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.partition_aligned:
            offset = (self.epoch * self.num_samples) % max(len(self.candidates), 1)
            candidates = np.concatenate(
                (self.candidates[offset:], self.candidates[:offset])
            )[: self.num_samples]
            blocks = [
                candidates[start : min(start + self.block, len(candidates))]
                for start in range(0, len(candidates), self.block)
            ]
            order = []
            for block_index in torch.randperm(len(blocks), generator=generator).tolist():
                values = torch.as_tensor(blocks[block_index], dtype=torch.long)
                permutation = torch.randperm(len(values), generator=generator)
                order.extend(values[permutation].tolist())
            return iter(order)
        blocks = [
            (start, min(start + self.block, len(self.dataset)))
            for start in range(0, len(self.dataset), self.block)
        ]
        order = []
        for block_index in torch.randperm(len(blocks), generator=generator).tolist():
            start, stop = blocks[block_index]
            local = torch.randperm(stop - start, generator=generator).add_(start)
            order.extend(local.tolist())
        usable = self.num_samples * self.replicas
        order = order[:usable]
        return iter(order[self.rank:usable:self.replicas])


def _balanced_mean(values: torch.Tensor, labels: torch.Tensor, classes) -> torch.Tensor:
    terms = [values[labels == value].mean() for value in classes if bool((labels == value).any())]
    return torch.stack(terms).mean() if terms else values.sum() * 0.0


def _parse_history_lags(value: str) -> tuple[int, ...]:
    lags = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not lags or len(set(lags)) != len(lags) or any(lag <= 0 for lag in lags):
        raise argparse.ArgumentTypeError("history lags must be unique positive integers")
    return lags


class TemporalFlowTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: QueryAwareTemporalResidual,
        *,
        model_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        loss_config,
        base_checkpoint: str,
        output_dir: str,
        learning_rate: float,
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        transition_loss_weight: float,
        history_gate_loss_weight: float,
        base_guard_weight: float,
        delta_l1_weight: float,
        transition_threshold: float,
        history_margin: float,
        wandb_run_id: str,
    ):
        super().__init__()
        self.model = model
        self.model_config = dict(model_config)
        self.data_config = dict(data_config)
        self.loss_config = loss_config
        self.base_checkpoint = str(base_checkpoint)
        self.output_dir = Path(output_dir)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs_config = int(max_epochs)
        self.transition_loss_weight = float(transition_loss_weight)
        self.history_gate_loss_weight = float(history_gate_loss_weight)
        self.base_guard_weight = float(base_guard_weight)
        self.delta_l1_weight = float(delta_l1_weight)
        self.transition_threshold = float(transition_threshold)
        self.history_margin = float(history_margin)
        self.wandb_run_id = str(wandb_run_id)
        self.best_loss = math.inf
        self.best_gain = -math.inf
        self._loading_resume = False
        self._train_sum = torch.tensor(0.0, dtype=torch.float64)
        self._train_count = torch.tensor(0.0, dtype=torch.float64)
        self._val_sum = torch.tensor(0.0, dtype=torch.float64)
        self._val_count = torch.tensor(0.0, dtype=torch.float64)
        self.train_loss_epoch_global = math.nan
        self.fused_metrics = PriorMetricAccumulator()
        self.base_metrics = PriorMetricAccumulator()

    def _resume_contract(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "data_config": self.data_config,
            "base_checkpoint_sha256": file_sha256(self.base_checkpoint),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs_config,
            "transition_loss_weight": self.transition_loss_weight,
            "history_gate_loss_weight": self.history_gate_loss_weight,
            "base_guard_weight": self.base_guard_weight,
            "delta_l1_weight": self.delta_l1_weight,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"] = {
            name: value for name, value in checkpoint.get("state_dict", {}).items()
            if name.startswith("model.")
        }
        checkpoint["format"] = "tactile_temporal_flow_resume_v1"
        checkpoint["resume_contract"] = self._resume_contract()
        checkpoint["wandb_run_id"] = self.wandb_run_id
        checkpoint["best_loss"] = self.best_loss
        checkpoint["best_gain"] = self.best_gain

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != "tactile_temporal_flow_resume_v1":
            return
        if checkpoint.get("resume_contract") != self._resume_contract():
            raise RuntimeError("Temporal-flow resume configuration mismatch")
        self.wandb_run_id = str(checkpoint.get("wandb_run_id") or self.wandb_run_id)
        self.best_loss = float(checkpoint.get("best_loss", math.inf))
        self.best_gain = float(checkpoint.get("best_gain", -math.inf))
        self._loading_resume = True

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self._loading_resume:
            return super().load_state_dict(state_dict, strict=strict)
        result = super().load_state_dict(state_dict, strict=False)
        self._loading_resume = False
        missing = [name for name in result.missing_keys if name.startswith("model.")]
        if missing or result.unexpected_keys:
            raise RuntimeError(
                f"Temporal resume state mismatch: missing={missing}, "
                f"unexpected={result.unexpected_keys}"
            )
        return result

    def _forward_loss(self, batch):
        if self.model.architecture == "signed_additive":
            output = self.model(
                batch["current_logits"],
                batch["history_logits"],
                pair_context(batch),
                batch["history_available"],
                history_quality_context(
                    batch,
                    self.model.history_lags,
                    nominal_fps=self.model.nominal_fps,
                ) if self.model.use_per_lag_quality else None,
            )
        else:
            output = self.model(
                batch["current_logits"], batch["previous_logits"], pair_context(batch)
            )
        ones = torch.ones_like(batch["tactile_signal"])
        tactile_loss, components = compute_tactile_loss(
            pred=output["pred_tactile"],
            logits=output["pred_logits"],
            target=batch["tactile_signal"],
            palm_mask=ones,
            valid_mask=batch["has_tactile"],
            dataset_batch="TouchAnything",
            config=self.loss_config,
            current_epoch=self.current_epoch,
            ramp_override=1.0,
        )
        anchors = output["anchor_local_indices"]
        target_anchor = batch["tactile_signal"][:, anchors].float()
        if self.model.architecture == "signed_additive":
            history_target_anchor = batch["history_tactile_signal"][
                :, :, anchors
            ].float().transpose(1, 2)
            history_available = batch["history_available"][:, None, :] > 0.5
            target_delta = target_anchor[:, :, None] - history_target_anchor
        else:
            history_target_anchor = batch["previous_tactile_signal"][
                :, anchors
            ].float()
            history_available = torch.ones_like(
                history_target_anchor, dtype=torch.bool
            )
            target_delta = target_anchor - history_target_anchor
        transition_target = torch.zeros_like(target_delta, dtype=torch.long)
        transition_target[target_delta > self.transition_threshold] = 1
        transition_target[target_delta < -self.transition_threshold] = 2
        transition_element = F.cross_entropy(
            output["anchor_transition_logits"].reshape(-1, 3),
            transition_target.reshape(-1),
            reduction="none",
        )
        transition_mask = history_available.expand_as(transition_target).reshape(-1)
        transition_loss = _balanced_mean(
            transition_element[transition_mask],
            transition_target.reshape(-1)[transition_mask],
            (0, 1, 2),
        )
        current_error = (
            output["base_pred_tactile"][:, anchors] - target_anchor
        ).abs()
        if self.model.architecture == "signed_additive":
            history_prediction_anchor = torch.sigmoid(
                batch["history_logits"][:, :, anchors].float()
            ).transpose(1, 2)
            previous_error = (
                history_prediction_anchor - target_anchor[:, :, None]
            ).abs()
            advantage = current_error[:, :, None] - previous_error
        else:
            previous_error = (
                torch.sigmoid(batch["previous_logits"][:, anchors].float())
                - target_anchor
            ).abs()
            advantage = current_error - previous_error
        clear = (advantage.abs() >= self.history_margin) & history_available
        history_target = advantage > 0.0
        gate_logits = output["anchor_history_gate_logits"]
        gate_element = F.binary_cross_entropy_with_logits(
            gate_logits, history_target.float(), reduction="none"
        )
        gate_loss = _balanced_mean(
            gate_element[clear], history_target[clear], (False, True)
        ) if bool(clear.any()) else gate_element.sum() * 0.0
        fused_frame_error = (
            output["pred_tactile"] - batch["tactile_signal"].float()
        ).square().mean(dim=1)
        base_frame_error = (
            output["base_pred_tactile"] - batch["tactile_signal"].float()
        ).square().mean(dim=1)
        base_guard = F.relu(fused_frame_error - base_frame_error.detach()).mean()
        delta_l1 = output["bounded_logit_delta"].abs().mean()
        pressure_objective = (
            10.0 * tactile_loss
            + self.base_guard_weight * base_guard
            + self.delta_l1_weight * delta_l1
        )
        loss = (
            pressure_objective
            + self.transition_loss_weight * transition_loss
            + self.history_gate_loss_weight * gate_loss
        )
        checkpoint_loss = (
            pressure_objective
            if self.model.architecture == "signed_additive"
            else loss
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite temporal loss at epoch={self.current_epoch}, step={self.global_step}"
            )
        auxiliary = {
            "tactile": tactile_loss,
            "transition": transition_loss,
            "history_gate": gate_loss,
            "base_guard": base_guard,
            "delta_l1": delta_l1,
            "history_better_fraction": history_target[
                history_available.expand_as(history_target)
            ].float().mean(),
            "transition_stable_fraction": (transition_target == 0)[
                history_available.expand_as(transition_target)
            ].float().mean(),
            "history_alpha_abs_mean": output["vertex_history_alpha"].abs().mean(),
            "logit_delta_rms": output["bounded_logit_delta"].square().mean().sqrt(),
            "optimization_loss": loss.detach(),
            "pressure_objective": pressure_objective.detach(),
        }
        if "effective_global_gate" in output:
            auxiliary["effective_global_gate"] = output["effective_global_gate"]
        if "vertex_history_alpha_per_lag" in output:
            coefficients = output["vertex_history_alpha_per_lag"]
            coefficient_mask = batch["history_available"][:, :, None].to(
                device=coefficients.device, dtype=coefficients.dtype
            )
            coefficient_count = (
                coefficient_mask.sum() * coefficients.shape[-1]
            ).clamp_min(1.0)
            auxiliary.update(
                {
                    "signed_alpha_positive_fraction": (
                        (coefficients > 0).to(coefficients) * coefficient_mask
                    ).sum()
                    / coefficient_count,
                    "signed_alpha_negative_fraction": (
                        (coefficients < 0).to(coefficients) * coefficient_mask
                    ).sum()
                    / coefficient_count,
                    "signed_alpha_saturation_fraction": (
                        (
                            coefficients.abs()
                            > 0.95 * self.model.max_history_alpha
                        ).to(coefficients)
                        * coefficient_mask
                    ).sum()
                    / coefficient_count,
                }
            )
            for column, lag in enumerate(self.model.history_lags):
                lag_mask = coefficient_mask[:, column]
                lag_count = (lag_mask.sum() * coefficients.shape[-1]).clamp_min(1.0)
                auxiliary[f"lag{lag}_alpha_abs_mean"] = (
                    coefficients[:, column].abs() * lag_mask
                ).sum() / lag_count
        return output, loss, checkpoint_loss, auxiliary, components

    def training_step(self, batch, batch_idx):
        del batch_idx
        output, loss, _, auxiliary, components = self._forward_loss(batch)
        batch_size = int(batch["tactile_signal"].shape[0])
        self._train_sum.add_(loss.detach().double() * batch_size)
        self._train_count.add_(batch_size)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        for name, value in auxiliary.items():
            self.log(f"train/{name}", value, on_epoch=True, batch_size=batch_size)
        for name in (
            "diagnostics_pressure_weight_mean",
            "diagnostics_weighted_to_direct_loss_ratio",
        ):
            if name in components:
                self.log(
                    f"train/{name.removeprefix('diagnostics_')}",
                    components[name],
                    on_epoch=True,
                    batch_size=batch_size,
                )
        return loss

    def on_train_epoch_start(self):
        sampler = getattr(self.trainer.train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(self.current_epoch))
        self._train_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._train_count = torch.zeros((), device=self.device, dtype=torch.float64)

    def _sync_average(self, total, count) -> float:
        values = torch.stack((total, count)).to(self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        return float((values[0] / values[1].clamp_min(1.0)).cpu())

    def on_train_epoch_end(self):
        self.train_loss_epoch_global = self._sync_average(self._train_sum, self._train_count)
        self.log("train/loss_epoch_global", self.train_loss_epoch_global, sync_dist=False)

    def on_validation_epoch_start(self):
        self.fused_metrics.reset()
        self.base_metrics.reset()
        self._val_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_count = torch.zeros((), device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        del batch_idx
        output, loss, checkpoint_loss, auxiliary, _ = self._forward_loss(batch)
        batch_size = int(batch["tactile_signal"].shape[0])
        self._val_sum.add_(checkpoint_loss.detach().double() * batch_size)
        self._val_count.add_(batch_size)
        palm = torch.ones_like(batch["tactile_signal"])
        common = (batch["tactile_signal"], palm, batch["has_tactile"])
        self.fused_metrics.update(output["pred_tactile"], *common)
        self.base_metrics.update(output["base_pred_tactile"], *common)
        for name, value in auxiliary.items():
            self.log(f"val/{name}", value, on_epoch=True, sync_dist=True, batch_size=batch_size)
        return loss

    def _compact(self, monitor: str, score: float):
        return temporal_checkpoint_payload(
            self.model,
            model_config=self.model_config,
            data_config=self.data_config,
            base_checkpoint=self.base_checkpoint,
            epoch=self.current_epoch,
            global_step=self.global_step,
            monitor=monitor,
            score=score,
        )

    def on_validation_epoch_end(self):
        self.fused_metrics.synchronize(self.device)
        self.base_metrics.synchronize(self.device)
        fused = self.fused_metrics.summary()
        base = self.base_metrics.summary()
        val_loss = self._sync_average(self._val_sum, self._val_count)
        gain = (fused["contact_iou"] - base["contact_iou"]) + (
            fused["volumetric_iou"] - base["volumetric_iou"]
        )
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=False)
        self.log("val/temporal_gain", gain, prog_bar=True, sync_dist=False)
        for prefix, values in (("fused", fused), ("base", base)):
            for name, value in values.items():
                self.log(f"val/{prefix}_{name}", value, sync_dist=False)
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        row = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "train_loss_epoch_global": self.train_loss_epoch_global,
            "val_loss": val_loss,
            "temporal_gain": gain,
            **{f"fused_{key}": value for key, value in fused.items()},
            **{f"base_{key}": value for key, value in base.items()},
        }
        path = self.output_dir / "val_metrics.csv"
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        checkpoint_dir = self.output_dir / "checkpoints"
        atomic_torch_save(self._compact("last", gain), checkpoint_dir / "last.ckpt")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            atomic_torch_save(
                self._compact("val/loss", val_loss), checkpoint_dir / "best_loss.ckpt"
            )
        if gain > self.best_gain:
            self.best_gain = gain
            atomic_torch_save(
                self._compact("val/temporal_gain", gain),
                checkpoint_dir / "temporal-best.ckpt",
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


def _pair_index(cache_root: str, manifests: tuple[str, ...], root: Path, split: str, args) -> Path:
    from tactile_input_priors.temporal_flow import PartitionedPalmCache

    cache = PartitionedPalmCache(cache_root, max_open_shards=args.max_open_shards)
    key = temporal_manifest_key(manifests)
    path = root / f"{split}-{cache.config_sha256[:12]}-{key}.npz"
    lock = path.with_suffix(".lock")
    acquired = False
    for _ in range(7200):
        try:
            lock.mkdir(parents=True)
            acquired = True
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 21600:
                    lock.rmdir()
                    continue
            except (FileNotFoundError, OSError):
                pass
            time.sleep(1.0)
    if not acquired:
        raise TimeoutError(f"Timed out waiting for temporal pair index lock: {lock}")
    try:
        return build_temporal_pair_index(cache, manifests, path, seed=args.seed)
    finally:
        lock.rmdir()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--train-query-manifests", required=True)
    parser.add_argument("--val-query-manifests", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--exp-name", default="ta_tflow_r256")
    parser.add_argument("--output-root", default=os.environ.get(
        "TEMPORAL_EXPERIMENT_ROOT", "/home/ma-user/work/cfzhao/input_prior_full/temporal_experiments"
    ))
    parser.add_argument("--pair-index-root", default=os.environ.get(
        "TEMPORAL_PAIR_ROOT", "/home/ma-user/work/cfzhao/input_prior_full/cache/temporal_pairs"
    ))
    parser.add_argument("--anchor-count", type=int, default=512)
    parser.add_argument("--anchor-neighbors", type=int, default=4)
    parser.add_argument("--graph-neighbors", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--graph-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--temporal-architecture",
        choices=("legacy_product", "signed_additive"),
        default="legacy_product",
    )
    parser.add_argument("--history-lags", type=_parse_history_lags, default=(1,))
    parser.add_argument(
        "--use-per-lag-quality", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--nominal-fps", type=float, default=30.0)
    parser.add_argument("--max-history-alpha", type=float, default=0.75)
    parser.add_argument("--max-logit-delta", type=float, default=0.50)
    parser.add_argument("--transition-loss-weight", type=float, default=0.10)
    parser.add_argument("--history-gate-loss-weight", type=float, default=0.10)
    parser.add_argument("--base-guard-weight", type=float, default=0.10)
    parser.add_argument("--delta-l1-weight", type=float, default=0.001)
    parser.add_argument("--transition-threshold", type=float, default=0.02)
    parser.add_argument("--history-margin", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-num-workers", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the small cache-only worker pool alive across epochs.",
    )
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
    if args.temporal_architecture == "legacy_product" and args.history_lags != (1,):
        raise ValueError("legacy_product supports only --history-lags 1")
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
        args.train_cache,
        train_pair,
        history_lags=args.history_lags,
        max_open_shards=args.max_open_shards,
    )
    val_dataset = TemporalPairDataset(
        args.val_cache,
        val_pair,
        history_lags=args.history_lags,
        max_open_shards=args.max_open_shards,
    )
    expected_sha = file_sha256(args.base_checkpoint)
    for name, dataset in (("train", train_dataset), ("val", val_dataset)):
        if dataset.cache.base_checkpoint_sha256 != expected_sha:
            raise RuntimeError(f"{name} cache was built from a different baseline checkpoint")
    model_config = {
        "palm_vertex_indices": train_dataset.cache.palm_vertex_indices.tolist(),
        "anchor_count": args.anchor_count,
        "anchor_neighbors": args.anchor_neighbors,
        "graph_neighbors": args.graph_neighbors,
        "hidden_dim": args.hidden_dim,
        "graph_layers": args.graph_layers,
        "dropout": args.dropout,
        "max_history_alpha": args.max_history_alpha,
        "max_logit_delta": args.max_logit_delta,
        "architecture": args.temporal_architecture,
        "history_lags": list(args.history_lags),
        "use_per_lag_quality": args.use_per_lag_quality,
        "nominal_fps": args.nominal_fps,
    }
    model = QueryAwareTemporalResidual(**model_config)
    base_payload = load_torch_checkpoint(args.base_checkpoint)
    loss_config = tactile_loss_config_from_checkpoint(base_payload, full_ramp=True)
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
        "use_per_lag_quality": args.use_per_lag_quality,
        "nominal_fps": args.nominal_fps,
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
    module = TemporalFlowTrainingModule(
        model,
        model_config=model_config,
        data_config=data_config,
        loss_config=loss_config,
        base_checkpoint=args.base_checkpoint,
        output_dir=str(output_dir),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        transition_loss_weight=args.transition_loss_weight,
        history_gate_loss_weight=args.history_gate_loss_weight,
        base_guard_weight=args.base_guard_weight,
        delta_l1_weight=args.delta_l1_weight,
        transition_threshold=args.transition_threshold,
        history_margin=args.history_margin,
        wandb_run_id=wandb_run_id,
    )
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    train_sampler = DistributedPairBlockSampler(
        train_dataset,
        replicas=len(gpu_ids),
        rank=rank,
        block=args.cache_shuffle_block_size,
        seed=args.seed,
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=len(gpu_ids), rank=rank, shuffle=False, drop_last=False
    )
    common = {
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    train_kwargs = dict(
        common,
        batch_size=args.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_kwargs = dict(
        common,
        batch_size=args.val_batch_size,
        sampler=val_sampler,
        num_workers=args.val_num_workers,
    )
    if args.num_workers:
        train_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
    if args.val_num_workers:
        val_kwargs.update(
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)
    run_config = output_dir / "run_config.json"
    run_config.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    logger = CSVLogger(str(output_dir), name="lightning")
    callbacks = [
        AtomicPriorResumeCheckpoint(
            checkpoint_dir, every_n_epochs=args.resume_save_every_n_epochs
        )
    ]
    if args.runtime_debug:
        debug = output_dir / "runtime_debug"
        callbacks.extend(
            (
                RuntimeBatchTimingCallback(debug, args.runtime_debug_flush_steps),
                RuntimeSystemMonitorCallback(debug, args.runtime_debug_interval),
            )
        )
    if not args.no_wandb:
        callbacks.append(
            DurableWandbEpochCallback(
                queue_dir=checkpoint_dir / "wandb" / "epoch_queue" / wandb_run_id,
                run_id=wandb_run_id,
                project=args.wandb_project,
                entity=args.wandb_entity,
                run_name=args.exp_name,
                config_path=run_config,
                retries=args.wandb_sync_retries,
                interval=args.wandb_sync_interval,
                local_logger=logger,
            )
        )
    if rank == 0:
        print(
            f"Temporal cache-only training: train_pairs={len(train_dataset):,}, "
            f"val_pairs={len(val_dataset):,}, batch/rank={args.batch_size}, "
            f"val_batch/rank={args.val_batch_size}, workers={args.num_workers}/"
            f"{args.val_num_workers}, cache_partition_aligned="
            f"{train_sampler.partition_aligned}, no DINO/HDF5 in epoch loop",
            flush=True,
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=len(gpu_ids),
        strategy="ddp" if len(gpu_ids) > 1 else "auto",
        precision=args.precision,
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        use_distributed_sampler=False,
        num_sanity_val_steps=2,
        log_every_n_steps=20,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_path or None)


if __name__ == "__main__":
    main()
