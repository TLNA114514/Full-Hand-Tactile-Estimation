#!/usr/bin/env python3
"""Train Depth/VLM adapters against a frozen Binary Grid contact selector."""

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

import cv2
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

cv2.setNumThreads(0)
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from hamer_tactile_ft.process_lifecycle import initialize_worker_parent_death_signal
from tactile_input_priors.runtime import (
    atomic_torch_save,
    build_prior_selector_model,
    file_sha256,
    parse_csv,
    selector_prior_checkpoint_payload,
    selector_prior_config_from_args,
)
from tactile_input_priors.selector_prior_metrics import (
    ContactMetricAccumulator,
    SequenceContactAPAccumulator,
)
from tactile_input_priors.train_prior_adapter import (
    AtomicPriorResumeCheckpoint,
    DurableWandbEpochCallback,
    RuntimeBatchTimingCallback,
    RuntimeSystemMonitorCallback,
    _dataset_from_args,
    add_data_arguments,
)


def balanced_clear_contact_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    palm_mask: torch.Tensor,
    has_tactile: torch.Tensor,
    *,
    no_contact_max: float,
    contact_min: float,
) -> torch.Tensor:
    logits = logits.float()
    target = target.float()
    valid = (palm_mask.float() > 0.5) & (has_tactile.reshape(-1) > 0.5)[:, None]
    clear = (target <= float(no_contact_max)) | (target >= float(contact_min))
    eligible = valid & clear
    labels = target >= float(contact_min)
    element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels.to(logits), reduction="none"
    )
    positive = eligible & labels
    negative = eligible & ~labels
    terms = []
    if bool(positive.any()):
        terms.append(element[positive].mean())
    if bool(negative.any()):
        terms.append(element[negative].mean())
    if not terms:
        raise RuntimeError("Contact batch contains no clear eligible vertices")
    return torch.stack(terms).mean()


def balanced_false_high_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    base_pressure: torch.Tensor,
    palm_mask: torch.Tensor,
    has_tactile: torch.Tensor,
    *,
    no_contact_max: float,
    contact_min: float,
) -> torch.Tensor:
    """Classify false-high candidates without turning the score into pressure."""

    logits = logits.float()
    target = target.float()
    valid = (palm_mask.float() > 0.5) & (has_tactile.reshape(-1) > 0.5)[:, None]
    clear = (target <= float(no_contact_max)) | (target >= float(contact_min))
    candidate = valid & clear & (base_pressure.detach().float() >= float(contact_min))
    labels = target <= float(no_contact_max)
    element = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels.to(logits), reduction="none"
    )
    terms = []
    for class_mask in (candidate & labels, candidate & ~labels):
        if bool(class_mask.any()):
            terms.append(element[class_mask].mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


class DistributedBlockShuffleSampler(Sampler[int]):
    """DDP sampler that shuffles mmap-local blocks instead of the whole cache."""

    def __init__(
        self,
        dataset,
        *,
        num_replicas: int,
        rank: int,
        block_size: int,
        seed: int,
    ):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("Invalid distributed sampler rank/world size")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        self.num_samples = int(math.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        blocks = [
            (start, min(start + self.block_size, len(self.dataset)))
            for start in range(0, len(self.dataset), self.block_size)
        ]
        block_order = torch.randperm(len(blocks), generator=generator).tolist()
        ordered = []
        for block_index in block_order:
            start, stop = blocks[block_index]
            local = torch.randperm(stop - start, generator=generator).add_(start)
            ordered.extend(local.tolist())
        if len(ordered) < self.total_size:
            missing = self.total_size - len(ordered)
            repeats = int(math.ceil(missing / max(len(ordered), 1)))
            ordered.extend((ordered * repeats)[:missing])
        start = self.rank * self.num_samples
        return iter(ordered[start : start + self.num_samples])


class PriorSelectorTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model,
        *,
        adapter_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        selector_checkpoint: str,
        dino_weights: str,
        output_dir: str,
        learning_rate: float,
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        counterfactual_control: str,
        control_identity_weight: float,
        feature_budget_penalty_weight: float,
        false_high_loss_weight: float,
        false_high_score_source: str,
        paired_controls: bool,
        validation_reference_metrics: bool,
        validation_sequence_metrics: bool,
        wandb_run_id: str,
    ):
        super().__init__()
        self.model = model
        self.adapter_config = dict(adapter_config)
        self.data_config = dict(data_config)
        self.selector_checkpoint = str(selector_checkpoint)
        self.dino_weights = str(dino_weights)
        self.output_dir = Path(output_dir)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs_config = int(max_epochs)
        self.counterfactual_control = str(counterfactual_control)
        self.control_identity_weight = float(control_identity_weight)
        self.feature_budget_penalty_weight = float(feature_budget_penalty_weight)
        self.false_high_loss_weight = float(false_high_loss_weight)
        self.false_high_score_source = str(false_high_score_source)
        self.paired_controls = bool(paired_controls)
        self.validation_reference_metrics = bool(validation_reference_metrics)
        self.validation_sequence_metrics = bool(validation_sequence_metrics)
        self.wandb_run_id = str(wandb_run_id)
        self.no_contact_max = float(model.base_model.support_selector_no_contact_max)
        self.contact_min = float(model.base_model.support_selector_contact_min)
        self.best_information_score = -math.inf
        self.best_val_loss = math.inf
        self.best_false_high_ap = -math.inf
        self.best_contact_ap = -math.inf
        self._loading_resume = False
        self._train_sum = torch.tensor(0.0, dtype=torch.float64)
        self._train_count = torch.tensor(0.0, dtype=torch.float64)
        self._val_sum = torch.tensor(0.0, dtype=torch.float64)
        self._val_count = torch.tensor(0.0, dtype=torch.float64)
        self.train_loss_epoch_global = math.nan
        self.fused_metrics = ContactMetricAccumulator(
            no_contact_max=self.no_contact_max,
            contact_min=self.contact_min,
            false_high_topk=(),
        )
        self.base_metrics = ContactMetricAccumulator(
            no_contact_max=self.no_contact_max,
            contact_min=self.contact_min,
            false_high_topk=(),
        )
        self.control_metrics = ContactMetricAccumulator(
            no_contact_max=self.no_contact_max,
            contact_min=self.contact_min,
            false_high_topk=(),
        )
        self.fused_sequence_metrics = SequenceContactAPAccumulator(
            no_contact_max=self.no_contact_max, contact_min=self.contact_min
        )
        self.base_sequence_metrics = SequenceContactAPAccumulator(
            no_contact_max=self.no_contact_max, contact_min=self.contact_min
        )
        self.control_sequence_metrics = SequenceContactAPAccumulator(
            no_contact_max=self.no_contact_max, contact_min=self.contact_min
        )
        self.save_hyperparameters(ignore=("model",))

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.base_model.eval()
        return self

    def _contract(self) -> dict[str, Any]:
        return {
            "adapter_config": self.adapter_config,
            "data_config": self.data_config,
            "selector_checkpoint": self.selector_checkpoint,
            "dino_weights": self.dino_weights,
            "counterfactual_control": self.counterfactual_control,
            "control_identity_weight": self.control_identity_weight,
            "feature_budget_penalty_weight": self.feature_budget_penalty_weight,
            "false_high_loss_weight": self.false_high_loss_weight,
            "false_high_score_source": self.false_high_score_source,
            "paired_controls": self.paired_controls,
            "validation_reference_metrics": self.validation_reference_metrics,
            "validation_sequence_metrics": self.validation_sequence_metrics,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs_config,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["state_dict"] = {
            name: value
            for name, value in checkpoint.get("state_dict", {}).items()
            if name.startswith("model.prior_adapter.")
        }
        checkpoint["format"] = "tactile_prior_selector_resume_v1"
        checkpoint["resume_contract"] = self._contract()
        checkpoint["best_information_score"] = self.best_information_score
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["best_false_high_ap"] = self.best_false_high_ap
        checkpoint["best_contact_ap"] = self.best_contact_ap
        checkpoint["wandb_run_id"] = self.wandb_run_id

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != "tactile_prior_selector_resume_v1":
            return
        stored = checkpoint.get("resume_contract")
        current = self._contract()
        if not isinstance(stored, Mapping):
            raise RuntimeError(
                "Selector-prior resume checkpoint has no valid resume_contract"
            )
        stored = dict(stored)
        stored.setdefault(
            "false_high_score_source",
            "head" if float(stored.get("false_high_loss_weight", 0.0)) > 0 else "contact",
        )
        stored.setdefault("paired_controls", True)
        stored.setdefault("validation_reference_metrics", True)
        stored.setdefault("validation_sequence_metrics", True)
        mismatches = {
            key: {
                "checkpoint": stored.get(key),
                "current": current.get(key),
            }
            for key in sorted(set(stored) | set(current))
            if stored.get(key) != current.get(key)
        }
        if mismatches:
            data_mismatch = mismatches.get("data_config", {})
            checkpoint_data = data_mismatch.get("checkpoint", {})
            current_data = data_mismatch.get("current", {})
            cache_mode_changed = (
                isinstance(checkpoint_data, Mapping)
                and isinstance(current_data, Mapping)
                and bool(checkpoint_data.get("cache_only", False))
                != bool(current_data.get("cache_only", False))
            )
            hint = (
                " The checkpoint and current run use different online/cache-only "
                "data flows; start a new experiment name instead of resuming."
                if cache_mode_changed
                else ""
            )
            raise RuntimeError(
                "Selector-prior resume configuration mismatch: "
                + json.dumps(mismatches, sort_keys=True)
                + hint
            )
        self.best_information_score = float(
            checkpoint.get("best_information_score", -math.inf)
        )
        self.best_val_loss = float(checkpoint.get("best_val_loss", math.inf))
        self.best_false_high_ap = float(
            checkpoint.get("best_false_high_ap", -math.inf)
        )
        self.best_contact_ap = float(checkpoint.get("best_contact_ap", -math.inf))
        checkpoint_run = str(checkpoint.get("wandb_run_id", "") or "")
        if checkpoint_run:
            self.wandb_run_id = checkpoint_run
        self._loading_resume = True

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self._loading_resume:
            return super().load_state_dict(state_dict, strict=strict)
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = [
            key for key in incompatible.missing_keys
            if key.startswith("model.prior_adapter.")
        ]
        self._loading_resume = False
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Selector-prior resume mismatch: missing={missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return incompatible

    def _forward_and_loss(self, batch):
        if self.paired_controls:
            output = self.model.forward_paired(
                batch, control=self.counterfactual_control
            )
        else:
            output = self.model.forward_step(batch)
            zero = output["fused_contact_logits"].sum() * 0.0
            output["control_identity_loss"] = zero
            output["control_prior_aux_losses"] = {}
        contact_loss = balanced_clear_contact_loss(
            output["fused_contact_logits"],
            batch["tactile_signal"],
            batch["palm_mask"],
            batch["has_tactile"],
            no_contact_max=self.no_contact_max,
            contact_min=self.contact_min,
        )
        if self.false_high_loss_weight > 0.0:
            false_high_loss = balanced_false_high_loss(
                output["false_high_logits"],
                batch["tactile_signal"],
                output["base_pred_tactile"],
                batch["palm_mask"],
                batch["has_tactile"],
                no_contact_max=self.no_contact_max,
                contact_min=self.contact_min,
            )
        else:
            false_high_loss = output["fused_contact_logits"].sum() * 0.0
        control_identity = output["control_identity_loss"]
        budget = contact_loss.new_zeros(())
        for value in output.get("prior_aux_losses", {}).values():
            budget = budget + value
        for value in output.get("control_prior_aux_losses", {}).values():
            budget = budget + value
        loss = (
            contact_loss
            + self.false_high_loss_weight * false_high_loss
            + self.control_identity_weight * control_identity
            + self.feature_budget_penalty_weight * budget
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite selector-prior loss at epoch={self.current_epoch}, "
                f"step={self.global_step}"
            )
        if not torch.equal(output["pred_logits"], output["base_pressure_logits"]):
            raise RuntimeError("Selector prior changed frozen pressure logits")
        return output, loss, contact_loss, false_high_loss, control_identity, budget

    def training_step(self, batch, batch_idx):
        del batch_idx
        output, loss, contact_loss, false_high_loss, identity, budget = self._forward_and_loss(batch)
        batch_size = int(batch["tactile_signal"].shape[0])
        self._train_sum.add_(loss.detach().double() * batch_size)
        self._train_count.add_(batch_size)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/contact_loss", contact_loss, on_epoch=True, batch_size=batch_size)
        self.log("train/false_high_loss", false_high_loss, on_epoch=True, batch_size=batch_size)
        self.log("train/control_identity", identity, on_epoch=True, batch_size=batch_size)
        self.log("train/feature_budget", budget, on_epoch=True, batch_size=batch_size)
        for name, value in output["prior_diagnostics"].items():
            self.log(f"train/prior_{name}", value, on_epoch=True, batch_size=batch_size)
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
        self.train_loss_epoch_global = self._sync_average(
            self._train_sum, self._train_count
        )
        self.log("train/loss_epoch_global", self.train_loss_epoch_global, sync_dist=False)

    def on_validation_epoch_start(self):
        self.fused_metrics.reset()
        if self.validation_sequence_metrics:
            self.fused_sequence_metrics.reset()
        if self.validation_reference_metrics:
            self.base_metrics.reset()
            if self.validation_sequence_metrics:
                self.base_sequence_metrics.reset()
            if self.paired_controls:
                self.control_metrics.reset()
                if self.validation_sequence_metrics:
                    self.control_sequence_metrics.reset()
        self._val_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_count = torch.zeros((), device=self.device, dtype=torch.float64)

    def _metric_false_high_logits(self, output, key="false_high_logits"):
        if self.false_high_score_source == "contact":
            return None
        return output[key]

    def validation_step(self, batch, batch_idx):
        del batch_idx
        output, loss, _, _, _, _ = self._forward_and_loss(batch)
        batch_size = int(batch["tactile_signal"].shape[0])
        self._val_sum.add_(loss.detach().double() * batch_size)
        self._val_count.add_(batch_size)
        common = (
            batch["tactile_signal"],
            batch["palm_mask"],
            batch["has_tactile"],
        )
        pressure = output["base_pred_tactile"]
        self.fused_metrics.update(
            output["fused_contact_logits"],
            *common,
            base_pressure=pressure,
            false_high_logits=self._metric_false_high_logits(output),
        )
        if self.validation_reference_metrics:
            self.base_metrics.update(
                output["base_contact_logits"], *common, base_pressure=pressure
            )
            if self.paired_controls:
                self.control_metrics.update(
                    output["control_contact_logits"],
                    *common,
                    base_pressure=pressure,
                    false_high_logits=self._metric_false_high_logits(
                        output, "control_false_high_logits"
                    ),
                )
        sequence_keys = [str(value) for value in batch.get("sequence_key", [])]
        if sequence_keys and self.validation_sequence_metrics:
            self.fused_sequence_metrics.update(
                output["fused_contact_logits"], *common, sequence_keys=sequence_keys
            )
            if self.validation_reference_metrics:
                self.base_sequence_metrics.update(
                    output["base_contact_logits"], *common, sequence_keys=sequence_keys
                )
                if self.paired_controls:
                    self.control_sequence_metrics.update(
                        output["control_contact_logits"],
                        *common,
                        sequence_keys=sequence_keys,
                    )
        return loss

    def _compact(self, monitor: str, score: float):
        payload = selector_prior_checkpoint_payload(
            self.model,
            adapter_config=self.adapter_config,
            selector_checkpoint=self.selector_checkpoint,
            dino_weights=self.dino_weights,
            data_config=self.data_config,
            epoch=self.current_epoch,
            global_step=self.global_step,
            monitor=monitor,
            score=score,
        )
        payload["training_config"] = {
            "counterfactual_control": self.counterfactual_control,
            "control_identity_weight": self.control_identity_weight,
            "feature_budget_penalty_weight": self.feature_budget_penalty_weight,
            "false_high_loss_weight": self.false_high_loss_weight,
            "false_high_score_source": self.false_high_score_source,
            "paired_controls": self.paired_controls,
            "validation_reference_metrics": self.validation_reference_metrics,
            "validation_sequence_metrics": self.validation_sequence_metrics,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs_config,
        }
        return payload

    def on_validation_epoch_end(self):
        self.fused_metrics.synchronize(self.device)
        if self.validation_sequence_metrics:
            self.fused_sequence_metrics.synchronize()
        if self.validation_reference_metrics:
            self.base_metrics.synchronize(self.device)
            if self.validation_sequence_metrics:
                self.base_sequence_metrics.synchronize()
            if self.paired_controls:
                self.control_metrics.synchronize(self.device)
                if self.validation_sequence_metrics:
                    self.control_sequence_metrics.synchronize()
        fused = self.fused_metrics.summary()
        if self.validation_sequence_metrics:
            fused.update(self.fused_sequence_metrics.summary())
        base = {}
        control = {}
        if self.validation_reference_metrics:
            base = self.base_metrics.summary()
            if self.validation_sequence_metrics:
                base.update(self.base_sequence_metrics.summary())
            if self.paired_controls:
                control = self.control_metrics.summary()
                if self.validation_sequence_metrics:
                    control.update(self.control_sequence_metrics.summary())
        val_loss = self._sync_average(self._val_sum, self._val_count)
        for prefix, values in (("fused", fused), ("base", base), ("control", control)):
            for name, value in values.items():
                self.log(f"val/{prefix}_{name}", value, sync_dist=False)
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=False)
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        contact_ap = fused["contact_ap"]
        false_high_ap = fused["false_high_candidate_ap"]
        sequence_ap = fused.get("sequence_macro_contact_ap", math.nan)
        absolute_score = sum(
            value
            for value in (contact_ap, false_high_ap, sequence_ap)
            if math.isfinite(value)
        )
        aligned_control_gain = 0.0
        if control:
            aligned_control_gain = sum(
                fused[name] - control[name]
                for name in (
                    "contact_ap",
                    "false_high_candidate_ap",
                    "sequence_macro_contact_ap",
                )
                if math.isfinite(fused[name]) and math.isfinite(control[name])
            )
        information_score = absolute_score + 0.5 * aligned_control_gain
        row = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "train_loss_epoch_global": self.train_loss_epoch_global,
            "val_loss": val_loss,
            "prior_information_score": information_score,
            "aligned_control_information_gain": aligned_control_gain,
            **{f"fused_{key}": value for key, value in fused.items()},
            **{f"base_{key}": value for key, value in base.items()},
            **{f"control_{key}": value for key, value in control.items()},
        }
        path = self.output_dir / "val_metrics.csv"
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        checkpoint_dir = self.output_dir / "checkpoints"
        atomic_torch_save(self._compact("last", information_score), checkpoint_dir / "last.ckpt")
        if math.isfinite(val_loss) and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            atomic_torch_save(
                self._compact("val_loss", val_loss),
                checkpoint_dir / "best_loss.ckpt",
            )
        if math.isfinite(false_high_ap) and false_high_ap > self.best_false_high_ap:
            self.best_false_high_ap = false_high_ap
            atomic_torch_save(
                self._compact("false_high_candidate_ap", false_high_ap),
                checkpoint_dir / "best_false_high.ckpt",
            )
        if math.isfinite(contact_ap) and contact_ap > self.best_contact_ap:
            self.best_contact_ap = contact_ap
            atomic_torch_save(
                self._compact("contact_ap", contact_ap),
                checkpoint_dir / "best_contact.ckpt",
            )
        if information_score > self.best_information_score:
            self.best_information_score = information_score
            atomic_torch_save(
                self._compact("prior_information_score", information_score),
                checkpoint_dir / "prior-info-best.ckpt",
            )

    def configure_optimizers(self):
        trainable = [
            parameter
            for parameter in self.model.trainable_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("Selector-prior experiment has no trainable parameters")
        optimizer = torch.optim.AdamW(
            trainable,
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
    parser.add_argument(
        "--adapter-type",
        choices=(
            "depth_mapping_rectifier",
            "depth_anchor_residual",
            "depth_anchor_query",
            "vlm_global_calibrator",
        ),
        required=True,
    )
    parser.add_argument("--prior-dim", type=int, required=True)
    parser.add_argument("--prior-control", default="real")
    parser.add_argument("--counterfactual-control", default="spatial_shuffle")
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--dino-weights", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "PRIOR_EXPERIMENT_ROOT", "/home/ma-user/work/cfzhao/input_prior_full/experiments"
        ),
    )
    parser.add_argument("--feature-rms-budget", type=float, default=0.05)
    parser.add_argument("--prior-dropout", type=float, default=0.0)
    parser.add_argument("--depth-hidden-channels", type=int, default=128)
    parser.add_argument("--depth-modulation-max-scale", type=float, default=0.10)
    parser.add_argument("--anchor-residual-max-logit", type=float, default=2.0)
    parser.add_argument("--anchor-query-dim", type=int, default=128)
    parser.add_argument("--anchor-query-heads", type=int, default=4)
    parser.add_argument("--anchor-query-layers", type=int, default=2)
    parser.add_argument("--vlm-rank", type=int, default=32)
    parser.add_argument("--vlm-residual-max-logit", type=float, default=1.0)
    parser.add_argument("--control-identity-weight", type=float, default=0.10)
    parser.add_argument("--feature-budget-penalty-weight", type=float, default=0.005)
    parser.add_argument("--false-high-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--false-high-score-source",
        choices=("auto", "contact", "head"),
        default="auto",
        help="Use sigmoid(-contact_logits) or the dedicated head for false-high AP.",
    )
    parser.add_argument(
        "--paired-controls", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--validation-reference-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validation-sequence-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=0,
        help="Per-rank validation batch; 0 reuses --batch-size.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--cache-shuffle-block-size",
        type=int,
        default=8192,
        help="Locality-preserving shuffle block for mmap cache-only training; 0 disables it.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--no-train-augmentation", action="store_true")
    parser.add_argument("--wandb-project", default="tactile-priors-v2")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-sync-retries", type=int, default=24)
    parser.add_argument("--wandb-sync-interval", type=int, default=300)
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--resume-save-every-n-epochs", type=int, default=1)
    parser.add_argument(
        "--runtime-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record loader gaps, step time, Linux D-state tasks, and I/O pressure.",
    )
    parser.add_argument("--runtime-debug-interval", type=float, default=2.0)
    parser.add_argument("--runtime-debug-flush-steps", type=int, default=64)
    add_data_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.adapter_type == "vlm_global_calibrator" and args.counterfactual_control == "spatial_shuffle":
        args.counterfactual_control = "context_shuffle"
    if args.false_high_loss_weight < 0:
        raise ValueError("--false-high-loss-weight cannot be negative")
    if args.batch_size < 1 or args.val_batch_size < 0:
        raise ValueError("Train batch size must be positive and val batch size non-negative")
    if args.false_high_score_source == "auto":
        args.false_high_score_source = (
            "head"
            if args.adapter_type == "depth_anchor_query"
            and args.false_high_loss_weight > 0.0
            else "contact"
        )
    if args.false_high_score_source == "head" and args.adapter_type != "depth_anchor_query":
        raise ValueError(
            "--false-high-score-source=head requires adapter-type=depth_anchor_query"
        )
    if args.cache_shuffle_block_size < 0:
        raise ValueError("--cache-shuffle-block-size cannot be negative")
    if args.anchor_query_layers < 1 or args.anchor_query_heads < 1:
        raise ValueError("Anchor query layers/heads must be positive")
    if args.anchor_query_dim < 1 or args.anchor_query_dim % args.anchor_query_heads:
        raise ValueError("Anchor query dimension must be divisible by its head count")
    if args.runtime_debug_interval <= 0 or args.runtime_debug_flush_steps < 1:
        raise ValueError("Runtime debug interval/flush steps must be positive")
    pl.seed_everything(args.seed, workers=True)
    gpu_ids = parse_csv(args.gpus)
    if not gpu_ids:
        raise ValueError("--gpus cannot be empty")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    output_dir = Path(args.output_root).expanduser().resolve(strict=False) / args.exp_name
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume_from_checkpoint
    if not resume_path and args.auto_resume and (checkpoint_dir / "resume.ckpt").is_file():
        resume_path = str(checkpoint_dir / "resume.ckpt")
    resume_run_id = ""
    if resume_path:
        header = torch.load(resume_path, map_location="cpu")
        if header.get("format") != "tactile_prior_selector_resume_v1":
            raise ValueError("Resume checkpoint is not a selector-prior trainer checkpoint")
        resume_run_id = str(header.get("wandb_run_id", "") or "")
    run_id_file = checkpoint_dir / "wandb_run_id.txt"
    wandb_run_id = resume_run_id or (
        run_id_file.read_text().strip() if run_id_file.is_file() else uuid.uuid4().hex[:8]
    )
    run_id_file.write_text(wandb_run_id + "\n", encoding="utf-8")

    train_dataset = _dataset_from_args(args, "train", True)
    val_dataset = _dataset_from_args(args, "val", False)
    expected_selector_sha = file_sha256(args.selector_checkpoint)
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        cached_sha = str(getattr(dataset, "base_checkpoint_sha256", "") or "")
        if cached_sha and cached_sha != expected_selector_sha:
            raise ValueError(
                f"{split_name} frozen feature cache was built from a different "
                "selector checkpoint; rebuild it with TACTILE_BASE_CHECKPOINT="
                f"{args.selector_checkpoint}"
            )
    common_loader = {
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    train_sampler = None
    val_sampler = None
    manual_distributed_samplers = bool(args.cache_only and args.cache_shuffle_block_size > 0)
    if manual_distributed_samplers:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        train_sampler = DistributedBlockShuffleSampler(
            train_dataset,
            num_replicas=len(gpu_ids),
            rank=rank,
            block_size=args.cache_shuffle_block_size,
            seed=args.seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=len(gpu_ids),
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if rank == 0:
            print(
                "Cache-local sampling enabled: "
                f"block={args.cache_shuffle_block_size}, batch={args.batch_size}, "
                f"val_batch={args.val_batch_size or args.batch_size}, "
                f"workers/rank={args.num_workers}, val_workers/rank={args.val_num_workers}, "
                f"paired_controls={args.paired_controls}, "
                f"validation_references={args.validation_reference_metrics}, "
                f"validation_sequences={args.validation_sequence_metrics}",
                flush=True,
            )
    train_kwargs = dict(
        common_loader,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_kwargs = dict(
        common_loader,
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.val_num_workers,
    )
    if args.num_workers:
        train_kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=False)
    if args.val_num_workers:
        val_kwargs.update(prefetch_factor=args.prefetch_factor, persistent_workers=False)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    val_loader = DataLoader(val_dataset, **val_kwargs)

    adapter_config = selector_prior_config_from_args(args)
    model, selector_payload = build_prior_selector_model(
        selector_checkpoint=args.selector_checkpoint,
        dino_weights=args.dino_weights,
        **adapter_config,
    )
    if args.false_high_loss_weight == 0.0 and args.false_high_score_source == "contact":
        false_high_head = getattr(model.prior_adapter, "false_high_head", None)
        if false_high_head is not None:
            for parameter in false_high_head.parameters():
                parameter.requires_grad_(False)
    if parse_csv(args.train_base_feature_cache or args.base_feature_cache):
        model.disable_online_backbone()
    data_config = {
        "datasets": args.datasets,
        "input_resolution": args.input_resolution,
        "bbox_rescale_factor": args.bbox_rescale_factor,
        "bbox_source_policy": args.bbox_source_policy,
        "depth_sidecar_root": args.depth_sidecar_root,
        "train_base_feature_cache": args.train_base_feature_cache,
        "val_base_feature_cache": args.val_base_feature_cache,
        "train_prior_feature_cache": args.train_prior_feature_cache,
        "val_prior_feature_cache": args.val_prior_feature_cache,
        "cache_only": args.cache_only,
        "selector_checkpoint_epoch": selector_payload.get("epoch"),
    }
    module = PriorSelectorTrainingModule(
        model,
        adapter_config=adapter_config,
        data_config=data_config,
        selector_checkpoint=args.selector_checkpoint,
        dino_weights=args.dino_weights,
        output_dir=str(output_dir),
        learning_rate=args.lr * len(gpu_ids),
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        counterfactual_control=args.counterfactual_control,
        control_identity_weight=args.control_identity_weight,
        feature_budget_penalty_weight=args.feature_budget_penalty_weight,
        false_high_loss_weight=args.false_high_loss_weight,
        false_high_score_source=args.false_high_score_source,
        paired_controls=args.paired_controls,
        validation_reference_metrics=args.validation_reference_metrics,
        validation_sequence_metrics=args.validation_sequence_metrics,
        wandb_run_id=wandb_run_id,
    )
    run_config = output_dir / "run_config.json"
    run_config.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")
    logger = CSVLogger(str(output_dir), name="lightning")
    callbacks = [
        AtomicPriorResumeCheckpoint(
            checkpoint_dir, every_n_epochs=args.resume_save_every_n_epochs
        )
    ]
    if args.runtime_debug:
        runtime_debug_dir = output_dir / "runtime_debug"
        callbacks.extend(
            (
                RuntimeBatchTimingCallback(
                    runtime_debug_dir,
                    flush_steps=args.runtime_debug_flush_steps,
                ),
                RuntimeSystemMonitorCallback(
                    runtime_debug_dir,
                    interval=args.runtime_debug_interval,
                ),
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
        use_distributed_sampler=not manual_distributed_samplers,
        num_sanity_val_steps=2,
        log_every_n_steps=20,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_path or None)


if __name__ == "__main__":
    main()
