#!/usr/bin/env python3
"""Train a feature-level Depth or VLM adapter on a frozen tactile baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.process_lifecycle import (
    configure_supervised_process,
    initialize_worker_parent_death_signal,
    set_parent_death_signal,
)

configure_supervised_process()

import cv2
import torch
from torch.utils.data import DataLoader

cv2.setNumThreads(0)
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import CSVLogger
except ImportError as exc:  # pragma: no cover - exercised on the training host
    raise ImportError("Prior training requires pytorch_lightning") from exc

from tactile_input_priors.prior_metrics import PriorMetricAccumulator
from tactile_input_priors.runtime import (
    CachedFeatureDataset,
    FeatureOnlyTactileDataset,
    adapter_config_from_args,
    atomic_torch_save,
    build_dataset,
    build_prior_model,
    parse_csv,
    prior_checkpoint_payload,
)

from hamer_tactile_ft.losses import compute_tactile_loss
class PriorAdapterTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model,
        *,
        loss_config,
        adapter_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        base_checkpoint: str,
        dino_weights: str,
        output_dir: str,
        learning_rate: float,
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        wandb_run_id: str = "",
        runtime_debug: bool = False,
    ):
        super().__init__()
        self.model = model
        self.loss_config = loss_config
        self.adapter_config = dict(adapter_config)
        self.data_config = dict(data_config)
        self.base_checkpoint = str(base_checkpoint)
        self.dino_weights = str(dino_weights)
        self.output_dir = Path(output_dir)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs_config = int(max_epochs)
        self.wandb_run_id = str(wandb_run_id)
        self.runtime_debug = bool(runtime_debug)
        self.counterfactual_control = str(
            self.adapter_config.get("counterfactual_control", "") or ""
        )
        self.control_identity_weight = float(
            self.adapter_config.get("control_identity_weight", 0.0)
        )
        self.feature_budget_penalty_weight = float(
            self.adapter_config.get("feature_budget_penalty_weight", 0.0)
        )
        self._loading_prior_resume = False
        self.tactile_loss_scale = 10.0
        self.best_loss = math.inf
        self.fused_metrics = PriorMetricAccumulator()
        self.base_metrics = PriorMetricAccumulator()
        self.control_metrics = PriorMetricAccumulator()
        self._val_loss_sum = torch.tensor(0.0, dtype=torch.float64)
        self._val_loss_count = torch.tensor(0.0, dtype=torch.float64)
        self._train_loss_sum = torch.tensor(0.0, dtype=torch.float64)
        self._train_loss_count = torch.tensor(0.0, dtype=torch.float64)
        self.train_loss_epoch_global = math.nan
        self.save_hyperparameters(
            {
                "adapter_config": self.adapter_config,
                "data_config": self.data_config,
                "base_checkpoint": self.base_checkpoint,
                "dino_weights": self.dino_weights,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_epochs": self.warmup_epochs,
                "max_epochs": self.max_epochs_config,
                "wandb_run_id": self.wandb_run_id,
                "runtime_debug": self.runtime_debug,
            }
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Keep worker timing tensors on CPU so debugging never synchronizes CUDA."""

        if not isinstance(batch, dict) or "_runtime_io_debug" not in batch:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)
        payload = dict(batch)
        runtime_io_debug = payload.pop("_runtime_io_debug")
        transferred = super().transfer_batch_to_device(payload, device, dataloader_idx)
        transferred["_runtime_io_debug"] = runtime_io_debug
        return transferred

    def _resume_contract(self) -> dict[str, Any]:
        return {
            "adapter_config": self.adapter_config,
            "data_config": self.data_config,
            "base_checkpoint": self.base_checkpoint,
            "dino_weights": self.dino_weights,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "max_epochs": self.max_epochs_config,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Retain exact trainer state without serializing the frozen RGB base."""

        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            name: value
            for name, value in state_dict.items()
            if name.startswith("model.prior_adapter.")
        }
        checkpoint["format"] = "tactile_prior_resume_v1"
        checkpoint["wandb_run_id"] = self.wandb_run_id
        checkpoint["best_loss"] = float(self.best_loss)
        checkpoint["resume_contract"] = self._resume_contract()

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("format") != "tactile_prior_resume_v1":
            return
        current = self._resume_contract()
        stored = checkpoint.get("resume_contract", {})
        mismatches = {
            key: {"checkpoint": stored.get(key), "current": value}
            for key, value in current.items()
            if stored.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Prior resume checkpoint configuration mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        checkpoint_run_id = str(checkpoint.get("wandb_run_id", "") or "")
        if self.wandb_run_id and checkpoint_run_id and self.wandb_run_id != checkpoint_run_id:
            raise RuntimeError(
                "Prior resume checkpoint WandB run ID mismatch: "
                f"checkpoint={checkpoint_run_id}, current={self.wandb_run_id}"
            )
        if checkpoint_run_id:
            self.wandb_run_id = checkpoint_run_id
        self.best_loss = float(checkpoint.get("best_loss", math.inf))
        self._loading_prior_resume = True

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self._loading_prior_resume:
            return super().load_state_dict(state_dict, strict=strict)
        incompatible = super().load_state_dict(state_dict, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            name
            for name in incompatible.missing_keys
            if name.startswith("model.prior_adapter.")
        ]
        self._loading_prior_resume = False
        if unexpected or missing:
            raise RuntimeError(
                "Prior resume state mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return incompatible

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.base_model.eval()
        return self

    def forward(self, batch):
        return self.model(batch)

    def _loss(self, batch, output):
        raw, components = compute_tactile_loss(
            pred=output["pred_tactile"],
            logits=output["pred_logits"],
            target=batch["tactile_signal"],
            palm_mask=batch["palm_mask"],
            valid_mask=batch["has_tactile"],
            dataset_batch=batch.get("dataset"),
            config=self.loss_config,
            current_epoch=self.current_epoch,
            sample_weight=batch.get("sample_weight"),
            ramp_override=1.0,
        )
        return self.tactile_loss_scale * raw, components

    def training_step(self, batch, batch_idx):
        del batch_idx
        if self.counterfactual_control:
            output = self.model.forward_paired(
                batch, train=True, control=self.counterfactual_control
            )
        else:
            output = self.model(batch, train=True)
        tactile_loss, components = self._loss(batch, output)
        identity_loss = output["pred_tactile"].new_zeros(())
        if self.counterfactual_control:
            identity_loss = output["control_identity_loss"]
        budget_loss = output["pred_tactile"].new_zeros(())
        for value in output.get("prior_aux_losses", {}).values():
            budget_loss = budget_loss + value
        if self.counterfactual_control:
            control_budget = output["pred_tactile"].new_zeros(())
            for value in output.get("control_prior_aux_losses", {}).values():
                control_budget = control_budget + value
            budget_loss = 0.5 * (budget_loss + control_budget)
        weighted_identity = self.control_identity_weight * identity_loss
        weighted_budget = self.feature_budget_penalty_weight * budget_loss
        loss = tactile_loss + weighted_identity + weighted_budget
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite prior-adapter loss at epoch={self.current_epoch}, "
                f"global_step={self.global_step}"
            )
        batch_size = int(batch["tactile_signal"].shape[0])
        self._train_loss_sum.add_(loss.detach().double() * batch_size)
        self._train_loss_count.add_(batch_size)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log(
            "train/tactile_loss",
            tactile_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/control_identity_raw",
            identity_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/control_identity_weighted",
            weighted_identity,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/feature_budget_raw",
            budget_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/feature_budget_weighted",
            weighted_budget,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        for name in (
            "loss_base_tactile",
            "loss_background",
            "loss_location",
            "pressure_weight_mean",
            "weighted_to_direct_loss_ratio",
        ):
            if name in components:
                self.log(
                    f"train/{name}",
                    components[name],
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )
        for name, value in output.get("prior_diagnostics", {}).items():
            self.log(
                f"train/prior_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        for name, value in output.get("control_prior_diagnostics", {}).items():
            self.log(
                f"train/control_prior_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return loss

    def on_train_epoch_start(self):
        self._train_loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._train_loss_count = torch.zeros((), device=self.device, dtype=torch.float64)

    def _sync_average(self, total: torch.Tensor, count: torch.Tensor) -> float:
        value = torch.stack((total, count)).to(self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return float((value[0] / value[1].clamp_min(1.0)).cpu())

    def on_train_epoch_end(self):
        self.train_loss_epoch_global = self._sync_average(
            self._train_loss_sum, self._train_loss_count
        )
        self.log(
            "train/loss_epoch_global",
            self.train_loss_epoch_global,
            sync_dist=False,
        )

    def on_validation_epoch_start(self):
        self.fused_metrics.reset()
        self.base_metrics.reset()
        self.control_metrics.reset()
        self._val_loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_loss_count = torch.zeros((), device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        del batch_idx
        if self.counterfactual_control:
            output = self.model.forward_paired(
                batch, train=False, control=self.counterfactual_control
            )
        else:
            output = self.model(batch, train=False)
        loss, _ = self._loss(batch, output)
        batch_size = int(batch["tactile_signal"].shape[0])
        self._val_loss_sum.add_(loss.detach().double() * batch_size)
        self._val_loss_count.add_(batch_size)
        self.fused_metrics.update(
            output["pred_tactile"],
            batch["tactile_signal"],
            batch["palm_mask"],
            batch["has_tactile"],
        )
        self.base_metrics.update(
            output["base_pred_tactile"],
            batch["tactile_signal"],
            batch["palm_mask"],
            batch["has_tactile"],
        )
        if self.counterfactual_control:
            self.control_metrics.update(
                output["control_pred_tactile"],
                batch["tactile_signal"],
                batch["palm_mask"],
                batch["has_tactile"],
            )
        for name, value in output.get("prior_diagnostics", {}).items():
            self.log(
                f"val/prior_{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        for name, value in output.get("control_prior_diagnostics", {}).items():
            self.log(
                f"val/control_prior_{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss

    def _sync_loss(self) -> float:
        return self._sync_average(self._val_loss_sum, self._val_loss_count)

    def _write_metrics(self, row: Mapping[str, Any]) -> None:
        path = self.output_dir / "val_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _compact_payload(self, monitor: str, score: float):
        return prior_checkpoint_payload(
            self.model,
            adapter_config=self.adapter_config,
            base_checkpoint=self.base_checkpoint,
            dino_weights=self.dino_weights,
            loss_config=self.loss_config,
            data_config=self.data_config,
            epoch=self.current_epoch,
            global_step=self.global_step,
            monitor=monitor,
            score=score,
        )

    def on_validation_epoch_end(self):
        self.fused_metrics.synchronize(self.device)
        self.base_metrics.synchronize(self.device)
        if self.counterfactual_control:
            self.control_metrics.synchronize(self.device)
        val_loss = self._sync_loss()
        fused = self.fused_metrics.summary()
        base = self.base_metrics.summary()
        control = self.control_metrics.summary() if self.counterfactual_control else {}
        self.log("val/loss", val_loss, prog_bar=True, sync_dist=False)
        for name, value in fused.items():
            self.log(f"val/fused_{name}", value, sync_dist=False)
        for name, value in base.items():
            self.log(f"val/base_{name}", value, sync_dist=False)
        for name, value in control.items():
            self.log(f"val/control_{name}", value, sync_dist=False)
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        row = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "train_loss_epoch_global": self.train_loss_epoch_global,
            "val_loss": val_loss,
            **{f"fused_{key}": value for key, value in fused.items()},
            **{f"base_{key}": value for key, value in base.items()},
            **{f"control_{key}": value for key, value in control.items()},
        }
        self._write_metrics(row)
        checkpoint_dir = self.output_dir / "checkpoints"
        atomic_torch_save(
            self._compact_payload("val/loss", val_loss), checkpoint_dir / "last.ckpt"
        )
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            atomic_torch_save(
                self._compact_payload("val/loss", val_loss),
                checkpoint_dir / "loss-best.ckpt",
            )

    def on_exception(self, exception: BaseException) -> None:
        del exception
        if self.trainer.is_global_zero:
            atomic_torch_save(
                self._compact_payload("interrupted", math.nan),
                self.output_dir / "checkpoints" / "last-interrupted.ckpt",
            )

    def configure_optimizers(self):
        parameters = list(self.model.trainable_parameters())
        if not parameters:
            raise RuntimeError("Prior adapter exposes no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=False,
        )

        def schedule(step: int) -> float:
            steps_per_epoch = max(1, int(self.trainer.estimated_stepping_batches) // max(self.max_epochs_config, 1))
            warmup_steps = self.warmup_epochs * steps_per_epoch
            total_steps = max(int(self.trainer.estimated_stepping_batches), warmup_steps + 1)
            if warmup_steps > 0 and step < warmup_steps:
                return max(float(step + 1) / float(warmup_steps), 1e-8)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class AtomicPriorResumeCheckpoint(Callback):
    """Atomically persist optimizer/scheduler/loop state once per epoch."""

    def __init__(self, directory: Path, every_n_epochs: int = 1):
        super().__init__()
        self.directory = Path(directory)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.last_saved_step = -1

    @property
    def state_key(self):
        return f"{self.__class__.__qualname__}:{self.directory}"

    def state_dict(self):
        return {"last_saved_step": int(self.last_saved_step)}

    def load_state_dict(self, state_dict):
        self.last_saved_step = int(state_dict.get("last_saved_step", -1))

    def _save(self, trainer) -> None:
        if int(trainer.global_step) == self.last_saved_step:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / "resume.ckpt"
        temporary = self.directory / ".resume.ckpt.tmp"
        trainer.save_checkpoint(str(temporary), weights_only=False)
        trainer.strategy.barrier("prior_resume_checkpoint_written")
        if trainer.is_global_zero:
            os.replace(temporary, destination)
            descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            info_path = self.directory / "resume_checkpoint_info.json"
            info_tmp = info_path.with_name(f".{info_path.name}.tmp-{os.getpid()}")
            info_tmp.write_text(
                json.dumps(
                    {
                        "format": "tactile_prior_resume_v1",
                        "epoch": int(trainer.current_epoch),
                        "global_step": int(trainer.global_step),
                        "path": str(destination),
                        "wandb_run_id": str(
                            getattr(trainer.lightning_module, "wandb_run_id", "") or ""
                        ),
                        "time_unix": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(info_tmp, info_path)
            print(
                f"Atomic prior resume checkpoint: epoch={trainer.current_epoch}, "
                f"global_step={trainer.global_step} -> {destination}",
                flush=True,
            )
        trainer.strategy.barrier("prior_resume_checkpoint_materialized")
        self.last_saved_step = int(trainer.global_step)

    def on_validation_end(self, trainer, pl_module) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        if (int(trainer.current_epoch) + 1) % self.every_n_epochs == 0:
            self._save(trainer)

    def on_fit_end(self, trainer, pl_module) -> None:
        del pl_module
        self._save(trainer)


def _scalar_metric_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    elif not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class DurableWandbEpochCallback(Callback):
    """Queue epoch metrics locally and upload outside the training process."""

    def __init__(
        self,
        *,
        queue_dir: Path,
        run_id: str,
        project: str,
        entity: str,
        run_name: str,
        config_path: Path,
        retries: int,
        interval: int,
        local_logger,
    ):
        super().__init__()
        self.queue_dir = Path(queue_dir)
        self.run_id = str(run_id)
        self.project = str(project)
        self.entity = str(entity)
        self.run_name = str(run_name)
        self.config_path = Path(config_path)
        self.retries = max(1, int(retries))
        self.interval = max(1, int(interval))
        self.local_logger = local_logger
        self._last_key = None
        self._upload_process = None
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def _enqueue(self, trainer) -> bool:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return False
        key = (int(trainer.current_epoch), int(trainer.global_step))
        if key == self._last_key:
            return False
        metrics = {}
        for name, value in trainer.callback_metrics.items():
            scalar = _scalar_metric_value(value)
            if scalar is not None:
                metrics[str(name)] = scalar
        metrics["trainer/epoch"] = float(key[0])
        metrics["trainer/global_step"] = float(key[1])
        try:
            self.local_logger.save()
        except Exception as exc:
            print(f"Warning: failed to flush local CSV logger: {exc}", flush=True)
        destination = self.queue_dir / f"epoch_{key[0]:04d}_step_{key[1]:012d}.json"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(
                {
                    "format": "tactile_prior_wandb_epoch_v1",
                    "run_id": self.run_id,
                    "epoch": key[0],
                    "global_step": key[1],
                    "created_unix": time.time(),
                    "metrics": dict(sorted(metrics.items())),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
        self._last_key = key
        return True

    def _launch(self) -> None:
        if self._upload_process is not None and self._upload_process.poll() is None:
            return
        log_path = self.queue_dir / "upload.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "wandb_epoch_sync.py"),
            "--queue-dir", str(self.queue_dir),
            "--run-id", self.run_id,
            "--project", self.project,
            "--name", self.run_name,
            "--config", str(self.config_path),
            "--retries", str(self.retries),
            "--interval", str(self.interval),
        ]
        if self.entity:
            command.extend(("--entity", self.entity))
        with log_path.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=False,
                preexec_fn=lambda: set_parent_death_signal(
                    signal.SIGKILL,
                    expected_parent_pid=os.getppid(),
                ),
            )
        self._upload_process = process
        print(
            f"Queued WandB epoch upload: pid={process.pid}, run_id={self.run_id}, "
            f"log={log_path}",
            flush=True,
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        del pl_module
        if trainer.is_global_zero:
            self._launch()

    def on_validation_end(self, trainer, pl_module) -> None:
        del pl_module
        if self._enqueue(trainer):
            self._launch()

    def on_fit_end(self, trainer, pl_module) -> None:
        del pl_module
        if self._enqueue(trainer):
            self._launch()

    def on_exception(self, trainer, pl_module, exception) -> None:
        del trainer, pl_module, exception
        self._stop_uploader()

    def teardown(self, trainer, pl_module, stage) -> None:
        del trainer, pl_module, stage
        self._stop_uploader()

    def _stop_uploader(self) -> None:
        process = self._upload_process
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class RuntimeBatchTimingCallback(Callback):
    """Write rank-local loader and item-stage latency without DDP collectives."""

    STAGES = (
        "source_handle_ms",
        "source_handle_hit",
        "jpeg_hdf5_ms",
        "jpeg_decode_ms",
        "pressure_hdf5_ms",
        "source_raw_ms",
        "rgb_transform_ms",
        "depth_handle_ms",
        "depth_handle_hit",
        "depth_hdf5_ms",
        "depth_warp_ms",
        "getitem_total_ms",
    )

    def __init__(self, output_dir: Path, flush_steps: int = 64):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.flush_steps = max(1, int(flush_steps))
        self._rows = []
        self._path = None
        self._last_batch_end = None
        self._batch_started = None
        self._pending = None

    @classmethod
    def _fieldnames(cls):
        fields = [
            "time_unix", "epoch", "global_step", "batch_idx", "rank",
            "loader_gap_ms", "step_wall_ms", "batch_size",
            "unique_source_hdf5", "unique_sequences", "unique_worker_pids",
            "slowest_sample_uid", "slowest_h5_path", "slowest_sequence_key",
        ]
        for stage in cls.STAGES:
            fields.extend((f"{stage}_mean", f"{stage}_p95", f"{stage}_max"))
        return fields

    @staticmethod
    def _strings(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    @staticmethod
    def _summary(value):
        if value is None:
            return (math.nan, math.nan, math.nan, None)
        tensor = torch.as_tensor(value).detach().reshape(-1).to(device="cpu", dtype=torch.float64)
        if tensor.numel() == 0:
            return (math.nan, math.nan, math.nan, None)
        count = int(tensor.numel())
        position = max(1, min(count, int(math.ceil(0.95 * count))))
        p95 = float(torch.kthvalue(tensor, position).values.item())
        maximum, index = torch.max(tensor, dim=0)
        return float(tensor.mean().item()), p95, float(maximum.item()), int(index.item())

    @staticmethod
    def _nominal_frame_count(batch: Mapping[str, Any]) -> int:
        """Return a schema-independent frame count without synchronizing CUDA."""

        tactile = batch.get("tactile_signal")
        if torch.is_tensor(tactile):
            if tactile.ndim < 1:
                raise ValueError("tactile_signal must include a batch dimension")
            return int(tactile.shape[0])
        clip_tactile = batch.get("clip_tactile_signal")
        if torch.is_tensor(clip_tactile):
            if clip_tactile.ndim < 2:
                raise ValueError(
                    "clip_tactile_signal must include batch and time dimensions"
                )
            return int(clip_tactile.shape[0] * clip_tactile.shape[1])
        available = ", ".join(sorted(str(key) for key in batch))
        raise KeyError(
            "Runtime timing could not infer frame count: expected "
            "'tactile_signal' or 'clip_tactile_signal'; "
            f"available keys=[{available}]"
        )

    def _flush(self):
        if not self._rows or self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._path.exists() or self._path.stat().st_size == 0
        with self._path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames())
            if write_header:
                writer.writeheader()
            writer.writerows(self._rows)
        self._rows.clear()

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        self._path = self.output_dir / f"batch_timing_rank_{int(trainer.global_rank):02d}.csv"

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del pl_module
        if not isinstance(batch, Mapping):
            raise TypeError(
                "Runtime timing expects a mapping batch, "
                f"got {type(batch).__name__}"
            )
        now = time.perf_counter()
        loader_gap_ms = (
            math.nan if self._last_batch_end is None else (now - self._last_batch_end) * 1000.0
        )
        io_debug = batch.get("_runtime_io_debug", {})
        row = {
            "time_unix": f"{time.time():.6f}",
            "epoch": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "batch_idx": int(batch_idx),
            "rank": int(trainer.global_rank),
            "loader_gap_ms": loader_gap_ms,
            "step_wall_ms": math.nan,
            "batch_size": self._nominal_frame_count(batch),
        }
        h5_paths = self._strings(batch.get("h5_path"))
        sequences = self._strings(batch.get("sequence_key"))
        sample_uids = self._strings(batch.get("sample_uid"))
        row["unique_source_hdf5"] = len(set(h5_paths))
        row["unique_sequences"] = len(set(sequences))
        worker_pids = io_debug.get("worker_pid")
        if worker_pids is None:
            row["unique_worker_pids"] = 0
        else:
            values = torch.as_tensor(worker_pids).detach().reshape(-1).cpu().tolist()
            row["unique_worker_pids"] = len(set(int(item) for item in values))
        slowest_index = None
        for stage in self.STAGES:
            mean, p95, maximum, maximum_index = self._summary(io_debug.get(stage))
            row[f"{stage}_mean"] = mean
            row[f"{stage}_p95"] = p95
            row[f"{stage}_max"] = maximum
            if stage == "getitem_total_ms":
                slowest_index = maximum_index
        if slowest_index is not None:
            row["slowest_sample_uid"] = (
                sample_uids[slowest_index] if slowest_index < len(sample_uids) else ""
            )
            row["slowest_h5_path"] = (
                h5_paths[slowest_index] if slowest_index < len(h5_paths) else ""
            )
            row["slowest_sequence_key"] = (
                sequences[slowest_index] if slowest_index < len(sequences) else ""
            )
        else:
            row["slowest_sample_uid"] = ""
            row["slowest_h5_path"] = ""
            row["slowest_sequence_key"] = ""
        self._pending = row
        self._batch_started = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del trainer, pl_module, outputs, batch, batch_idx
        ended = time.perf_counter()
        if self._pending is not None and self._batch_started is not None:
            self._pending["step_wall_ms"] = (ended - self._batch_started) * 1000.0
            self._rows.append(self._pending)
        self._pending = None
        self._batch_started = None
        self._last_batch_end = ended
        if len(self._rows) >= self.flush_steps:
            self._flush()

    def on_train_epoch_end(self, trainer, pl_module):
        del trainer, pl_module
        self._flush()

    def on_exception(self, trainer, pl_module, exception):
        del trainer, pl_module, exception
        self._flush()

    def teardown(self, trainer, pl_module, stage):
        del trainer, pl_module, stage
        self._flush()


class RuntimeSystemMonitorCallback(Callback):
    """Sample Linux D-state waits and I/O pressure outside the training loop."""

    def __init__(self, output_dir: Path, interval: float = 2.0):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.interval = max(float(interval), 0.25)
        self._process = None
        self._log_handle = None

    def on_fit_start(self, trainer, pl_module):
        del pl_module
        if not trainer.is_global_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.output_dir / "monitor.log"
        self._log_handle = log_path.open("ab", buffering=0)
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "runtime_debug.py"),
            "--session-id", str(os.getsid(0)),
            "--output-dir", str(self.output_dir),
            "--interval", str(self.interval),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=False,
            preexec_fn=lambda: set_parent_death_signal(
                signal.SIGKILL, expected_parent_pid=os.getppid()
            ),
        )
        print(
            f"Runtime I/O diagnostics enabled: dir={self.output_dir}, "
            f"monitor_pid={self._process.pid}, interval={self.interval:g}s",
            flush=True,
        )

    def _stop(self):
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=max(10.0, 2.0 * self.interval))
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def on_fit_end(self, trainer, pl_module):
        del trainer, pl_module
        self._stop()

    def on_exception(self, trainer, pl_module, exception):
        del trainer, pl_module, exception
        self._stop()

    def teardown(self, trainer, pl_module, stage):
        del trainer, pl_module, stage
        self._stop()


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--datasets", default="touchanything")
    parser.add_argument("--data-roots", default="")
    parser.add_argument("--train-query-manifests", default="")
    parser.add_argument("--val-query-manifests", default="")
    parser.add_argument("--bbox-manifests", default="")
    parser.add_argument("--bbox-source-policy", default="sam3_only")
    parser.add_argument("--bbox-rescale-factor", type=float, default=1.2)
    parser.add_argument("--input-resolution", default="256x192")
    parser.add_argument(
        "--depth-sidecar-root",
        default=os.environ.get("DEPTH_SIDECAR_ROOT", ""),
        help="MoGe sidecar root; defaults to DEPTH_SIDECAR_ROOT when set.",
    )
    parser.add_argument("--base-feature-cache", default="")
    parser.add_argument("--prior-feature-cache", default="")
    parser.add_argument("--train-base-feature-cache", default="")
    parser.add_argument("--val-base-feature-cache", default="")
    parser.add_argument("--train-prior-feature-cache", default="")
    parser.add_argument("--val-prior-feature-cache", default="")
    parser.add_argument("--hdf5-handle-cache-size", type=int, default=4)
    parser.add_argument("--hdf5-manifest-cache-dir", default="")
    parser.add_argument(
        "--hdf5-batch-read-mode",
        choices=("streaming", "grouped"),
        default="streaming",
        help="Streaming bounds full-resolution image memory; grouped is a locality benchmark.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Read targets, provenance, base features, and priors only from caches.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-type",
        choices=(
            "depth_spatial",
            "depth_causal_film",
            "depth_local_xattn",
            "vlm_lowrank",
        ),
        required=True,
    )
    parser.add_argument("--prior-dim", type=int, required=True)
    parser.add_argument("--prior-control", default="real")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--dino-weights", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "PRIOR_EXPERIMENT_ROOT",
            "/home/ma-user/work/cfzhao/input_prior_full/experiments",
        ),
    )
    parser.add_argument("--feature-rms-budget", type=float, default=0.05)
    parser.add_argument("--logit-delta-max", type=float, default=0.50)
    parser.add_argument("--prior-dropout", type=float, default=0.10)
    parser.add_argument("--depth-hidden-channels", type=int, default=128)
    parser.add_argument("--depth-modulation-max-scale", type=float, default=0.10)
    parser.add_argument("--depth-attention-heads", type=int, default=4)
    parser.add_argument("--depth-attention-window", type=int, default=5)
    parser.add_argument(
        "--zero-mean-logit-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--counterfactual-control",
        choices=("", "spatial_shuffle", "sample_shuffle", "global_mean", "zero"),
        default="",
    )
    parser.add_argument("--control-identity-weight", type=float, default=0.0)
    parser.add_argument("--feature-budget-penalty-weight", type=float, default=0.0)
    parser.add_argument("--vlm-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=1)
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
    parser.add_argument(
        "--wandb-sync-retries",
        type=int,
        default=24,
        help="Background epoch-upload attempts before leaving metrics queued locally.",
    )
    parser.add_argument(
        "--wandb-sync-interval",
        type=int,
        default=300,
        help="Seconds between background WandB upload retries.",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically restore checkpoints/resume.ckpt for this experiment.",
    )
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--resume-save-every-n-epochs", type=int, default=1)
    parser.add_argument(
        "--runtime-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write rank-local loader timings and Linux D-state/I/O-pressure diagnostics.",
    )
    parser.add_argument("--runtime-debug-interval", type=float, default=2.0)
    parser.add_argument("--runtime-debug-flush-steps", type=int, default=64)
    add_data_arguments(parser)
    return parser


def _dataset_from_args(args, split: str, train: bool):
    query = args.train_query_manifests if split == "train" else args.val_query_manifests
    base_cache_value = (
        args.train_base_feature_cache if split == "train" else args.val_base_feature_cache
    ) or args.base_feature_cache
    prior_cache_value = (
        args.train_prior_feature_cache if split == "train" else args.val_prior_feature_cache
    ) or args.prior_feature_cache
    base_caches = parse_csv(base_cache_value)
    prior_caches = parse_csv(prior_cache_value)
    cached_base = bool(base_caches)
    is_vlm = args.adapter_type in {"vlm_lowrank", "vlm_global_calibrator"}
    if args.cache_only:
        if not cached_base:
            raise ValueError("--cache-only requires a base feature cache")
        return FeatureOnlyTactileDataset(
            (*base_caches, *prior_caches), adapter_type=args.adapter_type
        )
    if train and cached_base and not args.no_train_augmentation:
        raise ValueError(
            "Frozen base feature caches require --no-train-augmentation; cached RGB "
            "features cannot be paired with a newly randomized crop"
        )
    dataset = build_dataset(
        split=split,
        datasets=args.datasets,
        input_resolution=args.input_resolution,
        bbox_rescale_factor=args.bbox_rescale_factor,
        train=train,
        augmentation_enabled=train and not args.no_train_augmentation and not cached_base,
        data_roots=args.data_roots,
        query_manifests=query,
        bbox_manifests=args.bbox_manifests,
        bbox_source_policy=args.bbox_source_policy,
        depth_sidecar_root=args.depth_sidecar_root or None,
        depth_output_hw=(16, 12),
        hdf5_handle_cache_size=args.hdf5_handle_cache_size,
        hdf5_manifest_cache_dir=args.hdf5_manifest_cache_dir or None,
        io_debug_enabled=bool(args.runtime_debug and train),
        hdf5_batch_read_mode=args.hdf5_batch_read_mode,
    )
    caches = (*base_caches, *prior_caches)
    required = ("vlm_embedding",) if is_vlm else ()
    if args.adapter_type.startswith("depth_") and not args.depth_sidecar_root:
        required = ("depth_grid",)
    if cached_base:
        base_required = (
            ("h_rgb",) if args.adapter_type == "vlm_lowrank" else ("z_rgb",)
        )
        required = tuple(dict.fromkeys((*required, *base_required)))
    if caches:
        dataset = CachedFeatureDataset(dataset, caches, require_fields=required)
    elif required:
        raise ValueError(f"{args.adapter_type} requires a matching feature cache or depth sidecar")
    return dataset


def main() -> None:
    args = build_parser().parse_args()
    if args.resume_save_every_n_epochs < 1:
        raise ValueError("--resume-save-every-n-epochs must be at least 1")
    if args.wandb_sync_retries < 1 or args.wandb_sync_interval < 1:
        raise ValueError("WandB sync retries and interval must both be positive")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1")
    if args.num_workers < 0 or args.val_num_workers < 0:
        raise ValueError("DataLoader worker counts cannot be negative")
    if args.runtime_debug_interval <= 0 or args.runtime_debug_flush_steps < 1:
        raise ValueError("Runtime debug interval and flush steps must be positive")
    if args.control_identity_weight < 0 or args.feature_budget_penalty_weight < 0:
        raise ValueError("Counterfactual auxiliary weights cannot be negative")
    if args.counterfactual_control and not args.adapter_type.startswith("depth_"):
        raise ValueError("Paired counterfactual training is only supported by Depth adapters")
    if args.depth_attention_heads < 1:
        raise ValueError("--depth-attention-heads must be positive")
    if args.depth_attention_window < 1 or args.depth_attention_window % 2 == 0:
        raise ValueError("--depth-attention-window must be a positive odd integer")
    pl.seed_everything(args.seed, workers=True)
    gpu_ids = parse_csv(args.gpus)
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    output_dir = Path(args.output_root).expanduser().resolve(strict=False) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = ""
    if args.resume_from_checkpoint:
        resume_checkpoint = str(
            Path(args.resume_from_checkpoint).expanduser().resolve(strict=True)
        )
    elif args.auto_resume and (checkpoint_dir / "resume.ckpt").is_file():
        resume_checkpoint = str((checkpoint_dir / "resume.ckpt").resolve())
    resume_wandb_run_id = ""
    if resume_checkpoint:
        resume_header = torch.load(resume_checkpoint, map_location="cpu")
        if resume_header.get("format") != "tactile_prior_resume_v1":
            raise ValueError(
                "Prior training resume requires format=tactile_prior_resume_v1; "
                "compact loss-best/last checkpoints do not contain optimizer state"
            )
        resume_wandb_run_id = str(resume_header.get("wandb_run_id", "") or "")
        del resume_header
        print(f"Exact prior training resume enabled: {resume_checkpoint}", flush=True)

    run_id_path = checkpoint_dir / "wandb_run_id.txt"
    wandb_run_id = resume_wandb_run_id
    if not wandb_run_id and run_id_path.is_file():
        wandb_run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not wandb_run_id:
        wandb_run_id = uuid.uuid4().hex[:8]
    run_id_tmp = run_id_path.with_name(f".{run_id_path.name}.tmp-{os.getpid()}")
    run_id_tmp.write_text(wandb_run_id + "\n", encoding="utf-8")
    os.replace(run_id_tmp, run_id_path)
    train_dataset = _dataset_from_args(args, "train", True)
    val_dataset = _dataset_from_args(args, "val", False)
    if parse_csv(args.train_base_feature_cache or args.base_feature_cache) and not parse_csv(
        args.val_base_feature_cache or args.base_feature_cache
    ):
        raise ValueError(
            "Cache-only training also requires --val-base-feature-cache so validation "
            "does not fall back to the released DINO backbone"
        )
    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.val_num_workers,
        "pin_memory": True,
        "worker_init_fn": initialize_worker_parent_death_signal,
    }
    print(
        "Prior DataLoader configuration: "
        f"train_workers={args.num_workers}/rank, "
        f"val_workers={args.val_num_workers}/rank, "
        f"prefetch_factor={args.prefetch_factor}, persistent_workers=False",
        flush=True,
    )
    if args.num_workers > 0:
        train_loader_kwargs.update(
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
        )
    if args.val_num_workers > 0:
        val_loader_kwargs.update(
            persistent_workers=False,
            prefetch_factor=args.prefetch_factor,
        )
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    adapter_config = adapter_config_from_args(args)
    prior_model, base_payload, loss_config = build_prior_model(
        base_checkpoint=args.base_checkpoint,
        dino_weights=args.dino_weights,
        **adapter_config,
    )
    if parse_csv(args.train_base_feature_cache or args.base_feature_cache):
        prior_model.disable_online_backbone()
    data_config = {
        "datasets": args.datasets,
        "input_resolution": args.input_resolution,
        "bbox_rescale_factor": args.bbox_rescale_factor,
        "bbox_source_policy": args.bbox_source_policy,
        "train_query_manifests": args.train_query_manifests,
        "val_query_manifests": args.val_query_manifests,
        "bbox_manifests": args.bbox_manifests,
        "depth_sidecar_root": args.depth_sidecar_root,
        "base_feature_cache": args.base_feature_cache,
        "prior_feature_cache": args.prior_feature_cache,
        "train_base_feature_cache": args.train_base_feature_cache,
        "val_base_feature_cache": args.val_base_feature_cache,
        "train_prior_feature_cache": args.train_prior_feature_cache,
        "val_prior_feature_cache": args.val_prior_feature_cache,
        "cache_only": args.cache_only,
        "base_checkpoint_epoch": base_payload.get("epoch"),
    }
    effective_lr = args.lr * len(gpu_ids)
    module = PriorAdapterTrainingModule(
        prior_model,
        loss_config=loss_config,
        adapter_config=adapter_config,
        data_config=data_config,
        base_checkpoint=args.base_checkpoint,
        dino_weights=args.dino_weights,
        output_dir=str(output_dir),
        learning_rate=effective_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        wandb_run_id=wandb_run_id,
        runtime_debug=args.runtime_debug,
    )
    run_config_path = output_dir / "run_config.json"
    run_config_path.write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    local_logger = CSVLogger(str(output_dir), name="lightning")
    resume_callback = AtomicPriorResumeCheckpoint(
        checkpoint_dir,
        every_n_epochs=args.resume_save_every_n_epochs,
    )
    callbacks = [resume_callback]
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
        queue_dir = checkpoint_dir / "wandb" / "epoch_queue" / wandb_run_id
        callbacks.append(
            DurableWandbEpochCallback(
                queue_dir=queue_dir,
                run_id=wandb_run_id,
                project=args.wandb_project,
                entity=args.wandb_entity,
                run_name=args.exp_name,
                config_path=run_config_path,
                retries=args.wandb_sync_retries,
                interval=args.wandb_sync_interval,
                local_logger=local_logger,
            )
        )
        print(
            f"Durable WandB epoch sync enabled: project={args.wandb_project}, "
            f"run_id={wandb_run_id}, resume={bool(resume_checkpoint)}, "
            f"queue={queue_dir}",
            flush=True,
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=len(gpu_ids),
        strategy="ddp" if len(gpu_ids) > 1 else "auto",
        precision=args.precision,
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        logger=local_logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        use_distributed_sampler=True,
        num_sanity_val_steps=2,
        log_every_n_steps=20,
    )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_checkpoint or None,
    )


if __name__ == "__main__":
    main()
