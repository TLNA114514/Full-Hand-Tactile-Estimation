#!/usr/bin/env python3
"""Train online or legacy cached temporal image-grid fusion."""

from __future__ import annotations

import argparse
import csv
import fcntl
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
import numpy as np
import torch
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Sampler

from hamer_tactile_ft.hamer_tactile import CANONICAL_MODEL_INITIALIZATION_ORDER
from hamer_tactile_ft.losses import (
    TactileLossConfig,
    compute_tactile_loss,
    loss_ramp,
)
from hamer_tactile_ft.process_lifecycle import (
    configure_supervised_process,
    initialize_worker_historical_lightning_seed,
    initialize_worker_parent_death_signal,
)
from tactile_input_priors.prior_metrics import PriorMetricAccumulator
from tactile_input_priors.runtime import (
    atomic_torch_save,
    build_dataset,
    build_frozen_base,
    file_sha256,
    load_torch_checkpoint,
    parse_csv,
    parse_resolution,
    tactile_loss_config_from_checkpoint,
)
from tactile_input_priors.online_temporal import (
    OnlineTemporalClipDataset,
    OnlineTemporalDataset,
    OnlineTemporalRecordIndex,
    build_online_temporal_pair_index,
    online_temporal_contract,
)
from tactile_input_priors.temporal_flow import (
    PartitionedPalmCache,
    TemporalPairDataset,
    build_prediction_control_bins,
)
from tactile_input_priors.temporal_grid import (
    TEMPORAL_ARCHITECTURES,
    TEMPORAL_GRID_FORMAT,
    TEMPORAL_GRID_RESUME_FORMAT,
    TEMPORAL_MEMORY_FORMAT,
    TEMPORAL_MEMORY_RESUME_FORMAT,
    TEMPORAL_TRUNK_FORMAT,
    TEMPORAL_TRUNK_RESUME_FORMAT,
    TEMPORAL_CLIP_FORMAT,
    TEMPORAL_CLIP_RESUME_FORMAT,
    TEMPORAL_FULLGRID_FORMAT,
    TEMPORAL_FULLGRID_RESUME_FORMAT,
    TEMPORAL_ONLINEHMR_FORMAT,
    TEMPORAL_ONLINEHMR_RESUME_FORMAT,
    CausalClipTransformerFusionV4,
    FullGrid6144SpatiotemporalFusionV5,
    FullGrid6144TemporalMainTrunkV5,
    OnlineHMRPatchKVFusionV6,
    OnlineHMRPatchKVTemporalMainTrunkV6,
    FrozenOnlineGridEncoder,
    HierarchicalTemporalMemoryFusionV3,
    TemporalMainTrunkV3,
    TemporalClipMainTrunkV4,
    TemporalGridAdapterV1,
    TemporalGridTactileModel,
    TemporalLocalMemoryFusionV2,
    build_fresh_temporal_base,
    load_frozen_fullgrid_decoder,
    module_state_sha256,
    temporal_grid_tiny_check,
    temporal_clip_tiny_check,
    temporal_fullgrid_tiny_check,
    temporal_onlinehmr_tiny_check,
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


CONTACT_HISTOGRAM_BINS = 1000
CLIP_TRUNK_ARCHITECTURES = frozenset(
    {
        "causal_clip_transformer_v4",
        "fullgrid6144_bidirectional_v5",
        "onlinehmr_patch_kv_v6",
    }
)
TWIN_HEAD_ARCHITECTURES = frozenset(
    {"fullgrid6144_bidirectional_v5", "onlinehmr_patch_kv_v6"}
)
VALIDATION_DIAGNOSTIC_NAMES = (
    "selection_loss",
    "joint_selection_loss",
    "objective_loss",
    "base_preservation",
    "tactile_loss",
    "contact_loss",
    "contact_bce",
    "contact_jaccard",
    "contact_ramp",
    "shared_gradient_cosine",
    "shared_tactile_gradient_norm",
    "shared_contact_gradient_norm",
    "reset_consistency",
    "counterfactual_identity",
    "reset_fraction",
    "feature_delta_rms",
    "feature_clamp_scale",
    "history_available_fraction",
    "history_motion_rms",
    "spatial_delta_rms",
    "temporal_delta_rms",
    "temporal_layer_scale",
    "fast_feature_delta_rms",
    "medium_feature_delta_rms",
    "logit_delta_rms",
    "match_confidence",
    "match_null_fraction",
    "match_entropy",
    "match_displacement",
    "match_similarity",
)


def independent_contact_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    frame_valid: torch.Tensor,
    *,
    threshold: float,
    jaccard_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class-balanced BCE plus soft Jaccard for an independent contact head."""

    logits = logits.float().reshape(-1, logits.shape[-1])
    target = target.float().reshape_as(logits)
    eligible = frame_valid.reshape(-1) > 0.5
    if not bool(eligible.any()):
        zero = logits.sum() * 0.0
        return zero, zero.detach(), zero.detach()
    logits = logits[eligible]
    labels = target[eligible] > float(threshold)
    point_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none"
    )
    positive = labels
    negative = ~labels
    positive_loss = (
        point_loss[positive].mean()
        if bool(positive.any())
        else point_loss.sum() * 0.0
    )
    negative_loss = (
        point_loss[negative].mean()
        if bool(negative.any())
        else point_loss.sum() * 0.0
    )
    balanced_bce = 0.5 * (positive_loss + negative_loss)
    probabilities = torch.sigmoid(logits)
    labels_float = labels.float()
    intersection = (probabilities * labels_float).sum(dim=1)
    union = (
        probabilities + labels_float - probabilities * labels_float
    ).sum(dim=1)
    soft_jaccard = (1.0 - (intersection + 1.0) / (union + 1.0)).mean()
    total = balanced_bce + float(jaccard_weight) * soft_jaccard
    return total, balanced_bce.detach(), soft_jaccard.detach()


@torch.no_grad()
def contact_histogram(
    logits: torch.Tensor,
    target: torch.Tensor,
    frame_valid: torch.Tensor,
    *,
    threshold: float,
    bins: int = CONTACT_HISTOGRAM_BINS,
) -> tuple[torch.Tensor, torch.Tensor]:
    eligible = frame_valid.reshape(-1) > 0.5
    histogram = torch.zeros(2, bins, device=logits.device, dtype=torch.float64)
    confusion = torch.zeros(4, device=logits.device, dtype=torch.float64)
    if not bool(eligible.any()):
        return histogram, confusion
    probabilities = torch.sigmoid(logits.float().reshape(-1, logits.shape[-1]))[
        eligible
    ].flatten()
    labels = (
        target.float().reshape(-1, target.shape[-1])[eligible].flatten()
        > float(threshold)
    )
    indices = torch.clamp((probabilities * bins).long(), max=bins - 1)
    histogram[1] = torch.bincount(
        indices[labels], minlength=bins
    ).to(dtype=torch.float64)
    histogram[0] = torch.bincount(
        indices[~labels], minlength=bins
    ).to(dtype=torch.float64)
    predicted = probabilities >= 0.5
    confusion[0] = (predicted & labels).sum()
    confusion[1] = (predicted & ~labels).sum()
    confusion[2] = (~predicted & labels).sum()
    confusion[3] = (~predicted & ~labels).sum()
    return histogram, confusion


def summarize_contact_histogram(
    histogram: torch.Tensor, confusion: torch.Tensor
) -> dict[str, float]:
    negative, positive = histogram
    true_positive_curve = positive.flip(0).cumsum(0)
    false_positive_curve = negative.flip(0).cumsum(0)
    precision_curve = true_positive_curve / (
        true_positive_curve + false_positive_curve
    ).clamp_min(1.0)
    positive_total = positive.sum().clamp_min(1.0)
    recall_increments = positive.flip(0) / positive_total
    average_precision = (precision_curve * recall_increments).sum()
    true_positive, false_positive, false_negative, true_negative = confusion
    precision = true_positive / (true_positive + false_positive).clamp_min(1.0)
    recall = true_positive / (true_positive + false_negative).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-12)
    iou = true_positive / (
        true_positive + false_positive + false_negative
    ).clamp_min(1.0)
    centers = (
        torch.arange(
            histogram.shape[1], device=histogram.device, dtype=torch.float64
        )
        + 0.5
    ) / float(histogram.shape[1])
    totals = positive + negative
    empirical = positive / totals.clamp_min(1.0)
    calibration_error = (
        (empirical - centers).abs() * totals
    ).sum() / totals.sum().clamp_min(1.0)
    brier = (
        positive * (1.0 - centers).square() + negative * centers.square()
    ).sum() / totals.sum().clamp_min(1.0)
    return {
        "contact_ap": float(average_precision.item()),
        "contact_precision": float(precision.item()),
        "contact_recall": float(recall.item()),
        "contact_f1": float(f1.item()),
        "contact_iou_at_05": float(iou.item()),
        "contact_ece": float(calibration_error.item()),
        "contact_brier": float(brier.item()),
        "contact_positive_count": float(positive.sum().item()),
        "contact_negative_count": float(negative.sum().item()),
        "contact_true_negative": float(true_negative.item()),
    }


def optimizer_parameter_groups(named_parameters, weight_decay: float):
    decay = []
    no_decay = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        target = (
            no_decay
            if any(token in name for token in ("bias", "norm", "LayerNorm", "scale"))
            else decay
        )
        target.append(parameter)
    return (
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    )


class ExactValidationSampler(Sampler[int]):
    """Shard validation without DistributedSampler's duplicate padding rows."""

    def __init__(self, length: int, rank: int, replicas: int):
        self.length = int(length)
        self.rank = int(rank)
        self.replicas = int(replicas)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.replicas))

    def __len__(self) -> int:
        return max(0, (self.length - self.rank + self.replicas - 1) // self.replicas)


class TemporalGridTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        model_config: Mapping[str, Any],
        data_config: Mapping[str, Any],
        loss_config,
        base_checkpoint: str,
        dino_weights: str,
        output_dir: str,
        history_source: str,
        temporal_architecture: str,
        learning_rate: float,
        weight_decay: float,
        warmup_epochs: int,
        max_epochs: int,
        tactile_loss_scale: float,
        history_reset_probability: float,
        reset_consistency_weight: float,
        counterfactual_identity_weight: float,
        base_preservation_weight: float,
        contact_loss_weight: float,
        contact_jaccard_weight: float,
        contact_pressure_threshold: float,
        contact_gradient_audit_every: int,
        fresh_trunk: bool,
        wandb_run_id: str,
    ):
        super().__init__()
        self.model = model
        self.model_config = dict(model_config)
        self.data_config = dict(data_config)
        self.loss_config = loss_config
        self.base_checkpoint = str(base_checkpoint)
        self.dino_weights = str(dino_weights)
        self.output_dir = Path(output_dir)
        self.history_source = str(history_source)
        self.temporal_architecture = str(temporal_architecture)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_epochs = int(warmup_epochs)
        self.max_epochs_config = int(max_epochs)
        self.tactile_loss_scale = float(tactile_loss_scale)
        self.history_reset_probability = float(history_reset_probability)
        self.reset_consistency_weight = float(reset_consistency_weight)
        self.counterfactual_identity_weight = float(counterfactual_identity_weight)
        self.base_preservation_weight = float(base_preservation_weight)
        self.contact_loss_weight = float(contact_loss_weight)
        self.contact_jaccard_weight = float(contact_jaccard_weight)
        self.contact_pressure_threshold = float(contact_pressure_threshold)
        self.contact_gradient_audit_every = int(contact_gradient_audit_every)
        self.fresh_trunk = bool(fresh_trunk)
        self.wandb_run_id = str(wandb_run_id)
        self.best_loss = math.inf
        self.best_joint_loss = math.inf
        self.best_contact_ap = -math.inf
        self._loading_resume = False
        self._train_loss_sum = torch.tensor(0.0, dtype=torch.float64)
        self._train_count = torch.tensor(0.0, dtype=torch.float64)
        self._val_loss_sum = torch.tensor(0.0, dtype=torch.float64)
        self._val_count = torch.tensor(0.0, dtype=torch.float64)
        self._val_diagnostic_sum = torch.zeros(
            len(VALIDATION_DIAGNOSTIC_NAMES), dtype=torch.float64
        )
        self._val_contact_histogram = torch.zeros(
            2, CONTACT_HISTOGRAM_BINS, dtype=torch.float64
        )
        self._val_contact_confusion = torch.zeros(4, dtype=torch.float64)
        self.fused_metrics = PriorMetricAccumulator()
        self.base_metrics = PriorMetricAccumulator()

    def _resume_contract(self) -> dict[str, Any]:
        return {
            "model_config": self.model_config,
            "data_config": self.data_config,
            "base_checkpoint_sha256": (
                file_sha256(self.base_checkpoint) if self.base_checkpoint else ""
            ),
            "dino_weights_sha256": (
                file_sha256(self.dino_weights) if self.dino_weights else ""
            ),
            "history_source": self.history_source,
            "temporal_architecture": self.temporal_architecture,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs_config,
            "tactile_loss_scale": self.tactile_loss_scale,
            "history_reset_probability": self.history_reset_probability,
            "reset_consistency_weight": self.reset_consistency_weight,
            "counterfactual_identity_weight": self.counterfactual_identity_weight,
            "base_preservation_weight": self.base_preservation_weight,
            "contact_loss_weight": self.contact_loss_weight,
            "contact_jaccard_weight": self.contact_jaccard_weight,
            "contact_pressure_threshold": self.contact_pressure_threshold,
            "contact_gradient_audit_every": self.contact_gradient_audit_every,
            "fresh_trunk": self.fresh_trunk,
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        prefixes = (
            (
                "model.base_model.tactile_head.",
                "model.fusion.",
                "model.contact_head.",
            )
            if self.fresh_trunk
            else ("model.adapter.",)
        )
        checkpoint["state_dict"] = {
            name: value
            for name, value in checkpoint.get("state_dict", {}).items()
            if name.startswith(prefixes)
        }
        checkpoint["format"] = self._resume_format()
        checkpoint["resume_contract"] = self._resume_contract()
        checkpoint["wandb_run_id"] = self.wandb_run_id
        checkpoint["best_loss"] = self.best_loss
        checkpoint["best_joint_loss"] = self.best_joint_loss
        checkpoint["best_contact_ap"] = self.best_contact_ap

    def on_load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        expected_format = self._resume_format()
        if checkpoint.get("format") != expected_format:
            return
        if checkpoint.get("resume_contract") != self._resume_contract():
            raise RuntimeError("Temporal-grid resume configuration mismatch")
        self.wandb_run_id = str(checkpoint.get("wandb_run_id") or self.wandb_run_id)
        self.best_loss = float(checkpoint.get("best_loss", math.inf))
        self.best_joint_loss = float(checkpoint.get("best_joint_loss", math.inf))
        self.best_contact_ap = float(checkpoint.get("best_contact_ap", -math.inf))
        self._loading_resume = True

    def load_state_dict(self, state_dict, strict: bool = True):
        if not self._loading_resume:
            return super().load_state_dict(state_dict, strict=strict)
        incompatible = super().load_state_dict(state_dict, strict=False)
        self._loading_resume = False
        unexpected = list(incompatible.unexpected_keys)
        prefixes = (
            (
                "model.base_model.tactile_head.",
                "model.fusion.",
                "model.contact_head.",
            )
            if self.fresh_trunk
            else ("model.adapter.",)
        )
        missing_trainable = [
            name for name in incompatible.missing_keys if name.startswith(prefixes)
        ]
        if unexpected or missing_trainable:
            raise RuntimeError(
                "Temporal-grid resume state mismatch: "
                f"missing_trainable={missing_trainable}, unexpected={unexpected}"
            )
        return incompatible

    def _resume_format(self) -> str:
        if self.temporal_architecture == "onlinehmr_patch_kv_v6":
            return TEMPORAL_ONLINEHMR_RESUME_FORMAT
        if self.temporal_architecture == "fullgrid6144_bidirectional_v5":
            return TEMPORAL_FULLGRID_RESUME_FORMAT
        if self.temporal_architecture == "causal_clip_transformer_v4":
            return TEMPORAL_CLIP_RESUME_FORMAT
        if self.fresh_trunk:
            return TEMPORAL_TRUNK_RESUME_FORMAT
        if self.temporal_architecture == "local_memory_v2":
            return TEMPORAL_MEMORY_RESUME_FORMAT
        return TEMPORAL_GRID_RESUME_FORMAT

    def _history_inputs(self, batch, *, training: bool):
        if self.history_source == "real":
            history = batch["history_grids"]
            time_gap = batch["history_time_gap"]
            available = batch["history_available"]
            crop_transform = batch.get("history_crop_transform")
        elif self.history_source == "cross_sequence":
            history = batch["control_history_grids"]
            time_gap = batch["control_history_time_gap"]
            available = batch["control_history_available"]
            crop_transform = batch.get("control_history_crop_transform")
        else:
            raise ValueError(f"Unsupported history_source={self.history_source!r}")
        reset_mask = torch.zeros(
            history.shape[0], device=history.device, dtype=torch.bool
        )
        if training and self.history_reset_probability > 0.0:
            reset_mask = (
                torch.rand(history.shape[0], device=history.device)
                < self.history_reset_probability
            )
            if bool(reset_mask.any()):
                history = history.clone()
                time_gap = time_gap.clone()
                available = available.clone()
                if crop_transform is not None:
                    crop_transform = crop_transform.clone()
                reset_current = batch["current_grid"][reset_mask, None].expand(
                    -1, history.shape[1], -1, -1, -1
                )
                history[reset_mask] = reset_current
                time_gap[reset_mask] = 0.0
                available[reset_mask] = 0.0
                if crop_transform is not None:
                    crop_transform[reset_mask] = torch.eye(
                        3,
                        device=crop_transform.device,
                        dtype=crop_transform.dtype,
                    )
        return history, time_gap, available, crop_transform, reset_mask

    def _forward_loss(self, batch, *, training: bool):
        if self.temporal_architecture in CLIP_TRUNK_ARCHITECTURES:
            return self._forward_clip_loss(batch, training=training)
        batch = self.model.materialize_online_features(batch)
        history, time_gap, available, crop_transform, reset_mask = self._history_inputs(
            batch, training=training
        )
        output = self.model(
            batch["current_grid"],
            history,
            time_gap,
            available,
            history_crop_transform=crop_transform,
            cached_base_logits=batch.get("current_logits"),
        )
        palm_mask = torch.ones_like(batch["tactile_signal"])
        tactile_loss, components = compute_tactile_loss(
            pred=output["pred_tactile"],
            logits=output["pred_logits"],
            target=batch["tactile_signal"],
            palm_mask=palm_mask,
            valid_mask=batch["has_tactile"],
            dataset_batch="TouchAnything",
            config=self.loss_config,
            current_epoch=self.current_epoch,
            ramp_override=None if self.fresh_trunk else 1.0,
        )
        selection_tactile_loss = tactile_loss
        if self.fresh_trunk and not training:
            # Checkpoint selection must be comparable across epochs. The
            # optimization loss keeps its five-epoch ramp, while loss-best uses
            # the fully enabled fused objective just like the RGB baseline.
            selection_tactile_loss, _ = compute_tactile_loss(
                pred=output["pred_tactile"],
                logits=output["pred_logits"],
                target=batch["tactile_signal"],
                palm_mask=palm_mask,
                valid_mask=batch["has_tactile"],
                dataset_batch="TouchAnything",
                config=self.loss_config,
                current_epoch=self.current_epoch,
                ramp_override=1.0,
            )
        base_preservation = tactile_loss.new_zeros(())
        if self.fresh_trunk and self.base_preservation_weight > 0.0:
            base_preservation, _ = compute_tactile_loss(
                pred=output["base_pred_tactile"],
                logits=output["base_pred_logits"],
                target=batch["tactile_signal"],
                palm_mask=palm_mask,
                valid_mask=batch["has_tactile"],
                dataset_batch="TouchAnything",
                config=self.loss_config,
                current_epoch=self.current_epoch,
            )
        logit_delta = output["decoder_logit_delta"].float()
        if bool(reset_mask.any()):
            reset_consistency = logit_delta[reset_mask].square().mean()
        else:
            reset_consistency = logit_delta.sum() * 0.0
        counterfactual_identity = logit_delta.sum() * 0.0
        if (
            self.counterfactual_identity_weight > 0.0
            and self.history_source == "real"
            and "control_history_grids" in batch
        ):
            # Wrong-sequence history is a writer-specific safety constraint. It
            # must not reshape the jointly trained RGB/ReZero trunk merely to
            # make the normalization denominator easier to satisfy.
            counterfactual_current = batch["current_grid"].detach()
            counterfactual_grid, _ = self.model.adapter(
                counterfactual_current,
                batch["control_history_grids"].detach(),
                batch["control_history_time_gap"].detach(),
                batch["control_history_available"].detach(),
                batch.get("control_history_crop_transform"),
            )
            counterfactual_delta = (
                counterfactual_grid.float() - counterfactual_current.float()
            )
            dimensions = tuple(range(1, counterfactual_delta.ndim))
            base_energy = counterfactual_current.float().pow(2).mean(
                dim=dimensions
            ).clamp_min(1e-12)
            delta_energy = counterfactual_delta.pow(2).mean(dim=dimensions)
            budget = float(self.model.adapter.feature_rms_budget)
            # This is exactly (delta_rms / (budget * base_rms)) ** 2, but
            # avoids differentiating sqrt(x) at x=0. The previous form yielded
            # 0 * inf -> NaN at the zero-initialized identity state.
            counterfactual_identity = (
                delta_energy / (budget * budget * base_energy)
            ).mean()
        loss = (
            self.tactile_loss_scale * tactile_loss
            + self.base_preservation_weight
            * self.tactile_loss_scale
            * base_preservation
            + self.reset_consistency_weight * reset_consistency
            + self.counterfactual_identity_weight * counterfactual_identity
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite temporal-grid loss at "
                f"epoch={self.current_epoch}, global_step={self.global_step}"
            )
        diagnostics = {
            "tactile_loss": tactile_loss.detach(),
            "selection_loss": (
                self.tactile_loss_scale * selection_tactile_loss
            ).detach(),
            "objective_loss": loss.detach(),
            "base_preservation": base_preservation.detach(),
            "reset_consistency": reset_consistency.detach(),
            "counterfactual_identity": counterfactual_identity.detach(),
            "reset_fraction": reset_mask.float().mean(),
            "feature_delta_rms": output["feature_delta_rms"].mean(),
            "feature_clamp_scale": output["feature_clamp_scale"].mean(),
            "history_available_fraction": output[
                "history_available_fraction"
            ].mean(),
            "history_motion_rms": output["history_motion_rms"].mean(),
            "fast_feature_delta_rms": output.get(
                "fast_feature_delta_rms", logit_delta.new_zeros(())
            ).mean(),
            "medium_feature_delta_rms": output.get(
                "medium_feature_delta_rms", logit_delta.new_zeros(())
            ).mean(),
            "logit_delta_rms": logit_delta.square().mean().sqrt(),
            "match_confidence": output.get(
                "match_confidence", logit_delta.new_zeros(())
            ).mean(),
            "match_null_fraction": output.get(
                "match_null_fraction", logit_delta.new_zeros(())
            ).mean(),
            "match_entropy": output.get(
                "match_entropy", logit_delta.new_zeros(())
            ).mean(),
            "match_displacement": output.get(
                "match_displacement", logit_delta.new_zeros(())
            ).mean(),
            "match_similarity": output.get(
                "match_similarity", logit_delta.new_zeros(())
            ).mean(),
        }
        for name, value in components.items():
            if torch.is_tensor(value):
                diagnostics[f"loss_component_{name}"] = value.detach()
        return output, loss, diagnostics

    def _forward_clip_loss(self, batch, *, training: bool):
        batch = self.model.materialize_online_features(batch)
        clip_valid = batch["clip_valid"].float()
        force_reset = torch.zeros(
            clip_valid.shape[0], device=clip_valid.device, dtype=torch.bool
        )
        if training and self.history_reset_probability > 0.0:
            force_reset = (
                torch.rand(clip_valid.shape[0], device=clip_valid.device)
                < self.history_reset_probability
            )
        output = self.model.forward_clip(
            batch["clip_grids"],
            batch["clip_time"],
            clip_valid,
            batch["clip_crop_affine"],
            force_reset=force_reset,
            decode_base=(not training or self.base_preservation_weight > 0.0),
        )
        target = batch["clip_tactile_signal"].reshape(
            -1, batch["clip_tactile_signal"].shape[-1]
        )
        pred = output["pred_tactile"].reshape(-1, target.shape[-1])
        logits = output["pred_logits"].reshape_as(pred)
        valid = (
            batch["clip_has_tactile"].float() * clip_valid
        ).reshape(-1)
        palm_mask = torch.ones_like(target)
        tactile_loss, components = compute_tactile_loss(
            pred=pred,
            logits=logits,
            target=target,
            palm_mask=palm_mask,
            valid_mask=valid,
            dataset_batch="TouchAnything",
            config=self.loss_config,
            current_epoch=self.current_epoch,
            ramp_override=None,
        )
        selection_tactile_loss = tactile_loss
        if not training:
            selection_tactile_loss, _ = compute_tactile_loss(
                pred=pred,
                logits=logits,
                target=target,
                palm_mask=palm_mask,
                valid_mask=valid,
                dataset_batch="TouchAnything",
                config=self.loss_config,
                current_epoch=self.current_epoch,
                ramp_override=1.0,
            )
        contact_zero = tactile_loss.new_zeros(())
        contact_loss = contact_zero
        contact_bce = contact_zero.detach()
        contact_jaccard = contact_zero.detach()
        contact_ramp = 0.0
        if "contact_logits" in output:
            contact_loss, contact_bce, contact_jaccard = independent_contact_loss(
                output["contact_logits"].reshape_as(target),
                target,
                valid,
                threshold=self.contact_pressure_threshold,
                jaccard_weight=self.contact_jaccard_weight,
            )
            contact_ramp = loss_ramp(self.loss_config, self.current_epoch)
        weighted_contact = (
            self.contact_loss_weight * float(contact_ramp) * contact_loss
        )
        joint_selection_loss = self.tactile_loss_scale * (
            selection_tactile_loss + self.contact_loss_weight * contact_loss
        )

        shared_gradient_cosine = contact_zero.detach()
        shared_tactile_gradient_norm = contact_zero.detach()
        shared_contact_gradient_norm = contact_zero.detach()
        audit_gradients = (
            training
            and "shared_frame_features" in output
            and self.contact_gradient_audit_every > 0
            and int(self.global_step) % self.contact_gradient_audit_every == 0
        )
        if audit_gradients:
            shared = output["shared_frame_features"]
            tactile_gradient = torch.autograd.grad(
                tactile_loss, shared, retain_graph=True, allow_unused=False
            )[0].float()
            contact_gradient = torch.autograd.grad(
                contact_loss, shared, retain_graph=True, allow_unused=False
            )[0].float()
            gradient_mask = clip_valid[:, :, None].to(dtype=tactile_gradient.dtype)
            tactile_gradient = tactile_gradient * gradient_mask
            contact_gradient = contact_gradient * gradient_mask
            shared_tactile_gradient_norm = tactile_gradient.square().sum().sqrt()
            shared_contact_gradient_norm = contact_gradient.square().sum().sqrt()
            denominator = (
                shared_tactile_gradient_norm * shared_contact_gradient_norm
            ).clamp_min(1e-12)
            shared_gradient_cosine = (
                tactile_gradient.mul(contact_gradient).sum() / denominator
            ).detach()
        base_preservation = tactile_loss.new_zeros(())
        if self.base_preservation_weight > 0.0:
            base_prediction = output["base_pred_tactile"].reshape_as(pred)
            base_logits = output["base_pred_logits"].reshape_as(pred)
            base_preservation, _ = compute_tactile_loss(
                pred=base_prediction,
                logits=base_logits,
                target=target,
                palm_mask=palm_mask,
                valid_mask=valid,
                dataset_batch="TouchAnything",
                config=self.loss_config,
                current_epoch=self.current_epoch,
            )
        logit_delta = output["decoder_logit_delta"].float()
        frame_mask = clip_valid[:, :, None]
        if bool(force_reset.any()) and output["base_decode_performed"]:
            reset_consistency = (
                logit_delta[force_reset].square()
                * frame_mask[force_reset]
            ).sum() / frame_mask[force_reset].sum().clamp_min(1.0) / logit_delta.shape[-1]
        else:
            reset_consistency = logit_delta.sum() * 0.0
        counterfactual_identity = logit_delta.sum() * 0.0
        loss = (
            self.tactile_loss_scale * (tactile_loss + weighted_contact)
            + self.base_preservation_weight
            * self.tactile_loss_scale
            * base_preservation
            + self.reset_consistency_weight * reset_consistency
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "Non-finite causal-clip loss at "
                f"epoch={self.current_epoch}, global_step={self.global_step}"
            )

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            weights = clip_valid.to(dtype=value.dtype)
            return (value * weights).sum() / weights.sum().clamp_min(1.0)

        zero = logit_delta.new_zeros(())
        diagnostics = {
            "tactile_loss": tactile_loss.detach(),
            "selection_loss": (
                self.tactile_loss_scale * selection_tactile_loss
            ).detach(),
            "joint_selection_loss": joint_selection_loss.detach(),
            "contact_loss": contact_loss.detach(),
            "contact_bce": contact_bce,
            "contact_jaccard": contact_jaccard,
            "contact_ramp": contact_loss.new_tensor(contact_ramp),
            "shared_gradient_cosine": shared_gradient_cosine,
            "shared_gradient_audit_active": contact_loss.new_tensor(
                float(audit_gradients)
            ),
            "shared_tactile_gradient_norm": shared_tactile_gradient_norm.detach(),
            "shared_contact_gradient_norm": shared_contact_gradient_norm.detach(),
            "objective_loss": loss.detach(),
            "base_preservation": base_preservation.detach(),
            "reset_consistency": reset_consistency.detach(),
            "counterfactual_identity": counterfactual_identity.detach(),
            "reset_fraction": force_reset.float().mean(),
            "feature_delta_rms": masked_mean(output["feature_delta_rms"]),
            "feature_clamp_scale": masked_mean(output["feature_clamp_scale"]),
            "history_available_fraction": masked_mean(
                output["history_available_fraction"]
            ),
            "history_motion_rms": masked_mean(output["history_motion_rms"]),
            "spatial_delta_rms": masked_mean(
                output.get("spatial_delta_rms", torch.zeros_like(clip_valid))
            ),
            "temporal_delta_rms": masked_mean(
                output.get("temporal_delta_rms", torch.zeros_like(clip_valid))
            ),
            "temporal_layer_scale": output.get(
                "temporal_layer_scale", zero
            ).mean(),
            "fast_feature_delta_rms": zero,
            "medium_feature_delta_rms": zero,
            "logit_delta_rms": (
                (logit_delta.square() * frame_mask).sum()
                / (frame_mask.sum().clamp_min(1.0) * logit_delta.shape[-1])
            ).sqrt(),
            "match_confidence": zero,
            "match_null_fraction": zero,
            "match_entropy": zero,
            "match_displacement": zero,
            "match_similarity": zero,
        }
        for name, value in components.items():
            if torch.is_tensor(value):
                diagnostics[f"loss_component_{name}"] = value.detach()
        output["flat_target"] = target
        output["flat_valid"] = valid
        output["valid_frame_count"] = valid.detach().sum()
        return output, loss, diagnostics

    def on_train_epoch_start(self) -> None:
        sampler = getattr(self.trainer.train_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(int(self.current_epoch))
        self._train_loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._train_count = torch.zeros((), device=self.device, dtype=torch.float64)

    def training_step(self, batch, batch_idx):
        del batch_idx
        output, loss, diagnostics = self._forward_loss(batch, training=True)
        batch_size: int | torch.Tensor = int(output["pred_tactile"].shape[0])
        if self.temporal_architecture in CLIP_TRUNK_ARCHITECTURES:
            batch_size = output["valid_frame_count"].double()
        self._train_loss_sum += loss.detach().double() * batch_size
        self._train_count += batch_size
        for name, value in diagnostics.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        self.log("train/loss", loss, on_step=True, on_epoch=False, sync_dist=False)
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        del optimizer
        named_parameters = (
            self.model.trainable_named_parameters()
            if hasattr(self.model, "trainable_named_parameters")
            else self.model.adapter.named_parameters()
        )
        for name, parameter in named_parameters:
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(
                    f"Temporal-grid parameter became non-finite before optimizer step: {name}"
                )
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(
                    f"Temporal-grid gradient became non-finite before optimizer step: {name}"
                )

    def on_train_epoch_end(self) -> None:
        values = torch.stack((self._train_loss_sum, self._train_count))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
        self.log(
            "train/loss_epoch",
            values[0] / values[1].clamp_min(1.0),
            sync_dist=False,
        )

    def on_validation_epoch_start(self) -> None:
        self._val_loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_count = torch.zeros((), device=self.device, dtype=torch.float64)
        self._val_diagnostic_sum = torch.zeros(
            len(VALIDATION_DIAGNOSTIC_NAMES),
            device=self.device,
            dtype=torch.float64,
        )
        self._val_contact_histogram = torch.zeros(
            2,
            CONTACT_HISTOGRAM_BINS,
            device=self.device,
            dtype=torch.float64,
        )
        self._val_contact_confusion = torch.zeros(
            4, device=self.device, dtype=torch.float64
        )
        self.fused_metrics.reset()
        self.base_metrics.reset()

    def validation_step(self, batch, batch_idx):
        del batch_idx
        output, loss, diagnostics = self._forward_loss(batch, training=False)
        batch_size: int | torch.Tensor = int(output["pred_tactile"].shape[0])
        if self.temporal_architecture in CLIP_TRUNK_ARCHITECTURES:
            batch_size = output["valid_frame_count"].double()
        self._val_loss_sum += diagnostics["selection_loss"].double() * batch_size
        self._val_count += batch_size
        self._val_diagnostic_sum += torch.stack(
            [
                diagnostics.get(name, loss.new_zeros(())).detach().double()
                for name in VALIDATION_DIAGNOSTIC_NAMES
            ]
        ) * batch_size
        if self.temporal_architecture in CLIP_TRUNK_ARCHITECTURES:
            target = output["flat_target"]
            valid = output["flat_valid"]
            fused_prediction = output["pred_tactile"].reshape_as(target)
            base_prediction = output["base_pred_tactile"].reshape_as(target)
            if "contact_logits" in output:
                histogram, confusion = contact_histogram(
                    output["contact_logits"],
                    target,
                    valid,
                    threshold=self.contact_pressure_threshold,
                )
                self._val_contact_histogram += histogram
                self._val_contact_confusion += confusion
        else:
            target = batch["tactile_signal"]
            valid = batch["has_tactile"]
            fused_prediction = output["pred_tactile"]
            base_prediction = output["base_pred_tactile"]
        palm_mask = torch.ones_like(target)
        self.fused_metrics.update(
            fused_prediction,
            target,
            palm_mask,
            valid,
        )
        self.base_metrics.update(
            base_prediction,
            target,
            palm_mask,
            valid,
        )

    def _compact(self, selection_metric: str, selection_value: float) -> dict[str, Any]:
        state = {
            name: value.detach().cpu()
            for name, value in (
                self.model.compact_state_dict().items()
                if self.fresh_trunk
                else self.model.fusion.state_dict().items()
            )
        }
        payload = {
            "format": (
                TEMPORAL_ONLINEHMR_FORMAT
                if self.temporal_architecture == "onlinehmr_patch_kv_v6"
                else TEMPORAL_FULLGRID_FORMAT
                if self.temporal_architecture == "fullgrid6144_bidirectional_v5"
                else TEMPORAL_CLIP_FORMAT
                if self.temporal_architecture == "causal_clip_transformer_v4"
                else TEMPORAL_TRUNK_FORMAT
                if self.fresh_trunk
                else (
                    TEMPORAL_MEMORY_FORMAT
                    if self.temporal_architecture == "local_memory_v2"
                    else TEMPORAL_GRID_FORMAT
                )
            ),
            "model_config": self.model_config,
            "data_config": self.data_config,
            "base_checkpoint": self.base_checkpoint,
            "base_checkpoint_sha256": (
                file_sha256(self.base_checkpoint) if self.base_checkpoint else ""
            ),
            "dino_weights": self.dino_weights,
            "dino_weights_sha256": (
                file_sha256(self.dino_weights) if self.dino_weights else ""
            ),
            "history_source": self.history_source,
            "temporal_architecture": self.temporal_architecture,
            "loss_config": {
                name: getattr(self.loss_config, name)
                for name in self.loss_config.__dataclass_fields__
            },
            "selection_metric": selection_metric,
            "selection_value": float(selection_value),
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "wandb_run_id": self.wandb_run_id,
            "model_initialization_order": self.model_config.get(
                "model_initialization_order", CANONICAL_MODEL_INITIALIZATION_ORDER
            ),
            "initial_tactile_head_sha256": self.model_config.get(
                "initial_tactile_head_sha256", ""
            ),
            "initial_fast_writer_sha256": self.model_config.get(
                "initial_fast_writer_sha256", ""
            ),
            "initial_temporal_module_sha256": self.model_config.get(
                "initial_temporal_module_sha256", ""
            ),
            "initial_fusion_sha256": self.model_config.get(
                "initial_fusion_sha256", ""
            ),
            "initial_spatial_module_sha256": self.model_config.get(
                "initial_spatial_module_sha256", ""
            ),
            "initial_contact_head_sha256": self.model_config.get(
                "initial_contact_head_sha256", ""
            ),
        }
        if self.fresh_trunk:
            payload["trunk_state_dict"] = state
        elif self.temporal_architecture == "local_memory_v2":
            payload["fusion_state_dict"] = state
        else:
            payload["adapter_state_dict"] = state
        return payload

    def on_validation_epoch_end(self) -> None:
        totals = torch.cat(
            (
                torch.stack((self._val_loss_sum, self._val_count)),
                self._val_diagnostic_sum,
            )
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
            torch.distributed.all_reduce(self._val_contact_histogram)
            torch.distributed.all_reduce(self._val_contact_confusion)
        self.fused_metrics.synchronize(self.device)
        self.base_metrics.synchronize(self.device)
        val_loss = float((totals[0] / totals[1].clamp_min(1.0)).item())
        diagnostics = totals[2:] / totals[1].clamp_min(1.0)
        fused = self.fused_metrics.summary()
        base = self.base_metrics.summary()
        contact = summarize_contact_histogram(
            self._val_contact_histogram, self._val_contact_confusion
        )
        self.log("val/loss", val_loss, sync_dist=False)
        for name, value in fused.items():
            self.log(f"val/fused_{name}", value, sync_dist=False)
        for name, value in base.items():
            self.log(f"val/base_{name}", value, sync_dist=False)
        if contact["contact_positive_count"] > 0.0:
            for name, value in contact.items():
                self.log(f"val/{name}", value, sync_dist=False)
        # Sanity validation runs before the first optimizer step. Saving from
        # it silently turns zero-init into the formal loss-best checkpoint.
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        row: dict[str, Any] = {
            "epoch": int(self.current_epoch),
            "global_step": int(self.global_step),
            "history_source": self.history_source,
            "val_loss": val_loss,
        }
        row.update(
            {
                f"val_{name}": float(diagnostics[index].item())
                for index, name in enumerate(VALIDATION_DIAGNOSTIC_NAMES)
            }
        )
        row.update({f"fused_{name}": value for name, value in fused.items()})
        row.update({f"base_{name}": value for name, value in base.items()})
        row.update(contact)
        for metric in (
            "rmse",
            "contact_iou",
            "volumetric_iou",
            "core_distribution_viou",
            "temporal_accuracy_frame",
            "false_high_excess_fraction",
        ):
            row[f"fused_minus_base_{metric}"] = fused[metric] - base[metric]
        metrics_path = self.output_dir / "val_metrics.csv"
        write_header = not metrics_path.is_file()
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        checkpoint_dir = self.output_dir / "checkpoints"
        atomic_torch_save(self._compact("last", val_loss), checkpoint_dir / "last.ckpt")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            atomic_torch_save(
                self._compact("val/loss", val_loss), checkpoint_dir / "best_loss.ckpt"
            )
        joint_loss = float(
            diagnostics[
                VALIDATION_DIAGNOSTIC_NAMES.index("joint_selection_loss")
            ].item()
        )
        if (
            contact["contact_positive_count"] > 0.0
            and joint_loss > 0.0
            and joint_loss < self.best_joint_loss
        ):
            self.best_joint_loss = joint_loss
            atomic_torch_save(
                self._compact("val/joint_selection_loss", joint_loss),
                checkpoint_dir / "best_joint.ckpt",
            )
        contact_ap = float(contact["contact_ap"])
        if (
            contact["contact_positive_count"] > 0.0
            and contact_ap > self.best_contact_ap
        ):
            self.best_contact_ap = contact_ap
            atomic_torch_save(
                self._compact("val/contact_ap", contact_ap),
                checkpoint_dir / "best_contact_ap.ckpt",
            )

    def configure_optimizers(self):
        parameters = (
            optimizer_parameter_groups(
                self.model.trainable_named_parameters(), self.weight_decay
            )
            if self.fresh_trunk
            else self.model.trainable_parameters()
        )
        optimizer_kwargs = {
            "lr": self.learning_rate,
            "weight_decay": 0.0 if self.fresh_trunk else self.weight_decay,
        }
        if not self.fresh_trunk:
            optimizer_kwargs["fused"] = False
        optimizer = torch.optim.AdamW(parameters, **optimizer_kwargs)

        if self.fresh_trunk:
            total = max(1, int(self.trainer.estimated_stepping_batches))
            warmup = int(
                round(
                    total
                    * max(self.warmup_epochs, 0)
                    / max(self.max_epochs_config, 1)
                )
            )
            if warmup > 0:
                warmup = min(warmup, max(total - 1, 1))
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0 / float(max(warmup, 1)),
                    end_factor=1.0,
                    total_iters=warmup,
                )
                decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(total - warmup, 1),
                    eta_min=self.learning_rate * 0.01,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=(warmup_scheduler, decay),
                    milestones=(warmup,),
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total,
                    eta_min=self.learning_rate * 0.01,
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

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
    parser.add_argument("--data-mode", choices=("online", "cache"), default="online")
    parser.add_argument("--train-cache", default="")
    parser.add_argument("--val-cache", default="")
    parser.add_argument("--train-query-manifests", required=True)
    parser.add_argument("--val-query-manifests", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default="",
        help="Required by frozen probes; forbidden as initialization for fresh trunk V3.",
    )
    parser.add_argument("--dino-weights", default="")
    parser.add_argument("--train-data-roots", default="")
    parser.add_argument("--val-data-roots", default="")
    parser.add_argument("--bbox-manifests", default="")
    parser.add_argument("--bbox-source-policy", default="sam3_only")
    parser.add_argument("--bbox-rescale-factor", type=float, default=1.2)
    parser.add_argument("--input-resolution", default="256x192")
    parser.add_argument(
        "--hdf5-manifest-index-dir",
        default=os.environ.get(
            "TEMPORAL_HDF5_INDEX_ROOT",
            "/home/ma-user/work/cfzhao/input_prior_full/state/hdf5_manifest_index",
        ),
        help=(
            "Small mmap sample index used by online HDF5 loading. It contains no "
            "RGB, DINO features, logits, or targets."
        ),
    )
    parser.add_argument("--hdf5-handle-cache-size", type=int, default=8)
    parser.add_argument("--online-encoder-chunk-size", type=int, default=128)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "TEMPORAL_EXPERIMENT_ROOT",
            "/home/ma-user/work/cfzhao/input_prior_full/temporal_experiments",
        ),
    )
    parser.add_argument(
        "--pair-index-root",
        default=os.environ.get(
            "TEMPORAL_PAIR_ROOT",
            "/home/ma-user/work/cfzhao/input_prior_full/cache/temporal_pairs",
        ),
    )
    parser.add_argument(
        "--history-source", choices=("real", "cross_sequence"), default="real"
    )
    parser.add_argument(
        "--temporal-architecture",
        choices=TEMPORAL_ARCHITECTURES,
        default="grid_difference_v1",
    )
    parser.add_argument("--history-lags", type=_parse_history_lags, default=(1, 2, 4, 8))
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--nominal-fps", type=float, default=30.0)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--feature-rms-budget", type=float, default=0.05)
    parser.add_argument("--search-window", type=int, default=5)
    parser.add_argument("--match-temperature", type=float, default=0.07)
    parser.add_argument("--null-similarity", type=float, default=0.40)
    parser.add_argument("--medium-null-similarity", type=float, default=0.50)
    parser.add_argument("--medium-feature-rms-budget", type=float, default=0.025)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--spatial-layers", type=int, default=1)
    parser.add_argument("--spatial-heads", type=int, default=4)
    parser.add_argument("--spatial-ffn-ratio", type=int, default=2)
    parser.add_argument("--clip-layers", type=int, default=2)
    parser.add_argument("--clip-heads", type=int, default=4)
    parser.add_argument("--clip-ffn-ratio", type=int, default=2)
    parser.add_argument("--clip-residual-dropout", type=float, default=0.10)
    parser.add_argument("--clip-layer-scale-init", type=float, default=1e-3)
    parser.add_argument("--onlinehmr-hidden-dim", type=int, default=512)
    parser.add_argument("--onlinehmr-memory-frames", type=int, default=2)
    parser.add_argument("--contact-loss-weight", type=float, default=0.10)
    parser.add_argument("--contact-jaccard-weight", type=float, default=0.25)
    parser.add_argument("--contact-pressure-threshold", type=float, default=0.10)
    parser.add_argument("--contact-gradient-audit-every", type=int, default=500)
    parser.add_argument("--history-reset-probability", type=float, default=0.20)
    parser.add_argument("--reset-consistency-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-identity-weight", type=float, default=0.0)
    parser.add_argument("--base-preservation-weight", type=float, default=0.0)
    parser.add_argument(
        "--model-initialization-order",
        default=CANONICAL_MODEL_INITIALIZATION_ORDER,
        choices=(CANONICAL_MODEL_INITIALIZATION_ORDER,),
    )
    parser.add_argument("--tactile-loss-scale", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-open-shards", type=int, default=4)
    parser.add_argument("--cache-shuffle-block-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
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
    parser.add_argument("--tiny-check", action="store_true")
    return parser


def _label_free_control_bins(
    cache_root: str,
    pair_path: Path,
    *,
    max_open_shards: int,
) -> tuple[np.ndarray, Path]:
    cache = PartitionedPalmCache(cache_root, max_open_shards=max_open_shards)
    sidecar = pair_path.with_name(f"{pair_path.stem}-rgbmax-control.npz")
    lock_path = sidecar.with_suffix(sidecar.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        build_prediction_control_bins(cache, pair_path, sidecar)
    with np.load(sidecar, allow_pickle=False) as payload:
        bins = np.asarray(payload["prediction_pressure_bin"], dtype=np.int64)
    return bins, sidecar


def main() -> None:
    args = build_parser().parse_args()
    if args.tiny_check:
        print(
            json.dumps(
                {
                    "temporal_grid": temporal_grid_tiny_check(),
                    "causal_clip": temporal_clip_tiny_check(),
                    "fullgrid6144": temporal_fullgrid_tiny_check(),
                    "onlinehmr_patch_kv": temporal_onlinehmr_tiny_check(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.history_lags[0] != 1:
        raise ValueError("Temporal fusion requires lag 1 as its nearest history")
    if (
        args.temporal_architecture == "local_memory_v2"
        and args.history_lags != (1, 2)
    ):
        raise ValueError("local_memory_v2 currently requires --history-lags 1,2")
    clip_trunk = args.temporal_architecture in CLIP_TRUNK_ARCHITECTURES
    fresh_trunk = args.temporal_architecture in (
        CLIP_TRUNK_ARCHITECTURES | {"hierarchical_memory_v3"}
    )
    if fresh_trunk:
        if not clip_trunk and args.history_lags not in {(1, 2), (1, 2, 4, 8, 16)}:
            raise ValueError(
                "hierarchical_memory_v3 requires --history-lags 1,2 or 1,2,4,8,16"
            )
        if args.data_mode != "online":
            raise ValueError("Fresh temporal trunk training is online-only")
        if args.base_checkpoint:
            raise ValueError(
                "Fresh temporal trunks are trained from scratch and must not receive "
                "--base-checkpoint; auto-resume only resumes the same fresh run"
            )
        if parse_resolution(args.input_resolution) != (256, 192):
            raise ValueError("Fresh temporal trunk comparison is fixed to 256x192")
    if not 0.0 <= args.history_reset_probability < 1.0:
        raise ValueError("history reset probability must lie in [0, 1)")
    if args.contact_loss_weight < 0.0 or args.contact_jaccard_weight < 0.0:
        raise ValueError("Contact loss weights must be nonnegative")
    if not 0.0 <= args.contact_pressure_threshold <= 1.0:
        raise ValueError("Contact pressure threshold must lie in [0, 1]")
    if args.contact_gradient_audit_every < 0:
        raise ValueError("Contact gradient audit interval must be nonnegative")
    if args.accumulate_grad_batches <= 0:
        raise ValueError("--accumulate-grad-batches must be positive")
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
    include_control = (
        args.history_source == "cross_sequence"
        or args.counterfactual_identity_weight > 0.0
    )
    decoder = None
    decoder_metadata: dict[str, Any] = {}
    if fresh_trunk:
        input_resolution = parse_resolution(args.input_resolution)
        expected_shape = (256, input_resolution[0] // 16, input_resolution[1] // 16)
    else:
        if not args.base_checkpoint:
            raise ValueError("Frozen temporal probes require --base-checkpoint")
        decoder, decoder_metadata = load_frozen_fullgrid_decoder(args.base_checkpoint)
        expected_shape = (
            int(decoder_metadata["grid_channels"]),
            *tuple(int(value) for value in decoder_metadata["grid_size"]),
        )
    online_encoder = None
    dino_weights = ""
    if args.data_mode == "online":
        if not args.dino_weights:
            raise ValueError("--dino-weights is required for --data-mode online")
        dino_weights = str(Path(args.dino_weights).expanduser().resolve(strict=True))
        input_resolution = parse_resolution(args.input_resolution)
        if (
            not fresh_trunk
            and tuple(decoder_metadata["input_resolution"]) != tuple(input_resolution)
        ):
            raise RuntimeError(
                "Online input resolution differs from the frozen decoder contract"
            )
        dataset_kwargs = {
            "datasets": "touchanything",
            "input_resolution": input_resolution,
            "bbox_rescale_factor": args.bbox_rescale_factor,
            "bbox_manifests": args.bbox_manifests or None,
            "bbox_source_policy": args.bbox_source_policy,
            "hdf5_handle_cache_size": args.hdf5_handle_cache_size,
            "hdf5_manifest_cache_dir": args.hdf5_manifest_index_dir or None,
            "hdf5_batch_read_mode": "grouped",
        }
        train_base_dataset = build_dataset(
            split="train",
            train=fresh_trunk,
            augmentation_enabled=fresh_trunk,
            data_roots=args.train_data_roots or None,
            query_manifests=train_manifests,
            **dataset_kwargs,
        )
        val_base_dataset = build_dataset(
            split="val",
            train=False,
            augmentation_enabled=False,
            data_roots=args.val_data_roots or None,
            query_manifests=val_manifests,
            **dataset_kwargs,
        )
        train_record_index = OnlineTemporalRecordIndex(
            train_base_dataset,
            input_resolution=input_resolution,
            bbox_rescale_factor=args.bbox_rescale_factor,
        )
        val_record_index = OnlineTemporalRecordIndex(
            val_base_dataset,
            input_resolution=input_resolution,
            bbox_rescale_factor=args.bbox_rescale_factor,
        )
        train_pair = build_online_temporal_pair_index(
            train_record_index,
            train_manifests,
            pair_root,
            "train",
            seed=args.seed,
        )
        val_pair = build_online_temporal_pair_index(
            val_record_index,
            val_manifests,
            pair_root,
            "val",
            seed=args.seed,
        )
        palm_vertex_indices = np.flatnonzero(
            np.asarray(train_base_dataset.palm_mask) > 0.5
        ).astype(np.int64)
        val_palm = np.flatnonzero(
            np.asarray(val_base_dataset.palm_mask) > 0.5
        ).astype(np.int64)
        if not np.array_equal(palm_vertex_indices, val_palm):
            raise RuntimeError("Train/validation palm masks differ")
        if clip_trunk:
            train_dataset = OnlineTemporalClipDataset(
                train_base_dataset,
                train_pair,
                palm_vertex_indices=palm_vertex_indices,
                clip_length=args.clip_length,
                include_control=False,
                seed=args.seed,
            )
            val_dataset = OnlineTemporalClipDataset(
                val_base_dataset,
                val_pair,
                palm_vertex_indices=palm_vertex_indices,
                clip_length=args.clip_length,
                include_control=False,
                seed=args.seed,
            )
            train_clip_lengths = np.asarray(train_dataset.clip_lengths, dtype=np.int64)
            val_clip_lengths = np.asarray(val_dataset.clip_lengths, dtype=np.int64)
        else:
            train_dataset = OnlineTemporalDataset(
                train_base_dataset,
                train_pair,
                palm_vertex_indices=palm_vertex_indices,
                history_lags=args.history_lags,
                include_control=include_control,
                pair_only=not fresh_trunk,
            )
            val_dataset = OnlineTemporalDataset(
                val_base_dataset,
                val_pair,
                palm_vertex_indices=palm_vertex_indices,
                history_lags=args.history_lags,
                include_control=include_control and not fresh_trunk,
                pair_only=not fresh_trunk,
            )
        if fresh_trunk:
            base_payload = {}
            loss_config = TactileLossConfig(
                loss_ramp_epochs=5,
                location_loss_weight=0.001,
            )
        else:
            frozen_base, base_payload, loss_config = build_frozen_base(
                args.base_checkpoint, dino_weights
            )
            if tuple(frozen_base.input_resolution) != tuple(input_resolution):
                raise RuntimeError(
                    "Frozen DINO base input resolution differs from online data"
                )
            online_encoder = FrozenOnlineGridEncoder(frozen_base)
        data_config = {
            "mode": "online",
            "train": online_temporal_contract(train_dataset, train_manifests),
            "val": online_temporal_contract(val_dataset, val_manifests),
            "input_resolution": list(input_resolution),
            "bbox_rescale_factor": float(args.bbox_rescale_factor),
            "bbox_source_policy": args.bbox_source_policy,
            "hdf5_manifest_index_dir": str(
                Path(args.hdf5_manifest_index_dir).expanduser().resolve(strict=False)
            ),
            "feature_cache_generated": False,
            "train_full_frame_distribution": bool(fresh_trunk),
            "non_overlapping_temporal_clips": bool(clip_trunk),
            "clip_attention_direction": (
                "bidirectional"
                if args.temporal_architecture == "fullgrid6144_bidirectional_v5"
                else "causal"
                if clip_trunk
                else "none"
            ),
            "clip_length": int(args.clip_length) if clip_trunk else None,
            "train_clip_count": len(train_dataset) if clip_trunk else None,
            "val_clip_count": len(val_dataset) if clip_trunk else None,
            "train_source_frame_count": (
                len(train_base_dataset) if clip_trunk else None
            ),
            "val_source_frame_count": (
                len(val_base_dataset) if clip_trunk else None
            ),
            "train_mean_valid_frames_per_clip": (
                float(train_clip_lengths.mean()) if clip_trunk else None
            ),
            "val_mean_valid_frames_per_clip": (
                float(val_clip_lengths.mean()) if clip_trunk else None
            ),
            "train_full_clip_fraction": (
                float((train_clip_lengths == args.clip_length).mean())
                if clip_trunk
                else None
            ),
            "val_full_clip_fraction": (
                float((val_clip_lengths == args.clip_length).mean())
                if clip_trunk
                else None
            ),
            "train_clip_padding_fraction": (
                1.0
                - len(train_base_dataset)
                / max(len(train_dataset) * int(args.clip_length), 1)
                if clip_trunk
                else None
            ),
            "val_clip_padding_fraction": (
                1.0
                - len(val_base_dataset)
                / max(len(val_dataset) * int(args.clip_length), 1)
                if clip_trunk
                else None
            ),
            "train_clip_contract_sha256": (
                train_dataset.config_sha256 if clip_trunk else None
            ),
            "val_clip_contract_sha256": (
                val_dataset.config_sha256 if clip_trunk else None
            ),
            "train_augmentation": bool(fresh_trunk),
            "temporal_augmentation_grouping": (
                "shared_scale_translation_per_clip"
                if clip_trunk and fresh_trunk
                else "per_sample"
            ),
            "crop_pipeline": "legacy_square_center",
            "hdf5_sample_order": "legacy_sample_dir_hand",
        }
    else:
        if not args.train_cache or not args.val_cache:
            raise ValueError(
                "--train-cache and --val-cache are required for --data-mode cache"
            )
        train_pair = _pair_index(
            args.train_cache, train_manifests, pair_root, "train", args
        )
        val_pair = _pair_index(
            args.val_cache, val_manifests, pair_root, "val", args
        )
        train_control_bins = val_control_bins = None
        train_control_sidecar = val_control_sidecar = None
        if include_control:
            train_control_bins, train_control_sidecar = _label_free_control_bins(
                args.train_cache, train_pair, max_open_shards=args.max_open_shards
            )
            val_control_bins, val_control_sidecar = _label_free_control_bins(
                args.val_cache, val_pair, max_open_shards=args.max_open_shards
            )
        uses_affine_matching = args.temporal_architecture == "local_memory_v2"
        train_dataset = TemporalPairDataset(
            args.train_cache,
            train_pair,
            include_control=include_control,
            include_dino_grid=True,
            include_crop_transform=uses_affine_matching,
            include_control_current_grid=False,
            history_lags=args.history_lags,
            max_open_shards=args.max_open_shards,
            control_pressure_bins=train_control_bins,
            control_crop_transform_from_current=uses_affine_matching,
        )
        val_dataset = TemporalPairDataset(
            args.val_cache,
            val_pair,
            include_control=include_control,
            include_dino_grid=True,
            include_crop_transform=uses_affine_matching,
            include_control_current_grid=False,
            history_lags=args.history_lags,
            max_open_shards=args.max_open_shards,
            control_pressure_bins=val_control_bins,
            control_crop_transform_from_current=uses_affine_matching,
        )
        expected_sha = file_sha256(args.base_checkpoint)
        for name, dataset in (("train", train_dataset), ("val", val_dataset)):
            if dataset.cache.base_checkpoint_sha256 != expected_sha:
                raise RuntimeError(
                    f"{name} cache was built from a different base checkpoint"
                )
        sample_shape = tuple(train_dataset[0]["current_grid"].shape)
        if sample_shape != expected_shape:
            raise RuntimeError(
                f"Cached grid shape {sample_shape} differs from decoder contract "
                f"{expected_shape}"
            )
        palm_vertex_indices = train_dataset.cache.palm_vertex_indices
        data_config = {
            "mode": "cache",
            "train_cache": str(Path(args.train_cache).resolve()),
            "val_cache": str(Path(args.val_cache).resolve()),
            "train_cache_sha256": train_dataset.cache.config_sha256,
            "val_cache_sha256": val_dataset.cache.config_sha256,
            "train_pair_index": str(train_pair),
            "val_pair_index": str(val_pair),
            "train_pair_sha256": file_sha256(train_pair),
            "val_pair_sha256": file_sha256(val_pair),
            "control_bin_source": (
                "frozen_rgb_prediction" if include_control else "none"
            ),
            "train_control_sidecar": (
                str(train_control_sidecar)
                if train_control_sidecar is not None
                else ""
            ),
            "val_control_sidecar": (
                str(val_control_sidecar) if val_control_sidecar is not None else ""
            ),
        }
        base_payload = load_torch_checkpoint(args.base_checkpoint)
        loss_config = tactile_loss_config_from_checkpoint(base_payload, full_ramp=True)
    initial_tactile_head_sha256 = ""
    initial_fast_writer_sha256 = ""
    initial_fusion_sha256 = ""
    initial_spatial_module_sha256 = ""
    initial_temporal_module_sha256 = ""
    initial_contact_head_sha256 = ""
    if fresh_trunk:
        # Match the canonical baseline RNG protocol: construct the complete RGB
        # base first, record it, then append temporal modules. No tactile ckpt is
        # loaded here.
        base_model = build_fresh_temporal_base(
            dino_weights,
            input_resolution=input_resolution,
            model_initialization_order=args.model_initialization_order,
        )
        initial_tactile_head_sha256 = module_state_sha256(base_model.tactile_head)
        if args.temporal_architecture == "fullgrid6144_bidirectional_v5":
            fullgrid_pool = base_model.tactile_head.decoder[0]
            patch_channels = getattr(fullgrid_pool, "output_channels", None)
            if patch_channels is None:
                raise RuntimeError(
                    "FullGrid6144 requires a decoder pool that exposes "
                    "output_channels"
                )
            adapter = FullGrid6144SpatiotemporalFusionV5(
                patch_channels=int(patch_channels),
                clip_length=args.clip_length,
                grid_height=expected_shape[1],
                grid_width=expected_shape[2],
                spatial_layers=args.spatial_layers,
                spatial_heads=args.spatial_heads,
                spatial_ffn_ratio=args.spatial_ffn_ratio,
                temporal_layers=args.clip_layers,
                temporal_heads=args.clip_heads,
                temporal_ffn_ratio=args.clip_ffn_ratio,
                nominal_fps=args.nominal_fps,
                residual_dropout=args.clip_residual_dropout,
                layer_scale_init=args.clip_layer_scale_init,
            )
            initial_fusion_sha256 = module_state_sha256(adapter)
            initial_spatial_module_sha256 = module_state_sha256(
                adapter.spatial_blocks
            )
            initial_temporal_module_sha256 = module_state_sha256(
                adapter.temporal_blocks
            )
            initial_fast_writer_sha256 = initial_fusion_sha256
        elif args.temporal_architecture == "onlinehmr_patch_kv_v6":
            fullgrid_pool = base_model.tactile_head.decoder[0]
            patch_channels = getattr(fullgrid_pool, "output_channels", None)
            if patch_channels is None:
                raise RuntimeError(
                    "OnlineHMR patch memory requires a FullGrid pool that "
                    "exposes output_channels"
                )
            adapter = OnlineHMRPatchKVFusionV6(
                patch_channels=int(patch_channels),
                hidden_dim=args.onlinehmr_hidden_dim,
                clip_length=args.clip_length,
                grid_height=expected_shape[1],
                grid_width=expected_shape[2],
                layers=args.clip_layers,
                heads=args.clip_heads,
                ffn_ratio=args.clip_ffn_ratio,
                max_memory_frames=args.onlinehmr_memory_frames,
                nominal_fps=args.nominal_fps,
                residual_dropout=args.clip_residual_dropout,
            )
            initial_fusion_sha256 = module_state_sha256(adapter)
            initial_temporal_module_sha256 = module_state_sha256(adapter.blocks)
            initial_fast_writer_sha256 = initial_fusion_sha256
        elif clip_trunk:
            adapter = CausalClipTransformerFusionV4(
                grid_channels=expected_shape[0],
                hidden_channels=args.hidden_channels,
                clip_length=args.clip_length,
                grid_height=expected_shape[1],
                grid_width=expected_shape[2],
                layers=args.clip_layers,
                heads=args.clip_heads,
                ffn_ratio=args.clip_ffn_ratio,
                nominal_fps=args.nominal_fps,
                feature_rms_budget=args.feature_rms_budget,
            )
            initial_fast_writer_sha256 = module_state_sha256(adapter)
            initial_fusion_sha256 = initial_fast_writer_sha256
            initial_temporal_module_sha256 = initial_fast_writer_sha256
        else:
            adapter = HierarchicalTemporalMemoryFusionV3(
                grid_channels=expected_shape[0],
                hidden_channels=args.hidden_channels,
                history_lags=args.history_lags,
                nominal_fps=args.nominal_fps,
                patch_size=16,
                search_window=args.search_window,
                match_temperature=args.match_temperature,
                null_similarity=args.null_similarity,
                medium_null_similarity=args.medium_null_similarity,
                feature_rms_budget=args.feature_rms_budget,
                medium_feature_rms_budget=args.medium_feature_rms_budget,
            )
            initial_fast_writer_sha256 = module_state_sha256(adapter.fast_writer)
        model_class = (
            OnlineHMRPatchKVTemporalMainTrunkV6
            if args.temporal_architecture == "onlinehmr_patch_kv_v6"
            else FullGrid6144TemporalMainTrunkV5
            if args.temporal_architecture == "fullgrid6144_bidirectional_v5"
            else TemporalClipMainTrunkV4
            if clip_trunk
            else TemporalMainTrunkV3
        )
        model = model_class(
            base_model,
            palm_vertex_indices,
            adapter,
            online_encoder_chunk_size=args.online_encoder_chunk_size,
        )
        if args.temporal_architecture in TWIN_HEAD_ARCHITECTURES:
            initial_contact_head_sha256 = module_state_sha256(model.contact_head)
        print(
            "Fresh temporal initialization: "
            f"head={initial_tactile_head_sha256}, "
            f"fusion={initial_fusion_sha256 or initial_fast_writer_sha256}, "
            f"spatial={initial_spatial_module_sha256 or 'n/a'}, "
            f"temporal={initial_temporal_module_sha256 or 'n/a'}, "
            f"contact={initial_contact_head_sha256 or 'n/a'}, "
            f"order={args.model_initialization_order}",
            flush=True,
        )
        decoder_metadata = {
            "grid_channels": expected_shape[0],
            "grid_size": list(expected_shape[1:]),
            "input_resolution": list(input_resolution),
            "pool_layout": "fullgrid32",
            "pool_output_channels": 32,
            "decoder_hidden_dim": 512,
        }
    elif args.temporal_architecture == "local_memory_v2":
        adapter = TemporalLocalMemoryFusionV2(
            grid_channels=expected_shape[0],
            hidden_channels=args.hidden_channels,
            history_lags=args.history_lags,
            nominal_fps=args.nominal_fps,
            patch_size=16,
            search_window=args.search_window,
            match_temperature=args.match_temperature,
            null_similarity=args.null_similarity,
            feature_rms_budget=args.feature_rms_budget,
        )
        model = TemporalGridTactileModel(
            decoder,
            palm_vertex_indices,
            adapter,
            online_encoder=online_encoder,
            online_encoder_chunk_size=args.online_encoder_chunk_size,
        )
    else:
        adapter = TemporalGridAdapterV1(
            grid_channels=expected_shape[0],
            hidden_channels=args.hidden_channels,
            history_lags=args.history_lags,
            nominal_fps=args.nominal_fps,
            temporal_kernel_size=args.temporal_kernel_size,
            feature_rms_budget=args.feature_rms_budget,
        )
        model = TemporalGridTactileModel(
            decoder,
            palm_vertex_indices,
            adapter,
            online_encoder=online_encoder,
            online_encoder_chunk_size=args.online_encoder_chunk_size,
        )
    model_config = {
        "temporal_architecture": args.temporal_architecture,
        "fusion": adapter.config(),
        "decoder": decoder_metadata,
        "palm_vertex_indices": palm_vertex_indices.tolist(),
        "history_source": args.history_source,
        "data_mode": args.data_mode,
        "fresh_trunk": fresh_trunk,
        "base_initialization": "from_scratch" if fresh_trunk else "checkpoint",
        "model_initialization_order": (
            args.model_initialization_order
            if fresh_trunk
            else decoder_metadata.get(
                "model_initialization_order", CANONICAL_MODEL_INITIALIZATION_ORDER
            )
        ),
        "initial_tactile_head_sha256": initial_tactile_head_sha256,
        "initial_fast_writer_sha256": initial_fast_writer_sha256,
        "initial_fusion_sha256": initial_fusion_sha256,
        "initial_spatial_module_sha256": initial_spatial_module_sha256,
        "initial_temporal_module_sha256": initial_temporal_module_sha256,
        "initial_contact_head_sha256": initial_contact_head_sha256,
        "contact_head": (
            {
                "topology": "independent_dense_v2_decoder_tail",
                "pressure_threshold": float(args.contact_pressure_threshold),
                "loss_weight": float(args.contact_loss_weight),
                "jaccard_weight": float(args.contact_jaccard_weight),
                "gates_tactile": False,
            }
            if args.temporal_architecture in TWIN_HEAD_ARCHITECTURES
            else {}
        ),
        "worker_seed_mode": "lightning_legacy",
        "hdf5_sample_order": "legacy_sample_dir_hand",
        "crop_pipeline": "legacy_square_center",
        "optimizer_backend_mode": "legacy_default",
        "seed": int(args.seed),
        "batch_size_per_gpu": int(args.batch_size),
        "accumulate_grad_batches": int(args.accumulate_grad_batches),
        "global_batch_size": int(
            args.batch_size * len(gpu_ids) * args.accumulate_grad_batches
        ),
        "effective_frames_per_gpu": int(
            args.batch_size * (args.clip_length if clip_trunk else 1)
        ),
        "effective_global_frame_batch": int(
            args.batch_size
            * (args.clip_length if clip_trunk else 1)
            * len(gpu_ids)
            * args.accumulate_grad_batches
        ),
        "base_model": {
            "tactile_head_type": "dense_v2_dino_rezero",
            "backbone_feature_layers": [8, 16, 24, 32],
            "dino_residual_max_scale": 0.10,
            "dino_residual_rms_budget": 0.50,
            "pool_layout": "fullgrid32",
            "decoder_dropout_scale": 1.0,
            "input_resolution": list(input_resolution),
            "pool_output_channels": 32,
            "decoder_hidden_dim": 512,
            "model_initialization_order": args.model_initialization_order,
        }
        if fresh_trunk
        else {},
    }
    resume_path = args.resume_from_checkpoint
    if not resume_path and args.auto_resume and (checkpoint_dir / "resume.ckpt").is_file():
        resume_path = str(checkpoint_dir / "resume.ckpt")
    run_id_file = checkpoint_dir / "wandb_run_id.txt"
    resume_run = ""
    if resume_path:
        resume_payload = torch.load(resume_path, map_location="cpu")
        resume_run = str(resume_payload.get("wandb_run_id") or "")
    wandb_run_id = resume_run or (
        run_id_file.read_text().strip()
        if run_id_file.is_file()
        else uuid.uuid4().hex[:8]
    )
    run_id_file.write_text(wandb_run_id + "\n", encoding="utf-8")
    module = TemporalGridTrainingModule(
        model,
        model_config=model_config,
        data_config=data_config,
        loss_config=loss_config,
        base_checkpoint=args.base_checkpoint,
        dino_weights=dino_weights,
        output_dir=str(output_dir),
        history_source=args.history_source,
        temporal_architecture=args.temporal_architecture,
        learning_rate=args.lr * len(gpu_ids) if fresh_trunk else args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        tactile_loss_scale=args.tactile_loss_scale,
        history_reset_probability=args.history_reset_probability,
        reset_consistency_weight=args.reset_consistency_weight,
        counterfactual_identity_weight=args.counterfactual_identity_weight,
        base_preservation_weight=args.base_preservation_weight,
        contact_loss_weight=args.contact_loss_weight,
        contact_jaccard_weight=args.contact_jaccard_weight,
        contact_pressure_threshold=args.contact_pressure_threshold,
        contact_gradient_audit_every=args.contact_gradient_audit_every,
        fresh_trunk=fresh_trunk,
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
    val_sampler = ExactValidationSampler(len(val_dataset), rank, len(gpu_ids))
    common = {
        "pin_memory": True,
        "worker_init_fn": (
            initialize_worker_historical_lightning_seed
            if fresh_trunk
            else initialize_worker_parent_death_signal
        ),
    }
    train_kwargs = {
        **common,
        "batch_size": args.batch_size,
        "sampler": train_sampler,
        "drop_last": True,
        "num_workers": args.num_workers,
    }
    val_kwargs = {
        **common,
        "batch_size": args.val_batch_size,
        "sampler": val_sampler,
        "num_workers": args.val_num_workers,
    }
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
    (output_dir / "model_config.json").write_text(
        json.dumps(model_config, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "schema": "tactile_temporal_training_provenance_v3",
                "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "model_config": model_config,
                "data_config": data_config,
                "dino_weights": dino_weights,
                "dino_weights_sha256": (
                    file_sha256(dino_weights) if dino_weights else ""
                ),
                "base_checkpoint": args.base_checkpoint,
                "base_checkpoint_sha256": (
                    file_sha256(args.base_checkpoint) if args.base_checkpoint else ""
                ),
                "fresh_training": fresh_trunk,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
        input_summary = (
            "feature_cache_generated=false"
            if args.data_mode == "online"
            else "legacy feature-cache input"
        )
        clip_summary = ""
        if clip_trunk:
            clip_summary = (
                f", source_frames={data_config['train_source_frame_count']:,}, "
                f"mean_valid_frames/clip="
                f"{data_config['train_mean_valid_frames_per_clip']:.3f}/"
                f"{args.clip_length}, padding="
                f"{100.0 * data_config['train_clip_padding_fraction']:.2f}%, "
                f"steps/rank/epoch={len(train_loader):,}"
            )
        print(
            f"Temporal image-grid fusion {args.data_mode} training: "
            f"architecture={args.temporal_architecture}, "
            f"source={args.history_source}, train_pairs={len(train_dataset):,}, "
            f"val_pairs={len(val_dataset):,}, lags={args.history_lags}, "
            f"batch/rank={args.batch_size}, workers={args.num_workers}/"
            f"{args.val_num_workers}, fresh_trunk={fresh_trunk}, "
            f"base_lr={args.lr:g}, effective_lr="
            f"{(args.lr * len(gpu_ids) if fresh_trunk else args.lr):g}, "
            f"initial_head={initial_tactile_head_sha256 or 'checkpoint'}, "
            f"{input_summary}{clip_summary}",
            flush=True,
        )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=len(gpu_ids),
        strategy="ddp" if len(gpu_ids) > 1 else "auto",
        precision=args.precision,
        max_epochs=args.epochs,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        use_distributed_sampler=False,
        num_sanity_val_steps=2,
        check_val_every_n_epoch=1,
        log_every_n_steps=20,
    )
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_path or None)


if __name__ == "__main__":
    main()
