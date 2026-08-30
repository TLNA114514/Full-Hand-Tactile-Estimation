"""Frozen RGB pressure/contact model with a prior-aware contact-only adapter."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from hamer_tactile_ft.hamer_tactile import DinoTactileModel, _canonical_mesh_assets

from .prior_adapters import (
    SUPPORTED_PRIOR_CONTROLS,
    apply_prior_control,
    per_sample_spatial_permutations,
)
from .selector_prior_adapters import (
    DepthAnchorQuerySelector,
    DepthAnchorResidual,
    DepthMappingRectifier,
    SUPPORTED_SELECTOR_PRIOR_ADAPTERS,
    VLMGlobalContactCalibrator,
)


class _CacheOnlyBackbone(nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "This selector-prior model is cache-only; provide frozen selector "
            "evidence (decoder bottleneck/base logits, contact neck, and anchor logits)"
        )


class PriorSelectorModel(nn.Module):
    """Add external evidence to a frozen Binary Grid contact selector.

    The pressure logits are copied from the frozen selector checkpoint and are
    never decoded from prior-modified features.  This is the central contract:
    Stage D1 measures whether a prior improves contact perception, not whether
    it can globally turn pressure up or down.
    """

    def __init__(
        self,
        selector_model: DinoTactileModel,
        *,
        adapter_type: str,
        prior_dim: int,
        prior_control: str = "real",
        control_seed: int = 521,
        feature_rms_budget: float = 0.05,
        prior_dropout: float = 0.0,
        depth_hidden_channels: int = 128,
        depth_modulation_max_scale: float = 0.10,
        anchor_residual_max_logit: float = 2.0,
        anchor_query_dim: int = 128,
        anchor_query_heads: int = 4,
        anchor_query_layers: int = 2,
        vlm_rank: int = 32,
        vlm_residual_max_logit: float = 1.0,
    ):
        super().__init__()
        adapter_type = str(adapter_type).strip().lower()
        if adapter_type not in SUPPORTED_SELECTOR_PRIOR_ADAPTERS:
            raise ValueError(
                f"Unsupported selector prior {adapter_type!r}; "
                f"choose one of {SUPPORTED_SELECTOR_PRIOR_ADAPTERS}"
            )
        if float(prior_dropout) != 0.0:
            raise ValueError(
                "Selector-prior adapters require prior_dropout=0 so paired "
                "causal subtraction uses identical deterministic paths"
            )
        prior_control = str(prior_control).strip().lower()
        if prior_control not in SUPPORTED_PRIOR_CONTROLS:
            raise ValueError(f"Unsupported prior control {prior_control!r}")
        if not isinstance(selector_model, DinoTactileModel):
            raise TypeError("selector_model must be DinoTactileModel")
        self.base_model = selector_model
        self.adapter_type = adapter_type
        self.prior_dim = int(prior_dim)
        self.default_control = prior_control
        self.control_seed = int(control_seed)
        self._validate_selector_contract()
        self._freeze_base()

        mapping = self._mapping
        common = {
            "prior_channels": self.prior_dim,
            "neck_channels": mapping.neck_channels,
            "hidden_channels": int(depth_hidden_channels),
            "dropout": float(prior_dropout),
        }
        if adapter_type == "depth_mapping_rectifier":
            self.prior_adapter = DepthMappingRectifier(
                **common,
                feature_rms_budget=float(feature_rms_budget),
                modulation_max_scale=float(depth_modulation_max_scale),
            )
        elif adapter_type == "depth_anchor_residual":
            self.prior_adapter = DepthAnchorResidual(
                **common,
                grid_size=mapping.grid_size,
                anchor_count=mapping.anchor_count,
                hidden_dim=self.base_model.support_selector_hidden_dim,
                residual_max_logit=float(anchor_residual_max_logit),
            )
        elif adapter_type == "depth_anchor_query":
            canonical_vertices, _ = _canonical_mesh_assets()
            anchor_coordinates = canonical_vertices.index_select(
                0, mapping.anchor_vertex_indices.cpu()
            )
            self.prior_adapter = DepthAnchorQuerySelector(
                **common,
                anchor_coordinates=anchor_coordinates,
                grid_size=mapping.grid_size,
                query_dim=int(anchor_query_dim),
                heads=int(anchor_query_heads),
                layers=int(anchor_query_layers),
                residual_max_logit=float(anchor_residual_max_logit),
            )
        else:
            self.prior_adapter = VLMGlobalContactCalibrator(
                prior_dim=self.prior_dim,
                rank=int(vlm_rank),
                residual_max_logit=float(vlm_residual_max_logit),
                dropout=float(prior_dropout),
            )
        self._last_diagnostics: Dict[str, torch.Tensor] = {}
        self.train(True)

    @property
    def tactile_dim(self) -> int:
        return int(self.base_model.tactile_dim)

    @property
    def input_resolution(self):
        return self.base_model.input_resolution

    @property
    def is_depth(self) -> bool:
        return self.adapter_type.startswith("depth_")

    @property
    def _support(self):
        return self.base_model.tactile_head.support_selector

    @property
    def _mapping(self):
        return self._support.selector

    def _validate_selector_contract(self) -> None:
        if self.base_model.tactile_head_type != "dense_v2_dino_support_selector":
            raise ValueError("Prior selector requires a support-selector checkpoint")
        if self.base_model.support_selector_mode != "contact":
            raise ValueError("Prior selector requires the binary contact checkpoint")
        if self.base_model.support_selector_architecture != "spatial_mlp":
            raise ValueError("Prior selector requires architecture=spatial_mlp")
        if self.base_model.support_selector_feature_source != "rezero_grid":
            raise ValueError("Prior selector requires feature_source=rezero_grid")
        if tuple(self.base_model.support_selector_thresholds) != (
            float(self.base_model.support_selector_contact_min),
        ):
            raise ValueError("Prior selector requires exactly one contact threshold")
        support = getattr(self.base_model.tactile_head, "support_selector", None)
        mapping = getattr(support, "selector", None)
        if mapping is None or not hasattr(mapping, "spatial_neck"):
            raise TypeError("Selector checkpoint does not expose the spatial mapping")
        if getattr(support, "raw_fusion", None) is not None:
            raise ValueError("Prior selector does not support a raw-DINO selector")
        if getattr(support, "base_conditioner", None) is not None:
            raise ValueError("Prior selector requires the independent Binary Grid head")

    def _freeze_base(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        self.prior_adapter.train(mode)
        return self

    def trainable_parameters(self):
        return self.prior_adapter.parameters()

    def assert_base_frozen(self) -> None:
        trainable = [
            name for name, parameter in self.base_model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"Frozen selector has trainable parameters: {trainable[:5]}")
        if self.base_model.training:
            raise RuntimeError("Frozen selector is in training mode")

    def disable_online_backbone(self) -> None:
        self.base_model.backbone = _CacheOnlyBackbone()
        self.base_model.backbone.eval()

    def _online_grid(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) != tuple(self.input_resolution):
            raise ValueError(
                f"Selector expects input {self.input_resolution}, got {tuple(image.shape[-2:])}"
            )
        with torch.no_grad():
            levels = self.base_model._extract_tactile_features(image)
            grid = self.base_model.tactile_head._fuse(levels)
        return grid.detach()

    def _scatter_anchor_logits(self, anchor_logits: torch.Tensor) -> torch.Tensor:
        mapping = self._mapping
        per_vertex = anchor_logits[:, mapping.vertex_anchor_indices]
        weights = mapping.vertex_anchor_weights.to(per_vertex)[None]
        logits = (per_vertex * weights).sum(dim=2)
        return logits * mapping.valid_palm_mask.to(logits)[None]

    def _base_evidence(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        grid = batch.get("frozen_base_grid")
        bottleneck = batch.get("frozen_base_bottleneck")
        pressure_logits = batch.get("frozen_base_logits")
        contact_logits = batch.get("frozen_contact_logits")
        neck = batch.get("frozen_contact_neck")
        anchor_logits = batch.get("frozen_contact_anchor_logits")
        has_cached_evidence = (
            neck is not None
            and anchor_logits is not None
            and (pressure_logits is not None or bottleneck is not None)
        )
        if bottleneck is not None:
            bottleneck = bottleneck.detach()
            expected_hidden_dim = int(self.base_model.decoder_hidden_dim)
            if bottleneck.ndim != 2 or bottleneck.shape[1] != expected_hidden_dim:
                raise ValueError(
                    "frozen_base_bottleneck must be "
                    f"[B,{expected_hidden_dim}], got "
                    f"{tuple(bottleneck.shape)}"
                )
        if grid is not None:
            grid = grid.detach()
            if grid.ndim != 4 or grid.shape[1] != 256:
                raise ValueError(
                    f"frozen_base_grid must be [B,256,H,W], got {tuple(grid.shape)}"
                )
        elif not has_cached_evidence:
            image = batch.get("img")
            if image is None:
                missing = [
                    name
                    for name, value in (
                        (
                            "frozen_base_bottleneck/frozen_base_logits",
                            bottleneck if bottleneck is not None else pressure_logits,
                        ),
                        ("frozen_contact_neck", neck),
                        ("frozen_contact_anchor_logits", anchor_logits),
                    )
                    if value is None
                ]
                raise KeyError(
                    "Batch needs img/frozen_base_grid or the complete frozen selector "
                    f"evidence cache; missing={missing}"
                )
            grid = self._online_grid(image)

        with torch.no_grad():
            if pressure_logits is None:
                if bottleneck is not None:
                    pressure_logits = self.base_model.tactile_head.decoder[5:](
                        bottleneck
                    )
                else:
                    pressure_logits = self.base_model.tactile_head.decoder(grid)
            if neck is None:
                neck = self._mapping.spatial_neck(grid)
            if anchor_logits is None:
                anchor_logits = self._mapping.anchor_decoder(neck.flatten(1)).reshape(
                    grid.shape[0], self._mapping.anchor_count
                )
            if contact_logits is None:
                contact_logits = self._scatter_anchor_logits(anchor_logits)
        batch_size = int(pressure_logits.shape[0])
        expected_neck = (
            batch_size,
            self._mapping.neck_channels,
            *self._mapping.grid_size,
        )
        if tuple(neck.shape) != expected_neck:
            raise ValueError(
                f"frozen_contact_neck must be {expected_neck}, got {tuple(neck.shape)}"
            )
        if tuple(anchor_logits.shape) != (batch_size, self._mapping.anchor_count):
            raise ValueError("frozen_contact_anchor_logits has an incompatible shape")
        if tuple(contact_logits.shape) != (batch_size, self.tactile_dim):
            raise ValueError("frozen_contact_logits has an incompatible shape")
        if tuple(pressure_logits.shape) != (batch_size, self.tactile_dim):
            raise ValueError("frozen_base_logits has an incompatible shape")
        evidence = {
            "pressure_logits": pressure_logits.detach(),
            "neck": neck.detach(),
            "anchor_logits": anchor_logits.detach(),
            "contact_logits": contact_logits.detach(),
        }
        if grid is not None:
            evidence["grid"] = grid.detach()
        return evidence

    def _resolve_prior(
        self,
        batch: Mapping[str, Any],
        prior: Optional[torch.Tensor],
        control: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        key = "depth_prior" if self.is_depth else "vlm_prior"
        availability_key = "depth_available" if self.is_depth else "vlm_available"
        if prior is None:
            prior = batch.get(key)
        if prior is None:
            raise KeyError(f"Batch is missing {key}")
        control_index = batch.get("prior_control_index")
        if control == "spatial_shuffle" and control_index is None and prior.ndim == 4:
            sample_uids = batch.get("sample_uid")
            if sample_uids is None:
                raise KeyError("sample_uid is required for spatial_shuffle")
            control_index = per_sample_spatial_permutations(
                [str(value) for value in sample_uids],
                prior.shape[-2] * prior.shape[-1],
                seed=self.control_seed,
                device=prior.device,
            )
        controlled = apply_prior_control(
            prior,
            control,
            alternate_prior=batch.get("alternate_prior"),
            control_index=control_index,
            seed=self.control_seed,
        )
        valid = batch.get("depth_valid") if self.is_depth else None
        if valid is not None and control != "real":
            alternate_valid = batch.get("alternate_depth_valid")
            valid = apply_prior_control(
                valid[:, None] if valid.ndim == 3 else valid,
                control,
                alternate_prior=(
                    alternate_valid[:, None]
                    if alternate_valid is not None and alternate_valid.ndim == 3
                    else alternate_valid
                ),
                control_index=control_index,
                seed=self.control_seed,
            )
        return controlled, batch.get(availability_key), valid

    def forward_step(
        self,
        batch: Mapping[str, Any],
        *,
        prior: Optional[torch.Tensor] = None,
        prior_control: Optional[str] = None,
        cached_evidence: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        self.assert_base_frozen()
        control = self.default_control if prior_control is None else str(prior_control).lower()
        evidence = (
            dict(cached_evidence) if cached_evidence is not None else self._base_evidence(batch)
        )
        controlled_prior, availability, depth_valid = self._resolve_prior(
            batch, prior, control
        )
        if self.adapter_type == "depth_mapping_rectifier":
            fused_neck, diagnostics, auxiliary = self.prior_adapter(
                evidence["neck"],
                controlled_prior,
                valid=depth_valid,
                availability=availability,
            )
            fused_anchor = self._mapping.anchor_decoder(fused_neck.flatten(1)).reshape(
                fused_neck.shape[0], self._mapping.anchor_count
            )
            anchor_residual = fused_anchor - evidence["anchor_logits"]
        elif self.adapter_type == "depth_anchor_residual":
            anchor_residual, diagnostics, auxiliary = self.prior_adapter(
                evidence["neck"],
                controlled_prior,
                valid=depth_valid,
                availability=availability,
            )
            fused_anchor = evidence["anchor_logits"] + anchor_residual
            false_high_anchor_logits = -fused_anchor
        elif self.adapter_type == "depth_anchor_query":
            anchor_vertices = self._mapping.anchor_vertex_indices
            base_pressure = torch.sigmoid(evidence["pressure_logits"]).index_select(
                1, anchor_vertices
            )
            (
                anchor_residual,
                false_high_anchor_logits,
                diagnostics,
                auxiliary,
            ) = self.prior_adapter(
                evidence["neck"],
                evidence["anchor_logits"],
                base_pressure,
                controlled_prior,
                valid=depth_valid,
                availability=availability,
            )
            fused_anchor = evidence["anchor_logits"] + anchor_residual
        else:
            anchor_vertices = self._mapping.anchor_vertex_indices
            base_pressure = torch.sigmoid(evidence["pressure_logits"]).index_select(
                1, anchor_vertices
            )
            anchor_residual, diagnostics, auxiliary = self.prior_adapter(
                evidence["anchor_logits"],
                base_pressure,
                controlled_prior,
                availability=availability,
            )
            fused_anchor = evidence["anchor_logits"] + anchor_residual
            false_high_anchor_logits = -fused_anchor
        if self.adapter_type == "depth_mapping_rectifier":
            false_high_anchor_logits = -fused_anchor
        fused_contact = self._scatter_anchor_logits(fused_anchor)
        false_high_logits = self._scatter_anchor_logits(false_high_anchor_logits)
        vertex_residual = fused_contact - evidence["contact_logits"]
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "contact_delta_rms": vertex_residual.detach().float().pow(2).mean().sqrt(),
                "contact_delta_mean": vertex_residual.detach().float().mean(),
                "contact_delta_up_fraction": (vertex_residual.detach() > 0).float().mean(),
                "contact_delta_down_fraction": (vertex_residual.detach() < 0).float().mean(),
            }
        )
        self._last_diagnostics = {
            name: value.detach().float() for name, value in diagnostics.items()
        }
        pressure_logits = evidence["pressure_logits"]
        prior_available = availability
        if prior_available is None:
            prior_available = pressure_logits.new_ones((pressure_logits.shape[0],))
        return {
            "pred_logits": pressure_logits,
            "pred_tactile": torch.sigmoid(pressure_logits),
            "base_pressure_logits": pressure_logits,
            "base_pred_tactile": torch.sigmoid(pressure_logits),
            "base_contact_logits": evidence["contact_logits"],
            "fused_contact_logits": fused_contact,
            "false_high_logits": false_high_logits,
            "prior_contact_residual": vertex_residual,
            "base_anchor_logits": evidence["anchor_logits"],
            "fused_anchor_logits": fused_anchor,
            "prior_aux_losses": auxiliary,
            "prior_control": control,
            "prior_available": prior_available,
            "prior_diagnostics": self._last_diagnostics,
            "_base_evidence": evidence,
        }

    def forward_paired(
        self,
        batch: Mapping[str, Any],
        *,
        control: str = "spatial_shuffle",
    ) -> Dict[str, Any]:
        primary_control = self.default_control
        real = self.forward_step(batch, prior_control=primary_control)
        counterfactual = self.forward_step(
            batch,
            prior_control=control,
            cached_evidence=real["_base_evidence"],
        )
        real["control_contact_logits"] = counterfactual["fused_contact_logits"]
        real["control_false_high_logits"] = counterfactual["false_high_logits"]
        real["control_contact_residual"] = counterfactual["prior_contact_residual"]
        real["control_identity_loss"] = counterfactual[
            "prior_contact_residual"
        ].float().square().mean()
        real["control_prior_diagnostics"] = counterfactual["prior_diagnostics"]
        real["control_prior_aux_losses"] = counterfactual["prior_aux_losses"]
        real["counterfactual_control"] = str(control)
        return real

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Dict[str, Any]:
        return self.forward_step(batch, **kwargs)

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_diagnostics
