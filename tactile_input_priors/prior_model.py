"""Frozen-base tactile model with feature-level Depth or VLM adapters."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from hamer_tactile_ft.hamer_tactile import DinoTactileModel

from .prior_adapters import (
    AdapterConfig,
    DepthCausalFiLMAdapter,
    DepthLocalCrossAttentionAdapter,
    DepthSpatialRectificationAdapter,
    SUPPORTED_PRIOR_CONTROLS,
    VLMLowRankModulationAdapter,
    apply_prior_control,
    detached_diagnostics,
    per_sample_spatial_permutations,
)


DEPTH_PRIOR_ADAPTERS = (
    "depth_spatial",
    "depth_causal_film",
    "depth_local_xattn",
)
SUPPORTED_PRIOR_ADAPTERS = (*DEPTH_PRIOR_ADAPTERS, "vlm_lowrank")


class _CacheOnlyBackbone(nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "This prior model was placed in cache-only mode; provide frozen_base_grid "
            "or frozen_base_bottleneck instead of an RGB image"
        )


def smooth_logit_delta(
    raw_fused_logits: torch.Tensor,
    base_logits: torch.Tensor,
    maximum_delta: float,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    zero_mean: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smoothly cap the adapter-induced change while preserving its sign."""

    if raw_fused_logits.shape != base_logits.shape:
        raise ValueError(
            f"raw/base logit shapes differ: {tuple(raw_fused_logits.shape)} vs "
            f"{tuple(base_logits.shape)}"
        )
    if float(maximum_delta) <= 0.0:
        raise ValueError("maximum logit delta must be positive")
    raw_delta = raw_fused_logits - base_logits
    mask = None
    if zero_mean:
        if valid_mask is None:
            raise ValueError("zero-mean logit residual requires a valid palm mask")
        mask = valid_mask.to(device=raw_delta.device, dtype=raw_delta.dtype)
        if mask.ndim == 1:
            mask = mask[None].expand(raw_delta.shape[0], -1)
        elif mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask.expand(raw_delta.shape[0], -1)
        if mask.shape != raw_delta.shape:
            raise ValueError(
                f"palm mask shape {tuple(mask.shape)} does not match logits {tuple(raw_delta.shape)}"
            )
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        raw_delta = (raw_delta - (raw_delta * mask).sum(dim=1, keepdim=True) / denominator) * mask
    bounded_delta = float(maximum_delta) * torch.tanh(raw_delta / float(maximum_delta))
    if zero_mean:
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        bounded_delta = bounded_delta - (
            bounded_delta * mask
        ).sum(dim=1, keepdim=True) / denominator
        bounded_delta = bounded_delta * mask
        maximum = bounded_delta.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(float(maximum_delta) / maximum, max=1.0)
        bounded_delta = bounded_delta * scale
    return base_logits + bounded_delta, bounded_delta


class FrozenBasePriorModel(nn.Module):
    """Wrap ``DinoTactileModel`` without changing its source implementation.

    The base is permanently frozen and kept in evaluation mode.  Gradients can
    still pass through its decoder operations to the adapter input, but no base
    parameter is part of optimization.  Cached base features may be supplied in
    the batch to bypass the frozen DINO computation.

    Batch/cache keys:

    - ``img``: normalized RGB input for an online base forward.
    - ``frozen_base_grid``: cached fused RGB grid ``[B,256,H,W]``.
    - ``frozen_base_bottleneck``: cached decoder bottleneck (VLM only).
    - ``frozen_base_logits``: cached base logits.
    - ``depth_prior`` and optional ``depth_valid``/``depth_available``.
    - ``vlm_prior`` and optional ``vlm_available``.
    - ``alternate_prior``: paired wrong-frame/query prior.
    - ``prior_control_index``: explicit sample or spatial permutation.
    """

    def __init__(
        self,
        base_model: DinoTactileModel,
        *,
        adapter_type: str,
        prior_dim: int,
        feature_rms_budget: float = 0.05,
        logit_delta_max: float = 0.50,
        prior_dropout: float = 0.10,
        control_seed: int = 521,
        depth_hidden_channels: int = 128,
        depth_modulation_max_scale: float = 0.10,
        depth_attention_heads: int = 4,
        depth_attention_window: int = 5,
        zero_mean_logit_residual: bool = False,
        vlm_rank: int = 32,
        default_control: str = "real",
    ):
        super().__init__()
        if not isinstance(base_model, DinoTactileModel):
            raise TypeError(
                "base_model must be an instance of hamer_tactile_ft.hamer_tactile."
                "DinoTactileModel"
            )
        adapter_type = str(adapter_type).strip().lower()
        if adapter_type not in SUPPORTED_PRIOR_ADAPTERS:
            raise ValueError(
                f"Unsupported adapter_type={adapter_type!r}; "
                f"choose one of {SUPPORTED_PRIOR_ADAPTERS}"
            )
        default_control = str(default_control).strip().lower()
        if default_control not in SUPPORTED_PRIOR_CONTROLS:
            raise ValueError(
                f"Unsupported default_control={default_control!r}; "
                f"choose one of {SUPPORTED_PRIOR_CONTROLS}"
            )
        if str(base_model.pool_layout) != "fullgrid32":
            raise ValueError(
                "Feature adapters currently require a FullGrid32 tactile base"
            )
        self.base_model = base_model
        self.adapter_type = adapter_type
        self.prior_dim = int(prior_dim)
        self.logit_delta_max = float(logit_delta_max)
        self.control_seed = int(control_seed)
        self.default_control = default_control
        self.zero_mean_logit_residual = bool(zero_mean_logit_residual)
        self._last_prior_diagnostics: Dict[str, torch.Tensor] = {}
        self._base_frozen_verified = False

        self._freeze_base()
        self._validate_decoder_contract()
        config = AdapterConfig(
            feature_rms_budget=feature_rms_budget,
            prior_dropout=prior_dropout,
            control_seed=control_seed,
        )
        if self.adapter_type == "depth_spatial":
            self.prior_adapter = DepthSpatialRectificationAdapter(
                prior_channels=self.prior_dim,
                feature_channels=self._grid_channels,
                hidden_channels=depth_hidden_channels,
                modulation_max_scale=depth_modulation_max_scale,
                config=config,
            )
        elif self.adapter_type == "depth_causal_film":
            self.prior_adapter = DepthCausalFiLMAdapter(
                prior_channels=self.prior_dim,
                feature_channels=self._grid_channels,
                hidden_channels=depth_hidden_channels,
                modulation_max_scale=depth_modulation_max_scale,
                config=config,
            )
        elif self.adapter_type == "depth_local_xattn":
            self.prior_adapter = DepthLocalCrossAttentionAdapter(
                prior_channels=self.prior_dim,
                feature_channels=self._grid_channels,
                hidden_channels=depth_hidden_channels,
                attention_heads=depth_attention_heads,
                window_size=depth_attention_window,
                config=config,
            )
        else:
            self.prior_adapter = VLMLowRankModulationAdapter(
                prior_dim=self.prior_dim,
                feature_dim=self._bottleneck_dim,
                rank=vlm_rank,
                config=config,
            )
        self.train(True)

    @property
    def tactile_dim(self) -> int:
        return int(self.base_model.tactile_dim)

    @property
    def input_resolution(self):
        return self.base_model.input_resolution

    @property
    def tactile_head(self):
        return self.base_model.tactile_head

    def _freeze_base(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.base_model.eval()
        self._base_frozen_verified = False

    def _validate_decoder_contract(self) -> None:
        head = self.base_model.tactile_head
        if not hasattr(head, "decoder") or not isinstance(head.decoder, nn.Sequential):
            raise TypeError("The tactile head must expose an nn.Sequential decoder")
        decoder = head.decoder
        if len(decoder) != 8:
            raise ValueError(
                "Expected the DenseV2 decoder layout "
                "pool/dropout/linear/norm/GELU/dropout/residual/output"
            )
        if not isinstance(decoder[2], nn.Linear) or decoder[2].out_features < 1:
            raise ValueError(
                "DenseV2 decoder does not expose a positive-width bottleneck"
            )
        self._grid_channels = 256
        projection = getattr(head, "base_projection", None)
        if isinstance(projection, nn.Sequential) and isinstance(projection[0], nn.Conv2d):
            self._grid_channels = int(projection[0].out_channels)
        self._bottleneck_dim = int(decoder[2].out_features)

    def train(self, mode: bool = True):
        super().train(mode)
        # nn.Module.train() recurses before returning. Restore the invariant
        # after every mode switch while leaving the adapter in the requested mode.
        self.base_model.eval()
        self.prior_adapter.train(mode)
        return self

    def trainable_parameters(self):
        return self.prior_adapter.parameters()

    def disable_online_backbone(self) -> None:
        """Release frozen DINO parameters when every sample has cached features."""

        self.base_model.backbone = _CacheOnlyBackbone()
        self.base_model.backbone.eval()

    def assert_base_frozen(self) -> None:
        if self._base_frozen_verified:
            return
        trainable = [name for name, value in self.base_model.named_parameters() if value.requires_grad]
        if trainable:
            raise RuntimeError(f"Frozen tactile base has trainable parameters: {trainable[:5]}")
        if self.base_model.training or any(
            module.training for module in self.base_model.modules()
        ):
            raise RuntimeError("Frozen tactile base or one of its submodules is in train mode")
        self._base_frozen_verified = True

    def _online_grid(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) != tuple(self.base_model.input_resolution):
            raise ValueError(
                f"Base expects input {self.base_model.input_resolution}, "
                f"got {tuple(image.shape[-2:])}"
            )
        with torch.no_grad():
            levels = self.base_model._extract_tactile_features(image)
            head = self.base_model.tactile_head
            if hasattr(head, "_fuse"):
                grid = head._fuse(levels)
            else:
                final_level = levels[-1] if isinstance(levels, (tuple, list)) else levels
                grid = head.base_projection(final_level)
        return grid.detach()

    def _decoder_bottleneck(self, grid: torch.Tensor) -> torch.Tensor:
        decoder = self.base_model.tactile_head.decoder
        value = grid
        for layer in decoder[:5]:
            value = layer(value)
        return value

    def _decoder_tail(self, bottleneck: torch.Tensor) -> torch.Tensor:
        value = bottleneck
        for layer in self.base_model.tactile_head.decoder[5:]:
            value = layer(value)
        return value

    def _resolve_prior(
        self,
        batch: Mapping[str, Any],
        prior: Optional[torch.Tensor],
        control: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        is_depth = self.adapter_type in DEPTH_PRIOR_ADAPTERS
        key = "depth_prior" if is_depth else "vlm_prior"
        available_key = (
            "depth_available" if is_depth else "vlm_available"
        )
        if prior is None:
            prior = batch.get(key)
        if prior is None:
            raise KeyError(f"Batch is missing required prior tensor {key!r}")
        alternate = batch.get("alternate_prior")
        control_index = batch.get("prior_control_index")
        if control == "spatial_shuffle" and control_index is None and prior.ndim == 4:
            sample_uids = batch.get("sample_uid")
            if sample_uids is None:
                raise KeyError(
                    "sample_uid is required for deterministic per-sample spatial shuffle"
                )
            control_index = per_sample_spatial_permutations(
                [str(value) for value in sample_uids],
                prior.shape[-2] * prior.shape[-1],
                seed=self.control_seed,
                device=prior.device,
            )
        controlled = apply_prior_control(
            prior,
            control,
            alternate_prior=alternate,
            control_index=control_index,
            seed=self.control_seed,
        )
        depth_valid = batch.get("depth_valid")
        if is_depth and depth_valid is not None:
            alternate_valid = batch.get("alternate_depth_valid")
            if depth_valid.ndim == 3:
                depth_valid = depth_valid[:, None]
            if alternate_valid is not None and alternate_valid.ndim == 3:
                alternate_valid = alternate_valid[:, None]
            depth_valid = apply_prior_control(
                depth_valid,
                control,
                alternate_prior=alternate_valid,
                control_index=control_index,
                seed=self.control_seed,
            )
        return controlled, batch.get(available_key), depth_valid

    def _base_grid_and_logits(
        self, batch: Mapping[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = batch.get("frozen_base_grid")
        if grid is None:
            image = batch.get("img")
            if image is None:
                raise KeyError("Batch needs either 'img' or 'frozen_base_grid'")
            grid = self._online_grid(image)
        else:
            grid = grid.detach()
        if grid.ndim != 4 or grid.shape[1] != self._grid_channels:
            raise ValueError(
                f"frozen base grid must be [B,{self._grid_channels},H,W], "
                f"got {tuple(grid.shape)}"
            )
        base_logits = batch.get("frozen_base_logits")
        if base_logits is None:
            with torch.no_grad():
                base_logits = self.base_model.tactile_head.decoder(grid)
        else:
            base_logits = base_logits.detach()
        if base_logits.ndim != 2 or base_logits.shape != (
            grid.shape[0],
            self.tactile_dim,
        ):
            raise ValueError(
                f"frozen base logits must be [B,{self.tactile_dim}], "
                f"got {tuple(base_logits.shape)}"
            )
        return grid, base_logits.detach()

    def _cached_vlm_base(
        self, batch: Mapping[str, Any]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        bottleneck = batch.get("frozen_base_bottleneck")
        base_logits = batch.get("frozen_base_logits")
        if bottleneck is None and base_logits is None:
            return None
        if bottleneck is None:
            raise KeyError(
                "The VLM cache path requires 'frozen_base_bottleneck'"
            )
        bottleneck = bottleneck.detach()
        if bottleneck.ndim != 2 or bottleneck.shape[1] != self._bottleneck_dim:
            raise ValueError(
                f"frozen bottleneck must be [B,{self._bottleneck_dim}], "
                f"got {tuple(bottleneck.shape)}"
            )
        if base_logits is None:
            with torch.no_grad():
                base_logits = self._decoder_tail(bottleneck)
        else:
            base_logits = base_logits.detach()
        if base_logits.ndim != 2 or base_logits.shape != (
            bottleneck.shape[0],
            self.tactile_dim,
        ):
            raise ValueError(
                f"frozen base logits must be [B,{self.tactile_dim}], "
                f"got {tuple(base_logits.shape)}"
            )
        return bottleneck, base_logits

    def forward_step(
        self,
        batch: Mapping[str, Any],
        train: bool = False,
        *,
        prior: Optional[torch.Tensor] = None,
        prior_control: Optional[str] = None,
        decode: bool = True,
    ) -> Dict[str, Any]:
        self.assert_base_frozen()
        control = self.default_control if prior_control is None else str(prior_control).lower()
        controlled_prior, availability, depth_valid = self._resolve_prior(
            batch, prior, control
        )
        if self.adapter_type in DEPTH_PRIOR_ADAPTERS:
            grid, base_logits = self._base_grid_and_logits(batch)
            fused_grid, diagnostics, auxiliary_losses = self.prior_adapter(
                grid,
                controlled_prior,
                valid=depth_valid,
                availability=availability,
            )
            feature_delta = fused_grid - grid
            fused_feature = fused_grid
        else:
            cached_base = self._cached_vlm_base(batch)
            if cached_base is None:
                grid, base_logits = self._base_grid_and_logits(batch)
                with torch.no_grad():
                    bottleneck = self._decoder_bottleneck(grid)
            else:
                bottleneck, base_logits = cached_base
            fused_bottleneck, diagnostics, auxiliary_losses = self.prior_adapter(
                bottleneck,
                controlled_prior,
                availability=availability,
            )
            feature_delta = fused_bottleneck - bottleneck
            fused_feature = fused_bottleneck

        if not decode:
            self._last_prior_diagnostics = detached_diagnostics(diagnostics)
            return {
                "feature_delta": feature_delta,
                "base_feature": grid if self.adapter_type in DEPTH_PRIOR_ADAPTERS else bottleneck,
                "prior_aux_losses": auxiliary_losses,
                "prior_control": control,
                "prior_diagnostics": self._last_prior_diagnostics,
            }

        if self.adapter_type in DEPTH_PRIOR_ADAPTERS:
            raw_fused_logits = self.base_model.tactile_head.decoder(fused_feature)
        else:
            raw_fused_logits = self._decoder_tail(fused_feature)

        fused_logits, bounded_logit_delta = smooth_logit_delta(
            raw_fused_logits,
            base_logits,
            self.logit_delta_max,
            valid_mask=batch.get("palm_mask"),
            zero_mean=self.zero_mean_logit_residual,
        )
        raw_logit_delta = raw_fused_logits - base_logits
        base_prediction = torch.sigmoid(base_logits)
        fused_prediction = torch.sigmoid(fused_logits)
        if not train:
            diagnostics.update(
                {
                    "logit_delta_rms_raw": raw_logit_delta.detach().float().pow(2).mean().sqrt(),
                    "logit_delta_rms_bounded": bounded_logit_delta.detach().float().pow(2).mean().sqrt(),
                    "logit_delta_abs_max": bounded_logit_delta.detach().float().abs().max(),
                    "logit_delta_saturation": (
                        raw_logit_delta.detach().float().abs()
                        > 3.0 * self.logit_delta_max
                    ).float().mean(),
                    "logit_delta_up_fraction": (
                        bounded_logit_delta.detach().float() > 0.0
                    ).float().mean(),
                    "logit_delta_down_fraction": (
                        bounded_logit_delta.detach().float() < 0.0
                    ).float().mean(),
                    "decoder_gain": (
                        bounded_logit_delta.detach().float().pow(2).mean().sqrt()
                        / diagnostics["feature_delta_rms_post"]
                        .detach()
                        .float()
                        .clamp_min(1e-12)
                    ),
                }
            )
            prediction_delta = (
                fused_prediction.detach().float() - base_prediction.detach().float()
            )
            diagnostics.update(
                {
                    "output_delta_up_volume": prediction_delta.clamp_min(0.0).sum(dim=1).mean(),
                    "output_delta_down_volume": (-prediction_delta).clamp_min(0.0).sum(dim=1).mean(),
                    "output_delta_net_volume": prediction_delta.sum(dim=1).mean(),
                }
            )
        self._last_prior_diagnostics = detached_diagnostics(diagnostics)
        return {
            "losses": {},
            "pred_logits": fused_logits,
            "pred_tactile": fused_prediction,
            "base_pred_logits": base_logits,
            "base_pred_tactile": base_prediction,
            "raw_fused_logits": raw_fused_logits,
            "feature_delta": feature_delta,
            "base_feature": grid if self.adapter_type in DEPTH_PRIOR_ADAPTERS else bottleneck,
            "prior_aux_losses": auxiliary_losses,
            "prior_control": control,
            "prior_diagnostics": self._last_prior_diagnostics,
        }

    def forward_paired(
        self,
        batch: Mapping[str, Any],
        *,
        control: str = "spatial_shuffle",
        train: bool = False,
        decode_control: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Evaluate aligned and counterfactual priors with one frozen RGB forward."""

        if self.adapter_type not in DEPTH_PRIOR_ADAPTERS:
            raise ValueError("Paired counterfactual forwarding is only defined for Depth adapters")
        real = self.forward_step(batch, train=train, prior_control="real")
        paired_batch = dict(batch)
        paired_batch["frozen_base_grid"] = real["base_feature"].detach()
        paired_batch["frozen_base_logits"] = real["base_pred_logits"].detach()
        if decode_control is None:
            decode_control = not train
        counterfactual = self.forward_step(
            paired_batch,
            train=train,
            prior_control=control,
            decode=bool(decode_control),
        )
        base_rms = real["base_feature"].detach().float().pow(2).mean(
            dim=tuple(range(1, real["base_feature"].ndim)), keepdim=True
        ).clamp_min(1e-24).sqrt()
        control_rms = counterfactual["feature_delta"].float().pow(2).mean(
            dim=tuple(range(1, counterfactual["feature_delta"].ndim)), keepdim=True
        ).clamp_min(1e-24).sqrt()
        budget = float(self.prior_adapter.rms_clamp.budget)
        identity_loss = ((control_rms / (budget * base_rms).clamp_min(1e-12)) ** 2).mean()
        correction_gap = (
            real["feature_delta"].detach().float()
            - counterfactual["feature_delta"].detach().float()
        ).pow(2).mean().sqrt()
        real["prior_diagnostics"] = dict(real["prior_diagnostics"])
        real["prior_diagnostics"].update(
            {
                "real_control_feature_gap_rms": correction_gap,
                "control_identity_raw": identity_loss.detach().float(),
            }
        )
        if decode_control:
            real["control_pred_logits"] = counterfactual["pred_logits"]
            real["control_pred_tactile"] = counterfactual["pred_tactile"]
        real["control_feature_delta"] = counterfactual["feature_delta"]
        real["control_identity_loss"] = identity_loss
        real["control_prior_aux_losses"] = counterfactual["prior_aux_losses"]
        real["control_prior_diagnostics"] = counterfactual["prior_diagnostics"]
        real["counterfactual_control"] = str(control)
        return real

    def forward(self, batch: Mapping[str, Any], **kwargs) -> Dict[str, Any]:
        return self.forward_step(batch, **kwargs)

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_prior_diagnostics
