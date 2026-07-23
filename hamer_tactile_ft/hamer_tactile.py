from pathlib import Path
from typing import Dict, Sequence

import pytorch_lightning as pl
import torch
import torch.nn as nn

from losses import TactileLossConfig, compute_tactile_loss


def count_obj_vertices(obj_path: Path) -> int:
    with obj_path.open("r") as handle:
        return sum(1 for line in handle if line.startswith("v "))


def default_tactile_dim() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    subdiv_obj = repo_root / "opentouch" / "preprocess" / "scratch" / "mano_right_neutral_subdiv.obj"
    return count_obj_vertices(subdiv_obj)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_probability: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout_probability))
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.drop(self.act(self.norm1(self.fc1(features))))
        features = self.norm2(self.fc2(features))
        return self.act(features + residual)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class FullGrid32SpatialPooling(nn.Module):
    """Preserve the complete 16x12 DINO grid while controlling decoder size."""

    output_dim = 32 * 16 * 12

    def __init__(self, input_channels: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(int(input_channels), 32, kernel_size=1),
            ChannelLayerNorm(32),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if tuple(features.shape[-2:]) != (16, 12):
            raise ValueError(
                f"FullGrid32 expects a 16x12 feature map, got {tuple(features.shape[-2:])}"
            )
        return self.projection(features).flatten(1)


class DenseV2DinoReZeroTactileHead(nn.Module):
    """Fuse multilevel DINO maps through one bounded ReZero residual."""

    def __init__(
        self,
        tactile_dim: int,
        layer_indices: Sequence[int],
        residual_max_scale: float = 0.10,
        residual_rms_budget: float = 0.50,
        input_channels: int = 1280,
        channels: int = 256,
        decoder_dropout_scale: float = 1.0,
    ):
        super().__init__()
        layer_indices = tuple(int(layer) for layer in layer_indices)
        if len(layer_indices) < 2 or tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError("DINO layers must contain at least two unique increasing indices")
        if not 0.0 < float(residual_max_scale) <= 1.0:
            raise ValueError("dino_residual_max_scale must lie in (0, 1]")
        if not 0.0 < float(residual_rms_budget) <= 1.0:
            raise ValueError("dino_residual_rms_budget must lie in (0, 1]")
        dropout_scale = float(decoder_dropout_scale)
        if not 0.0 <= dropout_scale <= 1.0:
            raise ValueError("decoder_dropout_scale must lie in [0, 1]")

        self.layer_indices = layer_indices
        self.refinement_layer_indices = tuple(reversed(layer_indices[:-1]))
        self.residual_max_scale = float(residual_max_scale)
        self.residual_rms_budget = float(residual_rms_budget)

        pool = FullGrid32SpatialPooling(input_channels=channels)
        self.decoder = nn.Sequential(
            pool,
            nn.Dropout(p=0.5 * dropout_scale),
            nn.Linear(pool.output_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3 * dropout_scale),
            ResidualBlock(512, dropout_probability=0.3 * dropout_scale),
            nn.Linear(512, int(tactile_dim)),
        )
        self.base_projection = nn.Sequential(
            nn.Conv2d(int(input_channels), channels, kernel_size=1),
            nn.GELU(),
        )

        self.projections = nn.ModuleDict()
        self.refiners = nn.ModuleDict()
        for layer in self.refinement_layer_indices:
            key = str(layer)
            self.projections[key] = nn.Sequential(
                nn.Conv2d(int(input_channels), channels, kernel_size=1),
                ChannelLayerNorm(channels),
                nn.GELU(),
            )
            refiner = nn.Sequential(
                nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
                ChannelLayerNorm(channels),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
            nn.init.normal_(refiner[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(refiner[-1].bias)
            self.refiners[key] = refiner

        self.level_logits = nn.Parameter(torch.zeros(len(self.refinement_layer_indices)))
        self.global_gate = nn.Parameter(torch.zeros(()))
        self._last_feature_diagnostics: Dict[str, torch.Tensor] = {}

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_feature_diagnostics

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.level_logits, dim=0)

    def effective_gate(self) -> torch.Tensor:
        return self.residual_max_scale * torch.tanh(self.global_gate)

    @staticmethod
    def _sample_rms(features: torch.Tensor) -> torch.Tensor:
        return features.float().pow(2).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-24).sqrt()

    def _fuse(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_levels) != len(self.layer_indices):
            raise ValueError(f"Expected {len(self.layer_indices)} DINO levels, got {len(feature_levels)}")
        features_by_layer = dict(zip(self.layer_indices, feature_levels))
        base = self.base_projection(features_by_layer[self.layer_indices[-1]])
        projected_values = []
        residual_logits_values = []
        unit_residual_values = []
        for layer in self.refinement_layer_indices:
            key = str(layer)
            projected = self.projections[key](features_by_layer[layer])
            if projected.shape[-2:] != base.shape[-2:]:
                projected = nn.functional.interpolate(
                    projected,
                    size=base.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            residual_logits = self.refiners[key](torch.cat([base, projected], dim=1))
            projected_values.append(projected)
            residual_logits_values.append(residual_logits)
            unit_residual_values.append(torch.tanh(residual_logits))

        weights = self.fusion_weights().to(dtype=base.dtype)
        raw_delta = sum(weight * residual for weight, residual in zip(weights, unit_residual_values))
        delta_pre_budget = self.effective_gate().to(dtype=base.dtype) * raw_delta
        base_rms_per_sample = self._sample_rms(base).detach()
        delta_rms_pre_per_sample = self._sample_rms(delta_pre_budget)
        allowed_rms = self.residual_rms_budget * base_rms_per_sample
        budget_scale = torch.clamp(
            allowed_rms / delta_rms_pre_per_sample.clamp_min(1e-12),
            max=1.0,
        )
        delta = delta_pre_budget * budget_scale.to(dtype=delta_pre_budget.dtype)
        fused = base + delta

        unit_rms = torch.stack(
            [value.detach().float().pow(2).mean().sqrt() for value in unit_residual_values]
        )
        weighted_rms = weights.detach().float() * unit_rms
        delta_rms_post_per_sample = self._sample_rms(delta)
        self._last_feature_diagnostics = {
            "gate_raw": self.global_gate.detach().float(),
            "gate_effective": self.effective_gate().detach().float(),
            "level_weight": weights.detach().float(),
            "projected_rms": torch.stack(
                [value.detach().float().pow(2).mean().sqrt() for value in projected_values]
            ),
            "raw_residual_rms": unit_rms,
            "residual_saturation": torch.stack(
                [
                    (value.detach().float().abs() > 3.0).float().mean()
                    for value in residual_logits_values
                ]
            ),
            "effective_contribution": weighted_rms / weighted_rms.sum().clamp_min(1e-12),
            "delta_rms_pre_budget": delta_rms_pre_per_sample.detach().mean(),
            "delta_rms_post_budget": delta_rms_post_per_sample.detach().mean(),
            "delta_to_base_rms": (
                delta_rms_post_per_sample.detach() / base_rms_per_sample.clamp_min(1e-12)
            ).mean(),
            "budget_clip_rate": (budget_scale.detach() < (1.0 - 1e-6)).float().mean(),
            "base_rms": base.detach().float().pow(2).mean().sqrt(),
            "final_rms": fused.detach().float().pow(2).mean().sqrt(),
        }
        return fused

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.decoder(self._fuse(feature_levels))


class DinoV3BackboneAdapter(nn.Module):
    """Frozen local DINOv3 H+/16 backbone returning BCHW patch maps."""

    MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"

    def __init__(self, weights_path: str, image_size=(256, 192)):
        super().__init__()
        weights = Path(weights_path).expanduser()
        if not weights.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
        import timm
        from timm.models._helpers import load_state_dict
        from timm.models.eva import checkpoint_filter_fn

        self.weights_path = str(weights.resolve())
        try:
            self.model = timm.create_model(
                self.MODEL_NAME,
                pretrained=False,
                img_size=tuple(image_size),
                dynamic_img_size=False,
                num_classes=0,
            )
            state_dict = load_state_dict(self.weights_path, use_ema=False, device="cpu")
            state_dict = checkpoint_filter_fn(state_dict, self.model)
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to strictly load local DINOv3 H+/16 weights from {self.weights_path}"
            ) from exc
        if int(self.model.num_features) != 1280:
            raise RuntimeError(f"Expected 1280 DINOv3 feature channels, got {self.model.num_features}")
        patch_size = tuple(int(value) for value in self.model.patch_embed.patch_size)
        if patch_size != (16, 16):
            raise RuntimeError(f"Expected DINOv3 patch size 16x16, got {patch_size}")
        self.num_prefix_tokens = int(self.model.num_prefix_tokens)
        if self.num_prefix_tokens != 5:
            raise RuntimeError(f"Expected 5 DINOv3 prefix tokens, got {self.num_prefix_tokens}")
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.train(False)

    def get_num_layers(self) -> int:
        return len(self.model.blocks)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(self, image: torch.Tensor, layer_indices: Sequence[int]) -> Sequence[torch.Tensor]:
        layer_indices = tuple(int(layer) for layer in layer_indices)
        depth = self.get_num_layers()
        if not layer_indices or tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError("DINO layers must contain unique increasing indices")
        if min(layer_indices) < 1 or max(layer_indices) > depth:
            raise ValueError(f"DINO layers must lie in [1, {depth}], got {layer_indices}")
        feature_maps = self.model.forward_intermediates(
            image,
            indices=[layer - 1 for layer in layer_indices],
            return_prefix_tokens=False,
            norm=True,
            stop_early=False,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        height, width = self.model.patch_embed.dynamic_feat_size(image.shape[-2:])
        for layer, feature_map in zip(layer_indices, feature_maps):
            expected_shape = (image.shape[0], 1280, height, width)
            if tuple(feature_map.shape) != expected_shape:
                raise RuntimeError(
                    f"DINO layer {layer} returned {tuple(feature_map.shape)}, expected {expected_shape}"
                )
        return feature_maps


class DinoTactileModel(pl.LightningModule):
    """Standalone frozen-DINO to canonical tactile model."""

    def __init__(
        self,
        cfg=None,
        init_renderer: bool = False,
        tactile_only_forward: bool = True,
        tactile_loss_scale: float = 10.0,
        tactile_head_type: str = "dense_v2_dino_rezero",
        backbone_feature_layers: Sequence[int] = (8, 16, 24, 32),
        visual_backbone: str = "dinov3_hplus",
        dino_weights: str = "",
        dino_rezero_source: str = "multilevel",
        dino_residual_max_scale: float = 0.10,
        dino_residual_rms_budget: float = 0.50,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
    ):
        super().__init__()
        if not tactile_only_forward:
            raise ValueError("The standalone tactile model only supports tactile_only_forward=True")
        if tactile_head_type != "dense_v2_dino_rezero":
            raise ValueError("Only tactile_head_type=dense_v2_dino_rezero is supported")
        if visual_backbone != "dinov3_hplus":
            raise ValueError("Only visual_backbone=dinov3_hplus is supported")
        if dino_rezero_source != "multilevel":
            raise ValueError("Only dino_rezero_source=multilevel is supported")
        if pool_layout != "fullgrid32":
            raise ValueError("Only pool_layout=fullgrid32 is supported")

        self.visual_backbone = str(visual_backbone)
        self.dino_weights = str(dino_weights)
        self.backbone = DinoV3BackboneAdapter(self.dino_weights, image_size=(256, 192))
        self.tactile_dim = default_tactile_dim()
        self.tactile_head_type = str(tactile_head_type)
        self.backbone_feature_layers = tuple(int(layer) for layer in backbone_feature_layers)
        if self.backbone_feature_layers[-1] != self.backbone.get_num_layers():
            raise ValueError("backbone_feature_layers must end at the final DINO block")
        self.dino_rezero_source = "multilevel"
        self.dino_residual_max_scale = float(dino_residual_max_scale)
        self.dino_residual_rms_budget = float(dino_residual_rms_budget)
        self.pool_layout = "fullgrid32"
        self.pool_grid_size = (16, 12)
        self.pool_valid_tokens = 192
        self.decoder_dropout_scale = float(decoder_dropout_scale)
        self.tactile_only_forward = True
        self.tactile_loss_scale = float(tactile_loss_scale)
        self.tactile_loss_config = TactileLossConfig()
        self.tactile_head = DenseV2DinoReZeroTactileHead(
            self.tactile_dim,
            layer_indices=self.backbone_feature_layers,
            residual_max_scale=self.dino_residual_max_scale,
            residual_rms_budget=self.dino_residual_rms_budget,
            decoder_dropout_scale=self.decoder_dropout_scale,
        )
        self.automatic_optimization = True

    def set_tactile_loss_config(self, config: TactileLossConfig) -> None:
        self.tactile_loss_config = config

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract_tactile_features(self, image: torch.Tensor):
        return self.backbone(image, self.backbone_feature_layers)

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        image = batch["img"][:, :, :, 32:-32]
        with torch.no_grad():
            conditioning_features = self._extract_tactile_features(image)
        pred_logits = self.tactile_head(conditioning_features)
        return {
            "losses": {},
            "pred_logits": pred_logits,
            "pred_tactile": torch.sigmoid(pred_logits),
        }

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        tactile_loss, tactile_losses = compute_tactile_loss(
            pred=output["pred_tactile"],
            logits=output["pred_logits"],
            target=batch["tactile_signal"],
            palm_mask=batch["palm_mask"],
            valid_mask=batch["has_tactile"],
            dataset_batch=batch.get("dataset"),
            config=self.tactile_loss_config,
            current_epoch=getattr(self, "current_epoch", 0),
            sample_weight=batch.get("sample_weight"),
        )
        total_loss = self.tactile_loss_scale * tactile_loss
        output["losses"].update(tactile_losses)
        output["losses"]["loss_total"] = total_loss.detach()
        output["losses"]["loss_current_ramp"] = total_loss.detach()
        output["losses"]["loss_direct_raw"] = tactile_losses["loss_base_tactile"]
        output["losses"]["loss_full_ramp_reference"] = (
            self.tactile_loss_scale * tactile_losses["loss_full_ramp"]
        ).detach()
        return total_loss
