from pathlib import Path
from typing import Dict, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

if __package__:
    from .losses import TactileLossConfig, compute_tactile_loss
else:
    from losses import TactileLossConfig, compute_tactile_loss


def count_obj_vertices(obj_path: Path) -> int:
    with obj_path.open("r") as handle:
        return sum(1 for line in handle if line.startswith("v "))


def default_tactile_dim() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    subdiv_obj = (
        repo_root
        / "opentouch"
        / "preprocess"
        / "scratch"
        / "mano_right_neutral_subdiv.obj"
    )
    return count_obj_vertices(subdiv_obj)


SUPPORTED_INPUT_RESOLUTIONS = ((256, 192), (320, 240), (384, 288))
SUPPORTED_TACTILE_HEAD_TYPES = ("dense_v2", "dense_v2_dino_rezero")
SUPPORTED_POOL_LAYOUTS = ("legacy5", "fullgrid32")


def parse_input_resolution(value) -> Tuple[int, int]:
    if isinstance(value, str):
        parts = value.lower().split("x")
        if len(parts) != 2:
            raise ValueError(
                "input_resolution must use HEIGHTxWIDTH, for example 320x240"
            )
        height, width = (int(part.strip()) for part in parts)
    elif isinstance(value, Sequence) and len(value) == 2:
        height, width = (int(part) for part in value)
    else:
        raise ValueError(
            "input_resolution must be HEIGHTxWIDTH or a two-value sequence"
        )
    resolution = (height, width)
    if resolution not in SUPPORTED_INPUT_RESOLUTIONS:
        allowed = ", ".join(
            f"{height}x{width}" for height, width in SUPPORTED_INPUT_RESOLUTIONS
        )
        raise ValueError(
            f"Unsupported input_resolution {height}x{width}; choose one of {allowed}"
        )
    if height % 16 or width % 16:
        raise ValueError(
            "input_resolution height and width must both be divisible by 16"
        )
    return resolution


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
        features = features.permute(0, 2, 3, 1)
        features = self.norm(features)
        return features.permute(0, 3, 1, 2).contiguous()


class AnatomicalSpatialPooling(nn.Module):
    """Original V2 5x5 pooling with the four bottom-corner cells removed."""

    def __init__(self, input_channels: int = 256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        mask = torch.ones((5, 5), dtype=torch.bool)
        mask[4, 0] = False
        mask[4, 1] = False
        mask[4, 3] = False
        mask[4, 4] = False
        self.register_buffer("mask", mask)
        self.valid_token_count = int(mask.sum().item())
        self.output_dim = int(input_channels) * self.valid_token_count

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.pool(features)
        batch_size, channels, height, width = features.shape
        features = features.reshape(batch_size, channels, height * width)
        return features[:, :, self.mask.reshape(-1)].reshape(batch_size, -1)


class FullGrid32SpatialPooling(nn.Module):
    """Preserve the complete DINO grid while controlling decoder size."""

    def __init__(
        self,
        input_channels: int = 256,
        grid_size: Sequence[int] = (16, 12),
        output_channels: int = 32,
    ):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        self.output_channels = int(output_channels)
        self.valid_token_count = self.grid_size[0] * self.grid_size[1]
        self.output_dim = self.output_channels * self.valid_token_count
        self.projection = nn.Sequential(
            nn.Conv2d(int(input_channels), self.output_channels, kernel_size=1),
            ChannelLayerNorm(self.output_channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if tuple(features.shape[-2:]) != self.grid_size:
            raise ValueError(
                f"FullGrid32 expects a {self.grid_size} feature map, "
                f"got {tuple(features.shape[-2:])}"
            )
        return self.projection(features).flatten(1)


def _build_dense_decoder(
    *,
    tactile_dim: int,
    channels: int,
    pool_layout: str,
    grid_size: Sequence[int],
    pool_output_channels: int,
    dropout_scale: float,
) -> Tuple[nn.Sequential, int, int]:
    if pool_layout == "legacy5":
        pool = AnatomicalSpatialPooling(input_channels=channels)
    elif pool_layout == "fullgrid32":
        pool = FullGrid32SpatialPooling(
            input_channels=channels,
            grid_size=grid_size,
            output_channels=pool_output_channels,
        )
    else:
        raise ValueError(
            f"Unsupported pool_layout={pool_layout}; choose one of {SUPPORTED_POOL_LAYOUTS}"
        )
    decoder = nn.Sequential(
        pool,
        nn.Dropout(p=0.5 * dropout_scale),
        nn.Linear(pool.output_dim, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(p=0.3 * dropout_scale),
        ResidualBlock(512, dropout_probability=0.3 * dropout_scale),
        nn.Linear(512, int(tactile_dim)),
    )
    return decoder, int(pool.output_dim), int(pool.valid_token_count)


class DenseV2TactileHead(nn.Module):
    """Final-DINO-feature control using the canonical V2 dense decoder."""

    def __init__(
        self,
        tactile_dim: int,
        input_channels: int = 1280,
        channels: int = 256,
        pool_layout: str = "legacy5",
        decoder_dropout_scale: float = 1.0,
        grid_size: Sequence[int] = (16, 12),
        pool_output_channels: int = 32,
    ):
        super().__init__()
        dropout_scale = float(decoder_dropout_scale)
        if not 0.0 <= dropout_scale <= 1.0:
            raise ValueError("decoder_dropout_scale must lie in [0, 1]")
        self.pool_layout = str(pool_layout)
        self.grid_size = tuple(int(value) for value in grid_size)
        self.base_projection = nn.Sequential(
            nn.Conv2d(int(input_channels), int(channels), kernel_size=1),
            nn.GELU(),
        )
        self.decoder, self.decoder_input_dim, self.pool_valid_tokens = (
            _build_dense_decoder(
                tactile_dim=tactile_dim,
                channels=channels,
                pool_layout=self.pool_layout,
                grid_size=self.grid_size,
                pool_output_channels=pool_output_channels,
                dropout_scale=dropout_scale,
            )
        )
        self.refinement_layer_indices = ()

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, features) -> torch.Tensor:
        if isinstance(features, (tuple, list)):
            if not features:
                raise ValueError("dense_v2 received no DINO feature maps")
            features = features[-1]
        return self.decoder(self.base_projection(features))


class DenseV2DinoReZeroTactileHead(DenseV2TactileHead):
    """Fuse multilevel DINO maps through one bounded ReZero residual."""

    def __init__(
        self,
        tactile_dim: int,
        layer_indices: Sequence[int],
        residual_max_scale: float = 0.10,
        residual_rms_budget: float = 0.50,
        input_channels: int = 1280,
        channels: int = 256,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
        grid_size: Sequence[int] = (16, 12),
        pool_output_channels: int = 32,
    ):
        layer_indices = tuple(int(layer) for layer in layer_indices)
        if len(layer_indices) < 2 or tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError(
                "DINO ReZero layers must contain at least two unique increasing indices"
            )
        if not 0.0 < float(residual_max_scale) <= 1.0:
            raise ValueError("dino_residual_max_scale must lie in (0, 1]")
        if not 0.0 < float(residual_rms_budget) <= 1.0:
            raise ValueError("dino_residual_rms_budget must lie in (0, 1]")
        super().__init__(
            tactile_dim=tactile_dim,
            input_channels=input_channels,
            channels=channels,
            pool_layout=pool_layout,
            decoder_dropout_scale=decoder_dropout_scale,
            grid_size=grid_size,
            pool_output_channels=pool_output_channels,
        )
        self.layer_indices = layer_indices
        self.refinement_layer_indices = tuple(reversed(layer_indices[:-1]))
        self.residual_max_scale = float(residual_max_scale)
        self.residual_rms_budget = float(residual_rms_budget)

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
        return (
            features.float()
            .pow(2)
            .mean(dim=(1, 2, 3), keepdim=True)
            .clamp_min(1e-24)
            .sqrt()
        )

    def _fuse(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_levels) != len(self.layer_indices):
            raise ValueError(
                f"Expected {len(self.layer_indices)} DINO levels, got {len(feature_levels)}"
            )
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
            residual_logits = self.refiners[key](
                torch.cat([base, projected], dim=1)
            )
            projected_values.append(projected)
            residual_logits_values.append(residual_logits)
            unit_residual_values.append(torch.tanh(residual_logits))

        weights = self.fusion_weights().to(dtype=base.dtype)
        raw_delta = sum(
            weight * residual
            for weight, residual in zip(weights, unit_residual_values)
        )
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

        if not self.training:
            unit_rms = torch.stack(
                [
                    value.detach().float().pow(2).mean().sqrt()
                    for value in unit_residual_values
                ]
            )
            weighted_rms = weights.detach().float() * unit_rms
            delta_rms_post_per_sample = self._sample_rms(delta)
            self._last_feature_diagnostics = {
                "gate_raw": self.global_gate.detach().float(),
                "gate_effective": self.effective_gate().detach().float(),
                "level_weight": weights.detach().float(),
                "projected_rms": torch.stack(
                    [
                        value.detach().float().pow(2).mean().sqrt()
                        for value in projected_values
                    ]
                ),
                "raw_residual_rms": unit_rms,
                "residual_saturation": torch.stack(
                    [
                        (value.detach().float().abs() > 3.0).float().mean()
                        for value in residual_logits_values
                    ]
                ),
                "effective_contribution": (
                    weighted_rms / weighted_rms.sum().clamp_min(1e-12)
                ),
                "delta_rms_pre_budget": delta_rms_pre_per_sample.detach().mean(),
                "delta_rms_post_budget": delta_rms_post_per_sample.detach().mean(),
                "delta_to_base_rms": (
                    delta_rms_post_per_sample.detach()
                    / base_rms_per_sample.clamp_min(1e-12)
                ).mean(),
                "budget_clip_rate": (
                    budget_scale.detach() < (1.0 - 1e-6)
                ).float().mean(),
                "base_rms": base.detach().float().pow(2).mean().sqrt(),
                "final_rms": fused.detach().float().pow(2).mean().sqrt(),
            }
        else:
            self._last_feature_diagnostics = {}
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
            state_dict = load_state_dict(
                self.weights_path, use_ema=False, device="cpu"
            )
            state_dict = checkpoint_filter_fn(state_dict, self.model)
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to strictly load local DINOv3 H+/16 weights from "
                f"{self.weights_path}"
            ) from exc
        if int(self.model.num_features) != 1280:
            raise RuntimeError(
                f"Expected 1280 DINOv3 feature channels, got {self.model.num_features}"
            )
        patch_size = tuple(int(value) for value in self.model.patch_embed.patch_size)
        if patch_size != (16, 16):
            raise RuntimeError(f"Expected DINOv3 patch size 16x16, got {patch_size}")
        self.num_prefix_tokens = int(self.model.num_prefix_tokens)
        if self.num_prefix_tokens != 5:
            raise RuntimeError(
                f"Expected 5 DINOv3 prefix tokens, got {self.num_prefix_tokens}"
            )
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.train(False)

    def get_num_layers(self) -> int:
        return len(self.model.blocks)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def forward(
        self, image: torch.Tensor, layer_indices: Sequence[int]
    ) -> Sequence[torch.Tensor]:
        layer_indices = tuple(int(layer) for layer in layer_indices)
        depth = self.get_num_layers()
        if not layer_indices or tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError("DINO layers must contain unique increasing indices")
        if min(layer_indices) < 1 or max(layer_indices) > depth:
            raise ValueError(
                f"DINO layers must lie in [1, {depth}], got {layer_indices}"
            )
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
                    f"DINO layer {layer} returned {tuple(feature_map.shape)}, "
                    f"expected {expected_shape}"
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
        dino_residual_max_scale: float = 0.10,
        dino_residual_rms_budget: float = 0.50,
        pool_layout: str = "fullgrid32",
        decoder_dropout_scale: float = 1.0,
        input_resolution: Sequence[int] = (256, 192),
        pool_output_channels: int = 32,
    ):
        super().__init__()
        if not tactile_only_forward:
            raise ValueError(
                "The standalone tactile model only supports tactile_only_forward=True"
            )
        if tactile_head_type not in SUPPORTED_TACTILE_HEAD_TYPES:
            raise ValueError(
                f"Unsupported tactile_head_type={tactile_head_type}; "
                f"choose one of {SUPPORTED_TACTILE_HEAD_TYPES}"
            )
        if visual_backbone != "dinov3_hplus":
            raise ValueError("Only visual_backbone=dinov3_hplus is supported")
        if pool_layout not in SUPPORTED_POOL_LAYOUTS:
            raise ValueError(
                f"Unsupported pool_layout={pool_layout}; "
                f"choose one of {SUPPORTED_POOL_LAYOUTS}"
            )

        self.visual_backbone = str(visual_backbone)
        self.dino_weights = str(dino_weights)
        self.input_resolution = parse_input_resolution(input_resolution)
        self.backbone = DinoV3BackboneAdapter(
            self.dino_weights,
            image_size=self.input_resolution,
        )
        self.tactile_dim = default_tactile_dim()
        self.tactile_head_type = str(tactile_head_type)
        requested_layers = tuple(int(layer) for layer in backbone_feature_layers)
        if not requested_layers or requested_layers[-1] != self.backbone.get_num_layers():
            raise ValueError(
                "backbone_feature_layers must end at the final DINO block"
            )
        if self.tactile_head_type == "dense_v2":
            self.backbone_feature_layers = (requested_layers[-1],)
        else:
            if len(requested_layers) < 2:
                raise ValueError(
                    "dense_v2_dino_rezero requires at least two DINO feature layers"
                )
            self.backbone_feature_layers = requested_layers

        self.dino_residual_max_scale = float(dino_residual_max_scale)
        self.dino_residual_rms_budget = float(dino_residual_rms_budget)
        self.pool_layout = str(pool_layout)
        self.feature_grid_size = (
            self.input_resolution[0] // 16,
            self.input_resolution[1] // 16,
        )
        self.pool_grid_size = self.feature_grid_size
        self.pool_output_channels = int(pool_output_channels)
        self.decoder_dropout_scale = float(decoder_dropout_scale)
        self.tactile_only_forward = True
        self.tactile_loss_scale = float(tactile_loss_scale)
        self.tactile_loss_config = TactileLossConfig()

        common_head_args = {
            "tactile_dim": self.tactile_dim,
            "pool_layout": self.pool_layout,
            "decoder_dropout_scale": self.decoder_dropout_scale,
            "grid_size": self.feature_grid_size,
            "pool_output_channels": self.pool_output_channels,
        }
        if self.tactile_head_type == "dense_v2":
            self.tactile_head = DenseV2TactileHead(**common_head_args)
        else:
            self.tactile_head = DenseV2DinoReZeroTactileHead(
                **common_head_args,
                layer_indices=self.backbone_feature_layers,
                residual_max_scale=self.dino_residual_max_scale,
                residual_rms_budget=self.dino_residual_rms_budget,
            )
        self.pool_valid_tokens = int(self.tactile_head.pool_valid_tokens)
        self.decoder_input_dim = int(self.tactile_head.decoder_input_dim)
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
        image = batch["img"]
        if tuple(image.shape[-2:]) != self.input_resolution:
            raise ValueError(
                f"Model expects input {self.input_resolution}, "
                f"got {tuple(image.shape[-2:])}"
            )
        with torch.no_grad():
            conditioning_features = self._extract_tactile_features(image)
        pred_logits = self.tactile_head(conditioning_features)
        return {
            "losses": {},
            "pred_logits": pred_logits,
            "pred_tactile": torch.sigmoid(pred_logits),
        }

    def compute_loss(
        self, batch: Dict, output: Dict, train: bool = True
    ) -> torch.Tensor:
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
        output["losses"]["loss_direct_raw"] = tactile_losses[
            "loss_base_tactile"
        ]
        output["losses"]["loss_full_ramp_reference"] = (
            self.tactile_loss_scale * tactile_losses["loss_full_ramp"]
        ).detach()
        return total_loss
