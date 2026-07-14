from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn

from hamer.models.hamer import HAMER
from losses import TactileLossConfig, compute_tactile_loss


def count_obj_vertices(obj_path: Path) -> int:
    with obj_path.open("r") as handle:
        return sum(1 for line in handle if line.startswith("v "))


def default_tactile_dim() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    subdiv_obj = repo_root / "opentouch" / "preprocess" / "scratch" / "mano_right_neutral_subdiv.obj"
    return count_obj_vertices(subdiv_obj)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = self.drop(self.act(self.norm1(self.fc1(features))))
        features = self.norm2(self.fc2(features))
        return self.act(features + residual)


class AnatomicalSpatialPooling(nn.Module):
    """The original V2 5x5 pooling with 21 retained spatial cells."""

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        mask = torch.ones((1, 1, 5, 5), dtype=torch.bool)
        mask[0, 0, 4, 0] = False
        mask[0, 0, 4, 1] = False
        mask[0, 0, 4, 3] = False
        mask[0, 0, 4, 4] = False
        self.register_buffer("mask", mask)
        self.valid_token_count = int(mask.sum().item())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.pool(features)
        batch_size, channels, height, width = features.shape
        features = features.reshape(batch_size, channels, height * width)
        return features[:, :, self.mask.reshape(-1)].reshape(batch_size, -1)


class DenseV2TactileHead(nn.Module):
    """High-capacity direct continuous decoder used by the original V2 model."""

    def __init__(self, tactile_dim: int):
        super().__init__()
        pool = AnatomicalSpatialPooling()
        self.layers = nn.Sequential(
            nn.LazyConv2d(out_channels=256, kernel_size=1),
            nn.GELU(),
            pool,
            nn.Dropout(p=0.5),
            nn.Linear(256 * pool.valid_token_count, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            ResidualBlock(512),
            nn.Linear(512, tactile_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels for a BCHW feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.permute(0, 2, 3, 1)
        features = self.norm(features)
        return features.permute(0, 3, 1, 2).contiguous()


class DenseV2MultilevelTactileHead(nn.Module):
    """Fuse multiple ViT stages, then retain the original V2 dense decoder."""

    def __init__(self, tactile_dim: int, num_levels: int, projected_channels: int = 256):
        super().__init__()
        if num_levels < 2:
            raise ValueError("dense_v2_multilevel requires at least two backbone feature levels")

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LazyConv2d(projected_channels, kernel_size=1),
                    ChannelLayerNorm(projected_channels),
                    nn.GELU(),
                )
                for _ in range(num_levels)
            ]
        )
        self.level_logits = nn.Parameter(torch.zeros(num_levels))
        pool = AnatomicalSpatialPooling()
        self.decoder = nn.Sequential(
            pool,
            nn.Dropout(p=0.5),
            nn.Linear(projected_channels * pool.valid_token_count, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            ResidualBlock(512),
            nn.Linear(512, tactile_dim),
        )

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.level_logits, dim=0)

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_levels) != len(self.projections):
            raise ValueError(
                f"Expected {len(self.projections)} feature levels, got {len(feature_levels)}"
            )
        projected = [projection(features) for projection, features in zip(self.projections, feature_levels)]
        reference_size = projected[-1].shape[-2:]
        projected = [
            features
            if features.shape[-2:] == reference_size
            else nn.functional.interpolate(features, size=reference_size, mode="bilinear", align_corners=False)
            for features in projected
        ]
        weights = self.fusion_weights().to(dtype=projected[0].dtype)
        fused = sum(weight * features for weight, features in zip(weights, projected))
        return self.decoder(fused)


class DenseV2MultilevelConcatTactileHead(nn.Module):
    """Preserve complementary ViT levels before the original V2 decoder."""

    def __init__(self, tactile_dim: int, num_levels: int, projected_channels: int = 256):
        super().__init__()
        if num_levels < 2:
            raise ValueError("dense_v2_multilevel_concat requires at least two feature levels")
        self.num_levels = int(num_levels)
        self.projected_channels = int(projected_channels)
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LazyConv2d(projected_channels, kernel_size=1),
                    ChannelLayerNorm(projected_channels),
                    nn.GELU(),
                )
                for _ in range(num_levels)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(num_levels * projected_channels, projected_channels, kernel_size=1),
            ChannelLayerNorm(projected_channels),
            nn.GELU(),
        )
        pool = AnatomicalSpatialPooling()
        self.decoder = nn.Sequential(
            pool,
            nn.Dropout(p=0.5),
            nn.Linear(projected_channels * pool.valid_token_count, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            ResidualBlock(512),
            nn.Linear(512, tactile_dim),
        )
        self._last_feature_diagnostics = {}

    def fusion_group_contributions(self) -> torch.Tensor:
        weight = self.fusion[0].weight
        groups = weight.reshape(
            weight.shape[0], self.num_levels, self.projected_channels, *weight.shape[2:]
        )
        norms = groups.float().pow(2).sum(dim=(0, 2, 3, 4)).sqrt()
        return norms / norms.sum().clamp_min(1e-12)

    def feature_diagnostics(self) -> Dict[str, torch.Tensor]:
        return self._last_feature_diagnostics

    def forward(self, feature_levels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_levels) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} levels, got {len(feature_levels)}")
        projected = [projection(features) for projection, features in zip(self.projections, feature_levels)]
        reference_size = projected[-1].shape[-2:]
        projected = [
            features
            if features.shape[-2:] == reference_size
            else nn.functional.interpolate(features, size=reference_size, mode="bilinear", align_corners=False)
            for features in projected
        ]
        fused = self.fusion(torch.cat(projected, dim=1))
        self._last_feature_diagnostics = {
            "projected_rms": torch.stack([features.detach().float().pow(2).mean().sqrt() for features in projected]),
            "fusion_rms": fused.detach().float().pow(2).mean().sqrt(),
        }
        return self.decoder(fused)


class DinoV3BackboneAdapter(nn.Module):
    """Frozen timm DINOv3 backbone returning a BCHW patch feature map."""

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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(image)
        patch_tokens = tokens[:, self.num_prefix_tokens :]
        height, width = self.model.patch_embed.dynamic_feat_size(image.shape[-2:])
        expected = int(height * width)
        if patch_tokens.shape[1] != expected:
            raise RuntimeError(
                f"DINOv3 returned {patch_tokens.shape[1]} patch tokens; expected {height}x{width}={expected}"
            )
        return patch_tokens.reshape(image.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()


class HAMER_Tactile(HAMER):
    def __init__(
        self,
        cfg,
        init_renderer: bool = False,
        tactile_only_forward: bool = True,
        tactile_loss_scale: float = 10.0,
        tactile_head_type: str = "dense_v2",
        backbone_feature_layers: Sequence[int] = (16, 24, 32),
        visual_backbone: str = "hamer",
        dino_weights: str = "",
    ):
        super().__init__(cfg, init_renderer=init_renderer)
        self.visual_backbone = str(visual_backbone)
        self.dino_weights = str(dino_weights or "")
        if self.visual_backbone == "dinov3_hplus":
            self.backbone = DinoV3BackboneAdapter(self.dino_weights, image_size=(256, 192))
        elif self.visual_backbone != "hamer":
            raise ValueError(f"Unsupported visual_backbone: {self.visual_backbone}")
        self.tactile_dim = default_tactile_dim()
        self.tactile_head_type = str(tactile_head_type)
        self.backbone_feature_layers = tuple(int(layer) for layer in backbone_feature_layers)
        backbone_depth = int(self.backbone.get_num_layers()) if hasattr(self.backbone, "get_num_layers") else None
        if self.tactile_head_type in {"dense_v2_multilevel", "dense_v2_multilevel_concat"}:
            if self.visual_backbone != "hamer":
                raise ValueError(f"{self.tactile_head_type} currently requires visual_backbone=hamer")
            if len(self.backbone_feature_layers) < 2 or len(set(self.backbone_feature_layers)) != len(
                self.backbone_feature_layers
            ):
                raise ValueError("backbone_feature_layers must contain at least two unique layer indices")
            if backbone_depth is not None and (
                min(self.backbone_feature_layers) < 1 or max(self.backbone_feature_layers) > backbone_depth
            ):
                raise ValueError(
                    f"backbone_feature_layers must be in [1, {backbone_depth}], "
                    f"got {self.backbone_feature_layers}"
                )
        self.pool_layout = "legacy5"
        self.pool_grid_size = 5
        self.pool_valid_tokens = 21
        self.tactile_only_forward = bool(tactile_only_forward)
        self.tactile_loss_scale = float(tactile_loss_scale)
        self.tactile_loss_config = TactileLossConfig()
        if self.tactile_head_type == "dense_v2":
            self.tactile_head = DenseV2TactileHead(self.tactile_dim)
        elif self.tactile_head_type == "dense_v2_multilevel":
            self.tactile_head = DenseV2MultilevelTactileHead(
                self.tactile_dim,
                num_levels=len(self.backbone_feature_layers),
            )
        elif self.tactile_head_type == "dense_v2_multilevel_concat":
            self.tactile_head = DenseV2MultilevelConcatTactileHead(
                self.tactile_dim,
                num_levels=len(self.backbone_feature_layers),
            )
        else:
            raise ValueError(f"Unsupported tactile_head_type: {self.tactile_head_type}")
        self.automatic_optimization = True

    def set_tactile_loss_config(self, config: TactileLossConfig) -> None:
        self.tactile_loss_config = config

    def _extract_tactile_features(self, image: torch.Tensor):
        if self.tactile_head_type in {"dense_v2_multilevel", "dense_v2_multilevel_concat"}:
            return self.backbone(image, return_intermediate_layers=self.backbone_feature_layers)
        return self.backbone(image)

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        if self.tactile_only_forward:
            output = {"losses": {}}
        else:
            output = super().forward_step(batch, train=train)

        image = batch["img"][:, :, :, 32:-32]
        with torch.set_grad_enabled(self.backbone.training and train):
            conditioning_features = self._extract_tactile_features(image)
        pred_logits = self.tactile_head(conditioning_features)
        output["pred_logits"] = pred_logits
        output["pred_tactile"] = torch.sigmoid(pred_logits)
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        if self.tactile_only_forward:
            base_loss = output["pred_tactile"].new_zeros(())
            output.setdefault("losses", {})
        else:
            base_loss = super().compute_loss(batch, output, train=train)

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
        output["losses"].update(tactile_losses)
        total_loss = base_loss + self.tactile_loss_scale * tactile_loss
        output["losses"]["loss_total"] = total_loss.detach()
        return total_loss
