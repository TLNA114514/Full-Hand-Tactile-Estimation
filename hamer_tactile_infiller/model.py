import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import count_obj_vertices
    from .opengl_guard import install_opengl_guard
except ImportError:
    from common import count_obj_vertices
    from opengl_guard import install_opengl_guard

install_opengl_guard()

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
FT_DIR = WORKSPACE_DIR / "hamer_tactile_ft"
HAMER_DIR = WORKSPACE_DIR / "hamer"
if str(HAMER_DIR) not in sys.path:
    sys.path.append(str(HAMER_DIR))
if str(FT_DIR) not in sys.path:
    sys.path.append(str(FT_DIR))

from hamer_tactile import HAMER_Tactile  # noqa: E402
from losses import TactileLossConfig, compute_tactile_loss  # noqa: E402


class TactileInfiller(nn.Module):
    def __init__(
        self,
        cfg,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        init_renderer=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.frame_model = HAMER_Tactile(cfg, init_renderer=init_renderer)
        self.tactile_dim = count_obj_vertices()
        self.hidden_dim = int(hidden_dim)

        self.tactile_proj = nn.Linear(self.tactile_dim, hidden_dim)
        self.visual_pool = nn.Sequential(
            nn.LazyConv2d(hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.mask_proj = nn.Linear(3, hidden_dim)
        self.dataset_embed = nn.Embedding(3, hidden_dim)
        self.hand_embed = nn.Embedding(2, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, self.tactile_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def freeze_frame_model(self):
        for param in self.frame_model.parameters():
            param.requires_grad = False

    def unfreeze_tactile_head(self):
        for param in self.frame_model.tactile_head.parameters():
            param.requires_grad = True

    def initialize_lazy_layers(self, device=None):
        device = device or next(self.parameters()).device
        image_size = int(self.cfg.MODEL.IMAGE_SIZE)
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            feats = self.frame_model.backbone(dummy[:, :, :, 32:-32])
            self.frame_model.tactile_head(feats)
            self.visual_pool(feats)

    def _dataset_ids(self, dataset_batch, batch_size, seq_len, device):
        mapping = {"opentouch": 0, "touchanything": 1, "egotactile": 2}
        ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        if isinstance(dataset_batch, (list, tuple)):
            for t, names_at_t in enumerate(dataset_batch):
                if isinstance(names_at_t, (list, tuple)):
                    names = names_at_t
                else:
                    names = [names_at_t] * batch_size
                for b, name in enumerate(names[:batch_size]):
                    ids[b, t] = mapping.get(str(name).lower(), 0)
        return ids

    def _hand_ids(self, batch, batch_size, seq_len, device):
        right = batch.get("right", None)
        if right is None:
            return torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
        if torch.is_tensor(right):
            vals = right.to(device).view(batch_size, 1).expand(batch_size, seq_len)
            return (vals > 0.5).long()
        return torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    def encode_observed_frames(self, imgs, observed_mask):
        bsz, seq_len, channels, height, width = imgs.shape
        flat_imgs = imgs.reshape(bsz * seq_len, channels, height, width)
        flat_mask = observed_mask.reshape(bsz * seq_len) > 0.5
        pred = torch.zeros((bsz * seq_len, self.tactile_dim), dtype=imgs.dtype, device=imgs.device)
        visual = torch.zeros((bsz * seq_len, self.hidden_dim), dtype=imgs.dtype, device=imgs.device)

        if flat_mask.any():
            obs_imgs = flat_imgs[flat_mask]
            with torch.set_grad_enabled(self.frame_model.tactile_head.training):
                feats = self.frame_model.backbone(obs_imgs[:, :, :, 32:-32])
                logits = self.frame_model.tactile_head(feats)
                pred[flat_mask] = torch.sigmoid(logits).to(dtype=pred.dtype)
                visual[flat_mask] = self.visual_pool(feats).to(dtype=visual.dtype)
        return pred.reshape(bsz, seq_len, self.tactile_dim), visual.reshape(bsz, seq_len, self.hidden_dim)

    def forward(self, batch):
        imgs = batch["img"]
        observed_mask = batch["observed_mask"]
        bbox_valid = batch["bbox_valid"]
        target_mask = batch["target_mask"]
        bsz, seq_len = observed_mask.shape

        observed_tactile, visual_tokens = self.encode_observed_frames(imgs, observed_mask)
        x = self.tactile_proj(observed_tactile) + visual_tokens

        mask_feats = torch.stack([observed_mask, bbox_valid, target_mask], dim=-1).to(imgs.dtype)
        x = x + self.mask_proj(mask_feats)
        dataset_ids = self._dataset_ids(batch.get("dataset"), bsz, seq_len, imgs.device)
        hand_ids = self._hand_ids(batch, bsz, seq_len, imgs.device)
        x = x + self.dataset_embed(dataset_ids) + self.hand_embed(hand_ids)
        x = x + self.pos_embed[:, :seq_len, :]

        encoded = self.temporal_encoder(x)
        logits = self.output_head(encoded)
        return {
            "observed_tactile": observed_tactile,
            "pred_logits": logits,
            "pred_tactile": torch.sigmoid(logits),
        }


def infiller_loss(batch, output, temporal_smooth_weight=0.05, tactile_loss_config=None, current_epoch=0):
    target = batch["tactile_signal"]
    pred = output["pred_tactile"]
    palm_mask = batch["palm_mask"]
    target_mask = batch["target_mask"]
    loss_weight = batch["loss_weight"]

    tactile_loss, tactile_losses = compute_tactile_loss(
        pred=pred,
        logits=output["pred_logits"],
        target=target,
        palm_mask=palm_mask,
        valid_mask=target_mask,
        dataset_batch=batch.get("dataset"),
        config=tactile_loss_config or TactileLossConfig(),
        current_epoch=current_epoch,
        sample_weight=loss_weight,
    )

    smooth_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if pred.shape[1] > 1 and temporal_smooth_weight > 0:
        both_valid = target_mask[:, 1:] * target_mask[:, :-1]
        if both_valid.sum() > 0:
            pred_delta = pred[:, 1:] - pred[:, :-1]
            target_delta = target[:, 1:] - target[:, :-1]
            smooth = F.smooth_l1_loss(pred_delta, target_delta, reduction="none") * palm_mask[:, 1:]
            smooth = smooth * both_valid.unsqueeze(-1)
            smooth_loss = smooth.sum() / (both_valid.sum() * palm_mask[0, 0].sum().clamp_min(1.0)).clamp_min(1.0)

    total = tactile_loss + temporal_smooth_weight * smooth_loss
    tactile_losses.update({
        "loss_temporal": smooth_loss.detach(),
        "loss_total": total.detach(),
    })
    return total, tactile_losses


def metrics(batch, output):
    pred = output["pred_tactile"]
    target = batch["tactile_signal"]
    palm_mask = batch["palm_mask"]
    target_mask = batch["target_mask"]
    bbox_valid = batch["bbox_valid"]

    def _subset(mask):
        denom = mask.sum() * palm_mask[0, 0].sum().clamp_min(1.0)
        if denom <= 0:
            zero = torch.zeros((), device=pred.device, dtype=pred.dtype)
            return zero, zero
        err = torch.abs(pred - target) * palm_mask * mask.unsqueeze(-1)
        sq = ((pred - target) ** 2) * palm_mask * mask.unsqueeze(-1)
        mae = err.sum() / denom
        rmse = torch.sqrt(sq.sum() / denom)
        return mae, rmse

    all_mae, all_rmse = _subset(target_mask)
    obs_mae, obs_rmse = _subset(target_mask * bbox_valid)
    miss_mae, miss_rmse = _subset(target_mask * (1.0 - bbox_valid))
    return {
        "all_mae": all_mae,
        "all_rmse": all_rmse,
        "observed_bbox_mae": obs_mae,
        "observed_bbox_rmse": obs_rmse,
        "missing_bbox_mae": miss_mae,
        "missing_bbox_rmse": miss_rmse,
    }
