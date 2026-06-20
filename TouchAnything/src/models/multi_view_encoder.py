"""
Multi-View Vision Encoder with View Dropout.

Processes multiple camera views (egocentric + wrist cameras) through a shared
DINOv2 backbone, adds view-specific embeddings, and fuses via cross-view attention.

Key features:
- Shared vision encoder across views (parameter efficient)
- Learnable view-type embeddings to distinguish camera perspectives
- View dropout during training → robust to missing views at inference
- Cross-view attention fusion → output same shape as single-view encoder

Training: randomly drop views so model learns all subsets
Inference: feed any subset of views (ego only, wrist only, or all three)

Output: (B, T, N, D) — identical to single-view VisionEncoder output,
        so all downstream modules (TemporalTransformer, Fusion, Decoder) are unchanged.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional


VIEW_NAMES = ['ego', 'wrist_left', 'wrist_right']


class MultiViewEncoder(nn.Module):
    """
    Multi-view vision encoder with view dropout and cross-view fusion.

    Architecture:
        1. Each view → shared DINOv2 → (B,T,N,D) patch tokens
        2. Add learnable view embedding per view type
        3. View dropout: randomly skip views during training
        4. Cross-view fusion: attention on view summaries → gated combination
        5. Output: (B,T,N,D) — same as single-view encoder
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        embed_dim: int = 768,
        num_cross_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        view_dropout_prob: float = 0.5,
    ):
        """
        Args:
            vision_encoder: Shared VisionEncoder (DINOv2) instance
            embed_dim: Feature dimension (must match vision_encoder output)
            num_cross_layers: Number of cross-view transformer layers
            num_heads: Attention heads for cross-view transformer
            mlp_ratio: MLP expansion ratio in cross-view transformer
            dropout: Dropout rate
            view_dropout_prob: Probability of dropping each view during training
                               (at least one view is always kept)
        """
        super().__init__()

        self.vision_encoder = vision_encoder
        self.embed_dim = embed_dim
        self.view_dropout_prob = view_dropout_prob

        # Learnable view-type embeddings (added to all patch tokens of a view)
        self.view_embeddings = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
            for name in VIEW_NAMES
        })
        for v in self.view_embeddings.values():
            nn.init.normal_(v, std=0.02)

        # Cross-view fusion transformer
        # Operates on V summary tokens (one per view), lightweight
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.cross_view_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_cross_layers,
        )

        # View gating: produces per-view importance weight from fused summaries
        self.view_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )

        # Output normalization
        self.output_norm = nn.LayerNorm(embed_dim)

        num_params = sum(p.numel() for p in self.parameters()) - \
                     sum(p.numel() for p in self.vision_encoder.parameters())
        print(f"[MultiViewEncoder] views={VIEW_NAMES}, "
              f"cross_layers={num_cross_layers}, heads={num_heads}, "
              f"view_dropout={view_dropout_prob}, "
              f"extra_params={num_params:,}")

    def _apply_view_dropout(self, available_views: List[str]) -> List[str]:
        """
        Randomly drop wrist views during training. Ego view is always kept.

        Args:
            available_views: List of view names that have input data

        Returns:
            keep: List of view names to actually process
        """
        if not self.training or len(available_views) <= 1:
            return available_views

        if self.view_dropout_prob <= 0:
            return available_views

        keep = []
        for name in available_views:
            if name == 'ego':
                # Ego view is always kept
                keep.append(name)
            elif torch.rand(1).item() > self.view_dropout_prob:
                keep.append(name)

        # Fallback: if ego wasn't in available_views, keep at least one
        if not keep:
            keep = [available_views[0]]

        return keep

    def forward(
        self,
        views: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            views: Dict mapping view_name → (B, T, 3, H, W) video tensor.
                   Any subset of VIEW_NAMES can be provided.
                   Must provide at least one view.

        Returns:
            features: (B, T, N, D) — fused multi-view features,
                      same shape as single-view VisionEncoder output.
        """
        # Validate input
        available_views = [n for n in VIEW_NAMES if n in views]
        assert len(available_views) > 0, \
            f"At least one view required. Got keys: {list(views.keys())}"

        # Apply view dropout (training only)
        active_views = self._apply_view_dropout(available_views)

        # Encode active views through shared backbone + add view embeddings
        encoded = {}  # name → (B, T, N, D)
        for name in active_views:
            feat = self.vision_encoder(views[name])  # (B, T, N, D)
            feat = feat + self.view_embeddings[name]
            encoded[name] = feat

        B, T, N, D = next(iter(encoded.values())).shape

        # Single view → skip cross-view attention
        if len(encoded) == 1:
            return self.output_norm(next(iter(encoded.values())))

        # --- Multi-view fusion ---
        # 1. Compute per-view summary tokens (mean pool patches)
        ordered_names = [n for n in VIEW_NAMES if n in encoded]
        features_list = []  # (B*T, N, D) per view
        summaries = []      # (B*T, D) per view

        for name in ordered_names:
            feat_flat = encoded[name].reshape(B * T, N, D)
            features_list.append(feat_flat)
            summaries.append(feat_flat.mean(dim=1))  # (B*T, D)

        V = len(summaries)

        # 2. Cross-view attention on summary tokens
        summary_stack = torch.stack(summaries, dim=1)  # (B*T, V, D)
        summary_fused = self.cross_view_transformer(summary_stack)  # (B*T, V, D)

        # 3. Compute gated view weights
        view_logits = self.view_gate(summary_fused).squeeze(-1)  # (B*T, V)
        view_weights = torch.softmax(view_logits, dim=-1)        # (B*T, V)

        # 4. Weighted combination of view features
        fused = torch.zeros(B * T, N, D, device=summary_stack.device,
                            dtype=summary_stack.dtype)
        for i in range(V):
            w = view_weights[:, i:i+1].unsqueeze(-1)  # (B*T, 1, 1)
            fused = fused + w * features_list[i]

        fused = self.output_norm(fused)
        return fused.reshape(B, T, N, D)


def build_multi_view_encoder(
    vision_encoder: nn.Module,
    config: dict,
    embed_dim: int,
) -> MultiViewEncoder:
    """
    Factory function: build MultiViewEncoder from config.

    Args:
        vision_encoder: Shared VisionEncoder instance
        config: multi_view config dict from YAML
        embed_dim: Model embedding dimension

    Returns:
        MultiViewEncoder instance
    """
    return MultiViewEncoder(
        vision_encoder=vision_encoder,
        embed_dim=embed_dim,
        num_cross_layers=config.get('num_cross_layers', 2),
        num_heads=config.get('num_heads', 8),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        dropout=config.get('dropout', 0.1),
        view_dropout_prob=config.get('view_dropout_prob', 0.5),
    )
