"""Deterministic ACT with matched structural-conditioning adapters."""

from __future__ import annotations

import torch
from torch import nn
from einops import rearrange, repeat
from torchvision.models import ResNet18_Weights, resnet18

from .transformer import Transformer


CONDITION_MODES = ("none", "global_sign", "pixel_jacobian")


class DeterministicACT(nn.Module):
    def __init__(
        self,
        condition_mode: str,
        chunk_size: int = 30,
        hidden_dim: int = 256,
        nheads: int = 8,
        ffn_dim: int = 1024,
        enc_layers: int = 2,
        dec_layers: int = 4,
        dropout: float = 0.0,
        imagenet: bool = False,
    ) -> None:
        super().__init__()
        if condition_mode not in CONDITION_MODES:
            raise ValueError(f"Unknown condition mode: {condition_mode}")
        self.condition_mode = condition_mode
        self.chunk_size = chunk_size

        weights = ResNet18_Weights.DEFAULT if imagenet else None
        resnet = resnet18(weights=weights)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.rgb_projection = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.jacobian_projection = nn.Sequential(
            nn.Conv2d(15, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.spatial_fusion = nn.Conv2d(
            2 * hidden_dim, hidden_dim, kernel_size=1
        )
        self.global_sign_projection = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.null_condition = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.source_position = nn.Embedding(16 * 16 + 1, hidden_dim)
        self.query_embedding = nn.Embedding(chunk_size, hidden_dim)

        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            ffn_dim=ffn_dim,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=True,
            return_intermediate_dec=True,
            norm_cls=nn.LayerNorm,
            activation="gelu",
        )
        self.action_head = nn.Linear(hidden_dim, 8)

    def forward(
        self,
        image: torch.Tensor,
        pixel_jacobian: torch.Tensor,
        global_sign: torch.Tensor,
    ) -> torch.Tensor:
        rgb_features = self.rgb_projection(self.rgb_backbone(image))
        if rgb_features.shape[-2:] != (16, 16):
            raise ValueError(
                f"Expected 16x16 RGB features, got {rgb_features.shape[-2:]}"
            )

        if self.condition_mode == "pixel_jacobian":
            structural_features = self.jacobian_projection(pixel_jacobian)
        else:
            structural_features = self.jacobian_projection(
                torch.zeros_like(pixel_jacobian)
            )
        spatial = self.spatial_fusion(
            torch.cat([rgb_features, structural_features], dim=1)
        )
        spatial = rearrange(spatial, "b d h w -> b (h w) d")

        if self.condition_mode == "global_sign":
            condition_token = self.global_sign_projection(global_sign)[:, None]
        else:
            condition_token = repeat(
                self.null_condition, "1 1 d -> b 1 d", b=image.shape[0]
            )
        source = torch.cat([spatial, condition_token], dim=1)
        source_position = repeat(
            self.source_position.weight,
            "s d -> b s d",
            b=image.shape[0],
        )
        queries = repeat(
            self.query_embedding.weight,
            "q d -> b q d",
            b=image.shape[0],
        )
        decoded = self.transformer(
            src=source,
            mask=None,
            query_embed=queries,
            pos_embed=source_position,
        )[0]
        return self.action_head(decoded)
