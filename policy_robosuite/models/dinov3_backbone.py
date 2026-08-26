"""Frozen DINOv3 ViT-B/16 dense-token backbone for ACT."""

from __future__ import annotations

import torch
from einops import rearrange, repeat
from torch import nn


class FrozenDinoV3Backbone(nn.Module):
    def __init__(self, model_path: str, hidden_dim: int, num_cameras: int = 1):
        super().__init__()
        from transformers import AutoModel

        self.vit = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        self.vit.requires_grad_(False)
        self.vit.eval()
        config = self.vit.config
        self.patch_size = int(config.patch_size)
        self.feature_dim = int(config.hidden_size)
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        self.input_proj = nn.Linear(self.feature_dim, hidden_dim)
        self.pos_embed = nn.Embedding(num_cameras * 16 * 16, hidden_dim)
        self.num_channels = hidden_dim

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.vit.eval()
        return self

    def forward(self, images: torch.Tensor):
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected [B,N,3,H,W] RGB input, got {tuple(images.shape)}")
        batch, num_cameras, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"H,W must be divisible by patch_size={self.patch_size}")
        pixels = rearrange(images, "b n c h w -> (b n) c h w")
        pixels = (pixels - self.image_mean) / self.image_std
        with torch.no_grad():
            outputs = self.vit(pixel_values=pixels.to(dtype=torch.bfloat16))
            tokens = outputs.last_hidden_state
            start = 1 + self.num_register_tokens
            expected = (height // self.patch_size) * (width // self.patch_size)
            tokens = tokens[:, start : start + expected]
            if tokens.shape[1] != expected:
                raise RuntimeError(
                    f"DINOv3 returned {tokens.shape[1]} patch tokens; expected {expected}"
                )
        features = self.input_proj(tokens.float())
        features = rearrange(features, "(b n) p d -> b (n p) d", b=batch, n=num_cameras)
        positions = repeat(
            self.pos_embed.weight[: num_cameras * expected],
            "p d -> b p d",
            b=batch,
        )
        return features, positions
