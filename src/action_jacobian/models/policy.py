"""Deterministic DINOv3+ACT with optional pixel-aligned Jacobian fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from .dinov3 import FrozenDinoV3Backbone
from .transformer import Transformer


class DeterministicDinoACTPolicy(nn.Module):
    """Map DINOv3 patch tokens to a fixed-length action chunk.

    Stage 1 keeps the exact visual-only architecture. Stage 2 compares a
    direct sign-array token, a globally pooled Jacobian token, and a matched
    adapter that fuses a 15-channel Jacobian into the aligned 16x16 grid.
    Only the two token conditions add one structural token; there are no
    camera-extrinsics, proprio, Robot ID, CVAE posterior, or latent tokens.
    """

    def __init__(self, args):
        super().__init__()
        self.chunk_size = args.chunk_size
        self.condition = getattr(args, "condition", "none")
        self.matched_jacobian_adapter = bool(
            getattr(args, "matched_jacobian_adapter", False)
        )
        if self.condition not in (
            "none",
            "sign_array",
            "global_token",
            "pixel_jacobian",
        ):
            raise ValueError(f"Unknown condition: {self.condition}")
        if self.condition == "pixel_jacobian" and not self.matched_jacobian_adapter:
            raise ValueError(
                "pixel_jacobian requires matched_jacobian_adapter=True"
            )
        self.backbone = FrozenDinoV3Backbone(
            model_path=args.dinov3_model_path,
            hidden_dim=args.hidden_dim,
            num_cameras=1,
        )
        self.use_pixel_adapter = self.matched_jacobian_adapter and self.condition in (
            "none",
            "pixel_jacobian",
        )
        if self.use_pixel_adapter:
            self.jacobian_projection = nn.Sequential(
                nn.Conv2d(15, args.hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=1),
            )
            self.spatial_fusion = nn.Linear(
                2 * args.hidden_dim, args.hidden_dim
            )
        elif self.condition == "global_token":
            self.global_jacobian_projection = nn.Sequential(
                nn.Linear(args.global_jacobian_dim, args.hidden_dim),
                nn.GELU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.GELU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
            )
            self.structural_pos_embed = nn.Parameter(
                torch.zeros(1, 1, args.hidden_dim)
            )
            nn.init.normal_(self.structural_pos_embed, std=0.02)
        elif self.condition == "sign_array":
            self.sign_array_projection = nn.Sequential(
                nn.Linear(args.sign_array_dim, args.hidden_dim),
                nn.GELU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.GELU(),
                nn.Linear(args.hidden_dim, args.hidden_dim),
            )
            self.structural_pos_embed = nn.Parameter(
                torch.zeros(1, 1, args.hidden_dim)
            )
            nn.init.normal_(self.structural_pos_embed, std=0.02)
        self.transformer = Transformer(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            ffn_dim=args.ffn_dim,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=bool(args.pre_norm),
            return_intermediate_dec=True,
            norm_cls=nn.LayerNorm,
            activation=args.activation,
        )
        self.action_queries = nn.Embedding(args.chunk_size, args.hidden_dim)
        self.action_head = nn.Linear(args.hidden_dim, args.action_dim)
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    def forward(self, data: dict[str, torch.Tensor]):
        image = data["image"]
        if image.ndim == 4:
            image = image.unsqueeze(1)
        visual_features, visual_positions = self.backbone(image)
        if self.use_pixel_adapter:
            jacobian = data.get("pixel_jacobian")
            if self.condition == "pixel_jacobian":
                if jacobian is None:
                    raise ValueError(
                        "pixel_jacobian condition requires pixel_jacobian input"
                    )
            else:
                jacobian = torch.zeros(
                    image.shape[0],
                    15,
                    16,
                    16,
                    device=visual_features.device,
                    dtype=visual_features.dtype,
                )
            jacobian_features = self.jacobian_projection(
                jacobian.to(dtype=visual_features.dtype)
            )
            jacobian_features = jacobian_features.flatten(2).transpose(1, 2)
            if jacobian_features.shape != visual_features.shape:
                raise ValueError(
                    "Pixel Jacobian and DINOv3 grids are not aligned: "
                    f"{tuple(jacobian_features.shape)} vs "
                    f"{tuple(visual_features.shape)}"
                )
            visual_features = self.spatial_fusion(
                torch.cat([visual_features, jacobian_features], dim=-1)
            )
        elif self.condition == "global_token":
            descriptor = data.get("global_jacobian")
            if descriptor is None:
                raise ValueError("global_token condition requires global_jacobian input")
            structural_token = self.global_jacobian_projection(
                descriptor.to(dtype=visual_features.dtype)
            ).unsqueeze(1)
            structural_pos = self.structural_pos_embed.expand(
                visual_features.shape[0], -1, -1
            )
            visual_features = torch.cat([visual_features, structural_token], dim=1)
            visual_positions = torch.cat([visual_positions, structural_pos], dim=1)
        elif self.condition == "sign_array":
            sign_array = data.get("global_sign")
            if sign_array is None:
                raise ValueError("sign_array condition requires global_sign input")
            structural_token = self.sign_array_projection(
                sign_array.to(dtype=visual_features.dtype)
            ).unsqueeze(1)
            structural_pos = self.structural_pos_embed.expand(
                visual_features.shape[0], -1, -1
            )
            visual_features = torch.cat([visual_features, structural_token], dim=1)
            visual_positions = torch.cat([visual_positions, structural_pos], dim=1)
        queries = repeat(
            self.action_queries.weight,
            "q d -> b q d",
            b=image.shape[0],
        )
        decoded = self.transformer(
            src=visual_features,
            mask=None,
            query_embed=queries,
            pos_embed=visual_positions,
        )[0]
        predicted_actions = self.action_head(decoded)

        actions = data.get("actions")
        if actions is None:
            return predicted_actions
        actions = actions[:, : self.chunk_size]
        is_pad = data["is_pad"][:, : self.chunk_size]
        all_l1 = F.l1_loss(actions, predicted_actions, reduction="none")
        l1 = (all_l1 * (~is_pad).unsqueeze(-1)).mean()
        return {"l1": l1, "loss": l1}

    def configure_optimizers(self):
        return self.optimizer
