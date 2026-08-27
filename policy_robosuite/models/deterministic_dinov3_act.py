"""Deterministic DINOv3+ACT baseline with visual tokens only."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from .dinov3_backbone import FrozenDinoV3Backbone
from .transformer import Transformer


class DeterministicDinoACTPolicy(nn.Module):
    """Map DINOv3 patch tokens directly to a fixed-length action chunk.

    The Transformer source contains only dense RGB patch tokens. There are no
    camera-extrinsics, proprio, Robot ID, CVAE posterior, or latent tokens.
    """

    def __init__(self, args):
        super().__init__()
        self.chunk_size = args.chunk_size
        self.backbone = FrozenDinoV3Backbone(
            model_path=args.dinov3_model_path,
            hidden_dim=args.hidden_dim,
            num_cameras=1,
        )
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
