# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from einops import rearrange, repeat
from models.backbone import BackboneResNet, BackboneLinear, BackboneMLP, BackboneLateConcat
from models.transformer import TransformerEncoder, TransformerEncoderLayer, Transformer

import numpy as np

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return rearrange(torch.FloatTensor(sinusoid_table), 'n d -> 1 n d')


class DETRVAE(nn.Module):
    def __init__(self, backbone, transformer, encoder, action_dim, obs_dim, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim

        # output heads
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(chunk_size, hidden_dim)

        self.backbone = backbone
        self.input_proj_robot_state = nn.Linear(obs_dim, hidden_dim)
        self.input_embed = nn.Embedding(2, hidden_dim)  # Only for proprio + latent

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(obs_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+chunk_size, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding

    def forward(self, data):
        qpos = data['qpos']
        image = data['image']
        actions = data.get('actions')
        is_pad = data.get('is_pad')
        
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = rearrange(qpos_embed, 'bs hidden_dim -> bs 1 hidden_dim') 
            cls_input = torch.zeros([bs, 1, self.hidden_dim], dtype=torch.float32).to(qpos.device)

            encoder_input = torch.cat([cls_input, qpos_embed, action_embed], axis=1)  # (bs, seq+2, hidden_dim)
            cls_joint_is_pad = torch.zeros(bs, 2, dtype=torch.bool, device=qpos.device)  # (bs,2)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+2)
            pos_embed = self.pos_table.clone().detach()  # (1, seq+2, hidden_dim)
            pos_embed = repeat(pos_embed, '1 seq hidden_dim -> bs seq hidden_dim', bs=bs)  # (bs, seq+2, hidden_dim)
            encoder_output = self.encoder(
                encoder_input,
                src_key_padding_mask=is_pad,
                pos=pos_embed
            )  # (bs, seq+2, hidden_dim)
            encoder_output = encoder_output[:, 0, :]  # (bs, hidden_dim)
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        camera_features, camera_pos_embed = self.backbone(image)  # (bs, seq, hidden_dim)

        proprio_input = self.input_proj_robot_state(qpos)  # (bs, hidden_dim)
        proprio_input = rearrange(proprio_input, 'bs hidden_dim -> bs 1 hidden_dim')  
        latent_input = rearrange(latent_input, 'bs hidden_dim -> bs 1 hidden_dim')  
        
        src = torch.cat([camera_features, proprio_input, latent_input], dim=1) 
        proprio_latent_pos_embed = repeat(self.input_embed.weight, 's d -> b s d', b=bs)  # (bs, 2, hidden_dim)
        pos_embed = torch.cat([camera_pos_embed, proprio_latent_pos_embed], dim=1)  # (bs, seq, hidden_dim)
        
        query_embed = repeat(self.query_embed.weight, 'c d -> b c d', b=bs)  # (bs, chunk_size, hidden_dim)
        
        hs = self.transformer(
            src=src,
            mask=None,
            query_embed=query_embed,
            pos_embed=pos_embed
        )[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

def build(args):

    if 'resnet' in args.backbone:
        backbone = BackboneResNet(
            hidden_dim=args.hidden_dim,
            use_plucker=args.use_plucker,
            imagenet=('imagenet' in args.backbone),
        )
    elif 'late' in args.backbone:
        backbone = BackboneLateConcat(
            hidden_dim=args.hidden_dim,
            use_plucker=args.use_plucker,
            imagenet=('imagenet' in args.backbone),
            use_r3m=('r3m' in args.backbone),
            latent_drop_prob=args.latent_drop_prob,
        )
    elif args.backbone == 'linear':
        backbone = BackboneLinear(
            hidden_dim=args.hidden_dim,
            patch_size=args.patch_size,
            use_plucker=args.use_plucker
        )
    elif args.backbone == 'mlp':
        backbone = BackboneMLP(
            hidden_dim=args.hidden_dim,
            patch_size=args.patch_size,
            use_plucker=args.use_plucker
        )
    else:
        raise ValueError(f"Invalid backbone: {args.backbone}")

    transformer = Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        ffn_dim=args.ffn_dim,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        norm_cls=nn.LayerNorm,
        activation=args.activation,
    )

    encoder_layer = TransformerEncoderLayer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        activation=args.activation,
        normalize_before=args.pre_norm,
        norm_cls=nn.LayerNorm
    )
    encoder_norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    encoder = TransformerEncoder(encoder_layer, args.enc_layers, encoder_norm)

    model = DETRVAE(
        backbone,
        transformer,
        encoder,
        action_dim=args.action_dim,
        obs_dim=args.obs_dim,
        chunk_size=args.chunk_size,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return model, optimizer 