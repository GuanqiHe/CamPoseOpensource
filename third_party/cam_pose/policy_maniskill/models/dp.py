import math
from typing import Optional, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def _replace_submodules(root_module: nn.Module, predicate, func) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        src_module = parent_module[int(k)] if isinstance(parent_module, nn.Sequential) else getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp: Optional[int] = 32):
        super().__init__()
        c, h, w = input_shape
        self._in_c, self._in_h, self._in_w = c, h, w
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class RgbEncoder(nn.Module):
    def __init__(self, image_shape=(3, 256, 256), backbone="resnet18", use_group_norm: bool = True, num_keypoints: int = 32, use_plucker: bool = False):
        super().__init__()
        self.use_plucker = use_plucker
        backbone_model = getattr(torchvision.models, backbone)(weights=None)
        
        # Modify first conv layer for Plucker if needed
        if use_plucker:
            original_conv = backbone_model.conv1
            backbone_model.conv1 = nn.Conv2d(
                9,  # 3 RGB + 6 Plucker channels
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
        
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if use_group_norm:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=max(1, x.num_features // 16), num_channels=x.num_features),
            )
        # infer feature map shape with a dummy tensor
        with torch.no_grad():
            # Adjust dummy tensor channels based on use_plucker
            if use_plucker:
                dummy_shape = (1, 9, image_shape[1], image_shape[2])  # 9 channels for Plucker
            else:
                dummy_shape = (1, *image_shape)
            dummy = torch.zeros(dummy_shape)
            fmap = self.backbone(dummy)
        fmap_shape = fmap.shape[1], fmap.shape[2], fmap.shape[3]
        self.pool = SpatialSoftmax((fmap_shape[0], fmap_shape[1], fmap_shape[2]), num_kp=num_keypoints)
        self.feature_dim = num_keypoints * 2
        self.proj = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W)
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.proj(x)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            cond_dim: int,
            kernel_size: int = 3,
            n_groups: int = 8,
            cond_predict_scale: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim: int,
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            self.local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups, cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: Tensor, timestep: Union[torch.Tensor, float, int], local_cond: Optional[Tensor] = None, global_cond: Optional[Tensor] = None, **kwargs) -> Tensor:
        # Accept (B, T, D); convert to (B, D, T) for Conv1d
        if sample.dim() != 3:
            raise ValueError("sample must be (B, T, D)")
        x = einops.rearrange(sample, 'b t h -> b h t')

        # time embedding
        if not torch.is_tensor(timestep):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timesteps = timestep[None].to(x.device)
        else:
            timesteps = timestep.to(x.device)
        timesteps = timesteps.expand(x.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)

        # local features
        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            # local_cond: (B, T, D) -> (B, D, T) for conv blocks
            lc = einops.rearrange(local_cond, 'b t h -> b h t')
            resnet1, resnet2 = self.local_cond_encoder
            x_local1 = resnet1(lc, global_feature)
            h_local.append(x_local1)
            x_local2 = resnet2(lc, global_feature)
            h_local.append(x_local2)

        # down path
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        # mid
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # up path
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
            # Add local high-resolution feature after the last upsample so T matches full length
            if idx == (len(self.up_modules) - 1) and len(h_local) > 0:
                x = x + h_local[1]

        x = self.final_conv(x)
        # back to (B, T, D)
        x = einops.rearrange(x, 'b h t -> b t h')
        return x

class DiffusionPolicy(nn.Module):
    """
    Diffusion policy with original DP conditioning:
    - local_cond (time-aligned) for proprio (qpos)
    - global_cond for image features (flattened across cameras and obs steps)
    Uses ConditionalUnet1D from diffusion_policy to mirror conditioning behavior.
    """

    def __init__(self, args):
        super().__init__()
        obs_dim = int(args.obs_dim)
        act_dim = int(args.action_dim)
        n_obs_steps = 1
        # chunk = int(args.chunk_size)
        chunk = 32 # TEMP
        horizon = chunk  # with S=1, T=chunk

        # cameras and image encoder
        image_h, image_w = 256, 256
        use_plucker = args.use_plucker
        image_c = 9 if use_plucker else 3  # 9 channels for Plucker (RGB + 6), 3 for RGB only
        # Use explicit num_side_cam from args
        num_cameras = int(args.num_side_cam)
        self.use_plucker = use_plucker
        self.rgb_encoder = RgbEncoder(image_shape=(image_c, image_h, image_w), backbone="resnet18", use_group_norm=True, num_keypoints=32, use_plucker=use_plucker)
        image_feat_dim_total = self.rgb_encoder.feature_dim * num_cameras * n_obs_steps

        # Conditional UNet mirroring original DP shapes
        self.unet = ConditionalUnet1D(
            input_dim=act_dim,  # actions only as input
            local_cond_dim=obs_dim,  # time-aligned proprio
            global_cond_dim=image_feat_dim_total,  # global image features
            diffusion_step_embed_dim=128,
            down_dims=(512, 1024, 2048),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )

        # DDPM scheduler, paper-style defaults
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=True,
            clip_sample_range=1.0,
        )
        self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps

        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.action_dim = act_dim
        self.obs_dim = obs_dim
        self.num_cameras = num_cameras
        self.chunk_size = chunk

        # Optimizer
        betas = (0.9, 0.999)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(args.lr), betas=betas, weight_decay=float(args.weight_decay))

    def configure_optimizers(self):
        return self.optimizer

    def _build_conds(self, data_dict: dict, device: torch.device, dtype: torch.dtype):
        # qpos: (B, D)
        qpos = data_dict["qpos"].to(device)
        images = data_dict["image"].to(device)  # (B, N, C, H, W)
        B = qpos.shape[0]
        T = self.horizon

        # local_cond: (B, T, obs_dim) — replicate current proprio across horizon
        local_cond = qpos.unsqueeze(1).expand(B, T, self.obs_dim)

        # global_cond: encode images for S=1, flatten across cameras
        # Handle channel selection based on use_plucker
        if not self.use_plucker and images.size(2) > 3:
            # If not using Plucker, only use RGB channels
            images = images[:, :, :3, ...]
        elif self.use_plucker and images.size(2) != 9:
            raise ValueError(f"When use_plucker=True, expected 9 channels (RGB + 6 Plucker) but got {images.size(2)}")
        
        # Ensure we actually have the expected number of cameras; if more were provided,
        # keep the first self.num_cameras.
        # Expect exactly num_cameras views
        if images.shape[1] != self.num_cameras:
            raise RuntimeError(f"Expected {self.num_cameras} camera views, got {images.shape[1]}")
        img_bsnchw = einops.rearrange(images, "b n c h w -> (b n) c h w")
        img_feats = self.rgb_encoder(img_bsnchw)  # ((B*N), F)
        img_feats = einops.rearrange(img_feats, "(b n) f -> b (n f)", b=B, n=self.num_cameras)
        global_cond = img_feats  # (B, N*F)

        return local_cond, global_cond

    @torch.no_grad()
    def _sample(self, batch_size: int, local_cond: Tensor, global_cond: Tensor, generator=None) -> Tensor:
        device = local_cond.device
        dtype = local_cond.dtype
        sample = torch.randn((batch_size, self.horizon, self.action_dim), dtype=dtype, device=device, generator=generator)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            t_vec = torch.full((batch_size,), t, dtype=torch.long, device=device)
            model_output = self.unet(sample, t_vec, local_cond=local_cond, global_cond=global_cond)
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
        return sample

    def __call__(self, data_dict: dict):
        device = data_dict["qpos"].device
        dtype = data_dict["qpos"].dtype
        local_cond, global_cond = self._build_conds(data_dict, device, dtype)

        actions = data_dict.get("actions")
        if actions is not None:
            # Training path
            act = actions[:, : self.horizon, :].to(device)
            eps = torch.randn_like(act)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (act.shape[0],), device=device).long()
            noisy = self.noise_scheduler.add_noise(act, eps, timesteps)
            pred = self.unet(noisy, timesteps, local_cond=local_cond, global_cond=global_cond)
            target = eps  # epsilon prediction
            loss = F.mse_loss(pred, target, reduction="none")  # (B, T, A)

            # Mask padded steps if provided; slice to match horizon (B, horizon)
            if "is_pad" in data_dict:
                is_pad = data_dict["is_pad"][:, : self.horizon].to(device).bool()  # (B, T)
                valid_mask_bt1 = (~is_pad).unsqueeze(-1)  # (B, T, 1)
                valid_mask_bta = valid_mask_bt1.expand_as(loss).float()  # (B, T, A)
                loss = (loss * valid_mask_bta).sum(dim=(1, 2))
                denom = valid_mask_bta.sum(dim=(1, 2)).clamp_min(1.0)  # (B)
                loss = (loss / denom).mean()
            else:
                loss = loss.mean()
            return {"loss": loss}

        # Inference path
        with torch.no_grad():
            B = local_cond.shape[0]
            pred_traj = self._sample(B, local_cond=local_cond, global_cond=global_cond)
            # With S=1, return first chunk_size steps directly
            return pred_traj[:, : self.chunk_size]


