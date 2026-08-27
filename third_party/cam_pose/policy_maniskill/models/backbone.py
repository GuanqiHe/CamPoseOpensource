import torch
import torchvision
from torch import nn
from typing import List
from einops import rearrange, repeat

class FrozenBatchNorm2d(torch.nn.Module):

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneResNet(nn.Module):
    def __init__(self, hidden_dim: int, use_plucker: bool, imagenet: bool):
        super().__init__()
        self.use_plucker = use_plucker
        
        shared = torchvision.models.resnet18(
            replace_stride_with_dilation=[False, False, False],
            pretrained=imagenet, norm_layer=FrozenBatchNorm2d)

        if use_plucker:
            original_conv = shared.conv1
            shared.conv1 = nn.Conv2d(
                9, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

        layers = list(shared.children())[:-2]
        self.resnet = nn.Sequential(*layers)
        
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.pos_embed = nn.Embedding(2 * 8 * 8, hidden_dim)
        self.num_channels = hidden_dim

    def forward(self, images):
        batch_size = images.shape[0]
        num_cams = images.shape[1]
        all_cam_features = []
        
        for cam_id in range(num_cams):
            cam_image = images[:, cam_id]
            if not self.use_plucker:
                cam_image = cam_image[:, :3]
            features = self.resnet(cam_image)
            features = self.input_proj(features)
            features = rearrange(features, 'b d h w -> b (h w) d')
            all_cam_features.append(features)
        
        camera_features = torch.cat(all_cam_features, dim=1)
        pos_tokens = num_cams * 8 * 8
        pos_embed = repeat(self.pos_embed.weight[:pos_tokens], 'n d -> b n d', b=batch_size)
        return camera_features, pos_embed
    
class BackboneLateConcat(nn.Module):
    def __init__(self, hidden_dim: int, use_plucker: bool, imagenet: bool, use_r3m: bool, latent_drop_prob: float = 0.0):
        super().__init__()
        self.use_plucker = use_plucker
        self.latent_drop_prob = float(latent_drop_prob)
        
        if use_r3m:
            import r3m
            r3m_model = r3m.load_r3m("resnet18")
            r3m_resnet = r3m_model.module.convnet
            self.resnet = nn.Sequential(*list(r3m_resnet.children())[:-2])
        else:
            shared_rgb = torchvision.models.resnet18(
                replace_stride_with_dilation=[False, False, False],
                pretrained=imagenet, norm_layer=FrozenBatchNorm2d)
            self.resnet = nn.Sequential(*list(shared_rgb.children())[:-2])

        if self.use_plucker:
            self.plucker_encoder = nn.Sequential(
                nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),
                FrozenBatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                FrozenBatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                FrozenBatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                FrozenBatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
                FrozenBatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

        self.pos_embed = nn.Embedding(2 * 8 * 8, hidden_dim)
        if self.use_plucker:
            self.input_proj = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        else:
            self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        self.num_channels = hidden_dim

    def forward(self, images):
        batch_size = images.shape[0]
        num_cams = images.shape[1]
        all_cam_features = []
        
        for cam_id in range(num_cams):
            cam_image = images[:, cam_id]
            if self.use_plucker:
                plucker_features = self.plucker_encoder(cam_image[:, 3:9])
                rgb_features = self.resnet(cam_image[:, :3])
                if self.latent_drop_prob > 0.0 and self.training:
                    b, c, h, w = rgb_features.shape
                    mask = (torch.rand(b, 1, h, w, device=rgb_features.device) < self.latent_drop_prob)
                    rgb_features = rgb_features.masked_fill(mask, 0.0)
                features = torch.cat([rgb_features, plucker_features], dim=1)
                features = self.input_proj(features)
            else:
                features = self.resnet(cam_image[:, :3])
                features = self.input_proj(features)
            features = rearrange(features, 'b d h w -> b (h w) d')
            all_cam_features.append(features)

        camera_features = torch.cat(all_cam_features, dim=1)
        pos_tokens = num_cams * 8 * 8
        pos_embed = repeat(self.pos_embed.weight[:pos_tokens], 'n d -> b n d', b=batch_size)
        return camera_features, pos_embed
                
    
class BackboneLinear(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int, use_plucker: bool):
        super().__init__()
        self.use_plucker = use_plucker
        self.patch_size = patch_size
        
        rgb_patch_dim = 3 * patch_size * patch_size
        self.rgb_proj = nn.Linear(rgb_patch_dim, hidden_dim, bias=False)
        
        if use_plucker:
            plucker_patch_dim = 6 * patch_size * patch_size
            self.plucker_proj = nn.Linear(plucker_patch_dim, hidden_dim, bias=False)
        else:
            self.pos_embed = nn.Embedding((256 // patch_size) ** 2 * 2, hidden_dim)
        
        self.num_channels = hidden_dim

    def forward(self, images):
        batch_size = images.shape[0]
        num_cams = images.shape[1]
        all_features, all_pos = [], []
        
        for cam_id in range(num_cams):
            cam_img = images[:, cam_id]
            rgb_patches = rearrange(cam_img[:, :3], 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                                  p1=self.patch_size, p2=self.patch_size)

            features = self.rgb_proj(rgb_patches)
            all_features.append(features)
            
            if self.use_plucker:
                plucker_patches = rearrange(cam_img[:, 3:9], 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                                          p1=self.patch_size, p2=self.patch_size)
                pos = self.plucker_proj(plucker_patches)
            else:
                num_patches_per_cam = (256 // self.patch_size) ** 2
                start_idx = cam_id * num_patches_per_cam
                indices = torch.arange(start_idx, start_idx + num_patches_per_cam, device=images.device)
                pos = repeat(self.pos_embed(indices), 'n d -> b n d', b=batch_size)
            
            all_pos.append(pos)
        
        return torch.cat(all_features, dim=1), torch.cat(all_pos, dim=1)
    

class BackboneMLP(nn.Module):
    def __init__(self, hidden_dim: int, patch_size: int, use_plucker: bool):
        super().__init__()
        self.use_plucker = use_plucker
        self.patch_size = patch_size
        
        rgb_patch_dim = 3 * patch_size * patch_size
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_patch_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        if use_plucker:
            plucker_patch_dim = 6 * patch_size * patch_size
            self.plucker_proj = nn.Sequential(
                nn.Linear(plucker_patch_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
        else:
            self.pos_embed = nn.Embedding((256 // patch_size) ** 2 * 2, hidden_dim)
        
        self.num_channels = hidden_dim

    def forward(self, images):
        batch_size = images.shape[0]
        num_cams = images.shape[1]
        all_features, all_pos = [], []
        
        for cam_id in range(num_cams):
            cam_img = images[:, cam_id]
            rgb_patches = rearrange(cam_img[:, :3], 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                                  p1=self.patch_size, p2=self.patch_size)

            features = self.rgb_proj(rgb_patches)
            all_features.append(features)
            
            if self.use_plucker:
                plucker_patches = rearrange(cam_img[:, 3:9], 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                                          p1=self.patch_size, p2=self.patch_size)
                pos = self.plucker_proj(plucker_patches)
            else:
                num_patches_per_cam = (256 // self.patch_size) ** 2
                start_idx = cam_id * num_patches_per_cam
                indices = torch.arange(start_idx, start_idx + num_patches_per_cam, device=images.device)
                pos = repeat(self.pos_embed(indices), 'n d -> b n d', b=batch_size)
            
            all_pos.append(pos)
        
        return torch.cat(all_features, dim=1), torch.cat(all_pos, dim=1)