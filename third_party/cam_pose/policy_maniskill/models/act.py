import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch
import einops
from torchvision.models.vision_transformer import EncoderBlock
from models.transformer import Transformer
from models.detr_vae import build

class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        model, optimizer = build(args)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        
        self.kl_weight = args.kl_weight
        self.prob_drop_proprio = args.prob_drop_proprio
        self.use_cam_pose = args.use_cam_pose
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, data_dict):
        qpos = data_dict['qpos']
        image = data_dict['image']
        actions = data_dict.get('actions', None)
        is_pad = data_dict.get('is_pad', None)
        cam_config = data_dict.get('cam_config', None)
        
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # Image format: [batch, num_cam, channel, height, width]
        assert image.size(2) == 3 or image.size(2) == 9
        image[:, :, :3] = normalize(image[:, :, :3])
        
        # No proprio dropping here; dataset may already handle augmentation if desired

        if self.use_cam_pose:
            zeros = torch.zeros(qpos.size(0), qpos.size(1) - cam_config.size(1)).to(qpos)
            qpos = torch.cat([cam_config, zeros], dim=1)

        # Prepare data for model
        model_data = {
            'qpos': qpos,
            'image': image,
            'actions': actions,
            'is_pad': is_pad
        }

        if actions is not None: # training time
            actions = actions[:, :self.model.chunk_size]
            is_pad = is_pad[:, :self.model.chunk_size]
            model_data['actions'] = actions
            model_data['is_pad'] = is_pad

            a_hat, is_pad_hat, (mu, logvar) = self.model(model_data)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(model_data) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
