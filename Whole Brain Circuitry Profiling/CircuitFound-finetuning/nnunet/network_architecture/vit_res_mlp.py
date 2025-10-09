from matplotlib.pyplot import grid
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
import argparse
from omegaconf import OmegaConf
import numpy as np
import pdb

from timm.layers.helpers import to_3tuple

from nnunet.network_architecture.custom_modules.vision_transformer import ViTBackbone
from nnunet.network_architecture.custom_modules.resnet3D import ResNet
from nnunet.network_architecture.neural_network import SegmentationNetwork

__all__ = [
    'get_inplanes',
    'conv3x3x3',
    'conv1x1x1',
    'Bottleneck',
    'ResNet',
    'ResNeXtBottleneck'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


def get_inplanes():
    return [64, 128, 256, 512]

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 in_chan_last=False):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            self.grid_size.append(im_size // pa_size)
        # self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.in_chans = in_chans
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten
        self.in_chan_last = in_chan_last

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W, D = x.shape
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2], \
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        # x.shape: (1, 768, 8, 8, 8)
        _, _, H, W, D = x.shape
        # pdb.set_trace()
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHWD -> BNC
        x = self.norm(x)
        return x, H, W, D


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXtBottleneck(Bottleneck):
    expansion = 2

    def __init__(self, in_planes, planes, cardinality, stride=1,
                 downsample=None):
        super().__init__(in_planes, planes, stride, downsample)

        mid_planes = cardinality * planes // 32
        self.conv1 = conv1x1x1(in_planes, mid_planes)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(mid_planes,
                               mid_planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               groups=cardinality,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = conv1x1x1(mid_planes, planes * self.expansion)


class ViT_Res_MLP(SegmentationNetwork):
    """General segmenter module for 3D medical images
    """

    def __init__(self, encoder, decoder):
        global input_size
        super().__init__()
        input_size = (128, 128, 128)

        # Do Deep Supervision
        self.do_ds = True
        self.conv_op = nn.Conv3d
        self.num_classes = 2

        self.vit = ViTBackbone(img_size=input_size,
                               patch_size=16,
                               in_chans=1,
                               embed_dim=768,
                               depth=12,
                               NetType='Vit_CNN',
                               num_heads=12,
                               drop_path_rate=0.1,
                               embed_layer=PatchEmbed3D,
                               use_learnable_pos_emb=True,
                               return_hidden_states=True)

        self.resnet = ResNet(block=Bottleneck,
                             layers=[3, 24, 36, 3],
                             block_inplanes=get_inplanes(),
                             n_input_channels=1,
                             n_classes=2)

        self.encoder = encoder(self.vit, self.resnet)


        self.decoder = decoder(
            in_channels=[256, 512, 1024, 2048],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=dict(type='BN3d', requires_grad=True),
            align_corners=False,
            decoder_params=dict(embed_dim=768)
            # loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        )

    def get_num_layers(self):
        return self.encoder.get_num_layers()

    @torch.jit.ignore
    def no_weight_decay(self):
        total_set = set()
        module_prefix_dict = {self.encoder: 'encoder',
                              self.decoder: 'decoder'}
        for module, prefix in module_prefix_dict.items():
            if hasattr(module, 'no_weight_decay'):
                for name in module.no_weight_decay():
                    total_set.add(f'{prefix}.{name}')
        print(f"{total_set} will skip weight decay")
        return total_set

    def forward(self, x_in):
        """
        x_in in shape of [BCHWD]
        """
        # x, hidden_states = self.encoder(x_in)

        x = self.encoder(x_in)

        # x[0].shape: torch.Size([1, 256, 32, 32, 32])
        # x[1].shape: torch.Size([1, 512, 16, 16, 16])
        # x[2].shape: torch.Size([1, 1024, 8, 8, 8])
        # x[3].shape: torch.Size([1, 2048, 4, 4, 4])

        out = self.decoder(x)

        return out
