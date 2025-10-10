# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import pdb

import numpy as np
import torch.nn as nn
import warnings
import torch
import torch.nn.functional as F
from mmcv.utils import Registry, build_from_cfg
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from .segformer_decode_head import BaseDecodeHead
import attr

HEADS = Registry('head')
from IPython import embed


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            conv_cfg=dict(type='Conv3d'),  # 指定使用3D卷积
            norm_cfg=dict(type='BN3d', requires_grad=True)
        )

        self.linear_pred = nn.Conv3d(embedding_dim, self.num_classes, kernel_size=1)
    #
    #     self.upsample_layer = nn.Upsample(
    #         size=(128, 128, 128),  # 目标尺寸
    #         mode='trilinear',  # 三线性插值，适用于3D数据
    #         align_corners=True  # 边角对齐方式·
    #     )
    #
    #     self.apply(self._init_weight_He_)
    #
    # def _init_weight_He_(self, module):
    #     if isinstance(module, nn.Conv3d) \
    #             or isinstance(module, nn.Conv2d) \
    #             or isinstance(module, nn.ConvTranspose2d) \
    #             or isinstance(module, nn.ConvTranspose3d):
    #         module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
    #         if module.bias is not None:
    #             module.bias = nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        # pdb.set_trace()

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w, d = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
        _c4 = resize(_c4, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
        _c3 = resize(_c3, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
        _c2 = resize(_c2, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = self.upsample_layer(x)
        return x


class SegFormerHead_1(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_1, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.neg_slope = 1e-2

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            conv_cfg=dict(type='Conv3d'),  # 指定使用3D卷积
            norm_cfg=dict(type='BN3d', requires_grad=True)
        )

        self.linear_pred = nn.Conv3d(embedding_dim, self.num_classes, kernel_size=1)

        # TASK: 1205_1
        # self.upsample_layer = nn.Sequential(*[
        #     nn.ConvTranspose3d(self.num_classes, self.num_classes, 4, 2, padding=1),
        #     nn.BatchNorm3d(self.num_classes),
        #     nn.GELU(),
        #     nn.ConvTranspose3d(self.num_classes, self.num_classes, 4, 2, padding=1)
        # ])

        # TASK: 1205_3 ,改进上方上采样层，在反卷积后面加入卷积
        # self.upsample_layer = nn.Sequential(*[
        #     nn.ConvTranspose3d(self.num_classes, self.num_classes, 4, 2, padding=1),
        #     nn.Conv3d(self.num_classes, self.num_classes, 3, 1, padding=1),
        #     nn.BatchNorm3d(self.num_classes),
        #     nn.GELU(),
        #     nn.ConvTranspose3d(self.num_classes, self.num_classes, 4, 2, padding=1),
        #     nn.Conv3d(self.num_classes, self.num_classes, 3, 1, padding=1),
        #     # nn.BatchNorm3d(self.num_classes),
        #     # nn.GELU()
        # ])

        # TASK: 1205_4
        self.upsample_layer = nn.Sequential(*[
            nn.ConvTranspose3d(self.num_classes, self.num_classes, 2, 2),
            nn.BatchNorm3d(self.num_classes),
            nn.GELU(),
            nn.ConvTranspose3d(self.num_classes, self.num_classes, 2, 2),
        ])

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w, d = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
        _c4 = resize(_c4, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
        _c3 = resize(_c3, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
        _c2 = resize(_c2, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        # x: torch.Size([1, 2, 32, 32, 32])
        # pdb.set_trace()
        x = self.upsample_layer(x)
        return x


class SegFormerHead_2(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_2, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            conv_cfg=dict(type='Conv3d'),  # 指定使用3D卷积
            norm_cfg=dict(type='BN3d', requires_grad=True)
        )
        self.conv1 = nn.Sequential(*[
            nn.Conv3d(embedding_dim, embedding_dim // 2, kernel_size=1),
            nn.BatchNorm3d(embedding_dim // 2),
            nn.GELU()])

        self.upsample1 = nn.ConvTranspose3d(embedding_dim // 2, embedding_dim // 2, 4, 2, padding=1)

        self.conv2 = nn.Sequential(*[
            nn.Conv3d(embedding_dim // 2, embedding_dim // 4, kernel_size=1),
            nn.BatchNorm3d(embedding_dim // 4),
            nn.GELU()])

        self.upsample2 = nn.ConvTranspose3d(embedding_dim // 4, embedding_dim // 4, 4, 2, padding=1)

        self.conv3 = nn.Sequential(*[
            nn.Conv3d(embedding_dim // 4, embedding_dim // 8, kernel_size=1),
            nn.BatchNorm3d(embedding_dim // 8),
            nn.GELU()])

        self.linear_pred = nn.Conv3d(embedding_dim // 8, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # inputs[0,1,2,3]
        # torch.Size([1, 256, 32, 32, 32])
        # torch.Size([1, 512, 16, 16, 16])
        # torch.Size([1, 1024, 8, 8, 8])
        # torch.Size([1, 2048, 4, 4, 4])
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w, d = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
        _c4 = resize(_c4, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
        _c3 = resize(_c3, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
        _c2 = resize(_c2, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # pdb.set_trace()
        # x: [1, 768, 32, 32, 32]
        # target: [1, 2, 128, 128, 128]
        # channel: 768--384--192--96
        # size: 32--64--128

        x = self.dropout(_c)

        x = self.conv1(x)
        x = self.upsample1(x)
        x = self.conv2(x)
        x = self.upsample2(x)
        x = self.conv3(x)

        x = self.linear_pred(x)
        return x

class SegFormerHead_3(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_3, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            conv_cfg=dict(type='Conv3d'),  # 指定使用3D卷积
            norm_cfg=dict(type='BN3d', requires_grad=True)
        )

        # TASK: 1213
        # self.upsample_layer = nn.Sequential(*[
        #     nn.ConvTranspose3d(embedding_dim, embedding_dim // 2, 4, 2, padding=1),
        #     nn.BatchNorm3d(embedding_dim // 2),
        #     nn.GELU(),
        #     nn.ConvTranspose3d(embedding_dim // 2, embedding_dim // 4, 4, 2, padding=1),
        # ])

        # TASK: 1218
        self.upsample_layer = nn.Sequential(*[
            nn.ConvTranspose3d(embedding_dim, embedding_dim // 4, 2, 2),
            nn.BatchNorm3d(embedding_dim // 4),
            nn.GELU(),
            nn.ConvTranspose3d(embedding_dim // 4, embedding_dim // 8, 2, 2),
            nn.GELU()
        ])

        self.linear_pred = nn.Conv3d(embedding_dim // 8, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # inputs[0,1,2,3]
        # torch.Size([1, 256, 32, 32, 32])
        # torch.Size([1, 512, 16, 16, 16])
        # torch.Size([1, 1024, 8, 8, 8])
        # torch.Size([1, 2048, 4, 4, 4])
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w, d = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3], c4.shape[4])
        _c4 = resize(_c4, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3], c3.shape[4])
        _c3 = resize(_c3, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3], c2.shape[4])
        _c2 = resize(_c2, size=c1.size()[2:], mode='trilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3], c1.shape[4])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # pdb.set_trace()
        # x: [1, 768, 32, 32, 32]
        # target: [1, 2, 128, 128, 128]
        # channel: 768--384--192--96
        # size: 32--64--128

        x = self.dropout(_c)

        x = self.upsample_layer(x)
        x = self.linear_pred(x)
        return x