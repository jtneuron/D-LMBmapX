import math
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import Block
from timm.layers import trunc_normal_, PatchEmbed
from timm.layers.helpers import to_2tuple, to_3tuple

from .mae_vit import build_2d_sincos_position_embedding
from lib.networks.patch_embed_layers import PatchEmbed3D

import time


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=128, patch_size=16, in_chans=1, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', use_learnable_pos_emb=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  # no distill here
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        else:
            grid_size = img_size // patch_size
            self.pos_embed = build_2d_sincos_position_embedding(grid_size, embed_dim, self.num_tokens)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # self.pe_control = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pe_rate = 1

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02, a=-.02, b=.02)
        trunc_normal_(self.cls_token, std=.02, a=-.02, b=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        pe_rate = self.pe_rate
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed * pe_rate)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def compute_grid_size(img_size, patch_size):
    if isinstance(img_size, int) and isinstance(patch_size, int):
        grid_size = img_size // patch_size
    else:
        if isinstance(img_size, (tuple, list)) and isinstance(patch_size, int):
            patch_size = [patch_size, ] * len(img_size)
        elif isinstance(patch_size, (tuple, list)) and isinstance(img_size, int):
            img_size = [img_size, ] * len(patch_size)
        grid_size = []
        for im_size, pa_size in zip(img_size, patch_size):
            grid_size.append(im_size // pa_size)
    return grid_size


class ViTBackbone(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, use_learnable_pos_emb=False, return_hidden_states=False,
                 pos_embed_builder=None, NetType=None, **kwargs):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.NetType = NetType
        self.patch_size = patch_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1  # no distill here
        self.return_hidden_states = return_hidden_states

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        else:
            assert pos_embed_builder is not None, \
                "When noting using learnable pos embed, pos embed builder should be specified"
            grid_size = compute_grid_size(img_size, patch_size)
            self.pos_embed = pos_embed_builder(grid_size, embed_dim, self.num_tokens)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # self.norm0 = norm_layer(embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.norm3 = norm_layer(embed_dim)
        self.norm4 = norm_layer(embed_dim)

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02, a=-.02, b=.02)
        trunc_normal_(self.cls_token, std=.02, a=-.02, b=.02)

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose3d(embed_dim, embed_dim, 2, 2),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose3d(embed_dim, embed_dim, 2, 2)
        ])
        self.up2 = nn.ConvTranspose3d(embed_dim, embed_dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def get_num_layers(self):
        return len(self.blocks)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, time_meters=None):
        # x shape: [1, 1, 128, 128, 128]
        ret_hids = self.return_hidden_states
        s_time = time.perf_counter()
        # x shape: [1,512,768]
        x, H, W, D = self.patch_embed(x)

        # For SwinUNETR
        f0 = x

        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['patchembed'].append(time.perf_counter() - s_time)
        s_time = time.perf_counter()
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x shape: [1,513,768]
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if time_meters is not None:
            torch.cuda.synchronize()
            time_meters['cls+pe'].append(time.perf_counter() - s_time)

        if ret_hids:
            hidden_states_out = []
        accum_block_time = 0
        for i, blk in enumerate(self.blocks):
            s_time = time.perf_counter()
            x = blk(x)
            if time_meters is not None:
                torch.cuda.synchronize()
                duration = time.perf_counter() - s_time
                time_meters[f'{i}_block'].append(duration)
                accum_block_time += duration
            # print(f"num tokens after layer {i+1} is {x.size(1)}")
            if ret_hids:
                hidden_states_out.append(x[:, 1:, :])

        if time_meters is not None:
            time_meters['accum_blocks'].append(accum_block_time)

        # x: torch.Size([1, 513, 768])
        x = self.norm(x)

        if ret_hids and self.NetType == "Vit_CNN":
            f1 = hidden_states_out[2]
            f2 = hidden_states_out[5]
            f3 = hidden_states_out[8]
            f4 = hidden_states_out[11]

            # f1, f2, f3, f4: torch.Size([1, 512, 768])
            # pdb.set_trace()

            B, n, emb_dim = f1.shape

            # Norm and change dim
            f1 = self.norm1(f1).transpose(1, 2).reshape(B, emb_dim, H, W, D)
            f2 = self.norm2(f2).transpose(1, 2).reshape(B, emb_dim, H, W, D)
            f3 = self.norm3(f3).transpose(1, 2).reshape(B, emb_dim, H, W, D)
            f4 = self.norm4(f4).transpose(1, 2).reshape(B, emb_dim, H, W, D)

            f1 = self.up1(f1).contiguous()
            f2 = self.up2(f2).contiguous()
            f3 = self.up3(f3).contiguous()
            f4 = self.up4(f4).contiguous()

            # f1 shape: [1, 768, 32, 32, 32]
            # f2 shape: [1, 768, 16, 16, 16]
            # f3 shape: [1, 768, 8, 8, 8]
            # f4 shape: [1, 768, 4, 4, 4]
            return x[:, 1:, :], [f1, f2, f3, f4]

        elif ret_hids and self.NetType == "SwinUNETR":
            f0 = self.pos_drop(f0)
            f1 = hidden_states_out[2]
            f2 = hidden_states_out[5]
            f3 = hidden_states_out[8]
            f4 = hidden_states_out[11]

            # f1, f2, f3, f4: torch.Size([1, 512, 768])
            # pdb.set_trace()

            B, n, emb_dim = f1.shape

            # Norm and change dim
            # f0 = self.norm0(f0).transpose(1, 2).reshape(B, emb_dim, H, W, D)
            f1 = self.norm1(f1).transpose(1, 2).reshape(B, emb_dim, H, W, D)
            f2 = self.norm2(f2).transpose(1, 2).reshape(B, emb_dim, H, W, D)
            f3 = self.norm3(f3).transpose(1, 2).reshape(B, emb_dim, H, W, D)
            f4 = self.norm4(f4).transpose(1, 2).reshape(B, emb_dim, H, W, D)

            return [f0, f1, f2, f3, f4]

        elif ret_hids and self.NetType == "UNETR":
            # x[:, 1:, :] == hidden_states_out[12]
            return x[:, 1:, :], hidden_states_out

        elif self.NetType == "SAM3D":

            enc_out = x[:, 1:, :]
            B, n, emb_dim = enc_out.shape
            enc_out = enc_out.transpose(1, 2).reshape(B, emb_dim, H, W, D)


            # [B, 768, ]
            return

        else:
            return x[:, 1:, :]
