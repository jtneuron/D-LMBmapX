import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp, to_2tuple, trunc_normal_

# from .common import ConvNormLayer


__all__ = ['vit_res']


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv):
        B, N_q, C_q = x.shape
        _, N_kv, C_kv = kv.shape

        q = self.q(x).reshape(B, N_q, self.num_heads, C_q // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(kv).reshape(B, N_kv, self.num_heads, C_kv // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(kv).reshape(B, N_kv, self.num_heads, C_kv // self.num_heads).permute(0, 2, 1, 3)

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C_q)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class vit_res(nn.Module):
    def __init__(self,
                 vit: nn.Module,
                 resnet: nn.Module,
                 is_sam: bool,
                 catn: bool):
        super(vit_res, self).__init__()

        self.vit = vit
        self.resnet = resnet
        self.is_sam = is_sam
        self.catn = catn

        # self.MLP_0 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=(1,1), stride=(1,1))
        # self.MLP_1 = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=(1,1), stride=(1,1))
        # self.MLP_2 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=(1,1), stride=(1,1))
        # self.MLP_3 = nn.Conv2d(in_channels=768, out_channels=2048, kernel_size=(1,1), stride=(1,1))

        # self.cnr_0 = ConvNormLayer(768, 256, 1, 1, act='relu')
        # self.cnr_1 = ConvNormLayer(768, 512, 1, 1, act='relu')
        # self.cnr_2 = ConvNormLayer(768, 1024, 1, 1, act='relu')
        # self.cnr_3 = ConvNormLayer(768, 2048, 1, 1, act='relu')

        self.fc1 = nn.Conv2d(in_channels=512, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(in_channels=2048, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)

        self.catn1 = CrossAttention(dim=768, num_heads=12, qkv_bias=True)
        self.catn2 = CrossAttention(dim=768, num_heads=12, qkv_bias=True)
        self.catn3 = CrossAttention(dim=768, num_heads=12, qkv_bias=True)

        self.norm1 = nn.LayerNorm(768, eps=1e-6)
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.norm3 = nn.LayerNorm(768, eps=1e-6)

        self.ffc1 = nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True)
        self.ffc2 = nn.Conv2d(in_channels=768, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.ffc3 = nn.Conv2d(in_channels=768, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=True)

        # self.vit._freeze_parameters()

    def forward(self, x):

        vit_x = self.vit(x)
        res_x = self.resnet(x)

        # # torch.Size([4, 768, 128, 128])
        # # torch.Size([4, 768, 64, 64])
        # # torch.Size([4, 768, 32, 32])
        # # torch.Size([4, 768, 16, 16])

        # # torch.Size([4, 512, 64, 64])
        # # torch.Size([4, 1024, 32, 32])
        # # torch.Size([4, 2048, 16, 16])
        # for i in res_x:
        #     print(i.shape)
        # quit()

        if self.catn:
            c1 = self.fc1(res_x[1])
            c2 = self.fc2(res_x[2])
            c3 = self.fc3(res_x[3])

            bs, dim, H1, W1 = c1.shape
            _, _, H2, W2 = c2.shape
            _, _, H3, W3 = c3.shape

            c1 = c1.view(bs, dim, -1).transpose(1, 2)
            c2 = c2.view(bs, dim, -1).transpose(1, 2)
            c3 = c3.view(bs, dim, -1).transpose(1, 2)

            hyb1 = self.catn1(c1, vit_x[1])
            hyb2 = self.catn2(c2, vit_x[2])
            hyb3 = self.catn3(c3, vit_x[3])

            hyb1 = self.norm1(hyb1).transpose(1, 2).reshape(bs, dim, H1, W1)
            hyb2 = self.norm2(hyb2).transpose(1, 2).reshape(bs, dim, H2, W2)
            hyb3 = self.norm3(hyb3).transpose(1, 2).reshape(bs, dim, H3, W3)

            hyb1 = self.ffc1(hyb1) + res_x[1]
            hyb2 = self.ffc2(hyb2) + res_x[2]
            hyb3 = self.ffc3(hyb3) + res_x[3]

            return [res_x[0], hyb1, hyb2, hyb3]

        hyb = []
        if self.is_sam == False:

            mid_0 = self.MLP_0(vit_x[0])
            mid_1 = self.MLP_1(vit_x[1])
            mid_2 = self.MLP_2(vit_x[2])
            mid_3 = self.MLP_3(vit_x[3])

            # mid_0 = self.cnr_0(vit_x[0])
            # mid_1 = self.cnr_1(vit_x[1])
            # mid_2 = self.cnr_2(vit_x[2])
            # mid_3 = self.cnr_3(vit_x[3])

            mid = [mid_0, mid_1, mid_2, mid_3]

            res_x = self.resnet(x, mid)

            return res_x

            hyb_x_0 = mid_1 + res_x[0]
            hyb_x_1 = mid_2 + res_x[1]
            hyb_x_2 = mid_3 + res_x[2]
            # hyb_x_0 = torch.cat([res_x[0], vit_x[1]], dim=1)
            # hyb_x_1 = torch.cat([res_x[1], vit_x[2]], dim=1)
            # hyb_x_2 = torch.cat([res_x[2], vit_x[3]], dim=1)
            hyb.append(hyb_x_0)
            hyb.append(hyb_x_1)
            hyb.append(hyb_x_2)
        else:
            # print(len(res_x))
            # for i in res_x:
            #     print(i.shape)
            # print(vit_x.shape)

            hyb_x_0 = res_x[0]
            hyb_x_1 = torch.cat([res_x[1], vit_x], dim=1)
            hyb_x_2 = res_x[2]
            hyb.append(hyb_x_0)
            hyb.append(hyb_x_1)
            hyb.append(hyb_x_2)

        return hyb
