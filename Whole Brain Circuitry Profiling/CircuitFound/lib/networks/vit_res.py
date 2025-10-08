import pdb
import torch.nn as nn

__all__ = ['CrossAttention', 'VitResnet']


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


class VitResnet(nn.Module):
    def __init__(self, vit: nn.Module, resnet: nn.Module):
        super().__init__()

        self.vit = vit
        self.resnet = resnet

        self.fc1 = nn.Conv3d(in_channels=256, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv3d(in_channels=512, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv3d(in_channels=1024, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv3d(in_channels=2048, out_channels=768, kernel_size=1, stride=1, padding=0, bias=True)

        self.catn1 = CrossAttention(dim=768, num_heads=12, qkv_bias=True)
        self.catn2 = CrossAttention(dim=768, num_heads=12, qkv_bias=True)
        self.catn3 = CrossAttention(dim=768, num_heads=12, qkv_bias=True)
        self.catn4 = CrossAttention(dim=768, num_heads=12, qkv_bias=True)

        self.norm1 = nn.LayerNorm(768, eps=1e-6)
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.norm3 = nn.LayerNorm(768, eps=1e-6)
        self.norm4 = nn.LayerNorm(768, eps=1e-6)

        self.ffc1 = nn.Conv3d(in_channels=768, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)
        self.ffc2 = nn.Conv3d(in_channels=768, out_channels=512, kernel_size=1, stride=1, padding=0, bias=True)
        self.ffc3 = nn.Conv3d(in_channels=768, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.ffc4 = nn.Conv3d(in_channels=768, out_channels=2048, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        _, vit_x = self.vit(x)
        res_x = self.resnet(x)

        v1 = vit_x[0]
        v2 = vit_x[1]
        v3 = vit_x[2]
        v4 = vit_x[3]

        r1 = self.fc1(res_x[0])
        r2 = self.fc2(res_x[1])
        r3 = self.fc3(res_x[2])
        r4 = self.fc4(res_x[3])

        bs, dim, H1, W1, D1 = r1.shape
        _, _, H2, W2, D2 = r2.shape
        _, _, H3, W3, D3 = r3.shape
        _, _, H4, W4, D4 = r4.shape

        # v1 = v1.view(bs, dim, -1).transpose(1, 2)
        v2 = v2.view(bs, dim, -1).transpose(1, 2)
        v3 = v3.view(bs, dim, -1).transpose(1, 2)
        v4 = v4.view(bs, dim, -1).transpose(1, 2)

        # r1 = r1.view(bs, dim, -1).transpose(1, 2)
        r2 = r2.view(bs, dim, -1).transpose(1, 2)
        r3 = r3.view(bs, dim, -1).transpose(1, 2)
        r4 = r4.view(bs, dim, -1).transpose(1, 2)

        # pdb.set_trace()

        # hyb1 = self.catn1(r1, v1)
        hyb2 = self.catn2(r2, v2)
        hyb3 = self.catn3(r3, v3)
        hyb4 = self.catn4(r4, v4)

        # hyb1 = self.norm1(hyb1).transpose(1, 2).reshape(bs, dim, H1, W1, D1)
        hyb2 = self.norm2(hyb2).transpose(1, 2).reshape(bs, dim, H2, W2, D2)
        hyb3 = self.norm3(hyb3).transpose(1, 2).reshape(bs, dim, H3, W3, D3)
        hyb4 = self.norm4(hyb4).transpose(1, 2).reshape(bs, dim, H4, W4, D4)

        # TODO: try to deal with that return like v1 = self.ffc1(v1), return output: [v1, hyb2, hyb3, hyb4]]

        # hyb1 = self.ffc1(hyb1) + res_x[0]
        hyb2 = self.ffc2(hyb2) + res_x[1]
        hyb3 = self.ffc3(hyb3) + res_x[2]
        hyb4 = self.ffc4(hyb4) + res_x[3]

        return [res_x[0], hyb2, hyb3, hyb4]
        # return [hyb1, hyb2, hyb3, hyb4]
