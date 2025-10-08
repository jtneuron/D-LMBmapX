import pdb

from matplotlib.pyplot import grid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
from timm.layers.helpers import to_3tuple
from concurrent.futures import ThreadPoolExecutor
# from lib.networks import build_3d_sincos_position_embedding
from lib.networks.patch_embed_layers import PatchEmbed3D

__all__ = ["MAE3DHOG"]


def build_3d_sincos_position_embedding(grid_size, embed_dim, num_tokens=1, temperature=10000.):
    grid_size = to_3tuple(grid_size)
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)
    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature ** omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat(
        [torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)],
        dim=1)[None, :, :]

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


def build_perceptron_position_embedding(grid_size, embed_dim, num_tokens=1):
    pos_emb = torch.rand([1, np.prod(grid_size), embed_dim])
    nn.init.normal_(pos_emb, std=.02)

    assert num_tokens == 1 or num_tokens == 0, "Number of tokens must be of 0 or 1"
    if num_tokens == 1:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    return pos_embed


def patchify_image(x, patch_size):
    """
    ATTENTION!!!!!!!
    Different from 2D version patchification: The final axis follows the order of [ph, pw, pd, c] instead of [c, ph, pw, pd]
    """
    # patchify input, [B,C,H,W,D] --> [B,C,gh,ph,gw,pw,gd,pd] --> [B,gh*gw*gd,ph*pw*pd*C]
    B, C, H, W, D = x.shape
    patch_size = to_3tuple(patch_size)
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x = x.reshape(B, C, grid_size[0], patch_size[0], grid_size[1], patch_size[1], grid_size[2],
                  patch_size[2])  # [B,C,gh,ph,gw,pw,gd,pd]
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, np.prod(grid_size),
                                                  np.prod(patch_size) * C)  # [B,gh*gw*gd,ph*pw*pd*C]

    return x


def unpatchify_image(x_patch, patch_size, original_shape):
    """
    Unpatchify the patchified image back to its original shape.
    """
    B, C, H, W, D = original_shape
    patch_size = to_3tuple(patch_size)
    grid_size = (H // patch_size[0], W // patch_size[1], D // patch_size[2])

    x_patch = x_patch.reshape(B, grid_size[0], grid_size[1], grid_size[2], patch_size[0], patch_size[1], patch_size[2],
                              C)  # [B,gh,gw,gd,ph,pw,pd,C]
    x_patch = x_patch.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, C, H, W, D)  # [B,C,H,W,D]

    return x_patch


def batched_shuffle_indices(batch_size, length, device):
    """
    Generate random permutations of specified length for batch_size times
    Motivated by https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    """
    rand = torch.rand(batch_size, length).to(device)
    batch_perm = rand.argsort(dim=1)
    return batch_perm


class MAE3DHOG(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 encoder,
                 decoder,
                 args):
        super().__init__()
        self.args = args
        input_size = to_3tuple(args.input_size)
        patch_size = to_3tuple(args.patch_size)
        self.input_size = input_size
        self.patch_size = patch_size
        # self.output = None
        # self.target = None
        # self.loss = None
        self.loss_pxiel = None
        self.loss_hog = None
        self.loss = None

        out_chans = args.in_chans * np.prod(self.patch_size)
        self.out_chans = out_chans

        grid_size = []
        for in_size, pa_size in zip(input_size, patch_size):
            assert in_size % pa_size == 0, "input size and patch size are not proper"
            grid_size.append(in_size // pa_size)
        self.grid_size = grid_size

        # build positional encoding for encoder and decoder
        if args.pos_embed_type == 'sincos':
            with torch.no_grad():
                self.encoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.encoder_embed_dim,
                                                                            num_tokens=0)
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.decoder_embed_dim,
                                                                            num_tokens=0)
        elif args.pos_embed_type == 'perceptron':
            self.encoder_pos_embed = build_perceptron_position_embedding(grid_size,
                                                                         args.encoder_embed_dim,
                                                                         num_tokens=0)
            with torch.no_grad():
                self.decoder_pos_embed = build_3d_sincos_position_embedding(grid_size,
                                                                            args.decoder_embed_dim,
                                                                            num_tokens=0)

        # build encoder and decoder
        from lib.networks import patch_embed_layers
        embed_layer = getattr(patch_embed_layers, args.patchembed)
        self.encoder = encoder(patch_size=patch_size,
                               in_chans=args.in_chans,
                               embed_dim=args.encoder_embed_dim,
                               depth=args.encoder_depth,
                               num_heads=args.encoder_num_heads,
                               embed_layer=embed_layer)
        self.decoder_pixel = decoder(patch_size=patch_size,
                               num_classes=out_chans,
                               embed_dim=args.decoder_embed_dim,
                               depth=args.decoder_depth,
                               num_heads=args.decoder_num_heads)

        self.decoder_hog = decoder(patch_size=patch_size,
                                     num_classes=out_chans,
                                     embed_dim=args.decoder_embed_dim,
                                     depth=args.decoder_depth,
                                     num_heads=args.decoder_num_heads)

        self.encoder_to_decoder = nn.Linear(args.encoder_embed_dim, args.decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_embed_dim))

        self.patch_norm = nn.LayerNorm(normalized_shape=(out_chans,), eps=1e-6, elementwise_affine=False)

        self.criterion = nn.MSELoss(reduction='mean')

        # initialize encoder_to_decoder and mask token
        nn.init.xavier_uniform_(self.encoder_to_decoder.weight)
        nn.init.normal_(self.mask_token, std=.02)


    def forward(self, images, hog_label, return_image=True):
        args = self.args
        batch_size = images.size(0)
        in_chans = images.size(1)
        original_shape = images.shape

        # pdb.set_trace()
        # print("Batch_size: {}, In_chans: {}".format(x.size(0), x.size(1)))
        assert in_chans == args.in_chans
        out_chans = self.out_chans
        x = patchify_image(images, self.patch_size)  # [B,gh*gw*gd,ph*pw*pd*C]
        l_x = patchify_image(hog_label, self.patch_size)

        # compute length for selected and masked
        length = np.prod(self.grid_size)
        sel_length = int(length * (1 - args.mask_ratio))
        msk_length = length - sel_length

        # generate batched shuffle indices
        shuffle_indices = batched_shuffle_indices(batch_size, length, device=x.device)
        unshuffle_indices = shuffle_indices.argsort(dim=1)

        # select and mask the input patches
        shuffled_x = x.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans))
        shuffled_label_x = l_x.gather(dim=1, index=shuffle_indices[:, :, None].expand(-1, -1, out_chans))

        sel_x = shuffled_x[:, :sel_length, :]
        msk_x = shuffled_x[:, -msk_length:, :]
        msk_label_x = shuffled_label_x[:, -msk_length:, :]

        # select and mask the indices
        # shuffle_indices = F.pad(shuffle_indices + 1, pad=(1, 0), mode='constant', value=0)
        sel_indices = shuffle_indices[:, :sel_length]
        # msk_indices = shuffle_indices[:, -msk_length:]

        # select the position embedings accordingly
        sel_encoder_pos_embed = self.encoder_pos_embed.expand(batch_size, -1, -1).gather(dim=1, index=sel_indices[:, :,
                                                                                                      None].expand(-1,
                                                                                                                   -1,
                                                                                                                   args.encoder_embed_dim))
        # forward encoder & proj to decoder dimension
        sel_x = self.encoder(sel_x, sel_encoder_pos_embed)
        sel_x = self.encoder_to_decoder(sel_x)

        # combine the selected tokens and mask tokens in the shuffled order
        all_x = torch.cat([sel_x, self.mask_token.expand(batch_size, msk_length, -1)], dim=1)

        # shuffle all the decoder positional encoding
        shuffled_decoder_pos_embed = self.decoder_pos_embed.expand(batch_size, -1, -1).gather(dim=1,
                                                                                              index=shuffle_indices[:,
                                                                                                    :, None].expand(-1,
                                                                                                                    -1,
                                                                                                                    args.decoder_embed_dim))
        # add the shuffled positional embedings to encoder output tokens
        all_x = all_x + shuffled_decoder_pos_embed

        # forward decoder
        all_x_pixel = self.decoder_pixel(all_x)
        self.loss_pxiel = self.criterion(input=all_x_pixel[:, -msk_length:, :], target=self.patch_norm(msk_x.detach()))

        # all_x_hog = all_x_hog.gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))
        # self.loss_hog = self.criterion(input=all_x_hog, target=shuffled_label_x)

        all_x_hog = self.decoder_hog(all_x)
        self.loss_hog = self.criterion(input=all_x_hog[:, -msk_length:, :], target=msk_label_x.detach())

        # 总loss计算
        self.loss = (1 - args.hog_weight) * self.loss_pxiel + args.hog_weight * self.loss_hog

        if return_image:
            # unshuffled all the tokens
            masked_x = torch.cat(
                [shuffled_x[:, :sel_length, :], 0. * torch.ones(batch_size, msk_length, out_chans).to(x.device)],
                dim=1).gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))

            # recon: (8, 512, 4096) == (8, 128, 128, 128)
            recon = torch.cat([self.patch_norm(shuffled_x[:, :sel_length, :]), all_x_pixel[:, -msk_length:, :]], dim=1) \
                .gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))

            recon_hog = torch.cat(
                [shuffled_label_x[:, :sel_length, :], all_x_hog[:, -msk_length:, :]], dim=1) \
                .gather(dim=1, index=unshuffle_indices[:, :, None].expand(-1, -1, out_chans))

            masked_x = unpatchify_image(masked_x, self.patch_size, original_shape)
            recon = unpatchify_image(recon, self.patch_size, original_shape)
            recon_hog = unpatchify_image(recon_hog, self.patch_size, original_shape)

            return self.loss, masked_x, recon, recon_hog
        else:
            return self.loss, self.loss_pxiel, self.loss_hog
