import os
import time
import random
from pathlib import Path
import numpy as np
import torch
import sys

sys.path.append('..')

import lib.models as models
import lib.networks as networks

import wandb
import pdb
import tifffile
from monai import data
from monai.data import load_decathlon_datalist
import torch.nn as nn
from packaging import version

_persistent_workers = False if version.parse(torch.__version__) < version.parse('1.8.2') else True
from .base_trainer import BaseTrainer
from lib.data.med_transforms import get_scratch_train_transforms, get_vis_transforms

from lib.data.med_datasets import get_train_loader, get_val_loader
from lib.tools.visualization import patches3d_to_grid
from timm.layers.helpers import to_3tuple


class MAE3DEdgeTrainer(BaseTrainer):
    r"""
    3D Masked Autoencoder Trainer
    """

    def __init__(self, args):
        super().__init__(args)
        self.iters_per_epoch = None
        self.train_ds = None
        self.model_name = 'MAE3DEdge'
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            self.model = getattr(models, self.model_name)(
                encoder=getattr(networks, args.enc_arch),
                decoder=getattr(networks, args.dec_arch),
                args=args)
            # 输出模型参数
            # for name, param in self.model.named_parameters():
            #     print(name, param.size())

            if args.pretrain is not None and args.pretrain_type == 'enc_dec' and os.path.exists(args.pretrain):
                print(f"=> Start loading the model weights from {args.pretrain} for test")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                state_dict = checkpoint['state_dict']
                # 提取decoder权重
                for key in list(state_dict.keys()):
                    if key.startswith('decoder.'):
                        new_key = key.replace('decoder.', 'decoder_pixel.', 1)
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]
                    else:
                        state_dict[key] = state_dict[key]
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

            elif args.pretrain is not None and args.pretrain_type == 'enc' and os.path.exists(args.pretrain):
                print(f"=> Start loading vit model weights from {args.pretrain} ")
                checkpoint = torch.load(args.pretrain, map_location='cpu')

                state_dict = checkpoint['state_dict']
                # pdb.set_trace()

                for key in list(state_dict.keys()):
                    if key == 'patch_embed.proj.weight' and \
                            state_dict[
                                'patch_embed.proj.weight'].shape != self.model.encoder.patch_embed.proj.weight.shape:
                        del state_dict['patch_embed.proj.weight']
                        del state_dict['patch_embed.proj.bias']
                        print("===> Del patch_embed.proj.weight and patch_embed.proj.bias")
                    if key == 'pos_embed' and \
                            state_dict['pos_embed'].shape != self.model.encoder_pos_embed.shape:
                        del state_dict[key]
                        print("===> Del pos_embed")
                    if key == 'mask_token' and \
                            state_dict['mask_token'].shape != self.model.mask_token.shape:
                        del state_dict[key]
                        print("===> Del mask_token")

                # 为每个键添加"encoder."前缀
                # state_dict = {"encoder." + key: value for key, value in state_dict.items()}
                msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

            self.wrap_model()

        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert (self.model is not None and self.wrapped_model is not None), \
            "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args

        optim_params = self.get_parameter_groups()
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params,
                                           lr=args.lr,
                                           betas=(args.beta1, args.beta2),
                                           weight_decay=args.weight_decay)

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating dataloader")
            args = self.args

            if args.dataset in ['btcv', 'msd_brats', 'TH_mixed', 'TH_P28']:
                train_transform = get_scratch_train_transforms(args)
                # print("============>batch size: {}".format(self.batch_size))
                self.dataloader = get_train_loader(args,
                                                   batch_size=self.batch_size,
                                                   workers=self.workers,
                                                   train_transform=train_transform)

                from torch.utils.data import DataLoader
                val_transform = get_scratch_train_transforms(args)
                self.val_dataloader = get_val_loader(args,
                                                     batch_size=args.vis_batch_size,
                                                     workers=self.workers,
                                                     val_transform=val_transform)
            elif args.dataset == 'brats20':
                # TODO
                raise NotImplementedError("brats20 transforms and dataloaders on MONAI has not been implemented yet.")
            else:
                raise ValueError("Currently only support brats2020 dataset")

            self.iters_per_epoch = len(self.dataloader)
            print(f"==> Length of train dataloader is {self.iters_per_epoch}")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")

    def run(self):
        args = self.args

        niters = args.start_epoch * self.iters_per_epoch
        if not os.path.exists(args.image_dir):
            os.mkdir(args.image_dir)

        for epoch in range(args.start_epoch, args.epochs):
            # if args.distributed:
            # self.dataloader.sampler.set_epoch(epoch)
            # torch.distributed.barrier()

            # if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
            #     if epoch == args.start_epoch:
            #         print("==> First visualization")
            #         self.vis_reconstruction(niters)

            # train for one epoch
            niters = self.epoch_train(epoch, niters)
            self.val_mae(epoch, niters)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    print(f"=> start saving checkpoint after epoch {epoch + 1}")
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                    }, is_best=False, filename=f'{args.ckpt_dir}/checkpoint_{epoch + 1:04d}.pth.tar')
                    print("=> finish saving checkpoint")

                if epoch + 1 == args.epochs:
                    print(f"=> start saving checkpoint after epoch {epoch + 1}")
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                    }, is_best=False, filename=f'{args.ckpt_dir}/checkpoint_final.pth.tar')
                    print("=> finish saving checkpoint")

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        # switch to train mode
        model.train()

        load_start_time = time.time()

        # cache_rate = 0.25, total_data = 20608, 20608*0.25 = 5152, 5152/32 = 161
        # if (niters * args.batch_size) % (len(datalist) * args.cache_rate) == 0:
        #     self.train_ds.update_cache()
        for i, batch_data in enumerate(train_loader):
            load_time = time.time() - load_start_time

            # pdb.set_trace()

            # adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            # For SSL pretraining, only image data is required for training
            batch_images = batch_data['image']
            # images = batch_data['image']
            edge_label = batch_data['label']

            # 对批次中的每个图像进行归一化
            normalized_images = []
            for img in batch_images:
                img = img / 65535  # 假设图像数据是uint16格式的，先转换为 float 类型
                img_min = img.min()
                img_max = img.max()
                normalized_img = (img - img_min) / (img_max - img_min)
                normalized_images.append(normalized_img)

            images = torch.stack(normalized_images)

            # images = images.cuda(0)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                edge_label = edge_label.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            forward_start_time = time.time()

            # here
            loss, loss_pixel, loss_edge = model(images, edge_label, return_image=False)
            # loss = model(images, edge_label, return_image=False)

            forward_time = time.time() - forward_start_time

            # pdb.set_trace()

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time

            # tifffile.imwrite(os.path.join(args.image_dir, 'train_image' + str(i) + '_' + str(epoch + 1) + '.tiff'),
            #                  image.squeeze().permute(2, 1, 0).detach().cpu().numpy())

            # Log to the screen
            if niters % args.print_freq == 0:
                print(f"Epoch: {epoch}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {optimizer.param_groups[0]['lr']:.08f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.05f} | "
                      # here
                      f"Loss_pixel: {loss_pixel.item():.05f} | "
                      f"loss_edge: {loss_edge.item():.05f}"
                )

                wandb.log(
                    {
                        "lr": optimizer.param_groups[0]['lr'],
                        "Loss": loss.item(),
                        # here
                        "loss_edge": loss_edge.item(),
                        "loss pixel": loss_pixel.item()
                    },
                    step=niters,
                )

            niters += 1
            load_start_time = time.time()
        return niters

    def val_mae(self, epoch, niters):
        print(f"=> start val after {epoch + 1} epochs")

        args = self.args
        val_loader = self.val_dataloader
        model = self.wrapped_model

        model.eval()

        total_loss = 0

        for i, batch_data in enumerate(val_loader):
            image, edge_label = batch_data['image'], batch_data['label']
            # image = image / 65535
            # image = ((image - image.min()) / (image.max() - image.min()))

            # image = image.cuda()
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                edge_label = edge_label.cuda(args.gpu, non_blocking=True)

            # with torch.cuda.amp.autocast(True):
            # here
            loss, masked_image, recon_pixel, recon_edge = model(image, edge_label, return_image=True)
            # loss, masked_image, recon_edge = model(image, edge_label, return_image=True)
            total_loss += loss.item()

            if epoch == 0 or (epoch + 1) % args.vis_freq == 0:
                tifffile.imwrite(os.path.join(args.image_dir, 'val_image' + str(i) + '.tiff'),
                                 image.squeeze().permute(2, 1, 0).detach().cpu().numpy())
                tifffile.imwrite(
                    os.path.join(args.image_dir, 'val_image' + str(i) + '_masked_' + str(epoch + 1) + '.tiff'),
                    masked_image.squeeze().permute(2, 1, 0).detach().cpu().numpy())

                # here
                tifffile.imwrite(
                    os.path.join(args.image_dir, 'val_image' + str(i) + '_recon_pixel_' + str(epoch + 1) + '.tiff'),
                    recon_pixel.squeeze().permute(2, 1, 0).detach().cpu().numpy())

                tifffile.imwrite(
                    os.path.join(args.image_dir, 'val_image' + str(i) + '_recon_edge_' + str(epoch + 1) + '.tiff'),
                    recon_edge.squeeze().permute(2, 1, 0).detach().cpu().numpy())

        mean_loss = total_loss / len(val_loader)

        wandb.log(
            {
                "val_loss": mean_loss,
            },
            step=niters,
        )

        print(f"Epoch: {epoch:03d}/{args.epochs} | "
              f"Val Loss: {mean_loss:.05f}")
        print("=> finish val and visualizing")

    def vis_reconstruction(self, niters=0):
        args = self.args
        loader = self.val_dataloader
        model = self.wrapped_model

        model.eval()

        for batch_data in loader:
            image = batch_data['image']
            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            _, x, recon, masked_x = model(image, return_image=True)

            vis_tensor = torch.cat([x, masked_x, recon], dim=0)

            # visualize
            grid_size = []
            for pa_size, in_size in zip(to_3tuple(args.patch_size), to_3tuple(args.input_size)):
                grid_size.append(in_size // pa_size)
            vis_grid_hw = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size,
                                            in_chans=args.in_chans, hidden_axis='d')
            # import pdb
            # pdb.set_trace()
            # vis_grid_hd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='w')
            # vis_grid_wd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='h')

            print("wandb logging")
            vis_grid_hw = np.array(vis_grid_hw.cpu())
            # pdb.set_trace()
            vis_grid_hw = np.transpose(vis_grid_hw, (1, 2, 0))
            vis_grid_hw = wandb.Image(vis_grid_hw, caption=f"hw_iter{niters:06d}")

            wandb.log(
                {
                    "vis_hw": vis_grid_hw,
                    # "vis_hd": vis_grid_hd,
                    # "vis_wd": vis_grid_wd
                },
                step=niters,
            )
            break
        print("finish wandb logging")

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler'])  # additional line compared with base imple
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
