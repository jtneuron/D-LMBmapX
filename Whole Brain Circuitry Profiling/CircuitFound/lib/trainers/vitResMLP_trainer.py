import os
import math
import time
from functools import partial
from matplotlib.pyplot import grid
import numpy as np
from numpy import nanmean, nonzero, percentile
from torchprofile import profile_macs

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import tifffile
import sys

sys.path.append('..')

import lib.models as models
import lib.networks as networks
from lib.utils import SmoothedValue, concat_all_gather, LayerDecayValueAssigner

import wandb

from lib.data.med_transforms import get_scratch_train_transforms, get_val_transforms, get_post_transforms, \
    get_vis_transforms, get_raw_transforms
from lib.data.med_datasets import get_msd_trainset, get_train_loader, get_val_loader, idx2label_all, btcv_8cls_idx
from lib.tools.visualization import patches3d_to_grid, images3d_to_grid
from .base_trainer import BaseTrainer

from timm.data import Mixup
from timm.utils import accuracy
from timm.layers.helpers import to_3tuple

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
# from monai.transforms import AsDiscrete
from monai.metrics import meandice, hausdorff_distance

from collections import defaultdict, OrderedDict

import pdb


class vitResMLP_trainer(BaseTrainer):
    r"""
    General Segmentation Trainer
    """

    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_name
        self.scaler = torch.cuda.amp.GradScaler()
        self.metric_funcs = OrderedDict([
            ('Dice',
             meandice)
        ])

    def build_model(self):

        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name}")

            self.loss_fn = DiceCELoss(softmax=True,
                                      to_onehot_y=True,
                                      squared_pred=True,
                                      smooth_nr=args.smooth_nr,
                                      smooth_dr=args.smooth_dr)

            self.post_pred, self.post_label = get_post_transforms(args)

            # setup mixup and loss functions
            if args.mixup > 0:
                raise NotImplemented("Mixup for segmentation has not been implemented.")
            else:
                self.mixup_fn = None

            self.model = getattr(models, self.model_name)(encoder=getattr(networks, args.enc_arch),
                                                          decoder=getattr(networks, args.dec_arch),
                                                          args=args)
            # 输出模型参数
            # for name, param in self.model.named_parameters():
            #     print(name, param.size())

            # load pretrained weights
            if args.pretrain is not None and args.pretrain_type == 'enc_dec' and os.path.exists(args.pretrain):
                print(f"=> Start loading the model weights from {args.pretrain} for test")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                state_dict = checkpoint['state_dict']
                msg = self.model.load_state_dict(state_dict, strict=False)
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")
            elif args.pretrain is not None and args.pretrain_type == 'enc' and os.path.exists(args.pretrain):
                print(f"=> Start loading encoder pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                # 遍历模型的参数和名称
                if self.model_name in ['UNETR3D', 'ViT_Res_MLP']:
                    for key in list(state_dict.keys()):
                        if key.startswith('encoder.'):
                            state_dict[key[len('encoder.'):]] = state_dict[key]
                            del state_dict[key]
                        if key.startswith('decoder.'):
                            del state_dict[key]
                        # need to concat and load pos embed. too
                        # TODO: unify the learning of pos embed of pretraining and finetuning
                        if key == 'encoder_pos_embed':
                            pe = torch.zeros([1, 1, state_dict[key].size(-1)])
                            state_dict['pos_embed'] = torch.cat([pe, state_dict[key]], dim=1)
                            del state_dict[key]
                        if key == 'patch_embed.proj.weight' and \
                                state_dict[
                                    'patch_embed.proj.weight'].shape != self.model.encoder.patch_embed.proj.weight.shape:
                            del state_dict['patch_embed.proj.weight']
                            del state_dict['patch_embed.proj.bias']
                        if key == 'pos_embed' and \
                                state_dict['pos_embed'].shape != self.model.encoder.pos_embed.shape:
                            del state_dict[key]
                    msg = self.model.encoder.vit.load_state_dict(state_dict, strict=False)
                elif self.model_name == 'DynSeg3d':
                    if args.pretrain_load == 'enc+dec':
                        for key in list(state_dict.keys()):
                            if key.startswith('decoder.head.') or (
                                    key.startswith('decoder.blocks.') and int(key[15]) > 7):
                                del state_dict[key]
                    elif args.pretrain_load == 'enc':
                        for key in list(state_dict.keys()):
                            if key.startswith('decoder.'):
                                del state_dict[key]
                    msg = self.model.load_state_dict(state_dict, strict=False)

                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

                # freeze layers
                if args.freeze_vit:
                    print("=======> freeze_vit")
                    for name, param in self.model.named_parameters():
                        if "vit.blocks" in name:
                            print("freeze " + name)
                            param.requires_grad = False

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
        model = self.model

        # num_layers = model.get_num_layers()
        # assigner = LayerDecayValueAssigner(
        #     list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        #
        # # optim_params = self.group_params(model)
        # optim_params = self.get_parameter_groups(get_layer_id=partial(assigner.get_layer_id, prefix='encoder.'),
        #                                          get_layer_scale=assigner.get_scale,
        #                                          verbose=False)
        # # TODO: create optimizer factory
        # self.optimizer = torch.optim.AdamW(optim_params,
        #                                    lr=args.lr,
        #                                    betas=(args.beta1, args.beta2),
        #                                    weight_decay=args.weight_decay)
        # 获取所有参数，不区分不同层
        optim_params = [{'params': filter(lambda p: p.requires_grad, model.parameters())}]

        # 创建优化器
        self.optimizer = torch.optim.AdamW(optim_params,
                                           lr=args.lr,
                                           betas=(args.beta1, args.beta2),
                                           weight_decay=args.weight_decay)

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args

            if args.dataset in ['btcv', 'msd_brats', 'TH_mixed', 'TH_P28']:
                # build train dataloader
                if not args.test:
                    train_transform = get_scratch_train_transforms(args)
                    self.dataloader = get_train_loader(args,
                                                       batch_size=self.batch_size,
                                                       workers=self.workers,
                                                       train_transform=train_transform)
                    self.iters_per_epoch = len(self.dataloader)
                    print(f"==> Length of train dataloader is {self.iters_per_epoch}")
                else:
                    self.dataloader = None
                # build val dataloader
                val_transform = get_val_transforms(args)
                self.val_dataloader = get_val_loader(args,
                                                     batch_size=args.val_batch_size,  # batch per gpu
                                                     workers=self.workers,
                                                     val_transform=val_transform)
            elif args.dataset == 'brats20':
                raise NotImplementedError("brats20 transforms and dataloaders on MONAI has not been implemented yet.")
            else:
                raise ValueError("Currently only support brats2020 dataset")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")

    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        if not os.path.exists(args.image_dir):
            os.mkdir(args.image_dir)

        best_metric = 0
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            if epoch == args.start_epoch:
                self.val_seg(epoch=epoch, niters=niters)

            # train for one epoch
            niters = self.epoch_train(epoch, niters)

            # eval after each epoch training
            if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
                metric = self.val_seg(epoch=epoch, niters=niters)
                if epoch == 0:
                    best_metric = metric
                # metric = metric_list[0]
                if metric < best_metric:
                    print(f"=> New val best metric: {metric} | Old val best metric: {best_metric}")

                    best_metric = metric

                    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                        self.save_checkpoint(
                            {
                                'proj_name': args.proj_name,
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'enc': args.enc_arch,
                                'dec': args.enc_arch,
                                'freeze_vit': args.freeze_vit,
                                'batch_size': args.batch_size,
                                'train_dataset': args.data_path,
                                'epochs': args.epochs,
                                'warmup_epochs': args.warmup_epochs,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                            },
                            is_best=False,
                            filename=f'{args.ckpt_dir}/best_model.pth.tar'
                        )
                        print("=> Finish saving best model.")
                else:
                    print(f"=> Still old val best metric: {best_metric}")

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if (epoch + 1) % args.save_freq == 0:
                    # TODO: save the best
                    self.save_checkpoint(
                        {
                            'proj_name': args.proj_name,
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'enc': args.enc_arch,
                            'dec': args.enc_arch,
                            'freeze_vit': args.freeze_vit,
                            'batch_size': args.batch_size,
                            'train_dataset': args.data_path,
                            'epochs': args.epochs,
                            'warmup_epochs': args.warmup_epochs,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(),  # additional line compared with base imple
                        },
                        is_best=False,
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )

                if epoch + 1 == args.epochs:
                    print(f"=> start saving checkpoint after epoch {epoch + 1}")
                    self.save_checkpoint({
                        'proj_name': args.proj_name,
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'enc': args.enc_arch,
                        'dec': args.enc_arch,
                        'freeze_vit': args.freeze_vit,
                        'batch_size': args.batch_size,
                        'train_dataset': args.data_path,
                        'epochs': args.epochs,
                        'warmup_epochs': args.warmup_epochs,
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
        mixup_fn = self.mixup_fn
        loss_fn = self.loss_fn

        # switch to train mode
        model.train()

        load_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
            load_time = time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            image = batch_data['image']
            target = batch_data['label']

            # print(image.shape)

            if args.gpu is not None:
                image = image.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if mixup_fn is not None:
                image, target = mixup_fn(image, target)

            # compute output and loss
            forward_start_time = time.time()
            # forward_start_time_1 = time.perf_counter()

            # with torch.cuda.amp.autocast(True):
            #     loss = self.train_class_batch(model, image, target, loss_fn)

            loss = self.train_class_batch(model, image, target, loss_fn)

            # with torch.cuda.amp.autocast(True):
            #     pred = model(image)
            #     loss = loss_fn(pred, target)

            # pred = model(image)
            # loss = loss_fn(pred, target)

            forward_time = time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time

            # torch.cuda.synchronize()
            # print(f"training iter time is {time.perf_counter() - forward_start_time_1}")

            # Log to the screen
            if i % args.print_freq == 0:
                if 'lr_scale' in optimizer.param_groups[0]:
                    last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
                else:
                    last_layer_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {last_layer_lr:.05f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.03f}")
                if args.rank == 0 and not args.disable_wandb:
                    wandb.log(
                        {
                            "lr": last_layer_lr,
                            "Loss": loss.item(),
                        },
                        step=niters,
                    )

            niters += 1
            # torch.cuda.empty_cache()
            load_start_time = time.time()
        return niters

    @staticmethod
    def train_class_batch(model, samples, target, criterion):
        outputs = model(samples)
        # loss = criterion(outputs[:,1,:,:,:].unsqueeze(1), target)
        loss = criterion(outputs, target)
        # pdb.set_trace()
        return loss

    def val_seg(self, epoch, niters):
        print(f"=> start val after {epoch + 1} epochs")
        args = self.args
        val_loader = self.val_dataloader
        model = self.wrapped_model
        loss_fn = self.loss_fn

        model.eval()

        total_loss = 0

        for i, batch_data in enumerate(val_loader):
            image, target = batch_data['image'], batch_data['label']
            image = image / 65535
            image = ((image - image.min()) / (image.max() - image.min()))

            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # with torch.cuda.amp.autocast(True):
            #     outputs = model(image)
            #     loss = loss_fn(outputs, target)

            outputs = model(image)
            loss = loss_fn(outputs, target)

            # pdb.set_trace()

            outputs = torch.softmax(outputs, dim=1)[:, 1, ...]
            # pdb.set_trace()
            # outputs = torch.sigmoid(outputs)[:, 1, ...]
            outputs = outputs.squeeze().permute(2, 1, 0).detach().cpu().numpy()
            outputs = outputs * 255

            total_loss += loss.item()

            if epoch == 0 or (epoch + 1) % args.vis_freq == 0:
                tifffile.imwrite(os.path.join(args.image_dir, 'val_image_' + str(i) + '.tiff'),
                                 image.squeeze().permute(2, 1, 0).detach().cpu().numpy())

                tifffile.imwrite(os.path.join(args.image_dir, 'val_label_' + str(i) + '.tiff'),
                                 target.squeeze().permute(2, 1, 0).detach().cpu().numpy())

                tifffile.imwrite(os.path.join(args.image_dir, 'val_output_' + str(i) + '_' + str(epoch + 1) + '.tiff'),
                                 outputs.astype(np.uint8))

        mean_loss = total_loss / len(val_loader)

        if args.rank == 0 and not args.disable_wandb:
            wandb.log(
                {
                    "val_loss": mean_loss,
                },
                step=niters,
            )

        print(f"Epoch: {epoch:03d}/{args.epochs} | "
              f"Val Loss: {mean_loss:.05f}")
        print("=> finish val and visualizing")

        # torch.cuda.empty_cache()

        return mean_loss

    @torch.no_grad()
    def evaluate(self, epoch=0, niters=0):
        print("=> Start Evaluating")
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader
        loss_fn = self.loss_fn

        if args.spatial_dim == 3:
            roi_size = (args.roi_x, args.roi_y, args.roi_z)
        elif args.spatial_dim == 2:
            roi_size = (args.roi_x, args.roi_y)
        else:
            raise ValueError(f"Do not support this spatial dimension (={args.spatial_dim}) for now")

        if hasattr(args, 'ts_ratio') and args.ts_ratio != 0:
            assert args.batch_size == 1, "Test mode requires batch size 1"
            ts_samples = int(len(val_loader) * args.ts_ratio)
            val_samples = len(val_loader) - ts_samples
            ts_meters = defaultdict(SmoothedValue)
        else:
            ts_samples = 0
            val_samples = len(val_loader)
            ts_meters = None
        print(f"val samples: {val_samples} and test samples: {ts_samples}")

        # switch to evaluation mode
        model.eval()
        meters = []
        for i, batch_data in enumerate(val_loader):
            image, target = batch_data['image'], batch_data['label']
            if args.gpu is not None:
                image = image.to(args.gpu, non_blocking=True)
                target = target.to(args.gpu, non_blocking=True)

            # compute output
            # with torch.cuda.amp.autocast():
            #     output = sliding_window_inference(image,
            #                                       roi_size=roi_size,
            #                                       sw_batch_size=4,
            #                                       predictor=model,
            #                                       overlap=args.infer_overlap)

            output = model(image)
            target_convert = torch.stack([self.post_label(target_tensor) for target_tensor in decollate_batch(target)],
                                         dim=0)
            output_convert = torch.stack([self.post_pred(output_tensor) for output_tensor in decollate_batch(output)],
                                         dim=0)
            if epoch == 0:
                tifffile.imwrite(
                    os.path.join(args.image_dir, 'eval_label_seq' + str(i) + '_' + str(epoch + 1) + '.tiff'),
                    target_convert.detach().cpu().numpy())

            tifffile.imwrite(
                os.path.join(args.image_dir, 'eval_output_image_seq' + str(i) + '_' + str(epoch + 1) + '.tiff'),
                output_convert[:, 1, :, :, :].detach().cpu().numpy())
            batch_size = image.size(0)

            # idx2label = idx2label_all[args.dataset]
            for metric_name, metric_func in self.metric_funcs.items():
                if i < val_samples:
                    log_meters = meters
                else:
                    log_meters = ts_meters

                metric = metric_func.compute_dice(y_pred=output_convert, y=target_convert, include_background=False)
                metric = metric.cpu().numpy()
                meters.append(metric)
                # pdb.set_trace()
            print(f'==> Evaluating on the {i + 1}th batch is finished.')
        return [np.mean(meters)]

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

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (
                    1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr
