import argparse
import os
import sys

sys.path.append("/phz/registration_3_mutual")
from current_script.script_util import (
    add_dict_to_argparser,
    parse_resume_step_from_filename, set_random_seed, args_to_dict,
)
from util import logger
from current_script.model_util import voxelmorph_defaults
import torch
import json
import time
from torch import nn
from util.data_util.dataset import TrainPairDataset, ValPairDataset
from torch.utils.data import DataLoader
from util.loss.NCCLoss import NCCLoss
from util.loss.MSELoss import MSELoss
from util.loss.DiceLoss import DiceLoss
from util.loss.GradientLoss2D import GradientLoss2D
from model.registration.voxelmorph.layers import SpatialTransformer
from util.metric.registration_metric import folds_percent_metric, ssim_metric, jacobian_determinant
from util.metric.segmentation_metric import dice_metric3
from util.ModelSaver import ModelSaver
from shutil import copyfile
from tqdm import tqdm
from torch.autograd import Variable
from util.visual.visual1.visual_image import cat_imgs, tensor2img
from util.visual.image_util import save_deformation_figure, save_det_figure, save_dvf_figure
from model.registration.voxelmorph.voxelmorph import VxmDense
import monai
import toml

def create_model_var1(
        spatial_dims,
        int_steps,
        use_probs,
        src_feats,
        trg_feats
):
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    return VxmDense(
        spatial_dims=spatial_dims,
        nb_unet_features=[enc_nf, dec_nf],
        int_steps=int_steps,
        use_probs=use_probs,
        src_feats=src_feats,
        trg_feats=trg_feats,
    )


def voxelmorph_defaults_var1():
    config = dict(
        spatial_dims=2,
        int_steps=0,
        use_probs=False,
        src_feats=2,
        trg_feats=2,
    )
    return config


def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def main():
    set_random_seed(seed=0)
    args = create_argparser().parse_args()
    if len(args.resolution) > 0:
        args.resolution = tuple([int(i) for i in args.resolution.split(',')])
    os.environ['LOGDIR'] = args.logdir
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
    logger.configure()
    logger.log(list(vars(args).items()))
    logger.log(list(voxelmorph_defaults_var1().items()))
    logger.log(f"start train time: {time.asctime()}")
    train_path = os.path.abspath(__file__)
    logger.log(f"train file path: {train_path}")
    copyfile(train_path, os.path.join(logger.get_dir(), os.path.basename(train_path)))

    model_r1 = create_model_var1(**args_to_dict(args, voxelmorph_defaults_var1().keys()))
    model_r2 = create_model_var1(**args_to_dict(args, voxelmorph_defaults_var1().keys()))
    model_c1 = create_model_var1(**args_to_dict(args, voxelmorph_defaults_var1().keys()))
    model_c2 = create_model_var1(**args_to_dict(args, voxelmorph_defaults_var1().keys()))

    total_params = sum([param.nelement() for param in model_r1.parameters()])
    logger.log("Number of parameter: %.2fM" % (total_params / 1e6))

    optimizer_r1 = torch.optim.Adam(model_r1.parameters(), lr=args.lr)
    optimizer_r2 = torch.optim.Adam(model_r2.parameters(), lr=args.lr)
    optimizer_c1 = torch.optim.Adam(model_c1.parameters(), lr=args.lr)
    optimizer_c2 = torch.optim.Adam(model_c2.parameters(), lr=args.lr)

    device = torch.device("cuda:0" if len(args.gpu) > 0 else "cpu")
    resume_epoch = 0
    if len(args.resume_checkpoint) > 0:
        logger.log(f"loading model from {args.resume_checkpoint}")
        resume_epoch = parse_resume_step_from_filename(args.resume_checkpoint)
        state_dict = torch.load(args.resume_checkpoint, map_location=device)
        model_r1.load_state_dict(state_dict['model_r1'])
        optimizer_r1.load_state_dict(state_dict['optimizer_r1'])
        model_c1.load_state_dict(state_dict['model_c1'])
        optimizer_c1.load_state_dict(state_dict['optimizer_c1'])

        for state in optimizer_r1.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        for state in optimizer_c1.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        model_r2.load_state_dict(state_dict['model_r2'])
        optimizer_r2.load_state_dict(state_dict['optimizer_r2'])
        model_c2.load_state_dict(state_dict['model_c2'])
        optimizer_c2.load_state_dict(state_dict['optimizer_c2'])

        for state in optimizer_r2.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        for state in optimizer_r2.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    model_r1.to(device)
    model_r2.to(device)
    model_c1.to(device)
    model_c2.to(device)

    gpu_num = len(args.gpu.split(","))

    if gpu_num > 1:
        model_r1 = torch.nn.DataParallel(model_r1, device_ids=[i for i in range(gpu_num)])
        model_r2 = torch.nn.DataParallel(model_r2, device_ids=[i for i in range(gpu_num)])
        model_c1 = torch.nn.DataParallel(model_c1, device_ids=[i for i in range(gpu_num)])
        model_c2 = torch.nn.DataParallel(model_c2, device_ids=[i for i in range(gpu_num)])
    model_without_ddp_r1 = model_r1.module if gpu_num > 1 else model_r1
    model_without_ddp_r2 = model_r2.module if gpu_num > 1 else model_r2
    model_without_ddp_c1 = model_c1.module if gpu_num > 1 else model_c1
    model_without_ddp_c2 = model_c2.module if gpu_num > 1 else model_c2

    if args.data_dir.endswith('toml'):
        with open(args.data_dir, 'r') as f:
            dataset_config = toml.load(f)
    elif args.data_dir.endswith('json'):
        with open(args.data_dir, 'r') as f:
            dataset_config = json.load(f)
    else:
        raise Exception("dataset config file format error")

    num_classes = dataset_config['region_number']
    registration_type = dataset_config["registration_type"]

    transform = None
    if len(args.resolution) > 0:
        transform = monai.transforms.ResizeD(spatial_size=args.resolution,
                                             keys=['volume1', 'volume2', 'label1_u', 'label1_i', 'label2'],
                                             mode=['trilinear', 'trilinear', 'nearest', 'nearest', 'nearest'])
    train_dataset = TrainPairDataset(dataset_config, dataset_type='train',
                                     registration_type=registration_type,
                                     transform=transform)

    val_dataset = TrainPairDataset(dataset_config, dataset_type='test',
                                 registration_type=registration_type,
                                 transform=transform)
    logger.log(f'train dataset size: {len(train_dataset)}')
    logger.log(f'val dataset size: {len(val_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    modelsaver = ModelSaver(args.max_save_num)
    
    mode = 'u'
    part_size = args.part_size
    for epoch in range(resume_epoch + 1, args.epoches + 1):
        flag = epoch // part_size
        if flag % 2 == 0:
            mode = 'u'
        else:
            mode = 'i' 
        run_train(args, model_r1, model_r2, model_c1, model_c2, optimizer_r1, optimizer_r2, optimizer_c1, optimizer_c2, train_dataloader, device, epoch, num_classes, mode)
        # if epoch % args.val_interval == 0:
        #     run_val(args, model_r1, model_r2, model_c1, model_c2, val_dataloader, device, epoch, num_classes, mode)
        if epoch % args.save_interval == 0:
            state_dict = {"model_r1": model_without_ddp_r1.state_dict(),
                          "model_r2": model_without_ddp_r2.state_dict(),
                          "model_c1": model_without_ddp_c1.state_dict(),
                          "model_c2": model_without_ddp_c2.state_dict(),
                          "optimizer_r1": optimizer_r1.state_dict(),
                          "optimizer_r2": optimizer_r2.state_dict(),
                          "optimizer_c1": optimizer_c1.state_dict(),
                          "optimizer_c2": optimizer_c2.state_dict()}
            modelsaver.save(logger.get_dir(), state_dict, epoch)
    logger.log(f"end train time: {time.asctime()}")


def extract_slices(img, i, batch_size, dim):
    if dim == 2:
        return img[:, :, i:i + batch_size, :, :]
    elif dim == 3:
        return img[:, :, :, i:i + batch_size, :]
    elif dim == 4:
        return img[:, :, :, :, i:i + batch_size]
    else:
        raise Exception("dim error")


def extract_reg_slices(volume1, volume2, label1, label2, i, batch_size, dim):
    img1 = extract_slices(volume1, i, batch_size, dim)
    img2 = extract_slices(volume2, i, batch_size, dim)
    mask1 = extract_slices(label1, i, batch_size, dim) if label1 != [] else []
    mask2 = extract_slices(label2, i, batch_size, dim) if label2 != [] else []
    return img1, img2, mask1, mask2


def permute_img(img, B, C, H, W, D, dim):
    if dim == 2:
        return img.permute((0, 2, 1, 3, 4)).reshape((B * H, C, W, D)).contiguous()
    elif dim == 3:
        return img.permute((0, 3, 1, 2, 4)).reshape((B * W, C, H, D)).contiguous()
    elif dim == 4:
        return img.permute((0, 4, 1, 2, 3)).reshape((B * D, C, H, W)).contiguous()
    else:
        raise Exception("dim error")


def permute_reg_imgs(img1, img2, mask1, mask2, B, C, H, W, D, dim):
    img1 = permute_img(img1, B, C, H, W, D, dim)
    img2 = permute_img(img2, B, C, H, W, D, dim)
    mask1 = permute_img(mask1, B, C, H, W, D, dim) if mask1 != [] else []
    mask2 = permute_img(mask2, B, C, H, W, D, dim) if mask2 != [] else []
    return img1, img2, mask1, mask2


def unpermute_img(img, B, C, H, W, D, dim):
    if img == []:
        return img

    if dim == 2:
        return img.reshape((B, H, C, W, D)).permute((0, 2, 1, 3, 4)).contiguous()
    elif dim == 3:
        return img.reshape((B, W, C, H, D)).permute((0, 2, 3, 1, 4)).contiguous()
    elif dim == 4:
        return img.reshape((B, D, C, H, W)).permute((0, 2, 3, 4, 1)).contiguous()
    else:
        raise Exception("dim error")


def concat_dvf_3d(dvf, dim):
    remaining_channel = torch.zeros((dvf.shape[0], 1) + tuple(dvf.shape[2:]), device=dvf.device).float()
    dvf = torch.cat((dvf, remaining_channel), dim=1)
    if dim == 2:
        dvf = dvf[:, [2, 0, 1], ...]
    elif dim == 3:
        dvf = dvf[:, [0, 2, 1], ...]
    return dvf


def run_train(args, model_r1, model_r2, model_c1, model_c2, optimizer_r1, optimizer_r2, optimizer_c1, optimizer_c2, train_dataloader, device, epoch, num_classes, mode):

    model_r1.train()
    model_r2.train()
    model_c1.train()
    model_c2.train()    

    if mode == 'u':
        freeze(model_c1)
        freeze(model_c2)
    if mode == 'i':
        freeze(model_r1)
        freeze(model_r2)

    batch_size = args.batch_size
    view_1_dim = args.view_1_dim
    view_2_dim = args.view_2_dim
    logger.logkv("epoch", epoch)
    for img_dict in tqdm(train_dataloader):
        volume1, volume2 = img_dict['volume1'], img_dict['volume2']
        label1, label2 = img_dict[f'label1_{mode}'], img_dict['label2']
        
        volume1 = volume1.to(device)
        label1 = label1.to(device) if label1 != [] else label1

        volume2 = volume2.to(device)
        label2 = label2.to(device) if label2 != [] else label2

        view_1_size = volume1.shape[view_1_dim]
        view_1_warp_volume1, view_1_warp_label1 = [], []
        for i in range(0, view_1_size, batch_size):

            img1, img2, mask1, mask2 = extract_reg_slices(volume1, volume2, label1, label2, i, batch_size, view_1_dim)
            B, C, H, W, D = img1.shape
            img1, img2, mask1, mask2 = permute_reg_imgs(img1, img2, mask1, mask2, B, C, H, W, D, view_1_dim)

            warp_img1, warp_mask1 = train_one_view(args, epoch, model_r1, model_c1, optimizer_r1, optimizer_c1, img1, img2, mask1, mask2, "view_1_", mode=mode, is_consistency=False)

            warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_1_dim)
            warp_mask1 = unpermute_img(warp_mask1, B, C, H, W, D, view_1_dim)

            view_1_warp_volume1.append(warp_img1)
            if warp_mask1 != []:
                view_1_warp_label1.append(warp_mask1)
        view_1_warp_volume1 = torch.cat(view_1_warp_volume1, dim=view_1_dim).detach()
        if view_1_warp_label1 != []:
            view_1_warp_label1 = torch.cat(view_1_warp_label1, dim=view_1_dim).detach()

        view_2_size = volume1.shape[view_2_dim]
        view_2_warp_volume1, view_2_warp_label1 = [], []
        for i in range(0, view_2_size, batch_size):

            img1, img2, mask1, mask2 = extract_reg_slices(volume1, volume2, label1, label2, i, batch_size, view_2_dim)
            B, C, H, W, D = img1.shape
            img1, img2, mask1, mask2 = permute_reg_imgs(img1, img2, mask1, mask2, B, C, H, W, D, view_2_dim)

            warp_img1, warp_mask1 = train_one_view(args, epoch, model_r2, model_c2, optimizer_r2, optimizer_c2, img1, img2, mask1, mask2, "view_2_", mode=mode, is_consistency=False)

            warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_2_dim)
            warp_mask1 = unpermute_img(warp_mask1, B, C, H, W, D, view_2_dim)

            view_2_warp_volume1.append(warp_img1)
            if warp_mask1 != []:
                view_2_warp_label1.append(warp_mask1)
        view_2_warp_volume1 = torch.cat(view_2_warp_volume1, dim=view_2_dim).detach()
        if view_2_warp_label1 != []:
            view_2_warp_label1 = torch.cat(view_2_warp_label1, dim=view_2_dim).detach()


        # 传播一致性损失
        volume2 = view_2_warp_volume1.to(device)
        label2 = view_2_warp_label1.to(device) if view_2_warp_label1 != [] else view_2_warp_label1
        view_1_size = volume1.shape[view_1_dim]
        for i in range(0, view_1_size, batch_size):

            img1, img2, mask1, mask2 = extract_reg_slices(volume1, volume2, label1, label2, i, batch_size, view_1_dim)
            B, C, H, W, D = img1.shape
            img1, img2, mask1, mask2 = permute_reg_imgs(img1, img2, mask1, mask2, B, C, H, W, D, view_1_dim)

            warp_img1, warp_mask1 = train_one_view(args, epoch, model_r1, model_c1, optimizer_r1, optimizer_c1, img1, img2, mask1, mask2, "view_1_consistency_", mode=mode, is_consistency=True)
        
        
        volume2 = view_1_warp_volume1.to(device)
        label2 = view_1_warp_label1.to(device) if view_1_warp_label1 != [] else view_1_warp_label1
        view_2_size = volume1.shape[view_2_dim]
        for i in range(0, view_2_size, batch_size):

            img1, img2, mask1, mask2 = extract_reg_slices(volume1, volume2, label1, label2, i, batch_size, view_2_dim)
            B, C, H, W, D = img1.shape
            img1, img2, mask1, mask2 = permute_reg_imgs(img1, img2, mask1, mask2, B, C, H, W, D, view_2_dim)

            warp_img1, warp_mask1 = train_one_view(args, epoch, model_r2, model_c2, optimizer_r2, optimizer_c2, img1, img2, mask1, mask2, "view_2_consistency_", mode=mode, is_consistency=True)

        
        # 打印3d一致性损失
        loss_dict = compute_consistency_loss(args, view_1_warp_volume1, view_2_warp_volume1,
                                             view_1_warp_label1, view_2_warp_label1)
        for key, value in loss_dict.items():
            logger.logkv_mean(key, value.item())

    logger.dumpkvs()
    if mode == 'u':
        unfreeze(model_c1)
        unfreeze(model_c2)
    if mode == 'i':
        unfreeze(model_r1)
        unfreeze(model_r2)
    
    

def train_one_view(args, epoch, model_r, model_c, optimizer_r, optimizer_c, img1, img2, mask1, mask2, identifier, mode='u', is_consistency=False):
    _img1 = torch.cat([img1, mask1.float()], dim=1)
    _img2 = torch.cat([img2, mask2.float()], dim=1)


    dvf_i_r = model_r(_img1, _img2)
    dvf_i_c = model_c(_img1, _img2)

    loss_dict = compute_lossv(args, epoch, dvf_i_r, dvf_i_c, img1, img2, mask1, mask2, mode, is_consistency)
    for key, value in loss_dict.items():
        logger.logkv_mean(identifier + key, value.item())
    optimizer_r.zero_grad()
    optimizer_c.zero_grad()
    loss_dict["total_loss"].backward()
    optimizer_r.step()
    optimizer_c.step()
    if mode == 'u':
        dvf_i = dvf_i_r
    if mode == 'i':
        dvf_i = dvf_i_c
    warp_img1 = SpatialTransformer(mode='bilinear')(img1, dvf_i)
    warp_mask1 = []
    if mask1 != []:
        warp_mask1 = SpatialTransformer(mode='bilinear')(mask1.float(), dvf_i)
    return warp_img1, warp_mask1


def run_val(args, model_r1, model_r2, model_c1, model_c2, val_dataloader, device, epoch, num_classes, mode):
    model_r1.eval()
    model_r2.eval()
    model_c1.eval()
    model_c2.eval()
    batch_size = args.batch_size
    view_1_dim = args.view_1_dim
    view_2_dim = args.view_2_dim
    with torch.no_grad():
        for img_dict in tqdm(val_dataloader):
            volume1, volume2 = img_dict['volume1'], img_dict['volume2']
            label1, label2 = img_dict[f'label1_{mode}'], img_dict['label2']
            id1, id2 = img_dict['id1'][0], img_dict['id2'][0]
            volume1 = volume1.to(device)
            label1 = label1.to(device) if label1 != [] else label1

            volume2 = volume2.to(device)
            label2 = label2.to(device) if label2 != [] else label2

            view_1_size = volume1.shape[view_1_dim]
            view_1_warp_volume1, view_1_warp_label1, view_1_dvf = [], [], []
            view_1_warp_label1_nearest = []
            for i in range(0, view_1_size, batch_size):
                img1, img2, mask1, mask2 = extract_reg_slices(volume1, volume2, label1, label2,
                                                              i, batch_size, view_1_dim)
                B, C, H, W, D = img1.shape
                img1, img2, mask1, mask2 = permute_reg_imgs(img1, img2, mask1, mask2, B, C, H, W, D, view_1_dim)
                
                # TEST 只输入主脑区的信息
                #mask1_with_main_label = mask1.clone()
                #mask1_with_main_label[mask1_with_main_label > 5] = 1
                #mask2_with_main_label = mask2.clone()
                #mask2_with_main_label[mask2_with_main_label > 5] = 1

                warp_img1, warp_mask1, dvf_i = val_one_view(args, epoch, model_r1, model_c1, img1, img2, mask1, mask2, "val_view_1_", mode=mode)

                if mask1 != []:
                    warp_mask1_nearest = SpatialTransformer(mode='nearest')(mask1.float(), dvf_i)
                    warp_mask1_nearest = unpermute_img(warp_mask1_nearest, B, C, H, W, D, view_1_dim)
                    view_1_warp_label1_nearest.append(warp_mask1_nearest)

                warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_1_dim)
                warp_mask1 = unpermute_img(warp_mask1, B, C, H, W, D, view_1_dim)
                dvf_i = unpermute_img(dvf_i, B, dvf_i.shape[1], H, W, D, view_1_dim)

                view_1_warp_volume1.append(warp_img1)
                if warp_mask1 != []:
                    view_1_warp_label1.append(warp_mask1)
                view_1_dvf.append(dvf_i)

            view_1_warp_volume1 = torch.cat(view_1_warp_volume1, dim=view_1_dim)
            if view_1_warp_label1 != []:
                view_1_warp_label1 = torch.cat(view_1_warp_label1, dim=view_1_dim)
            view_1_warp_label1_nearest = torch.cat(view_1_warp_label1_nearest, dim=view_1_dim)

            view_1_dvf = torch.cat(view_1_dvf, dim=view_1_dim)
            view_1_dvf = concat_dvf_3d(view_1_dvf, dim=view_1_dim)

            view_2_size = volume1.shape[view_2_dim]
            view_2_warp_volume1, view_2_warp_label1, view_2_dvf = [], [], []
            view_2_warp_label1_nearest = []
            for i in range(0, view_2_size, batch_size):
                img1, img2, mask1, mask2 = extract_reg_slices(volume1, volume2, label1, label2,
                                                              i, batch_size, view_2_dim)
                B, C, H, W, D = img1.shape
                img1, img2, mask1, mask2 = permute_reg_imgs(img1, img2, mask1, mask2, B, C, H, W, D, view_2_dim)

                # TEST 只输入主脑区的信息
                #mask1_with_main_label = mask1.clone()
                #mask1_with_main_label[mask1_with_main_label > 5] = 1
                #mask2_with_main_label = mask2.clone()
                #mask2_with_main_label[mask2_with_main_label > 5] = 1

                warp_img1, warp_mask1, dvf_i = val_one_view(args, epoch, model_r2, model_c2, img1, img2, mask1, mask2, "val_view_2_", mode=mode)

                if mask1 != []:
                    warp_mask1_nearest = SpatialTransformer(mode='nearest')(mask1.float(), dvf_i)
                    warp_mask1_nearest = unpermute_img(warp_mask1_nearest, B, C, H, W, D, view_2_dim)
                    view_2_warp_label1_nearest.append(warp_mask1_nearest)

                warp_img1 = unpermute_img(warp_img1, B, C, H, W, D, view_2_dim)
                warp_mask1 = unpermute_img(warp_mask1, B, C, H, W, D, view_2_dim)
                dvf_i = unpermute_img(dvf_i, B, dvf_i.shape[1], H, W, D, view_2_dim)

                view_2_warp_volume1.append(warp_img1)
                if warp_mask1 != []:
                    view_2_warp_label1.append(warp_mask1)
                view_2_dvf.append(dvf_i)

            view_2_warp_volume1 = torch.cat(view_2_warp_volume1, dim=view_2_dim)
            if view_2_warp_label1 != []:
                view_2_warp_label1 = torch.cat(view_2_warp_label1, dim=view_2_dim)
            view_2_warp_label1_nearest = torch.cat(view_2_warp_label1_nearest, dim=view_2_dim)

            view_2_dvf = torch.cat(view_2_dvf, dim=view_2_dim)
            view_2_dvf = concat_dvf_3d(view_2_dvf, dim=view_2_dim)

            loss_dict = compute_consistency_loss(args, view_1_warp_volume1, view_2_warp_volume1,
                                                 view_1_warp_label1, view_2_warp_label1)
            for key, value in loss_dict.items():
                logger.logkv_mean("val_" + key, value.item())

            view_1_metric_dict = compute_metric(view_1_dvf, view_1_warp_volume1, view_1_warp_label1,
                                                volume2, label2, num_classes)
            for key, value in view_1_metric_dict.items():
                logger.logkv_mean("val_view_1_" + key, value)

            view_2_metric_dict = compute_metric(view_2_dvf, view_2_warp_volume1, view_2_warp_label1,
                                                volume2, label2, num_classes)
            for key, value in view_2_metric_dict.items():
                logger.logkv_mean("val_view_2_" + key, value)
            output_dir = os.path.join(logger.get_dir(), "val_img", str(epoch), id1 + "_" + id2)
            os.makedirs(output_dir, exist_ok=True)
            save_img_list(img_list1=[volume1, volume2, view_1_warp_volume1],
                          img_list2=[label1, label2, view_1_warp_label1_nearest],
                          view_dim=view_1_dim, num_classes=num_classes,
                          output_path=os.path.join(output_dir, "view_1.png")
                          )
            save_img_list(img_list1=[volume1, volume2, view_2_warp_volume1],
                          img_list2=[label1, label2, view_2_warp_label1_nearest],
                          view_dim=view_2_dim, num_classes=num_classes,
                          output_path=os.path.join(output_dir, "view_2.png")
                          )
            save_dvf(view_1_dvf, output_dir, "view_1")
            save_dvf(view_2_dvf, output_dir, "view_2")
        # logger.dumpkvs()


def save_dvf(dvf, output_dir, view_tag):
    save_deformation_figure(output_dir, view_tag + '_deformation', dvf[0].detach().cpu(),
                            grid_spacing=dvf[0].shape[-1] // 50, linewidth=1, color='darkblue')
    save_dvf_figure(output_dir, view_tag + '_dvf', dvf[0].detach().cpu())
    det = jacobian_determinant(dvf[0].detach().cpu())
    save_det_figure(output_dir, view_tag + '_jacobian_det', det, normalize_by='slice', threshold=0,
                    cmap='Blues')
    # save_RGB_dvf_figure(output_dir, view_tag + '_rgb_dvf', dvf[0].detach().cpu())


def save_img_list(img_list1, img_list2, num_classes, output_path, view_dim):
    interval = img_list1[0].shape[view_dim] // 15
    img_list1 = [img.cpu() for img in img_list1]
    img_list2 = [img.cpu() / num_classes for img in img_list2]
    img_list = img_list1 + img_list2
    img = cat_imgs(img_list, interval=interval, view_dim=view_dim)
    img = tensor2img(img, nrow=1, min_max=(0, 1))

    save_img(img, output_path)


def val_one_view(args, epoch, model_r, model_c, img1, img2, mask1, mask2, identifier, mode='u'):
    _img1 = torch.cat([img1, mask1.float()], dim=1)
    _img2 = torch.cat([img2, mask2.float()], dim=1)


    dvf_i_r = model_r(_img1, _img2)
    dvf_i_c = model_c(_img1, _img2)

    loss_dict = compute_lossv(args, epoch, dvf_i_r, dvf_i_c, img1, img2, mask1, mask2, mode=mode)
    for key, value in loss_dict.items():
        logger.logkv_mean(identifier + key, value.item())
    if mode == 'u':
        dvf_i = dvf_i_r
    if mode == 'i':
        dvf_i = dvf_i_c
    warp_img1 = SpatialTransformer(mode='bilinear')(img1, dvf_i)
    warp_mask1 = []
    if mask1 != []:
        warp_mask1 = SpatialTransformer(mode='bilinear')(mask1.float(), dvf_i)
    return warp_img1, warp_mask1, dvf_i


def compute_metric(dvf, warp_volume1, warp_label1, volume2, label2, num_classes):
    metric_dict = dict()
    # metric_dict['dice'] = dice_metric3(warp_label1, label2, num_classes, reduction="mean").item()
    dice_list = dice_metric3(warp_label1, label2, num_classes, reduction=None)
    for i, value in enumerate(dice_list):
        metric_dict[f'dice_class_{i}'] = value
    # metric_dict['ASD'] = ASD_metric2(warp_label1, label2, num_classes).item()
    # metric_dict['HD'] = HD_metric2(warp_label1, label2, num_classes).item()
    metric_dict['SSIM'] = ssim_metric(warp_volume1, volume2).item()
    metric_dict['folds_percent'] = folds_percent_metric(dvf).item()
    # metric_dict['folds_count'] = folds_count_metric(dvf).item()
    # metric_dict['mse'] = mse_metric(warp_volume1, volume2).item()
    return metric_dict


def compute_consistency_loss(args, view_1_warp_volume1, view_2_warp_volume1,
                             view_1_warp_label1, view_2_warp_label1):
    # similarity_loss = NCCLoss(spatial_dims=3, kernel_size=9)(view_1_warp_volume1, view_2_warp_volume1)
    similarity_loss = MSELoss()(view_1_warp_volume1, view_2_warp_volume1)
    seg_loss = torch.tensor(0., device=view_1_warp_volume1.device)
    if view_1_warp_label1 != [] and view_2_warp_label1 != []:
        seg_loss = DiceLoss()(view_1_warp_label1, view_2_warp_label1)
    # similarity_loss = Variable(similarity_loss, requires_grad=True)
    # seg_loss = Variable(seg_loss, requires_grad=True)
    total_loss = similarity_loss + args.seg_loss_weight * seg_loss
    total_loss = Variable(total_loss, requires_grad=True)
    total_loss = args.consistency_loss_weight * total_loss
    loss_dict = {"consistency_total_loss": total_loss,
                 "consistency_similarity_loss": similarity_loss,
                 "consistency_seg_loss": seg_loss}
    return loss_dict


def compute_loss(args, dvf, img1, img2, mask1, mask2, is_consistency=False):
    warp_img1 = SpatialTransformer(mode='bilinear')(img1, dvf)
    similarity_loss = NCCLoss(spatial_dims=2, kernel_size=9)(warp_img1, img2)
    seg_loss = torch.tensor(0., device=dvf.device)
    if mask1 != [] and mask2 != []:
        num_class = torch.max(mask1)
        count = 0
        for i in range(1, num_class + 1):
            mask1_i = (mask1 == i).float()
            mask2_i = (mask2 == i).float()
            if torch.any(mask1_i) and torch.any(mask2_i):
                warp_mask1_i = SpatialTransformer(mode='bilinear')(mask1_i.float(), dvf)
                seg_loss = seg_loss + DiceLoss()(warp_mask1_i, mask2_i)
                count = count + 1
        seg_loss = seg_loss / count if count != 0 else seg_loss
    smooth_loss = GradientLoss2D()(dvf)
    total_loss = similarity_loss + args.seg_loss_weight * seg_loss + args.smooth_loss_weight * smooth_loss
    loss_dict = {"total_loss": total_loss, "similarity_loss": similarity_loss, "seg_loss": seg_loss,
                 "smooth_loss": smooth_loss}
    
    # 如果是一致性损失，调整权重
    if is_consistency:
        for key, item in loss_dict.items():
            loss_dict[key] *= args.consistency_loss_weight
    return loss_dict


def cc_mask(fix, reg, win=None):
    if win is None:
        win = [9, 9]
    I, J = fix, reg
    conv = nn.Conv2d(1, 1, win[0], padding=win[0]//2, bias=False)
    conv.register_parameter(name='weight',
                            param=nn.Parameter(torch.ones([1, 1, win[0], win[1]])))
    for param in conv.parameters():
        param.requires_grad = False
    conv = conv.to(I.device)
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv(I)
    J_sum = conv(J)
    I2_sum = conv(I2)
    J2_sum = conv(J2)
    IJ_sum = conv(IJ)

    win_size = win[0] * win[1] 
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)
    return cc
    
    
def get_cc_mask_map(map1, map2, mode):#learn map1
    if mode == 'u':
        r = map2 - map1
    if mode == 'i':
        r = map1 - map2
    r[r > 0] = 1
    r[r <= 0] = 0
    return r


def compute_lossv(args, epoch, dvf1, dvf2, img1, img2, mask1, mask2, mode, is_consistency=False):
    if mode == 'u':
        dvf = dvf1
    if mode == 'i':
        dvf = dvf2
    warp_img1 = SpatialTransformer(mode='bilinear')(img1, dvf1)
    warp_img2 = SpatialTransformer(mode='bilinear')(img1, dvf2)
    cc1 = cc_mask(img2, warp_img1)
    cc2 = cc_mask(img2, warp_img2)
    cc_mask_map = get_cc_mask_map(cc1, cc2, mode)
    
    similarity_loss = NCCLoss(spatial_dims=2, kernel_size=9)(warp_img1, img2)
    seg_loss = torch.tensor(0., device=dvf.device)
    if mask1 != [] and mask2 != []:
        num_class = torch.max(mask1)
        count = 0
        for i in range(1, num_class + 1):
            mask1_i = (mask1 == i).float()
            mask2_i = (mask2 == i).float()
            if torch.any(mask1_i) and torch.any(mask2_i):
                warp_mask1_i = SpatialTransformer(mode='bilinear')(mask1_i.float(), dvf)
                seg_loss = seg_loss + DiceLoss()(warp_mask1_i, mask2_i)
                count = count + 1
        seg_loss = seg_loss / count if count != 0 else seg_loss
    smooth_loss = GradientLoss2D()(dvf)
    dvf_loss = MSELoss()(dvf1*torch.sqrt(cc_mask_map), dvf2*torch.sqrt(cc_mask_map))
    if epoch < 20:
        total_loss = similarity_loss + args.seg_loss_weight * seg_loss + args.smooth_loss_weight * smooth_loss #+ args.dvf_loss_weight * dvf_loss
    else:
        total_loss = similarity_loss + args.seg_loss_weight * seg_loss + args.smooth_loss_weight * smooth_loss + args.dvf_loss_weight * dvf_loss
    loss_dict = {"total_loss": total_loss, "similarity_loss": similarity_loss, "seg_loss": seg_loss,
                 "smooth_loss": smooth_loss, 'dvf_loss': dvf_loss}
    
    # 如果是一致性损失，调整权重
    if is_consistency:
        for key, item in loss_dict.items():
            loss_dict[key] *= args.consistency_loss_weight
    return loss_dict


def create_argparser():
    defaults = dict(
        view_1_dim=2,
        view_2_dim=4,
        data_dir=r"/syq/registration_3/datasets/test.json",
        lr=1e-4,
        batch_size=8,
        log_interval=10,
        save_interval=50,
        resume_checkpoint="",
        logdir="/syq/registration_3/output/test",
        gpu='0',
        max_save_num=10,
        val_interval=10,
        part_size=10,
        epoches=200,
        registration_type=1,
        seg_loss_weight=5,
        smooth_loss_weight=5.,
        dvf_loss_weight=1,
        consistency_loss_weight=1.,
        resolution="",
    )
    defaults.update(voxelmorph_defaults_var1())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
